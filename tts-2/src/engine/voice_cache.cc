// Voice cache (see voice_cache.h).
#include "voice_cache.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>

namespace kt2n::engine {

namespace {
double steady_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void erase_key(std::vector<int>& v, int key) { v.erase(std::remove(v.begin(), v.end(), key), v.end()); }
}  // namespace

VoiceCache::VoiceCache(int mode, const VoiceCacheLimits& limits, VoiceLoader* loader, std::function<double()> clock_fn)
    : mode_(mode), lim_(limits), loader_(loader), clock_(clock_fn ? std::move(clock_fn) : std::function<double()>(steady_s)) {
  st_.mode = mode;
  st_.limits = limits;
}

int VoiceCache::add(const VoiceCost& cost, bool counted, uint64_t host_bytes) {
  std::lock_guard<std::mutex> lk(mu_);
  Entry x;
  x.cost = cost;
  x.counted = counted;
  x.host_bytes = host_bytes;
  v_.push_back(x);
  return int(v_.size()) - 1;
}

bool VoiceCache::fits_alone(const VoiceCost& c) const {
  if (lim_.max_voices < 0) return false;
  if (lim_.max_bytes && c.total() > lim_.max_bytes) return false;
  if (lim_.max_pages && c.pages > lim_.max_pages) return false;
  if (lim_.max_lm_bytes && c.lm_bytes > lim_.max_lm_bytes) return false;
  return true;
}

bool VoiceCache::fits(const Entry& x, bool projected, bool ignore_count) const {
  // projected: count the room of voices being dropped as free already. They still hold their state, but their room is
  // on its way back, so a load that fits once they are gone waits for them instead of dropping more voices.
  int uv = used_voices_, up = used_pages_;
  uint64_t ub = used_bytes_, ul = used_lm_;
  if (projected) {
    for (int k : unloading_) {
      const Entry& u = v_[size_t(k)];
      uv -= u.counted ? 1 : 0;
      up -= u.cost.pages;
      ub -= u.cost.total();
      ul -= u.cost.lm_bytes;
    }
  }
  if (!ignore_count && lim_.max_voices > 0 && x.counted && uv + 1 > lim_.max_voices) return false;
  if (lim_.max_bytes && ub + x.cost.total() > lim_.max_bytes) return false;
  if (lim_.max_pages && up + x.cost.pages > lim_.max_pages) return false;
  if (lim_.max_lm_bytes && ul + x.cost.lm_bytes > lim_.max_lm_bytes) return false;
  return true;
}

void VoiceCache::hold(Entry& x, bool on) {
  if (x.holds == on) return;
  const int s = on ? 1 : -1;
  x.holds = on;
  used_voices_ += x.counted ? s : 0;
  used_pages_ += s * x.cost.pages;
  if (on) {
    used_bytes_ += x.cost.total();
    used_lm_ += x.cost.lm_bytes;
  } else {
    used_bytes_ -= x.cost.total();
    used_lm_ -= x.cost.lm_bytes;
  }
}

void VoiceCache::pin(int key) {
  std::lock_guard<std::mutex> lk(mu_);
  if (key < 0 || size_t(key) >= v_.size()) return;
  Entry& x = v_[size_t(key)];
  x.pins += 1;
  x.gave_up = false;
  x.last_use = ++tick_;
  switch (x.res) {
    case Res::kResident:
      st_.hits++;
      break;
    case Res::kLoading:
    case Res::kQueued:
      st_.joins++;  // single flight: this request waits for the load already queued or running
      x.warm = false;
      break;
    case Res::kUnloading:
      st_.misses++;  // queued again once the drop completes
      break;
    case Res::kFailed:
      break;  // the request fails at admission; a later request retries once every pin is gone
    case Res::kCold:
      st_.misses++;
      x.res = Res::kQueued;
      queue_.push_back(key);
      break;
  }
}

void VoiceCache::unpin(int key) {
  std::lock_guard<std::mutex> lk(mu_);
  if (key < 0 || size_t(key) >= v_.size()) return;
  Entry& x = v_[size_t(key)];
  if (x.pins <= 0) return;
  x.pins -= 1;
  x.last_use = ++tick_;
  if (x.pins > 0) return;
  switch (x.res) {
    case Res::kFailed:  // its last load gave up while pinned; the requests holding it have failed
      x.res = Res::kCold;
      x.attempts = 0;
      x.retry_after = 0;
      x.gave_up = true;  // load_failed() stays true until the next pin or warm, as for a load that gave up unpinned
      break;
    case Res::kQueued:
      if (!x.warm) {
        queue_.erase(std::remove(queue_.begin(), queue_.end(), key), queue_.end());
        x.res = Res::kCold;
        x.attempts = 0;  // the next request starts without this one's retry backoff
        x.retry_after = 0;
        if (x.wait_t0 >= 0) {
          st_.wait_ms += (now() - x.wait_t0) * 1000.0;
          x.wait_t0 = -1;
        }
      }
      break;
    case Res::kResident:
      if ((mode_ == kNone && x.keeps == 0) || x.retired) begin_unload(key, false);
      break;
    default:
      break;  // loading: decided when the load finishes; unloading: continues
  }
}

bool VoiceCache::ready(int key) const {
  std::lock_guard<std::mutex> lk(mu_);
  return key >= 0 && size_t(key) < v_.size() && v_[size_t(key)].res == Res::kResident;
}

bool VoiceCache::failed(int key) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (key < 0 || size_t(key) >= v_.size()) return false;
  const Entry& x = v_[size_t(key)];
  return x.res == Res::kFailed || (x.retired && x.res != Res::kResident);
}

void VoiceCache::keep(int key) {
  std::lock_guard<std::mutex> lk(mu_);
  if (key < 0 || size_t(key) >= v_.size()) return;
  v_[size_t(key)].keeps += 1;
}

void VoiceCache::unkeep(int key) {
  std::lock_guard<std::mutex> lk(mu_);
  if (key < 0 || size_t(key) >= v_.size()) return;
  Entry& x = v_[size_t(key)];
  if (x.keeps <= 0 || --x.keeps > 0) return;
  if (x.pins == 0 && x.res == Res::kResident && (mode_ == kNone || x.retired)) begin_unload(key, false);
}

void VoiceCache::warm(int key) {
  std::lock_guard<std::mutex> lk(mu_);
  if (mode_ != kLru || key < 0 || size_t(key) >= v_.size()) return;
  Entry& x = v_[size_t(key)];
  if (x.res != Res::kCold || x.retired) return;
  x.warm = true;
  x.gave_up = false;
  x.res = Res::kQueued;
  x.last_use = ++tick_;  // a voice warmed for use (kt2n_voice_prepare) is not the first one dropped for room
  queue_.push_back(key);
}

bool VoiceCache::load_failed(int key) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (key < 0 || size_t(key) >= v_.size()) return false;
  const Entry& x = v_[size_t(key)];
  return x.res == Res::kFailed || (x.res == Res::kCold && x.gave_up);
}

void VoiceCache::retire(int key) {
  std::lock_guard<std::mutex> lk(mu_);
  if (key < 0 || size_t(key) >= v_.size()) return;
  Entry& x = v_[size_t(key)];
  if (x.retired) return;
  x.retired = true;
  st_.retire++;
  if (x.res == Res::kQueued) {
    queue_.erase(std::remove(queue_.begin(), queue_.end(), key), queue_.end());
    x.res = Res::kCold;
    if (x.wait_t0 >= 0) {
      st_.wait_ms += (now() - x.wait_t0) * 1000.0;
      x.wait_t0 = -1;
    }
  } else if (x.res == Res::kResident && x.pins == 0) {
    begin_unload(key, false);
  } else if (x.res == Res::kFailed && x.pins == 0) {
    x.res = Res::kCold;
  }
  forget_if_done(key);
}

void VoiceCache::forget_if_done(int key) {
  Entry& x = v_[size_t(key)];
  if (!x.retired || x.forgotten || x.res != Res::kCold || x.holds) return;
  x.forgotten = true;
  x.host_bytes = 0;
  loader_->forget(key);
}

bool VoiceCache::alone(int key) const {
  for (size_t i = 0; i < v_.size(); ++i)
    if (int(i) != key && v_[i].holds) return false;
  return true;
}

bool VoiceCache::evict_one(int keep, bool counted_only) {
  int best = -1;
  for (size_t i = 0; i < v_.size(); ++i) {
    const Entry& c = v_[i];
    if (int(i) == keep || c.res != Res::kResident || c.pins > 0 || (counted_only && !c.counted)) continue;
    if (best < 0 || c.last_use < v_[size_t(best)].last_use) best = int(i);
  }
  if (best < 0) return false;
  begin_unload(best, true);
  return true;
}

void VoiceCache::abandon_load(int key) {
  Entry& x = v_[size_t(key)];
  erase_key(loading_, key);
  x.room_t0 = -1;
  x.room_wait = false;
  x.warm = false;
  begin_unload(key, false, false);  // no pin: back to cold once its partial state is given back
}

void VoiceCache::begin_unload(int key, bool evicted, bool count) {
  Entry& x = v_[size_t(key)];
  x.res = Res::kUnloading;
  if (count && evicted)
    st_.evictions++;
  else if (count && !x.fail_after_unload)
    st_.drops++;
  if (std::find(unloading_.begin(), unloading_.end(), key) == unloading_.end()) unloading_.push_back(key);
  std::string err;
  const int rc = loader_->unload(key, &err);
  if (rc == VoiceLoader::kBusy) {
    st_.busy_retries++;
    return;
  }
  complete_unload(key);  // done (or a hard failure, which leaves nothing the cache can retry): the room is back
}

void VoiceCache::complete_unload(int key) {
  Entry& x = v_[size_t(key)];
  erase_key(unloading_, key);
  hold(x, false);
  const bool after_fail = x.fail_after_unload;
  x.fail_after_unload = false;
  if (after_fail && x.attempts >= kMaxAttempts) {
    x.res = x.pins > 0 ? Res::kFailed : Res::kCold;
    x.warm = false;  // a later pin and unpin must not leave a load nobody needs queued
    x.gave_up = x.pins == 0;
    if (x.pins == 0) x.attempts = 0;
    forget_if_done(key);
    return;
  }
  x.res = Res::kCold;
  if (!x.retired && (x.pins > 0 || (after_fail && x.warm))) {
    if (after_fail) x.retry_after = now() + 0.05 * double(1 << std::min(4, std::max(0, x.attempts - 1)));
    x.res = Res::kQueued;
    queue_.push_back(key);
  }
  forget_if_done(key);
}

void VoiceCache::note_load_ms(double ms) {
  if (load_ms_.size() < 1024) {
    load_ms_.push_back(ms);
  } else {
    load_ms_[load_ms_next_] = ms;
    load_ms_next_ = (load_ms_next_ + 1) % load_ms_.size();
  }
}

void VoiceCache::finish_load(int key, int rc, const std::string& err) {
  Entry& x = v_[size_t(key)];
  erase_key(loading_, key);
  x.room_t0 = -1;
  x.room_wait = false;
  if (rc == VoiceLoader::kDone) {
    x.res = Res::kResident;
    x.attempts = 0;
    x.gave_up = false;
    x.error.clear();
    x.warm = false;
    st_.loads++;
    note_load_ms((now() - x.load_t0) * 1000.0);
    if (x.retired || (mode_ == kNone && x.pins == 0 && x.keeps == 0)) begin_unload(key, false);
    return;
  }
  st_.load_failures++;
  x.attempts += 1;
  x.error = err;
  if (x.attempts == 1 || x.attempts >= kMaxAttempts)
    std::fprintf(stderr, "[kt2n voice cache] voice load failed (attempt %d): %s\n", x.attempts, err.c_str());
  x.fail_after_unload = true;
  begin_unload(key, false);  // give back whatever the partial load took
}

void VoiceCache::pump() {
  std::lock_guard<std::mutex> lk(mu_);
  if (queue_.empty() && loading_.empty() && unloading_.empty() && !need_room_) return;
  const double t = now();
  std::string err;
  // 1. Running loads.
  const std::vector<int> running = loading_;
  for (int key : running) {
    Entry& x = v_[size_t(key)];
    err.clear();
    const int rc = loader_->poll_load(key, &err);
    if (rc == VoiceLoader::kPending) {
      x.room_wait = false;
      continue;
    }
    if (rc == VoiceLoader::kWaitRoom) {
      if (x.pins == 0) {  // nothing needs this load (a warm load, or its request is gone): never drop voices for it
        abandon_load(key);
        continue;
      }
      if (!x.room_wait) st_.room_waits++;  // once per wait, not per poll
      x.room_wait = true;
      need_room_ = true;
      // Nothing else holds GPU state, yet a backend still has no room: the voice can never fit.
      if (alone(key)) {
        if (x.room_t0 < 0) {
          x.room_t0 = t;
        } else if (t - x.room_t0 > 2.0) {
          x.attempts = kMaxAttempts - 1;
          finish_load(key, VoiceLoader::kFailed, "the voice does not fit the voice cache budget");
        }
      } else {
        x.room_t0 = -1;
      }
      continue;
    }
    finish_load(key, rc, err);
  }
  // 2. Drops waiting for their readers.
  const std::vector<int> dropping = unloading_;
  for (int key : dropping) {
    err.clear();
    if (loader_->unload(key, &err) == VoiceLoader::kBusy) {
      st_.busy_retries++;
      continue;
    }
    complete_unload(key);
  }
  // 3. A backend asked for room (LM store fragmentation, renderer pages taken by rows): drop one more unpinned voice.
  // One at a time: wait for the previous room-making drop to finish, and at most one every 5 ms (a dropped LM range
  // is freed only after the LM releases its last rows' slots).
  if (need_room_ && unloading_.empty() && t - last_room_evict_ >= 0.005) {
    need_room_ = false;
    last_room_evict_ = t;
    evict_one(-1);
  }
  // 4. Queued loads, first come first served. The head waits (and nothing behind it starts) until it fits.
  while (!queue_.empty()) {
    const int key = queue_.front();
    Entry& x = v_[size_t(key)];
    if (x.res != Res::kQueued || x.retired || (x.pins == 0 && !x.warm)) {
      queue_.pop_front();
      if (x.res == Res::kQueued) x.res = Res::kCold;
      continue;
    }
    if (x.retry_after > t) {
      // A retry backoff: let the voices behind it go first.
      bool started_other = false;
      for (size_t i = 1; i < queue_.size(); ++i) {
        const int k2 = queue_[i];
        Entry& y = v_[size_t(k2)];
        if (y.res != Res::kQueued || y.retry_after > t || (y.pins == 0 && !y.warm) || y.retired || !fits(y, false))
          continue;
        queue_.erase(queue_.begin() + long(i));
        queue_.push_front(k2);
        started_other = true;
        break;
      }
      if (!started_other) break;
      continue;
    }
    if (!fits(x, false)) {
      if (x.warm && x.pins == 0) {  // a warm load never drops anything
        queue_.pop_front();
        x.res = Res::kCold;
        x.warm = false;
        continue;
      }
      while (!fits(x, true) && evict_one(key, fits(x, true, true))) {
      }
      if (!fits(x, false)) {  // wait for the drops (they free their room in a later pump)
        if (x.wait_t0 < 0) {
          x.wait_t0 = t;
          st_.waits++;
        }
        break;
      }
    }
    queue_.pop_front();
    if (x.wait_t0 >= 0) {
      st_.wait_ms += (t - x.wait_t0) * 1000.0;
      x.wait_t0 = -1;
    }
    hold(x, true);
    x.res = Res::kLoading;
    x.load_t0 = t;
    loading_.push_back(key);
    err.clear();
    const int rc = loader_->start_load(key, &err);
    if (rc == VoiceLoader::kPending) continue;
    if (rc == VoiceLoader::kWaitRoom) {
      if (x.pins == 0) {  // a warm load never drops anything
        abandon_load(key);
        continue;
      }
      st_.room_waits++;
      x.room_wait = true;
      need_room_ = true;
      continue;
    }
    finish_load(key, rc, err);
  }
}

void VoiceCache::clear_queue() {
  for (int key : queue_) {
    Entry& x = v_[size_t(key)];
    if (x.res == Res::kQueued) x.res = Res::kCold;
    if (x.wait_t0 >= 0) {
      st_.wait_ms += (now() - x.wait_t0) * 1000.0;
      x.wait_t0 = -1;
    }
  }
  queue_.clear();
}

void VoiceCache::clear_pending() {
  std::lock_guard<std::mutex> lk(mu_);
  clear_queue();
}

int VoiceCache::drain(double timeout_s) {
  const double t0 = steady_s();
  for (;;) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      clear_queue();
      const std::vector<int> running = loading_;
      for (int key : running) {
        if (!loader_->load_in_flight(key)) {
          // Nothing queued on the GPU for it (e.g. waiting for render pages that only closing rows give back, and no
          // row closes any more): give back what it holds now and fail it, without advancing it.
          Entry& x = v_[size_t(key)];
          erase_key(loading_, key);
          x.room_t0 = -1;
          x.room_wait = false;
          x.warm = false;
          x.attempts = kMaxAttempts;
          x.fail_after_unload = true;
          x.error = "the engine is shutting down";
          begin_unload(key, false, false);
          continue;
        }
        std::string err;
        const int rc = loader_->poll_load(key, &err);
        if (rc == VoiceLoader::kPending) continue;
        // kWaitRoom at shutdown: nothing will make room, give back what the load holds.
        finish_load(key, rc == VoiceLoader::kWaitRoom ? VoiceLoader::kFailed : rc, err);
      }
      clear_queue();  // a failed load may have queued its retry
      if (loading_.empty() || steady_s() - t0 > timeout_s) return int(loading_.size());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

int VoiceCache::resident_count() const {
  std::lock_guard<std::mutex> lk(mu_);
  int n = 0;
  for (const Entry& x : v_) n += (x.res == Res::kResident && x.counted) ? 1 : 0;
  return n;
}

VoiceCache::Stats VoiceCache::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  Stats s = st_;
  s.voices = int(v_.size());
  for (const Entry& x : v_) {
    s.resident += (x.res == Res::kResident && x.counted) ? 1 : 0;
    s.pinned += x.pins > 0 ? 1 : 0;
    s.loading += x.res == Res::kLoading ? 1 : 0;
    s.queued += x.res == Res::kQueued ? 1 : 0;
    s.unloading += x.res == Res::kUnloading ? 1 : 0;
    s.failed += x.res == Res::kFailed ? 1 : 0;
    s.host_bytes += x.host_bytes;
    if (x.holds) {
      s.gpu_lm_bytes += x.cost.lm_bytes;
      s.gpu_page_bytes += x.cost.page_bytes;
      s.gpu_table_bytes += x.cost.table_bytes;
      s.gpu_pages += x.cost.pages;
    }
  }
  if (!load_ms_.empty()) {
    std::vector<double> v = load_ms_;
    std::sort(v.begin(), v.end());
    auto pct = [&](double p) { return v[std::min(v.size() - 1, size_t(p * double(v.size() - 1) + 0.5))]; };
    s.load_ms_p50 = pct(0.50);
    s.load_ms_p95 = pct(0.95);
    s.load_ms_max = v.back();
  }
  return s;
}

std::string VoiceCache::stats_json() const {
  const Stats s = stats();
  auto u = [](uint64_t x) { return std::to_string(x); };
  auto i = [](int x) { return std::to_string(x); };
  auto d = [](double x) {
    char b[48];
    std::snprintf(b, sizeof b, "%.3f", x);
    return std::string(b);
  };
  const char* mode = s.mode == kNone ? "none" : s.mode == kLru ? "lru" : "all";
  return std::string("{\"mode\":\"") + mode + "\",\"voices\":" + i(s.voices) + ",\"resident\":" + i(s.resident) +
         ",\"pinned\":" + i(s.pinned) + ",\"loading\":" + i(s.loading) + ",\"queued\":" + i(s.queued) +
         ",\"unloading\":" + i(s.unloading) + ",\"failed\":" + i(s.failed) + ",\"max_voices\":" + i(s.limits.max_voices) +
         ",\"max_bytes\":" + u(s.limits.max_bytes) + ",\"max_pages\":" + i(s.limits.max_pages) +
         ",\"max_lm_bytes\":" + u(s.limits.max_lm_bytes) + ",\"gpu_bytes\":{\"lm\":" + u(s.gpu_lm_bytes) +
         ",\"pages\":" + u(s.gpu_page_bytes) + ",\"tables\":" + u(s.gpu_table_bytes) +
         ",\"total\":" + u(s.gpu_lm_bytes + s.gpu_page_bytes + s.gpu_table_bytes) + "},\"gpu_pages\":" + i(s.gpu_pages) +
         ",\"host_bytes\":" + u(s.host_bytes) + ",\"hits\":" + u(s.hits) + ",\"misses\":" + u(s.misses) +
         ",\"joins\":" + u(s.joins) + ",\"evictions\":" + u(s.evictions) + ",\"drops\":" + u(s.drops) +
         ",\"busy_retries\":" + u(s.busy_retries) + ",\"room_waits\":" + u(s.room_waits) + ",\"waits\":" + u(s.waits) +
         ",\"wait_ms\":" + d(s.wait_ms) + ",\"loads\":" + u(s.loads) + ",\"load_failures\":" + u(s.load_failures) +
         ",\"retired\":" + u(s.retire) + ",\"load_ms\":{\"p50\":" + d(s.load_ms_p50) + ",\"p95\":" + d(s.load_ms_p95) +
         ",\"max\":" + d(s.load_ms_max) + "}}";
}

}  // namespace kt2n::engine
