// Voice cache: which voices keep their GPU state (LM prefix K/V, renderer prefix pages, voice table row).
//
// Modes (kt2n_voice_cache_mode): ALL keeps every voice resident and does not use this class at all (the engine's
// original path). LRU prepares a voice on first use and keeps it until room is needed, dropping the least recently
// used voice that nothing is using. NONE drops a voice as soon as nothing is using it.
//
// Rules:
//   * A pin records that a request will use the voice. Pinned voices are never dropped. A pin does not mean the
//     voice is resident: a load that does not fit the budget waits (queued, FIFO) until unpinned voices can be
//     dropped. It never fails for lack of room and never drops a pinned voice.
//   * Loads are single-flight: every pin of a cold voice joins its one load.
//   * Budgets: at most max_voices counted voices (the hidden voice is not counted), max_bytes GPU bytes,
//     max_pages renderer pages and max_lm_bytes LM store bytes held by resident, loading and dropping voices.
//   * A failed load is retried (with a short backoff) while pins remain; after kMaxAttempts consecutive failures the
//     voice is marked failed and the requests pinning it fail. Once every pin is gone it may be loaded again.
//   * A load that nothing pins any more (a warm load, or a request cancelled while its voice loads) never makes room:
//     when it would have to wait for room it is abandoned instead (the voice goes back to cold).
//   * keep(): an idle voice that is likely to be used again soon (a realtime session between phrases) is not dropped
//     when its last pin goes in NONE mode; it is still dropped like any unpinned voice when a load needs room.
//
// Threading: every call except stats() comes from one thread (the conductor). stats() may be called from any thread.
// The loader is called from pump() / drain() on that thread; it must not call back into the cache.
#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace kt2n::engine {

struct VoiceCost {
  uint64_t lm_bytes = 0;     // LM voice store range
  int pages = 0;             // renderer prefix pages (upper bound)
  uint64_t page_bytes = 0;   // pages x bytes per page
  uint64_t table_bytes = 0;  // renderer voice table row
  uint64_t total() const { return lm_bytes + page_bytes + table_bytes; }
};

struct VoiceCacheLimits {  // 0 = no limit on that dimension
  int max_voices = 0;
  uint64_t max_bytes = 0;
  int max_pages = 0;
  uint64_t max_lm_bytes = 0;
};

// The GPU side of a load, implemented by the engine (and by fakes in tests).
class VoiceLoader {
 public:
  enum : int { kPending = 0, kDone = 1, kWaitRoom = 2, kFailed = -1, kBusy = -2 };
  virtual ~VoiceLoader() = default;
  // Begin preparing the voice: kPending (poll later), kDone, kWaitRoom (a backend has no room now; the cache drops
  // one more unpinned voice and polls again), or kFailed (+ err).
  virtual int start_load(int key, std::string* err) = 0;
  // Continue / check a load: kDone, kPending, kWaitRoom or kFailed (+ err).
  virtual int poll_load(int key, std::string* err) = 0;
  // Give the voice's GPU state back (also after a failed or partial load). 0 = done, kBusy = something still reads
  // it (retry later).
  virtual int unload(int key, std::string* err) = 0;
  // Shutdown: true when the running load has GPU work queued that must complete before its state can be given back
  // (drain() waits for it); false when the load can be given back at once (drain() abandons it without polling, so
  // no new GPU work starts during shutdown).
  virtual bool load_in_flight(int key) {
    (void)key;
    return true;
  }
  // The voice is retired and holds no GPU state: nothing reads its host record again, so it can be dropped.
  // Called once per voice, with the cache's lock held.
  virtual void forget(int key) { (void)key; }
};

class VoiceCache {
 public:
  enum Mode : int { kAll = 0, kLru = 1, kNone = 2 };
  enum class Res : uint8_t { kCold, kQueued, kLoading, kResident, kUnloading, kFailed };
  static constexpr int kMaxAttempts = 3;

  struct Stats {
    int mode = kLru;
    int voices = 0, resident = 0, pinned = 0, loading = 0, queued = 0, unloading = 0, failed = 0;
    uint64_t gpu_lm_bytes = 0, gpu_page_bytes = 0, gpu_table_bytes = 0, host_bytes = 0;
    int gpu_pages = 0;
    uint64_t hits = 0, misses = 0, joins = 0, evictions = 0, drops = 0, busy_retries = 0, room_waits = 0;
    uint64_t waits = 0;
    double wait_ms = 0;
    uint64_t loads = 0, load_failures = 0, retire = 0;
    double load_ms_p50 = 0, load_ms_p95 = 0, load_ms_max = 0;
    VoiceCacheLimits limits;
  };

  // clock: seconds on a monotonic clock (tests pass a fake one).
  VoiceCache(int mode, const VoiceCacheLimits& limits, VoiceLoader* loader, std::function<double()> clock_fn = {});

  // Registers a voice (cold). counted = false: not counted against max_voices (the hidden voice). Returns its key.
  int add(const VoiceCost& cost, bool counted, uint64_t host_bytes);
  // True when an otherwise empty cache could hold the voice.
  bool fits_alone(const VoiceCost& cost) const;
  const VoiceCost& cost(int key) const { return v_[size_t(key)].cost; }
  int size() const { return int(v_.size()); }

  void pin(int key);
  void unpin(int key);
  bool ready(int key) const;   // resident (usable by new rows)
  // The requests pinning it fail: its loads failed permanently while pinned, or it is retired and not resident (a
  // request that looked the voice up just before its handle was evicted and released can never be served).
  bool failed(int key) const;
  Res res(int key) const { return v_[size_t(key)].res; }
  int pins(int key) const { return v_[size_t(key)].pins; }
  // LRU only: queue a load without a pin (a fresh clone, kt2n_voice_prepare), if it fits without dropping anything.
  void warm(int key);
  // Its last load gave up after the retries and it is not resident (kFailed, or back to cold with nothing pinning it).
  bool load_failed(int key) const;
  // Keep an idle voice resident between uses (NONE mode: a realtime session's voice between phrases); unkeep() ends
  // it. Kept voices are not pinned: they are dropped for room like any unpinned voice. Counted per call.
  void keep(int key);
  void unkeep(int key);
  // The voice is gone for good (its handle was evicted and no request holds it): drop its GPU state when possible,
  // never load it again.
  void retire(int key);

  // One conductor turn: finish loads, drop voices (NONE mode, and to make room), start queued loads.
  void pump();
  // Work in progress (queued, loading or dropping voices): the conductor should come back soon.
  bool busy() const { return !queue_.empty() || !loading_.empty() || !unloading_.empty(); }
  // Engine failure: forget queued loads (loads already running are finished by drain()).
  void clear_pending();
  // Shutdown: abandon queued loads and running loads without queued GPU work (VoiceLoader::load_in_flight), and poll
  // the others until they finish (or timeout_s passes). It does not release resident voices: the backends free every
  // voice's GPU state when they are destroyed afterwards. Returns the number of loads still running at the timeout.
  int drain(double timeout_s);

  int resident_count() const;  // counted resident voices
  // Stats: `resident` counts the voices held against max_voices (the hidden voice is not counted, as in
  // resident_count); `loading`, `queued`, `unloading`, `failed` and `pinned` count every voice.
  Stats stats() const;
  std::string stats_json() const;

 private:
  struct Entry {
    VoiceCost cost;
    bool counted = true;
    bool retired = false;
    bool warm = false;
    int keeps = 0;       // keep() calls not yet ended (NONE: not dropped when the last pin goes)
    bool holds = false;  // counted in the used totals (loading, resident or dropping)
    Res res = Res::kCold;
    int pins = 0;
    uint64_t last_use = 0;
    uint64_t host_bytes = 0;
    int attempts = 0;           // consecutive failed loads
    double retry_after = 0;     // no new attempt before this time
    double load_t0 = 0;
    double wait_t0 = -1;        // waiting for room since (-1 = not waiting)
    double room_t0 = -1;        // kWaitRoom while nothing else holds GPU state, since
    bool room_wait = false;     // the load is waiting for room (counted once in room_waits)
    bool fail_after_unload = false;
    bool gave_up = false;       // the last load gave up after kMaxAttempts (back to cold, now or at its last unpin)
    bool forgotten = false;     // retired and cold: the loader dropped its host record (VoiceLoader::forget)
    std::string error;
  };
  void forget_if_done(int key);  // mu_ held: a retired voice that reached cold drops its host record
  bool fits(const Entry& x, bool projected, bool ignore_count = false) const;
  void clear_queue();  // mu_ held
  void hold(Entry& x, bool on);
  // Drop the least recently used resident unpinned voice; false when none. counted_only: only voices counted against
  // max_voices (when only the voice count is short, dropping the hidden voice frees nothing that is needed).
  bool evict_one(int keep, bool counted_only = false);
  void abandon_load(int key);                   // a load nothing pins any more that would wait for room
  void begin_unload(int key, bool evicted, bool count = true);
  void complete_unload(int key);
  void finish_load(int key, int rc, const std::string& err);
  void note_load_ms(double ms);
  bool alone(int key) const;
  double now() const { return clock_ ? clock_() : 0.0; }

  int mode_;
  VoiceCacheLimits lim_;
  VoiceLoader* loader_;
  std::function<double()> clock_;
  std::vector<Entry> v_;
  std::deque<int> queue_;        // loads to start (FIFO)
  std::vector<int> loading_;     // started loads
  std::vector<int> unloading_;   // drops in progress (kBusy retries)
  bool need_room_ = false;
  double last_room_evict_ = -1e9;
  uint64_t tick_ = 0;
  // Totals of voices that hold GPU state.
  int used_voices_ = 0, used_pages_ = 0;
  uint64_t used_bytes_ = 0, used_lm_ = 0;
  mutable std::mutex mu_;  // guards everything stats() reads
  Stats st_;
  std::vector<double> load_ms_;  // recent load times (ring of 1024)
  size_t load_ms_next_ = 0;
};

}  // namespace kt2n::engine
