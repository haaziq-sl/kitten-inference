// Voice cache state machine (src/engine/voice_cache.*) against a fake loader: modes, LRU order, pins, budgets,
// single-flight loads, failures and retries, drops that must wait for readers, shutdown, stats.
#include <gtest/gtest.h>

#include <chrono>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "voice_cache.h"

namespace {

using kt2n::engine::VoiceCache;
using kt2n::engine::VoiceCacheLimits;
using kt2n::engine::VoiceCost;
using kt2n::engine::VoiceLoader;
using Res = VoiceCache::Res;

// A loader whose loads take `polls` polls; scripted failures, busy drops and room waits per key. It also checks the
// cache's promises: a voice is unloaded only when no pin holds it, and never twice.
class FakeLoader final : public VoiceLoader {
 public:
  int polls = 0;                   // polls before a load is done (0 = done at start)
  std::map<int, int> fail;         // key -> remaining failing loads
  std::map<int, int> busy;         // key -> remaining kBusy answers to unload
  std::map<int, int> room;         // key -> remaining kWaitRoom answers
  std::map<int, int> left;         // key -> polls left of its running load
  std::map<int, int> starts;       // key -> start_load calls
  std::map<int, int> unloads;      // key -> completed unloads
  std::set<int> loaded;            // keys whose load completed and that still hold their (fake) GPU state
  std::set<int> has_state;         // keys with a started load not yet unloaded
  std::set<int> in_drop;           // keys whose unload answered kBusy (a drop in progress)
  std::set<int> no_gpu_work;       // keys whose running load has no queued GPU work (load_in_flight false)
  std::map<int, int> forgets;      // key -> forget calls (host record dropped)
  const VoiceCache* cache = nullptr;
  int violations = 0;

  int step(int key, std::string* err) {
    if (room[key] > 0) {
      --room[key];
      return kWaitRoom;
    }
    if (fail[key] > 0) {
      --fail[key];
      if (err) *err = "injected";
      return kFailed;
    }
    if (left[key] > 0) {
      --left[key];
      return kPending;
    }
    loaded.insert(key);
    return kDone;
  }
  int start_load(int key, std::string* err) override {
    starts[key]++;
    if (has_state.count(key)) violations++;  // a second load of a voice whose state was never given back
    has_state.insert(key);
    left[key] = polls;
    return step(key, err);
  }
  int poll_load(int key, std::string* err) override { return step(key, err); }
  bool load_in_flight(int key) override { return !no_gpu_work.count(key); }
  void forget(int key) override {
    forgets[key]++;
    if (has_state.count(key) || in_drop.count(key)) violations++;  // only once its GPU state is given back
  }
  int unload(int key, std::string*) override {
    if (!in_drop.count(key)) {
      // A drop starts (begin_unload): a voice whose load completed is never dropped while a pin holds it.
      if (cache && loaded.count(key) && cache->pins(key) > 0) violations++;
      in_drop.insert(key);
    }
    if (busy[key] > 0) {
      --busy[key];
      return kBusy;
    }
    in_drop.erase(key);
    if (!has_state.count(key)) violations++;  // never twice: each started load is given back once
    has_state.erase(key);
    loaded.erase(key);
    unloads[key]++;
    return 0;
  }
};

VoiceCost cost(uint64_t lm, int pages) {
  VoiceCost c;
  c.lm_bytes = lm;
  c.pages = pages;
  c.page_bytes = uint64_t(pages) * 1000;
  c.table_bytes = 10;
  return c;
}

struct Fixture {
  double t = 0;
  FakeLoader ld;
  VoiceCache vc;
  Fixture(int mode, VoiceCacheLimits lim, int n, VoiceCost c = cost(100, 2))
      : vc(mode, lim, &ld, [this] { return t; }) {
    ld.cache = &vc;
    for (int i = 0; i < n; ++i) vc.add(c, true, 1000);
  }
  void pump(int n = 1) {
    for (int i = 0; i < n; ++i) vc.pump();
  }
  // Pin, pump until resident.
  void use(int key) {
    vc.pin(key);
    for (int i = 0; i < 50 && !vc.ready(key); ++i) pump();
  }
};

VoiceCacheLimits voices(int n) {
  VoiceCacheLimits l;
  l.max_voices = n;
  return l;
}

TEST(VoiceCache, LruEvictsLeastRecentlyUsedUnpinned) {
  Fixture f(VoiceCache::kLru, voices(3), 5);
  for (int k : {0, 1, 2}) {
    f.use(k);
    ASSERT_TRUE(f.vc.ready(k));
    f.vc.unpin(k);
  }
  f.use(0);  // 0 is now the most recent: 1 is the least recently used
  f.vc.unpin(0);
  f.use(3);
  EXPECT_TRUE(f.vc.ready(3));
  EXPECT_EQ(f.vc.res(1), Res::kCold);
  EXPECT_TRUE(f.vc.ready(0));
  EXPECT_TRUE(f.vc.ready(2));
  EXPECT_EQ(f.vc.resident_count(), 3);
  EXPECT_EQ(f.vc.stats().evictions, 1u);
  EXPECT_EQ(f.ld.violations, 0);
}

TEST(VoiceCache, NoneDropsAsSoonAsUnpinned) {
  Fixture f(VoiceCache::kNone, voices(2), 3);
  f.use(0);
  ASSERT_TRUE(f.vc.ready(0));
  f.vc.unpin(0);
  EXPECT_EQ(f.vc.res(0), Res::kCold);
  EXPECT_EQ(f.ld.loaded.count(0), 0u);
  EXPECT_EQ(f.vc.stats().drops, 1u);
  // A second pin of the same voice while it is in use keeps it.
  f.use(1);
  f.vc.pin(1);
  f.vc.unpin(1);
  EXPECT_TRUE(f.vc.ready(1));
  f.vc.unpin(1);
  EXPECT_EQ(f.vc.res(1), Res::kCold);
}

TEST(VoiceCache, PinnedAndQueuedVoicesAreNeverEvicted) {
  Fixture f(VoiceCache::kLru, voices(2), 4);
  f.use(0);
  f.use(1);  // both pinned (requests queued or running on them)
  f.vc.pin(2);
  f.pump(20);
  EXPECT_EQ(f.vc.res(2), Res::kQueued);  // waits, never fails, never drops a pinned voice
  EXPECT_TRUE(f.vc.ready(0));
  EXPECT_TRUE(f.vc.ready(1));
  EXPECT_EQ(f.ld.starts[2], 0);
  EXPECT_EQ(f.vc.stats().waits, 1u);
  f.t = 0.25;
  f.vc.unpin(0);  // 0's request is done: room for 2
  f.pump(3);
  EXPECT_TRUE(f.vc.ready(2));
  EXPECT_EQ(f.vc.res(0), Res::kCold);
  EXPECT_TRUE(f.vc.ready(1));
  EXPECT_NEAR(f.vc.stats().wait_ms, 250.0, 1e-6);
  EXPECT_EQ(f.ld.violations, 0);
}

TEST(VoiceCache, ConcurrentColdRequestsShareOneLoad) {
  Fixture f(VoiceCache::kLru, voices(4), 2);
  f.ld.polls = 5;
  f.vc.pin(0);
  f.pump();
  f.vc.pin(0);
  f.vc.pin(0);
  f.pump(10);
  EXPECT_TRUE(f.vc.ready(0));
  EXPECT_EQ(f.ld.starts[0], 1);
  const auto st = f.vc.stats();
  EXPECT_EQ(st.misses, 1u);
  EXPECT_EQ(st.joins, 2u);
  EXPECT_EQ(st.loads, 1u);
  EXPECT_EQ(f.vc.pins(0), 3);
}

TEST(VoiceCache, ByteBudgetCountsLoadingAndResidentVoices) {
  VoiceCacheLimits l;
  l.max_bytes = 2 * cost(100, 2).total();
  Fixture f(VoiceCache::kLru, l, 3);
  f.ld.polls = 3;
  f.vc.pin(0);
  f.vc.pin(1);
  f.vc.pin(2);
  f.pump();
  EXPECT_EQ(f.vc.res(0), Res::kLoading);
  EXPECT_EQ(f.vc.res(1), Res::kLoading);
  EXPECT_EQ(f.vc.res(2), Res::kQueued);
  f.pump(10);
  EXPECT_EQ(f.vc.res(2), Res::kQueued);
  EXPECT_LE(f.vc.stats().gpu_lm_bytes + f.vc.stats().gpu_page_bytes + f.vc.stats().gpu_table_bytes, l.max_bytes);
  f.vc.unpin(1);
  f.pump(10);
  EXPECT_TRUE(f.vc.ready(2));
  EXPECT_EQ(f.vc.res(1), Res::kCold);
}

TEST(VoiceCache, PagesAndLmBudgetsAreEnforced) {
  VoiceCacheLimits l;
  l.max_pages = 5;
  l.max_lm_bytes = 1000;
  Fixture f(VoiceCache::kLru, l, 0);
  f.vc.add(cost(100, 3), true, 0);  // 0
  f.vc.add(cost(100, 3), true, 0);  // 1: pages 6 > 5 with 0
  f.vc.add(cost(950, 1), true, 0);  // 2: lm 1050 > 1000 with 0
  f.use(0);
  f.vc.pin(1);
  f.pump(5);
  EXPECT_EQ(f.vc.res(1), Res::kQueued);
  f.vc.unpin(1);
  f.vc.pin(2);
  f.pump(5);
  EXPECT_EQ(f.vc.res(2), Res::kQueued);
  EXPECT_FALSE(f.vc.fits_alone(cost(2000, 1)));
  EXPECT_FALSE(f.vc.fits_alone(cost(10, 6)));
  EXPECT_TRUE(f.vc.fits_alone(cost(1000, 5)));
}

TEST(VoiceCache, HiddenVoiceIsNotCounted) {
  Fixture f(VoiceCache::kLru, voices(1), 1);
  const int hidden = f.vc.add(cost(100, 2), false, 0);
  f.use(0);
  f.use(hidden);
  EXPECT_TRUE(f.vc.ready(0));
  EXPECT_TRUE(f.vc.ready(hidden));
  EXPECT_EQ(f.vc.resident_count(), 1);
}

TEST(VoiceCache, ClonesWarmLoadWithoutDroppingAnything) {
  Fixture f(VoiceCache::kLru, voices(2), 2);
  const int clone = f.vc.add(cost(120, 2), true, 0);  // registered later, like a clone
  f.vc.warm(clone);
  f.pump();
  EXPECT_TRUE(f.vc.ready(clone));  // room: warm-loaded without a pin
  f.use(0);
  f.vc.unpin(0);
  const int clone2 = f.vc.add(cost(120, 2), true, 0);
  f.vc.warm(clone2);
  f.pump(3);
  EXPECT_EQ(f.vc.res(clone2), Res::kCold);  // full: a warm load never drops anything
  EXPECT_TRUE(f.vc.ready(clone));
  EXPECT_TRUE(f.vc.ready(0));
  f.use(clone2);  // a request on it does
  EXPECT_TRUE(f.vc.ready(clone2));
  EXPECT_EQ(f.vc.res(clone), Res::kCold);  // the least recently used
  // NONE mode never warm-loads.
  Fixture g(VoiceCache::kNone, voices(2), 1);
  g.vc.warm(0);
  g.pump(3);
  EXPECT_EQ(g.vc.res(0), Res::kCold);
}

TEST(VoiceCache, CancelOrDeadlineDuringLoad) {
  for (int mode : {int(VoiceCache::kLru), int(VoiceCache::kNone)}) {
    Fixture f(mode, voices(2), 2);
    f.ld.polls = 4;
    f.vc.pin(0);
    f.pump();
    ASSERT_EQ(f.vc.res(0), Res::kLoading);
    f.vc.unpin(0);  // the request was cancelled (or hit its deadline) while its voice loaded
    f.pump(10);
    if (mode == VoiceCache::kLru)
      EXPECT_TRUE(f.vc.ready(0));  // kept, unpinned: the first to go when room is needed
    else
      EXPECT_EQ(f.vc.res(0), Res::kCold);
    // A queued (not started) load is forgotten when its only pin goes.
    f.ld.polls = 0;
    Fixture g(mode, voices(1), 2);
    g.use(0);
    g.vc.pin(1);
    g.pump();
    ASSERT_EQ(g.vc.res(1), Res::kQueued);
    g.vc.unpin(1);
    g.pump();
    EXPECT_EQ(g.vc.res(1), Res::kCold);
    EXPECT_EQ(g.ld.starts[1], 0);
  }
}

// The counters of a cancelled request, in the shape of the low-memory stress cell (LRU, one voice). The stress
// harness's replay oracle models these outcomes; a request cancelled before the conductor admits
// it never pins, so it changes no counter (engine.cc admit).
TEST(VoiceCache, CancelCountersFollowTheLoadState) {
  auto expect = [](const VoiceCache::Stats& s, uint64_t hits, uint64_t joins, uint64_t misses, uint64_t evictions,
                   uint64_t loads) {
    EXPECT_EQ(s.hits, hits);
    EXPECT_EQ(s.joins, joins);
    EXPECT_EQ(s.misses, misses);
    EXPECT_EQ(s.evictions, evictions);
    EXPECT_EQ(s.loads, loads);
  };
  {  // Cancelled while its load is queued (the resident voice is pinned): a miss, no eviction, no load.
    Fixture f(VoiceCache::kLru, voices(1), 2);
    f.use(0);
    f.vc.pin(1);
    f.pump();
    ASSERT_EQ(f.vc.res(1), Res::kQueued);
    f.vc.unpin(1);
    f.pump(5);
    EXPECT_EQ(f.vc.res(1), Res::kCold);
    EXPECT_TRUE(f.vc.ready(0));
    expect(f.vc.stats(), 0, 0, 2, 0, 1);
  }
  {  // Cancelled after its eviction started but before its load: a miss and an eviction, no load.
    Fixture f(VoiceCache::kLru, voices(1), 2);
    f.use(0);
    f.vc.unpin(0);
    f.ld.busy[0] = 3;  // the evicted voice's drop takes a few pumps
    f.vc.pin(1);
    f.pump();
    ASSERT_EQ(f.vc.res(0), Res::kUnloading);
    ASSERT_EQ(f.vc.res(1), Res::kQueued);
    f.vc.unpin(1);
    f.pump(5);
    EXPECT_EQ(f.vc.res(1), Res::kCold);
    EXPECT_EQ(f.ld.starts[1], 0);
    expect(f.vc.stats(), 0, 0, 2, 1, 1);
  }
  {  // Cancelled while its load runs: the load continues and the voice ends resident; the next request for it while it
     // still loads is a join, and one after it is a hit.
    Fixture f(VoiceCache::kLru, voices(1), 2);
    f.use(0);
    f.vc.unpin(0);
    f.ld.polls = 4;
    f.vc.pin(1);
    for (int i = 0; i < 10 && f.vc.res(1) != Res::kLoading; ++i) f.pump();
    ASSERT_EQ(f.vc.res(1), Res::kLoading);
    f.vc.unpin(1);
    f.vc.pin(1);
    for (int i = 0; i < 20 && !f.vc.ready(1); ++i) f.pump();
    ASSERT_TRUE(f.vc.ready(1));
    f.vc.unpin(1);
    expect(f.vc.stats(), 0, 1, 2, 1, 2);
    f.vc.pin(1);
    expect(f.vc.stats(), 1, 1, 2, 1, 2);
    f.vc.unpin(1);
  }
  {  // Cancelled while its load runs, with no later request: still resident in LRU (the first to go for room).
    Fixture f(VoiceCache::kLru, voices(1), 2);
    f.ld.polls = 4;
    f.vc.pin(1);
    f.pump();
    ASSERT_EQ(f.vc.res(1), Res::kLoading);
    f.vc.unpin(1);
    f.pump(10);
    EXPECT_TRUE(f.vc.ready(1));
    expect(f.vc.stats(), 0, 0, 1, 0, 1);
  }
}

TEST(VoiceCache, LoadFailureRetriesThenFailsThenRecovers) {
  Fixture f(VoiceCache::kLru, voices(2), 1);
  f.ld.fail[0] = 2;
  f.vc.pin(0);
  for (int i = 0; i < 40 && !f.vc.ready(0); ++i) {
    f.pump();
    f.t += 0.05;  // past the retry backoff
  }
  EXPECT_TRUE(f.vc.ready(0));
  EXPECT_EQ(f.ld.starts[0], 3);
  EXPECT_EQ(f.vc.stats().load_failures, 2u);
  f.vc.unpin(0);
  // Three failures in a row: the voice fails while pinned (its requests fail), and recovers once the pins are gone.
  Fixture g(VoiceCache::kLru, voices(2), 1);
  g.ld.fail[0] = 3;
  g.vc.pin(0);
  for (int i = 0; i < 40 && !g.vc.failed(0); ++i) {
    g.pump();
    g.t += 0.5;
  }
  EXPECT_TRUE(g.vc.failed(0));
  EXPECT_EQ(g.ld.starts[0], VoiceCache::kMaxAttempts);
  EXPECT_TRUE(g.ld.loaded.empty());
  g.vc.unpin(0);
  EXPECT_EQ(g.vc.res(0), Res::kCold);
  EXPECT_TRUE(g.vc.load_failed(0));  // kt2n_voice_prepare still reports the failure after the requests are gone
  g.use(0);
  EXPECT_TRUE(g.vc.ready(0));
  EXPECT_FALSE(g.vc.load_failed(0));
}

TEST(VoiceCache, BusyDropWaitsForItsReadersAndDropsNothingElse) {
  Fixture f(VoiceCache::kLru, voices(2), 3);
  f.use(0);
  f.vc.unpin(0);
  f.use(1);
  f.vc.unpin(1);
  f.ld.busy[0] = 3;  // a closing row's last render call still reads voice 0
  f.vc.pin(2);
  f.pump();
  EXPECT_EQ(f.vc.res(0), Res::kUnloading);
  EXPECT_TRUE(f.vc.ready(1));  // not dropped as well: 0's room is on its way back
  EXPECT_EQ(f.vc.res(2), Res::kQueued);
  f.pump(4);
  EXPECT_TRUE(f.vc.ready(2));
  EXPECT_TRUE(f.vc.ready(1));
  EXPECT_EQ(f.vc.res(0), Res::kCold);
  EXPECT_GE(f.vc.stats().busy_retries, 3u);
  EXPECT_EQ(f.vc.stats().evictions, 1u);
}

TEST(VoiceCache, RoomWaitDropsOneMoreVoiceOrFailsWhenAlone) {
  Fixture f(VoiceCache::kLru, voices(3), 3);
  f.use(0);
  f.vc.unpin(0);
  f.use(1);
  f.vc.unpin(1);
  f.ld.room[2] = 1;  // the LM store is fragmented: one more voice must go
  f.use(2);
  EXPECT_TRUE(f.vc.ready(2));
  EXPECT_EQ(f.vc.res(0), Res::kCold);
  EXPECT_EQ(f.vc.stats().room_waits, 1u);
  // Alone in the cache and still no room: the voice can never fit.
  Fixture g(VoiceCache::kLru, voices(3), 1);
  g.ld.room[0] = 1000000;
  g.vc.pin(0);
  for (int i = 0; i < 20 && !g.vc.failed(0); ++i) {
    g.pump();
    g.t += 0.5;
  }
  EXPECT_TRUE(g.vc.failed(0));
}

TEST(VoiceCache, RetiredVoiceIsDroppedAndNeverLoadedAgain) {
  Fixture f(VoiceCache::kLru, voices(3), 2);
  f.use(0);
  f.vc.unpin(0);
  f.vc.retire(0);
  EXPECT_EQ(f.vc.res(0), Res::kCold);
  f.vc.pin(0);  // a request that looked the voice up just before its handle was released
  f.pump(3);
  EXPECT_FALSE(f.vc.ready(0));
  EXPECT_TRUE(f.vc.failed(0));  // never served: the engine fails the request instead of letting it wait
  EXPECT_EQ(f.ld.starts[0], 1);
  EXPECT_EQ(f.ld.violations, 0);
}

// A retired voice drops its host record once, and only after its GPU state is given back (resident, busy drop,
// cold); its record is never read again (no load starts for a late pin).
TEST(VoiceCache, RetiredVoiceForgetsItsHostRecordOnceCold) {
  Fixture f(VoiceCache::kLru, voices(3), 3);
  f.use(0);
  f.vc.unpin(0);
  f.ld.busy[0] = 2;  // a reader still holds it for two polls
  f.vc.retire(0);
  EXPECT_EQ(f.ld.forgets[0], 0);
  f.pump(4);
  EXPECT_EQ(f.vc.res(0), Res::kCold);
  EXPECT_EQ(f.ld.forgets[0], 1);
  f.vc.retire(1);  // never loaded: cold at once
  EXPECT_EQ(f.ld.forgets[1], 1);
  f.vc.pin(0);
  f.vc.pin(1);
  f.pump(3);
  EXPECT_EQ(f.ld.starts[0], 1);
  EXPECT_EQ(f.ld.starts.count(1), 0u);
  EXPECT_EQ(f.ld.forgets[0], 1);
  EXPECT_EQ(f.ld.forgets[1], 1);
  EXPECT_EQ(f.vc.stats().host_bytes, 1000u);  // only voice 2's record is still counted
  EXPECT_EQ(f.ld.violations, 0);
}

// A load nothing pins (a warm load, or one whose request went away) never drops another voice to make room.
TEST(VoiceCache, UnpinnedLoadWaitingForRoomDropsNothing) {
  Fixture f(VoiceCache::kLru, voices(3), 3);
  f.use(0);
  f.vc.unpin(0);
  ASSERT_TRUE(f.vc.ready(0));
  f.ld.room[1] = 1;  // the LM store has no free range yet
  f.vc.warm(1);
  f.pump(5);
  EXPECT_EQ(f.vc.res(1), Res::kCold);
  EXPECT_TRUE(f.vc.ready(0));
  EXPECT_EQ(f.vc.stats().evictions, 0u);
  // A pinned load whose request is cancelled while it runs, then asked for room.
  f.ld.polls = 2;
  f.vc.pin(2);
  f.pump();
  ASSERT_EQ(f.vc.res(2), Res::kLoading);
  f.vc.unpin(2);
  f.ld.left[2] = 0;
  f.ld.room[2] = 1;
  f.pump(5);
  EXPECT_EQ(f.vc.res(2), Res::kCold);
  EXPECT_TRUE(f.vc.ready(0));
  EXPECT_EQ(f.vc.stats().evictions, 0u);
  EXPECT_EQ(f.vc.stats().load_failures, 0u);
  EXPECT_EQ(f.ld.violations, 0);
}

// Only the voice count is short: dropping the hidden (uncounted) voice frees nothing, so a counted one is dropped.
TEST(VoiceCache, CountOnlyShortageDropsACountedVoice) {
  Fixture f(VoiceCache::kLru, voices(1), 2);
  const int hidden = f.vc.add(cost(100, 2), false, 1000);
  f.use(hidden);
  f.vc.unpin(hidden);
  f.use(0);
  f.vc.unpin(0);
  f.use(1);
  EXPECT_TRUE(f.vc.ready(1));
  EXPECT_TRUE(f.vc.ready(hidden));
  EXPECT_EQ(f.vc.res(0), Res::kCold);
  EXPECT_EQ(f.vc.stats().evictions, 1u);
  EXPECT_EQ(f.ld.violations, 0);
}

// NONE: a kept voice (a realtime session between phrases) stays resident after its last pin, and is still dropped
// when another voice's load needs the room.
TEST(VoiceCache, NoneKeptVoiceStaysAndYieldsForRoom) {
  Fixture f(VoiceCache::kNone, voices(1), 2);
  f.use(0);
  f.vc.keep(0);
  f.vc.unpin(0);
  EXPECT_TRUE(f.vc.ready(0));
  f.use(0);
  EXPECT_EQ(f.ld.starts[0], 1);  // a hit, no reload
  f.vc.unpin(0);
  f.use(1);
  EXPECT_TRUE(f.vc.ready(1));
  EXPECT_EQ(f.vc.res(0), Res::kCold);
  f.vc.unpin(1);
  f.pump();
  EXPECT_EQ(f.vc.res(1), Res::kCold);  // not kept: dropped at once
  f.use(0);
  f.vc.unpin(0);
  EXPECT_TRUE(f.vc.ready(0));  // still kept
  f.vc.unkeep(0);
  f.pump();
  EXPECT_EQ(f.vc.res(0), Res::kCold);
  EXPECT_EQ(f.ld.violations, 0);
}

// A queued voice whose pins all go loses its retry backoff and its warm flag after a permanent failure.
TEST(VoiceCache, NoStaleBackoffOrWarmFlag) {
  Fixture f(VoiceCache::kLru, voices(3), 2);
  f.ld.fail[0] = 1;
  f.vc.pin(0);
  f.pump(2);  // fails once: queued again with a backoff
  ASSERT_EQ(f.vc.res(0), Res::kQueued);
  f.vc.unpin(0);
  EXPECT_EQ(f.vc.res(0), Res::kCold);
  f.vc.pin(0);
  f.pump(2);  // same time: no inherited backoff
  EXPECT_EQ(f.ld.starts[0], 2);
  EXPECT_TRUE(f.vc.ready(0));
  f.ld.fail[1] = VoiceCache::kMaxAttempts;
  f.vc.warm(1);
  for (int i = 0; i < 40 && f.ld.starts[1] < VoiceCache::kMaxAttempts; ++i) {
    f.pump();
    f.t += 1.0;
  }
  f.pump(3);
  ASSERT_EQ(f.vc.res(1), Res::kCold);
  f.vc.pin(1);
  f.vc.unpin(1);
  EXPECT_EQ(f.vc.res(1), Res::kCold);  // no load left queued that nothing needs
  f.pump(3);
  EXPECT_EQ(f.ld.starts[1], VoiceCache::kMaxAttempts);
  EXPECT_EQ(f.ld.violations, 0);
}

// Shutdown gives back at once a load that has no GPU work queued (it may wait for pages that no row returns any more)
// and waits only for the others.
TEST(VoiceCache, DrainAbandonsLoadsWithoutQueuedWork) {
  Fixture f(VoiceCache::kLru, voices(3), 2);
  f.ld.polls = 1000000;
  f.vc.pin(0);
  f.pump();
  ASSERT_EQ(f.vc.res(0), Res::kLoading);
  f.ld.no_gpu_work.insert(0);
  f.ld.polls = 3;
  f.vc.pin(1);
  f.pump();
  ASSERT_EQ(f.vc.res(1), Res::kLoading);
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_EQ(f.vc.drain(30.0), 0);
  EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), 1.0);
  EXPECT_TRUE(f.vc.failed(0));  // its request fails (the engine is going away)
  EXPECT_EQ(f.ld.unloads[0], 1);
  EXPECT_TRUE(f.vc.ready(1));
  EXPECT_EQ(f.vc.stats().load_failures, 0u);
  EXPECT_EQ(f.ld.violations, 0);
}

TEST(VoiceCache, DrainWaitsForLoadsInFlight) {
  Fixture f(VoiceCache::kLru, voices(3), 3);
  f.ld.polls = 30;
  f.vc.pin(0);
  f.vc.pin(1);
  f.pump();
  ASSERT_EQ(f.vc.res(0), Res::kLoading);
  EXPECT_EQ(f.vc.drain(5.0), 0);
  EXPECT_TRUE(f.vc.ready(0));
  EXPECT_TRUE(f.vc.ready(1));
  f.vc.pin(2);
  f.vc.clear_pending();  // engine failure: queued loads are forgotten
  f.pump(3);
  EXPECT_EQ(f.ld.starts[2], 0);
}

TEST(VoiceCache, StatsJson) {
  Fixture f(VoiceCache::kLru, voices(1), 2);
  f.ld.polls = 2;
  f.use(0);
  f.t = 1.0;
  f.vc.unpin(0);
  f.vc.pin(0);  // hit
  f.vc.unpin(0);
  f.use(1);     // miss + eviction of 0
  const auto st = f.vc.stats();
  EXPECT_EQ(st.hits, 1u);
  EXPECT_EQ(st.misses, 2u);
  EXPECT_EQ(st.evictions, 1u);
  EXPECT_EQ(st.resident, 1);
  EXPECT_EQ(st.pinned, 1);
  EXPECT_EQ(st.host_bytes, 2000u);
  EXPECT_EQ(st.gpu_lm_bytes, 100u);
  EXPECT_EQ(st.gpu_page_bytes, 2000u);
  EXPECT_EQ(st.gpu_table_bytes, 10u);
  const std::string j = f.vc.stats_json();
  for (const char* k : {"\"mode\":\"lru\"", "\"resident\":1", "\"pinned\":1", "\"hits\":1", "\"misses\":2",
                        "\"evictions\":1", "\"gpu_bytes\":{\"lm\":100,\"pages\":2000,\"tables\":10,\"total\":2110}",
                        "\"host_bytes\":2000", "\"load_ms\":{\"p50\":", "\"waits\":0", "\"busy_retries\":0",
                        "\"load_failures\":0"})
    EXPECT_NE(j.find(k), std::string::npos) << k << " in " << j;
}

// 47 voices round-robin through an LRU of 10 at concurrency 1, 2 and 4: every request completes, the budget holds at
// every step, pinned voices are never dropped.
TEST(VoiceCache, RoundRobin47VoicesLru10) {
  for (int conc : {1, 2, 4}) {
    Fixture f(VoiceCache::kLru, voices(10), 47);
    f.ld.polls = 2;
    struct Req {
      int voice = -1;
      int steps = 0;
      bool was_ready = false;
    };
    std::vector<Req> running;
    int next = 0, done = 0;
    const int total = 47 * 3;
    for (int turn = 0; turn < 20000 && done < total; ++turn) {
      while (int(running.size()) < conc && next < total) {
        running.push_back({next % 47, 0});
        f.vc.pin(next % 47);
        ++next;
      }
      f.pump();
      ASSERT_LE(f.vc.resident_count(), 10);
      ASSERT_LE(int(f.ld.loaded.size()), 10);
      for (auto it = running.begin(); it != running.end();) {
        // Once resident, a pinned voice stays resident until its request unpins it (never dropped and reloaded).
        if (it->was_ready) {
          ASSERT_TRUE(f.vc.ready(it->voice)) << "voice " << it->voice << " concurrency " << conc;
        }
        if (f.vc.ready(it->voice)) it->was_ready = true;
        if (f.vc.ready(it->voice) && ++it->steps >= 3) {  // it ran on its resident voice
          f.vc.unpin(it->voice);
          it = running.erase(it);
          ++done;
        } else {
          ++it;
        }
      }
    }
    EXPECT_EQ(done, total) << "concurrency " << conc;
    EXPECT_EQ(f.ld.violations, 0);
    const auto st = f.vc.stats();
    EXPECT_GE(st.evictions, uint64_t(total - 10)) << "concurrency " << conc;
    EXPECT_EQ(st.load_failures, 0u);
  }
}

}  // namespace
