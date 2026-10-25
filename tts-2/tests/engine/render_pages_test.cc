// The decoder backend's pool page rules (src/engine/render_pages.h) on a CPU pool model. FakeDecoder follows the order
// of calls L7Render makes (backend_l7.cu: add_voice, open_row, extend_noise, launch, close_row) and keeps the pool's
// physical pages the way the decoder does: a built prefix holds 8 pages, a row allocates pages as its commits grow, and a
// commit that finds too few free pages is the decoder's no_pages.
//
// The sizes are the RTX PRO 6000 stress run that failed (default profile, cache mode ALL, 8 streams, 47 built-in voices
// + 130 registered clones): a 1,030-page pool (catalogue 422 + 32 custom voices x 9 + 40 x 8 rows), prefix cap 710, a
// row of budget 334 on a 250-token prompt that reserves 12 pages and needs 22 once it runs past its budget.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

#include "render_pages.h"

using kt2n::engine::RenderPageBook;
using Admit = RenderPageBook::Admit;

namespace {

constexpr int kPrefixBound = 9;  // (2 x 250 + 63) / 64 + 1: the bound the backend checks
constexpr int kPrefixPages = 8;  // what a built 250-token prefix holds

struct FakeDecoder {
  struct Row {
    int voice = 0, reserved = 0, used = 0;
  };
  explicit FakeDecoder(int pool, int cap, bool evict_while_reserved = false, bool reserve = true)
      : free(pool), reserve_pages(reserve),
        book(pool, cap, evict_while_reserved, false, [this](int v) { return unbuild(v); }) {}

  // The decoder refuses to unbuild a voice that a row reads (-2).
  int unbuild(int v) {
    for (const auto& kv : rows)
      if (kv.second.voice == v) return -2;
    if (!book.built(v)) return -1;
    free += kPrefixPages;
    book.forget(v);
    ++evictions;
    return 0;
  }
  void build(int v) {
    ASSERT_GE(free, kPrefixPages);
    free -= kPrefixPages;
    book.on_built(v, kPrefixPages);
  }
  // add_voice with prefixes built at registration (cloning on, cache mode ALL).
  void add_voice(int v) {
    if (book.build_at_add(kPrefixBound)) build(v);
  }
  // open_row: a first-use prefix, then the row's reservation.
  Admit open_row(uint64_t id, int v, int need) {
    book.touch(v);
    if (!book.built(v)) {
      const auto a = book.admit_prefix(v, kPrefixBound, need);
      if (a != Admit::kOk) return a;
      build(v);
    }
    if (reserve_pages) {
      const auto a = book.admit_row(v, need);
      if (a != Admit::kOk) return a;
      book.reserve(need);
    }
    rows[id] = Row{v, reserve_pages ? need : 0, 0};
    return Admit::kOk;
  }
  void extend(uint64_t id, int full) {
    Row& r = rows.at(id);
    const int extra = std::max(0, full - r.reserved);
    book.extend(r.voice, extra);
    r.reserved += extra;
  }
  // A commit that grows the row to `pages` own pages; false = no_pages (the request fails).
  bool commit(uint64_t id, int pages) {
    Row& r = rows.at(id);
    const int need = pages - r.used;
    while (need > free && book.evict_one_idle()) ++launch_retries;
    if (need > free) return false;
    free -= need;
    r.used = pages;
    return true;
  }
  void close(uint64_t id) {
    Row r = rows.at(id);
    rows.erase(id);
    free += r.used;
    book.release(r.reserved);
  }

  int free;  // the pool's physical free pages
  bool reserve_pages;
  long evictions = 0, launch_retries = 0;
  std::map<uint64_t, Row> rows;
  RenderPageBook book;
};

constexpr int kPool = 1030, kCap = kPool - 40 * 8, kVoices = 47 + 130, kNeed = 12, kFull = 22;

void register_all(FakeDecoder& d) {
  for (int v = 0; v < kVoices; ++v) d.add_voice(v);
}
// Admit 8 rows on voices 0..7 (clients that each send one request); returns how many were admitted.
int admit_eight(FakeDecoder& d) {
  int admitted = 0;
  for (int i = 0; i < 8; ++i) admitted += d.open_row(uint64_t(i), i, kNeed) == Admit::kOk;
  return admitted;
}

}  // namespace

TEST(RenderPages, UncappedRegistrationTakesTheRowsShareAndSerialisesRequests) {
  // The rule before the cap (prefix_cap_pages 0): every registered voice builds while it fits the unreserved pages.
  FakeDecoder d(kPool, 0);
  register_all(d);
  EXPECT_EQ(d.book.prefix_pages(), 128 * kPrefixPages);  // 1,024 of 1,030 pages: the rows' 320 pages are gone
  EXPECT_LT(d.book.avail(), kNeed);
  // The first row evicts an idle prefix for itself (nothing reserved); every other row waits behind it.
  EXPECT_EQ(admit_eight(d), 1);
}

TEST(RenderPages, CapKeepsTheRowsShareSoEightRowsRunTogether) {
  FakeDecoder d(kPool, kCap);
  register_all(d);
  EXPECT_LE(d.book.prefix_pages(), kCap);
  EXPECT_GE(d.book.avail(), 40 * 8);
  EXPECT_EQ(admit_eight(d), 8);
  EXPECT_EQ(d.evictions, 0);
  // A voice registered past the cap builds on first use and evicts an idle prefix (least recently used) to stay under
  // it, while the 8 rows hold their reservations.
  ASSERT_FALSE(d.book.built(kVoices - 1));
  d.close(7);
  EXPECT_EQ(d.open_row(100, kVoices - 1, kNeed), Admit::kOk);
  EXPECT_TRUE(d.book.built(kVoices - 1));
  EXPECT_LE(d.book.prefix_pages(), kCap);
  EXPECT_EQ(d.evictions, 1);
  for (int v = 0; v < 7; ++v) EXPECT_TRUE(d.book.built(v)) << "a voice with an open row lost its prefix: " << v;
}

TEST(RenderPages, ExtensionPastBudgetTakesIdlePrefixPagesInsteadOfNoPages) {
  // The failing request's state: registration without the cap left the pool full of idle prefixes, and one row runs
  // past its budget.
  FakeDecoder d(kPool, 0);
  register_all(d);
  ASSERT_EQ(d.open_row(42, 0, kNeed), Admit::kOk);
  ASSERT_TRUE(d.commit(42, kNeed));  // the row grows to its reservation
  // No unreserved page is left for the extension's 10 pages, and fewer than 10 are physically free: before this rule
  // the extension was booked as an overcommit and a later commit failed with no_pages.
  ASSERT_LT(d.book.avail(), kFull - kNeed);
  ASSERT_LT(d.free, kFull - kNeed);
  d.extend(42, kFull);
  EXPECT_EQ(d.book.overcommits(), 0);
  EXPECT_GE(d.book.avail(), 0);
  for (int p = kNeed + 1; p <= kFull; ++p) ASSERT_TRUE(d.commit(42, p)) << "no_pages at " << p << " pages";
  EXPECT_EQ(d.launch_retries, 0);
  d.close(42);
  EXPECT_EQ(d.free + d.book.prefix_pages(), kPool);  // nothing leaked
  EXPECT_EQ(d.book.reserved(), 0);
}

TEST(RenderPages, EightRowsAllExtendingWithCloneChurnNeverRunOutOfPages) {
  // c=8 for many rounds: every request takes a voice in turn from the whole set (first uses past the cap evict idle
  // prefixes), and every request runs past its budget. No commit may meet an empty pool and no extension overcommits.
  FakeDecoder d(kPool, kCap);
  register_all(d);
  uint64_t next = 0;
  int voice = 0, failed = 0, waited = 0, max_open = 0;
  std::map<uint64_t, int> stage;  // row -> 0 open, 1 at reservation, 2 extended and done
  for (int round = 0; round < 400; ++round) {
    while (d.rows.size() < 8) {
      const auto a = d.open_row(next, voice, kNeed);
      if (a != Admit::kOk) {
        ASSERT_EQ(a, Admit::kWait);
        ++waited;
        break;
      }
      stage[next++] = 0;
      voice = (voice + 37) % kVoices;
    }
    max_open = std::max<int>(max_open, int(d.rows.size()));
    std::vector<uint64_t> done;
    for (auto& [id, s] : stage) {
      if (s == 0) {
        failed += !d.commit(id, kNeed);
        s = 1;
      } else if (s == 1) {
        d.extend(id, kFull);
        failed += !d.commit(id, kFull);
        s = 2;
      } else {
        done.push_back(id);
      }
    }
    for (uint64_t id : done) {
      d.close(id);
      stage.erase(id);
    }
  }
  EXPECT_EQ(failed, 0);
  EXPECT_EQ(d.book.overcommits(), 0);
  EXPECT_EQ(max_open, 8);
  EXPECT_EQ(waited, 0);
  EXPECT_LE(d.book.prefix_pages(), kCap);
  EXPECT_GT(d.evictions, 0);
}

TEST(RenderPages, LaunchWithoutFreePagesGivesBackIdlePrefixesOrFailsWhenNoneIsIdle) {
  // Rows without reservations (KT2N_L7_RESERVE=0) can meet a pool held by prefixes: the launch takes idle prefixes'
  // pages (least recently used first), and fails only when every prefix is in use.
  FakeDecoder d(64, 0, false, /*reserve=*/false);
  for (int v = 0; v < 6; ++v) d.add_voice(v);  // 48 prefix pages, 16 free
  ASSERT_EQ(d.book.prefix_pages(), 48);
  ASSERT_EQ(d.open_row(1, 0, 0), Admit::kOk);
  ASSERT_EQ(d.open_row(2, 1, 0), Admit::kOk);
  EXPECT_TRUE(d.commit(1, 30));  // 16 free + voices 2..5 idle (oldest first)
  EXPECT_EQ(d.launch_retries, 2);
  EXPECT_FALSE(d.book.built(2));
  EXPECT_FALSE(d.book.built(3));
  EXPECT_TRUE(d.book.built(4));
  EXPECT_TRUE(d.commit(2, 16));
  EXPECT_EQ(d.launch_retries, 4);
  EXPECT_FALSE(d.commit(1, 40));  // only voices 0 and 1 are left, and both are in use: no_pages
  EXPECT_TRUE(d.book.built(0));
  EXPECT_TRUE(d.book.built(1));
}

TEST(RenderPages, CapOffAndEngineManagedPrefixesKeepTheEarlierRules) {
  // No cap: registration builds while the unreserved pages hold a prefix, as before.
  RenderPageBook a(100, 0, false, false, nullptr);
  EXPECT_TRUE(a.prefix_fits(1000));
  EXPECT_TRUE(a.build_at_add(9));
  // Engine-managed prefixes (cache modes LRU and NONE): the backend never evicts, so the voice cache's accounting holds.
  int calls = 0;
  RenderPageBook m(40, 0, false, true, [&](int) {
    ++calls;
    return 0;
  });
  m.on_built(1, 8);
  m.reserve(30);
  m.extend(2, 10);
  EXPECT_EQ(m.overcommits(), 1);
  EXPECT_FALSE(m.evict_one_idle());
  EXPECT_EQ(m.admit_row(2, 12), Admit::kWait);
  EXPECT_EQ(calls, 0);
}
