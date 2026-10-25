// Pool page bookkeeping of the decoder backend (backend_l7.cu): the pages built voice prefixes hold, the pages open
// rows have reserved, and the rules that decide when a prefix is built, when a row waits (back-pressure), and which
// idle prefixes give their pages back. CPU only (no decoder types), so the rules have CPU tests
// (tests/engine/render_pages_test.cc).
//
//   avail()        total - prefix pages - reserved pages: what admission may still promise.
//   prefix cap     > 0 (default profile, cache mode ALL): prefixes hold at most this many pages, so the rows' share of
//                  the pool (40 pages per stream in the engine's sizing) stays theirs. Without it, registering more
//                  voices than the sizing counted built prefixes into every unreserved page, and requests were then
//                  admitted one at a time (an idle prefix is evicted for a row only when nothing is reserved).
//   eviction       an IDLE prefix (no row, call or build reads the voice: the decoder refuses the unbuild otherwise)
//                  gives its pages back, least recently used first; its voice rebuilds it on its next use.
//   managed        the engine's voice cache owns prefix residency (cache modes LRU and NONE): nothing is evicted here.
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kt2n::engine {

class RenderPageBook {
 public:
  // Unbuild one voice's prefix: 0 when its pages are back in the pool (the callee calls forget(voice)); anything else
  // when the voice is busy or has no prefix.
  using Unbuild = std::function<int(int voice)>;
  enum class Admit { kOk, kWait, kNever };  // kWait: back-pressure (rows hold the pages); kNever: it can never fit

  RenderPageBook() = default;
  RenderPageBook(int total_pages, int prefix_cap, bool evict_while_reserved, bool managed, Unbuild unbuild)
      : total_(total_pages), cap_(prefix_cap), evict_while_reserved_(evict_while_reserved), managed_(managed),
        unbuild_(std::move(unbuild)) {}

  int total() const { return total_; }
  int cap() const { return cap_; }
  int prefix_pages() const { return prefix_.load(); }
  int reserved() const { return reserved_.load(); }
  long overcommits() const { return overcommit_.load(); }
  int avail() const { return total_ - prefix_ - reserved_; }
  bool prefix_fits(int pages) const { return cap_ <= 0 || prefix_ + pages <= cap_; }
  bool built(int voice) const { return pages_of_.count(voice) != 0; }

  // Bookkeeping.
  void touch(int voice) { last_use_[voice] = ++tick_; }  // open_row: the LRU clock
  void on_built(int voice, int pages) {
    prefix_ += pages;
    pages_of_[voice] = pages;
  }
  void forget(int voice) {  // the voice's prefix pages are back in the pool (unbuilt or removed)
    auto it = pages_of_.find(voice);
    if (it == pages_of_.end()) return;
    prefix_ -= it->second;
    pages_of_.erase(it);
  }
  void erase_voice(int voice) {
    forget(voice);
    last_use_.erase(voice);
  }
  void reserve(int pages) { reserved_ += pages; }
  void release(int pages) { reserved_ -= pages; }

  // Rules.
  // Registration (prefixes built at add_voice): build now only when the prefix fits the unreserved pages and the cap;
  // else the voice builds it on its first use.
  bool build_at_add(int prefix_bound) const { return prefix_bound <= avail() && prefix_fits(prefix_bound); }
  // First use of a voice whose prefix is unbuilt: `prefix_bound` pages for the prefix, `row_bound` (an upper bound of the
  // row's own pages) is counted too when evicting under reservations, so the new prefix does not sit idle while its row
  // waits. Over the cap, idle prefixes are evicted down to it even while rows hold reservations: idle prefix pages are
  // never reserved pages. kOk: build the prefix now.
  Admit admit_prefix(int voice, int prefix_bound, int row_bound) {
    const bool under = reserved_ > 0 && evict_while_reserved_ && !managed_;
    const int want = under ? prefix_bound + row_bound : prefix_bound;
    const bool for_avail = reserved_ == 0 || evict_while_reserved_;
    if (!managed_ && (!prefix_fits(prefix_bound) || (want > avail() && for_avail)))
      evict_idle(for_avail ? want : 0, voice, prefix_bound);
    if (want > avail()) return reserved_ == 0 ? Admit::kNever : Admit::kWait;
    return Admit::kOk;
  }
  // A row reserving `need` pages: with nothing reserved (or evicting under reservations) idle prefixes make room.
  // kOk: the caller opens the row and then reserve(need).
  Admit admit_row(int voice, int need) {
    if (need > avail() && (reserved_ == 0 || evict_while_reserved_) && !managed_) evict_idle(need, voice);
    if (need > avail()) return reserved_ == 0 ? Admit::kNever : Admit::kWait;
    return Admit::kOk;
  }
  // A row past its budget books `extra` pages for its noise extension. They come out of idle prefixes when no
  // unreserved page is left; an extension the pool still cannot cover runs on the pool's real free pages and is counted
  // (overcommits()).
  void extend(int voice, int extra) {
    if (extra > 0 && extra > avail() && !managed_) evict_idle(extra, voice);
    if (extra > avail()) overcommit_++;
    reserved_ += extra;
  }
  // A launch found no free page: one idle prefix (least recently used) gives its pages back. false: none could
  // (every built prefix is in use, or the cache owns them).
  bool evict_one_idle() {
    if (managed_ || !unbuild_) return false;
    for (const auto& c : lru(-1))
      if (unbuild_(c.second) == 0) return true;
    return false;
  }
  // Idle prefixes, least recently used first (never `keep`), until `want` pages are available and, with `prefix_bound`
  // > 0, a new prefix of that many pages fits under the cap.
  void evict_idle(int want, int keep, int prefix_bound = 0) {
    if (!unbuild_) return;
    for (const auto& c : lru(keep)) {
      if (avail() >= want && (prefix_bound <= 0 || prefix_fits(prefix_bound))) break;
      unbuild_(c.second);
    }
  }

 private:
  std::vector<std::pair<uint64_t, int>> lru(int keep) const {
    std::vector<std::pair<uint64_t, int>> v;
    v.reserve(pages_of_.size());
    for (const auto& b : pages_of_)
      if (b.first != keep) {
        auto t = last_use_.find(b.first);
        v.push_back({t != last_use_.end() ? t->second : 0, b.first});
      }
    std::sort(v.begin(), v.end());
    return v;
  }
  int total_ = 0;
  int cap_ = 0;
  bool evict_while_reserved_ = false;
  bool managed_ = false;
  Unbuild unbuild_;
  // Atomics: the stats thread reads them (the rest is the conductor's).
  std::atomic<int> prefix_{0}, reserved_{0};
  std::atomic<long> overcommit_{0};
  std::unordered_map<int, int> pages_of_;       // built voice -> its prefix pages
  std::unordered_map<int, uint64_t> last_use_;  // voice -> open_row tick
  uint64_t tick_ = 0;
};

}  // namespace kt2n::engine
