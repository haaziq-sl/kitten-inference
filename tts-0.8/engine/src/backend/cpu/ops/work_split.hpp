#pragma once
// How the GEMM kernels split their output tiles among the threads of an OpenMP team.
//
// The kernels compute every output tile whole on one thread (each tile's sum runs over all of K in a fixed order),
// so how the tiles are split never changes a value: only which thread computes which tile.
//
// Static split (the default): thread tid takes the tid-th of nT equal ranges of tiles, as these kernels always did.
// Dynamic split (kt_dynamic_chunks() > 0): the tiles are cut into nT * kt_dynamic_chunks() ranges, handed out to
// the threads as they finish one (schedule(dynamic, 1)), so a slower core takes fewer of them.
#include <algorithm>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif

inline std::atomic<int>& kt_dynamic_chunks_setting() {
    static std::atomic<int> chunks{0};
    return chunks;
}

inline int kt_dynamic_chunks() { return kt_dynamic_chunks_setting().load(std::memory_order_relaxed); }

// Calls run(mi_s, mi_e, co_s, co_e, tail) for this thread's share of an m_tiles x co_tiles grid of output tiles.
// n_part: split over the co tiles (every share has all m tiles) instead of the m tiles. tail: this share also
// computes the M % MR leftover rows for its co range (once per co range, as in the static split).
// team_call: every thread of the current team makes this same call (the dynamic split is a worksharing loop, so it
// is used only then).
template <class Run>
inline void kt_split_tiles(bool n_part, int m_tiles, int co_tiles, int tid, int nT, bool team_call, Run&& run) {
    const int units = n_part ? co_tiles : m_tiles;
#ifdef _OPENMP
    const int per_thread = kt_dynamic_chunks();
    if (per_thread > 0 && team_call && nT > 1 && omp_get_num_threads() == nT) {
        const int chunks = std::max(1, std::min(units, nT * per_thread));
#pragma omp for schedule(dynamic, 1) nowait
        for (int c = 0; c < chunks; ++c) {
            const int s = (int)((long long)c * units / chunks);
            const int e = (int)((long long)(c + 1) * units / chunks);
            if (n_part) run(0, m_tiles, s, e, true);
            else        run(s, e, 0, co_tiles, c == 0);
        }
        return;
    }
#endif
    const int s = (int)((long long)tid * units / nT);
    const int e = (int)((long long)(tid + 1) * units / nT);
    if (n_part) run(0, m_tiles, s, e, true);
    else        run(s, e, 0, co_tiles, tid == 0);
}
