// kt2n p2-prefill: the prefill context and one prefill call (prefix build, single suffix, burst of suffixes).
// Composes the verified kernel-library operators; see include/kt2n/prefill.h for the contract.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "kt2n/kernels/attn.h"
#include "kt2n/kernels/ops.h"
#include "kt2n/kernels/samp.h"
#include "kt2n/kernels/tern_proj.h"
#include "kt2n/prefill.h"
#include "kt2n/cuda_caps.h"
#if defined(KT2N_HIP)
#include "kt2n/hip/wmma_kernels.h"
#endif
#include "pf_attn.h"
#include "pf_internal.h"
#include "pf_tern.h"

namespace kt2n_pf {

namespace {
thread_local std::string g_last_error;
}

void set_last_error(const std::string& msg) noexcept {
  try {
    g_last_error = msg;
  } catch (...) {
  }
}

namespace {

constexpr size_t kAlign = 256;
size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

// out[r, :] = in[idx[r], :] for two bf16 matrices with row length n (n % 8 == 0); one block per row.
__global__ void gather_two_kernel(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
                                  int64_t ld, const int32_t* __restrict__ idx, __nv_bfloat16* __restrict__ a_out,
                                  __nv_bfloat16* __restrict__ b_out, int64_t ld_out, int n) {
  const int r = blockIdx.x;
  const int64_t src = static_cast<int64_t>(idx[r]) * ld;
  const int64_t dst = static_cast<int64_t>(r) * ld_out;
  const uint4* a4 = reinterpret_cast<const uint4*>(a + src);
  const uint4* b4 = reinterpret_cast<const uint4*>(b + src);
  uint4* ao = reinterpret_cast<uint4*>(a_out + dst);
  uint4* bo = reinterpret_cast<uint4*>(b_out + dst);
  for (int i = threadIdx.x; i < n / 8; i += blockDim.x) {
    ao[i] = a4[i];
    bo[i] = b4[i];
  }
}

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw PfError(KT2N_PF_ECUDA, std::string(what) + ": " + cudaGetErrorString(e));
}
void kt(int rc, const char* what) {  // k1 codes
  if (rc == KT2N_TERN_ECUDA) throw PfError(KT2N_PF_ECUDA, std::string(what) + ": launch failed");
  if (rc != KT2N_TERN_OK) throw PfError(KT2N_PF_EKERNEL, std::string(what) + ": bad arguments (rc " + std::to_string(rc) + ")");
}
void ko(cudaError_t e, const char* what) {  // k2 codes
  if (e == cudaErrorInvalidValue) throw PfError(KT2N_PF_EKERNEL, std::string(what) + ": bad arguments");
  ck(e, what);
}
void ka(int rc, const char* what) {  // k3 codes
  if (rc == KT2N_ATTN_EINVAL) throw PfError(KT2N_PF_EKERNEL, std::string(what) + ": bad arguments");
  if (rc != 0) throw PfError(KT2N_PF_ECUDA, std::string(what) + ": " + cudaGetErrorString((cudaError_t)rc));
}
void ks(int rc, const char* what) {  // k4 codes
  if (rc < 0) throw PfError(KT2N_PF_EKERNEL, std::string(what) + ": bad arguments (rc " + std::to_string(rc) + ")");
  if (rc != 0) throw PfError(KT2N_PF_ECUDA, std::string(what) + ": " + cudaGetErrorString((cudaError_t)rc));
}

}  // namespace
}  // namespace kt2n_pf

struct kt2n_pf_ctx {
  kt2n_pf_weights w{};
  int device = 0;
  bool supports_pdl = false;
#if defined(KT2N_HIP)
  bool wmma = false;  // projections on the AMD matrix unit (kt2n/hip/wmma_kernels.h)
#endif
  int max_rows = 0, max_tokens = 0, max_keys = 0;
  // device arena
  void* arena = nullptr;
  size_t bytes = 0;
  bool owns_arena = true;
  __nv_bfloat16 *hidden = nullptr, *normed = nullptr, *qkv = nullptr, *qrot = nullptr, *attn = nullptr,
                *act = nullptr, *cos = nullptr, *sin = nullptr;
  __nv_bfloat16 *last_h = nullptr, *last_a = nullptr, *last_n = nullptr;
  char* d_in = nullptr;  // packed per-call inputs (mirror of the pinned buffer)
  size_t in_cap = 0;
  void* attn_ws = nullptr;
  size_t attn_ws_bytes = 0;
  // host
  char* h_in = nullptr;  // pinned
  // weights.host_embed: each call's new-token embedding rows are gathered here (pinned, [max_tokens][hidden] bf16)
  // and uploaded into the `attn` buffer (free until layer 0's attention), which the embed kernel then reads.
  char* h_rows = nullptr;
  cudaEvent_t h2d_done = nullptr;
  bool h2d_pending = false;
  // KT2N_PF_FLAG_OVERLAP_PAST: side stream for the past K/V copies + fork / per-layer-group events
  unsigned force_flags = 0;
  int q8_tq = 0;  // KT2N_PF_FLAG_ATTN_Q8 queries per tile: 0 = measured policy; env KT2N_PF_Q8_TQ (1..8) forces it
  cudaStream_t side = nullptr;
  cudaEvent_t ev_fork = nullptr;
  cudaEvent_t ev_grp[4] = {nullptr, nullptr, nullptr, nullptr};
};

namespace kt2n_pf {
namespace {

void ctx_free(kt2n_pf_ctx* c) {
  if (!c) return;
  int prev = 0;
  cudaGetDevice(&prev);
  cudaSetDevice(c->device);
  if (c->h2d_done) {
    cudaEventSynchronize(c->h2d_done);
    cudaEventDestroy(c->h2d_done);
  }
  if (c->side) {
    cudaStreamSynchronize(c->side);
    cudaStreamDestroy(c->side);
  }
  if (c->ev_fork) cudaEventDestroy(c->ev_fork);
  for (cudaEvent_t& e : c->ev_grp)
    if (e) cudaEventDestroy(e);
  if (c->arena && c->owns_arena) cudaFree(c->arena);
  if (c->h_in) cudaFreeHost(c->h_in);
  if (c->h_rows) cudaFreeHost(c->h_rows);
  cudaSetDevice(prev);
  delete c;
}

size_t input_bytes(int rows, int tokens) {
  // ids, positions, row_slot (int64), speaker_index (int32), two descriptor arrays, gather index,
  // decode-rows attention slot / key count (int32 per token; KT2N_PF_FLAG_ATTN_DECODE_ROWS)
  return align_up((size_t)tokens * 8) * 3 + align_up((size_t)tokens * 4) +
         2 * align_up((size_t)rows * sizeof(kt2n_attn_prefill_row)) + align_up((size_t)rows * 4) +
         2 * align_up((size_t)tokens * 4) + align_up((size_t)tokens * sizeof(kt2n_pf_q8_tile));
}

struct Layout {
  size_t hidden, normed, qkv, qrot, attn, act, cos, sin, lh, la, ln, in, ws, ws_bytes, in_cap, total;
};

// KT2N_PF_CTX_FIT_ATTN_WS (one-row calls): the most query tiles a one-row call can have and still take
// kt2n_attn_prefill's split-parallel shape on `device` (4 x tiles x HKV x rows < 3 x SMs, attn_prefill.cu), at most the
// default 17. 0 = the default layout.
int split_tiles_for(int device) {
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || sms < 1) {
    cudaGetLastError();
    return 17;
  }
  return std::max(1, std::min(17, (3 * sms - 1) / (4 * 8)));
}

Layout layout(int max_rows, int max_tokens, int max_keys, int split_tiles = 0) {
  const size_t H = 2048, I = 6144, T = (size_t)max_tokens, R = (size_t)max_rows, qkv_n = 4096;
  Layout L{};
  // attention workspace: k3 picks its split-parallel shape only when rows x query tiles (32 queries) <= 17
  // (4 * tiles * 8 heads * rows < 3/4 of 188 SMs on the RTX PRO 6000) and the workspace covers the call; any
  // other call runs the workspace-free sequential shape (same bits).  17 tiles over up to 2048 keys covers every
  // call that would choose split-parallel (prefix builds, single suffixes, small bursts).
  L.ws_bytes = kt2n_attn_prefill_workspace_bytes(1, 17 * 32, std::min(max_keys, 2048));
  // The same workspace serves the attention library's decode kernel (KT2N_PF_FLAG_ATTN_DECODE_ROWS: <= 64 rows per launch over up to
  // 2048 keys); stream-ordered launches of both kinds may share it (attn.h).  Smaller than the prefill part.
  L.ws_bytes = std::max(L.ws_bytes, kt2n_attn_decode_workspace_bytes(64, std::min(max_keys, 2048)));
  if (split_tiles > 0) {
    // KT2N_PF_CTX_FIT_ATTN_WS, calls of one row: every kernel picks its shape by "the workspace covers the call", so
    // the workspace here is the largest need, among the one-row calls this GPU can run split, that the default
    // workspace above also covers -- each call then makes the same choice as with the default workspace:
    //   kt2n_attn_prefill: split only at <= split_tiles tiles, over up to max_keys keys;
    //   kt2n_attn_decode_rows: <= 64 rows over up to 2048 keys (as above);
    //   kt2n_pf_q8_attention: <= KT2N_PF_Q8_MAX_T tokens in tiles of >= 8 queries (4 up to 64 tokens): <= 20 tiles.
    const size_t full = L.ws_bytes;
    size_t need = std::max(kt2n_attn_prefill_workspace_bytes(1, split_tiles * 32, max_keys),
                           kt2n_attn_decode_workspace_bytes(64, std::min(max_keys, 2048)));
    need = std::max(need, kt2n_pf_q8_workspace_bytes(std::max(64 / 4, (KT2N_PF_Q8_MAX_T + 7) / 8), max_keys));
    L.ws_bytes = std::min(full, need);
  }
  L.in_cap = input_bytes(max_rows, max_tokens);
  size_t off = 0;
  auto take = [&](size_t b) {
    size_t o = off;
    off = align_up(off + b);
    return o;
  };
  L.hidden = take(T * H * 2);
  L.normed = take(T * H * 2);
  L.qkv = take(T * qkv_n * 2);
  L.qrot = take(T * H * 2);
  L.attn = take(T * H * 2);
  L.act = take(T * I * 2);
  L.cos = take(T * 64 * 2);
  L.sin = take(T * 64 * 2);
  L.lh = take(R * H * 2);
  L.la = take(R * H * 2);
  L.ln = take(R * H * 2);
  L.in = take(L.in_cap);
  L.ws = take(L.ws_bytes);
  L.total = off;
  return L;
}

void check_limits(int max_rows, int max_tokens, int max_keys) {
  if (max_rows < 1 || max_rows > 64 || max_tokens < max_rows || max_tokens > (1 << 16) || max_keys < 1 ||
      max_keys > KT2N_ATTN_MAX_CTX)
    throw PfError(KT2N_PF_EINVAL, "bad context limits");
}

kt2n_pf_ctx* ctx_create(const kt2n_pf_weights* w, int device, int max_rows, int max_tokens, int max_keys,
                        void* workspace, size_t workspace_bytes, cudaStream_t stream, unsigned flags = 0) {
  if (!w) throw PfError(KT2N_PF_EINVAL, "weights is NULL");
  if (w->layers < 1 || w->layers > KT2N_PF_MAX_LAYERS || w->hidden != 2048 || w->inter != 6144 ||
      w->q_heads != 16 || w->kv_heads != 8 || w->head_dim != 128 || !w->embed || !w->final_norm)
    throw PfError(KT2N_PF_EINVAL, "weights geometry is not the supported model");
  {
    const int64_t rows = w->embed_rows > 0 ? w->embed_rows : w->vocab;
    if (w->embed_row0 < 0 || w->embed_row0 > KT2N_PF_AUDIO_LOW ||
        w->embed_row0 + rows < (int64_t)KT2N_PF_AUDIO_LOW + KT2N_PF_AUDIO_N || w->embed_row0 + rows > w->vocab)
      throw PfError(KT2N_PF_EINVAL, "the device embedding rows must cover the audio slice (the head)");
  }
  check_limits(max_rows, max_tokens, max_keys);
  const Layout L = layout(max_rows, max_tokens, max_keys, (flags & KT2N_PF_CTX_FIT_ATTN_WS) ? split_tiles_for(device) : 0);
  if (workspace && (workspace_bytes < L.total || reinterpret_cast<uintptr_t>(workspace) % kAlign))
    throw PfError(KT2N_PF_EINVAL, "caller workspace too small or not 256-byte aligned");
  ck(cudaSetDevice(device), "cudaSetDevice");
  std::unique_ptr<kt2n_pf_ctx, void (*)(kt2n_pf_ctx*)> c(new kt2n_pf_ctx, ctx_free);
  c->w = *w;
  c->device = device;
  c->supports_pdl = kt2n::cuda::device_caps(device).pdl();
#if defined(KT2N_HIP)
  c->wmma = kt2n::cuda::device_caps(device).wmma != 0;
#endif
  c->max_rows = max_rows;
  c->max_tokens = max_tokens;
  c->max_keys = max_keys;
  c->in_cap = L.in_cap;
  if (workspace) {
    c->arena = workspace;
    c->owns_arena = false;
  } else {
    ck(cudaMalloc(&c->arena, L.total), "cudaMalloc(prefill workspace)");
  }
  c->bytes = L.total;
  char* b = static_cast<char*>(c->arena);
  c->hidden = reinterpret_cast<__nv_bfloat16*>(b + L.hidden);
  c->normed = reinterpret_cast<__nv_bfloat16*>(b + L.normed);
  c->qkv = reinterpret_cast<__nv_bfloat16*>(b + L.qkv);
  c->qrot = reinterpret_cast<__nv_bfloat16*>(b + L.qrot);
  c->attn = reinterpret_cast<__nv_bfloat16*>(b + L.attn);
  c->act = reinterpret_cast<__nv_bfloat16*>(b + L.act);
  c->cos = reinterpret_cast<__nv_bfloat16*>(b + L.cos);
  c->sin = reinterpret_cast<__nv_bfloat16*>(b + L.sin);
  c->last_h = reinterpret_cast<__nv_bfloat16*>(b + L.lh);
  c->last_a = reinterpret_cast<__nv_bfloat16*>(b + L.la);
  c->last_n = reinterpret_cast<__nv_bfloat16*>(b + L.ln);
  c->d_in = b + L.in;
  c->attn_ws = b + L.ws;
  c->attn_ws_bytes = L.ws_bytes;
  ck(cudaMemsetAsync(c->arena, 0, L.total, stream), "cudaMemset(prefill workspace)");
  ka(kt2n_attn_workspace_init(c->attn_ws, L.ws_bytes, stream), "kt2n_attn_workspace_init");
  ck(cudaHostAlloc(reinterpret_cast<void**>(&c->h_in), c->in_cap, cudaHostAllocDefault), "cudaHostAlloc");
  if (w->host_embed)
    ck(cudaHostAlloc(reinterpret_cast<void**>(&c->h_rows), (size_t)max_tokens * w->hidden * 2, cudaHostAllocDefault),
       "cudaHostAlloc(embedding rows)");
  ck(cudaEventCreateWithFlags(&c->h2d_done, cudaEventDisableTiming), "cudaEventCreate");
  if (const char* f = std::getenv("KT2N_PF_FORCE_FLAGS")) c->force_flags = (unsigned)std::strtoul(f, nullptr, 0);
  // KT2N_PF_FLAG_OVERLAP_PAST: the side stream and its events are made here, not on first use, so that a first
  // call made under stream capture (any capture mode) never creates CUDA objects.
  ck(cudaStreamCreateWithFlags(&c->side, cudaStreamNonBlocking), "cudaStreamCreate(side)");
  ck(cudaEventCreateWithFlags(&c->ev_fork, cudaEventDisableTiming), "cudaEventCreate");
  for (cudaEvent_t& e : c->ev_grp) ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreate");
  if (const char* f = std::getenv("KT2N_PF_Q8_TQ")) c->q8_tq = std::max(1, std::min(8, std::atoi(f)));
  ck(cudaStreamSynchronize(stream), "prefill context init");
  return c.release();
}

void run(kt2n_pf_ctx* c, int R, const int32_t* new_tokens, const int64_t* ids, const int32_t* past_len,
         const void* const* past_kv, const void* speakers, void* kv_dest, int64_t kv_row_stride, int32_t kv_width,
         const int32_t* dest_row, void* last_hidden, void* logits, int64_t ld_logits, unsigned flags,
         cudaStream_t stream) {
  if (!c) throw PfError(KT2N_PF_EINVAL, "ctx is NULL");
  flags |= c->force_flags;
  if (R < 1 || R > c->max_rows) throw PfError(KT2N_PF_EINVAL, "num_rows out of range");
  if (!new_tokens || !ids || !past_len || !kv_dest) throw PfError(KT2N_PF_EINVAL, "NULL argument");
  const kt2n_pf_weights& w = c->w;
  const int L = w.layers, H = w.hidden, I = w.inter, KVH = w.kv_heads, HD = w.head_dim;
  const int QN = w.q_heads * HD, KVN = KVH * HD, QKVN = QN + 2 * KVN;
  const int64_t W = kv_width;
  const int64_t row_elems = 2LL * L * KVH * W * HD;
  if (W < 1 || W > c->max_keys || kv_row_stride < row_elems || kv_row_stride % 8)
    throw PfError(KT2N_PF_EINVAL, "bad kv_width / kv_row_stride");
  if (reinterpret_cast<uintptr_t>(kv_dest) % 16) throw PfError(KT2N_PF_EINVAL, "kv_dest not 16-byte aligned");
  const bool want_out = last_hidden || logits;
  if (logits && (R > KT2N_SAMP_HEAD_MAX_ROWS || ld_logits < KT2N_PF_AUDIO_N || ld_logits % 8))
    throw PfError(KT2N_PF_EINVAL, "logits need R <= 16 and ld_logits >= 6568, % 8 == 0");
  // ---- validate rows, build the packed inputs
  int T = 0, max_new = 0, max_keys = 0;
  std::vector<int> drow(R);
  for (int r = 0; r < R; ++r) {
    if (new_tokens[r] < 1 || past_len[r] < 0) throw PfError(KT2N_PF_EINVAL, "row with no new tokens / negative past");
    const int keys = past_len[r] + new_tokens[r];
    if (keys > W) throw PfError(KT2N_PF_EINVAL, "row does not fit kv_width");
    T += new_tokens[r];
    max_new = std::max(max_new, new_tokens[r]);
    max_keys = std::max(max_keys, keys);
    drow[r] = dest_row ? dest_row[r] : r;
    if (drow[r] < 0) throw PfError(KT2N_PF_EINVAL, "negative dest row");
    for (int q = 0; q < r; ++q)
      if (drow[q] == drow[r]) throw PfError(KT2N_PF_EINVAL, "dest rows must be distinct");
    if (past_kv && past_kv[r] && past_len[r] == 0) throw PfError(KT2N_PF_EINVAL, "past_kv given for past_len 0");
  }
  if (T > c->max_tokens) throw PfError(KT2N_PF_EINVAL, "call exceeds the context's max_tokens");
  ck(cudaSetDevice(c->device), "cudaSetDevice");
  if (c->h2d_pending) {  // the previous call's input copy must have read the pinned buffer
    ck(cudaEventSynchronize(c->h2d_done), "cudaEventSynchronize");
    c->h2d_pending = false;
  }
  char* h = c->h_in;
  size_t off = 0;
  auto take = [&](size_t bytes) {
    size_t o = off;
    off = align_up(off + bytes);
    return o;
  };
  const size_t o_ids = take((size_t)T * 8), o_pos = take((size_t)T * 8), o_slot = take((size_t)T * 8),
               o_spk = take((size_t)T * 4), o_desc = take((size_t)R * sizeof(kt2n_attn_prefill_row)),
               o_last = take((size_t)R * sizeof(kt2n_attn_prefill_row)), o_gat = take((size_t)R * 4),
               o_dslot = take((size_t)T * 4), o_dlen = take((size_t)T * 4),
               o_q8 = take((size_t)T * sizeof(kt2n_pf_q8_tile));
  auto* hi = reinterpret_cast<int64_t*>(h + o_ids);
  auto* hp = reinterpret_cast<int64_t*>(h + o_pos);
  auto* hs = reinterpret_cast<int64_t*>(h + o_slot);
  auto* hk = reinterpret_cast<int32_t*>(h + o_spk);
  auto* hd = reinterpret_cast<kt2n_attn_prefill_row*>(h + o_desc);
  auto* hl = reinterpret_cast<kt2n_attn_prefill_row*>(h + o_last);
  auto* hg = reinterpret_cast<int32_t*>(h + o_gat);
  auto* hds = reinterpret_cast<int32_t*>(h + o_dslot);
  auto* hdl = reinterpret_cast<int32_t*>(h + o_dlen);
  auto* hq8 = reinterpret_cast<kt2n_pf_q8_tile*>(h + o_q8);
  int q8_tiles = 0;
  // q8 tile size (KT2N_PF_FLAG_ATTN_Q8): 4 queries per tile up to 64 new tokens (more CTAs for short calls), 8 above
  // (A/B on the RTX PRO 6000: 4 -> 1.12x at 27 tokens, 1.06x at 53; 8 -> 1.03x at 103, where 4 is 0.98x).
  const int tq = c->q8_tq > 0 ? c->q8_tq : (T <= 64 ? 4 : 8);
  bool any_speaker = false;
  // weights.host_embed: token t's embedding row is copied to h_rows[t] here, and the embed kernel reads row t of
  // the uploaded copy (hi[t] = t; -1 for an id outside the vocabulary, a zero row like the device table's lookup).
  const char* hemb = static_cast<const char*>(w.host_embed);
  const size_t row_bytes = (size_t)H * 2;
  int t = 0;
  for (int r = 0; r < R; ++r) {
    const int s = new_tokens[r], p = past_len[r];
    std::memset(&hd[r], 0, sizeof(hd[r]));
    std::memset(&hl[r], 0, sizeof(hl[r]));
    hd[r].q_start = t;
    hd[r].q_len = s;
    hd[r].q_pos0 = p;
    hd[r].kv_slot = drow[r];
    hd[r].kv_col0 = 0;
    hl[r] = hd[r];
    hl[r].q_start = t + s - 1;
    hl[r].q_len = 1;
    hl[r].q_pos0 = p + s - 1;
    hg[r] = t + s - 1;
    for (int q0 = 0; q0 < s; q0 += tq)
      hq8[q8_tiles++] = kt2n_pf_q8_tile{t + q0, std::min(tq, s - q0), p + q0, drow[r]};
    for (int j = 0; j < s; ++j, ++t) {
      const bool inject = speakers && p == 0 && j == 0;
      if (hemb) {
        const bool in_vocab = ids[t] >= 0 && ids[t] < w.vocab;
        hi[t] = in_vocab ? t : -1;
        if (in_vocab && !inject)
          std::memcpy(c->h_rows + (size_t)t * row_bytes, hemb + (size_t)ids[t] * row_bytes, row_bytes);
      } else {
        hi[t] = ids[t];
      }
      hp[t] = p + j;
      hs[t] = drow[r];
      hk[t] = inject ? r : -1;
      hds[t] = drow[r];
      hdl[t] = p + j + 1;  // keys the token at logical position p + j attends
      any_speaker = any_speaker || inject;
    }
  }
  ck(cudaMemcpyAsync(c->d_in, h, off, cudaMemcpyHostToDevice, stream), "copy prefill inputs");
  if (hemb)  // the gathered rows go to `attn`, which nothing reads or writes before layer 0's attention
    ck(cudaMemcpyAsync(c->attn, c->h_rows, (size_t)T * row_bytes, cudaMemcpyHostToDevice, stream),
       "copy prefill embedding rows");
  ck(cudaEventRecord(c->h2d_done, stream), "cudaEventRecord");
  c->h2d_pending = true;
  const int64_t* d_ids = reinterpret_cast<const int64_t*>(c->d_in + o_ids);
  const int64_t* d_pos = reinterpret_cast<const int64_t*>(c->d_in + o_pos);
  const int64_t* d_slot = reinterpret_cast<const int64_t*>(c->d_in + o_slot);
  const int32_t* d_spk = reinterpret_cast<const int32_t*>(c->d_in + o_spk);
  const auto* d_desc = reinterpret_cast<const kt2n_attn_prefill_row*>(c->d_in + o_desc);
  const auto* d_last = reinterpret_cast<const kt2n_attn_prefill_row*>(c->d_in + o_last);
  const int32_t* d_gat = reinterpret_cast<const int32_t*>(c->d_in + o_gat);
  const int32_t* d_dslot = reinterpret_cast<const int32_t*>(c->d_in + o_dslot);
  const int32_t* d_dlen = reinterpret_cast<const int32_t*>(c->d_in + o_dlen);
  const auto* d_q8 = reinterpret_cast<const kt2n_pf_q8_tile*>(c->d_in + o_q8);

  auto* dest = static_cast<__nv_bfloat16*>(kv_dest);
  // ---- past K/V into the destination rows (columns [0, p))
  int past_rows = 0;
  if (past_kv)
    for (int r = 0; r < R; ++r) past_rows += past_kv[r] != nullptr;
  const bool any_past = past_rows > 0;
  // Overlapped variant: the copies go to a side stream in layer groups (K and V of the group's layers, one 3-D
  // copy per row and group); layer l's attention (the first reader of columns [0, p) of layer l) waits for its
  // group only.  The new columns [p, p + s) are written by the main stream: disjoint bytes.  Same bits.
  // Only for bursts (>= 2 rows with a past): for one row the copy is ~1-2% of the call and the extra stream /
  // event plumbing measured no gain in A/B tests.
  const bool overlap = past_rows >= 2 && (flags & KT2N_PF_FLAG_OVERLAP_PAST);
  int grp_first[5] = {0, 1, 3, 7, L};  // groups [0,1) [1,3) [3,7) [7,L) for 28 layers; clamped below
  int ngrp = 0;
  if (overlap) {
    for (int g = 1; g < 5; ++g) grp_first[g] = std::min(grp_first[g], L);
    ngrp = 4;
    ck(cudaEventRecord(c->ev_fork, stream), "cudaEventRecord(fork)");
    ck(cudaStreamWaitEvent(c->side, c->ev_fork, 0), "cudaStreamWaitEvent(side)");
    for (int g = 0; g < ngrp; ++g) {
      const int l0 = grp_first[g], l1 = grp_first[g + 1];
      if (l1 > l0) {
        for (int r = 0; r < R; ++r) {
          if (!past_kv[r]) continue;
          const size_t p = (size_t)past_len[r];
          cudaMemcpy3DParms cp{};
          cp.srcPtr = make_cudaPitchedPtr(const_cast<void*>(past_kv[r]), p * HD * 2, p * HD * 2, (size_t)L * KVH);
          cp.dstPtr = make_cudaPitchedPtr(dest + (int64_t)drow[r] * kv_row_stride, (size_t)W * HD * 2,
                                          (size_t)W * HD * 2, (size_t)L * KVH);
          cp.srcPos = make_cudaPos(0, (size_t)l0 * KVH, 0);
          cp.dstPos = make_cudaPos(0, (size_t)l0 * KVH, 0);
          cp.extent = make_cudaExtent(p * HD * 2, (size_t)(l1 - l0) * KVH, 2);
          cp.kind = cudaMemcpyDeviceToDevice;
          ck(cudaMemcpy3DAsync(&cp, c->side), "copy past K/V (side)");
        }
      }
      ck(cudaEventRecord(c->ev_grp[g], c->side), "cudaEventRecord(group)");
    }
  } else if (any_past) {
    for (int r = 0; r < R; ++r) {
      if (!past_kv[r]) continue;
      const size_t p = (size_t)past_len[r];
      ck(cudaMemcpy2DAsync(dest + (int64_t)drow[r] * kv_row_stride, (size_t)W * HD * 2, past_kv[r], p * HD * 2,
                           p * HD * 2, (size_t)2 * L * KVH, cudaMemcpyDeviceToDevice, stream),
         "copy past K/V");
    }
  }
  int grp_waited = 0;  // groups the main stream already waits for
  auto wait_groups_through = [&](int layer) {
    while (grp_waited < ngrp && grp_first[grp_waited] <= layer) {
      ck(cudaStreamWaitEvent(stream, c->ev_grp[grp_waited], 0), "cudaStreamWaitEvent(group)");
      ++grp_waited;
    }
  };
  const bool pdl = c->supports_pdl && !(flags & KT2N_PF_FLAG_NO_PDL);
  const unsigned tflags = (pdl ? KT2N_TERN_FLAG_PDL : 0u) | KT2N_TERN_FLAG_PREFILL;
  const unsigned aflags = (pdl ? KT2N_ATTN_FLAG_PDL : 0u) |
                          ((flags & KT2N_PF_FLAG_ATTN_SEQUENTIAL) ? KT2N_ATTN_FLAG_PREFILL_SEQUENTIAL : 0u);
  const int opdl = pdl ? 1 : 0;
  const bool ptable = (flags & KT2N_PF_FLAG_PROJ_TABLE) != 0;
  // One projection: the projection library's kt2n_tern_proj, or (KT2N_PF_FLAG_PROJ_TABLE, where the table has an entry for this family
  // and M) the lean GEMM or the projection library's kernels with another launch shape -- the same bits either way.
  auto proj = [&](int fam, const void* x, int64_t ldx, int m, const void* codes, const void* scales, int n, int k,
                  void* y, int64_t ldy, int epi, const void* res, int64_t ldr) -> int {
#if defined(KT2N_HIP)
    // AMD RDNA3 / RDNA4: every prefill projection (any M, so a row's bits never depend on the call's size) on the
    // matrix unit; EINVAL (shape it does not take) keeps the software path below.
    if (c->wmma) {
      const int wrc = kt2n_wmma_tern_proj(x, ldx, m, codes, scales, n, k, y, ldy, epi, res, ldr, 0, stream);
      if (wrc != KT2N_WMMA_EINVAL) return wrc == KT2N_WMMA_OK ? KT2N_TERN_OK : KT2N_TERN_ECUDA;
    }
#endif
    int lc = 0, snt = 0, swn = 0, smt = 0;
    if (ptable && kt2n_pf_proj_choice(fam, m, &lc, &snt, &swn, &smt)) {
      if (lc) {
        kt2n_pf_tern_args a{};
        a.family = fam;
        a.epi = epi;
        a.x = x;
        a.ldx = ldx;
        a.m = m;
        a.codes = codes;
        a.scales = scales;
        a.n = n;
        a.y = y;
        a.ldy = ldy;
        a.residual = res;
        a.ldr = ldr;
        a.pdl = pdl ? 1 : 0;
        a.stream = stream;
        return kt2n_pf_tern_proj(&a, lc);
      }
      return kt2n_tern_proj_ex(fam, x, ldx, m, codes, scales, n, k, y, ldy, epi, res, ldr, tflags, snt, swn, smt,
                               stream);
    }
    return kt2n_tern_proj(fam, x, ldx, m, codes, scales, n, k, y, ldy, epi, res, ldr, tflags, stream);
  };
  // Attention of layers 0..L-2 for short calls (same per-element arithmetic as the attention library's prefill kernel -- attention: prefill ==
  // decode at every position -- so the same bits): decode rows (one decode row per new token, <=
  // KT2N_PF_DECODE_ROWS_MAX_T tokens), else q8 tiles (<= KT2N_PF_Q8_MAX_T tokens); longer calls keep the attention library's prefill
  // kernel (measured faster there).
  const bool dec_attn = (flags & KT2N_PF_FLAG_ATTN_DECODE_ROWS) && T <= KT2N_PF_DECODE_ROWS_MAX_T &&
                        max_keys <= std::min(c->max_keys, 2048) &&
                        c->attn_ws_bytes >= kt2n_attn_decode_workspace_bytes(T, max_keys);
  const bool q8_attn = !dec_attn && (flags & KT2N_PF_FLAG_ATTN_Q8) && T <= KT2N_PF_Q8_MAX_T;
  const int q8_mode = (flags & KT2N_PF_FLAG_Q8_FORCE_PAR) ? 1 : (flags & KT2N_PF_FLAG_Q8_FORCE_SEQ) ? 2 : 0;
#if defined(KT2N_HIP)
  // AMD with KT2N_AMD_WMMA_DECODE=2: every token's attention (all layers, the last one included) through the attention
  // library's decode-rows kernel, whose matrix-unit form is the decode step's attention arithmetic in that mode, so a
  // prefill gives the decode step's bits at every position, except in the step's SPLIT64 buckets (<= 2 rows: another
  // softmax fold order for rows of <= 640 keys, as on NVIDIA). As many rows per launch as the workspace takes.
  int amd_rows_per_call = 0;
  if (c->wmma && kt2n_wmma_decode_mode() == 2) {
    for (int r = std::min(T, 64); r >= 1; r = r > 8 ? r - 8 : r - 1)
      if (c->attn_ws_bytes >= kt2n_attn_decode_workspace_bytes(r, max_keys)) {
        amd_rows_per_call = r;
        break;
      }
  }
#endif
  const float eps = w.rms_eps;
  auto bf = [](const void* p) { return static_cast<const __nv_bfloat16*>(p); };

  // ---- embedding (+ speaker at logical position 0) + layer 0 input norm + RoPE tables.  First kernel after
  // the copies: an ordinary launch.
  // The table: the uploaded rows of this call (host_embed: id t = row t), else the device embedding.
  const __nv_bfloat16* table = hemb ? c->attn : bf(w.embed);
  const int64_t table_row0 = hemb ? 0 : w.embed_row0;
  const int64_t table_rows = hemb ? T : (w.embed_rows > 0 ? w.embed_rows : w.vocab);
  ko(kt2n_ops_embed_rmsnorm_rows(d_ids, T, table, table_row0, table_rows, H, any_speaker ? bf(speakers) : nullptr, H,
                                 any_speaker ? d_spk : nullptr, -1, bf(w.layer[0].input_norm), eps, c->hidden, H,
                                 c->normed, H, d_pos, w.inv_freq, c->cos, c->sin, 64, 0, stream),
     "kt2n_ops_embed_rmsnorm");

  const int64_t layer_kv = (int64_t)KVH * W * HD;
  for (int l = 0; l < L; ++l) {
    const kt2n_pf_layer& lw = w.layer[l];
    const bool last = (l == L - 1);
    __nv_bfloat16* kc = dest + (int64_t)l * layer_kv;
    __nv_bfloat16* vc = dest + (int64_t)(L + l) * layer_kv;
    kt(proj(KT2N_TERN_FAM_QKV, c->normed, H, T, lw.qkv_codes, lw.qkv_scales, QKVN, H, c->qkv, QKVN,
            KT2N_TERN_EPI_STORE, nullptr, 0),
       "qkv projection");
    kt2n_ops_kv_target kvt;
    std::memset(&kvt, 0, sizeof(kvt));
    kvt.k = kc;
    kvt.v = vc;
    kvt.slot_stride = kv_row_stride;
    kvt.head_stride = W * HD;
    kvt.col_stride = HD;
    kvt.row_slot = d_slot;
    kvt.row_col = d_pos;
    kvt.num_cols = W;
    ko(kt2n_ops_qk_norm_rope(c->qkv, QKVN, c->qkv + QN, QKVN, c->qkv + QN + KVN, QKVN, bf(lw.q_norm),
                             bf(lw.k_norm), eps, d_pos, w.inv_freq, c->cos, c->sin, 64, c->qrot, QN, HD, nullptr,
                             0, 0, &kvt, T, w.q_heads, KVH, HD, opdl, stream),
       "kt2n_ops_qk_norm_rope");
    if (last && !want_out) break;  // K/V of every layer written; nothing reads the last layer's output
    if (overlap) wait_groups_through(l);  // this layer's past columns are read by its attention
    __nv_bfloat16 *x_res = c->hidden, *a_in = c->attn, *nrm = c->normed;
    int M = T;
#if defined(KT2N_HIP)
    if (amd_rows_per_call > 0) {
      for (int r0 = 0; r0 < T; r0 += amd_rows_per_call) {
        const int n = std::min(amd_rows_per_call, T - r0);
        ka(kt2n_attn_decode_rows(c->qrot + (int64_t)r0 * QN, QN, kc, vc, kv_row_stride, W * HD, d_dslot + r0, nullptr,
                                 d_dlen + r0, n, max_keys, c->attn + (int64_t)r0 * QN, QN, c->attn_ws, c->attn_ws_bytes,
                                 pdl ? KT2N_ATTN_FLAG_PDL : 0u, stream),
           "kt2n_attn_decode_rows (AMD)");
      }
      if (last) {  // as below: only each row's last token goes on (its attention row is at its token index)
        gather_two_kernel<<<R, 128, 0, stream>>>(c->hidden, c->attn, H, d_gat, c->last_h, c->last_a, H, H);
        ck(cudaGetLastError(), "gather");
        x_res = c->last_h;
        a_in = c->last_a;
        nrm = c->last_n;
        M = R;
      }
    } else
#endif
    if (!last && q8_attn) {
      ka(kt2n_pf_q8_attention(c->qrot, QN, kc, vc, kv_row_stride, W * HD, d_q8, q8_tiles, max_keys, c->attn, QN,
                              c->attn_ws, c->attn_ws_bytes, pdl ? 1 : 0, q8_mode, stream),
         "kt2n_pf_q8_attention");
    } else if (!last && dec_attn) {
      ka(kt2n_attn_decode_rows(c->qrot, QN, kc, vc, kv_row_stride, W * HD, d_dslot, nullptr, d_dlen, T, max_keys,
                               c->attn, QN, c->attn_ws, c->attn_ws_bytes, pdl ? KT2N_ATTN_FLAG_PDL : 0u, stream),
         "kt2n_attn_decode_rows");
    } else if (!last) {
      ka(kt2n_attn_prefill(c->qrot, QN, kc, vc, kv_row_stride, W * HD, d_desc, R, max_new, max_keys, c->attn, QN,
                           c->attn_ws, c->attn_ws_bytes, aflags, stream),
         "kt2n_attn_prefill");
    } else {
      // last layer: only each row's last token feeds the head -> attention, o_proj and the MLP on R rows.
      ka(kt2n_attn_prefill(c->qrot, QN, kc, vc, kv_row_stride, W * HD, d_last, R, 1, max_keys, c->attn, QN,
                           c->attn_ws, c->attn_ws_bytes, aflags, stream),
         "kt2n_attn_prefill(last)");
      gather_two_kernel<<<R, 128, 0, stream>>>(c->hidden, c->attn, H, d_gat, c->last_h, c->last_a, H, H);
      ck(cudaGetLastError(), "gather");
      x_res = c->last_h;
      a_in = c->last_a;
      nrm = c->last_n;
      M = R;
    }
    kt(proj(KT2N_TERN_FAM_O, a_in, QN, M, lw.o_codes, lw.o_scales, H, QN, x_res, H, KT2N_TERN_EPI_RESIDUAL, x_res,
            H),
       "o projection");
    ko(kt2n_ops_rmsnorm(x_res, H, bf(lw.post_norm), eps, nrm, H, M, H, opdl, stream), "post-attention rmsnorm");
    kt(proj(KT2N_TERN_FAM_GATE_UP, nrm, H, M, lw.gu_codes, lw.gu_scales, 2 * I, H, c->act, I,
            KT2N_TERN_EPI_SILU_MUL, nullptr, 0),
       "gate_up projection");
    kt(proj(KT2N_TERN_FAM_DOWN, c->act, I, M, lw.down_codes, lw.down_scales, H, I, x_res, H,
            KT2N_TERN_EPI_RESIDUAL, x_res, H),
       "down projection");
    if (!last) {
      ko(kt2n_ops_rmsnorm(c->hidden, H, bf(w.layer[l + 1].input_norm), eps, c->normed, H, T, H, opdl, stream),
         "input rmsnorm");
    } else {
      ko(kt2n_ops_rmsnorm(c->last_h, H, bf(w.final_norm), eps, c->last_n, H, R, H, opdl, stream), "final rmsnorm");
    }
  }
  if (overlap) wait_groups_through(L);  // join every side-stream copy before the call's last operation
  if (want_out) {
    if (last_hidden)
      ck(cudaMemcpyAsync(last_hidden, c->last_n, (size_t)R * H * 2, cudaMemcpyDeviceToDevice, stream),
         "copy last hidden");
    // the head: the audio slice of the (tied) embedding, wherever the device table starts
    const __nv_bfloat16* head_w = bf(w.embed) + (int64_t)(KT2N_PF_AUDIO_LOW - w.embed_row0) * H;
    if (logits)
      ks(kt2n_samp_head_prefill_bf16(c->last_n, H, head_w, H, logits, ld_logits, R, KT2N_PF_AUDIO_N, H, stream),
         "kt2n_samp_head_prefill_bf16");
  }
  ck(cudaGetLastError(), "prefill launch");
}

}  // namespace
}  // namespace kt2n_pf

extern "C" int kt2n_pf_ctx_create(const kt2n_pf_weights* weights, int device, int max_rows, int max_tokens,
                                  int max_keys, kt2n_pf_ctx** out) {
  return kt2n_pf::guarded([&] {
    if (!out) throw kt2n_pf::PfError(KT2N_PF_EINVAL, "out is NULL");
    *out = nullptr;
    *out = kt2n_pf::ctx_create(weights, device, max_rows, max_tokens, max_keys, nullptr, 0, nullptr);
  });
}

extern "C" int kt2n_pf_ctx_create2(const kt2n_pf_weights* weights, int device, int max_rows, int max_tokens,
                                   int max_keys, unsigned flags, kt2n_pf_ctx** out) {
  return kt2n_pf::guarded([&] {
    if (!out) throw kt2n_pf::PfError(KT2N_PF_EINVAL, "out is NULL");
    if (flags & ~KT2N_PF_CTX_FIT_ATTN_WS) throw kt2n_pf::PfError(KT2N_PF_EINVAL, "unknown context flags");
    *out = nullptr;
    *out = kt2n_pf::ctx_create(weights, device, max_rows, max_tokens, max_keys, nullptr, 0, nullptr, flags);
  });
}

extern "C" size_t kt2n_pf_ctx_workspace_bytes2(int max_rows, int max_tokens, int max_keys, unsigned flags, int device) {
  try {
    kt2n_pf::check_limits(max_rows, max_tokens, max_keys);
    return kt2n_pf::layout(max_rows, max_tokens, max_keys,
                           (flags & KT2N_PF_CTX_FIT_ATTN_WS) ? kt2n_pf::split_tiles_for(device) : 0).total;
  } catch (...) {
    return 0;
  }
}

extern "C" size_t kt2n_pf_ctx_workspace_bytes(int max_rows, int max_tokens, int max_keys) {
  try {
    kt2n_pf::check_limits(max_rows, max_tokens, max_keys);
    return kt2n_pf::layout(max_rows, max_tokens, max_keys).total;
  } catch (...) {
    return 0;
  }
}

extern "C" int kt2n_pf_ctx_create_in(const kt2n_pf_weights* weights, int device, int max_rows, int max_tokens,
                                     int max_keys, void* workspace, size_t workspace_bytes, cudaStream_t stream,
                                     kt2n_pf_ctx** out) {
  return kt2n_pf::guarded([&] {
    if (!out) throw kt2n_pf::PfError(KT2N_PF_EINVAL, "out is NULL");
    if (!workspace) throw kt2n_pf::PfError(KT2N_PF_EINVAL, "workspace is NULL");
    *out = nullptr;
    *out = kt2n_pf::ctx_create(weights, device, max_rows, max_tokens, max_keys, workspace, workspace_bytes, stream);
  });
}

extern "C" size_t kt2n_pf_ctx_bytes(const kt2n_pf_ctx* ctx) { return ctx ? ctx->bytes : 0; }

extern "C" void kt2n_pf_ctx_free(kt2n_pf_ctx* ctx) { kt2n_pf::ctx_free(ctx); }

extern "C" int kt2n_pf_run(kt2n_pf_ctx* ctx, int num_rows, const int32_t* host_new_tokens, const int64_t* host_ids,
                           const int32_t* host_past_len, const void* const* host_past_kv, const void* speakers,
                           void* kv_dest, int64_t kv_row_stride, int32_t kv_width, const int32_t* host_dest_row,
                           void* last_hidden, void* logits, int64_t ld_logits, unsigned flags, cudaStream_t stream) {
  return kt2n_pf::guarded([&] {
    kt2n_pf::run(ctx, num_rows, host_new_tokens, host_ids, host_past_len, host_past_kv, speakers, kv_dest,
                 kv_row_stride, kv_width, host_dest_row, last_hidden, logits, ld_logits, flags, stream);
  });
}

extern "C" const char* kt2n_pf_last_error(void) { return kt2n_pf::g_last_error.c_str(); }

extern "C" int kt2n_pf_api_version(void) { return KT2N_PF_API_VERSION; }
