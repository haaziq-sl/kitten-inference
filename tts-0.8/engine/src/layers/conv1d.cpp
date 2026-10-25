// conv1d.cpp — Conv1dFp32 and ConvTranspose1dFp32 layer implementations
//
// Layout convention: 1-D tensors use shape {C, T, 1}.
// Buffer memory: [T, C] (NHWC with H=T, W=1).
//
// Conv1dFp32 reuses conv2d_fp32_nhwc with kH=kernel_size, kW=1.
// ConvTranspose1dFp32 uses GEMM-per-kernel-position + scatter-add (groups==1)
// or scalar scatter-add fallback (groups > 1).

#include "conv1d.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/cpu/ops/profile_internal.hpp"
#include "backend/backend.hpp"
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

// ─────────────────────────────────────────────────────────────────
// helpers — use shared read_f32_tensor() from layer.hpp
// ─────────────────────────────────────────────────────────────────
static inline bool read_f32(const ReadTensorFn& r, const std::string& k,
                             std::vector<float>& out)
{ return read_f32_tensor(r, k, out); }

template <typename T>
static void release_vector(std::vector<T>& v)
{
    std::vector<T>().swap(v);
}

static float* conv1d_float_scratch(BufferView view,
                                   std::vector<float>& fallback,
                                   size_t elems)
{
    const size_t bytes = elems * sizeof(float);
    if (view.valid() && view.bytes >= bytes) return view.as<float>();
    if (fallback.size() < elems) fallback.resize(elems);
    return fallback.data();
}

static bool conv1d_direct_by_default()
{
#if defined(__AVX2__) || defined(__AVX512F__)
    return true;
#else
    return false;
#endif
}

static bool conv1d_use_direct_kernel(int kernel_size)
{
    const char* v = std::getenv("KITTEN_FP32_CONV1D_DIRECT");
    if (!v || !v[0]) return conv1d_direct_by_default();
    if (v[0] == '0' || !std::strcmp(v, "wino")) return false;
    if (!std::strcmp(v, "1") || !std::strcmp(v, "all")) return true;

    const char* p = v;
    while (*p) {
        while (*p == ',' || *p == ':' || *p == ';' || *p == ' ') ++p;
        char* end = nullptr;
        const long k = std::strtol(p, &end, 10);
        if (end == p) break;
        if (k == kernel_size) return true;
        p = end;
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────
// Conv1dFp32
// ─────────────────────────────────────────────────────────────────

void Conv1dFp32::load_weights(const ReadTensorFn& read)
{
    std::vector<float> raw_weight;
    read_f32(read, name + ".weight", raw_weight);
    read_f32(read, name + ".bias",   bias_fp32);

    const int C_in_g = C_in / std::max(groups, 1);
    const int K      = kernel_size * C_in_g;

    // Reorder NCHW [C_out, C_in/g, kH] → NHWC [C_out, kH, C_in/g]
    weight_fp32.resize((size_t)C_out * K);
    for (int co = 0; co < C_out; ++co)
        for (int kh = 0; kh < kernel_size; ++kh)
            for (int ci = 0; ci < C_in_g; ++ci)
                weight_fp32[(size_t)co * K + kh * C_in_g + ci] =
                    raw_weight[(size_t)co * C_in_g * kernel_size + ci * kernel_size + kh];

    if (groups == 1 && stride == 1) {
        std::vector<float> w_k((size_t)C_out * C_in_g);
        const size_t nb_per_pack = packed_f32_elems(C_out, C_in_g) * sizeof(float);

        const bool use_direct = conv1d_use_direct_kernel(kernel_size);

        if (kernel_size == 3 && dilation == 1 && !use_direct) {
            // Winograd F(4,3) filter transform G (6×3)
            static const float G[6][3] = {
                { 1.f/4,    0.f,      0.f   },
                {-1.f/6,  -1.f/6,  -1.f/6  },
                {-1.f/6,   1.f/6,  -1.f/6  },
                { 1.f/24,  1.f/12,  1.f/6  },
                { 1.f/24, -1.f/12,  1.f/6  },
                {   0.f,    0.f,     1.f   },
            };
            w_wino_packed_.resize(6);
            for (int p = 0; p < 6; ++p) {
                for (int co = 0; co < C_out; ++co) {
                    for (int ci = 0; ci < C_in_g; ++ci) {
                        float g0 = weight_fp32[(size_t)co*K + 0*C_in_g + ci];
                        float g1 = weight_fp32[(size_t)co*K + 1*C_in_g + ci];
                        float g2 = weight_fp32[(size_t)co*K + 2*C_in_g + ci];
                        w_k[co * C_in_g + ci] = G[p][0]*g0 + G[p][1]*g1 + G[p][2]*g2;
                    }
                }
                float* pk = pack_weights_f32(w_k.data(), C_out, C_in_g);
                w_wino_packed_[p] = Buffer(pk, nb_per_pack, MemSpace::Host,
                    [](void* p){ delete[] static_cast<float*>(p); });
            }
        } else if (kernel_size == 7 && dilation == 1 && !use_direct) {
            // Winograd F(2,7) filter transform G (8×7)
            static const float G[8][7] = {
                {-1.f/36,      0.f,      0.f,      0.f,      0.f,       0.f,     0.f},
                { 1.f/48,  1.f/48,  1.f/48,  1.f/48,  1.f/48,  1.f/48,  1.f/48},
                { 1.f/48, -1.f/48,  1.f/48, -1.f/48,  1.f/48, -1.f/48,  1.f/48},
                {-1.f/120,-1.f/60, -1.f/30, -1.f/15, -2.f/15, -4.f/15, -8.f/15},
                {-1.f/120, 1.f/60, -1.f/30,  1.f/15, -2.f/15,  4.f/15, -8.f/15},
                { 1.f/720, 1.f/240, 1.f/80,  3.f/80,  9.f/80, 27.f/80, 81.f/80},
                { 1.f/720,-1.f/240, 1.f/80, -3.f/80,  9.f/80,-27.f/80, 81.f/80},
                {      0.f,     0.f,     0.f,     0.f,     0.f,      0.f,    1.f},
            };
            w_wino_packed_.resize(8);
            for (int p = 0; p < 8; ++p) {
                for (int co = 0; co < C_out; ++co) {
                    for (int ci = 0; ci < C_in_g; ++ci) {
                        float acc = 0.f;
                        for (int k = 0; k < 7; ++k)
                            acc += G[p][k] * weight_fp32[(size_t)co*K + k*C_in_g + ci];
                        w_k[co * C_in_g + ci] = acc;
                    }
                }
                float* pk = pack_weights_f32(w_k.data(), C_out, C_in_g);
                w_wino_packed_[p] = Buffer(pk, nb_per_pack, MemSpace::Host,
                    [](void* p){ delete[] static_cast<float*>(p); });
            }
        } else if (kernel_size == 11 && dilation == 1 && !use_direct) {
            // Winograd F(2,11) filter transform G (12×11)
            static const float G[12][11] = {
                {-1.f/14400,      0.f,      0.f,      0.f,      0.f,       0.f,       0.f,       0.f,       0.f,       0.f,       0.f},
                { 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280},
                { 1.f/17280,-1.f/17280, 1.f/17280,-1.f/17280, 1.f/17280,-1.f/17280, 1.f/17280,-1.f/17280, 1.f/17280,-1.f/17280, 1.f/17280},
                {-1.f/30240,-2.f/30240,-4.f/30240,-8.f/30240,-16.f/30240,-32.f/30240,-64.f/30240,-128.f/30240,-256.f/30240,-512.f/30240,-1024.f/30240},
                {-1.f/30240, 2.f/30240,-4.f/30240, 8.f/30240,-16.f/30240, 32.f/30240,-64.f/30240, 128.f/30240,-256.f/30240, 512.f/30240,-1024.f/30240},
                { 1.f/80640, 3.f/80640, 9.f/80640,27.f/80640,81.f/80640,243.f/80640,729.f/80640,2187.f/80640,6561.f/80640,19683.f/80640,59049.f/80640},
                { 1.f/80640,-3.f/80640, 9.f/80640,-27.f/80640,81.f/80640,-243.f/80640,729.f/80640,-2187.f/80640,6561.f/80640,-19683.f/80640,59049.f/80640},
                {-1.f/362880,-4.f/362880,-16.f/362880,-64.f/362880,-256.f/362880,-1024.f/362880,-4096.f/362880,-16384.f/362880,-65536.f/362880,-262144.f/362880,-1048576.f/362880},
                {-1.f/362880, 4.f/362880,-16.f/362880,64.f/362880,-256.f/362880,1024.f/362880,-4096.f/362880,16384.f/362880,-65536.f/362880,262144.f/362880,-1048576.f/362880},
                { 1.f/3628800, 5.f/3628800,25.f/3628800,125.f/3628800,625.f/3628800,3125.f/3628800,15625.f/3628800,78125.f/3628800,390625.f/3628800,1953125.f/3628800,9765625.f/3628800},
                { 1.f/3628800,-5.f/3628800,25.f/3628800,-125.f/3628800,625.f/3628800,-3125.f/3628800,15625.f/3628800,-78125.f/3628800,390625.f/3628800,-1953125.f/3628800,9765625.f/3628800},
                {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f},
            };
            w_wino_packed_.resize(12);
            for (int p = 0; p < 12; ++p) {
                for (int co = 0; co < C_out; ++co) {
                    for (int ci = 0; ci < C_in_g; ++ci) {
                        float acc = 0.f;
                        for (int k = 0; k < 11; ++k)
                            acc += G[p][k] * weight_fp32[(size_t)co*K + k*C_in_g + ci];
                        w_k[co * C_in_g + ci] = acc;
                    }
                }
                float* pk = pack_weights_f32(w_k.data(), C_out, C_in_g);
                w_wino_packed_[p] = Buffer(pk, nb_per_pack, MemSpace::Host,
                    [](void* p){ delete[] static_cast<float*>(p); });
            }
        } else if (kernel_size == 1) {
            // Single kernel: use per-k pack (fast path via sgemm_f32).
            w_per_k_packed_.resize(1);
            for (int co = 0; co < C_out; ++co)
                memcpy(w_k.data() + (size_t)co * C_in_g,
                       weight_fp32.data() + (size_t)co * K,
                       C_in_g * sizeof(float));
            float* pk = pack_weights_f32(w_k.data(), C_out, C_in_g);
            w_per_k_packed_[0] = Buffer(pk, nb_per_pack, MemSpace::Host,
                [](void* p){ delete[] static_cast<float*>(p); });
        } else {
            // K-split path: merged pack [Co_t, kernel_size, C_in_g, TILE] for kfused GEMM.
            float* pk = pack_merged_weights_f32(weight_fp32.data(), C_out, C_in_g, kernel_size);
            const size_t nb = packed_f32_elems(C_out, C_in_g) * kernel_size * sizeof(float);
            w_kmerged_packed_ = Buffer(pk, nb, MemSpace::Host,
                [](void* p){ delete[] static_cast<float*>(p); });
        }

        // Keep weight_fp32 alive for CUDA cuDNN path (passed as raw weight pointer)
        if (active_backend()->mem_space == MemSpace::Host) {
            release_vector(weight_fp32);
        }
    } else {
        // Fallback path (stride>1, dilation>1, or groups>1): full-K pack
        // Always pack using CPU format: this layer always runs on CPU (required_mem_space=Host).
        float* packed = pack_weights_f32(weight_fp32.data(), C_out, K);
        size_t nb = (size_t)((C_out+3)/4) * K * 4 * sizeof(float);
        w_packed_f32 = Buffer(packed, nb, MemSpace::Host,
            [](void* p){ delete[] static_cast<float*>(p); });
        if (active_backend()->mem_space == MemSpace::Host && groups == 1) {
            release_vector(weight_fp32);
        }
    }
}

void Conv1dFp32::upload_weights(Allocator& /*alloc*/)
{
    // GPU path uses weight_fp32 (CPU buffer) directly; cuDNN uploads it via get_weight_gpu().
    // No persistent GPU copy needed here.
}

void Conv1dFp32::release_runtime_scratch()
{
    release_vector(pad_buf_);
    release_vector(tmp_buf_);
}

Shape Conv1dFp32::forward(const TensorView* inputs, int n_inputs,
                               BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const bool _profile_conv1d_fp32 = g_ops_profile_enabled;
    const double _t0_conv1d_fp32 = _profile_conv1d_fp32 ? now_ms() : 0.0;
    const Shape out_sh_early = output_shape(sh);

    // ── GPU path: delegate to backend conv2d_fp32_nhwc (cuDNN / CUDA only) ──
    if (in1.where == MemSpace::CUDA || in1.where == MemSpace::HIP) {
        active_backend()->conv2d_fp32_nhwc(
            in1.as<float>(), weight_fp32.data(),
            bias_fp32.empty() ? nullptr : bias_fp32.data(),
            out.as<float>(),
            sh.d0, sh.d1, sh.d2,   // C_in, H=T, W=1
            C_out, kernel_size, 1,
            stride, 1, padding, 0,
            relu,
            nullptr,            // w_packed unused by cuDNN path
            sc.buffers[Scratch::F32_A].as<float>(),
            dilation, 1, sc.stream);
        if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
        return out_sh_early;
    }

    // ── Metal GPU path: im2col + MPS MatMul via conv1d_fp32_nhwc ─────────────
    // groups > 1 not supported by metal_conv1d_fp32; falls through to CPU path.
    if (in1.where == MemSpace::Metal && groups == 1 && active_backend()->conv1d_fp32_nhwc) {
        active_backend()->conv1d_fp32_nhwc(
            in1.as<float>(), weight_fp32.data(),
            bias_fp32.empty() ? nullptr : bias_fp32.data(),
            out.as<float>(),
            sh.d0, sh.d1, C_out,  // C_in, T, C_out
            kernel_size, stride, padding, dilation,
            relu, sc.stream);
        if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
        return out_sh_early;
    }

    // ── CPU path ─────────────────────────────────────────────────────────────
    const float* in_buf  = in1.as<float>();
    float*       out_buf = out.as<float>();
    const Shape out_sh = output_shape(sh);

    // ── Winograd F(4,3) path (groups=1, stride=1, kernel_size=3, dilation=1) ──
    if (w_wino_packed_.size() == 6) {
        const float* ptrs[6];
        for (int p = 0; p < 6; ++p) ptrs[p] = w_wino_packed_[p].as<float>();
        conv1d_wino_f43_fp32(in_buf, ptrs,
                             bias_fp32.empty() ? nullptr : bias_fp32.data(), relu,
                             sh.d1, out_sh.d1, C_in, C_out, padding,
                             out_buf, sc.buffers[Scratch::F32_A].as<float>());
        if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
        return out_sh;
    }

    // ── Winograd F(2,7) path (groups=1, stride=1, kernel_size=7, dilation=1) ──
    if (w_wino_packed_.size() == 8) {
        const float* ptrs[8];
        for (int p = 0; p < 8; ++p) ptrs[p] = w_wino_packed_[p].as<float>();
        conv1d_wino_f27_fp32(in_buf, ptrs,
                             bias_fp32.empty() ? nullptr : bias_fp32.data(), relu,
                             sh.d1, out_sh.d1, C_in, C_out, padding,
                             out_buf, sc.buffers[Scratch::F32_A].as<float>());
        if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
        return out_sh;
    }

    // ── Winograd F(2,11) path (groups=1, stride=1, kernel_size=11, dilation=1) ──
    if (w_wino_packed_.size() == 12) {
        const float* ptrs[12];
        for (int p = 0; p < 12; ++p) ptrs[p] = w_wino_packed_[p].as<float>();
        conv1d_wino_f211_fp32(in_buf, ptrs,
                              bias_fp32.empty() ? nullptr : bias_fp32.data(), relu,
                              sh.d1, out_sh.d1, C_in, C_out, padding,
                              out_buf, sc.buffers[Scratch::F32_A].as<float>());
        if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
        return out_sh;
    }

    // ── K-split path (groups=1, stride=1): kernel_size==1 or kfused GEMM ──
    if (!w_per_k_packed_.empty() || w_kmerged_packed_.valid()) {
        const int T      = sh.d1;
        const int T_out  = out_sh.d1;
        const int C_in_g = C_in;

        // Pad input once (zero-pad left and right by `padding` time steps)
        const float* A_base;
        if (padding > 0) {
            const int T_padded = T + 2 * padding;
            const size_t pad_need = (size_t)T_padded * C_in_g;
            float* pad_buf = conv1d_float_scratch(sc.buffers[Scratch::F32_A], pad_buf_, pad_need);
            memset(pad_buf, 0,
                   (size_t)padding * C_in_g * sizeof(float));
            memcpy(pad_buf + (size_t)padding * C_in_g, in_buf,
                   (size_t)T * C_in_g * sizeof(float));
            memset(pad_buf + (size_t)(T + padding) * C_in_g, 0,
                   (size_t)padding * C_in_g * sizeof(float));
            A_base = pad_buf;
        } else {
            A_base = in_buf;
        }

        if (!w_per_k_packed_.empty()) {
            // kernel_size == 1: single sgemm with bias+relu fused
            sgemm_f32(A_base, w_per_k_packed_[0].as<float>(),
                      bias_fp32.empty() ? nullptr : bias_fp32.data(),
                      out_buf, relu, T_out, C_in_g, C_out);
        } else {
            // kernel_size > 1: fused K-split GEMM (accumulates all kp in one pass)
            conv1d_kfused_sgemm_f32(A_base, w_kmerged_packed_.as<float>(),
                                    bias_fp32.empty() ? nullptr : bias_fp32.data(),
                                    out_buf, relu, T_out, C_in_g, C_out,
                                    kernel_size, dilation);
        }
        if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
        return out_sh;
    }

    // ── Fallback: im2col + single SGEMM (stride>1 or groups>1) ──────────────
    const int T_out  = out_sh.d1;
    const int C_in_g = C_in / std::max(groups, 1);
    const size_t col_elems = (size_t)T_out * kernel_size * C_in_g;
    float* col_buf = conv1d_float_scratch(sc.buffers[Scratch::F32_A], pad_buf_, col_elems);
    conv2d_fp32_nhwc(
        in_buf, weight_fp32.data(),
        bias_fp32.empty() ? nullptr : bias_fp32.data(),
        out_buf,
        sh.d0, sh.d1, sh.d2, C_out,
        kernel_size, 1, stride, 1, padding, 0, relu,
        w_packed_f32.as<float>(), col_buf,
        dilation, 1);

    if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
    return out_sh;
}

// ─────────────────────────────────────────────────────────────────
// Conv1dInt8
// ─────────────────────────────────────────────────────────────────

static int8_t* conv1d_int8_scratch(BufferView view,
                                   std::vector<int8_t>& fallback,
                                   size_t elems)
{
    if (view.valid() && view.bytes >= elems) return view.as<int8_t>();
    if (fallback.size() < elems) fallback.resize(elems);
    return fallback.data();
}

void Conv1dInt8::scratch_needed(Shape in, size_t out[Scratch::N]) const
{
    if (active_backend()->mem_space != MemSpace::Host) return;

    const int T = std::max(in.d1, 1);
    const int T_out = std::max(output_shape(in).d1, 1);
    const int C_in_g = C_in / std::max(groups, 1);
    if (C_in_g <= 0 || C_out <= 0 || kernel_size <= 0) return;

    if (!w_per_k_packed_.empty()) {
        const int T_padded = T + 2 * padding;
        if (padding > 0) {
            out[Scratch::F32_A] = std::max(
                out[Scratch::F32_A],
                (size_t)T_padded * C_in_g * sizeof(float));
        }
        out[Scratch::I8_A] = std::max(
            out[Scratch::I8_A],
            (size_t)T_padded * C_in_g);
        if (kernel_size > 1) {
            out[Scratch::F32_B] = std::max(
                out[Scratch::F32_B],
                (size_t)T_out * C_out * sizeof(float));
        }
        out[Scratch::F32_C] = std::max(
            out[Scratch::F32_C],
            (size_t)T_padded * sizeof(float));
        return;
    }

    const int K = kernel_size * C_in_g;
    const size_t im2col_need = (size_t)T_out * K;
    out[Scratch::F32_A] = std::max(out[Scratch::F32_A],
                                   im2col_need * sizeof(float));
    out[Scratch::I8_A] = std::max(out[Scratch::I8_A], im2col_need);
    out[Scratch::F32_C] = std::max(out[Scratch::F32_C],
                                   (size_t)C_out * sizeof(float));
}

void Conv1dInt8::load_weights(const ReadTensorFn& read)
{
    std::vector<int32_t> shape;
    std::vector<uint8_t> data;
    std::vector<int8_t>  weight_i8;

    if (read(name + ".weight_i8", shape, data)) {
        weight_i8.resize(data.size());
        memcpy(weight_i8.data(), data.data(), data.size());
    }
    if (read(name + ".w_scales", shape, data)) {
        w_scales.resize(data.size() / 4);
        memcpy(w_scales.data(), data.data(), data.size());
    }
    if (read(name + ".bias", shape, data)) {
        bias_fp32.resize(data.size() / 4);
        memcpy(bias_fp32.data(), data.data(), data.size());
    }

    const int C_in_g = C_in / std::max(groups, 1);
    const int K      = kernel_size * C_in_g;
    if (!weight_i8.empty()) {
        if (stride == 1 && groups <= 1) {
            // Fast path: pack one [C_out, C_in_g] sub-matrix per kernel position.
            // weight_i8 layout: [C_out, K] with K = kernel_size * C_in_g (NHWC order).
            w_per_k_packed_.resize(kernel_size);
            w_per_k_row_sums_.resize(kernel_size);
            std::vector<int8_t> w_k((size_t)C_out * C_in_g);
            for (int k = 0; k < kernel_size; ++k) {
                w_per_k_row_sums_[k].assign(C_out, 0);
                for (int co = 0; co < C_out; ++co)
                    memcpy(w_k.data() + (size_t)co * C_in_g,
                           weight_i8.data() + (size_t)co * K + (size_t)k * C_in_g,
                           C_in_g * sizeof(int8_t));
                for (int co = 0; co < C_out; ++co) {
                    int32_t s = 0;
                    for (int ci = 0; ci < C_in_g; ++ci)
                        s += w_k[(size_t)co * C_in_g + ci];
                    w_per_k_row_sums_[k][co] = s;
                }
                int8_t* pk = pack_weights_sdot(w_k.data(), C_out, C_in_g);
                size_t nb = (size_t)((C_out+7)/8) * ((C_in_g+3)/4) * 32;
                w_per_k_packed_[k] = Buffer(pk, nb, MemSpace::Host,
                    [](void* p){ delete[] static_cast<int8_t*>(p); });
            }
        } else {
            // Fallback: pack full [C_out, K] matrix.
            w_full_row_sums_.assign(C_out, 0);
            for (int co = 0; co < C_out; ++co) {
                int32_t s = 0;
                for (int ki = 0; ki < K; ++ki)
                    s += weight_i8[(size_t)co * K + ki];
                w_full_row_sums_[co] = s;
            }
            int8_t* pk = pack_weights_sdot(weight_i8.data(), C_out, K);
            size_t nb = (size_t)((C_out+7)/8) * ((K+3)/4) * 32;
            w_packed_full_ = Buffer(pk, nb, MemSpace::Host,
                [](void* p){ delete[] static_cast<int8_t*>(p); });
        }

        if (active_backend()->mem_space != MemSpace::Host) {
            // GPU path: dense [C_out, K_pad] row-major INT8.
            // K_pad = ceil4(K) to satisfy GPU GEMM leading-dimension requirements.
            const int K_pad = (K + 3) / 4 * 4;
            std::vector<int8_t> w_dense((size_t)C_out * K_pad, 0);
            for (int co = 0; co < C_out; ++co)
                memcpy(w_dense.data() + (size_t)co * K_pad,
                       weight_i8.data() + (size_t)co * K,
                       K * sizeof(int8_t));
            int8_t* pd = new int8_t[(size_t)C_out * K_pad];
            memcpy(pd, w_dense.data(), (size_t)C_out * K_pad);
            w_gpu_packed_ = Buffer(pd, (size_t)C_out * K_pad, MemSpace::Host,
                [](void* p){ delete[] static_cast<int8_t*>(p); });
        }
    }
    eff_zeros.assign(C_out, 0LL);
}

void Conv1dInt8::upload_weights(Allocator& alloc)
{
    if (alloc.mem_space() == MemSpace::Host) return;
    if (!w_gpu_packed_.valid()) return;
    Buffer d = alloc.make_buffer(w_gpu_packed_.bytes());
    alloc.copy_h2d(d.ptr(), w_gpu_packed_.ptr(), w_gpu_packed_.bytes());
    w_gpu_packed_ = std::move(d);
}

void Conv1dInt8::release_runtime_scratch()
{
    release_vector(im2col_buf_);
    release_vector(pad_buf_);
    release_vector(tmp_buf_);
    release_vector(inp_scratch_);
    release_vector(req_scratch_);
    release_vector(row_scale_buf_);
}

Shape Conv1dInt8::forward(const TensorView* inputs, int n_inputs,
                               BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const Shape out_sh = output_shape(sh);

    // ── GPU path ──────────────────────────────────────────────────────────────
    if (in1.is_gpu()) {
        active_backend()->conv1d_int8_nhwc(
            in1.as<float>(), w_gpu_packed_.as<int8_t>(),
            w_scales.data(), bias_fp32.empty() ? nullptr : bias_fp32.data(),
            out.as<float>(),
            sh.d1, sh.d0, C_out,
            kernel_size, stride, padding, dilation,
            sc.stream);
        return out_sh;
    }

    // ── CPU path ──────────────────────────────────────────────────────────────
    const bool _profile_conv1d_int8 = g_ops_profile_enabled;
    const double _t0_conv1d_int8 = _profile_conv1d_int8 ? now_ms() : 0.0;
    const float* in_buf  = in1.as<float>();
    float*       out_buf = out.as<float>();
    const int T      = sh.d1;
    const int T_out  = out_sh.d1;
    const int C_in_g = C_in / std::max(groups, 1);

    // ── Fast K-split path (stride=1, groups=1) ───────────────────────────────
    // Per-row activation quantization: each time-step [1, C_in_g] gets its own
    // scale = max|row| / 127.  This avoids the "outlier inflation" issue with
    // per-tensor quantization, improving accuracy by ~5 dB SNR at negligible cost.
    //
    // GEMM is called once per kernel position k with req_scale = w_scales (no
    // in_scale multiplier).  After GEMM, each row t of the output is multiplied
    // by row_scale[t] to dequantize.
    if (!w_per_k_packed_.empty()) {
        // Pad input (single-threaded, outside OMP region)
        const float* A_base;
        const int T_padded = T + 2 * padding;
        if (padding > 0) {
            const size_t pad_need = (size_t)T_padded * C_in_g;
            float* pad_buf = conv1d_float_scratch(sc.buffers[Scratch::F32_A], pad_buf_, pad_need);
            memset(pad_buf, 0,
                   (size_t)padding * C_in_g * sizeof(float));
            memcpy(pad_buf + (size_t)padding * C_in_g, in_buf,
                   (size_t)T * C_in_g * sizeof(float));
            memset(pad_buf + (size_t)(T + padding) * C_in_g, 0,
                   (size_t)padding * C_in_g * sizeof(float));
            A_base = pad_buf;
        } else {
            A_base = in_buf;
        }

        const size_t inp_need = (size_t)T_padded * C_in_g;
        int8_t* inp_scratch = conv1d_int8_scratch(sc.buffers[Scratch::I8_A], inp_scratch_, inp_need);
        const float* req_scale = w_scales.data();
        float* row_scale = conv1d_float_scratch(sc.buffers[Scratch::F32_C], row_scale_buf_, T_padded);
        float* tmp_buf = nullptr;
        if (kernel_size > 1) {
            const size_t tmp_need = (size_t)T_out * C_out;
            tmp_buf = conv1d_float_scratch(sc.buffers[Scratch::F32_B], tmp_buf_, tmp_need);
        }

        const int nthreads = omp_get_max_threads();

        // ── Single-thread path ────────────────────────────────────────────────
        if (nthreads <= 1) {
            const float* bias_ptr = bias_fp32.empty() ? nullptr : bias_fp32.data();
            for (int t = 0; t < T_padded; ++t)
                quantize_row_fp32_to_int8(A_base + (size_t)t * C_in_g,
                                         inp_scratch + (size_t)t * C_in_g,
                                         row_scale[t], C_in_g);

            for (int k = 0; k < kernel_size; ++k) {
                const int tap_row = k * dilation;
                const int8_t* q_src = inp_scratch + (size_t)tap_row * C_in_g;
                const float* tap_row_scale = row_scale + tap_row;

                // Fused INT8 GEMM + dequantization. Conv1dInt8 always runs on
                // CPU (required_mem_space=Host), so call the CPU helper directly.
                gemm_int8_dequant_accum(q_src, w_per_k_packed_[k].as<int8_t>(),
                                        out_buf,
                                        T_out, C_in_g, C_out,
                                        w_scales.data(), tap_row_scale,
                                        bias_ptr, k != 0,
                                        w_per_k_row_sums_[k].empty() ? nullptr : w_per_k_row_sums_[k].data(),
                                        nullptr);
            }
            if (_profile_conv1d_int8) g_conv1d_int8_ms += now_ms() - _t0_conv1d_int8;
            return out_sh;
        }

        // ── Multi-thread path ─────────────────────────────────────────────────
        {
            const int8_t* k_ptrs_stack[32];
            std::vector<const int8_t*> k_ptrs_heap;
            const int8_t** k_ptrs = k_ptrs_stack;
            if (kernel_size > (int)(sizeof(k_ptrs_stack) / sizeof(k_ptrs_stack[0]))) {
                k_ptrs_heap.resize(kernel_size);
                k_ptrs = k_ptrs_heap.data();
            }
            for (int k = 0; k < kernel_size; ++k)
                k_ptrs[k] = w_per_k_packed_[k].as<int8_t>();
            const float* bias_ptr = bias_fp32.empty() ? nullptr : bias_fp32.data();
            conv1d_int8_ksplit(
                A_base, k_ptrs, eff_zeros.data(), req_scale,
                bias_ptr, out_buf, tmp_buf,
                inp_scratch, row_scale,
                T_out, C_in_g, C_out, kernel_size, dilation, nthreads);
        }

        if (_profile_conv1d_int8) g_conv1d_int8_ms += now_ms() - _t0_conv1d_int8;
        return out_sh;
    }

    // ── Fallback: stride>1 or groups>1 — im2col + single seqgemm_int8 ────────
    const int K = kernel_size * C_in_g;
    const size_t im2col_need = (size_t)T_out * K;
    float* im2col_buf = conv1d_float_scratch(sc.buffers[Scratch::F32_A], im2col_buf_, im2col_need);
    for (int t = 0; t < T_out; ++t) {
        float* row = im2col_buf + (size_t)t * K;
        for (int k = 0; k < kernel_size; ++k) {
            const int t_in = t * stride + k * dilation - padding;
            float* dst = row + (size_t)k * C_in_g;
            if (t_in >= 0 && t_in < T)
                memcpy(dst, in_buf + (size_t)t_in * C_in_g, C_in_g * sizeof(float));
            else
                memset(dst, 0, C_in_g * sizeof(float));
        }
    }
    int8_t* inp_scratch = conv1d_int8_scratch(sc.buffers[Scratch::I8_A], inp_scratch_, im2col_need);
    float* req_scratch = conv1d_float_scratch(sc.buffers[Scratch::F32_C], req_scratch_, C_out);
    // Use CPU seqgemm_int8 directly: Conv1dInt8 always runs on CPU.
    seqgemm_int8_with_rows(im2col_buf, w_packed_full_.as<int8_t>(),
                           w_scales.data(), eff_zeros.data(),
                           bias_fp32.empty() ? nullptr : bias_fp32.data(),
                           inp_scratch, req_scratch, out_buf,
                           T_out, K, C_out, false,
                           w_full_row_sums_.empty() ? nullptr : w_full_row_sums_.data(),
                           nullptr);
    if (_profile_conv1d_int8) g_conv1d_int8_ms += now_ms() - _t0_conv1d_int8;
    return out_sh;
}

// ─────────────────────────────────────────────────────────────────
// ConvTranspose1dFp32
// ─────────────────────────────────────────────────────────────────

void ConvTranspose1dFp32::load_weights(const ReadTensorFn& read)
{
    // PyTorch ConvTranspose1d weight: [C_in, C_out/groups, kernel_size]
    read_f32(read, name + ".weight", weight_fp32);
    read_f32(read, name + ".bias",   bias_fp32);

    // Pre-pack one [C_out_g, C_in_g] GEMM matrix per (group, kernel position)
    const int C_in_g  = C_in  / (groups > 0 ? groups : 1);
    const int C_out_g = C_out / (groups > 0 ? groups : 1);
    w_packed_k.resize((size_t)groups * kernel_size);
    std::vector<float> tmp((size_t)C_out_g * C_in_g);
    for (int g = 0; g < groups; ++g) {
        for (int k = 0; k < kernel_size; ++k) {
            for (int co = 0; co < C_out_g; ++co)
                for (int ci = 0; ci < C_in_g; ++ci)
                    tmp[(size_t)co * C_in_g + ci] =
                        weight_fp32[((size_t)(g*C_in_g + ci) * C_out_g + co) * kernel_size + k];
            float* packed = pack_weights_f32(tmp.data(), C_out_g, C_in_g);
            size_t nb = (size_t)(((C_out_g+3)/4)) * C_in_g * 4 * sizeof(float);
            w_packed_k[g * kernel_size + k] = Buffer(packed, nb, MemSpace::Host,
                [](void* p){ delete[] static_cast<float*>(p); });
        }
    }

    if (active_backend()->mem_space == MemSpace::Host) {
        if (groups == 1) release_vector(weight_fp32);
    } else {
        // GPU weight packing:
        // groups == 1: [kernel_size, C_out, C_in] row-major for MPS matmul.
        //   W_gpu[k, c_out, c_in] = weight_fp32[(c_in * C_out + c_out) * kernel_size + k]
        // depthwise (groups == C_in == C_out): [C, kernel_size] row-major.
        //   W_gpu[c, k] = weight_fp32[c * kernel_size + k]  (PyTorch [C,1,ksz] flattened)
        const bool is_depthwise = (groups > 1 && C_in_g == 1 && C_out_g == 1);
        if (groups == 1) {
            w_gpu_data.resize((size_t)kernel_size * C_out_g * C_in_g);
            for (int k = 0; k < kernel_size; ++k)
                for (int co = 0; co < C_out_g; ++co)
                    for (int ci = 0; ci < C_in_g; ++ci)
                        w_gpu_data[(size_t)k * C_out_g * C_in_g + (size_t)co * C_in_g + ci] =
                            weight_fp32[((size_t)ci * C_out_g + co) * kernel_size + k];
        } else if (is_depthwise) {
            // PyTorch layout: [C_in, C_out/groups, kernel_size] = [C, 1, ksz]
            // GPU layout: [C, ksz] — same order, just drop the trivial middle dim
            w_gpu_data.assign(weight_fp32.begin(), weight_fp32.end());
        }
    }
}

void ConvTranspose1dFp32::release_runtime_scratch()
{
    release_vector(pad_buf_);
}

Shape ConvTranspose1dFp32::forward(const TensorView* inputs, int n_inputs,
                                       BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const Shape out_sh = output_shape(sh);

    // ── Metal GPU path ────────────────────────────────────────────────────────
    // Supports groups==1 (GEMM+scatter) and depthwise (groups==C_in==C_out).
    const int C_in_g_  = C_in  / (groups > 0 ? groups : 1);
    const int C_out_g_ = C_out / (groups > 0 ? groups : 1);
    const bool is_depthwise_ = (groups > 1 && C_in_g_ == 1 && C_out_g_ == 1);
    if (in1.where == MemSpace::Metal &&
        (groups == 1 || is_depthwise_) &&
        !w_gpu_data.empty() &&
        active_backend()->conv_transpose1d_fp32_nhwc) {
        active_backend()->conv_transpose1d_fp32_nhwc(
            in1.as<float>(), w_gpu_data.data(),
            bias_fp32.empty() ? nullptr : bias_fp32.data(),
            out.as<float>(),
            sh.d0, sh.d1, C_out,
            kernel_size, stride, padding, output_padding,
            groups,
            sc.stream);
        return out_sh;
    }

    // ── CPU path ──────────────────────────────────────────────────────────────
    const bool _profile_conv_transpose = g_ops_profile_enabled;
    const double _t0_conv_transpose = _profile_conv_transpose ? now_ms() : 0.0;
    const float* in_buf  = in1.as<float>();
    float*       out_buf = out.as<float>();

    const int T     = sh.d1;
    const int C_in_ = sh.d0;
    const int T_out = out_sh.d1;
    const int C_out_g = C_out / (groups > 0 ? groups : 1);
    const int C_in_g  = C_in_ / (groups > 0 ? groups : 1);

    memset(out_buf, 0, (size_t)C_out * T_out * sizeof(float));

    if (groups == 1 && !w_packed_k.empty()) {
        // GEMM path: one sgemm per kernel position, then scatter-add.
        const size_t scratch_need = (size_t)T * C_out_g;
        float* scratch = conv1d_float_scratch(sc.buffers[Scratch::F32_A], pad_buf_, scratch_need);

        for (int k = 0; k < kernel_size; ++k) {
            sgemm_f32(in_buf, w_packed_k[k].as<float>(), nullptr, scratch,
                      false, T, C_in_g, C_out_g);

            for (int t = 0; t < T; ++t) {
                int t_out = t * stride + k - padding;
                if (t_out < 0 || t_out >= T_out) continue;
                float*       y = out_buf + (size_t)t_out * C_out;
                const float* s = scratch  + (size_t)t   * C_out_g;
                add_vectors_fp32(y, s, C_out_g);
            }
        }
    } else {
        conv_transpose1d_grouped_fp32(
            in_buf, out_buf, weight_fp32.data(),
            T, T_out, C_in_, C_out, C_in_g, C_out_g,
            groups, stride, padding, kernel_size);
    }

    if (!bias_fp32.empty())
        bias_add_rows_fp32(out_buf, bias_fp32.data(), T_out, C_out);

    if (_profile_conv_transpose) g_conv_transpose_ms += now_ms() - _t0_conv_transpose;
    return out_sh;
}
