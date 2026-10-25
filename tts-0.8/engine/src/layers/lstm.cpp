// lstm.cpp — LstmFp32: single-layer (optionally bidirectional) LSTM
//
// Layout: input {C_in, T, 1} → buffer [T, C_in]; output {H_out, T, 1} → [T, H_out]
// where H_out = hidden_size * (1 + bidirectional).
//
// Implementation:
//   1. Pre-compute input projections [T, 4H] via sgemm_f32
//   2. Sequential LSTM scan (forward + optional backward)
//   3. Concatenate forward/backward hidden states

#include "lstm.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"
#include <cstring>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Use shared read_f32_tensor() from layer.hpp; thin alias kept for call-site compatibility.
static inline bool read_f32_lstm(const ReadTensorFn& r, const std::string& k,
                                  std::vector<float>& out)
{ return read_f32_tensor(r, k, out); }

template <typename T>
static void release_vector(std::vector<T>& v)
{
    std::vector<T>().swap(v);
}

// ─────────────────────────────────────────────────────────────────
// LstmFp32::load_weights
// ─────────────────────────────────────────────────────────────────
void LstmFp32::load_weights(const ReadTensorFn& read)
{
    const int H = hidden_size;
    const std::string& n = name;

    read_f32_lstm(read, n + ".weight_ih_l0", w_ih);
    read_f32_lstm(read, n + ".weight_hh_l0", w_hh);
    read_f32_lstm(read, n + ".bias_ih_l0",   b_ih);
    read_f32_lstm(read, n + ".bias_hh_l0",   b_hh);

    // Infer C_in from w_ih [4H, C_in].
    // Always pack using CPU format: this layer always runs on CPU (required_mem_space=Host).
    auto make_tb_f32 = [](float* pk, int C_out, int C_in) -> Buffer {
        size_t nb = (size_t)((C_out+3)/4) * C_in * 4 * sizeof(float);
        return Buffer(pk, nb, MemSpace::Host,
            [](void* p){ delete[] static_cast<float*>(p); });
    };

    if (!w_ih.empty()) {
        int C_in_inferred = (int)w_ih.size() / (4 * H);
        float* pk = pack_weights_f32(w_ih.data(), 4*H, C_in_inferred);
        w_ih_packed = make_tb_f32(pk, 4*H, C_in_inferred);
    }
    if (!w_hh.empty()) {
        float* pk = pack_weights_f32(w_hh.data(), 4*H, H);
        w_hh_packed = make_tb_f32(pk, 4*H, H);
    }

    if (bidirectional) {
        read_f32_lstm(read, n + ".weight_ih_l0_reverse", w_ih_r);
        read_f32_lstm(read, n + ".weight_hh_l0_reverse", w_hh_r);
        read_f32_lstm(read, n + ".bias_ih_l0_reverse",   b_ih_r);
        read_f32_lstm(read, n + ".bias_hh_l0_reverse",   b_hh_r);

        if (!w_ih_r.empty()) {
            int C_in_inferred = (int)w_ih_r.size() / (4 * H);
            float* pk = pack_weights_f32(w_ih_r.data(), 4*H, C_in_inferred);
            w_ih_r_packed = make_tb_f32(pk, 4*H, C_in_inferred);
        }
        if (!w_hh_r.empty()) {
            float* pk = pack_weights_f32(w_hh_r.data(), 4*H, H);
            w_hh_r_packed = make_tb_f32(pk, 4*H, H);
        }
    }

    if (!active_backend()->lstm_fp32_metal) {
        release_vector(w_ih);
        release_vector(w_hh);
        release_vector(w_ih_r);
        release_vector(w_hh_r);
    }
}

void LstmFp32::upload_weights(Allocator& alloc)
{
    if (!active_backend()->lstm_fp32_metal) return;  // no GPU path
    // Upload all weights as Metal shared buffers via allocator
    auto upload = [&](const std::vector<float>& v) -> Buffer {
        if (v.empty()) return Buffer{};
        Buffer tb = alloc.make_buffer(v.size() * sizeof(float));
        memcpy(tb.as<float>(), v.data(), v.size() * sizeof(float));
        return tb;
    };
    w_ih_gpu = upload(w_ih);
    b_ih_gpu = upload(b_ih);
    w_hh_gpu = upload(w_hh);
    b_hh_gpu = upload(b_hh);
    if (bidirectional) {
        w_ih_r_gpu = upload(w_ih_r);
        b_ih_r_gpu = upload(b_ih_r);
        w_hh_r_gpu = upload(w_hh_r);
        b_hh_r_gpu = upload(b_hh_r);
    }
}

// ─────────────────────────────────────────────────────────────────
// LstmFp32::forward
// ─────────────────────────────────────────────────────────────────
Shape LstmFp32::forward(const TensorView* inputs, int n_inputs,
                             BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const float* input  = in1.as<float>();
    float*       output = out.as<float>();

    const int T      = sh.d1 * sh.d2;   // time steps
    const int C_in_  = sh.d0;           // input channels

    const int H      = hidden_size;
    const int H_out  = H * (bidirectional ? 2 : 1);

    // ── Metal GPU path ─────────────────────────────────────────────
    if (in1.where == MemSpace::Metal && active_backend()->lstm_fp32_metal) {
        active_backend()->lstm_fp32_metal(
            input,
            w_ih_gpu.as<float>(), b_ih_gpu.valid() ? b_ih_gpu.as<float>() : nullptr,
            w_hh_gpu.as<float>(), b_hh_gpu.valid() ? b_hh_gpu.as<float>() : nullptr,
            bidirectional ? w_ih_r_gpu.as<float>() : nullptr,
            bidirectional ? (b_ih_r_gpu.valid() ? b_ih_r_gpu.as<float>() : nullptr) : nullptr,
            bidirectional ? w_hh_r_gpu.as<float>() : nullptr,
            bidirectional ? (b_hh_r_gpu.valid() ? b_hh_r_gpu.as<float>() : nullptr) : nullptr,
            output, T, C_in_, H, bidirectional, sc.stream);
        return {H_out, sh.d1, sh.d2};
    }

    std::vector<float> h_fwd, h_bwd;

    if (!bidirectional) {
        // ── Unidirectional ────────────────────────────────────
        std::vector<float> proj(T * 4 * H);
        sgemm_f32(input, w_ih_packed.as<float>(),
                                     b_ih.empty() ? nullptr : b_ih.data(),
                                     proj.data(), /*relu=*/false, T, C_in_, 4*H);
        h_fwd.resize((size_t)T * H);
        lstm_scan_fp32(proj.data(), w_hh.data(),
                  b_hh.empty() ? nullptr : b_hh.data(),
                  w_hh_packed.as<float>(), h_fwd.data(), T, H, /*reverse=*/false);
        memcpy(output, h_fwd.data(), (size_t)T * H * sizeof(float));
    } else {
        // ── Bidirectional: run fwd + bwd in parallel OMP sections ─
        std::vector<float> proj_f(T * 4 * H);
        std::vector<float> proj_r(T * 4 * H);
        h_fwd.resize((size_t)T * H);
        h_bwd.resize((size_t)T * H);

#ifdef _OPENMP
        #pragma omp parallel sections if(!omp_in_parallel())
        {
            #pragma omp section
            {
                // in_parallel=false: sgemm detects omp_in_parallel()==true
                // and runs single-threaded on this thread, computing full GEMM.
                sgemm_f32(input, w_ih_packed.as<float>(),
                                             b_ih.empty() ? nullptr : b_ih.data(),
                                             proj_f.data(), false, T, C_in_, 4*H);
                lstm_scan_fp32(proj_f.data(), w_hh.data(),
                          b_hh.empty() ? nullptr : b_hh.data(),
                          w_hh_packed.as<float>(), h_fwd.data(), T, H, false);
            }
            #pragma omp section
            {
                sgemm_f32(input, w_ih_r_packed.as<float>(),
                                             b_ih_r.empty() ? nullptr : b_ih_r.data(),
                                             proj_r.data(), false, T, C_in_, 4*H);
                lstm_scan_fp32(proj_r.data(), w_hh_r.data(),
                          b_hh_r.empty() ? nullptr : b_hh_r.data(),
                          w_hh_r_packed.as<float>(), h_bwd.data(), T, H, true);
            }
        }
#else
        // Sequential fallback (no OMP)
        sgemm_f32(input, w_ih_packed.as<float>(),
                                     b_ih.empty() ? nullptr : b_ih.data(),
                                     proj_f.data(), false, T, C_in_, 4*H);
        lstm_scan_fp32(proj_f.data(), w_hh.data(),
                  b_hh.empty() ? nullptr : b_hh.data(),
                  w_hh_packed.as<float>(), h_fwd.data(), T, H, false);
        sgemm_f32(input, w_ih_r_packed.as<float>(),
                                     b_ih_r.empty() ? nullptr : b_ih_r.data(),
                                     proj_r.data(), false, T, C_in_, 4*H);
        lstm_scan_fp32(proj_r.data(), w_hh_r.data(),
                  b_hh_r.empty() ? nullptr : b_hh_r.data(),
                  w_hh_r_packed.as<float>(), h_bwd.data(), T, H, true);
#endif

        // Interleave fwd + bwd hidden states into output [T, H_out]
        for (int t = 0; t < T; ++t) {
            memcpy(output + (size_t)t * H_out,     h_fwd.data() + (size_t)t * H, H * sizeof(float));
            memcpy(output + (size_t)t * H_out + H, h_bwd.data() + (size_t)t * H, H * sizeof(float));
        }
    }

    return {H_out, sh.d1, sh.d2};
}
