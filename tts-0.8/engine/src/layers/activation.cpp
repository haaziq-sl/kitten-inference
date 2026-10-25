// activation.cpp — Activation layer classes: ActivationFp32, SnakeFp32, ScaleFp32

#include "activation.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"
#include <cstring>
#include <cmath>

Shape ActivationFp32::forward(const TensorView* inputs, int n_inputs,
                                   BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const float* in_ptr  = in1.as<float>();
    float*       out_ptr = out.as<float>();
    const int    N       = sh.d0 * sh.d1 * sh.d2;

    switch (act_type) {
    case ActType::LEAKY_RELU: active_backend()->leaky_relu_fp32(in_ptr, out_ptr, N, alpha, sc.stream);  break;
    case ActType::EXP:        active_backend()->exp_fp32       (in_ptr, out_ptr, N, sc.stream);          break;
    case ActType::SIN:        active_backend()->sin_fp32       (in_ptr, out_ptr, N, sc.stream);          break;
    case ActType::SIGMOID:    active_backend()->sigmoid_fp32   (in_ptr, out_ptr, N, sc.stream);          break;
    case ActType::TANH:    active_backend()->tanh_fp32    (in_ptr, out_ptr, N, sc.stream);          break;
    case ActType::GELU:
        if (in_ptr != out_ptr) {
            if (active_backend()->mem_space == MemSpace::Host)
                memcpy(out_ptr, in_ptr, (size_t)N * sizeof(float));
            else
                active_backend()->make_allocator()->copy_d2d(out_ptr, in_ptr, (size_t)N * sizeof(float));
        }
        active_backend()->gelu_fp32(out_ptr, N, sc.stream);
        break;
    case ActType::SILU:
        if (active_backend()->silu_fp32)
            active_backend()->silu_fp32(in_ptr, out_ptr, N, sc.stream);
        else { for (int i = 0; i < N; ++i) { float x = in_ptr[i]; out_ptr[i] = x / (1.f + expf(-x)); } }
        break;
    }
    return sh;
}

// ─────────────────────────────────────────────────────────────────
// SnakeFp32 — Snake1D activation x += sin(alpha*x)^2 / alpha
// ─────────────────────────────────────────────────────────────────

void SnakeFp32::load_weights(const ReadTensorFn& read)
{
    std::vector<int32_t> shape;
    std::vector<uint8_t> data;
    if (read(name + ".alpha", shape, data)) {
        alpha.resize(data.size() / sizeof(float));
        memcpy(alpha.data(), data.data(), data.size());
        inv_alpha.resize(alpha.size());
        for (size_t i = 0; i < alpha.size(); ++i)
            inv_alpha[i] = 1.f / (alpha[i] + 1e-9f);
    }
}

void SnakeFp32::upload_weights(Allocator& alloc)
{
    if (alloc.mem_space() == MemSpace::Host) return;
    auto up = [&](const std::vector<float>& src, Buffer& dst) {
        if (src.empty()) return;
        dst = alloc.make_buffer(src.size() * sizeof(float));
        alloc.copy_h2d(dst.ptr(), src.data(), src.size() * sizeof(float));
    };
    up(alpha,     alpha_d);
    up(inv_alpha, inv_alpha_d);
}

Shape SnakeFp32::forward(const TensorView* inputs, int n_inputs,
                             BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    active_backend()->snake1d_fp32(
        in1.as<float>(), out.as<float>(),
        alpha_d.valid()     ? alpha_d.as<float>()     : alpha.data(),
        inv_alpha_d.valid() ? inv_alpha_d.as<float>() : inv_alpha.data(),
        sh.d1 * sh.d2, sh.d0, sc.stream);
    return sh;
}

// ─────────────────────────────────────────────────────────────────
// ScaleFp32 — Multiply all elements by scalar constant
// ─────────────────────────────────────────────────────────────────

Shape ScaleFp32::forward(const TensorView* inputs, int n_inputs,
                             BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    active_backend()->scale_fp32(in1.as<float>(),
                                  out.as<float>(),
                                  (size_t)sh.d0 * sh.d1 * sh.d2, scalar, sc.stream);
    return sh;
}
