// norm.cpp — Normalization layer classes: LayerNorm, AdaIN, AdaLayerNorm

#include <cstdio>
#include "norm.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"
#include <cstring>

// Use shared read_f32_tensor() from layer.hpp; thin alias kept for call-site compatibility.
static inline bool read_f32_norm(const ReadTensorFn& r, const std::string& k,
                                  std::vector<float>& out)
{ return read_f32_tensor(r, k, out); }

template <typename T>
static void release_vector(std::vector<T>& v)
{
    std::vector<T>().swap(v);
}

// ─────────────────────────────────────────────────────────────────
// RmsNormFp32
// ─────────────────────────────────────────────────────────────────

void RmsNormFp32::load_weights(const ReadTensorFn& read)
{
    read_f32_norm(read, name + ".gamma", gamma);
}

Shape RmsNormFp32::forward(const TensorView* inputs, int n_inputs,
                            BufferView out, ScratchPads& sc)
{
    const Shape sh = inputs[0].shape;
    (void)n_inputs;
    const float* input  = inputs[0].view.as<float>();
    float*       output = out.as<float>();
    int N = sh.d1 * sh.d2;
    int C = sh.d0;
    active_backend()->rmsnorm_fp32(input, gamma.data(), output, N, C, eps, sc.stream);
    return sh;
}

// ─────────────────────────────────────────────────────────────────
// LayerNormFp32
// ─────────────────────────────────────────────────────────────────

void LayerNormFp32::load_weights(const ReadTensorFn& read)
{
    read_f32_norm(read, name + ".gamma", gamma);
    read_f32_norm(read, name + ".beta",  beta);
}

Shape LayerNormFp32::forward(const TensorView* inputs, int n_inputs,
                                  BufferView out, ScratchPads& sc)
{
    BufferView  in1       = inputs[0].view;
    Shape in_shape  = inputs[0].shape;
    (void)n_inputs;
    const float* input  = in1.as<float>();
    float*       output = out.as<float>();
    int N = in_shape.d1 * in_shape.d2;
    int C = in_shape.d0;
    active_backend()->layernorm_fp32(input, gamma.data(), beta.data(), output, N, C, eps, sc.stream);
    return in_shape;
}

// ─────────────────────────────────────────────────────────────────
// AdaIn1dFp32
// ─────────────────────────────────────────────────────────────────

void AdaIn1dFp32::load_weights(const ReadTensorFn& read)
{
    read_f32_or_i8_tensor(read, name + ".fc.weight", fc_weight);
    read_f32_norm(read, name + ".fc.bias",     fc_bias);
    read_f32_norm(read, name + ".norm.weight", norm_weight);  // InstanceNorm learnable scale
    read_f32_norm(read, name + ".norm.bias",   norm_bias);    // InstanceNorm learnable shift

    if (!fc_weight.empty()) {
        auto* be = active_backend();
        float* pk = be->pack_weights_fp32(fc_weight.data(), 2*C, C_style);
        size_t N_pad = (size_t)((2*C + 3) / 4) * 4;
        size_t K_pad = (size_t)((C_style + 3) / 4) * 4;
        size_t nb = N_pad * K_pad * sizeof(float);
        fc_packed = Buffer(pk, nb, MemSpace::Host,
            [](void* p){ active_backend()->free_packed_fp32(static_cast<float*>(p)); });
        if (be->mem_space == MemSpace::Host) {
            std::vector<float>().swap(fc_weight);
        }
    }

    // Pre-allocate CPU scratch (fixed size = 2*C). Avoids malloc on every forward() call.
    gamma_beta_buf_.resize(2 * C);
}

void AdaIn1dFp32::upload_weights(Allocator& alloc)
{
    if (alloc.mem_space() == MemSpace::Host) return;
    if (fc_packed.valid()) {
        Buffer d = alloc.make_buffer(fc_packed.bytes());
        alloc.copy_h2d(d.ptr(), fc_packed.ptr(), fc_packed.bytes());
        fc_packed = std::move(d);
    }
    gamma_beta_d_ = alloc.make_buffer(2 * C * sizeof(float));
}

Shape AdaIn1dFp32::forward(const TensorView* inputs, int n_inputs,
                               BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    BufferView  in2      = (n_inputs > 1) ? inputs[1].view  : BufferView{};
    Shape in2_sh   = (n_inputs > 1) ? inputs[1].shape : Shape{};
    (void)in2_sh;
    const float* feat    = in1.as<float>();
    const float* style   = in2.as<float>();
    float*       out_ptr = out.as<float>();
    const int T      = sh.d1 * sh.d2;
    const int C_feat = sh.d0;

    if (gamma_beta_d_.valid()) {
        // GPU path: GEMM → gamma/beta on GPU, then AdaIN on GPU
        active_backend()->gemm_fp32_vec(
            style, nullptr,
            fc_bias.empty() ? nullptr : fc_bias.data(),
            gamma_beta_d_.as<float>(),
            C_style, 2 * C_feat,
            fc_packed.as<float>(), sc.stream);
        active_backend()->ada_in1d_fp32(
            feat, out_ptr,
            gamma_beta_d_.as<float>(),
            gamma_beta_d_.as<float>() + C_feat,
            T, C_feat, eps, sc.stream,
            norm_weight.empty() ? nullptr : norm_weight.data(),
            norm_bias.empty()   ? nullptr : norm_bias.data());
    } else {
        // CPU path (gamma_beta_buf_ is pre-allocated in load_weights to 2*C)
        if (gamma_beta_buf_.size() < (size_t)2 * C_feat)
            gamma_beta_buf_.resize((size_t)2 * C_feat);
        gemm_fp32_vec(style, fc_weight.data(),
                      fc_bias.empty() ? nullptr : fc_bias.data(),
                      gamma_beta_buf_.data(), C_style, 2 * C_feat,
                      fc_packed.as<float>(), sc.stream);
        ada_in1d_fp32(feat, out_ptr,
                      gamma_beta_buf_.data(), gamma_beta_buf_.data() + C_feat,
                      T, C_feat, eps, nullptr,
                      norm_weight.empty() ? nullptr : norm_weight.data(),
                      norm_bias.empty()   ? nullptr : norm_bias.data());
    }
    return sh;
}

void AdaIn1dFp32::release_runtime_scratch()
{
    release_vector(gamma_beta_buf_);
}

// ─────────────────────────────────────────────────────────────────
// AdaLayerNormFp32
// ─────────────────────────────────────────────────────────────────

void AdaLayerNormFp32::load_weights(const ReadTensorFn& read)
{
    read_f32_or_i8_tensor(read, name + ".fc.weight", fc_weight);
    read_f32_norm(read, name + ".fc.bias",   fc_bias);

    if (!fc_weight.empty()) {
        auto* be = active_backend();
        float* pk = be->pack_weights_fp32(fc_weight.data(), 2*C, C_style);
        size_t N_pad = (size_t)((2*C + 3) / 4) * 4;
        size_t K_pad = (size_t)((C_style + 3) / 4) * 4;
        size_t nb = N_pad * K_pad * sizeof(float);
        fc_packed = Buffer(pk, nb, MemSpace::Host,
            [](void* p){ active_backend()->free_packed_fp32(static_cast<float*>(p)); });
        if (be->mem_space == MemSpace::Host) {
            std::vector<float>().swap(fc_weight);
        }
    }

    // Pre-allocate CPU scratch (fixed size = 2*C). Avoids malloc on every forward() call.
    gamma_beta_buf_.resize(2 * C);
}

void AdaLayerNormFp32::upload_weights(Allocator& alloc)
{
    if (alloc.mem_space() == MemSpace::Host) return;
    if (fc_packed.valid()) {
        Buffer d = alloc.make_buffer(fc_packed.bytes());
        alloc.copy_h2d(d.ptr(), fc_packed.ptr(), fc_packed.bytes());
        fc_packed = std::move(d);
    }
    gamma_beta_d_ = alloc.make_buffer(2 * C * sizeof(float));
}

Shape AdaLayerNormFp32::forward(const TensorView* inputs, int n_inputs,
                                    BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    BufferView  in2      = (n_inputs > 1) ? inputs[1].view  : BufferView{};
    Shape in2_sh   = (n_inputs > 1) ? inputs[1].shape : Shape{};
    (void)in2_sh;
    const float* feat    = in1.as<float>();
    const float* style   = in2.as<float>();
    float*       out_ptr = out.as<float>();
    const int T      = sh.d1 * sh.d2;
    const int C_feat = sh.d0;

    if (gamma_beta_d_.valid()) {
        // GPU path
        active_backend()->gemm_fp32_vec(
            style, nullptr,
            fc_bias.empty() ? nullptr : fc_bias.data(),
            gamma_beta_d_.as<float>(),
            C_style, 2 * C_feat,
            fc_packed.as<float>(), sc.stream);
        active_backend()->ada_layer_norm1d_fp32(
            feat, out_ptr,
            gamma_beta_d_.as<float>(),
            gamma_beta_d_.as<float>() + C_feat,
            T, C_feat, eps, sc.stream);
    } else {
        // CPU path (gamma_beta_buf_ is pre-allocated in load_weights to 2*C)
        if (gamma_beta_buf_.size() < (size_t)2 * C_feat)
            gamma_beta_buf_.resize((size_t)2 * C_feat);
        gemm_fp32_vec(style, fc_weight.data(),
                      fc_bias.empty() ? nullptr : fc_bias.data(),
                      gamma_beta_buf_.data(), C_style, 2 * C_feat,
                      fc_packed.as<float>(), sc.stream);
        ada_layer_norm1d_fp32(feat, out_ptr,
                              gamma_beta_buf_.data(), gamma_beta_buf_.data() + C_feat,
                              T, C_feat, eps);
    }
    return sh;
}

void AdaLayerNormFp32::release_runtime_scratch()
{
    release_vector(gamma_beta_buf_);
}
