#pragma once
#include "engine/layer.hpp"

// ── RMSNorm (used by LLaMA, Gemma, etc.) ─────────────────────────
// y = x / rms(x) * gamma,   rms = sqrt(mean(x^2) + eps)
// No beta (mean subtraction is absent).
class RmsNormFp32 : public ILayer {
public:
    float eps = 1e-6f;
    std::vector<float> gamma;

    void     load_weights(const ReadTensorFn&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return in; }
    // No required_mem_space() override — defaults to active backend's space.
    // Metal backend implements rmsnorm_k GPU kernel; CPU path uses CPU allocations.
};

class LayerNormFp32 : public ILayer {
public:
    float eps = 1e-6f;
    std::vector<float> gamma;
    std::vector<float> beta;

    void     load_weights(const ReadTensorFn&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return in; }
};

class AdaIn1dFp32 : public ILayer {
public:
    int C = 0, C_style = 0;
    float eps = 1e-5f;

    std::vector<float> fc_weight;
    std::vector<float> fc_bias;
    std::vector<float> norm_weight;
    std::vector<float> norm_bias;
    Buffer fc_packed;
    Buffer gamma_beta_d_;

    mutable std::vector<float> gamma_beta_buf_;

    void     load_weights(const ReadTensorFn&) override;
    void     upload_weights(Allocator&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    void release_runtime_scratch() override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return in; }
};

class AdaLayerNormFp32 : public ILayer {
public:
    int C = 0, C_style = 0;
    float eps = 1e-5f;

    std::vector<float> fc_weight;
    std::vector<float> fc_bias;
    Buffer fc_packed;
    Buffer gamma_beta_d_;

    mutable std::vector<float> gamma_beta_buf_;

    void     load_weights(const ReadTensorFn&) override;
    void     upload_weights(Allocator&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    void release_runtime_scratch() override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return in; }
};
