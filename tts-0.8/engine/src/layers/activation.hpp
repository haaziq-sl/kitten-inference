#pragma once
#include "engine/layer.hpp"

enum class ActType { LEAKY_RELU, EXP, SIN, SIGMOID, TANH, GELU, SILU };

class ActivationFp32 : public ILayer {
public:
    ActType act_type = ActType::LEAKY_RELU;
    float   alpha    = 0.2f;

    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return in; }
};

class SnakeFp32 : public ILayer {
public:
    std::vector<float> alpha;
    std::vector<float> inv_alpha;
    Buffer alpha_d;
    Buffer inv_alpha_d;

    void     load_weights(const ReadTensorFn&) override;
    void     upload_weights(Allocator&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return in; }
};

class ScaleFp32 : public ILayer {
public:
    float scalar = 1.0f;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return in; }
};
