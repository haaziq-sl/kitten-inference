#pragma once
#include "engine/layer.hpp"

class LengthRegulateFp32 : public ILayer {
public:
    float duration_scale = 1.0f;

    void set_duration_scale(float scale) { duration_scale = scale > 0.0f ? scale : 1.0f; }
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return in; }

    Shape dynamic_output_size(const void* const*, const Shape*, int) const override;
    MemSpace required_mem_space() const override { return MemSpace::Host; }
};

class Concat1dFp32 : public ILayer {
public:
    int C2 = 0;

    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return {in.d0 + C2, in.d1, in.d2}; }
};

class ReflectionPad1dFp32 : public ILayer {
public:
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape dynamic_output_size(const void* const*, const Shape* in_shapes, int) const override {
        return {in_shapes[0].d0, in_shapes[0].d1 * in_shapes[0].d2 + 1, 1};
    }
    Shape output_shape(Shape in) const override { return {in.d0, in.d1 * in.d2 + 1, 1}; }
    MemSpace required_mem_space() const override {
        MemSpace ms = active_backend()->mem_space;
        return (ms == MemSpace::CUDA || ms == MemSpace::HIP) ? MemSpace::Host : ms;
    }
};

class SliceChannelsFp32 : public ILayer {
public:
    int ch_start = 0, ch_end = 0;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return {ch_end - ch_start, in.d1, in.d2}; }
};

class SumChannelsFp32 : public ILayer {
public:
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return {1, in.d1, in.d2}; }
};

// Elementwise multiply of two same-shape tensors: out[i] = a[i] * b[i]
// inputs[0] = a, inputs[1] = b
class ElemwiseMulFp32 : public ILayer {
public:
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return in; }
    Shape output_shape(const Shape* inputs, int n) const override {
        return n > 0 ? inputs[0] : Shape{};
    }
};
