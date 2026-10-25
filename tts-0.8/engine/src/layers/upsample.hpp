#pragma once
#include "engine/layer.hpp"

class UpsampleBilinearFp32 : public ILayer {
public:
    float scale_h = 2.0f, scale_w = 2.0f;
    void     load_weights(const ReadTensorFn&) override {}
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override {
        return {in.d0, (int)(in.d1 * scale_h + 0.5f), (int)(in.d2 * scale_w + 0.5f)};
    }
};

class UpsampleNearest1dFp32 : public ILayer {
public:
    int scale_factor = 2;

    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override {
        return {in.d0, in.d1 * scale_factor, in.d2};
    }
};
