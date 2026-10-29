#pragma once
#include "engine/layer.hpp"

class AddInt8 : public ILayer {
public:
    float in1_scale = 1.f, in2_scale = 1.f, out_scale = 1.f;
    int   in1_zp = 0, in2_zp = 0, out_zp = 0;

    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::int8(); }
    Shape output_shape(Shape in) const override { return in; }
};

class AddFp32 : public ILayer {
public:
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return in; }
};
