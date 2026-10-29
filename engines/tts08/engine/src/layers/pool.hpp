#pragma once
#include "engine/layer.hpp"

class MaxPoolInt8 : public ILayer {
public:
    int kH = 3, kW = 3;
    int stride_h = 2, stride_w = 2;
    int pad_h = 0, pad_w = 0;
    int8_t in_zp = 0, out_zp = 0;

    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::int8(); }
    Shape output_shape(Shape in) const override {
        int oH = (in.d1 + 2*pad_h - kH) / stride_h + 1;
        int oW = (in.d2 + 2*pad_w - kW) / stride_w + 1;
        return {in.d0, oH, oW};
    }
};

class MaxPoolFp32 : public ILayer {
public:
    int kH = 3, kW = 3;
    int stride_h = 2, stride_w = 2;
    int pad_h = 0, pad_w = 0;

    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override {
        int oH = (in.d1 + 2*pad_h - kH) / stride_h + 1;
        int oW = (in.d2 + 2*pad_w - kW) / stride_w + 1;
        return {in.d0, oH, oW};
    }
};

class AvgPoolInt8 : public ILayer {
public:
    float in_scale = 1.f, out_scale = 1.f;
    int   in_zp = 0, out_zp = 0;

    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::int8(); }
    Shape output_shape(Shape in) const override { return {in.d0, 1, 1}; }
};

class AvgPoolFp32 : public ILayer {
public:
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return {in.d0, 1, 1}; }
};
