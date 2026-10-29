#pragma once
#include "engine/layer.hpp"

class ConvFp32 : public ILayer {
public:
    int C_out = 0, C_in = 0, kH = 1, kW = 1;
    int stride_h = 1, stride_w = 1;
    int pad_h = 0, pad_w = 0;
    int groups = 1;
    int dilation_h = 1, dilation_w = 1;
    int K = 0;

    std::vector<float> weight_fp32;
    std::vector<float> bias_fp32;
    Buffer w_packed_f32;

    bool wino_f32_available = false;
    Buffer w_wino_f32_packed[16];

    void     load_weights(const ReadTensorFn&) override;
    void     upload_weights(Allocator&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    void     scratch_needed(Shape in, size_t out[Scratch::N]) const override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override {
        int oH = (in.d1 + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
        int oW = (in.d2 + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;
        return {C_out, oH, oW};
    }
};
