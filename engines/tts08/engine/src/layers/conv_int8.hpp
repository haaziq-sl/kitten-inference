#pragma once
#include "engine/layer.hpp"

class ConvInt8 : public ILayer {
public:
    int C_out = 0, C_in = 0, kH = 1, kW = 1;
    int stride_h = 1, stride_w = 1;
    int pad_h = 0, pad_w = 0;
    int groups = 1;
    int8_t in_zp = 0, out_zp = 0;
    int K = 0;

    std::vector<int8_t>  weight;
    std::vector<float>   req_scale;
    std::vector<int32_t> req_mult;
    std::vector<int32_t> req_exp;
    std::vector<int64_t> eff_bias;
    std::vector<int64_t> wino_eff_bias;

    Buffer w_packed;
    Buffer w_packed_nhwc;
    std::vector<int32_t> w_row_sums;
    std::vector<int32_t> w_wino_row_sums[16];

    // NHWC depthwise kernel: pre-transposed weights [9, C] and truncated bias [C]
    std::vector<int8_t>  w_dw_hwc;
    std::vector<int32_t> dw_eff_bias32;

    bool  wino_available = false;
    float wino_weight_scale[16] = {};
    float wino_input_scale = 0.25f;
    Buffer w_wino_packed[16];

    void     load_weights(const ReadTensorFn&) override;
    void     upload_weights(Allocator&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    void     scratch_needed(Shape in, size_t out[Scratch::N]) const override;
    DType out_dtype() const override { return DType::int8(); }
    Shape output_shape(Shape in) const override {
        int oH = (in.d1 + 2*pad_h - kH) / stride_h + 1;
        int oW = (in.d2 + 2*pad_w - kW) / stride_w + 1;
        return {C_out, oH, oW};
    }
    MemSpace required_mem_space() const override {
        return active_backend()->mem_space;
    }
};
