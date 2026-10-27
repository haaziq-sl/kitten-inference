#pragma once
#include "engine/layer.hpp"

class GemmInt8 : public ILayer {
public:
    int C_in = 0, C_out = 0;
    int8_t in_zp = 0;

    std::vector<int8_t>  weight;
    std::vector<float>   req_scale;
    std::vector<int32_t> req_mult;
    std::vector<int32_t> req_exp;
    std::vector<int64_t> eff_bias;
    Buffer w_packed;
    std::vector<int32_t> w_row_sums;

    void     load_weights(const ReadTensorFn&) override;
    void     upload_weights(Allocator&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape) const override { return {C_out, 1, 1}; }
};

class GemmFp32 : public ILayer {
public:
    int C_in = 0, C_out = 0;

    std::vector<float> weight_fp32;
    std::vector<float> bias_fp32;
    Buffer w_packed_f32;

    void     load_weights(const ReadTensorFn&) override;
    void     upload_weights(Allocator&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape) const override { return {C_out, 1, 1}; }
};
