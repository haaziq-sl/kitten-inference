#pragma once
#include "engine/layer.hpp"

class LstmFp32 : public ILayer {
public:
    int  hidden_size  = 0;
    bool bidirectional = false;

    std::vector<float> w_ih;
    std::vector<float> w_hh;
    std::vector<float> b_ih;
    std::vector<float> b_hh;
    Buffer w_ih_packed;
    Buffer w_hh_packed;

    std::vector<float> w_ih_r;
    std::vector<float> w_hh_r;
    std::vector<float> b_ih_r;
    std::vector<float> b_hh_r;
    Buffer w_ih_r_packed;
    Buffer w_hh_r_packed;

    Buffer w_ih_gpu, b_ih_gpu, w_hh_gpu, b_hh_gpu;
    Buffer w_ih_r_gpu, b_ih_r_gpu, w_hh_r_gpu, b_hh_r_gpu;

    void     load_weights(const ReadTensorFn&) override;
    void     upload_weights(Allocator&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    MemSpace required_mem_space() const override {
        MemSpace ms = active_backend()->mem_space;
        return (ms == MemSpace::CUDA || ms == MemSpace::HIP) ? MemSpace::Host : ms;
    }
    Shape output_shape(Shape in) const override {
        return {hidden_size * (bidirectional ? 2 : 1), in.d1, in.d2};
    }
};
