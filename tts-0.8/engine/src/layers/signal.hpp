#pragma once
#include "engine/layer.hpp"

class StftFp32 : public ILayer {
public:
    int n_fft    = 20;
    int hop_size = 5;

    std::vector<float> w_real;
    std::vector<float> w_imag;

    void load_weights(const ReadTensorFn& read) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape dynamic_output_size(const void* const*, const Shape*, int) const override;
    Shape output_shape(Shape in) const override {
        int T_stft = in.d1 / hop_size + 1;
        return {n_fft + 2, T_stft, 1};
    }
    MemSpace required_mem_space() const override { return MemSpace::Host; }
};

class IstftFp32 : public ILayer {
public:
    int  n_fft      = 20;
    int  hop_size   = 5;
    bool normalized = false;

    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override {
        return {1, (in.d1 - 1) * hop_size + n_fft, 1};
    }
    MemSpace required_mem_space() const override {
        MemSpace ms = active_backend()->mem_space;
        return (ms == MemSpace::CUDA || ms == MemSpace::HIP) ? MemSpace::Host : ms;
    }
};

class SineGenFp32 : public ILayer {
public:
    int   sample_rate       = 24000;
    int   harmonic_num      = 8;
    float sine_amp          = 0.1f;
    float voiced_threshold  = 10.0f;
    float noise_std         = 0.0f;
    int   upsample_scale    = 0;

    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override {
        return {harmonic_num + 1, in.d1, in.d2};
    }
    MemSpace required_mem_space() const override { return MemSpace::Host; }
};
