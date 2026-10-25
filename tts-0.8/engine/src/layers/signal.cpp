// signal.cpp — Signal processing layer classes: STFT, ISTFT, SineGen

#include "signal.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"

// ─────────────────────────────────────────────────────────────────
// IstftFp32
//   in1 = magnitude  {n_fft/2+1, T, 1}
//   in2 = phase      {n_fft/2+1, T, 1}
//   out = waveform   {1, (T-1)*hop, 1}  (n_fft/2 trimmed each end)
// ─────────────────────────────────────────────────────────────────

Shape IstftFp32::forward(const TensorView* inputs, int n_inputs,
                             BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    BufferView  in2      = (n_inputs > 1) ? inputs[1].view  : BufferView{};
    Shape in2_sh   = (n_inputs > 1) ? inputs[1].shape : Shape{};
    (void)in2_sh;
    const int T_frames  = sh.d1 * sh.d2;
    const int out_len   = (T_frames - 1) * hop_size;   // trimmed length

    if (in1.where == MemSpace::Metal && active_backend()->istft_fp32_metal) {
        active_backend()->istft_fp32_metal(
            in1.as<float>(), in2.as<float>(),
            out.as<float>(),
            T_frames, n_fft, hop_size, normalized, sc.stream);
        return {1, out_len, 1};
    }

    istft_fp32(in1.as<float>(),
               in2.as<float>(),
               out.as<float>(),
               T_frames, n_fft, hop_size, normalized);
    return {1, out_len, 1};
}

// ─────────────────────────────────────────────────────────────────
// StftFp32
//   Input:  {1, T_audio, 1}
//   Output: {2*(n_fft/2+1), T_stft, 1}
// ─────────────────────────────────────────────────────────────────

void StftFp32::load_weights(const ReadTensorFn& read)
{
    // Load precomputed Hann-windowed cosine/sine filters from bin file.
    // These are extracted from the ONNX model's weight_forward_real/imag
    // Conv weights to ensure bit-exact match with ONNX float32 accumulation.
    // Stored as [K, n_fft] (row-major), K = n_fft/2+1.
    read_f32_tensor(read, name + ".w_real", w_real);
    read_f32_tensor(read, name + ".w_imag", w_imag);
    // If weights not found in bin, stft_fp32() falls back to computing them.
}

Shape StftFp32::dynamic_output_size(const void* const* /*in_ptrs*/,
                                        const Shape* in_shapes,
                                        int /*n_inputs*/) const
{
    int T_audio = in_shapes[0].d1 * in_shapes[0].d2;
    int T_stft  = T_audio / hop_size + 1;
    return {n_fft + 2, T_stft, 1};
}

Shape StftFp32::forward(const TensorView* inputs, int n_inputs,
                            BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const int T_audio = sh.d1 * sh.d2;
    const int T_stft  = T_audio / hop_size + 1;
    const int K       = n_fft / 2 + 1;
    const float* wr   = w_real.empty() ? nullptr : w_real.data();
    const float* wi   = w_imag.empty() ? nullptr : w_imag.data();


    if (in1.where == MemSpace::Metal && !w_real.empty() && !w_imag.empty()
        && active_backend()->stft_fp32_metal) {
        active_backend()->stft_fp32_metal(
            in1.as<float>(), wr, wi,
            out.as<float>(),
            T_audio, n_fft, hop_size, sc.stream);
        return {2 * K, T_stft, 1};
    }

    stft_fp32(in1.as<float>(),
              out.as<float>(),
              T_audio, n_fft, hop_size, wr, wi);
    return {2 * K, T_stft, 1};
}

// ─────────────────────────────────────────────────────────────────
// SineGenFp32
//   Input:  F0 {1, T, 1}  (Hz, audio-rate)
//   Output: harmonics {harmonic_num+1, T, 1}
// ─────────────────────────────────────────────────────────────────

Shape SineGenFp32::forward(const TensorView* inputs, int n_inputs,
                               BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const int H = harmonic_num + 1;

    if (in1.where == MemSpace::Metal && upsample_scale > 0
        && active_backend()->sine_gen_fp32_metal) {
        active_backend()->sine_gen_fp32_metal(
            in1.as<float>(), out.as<float>(),
            sh.d1 * sh.d2, H, sample_rate, upsample_scale,
            sine_amp, voiced_threshold, sc.stream);
        return {H, sh.d1, sh.d2};
    }

    sine_gen_fp32(in1.as<float>(),
                  out.as<float>(),
                  sh.d1 * sh.d2, harmonic_num, sample_rate,
                  sine_amp, voiced_threshold, noise_std, upsample_scale);
    return {H, sh.d1, sh.d2};
}
