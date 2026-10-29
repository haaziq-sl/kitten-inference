// tensor.cpp — ModelIOTensor manipulation layer classes

#include <cstdio>
#include "tensor.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>

// ─────────────────────────────────────────────────────────────────
// Concat1dFp32
//   in1 {C1, T, 1}, in2 {C2, T_in2, 1} where T_in2=1 or T_in2=T
//   out {C1+C2, T, 1}
// ─────────────────────────────────────────────────────────────────

Shape Concat1dFp32::forward(const TensorView* inputs, int n_inputs,
                                BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    BufferView  in2      = (n_inputs > 1) ? inputs[1].view  : BufferView{};
    Shape in2_sh   = (n_inputs > 1) ? inputs[1].shape : Shape{};
    const bool broadcast = (in2_sh.d1 * in2_sh.d2 <= 1);
    active_backend()->concat1d_fp32(in1.as<float>(),
                                     in2.as<float>(),
                                     out.as<float>(),
                                     sh.d1 * sh.d2, sh.d0, C2, broadcast, sc.stream);
    return {sh.d0 + C2, sh.d1, sh.d2};
}

// ─────────────────────────────────────────────────────────────────
// SliceChannelsFp32
//   Select channel range [ch_start, ch_end)
//   Input:  {C_in, T, 1} → Output: {ch_end - ch_start, T, 1}
// ─────────────────────────────────────────────────────────────────

Shape SliceChannelsFp32::forward(const TensorView* inputs, int n_inputs,
                                     BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const int T     = sh.d1 * sh.d2;
    const int C_out = ch_end - ch_start;
    active_backend()->slice_channels_fp32(in1.as<float>(),
                                           out.as<float>(),
                                           T, sh.d0, ch_start, C_out, sc.stream);
    return {C_out, T, 1};
}

// ─────────────────────────────────────────────────────────────────
// ReflectionPad1dFp32
//   Reflect-pad 1 sample on left side: out[0] = in[1]
//   Input:  {C, T, 1} → Output: {C, T+1, 1}
// ─────────────────────────────────────────────────────────────────

Shape ReflectionPad1dFp32::forward(const TensorView* inputs, int n_inputs,
                                       BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const float* in      = in1.as<float>();
    float*       out_ptr = out.as<float>();
    const int T   = sh.d1 * sh.d2;
    const int C   = sh.d0;

    if (in1.where == MemSpace::Metal && active_backend()->reflection_pad1d_fp32_metal) {
        active_backend()->reflection_pad1d_fp32_metal(in, out_ptr, T, C, sc.stream);
        return {C, T + 1, 1};
    }

    memcpy(out_ptr,     in + C, C * sizeof(float));
    memcpy(out_ptr + C, in,     (size_t)T * C * sizeof(float));
    return {C, T + 1, 1};
}

// ─────────────────────────────────────────────────────────────────
// LengthRegulateFp32
//   in1 = features {C, T, 1}
//   in2 = durations {1, T, 1}
//   out = expanded {C, T_out, 1}
// ─────────────────────────────────────────────────────────────────

Shape LengthRegulateFp32::dynamic_output_size(const void* const* in_ptrs,
                                                   const Shape* in_shapes,
                                                   int /*n_inputs*/) const
{
    const float* durs = static_cast<const float*>(in_ptrs[1]);
    if (!durs) return in_shapes[0];
    int T_out = 0;
    for (int t = 0; t < in_shapes[1].d1; t++) {
        T_out += std::max(1, (int)std::round(durs[t] * duration_scale));
    }
    return {in_shapes[0].d0, T_out, 1};
}

Shape LengthRegulateFp32::forward(const TensorView* inputs, int n_inputs,
                                      BufferView out, ScratchPads& /*sc*/)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    BufferView  in2      = (n_inputs > 1) ? inputs[1].view  : BufferView{};
    Shape in2_sh   = (n_inputs > 1) ? inputs[1].shape : Shape{};
    (void)in2_sh;
    const int C = sh.d0;
    int out_t = length_regulate_fp32(in1.as<float>(),
                                     in2.as<float>(),
                                     out.as<float>(),
                                     sh.d1 * sh.d2, C,
                                     duration_scale);
    return {C, out_t, 1};
}

// ─────────────────────────────────────────────────────────────────
// ElemwiseMulFp32
//   out[i] = inputs[0][i] * inputs[1][i]  (same shape)
// ─────────────────────────────────────────────────────────────────

Shape ElemwiseMulFp32::forward(const TensorView* inputs, int n_inputs,
                                   BufferView out, ScratchPads& sc)
{
    Shape sh = inputs[0].shape;
    (void)n_inputs;
    const int N = sh.d0 * sh.d1 * sh.d2;
    active_backend()->elemwise_mul_fp32(inputs[0].view.as<float>(),
                                        inputs[1].view.as<float>(),
                                        out.as<float>(), N, sc.stream);
    return sh;
}

// ─────────────────────────────────────────────────────────────────
// SumChannelsFp32
//   Sum all C channels per time step → {1, T, 1}
// ─────────────────────────────────────────────────────────────────

Shape SumChannelsFp32::forward(const TensorView* inputs, int n_inputs,
                                   BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const int T = sh.d1 * sh.d2;
    active_backend()->sum_channels_fp32(in1.as<float>(),
                                         out.as<float>(), T, sh.d0, sc.stream);
    return {1, T, 1};
}
