// upsample.cpp — Upsample layer classes: UpsampleBilinearFp32, UpsampleNearest1dFp32

#include "upsample.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"

Shape UpsampleBilinearFp32::forward(const TensorView* inputs, int n_inputs,
                                        BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const Shape osh = output_shape(sh);
    active_backend()->upsample_bilinear2d_fp32(in1.as<float>(), out.as<float>(),
                                                sh.d0, sh.d1, sh.d2, osh.d1, osh.d2, sc.stream);
    return osh;
}

// ─────────────────────────────────────────────────────────────────
// UpsampleNearest1dFp32
// ─────────────────────────────────────────────────────────────────

Shape UpsampleNearest1dFp32::forward(const TensorView* inputs, int n_inputs,
                                         BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const int T = sh.d1 * sh.d2;
    const int C = sh.d0;
    active_backend()->upsample_nearest1d_fp32(in1.as<float>(), out.as<float>(),
                                               T, C, scale_factor, sc.stream);
    return {C, T * scale_factor, 1};
}
