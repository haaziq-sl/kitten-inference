#include "pool.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"

Shape MaxPoolInt8::forward(const TensorView* inputs, int n_inputs,
                               BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    Shape osh = output_shape(sh);
    active_backend()->maxpool_int8_nhwc(in1.as<int8_t>(), out.as<int8_t>(),
                                         sh.d0, sh.d1, sh.d2,
                                         kH, kW, stride_h, stride_w, pad_h, pad_w, sc.stream);
    return osh;
}

Shape MaxPoolFp32::forward(const TensorView* inputs, int n_inputs,
                               BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    Shape osh = output_shape(sh);
    active_backend()->maxpool_fp32_nhwc(in1.as<float>(), out.as<float>(),
                                         sh.d0, sh.d1, sh.d2,
                                         kH, kW, stride_h, stride_w, pad_h, pad_w, sc.stream);
    return osh;
}

Shape AvgPoolInt8::forward(const TensorView* inputs, int n_inputs,
                               BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    active_backend()->avgpool_global_int8_nhwc(in1.as<int8_t>(), in_scale, in_zp, out_scale, out_zp,
                                                out.as<int8_t>(), sh.d0, sh.d1, sh.d2, sc.stream);
    return {sh.d0, 1, 1};
}

Shape AvgPoolFp32::forward(const TensorView* inputs, int n_inputs,
                               BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    active_backend()->avgpool_global_fp32_nhwc(in1.as<float>(), out.as<float>(),
                                                sh.d0, sh.d1, sh.d2, sc.stream);
    return {sh.d0, 1, 1};
}
