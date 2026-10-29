#include "add.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"

Shape AddInt8::forward(const TensorView* inputs, int n_inputs,
                           BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    BufferView  in2      = (n_inputs > 1) ? inputs[1].view  : BufferView{};
    Shape in2_sh   = (n_inputs > 1) ? inputs[1].shape : Shape{};
    (void)in2_sh;
    const int8_t* a = in1.as<int8_t>();
    const int8_t* b = in2.as<int8_t>();
    int8_t*       c = out.as<int8_t>();
    active_backend()->add_requant_int8(a, b, in1_scale, in1_zp, in2_scale, in2_zp,
                                        out_scale, out_zp, c, sh.d0 * sh.d1 * sh.d2, sc.stream);
    return sh;
}

Shape AddFp32::forward(const TensorView* inputs, int n_inputs,
                           BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    BufferView  in2      = (n_inputs > 1) ? inputs[1].view  : BufferView{};
    Shape in2_sh   = (n_inputs > 1) ? inputs[1].shape : Shape{};
    (void)in2_sh;
    const float* a = in1.as<float>();
    const float* b = in2.as<float>();
    float*       c = out.as<float>();
    active_backend()->add_fp32(a, b, c, sh.d0 * sh.d1 * sh.d2, relu, sc.stream);
    return sh;
}
