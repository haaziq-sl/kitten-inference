#include "gemm.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"
#include <cstring>
#include <vector>

// ── GemmInt8 ──────────────────────────────────────────────────

void GemmInt8::load_weights(const ReadTensorFn& read)
{
    const std::string base = name;
    std::vector<int32_t> shape;
    std::vector<uint8_t> data;

    if (read(base + ".weight", shape, data)) tensor_values(data, weight, base + ".weight");
    if (read(base + ".req_scale", shape, data)) tensor_values(data, req_scale, base + ".req_scale");
    if (read(base + ".eff_bias", shape, data)) tensor_values(data, eff_bias, base + ".eff_bias");
    need_values(weight, layer_elems(name, {C_out, C_in}), base + ".weight");
    need_values(req_scale, (size_t)C_out, base + ".req_scale");
    need_values(eff_bias, (size_t)C_out, base + ".eff_bias");

    // Pre-compute Q31 multipliers
    req_mult.resize(C_out);
    req_exp.resize(C_out);
    for (int c = 0; c < C_out; ++c)
        QuantizeMultiplier(req_scale[c], &req_mult[c], &req_exp[c]);

    // Pre-pack for SDOT
    auto* _be = active_backend();
    int8_t* pk = _be->pack_weights_int8(weight.data(), C_out, C_in);
    size_t nb = (size_t)((C_out+7)/8) * ((C_in+3)/4) * 32;
    w_packed = Buffer(pk, nb, MemSpace::Host, [_be](void* p){ _be->free_packed_int8(static_cast<int8_t*>(p)); });

    // Weight row sums
    w_row_sums.resize(C_out);
    for (int nn = 0; nn < C_out; ++nn) {
        int32_t s = 0;
        for (int k = 0; k < C_in; ++k) s += weight[(size_t)nn * C_in + k];
        w_row_sums[nn] = s;
    }
}

void GemmInt8::upload_weights(Allocator& alloc)
{
    if (alloc.mem_space() == MemSpace::Host || !w_packed.valid()) return;
    Buffer d = alloc.make_buffer(w_packed.bytes());
    alloc.copy_h2d(d.ptr(), w_packed.ptr(), w_packed.bytes());
    w_packed = std::move(d);
}

Shape GemmInt8::forward(const TensorView* inputs, int n_inputs,
                            BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape in_shape = inputs[0].shape;
    (void)in_shape; (void)n_inputs;
    const int8_t* A = in1.as<int8_t>();
    float*        C = out.as<float>();
    active_backend()->gemm_int8(
        A, w_packed.as<int8_t>(),
        eff_bias.data(), nullptr, nullptr, req_scale.data(),
        /*out_zp=*/0,
        C, /*is_float=*/true,
        /*M=*/1, /*K=*/C_in, /*N=*/C_out,
        /*nchw_out=*/false, /*in_parallel=*/false,
        w_row_sums.empty() ? nullptr : w_row_sums.data(),
        sc.stream);
    if (relu) {
        for (int i = 0; i < C_out; i++) C[i] = C[i] > 0.f ? C[i] : 0.f;
    }
    return {C_out, 1, 1};
}

// ── GemmFp32 ─────────────────────────────────────────────────

void GemmFp32::load_weights(const ReadTensorFn& read)
{
    const std::string base = name;
    std::vector<int32_t> shape;
    std::vector<uint8_t> data;

    if (read(base + ".weight", shape, data)) tensor_values(data, weight_fp32, base + ".weight");
    if (read(base + ".bias", shape, data)) tensor_values(data, bias_fp32, base + ".bias");
    need_values(weight_fp32, layer_elems(name, {C_out, C_in}), base + ".weight");
    allow_values(bias_fp32, (size_t)C_out, base + ".bias");

    // Pack weights for SGEMM
    auto* _be = active_backend();
    float* packed = _be->pack_weights_fp32(weight_fp32.data(), C_out, C_in);
    size_t nb = (size_t)((C_out+3)/4) * ((C_in+3)/4) * 16 * sizeof(float);
    w_packed_f32 = Buffer(packed, nb, MemSpace::Host, [_be](void* p){ _be->free_packed_fp32(static_cast<float*>(p)); });
    if (_be->mem_space == MemSpace::Host) {
        std::vector<float>().swap(weight_fp32);
    }
}

void GemmFp32::upload_weights(Allocator& alloc)
{
    if (alloc.mem_space() == MemSpace::Host || !w_packed_f32.valid()) return;
    Buffer d = alloc.make_buffer(w_packed_f32.bytes());
    alloc.copy_h2d(d.ptr(), w_packed_f32.ptr(), w_packed_f32.bytes());
    w_packed_f32 = std::move(d);
}

Shape GemmFp32::forward(const TensorView* inputs, int n_inputs,
                            BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape in_shape = inputs[0].shape;
    (void)in_shape; (void)n_inputs;
    const float* in_ptr  = in1.as<float>();
    float*       out_ptr = out.as<float>();
    active_backend()->gemm_fp32_vec(in_ptr, weight_fp32.data(),
                                     bias_fp32.empty() ? nullptr : bias_fp32.data(),
                                     out_ptr, C_in, C_out,
                                     w_packed_f32.as<float>(),
                                     sc.stream);
    if (relu)
        active_backend()->relu_fp32(out_ptr, C_out, sc.stream);
    return {C_out, 1, 1};
}
