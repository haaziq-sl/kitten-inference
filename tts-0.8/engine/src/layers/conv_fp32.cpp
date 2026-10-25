#include "conv_fp32.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"
#include <cstring>
#include <cstdio>
#include <vector>

void ConvFp32::load_weights(const ReadTensorFn& read)
{
    const std::string base = name;
    std::vector<int32_t> shape;
    std::vector<uint8_t> data;

    if (read(base + ".weight", shape, data)) {
        weight_fp32.resize(data.size() / 4);
        memcpy(weight_fp32.data(), data.data(), data.size());
    }
    if (read(base + ".bias", shape, data)) {
        bias_fp32.resize(data.size() / 4);
        memcpy(bias_fp32.data(), data.data(), data.size());
    }

    int C_in_g = C_in / (groups > 0 ? groups : 1);
    K = kH * kW * C_in_g;

    // FP32 Winograd F(2,3) weight pre-transform for 3x3 stride=1 pad=1 convs
    if (kH == 3 && kW == 3 && stride_h == 1 && stride_w == 1
        && pad_h == 1 && pad_w == 1 && groups == 1
        && dilation_h == 1 && dilation_w == 1)
    {
        std::vector<float> w_wino_tmp[16];
        for (int pos = 0; pos < 16; ++pos)
            w_wino_tmp[pos].assign((size_t)C_out * C_in_g, 0.f);

        for (int co = 0; co < C_out; ++co) {
            for (int ci = 0; ci < C_in_g; ++ci) {
                float g[3][3];
                for (int kh = 0; kh < 3; ++kh)
                    for (int kw_i = 0; kw_i < 3; ++kw_i)
                        g[kh][kw_i] = weight_fp32[(size_t)co*C_in_g*9 + ci*9 + kh*3 + kw_i];
                float Gg[4][3];
                for (int j = 0; j < 3; ++j) {
                    Gg[0][j] = g[0][j];
                    Gg[1][j] = 0.5f*(g[0][j]+g[1][j]+g[2][j]);
                    Gg[2][j] = 0.5f*(g[0][j]-g[1][j]+g[2][j]);
                    Gg[3][j] = g[2][j];
                }
                float GgGt[4][4];
                for (int ii = 0; ii < 4; ++ii) {
                    GgGt[ii][0] = Gg[ii][0];
                    GgGt[ii][1] = 0.5f*(Gg[ii][0]+Gg[ii][1]+Gg[ii][2]);
                    GgGt[ii][2] = 0.5f*(Gg[ii][0]-Gg[ii][1]+Gg[ii][2]);
                    GgGt[ii][3] = Gg[ii][2];
                }
                for (int pos = 0; pos < 16; ++pos)
                    w_wino_tmp[pos][(size_t)co*C_in_g + ci] = GgGt[pos/4][pos%4];
            }
        }
        for (int pos = 0; pos < 16; ++pos) {
            auto* _be = active_backend();
            float* pk = _be->pack_weights_fp32(w_wino_tmp[pos].data(), C_out, C_in_g);
            size_t nbw = (size_t)((C_out+3)/4) * ((C_in_g+3)/4) * 16 * sizeof(float);
            w_wino_f32_packed[pos] = Buffer(pk, nbw, MemSpace::Host,
                [_be](void* p){ _be->free_packed_fp32(static_cast<float*>(p)); });
        }
        wino_f32_available = true;
        // printf("  FP32 Winograd weights: %s (C_out=%d, C_in=%d)\n",
        //        name.c_str(), C_out, C_in_g);
    }

    {
        std::vector<float> w_nhwc((size_t)C_out * K);
        if (kH > 1 || kW > 1) {
            for (int co = 0; co < C_out; ++co)
                for (int kh = 0; kh < kH; ++kh)
                    for (int kw = 0; kw < kW; ++kw)
                        for (int ci = 0; ci < C_in_g; ++ci)
                            w_nhwc[co * K + (kh * kW + kw) * C_in_g + ci] =
                                weight_fp32[co * K + ci * kH * kW + kh * kW + kw];
        } else {
            w_nhwc = weight_fp32;
        }
        auto* _be = active_backend();
        float* packed = _be->pack_weights_fp32(w_nhwc.data(), C_out, K);
        size_t nb = (size_t)((C_out+3)/4) * ((K+3)/4) * 16 * sizeof(float);
        w_packed_f32 = Buffer(packed, nb, MemSpace::Host,
            [_be](void* p){ _be->free_packed_fp32(static_cast<float*>(p)); });
        // For CUDA: cuDNN expects NHWC filter layout, so replace NCHW weight_fp32
        // with the NHWC-converted copy (Conv1dFp32 does the same in its load_weights).
        if (_be->mem_space != MemSpace::Host)
            weight_fp32 = std::move(w_nhwc);
    }
}

void ConvFp32::upload_weights(Allocator& alloc)
{
    if (alloc.mem_space() == MemSpace::Host) return;
    auto upload = [&](Buffer& tb) {
        if (!tb.valid()) return;
        Buffer d = alloc.make_buffer(tb.bytes());
        alloc.copy_h2d(d.ptr(), tb.ptr(), tb.bytes());
        tb = std::move(d);
    };
    upload(w_packed_f32);
    for (int p = 0; p < 16; ++p) upload(w_wino_f32_packed[p]);
}

Shape ConvFp32::forward(const TensorView* inputs, int n_inputs,
                            BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const float* in_buf  = in1.as<float>();
    float*       out_buf = out.as<float>();
    int C = sh.d0, H = sh.d1, W = sh.d2;

    if (wino_f32_available && dilation_h == 1 && dilation_w == 1) {
        const float* ptrs[16];
        for (int p = 0; p < 16; ++p) ptrs[p] = w_wino_f32_packed[p].as<float>();
        active_backend()->conv2d_winograd_nhwc_fp32(
            in_buf, ptrs, bias_fp32.empty() ? nullptr : bias_fp32.data(),
            out_buf, C, H, W, C_out, relu, sc.buffers[Scratch::F32_A].as<float>(), sc.stream,
            weight_fp32.data());
    } else {
        active_backend()->conv2d_fp32_nhwc(
            in_buf, weight_fp32.data(), bias_fp32.empty() ? nullptr : bias_fp32.data(),
            out_buf, C, H, W, C_out, kH, kW,
            stride_h, stride_w, pad_h, pad_w, relu,
            w_packed_f32.as<float>(), sc.buffers[Scratch::F32_A].as<float>(),
            dilation_h, dilation_w, sc.stream);
    }
    return output_shape(sh);
}

void ConvFp32::scratch_needed(Shape in, size_t out[Scratch::N]) const
{
    if (in.d1 == 0 || in.d2 == 0 || in.d0 == 0) return;
    int oH = (in.d1 + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
    int oW = (in.d2 + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;
    size_t im2col = (size_t)oH * oW * kH * kW * in.d0 * sizeof(float);
    out[Scratch::F32_A] = std::max(out[Scratch::F32_A], im2col);
}
