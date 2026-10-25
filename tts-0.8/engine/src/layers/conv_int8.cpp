#include "conv_int8.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>
#include <stdexcept>

void ConvInt8::load_weights(const ReadTensorFn& read)
{
    const std::string base = name;
    std::vector<int32_t> shape;
    std::vector<uint8_t> data;

    if (read(base + ".weight", shape, data)) {
        weight.resize(data.size());
        memcpy(weight.data(), data.data(), data.size());
    }
    if (read(base + ".req_scale", shape, data)) {
        req_scale.resize(data.size() / 4);
        memcpy(req_scale.data(), data.data(), data.size());
    }
    if (read(base + ".eff_bias", shape, data)) {
        eff_bias.resize(data.size() / 8);
        memcpy(eff_bias.data(), data.data(), data.size());
    }

    int C_in_g = C_in / (groups > 0 ? groups : 1);
    K = C_in_g * kH * kW;

    // Pre-compute Q31 multipliers
    req_mult.resize(C_out);
    req_exp.resize(C_out);
    for (int c = 0; c < C_out; ++c)
        QuantizeMultiplier(req_scale[c], &req_mult[c], &req_exp[c]);

    // Pre-pack weights for SDOT (NCHW ordering)
    {
        auto* _be = active_backend();
        int8_t* pk = _be->pack_weights_int8(weight.data(), C_out, K);
        size_t nb = (size_t)((C_out+7)/8) * ((K+3)/4) * 32;
        w_packed = Buffer(pk, nb, MemSpace::Host, [_be](void* p){ _be->free_packed_int8(static_cast<int8_t*>(p)); });
    }

    // Weight row sums (AVX-512 VNNI correction)
    w_row_sums.resize(C_out);
    for (int nn = 0; nn < C_out; ++nn) {
        int32_t s = 0;
        for (int k = 0; k < K; ++k) s += weight[(size_t)nn * K + k];
        w_row_sums[nn] = s;
    }

    // Pre-pack NHWC ordering (kH, kW, C_in_g)
    if (kH > 1 || kW > 1) {
        std::vector<int8_t> w_nhwc((size_t)C_out * K);
        for (int co = 0; co < C_out; ++co)
            for (int kh = 0; kh < kH; ++kh)
                for (int kw = 0; kw < kW; ++kw)
                    for (int c = 0; c < C_in_g; ++c)
                        w_nhwc[(size_t)co * K + kh * kW * C_in_g + kw * C_in_g + c] =
                            weight[(size_t)co * K + c * kH * kW + kh * kW + kw];
        auto* _be2 = active_backend();
        int8_t* pn = _be2->pack_weights_int8(w_nhwc.data(), C_out, K);
        size_t nb2 = (size_t)((C_out+7)/8) * ((K+3)/4) * 32;
        w_packed_nhwc = Buffer(pn, nb2, MemSpace::Host, [_be2](void* p){ _be2->free_packed_int8(static_cast<int8_t*>(p)); });
    } else {
        // Share the same data: make a non-owning view pointing to w_packed's memory
        // (no delete on w_packed_nhwc — w_packed owns it)
        w_packed_nhwc = Buffer(w_packed.ptr(), w_packed.bytes(), MemSpace::Host, [](void*){});
    }

    // Winograd F(2,3) weight pre-transform for 3x3 stride=1 pad=1 convs
    if (kH == 3 && kW == 3 && stride_h == 1 && stride_w == 1
        && pad_h == 1 && pad_w == 1 && groups == 1)
    {
        std::vector<float> w_hat_f((size_t)16 * C_out * C_in_g);
        for (int co = 0; co < C_out; ++co) {
            for (int ci = 0; ci < C_in_g; ++ci) {
                float g[3][3];
                for (int kh = 0; kh < 3; ++kh)
                    for (int kw = 0; kw < 3; ++kw)
                        g[kh][kw] = (float)weight[(size_t)co * C_in_g * 9 + ci * 9 + kh * 3 + kw];
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
                for (int pos = 0; pos < 16; ++pos) {
                    int pi = pos/4, pj = pos%4;
                    w_hat_f[(size_t)pos * C_out * C_in_g + co * C_in_g + ci] = GgGt[pi][pj];
                }
            }
        }
        wino_input_scale = 0.25f;
        for (int pos = 0; pos < 16; ++pos) {
            const float* wf = w_hat_f.data() + (size_t)pos * C_out * C_in_g;
            int Nw = C_out * C_in_g;
            float max_abs = 0.0f;
            for (int k = 0; k < Nw; ++k) { float a = std::fabs(wf[k]); if (a > max_abs) max_abs = a; }
            float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            wino_weight_scale[pos] = scale;
            std::vector<int8_t> w_q(Nw);
            for (int k = 0; k < Nw; ++k) {
                float q = std::roundf(wf[k] / scale);
                if (q > 127.0f) q = 127.0f; if (q < -128.0f) q = -128.0f;
                w_q[k] = (int8_t)q;
            }
            auto* _be3 = active_backend();
            int8_t* pk = _be3->pack_weights_int8(w_q.data(), C_out, C_in_g);
            size_t nbw;
            if (_be3->mem_space != MemSpace::Host) {
                // CUDA backend: dense [N_pad, K_pad] layout
                size_t N_pad = (size_t)((C_out   + 3) / 4) * 4;
                size_t K_pad = (size_t)((C_in_g  + 3) / 4) * 4;
                nbw = N_pad * K_pad;
            } else {
#ifdef __ARM_FEATURE_MATMUL_INT8
                nbw = (size_t)((C_out+7)/8) * ((C_in_g+7)/8) * 64;  // SMMLA packing
#else
                nbw = (size_t)((C_out+7)/8) * ((C_in_g+3)/4) * 32;  // SDOT / x86 packing
#endif
            }
            w_wino_packed[pos] = Buffer(pk, nbw, MemSpace::Host, [_be3](void* p){ _be3->free_packed_int8(static_cast<int8_t*>(p)); });
            w_wino_row_sums[pos].resize(C_out);
            for (int nn = 0; nn < C_out; ++nn) {
                int32_t s = 0;
                for (int k = 0; k < C_in_g; ++k) s += w_q[(size_t)nn * C_in_g + k];
                w_wino_row_sums[pos][nn] = s;
            }
        }
        wino_available = true;

        // Compute wino_eff_bias correction for zero-point mismatch
        {
            int32_t cs5m, cs5e;
            QuantizeMultiplier(wino_weight_scale[5] / wino_input_scale, &cs5m, &cs5e);
            wino_eff_bias.resize(C_out);
            for (int co = 0; co < C_out; ++co) {
                int32_t wrs5 = w_wino_row_sums[5][co];
                int32_t wino_accum   = apply_q31_scalar((int32_t)in_zp * wrs5, cs5m, cs5e);
                int32_t direct_accum = (int32_t)in_zp * w_row_sums[co];
                wino_eff_bias[co] = eff_bias[co] + (int64_t)(direct_accum - wino_accum);
            }
        }
        // printf("  Winograd weights: %s (C_out=%d, C_in=%d)\n", name.c_str(), C_out, C_in_g);
    }

    // Pre-transpose weights to [9, C] layout for the native NHWC depthwise kernel.
    // Only for 3×3 depthwise (groups == C_in) on the host backend.
    // w_dw_hwc[t*C + c] = weight[c*9 + t] (tap-major order).
    if (groups == C_in && kH == 3 && kW == 3
        && (active_backend()->mem_space == MemSpace::Host
            || active_backend()->mem_space == MemSpace::Metal))
    {
        const int C = C_out;  // groups == C_in == C_out for depthwise
        w_dw_hwc.resize(9 * C);
        for (int c = 0; c < C; ++c)
            for (int t = 0; t < 9; ++t)
                w_dw_hwc[t * C + c] = weight[c * 9 + t];
        dw_eff_bias32.resize(C);
        for (int c = 0; c < C; ++c)
            dw_eff_bias32[c] = (int32_t)eff_bias[c];
    }
}

void ConvInt8::upload_weights(Allocator& alloc)
{
    if (alloc.mem_space() == MemSpace::Host) return;
    auto upload = [&](Buffer& tb) {
        if (!tb.valid()) return;
        Buffer d = alloc.make_buffer(tb.bytes());
        alloc.copy_h2d(d.ptr(), tb.ptr(), tb.bytes());
        tb = std::move(d);
    };
    // Upload w_packed_nhwc BEFORE w_packed: for 1×1 conv, w_packed_nhwc is a
    // non-owning view into w_packed's host buffer, so w_packed must not be
    // freed (uploaded) until after w_packed_nhwc has been copied to the device.
    upload(w_packed_nhwc);
    upload(w_packed);
    for (int p = 0; p < 16; ++p) upload(w_wino_packed[p]);
}

Shape ConvInt8::forward(const TensorView* inputs, int n_inputs,
                            BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const int8_t* in_buf  = in1.as<int8_t>();
    int8_t*       out_buf = out.as<int8_t>();
    int C = sh.d0, H = sh.d1, W = sh.d2;

    // Winograd for eligible 3x3 stride=1 pad=1 convs (CPU/Host only).
    // Metal and CUDA both skip this: Metal cannot correctly read/write
    // Metal-allocated activation buffers via CPU Winograd; CUDA uses cuDNN.
    if (wino_available && kH == 3 && kW == 3
        && stride_h == 1 && stride_w == 1
        && pad_h == 1 && pad_w == 1 && groups == 1
        && active_backend()->mem_space == MemSpace::Host)
    {
        const int8_t*  wino_ptrs[16];
        const int32_t* wino_rs[16];
        for (int p = 0; p < 16; ++p) {
            wino_ptrs[p] = w_wino_packed[p].as<int8_t>();
            wino_rs[p]   = w_wino_row_sums[p].empty() ? nullptr : w_wino_row_sums[p].data();
        }
        active_backend()->conv2d_winograd_nhwc_int8(
            in_buf, wino_ptrs,
            wino_weight_scale, wino_input_scale,
            wino_eff_bias.data(), req_mult.data(), req_exp.data(),
            req_scale.data(),
            in_zp, out_zp,
            out_buf,
            C, H, W, C_out,
            sc.buffers[Scratch::I8_A].as<int8_t>(), sc.buffers[Scratch::I32_A].as<int32_t>(),
            wino_rs, sc.stream);
        return output_shape(sh);
    }

    if (groups > 1) {
        const int oH = (H + 2*pad_h - kH) / stride_h + 1;
        const int oW = (W + 2*pad_w - kW) / stride_w + 1;

        if (active_backend()->mem_space == MemSpace::CUDA
            || active_backend()->mem_space == MemSpace::HIP
            || active_backend()->mem_space == MemSpace::Metal) {
            // GPU backends: pass NHWC buffers to backend->conv2d_int8.
            // The backend is responsible for flushing any pending GPU work
            // (Metal flushes before the CPU fallback; CUDA/HIP run on GPU).
            active_backend()->conv2d_int8(
                in_buf, weight.data(), w_packed_nhwc.as<int8_t>(),
                eff_bias.data(), req_mult.data(), req_exp.data(),
                in_zp, out_zp, out_buf,
                C, H, W, C_out, kH, kW,
                stride_h, stride_w, pad_h, pad_w, groups,
                sc.buffers[Scratch::I8_A].as<int8_t>(), /*nhwc=*/true,
                w_row_sums.empty() ? nullptr : w_row_sums.data(), sc.stream);
            return {C_out, oH, oW};
        }

        // Native NHWC depthwise kernel: avoids NHWC↔NCHW transpositions entirely.
        // Declared in ops_neon.hpp, implemented in src/backend/cpu/ops/conv.cpp.
        if (!w_dw_hwc.empty() && kH == 3 && kW == 3
            && pad_h == 1 && pad_w == 1 && (stride_h == 1 || stride_h == 2)
            && stride_w == stride_h)
        {
            conv2d_depthwise_nhwc_int8(in_buf, out_buf, H, W, C, oH, oW, stride_h,
                                       in_zp, w_dw_hwc.data(), dw_eff_bias32.data(),
                                       req_mult.data(), req_exp.data(), out_zp,
                                       /*nthreads=*/0, sc.stream);
            return {C_out, oH, oW};
        }

        // Fallback: NHWC→NCHW→conv2d_int8→NCHW→NHWC (ops owns the scratch buffers).
        conv2d_grouped_nhwc_fallback_int8(
            in_buf, weight.data(),
            eff_bias.data(), req_mult.data(), req_exp.data(),
            in_zp, out_zp, out_buf,
            C, H, W, C_out, kH, kW,
            stride_h, stride_w, pad_h, pad_w, groups,
            sc.buffers[Scratch::I8_A].as<int8_t>(),
            w_row_sums.empty() ? nullptr : w_row_sums.data(), sc.stream);
        return {C_out, oH, oW};
    }

    active_backend()->conv2d_int8(
        in_buf, weight.data(), w_packed_nhwc.as<int8_t>(),
        eff_bias.data(), req_mult.data(), req_exp.data(),
        in_zp, out_zp,
        out_buf,
        C, H, W,
        C_out, kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        groups,
        sc.buffers[Scratch::I8_A].as<int8_t>(),
        /*nhwc=*/true,
        w_row_sums.empty() ? nullptr : w_row_sums.data(), sc.stream);
    return output_shape(sh);
}

void ConvInt8::scratch_needed(Shape in, size_t out[Scratch::N]) const
{
    if (in.d1 == 0 || in.d2 == 0 || in.d0 == 0) return;
    const int oH = (in.d1 + 2*pad_h - kH) / stride_h + 1;
    const int oW = (in.d2 + 2*pad_w - kW) / stride_w + 1;

    // im2col scratch (INT8): oH * oW * kH * kW * C_in bytes
    size_t im2col = (size_t)oH * oW * kH * kW * in.d0;

    // Winograd scratch (eligible 3x3 stride=1 pad=1 layers)
    if (wino_available && kH == 3 && kW == 3
        && stride_h == 1 && stride_w == 1 && pad_h == 1 && pad_w == 1 && groups == 1)
    {
        int n_tiles_h = in.d1 / 2 + 1;
        int n_tiles_w = in.d2 / 2 + 1;
        size_t wino_in  = (size_t)16 * n_tiles_h * n_tiles_w * in.d0;
        size_t wino_out = (size_t)16 * n_tiles_h * n_tiles_w * C_out * sizeof(int32_t);
        out[Scratch::I8_A]  = std::max(out[Scratch::I8_A],  std::max(im2col, wino_in));
        out[Scratch::I32_A] = std::max(out[Scratch::I32_A], wino_out);
    } else {
        out[Scratch::I8_A] = std::max(out[Scratch::I8_A], im2col);
    }
}
