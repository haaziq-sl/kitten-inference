// src/backend/metal/shaders/kernels.metal
// All Metal compute kernels for the Metal backend.
#include <metal_stdlib>
using namespace metal;

// ── Activations (elementwise) ─────────────────────────────────────
kernel void relu_inplace_k(
    device float* d [[buffer(0)]],
    uint id [[thread_position_in_grid]])
{
    d[id] = max(d[id], 0.f);
}

kernel void gelu_inplace_k(
    device float* d [[buffer(0)]],
    uint id [[thread_position_in_grid]])
{
    float x = d[id];
    // Clamp tanh argument to ±44.0 to prevent FP32 exp overflow (exp(2*44) < FP32_MAX).
    // Metal's tanh(x) = (exp(2x)-1)/(exp(2x)+1); for |x| > ~44 exp overflows to inf → NaN.
    float inner = clamp(0.7978845608f * fma(0.044715f, x * x * x, x), -44.0f, 44.0f);
    d[id] = 0.5f * x * (1.0f + tanh(inner));
}

kernel void sigmoid_k(
    device const float* in  [[buffer(0)]],
    device       float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = 1.f / (1.f + exp(-in[id]));
}

kernel void tanh_k(
    device const float* in  [[buffer(0)]],
    device       float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = tanh(in[id]);
}

kernel void leaky_relu_k(
    device const float* in    [[buffer(0)]],
    device       float* out   [[buffer(1)]],
    constant     float& alpha [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = in[id] >= 0.f ? in[id] : alpha * in[id];
}

kernel void exp_k(
    device const float* in  [[buffer(0)]],
    device       float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = exp(in[id]);
}

kernel void sin_k(
    device const float* in  [[buffer(0)]],
    device       float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = sin(in[id]);
}

// ── LM activation ops ─────────────────────────────────────────────
kernel void silu_k(
    device const float* in  [[buffer(0)]],
    device       float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = in[id];
    out[id] = x / (1.0f + exp(-x));
}

kernel void elemwise_mul_k(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device       float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = a[id] * b[id];
}

// ── Vector ops ────────────────────────────────────────────────────
kernel void add_fp32_k(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device       float* c      [[buffer(2)]],
    constant     int&   do_relu [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    float v = a[id] + b[id];
    c[id] = do_relu ? max(v, 0.f) : v;
}

kernel void add_vectors_k(
    device       float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] += src[id];
}

kernel void scale_k(
    device const float* in  [[buffer(0)]],
    device       float* out [[buffer(1)]],
    constant     float& s   [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = in[id] * s;
}

kernel void bias_add_flat_k(
    device       float* out  [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant     int&   C    [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] += bias[id % uint(C)];
}

kernel void sum_channels_k(
    device const float* in  [[buffer(0)]],
    device       float* out [[buffer(1)]],
    constant     int&   C   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const device float* r = in + uint64_t(gid) * C;
    float s = 0.f;
    for (int c = 0; c < C; ++c) s += r[c];
    out[gid] = s;
}

kernel void add_requant_int8_k(
    device const int8_t* a  [[buffer(0)]],
    device const int8_t* b  [[buffer(1)]],
    device       int8_t* c  [[buffer(2)]],
    constant     int4&   q  [[buffer(3)]],   // {M1, M2, pre_bias, _pad}  Q12 params
    uint id [[thread_position_in_grid]])
{
    // Q12 fixed-point matching CPU ARM NEON vrshrn_n_s32(acc, 12):
    //   acc = a*M1 + b*M2 + pre_bias
    //   result = (acc + 2048) >> 12  (arithmetic right shift = round-half-up, same as VRSHRN)
    int acc = int(a[id]) * q.x + int(b[id]) * q.y + q.z;
    c[id] = int8_t(clamp((acc + 2048) >> 12, -128, 127));
}

// ── Pooling: max pool FP32 NHWC ───────────────────────────────────
// dims   = [H, W, C, kH]
// params = [kW, padH, padW, sH]
// oparams= [sW, oH, oW, 0]
kernel void maxpool_fp32_nhwc_k(
    device const float* in      [[buffer(0)]],
    device       float* out     [[buffer(1)]],
    constant     int4&  dims    [[buffer(2)]],
    constant     int4&  params  [[buffer(3)]],
    constant     int4&  oparams [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int H = dims.x, W = dims.y, C = dims.z, kH = dims.w;
    int kW = params.x, pH = params.y, pW = params.z, sH = params.w;
    int sW = oparams.x, oH = oparams.y, oW = oparams.z;

    int c  = int(gid) % C;
    int tmp= int(gid) / C;
    int ow = tmp % oW;
    int oh = tmp / oW;
    if (oh >= oH) return;

    float mx = -INFINITY;
    for (int kh = 0; kh < kH; ++kh) {
        int ih = oh * sH + kh - pH;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kW; ++kw) {
            int iw = ow * sW + kw - pW;
            if (iw < 0 || iw >= W) continue;
            mx = max(mx, in[(ih * W + iw) * C + c]);
        }
    }
    out[(oh * oW + ow) * C + c] = mx;
}

// ── Max pool INT8 NHWC ────────────────────────────────────────────
kernel void maxpool_int8_nhwc_k(
    device const int8_t* in      [[buffer(0)]],
    device       int8_t* out     [[buffer(1)]],
    constant     int4&   dims    [[buffer(2)]],
    constant     int4&   params  [[buffer(3)]],
    constant     int4&   oparams [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int H = dims.x, W = dims.y, C = dims.z, kH = dims.w;
    int kW = params.x, pH = params.y, pW = params.z, sH = params.w;
    int sW = oparams.x, oH = oparams.y, oW = oparams.z;

    int c  = int(gid) % C;
    int tmp= int(gid) / C;
    int ow = tmp % oW;
    int oh = tmp / oW;
    if (oh >= oH) return;

    int mx = -128;
    for (int kh = 0; kh < kH; ++kh) {
        int ih = oh * sH + kh - pH;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kW; ++kw) {
            int iw = ow * sW + kw - pW;
            if (iw < 0 || iw >= W) continue;
            mx = max(mx, (int)in[(ih * W + iw) * C + c]);
        }
    }
    out[(oh * oW + ow) * C + c] = int8_t(mx);
}

// ── Max pool INT8 NCHW ────────────────────────────────────────────
kernel void maxpool_int8_nchw_k(
    device const int8_t* in      [[buffer(0)]],
    device       int8_t* out     [[buffer(1)]],
    constant     int4&   dims    [[buffer(2)]],
    constant     int4&   params  [[buffer(3)]],
    constant     int4&   oparams [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int H = dims.x, W = dims.y, C = dims.z, kH = dims.w;
    int kW = params.x, pH = params.y, pW = params.z, sH = params.w;
    int sW = oparams.x, oH = oparams.y, oW = oparams.z;

    int ow = int(gid) % oW;
    int tmp= int(gid) / oW;
    int oh = tmp % oH;
    int c  = tmp / oH;
    if (c >= C) return;

    int mx = -128;
    for (int kh = 0; kh < kH; ++kh) {
        int ih = oh * sH + kh - pH;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kW; ++kw) {
            int iw = ow * sW + kw - pW;
            if (iw < 0 || iw >= W) continue;
            mx = max(mx, (int)in[c * H * W + ih * W + iw]);
        }
    }
    out[c * oH * oW + oh * oW + ow] = int8_t(mx);
}

// ── Avg pool global FP32 NHWC ─────────────────────────────────────
kernel void avgpool_fp32_nhwc_k(
    device const float* in   [[buffer(0)]],
    device       float* out  [[buffer(1)]],
    constant     int3&  dims [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int H = dims.x, W = dims.y, C = dims.z;
    if ((int)gid >= C) return;
    float s = 0.f;
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            s += in[(h * W + w) * C + gid];
    out[gid] = s / float(H * W);
}

// ── Avg pool global INT8 NHWC ─────────────────────────────────────
kernel void avgpool_int8_nhwc_k(
    device const int8_t* in   [[buffer(0)]],
    device       int8_t* out  [[buffer(1)]],
    constant     int3&   dims [[buffer(2)]],
    constant     float4& q    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int H = dims.x, W = dims.y, C = dims.z;
    if ((int)gid >= C) return;
    float s = 0.f;
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            s += (float(in[(h * W + w) * C + gid]) - q.y) * q.x;
    out[gid] = int8_t(clamp(round(s / float(H * W) / q.z) + q.w, -128.f, 127.f));
}

// ── Avg pool global INT8 NCHW ─────────────────────────────────────
kernel void avgpool_int8_nchw_k(
    device const int8_t* in   [[buffer(0)]],
    device       int8_t* out  [[buffer(1)]],
    constant     int3&   dims [[buffer(2)]],
    constant     float4& q    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int H = dims.x, W = dims.y;
    float s = 0.f;
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            s += (float(in[gid * H * W + h * W + w]) - q.y) * q.x;
    out[gid] = int8_t(clamp(round(s / float(H * W) / q.z) + q.w, -128.f, 127.f));
}

// ── Layer norm ────────────────────────────────────────────────────
kernel void layernorm_k(
    device const float* in    [[buffer(0)]],
    device       float* out   [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta  [[buffer(3)]],
    constant     int&   C     [[buffer(4)]],
    constant     float& eps   [[buffer(5)]],
    threadgroup  float* tg    [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    const uint TGS = 256;
    const device float* x = in  + uint64_t(row) * C;
    device       float* y = out + uint64_t(row) * C;

    float ls = 0.f;
    for (uint i = lid; i < (uint)C; i += TGS) ls += x[i];
    tg[lid] = ls;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TGS / 2; s > 0; s >>= 1) {
        if (lid < s) tg[lid] += tg[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = tg[0] / float(C);

    float lv = 0.f;
    for (uint i = lid; i < (uint)C; i += TGS) {
        float d = x[i] - mean;
        lv += d * d;
    }
    tg[lid] = lv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TGS / 2; s > 0; s >>= 1) {
        if (lid < s) tg[lid] += tg[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(tg[0] / float(C) + eps);

    for (uint i = lid; i < (uint)C; i += TGS)
        y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
}

// ── Upsample ──────────────────────────────────────────────────────
kernel void upsample_nearest1d_k(
    device const float* in  [[buffer(0)]],
    device       float* out [[buffer(1)]],
    constant     int3&  p   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int C = p.y, sf = p.z;
    int c  = int(gid) % C;
    int ot = int(gid) / C;
    out[gid] = in[(ot / sf) * C + c];
}

// in_dims = [C, iH, iW, oH]; oW is separate
kernel void upsample_bilinear2d_k(
    device const float* in      [[buffer(0)]],
    device       float* out     [[buffer(1)]],
    constant     int4&  in_dims [[buffer(2)]],
    constant     int&   oW      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int C  = in_dims.x, iH = in_dims.y, iW = in_dims.z, oH = in_dims.w;
    int c  = int(gid) % C;
    int tmp= int(gid) / C;
    int ow = tmp % oW;
    int oh = tmp / oW;
    if (oh >= oH) return;

    float fy = (float(oh) + 0.5f) * float(iH) / float(oH) - 0.5f;
    float fx = (float(ow) + 0.5f) * float(iW) / float(oW) - 0.5f;

    int y0_raw = (int)floor(fy);
    int x0_raw = (int)floor(fx);
    float wy = fy - float(y0_raw);
    float wx = fx - float(x0_raw);
    int y0 = clamp(y0_raw,     0, iH - 1);
    int y1 = clamp(y0_raw + 1, 0, iH - 1);
    int x0 = clamp(x0_raw,     0, iW - 1);
    int x1 = clamp(x0_raw + 1, 0, iW - 1);

    out[gid] = (1.f - wy) * (1.f - wx) * in[(y0 * iW + x0) * C + c]
             + (1.f - wy) *        wx  * in[(y0 * iW + x1) * C + c]
             +        wy  * (1.f - wx) * in[(y1 * iW + x0) * C + c]
             +        wy  *        wx  * in[(y1 * iW + x1) * C + c];
}

// ── Tensor utilities ──────────────────────────────────────────────
kernel void concat1d_k(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device       float* out [[buffer(2)]],
    constant     int4&  p   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int T = p.x, C1 = p.y, C2 = p.z, bcast = p.w;
    int Cout = C1 + C2;
    int c = int(gid) % Cout;
    int t = int(gid) / Cout;
    if (t >= T) return;
    out[gid] = (c < C1) ? a[t * C1 + c]
                         : b[(bcast ? 0 : t) * C2 + (c - C1)];
}

kernel void slice_channels_k(
    device const float* in  [[buffer(0)]],
    device       float* out [[buffer(1)]],
    constant     int4&  p   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int Cin = p.y, cs = p.z, Cout = p.w;
    int c = int(gid) % Cout;
    int t = int(gid) / Cout;
    out[gid] = in[t * Cin + cs + c];
}

kernel void embedding_lookup_k(
    device const float* ids [[buffer(0)]],
    device const float* w   [[buffer(1)]],
    device       float* out [[buffer(2)]],
    constant     int2&  p   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int D = p.y;
    int d = int(gid) % D;
    int t = int(gid) / D;
    out[gid] = w[int(ids[t]) * D + d];
}

kernel void snake1d_k(
    device const float* in        [[buffer(0)]],
    device       float* out       [[buffer(1)]],
    device const float* alpha     [[buffer(2)]],
    device const float* inv_alpha [[buffer(3)]],
    constant     int&   C         [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int   c  = int(gid) % C;
    float x  = in[gid];
    float a  = alpha[c];
    float ia = inv_alpha[c];
    float s  = sin(a * x);
    out[gid] = x + s * s * ia;
}

// After seqgemm: add bias row-wise. TC = [T, C_out]
kernel void seqgemm_bias_k(
    device       float* out  [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant     int2&  TC   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] += bias[gid % uint(TC.y)];
}

// ── INT8 GEMM: tiled 16×16, INT8 → INT8 with Q31 requantization ──
// A: [M, K] int8, B: [N, K] int8 row-major
// Output C: [M, N] int8
// dims: {M, K, N, out_zp}
// Requantization: Q31 fixed-point: result = round(acc * mult >> (31-exp)) + out_zp
constant int GEMM_TILE = 16;

kernel void gemm_int8_requant_k(
    device const char*  A     [[buffer(0)]],
    device const char*  B     [[buffer(1)]],
    device const long*  ebias [[buffer(2)]],
    device const int*   mult  [[buffer(3)]],
    device const int*   exp_  [[buffer(4)]],
    device       char*  C     [[buffer(5)]],
    constant     int4&  dims  [[buffer(6)]],
    threadgroup  char*  tgmem [[threadgroup(0)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    int M = dims.x, K = dims.y, N = dims.z;
    int m = int(tg.y) * GEMM_TILE + int(tid.y);
    int n = int(tg.x) * GEMM_TILE + int(tid.x);

    threadgroup char* tA = tgmem;                     // [TILE, TILE] A tile
    threadgroup char* tB = tgmem + GEMM_TILE * GEMM_TILE; // [TILE, TILE] B tile

    int acc = 0;
    for (int k0 = 0; k0 < K; k0 += GEMM_TILE) {
        // Thread (tx, ty) loads A[tg.y*T+ty, k0+tx] into tA[ty][tx]
        int a_k = k0 + int(tid.x);
        tA[int(tid.y) * GEMM_TILE + int(tid.x)] = (m < M && a_k < K) ? A[m * K + a_k] : char(0);

        // Thread (tx, ty) loads B[tg.x*T+tx, k0+ty] into tB[ty][tx]
        int b_n = int(tg.x) * GEMM_TILE + int(tid.x);
        int b_k = k0 + int(tid.y);
        tB[int(tid.y) * GEMM_TILE + int(tid.x)] = (b_n < N && b_k < K) ? B[b_n * K + b_k] : char(0);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = 0; t < GEMM_TILE; ++t)
            acc += int(tA[int(tid.y) * GEMM_TILE + t]) * int(tB[t * GEMM_TILE + int(tid.x)]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (m >= M || n >= N) return;

    // Q31 fixed-point requantization: match CPU ARM NEON which truncates eff_bias
    // to int32 before adding to acc (all NEON paths cast eff_bias[n] to int32_t).
    // int32 wrap-around on overflow matches CPU behaviour for resnet101-style layers.
    int  total    = int(long(acc) + ebias[n]);   // truncate to int32, matching CPU
    int  rshift   = 31 - exp_[n];
    long prod     = long(total) * long(mult[n]);
    long rounding = (rshift > 0) ? (long(1) << (rshift - 1)) : long(0);
    int  q        = int((prod + rounding) >> rshift) + dims.w;
    C[m * N + n] = char(clamp(q, -128, 127));
}

// ── INT8 GEMM: tiled 16×16, INT8 → float with per-channel scale ──
// Output C: [M, N] float
// dims: {M, K, N, 0}
kernel void gemm_int8_float_k(
    device const char*  A      [[buffer(0)]],
    device const char*  B      [[buffer(1)]],
    device const long*  ebias  [[buffer(2)]],
    device const float* rscale [[buffer(3)]],
    device       float* C      [[buffer(4)]],
    constant     int4&  dims   [[buffer(5)]],
    threadgroup  char*  tgmem  [[threadgroup(0)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    int M = dims.x, K = dims.y, N = dims.z;
    int m = int(tg.y) * GEMM_TILE + int(tid.y);
    int n = int(tg.x) * GEMM_TILE + int(tid.x);

    threadgroup char* tA = tgmem;
    threadgroup char* tB = tgmem + GEMM_TILE * GEMM_TILE;

    int acc = 0;
    for (int k0 = 0; k0 < K; k0 += GEMM_TILE) {
        int a_k = k0 + int(tid.x);
        tA[int(tid.y) * GEMM_TILE + int(tid.x)] = (m < M && a_k < K) ? A[m * K + a_k] : char(0);

        int b_n = int(tg.x) * GEMM_TILE + int(tid.x);
        int b_k = k0 + int(tid.y);
        tB[int(tid.y) * GEMM_TILE + int(tid.x)] = (b_n < N && b_k < K) ? B[b_n * K + b_k] : char(0);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = 0; t < GEMM_TILE; ++t)
            acc += int(tA[int(tid.y) * GEMM_TILE + t]) * int(tB[t * GEMM_TILE + int(tid.x)]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (m >= M || n >= N) return;
    C[m * N + n] = float(long(acc) + ebias[n]) * rscale[n];
}

// ── im2col: NHWC input → column matrix (FP32) ────────────────────
// in:  [H, W, C_in]  NHWC
// col: [oH*oW, kH*kW*C_in]  — one row per output position (NHWC kernel order)
// in_dims = {H, W, C_in, kH}
// kp      = {kW, padH, padW, strideH}
// op      = {strideW, oH, oW, dilH}
// dilW    = dilation_w
// One thread per output position (oh, ow).
kernel void im2col_fp32_nhwc_k(
    device const float* in      [[buffer(0)]],
    device       float* col     [[buffer(1)]],
    constant     int4&  in_dims [[buffer(2)]],
    constant     int4&  kp      [[buffer(3)]],
    constant     int4&  op      [[buffer(4)]],
    constant     int&   dilW    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int H = in_dims.x, W = in_dims.y, C = in_dims.z, kH = in_dims.w;
    int kW = kp.x, padH = kp.y, padW = kp.z, strideH = kp.w;
    int strideW = op.x, oH = op.y, oW = op.z, dilH = op.w;
    int oHW = oH * oW;
    if (int(gid) >= oHW) return;

    int ow  = int(gid) % oW;
    int oh  = int(gid) / oW;
    int K   = kH * kW * C;
    device float* row = col + int(gid) * K;

    int idx = 0;
    for (int kh = 0; kh < kH; ++kh) {
        int ih = oh * strideH + kh * dilH - padH;
        for (int kw = 0; kw < kW; ++kw) {
            int iw = ow * strideW + kw * dilW - padW;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                device const float* src = in + (ih * W + iw) * C;
                for (int c = 0; c < C; ++c) row[idx++] = src[c];
            } else {
                for (int c = 0; c < C; ++c) row[idx++] = 0.f;
            }
        }
    }
}

// ── im2col: NHWC input → column matrix (INT8) ────────────────────
// Padding fills with in_zp (zero point).
kernel void im2col_int8_nhwc_k(
    device const char* in      [[buffer(0)]],
    device       char* col     [[buffer(1)]],
    constant     int4& in_dims [[buffer(2)]],
    constant     int4& kp      [[buffer(3)]],
    constant     int4& op      [[buffer(4)]],
    constant     int&  dilW    [[buffer(5)]],
    constant     int&  in_zp   [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int H = in_dims.x, W = in_dims.y, C = in_dims.z, kH = in_dims.w;
    int kW = kp.x, padH = kp.y, padW = kp.z, strideH = kp.w;
    int strideW = op.x, oH = op.y, oW = op.z, dilH = op.w;
    int oHW = oH * oW;
    if (int(gid) >= oHW) return;

    int ow  = int(gid) % oW;
    int oh  = int(gid) / oW;
    int K   = kH * kW * C;
    device char* row = col + int(gid) * K;
    char zp = char(in_zp);

    int idx = 0;
    for (int kh = 0; kh < kH; ++kh) {
        int ih = oh * strideH + kh * dilH - padH;
        for (int kw = 0; kw < kW; ++kw) {
            int iw = ow * strideW + kw * dilW - padW;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                device const char* src = in + (ih * W + iw) * C;
                for (int c = 0; c < C; ++c) row[idx++] = src[c];
            } else {
                for (int c = 0; c < C; ++c) row[idx++] = zp;
            }
        }
    }
}

// ── Scatter-add for conv_transpose1d (GPU path) ───────────────────
// col[T, C_out] → output[T_out, C_out]
// Each thread handles one (t, c) pair; adds col[t,c] to
// output[t*stride + k_pos - padding, c] if in bounds.
// dims: {T, C_out, T_out, stride}
// kp:   {k_pos, padding}
kernel void col2im_scatter_k(
    device const float* col    [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     int4&  dims   [[buffer(2)]],
    constant     int2&  kp     [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    int C_out = dims.y;
    int c   = int(id) % C_out;
    int t   = int(id) / C_out;
    if (t >= dims.x) return;
    int t_out = t * dims.w + kp.x - kp.y;
    if (t_out < 0 || t_out >= dims.z) return;
    output[t_out * C_out + c] += col[t * C_out + c];
}

// ── Depthwise transposed 1D convolution ──────────────────────────
// Each thread computes one output element: output[t_out, c].
// weight: [C, kernel_size] row-major  (depthwise: groups == C_in == C_out)
// dims:   {T_in, C, T_out, kernel_size}
// params: {stride, padding}
kernel void conv_transpose1d_dw_k(
    device const float* input   [[buffer(0)]],  // [T_in, C]
    device const float* weight  [[buffer(1)]],  // [C, kernel_size]
    device       float* output  [[buffer(2)]],  // [T_out, C]
    constant     int4&  dims    [[buffer(3)]],  // T_in, C, T_out, kernel_size
    constant     int2&  params  [[buffer(4)]],  // stride, padding
    uint2 gid [[thread_position_in_grid]])       // gid.x = c, gid.y = t_out
{
    int C     = dims.y;
    int c     = (int)gid.x;
    int t_out = (int)gid.y;
    if (c >= C || t_out >= dims.z) return;

    int T      = dims.x;
    int ksz    = dims.w;
    int stride = params.x;
    int pad    = params.y;

    device const float* w = weight + c * ksz;
    float acc = 0.f;
    for (int k = 0; k < ksz; ++k) {
        int num = t_out + pad - k;
        if (num < 0 || num % stride != 0) continue;
        int t = num / stride;
        if (t >= T) continue;
        acc += input[t * C + c] * w[k];
    }
    output[t_out * C + c] = acc;
}

// ── Direct 1D Convolution (no im2col) ────────────────────────────
// Each thread computes one output element: output[t_out, c_out].
// weight: [C_out, kernel_size, C_in] row-major (NHWC with W=1).
// dims: {T_in, C_in, T_out, C_out}
// kp:   {kernel_size, stride, padding, dilation}
// flags: {has_bias, do_relu}
kernel void conv1d_direct_k(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device       float* output  [[buffer(3)]],
    constant     int4&  dims    [[buffer(4)]],
    constant     int4&  kp      [[buffer(5)]],
    constant     int2&  flags   [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    int c_out = (int)gid.x;
    int t_out = (int)gid.y;
    int T_in  = dims.x, C_in  = dims.y;
    int T_out = dims.z, C_out = dims.w;
    if (c_out >= C_out || t_out >= T_out) return;

    int ksz = kp.x, stride = kp.y, pad = kp.z, dil = kp.w;

    float acc = 0.f;
    device const float* w_base = weight + c_out * ksz * C_in;

    for (int k = 0; k < ksz; ++k) {
        int t_in = t_out * stride + k * dil - pad;
        if (t_in >= 0 && t_in < T_in) {
            device const float* in_ptr = input + t_in * C_in;
            device const float* w_ptr  = w_base + k * C_in;
            int ci = 0;
            for (; ci + 4 <= C_in; ci += 4) {
                float4 iv = float4(in_ptr[ci], in_ptr[ci+1], in_ptr[ci+2], in_ptr[ci+3]);
                float4 wv = float4(w_ptr[ci],  w_ptr[ci+1],  w_ptr[ci+2],  w_ptr[ci+3]);
                acc += dot(iv, wv);
            }
            for (; ci < C_in; ++ci)
                acc += in_ptr[ci] * w_ptr[ci];
        }
    }

    if (flags.x) acc += bias[c_out];
    if (flags.y && acc < 0.f) acc = 0.f;
    output[t_out * C_out + c_out] = acc;
}

// ── Bias add + optional ReLU for conv output [oHW, C_out] NHWC ───
// bias: [C_out], C_out = dims.y
// do_relu: 0 or 1
kernel void conv_bias_relu_k(
    device       float* out     [[buffer(0)]],
    device const float* bias    [[buffer(1)]],
    constant     int2&  dims    [[buffer(2)]],  // {oHW, C_out}
    constant     int&   do_relu [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int C_out = dims.y;
    int n = int(gid) % C_out;  // channel
    float v = out[gid] + bias[n];
    out[gid] = (do_relu && v < 0.f) ? 0.f : v;
}

// ── Softmax over rows of [N_rows, N_cols] matrix (in-place) ──────
// Each threadgroup handles one row.  Threadgroup memory: 256 floats.
// dims.x = N_cols (element count per row)
// dims.y = N_rows
// dims.z = row_stride in floats (0 → use N_cols; set to S_rb/sizeof(float) for MPS-padded buffers)
kernel void softmax_rows_k(
    device       float* data [[buffer(0)]],
    constant     int4&  dims [[buffer(1)]],
    threadgroup  float* tg   [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    const uint TGS = 256;
    int N      = dims.x;
    int stride = dims.z > 0 ? dims.z : N;
    if ((int)row >= dims.y) return;
    device float* rp = data + int(row) * stride;

    // Phase 1: parallel max
    float lmax = -INFINITY;
    for (uint i = lid; i < (uint)N; i += TGS)
        lmax = max(lmax, rp[i]);
    tg[lid] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TGS/2; s > 0; s >>= 1) {
        if (lid < s) tg[lid] = max(tg[lid], tg[lid+s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = tg[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);   // ensure all see row_max before tg[] reused

    // Phase 2: exp(x - max), accumulate sum
    float lsum = 0.f;
    for (uint i = lid; i < (uint)N; i += TGS) {
        float e = exp(rp[i] - row_max);
        rp[i] = e;
        lsum += e;
    }
    tg[lid] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TGS/2; s > 0; s >>= 1) {
        if (lid < s) tg[lid] += tg[lid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.f / tg[0];

    // Phase 3: normalize
    for (uint i = lid; i < (uint)N; i += TGS)
        rp[i] *= inv_sum;
}

// ── Adaptive Layer Norm 1D ────────────────────────────────────────
// Like layernorm_k but output = (1 + gamma[i]) * xnorm + beta[i].
// Used by TTS decoder blocks.
kernel void ada_layernorm_k(
    device const float* in    [[buffer(0)]],
    device       float* out   [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta  [[buffer(3)]],
    constant     int&   C     [[buffer(4)]],
    constant     float& eps   [[buffer(5)]],
    threadgroup  float* tg    [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    const uint TGS = 256;
    const device float* x = in  + uint64_t(row) * C;
    device       float* y = out + uint64_t(row) * C;

    float ls = 0.f;
    for (uint i = lid; i < (uint)C; i += TGS) ls += x[i];
    tg[lid] = ls;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TGS/2; s > 0; s >>= 1) {
        if (lid < s) tg[lid] += tg[lid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = tg[0] / float(C);

    float lv = 0.f;
    for (uint i = lid; i < (uint)C; i += TGS) {
        float d = x[i] - mean;
        lv += d * d;
    }
    tg[lid] = lv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TGS/2; s > 0; s >>= 1) {
        if (lid < s) tg[lid] += tg[lid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(tg[0] / float(C) + eps);

    for (uint i = lid; i < (uint)C; i += TGS) {
        float xn = (x[i] - mean) * inv_std;
        y[i] = (1.f + gamma[i]) * xn + beta[i];
    }
}

// ── Adaptive Instance Norm 1D ─────────────────────────────────────
// Input: feat[T, C].  One threadgroup per channel.
// For each channel c: compute mean/var over T, then apply
//   out[t,c] = (1 + gamma[c]) * (feat[t,c] - mean) * inv_std + beta[c]
// Threadgroup memory: 2 * 256 floats (sum buf + sumsq buf).
kernel void ada_in1d_k(
    device const float* feat  [[buffer(0)]],
    device       float* out   [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta  [[buffer(3)]],
    constant     int2&  dims  [[buffer(4)]],   // dims.x = T, dims.y = C
    constant     float& eps   [[buffer(5)]],
    threadgroup  float* tg    [[threadgroup(0)]],  // 256 floats (one pass at a time)
    uint ch  [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    const uint TGS = 256;
    int T = dims.x, C = dims.y;
    if ((int)ch >= C) return;

    // Two-pass variance (stable for large means — avoids E[x²]-mean² cancellation).
    // Pass 1: mean
    float lsum = 0.f;
    for (int t = (int)lid; t < T; t += (int)TGS)
        lsum += feat[t * C + int(ch)];
    tg[lid] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TGS/2; s > 0; s >>= 1) {
        if (lid < s) tg[lid] += tg[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = tg[0] / float(T);

    // Pass 2: centered variance
    float lsq = 0.f;
    for (int t = (int)lid; t < T; t += (int)TGS) {
        float d = feat[t * C + int(ch)] - mean;
        lsq += d * d;
    }
    tg[lid] = lsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TGS/2; s > 0; s >>= 1) {
        if (lid < s) tg[lid] += tg[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(tg[0] / float(T) + eps);
    float g = gamma[int(ch)];
    float b = beta[int(ch)];

    for (int t = (int)lid; t < T; t += (int)TGS) {
        float xn = (feat[t * C + int(ch)] - mean) * inv_std;
        out[t * C + int(ch)] = (1.f + g) * xn + b;
    }
}

// ── AdaIN 1D with inline norm_weight/norm_bias folding ───────────
// Same as ada_in1d_k but folds static InstanceNorm affine params.
// Correct formula: out = (1 + gamma) * (xraw*nw + nb) + beta
//   scale = (1 + gamma_raw[c]) * nw[c]
//   bias  = (1 + gamma_raw[c]) * nb[c] + beta_raw[c]
// This avoids reading GPU-computed gamma_beta on the CPU before the
// command buffer is committed.
kernel void ada_in1d_fold_k(
    device const float* feat       [[buffer(0)]],
    device       float* out        [[buffer(1)]],
    device const float* gamma_raw  [[buffer(2)]],
    device const float* beta_raw   [[buffer(3)]],
    device const float* nw         [[buffer(4)]],
    device const float* nb         [[buffer(5)]],
    constant     int2&  dims       [[buffer(6)]],
    constant     float& eps        [[buffer(7)]],
    threadgroup  float* tg         [[threadgroup(0)]],  // 256 floats
    uint ch  [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    const uint TGS = 256;
    int T = dims.x, C = dims.y;
    if ((int)ch >= C) return;

    // Two-pass variance (stable for large means — avoids E[x²]-mean² cancellation).
    // Pass 1: mean
    float lsum = 0.f;
    for (int t = (int)lid; t < T; t += (int)TGS)
        lsum += feat[t * C + int(ch)];
    tg[lid] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TGS/2; s > 0; s >>= 1) {
        if (lid < s) tg[lid] += tg[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = tg[0] / float(T);

    // Pass 2: centered variance
    float lsq = 0.f;
    for (int t = (int)lid; t < T; t += (int)TGS) {
        float d = feat[t * C + int(ch)] - mean;
        lsq += d * d;
    }
    tg[lid] = lsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TGS/2; s > 0; s >>= 1) {
        if (lid < s) tg[lid] += tg[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(tg[0] / float(T) + eps);

    // Correct fold: out = (1 + gamma) * (xraw*nw + nb) + beta
    //   scale = (1 + gamma) * nw,  bias = (1 + gamma) * nb + beta
    float scale = (1.f + gamma_raw[int(ch)]) * nw[int(ch)];
    float bias  = (1.f + gamma_raw[int(ch)]) * nb[int(ch)] + beta_raw[int(ch)];

    for (int t = (int)lid; t < T; t += (int)TGS) {
        float xn = (feat[t * C + int(ch)] - mean) * inv_std;
        out[t * C + int(ch)] = scale * xn + bias;
    }
}

// ── Quantize float → int8 with per-element scale ─────────────────
kernel void quantize_float_to_int8_k(
    device const float* in        [[buffer(0)]],
    device       char*  out       [[buffer(1)]],
    constant     float& inv_scale [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = char(clamp((int)round(in[id] * inv_scale), -128, 127));
}

// ── LSTM input projection: C[T, 4H] = A[T, C_in] × W[4H, C_in]^T + b[4H] ──
kernel void lstm_proj_k(
    device const float*  A      [[buffer(0)]],  // [T, C_in]
    device const float*  W      [[buffer(1)]],  // [4H, C_in]  w_ih
    device const float*  bias   [[buffer(2)]],  // [4H] nullable (buf not used if has_bias==0)
    device       float*  C      [[buffer(3)]],  // [T, 4H]
    constant     int4&   dims   [[buffer(4)]],  // {T, C_in, fourH, has_bias}
    uint2 gid [[thread_position_in_grid]])
{
    int t  = (int)gid.x, n = (int)gid.y;
    int T  = dims.x, M = dims.y, fourH = dims.z;
    if (t >= T || n >= fourH) return;
    float acc = dims.w ? bias[n] : 0.0f;
    device const float* a_row = A + t * M;
    device const float* w_row = W + n * M;
    for (int m = 0; m < M; m++)
        acc += a_row[m] * w_row[m];
    C[t * fourH + n] = acc;
}

// ── LSTM recurrence step, pass 1: compute pre-gate values ────────
// gates[i] = proj_row[i] + b_hh[i] + sum_j(w_hh[i,j] * h_in[j])
// Grid = 4H, threadgroup <= 256. Reads h_in from global memory.
// Avoids the 4H-thread single-threadgroup design that hits Metal's
// maxTotalThreadsPerThreadgroup limit when H=256 (4H=1024).
kernel void lstm_gemv_k(
    device const float*  proj_row [[buffer(0)]],  // [4H]
    device const float*  w_hh     [[buffer(1)]],  // [4H, H]
    device const float*  b_hh     [[buffer(2)]],  // [4H]
    device const float*  h_in     [[buffer(3)]],  // [H]
    device       float*  gates    [[buffer(4)]],  // [4H] output
    constant     int2&   dims     [[buffer(5)]],  // {H, has_b_hh}
    uint gid [[thread_position_in_grid]])
{
    int H     = dims.x;
    int has_b = dims.y;
    if ((int)gid >= 4 * H) return;
    int i = (int)gid;
    float acc = proj_row[i] + (has_b ? b_hh[i] : 0.0f);
    device const float* wrow = w_hh + i * H;
    for (int j = 0; j < H; j++)
        acc += wrow[j] * h_in[j];
    gates[i] = acc;
}

// ── LSTM recurrence step, pass 2: apply gates and update h/c ────
// Grid = H threads. Reads gates from global memory (written by lstm_gemv_k).
// Separate command encoder ensures sequential execution with pass 1.
kernel void lstm_update_k(
    device const float*  gates  [[buffer(0)]],  // [4H] from lstm_gemv_k
    device const float*  c_in   [[buffer(1)]],  // [H]
    device       float*  h_out  [[buffer(2)]],  // [H]
    device       float*  c_out  [[buffer(3)]],  // [H]
    device       float*  h_all  [[buffer(4)]],  // [T*H]
    constant     int2&   dims   [[buffer(5)]],  // {H, t}
    uint gid [[thread_position_in_grid]])
{
    int H = dims.x;
    int t = dims.y;
    if ((int)gid >= H) return;
    int j = (int)gid;
    float gi = 1.0f / (1.0f + exp(-gates[j]));
    float gf = 1.0f / (1.0f + exp(-gates[H + j]));
    // Clamp tanh argument to ±40 to prevent Metal tanh overflow.
    // Metal computes tanh(x) = (exp(2x)-1)/(exp(2x)+1); for |x| > 44 exp(2x)
    // overflows to inf → inf/inf = NaN. exp(2*40)=exp(80) << FLT_MAX, so ±40
    // is safe. tanh saturates to ±1.0 in FP32 well before ±40.
    float gg = tanh(clamp(gates[2 * H + j], -40.0f, 40.0f));
    float go = 1.0f / (1.0f + exp(-gates[3 * H + j]));
    float c_new = gf * c_in[j] + gi * gg;
    // Clamp cell state to ±40: prevents tanh(c_new) overflow (exp(2*40) is
    // well below FLT_MAX) and guards against FTZ denormal × inf = NaN.
    c_new = clamp(c_new, -40.0f, 40.0f);
    float h_new = go * tanh(c_new);
    h_out[j]         = h_new;
    c_out[j]         = c_new;
    h_all[t * H + j] = h_new;
}

// ── Interleave fwd+bwd hidden states for bidirectional LSTM ───────
// h_fwd[T, H], h_bwd[T, H] → out[T, 2H]
kernel void lstm_interleave_k(
    device const float*  h_fwd  [[buffer(0)]],
    device const float*  h_bwd  [[buffer(1)]],
    device       float*  out    [[buffer(2)]],
    constant     int2&   dims   [[buffer(3)]],  // {T, H}
    uint gid [[thread_position_in_grid]])
{
    int T = dims.x, H = dims.y;
    int total = T * 2 * H;
    if ((int)gid >= total) return;
    int t  = gid / (2 * H);
    int ch = gid % (2 * H);
    out[gid] = (ch < H) ? h_fwd[t * H + ch] : h_bwd[t * H + (ch - H)];
}

// ── STFT: parallel (frame, bin) computation ─────────────────────
// Audio is accessed with edge-clamp padding (no separate padded buffer).
// Dispatch: grid=(T_stft, K)
// Output layout: out[frame * 2*K + 0..K-1] = mag, out[..K..2K-1] = phase
kernel void stft_fp32_k(
    device const float*  audio  [[buffer(0)]],  // [T_audio]
    device const float*  w_real [[buffer(1)]],  // [K, n_fft]
    device const float*  w_imag [[buffer(2)]],  // [K, n_fft]
    device       float*  out    [[buffer(3)]],  // [T_stft, 2*K]
    constant     int4&   params [[buffer(4)]],  // {n_fft, hop_size, K, T_audio}
    uint2 gid [[thread_position_in_grid]])
{
    int frame    = (int)gid.x;
    int k        = (int)gid.y;
    int n_fft    = params.x, hop_size = params.y, K = params.z, T_audio = params.w;
    if (k >= K) return;

    int pad = n_fft / 2;
    device const float* wr = w_real + k * n_fft;
    device const float* wi = w_imag + k * n_fft;

    float re = 0.0f, im = 0.0f;
    for (int n = 0; n < n_fft; n++) {
        int audio_idx = frame * hop_size - pad + n;
        float s = audio[clamp(audio_idx, 0, T_audio - 1)];
        re += wr[n] * s;
        im += wi[n] * s;
    }

    int base = frame * 2 * K;
    out[base + k]     = sqrt(re * re + im * im);
    out[base + K + k] = atan2(im, re);
}

// ── ISTFT: each thread computes one output sample ────────────────
// No atomic needed: each output sample computed independently.
// out_len = (T_frames - 1) * hop_size
kernel void istft_sample_k(
    device const float*  mag    [[buffer(0)]],  // [T_frames, K]
    device const float*  phase  [[buffer(1)]],  // [T_frames, K]
    device       float*  out    [[buffer(2)]],  // [out_len]
    constant     int4&   params [[buffer(3)]],  // {T_frames, n_fft, hop_size, normalized}
    uint sid [[thread_position_in_grid]])
{
    int T_frames = params.x, n_fft = params.y, hop_size = params.z;
    bool normalized = params.w;
    int n_bins    = n_fft / 2 + 1;
    int trim      = n_fft / 2;
    int out_len   = (T_frames - 1) * hop_size;

    if ((int)sid >= out_len) return;

    int ola_pos = (int)sid + trim;

    int f_start = max(0,             (ola_pos - n_fft + 1 + hop_size - 1) / hop_size);
    int f_end   = min(T_frames - 1,  ola_pos / hop_size);

    const float two_pi_inv_nfft = 2.0f * M_PI_F / float(n_fft);
    float inv_nfft = 1.0f / float(n_fft);

    float acc      = 0.0f;
    float win_env  = 0.0f;

    for (int f = f_start; f <= f_end; f++) {
        int n = ola_pos - f * hop_size;
        float win_n = 0.5f * (1.0f - cos(2.0f * M_PI_F * float(n) / float(n_fft)));

        device const float* m = mag   + f * n_bins;
        device const float* p = phase + f * n_bins;

        float s = m[0] * cos(p[0]);
        if (normalized) {
            for (int k = 1; k < n_bins - 1; k++)
                s += 2.0f * m[k] * cos(p[k] + two_pi_inv_nfft * float(k * n));
        } else {
            for (int k = 1; k < n_bins - 1; k++)
                s += m[k] * cos(p[k] + two_pi_inv_nfft * float(k * n));
        }
        s += m[n_bins - 1] * cos(p[n_bins - 1] + M_PI_F * float(n));

        acc     += s * win_n * inv_nfft;
        win_env += win_n * win_n;
    }

    if (normalized && win_env > 1e-8f)
        out[sid] = acc / win_env;
    else
        out[sid] = acc;
}

// ── Reflection pad 1D (left-pad 1 sample) ────────────────────────
// in[T, C] → out[T+1, C]:  out[0, :] = in[1, :];  out[t+1, :] = in[t, :]
// Dispatch: grid=(T+1)*C
kernel void reflection_pad1d_k(
    device const float*  in     [[buffer(0)]],  // [T, C]
    device       float*  out    [[buffer(1)]],  // [T+1, C]
    constant     int2&   dims   [[buffer(2)]],  // {T, C}
    uint gid [[thread_position_in_grid]])
{
    int T = dims.x, C = dims.y;
    if ((int)gid >= (T + 1) * C) return;
    int t_out = (int)gid / C;
    int c     = (int)gid % C;
    int t_in  = (t_out == 0) ? 1 : (t_out - 1);
    out[gid]  = in[t_in * C + c];
}

// ── SineGen stage 1: compute rad_dn[H, T_dn] via bilinear downsample ──
kernel void sine_gen_rad_dn_k(
    device const float*  f0       [[buffer(0)]],  // [T]
    device       float*  rad_dn   [[buffer(1)]],  // [H, T_dn]
    constant     int4&   params   [[buffer(2)]],  // {T, H, T_dn, sr}
    constant     float&  scale    [[buffer(3)]],  // upsample_scale (float)
    uint2 gid [[thread_position_in_grid]])
{
    int h = (int)gid.x, i = (int)gid.y;
    int T = params.x, H = params.y, T_dn = params.z, sr = params.w;
    if (h >= H || i >= T_dn) return;

    float src = (float(i) + 0.5f) * scale - 0.5f;
    int i0 = (int)floor(src);
    int i1 = i0 + 1;
    float w1 = src - float(i0);
    float w0 = 1.0f - w1;
    i0 = clamp(i0, 0, T - 1);
    i1 = clamp(i1, 0, T - 1);

    float r0 = fmod(f0[i0] * float(h + 1) / float(sr), 1.0f);
    if (r0 < 0.0f) r0 += 1.0f;
    float r1 = fmod(f0[i1] * float(h + 1) / float(sr), 1.0f);
    if (r1 < 0.0f) r1 += 1.0f;

    rad_dn[h * T_dn + i] = w0 * r0 + w1 * r1;
}

// ── SineGen stage 2: cumulative sum of rad_dn[H, T_dn] → phase_dn[H, T_dn] ──
kernel void sine_gen_cumsum_k(
    device const float*  rad_dn   [[buffer(0)]],  // [H, T_dn]
    device       float*  phase_dn [[buffer(1)]],  // [H, T_dn]
    constant     int2&   dims     [[buffer(2)]],  // {H, T_dn}
    uint gid [[thread_position_in_grid]])
{
    int H = dims.x, T_dn = dims.y;
    int h = (int)gid;
    if (h >= H) return;
    float cum = 0.0f;
    device const float* src = rad_dn   + h * T_dn;
    device       float* dst = phase_dn + h * T_dn;
    for (int i = 0; i < T_dn; i++) {
        cum += src[i];
        dst[i] = cum;
    }
}

// ── SineGen stage 3: upsample phase_dn → full T resolution and compute sin ──
kernel void sine_gen_output_k(
    device const float*  phase_dn  [[buffer(0)]],  // [H, T_dn]
    device const float*  f0        [[buffer(1)]],  // [T] for voicing
    device       float*  out       [[buffer(2)]],  // [T, H]
    constant     int4&   params    [[buffer(3)]],  // {T, H, T_dn, 0}
    constant     float2& ap        [[buffer(4)]],  // {sine_amp, voiced_threshold}
    constant     float&  inv_scale [[buffer(5)]],  // 1.0 / upsample_scale
    constant     float&  scale     [[buffer(6)]],  // upsample_scale
    uint2 gid [[thread_position_in_grid]])
{
    int t = (int)gid.x, h = (int)gid.y;
    int T = params.x, H = params.y, T_dn = params.z;
    if (t >= T || h >= H) return;

    float sine_amp         = ap.x;
    float voiced_threshold = ap.y;

    float src = (float(t) + 0.5f) * inv_scale - 0.5f;
    int i0 = (int)floor(src);
    int i1 = i0 + 1;
    float w1 = src - float(i0);
    float w0 = 1.0f - w1;
    i0 = clamp(i0, 0, T_dn - 1);
    i1 = clamp(i1, 0, T_dn - 1);

    float ph_dn = w0 * phase_dn[h * T_dn + i0] + w1 * phase_dn[h * T_dn + i1];
    float ph = ph_dn * scale;

    bool voiced = f0[t] > voiced_threshold;
    out[t * H + h] = voiced ? sine_amp * sin(2.0f * M_PI_F * ph) : 0.0f;
}

// ── LM ops ────────────────────────────────────────────────────────

// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * gamma
// One threadgroup per token.  threadgroup memory: tg_size floats.
kernel void rmsnorm_k(
    device const float* input  [[buffer(0)]],
    device const float* gamma  [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant int2&      dims   [[buffer(3)]],   // {N, C}
    constant float&     eps    [[buffer(4)]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shmem [[threadgroup(0)]])
{
    int C = dims.y;
    device const float* x = input  + (int)tg_id * C;
    device float*       y = output + (int)tg_id * C;
    float ss = 0.f;
    for (int c = (int)lid; c < C; c += (int)tg_size)
        ss += x[c] * x[c];
    shmem[lid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size >> 1; s > 0; s >>= 1) {
        if (lid < s) shmem[lid] += shmem[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shmem[0] / float(C) + eps);
    for (int c = (int)lid; c < C; c += (int)tg_size)
        y[c] = x[c] * inv_rms * gamma[c];
}

// INT8 GEMV: 128 threads (4 SIMD groups) per output channel.
// Weight layout: row-major [C_out, C_in].
// 128 threads cover 128 consecutive bytes per iteration → full cache-line utilization.
// 4× more SIMD groups vs 32-thread design → better latency hiding for small C_out.
// threadgroup(0) must be set to 4*sizeof(float) by caller.
kernel void seqgemm_i8_matvec_k(
    device const float*  input    [[buffer(0)]],
    device const int8_t* weight   [[buffer(1)]],   // [C_out, C_in] row-major
    device const float*  w_scales [[buffer(2)]],
    device const float*  bias     [[buffer(3)]],
    device float*        output   [[buffer(4)]],
    constant int2&       dims     [[buffer(5)]],   // {C_in, C_out}
    constant int&        has_bias [[buffer(6)]],
    uint  tg_id   [[threadgroup_position_in_grid]],
    uint  tg_lid  [[thread_position_in_threadgroup]],
    uint  lane    [[thread_index_in_simdgroup]],
    uint  simd_i  [[simdgroup_index_in_threadgroup]],
    threadgroup float*   shmem   [[threadgroup(0)]])
{
    // One threadgroup = one output channel. TG size fixed at 128 = 4 SIMD groups.
    int C_in = dims.x, C_out = dims.y;
    int co = (int)tg_id;
    if (co >= C_out) return;
    device const int8_t* row = weight + co * C_in;
    float acc = 0.f;
    for (int k = (int)tg_lid; k < C_in; k += 128)
        acc += float(row[k]) * input[k];
    // Step 1: reduce 32-thread SIMD group → 1 partial sum
    acc = simd_sum(acc);
    // Step 2: collect 4 SIMD group sums in shared memory
    if (lane == 0) shmem[simd_i] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Step 3: thread 0 writes final output
    if (tg_lid == 0)
        output[co] = (shmem[0] + shmem[1] + shmem[2] + shmem[3])
                     * w_scales[co] + (has_bias ? bias[co] : 0.f);
}

// INT8 Batch GEMM for prefill (N>1).  2D dispatch: gid = {co, n}.
kernel void seqgemm_i8_batch_k(
    device const float*  input    [[buffer(0)]],
    device const int8_t* weight   [[buffer(1)]],
    device const float*  w_scales [[buffer(2)]],
    device const float*  bias     [[buffer(3)]],
    device float*        output   [[buffer(4)]],
    constant int4&       dims     [[buffer(5)]],   // {N, C_in, C_out, 0}
    constant int&        has_bias [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    int N = dims.x, C_in = dims.y, C_out = dims.z;
    int n = (int)gid.y, co = (int)gid.x;
    if (n >= N || co >= C_out) return;
    device const float*  x = input  + n * C_in;
    device const int8_t* w = weight + co * C_in;
    float acc = 0.f;
    for (int k = 0; k < C_in; k++)
        acc += float(w[k]) * x[k];
    output[n * C_out + co] = acc * w_scales[co] + (has_bias ? bias[co] : 0.f);
}

// Apply RoPE to Q (→ q_out) and K (→ directly into KV cache K slot).
// Dispatch: N * (nq + nkv) * (hd/2) threads.
kernel void rope_qk_and_kvcache_k(
    device const float* q_in     [[buffer(0)]],   // [N, nq*hd]
    device const float* k_in     [[buffer(1)]],   // [N, nkv*hd]
    device float*       q_out    [[buffer(2)]],   // [N, nq*hd]  Q after RoPE
    device float*       kv_cache [[buffer(3)]],   // [2, cache_len, nkv*hd]
    constant int4&      dims     [[buffer(4)]],   // {N, nq, nkv, hd}
    constant int2&      pos      [[buffer(5)]],   // {step_pos, cache_len}
    constant float&     theta    [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int N = dims.x, nq = dims.y, nkv = dims.z, hd = dims.w;
    int cache_len = pos.y;
    int hd2 = hd / 2;
    int n_total = nq + nkv;
    if ((int)gid >= N * n_total * hd2) return;

    int pair = (int)gid % hd2;
    int head = ((int)gid / hd2) % n_total;
    int tok  = (int)gid / (hd2 * n_total);
    int abs_pos = pos.x + tok;

    float inv_freq = 1.0f / pow(theta, 2.0f * float(pair) / float(hd));
    float angle = float(abs_pos) * inv_freq;
    float cs = cos(angle), sn = sin(angle);

    if (head < nq) {
        int base = (tok * nq + head) * hd;
        float x = q_in[base + pair], y = q_in[base + pair + hd2];
        q_out[base + pair]       = x * cs - y * sn;
        q_out[base + pair + hd2] = x * sn + y * cs;
    } else {
        int kh = head - nq;
        int k_base = (tok * nkv + kh) * hd;
        float x = k_in[k_base + pair], y = k_in[k_base + pair + hd2];
        float kr = x * cs - y * sn;
        float ki = x * sn + y * cs;
        int cache_pos = pos.x;
        int cp = cache_pos + tok;
        int kv_stride = nkv * hd;
        kv_cache[cp * kv_stride + kh * hd + pair]       = kr;
        kv_cache[cp * kv_stride + kh * hd + pair + hd2] = ki;
    }
}

// Copy V to KV-cache V slot.  Dispatch: N * nkv * hd threads.
kernel void kv_cache_v_write_k(
    device const float* v_in     [[buffer(0)]],
    device float*       kv_cache [[buffer(1)]],
    constant int4&      dims     [[buffer(2)]],   // {N, nkv, hd, 0}
    constant int2&      pos      [[buffer(3)]],   // {cache_pos, cache_len}
    uint gid [[thread_position_in_grid]])
{
    int N = dims.x, nkv = dims.y, hd = dims.z;
    int kv_stride = nkv * hd;
    int total = N * kv_stride;
    if ((int)gid >= total) return;
    int elem = (int)gid % kv_stride;
    int tok  = (int)gid / kv_stride;
    int cp   = pos.x + tok;
    int cache_len = pos.y;
    kv_cache[(long)cache_len * kv_stride + cp * kv_stride + elem] =
        v_in[tok * kv_stride + elem];
}

// Decode causal attention (N=1): one thread per query head.
kernel void causal_attn_decode_k(
    device const float* q_rope   [[buffer(0)]],   // [nq * hd]
    device const float* kv_cache [[buffer(1)]],   // [2, cache_len, nkv*hd]
    device float*       output   [[buffer(2)]],   // [nq * hd]
    device float*       scores   [[buffer(3)]],   // [nq * total_kv]
    constant int4&      dims     [[buffer(4)]],   // {nq, nkv, hd, total_kv}
    constant int&       cache_len[[buffer(5)]],
    uint qh [[thread_position_in_grid]])
{
    int nq = dims.x, nkv = dims.y, hd = dims.z, total_kv = dims.w;
    if ((int)qh >= nq) return;
    int group = nq / nkv;
    int kvh = (int)qh / group;
    int kv_stride = nkv * hd;
    float inv_sqrt = rsqrt(float(hd));
    device const float* q_head = q_rope + (int)qh * hd;
    device float* sc = scores + (int)qh * total_kv;

    float max_s = -1e30f;
    for (int kp = 0; kp < total_kv; kp++) {
        device const float* k_head = kv_cache + kp * kv_stride + kvh * hd;
        float s = 0.f;
        for (int d = 0; d < hd; d++) s += q_head[d] * k_head[d];
        s *= inv_sqrt;
        sc[kp] = s;
        max_s = max(max_s, s);
    }
    float sum_e = 0.f;
    for (int kp = 0; kp < total_kv; kp++) {
        float e = exp(sc[kp] - max_s);
        sc[kp] = e;
        sum_e += e;
    }
    float inv_sum = (sum_e > 0.f) ? 1.f / sum_e : 0.f;
    device const float* v_cache = kv_cache + (long)cache_len * kv_stride;
    device float* out_head = output + (int)qh * hd;
    for (int d = 0; d < hd; d++) {
        float val = 0.f;
        for (int kp = 0; kp < total_kv; kp++)
            val += sc[kp] * v_cache[kp * kv_stride + kvh * hd + d];
        out_head[d] = val * inv_sum;
    }
}

// Prefill causal attention (N>1): one thread per flattened (tok*nq + qh).
kernel void causal_attn_prefill_k(
    device const float* q_rope   [[buffer(0)]],   // [N, nq*hd]
    device const float* kv_cache [[buffer(1)]],   // [2, cache_len, nkv*hd]
    device float*       output   [[buffer(2)]],   // [N, nq*hd]
    device float*       scores   [[buffer(3)]],   // [N * nq * (cache_pos+N)]
    constant int4&      dims     [[buffer(4)]],   // {N, nq, nkv, hd}
    constant int2&      pos      [[buffer(5)]],   // {cache_pos, cache_len}
    uint gid [[thread_position_in_grid]])
{
    int N = dims.x, nq = dims.y, nkv = dims.z, hd = dims.w;
    int cache_pos = pos.x, cache_len = pos.y;
    int qh  = (int)gid % nq;
    int tok = (int)gid / nq;
    if (tok >= N) return;

    int group = nq / nkv;
    int kvh = qh / group;
    int kv_stride = nkv * hd;
    int total_kv = cache_pos + N;
    int causal_end = cache_pos + tok + 1;
    float inv_sqrt = rsqrt(float(hd));

    device const float* q_head = q_rope + (tok * nq + qh) * hd;
    device float* sc = scores + (tok * nq + qh) * total_kv;

    float max_s = -1e30f;
    for (int kp = 0; kp < causal_end; kp++) {
        device const float* k_head = kv_cache + kp * kv_stride + kvh * hd;
        float s = 0.f;
        for (int d = 0; d < hd; d++) s += q_head[d] * k_head[d];
        sc[kp] = s * inv_sqrt;
        max_s = max(max_s, sc[kp]);
    }
    float sum_e = 0.f;
    for (int kp = 0; kp < causal_end; kp++) {
        float e = exp(sc[kp] - max_s);
        sc[kp] = e;
        sum_e += e;
    }
    float inv_sum = (sum_e > 0.f) ? 1.f / sum_e : 0.f;
    device const float* v_cache = kv_cache + (long)cache_len * kv_stride;
    device float* out_head = output + (tok * nq + qh) * hd;
    for (int d = 0; d < hd; d++) {
        float val = 0.f;
        for (int kp = 0; kp < causal_end; kp++)
            val += sc[kp] * v_cache[kp * kv_stride + kvh * hd + d];
        out_head[d] = val * inv_sum;
    }
}
