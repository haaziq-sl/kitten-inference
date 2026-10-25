// attention.cpp — ILayer implementations for attention and sequence ops

#include "attention.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"
#include <cstring>
#include <stdexcept>
#if defined(__linux__)
#  include <sys/mman.h>
#endif

// Hint the kernel to back this memory with 2MB huge pages (reduces TLB misses).
static void hint_hugepages(void* ptr, size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (ptr && bytes > 0)
        madvise(ptr, bytes, MADV_HUGEPAGE);
#else
    (void)ptr; (void)bytes;
#endif
}

template <typename T>
static void release_vector(std::vector<T>& v)
{
    std::vector<T>().swap(v);
}

static Buffer share_host_buffer(std::shared_ptr<SharedPackedWeight>& owner,
                                const std::string& key,
                                Buffer&& buffer)
{
    if (active_backend()->mem_space != MemSpace::Host) return std::move(buffer);
    SharedPackedWeightView shared = share_host_packed_weight(key, std::move(buffer));
    owner = std::move(shared.owner);
    return std::move(shared.view);
}

// Use shared read_f32_tensor() from layer.hpp; thin alias kept for call-site compatibility.
static inline bool read_f32_attn(const ReadTensorFn& r, const std::string& k,
                                  std::vector<float>& out)
{ return read_f32_tensor(r, k, out); }

static bool read_i8_attn(const ReadTensorFn& read, const std::string& key,
                          std::vector<int8_t>& out)
{
    std::vector<int32_t> shape;
    std::vector<uint8_t> data;
    if (!read(key, shape, data)) return false;
    out.resize(data.size());
    memcpy(out.data(), data.data(), data.size());
    return true;
}

static void compute_i8_row_sums(const std::vector<int8_t>& weight,
                                int C_out, int C_in,
                                std::vector<int32_t>& row_sums)
{
    row_sums.assign(C_out, 0);
    if ((int64_t)weight.size() < (int64_t)C_out * C_in) return;

    for (int co = 0; co < C_out; ++co) {
        const int8_t* row = weight.data() + (size_t)co * C_in;
        int32_t sum = 0;
        for (int k = 0; k < C_in; ++k) sum += (int32_t)row[k];
        row_sums[co] = sum;
    }
}

// ─────────────────────────────────────────────────────────────────
// PatchPrepFp32
// ─────────────────────────────────────────────────────────────────

void PatchPrepFp32::load_weights(const ReadTensorFn& read)
{
    read_f32_attn(read, name + ".cls_token", cls_token);
    read_f32_attn(read, name + ".pos_embed", pos_embed);
    D = (int)cls_token.size();
}

Shape PatchPrepFp32::forward(const TensorView* inputs, int n_inputs,
                                  BufferView out, ScratchPads& sc)
{
    BufferView  in1       = inputs[0].view;
    Shape in_shape  = inputs[0].shape;
    (void)n_inputs; (void)sc;
    const float* patches = in1.as<float>();
    float*       output  = out.as<float>();
    int N_patches = in_shape.d1 * in_shape.d2;
    int d = (D > 0) ? D : in_shape.d0;

    patch_prep_fp32(patches, cls_token.data(), pos_embed.data(),
                    output, N_patches, d);
    return {in_shape.d0, N_patches + 1, 1};
}

// ─────────────────────────────────────────────────────────────────
// AttentionFp32
// ─────────────────────────────────────────────────────────────────

void AttentionFp32::load_weights(const ReadTensorFn& read)
{
    read_f32_attn(read, name + ".w_qkv",  w_qkv);
    read_f32_attn(read, name + ".b_qkv",  b_qkv);
    read_f32_attn(read, name + ".w_proj", w_proj);
    read_f32_attn(read, name + ".b_proj", b_proj);

    if (!w_qkv.empty()) {
        auto* _be = active_backend();
        float* pk_qkv = _be->pack_weights_fp32(w_qkv.data(), 3 * D, D);
        size_t nb_qkv = (size_t)((3*D+3)/4) * ((D+3)/4) * 16 * sizeof(float);
        w_qkv_packed = share_host_buffer(
            w_qkv_packed_owner,
            "AttentionFp32:" + name + ".w_qkv:" + std::to_string(3 * D) + "x" + std::to_string(D),
            Buffer(pk_qkv, nb_qkv, MemSpace::Host,
                [_be](void* p){ _be->free_packed_fp32(static_cast<float*>(p)); }));

        float* pk_proj = _be->pack_weights_fp32(w_proj.data(), D, D);
        size_t nb_proj = (size_t)((D+3)/4) * ((D+3)/4) * 16 * sizeof(float);
        w_proj_packed = share_host_buffer(
            w_proj_packed_owner,
            "AttentionFp32:" + name + ".w_proj:" + std::to_string(D) + "x" + std::to_string(D),
            Buffer(pk_proj, nb_proj, MemSpace::Host,
                [_be](void* p){ _be->free_packed_fp32(static_cast<float*>(p)); }));
        if (_be->mem_space == MemSpace::Host) {
            release_vector(w_qkv);
            release_vector(w_proj);
        }
    }
}

void AttentionFp32::upload_weights(Allocator& alloc)
{
    if (alloc.mem_space() == MemSpace::Host) return;
    auto upload = [&](Buffer& tb) {
        if (!tb.valid()) return;
        Buffer d = alloc.make_buffer(tb.bytes());
        alloc.copy_h2d(d.ptr(), tb.ptr(), tb.bytes());
        tb = std::move(d);
    };
    upload(w_qkv_packed);
    upload(w_proj_packed);
}

Shape AttentionFp32::forward(const TensorView* inputs, int n_inputs,
                                  BufferView out, ScratchPads& sc)
{
    BufferView  in1       = inputs[0].view;
    Shape in_shape  = inputs[0].shape;
    (void)n_inputs;
    const float* input  = in1.as<float>();
    float*       output = out.as<float>();
    int N = in_shape.d1 * in_shape.d2;
    int d = (D > 0) ? D : in_shape.d0;

    std::vector<float> dq_buf;
    if (use_dynamic_quant) {
        dq_buf.assign(input, input + (size_t)N * d);
        dynamic_quantize_dequant_f32(dq_buf.data(), N * d);
        input = dq_buf.data();
    }

    active_backend()->attention_flash_fp32(
        input,
        w_qkv.data(),
        b_qkv.empty()  ? nullptr : b_qkv.data(),
        w_proj.data(),
        b_proj.empty() ? nullptr : b_proj.data(),
        output, N, d, num_heads,
        sc.buffers[Scratch::F32_B].as<float>(),
        w_qkv_packed.as<float>(),
        w_proj_packed.as<float>(),
        sc.stream,
        use_dynamic_quant);
    return in_shape;
}

// ─────────────────────────────────────────────────────────────────
// SeqGemmFp32
// ─────────────────────────────────────────────────────────────────

void SeqGemmFp32::load_weights(const ReadTensorFn& read)
{
    read_f32_or_i8_tensor(read, name + ".weight", weight_fp32);
    read_f32_attn(read, name + ".bias",   bias_fp32);

    if (!weight_fp32.empty()) {
        auto* _be = active_backend();
        float* pk = _be->pack_weights_fp32(weight_fp32.data(), C_out, C_in);
        size_t nb = (size_t)((C_out+3)/4) * ((C_in+3)/4) * 16 * sizeof(float);
        w_packed_f32 = share_host_buffer(
            w_packed_f32_owner,
            "SeqGemmFp32:" + name + ".weight:" + std::to_string(C_out) + "x" + std::to_string(C_in),
            Buffer(pk, nb, MemSpace::Host,
                [_be](void* p){ _be->free_packed_fp32(static_cast<float*>(p)); }));

        // Per-channel INT8 quantization for single-token decode GEMV.
        // Reduces bandwidth 4× vs FP32 (critical for LM head: 32000×2048).
        w_i8_matvec_.resize((size_t)C_out * C_in);
        w_scales_matvec_.resize(C_out);
        for (int co = 0; co < C_out; ++co) {
            const float* row     = weight_fp32.data() + (size_t)co * C_in;
            int8_t*      out_row = w_i8_matvec_.data() + (size_t)co * C_in;
            float max_abs = 0.f;
            for (int k = 0; k < C_in; ++k) {
                float a = std::abs(row[k]);
                if (a > max_abs) max_abs = a;
            }
            float scale = (max_abs > 1e-8f) ? max_abs / 127.f : 1e-8f;
            float inv   = (max_abs > 1e-8f) ? 127.f / max_abs : 0.f;
            w_scales_matvec_[co] = scale;
            for (int k = 0; k < C_in; ++k) {
                int q    = (int)std::roundf(row[k] * inv);
                out_row[k] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
            }
        }
        if (_be->mem_space == MemSpace::Host && !keep_matvec_weight) {
            release_vector(w_i8_matvec_);
            release_vector(w_scales_matvec_);
        } else {
            // Hint kernel to back weight buffer with 2MB huge pages → fewer TLB misses.
            hint_hugepages(w_i8_matvec_.data(), w_i8_matvec_.size());
        }
        if (_be->mem_space == MemSpace::Host) {
            release_vector(weight_fp32);
        }
    }
}

void SeqGemmFp32::upload_weights(Allocator& alloc)
{
    if (alloc.mem_space() == MemSpace::Host) return;
    if (w_packed_f32.valid()) {
        Buffer d = alloc.make_buffer(w_packed_f32.bytes());
        alloc.copy_h2d(d.ptr(), w_packed_f32.ptr(), w_packed_f32.bytes());
        w_packed_f32 = std::move(d);
    }
    // Upload INT8 matvec weights for GPU decode path.
    if (!w_i8_matvec_.empty()) {
        size_t wi8_bytes = w_i8_matvec_.size();
        w_i8_matvec_buf_ = alloc.make_buffer(wi8_bytes);
        alloc.copy_h2d(w_i8_matvec_buf_.ptr(), w_i8_matvec_.data(), wi8_bytes);

        size_t wsc_bytes = w_scales_matvec_.size() * sizeof(float);
        w_scales_matvec_buf_ = alloc.make_buffer(wsc_bytes);
        alloc.copy_h2d(w_scales_matvec_buf_.ptr(), w_scales_matvec_.data(), wsc_bytes);
    }
}

Shape SeqGemmFp32::forward(const TensorView* inputs, int n_inputs,
                                BufferView out, ScratchPads& sc)
{
    BufferView  in1       = inputs[0].view;
    Shape in_shape  = inputs[0].shape;
    (void)n_inputs;
    const float* input  = in1.as<float>();
    float*       output = out.as<float>();
    int N = in_shape.d1 * in_shape.d2;

    auto* be = active_backend();

    // Single-token decode: use INT8 GEMV for 4× bandwidth reduction.
    // Use uploaded device buffer if available (Metal), else CPU vector.
    if (N == 1 && !w_i8_matvec_.empty() && be->seqgemm_int8_matvec) {
        const int8_t* wi8 = w_i8_matvec_buf_.valid()
            ? w_i8_matvec_buf_.as<int8_t>() : w_i8_matvec_.data();
        const float* wsc = w_scales_matvec_buf_.valid()
            ? w_scales_matvec_buf_.as<float>() : w_scales_matvec_.data();
        be->seqgemm_int8_matvec(input, wi8, wsc,
                                bias_fp32.empty() ? nullptr : bias_fp32.data(),
                                output, C_in, C_out, sc.stream);
        if (has_gelu) be->gelu_fp32(output, C_out, sc.stream);
        return {C_out, in_shape.d1, in_shape.d2};
    }

    std::vector<float> dq_buf;
    if (use_dynamic_quant) {
        dq_buf.assign(input, input + (size_t)N * C_in);
        dynamic_quantize_dequant_f32(dq_buf.data(), N * C_in);
        input = dq_buf.data();
    }

    be->seqgemm_fp32(input, weight_fp32.data(),
                     bias_fp32.empty() ? nullptr : bias_fp32.data(),
                     output, N, C_in, C_out,
                     has_gelu, w_packed_f32.as<float>(), sc.stream);
    return {C_out, in_shape.d1, in_shape.d2};
}

// ─────────────────────────────────────────────────────────────────
// ClsExtractFp32
// ─────────────────────────────────────────────────────────────────

Shape ClsExtractFp32::forward(const TensorView* inputs, int n_inputs,
                                   BufferView out, ScratchPads& sc)
{
    BufferView  in1       = inputs[0].view;
    Shape in_shape  = inputs[0].shape;
    (void)n_inputs; (void)sc;
    const float* input  = in1.as<float>();
    float*       output = out.as<float>();
    cls_extract_fp32(input, output, in_shape.d0);
    return {in_shape.d0, 1, 1};
}

// ─────────────────────────────────────────────────────────────────
// PosEmbAddFp32
// ─────────────────────────────────────────────────────────────────

void PosEmbAddFp32::load_weights(const ReadTensorFn& read)
{
    read_f32_attn(read, name + ".weight", weight);
}

Shape PosEmbAddFp32::forward(const TensorView* inputs, int n_inputs,
                                  BufferView out, ScratchPads& /*sc*/)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const float* in      = in1.as<float>();
    float*       out_ptr = out.as<float>();
    const size_t N   = (size_t)sh.d0 * sh.d1 * sh.d2;
    memcpy(out_ptr, in, N * sizeof(float));
    // Always on CPU (required_mem_space = Host).
    add_vectors_fp32(out_ptr, weight.data(), (int)N);
    return sh;
}

// ─────────────────────────────────────────────────────────────────
// SeqGemmInt8
// ─────────────────────────────────────────────────────────────────

void SeqGemmInt8::load_weights(const ReadTensorFn& read)
{
    read_i8_attn( read, name + ".weight_i8", weight_i8);
    read_f32_attn(read, name + ".w_scales",  w_scales);
    read_f32_attn(read, name + ".bias",      bias_fp32);

    if (!weight_i8.empty() && !w_scales.empty()) {
        compute_i8_row_sums(weight_i8, C_out, C_in, w_row_sums);

        auto* _be = active_backend();
        int8_t* pk = _be->pack_weights_int8(weight_i8.data(), C_out, C_in);
        size_t nb = (size_t)((C_out+7)/8) * ((C_in+3)/4) * 32;
        w_packed = share_host_buffer(
            w_packed_owner,
            "SeqGemmInt8:" + name + ".weight_i8:" + std::to_string(C_out) + "x" + std::to_string(C_in),
            Buffer(pk, nb, MemSpace::Host,
                [_be](void* p){ _be->free_packed_int8(static_cast<int8_t*>(p)); }));
        eff_zeros.assign(C_out, 0LL);
    }
    if (active_backend()->mem_space == MemSpace::Host && !keep_matvec_weight) {
        release_vector(weight_i8);
    } else if (!weight_i8.empty()) {
        // Hint kernel to back weight buffer with 2MB huge pages → fewer TLB misses.
        hint_hugepages(weight_i8.data(), weight_i8.size());
    }
}

void SeqGemmInt8::upload_weights(Allocator& alloc)
{
    if (alloc.mem_space() == MemSpace::Host) return;
    if (w_packed.valid()) {
        Buffer d = alloc.make_buffer(w_packed.bytes());
        alloc.copy_h2d(d.ptr(), w_packed.ptr(), w_packed.bytes());
        w_packed = std::move(d);
    }
    // Upload raw row-major int8 weights and per-channel scales for GPU GEMV path.
    if (!weight_i8.empty()) {
        weight_i8_buf_ = alloc.make_buffer(weight_i8.size());
        alloc.copy_h2d(weight_i8_buf_.ptr(), weight_i8.data(), weight_i8.size());
    }
    if (!w_scales.empty()) {
        size_t sb = w_scales.size() * sizeof(float);
        w_scales_buf_ = alloc.make_buffer(sb);
        alloc.copy_h2d(w_scales_buf_.ptr(), w_scales.data(), sb);
    }
}

Shape SeqGemmInt8::forward(const TensorView* inputs, int n_inputs,
                                BufferView out, ScratchPads& sc)
{
    BufferView  in1       = inputs[0].view;
    Shape in_shape  = inputs[0].shape;
    (void)n_inputs;
    const float* input  = in1.as<float>();
    float*       output = out.as<float>();
    int N = in_shape.d1 * in_shape.d2;

    auto* be = active_backend();
    if (N == 1 && !weight_i8.empty() && be->seqgemm_int8_matvec) {
        // Single-token decode GEMV. Use uploaded device buffers if available.
        const int8_t* wi8 = weight_i8_buf_.valid()
            ? weight_i8_buf_.as<int8_t>() : weight_i8.data();
        const float* wsc = w_scales_buf_.valid()
            ? w_scales_buf_.as<float>() : w_scales.data();
        be->seqgemm_int8_matvec(input, wi8, wsc,
                                bias_fp32.empty() ? nullptr : bias_fp32.data(),
                                output, C_in, C_out, sc.stream);
        if (has_gelu) be->gelu_fp32(output, C_out, sc.stream);
    } else if (be->mem_space == MemSpace::Host) {
        seqgemm_int8_with_rows(input, w_packed.as<int8_t>(), w_scales.data(), eff_zeros.data(),
                               bias_fp32.empty() ? nullptr : bias_fp32.data(),
                               sc.buffers[Scratch::I8_B].as<int8_t>(), sc.buffers[Scratch::F32_C].as<float>(),
                               output, N, C_in, C_out, has_gelu,
                               w_row_sums.empty() ? nullptr : w_row_sums.data(),
                               sc.stream);
    } else {
        be->seqgemm_int8(input, w_packed.as<int8_t>(), w_scales.data(), eff_zeros.data(),
                         bias_fp32.empty() ? nullptr : bias_fp32.data(),
                         sc.buffers[Scratch::I8_B].as<int8_t>(), sc.buffers[Scratch::F32_C].as<float>(),
                         output, N, C_in, C_out, has_gelu, sc.stream);
    }
    return {C_out, in_shape.d1, in_shape.d2};
}

// ─────────────────────────────────────────────────────────────────
// AttentionInt8
// ─────────────────────────────────────────────────────────────────

void AttentionInt8::load_weights(const ReadTensorFn& read)
{
    read_i8_attn( read, name + ".w_qkv_i8",    w_qkv_i8);
    read_f32_attn(read, name + ".w_qkv_scales", w_qkv_scales);
    read_f32_attn(read, name + ".b_qkv",        b_qkv);
    read_i8_attn( read, name + ".w_proj_i8",    w_proj_i8);
    read_f32_attn(read, name + ".w_proj_scales", w_proj_scales);
    read_f32_attn(read, name + ".b_proj",        b_proj);

    if (!w_qkv_i8.empty() && !w_qkv_scales.empty()) {
        int three_D = 3 * D;
        compute_i8_row_sums(w_qkv_i8,  three_D, D, w_qkv_row_sums);
        compute_i8_row_sums(w_proj_i8, D,       D, w_proj_row_sums);

        auto* _be = active_backend();
        int8_t* pk_qkv = _be->pack_weights_int8(w_qkv_i8.data(), three_D, D);
        size_t nb_qkv = (size_t)((three_D+3)/4) * ((D+3)/4) * 16;
        w_qkv_packed = share_host_buffer(
            w_qkv_packed_owner,
            "AttentionInt8:" + name + ".w_qkv_i8:" + std::to_string(three_D) + "x" + std::to_string(D),
            Buffer(pk_qkv, nb_qkv, MemSpace::Host,
                [_be](void* p){ _be->free_packed_int8(static_cast<int8_t*>(p)); }));

        int8_t* pk_proj = _be->pack_weights_int8(w_proj_i8.data(), D, D);
        size_t nb_proj = (size_t)((D+3)/4) * ((D+3)/4) * 16;
        w_proj_packed = share_host_buffer(
            w_proj_packed_owner,
            "AttentionInt8:" + name + ".w_proj_i8:" + std::to_string(D) + "x" + std::to_string(D),
            Buffer(pk_proj, nb_proj, MemSpace::Host,
                [_be](void* p){ _be->free_packed_int8(static_cast<int8_t*>(p)); }));

        eff_zeros.assign(three_D, 0LL);
        if (_be->mem_space == MemSpace::Host) {
            release_vector(w_qkv_i8);
            release_vector(w_proj_i8);
        }
    }
}

void AttentionInt8::upload_weights(Allocator& alloc)
{
    if (alloc.mem_space() == MemSpace::Host) return;
    auto upload = [&](Buffer& tb) {
        if (!tb.valid()) return;
        Buffer d = alloc.make_buffer(tb.bytes());
        alloc.copy_h2d(d.ptr(), tb.ptr(), tb.bytes());
        tb = std::move(d);
    };
    upload(w_qkv_packed);
    upload(w_proj_packed);
}

Shape AttentionInt8::forward(const TensorView* inputs, int n_inputs,
                                  BufferView out, ScratchPads& sc)
{
    BufferView  in1       = inputs[0].view;
    Shape in_shape  = inputs[0].shape;
    (void)n_inputs;
    const float* input  = in1.as<float>();
    float*       output = out.as<float>();
    int N = in_shape.d1 * in_shape.d2;
    int d = (D > 0) ? D : in_shape.d0;

    auto* be = active_backend();
    if (be->mem_space == MemSpace::Host) {
        attention_flash_int8_with_rows(
            input,
            w_qkv_packed.as<int8_t>(), w_qkv_scales.data(), eff_zeros.data(),
            b_qkv.empty()  ? nullptr : b_qkv.data(),
            w_proj_packed.as<int8_t>(), w_proj_scales.data(),
            eff_zeros.data(),
            b_proj.empty() ? nullptr : b_proj.data(),
            output, N, d, num_heads,
            sc.buffers[Scratch::F32_B].as<float>(), sc.buffers[Scratch::I8_B].as<int8_t>(), sc.buffers[Scratch::F32_C].as<float>(),
            w_qkv_row_sums.empty() ? nullptr : w_qkv_row_sums.data(),
            w_proj_row_sums.empty() ? nullptr : w_proj_row_sums.data(),
            sc.stream);
    } else {
        be->attention_flash_int8(
            input,
            w_qkv_packed.as<int8_t>(), w_qkv_scales.data(), eff_zeros.data(),
            b_qkv.empty()  ? nullptr : b_qkv.data(),
            w_proj_packed.as<int8_t>(), w_proj_scales.data(),
            eff_zeros.data(),
            b_proj.empty() ? nullptr : b_proj.data(),
            output, N, d, num_heads,
            sc.buffers[Scratch::F32_B].as<float>(), sc.buffers[Scratch::I8_B].as<int8_t>(), sc.buffers[Scratch::F32_C].as<float>(), sc.stream);
    }
    return in_shape;
}

void AttentionFp32::scratch_needed(Shape in, size_t out[Scratch::N]) const
{
    if (in.d1 == 0 && in.d2 == 0) return;
    int N = in.d1 * in.d2;
    int d = (D > 0) ? D : in.d0;
    out[Scratch::F32_B] = std::max(out[Scratch::F32_B], (size_t)N * 3 * d * sizeof(float));
}

void SeqGemmInt8::scratch_needed(Shape in, size_t out[Scratch::N]) const
{
    if (in.d1 == 0 && in.d2 == 0) return;
    int N = in.d1 * in.d2;
    out[Scratch::I8_B]  = std::max(out[Scratch::I8_B],  (size_t)N * C_in);
    out[Scratch::F32_C] = std::max(out[Scratch::F32_C], (size_t)C_out * sizeof(float));
}

void AttentionInt8::scratch_needed(Shape in, size_t out[Scratch::N]) const
{
    if (in.d1 == 0 && in.d2 == 0) return;
    int N = in.d1 * in.d2;
    int d = (D > 0) ? D : in.d0;
    out[Scratch::F32_B] = std::max(out[Scratch::F32_B], (size_t)N * 3 * d * sizeof(float));
    out[Scratch::I8_B]  = std::max(out[Scratch::I8_B],  (size_t)N * d);
    out[Scratch::F32_C] = std::max(out[Scratch::F32_C], (size_t)3 * d * sizeof(float));
}

// ─────────────────────────────────────────────────────────────────
// CausalAttnCoreFp32
// ─────────────────────────────────────────────────────────────────

void CausalAttnCoreFp32::reset_state(int max_seq_len, Allocator& alloc)
{
    cache_len_ = max_seq_len;
    size_t sz = (size_t)2 * max_seq_len * num_kv_heads * head_dim * sizeof(float);
    kv_cache_buf_ = alloc.make_buffer(sz);
    alloc.zero_memory(kv_cache_buf_.ptr(), sz);
}

void CausalAttnCoreFp32::release_runtime_scratch()
{
    release_vector(fwd_scratch_);
}

Shape CausalAttnCoreFp32::forward(const TensorView* inputs, int /*n_inputs*/,
                                       BufferView out, ScratchPads& sc)
{
    // Q: inputs[0] shape {nq*head_dim, N, 1}
    // K: inputs[1] shape {nkv*head_dim, N, 1}
    // V: inputs[2] shape {nkv*head_dim, N, 1}
    const Shape qsh = inputs[0].shape;
    const int N  = qsh.d1 * qsh.d2;  // token count for this call

    // Scratch: N*(nq+nkv)*hd  (RoPE buffer)  +  N*nq*(step_pos+N)  (attention scores)
    const size_t total_kv = (size_t)sc.step_pos + N;
    const size_t need = (size_t)N * (num_q_heads + num_kv_heads) * head_dim
                      + (size_t)N * num_q_heads * total_kv;
    if (fwd_scratch_.size() < need) fwd_scratch_.resize(need);

    active_backend()->causal_attn_core_fp32(
        inputs[0].view.as<float>(),
        inputs[1].view.as<float>(),
        inputs[2].view.as<float>(),
        (float*)kv_cache_buf_.ptr(),
        sc.step_pos,           // cache_pos = write offset
        cache_len_,
        out.as<float>(),
        N, num_q_heads, num_kv_heads, head_dim,
        sc.step_pos,           // first token's absolute position
        rope_theta,
        sc.is_decode,
        fwd_scratch_.data(),
        sc.stream);

    return qsh;
}
