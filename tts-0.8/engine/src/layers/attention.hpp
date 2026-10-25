#pragma once
#include "engine/layer.hpp"
#include "engine/weight_cache.hpp"

class PatchPrepFp32 : public ILayer {
public:
    int D = 0;
    std::vector<float> cls_token;
    std::vector<float> pos_embed;

    void     load_weights(const ReadTensorFn&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    MemSpace required_mem_space() const override { return MemSpace::Host; }
    Shape output_shape(Shape in) const override {
        return {in.d0, in.d1 * in.d2 + 1, 1};
    }
};

class AttentionFp32 : public ILayer {
public:
    int num_heads = 0;
    int D = 0;
    bool use_dynamic_quant = false;

    std::vector<float> w_qkv;
    std::vector<float> b_qkv;
    std::vector<float> w_proj;
    std::vector<float> b_proj;

    Buffer w_qkv_packed;
    Buffer w_proj_packed;
    std::shared_ptr<SharedPackedWeight> w_qkv_packed_owner;
    std::shared_ptr<SharedPackedWeight> w_proj_packed_owner;

    void     load_weights(const ReadTensorFn&) override;
    void     upload_weights(Allocator&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    void     scratch_needed(Shape in, size_t out[Scratch::N]) const override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return in; }
};

class SeqGemmFp32 : public ILayer {
public:
    int  C_in = 0, C_out = 0;
    bool has_gelu = false;
    bool use_dynamic_quant = false;
    bool keep_matvec_weight = true;

    std::vector<float>  weight_fp32;
    std::vector<float>  bias_fp32;
    Buffer              w_packed_f32;
    std::shared_ptr<SharedPackedWeight> w_packed_f32_owner;

    // Per-channel INT8 weights for single-token decode GEMV
    std::vector<int8_t> w_i8_matvec_;
    std::vector<float>  w_scales_matvec_;
    Buffer              w_i8_matvec_buf_;    // uploaded to device in upload_weights()
    Buffer              w_scales_matvec_buf_;

    void     load_weights(const ReadTensorFn&) override;
    void     upload_weights(Allocator&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return {C_out, in.d1, in.d2}; }
};

class ClsExtractFp32 : public ILayer {
public:
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    MemSpace required_mem_space() const override { return MemSpace::Host; }
    Shape output_shape(Shape in) const override { return {in.d0, 1, 1}; }
};

class PosEmbAddFp32 : public ILayer {
public:
    std::vector<float> weight;

    void     load_weights(const ReadTensorFn&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    MemSpace required_mem_space() const override { return MemSpace::Host; }
    Shape output_shape(Shape in) const override { return in; }
};

class SeqGemmInt8 : public ILayer {
public:
    int  C_in = 0, C_out = 0;
    bool has_gelu = false;
    bool keep_matvec_weight = true;

    std::vector<int8_t>  weight_i8;
    std::vector<float>   w_scales;
    std::vector<float>   bias_fp32;
    std::vector<int64_t> eff_zeros;
    std::vector<int32_t> w_row_sums;
    Buffer w_packed;
    std::shared_ptr<SharedPackedWeight> w_packed_owner;
    Buffer weight_i8_buf_;   // row-major int8 weights uploaded to device
    Buffer w_scales_buf_;    // per-channel scales uploaded to device

    void     load_weights(const ReadTensorFn&) override;
    void     upload_weights(Allocator&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    void     scratch_needed(Shape in, size_t out[Scratch::N]) const override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return {C_out, in.d1, in.d2}; }
};

class AttentionInt8 : public ILayer {
public:
    int num_heads = 0;
    int D = 0;

    std::vector<int8_t>  w_qkv_i8;
    std::vector<float>   w_qkv_scales;
    std::vector<float>   b_qkv;
    std::vector<int8_t>  w_proj_i8;
    std::vector<float>   w_proj_scales;
    std::vector<float>   b_proj;
    std::vector<int64_t> eff_zeros;
    std::vector<int32_t> w_qkv_row_sums;
    std::vector<int32_t> w_proj_row_sums;
    Buffer w_qkv_packed;
    Buffer w_proj_packed;
    std::shared_ptr<SharedPackedWeight> w_qkv_packed_owner;
    std::shared_ptr<SharedPackedWeight> w_proj_packed_owner;

    void     load_weights(const ReadTensorFn&) override;
    void     upload_weights(Allocator&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    void     scratch_needed(Shape in, size_t out[Scratch::N]) const override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override { return in; }
};

// ─────────────────────────────────────────────────────────────────
// CausalAttnCoreFp32
//   Stateful layer: owns KV cache, applies RoPE, runs causal attention.
//   inputs[0] = Q [N, nq*head_dim]
//   inputs[1] = K [N, nkv*head_dim]
//   inputs[2] = V [N, nkv*head_dim]
//   output    = [N, nq*head_dim]
// ─────────────────────────────────────────────────────────────────
class CausalAttnCoreFp32 : public ILayer {
public:
    int   num_q_heads  = 0;
    int   num_kv_heads = 0;
    int   head_dim     = 0;
    float rope_theta   = 10000.f;

    // KV cache: [2, max_seq_len, num_kv_heads * head_dim]
    // Allocated via reset_state's Allocator — Metal-shared on GPU backend.
    Buffer kv_cache_buf_;
    int    cache_len_ = 0;   // allocated capacity (max_seq_len)

    // Forward scratch for CPU path (ignored by GPU backend).
    mutable std::vector<float> fwd_scratch_;

    bool is_stateful() const override { return true; }
    void reset_state(int max_seq_len, Allocator& alloc) override;
    void release_runtime_scratch() override;

    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    // No required_mem_space() override — defaults to active backend's space.
    // Metal backend implements GPU attention; CPU path uses CPU allocation.
    Shape output_shape(Shape in) const override {
        // in = shape of Q input: {nq*head_dim, N, 1}
        return in;
    }
    Shape output_shape(const Shape* inputs, int n) const override {
        return n > 0 ? inputs[0] : Shape{};
    }
};
