// embedding.cpp — Embedding layer classes: EmbeddingFp32, BertEmbeddingsFp32

#include "embedding.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"
#include <cstring>
#include <cstdio>

// Use shared read_f32_tensor() from layer.hpp; thin alias kept for call-site compatibility.
static inline bool read_f32_emb(const ReadTensorFn& r, const std::string& k,
                                 std::vector<float>& out)
{ return read_f32_tensor(r, k, out); }

// ─────────────────────────────────────────────────────────────────
// EmbeddingFp32
//   Input:  {1, T, 1}  float token IDs
//   Output: {embedding_dim, T, 1}
// ─────────────────────────────────────────────────────────────────

void EmbeddingFp32::load_weights(const ReadTensorFn& read)
{
    std::vector<int32_t> shape;
    std::vector<uint8_t> data;
    if (read(name + ".weight", shape, data)) {
        if (shape.size() >= 2) {
            num_embeddings = shape[0];
            embedding_dim  = shape[1];
        }
        weight.resize(data.size() / sizeof(float));
        memcpy(weight.data(), data.data(), data.size());
    }
}

Shape EmbeddingFp32::forward(const TensorView* inputs, int n_inputs,
                                 BufferView out, ScratchPads& /*sc*/)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const float* ids     = in1.as<float>();
    float*       out_ptr = out.as<float>();
    // Always on CPU (required_mem_space = Host).
    embedding_lookup_fp32(ids, weight.data(), out_ptr,
                          sh.d1 * sh.d2, num_embeddings, embedding_dim);
    return {embedding_dim, sh.d1, sh.d2};
}

// ─────────────────────────────────────────────────────────────────
// BertEmbeddingsFp32
//   Combined word + position embedding lookup + LayerNorm
//   Input:  {1, T, 1}  float token IDs
//   Output: {emb_size, T, 1}
// ─────────────────────────────────────────────────────────────────

void BertEmbeddingsFp32::load_weights(const ReadTensorFn& read)
{
    auto rd = [&](const std::string& key, std::vector<float>& out) {
        read_f32_emb(read, key, out);
    };
    rd(name + ".word_weight",      word_weight);
    rd(name + ".pos_weight",       pos_weight);
    rd(name + ".type_weight",      type_weight);
    rd(name + ".layernorm.weight", ln_weight);
    rd(name + ".layernorm.bias",   ln_bias);
}

Shape BertEmbeddingsFp32::forward(const TensorView* inputs, int n_inputs,
                                      BufferView out, ScratchPads& /*sc*/)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const int T = sh.d1 * sh.d2;
    bert_embeddings_fp32(in1.as<float>(),
                         out.as<float>(), T, emb_size,
                         word_weight.data(), pos_weight.data(),
                         type_weight.empty() ? nullptr : type_weight.data(),
                         ln_weight.empty()   ? nullptr : ln_weight.data(),
                         ln_bias.empty()     ? nullptr : ln_bias.data(),
                         1e-12f);
    return {emb_size, T, 1};
}
