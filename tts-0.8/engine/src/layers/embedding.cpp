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
        tensor_values(data, weight, name + ".weight");
    }
    // The lookup clamps token ids to [0, num_embeddings).
    need_values(weight, layer_elems(name, {num_embeddings, embedding_dim}), name + ".weight");
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
    const size_t E = layer_elems(name, {emb_size});
    if (word_weight.empty() || word_weight.size() % E != 0)
        bad_weights(name + ".word_weight is not a whole number of rows of " + std::to_string(E));
    if (pos_weight.empty() || pos_weight.size() % E != 0)
        bad_weights(name + ".pos_weight is not a whole number of rows of " + std::to_string(E));
    allow_values(type_weight, E, name + ".type_weight");
    allow_values(ln_weight, E, name + ".layernorm.weight");
    allow_values(ln_bias, E, name + ".layernorm.bias");
}

Shape BertEmbeddingsFp32::forward(const TensorView* inputs, int n_inputs,
                                      BufferView out, ScratchPads& /*sc*/)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const int T = sh.d1 * sh.d2;
    // Token ids index the word table and positions the position table (no clamping in the kernel).
    covers(pos_weight, (size_t)T * emb_size, name, "pos_weight (one row per position)");
    const float* ids = in1.as<float>();
    const size_t words = word_weight.size() / emb_size;
    for (int t = 0; t < T; ++t)
        if (!(ids[t] >= 0.f && ids[t] < (float)words))
            throw std::runtime_error("model weights do not match the input: token id " + std::to_string(ids[t]) +
                                     " is outside " + name + ".word_weight (" + std::to_string(words) + " rows)");
    bert_embeddings_fp32(in1.as<float>(),
                         out.as<float>(), T, emb_size,
                         word_weight.data(), pos_weight.data(),
                         type_weight.empty() ? nullptr : type_weight.data(),
                         ln_weight.empty()   ? nullptr : ln_weight.data(),
                         ln_bias.empty()     ? nullptr : ln_bias.data(),
                         1e-12f);
    return {emb_size, T, 1};
}
