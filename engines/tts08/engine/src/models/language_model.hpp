#pragma once
// language_model.hpp — Generic LM wrapper using InferenceModel
// Supports Gemma3-style transformer LMs.

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include "engine/memory_mode.hpp"

class InferenceModel;

// ────────────────────────────────────────────────────────────────
// LanguageModel
//   Wraps InferenceModel to provide prefill + autoregressive decode.
//
//   Arch contract:
//     Input tensor  "token_ids" : shape {T}   (int token IDs, stored as float)
//     Output tensor "logits"    : shape {T, vocab_size}  (last token used for next prediction)
//
//   The arch JSON must use named inputs:
//     "token_ids"   → the token ID sequence (shape {T})
//   The arch JSON output must be "logits" with shape {T, vocab_size}.
// ────────────────────────────────────────────────────────────────
class LanguageModel {
public:
    // Throws std::runtime_error on load failure.
    LanguageModel(const std::string& arch_path,
                  const std::string& weights_path,
                  int max_seq_len = 2048,
                  InferenceMemoryMode memory_mode = InferenceMemoryMode::Latency);
    ~LanguageModel();

    // ── Sampling ──────────────────────────────────────────────────
    struct SampleParams {
        int   max_new_tokens = 200;
        bool  greedy         = true;   // argmax if true
        float temperature    = 1.f;
        float top_p          = 1.f;    // 1.0 = no nucleus filter
        int   top_k          = 0;      // 0 = no top-k filter
        int   eos_token_id   = 1;      // stop on this ID (Gemma: <eos>=1)
    };

    // Run prefill on prompt_ids, then decode up to max_new_tokens.
    // Returns generated token IDs (not including prompt).
    // progress_cb is called with each new token (may be nullptr).
    std::vector<int> generate(
        const std::vector<int>& prompt_ids,
        const SampleParams& params,
        std::function<void(int tok)> progress_cb = nullptr);

    // Low-level: run prefill, return logits for the last token.
    // Returns vocab_size floats.
    std::vector<float> prefill(const std::vector<int>& prompt_ids);

    // Low-level: decode one step at absolute position pos.
    // Returns vocab_size floats (logits for the next token).
    std::vector<float> decode_step(int token_id, int pos);

    // Reset KV caches (call before a new generation sequence).
    void reset_state();

    // Vocabulary size (inferred from logits output shape on first forward).
    int vocab_size() const { return vocab_size_; }

    // Per-layer timing from the last generate() call.
    const std::unordered_map<std::string, double>& layer_profile() const;
    void set_memory_mode(InferenceMemoryMode mode);

private:
    std::unique_ptr<InferenceModel> model_;
    int max_seq_len_ = 0;
    int vocab_size_  = 0;  // set after first prefill

    // Sample next token from logits.
    int sample(const float* logits, int vocab, const SampleParams& p) const;
};
