// language_model.cpp — Generic LM wrapper using InferenceModel

#include "language_model.hpp"
#include "engine/model.hpp"
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>

LanguageModel::LanguageModel(const std::string& arch_path,
                              const std::string& weights_path,
                              int max_seq_len,
                              InferenceMemoryMode memory_mode)
    : max_seq_len_(max_seq_len)
{
    model_ = std::make_unique<InferenceModel>(arch_path, weights_path, memory_mode);
    model_->reset_state(max_seq_len);
}

LanguageModel::~LanguageModel() = default;

void LanguageModel::reset_state()
{
    model_->reset_state(max_seq_len_);
}

const std::unordered_map<std::string, double>& LanguageModel::layer_profile() const
{
    return model_->layer_profile();
}

void LanguageModel::set_memory_mode(InferenceMemoryMode mode)
{
    model_->set_memory_mode(mode);
}

// ─────────────────────────────────────────────────────────────────
// prefill — run on full prompt, return last-token logits
// ─────────────────────────────────────────────────────────────────
std::vector<float> LanguageModel::prefill(const std::vector<int>& prompt_ids)
{
    int T = (int)prompt_ids.size();
    // Pack token IDs as float (the engine uses float throughout)
    std::vector<float> ids_f(T);
    for (int i = 0; i < T; ++i) ids_f[i] = (float)prompt_ids[i];

    // Shape {T, 1}: primary dim = T tokens, C = 1 (IDs stored as float)
    std::unordered_map<std::string, ModelIOTensor> in;
    in["token_ids"] = ModelIOTensor(ids_f.data(), {T, 1});

    auto out = model_->forward(in, 0, false);

    if (out.find("logits") == out.end())
        throw std::runtime_error("LanguageModel: model output 'logits' not found");

    const auto& logits_t = out.at("logits");
    // Shape {T, vocab} in compact form: {T, vocab} → NHWC {T, vocab, 1} → d0=vocab, d1=T
    // The flat buffer is [T * vocab] floats.  Last-token logits = last row.
    size_t total = logits_t.numel();
    if (total == 0)
        throw std::runtime_error("LanguageModel: logits tensor is empty");

    // Infer vocab_size if not known yet
    if (vocab_size_ == 0) {
        // shape is {T, vocab} => numel/T
        vocab_size_ = (int)(total / (size_t)T);
    }

    // Return the last token's logits
    const float* ptr = logits_t.as_float() + (size_t)(T - 1) * vocab_size_;
    return std::vector<float>(ptr, ptr + vocab_size_);
}

// ─────────────────────────────────────────────────────────────────
// decode_step — run single token at position pos
// ─────────────────────────────────────────────────────────────────
std::vector<float> LanguageModel::decode_step(int token_id, int pos)
{
    float id_f = (float)token_id;

    // Shape {1, 1}: 1 token with 1 channel
    std::unordered_map<std::string, ModelIOTensor> in;
    in["token_ids"] = ModelIOTensor(&id_f, {1, 1});

    auto out = model_->forward(in, pos, true);

    if (out.find("logits") == out.end())
        throw std::runtime_error("LanguageModel: model output 'logits' not found");

    const auto& logits_t = out.at("logits");
    size_t total = logits_t.numel();
    if (vocab_size_ == 0) vocab_size_ = (int)total;

    const float* ptr = logits_t.as_float();
    return std::vector<float>(ptr, ptr + vocab_size_);
}

// ─────────────────────────────────────────────────────────────────
// sample — pick next token from logits
// ─────────────────────────────────────────────────────────────────
int LanguageModel::sample(const float* logits, int vocab, const SampleParams& p) const
{
    if (p.greedy || p.temperature == 0.f) {
        // Argmax
        return (int)(std::max_element(logits, logits + vocab) - logits);
    }

    // Temperature scaling + softmax
    std::vector<float> probs(vocab);
    float inv_temp = 1.f / p.temperature;
    float max_l = *std::max_element(logits, logits + vocab);
    for (int i = 0; i < vocab; ++i) probs[i] = std::exp((logits[i] - max_l) * inv_temp);

    // Top-k filtering
    int k = (p.top_k > 0 && p.top_k < vocab) ? p.top_k : vocab;
    if (k < vocab) {
        // Partial sort to find k-th largest
        std::vector<int> idx(vocab);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b){ return probs[a] > probs[b]; });
        for (int i = k; i < vocab; ++i) probs[idx[i]] = 0.f;
    }

    // Top-p (nucleus) filtering
    if (p.top_p < 1.f) {
        std::vector<int> idx(vocab);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return probs[a] > probs[b]; });
        float cumsum = 0.f;
        float total_p = 0.f;
        for (int i : idx) total_p += probs[i];
        float threshold = p.top_p * total_p;
        for (int i : idx) {
            cumsum += probs[i];
            if (cumsum > threshold) {
                // Zero remaining
                for (; &i < idx.data() + vocab; ++i) probs[idx[i]] = 0.f;  // won't work
                break;
            }
        }
        // Simple top-p: zero tokens below threshold
        float cumsum2 = 0.f;
        bool zeroing = false;
        for (int id : idx) {
            if (!zeroing) {
                cumsum2 += probs[id];
                if (cumsum2 >= threshold) zeroing = true;
            } else {
                probs[id] = 0.f;
            }
        }
    }

    // Sample
    float total_p = 0.f;
    for (float v : probs) total_p += v;

    static thread_local std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, total_p);
    float r = dist(rng);
    float cumsum = 0.f;
    for (int i = 0; i < vocab; ++i) {
        cumsum += probs[i];
        if (r <= cumsum) return i;
    }
    return vocab - 1;
}

// ─────────────────────────────────────────────────────────────────
// generate
// ─────────────────────────────────────────────────────────────────
std::vector<int> LanguageModel::generate(
    const std::vector<int>& prompt_ids,
    const SampleParams& params,
    std::function<void(int)> progress_cb)
{
    reset_state();

    // Prefill on the prompt
    auto logits = prefill(prompt_ids);
    int next_tok = sample(logits.data(), vocab_size_, params);

    std::vector<int> generated;
    generated.reserve(params.max_new_tokens);

    int pos = (int)prompt_ids.size();

    for (int step = 0; step < params.max_new_tokens; ++step) {
        generated.push_back(next_tok);
        if (progress_cb) progress_cb(next_tok);
        if (next_tok == params.eos_token_id) break;

        logits = decode_step(next_tok, pos);
        next_tok = sample(logits.data(), vocab_size_, params);
        ++pos;
    }

    return generated;
}
