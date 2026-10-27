#pragma once
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "engine/memory_mode.hpp"

class InferenceModel;

// KittenTTS model wrapper. Voice pack size: (400, 256) float32.
// Voice pack: (400, 256) float32 rows.
// Row selection: clamp token count to the available voice rows, matching ONNX.
class KittenModel {
public:
    static constexpr int STYLE_DIM = 256;
    static constexpr int PACK_LEN  = 400;   // voices.npz shape[0]
    static constexpr int SAMPLE_RATE = 24000;

    KittenModel(const std::string& arch_path,
                const std::string& weights_path,
                InferenceMemoryMode memory_mode = InferenceMemoryMode::Latency);
    ~KittenModel();

    // Load voice binary: raw float32 [PACK_LEN * STYLE_DIM].
    void load_voice(const std::string& voice_path);

    // Synthesize from phoneme IDs (BOS=0 ... EOS=0).
    std::vector<float> synthesize(const std::vector<float>& phoneme_ids) const;
    int style_row_for_ids(int n_ids) const;

    static std::vector<float> parse_ids(const std::string& ids_str);
    static std::vector<float> load_ids_bin(const std::string& path);

    const std::unordered_map<std::string, double>& op_profile()    const;
    const std::unordered_map<std::string, double>& layer_profile() const;
    const std::unordered_map<std::string, std::string>& layer_op_types() const;
    void set_profile_enabled(bool enabled);
    void set_memory_report_enabled(bool enabled);
    void set_memory_mode(InferenceMemoryMode mode);
    void reset_profile();
    std::string name() const;
    void print_memory_report(const std::string& context = "") const;

private:
    std::unique_ptr<InferenceModel> model_;
    std::vector<float> raw_pack_;
    std::vector<float> style_dec_;   // [128]
    std::vector<float> style_pred_;  // [128]

    void apply_voice(int n_ids);
};
