#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "engine/memory_mode.hpp"

class InferenceModel;

// Result of a single image inference.
struct ImageResult {
    std::vector<float> logits;  // raw class logits
    int    top1 = -1;           // argmax(-1 if image failed to load)
    double ms   = 0.0;          // forward-pass time (preprocessing excluded)
    bool   ok   = false;        // false if image could not be loaded
};

// Image classifier wrapping InferenceModel.
// Handles ImageNet-style resize+crop+normalise preprocessing for both
// INT8 (quantised) and FP32 models.
class ImageClassifier {
public:
    // Throws std::runtime_error if arch or weights cannot be opened.
    ImageClassifier(const std::string& arch_path,
                    const std::string& weights_path,
                    InferenceMemoryMode memory_mode = InferenceMemoryMode::Latency);
    ~ImageClassifier();  // defined in .cpp so InferenceModel is complete

    // Run inference on a raw RGB pixel buffer (packed HxWx3, uint8).
    ImageResult run(const uint8_t* rgb, int width, int height) const;

    // Load an image from disk, preprocess, then run.
    // Returns ok=false if the file cannot be loaded.
    ImageResult run_file(const std::string& image_path) const;

    std::string name()        const;
    bool        is_fp32()     const;
    int         input_h()     const;
    int         input_w()     const;
    void        set_memory_mode(InferenceMemoryMode mode);

private:
    std::unique_ptr<InferenceModel> model_;

    void preprocess_i8 (const uint8_t* rgb, int W, int H,
                        std::vector<int8_t>& out) const;
    void preprocess_f32(const uint8_t* rgb, int W, int H,
                        std::vector<float>&  out) const;
};
