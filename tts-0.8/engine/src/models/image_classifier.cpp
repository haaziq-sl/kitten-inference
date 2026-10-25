// image_classifier.cpp — ImageClassifier: preprocessing + InferenceModel wrapper

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "utils/stb_image.h"
#include "utils/stb_image_resize2.h"

#include "models/image_classifier.hpp"
#include "engine/model.hpp"

#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>

// ImageNet per-channel normalisation constants
static const float IMGNET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMGNET_STD[3]  = {0.229f, 0.224f, 0.225f};

// ── Constructor / destructor ─────────────────────────────────────────────────

ImageClassifier::ImageClassifier(const std::string& arch_path,
                                 const std::string& weights_path,
                                 InferenceMemoryMode memory_mode)
    : model_(std::make_unique<InferenceModel>(arch_path, weights_path, memory_mode))
{}

ImageClassifier::~ImageClassifier() = default;

// ── Metadata ─────────────────────────────────────────────────────────────────

std::string ImageClassifier::name()        const { return model_->model_name(); }
bool        ImageClassifier::is_fp32()     const { return !model_->inputs().empty() && model_->inputs()[0].dtype == "float32"; }
int         ImageClassifier::input_h()     const { return model_->inputs()[0].shape.size() >= 2 ? model_->inputs()[0].shape[0] : 1; }
int         ImageClassifier::input_w()     const { return model_->inputs()[0].shape.size() >= 3 ? model_->inputs()[0].shape[1] : 1; }
void        ImageClassifier::set_memory_mode(InferenceMemoryMode mode) { model_->set_memory_mode(mode); }

// ── Preprocessing (from raw RGB buffer) ──────────────────────────────────────

void ImageClassifier::preprocess_i8(const uint8_t* rgb, int W, int H,
                                     std::vector<int8_t>& out) const
{
    const InputSpec& inp0_i8 = model_->inputs()[0];
    int ch = (int)inp0_i8.shape.size() >= 2 ? inp0_i8.shape[0] : 1;
    int cw = (int)inp0_i8.shape.size() >= 3 ? inp0_i8.shape[1] : 1;
    float scale = inp0_i8.scale;
    int   zp    = inp0_i8.zp;

    const int RSZ = std::max(ch, cw) + 32;
    int nW, nH;
    if (W <= H) { nW = RSZ; nH = (H * RSZ + W/2) / W; }
    else        { nH = RSZ; nW = (W * RSZ + H/2) / H; }

    std::vector<uint8_t> buf(nW * nH * 3);
    stbir_resize_uint8_linear(rgb, W, H, 0, buf.data(), nW, nH, 0, STBIR_RGB);

    int oh = (nH - ch) / 2, ow = (nW - cw) / 2;
    out.resize(ch * cw * 3);
    for (int c = 0; c < 3; ++c)
        for (int h = 0; h < ch; ++h)
            for (int w = 0; w < cw; ++w) {
                uint8_t px = buf[((h+oh)*nW + (w+ow))*3 + c];
                float   fp = (px / 255.0f - IMGNET_MEAN[c]) / IMGNET_STD[c];
                int     q  = (int)std::roundf(fp / scale) + zp;
                out[(h*cw + w)*3 + c] = (int8_t)std::clamp(q, -128, 127);
            }
}

void ImageClassifier::preprocess_f32(const uint8_t* rgb, int W, int H,
                                      std::vector<float>& out) const
{
    const InputSpec& inp0_f = model_->inputs()[0];
    int ch = (int)inp0_f.shape.size() >= 2 ? inp0_f.shape[0] : 1;
    int cw = (int)inp0_f.shape.size() >= 3 ? inp0_f.shape[1] : 1;

    const int RSZ = std::max(ch, cw) + 32;
    int nW, nH;
    if (W <= H) { nW = RSZ; nH = (H * RSZ + W/2) / W; }
    else        { nH = RSZ; nW = (W * RSZ + H/2) / H; }

    std::vector<uint8_t> buf(nW * nH * 3);
    stbir_resize_uint8_linear(rgb, W, H, 0, buf.data(), nW, nH, 0, STBIR_RGB);

    int oh = (nH - ch) / 2, ow = (nW - cw) / 2;
    out.resize(ch * cw * 3);
    for (int c = 0; c < 3; ++c)
        for (int h = 0; h < ch; ++h)
            for (int w = 0; w < cw; ++w) {
                uint8_t px = buf[((h+oh)*nW + (w+ow))*3 + c];
                out[(h*cw + w)*3 + c] =
                    (px / 255.0f - IMGNET_MEAN[c]) / IMGNET_STD[c];
            }
}

// ── Inference ────────────────────────────────────────────────────────────────

ImageResult ImageClassifier::run(const uint8_t* rgb, int width, int height) const
{
    ImageResult res;

    std::vector<float>  inp_f;
    std::vector<int8_t> inp_i;

    if (is_fp32()) preprocess_f32(rgb, width, height, inp_f);
    else           preprocess_i8 (rgb, width, height, inp_i);

    auto t0 = std::chrono::high_resolution_clock::now();
    {
        const InputSpec& inp0 = model_->inputs()[0];
        std::unordered_map<std::string, ModelIOTensor> dict;
        if (is_fp32())
            dict[inp0.name] = ModelIOTensor(inp_f.data(), inp0.shape);
        else
            dict[inp0.name] = ModelIOTensor(inp_i.data(), inp0.shape);
        auto outputs = model_->forward(dict);
        const auto& onames = model_->output_names();
        if (!onames.empty()) {
            auto it = outputs.find(onames[0]);
            if (it != outputs.end()) {
                const ModelIOTensor& ot = it->second;
                res.logits.assign(ot.as_float(), ot.as_float() + ot.numel());
            }
        }
    }
    res.ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    if (!res.logits.empty())
        res.top1 = (int)(std::max_element(res.logits.begin(),
                                          res.logits.end()) - res.logits.begin());
    res.ok = true;
    return res;
}

ImageResult ImageClassifier::run_file(const std::string& image_path) const
{
    int W, H, C;
    unsigned char* img = stbi_load(image_path.c_str(), &W, &H, &C, 3);
    if (!img) return {};
    ImageResult res = run(img, W, H);
    stbi_image_free(img);
    return res;
}
