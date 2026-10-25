// kitten.cpp — KittenModel: voice-style + synthesis wrapper (KittenTTS)

#include "models/kitten.hpp"
#include "engine/model.hpp"
#include "utils/file_open.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

KittenModel::KittenModel(const std::string& arch_path,
                         const std::string& weights_path,
                         InferenceMemoryMode memory_mode)
    : model_(std::make_unique<InferenceModel>(arch_path, weights_path, memory_mode))
    , style_dec_(128, 0.f)
    , style_pred_(128, 0.f)
{}

KittenModel::~KittenModel() = default;

std::string KittenModel::name() const { return model_->model_name(); }

const std::unordered_map<std::string, double>&
KittenModel::op_profile()    const { return model_->op_profile(); }
const std::unordered_map<std::string, double>&
KittenModel::layer_profile() const { return model_->layer_profile(); }
const std::unordered_map<std::string, std::string>&
KittenModel::layer_op_types() const { return model_->layer_op_types(); }

void KittenModel::set_profile_enabled(bool enabled) { model_->set_profile_enabled(enabled); }

void KittenModel::set_memory_report_enabled(bool enabled) { model_->set_memory_report_enabled(enabled); }

void KittenModel::set_memory_mode(InferenceMemoryMode mode) { model_->set_memory_mode(mode); }

void KittenModel::reset_profile() { model_->reset_profile(); }

void KittenModel::print_memory_report(const std::string& context) const {
    model_->print_memory_report(context.empty() ? nullptr : context.c_str(), stdout);
}

std::vector<float> KittenModel::parse_ids(const std::string& s)
{
    std::vector<float> ids;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty())
            try { ids.push_back((float)std::stoi(tok)); } catch (...) {}
    return ids;
}

std::vector<float> KittenModel::load_ids_bin(const std::string& path)
{
    FILE* f = fopen_utf8(path, "rb");
    if (!f) throw std::runtime_error("Cannot open IDs bin: " + path);
    fseek(f, 0, SEEK_END);
    long nb = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> v(nb / sizeof(float));
    if (!v.empty()) (void)fread(v.data(), sizeof(float), v.size(), f);
    fclose(f);
    return v;
}

void KittenModel::load_voice(const std::string& voice_path)
{
    FILE* f = fopen_utf8(voice_path, "rb");
    if (!f) throw std::runtime_error("Cannot open voice: " + voice_path);
    fseek(f, 0, SEEK_END);
    long nb = ftell(f);
    fseek(f, 0, SEEK_SET);
    raw_pack_.resize(nb / sizeof(float));
    if (!raw_pack_.empty()) (void)fread(raw_pack_.data(), sizeof(float), raw_pack_.size(), f);
    fclose(f);
}

int KittenModel::style_row_for_ids(int n_ids) const
{
    const int rows = (int)(raw_pack_.size() / STYLE_DIM);
    if (rows <= 0) return 0;
    if (n_ids < 0) return 0;
    return std::min(n_ids, rows - 1);
}

void KittenModel::apply_voice(int n_ids)
{
    if (raw_pack_.empty()) {
        std::fill(style_dec_.begin(),  style_dec_.end(),  0.f);
        std::fill(style_pred_.begin(), style_pred_.end(), 0.f);
        return;
    }
    const int row = style_row_for_ids(n_ids);
    size_t offset = (size_t)row * STYLE_DIM;
    if (offset + STYLE_DIM <= raw_pack_.size()) {
        // ONNX split: style[:,0:128] → decoder, style[:,128:] → predictor
        std::copy(raw_pack_.begin() + offset,       raw_pack_.begin() + offset + 128, style_dec_.begin());
        std::copy(raw_pack_.begin() + offset + 128, raw_pack_.begin() + offset + 256, style_pred_.begin());
    } else {
        std::fill(style_dec_.begin(),  style_dec_.end(),  0.f);
        std::fill(style_pred_.begin(), style_pred_.end(), 0.f);
    }
}

std::vector<float> KittenModel::synthesize(const std::vector<float>& phoneme_ids) const
{
    const_cast<KittenModel*>(this)->apply_voice((int)phoneme_ids.size());

    std::unordered_map<std::string, ModelIOTensor> inputs;
    inputs["phoneme_ids"] = ModelIOTensor(phoneme_ids.data(), {(int)phoneme_ids.size(), 1});
    inputs["style_dec"]   = ModelIOTensor(style_dec_.data(),  {(int)style_dec_.size()});
    inputs["style_pred"]  = ModelIOTensor(style_pred_.data(), {(int)style_pred_.size()});

    auto outputs = model_->forward(inputs);

    const auto& out_names = model_->output_names();
    if (out_names.empty()) return {};
    auto it = outputs.find(out_names[0]);
    if (it == outputs.end()) return {};
    const ModelIOTensor& ot = it->second;
    return std::vector<float>(ot.as_float(), ot.as_float() + ot.numel());
}
