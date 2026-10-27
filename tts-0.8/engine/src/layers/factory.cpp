#include "factory.hpp"
#include "conv_int8.hpp"
#include "conv_fp32.hpp"
#include "gemm.hpp"
#include "add.hpp"
#include "pool.hpp"
#include "activation.hpp"
#include "conv1d.hpp"
#include "lstm.hpp"
#include "norm.hpp"
#include "attention.hpp"
#include "embedding.hpp"
#include "signal.hpp"
#include "tensor.hpp"
#include "upsample.hpp"
#include <memory>

// ── Helpers ───────────────────────────────────────────────────
// Look up a field in the node JSON, checking top-level first then args sub-object.

static int get_int(const JsonValue& node_json, const char* key, int default_val) {
    if (node_json.contains(key)) return node_json[key].as_int();
    if (node_json.contains("args") && node_json["args"].contains(key)) return node_json["args"][key].as_int();
    return default_val;
}
static float get_float(const JsonValue& node_json, const char* key, float default_val) {
    if (node_json.contains(key)) return node_json[key].as_float();
    if (node_json.contains("args") && node_json["args"].contains(key)) return node_json["args"][key].as_float();
    return default_val;
}
static bool get_bool(const JsonValue& node_json, const char* key, bool default_val) {
    auto check = [&](const JsonValue& v) -> bool {
        return v.is_bool() ? v.as_bool() : (v.as_int() != 0);
    };
    if (node_json.contains(key)) return check(node_json[key]);
    if (node_json.contains("args") && node_json["args"].contains(key)) return check(node_json["args"][key]);
    return default_val;
}

static bool keep_decode_matvec_weight(const std::string& layer_name) {
    return layer_name.rfind("layers.", 0) == 0 || layer_name == "lm_head";
}

// ── Factory ───────────────────────────────────────────────────

std::unique_ptr<ILayer> make_layer(const std::string& layer_name,
                                    OpType             op,
                                    const JsonValue&   node_json,
                                    bool relu)
{
    std::unique_ptr<ILayer> layer;

    switch (op) {

    case OpType::ConvFp32: {
        auto* L       = new ConvFp32;
        L->C_out      = get_int(node_json, "C_out",      0);
        L->C_in       = get_int(node_json, "C_in",       0);
        L->kH         = get_int(node_json, "kH",         1);
        L->kW         = get_int(node_json, "kW",         1);
        L->stride_h   = get_int(node_json, "stride_h",   1);
        L->stride_w   = get_int(node_json, "stride_w",   1);
        L->pad_h      = get_int(node_json, "pad_h",      0);
        L->pad_w      = get_int(node_json, "pad_w",      0);
        L->groups     = get_int(node_json, "groups",     1);
        L->dilation_h = get_int(node_json, "dilation_h", 1);
        L->dilation_w = get_int(node_json, "dilation_w", 1);
        L->K = L->kH * L->kW * (L->C_in / (L->groups > 0 ? L->groups : 1));
        layer.reset(L);
        break;
    }
    case OpType::ConvInt8: {
        auto* L     = new ConvInt8;
        L->C_out    = get_int(node_json, "C_out",    0);
        L->C_in     = get_int(node_json, "C_in",     0);
        L->kH       = get_int(node_json, "kH",       1);
        L->kW       = get_int(node_json, "kW",       1);
        L->stride_h = get_int(node_json, "stride_h", 1);
        L->stride_w = get_int(node_json, "stride_w", 1);
        L->pad_h    = get_int(node_json, "pad_h",    0);
        L->pad_w    = get_int(node_json, "pad_w",    0);
        L->groups   = get_int(node_json, "groups",   1);
        L->in_zp    = (int8_t)get_int(node_json, "in_zp",  0);
        L->out_zp   = (int8_t)get_int(node_json, "out_zp", 0);
        L->K = (L->C_in / (L->groups > 0 ? L->groups : 1)) * L->kH * L->kW;
        layer.reset(L);
        break;
    }
    case OpType::GemmFp32: {
        auto* L  = new GemmFp32;
        L->C_out = get_int(node_json, "C_out", 0);
        L->C_in  = get_int(node_json, "C_in",  0);
        layer.reset(L);
        break;
    }
    case OpType::GemmInt8: {
        auto* L  = new GemmInt8;
        L->C_out = get_int(node_json, "C_out", 0);
        L->C_in  = get_int(node_json, "C_in",  0);
        L->in_zp = (int8_t)get_int(node_json, "in_zp", 0);
        layer.reset(L);
        break;
    }
    case OpType::AddFp32:
        layer.reset(new AddFp32);
        break;
    case OpType::AddInt8: {
        auto* L      = new AddInt8;
        L->in1_scale = get_float(node_json, "in1_scale", 1.f);
        L->in2_scale = get_float(node_json, "in2_scale", 1.f);
        L->out_scale = get_float(node_json, "out_scale", 1.f);
        L->in1_zp    = get_int(node_json, "in1_zp", 0);
        L->in2_zp    = get_int(node_json, "in2_zp", 0);
        L->out_zp    = get_int(node_json, "out_zp", 0);
        layer.reset(L);
        break;
    }
    case OpType::MaxPoolFp32: {
        auto* L     = new MaxPoolFp32;
        L->kH       = get_int(node_json, "kH",       3);
        L->kW       = get_int(node_json, "kW",       3);
        L->stride_h = get_int(node_json, "stride_h", 2);
        L->stride_w = get_int(node_json, "stride_w", 2);
        L->pad_h    = get_int(node_json, "pad_h",    0);
        L->pad_w    = get_int(node_json, "pad_w",    0);
        layer.reset(L);
        break;
    }
    case OpType::MaxPoolInt8: {
        auto* L     = new MaxPoolInt8;
        L->kH       = get_int(node_json, "kH",       3);
        L->kW       = get_int(node_json, "kW",       3);
        L->stride_h = get_int(node_json, "stride_h", 2);
        L->stride_w = get_int(node_json, "stride_w", 2);
        L->pad_h    = get_int(node_json, "pad_h",    0);
        L->pad_w    = get_int(node_json, "pad_w",    0);
        L->in_zp    = (int8_t)get_int(node_json, "in_zp",  0);
        L->out_zp   = (int8_t)get_int(node_json, "out_zp", 0);
        layer.reset(L);
        break;
    }
    case OpType::AvgPoolFp32:
        layer.reset(new AvgPoolFp32);
        break;
    case OpType::AvgPoolInt8: {
        auto* L      = new AvgPoolInt8;
        L->in_scale  = get_float(node_json, "in_scale",  1.f);
        L->out_scale = get_float(node_json, "out_scale", 1.f);
        L->in_zp     = get_int(node_json, "in_zp",  0);
        L->out_zp    = get_int(node_json, "out_zp", 0);
        layer.reset(L);
        break;
    }
    case OpType::PatchPrepFp32:
        layer.reset(new PatchPrepFp32);
        break;
    case OpType::LayernormFp32: {
        auto* L = new LayerNormFp32;
        L->eps  = get_float(node_json, "eps", 1e-6f);
        layer.reset(L);
        break;
    }
    case OpType::AttentionFp32: {
        auto* L              = new AttentionFp32;
        L->num_heads         = get_int(node_json, "num_heads",        0);
        L->D                 = get_int(node_json, "embed_dim",         0);
        L->use_dynamic_quant = get_bool(node_json, "dynamic_quantize", false);
        layer.reset(L);
        break;
    }
    case OpType::AttentionInt8: {
        auto* L      = new AttentionInt8;
        L->num_heads = get_int(node_json, "num_heads", 0);
        L->D         = get_int(node_json, "embed_dim", 0);
        layer.reset(L);
        break;
    }
    case OpType::SeqGemmFp32: {
        auto* L              = new SeqGemmFp32;
        L->has_gelu          = get_bool(node_json, "gelu",              false);
        L->C_out             = get_int(node_json, "C_out",             0);
        L->C_in              = get_int(node_json, "C_in",              0);
        L->use_dynamic_quant = get_bool(node_json, "dynamic_quantize",  false);
        L->keep_matvec_weight = keep_decode_matvec_weight(layer_name);
        layer.reset(L);
        break;
    }
    case OpType::SeqGemmInt8: {
        auto* L     = new SeqGemmInt8;
        L->has_gelu = get_bool(node_json, "gelu",  false);
        L->C_out    = get_int(node_json, "C_out", 0);
        L->C_in     = get_int(node_json, "C_in",  0);
        L->keep_matvec_weight = keep_decode_matvec_weight(layer_name);
        layer.reset(L);
        break;
    }
    case OpType::ClsExtractFp32:
        layer.reset(new ClsExtractFp32);
        break;
    case OpType::UpsampleBilinearFp32: {
        auto* L    = new UpsampleBilinearFp32;
        L->scale_h = get_float(node_json, "scale_h", 2.0f);
        L->scale_w = get_float(node_json, "scale_w", 2.0f);
        layer.reset(L);
        break;
    }
    case OpType::PosEmbAddFp32:
        layer.reset(new PosEmbAddFp32);
        break;

    // ── Point-wise activations ─────────────────────────────────
    case OpType::LeakyReluFp32: {
        auto* L     = new ActivationFp32;
        L->act_type = ActType::LEAKY_RELU;
        L->alpha    = get_float(node_json, "alpha", 0.2f);
        layer.reset(L);
        break;
    }
    case OpType::ExpFp32: {
        auto* L     = new ActivationFp32;
        L->act_type = ActType::EXP;
        layer.reset(L);
        break;
    }
    case OpType::SinFp32: {
        auto* L     = new ActivationFp32;
        L->act_type = ActType::SIN;
        layer.reset(L);
        break;
    }
    case OpType::SigmoidFp32: {
        auto* L     = new ActivationFp32;
        L->act_type = ActType::SIGMOID;
        layer.reset(L);
        break;
    }
    case OpType::TanhFp32: {
        auto* L     = new ActivationFp32;
        L->act_type = ActType::TANH;
        layer.reset(L);
        break;
    }
    case OpType::GeluFp32: {
        auto* L     = new ActivationFp32;
        L->act_type = ActType::GELU;
        layer.reset(L);
        break;
    }

    // ── 1-D conv ───────────────────────────────────────────────
    case OpType::Conv1dFp32: {
        auto* L        = new Conv1dFp32;
        L->C_out       = get_int(node_json, "C_out",       0);
        L->C_in        = get_int(node_json, "C_in",        0);
        L->kernel_size = get_int(node_json, "kernel_size",  1);
        L->stride      = get_int(node_json, "stride",       1);
        L->padding     = get_int(node_json, "padding",      0);
        L->dilation    = get_int(node_json, "dilation",     1);
        L->groups      = get_int(node_json, "groups",       1);
        layer.reset(L);
        break;
    }
    case OpType::Conv1dInt8: {
        auto* L        = new Conv1dInt8;
        L->C_out       = get_int(node_json, "C_out",       0);
        L->C_in        = get_int(node_json, "C_in",        0);
        L->kernel_size = get_int(node_json, "kernel_size",  1);
        L->stride      = get_int(node_json, "stride",       1);
        L->padding     = get_int(node_json, "padding",      0);
        L->dilation    = get_int(node_json, "dilation",     1);
        L->groups      = get_int(node_json, "groups",       1);
        layer.reset(L);
        break;
    }
    case OpType::ConvTranspose1dFp32: {
        auto* L           = new ConvTranspose1dFp32;
        L->C_out          = get_int(node_json, "C_out",          0);
        L->C_in           = get_int(node_json, "C_in",           0);
        L->kernel_size    = get_int(node_json, "kernel_size",     1);
        L->stride         = get_int(node_json, "stride",          1);
        L->padding        = get_int(node_json, "padding",         0);
        L->output_padding = get_int(node_json, "output_padding",  0);
        L->groups         = get_int(node_json, "groups",          1);
        layer.reset(L);
        break;
    }

    // ── LSTM ───────────────────────────────────────────────────
    case OpType::LstmFp32: {
        auto* L          = new LstmFp32;
        L->hidden_size   = get_int(node_json, "hidden_size",    0);
        L->bidirectional = get_bool(node_json, "bidirectional", false);
        layer.reset(L);
        break;
    }

    // ── Norm (1-D) ─────────────────────────────────────────────
    case OpType::AdaIn1dFp32: {
        auto* L    = new AdaIn1dFp32;
        L->C       = get_int(node_json, "C",       0);
        L->C_style = get_int(node_json, "C_style", 0);
        L->eps     = get_float(node_json, "eps",     1e-5f);
        layer.reset(L);
        break;
    }
    case OpType::AdaLayerNormFp32: {
        auto* L    = new AdaLayerNormFp32;
        L->C       = get_int(node_json, "C",       0);
        L->C_style = get_int(node_json, "C_style", 0);
        L->eps     = get_float(node_json, "eps",     1e-5f);
        layer.reset(L);
        break;
    }

    // ── Embeddings ─────────────────────────────────────────────
    case OpType::EmbeddingFp32: {
        auto* L           = new EmbeddingFp32;
        L->num_embeddings = get_int(node_json, "num_embeddings", 0);
        L->embedding_dim  = get_int(node_json, "embedding_dim",  0);
        layer.reset(L);
        break;
    }
    case OpType::BertEmbeddingsFp32: {
        auto* L     = new BertEmbeddingsFp32;
        L->emb_size = get_int(node_json, "emb_size", 128);
        layer.reset(L);
        break;
    }

    // ── TTS / sequence ─────────────────────────────────────────
    case OpType::Snake1dFp32:
        layer.reset(new SnakeFp32);
        break;
    case OpType::LengthRegulateFp32:
        layer.reset(new LengthRegulateFp32);
        break;
    case OpType::IstftFp32: {
        auto* L       = new IstftFp32;
        L->n_fft      = get_int(node_json, "n_fft",       20);
        L->hop_size   = get_int(node_json, "hop_size",      5);
        L->normalized = get_bool(node_json, "normalized", false);
        layer.reset(L);
        break;
    }
    case OpType::StftFp32: {
        auto* L     = new StftFp32;
        L->n_fft    = get_int(node_json, "n_fft",    20);
        L->hop_size = get_int(node_json, "hop_size",  5);
        layer.reset(L);
        break;
    }
    case OpType::Concat1dFp32: {
        auto* L = new Concat1dFp32;
        L->C2   = get_int(node_json, "C2", 0);
        layer.reset(L);
        break;
    }
    case OpType::UpsampleNearest1dFp32: {
        auto* L         = new UpsampleNearest1dFp32;
        L->scale_factor = get_int(node_json, "scale_factor", 2);
        layer.reset(L);
        break;
    }
    case OpType::SineGenFp32: {
        auto* L             = new SineGenFp32;
        L->sample_rate      = get_int(node_json, "sample_rate",       24000);
        L->harmonic_num     = get_int(node_json, "harmonic_num",           8);
        L->sine_amp         = get_float(node_json, "sine_amp",           0.1f);
        L->voiced_threshold = get_float(node_json, "voiced_threshold",  10.0f);
        L->noise_std        = get_float(node_json, "noise_std",          0.0f);
        L->upsample_scale   = get_int(node_json, "upsample_scale",        0);
        layer.reset(L);
        break;
    }
    case OpType::SumChannelsFp32:
        layer.reset(new SumChannelsFp32);
        break;
    case OpType::ScaleFp32: {
        auto* L   = new ScaleFp32;
        L->scalar = get_float(node_json, "scalar", 1.0f);
        layer.reset(L);
        break;
    }
    case OpType::ReflectionPad1dFp32:
        layer.reset(new ReflectionPad1dFp32);
        break;
    case OpType::SliceChannelsFp32: {
        auto* L     = new SliceChannelsFp32;
        L->ch_start = get_int(node_json, "ch_start", 0);
        L->ch_end   = get_int(node_json, "ch_end",   0);
        layer.reset(L);
        break;
    }

    // ── LM ops ─────────────────────────────────────────────────
    case OpType::RmsNormFp32: {
        auto* L = new RmsNormFp32;
        L->eps  = get_float(node_json, "eps", 1e-6f);
        layer.reset(L);
        break;
    }
    case OpType::SiluFp32: {
        auto* L     = new ActivationFp32;
        L->act_type = ActType::SILU;
        layer.reset(L);
        break;
    }
    case OpType::ElemwiseMulFp32:
        layer.reset(new ElemwiseMulFp32);
        break;
    case OpType::CausalAttnCoreFp32: {
        auto* L          = new CausalAttnCoreFp32;
        L->num_q_heads   = get_int(node_json, "num_q_heads",   0);
        L->num_kv_heads  = get_int(node_json, "num_kv_heads",  0);
        L->head_dim      = get_int(node_json, "head_dim",      0);
        L->rope_theta    = get_float(node_json, "rope_theta",  10000.f);
        layer.reset(L);
        break;
    }

    case OpType::Unknown:
    default:
        return nullptr;
    }

    if (layer) {
        layer->name = layer_name;
        layer->relu = relu;
    }
    return layer;
}
