#include "op_type.hpp"
#include <cstring>

// Single lookup table: maps every accepted JSON string → OpType.
// Listed name is also the canonical name returned by op_type_name().
static const struct { const char* name; OpType type; } OP_TABLE[] = {
    { "conv_int8",               OpType::ConvInt8            },
    { "conv_fp32",               OpType::ConvFp32            },
    { "gemm_int8",               OpType::GemmInt8            },
    { "gemm_fp32",               OpType::GemmFp32            },
    { "add_int8",                OpType::AddInt8             },
    { "add_fp32",                OpType::AddFp32             },
    { "maxpool_int8",            OpType::MaxPoolInt8         },
    { "maxpool_fp32",            OpType::MaxPoolFp32         },
    { "avgpool_int8",            OpType::AvgPoolInt8         },
    { "avgpool_fp32",            OpType::AvgPoolFp32         },
    { "patch_prep_fp32",         OpType::PatchPrepFp32       },
    { "layernorm_fp32",          OpType::LayernormFp32       },
    { "attention_fp32",          OpType::AttentionFp32       },
    { "attention_int8",          OpType::AttentionInt8       },
    { "seqgemm_fp32",            OpType::SeqGemmFp32         },
    { "seqgemm_int8",            OpType::SeqGemmInt8         },
    { "cls_extract_fp32",        OpType::ClsExtractFp32      },
    { "upsample_bilinear_fp32",  OpType::UpsampleBilinearFp32},
    { "pos_emb_add_fp32",        OpType::PosEmbAddFp32       },
    { "leaky_relu_fp32",         OpType::LeakyReluFp32       },
    { "exp_fp32",                OpType::ExpFp32             },
    { "sin_fp32",                OpType::SinFp32             },
    { "sigmoid_fp32",            OpType::SigmoidFp32         },
    { "tanh_fp32",               OpType::TanhFp32            },
    { "gelu_fp32",               OpType::GeluFp32            },
    { "conv1d_fp32",             OpType::Conv1dFp32          },
    { "conv1d_int8",             OpType::Conv1dInt8          },
    { "conv_transpose1d_fp32",   OpType::ConvTranspose1dFp32 },
    { "lstm_fp32",               OpType::LstmFp32            },
    { "ada_in1d_fp32",           OpType::AdaIn1dFp32         },
    { "ada_layer_norm_fp32",     OpType::AdaLayerNormFp32    },
    { "embedding_fp32",          OpType::EmbeddingFp32       },
    { "bert_embeddings_fp32",    OpType::BertEmbeddingsFp32  },
    { "snake1d_fp32",            OpType::Snake1dFp32         },
    { "length_regulate_fp32",    OpType::LengthRegulateFp32  },
    { "istft_fp32",              OpType::IstftFp32           },
    { "stft_fp32",               OpType::StftFp32            },
    { "concat1d_fp32",           OpType::Concat1dFp32        },
    { "upsample_nearest1d_fp32", OpType::UpsampleNearest1dFp32},
    { "sine_gen_fp32",           OpType::SineGenFp32         },
    { "sum_channels_fp32",       OpType::SumChannelsFp32     },
    { "scale_fp32",              OpType::ScaleFp32           },
    { "reflection_pad1d_fp32",   OpType::ReflectionPad1dFp32 },
    { "slice_channels_fp32",     OpType::SliceChannelsFp32   },
    // LM ops
    { "rms_norm_fp32",           OpType::RmsNormFp32         },
    { "silu_fp32",               OpType::SiluFp32            },
    { "elemwise_mul_fp32",       OpType::ElemwiseMulFp32     },
    { "causal_attn_core_fp32",   OpType::CausalAttnCoreFp32  },
    // Bare aliases accepted as INT8 variants
    { "conv",                    OpType::ConvInt8            },
    { "gemm",                    OpType::GemmInt8            },
    { "add",                     OpType::AddInt8             },
    { "maxpool",                 OpType::MaxPoolInt8         },
    { "avgpool",                 OpType::AvgPoolInt8         },
};

OpType op_type_from_string(const std::string& s) {
    for (const auto& e : OP_TABLE)
        if (s == e.name) return e.type;
    return OpType::Unknown;
}

const char* op_type_name(OpType t) {
    if (t == OpType::Unknown) return "unknown";
    // Canonical name is the first entry in OP_TABLE with that type.
    for (const auto& e : OP_TABLE)
        if (e.type == t) return e.name;
    return "unknown";
}
