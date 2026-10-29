#pragma once
#include <string>

// Typed enumeration of every recognised graph op.
// Use op_type_from_string() to parse JSON; op_type_name() for the canonical string.
enum class OpType {
    Unknown = 0,
    // 2-D convolution
    ConvInt8, ConvFp32,
    // GEMM (fully-connected)
    GemmInt8, GemmFp32,
    // Element-wise add
    AddInt8, AddFp32,
    // Pooling
    MaxPoolInt8, MaxPoolFp32,
    AvgPoolInt8, AvgPoolFp32,
    // ViT / attention
    PatchPrepFp32,
    LayernormFp32,
    AttentionFp32, AttentionInt8,
    SeqGemmFp32,   SeqGemmInt8,
    ClsExtractFp32,
    UpsampleBilinearFp32,
    PosEmbAddFp32,
    // Point-wise activations
    LeakyReluFp32,
    ExpFp32, SinFp32, SigmoidFp32, TanhFp32, GeluFp32,
    // 1-D conv
    Conv1dFp32, Conv1dInt8,
    ConvTranspose1dFp32,
    // LSTM
    LstmFp32,
    // Norm (1-D)
    AdaIn1dFp32, AdaLayerNormFp32,
    // Embeddings
    EmbeddingFp32,
    BertEmbeddingsFp32,
    // TTS / sequence
    Snake1dFp32,
    LengthRegulateFp32,
    IstftFp32, StftFp32,
    Concat1dFp32,
    UpsampleNearest1dFp32,
    SineGenFp32,
    SumChannelsFp32,
    ScaleFp32,
    ReflectionPad1dFp32,
    SliceChannelsFp32,
    // LM ops
    RmsNormFp32,
    SiluFp32,
    ElemwiseMulFp32,
    CausalAttnCoreFp32,
};

// Parse a JSON op string into OpType.  Returns OpType::Unknown for unrecognised names.
// Bare aliases ("conv", "gemm", ...) are accepted as INT8 variants.
OpType      op_type_from_string(const std::string& s);

// Canonical string for an OpType (stable, used as profiling map key).
// Returns "unknown" for OpType::Unknown.
const char* op_type_name(OpType t);
