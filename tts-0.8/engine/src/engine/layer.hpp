#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include "utils/json.hpp"
#include "backend/backend.hpp"

// ── Shared types ──────────────────────────────────────────────

// 3D tensor shape in the engine's internal layout (NHWC-derived).
// d0 = features/channels, d1 = primary spatial dim (H or T), d2 = secondary (W, defaults to 1).
struct Shape { int d0 = 0, d1 = 0, d2 = 0; };

// One entry in the input list passed to ILayer::forward().
// Bundles the buffer view and its current runtime shape together.
struct TensorView {
    BufferView  view;   // device or host pointer, size, memory space
    Shape shape;  // runtime shape of this tensor
};

// Maximum number of inputs a single layer can consume.
// Stack arrays in the engine are sized by this constant.
constexpr int MAX_LAYER_INPUTS = 8;

// Element dtype for a tensor slot or layer output.
struct DType {
    enum class Dtype { Float32, Int8 };
    Dtype dtype = Dtype::Float32;

    DType() = default;
    explicit DType(Dtype d) : dtype(d) {}

    static DType fp32() { return DType{Dtype::Float32}; }
    static DType int8() { return DType{Dtype::Int8}; }
    // Parses "int8" → int8(), anything else → fp32().
    static DType from_str(const std::string& s) {
        return s == "int8" ? int8() : fp32();
    }

    size_t      elem_bytes() const { return dtype == Dtype::Int8 ? 1 : 4; }
    const char* dtype_str()  const { return dtype == Dtype::Int8 ? "int8" : "float32"; }
    bool        is_fp32()    const { return dtype == Dtype::Float32; }
    bool        is_int8()    const { return dtype == Dtype::Int8; }
};

// ── Scratch slot indices ───────────────────────────────────────
// Named indices into the ScratchPads::scratch[] array.
// Each slot is a reusable temporary buffer; layers declare their
// requirements via scratch_needed() and use these constants to index sc.buffers[].
namespace Scratch {
    constexpr int I8_A  = 0;  // INT8 scratch A  — ConvInt8: im2col col / Winograd input
    constexpr int I32_A = 1;  // INT32 scratch   — ConvInt8: Winograd accumulator
    constexpr int F32_A = 2;  // Float scratch A — ConvFp32 / Conv1d: im2col / padding
    constexpr int F32_B = 3;  // Float scratch B — Attention: QKV buffer (3× activation)
    constexpr int I8_B  = 4;  // INT8 scratch B  — SeqGemmInt8 / AttentionInt8: quant input
    constexpr int F32_C = 5;  // Float scratch C — SeqGemmInt8 / AttentionInt8: per-ch scales
    constexpr int N     = 8;  // total slots (2 reserved)
}

// Bundle of scratch buffer views + metadata passed to every ILayer::forward() call.
struct ScratchPads {
    BufferView   buffers[Scratch::N] = {}; // temporary workspace buffers, indexed by Scratch::*
    StreamHandle stream = nullptr;         // CUDA/HIP stream; nullptr on CPU

    // ── Autoregressive generation context (zero/false for non-LM models) ──
    int  step_pos  = 0;      // absolute token position for RoPE and KV cache writes
    bool is_decode = false;  // false = prefill (full sequence), true = single-token decode
};

// Callback that reads a named tensor from the weights binary into shape + raw bytes.
// Returns false if the tensor is not present.
using ReadTensorFn = std::function<bool(const std::string&,
                                        std::vector<int32_t>&,
                                        std::vector<uint8_t>&)>;

// ── Shared weight-reading helpers ─────────────────────────────
// A model file can come from any Hugging Face repository or folder. A tensor that is missing, has a size that is not
// a whole number of values, or does not match the layer's sizes in the architecture file is refused with an error,
// so that no layer reads or writes past a weight array.

[[noreturn]] inline void bad_weights(const std::string& what) {
    throw std::runtime_error("model weights do not match the architecture: " + what);
}

// The raw bytes of a tensor as values of type T.
template <class T>
inline void tensor_values(const std::vector<uint8_t>& data, std::vector<T>& out, const std::string& key) {
    if (data.size() % sizeof(T) != 0)
        bad_weights(key + " has " + std::to_string(data.size()) + " bytes, not a whole number of " +
                    std::to_string(sizeof(T)) + "-byte values");
    out.resize(data.size() / sizeof(T));
    if (!data.empty()) memcpy(out.data(), data.data(), data.size());
}

// The product of a layer's sizes from the architecture file; each must be at least 1.
inline size_t layer_elems(const std::string& layer, std::initializer_list<long long> dims) {
    size_t n = 1;
    for (long long d : dims) {
        if (d < 1 || d > (1LL << 31))
            bad_weights(layer + ": a layer size in the architecture is " + std::to_string(d));
        if (n > ((size_t)1 << 40) / (size_t)d) bad_weights(layer + ": layer sizes too large");
        n *= (size_t)d;
    }
    return n;
}

// A tensor the layer needs, with exactly n values.
template <class T>
inline void need_values(const std::vector<T>& v, size_t n, const std::string& key) {
    if (v.empty()) bad_weights(key + " is missing");
    if (v.size() != n)
        bad_weights(key + " has " + std::to_string(v.size()) + " values; the layer needs " + std::to_string(n));
}

// An optional tensor (a bias): absent, or exactly n values.
template <class T>
inline void allow_values(const std::vector<T>& v, size_t n, const std::string& key) {
    if (!v.empty()) need_values(v, n, key);
}

// At run time: a weight array indexed by an input's size (channels, token ids, positions) must cover it.
template <class T>
inline void covers(const std::vector<T>& v, size_t n, const std::string& layer, const char* what) {
    if (v.size() < n)
        throw std::runtime_error("model weights do not match the input: " + layer + "." + what + " has " +
                                 std::to_string(v.size()) + " values, the input needs " + std::to_string(n));
}

// Reads a float32 tensor by name; returns false if not found.
inline bool read_f32_tensor(const ReadTensorFn& read, const std::string& key,
                             std::vector<float>& out)
{
    std::vector<int32_t> shape;
    std::vector<uint8_t> data;
    if (!read(key, shape, data)) return false;
    tensor_values(data, out, key);
    return true;
}

// Reads a tensor as float32, with automatic dequantization if an _i8 + w_scales pair exists.
inline bool read_f32_or_i8_tensor(const ReadTensorFn& read, const std::string& key,
                                   std::vector<float>& out)
{
    std::string scales_key = key.substr(0, key.rfind('.') + 1) + "w_scales";
    std::vector<int32_t> shape;
    std::vector<uint8_t> data_i8, data_sc;
    if (read(key + "_i8", shape, data_i8) && read(scales_key, shape, data_sc)) {
        std::vector<float> sc;
        tensor_values(data_sc, sc, scales_key);
        const int8_t*  wi8 = reinterpret_cast<const int8_t*>(data_i8.data());
        size_t total = data_i8.size();
        size_t C_out = sc.size();
        if (C_out == 0 || total % C_out != 0)
            bad_weights(key + "_i8 has " + std::to_string(total) + " values, not a whole number of rows for " +
                        std::to_string(C_out) + " scales");
        size_t K     = total / C_out;
        out.resize(total);
        for (size_t c = 0; c < C_out; ++c)
            for (size_t k = 0; k < K; ++k)
                out[c * K + k] = wi8[c * K + k] * sc[c];
        return true;
    }
    return read_f32_tensor(read, key, out);
}

// ── ILayer base class ─────────────────────────────────────────
// All layer types derive from ILayer.  The graph engine calls these
// virtual methods; concrete implementations live in src/layers/.
class ILayer {
public:
    std::string name; // matches the node name in the arch JSON and the tag in the weights binary
    bool relu = false; // whether to fuse a ReLU activation after this layer's output

    virtual ~ILayer() = default;

    // Load weights from the binary via the read callback.  Called once at model load time.
    virtual void load_weights(const ReadTensorFn&) {}

    // Run the layer: inputs[0..n_inputs-1] → out. Returns the output shape.
    // inputs[0] is always the primary input; inputs[1] is the secondary (e.g. skip connection,
    // style tensor, duration tensor). Layers that only use one input may ignore n_inputs > 1.
    virtual Shape forward(const TensorView* inputs, int n_inputs,
                              BufferView out, ScratchPads& sc) = 0;

    // Returns the element dtype of this layer's output tensor.
    virtual DType out_dtype() const = 0;

    // Returns the output shape given a static input shape (used during shape propagation).
    virtual Shape output_shape(Shape in) const = 0;

    // Returns the output shape given all static input shapes.
    // Default delegates to the single-input overload; override when secondary inputs affect shape.
    virtual Shape output_shape(const Shape* inputs, int n) const {
        return output_shape(n > 0 ? inputs[0] : Shape{});
    }

    // Returns the output shape when it depends on the runtime contents of the input buffers
    // (e.g. LengthRegulateFp32 reads duration values from inputs[1]).
    // in_ptrs[i] are host-side pointers (already staged from device if needed).
    // Defaults to output_shape(in_shapes[0]).
    virtual Shape dynamic_output_size(const void* const* in_ptrs,
                                          const Shape* in_shapes,
                                          int n_inputs) const {
        (void)in_ptrs;
        return n_inputs > 0 ? output_shape(in_shapes[0]) : Shape{};
    }

    // Fills out[] with the byte size needed for each Scratch slot given input shape in.
    // Slots that are not needed should be left at 0.
    virtual void scratch_needed(Shape /*in*/, size_t /*out*/[Scratch::N]) const {}

    // Upload weights to device memory after load_weights().  No-op on CPU.
    virtual void upload_weights(Allocator& /*alloc*/) {}

    // Release scratch owned by the layer itself, without touching weights or
    // persistent state. Extreme low-memory mode calls this after a safe sync.
    virtual void release_runtime_scratch() {}

    // ── Stateful layer support (for KV-cache attention layers) ────────────
    // Returns true if this layer holds persistent state between forward() calls.
    virtual bool is_stateful() const { return false; }

    // Allocates / resets persistent state for a sequence of up to max_seq_len tokens.
    // Called by InferenceModel::reset_state() before each new generation sequence.
    // No-op for layers where is_stateful() == false.
    virtual void reset_state(int /*max_seq_len*/, Allocator& /*alloc*/) {}

    // Returns the memory space this layer's forward() expects its buffers to live in.
    // Defaults to the active backend's memory space; CPU-only layers override this to Host.
    virtual MemSpace required_mem_space() const { return active_backend()->mem_space; }
};
