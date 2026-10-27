#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <unordered_map>
#include "layer.hpp"
#include "buffer.hpp"
#include "op_type.hpp"
#include "memory_mode.hpp"
#include "backend/backend.hpp"

// ──────────────────────────────────────────────
// ModelIOTensor — tensor descriptor with optional ownership
//
// For inputs:  construct with a raw pointer (non-owning); the caller
//              keeps the data alive for the duration of forward().
// For outputs: InferenceModel::forward() returns IOTensors with
//              owned storage accessible via data / as_float() / as_int8().
//
// Layout is always NHWC: shape {H, W, C}, {H, C} (W=1), or {C} (H=W=1).
// ──────────────────────────────────────────────
class InferenceModel;

struct ModelIOTensor {
    const void*      data  = nullptr;   // pointer to raw data
    std::vector<int> shape;             // NHWC: {H,W,C}, {H,C}, or {C}
    std::string      dtype = "float32"; // "float32" | "int8"

    ModelIOTensor() = default;

    // Non-owning input constructors
    ModelIOTensor(const float*  p, std::vector<int> sh)
        : data(p), shape(std::move(sh)), dtype("float32") {}
    ModelIOTensor(const int8_t* p, std::vector<int> sh)
        : data(p), shape(std::move(sh)), dtype("int8") {}
    ModelIOTensor(const void*   p, std::vector<int> sh, std::string dt)
        : data(p), shape(std::move(sh)), dtype(std::move(dt)) {}

    // Generic dimension accessor — returns 1 for out-of-bounds indices
    int    dim(int i) const { return (i < (int)shape.size()) ? shape[i] : 1; }
    size_t numel() const {
        size_t n = 1;
        for (int d : shape) n *= (size_t)d;
        return n;
    }
    size_t elem_bytes() const { return dtype == "int8" ? 1 : 4; }
    size_t nbytes()     const { return numel() * elem_bytes(); }

    // Typed data accessors
    const float*  as_float() const { return static_cast<const float*>(data); }
    const int8_t* as_int8()  const { return static_cast<const int8_t*>(data); }

    // Float iteration helpers (outputs are always float32 from forward())
    const float* begin() const { return as_float(); }
    const float* end()   const { return as_float() + numel(); }
    float  operator[](size_t i) const { return as_float()[i]; }
    size_t size() const { return numel(); }  // alias for range/size checks

    // Move semantics: fix data pointer if it was pointing into owned_
    ModelIOTensor(ModelIOTensor&& o) noexcept
        : shape(std::move(o.shape)), dtype(std::move(o.dtype))
        , owned_(std::move(o.owned_))
    {
        data   = owned_.empty() ? o.data : owned_.data();
        o.data = nullptr;
    }
    ModelIOTensor& operator=(ModelIOTensor&& o) noexcept {
        if (this == &o) return *this;
        shape  = std::move(o.shape);
        dtype  = std::move(o.dtype);
        owned_ = std::move(o.owned_);
        data   = owned_.empty() ? o.data : owned_.data();
        o.data = nullptr;
        return *this;
    }
    ModelIOTensor(const ModelIOTensor& o)
        : shape(o.shape), dtype(o.dtype), owned_(o.owned_)
    {
        data = owned_.empty() ? o.data : owned_.data();
    }
    ModelIOTensor& operator=(const ModelIOTensor& o) {
        if (this == &o) return *this;
        shape = o.shape; dtype = o.dtype; owned_ = o.owned_;
        data  = owned_.empty() ? o.data : owned_.data();
        return *this;
    }

private:
    std::vector<uint8_t> owned_; // non-empty only for model outputs
    friend class InferenceModel;
};

// ──────────────────────────────────────────────
// GraphNode — one node in the computation graph
// Loaded from the JSON arch file.
// ──────────────────────────────────────────────
struct GraphNode {
    std::string name;   // matches ILayer::name in the weights binary
    OpType      op  = OpType::Unknown; // typed op identifier
    std::vector<std::string> inputs;   // input tensor names
    std::vector<std::string> outputs;  // output tensor names
    bool relu = false;  // FP32 path: apply ReLU after op (from args.relu); stored in layer
};

// ──────────────────────────────────────────────
// InputSpec — descriptor for one model input
//
// shape is NHWC: {H, W, C}, {H, C} (W=1), or {C} (H=W=1).
// A dimension value of 0 means variable (supplied at runtime).
// ──────────────────────────────────────────────
struct InputSpec {
    std::string      name  = "x";
    std::vector<int> shape;            // NHWC: {H[, W], C}; 0 = variable dim
    float            scale = 1.0f;
    int              zp    = 0;
    std::string      dtype = "int8";   // "int8" or "float32"

    int dim(int i) const { return i < (int)shape.size() ? shape[i] : 1; }
};

// ──────────────────────────────────────────────
// ModelArch — loaded from JSON arch file
// ──────────────────────────────────────────────
struct ModelArch {
    std::string model;

    // Input tensors
    std::vector<InputSpec> all_inputs;

    // Output
    std::vector<std::string> output_names;

    std::vector<GraphNode> nodes;
};

// ──────────────────────────────────────────────
// LoadedArch — combined result of parsing one model:
//   arch   → graph topology, input/output specs
//   layers → one owned ILayer per node, matched by name
// ──────────────────────────────────────────────
struct LoadedArch {
    ModelArch                            arch;
    std::vector<std::unique_ptr<ILayer>> layers;
};

#include "tensor_pool.hpp"

struct ModelMemoryReport {
    std::string model_name;
    std::string backend_name;
    MemSpace    backend_mem_space = MemSpace::Host;

    size_t persistent_weights_bytes       = 0;
    size_t activation_slots_bytes         = 0;
    size_t scratch_buffers_bytes          = 0;
    size_t layer_local_scratch_bytes      = 0;
    size_t persistent_state_bytes         = 0;
    size_t staging_buffers_bytes          = 0;
    size_t weight_load_staging_peak_bytes = 0;
    size_t last_output_copy_bytes         = 0;
    size_t peak_output_copy_bytes         = 0;
    size_t manual_min_transient_peak_bytes = 0;
    size_t manual_min_activation_peak_bytes = 0;
    size_t manual_min_scratch_peak_bytes = 0;
    size_t manual_min_staging_peak_bytes = 0;
    size_t engine_transient_peak_bytes = 0;
    size_t engine_transient_over_min_bytes = 0;

    int activation_slot_count = 0;
    int active_activation_slots = 0;
    InferenceMemoryMode memory_mode = InferenceMemoryMode::Latency;

    size_t current_tracked_bytes() const {
        return persistent_weights_bytes + activation_slots_bytes
             + scratch_buffers_bytes + layer_local_scratch_bytes
             + persistent_state_bytes + staging_buffers_bytes
             + last_output_copy_bytes;
    }
};

// ──────────────────────────────────────────────
// InferenceModel — loads arch from JSON + weights from binary
// and runs inference on any supported model topology.
// ──────────────────────────────────────────────
class InferenceModel {
public:
    // Throws std::runtime_error on failure.
    InferenceModel(const std::string& arch_path,
                   const std::string& weights_path,
                   InferenceMemoryMode memory_mode = InferenceMemoryMode::Latency);

    // Forward pass: dict of named input tensors → dict of named output tensors.
    // Inputs must be NHWC: shape {H, W, C}, {H, C}, or {C}.
    // Output tensors are NHWC.  Data pointer must remain valid during the call.
    // Variable-size inputs (h=0 in arch) use H from the tensor's shape.
    //
    // step_pos / is_decode: for autoregressive LM inference only.
    //   step_pos=0, is_decode=false → prefill (process full prompt)
    //   step_pos=N, is_decode=true  → decode step at position N
    // Both default to the non-LM case (stateless forward pass).
    std::unordered_map<std::string, ModelIOTensor>
        forward(const std::unordered_map<std::string, ModelIOTensor>& inputs,
                int step_pos = 0, bool is_decode = false);

    int input_count() const { return (int)loaded_arch_.arch.all_inputs.size(); }
    const std::vector<InputSpec>& inputs() const { return loaded_arch_.arch.all_inputs; }
    const std::vector<std::string>& output_names() const { return loaded_arch_.arch.output_names; }
    const std::unordered_map<std::string, double>& op_profile() const { return op_ms_; }
    const std::unordered_map<std::string, double>& layer_profile() const { return layer_ms_; }
    const std::unordered_map<std::string, std::string>& layer_op_types() const { return layer_ops_; }
    void set_profile_enabled(bool enabled) { profiling_enabled_ = enabled; }
    bool profile_enabled() const { return profiling_enabled_; }
    void reset_profile();
    void set_memory_mode(InferenceMemoryMode mode);
    InferenceMemoryMode memory_mode() const { return memory_mode_; }
    void set_speed(float speed);
    float speed() const { return speed_; }

    const std::string& model_name() const { return loaded_arch_.arch.model; }
    void set_memory_report_enabled(bool enabled) {
        memory_report_enabled_ = enabled;
        if (!enabled) {
            last_output_copy_bytes_ = 0;
            peak_output_copy_bytes_ = 0;
        }
    }
    ModelMemoryReport memory_report() const;
    void print_memory_report(const char* context = nullptr, FILE* out = stdout) const;

    // ── Autoregressive generation support ────────────────────────────────
    // Allocate / reset KV caches in all stateful layers.
    // Must be called once before the first prefill/decode sequence.
    void reset_state(int max_seq_len);

private:
    LoadedArch  loaded_arch_; // parsed arch + owned layer instances

    BufferPool  pool_;    // activation buffers, slot assignment, node→layer map
    ScratchPool scratch_; // temporary workspace buffers (im2col, Winograd, attention, ...)

    Allocator* alloc_ = nullptr;  // backend allocator; non-owning pointer to a static singleton
    size_t weight_load_staging_peak_bytes_ = 0;
    size_t last_output_copy_bytes_ = 0;
    size_t peak_output_copy_bytes_ = 0;

    std::unordered_map<std::string, double> op_ms_;    // cumulative time per op type
    std::unordered_map<std::string, double> layer_ms_; // cumulative time per layer name
    std::unordered_map<std::string, std::string> layer_ops_; // layer name -> op type
    bool profiling_enabled_ = false;
    bool memory_report_enabled_ = false;
    InferenceMemoryMode memory_mode_ = InferenceMemoryMode::Latency;
    float speed_ = 1.0f;

    // Resolves concrete input shapes, propagates them through the graph,
    // and grows activation + scratch buffers as needed.  No-op for dims
    // that are already large enough.
    void update_buffers(const std::unordered_map<std::string, ModelIOTensor>& inputs);

    // Runs every graph node in topological order using the given scratch pads.
    void execute_graph(ScratchPads& scratch);

    // Releases non-persistent forward-pass buffers.  Weights and stateful
    // layer caches are intentionally untouched.
    void release_transient_buffers(bool release_activations);

    void reset_forward_memory_accounting();
    void remember_active_tensor_bytes(const std::string& name, size_t bytes);
    void forget_active_tensor_bytes(const std::string& name);
    bool is_graph_output(const std::string& name) const;
    void record_manual_transient_peak(size_t activation_bytes,
                                      size_t scratch_bytes,
                                      size_t staging_bytes);
    void record_engine_transient_peak();
    size_t current_engine_transient_bytes() const;

    // Host-side staging buffers used when a CPU-only layer runs under a GPU backend.
    // One entry per input slot (indexed 0..n_inputs-1), one for the output, and
    // host scratch pads for CPU fallback workspaces.
    Buffer h_stage_ins_[MAX_LAYER_INPUTS];
    Buffer h_stage_out_;
    Buffer h_stage_scratch_[Scratch::N];

    std::unordered_map<std::string, size_t> active_tensor_bytes_;
    size_t active_tensor_total_bytes_ = 0;
    size_t manual_min_transient_peak_bytes_ = 0;
    size_t manual_min_activation_peak_bytes_ = 0;
    size_t manual_min_scratch_peak_bytes_ = 0;
    size_t manual_min_staging_peak_bytes_ = 0;
    size_t engine_transient_peak_bytes_ = 0;
};
