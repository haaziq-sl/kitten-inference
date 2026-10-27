#pragma once
#include <vector>
#include <memory>
#include <array>
#include <string>
#include <unordered_map>
#include "layer.hpp"
#include "buffer.hpp"
#include "backend/backend.hpp"

struct ModelArch;

// ──────────────────────────────────────────────────────────────
// GraphSlots — maps tensor names to reusable buffer slot indices.
//
// A "slot" is a physical buffer index shared by tensors whose live
// ranges do not overlap (greedy linear-scan reuse).  Built once at
// model load time from graph connectivity alone; has no dependency
// on layers, memory, or dtypes.
// We do not want to assign separate memory for each layer.
// Hence, we share the same memory for different layers if they
// are not live at the same time.
// The number of slots equals the maximum number of tensors alive at
// the same time.
// ──────────────────────────────────────────────────────────────
struct GraphSlots {
    // Total number of distinct physical buffer slots needed for this model.
    int num_slots = 0;

    // Maps every tensor name (model inputs + intermediate outputs) to its
    // assigned slot index in [0, num_slots).
    std::unordered_map<std::string, int> slot_map;

    // Last graph node index that consumes each tensor.  Tensors with no entry
    // are not consumed by another node, commonly final graph outputs.
    std::unordered_map<std::string, int> last_use;

    // Runs liveness analysis + greedy slot assignment over arch.
    static GraphSlots build(const ModelArch& arch);
};

// ──────────────────────────────────────────────────────────────
// PropagateResult — output of BufferPool::propagate_shapes().
// Contains everything needed to size buffers without allocating.
// ──────────────────────────────────────────────────────────────
struct PropagateResult {
    // Required element count per slot (not bytes).
    // Multiply by slot_dtype[s].elem_bytes() to get byte size.
    std::vector<size_t> slot_sizes;

    // Concrete Shape for every tensor that appeared during propagation
    // (model inputs + each node output), keyed by tensor name.
    std::unordered_map<std::string, Shape> shapes;
};

// ──────────────────────────────────────────────────────────────
// BufferPool — groups the per-slot activation buffer data.
//
// Built once via BufferPool::build(); slot count and formats are
// fixed after construction.  Buffers may grow via grow() as
// variable-length inputs arrive.
// ──────────────────────────────────────────────────────────────
struct BufferPool {
    // Slot assignment (tensor name → slot index) and total slot count.
    GraphSlots slots;

    // Element dtype (float32 or int8) for each slot in [0, slots.num_slots).
    // Slots are basically a way to reuse buffer memory among different layers. 
    std::vector<DType> slot_dtype;

    // Raw (non-owning) pointer to the ILayer for each node in arch.nodes,
    // in the same order.  nullptr when no layer matched the node name.
    std::vector<ILayer*> node_layers;

    // Owning activation buffers, one per slot.  Invalid until first grow().
    std::vector<Buffer> buffers;

    // Runtime shape for each slot, updated after every node in execute_graph().
    std::vector<Shape> slot_shapes;

    // Constructs a BufferPool: infers dtypes, maps node names to layers,
    // and resizes all vectors.  Does not allocate device memory.
    static BufferPool build(GraphSlots&& graph_slots,
                            const ModelArch& arch,
                            const std::vector<std::unique_ptr<ILayer>>& layers);

    // Returns the raw pointer for slot s.
    void* raw_ptr(int s) { return buffers[s].ptr(); }

    // Ensures slot s holds at least n_elems elements; no-op if large enough.
    void grow(int s, size_t n_elems, Allocator& alloc);

    // Walks the graph in topological order calling output_shape() on each
    // layer to compute element counts and intermediate shapes.
    // Does not touch any allocator.
    PropagateResult propagate_shapes(
        const ModelArch& arch,
        const std::unordered_map<std::string, Shape>& input_shapes) const;
};

// ──────────────────────────────────────────────────────────────
// ScratchPool — temporary workspace buffers for layer execution.
//
// Scratch buffers are indexed by Scratch::I8_A, Scratch::F32_A, etc.
// They can be regrown on a per-node basis during execute_graph().
// ──────────────────────────────────────────────────────────────
struct ScratchPool {
    // ScratchPads buffers, one per Scratch slot.  Invalid until first grow().
    std::array<Buffer, Scratch::N> buffers = {};

    // CUDA/HIP stream handle; nullptr on CPU.
    StreamHandle stream = nullptr;

    // Builds a ScratchPads snapshot from current buffer views + stream.
    // Called once per forward() to produce the sc argument for execute_graph().
    ScratchPads make_pads() const;

    // Ensures scratch slot i has at least `need` bytes; no-op if large enough.
    void grow(int i, size_t need, Allocator& alloc);

    // Calls layer->scratch_needed(input_shape) and grows all slots as required.
    void grow_for_layer(ILayer* layer, Shape input_shape, Allocator& alloc);
};
