#include "tensor_pool.hpp"
#include "model.hpp"

// ──────────────────────────────────────────────────────────────
// GraphSlots::build — liveness analysis + greedy slot assignment
// ──────────────────────────────────────────────────────────────
GraphSlots GraphSlots::build(const ModelArch& arch) {
    GraphSlots slots;
    const auto& nodes = arch.nodes;

    // ── Liveness analysis ─────────────────────────────────────
    std::unordered_map<std::string, int> last_use;
    for (int i = 0; i < (int)nodes.size(); ++i)
        for (const auto& t : nodes[i].inputs) last_use[t] = i;
    slots.last_use = last_use;

    // ── Greedy slot assignment ────────────────────────────────
    std::vector<int> free_slots;
    int next_slot = 0;

    for (const auto& inp : arch.all_inputs)
        slots.slot_map[inp.name] = next_slot++;

    for (int i = 0; i < (int)nodes.size(); ++i) {
        const GraphNode& node = nodes[i];
        // Allocate output slot BEFORE freeing dead inputs (avoids aliasing)
        int slot;
        if (!free_slots.empty()) { slot = free_slots.back(); free_slots.pop_back(); }
        else { slot = next_slot++; }
        if (!node.outputs.empty()) slots.slot_map[node.outputs[0]] = slot;

        auto maybe_free = [&](const std::string& t) {
            if (t.empty()) return;
            auto it = last_use.find(t);
            if (it != last_use.end() && it->second == i) {
                auto slot_it = slots.slot_map.find(t);
                if (slot_it != slots.slot_map.end()) free_slots.push_back(slot_it->second);
            }
        };
        for (const auto& t : node.inputs) maybe_free(t);
    }

    slots.num_slots = next_slot;
    return slots;
}

// ──────────────────────────────────────────────────────────────
// BufferPool::build — dtype inference + node_layers mapping + resize
// ──────────────────────────────────────────────────────────────
BufferPool BufferPool::build(GraphSlots&&                                    graph_slots,
                              const ModelArch&                               arch,
                              const std::vector<std::unique_ptr<ILayer>>&   layers)
{
    BufferPool pool;
    pool.slots = std::move(graph_slots);
    const auto& nodes = arch.nodes;

    // ── Build node→layer map ──────────────────────────────────
    std::unordered_map<std::string, ILayer*> layer_map;
    for (auto& l : layers) layer_map[l->name] = l.get();
    pool.node_layers.resize(nodes.size(), nullptr);
    for (int node_idx = 0; node_idx < (int)nodes.size(); ++node_idx) {
        auto it = layer_map.find(nodes[node_idx].name);
        if (it != layer_map.end()) pool.node_layers[node_idx] = it->second;
    }

    // ── Per-slot element type ─────────────────────────────────
    pool.slot_dtype.assign(pool.slots.num_slots, DType::int8());
    for (int node_idx = 0; node_idx < (int)nodes.size(); ++node_idx) {
        if (nodes[node_idx].outputs.empty()) continue;
        int slot = pool.slots.slot_map.at(nodes[node_idx].outputs[0]);
        if (pool.node_layers[node_idx])
            pool.slot_dtype[slot] = pool.node_layers[node_idx]->out_dtype();
    }
    for (const auto& inp : arch.all_inputs) {
        int slot = pool.slots.slot_map.at(inp.name);
        if (pool.slot_dtype[slot].is_int8())  // don't override fp32 set by a layer (slot-reuse case)
            pool.slot_dtype[slot] = DType::from_str(inp.dtype);
    }

    pool.buffers.resize(pool.slots.num_slots);
    pool.slot_shapes.resize(pool.slots.num_slots);

    // printf("Buffer slots: %d  (model: %s)\n", pool.slots.num_slots, arch.model.c_str());
    return pool;
}

// ──────────────────────────────────────────────────────────────
// BufferPool::grow — grow-only single-slot allocation
// ──────────────────────────────────────────────────────────────
void BufferPool::grow(int slot, size_t n_elems, Allocator& alloc) {
    size_t need_bytes = n_elems * slot_dtype[slot].elem_bytes();
    if (need_bytes > buffers[slot].bytes())
        buffers[slot] = alloc.make_buffer(need_bytes);
}

// ──────────────────────────────────────────────────────────────
// BufferPool::propagate_shapes — pure shape propagation, no allocation
// ──────────────────────────────────────────────────────────────
PropagateResult BufferPool::propagate_shapes(
    const ModelArch& arch,
    const std::unordered_map<std::string, Shape>& input_shapes) const
{
    PropagateResult result;
    result.slot_sizes.assign(slots.num_slots, 0);
    result.shapes = input_shapes;

    for (const auto& inp : arch.all_inputs) {
        const Shape& shape = result.shapes.at(inp.name);
        int slot = slots.slot_map.at(inp.name);
        result.slot_sizes[slot] = std::max(result.slot_sizes[slot],
                                           (size_t)shape.d0 * (size_t)shape.d1 * (size_t)shape.d2);
    }

    for (int node_idx = 0; node_idx < (int)arch.nodes.size(); ++node_idx) {
        const GraphNode& node = arch.nodes[node_idx];
        int num_inputs = (int)node.inputs.size();
        Shape in_shapes[MAX_LAYER_INPUTS] = {};
        for (int j = 0; j < num_inputs && j < MAX_LAYER_INPUTS; ++j) {
            auto it = result.shapes.find(node.inputs[j]);
            if (it != result.shapes.end()) in_shapes[j] = it->second;
        }

        ILayer* layer = node_layers[node_idx];
        Shape output_shape = layer ? layer->output_shape(in_shapes, num_inputs)
                                   : (num_inputs > 0 ? in_shapes[0] : Shape{});

        if (!node.outputs.empty()) {
            result.shapes[node.outputs[0]] = output_shape;
            int slot = slots.slot_map.at(node.outputs[0]);
            size_t num_elems = (size_t)output_shape.d0 * (size_t)output_shape.d1 * (size_t)output_shape.d2;
            if (num_elems > result.slot_sizes[slot]) result.slot_sizes[slot] = num_elems;
        }
    }

    return result;
}

// ──────────────────────────────────────────────────────────────
// ScratchPool::make_pads — build ScratchPads from current buffers
// ──────────────────────────────────────────────────────────────
ScratchPads ScratchPool::make_pads() const {
    ScratchPads scratch;
    for (int i = 0; i < Scratch::N; ++i)
        scratch.buffers[i] = buffers[i].valid() ? buffers[i].view : BufferView{};
    scratch.stream = stream;
    return scratch;
}

// ──────────────────────────────────────────────────────────────
// ScratchPool::grow — single-slot grow
// ──────────────────────────────────────────────────────────────
void ScratchPool::grow(int i, size_t need, Allocator& alloc) {
    if (need > buffers[i].bytes())
        buffers[i] = alloc.make_buffer(need);
}

// ──────────────────────────────────────────────────────────────
// ScratchPool::grow_for_layer — grow all scratch slots for a layer
// ──────────────────────────────────────────────────────────────
void ScratchPool::grow_for_layer(ILayer* layer, Shape input_shape, Allocator& alloc) {
    size_t sizes[Scratch::N] = {};
    layer->scratch_needed(input_shape, sizes);
    for (int i = 0; i < Scratch::N; ++i)
        grow(i, sizes[i], alloc);
}
