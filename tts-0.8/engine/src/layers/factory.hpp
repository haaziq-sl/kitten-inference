#pragma once
#include "engine/layer.hpp"
#include "engine/op_type.hpp"
#include <memory>

// Creates the appropriate ILayer subclass for the given OpType.
// name:      layer name (matches weights binary)
// op:        typed op identifier
// node_json: full JSON node object (used to read args, num_heads, gelu, etc.)
// relu:      apply ReLU after op (FP32 ops only)
// Returns nullptr for OpType::Unknown.
std::unique_ptr<ILayer> make_layer(const std::string& layer_name,
                                    OpType             op,
                                    const JsonValue&   node_json,
                                    bool relu);
