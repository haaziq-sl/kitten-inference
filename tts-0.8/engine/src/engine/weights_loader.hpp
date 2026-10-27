#pragma once
#include <string>
#include <vector>
#include <memory>
#include "layer.hpp"

struct WeightLoadMemoryStats {
    // Largest tensor payload staged by the loader while reading the W1X1 file.
    // This buffer is temporary, but it can still contribute to peak RSS.
    size_t peak_tensor_bytes = 0;
};

// Reads a W1X1 binary weights file and calls load_weights() on each layer
// in the list, matched by the name tag stored in the binary index.
// Throws std::runtime_error on file or format errors.
void load_weights(const std::string& weights_path,
                  std::vector<std::unique_ptr<ILayer>>& layers,
                  WeightLoadMemoryStats* memory_stats = nullptr);
