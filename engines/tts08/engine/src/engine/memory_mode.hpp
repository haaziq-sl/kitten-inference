#pragma once
#include <stdexcept>
#include <string>

enum class InferenceMemoryMode {
    Latency,
    LowMemory,
    ExtremeLowMemory,
};

inline const char* inference_memory_mode_name(InferenceMemoryMode mode) {
    switch (mode) {
    case InferenceMemoryMode::Latency:          return "latency";
    case InferenceMemoryMode::LowMemory:        return "low-memory";
    case InferenceMemoryMode::ExtremeLowMemory: return "extreme-low-memory";
    }
    return "latency";
}

inline bool inference_memory_mode_releases_transients(InferenceMemoryMode mode) {
    return mode == InferenceMemoryMode::LowMemory
        || mode == InferenceMemoryMode::ExtremeLowMemory;
}

inline bool inference_memory_mode_releases_layer_scratch(InferenceMemoryMode mode) {
    return mode == InferenceMemoryMode::ExtremeLowMemory;
}

inline InferenceMemoryMode parse_inference_memory_mode(const std::string& value) {
    if (value == "latency" || value == "fast" || value == "default")
        return InferenceMemoryMode::Latency;
    if (value == "low-memory" || value == "low_memory" || value == "memory" || value == "low")
        return InferenceMemoryMode::LowMemory;
    if (value == "extreme-low-memory" || value == "extreme_low_memory" ||
        value == "extreme-memory" || value == "extreme" ||
        value == "minimum-memory" || value == "minimal" || value == "minimum")
        return InferenceMemoryMode::ExtremeLowMemory;
    throw std::runtime_error("memory mode must be 'latency', 'low-memory', or 'extreme-low-memory'");
}
