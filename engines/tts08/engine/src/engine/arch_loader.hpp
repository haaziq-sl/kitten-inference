#pragma once
#include <string>
#include <vector>
#include <memory>
#include "model.hpp"   // ModelArch, InputSpec, GraphNode
#include "layer.hpp"   // ILayer

class ArchLoader {
public:
    static LoadedArch load(const std::string& path);
};
