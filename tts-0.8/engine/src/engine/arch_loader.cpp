#include "arch_loader.hpp"
#include "utils/json.hpp"
#include "layers/factory.hpp"
#include <unordered_set>

// Build a compact NHWC shape from {h, w, c} read from JSON.
// Drops trailing W=1 then H=1 unless the dimension is variable (0).
static std::vector<int> make_input_shape(int h, int w, int c) {
    if (w != 1) return {h, w, c};
    if (h != 1) return {h, c};
    return {c};
}

LoadedArch ArchLoader::load(const std::string& path) {
    LoadedArch result;
    ModelArch& arch_   = result.arch;
    auto&      layers_ = result.layers;

    JsonValue root = json_load(path);

    if (root.contains("model"))
        arch_.model = root["model"].as_string();

    // ── Parse inputs ──────────────────────────────────────────
    if (root.contains("inputs") && root["inputs"].size() > 0) {
        const auto& inps = root["inputs"];
        arch_.all_inputs.clear();
        for (size_t ii = 0; ii < inps.size(); ++ii) {
            const auto& inp = inps[ii];
            InputSpec spec;
            if (inp.contains("name"))  spec.name = inp["name"].as_string();
            int height = 1, width = 1, channels = 1;
            if (inp.contains("shape")) {
                const auto& shape_json = inp["shape"];
                height = shape_json[1].as_int(); width = shape_json[2].as_int(); channels = shape_json[3].as_int();
            } else {
                if (inp.contains("h")) height   = inp["h"].as_int();
                if (inp.contains("w")) width    = inp["w"].as_int();
                if (inp.contains("c")) channels = inp["c"].as_int();
            }
            spec.shape = make_input_shape(height, width, channels);
            if (inp.contains("scale")) spec.scale = inp["scale"].as_float();
            if (inp.contains("zp"))    spec.zp    = inp["zp"].as_int();
            if (inp.contains("dtype")) spec.dtype = inp["dtype"].as_string();
            arch_.all_inputs.push_back(spec);
        }
    } else if (root.contains("input")) {
        const auto& inp = root["input"];
        InputSpec spec;
        int height = 1, width = 1, channels = 1;
        if (inp.contains("h")) height   = inp["h"].as_int();
        if (inp.contains("w")) width    = inp["w"].as_int();
        if (inp.contains("c")) channels = inp["c"].as_int();
        spec.shape = make_input_shape(height, width, channels);
        if (inp.contains("scale")) spec.scale = inp["scale"].as_float();
        if (inp.contains("zp"))    spec.zp    = inp["zp"].as_int();
        if (inp.contains("dtype")) spec.dtype = inp["dtype"].as_string();
        arch_.all_inputs.push_back(spec);
    }
    if (arch_.all_inputs.empty()) {
        arch_.all_inputs.push_back(InputSpec{});
    }

    // ── Parse outputs ─────────────────────────────────────────
    if (root.contains("outputs")) {
        const auto& outs = root["outputs"];
        for (size_t i = 0; i < outs.size(); ++i) {
            if (outs[i].contains("name"))
                arch_.output_names.push_back(outs[i]["name"].as_string());
        }
    }

    // ── Parse nodes ───────────────────────────────────────────
    const auto& nodes_j = root["nodes"];
    size_t n = nodes_j.size();
    arch_.nodes.resize(n);

    static const JsonValue empty_args;   // default-constructed Null JsonValue

    for (size_t i = 0; i < n; ++i) {
        const auto& node_json = nodes_j[i];
        GraphNode& graph_node = arch_.nodes[i];
        graph_node.name = node_json["name"].as_string();
        const std::string op_str = node_json["op"].as_string();
        graph_node.op = op_type_from_string(op_str);
        if (graph_node.op == OpType::Unknown)
            fprintf(stderr, "arch_loader: unknown op '%s' in node '%s'\n",
                    op_str.c_str(), graph_node.name.c_str());

        // New format: inputs/outputs as arrays
        if (node_json.contains("inputs")) {
            const auto& arr = node_json["inputs"];
            for (size_t j = 0; j < arr.size(); ++j)
                graph_node.inputs.push_back(arr[j].as_string());
        } else if (node_json.contains("in")) {
            graph_node.inputs.push_back(node_json["in"].as_string());
            if (node_json.contains("in2"))
                graph_node.inputs.push_back(node_json["in2"].as_string());
            if (node_json.contains("in3"))
                graph_node.inputs.push_back(node_json["in3"].as_string());
        }

        if (node_json.contains("outputs")) {
            const auto& arr = node_json["outputs"];
            for (size_t j = 0; j < arr.size(); ++j)
                graph_node.outputs.push_back(arr[j].as_string());
        } else if (node_json.contains("out")) {
            graph_node.outputs.push_back(node_json["out"].as_string());
        }

        // Parse relu at node level (legacy format: relu outside args)
        if (node_json.contains("relu") && !node_json.contains("args")) {
            const auto& relu_val = node_json["relu"];
            graph_node.relu = relu_val.is_bool() ? relu_val.as_bool() : (relu_val.as_int() != 0);
        }

        // Parse relu from args (overrides node-level)
        const JsonValue& args = node_json.contains("args") ? node_json["args"] : empty_args;
        if (args.contains("relu")) {
            const auto& relu_val = args["relu"];
            graph_node.relu = relu_val.is_bool() ? relu_val.as_bool() : (relu_val.as_int() != 0);
        }

        // Create layer via factory — pass full node JSON so factory can read
        // both top-level fields (e.g. num_heads, gelu) and args sub-object.
        auto layer = make_layer(graph_node.name, graph_node.op, node_json, graph_node.relu);
        if (layer) layers_.push_back(std::move(layer));
    }

    // ── Infer output_names if not explicitly specified ────────
    if (arch_.output_names.empty() && !arch_.nodes.empty()) {
        std::unordered_set<std::string> consumed;
        for (const auto& node : arch_.nodes)
            for (const auto& t : node.inputs) consumed.insert(t);
        for (auto it = arch_.nodes.rbegin();
             it != arch_.nodes.rend() && arch_.output_names.empty(); ++it) {
            for (const auto& out : it->outputs)
                if (!consumed.count(out)) arch_.output_names.push_back(out);
        }
        if (arch_.output_names.empty() && !arch_.nodes.back().outputs.empty())
            arch_.output_names.push_back(arch_.nodes.back().outputs[0]);
    }

    return result;
}
