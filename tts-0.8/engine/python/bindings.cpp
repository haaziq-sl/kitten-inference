/*
 * python/bindings.cpp — pybind11 bindings for the Kitten TTS 0.8 inference engine,
 * built as the extension module kitten_inference.tts08._engine.
 *
 * Exposed:
 *   set_backend(name)       → bool         select "cpu", "cuda", or "metal"
 *   active_backend()        → str          name of current backend
 *   InputSpec               data class     per-input shape/dtype/quant metadata
 *   InferenceModel          class          generic graph-based model
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "engine/model.hpp"
#include "backend/backend.hpp"

namespace py = pybind11;

// ── Helpers ──────────────────────────────────────────────────────────────────

// Ensure an array is C-contiguous float32 and return it.
static py::array_t<float, py::array::c_style>
to_contiguous_f32(py::array arr) {
    return py::array_t<float, py::array::c_style | py::array::forcecast>(arr);
}

// Ensure an array is C-contiguous int8 and return it.
static py::array_t<int8_t, py::array::c_style>
to_contiguous_i8(py::array arr) {
    return py::array_t<int8_t, py::array::c_style | py::array::forcecast>(arr);
}

// Wrap a std::vector<float> as a 1-D numpy array (zero-copy via move).
static py::array_t<float> vec_to_array(std::vector<float> v) {
    auto* data = new std::vector<float>(std::move(v));
    py::capsule free_when_done(data, [](void* p) {
        delete reinterpret_cast<std::vector<float>*>(p);
    });
    return py::array_t<float>(
        { (py::ssize_t)data->size() },
        { sizeof(float) },
        data->data(),
        free_when_done
    );
}

static const InputSpec& first_input(const InferenceModel& model) {
    const auto& inputs = model.inputs();
    if (inputs.empty()) {
        throw std::runtime_error("Model has no declared inputs.");
    }
    return inputs[0];
}

static int input_dim(const InferenceModel& model, int idx) {
    const auto& shape = first_input(model).shape;
    return idx < (int)shape.size() ? shape[idx] : 1;
}

static std::string lowercase(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static int classifier_num_classes_hint(const InferenceModel& model) {
    const std::string name = lowercase(model.model_name());
    if (name.find("resnet") != std::string::npos ||
        name.find("mobilenet") != std::string::npos ||
        name.find("vit") != std::string::npos) {
        return 1000;
    }
    return 0;
}

// ── Module ───────────────────────────────────────────────────────────────────

PYBIND11_MODULE(_engine, m) {
    m.doc() =
        "Native engine for kitten_inference.tts08.\n\n"
        "Quick start::\n\n"
        "    import numpy as np\n"
        "    from kitten_inference.tts08 import _engine as mi\n"
        "    model = mi.InferenceModel('model_defs/resnet50_fp32_arch.json',\n"
        "                              'weights/resnet50_fp32.bin')\n"
        "    logits = model.forward(np.zeros((224, 224, 3), dtype=np.float32))\n";

    // ── Backend selection ─────────────────────────────────────────────────
    m.def("set_backend",
        [](const std::string& name) -> bool {
            return set_active_backend(name.c_str());
        },
        py::arg("name") = "cpu",
        "Select the active backend.\n\n"
        "Args:\n"
        "    name: 'cpu' (default), 'cuda', or 'metal'\n\n"
        "Returns:\n"
        "    True if the backend was found and activated."
    );

    m.def("active_backend",
        []() -> std::string {
            auto* b = active_backend();
            return b ? std::string(b->name) : std::string("");
        },
        "Return the name of the currently active backend ('cpu', 'cuda', or 'metal')."
    );

    m.def("set_num_threads",
        [](int threads) {
            if (threads < 1)
                throw std::invalid_argument("threads must be at least 1");
#ifdef _OPENMP
            omp_set_dynamic(0);
            omp_set_num_threads(threads);
#else
            (void)threads;
#endif
        },
        py::arg("threads"),
        "Set the native CPU thread budget for forward calls made by this Python thread."
    );

    m.def("max_threads",
        []() -> int {
#ifdef _OPENMP
            return omp_get_max_threads();
#else
            return 1;
#endif
        },
        "Return the native CPU thread budget visible to this Python thread."
    );

    // ── InputSpec ─────────────────────────────────────────────────────────
    py::class_<InputSpec>(m, "InputSpec",
        "Descriptor for one model input (shape, dtype, and quantization params).")
        .def_readonly("name",  &InputSpec::name,  "ModelIOTensor name in the arch graph.")
        .def_property_readonly("shape", [](const InputSpec& s) { return s.shape; },
            "Input shape in NHWC order, e.g. [224,224,3] or [80] for flat. "
            "0 in a dimension means variable (supplied at runtime).")
        .def_readonly("scale", &InputSpec::scale,  "INT8 quantization scale.")
        .def_readonly("zp",    &InputSpec::zp,     "INT8 zero point.")
        .def_readonly("dtype", &InputSpec::dtype,  "'int8' or 'float32'.")
        .def("__repr__", [](const InputSpec& s) {
            std::string sh;
            for (int i = 0; i < (int)s.shape.size(); ++i) {
                if (i) sh += ",";
                sh += std::to_string(s.shape[i]);
            }
            return "<InputSpec name='" + s.name +
                   "' shape=[" + sh + "]" +
                   " dtype=" + s.dtype + ">";
        });

    // ── InferenceModel ────────────────────────────────────────────────────
    py::class_<InferenceModel>(m, "InferenceModel",
        "Generic graph-based inference model.\n\n"
        "Loads a JSON architecture file and a W1X1 binary weights file,\n"
        "then exposes a ``forward()`` method for running inference.\n\n"
        "Example (single input, float32 model)::\n\n"
        "    import numpy as np\n"
        "    from kitten_inference.tts08 import _engine as mi\n"
        "    model = mi.InferenceModel('model_defs/resnet50_fp32_arch.json',\n"
        "                              'weights/resnet50_fp32.bin')\n"
        "    x = np.zeros((224, 224, 3), dtype=np.float32)\n"
        "    logits = model.forward(x)\n\n"
        "Example (multi-input)::\n\n"
        "    outputs = model.forward({'x1': arr1, 'x2': arr2})\n"
        "    y1 = outputs['y1']\n")

        .def(py::init([](const std::string& arch_path, const std::string& weights_path) {
                return std::make_unique<InferenceModel>(arch_path, weights_path);
             }),
             py::arg("arch_path"), py::arg("weights_path"),
             "Load model from a JSON arch file and a W1X1 binary weights file.\n\n"
             "Raises RuntimeError if either file cannot be opened.")

        // Single-input forward: numpy array → 1-D numpy array
        .def("forward",
            [](InferenceModel& self, py::array arr) -> py::array_t<float> {
                if (arr.ndim() == 0)
                    throw std::runtime_error("Input array must have at least 1 dimension.");

                const InputSpec& inp0 = self.inputs()[0];
                // Use spec shape directly for fixed inputs; infer variable dims from element count.
                std::vector<int> shape;
                bool fixed = true;
                for (int d : inp0.shape) if (d == 0) { fixed = false; break; }
                if (fixed) {
                    shape = inp0.shape;
                } else {
                    int C = inp0.shape.empty() ? 1 : inp0.shape.back();
                    int W = (int)inp0.shape.size() >= 3 ? inp0.shape[1] : 1;
                    W = std::max(W, 1);
                    int H = (int)(arr.size() / ((size_t)C * W));
                    if (W > 1)      shape = {H, W, C};
                    else if (H > 1) shape = {H, C};
                    else            shape = {C};
                }

                char kind = arr.dtype().kind();
                std::unordered_map<std::string, ModelIOTensor> named;
                // Keep contiguous copies alive until forward() returns
                py::array_t<float,  py::array::c_style> held_f32;
                py::array_t<int8_t, py::array::c_style> held_i8;
                if (kind == 'f') {
                    held_f32 = to_contiguous_f32(arr);
                    named[inp0.name] = ModelIOTensor(held_f32.data(), shape);
                } else if (kind == 'i' || kind == 'b') {
                    held_i8 = to_contiguous_i8(arr);
                    named[inp0.name] = ModelIOTensor(held_i8.data(), shape);
                } else {
                    throw std::runtime_error(
                        "Unsupported array dtype '" + std::string(py::str(arr.dtype())) +
                        "'. Use float32 for FP32 models or int8 for INT8 models.");
                }
                std::unordered_map<std::string, ModelIOTensor> outputs;
                {
                    py::gil_scoped_release release;
                    outputs = self.forward(named);
                }
                std::vector<float> result;
                const auto& out_names = self.output_names();
                if (!out_names.empty()) {
                    auto it = outputs.find(out_names[0]);
                    if (it != outputs.end()) {
                        const ModelIOTensor& ot = it->second;
                        result.assign(ot.begin(), ot.end());
                    }
                }
                return vec_to_array(std::move(result));
            },
            py::arg("input"),
            "Run a single-input forward pass.\n\n"
            "Args:\n"
            "    input: numpy array (float32 for FP32 models, int8 for INT8 models).\n"
            "           Shape is inferred from the arch spec; flat 1-D arrays are accepted.\n\n"
            "Returns:\n"
            "    1-D float32 numpy array (e.g. class logits).")

        // Multi-input / multi-output forward: dict[str, array] → dict[str, array]
        .def("forward",
            [](InferenceModel& self,
               std::unordered_map<std::string, py::array> inputs_py)
               -> std::unordered_map<std::string, py::array_t<float>> {

                // Build name → InputSpec map for shape inference
                std::unordered_map<std::string, const InputSpec*> spec_map;
                for (const auto& s : self.inputs()) spec_map[s.name] = &s;

                std::unordered_map<std::string, ModelIOTensor> named;
                // Keep contiguous copies alive until forward() returns
                std::vector<py::array_t<float, py::array::c_style>> held;
                for (auto& [name, arr] : inputs_py) {
                    auto c = to_contiguous_f32(arr);
                    // Derive shape from arch spec so callers can pass flat 1-D arrays
                    std::vector<int> shape;
                    auto sit = spec_map.find(name);
                    if (sit != spec_map.end()) {
                        const InputSpec& spec = *sit->second;
                        bool fixed = true;
                        for (int d : spec.shape) if (d == 0) { fixed = false; break; }
                        if (fixed) {
                            shape = spec.shape;
                        } else {
                            int C = spec.shape.empty() ? 1 : spec.shape.back();
                            int W = (int)spec.shape.size() >= 3 ? spec.shape[1] : 1;
                            W = std::max(W, 1);
                            int H = (int)(c.size() / ((size_t)C * W));
                            if (W > 1)      shape = {H, W, C};
                            else if (H > 1) shape = {H, C};
                            else            shape = {C};
                        }
                    } else {
                        for (py::ssize_t d : c.request().shape) shape.push_back((int)d);
                    }
                    held.push_back(c);
                    named[name] = ModelIOTensor(held.back().data(), std::move(shape));
                }

                std::unordered_map<std::string, ModelIOTensor> outputs;
                {
                    py::gil_scoped_release release;
                    outputs = self.forward(named);
                }

                std::unordered_map<std::string, py::array_t<float>> result;
                for (auto& [name, ot] : outputs) {
                    std::vector<float> v(ot.begin(), ot.end());
                    result[name] = vec_to_array(std::move(v));
                }
                return result;
            },
            py::arg("inputs"),
            "Run a multi-input / multi-output forward pass.\n\n"
            "Args:\n"
            "    inputs: dict mapping tensor name → numpy array (float32).\n\n"
            "Returns:\n"
            "    dict mapping output tensor name → 1-D float32 numpy array.")

        .def_property_readonly("model_name",  &InferenceModel::model_name,
                               "Model name string from the arch JSON.")
        .def_property_readonly("input_count", &InferenceModel::input_count,
                               "Number of declared model inputs.")
        .def_property_readonly("inputs",      &InferenceModel::inputs,
                               "List of InputSpec descriptors (one per model input).")
        .def_property_readonly("output_names",&InferenceModel::output_names,
                               "List of declared output tensor names.")
        .def_property_readonly("is_fp32",
            [](const InferenceModel& m) {
                return first_input(m).dtype == "float32";
            },
            "True when the first declared input is float32.")
        .def_property_readonly("input_h",
            [](const InferenceModel& m) { return input_dim(m, 0); },
            "First input height in NHWC layout.")
        .def_property_readonly("input_w",
            [](const InferenceModel& m) { return input_dim(m, 1); },
            "First input width in NHWC layout.")
        .def_property_readonly("input_c",
            [](const InferenceModel& m) { return input_dim(m, 2); },
            "First input channel count in NHWC layout.")
        .def_property_readonly("num_classes", &classifier_num_classes_hint,
            "Classifier class-count hint. Returns 1000 for ImageNet classifier models, otherwise 0.")
        .def("op_profile",    &InferenceModel::op_profile,
             "Return per-op accumulated timing as dict[str, float] (milliseconds).")
        .def("layer_profile", &InferenceModel::layer_profile,
             "Return per-layer accumulated timing as dict[str, float] (milliseconds).")
        .def("set_profile_enabled", &InferenceModel::set_profile_enabled,
             py::arg("enabled"),
             "Enable or disable per-op and per-layer profiling.")
        .def_property_readonly("profile_enabled", &InferenceModel::profile_enabled,
                               "True when profiling is enabled.")
        .def("set_speed", &InferenceModel::set_speed,
             py::arg("speed"),
             "Set TTS duration speed. 1.0 is normal, larger is faster, smaller is slower.")
        .def_property_readonly("speed", &InferenceModel::speed,
                               "Current TTS duration speed.")
        .def("reset_profile", &InferenceModel::reset_profile,
             "Clear accumulated profiling timings.")
        .def("__repr__", [](const InferenceModel& m) {
            const InputSpec& inp0 = m.inputs()[0];
            std::string sh;
            for (int i = 0; i < (int)inp0.shape.size(); ++i) {
                if (i) sh += ",";
                sh += std::to_string(inp0.shape[i]);
            }
            return "<InferenceModel name='" + m.model_name() +
                   "' input=[" + sh + "]" +
                   ">";
        });
}
