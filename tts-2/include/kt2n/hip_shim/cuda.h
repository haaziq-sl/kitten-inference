// HIP builds only (on the include path when KT2N_HIP=ON). The CUDA driver API is used only by the green-context
// experiment (src/render/cu_driver.h), which HIP builds compile out.
#pragma once
#include "kt2n/gpu_runtime.h"
