// HIP builds only (on the include path when KT2N_HIP=ON). The engine uses cooperative groups only for
// thread-block clusters, which the portable build compiles out; this keeps the namespace available.
#pragma once
#include "kt2n/gpu_runtime.h"
#include <hip/hip_cooperative_groups.h>
