// HIP builds only (on the include path when KT2N_HIP=ON): the CUDA math constants the engine uses.
#pragma once
#include <hip/hip_math_constants.h>
#ifndef CUDART_INF_F
#define CUDART_INF_F HIP_INF_F
#endif
