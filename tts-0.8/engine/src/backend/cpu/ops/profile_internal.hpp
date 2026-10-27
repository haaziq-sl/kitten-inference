#pragma once
#include <chrono>

extern bool g_ops_profile_enabled;

struct OpsProfileCounter {
    double value = 0.0;

    OpsProfileCounter() = default;
    OpsProfileCounter(double v) : value(v) {}

    OpsProfileCounter& operator=(double v) {
        value = v;
        return *this;
    }

    OpsProfileCounter& operator+=(double v) {
        if (g_ops_profile_enabled) value += v;
        return *this;
    }

    operator double() const { return value; }
};

static inline double now_ms() {
    if (!g_ops_profile_enabled) return 0.0;
    using namespace std::chrono;
    return duration<double, std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

extern OpsProfileCounter g_im2col_1x1_ms;
extern OpsProfileCounter g_im2col_3x3_ms;
extern OpsProfileCounter g_im2col_gen_ms;
extern OpsProfileCounter g_gemm_ms;
extern OpsProfileCounter g_add_req_ms;
extern OpsProfileCounter g_maxpool_ms;
extern OpsProfileCounter g_wino_transform_ms;
extern OpsProfileCounter g_wino_gemm_ms;
extern OpsProfileCounter g_wino_output_ms;

// ViT timers
extern OpsProfileCounter g_vit_ln_ms;
extern OpsProfileCounter g_vit_gelu_ms;
extern OpsProfileCounter g_vit_seqgemm_ms;
extern OpsProfileCounter g_vit_attn_ms;

// Depthwise timer
extern OpsProfileCounter g_depthwise_ms;

// LM decode timers
extern OpsProfileCounter g_lm_matvec_ms;   // seqgemm_int8_matvec total
extern OpsProfileCounter g_lm_attn_ms;     // causal_attn_core total
extern OpsProfileCounter g_lm_rmsnorm_ms;  // RMSNorm total

// TTS generator timers
extern OpsProfileCounter g_conv1d_fp32_ms;
extern OpsProfileCounter g_conv1d_int8_ms;
extern OpsProfileCounter g_ada_in1d_ms;
extern OpsProfileCounter g_snake1d_ms;
extern OpsProfileCounter g_layernorm_ms;
extern OpsProfileCounter g_conv_transpose_ms;
extern OpsProfileCounter g_sgemm_total_ms;
