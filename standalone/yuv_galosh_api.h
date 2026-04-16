/*
 * yuv_galosh_api.h  —  Public C API for yuv_galosh_core.c
 * Include this from C++ code; compile yuv_galosh_core.c separately as C.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define GALOSH_YUV_420  420
#define GALOSH_YUV_444  444
#define GALOSH_TR_MAX   4

typedef struct
{
    float sigma_y;
    float sigma_c;
    int   stride_y;
    int   stride_c;
    int   tr;
    int   yuv_format;
    float gamma_curve;
    /* Noise sigma scale factor — corrects MAD underestimation on ISP-NR footage.
     * ISP NR spatially correlates residual noise; Laplacian MAD then
     * underestimates σ, making all GAT thresholds too small.
     * 0.0 or 1.0 = trust MAD estimator as-is.
     * 3.0 = "actual noise is 3× what MAD estimated" (good starting point for
     *        high-ISO night footage that has been through ISP NR).
     * When scale > 1, alpha is zeroed (ISP has removed the Poisson component). */
    float sigma_y_scale;  /* Y  plane σ scale  (0 or 1.0 = auto) */
    float sigma_c_scale;  /* U/V plane σ scale (0 or 1.0 = auto) */
    /* 3D WHT pilot mode (requires n_mc > 0).
     * 0 = current design: temporal-mean pilot + Wiener (fast, robust).
     * 1 = 3D WHT pilot:   full 3D BayesShrink on 8×8×T coefficients (better
     *     temporal texture, slightly more sensitive to MC quality). */
    int use_3dwht;
} galosh_yuv_params_t;

/* Declared extern so C++ can use it; definition lives in yuv_galosh_core.c */
extern const galosh_yuv_params_t GALOSH_YUV_DEFAULTS_INIT;

int galosh_yuv_denoise(
        float *y_out, float *u_out, float *v_out,
        const float *y_in, const float *u_in, const float *v_in,
        int width, int height,
        const float * const *mc_y,
        const float * const *mc_u,
        const float * const *mc_v,
        int n_mc,
        const galosh_yuv_params_t *params);

#ifdef __cplusplus
} /* extern "C" */
#endif
