/*
 * GALOSH GPU Full-Pipeline — All processing on GPU (ISP-class)
 *
 * Entire GALOSH denoising pipeline runs on GPU compute units:
 *   GAT → σ estimation → dark ref → WHT → denoise → L fullres → reconstruct
 * CPU only does file I/O and OpenCL setup.
 *
 * ISP story: proves GALOSH can run on a single-chip GPU/ISP
 * with no CPU involvement in the signal processing path.
 *
 * 全パイプライン GPU 化: ワンチップ ISP で完結する proof-of-concept.
 * CPU は入出力のみ。信号処理パスは全て GPU カーネルで実行。
 *
 * Usage: rawdenoise_gpu input.bin output.bin width height galosh
 *        strength luma_str chroma_str alpha sigma_sq [opencl_device_idx]
 *
 * Build (MSYS2 UCRT64):
 *   gcc -O3 -march=native -o rawdenoise_gpu.exe rawdenoise_gpu.c \
 *       -lOpenCL -lm
 *
 * Copyright (c) 2026 luxgrain. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
#include <sys/time.h>
#include <sys/stat.h>
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

/* OpenCL */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/* cl_half ↔ float conversion (software) */
static cl_half float_to_cl_half(float f)
{
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int exp = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3FF;
    if(exp <= 0)       return (cl_half)(sign);               /* flush subnormals to 0 */
    if(exp >= 31)      return (cl_half)(sign | 0x7C00);      /* inf */
    return (cl_half)(sign | (exp << 10) | mant);
}

static float cl_half_to_float(cl_half h)
{
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if(exp == 0)
        f = sign | (mant ? ((uint32_t)(mant << 13)) : 0); /* subnormal → flush to 0 (good enough for diag) */
    else if(exp == 31)
        f = sign | 0x7F800000 | (mant << 13); /* inf/nan */
    else
        f = sign | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f, 4);
    return result;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* GALOSH constants — must match galosh_fused.cl */
#define GALOSH_BS          8
#define GALOSH_BP         64
#define GALOSH_STRIDE      4
#define TILE_SIZE         48
#define HIST_BINS       4096
#define HIST_MAX        16.0f
#define REDUCE_WG_SIZE   256
#define N_REDUCE_WG       64
#define GAT_LUT_SIZE    4096

/* Params buffer layout indices */
#define P_SIGMA_CH0      0
#define P_SIGMA_CH1      1
#define P_SIGMA_CH2      2
#define P_SIGMA_CH3      3
#define P_UNIFIED_SIGMA  4
#define P_INV_SG         5
#define P_DARK_REF0      6
#define P_DARK_REF1      7
#define P_DARK_REF2      8
#define P_DARK_REF3      9
#define P_S_SCALE       10
#define P_LUMA_STR      11
#define P_CHROMA_STR    12
#define P_ALPHA         13
#define P_SIGMA_SQ      14
/* NOTE: PARAMS_SIZE bumped from 16 → 32 to accommodate Y-GAT mode's
 * per-plane chroma parameters and bivariate Wiener config. Bayer path
 * ignores indices ≥ 16. JP: Y-GAT モード用に chroma α/σ² 等を収容する
 * ため 16→32 に拡張。Bayer パスは idx 16+ を無視する。 */
#define PARAMS_SIZE     32

/* Y-GAT mode extended params (idx 16..23) */
#define P_YG_ALPHA_CB     16
#define P_YG_SIGMA_SQ_CB  17
#define P_YG_ALPHA_CR     18
#define P_YG_SIGMA_SQ_CR  19
#define P_YG_SIGMA_Y      20
#define P_YG_SIGMA_CB     21
#define P_YG_SIGMA_CR     22
#define P_YG_EPS_BIV      23

/* Memory allocation */
static inline float *alloc_float(size_t n) {
    size_t sz = sizeof(float) * ((n + 15) & ~15);
#ifdef _WIN32
    return (float *)_aligned_malloc(sz, 64);
#else
    return (float *)aligned_alloc(64, sz);
#endif
}
static inline void free_aligned(void *p) {
    if(!p) return;
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}


/* ================================================================
 * OpenCL helpers
 * ================================================================ */

#define CL_CHECK(err, msg) do { \
    if((err) != CL_SUCCESS) { \
        fprintf(stderr, "[CL ERROR] %s: err=%d\n", msg, (int)(err)); \
        goto cl_cleanup; \
    } \
} while(0)

static char *load_kernel_source(const char *filename, size_t *out_len) {
    FILE *f = fopen(filename, "rb");
    if(!f) { fprintf(stderr, "Cannot open %s\n", filename); return NULL; }
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = (char *)malloc(len + 1);
    fread(src, 1, len, f);
    src[len] = 0;
    fclose(f);
    if(out_len) *out_len = len;
    return src;
}

/* Per-kernel profiling events */
#define MAX_PROF_EVENTS 64
static cl_event prof_events[MAX_PROF_EVENTS];
static const char *prof_names[MAX_PROF_EVENTS];
static int prof_count = 0;

static double event_ms(cl_event ev) {
    cl_ulong t0, t1;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, NULL);
    return (double)(t1 - t0) * 1e-6;
}

static void prof_add(cl_event ev, const char *name) {
    if(prof_count < MAX_PROF_EVENTS) {
        prof_events[prof_count] = ev;
        prof_names[prof_count] = name;
        prof_count++;
    }
}

/* Simple 1D kernel dispatch helper */
static int dispatch_1d(cl_command_queue q, cl_kernel k,
                       size_t global, size_t local)
{
    cl_event ev;
    cl_int err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global,
                                        local > 0 ? &local : NULL,
                                        0, NULL, &ev);
    if(err != CL_SUCCESS) {
        fprintf(stderr, "[CL] 1D dispatch failed: %d\n", err);
        return -1;
    }
    prof_add(ev, NULL);
    return 0;
}

/* Named 1D dispatch for profiling */
static int dispatch_1d_named(cl_command_queue q, cl_kernel k,
                             size_t global, size_t local, const char *name)
{
    cl_event ev;
    cl_int err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global,
                                        local > 0 ? &local : NULL,
                                        0, NULL, &ev);
    if(err != CL_SUCCESS) {
        fprintf(stderr, "[CL] 1D dispatch failed: %d\n", err);
        return -1;
    }
    prof_add(ev, name);
    return 0;
}

/* Simple 2D kernel dispatch helper */
static int dispatch_2d(cl_command_queue q, cl_kernel k,
                       size_t gx, size_t gy, size_t lx, size_t ly)
{
    const size_t global[2] = { gx, gy };
    const size_t local[2]  = { lx, ly };
    cl_event ev;
    cl_int err = clEnqueueNDRangeKernel(q, k, 2, NULL, global,
                                        (lx > 0) ? local : NULL,
                                        0, NULL, &ev);
    if(err != CL_SUCCESS) {
        fprintf(stderr, "[CL] 2D dispatch failed: %d\n", err);
        return -1;
    }
    prof_add(ev, NULL);
    return 0;
}

/* Named 2D dispatch for profiling */
static int dispatch_2d_named(cl_command_queue q, cl_kernel k,
                             size_t gx, size_t gy, size_t lx, size_t ly,
                             const char *name)
{
    const size_t global[2] = { gx, gy };
    const size_t local[2]  = { lx, ly };
    cl_event ev;
    cl_int err = clEnqueueNDRangeKernel(q, k, 2, NULL, global,
                                        (lx > 0) ? local : NULL,
                                        0, NULL, &ev);
    if(err != CL_SUCCESS) {
        fprintf(stderr, "[CL] 2D dispatch failed: %d\n", err);
        return -1;
    }
    prof_add(ev, name);
    return 0;
}

/* Tile workgroup dimension: WGD × WGD work-items per tile.
 * Configurable via TILE_WG_DIM compile flag (default 16 → 16×16=256 WIs).
 * Smaller WG (e.g. 8×8=64) reduces register pressure at the cost of
 * fewer threads for latency hiding. Optimal value is device-independent
 * as the kernel adapts via get_local_size().
 *
 * タイルカーネルの WG サイズ。TILE_WG_DIM でコンパイル時指定可能。
 * 小さい WG はレジスタ圧を下げるが、レイテンシ隠蔽スレッド数が減る。
 * カーネルは get_local_size() で動的適応するため全 GPU 共通。 */
#ifndef TILE_WG_DIM
#define TILE_WG_DIM 8
#endif
static const int g_tile_wg_dim = TILE_WG_DIM;

/* Tile dispatch helper for fused/pass2 denoise kernels */
static int dispatch_tile(cl_command_queue q, cl_kernel k,
                         int width, int height)
{
    const int wgd = g_tile_wg_dim;
    const size_t tiles_x = (width  + TILE_SIZE - 1) / TILE_SIZE;
    const size_t tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    const size_t global[2] = { tiles_x * wgd, tiles_y * wgd };
    const size_t local[2]  = { wgd, wgd };
    cl_event ev;
    cl_int err = clEnqueueNDRangeKernel(q, k, 2, NULL, global, local,
                                        0, NULL, &ev);
    if(err != CL_SUCCESS) {
        fprintf(stderr, "[CL] tile dispatch failed: %d\n", err);
        return -1;
    }
    prof_add(ev, NULL);
    return 0;
}

/* Named tile dispatch for profiling */
static int dispatch_tile_named(cl_command_queue q, cl_kernel k,
                               int width, int height, const char *name)
{
    const int wgd = g_tile_wg_dim;
    const size_t tiles_x = (width  + TILE_SIZE - 1) / TILE_SIZE;
    const size_t tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    const size_t global[2] = { tiles_x * wgd, tiles_y * wgd };
    const size_t local[2]  = { wgd, wgd };
    cl_event ev;
    cl_int err = clEnqueueNDRangeKernel(q, k, 2, NULL, global, local,
                                        0, NULL, &ev);
    if(err != CL_SUCCESS) {
        fprintf(stderr, "[CL] tile dispatch failed: %d\n", err);
        return -1;
    }
    prof_add(ev, name);
    return 0;
}

/* Align global size to multiple of local size */
static inline size_t align_up(size_t n, size_t alignment) {
    return ((n + alignment - 1) / alignment) * alignment;
}


/* ================================================================
 * Single-plane mode: denoise a single float32 plane entirely on GPU.
 *
 * All-GPU pipeline (CPU does file I/O only):
 *   GPU K_SP1: σ histogram (Laplacian MAD, single channel)
 *   GPU K_SP2: σ finalize (histogram median → σ, 1/σ)
 *   GPU K_SP3: normalize (float32 / σ → half)
 *   GPU K13:   fused_pass12 (tiled WHT BayesShrink + Wiener)
 *   GPU K_SP4: denormalize (half × σ → float32)
 *
 * Used for YUV GALOSH GPU: Python handles RGB↔YCbCr, calls this per plane.
 * ================================================================ */
static int run_single_plane_gpu(const char *input_file, const char *output_file,
                                int width, int height, float strength,
                                int cl_device_idx)
{
    const size_t npix = (size_t)width * height;
    int ret = 1;

    /* Read input */
    float *plane = alloc_float(npix);
    {
        FILE *f = fopen(input_file, "rb");
        if(!f) { fprintf(stderr, "Cannot open %s\n", input_file); return 1; }
        fread(plane, sizeof(float), npix, f);
        fclose(f);
    }

    double t0 = get_time_ms();

    /* OpenCL setup */
    cl_int err;
    cl_platform_id platforms[8]; cl_uint n_plat = 0;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program prog = NULL;
    cl_kernel k_sigma_hist = NULL, k_sigma_final = NULL;
    cl_kernel k_normalize = NULL, k_denormalize = NULL;
    cl_kernel k_fused = NULL;
    cl_mem plane_buf = NULL, norm_buf = NULL, out_buf = NULL;
    cl_mem hist_buf = NULL, sigma_buf = NULL, result_buf = NULL;

    err = clGetPlatformIDs(8, platforms, &n_plat);
    if(err != CL_SUCCESS) { fprintf(stderr, "[CL] no platforms\n"); goto sp_cleanup; }

    cl_device_id all_devs[32]; int n_dev = 0;
    for(cl_uint p = 0; p < n_plat; p++) {
        cl_device_id devs[8]; cl_uint nd = 0;
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 8, devs, &nd);
        for(cl_uint d = 0; d < nd && n_dev < 32; d++)
            all_devs[n_dev++] = devs[d];
    }
    if(cl_device_idx >= n_dev) { fprintf(stderr, "[CL] device %d not found\n", cl_device_idx); goto sp_cleanup; }

    device = all_devs[cl_device_idx];
    {
        char name[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
        fprintf(stderr, "[SINGLE] device: %s\n", name);
    }
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if(err) { fprintf(stderr, "[CL] context err=%d\n", err); goto sp_cleanup; }
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if(err) { fprintf(stderr, "[CL] queue err=%d\n", err); goto sp_cleanup; }

    /* Load & build kernel source */
    {
        const char *cl_paths[] = {"galosh.cl",
            "C:/Users/luxgrain/GALOSH/standalone/galosh.cl", NULL};
        char *source = NULL; size_t src_len = 0;
        for(int i = 0; cl_paths[i]; i++) {
            source = load_kernel_source(cl_paths[i], &src_len);
            if(source) break;
        }
        if(!source) { fprintf(stderr, "[CL] Cannot find .cl\n"); goto sp_cleanup; }

        char opts[256];
        snprintf(opts, sizeof(opts),
                 "-DGALOSH_STRIDE=%d -DTILE_SIZE=%d -DHIST_BINS=%d -DREDUCE_WG_SIZE=%d",
                 GALOSH_STRIDE, TILE_SIZE, HIST_BINS, REDUCE_WG_SIZE);
        prog = clCreateProgramWithSource(context, 1, (const char **)&source, &src_len, &err);
        free(source);
        if(err) { fprintf(stderr, "[CL] program err=%d\n", err); goto sp_cleanup; }
        err = clBuildProgram(prog, 1, &device, opts, NULL, NULL);
        if(err) {
            char log[4096]; clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
            fprintf(stderr, "[CL] build err:\n%s\n", log); goto sp_cleanup;
        }
    }

    /* Create kernels */
    k_sigma_hist  = clCreateKernel(prog, "galosh_sigma_histogram_single", &err);
    if(err) { fprintf(stderr, "[CL] kernel sigma_hist err=%d\n", err); goto sp_cleanup; }
    k_sigma_final = clCreateKernel(prog, "galosh_sigma_finalize_single", &err);
    if(err) { fprintf(stderr, "[CL] kernel sigma_final err=%d\n", err); goto sp_cleanup; }
    k_normalize   = clCreateKernel(prog, "galosh_normalize_single", &err);
    if(err) { fprintf(stderr, "[CL] kernel normalize err=%d\n", err); goto sp_cleanup; }
    k_denormalize = clCreateKernel(prog, "galosh_denormalize_single", &err);
    if(err) { fprintf(stderr, "[CL] kernel denormalize err=%d\n", err); goto sp_cleanup; }
    k_fused       = clCreateKernel(prog, "galosh_fused_pass12", &err);
    if(err) { fprintf(stderr, "[CL] kernel fused err=%d\n", err); goto sp_cleanup; }

    /* Allocate GPU buffers */
    plane_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, npix * sizeof(float), NULL, &err);
    norm_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, npix * sizeof(cl_half), NULL, &err);
    out_buf    = clCreateBuffer(context, CL_MEM_READ_WRITE, npix * sizeof(cl_half), NULL, &err);
    result_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, npix * sizeof(float), NULL, &err);
    hist_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, HIST_BINS * sizeof(int), NULL, &err);
    sigma_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * sizeof(float), NULL, &err);
    if(err) { fprintf(stderr, "[CL] buffer err=%d\n", err); goto sp_cleanup; }

    /* Upload input */
    clEnqueueWriteBuffer(queue, plane_buf, CL_FALSE, 0,
                         npix * sizeof(float), plane, 0, NULL, NULL);

    /* Zero histogram and output accumulator */
    {
        int izero = 0;
        clEnqueueFillBuffer(queue, hist_buf, &izero, sizeof(int), 0,
                            HIST_BINS * sizeof(int), 0, NULL, NULL);
        cl_half hzero = 0;
        clEnqueueFillBuffer(queue, out_buf, &hzero, sizeof(cl_half), 0,
                            npix * sizeof(cl_half), 0, NULL, NULL);
    }

    double t_gpu_start = get_time_ms();

    /* K_SP1: Sigma histogram */
    {
        const int n_x = (width - 2) / 3 + 1;
        const int n_samp = n_x * height;
        clSetKernelArg(k_sigma_hist, 0, sizeof(cl_mem), &plane_buf);
        clSetKernelArg(k_sigma_hist, 1, sizeof(cl_mem), &hist_buf);
        clSetKernelArg(k_sigma_hist, 2, sizeof(int), &width);
        clSetKernelArg(k_sigma_hist, 3, sizeof(int), &height);
        dispatch_1d_named(queue, k_sigma_hist, align_up(n_samp, 256), 256, "K_SP1 sigma_hist");

        /* K_SP2: Sigma finalize */
        clSetKernelArg(k_sigma_final, 0, sizeof(cl_mem), &hist_buf);
        clSetKernelArg(k_sigma_final, 1, sizeof(cl_mem), &sigma_buf);
        clSetKernelArg(k_sigma_final, 2, sizeof(int), &n_samp);
        dispatch_1d_named(queue, k_sigma_final, 1, 0, "K_SP2 sigma_final");
    }

    /* K_SP3: Normalize float32 → half */
    {
        const int npix_i = (int)npix;
        clSetKernelArg(k_normalize, 0, sizeof(cl_mem), &plane_buf);
        clSetKernelArg(k_normalize, 1, sizeof(cl_mem), &norm_buf);
        clSetKernelArg(k_normalize, 2, sizeof(cl_mem), &sigma_buf);
        clSetKernelArg(k_normalize, 3, sizeof(int), &npix_i);
        dispatch_1d_named(queue, k_normalize, align_up(npix, 256), 256, "K_SP3 normalize");
    }

    /* K13: Fused Pass1+Pass2 */
    clSetKernelArg(k_fused, 0, sizeof(cl_mem), &norm_buf);
    clSetKernelArg(k_fused, 1, sizeof(cl_mem), &out_buf);
    clSetKernelArg(k_fused, 2, sizeof(int), &width);
    clSetKernelArg(k_fused, 3, sizeof(int), &height);
    clSetKernelArg(k_fused, 4, sizeof(float), &strength);
    dispatch_tile_named(queue, k_fused, width, height, "K13 fused_pass12");

    /* K_SP4: Denormalize half → float32 */
    {
        const int npix_i = (int)npix;
        clSetKernelArg(k_denormalize, 0, sizeof(cl_mem), &out_buf);
        clSetKernelArg(k_denormalize, 1, sizeof(cl_mem), &result_buf);
        clSetKernelArg(k_denormalize, 2, sizeof(cl_mem), &sigma_buf);
        clSetKernelArg(k_denormalize, 3, sizeof(int), &npix_i);
        dispatch_1d_named(queue, k_denormalize, align_up(npix, 256), 256, "K_SP4 denormalize");
    }

    clFinish(queue);
    double t_gpu = get_time_ms() - t_gpu_start;

    /* Read back sigma for logging */
    {
        float sp[2];
        clEnqueueReadBuffer(queue, sigma_buf, CL_TRUE, 0, 2 * sizeof(float), sp, 0, NULL, NULL);
        fprintf(stderr, "[SINGLE] sigma=%.6f (GPU Laplacian MAD)\n", sp[0]);
    }

    /* Download denoised float32 output */
    clEnqueueReadBuffer(queue, result_buf, CL_TRUE, 0,
                        npix * sizeof(float), plane, 0, NULL, NULL);

    /* Write output */
    {
        FILE *f = fopen(output_file, "wb");
        if(!f) { fprintf(stderr, "Cannot write %s\n", output_file); goto sp_cleanup; }
        fwrite(plane, sizeof(float), npix, f);
        fclose(f);
    }

    /* Profiling report */
    {
        fprintf(stderr, "\n[SINGLE] ====== PER-KERNEL PROFILING ======\n");
        double total_gpu = 0.0;
        for(int i = 0; i < prof_count; i++) {
            double ms = event_ms(prof_events[i]);
            total_gpu += ms;
            const char *name = prof_names[i] ? prof_names[i] : "(unnamed)";
            fprintf(stderr, "[SINGLE]   %-20s %7.3f ms\n", name, ms);
            clReleaseEvent(prof_events[i]);
        }
        fprintf(stderr, "[SINGLE]   %-20s %7.3f ms\n", "TOTAL (GPU time)", total_gpu);
        fprintf(stderr, "[SINGLE] ==================================\n");
    }

    fprintf(stderr, "[SINGLE] gpu=%.1fms total=%.1fms (%dx%d)\n",
            t_gpu, get_time_ms() - t0, width, height);
    fprintf(stderr, "[GPU_PIPELINE_TIME] %.2f\n", t_gpu);
    ret = 0;

sp_cleanup:
    if(k_sigma_hist)  clReleaseKernel(k_sigma_hist);
    if(k_sigma_final) clReleaseKernel(k_sigma_final);
    if(k_normalize)   clReleaseKernel(k_normalize);
    if(k_denormalize) clReleaseKernel(k_denormalize);
    if(k_fused)       clReleaseKernel(k_fused);
    if(plane_buf)  clReleaseMemObject(plane_buf);
    if(norm_buf)   clReleaseMemObject(norm_buf);
    if(out_buf)    clReleaseMemObject(out_buf);
    if(result_buf) clReleaseMemObject(result_buf);
    if(hist_buf)   clReleaseMemObject(hist_buf);
    if(sigma_buf)  clReleaseMemObject(sigma_buf);
    if(prog)    clReleaseProgram(prog);
    if(queue)   clReleaseCommandQueue(queue);
    if(context) clReleaseContext(context);
    free_aligned(plane);
    return ret;
}


/* ================================================================
 * YUV-GAT mode: linear-domain Y-GAT + Y-driven chroma VST
 *
 * EN: Full GALOSH pipeline for sRGB inputs. Converts sRGB → linear
 *     YCbCr, applies GAT on Y (Poisson-Gauss VST) and linear VST
 *     on Cb/Cr (÷√(α·Y+σ²)), runs LOSH shrinkage via
 *     galosh_fused_pass12 on each stabilised plane, then applies
 *     Makitalo-Foi inverse (Y) / linear inverse (Cb/Cr), bivariate
 *     Wiener coupling on (Cb,Cr), and reconstructs sRGB.
 *
 * JP: sRGB 入力を linear YCbCr に分解し、Y は GAT+LOSH+Makitalo 逆、
 *     Cb/Cr は Y 駆動線形 VST+LOSH+逆 VST + bivariate Wiener で
 *     処理する完全 GPU パイプライン。Foi+Alenius blind estimation で
 *     Y の (α,σ²) を推定し、Cb/Cr パラメータは解析式で導出。
 *
 * 2-source CL build: galosh_fused.cl + galosh_yuv_gat.cl
 * I/O: 3ch sRGB float32 interleaved (H×W×3)
 * ================================================================ */
static int run_yuv_gat_gpu(const char *input_file, const char *output_file,
                           int width, int height,
                           float strength_y, float strength_c,
                           int cl_device_idx)
{
    const size_t npix = (size_t)width * height;
    int ret = 1;
    prof_count = 0;

    /* --- Read 3ch sRGB input --- */
    float *srgb = alloc_float(3 * npix);
    {
        FILE *f = fopen(input_file, "rb");
        if(!f) { fprintf(stderr, "[YUV_GAT] Cannot open %s\n", input_file); return 1; }
        fread(srgb, sizeof(float), 3 * npix, f);
        fclose(f);
    }
    fprintf(stderr, "[YUV_GAT] %dx%d, strength_y=%.3f, strength_c=%.3f\n",
            width, height, strength_y, strength_c);

    double t0 = get_time_ms();

    /* --- OpenCL setup --- */
    cl_int err;
    cl_platform_id platforms[8]; cl_uint n_plat = 0;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program prog = NULL;

    /* Kernel handles (18 kernels) */
    cl_kernel k_srgb2ycbcr = NULL, k_ycbcr2srgb = NULL;
    cl_kernel k_blk_stats = NULL, k_dark_samp = NULL, k_noise_est = NULL;
    cl_kernel k_dark_lap = NULL, k_dark_final = NULL, k_chroma_derive = NULL;
    cl_kernel k_gat_fwd = NULL;
    cl_kernel k_f2h = NULL, k_h2f = NULL;
    cl_kernel k_build_lut = NULL, k_lut_fin = NULL;
    cl_kernel k_makitalo = NULL, k_fused = NULL;
    /* Step 7 (Phase 2a) — Y-guided filter, 4-kernel separable pipeline:
     *   1) moments_x        : 1D x-pass over (Y, Y², Cb, Cr, Y·Cb, Y·Cr)
     *   2) moments_y_and_ab : 1D y-pass + local linear regression → (a, b)
     *   3) apply_x          : 1D x-pass over (a, b) coefficient planes
     *   4) apply_y          : 1D y-pass + output = mean_a·Y + mean_b
     * Reduces per-pixel work from O((2r+1)²)=225 to O(2·(2r+1))=30 at r=7. */
    cl_kernel k_guided_moments_x   = NULL;
    cl_kernel k_guided_moments_yab = NULL;
    cl_kernel k_guided_apply_x     = NULL;
    cl_kernel k_guided_apply_y     = NULL;

    /* Buffer handles */
    cl_mem srgb_buf = NULL, y_buf = NULL, cb_buf = NULL, cr_buf = NULL;
    cl_mem y_stab_buf = NULL, cb_stab_buf = NULL, cr_stab_buf = NULL;
    cl_mem scale_cb_buf = NULL, scale_cr_buf = NULL;
    cl_mem half_in_buf = NULL, half_out_buf = NULL;
    cl_mem cb_biv_buf = NULL, cr_biv_buf = NULL;
    /* Phase 2a separable guided filter: 6 moments_x + 4 apply_x scratch.
     * Float32, npix each — ≈640 MB total at 5326×2998 (desktop-only). */
    cl_mem gf_sum_Y_x   = NULL, gf_sum_YY_x  = NULL;
    cl_mem gf_sum_Cb_x  = NULL, gf_sum_Cr_x  = NULL;
    cl_mem gf_sum_YCb_x = NULL, gf_sum_YCr_x = NULL;
    cl_mem gf_a_cb_x    = NULL, gf_b_cb_x    = NULL;
    cl_mem gf_a_cr_x    = NULL, gf_b_cr_x    = NULL;
    cl_mem blk_mean_buf = NULL, blk_var_buf = NULL;
    cl_mem dark_hist_buf = NULL, lap_hist_buf = NULL;
    cl_mem params_buf = NULL;
    cl_mem lut_d_buf = NULL, lut_x_buf = NULL, lut_params_buf = NULL;

    /* Platform / device enumeration */
    err = clGetPlatformIDs(8, platforms, &n_plat);
    if(err != CL_SUCCESS) { fprintf(stderr, "[YUV_GAT] no platforms\n"); goto yg_cleanup; }

    cl_device_id all_devs[32]; int n_dev = 0;
    for(cl_uint p = 0; p < n_plat; p++) {
        cl_device_id devs[8]; cl_uint nd = 0;
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 8, devs, &nd);
        for(cl_uint d = 0; d < nd && n_dev < 32; d++)
            all_devs[n_dev++] = devs[d];
    }
    if(cl_device_idx >= n_dev) {
        fprintf(stderr, "[YUV_GAT] device %d not found\n", cl_device_idx);
        goto yg_cleanup;
    }
    device = all_devs[cl_device_idx];
    {
        char name[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
        fprintf(stderr, "[YUV_GAT] device: %s\n", name);
    }
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if(err) { fprintf(stderr, "[YUV_GAT] context err=%d\n", err); goto yg_cleanup; }
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if(err) { fprintf(stderr, "[YUV_GAT] queue err=%d\n", err); goto yg_cleanup; }

    /* --- Load & build single-source CL program (galosh.cl) --- */
    {
        const char *cl_paths[] = {"galosh.cl",
            "C:/Users/luxgrain/GALOSH/standalone/galosh.cl", NULL};

        char *source = NULL;
        size_t src_len = 0;
        for(int i = 0; cl_paths[i]; i++) {
            source = load_kernel_source(cl_paths[i], &src_len);
            if(source) break;
        }
        if(!source) {
            fprintf(stderr, "[YUV_GAT] Cannot load galosh.cl\n");
            goto yg_cleanup;
        }

        prog = clCreateProgramWithSource(context, 1, (const char **)&source, &src_len, &err);
        free(source);
        if(err) { fprintf(stderr, "[YUV_GAT] program err=%d\n", err); goto yg_cleanup; }

        char opts[512];
        snprintf(opts, sizeof(opts),
            "-DCL_TARGET_OPENCL_VERSION=120 "
            "-DGALOSH_STRIDE=%d -DTILE_SIZE=%d -DHIST_BINS=%d -DREDUCE_WG_SIZE=%d",
            GALOSH_STRIDE, TILE_SIZE, HIST_BINS, REDUCE_WG_SIZE);
        err = clBuildProgram(prog, 1, &device, opts, NULL, NULL);
        if(err) {
            char log[8192];
            clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
            fprintf(stderr, "[YUV_GAT] build err:\n%s\n", log);
            goto yg_cleanup;
        }
    }

    /* --- Create kernels --- */
#define YG_KERNEL(var, name) do { \
    var = clCreateKernel(prog, name, &err); \
    if(err) { fprintf(stderr, "[YUV_GAT] kernel '%s' err=%d\n", name, err); goto yg_cleanup; } \
} while(0)

    YG_KERNEL(k_srgb2ycbcr,   "galosh_yuv_srgb_to_linear_ycbcr");
    YG_KERNEL(k_ycbcr2srgb,   "galosh_yuv_ycbcr_to_srgb");
    YG_KERNEL(k_blk_stats,    "galosh_yuv_noise_block_stats_Y");
    YG_KERNEL(k_dark_samp,    "galosh_yuv_noise_dark_samp_hist_Y");
    YG_KERNEL(k_noise_est,    "galosh_noise_estimate");       /* from fused.cl */
    YG_KERNEL(k_dark_lap,     "galosh_yuv_noise_dark_lap_hist_Y");
    YG_KERNEL(k_dark_final,   "galosh_yuv_noise_dark_finalize_Y");
    YG_KERNEL(k_chroma_derive,"galosh_yuv_chroma_params_derive");
    YG_KERNEL(k_gat_fwd,      "galosh_yuv_gat_forward_Y");
    YG_KERNEL(k_f2h,          "galosh_yuv_float_to_half");
    YG_KERNEL(k_h2f,          "galosh_yuv_half_to_float");
    YG_KERNEL(k_build_lut,    "galosh_build_inv_lut");        /* from fused.cl */
    YG_KERNEL(k_lut_fin,      "galosh_lut_finalize");         /* from fused.cl */
    YG_KERNEL(k_makitalo,     "galosh_yuv_makitalo_inverse_Y");
    YG_KERNEL(k_fused,        "galosh_fused_pass12");         /* from fused.cl */
    YG_KERNEL(k_guided_moments_x,   "galosh_yuv_guided_moments_x");
    YG_KERNEL(k_guided_moments_yab, "galosh_yuv_guided_moments_y_ab");
    YG_KERNEL(k_guided_apply_x,     "galosh_yuv_guided_apply_x");
    YG_KERNEL(k_guided_apply_y,     "galosh_yuv_guided_apply_y");
#undef YG_KERNEL

    /* --- Allocate GPU buffers --- */
    {
        const size_t fb = npix * sizeof(float);
        const size_t hb = npix * sizeof(cl_half);

        srgb_buf      = clCreateBuffer(context, CL_MEM_READ_WRITE, 3 * fb, NULL, &err);
        y_buf         = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        cb_buf        = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        cr_buf        = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        y_stab_buf    = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        cb_stab_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        cr_stab_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        scale_cb_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        scale_cr_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        half_in_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);
        half_out_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);
        cb_biv_buf    = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        cr_biv_buf    = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        /* Separable guided-filter scratch (6 moments_x + 4 apply_x). */
        gf_sum_Y_x   = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_sum_YY_x  = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_sum_Cb_x  = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_sum_Cr_x  = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_sum_YCb_x = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_sum_YCr_x = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_a_cb_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_b_cb_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_a_cr_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_b_cr_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        if(err) { fprintf(stderr, "[YUV_GAT] alloc plane bufs err=%d\n", err); goto yg_cleanup; }

        const int ne_n_bx = width / 16;
        const int ne_n_by = height / 16;
        const int ne_total = ne_n_bx * ne_n_by;
        blk_mean_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, ne_total * sizeof(float), NULL, &err);
        blk_var_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, ne_total * sizeof(float), NULL, &err);
        dark_hist_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(cl_int), NULL, &err);
        lap_hist_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, 2048 * sizeof(cl_int), NULL, &err);
        if(err) { fprintf(stderr, "[YUV_GAT] alloc noise bufs err=%d\n", err); goto yg_cleanup; }

        params_buf     = clCreateBuffer(context, CL_MEM_READ_WRITE, PARAMS_SIZE * sizeof(float), NULL, &err);
        lut_d_buf      = clCreateBuffer(context, CL_MEM_READ_WRITE, GAT_LUT_SIZE * sizeof(float), NULL, &err);
        lut_x_buf      = clCreateBuffer(context, CL_MEM_READ_WRITE, GAT_LUT_SIZE * sizeof(float), NULL, &err);
        lut_params_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * sizeof(float), NULL, &err);
        if(err) { fprintf(stderr, "[YUV_GAT] alloc lut bufs err=%d\n", err); goto yg_cleanup; }
    }

    /* --- Upload sRGB data + init params --- */
    clEnqueueWriteBuffer(queue, srgb_buf, CL_FALSE, 0,
                         3 * npix * sizeof(float), srgb, 0, NULL, NULL);
    {
        float h_params[PARAMS_SIZE];
        memset(h_params, 0, sizeof(h_params));
        /* P_YG_EPS_BIV retained for parameter-slot compatibility; bivariate
         * Wiener was retired in Step 7 (guided-filter-based chroma path). */
        h_params[P_YG_EPS_BIV] = 1e-3f;
        clEnqueueWriteBuffer(queue, params_buf, CL_FALSE, 0,
                             PARAMS_SIZE * sizeof(float), h_params, 0, NULL, NULL);
    }

    double t_pipe_start = get_time_ms();
    const int npix_i = (int)npix;

    /* ================================================================
     * Phase 1: sRGB → linear YCbCr
     * ================================================================ */
    clSetKernelArg(k_srgb2ycbcr, 0, sizeof(cl_mem), &srgb_buf);
    clSetKernelArg(k_srgb2ycbcr, 1, sizeof(cl_mem), &y_buf);
    clSetKernelArg(k_srgb2ycbcr, 2, sizeof(cl_mem), &cb_buf);
    clSetKernelArg(k_srgb2ycbcr, 3, sizeof(cl_mem), &cr_buf);
    clSetKernelArg(k_srgb2ycbcr, 4, sizeof(int), &npix_i);
    dispatch_1d_named(queue, k_srgb2ycbcr, align_up(npix, 256), 256, "YG1 srgb2ycbcr");

    /* ================================================================
     * Phase 2: Y blind noise estimation (Foi+Alenius 2008)
     * ================================================================ */
    {
        const int ne_n_bx = width / 16;
        const int ne_n_by = height / 16;
        const int ne_total = ne_n_bx * ne_n_by;

        /* 2a: block stats */
        clSetKernelArg(k_blk_stats, 0, sizeof(cl_mem), &y_buf);
        clSetKernelArg(k_blk_stats, 1, sizeof(cl_mem), &blk_mean_buf);
        clSetKernelArg(k_blk_stats, 2, sizeof(cl_mem), &blk_var_buf);
        clSetKernelArg(k_blk_stats, 3, sizeof(int), &width);
        clSetKernelArg(k_blk_stats, 4, sizeof(int), &height);
        clSetKernelArg(k_blk_stats, 5, sizeof(int), &ne_n_bx);
        clSetKernelArg(k_blk_stats, 6, sizeof(int), &ne_n_by);
        dispatch_1d_named(queue, k_blk_stats, align_up(ne_total, 64), 64, "YG2a blk_stats");

        /* 2b: dark sample histogram */
        {
            cl_int zero = 0;
            clEnqueueFillBuffer(queue, dark_hist_buf, &zero, sizeof(cl_int),
                                0, 1024 * sizeof(cl_int), 0, NULL, NULL);
        }
        const int samp_per_row = (width + 2) / 3;
        const int n_rows       = (height + 2) / 3;
        const int n_samples    = n_rows * samp_per_row;

        clSetKernelArg(k_dark_samp, 0, sizeof(cl_mem), &y_buf);
        clSetKernelArg(k_dark_samp, 1, sizeof(cl_mem), &dark_hist_buf);
        clSetKernelArg(k_dark_samp, 2, sizeof(int), &width);
        clSetKernelArg(k_dark_samp, 3, sizeof(int), &height);
        clSetKernelArg(k_dark_samp, 4, sizeof(int), &samp_per_row);
        clSetKernelArg(k_dark_samp, 5, sizeof(int), &n_samples);
        dispatch_1d_named(queue, k_dark_samp, align_up(n_samples, 64), 64, "YG2b dark_samp");

        /* 2c: noise model regression (reused from fused.cl) */
        clSetKernelArg(k_noise_est, 0, sizeof(cl_mem), &blk_mean_buf);
        clSetKernelArg(k_noise_est, 1, sizeof(cl_mem), &blk_var_buf);
        clSetKernelArg(k_noise_est, 2, sizeof(cl_mem), &dark_hist_buf);
        clSetKernelArg(k_noise_est, 3, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_noise_est, 4, sizeof(int), &ne_total);
        clSetKernelArg(k_noise_est, 5, sizeof(int), &n_samples);
        dispatch_1d_named(queue, k_noise_est, 64, 64, "YG2c noise_est");

        /* 2d: dark Laplacian histogram (needs dark_max from 2c) */
        {
            cl_int zero = 0;
            clEnqueueFillBuffer(queue, lap_hist_buf, &zero, sizeof(cl_int),
                                0, 2048 * sizeof(cl_int), 0, NULL, NULL);
        }
        const int pos_per_row_h = height * (width - 2);
        const int pos_per_ch    = pos_per_row_h + (height - 2) * width;

        clSetKernelArg(k_dark_lap, 0, sizeof(cl_mem), &y_buf);
        clSetKernelArg(k_dark_lap, 1, sizeof(cl_mem), &lap_hist_buf);
        clSetKernelArg(k_dark_lap, 2, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_dark_lap, 3, sizeof(int), &width);
        clSetKernelArg(k_dark_lap, 4, sizeof(int), &height);
        clSetKernelArg(k_dark_lap, 5, sizeof(int), &pos_per_row_h);
        clSetKernelArg(k_dark_lap, 6, sizeof(int), &pos_per_ch);
        dispatch_1d_named(queue, k_dark_lap, align_up(pos_per_ch, 64), 64, "YG2d dark_lap");

        /* 2e: dark finalize */
        clSetKernelArg(k_dark_final, 0, sizeof(cl_mem), &lap_hist_buf);
        clSetKernelArg(k_dark_final, 1, sizeof(cl_mem), &params_buf);
        dispatch_1d_named(queue, k_dark_final, 1, 0, "YG2e dark_final");
    }

    /* Readback α/σ² — only 2 scalar floats, needed as explicit kernel args */
    float alpha_y, sigma_sq_y;
    {
        clFinish(queue);
        float est[PARAMS_SIZE];
        clEnqueueReadBuffer(queue, params_buf, CL_TRUE, 0,
                            PARAMS_SIZE * sizeof(float), est, 0, NULL, NULL);
        alpha_y    = est[P_ALPHA];
        sigma_sq_y = est[P_SIGMA_SQ];
        fprintf(stderr, "[YUV_GAT] Blind est: alpha=%.6f sigma_sq=%.8f\n",
                alpha_y, sigma_sq_y);
    }

    /* ================================================================
     * Phase 3: Chroma params derive (analytical from Y)
     * ================================================================ */
    clSetKernelArg(k_chroma_derive, 0, sizeof(cl_mem), &params_buf);
    dispatch_1d_named(queue, k_chroma_derive, 1, 0, "YG3 chroma_derive");

    /* ================================================================
     * Phase 4: Y path — GAT → fused LOSH → Makitalo inverse
     * ================================================================ */
    /* 4a: GAT forward */
    clSetKernelArg(k_gat_fwd, 0, sizeof(cl_mem), &y_buf);
    clSetKernelArg(k_gat_fwd, 1, sizeof(cl_mem), &y_stab_buf);
    clSetKernelArg(k_gat_fwd, 2, sizeof(cl_mem), &params_buf);
    clSetKernelArg(k_gat_fwd, 3, sizeof(int), &npix_i);
    dispatch_1d_named(queue, k_gat_fwd, align_up(npix, 256), 256, "YG4a gat_fwd");

    /* 4b: float → half */
    clSetKernelArg(k_f2h, 0, sizeof(cl_mem), &y_stab_buf);
    clSetKernelArg(k_f2h, 1, sizeof(cl_mem), &half_in_buf);
    clSetKernelArg(k_f2h, 2, sizeof(int), &npix_i);
    dispatch_1d_named(queue, k_f2h, align_up(npix, 256), 256, "YG4b f2h_Y");

    /* 4c: build inverse GAT LUT */
    clSetKernelArg(k_build_lut, 0, sizeof(cl_mem), &lut_d_buf);
    clSetKernelArg(k_build_lut, 1, sizeof(cl_mem), &lut_x_buf);
    clSetKernelArg(k_build_lut, 2, sizeof(float), &alpha_y);
    clSetKernelArg(k_build_lut, 3, sizeof(float), &sigma_sq_y);
    dispatch_1d_named(queue, k_build_lut, GAT_LUT_SIZE, 256, "YG4c build_lut");

    /* 4d: LUT finalize */
    clSetKernelArg(k_lut_fin, 0, sizeof(cl_mem), &lut_d_buf);
    clSetKernelArg(k_lut_fin, 1, sizeof(cl_mem), &lut_params_buf);
    clSetKernelArg(k_lut_fin, 2, sizeof(float), &alpha_y);
    clSetKernelArg(k_lut_fin, 3, sizeof(float), &sigma_sq_y);
    dispatch_1d_named(queue, k_lut_fin, 1, 0, "YG4d lut_fin");

    /* 4e: zero half_out accumulator */
    {
        cl_half hzero = 0;
        clEnqueueFillBuffer(queue, half_out_buf, &hzero, sizeof(cl_half), 0,
                            npix * sizeof(cl_half), 0, NULL, NULL);
    }

    /* 4f: fused pass12 (LOSH shrinkage on Y) */
    clSetKernelArg(k_fused, 0, sizeof(cl_mem), &half_in_buf);
    clSetKernelArg(k_fused, 1, sizeof(cl_mem), &half_out_buf);
    clSetKernelArg(k_fused, 2, sizeof(int), &width);
    clSetKernelArg(k_fused, 3, sizeof(int), &height);
    clSetKernelArg(k_fused, 4, sizeof(float), &strength_y);
    dispatch_tile_named(queue, k_fused, width, height, "YG4f fused_Y");

    /* 4g: half → float */
    clSetKernelArg(k_h2f, 0, sizeof(cl_mem), &half_out_buf);
    clSetKernelArg(k_h2f, 1, sizeof(cl_mem), &y_stab_buf);
    clSetKernelArg(k_h2f, 2, sizeof(int), &npix_i);
    dispatch_1d_named(queue, k_h2f, align_up(npix, 256), 256, "YG4g h2f_Y");

    /* 4h: Makitalo-Foi inverse GAT */
    clSetKernelArg(k_makitalo, 0, sizeof(cl_mem), &y_stab_buf);
    clSetKernelArg(k_makitalo, 1, sizeof(cl_mem), &y_buf);   /* overwrite with denoised Y */
    clSetKernelArg(k_makitalo, 2, sizeof(cl_mem), &lut_d_buf);
    clSetKernelArg(k_makitalo, 3, sizeof(cl_mem), &lut_x_buf);
    clSetKernelArg(k_makitalo, 4, sizeof(cl_mem), &lut_params_buf);
    clSetKernelArg(k_makitalo, 5, sizeof(int), &npix_i);
    dispatch_1d_named(queue, k_makitalo, align_up(npix, 256), 256, "YG4h makitalo");

    /* ================================================================
     * Phase 5: Y-guided filter on (Cb, Cr) (Step 7).
     *
     * Two-kernel guided filter (He 2013):
     *   Kernel 1: compute per-pixel (a_Cb, b_Cb, a_Cr, b_Cr) via local
     *             linear regression C ≈ a·Y + b within window.
     *   Kernel 2: box-filter (a, b) coefficient planes and produce
     *             output = mean(a)·Y + mean(b).
     *
     * ε = strength_c² · (α_C·Y + σ²_C)   (Poisson-Gauss, user-scalable)
     *
     * Coefficient planes reuse existing scratch buffers:
     *   cb_stab_buf  ← a_Cb
     *   scale_cb_buf ← b_Cb
     *   cr_stab_buf  ← a_Cr
     *   scale_cr_buf ← b_Cr
     * Final output lands in cb_biv_buf / cr_biv_buf for ycbcr2srgb.
     * ================================================================ */
    /* Step 1: moments_x — 1D x-pass writes 6 intermediate sum buffers. */
    clSetKernelArg(k_guided_moments_x,  0, sizeof(cl_mem), &y_buf);
    clSetKernelArg(k_guided_moments_x,  1, sizeof(cl_mem), &cb_buf);
    clSetKernelArg(k_guided_moments_x,  2, sizeof(cl_mem), &cr_buf);
    clSetKernelArg(k_guided_moments_x,  3, sizeof(cl_mem), &gf_sum_Y_x);
    clSetKernelArg(k_guided_moments_x,  4, sizeof(cl_mem), &gf_sum_YY_x);
    clSetKernelArg(k_guided_moments_x,  5, sizeof(cl_mem), &gf_sum_Cb_x);
    clSetKernelArg(k_guided_moments_x,  6, sizeof(cl_mem), &gf_sum_Cr_x);
    clSetKernelArg(k_guided_moments_x,  7, sizeof(cl_mem), &gf_sum_YCb_x);
    clSetKernelArg(k_guided_moments_x,  8, sizeof(cl_mem), &gf_sum_YCr_x);
    clSetKernelArg(k_guided_moments_x,  9, sizeof(int),    &width);
    clSetKernelArg(k_guided_moments_x, 10, sizeof(int),    &height);
    dispatch_2d_named(queue, k_guided_moments_x,
                      align_up(width, 16), align_up(height, 16),
                      16, 16, "YG5a moments_x");

    /* Step 2: moments_y + (a, b) — 1D y-pass + local linear regression. */
    clSetKernelArg(k_guided_moments_yab,  0, sizeof(cl_mem), &gf_sum_Y_x);
    clSetKernelArg(k_guided_moments_yab,  1, sizeof(cl_mem), &gf_sum_YY_x);
    clSetKernelArg(k_guided_moments_yab,  2, sizeof(cl_mem), &gf_sum_Cb_x);
    clSetKernelArg(k_guided_moments_yab,  3, sizeof(cl_mem), &gf_sum_Cr_x);
    clSetKernelArg(k_guided_moments_yab,  4, sizeof(cl_mem), &gf_sum_YCb_x);
    clSetKernelArg(k_guided_moments_yab,  5, sizeof(cl_mem), &gf_sum_YCr_x);
    clSetKernelArg(k_guided_moments_yab,  6, sizeof(cl_mem), &y_buf);
    clSetKernelArg(k_guided_moments_yab,  7, sizeof(cl_mem), &params_buf);
    clSetKernelArg(k_guided_moments_yab,  8, sizeof(float),  &strength_c);
    clSetKernelArg(k_guided_moments_yab,  9, sizeof(cl_mem), &cb_stab_buf);   /* a_Cb */
    clSetKernelArg(k_guided_moments_yab, 10, sizeof(cl_mem), &scale_cb_buf);  /* b_Cb */
    clSetKernelArg(k_guided_moments_yab, 11, sizeof(cl_mem), &cr_stab_buf);   /* a_Cr */
    clSetKernelArg(k_guided_moments_yab, 12, sizeof(cl_mem), &scale_cr_buf);  /* b_Cr */
    clSetKernelArg(k_guided_moments_yab, 13, sizeof(int),    &width);
    clSetKernelArg(k_guided_moments_yab, 14, sizeof(int),    &height);
    dispatch_2d_named(queue, k_guided_moments_yab,
                      align_up(width, 16), align_up(height, 16),
                      16, 16, "YG5b moments_y_ab");

    /* Step 3: apply_x — 1D x-pass over (a, b) coefficient planes. */
    clSetKernelArg(k_guided_apply_x, 0, sizeof(cl_mem), &cb_stab_buf);
    clSetKernelArg(k_guided_apply_x, 1, sizeof(cl_mem), &scale_cb_buf);
    clSetKernelArg(k_guided_apply_x, 2, sizeof(cl_mem), &cr_stab_buf);
    clSetKernelArg(k_guided_apply_x, 3, sizeof(cl_mem), &scale_cr_buf);
    clSetKernelArg(k_guided_apply_x, 4, sizeof(cl_mem), &gf_a_cb_x);
    clSetKernelArg(k_guided_apply_x, 5, sizeof(cl_mem), &gf_b_cb_x);
    clSetKernelArg(k_guided_apply_x, 6, sizeof(cl_mem), &gf_a_cr_x);
    clSetKernelArg(k_guided_apply_x, 7, sizeof(cl_mem), &gf_b_cr_x);
    clSetKernelArg(k_guided_apply_x, 8, sizeof(int),    &width);
    clSetKernelArg(k_guided_apply_x, 9, sizeof(int),    &height);
    dispatch_2d_named(queue, k_guided_apply_x,
                      align_up(width, 16), align_up(height, 16),
                      16, 16, "YG5c apply_x");

    /* Step 4: apply_y — 1D y-pass + output = mean_a·Y + mean_b. */
    clSetKernelArg(k_guided_apply_y, 0, sizeof(cl_mem), &gf_a_cb_x);
    clSetKernelArg(k_guided_apply_y, 1, sizeof(cl_mem), &gf_b_cb_x);
    clSetKernelArg(k_guided_apply_y, 2, sizeof(cl_mem), &gf_a_cr_x);
    clSetKernelArg(k_guided_apply_y, 3, sizeof(cl_mem), &gf_b_cr_x);
    clSetKernelArg(k_guided_apply_y, 4, sizeof(cl_mem), &y_buf);
    clSetKernelArg(k_guided_apply_y, 5, sizeof(cl_mem), &cb_biv_buf);
    clSetKernelArg(k_guided_apply_y, 6, sizeof(cl_mem), &cr_biv_buf);
    clSetKernelArg(k_guided_apply_y, 7, sizeof(int),    &width);
    clSetKernelArg(k_guided_apply_y, 8, sizeof(int),    &height);
    dispatch_2d_named(queue, k_guided_apply_y,
                      align_up(width, 16), align_up(height, 16),
                      16, 16, "YG5d apply_y");

    cl_mem cb_final = cb_biv_buf;
    cl_mem cr_final = cr_biv_buf;

    /* ================================================================
     * Phase 9: YCbCr → sRGB
     * ================================================================ */
    clSetKernelArg(k_ycbcr2srgb, 0, sizeof(cl_mem), &y_buf);          /* Y_denoised */
    clSetKernelArg(k_ycbcr2srgb, 1, sizeof(cl_mem), &cb_final);       /* Cb final bilateral pass */
    clSetKernelArg(k_ycbcr2srgb, 2, sizeof(cl_mem), &cr_final);       /* Cr final bilateral pass */
    clSetKernelArg(k_ycbcr2srgb, 3, sizeof(cl_mem), &srgb_buf);
    clSetKernelArg(k_ycbcr2srgb, 4, sizeof(int), &npix_i);
    dispatch_1d_named(queue, k_ycbcr2srgb, align_up(npix, 256), 256, "YG9 ycbcr2srgb");

    /* ================================================================
     * Phase 10: Download + Write output
     * ================================================================ */
    clFinish(queue);
    double t_gpu = get_time_ms() - t_pipe_start;

    clEnqueueReadBuffer(queue, srgb_buf, CL_TRUE, 0,
                        3 * npix * sizeof(float), srgb, 0, NULL, NULL);

    {
        FILE *f = fopen(output_file, "wb");
        if(!f) { fprintf(stderr, "[YUV_GAT] Cannot write %s\n", output_file); goto yg_cleanup; }
        fwrite(srgb, sizeof(float), 3 * npix, f);
        fclose(f);
    }

    /* Profiling report */
    {
        fprintf(stderr, "\n[YUV_GAT] ====== PER-KERNEL PROFILING ======\n");
        double total_gpu = 0.0;
        for(int i = 0; i < prof_count; i++) {
            double ms = event_ms(prof_events[i]);
            total_gpu += ms;
            const char *name = prof_names[i] ? prof_names[i] : "(unnamed)";
            fprintf(stderr, "[YUV_GAT]   %-24s %7.3f ms\n", name, ms);
            clReleaseEvent(prof_events[i]);
        }
        fprintf(stderr, "[YUV_GAT]   %-24s %7.3f ms\n", "TOTAL (GPU time)", total_gpu);
        fprintf(stderr, "[YUV_GAT] ==================================\n");
        prof_count = 0;
    }

    fprintf(stderr, "[YUV_GAT] gpu=%.1fms total=%.1fms (%dx%d)\n",
            t_gpu, get_time_ms() - t0, width, height);
    fprintf(stderr, "[GPU_PIPELINE_TIME] %.2f\n", t_gpu);
    ret = 0;

yg_cleanup:
    if(k_srgb2ycbcr)   clReleaseKernel(k_srgb2ycbcr);
    if(k_ycbcr2srgb)   clReleaseKernel(k_ycbcr2srgb);
    if(k_blk_stats)    clReleaseKernel(k_blk_stats);
    if(k_dark_samp)    clReleaseKernel(k_dark_samp);
    if(k_noise_est)    clReleaseKernel(k_noise_est);
    if(k_dark_lap)     clReleaseKernel(k_dark_lap);
    if(k_dark_final)   clReleaseKernel(k_dark_final);
    if(k_chroma_derive)clReleaseKernel(k_chroma_derive);
    if(k_gat_fwd)      clReleaseKernel(k_gat_fwd);
    if(k_f2h)          clReleaseKernel(k_f2h);
    if(k_h2f)          clReleaseKernel(k_h2f);
    if(k_build_lut)    clReleaseKernel(k_build_lut);
    if(k_lut_fin)      clReleaseKernel(k_lut_fin);
    if(k_makitalo)     clReleaseKernel(k_makitalo);
    if(k_fused)        clReleaseKernel(k_fused);

    if(srgb_buf)       clReleaseMemObject(srgb_buf);
    if(y_buf)          clReleaseMemObject(y_buf);
    if(cb_buf)         clReleaseMemObject(cb_buf);
    if(cr_buf)         clReleaseMemObject(cr_buf);
    if(y_stab_buf)     clReleaseMemObject(y_stab_buf);
    if(cb_stab_buf)    clReleaseMemObject(cb_stab_buf);
    if(cr_stab_buf)    clReleaseMemObject(cr_stab_buf);
    if(scale_cb_buf)   clReleaseMemObject(scale_cb_buf);
    if(scale_cr_buf)   clReleaseMemObject(scale_cr_buf);
    if(half_in_buf)    clReleaseMemObject(half_in_buf);
    if(k_guided_moments_x)   clReleaseKernel(k_guided_moments_x);
    if(k_guided_moments_yab) clReleaseKernel(k_guided_moments_yab);
    if(k_guided_apply_x)     clReleaseKernel(k_guided_apply_x);
    if(k_guided_apply_y)     clReleaseKernel(k_guided_apply_y);
    if(half_out_buf)   clReleaseMemObject(half_out_buf);
    if(cb_biv_buf)     clReleaseMemObject(cb_biv_buf);
    if(cr_biv_buf)     clReleaseMemObject(cr_biv_buf);
    if(gf_sum_Y_x)   clReleaseMemObject(gf_sum_Y_x);
    if(gf_sum_YY_x)  clReleaseMemObject(gf_sum_YY_x);
    if(gf_sum_Cb_x)  clReleaseMemObject(gf_sum_Cb_x);
    if(gf_sum_Cr_x)  clReleaseMemObject(gf_sum_Cr_x);
    if(gf_sum_YCb_x) clReleaseMemObject(gf_sum_YCb_x);
    if(gf_sum_YCr_x) clReleaseMemObject(gf_sum_YCr_x);
    if(gf_a_cb_x)    clReleaseMemObject(gf_a_cb_x);
    if(gf_b_cb_x)    clReleaseMemObject(gf_b_cb_x);
    if(gf_a_cr_x)    clReleaseMemObject(gf_a_cr_x);
    if(gf_b_cr_x)    clReleaseMemObject(gf_b_cr_x);
    if(blk_mean_buf)   clReleaseMemObject(blk_mean_buf);
    if(blk_var_buf)    clReleaseMemObject(blk_var_buf);
    if(dark_hist_buf)  clReleaseMemObject(dark_hist_buf);
    if(lap_hist_buf)   clReleaseMemObject(lap_hist_buf);
    if(params_buf)     clReleaseMemObject(params_buf);
    if(lut_d_buf)      clReleaseMemObject(lut_d_buf);
    if(lut_x_buf)      clReleaseMemObject(lut_x_buf);
    if(lut_params_buf) clReleaseMemObject(lut_params_buf);

    if(prog)    clReleaseProgram(prog);
    if(queue)   clReleaseCommandQueue(queue);
    if(context) clReleaseContext(context);
    free_aligned(srgb);
    return ret;
}


/* ================================================================
 * main — 3-way dispatch: single / yuv_gat / galosh (Bayer)
 *
 * CLI:
 *   rawdenoise_gpu <in.bin> <out.bin> <W> <H> single  <strength> [cl_dev]
 *   rawdenoise_gpu <in.bin> <out.bin> <W> <H> yuv_gat <s_y> <s_c> [cl_dev]
 *   rawdenoise_gpu <in.bin> <out.bin> <W> <H> galosh  <strength> <luma_str> <chroma_str> <alpha> <sigma_sq> [cl_dev]
 *
 * Bayer path: fall-through to the legacy body below. The first few
 * locals (input_file/width/height/strength) are declared here so the
 * body (starting with `const float luma_str = atof(argv[7])`) stays
 * unchanged. JP: Bayer 経路は下の既存コードに落ちる。引数の先頭 5 個は
 * ここで宣言しておき、既存本体は無変更で再利用する。
 * ================================================================ */
int main(int argc, char **argv)
{
    if(argc < 6) {
        fprintf(stderr,
            "Usage:\n"
            "  %s <in.bin> <out.bin> <W> <H> single  <strength> [cl_dev]\n"
            "  %s <in.bin> <out.bin> <W> <H> yuv_gat <s_y> <s_c> [cl_dev]\n"
            "  %s <in.bin> <out.bin> <W> <H> galosh  <strength> <luma_str>"
            " <chroma_str> <alpha> <sigma_sq> [cl_dev]\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }
    const char *input_file  = argv[1];
    const char *output_file = argv[2];
    const int   width       = atoi(argv[3]);
    const int   height      = atoi(argv[4]);
    const char *mode        = argv[5];

    if(strcmp(mode, "single") == 0) {
        const float s    = (argc > 6) ? (float)atof(argv[6]) : 1.0f;
        const int   dev  = (argc > 7) ? atoi(argv[7]) : 0;
        return run_single_plane_gpu(input_file, output_file,
                                    width, height, s, dev);
    }
    if(strcmp(mode, "yuv_gat") == 0) {
        if(argc < 8) {
            fprintf(stderr, "yuv_gat mode requires <s_y> <s_c>\n");
            return 1;
        }
        const float sy  = (float)atof(argv[6]);
        const float sc  = (float)atof(argv[7]);
        const int   dev = (argc > 8) ? atoi(argv[8]) : 0;
        return run_yuv_gat_gpu(input_file, output_file,
                               width, height, sy, sc, dev);
    }
    if(strcmp(mode, "galosh") != 0) {
        fprintf(stderr, "Unknown mode '%s' (expected single|yuv_gat|galosh)\n", mode);
        return 1;
    }

    /* --- Bayer galosh mode: fall through to legacy body --- */
    const float strength = (argc > 6) ? (float)atof(argv[6]) : 1.0f;

    /* --- Original Bayer pipeline below --- */
    const float luma_str    = atof(argv[7]);
    const float chroma_str  = atof(argv[8]);
    float alpha       = atof(argv[9]);   /* overridden by GPU blind estimation */
    float sigma_sq    = atof(argv[10]); /* overridden by GPU blind estimation */
    const int cl_device_idx = (argc > 11) ? atoi(argv[11]) : 0;

    const int hw = width / 2, hh = height / 2;
    const size_t npix = (size_t)width * height;
    const size_t chsize = (size_t)hw * hh;

    fprintf(stderr, "[GPU] GALOSH full-pipeline: %dx%d, half=%dx%d, stride=%d, tile=%d\n",
            width, height, hw, hh, GALOSH_STRIDE, TILE_SIZE);
    fprintf(stderr, "[GPU] alpha=%.6f sigma_sq=%.6f luma_str=%.3f chroma_str=%.3f\n",
            alpha, sigma_sq, luma_str, chroma_str);

    /* --- List OpenCL devices --- */
    {
        cl_platform_id plats[8]; cl_uint np = 0;
        clGetPlatformIDs(8, plats, &np);
        fprintf(stderr, "[CL] Available GPU devices:\n");
        int idx = 0;
        for(cl_uint p = 0; p < np; p++) {
            cl_device_id devs[8]; cl_uint nd = 0;
            clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 8, devs, &nd);
            for(cl_uint d = 0; d < nd; d++) {
                char name[256]; cl_uint cu; cl_ulong lm;
                clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(name), name, NULL);
                clGetDeviceInfo(devs[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
                clGetDeviceInfo(devs[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lm), &lm, NULL);
                fprintf(stderr, "  [%d] %s (%u CU, %lu KB LDS)%s\n",
                        idx, name, cu, (unsigned long)(lm/1024),
                        (idx == cl_device_idx) ? " <<< SELECTED" : "");
                idx++;
            }
        }
    }

    double t_total_start = get_time_ms();

    /* --- Read input --- */
    float *raw = alloc_float(npix);
    {
        FILE *f = fopen(input_file, "rb");
        if(!f) { fprintf(stderr, "Cannot open %s\n", input_file); return 1; }
        fread(raw, sizeof(float), npix, f);
        fclose(f);
    }

    /* ================================================================
     * OpenCL setup: platform, device, context, queue, program
     * ================================================================ */
    cl_int err;
    cl_platform_id platforms[8];
    cl_uint n_platforms = 0;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;

    /* Kernel objects — 21 kernels for full pipeline (incl. blind noise estimation) */
    cl_kernel k_noise_block_stats = NULL;
    cl_kernel k_noise_dark_samp_hist = NULL;
    cl_kernel k_noise_estimate = NULL;
    cl_kernel k_noise_dark_lap_hist = NULL;
    cl_kernel k_noise_dark_finalize = NULL;
    cl_kernel k_gat_extract = NULL;
    cl_kernel k_build_inv_lut = NULL;
    cl_kernel k_lut_finalize = NULL;
    cl_kernel k_sigma_histogram = NULL;
    cl_kernel k_sigma_finalize = NULL;
    cl_kernel k_normalize = NULL;
    cl_kernel k_dark_ref_reduce = NULL;
    cl_kernel k_dark_ref_finalize = NULL;
    cl_kernel k_dark_ref_resid_reduce = NULL;
    cl_kernel k_dark_ref_resid_finalize = NULL;
    cl_kernel k_dark_ref_subtract = NULL;
    cl_kernel k_wht_decompose = NULL;
    cl_kernel k_pass1_only = NULL;
    cl_kernel k_fused_pass12 = NULL;
    /* RAW mode chroma guided-filter (C1/C2/C3 plane): reuses yuv kernels. */
    cl_kernel k_raw_chroma_derive      = NULL;
    cl_kernel k_raw_h2f                = NULL;  /* half→float for raw chroma guide */
    cl_kernel k_raw_f2h                = NULL;  /* float→half after apply_y      */
    cl_kernel k_raw_guided_moments_x   = NULL;
    cl_kernel k_raw_guided_moments_yab = NULL;
    cl_kernel k_raw_guided_apply_x     = NULL;
    cl_kernel k_raw_guided_apply_y     = NULL;
    cl_kernel k_raw_guided_loess       = NULL;  /* LOESS (bilateral-weighted) guided filter */
    cl_kernel k_compute_L_fullres = NULL;
    cl_kernel k_pass2_only = NULL;
    cl_kernel k_reconstruct = NULL;
    cl_kernel k_stream_preprocess = NULL;
    cl_kernel k_stream_postprocess = NULL;

    /* GPU buffers */
    cl_mem raw_buf = NULL;
    cl_mem ch0_buf = NULL, ch1_buf = NULL, ch2_buf = NULL, ch3_buf = NULL;
    cl_mem luma_buf = NULL, c1_buf = NULL, c2_buf = NULL, c3_buf = NULL;
    cl_mem l_out_buf = NULL, c1_out_buf = NULL, c2_out_buf = NULL, c3_out_buf = NULL;
    /* Raw-mode chroma guided-filter scratch (6 moments_x + 4 apply_x, each hw*hh). */
    cl_mem rgf_sum_Y_x = NULL, rgf_sum_YY_x = NULL;
    cl_mem rgf_sum_Cb_x = NULL, rgf_sum_Cr_x = NULL;
    cl_mem rgf_sum_YCb_x = NULL, rgf_sum_YCr_x = NULL;
    cl_mem rgf_a_cb_x = NULL, rgf_b_cb_x = NULL;
    cl_mem rgf_a_cr_x = NULL, rgf_b_cr_x = NULL;
    /* Raw mode: luma/c1/c2/c3 are stored as half; guided filter kernels
     * expect float32 — keep 4 converted float buffers here. */
    cl_mem luma_f_buf = NULL, c1_f_buf = NULL, c2_f_buf = NULL, c3_f_buf = NULL;
    cl_mem Lfr_noisy_buf = NULL, Lfr_pilot_buf = NULL, Lfr_den_buf = NULL;
    cl_mem lut_d_buf = NULL, lut_x_buf = NULL, lut_params_buf = NULL;
    cl_mem params_buf = NULL;
    cl_mem hist_buf = NULL;
    cl_mem partial_buf = NULL, partial_resid_buf = NULL;
    cl_mem pilot_tmp_buf = NULL;
    cl_mem blk_mean_buf = NULL, blk_var_buf = NULL;
    cl_mem ne_dark_hist_buf = NULL, ne_lap_hist_buf = NULL;

    double t_cl_setup = get_time_ms();

    err = clGetPlatformIDs(8, platforms, &n_platforms);
    CL_CHECK(err, "getPlatforms");

    /* Enumerate all GPU devices across platforms */
    cl_device_id all_devices[32];
    cl_platform_id device_platform[32];
    int n_devices_total = 0;
    for(cl_uint p = 0; p < n_platforms; p++) {
        cl_device_id devs[8]; cl_uint nd = 0;
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 8, devs, &nd);
        for(cl_uint d = 0; d < nd && n_devices_total < 32; d++) {
            device_platform[n_devices_total] = platforms[p];
            all_devices[n_devices_total] = devs[d];
            n_devices_total++;
        }
    }
    if(n_devices_total == 0) {
        fprintf(stderr, "[CL] No GPU devices found!\n");
        goto cl_cleanup;
    }
    if(cl_device_idx >= n_devices_total) {
        fprintf(stderr, "[CL] Device index %d out of range (max %d)\n",
                cl_device_idx, n_devices_total - 1);
        goto cl_cleanup;
    }

    device = all_devices[cl_device_idx];
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CL_CHECK(err, "createContext");
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err, "createQueue");

    /* Load and build single-source program (galosh.cl = unified RAW + YUV kernels). */
    {
        const char *cl_paths[] = {
            "galosh.cl",
            "C:/Users/luxgrain/GALOSH/standalone/galosh.cl",
            NULL
        };
        char *source = NULL;
        size_t src_len = 0;
        const char *cl_src_path = NULL;
        for(int i = 0; cl_paths[i]; i++) {
            source = load_kernel_source(cl_paths[i], &src_len);
            if(source) { cl_src_path = cl_paths[i]; break; }
        }
        if(!source) {
            fprintf(stderr, "[CL] Cannot find kernel source galosh.cl\n");
            goto cl_cleanup;
        }

        char build_opts[512];
        snprintf(build_opts, sizeof(build_opts),
                 "-DGALOSH_STRIDE=%d -DTILE_SIZE=%d "
                 "-DHIST_BINS=%d -DREDUCE_WG_SIZE=%d",
                 GALOSH_STRIDE, TILE_SIZE, HIST_BINS, REDUCE_WG_SIZE);

        /* Binary cache path — rev'd for single-source merged build so old
         * 2-source caches are invalidated automatically. */
        char cache_path[512];
        snprintf(cache_path, sizeof(cache_path),
                 "%s.unified_s%d_t%d_dev%d.bin",
                 cl_src_path, GALOSH_STRIDE, TILE_SIZE, cl_device_idx);

        int use_cache = 0;
#ifdef _WIN32
        WIN32_FILE_ATTRIBUTE_DATA src_attr, bin_attr;
        if(GetFileAttributesExA(cache_path, GetFileExInfoStandard, &bin_attr) &&
           GetFileAttributesExA(cl_src_path, GetFileExInfoStandard, &src_attr))
        {
            if(CompareFileTime(&bin_attr.ftLastWriteTime, &src_attr.ftLastWriteTime) >= 0)
                use_cache = 1;
        }
#else
        struct stat src_st, bin_st;
        if(stat(cache_path, &bin_st) == 0 && stat(cl_src_path, &src_st) == 0)
            if(bin_st.st_mtime >= src_st.st_mtime) use_cache = 1;
#endif

        double t_build = get_time_ms();

        if(use_cache) {
            size_t bin_len = 0;
            unsigned char *binary = (unsigned char *)load_kernel_source(cache_path, &bin_len);
            if(binary) {
                cl_int bin_status;
                program = clCreateProgramWithBinary(context, 1, &device,
                    &bin_len, (const unsigned char **)&binary, &bin_status, &err);
                free(binary);
                if(err == CL_SUCCESS && bin_status == CL_SUCCESS) {
                    err = clBuildProgram(program, 1, &device, build_opts, NULL, NULL);
                    if(err == CL_SUCCESS) {
                        t_build = get_time_ms() - t_build;
                        fprintf(stderr, "[CL] Loaded cached binary (%.1f ms)\n", t_build);
                        free(source);
                        goto kernel_ready;
                    }
                }
                if(program) { clReleaseProgram(program); program = NULL; }
                fprintf(stderr, "[CL] Cache invalid, recompiling\n");
            }
        }

        /* Compile from source (single-source unified build). */
        program = clCreateProgramWithSource(context, 1, (const char **)&source, &src_len, &err);
        free(source);
        CL_CHECK(err, "createProgram");

        err = clBuildProgram(program, 1, &device, build_opts, NULL, NULL);
        if(err != CL_SUCCESS) {
            char log[8192];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                 sizeof(log), log, NULL);
            fprintf(stderr, "[CL] Build failed:\n%s\n", log);
            goto cl_cleanup;
        }

        t_build = get_time_ms() - t_build;
        fprintf(stderr, "[CL] Compiled from source in %.1f ms\n", t_build);

        /* Save binary cache */
        {
            size_t bin_size = 0;
            clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size, NULL);
            if(bin_size > 0) {
                unsigned char *binary = (unsigned char *)malloc(bin_size);
                clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &binary, NULL);
                FILE *f = fopen(cache_path, "wb");
                if(f) { fwrite(binary, 1, bin_size, f); fclose(f);
                    fprintf(stderr, "[CL] Saved binary cache (%zu bytes)\n", bin_size); }
                free(binary);
            }
        }

    kernel_ready: ;
    }

    /* ================================================================
     * Create all 18 kernels (incl. blind noise estimation)
     * ================================================================ */
    k_noise_block_stats     = clCreateKernel(program, "galosh_noise_block_stats", &err);
    CL_CHECK(err, "kernel noise_block_stats");
    k_noise_dark_samp_hist  = clCreateKernel(program, "galosh_noise_dark_samp_hist", &err);
    CL_CHECK(err, "kernel noise_dark_samp_hist");
    k_noise_estimate        = clCreateKernel(program, "galosh_noise_estimate", &err);
    CL_CHECK(err, "kernel noise_estimate");
    k_noise_dark_lap_hist   = clCreateKernel(program, "galosh_noise_dark_lap_hist", &err);
    CL_CHECK(err, "kernel noise_dark_lap_hist");
    k_noise_dark_finalize   = clCreateKernel(program, "galosh_noise_dark_finalize", &err);
    CL_CHECK(err, "kernel noise_dark_finalize");
    k_gat_extract           = clCreateKernel(program, "galosh_gat_extract", &err);
    CL_CHECK(err, "kernel gat_extract");
    k_build_inv_lut         = clCreateKernel(program, "galosh_build_inv_lut", &err);
    CL_CHECK(err, "kernel build_inv_lut");
    k_lut_finalize          = clCreateKernel(program, "galosh_lut_finalize", &err);
    CL_CHECK(err, "kernel lut_finalize");
    k_sigma_histogram       = clCreateKernel(program, "galosh_sigma_histogram", &err);
    CL_CHECK(err, "kernel sigma_histogram");
    k_sigma_finalize        = clCreateKernel(program, "galosh_sigma_finalize", &err);
    CL_CHECK(err, "kernel sigma_finalize");
    k_normalize             = clCreateKernel(program, "galosh_normalize", &err);
    CL_CHECK(err, "kernel normalize");
    k_dark_ref_reduce       = clCreateKernel(program, "galosh_dark_ref_reduce", &err);
    CL_CHECK(err, "kernel dark_ref_reduce");
    k_dark_ref_finalize     = clCreateKernel(program, "galosh_dark_ref_finalize", &err);
    CL_CHECK(err, "kernel dark_ref_finalize");
    k_dark_ref_resid_reduce = clCreateKernel(program, "galosh_dark_ref_resid_reduce", &err);
    CL_CHECK(err, "kernel dark_ref_resid_reduce");
    k_dark_ref_resid_finalize = clCreateKernel(program, "galosh_dark_ref_resid_finalize", &err);
    CL_CHECK(err, "kernel dark_ref_resid_finalize");
    k_dark_ref_subtract     = clCreateKernel(program, "galosh_dark_ref_subtract", &err);
    CL_CHECK(err, "kernel dark_ref_subtract");
    k_wht_decompose         = clCreateKernel(program, "galosh_wht_decompose", &err);
    CL_CHECK(err, "kernel wht_decompose");
    k_pass1_only            = clCreateKernel(program, "galosh_pass1_only", &err);
    CL_CHECK(err, "kernel pass1_only");
    k_fused_pass12          = clCreateKernel(program, "galosh_fused_pass12", &err);
    /* Raw-mode chroma guided filter kernels (reuse yuv kernels). */
    k_raw_chroma_derive      = clCreateKernel(program, "galosh_yuv_chroma_params_derive", &err);
    k_raw_h2f                = clCreateKernel(program, "galosh_yuv_half_to_float", &err);
    k_raw_f2h                = clCreateKernel(program, "galosh_yuv_float_to_half", &err);
    k_raw_guided_moments_x   = clCreateKernel(program, "galosh_yuv_guided_moments_x", &err);
    k_raw_guided_moments_yab = clCreateKernel(program, "galosh_yuv_guided_moments_y_ab", &err);
    k_raw_guided_apply_x     = clCreateKernel(program, "galosh_yuv_guided_apply_x", &err);
    k_raw_guided_apply_y     = clCreateKernel(program, "galosh_yuv_guided_apply_y", &err);
    k_raw_guided_loess       = clCreateKernel(program, "galosh_yuv_guided_loess", &err);
    CL_CHECK(err, "kernel fused_pass12");
    k_compute_L_fullres     = clCreateKernel(program, "galosh_compute_L_fullres", &err);
    CL_CHECK(err, "kernel compute_L_fullres");
    k_pass2_only            = clCreateKernel(program, "galosh_pass2_only", &err);
    CL_CHECK(err, "kernel pass2_only");
    k_reconstruct           = clCreateKernel(program, "galosh_reconstruct", &err);
    CL_CHECK(err, "kernel reconstruct");
    k_stream_preprocess     = clCreateKernel(program, "galosh_stream_preprocess", &err);
    CL_CHECK(err, "kernel stream_preprocess");
    k_stream_postprocess    = clCreateKernel(program, "galosh_stream_postprocess", &err);
    CL_CHECK(err, "kernel stream_postprocess");

    fprintf(stderr, "[CL] All 24 kernels created\n");

    /* ================================================================
     * Allocate GPU buffers
     * ================================================================ */
    {
        const size_t ch_bytes   = chsize * sizeof(float);
        const size_t full_bytes = npix * sizeof(float);
        const size_t ch_bytes_h   = chsize * sizeof(cl_half);   /* FP16 channel buffers */
        const size_t full_bytes_h = npix * sizeof(cl_half);
        const size_t lut_bytes  = GAT_LUT_SIZE * sizeof(float);
        const size_t hist_bytes = 4 * HIST_BINS * sizeof(cl_int);
        const size_t part_bytes = N_REDUCE_WG * 5 * sizeof(float);
        const size_t resid_bytes = N_REDUCE_WG * 2 * sizeof(float);

        raw_buf       = clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes, NULL, &err);
        CL_CHECK(err, "alloc raw_buf");

        ch0_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes, NULL, &err);
        ch1_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes, NULL, &err);
        ch2_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes, NULL, &err);
        ch3_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes, NULL, &err);
        CL_CHECK(err, "alloc ch_bufs");

        luma_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        c1_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        c2_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        c3_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        CL_CHECK(err, "alloc luma/c_bufs (FP16)");

        l_out_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        c1_out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        c2_out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        c3_out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        /* Guided-filter scratch for raw-mode chroma — all float32, so the
         * buffer size is hw*hh*sizeof(float), NOT ch_bytes_h (which is for
         * half precision = ×1/2 the size). */
        const size_t ch_bytes_f = (size_t)hw * (size_t)hh * sizeof(float);
        rgf_sum_Y_x   = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_sum_YY_x  = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_sum_Cb_x  = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_sum_Cr_x  = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_sum_YCb_x = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_sum_YCr_x = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_a_cb_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_b_cb_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_a_cr_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_b_cr_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        /* Float32 copies of half luma/c1/c2/c3 used by guided filter. */
        luma_f_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        c1_f_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        c2_f_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        c3_f_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        CL_CHECK(err, "alloc out_bufs (FP16)");

        Lfr_noisy_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes_h, NULL, &err);
        Lfr_pilot_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes_h, NULL, &err);
        Lfr_den_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes_h, NULL, &err);
        CL_CHECK(err, "alloc Lfr_bufs (FP16)");

        pilot_tmp_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        CL_CHECK(err, "alloc pilot_tmp_buf (FP16)");

        lut_d_buf      = clCreateBuffer(context, CL_MEM_READ_WRITE, lut_bytes, NULL, &err);
        lut_x_buf      = clCreateBuffer(context, CL_MEM_READ_WRITE, lut_bytes, NULL, &err);
        lut_params_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * sizeof(float), NULL, &err);
        CL_CHECK(err, "alloc lut_bufs");

        params_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, PARAMS_SIZE * sizeof(float), NULL, &err);
        CL_CHECK(err, "alloc params_buf");

        hist_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, hist_bytes, NULL, &err);
        CL_CHECK(err, "alloc hist_buf");

        partial_buf       = clCreateBuffer(context, CL_MEM_READ_WRITE, part_bytes, NULL, &err);
        partial_resid_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, resid_bytes, NULL, &err);
        CL_CHECK(err, "alloc partial_bufs");

        /* Noise estimation buffers */
        const int ne_n_bx = hw / 16;  /* NE_BLOCK_SZ(8) × NE_BLOCK_STEP(2) */
        const int ne_n_by = hh / 16;
        const int ne_total_blocks = 4 * ne_n_bx * ne_n_by;
        const size_t ne_blk_bytes = ne_total_blocks * sizeof(float);
        blk_mean_buf     = clCreateBuffer(context, CL_MEM_READ_WRITE, ne_blk_bytes, NULL, &err);
        blk_var_buf      = clCreateBuffer(context, CL_MEM_READ_WRITE, ne_blk_bytes, NULL, &err);
        ne_dark_hist_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(cl_int), NULL, &err);
        ne_lap_hist_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, 2048 * sizeof(cl_int), NULL, &err);
        CL_CHECK(err, "alloc noise_est_bufs");
    }

    t_cl_setup = get_time_ms() - t_cl_setup;
    fprintf(stderr, "[GPU] CL setup (ctx+build+alloc): %.1f ms\n", t_cl_setup);

    /* ================================================================
     * Upload raw data + initialize params
     * ================================================================ */
    err = clEnqueueWriteBuffer(queue, raw_buf, CL_TRUE, 0,
                               npix * sizeof(float), raw, 0, NULL, NULL);
    CL_CHECK(err, "upload raw");

    /* Initialize params buffer on host — only strengths are known a priori.
     * alpha, sigma_sq, s_scale will be estimated by GPU kernels. */
    {
        float h_params[PARAMS_SIZE];
        memset(h_params, 0, sizeof(h_params));
        h_params[P_LUMA_STR]   = strength * luma_str;
        h_params[P_CHROMA_STR] = strength * chroma_str;
        err = clEnqueueWriteBuffer(queue, params_buf, CL_TRUE, 0,
                                   PARAMS_SIZE * sizeof(float), h_params, 0, NULL, NULL);
        CL_CHECK(err, "upload params");
    }

    /* ================================================================
     * GPU PIPELINE — All processing on GPU
     *
     * CPU does nothing from here until download.
     * Phase 0: blind noise model estimation (Foi 2008)
     * Phase 1: GAT + σ estimation + normalization
     * Phase 2: dark anchor + WHT
     * Phase 3: chroma denoise
     * Phase 4: GALOSH_F (half-res L + full-res L + reconstruct)
     * ================================================================ */
    double t_pipe_start = get_time_ms();

    /* ---- K0a: Block statistics for noise estimation ---- */
    /* ---- K0c: Dark sample histogram (parallel, independent of K0a) ---- */
    {
        const int ne_n_bx = hw / 16;  /* NE_BLOCK_SZ(8) × NE_BLOCK_STEP(2) */
        const int ne_n_by = hh / 16;
        const int ne_total = 4 * ne_n_bx * ne_n_by;

        /* K0a: block stats */
        clSetKernelArg(k_noise_block_stats, 0, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_noise_block_stats, 1, sizeof(cl_mem), &blk_mean_buf);
        clSetKernelArg(k_noise_block_stats, 2, sizeof(cl_mem), &blk_var_buf);
        clSetKernelArg(k_noise_block_stats, 3, sizeof(int), &width);
        clSetKernelArg(k_noise_block_stats, 4, sizeof(int), &height);
        clSetKernelArg(k_noise_block_stats, 5, sizeof(int), &ne_n_bx);
        clSetKernelArg(k_noise_block_stats, 6, sizeof(int), &ne_n_by);
        dispatch_1d_named(queue, k_noise_block_stats, align_up(ne_total, 64), 64, "K0a block_stats");

        /* K0c: dark sample histogram (parallel, stride-3 sampling) */
        const int samp_per_ch_row = (hw + 2) / 3;
        const int samp_per_ch     = ((hh + 2) / 3) * samp_per_ch_row;
        const int n_dark_samples  = 4 * samp_per_ch;

        {
            cl_int zero = 0;
            clEnqueueFillBuffer(queue, ne_dark_hist_buf, &zero, sizeof(cl_int),
                                0, 1024 * sizeof(cl_int), 0, NULL, NULL);
        }
        clSetKernelArg(k_noise_dark_samp_hist, 0, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_noise_dark_samp_hist, 1, sizeof(cl_mem), &ne_dark_hist_buf);
        clSetKernelArg(k_noise_dark_samp_hist, 2, sizeof(int), &width);
        clSetKernelArg(k_noise_dark_samp_hist, 3, sizeof(int), &height);
        clSetKernelArg(k_noise_dark_samp_hist, 4, sizeof(int), &samp_per_ch_row);
        clSetKernelArg(k_noise_dark_samp_hist, 5, sizeof(int), &samp_per_ch);
        clSetKernelArg(k_noise_dark_samp_hist, 6, sizeof(int), &n_dark_samples);
        dispatch_1d_named(queue, k_noise_dark_samp_hist, align_up(n_dark_samples, 64), 64, "K0c dark_samp_hist");

        /* K0b: noise model estimation (needs K0a + K0c results) */
        clSetKernelArg(k_noise_estimate, 0, sizeof(cl_mem), &blk_mean_buf);
        clSetKernelArg(k_noise_estimate, 1, sizeof(cl_mem), &blk_var_buf);
        clSetKernelArg(k_noise_estimate, 2, sizeof(cl_mem), &ne_dark_hist_buf);
        clSetKernelArg(k_noise_estimate, 3, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_noise_estimate, 4, sizeof(int), &ne_total);
        clSetKernelArg(k_noise_estimate, 5, sizeof(int), &n_dark_samples);
        dispatch_1d_named(queue, k_noise_estimate, 64, 64, "K0b noise_estimate");

        /* K0d: dark Laplacian histogram (parallel, needs dark_max from K0b) */
        const int pos_per_ch_h = hh * (hw - 2);
        const int pos_per_ch_v = (hh - 2) * hw;
        const int pos_per_ch   = pos_per_ch_h + pos_per_ch_v;
        const int total_lap_pos = 4 * pos_per_ch;

        {
            cl_int zero = 0;
            clEnqueueFillBuffer(queue, ne_lap_hist_buf, &zero, sizeof(cl_int),
                                0, 2048 * sizeof(cl_int), 0, NULL, NULL);
        }
        clSetKernelArg(k_noise_dark_lap_hist, 0, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_noise_dark_lap_hist, 1, sizeof(cl_mem), &ne_lap_hist_buf);
        clSetKernelArg(k_noise_dark_lap_hist, 2, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_noise_dark_lap_hist, 3, sizeof(int), &width);
        clSetKernelArg(k_noise_dark_lap_hist, 4, sizeof(int), &height);
        clSetKernelArg(k_noise_dark_lap_hist, 5, sizeof(int), &pos_per_ch_h);
        clSetKernelArg(k_noise_dark_lap_hist, 6, sizeof(int), &pos_per_ch);
        dispatch_1d_named(queue, k_noise_dark_lap_hist, align_up(total_lap_pos, 64), 64, "K0d dark_lap_hist");

        /* K0e: dark sigma finalize (single WI) */
        clSetKernelArg(k_noise_dark_finalize, 0, sizeof(cl_mem), &ne_lap_hist_buf);
        clSetKernelArg(k_noise_dark_finalize, 1, sizeof(cl_mem), &params_buf);
        dispatch_1d_named(queue, k_noise_dark_finalize, 1, 0, "K0e dark_finalize");

        /* Readback estimated alpha/sigma_sq for kernel arg setup.
         * NOT CPU processing — just transferring 2 scalars computed on GPU. */
        clFinish(queue);
        float est_params[PARAMS_SIZE];
        clEnqueueReadBuffer(queue, params_buf, CL_TRUE, 0,
                            PARAMS_SIZE * sizeof(float), est_params, 0, NULL, NULL);
        alpha    = est_params[13];
        sigma_sq = est_params[14];
        fprintf(stderr, "[GPU] Blind noise est: alpha=%.6f sigma_sq=%.8f s_scale=%.6f\n",
                alpha, sigma_sq, est_params[10]);
    }

    /* ---- K1: GAT forward + Bayer extraction (uses estimated alpha/sigma_sq) ---- */
    {
        clSetKernelArg(k_gat_extract, 0, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_gat_extract, 1, sizeof(cl_mem), &ch0_buf);
        clSetKernelArg(k_gat_extract, 2, sizeof(cl_mem), &ch1_buf);
        clSetKernelArg(k_gat_extract, 3, sizeof(cl_mem), &ch2_buf);
        clSetKernelArg(k_gat_extract, 4, sizeof(cl_mem), &ch3_buf);
        clSetKernelArg(k_gat_extract, 5, sizeof(int), &width);
        clSetKernelArg(k_gat_extract, 6, sizeof(int), &height);
        clSetKernelArg(k_gat_extract, 7, sizeof(float), &alpha);
        clSetKernelArg(k_gat_extract, 8, sizeof(float), &sigma_sq);
        dispatch_2d_named(queue, k_gat_extract,
                    align_up(hw, 16), align_up(hh, 16), 16, 16, "K1 gat_extract");
    }

    /* ---- K2: Build inverse GAT LUT (uses estimated alpha/sigma_sq) ---- */
    {
        clSetKernelArg(k_build_inv_lut, 0, sizeof(cl_mem), &lut_d_buf);
        clSetKernelArg(k_build_inv_lut, 1, sizeof(cl_mem), &lut_x_buf);
        clSetKernelArg(k_build_inv_lut, 2, sizeof(float), &alpha);
        clSetKernelArg(k_build_inv_lut, 3, sizeof(float), &sigma_sq);
        dispatch_1d_named(queue, k_build_inv_lut, 4096, 256, "K2 build_inv_lut");
    }

    /* ---- K3: LUT finalize ---- */
    {
        clSetKernelArg(k_lut_finalize, 0, sizeof(cl_mem), &lut_d_buf);
        clSetKernelArg(k_lut_finalize, 1, sizeof(cl_mem), &lut_params_buf);
        clSetKernelArg(k_lut_finalize, 2, sizeof(float), &alpha);
        clSetKernelArg(k_lut_finalize, 3, sizeof(float), &sigma_sq);
        dispatch_1d_named(queue, k_lut_finalize, 1, 0, "K3 lut_finalize");
    }

    /* ---- K4: Sigma histogram ---- */
    {
        /* Zero histogram buffer */
        cl_int zero = 0;
        clEnqueueFillBuffer(queue, hist_buf, &zero, sizeof(cl_int), 0,
                            4 * HIST_BINS * sizeof(cl_int), 0, NULL, NULL);

        clSetKernelArg(k_sigma_histogram, 0, sizeof(cl_mem), &ch0_buf);
        clSetKernelArg(k_sigma_histogram, 1, sizeof(cl_mem), &ch1_buf);
        clSetKernelArg(k_sigma_histogram, 2, sizeof(cl_mem), &ch2_buf);
        clSetKernelArg(k_sigma_histogram, 3, sizeof(cl_mem), &ch3_buf);
        clSetKernelArg(k_sigma_histogram, 4, sizeof(cl_mem), &hist_buf);
        clSetKernelArg(k_sigma_histogram, 5, sizeof(int), &hw);
        clSetKernelArg(k_sigma_histogram, 6, sizeof(int), &hh);

        const int n_x_per_row = (hw - 2) / 3 + 1;
        const int n_sigma_samples = n_x_per_row * hh;
        dispatch_1d_named(queue, k_sigma_histogram,
                    align_up(n_sigma_samples, 256), 256, "K4 sigma_hist");

        /* ---- K5: Sigma finalize ---- */
        clSetKernelArg(k_sigma_finalize, 0, sizeof(cl_mem), &hist_buf);
        clSetKernelArg(k_sigma_finalize, 1, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_sigma_finalize, 2, sizeof(int), &n_sigma_samples);
        dispatch_1d_named(queue, k_sigma_finalize, 1, 0, "K5 sigma_finalize");
    }

    /* ---- K6: Normalize channels ---- */
    {
        const int chsize_i = (int)chsize;
        clSetKernelArg(k_normalize, 0, sizeof(cl_mem), &ch0_buf);
        clSetKernelArg(k_normalize, 1, sizeof(cl_mem), &ch1_buf);
        clSetKernelArg(k_normalize, 2, sizeof(cl_mem), &ch2_buf);
        clSetKernelArg(k_normalize, 3, sizeof(cl_mem), &ch3_buf);
        clSetKernelArg(k_normalize, 4, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_normalize, 5, sizeof(int), &chsize_i);
        dispatch_1d_named(queue, k_normalize, align_up(chsize, 256), 256, "K6 normalize");
    }

    /* ---- K7-K10: Dark reference estimation (3 iterations) ----
     *
     * CPU equivalent: 3 iterations of weighted reduction with
     * achromatic filter + scale refinement via residual std.
     * Each iteration: reduce → finalize (→ resid_reduce → resid_finalize)
     * ---- */
    {
        const int n_iter = 1;  /* 2 loops sufficient for IRLS convergence */
        const float s_init = sigma_sq / fmaxf(alpha, 1e-12f);
        const float s_min = 0.05f * s_init;
        const float s_max = 50.0f * s_init;
        const size_t reduce_global = (size_t)N_REDUCE_WG * REDUCE_WG_SIZE;
        const int n_wg = N_REDUCE_WG;

        /* Set static args for dark_ref_reduce */
        clSetKernelArg(k_dark_ref_reduce, 0, sizeof(cl_mem), &ch0_buf);
        clSetKernelArg(k_dark_ref_reduce, 1, sizeof(cl_mem), &ch1_buf);
        clSetKernelArg(k_dark_ref_reduce, 2, sizeof(cl_mem), &ch2_buf);
        clSetKernelArg(k_dark_ref_reduce, 3, sizeof(cl_mem), &ch3_buf);
        clSetKernelArg(k_dark_ref_reduce, 4, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_dark_ref_reduce, 5, sizeof(cl_mem), &partial_buf);
        clSetKernelArg(k_dark_ref_reduce, 6, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_dark_ref_reduce, 7, sizeof(int), &hw);
        clSetKernelArg(k_dark_ref_reduce, 8, sizeof(int), &hh);
        clSetKernelArg(k_dark_ref_reduce, 9, sizeof(int), &width);

        /* Set static args for dark_ref_finalize */
        clSetKernelArg(k_dark_ref_finalize, 0, sizeof(cl_mem), &partial_buf);
        clSetKernelArg(k_dark_ref_finalize, 1, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_dark_ref_finalize, 2, sizeof(int), &n_wg);

        /* Set static args for resid_reduce */
        clSetKernelArg(k_dark_ref_resid_reduce, 0, sizeof(cl_mem), &ch0_buf);
        clSetKernelArg(k_dark_ref_resid_reduce, 1, sizeof(cl_mem), &ch1_buf);
        clSetKernelArg(k_dark_ref_resid_reduce, 2, sizeof(cl_mem), &ch2_buf);
        clSetKernelArg(k_dark_ref_resid_reduce, 3, sizeof(cl_mem), &ch3_buf);
        clSetKernelArg(k_dark_ref_resid_reduce, 4, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_dark_ref_resid_reduce, 5, sizeof(cl_mem), &partial_resid_buf);
        clSetKernelArg(k_dark_ref_resid_reduce, 6, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_dark_ref_resid_reduce, 7, sizeof(int), &hw);
        clSetKernelArg(k_dark_ref_resid_reduce, 8, sizeof(int), &hh);
        clSetKernelArg(k_dark_ref_resid_reduce, 9, sizeof(int), &width);

        /* Set static args for resid_finalize */
        clSetKernelArg(k_dark_ref_resid_finalize, 0, sizeof(cl_mem), &partial_resid_buf);
        clSetKernelArg(k_dark_ref_resid_finalize, 1, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_dark_ref_resid_finalize, 2, sizeof(int), &n_wg);
        clSetKernelArg(k_dark_ref_resid_finalize, 3, sizeof(float), &s_min);
        clSetKernelArg(k_dark_ref_resid_finalize, 4, sizeof(float), &s_max);

        for(int iter = 0; iter <= n_iter; iter++)
        {
            /* Zero partial sums */
            float fzero = 0.0f;
            clEnqueueFillBuffer(queue, partial_buf, &fzero, sizeof(float), 0,
                                N_REDUCE_WG * 5 * sizeof(float), 0, NULL, NULL);

            /* K7: dark_ref_reduce */
            dispatch_1d_named(queue, k_dark_ref_reduce, reduce_global, REDUCE_WG_SIZE, "K7 dark_reduce");

            /* K8: dark_ref_finalize */
            dispatch_1d_named(queue, k_dark_ref_finalize, 1, 0, "K8 dark_finalize");

            if(iter == n_iter) break;

            /* K9: resid_reduce */
            clEnqueueFillBuffer(queue, partial_resid_buf, &fzero, sizeof(float), 0,
                                N_REDUCE_WG * 2 * sizeof(float), 0, NULL, NULL);
            dispatch_1d_named(queue, k_dark_ref_resid_reduce, reduce_global, REDUCE_WG_SIZE, "K9 resid_reduce");

            /* K10: resid_finalize (updates s_scale in params) */
            dispatch_1d_named(queue, k_dark_ref_resid_finalize, 1, 0, "K10 resid_final");
        }
    }

    /* ---- K_SP: Stream Preprocess (fused normalize + dark_sub + WHT) ---- */
    {
        const int chsize_i = (int)chsize;
        clSetKernelArg(k_stream_preprocess, 0, sizeof(cl_mem), &ch0_buf);
        clSetKernelArg(k_stream_preprocess, 1, sizeof(cl_mem), &ch1_buf);
        clSetKernelArg(k_stream_preprocess, 2, sizeof(cl_mem), &ch2_buf);
        clSetKernelArg(k_stream_preprocess, 3, sizeof(cl_mem), &ch3_buf);
        clSetKernelArg(k_stream_preprocess, 4, sizeof(cl_mem), &luma_buf);
        clSetKernelArg(k_stream_preprocess, 5, sizeof(cl_mem), &c1_buf);
        clSetKernelArg(k_stream_preprocess, 6, sizeof(cl_mem), &c2_buf);
        clSetKernelArg(k_stream_preprocess, 7, sizeof(cl_mem), &c3_buf);
        clSetKernelArg(k_stream_preprocess, 8, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_stream_preprocess, 9, sizeof(int), &chsize_i);
        dispatch_1d_named(queue, k_stream_preprocess, align_up(chsize, 256), 256, "K_SP preprocess");
    }

    /* ---- K13: Fused Pass1+Pass2 denoise (half-res, 4 channels) ----
     *
     * Luma: luma_buf → l_out_buf  (luma_strength)
     * C1:   c1_buf   → c1_out_buf (chroma_strength)
     * C2:   c2_buf   → c2_out_buf (chroma_strength)
     * C3:   c3_buf   → c3_out_buf (chroma_strength)
     * ---- */
    {
        const float luma_strength   = strength * luma_str;
        const float chroma_strength = strength * chroma_str;

        /* Luma */
        clSetKernelArg(k_fused_pass12, 0, sizeof(cl_mem), &luma_buf);
        clSetKernelArg(k_fused_pass12, 1, sizeof(cl_mem), &l_out_buf);
        clSetKernelArg(k_fused_pass12, 2, sizeof(int), &hw);
        clSetKernelArg(k_fused_pass12, 3, sizeof(int), &hh);
        clSetKernelArg(k_fused_pass12, 4, sizeof(float), &luma_strength);
        dispatch_tile_named(queue, k_fused_pass12, hw, hh, "K13 fused_L");

        /* ========================================================
         * C1/C2/C3 — Y-guided filter (Step 8: unified L/C split).
         *
         * Luma plane (L = 2×2-WHT DC) stays on WHT LOSH above; the three
         * chroma sub-bands (C1, C2, C3) move to a Y-guided filter that
         * mirrors the galosh_yuv_gpu chroma path:
         *
         *   C_out(x, y) ≈ mean_a(x,y)·L(x,y) + mean_b(x,y)
         *
         * ε_pixel = strength_c² · (α·L + σ²)  with α, σ² = raw blind
         * estimates.  Guide = luma_buf (GAT'd noisy L plane — good enough
         * as cross-guidance; we do not require the denoised L because
         * edges in luminance are already 5-10× SNR over chroma).
         *
         * 2 dispatch runs (joint-pair + singleton):
         *   Run 1: Cb slot=C1, Cr slot=C2 → C1_out, C2_out
         *   Run 2: Cb slot=C3, Cr slot=C3 → C3_out (Cr output = same, discarded)
         *
         * ε α/σ² plumbed through params[16..19] (yuv chroma slots) —
         * pre-write raw α/σ² there so the shared kernel reads valid values.
         * ======================================================== */
        {
            /* Raw-mode pipeline uses half precision for luma/c1/c2/c3; the
             * shared yuv guided-filter kernels operate on fp32.  Convert
             * half → float once into dedicated scratch buffers, run the
             * 4-stage guided filter, then convert the float output back to
             * half to feed the downstream (half-based) reconstruct stages. */
            const int chsize = hw * hh;
            /* half → float for all 4 half buffers. */
            clSetKernelArg(k_raw_h2f, 0, sizeof(cl_mem), &luma_buf);
            clSetKernelArg(k_raw_h2f, 1, sizeof(cl_mem), &luma_f_buf);
            clSetKernelArg(k_raw_h2f, 2, sizeof(int),    &chsize);
            dispatch_1d_named(queue, k_raw_h2f, align_up(chsize, 256), 256, "K13.a h2f_L");
            clSetKernelArg(k_raw_h2f, 0, sizeof(cl_mem), &c1_buf);
            clSetKernelArg(k_raw_h2f, 1, sizeof(cl_mem), &c1_f_buf);
            dispatch_1d_named(queue, k_raw_h2f, align_up(chsize, 256), 256, "K13.a h2f_C1");
            clSetKernelArg(k_raw_h2f, 0, sizeof(cl_mem), &c2_buf);
            clSetKernelArg(k_raw_h2f, 1, sizeof(cl_mem), &c2_f_buf);
            dispatch_1d_named(queue, k_raw_h2f, align_up(chsize, 256), 256, "K13.a h2f_C2");
            clSetKernelArg(k_raw_h2f, 0, sizeof(cl_mem), &c3_buf);
            clSetKernelArg(k_raw_h2f, 1, sizeof(cl_mem), &c3_f_buf);
            dispatch_1d_named(queue, k_raw_h2f, align_up(chsize, 256), 256, "K13.a h2f_C3");

            /* Derive chroma (α, σ²) from params[13,14] into [16..19]. */
            clSetKernelArg(k_raw_chroma_derive, 0, sizeof(cl_mem), &params_buf);
            dispatch_1d_named(queue, k_raw_chroma_derive, 1, 0, "K13.0 chroma_derive");

            /* LOESS (bilateral-weighted guided filter) — single 2D pass that
             * replaces the old 4-kernel separable-guided pipeline.  Replaces
             *   moments_x → moments_y_ab → apply_x → apply_y
             * with one non-separable kernel that re-weights neighbours by
             * exp(-(Y_i - Y_c)² / (2σ²)), excluding specular highlights from
             * silver windows (see yuv_gat.cl docstring for derivation).
             * Outputs to float scratch, then f2h to downstream c*_out_buf. */
            for(int pass = 0; pass < 2; pass++) {
                cl_mem cb_in      = (pass == 0) ? c1_f_buf : c3_f_buf;
                cl_mem cr_in      = (pass == 0) ? c2_f_buf : c3_f_buf;
                /* Float output scratch (not aliased with inputs): reuse the
                 * now-unused rgf_a_cb_x / rgf_a_cr_x scalar-plane scratch. */
                cl_mem cb_f_out   = rgf_a_cb_x;
                cl_mem cr_f_out   = rgf_a_cr_x;
                cl_mem cb_half_out = (pass == 0) ? c1_out_buf : c3_out_buf;
                cl_mem cr_half_out = (pass == 0) ? c2_out_buf : rgf_b_cr_x;  /* unused */

                clSetKernelArg(k_raw_guided_loess, 0, sizeof(cl_mem), &luma_f_buf);
                clSetKernelArg(k_raw_guided_loess, 1, sizeof(cl_mem), &cb_in);
                clSetKernelArg(k_raw_guided_loess, 2, sizeof(cl_mem), &cr_in);
                clSetKernelArg(k_raw_guided_loess, 3, sizeof(cl_mem), &params_buf);
                clSetKernelArg(k_raw_guided_loess, 4, sizeof(float),  &chroma_strength);
                clSetKernelArg(k_raw_guided_loess, 5, sizeof(cl_mem), &cb_f_out);
                clSetKernelArg(k_raw_guided_loess, 6, sizeof(cl_mem), &cr_f_out);
                clSetKernelArg(k_raw_guided_loess, 7, sizeof(int),    &hw);
                clSetKernelArg(k_raw_guided_loess, 8, sizeof(int),    &hh);
                dispatch_2d_named(queue, k_raw_guided_loess,
                                  align_up(hw, 16), align_up(hh, 16),
                                  16, 16, "K13 raw_gf_loess");

                /* Float → half: write to downstream c{1,2,3}_out_buf. */
                clSetKernelArg(k_raw_f2h, 0, sizeof(cl_mem), &cb_f_out);
                clSetKernelArg(k_raw_f2h, 1, sizeof(cl_mem), &cb_half_out);
                clSetKernelArg(k_raw_f2h, 2, sizeof(int),    &chsize);
                dispatch_1d_named(queue, k_raw_f2h, align_up(chsize, 256), 256, "K13 f2h_cb");
                if(pass == 0) {
                    clSetKernelArg(k_raw_f2h, 0, sizeof(cl_mem), &cr_f_out);
                    clSetKernelArg(k_raw_f2h, 1, sizeof(cl_mem), &cr_half_out);
                    dispatch_1d_named(queue, k_raw_f2h, align_up(chsize, 256), 256, "K13 f2h_cr");
                }
            }
        }

    }

    /* ---- K14: Compute L fullres (noisy + pilot) ---- */
    {
        clSetKernelArg(k_compute_L_fullres, 0, sizeof(cl_mem), &luma_buf);
        clSetKernelArg(k_compute_L_fullres, 1, sizeof(cl_mem), &c1_buf);
        clSetKernelArg(k_compute_L_fullres, 2, sizeof(cl_mem), &c2_buf);
        clSetKernelArg(k_compute_L_fullres, 3, sizeof(cl_mem), &c3_buf);
        clSetKernelArg(k_compute_L_fullres, 4, sizeof(cl_mem), &Lfr_noisy_buf);
        clSetKernelArg(k_compute_L_fullres, 5, sizeof(int), &hw);
        clSetKernelArg(k_compute_L_fullres, 6, sizeof(int), &hh);
        dispatch_2d_named(queue, k_compute_L_fullres,
                    align_up(hw, 16), align_up(hh, 16), 16, 16, "K14 Lfr_noisy");

        clSetKernelArg(k_compute_L_fullres, 0, sizeof(cl_mem), &l_out_buf);
        clSetKernelArg(k_compute_L_fullres, 1, sizeof(cl_mem), &c1_out_buf);
        clSetKernelArg(k_compute_L_fullres, 2, sizeof(cl_mem), &c2_out_buf);
        clSetKernelArg(k_compute_L_fullres, 3, sizeof(cl_mem), &c3_out_buf);
        clSetKernelArg(k_compute_L_fullres, 4, sizeof(cl_mem), &Lfr_pilot_buf);
        dispatch_2d_named(queue, k_compute_L_fullres,
                    align_up(hw, 16), align_up(hh, 16), 16, 16, "K14 Lfr_pilot");
    }

    /* ---- K15: Pass2-only for full-res L ---- */
    {
        const float luma_strength = strength * luma_str;
        clSetKernelArg(k_pass2_only, 0, sizeof(cl_mem), &Lfr_noisy_buf);
        clSetKernelArg(k_pass2_only, 1, sizeof(cl_mem), &Lfr_pilot_buf);
        clSetKernelArg(k_pass2_only, 2, sizeof(cl_mem), &Lfr_den_buf);
        clSetKernelArg(k_pass2_only, 3, sizeof(int), &width);
        clSetKernelArg(k_pass2_only, 4, sizeof(int), &height);
        clSetKernelArg(k_pass2_only, 5, sizeof(float), &luma_strength);
        dispatch_tile_named(queue, k_pass2_only, width, height, "K15 pass2_Lfr");
    }

    /* ---- K16: Reconstruct ---- */
    {
        clSetKernelArg(k_reconstruct,  0, sizeof(cl_mem), &Lfr_den_buf);
        clSetKernelArg(k_reconstruct,  1, sizeof(cl_mem), &c1_out_buf);
        clSetKernelArg(k_reconstruct,  2, sizeof(cl_mem), &c2_out_buf);
        clSetKernelArg(k_reconstruct,  3, sizeof(cl_mem), &c3_out_buf);
        clSetKernelArg(k_reconstruct,  4, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_reconstruct,  5, sizeof(cl_mem), &lut_d_buf);
        clSetKernelArg(k_reconstruct,  6, sizeof(cl_mem), &lut_x_buf);
        clSetKernelArg(k_reconstruct,  7, sizeof(cl_mem), &lut_params_buf);
        clSetKernelArg(k_reconstruct,  8, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_reconstruct,  9, sizeof(int), &width);
        clSetKernelArg(k_reconstruct, 10, sizeof(int), &height);
        dispatch_2d_named(queue, k_reconstruct,
                    align_up(hw, 16), align_up(hh, 16), 16, 16, "K16 reconstruct");
    }

    /* Wait for all GPU work to complete */
    clFinish(queue);
    double t_pipe = get_time_ms() - t_pipe_start;

    /* Per-kernel profiling report */
    {
        fprintf(stderr, "\n[GPU] ====== PER-KERNEL PROFILING ======\n");
        double total_gpu = 0.0;
        for(int i = 0; i < prof_count; i++) {
            double ms = event_ms(prof_events[i]);
            total_gpu += ms;
            const char *name = prof_names[i] ? prof_names[i] : "(unnamed)";
            fprintf(stderr, "[GPU]   %-20s %7.3f ms\n", name, ms);
            clReleaseEvent(prof_events[i]);
        }
        fprintf(stderr, "[GPU]   %-20s %7.3f ms\n", "TOTAL (GPU time)", total_gpu);
        fprintf(stderr, "[GPU] ==================================\n");
    }

    /* ================================================================
     * Download output + write file
     * ================================================================ */
    err = clEnqueueReadBuffer(queue, raw_buf, CL_TRUE, 0,
                              npix * sizeof(float), raw, 0, NULL, NULL);
    CL_CHECK(err, "download output");

    {
        FILE *f = fopen(output_file, "wb");
        if(!f) { fprintf(stderr, "Cannot write %s\n", output_file); goto cl_cleanup; }
        fwrite(raw, sizeof(float), npix, f);
        fclose(f);
    }

    double t_total = get_time_ms() - t_total_start;

    /* ================================================================
     * Timing summary
     * ================================================================ */
    fprintf(stderr, "\n[GPU] ====== FULL-PIPELINE TIMING ======\n");
    fprintf(stderr, "[GPU]   GPU pipeline:    %7.1f ms  (all 16 kernels)\n", t_pipe);
    fprintf(stderr, "[GPU]   CL setup:        %7.1f ms  (ctx+build+alloc)\n", t_cl_setup);
    fprintf(stderr, "[GPU]   TOTAL:           %7.1f ms  (incl. file I/O)\n", t_total);
    fprintf(stderr, "[GPU] ==================================\n");

    /* Report GPU-only time for benchmarks (parseable) */
    fprintf(stderr, "[GPU_PIPELINE_TIME] %.2f\n", t_pipe);

    /* Read back params for debug info */
    {
        float h_params[PARAMS_SIZE];
        clEnqueueReadBuffer(queue, params_buf, CL_TRUE, 0,
                            PARAMS_SIZE * sizeof(float), h_params, 0, NULL, NULL);
        fprintf(stderr, "[GPU] Sigma: unified=%.4f (ch: %.4f %.4f %.4f %.4f)\n",
                h_params[P_UNIFIED_SIGMA],
                h_params[P_SIGMA_CH0], h_params[P_SIGMA_CH1],
                h_params[P_SIGMA_CH2], h_params[P_SIGMA_CH3]);
        fprintf(stderr, "[GPU] Dark ref: %.4f %.4f %.4f %.4f\n",
                h_params[P_DARK_REF0], h_params[P_DARK_REF1],
                h_params[P_DARK_REF2], h_params[P_DARK_REF3]);

        /* Diagnostic: GPU LUT values and intermediate buffers */
        float lp[5];
        clEnqueueReadBuffer(queue, lut_params_buf, CL_TRUE, 0, 5*sizeof(float), lp, 0, NULL, NULL);
        fprintf(stderr, "[DIAG] lut_params: d_min=%.6f d_max=%.6f y_break=%.6f t_break=%.4f sigma_raw=%.6f\n",
                lp[0], lp[1], lp[2], lp[3], lp[4]);

        int lut_idx[] = {0, 100, 500, 1000, 2000, 4095};
        for(int i = 0; i < 6; i++) {
            float d_val, x_val;
            clEnqueueReadBuffer(queue, lut_d_buf, CL_TRUE, lut_idx[i]*sizeof(float),
                                sizeof(float), &d_val, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, lut_x_buf, CL_TRUE, lut_idx[i]*sizeof(float),
                                sizeof(float), &x_val, 0, NULL, NULL);
            fprintf(stderr, "[DIAG] LUT D[%d]=%.8f x[%d]=%.8f\n",
                    lut_idx[i], d_val, lut_idx[i], x_val);
        }

        /* Read a few intermediate buffer values (FP16 buffers → cl_half) */
        cl_half diag_h;
        clEnqueueReadBuffer(queue, luma_buf, CL_TRUE, 0, sizeof(cl_half), &diag_h, 0, NULL, NULL);
        fprintf(stderr, "[DIAG] luma[0]=%.6f", (double)cl_half_to_float(diag_h));
        clEnqueueReadBuffer(queue, l_out_buf, CL_TRUE, 0, sizeof(cl_half), &diag_h, 0, NULL, NULL);
        fprintf(stderr, " l_out[0]=%.6f", (double)cl_half_to_float(diag_h));
        clEnqueueReadBuffer(queue, Lfr_den_buf, CL_TRUE, 0, sizeof(cl_half), &diag_h, 0, NULL, NULL);
        fprintf(stderr, " Lfr_den[0]=%.6f\n", (double)cl_half_to_float(diag_h));
    }

cl_cleanup:
    /* Release kernels */
    if(k_noise_block_stats) clReleaseKernel(k_noise_block_stats);
    if(k_noise_dark_samp_hist) clReleaseKernel(k_noise_dark_samp_hist);
    if(k_noise_estimate) clReleaseKernel(k_noise_estimate);
    if(k_noise_dark_lap_hist) clReleaseKernel(k_noise_dark_lap_hist);
    if(k_noise_dark_finalize) clReleaseKernel(k_noise_dark_finalize);
    if(k_gat_extract) clReleaseKernel(k_gat_extract);
    if(k_build_inv_lut) clReleaseKernel(k_build_inv_lut);
    if(k_lut_finalize) clReleaseKernel(k_lut_finalize);
    if(k_sigma_histogram) clReleaseKernel(k_sigma_histogram);
    if(k_sigma_finalize) clReleaseKernel(k_sigma_finalize);
    if(k_normalize) clReleaseKernel(k_normalize);
    if(k_dark_ref_reduce) clReleaseKernel(k_dark_ref_reduce);
    if(k_dark_ref_finalize) clReleaseKernel(k_dark_ref_finalize);
    if(k_dark_ref_resid_reduce) clReleaseKernel(k_dark_ref_resid_reduce);
    if(k_dark_ref_resid_finalize) clReleaseKernel(k_dark_ref_resid_finalize);
    if(k_dark_ref_subtract) clReleaseKernel(k_dark_ref_subtract);
    if(k_wht_decompose) clReleaseKernel(k_wht_decompose);
    if(k_pass1_only) clReleaseKernel(k_pass1_only);
    if(k_fused_pass12) clReleaseKernel(k_fused_pass12);
    if(k_raw_chroma_derive)      clReleaseKernel(k_raw_chroma_derive);
    if(k_raw_h2f)                clReleaseKernel(k_raw_h2f);
    if(k_raw_f2h)                clReleaseKernel(k_raw_f2h);
    if(k_raw_guided_moments_x)   clReleaseKernel(k_raw_guided_moments_x);
    if(k_raw_guided_moments_yab) clReleaseKernel(k_raw_guided_moments_yab);
    if(k_raw_guided_apply_x)     clReleaseKernel(k_raw_guided_apply_x);
    if(k_raw_guided_apply_y)     clReleaseKernel(k_raw_guided_apply_y);
    if(k_raw_guided_loess)       clReleaseKernel(k_raw_guided_loess);
    if(k_compute_L_fullres) clReleaseKernel(k_compute_L_fullres);
    if(k_pass2_only) clReleaseKernel(k_pass2_only);
    if(k_reconstruct) clReleaseKernel(k_reconstruct);
    if(k_stream_preprocess) clReleaseKernel(k_stream_preprocess);
    if(k_stream_postprocess) clReleaseKernel(k_stream_postprocess);

    /* Release buffers */
    if(raw_buf) clReleaseMemObject(raw_buf);
    if(ch0_buf) clReleaseMemObject(ch0_buf);
    if(ch1_buf) clReleaseMemObject(ch1_buf);
    if(ch2_buf) clReleaseMemObject(ch2_buf);
    if(ch3_buf) clReleaseMemObject(ch3_buf);
    if(luma_buf) clReleaseMemObject(luma_buf);
    if(c1_buf) clReleaseMemObject(c1_buf);
    if(c2_buf) clReleaseMemObject(c2_buf);
    if(c3_buf) clReleaseMemObject(c3_buf);
    if(l_out_buf) clReleaseMemObject(l_out_buf);
    if(c1_out_buf) clReleaseMemObject(c1_out_buf);
    if(c2_out_buf) clReleaseMemObject(c2_out_buf);
    if(c3_out_buf) clReleaseMemObject(c3_out_buf);
    if(rgf_sum_Y_x)   clReleaseMemObject(rgf_sum_Y_x);
    if(rgf_sum_YY_x)  clReleaseMemObject(rgf_sum_YY_x);
    if(rgf_sum_Cb_x)  clReleaseMemObject(rgf_sum_Cb_x);
    if(rgf_sum_Cr_x)  clReleaseMemObject(rgf_sum_Cr_x);
    if(rgf_sum_YCb_x) clReleaseMemObject(rgf_sum_YCb_x);
    if(rgf_sum_YCr_x) clReleaseMemObject(rgf_sum_YCr_x);
    if(rgf_a_cb_x)    clReleaseMemObject(rgf_a_cb_x);
    if(rgf_b_cb_x)    clReleaseMemObject(rgf_b_cb_x);
    if(rgf_a_cr_x)    clReleaseMemObject(rgf_a_cr_x);
    if(rgf_b_cr_x)    clReleaseMemObject(rgf_b_cr_x);
    if(luma_f_buf)    clReleaseMemObject(luma_f_buf);
    if(c1_f_buf)      clReleaseMemObject(c1_f_buf);
    if(c2_f_buf)      clReleaseMemObject(c2_f_buf);
    if(c3_f_buf)      clReleaseMemObject(c3_f_buf);
    if(Lfr_noisy_buf) clReleaseMemObject(Lfr_noisy_buf);
    if(Lfr_pilot_buf) clReleaseMemObject(Lfr_pilot_buf);
    if(Lfr_den_buf) clReleaseMemObject(Lfr_den_buf);
    if(pilot_tmp_buf) clReleaseMemObject(pilot_tmp_buf);
    if(lut_d_buf) clReleaseMemObject(lut_d_buf);
    if(lut_x_buf) clReleaseMemObject(lut_x_buf);
    if(lut_params_buf) clReleaseMemObject(lut_params_buf);
    if(params_buf) clReleaseMemObject(params_buf);
    if(hist_buf) clReleaseMemObject(hist_buf);
    if(partial_buf) clReleaseMemObject(partial_buf);
    if(partial_resid_buf) clReleaseMemObject(partial_resid_buf);
    if(blk_mean_buf) clReleaseMemObject(blk_mean_buf);
    if(blk_var_buf) clReleaseMemObject(blk_var_buf);
    if(ne_dark_hist_buf) clReleaseMemObject(ne_dark_hist_buf);
    if(ne_lap_hist_buf) clReleaseMemObject(ne_lap_hist_buf);

    /* Release CL objects */
    if(program) clReleaseProgram(program);
    if(queue) clReleaseCommandQueue(queue);
    if(context) clReleaseContext(context);

    free_aligned(raw);
    return 0;
}
