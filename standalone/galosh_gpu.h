/* galosh_gpu.h  --  GALOSH GPU-side common infrastructure.
 *
 * Shared between galosh_raw_gpu.c, galosh_yuv_gpu.c, and (legacy debug)
 * galosh_single_gpu.c.  Everything declared here is `static` so each
 * binary gets its own copy (mirrors galosh_cpu.h's convention).
 *
 * Sections (search "Section:" to navigate):
 *   1. cl_half <-> float conversion (FP16 LDS path).
 *   2. GALOSH GPU constants (BS / BP / TILE_SIZE / PARAMS_SIZE indices)
 *      -- must match galosh.cl.
 *   3. Pinned-memory alloc helpers (alloc_float / free_aligned).
 *   4. Wall-clock timing (get_time_ms).
 *   5. CL_CHECK macro + load_kernel_source.
 *   6. Per-kernel profiling (prof_add / event_ms / prof_events array).
 *   7. Dispatch helpers (dispatch_1d / dispatch_2d / dispatch_tile +
 *      named variants for profiling).
 *   8. align_up.
 *
 * Pipeline-specific orchestration (run_galosh_raw_gpu / run_yuv_gat_gpu /
 * run_single_plane_gpu) lives in the per-modality .c files.  Kernels are
 * all in galosh.cl.
 */
#ifndef GALOSH_GPU_H
#define GALOSH_GPU_H

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
/* BUG FIX 2026-05-10: GALOSH_STRIDE was 4 (= host build_opts -D pass to
 * OpenCL kernel) which overrode galosh.cl's default `#define GALOSH_STRIDE 2`.
 * CPU `galosh_pass1_blocked` is called with stride=2 (hardcoded parameter
 * in galosh_pass12_multiorient_blocked).  Mismatch caused GPU's pass12_o32
 * to iterate with stride=4 → ~half block coverage per pixel (4 vs 16
 * contributions for interior pixels) → degraded Phase 5 BayesShrink
 * averaging → ~1.5 dB SIDD val PSNR gap vs CPU O.  Verified via diagnostic
 * printf in pass1_o32_dump showing GPU iterates by * 4 = ref_r ∈ {0, 4, 8,
 * ...} instead of CPU's by * 2 = ref_r ∈ {0, 2, 4, ...}.  HALO formula
 * `8 - GALOSH_STRIDE` followed suit (= 4 vs CPU-equivalent 6). */
#define GALOSH_STRIDE      2
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

/* Directory of the running executable (kernels ship next to the exe).
 * 実行ファイルのあるディレクトリ（カーネル .cl は exe と同じ場所に置かれる）。 */
static const char *galosh_exe_dir(void) {
    static char dir[1024];
    static int done = 0;
    if(!done) {
        dir[0] = 0;
#ifdef _WIN32
        DWORD n = GetModuleFileNameA(NULL, dir, (DWORD)sizeof(dir) - 1);
        if(n > 0) {
            dir[n] = 0;
            for(char *p = dir + n; p > dir; p--)
                if(*p == '\\' || *p == '/') { *p = 0; break; }
        }
#else
        ssize_t n = readlink("/proc/self/exe", dir, sizeof(dir) - 1);
        if(n > 0) {
            dir[n] = 0;
            for(char *p = dir + n; p > dir; p--)
                if(*p == '/') { *p = 0; break; }
        }
#endif
        done = 1;
    }
    return dir;
}

/* Load a kernel/source file. Resolution order:
 *   1. the path as given (relative to the CWD, e.g. run from the repo root);
 *   2. <exe dir>/<basename> — so the CLI works from ANY working directory
 *      as long as the .cl/.clh files sit next to the executable (they do).
 * CWD 依存を排除：与えられたパス → exe と同じディレクトリの basename の順で解決。 */
static char *load_kernel_source(const char *filename, size_t *out_len) {
    FILE *f = fopen(filename, "rb");
    if(!f) {
        const char *base = filename;
        for(const char *p = filename; *p; p++)
            if(*p == '/' || *p == '\\') base = p + 1;
        char alt[1200];
        snprintf(alt, sizeof(alt), "%s/%s", galosh_exe_dir(), base);
        f = fopen(alt, "rb");
        if(!f) { fprintf(stderr, "Cannot open %s (also tried %s)\n", filename, alt); return NULL; }
    }
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


#endif /* GALOSH_GPU_H */
