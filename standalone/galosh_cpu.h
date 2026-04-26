/* galosh_cpu.h  --  GALOSH CPU-side complete library (algorithm core +
 * runtime infrastructure).  Single header, `static` definitions so each
 * translation unit gets its own copy (matches darktable plugin convention
 * where a single .c file is dropped into an IOP and can't share symbols).
 *
 * Sections (search "Section:" to navigate):
 *   1. Runtime infrastructure (alloc shim, OMP macros, prof helpers,
 *      median selection helper).
 *   2. Noise model: Foi-Mäkitalo Poisson-Gaussian + GAT (forward / exact
 *      unbiased inverse / blind sigma estimation).
 *   3. WHT primitives (8x8 + 4x4 in sequency order).
 *   4. Kaiser windows (8x8 + 4x4 overlap-add weights).
 *   5. GALOSH passes (block-size-parameterized BayesShrink Pass1 +
 *      empirical Wiener Pass2 + multi-orientation wrapper).  Pass1
 *      supports MAD-based robust sigma_Y for noise-cluster killing.
 *   6. LOESS chroma denoise (luma-guided locally-weighted regression).
 *   7. EWA Jinc-Lanczos-3 generic 2x upsample (used by chroma_up=1).
 *
 * RAW-specific orchestration (K11 2x2 WHT decompose + K14 compute_L_fullres
 * + K16 inverse 2x2 WHT + main pipeline) lives in galosh_raw_cpu.c.
 * YUV-specific orchestration (sRGB <-> linear, BT.709 YCbCr, Y-GAT
 * pipeline + main) lives in galosh_yuv_cpu.c.
 */
#ifndef GALOSH_CPU_H
#define GALOSH_CPU_H

/* Introspection TUs see every static function here but only use
 * the params struct from rawdenoise.c.  Silence -Wunused-function so
 * -Werror doesn't kill the auto-generated introspection compile. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

/*
 * Standalone raw denoiser -- GALOSH (GAT L/C-decomposed Overlap Shrinkage)
 *
 * Replaces BM3D with a purely local WHT-based shrinkage denoiser.
 * No block matching, no non-local processing.
 *
 * Usage: rawdenoise_v6 input.bin output.bin width height
 *        [method] [strength] [luma_str] [chroma_str] [alpha] [sigma_sq]
 *
 * Input/output: 32-bit float raw Bayer mosaic (RGGB), row-major
 *
 * Build:
 *   gcc -O3 -march=native -msse4.1 -fopenmp -o rawdenoise_v6 rawdenoise_v6.c -lm
 *
 * Theory:
 *   GALOSH is a two-pass local denoiser that relies on the Generalized
 *   Anscombe Transform (GAT) to normalize Poisson-Gaussian noise to
 *   i.i.d. Gaussian (sigma=1). In this stabilized domain, local adaptive
 *   shrinkage in the Walsh-Hadamard Transform (WHT) domain is sufficient
 *   without non-local block matching.
 *
 *   Pass 1: 8x8 WHT + BayesShrink adaptive hard threshold (pilot)
 *     Ref: Chang, Yu & Vetterli, "Adaptive wavelet thresholding for image
 *          denoising and compression", IEEE TIP, 2000
 *   Pass 2: 8x8 WHT + empirical Wiener shrinkage using pilot
 *     Ref: Textbook James-Stein / empirical Wiener estimator
 *
 *   The L/C decomposition via 2x2 WHT separates luma and chroma at half
 *   resolution, allowing independent strength control.
 *     Ref: Danielyan et al. "Cross-color BM3D" (LNLA 2009) -- for the
 *          L/C decomposition idea (we replace BM3D with local shrinkage)
 *
 *   GAT and exact unbiased inverse:
 *     Ref: Anscombe (1948), Foi et al. (Sig.Proc. 2008),
 *          Makitalo & Foi (TIP 2013)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <omp.h>
#include <time.h>
#include <xmmintrin.h>  /* SSE  */
#include <emmintrin.h>  /* SSE2 */
#include <pmmintrin.h>  /* SSE3: _mm_hadd_ps */

/* =====================================================================
 *  Environment compat layer
 *
 *  This header is used in three contexts:
 *    (a) standalone CPU drivers (galosh_raw_cpu.c, galosh_yuv_cpu.c),
 *        built without darktable — we provide every symbol locally.
 *    (b) darktable IOP (rawdenoise.c) — the real dt_* symbols come from
 *        darktable headers that the IOP already included BEFORE this
 *        header is pulled in.  Define GALOSH_DT_ENV before including
 *        galosh_core.h in that case to skip the standalone stubs.
 *    (c) external plugins — decide per host.
 *
 *  The standalone stubs block below only compiles when !GALOSH_DT_ENV.
 * ===================================================================== */
#ifdef GALOSH_DT_ENV
/* darktable exposes `dt_alloc_aligned(size)` (1-arg, implicit cacheline
 * alignment via DT_CACHELINE_BYTES).  Our core calls `dt_alloc_align(
 * alignment, sz)` (2-arg) in a few places where the standalone build
 * picks a tighter alignment.  Wrap to the 1-arg form and discard the
 * explicit alignment — DT_CACHELINE_BYTES satisfies every SSE/AVX load
 * we do. */
static inline void *dt_alloc_align(size_t alignment, size_t sz)
{
  (void)alignment;
  return dt_alloc_aligned(sz);
}
#endif

#ifndef GALOSH_DT_ENV

#define DT_OMP_FOR() _Pragma("omp parallel for schedule(static)")

typedef int gboolean;
#define TRUE 1
#define FALSE 0

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

static inline float *dt_alloc_align_float(size_t n) {
    size_t sz = sizeof(float) * ((n + 15) & ~15);
#ifdef _WIN32
    return (float *)_aligned_malloc(sz, 64);
#else
    return (float *)aligned_alloc(64, sz);
#endif
}
static inline void *dt_alloc_align(size_t alignment, size_t sz) {
#ifdef _WIN32
    return _aligned_malloc(sz, alignment);
#else
    return aligned_alloc(alignment, sz);
#endif
}
static inline void dt_free_align(void *p) {
    if(!p) return;
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

typedef struct { int width, height; } dt_iop_roi_t;

#endif /* !GALOSH_DT_ENV */


/* ================================================================
 * Per-step profiling helpers (CPU side).
 *
 * Mirror of the GPU prof helpers in galosh_gpu_common.h (pending Stage 3
 * file split): same `prof_add(name, ms)` / `prof_print()` API so the two
 * sides can be benched with the same harness.  Compile with -DGALOSH_PROF
 * to enable; otherwise all helpers compile to no-ops and incur zero
 * runtime cost.  Uses portable wall-clock (clock_gettime / QueryPerformance
 * Counter) so it captures real elapsed time including I/O stalls.
 * ================================================================ */
#if defined(_WIN32)
  #include <windows.h>
  static inline double galosh_get_time_ms(void)
  {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
  }
#else
  #include <time.h>
  static inline double galosh_get_time_ms(void)
  {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + 1e-6 * (double)ts.tv_nsec;
  }
#endif

#ifdef GALOSH_PROF
  #define GALOSH_PROF_MAX_STEPS 64
  typedef struct { const char *name; double ms; int n; } galosh_prof_entry_t;
  static galosh_prof_entry_t galosh_prof_table[GALOSH_PROF_MAX_STEPS];
  static int galosh_prof_count = 0;

  static inline void galosh_prof_add(const char *name, double ms)
  {
    for(int i = 0; i < galosh_prof_count; i++)
      if(galosh_prof_table[i].name == name ||
         (galosh_prof_table[i].name && name &&
          strcmp(galosh_prof_table[i].name, name) == 0))
      {
        galosh_prof_table[i].ms += ms;
        galosh_prof_table[i].n  += 1;
        return;
      }
    if(galosh_prof_count < GALOSH_PROF_MAX_STEPS)
    {
      galosh_prof_table[galosh_prof_count].name = name;
      galosh_prof_table[galosh_prof_count].ms   = ms;
      galosh_prof_table[galosh_prof_count].n    = 1;
      galosh_prof_count++;
    }
  }
  static inline void galosh_prof_reset(void) { galosh_prof_count = 0; }
  static inline void galosh_prof_print(const char *header)
  {
    fprintf(stderr, "\n[GALOSH_PROF] ====== %s ======\n", header ? header : "per-step");
    double total = 0.0;
    for(int i = 0; i < galosh_prof_count; i++) total += galosh_prof_table[i].ms;
    for(int i = 0; i < galosh_prof_count; i++)
      fprintf(stderr, "[GALOSH_PROF] %-32s  %8.2f ms  (n=%d, %.1f%%)\n",
              galosh_prof_table[i].name ? galosh_prof_table[i].name : "(null)",
              galosh_prof_table[i].ms, galosh_prof_table[i].n,
              total > 0.0 ? 100.0 * galosh_prof_table[i].ms / total : 0.0);
    fprintf(stderr, "[GALOSH_PROF] %-32s  %8.2f ms\n", "TOTAL", total);
  }
  /* Convenience: wrap a block in start/end timestamps and accumulate. */
  #define GALOSH_PROF_BEGIN(name) \
      const double _galosh_prof_t0_##name = galosh_get_time_ms()
  #define GALOSH_PROF_END(name) \
      galosh_prof_add(#name, galosh_get_time_ms() - _galosh_prof_t0_##name)
#else
  static inline void galosh_prof_add(const char *name, double ms) { (void)name; (void)ms; }
  static inline void galosh_prof_reset(void) {}
  static inline void galosh_prof_print(const char *header) { (void)header; }
  #define GALOSH_PROF_BEGIN(name) ((void)0)
  #define GALOSH_PROF_END(name)   ((void)0)
#endif


/* ===================== GALOSH Constants =====================
 * GALOSH: GAT L/C-decomposed Overlap Shrinkage
 *
 * Purely local denoiser -- no block matching, no non-local processing.
 * Relies on GAT normalizing noise to i.i.d. Gaussian (sigma=1) so that
 * local adaptive shrinkage in the WHT domain is sufficient.
 *
 * Two-pass design:
 *   Pass 1: 8x8 WHT + BayesShrink hard threshold (pilot estimate)
 *     Ref: Chang, Yu & Vetterli, IEEE TIP, 2000
 *   Pass 2: 8x8 WHT + Wiener shrinkage using pilot
 *     Ref: Textbook empirical Wiener / James-Stein estimator
 */
#define GALOSH_BLOCK_SIZE   8
#define GALOSH_BLOCK_PIXELS (GALOSH_BLOCK_SIZE * GALOSH_BLOCK_SIZE)  /* 64 */
#define GALOSH_HALF_BLOCK   4
#define GALOSH_HALF_PIXELS  (GALOSH_HALF_BLOCK * GALOSH_HALF_BLOCK)  /* 16 */
#define GALOSH_STRIDE       4   /* 50% overlap for 8x8 blocks */

/* Wiener floor: minimum gain to prevent complete signal annihilation.
 *
 * Derived from perceptual threshold: human vision tolerates residual
 * noise up to ~1/N of signal range (Weber fraction ~1%).
 * For N=64 coefficients, the minimum fraction of each coefficient
 * to preserve: 1/N = 0.016. We use 1/sqrt(N) = 0.125 ≈ 0.1 as a
 * compromise: sqrt accounts for energy (not amplitude) domain.
 *
 * In practice, floor=0 risks creating black holes in uniform areas
 * where all coefficients are killed; floor > 0 preserves DC leakage
 * and prevents blocking artifacts at block boundaries.
 *
 * 1/sqrt(N) = 1/8 = 0.125 for 8x8 blocks.
 */
#define GALOSH_WIENER_FLOOR (1.0f / GALOSH_BLOCK_SIZE)  /* 0.125 */

/* CFA frequency indices in 8x8 WHT (full-res raw Bayer).
 *
 * The RGGB Bayer pattern has 2-pixel periodicity in both row and column.
 * In the 8-point WHT (this implementation's sequency order), the
 * 2-pixel alternation [+,-,+,-,+,-,+,-] maps to index 1.
 *
 * In 2D, the CFA color information occupies 3 coefficients:
 *   (row=0, col=1) = idx  1: horizontal color diff (R−Gr / Gb−B DC)
 *   (row=1, col=0) = idx  8: vertical color diff   (R−Gb / Gr−B DC)
 *   (row=1, col=1) = idx  9: checkerboard           (R−Gr−Gb+B DC)
 *
 * These carry the "DC of color" — the average inter-channel difference
 * across the 8×8 block. Shrinking them in full-res raw GALOSH destroys
 * color information, causing green shift (G-biased luma dominates).
 *
 * CFA 色情報が乗る WHT 係数。2 pixel 周期の交互パターンは WHT index 1
 * に対応。2D では (0,1), (1,0), (1,1) の 3 係数が CFA 色 DC を担う。
 * これらを Phase 4 full-res GALOSH で shrinkage から保護すると、
 * 色情報が保持され green shift が防止される。
 */
/* Achromatic filter threshold for dark anchor estimation.
 *
 * In sigma-normalized GAT domain (noise std = 1), the range (max - min)
 * of 4 i.i.d. N(mu, 1) samples has E[range] ≈ 2.06, std ≈ 0.73.
 * Threshold 4.0 covers ~98% of noise-only pixels while excluding
 * pixels with real spectral differences (e.g., dark skin: R ≠ G
 * due to melanin absorption).
 *
 * This prevents dark-area color shift caused by contamination of
 * the dark anchor with spectrally non-neutral signal.
 *
 * Ref: achromatic pixel selection is established in white balance
 *      estimation (Gray World variants); same principle applied here
 *      to dark-area DC bias removal.
 */
#define GALOSH_ACHROMATIC_RANGE 4.0f

/* Kaiser-Bessel window for overlap-add aggregation.
 * Without windowing, patch boundaries create visible grid artifacts in
 * smooth gradients because the set of contributing patches changes
 * abruptly at every stride pixels. The window tapers each patch's
 * contribution: high weight at center, near-zero at edges, ensuring
 * smooth transitions between overlapping patches.
 *
 * w(n) = I0(beta * sqrt(1 - ((2n/(N-1))-1)^2)) / I0(beta)
 * beta = 2.0 (moderate taper), N = GALOSH_BLOCK_SIZE = 8.
 * 2D window = outer product of 1D window. */
static const float galosh_kaiser_1d[GALOSH_BLOCK_SIZE] = {
  0.34012f, 0.59885f, 0.84123f, 0.97659f,
  0.97659f, 0.84123f, 0.59885f, 0.34012f
};
static float galosh_kaiser_2d[GALOSH_BLOCK_PIXELS];
/* Half-res 4x4 Kaiser: averaged from consecutive pairs of the 8-point window.
 * Point-sampling at stride-2 produces an asymmetric window [0.34,0.84,0.98,0.60]
 * which causes visible grid artifacts. Pair-averaging gives symmetric
 * [0.47,0.91,0.91,0.47] for proper overlap-add in half-res aggregation. */
static float galosh_kaiser_half_2d[GALOSH_HALF_PIXELS];
static void init_galosh_kaiser(void)
{
  for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
    for(int dx = 0; dx < GALOSH_BLOCK_SIZE; dx++)
      galosh_kaiser_2d[dy * GALOSH_BLOCK_SIZE + dx] = galosh_kaiser_1d[dy] * galosh_kaiser_1d[dx];

  /* Build symmetric 4-point Kaiser by averaging consecutive pairs */
  float kaiser_half_1d[4];
  for(int i = 0; i < 4; i++)
    kaiser_half_1d[i] = (galosh_kaiser_1d[2 * i] + galosh_kaiser_1d[2 * i + 1]) * 0.5f;
  for(int dy = 0; dy < 4; dy++)
    for(int dx = 0; dx < 4; dx++)
      galosh_kaiser_half_2d[dy * 4 + dx] = kaiser_half_1d[dy] * kaiser_half_1d[dx];
}

/* ================================================================
 * 8-point Walsh-Hadamard Transform (in-place, sequency order)
 *
 * Pure add/subtract -- no multiplication. O(N log N).
 *
 * The WHT is self-inverse up to a scale factor:
 *   WHT(WHT(x)) = N * x
 * So forward and inverse share the same butterfly structure;
 * the only difference is the normalization (1/N for inverse).
 *
 * Sequency ordering ensures that higher-index coefficients
 * correspond to higher spatial frequencies (analogous to DCT),
 * which is important for the shrinkage thresholding to work
 * correctly (low-index = low-frequency = signal, high-index =
 * high-frequency = noise).
 * ================================================================ */
static inline void wht8_inplace(float *x)
{
    /* Stage 1: pairs */
    float a0 = x[0] + x[1], a1 = x[0] - x[1];
    float a2 = x[2] + x[3], a3 = x[2] - x[3];
    float a4 = x[4] + x[5], a5 = x[4] - x[5];
    float a6 = x[6] + x[7], a7 = x[6] - x[7];
    /* Stage 2: quads */
    float b0 = a0 + a2, b1 = a1 + a3;
    float b2 = a0 - a2, b3 = a1 - a3;
    float b4 = a4 + a6, b5 = a5 + a7;
    float b6 = a4 - a6, b7 = a5 - a7;
    /* Stage 3: octets (sequency order) */
    x[0] = b0 + b4;
    x[1] = b1 + b5;
    x[2] = b2 + b6;
    x[3] = b3 + b7;
    x[4] = b0 - b4;
    x[5] = b1 - b5;
    x[6] = b2 - b6;
    x[7] = b3 - b7;
}

/* 2D separable WHT on an 8x8 block (in-place).
 * Forward: call with normalize=false, then coefficients are scaled by 64.
 * Inverse: call with normalize=true, divides all by 64.
 *
 * The separable structure means 2D WHT = row WHT + column WHT,
 * which is O(N^2 log N) instead of O(N^2 * N^2) for direct 2D. */
static void wht2d_8x8(float block[GALOSH_BLOCK_PIXELS], const int normalize)
{
    /* Rows */
    for(int r = 0; r < GALOSH_BLOCK_SIZE; r++)
        wht8_inplace(block + r * GALOSH_BLOCK_SIZE);
    /* Columns */
    for(int c = 0; c < GALOSH_BLOCK_SIZE; c++)
    {
        float col[GALOSH_BLOCK_SIZE];
        for(int r = 0; r < GALOSH_BLOCK_SIZE; r++)
            col[r] = block[r * GALOSH_BLOCK_SIZE + c];
        wht8_inplace(col);
        for(int r = 0; r < GALOSH_BLOCK_SIZE; r++)
            block[r * GALOSH_BLOCK_SIZE + c] = col[r];
    }
    if(normalize)
    {
        const float inv = 1.0f / (float)GALOSH_BLOCK_PIXELS;
        for(int i = 0; i < GALOSH_BLOCK_PIXELS; i++)
            block[i] *= inv;
    }
}

/* ================================================================
 * 4-point WHT (in-place, sequency order) — same family as wht8_inplace
 *
 * EN: Sequency-ordered 4-point Hadamard, used by the 4x4 2D-WHT below
 *     for K13 (half-res L) when block_size = 4.  In sequency order:
 *         y[0] = (++++) · x  = x0+x1+x2+x3   (DC)
 *         y[1] = (++--) · x  = x0+x1-x2-x3   (1 sign change)
 *         y[2] = (+--+) · x  = x0-x1-x2+x3   (2 sign changes)
 *         y[3] = (+-+-) · x  = x0-x1+x2-x3   (3 sign changes)
 *     Implemented via 2-stage radix-2 butterfly (same template as
 *     wht8_inplace).  Self-inverse up to factor 4.
 *
 * JP: 4 点 sequency-ordered WHT。8 点版 (wht8_inplace) と同じ構造の
 *     2 段 butterfly。WHT2D_4x4 の row/col primitive。 */
static inline void wht4_inplace(float *x)
{
    /* Stage 1: pairs */
    const float a0 = x[0] + x[1];
    const float a1 = x[0] - x[1];
    const float a2 = x[2] + x[3];
    const float a3 = x[2] - x[3];
    /* Stage 2: sequency-ordered output (low → high frequency) */
    x[0] = a0 + a2;  /* (+ + + +) DC */
    x[1] = a0 - a2;  /* (+ + - -) lowest non-DC */
    x[2] = a1 - a3;  /* (+ - - +) */
    x[3] = a1 + a3;  /* (+ - + -) Nyquist */
}

/* 2D separable WHT on a 4x4 block (in-place).
 * Used by K13 (half-res L Pass1+2) when --k13-block=4 to match the
 * full-res grain scale produced by K15 8×8 (= 2 × 4 = 8 px at full-res
 * after K14 ×2 upsample, equal to K15's 8 px native scale).
 *
 * Forward: normalize=0 (coefficients scaled by 16).
 * Inverse: normalize=1 (divides by 16).
 *
 * 配列は 16 連続 float (row-major 4×4)。row 4 個 + col 4 個の WHT。
 * normalize=0 で前方 (係数は 16 倍スケール)、=1 で逆方向 (1/16 で除す)。 */
static void wht2d_4x4(float *block, const int normalize)
{
    /* Rows */
    for(int r = 0; r < 4; r++)
        wht4_inplace(block + r * 4);
    /* Columns */
    for(int c = 0; c < 4; c++)
    {
        float col[4];
        for(int r = 0; r < 4; r++)
            col[r] = block[r * 4 + c];
        wht4_inplace(col);
        for(int r = 0; r < 4; r++)
            block[r * 4 + c] = col[r];
    }
    if(normalize)
    {
        const float inv = 1.0f / 16.0f;
        for(int i = 0; i < 16; i++)
            block[i] *= inv;
    }
}

/* WHT decomposition: 8x8 Bayer patch -> 4x4 x 4ch (L, C1, C2, C3).
 * Each 2x2 cell in the 8x8 patch is transformed independently.
 * L = average (luma-like), C1/C2/C3 = differences (chroma-like).
 * Noise variance is preserved: if input noise sigma^2=1 per pixel,
 * each WHT component also has sigma^2=1 (orthogonal with /2 norm).
 *
 * SSE version: processes all 4 columns (j=0..3) simultaneously.
 * Row pair (2i, 2i+1) of the 8x8 patch = 8+8 = 16 consecutive floats.
 * Deinterleave even/odd columns -> a,b,c,d as __m128 -> add/sub. */
static float quick_select_kth(float *arr, const int n, const int k)
{
  if(n <= 0) return 0.0f;
  int lo = 0, hi = n - 1;
  const int mid = k;
  while(lo < hi)
  {
    const float pivot = arr[mid];
    int i = lo, j = hi;
    do {
      while(arr[i] < pivot) i++;
      while(arr[j] > pivot) j--;
      if(i <= j) { const float t = arr[i]; arr[i] = arr[j]; arr[j] = t; i++; j--; }
    } while(i <= j);
    if(j < mid) lo = i;
    if(i > mid) hi = j;
  }
  return arr[mid];
}

static inline float quick_select_median(float *arr, const int n)
{
  return quick_select_kth(arr, n, n / 2);
}

typedef struct galosh_noise_params_t { float alpha, sigma_sq; } galosh_noise_params_t;

static int compare_floats_galosh(const void *a, const void *b)
{
  const float fa = *(const float *)a, fb = *(const float *)b;
  /* NaN-safe: NaN sorts to end */
  if(fa != fa) return 1;
  if(fb != fb) return -1;
  return (fa > fb) - (fa < fb);
}

/* Robust Poisson-Gaussian noise estimation using Laplacian (second-order
 * difference) and MAD. Uses ALL Bayer pixels so that all 4 sub-channels
 * share exactly the same noise parameters -- critical to prevent jaggies.
 *
 * Key insight: the first-difference estimator d = raw[x+2] - raw[x]
 * is contaminated by signal gradients: Var(d) = 2*sigma^2 + gradient^2.
 * For typical images the gradient term dominates, overestimating noise
 * by 10-50x, which makes the GAT over-stabilize (sigma_GAT << 1).
 *
 * The Laplacian (second difference) d = raw[x] - 2*raw[x+2] + raw[x+4]
 * perfectly cancels any linear signal component:
 *   d_signal = f - 2(f+f'h) + (f+2f'h) = 0   (h = 2 pixels)
 * Only noise remains: Var(d) = (1^2 + 2^2 + 1^2)*sigma^2 = 6*sigma^2
 *
 * Binned by intensity -> median(d^2) -> robust sigma^2 per bin -> linear fit
 * gives the Poisson-Gaussian model: noise_var = alpha*intensity + sigma^2.
 *
 * Ref: Foi et al. (2008) lower envelope method */
static galosh_noise_params_t galosh_estimate_noise(const float *raw,
                                                const int width, const int height)
{
  /* Block-based Laplacian + lower envelope + Huber M-estimator + dark pixel pass
   * Estimates noise parameters from raw Bayer data: Var(x) = alpha * E(x) + sigma_sq
   *   alpha: shot noise gain, sigma_sq: read noise variance
   *
   * Algorithm (Foi et al. 2008 lower envelope method):
   *   1. Partition each CFA channel into non-overlapping 8x8 blocks (half resolution)
   *   2. Per block: compute mean and Laplacian-based noise variance (horizontal+vertical)
   *      Intra-block MAD(|L|) is robust even when edge pixels are present
   *   3. Bin by mean intensity. Take 5-20th percentile of block variances (lower envelope =
   *      flattest patches = noise floor)
   *   4. Robust WLS fit using Huber M-estimator (k derived from residual MAD)
   *   5. Refine sigma_sq directly from darkest pixels */
  galosh_noise_params_t result = { 1e-4f, 1e-6f };

#define NE_NBINS 32
#define NE_BLOCK_SZ 8

  const int halfwidth = (width + 1) / 2;
  const int halfheight = (height + 1) / 2;
  const int offsets[4][2] = {{0,0},{0,1},{1,0},{1,1}};

  const int n_bx = halfwidth / NE_BLOCK_SZ;
  const int n_by = halfheight / NE_BLOCK_SZ;
  const int n_blocks_per_ch = n_bx * n_by;
  const int total_blocks = 4 * n_blocks_per_ch;

  if(total_blocks < 100) return result;

  float *blk_mean = dt_alloc_align_float(total_blocks);
  float *blk_var  = dt_alloc_align_float(total_blocks);
  if(!blk_mean || !blk_var)
  {
    dt_free_align(blk_mean); dt_free_align(blk_var);
    return result;
  }

  /* Steps 1-2: per-block mean and Laplacian noise variance (horizontal+vertical) */
  int bi = 0;
  for(int ch = 0; ch < 4; ch++)
  {
    const int dy0 = offsets[ch][0], dx0 = offsets[ch][1];
    for(int by = 0; by < n_by; by++)
    {
      for(int bx = 0; bx < n_bx; bx++)
      {
        const int y0 = by * NE_BLOCK_SZ;
        const int x0 = bx * NE_BLOCK_SZ;

        double sum = 0;
        int np = 0;
        for(int y = y0; y < y0 + NE_BLOCK_SZ; y++)
          for(int x = x0; x < x0 + NE_BLOCK_SZ; x++)
          {
            sum += raw[(2*y+dy0) * width + (2*x+dx0)];
            np++;
          }
        const float bm = (float)(sum / np);

        /* Collect horizontal+vertical Laplacians within the block */
        float laps[256];
        int nl = 0;
        for(int y = y0; y < y0 + NE_BLOCK_SZ; y++)
          for(int x = x0; x < x0 + NE_BLOCK_SZ - 2; x++)
          {
            const float v0 = raw[(2*y+dy0) * width + (2*x+dx0)];
            const float v1 = raw[(2*y+dy0) * width + (2*(x+1)+dx0)];
            const float v2 = raw[(2*y+dy0) * width + (2*(x+2)+dx0)];
            laps[nl++] = fabsf(v0 - 2.0f*v1 + v2);
          }
        for(int y = y0; y < y0 + NE_BLOCK_SZ - 2; y++)
          for(int x = x0; x < x0 + NE_BLOCK_SZ; x++)
          {
            const float v0 = raw[(2*y+dy0) * width + (2*x+dx0)];
            const float v1 = raw[(2*(y+1)+dy0) * width + (2*x+dx0)];
            const float v2 = raw[(2*(y+2)+dy0) * width + (2*x+dx0)];
            laps[nl++] = fabsf(v0 - 2.0f*v1 + v2);
          }

        if(nl > 10)
        {
          const float med = quick_select_median(laps, nl);
          const float sigma_lap = med / 0.6745f;
          blk_var[bi] = (sigma_lap * sigma_lap) / 6.0f;
        }
        else
        {
          blk_var[bi] = 1e10f;
        }
        blk_mean[bi] = bm;
        bi++;
      }
    }
  }
  const int n_total = bi;

  /* Step 3: bin by mean intensity, take lower envelope (5-20th percentile) */
  float global_min = FLT_MAX, global_max = 0.0f;
  for(int i = 0; i < n_total; i++)
  {
    if(blk_mean[i] > 0.003f && blk_mean[i] < 0.97f)
    {
      if(blk_mean[i] < global_min) global_min = blk_mean[i];
      if(blk_mean[i] > global_max) global_max = blk_mean[i];
    }
  }
  const float bw = (global_max - global_min) / NE_NBINS;
  if(bw < 1e-10f)
  {
    dt_free_align(blk_mean); dt_free_align(blk_var);
    return result;
  }

  float bin_mean_arr[NE_NBINS], bin_var_arr[NE_NBINS];
  int bin_valid[NE_NBINS], bin_cnt_arr[NE_NBINS];

  float *sort_buf = dt_alloc_align_float(n_total);
  if(!sort_buf)
  {
    dt_free_align(blk_mean); dt_free_align(blk_var);
    return result;
  }

  int n_valid = 0;
  for(int b = 0; b < NE_NBINS; b++)
  {
    const float bin_lo = global_min + b * bw;
    const float bin_hi = bin_lo + bw;
    bin_valid[b] = 0;

    int cnt = 0;
    double msum = 0;
    for(int i = 0; i < n_total; i++)
    {
      if(blk_mean[i] >= bin_lo && blk_mean[i] < bin_hi
         && blk_mean[i] > 0.003f && blk_mean[i] < 0.97f
         && blk_var[i] < 1e9f)
      {
        sort_buf[cnt] = blk_var[i];
        msum += blk_mean[i];
        cnt++;
      }
    }
    if(cnt < 20) continue;

    /* Sort variances: lower envelope = 5-20th percentile */
    qsort(sort_buf, cnt, sizeof(float), compare_floats_galosh);
    const int p5  = cnt / 20;
    const int p20 = cnt / 5;
    double var_sum = 0;
    int var_cnt = 0;
    for(int i = p5; i <= p20 && i < cnt; i++)
    {
      var_sum += sort_buf[i];
      var_cnt++;
    }
    if(var_cnt > 0)
    {
      bin_var_arr[b] = (float)(var_sum / var_cnt);
      bin_mean_arr[b] = (float)(msum / cnt);
      bin_cnt_arr[b] = var_cnt;
      bin_valid[b] = 1;
      n_valid++;
    }
  }

  dt_free_align(sort_buf);
  dt_free_align(blk_mean);
  dt_free_align(blk_var);

  if(n_valid < 4)
  {
    fprintf(stderr, "[rawdenoise] noise_est: too few valid bins (%d)\n", n_valid);
    return result;
  }

  /* Step 4: robust WLS fit Var = alpha * mean + sigma_sq
   * Huber M-estimator: k = 1.345 * MAD(residuals) / 0.6745 */
  float alpha_est = 0.01f, sigma_sq_est = 0.0f;

  for(int iter = 0; iter < 5; iter++)
  {
    double huber_k = 1e10;
    if(iter > 0)
    {
      float resids[NE_NBINS];
      int nr = 0;
      for(int b = 0; b < NE_NBINS; b++)
      {
        if(!bin_valid[b]) continue;
        resids[nr++] = fabsf(bin_var_arr[b] - (alpha_est * bin_mean_arr[b] + sigma_sq_est));
      }
      const float resid_mad = quick_select_median(resids, nr) / 0.6745f;
      huber_k = 1.345 * fmax(resid_mad, 1e-12);
    }

    double Sw = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    for(int b = 0; b < NE_NBINS; b++)
    {
      if(!bin_valid[b]) continue;
      double w = (double)bin_cnt_arr[b];
      if(iter > 0)
      {
        const double pred = (double)alpha_est * bin_mean_arr[b] + sigma_sq_est;
        const double resid = fabs((double)bin_var_arr[b] - pred);
        if(resid > huber_k) w *= huber_k / resid;
      }
      const double x = bin_mean_arr[b], y = bin_var_arr[b];
      Sw += w; Sx += w*x; Sy += w*y; Sxx += w*x*x; Sxy += w*x*y;
    }
    const double det = Sw * Sxx - Sx * Sx;
    if(fabs(det) > 1e-30)
    {
      float new_alpha = (float)((Sw * Sxy - Sx * Sy) / det);
      float new_sq    = (float)((Sxx * Sy - Sx * Sxy) / det);
      if(new_alpha > 0) alpha_est = new_alpha;
      if(new_sq >= 0) sigma_sq_est = new_sq;
    }
  }

  alpha_est = fmaxf(alpha_est, 1e-8f);

  /* Step 5: estimate sigma_sq directly from darkest pixels */
  {
    const int samp_max = 50000;
    float *samp = dt_alloc_align_float(samp_max);
    if(samp)
    {
      int ns = 0;
      for(int ch = 0; ch < 4 && ns < samp_max; ch++)
      {
        const int dy0 = offsets[ch][0], dx0 = offsets[ch][1];
        for(int y = 0; y < halfheight && ns < samp_max; y += 3)
          for(int x = 0; x < halfwidth && ns < samp_max; x += 3)
            samp[ns++] = raw[(2*y+dy0) * width + (2*x+dx0)];
      }
      const float dark_thresh = quick_select_kth(samp, ns, ns / 10);
      const float dark_max = dark_thresh + 0.02f;

      float *dark_laps = dt_alloc_align_float(samp_max);
      if(dark_laps)
      {
        int ndl = 0;
        for(int ch = 0; ch < 4 && ndl < samp_max; ch++)
        {
          const int dy0 = offsets[ch][0], dx0 = offsets[ch][1];
          /* Horizontal */
          for(int y = 0; y < halfheight && ndl < samp_max; y++)
            for(int x = 0; x < halfwidth - 2 && ndl < samp_max; x++)
            {
              const float v0 = raw[(2*y+dy0) * width + (2*x+dx0)];
              const float v1 = raw[(2*y+dy0) * width + (2*(x+1)+dx0)];
              const float v2 = raw[(2*y+dy0) * width + (2*(x+2)+dx0)];
              if(v0 > dark_max || v1 > dark_max || v2 > dark_max) continue;
              dark_laps[ndl++] = fabsf(v0 - 2.0f*v1 + v2);
            }
          /* Vertical */
          for(int y = 0; y < halfheight - 2 && ndl < samp_max; y++)
            for(int x = 0; x < halfwidth && ndl < samp_max; x++)
            {
              const float v0 = raw[(2*y+dy0) * width + (2*x+dx0)];
              const float v1 = raw[(2*(y+1)+dy0) * width + (2*x+dx0)];
              const float v2 = raw[(2*(y+2)+dy0) * width + (2*x+dx0)];
              if(v0 > dark_max || v1 > dark_max || v2 > dark_max) continue;
              dark_laps[ndl++] = fabsf(v0 - 2.0f*v1 + v2);
            }
        }

        if(ndl > 100)
        {
          const float med = quick_select_median(dark_laps, ndl);
          const float sigma_lap = med / 0.6745f;
          const float dark_var = (sigma_lap * sigma_lap) / 6.0f;
          const float dark_mean = dark_thresh * 0.5f;
          sigma_sq_est = fmaxf(dark_var - alpha_est * dark_mean, 0.0f);
        }
        dt_free_align(dark_laps);
      }
      dt_free_align(samp);
    }
  }

  fprintf(stderr, "[rawdenoise] noise_est: %d bins, alpha=%.6f sigma_sq=%.8f\n",
          n_valid, alpha_est, sigma_sq_est);

  result.alpha = alpha_est;
  result.sigma_sq = sigma_sq_est;
  return result;

#undef NE_NBINS
#undef NE_BLOCK_SZ
}

/* --- GAT (forward / inverse / LUT) ---
 *
 * GAT: Generalized Anscombe Transform (variance stabilization)
 *
 * Forward (piecewise C1 VST extending Foi GAT to all reals):
 *   For y >= y_break = -3*alpha/8:
 *      T(y) = (2/alpha) * sqrt(alpha*y + 3*alpha^2/8 + sigma^2)  (Foi sqrt branch)
 *   For y < y_break:
 *      T(y) = T_break + (y - y_break)/sigma                     (linear branch)
 *   where T_break = 2*sigma/alpha (continuity) and slope = 1/sigma (C1).
 *
 *   Variance is stabilized in BOTH branches to approx 1:
 *     - sqrt branch: stabilizes Poisson(alpha*X) + Gaussian(sigma^2) -> Var ~ 1
 *     - linear branch: in deep negatives, signal-dependent component is
 *       negligible, observed variance is approx sigma^2, linear scaling by 1/sigma
 *       gives Var ~ 1 as well.
 *   No input clamping -> distribution shape preserved across channels
 *   (critical for L/C decomposition where 4ch shape mismatch creates false chroma).
 *
 * Inverse: Exact unbiased inverse via Gauss-Hermite quadrature LUT
 *   in the sqrt branch; analytical linear inverse in the linear branch.
 *   Ref: Anscombe (1948), Foi et al. (Sig.Proc. 2008),
 *        Makitalo & Foi (TIP 2013) */

static inline float gat_forward(const float x, const float alpha, const float sigma_sq)
{
  const float x_safe = (x == x) ? x : 0.0f;  /* NaN guard only -- NO clamping */
  const float y_break = -0.375f * alpha;     /* -3*alpha/8 */
  if(x_safe >= y_break)
  {
    /* sqrt branch: classical Foi GAT */
    return (2.0f / alpha) * sqrtf(alpha * x_safe + 0.375f * alpha * alpha + sigma_sq);
  }
  else
  {
    /* linear branch: C1 extension */
    const float sigma_raw = sqrtf(fmaxf(sigma_sq, 1e-20f));
    const float t_break = 2.0f * sigma_raw / alpha;
    return t_break + (x_safe - y_break) / sigma_raw;
  }
}

/* ================================================================
 * Foi exact unbiased inverse GAT (Makitalo & Foi 2013)
 *
 * Precompute E[GAT(Y)] for Y ~ Poisson-Gaussian(x, alpha, sigma_sq)
 * via Poisson summation + 10-point Gauss-Hermite quadrature.
 * Inverse transform by binary search on monotone lookup table.
 * Significantly reduces bias compared to closed-form approximations,
 * especially at high ISO.
 * ================================================================ */

#define GAT_INV_TABLE_SIZE 4096

typedef struct gat_inv_table_t
{
  float d[GAT_INV_TABLE_SIZE]; /* Expected GAT output values (sorted, monotone) */
  float x[GAT_INV_TABLE_SIZE]; /* Corresponding signal values [0, 1] */
  float d_min, d_max;
  /* Piecewise C1 VST parameters (set at table build time) */
  float alpha;
  float sigma_raw;             /* sqrt(sigma_sq) */
  float y_break;               /* -3*alpha/8: forward-domain transition */
  float t_break;               /* 2*sigma/alpha : GAT-domain transition */
  int valid;
} gat_inv_table_t;

static gat_inv_table_t gat_inv_table = { .valid = 0 };

/* 10-point Gauss-Hermite quadrature (weight function e^{-x^2}).
 * Makitalo & Foi (TIP 2013) use 10 points for the exact unbiased inverse. */
static const double gh_nodes[10] = {
  -3.436159, -2.532732, -1.756684, -1.036611, -0.342901,
   0.342901,  1.036611,  1.756684,  2.532732,  3.436159
};
static const double gh_weights[10] = {
  7.640432855232641e-06, 1.343645746781232e-03, 3.387439445548111e-02,
  2.401386110823147e-01, 6.108626337353258e-01,
  6.108626337353258e-01, 2.401386110823147e-01, 3.387439445548111e-02,
  1.343645746781232e-03, 7.640432855232641e-06
};

static void gat_build_inverse_table(const float alpha, const float sigma_sq)
{
  const double a = (double)alpha;
  const double sq = (double)sigma_sq;
  const double sig = sqrt(fmax(sq, 1e-20));
  const double y_break_d = -0.375 * a;
  const double t_break_d = 2.0 * sig / a;

  for(int i = 0; i < GAT_INV_TABLE_SIZE; i++)
  {
    const double x_val = (double)i / (double)(GAT_INV_TABLE_SIZE - 1); /* [0, 1] */
    const double lambda = x_val / a;  /* Poisson rate */

    /* Summation over Poisson distribution: E[T(Poisson(lambda)*alpha + N(0, sigma_sq))] */
    double expected_gat = 0.0;
    const int k_max = (int)(lambda + 8.0 * sqrt(fmax(lambda, 1.0))) + 20;
    double log_prob = -lambda;  /* log P(0; lambda) */

    for(int k = 0; k <= k_max; k++)
    {
      if(k > 0) log_prob += log(lambda) - log((double)k);
      const double prob = exp(log_prob);
      if(prob < 1e-15 && k > (int)lambda + 1) break;

      double eg = 0.0;
      for(int g = 0; g < 10; g++)
      {
        const double z = 1.4142135623730951 * sig * gh_nodes[g]; /* sqrt(2)*sig*node */
        const double noisy_y = (double)k * a + z;
        double T;
        if(noisy_y >= y_break_d)
        {
          const double arg = a * noisy_y + 0.375 * a * a + sq;
          T = (2.0 / a) * sqrt(fmax(arg, 0.0));
        }
        else
        {
          T = t_break_d + (noisy_y - y_break_d) / sig;
        }
        eg += gh_weights[g] * T;
      }
      eg *= 0.5641895835477563;  /* 1/sqrt(pi) */
      expected_gat += prob * eg;
    }

    gat_inv_table.x[i] = (float)x_val;
    gat_inv_table.d[i] = (float)expected_gat;
  }

  gat_inv_table.d_min = gat_inv_table.d[0];
  gat_inv_table.d_max = gat_inv_table.d[GAT_INV_TABLE_SIZE - 1];
  gat_inv_table.alpha = alpha;
  gat_inv_table.sigma_raw = (float)sig;
  gat_inv_table.y_break = (float)y_break_d;
  gat_inv_table.t_break = (float)t_break_d;
  gat_inv_table.valid = 1;
  fprintf(stderr, "[rawdenoise] piecewise GAT inv table: D range [%.4f, %.4f] "
                  "alpha=%.6f sigma_sq=%.8f y_break=%.6f t_break=%.4f\n",
          gat_inv_table.d_min, gat_inv_table.d_max, alpha, sigma_sq,
          gat_inv_table.y_break, gat_inv_table.t_break);
}

static inline float gat_inverse_exact(const float D)
{
  if(!gat_inv_table.valid) return 0.0f;

  /* Below table coverage: use analytical linear-branch inverse.
   * In the linear branch, T(y) = t_break + (y - y_break)/sigma
   *   => y = y_break + sigma * (T - t_break).
   * No clamp to 0 -- preserves distribution shape. */
  if(D <= gat_inv_table.d_min)
  {
    return gat_inv_table.y_break
         + gat_inv_table.sigma_raw * (D - gat_inv_table.t_break);
  }
  if(D >= gat_inv_table.d_max) return 1.0f;

  /* Binary search on monotone d[] array (sqrt branch, unbiased) */
  int lo = 0, hi = GAT_INV_TABLE_SIZE - 1;
  while(lo < hi - 1)
  {
    const int mid = (lo + hi) >> 1;
    if(gat_inv_table.d[mid] <= D) lo = mid;
    else hi = mid;
  }

  /* Linear interpolation */
  const float d0 = gat_inv_table.d[lo], d1 = gat_inv_table.d[hi];
  const float t = (D - d0) / fmaxf(d1 - d0, 1e-10f);
  return gat_inv_table.x[lo] + t * (gat_inv_table.x[hi] - gat_inv_table.x[lo]);
}

/* Measure actual sigma in GAT domain using MAD of Laplacian (second differences).
 *
 * Laplacian: L = data[x] - 2*data[x+1] + data[x+2] cancels linear gradients.
 * For iid noise with variance sigma^2: Var(L) = (1+4+1)*sigma^2 = 6*sigma^2
 * |L| ~ sqrt(6)*sigma * |N(0,1)|, so MAD(|L|) = sqrt(6)*sigma*0.6745
 * => sigma = MAD / (0.6745 * sqrt(6)) = MAD / 1.6521 */

/* Full-mosaic sigma estimation: uses stride-2 Laplacian to stay within
 * the same Bayer channel. */
static float estimate_gat_sigma_mosaic(const float *data, const int width, const int height)
{
  const int n_samples = MIN(width * height / 6, 200000);
  float *abs_laps = dt_alloc_align_float(n_samples + 1);
  if(!abs_laps) return 1.0f;

  int count = 0;
  for(int y = 0; y < height && count < n_samples; y++)
  {
    const float *row = data + (size_t)y * width;
    for(int x = 0; x < width - 4 && count < n_samples; x += 3)
    {
      const float lap = row[x] - 2.0f * row[x + 2] + row[x + 4];
      abs_laps[count++] = fabsf(lap);
    }
  }

  if(count < 100) { dt_free_align(abs_laps); return 1.0f; }

  const float mad = quick_select_median(abs_laps, count);
  dt_free_align(abs_laps);

  const float sigma = mad / 1.6521f;
  return fmaxf(sigma, 0.01f);
}

static float estimate_gat_sigma_halfres(const float *data, const int width, const int height)
{
  const int n_samples = MIN(width * height / 3, 200000);
  float *abs_laps = dt_alloc_align_float(n_samples + 1);
  if(!abs_laps) return 1.0f;

  int count = 0;
  for(int y = 0; y < height && count < n_samples; y++)
  {
    const float *row = data + (size_t)y * width;
    for(int x = 0; x < width - 2 && count < n_samples; x += 3)
    {
      const float lap = row[x] - 2.0f * row[x + 1] + row[x + 2];
      abs_laps[count++] = fabsf(lap);
    }
  }

  if(count < 100) { dt_free_align(abs_laps); return 1.0f; }

  const float mad = quick_select_median(abs_laps, count);
  dt_free_align(abs_laps);

  const float sigma = mad / 1.6521f;
  return fmaxf(sigma, 0.01f);
}

/* ================================================================
 * GALOSH Pass 1: BayesShrink adaptive hard thresholding
 *
 * For each overlapping 8x8 block in a single 2D plane:
 *   1. Forward 2D WHT
 *   2. Estimate signal variance: sigma_x^2 = max(sigma_Y^2 - sigma^2, 0)
 *      where sigma_Y^2 = (1/N) sum coeff^2 (excluding DC),
 *      sigma = noise std (=1 after GAT normalization)
 *   3. BayesShrink threshold: lambda = sigma^2 / sigma_x
 *      Ref: Chang, Yu & Vetterli, "Adaptive wavelet thresholding for image
 *           denoising and compression", IEEE TIP, 2000
 *   4. Hard threshold: |coeff| < lambda -> 0, DC bin always preserved
 *   5. Inverse 2D WHT
 *   6. Overlap-add with Kaiser window weighting
 *
 * sigma_strength: controls effective noise level (luma_strength or chroma_strength)
 *   sigma_eff = sigma_strength (in GAT-normalized space where true sigma=1)
 *
 * This function operates on a SINGLE 2D plane (e.g. L, C1, C2, or C3).
 * Called 4 times in the pipeline: once per L/C plane.
 * ================================================================ */

/* Forward declarations -- the multiorient wrappers and EWA helper below
 * call galosh_pass1, galosh_pass2 and j1f, all of which are defined
 * later in this header (or in libm).  Explicit decls avoid implicit-
 * declaration warnings under -Wall and the static-after-implicit error
 * that follows. */
static void galosh_pass1(const float *restrict input,
                         float *restrict output,
                         const int width, const int height,
                         const float sigma_strength,
                         const int stride);
static void galosh_pass2(const float *restrict noisy,
                         const float *restrict pilot,
                         float *restrict output,
                         const int width, const int height,
                         const float sigma_strength,
                         const int stride);
/* libm has double j1(double); UCRT64 mingw lacks the float variant j1f.
 * The jinc kernel evaluates at small x values where double precision is
 * well within float range, so casting is harmless. */
extern double j1(double x);

/* ================================================================
 * Bilinear plane rotation -- helper for multi-orientation WHT averaging.
 *
 * Output(x,y) = Input(R(-angle)*(x,y)) with bilinear interpolation,
 * boundary clamp.  Same in/out dimensions.
 *
 * Used by galosh_pass12_multiorient (variant B).  Bilinear is sufficient
 * here because the four orientations average post-shrinkage anyway, and
 * the rotated grid noise re-correlation is bounded by 1 sample in each
 * direction; sinc rotation would be overkill for what is essentially a
 * 4-sample symmetric average over rotational frame mismatches.
 * ================================================================ */
static void galosh_rotate_bilinear(const float *restrict in,
                                    float *restrict out,
                                    const int width, const int height,
                                    const float angle_deg)
{
    const float c = cosf(-angle_deg * (float)M_PI / 180.0f);
    const float s = sinf(-angle_deg * (float)M_PI / 180.0f);
    const float cx = width  * 0.5f;
    const float cy = height * 0.5f;

    DT_OMP_FOR()
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            const float dx = x - cx, dy = y - cy;
            const float sx = c * dx - s * dy + cx;
            const float sy = s * dx + c * dy + cy;
            int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
            float fx = sx - x0, fy = sy - y0;
            int x0c = (x0 < 0) ? 0 : (x0 >= width  ? width  - 1 : x0);
            int y0c = (y0 < 0) ? 0 : (y0 >= height ? height - 1 : y0);
            int x1c = (x0 + 1 < 0) ? 0 : (x0 + 1 >= width  ? width  - 1 : x0 + 1);
            int y1c = (y0 + 1 < 0) ? 0 : (y0 + 1 >= height ? height - 1 : y0 + 1);
            const float v00 = in[(size_t)y0c * width + x0c];
            const float v01 = in[(size_t)y0c * width + x1c];
            const float v10 = in[(size_t)y1c * width + x0c];
            const float v11 = in[(size_t)y1c * width + x1c];
            out[(size_t)y * width + x] =
                (1.0f - fx) * (1.0f - fy) * v00 +
                fx          * (1.0f - fy) * v01 +
                (1.0f - fx) * fy          * v10 +
                fx          * fy          * v11;
        }
    }
}

/* ================================================================
 * Multi-orientation WHT shrinkage (Pass1 + Pass2) wrapper -- variant B.
 *
 * Standard WHT 8x8 shrinkage uses an axis-aligned (Walsh sequency) basis,
 * whose frequency response is square in 2D.  Diagonal features at 45 deg
 * project onto many basis functions and are not sparse, so shrinkage
 * leaves a residual stair-step ("rotation-variant artefact").
 *
 * The textbook fix (Yu & Sapiro & Mallat; rotation-invariant denoising)
 * is to apply the transform in N rotated orientations and average.  For
 * n_orient = 4 we use {0, 45, 90, 135} deg.  At each orientation:
 *   1. rotate input plane by +theta
 *   2. run pass1+pass2 (axis-aligned WHT shrinkage)
 *   3. rotate result back by -theta
 *   4. accumulate
 * Final output is the n_orient-averaged result.
 *
 * Cost: 4x the pass1+pass2 work, but pass1+pass2 itself is ~20% of total
 * pipeline so absolute slowdown is ~1.6x.
 *
 * n_orient == 1 collapses to a single pass1+pass2 call (no rotation).
 * n_orient == 4 uses the four orientations above.  Other values are
 * treated as 1.
 * ================================================================ */
/* Forward declarations for the block-size-parameterized Pass1/Pass2.
 * Definitions live further down (after wht2d primitives + Kaiser init);
 * the multiorient wrapper calls into them, so it needs the prototypes
 * up here. */
static void galosh_pass1_blocked(const float *restrict input,
                                 float *restrict output,
                                 const int width, const int height,
                                 const float sigma_strength,
                                 const int block_size,
                                 const int stride,
                                 const int use_robust_shrink);
static void galosh_pass2_blocked(const float *restrict noisy,
                                  const float *restrict pilot,
                                  float *restrict output,
                                  const int width, const int height,
                                  const float sigma_strength,
                                  const float wiener_floor,
                                  const int block_size,
                                  const int stride);

/* Block-size-parameterized multi-orientation Pass1+Pass2 wrapper.
 * wiener_floor defaults to 1/B (=1/sqrt(N)) per the GALOSH convention.
 *
 * use_robust_shrink controls Pass1's σ_Y estimator (MAD vs mean) — see
 * galosh_pass1_blocked.  Pass2 is empirical Wiener with pilot, unaffected
 * by this flag (pilot quality from Pass1 is what propagates the robust
 * estimate's benefit through to Pass2's surviving-coefficient selection). */
static void galosh_pass12_multiorient_blocked(const float *restrict noisy,
                                               float *restrict output,
                                               const int width, const int height,
                                               const float sigma_strength,
                                               const int block_size,
                                               const int stride,
                                               const int n_orient,
                                               const int use_robust_shrink)
{
    const size_t npix = (size_t)width * height;
    const float wiener_floor = 1.0f / (float)block_size;  /* 1/sqrt(N) */

    /* Single-orientation path: matches the legacy pass1->pass2 sequence. */
    if(n_orient <= 1)
    {
        float *pilot = dt_alloc_align_float(npix);
        if(!pilot) { memcpy(output, noisy, sizeof(float) * npix); return; }
        galosh_pass1_blocked(noisy, pilot, width, height,
                              sigma_strength, block_size, stride,
                              use_robust_shrink);
        galosh_pass2_blocked(noisy, pilot, output, width, height,
                              sigma_strength, wiener_floor, block_size, stride);
        dt_free_align(pilot);
        return;
    }

    /* 4-orientation averaging. */
    const float angles[4] = { 0.0f, 45.0f, 90.0f, 135.0f };

    float *acc          = dt_alloc_align_float(npix);
    float *rotated      = dt_alloc_align_float(npix);
    float *pilot        = dt_alloc_align_float(npix);
    float *denoised_rot = dt_alloc_align_float(npix);
    float *derotated    = dt_alloc_align_float(npix);

    if(!acc || !rotated || !pilot || !denoised_rot || !derotated)
    {
        dt_free_align(acc); dt_free_align(rotated); dt_free_align(pilot);
        dt_free_align(denoised_rot); dt_free_align(derotated);
        memcpy(output, noisy, sizeof(float) * npix);
        return;
    }

    memset(acc, 0, sizeof(float) * npix);

    for(int i = 0; i < 4; i++)
    {
        const float ang = angles[i];
        const float *src;
        if(ang == 0.0f)
        {
            src = noisy;  /* identity rotation: skip the bilinear pass */
        }
        else
        {
            galosh_rotate_bilinear(noisy, rotated, width, height, ang);
            src = rotated;
        }

        galosh_pass1_blocked(src, pilot, width, height,
                              sigma_strength, block_size, stride,
                              use_robust_shrink);
        galosh_pass2_blocked(src, pilot, denoised_rot, width, height,
                              sigma_strength, wiener_floor, block_size, stride);

        if(ang == 0.0f)
        {
            DT_OMP_FOR()
            for(size_t k = 0; k < npix; k++) acc[k] += denoised_rot[k];
        }
        else
        {
            galosh_rotate_bilinear(denoised_rot, derotated, width, height, -ang);
            DT_OMP_FOR()
            for(size_t k = 0; k < npix; k++) acc[k] += derotated[k];
        }
    }

    const float inv_n = 1.0f / 4.0f;
    DT_OMP_FOR()
    for(size_t k = 0; k < npix; k++) output[k] = acc[k] * inv_n;

    dt_free_align(acc);
    dt_free_align(rotated);
    dt_free_align(pilot);
    dt_free_align(denoised_rot);
    dt_free_align(derotated);
}

/* Backward-compat wrapper: legacy 8×8 multi-orient Pass1+2, mean-based
 * sigma_Y (use_robust_shrink=0).  External callers (galosh_yuv_cpu.c) and
 * legacy non-GALOSH_F path see no change. */
static inline void galosh_pass12_multiorient(const float *restrict noisy,
                                              float *restrict output,
                                              const int width, const int height,
                                              const float sigma_strength,
                                              const int stride,
                                              const int n_orient)
{
    galosh_pass12_multiorient_blocked(noisy, output, width, height,
                                       sigma_strength, GALOSH_BLOCK_SIZE,
                                       stride, n_orient,
                                       /*use_robust_shrink=*/0);
}

/* ================================================================
 * Multi-orientation Pass2-only wrapper (with precomputed pilot).
 *
 * Used by the RAW pipeline's full-res L refinement step (compute_L_fullres
 * already produced a pilot, so we only need Pass2 / Wiener).  Same
 * 4-orientation averaging as galosh_pass12_multiorient, but operates on
 * an externally supplied (noisy, pilot) pair.
 *
 * Both `noisy` and `pilot` are rotated together at each orientation to
 * keep the Wiener weighting consistent across the pair.
 * ================================================================ */
static void galosh_pass2_multiorient(const float *restrict noisy,
                                      const float *restrict pilot,
                                      float *restrict output,
                                      const int width, const int height,
                                      const float sigma_strength,
                                      const int stride,
                                      const int n_orient)
{
    const size_t npix = (size_t)width * height;

    if(n_orient <= 1)
    {
        galosh_pass2(noisy, pilot, output, width, height, sigma_strength, stride);
        return;
    }

    const float angles[4] = { 0.0f, 45.0f, 90.0f, 135.0f };

    float *acc           = dt_alloc_align_float(npix);
    float *rot_noisy     = dt_alloc_align_float(npix);
    float *rot_pilot     = dt_alloc_align_float(npix);
    float *denoised_rot  = dt_alloc_align_float(npix);
    float *derotated     = dt_alloc_align_float(npix);

    if(!acc || !rot_noisy || !rot_pilot || !denoised_rot || !derotated)
    {
        dt_free_align(acc); dt_free_align(rot_noisy); dt_free_align(rot_pilot);
        dt_free_align(denoised_rot); dt_free_align(derotated);
        memcpy(output, noisy, sizeof(float) * npix);
        return;
    }

    memset(acc, 0, sizeof(float) * npix);

    for(int i = 0; i < 4; i++)
    {
        const float ang = angles[i];
        const float *src_noisy, *src_pilot;
        if(ang == 0.0f)
        {
            src_noisy = noisy;
            src_pilot = pilot;
        }
        else
        {
            galosh_rotate_bilinear(noisy, rot_noisy, width, height, ang);
            galosh_rotate_bilinear(pilot, rot_pilot, width, height, ang);
            src_noisy = rot_noisy;
            src_pilot = rot_pilot;
        }

        galosh_pass2(src_noisy, src_pilot, denoised_rot, width, height, sigma_strength, stride);

        if(ang == 0.0f)
        {
            DT_OMP_FOR()
            for(size_t k = 0; k < npix; k++) acc[k] += denoised_rot[k];
        }
        else
        {
            galosh_rotate_bilinear(denoised_rot, derotated, width, height, -ang);
            DT_OMP_FOR()
            for(size_t k = 0; k < npix; k++) acc[k] += derotated[k];
        }
    }

    const float inv_n = 1.0f / 4.0f;
    DT_OMP_FOR()
    for(size_t k = 0; k < npix; k++) output[k] = acc[k] * inv_n;

    dt_free_align(acc);
    dt_free_align(rot_noisy);
    dt_free_align(rot_pilot);
    dt_free_align(denoised_rot);
    dt_free_align(derotated);
}

/* Forward declaration -- galosh_jinc is defined further down (alongside
 * galosh_compute_L_fullres_ewajl3) but the generic upsample below is
 * placed earlier so it can be called from anywhere in the pipeline. */
static inline float galosh_jinc(float x);

/* ================================================================
 * Generic single-plane 2x upsample via EWA jinc-windowed-jinc-3.
 *
 * Input:  src half-res plane sized halfwidth x halfheight
 * Output: dst full-res plane sized (2*halfwidth) x (2*halfheight)
 *
 * Used by the *unified* RAW reconstruction path to lift each of L,
 * C1, C2, C3 to full resolution before the inverse 2x2 WHT, replacing
 * the legacy per-block reconstruction that left chroma block-
 * replicated and visibly stair-stepped on edges.
 *
 * Sub-pixel offset convention (in half-res units):
 *   (fr+0, fc+0) : ( 0.0,  0.0)  -- block centroid
 *   (fr+0, fc+1) : ( 0.0, +0.5)  -- horizontal midpoint between this and right block
 *   (fr+1, fc+0) : (+0.5,  0.0)  -- vertical midpoint between this and below block
 *   (fr+1, fc+1) : (+0.5, +0.5)  -- diagonal midpoint between 4 neighbours
 *
 * Kernel: jinc(r_full) * jinc(r_full / 3) for r_full < 3 (output px units).
 * Window: 5x5 half-res samples (W=2).  Negative-ringing wsum is OK
 *   numerically: every output is sum * (1/wsum) and sum has the same
 *   sign as wsum so the normalisation just rescales positive
 *   contributions; tested stable for all four sub-pixel positions.
 * ================================================================ */
static void galosh_upsample_2x_ewajl3(const float *restrict src,
                                       float *restrict dst,
                                       const int halfwidth, const int halfheight)
{
    const int fw = 2 * halfwidth;
    const int fh = 2 * halfheight;

    const float subpix[4][2] = {
        {  0.00f,  0.00f },
        {  0.00f, +0.50f },
        { +0.50f,  0.00f },
        { +0.50f, +0.50f },
    };

    const int W  = 2;
    const int kw = 2 * W + 1;
    float weights[4][5][5];

    for(int si = 0; si < 4; si++)
    {
        const float oy = subpix[si][0];
        const float ox = subpix[si][1];
        float wsum = 0.0f;
        for(int dy = -W; dy <= W; dy++)
        {
            for(int dx = -W; dx <= W; dx++)
            {
                const float ry = (float)dy - oy;
                const float rx = (float)dx - ox;
                const float r_half = sqrtf(rx * rx + ry * ry);
                const float r_full = r_half * 2.0f;
                float w_val = 0.0f;
                if(r_full < 3.0f)
                    w_val = galosh_jinc(r_full) * galosh_jinc(r_full / 3.0f);
                weights[si][dy + W][dx + W] = w_val;
                wsum += w_val;
            }
        }
        /* wsum can be negative for the (+0.5, +0.5) case; that is fine
         * because every contributing weight has the same sign as wsum
         * so the normalised weights all come out positive. */
        const float inv_wsum = 1.0f / wsum;
        for(int dy = 0; dy < kw; dy++)
            for(int dx = 0; dx < kw; dx++)
                weights[si][dy][dx] *= inv_wsum;
    }

    DT_OMP_FOR()
    for(int hy = 0; hy < halfheight; hy++)
    {
        for(int hx = 0; hx < halfwidth; hx++)
        {
            for(int si = 0; si < 4; si++)
            {
                const int sub_dy = si / 2;
                const int sub_dx = si % 2;
                const int fr = 2 * hy + sub_dy;
                const int fc = 2 * hx + sub_dx;
                if(fr >= fh || fc >= fw) continue;

                float sum = 0.0f;
                for(int dy = -W; dy <= W; dy++)
                {
                    int hyi = hy + dy;
                    if(hyi < 0)            hyi = 0;
                    if(hyi >= halfheight)  hyi = halfheight - 1;
                    for(int dx = -W; dx <= W; dx++)
                    {
                        int hxi = hx + dx;
                        if(hxi < 0)           hxi = 0;
                        if(hxi >= halfwidth)  hxi = halfwidth - 1;
                        sum += src[(size_t)hyi * halfwidth + hxi]
                             * weights[si][dy + W][dx + W];
                    }
                }
                dst[(size_t)fr * fw + fc] = sum;
            }
        }
    }
}

/* ================================================================
 * EWA Jinc-Lanczos-3 jinc helper (shared between upsample_2x and the
 * RAW-specific compute_L_fullres variant C).
 *
 * jinc(x) = 2 * J1(pi*x) / (pi*x),  jinc(0) = 1.
 * The 2D Fourier dual of a circular aperture (= 2D sinc analogue);
 * frequency response is a perfect circle, so EWA reconstruction does
 * not exhibit the axis-aligned anisotropy that separable Lanczos has.
 * jinc(r) * jinc(r/3) is the windowed kernel ("EWA Lanczos-Jinc-3"),
 * same family as mpv's `ewa_lanczossharp`.
 * ================================================================ */
static inline float galosh_jinc(float x)
{
    /* Limit at x->0: 2*J1(pi*x)/(pi*x) -> 1.  Taylor series for stability. */
    if(fabsf(x) < 1e-5f) return 1.0f;
    const double pix = M_PI * (double)x;
    return (float)(2.0 * j1(pix) / pix);
}

/* galosh_compute_L_fullres_ewajl3 (RAW-specific variant C alternative
 * upsample) was moved to galosh_raw_cpu.c since it operates on RAW Bayer
 * 2x2 sub-pixel offsets and is invoked only from gat_galosh_denoise_rawlc. */

/* ================================================================
 * GALOSH Pass 1 (block-size-parameterized).
 *
 * EN: Empirical Bayes hard-threshold pilot, generalized over block size B
 *     (2D B×B WHT-LOSH).  Switching B between 8 (legacy K15 / original
 *     GALOSH) and 4 (K13 4×4 grain-scale matched variant) is purely a
 *     parameter — same BayesShrink + VisuShrink-fallback formulae apply
 *     with N = B*B, only the WHT primitive and Kaiser window are
 *     dispatched on B.  K13 4×4 + K15 8×8 yields matched 8-pixel grain
 *     scale at full resolution (= 2·B_K13 = B_K15) which is the only
 *     block-size pair consistent with the half→full-res hierarchy that
 *     K11's fixed 2×2 Bayer decompose imposes.
 *
 * JP: BayesShrink hard-threshold pilot を block size B でパラメータ化。
 *     B=8 (legacy K15 / 全res WHT-LOSH) と B=4 (K13 4×4: K15 8×8 と
 *     full-res grain scale を 8 px に揃える variant) で同じ BayesShrink
 *     公式が成立、WHT primitive と Kaiser window だけ B で切り替え。
 *     K13 4×4 + K15 8×8 が full-res で grain scale 一致 (2*4=8=8) する
 *     唯一の組合せで、K11 の Bayer 2×2 階層から自然に導かれる。
 * ================================================================ */
static void galosh_pass1_blocked(const float *restrict input,
                                 float *restrict output,
                                 const int width, const int height,
                                 const float sigma_strength,
                                 const int block_size,
                                 const int stride,
                                 const int use_robust_shrink)
{
    const int B = block_size;
    const int N = B * B;
    const int rmax = height - B;
    const int cmax = width  - B;
    const int npix = width * height;

    /* Dispatch WHT primitive and Kaiser window on block size.
     * (4 and 8 are the only supported B; K11 2×2 Bayer hierarchy fixes
     * the meaningful block sizes for K13/K15 to {4, 8}.) */
    void (*wht_fn)(float *, int);
    const float *kaiser;
    if(B == 8)      { wht_fn = wht2d_8x8;   kaiser = galosh_kaiser_2d;      }
    else if(B == 4) { wht_fn = wht2d_4x4;   kaiser = galosh_kaiser_half_2d; }
    else            { memcpy(output, input, sizeof(float) * npix); return; }

    /* Per-thread accumulation buffers (allocated up front so the post-loop
     * merge can be a parallel-for reduction instead of a serializing
     * `#pragma omp critical` loop -- the merge had been ~64 M sequential
     * float adds per pass on a 16 MP image with 4 threads, now runs at
     * thread parallelism over npix). */
    const int n_threads_max = omp_get_max_threads();
    float **t_numer = (float **)malloc(n_threads_max * sizeof(float *));
    float **t_denom = (float **)malloc(n_threads_max * sizeof(float *));
    if(!t_numer || !t_denom)
    {
        free(t_numer); free(t_denom);
        memcpy(output, input, sizeof(float) * npix);
        return;
    }
    int t_alloc_ok = 1;
    for(int t = 0; t < n_threads_max; t++)
    {
        t_numer[t] = (float *)dt_alloc_align(64, sizeof(float) * npix);
        t_denom[t] = (float *)dt_alloc_align(64, sizeof(float) * npix);
        if(!t_numer[t] || !t_denom[t]) { t_alloc_ok = 0; }
        else { memset(t_numer[t], 0, sizeof(float) * npix);
               memset(t_denom[t], 0, sizeof(float) * npix); }
    }
    if(!t_alloc_ok)
    {
        for(int t = 0; t < n_threads_max; t++) {
            if(t_numer[t]) dt_free_align(t_numer[t]);
            if(t_denom[t]) dt_free_align(t_denom[t]);
        }
        free(t_numer); free(t_denom);
        memcpy(output, input, sizeof(float) * npix);
        return;
    }

    const float sigma_sq = sigma_strength * sigma_strength;
    /* VisuShrink fallback for flat blocks: lambda_max = sigma * sqrt(2 ln N) */
    const float lambda_max = sigma_strength * sqrtf(2.0f * logf((float)N));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        float *my_numer = t_numer[tid];
        float *my_denom = t_denom[tid];

        #pragma omp for schedule(dynamic, 4)
        for(int ref_r = 0; ref_r <= rmax; ref_r += stride)
        {
            for(int ref_c = 0; ref_c <= cmax; ref_c += stride)
            {
                /* Extract B×B block (max stack alloc 64 floats covers both B=4 and B=8) */
                float block[GALOSH_BLOCK_PIXELS];
                for(int dy = 0; dy < B; dy++)
                    memcpy(block + dy * B,
                           input + (ref_r + dy) * width + ref_c,
                           B * sizeof(float));

                /* Forward 2D WHT (unnormalized: coefficients scaled by N) */
                wht_fn(block, 0);

                /* sigma_Y^2 from AC coefficients (exclude DC = block[0]).
                 * Two estimators:
                 *
                 * mean (use_robust_shrink == 0, legacy):
                 *   sigma_Y^2 = sum_sq / ((N-1) * N)
                 *   Maximum-likelihood under iid Gaussian, but L2 sensitive
                 *   to outliers — when noise happens to cluster spatially
                 *   the WHT compacts it into a few large coefficients, sum_sq
                 *   is inflated, lambda becomes too small, and the cluster
                 *   survives shrinkage as a "false signal" blob.
                 *
                 * MAD (use_robust_shrink == 1, robust):
                 *   sigma_Y = (median |coef|) / 0.6745 in *unnormalized* WHT
                 *   scale, since median(|N(0, N·σ²)|) = 0.6745·sqrt(N)·σ.
                 *   Then sigma_Y^2_per_pixel = (mad / 0.6745)^2 / N.
                 *   Robust to up to ~25% outlier coefficients (median is in
                 *   the bulk of the distribution).  When a few coefficients
                 *   are spuriously large from a noise cluster, the median
                 *   stays at the typical noise magnitude, sigma_Y stays ≈ 1,
                 *   sigma_X estimates 0, lambda → ∞, and the cluster is
                 *   killed.  This is Donoho-Johnstone (1995) robust noise
                 *   estimation transplanted from wavelet shrinkage to
                 *   GALOSH's local WHT-LOSH.
                 *
                 * MAD は GALOSH の GPU/streaming 互換性を保ったまま、孤立
                 * noise cluster を構造的に消す唯一の局所手法 (NL に踏み込まず
                 * BM3D-style cluster suppression に近づける)。 */
                float sigma_y_sq;
                if(use_robust_shrink)
                {
                    float abs_coefs[GALOSH_BLOCK_PIXELS];  /* max 64, holds N-1 ACs */
                    for(int i = 1; i < N; i++)
                        abs_coefs[i - 1] = fabsf(block[i]);
                    const float mad = quick_select_median(abs_coefs, N - 1);
                    /* mad estimates 0.6745·sqrt(N)·σ_Y in unnormalized scale,
                     * so per-pixel σ_Y² = (mad / 0.6745)² / N. */
                    const float scale = mad / 0.6745f;
                    sigma_y_sq = (scale * scale) / (float)N;
                }
                else
                {
                    float sum_sq = 0.0f;
                    for(int i = 1; i < N; i++)
                        sum_sq += block[i] * block[i];
                    sigma_y_sq = sum_sq / ((float)(N - 1) * (float)N);
                }

                /* sigma_X^2 = max(sigma_Y^2 - sigma^2, 0) */
                const float sigma_x_sq = fmaxf(sigma_y_sq - sigma_sq, 0.0f);

                /* BayesShrink threshold (in unnormalized WHT scale) */
                float lambda;
                if(sigma_x_sq < 1e-10f)
                {
                    /* Flat block: kill all AC (only noise) */
                    lambda = 1e30f;
                }
                else
                {
                    /* lambda = sigma^2 / sigma_x scaled by sqrt(N) for unnormalized coef domain */
                    lambda = (sigma_sq / sqrtf(sigma_x_sq)) * sqrtf((float)N);
                    const float lambda_max_unorm = lambda_max * sqrtf((float)N);
                    if(lambda > lambda_max_unorm) lambda = lambda_max_unorm;
                }

                /* Hard threshold: zero coefficients below lambda, preserve DC */
                int n_nonzero = 1; /* DC always kept */
                for(int i = 1; i < N; i++)
                {
                    if(fabsf(block[i]) < lambda)
                        block[i] = 0.0f;
                    else
                        n_nonzero++;
                }

                /* Inverse 2D WHT (with normalization: divide by N) */
                wht_fn(block, 1);

                /* Overlap-add with Kaiser window */
                const float weight = 1.0f / (float)n_nonzero;
                for(int dy = 0; dy < B; dy++)
                    for(int dx = 0; dx < B; dx++)
                    {
                        const int pos = (ref_r + dy) * width + (ref_c + dx);
                        const float kw = kaiser[dy * B + dx];
                        my_numer[pos] += weight * kw * block[dy * B + dx];
                        my_denom[pos] += weight * kw;
                    }
            }
        }
    }  /* end omp parallel */

    /* Parallel reduction across threads + per-pixel normalization in one
     * pass.  Replaces the legacy omp-critical merge (which was sequential
     * O(npix * n_threads)) with a thread-parallel sum, fixing the merge
     * bottleneck on multi-core machines. */
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < npix; i++)
    {
        float n = 0.0f, d = 0.0f;
        for(int t = 0; t < n_threads_max; t++)
        {
            n += t_numer[t][i];
            d += t_denom[t][i];
        }
        output[i] = (d > 1e-10f) ? n / d : input[i];
    }

    for(int t = 0; t < n_threads_max; t++)
    {
        dt_free_align(t_numer[t]);
        dt_free_align(t_denom[t]);
    }
    free(t_numer);
    free(t_denom);
}

/* Backward-compat wrapper: legacy 8×8 BayesShrink pilot, mean-based
 * sigma_Y estimate (use_robust_shrink=0).  Existing callers (rawdenoise_v6.c,
 * yuv_galosh_core.c, galosh_yuv_cpu.c, raw_cpu.c legacy non-GALOSH_F path)
 * see no behaviour change. */
static inline void galosh_pass1(const float *restrict input,
                                float *restrict output,
                                const int width, const int height,
                                const float sigma_strength,
                                const int stride)
{
    galosh_pass1_blocked(input, output, width, height,
                          sigma_strength, GALOSH_BLOCK_SIZE, stride,
                          /*use_robust_shrink=*/0);
}

/* ================================================================
 * GALOSH Pass 2: Wiener shrinkage using pilot estimate
 *
 * Uses Pass 1 output as pilot (signal estimate) to compute
 * empirical Wiener gains:
 *   w(k) = |Y_pilot(k)|^2 / (|Y_pilot(k)|^2 + sigma^2)
 *
 * This is the textbook James-Stein / empirical Wiener estimator.
 * With a good pilot, this achieves near-oracle performance.
 *
 * Like Pass 1, this operates on a SINGLE 2D plane.
 * Called 4 times in the pipeline: once per L/C plane.
 *
 * WHT normalization note:
 *   Unnormalized WHT scales coefficients by N (=64 for 8x8).
 *   Coefficient energy in unnormalized domain: coeff^2 = (N * true_coeff)^2
 *   Noise variance in unnormalized domain: sigma^2 * N^2
 *   These cancel in the Wiener ratio, so the formula works directly
 *   on unnormalized coefficients with sigma^2_unorm = sigma^2 * N^2.
 * ================================================================ */
/* ================================================================
 * GALOSH Pass 2 (block-size-parameterized).
 *
 * EN: Empirical Wiener using pilot estimate, generalized over block size
 *     B (= 4 or 8).  Wiener floor and DC protection apply identically
 *     regardless of B; only the WHT primitive and Kaiser window switch.
 *     wiener_floor scaling: 1/sqrt(N) gives the per-block-size default
 *     (B=8 → 0.125, B=4 → 0.25), but caller passes it explicitly.
 *
 * JP: Wiener pass を block size B でパラメータ化。Pass1 と同じ B={4,8}
 *     dispatch、wiener_floor は明示渡し。default は 1/sqrt(N) で B=4 だと
 *     0.25 (8×8 の倍に粗く保護)。
 * ================================================================ */
static void galosh_pass2_blocked(const float *restrict noisy,
                                  const float *restrict pilot,
                                  float *restrict output,
                                  const int width, const int height,
                                  const float sigma_strength,
                                  const float wiener_floor,
                                  const int block_size,
                                  const int stride)
{
    const int B = block_size;
    const int N = B * B;
    const int rmax = height - B;
    const int cmax = width  - B;
    const int npix = width * height;

    void (*wht_fn)(float *, int);
    const float *kaiser;
    if(B == 8)      { wht_fn = wht2d_8x8;   kaiser = galosh_kaiser_2d;      }
    else if(B == 4) { wht_fn = wht2d_4x4;   kaiser = galosh_kaiser_half_2d; }
    else            { memcpy(output, noisy, sizeof(float) * npix); return; }

    /* Per-thread accumulation buffers (see pass1_blocked for rationale --
     * eliminates the omp-critical merge bottleneck via parallel-for
     * reduction across threads). */
    const int n_threads_max = omp_get_max_threads();
    float **t_numer = (float **)malloc(n_threads_max * sizeof(float *));
    float **t_denom = (float **)malloc(n_threads_max * sizeof(float *));
    if(!t_numer || !t_denom)
    {
        free(t_numer); free(t_denom);
        memcpy(output, noisy, sizeof(float) * npix);
        return;
    }
    int t_alloc_ok = 1;
    for(int t = 0; t < n_threads_max; t++)
    {
        t_numer[t] = (float *)dt_alloc_align(64, sizeof(float) * npix);
        t_denom[t] = (float *)dt_alloc_align(64, sizeof(float) * npix);
        if(!t_numer[t] || !t_denom[t]) { t_alloc_ok = 0; }
        else { memset(t_numer[t], 0, sizeof(float) * npix);
               memset(t_denom[t], 0, sizeof(float) * npix); }
    }
    if(!t_alloc_ok)
    {
        for(int t = 0; t < n_threads_max; t++) {
            if(t_numer[t]) dt_free_align(t_numer[t]);
            if(t_denom[t]) dt_free_align(t_denom[t]);
        }
        free(t_numer); free(t_denom);
        memcpy(output, noisy, sizeof(float) * npix);
        return;
    }

    /* sigma^2 in unnormalized WHT domain: sigma^2 * N */
    const float sigma_sq_unorm = sigma_strength * sigma_strength * (float)N;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        float *my_numer = t_numer[tid];
        float *my_denom = t_denom[tid];

        #pragma omp for schedule(dynamic, 4)
        for(int ref_r = 0; ref_r <= rmax; ref_r += stride)
        {
            for(int ref_c = 0; ref_c <= cmax; ref_c += stride)
            {
                float blk_noisy[GALOSH_BLOCK_PIXELS];
                float blk_pilot[GALOSH_BLOCK_PIXELS];
                for(int dy = 0; dy < B; dy++)
                {
                    memcpy(blk_noisy + dy * B,
                           noisy + (ref_r + dy) * width + ref_c,
                           B * sizeof(float));
                    memcpy(blk_pilot + dy * B,
                           pilot + (ref_r + dy) * width + ref_c,
                           B * sizeof(float));
                }

                wht_fn(blk_noisy, 0);
                wht_fn(blk_pilot, 0);

                float wiener_energy = 0.0f;
                for(int i = 0; i < N; i++)
                {
                    float w;
                    if(i == 0)
                    {
                        w = 1.0f; /* DC protected */
                    }
                    else
                    {
                        const float s2 = blk_pilot[i] * blk_pilot[i];
                        w = s2 / (s2 + sigma_sq_unorm);
                        if(w < wiener_floor) w = wiener_floor;
                    }
                    blk_noisy[i] *= w;
                    wiener_energy += w * w;
                }

                wht_fn(blk_noisy, 1);

                const float weight = 1.0f / fmaxf(wiener_energy, 1e-6f);
                for(int dy = 0; dy < B; dy++)
                    for(int dx = 0; dx < B; dx++)
                    {
                        const int pos = (ref_r + dy) * width + (ref_c + dx);
                        const float kw = kaiser[dy * B + dx];
                        my_numer[pos] += weight * kw * blk_noisy[dy * B + dx];
                        my_denom[pos] += weight * kw;
                    }
            }
        }
    }  /* end omp parallel */

    /* Parallel reduction + per-pixel normalization (replaces omp-critical
     * merge -- see pass1_blocked). */
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < npix; i++)
    {
        float n = 0.0f, d = 0.0f;
        for(int t = 0; t < n_threads_max; t++)
        {
            n += t_numer[t][i];
            d += t_denom[t][i];
        }
        output[i] = (d > 1e-10f) ? n / d : noisy[i];
    }

    for(int t = 0; t < n_threads_max; t++)
    {
        dt_free_align(t_numer[t]);
        dt_free_align(t_denom[t]);
    }
    free(t_numer);
    free(t_denom);
}

/* Backward-compat wrapper: legacy 8×8 Pass2 with caller-specified floor. */
static inline void galosh_pass2_ex(const float *restrict noisy,
                                    const float *restrict pilot,
                                    float *restrict output,
                                    const int width, const int height,
                                    const float sigma_strength,
                                    const float wiener_floor,
                                    const int stride)
{
    galosh_pass2_blocked(noisy, pilot, output, width, height,
                          sigma_strength, wiener_floor,
                          GALOSH_BLOCK_SIZE, stride);
}

/* Convenience wrapper: standard Wiener (original floor) */
static void galosh_pass2(const float *restrict noisy,
                         const float *restrict pilot,
                         float *restrict output,
                         const int width, const int height,
                         const float sigma_strength,
                         const int stride)
{
    galosh_pass2_ex(noisy, pilot, output, width, height,
                    sigma_strength, GALOSH_WIENER_FLOOR, stride);
}

/* ================================================================
 * CFA-protected GALOSH Pass 1 & 2 for full-res raw Bayer.
 *
 * On full-res raw, the CFA Bayer pattern creates color information
 * at specific WHT frequency bins: (0,1), (1,0), (1,1). These carry
 * the DC of inter-channel differences (R−Gr, R−Gb, R−Gr−Gb+B).
 * Standard GALOSH shrinks these in dark areas → homogenizes channels
 * toward G-biased luma → green color shift.
 *
 * CFA-protected variants: treat indices {0, 1, 8, 9} the same as DC
 * — always preserved, never shrunk. Only 3 extra protected coefficients
 * out of 64 (4.7%), so denoising impact is minimal.
 *
 * 全解像度 raw Bayer 上の GALOSH で、CFA 色 DC 係数 (0,1), (1,0), (1,1)
 * を shrinkage から保護。暗部でのチャンネル均一化を防ぎ、green shift を
 * 抑制する。保護するのは 64 係数中 3 個 (4.7%) のみ。
 * ================================================================ */

/* ================================================================
 * galosh_loess_chroma: luma-guided chroma denoise via locally-weighted
 * linear regression (LOESS / bilateral-weighted guided filter).
 *
 * EN: For each half-res pixel (x,y), fit the local linear model
 *         Cb_i = a·Y_i + b + ε_i,  i ∈ (2·R+1)² window
 *     weighted by w_i = exp(-(Y_i-Y_c)²/(2·σ²)) so neighbours whose
 *     luma differs from the centre's by >σ are excluded (this makes
 *     the regression robust to specular highlights in silver surfaces).
 *     Bayes MAP with Gaussian prior a~N(0,τ²) gives
 *         a = cov(Y,Cb) / (var(Y) + 1/τ²),   b = mean_Cb - a·mean_Y
 *     yielding  Cb_out = a·Y_c + b.  In GAT-stabilised space per-pixel
 *     noise is uniformly ≈1, so τ²=1 (|a|≤2 prior) gives ε=1/τ²=1.
 *
 *     Drop-in replacement for the 3-plane WHT-LOSH chroma pipeline
 *     (galosh_pass1 + galosh_pass2 on each of C1, C2, C3); mirrors the
 *     GPU `galosh_yuv_guided_loess` kernel exactly (same formula, same
 *     τ², same bandwidth σ).  The L plane keeps WHT-LOSH; only the
 *     three chroma sub-bands switch to LOESS.
 *
 * JP: 半解像度各画素 (x,y) を中心に (2R+1)² 窓を bilateral 重み
 *     w_i = exp(-(Y_i-Y_c)²/(2σ²)) 付きで局所線形回帰。σ=3 (GAT per-pixel
 *     std=1 の 3σ) で silver+specular bimodal window を silver 側のみに
 *     絞り込み、silver 面の赤青粒を防ぐ。Bayes MAP から ε=1/τ²=1。
 *     WHT-LOSH の pass1+pass2 chroma をこの 1 関数で置換。GPU の
 *     galosh_yuv_guided_loess と完全同一理論。 */
#ifndef GALOSH_LOESS_RADIUS
#define GALOSH_LOESS_RADIUS 7          /* 15×15 window matches GPU default */
#endif
#ifndef GALOSH_LOESS_BW
#define GALOSH_LOESS_BW 3.0f           /* σ in GAT units (per-pixel noise ≈1) */
#endif
#ifndef GALOSH_LOESS_TAU_SQ_INV
#define GALOSH_LOESS_TAU_SQ_INV 1.0f   /* ε = 1/τ²  (τ²=1 → |a|≤2 prior) */
#endif

static inline int reflect_idx(int i, int n)
{
  if(i < 0) return -i;
  if(i >= n) return 2 * n - i - 2;
  return i;
}

static void galosh_loess_chroma(const float *restrict y_guide,
                                const float *restrict cb_in,
                                const float *restrict cr_in,
                                float *restrict cb_out,
                                float *restrict cr_out,
                                const int width, const int height,
                                const float strength_c)
{
  const float eps_gat = strength_c * strength_c * GALOSH_LOESS_TAU_SQ_INV;
  const float inv_2sigma_sq = 1.0f / (2.0f * GALOSH_LOESS_BW * GALOSH_LOESS_BW);
  const int R = GALOSH_LOESS_RADIUS;

  const int x0_int = (R < width)  ? R          : width;
  const int x1_int = (width  > R) ? width - R  : 0;
  const int y0_int = (R < height) ? R          : height;
  const int y1_int = (height > R) ? height - R : 0;

  DT_OMP_FOR()
  for(int y = 0; y < height; y++)
  {
    const int y_interior = (y >= y0_int && y < y1_int);
    for(int x = 0; x < width; x++)
    {
      const size_t cx = (size_t)y * width + x;
      const float Y_c = y_guide[cx];

      float sumW = 0.f, sumY = 0.f, sumYY = 0.f;
      float sumCb = 0.f, sumCr = 0.f;
      float sumYCb = 0.f, sumYCr = 0.f;

      if(y_interior && x >= x0_int && x < x1_int)
      {
        /* Interior fast path: no boundary reflection, contiguous reads.
         * Inner (2R+1) loop is linear-stride, gcc auto-vectorises. */
        for(int dy = -R; dy <= R; dy++)
        {
          const size_t row_off = (size_t)(y + dy) * width;
          const float *rowY  = y_guide + row_off;
          const float *rowCb = cb_in   + row_off;
          const float *rowCr = cr_in   + row_off;
          for(int dx = -R; dx <= R; dx++)
          {
            const int xi = x + dx;
            const float Yi  = rowY [xi];
            const float Cbi = rowCb[xi];
            const float Cri = rowCr[xi];
            const float dY  = Yi - Y_c;
            const float w   = expf(-dY * dY * inv_2sigma_sq);
            sumW   += w;
            sumY   += w * Yi;
            sumYY  += w * Yi * Yi;
            sumCb  += w * Cbi;
            sumCr  += w * Cri;
            sumYCb += w * Yi * Cbi;
            sumYCr += w * Yi * Cri;
          }
        }
      }
      else
      {
        /* Slow path: reflect padding at image edges (≤1% of pixels). */
        for(int dy = -R; dy <= R; dy++)
        {
          const int yi = reflect_idx(y + dy, height);
          const float *rowY  = y_guide + (size_t)yi * width;
          const float *rowCb = cb_in   + (size_t)yi * width;
          const float *rowCr = cr_in   + (size_t)yi * width;
          for(int dx = -R; dx <= R; dx++)
          {
            const int xi = reflect_idx(x + dx, width);
            const float Yi  = rowY [xi];
            const float Cbi = rowCb[xi];
            const float Cri = rowCr[xi];
            const float dY  = Yi - Y_c;
            const float w   = expf(-dY * dY * inv_2sigma_sq);
            sumW   += w;
            sumY   += w * Yi;
            sumYY  += w * Yi * Yi;
            sumCb  += w * Cbi;
            sumCr  += w * Cri;
            sumYCb += w * Yi * Cbi;
            sumYCr += w * Yi * Cri;
          }
        }
      }

      const float invW     = 1.0f / fmaxf(sumW, 1e-10f);
      const float mean_Y   = sumY   * invW;
      const float mean_YY  = sumYY  * invW;
      const float mean_Cb  = sumCb  * invW;
      const float mean_Cr  = sumCr  * invW;
      const float mean_YCb = sumYCb * invW;
      const float mean_YCr = sumYCr * invW;

      const float var_Y   = fmaxf(mean_YY - mean_Y * mean_Y, 0.0f);
      const float cov_YCb = mean_YCb - mean_Y * mean_Cb;
      const float cov_YCr = mean_YCr - mean_Y * mean_Cr;
      const float denom   = fmaxf(var_Y + eps_gat, 1e-6f);
      const float a_cb    = cov_YCb / denom;
      const float a_cr    = cov_YCr / denom;
      const float b_cb    = mean_Cb - a_cb * mean_Y;
      const float b_cr    = mean_Cr - a_cr * mean_Y;

      cb_out[cx] = a_cb * Y_c + b_cb;
      cr_out[cx] = a_cr * Y_c + b_cr;
    }
  }
}

#pragma GCC diagnostic pop

#endif /* GALOSH_CPU_H */
