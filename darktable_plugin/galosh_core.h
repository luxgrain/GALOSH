/* galosh_core.h — shared GALOSH algorithm core (noise model, GAT, WHT/LOSH,
 * LOESS guided-chroma).  Included by both galosh_raw_cpu.c (Bayer RAW
 * pipeline) and galosh_yuv_cpu.c (sRGB YCbCr pipeline).  All core
 * functions are `static` so each translation unit gets its own copy
 * (matches darktable plugin convention where a single .c file is dropped
 * into an IOP and can't share symbols).
 *
 * This file was extracted from the original self-contained rawdenoise_v6.c;
 * raw-specific bits (is_cfa_protected, wht_decompose_8x8, pass1_cfa,
 * pass2_cfa, compute_L_fullres, gat_galosh_denoise_rawlc, main) stay in
 * galosh_raw_cpu.c.  Y-GAT / sRGB conversion / Makitalo inverse for Y
 * plane are provided by galosh_yuv_cpu.c.
 */
#ifndef GALOSH_CORE_H
#define GALOSH_CORE_H

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


static void galosh_pass1(const float *restrict input,
                         float *restrict output,
                         const int width, const int height,
                         const float sigma_strength,
                         const int stride)
{
    const int rmax = height - GALOSH_BLOCK_SIZE;
    const int cmax = width  - GALOSH_BLOCK_SIZE;
    const int npix = width * height;

    /* Accumulation buffers */
    float *numer = (float *)dt_alloc_align(64, sizeof(float) * npix);
    float *denom = (float *)dt_alloc_align(64, sizeof(float) * npix);
    if(!numer || !denom)
    {
        if(numer) dt_free_align(numer);
        if(denom) dt_free_align(denom);
        memcpy(output, input, sizeof(float) * npix);
        return;
    }
    memset(numer, 0, sizeof(float) * npix);
    memset(denom, 0, sizeof(float) * npix);

    const float sigma_sq = sigma_strength * sigma_strength;
    /* VisuShrink fallback for flat blocks: lambda_max = sigma * sqrt(2 ln N)
     * N = 64, so sqrt(2 ln 64) ~ 2.884 */
    const float lambda_max = sigma_strength * sqrtf(2.0f * logf((float)GALOSH_BLOCK_PIXELS));

    #pragma omp parallel
    {
        float *my_numer = (float *)dt_alloc_align(64, sizeof(float) * npix);
        float *my_denom = (float *)dt_alloc_align(64, sizeof(float) * npix);
        if(my_numer && my_denom)
        {
        memset(my_numer, 0, sizeof(float) * npix);
        memset(my_denom, 0, sizeof(float) * npix);

        #pragma omp for schedule(dynamic, 4)
        for(int ref_r = 0; ref_r <= rmax; ref_r += stride)
        {
            for(int ref_c = 0; ref_c <= cmax; ref_c += stride)
            {
                /* Extract 8x8 block */
                float block[GALOSH_BLOCK_PIXELS];
                for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                    memcpy(block + dy * GALOSH_BLOCK_SIZE,
                           input + (ref_r + dy) * width + ref_c,
                           GALOSH_BLOCK_SIZE * sizeof(float));

                /* Forward 2D WHT (unnormalized: coefficients scaled by 64) */
                wht2d_8x8(block, 0);

                /* Estimate sigma_Y^2 from AC coefficients (exclude DC = block[0])
                 * In unnormalized 2D WHT, each coeff variance = N * sigma^2
                 * (N = GALOSH_BLOCK_PIXELS = 64), so we divide sum_sq by
                 * (N-1) * N to get per-pixel variance estimate. */
                float sum_sq = 0.0f;
                for(int i = 1; i < GALOSH_BLOCK_PIXELS; i++)
                    sum_sq += block[i] * block[i];
                const float sigma_y_sq = sum_sq / ((float)(GALOSH_BLOCK_PIXELS - 1) * (float)GALOSH_BLOCK_PIXELS);

                /* Signal variance estimate: sigma_x^2 = max(sigma_Y^2 - sigma^2, 0) */
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
                    /* lambda = sigma^2 / sigma_x, scaled to unnormalized WHT domain (x sqrt(N))
                     * because unnormalized coeff std = sqrt(N) * normalized std */
                    lambda = (sigma_sq / sqrtf(sigma_x_sq)) * sqrtf((float)GALOSH_BLOCK_PIXELS);
                    /* Clamp to VisuShrink maximum (also in unnormalized scale) */
                    const float lambda_max_unorm = lambda_max * sqrtf((float)GALOSH_BLOCK_PIXELS);
                    if(lambda > lambda_max_unorm) lambda = lambda_max_unorm;
                }

                /* Hard threshold: zero coefficients below lambda, preserve DC */
                int n_nonzero = 1; /* DC always kept */
                for(int i = 1; i < GALOSH_BLOCK_PIXELS; i++)
                {
                    if(fabsf(block[i]) < lambda)
                        block[i] = 0.0f;
                    else
                        n_nonzero++;
                }

                /* Inverse 2D WHT (with normalization: divide by N=64) */
                wht2d_8x8(block, 1);

                /* Overlap-add with Kaiser window */
                const float weight = 1.0f / (float)n_nonzero;
                for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                    for(int dx = 0; dx < GALOSH_BLOCK_SIZE; dx++)
                    {
                        const int pos = (ref_r + dy) * width + (ref_c + dx);
                        const float kw = galosh_kaiser_2d[dy * GALOSH_BLOCK_SIZE + dx];
                        my_numer[pos] += weight * kw * block[dy * GALOSH_BLOCK_SIZE + dx];
                        my_denom[pos] += weight * kw;
                    }
            }
        }

        /* Merge thread-local buffers */
        #pragma omp critical
        {
            for(int i = 0; i < npix; i++)
            {
                numer[i] += my_numer[i];
                denom[i] += my_denom[i];
            }
        }
        } /* end if alloc ok */
        dt_free_align(my_numer);
        dt_free_align(my_denom);
    }

    /* Normalize */
    for(int i = 0; i < npix; i++)
        output[i] = (denom[i] > 1e-10f) ? numer[i] / denom[i] : input[i];

    dt_free_align(numer);
    dt_free_align(denom);
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
static void galosh_pass2_ex(const float *restrict noisy,
                           const float *restrict pilot,
                           float *restrict output,
                           const int width, const int height,
                           const float sigma_strength,
                           const float wiener_floor,
                           const int stride)
{
    const int rmax = height - GALOSH_BLOCK_SIZE;
    const int cmax = width  - GALOSH_BLOCK_SIZE;
    const int npix = width * height;

    float *numer = (float *)dt_alloc_align(64, sizeof(float) * npix);
    float *denom = (float *)dt_alloc_align(64, sizeof(float) * npix);
    if(!numer || !denom)
    {
        if(numer) dt_free_align(numer);
        if(denom) dt_free_align(denom);
        memcpy(output, noisy, sizeof(float) * npix);
        return;
    }
    memset(numer, 0, sizeof(float) * npix);
    memset(denom, 0, sizeof(float) * npix);

    /* sigma^2 in unnormalized WHT domain: sigma^2 * N
     * (each unnormalized 2D WHT coeff has variance = N * pixel_variance) */
    const float sigma_sq_unorm = sigma_strength * sigma_strength
                                * (float)GALOSH_BLOCK_PIXELS;

    #pragma omp parallel
    {
        float *my_numer = (float *)dt_alloc_align(64, sizeof(float) * npix);
        float *my_denom = (float *)dt_alloc_align(64, sizeof(float) * npix);
        if(my_numer && my_denom)
        {
        memset(my_numer, 0, sizeof(float) * npix);
        memset(my_denom, 0, sizeof(float) * npix);

        #pragma omp for schedule(dynamic, 4)
        for(int ref_r = 0; ref_r <= rmax; ref_r += stride)
        {
            for(int ref_c = 0; ref_c <= cmax; ref_c += stride)
            {
                /* Extract noisy and pilot blocks */
                float blk_noisy[GALOSH_BLOCK_PIXELS];
                float blk_pilot[GALOSH_BLOCK_PIXELS];
                for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                {
                    memcpy(blk_noisy + dy * GALOSH_BLOCK_SIZE,
                           noisy + (ref_r + dy) * width + ref_c,
                           GALOSH_BLOCK_SIZE * sizeof(float));
                    memcpy(blk_pilot + dy * GALOSH_BLOCK_SIZE,
                           pilot + (ref_r + dy) * width + ref_c,
                           GALOSH_BLOCK_SIZE * sizeof(float));
                }

                /* Forward WHT (unnormalized) */
                wht2d_8x8(blk_noisy, 0);
                wht2d_8x8(blk_pilot, 0);

                /* Wiener filtering */
                float wiener_energy = 0.0f;
                for(int i = 0; i < GALOSH_BLOCK_PIXELS; i++)
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

                /* Inverse WHT */
                wht2d_8x8(blk_noisy, 1);

                /* Overlap-add */
                const float weight = 1.0f / fmaxf(wiener_energy, 1e-6f);
                for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                    for(int dx = 0; dx < GALOSH_BLOCK_SIZE; dx++)
                    {
                        const int pos = (ref_r + dy) * width + (ref_c + dx);
                        const float kw = galosh_kaiser_2d[dy * GALOSH_BLOCK_SIZE + dx];
                        my_numer[pos] += weight * kw * blk_noisy[dy * GALOSH_BLOCK_SIZE + dx];
                        my_denom[pos] += weight * kw;
                    }
            }
        }

        #pragma omp critical
        {
            for(int i = 0; i < npix; i++)
            {
                numer[i] += my_numer[i];
                denom[i] += my_denom[i];
            }
        }
        } /* end if alloc ok */
        dt_free_align(my_numer);
        dt_free_align(my_denom);
    }

    for(int i = 0; i < npix; i++)
        output[i] = (denom[i] > 1e-10f) ? numer[i] / denom[i] : noisy[i];

    dt_free_align(numer);
    dt_free_align(denom);
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

#endif /* GALOSH_CORE_H */
