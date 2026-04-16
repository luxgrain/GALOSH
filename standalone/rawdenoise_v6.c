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

/* --- darktable compatibility stubs --- */
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
#define CFA_IDX_HORIZ  1   /* row=0, col=1 → 0*8+1 */
#define CFA_IDX_VERT   8   /* row=1, col=0 → 1*8+0 */
#define CFA_IDX_DIAG   9   /* row=1, col=1 → 1*8+1 */

static inline int is_cfa_protected(int i)
{
    return (i == 0 || i == CFA_IDX_HORIZ || i == CFA_IDX_VERT || i == CFA_IDX_DIAG);
}

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
static inline void wht_decompose_8x8(const float patch[GALOSH_BLOCK_PIXELS],
                                       float lc[4][GALOSH_HALF_PIXELS])
{
  const __m128 half = _mm_set1_ps(0.5f);
  for(int i = 0; i < GALOSH_HALF_BLOCK; i++)
  {
    /* Load 8 floats from even row: [a0,b0,a1,b1,a2,b2,a3,b3] */
    const float *row0 = patch + (2 * i) * GALOSH_BLOCK_SIZE;
    const float *row1 = patch + (2 * i + 1) * GALOSH_BLOCK_SIZE;
    const __m128 r0lo = _mm_loadu_ps(row0);      /* a0,b0,a1,b1 */
    const __m128 r0hi = _mm_loadu_ps(row0 + 4);  /* a2,b2,a3,b3 */
    const __m128 r1lo = _mm_loadu_ps(row1);      /* c0,d0,c1,d1 */
    const __m128 r1hi = _mm_loadu_ps(row1 + 4);  /* c2,d2,c3,d3 */
    /* Deinterleave: a = even cols, b = odd cols */
    const __m128 a = _mm_shuffle_ps(r0lo, r0hi, _MM_SHUFFLE(2, 0, 2, 0)); /* a0,a1,a2,a3 */
    const __m128 b = _mm_shuffle_ps(r0lo, r0hi, _MM_SHUFFLE(3, 1, 3, 1)); /* b0,b1,b2,b3 */
    const __m128 c = _mm_shuffle_ps(r1lo, r1hi, _MM_SHUFFLE(2, 0, 2, 0)); /* c0,c1,c2,c3 */
    const __m128 d = _mm_shuffle_ps(r1lo, r1hi, _MM_SHUFFLE(3, 1, 3, 1)); /* d0,d1,d2,d3 */
    const __m128 ab_sum = _mm_add_ps(a, b);
    const __m128 ab_dif = _mm_sub_ps(a, b);
    const __m128 cd_sum = _mm_add_ps(c, d);
    const __m128 cd_dif = _mm_sub_ps(c, d);
    _mm_storeu_ps(lc[0] + i * GALOSH_HALF_BLOCK, _mm_mul_ps(_mm_add_ps(ab_sum, cd_sum), half));
    _mm_storeu_ps(lc[1] + i * GALOSH_HALF_BLOCK, _mm_mul_ps(_mm_add_ps(ab_dif, cd_dif), half));
    _mm_storeu_ps(lc[2] + i * GALOSH_HALF_BLOCK, _mm_mul_ps(_mm_sub_ps(ab_sum, cd_sum), half));
    _mm_storeu_ps(lc[3] + i * GALOSH_HALF_BLOCK, _mm_mul_ps(_mm_sub_ps(ab_dif, cd_dif), half));
  }
}

/* SSE inverse WHT: 4x4 x 4ch -> 8x8 Bayer patch.
 * Processes all 4 columns simultaneously, interleaving back to 8-wide rows. */
static inline void wht_reconstruct_8x8(const float lc[4][GALOSH_HALF_PIXELS],
                                         float patch[GALOSH_BLOCK_PIXELS])
{
  const __m128 half = _mm_set1_ps(0.5f);
  for(int i = 0; i < GALOSH_HALF_BLOCK; i++)
  {
    const __m128 L  = _mm_loadu_ps(lc[0] + i * GALOSH_HALF_BLOCK);
    const __m128 C1 = _mm_loadu_ps(lc[1] + i * GALOSH_HALF_BLOCK);
    const __m128 C2 = _mm_loadu_ps(lc[2] + i * GALOSH_HALF_BLOCK);
    const __m128 C3 = _mm_loadu_ps(lc[3] + i * GALOSH_HALF_BLOCK);
    const __m128 LC1_sum = _mm_add_ps(L, C1);
    const __m128 LC1_dif = _mm_sub_ps(L, C1);
    const __m128 C2C3_sum = _mm_add_ps(C2, C3);
    const __m128 C2C3_dif = _mm_sub_ps(C2, C3);
    /* even-col (a), odd-col (b) for even row */
    const __m128 a = _mm_mul_ps(_mm_add_ps(LC1_sum, C2C3_sum), half);  /* L+C1+C2+C3 */
    const __m128 b = _mm_mul_ps(_mm_add_ps(LC1_dif, C2C3_dif), half);  /* L-C1+C2-C3 */
    /* even-col (c), odd-col (d) for odd row */
    const __m128 cc = _mm_mul_ps(_mm_sub_ps(LC1_sum, C2C3_sum), half); /* L+C1-C2-C3 */
    const __m128 dd = _mm_mul_ps(_mm_sub_ps(LC1_dif, C2C3_dif), half); /* L-C1-C2+C3 */
    /* Interleave even/odd columns back to 8-wide rows */
    float *out0 = patch + (2 * i) * GALOSH_BLOCK_SIZE;
    float *out1 = patch + (2 * i + 1) * GALOSH_BLOCK_SIZE;
    _mm_storeu_ps(out0,     _mm_unpacklo_ps(a, b));  /* a0,b0,a1,b1 */
    _mm_storeu_ps(out0 + 4, _mm_unpackhi_ps(a, b));  /* a2,b2,a3,b3 */
    _mm_storeu_ps(out1,     _mm_unpacklo_ps(cc, dd)); /* c0,d0,c1,d1 */
    _mm_storeu_ps(out1 + 4, _mm_unpackhi_ps(cc, dd)); /* c2,d2,c3,d3 */
  }
}


/* --- Noise estimation --- */

/* O(n) k-th element selection via quickselect.
 * Avoids O(n log n) qsort when only one order statistic is needed. */
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
#ifdef GALOSH_CFA_PROTECT
static void galosh_pass1_cfa(const float *restrict input,
                             float *restrict output,
                             const int width, const int height,
                             const float sigma_strength,
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
        memcpy(output, input, sizeof(float) * npix);
        return;
    }
    memset(numer, 0, sizeof(float) * npix);
    memset(denom, 0, sizeof(float) * npix);

    const float sigma_sq = sigma_strength * sigma_strength;
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
                float block[GALOSH_BLOCK_PIXELS];
                for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                    memcpy(block + dy * GALOSH_BLOCK_SIZE,
                           input + (ref_r + dy) * width + ref_c,
                           GALOSH_BLOCK_SIZE * sizeof(float));

                wht2d_8x8(block, 0);

                /* Estimate sigma_Y^2 from non-CFA AC coefficients */
                float sum_sq = 0.0f;
                int n_ac = 0;
                for(int i = 0; i < GALOSH_BLOCK_PIXELS; i++)
                {
                    if(!is_cfa_protected(i))
                    {
                        sum_sq += block[i] * block[i];
                        n_ac++;
                    }
                }
                const float sigma_y_sq = (n_ac > 0) ? sum_sq / ((float)n_ac * (float)GALOSH_BLOCK_PIXELS) : 0.0f;
                const float sigma_x_sq = fmaxf(sigma_y_sq - sigma_sq, 0.0f);

                float lambda;
                if(sigma_x_sq < 1e-10f)
                    lambda = 1e30f;
                else
                {
                    lambda = (sigma_sq / sqrtf(sigma_x_sq)) * sqrtf((float)GALOSH_BLOCK_PIXELS);
                    const float lambda_max_unorm = lambda_max * sqrtf((float)GALOSH_BLOCK_PIXELS);
                    if(lambda > lambda_max_unorm) lambda = lambda_max_unorm;
                }

                /* Hard threshold: preserve DC + CFA frequency bins */
                int n_nonzero = 4; /* DC + 3 CFA bins always kept */
                for(int i = 0; i < GALOSH_BLOCK_PIXELS; i++)
                {
                    if(is_cfa_protected(i)) continue; /* skip protected bins */
                    if(fabsf(block[i]) < lambda)
                        block[i] = 0.0f;
                    else
                        n_nonzero++;
                }

                wht2d_8x8(block, 1);

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
        output[i] = (denom[i] > 1e-10f) ? numer[i] / denom[i] : input[i];

    dt_free_align(numer);
    dt_free_align(denom);
}

/* CFA-protected Wiener: protect DC + CFA bins from shrinkage */
static void galosh_pass2_cfa(const float *restrict noisy,
                             const float *restrict pilot,
                             float *restrict output,
                             const int width, const int height,
                             const float sigma_strength,
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

                wht2d_8x8(blk_noisy, 0);
                wht2d_8x8(blk_pilot, 0);

                float wiener_energy = 0.0f;
                for(int i = 0; i < GALOSH_BLOCK_PIXELS; i++)
                {
                    float w;
                    if(is_cfa_protected(i))
                    {
                        w = 1.0f; /* DC + CFA bins protected */
                    }
                    else
                    {
                        const float s2 = blk_pilot[i] * blk_pilot[i];
                        w = s2 / (s2 + sigma_sq_unorm);
                        if(w < GALOSH_WIENER_FLOOR) w = GALOSH_WIENER_FLOOR;
                    }
                    blk_noisy[i] *= w;
                    wiener_energy += w * w;
                }

                wht2d_8x8(blk_noisy, 1);

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
#endif /* GALOSH_CFA_PROTECT */


/* ================================================================
 * Compute L_fullres from half-res L/C planes.
 *
 * Half-res L/C produces one L value per 2x2 Bayer block.
 * Under strong shrinkage this creates a visible 2x2 plateau artifact.
 * To eliminate it we compute L at every raw pixel position using
 * a sliding 2x2 WHT-DC.
 *
 * Derivation -- inv-WHT of sliding 2x2 blocks yields:
 *   L_fullres(2hy,   2hx)   = L(hy,hx)
 *   L_fullres(2hy,   2hx+1) = [(L-C2)@(hy,hx) + (L+C2)@(hy,hx+1)] / 2
 *   L_fullres(2hy+1, 2hx)   = [(L-C1)@(hy,hx) + (L+C1)@(hy+1,hx)] / 2
 *   L_fullres(2hy+1, 2hx+1) = sum [(L +/- C1 +/- C2 +/- C3) at 4 blocks] / 4
 *
 * Output: L_out of size (2*halfwidth) x (2*halfheight) = full-res.
 * ================================================================ */
static void compute_L_fullres(const float *restrict L,
                               const float *restrict C1,
                               const float *restrict C2,
                               const float *restrict C3,
                               const int halfwidth, const int halfheight,
                               float *restrict L_out)
{
  const int fw = 2 * halfwidth;

  DT_OMP_FOR()
  for(int hy = 0; hy < halfheight; hy++)
  {
    for(int hx = 0; hx < halfwidth; hx++)
    {
      const size_t hi = (size_t)hy * halfwidth + hx;
      /* Neighbor indices (clamped at boundaries) */
      const int hx1 = MIN(hx + 1, halfwidth - 1);
      const int hy1 = MIN(hy + 1, halfheight - 1);
      const size_t hi_r  = (size_t)hy  * halfwidth + hx1;
      const size_t hi_d  = (size_t)hy1 * halfwidth + hx;
      const size_t hi_dr = (size_t)hy1 * halfwidth + hx1;

      const int fr = 2 * hy, fc = 2 * hx;

      /* (2hy, 2hx): block-aligned = L itself */
      L_out[(size_t)fr * fw + fc] = L[hi];

      /* (2hy, 2hx+1): horizontal sliding */
      L_out[(size_t)fr * fw + fc + 1]
        = ((L[hi] - C2[hi]) + (L[hi_r] + C2[hi_r])) * 0.5f;

      /* (2hy+1, 2hx): vertical sliding */
      L_out[(size_t)(fr + 1) * fw + fc]
        = ((L[hi] - C1[hi]) + (L[hi_d] + C1[hi_d])) * 0.5f;

      /* (2hy+1, 2hx+1): diagonal sliding */
      {
        const float Ls = L[hi] + L[hi_r] + L[hi_d] + L[hi_dr];
        const float C1s = -C1[hi] - C1[hi_r] + C1[hi_d] + C1[hi_dr];
        const float C2s = -C2[hi] + C2[hi_r] - C2[hi_d] + C2[hi_dr];
        const float C3s =  C3[hi] - C3[hi_r] - C3[hi_d] + C3[hi_dr];
        L_out[(size_t)(fr + 1) * fw + fc + 1]
          = (Ls + C1s + C2s + C3s) * 0.25f;
      }
    }
  }
}


/* ================================================================
 *  GALOSH -- Full pipeline (RAW L/C decomposed)
 *
 *  Pre-demosaic denoiser operating on RGGB before WB and demosaic.
 *  Fully blind: no white-balance coefficients, no per-channel QE prior.
 *
 *  Pipeline:
 *   1. Half-res RGGB extraction -> piecewise C1 GAT -> RMS unified sigma-normalize
 *   2. Self-consistent dark anchor (per-ch DC subtract) -> WHT -> L/C1/C2/C3
 *   3. Pass 1: BayesShrink hard-threshold pilot on each half-res L/C plane
 *   4. Pass 2: Wiener shrinkage on each half-res L/C plane (independent sigma_L / sigma_C)
 *   5. Full-res L: sliding WHT-DC -> GALOSH Pass 2 on L plane
 *   6. Inverse WHT with full-res L + half-res C (anchor restore)
 *   7. sigma-denormalize -> exact inverse GAT -> write back
 *
 *  Key difference from BM3D: GALOSH operates on ONE plane at a time
 *  (L, C1, C2, C3 independently). No block matching, no non-local processing.
 *  GAT normalization ensures noise is i.i.d. Gaussian (sigma=1), so local
 *  adaptive shrinkage in the WHT domain is sufficient.
 *
 *  Theoretical highlights:
 *  (a) Piecewise C1 VST (extends Foi GAT to all reals)
 *  (b) Unified sigma normalization (RMS, makes Var[L]=Var[Ck]=1 exactly)
 *  (c) Self-consistent dark anchor (per-channel DC protection)
 *  See detailed derivations in comments below.
 *
 *  Ref: Danielyan et al. "Cross-color BM3D" (LNLA 2009) -- L/C decomposition
 *       Foi et al. (Sig.Proc. 2008) -- Poisson-Gaussian noise model
 *       Makitalo & Foi (TIP 2013) -- exact unbiased inverse GAT
 *       Chang, Yu & Vetterli (IEEE TIP 2000) -- BayesShrink
 * ================================================================ */
static void gat_galosh_denoise_rawlc(const float *const restrict in, float *const restrict out,
                                    const dt_iop_roi_t *const roi,
                                    const float luma_strength, const float chroma_strength,
                                    const uint32_t filters)
{
  const int width = roi->width, height = roi->height;
  const size_t npixels = (size_t)width * height;
  memcpy(out, in, sizeof(float) * npixels);

  if(luma_strength <= 0.0f) return;

  const int halfwidth = (width + 1) / 2;
  const int halfheight = (height + 1) / 2;
  if(halfwidth < GALOSH_BLOCK_SIZE * 2 || halfheight < GALOSH_BLOCK_SIZE * 2) return;

  const size_t chsize = (size_t)halfwidth * halfheight;

  const galosh_noise_params_t np = galosh_estimate_noise(in, width, height);
  gat_build_inverse_table(np.alpha, np.sigma_sq);

  /* Pre-declare all buffers */
  float *ch_gat[4] = { NULL, NULL, NULL, NULL };
  float *luma = NULL, *chroma1 = NULL, *chroma2 = NULL, *chroma3 = NULL;
  float *c1_pilot = NULL, *c2_pilot = NULL, *c3_pilot = NULL;
  float *c1_out = NULL, *c2_out = NULL, *c3_out = NULL;

  /* ================================================================
   * Phase 1: Extract, GAT, normalize half-res RGGB
   * ================================================================ */
  float sigma_gat_ch[4];
  for(int c = 0; c < 4; c++)
  {
    const int row_offset = c & 1, col_offset = (c >> 1) & 1;
    ch_gat[c] = dt_alloc_align_float(chsize);
    if(!ch_gat[c]) goto cleanup_rawlc;

    DT_OMP_FOR()
    for(int row = row_offset; row < height; row += 2)
      for(int col = col_offset; col < width; col += 2)
        ch_gat[c][((row - row_offset) / 2) * halfwidth + (col - col_offset) / 2]
          = in[(size_t)row * width + col];

    DT_OMP_FOR()
    for(size_t i = 0; i < chsize; i++)
      ch_gat[c][i] = gat_forward(ch_gat[c][i], np.alpha, np.sigma_sq);

    sigma_gat_ch[c] = estimate_gat_sigma_halfres(ch_gat[c], halfwidth, halfheight);
  }

  /* RMS unified sigma normalization.
   *
   * unified_sigma := sqrt( mean(sigma_c^2) )
   *
   * This choice makes Var[L] = Var[Ck] = 1 exactly after WHT,
   * regardless of per-channel GAT-domain variance non-uniformity
   * caused by single-shot noise estimation bias toward G.
   * Per-channel normalization would fix per-ch Var=1 but break
   * WHT signal proportionality (false chroma on uniform input). */
  {
    const float mean_var = 0.25f * (sigma_gat_ch[0] * sigma_gat_ch[0]
                                  + sigma_gat_ch[1] * sigma_gat_ch[1]
                                  + sigma_gat_ch[2] * sigma_gat_ch[2]
                                  + sigma_gat_ch[3] * sigma_gat_ch[3]);
    const float unified_sigma = sqrtf(fmaxf(mean_var, 1e-12f));

    const float post_mean_var = mean_var / (unified_sigma * unified_sigma);

    fprintf(stderr, "[rawdenoise] GALOSH: alpha=%.8f sigma_sq=%.10f | "
                     "unified_sigma=%.4f [RMS] (per-ch: %.4f %.4f %.4f %.4f) | "
                     "post-norm mean Var=%.4f (target=1.0) | "
                     "size=%dx%d (half=%dx%d) | "
                     "sigma_L=%.3f sigma_C=%.3f (independent, GAT-norm space)\n",
            np.alpha, np.sigma_sq, unified_sigma,
            sigma_gat_ch[0], sigma_gat_ch[1], sigma_gat_ch[2], sigma_gat_ch[3],
            post_mean_var,
            width, height, halfwidth, halfheight,
            luma_strength, chroma_strength);
    for(int c = 0; c < 4; c++) sigma_gat_ch[c] = unified_sigma;
    const float inv_sg = 1.0f / unified_sigma;
    for(int c = 0; c < 4; c++)
    {
      DT_OMP_FOR()
      for(size_t i = 0; i < chsize; i++)
        ch_gat[c][i] *= inv_sg;
    }
  }

  /* ================================================================
   * Phase 2: Self-consistent dark anchor + WHT -> L/C1/C2/C3
   *
   * Per-channel QE differences are REAL signal to be PRESERVED in bright
   * regions. In dark regions, however, QE x signal ~ 0, while the GAT
   * linear branch and read-noise statistics leave residual per-channel
   * DC offsets that, after WHT, manifest as false dark-area chroma.
   *
   * Fix: estimate a per-channel DC anchor over a noise-dominated
   * cohort and subtract it before WHT, then restore it after shrinkage.
   * The cohort is selected by smooth weights w(L_i) = 1 / (1 + (L_i/s)^4).
   * Scale s is self-consistently estimated (2 iterations).
   * ================================================================ */
  float ch_dark_ref[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  {
    const double s_init = (double)np.sigma_sq / fmax((double)np.alpha, 1e-12);
    double s_scale = s_init;
    const double s_min = 0.05 * s_init;
    const double s_max = 50.0 * s_init;
    const int n_iter = 2;

    for(int iter = 0; iter <= n_iter; iter++)
    {
      const double inv_s = 1.0 / fmax(s_scale, 1e-20);
      double sum_w = 0.0;
      double sum_wch[4] = {0.0, 0.0, 0.0, 0.0};

      for(int hy = 0; hy < halfheight; hy++)
        for(int hx = 0; hx < halfwidth; hx++)
        {
          const size_t hpos = (size_t)hy * halfwidth + hx;

          /* Achromatic filter: exclude pixels with inter-channel spread
           * exceeding noise expectation. Dark skin (melanin: R < G) and
           * other spectrally non-neutral dark objects are rejected.
           * In sigma-normalized GAT domain, noise-only range ≈ 2.06 ± 0.73. */
          const float g0 = ch_gat[0][hpos], g1 = ch_gat[1][hpos];
          const float g2 = ch_gat[2][hpos], g3 = ch_gat[3][hpos];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const int fy = hy * 2, fx = hx * 2;
          const float iv0 = in[(size_t)fy * width + fx];
          const float iv1 = in[(size_t)(fy + 1) * width + fx];
          const float iv2 = in[(size_t)fy * width + fx + 1];
          const float iv3 = in[(size_t)(fy + 1) * width + fx + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          sum_w += w;
          sum_wch[0] += w * g0;
          sum_wch[1] += w * g1;
          sum_wch[2] += w * g2;
          sum_wch[3] += w * g3;
        }

      const double inv_sw = 1.0 / fmax(sum_w, 1e-20);
      for(int c = 0; c < 4; c++)
        ch_dark_ref[c] = (float)(sum_wch[c] * inv_sw);

      if(iter == n_iter) break;

      /* Weighted per-ch residual std in GAT-norm (target = 1.0) */
      double sum_wresid2 = 0.0;
      double sum_w2 = 0.0;
      for(int hy = 0; hy < halfheight; hy++)
        for(int hx = 0; hx < halfwidth; hx++)
        {
          const size_t hpos = (size_t)hy * halfwidth + hx;

          /* Same achromatic filter as above */
          const float g0 = ch_gat[0][hpos], g1 = ch_gat[1][hpos];
          const float g2 = ch_gat[2][hpos], g3 = ch_gat[3][hpos];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const int fy = hy * 2, fx = hx * 2;
          const float iv0 = in[(size_t)fy * width + fx];
          const float iv1 = in[(size_t)(fy + 1) * width + fx];
          const float iv2 = in[(size_t)fy * width + fx + 1];
          const float iv3 = in[(size_t)(fy + 1) * width + fx + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          sum_w2 += w;
          double resid2 = 0.0;
          for(int c = 0; c < 4; c++)
          {
            const double d = (double)ch_gat[c][hpos] - ch_dark_ref[c];
            resid2 += d * d;
          }
          sum_wresid2 += w * resid2 * 0.25;
        }
      const double inv_sw2 = 1.0 / fmax(sum_w2, 1e-20);
      const double measured_std =
          sqrt(fmax(sum_wresid2 * inv_sw2, 1e-20));
      const double ratio = 1.0 / measured_std;
      s_scale *= sqrt(ratio);
      if(s_scale < s_min) s_scale = s_min;
      if(s_scale > s_max) s_scale = s_max;
    }

    fprintf(stderr, "[rawdenoise] dark anchor self-consistent: s_init=%.6e s_final=%.6e | "
                    "ch_dark_ref: B=%.4f Gb=%.4f Gr=%.4f R=%.4f\n",
            s_init, s_scale,
            ch_dark_ref[0], ch_dark_ref[1], ch_dark_ref[2], ch_dark_ref[3]);

    for(int c = 0; c < 4; c++)
    {
      const float ref = ch_dark_ref[c];
      DT_OMP_FOR()
      for(size_t i = 0; i < chsize; i++) ch_gat[c][i] -= ref;
    }
  }

  /* WHT decomposition: 4 half-res RGGB channels -> L/C1/C2/C3 */
  luma    = dt_alloc_align_float(chsize);
  chroma1 = dt_alloc_align_float(chsize);
  chroma2 = dt_alloc_align_float(chsize);
  chroma3 = dt_alloc_align_float(chsize);
  if(!luma || !chroma1 || !chroma2 || !chroma3) goto cleanup_rawlc;

  DT_OMP_FOR()
  for(size_t i = 0; i < chsize; i++)
  {
    const float a = ch_gat[0][i], b = ch_gat[1][i];
    const float cc = ch_gat[2][i], d = ch_gat[3][i];
    luma[i]    = (a + b + cc + d) * 0.5f;
    chroma1[i] = (a - b + cc - d) * 0.5f;
    chroma2[i] = (a + b - cc - d) * 0.5f;
    chroma3[i] = (a - b - cc + d) * 0.5f;
  }

  /* ================================================================
   * Phase 3: Independent chroma 2-pass denoising
   *
   * Each chroma plane (C1, C2, C3) gets its own BayesShrink pilot
   * (Pass 1) and Wiener shrinkage (Pass 2). This is the standard
   * 2-pass design applied independently to each L/C plane.
   *
   * Key difference from GALOSH_LG: chroma pilots come from chroma
   * itself, not from luma. This preserves chroma-specific noise
   * structure and avoids the edge-blocking artifacts that occur
   * when a cross-channel pilot mismatches chroma's own structure.
   *
   * Color shift mitigation: Pass 2 uses galosh_pass2_chroma() with
   * a higher Wiener floor (GALOSH_CHROMA_WIENER_FLOOR = 0.3) to
   * prevent over-shrinkage of chroma AC in dark areas. Standard
   * floor 0.125 kills spatial color variation, leaving Phase 4's
   * G-biased luma to dominate the output → green shift.
   *
   * Phase 3: 独立 chroma 2パスデノイズ
   * 各 chroma plane が自身の BayesShrink pilot + Wiener を持つ。
   * LG のように luma pilot を流用せず、chroma 固有のノイズ構造を保持。
   * 色シフト対策: chroma Wiener floor を 0.3 に引き上げ、暗部の
   * chroma AC 過剰 shrinkage を防止。
   * ================================================================ */
  {
    /* Chroma Pass 1 (BayesShrink pilot) + Pass 2 (Wiener with higher floor) */
    c1_pilot = dt_alloc_align_float(chsize);
    c2_pilot = dt_alloc_align_float(chsize);
    c3_pilot = dt_alloc_align_float(chsize);
    c1_out   = dt_alloc_align_float(chsize);
    c2_out   = dt_alloc_align_float(chsize);
    c3_out   = dt_alloc_align_float(chsize);
    if(!c1_pilot || !c2_pilot || !c3_pilot || !c1_out || !c2_out || !c3_out)
      goto cleanup_rawlc;

    /* Stride selection for half-res L/C processing:
     *   GALOSH_F_HALF or GALOSH_F: stride=2 (75% overlap)
     *   GALOSH_F_FULL or baseline:  stride=4 (50% overlap) */
#if defined(GALOSH_F_HALF) || defined(GALOSH_F)
    const int chroma_stride = 2;
#else
    const int chroma_stride = GALOSH_STRIDE;
#endif

    galosh_pass1(chroma1, c1_pilot, halfwidth, halfheight, chroma_strength, chroma_stride);
    galosh_pass1(chroma2, c2_pilot, halfwidth, halfheight, chroma_strength, chroma_stride);
    galosh_pass1(chroma3, c3_pilot, halfwidth, halfheight, chroma_strength, chroma_stride);

    fprintf(stderr, "[rawdenoise] GALOSH: chroma Pass1 pilots done (sigma_C=%.3f, stride=%d)\n",
            chroma_strength, chroma_stride);

    galosh_pass2(chroma1, c1_pilot, c1_out, halfwidth, halfheight, chroma_strength, chroma_stride);
    galosh_pass2(chroma2, c2_pilot, c2_out, halfwidth, halfheight, chroma_strength, chroma_stride);
    galosh_pass2(chroma3, c3_pilot, c3_out, halfwidth, halfheight, chroma_strength, chroma_stride);

    fprintf(stderr, "[rawdenoise] GALOSH: chroma Pass2 Wiener done (sigma_C=%.3f)\n",
            chroma_strength);
  }

  dt_free_align(c1_pilot); c1_pilot = NULL;
  dt_free_align(c2_pilot); c2_pilot = NULL;
  dt_free_align(c3_pilot); c3_pilot = NULL;

#if defined(GALOSH_F) || defined(GALOSH_F_HALF) || defined(GALOSH_F_FULL)
  /* ================================================================
   * Phase 4 — GALOSH_F: Half-Res L/C + Full-Res L Refinement
   *
   * Color-shift-free architecture. Instead of applying GALOSH on the
   * full-res raw Bayer (which mixes CFA channels and causes green
   * shift), we process luma and chroma separately:
   *   (a) Denoise half-res luma with Pass1 + Pass2 (stride=2, 75% overlap)
   *   (b) Compute L_fullres from noisy and pilot half-res L/C planes
   *       via sliding 2x2 WHT-DC
   *   (c) Apply GALOSH Pass2 on L_fullres for full-res luma refinement
   *       (stride=2, 75% overlap)
   *   (d) Reconstruct full-res RGGB via inverse 2x2 WHT:
   *       full-res L_den + half-res C_den + dark anchor
   *
   * Key advantages:
   *   - Chroma channels are NEVER processed at full-res
   *     → no inter-channel homogenization → no color shift
   *   - stride=2 overlap-add gives 4x block contributions per pixel
   *     → smoother estimates, better high-frequency luma resolution
   *   - Fully GPU-parallelizable (OpenCL): all passes are independent
   *     block-local operations with overlap-add aggregation
   *
   * GALOSH_F: 半解像度 L/C + 全解像度 L リファインメント。
   * raw Bayer 全体に GALOSH をかけず、L のみ全解像度化して Pass2 処理。
   * C は半解像度のまま Phase 3 の結果を使用。stride=2 (75%オーバーラップ)
   * により overlap-add 精度向上、ハイコントラスト・ルマ解像感を改善。
   * ================================================================ */
  {
    /* Step (a): Half-res luma Pass1 (BayesShrink pilot) + Pass2 (Wiener) */
    float *l_pilot = dt_alloc_align_float(chsize);
    float *l_out   = dt_alloc_align_float(chsize);
    if(!l_pilot || !l_out)
    {
      dt_free_align(l_pilot);
      dt_free_align(l_out);
      goto cleanup_rawlc;
    }

    /* Half-res luma stride: same logic as chroma stride.
     *   GALOSH_F_HALF or GALOSH_F: stride=2 (75% overlap)
     *   GALOSH_F_FULL:             stride=4 (50% overlap) */
#if defined(GALOSH_F_HALF) || defined(GALOSH_F)
    const int halfres_stride = 2;
#else
    const int halfres_stride = GALOSH_STRIDE;
#endif

    fprintf(stderr, "[rawdenoise] GALOSH_F: half-res luma Pass1+2 "
                     "(%dx%d, sigma_L=%.3f, stride=%d)\n",
                     halfwidth, halfheight, luma_strength, halfres_stride);

    galosh_pass1(luma, l_pilot, halfwidth, halfheight, luma_strength, halfres_stride);
    galosh_pass2(luma, l_pilot, l_out, halfwidth, halfheight, luma_strength, halfres_stride);

    fprintf(stderr, "[rawdenoise] GALOSH_F: half-res luma done\n");

    /* Step (b): Compute L_fullres from noisy and pilot half-res L/C.
     *
     * We build TWO full-res L planes:
     *   L_fullres_noisy: from noisy half-res (luma, chroma1/2/3)
     *   L_fullres_pilot: from denoised half-res (l_out, c1_out/2/3)
     *
     * The pilot provides the Wiener reference for Pass2 at full-res. */
    const size_t fullsize = (size_t)width * height;
    float *L_fr_noisy = dt_alloc_align_float(fullsize);
    float *L_fr_pilot = dt_alloc_align_float(fullsize);
    float *L_fr_den   = dt_alloc_align_float(fullsize);
    if(!L_fr_noisy || !L_fr_pilot || !L_fr_den)
    {
      dt_free_align(L_fr_noisy);
      dt_free_align(L_fr_pilot);
      dt_free_align(L_fr_den);
      dt_free_align(l_pilot);
      dt_free_align(l_out);
      goto cleanup_rawlc;
    }

    /* Full-res L stride selection:
     *   GALOSH_F_FULL or GALOSH_F: stride=2 (75% overlap)
     *   GALOSH_F_HALF or baseline:  stride=4 (50% overlap) */
#if defined(GALOSH_F_FULL) || defined(GALOSH_F)
    const int fullres_stride = 2;
#else
    const int fullres_stride = GALOSH_STRIDE;
#endif

    compute_L_fullres(luma, chroma1, chroma2, chroma3,
                      halfwidth, halfheight, L_fr_noisy);
    compute_L_fullres(l_out, c1_out, c2_out, c3_out,
                      halfwidth, halfheight, L_fr_pilot);

    fprintf(stderr, "[rawdenoise] GALOSH_F: L_fullres computed (%dx%d), "
                     "full-res stride=%d\n", width, height, fullres_stride);

    /* Step (c): GALOSH Pass2 on full-res L.
     * Only Pass2 (Wiener) — the pilot from step (b) is already denoised.
     * No Pass1 needed because the pilot is built from half-res denoised L/C. */
    galosh_pass2(L_fr_noisy, L_fr_pilot, L_fr_den, width, height, luma_strength, fullres_stride);

    dt_free_align(L_fr_noisy); L_fr_noisy = NULL;
    dt_free_align(L_fr_pilot); L_fr_pilot = NULL;
    dt_free_align(l_pilot);    l_pilot = NULL;

    fprintf(stderr, "[rawdenoise] GALOSH_F: full-res L Pass2 done\n");

    /* Step (d): Inverse WHT reconstruction.
     *
     * For each 2x2 Bayer block, reconstruct full-res RGGB from:
     *   - L_fr_den: full-res denoised luma (per-pixel)
     *   - c1_out, c2_out, c3_out: half-res denoised chroma (per-block)
     *   - ch_dark_ref: dark anchor offsets (per-channel)
     *
     * Inverse 2x2 WHT:
     *   R  = (L + C1 + C2 + C3) / 2 = L/2 + (C1+C2+C3)/2
     *   Gb = (L - C1 + C2 - C3) / 2
     *   Gr = (L + C1 - C2 - C3) / 2
     *   B  = (L - C1 - C2 + C3) / 2
     *
     * Note: ch_gat values had dark_ref subtracted in Phase 2, so
     * our L/C planes are dark-ref-free. We add dark_ref back during
     * reconstruction so the sigma-denorm + inv-GAT step gets absolute
     * GAT-normalized values.
     *
     * 逆 2x2 WHT: 全解像度 L_den + 半解像度 C_den → RGGB。
     * dark_ref を加算して absolute GAT-norm 空間に戻す。 */
    {
      const float sg = sigma_gat_ch[0]; /* = unified_sigma */
      DT_OMP_FOR()
      for(int hy = 0; hy < halfheight; hy++)
        for(int hx = 0; hx < halfwidth; hx++)
        {
          const size_t hi = (size_t)hy * halfwidth + hx;
          const int fr = 2 * hy, fc = 2 * hx;

          /* Half-res chroma (dark-ref-free, sigma-normalized) */
          const float c1 = c1_out[hi];
          const float c2 = c2_out[hi];
          const float c3 = c3_out[hi];

          /* Use block-aligned L only (2hy, 2hx) to avoid pixel shift.
           *
           * L_fullres at non-aligned positions is interpolated from adjacent
           * blocks, but C values are from this block only → spatial mismatch.
           * Using L at block-aligned position ensures L and C are consistent.
           * The full-res GALOSH Pass2 still improves L estimation quality
           * via cross-block spatial context in 8x8 WHT shrinkage.
           *
           * ブロック整合位置 (2hy, 2hx) の L のみ使用。非整合位置の
           * L_fullres は隣接ブロックから補間されており、自ブロックの C と
           * 空間的に不整合 → ピクセルシフトの原因。 */
          const float L_block = L_fr_den[(size_t)fr * width + fc];

          /* Inverse 2x2 WHT + dark_ref restore → absolute GAT-norm space.
           * Then sigma-denormalize (*sg) → inverse GAT → linear raw. */
          const float val_R  = (L_block + c1 + c2 + c3) * 0.5f + ch_dark_ref[0];
          const float val_Gb = (L_block - c1 + c2 - c3) * 0.5f + ch_dark_ref[1];
          const float val_Gr = (L_block + c1 - c2 - c3) * 0.5f + ch_dark_ref[2];
          const float val_B  = (L_block - c1 - c2 + c3) * 0.5f + ch_dark_ref[3];

          out[(size_t)fr       * width + fc]     = gat_inverse_exact(val_R  * sg);
          out[(size_t)(fr + 1) * width + fc]     = gat_inverse_exact(val_Gb * sg);
          out[(size_t)fr       * width + fc + 1] = gat_inverse_exact(val_Gr * sg);
          out[(size_t)(fr + 1) * width + fc + 1] = gat_inverse_exact(val_B  * sg);
        }
    }

    dt_free_align(L_fr_den);  L_fr_den = NULL;
    dt_free_align(l_out);     l_out = NULL;

    fprintf(stderr, "[rawdenoise] GALOSH_F: inverse WHT + inv-GAT done\n");
  }

#else /* legacy full-res raw GALOSH (stride=4) */

  /* ================================================================
   * Phase 4: Full-res raw Bayer GALOSH (Pass 1 + Pass 2)
   *
   * Apply GALOSH directly on the GAT-normalized raw Bayer at full
   * resolution. This denoises luma (and partially chroma) without
   * the half-resolution bottleneck, preserving sub-pixel features.
   *
   * The GAT-normalized raw is reconstructed from ch_gat[0..3] into
   * a single full-res buffer (interleaved RGGB).
   * ================================================================ */
  {
    const size_t fullsize = (size_t)width * height;
    float *raw_gat_full    = dt_alloc_align_float(fullsize);
    float *raw_gat_pilot   = dt_alloc_align_float(fullsize);
    float *raw_gat_den     = dt_alloc_align_float(fullsize);
    if(!raw_gat_full || !raw_gat_pilot || !raw_gat_den)
    {
      dt_free_align(raw_gat_full);
      dt_free_align(raw_gat_pilot);
      dt_free_align(raw_gat_den);
      goto cleanup_rawlc;
    }

    /* Reconstruct full-res GAT-normalized raw from half-res channels.
     * ch_gat[c] has dark_ref already subtracted; add it back for raw. */
    DT_OMP_FOR()
    for(int row = 0; row < height; row++)
      for(int col = 0; col < width; col++)
      {
        /* Determine which Bayer channel: c = (row&1) | ((col&1)<<1)
         * c=0: R(even row, even col), c=1: Gb(odd row, even col)
         * c=2: Gr(even row, odd col), c=3: B(odd row, odd col) */
        const int c = (row & 1) | ((col & 1) << 1);
        const size_t hi = (size_t)(row / 2) * halfwidth + (col / 2);
        raw_gat_full[(size_t)row * width + col] = ch_gat[c][hi] + ch_dark_ref[c];
      }

    fprintf(stderr, "[rawdenoise] GALOSH: starting full-res raw Pass 1+2 "
                     "(%dx%d, sigma_L=%.3f)\n", width, height, luma_strength);

#ifdef GALOSH_CFA_PROTECT
    /* CFA frequency protection (legacy, superseded by GALOSH_F).
     * Protects WHT bins {0,1,8,9} from shrinkage. Does NOT fully fix
     * color shift — GALOSH_F is the recommended path. */
    galosh_pass1_cfa(raw_gat_full, raw_gat_pilot, width, height, luma_strength, GALOSH_STRIDE);
    galosh_pass2_cfa(raw_gat_full, raw_gat_pilot, raw_gat_den, width, height, luma_strength, GALOSH_STRIDE);
#else
    galosh_pass1(raw_gat_full, raw_gat_pilot, width, height, luma_strength, GALOSH_STRIDE);
    galosh_pass2(raw_gat_full, raw_gat_pilot, raw_gat_den, width, height, luma_strength, GALOSH_STRIDE);
#endif

    dt_free_align(raw_gat_pilot); raw_gat_pilot = NULL;

    fprintf(stderr, "[rawdenoise] GALOSH: full-res raw Pass 1+2 done\n");

    /* ================================================================
     * Phase 5: Chroma replacement.
     *
     * raw_gat_den has good luma but mediocre chroma denoising (only
     * luma_strength was used). Replace its chroma component with the
     * properly denoised half-res C1/C2/C3 from Phase 3.
     *
     * For each 2x2 Bayer block at half-res (hy, hx):
     *   1. Extract the 4 full-res denoised values → compute C1', C2', C3'
     *   2. Compute delta: dCk = Ck_den - Ck'
     *   3. Apply chroma correction with WHT sign pattern:
     *        R  (0,0): +dC1, +dC2, +dC3
     *        Gb (1,0): -dC1, +dC2, -dC3
     *        Gr (0,1): +dC1, -dC2, -dC3
     *        B  (1,1): -dC1, -dC2, +dC3
     *
     * Sign convention follows the WHT decomposition in Phase 2:
     *   ch_gat order: [0]=R, [1]=Gb, [2]=Gr, [3]=B
     *   L  = (R + Gb + Gr + B) / 2
     *   C1 = (R - Gb + Gr - B) / 2
     *   C2 = (R + Gb - Gr - B) / 2
     *   C3 = (R - Gb - Gr + B) / 2
     * ================================================================ */
    DT_OMP_FOR()
    for(int hy = 0; hy < halfheight; hy++)
      for(int hx = 0; hx < halfwidth; hx++)
      {
        const size_t hi = (size_t)hy * halfwidth + hx;
        const int fr = 2 * hy, fc = 2 * hx;

        /* Full-res denoised Bayer values at the 4 positions */
        const float den_R  = raw_gat_den[(size_t)fr       * width + fc];      /* ch0: R */
        const float den_Gb = raw_gat_den[(size_t)(fr + 1) * width + fc];      /* ch1: Gb */
        const float den_Gr = raw_gat_den[(size_t)fr       * width + fc + 1];  /* ch2: Gr */
        const float den_B  = raw_gat_den[(size_t)(fr + 1) * width + fc + 1];  /* ch3: B */

        /* Chroma of full-res denoised (same WHT as Phase 2) */
        const float c1_fr = (den_R - den_Gb + den_Gr - den_B) * 0.5f;
        const float c2_fr = (den_R + den_Gb - den_Gr - den_B) * 0.5f;
        const float c3_fr = (den_R - den_Gb - den_Gr + den_B) * 0.5f;

        /* Chroma correction: replace full-res chroma with half-res denoised */
        const float dC1 = c1_out[hi] - c1_fr;
        const float dC2 = c2_out[hi] - c2_fr;
        const float dC3 = c3_out[hi] - c3_fr;

        /* Apply chroma correction to each full-res pixel.
         * raw_gat_den values include dark_ref (absolute GAT-norm space).
         * Inverse 2x2 WHT sign pattern applied to chroma delta. */
        const float corr_R  = ( dC1 + dC2 + dC3) * 0.5f;
        const float corr_Gb = (-dC1 + dC2 - dC3) * 0.5f;
        const float corr_Gr = ( dC1 - dC2 - dC3) * 0.5f;
        const float corr_B  = (-dC1 - dC2 + dC3) * 0.5f;

        raw_gat_den[(size_t)fr       * width + fc]     = den_R  + corr_R;
        raw_gat_den[(size_t)(fr + 1) * width + fc]     = den_Gb + corr_Gb;
        raw_gat_den[(size_t)fr       * width + fc + 1]  = den_Gr + corr_Gr;
        raw_gat_den[(size_t)(fr + 1) * width + fc + 1]  = den_B  + corr_B;
      }

    fprintf(stderr, "[rawdenoise] GALOSH: chroma replacement done\n");

    /* ================================================================
     * Phase 6: sigma-denormalize -> inverse GAT -> write back (full-res)
     *
     * raw_gat_den is in absolute GAT-normalized space:
     *   value = GAT(raw) / unified_sigma
     * To invert: value * unified_sigma → gat_inverse_exact()
     * ================================================================ */
    {
      const float sg = sigma_gat_ch[0]; /* = unified_sigma for all channels */
      DT_OMP_FOR()
      for(int row = 0; row < height; row++)
        for(int col = 0; col < width; col++)
        {
          const size_t pos = (size_t)row * width + col;
          out[pos] = gat_inverse_exact(raw_gat_den[pos] * sg);
        }
    }

    dt_free_align(raw_gat_den);  raw_gat_den = NULL;
    dt_free_align(raw_gat_full); raw_gat_full = NULL;
  }
#endif /* GALOSH_F / GALOSH_F_HALF / GALOSH_F_FULL */

  /* Free half-res L/C buffers */
  dt_free_align(luma);    luma = NULL;
  dt_free_align(chroma1); chroma1 = NULL;
  dt_free_align(chroma2); chroma2 = NULL;
  dt_free_align(chroma3); chroma3 = NULL;
  dt_free_align(c1_out);  c1_out = NULL;
  dt_free_align(c2_out);  c2_out = NULL;
  dt_free_align(c3_out);  c3_out = NULL;

  fprintf(stderr, "[rawdenoise] GALOSH: done\n");

  for(int c = 0; c < 4; c++) { dt_free_align(ch_gat[c]); ch_gat[c] = NULL; }
  return;

cleanup_rawlc:
  for(int c = 0; c < 4; c++) dt_free_align(ch_gat[c]);
  dt_free_align(luma);
  dt_free_align(chroma1);
  dt_free_align(chroma2);
  dt_free_align(chroma3);
  dt_free_align(c1_pilot);
  dt_free_align(c2_pilot);
  dt_free_align(c3_pilot);
  dt_free_align(c1_out);
  dt_free_align(c2_out);
  dt_free_align(c3_out);
}


/* --- main() --- */
int main(int argc, char **argv)
{
  if(argc < 5)
  {
    fprintf(stderr,
      "Usage: %s input.bin output.bin width height\n"
      "       [method] [strength] [luma_str] [chroma_str]\n"
      "       [alpha] [sigma_sq]\n"
      "\n"
      "  method:  'galosh' or 'ours' (GALOSH local WHT shrinkage, default)\n"
      "  luma_str:   sigma_L for luma shrinkage (default 0.5, user-tunable)\n"
      "  chroma_str: sigma_C for chroma shrinkage (default 1.0, user-tunable)\n"
      "  alpha:      P-G shot noise gain (auto if <= 0)\n"
      "  sigma_sq:   read noise variance (auto if <= 0)\n",
      argv[0]);
    return 1;
  }

  const char *input_file = argv[1];
  const char *output_file = argv[2];
  const int width = atoi(argv[3]);
  const int height = atoi(argv[4]);

  const char *method = (argc > 5) ? argv[5] : "galosh";
  const float strength = (argc > 6) ? (float)atof(argv[6]) : 1.0f;

  /* sigma_L (luma shrinkage strength):
   *   In GAT-normalized space, true noise sigma = 1.0.
   *   sigma_L < 1.0 means conservative denoising (preserve detail).
   *   Default 1.0: full-strength denoising matching the noise model.
   *   Applied to full-res raw GALOSH (Phase 4) where sub-pixel
   *   detail matters; also used for half-res luma pilot (Phase 3).
   *
   * sigma_C (chroma shrinkage strength):
   *   Applied in Phase 3 luma-guided chroma Wiener.
   *   Default 1.0: matches GAT-normalized noise sigma exactly.
   *   Since chroma Wiener uses luma pilot (not chroma pilot),
   *   sigma_C directly controls the denoising/preservation tradeoff
   *   without affecting the pilot quality. */
  const float luma_str = (argc > 7) ? (float)atof(argv[7]) : 0.5f;
  const float chroma_str = (argc > 8) ? (float)atof(argv[8]) : 1.0f;
  /* alpha / sigma_sq currently ignored -- fully blind estimation used */
  (void)strength; /* strength is absorbed into luma_str/chroma_str */

  if(width <= 0 || height <= 0)
  {
    fprintf(stderr, "Invalid dimensions: %dx%d\n", width, height);
    return 1;
  }

  fprintf(stderr, "Raw Denoiser Standalone v6 (GALOSH)\n");
  fprintf(stderr, "  Input:  %s (%dx%d)\n", input_file, width, height);
  fprintf(stderr, "  Method: %s\n", method);
  fprintf(stderr, "  Params: luma=%.2f chroma=%.2f\n", luma_str, chroma_str);

  /* Initialize Kaiser window */
  init_galosh_kaiser();

  /* Read input */
  const size_t npixels = (size_t)width * height;
  float *in = dt_alloc_align_float(npixels);
  float *out = dt_alloc_align_float(npixels);
  if(!in || !out) { fprintf(stderr, "Memory allocation failed\n"); return 1; }

  FILE *fin = fopen(input_file, "rb");
  if(!fin) { fprintf(stderr, "Cannot open %s\n", input_file); return 1; }
  size_t nread = fread(in, sizeof(float), npixels, fin);
  fclose(fin);
  if(nread != npixels)
  {
    fprintf(stderr, "Read %zu floats, expected %zu\n", nread, npixels);
    return 1;
  }

  /* Process */
  dt_iop_roi_t roi = { .width = width, .height = height };
  double t_start = omp_get_wtime();

  if(strcmp(method, "galosh") == 0 || strcmp(method, "ours") == 0)
  {
    gat_galosh_denoise_rawlc(in, out, &roi, luma_str, chroma_str, 0);
  }
  else
  {
    fprintf(stderr, "Unknown method '%s'. Use 'galosh' or 'ours'.\n", method);
    dt_free_align(in);
    dt_free_align(out);
    return 1;
  }

  double elapsed = omp_get_wtime() - t_start;
  fprintf(stderr, "  Elapsed: %.2f seconds\n", elapsed);

  /* Write output */
  FILE *fout = fopen(output_file, "wb");
  if(!fout) { fprintf(stderr, "Cannot open %s for writing\n", output_file); return 1; }
  fwrite(out, sizeof(float), npixels, fout);
  fclose(fout);

  fprintf(stderr, "  Output: %s\n", output_file);

  dt_free_align(in);
  dt_free_align(out);
  return 0;
}
