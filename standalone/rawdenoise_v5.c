/*
 * Standalone raw denoiser — extracted from darktable rawdenoise.c (v4)
 * RAW L/C BM3D + BM3D-CFA
 *
 * Usage: rawdenoise_v4 input.bin output.bin width height
 *        [method] [strength] [luma_str] [chroma_str] [alpha] [sigma_sq]
 *
 * Input/output: 32-bit float raw Bayer mosaic (RGGB), row-major
 *
 * Build (MSYS2 UCRT64):
 *   gcc -O2 -march=native -msse2 -msse3 -fopenmp -lm -o rawdenoise_v4 rawdenoise_v4.c
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

/* ─── darktable compatibility stubs ─────────────────── */
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
static inline void dt_free_align(void *p) {
    if(!p) return;
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

typedef struct { int width, height; } dt_iop_roi_t;

/* ─── BEGIN rawdenoise.c core (line 602-4094) ─── */


/* --- 2a. Transform kernels (DCT, WHT) --- */

#define BM3D_PATCH_SIZE 8
#define BM3D_PATCH_PIXELS (BM3D_PATCH_SIZE * BM3D_PATCH_SIZE)
#define BM3D_STEP         2      /* Step 2 reference patch stride */
#define BM3D_STEP1_STRIDE 4      /* Step 1 (NL-Means pilot) stride — coarser is OK */
#define BM3D_SEARCH_RAD   16
#define BM3D_MAX_MATCHED  32
#define BM3D_TAU_MATCH    2500.0f  /* Step 1: matching threshold on noisy data (Lebrun, IPOL 2012) */
#define BM3D_LAMBDA_3D    2.7f
#define BM3D_TAU_MATCH_W  400.0f  /* Step 2: matching threshold on pilot estimate */

/* Kaiser-Bessel window for patch aggregation (Lebrun, IPOL 2012).
 * Without windowing, patch boundaries create visible grid artifacts in
 * smooth gradients because the set of contributing patches changes
 * abruptly at every `step` pixels.  The window tapers each patch's
 * contribution: high weight at center, near-zero at edges, ensuring
 * smooth transitions between overlapping patches.
 *
 * w(n) = I0(beta * sqrt(1 - ((2n/(N-1))-1)^2)) / I0(beta)
 * beta = 2.0 (moderate taper), N = BM3D_PATCH_SIZE = 8.
 * 2D window = outer product of 1D window. */
static const float bm3d_kaiser_1d[BM3D_PATCH_SIZE] = {
  0.34012f, 0.59885f, 0.84123f, 0.97659f,
  0.97659f, 0.84123f, 0.59885f, 0.34012f
};
static float bm3d_kaiser_2d[BM3D_PATCH_PIXELS];
/* Half-res 4×4 Kaiser: averaged from consecutive pairs of the 8-point window.
 * Point-sampling at stride-2 produces an asymmetric window [0.34,0.84,0.98,0.60]
 * which causes visible grid artifacts. Pair-averaging gives symmetric
 * [0.47,0.91,0.91,0.47] for proper overlap-add in half-res aggregation. */
static float bm3d_kaiser_half_2d[16]; /* 4×4 = BM3D_HALF_PIXELS */
static void init_kaiser_window(void)
{
  for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
    for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++)
      bm3d_kaiser_2d[dy * BM3D_PATCH_SIZE + dx] = bm3d_kaiser_1d[dy] * bm3d_kaiser_1d[dx];

  /* Build symmetric 4-point Kaiser by averaging consecutive pairs */
  float kaiser_half_1d[4];
  for(int i = 0; i < 4; i++)
    kaiser_half_1d[i] = (bm3d_kaiser_1d[2 * i] + bm3d_kaiser_1d[2 * i + 1]) * 0.5f;
  for(int dy = 0; dy < 4; dy++)
    for(int dx = 0; dx < 4; dx++)
      bm3d_kaiser_half_2d[dy * 4 + dx] = kaiser_half_1d[dy] * kaiser_half_1d[dx];
}
#define BM3D_SEARCH_RAD2  12     /* Step 2: smaller search radius (pilot is smoother, matches cluster nearby) */
#define BM3D_SEARCH_RAD2_HALF (BM3D_SEARCH_RAD2 / 2)  /* Half-res equivalent for L/C-plane Step 2 */
#define BM3D_EST_BLOCK    32

/* Precomputed DCT-II basis and its transpose (for inverse DCT) */
static float dct_basis[BM3D_PATCH_SIZE][BM3D_PATCH_SIZE];
static float dct_basis_t[BM3D_PATCH_SIZE][BM3D_PATCH_SIZE]; /* transposed */
static int dct_basis_initialized = 0;

static void init_dct_basis(void)
{
  if(dct_basis_initialized) return;
  for(int k = 0; k < BM3D_PATCH_SIZE; k++)
  {
    const float alpha = (k == 0) ? 1.0f / sqrtf((float)BM3D_PATCH_SIZE)
                                 : sqrtf(2.0f / (float)BM3D_PATCH_SIZE);
    for(int n = 0; n < BM3D_PATCH_SIZE; n++)
    {
      dct_basis[k][n] = alpha * cosf((float)M_PI * (2.0f * n + 1.0f) * k
                                     / (2.0f * BM3D_PATCH_SIZE));
      dct_basis_t[n][k] = dct_basis[k][n];  /* transposed: [n][k] = [k][n] */
    }
  }
  dct_basis_initialized = 1;
}

/* SSE-accelerated 1D DCT-II / inverse DCT-II (length 8).
 * Process one row/column: 8 values × 8 basis vectors.
 * Each basis vector is loaded as 2×__m128, dot product via SSE.
 * Uses pre-transposed dct_basis_t for inverse to avoid gather. */
static inline void dct1d_8_forward(const float *restrict in, float *restrict out,
                                    const int in_stride, const int out_stride)
{
  float inv[8] __attribute__((aligned(16)));
  for(int i = 0; i < 8; i++) inv[i] = in[i * in_stride];
  const __m128 v0 = _mm_load_ps(inv);
  const __m128 v1 = _mm_load_ps(inv + 4);
  for(int k = 0; k < 8; k++)
  {
    const __m128 b0 = _mm_loadu_ps(&dct_basis[k][0]);
    const __m128 b1 = _mm_loadu_ps(&dct_basis[k][4]);
    __m128 p = _mm_add_ps(_mm_mul_ps(b0, v0), _mm_mul_ps(b1, v1));
    p = _mm_add_ps(p, _mm_movehl_ps(p, p));
    p = _mm_add_ss(p, _mm_shuffle_ps(p, p, 1));
    _mm_store_ss(&out[k * out_stride], p);
  }
}

static inline void dct1d_8_inverse(const float *restrict in, float *restrict out,
                                    const int in_stride, const int out_stride)
{
  float inv[8] __attribute__((aligned(16)));
  for(int i = 0; i < 8; i++) inv[i] = in[i * in_stride];
  const __m128 v0 = _mm_load_ps(inv);
  const __m128 v1 = _mm_load_ps(inv + 4);
  for(int n = 0; n < 8; n++)
  {
    /* dct_basis_t[n] = { basis[0][n], basis[1][n], ..., basis[7][n] } */
    const __m128 b0 = _mm_loadu_ps(&dct_basis_t[n][0]);
    const __m128 b1 = _mm_loadu_ps(&dct_basis_t[n][4]);
    __m128 p = _mm_add_ps(_mm_mul_ps(b0, v0), _mm_mul_ps(b1, v1));
    p = _mm_add_ps(p, _mm_movehl_ps(p, p));
    p = _mm_add_ss(p, _mm_shuffle_ps(p, p, 1));
    _mm_store_ss(&out[n * out_stride], p);
  }
}

static void dct2d_forward(const float *restrict in, float *restrict out)
{
  float tmp[BM3D_PATCH_PIXELS];
  for(int row = 0; row < BM3D_PATCH_SIZE; row++)
    dct1d_8_forward(in + row * BM3D_PATCH_SIZE, tmp + row * BM3D_PATCH_SIZE, 1, 1);
  for(int col = 0; col < BM3D_PATCH_SIZE; col++)
    dct1d_8_forward(tmp + col, out + col, BM3D_PATCH_SIZE, BM3D_PATCH_SIZE);
}

static void dct2d_inverse(const float *restrict in, float *restrict out)
{
  float tmp[BM3D_PATCH_PIXELS];
  for(int col = 0; col < BM3D_PATCH_SIZE; col++)
    dct1d_8_inverse(in + col, tmp + col, BM3D_PATCH_SIZE, BM3D_PATCH_SIZE);
  for(int row = 0; row < BM3D_PATCH_SIZE; row++)
    dct1d_8_inverse(tmp + row * BM3D_PATCH_SIZE, out + row * BM3D_PATCH_SIZE, 1, 1);
}

/* 4×4 DCT-II basis for WHT-decomposed sub-patches in full-res BM3D.
 * After WHT, each 8×8 Bayer patch becomes 4×4 × 4ch (L,C1,C2,C3).
 * DCT is applied to each 4×4 sub-patch independently. */
#define BM3D_HALF_PATCH 4
#define BM3D_HALF_PIXELS (BM3D_HALF_PATCH * BM3D_HALF_PATCH)

/* ----------------------------------------------------------------
 * Design note — no local noise variance map.
 *
 * A local variance map (box-mean of (noisy − pilot)²) was tested
 * as a per-pixel σ² multiplier inside Step2 Wiener. However, the
 * residual variance after Step1 is intrinsically ~0.25× the input
 * σ², causing the map to collapse to a near-constant 0.25 clamp
 * across the entire image. This halved effective Wiener strength
 * uniformly rather than providing local adaptation.
 *
 * Current design: no varmap. RMS unified σ normalization guarantees
 * Var[L]=Var[Cₖ]=1 exactly after WHT, so Step2 uses the global σ²
 * directly. Per-channel deviation from single-shot estimator bias
 * is absorbed by WHT averaging — no local correction needed.
 *
 * 設計メモ — local variance map なし。varmap は Step1 後の residual
 * 分散が本質的に ~0.25×σ² のため定数減衰として働き、local 適応器
 * にならなかった。RMS 統一 σ で Var[L/Cₖ]=1 が厳密に成立するため
 * Step2 は global σ² を直接使用する。
 * ---------------------------------------------------------------- */

static float dct4_basis[BM3D_HALF_PATCH][BM3D_HALF_PATCH];
static float dct4_basis_t[BM3D_HALF_PATCH][BM3D_HALF_PATCH];
/* SSE: each basis row as __m128 for vectorised 4×4 DCT */
static __m128 dct4_basis_sse[BM3D_HALF_PATCH];
static __m128 dct4_basis_t_sse[BM3D_HALF_PATCH];
static int dct4_basis_initialized = 0;

static void init_dct4_basis(void)
{
  if(dct4_basis_initialized) return;
  for(int k = 0; k < BM3D_HALF_PATCH; k++)
  {
    const float alpha = (k == 0) ? 1.0f / sqrtf((float)BM3D_HALF_PATCH)
                                 : sqrtf(2.0f / (float)BM3D_HALF_PATCH);
    for(int n = 0; n < BM3D_HALF_PATCH; n++)
    {
      dct4_basis[k][n] = alpha * cosf((float)M_PI * (2.0f * n + 1.0f) * k
                                       / (2.0f * BM3D_HALF_PATCH));
      dct4_basis_t[n][k] = dct4_basis[k][n];
    }
  }
  /* Pre-load basis rows as __m128 for SSE 4×4 DCT */
  for(int k = 0; k < BM3D_HALF_PATCH; k++)
  {
    dct4_basis_sse[k]   = _mm_loadu_ps(dct4_basis[k]);
    dct4_basis_t_sse[k] = _mm_loadu_ps(dct4_basis_t[k]);
  }
  dct4_basis_initialized = 1;
}

/* SSE-optimised 4×4 2-D DCT forward transform.
 * Row transform: each row is 4 floats = 1 __m128.
 * Dot product via mul + 2 hadd (SSE3). Column transform is identical
 * because the 4×4 tmp matrix is stored row-major → columns are gathered
 * with shuffles and processed the same way.
 *
 * Separable: out = B · in · B^T  (row-first, then column-first). */
static void dct2d_4x4_forward(const float *restrict in, float *restrict out)
{
  /* --- Pass 1: row transform  tmp[row][k] = Σ_n B[k][n] · in[row][n] --- */
  __m128 tmp_rows[BM3D_HALF_PATCH];
  for(int row = 0; row < BM3D_HALF_PATCH; row++)
  {
    const __m128 r = _mm_loadu_ps(in + row * BM3D_HALF_PATCH);
    /* Dot-product of r with each basis row k: dp = Σ r[i]*B[k][i] */
    __m128 d0 = _mm_mul_ps(r, dct4_basis_sse[0]);
    __m128 d1 = _mm_mul_ps(r, dct4_basis_sse[1]);
    __m128 d2 = _mm_mul_ps(r, dct4_basis_sse[2]);
    __m128 d3 = _mm_mul_ps(r, dct4_basis_sse[3]);
    /* Horizontal reduction: hadd pairs then hadd again */
    __m128 s01 = _mm_hadd_ps(d0, d1);   /* [d0.01, d0.23, d1.01, d1.23] */
    __m128 s23 = _mm_hadd_ps(d2, d3);
    tmp_rows[row] = _mm_hadd_ps(s01, s23); /* [dp0, dp1, dp2, dp3] */
  }

  /* --- Pass 2: column transform  out[k][col] = Σ_n B[k][n] · tmp[n][col]
   * tmp is stored as 4 __m128 rows. Column j across rows = gather.
   * Instead of gathering columns, we use the identity:
   *   out_row_k = Σ_n B[k][n] * tmp_row_n   (broadcast scalar, mul vec)
   * This is a standard matrix-vector product per output row. --- */
  for(int k = 0; k < BM3D_HALF_PATCH; k++)
  {
    __m128 acc = _mm_mul_ps(_mm_set1_ps(dct4_basis[k][0]), tmp_rows[0]);
    acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(dct4_basis[k][1]), tmp_rows[1]));
    acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(dct4_basis[k][2]), tmp_rows[2]));
    acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(dct4_basis[k][3]), tmp_rows[3]));
    _mm_storeu_ps(out + k * BM3D_HALF_PATCH, acc);
  }
}

/* SSE-optimised 4×4 2-D DCT inverse transform.
 * out = B^T · in · B   (column-first, then row-first).
 * Same SSE strategy as forward but with transposed basis. */
static void dct2d_4x4_inverse(const float *restrict in, float *restrict out)
{
  /* --- Pass 1: column transform  tmp[n][col] = Σ_k B^T[n][k] · in[k][col]
   *     = Σ_k B^T[n][k] * in_row_k   (broadcast scalar, mul vec) --- */
  __m128 tmp_rows[BM3D_HALF_PATCH];
  for(int n = 0; n < BM3D_HALF_PATCH; n++)
  {
    __m128 acc = _mm_mul_ps(_mm_set1_ps(dct4_basis_t[n][0]),
                            _mm_loadu_ps(in));
    acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(dct4_basis_t[n][1]),
                                     _mm_loadu_ps(in + BM3D_HALF_PATCH)));
    acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(dct4_basis_t[n][2]),
                                     _mm_loadu_ps(in + 2 * BM3D_HALF_PATCH)));
    acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(dct4_basis_t[n][3]),
                                     _mm_loadu_ps(in + 3 * BM3D_HALF_PATCH)));
    tmp_rows[n] = acc;
  }

  /* --- Pass 2: row transform  out[row][n] = Σ_k B^T[n][k] · tmp[row][k]
   *     Each row: dot-product of tmp_row with each B^T row → hadd --- */
  for(int row = 0; row < BM3D_HALF_PATCH; row++)
  {
    __m128 d0 = _mm_mul_ps(tmp_rows[row], dct4_basis_t_sse[0]);
    __m128 d1 = _mm_mul_ps(tmp_rows[row], dct4_basis_t_sse[1]);
    __m128 d2 = _mm_mul_ps(tmp_rows[row], dct4_basis_t_sse[2]);
    __m128 d3 = _mm_mul_ps(tmp_rows[row], dct4_basis_t_sse[3]);
    __m128 s01 = _mm_hadd_ps(d0, d1);
    __m128 s23 = _mm_hadd_ps(d2, d3);
    _mm_storeu_ps(out + row * BM3D_HALF_PATCH, _mm_hadd_ps(s01, s23));
  }
}

/* WHT decomposition: 8×8 Bayer patch → 4×4 × 4ch (L, C1, C2, C3).
 * Each 2×2 cell in the 8×8 patch is transformed independently.
 * L = average (luma-like), C1/C2/C3 = differences (chroma-like).
 * Noise variance is preserved: if input noise σ²=1 per pixel,
 * each WHT component also has σ²=1 (orthogonal with /2 norm).
 *
 * SSE version: processes all 4 columns (j=0..3) simultaneously.
 * Row pair (2i, 2i+1) of the 8×8 patch = 8+8 = 16 consecutive floats.
 * Deinterleave even/odd columns → a,b,c,d as __m128 → add/sub. */
static inline void wht_decompose_8x8(const float patch[BM3D_PATCH_PIXELS],
                                       float lc[4][BM3D_HALF_PIXELS])
{
  const __m128 half = _mm_set1_ps(0.5f);
  for(int i = 0; i < BM3D_HALF_PATCH; i++)
  {
    /* Load 8 floats from even row: [a0,b0,a1,b1,a2,b2,a3,b3] */
    const float *row0 = patch + (2 * i) * BM3D_PATCH_SIZE;
    const float *row1 = patch + (2 * i + 1) * BM3D_PATCH_SIZE;
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
    _mm_storeu_ps(lc[0] + i * BM3D_HALF_PATCH, _mm_mul_ps(_mm_add_ps(ab_sum, cd_sum), half));
    _mm_storeu_ps(lc[1] + i * BM3D_HALF_PATCH, _mm_mul_ps(_mm_add_ps(ab_dif, cd_dif), half));
    _mm_storeu_ps(lc[2] + i * BM3D_HALF_PATCH, _mm_mul_ps(_mm_sub_ps(ab_sum, cd_sum), half));
    _mm_storeu_ps(lc[3] + i * BM3D_HALF_PATCH, _mm_mul_ps(_mm_sub_ps(ab_dif, cd_dif), half));
  }
}

/* SSE inverse WHT: 4×4 × 4ch → 8×8 Bayer patch.
 * Processes all 4 columns simultaneously, interleaving back to 8-wide rows. */
static inline void wht_reconstruct_8x8(const float lc[4][BM3D_HALF_PIXELS],
                                         float patch[BM3D_PATCH_PIXELS])
{
  const __m128 half = _mm_set1_ps(0.5f);
  for(int i = 0; i < BM3D_HALF_PATCH; i++)
  {
    const __m128 L  = _mm_loadu_ps(lc[0] + i * BM3D_HALF_PATCH);
    const __m128 C1 = _mm_loadu_ps(lc[1] + i * BM3D_HALF_PATCH);
    const __m128 C2 = _mm_loadu_ps(lc[2] + i * BM3D_HALF_PATCH);
    const __m128 C3 = _mm_loadu_ps(lc[3] + i * BM3D_HALF_PATCH);
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
    float *out0 = patch + (2 * i) * BM3D_PATCH_SIZE;
    float *out1 = patch + (2 * i + 1) * BM3D_PATCH_SIZE;
    _mm_storeu_ps(out0,     _mm_unpacklo_ps(a, b));  /* a0,b0,a1,b1 */
    _mm_storeu_ps(out0 + 4, _mm_unpackhi_ps(a, b));  /* a2,b2,a3,b3 */
    _mm_storeu_ps(out1,     _mm_unpacklo_ps(cc, dd)); /* c0,d0,c1,d1 */
    _mm_storeu_ps(out1 + 4, _mm_unpackhi_ps(cc, dd)); /* c2,d2,c3,d3 */
  }
}

static void hadamard_transform(float *data, const int len)
{
  for(int step = 1; step < len; step *= 2)
    for(int i = 0; i < len; i += 2 * step)
      for(int j = i; j < i + step; j++)
      {
        const float a = data[j], b = data[j + step];
        data[j] = a + b;
        data[j + step] = a - b;
      }
  const float norm = 1.0f / sqrtf((float)len);
  for(int i = 0; i < len; i++) data[i] *= norm;
}

/* --- 2b. Noise estimation --- */

static int compare_floats_bm3d(const void *a, const void *b)
{
  const float fa = *(const float *)a, fb = *(const float *)b;
  /* NaN-safe: NaN sorts to end */
  if(fa != fa) return 1;
  if(fb != fb) return -1;
  return (fa > fb) - (fa < fb);
}

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

/* fast_expf is already provided by openmp_maths.h */

typedef struct bm3d_noise_params_t { float alpha, sigma_sq; } bm3d_noise_params_t;

/* Robust Poisson-Gaussian noise estimation using Laplacian (second-order
 * difference) and MAD.  Uses ALL Bayer pixels so that all 4 sub-channels
 * share exactly the same noise parameters — critical to prevent jaggies.
 *
 * Key insight: the first-difference estimator  d = raw[x+2] - raw[x]
 * is contaminated by signal gradients:  Var(d) = 2σ² + gradient².
 * For typical images the gradient term dominates, overestimating noise
 * by 10–50×, which makes the GAT over-stabilize (σ_GAT ≪ 1).
 *
 * The Laplacian (second difference) d = raw[x] - 2·raw[x+2] + raw[x+4]
 * perfectly cancels any linear signal component:
 *   d_signal = f - 2(f+f'h) + (f+2f'h) = 0   (h = 2 pixels)
 * Only noise remains:  Var(d) = (1² + 2² + 1²)·σ² = 6σ²
 *
 * Binned by intensity → median(d²) → robust σ² per bin → linear fit
 * gives the Poisson-Gaussian model: noise_var = α·intensity + σ².       */
static bm3d_noise_params_t bm3d_estimate_noise(const float *raw,
                                                const int width, const int height)
{
  /* V3 block-based Laplacian + lower envelope + Huber M-estimator + dark pixel pass
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
  bm3d_noise_params_t result = { 1e-4f, 1e-6f };

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

  /* Steps 1-2: per-block mean and Laplacian noise variance (horizontal+vertical)
   * Collect ~96 Laplacian samples (48 horizontal + 48 vertical) within each 8x8 block
   * Intra-block MAD(|L|) is robust even with some edge pixels present */
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
    qsort(sort_buf, cnt, sizeof(float), compare_floats_bm3d);
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

  /* Step 5: estimate sigma_sq directly from darkest pixels
   * For x ~ 0, Var(noise) ~ sigma_sq (shot noise is negligible)
   * Use 10th percentile of pixel values as threshold, collect
   * Laplacians from pixels below it, compute sigma_sq via MAD */
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

/* --- 2c. GAT (forward / inverse / LUT) --- */

/* ----------------------------------------------------------------
 *  GAT: Generalized Anscombe Transform (variance stabilization)
 *  GAT: 一般化Anscombe変換（分散安定化）
 *
 *  Forward (piecewise C¹ VST extending Foi GAT to all reals):
 *    For y >= y_break = -3α/8:
 *       T(y) = (2/α) * sqrt(αy + 3α²/8 + σ²)              (Foi sqrt branch)
 *    For y <  y_break:
 *       T(y) = T_break + (y - y_break)/σ                  (linear branch)
 *    where T_break = 2σ/α (continuity) and slope = 1/σ (C¹).
 *
 *    Variance is stabilized in BOTH branches to ≈ 1:
 *      - sqrt branch: stabilizes Poisson(αX) + Gaussian(σ²) → Var≈1
 *      - linear branch: in deep negatives, signal-dependent component is
 *        negligible, observed variance is ≈ σ², linear scaling by 1/σ
 *        gives Var≈1 as well.
 *    No input clamping → distribution shape preserved across channels
 *    (critical for L/C BM3D where 4ch shape mismatch creates false chroma).
 *
 *  Inverse: Exact unbiased inverse via Gauss-Hermite quadrature LUT
 *    in the sqrt branch; analytical linear inverse in the linear branch.
 *    逆変換: sqrt 領域は Gauss-Hermite 求積による厳密不偏逆変換、
 *    線形領域は解析逆。
 *    Ref: Anscombe (1948), Foi et al. (Sig.Proc. 2008),
 *         Makitalo & Foi (TIP 2013)
 * ---------------------------------------------------------------- */

static inline float gat_forward(const float x, const float alpha, const float sigma_sq)
{
  const float x_safe = (x == x) ? x : 0.0f;  /* NaN guard only — NO clamping */
  const float y_break = -0.375f * alpha;     /* -3α/8 */
  if(x_safe >= y_break)
  {
    /* sqrt branch: classical Foi GAT */
    return (2.0f / alpha) * sqrtf(alpha * x_safe + 0.375f * alpha * alpha + sigma_sq);
  }
  else
  {
    /* linear branch: C¹ extension */
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
  /* Piecewise C¹ VST parameters (set at table build time) */
  float alpha;
  float sigma_raw;             /* sqrt(sigma_sq) */
  float y_break;               /* -3α/8: forward-domain transition */
  float t_break;               /* 2σ/α : GAT-domain transition */
  int valid;
} gat_inv_table_t;

/* NOTE: gat_inv_table is a static singleton. This is safe because darktable's
 * pixelpipe processes IOPs sequentially per pipe, and the table is rebuilt
 * per-image with image-specific noise parameters. If concurrent processing
 * is ever needed, this must be made per-invocation. */
static gat_inv_table_t gat_inv_table = { .valid = 0 };

/* 10-point Gauss-Hermite quadrature (weight function e^{-x²}).
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

    /* Summation over Poisson distribution: E[T(Poisson(lambda)*alpha + N(0, sigma_sq))]
     * where T is the piecewise C¹ VST (sqrt + linear). */
    double expected_gat = 0.0;
    const int k_max = (int)(lambda + 8.0 * sqrt(fmax(lambda, 1.0))) + 20;
    double log_prob = -lambda;  /* log P(0; lambda) */

    for(int k = 0; k <= k_max; k++)
    {
      if(k > 0) log_prob += log(lambda) - log((double)k);
      const double prob = exp(log_prob);
      if(prob < 1e-15 && k > (int)lambda + 1) break;

      /* E[T(k*alpha + Z)], Z ~ N(0, sigma_sq), via Gauss-Hermite.
       * Piecewise: sqrt branch for noisy_y >= y_break, linear branch otherwise. */
      double eg = 0.0;
      for(int g = 0; g < 10; g++)
      {
        const double z = 1.4142135623730951 * sig * gh_nodes[g]; /* sqrt(2)*sig*node */
        const double noisy_y = (double)k * a + z;      /* observed signal */
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
   *   ⇒ y = y_break + sigma * (T - t_break).
   * No clamp to 0 — preserves distribution shape so downstream phases
   * (mean restore, BM3D match) see symmetric residuals. */
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

  /* Linear interpolation — no clamp to 0, see comment above. */
  const float d0 = gat_inv_table.d[lo], d1 = gat_inv_table.d[hi];
  const float t = (D - d0) / fmaxf(d1 - d0, 1e-10f);
  return gat_inv_table.x[lo] + t * (gat_inv_table.x[hi] - gat_inv_table.x[lo]);
}

/* Measure actual sigma in GAT domain using MAD of Laplacian (second differences).
 * First differences (row[x+1] - row[x]) are contaminated by signal gradients,
 * causing overestimation in images with strong gradients (the same problem that
 * bm3d_estimate_noise() solves with its Laplacian estimator).
 *
 * Laplacian: L = data[x] - 2*data[x+1] + data[x+2] cancels linear gradients.
 * For iid noise with variance sigma^2:  Var(L) = (1+4+1)*sigma^2 = 6*sigma^2
 * |L| ~ sqrt(6)*sigma * |N(0,1)|, so MAD(|L|) = sqrt(6)*sigma*0.6745
 * => sigma = MAD / (0.6745 * sqrt(6)) = MAD / 1.6521 */

/* Full-mosaic sigma estimation: uses stride-2 Laplacian to stay within
 * the same Bayer channel, avoiding contamination from cross-channel
 * signal differences (R-G-B DC offsets would appear as huge gradients
 * in stride-1 Laplacian, grossly overestimating noise).
 *
 * L = data[x] - 2*data[x+2] + data[x+4]   (all same Bayer channel)
 * Var(L) = 6*sigma^2 => sigma = MAD(|L|) / (0.6745*sqrt(6)) */
static float estimate_gat_sigma_mosaic(const float *data, const int width, const int height)
{
  const int n_samples = MIN(width * height / 6, 200000);
  float *abs_laps = dt_alloc_align_float(n_samples + 1);
  if(!abs_laps) return 1.0f;

  int count = 0;
  for(int y = 0; y < height && count < n_samples; y++)
  {
    const float *row = data + (size_t)y * width;
    /* Stride-2 sampling: row[x], row[x+2], row[x+4] are same Bayer channel */
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

/* --- 2d. Block matching utilities --- */

static int bm3d_next_pow2(int n)
{
  if(n <= 1) return 1;
  if(n <= 2) return 2;
  if(n <= 4) return 4;
  if(n <= 8) return 8;
  if(n <= 16) return 16;
  return 32;
}

/* Compute patch SSD with early termination + SSE vectorization.
 * Each row of the 8×8 patch is processed as two __m128 (4+4 floats).
 * Early exit after every 2 rows balances branch cost vs. skip savings. */
#include <xmmintrin.h>  /* SSE  */
#include <emmintrin.h>  /* SSE2 */
#include <pmmintrin.h>  /* SSE3: _mm_hadd_ps for 4×4 DCT */

static inline float bm3d_patch_ssd(const float *img, const int width,
                                   const int r1, const int c1,
                                   const int r2, const int c2,
                                   const float threshold)
{
  __m128 acc = _mm_setzero_ps();
  for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
  {
    const float *p1 = img + (size_t)(r1 + dy) * width + c1;
    const float *p2 = img + (size_t)(r2 + dy) * width + c2;
    /* BM3D_PATCH_SIZE == 8: two groups of 4 floats */
    const __m128 a0 = _mm_loadu_ps(p1);
    const __m128 b0 = _mm_loadu_ps(p2);
    const __m128 d0 = _mm_sub_ps(a0, b0);
    acc = _mm_add_ps(acc, _mm_mul_ps(d0, d0));
    const __m128 a1 = _mm_loadu_ps(p1 + 4);
    const __m128 b1 = _mm_loadu_ps(p2 + 4);
    const __m128 d1 = _mm_sub_ps(a1, b1);
    acc = _mm_add_ps(acc, _mm_mul_ps(d1, d1));
    /* Early termination every 2 rows (16 floats processed) */
    if((dy & 1) == 1)
    {
      /* horizontal sum of acc to check threshold */
      __m128 s = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
      s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
      float ssd;
      _mm_store_ss(&ssd, s);
      if(ssd >= threshold) return ssd;
    }
  }
  /* Final horizontal sum */
  __m128 s = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  float ssd;
  _mm_store_ss(&ssd, s);
  return ssd;
}

typedef struct bm3d_match_t { float dist; int row, col; } bm3d_match_t;

/* For N ≤ 32, insertion sort beats qsort by 2-3× due to no function-pointer overhead */
static void sort_matches(bm3d_match_t *m, const int n)
{
  for(int i = 1; i < n; i++)
  {
    const bm3d_match_t key = m[i];
    int j = i - 1;
    while(j >= 0 && m[j].dist > key.dist) { m[j + 1] = m[j]; j--; }
    m[j + 1] = key;
  }
}

/* Precompute patch mean for every possible patch position.
 * Used as a cheap pre-filter: if N·(mean1-mean2)² > tau, the SSD must
 * also exceed tau → skip the expensive 64-element SSD computation.
 * This rejects ~60-80% of candidates before touching the pixel data. */
static float *bm3d_precompute_patch_means(const float *img, const int width, const int height)
{
  const int rmax = height - BM3D_PATCH_SIZE;
  const int cmax = width - BM3D_PATCH_SIZE;
  const int mwidth = cmax + 1;
  const int mheight = rmax + 1;
  float *means = dt_alloc_align_float((size_t)mwidth * mheight);
  if(!means) return NULL;

  /* Use integral image for O(1) per-patch mean computation */
  const size_t iw = (size_t)(width + 1);
  double *integral = calloc(iw * (height + 1), sizeof(double));
  if(!integral) { dt_free_align(means); return NULL; }

  for(int y = 0; y < height; y++)
    for(int x = 0; x < width; x++)
      integral[(y + 1) * iw + (x + 1)] = (double)img[(size_t)y * width + x]
        + integral[y * iw + (x + 1)]
        + integral[(y + 1) * iw + x]
        - integral[y * iw + x];

  const float inv_n = 1.0f / (float)BM3D_PATCH_PIXELS;
  DT_OMP_FOR()
  for(int r = 0; r <= rmax; r++)
    for(int c = 0; c <= cmax; c++)
    {
      const double sum = integral[(r + BM3D_PATCH_SIZE) * iw + (c + BM3D_PATCH_SIZE)]
                       - integral[r * iw + (c + BM3D_PATCH_SIZE)]
                       - integral[(r + BM3D_PATCH_SIZE) * iw + c]
                       + integral[r * iw + c];
      means[r * mwidth + c] = (float)sum * inv_n;
    }

  free(integral);
  return means;
}

/* (Variance map removed — see design note near the top of the BM3D
 * section. Step2 uses global σ² directly; Var[L]=Var[Cₖ]=1 enforced
 * exactly by RMS unified σ normalization.) */

/* ================================================================
 * Compute 4-plane sum of SSDs for block matching (8×8 patch).
 * Basis-agnostic: works for both RGGB Bayer planes (Step1) and
 * L/C/C/C WHT-decomposed planes (would be Parseval-equivalent).
 * Sum across 4 planes preserves color boundary information that
 * luma-only matching misses.
 * ================================================================ */
static inline float bm3d_patch_ssd_4plane(
    const float *const ch[4], const int width,
    const int r1, const int c1, const int r2, const int c2,
    const float threshold)
{
  float total = 0.0f;
  for(int k = 0; k < 4; k++)
  {
    const float d = bm3d_patch_ssd(ch[k], width, r1, c1, r2, c2, threshold - total);
    total += d;
    if(total >= threshold) return total;
  }
  return total;
}

/* ================================================================
 * 4-plane SSD for 4×4 patches (used by half-res L/C Step 2).
 * Inline scalar — 4×4 = 16 floats per plane, no SSE benefit.
 * Early termination after each plane.
 * ================================================================ */
static inline float bm3d_patch_ssd_4plane_half(
    const float *const ch[4], const int width,
    const int r1, const int c1, const int r2, const int c2,
    const float threshold)
{
  float total = 0.0f;
  for(int k = 0; k < 4; k++)
  {
    const float *p = ch[k];
    float ssd = 0.0f;
    for(int dy = 0; dy < 4; dy++)
    {
      const float *p1 = p + (size_t)(r1 + dy) * width + c1;
      const float *p2 = p + (size_t)(r2 + dy) * width + c2;
      for(int dx = 0; dx < 4; dx++)
      {
        const float d = p1[dx] - p2[dx];
        ssd += d * d;
      }
    }
    total += ssd;
    if(total >= threshold) return total;
  }
  return total;
}

/* ================================================================
 *  RAW L/C BM3D — Step functions
 *  RAW L/C BM3D — ステップ関数
 *
 *  WHT (Walsh-Hadamard Transform) decomposes 2×2 Bayer blocks into
 *  luma (H0) and chroma (H1, H2, H3) at half resolution.
 *  BM3D operates on these L/C planes with separate strength control.
 *  WHT（Walsh-Hadamard変換）により2×2 Bayerブロックを輝度(H0)と
 *  色差(H1,H2,H3)に半解像度で分解し、L/C別の強度制御でBM3Dを適用。
 *
 *  Ref: Danielyan et al. "Cross-color BM3D filtering of noisy raw
 *       data" (LNLA 2009)
 * ================================================================ */

/* BM3D Step 1: DCT + Hadamard + Hard Thresholding pilot
 * on all 4 WHT components with SHARED aggregation weights.
 *
 * Matching uses RGGB 4ch SSD (before WHT) for color-aware grouping.
 * All 4 channels share the same matches and aggregation denominator
 * to prevent 2×2 grid artifact.
 *
 * Standard BM3D Step 1: 2D-DCT → 1D-Hadamard → hard threshold
 *                       → inverse → aggregate (1/N_nonzero)
 * ================================================================ */
static void bm3d_step1_lc(
    const float *const ch_rggb[4],  /* 4 RGGB channels for matching */
    const float *restrict luma_in,
    const float *restrict c1_in,
    const float *restrict c2_in,
    const float *restrict c3_in,
    float *restrict luma_out,
    float *restrict c1_out,
    float *restrict c2_out,
    float *restrict c3_out,
    const int width, const int height, const float sigma)
{
  const float tau = BM3D_TAU_MATCH * sigma * sigma;
  const float tau_4ch = tau * 4.0f;  /* 4ch SSD threshold */
  const float lambda_thr = BM3D_LAMBDA_3D * sigma;
  const size_t npixels = (size_t)width * height;

  const int step1 = BM3D_STEP1_STRIDE;
  const int rmax = height - BM3D_PATCH_SIZE, cmax = width - BM3D_PATCH_SIZE;
  const int mwidth = cmax + 1;

  /* Precompute patch means on luma for fast rejection */
  float *pmeans = bm3d_precompute_patch_means(luma_in, width, height);
  const float mean_thr = sqrtf(tau / (float)BM3D_PATCH_PIXELS);

  const float *ch_in[4] = { luma_in, c1_in, c2_in, c3_in };
  float *ch_out[4] = { luma_out, c1_out, c2_out, c3_out };

#ifdef _OPENMP
  const int max_threads = omp_get_max_threads();
#else
  const int max_threads = 1;
#endif
  const int nthreads = MIN(max_threads, 16);

  /* Per-thread accumulation: 4 numerators + 1 SHARED denominator */
  float **t_numer[4];
  float **t_denom;
  for(int c = 0; c < 4; c++)
    t_numer[c] = calloc(nthreads, sizeof(float *));
  t_denom = calloc(nthreads, sizeof(float *));

  gboolean alloc_ok = TRUE;
  for(int t = 0; t < nthreads && alloc_ok; t++)
  {
    t_denom[t] = dt_alloc_align_float(npixels);
    if(!t_denom[t]) { alloc_ok = FALSE; break; }
    memset(t_denom[t], 0, sizeof(float) * npixels);
    for(int c = 0; c < 4; c++)
    {
      t_numer[c][t] = dt_alloc_align_float(npixels);
      if(!t_numer[c][t]) { alloc_ok = FALSE; break; }
      memset(t_numer[c][t], 0, sizeof(float) * npixels);
    }
  }

  if(!alloc_ok)
  {
    for(int t = 0; t < nthreads; t++)
    {
      dt_free_align(t_denom[t]);
      for(int c = 0; c < 4; c++) dt_free_align(t_numer[c][t]);
    }
    free(t_denom);
    for(int c = 0; c < 4; c++) { free(t_numer[c]); memcpy(ch_out[c], ch_in[c], sizeof(float) * npixels); }
    dt_free_align(pmeans);
    return;
  }

  const int chunk_s1 = MAX(4, (rmax + step1) / step1 / (nthreads * 8));
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, chunk_s1)
#endif
  for(int ref_r = 0; ref_r <= rmax; ref_r += step1)
  {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif

    for(int ref_c = 0; ref_c <= cmax; ref_c += step1)
    {
      /* Block matching using RGGB 4ch SSD */
      bm3d_match_t matches[BM3D_MAX_MATCHED];
      int nmatches = 1;
      matches[0] = (bm3d_match_t){ 0.0f, ref_r, ref_c };
      float worst_dist = tau_4ch;

      const float ref_mean = pmeans ? pmeans[ref_r * mwidth + ref_c] : 0.0f;

      const int sr0 = MAX(0, ref_r - BM3D_SEARCH_RAD), sr1 = MIN(rmax, ref_r + BM3D_SEARCH_RAD);
      const int sc0 = MAX(0, ref_c - BM3D_SEARCH_RAD), sc1 = MIN(cmax, ref_c + BM3D_SEARCH_RAD);

      for(int sr = sr0; sr <= sr1; sr += BM3D_STEP)
        for(int sc = sc0; sc <= sc1; sc += BM3D_STEP)
        {
          if(sr == ref_r && sc == ref_c) continue;

          /* Fast rejection on luma mean */
          if(pmeans)
          {
            const float dm = ref_mean - pmeans[sr * mwidth + sc];
            if(dm > mean_thr || dm < -mean_thr) continue;
          }

          /* 4-plane SSD matching (RGGB-space for Step1) */
          const float d = bm3d_patch_ssd_4plane(ch_rggb, width, ref_r, ref_c, sr, sc, worst_dist);
          if(d < tau_4ch)
          {
            if(nmatches < BM3D_MAX_MATCHED)
            {
              matches[nmatches++] = (bm3d_match_t){ d, sr, sc };
              if(nmatches == BM3D_MAX_MATCHED)
              {
                worst_dist = 0;
                for(int i = 0; i < nmatches; i++)
                  if(matches[i].dist > worst_dist) worst_dist = matches[i].dist;
              }
            }
            else if(d < worst_dist)
            {
              int wi = 0;
              for(int i = 1; i < nmatches; i++) if(matches[i].dist > matches[wi].dist) wi = i;
              matches[wi] = (bm3d_match_t){ d, sr, sc };
              worst_dist = 0;
              for(int i = 0; i < nmatches; i++) if(matches[i].dist > worst_dist) worst_dist = matches[i].dist;
            }
          }
        }

      /* Sort matches + compute power-of-2 group size for Hadamard */
      sort_matches(matches, nmatches);
      const int group_size = bm3d_next_pow2(nmatches);

      /* Extract patches → 2D DCT for all 4 WHT channels */
      float dct_buf[4][BM3D_MAX_MATCHED][BM3D_PATCH_PIXELS];
      for(int c = 0; c < 4; c++)
      {
        for(int i = 0; i < nmatches; i++)
        {
          float patch[BM3D_PATCH_PIXELS];
          const int mr = matches[i].row, mc = matches[i].col;
          for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
            memcpy(patch + dy * BM3D_PATCH_SIZE,
                   ch_in[c] + (size_t)(mr + dy) * width + mc,
                   BM3D_PATCH_SIZE * sizeof(float));
          dct2d_forward(patch, dct_buf[c][i]);
        }
        for(int i = nmatches; i < group_size; i++)
          memcpy(dct_buf[c][i], dct_buf[c][0], sizeof(float) * BM3D_PATCH_PIXELS);
      }

      /* 1D Hadamard + hard thresholding (all 4 channels, shared nonzero count) */
      int n_nonzero = 0;
      for(int p = 0; p < BM3D_PATCH_PIXELS; p++)
      {
        for(int c = 0; c < 4; c++)
        {
          float had[BM3D_MAX_MATCHED];
          for(int i = 0; i < group_size; i++) had[i] = dct_buf[c][i][p];
          hadamard_transform(had, group_size);
          for(int i = 0; i < group_size; i++)
          {
            if(fabsf(had[i]) < lambda_thr)
              had[i] = 0.0f;
            else
              n_nonzero++;
          }
          hadamard_transform(had, group_size);
          for(int i = 0; i < group_size; i++) dct_buf[c][i][p] = had[i];
        }
      }

      /* Inverse DCT + aggregate with Kaiser window (shared denominator) */
      const float ht_weight = (n_nonzero > 0) ? 1.0f / (float)n_nonzero : 1.0f;
      float *my_denom = t_denom[tid];

      for(int c = 0; c < 4; c++)
      {
        float *my_numer = t_numer[c][tid];
        for(int i = 0; i < nmatches; i++)
        {
          float patch[BM3D_PATCH_PIXELS];
          dct2d_inverse(dct_buf[c][i], patch);
          const int mr = matches[i].row, mc = matches[i].col;
          for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
            for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++)
            {
              const size_t pos = (size_t)(mr + dy) * width + (mc + dx);
              const float kw = bm3d_kaiser_2d[dy * BM3D_PATCH_SIZE + dx];
              my_numer[pos] += ht_weight * kw * patch[dy * BM3D_PATCH_SIZE + dx];
              /* Shared denom: only add once (from channel 0) */
              if(c == 0) my_denom[pos] += ht_weight * kw;
            }
        }
      }
    }
  }

  /* Merge per-thread buffers and normalize — all 4 channels share denominator */
  for(int c = 0; c < 4; c++)
  {
    DT_OMP_FOR()
    for(int i = 0; i < (int)npixels; i++)
    {
      float n = 0.0f, d = 0.0f;
      for(int t = 0; t < nthreads; t++)
      { n += t_numer[c][t][i]; d += t_denom[t][i]; }
      ch_out[c][i] = (d > 0.0f) ? n / d : ch_in[c][i];
    }
  }

  /* Debug: measure pilot smoothness (RMS of luma residual) */
  {
    double rms_sum = 0.0;
    int rms_count = 0;
    for(size_t i = 0; i < npixels; i++)
    {
      const float diff = ch_out[0][i] - ch_in[0][i];
      rms_sum += (double)(diff * diff);
      rms_count++;
    }
    const double rms = sqrt(rms_sum / rms_count);
    const double neff_est = 1.0 / fmax(1.0 - rms * rms, 0.01);
    fprintf(stderr, "[rawdenoise] Step1 pilot: luma_residual_rms=%.4f neff_est=%.1f (%.0f pixels)\n",
            rms, neff_est, (double)rms_count);
  }

  for(int t = 0; t < nthreads; t++)
  {
    dt_free_align(t_denom[t]);
    for(int c = 0; c < 4; c++) dt_free_align(t_numer[c][t]);
  }
  free(t_denom);
  for(int c = 0; c < 4; c++) free(t_numer[c]);
  dt_free_align(pmeans);
}


/* ================================================================
 * BM3D Step 2 on half-res L/C planes.
 *
 * Operates directly on 4 half-res planes (L, C₁, C₂, C₃) produced
 * by WHT decomposition.  WHT 2×2 of 8×8 Bayer patch ≡ 4×4 patch on
 * L/C plane (Parseval), so matching and aggregation are natively
 * half-res with no Bayer round-trip.
 *
 * Pipeline per matched group of 4×4 L/C-plane patches:
 *   1. DCT(4×4) per plane (no WHT — planes are already in L/C basis)
 *   2. Hadamard along group axis
 *   3. Wiener: L with σ_L², C₁–C₃ with σ_C²
 *   4. Inverse Hadamard → inverse DCT
 *   5. Aggregate each plane separately in half-res
 *
 * σ_L and σ_C are independent — decoupling lets the user push chroma
 * denoising strong while keeping luma gentle (e.g. σ_L=0.5, σ_C=2.0
 * for delicate detail + strong false-chroma suppression).
 *
 * σ_L / σ_C 独立制御。ルマを軽く抑えつつクロマを強く叩く構成が
 * 直接表現可能（例: σ_L=0.5, σ_C=2.0）。
 * ================================================================ */
static void bm3d_step2_lc(
    const float *const noisy[4],   /* L, C1, C2, C3 half-res, anchor-subtracted */
    const float *const pilot[4],   /* L, C1, C2, C3 half-res, anchor-subtracted */
    float       *const output[4],  /* L, C1, C2, C3 half-res, anchor-subtracted */
    const int halfwidth, const int halfheight,
    const float sigma_L,
    const float sigma_C,
    const int ref_step)            /* half-res units (typically 2) */
{
  const float sigma_L_sq = sigma_L * sigma_L;
  const float sigma_C_sq = sigma_C * sigma_C;
  /* Matching uses σ=1 (data is GAT-normalized); σ_L/σ_C only control Wiener.
   * tau is the same value as the old full-res Step2 because Parseval
   * guarantees: sum-of-4-plane SSD on 4×4 L/C patches == 8×8 Bayer SSD. */
  const float tau = BM3D_TAU_MATCH_W;
  const size_t halfpixels = (size_t)halfwidth * halfheight;

  /* Patch positions on half-res L/C plane: 4×4 patches. */
  const int rmax = halfheight - BM3D_HALF_PATCH;
  const int cmax = halfwidth  - BM3D_HALF_PATCH;
  if(rmax < 0 || cmax < 0) return;

#ifdef _OPENMP
  const int max_threads = omp_get_max_threads();
#else
  const int max_threads = 1;
#endif
  const int nthreads = MIN(max_threads, 8);

  /* Aggregation buffers: 4 L/C planes × half-res */
  float **t_numer[4];
  float **t_denom = calloc(nthreads, sizeof(float *));

  for(int c = 0; c < 4; c++)
    t_numer[c] = calloc(nthreads, sizeof(float *));

  gboolean alloc_ok = TRUE;
  for(int t = 0; t < nthreads && alloc_ok; t++)
  {
    t_denom[t] = dt_alloc_align_float(halfpixels);
    if(!t_denom[t]) { alloc_ok = FALSE; break; }
    memset(t_denom[t], 0, sizeof(float) * halfpixels);
    for(int c = 0; c < 4; c++)
    {
      t_numer[c][t] = dt_alloc_align_float(halfpixels);
      if(!t_numer[c][t]) { alloc_ok = FALSE; break; }
      memset(t_numer[c][t], 0, sizeof(float) * halfpixels);
    }
  }
  if(!alloc_ok)
  {
    for(int t = 0; t < nthreads; t++)
    {
      dt_free_align(t_denom[t]);
      for(int c = 0; c < 4; c++) dt_free_align(t_numer[c][t]);
    }
    free(t_denom);
    for(int c = 0; c < 4; c++) free(t_numer[c]);
    /* Fallback: copy noisy → output */
    for(int c = 0; c < 4; c++)
      memcpy(output[c], noisy[c], sizeof(float) * halfpixels);
    return;
  }

  /* === DIAG-MATCH / DIAG-WIENER accumulators === */
  long long diag_total_refs = 0;
  long long diag_nmatches_sum = 0;
  int diag_nmatches_min = BM3D_MAX_MATCHED + 1;
  int diag_nmatches_max = 0;
  long long diag_group_hist[6] = {0};  /* g1, g2, g4, g8, g16, g32 */
  /* w_L / w_C histograms: 6 bins [0.03,0.1)[0.1,0.3)[0.3,0.5)[0.5,0.7)[0.7,0.9)[0.9,1.0] */
  long long diag_wL_hist[6] = {0};
  long long diag_wC_hist[6] = {0};
  double diag_wL_sum = 0.0, diag_wL_sum2 = 0.0;
  double diag_wC_sum = 0.0, diag_wC_sum2 = 0.0;
  long long diag_w_count = 0;

  const int chunk_s2 = MAX(4, (rmax + ref_step) / ref_step / (nthreads * 8));
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, chunk_s2) \
    reduction(+:diag_total_refs, diag_nmatches_sum, diag_w_count, diag_wL_sum, diag_wL_sum2, diag_wC_sum, diag_wC_sum2)
#endif
  for(int ref_r = 0; ref_r <= rmax; ref_r += ref_step)
  {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif

    for(int ref_c = 0; ref_c <= cmax; ref_c += ref_step)
    {
      /* Block matching on pilot 4-plane SSD (basis = L/C planes) */
      bm3d_match_t matches[BM3D_MAX_MATCHED];
      int nmatches = 1;
      matches[0] = (bm3d_match_t){ 0.0f, ref_r, ref_c };
      float worst_dist = tau;

      const int sr0 = MAX(0, ref_r - BM3D_SEARCH_RAD2_HALF);
      const int sr1 = MIN(rmax, ref_r + BM3D_SEARCH_RAD2_HALF);
      const int sc0 = MAX(0, ref_c - BM3D_SEARCH_RAD2_HALF);
      const int sc1 = MIN(cmax, ref_c + BM3D_SEARCH_RAD2_HALF);

      for(int sr = sr0; sr <= sr1; sr += 1)
        for(int sc = sc0; sc <= sc1; sc += 1)
        {
          if(sr == ref_r && sc == ref_c) continue;

          const float d = bm3d_patch_ssd_4plane_half(pilot, halfwidth, ref_r, ref_c, sr, sc, worst_dist);
          if(d < tau)
          {
            if(nmatches < BM3D_MAX_MATCHED)
            {
              matches[nmatches++] = (bm3d_match_t){ d, sr, sc };
              if(nmatches == BM3D_MAX_MATCHED)
              {
                worst_dist = 0;
                for(int i = 0; i < nmatches; i++)
                  if(matches[i].dist > worst_dist) worst_dist = matches[i].dist;
              }
            }
            else if(d < worst_dist)
            {
              int wi = 0;
              for(int i = 1; i < nmatches; i++) if(matches[i].dist > matches[wi].dist) wi = i;
              matches[wi] = (bm3d_match_t){ d, sr, sc };
              worst_dist = 0;
              for(int i = 0; i < nmatches; i++) if(matches[i].dist > worst_dist) worst_dist = matches[i].dist;
            }
          }
        }

      sort_matches(matches, nmatches);
      const int group_size = bm3d_next_pow2(nmatches);

      /* DIAG-MATCH stats */
      diag_total_refs++;
      diag_nmatches_sum += nmatches;
      if(nmatches < diag_nmatches_min) diag_nmatches_min = nmatches;
      if(nmatches > diag_nmatches_max) diag_nmatches_max = nmatches;
      {
        int gi = 0;
        int gs = group_size;
        while(gs > 1 && gi < 5) { gs >>= 1; gi++; }
#ifdef _OPENMP
#pragma omp atomic
#endif
        diag_group_hist[gi]++;
      }

      /* Wiener noise variance: no local variance map.
       * RMS unified σ guarantees Var[L]=Var[Cₖ]=1 (see highlight (b)),
       * so σ_L / σ_C are used directly with no spatial scaling.
       * varmap なし — RMS 統一 σ で Var[L/Cₖ]=1 厳密。 */
      const float local_sigma_L_sq = fmaxf(sigma_L_sq, 1e-6f);
      const float local_sigma_C_sq = fmaxf(sigma_C_sq, 1e-6f);

      /* Phase 1: Extract 4×4 L/C patches → DCT(4×4) for pilot and noisy.
       * No WHT needed — planes are already in L/C basis. */
      float all_pilot_dct[4][BM3D_MAX_MATCHED][BM3D_HALF_PIXELS];
      float all_noisy_dct[4][BM3D_MAX_MATCHED][BM3D_HALF_PIXELS];

      for(int m = 0; m < nmatches; m++)
      {
        float patch_buf[BM3D_HALF_PIXELS];
        const int mr = matches[m].row, mc = matches[m].col;
        for(int c = 0; c < 4; c++)
        {
          /* Pilot patch (4×4) → DCT */
          for(int dy = 0; dy < BM3D_HALF_PATCH; dy++)
            memcpy(patch_buf + dy * BM3D_HALF_PATCH,
                   pilot[c] + (size_t)(mr + dy) * halfwidth + mc,
                   BM3D_HALF_PATCH * sizeof(float));
          dct2d_4x4_forward(patch_buf, all_pilot_dct[c][m]);

          /* Noisy patch (4×4) → DCT */
          for(int dy = 0; dy < BM3D_HALF_PATCH; dy++)
            memcpy(patch_buf + dy * BM3D_HALF_PATCH,
                   noisy[c] + (size_t)(mr + dy) * halfwidth + mc,
                   BM3D_HALF_PATCH * sizeof(float));
          dct2d_4x4_forward(patch_buf, all_noisy_dct[c][m]);
        }
      }
      /* Copy-pad to power-of-2 group size */
      for(int m = nmatches; m < group_size; m++)
        for(int c = 0; c < 4; c++)
        {
          memcpy(all_pilot_dct[c][m], all_pilot_dct[c][0], sizeof(float) * BM3D_HALF_PIXELS);
          memcpy(all_noisy_dct[c][m], all_noisy_dct[c][0], sizeof(float) * BM3D_HALF_PIXELS);
        }

      /* ==============================================================
       * Phase 2: 3D Hadamard + L/C split Wiener (B3 + B4)
       *
       * Separate gains for luma (c=0) and chroma (c=1,2,3). The c=1,2,3
       * planes share a single w_C to preserve RGGB balance in the
       * caller's inverse WHT (Var[L]=Var[Cₖ] under unified σ).
       *
       * --------------------------------------------------------------
       * B4: textbook BM3D empirical Wiener
       *
       *   w = |Y_p|² / (|Y_p|² + σ²)
       *
       * where Y_p is the pilot DCT-Hadamard coefficient. This is the
       * standard Step2 form from Dabov et al. 2007. The previous code
       * used a debiased variant `max(s²-σ², 0) / (max(...) + σ²)` which
       * over-corrected for pilot bias (it assumed the pilot still had
       * full noise variance σ², but Step1 has already reduced it).
       * That over-correction created a flat-region pile-up at the gain
       * floor (a spurious [0.03, 0.1) spike in the w_L histogram, ~33%
       * of bins) and bimodal w_L distribution.
       *
       * The textbook form has an inherent slight upward bias because
       * E[|Y_p|²] = signal² + σ_pilot² with σ_pilot² < σ², so w is
       * pulled toward 1 a bit more than the true Wiener — but this is
       * harmless: extra preservation, never extra suppression. The
       * gain floor (0.03) is retained for aggregation-weight stability.
       *
       * B4: 教科書通り BM3D 経験 Wiener
       *   w = |Y_p|² / (|Y_p|² + σ²)
       * 旧コードの `max(s²-σ², 0)` 形式は pilot にフル σ² ノイズが
       * 残る前提の過剰補正で、flat 領域で w が floor に張り付く挙動
       * （ヒストグラム [0.03, 0.1) に 33% 集中）を生んでいた。
       * 本式は Step1 でノイズ低減済みの pilot を pilot² 自身で評価
       * するため上方バイアスはあるが「保存しすぎ」方向で安全。
       *
       * --------------------------------------------------------------
       * B3: DC bin protection
       *
       * The (p=0, i=0) coefficient — 4×4 DCT DC × group-Hadamard DC —
       * encodes the average patch mean across the matched group. It
       * already enjoys √(group_size × 16) noise reduction from the
       * combined averaging, so additional Wiener filtering yields
       * negligible SNR gain while injecting per-channel DC drift
       * (observed +5.6% B/G shift without DC protection). We force
       * w_L = w_C = 1 at this bin to preserve DC exactly.
       *
       * Theoretical alignment with Phase 2 dark anchor: the dark anchor
       * promises "DC = signal, Wiener = AC only". B3 makes the Wiener
       * step honor that contract for the (DC, DC) bin. Per-channel DC
       * offsets retained by the dark anchor pass through Step2 intact
       * and re-emerge after Phase 6's anchor restoration.
       *
       * Note: only the (p=0, i=0) bin is fully protected. Other DCT
       * bins at i=0 (Hadamard DC across matches) and i>0 at p=0 still
       * carry meaningful AC information (spatial AC averaged over
       * matches, or DC-difference between matches) and remain filtered.
       *
       * B3: DC bin 保護
       * (p=0, i=0) 係数（4×4 DCT の DC × group Hadamard の DC）は
       * マッチ群の平均パッチ平均を表す。√(group_size × 16) の平均化
       * ノイズ削減が既に効いており、追加 Wiener の SNR 利得は無視
       * できる一方、per-ch DC ドリフトを注入する。w_L=w_C=1 で固定。
       * Phase 2 ダークアンカーが per-ch DC を退避→Phase 6 で復元する
       * 設計と整合し、Wiener は AC のみ動かす契約を完成させる。
       * ============================================================== */
      float wiener_energy = 0.0f;

      for(int p = 0; p < BM3D_HALF_PIXELS; p++)
      {
        float had_pilot[4][BM3D_MAX_MATCHED], had_noisy[4][BM3D_MAX_MATCHED];
        for(int c = 0; c < 4; c++)
        {
          for(int i = 0; i < group_size; i++) had_pilot[c][i] = all_pilot_dct[c][i][p];
          for(int i = 0; i < group_size; i++) had_noisy[c][i] = all_noisy_dct[c][i][p];
          hadamard_transform(had_pilot[c], group_size);
          hadamard_transform(had_noisy[c], group_size);
        }

        for(int i = 0; i < group_size; i++)
        {
          /* B3 DC-protect mask: identity gain at the (DC-DCT, DC-Hadamard) bin */
          const int is_dc_bin = (p == 0 && i == 0);

          /* Luma Wiener (c=0): textbook empirical w_L = s²/(s²+σ²) */
          const float s2_L = had_pilot[0][i] * had_pilot[0][i];
          const float w_L = is_dc_bin
              ? 1.0f
              : fmaxf(s2_L / (s2_L + local_sigma_L_sq + 1e-10f), 0.03f);

          /* Chroma Wiener (c=1,2,3): pooled across 3 chroma planes,
           * single w_C maintains RGGB balance under inverse WHT.
           *   w_C = (Σₖ |Yₖ|²) / (Σₖ |Yₖ|² + 3σ²)
           * The factor 3 accounts for 3 independent chroma planes
           * each carrying variance σ². */
          float s2_C = 0.0f;
          for(int c = 1; c < 4; c++)
          {
            const float v = had_pilot[c][i];
            s2_C += v * v;
          }
          const float w_C = is_dc_bin
              ? 1.0f
              : fmaxf(s2_C / (s2_C + 3.0f * local_sigma_C_sq + 1e-10f), 0.03f);

          had_noisy[0][i] *= w_L;
          for(int c = 1; c < 4; c++)
            had_noisy[c][i] *= w_C;
          wiener_energy += w_L * w_L + w_C * w_C;

          /* DIAG-WIENER accumulation */
          diag_wL_sum  += w_L;       diag_wL_sum2 += w_L * w_L;
          diag_wC_sum  += w_C;       diag_wC_sum2 += w_C * w_C;
          diag_w_count++;
          {
            int bL = (w_L < 0.1f) ? 0 : (w_L < 0.3f) ? 1 : (w_L < 0.5f) ? 2
                   : (w_L < 0.7f) ? 3 : (w_L < 0.9f) ? 4 : 5;
            int bC = (w_C < 0.1f) ? 0 : (w_C < 0.3f) ? 1 : (w_C < 0.5f) ? 2
                   : (w_C < 0.7f) ? 3 : (w_C < 0.9f) ? 4 : 5;
#ifdef _OPENMP
#pragma omp atomic
#endif
            diag_wL_hist[bL]++;
#ifdef _OPENMP
#pragma omp atomic
#endif
            diag_wC_hist[bC]++;
          }
        }

        for(int c = 0; c < 4; c++)
        {
          hadamard_transform(had_noisy[c], group_size);
          for(int i = 0; i < group_size; i++) all_noisy_dct[c][i][p] = had_noisy[c][i];
        }
      }

      /* Phase 3: Inverse DCT → aggregate directly into half-res L/C planes.
       * Patch position is already in half-res coordinates (mr, mc). */
      const float shared_weight = 1.0f / fmaxf(wiener_energy, 1e-6f);

      for(int m = 0; m < nmatches; m++)
      {
        float lc_buf[4][BM3D_HALF_PIXELS];
        for(int c = 0; c < 4; c++)
          dct2d_4x4_inverse(all_noisy_dct[c][m], lc_buf[c]);

        const int mr = matches[m].row, mc = matches[m].col;

        for(int c = 0; c < 4; c++)
        {
          float *my_numer_c = t_numer[c][tid];
          float *my_denom_c = (c == 0) ? t_denom[tid] : NULL; /* shared denominator */
          for(int dy = 0; dy < BM3D_HALF_PATCH; dy++)
            for(int dx = 0; dx < BM3D_HALF_PATCH; dx++)
            {
              const int hy = mr + dy, hx = mc + dx;
              if(hy >= halfheight || hx >= halfwidth) continue;
              const size_t hpos = (size_t)hy * halfwidth + hx;
              /* Kaiser window: symmetric 4×4 (pair-averaged from 8-point) */
              const float kw = bm3d_kaiser_half_2d[dy * BM3D_HALF_PATCH + dx];
              my_numer_c[hpos] += shared_weight * kw * lc_buf[c][dy * BM3D_HALF_PATCH + dx];
              if(c == 0) my_denom_c[hpos] += shared_weight * kw;
            }
        }
      }
    } /* end ref_c */
  } /* end ref_r */

  /* Merge per-thread buffers → output[c] (half-res, mean-subtracted).
   * Edges with denom==0 fall back to noisy[c]. */
  DT_OMP_FOR()
  for(int i = 0; i < (int)halfpixels; i++)
  {
    float d = 0.0f;
    for(int t = 0; t < nthreads; t++)
      d += t_denom[t][i];
    if(d > 0.0f)
    {
      const float inv_d = 1.0f / d;
      for(int c = 0; c < 4; c++)
      {
        float n = 0.0f;
        for(int t = 0; t < nthreads; t++)
          n += t_numer[c][t][i];
        output[c][i] = n * inv_d;
      }
    }
    else
    {
      for(int c = 0; c < 4; c++)
        output[c][i] = noisy[c][i];
    }
  }

  /* === DIAG-MATCH / DIAG-WIENER / DIAG-STEP2-RESID emit === */
  if(diag_total_refs > 0)
  {
    fprintf(stderr, "[rawdenoise] DIAG-MATCH: refs=%lld nmatches: avg=%.1f min=%d max=%d\n",
            diag_total_refs,
            (double)diag_nmatches_sum / (double)diag_total_refs,
            diag_nmatches_min, diag_nmatches_max);
    fprintf(stderr, "[rawdenoise]   group_size: g1=%lld g2=%lld g4=%lld g8=%lld g16=%lld g32=%lld\n",
            diag_group_hist[0], diag_group_hist[1], diag_group_hist[2],
            diag_group_hist[3], diag_group_hist[4], diag_group_hist[5]);
  }
  if(diag_w_count > 0)
  {
    const double mL = diag_wL_sum / diag_w_count;
    const double vL = diag_wL_sum2 / diag_w_count - mL * mL;
    const double mC = diag_wC_sum / diag_w_count;
    const double vC = diag_wC_sum2 / diag_w_count - mC * mC;
    fprintf(stderr, "[rawdenoise] DIAG-WIENER: w_L mean=%.3f std=%.3f  "
                    "w_C mean=%.3f std=%.3f  (n=%lld bins)\n",
            mL, (vL > 0.0) ? sqrt(vL) : 0.0,
            mC, (vC > 0.0) ? sqrt(vC) : 0.0, diag_w_count);
    const double inv = 100.0 / (double)diag_w_count;
    fprintf(stderr, "[rawdenoise]   w_L hist (%%): [.03,.1)=%.1f [.1,.3)=%.1f [.3,.5)=%.1f "
                    "[.5,.7)=%.1f [.7,.9)=%.1f [.9,1.]=%.1f\n",
            diag_wL_hist[0]*inv, diag_wL_hist[1]*inv, diag_wL_hist[2]*inv,
            diag_wL_hist[3]*inv, diag_wL_hist[4]*inv, diag_wL_hist[5]*inv);
    fprintf(stderr, "[rawdenoise]   w_C hist (%%): [.03,.1)=%.1f [.1,.3)=%.1f [.3,.5)=%.1f "
                    "[.5,.7)=%.1f [.7,.9)=%.1f [.9,1.]=%.1f\n",
            diag_wC_hist[0]*inv, diag_wC_hist[1]*inv, diag_wC_hist[2]*inv,
            diag_wC_hist[3]*inv, diag_wC_hist[4]*inv, diag_wC_hist[5]*inv);
  }
  {
    const char *cname[4] = { "L ", "C1", "C2", "C3" };
    for(int c = 0; c < 4; c++)
    {
      double rms_sum = 0.0;
      for(size_t i = 0; i < halfpixels; i++)
      {
        const float diff = output[c][i] - noisy[c][i];
        rms_sum += (double)(diff * diff);
      }
      const double rms = sqrt(rms_sum / (double)halfpixels);
      fprintf(stderr, "[rawdenoise] DIAG-STEP2-RESID: %s plane rms=%.4f\n", cname[c], rms);
    }
  }

  for(int t = 0; t < nthreads; t++)
  {
    dt_free_align(t_denom[t]);
    for(int c = 0; c < 4; c++) dt_free_align(t_numer[c][t]);
  }
  free(t_denom);
  for(int c = 0; c < 4; c++) free(t_numer[c]);
}


/* ================================================================
 *  Section 4: BM3D-CFA
 *  セクション4: BM3D-CFA
 *
 *  Standard BM3D applied directly to the full-resolution CFA mosaic
 *  with CFA-phase-aware block matching (stride-2 search).
 *  CFA位相整合ブロックマッチング（stride-2探索）による
 *  フル解像度CFAモザイク上の標準BM3D。
 *
 *  Step 1: grouping + 2D-DCT + 1D-Hadamard + hard thresholding
 *          + inverse transforms + weighted aggregation (1/N_nonzero)
 *  Step 2: grouping on Step1 pilot + 2D-DCT + 1D-Hadamard
 *          + Wiener shrinkage + inverse + aggregation (1/wiener_energy)
 *
 *  Ref: Danielyan, Foi, Katkovnik, "BM3D frames and variational
 *       image deblurring and denoising" (TIP 2012)
 *
 *  GAT (forward/inverse) and noise estimation are shared with other methods.
 * ================================================================ */

/* BM3D-CFA Step 1: Hard thresholding collaborative filter
 * Data is σ-normalized (σ=1), so matching uses σ=1 fixed.
 * strength controls only the hard threshold: λ = 2.7 * strength */
static void bm3d_cfa_step1(const float *restrict noisy, float *restrict output,
                            const int width, const int height, const float strength)
{
  const float tau = BM3D_TAU_MATCH;  /* σ=1 normalized → tau is fixed */
  const float lambda_thr = BM3D_LAMBDA_3D * strength;
  const size_t npixels = (size_t)width * height;

  const int step = 4;  /* must be even for CFA alignment */
  const int rmax = height - BM3D_PATCH_SIZE, cmax = width - BM3D_PATCH_SIZE;
  const int mwidth = cmax + 1;

  float *pmeans = bm3d_precompute_patch_means(noisy, width, height);
  const float mean_thr = sqrtf(tau / (float)BM3D_PATCH_PIXELS);

#ifdef _OPENMP
  const int max_threads = omp_get_max_threads();
#else
  const int max_threads = 1;
#endif
  const int nthreads = MIN(max_threads, 8);

  float **t_numer = calloc(nthreads, sizeof(float *));
  float **t_denom = calloc(nthreads, sizeof(float *));

  gboolean alloc_ok = TRUE;
  for(int t = 0; t < nthreads && alloc_ok; t++)
  {
    t_numer[t] = dt_alloc_align_float(npixels);
    t_denom[t] = dt_alloc_align_float(npixels);
    if(!t_numer[t] || !t_denom[t]) { alloc_ok = FALSE; break; }
    memset(t_numer[t], 0, sizeof(float) * npixels);
    memset(t_denom[t], 0, sizeof(float) * npixels);
  }

  if(!alloc_ok)
  {
    for(int t = 0; t < nthreads; t++)
    { dt_free_align(t_numer[t]); dt_free_align(t_denom[t]); }
    free(t_numer); free(t_denom);
    dt_free_align(pmeans);
    memcpy(output, noisy, sizeof(float) * npixels);
    return;
  }

  const int chunk_cfa = MAX(4, (rmax + step) / step / (nthreads * 8));
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, chunk_cfa)
#endif
  for(int ref_r = 0; ref_r <= rmax; ref_r += step)
  {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif

    for(int ref_c = 0; ref_c <= cmax; ref_c += step)
    {
      /* Block matching with CFA phase alignment */
      bm3d_match_t matches[BM3D_MAX_MATCHED];
      int nmatches = 1;
      matches[0] = (bm3d_match_t){ 0.0f, ref_r, ref_c };
      float worst_dist = tau;

      const float ref_mean = pmeans ? pmeans[ref_r * mwidth + ref_c] : 0.0f;

      int sr0 = MAX(0, ref_r - BM3D_SEARCH_RAD);
      if(((sr0 ^ ref_r) & 1)) sr0++;
      const int sr1 = MIN(rmax, ref_r + BM3D_SEARCH_RAD);
      int sc0 = MAX(0, ref_c - BM3D_SEARCH_RAD);
      if(((sc0 ^ ref_c) & 1)) sc0++;
      const int sc1 = MIN(cmax, ref_c + BM3D_SEARCH_RAD);

      for(int sr = sr0; sr <= sr1; sr += 2)
        for(int sc = sc0; sc <= sc1; sc += 2)
        {
          if(sr == ref_r && sc == ref_c) continue;

          if(pmeans)
          {
            const float dm = ref_mean - pmeans[sr * mwidth + sc];
            if(dm > mean_thr || dm < -mean_thr) continue;
          }

          const float d = bm3d_patch_ssd(noisy, width, ref_r, ref_c, sr, sc, worst_dist);
          if(d < tau)
          {
            if(nmatches < BM3D_MAX_MATCHED)
            {
              matches[nmatches++] = (bm3d_match_t){ d, sr, sc };
              if(nmatches == BM3D_MAX_MATCHED)
              {
                worst_dist = 0;
                for(int i = 0; i < nmatches; i++)
                  if(matches[i].dist > worst_dist) worst_dist = matches[i].dist;
              }
            }
            else if(d < worst_dist)
            {
              int wi = 0;
              for(int i = 1; i < nmatches; i++) if(matches[i].dist > matches[wi].dist) wi = i;
              matches[wi] = (bm3d_match_t){ d, sr, sc };
              worst_dist = 0;
              for(int i = 0; i < nmatches; i++) if(matches[i].dist > worst_dist) worst_dist = matches[i].dist;
            }
          }
        }

      sort_matches(matches, nmatches);
      const int group_size = bm3d_next_pow2(nmatches);

      /* Extract patches → 2D DCT */
      float dct_buf[BM3D_MAX_MATCHED][BM3D_PATCH_PIXELS];
      for(int m = 0; m < nmatches; m++)
      {
        float patch[BM3D_PATCH_PIXELS];
        for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
          memcpy(patch + dy * BM3D_PATCH_SIZE,
                 noisy + (size_t)(matches[m].row + dy) * width + matches[m].col,
                 BM3D_PATCH_SIZE * sizeof(float));
        dct2d_forward(patch, dct_buf[m]);
      }
      for(int m = nmatches; m < group_size; m++)
        memcpy(dct_buf[m], dct_buf[0], sizeof(float) * BM3D_PATCH_PIXELS);

      /* 1D Hadamard along group axis + hard thresholding */
      int n_nonzero = 0;
      for(int p = 0; p < BM3D_PATCH_PIXELS; p++)
      {
        float had[BM3D_MAX_MATCHED];
        for(int i = 0; i < group_size; i++) had[i] = dct_buf[i][p];
        hadamard_transform(had, group_size);

        for(int i = 0; i < group_size; i++)
        {
          if(fabsf(had[i]) < lambda_thr)
            had[i] = 0.0f;
          else
            n_nonzero++;
        }

        hadamard_transform(had, group_size);
        for(int i = 0; i < group_size; i++) dct_buf[i][p] = had[i];
      }

      /* Inverse DCT → aggregate with weight = 1/N_nonzero */
      const float ht_weight = (n_nonzero > 0) ? 1.0f / (float)n_nonzero : 1.0f;
      float *my_numer = t_numer[tid];
      float *my_denom = t_denom[tid];

      for(int m = 0; m < nmatches; m++)
      {
        float patch[BM3D_PATCH_PIXELS];
        dct2d_inverse(dct_buf[m], patch);

        const int mr = matches[m].row, mc = matches[m].col;
        for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
          for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++)
          {
            const size_t pos = (size_t)(mr + dy) * width + (mc + dx);
            const float kw = bm3d_kaiser_2d[dy * BM3D_PATCH_SIZE + dx];
            my_numer[pos] += ht_weight * kw * patch[dy * BM3D_PATCH_SIZE + dx];
            my_denom[pos] += ht_weight * kw;
          }
      }
    }
  }

  /* Merge per-thread buffers */
  DT_OMP_FOR()
  for(int i = 0; i < (int)npixels; i++)
  {
    float n = 0.0f, d = 0.0f;
    for(int t = 0; t < nthreads; t++)
    { n += t_numer[t][i]; d += t_denom[t][i]; }
    output[i] = (d > 0.0f) ? n / d : noisy[i];
  }

  for(int t = 0; t < nthreads; t++)
  { dt_free_align(t_numer[t]); dt_free_align(t_denom[t]); }
  free(t_numer); free(t_denom);
  dt_free_align(pmeans);
}

/* BM3D-CFA Step 2: Wiener collaborative filter
 * Data is σ-normalized (σ=1), so matching uses σ=1 fixed.
 * strength controls Wiener shrinkage: σ² = strength² */
static void bm3d_cfa_step2(const float *restrict noisy, const float *restrict pilot,
                            float *restrict output, const int width, const int height,
                            const float strength)
{
  const float sigma_sq = strength * strength;
  const float tau = BM3D_TAU_MATCH_W;  /* σ=1 normalized → tau is fixed */
  const size_t npixels = (size_t)width * height;

  const int step = 4;  /* must be even for CFA alignment */
  const int rmax = height - BM3D_PATCH_SIZE, cmax = width - BM3D_PATCH_SIZE;
  const int mwidth = cmax + 1;

  float *pmeans = bm3d_precompute_patch_means(pilot, width, height);
  const float mean_thr = sqrtf(tau / (float)BM3D_PATCH_PIXELS);

#ifdef _OPENMP
  const int max_threads = omp_get_max_threads();
#else
  const int max_threads = 1;
#endif
  const int nthreads = MIN(max_threads, 8);

  float **t_numer = calloc(nthreads, sizeof(float *));
  float **t_denom = calloc(nthreads, sizeof(float *));

  gboolean alloc_ok = TRUE;
  for(int t = 0; t < nthreads && alloc_ok; t++)
  {
    t_numer[t] = dt_alloc_align_float(npixels);
    t_denom[t] = dt_alloc_align_float(npixels);
    if(!t_numer[t] || !t_denom[t]) { alloc_ok = FALSE; break; }
    memset(t_numer[t], 0, sizeof(float) * npixels);
    memset(t_denom[t], 0, sizeof(float) * npixels);
  }
  if(!alloc_ok)
  {
    for(int t = 0; t < nthreads; t++)
    { dt_free_align(t_numer[t]); dt_free_align(t_denom[t]); }
    free(t_numer); free(t_denom);
    dt_free_align(pmeans);
    memcpy(output, noisy, sizeof(float) * npixels);
    return;
  }

  const int chunk_cfa2 = MAX(4, (rmax + step) / step / (nthreads * 8));
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, chunk_cfa2)
#endif
  for(int ref_r = 0; ref_r <= rmax; ref_r += step)
  {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif

    for(int ref_c = 0; ref_c <= cmax; ref_c += step)
    {
      /* Block matching on pilot with CFA phase alignment */
      bm3d_match_t matches[BM3D_MAX_MATCHED];
      int nmatches = 1;
      matches[0] = (bm3d_match_t){ 0.0f, ref_r, ref_c };
      float worst_dist = tau;

      const float ref_mean = pmeans ? pmeans[ref_r * mwidth + ref_c] : 0.0f;

      int sr0 = MAX(0, ref_r - BM3D_SEARCH_RAD);
      if(((sr0 ^ ref_r) & 1)) sr0++;
      const int sr1 = MIN(rmax, ref_r + BM3D_SEARCH_RAD);
      int sc0 = MAX(0, ref_c - BM3D_SEARCH_RAD);
      if(((sc0 ^ ref_c) & 1)) sc0++;
      const int sc1 = MIN(cmax, ref_c + BM3D_SEARCH_RAD);

      for(int sr = sr0; sr <= sr1; sr += 2)
        for(int sc = sc0; sc <= sc1; sc += 2)
        {
          if(sr == ref_r && sc == ref_c) continue;

          if(pmeans)
          {
            const float dm = ref_mean - pmeans[sr * mwidth + sc];
            if(dm > mean_thr || dm < -mean_thr) continue;
          }

          const float d = bm3d_patch_ssd(pilot, width, ref_r, ref_c, sr, sc, worst_dist);
          if(d < tau)
          {
            if(nmatches < BM3D_MAX_MATCHED)
            {
              matches[nmatches++] = (bm3d_match_t){ d, sr, sc };
              if(nmatches == BM3D_MAX_MATCHED)
              {
                worst_dist = 0;
                for(int i = 0; i < nmatches; i++)
                  if(matches[i].dist > worst_dist) worst_dist = matches[i].dist;
              }
            }
            else if(d < worst_dist)
            {
              int wi = 0;
              for(int i = 1; i < nmatches; i++) if(matches[i].dist > matches[wi].dist) wi = i;
              matches[wi] = (bm3d_match_t){ d, sr, sc };
              worst_dist = 0;
              for(int i = 0; i < nmatches; i++) if(matches[i].dist > worst_dist) worst_dist = matches[i].dist;
            }
          }
        }

      sort_matches(matches, nmatches);
      const int group_size = bm3d_next_pow2(nmatches);

      /* No varmap — use global sigma_sq directly */
      const float local_sigma_sq = sigma_sq;

      /* Extract patches → 2D DCT for both pilot and noisy */
      float pilot_dct[BM3D_MAX_MATCHED][BM3D_PATCH_PIXELS];
      float noisy_dct[BM3D_MAX_MATCHED][BM3D_PATCH_PIXELS];

      for(int m = 0; m < nmatches; m++)
      {
        float patch[BM3D_PATCH_PIXELS];

        for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
          memcpy(patch + dy * BM3D_PATCH_SIZE,
                 pilot + (size_t)(matches[m].row + dy) * width + matches[m].col,
                 BM3D_PATCH_SIZE * sizeof(float));
        dct2d_forward(patch, pilot_dct[m]);

        for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
          memcpy(patch + dy * BM3D_PATCH_SIZE,
                 noisy + (size_t)(matches[m].row + dy) * width + matches[m].col,
                 BM3D_PATCH_SIZE * sizeof(float));
        dct2d_forward(patch, noisy_dct[m]);
      }
      for(int m = nmatches; m < group_size; m++)
      {
        memcpy(pilot_dct[m], pilot_dct[0], sizeof(float) * BM3D_PATCH_PIXELS);
        memcpy(noisy_dct[m], noisy_dct[0], sizeof(float) * BM3D_PATCH_PIXELS);
      }

      /* 1D Hadamard + Wiener shrinkage */
      float wiener_energy = 0.0f;
      for(int p = 0; p < BM3D_PATCH_PIXELS; p++)
      {
        float had_pilot[BM3D_MAX_MATCHED], had_noisy[BM3D_MAX_MATCHED];
        for(int i = 0; i < group_size; i++) had_pilot[i] = pilot_dct[i][p];
        for(int i = 0; i < group_size; i++) had_noisy[i] = noisy_dct[i][p];
        hadamard_transform(had_pilot, group_size);
        hadamard_transform(had_noisy, group_size);

        for(int i = 0; i < group_size; i++)
        {
          const float s2 = had_pilot[i] * had_pilot[i];
          const float signal_est = fmaxf(s2 - local_sigma_sq, 0.0f);
          const float w = fmaxf(signal_est / (signal_est + local_sigma_sq + 1e-10f), 0.03f);
          had_noisy[i] *= w;
          wiener_energy += w * w;
        }

        hadamard_transform(had_noisy, group_size);
        for(int i = 0; i < group_size; i++) noisy_dct[i][p] = had_noisy[i];
      }

      /* Inverse DCT → aggregate */
      const float wie_weight = 1.0f / fmaxf(wiener_energy, 1e-6f);
      float *my_numer = t_numer[tid];
      float *my_denom = t_denom[tid];

      for(int m = 0; m < nmatches; m++)
      {
        float patch[BM3D_PATCH_PIXELS];
        dct2d_inverse(noisy_dct[m], patch);

        const int mr = matches[m].row, mc = matches[m].col;
        for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
          for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++)
          {
            const size_t pos = (size_t)(mr + dy) * width + (mc + dx);
            const float kw = bm3d_kaiser_2d[dy * BM3D_PATCH_SIZE + dx];
            my_numer[pos] += wie_weight * kw * patch[dy * BM3D_PATCH_SIZE + dx];
            my_denom[pos] += wie_weight * kw;
          }
      }
    }
  }

  /* Merge per-thread buffers */
  DT_OMP_FOR()
  for(int i = 0; i < (int)npixels; i++)
  {
    float n = 0.0f, d = 0.0f;
    for(int t = 0; t < nthreads; t++)
    { n += t_numer[t][i]; d += t_denom[t][i]; }
    output[i] = (d > 0.0f) ? n / d : noisy[i];
  }

  for(int t = 0; t < nthreads; t++)
  { dt_free_align(t_numer[t]); dt_free_align(t_denom[t]); }
  free(t_numer); free(t_denom);
  dt_free_align(pmeans);
}

/* ================================================================
 * Full-resolution luma computation and BM3D Step2.
 *
 * Half-res L/C BM3D produces one L value per 2×2 Bayer block.
 * Under strong Wiener this creates a visible 2×2 plateau artifact.
 * To eliminate it we compute L at every raw pixel position using
 * a sliding 2×2 WHT-DC, then denoise L_fullres with a dedicated
 * single-channel BM3D Step2 at full resolution.
 *
 * Derivation — inv-WHT of sliding 2×2 blocks yields:
 *   L_fullres(2hy,   2hx)   = L(hy,hx)
 *   L_fullres(2hy,   2hx+1) = [(L−C₂)@(hy,hx) + (L+C₂)@(hy,hx+1)] / 2
 *   L_fullres(2hy+1, 2hx)   = [(L−C₁)@(hy,hx) + (L+C₁)@(hy+1,hx)] / 2
 *   L_fullres(2hy+1, 2hx+1) = Σ [(L ± C₁ ± C₂ ± C₃) at 4 blocks] / 4
 *
 * Noise properties: each L_fullres sample is a sum of 4 raw values / 2,
 * so Var[L_fullres] = 1 (same as half-res L under RMS normalization).
 * Adjacent samples share 2 of 4 raw values → ρ ≈ 0.5 spatial
 * correlation, but BM3D is robust to moderate correlation.
 *
 * full-res L 計算 + BM3D Step2。half-res L は 2×2 block あたり 1 値
 * のため強 Wiener 時に plateau が出る。sliding WHT-DC で全 raw pixel
 * 位置の L を計算し、full-res BM3D Step2 で edge-aware デノイズする。
 * ================================================================ */

/* Compute L_fullres from half-res L/C planes.
 * Output: L_out of size (2*halfwidth) × (2*halfheight) = full-res. */
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

      /* (2hy, 2hx+1): horizontal sliding —
       *   = [(L − C2)@here + (L + C2)@right] / 2 */
      L_out[(size_t)fr * fw + fc + 1]
        = ((L[hi] - C2[hi]) + (L[hi_r] + C2[hi_r])) * 0.5f;

      /* (2hy+1, 2hx): vertical sliding —
       *   = [(L − C1)@here + (L + C1)@below] / 2 */
      L_out[(size_t)(fr + 1) * fw + fc]
        = ((L[hi] - C1[hi]) + (L[hi_d] + C1[hi_d])) * 0.5f;

      /* (2hy+1, 2hx+1): diagonal sliding —
       * Full 4-block formula with C gradient terms */
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


/* BM3D Step2 Wiener on single-channel full-res L plane.
 * Based on bm3d_cfa_step2 but without CFA parity constraints.
 * Uses textbook Wiener w = s²/(s²+σ²) with DC bin protection. */
static void bm3d_step2_fullres_L(const float *restrict noisy,
                                  const float *restrict pilot,
                                  float *restrict output,
                                  const int width, const int height,
                                  const float sigma_L,
                                  const int ref_step)
{
  const float sigma_sq = sigma_L * sigma_L;
  const float tau = BM3D_TAU_MATCH_W;
  const size_t npixels = (size_t)width * height;

  const int rmax = height - BM3D_PATCH_SIZE;
  const int cmax = width  - BM3D_PATCH_SIZE;
  const int mwidth = cmax + 1;

  float *pmeans = bm3d_precompute_patch_means(pilot, width, height);
  const float mean_thr = sqrtf(tau / (float)BM3D_PATCH_PIXELS);

#ifdef _OPENMP
  const int max_threads = omp_get_max_threads();
#else
  const int max_threads = 1;
#endif
  const int nthreads = MIN(max_threads, 8);

  float **t_numer = calloc(nthreads, sizeof(float *));
  float **t_denom = calloc(nthreads, sizeof(float *));

  gboolean alloc_ok = TRUE;
  for(int t = 0; t < nthreads && alloc_ok; t++)
  {
    t_numer[t] = dt_alloc_align_float(npixels);
    t_denom[t] = dt_alloc_align_float(npixels);
    if(!t_numer[t] || !t_denom[t]) { alloc_ok = FALSE; break; }
    memset(t_numer[t], 0, sizeof(float) * npixels);
    memset(t_denom[t], 0, sizeof(float) * npixels);
  }
  if(!alloc_ok)
  {
    for(int t = 0; t < nthreads; t++)
    { dt_free_align(t_numer[t]); dt_free_align(t_denom[t]); }
    free(t_numer); free(t_denom);
    dt_free_align(pmeans);
    memcpy(output, noisy, sizeof(float) * npixels);
    return;
  }

  const int chunk = MAX(4, (rmax + ref_step) / ref_step / (nthreads * 8));
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, chunk)
#endif
  for(int ref_r = 0; ref_r <= rmax; ref_r += ref_step)
  {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif

    for(int ref_c = 0; ref_c <= cmax; ref_c += ref_step)
    {
      /* Block matching on pilot — no CFA parity constraint */
      bm3d_match_t matches[BM3D_MAX_MATCHED];
      int nmatches = 1;
      matches[0] = (bm3d_match_t){ 0.0f, ref_r, ref_c };
      float worst_dist = tau;

      const float ref_mean = pmeans ? pmeans[ref_r * mwidth + ref_c] : 0.0f;

      const int sr0 = MAX(0, ref_r - BM3D_SEARCH_RAD2);
      const int sr1 = MIN(rmax, ref_r + BM3D_SEARCH_RAD2);
      const int sc0 = MAX(0, ref_c - BM3D_SEARCH_RAD2);
      const int sc1 = MIN(cmax, ref_c + BM3D_SEARCH_RAD2);

      for(int sr = sr0; sr <= sr1; sr += 2)    /* step 2: L is smooth, fine matches are nearby */
        for(int sc = sc0; sc <= sc1; sc += 2)
        {
          if(sr == ref_r && sc == ref_c) continue;

          if(pmeans)
          {
            const float dm = ref_mean - pmeans[sr * mwidth + sc];
            if(dm > mean_thr || dm < -mean_thr) continue;
          }

          const float d = bm3d_patch_ssd(pilot, width, ref_r, ref_c, sr, sc, worst_dist);
          if(d < tau)
          {
            if(nmatches < BM3D_MAX_MATCHED)
            {
              matches[nmatches++] = (bm3d_match_t){ d, sr, sc };
              if(nmatches == BM3D_MAX_MATCHED)
              {
                worst_dist = 0;
                for(int i = 0; i < nmatches; i++)
                  if(matches[i].dist > worst_dist) worst_dist = matches[i].dist;
              }
            }
            else if(d < worst_dist)
            {
              int wi = 0;
              for(int i = 1; i < nmatches; i++) if(matches[i].dist > matches[wi].dist) wi = i;
              matches[wi] = (bm3d_match_t){ d, sr, sc };
              worst_dist = 0;
              for(int i = 0; i < nmatches; i++) if(matches[i].dist > worst_dist) worst_dist = matches[i].dist;
            }
          }
        }

      sort_matches(matches, nmatches);
      const int group_size = bm3d_next_pow2(nmatches);

      /* Extract patches → 8×8 DCT for pilot and noisy */
      float pilot_dct[BM3D_MAX_MATCHED][BM3D_PATCH_PIXELS];
      float noisy_dct[BM3D_MAX_MATCHED][BM3D_PATCH_PIXELS];

      for(int m = 0; m < nmatches; m++)
      {
        float patch[BM3D_PATCH_PIXELS];

        for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
          memcpy(patch + dy * BM3D_PATCH_SIZE,
                 pilot + (size_t)(matches[m].row + dy) * width + matches[m].col,
                 BM3D_PATCH_SIZE * sizeof(float));
        dct2d_forward(patch, pilot_dct[m]);

        for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
          memcpy(patch + dy * BM3D_PATCH_SIZE,
                 noisy + (size_t)(matches[m].row + dy) * width + matches[m].col,
                 BM3D_PATCH_SIZE * sizeof(float));
        dct2d_forward(patch, noisy_dct[m]);
      }
      for(int m = nmatches; m < group_size; m++)
      {
        memcpy(pilot_dct[m], pilot_dct[0], sizeof(float) * BM3D_PATCH_PIXELS);
        memcpy(noisy_dct[m], noisy_dct[0], sizeof(float) * BM3D_PATCH_PIXELS);
      }

      /* 1D Hadamard + textbook Wiener with DC bin protection */
      float wiener_energy = 0.0f;
      for(int p = 0; p < BM3D_PATCH_PIXELS; p++)
      {
        float had_pilot[BM3D_MAX_MATCHED], had_noisy[BM3D_MAX_MATCHED];
        for(int i = 0; i < group_size; i++) had_pilot[i] = pilot_dct[i][p];
        for(int i = 0; i < group_size; i++) had_noisy[i] = noisy_dct[i][p];
        hadamard_transform(had_pilot, group_size);
        hadamard_transform(had_noisy, group_size);

        for(int i = 0; i < group_size; i++)
        {
          /* DC bin (p=0, i=0): preserve group-average patch mean */
          const float s2 = had_pilot[i] * had_pilot[i];
          const float w = (p == 0 && i == 0)
              ? 1.0f
              : fmaxf(s2 / (s2 + sigma_sq + 1e-10f), 0.03f);
          had_noisy[i] *= w;
          wiener_energy += w * w;
        }

        hadamard_transform(had_noisy, group_size);
        for(int i = 0; i < group_size; i++) noisy_dct[i][p] = had_noisy[i];
      }

      /* Inverse DCT → aggregate with Kaiser window */
      const float wie_weight = 1.0f / fmaxf(wiener_energy, 1e-6f);
      float *my_numer = t_numer[tid];
      float *my_denom = t_denom[tid];

      for(int m = 0; m < nmatches; m++)
      {
        float patch[BM3D_PATCH_PIXELS];
        dct2d_inverse(noisy_dct[m], patch);

        const int mr = matches[m].row, mc = matches[m].col;
        for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
          for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++)
          {
            const size_t pos = (size_t)(mr + dy) * width + (mc + dx);
            const float kw = bm3d_kaiser_2d[dy * BM3D_PATCH_SIZE + dx];
            my_numer[pos] += wie_weight * kw * patch[dy * BM3D_PATCH_SIZE + dx];
            my_denom[pos] += wie_weight * kw;
          }
      }
    }
  }

  /* Merge per-thread buffers */
  DT_OMP_FOR()
  for(int i = 0; i < (int)npixels; i++)
  {
    float n = 0.0f, d = 0.0f;
    for(int t = 0; t < nthreads; t++)
    { n += t_numer[t][i]; d += t_denom[t][i]; }
    output[i] = (d > 0.0f) ? n / d : noisy[i];
  }

  for(int t = 0; t < nthreads; t++)
  { dt_free_align(t_numer[t]); dt_free_align(t_denom[t]); }
  free(t_numer); free(t_denom);
  dt_free_align(pmeans);
}


/* BM3D-CFA full pipeline */
static void gat_bm3d_denoise_cfa(const float *const restrict in, float *const restrict out,
                                   const dt_iop_roi_t *const roi, const float strength,
                                   const uint32_t filters,
                                   const float known_alpha, const float known_sigma_sq)
{
  const int width = roi->width, height = roi->height;
  const size_t npixels = (size_t)width * height;
  memcpy(out, in, sizeof(float) * npixels);

  if(strength <= 0.0f) return;
  if(width < BM3D_PATCH_SIZE * 2 || height < BM3D_PATCH_SIZE * 2) return;

  /* Phase 0: Noise parameter estimation — use known values if provided */
  bm3d_noise_params_t np;
  if(known_alpha > 0.0f)
  {
    np.alpha = known_alpha;
    np.sigma_sq = known_sigma_sq;
    fprintf(stderr, "[rawdenoise] BM3D-CFA: using KNOWN noise params: alpha=%.8f sigma_sq=%.10f\n",
            np.alpha, np.sigma_sq);
  }
  else
  {
    np = bm3d_estimate_noise(in, width, height);
    fprintf(stderr, "[rawdenoise] BM3D-CFA: using BLIND noise estimation: alpha=%.8f sigma_sq=%.10f\n",
            np.alpha, np.sigma_sq);
  }
  gat_build_inverse_table(np.alpha, np.sigma_sq);

  float *gat_mosaic = NULL, *noisy = NULL, *pilot = NULL;

  /* Phase 1: GAT forward */
  gat_mosaic = dt_alloc_align_float(npixels);
  if(!gat_mosaic) return;

  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++)
    gat_mosaic[i] = gat_forward(in[i], np.alpha, np.sigma_sq);

  /* Phase 2: σ estimation and normalization */
  const float sigma_gat = estimate_gat_sigma_mosaic(gat_mosaic, width, height);
  {
    const float inv_sg = 1.0f / sigma_gat;
    DT_OMP_FOR()
    for(size_t i = 0; i < npixels; i++)
      gat_mosaic[i] *= inv_sg;
  }

  noisy = dt_alloc_align_float(npixels);
  if(!noisy) goto cleanup_cfa;
  memcpy(noisy, gat_mosaic, sizeof(float) * npixels);

  fprintf(stderr, "[rawdenoise] BM3D-CFA: alpha=%.8f sigma_sq=%.10f | "
                   "GAT sigma=%.4f | size=%dx%d | str=%.2f\n",
          np.alpha, np.sigma_sq, sigma_gat, width, height, strength);

  /* Phase 3: Step 1 — hard thresholding pilot */
  pilot = dt_alloc_align_float(npixels);
  if(!pilot) goto cleanup_cfa;
  bm3d_cfa_step1(noisy, pilot, width, height, strength);
  fprintf(stderr, "[rawdenoise] BM3D-CFA: Step1 (hard threshold) done\n");

  /* Phase 4: Step 2 — Wiener (no varmap; global σ² used directly) */
  bm3d_cfa_step2(noisy, pilot, gat_mosaic, width, height, strength);

  dt_free_align(noisy); noisy = NULL;
  dt_free_align(pilot); pilot = NULL;

  fprintf(stderr, "[rawdenoise] BM3D-CFA: Step2 (Wiener) done\n");

  /* Phase 5: σ-denormalize → inverse GAT */
  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++)
    out[i] = gat_inverse_exact(gat_mosaic[i] * sigma_gat);

  dt_free_align(gat_mosaic); gat_mosaic = NULL;

  fprintf(stderr, "[rawdenoise] BM3D-CFA: done\n");
  return;

cleanup_cfa:
  dt_free_align(gat_mosaic);
  dt_free_align(noisy);
  dt_free_align(pilot);
}

/* ================================================================
 *  RAW L/C BM3D — Full pipeline
 *  RAW L/C BM3D — フルパイプライン
 *
 *  Pre-demosaic denoiser operating on RGGB before WB and demosaic.
 *  Fully blind: no white-balance coefficients, no per-channel QE prior.
 *  WB / デモザイク前の RGGB 上で動作。
 *  Fully blind（WB も per-ch QE prior も使わない）。
 *
 *  Pipeline / パイプライン:
 *   1. Half-res RGGB extraction → piecewise C¹ GAT → RMS unified σ-normalize
 *      半解像度RGGB抽出 → 区分 C¹ 級 GAT → RMS 統一 σ 正規化
 *   2. Self-consistent dark anchor (per-ch DC subtract) → WHT → L/C₁/C₂/C₃
 *      自己整合ダークアンカー（per-ch DC 減算）→ WHT → 輝度/色差分解
 *   3. Step1: hard-threshold pilot on half-res L/C (shared matches)
 *      Step1: 半解像度 L/C 上でハード閾値パイロット（共有マッチング）
 *   4. Step2: Wiener on half-res L/C (independent σ_L / σ_C)
 *      Step2: 半解像度 L/C Wiener（σ_L / σ_C 独立制御）
 *   5. Full-res L: sliding WHT-DC → full-res BM3D Step2 on L plane
 *      Full-res L: sliding WHT-DC → L 平面の full-res BM3D Step2
 *   6. Inverse WHT with full-res L + half-res C (anchor restore)
 *      逆 WHT（full-res L + half-res C、アンカー復元）
 *   7. σ-denormalize → exact inverse GAT → write back
 *      σ 逆正規化 → 厳密逆 GAT → 出力
 *
 *  ----------------------------------------------------------------
 *  Theoretical highlights / 理論ポイント
 *  ----------------------------------------------------------------
 *
 *  (a) Piecewise C¹ VST (extends Foi GAT to all reals)
 *      区分 C¹ 級 VST（Foi GAT を実数全域に拡張）
 *
 *      The classical Foi GAT sqrt branch is only valid for y ≥ -3α/8.
 *      Below that, naive clamping creates a δ-function pile-up at the
 *      clamp boundary, which (i) breaks per-channel distribution shape
 *      and (ii) injects false chroma DC after WHT. We extend GAT with
 *      a C¹-continuous linear branch:
 *
 *          T(y) = (2/α)·sqrt(αy + 3α²/8 + σ²)        (y ≥ -3α/8, sqrt)
 *          T(y) = T_break + (y - y_break)/σ          (y <  -3α/8, linear)
 *          T_break = 2σ/α     (continuity)
 *          slope  = 1/σ       (C¹ + variance stabilized in linear branch)
 *
 *      Variance ≈ 1 in BOTH branches (in the linear branch, signal-
 *      dependent variance is negligible so observed variance ≈ σ²,
 *      and dividing by σ rescales it to 1). Negative-valued read-noise
 *      samples flow through without distortion, preserving symmetric
 *      per-channel statistics.
 *
 *      Foi GAT の sqrt 分岐は y ≥ -3α/8 でのみ妥当。下方クランプは
 *      δ ピーク pile-up を作り、per-ch 分布形状を歪めて WHT 後に偽
 *      chroma DC を生む。本実装では C¹ 連続で線形分岐を継ぎ足す。
 *      両分岐で分散 ≈ 1。負値 read noise 標本が無歪で通り、各 ch の
 *      対称統計が保たれる。
 *
 *  (b) Unified σ normalization (RMS, makes Var[L]=Var[Cₖ]=1 exactly)
 *      統一 σ 正規化（RMS、Var[L]=Var[Cₖ]=1 を厳密に満たす）
 *
 *      All 4 channels are divided by the SAME σ — but the choice of
 *      σ is not arithmetic mean of per-ch GAT-domain stds, it is the
 *      RMS:  unified_sigma = √( mean(σ_c²) ).
 *
 *      The single-shot Poissonian-Gaussian noise estimator fits one
 *      (α, σ²) globally and is biased toward the dominant population
 *      (G, 50% of pixels). The fitted parameters do not perfectly fit
 *      B/R, leaving a per-channel post-GAT variance non-uniformity
 *      (e.g. observed: σ_B≈0.745, σ_G/R≈1.02 → Var_B≈0.55, Var_G/R≈1.04).
 *      This is a noise-MODEL mis-specification, not a normalization bug.
 *
 *      Per-ch normalization (different σ per channel) would FIX the
 *      per-ch Var=1 condition but BREAK the WHT chroma fidelity:
 *      different scaling per channel rescales R/G/B asymmetrically, so
 *      uniform-signal input produces non-zero C₁/C₂/C₃ planes (false
 *      chroma). The unified-σ design avoids this entirely.
 *
 *      The RMS choice has a closed-form property: WHT outputs satisfy
 *
 *          Var[L] = Var[C₁] = Var[C₂] = Var[C₃]
 *               = (1/4) Σ_c (σ_c / unified_sigma)²
 *               = mean(σ_c²) / unified_sigma²
 *               = 1   (exactly, by RMS definition)
 *
 *      So the per-ch deviation cancels under WHT averaging, and L/C
 *      planes have unit variance EXACTLY. This is what the dark anchor
 *      self-consistency iteration uses as its mathematically exact
 *      target std=1.0, and what lets Step2 Wiener apply σ_user directly
 *      with no further scaling.
 *
 *      Arithmetic mean (the previous choice) gives Var[L]≈1.016 — small
 *      but non-zero. RMS removes the residual entirely.
 *
 *      4 ch を 同一 σ で正規化するが、その値は per-ch GAT-norm std の
 *      算術平均ではなく **RMS**: unified_sigma = √( mean(σ_c²) )。
 *      single-shot 推定は G に bias し、B/R の真ノイズには合わない。
 *      観測: σ_B≈0.745, σ_G/R≈1.02 → Var_B≈0.55, Var_G/R≈1.04。
 *      これは正規化のバグではなく noise model の mis-specification が
 *      GAT 経由で観測される現象。per-ch 正規化は per-ch Var=1 を実現
 *      する代わりに WHT chroma fidelity を破壊（均一信号から偽 chroma）。
 *      RMS 化により WHT 後 Var[L]=Var[Cₖ]=1 が定義より厳密に成立する。
 *      これが (c) の自己整合反復が依拠する厳密な基準となる。
 *      算術平均では Var[L]≈1.016（残差 1.6%）、RMS でこれを 0 にする。
 *
 *  (c) Self-consistent dark anchor (per-channel DC protection)
 *      自己整合ダークアンカー（per-ch DC 保護）
 *
 *      Per-channel QE differences (e.g. B has roughly 1/7 the QE of G
 *      under typical CFAs) are REAL signal to be PRESERVED in bright
 *      regions — they encode color information. In dark regions,
 *      however, QE × signal ≈ 0, while the GAT linear branch and
 *      read-noise statistics leave residual per-channel DC offsets
 *      that, after WHT, manifest as false dark-area chroma.
 *
 *      Fix: estimate a per-channel DC anchor over a noise-dominated
 *      cohort and subtract it before WHT, then restore it after
 *      Wiener. The cohort is selected by smooth weights
 *
 *          w(L_i) = 1 / (1 + (L_i / s)^4)
 *
 *      on the raw-linear 2×2 luma L_i = mean of input block. The high
 *      polynomial decay (1/x⁴) is robust to scale-estimation error
 *      compared with a Gaussian (which has long tails). Bright pixels
 *      get w → 0 and are excluded — bright per-ch QE differences pass
 *      through ENTIRELY UNTOUCHED (no alignment, no correction).
 *
 *      Scale s — self-consistent estimation:
 *        - Initial: s₀ = σ²/α  (read-noise-equivalent luma; the level
 *          at which Poisson-detectable signal emerges from read noise).
 *        - Update: measure the weighted per-ch residual std in GAT-norm
 *          space (target = 1.0 by construction of (b)) and rescale
 *          s ← s · √(1/std). 2 iterations.
 *        - Bounded: s ∈ [0.05·s₀, 50·s₀] for safety.
 *
 *      Self-consistency decouples Option α from α/σ² point-estimate
 *      accuracy: the noise model only seeds initialization, while the
 *      final scale is calibrated by the data's own GAT-norm dispersion.
 *      This is what strengthens the fully blind property — even if
 *      noise estimation is imperfect, the dark anchor self-corrects.
 *
 *      per-ch QE 差は明部では保存すべき信号だが、暗部では信号 ≈ 0 で
 *      QE 差 × 信号 ≈ 0 になる。GAT 線形分岐 / read noise 非対称が
 *      per-ch DC オフセットを残し WHT 後に偽 chroma を生む。
 *      解決策は noise-only コホート上で per-ch DC アンカーを推定し、
 *      WHT 前に減算 → Wiener 後に復元する。コホートは raw linear luma
 *      に対する滑らかな重み w(L)=1/(1+(L/s)⁴) で選び、明部は w→0 で
 *      除外される（明部の per-ch QE 差は完全に無加工で通る）。
 *      スケール s は ノイズ等価輝度 σ²/α で初期化し、GAT-norm 残差
 *      std を測って自己整合的に補正する（target=1.0、2 反復）。
 *      これにより α, σ² 点推定への依存を弱め、fully blind 性を強化する。
 *
 *  Ref: Danielyan et al. "Cross-color BM3D" (LNLA 2009)
 *       Foi et al. "Practical Poissonian-Gaussian noise modeling..." (Sig.Proc. 2008)
 *       Makitalo & Foi, "Optimal inversion of the generalized Anscombe
 *       transformation for Poisson-Gaussian noise" (TIP 2013)
 * ================================================================ */
/* luma_strength / chroma_strength semantics:
 *
 *   σ_L = luma_strength    (Wiener strength on L plane)
 *   σ_C = chroma_strength  (Wiener strength on C₁/C₂/C₃ planes)
 *
 *   Both are direct values in GAT-normalized noise space where the
 *   true noise σ = 1 by construction (see Theoretical highlight (b)).
 *   σ_L and σ_C are fully independent — the user can push chroma
 *   denoising strong while keeping luma gentle (e.g. σ_L=0.5, σ_C=2.0
 *   for delicate detail + strong false-chroma suppression).
 *
 *   Defaults: σ_L = 0.5 (gentle luma), σ_C = 1.0 (textbook Wiener).
 *
 *   Step1 uses a single σ (= luma_strength) for all 4 L/C planes —
 *   its role is to provide a clean pilot for Step2's matching. Using
 *   a single threshold preserves the shared-matches / shared-aggregation
 *   invariant that prevents the 2×2 grid artifact. Only Step2 applies
 *   independent σ_L / σ_C Wiener.
 *
 *   σ_L / σ_C 独立制御。GAT-normalized 空間での直接値（真ノイズ σ=1）。
 *   デフォルト: σ_L=0.5（穏やか）, σ_C=1.0（教科書 Wiener）。
 *   Step1 は単一 σ で pilot 生成（shared matches で 2×2 artifact 防止）。
 *   Step2 のみが独立 Wiener を実行する。
 */
static void gat_bm3d_denoise_rawlc(const float *const restrict in, float *const restrict out,
                                    const dt_iop_roi_t *const roi,
                                    const float luma_strength, const float chroma_strength,
                                    const uint32_t filters,
                                    const float known_alpha, const float known_sigma_sq)
{
  const int width = roi->width, height = roi->height;
  const size_t npixels = (size_t)width * height;
  memcpy(out, in, sizeof(float) * npixels);

  if(luma_strength <= 0.0f) return;

  const int halfwidth = (width + 1) / 2;
  const int halfheight = (height + 1) / 2;
  if(halfwidth < BM3D_PATCH_SIZE * 2 || halfheight < BM3D_PATCH_SIZE * 2) return;

  const size_t chsize = (size_t)halfwidth * halfheight;

  /* Noise parameter estimation — use known values if provided */
  bm3d_noise_params_t np;
  if(known_alpha > 0.0f)
  {
    np.alpha = known_alpha;
    np.sigma_sq = known_sigma_sq;
    fprintf(stderr, "[rawdenoise] RAW L/C BM3D: using KNOWN noise params: alpha=%.8f sigma_sq=%.10f\n",
            np.alpha, np.sigma_sq);
  }
  else
  {
    np = bm3d_estimate_noise(in, width, height);
    fprintf(stderr, "[rawdenoise] RAW L/C BM3D: using BLIND noise estimation: alpha=%.8f sigma_sq=%.10f\n",
            np.alpha, np.sigma_sq);
  }
  gat_build_inverse_table(np.alpha, np.sigma_sq);

  /* Pre-declare all buffers */
  float *ch_gat[4] = { NULL, NULL, NULL, NULL };
  float *luma = NULL, *chroma1 = NULL, *chroma2 = NULL, *chroma3 = NULL;
  float *luma_pilot = NULL, *c1_pilot = NULL, *c2_pilot = NULL, *c3_pilot = NULL;
  float *luma_out = NULL, *c1_out = NULL, *c2_out = NULL, *c3_out = NULL;
  /* Full-res L plane buffers (Phase 5) */
  float *L_fullres_noisy = NULL, *L_fullres_pilot = NULL, *L_fullres_out = NULL;

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

  /* RMS unified σ normalization — see Theoretical highlight (b).
   *
   * unified_sigma := √( mean(σ_c²) )
   *
   * This choice makes Var[L] = Var[Cₖ] = 1 exactly after WHT,
   * regardless of per-channel GAT-domain variance non-uniformity
   * caused by single-shot noise estimation bias toward G.
   * Per-channel normalization would fix per-ch Var=1 but break
   * WHT signal proportionality (false chroma on uniform input).
   *
   * RMS 統一 σ — 理論ポイント (b) 参照。
   * Var[L]=Var[Cₖ]=1 を厳密に保証。per-ch 正規化は偽 chroma を生む。
   */
  {
    /* RMS of per-channel post-GAT std → makes Var[L]=Var[Cₖ]=1 exactly */
    const float mean_var = 0.25f * (sigma_gat_ch[0] * sigma_gat_ch[0]
                                  + sigma_gat_ch[1] * sigma_gat_ch[1]
                                  + sigma_gat_ch[2] * sigma_gat_ch[2]
                                  + sigma_gat_ch[3] * sigma_gat_ch[3]);
    const float unified_sigma = sqrtf(fmaxf(mean_var, 1e-12f));

    /* Diagnostic: mean of per-ch variance AFTER normalization should be ≈ 1.
     * This is an observation, not a correction — the Wiener path receives
     * no explicit variance argument. RMS definition guarantees this by
     * construction; the log line lets us verify it on every run. */
    const float post_mean_var = mean_var / (unified_sigma * unified_sigma);

    fprintf(stderr, "[rawdenoise] RAW L/C BM3D: alpha=%.8f sigma_sq=%.10f | "
                     "unified_sigma=%.4f [RMS] (per-ch: %.4f %.4f %.4f %.4f) | "
                     "post-norm mean Var=%.4f (target=1.0) | "
                     "size=%dx%d (half=%dx%d) | "
                     "σ_L=%.3f σ_C=%.3f (independent, GAT-norm space)\n",
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

  /* === DEBUG DIAG-A: dark area GAT-domain values (after GAT+norm, before mean-sub) === */
  {
    const float dark_thresh = 0.002f;
    int dark_n = 0;
    double gat_sum[4] = {0}, in_sum[4] = {0};
    int neg_per_ch[4] = {0};
    for(int hy = 0; hy < halfheight; hy++)
      for(int hx = 0; hx < halfwidth; hx++)
      {
        const int fy = hy * 2, fx = hx * 2;
        /* input 2x2 block average as darkness criterion */
        const float iv0 = in[(size_t)fy * width + fx];
        const float iv1 = in[(size_t)(fy + 1) * width + fx];
        const float iv2 = in[(size_t)fy * width + fx + 1];
        const float iv3 = in[(size_t)(fy + 1) * width + fx + 1];
        const float luma_in = (iv0 + iv1 + iv2 + iv3) * 0.25f;
        if(luma_in >= dark_thresh) continue;
        dark_n++;
        const size_t hpos = (size_t)hy * halfwidth + hx;
        gat_sum[0] += ch_gat[0][hpos]; gat_sum[1] += ch_gat[1][hpos];
        gat_sum[2] += ch_gat[2][hpos]; gat_sum[3] += ch_gat[3][hpos];
        in_sum[0] += iv0; in_sum[1] += iv1; in_sum[2] += iv2; in_sum[3] += iv3;
        if(iv0 < 0) neg_per_ch[0]++;
        if(iv1 < 0) neg_per_ch[1]++;
        if(iv2 < 0) neg_per_ch[2]++;
        if(iv3 < 0) neg_per_ch[3]++;
      }
    if(dark_n > 0)
    {
      const float ig = (float)((in_sum[1] + in_sum[2]) / (2.0 * dark_n));
      fprintf(stderr, "[rawdenoise] DIAG-A dark (luma<%.4f): %d blocks\n", dark_thresh, dark_n);
      fprintf(stderr, "[rawdenoise]   INPUT mean: ch0=%.7f ch1=%.7f ch2=%.7f ch3=%.7f\n",
              (float)(in_sum[0]/dark_n), (float)(in_sum[1]/dark_n),
              (float)(in_sum[2]/dark_n), (float)(in_sum[3]/dark_n));
      fprintf(stderr, "[rawdenoise]   INPUT neg%%: ch0=%.1f%% ch1=%.1f%% ch2=%.1f%% ch3=%.1f%%\n",
              100.0*neg_per_ch[0]/dark_n, 100.0*neg_per_ch[1]/dark_n,
              100.0*neg_per_ch[2]/dark_n, 100.0*neg_per_ch[3]/dark_n);
      fprintf(stderr, "[rawdenoise]   INPUT B/G=%.4f R/G=%.4f (G=avg(Gb,Gr)=%.7f)\n",
              (float)(in_sum[0]/dark_n) / fmaxf(ig, 1e-10f),
              (float)(in_sum[3]/dark_n) / fmaxf(ig, 1e-10f), ig);
      fprintf(stderr, "[rawdenoise]   GAT-NORM mean: ch0=%.4f ch1=%.4f ch2=%.4f ch3=%.4f\n",
              (float)(gat_sum[0]/dark_n), (float)(gat_sum[1]/dark_n),
              (float)(gat_sum[2]/dark_n), (float)(gat_sum[3]/dark_n));
    }
  }

  /* ================================================================
   * Phase 2: Self-consistent dark anchor + WHT → L/C₁/C₂/C₃
   *
   * See section header (Theoretical highlight (c)) for the full
   * derivation. Summary:
   *
   *   1. Select a noise-dominated cohort with smooth weights
   *        w(L_i) = 1 / (1 + (L_i / s)^4)
   *      where L_i = raw-linear luma of the 2×2 input block.
   *   2. Compute weighted per-ch mean → ch_dark_ref[c].
   *   3. Self-consistent scale update: measure weighted per-ch residual
   *      std in GAT-norm space, target = 1.0, rescale s ← s·√(1/std).
   *   4. Iterate (n_iter = 2), then subtract ch_dark_ref[c] from ch_gat.
   *   5. Phase 6 will restore ch_dark_ref[c] after Wiener.
   *
   *   - Bright pixels: w → 0 → per-ch QE differences pass untouched.
   *   - Dark pixels:   w → 1 → anchor captures GAT linear-branch and
   *                            read-noise residual DC offsets only.
   *   - α, σ² seed only the initial scale; final s is calibrated by
   *     the data's own GAT-norm dispersion → fully blind robustness.
   *
   * Phase 2: 自己整合ダークアンカー + WHT → L/C₁/C₂/C₃
   * セクション冒頭の理論ポイント (c) を参照。明部の per-ch QE 差を
   * 一切いじらずに、暗部 chroma DC バイアスのみを per-ch 退避し、
   * Wiener 後 (Phase 6) に復元する。スケール s は σ²/α で初期化し、
   * GAT-norm 残差 std を測って自己整合的に補正（目標=1.0、2 反復）。
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
          const size_t hpos = (size_t)hy * halfwidth + hx;
          sum_wch[0] += w * ch_gat[0][hpos];
          sum_wch[1] += w * ch_gat[1][hpos];
          sum_wch[2] += w * ch_gat[2][hpos];
          sum_wch[3] += w * ch_gat[3][hpos];
        }

      const double inv_sw = 1.0 / fmax(sum_w, 1e-20);
      for(int c = 0; c < 4; c++)
        ch_dark_ref[c] = (float)(sum_wch[c] * inv_sw);

      if(iter == n_iter) break;  /* skip scale update on last iter */

      /* Weighted per-ch residual std in GAT-norm (target = 1.0) */
      double sum_wresid2 = 0.0;
      for(int hy = 0; hy < halfheight; hy++)
        for(int hx = 0; hx < halfwidth; hx++)
        {
          const int fy = hy * 2, fx = hx * 2;
          const float iv0 = in[(size_t)fy * width + fx];
          const float iv1 = in[(size_t)(fy + 1) * width + fx];
          const float iv2 = in[(size_t)fy * width + fx + 1];
          const float iv3 = in[(size_t)(fy + 1) * width + fx + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          const size_t hpos = (size_t)hy * halfwidth + hx;
          double resid2 = 0.0;
          for(int c = 0; c < 4; c++)
          {
            const double d = (double)ch_gat[c][hpos] - ch_dark_ref[c];
            resid2 += d * d;
          }
          sum_wresid2 += w * resid2 * 0.25;  /* per-ch average */
        }
      const double measured_std =
          sqrt(fmax(sum_wresid2 * inv_sw, 1e-20));
      /* Damped multiplicative update: target std = 1.0 */
      const double ratio = 1.0 / measured_std;
      s_scale *= sqrt(ratio);
      if(s_scale < s_min) s_scale = s_min;
      if(s_scale > s_max) s_scale = s_max;
    }

    fprintf(stderr, "[rawdenoise] Option-α self-consistent: s_init=%.6e s_final=%.6e | "
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

  /* === DEBUG DIAG-D2: per-channel (value - global_mean) histogram in dark area ===
   * Inspects the actual signal that enters WHT.
   * Bins are in GAT-NORM units (i.e. multiples of unified_sigma).
   * If B distribution is asymmetric vs G/R, that explains why Wiener+iWHT
   * pushes B and G in opposite directions in dark regions. */
  {
    fprintf(stderr, "[rawdenoise] DIAG-D2 ch_dark_ref (10th pctile, GAT-NORM): B=%.4f Gb=%.4f Gr=%.4f R=%.4f\n",
            ch_dark_ref[0], ch_dark_ref[1], ch_dark_ref[2], ch_dark_ref[3]);
    const float dark_thresh = 0.002f;
    const int   NB = 41;       /* bins -20.5..-19.5 ... 19.5..20.5 */
    const float BIN_W = 1.0f;
    const float BIN_LO = -20.5f;
    long long hist[4][41];
    double sum[4] = {0}, sum2[4] = {0};
    double minv[4] = { 1e30, 1e30, 1e30, 1e30};
    double maxv[4] = {-1e30,-1e30,-1e30,-1e30};
    int dark_n = 0;
    for(int c = 0; c < 4; c++) for(int b = 0; b < NB; b++) hist[c][b] = 0;
    for(int hy = 0; hy < halfheight; hy++)
      for(int hx = 0; hx < halfwidth; hx++)
      {
        const int fy = hy * 2, fx = hx * 2;
        const float iv0 = in[(size_t)fy * width + fx];
        const float iv1 = in[(size_t)(fy + 1) * width + fx];
        const float iv2 = in[(size_t)fy * width + fx + 1];
        const float iv3 = in[(size_t)(fy + 1) * width + fx + 1];
        const float luma_in = (iv0 + iv1 + iv2 + iv3) * 0.25f;
        if(luma_in >= dark_thresh) continue;
        dark_n++;
        const size_t hpos = (size_t)hy * halfwidth + hx;
        for(int c = 0; c < 4; c++)
        {
          const float v = ch_gat[c][hpos]; /* already mean-subtracted, in sigma units */
          sum[c]  += v;
          sum2[c] += (double)v * v;
          if(v < minv[c]) minv[c] = v;
          if(v > maxv[c]) maxv[c] = v;
          int b = (int)floorf((v - BIN_LO) / BIN_W);
          if(b < 0) b = 0;
          if(b >= NB) b = NB - 1;
          hist[c][b]++;
        }
      }
    if(dark_n > 0)
    {
      const char *cname[4] = {"B ", "Gb", "Gr", "R "};
      fprintf(stderr, "[rawdenoise] DIAG-D2 dark (luma<%.4f): %d blocks (each ch sigma-units)\n",
              dark_thresh, dark_n);
      for(int c = 0; c < 4; c++)
      {
        const double m = sum[c] / dark_n;
        const double v = sum2[c] / dark_n - m * m;
        const double sd = (v > 0.0) ? sqrt(v) : 0.0;
        /* skewness via 3rd moment (single pass already used; do quick second loop) */
        fprintf(stderr, "[rawdenoise]   %s: mean=%+.4f std=%.4f min=%+.3f max=%+.3f\n",
                cname[c], m, sd, minv[c], maxv[c]);
      }
      /* Print histograms (one line per channel, bins -20..20) */
      for(int c = 0; c < 4; c++)
      {
        fprintf(stderr, "[rawdenoise]   %s hist:", cname[c]);
        for(int b = 0; b < NB; b++)
          fprintf(stderr, " %lld", hist[c][b]);
        fprintf(stderr, "\n");
      }
      /* Compact tail summary: count of |v|>3, >5, >10 sigma */
      for(int c = 0; c < 4; c++)
      {
        long long lt_m10 = 0, m10_m5 = 0, m5_m3 = 0, m3_p3 = 0, p3_p5 = 0, p5_p10 = 0, gt_p10 = 0;
        for(int b = 0; b < NB; b++)
        {
          const float lo = BIN_LO + b * BIN_W;
          const long long n = hist[c][b];
          if(lo < -10.0f) lt_m10 += n;
          else if(lo < -5.0f) m10_m5 += n;
          else if(lo < -3.0f) m5_m3 += n;
          else if(lo < 3.0f) m3_p3 += n;
          else if(lo < 5.0f) p3_p5 += n;
          else if(lo < 10.0f) p5_p10 += n;
          else gt_p10 += n;
        }
        fprintf(stderr, "[rawdenoise]   %s tails: <-10=%lld [-10..-5)=%lld [-5..-3)=%lld [-3..3)=%lld [3..5)=%lld [5..10)=%lld >=10=%lld\n",
                cname[c], lt_m10, m10_m5, m5_m3, m3_p3, p3_p5, p5_p10, gt_p10);
      }
    }
  }

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
   * Phase 3: Step1 pilot on half-res L/C
   * ================================================================ */
  luma_pilot = dt_alloc_align_float(chsize);
  c1_pilot   = dt_alloc_align_float(chsize);
  c2_pilot   = dt_alloc_align_float(chsize);
  c3_pilot   = dt_alloc_align_float(chsize);
  if(!luma_pilot || !c1_pilot || !c2_pilot || !c3_pilot) goto cleanup_rawlc;

  {
    const float *rggb_match[4] = { ch_gat[0], ch_gat[1], ch_gat[2], ch_gat[3] };
    /* Step1 uses luma_strength as σ: same "under-denoise multiplier" semantics
     * as Step2. Using 1.0 here would kill too many coefficients in the pilot,
     * making p²≈0 and forcing Step2 Wiener to the floor in flat regions. */
    bm3d_step1_lc(rggb_match, luma, chroma1, chroma2, chroma3,
                  luma_pilot, c1_pilot, c2_pilot, c3_pilot,
                  halfwidth, halfheight, luma_strength);
  }
  fprintf(stderr, "[rawdenoise] RAW L/C BM3D: Step1 half-res pilot done\n");

  /* ================================================================
   * Phase 4: Step2 Wiener on half-res L/C planes.
   *
   *   pilot[c]  = luma_pilot / c{1,2,3}_pilot   (Step1, anchor-subtracted)
   *   noisy[c]  = luma       / chroma{1,2,3}    (Phase 2, anchor-subtracted)
   *   output[c] = luma_out   / c{1,2,3}_out     (denoised, still subtracted)
   *
   * Pilot and noisy share the same dark anchor basis → DC consistency
   * is automatic. RMS unified σ guarantees Var[L]=Var[Cₖ]=1 exactly,
   * so no local variance map is needed — global σ_L/σ_C suffices.
   *
   * Note: luma_out from this phase is NOT used in the final output.
   * Phase 5 computes and denoises L at full resolution instead.
   * Only C₁/C₂/C₃ outputs (c1_out, c2_out, c3_out) carry forward.
   *
   * Phase 4: 半解像度 L/C Wiener。luma_out はこのフェーズでは最終出力
   * に使われない（Phase 5 で full-res L を別途デノイズ）。C₁/C₂/C₃ のみ
   * 後段で使用。
   * ================================================================ */
  luma_out = dt_alloc_align_float(chsize);
  c1_out   = dt_alloc_align_float(chsize);
  c2_out   = dt_alloc_align_float(chsize);
  c3_out   = dt_alloc_align_float(chsize);
  if(!luma_out || !c1_out || !c2_out || !c3_out) goto cleanup_rawlc;

  {
    const float *noisy_lc[4] = { luma,       chroma1,  chroma2,  chroma3 };
    const float *pilot_lc[4] = { luma_pilot, c1_pilot, c2_pilot, c3_pilot };
    float       *out_lc[4]   = { luma_out,   c1_out,   c2_out,   c3_out };

    bm3d_step2_lc(noisy_lc, pilot_lc, out_lc,
                  halfwidth, halfheight, luma_strength,
                  chroma_strength, 2);
  }

  fprintf(stderr, "[rawdenoise] RAW L/C BM3D: Step2 L/C split Wiener done "
                   "(σ_L=%.3f σ_C=%.3f)\n", luma_strength, chroma_strength);

  /* ================================================================
   * Phase 5: Full-res L BM3D Step2.
   *
   * Compute L_fullres at every raw pixel from the half-res L/C planes
   * (noisy and pilot) via sliding 2×2 WHT-DC, then run a dedicated
   * single-channel BM3D Step2 at full resolution. The denoised
   * L_fullres replaces the half-res luma_out in Phase 6's inverse WHT,
   * giving each raw pixel its own L value and eliminating the 2×2
   * plateau artifact that block-constant L creates under strong Wiener.
   *
   * Phase 5: full-res L BM3D Step2。sliding WHT-DC で全 raw pixel の
   * L を計算し full-res BM3D Step2 でデノイズ。Phase 6 の逆 WHT で
   * 使用し、2×2 plateau artifact を解消。
   * ================================================================ */
  {
    const size_t fullsize = (size_t)width * height;
    L_fullres_noisy = dt_alloc_align_float(fullsize);
    L_fullres_pilot = dt_alloc_align_float(fullsize);
    L_fullres_out   = dt_alloc_align_float(fullsize);
    if(!L_fullres_noisy || !L_fullres_pilot || !L_fullres_out) goto cleanup_rawlc;

    /* Compute L_fullres from noisy L/C planes */
    compute_L_fullres(luma, chroma1, chroma2, chroma3,
                      halfwidth, halfheight, L_fullres_noisy);

    /* Compute L_fullres from Step1 pilot L/C planes */
    compute_L_fullres(luma_pilot, c1_pilot, c2_pilot, c3_pilot,
                      halfwidth, halfheight, L_fullres_pilot);

    /* Full-res BM3D Step2 on L_fullres (8×8 patches, no CFA constraint).
     * ref_step=2 on full-res ≈ ref_step=1 on half-res in density. */
    const int fullres_width = 2 * halfwidth;
    const int fullres_height = 2 * halfheight;
    fprintf(stderr, "[rawdenoise] RAW L/C BM3D: starting full-res L Step2 "
                     "(%dx%d, ref_step=2, σ_L=%.3f)\n",
            fullres_width, fullres_height, luma_strength);

    bm3d_step2_fullres_L(L_fullres_noisy, L_fullres_pilot, L_fullres_out,
                          fullres_width, fullres_height,
                          luma_strength, 2);

    fprintf(stderr, "[rawdenoise] RAW L/C BM3D: full-res L Step2 done\n");

    dt_free_align(L_fullres_noisy);  L_fullres_noisy = NULL;
    dt_free_align(L_fullres_pilot);  L_fullres_pilot = NULL;
  }

  /* Free L/C noisy and pilot planes (no longer needed) */
  dt_free_align(luma);    luma = NULL;
  dt_free_align(chroma1); chroma1 = NULL;
  dt_free_align(chroma2); chroma2 = NULL;
  dt_free_align(chroma3); chroma3 = NULL;
  dt_free_align(luma_pilot); luma_pilot = NULL;
  dt_free_align(c1_pilot);   c1_pilot = NULL;
  dt_free_align(c2_pilot);   c2_pilot = NULL;
  dt_free_align(c3_pilot);   c3_pilot = NULL;

  /* ================================================================
   * Phase 6: Inverse WHT with full-res L + half-res C.
   *
   * L_fullres_out (denoised at full resolution in Phase 5) provides
   * each raw pixel its own L value. C₁/C₂/C₃ remain half-res
   * (block-wise, from Phase 4). This eliminates the 2×2 plateau
   * artifact that block-constant L creates under strong Wiener.
   *
   * Pixel–channel mapping (from Phase 7 write-back convention):
   *   ch_gat[0] → (2hy,   2hx)   = block top-left
   *   ch_gat[1] → (2hy+1, 2hx)   = block bottom-left
   *   ch_gat[2] → (2hy,   2hx+1) = block top-right
   *   ch_gat[3] → (2hy+1, 2hx+1) = block bottom-right
   *
   * Phase 6: full-res L + half-res C の逆 WHT。各 raw pixel が固有の
   * L を持つため、block-constant L の 2×2 plateau が消滅する。
   * ================================================================ */
  {
    const int fw = 2 * halfwidth;
    DT_OMP_FOR()
    for(int hy = 0; hy < halfheight; hy++)
      for(int hx = 0; hx < halfwidth; hx++)
      {
        const size_t i = (size_t)hy * halfwidth + hx;
        const int fr = 2 * hy, fc = 2 * hx;

        /* Full-res denoised L at each of the 4 pixel positions */
        const float L_tl = L_fullres_out[(size_t)fr       * fw + fc];
        const float L_bl = L_fullres_out[(size_t)(fr + 1) * fw + fc];
        const float L_tr = L_fullres_out[(size_t)fr       * fw + fc + 1];
        const float L_br = L_fullres_out[(size_t)(fr + 1) * fw + fc + 1];

        const float h1 = c1_out[i];    /* C₁ (half-res, block-wise) */
        const float h2 = c2_out[i];    /* C₂ */
        const float h3 = c3_out[i];    /* C₃ */

        /* Inverse WHT per pixel, each with its own denoised L + anchor restore.
         * Signs: ch0=(+,+,+,+) ch1=(+,-,+,-) ch2=(+,+,-,-) ch3=(+,-,-,+) */
        ch_gat[0][i] = (L_tl + h1 + h2 + h3) * 0.5f + ch_dark_ref[0];
        ch_gat[1][i] = (L_bl - h1 + h2 - h3) * 0.5f + ch_dark_ref[1];
        ch_gat[2][i] = (L_tr + h1 - h2 - h3) * 0.5f + ch_dark_ref[2];
        ch_gat[3][i] = (L_br - h1 - h2 + h3) * 0.5f + ch_dark_ref[3];
      }
  }

  dt_free_align(L_fullres_out); L_fullres_out = NULL;
  dt_free_align(luma_out); luma_out = NULL;
  dt_free_align(c1_out);   c1_out = NULL;
  dt_free_align(c2_out);   c2_out = NULL;
  dt_free_align(c3_out);   c3_out = NULL;

  /* === DEBUG DIAG-B: dark area GAT-domain after Step2 + mean restore === */
  {
    const float dark_thresh = 0.002f;
    int dark_n = 0;
    double gat_sum[4] = {0};
    for(int hy = 0; hy < halfheight; hy++)
      for(int hx = 0; hx < halfwidth; hx++)
      {
        const int fy = hy * 2, fx = hx * 2;
        const float iv0 = in[(size_t)fy * width + fx];
        const float iv1 = in[(size_t)(fy + 1) * width + fx];
        const float iv2 = in[(size_t)fy * width + fx + 1];
        const float iv3 = in[(size_t)(fy + 1) * width + fx + 1];
        if((iv0 + iv1 + iv2 + iv3) * 0.25f >= dark_thresh) continue;
        dark_n++;
        const size_t hpos = (size_t)hy * halfwidth + hx;
        gat_sum[0] += ch_gat[0][hpos]; gat_sum[1] += ch_gat[1][hpos];
        gat_sum[2] += ch_gat[2][hpos]; gat_sum[3] += ch_gat[3][hpos];
      }
    if(dark_n > 0)
    {
      fprintf(stderr, "[rawdenoise] DIAG-B dark (after Step2+mean): %d blocks\n", dark_n);
      fprintf(stderr, "[rawdenoise]   GAT-NORM mean: ch0=%.4f ch1=%.4f ch2=%.4f ch3=%.4f\n",
              (float)(gat_sum[0]/dark_n), (float)(gat_sum[1]/dark_n),
              (float)(gat_sum[2]/dark_n), (float)(gat_sum[3]/dark_n));
    }
  }

  /* ================================================================
   * Phase 7: σ-denormalize → inverse GAT per channel → write back
   * ================================================================ */
  for(int c = 0; c < 4; c++)
  {
    const int row_offset = c & 1, col_offset = (c >> 1) & 1;
    const float sg = sigma_gat_ch[c];

    DT_OMP_FOR()
    for(size_t i = 0; i < chsize; i++)
      ch_gat[c][i] = gat_inverse_exact(ch_gat[c][i] * sg);

    DT_OMP_FOR()
    for(int row = row_offset; row < height; row += 2)
      for(int col = col_offset; col < width; col += 2)
        out[(size_t)row * width + col]
          = ch_gat[c][((row - row_offset) / 2) * halfwidth + (col - col_offset) / 2];
  }

  /* === DEBUG: output data statistics per Bayer channel === */
  {
    float ch_min[4] = {1e30f, 1e30f, 1e30f, 1e30f};
    float ch_max[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    double ch_sum[4] = {0, 0, 0, 0};
    int ch_neg[4] = {0, 0, 0, 0};
    for(int c = 0; c < 4; c++)
    {
      const int row_off = c & 1, col_off = (c >> 1) & 1;
      int cnt = 0;
      for(int row = row_off; row < height; row += 2)
        for(int col = col_off; col < width; col += 2)
        {
          const float v = out[(size_t)row * width + col];
          if(v < ch_min[c]) ch_min[c] = v;
          if(v > ch_max[c]) ch_max[c] = v;
          ch_sum[c] += (double)v;
          if(v < 0.0f) ch_neg[c]++;
          cnt++;
        }
      ch_sum[c] /= cnt;
    }
    fprintf(stderr, "[rawdenoise] DEBUG OUTPUT (after inv-GAT):\n");
    for(int c = 0; c < 4; c++)
      fprintf(stderr, "[rawdenoise]   ch[%d]: min=%.6f max=%.6f mean=%.6f neg=%d\n",
              c, ch_min[c], ch_max[c], (float)ch_sum[c], ch_neg[c]);
  }

  /* === DEBUG DIAG-C: dark area final output vs input === */
  {
    const float dark_thresh = 0.002f;
    int dark_n = 0;
    double in_sum[4] = {0}, out_sum[4] = {0};
    double in_sum_sq[4] = {0}, out_sum_sq[4] = {0};
    for(int hy = 0; hy < halfheight; hy++)
      for(int hx = 0; hx < halfwidth; hx++)
      {
        const int fy = hy * 2, fx = hx * 2;
        const float iv[4] = {
          in[(size_t)fy * width + fx],
          in[(size_t)(fy + 1) * width + fx],
          in[(size_t)fy * width + fx + 1],
          in[(size_t)(fy + 1) * width + fx + 1]
        };
        if((iv[0] + iv[1] + iv[2] + iv[3]) * 0.25f >= dark_thresh) continue;
        const float ov[4] = {
          out[(size_t)fy * width + fx],
          out[(size_t)(fy + 1) * width + fx],
          out[(size_t)fy * width + fx + 1],
          out[(size_t)(fy + 1) * width + fx + 1]
        };
        dark_n++;
        for(int c = 0; c < 4; c++)
        {
          in_sum[c] += iv[c]; out_sum[c] += ov[c];
          in_sum_sq[c] += (double)iv[c] * iv[c];
          out_sum_sq[c] += (double)ov[c] * ov[c];
        }
      }
    if(dark_n > 0)
    {
      const double n = dark_n;
      const float ig = (float)((in_sum[1] + in_sum[2]) / (2.0 * n));
      const float og = (float)((out_sum[1] + out_sum[2]) / (2.0 * n));
      fprintf(stderr, "[rawdenoise] DIAG-C dark output (luma<%.4f): %d blocks\n",
              dark_thresh, dark_n);
      fprintf(stderr, "[rawdenoise]   OUTPUT mean: ch0=%.7f ch1=%.7f ch2=%.7f ch3=%.7f\n",
              (float)(out_sum[0]/n), (float)(out_sum[1]/n),
              (float)(out_sum[2]/n), (float)(out_sum[3]/n));
      fprintf(stderr, "[rawdenoise]   OUTPUT stdv: ch0=%.7f ch1=%.7f ch2=%.7f ch3=%.7f\n",
              (float)sqrt(out_sum_sq[0]/n - (out_sum[0]/n)*(out_sum[0]/n)),
              (float)sqrt(out_sum_sq[1]/n - (out_sum[1]/n)*(out_sum[1]/n)),
              (float)sqrt(out_sum_sq[2]/n - (out_sum[2]/n)*(out_sum[2]/n)),
              (float)sqrt(out_sum_sq[3]/n - (out_sum[3]/n)*(out_sum[3]/n)));
      fprintf(stderr, "[rawdenoise]   OUTPUT B/G=%.4f R/G=%.4f (G=%.7f)\n",
              (float)(out_sum[0]/n) / fmaxf(og, 1e-10f),
              (float)(out_sum[3]/n) / fmaxf(og, 1e-10f), og);
      fprintf(stderr, "[rawdenoise]   INPUT  B/G=%.4f R/G=%.4f (G=%.7f)\n",
              (float)(in_sum[0]/n) / fmaxf(ig, 1e-10f),
              (float)(in_sum[3]/n) / fmaxf(ig, 1e-10f), ig);
      fprintf(stderr, "[rawdenoise]   RATIO CHANGE: B/G %.4f→%.4f (%+.1f%%)  R/G %.4f→%.4f (%+.1f%%)\n",
              (float)(in_sum[0]/n) / fmaxf(ig, 1e-10f),
              (float)(out_sum[0]/n) / fmaxf(og, 1e-10f),
              100.0 * ((out_sum[0]/n) / fmaxf(og, 1e-10) / fmaxf((in_sum[0]/n) / fmaxf(ig, 1e-10), 1e-10) - 1.0),
              (float)(in_sum[3]/n) / fmaxf(ig, 1e-10f),
              (float)(out_sum[3]/n) / fmaxf(og, 1e-10f),
              100.0 * ((out_sum[3]/n) / fmaxf(og, 1e-10) / fmaxf((in_sum[3]/n) / fmaxf(ig, 1e-10), 1e-10) - 1.0));
    }
  }

  fprintf(stderr, "[rawdenoise] RAW L/C BM3D: done\n");

  for(int c = 0; c < 4; c++) { dt_free_align(ch_gat[c]); ch_gat[c] = NULL; }
  return;

cleanup_rawlc:
  for(int c = 0; c < 4; c++) dt_free_align(ch_gat[c]);
  dt_free_align(luma);
  dt_free_align(chroma1);
  dt_free_align(chroma2);
  dt_free_align(chroma3);
  dt_free_align(luma_pilot);
  dt_free_align(c1_pilot);
  dt_free_align(c2_pilot);
  dt_free_align(c3_pilot);
  dt_free_align(luma_out);
  dt_free_align(c1_out);
  dt_free_align(c2_out);
  dt_free_align(c3_out);
  dt_free_align(L_fullres_noisy);
  dt_free_align(L_fullres_pilot);
  dt_free_align(L_fullres_out);
}

/* ─── END rawdenoise.c core ─── */

/* ─── main() ────────────────────────────────────────── */
int main(int argc, char **argv)
{
  if(argc < 5)
  {
    fprintf(stderr,
      "Usage: %s input.bin output.bin width height\n"
      "       [method] [strength] [luma_str] [chroma_str]\n"
      "       [alpha] [sigma_sq]\n"
      "\n"
      "  method:  'ours' (RAW L/C BM3D, default)\n"
      "           'bm3dcfa' (BM3D-CFA comparison)\n"
      "  luma_str:   sigma_L for Wiener (default 0.5)\n"
      "  chroma_str: sigma_C for Wiener (default 1.0)\n"
      "  alpha:      P-G shot noise gain (auto if <= 0)\n"
      "  sigma_sq:   read noise variance (auto if <= 0)\n",
      argv[0]);
    return 1;
  }

  const char *input_file = argv[1];
  const char *output_file = argv[2];
  const int width = atoi(argv[3]);
  const int height = atoi(argv[4]);

  const char *method = (argc > 5) ? argv[5] : "ours";
  const float strength = (argc > 6) ? (float)atof(argv[6]) : 1.0f;
  const float luma_str = (argc > 7) ? (float)atof(argv[7]) : 0.5f;
  const float chroma_str = (argc > 8) ? (float)atof(argv[8]) : 1.0f;
  const float known_alpha = (argc > 9) ? (float)atof(argv[9]) : -1.0f;
  const float known_sigma_sq = (argc > 10) ? (float)atof(argv[10]) : -1.0f;

  if(width <= 0 || height <= 0)
  {
    fprintf(stderr, "Invalid dimensions: %dx%d\n", width, height);
    return 1;
  }

  fprintf(stderr, "Raw Denoiser Standalone v5 (darktable rawdenoise.c)\n");
  fprintf(stderr, "  Input:  %s (%dx%d)\n", input_file, width, height);
  fprintf(stderr, "  Method: %s\n", method);
  fprintf(stderr, "  Params: strength=%.2f luma=%.2f chroma=%.2f\n",
          strength, luma_str, chroma_str);
  fprintf(stderr, "  Noise:  alpha=%.6f sigma_sq=%.8f (%s)\n",
          known_alpha, known_sigma_sq,
          known_alpha > 0.0f ? "KNOWN" : "BLIND");

  /* Initialize */
  init_kaiser_window();
  init_dct_basis();
  init_dct4_basis();

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

  if(strcmp(method, "bm3dcfa") == 0)
  {
    gat_bm3d_denoise_cfa(in, out, &roi, strength, 0,
                          known_alpha, known_sigma_sq);
  }
  else /* "ours" = RAW L/C BM3D */
  {
    gat_bm3d_denoise_rawlc(in, out, &roi, luma_str, chroma_str, 0,
                            known_alpha, known_sigma_sq);
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
