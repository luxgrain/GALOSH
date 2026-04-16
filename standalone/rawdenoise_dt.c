/*
 * Standalone raw denoiser - extracted from darktable rawdenoise.c (culmination version)
 * Features: V3 noise estimation, Foi exact inverse GAT, unified Step2, NL-Means Step1
 *           SSE-accelerated DCT, patch mean pre-filter, iterative VST refinement
 *
 * Usage: rawdenoise_dt input.bin output.bin width height
 *        [method] [strength] [luma_str] [chroma_str] [alpha] [sigma_sq]
 *        [iterations] [search_window]
 *
 * Input/output: 32-bit float raw Bayer mosaic (RGGB), row-major
 *
 * Build (MSYS2/MinGW):
 *   gcc -O2 -march=native -msse2 -fopenmp -lm -o rawdenoise_dt rawdenoise_dt.c
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

/* ─── darktable compatibility stubs ─────────────────── */
#define DT_OMP_FOR() _Pragma("omp parallel for schedule(static)")
#define DT_OMP_FOR_NUM(n) _Pragma("omp parallel for schedule(static)")

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

static inline float fast_expf(float x) {
    if(x < -80.0f) return 0.0f;
    if(x > 80.0f) return expf(80.0f);
    return expf(x);
}

/* ─── BM3D Constants ────────────────────────────────── */
#define BM3D_PATCH_SIZE 8
#define BM3D_PATCH_PIXELS (BM3D_PATCH_SIZE * BM3D_PATCH_SIZE)
#define BM3D_STEP         2
#define BM3D_STEP1_STRIDE 4
#define BM3D_SEARCH_RAD   16
#define BM3D_MAX_MATCHED  32
#define BM3D_TAU_MATCH    400.0f
#define BM3D_LAMBDA_3D    2.7f
#define BM3D_TAU_MATCH_W  400.0f
#define BM3D_SEARCH_RAD2  12
#define BM3D_EST_BLOCK    32

/* ─── Kaiser-Bessel Window ──────────────────────────── */
static const float bm3d_kaiser_1d[BM3D_PATCH_SIZE] = {
  0.34012f, 0.59885f, 0.84123f, 0.97659f,
  0.97659f, 0.84123f, 0.59885f, 0.34012f
};
static float bm3d_kaiser_2d[BM3D_PATCH_PIXELS];
static void init_kaiser_window(void)
{
  for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
    for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++)
      bm3d_kaiser_2d[dy * BM3D_PATCH_SIZE + dx] = bm3d_kaiser_1d[dy] * bm3d_kaiser_1d[dx];
}

/* ─── DCT Basis ─────────────────────────────────────── */
static float dct_basis[BM3D_PATCH_SIZE][BM3D_PATCH_SIZE];
static float dct_basis_t[BM3D_PATCH_SIZE][BM3D_PATCH_SIZE];
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
      dct_basis_t[n][k] = dct_basis[k][n];
    }
  }
  dct_basis_initialized = 1;
}

/* ─── SSE-accelerated 1D DCT ───────────────────────── */
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

static int compare_floats_bm3d(const void *a, const void *b)
{
  const float fa = *(const float *)a, fb = *(const float *)b;
  /* NaN-safe: NaN sorts to end */
  if(fa != fa) return 1;
  if(fb != fb) return -1;
  return (fa > fb) - (fa < fb);
}

/* O(n) quickselect for median finding — replaces qsort+median */
static float quick_select_kth(float *arr, int n, int k)
{
  int lo = 0, hi = n - 1;
  while(lo < hi)
  {
    const float pivot = arr[(lo + hi) / 2];
    int i = lo, j = hi;
    while(i <= j)
    {
      while(arr[i] < pivot) i++;
      while(arr[j] > pivot) j--;
      if(i <= j) { const float t = arr[i]; arr[i] = arr[j]; arr[j] = t; i++; j--; }
    }
    if(k <= j) hi = j;
    else if(k >= i) lo = i;
    else break;
  }
  return arr[k];
}

static inline float quick_select_median(float *arr, int n)
{
  return quick_select_kth(arr, n, n / 2);
}

/* ─── Noise Estimation (V3) ─────────────────────────── */
typedef struct bm3d_noise_params_t { float alpha, sigma_sq; } bm3d_noise_params_t;

static bm3d_noise_params_t bm3d_estimate_noise(const float *raw,
                                                const int width, const int height)
{
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

    quick_select_kth(sort_buf, cnt, cnt / 5);
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

  /* Step 4: robust WLS fit with Huber M-estimator */
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

  /* Step 5: sigma_sq from darkest pixels */
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
      quick_select_kth(samp, ns, ns / 10);
      const float dark_thresh = samp[ns / 10];
      const float dark_max = dark_thresh + 0.02f;

      float *dark_laps = dt_alloc_align_float(samp_max);
      if(dark_laps)
      {
        int ndl = 0;
        for(int ch = 0; ch < 4 && ndl < samp_max; ch++)
        {
          const int dy0 = offsets[ch][0], dx0 = offsets[ch][1];
          for(int y = 0; y < halfheight && ndl < samp_max; y++)
            for(int x = 0; x < halfwidth - 2 && ndl < samp_max; x++)
            {
              const float v0 = raw[(2*y+dy0) * width + (2*x+dx0)];
              const float v1 = raw[(2*y+dy0) * width + (2*(x+1)+dx0)];
              const float v2 = raw[(2*y+dy0) * width + (2*(x+2)+dx0)];
              if(v0 > dark_max || v1 > dark_max || v2 > dark_max) continue;
              dark_laps[ndl++] = fabsf(v0 - 2.0f*v1 + v2);
            }
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

/* ─── GAT (Generalized Anscombe Transform) ──────────── */
static inline float gat_forward(const float x, const float alpha, const float sigma_sq)
{
  const float x_safe = (x == x) ? fmaxf(x, 0.0f) : 0.0f; /* NaN guard */
  return (2.0f / alpha) * sqrtf(fmaxf(alpha * x_safe + 0.375f * alpha * alpha + sigma_sq, 0.0f));
}

static inline float gat_inverse(const float D, const float alpha, const float sigma_sq)
{
  if(D <= 0.0f) return 0.0f;
  const float D_inv = 1.0f / fmaxf(D, 1e-8f);
  const float y = 0.25f * D * D + 0.25f * 1.2247448713916f * D_inv
                - 11.0f / 8.0f * D_inv * D_inv + 5.0f / 8.0f * 1.2247448713916f * D_inv * D_inv * D_inv
                - 1.0f / 8.0f;
  return fmaxf(alpha * y - sigma_sq / alpha, 0.0f);
}

/* ─── Foi exact unbiased inverse GAT (Makitalo & Foi 2013) ─── */
#define GAT_INV_TABLE_SIZE 4096

typedef struct gat_inv_table_t
{
  float d[GAT_INV_TABLE_SIZE];
  float x[GAT_INV_TABLE_SIZE];
  float d_min, d_max;
  int valid;
} gat_inv_table_t;

static gat_inv_table_t gat_inv_table = { .valid = 0 };

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
  const double sig = sqrt(sq);

  for(int i = 0; i < GAT_INV_TABLE_SIZE; i++)
  {
    const double x_val = (double)i / (double)(GAT_INV_TABLE_SIZE - 1);
    const double lambda = x_val / a;

    double expected_gat = 0.0;
    const int k_max = (int)(lambda + 8.0 * sqrt(fmax(lambda, 1.0))) + 20;
    double log_prob = -lambda;

    for(int k = 0; k <= k_max; k++)
    {
      if(k > 0) log_prob += log(lambda) - log((double)k);
      const double prob = exp(log_prob);
      if(prob < 1e-15 && k > (int)lambda + 1) break;

      const double base = (double)k * a * a + 0.375 * a * a + sq;
      double eg = 0.0;
      for(int g = 0; g < 10; g++)
      {
        const double z = 1.4142135623730951 * sig * gh_nodes[g];
        const double arg = base + a * z;
        if(arg > 0.0) eg += gh_weights[g] * sqrt(arg);
      }
      eg *= 0.5641895835477563;  /* 1/sqrt(pi) */
      expected_gat += prob * (2.0 / a) * eg;
    }

    gat_inv_table.x[i] = (float)x_val;
    gat_inv_table.d[i] = (float)expected_gat;
  }

  gat_inv_table.d_min = gat_inv_table.d[0];
  gat_inv_table.d_max = gat_inv_table.d[GAT_INV_TABLE_SIZE - 1];
  gat_inv_table.valid = 1;
  fprintf(stderr, "[rawdenoise] Foi inverse table: D range [%.4f, %.4f] alpha=%.6f sigma_sq=%.8f\n",
          gat_inv_table.d_min, gat_inv_table.d_max, alpha, sigma_sq);
}

static inline float gat_inverse_exact(const float D)
{
  if(!gat_inv_table.valid) return 0.0f;
  if(D <= gat_inv_table.d_min) return 0.0f;
  if(D >= gat_inv_table.d_max) return 1.0f;

  int lo = 0, hi = GAT_INV_TABLE_SIZE - 1;
  while(lo < hi - 1)
  {
    const int mid = (lo + hi) >> 1;
    if(gat_inv_table.d[mid] <= D) lo = mid;
    else hi = mid;
  }

  const float d0 = gat_inv_table.d[lo], d1 = gat_inv_table.d[hi];
  const float t = (D - d0) / fmaxf(d1 - d0, 1e-10f);
  return fmaxf(gat_inv_table.x[lo] + t * (gat_inv_table.x[hi] - gat_inv_table.x[lo]), 0.0f);
}

/* ─── GAT sigma estimation ──────────────────────────── */
static float estimate_gat_sigma(const float *data, const int width, const int height)
{
  const int n_samples = MIN(width * height / 3, 200000);
  const int step = MAX(1, (width * (height - 1)) / n_samples);
  float *abs_laps = dt_alloc_align_float(n_samples + 1);
  if(!abs_laps) return 1.0f;

  int count = 0;
  for(int y = 0; y < height && count < n_samples; y++)
  {
    const float *row = data + (size_t)y * width;
    for(int x = 0; x < width - 2 && count < n_samples; x += step)
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

static int bm3d_next_pow2(int n)
{
  if(n <= 1) return 1;
  if(n <= 2) return 2;
  if(n <= 4) return 4;
  if(n <= 8) return 8;
  if(n <= 16) return 16;
  return 32;
}

/* ─── SSE patch SSD with early termination ──────────── */
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
    const __m128 a0 = _mm_loadu_ps(p1);
    const __m128 b0 = _mm_loadu_ps(p2);
    const __m128 d0 = _mm_sub_ps(a0, b0);
    acc = _mm_add_ps(acc, _mm_mul_ps(d0, d0));
    const __m128 a1 = _mm_loadu_ps(p1 + 4);
    const __m128 b1 = _mm_loadu_ps(p2 + 4);
    const __m128 d1 = _mm_sub_ps(a1, b1);
    acc = _mm_add_ps(acc, _mm_mul_ps(d1, d1));
    if((dy & 1) == 1)
    {
      __m128 s = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
      s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
      float ssd;
      _mm_store_ss(&ssd, s);
      if(ssd >= threshold) return ssd;
    }
  }
  __m128 s = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  float ssd;
  _mm_store_ss(&ssd, s);
  return ssd;
}

typedef struct bm3d_match_t { float dist; int row, col; } bm3d_match_t;

/* For N <= 32, insertion sort beats qsort by 2-3x */
static void sort_matches(bm3d_match_t *m, int n)
{
  for(int i = 1; i < n; i++)
  {
    bm3d_match_t key = m[i];
    int j = i - 1;
    while(j >= 0 && m[j].dist > key.dist) { m[j + 1] = m[j]; j--; }
    m[j + 1] = key;
  }
}

/* ─── Patch mean precomputation (integral image) ────── */
static float *bm3d_precompute_patch_means(const float *img, const int width, const int height)
{
  const int rmax = height - BM3D_PATCH_SIZE;
  const int cmax = width - BM3D_PATCH_SIZE;
  const int mwidth = cmax + 1;
  const int mheight = rmax + 1;
  float *means = dt_alloc_align_float((size_t)mwidth * mheight);
  if(!means) return NULL;

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

/* ─── Noise variance map from residual ──────────────── */
static float *bm3d_compute_noise_variance_map(const float *noisy, const float *pilot,
                                               const int width, const int height,
                                               const int window_radius)
{
  const size_t npixels = (size_t)width * height;
  const size_t iw = (size_t)(width + 1);
  double *integral = calloc(iw * (height + 1), sizeof(double));
  if(!integral) return NULL;

  for(int y = 0; y < height; y++)
    for(int x = 0; x < width; x++)
    {
      const float r = noisy[(size_t)y * width + x] - pilot[(size_t)y * width + x];
      integral[(y + 1) * iw + (x + 1)] = (double)(r * r)
        + integral[y * iw + (x + 1)]
        + integral[(y + 1) * iw + x]
        - integral[y * iw + x];
    }

  float *varmap = dt_alloc_align_float(npixels);
  if(!varmap) { free(integral); return NULL; }

  DT_OMP_FOR()
  for(int y = 0; y < height; y++)
    for(int x = 0; x < width; x++)
    {
      const int y0 = MAX(0, y - window_radius);
      const int y1 = MIN(height, y + window_radius + 1);
      const int x0 = MAX(0, x - window_radius);
      const int x1 = MIN(width, x + window_radius + 1);
      const double sum = integral[y1 * iw + x1] - integral[y0 * iw + x1]
                       - integral[y1 * iw + x0] + integral[y0 * iw + x0];
      const int count = (y1 - y0) * (x1 - x0);
      varmap[(size_t)y * width + x] = (float)(sum / count);
    }

  free(integral);
  return varmap;
}

/* ─── BM3D Step 1: NL-Means pilot estimation ────────── */
static void bm3d_step1(const float *restrict input, float *restrict output,
                       const int width, const int height, const float sigma)
{
  const float tau = BM3D_TAU_MATCH * sigma * sigma;
  const float h_sq = sigma * sigma;
  const size_t npixels = (size_t)width * height;

  const int step1 = BM3D_STEP1_STRIDE;
  const int rmax = height - BM3D_PATCH_SIZE, cmax = width - BM3D_PATCH_SIZE;
  const int mwidth = cmax + 1;

  float *pmeans = bm3d_precompute_patch_means(input, width, height);
  const float mean_thr = sqrtf(tau / (float)BM3D_PATCH_PIXELS);

#ifdef _OPENMP
  const int max_threads = omp_get_max_threads();
#else
  const int max_threads = 1;
#endif
  const int nthreads = MIN(max_threads, 16);

  float **t_numer = calloc(nthreads, sizeof(float *));
  float **t_denom = calloc(nthreads, sizeof(float *));
  if(!t_numer || !t_denom)
  {
    free(t_numer); free(t_denom);
    dt_free_align(pmeans);
    memcpy(output, input, sizeof(float) * npixels);
    return;
  }

  gboolean alloc_ok = TRUE;
  for(int t = 0; t < nthreads; t++)
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
    memcpy(output, input, sizeof(float) * npixels);
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
    float *my_numer = t_numer[tid];
    float *my_denom = t_denom[tid];

    for(int ref_c = 0; ref_c <= cmax; ref_c += step1)
    {
      bm3d_match_t matches[BM3D_MAX_MATCHED];
      int nmatches = 1;
      matches[0] = (bm3d_match_t){ 0.0f, ref_r, ref_c };
      float worst_dist = tau;

      const float ref_mean = pmeans ? pmeans[ref_r * mwidth + ref_c] : 0.0f;

      const int sr0 = MAX(0, ref_r - BM3D_SEARCH_RAD), sr1 = MIN(rmax, ref_r + BM3D_SEARCH_RAD);
      const int sc0 = MAX(0, ref_c - BM3D_SEARCH_RAD), sc1 = MIN(cmax, ref_c + BM3D_SEARCH_RAD);

      for(int sr = sr0; sr <= sr1; sr += BM3D_STEP)
        for(int sc = sc0; sc <= sc1; sc += BM3D_STEP)
        {
          if(sr == ref_r && sc == ref_c) continue;

          if(pmeans)
          {
            const float dm = ref_mean - pmeans[sr * mwidth + sc];
            if(dm > mean_thr || dm < -mean_thr) continue;
          }

          const float d = bm3d_patch_ssd(input, width, ref_r, ref_c, sr, sc, worst_dist);
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

      /* NL-Means weighted averaging with Salmon correction */
      const float h_norm = 2.0f * BM3D_PATCH_PIXELS * h_sq;
      const float noise_floor = h_norm;
      float avg_patch[BM3D_PATCH_PIXELS];
      memset(avg_patch, 0, sizeof(avg_patch));
      float total_weight = 0.0f;

      for(int i = 0; i < nmatches; i++)
      {
        const float corrected_dist = fmaxf(matches[i].dist - noise_floor, 0.0f);
        const float w = fast_expf(-corrected_dist / fmaxf(h_norm, 1e-8f));
        total_weight += w;
        const int mr = matches[i].row, mc = matches[i].col;
        for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
        {
          const float *src = input + (size_t)(mr + dy) * width + mc;
          float *dst = avg_patch + dy * BM3D_PATCH_SIZE;
          for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++)
            dst[dx] += w * src[dx];
        }
      }

      if(total_weight > 0.0f)
      {
        const float inv_w = 1.0f / total_weight;
        for(int p = 0; p < BM3D_PATCH_PIXELS; p++)
          avg_patch[p] *= inv_w;
      }

      const float agg_weight = (float)nmatches;
      for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
        for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++)
        {
          const size_t pos = (size_t)(ref_r + dy) * width + (ref_c + dx);
          const float kw = bm3d_kaiser_2d[dy * BM3D_PATCH_SIZE + dx];
          my_numer[pos] += agg_weight * kw * avg_patch[dy * BM3D_PATCH_SIZE + dx];
          my_denom[pos] += agg_weight * kw;
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
    output[i] = (d > 0.0f) ? n / d : input[i];
  }

  for(int t = 0; t < nthreads; t++)
  { dt_free_align(t_numer[t]); dt_free_align(t_denom[t]); }
  free(t_numer); free(t_denom);
  dt_free_align(pmeans);
}

/* ─── Unified Step2: single matching + 4-channel Wiener ─── */
static void bm3d_step2_lc_unified(
    const float *const *ch_noisy,
    const float *const *ch_pilot,
    float *const *ch_output,
    const float *luma_pilot_match,
    const int width, const int height,
    const float sigma[4],
    const float *noise_varmap,
    const int pilot_neff_luma,
    const int pilot_neff_chroma)
{
  const float tau = BM3D_TAU_MATCH_W * sigma[0] * sigma[0];
  const size_t npixels = (size_t)width * height;

  const int rmax = height - BM3D_PATCH_SIZE, cmax = width - BM3D_PATCH_SIZE;
  const int mwidth = cmax + 1;

  float *pmeans = bm3d_precompute_patch_means(luma_pilot_match, width, height);
  const float mean_thr = sqrtf(tau / (float)BM3D_PATCH_PIXELS);

#ifdef _OPENMP
  const int max_threads = omp_get_max_threads();
#else
  const int max_threads = 1;
#endif
  const int nthreads = MIN(max_threads, 16);

  float **t_numer[4], **t_denom[4];
  for(int c = 0; c < 4; c++)
  {
    t_numer[c] = calloc(nthreads, sizeof(float *));
    t_denom[c] = calloc(nthreads, sizeof(float *));
  }

  gboolean alloc_ok = TRUE;
  for(int t = 0; t < nthreads && alloc_ok; t++)
  {
    for(int c = 0; c < 4; c++)
    {
      t_numer[c][t] = dt_alloc_align_float(npixels);
      t_denom[c][t] = dt_alloc_align_float(npixels);
      if(!t_numer[c][t] || !t_denom[c][t]) { alloc_ok = FALSE; break; }
      memset(t_numer[c][t], 0, sizeof(float) * npixels);
      memset(t_denom[c][t], 0, sizeof(float) * npixels);
    }
  }
  if(!alloc_ok)
  {
    for(int c = 0; c < 4; c++)
    {
      for(int t = 0; t < nthreads; t++)
      { dt_free_align(t_numer[c][t]); dt_free_align(t_denom[c][t]); }
      free(t_numer[c]); free(t_denom[c]);
    }
    dt_free_align(pmeans);
    for(int c = 0; c < 4; c++)
      memcpy(ch_output[c], ch_noisy[c], sizeof(float) * npixels);
    return;
  }

  const int chunk_s2 = MAX(4, (rmax + BM3D_STEP) / BM3D_STEP / (nthreads * 8));
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, chunk_s2)
#endif
  for(int ref_r = 0; ref_r <= rmax; ref_r += BM3D_STEP)
  {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif

    for(int ref_c = 0; ref_c <= cmax; ref_c += BM3D_STEP)
    {
      bm3d_match_t matches[BM3D_MAX_MATCHED];
      int nmatches = 1;
      matches[0] = (bm3d_match_t){ 0.0f, ref_r, ref_c };
      float worst_dist = tau;

      const float ref_mean = pmeans ? pmeans[ref_r * mwidth + ref_c] : 0.0f;

      const int sr0 = MAX(0, ref_r - BM3D_SEARCH_RAD), sr1 = MIN(rmax, ref_r + BM3D_SEARCH_RAD);
      const int sc0 = MAX(0, ref_c - BM3D_SEARCH_RAD), sc1 = MIN(cmax, ref_c + BM3D_SEARCH_RAD);

      for(int sr = sr0; sr <= sr1; sr += BM3D_STEP)
        for(int sc = sc0; sc <= sc1; sc += BM3D_STEP)
        {
          if(sr == ref_r && sc == ref_c) continue;

          if(pmeans)
          {
            const float dm = ref_mean - pmeans[sr * mwidth + sc];
            if(dm > mean_thr || dm < -mean_thr) continue;
          }

          const float d = bm3d_patch_ssd(luma_pilot_match, width, ref_r, ref_c, sr, sc, worst_dist);
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

      /* Wiener filter each of 4 channels using same match positions */
      for(int c = 0; c < 4; c++)
      {
        const float sigma_sq = sigma[c] * sigma[c];
        const int neff = (c == 0) ? pilot_neff_luma : pilot_neff_chroma;

        const float local_sigma_sq = fmaxf(
          (c == 0 && noise_varmap)
            ? noise_varmap[(size_t)(ref_r + BM3D_PATCH_SIZE / 2) * width
                           + (ref_c + BM3D_PATCH_SIZE / 2)]
            : sigma_sq,
          1e-6f);
        const float sigma_sq_pilot = local_sigma_sq / fmaxf((float)neff, 1.0f);

        float pilot_dct[BM3D_MAX_MATCHED][BM3D_PATCH_PIXELS];
        float noisy_dct[BM3D_MAX_MATCHED][BM3D_PATCH_PIXELS];
        float patch_buf[BM3D_PATCH_PIXELS];

        for(int i = 0; i < nmatches; i++)
        {
          for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
            memcpy(patch_buf + dy * BM3D_PATCH_SIZE,
                   ch_pilot[c] + (matches[i].row + dy) * width + matches[i].col,
                   BM3D_PATCH_SIZE * sizeof(float));
          dct2d_forward(patch_buf, pilot_dct[i]);

          for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
            memcpy(patch_buf + dy * BM3D_PATCH_SIZE,
                   ch_noisy[c] + (matches[i].row + dy) * width + matches[i].col,
                   BM3D_PATCH_SIZE * sizeof(float));
          dct2d_forward(patch_buf, noisy_dct[i]);
        }
        for(int i = nmatches; i < group_size; i++)
        {
          memcpy(pilot_dct[i], pilot_dct[0], sizeof(float) * BM3D_PATCH_PIXELS);
          memcpy(noisy_dct[i], noisy_dct[0], sizeof(float) * BM3D_PATCH_PIXELS);
        }

        /* 3D Hadamard + Bayesian Wiener */
        float had_pilot[BM3D_MAX_MATCHED], had_noisy[BM3D_MAX_MATCHED];
        float wiener_energy = 0.0f;
        for(int p = 0; p < BM3D_PATCH_PIXELS; p++)
        {
          for(int i = 0; i < group_size; i++) had_pilot[i] = pilot_dct[i][p];
          for(int i = 0; i < group_size; i++) had_noisy[i] = noisy_dct[i][p];
          hadamard_transform(had_pilot, group_size);
          hadamard_transform(had_noisy, group_size);

          for(int i = 0; i < group_size; i++)
          {
            const float s2 = had_pilot[i] * had_pilot[i];
            const float signal_est = s2 * s2 / fmaxf(s2 + sigma_sq_pilot, 1e-10f);
            const float denom = signal_est + local_sigma_sq;
            const float w = denom > 1e-10f ? signal_est / denom : 0.0f;
            had_noisy[i] *= w;
            wiener_energy += w * w;
          }

          hadamard_transform(had_noisy, group_size);
          for(int i = 0; i < group_size; i++) noisy_dct[i][p] = had_noisy[i];
        }

        const float weight = 1.0f / fmaxf(wiener_energy, 1e-6f);
        float *my_numer = t_numer[c][tid];
        float *my_denom = t_denom[c][tid];
        for(int i = 0; i < nmatches; i++)
        {
          float denoised[BM3D_PATCH_PIXELS];
          dct2d_inverse(noisy_dct[i], denoised);
          const int mr = matches[i].row, mc = matches[i].col;
          for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
            for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++)
            {
              const size_t pos = (size_t)(mr + dy) * width + (mc + dx);
              const float kw = bm3d_kaiser_2d[dy * BM3D_PATCH_SIZE + dx];
              my_numer[pos] += weight * kw * denoised[dy * BM3D_PATCH_SIZE + dx];
              my_denom[pos] += weight * kw;
            }
        }
      } /* end for c */
    } /* end ref_c */
  } /* end ref_r */

  /* Merge per-thread buffers */
  for(int c = 0; c < 4; c++)
  {
    DT_OMP_FOR()
    for(int i = 0; i < (int)npixels; i++)
    {
      float n = 0.0f, d = 0.0f;
      for(int t = 0; t < nthreads; t++)
      { n += t_numer[c][t][i]; d += t_denom[c][t][i]; }
      ch_output[c][i] = (d > 0.0f) ? n / d : ch_noisy[c][i];
    }
  }

  for(int c = 0; c < 4; c++)
  {
    for(int t = 0; t < nthreads; t++)
    { dt_free_align(t_numer[c][t]); dt_free_align(t_denom[c][t]); }
    free(t_numer[c]); free(t_denom[c]);
  }
  dt_free_align(pmeans);
}

/* ─── Main denoise pipeline ─────────────────────────── */
static void gat_bm3d_denoise(const float *in, float *out,
                             const int width, const int height,
                             const float strength, const int iterations,
                             const float luma_strength, const float chroma_strength,
                             const float override_alpha, const float override_sigma_sq)
{
  memcpy(out, in, sizeof(float) * width * height);
  if(strength <= 0.0f) return;

  const int niters = CLAMP(iterations, 1, 4);

  /* Estimate or use provided noise parameters */
  bm3d_noise_params_t np;
  if(override_alpha > 0 && override_sigma_sq >= 0)
  {
    np.alpha = override_alpha;
    np.sigma_sq = override_sigma_sq;
    fprintf(stderr, "[rawdenoise] Using provided noise params: alpha=%.6f sigma_sq=%.8f\n",
            np.alpha, np.sigma_sq);
  }
  else
  {
    np = bm3d_estimate_noise(in, width, height);
  }

  /* Build Foi exact inverse GAT lookup table */
  gat_build_inverse_table(np.alpha, np.sigma_sq);

  const int halfwidth = (width + 1) / 2;
  const int halfheight = (height + 1) / 2;
  if(halfwidth < BM3D_PATCH_SIZE * 2 || halfheight < BM3D_PATCH_SIZE * 2) return;

  const size_t chsize = (size_t)halfwidth * halfheight;

  float *luma = NULL, *chroma1 = NULL, *chroma2 = NULL, *chroma3 = NULL;
  float *luma_pilot = NULL, *luma_denoised = NULL;
  float *c1_denoised = NULL, *c2_denoised = NULL, *c3_denoised = NULL;
  float *noise_varmap = NULL;

  /* Phase 1: Extract, GAT, normalize all 4 Bayer channels */
  float *ch_gat[4] = { NULL };
  float *ch_original[4] = { NULL };
  float sigma_gat_ch[4] = { 1.0f };

  for(int c = 0; c < 4; c++)
  {
    const int row_offset = c & 1, col_offset = (c >> 1) & 1;
    ch_gat[c] = dt_alloc_align_float(chsize);
    ch_original[c] = dt_alloc_align_float(chsize);
    if(!ch_gat[c] || !ch_original[c]) goto cleanup;

    DT_OMP_FOR()
    for(int row = row_offset; row < height; row += 2)
      for(int col = col_offset; col < width; col += 2)
        ch_gat[c][((row - row_offset) / 2) * halfwidth + (col - col_offset) / 2]
          = in[(size_t)row * width + col];

    memcpy(ch_original[c], ch_gat[c], sizeof(float) * chsize);

    DT_OMP_FOR()
    for(size_t i = 0; i < chsize; i++)
      ch_gat[c][i] = gat_forward(ch_gat[c][i], np.alpha, np.sigma_sq);

    sigma_gat_ch[c] = estimate_gat_sigma(ch_gat[c], halfwidth, halfheight);
    const float inv_sg = 1.0f / sigma_gat_ch[c];
    DT_OMP_FOR()
    for(size_t i = 0; i < chsize; i++)
      ch_gat[c][i] *= inv_sg;
  }

  fprintf(stderr, "[rawdenoise] alpha=%.8f sigma_sq=%.10f | GAT sigma=[%.4f %.4f %.4f %.4f] | "
                  "size=%dx%d | str=%.2f luma=%.2f chroma=%.2f\n",
          np.alpha, np.sigma_sq,
          sigma_gat_ch[0], sigma_gat_ch[1], sigma_gat_ch[2], sigma_gat_ch[3],
          halfwidth, halfheight, strength, luma_strength, chroma_strength);

  /* Phase 2: L/C decomposition */
  luma    = dt_alloc_align_float(chsize);
  chroma1 = dt_alloc_align_float(chsize);
  chroma2 = dt_alloc_align_float(chsize);
  chroma3 = dt_alloc_align_float(chsize);
  if(!luma || !chroma1 || !chroma2 || !chroma3) goto cleanup;

  DT_OMP_FOR()
  for(size_t i = 0; i < chsize; i++)
  {
    const float r = ch_gat[0][i], gr = ch_gat[1][i];
    const float gb = ch_gat[2][i], b = ch_gat[3][i];
    const float l = (r + gr + gb + b) * 0.25f;
    luma[i]    = l;
    chroma1[i] = r - l;
    chroma2[i] = b - l;
    chroma3[i] = (gr - gb) * 0.5f;
  }

  /* Phase 3: BM3D on luma/chroma */
  const float noise_sigma_l  = 0.5f;
  const float noise_sigma_c1 = 0.866f;
  const float noise_sigma_c2 = 0.866f;
  const float noise_sigma_c3 = 0.707f;

  DT_OMP_FOR()
  for(size_t i = 0; i < chsize; i++) luma[i]    /= noise_sigma_l;
  DT_OMP_FOR()
  for(size_t i = 0; i < chsize; i++) chroma1[i] /= noise_sigma_c1;
  DT_OMP_FOR()
  for(size_t i = 0; i < chsize; i++) chroma2[i] /= noise_sigma_c2;
  DT_OMP_FOR()
  for(size_t i = 0; i < chsize; i++) chroma3[i] /= noise_sigma_c3;

  const float sigma_l_bm3d  = 1.0f * strength * luma_strength;
  const float sigma_c_bm3d  = 1.0f * strength * chroma_strength;

  luma_pilot    = dt_alloc_align_float(chsize);
  luma_denoised = dt_alloc_align_float(chsize);
  c1_denoised   = dt_alloc_align_float(chsize);
  c2_denoised   = dt_alloc_align_float(chsize);
  c3_denoised   = dt_alloc_align_float(chsize);
  if(!luma_pilot || !luma_denoised || !c1_denoised || !c2_denoised || !c3_denoised)
    goto cleanup;

  bm3d_step1(luma, luma_pilot, halfwidth, halfheight, sigma_l_bm3d);

  noise_varmap = bm3d_compute_noise_variance_map(luma, luma_pilot, halfwidth, halfheight, 16);

  /* Unified Step2 */
  {
    const float *u_noisy[4]  = { luma,    chroma1, chroma2, chroma3 };
    const float *u_pilot[4]  = { luma_pilot, chroma1, chroma2, chroma3 };
    float *u_output[4]       = { luma_denoised, c1_denoised, c2_denoised, c3_denoised };
    const float u_sigma[4]   = { sigma_l_bm3d, sigma_c_bm3d, sigma_c_bm3d, sigma_c_bm3d };

    bm3d_step2_lc_unified(u_noisy, u_pilot, u_output, luma_pilot,
                           halfwidth, halfheight, u_sigma, noise_varmap, 8, 1);
  }

  dt_free_align(noise_varmap); noise_varmap = NULL;

  DT_OMP_FOR()
  for(size_t i = 0; i < chsize; i++) luma_denoised[i] *= noise_sigma_l;
  DT_OMP_FOR()
  for(size_t i = 0; i < chsize; i++) c1_denoised[i]   *= noise_sigma_c1;
  DT_OMP_FOR()
  for(size_t i = 0; i < chsize; i++) c2_denoised[i]   *= noise_sigma_c2;
  DT_OMP_FOR()
  for(size_t i = 0; i < chsize; i++) c3_denoised[i]   *= noise_sigma_c3;

  /* Phase 4: Inverse L/C transform */
  DT_OMP_FOR()
  for(size_t i = 0; i < chsize; i++)
  {
    const float l  = luma_denoised[i];
    const float c1 = c1_denoised[i];
    const float c2 = c2_denoised[i];
    const float c3 = c3_denoised[i];
    const float r  = l + c1;
    const float gr = l - 0.5f * c1 - 0.5f * c2 + c3;
    const float gb = l - 0.5f * c1 - 0.5f * c2 - c3;
    const float b  = l + c2;
    ch_gat[0][i] = isfinite(r)  ? r  : l;
    ch_gat[1][i] = isfinite(gr) ? gr : l;
    ch_gat[2][i] = isfinite(gb) ? gb : l;
    ch_gat[3][i] = isfinite(b)  ? b  : l;
  }

  dt_free_align(luma); luma = NULL;
  dt_free_align(chroma1); chroma1 = NULL;
  dt_free_align(chroma2); chroma2 = NULL;
  dt_free_align(chroma3); chroma3 = NULL;
  dt_free_align(luma_pilot); luma_pilot = NULL;
  dt_free_align(luma_denoised); luma_denoised = NULL;
  dt_free_align(c1_denoised); c1_denoised = NULL;
  dt_free_align(c2_denoised); c2_denoised = NULL;
  dt_free_align(c3_denoised); c3_denoised = NULL;

  /* Phase 5: Denormalize + Foi exact inverse GAT + write back */
  for(int c = 0; c < 4; c++)
  {
    const int row_offset = c & 1, col_offset = (c >> 1) & 1;

    DT_OMP_FOR()
    for(size_t i = 0; i < chsize; i++)
      ch_gat[c][i] *= sigma_gat_ch[c];

    DT_OMP_FOR()
    for(size_t i = 0; i < chsize; i++)
      ch_gat[c][i] = gat_inverse_exact(ch_gat[c][i]);

    DT_OMP_FOR()
    for(int row = row_offset; row < height; row += 2)
      for(int col = col_offset; col < width; col += 2)
        out[(size_t)row * width + col]
          = ch_gat[c][((row - row_offset) / 2) * halfwidth + (col - col_offset) / 2];
  }

  /* Iterations 2..N: iterative VST refinement */
  for(int k = 1; k < niters; k++)
  {
    const float iter_alpha = np.alpha * 0.5f;
    const float iter_sigma_sq = np.sigma_sq * 0.5f;

    gat_build_inverse_table(iter_alpha, iter_sigma_sq);

    for(int c = 0; c < 4; c++)
    {
      const int row_offset = c & 1, col_offset = (c >> 1) & 1;

      DT_OMP_FOR()
      for(int row = row_offset; row < height; row += 2)
        for(int col = col_offset; col < width; col += 2)
        {
          const size_t hi = ((row - row_offset) / 2) * halfwidth + (col - col_offset) / 2;
          ch_gat[c][hi] = 0.5f * (ch_original[c][hi] + out[(size_t)row * width + col]);
        }

      DT_OMP_FOR()
      for(size_t i = 0; i < chsize; i++)
        ch_gat[c][i] = gat_forward(ch_gat[c][i], iter_alpha, iter_sigma_sq);

      sigma_gat_ch[c] = estimate_gat_sigma(ch_gat[c], halfwidth, halfheight);
      const float inv_sg = 1.0f / sigma_gat_ch[c];
      DT_OMP_FOR()
      for(size_t i = 0; i < chsize; i++)
        ch_gat[c][i] *= inv_sg;
    }

    luma    = dt_alloc_align_float(chsize);
    chroma1 = dt_alloc_align_float(chsize);
    chroma2 = dt_alloc_align_float(chsize);
    chroma3 = dt_alloc_align_float(chsize);
    luma_pilot    = dt_alloc_align_float(chsize);
    luma_denoised = dt_alloc_align_float(chsize);
    c1_denoised   = dt_alloc_align_float(chsize);
    c2_denoised   = dt_alloc_align_float(chsize);
    c3_denoised   = dt_alloc_align_float(chsize);
    if(!luma || !chroma1 || !chroma2 || !chroma3 ||
       !luma_pilot || !luma_denoised || !c1_denoised || !c2_denoised || !c3_denoised)
      goto cleanup;

    DT_OMP_FOR()
    for(size_t i = 0; i < chsize; i++)
    {
      const float r = ch_gat[0][i], gr = ch_gat[1][i];
      const float gb = ch_gat[2][i], b = ch_gat[3][i];
      const float l = (r + gr + gb + b) * 0.25f;
      luma[i]    = l / noise_sigma_l;
      chroma1[i] = (r - l) / noise_sigma_c1;
      chroma2[i] = (b - l) / noise_sigma_c2;
      chroma3[i] = ((gr - gb) * 0.5f) / noise_sigma_c3;
    }

    bm3d_step1(luma, luma_pilot, halfwidth, halfheight, sigma_l_bm3d);
    noise_varmap = bm3d_compute_noise_variance_map(luma, luma_pilot, halfwidth, halfheight, 16);

    {
      const float *u_noisy[4]  = { luma,    chroma1, chroma2, chroma3 };
      const float *u_pilot[4]  = { luma_pilot, chroma1, chroma2, chroma3 };
      float *u_output[4]       = { luma_denoised, c1_denoised, c2_denoised, c3_denoised };
      const float u_sigma[4]   = { sigma_l_bm3d, sigma_c_bm3d, sigma_c_bm3d, sigma_c_bm3d };
      bm3d_step2_lc_unified(u_noisy, u_pilot, u_output, luma_pilot,
                             halfwidth, halfheight, u_sigma, noise_varmap, 8, 1);
    }
    dt_free_align(noise_varmap); noise_varmap = NULL;

    DT_OMP_FOR()
    for(size_t i = 0; i < chsize; i++)
    {
      luma_denoised[i] *= noise_sigma_l;
      c1_denoised[i]   *= noise_sigma_c1;
      c2_denoised[i]   *= noise_sigma_c2;
      c3_denoised[i]   *= noise_sigma_c3;
    }

    DT_OMP_FOR()
    for(size_t i = 0; i < chsize; i++)
    {
      const float l  = luma_denoised[i];
      const float cc1 = c1_denoised[i], cc2 = c2_denoised[i], cc3 = c3_denoised[i];
      const float r  = l + cc1;
      const float gr = l - 0.5f * cc1 - 0.5f * cc2 + cc3;
      const float gb = l - 0.5f * cc1 - 0.5f * cc2 - cc3;
      const float b  = l + cc2;
      ch_gat[0][i] = isfinite(r)  ? r  : l;
      ch_gat[1][i] = isfinite(gr) ? gr : l;
      ch_gat[2][i] = isfinite(gb) ? gb : l;
      ch_gat[3][i] = isfinite(b)  ? b  : l;
    }

    dt_free_align(luma); luma = NULL;
    dt_free_align(chroma1); chroma1 = NULL;
    dt_free_align(chroma2); chroma2 = NULL;
    dt_free_align(chroma3); chroma3 = NULL;
    dt_free_align(luma_pilot); luma_pilot = NULL;
    dt_free_align(luma_denoised); luma_denoised = NULL;
    dt_free_align(c1_denoised); c1_denoised = NULL;
    dt_free_align(c2_denoised); c2_denoised = NULL;
    dt_free_align(c3_denoised); c3_denoised = NULL;

    for(int c = 0; c < 4; c++)
    {
      const int row_offset = c & 1, col_offset = (c >> 1) & 1;
      DT_OMP_FOR()
      for(size_t i = 0; i < chsize; i++)
        ch_gat[c][i] *= sigma_gat_ch[c];
      DT_OMP_FOR()
      for(size_t i = 0; i < chsize; i++)
        ch_gat[c][i] = gat_inverse_exact(ch_gat[c][i]);
      DT_OMP_FOR()
      for(int row = row_offset; row < height; row += 2)
        for(int col = col_offset; col < width; col += 2)
          out[(size_t)row * width + col]
            = ch_gat[c][((row - row_offset) / 2) * halfwidth + (col - col_offset) / 2];
    }
  }

cleanup:
  for(int c = 0; c < 4; c++)
  {
    dt_free_align(ch_gat[c]);
    dt_free_align(ch_original[c]);
  }
  dt_free_align(luma); dt_free_align(chroma1);
  dt_free_align(chroma2); dt_free_align(chroma3);
  dt_free_align(luma_pilot); dt_free_align(luma_denoised);
  dt_free_align(c1_denoised); dt_free_align(c2_denoised);
  dt_free_align(c3_denoised);
  dt_free_align(noise_varmap);
}

/* ─── main() ────────────────────────────────────────── */
int main(int argc, char **argv)
{
  if(argc < 5)
  {
    fprintf(stderr,
      "Usage: %s input.bin output.bin width height\n"
      "       [method] [strength] [luma_str] [chroma_str]\n"
      "       [alpha] [sigma_sq] [iterations] [search_window]\n"
      "\n"
      "  method:      'ours' (default) or 'perchannel' (fallback)\n"
      "  strength:    global strength (default 1.0)\n"
      "  luma_str:    luma strength multiplier (default 0.7)\n"
      "  chroma_str:  chroma strength multiplier (default 1.5)\n"
      "  alpha:       Poisson-Gaussian shot noise gain (auto if -1)\n"
      "  sigma_sq:    read noise variance (auto if -1)\n"
      "  iterations:  1-4 (default 1)\n"
      "  search_window: search radius (default 16)\n",
      argv[0]);
    return 1;
  }

  const char *input_file = argv[1];
  const char *output_file = argv[2];
  const int width = atoi(argv[3]);
  const int height = atoi(argv[4]);

  const char *method = (argc > 5) ? argv[5] : "ours";
  const float strength = (argc > 6) ? (float)atof(argv[6]) : 1.0f;
  const float luma_str = (argc > 7) ? (float)atof(argv[7]) : 0.25f;
  const float chroma_str = (argc > 8) ? (float)atof(argv[8]) : 1.5f;
  float alpha = (argc > 9) ? (float)atof(argv[9]) : -1.0f;
  float sigma_sq = (argc > 10) ? (float)atof(argv[10]) : -1.0f;
  const int iterations = (argc > 11) ? atoi(argv[11]) : 1;
  /* search_window arg kept for CLI compatibility but not used (hardcoded in defines) */

  if(width <= 0 || height <= 0)
  {
    fprintf(stderr, "Invalid dimensions: %dx%d\n", width, height);
    return 1;
  }

  fprintf(stderr, "Raw Denoiser Standalone (darktable culmination)\n");
  fprintf(stderr, "  Input:  %s (%dx%d)\n", input_file, width, height);
  fprintf(stderr, "  Method: %s\n", method);
  fprintf(stderr, "  Params: strength=%.2f luma=%.2f chroma=%.2f alpha=%.6f sigma_sq=%.8f iter=%d\n",
          strength, luma_str, chroma_str, alpha, sigma_sq, iterations);

  /* Initialize */
  init_kaiser_window();
  init_dct_basis();

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

  /* Override alpha/sigma_sq: negative means auto-estimate */
  float use_alpha = (alpha > 0) ? alpha : -1.0f;
  float use_sigma_sq = (sigma_sq >= 0 && alpha > 0) ? sigma_sq : -1.0f;

  /* Process */
  double t_start = omp_get_wtime();

  gat_bm3d_denoise(in, out, width, height, strength, iterations,
                    luma_str, chroma_str, use_alpha, use_sigma_sq);

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
