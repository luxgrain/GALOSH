/*
 * yuv_galosh_core.c
 *
 * YUV GALOSH — WHT-based local shrinkage denoiser for YUV420/YUV444 video.
 *
 * Extends GALOSH (rawdenoise_v6.c / darktable rawdenoise IOP) to processed
 * video.  In rawdenoise, a 2×2 WHT decomposes the Bayer mosaic into a luma-
 * like L plane and three chroma-like C planes at half resolution; GALOSH then
 * denoises each plane independently.  YUV already performs that separation:
 *   Y  ≈ L   (full resolution)
 *   U/V ≈ C1/C2  (half resolution in YUV420)
 * so the 2×2 Bayer decomposition is not needed here.
 *
 * Each plane (Y, U, V) is processed by the same two-pass WHT pipeline:
 *   Pass 1 — BayesShrink hard threshold (pilot estimate)
 *   Pass 2 — Empirical Wiener shrinkage using pilot
 *
 * 3D mode (tr > 0):
 *   Instead of a BayesShrink pilot, the temporal mean of WHT coefficients
 *   across N = 2·tr+1 motion-compensated frames is used as the pilot.
 *   Temporal averaging gives an SNR boost of √N, providing a far better
 *   signal estimate than any single-frame spatial pilot.
 *   Caller supplies MC frames (e.g. from AviSynth MVTools / MCompensate).
 *
 * Noise model (v1): Gaussian (α = 0).
 *   Per-plane σ is estimated via Laplacian MAD.  σ normalisation maps the
 *   plane to unit-variance domain (equivalent to a Gaussian GAT), where
 *   BayesShrink and Wiener thresholds are exact.
 *   → Upgrade path: full Poisson-Gaussian GAT for Y (marked TODO below).
 *
 * Usage (standalone test):
 *   yuv_galosh.exe in.yuv out.yuv W H [420|444] [sy] [sc] [stride_y] [stride_c]
 *      in.yuv   : planar float32 YUV (Y plane then U then V)
 *      sy / sc  : denoising strength for Y / chroma  (default 1.0)
 *      stride   : block stride (2 = 75% overlap, 4 = 50%)
 *
 * Build (MSYS2 UCRT64):
 *   gcc -O3 -march=native -msse4.1 -fopenmp -o yuv_galosh.exe yuv_galosh_core.c -lm
 *
 * Theory:
 *   BayesShrink:  Chang, Yu & Vetterli, IEEE TIP 2000
 *   Wiener:       James-Stein / empirical Wiener (textbook)
 *   GAT upgrade:  Makitalo & Foi, IEEE TIP 2013
 *   Noise est:    Foi et al., Signal Processing 2008 (lower envelope method)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <omp.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>

/* ================================================================
 * Platform helpers
 * ================================================================ */
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

static inline float *dt_alloc_align_float(size_t n)
{
    size_t sz = sizeof(float) * ((n + 15) & ~15);
#ifdef _WIN32
    return (float *)_aligned_malloc(sz, 64);
#else
    return (float *)aligned_alloc(64, sz);
#endif
}
static inline void *dt_alloc_align(size_t alignment, size_t sz)
{
#ifdef _WIN32
    return _aligned_malloc(sz, alignment);
#else
    return aligned_alloc(alignment, sz);
#endif
}
static inline void dt_free_align(void *p)
{
    if(!p) return;
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* ================================================================
 * GALOSH constants
 * ================================================================ */
#define GALOSH_BLOCK_SIZE   8
#define GALOSH_BLOCK_PIXELS (GALOSH_BLOCK_SIZE * GALOSH_BLOCK_SIZE)   /* 64 */

/* Wiener floor: prevents complete signal annihilation in flat blocks.
 * 1/sqrt(N) = 1/8 = 0.125 for 8×8 blocks — same reasoning as rawdenoise_v6.c. */
#define GALOSH_WIENER_FLOOR (1.0f / GALOSH_BLOCK_SIZE)

/* YUV format tags */
#define GALOSH_YUV_420  420
#define GALOSH_YUV_444  444

/* Max temporal radius */
#define GALOSH_TR_MAX   4

/* ================================================================
 * User parameters
 * ================================================================ */
typedef struct
{
    float sigma_y;     /* Y  denoising strength multiplier  (default 1.0) */
    float sigma_c;     /* U/V denoising strength multiplier (default 1.0) */
    int   stride_y;    /* Y  block stride: 2 = 75% overlap, 4 = 50%       */
    int   stride_c;    /* U/V block stride                                 */
    int   tr;          /* Temporal radius 0-4 (0 = 2D spatial only)       */
    int   yuv_format;  /* GALOSH_YUV_420 or GALOSH_YUV_444                */
    float gamma_curve; /* Display gamma for alpha estimation: 1.0=linear RAW (default),
                        * 2.2=sRGB approximation.  The variance-mean regression for
                        * Poisson alpha (Var=alpha*x+sigma^2) is only valid in the linear
                        * photon domain.  When input is gamma-compressed (e.g. ISP sRGB),
                        * set gamma_curve to the encoding gamma so the estimator can
                        * compensate: Var_lin = Var_sRGB / (gamma * x_lin^(gamma-1))^2,
                        * x_lin = x_sRGB^(1/gamma).  Has no effect when gamma_curve=1.0. */
    float sigma_y_scale; /* Y  sigma scale factor: 0/1.0=auto, >1.0=multiply MAD estimate */
    float sigma_c_scale; /* U/V sigma scale factor */
    int   use_3dwht;     /* 0=temporal-mean+Wiener, 1=3D WHT BayesShrink pilot */
} galosh_yuv_params_t;

static const galosh_yuv_params_t GALOSH_YUV_DEFAULTS = {
    .sigma_y       = 1.0f,
    .sigma_c       = 1.0f,
    .stride_y      = 2,
    .stride_c      = 2,
    .tr            = 0,
    .yuv_format    = GALOSH_YUV_420,
    .gamma_curve   = 1.0f,
    .sigma_y_scale = 0.0f,
    .sigma_c_scale = 0.0f,
    .use_3dwht     = 0
};

/* ================================================================
 * Kaiser-Bessel 2D window — identical to rawdenoise_v6.c.
 *
 * Tapers each overlapping patch's contribution to suppress block-boundary
 * grid artifacts in smooth gradients.  beta=2.0, N=8.
 * ================================================================ */
static const float galosh_kaiser_1d[GALOSH_BLOCK_SIZE] = {
    0.34012f, 0.59885f, 0.84123f, 0.97659f,
    0.97659f, 0.84123f, 0.59885f, 0.34012f
};
static float galosh_kaiser_2d[GALOSH_BLOCK_PIXELS];
static int   galosh_kaiser_ready = 0;

static void init_galosh_kaiser(void)
{
    if(galosh_kaiser_ready) return;
    for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
        for(int dx = 0; dx < GALOSH_BLOCK_SIZE; dx++)
            galosh_kaiser_2d[dy * GALOSH_BLOCK_SIZE + dx] =
                galosh_kaiser_1d[dy] * galosh_kaiser_1d[dx];
    galosh_kaiser_ready = 1;
}

/* ================================================================
 * O(n) quickselect — identical to rawdenoise_v6.c.
 * ================================================================ */
static float quick_select_kth(float *arr, const int n, const int k)
{
    if(n <= 0) return 0.0f;
    int lo = 0, hi = n - 1;
    while(lo < hi)
    {
        const float pivot = arr[k];
        int i = lo, j = hi;
        do {
            while(arr[i] < pivot) i++;
            while(arr[j] > pivot) j--;
            if(i <= j)
            {
                const float t = arr[i]; arr[i] = arr[j]; arr[j] = t;
                i++; j--;
            }
        } while(i <= j);
        if(j < k) lo = i;
        if(i > k) hi = j;
    }
    return arr[k];
}
static inline float quick_select_median(float *arr, const int n)
{
    return quick_select_kth(arr, n, n / 2);
}

/* ================================================================
 * 8-point WHT (in-place, sequency order) — identical to rawdenoise_v6.c.
 *
 * Self-inverse up to scale: WHT(WHT(x)) = N·x.
 * Higher index → higher spatial frequency (analogous to DCT).
 * ================================================================ */
static inline void wht8_inplace(float *x)
{
    float a0=x[0]+x[1], a1=x[0]-x[1], a2=x[2]+x[3], a3=x[2]-x[3];
    float a4=x[4]+x[5], a5=x[4]-x[5], a6=x[6]+x[7], a7=x[6]-x[7];
    float b0=a0+a2, b1=a1+a3, b2=a0-a2, b3=a1-a3;
    float b4=a4+a6, b5=a5+a7, b6=a4-a6, b7=a5-a7;
    x[0]=b0+b4; x[1]=b1+b5; x[2]=b2+b6; x[3]=b3+b7;
    x[4]=b0-b4; x[5]=b1-b5; x[6]=b2-b6; x[7]=b3-b7;
}

/* 2D separable WHT on 8×8 block (in-place).
 * normalize=0: forward (coefficients × 64).
 * normalize=1: inverse (divides by 64). */
static void wht2d_8x8(float block[GALOSH_BLOCK_PIXELS], const int normalize)
{
    for(int r = 0; r < GALOSH_BLOCK_SIZE; r++)
        wht8_inplace(block + r * GALOSH_BLOCK_SIZE);
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
        for(int i = 0; i < GALOSH_BLOCK_PIXELS; i++) block[i] *= inv;
    }
}

/* ================================================================
 * Per-plane noise estimation — Gaussian model (v1).
 *
 * Estimates σ² for a single float plane using Laplacian MAD.
 *   Laplacian: L = p[x-1] - 2·p[x] + p[x+1]  cancels linear signal.
 *   Var(L) = 6·σ²  (iid Gaussian)
 *   σ = MAD(|L|) / (0.6745 · √6) = MAD / 1.6521
 *
 * Additionally fits a simple Poisson-Gaussian model (Var = α·μ + σ²) using
 * a 4-bin variance vs. mean regression — robust enough for video.
 * For chroma (U/V): α expected to be near 0 (difference channel); caller
 * may pass use_poisson=0 to skip α estimation and return α=0.
 *
 * TODO (upgrade): Replace with full Foi lower-envelope method (as in
 *      rawdenoise_v6.c galosh_estimate_noise) for better α precision on Y.
 * ================================================================ */
typedef struct { float alpha, sigma_sq; } galosh_noise_t;

static galosh_noise_t galosh_estimate_noise_plane(
        const float *plane, const int width, const int height,
        const int use_poisson, const float gamma_curve)
{
    galosh_noise_t r = { 0.0f, 1e-8f };

    /* ---- Step 1: Laplacian MAD → σ ---- */
    const int nsamp = MIN(width * height / 3, 300000);
    float *laps = dt_alloc_align_float(nsamp + 1);
    if(!laps) return r;

    int cnt = 0;
    /* Horizontal second-difference, stride 2 for speed */
    for(int y = 0; y < height && cnt < nsamp; y++)
    {
        const float *row = plane + (size_t)y * width;
        for(int x = 1; x < width - 1 && cnt < nsamp; x += 2)
            laps[cnt++] = fabsf(row[x - 1] - 2.0f * row[x] + row[x + 1]);
    }
    /* Vertical second-difference, stride 2 */
    for(int y = 1; y < height - 1 && cnt < nsamp; y += 2)
    {
        const float *r0 = plane + (size_t)(y - 1) * width;
        const float *r1 = plane + (size_t) y      * width;
        const float *r2 = plane + (size_t)(y + 1) * width;
        for(int x = 0; x < width && cnt < nsamp; x += 3)
            laps[cnt++] = fabsf(r0[x] - 2.0f * r1[x] + r2[x]);
    }

    if(cnt < 20) { dt_free_align(laps); return r; }

    const float mad  = quick_select_median(laps, cnt);
    dt_free_align(laps);
    /* σ for the Laplacian  =  MAD / 0.6745
     * σ for the pixel      =  σ_lap / √6    →  combined: MAD / 1.6521 */
    const float sigma = mad / 1.6521f;
    r.sigma_sq = fmaxf(sigma * sigma, 1e-12f);

    if(!use_poisson) return r;   /* Gaussian model: α = 0 */

    /* ---- Step 2: 4-bin variance vs. mean → Poisson α (simple fit) ----
     *
     * Var(x) = α·x + σ²  →  α = (Var_hi - Var_lo) / (Mean_hi - Mean_lo)
     * Use 4 brightness bins, each bin's variance from block-local Laplacian.
     * Rougher than rawdenoise_v6.c's full lower-envelope but fast enough.
     */
#define NE_BINS   4
#define NE_BLK    16   /* scan 16×16 macro-blocks for bin mean */
    float bin_mean[NE_BINS] = {0};
    float bin_var [NE_BINS] = {0};
    int   bin_cnt [NE_BINS] = {0};

    /* Global intensity range for binning */
    float vmin = FLT_MAX, vmax = -FLT_MAX;
    {
        const int stride4 = MAX(1, (width * height) / 4096);
        for(int i = 0; i < width * height; i += stride4)
        {
            if(plane[i] < vmin) vmin = plane[i];
            if(plane[i] > vmax) vmax = plane[i];
        }
    }
    const float vrange = fmaxf(vmax - vmin, 1e-10f);

    const int bw = NE_BLK, bh = NE_BLK;
    for(int by = 0; by + bh <= height; by += bh)
    {
        for(int bx = 0; bx + bw <= width; bx += bw)
        {
            /* Block mean */
            float bsum = 0.0f;
            for(int dy = 0; dy < bh; dy++)
            {
                const float *row = plane + (size_t)(by + dy) * width + bx;
                for(int dx = 0; dx < bw; dx++) bsum += row[dx];
            }
            const float bmean = bsum / (float)(bw * bh);
            const int b = CLAMP((int)((bmean - vmin) / vrange * NE_BINS), 0, NE_BINS - 1);

            /* Block noise variance via horizontal Laplacian MAD */
            float bloc[NE_BLK * NE_BLK];  int nl = 0;
            for(int dy = 0; dy < bh; dy++)
            {
                const float *row = plane + (size_t)(by + dy) * width + bx;
                for(int dx = 1; dx < bw - 1; dx++)
                    bloc[nl++] = fabsf(row[dx - 1] - 2.0f * row[dx] + row[dx + 1]);
            }
            if(nl < 4) continue;
            const float bsig = quick_select_median(bloc, nl) / 1.6521f;
            bin_mean[b] += bmean;
            bin_var [b] += bsig * bsig;
            bin_cnt [b]++;
        }
    }
    for(int b = 0; b < NE_BINS; b++)
    {
        if(bin_cnt[b] > 0)
        {
            bin_mean[b] /= bin_cnt[b];
            bin_var [b] /= bin_cnt[b];
        }
    }

    /* Simple OLS fit on valid bins: Var(x) = alpha*x + sigma^2
     *
     * Gamma-aware compensation (gamma_curve != 1.0):
     *   In a gamma-compressed domain (e.g. sRGB, gamma ~2.2), the linear
     *   Poisson-Gaussian relationship Var_lin = alpha*x_lin + sigma^2 does NOT
     *   hold directly.  The chain rule gives:
     *   sRGB power-law: x_sRGB = x_lin^(1/gamma)
     *   Inverse:        x_lin  = x_sRGB^gamma
     *   Jacobian:       d(x_sRGB)/d(x_lin) = (1/gamma) * x_sRGB^(1-gamma)
     *   Delta method:   Var_lin = Var_sRGB / jacob^2
     *                           = Var_sRGB * gamma^2 * x_sRGB^(2*gamma-2)
     *   For gamma=2.2:  Var_lin = Var_sRGB * 4.84 * x_sRGB^2.4
     *   We then fit Var_lin = alpha*x_lin + sigma^2 as usual.
     *   Bins where x_sRGB < MIN_MU_GAMMA are skipped: jacob→∞ near black
     *   (since 1-gamma < 0 for gamma>1), making compensation numerically unstable. */
#define MIN_MU_GAMMA 0.04f   /* skip bins darker than ~4% sRGB (~0.13% linear) */
    double Sw = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    for(int b = 0; b < NE_BINS; b++)
    {
        if(bin_cnt[b] < 4) continue;
        double x = bin_mean[b];
        double y = bin_var [b];
        if(gamma_curve > 1.0f + 1e-3f)
        {
            /* Compensate to linear domain.
             * gamma_curve = display gamma (2.2 for sRGB power-law approx).
             * Encoding: x_sRGB = x_lin^(1/g)  →  x_lin = x_sRGB^g
             * Jacobian d(x_sRGB)/d(x_lin), expressed via x_sRGB:
             *   jacob = (1/g) * x_sRGB^(1−g)
             * Delta method: Var_lin = Var_sRGB / jacob²
             *   = Var_sRGB * g² * x_sRGB^(2g−2)
             * For g=2.2: Var_lin = Var_sRGB * 4.84 * x_sRGB^2.4            */
            if((float)x < MIN_MU_GAMMA) continue;   /* unstable near black: jacob→∞ */
            const double g     = (double)gamma_curve;
            const double x_lin = pow(x, g);                /* x_sRGB → x_linear         */
            const double jacob = (1.0/g) * pow(x, 1.0-g); /* d(x_sRGB)/d(x_lin) at x_sRGB */
            x = x_lin;
            y = y / (jacob * jacob);
        }
        const double w = bin_cnt[b];
        Sw += w; Sx += w*x; Sy += w*y; Sxx += w*x*x; Sxy += w*x*y;
    }
    const double det = Sw * Sxx - Sx * Sx;
    if(fabs(det) > 1e-30)
    {
        const float a = (float)((Sw * Sxy - Sx * Sy) / det);
        const float s = (float)((Sxx * Sy - Sx * Sxy) / det);
        if(a > 1e-8f) r.alpha    = a;
        if(s > 0.0f)  r.sigma_sq = s;
    }
#undef NE_BINS
#undef NE_BLK
#undef MIN_MU_GAMMA
    return r;
}

/* ================================================================
 * GAT: Generalized Anscombe Transform — Poisson-Gaussian variance stabilisation
 *
 * Forward (piecewise C1 VST, Foi 2008):
 *   For x ≥ y_break = −3α/8:
 *     T(x) = (2/α) · √(α·x + 3α²/8 + σ²)    [sqrt branch]
 *   For x < y_break:
 *     T(x) = t_break + (x − y_break) / σ_raw  [linear branch, C1 at y_break]
 *   where t_break = 2·σ_raw/α  (continuity).
 *   After transform: Var(T(x)) ≈ 1 everywhere.
 *
 * Inverse: Exact unbiased inverse via 10-point Gauss-Hermite quadrature LUT
 *   (Makitalo & Foi, IEEE TIP 2013).  Reduces bias at high ISO vs. closed-form.
 *
 * Gaussian fallback (α < GAT_ALPHA_THRESHOLD):
 *   Forward:  T(x) = x / σ_raw   (simple unit-variance normalisation)
 *   Inverse:  x    = T · σ_raw
 *   Used for U/V chroma planes where the Poisson component is negligible.
 *
 * Per-plane tables (gat_inv_table_t) are heap-allocated and freed per frame,
 * making this thread-safe for AviSynth+ multi-frame processing.
 * ================================================================ */

/* Below this threshold alpha is treated as zero (Gaussian model). */
#define GAT_ALPHA_THRESHOLD 1e-4f

/* Size of the inverse LUT.  4096 entries → linear interpolation error < 1e-5. */
#define GAT_INV_TABLE_SIZE 4096

typedef struct
{
    float d[GAT_INV_TABLE_SIZE];  /* expected GAT output values (monotone) */
    float x[GAT_INV_TABLE_SIZE];  /* corresponding signal values in [0, 1] */
    float d_min, d_max;
    float alpha, sigma_raw;
    float y_break, t_break;
    int   valid;
} gat_inv_table_t;

/* 10-point Gauss-Hermite quadrature (weight e^{-x²}).
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

/* Scalar forward GAT — identical to rawdenoise_v6.c gat_forward(). */
static inline float gat_forward_scalar(const float x,
                                        const float alpha,
                                        const float sigma_sq)
{
    const float xn = (x == x) ? x : 0.0f;   /* NaN guard, no clamp */
    const float y_break = -0.375f * alpha;
    if(xn >= y_break)
        return (2.0f / alpha)
             * sqrtf(alpha * xn + 0.375f * alpha * alpha + sigma_sq);
    else
    {
        const float sigma_raw = sqrtf(fmaxf(sigma_sq, 1e-20f));
        const float t_break   = 2.0f * sigma_raw / alpha;
        return t_break + (xn - y_break) / sigma_raw;
    }
}

/* Build the per-plane inverse LUT via Poisson + Gauss-Hermite summation.
 * Signal range [0, 1] covers normalised float video Y (0–1 after /max). */
static void gat_build_inverse_table(const float alpha, const float sigma_sq,
                                     gat_inv_table_t *tbl)
{
    const double a   = (double)alpha;
    const double sq  = (double)sigma_sq;
    const double sig = sqrt(fmax(sq, 1e-20));
    const double y_break_d = -0.375 * a;
    const double t_break_d = 2.0 * sig / a;

    for(int i = 0; i < GAT_INV_TABLE_SIZE; i++)
    {
        const double x_val  = (double)i / (double)(GAT_INV_TABLE_SIZE - 1);
        const double lambda = x_val / a;   /* Poisson rate */

        /* E[T(Poisson(λ)·α + N(0,σ²))] via Poisson summation + GH quadrature */
        double expected_gat = 0.0;
        const int k_max = (int)(lambda + 8.0 * sqrt(fmax(lambda, 1.0))) + 20;
        double log_prob = -lambda;

        for(int k = 0; k <= k_max; k++)
        {
            if(k > 0) log_prob += log(lambda + 1e-300) - log((double)k);
            const double prob = exp(log_prob);
            if(prob < 1e-15 && k > (int)lambda + 1) break;

            double eg = 0.0;
            for(int g = 0; g < 10; g++)
            {
                const double z      = 1.4142135623730951 * sig * gh_nodes[g];
                const double noisy  = (double)k * a + z;
                double T;
                if(noisy >= y_break_d)
                    T = (2.0 / a) * sqrt(fmax(a * noisy + 0.375*a*a + sq, 0.0));
                else
                    T = t_break_d + (noisy - y_break_d) / sig;
                eg += gh_weights[g] * T;
            }
            expected_gat += prob * eg * 0.5641895835477563; /* ×1/√π */
        }

        tbl->x[i] = (float)x_val;
        tbl->d[i] = (float)expected_gat;
    }

    tbl->d_min    = tbl->d[0];
    tbl->d_max    = tbl->d[GAT_INV_TABLE_SIZE - 1];
    tbl->alpha    = alpha;
    tbl->sigma_raw = (float)sig;
    tbl->y_break  = (float)y_break_d;
    tbl->t_break  = (float)t_break_d;
    tbl->valid    = 1;

    fprintf(stderr, "  [GAT LUT] alpha=%.6f sigma_sq=%.2e "
                    "d_range=[%.3f, %.3f]\n",
            alpha, sigma_sq, tbl->d_min, tbl->d_max);
}

/* Exact unbiased inverse via binary search on the monotone LUT. */
static inline float gat_inverse_exact(const float D, const gat_inv_table_t *tbl)
{
    if(!tbl->valid) return D;

    /* Below LUT range: analytical linear-branch inverse */
    if(D <= tbl->d_min)
        return tbl->y_break + tbl->sigma_raw * (D - tbl->t_break);
    if(D >= tbl->d_max)
        return 1.0f;

    /* Binary search */
    int lo = 0, hi = GAT_INV_TABLE_SIZE - 1;
    while(lo < hi - 1)
    {
        const int mid = (lo + hi) >> 1;
        if(tbl->d[mid] <= D) lo = mid; else hi = mid;
    }
    const float d0 = tbl->d[lo], d1 = tbl->d[hi];
    const float t  = (D - d0) / fmaxf(d1 - d0, 1e-10f);
    return tbl->x[lo] + t * (tbl->x[hi] - tbl->x[lo]);
}

/* ---- Plane-level GAT (in-place) ---- */

/* Forward GAT on a whole plane.
 * alpha < GAT_ALPHA_THRESHOLD → Gaussian fallback (x / σ_raw). */
static void gat_forward_plane(float *plane, const int npix,
                               const float alpha, const float sigma_sq)
{
    if(alpha < GAT_ALPHA_THRESHOLD)
    {
        /* Gaussian model: normalise to unit variance */
        const float inv = 1.0f / fmaxf(sqrtf(sigma_sq), 1e-10f);
        for(int i = 0; i < npix; i++) plane[i] *= inv;
    }
    else
    {
        for(int i = 0; i < npix; i++)
            plane[i] = gat_forward_scalar(plane[i], alpha, sigma_sq);
    }
}

/* Inverse GAT on a whole plane.
 * tbl must be pre-built (gat_build_inverse_table) when alpha is significant;
 * pass NULL for Gaussian fallback (tbl unused when alpha < threshold). */
static void gat_inverse_plane(float *plane, const int npix,
                               const float alpha, const float sigma_sq,
                               const gat_inv_table_t *tbl)
{
    if(alpha < GAT_ALPHA_THRESHOLD)
    {
        const float sigma_raw = sqrtf(fmaxf(sigma_sq, 1e-20f));
        for(int i = 0; i < npix; i++) plane[i] *= sigma_raw;
    }
    else
    {
        for(int i = 0; i < npix; i++)
            plane[i] = gat_inverse_exact(plane[i], tbl);
    }
}

/* ================================================================
 * Pass 1: BayesShrink adaptive hard thresholding — identical logic to
 * rawdenoise_v6.c galosh_pass1().
 *
 * For each overlapping 8×8 block:
 *   1. Forward 2D WHT (unnormalised; coefficients × 64)
 *   2. Estimate signal variance: σ_x² = max(σ_Y² − σ², 0)
 *      where σ_Y² = mean(coeff²) / N over AC coefficients
 *   3. BayesShrink threshold: λ = σ² / σ_x  (in unnorm. scale: × √N)
 *   4. Hard threshold all AC coefficients; DC always preserved
 *   5. Inverse WHT, overlap-add with Kaiser window
 *
 * sigma_strength: effective noise std in the (GAT-normalised) input plane.
 *   After gat_forward_plane(), noise is ≈ N(0,1), so sigma_strength = 1 is
 *   exact; user multiplier scales the aggressiveness.
 * ================================================================ */
static void galosh_pass1(const float *restrict input,
                          float *restrict output,
                          const int width, const int height,
                          const float sigma_strength,
                          const int stride)
{
    const int rmax  = height - GALOSH_BLOCK_SIZE;
    const int cmax  = width  - GALOSH_BLOCK_SIZE;
    const int npix  = width  * height;
    const float sigma_sq   = sigma_strength * sigma_strength;
    const float lambda_max = sigma_strength
                           * sqrtf(2.0f * logf((float)GALOSH_BLOCK_PIXELS));

    float *numer = (float *)dt_alloc_align(64, sizeof(float) * npix);
    float *denom = (float *)dt_alloc_align(64, sizeof(float) * npix);
    if(!numer || !denom)
    {
        dt_free_align(numer); dt_free_align(denom);
        memcpy(output, input, sizeof(float) * npix);
        return;
    }
    memset(numer, 0, sizeof(float) * npix);
    memset(denom, 0, sizeof(float) * npix);

    #pragma omp parallel
    {
        float *mn = (float *)dt_alloc_align(64, sizeof(float) * npix);
        float *md = (float *)dt_alloc_align(64, sizeof(float) * npix);
        if(mn && md)
        {
            memset(mn, 0, sizeof(float) * npix);
            memset(md, 0, sizeof(float) * npix);

            {
            int ref_r;
            #pragma omp for schedule(dynamic, 4)
            for(ref_r = 0; ref_r <= rmax; ref_r += stride)
            {
                for(int ref_c = 0; ref_c <= cmax; ref_c += stride)
                {
                    float block[GALOSH_BLOCK_PIXELS];
                    for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                        memcpy(block + dy * GALOSH_BLOCK_SIZE,
                               input + (ref_r + dy) * width + ref_c,
                               GALOSH_BLOCK_SIZE * sizeof(float));

                    wht2d_8x8(block, 0);

                    /* σ_Y² from unnorm. AC coefficients.
                     * Var(unnorm. coeff) = N · σ²  → divide sum_sq by (N-1)·N */
                    float sum_sq = 0.0f;
                    for(int i = 1; i < GALOSH_BLOCK_PIXELS; i++)
                        sum_sq += block[i] * block[i];
                    const float sigma_y_sq = sum_sq
                        / ((float)(GALOSH_BLOCK_PIXELS - 1) * (float)GALOSH_BLOCK_PIXELS);
                    const float sigma_x_sq = fmaxf(sigma_y_sq - sigma_sq, 0.0f);

                    float lambda;
                    if(sigma_x_sq < 1e-10f)
                    {
                        lambda = 1e30f;   /* flat block: kill all AC */
                    }
                    else
                    {
                        lambda = (sigma_sq / sqrtf(sigma_x_sq))
                               * sqrtf((float)GALOSH_BLOCK_PIXELS);
                        const float lmax_u = lambda_max
                                           * sqrtf((float)GALOSH_BLOCK_PIXELS);
                        if(lambda > lmax_u) lambda = lmax_u;
                    }

                    int n_nonzero = 1;   /* DC always kept */
                    for(int i = 1; i < GALOSH_BLOCK_PIXELS; i++)
                    {
                        if(fabsf(block[i]) < lambda) block[i] = 0.0f;
                        else                          n_nonzero++;
                    }

                    wht2d_8x8(block, 1);

                    const float weight = 1.0f / (float)n_nonzero;
                    for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                        for(int dx = 0; dx < GALOSH_BLOCK_SIZE; dx++)
                        {
                            const int pos = (ref_r + dy) * width + (ref_c + dx);
                            const float kw = galosh_kaiser_2d[dy*GALOSH_BLOCK_SIZE + dx];
                            mn[pos] += weight * kw * block[dy*GALOSH_BLOCK_SIZE + dx];
                            md[pos] += weight * kw;
                        }
                }
            }
            } /* end ref_r scope */

            #pragma omp critical
            {
                for(int i = 0; i < npix; i++)
                {
                    numer[i] += mn[i];
                    denom[i] += md[i];
                }
            }
        }
        dt_free_align(mn);
        dt_free_align(md);
    }

    for(int i = 0; i < npix; i++)
        output[i] = (denom[i] > 1e-10f) ? numer[i] / denom[i] : input[i];

    dt_free_align(numer);
    dt_free_align(denom);
}

/* ================================================================
 * Pass 2: Empirical Wiener shrinkage — identical logic to
 * rawdenoise_v6.c galosh_pass2_ex().
 *
 * For each overlapping 8×8 block:
 *   noisy block  →  forward WHT  →  w_k
 *   pilot block  →  forward WHT  →  p_k
 *   Wiener gain: G_k = max(G_floor,  p_k² / (p_k² + σ²_unorm))
 *     σ²_unorm = σ² · N  (unnorm. WHT coeff variance = N · pixel variance)
 *   Output: G_k · w_k  →  inverse WHT  →  overlap-add
 *
 * DC coefficient always gets G = 1 (fully preserved).
 * ================================================================ */
static void galosh_pass2(const float *restrict noisy,
                          const float *restrict pilot,
                          float *restrict output,
                          const int width, const int height,
                          const float sigma_strength,
                          const int stride)
{
    const int rmax  = height - GALOSH_BLOCK_SIZE;
    const int cmax  = width  - GALOSH_BLOCK_SIZE;
    const int npix  = width  * height;
    /* Noise variance in unnorm. WHT domain: each coeff Var = N · pixel Var */
    const float sigma_sq_unorm = sigma_strength * sigma_strength
                               * (float)GALOSH_BLOCK_PIXELS;

    float *numer = (float *)dt_alloc_align(64, sizeof(float) * npix);
    float *denom = (float *)dt_alloc_align(64, sizeof(float) * npix);
    if(!numer || !denom)
    {
        dt_free_align(numer); dt_free_align(denom);
        memcpy(output, noisy, sizeof(float) * npix);
        return;
    }
    memset(numer, 0, sizeof(float) * npix);
    memset(denom, 0, sizeof(float) * npix);

    #pragma omp parallel
    {
        float *mn = (float *)dt_alloc_align(64, sizeof(float) * npix);
        float *md = (float *)dt_alloc_align(64, sizeof(float) * npix);
        if(mn && md)
        {
            memset(mn, 0, sizeof(float) * npix);
            memset(md, 0, sizeof(float) * npix);

            {
            int ref_r;
            #pragma omp for schedule(dynamic, 4)
            for(ref_r = 0; ref_r <= rmax; ref_r += stride)
            {
                for(int ref_c = 0; ref_c <= cmax; ref_c += stride)
                {
                    float bn[GALOSH_BLOCK_PIXELS], bp[GALOSH_BLOCK_PIXELS];
                    for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                    {
                        memcpy(bn + dy*GALOSH_BLOCK_SIZE,
                               noisy + (ref_r+dy)*width + ref_c,
                               GALOSH_BLOCK_SIZE * sizeof(float));
                        memcpy(bp + dy*GALOSH_BLOCK_SIZE,
                               pilot + (ref_r+dy)*width + ref_c,
                               GALOSH_BLOCK_SIZE * sizeof(float));
                    }

                    wht2d_8x8(bn, 0);
                    wht2d_8x8(bp, 0);

                    float wiener_energy = 0.0f;
                    for(int i = 0; i < GALOSH_BLOCK_PIXELS; i++)
                    {
                        float g;
                        if(i == 0)
                        {
                            g = 1.0f;   /* DC: always fully preserved */
                        }
                        else
                        {
                            const float s2 = bp[i] * bp[i];
                            g = s2 / (s2 + sigma_sq_unorm);
                            if(g < GALOSH_WIENER_FLOOR) g = GALOSH_WIENER_FLOOR;
                        }
                        bn[i] *= g;
                        wiener_energy += g * g;
                    }

                    wht2d_8x8(bn, 1);

                    const float weight = 1.0f / fmaxf(wiener_energy, 1e-6f);
                    for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                        for(int dx = 0; dx < GALOSH_BLOCK_SIZE; dx++)
                        {
                            const int pos = (ref_r+dy)*width + (ref_c+dx);
                            const float kw = galosh_kaiser_2d[dy*GALOSH_BLOCK_SIZE+dx];
                            mn[pos] += weight * kw * bn[dy*GALOSH_BLOCK_SIZE+dx];
                            md[pos] += weight * kw;
                        }
                }
            }
            } /* end ref_r scope */

            #pragma omp critical
            {
                for(int i = 0; i < npix; i++)
                {
                    numer[i] += mn[i];
                    denom[i] += md[i];
                }
            }
        }
        dt_free_align(mn);
        dt_free_align(md);
    }

    for(int i = 0; i < npix; i++)
        output[i] = (denom[i] > 1e-10f) ? numer[i] / denom[i] : noisy[i];

    dt_free_align(numer);
    dt_free_align(denom);
}

/* ================================================================
 * Pass 1 — TEMPORAL (3D mode only)
 *
 * Replaces BayesShrink with a temporal-mean pilot when N ≥ 2 frames are
 * available (motion-compensated by caller).
 *
 * For each overlapping 8×8 block:
 *   1. Extract the same block from all N frames
 *   2. Forward WHT on each → N × 64 coefficient sets
 *   3. Per-coefficient temporal mean μ_k = (1/N) Σ w_k^(i)
 *      Noise variance of μ_k = σ² / N  (averaging √N SNR gain)
 *   4. Inverse WHT(μ_k) → pilot pixels, overlap-add with Kaiser
 *
 * gat_frames[0..n_frames-1]: all frames including the current frame,
 *   already GAT-normalised (unit-variance noise).  Caller arranges
 *   temporal order; the function does not need to know which is "current".
 *
 * The resulting pilot is passed to galosh_pass2() which operates on the
 * current (noisy) frame to produce the final denoised output.
 * ================================================================ */
static void galosh_pass1_temporal(const float * const *gat_frames,
                                   const int n_frames,
                                   float *pilot_out,
                                   const int width, const int height,
                                   const int stride)
{
    const int rmax = height - GALOSH_BLOCK_SIZE;
    const int cmax = width  - GALOSH_BLOCK_SIZE;
    const int npix = width  * height;
    const float inv_n = 1.0f / (float)n_frames;

    float *numer = (float *)dt_alloc_align(64, sizeof(float) * npix);
    float *denom = (float *)dt_alloc_align(64, sizeof(float) * npix);
    if(!numer || !denom)
    {
        dt_free_align(numer); dt_free_align(denom);
        /* Fallback: pilot = first frame */
        memcpy(pilot_out, gat_frames[0], sizeof(float) * npix);
        return;
    }
    memset(numer, 0, sizeof(float) * npix);
    memset(denom, 0, sizeof(float) * npix);

    #pragma omp parallel
    {
        float *mn = (float *)dt_alloc_align(64, sizeof(float) * npix);
        float *md = (float *)dt_alloc_align(64, sizeof(float) * npix);
        if(mn && md)
        {
            memset(mn, 0, sizeof(float) * npix);
            memset(md, 0, sizeof(float) * npix);

            {
            int ref_r;
            #pragma omp for schedule(dynamic, 4)
            for(ref_r = 0; ref_r <= rmax; ref_r += stride)
            {
                for(int ref_c = 0; ref_c <= cmax; ref_c += stride)
                {
                    /* Accumulate WHT coefficients across all frames */
                    float avg[GALOSH_BLOCK_PIXELS];
                    memset(avg, 0, sizeof(avg));

                    for(int fi = 0; fi < n_frames; fi++)
                    {
                        float blk[GALOSH_BLOCK_PIXELS];
                        for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                            memcpy(blk + dy * GALOSH_BLOCK_SIZE,
                                   gat_frames[fi] + (ref_r + dy) * width + ref_c,
                                   GALOSH_BLOCK_SIZE * sizeof(float));
                        wht2d_8x8(blk, 0);
                        for(int k = 0; k < GALOSH_BLOCK_PIXELS; k++)
                            avg[k] += blk[k];
                    }

                    /* Temporal mean = pilot WHT coefficients (√N SNR gain) */
                    for(int k = 0; k < GALOSH_BLOCK_PIXELS; k++)
                        avg[k] *= inv_n;

                    wht2d_8x8(avg, 1);

                    /* Overlap-add with Kaiser (uniform weight — no thresholding) */
                    for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                        for(int dx = 0; dx < GALOSH_BLOCK_SIZE; dx++)
                        {
                            const int pos = (ref_r+dy)*width + (ref_c+dx);
                            const float kw = galosh_kaiser_2d[dy*GALOSH_BLOCK_SIZE+dx];
                            mn[pos] += kw * avg[dy*GALOSH_BLOCK_SIZE+dx];
                            md[pos] += kw;
                        }
                }
            }
            } /* end ref_r scope */

            #pragma omp critical
            {
                for(int i = 0; i < npix; i++)
                {
                    numer[i] += mn[i];
                    denom[i] += md[i];
                }
            }
        }
        dt_free_align(mn);
        dt_free_align(md);
    }

    for(int i = 0; i < npix; i++)
        pilot_out[i] = (denom[i] > 1e-10f) ? numer[i] / denom[i] : gat_frames[0][i];

    dt_free_align(numer);
    dt_free_align(denom);
}

/* ================================================================
 * Temporal orthogonal transforms for 3D WHT mode.
 *
 * Applied per WHT coefficient k across n_frames (the temporal axis).
 * All transforms are orthonormal: noise N(0,σ²) remains N(0,σ²) in
 * the transformed domain — BayesShrink threshold math is unchanged.
 *
 * N=3 (tr=1):
 *   Explicit 3×3 orthonormal matrix (DC, antisymmetric diff, Laplacian):
 *     row 0: [1, 1, 1] / √3
 *     row 1: [1, 0,-1] / √2
 *     row 2: [1,-2, 1] / √6
 *
 * N=5 (tr=2):
 *   Orthonormal DCT-2 for N=5 (precomputed constants).
 *
 * General N (N>5 or fallback): QR-style pair-averaging tree (not used
 *   since GALOSH_TR_MAX = 4, so N_total = 2*tr+1 ≤ 9 with tr≤4;
 *   but AVS plugin enforces tr≤2, so N≤5 in practice).
 * ================================================================ */

/* N=3 constants */
#define T3_C0  0.57735027f   /* 1/√3 */
#define T3_C1  0.70710678f   /* 1/√2 */
#define T3_C2H 0.40824829f   /* 1/√6 */
#define T3_C2D 0.81649658f   /* 2/√6 */

static inline void temporal_fwd_3(float v[3])
{
    const float a = v[0], b = v[1], c = v[2];
    v[0] = (a + b + c) * T3_C0;
    v[1] = (a     - c) * T3_C1;
    v[2] = (a - 2*b+ c)* T3_C2H;
}
static inline void temporal_inv_3(float v[3])
{
    /* Inverse = transpose (orthogonal matrix) */
    const float d = v[0], e = v[1], f = v[2];
    v[0] = d * T3_C0 + e * T3_C1 + f * T3_C2H;
    v[1] = d * T3_C0              - f * T3_C2D;
    v[2] = d * T3_C0 - e * T3_C1 + f * T3_C2H;
}

/* N=5 orthonormal DCT-2 (precomputed to 7 significant figures) */
static const float dct5[5][5] = {
    /* row k=0: DC */
    { 0.4472136f,  0.4472136f,  0.4472136f,  0.4472136f,  0.4472136f },
    /* row k=1 */
    { 0.6015009f,  0.3717229f,  0.0000000f, -0.3717229f, -0.6015009f },
    /* row k=2 */
    { 0.5117310f, -0.1954423f, -0.6324555f, -0.1954423f,  0.5117310f },
    /* row k=3 */
    { 0.3717229f, -0.6015009f,  0.0000000f,  0.6015009f, -0.3717229f },
    /* row k=4 */
    { 0.1954423f, -0.5117310f,  0.6324555f, -0.5117310f,  0.1954423f }
};

static inline void temporal_fwd_5(float v[5])
{
    float out[5] = {0};
    for(int k = 0; k < 5; k++)
        for(int n = 0; n < 5; n++)
            out[k] += dct5[k][n] * v[n];
    for(int k = 0; k < 5; k++) v[k] = out[k];
}
static inline void temporal_inv_5(float v[5])
{
    /* Inverse = transpose (DCT-2 is orthogonal) */
    float out[5] = {0};
    for(int n = 0; n < 5; n++)
        for(int k = 0; k < 5; k++)
            out[n] += dct5[k][n] * v[k];   /* note: dct5[k][n] = dct5^T[n][k] */
    for(int n = 0; n < 5; n++) v[n] = out[n];
}

/* Dispatch: forward/inverse temporal transform for n frames */
static inline void temporal_fwd(float *v, int n)
{
    if     (n == 3) temporal_fwd_3(v);
    else if(n == 5) temporal_fwd_5(v);
    /* n==1: identity (2D mode, shouldn't reach here) */
}
static inline void temporal_inv(float *v, int n)
{
    if     (n == 3) temporal_inv_3(v);
    else if(n == 5) temporal_inv_5(v);
}

/* ================================================================
 * galosh_pass1_3dwht: 3D WHT pilot estimation
 *
 * Replaces galosh_pass1_temporal when use_3dwht=1.
 *
 * For each overlapping 8×8 spatial block:
 *   1. Extract 8×8 block from all n_frames frames
 *   2. Forward 2D WHT on each frame → n_frames × 64 coefficients
 *   3. Forward temporal transform (N-point orthogonal) per WHT coeff k
 *      → full 3D coefficient tensor [n_frames × 64]
 *   4. BayesShrink on ALL (n_frames*64 − 1) 3D-AC coefficients:
 *        σ_Y² = mean(coeff²) / n_total    (includes all n_frames)
 *        σ_X² = max(σ_Y² − σ², 0)
 *        λ    = σ² / √σ_X² × √(n_frames * N_pix)
 *      3D DC (temporal-DC of spatial-DC) always preserved.
 *   5. Inverse temporal transform per coefficient
 *   6. Extract center frame coefficients (index n_frames/2)
 *   7. Inverse 2D WHT → pilot pixels → overlap-add with Kaiser
 *
 * Because both 2D WHT and temporal transforms are orthogonal, noise in
 * the 3D domain remains N(0, σ²) — BayesShrink is exact without
 * any domain-dependent scaling factors.
 *
 * sigma_strength: user multiplier (same semantics as galosh_pass1).
 * ================================================================ */
#define GALOSH_MAX_FRAMES  (GALOSH_TR_MAX * 2 + 1)  /* up to 9 */

static void galosh_pass1_3dwht(const float * const *gat_frames,
                                const int n_frames,
                                float *pilot_out,
                                const int width, const int height,
                                const float sigma_strength,
                                const int stride)
{
    const int rmax   = height - GALOSH_BLOCK_SIZE;
    const int cmax   = width  - GALOSH_BLOCK_SIZE;
    const int npix   = width  * height;
    const int center = n_frames / 2;
    const float sigma_sq = sigma_strength * sigma_strength;
    /* lambda cap: √(2 ln(n_frames × N_pix)) × σ × √(unnorm scale) */
    const float lambda_max_base = sigma_strength
        * sqrtf(2.0f * logf((float)(n_frames * GALOSH_BLOCK_PIXELS)));
    const float lambda_max = lambda_max_base * sqrtf((float)GALOSH_BLOCK_PIXELS);

    float *numer = (float *)dt_alloc_align(64, sizeof(float) * npix);
    float *denom = (float *)dt_alloc_align(64, sizeof(float) * npix);
    if(!numer || !denom)
    {
        dt_free_align(numer); dt_free_align(denom);
        /* Fallback: copy center frame */
        memcpy(pilot_out, gat_frames[center], sizeof(float) * npix);
        return;
    }
    memset(numer, 0, sizeof(float) * npix);
    memset(denom, 0, sizeof(float) * npix);

    #pragma omp parallel
    {
        float *mn = (float *)dt_alloc_align(64, sizeof(float) * npix);
        float *md = (float *)dt_alloc_align(64, sizeof(float) * npix);
        if(mn && md)
        {
            memset(mn, 0, sizeof(float) * npix);
            memset(md, 0, sizeof(float) * npix);

            {
            int ref_r;
            #pragma omp for schedule(dynamic, 4)
            for(ref_r = 0; ref_r <= rmax; ref_r += stride)
            {
                for(int ref_c = 0; ref_c <= cmax; ref_c += stride)
                {
                    /* Step 1+2: extract and forward 2D WHT all frames */
                    float blks[GALOSH_MAX_FRAMES][GALOSH_BLOCK_PIXELS];
                    for(int fi = 0; fi < n_frames; fi++)
                    {
                        for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                            memcpy(blks[fi] + dy * GALOSH_BLOCK_SIZE,
                                   gat_frames[fi] + (ref_r + dy) * width + ref_c,
                                   GALOSH_BLOCK_SIZE * sizeof(float));
                        wht2d_8x8(blks[fi], 0);   /* forward, unnorm (×64) */
                    }

                    /* Step 3: temporal forward transform per WHT coefficient */
                    float tmp[GALOSH_MAX_FRAMES];
                    for(int k = 0; k < GALOSH_BLOCK_PIXELS; k++)
                    {
                        for(int fi = 0; fi < n_frames; fi++) tmp[fi] = blks[fi][k];
                        temporal_fwd(tmp, n_frames);
                        for(int fi = 0; fi < n_frames; fi++) blks[fi][k] = tmp[fi];
                    }

                    /* Step 4: 3D BayesShrink
                     * Count all 3D-AC coefficients: skip only blks[0][0]
                     * (temporal DC of spatial DC = true 3D DC).
                     * Noise variance in unnorm. 2D-WHT domain: Var = N_pix × σ²
                     * Temporal transform is orthogonal → noise unchanged.
                     * Combined unnorm variance: N_pix × σ² per coeff. */
                    float sum_sq = 0.0f;
                    int   n_ac   = 0;
                    for(int fi = 0; fi < n_frames; fi++)
                        for(int k = 0; k < GALOSH_BLOCK_PIXELS; k++)
                        {
                            if(fi == 0 && k == 0) continue;  /* 3D DC */
                            sum_sq += blks[fi][k] * blks[fi][k];
                            n_ac++;
                        }

                    const float sigma_y_sq  = sum_sq / ((float)n_ac
                                            * (float)GALOSH_BLOCK_PIXELS);
                    const float sigma_x_sq  = fmaxf(sigma_y_sq - sigma_sq, 0.0f);

                    float lambda;
                    if(sigma_x_sq < 1e-10f)
                    {
                        lambda = 1e30f;
                    }
                    else
                    {
                        lambda = (sigma_sq / sqrtf(sigma_x_sq))
                               * sqrtf((float)GALOSH_BLOCK_PIXELS);
                        if(lambda > lambda_max) lambda = lambda_max;
                    }

                    int n_nonzero = 1;   /* 3D DC always kept */
                    for(int fi = 0; fi < n_frames; fi++)
                        for(int k = 0; k < GALOSH_BLOCK_PIXELS; k++)
                        {
                            if(fi == 0 && k == 0) continue;
                            if(fabsf(blks[fi][k]) < lambda) blks[fi][k] = 0.0f;
                            else n_nonzero++;
                        }

                    /* Step 5: inverse temporal transform */
                    for(int k = 0; k < GALOSH_BLOCK_PIXELS; k++)
                    {
                        for(int fi = 0; fi < n_frames; fi++) tmp[fi] = blks[fi][k];
                        temporal_inv(tmp, n_frames);
                        for(int fi = 0; fi < n_frames; fi++) blks[fi][k] = tmp[fi];
                    }

                    /* Step 6+7: inverse 2D WHT on center frame, overlap-add */
                    wht2d_8x8(blks[center], 1);

                    const float weight = 1.0f / (float)n_nonzero;
                    for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                        for(int dx = 0; dx < GALOSH_BLOCK_SIZE; dx++)
                        {
                            const int pos = (ref_r + dy) * width + (ref_c + dx);
                            const float kw = galosh_kaiser_2d[dy*GALOSH_BLOCK_SIZE + dx];
                            mn[pos] += weight * kw * blks[center][dy*GALOSH_BLOCK_SIZE + dx];
                            md[pos] += weight * kw;
                        }
                }
            }
            } /* end ref_r scope */

            #pragma omp critical
            {
                for(int i = 0; i < npix; i++)
                {
                    numer[i] += mn[i];
                    denom[i] += md[i];
                }
            }
        }
        dt_free_align(mn);
        dt_free_align(md);
    }

    for(int i = 0; i < npix; i++)
        pilot_out[i] = (denom[i] > 1e-10f) ? numer[i] / denom[i] : gat_frames[center][i];

    dt_free_align(numer);
    dt_free_align(denom);
}

/* ================================================================
 * Per-plane pipeline
 *
 * Runs the full GALOSH pipeline on a single float plane:
 *   A. Estimate σ (Laplacian MAD)
 *   B. GAT forward: plane → plane / σ  (unit-variance Gaussian domain)
 *   C. Pass 1 pilot:
 *        2D  (n_mc == 0): BayesShrink spatial pilot
 *        3D  (n_mc  > 0): temporal mean pilot from MC frames
 *   D. Pass 2: Wiener shrinkage
 *   E. GAT inverse: multiply back by σ
 *
 * mc_frames[0..n_mc-1]: motion-compensated frames for 3D mode (NOT GAT'd
 *   yet; this function applies GAT to them internally using the same σ as
 *   the current frame — sensor noise parameters are constant across frames).
 *   Pass NULL / n_mc=0 for 2D mode.
 *
 * sigma_mult: user strength multiplier (galosh_yuv_params_t.sigma_y / .sigma_c)
 * ================================================================ */
/* np: pre-estimated noise params for this plane.
 *   Y  → caller uses use_poisson=1; alpha may be significant → full PG GAT.
 *   U/V → caller uses use_poisson=0; alpha=0  → Gaussian GAT (x/sigma).
 * Input must already be in linear domain.  Gamma linearisation/re-compression
 * is handled by the caller (galosh_yuv_denoise). */
static void galosh_yuv_denoise_plane(
        const float *in, float *out,
        const int width, const int height,
        const float sigma_mult,
        const int stride,
        const galosh_noise_t np,
        const float * const *mc_frames,
        const int n_mc,
        const int use_3dwht)
{
    const int npix = width * height;

    fprintf(stderr, "  [plane %dx%d] alpha=%.2e sigma_sq=%.2e sigma_mult=%.2f "
                    "stride=%d n_mc=%d\n",
            width, height, np.alpha, np.sigma_sq, sigma_mult, stride, n_mc);

    /* ---- B. GAT forward (in-place on private copy) ---- */
    float *gat_cur = dt_alloc_align_float(npix);
    if(!gat_cur) { memcpy(out, in, sizeof(float) * npix); return; }
    memcpy(gat_cur, in, sizeof(float) * npix);
    gat_forward_plane(gat_cur, npix, np.alpha, np.sigma_sq);

    /* Build inverse LUT when using full Poisson-Gaussian GAT.
     * For Gaussian fallback (alpha < threshold), tbl=NULL and
     * gat_inverse_plane() uses x·σ automatically. */
    gat_inv_table_t *tbl = NULL;
    if(np.alpha >= GAT_ALPHA_THRESHOLD)
    {
        tbl = (gat_inv_table_t *)malloc(sizeof(gat_inv_table_t));
        if(tbl) gat_build_inverse_table(np.alpha, np.sigma_sq, tbl);
    }

    /* After GAT, noise ≈ N(0,1) everywhere.
     * sigma_mult scales aggressiveness; 1.0 = exact noise match. */
    const float sigma_str = sigma_mult;

    /* ---- C. Pass 1: pilot ---- */
    float *pilot = dt_alloc_align_float(npix);
    if(!pilot)
    {
        memcpy(out, in, sizeof(float) * npix);
        dt_free_align(gat_cur); free(tbl);
        return;
    }

    if(n_mc == 0)
    {
        /* 2D: BayesShrink spatial pilot */
        galosh_pass1(gat_cur, pilot, width, height, sigma_str, stride);
    }
    else
    {
        /* 3D: temporal mean pilot.
         * Sensor noise params are constant across frames (same ISO),
         * so we GAT all MC frames with the same (alpha, sigma_sq). */
        const int n_total = n_mc + 1;
        const int cur_idx = n_mc / 2;
        const float **gat_all    = (const float **)malloc(sizeof(float *) * n_total);
        float       **gat_mc_buf = (float **)malloc(sizeof(float *) * n_mc);
        if(!gat_all || !gat_mc_buf)
        {
            free(gat_all); free(gat_mc_buf);
            galosh_pass1(gat_cur, pilot, width, height, sigma_str, stride);
        }
        else
        {
            for(int fi = 0; fi < n_mc; fi++)
            {
                gat_mc_buf[fi] = dt_alloc_align_float(npix);
                if(gat_mc_buf[fi])
                {
                    memcpy(gat_mc_buf[fi], mc_frames[fi], sizeof(float) * npix);
                    gat_forward_plane(gat_mc_buf[fi], npix, np.alpha, np.sigma_sq);
                }
            }
            for(int fi = 0; fi < cur_idx; fi++)
                gat_all[fi] = gat_mc_buf[fi] ? gat_mc_buf[fi] : gat_cur;
            gat_all[cur_idx] = gat_cur;
            for(int fi = cur_idx; fi < n_mc; fi++)
                gat_all[fi + 1] = gat_mc_buf[fi] ? gat_mc_buf[fi] : gat_cur;

            if(use_3dwht && n_total >= 3)
                galosh_pass1_3dwht(gat_all, n_total, pilot, width, height,
                                   sigma_str, stride);
            else
                galosh_pass1_temporal(gat_all, n_total, pilot, width, height, stride);

            for(int fi = 0; fi < n_mc; fi++) dt_free_align(gat_mc_buf[fi]);
            free(gat_mc_buf); free(gat_all);
        }
    }

    /* ---- D. Pass 2: Wiener ---- */
    float *denoised_gat = dt_alloc_align_float(npix);
    if(!denoised_gat)
    {
        memcpy(out, in, sizeof(float) * npix);
        dt_free_align(gat_cur); dt_free_align(pilot); free(tbl);
        return;
    }

    galosh_pass2(gat_cur, pilot, denoised_gat, width, height, sigma_str, stride);
    dt_free_align(pilot);
    dt_free_align(gat_cur);

    /* ---- E. GAT inverse → output ---- */
    gat_inverse_plane(denoised_gat, npix, np.alpha, np.sigma_sq, tbl);
    memcpy(out, denoised_gat, sizeof(float) * npix);
    dt_free_align(denoised_gat);
    free(tbl);
}

/* ================================================================
 * Gamma pipeline helpers (power-law approximation, float32 throughout).
 *
 * inv_gamma: x_sRGB → x_linear = clamp(x,0,1)^gamma_curve
 * fwd_gamma: x_linear → x_sRGB  = clamp(x,0,1)^(1/gamma_curve)
 *
 * Allocates a new buffer; caller must dt_free_align() it.
 * Returns NULL on allocation failure.
 * ================================================================ */
static float *plane_linearize(const float *src, int npix, float gamma_curve)
{
    float *dst = dt_alloc_align_float(npix);
    if(!dst) return NULL;
    const float g = gamma_curve;
    for(int i = 0; i < npix; i++)
        dst[i] = powf(fmaxf(src[i], 0.0f), g);
    return dst;
}

static void plane_apply_gamma_inplace(float *plane, int npix, float gamma_curve)
{
    const float inv_g = 1.0f / gamma_curve;
    for(int i = 0; i < npix; i++)
        plane[i] = powf(fmaxf(plane[i], 0.0f), inv_g);
}

/* ================================================================
 * Top-level YUV denoiser
 *
 * Processes Y, U, V planes independently.  Plane dimensions depend on
 * yuv_format: YUV420 → U/V at (w/2) × (h/2); YUV444 → all at w × h.
 *
 * mc_y / mc_u / mc_v: arrays of n_mc motion-compensated frame pointers,
 *   one per plane.  Pass NULL arrays and n_mc=0 for 2D mode.
 *
 * Returns 0 on success, -1 on allocation failure.
 * ================================================================ */
int galosh_yuv_denoise(
        float *y_out, float *u_out, float *v_out,
        const float *y_in, const float *u_in, const float *v_in,
        const int width, const int height,
        const float * const *mc_y,
        const float * const *mc_u,
        const float * const *mc_v,
        const int n_mc,
        const galosh_yuv_params_t *params)
{
    if(!params) params = &GALOSH_YUV_DEFAULTS;
    init_galosh_kaiser();

    /* Chroma plane dimensions */
    const int cw = (params->yuv_format == GALOSH_YUV_420) ? width  / 2 : width;
    const int ch = (params->yuv_format == GALOSH_YUV_420) ? height / 2 : height;

    fprintf(stderr, "[yuv_galosh] %dx%d YUV%d  tr=%d  sy=%.2f sc=%.2f"
                    " stride_y=%d stride_c=%d  gamma=%.2f\n",
            width, height, params->yuv_format, params->tr,
            params->sigma_y, params->sigma_c,
            params->stride_y, params->stride_c,
            params->gamma_curve);

    /* Gamma pipeline: GAT requires linear (photon-count) domain.
     * gamma_curve > 1.0 => linearise input, denoise, re-apply gamma. */
    const float gc        = params->gamma_curve;
    const int   use_gamma = (gc > 1.0f + 1e-3f);

    const size_t y_npix = (size_t)width * height;
    const size_t c_npix = (size_t)cw    * ch;

    /* Gamma pipeline applies to LUMA (Y) only.
     * Chroma (Cb/Cr) is a signed difference signal in [-0.5, 0.5] — it has no
     * meaningful gamma encoding and cannot be linearised with a pow() clamp. */
    float *y_lin = NULL;
    float **mc_y_lin = NULL;

    if(use_gamma)
    {
        y_lin = plane_linearize(y_in, (int)y_npix, gc);
        if(!y_lin)
        {
            fprintf(stderr, "[yuv_galosh] alloc failed in gamma linearise\n");
            return -1;
        }
        if(n_mc > 0)
        {
            mc_y_lin = calloc(n_mc, sizeof(float *));
            for(int fi = 0; fi < n_mc; fi++)
                if(mc_y && mc_y[fi]) mc_y_lin[fi] = plane_linearize(mc_y[fi], (int)y_npix, gc);
        }
    }

    const float *y_src       = use_gamma ? y_lin : y_in;
    const float *u_src       = u_in;   /* chroma: always treat as linear */
    const float *v_src       = v_in;
    const float * const *mcy = (use_gamma && mc_y_lin) ? (const float * const *)mc_y_lin : mc_y;
    const float * const *mcu = mc_u;
    const float * const *mcv = mc_v;

    const float gc_est = use_gamma ? 1.0f : gc;
    galosh_noise_t np_y = galosh_estimate_noise_plane(y_src, width, height, 1, gc_est);
    galosh_noise_t np_u = galosh_estimate_noise_plane(u_src, cw,    ch,    0, gc_est);
    galosh_noise_t np_v = galosh_estimate_noise_plane(v_src, cw,    ch,    0, gc_est);

    /* sigma_y/c_scale: corrects MAD underestimation for ISP-NR footage.
     * ISP NR introduces spatial correlation; Laplacian MAD sees smaller 2nd
     * differences → underestimates σ.  Scale multiplies the estimated sigma:
     *   scale=1.0 (or 0.0) → use MAD result as-is
     *   scale=3.0           → treat actual noise as 3× the MAD estimate
     * When scale > 1, Poisson alpha is zeroed (ISP already removed shot noise). */
    {
        const float sy_scale = (params->sigma_y_scale > 1e-3f) ? params->sigma_y_scale : 1.0f;
        if(sy_scale > 1.0f + 1e-3f)
        {
            np_y.sigma_sq *= sy_scale * sy_scale;
            np_y.alpha     = 0.0f;   /* ISP removed Poisson component */
            fprintf(stderr, "[yuv_galosh] noise  Y: MAD × %.2f → sigma=%.4f sigma_sq=%.2e"
                            " -> Gaussian GAT\n",
                    sy_scale, sqrtf(np_y.sigma_sq), np_y.sigma_sq);
        }
        else
        {
            fprintf(stderr, "[yuv_galosh] noise  Y: alpha=%.2e sigma_sq=%.2e  %s\n",
                    np_y.alpha, np_y.sigma_sq,
                    np_y.alpha >= GAT_ALPHA_THRESHOLD ? "-> full PG GAT" : "-> Gaussian GAT");
        }
    }
    {
        const float sc_scale = (params->sigma_c_scale > 1e-3f) ? params->sigma_c_scale : 1.0f;
        if(sc_scale > 1.0f + 1e-3f)
        {
            np_u.alpha = 0.0f;  np_u.sigma_sq *= sc_scale * sc_scale;
            np_v.alpha = 0.0f;  np_v.sigma_sq *= sc_scale * sc_scale;
            fprintf(stderr, "[yuv_galosh] noise  U/V: MAD × %.2f → sigma=%.4f sigma_sq=%.2e"
                            " -> Gaussian GAT\n",
                    sc_scale, sqrtf(np_u.sigma_sq), np_u.sigma_sq);
        }
        else
        {
            fprintf(stderr, "[yuv_galosh] noise  U: alpha=%.2e sigma_sq=%.2e  -> Gaussian GAT\n",
                    np_u.alpha, np_u.sigma_sq);
            fprintf(stderr, "[yuv_galosh] noise  V: alpha=%.2e sigma_sq=%.2e  -> Gaussian GAT\n",
                    np_v.alpha, np_v.sigma_sq);
        }
    }

    const int w3d = params->use_3dwht;
    fprintf(stderr, "[yuv_galosh] pilot mode: %s\n",
            (w3d && n_mc > 0) ? "3D WHT BayesShrink" : (n_mc > 0 ? "temporal mean + Wiener" : "spatial BayesShrink + Wiener"));

    fprintf(stderr, "[yuv_galosh] Processing Y plane...\n");
    galosh_yuv_denoise_plane(y_src, y_out, width, height,
                              params->sigma_y, params->stride_y,
                              np_y, mcy, n_mc, w3d);
    if(use_gamma) plane_apply_gamma_inplace(y_out, (int)y_npix, gc);

    fprintf(stderr, "[yuv_galosh] Processing U plane...\n");
    galosh_yuv_denoise_plane(u_src, u_out, cw, ch,
                              params->sigma_c, params->stride_c,
                              np_u, mcu, n_mc, w3d);
    /* no re-gamma for chroma — chroma was never linearised */

    fprintf(stderr, "[yuv_galosh] Processing V plane...\n");
    galosh_yuv_denoise_plane(v_src, v_out, cw, ch,
                              params->sigma_c, params->stride_c,
                              np_v, mcv, n_mc, w3d);
    /* no re-gamma for chroma */

    if(use_gamma)
    {
        dt_free_align(y_lin);
        if(mc_y_lin) { for(int fi=0;fi<n_mc;fi++) dt_free_align(mc_y_lin[fi]); free(mc_y_lin); }
    }

    fprintf(stderr, "[yuv_galosh] Done.\n");
    return 0;
}

/* ================================================================
 * Standalone test harness
 *
 * Input file: raw planar float32 YUV
 *   Layout: Y[H*W] U[cH*cW] V[cH*cW]  (float32, native endian)
 * Output file: same format.
 * ================================================================ */
#ifdef GALOSH_STANDALONE
int main(int argc, char **argv)
{
    if(argc < 5)
    {
        fprintf(stderr,
            "Usage: %s in.yuv out.yuv W H [fmt] [sy] [sc] [stride_y] [stride_c] [gamma] [mc*.yuv ...]\n"
            "  fmt       420 or 444 (default 420)\n"
            "  sy        Y  denoising strength (default 1.0)\n"
            "  sc        UV denoising strength (default 1.0)\n"
            "  stride_y  Y  block stride 2=75%% overlap, 4=50%% (default 2)\n"
            "  stride_c  UV block stride (default 2)\n"
            "  gamma     display gamma for alpha estimation: 1.0=linear RAW (default),\n"
            "            2.2=sRGB approx.  Compensates variance-mean regression so that\n"
            "            Poisson alpha is correctly estimated in gamma-compressed input.\n"
            "  mc*.yuv   motion-compensated frames for 3D mode\n",
            argv[0]);
        return 1;
    }

    const char *fin  = argv[1];
    const char *fout = argv[2];
    const int width  = atoi(argv[3]);
    const int height = atoi(argv[4]);
    if(width <= 0 || height <= 0)
    { fprintf(stderr, "Bad dimensions\n"); return 1; }

    galosh_yuv_params_t p = GALOSH_YUV_DEFAULTS;
    if(argc > 5)  p.yuv_format  = atoi(argv[5]);
    if(argc > 6)  p.sigma_y     = (float)atof(argv[6]);
    if(argc > 7)  p.sigma_c     = (float)atof(argv[7]);
    if(argc > 8)  p.stride_y    = atoi(argv[8]);
    if(argc > 9)  p.stride_c    = atoi(argv[9]);
    if(argc > 10) p.gamma_curve = (float)atof(argv[10]);

    const int cw = (p.yuv_format == GALOSH_YUV_420) ? width  / 2 : width;
    const int ch = (p.yuv_format == GALOSH_YUV_420) ? height / 2 : height;
    const size_t y_npix      = (size_t)width  * height;
    const size_t c_npix      = (size_t)cw     * ch;
    const size_t frame_floats = y_npix + 2 * c_npix;
    const size_t frame_bytes  = frame_floats * sizeof(float);

    float *buf_in = dt_alloc_align_float(frame_floats);
    if(!buf_in) { fprintf(stderr, "alloc failed\n"); return 1; }
    {
        FILE *fi = fopen(fin, "rb");
        if(!fi) { fprintf(stderr, "Cannot open %s\n", fin); return 1; }
        if(fread(buf_in, 1, frame_bytes, fi) != frame_bytes)
            fprintf(stderr, "Warning: short read on %s\n", fin);
        fclose(fi);
    }

    const float *y_in = buf_in;
    const float *u_in = buf_in + y_npix;
    const float *v_in = buf_in + y_npix + c_npix;

    /* Optional MC frames */
    const int mc_arg_start = 11;
    const int n_mc = MAX(0, argc - mc_arg_start);
    p.tr = n_mc / 2;
    float        **mc_buf   = (float **)calloc(MAX(n_mc,1), sizeof(float *));
    const float **mc_y_arr  = (const float **)calloc(MAX(n_mc,1), sizeof(float *));
    const float **mc_u_arr  = (const float **)calloc(MAX(n_mc,1), sizeof(float *));
    const float **mc_v_arr  = (const float **)calloc(MAX(n_mc,1), sizeof(float *));
    for(int i = 0; i < n_mc; i++)
    {
        mc_buf[i] = dt_alloc_align_float(frame_floats);
        if(!mc_buf[i]) { fprintf(stderr, "alloc failed\n"); return 1; }
        FILE *mf = fopen(argv[mc_arg_start + i], "rb");
        if(!mf) { fprintf(stderr, "Cannot open %s\n", argv[mc_arg_start+i]); return 1; }
        if(fread(mc_buf[i], 1, frame_bytes, mf) != frame_bytes)
            fprintf(stderr, "Warning: short read on MC frame %d\n", i);
        fclose(mf);
        mc_y_arr[i] = mc_buf[i];
        mc_u_arr[i] = mc_buf[i] + y_npix;
        mc_v_arr[i] = mc_buf[i] + y_npix + c_npix;
    }
    if(n_mc > 0)
        fprintf(stderr, "[yuv_galosh] 3D mode: n_mc=%d tr=%d\n", n_mc, p.tr);

    float *buf_out = dt_alloc_align_float(frame_floats);
    if(!buf_out) { fprintf(stderr, "alloc failed\n"); return 1; }
    float *y_out = buf_out;
    float *u_out = buf_out + y_npix;
    float *v_out = buf_out + y_npix + c_npix;

    int ret = galosh_yuv_denoise(y_out, u_out, v_out,
                                  y_in,  u_in,  v_in,
                                  width, height,
                                  mc_y_arr, mc_u_arr, mc_v_arr, n_mc, &p);
    if(ret != 0) { fprintf(stderr, "galosh_yuv_denoise failed\n"); return 1; }

    {
        FILE *fo = fopen(fout, "wb");
        if(!fo) { fprintf(stderr, "Cannot open %s for write\n", fout); return 1; }
        fwrite(buf_out, 1, frame_bytes, fo);
        fclose(fo);
    }

    for(int i = 0; i < n_mc; i++) dt_free_align(mc_buf[i]);
    free(mc_buf); free(mc_y_arr); free(mc_u_arr); free(mc_v_arr);
    dt_free_align(buf_in);
    dt_free_align(buf_out);
    return 0;
}
#endif /* GALOSH_STANDALONE */
