/*
 * Standalone raw denoiser - extracted from darktable rawdenoise.c
 * GAT + BM3D + L/C separation + cross-channel matching
 *
 * Usage: rawdenoise_standalone input.bin output.bin width height
 *        [strength] [luma_str] [chroma_str] [alpha] [sigma_sq]
 *
 * Input/output: 32-bit float raw Bayer mosaic (RGGB), row-major
 *
 * Build: gcc -O3 -march=native -fopenmp -lm -o rawdenoise rawdenoise_standalone.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <omp.h>
#include <time.h>

/* ─── darktable compatibility stubs ─────────────────── */
#define DT_OMP_FOR() _Pragma("omp parallel for schedule(static)")
#define DT_OMP_FOR_NUM(n) _Pragma("omp parallel for schedule(static)")
#define restrict __restrict__

static inline float *dt_alloc_align_float(size_t n) {
    size_t sz = sizeof(float) * ((n + 15) & ~15);
#ifdef _WIN32
    return (float *)_aligned_malloc(sz, 64);
#else
    return (float *)aligned_alloc(64, sz);
#endif
}
static inline void dt_free_align(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

typedef struct { int width, height; } dt_iop_roi_t;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── Fast math ─────────────────────────────────────── */
static inline float fast_expf(float x) {
    /* Schraudolph's fast exp approximation */
    if(x < -80.0f) return 0.0f;
    if(x > 80.0f) return expf(80.0f);
    return expf(x);  /* Use standard for correctness in benchmark */
}

/* ─── BM3D Constants ────────────────────────────────── */
#define BM3D_PATCH_SIZE 8
#define BM3D_PATCH_PIXELS (BM3D_PATCH_SIZE * BM3D_PATCH_SIZE)
#define BM3D_STEP 2
#define BM3D_STEP1_STRIDE 4
#define BM3D_SEARCH_RAD 16         /* default Step1 search radius */
#define BM3D_MAX_MATCHED 32
#define BM3D_TAU_MATCH 400.0f
#define BM3D_LAMBDA_3D 2.7f
#define BM3D_TAU_MATCH_W 400.0f
#define BM3D_SEARCH_RAD2 12        /* default Step2 search radius */
#define BM3D_NUM_CHANNELS 4        /* L, C1, C2, C3 */

/* ─── Kaiser-Bessel Window ──────────────────────────── */
static const float bm3d_kaiser_1d[BM3D_PATCH_SIZE] = {
    0.34012f, 0.59885f, 0.84123f, 0.97659f,
    0.97659f, 0.84123f, 0.59885f, 0.34012f
};
static float bm3d_kaiser_2d[BM3D_PATCH_PIXELS];

static void init_kaiser_window(void) {
    for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
        for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++)
            bm3d_kaiser_2d[dy * BM3D_PATCH_SIZE + dx] =
                bm3d_kaiser_1d[dy] * bm3d_kaiser_1d[dx];
}

/* ─── DCT Basis ─────────────────────────────────────── */
static float dct_basis[BM3D_PATCH_SIZE][BM3D_PATCH_SIZE];
static float dct_basis_t[BM3D_PATCH_SIZE][BM3D_PATCH_SIZE];

static void init_dct_basis(void) {
    for(int k = 0; k < BM3D_PATCH_SIZE; k++) {
        const float alpha = (k == 0) ? 1.0f / sqrtf((float)BM3D_PATCH_SIZE)
                                     : sqrtf(2.0f / (float)BM3D_PATCH_SIZE);
        for(int n = 0; n < BM3D_PATCH_SIZE; n++) {
            dct_basis[k][n] = alpha * cosf((float)M_PI * (2.0f * n + 1.0f) * k
                                           / (2.0f * BM3D_PATCH_SIZE));
            dct_basis_t[n][k] = dct_basis[k][n];
        }
    }
}

/* ─── Comparison function for qsort ─────────────────── */
static int compare_floats_bm3d(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

/* ─── Blind Poisson-Gaussian Noise Estimation ─────────────────── */
/* Estimates alpha (shot noise gain) and sigma_sq (read noise variance)
 * from raw Bayer data using: Var_noise(x) = alpha * E(x) + sigma_sq
 *
 * Algorithm (Foi et al. 2008 lower-envelope approach):
 *   1. Divide each CFA channel into non-overlapping 8×8 blocks (half-res).
 *   2. Per block: compute mean and Laplacian-based noise variance (H+V).
 *      Laplacian L = x[i]-2x[i+1]+x[i+2], Var(L) = 6σ² for iid noise.
 *      Within-block MAD of |L| → robust σ estimate even with a few edges.
 *   3. Bin blocks by mean intensity. Take 5th-20th percentile of block
 *      variance (lower envelope = smoothest patches = noise floor).
 *   4. Robust WLS fit with Huber M-estimator (k from residual MAD).
 */
static void estimate_noise_params(const float *raw, const int width, const int height,
                                   float *out_alpha, float *out_sigma_sq) {
    const int N_BINS = 32;
    const int BLOCK_SZ = 8;  /* block size in half-resolution */
    const int halfwidth = width / 2;
    const int halfheight = height / 2;
    const int offsets[4][2] = {{0,0},{0,1},{1,0},{1,1}};

    const int n_bx = halfwidth / BLOCK_SZ;
    const int n_by = halfheight / BLOCK_SZ;
    const int n_blocks_per_ch = n_bx * n_by;
    const int total_blocks = 4 * n_blocks_per_ch;

    if(total_blocks < 100) {
        *out_alpha = 1.0f; *out_sigma_sq = 0.0f;
        return;
    }

    float *blk_mean = dt_alloc_align_float(total_blocks);
    float *blk_var  = dt_alloc_align_float(total_blocks);
    if(!blk_mean || !blk_var) {
        if(blk_mean) dt_free_align(blk_mean);
        if(blk_var) dt_free_align(blk_var);
        *out_alpha = 1.0f; *out_sigma_sq = 0.0f;
        return;
    }

    /* Step 1-2: Per-block mean and Laplacian noise variance (H+V).
     * Within each 8×8 block: ~96 Laplacian samples (48H + 48V).
     * MAD of |L| is robust even if some edge pixels exist in the block. */
    int bi = 0;
    for(int ch = 0; ch < 4; ch++) {
        const int dy0 = offsets[ch][0], dx0 = offsets[ch][1];
        for(int by = 0; by < n_by; by++) {
            for(int bx = 0; bx < n_bx; bx++) {
                const int y0 = by * BLOCK_SZ;
                const int x0 = bx * BLOCK_SZ;

                double sum = 0;
                int np = 0;
                for(int y = y0; y < y0 + BLOCK_SZ; y++) {
                    for(int x = x0; x < x0 + BLOCK_SZ; x++) {
                        sum += raw[(2*y+dy0) * width + (2*x+dx0)];
                        np++;
                    }
                }
                const float bm = (float)(sum / np);

                /* Collect H+V Laplacians within block */
                float laps[256];
                int nl = 0;
                for(int y = y0; y < y0 + BLOCK_SZ; y++) {
                    for(int x = x0; x < x0 + BLOCK_SZ - 2; x++) {
                        const float v0 = raw[(2*y+dy0) * width + (2*x+dx0)];
                        const float v1 = raw[(2*y+dy0) * width + (2*(x+1)+dx0)];
                        const float v2 = raw[(2*y+dy0) * width + (2*(x+2)+dx0)];
                        laps[nl++] = fabsf(v0 - 2.0f*v1 + v2);
                    }
                }
                for(int y = y0; y < y0 + BLOCK_SZ - 2; y++) {
                    for(int x = x0; x < x0 + BLOCK_SZ; x++) {
                        const float v0 = raw[(2*y+dy0) * width + (2*x+dx0)];
                        const float v1 = raw[(2*(y+1)+dy0) * width + (2*x+dx0)];
                        const float v2 = raw[(2*(y+2)+dy0) * width + (2*x+dx0)];
                        laps[nl++] = fabsf(v0 - 2.0f*v1 + v2);
                    }
                }

                if(nl > 10) {
                    qsort(laps, nl, sizeof(float), compare_floats_bm3d);
                    const float med = laps[nl / 2];
                    const float sigma_lap = med / 0.6745f;
                    blk_var[bi] = (sigma_lap * sigma_lap) / 6.0f;
                } else {
                    blk_var[bi] = 1e10f;
                }
                blk_mean[bi] = bm;
                bi++;
            }
        }
    }
    const int n_total = bi;

    /* Step 3: Bin by mean, take lower envelope (5th-20th percentile). */
    float global_min = 1.0f, global_max = 0.0f;
    for(int i = 0; i < n_total; i++) {
        if(blk_mean[i] > 0.003f && blk_mean[i] < 0.97f) {
            if(blk_mean[i] < global_min) global_min = blk_mean[i];
            if(blk_mean[i] > global_max) global_max = blk_mean[i];
        }
    }
    const float bw = (global_max - global_min) / N_BINS;
    if(bw < 1e-10f) {
        dt_free_align(blk_mean); dt_free_align(blk_var);
        *out_alpha = 1.0f; *out_sigma_sq = 0.0f;
        return;
    }

    float bin_mean_out[32], bin_var_out[32];
    int bin_valid[32], bin_cnt_out[32];

    float *sort_buf = dt_alloc_align_float(n_total);
    if(!sort_buf) {
        dt_free_align(blk_mean); dt_free_align(blk_var);
        *out_alpha = 1.0f; *out_sigma_sq = 0.0f;
        return;
    }

    int n_valid = 0;
    for(int b = 0; b < N_BINS; b++) {
        const float bin_lo = global_min + b * bw;
        const float bin_hi = bin_lo + bw;
        bin_valid[b] = 0;

        int cnt = 0;
        double msum = 0;
        for(int i = 0; i < n_total; i++) {
            if(blk_mean[i] >= bin_lo && blk_mean[i] < bin_hi
               && blk_mean[i] > 0.003f && blk_mean[i] < 0.97f
               && blk_var[i] < 1e9f) {
                sort_buf[cnt] = blk_var[i];
                msum += blk_mean[i];
                cnt++;
            }
        }
        if(cnt < 20) continue;

        /* Sort variances: compute both lower-envelope and median */
        qsort(sort_buf, cnt, sizeof(float), compare_floats_bm3d);
        /* Lower envelope: 5th-20th percentile (for slope/alpha) */
        const int p5  = cnt / 20;
        const int p20 = cnt / 5;
        double var_sum = 0;
        int var_cnt = 0;
        for(int i = p5; i <= p20 && i < cnt; i++) {
            var_sum += sort_buf[i];
            var_cnt++;
        }
        if(var_cnt > 0) {
            bin_var_out[b] = (float)(var_sum / var_cnt);
            bin_mean_out[b] = (float)(msum / cnt);
            bin_cnt_out[b] = var_cnt;
            bin_valid[b] = 1;
            n_valid++;
        }
    }

    dt_free_align(sort_buf);
    dt_free_align(blk_mean);
    dt_free_align(blk_var);

    if(n_valid < 4) {
        *out_alpha = 1.0f; *out_sigma_sq = 0.0f;
        fprintf(stderr, "[noise_est] Too few valid bins (%d)\n", n_valid);
        return;
    }

    /* Step 4: Robust WLS fit  Var = alpha * mean + sigma_sq
     * Huber M-estimator: k = 1.345 * MAD(residuals) / 0.6745 */
    float alpha_est = 0.01f, sigma_sq_est = 0.0f;

    for(int iter = 0; iter < 5; iter++) {
        double huber_k = 1e10;
        if(iter > 0) {
            float resids[32];
            int nr = 0;
            for(int b = 0; b < N_BINS; b++) {
                if(!bin_valid[b]) continue;
                resids[nr++] = fabsf(bin_var_out[b] - (alpha_est * bin_mean_out[b] + sigma_sq_est));
            }
            qsort(resids, nr, sizeof(float), compare_floats_bm3d);
            const float resid_mad = resids[nr / 2] / 0.6745f;
            huber_k = 1.345 * fmax(resid_mad, 1e-12);
        }

        double Sw = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
        for(int b = 0; b < N_BINS; b++) {
            if(!bin_valid[b]) continue;
            double w = (double)bin_cnt_out[b];
            if(iter > 0) {
                const double pred = (double)alpha_est * bin_mean_out[b] + sigma_sq_est;
                const double resid = fabs((double)bin_var_out[b] - pred);
                if(resid > huber_k) w *= huber_k / resid;
            }
            const double x = bin_mean_out[b], y = bin_var_out[b];
            Sw += w; Sx += w*x; Sy += w*y; Sxx += w*x*x; Sxy += w*x*y;
        }
        const double det = Sw * Sxx - Sx * Sx;
        if(fabs(det) > 1e-30) {
            float new_alpha = (float)((Sw * Sxy - Sx * Sy) / det);
            float new_sq    = (float)((Sxx * Sy - Sx * Sxy) / det);
            if(new_alpha > 0) alpha_est = new_alpha;
            if(new_sq >= 0) sigma_sq_est = new_sq;
        }
    }

    alpha_est = fmaxf(alpha_est, 1e-8f);

    /* Pass 2: Estimate sigma_sq from darkest pixels directly.
     * For x ≈ 0: Var(noise) ≈ sigma_sq (shot noise negligible).
     * Find 5th percentile of pixel values, collect Laplacians from
     * pixels below that threshold, MAD → sigma_sq. */
    {
        /* Find intensity distribution to get dark threshold */
        const int samp_max = 50000;
        float *samp = dt_alloc_align_float(samp_max);
        int ns = 0;
        for(int ch = 0; ch < 4 && ns < samp_max; ch++) {
            const int dy0 = offsets[ch][0], dx0 = offsets[ch][1];
            for(int y = 0; y < halfheight && ns < samp_max; y += 3) {
                for(int x = 0; x < halfwidth && ns < samp_max; x += 3) {
                    samp[ns++] = raw[(2*y+dy0) * width + (2*x+dx0)];
                }
            }
        }
        qsort(samp, ns, sizeof(float), compare_floats_bm3d);
        /* Dark threshold: 10th percentile (pixels near black level) */
        const float dark_thresh = samp[ns / 10];
        /* Use dark threshold + a small margin based on expected read noise */
        const float dark_max = dark_thresh + 0.02f;

        /* Collect Laplacians from dark pixels (H+V) */
        float *dark_laps = dt_alloc_align_float(samp_max);
        int ndl = 0;
        for(int ch = 0; ch < 4 && ndl < samp_max; ch++) {
            const int dy0 = offsets[ch][0], dx0 = offsets[ch][1];
            /* Horizontal */
            for(int y = 0; y < halfheight && ndl < samp_max; y++) {
                for(int x = 0; x < halfwidth - 2 && ndl < samp_max; x++) {
                    const float v0 = raw[(2*y+dy0) * width + (2*x+dx0)];
                    const float v1 = raw[(2*y+dy0) * width + (2*(x+1)+dx0)];
                    const float v2 = raw[(2*y+dy0) * width + (2*(x+2)+dx0)];
                    if(v0 > dark_max || v1 > dark_max || v2 > dark_max) continue;
                    dark_laps[ndl++] = fabsf(v0 - 2.0f*v1 + v2);
                }
            }
            /* Vertical */
            for(int y = 0; y < halfheight - 2 && ndl < samp_max; y++) {
                for(int x = 0; x < halfwidth && ndl < samp_max; x++) {
                    const float v0 = raw[(2*y+dy0) * width + (2*x+dx0)];
                    const float v1 = raw[(2*(y+1)+dy0) * width + (2*x+dx0)];
                    const float v2 = raw[(2*(y+2)+dy0) * width + (2*x+dx0)];
                    if(v0 > dark_max || v1 > dark_max || v2 > dark_max) continue;
                    dark_laps[ndl++] = fabsf(v0 - 2.0f*v1 + v2);
                }
            }
        }

        if(ndl > 100) {
            qsort(dark_laps, ndl, sizeof(float), compare_floats_bm3d);
            const float med = dark_laps[ndl / 2];
            const float sigma_lap = med / 0.6745f;
            const float dark_var = (sigma_lap * sigma_lap) / 6.0f;
            /* dark_var ≈ alpha*dark_mean + sigma_sq, dark_mean ≈ dark_thresh/2 */
            const float dark_mean = dark_thresh * 0.5f;
            sigma_sq_est = fmaxf(dark_var - alpha_est * dark_mean, 0.0f);
        }

        dt_free_align(samp);
        dt_free_align(dark_laps);
    }

    fprintf(stderr, "[noise_est] %d bins valid: alpha=%.6f sigma_sq=%.8f\n",
            n_valid, alpha_est, sigma_sq_est);

    *out_alpha = alpha_est;
    *out_sigma_sq = sigma_sq_est;
}

/* ─── GAT Transform ─────────────────────────────────── */
static inline float gat_forward(const float x, const float alpha, const float sigma_sq) {
    return (2.0f / alpha) * sqrtf(fmaxf(alpha * fmaxf(x, 0.0f) + 0.375f * alpha * alpha + sigma_sq, 0.0f));
}

static inline float gat_inverse(const float D, const float alpha, const float sigma_sq) {
    if(D <= 0.0f) return 0.0f;
    const float D_inv = 1.0f / fmaxf(D, 1e-8f);
    const float y = 0.25f * D * D + 0.25f * 1.2247448713916f * D_inv
                  - 11.0f / 8.0f * D_inv * D_inv + 5.0f / 8.0f * 1.2247448713916f * D_inv * D_inv * D_inv
                  - 1.0f / 8.0f;
    return fmaxf(alpha * y - sigma_sq / alpha, 0.0f);
}

/* ─── Foi Exact Unbiased Inverse GAT (Makitalo & Foi 2013) ─── */
/* Precompute E[GAT(Y)] for Y ~ Poisson-Gaussian(x, alpha, sigma_sq)
 * via Poisson summation + Gauss-Hermite quadrature.
 * Invert by binary search on the monotonic table. */

#define GAT_INV_TABLE_SIZE 4096

static struct {
    float d[GAT_INV_TABLE_SIZE]; /* Expected GAT output values (sorted, monotonic) */
    float x[GAT_INV_TABLE_SIZE]; /* Corresponding signal values [0, 1] */
    float d_min, d_max;
    int valid;
} gat_inv_table = {.valid = 0};

/* 7-point Gauss-Hermite quadrature (weight function e^{-x^2}) */
static const double gh_nodes[7]   = {-2.651962, -1.673552, -0.816288, 0.0, 0.816288, 1.673552, 2.651962};
static const double gh_weights[7] = {0.000972, 0.054516, 0.425607, 0.810264, 0.425607, 0.054516, 0.000972};

static void gat_build_inverse_table(const float alpha, const float sigma_sq) {
    const double a = (double)alpha;
    const double sq = (double)sigma_sq;
    const double sig = sqrt(sq);  /* std of read noise */

    for(int i = 0; i < GAT_INV_TABLE_SIZE; i++) {
        const double x_val = (double)i / (double)(GAT_INV_TABLE_SIZE - 1); /* [0, 1] */
        const double lambda = x_val / a;  /* Poisson rate */

        /* Sum over Poisson distribution: E[GAT(Poisson(lambda)*alpha + N(0, sigma_sq))] */
        double expected_gat = 0.0;
        const int k_max = (int)(lambda + 8.0 * sqrt(fmax(lambda, 1.0))) + 20;
        double log_prob = -lambda;  /* log P(0; lambda) */

        for(int k = 0; k <= k_max; k++) {
            if(k > 0) log_prob += log(lambda) - log((double)k);
            const double prob = exp(log_prob);
            if(prob < 1e-15 && k > (int)lambda + 1) break;

            /* E[GAT(k*alpha + Z)] where Z ~ N(0, sigma_sq)
             * = E[(2/alpha) * sqrt(k*alpha^2 + alpha*Z + 3*alpha^2/8 + sigma_sq)]
             * Use Gauss-Hermite for E[sqrt(base + alpha*Z)] */
            const double base = (double)k * a * a + 0.375 * a * a + sq;
            double eg = 0.0;
            for(int g = 0; g < 7; g++) {
                const double z = 1.4142135623730951 * sig * gh_nodes[g]; /* sqrt(2)*sig*node */
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
    fprintf(stderr, "[Foi inverse] Table built: D range [%.4f, %.4f] for alpha=%.6f sigma_sq=%.8f\n",
            gat_inv_table.d_min, gat_inv_table.d_max, alpha, sigma_sq);
}

static inline float gat_inverse_exact(const float D) {
    if(D <= gat_inv_table.d_min) return 0.0f;
    if(D >= gat_inv_table.d_max) return 1.0f;

    /* Binary search on monotonic d[] array */
    int lo = 0, hi = GAT_INV_TABLE_SIZE - 1;
    while(lo < hi - 1) {
        const int mid = (lo + hi) >> 1;
        if(gat_inv_table.d[mid] <= D) lo = mid;
        else hi = mid;
    }

    /* Linear interpolation */
    const float d0 = gat_inv_table.d[lo], d1 = gat_inv_table.d[hi];
    const float t = (D - d0) / fmaxf(d1 - d0, 1e-10f);
    return fmaxf(gat_inv_table.x[lo] + t * (gat_inv_table.x[hi] - gat_inv_table.x[lo]), 0.0f);
}

/* ─── Sigma Estimation ──────────────────────────────── */
static float estimate_gat_sigma(const float *data, const int width, const int height) {
    const int n_samples = (width * height / 3 < 200000) ? width * height / 3 : 200000;
    const int step = (width * (height - 1)) / n_samples;
    float *abs_laps = dt_alloc_align_float(n_samples + 1);
    if(!abs_laps) return 1.0f;

    int count = 0;
    for(int y = 0; y < height && count < n_samples; y++) {
        const float *row = data + (size_t)y * width;
        for(int x = 0; x < width - 2 && count < n_samples; x += (step > 1 ? step : 1)) {
            const float lap = row[x] - 2.0f * row[x + 1] + row[x + 2];
            abs_laps[count++] = fabsf(lap);
        }
    }

    if(count < 100) { dt_free_align(abs_laps); return 1.0f; }
    qsort(abs_laps, count, sizeof(float), compare_floats_bm3d);
    const float mad = abs_laps[count / 2];
    dt_free_align(abs_laps);
    return fmaxf(mad / 1.6521f, 0.01f);
}

/* ─── Hadamard Transform ────────────────────────────── */
static void hadamard_transform(float *data, const int len) {
    for(int step = 1; step < len; step *= 2)
        for(int i = 0; i < len; i += 2 * step)
            for(int j = i; j < i + step; j++) {
                const float a = data[j], b = data[j + step];
                data[j] = a + b;
                data[j + step] = a - b;
            }
    const float norm = 1.0f / sqrtf((float)len);
    for(int i = 0; i < len; i++) data[i] *= norm;
}

/* ─── 2D DCT on 8x8 patch ──────────────────────────── */
static void dct2d_8x8(const float *in, float *out) {
    float tmp[64];
    /* rows */
    for(int r = 0; r < 8; r++) {
        for(int k = 0; k < 8; k++) {
            float s = 0;
            for(int n = 0; n < 8; n++) s += dct_basis[k][n] * in[r * 8 + n];
            tmp[r * 8 + k] = s;
        }
    }
    /* cols */
    for(int c = 0; c < 8; c++) {
        for(int k = 0; k < 8; k++) {
            float s = 0;
            for(int n = 0; n < 8; n++) s += dct_basis[k][n] * tmp[n * 8 + c];
            out[k * 8 + c] = s;
        }
    }
}

static void idct2d_8x8(const float *in, float *out) {
    float tmp[64];
    /* rows with transposed basis */
    for(int r = 0; r < 8; r++) {
        for(int k = 0; k < 8; k++) {
            float s = 0;
            for(int n = 0; n < 8; n++) s += dct_basis_t[k][n] * in[r * 8 + n];
            tmp[r * 8 + k] = s;
        }
    }
    /* cols with transposed basis */
    for(int c = 0; c < 8; c++) {
        for(int k = 0; k < 8; k++) {
            float s = 0;
            for(int n = 0; n < 8; n++) s += dct_basis_t[k][n] * tmp[n * 8 + c];
            out[k * 8 + c] = s;
        }
    }
}

/* ─── Patch SSD ─────────────────────────────────────── */
static inline float patch_ssd(const float *restrict a, const float *restrict b,
                               const int stride, const float early_exit) {
    float ssd = 0.0f;
    for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++) {
        for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++) {
            float d = a[dy * stride + dx] - b[dy * stride + dx];
            ssd += d * d;
        }
        if(ssd > early_exit) return ssd;
    }
    return ssd;
}

/* ─── Block matching ────────────────────────────────── */
typedef struct { int row, col; float dist; } bm3d_match;

static int compare_matches(const void *a, const void *b) {
    float da = ((const bm3d_match *)a)->dist;
    float db = ((const bm3d_match *)b)->dist;
    return (da > db) - (da < db);
}

static int block_match(const float *restrict img, const int width, const int height,
                       const int ref_r, const int ref_c, const float tau,
                       const int search_rad, bm3d_match *matches) {
    int count = 0;
    const float *ref = img + (size_t)ref_r * width + ref_c;

    const int r0 = (ref_r - search_rad > 0) ? ref_r - search_rad : 0;
    const int r1 = (ref_r + search_rad < height - BM3D_PATCH_SIZE)
                   ? ref_r + search_rad : height - BM3D_PATCH_SIZE;
    const int c0 = (ref_c - search_rad > 0) ? ref_c - search_rad : 0;
    const int c1 = (ref_c + search_rad < width - BM3D_PATCH_SIZE)
                   ? ref_c + search_rad : width - BM3D_PATCH_SIZE;

    for(int r = r0; r <= r1; r++) {
        for(int c = c0; c <= c1; c++) {
            const float *cand = img + (size_t)r * width + c;
            float ssd = patch_ssd(ref, cand, width, tau);
            if(ssd < tau) {
                matches[count].row = r;
                matches[count].col = c;
                matches[count].dist = ssd;
                count++;
                if(count >= BM3D_MAX_MATCHED) goto done;
            }
        }
    }
done:
    qsort(matches, count, sizeof(bm3d_match), compare_matches);
    return count;
}

/* ─── Mahalanobis distance block matching (all L/C channels) ─── */
/* d(p,q) = sum_c (1/sigma_c^2) * SSD_c(p,q)
 * Statistically optimal under Gaussian noise (post-GAT). */
static int block_match_multichannel(
    const float *restrict ch[BM3D_NUM_CHANNELS],
    const float match_weight[BM3D_NUM_CHANNELS],  /* 1/sigma_c^2 per channel */
    const int width, const int height,
    const int ref_r, const int ref_c,
    const float tau,
    const int search_rad,
    bm3d_match *matches)
{
    int count = 0;

    const int r0 = (ref_r - search_rad > 0) ? ref_r - search_rad : 0;
    const int r1 = (ref_r + search_rad < height - BM3D_PATCH_SIZE)
                   ? ref_r + search_rad : height - BM3D_PATCH_SIZE;
    const int c0 = (ref_c - search_rad > 0) ? ref_c - search_rad : 0;
    const int c1 = (ref_c + search_rad < width - BM3D_PATCH_SIZE)
                   ? ref_c + search_rad : width - BM3D_PATCH_SIZE;

    for(int r = r0; r <= r1; r++) {
        for(int c = c0; c <= c1; c++) {
            float dist = 0.0f;
            int bail = 0;
            for(int ch_i = 0; ch_i < BM3D_NUM_CHANNELS && !bail; ch_i++) {
                const float *ref_p = ch[ch_i] + (size_t)ref_r * width + ref_c;
                const float *cand_p = ch[ch_i] + (size_t)r * width + c;
                float ssd = 0.0f;
                for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++) {
                    for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++) {
                        const float d = ref_p[dy * width + dx] - cand_p[dy * width + dx];
                        ssd += d * d;
                    }
                }
                dist += match_weight[ch_i] * ssd;
                if(dist > tau) bail = 1;  /* early exit */
            }
            if(!bail && dist < tau) {
                matches[count].row = r;
                matches[count].col = c;
                matches[count].dist = dist;
                count++;
                if(count >= BM3D_MAX_MATCHED) goto done_mc;
            }
        }
    }
done_mc:
    qsort(matches, count, sizeof(bm3d_match), compare_matches);
    return count;
}

/* ─── Noise Variance Map ────────────────────────────── */
static void bm3d_compute_noise_variance_map(const float *restrict noisy,
                                              const float *restrict pilot,
                                              float *restrict varmap,
                                              const int width, const int height) {
    /* Compute residual = noisy - pilot, then local variance via integral image */
    const int radius = 16;
    double *integral = (double *)calloc((size_t)(height + 1) * (width + 1), sizeof(double));
    double *integral_sq = (double *)calloc((size_t)(height + 1) * (width + 1), sizeof(double));
    if(!integral || !integral_sq) {
        free(integral); free(integral_sq);
        /* Fill with 1.0 as fallback */
        for(size_t i = 0; i < (size_t)width * height; i++) varmap[i] = 1.0f;
        return;
    }

    /* Build integral images of residual and residual^2 */
    for(int y = 0; y < height; y++) {
        double row_sum = 0, row_sq = 0;
        for(int x = 0; x < width; x++) {
            float r = noisy[(size_t)y * width + x] - pilot[(size_t)y * width + x];
            row_sum += r;
            row_sq += (double)r * r;
            integral[(y + 1) * (width + 1) + (x + 1)] =
                integral[y * (width + 1) + (x + 1)] + row_sum;
            integral_sq[(y + 1) * (width + 1) + (x + 1)] =
                integral_sq[y * (width + 1) + (x + 1)] + row_sq;
        }
    }

    /* Compute local variance */
    DT_OMP_FOR()
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int y0 = (y - radius > 0) ? y - radius : 0;
            int y1 = (y + radius + 1 < height) ? y + radius + 1 : height;
            int x0 = (x - radius > 0) ? x - radius : 0;
            int x1 = (x + radius + 1 < width) ? x + radius + 1 : width;
            int n = (y1 - y0) * (x1 - x0);

            double sum = integral[y1 * (width + 1) + x1]
                       - integral[y0 * (width + 1) + x1]
                       - integral[y1 * (width + 1) + x0]
                       + integral[y0 * (width + 1) + x0];
            double sum_sq = integral_sq[y1 * (width + 1) + x1]
                          - integral_sq[y0 * (width + 1) + x1]
                          - integral_sq[y1 * (width + 1) + x0]
                          + integral_sq[y0 * (width + 1) + x0];

            double mean = sum / n;
            double var = sum_sq / n - mean * mean;
            varmap[(size_t)y * width + x] = fmaxf((float)var, 1e-6f);
        }
    }

    free(integral);
    free(integral_sq);
}

/* ──────────────────────────────────────────────────────
   Now we include the ACTUAL rawdenoise.c BM3D functions.
   Since the standalone extraction is complex, we'll
   compile rawdenoise.c with stubs and link.
   ────────────────────────────────────────────────────── */

/* For the standalone tool, we re-implement the core pipeline
   using the functions above. This is equivalent to gat_bm3d_denoise(). */

static void bm3d_step1(const float *restrict input, float *restrict output,
                        const int width, const int height, const float sigma) {
    const float tau = BM3D_TAU_MATCH * sigma * sigma;
    const float lambda_thr = BM3D_LAMBDA_3D * sigma;

    /* Thread-local accumulation buffers */
    float *numer = dt_alloc_align_float((size_t)width * height);
    float *denom = dt_alloc_align_float((size_t)width * height);
    if(!numer || !denom) { dt_free_align(numer); dt_free_align(denom); return; }
    memset(numer, 0, sizeof(float) * width * height);
    memset(denom, 0, sizeof(float) * width * height);

    /* Precompute patch means for NL-means weighting */
    const float h_norm = (float)BM3D_PATCH_PIXELS * sigma * sigma;

    for(int ref_r = 0; ref_r <= height - BM3D_PATCH_SIZE; ref_r += BM3D_STEP1_STRIDE) {
        for(int ref_c = 0; ref_c <= width - BM3D_PATCH_SIZE; ref_c += BM3D_STEP1_STRIDE) {
            /* Block matching */
            bm3d_match matches[BM3D_MAX_MATCHED];
            int n_matches = block_match(input, width, height, ref_r, ref_c,
                                        tau, BM3D_SEARCH_RAD, matches);
            if(n_matches < 1) continue;

            /* Round n_matches down to nearest power of 2 (Hadamard requirement) */
            {
                int p = 1;
                while(p * 2 <= n_matches) p *= 2;
                n_matches = p;
            }
            if(n_matches < 1) continue;

            /* Extract patches, 2D DCT, stack into 3D group */
            float group[BM3D_MAX_MATCHED][BM3D_PATCH_PIXELS];
            for(int m = 0; m < n_matches; m++) {
                float patch[BM3D_PATCH_PIXELS];
                for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
                    for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++)
                        patch[dy * BM3D_PATCH_SIZE + dx] =
                            input[(size_t)(matches[m].row + dy) * width + matches[m].col + dx];
                dct2d_8x8(patch, group[m]);
            }

            /* 1D Hadamard along 3rd dimension + hard threshold */
            int n_nonzero = 0;
            for(int i = 0; i < BM3D_PATCH_PIXELS; i++) {
                float col[BM3D_MAX_MATCHED];
                for(int m = 0; m < n_matches; m++) col[m] = group[m][i];
                hadamard_transform(col, n_matches);

                for(int m = 0; m < n_matches; m++) {
                    if(fabsf(col[m]) <= lambda_thr)
                        col[m] = 0.0f;
                    else
                        n_nonzero++;
                }

                hadamard_transform(col, n_matches);
                for(int m = 0; m < n_matches; m++) group[m][i] = col[m];
            }

            /* Aggregate with Kaiser window */
            const float weight = 1.0f / fmaxf((float)n_nonzero, 1.0f);
            for(int m = 0; m < n_matches; m++) {
                /* NL-means weight correction (Salmon 2010) */
                const float noise_floor = h_norm;
                const float corrected_dist = fmaxf(matches[m].dist - noise_floor, 0.0f);
                const float w = fast_expf(-corrected_dist / fmaxf(h_norm, 1e-8f));
                const float total_w = weight * w;

                float denoised[BM3D_PATCH_PIXELS];
                idct2d_8x8(group[m], denoised);

                for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
                    for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++) {
                        const size_t pos = (size_t)(matches[m].row + dy) * width + matches[m].col + dx;
                        const float kw = bm3d_kaiser_2d[dy * BM3D_PATCH_SIZE + dx];
                        numer[pos] += total_w * kw * denoised[dy * BM3D_PATCH_SIZE + dx];
                        denom[pos] += total_w * kw;
                    }
            }
        }
    }

    /* Final division */
    DT_OMP_FOR()
    for(size_t i = 0; i < (size_t)width * height; i++)
        output[i] = denom[i] > 1e-10f ? numer[i] / denom[i] : input[i];

    dt_free_align(numer);
    dt_free_align(denom);
}

static void bm3d_step2(const float *restrict noisy, const float *restrict pilot,
                        float *restrict output, const int width, const int height,
                        const float sigma, const float *restrict noise_varmap,
                        const int pilot_neff,
                        const float *restrict match_on) {
    const float sigma_sq = sigma * sigma;
    const float tau = BM3D_TAU_MATCH_W * sigma * sigma;
    const float *match_img = match_on ? match_on : pilot;

    float *numer = dt_alloc_align_float((size_t)width * height);
    float *denom = dt_alloc_align_float((size_t)width * height);
    if(!numer || !denom) { dt_free_align(numer); dt_free_align(denom); return; }
    memset(numer, 0, sizeof(float) * width * height);
    memset(denom, 0, sizeof(float) * width * height);

    for(int ref_r = 0; ref_r <= height - BM3D_PATCH_SIZE; ref_r += BM3D_STEP) {
        for(int ref_c = 0; ref_c <= width - BM3D_PATCH_SIZE; ref_c += BM3D_STEP) {
            /* Block matching on match_img (or pilot) */
            bm3d_match matches[BM3D_MAX_MATCHED];
            int n_matches = block_match(match_img, width, height, ref_r, ref_c,
                                        tau, BM3D_SEARCH_RAD2, matches);
            if(n_matches < 1) continue;

            /* Round n_matches down to nearest power of 2 (Hadamard requirement) */
            {
                int p = 1;
                while(p * 2 <= n_matches) p *= 2;
                n_matches = p;
            }

            /* Local sigma from variance map or uniform */
            const float local_sigma_sq = fmaxf(
                noise_varmap
                    ? noise_varmap[(size_t)(ref_r + BM3D_PATCH_SIZE / 2) * width
                                   + (ref_c + BM3D_PATCH_SIZE / 2)]
                    : sigma_sq,
                1e-6f);
            const float sigma_sq_pilot = local_sigma_sq / fmaxf((float)pilot_neff, 1.0f);

            /* Extract noisy and pilot patches */
            float group_noisy[BM3D_MAX_MATCHED][BM3D_PATCH_PIXELS];
            float group_pilot[BM3D_MAX_MATCHED][BM3D_PATCH_PIXELS];
            for(int m = 0; m < n_matches; m++) {
                float pn[BM3D_PATCH_PIXELS], pp[BM3D_PATCH_PIXELS];
                for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
                    for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++) {
                        size_t pos = (size_t)(matches[m].row + dy) * width + matches[m].col + dx;
                        pn[dy * BM3D_PATCH_SIZE + dx] = noisy[pos];
                        pp[dy * BM3D_PATCH_SIZE + dx] = pilot[pos];
                    }
                dct2d_8x8(pn, group_noisy[m]);
                dct2d_8x8(pp, group_pilot[m]);
            }

            /* 1D Hadamard + Wiener */
            float wiener_energy = 0.0f;
            for(int i = 0; i < BM3D_PATCH_PIXELS; i++) {
                float col_noisy[BM3D_MAX_MATCHED], col_pilot[BM3D_MAX_MATCHED];
                for(int m = 0; m < n_matches; m++) {
                    col_noisy[m] = group_noisy[m][i];
                    col_pilot[m] = group_pilot[m][i];
                }
                hadamard_transform(col_noisy, n_matches);
                hadamard_transform(col_pilot, n_matches);

                for(int m = 0; m < n_matches; m++) {
                    /* Soft Wiener estimation */
                    const float s2 = col_pilot[m] * col_pilot[m];
                    const float signal_est = s2 * s2 / fmaxf(s2 + sigma_sq_pilot, 1e-10f);
                    const float d = signal_est + local_sigma_sq;
                    const float w = d > 1e-10f ? signal_est / d : 0.0f;
                    col_noisy[m] *= w;
                    wiener_energy += w * w;
                }

                hadamard_transform(col_noisy, n_matches);
                for(int m = 0; m < n_matches; m++) group_noisy[m][i] = col_noisy[m];
            }

            /* Aggregate */
            const float weight = 1.0f / fmaxf(wiener_energy, 1e-6f);
            for(int m = 0; m < n_matches; m++) {
                float denoised[BM3D_PATCH_PIXELS];
                idct2d_8x8(group_noisy[m], denoised);

                for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
                    for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++) {
                        const size_t pos = (size_t)(matches[m].row + dy) * width + matches[m].col + dx;
                        const float kw = bm3d_kaiser_2d[dy * BM3D_PATCH_SIZE + dx];
                        numer[pos] += weight * kw * denoised[dy * BM3D_PATCH_SIZE + dx];
                        denom[pos] += weight * kw;
                    }
            }
        }
    }

    DT_OMP_FOR()
    for(size_t i = 0; i < (size_t)width * height; i++)
        output[i] = denom[i] > 1e-10f ? numer[i] / denom[i] : noisy[i];

    dt_free_align(numer);
    dt_free_align(denom);
}

/* ─── BM3D Step 1 with cross-channel matching ──────── */
static void bm3d_step1_cross(const float *restrict input, float *restrict output,
                              const int width, const int height, const float sigma,
                              const float *restrict match_on) {
    /* Like bm3d_step1 but block-matches on match_on image, processes input */
    const float tau = BM3D_TAU_MATCH * sigma * sigma;
    const float lambda_thr = BM3D_LAMBDA_3D * sigma;
    const float *match_img = match_on ? match_on : input;

    float *numer = dt_alloc_align_float((size_t)width * height);
    float *denom = dt_alloc_align_float((size_t)width * height);
    if(!numer || !denom) { dt_free_align(numer); dt_free_align(denom); return; }
    memset(numer, 0, sizeof(float) * width * height);
    memset(denom, 0, sizeof(float) * width * height);

    const float h_norm = (float)BM3D_PATCH_PIXELS * sigma * sigma;

    for(int ref_r = 0; ref_r <= height - BM3D_PATCH_SIZE; ref_r += BM3D_STEP1_STRIDE) {
        for(int ref_c = 0; ref_c <= width - BM3D_PATCH_SIZE; ref_c += BM3D_STEP1_STRIDE) {
            /* Block matching on match_img (luma pilot for cross-channel) */
            bm3d_match matches[BM3D_MAX_MATCHED];
            int n_matches = block_match(match_img, width, height, ref_r, ref_c,
                                        tau, BM3D_SEARCH_RAD, matches);
            if(n_matches < 1) continue;

            /* Power-of-2 for Hadamard */
            {
                int p = 1;
                while(p * 2 <= n_matches) p *= 2;
                n_matches = p;
            }
            if(n_matches < 1) continue;

            /* Extract patches from INPUT (not match_img), DCT */
            float group[BM3D_MAX_MATCHED][BM3D_PATCH_PIXELS];
            for(int m = 0; m < n_matches; m++) {
                float patch[BM3D_PATCH_PIXELS];
                for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
                    for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++)
                        patch[dy * BM3D_PATCH_SIZE + dx] =
                            input[(size_t)(matches[m].row + dy) * width + matches[m].col + dx];
                dct2d_8x8(patch, group[m]);
            }

            /* Hadamard + hard threshold */
            int n_nonzero = 0;
            for(int i = 0; i < BM3D_PATCH_PIXELS; i++) {
                float col[BM3D_MAX_MATCHED];
                for(int m = 0; m < n_matches; m++) col[m] = group[m][i];
                hadamard_transform(col, n_matches);
                for(int m = 0; m < n_matches; m++) {
                    if(fabsf(col[m]) <= lambda_thr)
                        col[m] = 0.0f;
                    else
                        n_nonzero++;
                }
                hadamard_transform(col, n_matches);
                for(int m = 0; m < n_matches; m++) group[m][i] = col[m];
            }

            /* Aggregate */
            const float weight = 1.0f / fmaxf((float)n_nonzero, 1.0f);
            for(int m = 0; m < n_matches; m++) {
                const float noise_floor = h_norm;
                const float corrected_dist = fmaxf(matches[m].dist - noise_floor, 0.0f);
                const float w = fast_expf(-corrected_dist / fmaxf(h_norm, 1e-8f));
                const float total_w = weight * w;

                float denoised[BM3D_PATCH_PIXELS];
                idct2d_8x8(group[m], denoised);

                for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
                    for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++) {
                        const size_t pos = (size_t)(matches[m].row + dy) * width + matches[m].col + dx;
                        const float kw = bm3d_kaiser_2d[dy * BM3D_PATCH_SIZE + dx];
                        numer[pos] += total_w * kw * denoised[dy * BM3D_PATCH_SIZE + dx];
                        denom[pos] += total_w * kw;
                    }
            }
        }
    }

    DT_OMP_FOR()
    for(size_t i = 0; i < (size_t)width * height; i++)
        output[i] = denom[i] > 1e-10f ? numer[i] / denom[i] : input[i];

    dt_free_align(numer);
    dt_free_align(denom);
}

/* ─── Step 1: Match on L (best SNR), hard threshold in L/C ─────── */
/* L channel has σ_L=0.5 after GAT (vs σ=1 per RGGB channel),
 * giving 4× better distance SNR than any single RGGB channel. */
static void bm3d_step1_rggb_lc(
    const float *restrict rggb[4],                  /* unused (kept for API compat) */
    const float rggb_sigma,                          /* unused */
    const float *restrict lc[BM3D_NUM_CHANNELS],     /* L/C channels; lc[0]=L for matching */
    float *restrict lc_out[BM3D_NUM_CHANNELS],       /* L/C filtered output */
    const int width, const int height,
    const float sigma_lc[BM3D_NUM_CHANNELS],         /* L/C thresholding sigmas */
    const float *restrict noise_varmap,
    const int search_rad)
{
    (void)rggb; (void)rggb_sigma;
    const float luma_sigma = sigma_lc[0];
    const float tau = BM3D_TAU_MATCH * luma_sigma * luma_sigma;
    const float h_norm = (float)BM3D_PATCH_PIXELS * luma_sigma * luma_sigma;

    float *numer[BM3D_NUM_CHANNELS], *denom[BM3D_NUM_CHANNELS];
    const size_t npix = (size_t)width * height;
    for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
        numer[c] = dt_alloc_align_float(npix);
        denom[c] = dt_alloc_align_float(npix);
        if(!numer[c] || !denom[c]) goto cleanup_s1mc;
        memset(numer[c], 0, sizeof(float) * npix);
        memset(denom[c], 0, sizeof(float) * npix);
    }

    {

    for(int ref_r = 0; ref_r <= height - BM3D_PATCH_SIZE; ref_r += BM3D_STEP1_STRIDE) {
        for(int ref_c = 0; ref_c <= width - BM3D_PATCH_SIZE; ref_c += BM3D_STEP1_STRIDE) {
            /* Block matching on L channel (best distance SNR) */
            bm3d_match matches[BM3D_MAX_MATCHED];
            int n_matches = block_match(lc[0], width, height, ref_r, ref_c,
                                        tau, search_rad, matches);
            if(n_matches < 1) continue;

            /* Power-of-2 for Hadamard */
            { int p = 1; while(p * 2 <= n_matches) p *= 2; n_matches = p; }
            if(n_matches < 1) continue;

            /* D: local sigma from variance map */
            float local_sigma[BM3D_NUM_CHANNELS];
            if(noise_varmap) {
                const float local_var = fmaxf(
                    noise_varmap[(size_t)(ref_r + BM3D_PATCH_SIZE/2) * width
                                 + (ref_c + BM3D_PATCH_SIZE/2)], 1e-6f);
                const float var_ratio = sqrtf(local_var) / fmaxf(sigma_lc[0], 1e-6f);
                for(int c = 0; c < BM3D_NUM_CHANNELS; c++)
                    local_sigma[c] = sigma_lc[c] * var_ratio;
            } else {
                for(int c = 0; c < BM3D_NUM_CHANNELS; c++)
                    local_sigma[c] = sigma_lc[c];
            }

            /* Extract patches from L/C channels at RGGB-matched positions, DCT */
            float group[BM3D_NUM_CHANNELS][BM3D_MAX_MATCHED][BM3D_PATCH_PIXELS];
            for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
                for(int m = 0; m < n_matches; m++) {
                    float patch[BM3D_PATCH_PIXELS];
                    for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
                        for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++)
                            patch[dy * BM3D_PATCH_SIZE + dx] =
                                lc[c][(size_t)(matches[m].row + dy) * width + matches[m].col + dx];
                    dct2d_8x8(patch, group[c][m]);
                }
            }

            /* 1D Hadamard + per-channel hard threshold */
            int n_nonzero = 0;
            for(int i = 0; i < BM3D_PATCH_PIXELS; i++) {
                float col[BM3D_NUM_CHANNELS][BM3D_MAX_MATCHED];
                for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
                    for(int m = 0; m < n_matches; m++) col[c][m] = group[c][m][i];
                    hadamard_transform(col[c], n_matches);
                }

                for(int m = 0; m < n_matches; m++) {
                    for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
                        const float lambda_thr = BM3D_LAMBDA_3D * local_sigma[c];
                        if(fabsf(col[c][m]) <= lambda_thr)
                            col[c][m] = 0.0f;
                        else
                            n_nonzero++;
                    }
                }

                for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
                    hadamard_transform(col[c], n_matches);
                    for(int m = 0; m < n_matches; m++) group[c][m][i] = col[c][m];
                }
            }

            /* Aggregate L/C channels with shared weight */
            const float weight = 1.0f / fmaxf((float)n_nonzero, 1.0f);
            for(int m = 0; m < n_matches; m++) {
                const float noise_floor = h_norm;
                const float corrected_dist = fmaxf(matches[m].dist - noise_floor, 0.0f);
                const float w = fast_expf(-corrected_dist / fmaxf(h_norm, 1e-8f));
                const float total_w = weight * w;

                for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
                    float denoised[BM3D_PATCH_PIXELS];
                    idct2d_8x8(group[c][m], denoised);
                    for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
                        for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++) {
                            const size_t pos = (size_t)(matches[m].row + dy) * width + matches[m].col + dx;
                            const float kw = bm3d_kaiser_2d[dy * BM3D_PATCH_SIZE + dx];
                            numer[c][pos] += total_w * kw * denoised[dy * BM3D_PATCH_SIZE + dx];
                            denom[c][pos] += total_w * kw;
                        }
                }
            }
        }
    }
    }

    /* Final division */
    for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
        DT_OMP_FOR()
        for(size_t i = 0; i < npix; i++)
            lc_out[c][i] = denom[c][i] > 1e-10f ? numer[c][i] / denom[c][i] : lc[c][i];
    }

cleanup_s1mc:
    for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
        dt_free_align(numer[c]); dt_free_align(denom[c]);
    }
}

/* ─── Multichannel BM3D Step 2 (A: match once on pilot_L, Wiener all 4) ── */
/* ─── Step 2: Match on RGGB pilot, Wiener in L/C ───── */
static void bm3d_step2_rggb_lc(
    const float *restrict rggb_pilot[4],             /* RGGB pilot for matching */
    const float rggb_sigma,                           /* avg RGGB noise sigma for tau */
    const float *restrict lc_noisy[BM3D_NUM_CHANNELS], /* L/C noisy for Wiener input */
    const float *restrict lc_pilot[BM3D_NUM_CHANNELS], /* L/C pilot for signal estimation */
    float *restrict lc_out[BM3D_NUM_CHANNELS],         /* L/C Wiener output */
    const int width, const int height,
    const float sigma_lc[BM3D_NUM_CHANNELS],
    const float *restrict varmap_lc[BM3D_NUM_CHANNELS],
    const int pilot_neff,
    const int search_rad)
{
    (void)rggb_pilot; (void)rggb_sigma;  /* match on L pilot for best SNR */
    const float luma_sigma = sigma_lc[0];
    const float tau = BM3D_TAU_MATCH_W * luma_sigma * luma_sigma;

    float *numer[BM3D_NUM_CHANNELS], *denom[BM3D_NUM_CHANNELS];
    const size_t npix = (size_t)width * height;
    for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
        numer[c] = dt_alloc_align_float(npix);
        denom[c] = dt_alloc_align_float(npix);
        if(!numer[c] || !denom[c]) goto cleanup_s2mc;
        memset(numer[c], 0, sizeof(float) * npix);
        memset(denom[c], 0, sizeof(float) * npix);
    }

    {
    for(int ref_r = 0; ref_r <= height - BM3D_PATCH_SIZE; ref_r += BM3D_STEP) {
        for(int ref_c = 0; ref_c <= width - BM3D_PATCH_SIZE; ref_c += BM3D_STEP) {
            /* Block matching on L pilot (best distance SNR) */
            bm3d_match matches[BM3D_MAX_MATCHED];
            int n_matches = block_match(lc_pilot[0], width, height, ref_r, ref_c,
                                        tau, search_rad, matches);
            if(n_matches < 1) continue;

            { int p = 1; while(p * 2 <= n_matches) p *= 2; n_matches = p; }

            /* D: local sigma from per-channel L/C variance maps */
            float local_sigma_sq[BM3D_NUM_CHANNELS];
            for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
                const float base_sq = sigma_lc[c] * sigma_lc[c];
                if(varmap_lc[c]) {
                    local_sigma_sq[c] = fmaxf(
                        varmap_lc[c][(size_t)(ref_r + BM3D_PATCH_SIZE/2) * width
                                      + (ref_c + BM3D_PATCH_SIZE/2)], 1e-6f);
                } else {
                    local_sigma_sq[c] = base_sq;
                }
            }

            /* Extract L/C noisy and pilot patches at RGGB-matched positions */
            float gn[BM3D_NUM_CHANNELS][BM3D_MAX_MATCHED][BM3D_PATCH_PIXELS];
            float gp[BM3D_NUM_CHANNELS][BM3D_MAX_MATCHED][BM3D_PATCH_PIXELS];
            for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
                for(int m = 0; m < n_matches; m++) {
                    float pn[BM3D_PATCH_PIXELS], pp[BM3D_PATCH_PIXELS];
                    for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
                        for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++) {
                            size_t pos = (size_t)(matches[m].row + dy) * width + matches[m].col + dx;
                            pn[dy * BM3D_PATCH_SIZE + dx] = lc_noisy[c][pos];
                            pp[dy * BM3D_PATCH_SIZE + dx] = lc_pilot[c][pos];
                        }
                    dct2d_8x8(pn, gn[c][m]);
                    dct2d_8x8(pp, gp[c][m]);
                }
            }

            /* 1D Hadamard + per-channel Wiener */
            float wiener_energy = 0.0f;
            for(int i = 0; i < BM3D_PATCH_PIXELS; i++) {
                float cn[BM3D_NUM_CHANNELS][BM3D_MAX_MATCHED];
                float cp_w[BM3D_NUM_CHANNELS][BM3D_MAX_MATCHED];
                for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
                    for(int m = 0; m < n_matches; m++) {
                        cn[c][m] = gn[c][m][i];
                        cp_w[c][m] = gp[c][m][i];
                    }
                    hadamard_transform(cn[c], n_matches);
                    hadamard_transform(cp_w[c], n_matches);
                }

                for(int m = 0; m < n_matches; m++) {
                    for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
                        const float sigma_sq_pilot = local_sigma_sq[c]
                                                     / fmaxf((float)pilot_neff, 1.0f);
                        const float s2 = cp_w[c][m] * cp_w[c][m];
                        const float signal_est = s2 * s2 / fmaxf(s2 + sigma_sq_pilot, 1e-10f);
                        const float d = signal_est + local_sigma_sq[c];
                        const float w = d > 1e-10f ? signal_est / d : 0.0f;
                        cn[c][m] *= w;
                        wiener_energy += w * w;
                    }
                }

                for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
                    hadamard_transform(cn[c], n_matches);
                    for(int m = 0; m < n_matches; m++) gn[c][m][i] = cn[c][m];
                }
            }

            /* Aggregate all channels with shared weight */
            const float weight = 1.0f / fmaxf(wiener_energy, 1e-6f);
            for(int m = 0; m < n_matches; m++) {
                for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
                    float denoised[BM3D_PATCH_PIXELS];
                    idct2d_8x8(gn[c][m], denoised);
                    for(int dy = 0; dy < BM3D_PATCH_SIZE; dy++)
                        for(int dx = 0; dx < BM3D_PATCH_SIZE; dx++) {
                            const size_t pos = (size_t)(matches[m].row + dy) * width + matches[m].col + dx;
                            const float kw = bm3d_kaiser_2d[dy * BM3D_PATCH_SIZE + dx];
                            numer[c][pos] += weight * kw * denoised[dy * BM3D_PATCH_SIZE + dx];
                            denom[c][pos] += weight * kw;
                        }
                }
            }
        }
    }
    }

    /* Final division */
    for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
        DT_OMP_FOR()
        for(size_t i = 0; i < npix; i++)
            lc_out[c][i] = denom[c][i] > 1e-10f ? numer[c][i] / denom[c][i] : lc_noisy[c][i];
    }

cleanup_s2mc:
    for(int c = 0; c < BM3D_NUM_CHANNELS; c++) {
        dt_free_align(numer[c]); dt_free_align(denom[c]);
    }
}

/* ─── Main pipeline: GAT + BM3D + L/C ──────────────── */
static void gat_bm3d_denoise(const float *restrict in, float *restrict out,
                              const int full_width, const int full_height,
                              const float strength, const int iterations,
                              const float luma_strength, const float chroma_strength,
                              const float alpha, const float sigma_sq,
                              const int search_rad1, const int search_rad2) {
    const int halfwidth = full_width / 2;
    const int halfheight = full_height / 2;
    const size_t chsize = (size_t)halfwidth * halfheight;

    /* Allocate channels */
    float *ch_gat[4] = {NULL, NULL, NULL, NULL};
    float *luma = NULL, *chroma1 = NULL, *chroma2 = NULL, *chroma3 = NULL;
    float *luma_pilot = NULL, *luma_denoised = NULL;
    float *c1_denoised = NULL, *c2_denoised = NULL, *c3_denoised = NULL;
    float *varmap = NULL;

    for(int i = 0; i < 4; i++) ch_gat[i] = dt_alloc_align_float(chsize);
    luma = dt_alloc_align_float(chsize);
    chroma1 = dt_alloc_align_float(chsize);
    chroma2 = dt_alloc_align_float(chsize);
    chroma3 = dt_alloc_align_float(chsize);
    luma_pilot = dt_alloc_align_float(chsize);
    luma_denoised = dt_alloc_align_float(chsize);
    c1_denoised = dt_alloc_align_float(chsize);
    c2_denoised = dt_alloc_align_float(chsize);
    c3_denoised = dt_alloc_align_float(chsize);
    varmap = dt_alloc_align_float(chsize);

    if(!ch_gat[0] || !ch_gat[1] || !ch_gat[2] || !ch_gat[3] ||
       !luma || !chroma1 || !chroma2 || !chroma3 ||
       !luma_pilot || !luma_denoised || !c1_denoised || !c2_denoised ||
       !c3_denoised || !varmap) goto cleanup;

    /* Build Foi exact inverse table (Makitalo & Foi 2013) */
    gat_build_inverse_table(alpha, sigma_sq);

    /* Phase 1: Extract Bayer channels + GAT */
    DT_OMP_FOR()
    for(int y = 0; y < halfheight; y++) {
        for(int x = 0; x < halfwidth; x++) {
            const size_t idx = (size_t)y * halfwidth + x;
            /* RGGB pattern */
            ch_gat[0][idx] = gat_forward(in[(2*y  ) * full_width + (2*x  )], alpha, sigma_sq); /* R */
            ch_gat[1][idx] = gat_forward(in[(2*y  ) * full_width + (2*x+1)], alpha, sigma_sq); /* Gr */
            ch_gat[2][idx] = gat_forward(in[(2*y+1) * full_width + (2*x  )], alpha, sigma_sq); /* Gb */
            ch_gat[3][idx] = gat_forward(in[(2*y+1) * full_width + (2*x+1)], alpha, sigma_sq); /* B */
        }
    }

    /* Estimate sigma per channel (should be ~1.0 after GAT) */
    float gat_sigma[4];
    for(int i = 0; i < 4; i++)
        gat_sigma[i] = estimate_gat_sigma(ch_gat[i], halfwidth, halfheight);
    fprintf(stderr, "[standalone] GAT sigma=[%.4f %.4f %.4f %.4f]\n",
            gat_sigma[0], gat_sigma[1], gat_sigma[2], gat_sigma[3]);

    /* Phase 2: L/C forward transform */
    const float noise_sigma_l  = 0.5f;
    const float noise_sigma_c1 = 0.866f;
    const float noise_sigma_c2 = 0.866f;
    const float noise_sigma_c3 = 0.707f;

    DT_OMP_FOR()
    for(size_t i = 0; i < chsize; i++) {
        const float r = ch_gat[0][i], gr = ch_gat[1][i];
        const float gb = ch_gat[2][i], b = ch_gat[3][i];
        const float l = (r + gr + gb + b) * 0.25f;
        luma[i]    = l / noise_sigma_l;     /* normalize to sigma=1 */
        chroma1[i] = (r - l) / noise_sigma_c1;
        chroma2[i] = (b - l) / noise_sigma_c2;
        chroma3[i] = ((gr - gb) * 0.5f) / noise_sigma_c3;
    }

    /* Estimate actual noise σ in each L/C component via MAD.
     * After GAT + L/C normalization, noise should theoretically be σ=1,
     * but in practice varies due to imperfect noise model estimation.
     * MAD adapts to the actual noise level (same approach as perchannel). */
    const float mad_sigma_l  = estimate_gat_sigma(luma,    halfwidth, halfheight);
    const float mad_sigma_c1 = estimate_gat_sigma(chroma1, halfwidth, halfheight);
    const float mad_sigma_c2 = estimate_gat_sigma(chroma2, halfwidth, halfheight);
    const float mad_sigma_c3 = estimate_gat_sigma(chroma3, halfwidth, halfheight);
    fprintf(stderr, "[standalone] MAD sigma: L=%.4f C1=%.4f C2=%.4f C3=%.4f\n",
            mad_sigma_l, mad_sigma_c1, mad_sigma_c2, mad_sigma_c3);

    /* RGGB matching sigma: average GAT sigma across channels */
    const float rggb_sigma = (gat_sigma[0] + gat_sigma[1] + gat_sigma[2] + gat_sigma[3]) * 0.25f;
    fprintf(stderr, "[standalone] RGGB match sigma: %.4f\n", rggb_sigma);

    /* Allocate chroma pilots, RGGB pilot, and variance maps */
    float *c1_pilot = dt_alloc_align_float(chsize);
    float *c2_pilot = dt_alloc_align_float(chsize);
    float *c3_pilot = dt_alloc_align_float(chsize);
    float *varmap_c1 = dt_alloc_align_float(chsize);
    float *varmap_c2 = dt_alloc_align_float(chsize);
    float *varmap_c3 = dt_alloc_align_float(chsize);
    /* RGGB pilot for Step2 matching (reconstructed from L/C pilot) */
    float *rggb_pilot[4] = {NULL, NULL, NULL, NULL};
    for(int i = 0; i < 4; i++) rggb_pilot[i] = dt_alloc_align_float(chsize);
    if(!c1_pilot || !c2_pilot || !c3_pilot ||
       !varmap_c1 || !varmap_c2 || !varmap_c3 ||
       !rggb_pilot[0] || !rggb_pilot[1] || !rggb_pilot[2] || !rggb_pilot[3]) {
        dt_free_align(c1_pilot); dt_free_align(c2_pilot); dt_free_align(c3_pilot);
        dt_free_align(varmap_c1); dt_free_align(varmap_c2); dt_free_align(varmap_c3);
        for(int i = 0; i < 4; i++) dt_free_align(rggb_pilot[i]);
        goto cleanup;
    }

    /* Phase 3: BM3D denoise — match RGGB, filter L/C */
    for(int iter = 0; iter < iterations; iter++) {
        const float iter_scale = (iter == 0) ? 1.0f : 0.5f;
        const float match_sigma = rggb_sigma * strength * iter_scale;
        const float sigma_l_bm3d  = mad_sigma_l  * strength * luma_strength   * iter_scale;
        const float sigma_c1_bm3d = mad_sigma_c1 * strength * chroma_strength * iter_scale;
        const float sigma_c2_bm3d = mad_sigma_c2 * strength * chroma_strength * iter_scale;
        const float sigma_c3_bm3d = mad_sigma_c3 * strength * chroma_strength * iter_scale;

        fprintf(stderr, "[standalone] iter=%d match_sigma=%.4f sigma_l=%.4f sigma_c=[%.4f %.4f %.4f] search=%d/%d\n",
                iter, match_sigma, sigma_l_bm3d, sigma_c1_bm3d, sigma_c2_bm3d, sigma_c3_bm3d,
                search_rad1, search_rad2);

        /* Step 1: Match on RGGB, hard threshold in L/C */
        {
            const float *rggb_in[4] = {ch_gat[0], ch_gat[1], ch_gat[2], ch_gat[3]};
            const float *lc_in[BM3D_NUM_CHANNELS] = {luma, chroma1, chroma2, chroma3};
            float *lc_out_s1[BM3D_NUM_CHANNELS] = {luma_pilot, c1_pilot, c2_pilot, c3_pilot};
            const float sig_lc[BM3D_NUM_CHANNELS] = {sigma_l_bm3d, sigma_c1_bm3d, sigma_c2_bm3d, sigma_c3_bm3d};
            bm3d_step1_rggb_lc(rggb_in, match_sigma, lc_in, lc_out_s1,
                                halfwidth, halfheight, sig_lc, NULL, search_rad1);
        }

        /* Reconstruct RGGB pilot from L/C pilot (for Step2 matching) */
        DT_OMP_FOR()
        for(size_t i = 0; i < chsize; i++) {
            const float l  = luma_pilot[i] * noise_sigma_l;
            const float c1 = c1_pilot[i]   * noise_sigma_c1;
            const float c2 = c2_pilot[i]   * noise_sigma_c2;
            const float c3 = c3_pilot[i]   * noise_sigma_c3;
            rggb_pilot[0][i] = l + c1;                              /* R */
            rggb_pilot[1][i] = l - 0.5f * c1 - 0.5f * c2 + c3;     /* Gr */
            rggb_pilot[2][i] = l - 0.5f * c1 - 0.5f * c2 - c3;     /* Gb */
            rggb_pilot[3][i] = l + c2;                              /* B */
        }

        /* D: Variance maps from L/C residuals */
        bm3d_compute_noise_variance_map(luma, luma_pilot, varmap,
                                         halfwidth, halfheight);
        bm3d_compute_noise_variance_map(chroma1, c1_pilot, varmap_c1,
                                         halfwidth, halfheight);
        bm3d_compute_noise_variance_map(chroma2, c2_pilot, varmap_c2,
                                         halfwidth, halfheight);
        bm3d_compute_noise_variance_map(chroma3, c3_pilot, varmap_c3,
                                         halfwidth, halfheight);

        /* Step 2: Match on RGGB pilot, Wiener in L/C */
        {
            const float *rggb_p[4] = {rggb_pilot[0], rggb_pilot[1], rggb_pilot[2], rggb_pilot[3]};
            const float *lc_noisy[BM3D_NUM_CHANNELS] = {luma, chroma1, chroma2, chroma3};
            const float *lc_pilot_ch[BM3D_NUM_CHANNELS] = {luma_pilot, c1_pilot, c2_pilot, c3_pilot};
            float *lc_out_s2[BM3D_NUM_CHANNELS] = {luma_denoised, c1_denoised, c2_denoised, c3_denoised};
            const float sig_lc[BM3D_NUM_CHANNELS] = {sigma_l_bm3d, sigma_c1_bm3d, sigma_c2_bm3d, sigma_c3_bm3d};
            const float *vmaps[BM3D_NUM_CHANNELS] = {varmap, varmap_c1, varmap_c2, varmap_c3};
            bm3d_step2_rggb_lc(rggb_p, match_sigma, lc_noisy, lc_pilot_ch, lc_out_s2,
                                halfwidth, halfheight, sig_lc, vmaps,
                                1, search_rad2);
        }

        /* For next iteration, use denoised as input */
        if(iter < iterations - 1) {
            memcpy(luma, luma_denoised, sizeof(float) * chsize);
            memcpy(chroma1, c1_denoised, sizeof(float) * chsize);
            memcpy(chroma2, c2_denoised, sizeof(float) * chsize);
            memcpy(chroma3, c3_denoised, sizeof(float) * chsize);
        }
    }

    dt_free_align(c1_pilot); dt_free_align(c2_pilot); dt_free_align(c3_pilot);
    dt_free_align(varmap_c1); dt_free_align(varmap_c2); dt_free_align(varmap_c3);

    /* Phase 4: Denormalize + Inverse L/C */
    DT_OMP_FOR()
    for(size_t i = 0; i < chsize; i++) {
        const float l  = luma_denoised[i] * noise_sigma_l;
        const float c1 = c1_denoised[i] * noise_sigma_c1;
        const float c2 = c2_denoised[i] * noise_sigma_c2;
        const float c3 = c3_denoised[i] * noise_sigma_c3;
        const float r  = l + c1;
        const float gr = l - 0.5f * c1 - 0.5f * c2 + c3;
        const float gb = l - 0.5f * c1 - 0.5f * c2 - c3;
        const float b  = l + c2;
        ch_gat[0][i] = isfinite(r)  ? r  : l;
        ch_gat[1][i] = isfinite(gr) ? gr : l;
        ch_gat[2][i] = isfinite(gb) ? gb : l;
        ch_gat[3][i] = isfinite(b)  ? b  : l;
    }

    /* Phase 5: Foi exact unbiased inverse GAT + write back */
    DT_OMP_FOR()
    for(int y = 0; y < halfheight; y++) {
        for(int x = 0; x < halfwidth; x++) {
            const size_t idx = (size_t)y * halfwidth + x;
            for(int c = 0; c < 4; c++)
                ch_gat[c][idx] = fmaxf(ch_gat[c][idx], 0.0f);
            out[(2*y  ) * full_width + (2*x  )] = fminf(gat_inverse_exact(ch_gat[0][idx]), 1.0f);
            out[(2*y  ) * full_width + (2*x+1)] = fminf(gat_inverse_exact(ch_gat[1][idx]), 1.0f);
            out[(2*y+1) * full_width + (2*x  )] = fminf(gat_inverse_exact(ch_gat[2][idx]), 1.0f);
            out[(2*y+1) * full_width + (2*x+1)] = fminf(gat_inverse_exact(ch_gat[3][idx]), 1.0f);
        }
    }

cleanup:
    for(int i = 0; i < 4; i++) dt_free_align(ch_gat[i]);
    for(int i = 0; i < 4; i++) dt_free_align(rggb_pilot[i]);
    dt_free_align(luma); dt_free_align(chroma1);
    dt_free_align(chroma2); dt_free_align(chroma3);
    dt_free_align(luma_pilot); dt_free_align(luma_denoised);
    dt_free_align(c1_denoised); dt_free_align(c2_denoised);
    dt_free_align(c3_denoised); dt_free_align(varmap);
}

/* Also provide per-channel BM3D (no L/C) for comparison */
static void bm3d_perchannel_denoise(const float *restrict in, float *restrict out,
                                     const int full_width, const int full_height,
                                     const float alpha, const float sigma_sq) {
    const int halfwidth = full_width / 2;
    const int halfheight = full_height / 2;
    const size_t chsize = (size_t)halfwidth * halfheight;

    float *ch = dt_alloc_align_float(chsize);
    float *pilot = dt_alloc_align_float(chsize);
    float *denoised = dt_alloc_align_float(chsize);
    if(!ch || !pilot || !denoised) {
        dt_free_align(ch); dt_free_align(pilot); dt_free_align(denoised);
        return;
    }

    /* Process each channel independently: R(0,0), Gr(0,1), Gb(1,0), B(1,1) */
    int offsets[4][2] = {{0,0},{0,1},{1,0},{1,1}};

    for(int c = 0; c < 4; c++) {
        int dy0 = offsets[c][0], dx0 = offsets[c][1];

        /* Extract + GAT */
        for(int y = 0; y < halfheight; y++)
            for(int x = 0; x < halfwidth; x++)
                ch[(size_t)y * halfwidth + x] =
                    gat_forward(in[(2*y+dy0) * full_width + (2*x+dx0)], alpha, sigma_sq);

        float sig = estimate_gat_sigma(ch, halfwidth, halfheight);
        fprintf(stderr, "[perchannel] ch%d sigma=%.4f\n", c, sig);

        /* Step 1 + Step 2 */
        bm3d_step1(ch, pilot, halfwidth, halfheight, sig);
        bm3d_step2(ch, pilot, denoised, halfwidth, halfheight, sig, NULL, 1, NULL);

        /* Inverse GAT + write back (with clipping) */
        const float gat_max_pc = gat_forward(1.0f, alpha, sigma_sq) * 1.2f;
        for(int y = 0; y < halfheight; y++)
            for(int x = 0; x < halfwidth; x++) {
                float v = denoised[(size_t)y * halfwidth + x];
                v = fminf(fmaxf(v, 0.0f), gat_max_pc);
                out[(2*y+dy0) * full_width + (2*x+dx0)] = fminf(gat_inverse(v, alpha, sigma_sq), 1.0f);
            }
    }

    dt_free_align(ch); dt_free_align(pilot); dt_free_align(denoised);
}

/* ─── Main ──────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    if(argc < 5) {
        fprintf(stderr,
            "Usage: %s input.bin output.bin width height [method] [strength] "
            "[luma_str] [chroma_str] [alpha] [sigma_sq] [iterations] [search_window]\n"
            "  method: ours | perchannel\n"
            "  search_window: 19-39 (default 33, step1 radius=window/2, step2=window*3/8)\n"
            "  Input/output: 32-bit float raw Bayer (RGGB), row-major\n",
            argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    const int width = atoi(argv[3]);
    const int height = atoi(argv[4]);
    const char *method = argc > 5 ? argv[5] : "ours";
    const float strength = argc > 6 ? atof(argv[6]) : 1.0f;
    const float luma_str = argc > 7 ? atof(argv[7]) : 0.7f;
    const float chroma_str = argc > 8 ? atof(argv[8]) : 1.5f;
    float alpha = argc > 9 ? atof(argv[9]) : -1.0f;      /* -1 = auto-estimate */
    float sigma_sq = argc > 10 ? atof(argv[10]) : -1.0f; /* -1 = auto-estimate */
    const int iterations = argc > 11 ? atoi(argv[11]) : 1;
    const int search_window = argc > 12 ? atoi(argv[12]) : 33; /* B: configurable, default=33 (was BM3D_SEARCH_RAD=16) */
    const int s_rad1 = search_window / 2;        /* Step 1 search radius */
    const int s_rad2 = search_window * 3 / 8;    /* Step 2: smaller (75% ratio) */

    fprintf(stderr, "Raw Denoiser Standalone\n");
    fprintf(stderr, "  Input:  %s (%dx%d)\n", input_file, width, height);
    fprintf(stderr, "  Method: %s\n", method);
    fprintf(stderr, "  Params: strength=%.2f luma=%.2f chroma=%.2f alpha=%.6f sigma_sq=%.8f iter=%d search=%d\n",
            strength, luma_str, chroma_str, alpha, sigma_sq, iterations, search_window);

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
    size_t read = fread(in, sizeof(float), npixels, fin);
    fclose(fin);
    if(read != npixels) {
        fprintf(stderr, "Read %zu floats, expected %zu\n", read, npixels);
        return 1;
    }

    /* Auto-estimate noise parameters if not provided */
    if(alpha < 0 || sigma_sq < 0) {
        estimate_noise_params(in, width, height, &alpha, &sigma_sq);
        fprintf(stderr, "  Auto-estimated: alpha=%.6f sigma_sq=%.8f\n", alpha, sigma_sq);
    }

    /* Process */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if(strcmp(method, "perchannel") == 0) {
        bm3d_perchannel_denoise(in, out, width, height, alpha, sigma_sq);
    } else {
        gat_bm3d_denoise(in, out, width, height, strength, iterations,
                          luma_str, chroma_str, alpha, sigma_sq, s_rad1, s_rad2);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
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
