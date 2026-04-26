/* galosh_raw_cpu.c — RAW Bayer GALOSH denoiser (CPU reference).
 *
 * Usage: galosh_raw_cpu in.bin out.bin width height
 *          [method] [strength] [luma_str] [chroma_str] [alpha] [sigma_sq]
 * Build:  gcc -O3 -march=native -ffast-math -funroll-loops -fopenmp
 *             -DGALOSH_F -o galosh_raw_cpu galosh_raw_cpu.c -lm
 *
 * Algorithm details in galosh_core.h; this file contains only the RAW-
 * specific orchestration (8x8 WHT-LOSH with 2x2 L/C decomposition and
 * CFA-protected chroma threshold).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "galosh_cpu.h"

/* ----------------------------------------------------------------------
 * Diagonal-quality bench variant flags (set from CLI in main()).
 * Used to A/B/C-test the proposed jaggy-fix combinations without
 * recompiling for every cell of the bench matrix.
 *
 *   g_galosh_stride       : 1 or 2.  stride=1 enables full cycle-spinning
 *                           (variant A).  Default 2 = legacy behaviour.
 *   g_galosh_n_orient     : 1 or 4.  n_orient=4 enables 4-orientation
 *                           WHT averaging (variant B).
 *   g_galosh_lfr_kernel   : 0 = legacy box compute_L_fullres,
 *                           1 = EWA jinc-windowed-jinc-3 (variant C).
 *
 * Both variants A and B affect every WHT-LOSH call (half-res L pass1+
 * pass2 and full-res L pass2).  Variant C affects only the half-res ->
 * full-res L upsample step inside gat_galosh_denoise_rawlc.
 * ---------------------------------------------------------------------- */
static int g_galosh_stride     = 2;
static int g_galosh_n_orient   = 1;
static int g_galosh_lfr_kernel = 0;
static int g_galosh_unified    = 0;  /* legacy exploration path (v4 archived):
                                       *   1 = upsample L+C with EWA-JL3, K15 抜き
                                       *   var=1 不変性破壊で denoise 性能↓
                                       *   採用しない、bench archive 用にだけ残す */
/* GALOSH_RAW_G adopts two structural features unconditionally:
 *
 *   (i)  K16 chroma full-res reconstruction:
 *        c1/c2/c3 are upsampled to full-res via EWA Jinc-Lanczos-3 and
 *        the inverse 2x2 WHT is applied per-pixel with the Bayer-aware
 *        sign tables.  Replaces the legacy block-replicated inverse
 *        (which produced visible 2x2 stair-steps on diagonal edges)
 *        without breaking the K11/K14/K15 var=1 noise invariance.
 *
 *   (ii) Pass1 BayesShrink with MAD-based sigma_Y estimator:
 *        sigma_Y = median(|AC coef|) / 0.6745.  Robust to ~25% outlier
 *        coefficients (Donoho-Johnstone 1995); kills the spatial noise
 *        clusters that the legacy L2 sum_sq estimator over-includes as
 *        "false signal" (the BM3D-CFA-clearable residual).  Pass1's
 *        improved pilot propagates to Pass2's empirical Wiener.
 *
 * Both are baked into the standard pipeline and cannot be disabled.
 * Earlier ad-hoc names "v1 chromaup" / "v5 robust-MAD" referred to the
 * development variants that introduced (i) and (ii) respectively; the
 * final pipeline subsumes both. */

/* K13 4x4 grain-scale-matched experiment (archived 2026-04-26).
 *   8 = GALOSH_RAW_G default (full-res grain dominated by K15 8x8)
 *   4 = K13 4x4 -- theory said matches K15 grain scale, bench showed
 *       K15 dominates and 4x4 just adds noisier-pilot trade-off
 * Exposed for reproducibility / future K15-4x4 ablation work. */
static int g_galosh_k13_block = 8;

#define CFA_IDX_HORIZ  1   /* row=0, col=1 → 0*8+1 */
#define CFA_IDX_VERT   8   /* row=1, col=0 → 1*8+0 */
#define CFA_IDX_DIAG   9   /* row=1, col=1 → 1*8+1 */

static inline int is_cfa_protected(int i)
{
    return (i == 0 || i == CFA_IDX_HORIZ || i == CFA_IDX_VERT || i == CFA_IDX_DIAG);
}

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
 * EWA jinc-windowed-jinc 3-tap (Lanczos-Jinc-3) compute_L_fullres
 * variant — RAW-specific archived experiment (variant C).
 *
 * Replaces the legacy box-like 4-tap reconstruction (compute_L_fullres
 * above) with a circularly symmetric, band-limited reconstruction
 * kernel.  Active only when --lfr-kernel=ewajl3 is passed; otherwise
 * the legacy compute_L_fullres is used.  Archived because pure EWA-JL3
 * reconstruction at the 4 sub-pixel positions used by the inverse 2x2
 * WHT introduces ~1 px shift relative to the legacy box convention
 * (PSNR -0.5 dB in bench).  Kept here for reproducibility of the
 * variant-C numbers in the bench archive.
 *
 * Lives in galosh_raw_cpu.c rather than galosh_core.h because the
 * subpixel offsets and the 4-channel API are RAW-Bayer-specific; the
 * generic 2x EWA-JL3 upsample for v1 chromaup chroma upsampling stays
 * in galosh_core.h as galosh_upsample_2x_ewajl3().
 *
 * Sub-pixel offsets (half-res units, relative to half-res sample
 * centre): TL(-0.25,-0.25), TR(-0.25,+0.25), BL(+0.25,-0.25),
 * BR(+0.25,+0.25).  We sample a 5x5 half-res window and apply a
 * precomputed normalized weight table per sub-pixel.
 * ================================================================ */
static void galosh_compute_L_fullres_ewajl3(const float *restrict L,
                                             const float *restrict C1,
                                             const float *restrict C2,
                                             const float *restrict C3,
                                             const int halfwidth, const int halfheight,
                                             float *restrict L_out)
{
    /* C1/C2/C3 unused: pure spatial reconstruction from the L plane.
     * The chroma-derived L refinement of the legacy formulas is sacrificed
     * for bandlimited diagonal accuracy -- this trade-off is intentional
     * for variant C and is part of what the bench is measuring. */
    (void)C1; (void)C2; (void)C3;

    const int fw = 2 * halfwidth;
    const int fh = 2 * halfheight;

    /* NOTE on alignment.
     *
     * Pure EWA-JL3 reconstruction at the geometric pixel centre offset
     * (TL at (-0.25,-0.25) half-res, etc.) produces a clean L plane that
     * is band-limited but is HALF A PIXEL DOWN-RIGHT of legacy's
     * block-centroid-stored-at-TL convention.  The downstream inverse
     * 2x2 WHT was tuned around legacy's built-in shift, so plain EWA-
     * JL3 ends up displaying RGGB output shifted ~1 px DOWN-RIGHT
     * relative to the base / A / B variants, as flagged in the dartboard
     * bench.
     *
     * Trying to match legacy by sampling at offsets {0, +0.5, +0.5, +0.5}
     * collides with jinc's negative ring at ~r_full=1.41 and produces
     * numerically unstable EWA weights for the BR position (only four
     * samples land inside support, all with negative inner-jinc and
     * positive outer-jinc contributions, leaving wsum tiny and the
     * normalised kernel magnifying small noise into huge values).
     *
     * Conclusion: C-1 (pure EWA-JL3 on L) is structurally incompatible
     * with the legacy WHT inverse pipeline.  The right fix is C-2 --
     * keep legacy compute_L_fullres formulas (which have the correct
     * built-in shift convention) and add a band-limited refinement on
     * top.  Until C-2 is implemented we stay with the original geometric
     * offsets here, which the user already saw shifts by ~1 px but at
     * least produces stable values to feed into the bench. */
    const float subpix[4][2] = {
        { -0.25f, -0.25f },  /* TL */
        { -0.25f, +0.25f },  /* TR */
        { +0.25f, -0.25f },  /* BL */
        { +0.25f, +0.25f },  /* BR */
    };

    /* Precompute 4 normalized 5x5 weight tables. */
    const int W = 2;          /* half-window radius in half-res samples */
    const int kw = 2 * W + 1; /* 5 */
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
                const float r_full = r_half * 2.0f;  /* output pixel units */
                float w_val = 0.0f;
                if(r_full < 3.0f)
                    w_val = galosh_jinc(r_full) * galosh_jinc(r_full / 3.0f);
                weights[si][dy + W][dx + W] = w_val;
                wsum += w_val;
            }
        }
        const float inv_wsum = 1.0f / fmaxf(wsum, 1e-20f);
        for(int dy = 0; dy < kw; dy++)
            for(int dx = 0; dx < kw; dx++)
                weights[si][dy][dx] *= inv_wsum;
    }

    /* Apply weights to produce 4 full-res samples per half-res block. */
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
                        sum += L[(size_t)hyi * halfwidth + hxi]
                             * weights[si][dy + W][dx + W];
                    }
                }
                L_out[(size_t)fr * fw + fc] = sum;
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
 *       Cleveland (J. Am. Stat. Assoc., 1979) -- LOESS (see galosh_core.h)
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

    /* Chroma denoise: LOESS (luma-guided locally-weighted linear regression).
     * Noisy half-res L is the guide; bilateral weight exp(-(Y_i-Y_c)²/(2σ²))
     * excludes specular highlights from silver windows.  Two calls handle the
     * 3 chroma planes (Cb/Cr pair per call).
     *
     * Archived alternatives (removed 2026-04-26):
     *   - v2 chromawiener (Lee residual reinjection): no effect in flat regions
     *     (α=0), worsened textured regions; PSNR -1.16 dB / LPIPS +0.034
     *   - v3 chromawhtlosh (B-cfa: WHT-LOSH on C with low-freq 2x2 protect):
     *     8x8 block grid artifact, PSNR -3.46 dB / SSIM -0.13 / LPIPS +0.12
     * Both removed from code; reproducibility lives in git history. */
    galosh_loess_chroma(luma, chroma1, chroma2, c1_out, c2_out,
                        halfwidth, halfheight, chroma_strength);
    galosh_loess_chroma(luma, chroma3, chroma3, c3_out, c3_pilot /*dummy*/,
                        halfwidth, halfheight, chroma_strength);

    fprintf(stderr, "[rawdenoise] GALOSH: chroma LOESS done (sigma_C=%.3f, R=%d, BW=%.1f)\n",
            chroma_strength, GALOSH_LOESS_RADIUS, GALOSH_LOESS_BW);

    (void)chroma_stride; (void)c1_pilot; (void)c2_pilot;  /* no longer used */
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

    /* Half-res luma stride.  legacy default: stride=2 (= 75% overlap on
     *   8×8 block).  When K13 block = 4, stride=2 means only 50% overlap;
     *   to match the legacy 75% overlap quality (and required to suppress
     *   block grid given the smaller block), default stride to 1 instead.
     *   User may still override via --stride=N. */
    int halfres_stride = g_galosh_stride;
    if(g_galosh_k13_block == 4 && halfres_stride == 2)
      halfres_stride = 1;

    fprintf(stderr, "[rawdenoise] GALOSH_F: half-res luma Pass1+2 "
                     "(%dx%d, sigma_L=%.3f, block=%d, stride=%d, n_orient=%d)\n",
                     halfwidth, halfheight, luma_strength,
                     g_galosh_k13_block, halfres_stride, g_galosh_n_orient);

    /* Pass1+Pass2 with optional 4-orientation averaging.  When
     * g_galosh_k13_block == 4, the half-res L is processed with 4×4 WHT-
     * LOSH so that K15's 8×8 full-res WHT-LOSH and K13's effective
     * full-res grain scale (2 × 4 = 8 pixels) match. */
    /* GALOSH_RAW_G: use_robust_shrink hardcoded to 1 (MAD-based BayesShrink
     * is part of the adopted pipeline definition; see file-top comment). */
    galosh_pass12_multiorient_blocked(luma, l_out, halfwidth, halfheight,
                                       luma_strength, g_galosh_k13_block,
                                       halfres_stride, g_galosh_n_orient,
                                       /*use_robust_shrink=*/1);
    (void)l_pilot;  /* internal pilot now lives inside the wrapper */

    fprintf(stderr, "[rawdenoise] GALOSH_F: half-res luma done\n");

    /* ----------------------------------------------------------------
     * Unified-reconstruction path (g_galosh_unified == 1).
     *
     * Replaces Step (b) compute_L_fullres + Step (c) full-res LOSH +
     * Step (d) block-replicated inverse 2x2 WHT with a single coherent
     * 4-plane upsample + per-pixel reconstruction.
     *
     * Why: legacy Step (b) reconstructs full-res L by mixing half-res L
     * and chroma corrections via 4 box-weighted formulas, then Step (d)
     * REPLICATES half-res chroma to all 4 pixels of every 2x2 block.
     * This produces visible 2x2 stair-stepping on diagonal edges (the
     * specific bug the user observed: hard_baseline clean, guided_c
     * blocky on text edges).
     *
     * Unified path: bandlimit-upsample L AND each of C1/C2/C3 to full
     * res with the same EWA jinc-windowed-jinc-3 kernel, then run the
     * inverse 2x2 WHT PER-PIXEL using the upsampled values and Bayer-
     * pattern-aware sign tables.  Chroma now varies smoothly across
     * the 2x2 block boundary, killing the stair.
     *
     * Theory: the half-res L and C planes are samples of bandlimited
     * signals at 1/2 Nyquist; EWA-JL3 is a near-optimal isotropic
     * reconstruction kernel.  Each Bayer pixel value is then a linear
     * combination of those four reconstructed values with the
     * channel-specific WHT inverse signs -- exactly the per-block
     * inverse the legacy code applies, but evaluated AT EACH PIXEL
     * rather than once per block. */
    if(g_galosh_unified)
    {
      const size_t fullsize_unified = (size_t)width * height;
      float *L_full  = dt_alloc_align_float(fullsize_unified);
      float *C1_full = dt_alloc_align_float(fullsize_unified);
      float *C2_full = dt_alloc_align_float(fullsize_unified);
      float *C3_full = dt_alloc_align_float(fullsize_unified);
      if(!L_full || !C1_full || !C2_full || !C3_full)
      {
        dt_free_align(L_full);  dt_free_align(C1_full);
        dt_free_align(C2_full); dt_free_align(C3_full);
        dt_free_align(l_pilot); dt_free_align(l_out);
        goto cleanup_rawlc;
      }

      galosh_upsample_2x_ewajl3(l_out,  L_full,  halfwidth, halfheight);
      galosh_upsample_2x_ewajl3(c1_out, C1_full, halfwidth, halfheight);
      galosh_upsample_2x_ewajl3(c2_out, C2_full, halfwidth, halfheight);
      galosh_upsample_2x_ewajl3(c3_out, C3_full, halfwidth, halfheight);

      fprintf(stderr, "[rawdenoise] GALOSH_F unified: 4-plane EWA-JL3 upsample done\n");

      /* Channel-slot lookup from 2x2 cell offset, RGGB hardcoded for
       * the standalone CPU reference (filters arg is ignored here as
       * elsewhere in this file -- the darktable build uses
       * galosh_bayer.h which derives co_row/co_col from FC()).  RGGB
       * gives R at (0,0), Gb at (1,0), Gr at (0,1), B at (1,1). */
      int ch_lut[2][2];
      ch_lut[0][0] = 0;  /* R  at TL */
      ch_lut[1][0] = 1;  /* Gb at BL */
      ch_lut[0][1] = 2;  /* Gr at TR */
      ch_lut[1][1] = 3;  /* B  at BR */

      /* Inverse 2x2 WHT signs: for output channel ch the linear
       * combination is L + s1*C1 + s2*C2 + s3*C3, all scaled by 1/2.
       * SIGNS[ch] = {s1, s2, s3} matches the legacy formulas:
       *   R  = (L + C1 + C2 + C3)/2
       *   Gb = (L - C1 + C2 - C3)/2
       *   Gr = (L + C1 - C2 - C3)/2
       *   B  = (L - C1 - C2 + C3)/2
       */
      static const float SIGNS[4][3] = {
        { +1.0f, +1.0f, +1.0f },  /* R  */
        { -1.0f, +1.0f, -1.0f },  /* Gb */
        { +1.0f, -1.0f, -1.0f },  /* Gr */
        { -1.0f, -1.0f, +1.0f },  /* B  */
      };

      const float sg_unified = sigma_gat_ch[0];  /* unified_sigma */

      DT_OMP_FOR()
      for(int fr = 0; fr < height; fr++)
      {
        for(int fc = 0; fc < width; fc++)
        {
          const int ch = ch_lut[fr & 1][fc & 1];
          const size_t pos = (size_t)fr * width + fc;
          const float val = 0.5f * (L_full[pos]
                                  + SIGNS[ch][0] * C1_full[pos]
                                  + SIGNS[ch][1] * C2_full[pos]
                                  + SIGNS[ch][2] * C3_full[pos])
                          + ch_dark_ref[ch];
          out[pos] = gat_inverse_exact(val * sg_unified);
        }
      }

      fprintf(stderr, "[rawdenoise] GALOSH_F unified: per-pixel inverse WHT + inv-GAT done\n");

      dt_free_align(L_full);  dt_free_align(C1_full);
      dt_free_align(C2_full); dt_free_align(C3_full);
      dt_free_align(l_pilot); dt_free_align(l_out);
      goto cleanup_rawlc;
    }

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

    /* Full-res L stride: bench-overrideable for variant A. */
    const int fullres_stride = g_galosh_stride;

    /* compute_L_fullres kernel selection (variant C):
     *   0 = legacy 4-tap box (axis-aligned, square frequency response)
     *   1 = EWA jinc-windowed-jinc-3 (isotropic, circular freq response,
     *       targets diagonal stair-step from upsample aliasing) */
    if(g_galosh_lfr_kernel == 1)
    {
        galosh_compute_L_fullres_ewajl3(luma, chroma1, chroma2, chroma3,
                                         halfwidth, halfheight, L_fr_noisy);
        galosh_compute_L_fullres_ewajl3(l_out, c1_out, c2_out, c3_out,
                                         halfwidth, halfheight, L_fr_pilot);
    }
    else
    {
        compute_L_fullres(luma, chroma1, chroma2, chroma3,
                          halfwidth, halfheight, L_fr_noisy);
        compute_L_fullres(l_out, c1_out, c2_out, c3_out,
                          halfwidth, halfheight, L_fr_pilot);
    }

    fprintf(stderr, "[rawdenoise] GALOSH_F: L_fullres computed (%dx%d), "
                     "full-res stride=%d, lfr_kernel=%s\n",
                     width, height, fullres_stride,
                     g_galosh_lfr_kernel == 1 ? "EWA-JL3" : "box");

    /* Step (c): GALOSH Pass2 on full-res L.
     * Only Pass2 (Wiener) — the pilot from step (b) is already denoised.
     * Multi-orientation averaging (variant B) wraps the pass2 call. */
    galosh_pass2_multiorient(L_fr_noisy, L_fr_pilot, L_fr_den,
                             width, height, luma_strength,
                             fullres_stride, g_galosh_n_orient);

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

        /* ------------------------------------------------------------------
     * GALOSH_RAW_G K16 chroma full-res reconstruction:
     *   - Legacy K14 (compute_L_fullres) + K15 Pass2 produce L_fr_den
     *     in GAT-normalized var=1 space (already done above).
     *   - K16 upsamples c1/c2/c3 to full resolution via EWA Jinc-Lanczos-3
     *     and applies the inverse 2x2 WHT per-pixel with Bayer-aware sign
     *     tables: val[fr,fc] = (L_fr_den + signs[ch] dot C_full) / 2 + dark_ref[ch].
     *   - Replaces the legacy block-replicated inverse (which produced
     *     visible 2x2 stair-steps on diagonal edges) without breaking
     *     K11/K14/K15's var=1 noise invariance.
     *
     * 全 res L_den + EWA-JL3 で full-res 化した chroma + per-pixel
     * inverse 2x2 WHT。block 階段除去 + var=1 不変性保存。 */
    {
      const size_t fullsize = (size_t)width * height;
      float *C1_full = dt_alloc_align_float(fullsize);
      float *C2_full = dt_alloc_align_float(fullsize);
      float *C3_full = dt_alloc_align_float(fullsize);
      if(!C1_full || !C2_full || !C3_full)
      {
        dt_free_align(C1_full); dt_free_align(C2_full); dt_free_align(C3_full);
        dt_free_align(L_fr_den); dt_free_align(l_out);
        goto cleanup_rawlc;
      }

      galosh_upsample_2x_ewajl3(c1_out, C1_full, halfwidth, halfheight);
      galosh_upsample_2x_ewajl3(c2_out, C2_full, halfwidth, halfheight);
      galosh_upsample_2x_ewajl3(c3_out, C3_full, halfwidth, halfheight);

      /* Channel-slot lookup (RGGB hardcoded for the standalone CPU
       * reference; darktable build uses galosh_bayer.h's FC()-derived
       * co_row/co_col which already supports any Bayer pattern).
       *
       * Slot order matches the channel extraction in Phase 1:
       *   0 = R  at TL (fr%2=0, fc%2=0)
       *   1 = Gb at BL (fr%2=1, fc%2=0)
       *   2 = Gr at TR (fr%2=0, fc%2=1)
       *   3 = B  at BR (fr%2=1, fc%2=1) */
      int ch_lut[2][2];
      ch_lut[0][0] = 0;  /* R  */
      ch_lut[1][0] = 1;  /* Gb */
      ch_lut[0][1] = 2;  /* Gr */
      ch_lut[1][1] = 3;  /* B  */

      /* Inverse 2x2 WHT sign tables (per Bayer channel). */
      static const float SIGNS[4][3] = {
        { +1.0f, +1.0f, +1.0f },  /* R  */
        { -1.0f, +1.0f, -1.0f },  /* Gb */
        { +1.0f, -1.0f, -1.0f },  /* Gr */
        { -1.0f, -1.0f, +1.0f },  /* B  */
      };

      const float sg = sigma_gat_ch[0];

      DT_OMP_FOR()
      for(int fr = 0; fr < height; fr++)
      {
        for(int fc = 0; fc < width; fc++)
        {
          const int ch = ch_lut[fr & 1][fc & 1];
          const size_t pos = (size_t)fr * width + fc;
          const float val = 0.5f * (L_fr_den[pos]
                                  + SIGNS[ch][0] * C1_full[pos]
                                  + SIGNS[ch][1] * C2_full[pos]
                                  + SIGNS[ch][2] * C3_full[pos])
                          + ch_dark_ref[ch];
          out[pos] = gat_inverse_exact(val * sg);
        }
      }

      dt_free_align(C1_full); dt_free_align(C2_full); dt_free_align(C3_full);
      fprintf(stderr, "[GALOSH_RAW_G] K16 EWA-JL3 chroma + per-pixel inverse done\n");
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
  /* Strip --stride / --orient / --lfr-kernel flags out of argv before
   * parsing positional args.  Order-independent so the bench harness
   * can append them anywhere on the CLI. */
  int new_argc = 0;
  char *positional[32];
  for(int i = 0; i < argc; i++)
  {
    const char *a = argv[i];
    if(strncmp(a, "--stride=", 9) == 0)
    {
      g_galosh_stride = atoi(a + 9);
      if(g_galosh_stride < 1 || g_galosh_stride > 8) g_galosh_stride = 2;
    }
    else if(strncmp(a, "--orient=", 9) == 0)
    {
      g_galosh_n_orient = atoi(a + 9);
      if(g_galosh_n_orient != 1 && g_galosh_n_orient != 4) g_galosh_n_orient = 1;
    }
    else if(strncmp(a, "--lfr-kernel=", 13) == 0)
    {
      const char *k = a + 13;
      if(strcmp(k, "ewajl3") == 0 || strcmp(k, "ewa-jl3") == 0)
        g_galosh_lfr_kernel = 1;
      else
        g_galosh_lfr_kernel = 0;
    }
    else if(strncmp(a, "--unified=", 10) == 0)
    {
      g_galosh_unified = (atoi(a + 10) != 0);
    }
    else if(strncmp(a, "--k13-block=", 12) == 0)
    {
      const int b = atoi(a + 12);
      if(b == 4 || b == 8) g_galosh_k13_block = b;
      else                 g_galosh_k13_block = 8;
    }
    else if(strncmp(a, "--chroma-up=",     12) == 0 ||
            strncmp(a, "--robust-shrink=", 16) == 0 ||
            strncmp(a, "--chroma-wiener=", 16) == 0 ||
            strncmp(a, "--chroma-method=", 16) == 0)
    {
      /* Deprecated flags from the v0..v6 development variants -- these
       * features are now baked into GALOSH_RAW_G's definition (chroma-up
       * + robust-shrink) or removed as archived experiments (chroma-
       * wiener / chroma-method).  Silently accepted for bench-script
       * backward compatibility. */
    }
    else
    {
      if(new_argc < 32) positional[new_argc++] = (char *)a;
    }
  }
  argv = positional;
  argc = new_argc;

  if(argc < 5)
  {
    fprintf(stderr,
      "Usage: %s input.bin output.bin width height\n"
      "       [method] [strength] [luma_str] [chroma_str]\n"
      "       [alpha] [sigma_sq]\n"
      "       [--stride=N]      (1 or 2, default 2; A enables stride=1)\n"
      "       [--orient=N]      (1 or 4, default 1; B enables orient=4)\n"
      "       [--lfr-kernel=K]  (box | ewajl3, default box; C enables ewajl3)\n"
      "\n"
      "  method:  'galosh' or 'ours' (GALOSH local WHT shrinkage, default)\n"
      "  luma_str:   sigma_L for luma shrinkage (default 0.5, user-tunable)\n"
      "  chroma_str: sigma_C for chroma shrinkage (default 1.0, user-tunable)\n"
      "  alpha:      P-G shot noise gain (auto if <= 0)\n"
      "  sigma_sq:   read noise variance (auto if <= 0)\n",
      argv[0]);
    return 1;
  }

  fprintf(stderr, "[GALOSH_RAW_G] stride=%d orient=%d lfr_kernel=%s "
                  "unified=%d k13_block=%d\n",
          g_galosh_stride, g_galosh_n_orient,
          g_galosh_lfr_kernel == 1 ? "ewajl3" : "box",
          g_galosh_unified, g_galosh_k13_block);

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
