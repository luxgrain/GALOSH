/* galosh_bayer.h - RAW Bayer GALOSH orchestrator as header-only module.
 *
 * Extracted from GALOSH/standalone/galosh_raw_cpu.c (minus main()).
 * Include from darktable rawdenoise.c after:
 *   #define GALOSH_DT_ENV
 *   #include "galosh_core.h"
 *
 * Contains:
 *   - CFA-protected DC coefficient map for 8x8 WHT chroma
 *   - 2x2 WHT decompose / reconstruct (SSE)
 *   - compute_L_fullres (half-res L + C -> full-res L)
 *   - gat_galosh_denoise_rawlc (the RAW Bayer entry point)
 *
 * All functions are static.
 */
#ifndef GALOSH_BAYER_H
#define GALOSH_BAYER_H

/* The standalone reference (galosh_raw_cpu.c) is built with -DGALOSH_F,
 * which selects the half-res L/C + full-res L refinement pipeline
 * (Phase 4 'GALOSH_F').  Without it the file falls back to the legacy
 * full-res raw GALOSH (stride=4 + chroma replacement) which has known
 * colour-cast bugs and is the path that produced the ringing /
 * colour-shift on the first darktable smoke test.  Force the modern
 * path for the IOP build so it matches the reference. */
#ifndef GALOSH_F
#define GALOSH_F 1
#endif

/* darktable's introspection parser auto-generates a translation unit that
 * #includes rawdenoise.c (and through it this header) only to walk the
 * params struct.  That TU doesn't call gat_galosh_denoise_rawlc or any of
 * its helpers, so -Wunused-function + -Werror = build break.  Suppress
 * the unused-function warning for the static helpers contained here;
 * rawdenoise.c itself does reference them so the real compile path is
 * warning-free. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

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

  /* ----------------------------------------------------------------
   * Pattern-aware 2x2 channel offsets.
   *
   * The standalone reference assumed RGGB (R at TL of each 2x2 block);
   * darktable's `filters` mosaic mask can also encode BGGR / GRBG / GBRG.
   * Locate R via FC, derive B as the diagonal partner and the two
   * greens as the off-diagonal partners.  Channel slot order is kept
   * canonical (c=0 R, c=1 Gb, c=2 Gr, c=3 B) so the WHT decompose,
   * shrinkage, compute_L_fullres, and inverse WHT formulas (which all
   * assume that order) need no change -- only the pixel positions we
   * read from / write to the full-res `in`/`out` buffers shift to
   * follow the actual Bayer phase.
   *
   * dt_iop_roi_t carries no Bayer phase shift, so FC indices line up
   * with the input buffer directly. */
  int rrow = 0, rcol = 0;
  for(int r = 0; r < 2; r++)
    for(int c = 0; c < 2; c++)
      if(FC(r, c, filters) == 0) { rrow = r; rcol = c; }
  const int brow = 1 - rrow, bcol = 1 - rcol;
  const int co_row[4] = { rrow, brow, rrow, brow };  /* R, Gb, Gr, B */
  const int co_col[4] = { rcol, rcol, bcol, bcol };

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
    const int row_offset = co_row[c], col_offset = co_col[c];
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

    /* ----------------------------------------------------------------
     * LOESS chroma (replaces WHT-LOSH pass1+pass2 on C1/C2/C3).
     * Noisy half-res L (`luma`) is the guide; bilateral weight exp(-(Y_i
     * -Y_c)²/(2σ²)) excludes specular highlights from silver windows.
     * Two calls handle 3 chroma planes (Cb/Cr pair per call).
     * ---------------------------------------------------------------- */
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

    /* Half-res luma stride: same logic as chroma stride.
     *   GALOSH_F_HALF or GALOSH_F: stride=2 (75% overlap)
     *   GALOSH_F_FULL:             stride=4 (50% overlap) */
#if defined(GALOSH_F_HALF) || defined(GALOSH_F)
    const int halfres_stride = 2;
#else
    const int halfres_stride = GALOSH_STRIDE;
#endif

    fprintf(stderr, "[GALOSH_RAW_G] half-res luma Pass1+2 "
                     "(%dx%d, sigma_L=%.3f, stride=%d, MAD-BayesShrink)\n",
                     halfwidth, halfheight, luma_strength, halfres_stride);

    /* GALOSH_RAW_G: K13 luma uses Pass1 BayesShrink with MAD-based
     * sigma_Y estimator (use_robust_shrink=1) -- the partial-selection-sort
     * robust noise estimator that kills spatial noise clusters fooling
     * the L2 sum_sq estimator (Donoho-Johnstone 1995).  Single-orientation
     * collapses to plain pass1+pass2 inside the wrapper; n_orient=1 here.
     * SIDD Medium 80-pair: +0.21 dB PSNR / -7.7% LPIPS / -2.5% DISTS. */
    galosh_pass12_multiorient_blocked(luma, l_out, halfwidth, halfheight,
                                       luma_strength, GALOSH_BLOCK_SIZE,
                                       halfres_stride, /*n_orient=*/1,
                                       /*use_robust_shrink=*/1);
    (void)l_pilot;  /* internal pilot now lives inside the wrapper */

    fprintf(stderr, "[GALOSH_RAW_G] half-res luma done\n");

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
    /* GALOSH_RAW_G K16: chroma full-res reconstruction.
     *   Upsample c1/c2/c3 to full-resolution via EWA Jinc-Lanczos-3
     *   (galosh_upsample_2x_ewajl3 in galosh_core.h), then apply the
     *   inverse 2x2 WHT per-pixel using the upsampled chroma + Bayer-
     *   aware sign tables.  Replaces the legacy block-replicated chroma
     *   inverse (which produced visible 2x2 stair-steps on diagonal
     *   edges) without breaking K11/K14/K15's var=1 noise invariance.
     *
     *   The sign tables are *canonical* (R, Gb, Gr, B in fixed order);
     *   the Bayer-pattern-aware slot lookup ch_lut[fr&1][fc&1] handles
     *   RGGB / BGGR / GRBG / GBRG by inverting co_row[] / co_col[]. */
    {
      const size_t fullsize_k16 = (size_t)width * height;
      float *C1_full = dt_alloc_align_float(fullsize_k16);
      float *C2_full = dt_alloc_align_float(fullsize_k16);
      float *C3_full = dt_alloc_align_float(fullsize_k16);
      if(!C1_full || !C2_full || !C3_full)
      {
        dt_free_align(C1_full); dt_free_align(C2_full); dt_free_align(C3_full);
        dt_free_align(L_fr_den); dt_free_align(l_out);
        goto cleanup_rawlc;
      }

      galosh_upsample_2x_ewajl3(c1_out, C1_full, halfwidth, halfheight);
      galosh_upsample_2x_ewajl3(c2_out, C2_full, halfwidth, halfheight);
      galosh_upsample_2x_ewajl3(c3_out, C3_full, halfwidth, halfheight);

      /* Inverse Bayer slot LUT: ch_lut[r%2][c%2] = canonical slot index
       * (0=R, 1=Gb, 2=Gr, 3=B).  Built by inverting co_row[] / co_col[]. */
      int ch_lut[2][2] = {{0,0},{0,0}};
      for(int c = 0; c < 4; c++)
        ch_lut[co_row[c]][co_col[c]] = c;

      /* Inverse 2x2 WHT signs in canonical (R, Gb, Gr, B) order. */
      static const float SIGNS[4][3] = {
        { +1.0f, +1.0f, +1.0f },  /* R  */
        { -1.0f, +1.0f, -1.0f },  /* Gb */
        { +1.0f, -1.0f, -1.0f },  /* Gr */
        { -1.0f, -1.0f, +1.0f },  /* B  */
      };

      const float sg = sigma_gat_ch[0]; /* = unified_sigma */

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


/* ================================================================
 * [LATEST: GALOSH_RAW_O] gat_galosh_denoise_rawlc_o — O pipeline entry.
 *
 * O = L pipeline + multi-scale LOESS chroma pyramid + smoothstep slider
 *   walk over 4 anchors:
 *     slider = 0   → C_h         (= noisy, no denoise)
 *     slider = 1   → C_loess_h   (= L baseline ≈ 30 raw px receptive field)
 *     slider = 2   → C_q_up      (= 60 raw px, mid-grain blotch removal)
 *     slider = 3   → C_e_up      (= 120 raw px, large-grain blotch removal)
 *
 *   Smoothstep blend within each segment for C¹-continuous slider feel.
 *   Math identical to standalone gat_galosh_denoise_rawlc_o; see
 *   standalone/galosh_raw_cpu.c for full design rationale.
 *
 * (日) GALOSH_RAW_O: chroma blotch killer with functional flat-region
 *   slider.  L パイプラインの chroma LOESS を 3-level LOESS pyramid に
 *   拡張、 slider 値で smoothstep blend (= 0 noisy / 1 標準 / 3 max)。
 *   既存 L からの非破壊な改良 (= slider=1 で L と完全一致)。
 * ================================================================ */
static void gat_galosh_denoise_rawlc_o(const float *const restrict in,
                                       float *const restrict out,
                                       const dt_iop_roi_t *const roi,
                                       const float luma_strength,
                                       const float chroma_strength,
                                       const uint32_t filters)
{
  const int width = roi->width, height = roi->height;
  const size_t npixels = (size_t)width * height;
  memcpy(out, in, sizeof(float) * npixels);

  if(luma_strength <= 0.0f) return;
  if(width < GALOSH_BLOCK_SIZE * 2 || height < GALOSH_BLOCK_SIZE * 2) return;

  const int co_row = 0, co_col = 0;
  (void)filters;

  const int halfwidth  = (width  + 1) / 2;
  const int halfheight = (height + 1) / 2;
  const size_t chsize  = (size_t)halfwidth * halfheight;

  float *in_gat = NULL;
  float *L_cs = NULL;
  float *L_cs_den = NULL;
  float *L_pixel = NULL;
  float *L_h_den = NULL;
  float *C1_h = NULL, *C2_h = NULL, *C3_h = NULL;
  float *L_q = NULL, *L_e = NULL;
  float *C1_q = NULL, *C2_q = NULL, *C3_q = NULL;
  float *C1_e = NULL, *C2_e = NULL, *C3_e = NULL;
  float *C1_loess_h = NULL, *C2_loess_h = NULL, *C3_loess_h = NULL;
  float *C1_loess_q = NULL, *C2_loess_q = NULL, *C3_loess_q = NULL;
  float *C1_loess_e = NULL, *C2_loess_e = NULL, *C3_loess_e = NULL;
  float *C1_q_up = NULL, *C2_q_up = NULL, *C3_q_up = NULL;
  float *C1_e_to_q = NULL, *C2_e_to_q = NULL, *C3_e_to_q = NULL;
  float *C1_e_up = NULL, *C2_e_up = NULL, *C3_e_up = NULL;
  float *C1_h_den = NULL, *C2_h_den = NULL, *C3_h_den = NULL;
  float *C1_aligned = NULL, *C2_aligned = NULL, *C3_aligned = NULL;

  in_gat = dt_alloc_align_float(npixels);
  if(!in_gat) goto cleanup_rawlc_o;

  /* Phase 0: blind α / σ² */
  const galosh_noise_params_t np = galosh_estimate_noise(in, width, height);
  gat_build_inverse_table(np.alpha, np.sigma_sq);

  /* Phase 1: GAT forward + per-CFA σ + RMS unified + scalar normalize */
  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++)
    in_gat[i] = gat_forward(in[i], np.alpha, np.sigma_sq);

  float sigma_gat_ch[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  for(int s = 0; s < 4; s++)
  {
    const int ro = ((s & 1)        + co_row) & 1;
    const int co = (((s >> 1) & 1) + co_col) & 1;
    const int hw = (width  - co + 1) / 2;
    const int hh = (height - ro + 1) / 2;
    if(hw < 4 || hh < 4) continue;

    float *tmp = dt_alloc_align_float((size_t)hw * hh);
    if(!tmp) continue;
    DT_OMP_FOR()
    for(int rr = 0; rr < hh; rr++)
      for(int cc = 0; cc < hw; cc++)
        tmp[(size_t)rr * hw + cc] = in_gat[(size_t)(ro + 2*rr) * width + (co + 2*cc)];
    sigma_gat_ch[s] = estimate_gat_sigma_halfres(tmp, hw, hh);
    dt_free_align(tmp);
  }

  const float mean_var = 0.25f * (sigma_gat_ch[0]*sigma_gat_ch[0]
                                + sigma_gat_ch[1]*sigma_gat_ch[1]
                                + sigma_gat_ch[2]*sigma_gat_ch[2]
                                + sigma_gat_ch[3]*sigma_gat_ch[3]);
  const float unified_sigma = sqrtf(fmaxf(mean_var, 1e-12f));
  const float inv_sg = 1.0f / unified_sigma;

  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++) in_gat[i] *= inv_sg;

  /* Phase 2: dark_ref IRLS + per-pixel CFA-aware subtract */
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
      double sum_w0 = 0.0, sum_w1 = 0.0, sum_w2 = 0.0, sum_w3 = 0.0;

      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_w,sum_w0,sum_w1,sum_w2,sum_w3)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          sum_w  += w;
          sum_w0 += w * g0;
          sum_w1 += w * g1;
          sum_w2 += w * g2;
          sum_w3 += w * g3;
        }

      const double inv_sw = 1.0 / fmax(sum_w, 1e-20);
      ch_dark_ref[0] = (float)(sum_w0 * inv_sw);
      ch_dark_ref[1] = (float)(sum_w1 * inv_sw);
      ch_dark_ref[2] = (float)(sum_w2 * inv_sw);
      ch_dark_ref[3] = (float)(sum_w3 * inv_sw);

      if(iter == n_iter) break;

      double sum_wresid2 = 0.0;
      double sum_wW = 0.0;
      const float dr0 = ch_dark_ref[0], dr1 = ch_dark_ref[1];
      const float dr2 = ch_dark_ref[2], dr3 = ch_dark_ref[3];
      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_wresid2,sum_wW)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          const double d0 = (double)g0 - dr0;
          const double d1 = (double)g1 - dr1;
          const double d2 = (double)g2 - dr2;
          const double d3 = (double)g3 - dr3;
          const double resid2 = d0*d0 + d1*d1 + d2*d2 + d3*d3;
          sum_wW      += w;
          sum_wresid2 += w * resid2 * 0.25;
        }
      const double inv_sw2 = 1.0 / fmax(sum_wW, 1e-20);
      const double measured_std = sqrt(fmax(sum_wresid2 * inv_sw2, 1e-20));
      const double ratio = 1.0 / measured_std;
      s_scale *= sqrt(ratio);
      if(s_scale < s_min) s_scale = s_min;
      if(s_scale > s_max) s_scale = s_max;
    }

    DT_OMP_FOR()
    for(int r = 0; r < height; r++)
    {
      const int r_off = (r - co_row) & 1;
      for(int c = 0; c < width; c++)
      {
        const int c_off = (c - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        in_gat[(size_t)r * width + c] -= ch_dark_ref[slot];
      }
    }
  }

  /* Phase 3: stride=1 forward 2x2 WHT (L-only) */
  L_cs = dt_alloc_align_float(npixels);
  if(!L_cs) goto cleanup_rawlc_o;
  gat_h_forward_l_only_stride1(in_gat, L_cs, width, height);

  /* Phase 4: half-res chroma extract */
  C1_h = dt_alloc_align_float(chsize);
  C2_h = dt_alloc_align_float(chsize);
  C3_h = dt_alloc_align_float(chsize);
  if(!C1_h || !C2_h || !C3_h) goto cleanup_rawlc_o;
  gat_j_forward_c_halfres(in_gat, C1_h, C2_h, C3_h,
                           width, height, halfwidth, halfheight);
  dt_free_align(in_gat); in_gat = NULL;

  /* Phase 5: single-scale WHT-LOSH on L_cs */
  L_cs_den = dt_alloc_align_float(npixels);
  if(!L_cs_den) goto cleanup_rawlc_o;
  galosh_pass12_multiorient_blocked(L_cs, L_cs_den, width, height,
                                     luma_strength, /*block=*/8,
                                     /*stride=*/2, /*n_orient=*/1,
                                     /*use_robust_shrink=*/1);
  dt_free_align(L_cs); L_cs = NULL;

  /* Phase 6: L_pixel + L_h_den */
  L_pixel = dt_alloc_align_float(npixels);
  L_h_den = dt_alloc_align_float(chsize);
  if(!L_pixel || !L_h_den) goto cleanup_rawlc_o;

  gat_i_lpixel_overlap_avg(L_cs_den, L_pixel, width, height);

  DT_OMP_FOR()
  for(int hr = 0; hr < halfheight; hr++)
  {
    const int fr = 2 * hr;
    if(fr >= height) continue;
    for(int hc = 0; hc < halfwidth; hc++)
    {
      const int fc = 2 * hc;
      if(fc >= width) continue;
      L_h_den[(size_t)hr * halfwidth + hc] = L_cs_den[(size_t)fr * width + fc];
    }
  }
  dt_free_align(L_cs_den); L_cs_den = NULL;

  /* Phase 7: multi-scale LOESS pyramid + L-guided K16 upsample chain */
  const int cq_w = halfwidth / 2;
  const int cq_h = halfheight / 2;
  const int ce_w = cq_w / 2;
  const int ce_h = cq_h / 2;
  const size_t cqsize = (size_t)cq_w * cq_h;
  const size_t cesize = (size_t)ce_w * ce_h;

  if(cq_w < 4 || cq_h < 4 || ce_w < 4 || ce_h < 4)
    goto cleanup_rawlc_o;  /* Image too small for 3-level pyramid */

  /* L pyramid */
  L_q = dt_alloc_align_float(cqsize);
  L_e = dt_alloc_align_float(cesize);
  if(!L_q || !L_e) goto cleanup_rawlc_o;
  gat_box_downsample_2x(L_h_den, L_q, halfwidth, halfheight);
  gat_box_downsample_2x(L_q,     L_e, cq_w,      cq_h);

  /* Chroma pyramid inputs */
  C1_q = dt_alloc_align_float(cqsize);
  C2_q = dt_alloc_align_float(cqsize);
  C3_q = dt_alloc_align_float(cqsize);
  C1_e = dt_alloc_align_float(cesize);
  C2_e = dt_alloc_align_float(cesize);
  C3_e = dt_alloc_align_float(cesize);
  if(!C1_q || !C2_q || !C3_q || !C1_e || !C2_e || !C3_e) goto cleanup_rawlc_o;
  gat_box_downsample_2x(C1_h, C1_q, halfwidth, halfheight);
  gat_box_downsample_2x(C2_h, C2_q, halfwidth, halfheight);
  gat_box_downsample_2x(C3_h, C3_q, halfwidth, halfheight);
  gat_box_downsample_2x(C1_q, C1_e, cq_w, cq_h);
  gat_box_downsample_2x(C2_q, C2_e, cq_w, cq_h);
  gat_box_downsample_2x(C3_q, C3_e, cq_w, cq_h);

  /* LOESS at each scale */
  C1_loess_h = dt_alloc_align_float(chsize);
  C2_loess_h = dt_alloc_align_float(chsize);
  C3_loess_h = dt_alloc_align_float(chsize);
  C1_loess_q = dt_alloc_align_float(cqsize);
  C2_loess_q = dt_alloc_align_float(cqsize);
  C3_loess_q = dt_alloc_align_float(cqsize);
  C1_loess_e = dt_alloc_align_float(cesize);
  C2_loess_e = dt_alloc_align_float(cesize);
  C3_loess_e = dt_alloc_align_float(cesize);
  if(!C1_loess_h || !C2_loess_h || !C3_loess_h ||
     !C1_loess_q || !C2_loess_q || !C3_loess_q ||
     !C1_loess_e || !C2_loess_e || !C3_loess_e)
    goto cleanup_rawlc_o;

  const float loess_strength = 1.0f;  /* fixed; slider walk acts on segment blend below */
  galosh_loess_chroma_3ch_r(L_h_den, C1_h, C2_h, C3_h,
                             C1_loess_h, C2_loess_h, C3_loess_h,
                             halfwidth, halfheight, loess_strength,
                             GALOSH_LOESS_RADIUS, GALOSH_LOESS_BW);
  galosh_loess_chroma_3ch_r(L_q, C1_q, C2_q, C3_q,
                             C1_loess_q, C2_loess_q, C3_loess_q,
                             cq_w, cq_h, loess_strength,
                             GALOSH_LOESS_RADIUS, GALOSH_LOESS_BW);
  galosh_loess_chroma_3ch_r(L_e, C1_e, C2_e, C3_e,
                             C1_loess_e, C2_loess_e, C3_loess_e,
                             ce_w, ce_h, loess_strength,
                             GALOSH_LOESS_RADIUS, GALOSH_LOESS_BW);

  dt_free_align(C1_q); dt_free_align(C2_q); dt_free_align(C3_q);
  C1_q = C2_q = C3_q = NULL;
  dt_free_align(C1_e); dt_free_align(C2_e); dt_free_align(C3_e);
  C1_e = C2_e = C3_e = NULL;

  /* K16 upsample chain (= 3 calls with stride correction) */
  C1_q_up = dt_alloc_align_float(chsize);
  C2_q_up = dt_alloc_align_float(chsize);
  C3_q_up = dt_alloc_align_float(chsize);
  C1_e_up = dt_alloc_align_float(chsize);
  C2_e_up = dt_alloc_align_float(chsize);
  C3_e_up = dt_alloc_align_float(chsize);
  if(!C1_q_up || !C2_q_up || !C3_q_up ||
     !C1_e_up || !C2_e_up || !C3_e_up)
    goto cleanup_rawlc_o;

  /* quarter → half (L_h_den guide) — stride-corrected */
  {
    const int fw = 2 * cq_w;
    const int fh = 2 * cq_h;
    const size_t fsize = (size_t)fw * fh;

    float *L_for_q = dt_alloc_align_float(fsize);
    float *C1_q_up_raw = dt_alloc_align_float(fsize);
    float *C2_q_up_raw = dt_alloc_align_float(fsize);
    float *C3_q_up_raw = dt_alloc_align_float(fsize);
    if(!L_for_q || !C1_q_up_raw || !C2_q_up_raw || !C3_q_up_raw)
    {
      dt_free_align(L_for_q);
      dt_free_align(C1_q_up_raw); dt_free_align(C2_q_up_raw); dt_free_align(C3_q_up_raw);
      goto cleanup_rawlc_o;
    }
    gat_crop_2d_topleft(L_h_den, halfwidth, halfheight, L_for_q, fw, fh);
    gat_k16_joint_bilateral_upsample(C1_loess_q, C2_loess_q, C3_loess_q, L_for_q,
                                      C1_q_up_raw, C2_q_up_raw, C3_q_up_raw,
                                      cq_w, cq_h, /*BW=*/1.5f);
    gat_pad_2d_edge(C1_q_up_raw, fw, fh, C1_q_up, halfwidth, halfheight);
    gat_pad_2d_edge(C2_q_up_raw, fw, fh, C2_q_up, halfwidth, halfheight);
    gat_pad_2d_edge(C3_q_up_raw, fw, fh, C3_q_up, halfwidth, halfheight);
    dt_free_align(L_for_q);
    dt_free_align(C1_q_up_raw);
    dt_free_align(C2_q_up_raw);
    dt_free_align(C3_q_up_raw);
  }

  /* eighth → quarter → half (chained, stride-corrected) */
  {
    const int fw_eq = 2 * ce_w;
    const int fh_eq = 2 * ce_h;
    const size_t fsize_eq = (size_t)fw_eq * fh_eq;

    float *L_for_e = dt_alloc_align_float(fsize_eq);
    float *C1_e_to_q_raw = dt_alloc_align_float(fsize_eq);
    float *C2_e_to_q_raw = dt_alloc_align_float(fsize_eq);
    float *C3_e_to_q_raw = dt_alloc_align_float(fsize_eq);
    if(!L_for_e || !C1_e_to_q_raw || !C2_e_to_q_raw || !C3_e_to_q_raw)
    {
      dt_free_align(L_for_e);
      dt_free_align(C1_e_to_q_raw); dt_free_align(C2_e_to_q_raw); dt_free_align(C3_e_to_q_raw);
      goto cleanup_rawlc_o;
    }
    gat_crop_2d_topleft(L_q, cq_w, cq_h, L_for_e, fw_eq, fh_eq);
    gat_k16_joint_bilateral_upsample(C1_loess_e, C2_loess_e, C3_loess_e, L_for_e,
                                      C1_e_to_q_raw, C2_e_to_q_raw, C3_e_to_q_raw,
                                      ce_w, ce_h, /*BW=*/1.5f);
    C1_e_to_q = dt_alloc_align_float(cqsize);
    C2_e_to_q = dt_alloc_align_float(cqsize);
    C3_e_to_q = dt_alloc_align_float(cqsize);
    if(!C1_e_to_q || !C2_e_to_q || !C3_e_to_q)
    {
      dt_free_align(L_for_e);
      dt_free_align(C1_e_to_q_raw); dt_free_align(C2_e_to_q_raw); dt_free_align(C3_e_to_q_raw);
      goto cleanup_rawlc_o;
    }
    gat_pad_2d_edge(C1_e_to_q_raw, fw_eq, fh_eq, C1_e_to_q, cq_w, cq_h);
    gat_pad_2d_edge(C2_e_to_q_raw, fw_eq, fh_eq, C2_e_to_q, cq_w, cq_h);
    gat_pad_2d_edge(C3_e_to_q_raw, fw_eq, fh_eq, C3_e_to_q, cq_w, cq_h);
    dt_free_align(L_for_e);
    dt_free_align(C1_e_to_q_raw);
    dt_free_align(C2_e_to_q_raw);
    dt_free_align(C3_e_to_q_raw);

    const int fw_qh = 2 * cq_w;
    const int fh_qh = 2 * cq_h;
    const size_t fsize_qh = (size_t)fw_qh * fh_qh;

    float *L_for_q2 = dt_alloc_align_float(fsize_qh);
    float *C1_e_up_raw = dt_alloc_align_float(fsize_qh);
    float *C2_e_up_raw = dt_alloc_align_float(fsize_qh);
    float *C3_e_up_raw = dt_alloc_align_float(fsize_qh);
    if(!L_for_q2 || !C1_e_up_raw || !C2_e_up_raw || !C3_e_up_raw)
    {
      dt_free_align(L_for_q2);
      dt_free_align(C1_e_up_raw); dt_free_align(C2_e_up_raw); dt_free_align(C3_e_up_raw);
      goto cleanup_rawlc_o;
    }
    gat_crop_2d_topleft(L_h_den, halfwidth, halfheight, L_for_q2, fw_qh, fh_qh);
    gat_k16_joint_bilateral_upsample(C1_e_to_q, C2_e_to_q, C3_e_to_q, L_for_q2,
                                      C1_e_up_raw, C2_e_up_raw, C3_e_up_raw,
                                      cq_w, cq_h, /*BW=*/1.5f);
    gat_pad_2d_edge(C1_e_up_raw, fw_qh, fh_qh, C1_e_up, halfwidth, halfheight);
    gat_pad_2d_edge(C2_e_up_raw, fw_qh, fh_qh, C2_e_up, halfwidth, halfheight);
    gat_pad_2d_edge(C3_e_up_raw, fw_qh, fh_qh, C3_e_up, halfwidth, halfheight);
    dt_free_align(L_for_q2);
    dt_free_align(C1_e_up_raw);
    dt_free_align(C2_e_up_raw);
    dt_free_align(C3_e_up_raw);
  }

  dt_free_align(C1_loess_q); dt_free_align(C2_loess_q); dt_free_align(C3_loess_q);
  C1_loess_q = C2_loess_q = C3_loess_q = NULL;
  dt_free_align(C1_loess_e); dt_free_align(C2_loess_e); dt_free_align(C3_loess_e);
  C1_loess_e = C2_loess_e = C3_loess_e = NULL;
  dt_free_align(C1_e_to_q); dt_free_align(C2_e_to_q); dt_free_align(C3_e_to_q);
  C1_e_to_q = C2_e_to_q = C3_e_to_q = NULL;
  dt_free_align(L_q); dt_free_align(L_e);
  L_q = L_e = NULL;

  /* Phase 8: smoothstep slider walk */
  C1_h_den = dt_alloc_align_float(chsize);
  C2_h_den = dt_alloc_align_float(chsize);
  C3_h_den = dt_alloc_align_float(chsize);
  if(!C1_h_den || !C2_h_den || !C3_h_den) goto cleanup_rawlc_o;

  {
    const float s = chroma_strength;
    int segment;
    float t_raw;
    const float *A1, *A2, *A3, *B1, *B2, *B3;
    if(s <= 0.0f)
    {
      segment = -1;
      t_raw = 0.0f;
      A1 = A2 = A3 = B1 = B2 = B3 = NULL;
    }
    else if(s <= 1.0f)
    {
      segment = 0;
      t_raw = s;
      A1 = C1_h;       A2 = C2_h;       A3 = C3_h;
      B1 = C1_loess_h; B2 = C2_loess_h; B3 = C3_loess_h;
    }
    else if(s <= 2.0f)
    {
      segment = 1;
      t_raw = s - 1.0f;
      A1 = C1_loess_h; A2 = C2_loess_h; A3 = C3_loess_h;
      B1 = C1_q_up;    B2 = C2_q_up;    B3 = C3_q_up;
    }
    else if(s <= 3.0f)
    {
      segment = 2;
      t_raw = s - 2.0f;
      A1 = C1_q_up;    A2 = C2_q_up;    A3 = C3_q_up;
      B1 = C1_e_up;    B2 = C2_e_up;    B3 = C3_e_up;
    }
    else
    {
      segment = 3;
      t_raw = 1.0f;
      A1 = C1_e_up;    A2 = C2_e_up;    A3 = C3_e_up;
      B1 = NULL;       B2 = NULL;       B3 = NULL;
    }

    if(segment < 0)
    {
      memcpy(C1_h_den, C1_h, sizeof(float) * chsize);
      memcpy(C2_h_den, C2_h, sizeof(float) * chsize);
      memcpy(C3_h_den, C3_h, sizeof(float) * chsize);
    }
    else if(segment >= 3 || B1 == NULL)
    {
      memcpy(C1_h_den, A1, sizeof(float) * chsize);
      memcpy(C2_h_den, A2, sizeof(float) * chsize);
      memcpy(C3_h_den, A3, sizeof(float) * chsize);
    }
    else
    {
      const float t = t_raw * t_raw * (3.0f - 2.0f * t_raw);
      const float oneMt = 1.0f - t;
      DT_OMP_FOR()
      for(size_t i = 0; i < chsize; i++)
      {
        C1_h_den[i] = oneMt * A1[i] + t * B1[i];
        C2_h_den[i] = oneMt * A2[i] + t * B2[i];
        C3_h_den[i] = oneMt * A3[i] + t * B3[i];
      }
    }
  }

  dt_free_align(C1_h); dt_free_align(C2_h); dt_free_align(C3_h);
  C1_h = C2_h = C3_h = NULL;
  dt_free_align(C1_loess_h); dt_free_align(C2_loess_h); dt_free_align(C3_loess_h);
  C1_loess_h = C2_loess_h = C3_loess_h = NULL;
  dt_free_align(C1_q_up); dt_free_align(C2_q_up); dt_free_align(C3_q_up);
  C1_q_up = C2_q_up = C3_q_up = NULL;
  dt_free_align(C1_e_up); dt_free_align(C2_e_up); dt_free_align(C3_e_up);
  C1_e_up = C2_e_up = C3_e_up = NULL;
  dt_free_align(L_h_den); L_h_den = NULL;

  /* Phase 9: final K16 joint bilateral upsample to full-res */
  C1_aligned = dt_alloc_align_float(npixels);
  C2_aligned = dt_alloc_align_float(npixels);
  C3_aligned = dt_alloc_align_float(npixels);
  if(!C1_aligned || !C2_aligned || !C3_aligned) goto cleanup_rawlc_o;

  gat_k16_joint_bilateral_upsample(C1_h_den, C2_h_den, C3_h_den, L_pixel,
                                    C1_aligned, C2_aligned, C3_aligned,
                                    halfwidth, halfheight, /*BW=*/1.5f);
  dt_free_align(C1_h_den); dt_free_align(C2_h_den); dt_free_align(C3_h_den);
  C1_h_den = C2_h_den = C3_h_den = NULL;

  /* Phase 10: per-pixel inverse 2x2 WHT + dark_ref + ×unified_sigma + inv GAT */
  {
    static const float SIGNS[4][3] = {
      { +1.0f, +1.0f, +1.0f },  /* R  */
      { -1.0f, +1.0f, -1.0f },  /* Gb */
      { +1.0f, -1.0f, -1.0f },  /* Gr */
      { -1.0f, -1.0f, +1.0f },  /* B  */
    };

    DT_OMP_FOR()
    for(int fr = 0; fr < height; fr++)
    {
      const int r_off = (fr - co_row) & 1;
      for(int fc = 0; fc < width; fc++)
      {
        const int c_off = (fc - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        const size_t pos = (size_t)fr * width + fc;
        const float val = 0.5f * (L_pixel[pos]
                                + SIGNS[slot][0] * C1_aligned[pos]
                                + SIGNS[slot][1] * C2_aligned[pos]
                                + SIGNS[slot][2] * C3_aligned[pos])
                        + ch_dark_ref[slot];
        out[pos] = gat_inverse_exact(val * unified_sigma);
      }
    }
  }

cleanup_rawlc_o:
  dt_free_align(in_gat);
  dt_free_align(L_cs);
  dt_free_align(L_cs_den);
  dt_free_align(L_pixel);
  dt_free_align(L_h_den);
  dt_free_align(L_q); dt_free_align(L_e);
  dt_free_align(C1_h); dt_free_align(C2_h); dt_free_align(C3_h);
  dt_free_align(C1_q); dt_free_align(C2_q); dt_free_align(C3_q);
  dt_free_align(C1_e); dt_free_align(C2_e); dt_free_align(C3_e);
  dt_free_align(C1_loess_h); dt_free_align(C2_loess_h); dt_free_align(C3_loess_h);
  dt_free_align(C1_loess_q); dt_free_align(C2_loess_q); dt_free_align(C3_loess_q);
  dt_free_align(C1_loess_e); dt_free_align(C2_loess_e); dt_free_align(C3_loess_e);
  dt_free_align(C1_q_up); dt_free_align(C2_q_up); dt_free_align(C3_q_up);
  dt_free_align(C1_e_to_q); dt_free_align(C2_e_to_q); dt_free_align(C3_e_to_q);
  dt_free_align(C1_e_up); dt_free_align(C2_e_up); dt_free_align(C3_e_up);
  dt_free_align(C1_h_den); dt_free_align(C2_h_den); dt_free_align(C3_h_den);
  dt_free_align(C1_aligned); dt_free_align(C2_aligned); dt_free_align(C3_aligned);
}


/* --- main() --- */

#pragma GCC diagnostic pop

#endif /* GALOSH_BAYER_H */
