/* galosh_yuv.h - GALOSH YUV denoiser, linear RGB entry (header-only).
 *
 * This header is a darktable-specific re-rolling of standalone
 * galosh_yuv_cpu.c.  Differences from the CLI reference:
 *
 *   1. Input/output are LINEAR scene-referred RGB (the convention of
 *      darktable's IOP_CS_RGB at this pipeline stage), NOT gamma-encoded
 *      sRGB.  The sRGB <-> linear conversions of the CLI reference are
 *      removed -- applying inverse gamma to already-linear data was the
 *      cause of magenta highlights and global colour-cast.
 *
 *   2. alpha / sigma_sq are no longer accepted -- the entry is fully
 *      blind (Foi-Alenius MAD on Y, quasi-linear GAT).  The IOP UI
 *      exposes only luma_strength and chroma_strength as designed.
 *
 *   3. Debug fprintf() spam removed -- darktable rebuilds the pipeline
 *      tile on every redraw, the CLI logging would flood stderr.
 *
 * The algorithm itself (BT.709 Y/Cb/Cr split, GAT-stabilised Y with
 * 2-pass WHT-LOSH, Y-guided LOESS chroma) is unchanged from the CLI
 * reference.
 *
 * Public entry point:
 *   galosh_yuv_denoise_linear(in_rgb, out_rgb, W, H,
 *                              strength_y, strength_c)
 *
 *  Refs: Foi et al. (Sig.Proc. 2008) - Poisson-Gaussian noise model
 *        Makitalo & Foi (TIP 2013)   - exact unbiased inverse GAT
 *        Chang, Yu & Vetterli (TIP 2000) - BayesShrink
 *        Cleveland (JASA 1979)       - LOESS local regression
 */
#ifndef GALOSH_YUV_H
#define GALOSH_YUV_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

/* ================================================================
 * BT.709 full-range YCbCr (linear domain)
 *
 * Same coefficients as the GPU galosh_yuv kernel and the standalone
 * CLI reference -- only the input/output here are linear, not sRGB.
 * ================================================================ */
static inline void galosh_yuv_rgb_to_ycbcr(float R, float G, float B,
                                           float *Y, float *Cb, float *Cr)
{
  *Y  =  0.2126f * R + 0.7152f * G + 0.0722f * B;
  *Cb = -0.1146f * R - 0.3854f * G + 0.5000f * B;
  *Cr =  0.5000f * R - 0.4542f * G - 0.0458f * B;
}

static inline void galosh_yuv_ycbcr_to_rgb(float Y, float Cb, float Cr,
                                           float *R, float *G, float *B)
{
  *R = Y                 + 1.5748f * Cr;
  *G = Y - 0.1873f * Cb  - 0.4681f * Cr;
  *B = Y + 1.8556f * Cb;
}

/* ================================================================
 * Blind sigma estimation on a single plane (Y) via Laplacian MAD.
 *
 * Mirrors estimate_gat_sigma_mosaic but on a contiguous plane (no
 * Bayer stride).  Median-of-absolute-Laplacians scaled by
 * 1/(0.6745 * sqrt(6)) = 1/1.6521.
 * ================================================================ */
static float galosh_yuv_estimate_sigma(const float *plane,
                                       const int width, const int height)
{
  const int n_samples = MIN(width * height / 6, 200000);
  float *abs_laps = dt_alloc_align_float(n_samples + 1);
  if(!abs_laps) return 1.0f;

  int count = 0;
  for(int y = 0; y < height && count < n_samples; y++)
  {
    const float *row = plane + (size_t)y * width;
    for(int x = 0; x < width - 4 && count < n_samples; x += 3)
    {
      const float lap = row[x] - 2.0f * row[x + 2] + row[x + 4];
      abs_laps[count++] = fabsf(lap);
    }
  }
  if(count < 100) { dt_free_align(abs_laps); return 1.0f; }
  const float mad = quick_select_median(abs_laps, count);
  dt_free_align(abs_laps);
  return mad / 1.6521f;
}

/* ================================================================
 * galosh_yuv_denoise_linear -- main pipeline (linear RGB in/out).
 *
 * The pipeline matches the CLI reference exactly except for the two
 * skipped gamma conversions and the always-blind noise estimation:
 *
 *   in_rgb (linear)
 *     -> BT.709 Y, Cb, Cr (linear)
 *     -> blind MAD sigma on Y -> alpha = 0.1*sigma, sigma_sq = sigma^2
 *     -> GAT forward on Y, normalise to unit variance in GAT space
 *     -> LOSH 2-pass (BayesShrink pilot + Wiener) on Y_stab
 *     -> LOESS Y-guided regression on Cb, Cr
 *     -> inverse GAT on Y -> ycbcr_to_rgb -> out_rgb (linear)
 * ================================================================ */
static void galosh_yuv_denoise_linear(const float *restrict in_rgb,
                                      float *restrict out_rgb,
                                      const int width, const int height,
                                      const float strength_y,
                                      const float strength_c)
{
  const size_t npx = (size_t)width * (size_t)height;

  float *Y_stab  = dt_alloc_align_float(npx);
  float *Cb_lin  = dt_alloc_align_float(npx);
  float *Cr_lin  = dt_alloc_align_float(npx);
  float *Y_pilot = dt_alloc_align_float(npx);
  float *Y_den   = dt_alloc_align_float(npx);
  float *Cb_den  = dt_alloc_align_float(npx);
  float *Cr_den  = dt_alloc_align_float(npx);
  if(!Y_stab || !Cb_lin || !Cr_lin || !Y_pilot || !Y_den || !Cb_den || !Cr_den)
  {
    /* On alloc failure pass-through and bail. */
    if(in_rgb != out_rgb)
      memcpy(out_rgb, in_rgb, npx * 3 * sizeof(float));
    goto cleanup;
  }

  /* Step 1: linear RGB -> BT.709 YCbCr (linear). */
  DT_OMP_FOR()
  for(int y = 0; y < height; y++)
  {
    for(int x = 0; x < width; x++)
    {
      const size_t i = (size_t)y * width + x;
      const float R = in_rgb[3 * i + 0];
      const float G = in_rgb[3 * i + 1];
      const float B = in_rgb[3 * i + 2];
      float Y, Cb, Cr;
      galosh_yuv_rgb_to_ycbcr(R, G, B, &Y, &Cb, &Cr);
      Y_stab[i] = Y;
      Cb_lin[i] = Cb;
      Cr_lin[i] = Cr;
    }
  }

  /* Step 2: blind noise estimation on Y. */
  const float sigma_lin = galosh_yuv_estimate_sigma(Y_stab, width, height);
  const float sigma_sq  = fmaxf(sigma_lin * sigma_lin, 1e-8f);
  const float alpha     = fmaxf(sigma_lin * 0.1f, 1e-5f);

  /* Step 3: GAT forward on Y. */
  gat_build_inverse_table(alpha, sigma_sq);
  DT_OMP_FOR()
  for(size_t i = 0; i < npx; i++)
    Y_stab[i] = gat_forward(Y_stab[i], alpha, sigma_sq);

  /* Step 4: normalise Y_stab to unit variance in GAT space. */
  const float sigma_gat = galosh_yuv_estimate_sigma(Y_stab, width, height);
  const float unified_sigma = fmaxf(sigma_gat, 1e-6f);
  const float inv_sg = 1.0f / unified_sigma;
  DT_OMP_FOR()
  for(size_t i = 0; i < npx; i++) Y_stab[i] *= inv_sg;

  /* Step 5: 2-pass LOSH on Y (Pass1 BayesShrink + Pass2 Wiener, stride=2).
   * GALOSH_YUV_G: MAD-based sigma_Y in Pass1 (use_robust_shrink=1) -- the
   * partial-selection-sort robust noise estimator that kills spatial noise
   * clusters fooling the L2 sum_sq estimator (Donoho-Johnstone 1995).
   * SIDD Medium 80-pair: +0.84 dB PSNR / -19% LPIPS / -7.9% DISTS vs the
   * legacy mean-based pass1.  Single-orientation collapses to plain
   * pass1+pass2 inside the wrapper. */
  const int losh_stride = 2;
  galosh_pass12_multiorient_blocked(Y_stab, Y_den, width, height,
                                     strength_y, GALOSH_BLOCK_SIZE,
                                     losh_stride, /*n_orient=*/1,
                                     /*use_robust_shrink=*/1);
  (void)Y_pilot;  /* internal pilot now lives inside the wrapper */

  /* Step 6: LOESS Y-guided chroma. */
  galosh_loess_chroma(Y_stab, Cb_lin, Cr_lin, Cb_den, Cr_den,
                      width, height, strength_c);

  /* Step 7: inverse GAT on Y, rebuild linear RGB. */
  DT_OMP_FOR()
  for(int y = 0; y < height; y++)
  {
    for(int x = 0; x < width; x++)
    {
      const size_t i = (size_t)y * width + x;
      const float Y_inv = gat_inverse_exact(Y_den[i] * unified_sigma);
      float R, G, B;
      galosh_yuv_ycbcr_to_rgb(Y_inv, Cb_den[i], Cr_den[i], &R, &G, &B);
      out_rgb[3 * i + 0] = R;
      out_rgb[3 * i + 1] = G;
      out_rgb[3 * i + 2] = B;
    }
  }

cleanup:
  dt_free_align(Y_stab);  dt_free_align(Y_pilot); dt_free_align(Y_den);
  dt_free_align(Cb_lin);  dt_free_align(Cr_lin);
  dt_free_align(Cb_den);  dt_free_align(Cr_den);
}

#pragma GCC diagnostic pop

#endif /* GALOSH_YUV_H */
