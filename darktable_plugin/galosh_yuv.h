/* galosh_yuv.h - sRGB / YCbCr GALOSH denoise as header-only module.
 *
 * Extracted from GALOSH/standalone/galosh_yuv_cpu.c (minus main()).
 * Include from darktable src/iop/galosh_yuv.c after:
 *   #define GALOSH_DT_ENV
 *   #include "galosh_core.h"
 *
 * Public entry point:
 *   galosh_yuv_denoise_srgb(in, out, W, H, strength_y, strength_c,
 *                            alpha, sigma_sq)
 * Input / output are float32 sRGB, 3 channels interleaved (HxWx3).
 *
 * All helpers are static.  Diagnostic pragma suppresses the spurious
 * -Wunused-function from darktable introspection parser.
 */
#ifndef GALOSH_YUV_H
#define GALOSH_YUV_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

/* ================================================================
 * sRGB ↔ linear + BT.709 YCbCr helpers
 * ================================================================ */

static inline float srgb_to_linear_f(float x)
{
  /* IEC 61966-2-1 piecewise inverse gamma. */
  return (x <= 0.04045f) ? x / 12.92f
                         : powf((x + 0.055f) / 1.055f, 2.4f);
}

static inline float linear_to_srgb_f(float x)
{
  if(x <= 0.0f) return 0.0f;
  if(x >= 1.0f) return 1.0f;
  return (x <= 0.0031308f) ? 12.92f * x
                           : 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
}

/* BT.709 full-range YCbCr (same convention as galosh_yuv_gpu.cl). */
static inline void rgb_to_ycbcr(float R, float G, float B,
                                float *Y, float *Cb, float *Cr)
{
  *Y  =  0.2126f * R + 0.7152f * G + 0.0722f * B;
  *Cb = -0.1146f * R - 0.3854f * G + 0.5000f * B;
  *Cr =  0.5000f * R - 0.4542f * G - 0.0458f * B;
}

static inline void ycbcr_to_rgb(float Y, float Cb, float Cr,
                                float *R, float *G, float *B)
{
  *R = Y                 + 1.5748f * Cr;
  *G = Y - 0.1873f * Cb  - 0.4681f * Cr;
  *B = Y + 1.8556f * Cb;
}

/* ================================================================
 * Blind σ estimation for a single plane (Y) via Laplacian MAD.
 * Identical math to estimate_gat_sigma_mosaic but on a single
 * plane (no Bayer stride).  Used when user passes alpha=0.
 * ================================================================ */
static float estimate_sigma_plane(const float *plane,
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
  /* MAD → std: Var(lap) = 6·sigma² → sigma = MAD/(0.6745·sqrt(6)) */
  return mad / 1.6521f;
}

/* ================================================================
 * galosh_yuv_denoise_srgb — main pipeline.
 * ================================================================ */
static void galosh_yuv_denoise_srgb(const float *restrict in_srgb,
                                    float *restrict out_srgb,
                                    const int width, const int height,
                                    const float strength_y,
                                    const float strength_c,
                                    float alpha, float sigma_sq)
{
  const size_t npx = (size_t)width * (size_t)height;

  /* Step 1: Allocate planar Y, Cb, Cr in GAT / linear domain. */
  float *Y_stab  = dt_alloc_align_float(npx);
  float *Cb_lin  = dt_alloc_align_float(npx);
  float *Cr_lin  = dt_alloc_align_float(npx);
  float *Y_pilot = dt_alloc_align_float(npx);
  float *Y_den   = dt_alloc_align_float(npx);
  float *Cb_den  = dt_alloc_align_float(npx);
  float *Cr_den  = dt_alloc_align_float(npx);
  if(!Y_stab || !Cb_lin || !Cr_lin || !Y_pilot || !Y_den || !Cb_den || !Cr_den)
  {
    fprintf(stderr, "[yuv_cpu] alloc failed\n");
    goto cleanup;
  }

  /* Step 2: sRGB → linear RGB → BT.709 YCbCr (linear domain). */
  DT_OMP_FOR()
  for(int y = 0; y < height; y++)
  {
    for(int x = 0; x < width; x++)
    {
      const size_t i = (size_t)y * width + x;
      const float R = srgb_to_linear_f(in_srgb[3 * i + 0]);
      const float G = srgb_to_linear_f(in_srgb[3 * i + 1]);
      const float B = srgb_to_linear_f(in_srgb[3 * i + 2]);
      float Y, Cb, Cr;
      rgb_to_ycbcr(R, G, B, &Y, &Cb, &Cr);
      Y_stab[i] = Y;    /* temp: holds linear Y, will be overwritten by GAT */
      Cb_lin[i] = Cb;
      Cr_lin[i] = Cr;
    }
  }

  /* Step 3: Blind noise estimate on Y (if user didn't supply α, σ²).
   *
   * We need α > 0 for the GAT to be well-defined (t_break = 2σ/α).  When
   * the user passes α≤0 we treat σ² as signal-INdependent variance and
   * synthesise a tiny α via the sqrt(σ²) heuristic: α = σ_lin · 0.1 is
   * enough to make GAT behave as approximately-linear normalisation
   * (matches "quasi-Gaussian" regime for low-ISO captures). */
  if(alpha <= 0.0f || sigma_sq <= 0.0f)
  {
    const float sigma_lin = estimate_sigma_plane(Y_stab, width, height);
    sigma_sq = fmaxf(sigma_lin * sigma_lin, 1e-8f);
    alpha    = fmaxf(sigma_lin * 0.1f, 1e-5f);  /* tiny α — quasi-linear GAT */
    fprintf(stderr, "[yuv_cpu] Blind σ (MAD) = %.5f, using α=%.6g σ²=%.6e\n",
            sigma_lin, alpha, sigma_sq);
  }

  /* Step 4: GAT forward on Y.  α=0 degenerates to (Y - 0) / σ — pure
   * linear normalisation.  With α>0, the full variance-stabilising
   * transform is applied. */
  gat_build_inverse_table(alpha, sigma_sq);
  DT_OMP_FOR()
  for(size_t i = 0; i < npx; i++)
    Y_stab[i] = gat_forward(Y_stab[i], alpha, sigma_sq);

  /* Step 5: Normalise Y_stab to unit variance in GAT space.  Measure
   * σ_gat on the post-GAT plane and divide.  This matches the RAW path
   * convention so downstream strength values are scale-consistent. */
  const float sigma_gat = estimate_sigma_plane(Y_stab, width, height);
  const float unified_sigma = fmaxf(sigma_gat, 1e-6f);
  const float inv_sg = 1.0f / unified_sigma;
  DT_OMP_FOR()
  for(size_t i = 0; i < npx; i++) Y_stab[i] *= inv_sg;

  fprintf(stderr, "[yuv_cpu] alpha=%.6g sigma_sq=%.6g  sigma_gat=%.4f  "
                  "size=%dx%d  strength_y=%.3f strength_c=%.3f\n",
          alpha, sigma_sq, sigma_gat,
          width, height, strength_y, strength_c);

  /* Step 6: LOSH on Y (pass1 pilot + pass2 Wiener, 75% overlap stride=2).
   * GALOSH_YUV_G uses MAD-based sigma_Y unconditionally (use_robust_shrink=1)
   * — the partial-selection-sort robust noise estimator that kills spatial
   * noise clusters fooling the L2 sum_sq estimator (Donoho-Johnstone 1995).
   * Single-orientation path; n_orient=1 collapses to the legacy pass1+pass2
   * sequence inside the wrapper. */
  const int losh_stride = 2;
  galosh_pass12_multiorient_blocked(Y_stab, Y_den, width, height,
                                     strength_y, GALOSH_BLOCK_SIZE,
                                     losh_stride, /*n_orient=*/1,
                                     /*use_robust_shrink=*/1);
  (void)Y_pilot;  /* internal pilot now lives inside the wrapper */
  fprintf(stderr, "[GALOSH_YUV_G] Y LOSH pass1(MAD)+pass2 done\n");

  /* Step 7: LOESS on Cb, Cr with noisy Y as guide (matches GPU raw path;
   * using the denoised Y gives slightly smoother chroma but also
   * double-filters — the noisy Y already contains enough high-SNR edge
   * information for bilateral weighting). */
  galosh_loess_chroma(Y_stab, Cb_lin, Cr_lin, Cb_den, Cr_den,
                      width, height, strength_c);
  fprintf(stderr, "[yuv_cpu] Cb/Cr LOESS done (R=%d BW=%.1f)\n",
          GALOSH_LOESS_RADIUS, GALOSH_LOESS_BW);

  /* Step 8: Inverse GAT on Y and rebuild sRGB. */
  DT_OMP_FOR()
  for(int y = 0; y < height; y++)
  {
    for(int x = 0; x < width; x++)
    {
      const size_t i = (size_t)y * width + x;
      const float Y_inv = gat_inverse_exact(Y_den[i] * unified_sigma);
      float R, G, B;
      ycbcr_to_rgb(Y_inv, Cb_den[i], Cr_den[i], &R, &G, &B);
      out_srgb[3 * i + 0] = linear_to_srgb_f(R);
      out_srgb[3 * i + 1] = linear_to_srgb_f(G);
      out_srgb[3 * i + 2] = linear_to_srgb_f(B);
    }
  }

cleanup:
  dt_free_align(Y_stab);  dt_free_align(Y_pilot); dt_free_align(Y_den);
  dt_free_align(Cb_lin);  dt_free_align(Cr_lin);
  dt_free_align(Cb_den);  dt_free_align(Cr_den);
}

/* ================================================================
 * main: CLI driver (same binary I/O style as galosh_raw_cpu).
 * ================================================================ */

#pragma GCC diagnostic pop

#endif /* GALOSH_YUV_H */
