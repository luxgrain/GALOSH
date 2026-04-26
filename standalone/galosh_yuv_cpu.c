/* galosh_yuv_cpu.c — sRGB / YCbCr GALOSH denoiser (CPU reference).
 *
 * Mirrors the GPU `galosh_yuv_gpu` pipeline (linear Y-GAT + Y-guided
 * chroma VST) on CPU as a paper-reference / plugin template.  Uses the
 * same `galosh_core.h` algorithm primitives as the RAW driver.
 *
 * Usage: galosh_yuv_cpu in.bin out.bin width height
 *          [strength_y] [strength_c] [alpha] [sigma_sq]
 * Input:  float32 sRGB, 3 channels, HxWx3 row-major
 * Output: float32 sRGB, same layout
 * Build:  gcc -O3 -march=native -ffast-math -funroll-loops -fopenmp
 *             -DGALOSH_F -o galosh_yuv_cpu galosh_yuv_cpu.c -lm
 *
 * Pipeline:
 *   sRGB → linear RGB (inverse gamma)
 *       → BT.709 Y / Cb / Cr
 *       → Y: GAT forward → LOSH pass1+pass2 → Makitalo inverse = Y_den
 *       → Cb, Cr: LOESS guided-chroma (Y_den as guide)
 *       → linear RGB (BT.709 inverse) → sRGB (gamma)
 *
 *  Refs: Foi et al. (Sig.Proc. 2008) – Poisson-Gaussian noise model
 *        Makitalo & Foi (TIP 2013)   – exact unbiased inverse GAT
 *        Chang, Yu & Vetterli (TIP 2000) – BayesShrink
 *        Cleveland (JASA 1979)       – LOESS local regression
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "galosh_cpu.h"

/* Diagonal-quality bench variant flags (set from CLI in main()).
 *   g_yuv_stride    : 1 or 2 (variant A: stride=1)
 *   g_yuv_n_orient  : 1 or 4 (variant B: 4-orient averaging)
 * Variant C (EWA-JL3 compute_L_fullres) is RAW-specific so YUV
 * pipeline doesn't expose --lfr-kernel. */
static int g_yuv_stride   = 2;
static int g_yuv_n_orient = 1;

/* GALOSH_YUV_G adopts MAD-based BayesShrink (Pass1 sigma_Y estimator)
 * unconditionally.  sigma_Y = median(|AC coef|)/0.6745, robust to ~25%
 * outlier coefficients (Donoho-Johnstone 1995).  Kills the spatial noise
 * clusters that the L2 sum_sq estimator over-includes as "false signal"
 * (the BM3D-CFA-clearable residual).  Applied to Y plane Pass1; Cb/Cr
 * LOESS path is unaffected (LOESS is a separate guided-regression
 * estimator).
 *
 * Earlier ad-hoc name "v5 robust-MAD" referred to the development variant
 * that introduced this on YUV; the final pipeline subsumes it.  No
 * runtime flag exposes it -- it is part of GALOSH_YUV_G's definition. */

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

  /* Step 6: LOSH on Y (Pass1 BayesShrink pilot + Pass2 empirical Wiener).
   * block_size = 8.  GALOSH_YUV_G uses MAD-based sigma_Y unconditionally
   * (use_robust_shrink hardcoded to 1).  Stride / orientation remain
   * bench-overrideable for archived variants A/B. */
  const int losh_stride = g_yuv_stride;
  galosh_pass12_multiorient_blocked(Y_stab, Y_den, width, height,
                                     strength_y, GALOSH_BLOCK_SIZE,
                                     losh_stride, g_yuv_n_orient,
                                     /*use_robust_shrink=*/1);
  (void)Y_pilot;  /* pilot now lives inside the multiorient wrapper */
  fprintf(stderr, "[GALOSH_YUV_G] Y LOSH Pass1(MAD)+Pass2 done\n");

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
int main(int argc, char **argv)
{
  /* Strip --stride / --orient flags before positional parsing. */
  int new_argc = 0;
  char *positional[32];
  for(int i = 0; i < argc; i++)
  {
    const char *a = argv[i];
    if(strncmp(a, "--stride=", 9) == 0)
    {
      g_yuv_stride = atoi(a + 9);
      if(g_yuv_stride < 1 || g_yuv_stride > 8) g_yuv_stride = 2;
    }
    else if(strncmp(a, "--orient=", 9) == 0)
    {
      g_yuv_n_orient = atoi(a + 9);
      if(g_yuv_n_orient != 1 && g_yuv_n_orient != 4) g_yuv_n_orient = 1;
    }
    else if(strncmp(a, "--robust-shrink=", 16) == 0)
    {
      /* Deprecated -- MAD-based BayesShrink is now part of GALOSH_YUV_G's
       * definition.  Silently accepted for bench-script back-compat. */
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
            "Usage: %s in.bin out.bin width height "
            "[strength_y=1.0] [strength_c=1.0] [alpha=0] [sigma_sq=0]\n"
            "       [--stride=1|2] [--orient=1|4]\n",
            argv[0]);
    return 1;
  }

  fprintf(stderr, "[GALOSH_YUV_G] stride=%d orient=%d\n",
          g_yuv_stride, g_yuv_n_orient);

  const char *in_path  = argv[1];
  const char *out_path = argv[2];
  const int width  = atoi(argv[3]);
  const int height = atoi(argv[4]);
  const float strength_y = (argc > 5) ? (float)atof(argv[5]) : 1.0f;
  const float strength_c = (argc > 6) ? (float)atof(argv[6]) : 1.0f;
  const float alpha_cli  = (argc > 7) ? (float)atof(argv[7]) : 0.0f;
  const float sigma_cli  = (argc > 8) ? (float)atof(argv[8]) : 0.0f;

  init_galosh_kaiser();

  const size_t npx = (size_t)width * (size_t)height;
  const size_t nbytes = npx * 3 * sizeof(float);

  float *in_buf  = dt_alloc_align_float(npx * 3);
  float *out_buf = dt_alloc_align_float(npx * 3);
  if(!in_buf || !out_buf) { fprintf(stderr, "alloc failed\n"); return 1; }

  FILE *fi = fopen(in_path, "rb");
  if(!fi) { fprintf(stderr, "cannot open %s\n", in_path); return 1; }
  const size_t rd = fread(in_buf, 1, nbytes, fi);
  fclose(fi);
  if(rd != nbytes)
  {
    fprintf(stderr, "short read (%zu of %zu)\n", rd, nbytes);
    return 1;
  }

  const clock_t t0 = clock();
  galosh_yuv_denoise_srgb(in_buf, out_buf, width, height,
                          strength_y, strength_c, alpha_cli, sigma_cli);
  const double t = (double)(clock() - t0) / CLOCKS_PER_SEC;
  fprintf(stderr, "[yuv_cpu] total time: %.2fs\n", t);

  FILE *fo = fopen(out_path, "wb");
  if(!fo) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }
  fwrite(out_buf, 1, nbytes, fo);
  fclose(fo);

  dt_free_align(in_buf);
  dt_free_align(out_buf);
  return 0;
}
