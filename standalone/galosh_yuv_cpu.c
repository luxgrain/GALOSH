/* galosh_yuv_cpu.c — sRGB / YCbCr GALOSH denoiser (CPU reference).
 *
 * ##############################################################
 * # [LATEST] GALOSH_YUV_G  (commit a48e716, default)
 * ##############################################################
 *   Production canonical pipeline.  Mirrors the GPU `galosh_yuv_gpu`
 *   driver and uses the same `galosh_cpu.h` algorithm primitives as
 *   the RAW driver.  No compile flag needed (LATEST is default).
 *
 *   GALOSH_YUV_G full flow:
 *     Phase 0  sRGB → linear RGB (inverse gamma)
 *     Phase 1  Linear RGB → BT.709 Y / Cb / Cr  (linear domain)
 *     Phase 2  Y plane:
 *              ① Foi-Alenius blind α / σ² (estimate_sigma_plane MAD,
 *                 if alpha/sigma_sq <= 0)
 *              ② GAT forward on Y  (gat_forward)
 *              ③ unit-variance normalise Y_stab via σ_GAT MAD
 *              ④ LOSH Pass1+Pass2 with use_robust_shrink=1
 *                 (= MAD-based BayesShrink + Wiener; same K13-style as
 *                  RAW Phase 4(a)).
 *              ⑤ Makitalo exact-unbiased inverse GAT → Y_den
 *     Phase 3  Cb / Cr planes:
 *              galosh_loess_chroma (Y-guided bilateral LOESS, R=7,
 *              BW=3) using NOISY Y as guide → Cb_den, Cr_den
 *     Phase 4  YCbCr → linear RGB (BT.709 inverse) → sRGB (gamma)
 *
 *   Two structural features adopted unconditionally vs pre-G:
 *     (i)  Pass1 σ_Y via MAD partial-selection-sort robust estimator
 *          (= "v5 robust-MAD" in old ad-hoc naming)
 *     (ii) Cb/Cr denoise via Y-guided BILATERAL LOESS (replaces the
 *          pre-G separable mean-only guided filter — bilateral weight
 *          excludes specular highlights from the regression).
 *
 * ##############################################################
 * # [ARCHIVED] pre-GALOSH_YUV_G variants
 * ##############################################################
 *   No #ifdef-guarded legacy path in this file (unlike galosh_raw_cpu.c).
 *   Old YUV variants lived only in archived bench scripts:
 *     - separable mean guided filter (Box approximation):
 *       moments_x → moments_y_ab → apply_x → apply_y, all 4-pass with
 *       no bilateral weight.  The pre-G GPU YUV used this; CPU never had
 *       a compile-flag fallback for it.  Now replaced by single
 *       galosh_loess_chroma call (= LATEST).
 *     - L2 sum_sq σ_Y for Y-plane Pass1 (= use_robust_shrink=0).
 *       Replaced by MAD via use_robust_shrink=1 — the only call site
 *       below uses 1 unconditionally; no bench-only fallback path.
 *
 * ##############################################################
 *
 * Usage: galosh_yuv_cpu in.bin out.bin width height
 *          [strength_y] [strength_c] [alpha] [sigma_sq]
 * Input:  float32 sRGB, 3 channels, HxWx3 row-major
 * Output: float32 sRGB, same layout
 *
 * Build (LATEST / production = GALOSH_YUV_G, default):
 *   gcc -O3 -march=native -ffast-math -funroll-loops -fopenmp \
 *       -o galosh_yuv_cpu galosh_yuv_cpu.c -lm
 *   (-DGALOSH_F is accepted as a no-op alias for backwards compat
 *    with old bench scripts.)
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
#include "galosh_yuv420.h"   /* GALOSH-420 planar front-end (spec:
                              * docs/yuv420_frontend_spec.md) — pure
                              * container handling, no denoise math. */

/* ============================================================
 * [MIXED] Pipeline tuning flags (set from CLI in main()).
 *   g_yuv_stride    : [LATEST] 2 = GALOSH_YUV_G default
 *                     [ARCHIVED] 1 = full cycle-spinning (variant A,
 *                                    no measurable quality gain)
 *   g_yuv_n_orient  : [LATEST] 1 = GALOSH_YUV_G default
 *                     [ARCHIVED] 4 = 4-orient averaging (variant B,
 *                                    no quality gain)
 * Variant C (EWA-JL3 compute_L_fullres) is RAW-specific so YUV
 * pipeline doesn't expose --lfr-kernel.
 *
 * Defaults below produce [LATEST] GALOSH_YUV_G canonical pipeline.
 * ============================================================ */
static int g_yuv_stride   = 2;   /* [LATEST] */
static int g_yuv_n_orient = 1;   /* [LATEST] */

/* [GALOSH-420 2026-07-12] Half-res 4:2:0 chroma flag.  When set (by the
 * 420 driver ONLY, and only for pix==420), the linear-domain chroma LOESS
 * uses a noise-adaptive radius (galosh_yuv420_chroma_radius) instead of the
 * full-res GALOSH_LOESS_RADIUS.  0 = legacy/full-res behaviour (unchanged
 * for 4:2:2 / 4:4:4 / the legacy sRGB path). */
static int g_yuv420_halfres_chroma = 0;

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
 * Pipeline-variant selector flags.
 *
 * g_galosh_yuv_use_q (= [LATEST: GALOSH_YUV_Q]):
 *   0 → [LATEST: GALOSH_YUV_O] = production = single-scale full-res
 *       Cb/Cr LOESS R=GALOSH_LOESS_RADIUS, per-block BayesShrink λ.
 *   1 → [LATEST: GALOSH_YUV_Q] = production candidate = 3-anchor multi-
 *       scale chroma LOESS pyramid {full / half-up / quarter-up} with
 *       smoothstep slider walk and Y_stab-guided EWA Jinc-Lanczos-3
 *       joint bilateral upsample (= mirror of RAW O Phase 7-9).
 *       Luma path identical to _O (= no spatially-smoothed BayesShrink).
 *
 * g_galosh_yuv_use_p (= [DEPRECATED: GALOSH_YUV_P]):
 *   1 → [DEPRECATED: GALOSH_YUV_P] = failed try = same chroma pyramid
 *       as _Q PLUS spatially-smoothed BayesShrink lambda on luma
 *       (= sets g_galosh_lambda_smooth=1).  Retained for archival
 *       reproducibility; SIDD 80-pair bench 2026-05-08 showed -0.46 dB
 *       PSNR regression vs _O traced to the luma sigma_y_sq smoothing
 *       path (= reduces BayesShrink local adaptivity).
 *
 * Mutually exclusive at orchestrator entry.  Process-wide single-writer.
 * ================================================================ */
static int g_galosh_yuv_use_q = 0;  /* [LATEST: GALOSH_YUV_Q] */
static int g_galosh_yuv_use_p = 0;  /* [DEPRECATED: GALOSH_YUV_P] */

/* Shared LOSH Pass1-mode global: declared `extern` in galosh_cpu.h and used by the
 * shared luma-shrink code, so every standalone driver must provide one definition.
 * YUV keeps the canonical default (0 = vanilla BayesShrink; A1 is a RAW-only CLI opt).
 * (Was missing -> YUV link broke after galosh_cpu.h added the extern; fixed Ph3.) */
int g_galosh_pass1_mode = 0;

/* ================================================================
 * [LATEST: GALOSH_YUV_P] galosh_yuv_chroma_pyramid_p — 3-anchor
 * multi-scale LOESS chroma pyramid with smoothstep slider walk.
 *
 * Mirrors RAW O Phase 7-9: full-res LOESS (anchor 0) + box-down 2x +
 * LOESS at half-res + K16 EWA-JL3 up (anchor 1) + box-down 4x + LOESS
 * at quarter-res + K16 up to half + K16 up to full (anchor 2).
 *
 * Smoothstep blend by chroma_strength ∈ [0, 2]:
 *   strength ∈ [0, 1]: blend(C_full, C_h_up)
 *   strength ∈ [1, 2]: blend(C_h_up, C_q_up)
 *
 * y_guide = Y_stab (= post-GAT, post-unified_sigma, pre-LOSH luma; = the
 * "noisy Y" used for bilateral edge alignment, matching CPU O comment
 * "use NOISY Y as guide; denoised would double-filter").
 * ================================================================ */
static inline float galosh_smoothstep_p(float t)
{
  if(t <= 0.0f) return 0.0f;
  if(t >= 1.0f) return 1.0f;
  return t * t * (3.0f - 2.0f * t);
}

static void galosh_yuv_chroma_pyramid_p(const float *restrict y_guide,
                                        const float *restrict cb_in,
                                        const float *restrict cr_in,
                                        float *restrict cb_out,
                                        float *restrict cr_out,
                                        const int width, const int height,
                                        const float chroma_strength)
{
  const size_t npx = (size_t)width * (size_t)height;
  /* gat_box_downsample_2x convention: dw = sw/2 (floor).  Match that
   * for buffer alloc to avoid off-by-one access violations. */
  const int hw = width  / 2;
  const int hh = height / 2;
  const int qw = hw / 2;
  const int qh = hh / 2;
  const size_t hsize = (size_t)hw * hh;
  const size_t qsize = (size_t)qw * qh;
  /* K16 stride: K16 outputs at (2*half_w, 2*half_h).  When this differs
   * from the next pyramid level's native dim (= due to integer / 2
   * floor truncation), we crop guide / pad output via the standard
   * GALOSH_RAW_O round-trip helpers. */
  const int kfw_half = 2 * hw;   /* K16 q→h output width (= 2*qw) */
  const int kfh_half = 2 * hh;
  const int kfw_q   = 2 * qw;
  const int kfh_q   = 2 * qh;
  const size_t kf_q_size = (size_t)kfw_q * kfh_q;

  float *Cb_full = dt_alloc_align_float(npx);
  float *Cr_full = dt_alloc_align_float(npx);
  /* half-res buffers */
  float *Y_h    = dt_alloc_align_float(hsize);
  float *Cb_h   = dt_alloc_align_float(hsize);
  float *Cr_h   = dt_alloc_align_float(hsize);
  float *Cb_h_d = dt_alloc_align_float(hsize);
  float *Cr_h_d = dt_alloc_align_float(hsize);
  float *Cb_h_up = dt_alloc_align_float(npx);
  float *Cr_h_up = dt_alloc_align_float(npx);
  /* quarter-res buffers */
  float *Y_q    = dt_alloc_align_float(qsize);
  float *Cb_q   = dt_alloc_align_float(qsize);
  float *Cr_q   = dt_alloc_align_float(qsize);
  float *Cb_q_d = dt_alloc_align_float(qsize);
  float *Cr_q_d = dt_alloc_align_float(qsize);
  /* K16 q→h: raw output at kf_q_size (= 2*qw × 2*qh). */
  float *Y_for_q       = dt_alloc_align_float(kf_q_size);
  float *Cb_q_to_h_raw = dt_alloc_align_float(kf_q_size);
  float *Cr_q_to_h_raw = dt_alloc_align_float(kf_q_size);
  float *zero_kfq      = dt_alloc_align_float(kf_q_size);
  /* Padded q→h output at hw × hh for K16 h→full. */
  float *Cb_q_to_h = dt_alloc_align_float(hsize);
  float *Cr_q_to_h = dt_alloc_align_float(hsize);
  /* K16 h→full: raw output at (kfw_half × kfh_half) = (2hw, 2hh), pad to npx. */
  float *Cb_q_up_raw = dt_alloc_align_float((size_t)kfw_half * kfh_half);
  float *Cr_q_up_raw = dt_alloc_align_float((size_t)kfw_half * kfh_half);
  float *Cb_h_up_raw = dt_alloc_align_float((size_t)kfw_half * kfh_half);
  float *Cr_h_up_raw = dt_alloc_align_float((size_t)kfw_half * kfh_half);
  float *Y_for_h     = dt_alloc_align_float((size_t)kfw_half * kfh_half);
  float *zero_kfh    = dt_alloc_align_float((size_t)kfw_half * kfh_half);
  /* Dummy zero buffers (= K16 helper is 3-channel; YUV uses 2). */
  float *zero_h    = dt_alloc_align_float(hsize);
  float *zero_q    = dt_alloc_align_float(qsize);
  float *zero_full = dt_alloc_align_float(npx);

  if(!Cb_full || !Cr_full || !Y_h || !Cb_h || !Cr_h || !Cb_h_d || !Cr_h_d
     || !Cb_h_up || !Cr_h_up || !Y_q || !Cb_q || !Cr_q || !Cb_q_d || !Cr_q_d
     || !Y_for_q || !Cb_q_to_h_raw || !Cr_q_to_h_raw || !zero_kfq
     || !Cb_q_to_h || !Cr_q_to_h
     || !Cb_q_up_raw || !Cr_q_up_raw || !Cb_h_up_raw || !Cr_h_up_raw
     || !Y_for_h || !zero_kfh
     || !zero_h || !zero_q || !zero_full)
  {
    fprintf(stderr, "[yuv_p] chroma_pyramid alloc failed\n");
    goto cleanup_pyr;
  }
  /* Zero buffers (= 3-channel K16 helper requires dummy 3rd input/output;
   * 2-channel YUV path passes zeros).  memset is fast — keep sequential. */
  memset(zero_h,    0, hsize * sizeof(float));
  memset(zero_q,    0, qsize * sizeof(float));
  memset(zero_full, 0, npx   * sizeof(float));
  memset(zero_kfq,  0, kf_q_size * sizeof(float));
  memset(zero_kfh,  0, (size_t)kfw_half * kfh_half * sizeof(float));

  /* ============================================================
   * Optimization — lazy anchor compute based on chroma_strength.
   *
   * EN: The smoothstep slider walk reads only the anchor pair adjacent
   *     to the current slider segment:
   *       cs ∈ [0, 1]: blend(noisy, C_full)            → need C_full
   *       cs ∈ [1, 2]: blend(C_full, C_h_up)           → need C_full + C_h_up
   *       cs ∈ [2, 3]: blend(C_h_up, C_q_up)           → need C_h_up + C_q_up
   *     Anchors outside the active pair are unused, so we skip their
   *     compute entirely.  At the calibrated default (cs = 1.0) only
   *     C_full is needed → pyramid Anchor 1 + Anchor 2 are skipped,
   *     matching GALOSH_YUV_O wall-time exactly.
   *
   * JP: smoothstep walk は隣接 anchor pair しか読まないので、 不要な
   *     anchor の compute をスキップ。 cs=1.0 default では Anchor 0
   *     のみ計算 → _O と同速度。
   * ============================================================ */
  const float cs_clamped = (chroma_strength < 0.0f) ? 0.0f
                         : (chroma_strength > 3.0f) ? 3.0f : chroma_strength;
  const int need_full = (cs_clamped <= 2.0f);  /* segments 1, 2 */
  const int need_h_up = (cs_clamped >  1.0f);  /* segments 2, 3 */
  const int need_q_up = (cs_clamped >  2.0f);  /* segment 3 */

  /* --- Anchor 0: full-res LOESS (= calibrated default chroma). --- */
  if(need_full)
  {
    galosh_loess_chroma(y_guide, cb_in, cr_in, Cb_full, Cr_full,
                        width, height, /*strength_c=*/1.0f);
  }

  /* --- Anchor 1: half-res LOESS + K16 up half→full. --- */
  if(need_h_up)
  {
    gat_box_downsample_2x(y_guide,  Y_h,  width, height);
    gat_box_downsample_2x(cb_in,    Cb_h, width, height);
    gat_box_downsample_2x(cr_in,    Cr_h, width, height);
    galosh_loess_chroma(Y_h, Cb_h, Cr_h, Cb_h_d, Cr_h_d,
                        hw, hh, /*strength_c=*/1.0f);
    gat_crop_2d_topleft(y_guide, width, height, Y_for_h, kfw_half, kfh_half);
    gat_k16_joint_bilateral_upsample(Cb_h_d, Cr_h_d, zero_h,
                                     Y_for_h,
                                     Cb_h_up_raw, Cr_h_up_raw, zero_kfh,
                                     hw, hh, /*BW=*/3.0f);
    gat_pad_2d_edge(Cb_h_up_raw, kfw_half, kfh_half, Cb_h_up, width, height);
    gat_pad_2d_edge(Cr_h_up_raw, kfw_half, kfh_half, Cr_h_up, width, height);
  }

  /* --- Anchor 2: quarter-res LOESS + K16 q→h + K16 h→full.
   * Reads Y_h / Cb_h / Cr_h written by Anchor 1 (= guarded jointly). --- */
  if(need_q_up)
  {
    gat_box_downsample_2x(Y_h,  Y_q,  hw, hh);
    gat_box_downsample_2x(Cb_h, Cb_q, hw, hh);
    gat_box_downsample_2x(Cr_h, Cr_q, hw, hh);
    galosh_loess_chroma(Y_q, Cb_q, Cr_q, Cb_q_d, Cr_q_d,
                        qw, qh, /*strength_c=*/1.0f);
    gat_crop_2d_topleft(Y_h, hw, hh, Y_for_q, kfw_q, kfh_q);
    gat_k16_joint_bilateral_upsample(Cb_q_d, Cr_q_d, zero_q,
                                     Y_for_q,
                                     Cb_q_to_h_raw, Cr_q_to_h_raw, zero_kfq,
                                     qw, qh, /*BW=*/3.0f);
  }
  /* --- Anchor 2 (cont.) — q→h pad + K16 h→full + h→full pad.
   * Gated by need_q_up (= matches Anchor 2 LOESS / K16 q→h block above). --- */
  float *Cb_q_up = NULL;
  float *Cr_q_up = NULL;
  if(need_q_up)
  {
    /* Pad raw K16 q→h output (kfw_q × kfh_q) → half-res (hw × hh). */
    gat_pad_2d_edge(Cb_q_to_h_raw, kfw_q, kfh_q, Cb_q_to_h, hw, hh);
    gat_pad_2d_edge(Cr_q_to_h_raw, kfw_q, kfh_q, Cr_q_to_h, hw, hh);
    /* K16 h→full (= same crop/pad pattern as Anchor 1). */
    gat_k16_joint_bilateral_upsample(Cb_q_to_h, Cr_q_to_h, zero_h,
                                     Y_for_h,
                                     Cb_q_up_raw, Cr_q_up_raw, zero_kfh,
                                     hw, hh, /*BW=*/3.0f);
    Cb_q_up = dt_alloc_align_float(npx);
    Cr_q_up = dt_alloc_align_float(npx);
    if(!Cb_q_up || !Cr_q_up) { fprintf(stderr, "[yuv_p] q_up alloc failed\n"); goto cleanup_pyr; }
    gat_pad_2d_edge(Cb_q_up_raw, kfw_half, kfh_half, Cb_q_up, width, height);
    gat_pad_2d_edge(Cr_q_up_raw, kfw_half, kfh_half, Cr_q_up, width, height);
  }

  /* --- Smoothstep slider walk: chroma_strength ∈ [0, 3].
   *
   * Calibrated semantic — chroma_strength=1.0 (default) = "expected
   * denoising" = C_full (= equivalent to GALOSH_YUV_O single-scale LOESS).
   * Lower values reduce denoising toward input; higher values enable
   * larger-scale chroma blob removal via box-down + LOESS + K16-up.
   *
   *   cs ∈ [0, 1]: blend(noisy chroma, C_full)         (= no-denoise → calibrated)
   *   cs ∈ [1, 2]: blend(C_full, C_h_up)               (= calibrated → mid-scale)
   *   cs ∈ [2, 3]: blend(C_h_up, C_q_up)               (= mid → max smoothing)
   *
   * Anchors increase smoothing reach as the slider moves up; calibrated
   * default sits at the "C_full" anchor at slider=1. */
  const float cs = chroma_strength;
  DT_OMP_FOR()
  for(size_t i = 0; i < npx; i++)
  {
    float cb, cr;
    if(cs <= 1.0f)
    {
      const float t = galosh_smoothstep_p(cs);
      cb = (1.0f - t) * cb_in[i]    + t * Cb_full[i];
      cr = (1.0f - t) * cr_in[i]    + t * Cr_full[i];
    }
    else if(cs <= 2.0f)
    {
      const float t = galosh_smoothstep_p(cs - 1.0f);
      cb = (1.0f - t) * Cb_full[i]  + t * Cb_h_up[i];
      cr = (1.0f - t) * Cr_full[i]  + t * Cr_h_up[i];
    }
    else
    {
      const float t = galosh_smoothstep_p(cs - 2.0f);
      cb = (1.0f - t) * Cb_h_up[i]  + t * Cb_q_up[i];
      cr = (1.0f - t) * Cr_h_up[i]  + t * Cr_q_up[i];
    }
    cb_out[i] = cb;
    cr_out[i] = cr;
  }

cleanup_pyr:
  dt_free_align(Cb_full);     dt_free_align(Cr_full);
  dt_free_align(Y_h);         dt_free_align(Cb_h);     dt_free_align(Cr_h);
  dt_free_align(Cb_h_d);      dt_free_align(Cr_h_d);
  dt_free_align(Cb_h_up);     dt_free_align(Cr_h_up);
  dt_free_align(Y_q);         dt_free_align(Cb_q);     dt_free_align(Cr_q);
  dt_free_align(Cb_q_d);      dt_free_align(Cr_q_d);
  dt_free_align(Y_for_q);     dt_free_align(Cb_q_to_h_raw);
  dt_free_align(Cr_q_to_h_raw); dt_free_align(zero_kfq);
  dt_free_align(Cb_q_to_h);   dt_free_align(Cr_q_to_h);
  dt_free_align(Cb_q_up_raw); dt_free_align(Cr_q_up_raw);
  dt_free_align(Cb_h_up_raw); dt_free_align(Cr_h_up_raw);
  dt_free_align(Y_for_h);     dt_free_align(zero_kfh);
  dt_free_align(Cb_q_up);     dt_free_align(Cr_q_up);
  dt_free_align(zero_h);      dt_free_align(zero_q);   dt_free_align(zero_full);
}

/* ================================================================
 * [LATEST: GALOSH_YUV_G] galosh_yuv_denoise_srgb — main entry point.
 *
 * Full GALOSH_YUV_G pipeline.  Called by main() (CLI) and by the
 * darktable IOP via galosh_yuv_denoise_linear() in galosh_yuv.h
 * (which omits Phase 0 / Phase 4 sRGB <-> linear conversions because
 * darktable's pipeline already runs on linear scene-referred RGB).
 *
 * Phases — see file-top docstring for canonical numbering:
 *   Phase 0   sRGB → linear RGB (inverse gamma)
 *   Phase 1   Linear → BT.709 YCbCr
 *   Phase 2   Y plane GAT + LOSH (use_robust_shrink=1) + Makitalo inv-GAT
 *   Phase 3   Cb/Cr Y-guided bilateral LOESS
 *   Phase 4   YCbCr → linear → sRGB gamma
 * ================================================================ */
static void galosh_yuv_denoise_srgb(const float *restrict in_srgb,
                                    float *restrict out_srgb,
                                    const int width, const int height,
                                    const float strength_y,
                                    const float strength_c,
                                    float alpha, float sigma_sq)
{
  const size_t npx = (size_t)width * (size_t)height;

  /* ============================================================
   * [LATEST: GALOSH_YUV_G] Step 1 — allocate planar Y/Cb/Cr buffers.
   * ============================================================ */
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

  /* ============================================================
   * [LATEST: GALOSH_YUV_G] Phase 0+1 — sRGB → linear → BT.709 YCbCr.
   * (darktable port skips sRGB↔linear because pipeline is already
   * linear scene-referred RGB; see galosh_yuv.h.)
   * ============================================================ */
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

  /* ============================================================
   * [LATEST: GALOSH_YUV_G] Phase 2 ① — blind α / σ² (if not supplied).
   * Uses Laplacian MAD on Y plane (estimate_sigma_plane).
   * ============================================================
   * Step 3: Blind noise estimate on Y (if user didn't supply α, σ²).
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

  /* ============================================================
   * [LATEST: GALOSH_YUV_G] Phase 2 ② — GAT forward on Y plane.
   * Variance-stabilising transform (Foi-Mäkitalo 2008/2013).
   * ============================================================
   * Step 4: GAT forward on Y.  α=0 degenerates to (Y - 0) / σ — pure
   * linear normalisation.  With α>0, the full variance-stabilising
   * transform is applied. */
  gat_build_inverse_table(alpha, sigma_sq);
  DT_OMP_FOR()
  for(size_t i = 0; i < npx; i++)
    Y_stab[i] = gat_forward(Y_stab[i], alpha, sigma_sq);

  /* ============================================================
   * [LATEST: GALOSH_YUV_G] Phase 2 ③ — unit-variance normalise Y_stab.
   * Measure σ_GAT via MAD, divide.  Matches RAW unified_sigma convention.
   * ============================================================
   * Step 5: Normalise Y_stab to unit variance in GAT space.  Measure
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

  /* ============================================================
   * [LATEST: GALOSH_YUV_G] Phase 2 ④ — Y plane LOSH (Pass1 + Pass2).
   * BayesShrink pilot + empirical Wiener via galosh_pass12_multiorient_blocked.
   * use_robust_shrink=1 (= MAD-based σ_Y, same K13-style as RAW Phase 4(a)).
   * Stride / n_orient bench-overrideable for [ARCHIVED] variants A/B.
   * ============================================================ */
  const int losh_stride = g_yuv_stride;
  galosh_pass12_multiorient_blocked(Y_stab, Y_den, width, height,
                                     strength_y, GALOSH_BLOCK_SIZE,
                                     losh_stride, g_yuv_n_orient,
                                     /*use_robust_shrink=*/1);
  (void)Y_pilot;  /* pilot now lives inside the multiorient wrapper */
  fprintf(stderr, "[GALOSH_YUV_G] Y LOSH Pass1(MAD)+Pass2 done\n");

  /* ============================================================
   * Phase 3 — Cb/Cr Y-guided chroma denoise.
   *
   * [LATEST: GALOSH_YUV_O] (default): single-scale full-res bilateral
   *   LOESS R=GALOSH_LOESS_RADIUS.  eps_gat = strength_c² × GALOSH_LOESS_
   *   TAU_SQ_INV controls local-vs-global ridge regularization.  Cannot
   *   reach large-scale (50+ px) chroma blobs (= R fixed at 7).
   *
   * [LATEST: GALOSH_YUV_Q] (--variant=q): 3-anchor multi-scale LOESS
   *   pyramid {full / half-up / quarter-up} + smoothstep slider walk +
   *   Y_stab-guided EWA Jinc-Lanczos-3 joint bilateral upsample.
   *   Mirrors RAW O Phase 7-9.  chroma_strength ∈ [0, 3] selects anchor:
   *     0 → blend(noisy, C_full)        = bypass / minimal
   *     1 → C_full                       = calibrated (= _O equivalent)
   *     2 → C_h_up                       = half-res reach (~25 px)
   *     3 → C_q_up                       = quarter-res reach (~50 px)
   *
   * [DEPRECATED: GALOSH_YUV_P] (--variant=p): same chroma pyramid as _Q
   *   PLUS spatially-smoothed BayesShrink lambda on luma (= -0.46 dB
   *   regression).  Retained for archival reproducibility only.
   *
   * All variants use NOISY Y_stab as bilateral guide (denoised Y would
   * double-filter; noisy Y has enough SNR for edge weighting).
   * ============================================================ */
  if(g_galosh_yuv_use_q || g_galosh_yuv_use_p)
  {
    galosh_yuv_chroma_pyramid_p(Y_stab, Cb_lin, Cr_lin, Cb_den, Cr_den,
                                width, height, strength_c);
    fprintf(stderr, "[yuv_cpu_%c] chroma pyramid done (3-anchor smoothstep, "
                    "chroma_strength=%.2f)\n",
            g_galosh_yuv_use_q ? 'Q' : 'P', strength_c);
  }
  else
  {
    galosh_loess_chroma(Y_stab, Cb_lin, Cr_lin, Cb_den, Cr_den,
                        width, height, strength_c);
    fprintf(stderr, "[yuv_cpu_O] Cb/Cr LOESS done (R=%d BW=%.1f)\n",
            GALOSH_LOESS_RADIUS, GALOSH_LOESS_BW);
  }

  /* ============================================================
   * [LATEST: GALOSH_YUV_G] Phase 2 ⑤ + Phase 4 — Y inverse GAT + sRGB.
   * Makitalo-Foi exact-unbiased inverse GAT on Y_den, then YCbCr →
   * linear → sRGB gamma.
   * ============================================================ */
  DT_OMP_FOR()
  for(int y = 0; y < height; y++)
  {
    for(int x = 0; x < width; x++)
    {
      const size_t i = (size_t)y * width + x;
      const float Y_inv = gat_inverse_exact(Y_den[i] * unified_sigma);
      float R, G, B;
      ycbcr_to_rgb(Y_inv, Cb_den[i], Cr_den[i], &R, &G, &B);
      /* clip linear RGB to [0,1] BEFORE the OETF (Ph3 2026-06-28): the BT.709
       * inverse + chroma denoise can push linear RGB slightly out of gamut; clipping
       * here keeps the sRGB OETF well-defined and emits a valid [0,1] image.  Matches
       * the GPU galosh_yuv_ycbcr_to_srgb (CPU<->GPU consistent). */
      out_srgb[3 * i + 0] = linear_to_srgb_f(fminf(fmaxf(R, 0.0f), 1.0f));
      out_srgb[3 * i + 1] = linear_to_srgb_f(fminf(fmaxf(G, 0.0f), 1.0f));
      out_srgb[3 * i + 2] = linear_to_srgb_f(fminf(fmaxf(B, 0.0f), 1.0f));
    }
  }

cleanup:
  dt_free_align(Y_stab);  dt_free_align(Y_pilot); dt_free_align(Y_den);
  dt_free_align(Cb_lin);  dt_free_align(Cr_lin);
  dt_free_align(Cb_den);  dt_free_align(Cr_den);
}

/* ================================================================
 * [GALOSH-420 2026-07-12] Noise-adaptive chroma LOESS radius for the
 * NATIVE half-resolution 4:2:0 lattice.
 *
 * EN: GALOSH_LOESS_RADIUS (=7, 15x15 window) is calibrated for a FULL-
 *     resolution chroma lattice.  Applied unchanged on the 4:2:0 half-res
 *     lattice it covers ~30x30 full-res pixels (4x the area), and because
 *     galosh_loess_chroma is a pure guide-Y projection with NO fidelity
 *     term on the input chroma (out = a*Y_c + b), that oversized window
 *     annihilates chroma structure orthogonal to local luma (iso-luminant
 *     hue edges).  Measured effect: on Set8 the native-420 Cr fell BELOW
 *     the noisy input at sigma20 (34.01 vs 34.53 dB); the 444-then-
 *     downsample route did not (34.88).  Matching the footprint to the
 *     444 route restores it.  The optimum is mildly noise-dependent
 *     (bias-variance crossover): low noise favours a tight window
 *     (structure preservation), high noise a slightly wider one (variance
 *     reduction).  Calibrated on Set8 (tractor+snowboard f40, sigma 10-50;
 *     benchmark/results_set8_video/_diag): the per-sigma argmax over
 *     R in {1..7} is R=2 for sigma_lin<=~0.023 (sigma10/20) and R=3 for
 *     sigma_lin>=~0.031 (sigma30-50).  Threshold 0.027 (the midpoint)
 *     recovers Cr ~+1.8 dB at sigma20 and ~+0.6 at sigma50 and beats the
 *     444 route, while R=2/R=3 differ by <0.1 dB right at the boundary
 *     (insensitive).  sigma_lin = sqrt(sigma_sq) = the blind MAD estimate.
 *     Only 4:2:0 uses this (4:2:2 / 4:4:4 chroma is processed at full res,
 *     where GALOSH_LOESS_RADIUS is already correct).  Env
 *     GALOSH_YUV420_CHROMA_R=<1..7> pins a fixed radius (A/B, regression
 *     bisection, or extreme-noise override beyond the calibrated range).
 * JP: 半解像度4:2:0クロマのLOESS半径をブラインドσから決定。R=7は
 *     フル解像度較正値で、半解像度では実効footprintが約2倍(面積4倍)に
 *     なり、fidelity項の無いguide-Y射影(out=a*Y_c+b)が等輝度色エッジ
 *     (色相のみの境界)を過平均で消す(Set8 σ20でnative-420のCrがnoisy
 *     未満34.01<34.53に劣化、444経由は34.88で無事)。444経路とfootprintを
 *     揃えると回復。最適はσ依存(bias-variance crossover): σ小→R小(構造
 *     保存)/σ大→Rやや大(分散低減)。Set8較正でσ_lin<0.027→R=2, 以上→R=3
 *     (境界でのR=2/3差は<0.1dBで鈍感)。422/444はフル解像度なので不変。
 *     GALOSH_YUV420_CHROMA_R で固定Rを強制(A/B・回帰二分・範囲外の高σ)。
 * ================================================================ */
static int galosh_yuv420_chroma_radius(float sigma_lin)
{
  const char *e = getenv("GALOSH_YUV420_CHROMA_R");
  if(e && e[0]) { const int v = atoi(e); if(v >= 1 && v <= 7) return v; }
  return (sigma_lin < 0.027f) ? 2 : 3;
}

/* ================================================================
 * [LATEST: GALOSH-420] galosh_yuv_denoise_linear_rgb — LINEAR-domain twin
 * of galosh_yuv_denoise_srgb (Phase 1-3 identical; Phase 0/4 gamma codecs
 * removed; the final linear [0,1] clip is retained).
 *
 * EN: DELIBERATE COPY, not a refactor: the legacy sRGB entry point above
 *     is the validated production path and must stay byte-identical under
 *     -O3 (identity harness 2026-07-11); routing it through a shared body
 *     risks codegen drift.  This twin is the half-resolution chroma-lattice
 *     core of the GALOSH-420 path (spec §eotf): the caller reconstructs
 *     half-res gamma R'G'B' from (siting-phased guide Y', Cb, Cr),
 *     linearises with the container EOTF, calls this, re-encodes, and
 *     extracts denoised Cb/Cr.  Exactly the form validated by the A/B rig
 *     (arm A = native, ab_yuv420.py, 2026-07-11: +0.3..0.5 dB vs 444-first).
 * JP: 意図的なコピー（リファクタ禁止）— レガシー sRGB パスのバイト一致を
 *     保証するため。420 パスの半解像度クロマ格子コア。A/B リグ A アーム
 *     （ネイティブ勝ち）と同一形態。
 * ================================================================ */
static void galosh_yuv_denoise_linear_rgb(const float *restrict in_lin,
                                          float *restrict out_lin,
                                          const int width, const int height,
                                          const float strength_y,
                                          const float strength_c,
                                          float alpha, float sigma_sq)
{
  const size_t npx = (size_t)width * (size_t)height;

  float *Y_stab  = dt_alloc_align_float(npx);
  float *Cb_lin  = dt_alloc_align_float(npx);
  float *Cr_lin  = dt_alloc_align_float(npx);
  float *Y_den   = dt_alloc_align_float(npx);
  float *Cb_den  = dt_alloc_align_float(npx);
  float *Cr_den  = dt_alloc_align_float(npx);
  if(!Y_stab || !Cb_lin || !Cr_lin || !Y_den || !Cb_den || !Cr_den)
  {
    fprintf(stderr, "[yuv420_core] alloc failed\n");
    goto cleanup;
  }

  /* Phase 1 — linear RGB → BT.709 internal working YCbCr (same internal
   * space as the legacy path; the container matrix lives in the caller). */
  DT_OMP_FOR()
  for(size_t i = 0; i < npx; i++)
  {
    float Y, Cb, Cr;
    rgb_to_ycbcr(in_lin[3 * i + 0], in_lin[3 * i + 1], in_lin[3 * i + 2],
                 &Y, &Cb, &Cr);
    Y_stab[i] = Y;
    Cb_lin[i] = Cb;
    Cr_lin[i] = Cr;
  }

  /* Phase 2 ① — blind α / σ² on the internal Y (if not supplied). */
  if(alpha <= 0.0f || sigma_sq <= 0.0f)
  {
    const float sigma_lin = estimate_sigma_plane(Y_stab, width, height);
    sigma_sq = fmaxf(sigma_lin * sigma_lin, 1e-8f);
    alpha    = fmaxf(sigma_lin * 0.1f, 1e-5f);
    fprintf(stderr, "[yuv420_core] Blind σ (MAD) = %.5f, α=%.6g σ²=%.6e\n",
            sigma_lin, alpha, sigma_sq);
  }

  /* Phase 2 ②③ — GAT forward + unit-variance normalise. */
  gat_build_inverse_table(alpha, sigma_sq);
  DT_OMP_FOR()
  for(size_t i = 0; i < npx; i++)
    Y_stab[i] = gat_forward(Y_stab[i], alpha, sigma_sq);
  const float sigma_gat = estimate_sigma_plane(Y_stab, width, height);
  const float unified_sigma = fmaxf(sigma_gat, 1e-6f);
  const float inv_sg = 1.0f / unified_sigma;
  DT_OMP_FOR()
  for(size_t i = 0; i < npx; i++) Y_stab[i] *= inv_sg;

  /* Phase 2 ④ — Y LOSH Pass1(MAD)+Pass2. */
  galosh_pass12_multiorient_blocked(Y_stab, Y_den, width, height,
                                     strength_y, GALOSH_BLOCK_SIZE,
                                     g_yuv_stride, g_yuv_n_orient,
                                     /*use_robust_shrink=*/1);

  /* Phase 3 — Cb/Cr Y_stab-guided chroma denoise (variant O/Q/P as set). */
  if(g_galosh_yuv_use_q || g_galosh_yuv_use_p)
    galosh_yuv_chroma_pyramid_p(Y_stab, Cb_lin, Cr_lin, Cb_den, Cr_den,
                                width, height, strength_c);
  else if(g_yuv420_halfres_chroma)
  {
    /* [GALOSH-420 2026-07-12] noise-adaptive half-res radius; see
     * galosh_yuv420_chroma_radius above.  sigma_lin = sqrt(sigma_sq). */
    const float sigma_lin = sqrtf(sigma_sq);
    const int R = galosh_yuv420_chroma_radius(sigma_lin);
    fprintf(stderr, "[yuv420_core] half-res chroma LOESS R=%d "
            "(adaptive, sigma_lin=%.5f)\n", R, sigma_lin);
    galosh_loess_chroma_r(Y_stab, Cb_lin, Cr_lin, Cb_den, Cr_den,
                          width, height, strength_c, R);
  }
  else
    galosh_loess_chroma(Y_stab, Cb_lin, Cr_lin, Cb_den, Cr_den,
                        width, height, strength_c);

  /* Phase 2 ⑤ + linear re-assembly.  Clip linear RGB to [0,1] exactly as
   * the legacy path does before its OETF (validated A/B chain). */
  DT_OMP_FOR()
  for(size_t i = 0; i < npx; i++)
  {
    const float Y_inv = gat_inverse_exact(Y_den[i] * unified_sigma);
    float R, G, B;
    ycbcr_to_rgb(Y_inv, Cb_den[i], Cr_den[i], &R, &G, &B);
    out_lin[3 * i + 0] = fminf(fmaxf(R, 0.0f), 1.0f);
    out_lin[3 * i + 1] = fminf(fmaxf(G, 0.0f), 1.0f);
    out_lin[3 * i + 2] = fminf(fmaxf(B, 0.0f), 1.0f);
  }

cleanup:
  dt_free_align(Y_stab); dt_free_align(Y_den);
  dt_free_align(Cb_lin); dt_free_align(Cr_lin);
  dt_free_align(Cb_den); dt_free_align(Cr_den);
}

/* ================================================================
 * [LATEST: GALOSH-420] galosh_yuv_denoise_luma_plane — Phase 2 only
 * (GAT + LOSH + exact-unbiased inverse) on a single LINEAR luma plane.
 *
 * EN: The GALOSH-420 full-resolution Y' path: chroma-INdependent by
 *     construction (container Y'_den = EOTF(denoise(EOTF^-1(Y'))); the
 *     denoised chroma never leaks into the output luma).  NCL note (spec
 *     §eotf, documented approximation): EOTF^-1(Y') is not true luminance;
 *     the blind PG fit runs on that domain and is self-consistent.
 * JP: 420 のフル解像度ルマ経路。出力 Y' はクロマ非依存（NCL 近似は
 *     spec に明記、ブラインド適合はその領域で自己整合）。
 * ================================================================ */
static void galosh_yuv_denoise_luma_plane(const float *restrict y_lin,
                                          float *restrict y_out,
                                          const int width, const int height,
                                          const float strength_y,
                                          float alpha, float sigma_sq)
{
  const size_t npx = (size_t)width * (size_t)height;
  float *Y_stab = dt_alloc_align_float(npx);
  float *Y_den  = dt_alloc_align_float(npx);
  if(!Y_stab || !Y_den)
  {
    fprintf(stderr, "[yuv420_luma] alloc failed\n");
    goto cleanup;
  }

  if(alpha <= 0.0f || sigma_sq <= 0.0f)
  {
    const float sigma_lin = estimate_sigma_plane(y_lin, width, height);
    sigma_sq = fmaxf(sigma_lin * sigma_lin, 1e-8f);
    alpha    = fmaxf(sigma_lin * 0.1f, 1e-5f);
    fprintf(stderr, "[yuv420_luma] Blind σ (MAD) = %.5f, α=%.6g σ²=%.6e\n",
            sigma_lin, alpha, sigma_sq);
  }

  gat_build_inverse_table(alpha, sigma_sq);
  DT_OMP_FOR()
  for(size_t i = 0; i < npx; i++)
    Y_stab[i] = gat_forward(y_lin[i], alpha, sigma_sq);
  const float sigma_gat = estimate_sigma_plane(Y_stab, width, height);
  const float unified_sigma = fmaxf(sigma_gat, 1e-6f);
  const float inv_sg = 1.0f / unified_sigma;
  DT_OMP_FOR()
  for(size_t i = 0; i < npx; i++) Y_stab[i] *= inv_sg;

  galosh_pass12_multiorient_blocked(Y_stab, Y_den, width, height,
                                     strength_y, GALOSH_BLOCK_SIZE,
                                     g_yuv_stride, g_yuv_n_orient,
                                     /*use_robust_shrink=*/1);

  DT_OMP_FOR()
  for(size_t i = 0; i < npx; i++)
    y_out[i] = gat_inverse_exact(Y_den[i] * unified_sigma);

cleanup:
  dt_free_align(Y_stab);
  dt_free_align(Y_den);
}

/* ================================================================
 * [LATEST: GALOSH-420] planar-container driver.
 *
 * Pipeline (4:2:0; spec docs/yuv420_frontend_spec.md):
 *   1. dequantise container codes → float gamma-domain Y' / Cb / Cr
 *      (range/depth per flags; excursions preserved, no input clamp)
 *   2. LUMA  (full res):  Y'_den = EOTF(luma_plane(EOTF^-1(Y')))
 *   3. CHROMA (half res, NATIVE lattice — the A/B-validated form):
 *      a. guide Yg' = galosh420_down_luma(Y', siting)   [phase-matched!]
 *      b. R'G'B'_h  = NCL^-1(Yg', Cb, Cr)  (container matrix)
 *      c. RGB_h     = EOTF^-1(R'G'B'_h)
 *      d. galosh_yuv_denoise_linear_rgb() at the chroma lattice
 *      e. R'G'B'_h_den = EOTF(...) → NCL → keep (Cb_den, Cr_den)
 *   4. requantise; write planar (format-preserving: Y full res + chroma
 *      at its native lattice; output clipped to [0,1] gamma / code range)
 *   4:2:2 → provisional 444 (horizontal co-sited up/down around the
 *   full-res core); 4:4:4 planar → direct full-res core; 4:0:0 → Y only.
 * ================================================================ */
static int galosh_yuv420_main(const char *in_path, const char *out_path,
                              const int W, const int H,
                              const float strength_y, const float strength_c,
                              const float alpha_cli, const float sigma_cli,
                              const galosh420_pix_t pix,
                              const galosh420_siting_t siting,
                              const galosh420_eotf_t eotf,
                              const galosh420_matrix_t mat,
                              const galosh420_range_t range,
                              const int depth)
{
  if(pix == GALOSH420_PIX_420 && ((W | H) & 1))
  { fprintf(stderr, "[yuv420] 4:2:0 requires even W and H (got %dx%d)\n", W, H); return 1; }
  if(pix == GALOSH420_PIX_422 && (W & 1))
  { fprintf(stderr, "[yuv420] 4:2:2 requires even W (got %d)\n", W); return 1; }

  const int cw = (pix == GALOSH420_PIX_400) ? 0
               : (pix == GALOSH420_PIX_444) ? W : W / 2;
  const int ch = (pix == GALOSH420_PIX_400) ? 0
               : (pix == GALOSH420_PIX_420) ? H / 2 : H;
  const size_t ysz = (size_t)W * H;
  const size_t csz = (size_t)cw * ch;
  const int wide = (depth > 8);
  const size_t code_bytes = (ysz + 2 * csz) * (wide ? 2 : 1);

  fprintf(stderr, "[yuv420] pix=%d %dx%d chroma %dx%d depth=%d range=%s "
                  "matrix Kr=%.4f Kb=%.4f eotf=%d siting=%d\n",
          (int)pix, W, H, cw, ch, depth,
          range == GALOSH420_RANGE_LIMITED ? "limited" : "full",
          mat.kr, mat.kb, (int)eotf, (int)siting);

  /* ---- read + dequantise --------------------------------------- */
  uint8_t *raw = (uint8_t *)malloc(code_bytes);
  float *Yp = dt_alloc_align_float(ysz);
  float *Cb = csz ? dt_alloc_align_float(csz) : NULL;
  float *Cr = csz ? dt_alloc_align_float(csz) : NULL;
  float *Ywork = dt_alloc_align_float(ysz);
  if(!raw || !Yp || !Ywork || (csz && (!Cb || !Cr)))
  { fprintf(stderr, "[yuv420] alloc failed\n"); return 1; }

  FILE *fi = fopen(in_path, "rb");
  if(!fi) { fprintf(stderr, "cannot open %s\n", in_path); return 1; }
  const size_t rd = fread(raw, 1, code_bytes, fi);
  fclose(fi);
  if(rd != code_bytes)
  { fprintf(stderr, "[yuv420] short read (%zu of %zu — check --pix/--depth/W/H)\n",
            rd, code_bytes); return 1; }

  const uint16_t *raw16 = (const uint16_t *)raw;   /* LE host */
  DT_OMP_FOR()
  for(size_t i = 0; i < ysz; i++)
    Yp[i] = galosh420_dequant_y(wide ? (float)raw16[i] : (float)raw[i],
                                depth, range);
  DT_OMP_FOR()
  for(size_t i = 0; i < csz; i++)
  {
    Cb[i] = galosh420_dequant_c(wide ? (float)raw16[ysz + i]
                                     : (float)raw[ysz + i], depth, range);
    Cr[i] = galosh420_dequant_c(wide ? (float)raw16[ysz + csz + i]
                                     : (float)raw[ysz + csz + i], depth, range);
  }

  /* ---- LUMA path (full res, chroma-independent) ------------------ */
  DT_OMP_FOR()
  for(size_t i = 0; i < ysz; i++)
    Ywork[i] = galosh420_eotf_inv_f(Yp[i], eotf);
  float *Yden = dt_alloc_align_float(ysz);
  if(!Yden) { fprintf(stderr, "[yuv420] alloc failed\n"); return 1; }
  galosh_yuv_denoise_luma_plane(Ywork, Yden, W, H,
                                strength_y, alpha_cli, sigma_cli);
  fprintf(stderr, "[yuv420] luma done\n");

  /* ---- CHROMA path ------------------------------------------------ */
  float *CbD = NULL, *CrD = NULL;    /* denoised chroma at native lattice */
  if(pix != GALOSH420_PIX_400)
  {
    const int pw = (pix == GALOSH420_PIX_420) ? cw : W;   /* processing dims */
    const int ph = (pix == GALOSH420_PIX_420) ? ch : H;
    const size_t psz = (size_t)pw * ph;

    /* guide Y' + chroma at the processing lattice (gamma domain) */
    float *Yg  = dt_alloc_align_float(psz);
    float *Cbp = dt_alloc_align_float(psz);
    float *Crp = dt_alloc_align_float(psz);
    float *rgb = dt_alloc_align_float(psz * 3);
    if(!Yg || !Cbp || !Crp || !rgb)
    { fprintf(stderr, "[yuv420] alloc failed\n"); return 1; }

    if(pix == GALOSH420_PIX_420)
    {
      galosh420_down_luma(Yp, W, H, Yg, siting);   /* siting-phased guide */
      memcpy(Cbp, Cb, csz * sizeof(float));
      memcpy(Crp, Cr, csz * sizeof(float));
    }
    else if(pix == GALOSH420_PIX_422)
    {
      memcpy(Yg, Yp, ysz * sizeof(float));         /* co-located after upH */
      galosh420_up422_h(Cb, cw, H, Cbp);
      galosh420_up422_h(Cr, cw, H, Crp);
    }
    else /* 444 planar */
    {
      memcpy(Yg, Yp, ysz * sizeof(float));
      memcpy(Cbp, Cb, csz * sizeof(float));
      memcpy(Crp, Cr, csz * sizeof(float));
    }

    /* NCL^-1 + EOTF^-1 → linear RGB at the processing lattice */
    DT_OMP_FOR()
    for(size_t i = 0; i < psz; i++)
    {
      float R, G, B;
      galosh420_ncl_inv(Yg[i], Cbp[i], Crp[i], mat, &R, &G, &B);
      rgb[3 * i + 0] = galosh420_eotf_inv_f(R, eotf);
      rgb[3 * i + 1] = galosh420_eotf_inv_f(G, eotf);
      rgb[3 * i + 2] = galosh420_eotf_inv_f(B, eotf);
    }

    /* [GALOSH-420 2026-07-12] enable the noise-adaptive chroma radius ONLY
     * on the native half-res 4:2:0 lattice (4:2:2 / 4:4:4 process chroma at
     * full res, where the full-res GALOSH_LOESS_RADIUS is already correct). */
    g_yuv420_halfres_chroma = (pix == GALOSH420_PIX_420) ? 1 : 0;
    galosh_yuv_denoise_linear_rgb(rgb, rgb, pw, ph,
                                  strength_y, strength_c,
                                  alpha_cli, sigma_cli);
    g_yuv420_halfres_chroma = 0;

    /* EOTF + NCL → extract denoised Cb/Cr at the processing lattice */
    DT_OMP_FOR()
    for(size_t i = 0; i < psz; i++)
    {
      const float Rp = galosh420_eotf_fwd_f(rgb[3 * i + 0], eotf);
      const float Gp = galosh420_eotf_fwd_f(rgb[3 * i + 1], eotf);
      const float Bp = galosh420_eotf_fwd_f(rgb[3 * i + 2], eotf);
      float yy;
      galosh420_ncl_fwd(Rp, Gp, Bp, mat, &yy, &Cbp[i], &Crp[i]);
    }

    CbD = dt_alloc_align_float(csz);
    CrD = dt_alloc_align_float(csz);
    if(!CbD || !CrD) { fprintf(stderr, "[yuv420] alloc failed\n"); return 1; }
    if(pix == GALOSH420_PIX_422)
    {
      galosh420_down422_h(Cbp, W, H, CbD);
      galosh420_down422_h(Crp, W, H, CrD);
    }
    else
    {
      memcpy(CbD, Cbp, csz * sizeof(float));
      memcpy(CrD, Crp, csz * sizeof(float));
    }
    dt_free_align(Yg); dt_free_align(Cbp); dt_free_align(Crp);
    dt_free_align(rgb);
    fprintf(stderr, "[yuv420] chroma done (native lattice %dx%d)\n", pw, ph);
  }

  /* ---- requantise + write (format-preserving) --------------------- */
  DT_OMP_FOR()
  for(size_t i = 0; i < ysz; i++)
  {
    const float yl = fminf(fmaxf(Yden[i], 0.0f), 1.0f);
    const int c = galosh420_requant_y(galosh420_eotf_fwd_f(yl, eotf),
                                      depth, range);
    if(wide) ((uint16_t *)raw)[i] = (uint16_t)c;
    else     raw[i] = (uint8_t)c;
  }
  DT_OMP_FOR()
  for(size_t i = 0; i < csz; i++)
  {
    const int cb = galosh420_requant_c(CbD[i], depth, range);
    const int cr = galosh420_requant_c(CrD[i], depth, range);
    if(wide)
    {
      ((uint16_t *)raw)[ysz + i]       = (uint16_t)cb;
      ((uint16_t *)raw)[ysz + csz + i] = (uint16_t)cr;
    }
    else
    {
      raw[ysz + i]       = (uint8_t)cb;
      raw[ysz + csz + i] = (uint8_t)cr;
    }
  }

  FILE *fo = fopen(out_path, "wb");
  if(!fo) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }
  const size_t wr = fwrite(raw, 1, code_bytes, fo);
  fclose(fo);
  if(wr != code_bytes)
  { fprintf(stderr, "short write (%zu of %zu)\n", wr, code_bytes); return 1; }

  free(raw);
  dt_free_align(Yp); dt_free_align(Cb); dt_free_align(Cr);
  dt_free_align(Ywork); dt_free_align(Yden);
  dt_free_align(CbD); dt_free_align(CrD);
  return 0;
}

/* ================================================================
 * main: CLI driver (same binary I/O style as galosh_raw_cpu).
 *
 * EN: GALOSH_YUV_NOMAIN excludes this CLI entry so the whole file can be
 *     #include'd as a library translation unit by embedders (VapourSynth
 *     plugin, darktable IOP) — the static pipeline functions above become
 *     directly callable in the including TU.  The canonical exe build is
 *     untouched (flag absent ⇒ byte-identical, identity harness).
 * JP: 組み込み用（VS プラグイン / darktable IOP が .c ごと include）。
 *     フラグ無しの正規 exe ビルドはバイト一致のまま。
 * ================================================================ */
#ifndef GALOSH_YUV_NOMAIN
int main(int argc, char **argv)
{
  /* GALOSH-420 planar-container options (spec docs/yuv420_frontend_spec.md);
   * all externally specified, no auto-detection. */
  galosh420_pix_t    g420_pix    = GALOSH420_PIX_444;  /* 444 = legacy sRGB float mode */
  int                g420_planar = 0;                  /* any --pix= given */
  galosh420_siting_t g420_siting = GALOSH420_SITING_CENTER;
  galosh420_eotf_t   g420_eotf   = GALOSH420_EOTF_SRGB;
  galosh420_matrix_t g420_mat    = GALOSH420_MAT_BT709;
  galosh420_range_t  g420_range  = GALOSH420_RANGE_FULL;
  int                g420_depth  = 8;

  /* Strip --stride / --orient flags before positional parsing. */
  int new_argc = 0;
  char *positional[32];
  for(int i = 0; i < argc; i++)
  {
    const char *a = argv[i];
    if(strcmp(a, "--selftest-phase") == 0)
    {
      /* MANDATORY affine-field phase verification (spec §guide
       * construction).  Machine-proves the siting-phased guide. */
      return galosh420_phase_selftest();
    }
    else if(strncmp(a, "--pix=", 6) == 0)
    {
      const char *v = a + 6;
      if(!strcmp(v, "420"))      g420_pix = GALOSH420_PIX_420;
      else if(!strcmp(v, "422")) g420_pix = GALOSH420_PIX_422;
      else if(!strcmp(v, "400")) g420_pix = GALOSH420_PIX_400;
      else if(!strcmp(v, "444")) g420_pix = GALOSH420_PIX_444;
      else { fprintf(stderr, "bad --pix=%s (420|422|444|400)\n", v); return 1; }
      g420_planar = 1;
    }
    else if(strncmp(a, "--siting=", 9) == 0)
    {
      if(galosh420_parse_siting(a + 9, &g420_siting))
      { fprintf(stderr, "bad --siting=%s (center|left|topleft)\n", a + 9); return 1; }
    }
    else if(strncmp(a, "--eotf=", 7) == 0)
    {
      if(galosh420_parse_eotf(a + 7, &g420_eotf))
      { fprintf(stderr, "bad --eotf=%s (srgb|g22|g24|bt709|hlg|pq|linear)\n", a + 7); return 1; }
    }
    else if(strncmp(a, "--matrix=", 9) == 0)
    {
      if(galosh420_parse_matrix(a + 9, &g420_mat))
      { fprintf(stderr, "bad --matrix=%s (bt601|bt709|bt2020|custom:Kr,Kb)\n", a + 9); return 1; }
    }
    else if(strncmp(a, "--range=", 8) == 0)
    {
      if(galosh420_parse_range(a + 8, &g420_range))
      { fprintf(stderr, "bad --range=%s (full|limited)\n", a + 8); return 1; }
    }
    else if(strncmp(a, "--depth=", 8) == 0)
    {
      g420_depth = atoi(a + 8);
      if(g420_depth < 8 || g420_depth > 16)
      { fprintf(stderr, "bad --depth=%d (8..16)\n", g420_depth); return 1; }
    }
    else if(strncmp(a, "--stride=", 9) == 0)
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
    else if(strncmp(a, "--variant=", 10) == 0)
    {
      /* Pipeline variant selector.
       *   --variant=o (default)  = [LATEST: GALOSH_YUV_O] = production
       *                            (per-block BayesShrink lambda;
       *                             single-scale chroma LOESS).
       *   --variant=q            = [LATEST: GALOSH_YUV_Q] = production
       *                            candidate (multi-scale chroma LOESS
       *                            pyramid mirroring RAW O Phase 7-9;
       *                            luma identical to _O).
       *   --variant=p            = [DEPRECATED: GALOSH_YUV_P] = failed
       *                            try (spatially-smoothed BayesShrink
       *                            on luma + same multi-scale chroma;
       *                            archived only). */
      const char ch = (a[10] == 'P') ? 'p' : (a[10] == 'Q') ? 'q' : a[10];
      g_galosh_yuv_use_q     = (ch == 'q') ? 1 : 0;
      g_galosh_yuv_use_p     = (ch == 'p') ? 1 : 0;
      /* Spatially-smoothed BayesShrink only on [DEPRECATED] _P. */
      g_galosh_lambda_smooth = g_galosh_yuv_use_p;
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
            "       [--stride=1|2] [--orient=1|4] [--variant=o|q|p]\n"
            "  GALOSH-420 planar container mode (spec docs/yuv420_frontend_spec.md):\n"
            "       [--pix=420|422|444|400] [--depth=8..16] [--range=full|limited]\n"
            "       [--matrix=bt601|bt709|bt2020|custom:Kr,Kb]\n"
            "       [--eotf=srgb|g22|g24|bt709|hlg|pq|linear]\n"
            "       [--siting=center|left|topleft]  [--selftest-phase]\n"
            "  planar in/out = Y plane + Cb + Cr, row-major, uint8 (depth 8)\n"
            "  or uint16 LE (depth 9-16); chroma at native lattice.\n",
            argv[0]);
    return 1;
  }

  const char vtag = g_galosh_yuv_use_p ? 'P'
                  : g_galosh_yuv_use_q ? 'Q' : 'O';
  fprintf(stderr, "[GALOSH_YUV_%c] stride=%d orient=%d lambda_smooth=%d\n",
          vtag, g_yuv_stride, g_yuv_n_orient, g_galosh_lambda_smooth);

  const char *in_path  = argv[1];
  const char *out_path = argv[2];
  const int width  = atoi(argv[3]);
  const int height = atoi(argv[4]);
  const float strength_y = (argc > 5) ? (float)atof(argv[5]) : 1.0f;
  const float strength_c = (argc > 6) ? (float)atof(argv[6]) : 1.0f;
  const float alpha_cli  = (argc > 7) ? (float)atof(argv[7]) : 0.0f;
  const float sigma_cli  = (argc > 8) ? (float)atof(argv[8]) : 0.0f;

  if(width <= 0 || height <= 0 ||
     (size_t)width > SIZE_MAX / (3 * sizeof(float)) / (size_t)height)
  {
    fprintf(stderr, "invalid dimensions %dx%d\n", width, height);
    return 1;
  }

  init_galosh_kaiser();

  /* GALOSH-420 planar-container route (--pix given).  The flags-off
   * default remains the legacy sRGB float path below, byte-identical
   * (identity harness 2026-07-11). */
  if(g420_planar)
    return galosh_yuv420_main(in_path, out_path, width, height,
                              strength_y, strength_c, alpha_cli, sigma_cli,
                              g420_pix, g420_siting, g420_eotf,
                              g420_mat, g420_range, g420_depth);

  const size_t npx = (size_t)width * (size_t)height;
  const size_t nbytes = npx * 3 * sizeof(float);

  float *in_buf  = dt_alloc_align_float(npx * 3);
  float *out_buf = dt_alloc_align_float(npx * 3);
  if(!in_buf || !out_buf)
  {
    fprintf(stderr, "alloc failed\n");
    dt_free_align(in_buf); dt_free_align(out_buf);
    return 1;
  }

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
  const size_t wr = fwrite(out_buf, 1, nbytes, fo);
  fclose(fo);
  if(wr != nbytes) { fprintf(stderr, "short write (%zu of %zu)\n", wr, nbytes); return 1; }

  dt_free_align(in_buf);
  dt_free_align(out_buf);
  return 0;
}
#endif /* GALOSH_YUV_NOMAIN */
