/*
 * ================================================================
 * galosh_yuv_gat.cl — Linear-domain Y-GAT + Y-driven chroma VST
 *
 * EN: Extension of GALOSH to sRGB inputs via linear-domain YCbCr
 *     decomposition. The luma plane follows the original Poisson-Gauss
 *     GALOSH pipeline (GAT → LOSH → Makitalo-Foi inverse), while the
 *     chroma planes use a Y-driven *linear* variance-stabilising
 *     transform `Cb_stab = Cb / sqrt(α_Cb · Y_lin + σ²_Cb)` that is
 *     the exact VST for Gaussian noise whose variance depends only on
 *     the luminance of the same pixel.
 *
 * JP: GALOSH を sRGB 入力に拡張する実装。sRGB → linear → YCbCr に
 *     戻したうえで、Y 平面には従来の Poisson-Gauss GAT+LOSH を、
 *     Cb/Cr 平面には Y 駆動の線形 VST (`÷√(α_Cb·Y+σ²_Cb)`) を適用。
 *     Cb/Cr のノイズは R/G/B の shot/read noise の線形和だから、
 *     分散は Cb 自身ではなく同画素の Y で決まる ─ この事実を利用した
 *     1 回の blind estimation で 3 平面全てをカバーする設計。
 *
 * Physical rationale (see plan cosmic-brewing-toast.md):
 *     Var[Y_lin]  ≈ α_Y · Y_lin + σ²_Y           → GAT (2√) non-linear
 *     Var[Cb_lin] ≈ α_Cb · Y_lin + σ²_Cb         → linear ÷√ scaling
 *     Var[Cr_lin] ≈ α_Cr · Y_lin + σ²_Cr         → linear ÷√ scaling
 *     α_Cb ≈ 0.375·α_Y   σ²_Cb ≈ 0.375·σ²_Y      (BT.709 + iid shot)
 *     α_Cr ≈ 0.605·α_Y   σ²_Cr ≈ 0.605·σ²_Y
 *
 * All kernels operate entirely on GPU; CPU does only input-read and
 * output-write (1 clEnqueueWriteBuffer + 1 clEnqueueReadBuffer).
 *
 * Copyright (c) 2026 luxgrain. All rights reserved.
 * ================================================================ */

/* sRGB EOTF parameters (IEC 61966-2-1) */
#define SRGB_A     0.055f
#define SRGB_ONEP  1.055f
#define SRGB_GAMMA 2.4f
#define SRGB_PHI   12.92f
#define SRGB_THRS  0.04045f
#define SRGB_THRS_L 0.0031308f

/* BT.709 linear YCbCr coefficients */
#define BT709_KR 0.2126f
#define BT709_KG 0.7152f
#define BT709_KB 0.0722f
/* Denominators from (B-Y)/(1-K_B) and (R-Y)/(1-K_R) — full-range video
 * style; we keep Cb/Cr CENTRED AT 0 (not +0.5) for ease of VST. */
#define BT709_CB_DEN 1.8556f   /* 2 * (1 - KB) */
#define BT709_CR_DEN 1.5748f   /* 2 * (1 - KR) */

/* Chroma variance-coefficient scaling.
 * Derived from the fact that Cb = (B - Y)/(2(1-KB)); noise in (B,Y) is
 * almost independent (cov ≈ 0.07·α·Y), so Var[Cb] = (Var[B]+(1-KB)²·
 * Var[Y] - 2·(1-KB)·Cov[B,Y]) / (2(1-KB))². Plugging in the Poisson-
 * Gauss shot-noise model gives α_Cb/α_Y ≈ 0.375, α_Cr/α_Y ≈ 0.605.
 * σ² ratios follow the same identity (same linear combination). */
#define CHROMA_RATIO_CB 0.375f
#define CHROMA_RATIO_CR 0.605f

/* Params buffer indices for Y-GAT mode (extend of PARAMS_SIZE=32) */
#define YG_P_LUMA_STR     11
#define YG_P_CHROMA_STR   12
#define YG_P_ALPHA_Y      13
#define YG_P_SIGMA_SQ_Y   14
#define YG_P_DARK_MAX     15
#define YG_P_ALPHA_CB     16
#define YG_P_SIGMA_SQ_CB  17
#define YG_P_ALPHA_CR     18
#define YG_P_SIGMA_SQ_CR  19
#define YG_P_SIGMA_Y      20  /* mean √(α·Y+σ²) for Y — auto-selected sigma_strength */
#define YG_P_SIGMA_CB     21  /* mean √(α_Cb·Y+σ²_Cb) — for LOSH strength tuning */
#define YG_P_SIGMA_CR     22
#define YG_P_EPS_BIV      23  /* regularisation for bivariate Wiener */

/* Y-plane block-stats layout: same 8×8 block, 2-stride as Bayer path,
 * but only one channel (no CFA offset). Block size constants reuse
 * the Bayer defines. */
#ifndef NE_BLOCK_SZ
#define NE_BLOCK_SZ 8
#endif
#ifndef NE_BLOCK_STEP
#define NE_BLOCK_STEP 2
#endif
#ifndef NE_BSEARCH_ITER
#define NE_BSEARCH_ITER 14
#endif


/* ================================================================
 * Utility: sRGB <-> linear conversion (per-pixel)
 * ================================================================ */
static inline float srgb_to_lin(float c)
{
  return (c <= SRGB_THRS) ? (c / SRGB_PHI)
                          : pow((c + SRGB_A) / SRGB_ONEP, SRGB_GAMMA);
}
static inline float lin_to_srgb(float c)
{
  return (c <= SRGB_THRS_L) ? (SRGB_PHI * c)
                            : (SRGB_ONEP * pow(c, 1.0f / SRGB_GAMMA) - SRGB_A);
}


/* ================================================================
 * YG1: sRGB → linear YCbCr (fused, per-pixel)
 *
 * Dispatch: 1D, global = npix = W×H.
 * Input: 3-ch interleaved sRGB float32  (rgb[0..3*npix-1])
 * Output: 3 separate float32 planes (Y_lin, Cb_lin, Cr_lin), each length npix.
 *
 * Fused sRGB EOTF → linear RGB → BT.709 YCbCr. Saves 1 global RT.
 *
 * JP: sRGB 3ch 入力を per-pixel 1 kernel で linear YCbCr 3 平面に展開。
 *     global memory を 1 回だけ round-trip → register 内で完結。
 * ================================================================ */
kernel void galosh_yuv_srgb_to_linear_ycbcr(
    global const float *restrict srgb_rgb,     /* length 3*npix, interleaved */
    global float       *restrict Y_out,        /* length npix */
    global float       *restrict Cb_out,       /* length npix */
    global float       *restrict Cr_out,       /* length npix */
    const int npix)
{
  const int i = get_global_id(0);
  if(i >= npix) return;

  const int base = 3 * i;
  const float R_s = srgb_rgb[base + 0];
  const float G_s = srgb_rgb[base + 1];
  const float B_s = srgb_rgb[base + 2];

  /* sRGB EOTF inverse */
  const float R = srgb_to_lin(R_s);
  const float G = srgb_to_lin(G_s);
  const float B = srgb_to_lin(B_s);

  /* BT.709 linear YCbCr (centred Cb/Cr, NOT +0.5) */
  const float Y  = BT709_KR * R + BT709_KG * G + BT709_KB * B;
  const float Cb = (B - Y) * (1.0f / BT709_CB_DEN);
  const float Cr = (R - Y) * (1.0f / BT709_CR_DEN);

  Y_out[i]  = Y;
  Cb_out[i] = Cb;
  Cr_out[i] = Cr;
}


/* ================================================================
 * YG11: linear YCbCr → sRGB (fused, per-pixel)
 *
 * Dispatch: 1D, global = npix.
 * Inverse of YG1. Clips output to [0,1] after OETF.
 * ================================================================ */
kernel void galosh_yuv_ycbcr_to_srgb(
    global const float *restrict Y_in,
    global const float *restrict Cb_in,
    global const float *restrict Cr_in,
    global float       *restrict srgb_rgb_out,
    const int npix)
{
  const int i = get_global_id(0);
  if(i >= npix) return;

  const float Y  = Y_in[i];
  const float Cb = Cb_in[i];
  const float Cr = Cr_in[i];

  /* YCbCr → linear RGB (BT.709 inverse) */
  const float R = Y + BT709_CR_DEN * Cr;
  const float G = Y - (BT709_KR * BT709_CR_DEN / BT709_KG) * Cr
                    - (BT709_KB * BT709_CB_DEN / BT709_KG) * Cb;
  const float B = Y + BT709_CB_DEN * Cb;

  /* Clip linear to [0,1] to keep OETF well-defined */
  const float Rc = fmax(fmin(R, 1.0f), 0.0f);
  const float Gc = fmax(fmin(G, 1.0f), 0.0f);
  const float Bc = fmax(fmin(B, 1.0f), 0.0f);

  /* sRGB OETF */
  const int base = 3 * i;
  srgb_rgb_out[base + 0] = lin_to_srgb(Rc);
  srgb_rgb_out[base + 1] = lin_to_srgb(Gc);
  srgb_rgb_out[base + 2] = lin_to_srgb(Bc);
}


/* ================================================================
 * YG2: Y-plane block statistics (for Foi+Alenius blind estimation)
 *
 * Dispatch: 1D (n_total = n_bx * n_by), local = 64.
 * Same algorithm as galosh_noise_block_stats for Bayer but operates
 * on a single full-resolution plane (no CFA offset).
 *
 * Writes blk_mean[] and blk_var[] arrays (length n_total) that feed
 * galosh_noise_estimate.
 *
 * JP: Y 平面用の block statistics kernel。Bayer 版と数式は同一だが
 *     CFA オフセットなし。64×64 pixel 領域ごとに mean と
 *     sigma_lap² を計算し、その後の Foi 回帰で α/σ² を推定。
 * ================================================================ */
kernel void galosh_yuv_noise_block_stats_Y(
    global const float *restrict plane,
    global float       *restrict blk_mean,
    global float       *restrict blk_var,
    const int width,
    const int height,
    const int n_bx,
    const int n_by)
{
  const int gid = get_global_id(0);
  const int total = n_bx * n_by;
  if(gid >= total) return;

  const int by = gid / n_bx;
  const int bx = gid % n_bx;

  /* 64×64 pixel region (NE_BLOCK_SZ * NE_BLOCK_STEP = 8 * 2 = 16 in
   * CFA space, but here in full-res Y we double the foot-print to
   * keep the same absolute area: step 2 × block 8 = 16 pixels). */
  const int y0 = by * (NE_BLOCK_SZ * NE_BLOCK_STEP);
  const int x0 = bx * (NE_BLOCK_SZ * NE_BLOCK_STEP);

  /* Mean over 8×8 sub-block at upper-left */
  float sum = 0.0f;
  for(int y = y0; y < y0 + NE_BLOCK_SZ; y++)
    for(int x = x0; x < x0 + NE_BLOCK_SZ; x++)
      sum += plane[y * width + x];
  const float bm = sum * (1.0f / (float)(NE_BLOCK_SZ * NE_BLOCK_SZ));

  /* Binary-search median of |Laplacian| (horizontal + vertical). */
  float lap_min = 1e30f, lap_max = 0.0f;
  int nl = 0;

  for(int y = y0; y < y0 + NE_BLOCK_SZ; y++)
    for(int x = x0; x < x0 + NE_BLOCK_SZ - 2; x++)
    {
      const float v0 = plane[y * width + x];
      const float v1 = plane[y * width + x + 1];
      const float v2 = plane[y * width + x + 2];
      const float L = fabs(v0 - 2.0f * v1 + v2);
      lap_min = fmin(lap_min, L);
      lap_max = fmax(lap_max, L);
      nl++;
    }
  for(int y = y0; y < y0 + NE_BLOCK_SZ - 2; y++)
    for(int x = x0; x < x0 + NE_BLOCK_SZ; x++)
    {
      const float v0 = plane[y * width + x];
      const float v1 = plane[(y + 1) * width + x];
      const float v2 = plane[(y + 2) * width + x];
      const float L = fabs(v0 - 2.0f * v1 + v2);
      lap_min = fmin(lap_min, L);
      lap_max = fmax(lap_max, L);
      nl++;
    }

  if(nl <= 10) { blk_mean[gid] = bm; blk_var[gid] = 1e10f; return; }

  const int med_idx = nl >> 1;
  float lo = lap_min, hi = lap_max;
  for(int iter = 0; iter < NE_BSEARCH_ITER; iter++)
  {
    const float mid = (lo + hi) * 0.5f;
    int cnt = 0;
    for(int y = y0; y < y0 + NE_BLOCK_SZ; y++)
      for(int x = x0; x < x0 + NE_BLOCK_SZ - 2; x++)
      {
        const float v0 = plane[y * width + x];
        const float v1 = plane[y * width + x + 1];
        const float v2 = plane[y * width + x + 2];
        if(fabs(v0 - 2.0f * v1 + v2) <= mid) cnt++;
      }
    for(int y = y0; y < y0 + NE_BLOCK_SZ - 2; y++)
      for(int x = x0; x < x0 + NE_BLOCK_SZ; x++)
      {
        const float v0 = plane[y * width + x];
        const float v1 = plane[(y + 1) * width + x];
        const float v2 = plane[(y + 2) * width + x];
        if(fabs(v0 - 2.0f * v1 + v2) <= mid) cnt++;
      }
    if(cnt <= med_idx) lo = mid; else hi = mid;
  }
  const float med = (lo + hi) * 0.5f;
  const float sigma_lap = med / 0.6745f;
  const float sigma_lap_sq = sigma_lap * sigma_lap;

  blk_mean[gid] = bm;
  blk_var[gid]  = sigma_lap_sq / 6.0f;
}


/* ================================================================
 * YG3: Y-plane dark-sample histogram
 *
 * Dispatch: 1D, global = n_samples = ceil(H/3)*ceil(W/3).
 * Each WI samples one pixel (stride-3 in x and y), atomic_inc into
 * 1024-bin dark histogram. Same role as galosh_noise_dark_samp_hist
 * but single-plane.
 * ================================================================ */
kernel void galosh_yuv_noise_dark_samp_hist_Y(
    global const float *restrict plane,
    global int         *restrict dark_hist,   /* 1024 bins, bin_scale=1024 over [0,1] */
    const int width,
    const int height,
    const int samp_per_row,
    const int n_samples)
{
  const int gid = get_global_id(0);
  if(gid >= n_samples) return;
  const int sy = gid / samp_per_row;
  const int sx = gid % samp_per_row;
  const int y = sy * 3;
  const int x = sx * 3;
  if(x >= width || y >= height) return;
  const float v = plane[y * width + x];
  const int bin = clamp((int)(v * 1024.0f), 0, 1023);
  atomic_inc(&dark_hist[bin]);
}


/* ================================================================
 * YG4: Y-plane dark-Laplacian histogram (refines σ² on truly dark
 * pixels identified by dark_max threshold from params[YG_P_DARK_MAX]).
 *
 * Dispatch: 1D (H*(W-2) + (H-2)*W positions).
 * ================================================================ */
kernel void galosh_yuv_noise_dark_lap_hist_Y(
    global const float *restrict plane,
    global int         *restrict lap_hist,   /* 2048 bins */
    global const float *restrict params,
    const int width,
    const int height,
    const int pos_per_row_h,
    const int pos_per_ch)
{
  const int gid = get_global_id(0);
  if(gid >= pos_per_ch) return;
  const float dark_max = params[YG_P_DARK_MAX];

  int x, y;
  int is_h;
  if(gid < pos_per_row_h)
  {
    /* horizontal Laplacian: (v0 - 2*v1 + v2) along x */
    is_h = 1;
    y = gid / (width - 2);
    x = gid % (width - 2);
  }
  else
  {
    /* vertical Laplacian along y */
    is_h = 0;
    const int g2 = gid - pos_per_row_h;
    y = g2 / width;
    x = g2 % width;
  }

  float v0, v1, v2;
  if(is_h)
  {
    v0 = plane[y * width + x];
    v1 = plane[y * width + x + 1];
    v2 = plane[y * width + x + 2];
  }
  else
  {
    v0 = plane[y * width + x];
    v1 = plane[(y + 1) * width + x];
    v2 = plane[(y + 2) * width + x];
  }

  /* Only use dark pixels (center value below dark_max) */
  if(v1 > dark_max) return;
  const float L = fabs(v0 - 2.0f * v1 + v2);
  const int bin = clamp((int)(L * 4096.0f), 0, 2047);
  atomic_inc(&lap_hist[bin]);
}


/* ================================================================
 * YG5: Dark sigma_sq finalise (for Y plane).
 *
 * Dispatch: 1 WI. Scans lap_hist (2048 bins), picks median, converts
 * to σ²_Y (same formula as galosh_noise_dark_finalize). Writes back
 * into params[YG_P_SIGMA_SQ_Y].
 * ================================================================ */
kernel void galosh_yuv_noise_dark_finalize_Y(
    global const int   *restrict lap_hist,
    global float       *restrict params)
{
  if(get_global_id(0) != 0) return;

  int total = 0;
  for(int i = 0; i < 2048; i++) total += lap_hist[i];
  if(total < 100) return;  /* too few dark pixels — keep bright estimate */

  const int med_idx = total / 2;
  int cum = 0, mb = 0;
  for(int i = 0; i < 2048; i++)
  {
    cum += lap_hist[i];
    if(cum >= med_idx) { mb = i; break; }
  }
  const float med = ((float)mb + 0.5f) / 4096.0f;
  const float sigma_lap = med / 0.6745f;
  const float sigma_sq_dark = (sigma_lap * sigma_lap) / 6.0f;

  /* Take the smaller of bright-side fit σ² and dark-only σ² (dark is
   * the more reliable since the shot-noise contribution vanishes). */
  const float sy2_cur = params[YG_P_SIGMA_SQ_Y];
  params[YG_P_SIGMA_SQ_Y] = fmin(sy2_cur, sigma_sq_dark);
}


/* ================================================================
 * YG6: Chroma noise-parameter derivation (closed-form).
 *
 * Dispatch: 1 WI.
 * Reads (α_Y, σ²_Y) from params, writes analytically-derived chroma
 * parameters (α_Cb, σ²_Cb, α_Cr, σ²_Cr) into the extended slots.
 *
 * See top of file for the derivation (BT.709 linear combination of
 * independent R/G/B Poisson-Gauss noise).
 * ================================================================ */
kernel void galosh_yuv_chroma_params_derive(
    global float *restrict params)
{
  if(get_global_id(0) != 0) return;
  const float a_Y  = params[YG_P_ALPHA_Y];
  const float s2_Y = params[YG_P_SIGMA_SQ_Y];
  params[YG_P_ALPHA_CB]    = CHROMA_RATIO_CB * a_Y;
  params[YG_P_SIGMA_SQ_CB] = CHROMA_RATIO_CB * s2_Y;
  params[YG_P_ALPHA_CR]    = CHROMA_RATIO_CR * a_Y;
  params[YG_P_SIGMA_SQ_CR] = CHROMA_RATIO_CR * s2_Y;
}


/* ================================================================
 * YG7: Y-plane forward GAT
 *
 * Dispatch: 1D, global = npix.
 * Applies generalised Anscombe transform:
 *     Y_stab = (2/α) · sqrt(α·Y + 0.375·α² + σ²_Y)
 * Output variance ≈ 1.  Written as float (precision preserved);
 * a separate float→half conversion kernel feeds galosh_fused_pass12.
 *
 * JP: Y 平面 forward GAT。出力 variance ≈ 1。結果は float で返し、
 *     次段で half に落とす (fused_pass12 は half 入出力)。
 * ================================================================ */
kernel void galosh_yuv_gat_forward_Y(
    global const float *restrict Y_lin,
    global float       *restrict Y_stab,
    global const float *restrict params,
    const int npix)
{
  const int i = get_global_id(0);
  if(i >= npix) return;
  const float a  = params[YG_P_ALPHA_Y];
  const float s2 = params[YG_P_SIGMA_SQ_Y];
  const float c  = 0.375f * a * a + s2;
  const float inv_a = 2.0f / fmax(a, 1e-12f);
  Y_stab[i] = inv_a * sqrt(fmax(a * Y_lin[i] + c, 0.0f));
}


/* ================================================================
 * YG8: Chroma forward linear VST
 *
 * Dispatch: 1D, global = npix.
 * Computes `C_stab = C / sqrt(α_C · Y + σ²_C)` per pixel.
 * Simultaneously writes the scale plane into `scale_out` so the
 * inverse kernel (YG10) can re-use it without recomputing sqrt.
 *
 * `chan = 0` → Cb uses (α_Cb, σ²_Cb); `chan = 1` → Cr.
 *
 * JP: Cb/Cr 共通の線形 VST forward。scale (√(αY+σ²)) を別 buffer に
 *     書き出し、inverse で再計算を省く。
 * ================================================================ */
kernel void galosh_yuv_vst_forward_C(
    global const float *restrict C_in,
    global const float *restrict Y_lin,
    global float       *restrict C_stab,
    global float       *restrict scale_out,
    global const float *restrict params,
    const int chan,        /* 0 = Cb, 1 = Cr */
    const int npix)
{
  const int i = get_global_id(0);
  if(i >= npix) return;
  const float a  = (chan == 0) ? params[YG_P_ALPHA_CB]
                               : params[YG_P_ALPHA_CR];
  const float s2 = (chan == 0) ? params[YG_P_SIGMA_SQ_CB]
                               : params[YG_P_SIGMA_SQ_CR];
  const float scale = sqrt(fmax(a * Y_lin[i] + s2, 1e-20f));
  C_stab[i]    = C_in[i] / scale;
  scale_out[i] = scale;
}


/* ================================================================
 * YG9: Chroma inverse linear VST
 *
 * Dispatch: 1D, global = npix.
 * Applies `C_out = C_stab_denoised × scale(precomputed)` to restore
 * the original variance scale after LOSH shrinkage.
 * ================================================================ */
kernel void galosh_yuv_vst_inverse_C(
    global const float *restrict C_stab,
    global const float *restrict scale_in,
    global float       *restrict C_out,
    const int npix)
{
  const int i = get_global_id(0);
  if(i >= npix) return;
  C_out[i] = C_stab[i] * scale_in[i];
}


/* ================================================================
 * YG10: Makitalo-Foi exact unbiased inverse GAT
 *
 * Dispatch: 1D, global = npix.
 * Uses the 4096-entry LUT built by galosh_build_inv_lut / lut_finalize
 * (both of which only depend on α, σ² — so we re-run those on Y
 * parameters from the yuv_gat driver).
 *
 * LUT layout:
 *   lut_d[i]      expected GAT(Poisson(x/α)·α + N(0,σ²)) for x = i/4095
 *   lut_x[i]      clean x values
 *   lut_params[0] d_min (lowest d value in LUT)
 *   lut_params[1] d_max
 *   lut_params[2] y_break = -0.375·α
 *   lut_params[3] t_break = 2·σ/α
 *   lut_params[4] sigma_raw = σ
 *
 * Given denoised (stabilised) d value, interpolate lut_x(d).
 *
 * JP: Makitalo-Foi 逆 GAT. LUT は既存 build_inv_lut/lut_finalize を
 *     Y-plane 用に再利用して Y の α/σ² で再構築済み。
 * ================================================================ */
kernel void galosh_yuv_makitalo_inverse_Y(
    global const float *restrict d_stab,    /* denoised stabilised Y */
    global float       *restrict x_out,     /* linear Y output */
    global const float *restrict lut_d,
    global const float *restrict lut_x,
    global const float *restrict lut_params,
    const int npix)
{
  const int i = get_global_id(0);
  if(i >= npix) return;
  const float d = d_stab[i];
  const float d_min = lut_params[0];
  const float d_max = lut_params[1];

  /* Binary search within lut_d[] for the interval containing d. */
  float result;
  if(d <= d_min) {
    result = lut_x[0];
  } else if(d >= d_max) {
    result = lut_x[4095];
  } else {
    int lo = 0, hi = 4095;
    while(hi - lo > 1)
    {
      const int mid = (lo + hi) >> 1;
      if(lut_d[mid] <= d) lo = mid; else hi = mid;
    }
    const float d0 = lut_d[lo], d1 = lut_d[hi];
    const float t  = (d - d0) / fmax(d1 - d0, 1e-20f);
    result = lut_x[lo] * (1.0f - t) + lut_x[hi] * t;
  }
  x_out[i] = fmax(result, 0.0f);
}


/* ================================================================
 * YG12: Float ↔ Half helper kernels (required because
 * galosh_fused_pass12 operates on half buffers, but our colour-
 * space / VST kernels work in float).
 *
 * Dispatch: 1D, global = npix.
 * ================================================================ */
kernel void galosh_yuv_float_to_half(
    global const float *restrict src,
    global half        *restrict dst,
    const int npix)
{
  const int i = get_global_id(0);
  if(i >= npix) return;
  dst[i] = (half)src[i];
}

kernel void galosh_yuv_half_to_float(
    global const half  *restrict src,
    global float       *restrict dst,
    const int npix)
{
  const int i = get_global_id(0);
  if(i >= npix) return;
  dst[i] = (float)src[i];
}


/* ================================================================
 * YG13: Bivariate Wiener shrinkage on (Cb, Cr)
 *
 * Dispatch: 1D, global = npix.
 * Reads stabilised denoised Cb/Cr (variance ≈ 1 each) and applies
 * a simple local 2×2 Wiener gain using a 5×5 neighbourhood local
 * signal covariance estimate. Noise covariance is identity (since
 * both planes are stabilised to unit variance and the cross-
 * correlation of chroma shot-noise ≈ 0 in the BT.709 transform).
 *
 * For each pixel:
 *   Σ_s_hat = local 2×2 sample covariance over 5×5 window  (signal+noise)
 *   Σ_n     = I₂ + ε·I₂                                     (noise)
 *   Σ_x_hat = max(Σ_s - Σ_n, 0)                             (clean signal)
 *   gain    = Σ_x_hat · (Σ_x_hat + Σ_n)⁻¹
 *   [Cb', Cr']ᵀ = gain · [Cb, Cr]ᵀ
 *
 * Solved in closed form for 2×2 (no linalg needed).
 *
 * Note: runs AFTER independent LOSH shrinkage, as a final coupling
 * step. This is cheaper than doing it inside the WHT blocks and
 * operates directly in the stabilised domain.
 *
 * JP: Cb/Cr の 2×2 joint Wiener 縮小。LOSH 独立後の coupling 段。
 *     5×5 ウィンドウで局所共分散を推定し、2×2 の閉じ形で gain を解く。
 * ================================================================ */
kernel void galosh_yuv_bivariate_wiener_cbcr(
    global const float *restrict Cb_in,
    global const float *restrict Cr_in,
    global float       *restrict Cb_out,
    global float       *restrict Cr_out,
    global const float *restrict params,
    const int width,
    const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;

  /* 5×5 local sample covariance, with reflect-border handling. */
  const int r = 2;
  float s_bb = 0.0f, s_rr = 0.0f, s_br = 0.0f;
  int n = 0;
  for(int dy = -r; dy <= r; dy++)
    for(int dx = -r; dx <= r; dx++)
    {
      int xi = x + dx, yi = y + dy;
      if(xi < 0) xi = -xi;
      if(yi < 0) yi = -yi;
      if(xi >= width) xi = 2 * width - xi - 2;
      if(yi >= height) yi = 2 * height - yi - 2;
      if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
      const float b = Cb_in[yi * width + xi];
      const float rr = Cr_in[yi * width + xi];
      s_bb += b * b;
      s_rr += rr * rr;
      s_br += b * rr;
      n++;
    }
  const float invn = 1.0f / (float)n;
  s_bb *= invn;  s_rr *= invn;  s_br *= invn;

  /* Signal covariance = max(Σ_s - Σ_n, 0) with Σ_n ≈ I₂. */
  const float eps = params[YG_P_EPS_BIV];
  const float sn  = 1.0f + eps;
  float x_bb = fmax(s_bb - sn, 0.0f);
  float x_rr = fmax(s_rr - sn, 0.0f);
  float x_br = s_br;   /* noise cross ≈ 0 */
  /* Keep Σ_x SPD: cross clipped by Cauchy-Schwarz */
  const float bound = sqrt(x_bb * x_rr);
  if(x_br >  bound) x_br =  bound;
  if(x_br < -bound) x_br = -bound;

  /* Gain = Σ_x (Σ_x + Σ_n)⁻¹.  A = Σ_x + Σ_n = [[a, c], [c, d]].
   * A⁻¹ = 1/det · [[d, -c], [-c, a]]. G = Σ_x · A⁻¹. */
  const float a = x_bb + sn, d = x_rr + sn, cc = x_br;
  const float det = fmax(a * d - cc * cc, 1e-20f);
  const float inv_det = 1.0f / det;
  const float A_inv_00 =  d * inv_det;
  const float A_inv_11 =  a * inv_det;
  const float A_inv_01 = -cc * inv_det;

  const float G_00 = x_bb * A_inv_00 + x_br * A_inv_01;
  const float G_01 = x_bb * A_inv_01 + x_br * A_inv_11;
  const float G_10 = x_br * A_inv_00 + x_rr * A_inv_01;
  const float G_11 = x_br * A_inv_01 + x_rr * A_inv_11;

  const float b_in = Cb_in[idx], r_in = Cr_in[idx];
  Cb_out[idx] = G_00 * b_in + G_01 * r_in;
  Cr_out[idx] = G_10 * b_in + G_11 * r_in;
}


/* End of galosh_yuv_gat.cl */
