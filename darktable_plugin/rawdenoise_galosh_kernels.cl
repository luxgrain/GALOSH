/*
 * GALOSH OpenCL kernels — Generalized Anscombe LOcal SHrinkage
 *
 * GPU implementation of the GALOSH denoiser for darktable rawdenoise IOP.
 * All kernels are local (no block matching) → perfect GPU parallelism.
 *
 * Architecture:
 *   Phase 1: GAT forward + Bayer channel extraction (CPU pre-pass for noise est.)
 *   Phase 2: Sigma normalization + dark anchor + WHT decomposition
 *   Phase 3: BayesShrink (Pass1) + Wiener (Pass2) via GATHER approach
 *   Phase 3b: Luma-guided filter for chroma (σ_C > 1.0)
 *   Phase 4: Full-res L computation + Wiener
 *   Phase 5: Inverse WHT + inverse GAT + output
 *
 * Core design: GATHER approach for overlap-add
 *   Each output pixel gathers contributions from all overlapping blocks
 *   (up to 16 blocks for stride=2, 8×8 blocks). Zero synchronization,
 *   no float atomics, fully parallel.
 *
 * Copyright (c) 2026 luxgrain. All rights reserved.
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "common.h"


/* ================================================================
 * Constants
 * ================================================================ */

#define GALOSH_BS    8    /* block size */
#define GALOSH_BP   64    /* block pixels = BS*BS */
#define GALOSH_HB    4    /* half block = BS/2 */
#define GALOSH_HP   16    /* half block pixels = HB*HB */
#define GALOSH_WIENER_FLOOR  0.125f   /* 1/BS */


/* ================================================================
 * Kaiser window (precomputed 1D, beta=2.0, N=8)
 * ================================================================ */

constant float galosh_kaiser_1d[8] = {
  0.34012f, 0.59885f, 0.84123f, 0.97659f,
  0.97659f, 0.84123f, 0.59885f, 0.34012f
};


/* ================================================================
 * WHT 8-point in-place (sequency order) — add/subtract only
 * ================================================================ */

inline void wht8(float *x)
{
  float a0 = x[0]+x[1], a1 = x[0]-x[1];
  float a2 = x[2]+x[3], a3 = x[2]-x[3];
  float a4 = x[4]+x[5], a5 = x[4]-x[5];
  float a6 = x[6]+x[7], a7 = x[6]-x[7];
  float b0 = a0+a2, b1 = a1+a3;
  float b2 = a0-a2, b3 = a1-a3;
  float b4 = a4+a6, b5 = a5+a7;
  float b6 = a4-a6, b7 = a5-a7;
  x[0] = b0+b4; x[1] = b1+b5; x[2] = b2+b6; x[3] = b3+b7;
  x[4] = b0-b4; x[5] = b1-b5; x[6] = b2-b6; x[7] = b3-b7;
}

/* 2D separable 8×8 WHT. normalize: 0=forward, 1=inverse (/64) */
inline void wht2d_8x8(float *block, const int normalize)
{
  for(int r = 0; r < 8; r++)
    wht8(block + r * 8);
  for(int c = 0; c < 8; c++)
  {
    float col[8];
    for(int r = 0; r < 8; r++) col[r] = block[r * 8 + c];
    wht8(col);
    for(int r = 0; r < 8; r++) block[r * 8 + c] = col[r];
  }
  if(normalize)
  {
    const float inv = 1.0f / 64.0f;
    for(int i = 0; i < 64; i++) block[i] *= inv;
  }
}


/* (atomic_add_float removed — phased approach eliminates all atomics) */


/* ================================================================
 * Kernel 1: GAT forward + Bayer channel extraction
 *
 * Full-res input → 4 half-res GAT-transformed channels
 * Each workitem processes one full-res pixel.
 * ================================================================ */

kernel void galosh_gat_forward_extract(
    read_only image2d_t in,
    global float *restrict ch0,
    global float *restrict ch1,
    global float *restrict ch2,
    global float *restrict ch3,
    const int width,
    const int height,
    const float alpha,
    const float sigma_sq)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const float val = read_imagef(in, sampleri, (int2)(x, y)).x;

  /* GAT forward (piecewise C1) */
  const float x_safe = isnan(val) ? 0.0f : val;
  const float y_break = -0.375f * alpha;
  float gat;
  if(x_safe >= y_break)
    gat = (2.0f / alpha) * dtcl_sqrt(alpha * x_safe + 0.375f * alpha * alpha + sigma_sq);
  else
  {
    const float sigma_raw = dtcl_sqrt(max(sigma_sq, 1e-20f));
    gat = 2.0f * sigma_raw / alpha + (x_safe - y_break) / sigma_raw;
  }

  /* Write to appropriate half-res channel based on Bayer position */
  const int hy = y / 2, hx = x / 2;
  const int halfwidth = (width + 1) / 2;
  const int ch_idx = (y & 1) * 2 + (x & 1);  /* 0=TL, 1=TR, 2=BL, 3=BR */
  const size_t hpos = (size_t)hy * halfwidth + hx;

  /* Note: Bayer layout uses row_offset = c & 1, col_offset = (c >> 1) & 1
   * ch0: row_offset=0, col_offset=0 → even row, even col → (y&1)==0, (x&1)==0
   * ch1: row_offset=1, col_offset=0 → odd row, even col  → (y&1)==1, (x&1)==0
   * ch2: row_offset=0, col_offset=1 → even row, odd col  → (y&1)==0, (x&1)==1
   * ch3: row_offset=1, col_offset=1 → odd row, odd col   → (y&1)==1, (x&1)==1
   * So ch_idx mapping: TL(0,0)→ch0, BL(1,0)→ch1, TR(0,1)→ch2, BR(1,1)→ch3 */
  if(ch_idx == 0)      ch0[hpos] = gat;
  else if(ch_idx == 2)  ch1[hpos] = gat;  /* odd row, even col */
  else if(ch_idx == 1)  ch2[hpos] = gat;  /* even row, odd col */
  else                  ch3[hpos] = gat;  /* odd row, odd col */
}


/* ================================================================
 * Kernel 2: Sigma normalization (per-pixel multiply)
 * ================================================================ */

kernel void galosh_normalize_sigma(
    global float *restrict data,
    const int width,
    const int height,
    const float inv_sigma)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  data[(size_t)y * width + x] *= inv_sigma;
}


/* ================================================================
 * Kernel 3: Dark reference subtraction
 * ================================================================ */

kernel void galosh_subtract_dark_ref(
    global float *restrict data,
    const int width,
    const int height,
    const float dark_ref)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  data[(size_t)y * width + x] -= dark_ref;
}


/* ================================================================
 * Kernel 4: 2×2 WHT decomposition (4 RGGB channels → L/C1/C2/C3)
 * ================================================================ */

kernel void galosh_wht_decompose(
    global const float *restrict ch0,
    global const float *restrict ch1,
    global const float *restrict ch2,
    global const float *restrict ch3,
    global float *restrict luma,
    global float *restrict chroma1,
    global float *restrict chroma2,
    global float *restrict chroma3,
    const int halfwidth,
    const int halfheight)
{
  const int hx = get_global_id(0);
  const int hy = get_global_id(1);
  if(hx >= halfwidth || hy >= halfheight) return;

  const size_t i = (size_t)hy * halfwidth + hx;
  const float a = ch0[i], b = ch1[i], c = ch2[i], d = ch3[i];

  luma[i]    = (a + b + c + d) * 0.5f;
  chroma1[i] = (a - b + c - d) * 0.5f;
  chroma2[i] = (a + b - c - d) * 0.5f;
  chroma3[i] = (a - b - c + d) * 0.5f;
}


/* ================================================================
 * Kernel 5a: Pass1 — BayesShrink (PHASED scatter, zero-atomic)
 *
 * Phase-based scatter: blocks are grouped by (bx % phase_mod, by % phase_mod).
 * Within each phase, no two blocks write to the same output pixel,
 * so plain += is safe without atomics. phase_mod = BS / stride.
 *
 * Dispatch grid: (phase_blocks_x, phase_blocks_y) per phase.
 * Total phases = phase_mod² = 16 for stride=2, BS=8.
 *
 * GPU-optimal: 1 WHT pair/workitem, 64 float private memory,
 * zero atomics, zero synchronization.
 * ================================================================ */

kernel void galosh_pass1_scatter(
    global const float *restrict input,
    global float *numer,
    global float *denom,
    const int width,
    const int height,
    const float sigma_strength,
    const int stride,
    const int phase_x,
    const int phase_y,
    const int phase_mod)
{
  const int pbx = get_global_id(0);  /* phase-local block index */
  const int pby = get_global_id(1);
  const int bx = pbx * phase_mod + phase_x;  /* actual block index */
  const int by = pby * phase_mod + phase_y;
  const int ref_c = bx * stride;
  const int ref_r = by * stride;
  if(ref_c + GALOSH_BS > width || ref_r + GALOSH_BS > height) return;

  const float sigma_sq = sigma_strength * sigma_strength;
  const float lambda_max = sigma_strength * dtcl_sqrt(2.0f * dtcl_log((float)GALOSH_BP));

  /* Load 8×8 block into private memory */
  float block[GALOSH_BP];
  for(int dy = 0; dy < GALOSH_BS; dy++)
    for(int dx = 0; dx < GALOSH_BS; dx++)
      block[dy * GALOSH_BS + dx] = input[(size_t)(ref_r + dy) * width + (ref_c + dx)];

  /* Forward WHT */
  wht2d_8x8(block, 0);

  /* BayesShrink threshold */
  float sum_sq = 0.0f;
  for(int i = 1; i < GALOSH_BP; i++)
    sum_sq += block[i] * block[i];
  const float sigma_y_sq = sum_sq / ((float)(GALOSH_BP - 1) * (float)GALOSH_BP);
  const float sigma_x_sq = max(sigma_y_sq - sigma_sq, 0.0f);

  float lambda;
  if(sigma_x_sq < 1e-10f)
    lambda = 1e30f;
  else
  {
    lambda = (sigma_sq / dtcl_sqrt(sigma_x_sq)) * dtcl_sqrt((float)GALOSH_BP);
    const float lambda_max_unorm = lambda_max * dtcl_sqrt((float)GALOSH_BP);
    if(lambda > lambda_max_unorm) lambda = lambda_max_unorm;
  }

  /* Hard thresholding */
  int n_nonzero = 1;
  for(int i = 1; i < GALOSH_BP; i++)
  {
    if(fabs(block[i]) < lambda)
      block[i] = 0.0f;
    else
      n_nonzero++;
  }

  /* Inverse WHT */
  wht2d_8x8(block, 1);

  /* Write weighted results — no atomic needed (phase guarantees no overlap) */
  const float weight = 1.0f / (float)n_nonzero;
  for(int dy = 0; dy < GALOSH_BS; dy++)
  {
    for(int dx = 0; dx < GALOSH_BS; dx++)
    {
      const float kw = galosh_kaiser_1d[dy] * galosh_kaiser_1d[dx];
      const float wkw = weight * kw;
      const size_t pos = (size_t)(ref_r + dy) * width + (ref_c + dx);
      numer[pos] += wkw * block[dy * GALOSH_BS + dx];
      denom[pos] += wkw;
    }
  }
}


/* ================================================================
 * Kernel 6a: Pass2 — Wiener shrinkage (SCATTER approach)
 *
 * Phased scatter: blocks grouped by (bx % phase_mod, by % phase_mod).
 * Within each phase, output regions don't overlap → plain += (no atomics).
 * phase_mod = BS / stride = 8/2 = 4 → 16 phases.
 *
 * Dispatch grid: (phase_blocks_x, phase_blocks_y) per phase.
 * Total phases = phase_mod² = 16 for stride=2, BS=8.
 * ================================================================ */

kernel void galosh_pass2_scatter(
    global const float *restrict noisy,
    global const float *restrict pilot,
    global float *numer,
    global float *denom,
    const int width,
    const int height,
    const float sigma_strength,
    const float wiener_floor,
    const int stride,
    const int phase_x,
    const int phase_y,
    const int phase_mod)
{
  const int pbx = get_global_id(0);  /* phase-local block index */
  const int pby = get_global_id(1);
  const int bx = pbx * phase_mod + phase_x;  /* actual block index */
  const int by = pby * phase_mod + phase_y;
  const int ref_c = bx * stride;
  const int ref_r = by * stride;
  if(ref_c + GALOSH_BS > width || ref_r + GALOSH_BS > height) return;

  const float sigma_sq_unorm = sigma_strength * sigma_strength * (float)GALOSH_BP;

  float blk_noisy[GALOSH_BP];
  float blk_pilot[GALOSH_BP];

  for(int dy = 0; dy < GALOSH_BS; dy++)
    for(int dx = 0; dx < GALOSH_BS; dx++)
    {
      const size_t pos = (size_t)(ref_r + dy) * width + (ref_c + dx);
      blk_noisy[dy * GALOSH_BS + dx] = noisy[pos];
      blk_pilot[dy * GALOSH_BS + dx] = pilot[pos];
    }

  wht2d_8x8(blk_noisy, 0);
  wht2d_8x8(blk_pilot, 0);

  /* Wiener shrinkage */
  float wiener_energy = 0.0f;
  for(int i = 0; i < GALOSH_BP; i++)
  {
    float w;
    if(i == 0)
      w = 1.0f;
    else
    {
      const float s2 = blk_pilot[i] * blk_pilot[i];
      w = s2 / (s2 + sigma_sq_unorm);
      if(w < wiener_floor) w = wiener_floor;
    }
    blk_noisy[i] *= w;
    wiener_energy += w * w;
  }

  wht2d_8x8(blk_noisy, 1);

  /* Scatter */
  const float weight = 1.0f / max(wiener_energy, 1e-6f);
  for(int dy = 0; dy < GALOSH_BS; dy++)
  {
    for(int dx = 0; dx < GALOSH_BS; dx++)
    {
      const float kw = galosh_kaiser_1d[dy] * galosh_kaiser_1d[dx];
      const float wkw = weight * kw;
      const size_t pos = (size_t)(ref_r + dy) * width + (ref_c + dx);
      numer[pos] += wkw * blk_noisy[dy * GALOSH_BS + dx];
      denom[pos] += wkw;
    }
  }
}


/* ================================================================
 * Kernel 5b/6b: Scatter finalize — output = numer/denom
 *
 * Shared by pass1 and pass2. Falls back to input (noisy) when
 * denominator is zero (border pixels outside all blocks).
 * ================================================================ */

kernel void galosh_scatter_finalize(
    global const float *restrict fallback,
    global const float *restrict numer,
    global const float *restrict denom,
    global float *restrict output,
    const int width,
    const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const size_t pos = (size_t)y * width + x;
  const float d = denom[pos];
  output[pos] = (d > 1e-10f) ? numer[pos] / d : fallback[pos];
}


/* ================================================================
 * Zero-fill buffer kernel
 * ================================================================ */

kernel void galosh_zero_buffer(
    global float *restrict buf,
    const int width,
    const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  buf[(size_t)y * width + x] = 0.0f;
}


/* ================================================================
 * Kernel 7a: Box filter mean — horizontal pass
 * One workitem per row.
 * ================================================================ */

kernel void galosh_boxmean_horiz(
    global const float *restrict input,
    global float *restrict output,
    const int width,
    const int height,
    const int radius)
{
  const int y = get_global_id(0);
  if(y >= height) return;

  const int r = radius;
  const float inv_n = 1.0f / (float)(2 * r + 1);

  /* Initialize running sum */
  float sum = 0.0f;
  for(int k = -r; k <= r; k++)
  {
    const int sx = clamp(k, 0, width - 1);
    sum += input[(size_t)y * width + sx];
  }
  output[(size_t)y * width + 0] = sum * inv_n;

  for(int x = 1; x < width; x++)
  {
    const int add_x = min(x + r, width - 1);
    const int rem_x = max(x - r - 1, 0);
    sum += input[(size_t)y * width + add_x];
    sum -= input[(size_t)y * width + rem_x];
    output[(size_t)y * width + x] = sum * inv_n;
  }
}


/* ================================================================
 * Kernel 7b: Box filter mean — vertical pass
 * One workitem per column.
 * ================================================================ */

kernel void galosh_boxmean_vert(
    global const float *restrict input,
    global float *restrict output,
    const int width,
    const int height,
    const int radius)
{
  const int x = get_global_id(0);
  if(x >= width) return;

  const int r = radius;
  const float inv_n = 1.0f / (float)(2 * r + 1);

  float sum = 0.0f;
  for(int k = -r; k <= r; k++)
  {
    const int sy = clamp(k, 0, height - 1);
    sum += input[(size_t)sy * width + x];
  }
  output[x] = sum * inv_n;

  for(int y = 1; y < height; y++)
  {
    const int add_y = min(y + r, height - 1);
    const int rem_y = max(y - r - 1, 0);
    sum += input[(size_t)add_y * width + x];
    sum -= input[(size_t)rem_y * width + x];
    output[(size_t)y * width + x] = sum * inv_n;
  }
}


/* ================================================================
 * Kernel 8: Guided filter — compute covariance products (pointwise)
 * ================================================================ */

kernel void galosh_guided_covar(
    global const float *restrict luma,
    global const float *restrict chroma,
    global float *restrict Ip_out,
    global float *restrict II_out,
    const int width,
    const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const size_t i = (size_t)y * width + x;
  Ip_out[i] = luma[i] * chroma[i];
  II_out[i] = luma[i] * luma[i];
}


/* ================================================================
 * Kernel 9: Guided filter — solve a,b coefficients (pointwise)
 * a = cov(I,p) / (var(I) + eps), b = mean_p - a * mean_I
 * ================================================================ */

kernel void galosh_guided_solve(
    global const float *restrict mean_I,
    global const float *restrict mean_p,
    global const float *restrict mean_Ip,
    global const float *restrict mean_II,
    global float *restrict a_out,
    global float *restrict b_out,
    const int width,
    const int height,
    const float eps)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const size_t i = (size_t)y * width + x;
  const float cov_Ip = mean_Ip[i] - mean_I[i] * mean_p[i];
  const float var_I  = mean_II[i] - mean_I[i] * mean_I[i];
  const float a = cov_Ip / (var_I + eps);
  const float b = mean_p[i] - a * mean_I[i];
  a_out[i] = a;
  b_out[i] = b;
}


/* ================================================================
 * Kernel 10: Guided filter — apply (pointwise)
 * chroma_out = mean_a * I + mean_b
 * ================================================================ */

kernel void galosh_guided_apply(
    global const float *restrict mean_a,
    global const float *restrict mean_b,
    global const float *restrict luma,
    global float *restrict chroma_out,
    const int width,
    const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const size_t i = (size_t)y * width + x;
  chroma_out[i] = mean_a[i] * luma[i] + mean_b[i];
}


/* ================================================================
 * Kernel 11: Compute full-res L from half-res L/C planes
 * (Sliding 2×2 WHT-DC interpolation)
 *
 * Each workitem processes one half-res position → writes 4 full-res pixels.
 * ================================================================ */

kernel void galosh_compute_L_fullres(
    global const float *restrict L,
    global const float *restrict C1,
    global const float *restrict C2,
    global const float *restrict C3,
    global float *restrict L_out,
    const int halfwidth,
    const int halfheight)
{
  const int hx = get_global_id(0);
  const int hy = get_global_id(1);
  if(hx >= halfwidth || hy >= halfheight) return;

  const int fw = 2 * halfwidth;
  const size_t hi = (size_t)hy * halfwidth + hx;
  const int hx1 = min(hx + 1, halfwidth - 1);
  const int hy1 = min(hy + 1, halfheight - 1);
  const size_t hi_r  = (size_t)hy  * halfwidth + hx1;
  const size_t hi_d  = (size_t)hy1 * halfwidth + hx;
  const size_t hi_dr = (size_t)hy1 * halfwidth + hx1;

  const int fr = 2 * hy, fc = 2 * hx;

  /* Block-aligned: L itself */
  L_out[(size_t)fr * fw + fc] = L[hi];

  /* Horizontal sliding */
  L_out[(size_t)fr * fw + fc + 1]
    = ((L[hi] - C2[hi]) + (L[hi_r] + C2[hi_r])) * 0.5f;

  /* Vertical sliding */
  L_out[(size_t)(fr + 1) * fw + fc]
    = ((L[hi] - C1[hi]) + (L[hi_d] + C1[hi_d])) * 0.5f;

  /* Diagonal sliding */
  const float Ls  = L[hi] + L[hi_r] + L[hi_d] + L[hi_dr];
  const float C1s = -C1[hi] - C1[hi_r] + C1[hi_d] + C1[hi_dr];
  const float C2s = -C2[hi] + C2[hi_r] - C2[hi_d] + C2[hi_dr];
  const float C3s =  C3[hi] - C3[hi_r] - C3[hi_d] + C3[hi_dr];
  L_out[(size_t)(fr + 1) * fw + fc + 1]
    = (Ls + C1s + C2s + C3s) * 0.25f;
}


/* ================================================================
 * Kernel 12: Inverse WHT + inverse GAT + output
 *
 * Reads: full-res L_den (block-aligned only), half-res C1/C2/C3
 * Writes: 4 full-res output pixels per workitem
 * Uses inverse GAT LUT via binary search.
 * ================================================================ */

kernel void galosh_reconstruct(
    global const float *restrict L_fr_den,
    global const float *restrict c1_out,
    global const float *restrict c2_out,
    global const float *restrict c3_out,
    write_only image2d_t out,
    global const float *restrict gat_inv_d,
    global const float *restrict gat_inv_x,
    const int halfwidth,
    const int halfheight,
    const int fullwidth,
    const float unified_sigma,
    const float dark_ref0,
    const float dark_ref1,
    const float dark_ref2,
    const float dark_ref3,
    const float gat_inv_d_min,
    const float gat_inv_d_max,
    const float gat_inv_y_break,
    const float gat_inv_sigma_raw,
    const float gat_inv_t_break)
{
  const int hx = get_global_id(0);
  const int hy = get_global_id(1);
  if(hx >= halfwidth || hy >= halfheight) return;

  const size_t hi = (size_t)hy * halfwidth + hx;
  const int fr = 2 * hy, fc = 2 * hx;

  const float c1 = c1_out[hi];
  const float c2 = c2_out[hi];
  const float c3 = c3_out[hi];
  const float L_block = L_fr_den[(size_t)fr * fullwidth + fc];

  /* Inverse WHT + dark_ref restore */
  float vals[4];
  vals[0] = (L_block + c1 + c2 + c3) * 0.5f + dark_ref0;
  vals[1] = (L_block - c1 + c2 - c3) * 0.5f + dark_ref1;
  vals[2] = (L_block + c1 - c2 - c3) * 0.5f + dark_ref2;
  vals[3] = (L_block - c1 - c2 + c3) * 0.5f + dark_ref3;

  /* Sigma denormalization + inverse GAT via LUT binary search */
  const float sg = unified_sigma;
  const int fw = fullwidth;

  for(int ch = 0; ch < 4; ch++)
  {
    const float D = vals[ch] * sg;
    float result;

    if(D <= gat_inv_d_min)
      result = gat_inv_y_break + gat_inv_sigma_raw * (D - gat_inv_t_break);
    else if(D >= gat_inv_d_max)
      result = 1.0f;
    else
    {
      /* Binary search on monotone LUT */
      int lo = 0, hi_idx = 4095;
      while(lo < hi_idx - 1)
      {
        const int mid = (lo + hi_idx) >> 1;
        if(gat_inv_d[mid] <= D) lo = mid;
        else hi_idx = mid;
      }
      const float d0 = gat_inv_d[lo], d1 = gat_inv_d[hi_idx];
      const float t = (D - d0) / max(d1 - d0, 1e-10f);
      result = gat_inv_x[lo] + t * (gat_inv_x[hi_idx] - gat_inv_x[lo]);
    }

    /* Write to correct full-res position based on channel index.
     * ch0: (fr, fc), ch1: (fr+1, fc), ch2: (fr, fc+1), ch3: (fr+1, fc+1) */
    const int oy = fr + (ch >> 1);
    const int ox = fc + (ch & 1);
    write_imagef(out, (int2)(ox, oy), (float4)(result, 0.0f, 0.0f, 0.0f));
  }
}
