/*
 * GALOSH Fused Tile Kernel — FP16 LDS + stride=4 variant
 *
 * Optimizations over galosh_fused.cl:
 *   (1) LDS buffers use half (FP16) → 50% LDS reduction, 2x LDS bandwidth
 *   (2) Default stride=4 → 1/4 blocks, N_PHASES=4 vs 16
 *   (3) Block-level computation stays float (private registers) for precision
 *   (4) Only LDS ↔ register transfers do half↔float conversion
 *
 * LDS budget (TILE_SIZE=48, stride=4, HALO=4):
 *   4 × (48+8)² × 2 = 19,712 bytes = 19.3 KB  (vs 57.6 KB FP32 stride=2)
 *
 * Combined speedup: stride 4x × FP16 ~1.5x = ~6x over FP32 stride=2
 *
 * FP16 LDS + FP32 演算: LDS帯域2倍 + 計算精度維持のハイブリッド方式。
 * WHT は加減算のみなので FP16 でも精度十分だが、BayesShrink の
 * σ推定と Wiener 除算は FP32 で行い暗部精度を確保。
 *
 * Copyright (c) 2026 luxgrain. All rights reserved.
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable


/* ================================================================
 * Compile-time constants (overridden by host -D flags)
 * ================================================================ */

#ifndef GALOSH_BS
#define GALOSH_BS      8
#endif

#ifndef GALOSH_BP
#define GALOSH_BP     64
#endif

#ifndef GALOSH_STRIDE
#define GALOSH_STRIDE  4
#endif

#ifndef TILE_SIZE
#define TILE_SIZE     48
#endif

#define PHASE_MOD  (GALOSH_BS / GALOSH_STRIDE)   /* 2 for stride=4 */
#define N_PHASES   (PHASE_MOD * PHASE_MOD)        /* 4 for stride=4 */
#define HALO       (GALOSH_BS - GALOSH_STRIDE)    /* 4 for stride=4 */
#define TILE_W     (TILE_SIZE + 2 * HALO)         /* 56 for stride=4,tile=48 */
#define TILE_PIXELS (TILE_W * TILE_W)

#define WIENER_FLOOR  0.125f


/* ================================================================
 * Kaiser window (beta=2.0, N=8)
 * ================================================================ */

constant float kaiser_1d[8] = {
  0.34012f, 0.59885f, 0.84123f, 0.97659f,
  0.97659f, 0.84123f, 0.59885f, 0.34012f
};


/* ================================================================
 * WHT 8-point in-place — pure add/sub (FP16-safe)
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


/* ================================================================
 * Fused Tile Kernel: Pass1 (BayesShrink) + Pass2 (Wiener)
 *
 * FP16 LDS variant: local memory uses half precision.
 * Block-level WHT + shrinkage uses float (private registers).
 *
 * LDS layout (all half):
 *   tile_in[TILE_PIXELS]  — input data (persistent across passes)
 *   numer[TILE_PIXELS]    — overlap-add numerator (reused per pass)
 *   denom[TILE_PIXELS]    — overlap-add denominator (reused per pass)
 *   pilot[TILE_PIXELS]    — Pass1 output, read in Pass2
 * ================================================================ */

kernel void galosh_fused_pass12(
    global const float *restrict input,
    global float *restrict output,
    const int width,
    const int height,
    const float sigma_strength)
{
  const int tile_x = get_group_id(0) * TILE_SIZE;
  const int tile_y = get_group_id(1) * TILE_SIZE;
  const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
  const int wg_size = get_local_size(0) * get_local_size(1);

  /* FP16 local memory buffers — half the LDS footprint */
  local half tile_in[TILE_PIXELS];
  local half numer[TILE_PIXELS];
  local half denom[TILE_PIXELS];
  local half pilot[TILE_PIXELS];

  /* ---- Step 1: Load input tile (global FP32 → local FP16) ---- */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    const int lx = i % TILE_W;
    const int ly = i / TILE_W;
    const int gx = tile_x - HALO + lx;
    const int gy = tile_y - HALO + ly;
    float val = (gx >= 0 && gx < width && gy >= 0 && gy < height)
                ? input[(size_t)gy * width + gx] : 0.0f;
    tile_in[i] = (half)val;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Step 2: Zero accumulators ---- */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    numer[i] = (half)0.0f;
    denom[i] = (half)0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Step 3: Pass1 BayesShrink ---- */
  {
    const int n_blocks_dim = (TILE_W - GALOSH_BS) / GALOSH_STRIDE + 1;
    const float sigma_sq = sigma_strength * sigma_strength;
    const float lambda_max_base = sigma_strength * sqrt(2.0f * log((float)GALOSH_BP));

    for(int phase = 0; phase < N_PHASES; phase++)
    {
      const int px = phase % PHASE_MOD;
      const int py = phase / PHASE_MOD;
      const int bpd   = (n_blocks_dim - px + PHASE_MOD - 1) / PHASE_MOD;
      const int bpd_y = (n_blocks_dim - py + PHASE_MOD - 1) / PHASE_MOD;
      const int n_blocks = bpd * bpd_y;

      for(int bi = lid; bi < n_blocks; bi += wg_size)
      {
        const int pbx = bi % bpd;
        const int pby = bi / bpd;
        const int bx = pbx * PHASE_MOD + px;
        const int by = pby * PHASE_MOD + py;
        if(bx >= n_blocks_dim || by >= n_blocks_dim) continue;

        const int ref_c = bx * GALOSH_STRIDE;
        const int ref_r = by * GALOSH_STRIDE;

        /* Load 8×8 block: LDS half → private float */
        float block[GALOSH_BP];
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
            block[dy * GALOSH_BS + dx] = (float)tile_in[(ref_r + dy) * TILE_W + (ref_c + dx)];

        /* Forward WHT (float precision) */
        wht2d_8x8(block, 0);

        /* BayesShrink threshold (float precision for sigma estimation) */
        float sum_sq = 0.0f;
        for(int i = 1; i < GALOSH_BP; i++)
          sum_sq += block[i] * block[i];
        const float sigma_y_sq = sum_sq / ((float)(GALOSH_BP - 1) * (float)GALOSH_BP);
        const float sigma_x_sq = fmax(sigma_y_sq - sigma_sq, 0.0f);

        float lambda;
        if(sigma_x_sq < 1e-10f)
          lambda = 1e30f;
        else
        {
          lambda = (sigma_sq / sqrt(sigma_x_sq)) * sqrt((float)GALOSH_BP);
          const float lambda_max_unorm = lambda_max_base * sqrt((float)GALOSH_BP);
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

        /* Scatter: float → half into LDS accumulators */
        const float weight = 1.0f / (float)n_nonzero;
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const float kw = kaiser_1d[dy] * kaiser_1d[dx];
            const float wkw = weight * kw;
            const int pos = (ref_r + dy) * TILE_W + (ref_c + dx);
            numer[pos] += (half)(wkw * block[dy * GALOSH_BS + dx]);
            denom[pos] += (half)wkw;
          }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  /* ---- Step 4: Finalize Pass1 → pilot ---- */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    float n = (float)numer[i];
    float d = (float)denom[i];
    pilot[i] = (d > 1e-6f) ? (half)(n / d) : tile_in[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Step 5: Zero accumulators for Pass2 ---- */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    numer[i] = (half)0.0f;
    denom[i] = (half)0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Step 6: Pass2 Wiener shrinkage ---- */
  {
    const int n_blocks_dim = (TILE_W - GALOSH_BS) / GALOSH_STRIDE + 1;
    const float sigma_sq_unorm = sigma_strength * sigma_strength * (float)GALOSH_BP;

    for(int phase = 0; phase < N_PHASES; phase++)
    {
      const int px = phase % PHASE_MOD;
      const int py = phase / PHASE_MOD;
      const int bpd   = (n_blocks_dim - px + PHASE_MOD - 1) / PHASE_MOD;
      const int bpd_y = (n_blocks_dim - py + PHASE_MOD - 1) / PHASE_MOD;
      const int n_blocks = bpd * bpd_y;

      for(int bi = lid; bi < n_blocks; bi += wg_size)
      {
        const int pbx = bi % bpd;
        const int pby = bi / bpd;
        const int bx = pbx * PHASE_MOD + px;
        const int by = pby * PHASE_MOD + py;
        if(bx >= n_blocks_dim || by >= n_blocks_dim) continue;

        const int ref_c = bx * GALOSH_STRIDE;
        const int ref_r = by * GALOSH_STRIDE;

        /* Load noisy + pilot: LDS half → private float */
        float blk_noisy[GALOSH_BP];
        float blk_pilot[GALOSH_BP];
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const int pos = (ref_r + dy) * TILE_W + (ref_c + dx);
            blk_noisy[dy * GALOSH_BS + dx] = (float)tile_in[pos];
            blk_pilot[dy * GALOSH_BS + dx] = (float)pilot[pos];
          }

        /* Forward WHT on both (float) */
        wht2d_8x8(blk_noisy, 0);
        wht2d_8x8(blk_pilot, 0);

        /* Wiener shrinkage (float precision for division) */
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
            if(w < WIENER_FLOOR) w = WIENER_FLOOR;
          }
          blk_noisy[i] *= w;
          wiener_energy += w * w;
        }

        /* Inverse WHT */
        wht2d_8x8(blk_noisy, 1);

        /* Scatter: float → half into LDS */
        const float weight = 1.0f / fmax(wiener_energy, 1e-6f);
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const float kw = kaiser_1d[dy] * kaiser_1d[dx];
            const float wkw = weight * kw;
            const int pos = (ref_r + dy) * TILE_W + (ref_c + dx);
            numer[pos] += (half)(wkw * blk_noisy[dy * GALOSH_BS + dx]);
            denom[pos] += (half)wkw;
          }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  /* ---- Step 7: Finalize Pass2 → global output (FP16 → FP32) ---- */
  for(int i = lid; i < TILE_SIZE * TILE_SIZE; i += wg_size)
  {
    const int lx = i % TILE_SIZE + HALO;
    const int ly = i / TILE_SIZE + HALO;
    const int idx = ly * TILE_W + lx;
    const int gx = tile_x + (i % TILE_SIZE);
    const int gy = tile_y + (i / TILE_SIZE);
    if(gx < width && gy < height)
    {
      float n = (float)numer[idx];
      float d = (float)denom[idx];
      output[(size_t)gy * width + gx] = (d > 1e-6f)
          ? n / d : (float)tile_in[idx];
    }
  }
}


/* ================================================================
 * Pass2-only Kernel (FP16 LDS variant)
 *
 * For full-res L refinement with pre-computed pilot.
 * ================================================================ */

kernel void galosh_pass2_only(
    global const float *restrict input,
    global const float *restrict pilot_global,
    global float *restrict output,
    const int width,
    const int height,
    const float sigma_strength)
{
  const int tile_x = get_group_id(0) * TILE_SIZE;
  const int tile_y = get_group_id(1) * TILE_SIZE;
  const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
  const int wg_size = get_local_size(0) * get_local_size(1);

  local half tile_in[TILE_PIXELS];
  local half tile_pilot[TILE_PIXELS];
  local half numer[TILE_PIXELS];
  local half denom[TILE_PIXELS];

  /* ---- Load input + pilot (FP32 global → FP16 local) ---- */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    const int lx = i % TILE_W;
    const int ly = i / TILE_W;
    const int gx = tile_x - HALO + lx;
    const int gy = tile_y - HALO + ly;
    const int valid = (gx >= 0 && gx < width && gy >= 0 && gy < height);
    const size_t gpos = valid ? (size_t)gy * width + gx : 0;
    tile_in[i]    = valid ? (half)input[gpos]        : (half)0.0f;
    tile_pilot[i] = valid ? (half)pilot_global[gpos] : (half)0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Zero accumulators ---- */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    numer[i] = (half)0.0f;
    denom[i] = (half)0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Pass2 Wiener ---- */
  {
    const int n_blocks_dim = (TILE_W - GALOSH_BS) / GALOSH_STRIDE + 1;
    const float sigma_sq_unorm = sigma_strength * sigma_strength * (float)GALOSH_BP;

    for(int phase = 0; phase < N_PHASES; phase++)
    {
      const int px = phase % PHASE_MOD;
      const int py = phase / PHASE_MOD;
      const int bpd   = (n_blocks_dim - px + PHASE_MOD - 1) / PHASE_MOD;
      const int bpd_y = (n_blocks_dim - py + PHASE_MOD - 1) / PHASE_MOD;
      const int n_blocks = bpd * bpd_y;

      for(int bi = lid; bi < n_blocks; bi += wg_size)
      {
        const int pbx = bi % bpd;
        const int pby = bi / bpd;
        const int bx = pbx * PHASE_MOD + px;
        const int by = pby * PHASE_MOD + py;
        if(bx >= n_blocks_dim || by >= n_blocks_dim) continue;

        const int ref_c = bx * GALOSH_STRIDE;
        const int ref_r = by * GALOSH_STRIDE;

        float blk_noisy[GALOSH_BP];
        float blk_pilot[GALOSH_BP];
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const int pos = (ref_r + dy) * TILE_W + (ref_c + dx);
            blk_noisy[dy * GALOSH_BS + dx] = (float)tile_in[pos];
            blk_pilot[dy * GALOSH_BS + dx] = (float)tile_pilot[pos];
          }

        wht2d_8x8(blk_noisy, 0);
        wht2d_8x8(blk_pilot, 0);

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
            if(w < WIENER_FLOOR) w = WIENER_FLOOR;
          }
          blk_noisy[i] *= w;
          wiener_energy += w * w;
        }

        wht2d_8x8(blk_noisy, 1);

        const float weight = 1.0f / fmax(wiener_energy, 1e-6f);
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const float kw = kaiser_1d[dy] * kaiser_1d[dx];
            const float wkw = weight * kw;
            const int pos = (ref_r + dy) * TILE_W + (ref_c + dx);
            numer[pos] += (half)(wkw * blk_noisy[dy * GALOSH_BS + dx]);
            denom[pos] += (half)wkw;
          }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  /* ---- Finalize → global FP32 ---- */
  for(int i = lid; i < TILE_SIZE * TILE_SIZE; i += wg_size)
  {
    const int lx = i % TILE_SIZE + HALO;
    const int ly = i / TILE_SIZE + HALO;
    const int idx = ly * TILE_W + lx;
    const int gx = tile_x + (i % TILE_SIZE);
    const int gy = tile_y + (i / TILE_SIZE);
    if(gx < width && gy < height)
    {
      float n = (float)numer[idx];
      float d = (float)denom[idx];
      output[(size_t)gy * width + gx] = (d > 1e-6f)
          ? n / d : (float)tile_in[idx];
    }
  }
}
