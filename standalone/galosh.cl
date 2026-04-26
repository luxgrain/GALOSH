/* ================================================================
 * GALOSH (Generalized Anscombe LOcal SHrinkage) — unified kernels
 *
 * Single OpenCL source covering both RAW and YUV pipelines.
 * Organised into three thematic sections — core kernels shared by
 * both modes (§1), RAW-mode specifics (§2), and YUV/Y-GAT-mode
 * specifics (§3).  The guided_loess (LOESS chroma) kernel at the
 * end of §3 is used by both pipelines.
 *
 * Build: loaded as a single-source program by rawdenoise_gpu.c
 * (no runtime concatenation).  See README / paper for derivations.
 * ================================================================
 */

/* ================================================================
 * §1 + §2. Core (noise, GAT, WHT/LOSH, helpers) + RAW-mode kernels
 * ================================================================*/
/*
 * GALOSH Fused Tile Kernel — Pass1(BayesShrink) + Pass2(Wiener) in one dispatch
 *
 * Core innovation: Entire 2-pass denoising runs inside local memory.
 * Global memory accessed only twice: load input tile, store output tile.
 * Zero atomics, zero inter-workgroup sync.
 *
 * Tile-based overlap-add:
 *   Each workgroup owns a TILE_SIZE×TILE_SIZE interior region + halo.
 *   Blocks within the tile are processed in PHASE_MOD² phases to avoid
 *   local memory write conflicts (same phase = blocks spaced PHASE_MOD apart).
 *
 * Memory layout (FP32, TILE_SIZE=32):
 *   tile_in[46×46]  = 8.5 KB   (input + halo, persistent)
 *   numer[46×46]    = 8.5 KB   (accumulator, reused between passes)
 *   denom[46×46]    = 8.5 KB   (accumulator, reused between passes)
 *   pilot[46×46]    = 8.5 KB   (Pass1 output, read in Pass2)
 *   Total: 33.8 KB  → fits 48/64 KB LDS
 *
 * Target: stride=2, BS=8, 4K real-time on mobile GPU (FP16 future)
 *
 * Copyright (c) 2026 luxgrain. All rights reserved.
 */


#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* ================================================================
 * Compile-time constants (can be overridden with -D)
 * ================================================================ */

#ifndef GALOSH_BS
#define GALOSH_BS      8
#endif

#ifndef GALOSH_BP
#define GALOSH_BP     64    /* BS*BS */
#endif

#ifndef GALOSH_STRIDE
#define GALOSH_STRIDE  2
#endif

#ifndef TILE_SIZE
#define TILE_SIZE     32    /* interior pixels per tile dimension */
#endif

#define PHASE_MOD  (GALOSH_BS / GALOSH_STRIDE)   /* 4 for stride=2 */
#define N_PHASES   (PHASE_MOD * PHASE_MOD)        /* 16 */
#define HALO       (GALOSH_BS - GALOSH_STRIDE)     /* 6: ensures stride-aligned blocks */
#define TILE_W     (TILE_SIZE + 2 * HALO)         /* 46 for TILE_SIZE=32 */
#define TILE_PIXELS (TILE_W * TILE_W)             /* 2116 */

#define WIENER_FLOOR  0.125f   /* 1/BS */


/* ================================================================
 * Kaiser window (beta=2.0, N=8, precomputed)
 * ================================================================ */

constant float kaiser_1d[8] = {
  0.34012f, 0.59885f, 0.84123f, 0.97659f,
  0.97659f, 0.84123f, 0.59885f, 0.34012f
};


/* ================================================================
 * WHT 8-point in-place (sequency order) — addition/subtraction only
 *
 * Ref: Walsh-Hadamard Transform, natural/sequency ordering.
 * Pure add/sub → no precision loss in FP16, ideal for ISP/FPGA.
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

/* ---- FP16 WHT variants ---- */
inline void wht8_h(half *x)
{
  half a0 = x[0]+x[1], a1 = x[0]-x[1];
  half a2 = x[2]+x[3], a3 = x[2]-x[3];
  half a4 = x[4]+x[5], a5 = x[4]-x[5];
  half a6 = x[6]+x[7], a7 = x[6]-x[7];
  half b0 = a0+a2, b1 = a1+a3;
  half b2 = a0-a2, b3 = a1-a3;
  half b4 = a4+a6, b5 = a5+a7;
  half b6 = a4-a6, b7 = a5-a7;
  x[0] = b0+b4; x[1] = b1+b5; x[2] = b2+b6; x[3] = b3+b7;
  x[4] = b0-b4; x[5] = b1-b5; x[6] = b2-b6; x[7] = b3-b7;
}

inline void wht2d_8x8_h(half *block, const int normalize)
{
  for(int r = 0; r < 8; r++)
    wht8_h(block + r * 8);
  for(int c = 0; c < 8; c++)
  {
    half col[8];
    for(int r = 0; r < 8; r++) col[r] = block[r * 8 + c];
    wht8_h(col);
    for(int r = 0; r < 8; r++) block[r * 8 + c] = col[r];
  }
  if(normalize)
  {
    const half inv = (half)(1.0f / 64.0f);
    for(int i = 0; i < 64; i++) block[i] *= inv;
  }
}


/* ================================================================
 * MAD-based sigma_Y estimator for BayesShrink (GALOSH_*_G adoption).
 *
 * EN: Replaces the L2 sum_sq estimator with median absolute deviation
 *     of the AC coefficients.  Robust to ~25% outliers (Donoho-Johnstone
 *     1995); kills the spatial noise clusters that L2 sum_sq over-
 *     includes as "false signal" (the BM3D-CFA-clearable residual).
 *     Returns sigma_Y^2 in *per-pixel* scale (unnormalized WHT scale
 *     compensation: 1/N).
 *
 *     Implementation: partial selection sort that locates the median
 *     element (rank (N-1)/2 = 31 for N=64) of |block[1..N-1]| in O(rank·N)
 *     ≈ 31·63 ≈ 2000 ops per block.  Fast enough; the Pass1 hot path
 *     is dominated by the 2D WHT and overlap-add, not by sigma estimation.
 *
 * JP: AC 係数の MAD ベース sigma_Y 推定。L2 sum_sq の cluster-騙され
 *     問題 (BM3D-CFA で消える "誤信号" 残留) を構造的に解消、
 *     GPU/streaming 互換性は維持。partial selection sort で median を
 *     取得、ブロック当たり数千 op で十分速い。
 * ================================================================ */
inline float mad_sigma_y_sq_h(const half *restrict block)
{
    /* Copy |AC coefs| into a register array */
    float abs_ac[GALOSH_BP - 1];  /* 63 elements for GALOSH_BP=64 */
    for(int i = 1; i < GALOSH_BP; i++)
        abs_ac[i - 1] = fabs((float)block[i]);

    /* Partial selection: find smallest 32 elements; the 32nd-smallest
     * (== median of 63 = element at rank 31, 0-indexed) is the MAD.
     *
     * Why partial-selection wins on GPU vs O(N) quickselect / O(N log²N)
     * bitonic on this kernel: SIMT lockstep execution makes constant-
     * work-per-iteration algorithms outperform asymptotically-faster
     * branchy ones.  Quickselect's data-dependent branches cause warp
     * divergence; bitonic's nested ifs increase register pressure and
     * spills.  Empirical (12 MP image, 4046×3042):
     *   partial selection: 1.9 ms / fused_L
     *   bitonic sort:      3.0 ms / fused_L  (-58%)
     *   quickselect:       4.5 ms / fused_L  (-136%)
     */
    const int target = (GALOSH_BP - 1) / 2;  /* = 31 for N=64 -> 63 ACs -> rank 31 */
    for(int k = 0; k <= target; k++)
    {
        int min_idx = k;
        float min_val = abs_ac[k];
        for(int j = k + 1; j < GALOSH_BP - 1; j++)
        {
            if(abs_ac[j] < min_val) { min_val = abs_ac[j]; min_idx = j; }
        }
        abs_ac[min_idx] = abs_ac[k];
        abs_ac[k] = min_val;
    }
    const float mad = abs_ac[target];

    /* mad approximates 0.6745 * sqrt(N) * sigma_Y in unnormalized WHT
     * scale (because median(|N(0, N*sigma^2)|) = 0.6745 * sqrt(N) * sigma);
     * convert to per-pixel sigma_Y^2: (mad / 0.6745)^2 / N. */
    const float sy = mad / 0.6745f;
    return (sy * sy) / (float)GALOSH_BP;
}


/* ================================================================
 * Pass1-only Tile Kernel: BayesShrink hard thresholding
 *
 * Separated from fused_pass12 to reduce register pressure.
 * Each WG processes one tile: load input → Pass1 BayesShrink →
 * write pilot estimate to global memory.
 *
 * fused_pass12 からの分離版。レジスタ圧を半減することで
 * Intel Arc 等レジスタファイルの小さい GPU でのスピルを抑制。
 * 品質は fused_pass12 と完全一致（ビット完全同一）。
 *
 * LDS: tile_in + numer + denom = 3 × TILE_W² × sizeof(half)
 * ================================================================ */

kernel void galosh_pass1_only(
    global const half *restrict input,
    global half *restrict output,
    const int width,
    const int height,
    const float sigma_strength)
{
  const int tile_x = get_group_id(0) * TILE_SIZE;
  const int tile_y = get_group_id(1) * TILE_SIZE;
  const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
  const int wg_size = get_local_size(0) * get_local_size(1);

  local half tile_in[TILE_PIXELS];
  local half numer[TILE_PIXELS];
  local half denom[TILE_PIXELS];

  /* ---- Load input tile (with halo) ---- */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    const int lx = i % TILE_W;
    const int ly = i / TILE_W;
    const int gx = tile_x - HALO + lx;
    const int gy = tile_y - HALO + ly;
    tile_in[i] = (gx >= 0 && gx < width && gy >= 0 && gy < height)
                 ? input[(size_t)gy * width + gx] : (half)0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Zero accumulators ---- */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    numer[i] = (half)0.0f;
    denom[i] = (half)0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- BayesShrink (phased overlap-add, FP16 blocks + WHT, FP32 threshold) ---- */
  {
    const int n_blocks_dim = (TILE_W - GALOSH_BS) / GALOSH_STRIDE + 1;
    const float sigma_sq = sigma_strength * sigma_strength;
    const float lambda_max_base = sigma_strength * sqrt(2.0f * log((float)GALOSH_BP));

    for(int phase = 0; phase < N_PHASES; phase++)
    {
      const int px = phase % PHASE_MOD;
      const int py = phase / PHASE_MOD;
      const int bpd = (n_blocks_dim - px + PHASE_MOD - 1) / PHASE_MOD;
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

        half block[GALOSH_BP];
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
            block[dy * GALOSH_BS + dx] = tile_in[(ref_r + dy) * TILE_W + (ref_c + dx)];

        wht2d_8x8_h(block, 0);

        /* GALOSH_*_G: MAD-based sigma_Y (replaces L2 sum_sq).  See
         * mad_sigma_y_sq_h() above for derivation and rationale. */
        const float sigma_y_sq = mad_sigma_y_sq_h(block);
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

        /* Pass1 shrinkage — soft instead of hard.
         * EN: Hard thresholding ( |c|<λ → 0 else c ) creates a discontinuity
         *     at |c|=λ.  In yuv_gpu this leaks into the chroma path as a
         *     blocky overlap-add artifact and inflates LPIPS.  Soft
         *     thresholding ( c → sign(c)·max(|c|-λ, 0) ) is continuous and
         *     preserves edge gradient direction.  Pass2 Wiener still applies
         *     so noise floor is maintained.
         * JP: Pass1 を soft 化。hard 不連続 → over-smooth + ブロック化、
         *     perceptual を悪化させていた。soft で連続化、Pass2 Wiener が
         *     後段で noise level 確保。 */
        int n_nonzero = 1;
        for(int i = 1; i < GALOSH_BP; i++)
        {
          const float v = (float)block[i];
          const float a = fabs(v);
          if(a < lambda)
            block[i] = (half)0.0f;
          else
          {
            block[i] = (half)copysign(a - lambda, v);
            n_nonzero++;
          }
        }

        wht2d_8x8_h(block, 1);

        const float weight = 1.0f / (float)n_nonzero;
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const float kw = kaiser_1d[dy] * kaiser_1d[dx];
            const float wkw = weight * kw;
            const int pos = (ref_r + dy) * TILE_W + (ref_c + dx);
            numer[pos] += (half)(wkw * (float)block[dy * GALOSH_BS + dx]);
            denom[pos] += (half)wkw;
          }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  /* ---- Finalize → pilot output (interior only) ---- */
  for(int i = lid; i < TILE_SIZE * TILE_SIZE; i += wg_size)
  {
    const int lx = i % TILE_SIZE + HALO;
    const int ly = i / TILE_SIZE + HALO;
    const int idx = ly * TILE_W + lx;
    const int gx = tile_x + (i % TILE_SIZE);
    const int gy = tile_y + (i / TILE_SIZE);
    if(gx < width && gy < height)
    {
      const float d = (float)denom[idx];
      output[(size_t)gy * width + gx] = (d > 1e-6f)
          ? (half)((float)numer[idx] / d) : tile_in[idx];
    }
  }
}


/* ================================================================
 * Fused Tile Kernel: Pass1 (BayesShrink) + Pass2 (Wiener)
 *
 * One workgroup = one tile of the image.
 * All intermediate data stays in local memory.
 * Global memory: 1 read (input) + 1 write (output).
 *
 * Dispatch: (ceil(width/TILE_SIZE), ceil(height/TILE_SIZE))
 * Workgroup size: 256 (16×16) recommended
 * ================================================================ */

kernel void galosh_fused_pass12(
    global const half *restrict input,
    global half *restrict output,
    const int width,
    const int height,
    const float sigma_strength)
{
  const int tile_x = get_group_id(0) * TILE_SIZE;
  const int tile_y = get_group_id(1) * TILE_SIZE;
  const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
  const int wg_size = get_local_size(0) * get_local_size(1);

  local half tile_in[TILE_PIXELS];
  local half numer[TILE_PIXELS];
  local half denom[TILE_PIXELS];
  local half pilot[TILE_PIXELS];

  /* ---- Step 1: Load input tile (with halo) ---- */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    const int lx = i % TILE_W;
    const int ly = i / TILE_W;
    const int gx = tile_x - HALO + lx;
    const int gy = tile_y - HALO + ly;
    tile_in[i] = (gx >= 0 && gx < width && gy >= 0 && gy < height)
                 ? input[(size_t)gy * width + gx] : (half)0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Step 2: Zero accumulators ---- */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    numer[i] = (half)0.0f;
    denom[i] = (half)0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Step 3: Pass1 BayesShrink (phased overlap-add, FP16 blocks + WHT, FP32 threshold math) ---- */
  {
    const int n_blocks_dim = (TILE_W - GALOSH_BS) / GALOSH_STRIDE + 1;
    const float sigma_sq = sigma_strength * sigma_strength;
    const float lambda_max_base = sigma_strength * sqrt(2.0f * log((float)GALOSH_BP));

    for(int phase = 0; phase < N_PHASES; phase++)
    {
      const int px = phase % PHASE_MOD;
      const int py = phase / PHASE_MOD;
      const int bpd = (n_blocks_dim - px + PHASE_MOD - 1) / PHASE_MOD;
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

        half block[GALOSH_BP];
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
            block[dy * GALOSH_BS + dx] = tile_in[(ref_r + dy) * TILE_W + (ref_c + dx)];

        wht2d_8x8_h(block, 0);

        /* BayesShrink threshold in FP32 for precision */
        /* GALOSH_*_G: MAD-based sigma_Y (replaces L2 sum_sq).  See
         * mad_sigma_y_sq_h() above for derivation and rationale. */
        const float sigma_y_sq = mad_sigma_y_sq_h(block);
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

        /* Pass1 shrinkage — soft instead of hard.
         * EN: Hard thresholding ( |c|<λ → 0 else c ) creates a discontinuity
         *     at |c|=λ.  In yuv_gpu this leaks into the chroma path as a
         *     blocky overlap-add artifact and inflates LPIPS.  Soft
         *     thresholding ( c → sign(c)·max(|c|-λ, 0) ) is continuous and
         *     preserves edge gradient direction.  Pass2 Wiener still applies
         *     so noise floor is maintained.
         * JP: Pass1 を soft 化。hard 不連続 → over-smooth + ブロック化、
         *     perceptual を悪化させていた。soft で連続化、Pass2 Wiener が
         *     後段で noise level 確保。 */
        int n_nonzero = 1;
        for(int i = 1; i < GALOSH_BP; i++)
        {
          const float v = (float)block[i];
          const float a = fabs(v);
          if(a < lambda)
            block[i] = (half)0.0f;
          else
          {
            block[i] = (half)copysign(a - lambda, v);
            n_nonzero++;
          }
        }

        wht2d_8x8_h(block, 1);

        const float weight = 1.0f / (float)n_nonzero;
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const float kw = kaiser_1d[dy] * kaiser_1d[dx];
            const float wkw = weight * kw;
            const int pos = (ref_r + dy) * TILE_W + (ref_c + dx);
            numer[pos] += (half)(wkw * (float)block[dy * GALOSH_BS + dx]);
            denom[pos] += (half)wkw;
          }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  /* ---- Step 4: Finalize Pass1 → pilot ---- */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    const float d = (float)denom[i];
    pilot[i] = (d > 1e-6f) ? (half)((float)numer[i] / d) : tile_in[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Step 5: Zero accumulators for Pass2 ---- */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    numer[i] = (half)0.0f;
    denom[i] = (half)0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Step 6: Pass2 Wiener shrinkage (phased overlap-add, FP16 blocks + WHT, FP32 gain math) ---- */
  {
    const int n_blocks_dim = (TILE_W - GALOSH_BS) / GALOSH_STRIDE + 1;
    const float sigma_sq_unorm = sigma_strength * sigma_strength * (float)GALOSH_BP;

    for(int phase = 0; phase < N_PHASES; phase++)
    {
      const int px = phase % PHASE_MOD;
      const int py = phase / PHASE_MOD;
      const int bpd = (n_blocks_dim - px + PHASE_MOD - 1) / PHASE_MOD;
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

        half blk_noisy[GALOSH_BP];
        half blk_pilot[GALOSH_BP];
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const int pos = (ref_r + dy) * TILE_W + (ref_c + dx);
            blk_noisy[dy * GALOSH_BS + dx] = tile_in[pos];
            blk_pilot[dy * GALOSH_BS + dx] = pilot[pos];
          }

        wht2d_8x8_h(blk_noisy, 0);
        wht2d_8x8_h(blk_pilot, 0);

        /* Wiener gain in FP32, apply to half */
        float wiener_energy = 0.0f;
        for(int i = 0; i < GALOSH_BP; i++)
        {
          float w;
          if(i == 0)
            w = 1.0f;
          else
          {
            const float p = (float)blk_pilot[i];
            const float s2 = p * p;
            w = s2 / (s2 + sigma_sq_unorm);
            if(w < WIENER_FLOOR) w = WIENER_FLOOR;
          }
          blk_noisy[i] = (half)((float)blk_noisy[i] * w);
          wiener_energy += w * w;
        }

        wht2d_8x8_h(blk_noisy, 1);

        const float weight = 1.0f / fmax(wiener_energy, 1e-6f);
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const float kw = kaiser_1d[dy] * kaiser_1d[dx];
            const float wkw = weight * kw;
            const int pos = (ref_r + dy) * TILE_W + (ref_c + dx);
            numer[pos] += (half)(wkw * (float)blk_noisy[dy * GALOSH_BS + dx]);
            denom[pos] += (half)wkw;
          }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  /* ---- Step 7: Finalize Pass2 → global output (interior only) ---- */
  for(int i = lid; i < TILE_SIZE * TILE_SIZE; i += wg_size)
  {
    const int lx = i % TILE_SIZE + HALO;
    const int ly = i / TILE_SIZE + HALO;
    const int idx = ly * TILE_W + lx;
    const int gx = tile_x + (i % TILE_SIZE);
    const int gy = tile_y + (i / TILE_SIZE);
    if(gx < width && gy < height)
    {
      const float d = (float)denom[idx];
      output[(size_t)gy * width + gx] = (d > 1e-6f)
          ? (half)((float)numer[idx] / d) : tile_in[idx];
    }
  }
}


/* ================================================================
 * Pass2-only Fused Tile Kernel
 *
 * For full-res L refinement: uses a PRE-COMPUTED pilot (from denoised
 * half-res channels) instead of self-generated Pass1 pilot.
 * This matches the CPU pipeline exactly.
 *
 * Local memory: 3 buffers (input, pilot, numer/denom reused → 4 total
 * but pilot is loaded from global, not computed locally).
 *
 * LDS: tile_in + tile_pilot + numer + denom = 4 × TILE_W² × 4 bytes
 *      Same footprint as fused_pass12.
 * ================================================================ */

kernel void galosh_pass2_only(
    global const half *restrict input,
    global const half *restrict pilot_global,
    global half *restrict output,
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

  /* ---- Load input + pilot tiles (with halo) ---- */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    const int lx = i % TILE_W;
    const int ly = i / TILE_W;
    const int gx = tile_x - HALO + lx;
    const int gy = tile_y - HALO + ly;
    const int valid = (gx >= 0 && gx < width && gy >= 0 && gy < height);
    const size_t gpos = valid ? (size_t)gy * width + gx : 0;
    tile_in[i]    = valid ? input[gpos]        : (half)0.0f;
    tile_pilot[i] = valid ? pilot_global[gpos] : (half)0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Zero accumulators ---- */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    numer[i] = (half)0.0f;
    denom[i] = (half)0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Pass2 Wiener (phased, FP16 blocks + WHT, FP32 gain math) ---- */
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

        half blk_noisy[GALOSH_BP];
        half blk_pilot[GALOSH_BP];
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const int pos = (ref_r + dy) * TILE_W + (ref_c + dx);
            blk_noisy[dy * GALOSH_BS + dx] = tile_in[pos];
            blk_pilot[dy * GALOSH_BS + dx] = tile_pilot[pos];
          }

        wht2d_8x8_h(blk_noisy, 0);
        wht2d_8x8_h(blk_pilot, 0);

        /* Wiener gain in FP32, apply to half */
        float wiener_energy = 0.0f;
        for(int i = 0; i < GALOSH_BP; i++)
        {
          float w;
          if(i == 0)
            w = 1.0f;
          else
          {
            const float p = (float)blk_pilot[i];
            const float s2 = p * p;
            w = s2 / (s2 + sigma_sq_unorm);
            if(w < WIENER_FLOOR) w = WIENER_FLOOR;
          }
          blk_noisy[i] = (half)((float)blk_noisy[i] * w);
          wiener_energy += w * w;
        }

        wht2d_8x8_h(blk_noisy, 1);

        const float weight = 1.0f / fmax(wiener_energy, 1e-6f);
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const float kw = kaiser_1d[dy] * kaiser_1d[dx];
            const float wkw = weight * kw;
            const int pos = (ref_r + dy) * TILE_W + (ref_c + dx);
            numer[pos] += (half)(wkw * (float)blk_noisy[dy * GALOSH_BS + dx]);
            denom[pos] += (half)wkw;
          }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  /* ---- Finalize → global output (interior only) ---- */
  for(int i = lid; i < TILE_SIZE * TILE_SIZE; i += wg_size)
  {
    const int lx = i % TILE_SIZE + HALO;
    const int ly = i / TILE_SIZE + HALO;
    const int idx = ly * TILE_W + lx;
    const int gx = tile_x + (i % TILE_SIZE);
    const int gy = tile_y + (i / TILE_SIZE);
    if(gx < width && gy < height)
    {
      const float d = (float)denom[idx];
      output[(size_t)gy * width + gx] = (d > 1e-3f)
          ? (half)((float)numer[idx] / d) : tile_in[idx];
    }
  }
}


/* ================================================================
 * GALOSH Full-Pipeline GPU Kernels
 *
 * Pre/Post-processing: GAT, sigma estimation, dark reference,
 * WHT decompose, L fullres, and reconstruction — all on GPU.
 *
 * ISP story: entire GALOSH pipeline runs on GPU compute units.
 * CPU involvement limited to file I/O and OpenCL setup.
 * Suitable for single-chip ISP implementation.
 *
 * 全パイプライン GPU 化: GAT → σ推定 → dark_ref → WHT → denoise
 * → L fullres → 再構成 を全て GPU カーネルで実行。
 * ワンチップ ISP 実装のための proof-of-concept.
 *
 * params buffer layout:
 *   [0..3]  sigma_ch[4]      per-channel σ in GAT domain
 *   [4]     unified_sigma    √(mean(σ²))
 *   [5]     inv_sg           1/unified_sigma
 *   [6..9]  ch_dark_ref[4]   per-channel dark DC offset
 *   [10]    s_scale          dark ref iteration scale
 *   [11]    luma_strength    strength * luma_str
 *   [12]    chroma_strength  strength * chroma_str
 *   [13]    alpha            estimated Poisson gain
 *   [14]    sigma_sq         estimated read noise variance
 *
 * lut_params buffer layout:
 *   [0] d_min   [1] d_max   [2] y_break   [3] t_break   [4] sigma_raw
 * ================================================================ */

#ifndef HIST_BINS
#define HIST_BINS      4096
#endif
#ifndef HIST_MAX
#define HIST_MAX       16.0f
#endif
#ifndef REDUCE_WG_SIZE
#define REDUCE_WG_SIZE 256
#endif
#define ACHROMATIC_RANGE 4.0f
#define GAT_LUT_SIZE   4096

/* Noise estimation constants */
#define NE_BLOCK_SZ      8
#define NE_BLOCK_STEP    2     /* subsample: every 2nd block in x/y → 4× fewer blocks */
#define NE_BSEARCH_ITER  5     /* binary search iterations for median (1/32 precision) */
#define NE_NBINS        32
#define NE_VAR_SUBBINS 128
#define NE_DARK_HIST  1024
#define NE_LAP_HIST   2048
#define NE_LAP_RANGE  0.1f


/* ================================================================
 * K0a: Block statistics for blind noise model estimation
 *       (Foi et al. 2008 lower-envelope method)
 *
 * Dispatch: 1D (total_blocks = 4 * n_bx * n_by)
 * Each WI computes one 8×8 half-res block's mean and Laplacian-based
 * noise variance. Laplacian MAD is robust to edges within the block.
 *
 * 各 WI が 1 ブロック (8×8 半解像度) の平均と Laplacian ベースの
 * ノイズ分散を計算。Laplacian MAD はエッジに対してロバスト。
 * ================================================================ */

kernel void galosh_noise_block_stats(
    global const float *restrict raw,
    global float *restrict blk_mean,
    global float *restrict blk_var,
    const int width,
    const int height,
    const int n_bx,
    const int n_by)
{
  const int gid = get_global_id(0);
  const int blocks_per_ch = n_bx * n_by;
  const int total = 4 * blocks_per_ch;
  if(gid >= total) return;

  const int ch = gid / blocks_per_ch;
  const int bi = gid % blocks_per_ch;
  const int by = bi / n_bx;
  const int bx = bi % n_bx;

  /* CFA channel offsets: ch0=(0,0) ch1=(1,0) ch2=(0,1) ch3=(1,1) */
  const int dy0 = ch & 1;
  const int dx0 = (ch >> 1) & 1;
  const int y0 = by * (NE_BLOCK_SZ * NE_BLOCK_STEP);
  const int x0 = bx * (NE_BLOCK_SZ * NE_BLOCK_STEP);

  /* Block mean */
  float sum = 0.0f;
  for(int y = y0; y < y0 + NE_BLOCK_SZ; y++)
    for(int x = x0; x < x0 + NE_BLOCK_SZ; x++)
      sum += raw[(2 * y + dy0) * width + (2 * x + dx0)];
  const float bm = sum * (1.0f / (float)(NE_BLOCK_SZ * NE_BLOCK_SZ));

  /* Binary search for median of |Laplacian| — O(n × NE_BSEARCH_ITER).
   * Recomputes Laplacians each iteration; same 8×8 block (64 pixels)
   * stays in L1 cache.  No private array → no scratch spill. */

  /* Pass 0: count + min/max */
  float lap_min = 1e30f, lap_max = 0.0f;
  int nl = 0;

  for(int y = y0; y < y0 + NE_BLOCK_SZ; y++)
    for(int x = x0; x < x0 + NE_BLOCK_SZ - 2; x++)
    {
      const float v0 = raw[(2*y+dy0) * width + (2*x+dx0)];
      const float v1 = raw[(2*y+dy0) * width + (2*(x+1)+dx0)];
      const float v2 = raw[(2*y+dy0) * width + (2*(x+2)+dx0)];
      const float L = fabs(v0 - 2.0f * v1 + v2);
      lap_min = fmin(lap_min, L);
      lap_max = fmax(lap_max, L);
      nl++;
    }
  for(int y = y0; y < y0 + NE_BLOCK_SZ - 2; y++)
    for(int x = x0; x < x0 + NE_BLOCK_SZ; x++)
    {
      const float v0 = raw[(2*y+dy0) * width + (2*x+dx0)];
      const float v1 = raw[(2*(y+1)+dy0) * width + (2*x+dx0)];
      const float v2 = raw[(2*(y+2)+dy0) * width + (2*x+dx0)];
      const float L = fabs(v0 - 2.0f * v1 + v2);
      lap_min = fmin(lap_min, L);
      lap_max = fmax(lap_max, L);
      nl++;
    }

  if(nl <= 10)
  {
    blk_mean[gid] = bm;
    blk_var[gid] = 1e10f;
    return;
  }

  /* Binary search: find threshold where count(L <= threshold) == nl/2 */
  const int med_idx = nl >> 1;
  float lo = lap_min, hi = lap_max;

  for(int iter = 0; iter < NE_BSEARCH_ITER; iter++)
  {
    const float mid = (lo + hi) * 0.5f;
    int cnt = 0;

    for(int y = y0; y < y0 + NE_BLOCK_SZ; y++)
      for(int x = x0; x < x0 + NE_BLOCK_SZ - 2; x++)
      {
        const float v0 = raw[(2*y+dy0) * width + (2*x+dx0)];
        const float v1 = raw[(2*y+dy0) * width + (2*(x+1)+dx0)];
        const float v2 = raw[(2*y+dy0) * width + (2*(x+2)+dx0)];
        if(fabs(v0 - 2.0f * v1 + v2) <= mid) cnt++;
      }
    for(int y = y0; y < y0 + NE_BLOCK_SZ - 2; y++)
      for(int x = x0; x < x0 + NE_BLOCK_SZ; x++)
      {
        const float v0 = raw[(2*y+dy0) * width + (2*x+dx0)];
        const float v1 = raw[(2*(y+1)+dy0) * width + (2*x+dx0)];
        const float v2 = raw[(2*(y+2)+dy0) * width + (2*x+dx0)];
        if(fabs(v0 - 2.0f * v1 + v2) <= mid) cnt++;
      }

    if(cnt <= med_idx) lo = mid; else hi = mid;
  }

  const float med = (lo + hi) * 0.5f;
  const float sigma_lap = med / 0.6745f;
  const float sigma_lap_sq = sigma_lap * sigma_lap;

  blk_mean[gid] = bm;
  blk_var[gid] = sigma_lap_sq / 6.0f;
}


/* ================================================================
 * K0b: Noise model estimation — histogram-based percentile + Huber fit
 *       + dark σ² refinement.  Optimized for GPU: NO sorting,
 *       all intermediate data in local memory.
 *
 * Dispatch: 1D (64, local_size=64) — only WI 0 works;
 *           WG needed for LDS allocation.
 *
 * Perf notes (vs v1 single-WI with global scratch):
 *   - Histogram-based percentile replaces insertion sort: O(n) vs O(n²)
 *   - Local memory histograms: ~10ns vs ~200ns per access
 *   - 2 passes over block data (vs 32×n_total in v1)
 *   - Dark histograms in LDS: no global scratch dependency
 *
 * LDS layout (total ~16.3 KB, fits in 32 KB ISP target):
 *   var_hist  [32 × 128]   = 16 KB   per-bin variance histogram
 *   bin_cnt   [32]          = 128 B   block count per bin
 *   bin_msum  [32]          = 128 B   mean accumulator per bin
 *
 * Dark pixel scanning is handled by separate parallel kernels
 * (K0c/K0d/K0e) that build histograms with atomic_inc.
 *
 * ヒストグラムベース percentile: ソート不要、O(n)。
 * 暗部スキャンは別カーネルで並列化。
 * ================================================================ */

kernel void galosh_noise_estimate(
    global const float *restrict blk_mean,
    global const float *restrict blk_var,
    global const int *restrict dark_samp_hist,
    global float *restrict params,
    const int n_total,
    const int n_dark_samples)
{
  const int lid = get_local_id(0);
  const int wg  = get_local_size(0);   /* 64 */

  /* ---- LDS allocation ---- */
  local int   var_hist[NE_NBINS * NE_VAR_SUBBINS]; /* 32×128 = 16KB */
  local int   bin_cnt[NE_NBINS];
  local int   bin_msum_fp[NE_NBINS];  /* fixed-point ×65536 for atomic add */
  local float lds_mm[4 * 64];         /* 4 arrays × 64 WIs for reduction */
  local float s_gmin, s_gmax, s_bw, s_vmin, s_inv_vbw;
  local int   s_early_exit;

  /* ---- Pass 1: parallel min/max reduction ----
   * Each WI scans n_total/wg blocks, then tree-reduce in LDS. */
  float my_gmin = 1e30f, my_gmax = 0.0f;
  float my_vmin = 1e30f, my_vmax = 0.0f;

  for(int i = lid; i < n_total; i += wg)
  {
    const float m = blk_mean[i];
    const float v = blk_var[i];
    if(m > 0.003f && m < 0.97f)
    {
      my_gmin = fmin(my_gmin, m);
      my_gmax = fmax(my_gmax, m);
      if(v < 1e9f)
      {
        my_vmin = fmin(my_vmin, v);
        my_vmax = fmax(my_vmax, v);
      }
    }
  }

  lds_mm[lid]        = my_gmin;
  lds_mm[64 + lid]   = my_gmax;
  lds_mm[128 + lid]  = my_vmin;
  lds_mm[192 + lid]  = my_vmax;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int s = wg >> 1; s > 0; s >>= 1)
  {
    if(lid < s)
    {
      lds_mm[lid]       = fmin(lds_mm[lid],       lds_mm[lid + s]);
      lds_mm[64 + lid]  = fmax(lds_mm[64 + lid],  lds_mm[64 + lid + s]);
      lds_mm[128 + lid] = fmin(lds_mm[128 + lid], lds_mm[128 + lid + s]);
      lds_mm[192 + lid] = fmax(lds_mm[192 + lid], lds_mm[192 + lid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(lid == 0)
  {
    float gmin_l = lds_mm[0], gmax_l = lds_mm[64];
    float vmin_l = lds_mm[128], vmax_l = lds_mm[192];
    float bw_l = (gmax_l - gmin_l) / (float)NE_NBINS;
    s_early_exit = (bw_l < 1e-10f) ? 1 : 0;
    if(s_early_exit)
    {
      params[13] = 1e-4f;
      params[14] = 1e-6f;
      params[10] = 1e-6f / 1e-4f;
    }
    else
    {
      s_gmin = gmin_l; s_gmax = gmax_l; s_bw = bw_l;
      s_vmin = vmin_l;
      float vbw_l = (vmax_l - vmin_l) / (float)NE_VAR_SUBBINS;
      s_inv_vbw = (vbw_l > 1e-20f) ? 1.0f / vbw_l : 0.0f;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if(s_early_exit) return;

  /* ---- Clear histograms in parallel ---- */
  for(int i = lid; i < NE_NBINS * NE_VAR_SUBBINS; i += wg) var_hist[i] = 0;
  if(lid < NE_NBINS) { bin_cnt[lid] = 0; bin_msum_fp[lid] = 0; }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Pass 2: parallel histogram + atomic bin assignment ----
   * var_hist & bin_cnt: atomic_add in LDS.
   * bin_msum: fixed-point ×65536 atomic_add (quality-neutral). */
  for(int i = lid; i < n_total; i += wg)
  {
    const float m = blk_mean[i];
    const float v = blk_var[i];
    if(m < 0.003f || m > 0.97f || v > 1e9f) continue;
    const int mb = clamp((int)((m - s_gmin) / s_bw), 0, NE_NBINS - 1);
    const int vb = clamp((int)((v - s_vmin) * s_inv_vbw), 0, NE_VAR_SUBBINS - 1);
    atomic_add(&var_hist[mb * NE_VAR_SUBBINS + vb], 1);
    atomic_add(&bin_cnt[mb], 1);
    atomic_add(&bin_msum_fp[mb], (int)(m * 65536.0f));
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Remaining work: WI 0 only (scans 32 bins → microseconds) ---- */
  if(lid != 0) return;

  /* Recover range vars from LDS for percentile extraction */
  const float vmin = s_vmin;
  const float vbw  = (s_inv_vbw > 0.0f) ? 1.0f / s_inv_vbw : 0.0f;

  /* ---- Extract 5-20th percentile from var histograms ---- */
  float bin_mean_arr[NE_NBINS], bin_var_arr[NE_NBINS];
  int bin_cnt_arr[NE_NBINS], bin_valid[NE_NBINS];
  int n_valid = 0;

  for(int b = 0; b < NE_NBINS; b++)
  {
    bin_valid[b] = 0;
    if(bin_cnt[b] < 20) continue;

    const int p5_tgt  = bin_cnt[b] / 20;
    const int p20_tgt = bin_cnt[b] / 5;
    int cum = 0;
    float wsum = 0.0f;
    int wcnt = 0;

    for(int vb = 0; vb < NE_VAR_SUBBINS; vb++)
    {
      const int c = var_hist[b * NE_VAR_SUBBINS + vb];
      if(c == 0) continue;
      const int prev = cum;
      cum += c;
      if(cum > p5_tgt && prev < p20_tgt)
      {
        const int lo = max(p5_tgt  - prev, 0);
        const int hi = min(cum, p20_tgt) - prev;
        const int use = hi - lo;
        const float v_center = vmin + ((float)vb + 0.5f) * vbw;
        wsum += v_center * (float)use;
        wcnt += use;
      }
    }
    if(wcnt > 0)
    {
      bin_var_arr[b]  = wsum / (float)wcnt;
      bin_mean_arr[b] = (float)bin_msum_fp[b] / (65536.0f * (float)bin_cnt[b]);
      bin_cnt_arr[b]  = wcnt;
      bin_valid[b]    = 1;
      n_valid++;
    }
  }

  if(n_valid < 4)
  {
    params[13] = 1e-4f;
    params[14] = 1e-6f;
    params[10] = 1e-6f / 1e-4f;
    return;
  }

  /* ---- Huber M-estimator WLS fit: Var = α·mean + σ² (5 iter) ---- */
  float alpha_est = 0.01f, sigma_sq_est = 0.0f;

  for(int iter = 0; iter < 5; iter++)
  {
    float huber_k = 1e10f;
    if(iter > 0)
    {
      float resids[NE_NBINS];
      int nr = 0;
      for(int b = 0; b < NE_NBINS; b++)
      {
        if(!bin_valid[b]) continue;
        resids[nr++] = fabs(bin_var_arr[b]
                            - (alpha_est * bin_mean_arr[b] + sigma_sq_est));
      }
      for(int i = 1; i < nr; i++)
      {
        float key = resids[i];
        int j = i - 1;
        while(j >= 0 && resids[j] > key) { resids[j+1] = resids[j]; j--; }
        resids[j+1] = key;
      }
      huber_k = 1.345f * fmax(resids[nr >> 1] / 0.6745f, 1e-12f);
    }

    float Sw = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    for(int b = 0; b < NE_NBINS; b++)
    {
      if(!bin_valid[b]) continue;
      float w = (float)bin_cnt_arr[b];
      if(iter > 0)
      {
        const float resid = fabs(bin_var_arr[b]
                                 - (alpha_est * bin_mean_arr[b] + sigma_sq_est));
        if(resid > huber_k) w *= huber_k / resid;
      }
      const float x = bin_mean_arr[b], y = bin_var_arr[b];
      Sw += w; Sx += w*x; Sy += w*y; Sxx += w*x*x; Sxy += w*x*y;
    }
    const float det = Sw * Sxx - Sx * Sx;
    if(fabs(det) > 1e-30f)
    {
      const float na = (Sw * Sxy - Sx * Sy) / det;
      const float ns = (Sxx * Sy - Sx * Sxy) / det;
      if(na > 0) alpha_est = na;
      if(ns >= 0) sigma_sq_est = ns;
    }
  }
  alpha_est = fmax(alpha_est, 1e-8f);

  /* ---- Scan dark sample histogram → dark threshold ---- */
  int cum = 0;
  const int tgt = n_dark_samples / 10;
  float dark_thresh = 0.0f;
  for(int i = 0; i < NE_DARK_HIST; i++)
  {
    cum += dark_samp_hist[i];
    if(cum >= tgt)
    {
      dark_thresh = (float)i / (float)NE_DARK_HIST;
      break;
    }
  }

  /* Write alpha + dark_thresh (sigma_sq finalized by K0e) */
  params[13] = alpha_est;
  params[14] = sigma_sq_est;          /* initial, refined by K0e */
  params[10] = sigma_sq_est / fmax(alpha_est, 1e-12f);
  params[15] = dark_thresh + 0.02f;   /* dark_max for K0d */
}


/* ================================================================
 * K0c: Dark sample histogram — parallel
 *
 * Dispatch: 1D (n_dark_samples)
 * Each WI samples one raw pixel (stride-3 in x/y, 4 CFA channels),
 * atomic_inc into NE_DARK_HIST-bin histogram.
 * Runs concurrently with K0a (independent of block stats).
 *
 * 暗部閾値推定用の raw サンプルヒストグラム。atomic_inc で並列。
 * ================================================================ */

kernel void galosh_noise_dark_samp_hist(
    global const float *restrict raw,
    global int *restrict dark_hist,
    const int width,
    const int height,
    const int samp_per_ch_row,
    const int samp_per_ch,
    const int n_total_samples)
{
  const int gid = get_global_id(0);
  if(gid >= n_total_samples) return;

  const int ch  = gid / samp_per_ch;
  const int rem = gid % samp_per_ch;
  const int sy  = rem / samp_per_ch_row;
  const int sx  = rem % samp_per_ch_row;
  const int y   = sy * 3;
  const int x   = sx * 3;
  const int dy0 = ch & 1;
  const int dx0 = (ch >> 1) & 1;

  const float v = raw[(2 * y + dy0) * width + (2 * x + dx0)];
  const int bin = clamp((int)(v * (float)NE_DARK_HIST), 0, NE_DARK_HIST - 1);
  atomic_inc(&dark_hist[bin]);
}


/* ================================================================
 * K0d: Dark Laplacian histogram — parallel
 *
 * Dispatch: 1D (total_lap_positions = 4 * (hh*(hw-2) + (hh-2)*hw))
 * Each WI computes one horizontal or vertical Laplacian at a dark pixel
 * position. Reads dark_max from params[15]. atomic_inc into histogram.
 *
 * 暗部 Laplacian のヒストグラム。全ピクセル位置を並列処理。
 * ================================================================ */

kernel void galosh_noise_dark_lap_hist(
    global const float *restrict raw,
    global int *restrict lap_hist,
    global const float *restrict params,
    const int width,
    const int height,
    const int pos_per_ch_h,
    const int pos_per_ch)
{
  const int gid = get_global_id(0);
  const int total = 4 * pos_per_ch;
  if(gid >= total) return;

  const int hw = width >> 1;
  const int hh = height >> 1;
  const int ch = gid / pos_per_ch;
  const int rem = gid % pos_per_ch;
  const int dy0 = ch & 1;
  const int dx0 = (ch >> 1) & 1;

  const float dark_max = params[15];

  float v0, v1, v2;

  if(rem < pos_per_ch_h)
  {
    /* Horizontal Laplacian */
    const int y = rem / (hw - 2);
    const int x = rem % (hw - 2);
    v0 = raw[(2*y+dy0) * width + (2*x+dx0)];
    v1 = raw[(2*y+dy0) * width + (2*(x+1)+dx0)];
    v2 = raw[(2*y+dy0) * width + (2*(x+2)+dx0)];
  }
  else
  {
    /* Vertical Laplacian */
    const int li = rem - pos_per_ch_h;
    const int y = li / hw;
    const int x = li % hw;
    v0 = raw[(2*y+dy0) * width + (2*x+dx0)];
    v1 = raw[(2*(y+1)+dy0) * width + (2*x+dx0)];
    v2 = raw[(2*(y+2)+dy0) * width + (2*x+dx0)];
  }

  if(v0 > dark_max || v1 > dark_max || v2 > dark_max) return;

  const float lap = fabs(v0 - 2.0f * v1 + v2);
  const int bin = clamp((int)(lap / NE_LAP_RANGE * (float)NE_LAP_HIST),
                        0, NE_LAP_HIST - 1);
  atomic_inc(&lap_hist[bin]);
}


/* ================================================================
 * K0e: Dark sigma finalize — single WI
 *
 * Dispatch: 1D (1)
 * Scans dark Laplacian histogram → median → σ² refinement.
 * Writes final alpha, sigma_sq, s_scale to params.
 *
 * 暗部 Laplacian ヒストグラムからメディアン → σ² 精錬。
 * ================================================================ */

kernel void galosh_noise_dark_finalize(
    global const int *restrict lap_hist,
    global float *restrict params)
{
  if(get_global_id(0) != 0) return;

  /* Count total dark Laplacians */
  int ndl = 0;
  for(int i = 0; i < NE_LAP_HIST; i++) ndl += lap_hist[i];

  if(ndl <= 100) return; /* keep initial sigma_sq from Huber fit */

  /* Scan for median */
  const int med_tgt = ndl / 2;
  int cum = 0;
  float med = 0.0f;
  for(int i = 0; i < NE_LAP_HIST; i++)
  {
    cum += lap_hist[i];
    if(cum >= med_tgt)
    {
      med = ((float)i + 0.5f) * NE_LAP_RANGE / (float)NE_LAP_HIST;
      break;
    }
  }

  const float sigma_lap = med / 0.6745f;
  const float dark_var  = (sigma_lap * sigma_lap) / 6.0f;
  const float alpha     = params[13];
  const float dark_max  = params[15];
  const float dark_mean = (dark_max - 0.02f) * 0.5f; /* dark_thresh = dark_max - 0.02 */
  const float sigma_sq  = fmax(dark_var - alpha * dark_mean, 0.0f);

  params[14] = sigma_sq;
  params[10] = sigma_sq / fmax(alpha, 1e-12f);
}


/* ================================================================
 * K1: GAT forward + Bayer extraction
 *
 * Dispatch: 2D (hw, hh)
 * Input:  raw Bayer [width × height]
 * Output: 4 channels [hw × hh], ordering TL/BL/TR/BR
 * ================================================================ */

kernel void galosh_gat_extract(
    global const float *restrict raw,
    global float *restrict ch0,
    global float *restrict ch1,
    global float *restrict ch2,
    global float *restrict ch3,
    const int width,
    const int height,
    const float alpha,
    const float sigma_sq)
{
  const int hx = get_global_id(0);
  const int hy = get_global_id(1);
  const int hw = width >> 1;
  const int hh = height >> 1;
  if(hx >= hw || hy >= hh) return;

  const int hi = hy * hw + hx;
  const int fy = hy << 1, fx = hx << 1;
  const float c = 0.375f * alpha * alpha + sigma_sq;
  const float inv_a = 2.0f / alpha;

  /* Channel ordering: c0=TL(R), c1=BL(Gb), c2=TR(Gr), c3=BR(B)
   * Must match CPU rawdenoise_v6.c for correct WHT L/C semantics. */
  ch0[hi] = inv_a * sqrt(fmax(alpha * raw[fy     * width + fx]     + c, 0.0f));
  ch1[hi] = inv_a * sqrt(fmax(alpha * raw[(fy+1) * width + fx]     + c, 0.0f));
  ch2[hi] = inv_a * sqrt(fmax(alpha * raw[fy     * width + fx + 1] + c, 0.0f));
  ch3[hi] = inv_a * sqrt(fmax(alpha * raw[(fy+1) * width + fx + 1] + c, 0.0f));
}


/* ================================================================
 * K2: Build inverse GAT lookup table (Gauss-Hermite quadrature)
 *
 * Dispatch: 1D (4096)
 * Each WI computes E[GAT(Poisson(x/α)·α + N(0,σ²))] for one x_val.
 * FP32 precision (sufficient for 4096-entry LUT with interpolation).
 *
 * Ref: Makitalo & Foi (IEEE TIP 2013) exact unbiased inverse GAT
 * ================================================================ */

constant float gh_nodes[10] = {
  -3.436159f, -2.532732f, -1.756684f, -1.036611f, -0.342901f,
   0.342901f,  1.036611f,  1.756684f,  2.532732f,  3.436159f
};
constant float gh_wts[10] = {
  7.640433e-06f, 1.343646e-03f, 3.387439e-02f,
  2.401386e-01f, 6.108626e-01f,
  6.108626e-01f, 2.401386e-01f, 3.387439e-02f,
  1.343646e-03f, 7.640433e-06f
};

kernel void galosh_build_inv_lut(
    global float *restrict lut_d,
    global float *restrict lut_x,
    const float alpha,
    const float sigma_sq)
{
  const int i = get_global_id(0);
  if(i >= GAT_LUT_SIZE) return;

  const float sig = sqrt(fmax(sigma_sq, 1e-20f));
  const float y_break = -0.375f * alpha;
  const float t_break = 2.0f * sig / alpha;
  const float x_val = (float)i / (float)(GAT_LUT_SIZE - 1);
  const float lambda = x_val / alpha;

  float expected_gat = 0.0f;
  const int k_max = (int)(lambda + 8.0f * sqrt(fmax(lambda, 1.0f))) + 20;
  float log_prob = -lambda;

  for(int k = 0; k <= k_max; k++)
  {
    if(k > 0) log_prob += log(lambda) - log((float)k);
    const float prob = exp(log_prob);
    if(prob < 1e-12f && k > (int)lambda + 1) break;

    float eg = 0.0f;
    for(int g = 0; g < 10; g++)
    {
      const float z = 1.4142135624f * sig * gh_nodes[g];
      const float noisy_y = (float)k * alpha + z;
      float T;
      if(noisy_y >= y_break)
      {
        const float arg = alpha * noisy_y + 0.375f * alpha * alpha + sigma_sq;
        T = (2.0f / alpha) * sqrt(fmax(arg, 0.0f));
      }
      else
        T = t_break + (noisy_y - y_break) / sig;
      eg += gh_wts[g] * T;
    }
    eg *= 0.5641895835f; /* 1/sqrt(pi) */
    expected_gat += prob * eg;
  }

  lut_x[i] = x_val;
  lut_d[i] = expected_gat;
}


/* ================================================================
 * K3: LUT finalize — extract d_min/d_max and derived constants
 *
 * Dispatch: 1D (1)
 * ================================================================ */

kernel void galosh_lut_finalize(
    global const float *restrict lut_d,
    global float *restrict lut_params,
    const float alpha,
    const float sigma_sq)
{
  if(get_global_id(0) != 0) return;
  const float sig = sqrt(fmax(sigma_sq, 1e-20f));
  lut_params[0] = lut_d[0];              /* d_min */
  lut_params[1] = lut_d[GAT_LUT_SIZE-1]; /* d_max */
  lut_params[2] = -0.375f * alpha;       /* y_break */
  lut_params[3] = 2.0f * sig / alpha;    /* t_break */
  lut_params[4] = sig;                   /* sigma_raw */
  lut_params[5] = alpha;                 /* for O(1) GAT inverse index */
  lut_params[6] = sigma_sq;              /* for O(1) GAT inverse index */
}


/* ================================================================
 * K4: Sigma estimation — histogram of |Laplacian| per channel
 *
 * Dispatch: 1D (n_samples), where n_samples = n_x_per_row * hh
 * n_x_per_row = (hw - 2) / 3 + 1  (stride-3 subsampling in x)
 *
 * Laplacian: L = data[x] - 2·data[x+1] + data[x+2]
 * For iid noise σ: Var(L) = 6σ² → MAD(|L|) = σ·√6·0.6745
 * → σ = MAD / 1.6521
 *
 * Atomic increment into global histogram (low contention:
 * ~30K samples / 4096 bins ≈ 7 hits/bin on average).
 * ================================================================ */

kernel void galosh_sigma_histogram(
    global const float *restrict ch0,
    global const float *restrict ch1,
    global const float *restrict ch2,
    global const float *restrict ch3,
    global volatile int *restrict hist,
    const int hw,
    const int hh)
{
  const int gid = get_global_id(0);
  const int n_x = (hw - 2) / 3 + 1;
  const int total = n_x * hh;
  if(gid >= total) return;

  const int y = gid / n_x;
  const int x = (gid % n_x) * 3;
  if(x + 2 >= hw) return;

  const int base = y * hw + x;
  const float bin_scale = (float)HIST_BINS / HIST_MAX;

  /* Process all 4 channels */
  float v0, v1, v2, lap;
  int bin;

  /* ch0 */
  v0 = ch0[base]; v1 = ch0[base+1]; v2 = ch0[base+2];
  lap = fabs(v0 - 2.0f * v1 + v2);
  bin = clamp((int)(lap * bin_scale), 0, HIST_BINS - 1);
  atomic_inc(&hist[0 * HIST_BINS + bin]);

  /* ch1 */
  v0 = ch1[base]; v1 = ch1[base+1]; v2 = ch1[base+2];
  lap = fabs(v0 - 2.0f * v1 + v2);
  bin = clamp((int)(lap * bin_scale), 0, HIST_BINS - 1);
  atomic_inc(&hist[1 * HIST_BINS + bin]);

  /* ch2 */
  v0 = ch2[base]; v1 = ch2[base+1]; v2 = ch2[base+2];
  lap = fabs(v0 - 2.0f * v1 + v2);
  bin = clamp((int)(lap * bin_scale), 0, HIST_BINS - 1);
  atomic_inc(&hist[2 * HIST_BINS + bin]);

  /* ch3 */
  v0 = ch3[base]; v1 = ch3[base+1]; v2 = ch3[base+2];
  lap = fabs(v0 - 2.0f * v1 + v2);
  bin = clamp((int)(lap * bin_scale), 0, HIST_BINS - 1);
  atomic_inc(&hist[3 * HIST_BINS + bin]);
}


/* ================================================================
 * K5: Sigma finalize — scan histograms → median → σ → unified
 *
 * Dispatch: 1D (1)
 * Reads 4×4096 histogram bins, finds median for each channel,
 * computes unified_sigma = √(mean(σ²)), inv_sg = 1/unified_sigma.
 * Writes results to params buffer.
 * ================================================================ */

kernel void galosh_sigma_finalize(
    global const int *restrict hist,
    global float *restrict params,
    const int n_samples_per_ch)
{
  if(get_global_id(0) != 0) return;

  const int median_target = n_samples_per_ch / 2;
  const float inv_scale = HIST_MAX / (float)HIST_BINS;
  float mean_var = 0.0f;

  for(int c = 0; c < 4; c++)
  {
    int cumsum = 0;
    int median_bin = 0;
    for(int b = 0; b < HIST_BINS; b++)
    {
      cumsum += hist[c * HIST_BINS + b];
      if(cumsum >= median_target) { median_bin = b; break; }
    }
    float mad = ((float)median_bin + 0.5f) * inv_scale;
    float sigma = fmax(mad / 1.6521f, 0.01f);
    params[c] = sigma;         /* sigma_ch[c] */
    mean_var += sigma * sigma;
  }
  mean_var *= 0.25f;
  float unified = sqrt(fmax(mean_var, 1e-12f));
  params[4] = unified;        /* unified_sigma */
  params[5] = 1.0f / unified; /* inv_sg */
}


/* ================================================================
 * K6: Normalize all channels by inv_sg (σ → 1)
 *
 * Dispatch: 1D (chsize)
 * ================================================================ */

kernel void galosh_normalize(
    global float *restrict ch0,
    global float *restrict ch1,
    global float *restrict ch2,
    global float *restrict ch3,
    global const float *restrict params,
    const int chsize)
{
  const int i = get_global_id(0);
  if(i >= chsize) return;
  const float inv_sg = params[5];
  ch0[i] *= inv_sg;
  ch1[i] *= inv_sg;
  ch2[i] *= inv_sg;
  ch3[i] *= inv_sg;
}


/* ================================================================
 * K7: Dark reference — weighted reduction (main pass)
 *
 * Dispatch: 1D (n_reduce_wg × REDUCE_WG_SIZE)
 * Each WG computes partial weighted sums, writes to partial buffer.
 *
 * Weight: w = 1/(1 + (L_raw/s_scale)^4)  — heavy dark weighting
 * Achromatic filter: reject pixels with inter-ch spread > threshold
 *
 * self-consistent dark anchor 推定: 暗部ピクセルのチャネル間
 * DC オフセットを加重平均で求める (2反復, scale refinement 付き).
 * ================================================================ */

kernel void galosh_dark_ref_reduce(
    global const float *restrict ch0,
    global const float *restrict ch1,
    global const float *restrict ch2,
    global const float *restrict ch3,
    global const float *restrict raw,
    global float *restrict partial,
    global const float *restrict params,
    const int hw,
    const int hh,
    const int width)
{
  const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  const int wg_id = get_group_id(0);
  const int n_global = get_global_size(0);
  const int total = hw * hh;

  float my_w = 0.0f;
  float my_c0 = 0.0f, my_c1 = 0.0f, my_c2 = 0.0f, my_c3 = 0.0f;
  const float inv_s = 1.0f / fmax(params[10], 1e-20f);

  for(int idx = gid; idx < total; idx += n_global)
  {
    const int hy = idx / hw, hx = idx % hw;
    const float g0 = ch0[idx], g1 = ch1[idx];
    const float g2 = ch2[idx], g3 = ch3[idx];
    const float cmax = fmax(fmax(g0, g1), fmax(g2, g3));
    const float cmin = fmin(fmin(g0, g1), fmin(g2, g3));
    if(cmax - cmin > ACHROMATIC_RANGE) continue;

    const int fy = hy * 2, fx = hx * 2;
    const float iv0 = raw[fy * width + fx];
    const float iv1 = raw[(fy+1) * width + fx];
    const float iv2 = raw[fy * width + fx + 1];
    const float iv3 = raw[(fy+1) * width + fx + 1];
    const float L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25f;
    const float r = L_raw * inv_s;
    const float r2 = r * r;
    const float w = 1.0f / (1.0f + r2 * r2);

    my_w  += w;
    my_c0 += w * g0; my_c1 += w * g1;
    my_c2 += w * g2; my_c3 += w * g3;
  }

  /* WG-level tree reduction */
  local float lw[REDUCE_WG_SIZE];
  local float lc[REDUCE_WG_SIZE * 4];
  lw[lid] = my_w;
  lc[lid*4+0] = my_c0; lc[lid*4+1] = my_c1;
  lc[lid*4+2] = my_c2; lc[lid*4+3] = my_c3;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int s = REDUCE_WG_SIZE / 2; s > 0; s >>= 1)
  {
    if(lid < s)
    {
      lw[lid] += lw[lid + s];
      lc[lid*4+0] += lc[(lid+s)*4+0];
      lc[lid*4+1] += lc[(lid+s)*4+1];
      lc[lid*4+2] += lc[(lid+s)*4+2];
      lc[lid*4+3] += lc[(lid+s)*4+3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(lid == 0)
  {
    partial[wg_id*5+0] = lw[0];
    partial[wg_id*5+1] = lc[0]; partial[wg_id*5+2] = lc[1];
    partial[wg_id*5+3] = lc[2]; partial[wg_id*5+4] = lc[3];
  }
}


/* ================================================================
 * K8: Dark reference finalize — sum partials → dark_ref
 *
 * Dispatch: 1D (1)
 * Reads partial sums from all WGs, computes weighted mean.
 * Writes ch_dark_ref[4] to params[6..9].
 * ================================================================ */

kernel void galosh_dark_ref_finalize(
    global const float *restrict partial,
    global float *restrict params,
    const int n_wg)
{
  if(get_global_id(0) != 0) return;

  float sum_w = 0.0f;
  float sum_c0 = 0.0f, sum_c1 = 0.0f, sum_c2 = 0.0f, sum_c3 = 0.0f;
  for(int i = 0; i < n_wg; i++)
  {
    sum_w  += partial[i*5+0];
    sum_c0 += partial[i*5+1]; sum_c1 += partial[i*5+2];
    sum_c2 += partial[i*5+3]; sum_c3 += partial[i*5+4];
  }
  const float inv_sw = 1.0f / fmax(sum_w, 1e-20f);
  params[6] = sum_c0 * inv_sw;
  params[7] = sum_c1 * inv_sw;
  params[8] = sum_c2 * inv_sw;
  params[9] = sum_c3 * inv_sw;
}


/* ================================================================
 * K9: Dark reference residual reduction (for scale refinement)
 *
 * Dispatch: 1D (n_reduce_wg × REDUCE_WG_SIZE)
 * Computes weighted sum of squared residuals from current dark_ref.
 * Used to refine s_scale between iterations.
 * ================================================================ */

kernel void galosh_dark_ref_resid_reduce(
    global const float *restrict ch0,
    global const float *restrict ch1,
    global const float *restrict ch2,
    global const float *restrict ch3,
    global const float *restrict raw,
    global float *restrict partial_resid,
    global const float *restrict params,
    const int hw,
    const int hh,
    const int width)
{
  const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  const int wg_id = get_group_id(0);
  const int n_global = get_global_size(0);
  const int total = hw * hh;

  float my_w2 = 0.0f, my_wr2 = 0.0f;
  const float inv_s = 1.0f / fmax(params[10], 1e-20f);
  const float dr0 = params[6], dr1 = params[7];
  const float dr2 = params[8], dr3 = params[9];

  for(int idx = gid; idx < total; idx += n_global)
  {
    const int hy = idx / hw, hx = idx % hw;
    const float g0 = ch0[idx], g1 = ch1[idx];
    const float g2 = ch2[idx], g3 = ch3[idx];
    const float cmax = fmax(fmax(g0, g1), fmax(g2, g3));
    const float cmin = fmin(fmin(g0, g1), fmin(g2, g3));
    if(cmax - cmin > ACHROMATIC_RANGE) continue;

    const int fy = hy * 2, fx = hx * 2;
    const float iv0 = raw[fy * width + fx];
    const float iv1 = raw[(fy+1) * width + fx];
    const float iv2 = raw[fy * width + fx + 1];
    const float iv3 = raw[(fy+1) * width + fx + 1];
    const float L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25f;
    const float r = L_raw * inv_s;
    const float r2 = r * r;
    const float w = 1.0f / (1.0f + r2 * r2);

    my_w2 += w;
    const float e0 = g0 - dr0, e1 = g1 - dr1;
    const float e2 = g2 - dr2, e3 = g3 - dr3;
    my_wr2 += w * (e0*e0 + e1*e1 + e2*e2 + e3*e3) * 0.25f;
  }

  local float lw2[REDUCE_WG_SIZE];
  local float lwr2[REDUCE_WG_SIZE];
  lw2[lid] = my_w2;
  lwr2[lid] = my_wr2;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int s = REDUCE_WG_SIZE / 2; s > 0; s >>= 1)
  {
    if(lid < s)
    {
      lw2[lid] += lw2[lid + s];
      lwr2[lid] += lwr2[lid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(lid == 0)
  {
    partial_resid[wg_id*2+0] = lw2[0];
    partial_resid[wg_id*2+1] = lwr2[0];
  }
}


/* ================================================================
 * K10: Dark reference residual finalize — update s_scale
 *
 * Dispatch: 1D (1)
 * s_scale *= sqrt(1/measured_std) with clamping
 * ================================================================ */

kernel void galosh_dark_ref_resid_finalize(
    global const float *restrict partial_resid,
    global float *restrict params,
    const int n_wg,
    const float s_min,
    const float s_max)
{
  if(get_global_id(0) != 0) return;

  float sum_w2 = 0.0f, sum_wr2 = 0.0f;
  for(int i = 0; i < n_wg; i++)
  {
    sum_w2 += partial_resid[i*2+0];
    sum_wr2 += partial_resid[i*2+1];
  }
  const float measured_std = sqrt(fmax(sum_wr2 / fmax(sum_w2, 1e-20f), 1e-20f));
  const float ratio = 1.0f / measured_std;
  float s = params[10] * sqrt(ratio);
  s = clamp(s, s_min, s_max);
  params[10] = s;
}


/* ================================================================
 * K11: Subtract dark reference from all channels (in-place)
 *
 * Dispatch: 1D (chsize)
 * ================================================================ */

kernel void galosh_dark_ref_subtract(
    global float *restrict ch0,
    global float *restrict ch1,
    global float *restrict ch2,
    global float *restrict ch3,
    global const float *restrict params,
    const int chsize)
{
  const int i = get_global_id(0);
  if(i >= chsize) return;
  ch0[i] -= params[6];
  ch1[i] -= params[7];
  ch2[i] -= params[8];
  ch3[i] -= params[9];
}


/* ================================================================
 * K12: 2×2 WHT decompose — 4 Bayer channels → L, C1, C2, C3
 *
 * Dispatch: 1D (chsize)
 * Input:  ch0(TL), ch1(BL), ch2(TR), ch3(BR) — normalized, dark-subtracted
 * Output: luma, chroma1, chroma2, chroma3
 * ================================================================ */

kernel void galosh_wht_decompose(
    global const float *restrict ch0,
    global const float *restrict ch1,
    global const float *restrict ch2,
    global const float *restrict ch3,
    global half *restrict luma,
    global half *restrict chroma1,
    global half *restrict chroma2,
    global half *restrict chroma3,
    const int chsize)
{
  const int i = get_global_id(0);
  if(i >= chsize) return;
  const float a = ch0[i], b = ch1[i], c = ch2[i], d = ch3[i];
  luma[i]    = (half)((a + b + c + d) * 0.5f);
  chroma1[i] = (half)((a - b + c - d) * 0.5f);
  chroma2[i] = (half)((a + b - c - d) * 0.5f);
  chroma3[i] = (half)((a - b - c + d) * 0.5f);
}


/* ================================================================
 * K13: Compute L fullres — half-res L/C → full-res luminance
 *
 * Dispatch: 2D (hw, hh)
 * Each WI writes 4 full-res pixels from one 2×2 block.
 * Matches CPU compute_L_fullres exactly.
 * ================================================================ */

kernel void galosh_compute_L_fullres(
    global const half *restrict L,
    global const half *restrict C1,
    global const half *restrict C2,
    global const half *restrict C3,
    global half *restrict out,
    const int hw,
    const int hh)
{
  const int hx = get_global_id(0);
  const int hy = get_global_id(1);
  if(hx >= hw || hy >= hh) return;

  const int fw = hw * 2;
  const int hi = hy * hw + hx;
  const int hx1 = min(hx + 1, hw - 1);
  const int hy1 = min(hy + 1, hh - 1);
  const int hi_r  = hy  * hw + hx1;
  const int hi_d  = hy1 * hw + hx;
  const int hi_dr = hy1 * hw + hx1;
  const int fr = hy * 2, fc = hx * 2;

  /* Promote to float for interpolation math, write back as half */
  const float fL = (float)L[hi], fL_r = (float)L[hi_r],
              fL_d = (float)L[hi_d], fL_dr = (float)L[hi_dr];
  const float fC1 = (float)C1[hi], fC1_r = (float)C1[hi_r],
              fC1_d = (float)C1[hi_d], fC1_dr = (float)C1[hi_dr];
  const float fC2 = (float)C2[hi], fC2_r = (float)C2[hi_r],
              fC2_d = (float)C2[hi_d], fC2_dr = (float)C2[hi_dr];
  const float fC3 = (float)C3[hi], fC3_r = (float)C3[hi_r],
              fC3_d = (float)C3[hi_d], fC3_dr = (float)C3[hi_dr];

  /* TL: block-aligned position */
  out[fr * fw + fc] = L[hi];
  /* TR: horizontal interpolation */
  out[fr * fw + fc + 1]
      = (half)(((fL - fC2) + (fL_r + fC2_r)) * 0.5f);
  /* BL: vertical interpolation */
  out[(fr+1) * fw + fc]
      = (half)(((fL - fC1) + (fL_d + fC1_d)) * 0.5f);
  /* BR: 4-corner interpolation */
  {
    const float Ls  = fL  + fL_r  + fL_d  + fL_dr;
    const float C1s = -fC1 - fC1_r + fC1_d + fC1_dr;
    const float C2s = -fC2 + fC2_r - fC2_d + fC2_dr;
    const float C3s =  fC3 - fC3_r - fC3_d + fC3_dr;
    out[(fr+1) * fw + fc + 1] = (half)((Ls + C1s + C2s + C3s) * 0.25f);
  }
}


/* ================================================================
 * K14: Reconstruct — inverse 2×2 WHT + dark_ref restore +
 *      σ denormalization + exact inverse GAT (LUT binary search)
 *
 * Dispatch: 2D (hw, hh)
 * Each WI reconstructs one 2×2 Bayer block (4 output pixels).
 *
 * Matches CPU reconstruction exactly:
 *   (1) Block-aligned L (avoid spatial mismatch with per-block C)
 *   (2) Inverse WHT + dark_ref restoration
 *   (3) × unified_sigma (denormalize)
 *   (4) LUT-based exact inverse GAT (binary search)
 * ================================================================ */

inline float gat_inv_lut(
    const float D,
    global const float *lut_d,
    global const float *lut_x,
    const float d_min,
    const float d_max,
    const float y_break,
    const float t_break,
    const float sigma_raw,
    const float alpha,
    const float sigma_sq)
{
  if(D <= d_min)
    return y_break + sigma_raw * (D - t_break);
  if(D >= d_max)
    return 1.0f;

  /* Bisection search for the LUT bin [lo, lo+1] containing D.
   *
   * EN: The earlier "O(1) algebraic estimate + 8-step refinement" failed
   *     on bimodal chroma windows near specular highlights.  The algebraic
   *     inverse x_est = (αD/2)² / α is only valid when D = α·GAT(x); but
   *     in K16 D = (L_block ± C1 ± C2 ± C3)·sg/2 + dark_ref, mixing post-
   *     WHT values with the unified_sigma normalisation.  x_est overshot
   *     by >10× on chroma outliers, forcing lo to the LUT end, and 8
   *     refinement steps could not climb back to the true bin.  Observed
   *     gat_inv_lut returning 3.89e5 at D=175.8 whose true x ≈ 0.23 —
   *     this injected saturated pixels into silver regions as the red/
   *     blue grains visible on 6/80 SIDD Medium scenes (0077/0078 G4,
   *     0083 GP).  Bisection is O(log N) = 12 reads, still cheap, and
   *     guarantees a bracketing bin regardless of upstream val scale.
   *
   * JP: 旧 O(1) 代数推定 + ±8 step 補正は bimodal chroma で破綻。
   *     x_est が >10× overshoot → lo が LUT 末尾にクランプ → 8 step では
   *     戻れず内挿で 389239 を返す事故を観測 (silver 面の赤青粒)。
   *     bisection の O(log N) ≈ 12 read に置換。L+C スケールに依らず
   *     正しい bin が決まる。 */
  int lo = 0, hi = GAT_LUT_SIZE - 1;
  while(lo + 1 < hi){
    const int mid = (lo + hi) >> 1;
    if(lut_d[mid] <= D) lo = mid; else hi = mid;
  }
  const float d0 = lut_d[lo], d1 = lut_d[lo + 1];
  const float t = (D - d0) / fmax(d1 - d0, 1e-10f);
  return lut_x[lo] + t * (lut_x[lo + 1] - lut_x[lo]);
}

kernel void galosh_reconstruct(
    global const half *restrict L_fr_den,
    global const half *restrict c1_out,
    global const half *restrict c2_out,
    global const half *restrict c3_out,
    global float *restrict output,
    global const float *restrict lut_d,
    global const float *restrict lut_x,
    global const float *restrict lut_params,
    global const float *restrict params,
    const int width,
    const int height)
{
  const int hx = get_global_id(0);
  const int hy = get_global_id(1);
  const int hw = width >> 1;
  const int hh = height >> 1;
  if(hx >= hw || hy >= hh) return;

  const int hi = hy * hw + hx;
  const int fy = hy << 1, fx = hx << 1;
  const float sg = params[4]; /* unified_sigma */

  /* Block-aligned L: use (fy, fx) position for all 4 sub-pixels.
   * Promote half → float for reconstruction math.
   * ブロック整合位置の L のみ使用 → C との空間整合を保証 */
  const float L_block = (float)L_fr_den[fy * width + fx];
  const float c1 = (float)c1_out[hi], c2 = (float)c2_out[hi], c3 = (float)c3_out[hi];

  /* Inverse 2×2 WHT + dark_ref restore */
  const float val_R  = (L_block + c1 + c2 + c3) * 0.5f + params[6];
  const float val_Gb = (L_block - c1 + c2 - c3) * 0.5f + params[7];
  const float val_Gr = (L_block + c1 - c2 - c3) * 0.5f + params[8];
  const float val_B  = (L_block - c1 - c2 + c3) * 0.5f + params[9];

  /* Denormalize (× unified_sigma) → exact inverse GAT via LUT */
  const float dm = lut_params[0], dx = lut_params[1];
  const float yb = lut_params[2], tb = lut_params[3], sr = lut_params[4];
  const float al = lut_params[5], sq = lut_params[6];

  /* Defensive clamp: gat_inv_lut can propagate NaN/Inf if upstream emits
   * such poison; raw output is defined on [0,1]. */
  output[fy     * width + fx]     = clamp(gat_inv_lut(val_R  * sg, lut_d, lut_x, dm, dx, yb, tb, sr, al, sq), 0.0f, 1.0f);
  output[fy     * width + fx + 1] = clamp(gat_inv_lut(val_Gr * sg, lut_d, lut_x, dm, dx, yb, tb, sr, al, sq), 0.0f, 1.0f);
  output[(fy+1) * width + fx]     = clamp(gat_inv_lut(val_Gb * sg, lut_d, lut_x, dm, dx, yb, tb, sr, al, sq), 0.0f, 1.0f);
  output[(fy+1) * width + fx + 1] = clamp(gat_inv_lut(val_B  * sg, lut_d, lut_x, dm, dx, yb, tb, sr, al, sq), 0.0f, 1.0f);
}


/* ================================================================
 * STREAMING FUSED KERNELS
 *
 * Kernel fusion for streaming / ISP pipeline:
 *   (1) stream_preprocess:  normalize + dark_sub + WHT → 1 kernel
 *   (2) stream_postprocess: compute_Lfr + pass2_only + reconstruct → 1 kernel
 *
 * Eliminates intermediate global memory round-trips.
 * Quality-neutral: same math, just fused dispatch.
 *
 * ストリーミング融合カーネル:
 *   前処理3カーネル → 1 / 後処理4カーネル → 1 に統合。
 *   中間バッファの global memory 往復を削減。品質変化なし。
 * ================================================================ */


/* ================================================================
 * K_SP: Stream Preprocess — normalize + dark_ref_subtract + wht_decompose
 *
 * Dispatch: 1D (chsize)
 * Input:  float ch0-ch3 (GAT domain, after sigma/dark estimation)
 * Output: half L, C1, C2, C3 (normalized, dark-subtracted, WHT'd)
 *
 * Fuses K6 normalize + K11 dark_ref_subtract + K12 wht_decompose.
 * 3 global memory round-trips → 1.
 * ================================================================ */

kernel void galosh_stream_preprocess(
    global const float *restrict ch0,
    global const float *restrict ch1,
    global const float *restrict ch2,
    global const float *restrict ch3,
    global half *restrict luma,
    global half *restrict chroma1,
    global half *restrict chroma2,
    global half *restrict chroma3,
    global const float *restrict params,
    const int chsize)
{
  const int i = get_global_id(0);
  if(i >= chsize) return;

  /* dark_ref subtract + 2×2 WHT decompose in one pass.
   * Note: ch0-ch3 are already normalized (×inv_sg) by K6 normalize,
   * which must run before dark_ref estimation. */
  const float a = ch0[i] - params[6];
  const float b = ch1[i] - params[7];
  const float c = ch2[i] - params[8];
  const float d = ch3[i] - params[9];

  luma[i]    = (half)((a + b + c + d) * 0.5f);
  chroma1[i] = (half)((a - b + c - d) * 0.5f);
  chroma2[i] = (half)((a + b - c - d) * 0.5f);
  chroma3[i] = (half)((a - b - c + d) * 0.5f);
}


/* ================================================================
 * K_SPOST: Stream Postprocess — compute_Lfr + pass2_Lfr + reconstruct
 *
 * Dispatch: tile-based 2D (full-res width, height), WG=16×16
 * Input:  half L/C1/C2/C3 noisy + denoised (8 half-res buffers)
 * Output: float reconstructed raw (full-res)
 *
 * Fuses K14 compute_L_fullres (×2) + K15 pass2_only + K16 reconstruct.
 * Eliminates Lfr_noisy_buf, Lfr_pilot_buf, Lfr_den_buf entirely.
 *
 * LDS: 4 × TILE_PIXELS × sizeof(half) = same as pass2_only.
 * L_fullres computed on-the-fly during tile load from half-res channels.
 *
 * ストリーミング後処理: L全解像度計算 → Pass2 → 再構成を1カーネルに融合。
 * 3つの中間バッファ（Lfr_noisy/pilot/den）を完全に排除。
 * ================================================================ */

kernel void galosh_stream_postprocess(
    global const half *restrict L_noisy,
    global const half *restrict C1_noisy,
    global const half *restrict C2_noisy,
    global const half *restrict C3_noisy,
    global const half *restrict L_den,
    global const half *restrict C1_den,
    global const half *restrict C2_den,
    global const half *restrict C3_den,
    global float *restrict output,
    global const float *restrict lut_d,
    global const float *restrict lut_x,
    global const float *restrict lut_params,
    global const float *restrict params,
    const int width,
    const int height)
{
  const int tile_x = get_group_id(0) * TILE_SIZE;
  const int tile_y = get_group_id(1) * TILE_SIZE;
  const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
  const int wg_size = get_local_size(0) * get_local_size(1);
  const int hw = width >> 1, hh = height >> 1;

  local half tile_in[TILE_PIXELS];     /* L_fr_noisy */
  local half tile_pilot[TILE_PIXELS];  /* L_fr_pilot */
  local half numer[TILE_PIXELS];
  local half denom[TILE_PIXELS];

  /* ---- Step 1: Compute L_fullres on-the-fly from half-res channels ----
   * Each full-res pixel computed by reading 4 half-res neighbors.
   * Replaces two separate compute_L_fullres dispatches. */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    const int lx = i % TILE_W;
    const int ly = i / TILE_W;
    const int gx = tile_x - HALO + lx;
    const int gy = tile_y - HALO + ly;

    if(gx < 0 || gx >= width || gy < 0 || gy >= height)
    {
      tile_in[i] = (half)0.0f;
      tile_pilot[i] = (half)0.0f;
      continue;
    }

    /* Map full-res → half-res coordinates */
    const int hx = gx >> 1;
    const int hy = gy >> 1;
    const int sub_x = gx & 1;
    const int sub_y = gy & 1;
    const int hx1 = min(hx + 1, hw - 1);
    const int hy1 = min(hy + 1, hh - 1);
    const int hi    = hy  * hw + hx;
    const int hi_r  = hy  * hw + hx1;
    const int hi_d  = hy1 * hw + hx;
    const int hi_dr = hy1 * hw + hx1;

    /* Compute L_fullres for noisy channels (matches compute_L_fullres exactly) */
    float Ln, Lp;
    if(sub_x == 0 && sub_y == 0)
    {
      /* TL: block-aligned */
      Ln = (float)L_noisy[hi];
      Lp = (float)L_den[hi];
    }
    else if(sub_x == 1 && sub_y == 0)
    {
      /* TR: horizontal interpolation */
      Ln = ((float)L_noisy[hi] - (float)C2_noisy[hi]
          + (float)L_noisy[hi_r] + (float)C2_noisy[hi_r]) * 0.5f;
      Lp = ((float)L_den[hi] - (float)C2_den[hi]
          + (float)L_den[hi_r] + (float)C2_den[hi_r]) * 0.5f;
    }
    else if(sub_x == 0 && sub_y == 1)
    {
      /* BL: vertical interpolation */
      Ln = ((float)L_noisy[hi] - (float)C1_noisy[hi]
          + (float)L_noisy[hi_d] + (float)C1_noisy[hi_d]) * 0.5f;
      Lp = ((float)L_den[hi] - (float)C1_den[hi]
          + (float)L_den[hi_d] + (float)C1_den[hi_d]) * 0.5f;
    }
    else
    {
      /* BR: 4-corner interpolation */
      const float nLs  = (float)L_noisy[hi]  + (float)L_noisy[hi_r]  + (float)L_noisy[hi_d]  + (float)L_noisy[hi_dr];
      const float nC1s = -(float)C1_noisy[hi] - (float)C1_noisy[hi_r] + (float)C1_noisy[hi_d] + (float)C1_noisy[hi_dr];
      const float nC2s = -(float)C2_noisy[hi] + (float)C2_noisy[hi_r] - (float)C2_noisy[hi_d] + (float)C2_noisy[hi_dr];
      const float nC3s =  (float)C3_noisy[hi] - (float)C3_noisy[hi_r] - (float)C3_noisy[hi_d] + (float)C3_noisy[hi_dr];
      Ln = (nLs + nC1s + nC2s + nC3s) * 0.25f;

      const float dLs  = (float)L_den[hi]  + (float)L_den[hi_r]  + (float)L_den[hi_d]  + (float)L_den[hi_dr];
      const float dC1s = -(float)C1_den[hi] - (float)C1_den[hi_r] + (float)C1_den[hi_d] + (float)C1_den[hi_dr];
      const float dC2s = -(float)C2_den[hi] + (float)C2_den[hi_r] - (float)C2_den[hi_d] + (float)C2_den[hi_dr];
      const float dC3s =  (float)C3_den[hi] - (float)C3_den[hi_r] - (float)C3_den[hi_d] + (float)C3_den[hi_dr];
      Lp = (dLs + dC1s + dC2s + dC3s) * 0.25f;
    }

    tile_in[i] = (half)Ln;
    tile_pilot[i] = (half)Lp;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Step 2: Zero accumulators ---- */
  for(int i = lid; i < TILE_PIXELS; i += wg_size)
  {
    numer[i] = (half)0.0f;
    denom[i] = (half)0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Step 3: Pass2 Wiener on full-res L (same as pass2_only) ---- */
  {
    const float sigma_strength = params[11];  /* luma_strength from params */
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

        half blk_noisy[GALOSH_BP];
        half blk_pilot[GALOSH_BP];
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const int pos = (ref_r + dy) * TILE_W + (ref_c + dx);
            blk_noisy[dy * GALOSH_BS + dx] = tile_in[pos];
            blk_pilot[dy * GALOSH_BS + dx] = tile_pilot[pos];
          }

        wht2d_8x8_h(blk_noisy, 0);
        wht2d_8x8_h(blk_pilot, 0);

        float wiener_energy = 0.0f;
        for(int j = 0; j < GALOSH_BP; j++)
        {
          float w;
          if(j == 0)
            w = 1.0f;
          else
          {
            const float p = (float)blk_pilot[j];
            const float s2 = p * p;
            w = s2 / (s2 + sigma_sq_unorm);
            if(w < WIENER_FLOOR) w = WIENER_FLOOR;
          }
          blk_noisy[j] = (half)((float)blk_noisy[j] * w);
          wiener_energy += w * w;
        }

        wht2d_8x8_h(blk_noisy, 1);

        const float weight = 1.0f / fmax(wiener_energy, 1e-6f);
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const float kw = kaiser_1d[dy] * kaiser_1d[dx];
            const float wkw = weight * kw;
            const int pos = (ref_r + dy) * TILE_W + (ref_c + dx);
            numer[pos] += (half)(wkw * (float)blk_noisy[dy * GALOSH_BS + dx]);
            denom[pos] += (half)wkw;
          }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  /* ---- Step 4: Finalize pass2 + Reconstruct → output ----
   * Fuses pass2 finalize + inverse WHT + dark_ref + denormalize + inverse GAT.
   * Each output pixel: L_fr from LDS, C from global, then full reconstruction. */
  {
    const float sg = params[4];  /* unified_sigma */
    const float dm = lut_params[0], dx = lut_params[1];
    const float yb = lut_params[2], tb = lut_params[3], sr = lut_params[4];
    const float al = lut_params[5], sq = lut_params[6];

    for(int i = lid; i < TILE_SIZE * TILE_SIZE; i += wg_size)
    {
      const int gx = tile_x + (i % TILE_SIZE);
      const int gy = tile_y + (i / TILE_SIZE);
      if(gx >= width || gy >= height) continue;

      /* Get block-aligned L from pass2 LDS output
       * Block-aligned = even full-res coordinates (fy,fx) for all 4 sub-pixels */
      const int ax = gx & ~1;
      const int ay = gy & ~1;
      const int a_lx = ax - tile_x + HALO;
      const int a_ly = ay - tile_y + HALO;
      const int a_idx = a_ly * TILE_W + a_lx;
      const float da = (float)denom[a_idx];
      const float L_block = (da > 1e-3f)
          ? (float)numer[a_idx] / da
          : (float)tile_in[a_idx];

      /* Read denoised C channels from global memory */
      const int hx = gx >> 1, hy = gy >> 1;
      const int hi = hy * hw + hx;
      const float c1 = (float)C1_den[hi];
      const float c2 = (float)C2_den[hi];
      const float c3 = (float)C3_den[hi];

      /* Inverse 2×2 WHT + dark_ref restore (Bayer sub-pixel selection) */
      const int sub_x = gx & 1;
      const int sub_y = gy & 1;
      float val;
      if(sub_x == 0 && sub_y == 0)
        val = (L_block + c1 + c2 + c3) * 0.5f + params[6];   /* R (TL) */
      else if(sub_x == 1 && sub_y == 0)
        val = (L_block + c1 - c2 - c3) * 0.5f + params[8];   /* Gr (TR) */
      else if(sub_x == 0 && sub_y == 1)
        val = (L_block - c1 + c2 - c3) * 0.5f + params[7];   /* Gb (BL) */
      else
        val = (L_block - c1 - c2 + c3) * 0.5f + params[9];   /* B (BR) */

      /* Denormalize (× unified_sigma) → inverse GAT via LUT */
      output[(size_t)gy * width + gx] = gat_inv_lut(val * sg,
          lut_d, lut_x, dm, dx, yb, tb, sr, al, sq);
    }
  }
}


/* ================================================================
 * Single-plane kernels — for YUV GALOSH GPU
 *
 * Minimal kernel set to denoise one float32 plane entirely on GPU:
 *   1. galosh_sigma_histogram_single — Laplacian MAD histogram (1 channel)
 *   2. galosh_sigma_finalize_single  — histogram → σ
 *   3. galosh_normalize_single       — float32 / σ → half (unit variance)
 *   4. galosh_denormalize_single     — half × σ → float32
 * ================================================================ */

/* K_SP1: Single-channel Laplacian MAD histogram.
 * Dispatch: 1D, global = ceil((width-2)/3) * height */
kernel void galosh_sigma_histogram_single(
    global const float *restrict plane,
    global volatile int *restrict hist,
    const int width,
    const int height)
{
  const int gid = get_global_id(0);
  const int n_x = (width - 2) / 3 + 1;
  const int total = n_x * height;
  if(gid >= total) return;

  const int y = gid / n_x;
  const int x = (gid % n_x) * 3;
  if(x + 2 >= width) return;

  const int base = y * width + x;
  const float bin_scale = (float)HIST_BINS / HIST_MAX;

  float v0 = plane[base], v1 = plane[base + 1], v2 = plane[base + 2];
  float lap = fabs(v0 - 2.0f * v1 + v2);
  int bin = clamp((int)(lap * bin_scale), 0, HIST_BINS - 1);
  atomic_inc(&hist[bin]);
}

/* K_SP2: Single-channel sigma finalize.
 * Dispatch: 1D (1 work-item).
 * Writes sigma and inv_sigma to sigma_out[0] and sigma_out[1]. */
kernel void galosh_sigma_finalize_single(
    global const int *restrict hist,
    global float *restrict sigma_out,
    const int n_samples)
{
  if(get_global_id(0) != 0) return;
  const int target = n_samples / 2;
  const float inv_scale = HIST_MAX / (float)HIST_BINS;
  int cumsum = 0, median_bin = 0;
  for(int b = 0; b < HIST_BINS; b++) {
    cumsum += hist[b];
    if(cumsum >= target) { median_bin = b; break; }
  }
  float mad   = ((float)median_bin + 0.5f) * inv_scale;
  float sigma = fmax(mad / 1.6521f, 0.01f);
  sigma_out[0] = sigma;
  sigma_out[1] = 1.0f / sigma;
}

/* K_SP3: Normalize float32 → half (FP16), dividing by sigma.
 * Dispatch: 1D, global = npix */
kernel void galosh_normalize_single(
    global const float *restrict plane_in,
    global half *restrict plane_out,
    global const float *restrict sigma_buf,
    const int npix)
{
  const int i = get_global_id(0);
  if(i >= npix) return;
  const float inv_sigma = sigma_buf[1];   /* sigma_out[1] = 1/sigma */
  plane_out[i] = (half)(plane_in[i] * inv_sigma);
}

/* K_SP4: Denormalize half → float32, multiplying by sigma.
 * Dispatch: 1D, global = npix */
kernel void galosh_denormalize_single(
    global const half *restrict plane_in,
    global float *restrict plane_out,
    global const float *restrict sigma_buf,
    const int npix)
{
  const int i = get_global_id(0);
  if(i >= npix) return;
  const float sigma = sigma_buf[0];   /* sigma_out[0] = sigma */
  plane_out[i] = (float)plane_in[i] * sigma;
}


/* ================================================================
 * §3. YUV/Y-GAT mode + shared chroma utilities
 *
 * Originally split as galosh_yuv_gat.cl; merged here as §3.
 * Includes the shared LOESS chroma kernel used by both modes.
 * ================================================================*/
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
/* Chroma noise-variance ratio against Y (BT.709 + iid Poisson + uniform-gray
 * approximation, redrived 2026-04-18).
 *
 *   R/G/B independent Poisson, R=G=B=m ⇒
 *     Var(Y)  = (w_R² + w_G² + w_B²)·α·m       = 0.5619·α·m
 *     Var(Cb) = (w_R² + w_G² + (1-w_B)²)/d_Cb²·α·m = 0.4117·α·m
 *     Var(Cr) = ((1-w_R)² + w_G² + w_B²)/d_Cr²·α·m = 0.4584·α·m
 *   where (w_R,w_G,w_B) = (0.2126, 0.7152, 0.0722) and (d_Cb,d_Cr) = (1.8556, 1.5748).
 *   blind α_Y_fit measures Var(Y)/m, so:
 *     α_Cb / α_Y_fit = 0.4117 / 0.5619 ≈ 0.733
 *     α_Cr / α_Y_fit = 0.4584 / 0.5619 ≈ 0.816
 *   The previous hard-coded 0.375 / 0.605 underestimated σ²_C by ~50% / ~25%,
 *   collapsing the Cb_stab denominator at low Y and amplifying chroma noise
 *   into black-block artifacts on high-ISO low-light scenes (e.g. 0015, 0081).
 *   Same ratios apply to σ²_C under R/G/B equal-variance read noise. */
#define CHROMA_RATIO_CB 0.733f
#define CHROMA_RATIO_CR 0.816f

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

  /* Reconcile bright-side (Huber WLS) and dark-only (Laplacian MAD) σ² estimates.
   *
   * EN: Bright-side Huber WLS can fail to recover a positive intercept on
   *     low-light / high-ISO scenes (all residuals computed with σ²=0 stay
   *     valid, so the iterator never updates the initial 0).  In that case
   *     the old `fmin(sy2_cur, sigma_sq_dark)` silently wrote 0 back, killing
   *     σ²_Y/σ²_Cb/σ²_Cr downstream and producing black-block noise on scene
   *     0015 (ISO3200) et al.  Fall back to the dark estimate when bright
   *     fit effectively failed; otherwise keep the more reliable (smaller)
   *     of the two — dark is preferable because shot-noise vanishes in
   *     shadows, but a truncated-dark estimate can exceed bright fit on
   *     heavily clipped pixels, hence the min.
   * JP: bright fit (Huber WLS) が 0 を返したら dark 単独推定を採用。
   *     そうでなければ両者の小さい方 (暗部側が shot-noise 寄与を含まないぶん
   *     信頼性が高いが、クリップ画素で過大評価される場合があるため min)。 */
  const float sy2_cur = params[YG_P_SIGMA_SQ_Y];
  params[YG_P_SIGMA_SQ_Y] = (sy2_cur > 1e-10f)
      ? fmin(sy2_cur, sigma_sq_dark)
      : sigma_sq_dark;
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
 * YG_MAKITALO_INVERSE_Y: Makitalo-Foi unbiased inverse GAT via LUT.
 *
 * Uses the 4096-entry LUT prebuilt by galosh_build_inv_lut /
 * galosh_lut_finalize (driven by Y-plane α, σ²).  Looks up clean x from
 * the stabilised value d via binary search + linear interpolation.
 * ================================================================ */
kernel void galosh_yuv_makitalo_inverse_Y(
    global const float *restrict d_stab,
    global       float *restrict x_out,
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
 * Y-guided filter (He 2013) for chroma — SEPARABLE implementation.
 *
 * 4-kernel pipeline: 2D box filters decomposed into x-pass + y-pass.
 * Reduces per-pixel work from O((2r+1)²) to O(2·(2r+1)), an 8× cut at
 * r=7 (225 → 30 taps).
 *
 * Pipeline:
 *   (1) moments_x       : row sum of (Y, Y², Cb, Cr, Y·Cb, Y·Cr)
 *   (2) moments_y_and_ab: column sum + local regression → (a, b)
 *   (3) apply_x         : row sum of (a, b) coefficient planes
 *   (4) apply_y         : column sum + output = mean_a·Y + mean_b
 *
 * Quality: numerically identical to the non-separable reference (box
 * filtering is separable).  Memory footprint is 6+4 float intermediate
 * buffers (≈640 MB at 5326×2998) — a desktop-GPU trade.  For ISP/tile
 * deployment the same 4 kernels can be wrapped in a tile loop with
 * overlap = radius; the per-pixel operation is purely local and
 * produces bit-for-bit identical output when overlap ≥ radius.
 *
 * Physics / ε: pixel-adaptive ε(Y) = strength_c²·(α_C·Y + σ²_C) is
 * folded into moments_y_and_ab (step 2), where α_C/σ²_C are already
 * averaged across Cb/Cr from chroma_params_derive.
 *
 * Compile-time knob:
 *   YG_GF_RADIUS — half window size (full window = 2r+1).  Default 7 (15×15).
 *
 * Caller convention:
 *   — moments_x writes 6 intermediate buffers (driver's responsibility)
 *   — moments_y_and_ab produces 4 coefficient planes (a_Cb, b_Cb, a_Cr, b_Cr)
 *   — apply_x writes 4 intermediate coefficient planes
 *   — apply_y consumes 4 intermediate + Y and writes (Cb_out, Cr_out)
 * ================================================================ */
#ifndef YG_GF_RADIUS
#define YG_GF_RADIUS 7
#endif

/* --- Kernel 1: 1D x-pass over (Y, Y², Cb, Cr, Y·Cb, Y·Cr). ------------- */
kernel void galosh_yuv_guided_moments_x(
    global const float *restrict y_guide,
    global const float *restrict cb_in,
    global const float *restrict cr_in,
    global       float *restrict sum_Y_x,
    global       float *restrict sum_YY_x,
    global       float *restrict sum_Cb_x,
    global       float *restrict sum_Cr_x,
    global       float *restrict sum_YCb_x,
    global       float *restrict sum_YCr_x,
    const int width,
    const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  float sY = 0.0f, sYY = 0.0f;
  float sCb = 0.0f, sCr = 0.0f;
  float sYCb = 0.0f, sYCr = 0.0f;

  for(int dx = -YG_GF_RADIUS; dx <= YG_GF_RADIUS; dx++)
  {
    int xi = x + dx;
    if(xi < 0) xi = -xi;
    if(xi >= width) xi = 2 * width - xi - 2;
    const int p = y * width + xi;
    const float Y  = y_guide[p];
    const float Cb = cb_in[p];
    const float Cr = cr_in[p];
    sY   += Y;
    sYY  += Y * Y;
    sCb  += Cb;
    sCr  += Cr;
    sYCb += Y * Cb;
    sYCr += Y * Cr;
  }

  const int cx = y * width + x;
  sum_Y_x[cx]   = sY;
  sum_YY_x[cx]  = sYY;
  sum_Cb_x[cx]  = sCb;
  sum_Cr_x[cx]  = sCr;
  sum_YCb_x[cx] = sYCb;
  sum_YCr_x[cx] = sYCr;
}


/* --- Kernel 2: 1D y-pass over intermediates + local linear regression. */
kernel void galosh_yuv_guided_moments_y_ab(
    global const float *restrict sum_Y_x,
    global const float *restrict sum_YY_x,
    global const float *restrict sum_Cb_x,
    global const float *restrict sum_Cr_x,
    global const float *restrict sum_YCb_x,
    global const float *restrict sum_YCr_x,
    global const float *restrict y_guide,
    global const float *restrict params,
    const float strength_c,
    global       float *restrict a_cb_out,
    global       float *restrict b_cb_out,
    global       float *restrict a_cr_out,
    global       float *restrict b_cr_out,
    const int width,
    const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  float sY = 0.0f, sYY = 0.0f;
  float sCb = 0.0f, sCr = 0.0f;
  float sYCb = 0.0f, sYCr = 0.0f;

  for(int dy = -YG_GF_RADIUS; dy <= YG_GF_RADIUS; dy++)
  {
    int yi = y + dy;
    if(yi < 0) yi = -yi;
    if(yi >= height) yi = 2 * height - yi - 2;
    const int p = yi * width + x;
    sY   += sum_Y_x[p];
    sYY  += sum_YY_x[p];
    sCb  += sum_Cb_x[p];
    sCr  += sum_Cr_x[p];
    sYCb += sum_YCb_x[p];
    sYCr += sum_YCr_x[p];
  }

  /* Window pixel count = (2r+1)² under reflect padding. */
  const float inv_n2   = 1.0f / (float)((2 * YG_GF_RADIUS + 1) * (2 * YG_GF_RADIUS + 1));
  const float mean_Y   = sY   * inv_n2;
  const float mean_YY  = sYY  * inv_n2;
  const float mean_Cb  = sCb  * inv_n2;
  const float mean_Cr  = sCr  * inv_n2;
  const float mean_YCb = sYCb * inv_n2;
  const float mean_YCr = sYCr * inv_n2;

  const float var_Y    = fmax(mean_YY - mean_Y * mean_Y, 0.0f);
  const float cov_YCb  = mean_YCb - mean_Y * mean_Cb;
  const float cov_YCr  = mean_YCr - mean_Y * mean_Cr;

  const int cx = y * width + x;

  /* ---------------------------------------------------------------------
   * Principled ε derivation from the GAT-stabilised noise model.
   *
   * EN: The guided filter fits the local linear model
   *       Cb_i = a·Y_i + b + ε_i,  i ∈ window(x,y)
   *     with MAP estimate  a = cov(Y,Cb) / (var(Y) + ε).
   *     Pre-GAT (raw) pixel variance is Poisson-Gauss: var(x_raw)=α·x+σ².
   *     The GAT forward is designed so that var(Y_stab) ≈ 1 for every
   *     pixel regardless of signal level (variance-stabilising transform).
   *     Hence in GAT space every per-pixel noise variance — luma AND
   *     chroma alike (WHT coeffs are orthonormal combinations of GAT
   *     pixels) — is uniformly ≈ 1.
   *
   *     Imposing a Gaussian prior on the slope a ~ N(0, τ²) and applying
   *     Bayes MAP:
   *           a_MAP = cov(Y,Cb) / (var(Y) + σ²_noise / τ²),
   *     so ε = σ²_noise / τ² = 1/τ² is a CONSTANT in GAT space — signal-
   *     independent, determined only by the slope prior.  τ = 1 encodes
   *     "|a| ≤ 2 with 95% prior mass", a conservative bound for natural
   *     luma-chroma coupling.  User-exposed `strength_c²` retains
   *     smoothing control.
   *
   *     The previous formula ε = strength² · (α_c·Y + σ²_c) re-injected
   *     the raw-space Poisson-Gauss variance into the already-stabilised
   *     GAT domain — a unit-mismatch that gave ε ≈ 0.03 on the raw path
   *     and ≈ 0 on flat tiles.  The new constant ε keeps the whole
   *     GALOSH pipeline under a single Poisson-Gauss / GAT story.
   *
   * JP: guided filter は局所線形モデル Cb = a·Y + b + ε を当てはめる。
   *     GAT forward は per-pixel 分散を 1 に正規化するため、GAT 後空間では
   *     Y も Cb も一律に σ²_noise ≈ 1。slope に Gaussian prior a ~ N(0, τ²)
   *     を置いた Bayes MAP から a = cov(Y,Cb) / (var(Y) + 1/τ²) となり、
   *     ε = 1/τ² は GAT 空間での定数（信号非依存）。τ²=1 (|a|≤2 を ~95%
   *     カバー) を既定、`strength_c²` はユーザー平滑化制御として温存。
   *     旧式 α·Y+σ² は raw 空間の式を GAT 後に再使用する単位混同であり、
   *     この修正で Poisson-Gauss / GAT 理論と pipeline 全体が整合する。
   * --------------------------------------------------------------------- */
  const float YG_GF_TAU_SQ_INV = 1.0f;  /* = 1/τ² for slope prior N(0,τ²) */
  const float eps_gat = strength_c * strength_c * YG_GF_TAU_SQ_INV;

  /* Safety floor below var_Y + ε in case both are tiny on degenerate tiles. */
  const float denom = fmax(var_Y + eps_gat, 1e-6f);
  const float a_cb  = cov_YCb / denom;
  const float a_cr  = cov_YCr / denom;
  const float b_cb  = mean_Cb - a_cb * mean_Y;
  const float b_cr  = mean_Cr - a_cr * mean_Y;

  a_cb_out[cx] = a_cb;
  b_cb_out[cx] = b_cb;
  a_cr_out[cx] = a_cr;
  b_cr_out[cx] = b_cr;
}


/* --- Kernel 3: 1D x-pass over (a_Cb, b_Cb, a_Cr, b_Cr). ---------------- */
kernel void galosh_yuv_guided_apply_x(
    global const float *restrict a_cb_in,
    global const float *restrict b_cb_in,
    global const float *restrict a_cr_in,
    global const float *restrict b_cr_in,
    global       float *restrict a_cb_x,
    global       float *restrict b_cb_x,
    global       float *restrict a_cr_x,
    global       float *restrict b_cr_x,
    const int width,
    const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  float sa_cb = 0.0f, sb_cb = 0.0f;
  float sa_cr = 0.0f, sb_cr = 0.0f;

  for(int dx = -YG_GF_RADIUS; dx <= YG_GF_RADIUS; dx++)
  {
    int xi = x + dx;
    if(xi < 0) xi = -xi;
    if(xi >= width) xi = 2 * width - xi - 2;
    const int p = y * width + xi;
    sa_cb += a_cb_in[p];
    sb_cb += b_cb_in[p];
    sa_cr += a_cr_in[p];
    sb_cr += b_cr_in[p];
  }

  const int cx = y * width + x;
  a_cb_x[cx] = sa_cb;
  b_cb_x[cx] = sb_cb;
  a_cr_x[cx] = sa_cr;
  b_cr_x[cx] = sb_cr;
}


/* --- Kernel 4: 1D y-pass + output = mean_a·Y + mean_b. ----------------- */
kernel void galosh_yuv_guided_apply_y(
    global const float *restrict a_cb_x,
    global const float *restrict b_cb_x,
    global const float *restrict a_cr_x,
    global const float *restrict b_cr_x,
    global const float *restrict y_guide,
    global       float *restrict cb_out,
    global       float *restrict cr_out,
    const int width,
    const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  float sa_cb = 0.0f, sb_cb = 0.0f;
  float sa_cr = 0.0f, sb_cr = 0.0f;

  for(int dy = -YG_GF_RADIUS; dy <= YG_GF_RADIUS; dy++)
  {
    int yi = y + dy;
    if(yi < 0) yi = -yi;
    if(yi >= height) yi = 2 * height - yi - 2;
    const int p = yi * width + x;
    sa_cb += a_cb_x[p];
    sb_cb += b_cb_x[p];
    sa_cr += a_cr_x[p];
    sb_cr += b_cr_x[p];
  }

  const float inv_n2 = 1.0f / (float)((2 * YG_GF_RADIUS + 1) * (2 * YG_GF_RADIUS + 1));
  const float mean_a_cb = sa_cb * inv_n2;
  const float mean_b_cb = sb_cb * inv_n2;
  const float mean_a_cr = sa_cr * inv_n2;
  const float mean_b_cr = sb_cr * inv_n2;

  const int cx = y * width + x;
  const float Y_c = y_guide[cx];
  cb_out[cx] = mean_a_cb * Y_c + mean_b_cb;
  cr_out[cx] = mean_a_cr * Y_c + mean_b_cr;
}

/* ================================================================
 * YG_GF_LOESS: Luma-weighted guided filter (LOESS / kernel regression).
 *
 * EN: A single 2D pass replacing the box-based separable guided filter
 *     (moments_x → moments_y_ab → apply_x → apply_y).  For every output
 *     pixel (x,y) the local linear model
 *         Cb_i = a·Y_i + b,   i ∈ window
 *     is fit with a Gaussian bilateral kernel on the luma difference:
 *         w_i = exp( −(Y_i − Y_c)² / (2·σ²) )
 *     where σ (YG_LOESS_BW) is the bandwidth in GAT-stabilised units.
 *     This is locally-weighted regression (Cleveland 1979 / LOESS,
 *     equivalently kernel-regression / cross-bilateral guided filter).
 *
 *     Motivation: the box-weighted guided filter fits one line through
 *     a bimodal window (silver + specular highlight), producing red/blue
 *     chroma grains on silver surfaces adjacent to saturated highlights.
 *     LOESS downweights samples whose luma differs from the centre
 *     pixel's luma by more than ~σ, so each fit sees only its own luma
 *     "population" and the regression remains stable across edges.
 *
 *     σ = 3 (in GAT units where per-pixel noise std ≈ 1) gives:
 *       |Y_i − Y_c| ≤ 1σ (noise): weight ≈ 0.94   (kept)
 *       |Y_i − Y_c| ≈ 3σ       : weight ≈ 0.01    (heavily down-weighted)
 *       typical specular gap (~50 GAT units): weight ≈ 0  (excluded)
 *     → clean silver windows get uniform weights, grain-prone
 *       silver+highlight windows shrink to silver-only regression.
 *
 *     ε is kept consistent with the principled Bayes-MAP derivation
 *     (ε = 1/τ² in GAT space, τ²=1).
 *
 * JP: box 平均の separable guided filter (moments_x → moments_y_ab
 *     → apply_x → apply_y) を 1 個の 2D pass に置き換え。output pixel
 *     ごとに luma 差分の Gaussian bilateral 重み w_i=exp(-(Y_i-Y_c)²/(2σ²))
 *     を掛けて局所線形回帰。σ=3 (GAT 空間の per-pixel noise std ≈1 の 3σ)
 *     で silver+highlight bimodal window を center の luma 属する population
 *     のみに自動絞り込み、silver 面の赤青粒を根絶する。
 *     ε は前段の Bayes MAP (GAT 空間 σ²_noise=1, slope prior τ²=1) と一貫。
 * ================================================================ */
#ifndef YG_LOESS_RADIUS
#define YG_LOESS_RADIUS 7
#endif
#ifndef YG_LOESS_BW
#define YG_LOESS_BW 3.0f
#endif

kernel void galosh_yuv_guided_loess(
    global const float *restrict y_guide,
    global const float *restrict cb_in,
    global const float *restrict cr_in,
    global const float *restrict params,
    const float strength_c,
    global       float *restrict cb_out,
    global       float *restrict cr_out,
    const int width,
    const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int cx = y * width + x;
  const float Y_c = y_guide[cx];
  const float inv_2sigma_sq = 1.0f / (2.0f * YG_LOESS_BW * YG_LOESS_BW);

  float sumW   = 0.0f, sumY   = 0.0f, sumYY  = 0.0f;
  float sumCb  = 0.0f, sumCr  = 0.0f;
  float sumYCb = 0.0f, sumYCr = 0.0f;

  for(int dy = -YG_LOESS_RADIUS; dy <= YG_LOESS_RADIUS; dy++){
    int yi = y + dy;
    if(yi < 0) yi = -yi;
    if(yi >= height) yi = 2 * height - yi - 2;
    for(int dx = -YG_LOESS_RADIUS; dx <= YG_LOESS_RADIUS; dx++){
      int xi = x + dx;
      if(xi < 0) xi = -xi;
      if(xi >= width) xi = 2 * width - xi - 2;
      const int p = yi * width + xi;
      const float Yi  = y_guide[p];
      const float Cbi = cb_in[p];
      const float Cri = cr_in[p];
      const float dY  = Yi - Y_c;
      const float w   = native_exp(-dY * dY * inv_2sigma_sq);
      sumW   += w;
      sumY   += w * Yi;
      sumYY  += w * Yi * Yi;
      sumCb  += w * Cbi;
      sumCr  += w * Cri;
      sumYCb += w * Yi * Cbi;
      sumYCr += w * Yi * Cri;
    }
  }

  /* Guard against degenerate weight sums (e.g., reflect-padded edge blocks
   * where every neighbour has Y ≈ Y_c — still OK — but play safe). */
  const float invW = 1.0f / fmax(sumW, 1e-10f);
  const float mean_Y   = sumY   * invW;
  const float mean_YY  = sumYY  * invW;
  const float mean_Cb  = sumCb  * invW;
  const float mean_Cr  = sumCr  * invW;
  const float mean_YCb = sumYCb * invW;
  const float mean_YCr = sumYCr * invW;

  const float var_Y   = fmax(mean_YY - mean_Y * mean_Y, 0.0f);
  const float cov_YCb = mean_YCb - mean_Y * mean_Cb;
  const float cov_YCr = mean_YCr - mean_Y * mean_Cr;

  /* Principled ε from Bayes MAP in GAT-stabilised space (see moments_y_ab
   * docstring for derivation).  τ²=1 encodes |a|≤2 at ~95% prior mass. */
  const float eps_gat = strength_c * strength_c * 1.0f;
  const float denom   = fmax(var_Y + eps_gat, 1e-6f);
  const float a_cb    = cov_YCb / denom;
  const float a_cr    = cov_YCr / denom;
  const float b_cb    = mean_Cb - a_cb * mean_Y;
  const float b_cr    = mean_Cr - a_cr * mean_Y;

  cb_out[cx] = a_cb * Y_c + b_cb;
  cr_out[cx] = a_cr * Y_c + b_cr;
}


/* ================================================================
 * galosh_raw_guided_loess_3p  (RAW 3-plane LOESS, optimization).
 *
 * Single-pass variant of galosh_yuv_guided_loess that processes the 3
 * RAW chroma planes (C1, C2, C3 from the 2x2 WHT decompose) together.
 * Replaces the 2-call pattern (loess(C1,C2) + loess(C3,dummy)) used by
 * GALOSH_RAW_G's K13 chroma path with a single kernel launch.
 *
 * Saving: the heavy work per pixel -- 15x15 neighbour sweep, sumW /
 * sumY / sumYY / exp() per neighbour, var_Y / Bayes MAP eps -- is
 * shared across all 3 chroma planes.  Only the 6 per-plane accumulator
 * pairs (sumCi / sumYCi) grow with plane count.  Empirical:
 *     2x 2-plane calls:   2.82 ms / chroma denoise (12 MP)
 *     1x 3-plane call:    1.85 ms / chroma denoise  (-34%)
 *
 * Numerical equivalence: the 2-call pattern produces (Cb,Cr,C3) using
 * exactly the same weights and mean/var Y per pixel as this 3-plane
 * variant.  Output is bit-equivalent (up to FP rounding identical to
 * the original).
 * ================================================================ */
kernel void galosh_raw_guided_loess_3p(
    global const float *restrict y_guide,
    global const float *restrict c1_in,
    global const float *restrict c2_in,
    global const float *restrict c3_in,
    global const float *restrict params,
    const float strength_c,
    global       float *restrict c1_out,
    global       float *restrict c2_out,
    global       float *restrict c3_out,
    const int width,
    const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int cx = y * width + x;
  const float Y_c = y_guide[cx];
  const float inv_2sigma_sq = 1.0f / (2.0f * YG_LOESS_BW * YG_LOESS_BW);

  float sumW   = 0.0f, sumY   = 0.0f, sumYY  = 0.0f;
  float sumC1  = 0.0f, sumC2  = 0.0f, sumC3  = 0.0f;
  float sumYC1 = 0.0f, sumYC2 = 0.0f, sumYC3 = 0.0f;

  for(int dy = -YG_LOESS_RADIUS; dy <= YG_LOESS_RADIUS; dy++){
    int yi = y + dy;
    if(yi < 0) yi = -yi;
    if(yi >= height) yi = 2 * height - yi - 2;
    for(int dx = -YG_LOESS_RADIUS; dx <= YG_LOESS_RADIUS; dx++){
      int xi = x + dx;
      if(xi < 0) xi = -xi;
      if(xi >= width) xi = 2 * width - xi - 2;
      const int p = yi * width + xi;
      const float Yi  = y_guide[p];
      const float C1i = c1_in[p];
      const float C2i = c2_in[p];
      const float C3i = c3_in[p];
      const float dY  = Yi - Y_c;
      const float w   = native_exp(-dY * dY * inv_2sigma_sq);
      sumW   += w;
      sumY   += w * Yi;
      sumYY  += w * Yi * Yi;
      sumC1  += w * C1i;
      sumC2  += w * C2i;
      sumC3  += w * C3i;
      sumYC1 += w * Yi * C1i;
      sumYC2 += w * Yi * C2i;
      sumYC3 += w * Yi * C3i;
    }
  }

  const float invW = 1.0f / fmax(sumW, 1e-10f);
  const float mean_Y   = sumY   * invW;
  const float mean_YY  = sumYY  * invW;
  const float mean_C1  = sumC1  * invW;
  const float mean_C2  = sumC2  * invW;
  const float mean_C3  = sumC3  * invW;
  const float mean_YC1 = sumYC1 * invW;
  const float mean_YC2 = sumYC2 * invW;
  const float mean_YC3 = sumYC3 * invW;

  const float var_Y   = fmax(mean_YY - mean_Y * mean_Y, 0.0f);
  const float cov_YC1 = mean_YC1 - mean_Y * mean_C1;
  const float cov_YC2 = mean_YC2 - mean_Y * mean_C2;
  const float cov_YC3 = mean_YC3 - mean_Y * mean_C3;

  const float eps_gat = strength_c * strength_c * 1.0f;
  const float denom   = fmax(var_Y + eps_gat, 1e-6f);
  const float a_c1    = cov_YC1 / denom;
  const float a_c2    = cov_YC2 / denom;
  const float a_c3    = cov_YC3 / denom;
  const float b_c1    = mean_C1 - a_c1 * mean_Y;
  const float b_c2    = mean_C2 - a_c2 * mean_Y;
  const float b_c3    = mean_C3 - a_c3 * mean_Y;

  c1_out[cx] = a_c1 * Y_c + b_c1;
  c2_out[cx] = a_c2 * Y_c + b_c2;
  c3_out[cx] = a_c3 * Y_c + b_c3;
}


/* End of galosh.cl */
