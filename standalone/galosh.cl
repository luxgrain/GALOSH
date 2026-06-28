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
 *
 * 2026-05-09: cl_khr_fp64 enabled to allow §7.8b/c/d/e dark anchor
 * IRLS to mirror CPU's `double` precision (CPU galosh_raw_cpu.c lines
 * 4684-4775).  Used ONLY in dark IRLS to match CPU's FP64 sum + per-
 * block r/w computation; rest of o32 pipeline stays FP32 (= ISP target
 * spec).  o16 (FP16 production for ISP target) does not use FP64 in
 * its dark IRLS — FP32 there is acceptable since o16 is the
 * streaming-faithful approximation, NOT the CPU faithful mirror.
 * ================================================================
 */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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

  /* 2026-05-10: FP64 (cl_khr_fp64) computation throughout to mirror CPU
   * `gat_build_inverse_table` (galosh_cpu.h line 1180+) exactly.  Previously
   * float-only computation gave d_max = 39.5332 vs CPU's 39.5346 (= 0.0014
   * diff) due to: (1) Poisson break threshold 1e-12f vs CPU 1e-15;
   * (2) sqrt(2) constant precision (1.4142135624 vs 1.4142135623730951);
   * (3) 1/sqrt(pi) precision (0.5641895835 vs 0.5641895835477563);
   * (4) all FP arithmetic in float vs CPU's double.  Effect: dark-image
   * Phase 10 inverse-GAT LUT lookup gave systematic ~0.025 mean|d| pixel
   * shift.  Cost: LUT = 4096 entries built ONCE per image, negligible. */
  const double a = (double)alpha;
  const double sq = (double)sigma_sq;
  const double sig = sqrt(fmax(sq, 1e-20));
  const double y_break = -0.375 * a;
  const double t_break = 2.0 * sig / a;
  const double x_val = (double)i / (double)(GAT_LUT_SIZE - 1);
  const double lambda = x_val / a;

  double expected_gat = 0.0;
  const int k_max = (int)(lambda + 8.0 * sqrt(fmax(lambda, 1.0))) + 20;
  double log_prob = -lambda;

  for(int k = 0; k <= k_max; k++)
  {
    if(k > 0) log_prob += log(lambda) - log((double)k);
    const double prob = exp(log_prob);
    if(prob < 1e-15 && k > (int)lambda + 1) break;

    double eg = 0.0;
    for(int g = 0; g < 10; g++)
    {
      const double z = 1.4142135623730951 * sig * (double)gh_nodes[g];
      const double noisy_y = (double)k * a + z;
      double T;
      if(noisy_y >= y_break)
      {
        const double arg = a * noisy_y + 0.375 * a * a + sq;
        T = (2.0 / a) * sqrt(fmax(arg, 0.0));
      }
      else
        T = t_break + (noisy_y - y_break) / sig;
      eg += (double)gh_wts[g] * T;
    }
    eg *= 0.5641895835477563;  /* 1/sqrt(pi), full double precision */
    expected_gat += prob * eg;
  }

  lut_x[i] = (float)x_val;
  lut_d[i] = (float)expected_gat;
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
    /* Clamp σ_ch at 1e-3 (was 0.01).  Old value was 10x SIDD median σ,
     * over-clamping low-noise RawNIND samples (true σ ~ 0.001-0.005) and
     * causing σ-normalize overshoot → FP16 cascade overflow downstream.
     * 1e-3 keeps inv_sg ≤ 1000 (FP16-safe) while accommodating real low-
     * noise data.  SIDD per-channel σ min = 0.0014 (above clamp), so SIDD
     * results unchanged. */
    float sigma = fmax(mad / 1.6521f, 0.01f);  /* original SIDD-tuned clamp */
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
 * GALOSH_RAW_G K16: per-pixel inverse 2x2 WHT with EWA Jinc-Lanczos-3
 * chroma upsample.
 *
 * Mirrors CPU galosh_raw_cpu.c Phase 4 step (d) (galosh_upsample_2x_ewajl3
 * + per-pixel inverse 2x2 WHT + dark_ref restore + sigma-denormalize +
 * inverse GAT) — all fused into one kernel.  Replaces the legacy
 * `galosh_reconstruct` which used block-replicated half-res chroma (=
 * 2x2 stair-stepping on diagonal edges).
 *
 * EWA-JL3 weights are precomputed in float64 with full j1() and embedded
 * as a __constant 4×5×5 = 100-float table.  Index: si*25 + (dy+2)*5 + (dx+2)
 * where si = sub_y*2 + sub_x is the four 2x upsample sub-pixel offsets:
 *   si=0: (oy=0,   ox=0)    -> block-aligned position (R   in RGGB)
 *   si=1: (oy=0,   ox=0.5)  -> horizontal half-step   (Gr  in RGGB)
 *   si=2: (oy=0.5, ox=0)    -> vertical half-step     (Gb  in RGGB)
 *   si=3: (oy=0.5, ox=0.5)  -> diagonal half-step     (B   in RGGB)
 * Weights sum to exactly 1.0 per si (verified at generation).
 *
 * Per-pixel logic at full-res (fx, fy):
 *   1. Identify Bayer slot ch from (fy&1, fx&1) via RGGB-hardcoded ch_lut
 *      (matches CPU standalone reference; darktable's FC()-aware mapping
 *      is handled host-side — non-RGGB phases fall back to CPU).
 *   2. Sample C1, C2, C3 via 5x5 EWA-JL3 over half-res neighbourhood
 *      around (hx, hy) = (fx>>1, fy>>1).  Border by edge-clamp.
 *   3. val = 0.5 * (L_fr_den[fy,fx] + SIGNS[ch] · (C1, C2, C3))
 *           + dark_ref[ch]
 *      with SIGNS table from inverse 2x2 Walsh-Hadamard:
 *        R : (+1, +1, +1) ; Gb: (-1, +1, -1)
 *        Gr: (+1, -1, -1) ; B : (-1, -1, +1)
 *   4. output[fy,fx] = clamp(gat_inv_lut(val * unified_sigma), 0, 1)
 *
 * Reads per output pixel:
 *   3 × 25 = 75 half-res FP16 chroma reads + 1 full-res FP16 L read.
 *   Cache locality is high: 16×16 work-group → footprint 12×12 half-res
 *   chroma (3.5 KB FP16) → comfortably within L1.  No LDS tiling
 *   required for correctness; could be added later as an optimisation.
 *
 * SIDD Medium 80-pair (CPU bench reference): +0.21 dB PSNR / -7.7% LPIPS
 * vs block-replicated reconstruct.
 * ================================================================ */
__constant float galosh_ewajl3_w[100] = {
  /* si=0 (oy=0.00, ox=0.00) */
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  +0.0000000000e+00f, +1.5198065889e-02f, -4.0430830644e-02f, +1.5198065889e-02f, +0.0000000000e+00f,
  +0.0000000000e+00f, -4.0430830644e-02f, +1.1009310590e+00f, -4.0430830644e-02f, +0.0000000000e+00f,
  +0.0000000000e+00f, +1.5198065889e-02f, -4.0430830644e-02f, +1.5198065889e-02f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  /* si=1 (oy=0.00, ox=0.50) */
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +1.1319775700e-03f, +1.1319775700e-03f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +4.9773604486e-01f, +4.9773604486e-01f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +1.1319775700e-03f, +1.1319775700e-03f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  /* si=2 (oy=0.50, ox=0.00) */
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  +0.0000000000e+00f, +1.1319775700e-03f, +4.9773604486e-01f, +1.1319775700e-03f, +0.0000000000e+00f,
  +0.0000000000e+00f, +1.1319775700e-03f, +4.9773604486e-01f, +1.1319775700e-03f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  /* si=3 (oy=0.50, ox=0.50) */
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +2.5000000000e-01f, +2.5000000000e-01f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +2.5000000000e-01f, +2.5000000000e-01f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f
};

/* Inverse 2x2 WHT signs in canonical (R, Gb, Gr, B) order. SIGNS[ch*3+i]
 * for i = 0..2 → (s1, s2, s3) so val = (L + s1·C1 + s2·C2 + s3·C3) / 2. */
__constant float galosh_chromaup_signs[12] = {
  +1.0f, +1.0f, +1.0f,  /* R  */
  -1.0f, +1.0f, -1.0f,  /* Gb */
  +1.0f, -1.0f, -1.0f,  /* Gr */
  -1.0f, -1.0f, +1.0f   /* B  */
};

/* o32 K16 joint bilateral upsample weights — matches CPU
 * gat_k16_joint_bilateral_upsample's runtime-computed jw[][][] values
 * (NOT galosh_ewajl3_w which is for K16 chromaup with different
 * normalization).  Computed offline via:
 *   jinc(x) = 2*J1(pi*x)/(pi*x)
 *   for each si (oy, ox), each dy/dx in [-2,2]:
 *     ry = dy-oy, rx = dx-ox, r_full = 2*sqrt(rx²+ry²)
 *     w = jinc(r_full) * jinc(r_full/3) if r_full < 3 else 0
 * Values include negative side-lobes; output normalisation by sum_w
 * inside the kernel handles sign correctly. */
__constant float galosh_o32_k16_jbu_w[100] = {
  /* si=0 (oy=0.00, ox=0.00) */
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  +0.0000000000e+00f, +1.3804738955e-02f, -3.6724216895e-02f, +1.3804738955e-02f, +0.0000000000e+00f,
  +0.0000000000e+00f, -3.6724216895e-02f, +1.0000000000e+00f, -3.6724216895e-02f, +0.0000000000e+00f,
  +0.0000000000e+00f, +1.3804738955e-02f, -3.6724216895e-02f, +1.3804738955e-02f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  /* si=1 (oy=0.00, ox=0.50) */
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +3.5811194091e-04f, +3.5811194091e-04f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +1.5746368940e-01f, +1.5746368940e-01f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +3.5811194091e-04f, +3.5811194091e-04f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  /* si=2 (oy=0.50, ox=0.00) */
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  +0.0000000000e+00f, +3.5811194091e-04f, +1.5746368940e-01f, +3.5811194091e-04f, +0.0000000000e+00f,
  +0.0000000000e+00f, +3.5811194091e-04f, +1.5746368940e-01f, +3.5811194091e-04f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  /* si=3 (oy=0.50, ox=0.50) */
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, -7.2646353170e-02f, -7.2646353170e-02f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, -7.2646353170e-02f, -7.2646353170e-02f, +0.0000000000e+00f,
  +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f, +0.0000000000e+00f
};

/* RGGB hardcoded slot lookup: ch_by_si[si] where si = sub_y*2 + sub_x.
 *   si=0 (TL): R  → ch=0
 *   si=1 (TR): Gr → ch=2
 *   si=2 (BL): Gb → ch=1
 *   si=3 (BR): B  → ch=3
 * Non-RGGB phases (BGGR / GRBG / GBRG) are routed to CPU host-side. */
__constant int galosh_chromaup_ch_by_si[4] = { 0, 2, 1, 3 };

kernel void galosh_reconstruct_chromaup(
    global const half  *restrict L_fr_den,    /* full-res FP16 luma   */
    global const half  *restrict c1_out,      /* half-res FP16 chroma */
    global const half  *restrict c2_out,
    global const half  *restrict c3_out,
    global       float *restrict output,      /* full-res FP32 output */
    global const float *restrict lut_d,
    global const float *restrict lut_x,
    global const float *restrict lut_params,
    global const float *restrict params,
    const int width,
    const int height)
{
  const int fx = get_global_id(0);
  const int fy = get_global_id(1);
  if(fx >= width || fy >= height) return;

  const int hw = width  >> 1;
  const int hh = height >> 1;
  const int hx = fx >> 1;
  const int hy = fy >> 1;
  const int sub_x = fx & 1;
  const int sub_y = fy & 1;
  const int si = sub_y * 2 + sub_x;            /* 0..3 */
  const int ch = galosh_chromaup_ch_by_si[si]; /* 0..3 in (R,Gb,Gr,B) */

  /* 5x5 EWA-JL3 sweep over half-res chroma, edge-clamp at borders. */
  float c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
  for(int dy = -2; dy <= 2; dy++)
  {
    int hyi = hy + dy;
    hyi = clamp(hyi, 0, hh - 1);
    for(int dx = -2; dx <= 2; dx++)
    {
      int hxi = hx + dx;
      hxi = clamp(hxi, 0, hw - 1);
      const int hi = hyi * hw + hxi;
      const float w = galosh_ewajl3_w[si * 25 + (dy + 2) * 5 + (dx + 2)];
      if(w == 0.0f) continue;  /* skip zero-weight border samples */
      c1 += w * (float)c1_out[hi];
      c2 += w * (float)c2_out[hi];
      c3 += w * (float)c3_out[hi];
    }
  }

  const float L = (float)L_fr_den[(size_t)fy * width + fx];

  /* Inverse 2x2 WHT + dark_ref restore. params[6..9] = dark_ref[0..3]. */
  const float s1 = galosh_chromaup_signs[ch * 3 + 0];
  const float s2 = galosh_chromaup_signs[ch * 3 + 1];
  const float s3 = galosh_chromaup_signs[ch * 3 + 2];
  const float val = 0.5f * (L + s1 * c1 + s2 * c2 + s3 * c3) + params[6 + ch];

  /* Sigma-denormalize → inverse GAT via LUT.  params[4] = unified_sigma. */
  const float sg = params[4];
  const float dm = lut_params[0], dx_ = lut_params[1];
  const float yb = lut_params[2], tb_ = lut_params[3], sr = lut_params[4];
  const float al = lut_params[5], sq = lut_params[6];

  output[(size_t)fy * width + fx] = clamp(
      gat_inv_lut(val * sg, lut_d, lut_x, dm, dx_, yb, tb_, sr, al, sq),
      0.0f, 1.0f);
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


/* ================================================================
 * §4. Multi-scale chroma kernels (formerly RAW O pipeline).
 *
 * 2026-05-09: 8 RAW-O-specific kernels archived to archived_o-broke/
 *   (= structurally broken, scrap-and-rewrite as o32 / o16 underway).
 * Remaining 6 kernels stay because galosh_yuv_gpu.c (YUV-Q) reuses them.
 *
 * Surviving kernels (= shared with YUV-Q production):
 *   galosh_o_box_downsample_2x_3p           — chroma pyramid step (3-plane)
 *   galosh_o_loess_chroma_3p_fp16_tiled     — LDS-tiled Y-guided LOESS
 *   galosh_o_k16_joint_bilateral_upsample_3p — K16 EWA-JL3 with L bilateral
 *   galosh_o_pad_2d_edge / galosh_o_crop_2d_topleft — boundary correction
 *   galosh_o_smoothstep_blend_3p            — multi-anchor smoothstep walk
 *
 * Archived (= see archived_o-broke/galosh_o_kernels_archived.cl.fragment):
 *   galosh_o_assemble_fullres_gat
 *   galosh_o_forward_l_stride1
 *   galosh_o_chroma_extract_halfres
 *   galosh_o_lpixel_overlap_avg
 *   galosh_o_l_h_den_subsample
 *   galosh_o_box_downsample_2x          (single-channel; _3p version stays)
 *   galosh_o_loess_chroma_3p_fp16       (non-tiled; tiled version stays)
 *   galosh_o_inverse_wht_dark_gat
 *
 * Constraints (still apply to surviving kernels): FP16 throughout,
 * LDS ≤ 32 KB per work-group, GPU-only (no host readbacks).
 * ================================================================ */

#ifndef GALOSH_O_LOESS_R
#define GALOSH_O_LOESS_R 7    /* LOESS half-window radius (matches CPU GALOSH_LOESS_RADIUS) */
#endif

#ifndef GALOSH_O_LOESS_BW
#define GALOSH_O_LOESS_BW 3.0f  /* LOESS bilateral bandwidth on Y guide */
#endif

#ifndef GALOSH_O_K16_BW
#define GALOSH_O_K16_BW 1.5f  /* K16 joint bilateral bandwidth on L_pixel guide */
#endif


/* ================================================================
 * §4.0 Helper: K16 EWA-JL3 jinc weights (= shared with G's K16; defined
 * via galosh_ewajl3_w / galosh_chromaup_signs / galosh_chromaup_ch_by_si
 * constants from earlier in this file).
 * ================================================================ */


/* §4.1-§4.6 K_O0..K_O5 — archived to archived_o-broke/galosh_o_kernels_archived.cl.fragment
 * (assemble_fullres_gat, forward_l_stride1, chroma_extract_halfres,
 *  lpixel_overlap_avg, l_h_den_subsample, box_downsample_2x single-channel)
 */


/* ================================================================
 * §4.7 K_O6 — Phase 7: 3-channel fused 2x2 box-down (chroma pyramid).
 *
 * Same op as K_O5 applied to 3 channels in one launch (= avoids
 * re-reading the source positions 3×).
 * Dispatch: 2D over (sw/2, sh/2).
 * LDS: none.
 * ================================================================ */
kernel void galosh_o_box_downsample_2x_3p(
    global const half *restrict src1,
    global const half *restrict src2,
    global const half *restrict src3,
    global       half *restrict dst1,
    global       half *restrict dst2,
    global       half *restrict dst3,
    const int sw,
    const int sh)
{
  const int dw = sw >> 1;
  const int dh = sh >> 1;
  const int dx = get_global_id(0);
  const int dy = get_global_id(1);
  if(dx >= dw || dy >= dh) return;

  const int sx = 2 * dx;
  const int sy = 2 * dy;
  const size_t p00 = (size_t)sy       * sw + sx;
  const size_t p01 = (size_t)sy       * sw + (sx + 1);
  const size_t p10 = (size_t)(sy + 1) * sw + sx;
  const size_t p11 = (size_t)(sy + 1) * sw + (sx + 1);
  const int dp = dy * dw + dx;
  dst1[dp] = (half)0.25f * (src1[p00] + src1[p01] + src1[p10] + src1[p11]);
  dst2[dp] = (half)0.25f * (src2[p00] + src2[p01] + src2[p10] + src2[p11]);
  dst3[dp] = (half)0.25f * (src3[p00] + src3[p01] + src3[p10] + src3[p11]);
}


/* §4.8 K_O7 (non-tiled loess_chroma_3p_fp16) — archived; tiled version below
 * is kept and preferred (= used by YUV-Q production).  See
 * archived_o-broke/galosh_o_kernels_archived.cl.fragment.
 */


/* ================================================================
 * §4.8a FP16 OVERFLOW FIX for bright scenes:
 *
 * Pure FP16 LOESS with R=7 (= 225-tap window) accumulator sum_YY
 * overflows for bright scenes where Yi values exceed ~30 in normalized
 * GAT space (= Yi² × bilateral_w_eff_count > 65504 FP16 max).
 *
 * Symptom (= GPU O bench 80 scenes): scenes with noisy.max > 0.5 produce
 * PSNR 9-15 dB instead of expected 30-40 dB.  Diagnosed as Yi² overflow:
 *   bright scene: Yi_max ~ 80, Yi² ~ 6400, sum 30 × 6400 = 192k → overflow
 *   dim scene:    Yi_max ~ 22, Yi² ~  484, sum 30 ×  484 =  15k → safe
 *
 * Fix: pre-scale Y_guide and chroma C inputs by 1/LOESS_PRESCALE (= 1/32)
 * inside the LOESS kernel.  Regression math is scale-invariant when ε
 * is scaled by 1/PRESCALE² and final output is multiplied by PRESCALE.
 * Storage + accumulators remain strict FP16 (= no FP32 exception).
 *
 * (日) 明るい scene (= max raw > 0.5) で LOESS の sum_YY が FP16 overflow
 *   (= 65504 上限超え)。 修正: kernel 内で Y/C を 1/32 に pre-scale して
 *   accumulate、 ε と output で再 scale (= 数学的に scale-invariant)。
 *   FP16 strict 維持。
 * ================================================================ */
#define LOESS_PRESCALE        32.0f
#define LOESS_PRESCALE_INV    0.03125f   /* = 1/32 */
#define LOESS_PRESCALE_SQ     1024.0f    /* = 32² */
#define LOESS_PRESCALE_SQ_INV 0.0009765625f  /* = 1/1024 */


/* ================================================================
 * §4.8b K_O7_tiled — LDS-tiled version of galosh_o_loess_chroma_3p_fp16.
 *
 * Same math as K_O7 but with cooperative LDS tile load:
 *   - Each work-group (16×16) loads a (16+2R)×(16+2R) = 30×30 tile of
 *     {Y, C1, C2, C3} into LDS (= 30·30·4·2 bytes = 7.2 KB, fits 32 KB).
 *   - Per-pixel 225-tap sweep reads from LDS instead of global memory.
 *   - Reduces global memory traffic by ~256× per pixel.
 *
 * Reflective boundary handled at tile load (= matches CPU reflect_idx).
 *
 * LDS budget (= 7.2 KB):  4 channels × 30×30 half × 2 bytes = 7,200 bytes
 *
 * Expected speedup vs untiled: 1.5-2× (= memory-bound to compute-bound).
 *
 * (日) LDS タイル化版。 半径 R=7 sweep を 16×16 work-group の cooperative
 *   tile load + per-pixel LDS read で実現、 global memory 帯域を ~256x 削減。
 *   品質は K_O7 と数学的同一。
 * ================================================================ */
/* Tile DIM 24 maximizes interior/halo ratio within 28 KB LDS:
 *   24 + 2×7 = 38 (TILE_W), 38² = 1444, × 4 bufs × 4 bytes = 23.1 KB ✓
 *   Halo ratio: (38² - 24²) / 38² = 60% halo (vs 72% halo for TILE_DIM=16)
 *   Tile count savings: 4MP / 24² vs 4MP / 16² = 56% fewer tiles
 *   WG size: 24² = 576 WIs (NVIDIA SM hosts 1-2 WGs/SM at 1024 thread/SM) */
#define GALOSH_O_LOESS_TILE_DIM   24
#define GALOSH_O_LOESS_TILE_W     (GALOSH_O_LOESS_TILE_DIM + 2 * GALOSH_O_LOESS_R)
#define GALOSH_O_LOESS_TILE_PIX   (GALOSH_O_LOESS_TILE_W * GALOSH_O_LOESS_TILE_W)

kernel void galosh_o_loess_chroma_3p_fp16_tiled(
    global const half *restrict y_guide,
    global const half *restrict c1_in,
    global const half *restrict c2_in,
    global const half *restrict c3_in,
    global       half *restrict c1_out,
    global       half *restrict c2_out,
    global       half *restrict c3_out,
    const int width,
    const int height,
    const float strength_c)
{
  local half tile_Y [GALOSH_O_LOESS_TILE_PIX];
  local half tile_C1[GALOSH_O_LOESS_TILE_PIX];
  local half tile_C2[GALOSH_O_LOESS_TILE_PIX];
  local half tile_C3[GALOSH_O_LOESS_TILE_PIX];

  const int tile_x = get_group_id(0) * GALOSH_O_LOESS_TILE_DIM;
  const int tile_y = get_group_id(1) * GALOSH_O_LOESS_TILE_DIM;
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int lid = ly * GALOSH_O_LOESS_TILE_DIM + lx;
  const int wg_size = GALOSH_O_LOESS_TILE_DIM * GALOSH_O_LOESS_TILE_DIM;

  /* ---- Load tile with R-halo + reflective boundary + pre-scale by 1/32 ---- */
  for(int i = lid; i < GALOSH_O_LOESS_TILE_PIX; i += wg_size)
  {
    const int tx = i % GALOSH_O_LOESS_TILE_W;
    const int ty = i / GALOSH_O_LOESS_TILE_W;
    int gx = tile_x - GALOSH_O_LOESS_R + tx;
    int gy = tile_y - GALOSH_O_LOESS_R + ty;
    if(gx < 0)        gx = -gx;
    if(gx >= width)   gx = 2 * width - gx - 2;
    if(gx < 0)        gx = 0;
    if(gx >= width)   gx = width - 1;
    if(gy < 0)        gy = -gy;
    if(gy >= height)  gy = 2 * height - gy - 2;
    if(gy < 0)        gy = 0;
    if(gy >= height)  gy = height - 1;
    const size_t gp = (size_t)gy * width + gx;
    /* Pre-scale Y and C by 1/32 to prevent FP16 overflow in sum_YY for
     * bright scenes (= Yi_max ~ 80 → Yi/32 ~ 2.5 → Yi²/1024 ~ 6 → sum
     * 30 × 6 = 180 in FP16-safe range).  Output un-scaled by ×32 below. */
    tile_Y [i] = (half)((float)y_guide[gp] * LOESS_PRESCALE_INV);
    tile_C1[i] = (half)((float)c1_in[gp]   * LOESS_PRESCALE_INV);
    tile_C2[i] = (half)((float)c2_in[gp]   * LOESS_PRESCALE_INV);
    tile_C3[i] = (half)((float)c3_in[gp]   * LOESS_PRESCALE_INV);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Per work-item LOESS sweep using LDS reads (= all values pre-scaled) ---- */
  const int ox = tile_x + lx;
  const int oy = tile_y + ly;
  if(ox >= width || oy >= height) return;

  /* Center position in LDS tile: (R + ly, R + lx) */
  const int cx_lds = GALOSH_O_LOESS_R + lx;
  const int cy_lds = GALOSH_O_LOESS_R + ly;
  const half Y_c = tile_Y[cy_lds * GALOSH_O_LOESS_TILE_W + cx_lds];  /* scaled */
  /* BW too in scaled space: dY = (Yi - Y_c) is scaled, so original BW=3.0
   * in unscaled space becomes BW/32 in scaled space → inv_2sigma_sq × 32². */
  const float inv_2sigma_sq = LOESS_PRESCALE_SQ /
                              (2.0f * GALOSH_O_LOESS_BW * GALOSH_O_LOESS_BW);

  half sumW = (half)0.0f, sumY = (half)0.0f, sumYY = (half)0.0f;
  half sumC1 = (half)0.0f, sumC2 = (half)0.0f, sumC3 = (half)0.0f;
  half sumYC1 = (half)0.0f, sumYC2 = (half)0.0f, sumYC3 = (half)0.0f;
  /* GALOSH_CHROMA_CLAMP mirror (= CPU galosh_loess_chroma_3ch_r): full-window
   * input chroma range (in SCALED space) to clamp the degree-1 regression
   * extrapolation below. */
  float cmin1= 1e30f, cmin2= 1e30f, cmin3= 1e30f;
  float cmax1=-1e30f, cmax2=-1e30f, cmax3=-1e30f;

  for(int dy = -GALOSH_O_LOESS_R; dy <= GALOSH_O_LOESS_R; dy++)
  {
    const int ty_lds = cy_lds + dy;
    for(int dx = -GALOSH_O_LOESS_R; dx <= GALOSH_O_LOESS_R; dx++)
    {
      const int tx_lds = cx_lds + dx;
      const int p = ty_lds * GALOSH_O_LOESS_TILE_W + tx_lds;
      const half Yi  = tile_Y [p];
      const half C1i = tile_C1[p];
      const half C2i = tile_C2[p];
      const half C3i = tile_C3[p];
      cmin1=fmin(cmin1,(float)C1i); cmax1=fmax(cmax1,(float)C1i);
      cmin2=fmin(cmin2,(float)C2i); cmax2=fmax(cmax2,(float)C2i);
      cmin3=fmin(cmin3,(float)C3i); cmax3=fmax(cmax3,(float)C3i);
      const half dY  = Yi - Y_c;
      const half w = (half)native_exp(-(float)dY * (float)dY * inv_2sigma_sq);
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

  /* Post-loop reduction in FP32 scalar (= scale-invariant regression).
   * Values are still in scaled space (= ×1/32).  ε is scaled by 1/32²
   * to keep regression equivalent to original.  Final output × 32 to
   * restore original scale.
   *
   * Derivation: if Y_internal = Y/k and C_internal = C/k:
   *   var_Y_int = var_Y/k², cov_YC_int = cov_YC/k², ε_int = ε/k²
   *   a_int = cov_YC_int / (var_Y_int + ε_int) = a (unchanged!)
   *   b_int = mean_C_int - a × mean_Y_int = (mean_C - a mean_Y)/k = b/k
   *   out_int = a × Y_c_int + b_int = (a Y + b)/k = out/k
   * So final output = out_int × k. */
  const float invW = 1.0f / fmax((float)sumW, 1e-10f);
  const float mean_Y_s   = (float)sumY  * invW;   /* scaled */
  const float mean_YY_s  = (float)sumYY * invW;
  const float mean_C1_s  = (float)sumC1 * invW;
  const float mean_C2_s  = (float)sumC2 * invW;
  const float mean_C3_s  = (float)sumC3 * invW;
  const float mean_YC1_s = (float)sumYC1 * invW;
  const float mean_YC2_s = (float)sumYC2 * invW;
  const float mean_YC3_s = (float)sumYC3 * invW;

  const float var_Y_s   = fmax(mean_YY_s - mean_Y_s * mean_Y_s, 0.0f);
  const float cov_YC1_s = mean_YC1_s - mean_Y_s * mean_C1_s;
  const float cov_YC2_s = mean_YC2_s - mean_Y_s * mean_C2_s;
  const float cov_YC3_s = mean_YC3_s - mean_Y_s * mean_C3_s;

  /* ε scaled by 1/32² to maintain scale invariance of a coefficient. */
  const float eps_s = strength_c * strength_c * LOESS_PRESCALE_SQ_INV;
  const float denom = fmax(var_Y_s + eps_s, 1e-6f);
  const float a_c1 = cov_YC1_s / denom;   /* a unchanged from unscaled */
  const float a_c2 = cov_YC2_s / denom;
  const float a_c3 = cov_YC3_s / denom;
  const float b_c1_s = mean_C1_s - a_c1 * mean_Y_s;   /* b/k */
  const float b_c2_s = mean_C2_s - a_c2 * mean_Y_s;
  const float b_c3_s = mean_C3_s - a_c3 * mean_Y_s;
  const float Yc_f_s = (float)Y_c;   /* scaled */

  /* Output in scaled space; clamp the regression extrapolation to the local
   * input chroma band (scaled), then multiply by PRESCALE to restore original. */
  float os1 = a_c1 * Yc_f_s + b_c1_s;
  float os2 = a_c2 * Yc_f_s + b_c2_s;
  float os3 = a_c3 * Yc_f_s + b_c3_s;
  if(cmax1>=cmin1) os1 = clamp(os1, cmin1, cmax1);
  if(cmax2>=cmin2) os2 = clamp(os2, cmin2, cmax2);
  if(cmax3>=cmin3) os3 = clamp(os3, cmin3, cmax3);
  const size_t op = (size_t)oy * width + ox;
  c1_out[op] = (half)(os1 * LOESS_PRESCALE);
  c2_out[op] = (half)(os2 * LOESS_PRESCALE);
  c3_out[op] = (half)(os3 * LOESS_PRESCALE);
}


/* ================================================================
 * §4.9 K_O8 — Phase 7+9: Joint bilateral K16 EWA-JL3 upsample, 3-channel.
 *
 * 2x upsample from (in_w, in_h) chroma to (2·in_w, 2·in_h) full-res chroma
 * with bilateral guidance from L (at output resolution).
 *
 * Per output pixel (fy, fx):
 *   si = (fy & 1) * 2 + (fx & 1)        — sub-pixel index 0..3
 *   For each (dy, dx) in 5x5 (centred at corresponding half-res sample):
 *     w_jinc = jinc(r_full) · jinc(r_full / 3)   (= K16 EWA-JL3)
 *     w_bilat = exp(-(L_pixel[fy_at_h] - L_pixel[fy, fx])² / (2·BW²))
 *     w = w_jinc · w_bilat
 *   C1_out[fy, fx] = Σ w · C1_in[…] / Σ w   (similarly for C2, C3)
 *
 * Per-output-pixel sums in FP16, post-loop division in FP32 scalar.
 *
 * Stride correctness: caller MUST allocate L_pixel guide at exactly
 * (2·in_w, 2·in_h) (= matches output stride).  CPU O ensures this via
 * gat_crop_2d_topleft when source halfwidth is odd.
 *
 * Dispatch: 2D over output dims (2·in_w, 2·in_h).
 * LDS: none.
 * ================================================================ */
kernel void galosh_o_k16_joint_bilateral_upsample_3p(
    global const half *restrict c1_in,        /* half-res input */
    global const half *restrict c2_in,
    global const half *restrict c3_in,
    global const half *restrict L_pixel,      /* full-res L guide (2·in_w × 2·in_h) */
    global       half *restrict c1_out,       /* full-res output */
    global       half *restrict c2_out,
    global       half *restrict c3_out,
    const int in_w,                           /* "halfwidth" relative to the K16 step */
    const int in_h,
    const float bw)
{
  const int out_w = 2 * in_w;
  const int out_h = 2 * in_h;
  const int fx = get_global_id(0);
  const int fy = get_global_id(1);
  if(fx >= out_w || fy >= out_h) return;

  const int hx = fx >> 1;
  const int hy = fy >> 1;
  const int sub_x = fx & 1;
  const int sub_y = fy & 1;
  const int si = sub_y * 2 + sub_x;
  const int ch = galosh_chromaup_ch_by_si[si];
  (void)ch;  /* not needed for this kernel; output is per-channel chroma only */

  const half L_c = L_pixel[(size_t)fy * out_w + fx];
  const float inv_2bw_sq = 1.0f / (2.0f * bw * bw);

  half sum_w  = (half)0.0f;
  half sum_c1 = (half)0.0f, sum_c2 = (half)0.0f, sum_c3 = (half)0.0f;
  /* GALOSH_CHROMA_CLAMP mirror (canonical, = CPU gat_k16_joint_bilateral_upsample):
   * track the FULL-window input chroma range, clamp the jinc-ringing output to it. */
  float cmin1= 1e30f, cmin2= 1e30f, cmin3= 1e30f;
  float cmax1=-1e30f, cmax2=-1e30f, cmax3=-1e30f;

  for(int dy = -2; dy <= 2; dy++)
  {
    int hyi = hy + dy;
    hyi = clamp(hyi, 0, in_h - 1);
    for(int dx = -2; dx <= 2; dx++)
    {
      int hxi = hx + dx;
      hxi = clamp(hxi, 0, in_w - 1);
      const int hi = hyi * in_w + hxi;
      /* full-window input range (every sample, before the zero-weight skip) */
      const float iv1=(float)c1_in[hi], iv2=(float)c2_in[hi], iv3=(float)c3_in[hi];
      cmin1=fmin(cmin1,iv1); cmax1=fmax(cmax1,iv1);
      cmin2=fmin(cmin2,iv2); cmax2=fmax(cmax2,iv2);
      cmin3=fmin(cmin3,iv3); cmax3=fmax(cmax3,iv3);
      const float w_jinc = galosh_ewajl3_w[si * 25 + (dy + 2) * 5 + (dx + 2)];
      if(w_jinc == 0.0f) continue;

      /* Sample L at the half-res chroma sample's full-res TL position */
      int fri = 2 * hyi;
      int fci = 2 * hxi;
      if(fri >= out_h) fri = out_h - 1;
      if(fci >= out_w) fci = out_w - 1;
      const half L_i = L_pixel[(size_t)fri * out_w + fci];
      const half dL = L_i - L_c;
      const float w_bilat = native_exp(-(float)dL * (float)dL * inv_2bw_sq);
      const half w = (half)(w_jinc * w_bilat);

      sum_w  += w;
      sum_c1 += w * c1_in[hi];
      sum_c2 += w * c2_in[hi];
      sum_c3 += w * c3_in[hi];
    }
  }

  /* Post-loop division in FP32 scalar, SIGN-PRESERVING |sum_w| floor (the old
   * fmax(...,1e-10) dropped the sign and flipped chroma when the jinc negative
   * lobes cancelled -> magenta spike). */
  const float swf = (float)sum_w;
  const float safe_sw = (fabs(swf) > 1e-10f) ? swf : ((swf < 0.0f) ? -1e-10f : 1e-10f);
  const float invSw = 1.0f / safe_sw;
  float oc1 = (float)sum_c1 * invSw, oc2 = (float)sum_c2 * invSw, oc3 = (float)sum_c3 * invSw;
  /* clamp jinc-ringing overshoot to the local input chroma band */
  if(cmax1>=cmin1) oc1 = clamp(oc1, cmin1, cmax1);
  if(cmax2>=cmin2) oc2 = clamp(oc2, cmin2, cmax2);
  if(cmax3>=cmin3) oc3 = clamp(oc3, cmin3, cmax3);
  const size_t op = (size_t)fy * out_w + fx;
  c1_out[op] = (half)oc1;
  c2_out[op] = (half)oc2;
  c3_out[op] = (half)oc3;
}


/* ================================================================
 * §4.9b K_O8b — Helper: 2D top-left pad with edge replication.
 *
 * Copies src (sw × sh) into dst (dw × dh) such that dst[y, x]
 * = src[min(y, sh-1), min(x, sw-1)].  Used to bridge K16 raw output
 * dim (= 2·in_w × 2·in_h, may be smaller than chsize when halfwidth
 * is odd) to chsize buffer for smoothstep blend compatibility.
 * (= CPU gat_pad_2d_edge equivalent)
 *
 * Caller guarantees dw >= sw and dh >= sh.
 * Dispatch: 2D over (dw, dh).
 * LDS: none.
 * ================================================================ */
kernel void galosh_o_pad_2d_edge(
    global const half *restrict src,
    global       half *restrict dst,
    const int sw,
    const int sh,
    const int dw,
    const int dh)
{
  const int dx = get_global_id(0);
  const int dy = get_global_id(1);
  if(dx >= dw || dy >= dh) return;

  const int sx = (dx < sw) ? dx : sw - 1;
  const int sy = (dy < sh) ? dy : sh - 1;
  dst[(size_t)dy * dw + dx] = src[(size_t)sy * sw + sx];
}

/* §4.9c K_O8c — Helper: 2D top-left crop (= take upper-left dw × dh
 * of a src buffer at stride sw, write to dst at stride dw).  Inverse
 * of pad_edge for boundary handling.  Used when K16 expects L guide
 * at exactly (2·in_w × 2·in_h) stride but the source buffer is at a
 * larger chsize stride.
 * (= CPU gat_crop_2d_topleft equivalent) */
kernel void galosh_o_crop_2d_topleft(
    global const half *restrict src,
    global       half *restrict dst,
    const int sw,
    const int sh,
    const int dw,
    const int dh)
{
  const int dx = get_global_id(0);
  const int dy = get_global_id(1);
  (void)sh;
  if(dx >= dw || dy >= dh) return;
  dst[(size_t)dy * dw + dx] = src[(size_t)dy * sw + dx];
}


/* ================================================================
 * §4.10 K_O9 — Phase 8: smoothstep slider walk (3-channel).
 *
 * 4 anchor inputs at half-res:
 *   anchor 0 = C_h         (slider = 0,  noisy)
 *   anchor 1 = C_loess_h   (slider = 1,  L baseline)
 *   anchor 2 = C_q_up      (slider = 2,  quarter-res LOESS, ~60 raw px)
 *   anchor 3 = C_e_up      (slider = 3,  eighth-res LOESS, ~120 raw px)
 *
 * Per pixel:
 *   if slider <= 0:        out = C_h
 *   elif slider <= 1:      out = lerp(C_h,        C_loess_h, smoothstep(s    ))
 *   elif slider <= 2:      out = lerp(C_loess_h,  C_q_up,    smoothstep(s - 1))
 *   elif slider <= 3:      out = lerp(C_q_up,     C_e_up,    smoothstep(s - 2))
 *   else:                  out = C_e_up
 *
 * smoothstep(t) = 3t² - 2t³ (= C¹ at t=0,1)
 *
 * Dispatch: 2D over (halfwidth, halfheight).
 * LDS: none.
 * ================================================================ */
kernel void galosh_o_smoothstep_blend_3p(
    global const half *restrict c1_h,         /* anchor 0: noisy */
    global const half *restrict c2_h,
    global const half *restrict c3_h,
    global const half *restrict c1_loess_h,   /* anchor 1: L baseline */
    global const half *restrict c2_loess_h,
    global const half *restrict c3_loess_h,
    global const half *restrict c1_q_up,      /* anchor 2: 60 raw px */
    global const half *restrict c2_q_up,
    global const half *restrict c3_q_up,
    global const half *restrict c1_e_up,      /* anchor 3: 120 raw px */
    global const half *restrict c2_e_up,
    global const half *restrict c3_e_up,
    global       half *restrict c1_out,       /* blended */
    global       half *restrict c2_out,
    global       half *restrict c3_out,
    const int width,
    const int height,
    const float slider)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int p = y * width + x;
  const half a1 = c1_h[p],       a2 = c2_h[p],       a3 = c3_h[p];
  const half b1 = c1_loess_h[p], b2 = c2_loess_h[p], b3 = c3_loess_h[p];
  const half c1 = c1_q_up[p],    c2 = c2_q_up[p],    c3 = c3_q_up[p];
  const half d1 = c1_e_up[p],    d2 = c2_e_up[p],    d3 = c3_e_up[p];

  half o1, o2, o3;
  if(slider <= 0.0f)
  {
    o1 = a1; o2 = a2; o3 = a3;
  }
  else if(slider >= 3.0f)
  {
    o1 = d1; o2 = d2; o3 = d3;
  }
  else
  {
    /* select segment + raw t */
    half lo1, lo2, lo3, hi1, hi2, hi3;
    float t_raw;
    if(slider <= 1.0f)
    {
      t_raw = slider;
      lo1 = a1; lo2 = a2; lo3 = a3;
      hi1 = b1; hi2 = b2; hi3 = b3;
    }
    else if(slider <= 2.0f)
    {
      t_raw = slider - 1.0f;
      lo1 = b1; lo2 = b2; lo3 = b3;
      hi1 = c1; hi2 = c2; hi3 = c3;
    }
    else
    {
      t_raw = slider - 2.0f;
      lo1 = c1; lo2 = c2; lo3 = c3;
      hi1 = d1; hi2 = d2; hi3 = d3;
    }
    /* smoothstep in FP32 scalar (= scalar work, not accumulator). */
    const float t = t_raw * t_raw * (3.0f - 2.0f * t_raw);
    const float oneMt = 1.0f - t;
    o1 = (half)(oneMt * (float)lo1 + t * (float)hi1);
    o2 = (half)(oneMt * (float)lo2 + t * (float)hi2);
    o3 = (half)(oneMt * (float)lo3 + t * (float)hi3);
  }

  c1_out[p] = o1;
  c2_out[p] = o2;
  c3_out[p] = o3;
}


/* §4.11 K_O10 (inverse_wht_dark_gat) — archived; see
 * archived_o-broke/galosh_o_kernels_archived.cl.fragment.
 */


/* End of §4 multi-scale chroma kernels */


/* ================================================================
 * §5 GALOSH_YUV_Q kernels — multi-scale chroma + Laplacian-MAD σ
 *
 * GALOSH_YUV_Q replaces GALOSH_YUV_O's Foi-Alenius (block-based) σ
 * estimator with the Laplacian-MAD estimator used by the CPU path
 * (galosh_yuv_cpu --variant=q).  The MAD output drives both:
 *   - Stage 1: blind α / σ² for GAT (= synthetic α = σ_lin × 0.1)
 *   - Stage 2: post-GAT unified_sigma normalize (= Y_stab /= σ_gat)
 *
 * Single new histogram-based kernel `galosh_yuv_q_lap_mad` services
 * both stages.  Stage 2 normalize is `galosh_yuv_q_unified_sigma_norm`.
 *
 * Chroma multi-scale pyramid (= mirror RAW O Phase 7-9) reuses the
 * existing §4 galosh_o_* kernels with 2-channel input + dummy 3rd
 * channel (= K16/box-downsample/pad/crop/blend are 3p variants).
 *
 * FP16 + LDS ≤ 32KB constraints inherited from RAW O GPU port.
 * ================================================================ */

/* [LATEST: GALOSH_YUV_Q] galosh_yuv_q_lap_mad — Laplacian MAD σ
 * estimator on a single float plane via histogram-based median.
 *
 * Single workgroup of NE_LAPMAD_WG (= 64) work-items collectively
 * scans the input plane at x-stride=3 (= matches CPU's
 * estimate_sigma_plane sampling pattern).  Each WI:
 *   1. Loops over its assigned strip of rows.
 *   2. Computes |Lap| = |row[x] - 2*row[x+2] + row[x+4]| at sampled x.
 *   3. Atomic-increments LDS histogram bin = clamp(|Lap|*scale, 0, B-1).
 * After WG barrier, WI 0 cumsum-scans the histogram, finds the median
 * bin, recovers MAD ≈ bin_center, and writes σ = MAD / 1.6521 to
 * params[result_idx].
 *
 * Histogram: NE_LAPMAD_BINS (= 4096) bins × 4 bytes = 16 KB LDS.
 * Range: [0, NE_LAPMAD_MAX] where NE_LAPMAD_MAX is large enough to
 * contain post-GAT Y_stab (σ ≈ 1) and linear-domain Y (σ ≈ 0.05).
 *
 * Used twice per pipeline:
 *   Stage 1: input = linear Y, output → params[P_YG_SIGMA_Y_LIN]
 *   Stage 2: input = Y_stab (post-GAT), output → params[P_YG_SIGMA_Y_GAT]
 * ================================================================ */
#define NE_LAPMAD_WG       64
#define NE_LAPMAD_BINS     4096
#define NE_LAPMAD_MAX_INV  16.0f   /* bin_idx = clamp(|Lap| * MAX_INV * BINS, 0, BINS-1) */

kernel void galosh_yuv_q_lap_mad(
    global const float *restrict plane,
    const int width,
    const int height,
    const int x_stride,             /* 3 — matches CPU sampling */
    const float lap_max,            /* clip range for histogram (= max |Lap|) */
    const int result_idx,           /* index into params buffer */
    global float *restrict params)
{
  const int lid = get_local_id(0);
  const int wg  = get_local_size(0);  /* NE_LAPMAD_WG */

  local int hist[NE_LAPMAD_BINS];
  local int total_count;

  /* Zero histogram. */
  for(int i = lid; i < NE_LAPMAD_BINS; i += wg) hist[i] = 0;
  if(lid == 0) total_count = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Each WI scans a row stripe.  x-stride=3 gives ~width/3 samples per row. */
  const float bin_scale = (float)NE_LAPMAD_BINS / lap_max;
  int my_count = 0;
  for(int y = lid; y < height; y += wg)
  {
    const int x_end = width - 4;
    for(int x = 0; x <= x_end; x += x_stride)
    {
      const float a = plane[(size_t)y * width + x];
      const float b = plane[(size_t)y * width + x + 2];
      const float c = plane[(size_t)y * width + x + 4];
      const float lap = fabs(a - 2.0f * b + c);
      int bin = (int)(lap * bin_scale);
      if(bin >= NE_LAPMAD_BINS) bin = NE_LAPMAD_BINS - 1;
      atomic_inc(&hist[bin]);
      my_count++;
    }
  }
  atomic_add(&total_count, my_count);
  barrier(CLK_LOCAL_MEM_FENCE);

  /* WI 0: cumsum-scan to find median bin, derive σ. */
  if(lid == 0)
  {
    const int median_target = total_count / 2;
    int cum = 0, median_bin = 0;
    for(int i = 0; i < NE_LAPMAD_BINS; i++)
    {
      cum += hist[i];
      if(cum >= median_target) { median_bin = i; break; }
    }
    const float mad = ((float)median_bin + 0.5f) / bin_scale;
    /* MAD → σ: Var(lap)=6σ² → σ = MAD/(0.6745·sqrt(6)) = MAD/1.6521. */
    const float sigma = fmax(mad / 1.6521f, 0.01f);
    params[result_idx] = sigma;
  }
}

/* [LATEST: GALOSH_YUV_Q] galosh_yuv_q_unified_sigma_norm — in-place
 * per-pixel y *= 1/σ for unit-variance normalization in GAT space.
 *
 * Reads sigma from params[sigma_idx]; broadcasts via local memory for
 * cache friendliness.  Used in Stage 2 (= post-GAT Y_stab normalize)
 * to match CPU's Y_stab[i] *= inv_sg pattern. */
kernel void galosh_yuv_q_unified_sigma_norm(
    global float *restrict plane,
    global const float *restrict params,
    const int sigma_idx,
    const int npix)
{
  const int gid = get_global_id(0);
  if(gid >= npix) return;
  const float sigma = params[sigma_idx];
  const float inv = 1.0f / fmax(sigma, 1e-6f);
  plane[gid] *= inv;
}

/* [LATEST: GALOSH_YUV_Q] galosh_yuv_q_unified_sigma_denorm — in-place
 * per-pixel y *= σ to undo the Stage 2 normalize before inverse GAT.
 * CPU galosh_yuv_cpu does `gat_inverse_exact(Y_den[i] * unified_sigma)`;
 * the GPU pipeline mirrors this by calling denorm immediately before
 * the Makitalo inverse-GAT kernel. */
kernel void galosh_yuv_q_unified_sigma_denorm(
    global float *restrict plane,
    global const float *restrict params,
    const int sigma_idx,
    const int npix)
{
  const int gid = get_global_id(0);
  if(gid >= npix) return;
  const float sigma = params[sigma_idx];
  plane[gid] *= sigma;
}

/* [LATEST: GALOSH_YUV_Q] galosh_yuv_q_synth_alpha_sigma_sq — derive
 * synthetic α and σ² from σ_lin for GAT setup.
 *   α       = max(σ_lin × 0.1, 1e-5)
 *   σ²      = max(σ_lin², 1e-8)
 * Single WI, writes to params[P_ALPHA] and params[P_SIGMA_SQ] (= same
 * indices used by GAT forward / Makitalo inverse downstream). */
kernel void galosh_yuv_q_synth_alpha_sigma_sq(
    global float *restrict params,
    const int sigma_lin_idx,
    const int alpha_idx,
    const int sigma_sq_idx)
{
  if(get_global_id(0) != 0) return;
  const float s = params[sigma_lin_idx];
  const float alpha = fmax(s * 0.1f, 1e-5f);
  const float ssq   = fmax(s * s,    1e-8f);
  params[alpha_idx]    = alpha;
  params[sigma_sq_idx] = ssq;
}

/* End of §5 GALOSH_YUV_Q kernels */


/* ================================================================
 * §6. GALOSH_RAW_O32 kernels — clean FP32 port of CPU O pipeline.
 *
 * 2026-05-09: scrap-and-rewrite of broken §4 RAW O.  o32 = GPU FP32
 * mirror of CPU O (= bit-near reference); o16 = FP16 production
 * (derived from o32 once verified).
 *
 * Numerics policy (= explicit, deviates from §4 FP16-throughout):
 *   - All buffer storage: float (FP32)
 *   - All sum-type accumulators: float (FP32)
 *   - Goal: bit-near equivalence to CPU O (galosh_raw_cpu.c
 *     gat_galosh_denoise_rawlc_o), measured per-Phase via readback +
 *     Python diff against CPU intermediates.
 *
 * Phase coverage (= matches CPU 10-phase pipeline):
 *   Phase 0..2  REUSED from existing G (K0a..K10) — Phase 0 blind α/σ²
 *               + Phase 1 GAT forward + per-CFA σ + RMS unified_sigma +
 *               normalize + Phase 2 dark anchor IRLS.  After K10 we
 *               have ch0..3 (half-res FP32 GAT, post-K6 normalize) and
 *               params[4]=unified_sigma, params[6..9]=ch_dark_ref[4].
 *   Phase 3     galosh_o32_assemble_fullres_gat (= reassemble + dark_sub)
 *               then galosh_o32_forward_l_stride1 (= cycle-spun WHT L).
 *   Phase 4     galosh_o32_chroma_extract_halfres (= stride=2 WHT C).
 *   Phase 5     galosh_pass12_o32 (= FP32 mirror of K13 fused_pass12,
 *               BayesShrink + Pass2 Wiener, single-orient).
 *   Phase 6     galosh_o32_lpixel_overlap_avg + galosh_o32_l_h_den_subsample
 *               (= L_pixel full-res guide + L_h_den half-res guide).
 *   Phase 7     box_downsample_2x + box_downsample_2x_3p + loess_chroma_3p
 *               + k16_joint_bilateral_upsample_3p + pad_2d_edge +
 *               crop_2d_topleft (= multi-scale LOESS pyramid + K16 chain).
 *   Phase 8-10  PENDING (smoothstep blend + final K16 + inverse WHT/dark/GAT).
 *
 * (日) §6: GALOSH_RAW_O32 = CPU O の clean な FP32 GPU 移植。 §4 の
 *   FP16 版 O が壊れていた (PSNR 0.4-0.8 dB on 明るい入力)。 解析よりも
 *   書き直しが速いと判断、 まず o32 で「CPU と数値一致」を取り、
 *   検証後に FP16 化したものを o16 として本番採用する 2 段階計画。
 *   Phase 0-2 は既存 G (K0a..K10) の出力をそのまま流用、 Phase 3 から
 *   分岐する。 各 Phase の中間 (in_gat_full, L_cs, C1/2/3_h, L_cs_den,
 *   L_pixel, L_h_den, etc.) は readback 可能で、 CPU 中間 dump と
 *   diff 取って正しさを担保する。
 * ================================================================ */


/* ================================================================
 * §6.1 galosh_o32_assemble_fullres_gat — Phase 3 (1/2): Reassemble
 *   full-res GAT-domain image from per-CFA half-res sub-channels +
 *   per-pixel CFA-aware dark_ref subtraction.
 *
 *   Input: ch0..3 (half-res FP32 GAT, post-K6 normalize, NOT yet dark-
 *          subbed); params[6..9] = ch_dark_ref[0..3].
 *   Output: in_gat_full (full-res FP32, post normalize + post dark_sub
 *          = same numeric domain as CPU's `in_gat[]` after Phase 2).
 *
 *   Per-pixel (fy, fx):
 *     slot = (fy & 1) | ((fx & 1) << 1)        [CFA slot 0..3]
 *     hp   = (fy/2) * (width/2) + (fx/2)
 *     val  = ch{slot}[hp]
 *     in_gat_full[fy, fx] = val - params[6 + slot]
 *
 * Dispatch: 2D over (width, height).
 * LDS: none.
 *
 * (日) Phase 3 前段。 G 既存 K0a..K10 の出力 ch0..3 (= half-res FP32 GAT,
 *   K6 で unified_sigma 除算済、 dark_ref 未減算) を full-res に再構成し、
 *   per-pixel CFA slot に応じて params[6..9] の dark_ref を減算。 出力は
 *   CPU O Phase 2 末尾の in_gat[] と同じ数値域。
 * ================================================================ */
kernel void galosh_o32_assemble_fullres_gat(
    global const float *restrict ch0,
    global const float *restrict ch1,
    global const float *restrict ch2,
    global const float *restrict ch3,
    global const float *restrict params,        /* [6..9]=ch_dark_ref */
    global       float *restrict in_gat_full,
    const int width,
    const int height)
{
  const int fx = get_global_id(0);
  const int fy = get_global_id(1);
  if(fx >= width || fy >= height) return;

  const int slot_r = fy & 1;
  const int slot_c = fx & 1;
  const int slot   = slot_r | (slot_c << 1);

  const int hw = width >> 1;
  const int hp = (fy >> 1) * hw + (fx >> 1);

  float val;
  if(slot == 0)      val = ch0[hp];
  else if(slot == 1) val = ch1[hp];
  else if(slot == 2) val = ch2[hp];
  else               val = ch3[hp];

  const float dark_ref = params[6 + slot];
  in_gat_full[(size_t)fy * width + fx] = val - dark_ref;
}


/* ================================================================
 * §6.2 galosh_o32_forward_l_stride1 — Phase 3 (2/2): Stride=1
 *   cycle-spun forward 2x2 WHT, L plane only, mirror-padded boundary.
 *   Mirrors CPU `gat_h_forward_l_only_stride1`.
 *
 *   For each pixel (y, x):
 *     yb = mirror(y + 1, height)
 *     xb = mirror(x + 1, width)
 *     a = in_gat_full[y , x ]
 *     b = in_gat_full[yb, x ]
 *     c = in_gat_full[y , xb]
 *     d = in_gat_full[yb, xb]
 *     L_cs[y, x] = (a + b + c + d) * 0.5
 *
 *   Mirror rule (matches CPU galosh_h_mirror_idx): if y+1 >= height
 *   reflect to height-2; if x+1 >= width reflect to width-2.
 *
 *   Scale 0.5 = CPU L_cs convention (= L_cs values are 2× the per-pixel
 *   mean of the 2x2 cycle-spin block).
 *
 * Dispatch: 2D over (width, height).
 * LDS: none.
 *
 * (日) Phase 3 後段。 stride=1 cycle-spun 2x2 WHT の L 成分のみ計算。
 *   境界は mirror reflection (CPU galosh_h_mirror_idx 準拠)。 出力 L_cs は
 *   per-pixel mean の 2× scale (CPU 慣例)。
 * ================================================================ */
kernel void galosh_o32_forward_l_stride1(
    global const float *restrict in_gat_full,
    global       float *restrict L_cs,
    const int width,
    const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  int yb = y + 1;
  if(yb >= height) yb = height - 2;
  if(yb < 0)       yb = 0;
  int xb = x + 1;
  if(xb >= width)  xb = width - 2;
  if(xb < 0)       xb = 0;

  const float a  = in_gat_full[(size_t)y  * width + x ];
  const float b  = in_gat_full[(size_t)yb * width + x ];
  const float cc = in_gat_full[(size_t)y  * width + xb];
  const float d  = in_gat_full[(size_t)yb * width + xb];
  L_cs[(size_t)y * width + x] = 0.5f * (a + b + cc + d);
}


/* ================================================================
 * §6.3 galosh_o32_chroma_extract_halfres — Phase 4: half-res chroma
 *   extract via stride=2 forward 2x2 WHT.  Mirrors CPU
 *   `gat_j_forward_c_halfres`.
 *
 *   For each half-res position (hr, hc):
 *     fr = 2*hr, fc = 2*hx
 *     R  = in_gat[fr,    fc   ]    [TL]
 *     Gb = in_gat[fr+1,  fc   ]    [BL]
 *     Gr = in_gat[fr,    fc+1 ]    [TR]
 *     B  = in_gat[fr+1,  fc+1 ]    [BR]   (assuming RGGB layout)
 *     C1 = (R - Gb + Gr - B) / 2
 *     C2 = (R + Gb - Gr - B) / 2
 *     C3 = (R - Gb - Gr + B) / 2
 *
 *   Boundary: if fr+1 >= height or fc+1 >= width, write 0 (matches CPU
 *   "skip if 2hr+1, 2hc+1 outside").
 *
 * Dispatch: 2D over (halfwidth, halfheight).
 * LDS: none.
 *
 * (日) Phase 4。 stride=2 forward 2x2 WHT の chroma 成分 3 plane 出力。
 *   RGGB CFA を仮定 (= TL=R, BL=Gb, TR=Gr, BR=B)。 境界 (= 2hr+1 / 2hc+1
 *   が外) は 0 (CPU 動作と整合)。
 * ================================================================ */
kernel void galosh_o32_chroma_extract_halfres(
    global const float *restrict in_gat_full,
    global       float *restrict C1_h,
    global       float *restrict C2_h,
    global       float *restrict C3_h,
    const int width,
    const int height,
    const int halfwidth,
    const int halfheight)
{
  const int hx = get_global_id(0);
  const int hy = get_global_id(1);
  if(hx >= halfwidth || hy >= halfheight) return;

  const int fr = 2 * hy;
  const int fc = 2 * hx;
  const int hp = hy * halfwidth + hx;

  if(fr + 1 >= height || fc + 1 >= width)
  {
    C1_h[hp] = 0.0f;
    C2_h[hp] = 0.0f;
    C3_h[hp] = 0.0f;
    return;
  }

  const float R  = in_gat_full[(size_t)fr       * width + fc      ];
  const float Gb = in_gat_full[(size_t)(fr + 1) * width + fc      ];
  const float Gr = in_gat_full[(size_t)fr       * width + (fc + 1)];
  const float B  = in_gat_full[(size_t)(fr + 1) * width + (fc + 1)];

  C1_h[hp] = 0.5f * (R - Gb + Gr - B);
  C2_h[hp] = 0.5f * (R + Gb - Gr - B);
  C3_h[hp] = 0.5f * (R - Gb - Gr + B);
}


/* ================================================================
 * §6.4 helpers: FP32 versions of wht8 / wht2d_8x8 / mad_sigma_y_sq.
 *
 * The §1 helpers (wht8_h / wht2d_8x8_h / mad_sigma_y_sq_h) accept half
 * blocks and intermix half storage with FP32 arithmetic.  The o32 path
 * uses float blocks throughout, so we duplicate the helpers as `_f`
 * variants here (= local to §6, no impact on existing G).
 *
 * (日) §6.4 用 helper。 §1 の half 版は GALOSH_RAW_G で使用継続。 o32 は
 *   FP32 throughout なので float 版を用意。
 * ================================================================ */
inline void wht8_f(float *x)
{
  /* 3-stage 8-point Hadamard butterfly (mirror of §1 wht8 / wht8_h).
   * BUG FIX 2026-05-09: stage 1 (pair butterfly) was missing in initial
   * o32 port — caused Phase 5 pass12_o32 to give 80% mean attenuation
   * (= verified via tools/verify_o32_phases.py which flagged Phase 5 as
   * first divergence vs CPU). */
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

inline void wht2d_8x8_f(float *block, const int normalize)
{
  for(int r = 0; r < 8; r++) wht8_f(block + r * 8);
  for(int c = 0; c < 8; c++)
  {
    float col[8];
    for(int r = 0; r < 8; r++) col[r] = block[r * 8 + c];
    wht8_f(col);
    for(int r = 0; r < 8; r++) block[r * 8 + c] = col[r];
  }
  if(normalize)
  {
    const float inv = 1.0f / 64.0f;
    for(int i = 0; i < 64; i++) block[i] *= inv;
  }
}

inline float mad_sigma_y_sq_f(const float *restrict block)
{
  float abs_ac[GALOSH_BP - 1];   /* 63 */
  for(int i = 1; i < GALOSH_BP; i++) abs_ac[i - 1] = fabs(block[i]);
  const int target = (GALOSH_BP - 1) / 2;   /* 31 */
  for(int k = 0; k <= target; k++)
  {
    int min_idx = k;
    float min_val = abs_ac[k];
    for(int j = k + 1; j < GALOSH_BP - 1; j++)
      if(abs_ac[j] < min_val) { min_val = abs_ac[j]; min_idx = j; }
    abs_ac[min_idx] = abs_ac[k];
    abs_ac[k] = min_val;
  }
  const float mad = abs_ac[target];
  const float sy = mad / 0.6745f;
  return (sy * sy) / (float)GALOSH_BP;
}


/* ================================================================
 * §6.5 galosh_pass12_o32 — Phase 5: FP32 mirror of K13 fused_pass12.
 *
 * Bit-near-equivalent to CPU `galosh_pass12_multiorient_blocked` with
 * block_size=8, stride=2, n_orient=1, use_robust_shrink=1; storage +
 * accumulators are FP32 throughout.
 *
 * 2026-05-09 ADDITION: validity filter inside both Pass1 and Pass2
 *   block loops — skip blocks whose image-space ref is outside CPU's
 *   iteration range [0, dim - BS].  CPU never iterates blocks that
 *   span the image boundary (rmax = height - BS, cmax = width - BS);
 *   without this filter the GPU was processing tile-edge blocks with
 *   zero-padded halo content, producing image-edge artifacts (= rows
 *   0-5 / 252-255 / cols 0-3 PSNR drop on dark SIDD val crops, e.g.
 *   s00_p00 Phase 5 max|d|=1.08).  No DRAM cost (= LDS-only fix), no
 *   extra streaming buffers, no second kernel pass.
 *
 * LDS: tile_in + numer + denom + pilot = 4 × 1600 × 4 B = 25.6 KB
 *   (within 32 KB virtual ISP target).
 *
 * (日) Phase 5 (= L_cs WHT-LOSH single-orient)。 既存 K13 fused_pass12 の
 *   FP32 mirror。 storage + accumulator 全て FP32。 CPU
 *   galosh_pass12_multiorient_blocked (block=8, stride=2, n_orient=1,
 *   robust=1) と数値一致を目標。
 *   2026-05-09 追加: block iteration 内に "image-ref が OOB なら skip"
 *   の validity filter。 CPU の rmax/cmax と等価、 zero-padded halo
 *   block を BayesShrink/Wiener に喰わせない。 LDS / DRAM 増分なし、
 *   streaming structure 不変。
 * ================================================================ */
#ifndef O32_TILE_SIZE
/* Tile size 28: LDS = 4 × 1600 × 4 = 25.6 KB ISP-target, leaves ~6 KB
 * margin within the 32 KB virtual budget.  Halo overhead: 28² interior
 * / 40² total = 49% halo, 51% interior. */
#define O32_TILE_SIZE 28
#endif
#define O32_TILE_W      (O32_TILE_SIZE + 2 * HALO)         /* 40 */
#define O32_TILE_PIXELS (O32_TILE_W * O32_TILE_W)           /* 1600 */

kernel void galosh_pass12_o32(
    global const float *restrict input,
    global       float *restrict output,
    const int width,
    const int height,
    const float sigma_strength,
    const int phase_stride)   /* cycle-spin subsample: 1=16 phases(full), 2=4, 4=1 */
{
  const int tile_x = get_group_id(0) * O32_TILE_SIZE;
  const int tile_y = get_group_id(1) * O32_TILE_SIZE;
  const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
  const int wg_size = get_local_size(0) * get_local_size(1);

  local float tile_in[O32_TILE_PIXELS];
  local float numer  [O32_TILE_PIXELS];
  local float denom  [O32_TILE_PIXELS];
  local float pilot  [O32_TILE_PIXELS];

  /* CPU's iteration range, in image coords: ref ∈ [0, dim - BS]. */
  const int img_rmax = height - GALOSH_BS;
  const int img_cmax = width  - GALOSH_BS;

  /* ---- Step 1: Load input tile (with halo) ---- */
  for(int i = lid; i < O32_TILE_PIXELS; i += wg_size)
  {
    const int lx = i % O32_TILE_W;
    const int ly = i / O32_TILE_W;
    const int gx = tile_x - HALO + lx;
    const int gy = tile_y - HALO + ly;
    tile_in[i] = (gx >= 0 && gx < width && gy >= 0 && gy < height)
                 ? input[(size_t)gy * width + gx] : 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Step 2: Zero accumulators ---- */
  for(int i = lid; i < O32_TILE_PIXELS; i += wg_size)
  {
    numer[i] = 0.0f;
    denom[i] = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Step 3: Pass1 BayesShrink (FP32 throughout) ---- */
  {
    const int n_blocks_dim = (O32_TILE_W - GALOSH_BS) / GALOSH_STRIDE + 1;
    const float sigma_sq = sigma_strength * sigma_strength;
    const float lambda_max_base = sigma_strength * sqrt(2.0f * log((float)GALOSH_BP));

    for(int phase = 0; phase < N_PHASES; phase++)
    {
      const int px = phase % PHASE_MOD;
      const int py = phase / PHASE_MOD;
      if(phase_stride > 1 && ((px % phase_stride) || (py % phase_stride))) continue;  /* cycle-spin subsample */
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

        /* Validity filter: skip blocks whose image-space ref is OOB.
         * Mirrors CPU's `for(ref_r=0; ref_r<=rmax; ref_r+=stride)`. */
        const int img_ref_r = tile_y - HALO + ref_r;
        const int img_ref_c = tile_x - HALO + ref_c;
        if(img_ref_r < 0 || img_ref_r > img_rmax) continue;
        if(img_ref_c < 0 || img_ref_c > img_cmax) continue;

        float block[GALOSH_BP];
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
            block[dy * GALOSH_BS + dx] = tile_in[(ref_r + dy) * O32_TILE_W + (ref_c + dx)];

        wht2d_8x8_f(block, 0);

        const float sigma_y_sq = mad_sigma_y_sq_f(block);
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

        /* Hard threshold (= CPU O semantic, galosh_cpu.h pass1_blocked
         * line 2107).  PRIOR REGRESSION 2026-05-09: this kernel inherited
         * the K13 G-variant soft-threshold (a-λ shrinkage) which created
         * a CPU/GPU algorithmic mismatch — verified via per-Phase harness
         * that GPU o32 pilot at SIDD val s00_p00 dampened by ~0.34/pixel
         * vs CPU O on dark-image content edges, propagating to -1.5 dB
         * SIDD val mean PSNR.  The o32 port spec is a faithful FP32
         * mirror of CPU O, so we restore CPU O's hard-threshold
         * semantics here. */
        int n_nonzero = 1;
        for(int i = 1; i < GALOSH_BP; i++)
        {
          if(fabs(block[i]) < lambda)
            block[i] = 0.0f;
          else
            n_nonzero++;
        }

        wht2d_8x8_f(block, 1);

        const float weight = 1.0f / (float)n_nonzero;
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const float kw = kaiser_1d[dy] * kaiser_1d[dx];
            const float wkw = weight * kw;
            const int pos = (ref_r + dy) * O32_TILE_W + (ref_c + dx);
            numer[pos] += wkw * block[dy * GALOSH_BS + dx];
            denom[pos] += wkw;
          }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  /* ---- Step 4: Finalize Pass1 → pilot.  CPU fallback threshold = 1e-10f. */
  for(int i = lid; i < O32_TILE_PIXELS; i += wg_size)
  {
    const float d = denom[i];
    pilot[i] = (d > 1e-10f) ? (numer[i] / d) : tile_in[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Step 5: Zero accumulators for Pass2 ---- */
  for(int i = lid; i < O32_TILE_PIXELS; i += wg_size)
  {
    numer[i] = 0.0f;
    denom[i] = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Step 6: Pass2 Wiener shrinkage (FP32 throughout) ---- */
  {
    const int n_blocks_dim = (O32_TILE_W - GALOSH_BS) / GALOSH_STRIDE + 1;
    const float sigma_sq_unorm = sigma_strength * sigma_strength * (float)GALOSH_BP;

    for(int phase = 0; phase < N_PHASES; phase++)
    {
      const int px = phase % PHASE_MOD;
      const int py = phase / PHASE_MOD;
      if(phase_stride > 1 && ((px % phase_stride) || (py % phase_stride))) continue;  /* cycle-spin subsample */
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

        /* Validity filter (Pass2): same as Pass1 — match CPU pass2 range. */
        const int img_ref_r = tile_y - HALO + ref_r;
        const int img_ref_c = tile_x - HALO + ref_c;
        if(img_ref_r < 0 || img_ref_r > img_rmax) continue;
        if(img_ref_c < 0 || img_ref_c > img_cmax) continue;

        float blk_noisy[GALOSH_BP];
        float blk_pilot[GALOSH_BP];
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const int pos = (ref_r + dy) * O32_TILE_W + (ref_c + dx);
            blk_noisy[dy * GALOSH_BS + dx] = tile_in[pos];
            blk_pilot[dy * GALOSH_BS + dx] = pilot[pos];
          }

        wht2d_8x8_f(blk_noisy, 0);
        wht2d_8x8_f(blk_pilot, 0);

        float wiener_energy = 0.0f;
        for(int i = 0; i < GALOSH_BP; i++)
        {
          float w;
          if(i == 0)
            w = 1.0f;
          else
          {
            const float p = blk_pilot[i];
            const float s2 = p * p;
            w = s2 / (s2 + sigma_sq_unorm);
            if(w < WIENER_FLOOR) w = WIENER_FLOOR;
          }
          blk_noisy[i] *= w;
          wiener_energy += w * w;
        }

        wht2d_8x8_f(blk_noisy, 1);

        const float weight = 1.0f / fmax(wiener_energy, 1e-6f);
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const float kw = kaiser_1d[dy] * kaiser_1d[dx];
            const float wkw = weight * kw;
            const int pos = (ref_r + dy) * O32_TILE_W + (ref_c + dx);
            numer[pos] += wkw * blk_noisy[dy * GALOSH_BS + dx];
            denom[pos] += wkw;
          }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  /* ---- Step 7: Finalize Pass2 → global output (interior only).
   * CPU fallback to noisy input when no block contributed (line 2343). */
  for(int i = lid; i < O32_TILE_SIZE * O32_TILE_SIZE; i += wg_size)
  {
    const int lx = i % O32_TILE_SIZE + HALO;
    const int ly = i / O32_TILE_SIZE + HALO;
    const int idx = ly * O32_TILE_W + lx;
    const int gx = tile_x + (i % O32_TILE_SIZE);
    const int gy = tile_y + (i / O32_TILE_SIZE);
    if(gx < width && gy < height)
    {
      const float d = denom[idx];
      output[(size_t)gy * width + gx] = (d > 1e-10f) ? (numer[idx] / d) : tile_in[idx];
    }
  }
}


/* ================================================================
 * §6.5d galosh_pass1_o32_dump — DEBUG-ONLY Pass1 extractor.
 *
 * Bit-identical to galosh_pass12_o32 Steps 1-4 (= load tile, zero
 * accumulators, BayesShrink-per-block with validity filter, finalize
 * pilot from numer/denom), but writes the per-tile pilot to a GLOBAL
 * buffer instead of consuming it via in-kernel Pass2.  Production
 * pipeline does NOT use this kernel; host invokes it only when
 * GALOSH_DUMP_DIR is set, so it has zero cost in production runs.
 *
 * Purpose: lets verify_o32_phases.py dump GPU pilot for direct
 * comparison against CPU pilot (which is naturally exposed at the
 * Phase 5 call site as the intermediate buffer between
 * galosh_pass1_blocked and galosh_pass2_blocked).  Localises any
 * Pass1-vs-Pass2 origin of the s00_p00 col-216 divergence.
 *
 * LDS: tile_in + numer + denom = 3 × 1600 × 4 B = 19.2 KB (under
 *   the 32 KB virtual ISP target; debug-only so additional buffer
 *   layout is fine).
 * ================================================================ */
kernel void galosh_pass1_o32_dump(
    global const float *restrict input,
    global       float *restrict pilot_global,
    const int width,
    const int height,
    const float sigma_strength)
{
  const int tile_x = get_group_id(0) * O32_TILE_SIZE;
  const int tile_y = get_group_id(1) * O32_TILE_SIZE;
  const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
  const int wg_size = get_local_size(0) * get_local_size(1);

  local float tile_in[O32_TILE_PIXELS];
  local float numer  [O32_TILE_PIXELS];
  local float denom  [O32_TILE_PIXELS];

  const int img_rmax = height - GALOSH_BS;
  const int img_cmax = width  - GALOSH_BS;

  for(int i = lid; i < O32_TILE_PIXELS; i += wg_size)
  {
    const int lx = i % O32_TILE_W;
    const int ly = i / O32_TILE_W;
    const int gx = tile_x - HALO + lx;
    const int gy = tile_y - HALO + ly;
    tile_in[i] = (gx >= 0 && gx < width && gy >= 0 && gy < height)
                 ? input[(size_t)gy * width + gx] : 0.0f;
  }
  for(int i = lid; i < O32_TILE_PIXELS; i += wg_size)
  {
    numer[i] = 0.0f;
    denom[i] = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  {
    const int n_blocks_dim = (O32_TILE_W - GALOSH_BS) / GALOSH_STRIDE + 1;
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

        const int img_ref_r = tile_y - HALO + ref_r;
        const int img_ref_c = tile_x - HALO + ref_c;
        if(img_ref_r < 0 || img_ref_r > img_rmax) continue;
        if(img_ref_c < 0 || img_ref_c > img_cmax) continue;

        float block[GALOSH_BP];
        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
            block[dy * GALOSH_BS + dx] = tile_in[(ref_r + dy) * O32_TILE_W + (ref_c + dx)];

        wht2d_8x8_f(block, 0);

        const float sigma_y_sq = mad_sigma_y_sq_f(block);
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

        /* Hard threshold (= CPU O semantic, galosh_cpu.h pass1_blocked
         * line 2107).  Same fix as production galosh_pass12_o32. */
        int n_nonzero = 1;
        for(int i = 1; i < GALOSH_BP; i++)
        {
          if(fabs(block[i]) < lambda)
            block[i] = 0.0f;
          else
            n_nonzero++;
        }

        wht2d_8x8_f(block, 1);

        const float weight = 1.0f / (float)n_nonzero;

        for(int dy = 0; dy < GALOSH_BS; dy++)
          for(int dx = 0; dx < GALOSH_BS; dx++)
          {
            const float kw = kaiser_1d[dy] * kaiser_1d[dx];
            const float wkw = weight * kw;
            const int pos = (ref_r + dy) * O32_TILE_W + (ref_c + dx);
            numer[pos] += wkw * block[dy * GALOSH_BS + dx];
            denom[pos] += wkw;
          }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  /* Write tile interior pilot to GLOBAL pilot_global (debug). */
  for(int i = lid; i < O32_TILE_SIZE * O32_TILE_SIZE; i += wg_size)
  {
    const int lx = i % O32_TILE_SIZE + HALO;
    const int ly = i / O32_TILE_SIZE + HALO;
    const int idx = ly * O32_TILE_W + lx;
    const int gx = tile_x + (i % O32_TILE_SIZE);
    const int gy = tile_y + (i / O32_TILE_SIZE);
    if(gx < width && gy < height)
    {
      const float d = denom[idx];
      pilot_global[(size_t)gy * width + gx] =
          (d > 1e-10f) ? (numer[idx] / d) : tile_in[idx];
    }
  }
}


/* ================================================================
 * §6.6 galosh_o32_lpixel_overlap_avg — Phase 6 (1/2): L_pixel = 2x2
 *   overlap-avg of L_cs_den.  Mirrors CPU `gat_i_lpixel_overlap_avg`.
 *
 *   For each output pixel (fy, fx), averages the (up to 4) L_cs_den
 *   values whose 2x2 cycle-spun blocks include (fy, fx):
 *     sum  = L_cs_den[fy, fx]                              (count 1)
 *          + L_cs_den[fy-1, fx]      if fy > 0            (count 2)
 *          + L_cs_den[fy,   fx-1]    if fx > 0            (count 3)
 *          + L_cs_den[fy-1, fx-1]    if fy > 0 && fx > 0  (count 4)
 *     L_pixel[fy, fx] = sum / count
 *
 * Dispatch: 2D over (width, height).
 * LDS: none.
 *
 * (日) Phase 6 前段。 L_cs_den の 2x2 cycle-spun overlap 平均で full-res
 *   L_pixel guide を構築。 Phase 9 の K16 joint bilateral upsample が
 *   この L_pixel を chroma 復元時の bilateral guide として使う。
 * ================================================================ */
kernel void galosh_o32_lpixel_overlap_avg(
    global const float *restrict L_cs_den,
    global       float *restrict L_pixel,
    const int width,
    const int height)
{
  const int fx = get_global_id(0);
  const int fy = get_global_id(1);
  if(fx >= width || fy >= height) return;

  float sum = L_cs_den[(size_t)fy * width + fx];
  int count = 1;
  if(fy > 0)
  {
    sum += L_cs_den[(size_t)(fy - 1) * width + fx];
    count++;
  }
  if(fx > 0)
  {
    sum += L_cs_den[(size_t)fy * width + (fx - 1)];
    count++;
  }
  if(fy > 0 && fx > 0)
  {
    sum += L_cs_den[(size_t)(fy - 1) * width + (fx - 1)];
    count++;
  }
  L_pixel[(size_t)fy * width + fx] = sum / (float)count;
}


/* ================================================================
 * §6.7 galosh_o32_l_h_den_subsample — Phase 6 (2/2): L_h_den = subsample
 *   of L_cs_den at every-other position.  Mirrors CPU L_h_den convention
 *   (see galosh_raw_cpu.c gat_galosh_denoise_rawlc_o lines 4838-4849).
 *
 *   L_h_den[hr, hc] = L_cs_den[2*hr, 2*hc]
 *
 *   Used by Phase 7 LOESS at half-res as the L bilateral guide and as
 *   the Phase 7 K16 destination guide for quarter→half upsample.
 *
 * Dispatch: 2D over (halfwidth, halfheight).
 * LDS: none.
 *
 * (日) Phase 6 後段。 L_cs_den を every-other (= stride=2) で subsample
 *   して half-res L guide を作成。 Phase 7 の half-res LOESS の bilateral
 *   guide および K16 q->h 時の guide として利用。
 * ================================================================ */
kernel void galosh_o32_l_h_den_subsample(
    global const float *restrict L_cs_den,
    global       float *restrict L_h_den,
    const int width,
    const int height,
    const int halfwidth,
    const int halfheight)
{
  const int hx = get_global_id(0);
  const int hy = get_global_id(1);
  if(hx >= halfwidth || hy >= halfheight) return;

  const int fr = 2 * hy;
  const int fc = 2 * hx;
  if(fr >= height || fc >= width)
  {
    L_h_den[hy * halfwidth + hx] = 0.0f;
    return;
  }
  L_h_den[hy * halfwidth + hx] = L_cs_den[(size_t)fr * width + fc];
}


/* ================================================================
 * §6.6+7 fused — per-pixel L_pixel (overlap-avg) + conditional L_h_den
 *   (subsample) in one dispatch.  Saves 1 dispatch overhead + halves
 *   L_cs_den global reads (= each 2x2 quad reads L_cs_den 4 times for
 *   L_pixel overlap, then §6.7 used to re-read 1 of those 4 — now reused).
 *
 * Dispatch: 2D over (width, height).
 * LDS: none.  Quality identical to §6.6+§6.7 separate calls.
 *
 * (日) §6.6 + §6.7 を 1 kernel に融合。 各 WI が L_cs_den[fy,fx] + 3 近傍
 *   を読んで L_pixel 計算、 (fy,fx) が 2x2 の TL のとき同時に L_h_den も
 *   書く。 dispatch オーバヘッド -1、 L_cs_den 読み -25%。
 * ================================================================ */
kernel void galosh_o32_lpixel_lh_den_fused(
    global const float *restrict L_cs_den,
    global       float *restrict L_pixel,
    global       float *restrict L_h_den,
    const int width,
    const int height,
    const int halfwidth)
{
  const int fx = get_global_id(0);
  const int fy = get_global_id(1);
  if(fx >= width || fy >= height) return;

  /* Read own + neighbors once; reuse for L_pixel + L_h_den. */
  const float own = L_cs_den[(size_t)fy * width + fx];

  float sum = own;
  int count = 1;
  if(fy > 0)
  {
    sum += L_cs_den[(size_t)(fy - 1) * width + fx];
    count++;
  }
  if(fx > 0)
  {
    sum += L_cs_den[(size_t)fy * width + (fx - 1)];
    count++;
  }
  if(fy > 0 && fx > 0)
  {
    sum += L_cs_den[(size_t)(fy - 1) * width + (fx - 1)];
    count++;
  }
  L_pixel[(size_t)fy * width + fx] = sum / (float)count;

  /* L_h_den write only for TL of each 2x2 (= matches §6.7 stride=2 sample);
   * reuses 'own' which was already loaded above (= no extra global read). */
  if((fy & 1) == 0 && (fx & 1) == 0)
  {
    L_h_den[(fy >> 1) * halfwidth + (fx >> 1)] = own;
  }
}


/* ================================================================
 * §6.8 galosh_o32_box_downsample_2x — single-channel 2x2 box-down (L pyramid).
 *   dst[hy, hx] = mean of src[2hy, 2hx], src[2hy+1, 2hx],
 *                       src[2hy, 2hx+1], src[2hy+1, 2hx+1]
 *   sw, sh: source dims; output dim = (sw/2, sh/2).
 *   (= CPU gat_box_downsample_2x, FP32 mirror)
 *
 * Dispatch: 2D over (sw/2, sh/2).
 * LDS: none.
 *
 * (日) Phase 7 で L pyramid (L_h_den → L_q → L_e) を構築する 1ch box down。
 * ================================================================ */
kernel void galosh_o32_box_downsample_2x(
    global const float *restrict src,
    global       float *restrict dst,
    const int sw,
    const int sh)
{
  const int dw = sw >> 1;
  const int dh = sh >> 1;
  const int dx = get_global_id(0);
  const int dy = get_global_id(1);
  if(dx >= dw || dy >= dh) return;

  const int sx = 2 * dx;
  const int sy = 2 * dy;
  const float a = src[(size_t)sy       * sw + sx      ];
  const float b = src[(size_t)sy       * sw + (sx + 1)];
  const float c = src[(size_t)(sy + 1) * sw + sx      ];
  const float d = src[(size_t)(sy + 1) * sw + (sx + 1)];
  dst[dy * dw + dx] = 0.25f * (a + b + c + d);
}


/* ================================================================
 * §6.9 galosh_o32_box_downsample_2x_3p — 3-channel fused 2x2 box-down
 *   (chroma pyramid).  Same op as §6.8 applied to 3 channels in one
 *   launch (avoids re-reading the source positions 3×).
 *
 * Dispatch: 2D over (sw/2, sh/2).
 * LDS: none.
 *
 * (日) Phase 7 で chroma pyramid (C{1,2,3}_h → _q → _e) を 1 launch で
 *   構築。 §4 の half 版と同じ計算、 FP32 storage。
 * ================================================================ */
kernel void galosh_o32_box_downsample_2x_3p(
    global const float *restrict src1,
    global const float *restrict src2,
    global const float *restrict src3,
    global       float *restrict dst1,
    global       float *restrict dst2,
    global       float *restrict dst3,
    const int sw,
    const int sh)
{
  const int dw = sw >> 1;
  const int dh = sh >> 1;
  const int dx = get_global_id(0);
  const int dy = get_global_id(1);
  if(dx >= dw || dy >= dh) return;

  const int sx = 2 * dx;
  const int sy = 2 * dy;
  const size_t p00 = (size_t)sy       * sw + sx;
  const size_t p01 = (size_t)sy       * sw + (sx + 1);
  const size_t p10 = (size_t)(sy + 1) * sw + sx;
  const size_t p11 = (size_t)(sy + 1) * sw + (sx + 1);
  const int dp = dy * dw + dx;
  dst1[dp] = 0.25f * (src1[p00] + src1[p01] + src1[p10] + src1[p11]);
  dst2[dp] = 0.25f * (src2[p00] + src2[p01] + src2[p10] + src2[p11]);
  dst3[dp] = 0.25f * (src3[p00] + src3[p01] + src3[p10] + src3[p11]);
}


/* ================================================================
 * §6.10 galosh_o32_loess_chroma_3p — Phase 7: 3-channel Y-guided
 *   bilateral-weighted local linear regression LOESS (FP32).  Mirrors
 *   CPU `galosh_loess_chroma_3ch_r` exactly.
 *
 *   Per output pixel (x, y):
 *     window: (2R+1)×(2R+1) centred on (y, x) (= 225 taps for R=7)
 *     w_i = exp(-(Y_i - Y_c)² / (2 BW²))
 *     accumulate sumW, sumY, sumYY, sumC*, sumYC* in FP32
 *     post-loop: var_Y, cov_YC, a = cov / (var + ε), b = mean_C - a·mean_Y
 *     out = a · Y_c + b
 *   Boundary: reflect (matches CPU reflect_idx).
 *   eps_gat = strength_c² × τ⁻²  (τ⁻² = 1.0)
 *
 *   FP32 vs FP16 §4 version: NO prescale needed (FP32 doesn't overflow
 *   on sum_YY = 225 × Y²; the §4 LOESS_PRESCALE=1/32 was a workaround
 *   for FP16's 65504 max).
 *
 * Args:
 *   y_guide, c1_in, c2_in, c3_in : input planes (FP32, w×h)
 *   c1_out, c2_out, c3_out       : output planes (FP32, w×h)
 *   width, height                : plane dimensions
 *   strength_c                   : LOESS regularization (= 1.0 baseline)
 *   R                            : window radius (= 7 GALOSH default)
 *   BW                           : bilateral bandwidth on Y (= 3.0 default)
 *
 * Dispatch: 2D over (width, height).
 * LDS: none (= per-thread sweep with edge-reflected reads).
 *
 * (日) Phase 7 multi-scale LOESS の単段 (half / quarter / eighth で同じ
 *   kernel を異なる解像度で 3 回呼ぶ)。 FP32 なので prescale 不要、
 *   CPU galosh_loess_chroma_3ch_r と数値一致を取りやすい。
 * ================================================================ */
kernel void galosh_o32_loess_chroma_3p(
    global const float *restrict y_guide,
    global const float *restrict c1_in,
    global const float *restrict c2_in,
    global const float *restrict c3_in,
    global       float *restrict c1_out,
    global       float *restrict c2_out,
    global       float *restrict c3_out,
    const int width,
    const int height,
    const float strength_c,
    const int R,
    const float BW)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int cx = y * width + x;
  const float Y_c = y_guide[cx];
  const float inv_2sigma_sq = 1.0f / (2.0f * BW * BW);

  float sumW = 0.0f, sumY = 0.0f, sumYY = 0.0f;
  float sumC1 = 0.0f, sumC2 = 0.0f, sumC3 = 0.0f;
  float sumYC1 = 0.0f, sumYC2 = 0.0f, sumYC3 = 0.0f;

  for(int dy = -R; dy <= R; dy++)
  {
    /* Reflect: matches CPU reflect_idx (= mirror-reflection at boundary). */
    int yi = y + dy;
    if(yi < 0)        yi = -yi;
    if(yi >= height)  yi = 2 * height - yi - 2;
    if(yi < 0)        yi = 0;
    if(yi >= height)  yi = height - 1;
    const size_t row_off = (size_t)yi * width;
    for(int dx = -R; dx <= R; dx++)
    {
      int xi = x + dx;
      if(xi < 0)       xi = -xi;
      if(xi >= width)  xi = 2 * width - xi - 2;
      if(xi < 0)       xi = 0;
      if(xi >= width)  xi = width - 1;
      const size_t p = row_off + xi;
      const float Yi  = y_guide[p];
      const float C1i = c1_in[p];
      const float C2i = c2_in[p];
      const float C3i = c3_in[p];
      const float dY  = Yi - Y_c;
      /* Use spec-compliant exp (3-4 ULP) instead of native_exp (= fast
       * approximation, several percent error).  CPU
       * galosh_loess_chroma_3ch_r uses libm expf which is IEEE-strict;
       * native_exp diverges algorithmically across (2R+1)² bilateral
       * samples (R=7 → 225 samples per output pixel). */
      const float w   = exp(-dY * dY * inv_2sigma_sq);
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

  const float invW    = 1.0f / fmax(sumW, 1e-10f);
  const float meanY   = sumY  * invW;
  const float meanYY  = sumYY * invW;
  const float meanC1  = sumC1 * invW;
  const float meanC2  = sumC2 * invW;
  const float meanC3  = sumC3 * invW;
  const float meanYC1 = sumYC1 * invW;
  const float meanYC2 = sumYC2 * invW;
  const float meanYC3 = sumYC3 * invW;

  const float var_Y = fmax(meanYY - meanY * meanY, 0.0f);
  const float eps   = strength_c * strength_c;   /* GALOSH_LOESS_TAU_SQ_INV = 1.0 */
  const float denom = fmax(var_Y + eps, 1e-6f);
  const float inv_denom = 1.0f / denom;

  const float a_c1 = (meanYC1 - meanY * meanC1) * inv_denom;
  const float a_c2 = (meanYC2 - meanY * meanC2) * inv_denom;
  const float a_c3 = (meanYC3 - meanY * meanC3) * inv_denom;
  const float b_c1 = meanC1 - a_c1 * meanY;
  const float b_c2 = meanC2 - a_c2 * meanY;
  const float b_c3 = meanC3 - a_c3 * meanY;

  c1_out[cx] = a_c1 * Y_c + b_c1;
  c2_out[cx] = a_c2 * Y_c + b_c2;
  c3_out[cx] = a_c3 * Y_c + b_c3;
}


/* ================================================================
 * §6.10b galosh_o32_loess_chroma_3p_tiled — Phase 7 LDS-tiled FP32
 *   LOESS variant.  Mirrors §4 galosh_o_loess_chroma_3p_fp16_tiled but
 *   FP32 throughout (= no prescale needed since FP32 sum_YY doesn't
 *   overflow on bright scenes).
 *
 *   Algorithm: 16×16 work-group cooperatively loads (16+2R)² = 30² tile
 *   of (Y, C1, C2, C3) into LDS, then each WI does its R=7 sweep
 *   reading from LDS (no global memory reads in inner loop).
 *
 *   Speedup vs naive §6.10: ~16× fewer global memory reads (= 14 reads
 *   per WI instead of 225).  Critical for 4MP+ images where naive LOESS
 *   becomes catastrophically memory-bound (= 225M global reads → ~12 sec).
 *
 *   Constraints:
 *     R = GALOSH_O_LOESS_R = 7 (fixed, matches CPU GALOSH_LOESS_RADIUS)
 *     BW = GALOSH_O_LOESS_BW = 3.0f (fixed)
 *     LDS: 4 buffers × 900 floats × 4 bytes = 14.4 KB.  Fits 32 KB.
 *
 *   Reflective boundary handled at tile load (= matches CPU reflect_idx).
 *
 * (日) §6.10 の LDS タイル版。 R=7 fixed、 FP32 throughout (=  prescale 不要)。
 *   per-WI 225 global reads → tile load 14 reads/WI に削減、 4MP scaling 改善。
 * ================================================================ */
kernel void galosh_o32_loess_chroma_3p_tiled(
    global const float *restrict y_guide,
    global const float *restrict c1_in,
    global const float *restrict c2_in,
    global const float *restrict c3_in,
    global       float *restrict c1_out,
    global       float *restrict c2_out,
    global       float *restrict c3_out,
    const int width,
    const int height,
    const float strength_c)
{
  /* Reuse §4 tile size constants (= same R=7, TILE_DIM=16, TILE_W=30). */
  local float t_Y [GALOSH_O_LOESS_TILE_PIX];
  local float t_C1[GALOSH_O_LOESS_TILE_PIX];
  local float t_C2[GALOSH_O_LOESS_TILE_PIX];
  local float t_C3[GALOSH_O_LOESS_TILE_PIX];

  const int tile_x = get_group_id(0) * GALOSH_O_LOESS_TILE_DIM;
  const int tile_y = get_group_id(1) * GALOSH_O_LOESS_TILE_DIM;
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int lid = ly * GALOSH_O_LOESS_TILE_DIM + lx;
  const int wg_size = GALOSH_O_LOESS_TILE_DIM * GALOSH_O_LOESS_TILE_DIM;

  /* Cooperative tile load with R-halo + reflective boundary.  No prescale. */
  for(int i = lid; i < GALOSH_O_LOESS_TILE_PIX; i += wg_size)
  {
    const int tx = i % GALOSH_O_LOESS_TILE_W;
    const int ty = i / GALOSH_O_LOESS_TILE_W;
    int gx = tile_x - GALOSH_O_LOESS_R + tx;
    int gy = tile_y - GALOSH_O_LOESS_R + ty;
    if(gx < 0)        gx = -gx;
    if(gx >= width)   gx = 2 * width - gx - 2;
    if(gx < 0)        gx = 0;
    if(gx >= width)   gx = width - 1;
    if(gy < 0)        gy = -gy;
    if(gy >= height)  gy = 2 * height - gy - 2;
    if(gy < 0)        gy = 0;
    if(gy >= height)  gy = height - 1;
    const size_t gp = (size_t)gy * width + gx;
    t_Y [i] = y_guide[gp];
    t_C1[i] = c1_in[gp];
    t_C2[i] = c2_in[gp];
    t_C3[i] = c3_in[gp];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const int ox = tile_x + lx;
  const int oy = tile_y + ly;
  if(ox >= width || oy >= height) return;

  const int cx_lds = GALOSH_O_LOESS_R + lx;
  const int cy_lds = GALOSH_O_LOESS_R + ly;
  const float Y_c = t_Y[cy_lds * GALOSH_O_LOESS_TILE_W + cx_lds];
  const float inv_2sigma_sq = 1.0f / (2.0f * GALOSH_O_LOESS_BW * GALOSH_O_LOESS_BW);

  float sumW = 0.0f, sumY = 0.0f, sumYY = 0.0f;
  float sumC1 = 0.0f, sumC2 = 0.0f, sumC3 = 0.0f;
  float sumYC1 = 0.0f, sumYC2 = 0.0f, sumYC3 = 0.0f;
  /* GALOSH_CHROMA_CLAMP mirror (= CPU galosh_loess_chroma_3ch_r): full-window
   * input chroma range to clamp the degree-1 regression extrapolation below. */
  float cmin1= 1e30f, cmin2= 1e30f, cmin3= 1e30f;
  float cmax1=-1e30f, cmax2=-1e30f, cmax3=-1e30f;

  for(int dy = -GALOSH_O_LOESS_R; dy <= GALOSH_O_LOESS_R; dy++)
  {
    const int ty_lds = cy_lds + dy;
    for(int dx = -GALOSH_O_LOESS_R; dx <= GALOSH_O_LOESS_R; dx++)
    {
      const int tx_lds = cx_lds + dx;
      const int p = ty_lds * GALOSH_O_LOESS_TILE_W + tx_lds;
      const float Yi  = t_Y [p];
      const float C1i = t_C1[p];
      const float C2i = t_C2[p];
      const float C3i = t_C3[p];
      cmin1=fmin(cmin1,C1i); cmax1=fmax(cmax1,C1i);
      cmin2=fmin(cmin2,C2i); cmax2=fmax(cmax2,C2i);
      cmin3=fmin(cmin3,C3i); cmax3=fmax(cmax3,C3i);
      const float dY  = Yi - Y_c;
      /* Use spec-compliant exp instead of native_exp (= matches CPU expf). */
      const float w = exp(-dY * dY * inv_2sigma_sq);
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
  const float mean_Y   = sumY  * invW;
  const float mean_YY  = sumYY * invW;
  const float mean_C1  = sumC1 * invW;
  const float mean_C2  = sumC2 * invW;
  const float mean_C3  = sumC3 * invW;
  const float mean_YC1 = sumYC1 * invW;
  const float mean_YC2 = sumYC2 * invW;
  const float mean_YC3 = sumYC3 * invW;

  const float var_Y = fmax(mean_YY - mean_Y * mean_Y, 0.0f);
  const float eps   = strength_c * strength_c;   /* GALOSH_LOESS_TAU_SQ_INV = 1.0 */
  const float denom = fmax(var_Y + eps, 1e-6f);
  const float inv_denom = 1.0f / denom;

  const float a_c1 = (mean_YC1 - mean_Y * mean_C1) * inv_denom;
  const float a_c2 = (mean_YC2 - mean_Y * mean_C2) * inv_denom;
  const float a_c3 = (mean_YC3 - mean_Y * mean_C3) * inv_denom;
  const float b_c1 = mean_C1 - a_c1 * mean_Y;
  const float b_c2 = mean_C2 - a_c2 * mean_Y;
  const float b_c3 = mean_C3 - a_c3 * mean_Y;

  float oc1 = a_c1 * Y_c + b_c1;
  float oc2 = a_c2 * Y_c + b_c2;
  float oc3 = a_c3 * Y_c + b_c3;
  /* clamp degree-1 regression extrapolation to the local input chroma band */
  if(cmax1>=cmin1) oc1 = clamp(oc1, cmin1, cmax1);
  if(cmax2>=cmin2) oc2 = clamp(oc2, cmin2, cmax2);
  if(cmax3>=cmin3) oc3 = clamp(oc3, cmin3, cmax3);
  const size_t op = (size_t)oy * width + ox;
  c1_out[op] = oc1;
  c2_out[op] = oc2;
  c3_out[op] = oc3;
}


/* ================================================================
 * §6.11 galosh_o32_k16_joint_bilateral_upsample_3p — Phase 7 + Phase 9:
 *   Joint bilateral K16 EWA-JL3 upsample, 3-channel, FP32.  Mirrors
 *   CPU `gat_k16_joint_bilateral_upsample`.
 *
 *   Per output pixel (fy, fx):
 *     si = (fy & 1) * 2 + (fx & 1)        sub-pixel index 0..3
 *     For each (dy, dx) in 5x5 (centred at corresponding half-res sample):
 *       w_jinc = jinc(r_full) · jinc(r_full / 3)   (= K16 EWA-JL3)
 *       w_bilat = exp(-(L_at_h - L_c)² / (2·BW²))
 *       w = w_jinc · w_bilat
 *     C_out[fy, fx] = Σ w · C_in[…] / Σ w
 *
 *   Stride correctness: caller MUST allocate L guide at exactly
 *   (2·in_w × 2·in_h) (= matches the K16 internal stride).  CPU O
 *   handles this via gat_crop_2d_topleft when source halfwidth is odd.
 *
 * Dispatch: 2D over output dims (2·in_w, 2·in_h).
 * LDS: none.
 *
 * (日) FP32 K16。 §4 の half 版と同じ jinc 重み定数 (= galosh_ewajl3_w
 *   constant) を再利用、 storage のみ FP32。 Phase 7 の 3 段 K16 chain
 *   (q→h, e→q, q→h) と Phase 9 の最終 half→full upsample で使う。
 * ================================================================ */
kernel void galosh_o32_k16_joint_bilateral_upsample_3p(
    global const float *restrict c1_in,        /* half-res input */
    global const float *restrict c2_in,
    global const float *restrict c3_in,
    global const float *restrict L_pixel,      /* full-res L guide (2·in_w × 2·in_h) */
    global       float *restrict c1_out,       /* full-res output */
    global       float *restrict c2_out,
    global       float *restrict c3_out,
    const int in_w,                            /* "halfwidth" relative to K16 step */
    const int in_h,
    const float bw)
{
  const int out_w = 2 * in_w;
  const int out_h = 2 * in_h;
  const int fx = get_global_id(0);
  const int fy = get_global_id(1);
  if(fx >= out_w || fy >= out_h) return;

  const int hx = fx >> 1;
  const int hy = fy >> 1;
  const int sub_x = fx & 1;
  const int sub_y = fy & 1;
  const int si = sub_y * 2 + sub_x;

  const float L_c = L_pixel[(size_t)fy * out_w + fx];
  const float inv_2bw_sq = 1.0f / (2.0f * bw * bw);

  float sum_w  = 0.0f;
  float sum_c1 = 0.0f, sum_c2 = 0.0f, sum_c3 = 0.0f;
  /* Anti-ringing (canonical, = CPU gat_k16_joint_bilateral_upsample): clamp the
   * jinc side-lobe overshoot to the convex hull of the NEAREST 2x2 source chroma
   * samples (ImageMagick/mpv "+clamp" antiring) -> removes the ring-magenta GALOSH
   * made from clean input at high-contrast L edges while preserving jinc sharpness.
   * Same nearest-2x2 mask as r32/i16 (sub_y=fy&1, sub_x=fx&1 = si). */
  float cmin1= 1e30f, cmin2= 1e30f, cmin3= 1e30f;
  float cmax1=-1e30f, cmax2=-1e30f, cmax3=-1e30f;

  for(int dy = -2; dy <= 2; dy++)
  {
    int hyi = hy + dy;
    hyi = clamp(hyi, 0, in_h - 1);
    for(int dx = -2; dx <= 2; dx++)
    {
      int hxi = hx + dx;
      hxi = clamp(hxi, 0, in_w - 1);
      const int hi = hyi * in_w + hxi;
      /* nearest-2x2 source-sample hull (anti-ringing), before the zero-weight skip */
      if(((dy==0)||(sub_y&&dy==1)) && ((dx==0)||(sub_x&&dx==1))) {
        const float iv1=c1_in[hi], iv2=c2_in[hi], iv3=c3_in[hi];
        cmin1=fmin(cmin1,iv1); cmax1=fmax(cmax1,iv1);
        cmin2=fmin(cmin2,iv2); cmax2=fmax(cmax2,iv2);
        cmin3=fmin(cmin3,iv3); cmax3=fmax(cmax3,iv3);
      }
      const float w_jinc = galosh_o32_k16_jbu_w[si * 25 + (dy + 2) * 5 + (dx + 2)];
      if(w_jinc == 0.0f) continue;

      /* Sample L at the half-res chroma sample's full-res TL position. */
      int fri = 2 * hyi;
      int fci = 2 * hxi;
      if(fri >= out_h) fri = out_h - 1;
      if(fci >= out_w) fci = out_w - 1;
      const float L_i = L_pixel[(size_t)fri * out_w + fci];
      const float dL = L_i - L_c;
      /* Use spec-compliant exp instead of native_exp (= matches CPU bilateral
       * weight in gat_k16_joint_bilateral_upsample). */
      const float w_bilat = exp(-dL * dL * inv_2bw_sq);
      const float w = w_jinc * w_bilat;

      sum_w  += w;
      sum_c1 += w * c1_in[hi];
      sum_c2 += w * c2_in[hi];
      sum_c3 += w * c3_in[hi];
    }
  }

  /* SIGN-PRESERVING |sum_w| floor (the old fmax-style positive floor flipped the
   * chroma sign when the jinc negative lobes cancelled -> magenta spike). */
  const float safe_w = (fabs(sum_w) > 1e-6f) ? sum_w : ((sum_w < 0.0f) ? -1e-6f : 1e-6f);
  const float inv_w = 1.0f / safe_w;
  float oc1 = sum_c1 * inv_w, oc2 = sum_c2 * inv_w, oc3 = sum_c3 * inv_w;
  /* clamp jinc-ringing overshoot to the local input chroma band */
  if(cmax1>=cmin1) oc1 = clamp(oc1, cmin1, cmax1);
  if(cmax2>=cmin2) oc2 = clamp(oc2, cmin2, cmax2);
  if(cmax3>=cmin3) oc3 = clamp(oc3, cmin3, cmax3);
  const size_t op = (size_t)fy * out_w + fx;
  c1_out[op] = oc1;
  c2_out[op] = oc2;
  c3_out[op] = oc3;
}


/* ================================================================
 * §6.12 galosh_o32_pad_2d_edge — 2D top-left pad with edge replication.
 *   Copies src (sw × sh) into dst (dw × dh) such that
 *     dst[y, x] = src[min(y, sh-1), min(x, sw-1)].
 *   Caller guarantees dw >= sw and dh >= sh.
 *
 * Dispatch: 2D over (dw, dh). LDS: none.
 *
 * (日) Phase 7 で K16 raw 出力 (= 2·in_w × 2·in_h) を chsize buffer
 *   stride に edge-replicate pad (smoothstep blend 互換のため)。
 * ================================================================ */
kernel void galosh_o32_pad_2d_edge(
    global const float *restrict src,
    global       float *restrict dst,
    const int sw,
    const int sh,
    const int dw,
    const int dh)
{
  const int dx = get_global_id(0);
  const int dy = get_global_id(1);
  if(dx >= dw || dy >= dh) return;
  const int sx = (dx < sw) ? dx : sw - 1;
  const int sy = (dy < sh) ? dy : sh - 1;
  dst[(size_t)dy * dw + dx] = src[(size_t)sy * sw + sx];
}


/* ================================================================
 * §6.13 galosh_o32_crop_2d_topleft — 2D top-left crop.  Inverse of
 *   pad_edge: takes upper-left dw × dh of src at stride sw, writes to
 *   dst at stride dw.
 *
 * Dispatch: 2D over (dw, dh). LDS: none.
 *
 * (日) Phase 7 で chsize stride の L_h_den / L_q を K16 stride matching
 *   のため (2·in_w × 2·in_h) に top-left crop。
 * ================================================================ */
kernel void galosh_o32_crop_2d_topleft(
    global const float *restrict src,
    global       float *restrict dst,
    const int sw,
    const int sh,
    const int dw,
    const int dh)
{
  const int dx = get_global_id(0);
  const int dy = get_global_id(1);
  (void)sh;
  if(dx >= dw || dy >= dh) return;
  dst[(size_t)dy * dw + dx] = src[(size_t)dy * sw + dx];
}


/* ================================================================
 * §6.14 galosh_o32_smoothstep_blend_3p — Phase 8: cubic smoothstep
 *   slider walk over 4 anchors at half-res, FP32.  Mirrors CPU
 *   galosh_raw_cpu.c gat_galosh_denoise_rawlc_o lines 5100-5187.
 *
 *   Anchors:
 *     0: C_h        (slider = 0, noisy)
 *     1: C_loess_h  (slider = 1, L baseline LOESS at half-res)
 *     2: C_q_up     (slider = 2, ~60 raw px receptive field)
 *     3: C_e_up     (slider = 3, ~120 raw px receptive field)
 *
 *   Per pixel:
 *     slider <= 0:  out = C_h
 *     slider in (0,1]: t = smoothstep(s);     out = lerp(C_h, C_loess_h, t)
 *     slider in (1,2]: t = smoothstep(s-1);   out = lerp(C_loess_h, C_q_up, t)
 *     slider in (2,3]: t = smoothstep(s-2);   out = lerp(C_q_up, C_e_up, t)
 *     slider >= 3:  out = C_e_up
 *   smoothstep(t) = t² (3 - 2t)  (= C¹ at t=0,1, no "click" at integer slider)
 *
 * Dispatch: 2D over (halfwidth, halfheight).  LDS: none.
 *
 * (日) Phase 8 slider walk。 整数値で C¹ 連続な smoothstep blend、
 *   slider 値で受容野を 0 → 30 → 60 → 120 raw px に walk させる。
 *   chroma blotch (大粒度色むら) を slider で確実に reach できる O 設計。
 * ================================================================ */
kernel void galosh_o32_smoothstep_blend_3p(
    global const float *restrict c1_h,         /* anchor 0: noisy */
    global const float *restrict c2_h,
    global const float *restrict c3_h,
    global const float *restrict c1_loess_h,   /* anchor 1: L baseline */
    global const float *restrict c2_loess_h,
    global const float *restrict c3_loess_h,
    global const float *restrict c1_q_up,      /* anchor 2: 60 raw px */
    global const float *restrict c2_q_up,
    global const float *restrict c3_q_up,
    global const float *restrict c1_e_up,      /* anchor 3: 120 raw px */
    global const float *restrict c2_e_up,
    global const float *restrict c3_e_up,
    global       float *restrict c1_out,
    global       float *restrict c2_out,
    global       float *restrict c3_out,
    const int width,
    const int height,
    const float slider)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int p = y * width + x;
  const float a1 = c1_h[p],       a2 = c2_h[p],       a3 = c3_h[p];
  const float b1 = c1_loess_h[p], b2 = c2_loess_h[p], b3 = c3_loess_h[p];
  const float cc1 = c1_q_up[p],   cc2 = c2_q_up[p],   cc3 = c3_q_up[p];
  const float d1 = c1_e_up[p],    d2 = c2_e_up[p],    d3 = c3_e_up[p];

  float o1, o2, o3;
  if(slider <= 0.0f)
  {
    o1 = a1; o2 = a2; o3 = a3;
  }
  else if(slider >= 3.0f)
  {
    o1 = d1; o2 = d2; o3 = d3;
  }
  else
  {
    float lo1, lo2, lo3, hi1, hi2, hi3, t_raw;
    if(slider <= 1.0f)
    {
      t_raw = slider;
      lo1 = a1;  lo2 = a2;  lo3 = a3;
      hi1 = b1;  hi2 = b2;  hi3 = b3;
    }
    else if(slider <= 2.0f)
    {
      t_raw = slider - 1.0f;
      lo1 = b1;  lo2 = b2;  lo3 = b3;
      hi1 = cc1; hi2 = cc2; hi3 = cc3;
    }
    else
    {
      t_raw = slider - 2.0f;
      lo1 = cc1; lo2 = cc2; lo3 = cc3;
      hi1 = d1;  hi2 = d2;  hi3 = d3;
    }
    const float t = t_raw * t_raw * (3.0f - 2.0f * t_raw);
    const float oneMt = 1.0f - t;
    o1 = oneMt * lo1 + t * hi1;
    o2 = oneMt * lo2 + t * hi2;
    o3 = oneMt * lo3 + t * hi3;
  }

  c1_out[p] = o1;
  c2_out[p] = o2;
  c3_out[p] = o3;
}


/* ================================================================
 * §6.15 galosh_o32_inverse_wht_dark_gat — Phase 10: fused per-pixel
 *   inverse 2x2 WHT + dark_ref restore + ×unified_sigma + inverse GAT
 *   (LUT).  Final output FP32.  Mirrors CPU lines 5217-5247.
 *
 *   Per output pixel (fy, fx):
 *     slot = (fy & 1) | ((fx & 1) << 1)        CFA slot 0..3
 *     ch   = galosh_chromaup_ch_by_si[slot]    R/Gb/Gr/B index
 *     s_k  = galosh_chromaup_signs[ch * 3 + k]  for k=0,1,2
 *     val = 0.5 * (L_pixel + s0*C1 + s1*C2 + s2*C3) + dark_ref[slot]
 *     out = inv_GAT_LUT(val * unified_sigma)
 *
 * Args:
 *   L_pixel, C1/2/3_aligned : full-res FP32
 *   output                  : final FP32 in [0, 1]
 *   lut_d, lut_x, lut_params : inverse GAT LUT (= built by K2/K3)
 *   params                  : [4]=unified_sigma, [6..9]=ch_dark_ref
 *
 * Dispatch: 2D over (width, height).  LDS: none.
 *
 * (日) Phase 10 最終再構成。 per-pixel CFA slot ごとに L+C 線形結合 → 0.5x →
 *   dark_ref 加算 → unified_sigma 乗算 → inverse GAT LUT で最終 [0,1] 出力。
 *   既存 G の K16 reconstruct と同じ数式、 storage のみ FP32。
 * ================================================================ */
kernel void galosh_o32_inverse_wht_dark_gat(
    global const float *restrict L_pixel,
    global const float *restrict C1_aligned,
    global const float *restrict C2_aligned,
    global const float *restrict C3_aligned,
    global       float *restrict output,
    global const float *restrict lut_d,
    global const float *restrict lut_x,
    global const float *restrict lut_params,
    global const float *restrict params,
    const int width,
    const int height)
{
  const int fx = get_global_id(0);
  const int fy = get_global_id(1);
  if(fx >= width || fy >= height) return;

  const int slot_r = fy & 1;
  const int slot_c = fx & 1;
  const int slot = slot_r | (slot_c << 1);

  const size_t p = (size_t)fy * width + fx;
  const float L  = L_pixel[p];
  const float C1 = C1_aligned[p];
  const float C2 = C2_aligned[p];
  const float C3 = C3_aligned[p];

  /* BUG FIX 2026-05-10: previously used `ch = galosh_chromaup_ch_by_si[slot]`
   * (= {0, 2, 1, 3}) then `signs[ch * 3 + k]`, which SWAPPED Gr (slot 2)
   * and Gb (slot 1) signs vs CPU.  CPU `gat_galosh_denoise_rawlc_o`
   * Phase 10 (galosh_raw_cpu.c lines 5252-5279) uses inline SIGNS[slot][k]
   * directly without ch remapping.  Mirror that here. */
  const float s1 = galosh_chromaup_signs[slot * 3 + 0];
  const float s2 = galosh_chromaup_signs[slot * 3 + 1];
  const float s3 = galosh_chromaup_signs[slot * 3 + 2];

  const float dark_ref = params[6 + slot];
  const float val = 0.5f * (L + s1 * C1 + s2 * C2 + s3 * C3) + dark_ref;

  const float sg = params[4];                /* unified_sigma */
  const float dm = lut_params[0], dx_ = lut_params[1];
  const float yb = lut_params[2], tb_ = lut_params[3], sr = lut_params[4];
  const float al = lut_params[5], sq = lut_params[6];

  output[p] = clamp(
      gat_inv_lut(val * sg, lut_d, lut_x, dm, dx_, yb, tb_, sr, al, sq),
      0.0f, 1.0f);
}


/* End of §6 GALOSH_RAW_O32 kernels (Phase 3-10 = full pipeline implemented). */


/* ================================================================
 * §7. GALOSH_RAW_O32 Phase 0-2 kernels — blind noise estimation +
 *     GAT forward + per-CFA σ + unified normalize + dark anchor IRLS.
 *
 * 2026-05-09: Phase 0-2 ported from CPU galosh_raw_cpu.c (galosh_estimate_noise
 * + gat_build_inverse_table + Phase 1 GAT + Phase 2 dark IRLS) to FP32 GPU
 * kernels.  Replaces previous mistaken reuse of G's K0a..K10, which used a
 * different blind estimator and produced α/σ² values incompatible with
 * CPU O (= structural divergence: G α=0.01 vs CPU α=0.000462 on real raw).
 *
 * Phase 0 design:
 *   §7.1 galosh_o32_ne_block_stats    (per-block mean + Laplacian MAD variance)
 *   §7.2 galosh_o32_ne_finalize       (bin envelope + WLS Huber + dark refine)
 *   §7.3 galosh_o32_build_inv_lut     (Poisson + Gauss-Hermite LUT, per-entry parallel)
 * Phase 1-2 implementation continues in §7.4+ (next session).
 *
 * (日) §7: Phase 0-2 を CPU から GPU に正しく FP32 移植。 §6 で既存 G 流用
 *   していたのを廃止 (CPU O と数値が違うため "完全移植" にならなかった)。
 *   GPU only pipeline policy 厳守、 CPU 関数を呼ばない。
 * ================================================================ */

#define O32_NE_BLOCK_SZ 8
#define O32_NE_NBINS    32

/* Params buffer indices (mirror galosh_gpu.h host-side defines).  Must
 * stay in sync. */
#ifndef P_SIGMA_CH0
#define P_SIGMA_CH0      0
#endif
#ifndef P_SIGMA_CH1
#define P_SIGMA_CH1      1
#endif
#ifndef P_SIGMA_CH2
#define P_SIGMA_CH2      2
#endif
#ifndef P_SIGMA_CH3
#define P_SIGMA_CH3      3
#endif
#ifndef P_UNIFIED_SIGMA
#define P_UNIFIED_SIGMA  4
#endif
#ifndef P_INV_SG
#define P_INV_SG         5
#endif
#ifndef P_ALPHA
#define P_ALPHA          13
#endif
#ifndef P_SIGMA_SQ
#define P_SIGMA_SQ       14
#endif


/* ================================================================
 * §7.1 galosh_o32_ne_block_stats — Phase 0 (1/3): per-block statistics.
 *
 * Mirrors CPU galosh_estimate_noise Step 1-2 (line 637-692).  For each
 * (channel, block_y, block_x) compute:
 *   blk_mean = mean of 64 pixels in 8x8 half-res block (CFA-aware stride 2)
 *   blk_var  = (MAD(|H+V Laplacians|) / 0.6745)² / 6
 *
 * Laplacian: |v0 - 2*v1 + v2| over 3 consecutive samples in same CFA channel.
 * 96 Laplacians per block (= 48 horizontal + 48 vertical).
 *
 * Dispatch: 1 WI per block (= 4 channels × n_bx × n_by total blocks).
 *           Workgroup 64×1.
 *
 * (日) ブロックごとに mean と Laplacian MAD ベースの noise variance を計算。
 *   Foi+Alenius "lower envelope" 法の Step 1。 1 WI = 1 block。
 * ================================================================ */
inline float partial_select_median_o32_ne(float *arr, int n)
{
  /* Find element at rank n/2 (lower median) via partial selection sort. */
  const int target = n / 2;
  for(int k = 0; k <= target; k++)
  {
    int min_idx = k;
    float min_val = arr[k];
    for(int j = k + 1; j < n; j++)
      if(arr[j] < min_val) { min_val = arr[j]; min_idx = j; }
    arr[min_idx] = arr[k];
    arr[k] = min_val;
  }
  return arr[target];
}

kernel void galosh_o32_ne_block_stats(
    global const float *restrict raw,
    const int width,
    const int height,
    const int n_bx,
    const int n_by,
    const int n_blocks_per_ch,
    global float *restrict blk_mean,
    global float *restrict blk_var)
{
  const int bi_global = get_global_id(0);
  const int total_blocks = 4 * n_blocks_per_ch;
  if(bi_global >= total_blocks) return;

  /* Decode (ch, by, bx). */
  const int ch       = bi_global / n_blocks_per_ch;
  const int bi_in_ch = bi_global - ch * n_blocks_per_ch;
  const int by       = bi_in_ch / n_bx;
  const int bx       = bi_in_ch - by * n_bx;
  /* CFA offsets {{0,0},{0,1},{1,0},{1,1}}. */
  const int dy0 = (ch >> 1) & 1;
  const int dx0 = ch & 1;

  const int y0 = by * O32_NE_BLOCK_SZ;
  const int x0 = bx * O32_NE_BLOCK_SZ;

  /* Step 1: block mean of 64 pixels. */
  float sum = 0.0f;
  for(int y = y0; y < y0 + O32_NE_BLOCK_SZ; y++)
    for(int x = x0; x < x0 + O32_NE_BLOCK_SZ; x++)
      sum += raw[(size_t)(2 * y + dy0) * width + (2 * x + dx0)];
  const float bm = sum / (float)(O32_NE_BLOCK_SZ * O32_NE_BLOCK_SZ);

  /* Step 2: collect 96 Laplacians (48 H + 48 V) within the block. */
  float laps[96];
  int nl = 0;
  /* Horizontal: 8 rows × 6 (= BS-2) per row. */
  for(int y = y0; y < y0 + O32_NE_BLOCK_SZ; y++)
    for(int x = x0; x < x0 + O32_NE_BLOCK_SZ - 2; x++)
    {
      const float v0 = raw[(size_t)(2 * y + dy0) * width + (2 *  x      + dx0)];
      const float v1 = raw[(size_t)(2 * y + dy0) * width + (2 * (x + 1) + dx0)];
      const float v2 = raw[(size_t)(2 * y + dy0) * width + (2 * (x + 2) + dx0)];
      laps[nl++] = fabs(v0 - 2.0f * v1 + v2);
    }
  /* Vertical: 6 (= BS-2) rows × 8 per row. */
  for(int y = y0; y < y0 + O32_NE_BLOCK_SZ - 2; y++)
    for(int x = x0; x < x0 + O32_NE_BLOCK_SZ; x++)
    {
      const float v0 = raw[(size_t)(2 *  y      + dy0) * width + (2 * x + dx0)];
      const float v1 = raw[(size_t)(2 * (y + 1) + dy0) * width + (2 * x + dx0)];
      const float v2 = raw[(size_t)(2 * (y + 2) + dy0) * width + (2 * x + dx0)];
      laps[nl++] = fabs(v0 - 2.0f * v1 + v2);
    }

  /* MAD via partial-selection median, then sigma_lap = MAD / 0.6745,
   * blk_var = sigma_lap² / 6 (Var of 3-tap Laplacian on iid noise = 6σ²). */
  const float med = partial_select_median_o32_ne(laps, nl);
  const float sigma_lap = med / 0.6745f;
  blk_var[bi_global]  = (sigma_lap * sigma_lap) / 6.0f;
  blk_mean[bi_global] = bm;
}


/* ================================================================
 * §7.2 galosh_o32_ne_finalize — Phase 0 (2/3): single-WG sequential
 *   finalize pass.  Mirrors CPU galosh_estimate_noise Steps 3-5
 *   (line 695-879): bin envelope → WLS Huber → dark refine.
 *
 * Input: blk_mean[total_blocks], blk_var[total_blocks] from §7.1.
 * Output: params[P_ALPHA], params[P_SIGMA_SQ].
 *
 * Note: this kernel is dispatched as a single WG of NE_FIN_WG (= 32) WIs.
 * Most heavy work (= sorting per-bin variances via partial selection) is
 * parallelized across WIs by bin index.  Sequential parts (WLS Huber +
 * dark refine) executed by WI 0.  Runs once per image so perf is not
 * critical (= O(n_total · NBINS) = ~125k ops, < 1 ms even single-WI).
 *
 * (日) Phase 0 後段。 32-bin に分割して下方 envelope (5-20%ile)、
 *   WLS Huber 5 iter、 暗部 pixel から σ² 補正。 single WG 32 WI で
 *   per-bin sort 並列化、 sequential 部分は WI 0 が処理。
 * ================================================================ */
#define NE_FIN_WG 32
#define NE_FIN_VAR_BINS 128   /* per-bin var histogram resolution */

kernel void galosh_o32_ne_finalize(
    global const float *restrict blk_mean,
    global const float *restrict blk_var,
    global const float *restrict raw,
    const int width,
    const int height,
    const int total_blocks,
    global float *restrict params)
{
  const int lid = get_local_id(0);
  const int wg  = get_local_size(0);

  /* Per-bin shared state. */
  local float bin_mean_arr[O32_NE_NBINS];
  local float bin_var_arr [O32_NE_NBINS];
  local int   bin_cnt_arr [O32_NE_NBINS];
  local int   bin_valid   [O32_NE_NBINS];
  local float global_min, global_max;
  local int   n_total_valid;
  local int lds_var_hist[O32_NE_NBINS * NE_FIN_VAR_BINS];   /* 16 KB */
  /* PARALLELIZED 2026-06-21: was launched 32×32 (1 WI = 1 bin, each WI
   * re-scanned ALL blocks twice = 12.8 ms @4K).  Now launch 256×256:
   *   - global min/max via full-WG reduction
   *   - tpb = 256/32 = 8 threads cooperate per mean-bin: split the block
   *     scan 8 ways, LDS-reduce the stats, atomic_inc the var-histogram.
   *   - percentile + WLS unchanged.  FP reduction order differs from the
   *     serial path, so alpha/σ² shift by FP rounding only (o32 = bit-near,
   *     quality-validated unchanged).  ~8× → ~1.6 ms. */
  const int tpb = wg / O32_NE_NBINS;     /* threads per bin (= 8 at lws 256) */
  const int bin = lid / tpb;             /* 0..O32_NE_NBINS-1 */
  const int sub = lid % tpb;             /* 0..tpb-1 */
  local float l_msum[O32_NE_NBINS], l_vmin[O32_NE_NBINS], l_vmax[O32_NE_NBINS];
  local int   l_cnt [O32_NE_NBINS];
  local int   r_cnt[256];
  local float r_a[256], r_b[256], r_c[256];   /* reduction scratch (msum/vmin/vmax) */

  /* ---- global min/max/nvalid via full-WG reduction ---- */
  {
    float gmin = FLT_MAX, gmax = 0.0f; int nv = 0;
    for(int i = lid; i < total_blocks; i += wg)
    {
      const float bm = blk_mean[i], bv = blk_var[i];
      if(bm > 0.003f && bm < 0.97f && bv < 1e9f)
      { gmin = fmin(gmin, bm); gmax = fmax(gmax, bm); nv++; }
    }
    r_b[lid] = gmin; r_c[lid] = gmax; r_cnt[lid] = nv;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int s = wg >> 1; s > 0; s >>= 1)
    {
      if(lid < s)
      { r_b[lid] = fmin(r_b[lid], r_b[lid + s]); r_c[lid] = fmax(r_c[lid], r_c[lid + s]);
        r_cnt[lid] += r_cnt[lid + s]; }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0) { global_min = r_b[0]; global_max = r_c[0]; n_total_valid = r_cnt[0]; }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  /* Early bail: nothing usable. */
  if(global_max - global_min < 1e-10f)
  {
    if(lid == 0) { params[P_ALPHA] = 1e-4f; params[P_SIGMA_SQ] = 1e-6f; }
    return;
  }

  const float bw     = (global_max - global_min) / (float)O32_NE_NBINS;
  const float bin_lo = global_min + (float)bin * bw;
  const float bin_hi = bin_lo + bw;

  /* ---- Pass1: per-bin cnt/msum/vmin/vmax (tpb threads cooperate per bin) ---- */
  {
    float msum = 0.0f, vmin = FLT_MAX, vmax = 0.0f; int cnt = 0;
    for(int i = sub; i < total_blocks; i += tpb)
    {
      const float bm = blk_mean[i], bv = blk_var[i];
      if(bm >= bin_lo && bm < bin_hi && bm > 0.003f && bm < 0.97f && bv < 1e9f)
      { msum += bm; cnt++; vmin = fmin(vmin, bv); vmax = fmax(vmax, bv); }
    }
    r_cnt[lid] = cnt; r_a[lid] = msum; r_b[lid] = vmin; r_c[lid] = vmax;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int s = tpb >> 1; s > 0; s >>= 1)
    {
      if(sub < s)
      { const int o = lid + s; r_cnt[lid] += r_cnt[o]; r_a[lid] += r_a[o];
        r_b[lid] = fmin(r_b[lid], r_b[o]); r_c[lid] = fmax(r_c[lid], r_c[o]); }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(sub == 0)
    { l_cnt[bin] = r_cnt[lid]; l_msum[bin] = r_a[lid]; l_vmin[bin] = r_b[lid]; l_vmax[bin] = r_c[lid]; }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const int   cnt  = l_cnt[bin];
  const float msum = l_msum[bin];
  const float vmin = l_vmin[bin];
  const float vmax = l_vmax[bin];
  const float vrange = fmax(vmax - vmin, 1e-12f);
  const float vscale = (float)NE_FIN_VAR_BINS / vrange;

  /* clear this bin's var-histogram (tpb threads) */
  for(int i = sub; i < NE_FIN_VAR_BINS; i += tpb)
    lds_var_hist[bin * NE_FIN_VAR_BINS + i] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- Pass2: per-bin var histogram (tpb threads, atomic_inc) ---- */
  if(cnt >= 20)
  {
    volatile local int *vhist = &lds_var_hist[bin * NE_FIN_VAR_BINS];
    for(int i = sub; i < total_blocks; i += tpb)
    {
      const float bm = blk_mean[i], bv = blk_var[i];
      if(bm >= bin_lo && bm < bin_hi && bm > 0.003f && bm < 0.97f && bv < 1e9f)
      {
        int vbin = (int)((bv - vmin) * vscale);
        if(vbin < 0) vbin = 0;
        if(vbin >= NE_FIN_VAR_BINS) vbin = NE_FIN_VAR_BINS - 1;
        atomic_inc(&vhist[vbin]);
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- percentile + bin_mean/var (sub==0 owns each bin) ---- */
  if(sub == 0)
  {
    if(cnt < 20) { bin_valid[bin] = 0; }
    else
    {
      local int *vhist = &lds_var_hist[bin * NE_FIN_VAR_BINS];
      const int p5_target  = cnt / 20;
      const int p20_target = cnt / 5;
      int cum = 0, p5_bin = 0, p20_bin = NE_FIN_VAR_BINS - 1, found_p5 = 0;
      for(int i = 0; i < NE_FIN_VAR_BINS; i++)
      {
        cum += vhist[i];
        if(!found_p5 && cum >= p5_target) { p5_bin = i; found_p5 = 1; }
        if(cum >= p20_target) { p20_bin = i; break; }
      }
      float vsum = 0.0f; int vcnt = 0;
      for(int i = p5_bin; i <= p20_bin; i++)
      {
        const float bin_center = vmin + ((float)i + 0.5f) / vscale;
        const int n = vhist[i];
        vsum += bin_center * (float)n; vcnt += n;
      }
      bin_var_arr[bin]  = (vcnt > 0) ? (vsum / (float)vcnt) : (vmin + 0.5f / vscale);
      bin_mean_arr[bin] = msum / (float)cnt;
      bin_cnt_arr[bin]  = (vcnt > 0) ? vcnt : 1;
      bin_valid[bin]    = 1;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* WLS Huber regression (5 iterations) — single-WI sequential. */
  if(lid != 0) return;

  int n_valid = 0;
  for(int b = 0; b < O32_NE_NBINS; b++) if(bin_valid[b]) n_valid++;

  if(n_valid < 4)
  {
    params[P_ALPHA]    = 1e-4f;
    params[P_SIGMA_SQ] = 1e-6f;
    return;
  }

  float alpha_est    = 0.01f;
  float sigma_sq_est = 0.0f;

  for(int iter = 0; iter < 5; iter++)
  {
    float huber_k = 1e10f;
    if(iter > 0)
    {
      /* Compute residual MAD (= per-bin |bv - (alpha*bm + sq)|). */
      float resids[O32_NE_NBINS];
      int nr = 0;
      for(int b = 0; b < O32_NE_NBINS; b++)
      {
        if(!bin_valid[b]) continue;
        resids[nr++] = fabs(bin_var_arr[b] - (alpha_est * bin_mean_arr[b] + sigma_sq_est));
      }
      /* Median via partial selection. */
      const int target_r = nr / 2;
      for(int k = 0; k <= target_r; k++)
      {
        int min_idx = k;
        float min_val = resids[k];
        for(int j = k + 1; j < nr; j++)
          if(resids[j] < min_val) { min_val = resids[j]; min_idx = j; }
        resids[min_idx] = resids[k];
        resids[k] = min_val;
      }
      const float resid_mad = resids[target_r] / 0.6745f;
      huber_k = 1.345f * fmax(resid_mad, 1e-12f);
    }

    /* WLS accumulators. */
    float Sw = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    for(int b = 0; b < O32_NE_NBINS; b++)
    {
      if(!bin_valid[b]) continue;
      float w = (float)bin_cnt_arr[b];
      if(iter > 0)
      {
        const float pred  = alpha_est * bin_mean_arr[b] + sigma_sq_est;
        const float resid = fabs(bin_var_arr[b] - pred);
        if(resid > huber_k) w *= huber_k / resid;
      }
      const float x = bin_mean_arr[b], y = bin_var_arr[b];
      Sw += w; Sx += w * x; Sy += w * y; Sxx += w * x * x; Sxy += w * x * y;
    }
    const float det = Sw * Sxx - Sx * Sx;
    if(fabs(det) > 1e-30f)
    {
      const float new_alpha = (Sw * Sxy - Sx * Sy) / det;
      const float new_sq    = (Sxx * Sy - Sx * Sxy) / det;
      if(new_alpha > 0)  alpha_est    = new_alpha;
      if(new_sq    >= 0) sigma_sq_est = new_sq;
    }
  }
  alpha_est = fmax(alpha_est, 1e-8f);

  /* Step 5 (dark refinement) is now in §7.2c-§7.2f histogram-based parallel
   * kernels (= ISP-streaming friendly, no sample buffer cap).  This kernel
   * only writes alpha_est now; sigma_sq_est is set by §7.2f. */
  params[P_ALPHA]    = alpha_est;
  params[P_SIGMA_SQ] = 0.0f;   /* placeholder — overwritten by §7.2f */
}


/* ================================================================
 * §7.2c-§7.2f Histogram-based dark refinement.
 *
 * Replaces CPU's quick_select on 50000-element sample buffer with two
 * histogram passes:
 *   §7.2c dark_thresh_hist     — multi-WG parallel, atomic-add to global hist
 *                                of halfres raw values.  Range [0, 1].
 *   §7.2d dark_thresh_finalize — single WI cumsum scan, find 10th percentile
 *                                = dark_thresh, write to params[P_S_SCALE+1]
 *                                (= temp slot, not final).
 *   §7.2e dark_lap_hist        — multi-WG parallel, atomic-add of |H+V Lap|
 *                                values for triplets where all 3 samples
 *                                ≤ dark_max.  Range [0, O32_DARK_LAP_MAX].
 *   §7.2f dark_finalize        — single WI cumsum scan, find median = MAD,
 *                                σ² = (MAD/0.6745)²/6 - α · dark_thresh/2,
 *                                write to params[P_SIGMA_SQ].
 *
 * Histograms in global memory (= multi-WG can atomic-add across WGs).
 * Bin counts: 4096 each.  Range scaling: dark_thresh hist clips at 1.0,
 * dark_lap hist clips at O32_DARK_LAP_MAX=0.1 (= ~10x typical noise std).
 *
 * (日) histogram-based dark refine。 sample buffer cap 撤廃、 multi-WG 並列。
 *   ISP streaming silicon でも実装可能な per-pixel atomic-increment 設計。
 * ================================================================ */
#define O32_DARK_HIST_BINS 4096
#define O32_DARK_LAP_MAX   0.1f

kernel void galosh_o32_ne_dark_thresh_hist(
    global const float *restrict raw,
    const int width,
    const int height,
    global int *restrict dark_thresh_hist)
{
  const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);
  const int halfwidth  = (width  + 1) / 2;
  const int halfheight = (height + 1) / 2;
  /* Sub-sample stride=3 per CFA channel; iterate 4 channels per WI. */
  if(gid_x * 3 >= halfwidth || gid_y * 3 >= halfheight) return;

  const float scale = (float)O32_DARK_HIST_BINS;   /* range [0,1) → bin */
  for(int ch = 0; ch < 4; ch++)
  {
    const int dy0 = (ch >> 1) & 1, dx0 = ch & 1;
    const int hr = gid_y * 3;
    const int hc = gid_x * 3;
    if(hr >= halfheight || hc >= halfwidth) continue;
    const int fr = 2 * hr + dy0;
    const int fc = 2 * hc + dx0;
    if(fr >= height || fc >= width) continue;
    const float v = raw[(size_t)fr * width + fc];
    int bin = (int)(v * scale);
    if(bin < 0) bin = 0;
    if(bin >= O32_DARK_HIST_BINS) bin = O32_DARK_HIST_BINS - 1;
    atomic_inc(&dark_thresh_hist[bin]);
  }
}

kernel void galosh_o32_ne_dark_thresh_finalize(
    global const int *restrict dark_thresh_hist,
    global float *restrict params,
    const int dark_thresh_slot)   /* index into params for storing dark_thresh */
{
  if(get_global_id(0) != 0) return;
  /* Total count. */
  int total = 0;
  for(int i = 0; i < O32_DARK_HIST_BINS; i++) total += dark_thresh_hist[i];
  if(total < 100) { params[dark_thresh_slot] = 0.01f; return; }

  /* 10th percentile cumsum. */
  const int target = total / 10;
  int cum = 0, dark_bin = 0;
  for(int i = 0; i < O32_DARK_HIST_BINS; i++)
  {
    cum += dark_thresh_hist[i];
    if(cum >= target) { dark_bin = i; break; }
  }
  const float dark_thresh = ((float)dark_bin + 0.5f) / (float)O32_DARK_HIST_BINS;
  params[dark_thresh_slot] = dark_thresh;
}

kernel void galosh_o32_ne_dark_lap_hist(
    global const float *restrict raw,
    const int width,
    const int height,
    global const float *restrict params,
    const int dark_thresh_slot,
    global int *restrict dark_lap_hist)
{
  const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);
  const int halfwidth  = (width  + 1) / 2;
  const int halfheight = (height + 1) / 2;
  if(gid_x >= halfwidth || gid_y >= halfheight) return;

  const float dark_max = params[dark_thresh_slot] + 0.02f;
  const float scale = (float)O32_DARK_HIST_BINS / O32_DARK_LAP_MAX;

  /* Process 4 channels per WI; stride=1 (= match CPU). */
  for(int ch = 0; ch < 4; ch++)
  {
    const int dy0 = (ch >> 1) & 1, dx0 = ch & 1;
    const int hr = gid_y;
    const int hc = gid_x;
    /* Horizontal Laplacian: x ∈ [0, halfwidth-2). */
    if(hc < halfwidth - 2)
    {
      const int fr = 2 * hr + dy0;
      if(fr < height)
      {
        const float v0 = raw[(size_t)fr * width + (2 *  hc      + dx0)];
        const float v1 = raw[(size_t)fr * width + (2 * (hc + 1) + dx0)];
        const float v2 = raw[(size_t)fr * width + (2 * (hc + 2) + dx0)];
        if(!(v0 > dark_max || v1 > dark_max || v2 > dark_max))
        {
          const float lap = fabs(v0 - 2.0f * v1 + v2);
          int bin = (int)(lap * scale);
          if(bin < 0) bin = 0;
          if(bin >= O32_DARK_HIST_BINS) bin = O32_DARK_HIST_BINS - 1;
          atomic_inc(&dark_lap_hist[bin]);
        }
      }
    }
    /* Vertical Laplacian: y ∈ [0, halfheight-2). */
    if(hr < halfheight - 2)
    {
      const int fc = 2 * hc + dx0;
      if(fc < width)
      {
        const float v0 = raw[(size_t)(2 *  hr      + dy0) * width + fc];
        const float v1 = raw[(size_t)(2 * (hr + 1) + dy0) * width + fc];
        const float v2 = raw[(size_t)(2 * (hr + 2) + dy0) * width + fc];
        if(!(v0 > dark_max || v1 > dark_max || v2 > dark_max))
        {
          const float lap = fabs(v0 - 2.0f * v1 + v2);
          int bin = (int)(lap * scale);
          if(bin < 0) bin = 0;
          if(bin >= O32_DARK_HIST_BINS) bin = O32_DARK_HIST_BINS - 1;
          atomic_inc(&dark_lap_hist[bin]);
        }
      }
    }
  }
}

kernel void galosh_o32_ne_dark_finalize(
    global const int *restrict dark_lap_hist,
    global float *restrict params,
    const int dark_thresh_slot)
{
  if(get_global_id(0) != 0) return;
  int total = 0;
  for(int i = 0; i < O32_DARK_HIST_BINS; i++) total += dark_lap_hist[i];
  if(total < 100) return;   /* keep alpha-only sigma_sq_est = 0 */

  const int target = total / 2;
  int cum = 0, med_bin = 0;
  for(int i = 0; i < O32_DARK_HIST_BINS; i++)
  {
    cum += dark_lap_hist[i];
    if(cum >= target) { med_bin = i; break; }
  }
  const float bin_scale = (float)O32_DARK_HIST_BINS / O32_DARK_LAP_MAX;
  const float mad = ((float)med_bin + 0.5f) / bin_scale;
  const float sigma_lap = mad / 0.6745f;
  const float dark_var = (sigma_lap * sigma_lap) / 6.0f;
  const float dark_thresh = params[dark_thresh_slot];
  const float dark_mean = dark_thresh * 0.5f;
  const float alpha = params[P_ALPHA];
  const float sigma_sq = fmax(dark_var - alpha * dark_mean, 0.0f);
  params[P_SIGMA_SQ] = sigma_sq;
}


/* ================================================================
 * §7.3 galosh_o32_build_inv_lut — Phase 0 (3/3): build inverse GAT LUT
 *   via Poisson summation + 10-point Gauss-Hermite quadrature.
 *   Mirrors CPU gat_build_inverse_table.
 *
 * Per LUT entry i ∈ [0, GAT_LUT_SIZE):
 *   x_val = i / (LUT_SIZE - 1)               signal in [0,1]
 *   λ     = x_val / α                         Poisson rate
 *   D     = E[GAT(Y)] for Y ~ α·Poisson(λ) + N(0, σ²)
 *
 * Output:
 *   lut_d[i], lut_x[i]                       d-x table (sorted, monotone)
 *   lut_params[0..6]                         d_min, d_max, y_break, t_break,
 *                                            sigma_raw, alpha, sigma_sq
 *
 * Dispatch: GAT_LUT_SIZE WIs (= 4096 work-items, parallel per entry).
 * Workgroup 256.
 *
 * (日) inv-GAT LUT 構築。 各エントリは独立に E[GAT(Poisson*α + N(0,σ²))] を
 *   Poisson 級数 + 10-point Gauss-Hermite 求積で計算。 4096 WI 並列。
 * ================================================================ */
__constant float gh_nodes_o32[10] = {
  -3.436159f, -2.532732f, -1.756684f, -1.036611f, -0.342901f,
   0.342901f,  1.036611f,  1.756684f,  2.532732f,  3.436159f
};
__constant float gh_weights_o32[10] = {
  7.640432855232641e-06f, 1.343645746781232e-03f, 3.387439445548111e-02f,
  2.401386110823147e-01f, 6.108626337353258e-01f,
  6.108626337353258e-01f, 2.401386110823147e-01f, 3.387439445548111e-02f,
  1.343645746781232e-03f, 7.640432855232641e-06f
};

kernel void galosh_o32_build_inv_lut(
    global const float *restrict params,
    global       float *restrict lut_d,
    global       float *restrict lut_x,
    global       float *restrict lut_params)
{
  const int i = get_global_id(0);
  if(i >= GAT_LUT_SIZE) return;

  /* 2026-05-10: FP64 (cl_khr_fp64) computation throughout to mirror CPU
   * `gat_build_inverse_table` (galosh_cpu.h line 1180+) exactly.  Cost: LUT
   * = 4096 entries built ONCE per image, FP64 overhead negligible (= ~ms).
   * Required for inverse-GAT lookup to match CPU bit-near in dark-image
   * Phase 10 reconstruction (= dominated final-output divergence). */
  const double a   = (double)params[P_ALPHA];
  const double sq  = (double)params[P_SIGMA_SQ];
  const double sig = sqrt(fmax(sq, 1e-20));
  const double y_break = -0.375 * a;
  const double t_break = 2.0 * sig / a;

  const double x_val  = (double)i / (double)(GAT_LUT_SIZE - 1);
  const double lambda = x_val / a;

  /* Poisson summation: log_prob recurrence avoids underflow. */
  double expected_gat = 0.0;
  const int k_max = (int)(lambda + 8.0 * sqrt(fmax(lambda, 1.0))) + 20;
  double log_prob = -lambda;

  for(int k = 0; k <= k_max; k++)
  {
    if(k > 0) log_prob += log(lambda) - log((double)k);
    const double prob = exp(log_prob);
    if(prob < 1e-15 && k > (int)lambda + 1) break;

    /* 10-point Gauss-Hermite quadrature over Gaussian noise. */
    double eg = 0.0;
    for(int g = 0; g < 10; g++)
    {
      const double z = 1.4142135623730951 * sig * (double)gh_nodes_o32[g];
      const double noisy_y = (double)k * a + z;
      double T;
      if(noisy_y >= y_break)
      {
        const double arg = a * noisy_y + 0.375 * a * a + sq;
        T = (2.0 / a) * sqrt(fmax(arg, 0.0));
      }
      else
      {
        T = t_break + (noisy_y - y_break) / sig;
      }
      eg += (double)gh_weights_o32[g] * T;
    }
    eg *= 0.5641895835477563;  /* 1 / sqrt(pi), full double precision */
    expected_gat += prob * eg;
  }

  lut_d[i] = (float)expected_gat;
  lut_x[i] = (float)x_val;

  /* WI 0: write lut_params (= matches §1 K_LUT_FINALIZE layout):
   *   [0]=d_min (deferred to §7.3b), [1]=d_max (deferred to §7.3b),
   *   [2]=y_break, [3]=t_break, [4]=sigma_raw, [5]=alpha, [6]=sigma_sq. */
  if(i == 0)
  {
    lut_params[2] = y_break;
    lut_params[3] = t_break;
    lut_params[4] = sig;
    lut_params[5] = a;
    lut_params[6] = sq;
  }
  /* d_min (= lut_d[0]) and d_max (= lut_d[LUT_SIZE-1]) finalized in §7.3b
   * after all entries computed. */
}

/* §7.3b galosh_o32_lut_finalize — write d_min and d_max into lut_params
 * after build_inv_lut has filled lut_d[].  Trivial 1-WI helper.
 *
 * BUG FIX 2026-05-09: previously wrote only d_min, leaving lut_params[1]
 * containing the placeholder dx = 1/(LUT_SIZE-1) ≈ 2.4e-4.  gat_inv_lut
 * helper reads lut_params[1] as d_max → for any D > 2.4e-4 returned 1.0
 * (= saturated), causing Phase 10 output mean to clip to 1.0 and PSNR
 * to drop to 1 dB.  Now writes proper d_max = lut_d[LUT_SIZE-1]. */
kernel void galosh_o32_lut_finalize(
    global const float *restrict lut_d,
    global       float *restrict lut_params)
{
  if(get_global_id(0) != 0) return;
  lut_params[0] = lut_d[0];                          /* d_min */
  lut_params[1] = lut_d[GAT_LUT_SIZE - 1];           /* d_max */
}


/* ================================================================
 * §7.4 galosh_o32_gat_forward_full — Phase 1 (1/4): per-pixel GAT
 *   forward.  Mirrors CPU `gat_forward` (piecewise C¹ VST).
 *
 *   For x >= y_break = -3α/8:
 *     T(x) = (2/α) · sqrt(α·x + 3α²/8 + σ²)             (Foi sqrt branch)
 *   For x < y_break:
 *     T(x) = t_break + (x - y_break) / σ_raw            (linear branch)
 *   where t_break = 2σ/α, σ_raw = sqrt(σ²).
 *
 *   Output: in_gat_full (full-res FP32) AND 4 half-res CFA-channel
 *   views (ch0..3) for downstream G K7-K10 dark-anchor compatibility
 *   (= will be removed once Phase 2 IRLS port lands).
 *
 *   α/σ² are read from params[P_ALPHA], params[P_SIGMA_SQ] (= written
 *   by §7.2 ne_finalize).
 *
 * Dispatch: 2D over (width, height).  LDS: none.
 *
 * (日) Phase 1 1/4: 全 pixel に GAT forward (piecewise C¹)。 in_gat_full と
 *   ch0..3 (= per-CFA half-res view) を同時に書く。 後者は G K7-K10
 *   dark anchor が ch0..3 を入力としているため Phase 2 港まで併用。
 * ================================================================ */
kernel void galosh_o32_gat_forward_full(
    global const float *restrict raw,
    global       float *restrict in_gat_full,
    global       float *restrict ch0,
    global       float *restrict ch1,
    global       float *restrict ch2,
    global       float *restrict ch3,
    global const float *restrict params,
    const int width,
    const int height)
{
  const int fx = get_global_id(0);
  const int fy = get_global_id(1);
  if(fx >= width || fy >= height) return;

  const float a  = params[P_ALPHA];
  const float sq = params[P_SIGMA_SQ];
  const float sigma_raw = sqrt(fmax(sq, 1e-20f));
  const float y_break = -0.375f * a;
  const float t_break = 2.0f * sigma_raw / a;

  const float x = raw[(size_t)fy * width + fx];
  /* NaN guard, no clamping. */
  const float x_safe = (x == x) ? x : 0.0f;

  float T;
  if(x_safe >= y_break)
  {
    /* Foi sqrt branch */
    const float arg = a * x_safe + 0.375f * a * a + sq;
    T = (2.0f / a) * sqrt(fmax(arg, 0.0f));
  }
  else
  {
    /* C¹ linear branch */
    T = t_break + (x_safe - y_break) / sigma_raw;
  }

  in_gat_full[(size_t)fy * width + fx] = T;

  /* Per-CFA half-res view (= for K7-K10 dark anchor compat).
   * Slot encoding matches CPU offsets {{0,0},{0,1},{1,0},{1,1}}:
   *   slot = (fy & 1) * 2 + (fx & 1)  =>  {0:RR, 1:RC, 2:CR, 3:CC}
   * Half-res index: hp = (fy/2) * (width/2) + (fx/2). */
  const int slot_r = fy & 1;
  const int slot_c = fx & 1;
  const int slot   = slot_r | (slot_c << 1);   /* matches §6.1 + G convention */
  const int hw     = width >> 1;
  const int hp     = (fy >> 1) * hw + (fx >> 1);
  if(slot == 0)      ch0[hp] = T;
  else if(slot == 1) ch1[hp] = T;
  else if(slot == 2) ch2[hp] = T;
  else               ch3[hp] = T;
}


/* ================================================================
 * §7.5 galosh_o32_sigma_per_cfa — Phase 1 (2/4): per-CFA σ estimation.
 *   Mirrors CPU `estimate_gat_sigma_halfres` for each of 4 CFA channels.
 *
 *   For each channel s: extract half-res view of in_gat_full (= every-2
 *   sampling at offset (s>>1, s&1)), compute |Laplacian| via 3-tap
 *   horizontal stride-1 within the channel, populate histogram, find
 *   median bin → MAD, σ = MAD / 1.6521.
 *
 *   Histogram: 4096 bins × 4 bytes = 16 KB LDS (one per WG, one WG per
 *   channel = 4 launches in 4 WGs).
 *
 *   Dispatch: 4 work-groups × 64 WIs (= 256 total WIs, group_id = ch).
 *
 * Output: params[P_SIGMA_CH0..3]
 *
 * (日) Phase 1 2/4: 各 CFA channel ごとに Laplacian MAD で σ 推定。
 *   3-tap stride=1 水平 |Lap| を histogram で median → MAD/1.6521 = σ。
 *   YUV-Q galosh_yuv_q_lap_mad と同じパターン。 4 channel × 4096-bin
 *   histogram (= 16 KB LDS / WG)。
 * ================================================================ */
#define O32_SIGMA_NBINS 4096
#define O32_SIGMA_LAP_MAX 16.0f   /* clip range for histogram */
#define O32_SIGMA_WG 64

kernel void galosh_o32_sigma_per_cfa(
    global const float *restrict in_gat_full,
    const int width,
    const int height,
    global float *restrict params)
{
  const int ch  = get_group_id(0);   /* 0..3, one WG per channel */
  const int lid = get_local_id(0);
  const int wg  = get_local_size(0);

  /* CFA slot encoding (matches CPU offsets[][] = {{0,0},{0,1},{1,0},{1,1}}):
   *   ch=0 → (dy0=0, dx0=0)   even row, even col
   *   ch=1 → (dy0=0, dx0=1)   even row, odd col
   *   ch=2 → (dy0=1, dx0=0)   odd row,  even col
   *   ch=3 → (dy0=1, dx0=1)   odd row,  odd col
   * BUG FIX 2026-05-09: previously had dy0/dx0 swapped, leading to
   * params[P_SIGMA_CH1] / [P_SIGMA_CH2] holding swapped values.  No
   * functional impact (unified_sigma is symmetric Σσ²) but fixed for
   * cosmetic correctness of [GPU] Sigma: log line. */
  const int dy0 = ch & 1;
  const int dx0 = (ch >> 1) & 1;
  const int hw  = (width  - dx0 + 1) / 2;
  const int hh  = (height - dy0 + 1) / 2;

  local int hist[O32_SIGMA_NBINS];
  local int total_count;

  /* Zero histogram. */
  for(int i = lid; i < O32_SIGMA_NBINS; i += wg) hist[i] = 0;
  if(lid == 0) total_count = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Each WI scans a row stripe.  3-tap stride=1 |Laplacian| in half-res. */
  const float bin_scale = (float)O32_SIGMA_NBINS / O32_SIGMA_LAP_MAX;
  int my_count = 0;
  for(int hr = lid; hr < hh; hr += wg)
  {
    const int fr = 2 * hr + dy0;
    if(fr >= height) continue;
    /* Half-res sampling: x_h ∈ [0, hw-2), 3-tap on (x_h, x_h+1, x_h+2). */
    for(int hc = 0; hc < hw - 2; hc += 3)   /* CPU uses x += 3 stride */
    {
      const int fc0 = 2 * hc       + dx0;
      const int fc1 = 2 * (hc + 1) + dx0;
      const int fc2 = 2 * (hc + 2) + dx0;
      if(fc2 >= width) break;
      const float a = in_gat_full[(size_t)fr * width + fc0];
      const float b = in_gat_full[(size_t)fr * width + fc1];
      const float c = in_gat_full[(size_t)fr * width + fc2];
      const float lap = fabs(a - 2.0f * b + c);
      int bin = (int)(lap * bin_scale);
      if(bin >= O32_SIGMA_NBINS) bin = O32_SIGMA_NBINS - 1;
      atomic_inc(&hist[bin]);
      my_count++;
    }
  }
  atomic_add(&total_count, my_count);
  barrier(CLK_LOCAL_MEM_FENCE);

  /* WI 0: cumsum-scan to find median bin → MAD → σ. */
  if(lid == 0)
  {
    const int median_target = total_count / 2;
    int cum = 0, median_bin = 0;
    for(int i = 0; i < O32_SIGMA_NBINS; i++)
    {
      cum += hist[i];
      if(cum >= median_target) { median_bin = i; break; }
    }
    const float mad = ((float)median_bin + 0.5f) / bin_scale;
    /* 3-tap iid noise: Var(Lap) = 6σ² → σ = MAD / (0.6745 · sqrt(6)) = MAD / 1.6521. */
    const float sigma = fmax(mad / 1.6521f, 0.01f);
    params[P_SIGMA_CH0 + ch] = sigma;
  }
}


/* ================================================================
 * §7.6 galosh_o32_unified_sigma — Phase 1 (3/4): compute unified_sigma.
 *   Mirrors CPU lines 4667-4673:
 *     unified = sqrt( max( 0.25 * Σ σ_ch² , 1e-12) )
 *     inv_sg  = 1 / unified
 *
 *   Single WI; reads params[P_SIGMA_CH0..3], writes params[P_UNIFIED_SIGMA]
 *   and params[P_INV_SG].
 *
 * (日) Phase 1 3/4: per-CFA σ から RMS 平均で unified_sigma を 1 WI で計算。
 * ================================================================ */
kernel void galosh_o32_unified_sigma(
    global float *restrict params)
{
  if(get_global_id(0) != 0) return;
  const float s0 = params[P_SIGMA_CH0];
  const float s1 = params[P_SIGMA_CH1];
  const float s2 = params[P_SIGMA_CH2];
  const float s3 = params[P_SIGMA_CH3];
  const float mean_var = 0.25f * (s0*s0 + s1*s1 + s2*s2 + s3*s3);
  const float unified = sqrt(fmax(mean_var, 1e-12f));
  params[P_UNIFIED_SIGMA] = unified;
  params[P_INV_SG]        = 1.0f / unified;
}


/* ================================================================
 * §7.7 galosh_o32_normalize_apply — Phase 1 (4/4): apply unified_sigma
 *   normalization to in_gat_full + ch0..3.  Mirrors CPU
 *   `for(i) in_gat[i] *= inv_sg`.
 *
 *   Reads params[P_INV_SG], multiplies in_gat_full[fy, fx] and the
 *   corresponding ch[slot][hp] by inv_sg.  In-place.
 *
 * Dispatch: 2D over (width, height).  LDS: none.
 *
 * (日) Phase 1 4/4: in_gat_full と ch0..3 を unified_sigma で割って
 *   normalize。 in-place per-pixel multiply。
 * ================================================================ */
kernel void galosh_o32_normalize_apply(
    global       float *restrict in_gat_full,
    global       float *restrict ch0,
    global       float *restrict ch1,
    global       float *restrict ch2,
    global       float *restrict ch3,
    global const float *restrict params,
    const int width,
    const int height)
{
  const int fx = get_global_id(0);
  const int fy = get_global_id(1);
  if(fx >= width || fy >= height) return;

  const float inv_sg = params[P_INV_SG];
  const size_t p = (size_t)fy * width + fx;
  in_gat_full[p] *= inv_sg;

  /* Per-CFA half-res view (write only when (fy, fx) is the slot's TL). */
  const int slot_r = fy & 1;
  const int slot_c = fx & 1;
  const int slot   = slot_r | (slot_c << 1);
  const int hw     = width >> 1;
  const int hp     = (fy >> 1) * hw + (fx >> 1);
  if(slot == 0)      ch0[hp] *= inv_sg;
  else if(slot == 1) ch1[hp] *= inv_sg;
  else if(slot == 2) ch2[hp] *= inv_sg;
  else               ch3[hp] *= inv_sg;
}


/* ================================================================
 * §7.8 galosh_o32_dark_ref_irls — Phase 2 (1/2): dark anchor IRLS.
 *   Mirrors CPU galosh_raw_cpu.c lines 4684-4775.
 *
 * Algorithm (3 iterations, n_iter=2):
 *   s_scale (init) = σ² / α     (s_min = 0.05·s_init, s_max = 50·s_init)
 *   For iter ∈ {0, 1, 2}:
 *     For each 2x2 block (br, bc step 2):
 *       g0..3 = in_gat at 4 CFA slots
 *       ch_max - ch_min > GALOSH_ACHROMATIC_RANGE → skip (= edge filter)
 *       L_raw = mean of 4 raw values
 *       w = 1 / (1 + (L_raw / s_scale)^4)         (= Tukey-bisquare)
 *       sum_w  += w
 *       sum_wi += w · gi   (i ∈ 0..3)
 *     dark_ref[i] = sum_wi / sum_w
 *     If iter < 2:
 *       sum_resid² = Σ w·(g - dark_ref)²·0.25
 *       measured_std = sqrt(sum_resid² / sum_wW)
 *       s_scale *= sqrt(1 / measured_std)
 *       clamp(s_scale, s_min, s_max)
 *
 * Single WG of 32 WIs.  All 3 iterations done in-kernel; reductions
 * happen in LDS.  Outputs dark_ref to params[P_DARK_REF0..3] and final
 * s_scale to params[P_S_SCALE].
 *
 * (日) Phase 2 1/3: dark_ref を 3 iter Tukey-bisquare IRLS で推定。
 *   Tukey-bisquare weight: w = 1/(1 + (L_raw/s)⁴)。 Residual std で
 *   s_scale を update し次 iter へ。 single WG 32 WI、 3 iter 全部 in-kernel。
 * ================================================================ */
#ifndef P_DARK_REF0
#define P_DARK_REF0       6
#endif
#ifndef P_S_SCALE
#define P_S_SCALE        10
#endif

#ifndef GALOSH_ACHROMATIC_RANGE_O32
#define GALOSH_ACHROMATIC_RANGE_O32 4.0f   /* matches CPU galosh_cpu.h:364 */
#endif

#define O32_DR_WG 32

kernel void galosh_o32_dark_ref_irls(
    global const float *restrict in_gat_full,
    global const float *restrict raw,
    const int width,
    const int height,
    global float *restrict params)
{
  const int lid = get_local_id(0);
  const int wg  = get_local_size(0);   /* O32_DR_WG = 32 */

  /* Per-WI partial sums; reduced via LDS. */
  local float lds_sw [O32_DR_WG];
  local float lds_sw0[O32_DR_WG];
  local float lds_sw1[O32_DR_WG];
  local float lds_sw2[O32_DR_WG];
  local float lds_sw3[O32_DR_WG];
  local float lds_swr[O32_DR_WG];
  local float lds_sww[O32_DR_WG];

  /* Final reduced values shared via LDS for all WIs to read after barrier. */
  local float dr0_lds, dr1_lds, dr2_lds, dr3_lds;
  local float s_scale_lds;

  if(lid == 0)
  {
    const float a  = params[P_ALPHA];
    const float sq = params[P_SIGMA_SQ];
    const float s_init = sq / fmax(a, 1e-12f);
    s_scale_lds = s_init;
    dr0_lds = dr1_lds = dr2_lds = dr3_lds = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const float a = params[P_ALPHA];
  const float sq = params[P_SIGMA_SQ];
  const float s_init = sq / fmax(a, 1e-12f);
  const float s_min = 0.05f * s_init;
  const float s_max = 50.0f * s_init;

  /* Even rows of 2x2 blocks: br = 0, 2, 4, ..., height-2.
   * BUG FIX 2026-05-09: previous formula `(height-1)/2` undercounted by 1
   * row for EVEN heights (= 256/512/etc.).  CPU iterates `br < height-1`
   * which gives br ∈ {0, 2, ..., height-2} = height/2 values for even
   * height.  Off-by-one caused 1 missed row of dark anchor blocks =
   * different sample set vs CPU = dark_ref divergence. */
  const int n_block_rows = height / 2;   /* matches CPU "br < height-1" */

  for(int iter = 0; iter <= 2; iter++)
  {
    const float inv_s = 1.0f / fmax(s_scale_lds, 1e-20f);

    /* Pass 1: weighted sums for dark_ref. */
    float my_sw = 0.0f, my_sw0 = 0.0f, my_sw1 = 0.0f, my_sw2 = 0.0f, my_sw3 = 0.0f;
    for(int br_idx = lid; br_idx < n_block_rows; br_idx += wg)
    {
      const int br = 2 * br_idx;
      for(int bc = 0; bc < width - 1; bc += 2)
      {
        const float g0 = in_gat_full[(size_t)br       * width + bc      ];
        const float g1 = in_gat_full[(size_t)(br + 1) * width + bc      ];
        const float g2 = in_gat_full[(size_t)br       * width + (bc + 1)];
        const float g3 = in_gat_full[(size_t)(br + 1) * width + (bc + 1)];
        const float ch_max = fmax(fmax(g0, g1), fmax(g2, g3));
        const float ch_min = fmin(fmin(g0, g1), fmin(g2, g3));
        if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE_O32) continue;

        const float iv0 = raw[(size_t)br       * width + bc      ];
        const float iv1 = raw[(size_t)(br + 1) * width + bc      ];
        const float iv2 = raw[(size_t)br       * width + (bc + 1)];
        const float iv3 = raw[(size_t)(br + 1) * width + (bc + 1)];
        const float L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25f;
        const float r  = L_raw * inv_s;
        const float r2 = r * r;
        const float w  = 1.0f / (1.0f + r2 * r2);
        my_sw  += w;
        my_sw0 += w * g0;
        my_sw1 += w * g1;
        my_sw2 += w * g2;
        my_sw3 += w * g3;
      }
    }
    lds_sw [lid] = my_sw;
    lds_sw0[lid] = my_sw0;
    lds_sw1[lid] = my_sw1;
    lds_sw2[lid] = my_sw2;
    lds_sw3[lid] = my_sw3;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Tree reduction (32 -> 1). */
    for(int s = wg / 2; s > 0; s >>= 1)
    {
      if(lid < s)
      {
        lds_sw [lid] += lds_sw [lid + s];
        lds_sw0[lid] += lds_sw0[lid + s];
        lds_sw1[lid] += lds_sw1[lid + s];
        lds_sw2[lid] += lds_sw2[lid + s];
        lds_sw3[lid] += lds_sw3[lid + s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0)
    {
      const float inv_sw = 1.0f / fmax(lds_sw[0], 1e-20f);
      dr0_lds = lds_sw0[0] * inv_sw;
      dr1_lds = lds_sw1[0] * inv_sw;
      dr2_lds = lds_sw2[0] * inv_sw;
      dr3_lds = lds_sw3[0] * inv_sw;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(iter == 2) break;

    /* Pass 2: weighted residual std for s_scale update. */
    const float dr0 = dr0_lds, dr1 = dr1_lds, dr2 = dr2_lds, dr3 = dr3_lds;
    float my_swr = 0.0f, my_sww = 0.0f;
    for(int br_idx = lid; br_idx < n_block_rows; br_idx += wg)
    {
      const int br = 2 * br_idx;
      for(int bc = 0; bc < width - 1; bc += 2)
      {
        const float g0 = in_gat_full[(size_t)br       * width + bc      ];
        const float g1 = in_gat_full[(size_t)(br + 1) * width + bc      ];
        const float g2 = in_gat_full[(size_t)br       * width + (bc + 1)];
        const float g3 = in_gat_full[(size_t)(br + 1) * width + (bc + 1)];
        const float ch_max = fmax(fmax(g0, g1), fmax(g2, g3));
        const float ch_min = fmin(fmin(g0, g1), fmin(g2, g3));
        if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE_O32) continue;

        const float iv0 = raw[(size_t)br       * width + bc      ];
        const float iv1 = raw[(size_t)(br + 1) * width + bc      ];
        const float iv2 = raw[(size_t)br       * width + (bc + 1)];
        const float iv3 = raw[(size_t)(br + 1) * width + (bc + 1)];
        const float L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25f;
        const float r  = L_raw * inv_s;
        const float r2 = r * r;
        const float w  = 1.0f / (1.0f + r2 * r2);
        const float d0 = g0 - dr0, d1 = g1 - dr1, d2 = g2 - dr2, d3 = g3 - dr3;
        const float resid2 = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        my_sww += w;
        my_swr += w * resid2 * 0.25f;
      }
    }
    lds_swr[lid] = my_swr;
    lds_sww[lid] = my_sww;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int s = wg / 2; s > 0; s >>= 1)
    {
      if(lid < s)
      {
        lds_swr[lid] += lds_swr[lid + s];
        lds_sww[lid] += lds_sww[lid + s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0)
    {
      const float inv_sw2 = 1.0f / fmax(lds_sww[0], 1e-20f);
      const float measured_std = sqrt(fmax(lds_swr[0] * inv_sw2, 1e-20f));
      const float ratio = 1.0f / measured_std;
      float new_s = s_scale_lds * sqrt(ratio);
      if(new_s < s_min) new_s = s_min;
      if(new_s > s_max) new_s = s_max;
      s_scale_lds = new_s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  /* Final write to params (= WI 0 only). */
  if(lid == 0)
  {
    params[P_DARK_REF0 + 0] = dr0_lds;
    params[P_DARK_REF0 + 1] = dr1_lds;
    params[P_DARK_REF0 + 2] = dr2_lds;
    params[P_DARK_REF0 + 3] = dr3_lds;
    params[P_S_SCALE]       = s_scale_lds;
  }
}


/* ================================================================
 * §7.8b/c/d/e — Multi-WG version of dark IRLS (ISP-streaming friendly).
 *
 * Current §7.8 single-WG bottleneck: 0.46 ms on 256² scales ~linearly
 * to 29 ms on 4MP (= 64× more blocks).  Multi-WG with partial-sum
 * reduction pattern (= mirror of existing G K7-K10) decouples runtime
 * from block count.
 *
 * Pipeline (host iterates 3 times):
 *   §7.8b reduce      — multi-WG (N_REDUCE_WG × REDUCE_WG_SIZE), each WI
 *                       scans a slice of blocks, WG reduces sum_w + sum_wi
 *                       in LDS, writes partial[wg_id*5..wg_id*5+4]
 *   §7.8c finalize_dr — 1 WG aggregates partials → dark_ref[0..3]
 *   §7.8d resid_reduce — multi-WG, sum_w·resid² + sum_w
 *   §7.8e resid_finalize — 1 WG aggregates → s_scale update
 *
 * (日) §7.8 を multi-WG パラレル化 (= G K7-K10 と同じ 64-WG × 256-WI
 *   reduction pattern)。 4MP ターゲットで 29 ms → ~1 ms に。
 * ================================================================ */
#define O32_DR_MWG_NWG    64
#define O32_DR_MWG_WGSIZE 256

/* §7.8b: per-WI scan blocks, WG-level tree reduce sum_w + sum_w0..3,
 * write 5 floats per WG to partial_buf. */
/* §7.8b dark_ref reduce — multi-WG partial sums.
 *
 * 2026-05-09: FP64 (cl_khr_fp64) accumulation throughout to mirror CPU's
 * `double` per-block w/r computation + sum accumulation (galosh_raw_cpu.c
 * lines 4697+).  LDS: 5 accumulators × 8B = 10 KB at WGSIZE=256, same
 * budget as prior FP32-Kahan path (= 5 × 2 × 4B).  Partial_buf written
 * as float (= consumer §7.8c re-aggregates in double internally). */
kernel void galosh_o32_dark_ref_reduce_mwg(
    global const float *restrict in_gat_full,
    global const float *restrict raw,
    const int width,
    const int height,
    global const float *restrict params,    /* read params[P_S_SCALE] */
    global       double *restrict partial_buf)  /* [n_wg * 5] doubles */
{
  const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  const int wgid = get_group_id(0);
  const int wgsize = get_local_size(0);
  const int total_wis = get_global_size(0);

  const double s_scale = (double)params[P_S_SCALE];
  const double inv_s = 1.0 / fmax(s_scale, 1e-20);

  /* BUG FIX 2026-05-09: was `(height-1)/2` / `(width-1)/2` which
   * undercounts by 1 row/col for even dims. */
  const int n_block_rows = height / 2;
  const int n_block_cols = width  / 2;
  const int total_blocks = n_block_rows * n_block_cols;

  /* Per-WI accumulators in double (= matches CPU double sum_w0..3). */
  double my_sw  = 0.0;
  double my_sw0 = 0.0;
  double my_sw1 = 0.0;
  double my_sw2 = 0.0;
  double my_sw3 = 0.0;

  for(int bi = gid; bi < total_blocks; bi += total_wis)
  {
    const int br_idx = bi / n_block_cols;
    const int bc_idx = bi - br_idx * n_block_cols;
    const int br = 2 * br_idx;
    const int bc = 2 * bc_idx;

    const float g0 = in_gat_full[(size_t)br       * width + bc      ];
    const float g1 = in_gat_full[(size_t)(br + 1) * width + bc      ];
    const float g2 = in_gat_full[(size_t)br       * width + (bc + 1)];
    const float g3 = in_gat_full[(size_t)(br + 1) * width + (bc + 1)];
    const float ch_max = fmax(fmax(g0, g1), fmax(g2, g3));
    const float ch_min = fmin(fmin(g0, g1), fmin(g2, g3));
    if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE_O32) continue;

    const float iv0 = raw[(size_t)br       * width + bc      ];
    const float iv1 = raw[(size_t)(br + 1) * width + bc      ];
    const float iv2 = raw[(size_t)br       * width + (bc + 1)];
    const float iv3 = raw[(size_t)(br + 1) * width + (bc + 1)];
    /* L_raw / r / r² / w in DOUBLE to mirror CPU. */
    const double L_raw = ((double)iv0 + (double)iv1 + (double)iv2 + (double)iv3) * 0.25;
    const double r  = L_raw * inv_s;
    const double r2 = r * r;
    const double w  = 1.0 / (1.0 + r2 * r2);
    my_sw  += w;
    my_sw0 += w * (double)g0;
    my_sw1 += w * (double)g1;
    my_sw2 += w * (double)g2;
    my_sw3 += w * (double)g3;
  }

  /* WG-level tree reduction in double LDS.  LDS: 5 doubles × WGSIZE × 8B
   * = 256 × 5 × 8 = 10 KB at WGSIZE=256, fits 32 KB ISP budget. */
  local double lds[O32_DR_MWG_WGSIZE * 5];
  lds[lid * 5 + 0] = my_sw;
  lds[lid * 5 + 1] = my_sw0;
  lds[lid * 5 + 2] = my_sw1;
  lds[lid * 5 + 3] = my_sw2;
  lds[lid * 5 + 4] = my_sw3;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int s = wgsize / 2; s > 0; s >>= 1)
  {
    if(lid < s)
    {
      lds[lid * 5 + 0] += lds[(lid + s) * 5 + 0];
      lds[lid * 5 + 1] += lds[(lid + s) * 5 + 1];
      lds[lid * 5 + 2] += lds[(lid + s) * 5 + 2];
      lds[lid * 5 + 3] += lds[(lid + s) * 5 + 3];
      lds[lid * 5 + 4] += lds[(lid + s) * 5 + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(lid == 0)
  {
    /* Write final partials as double (= consumer §7.8c reads as double).
     * Casting to float here would destroy the FP64 accumulation precision. */
    partial_buf[wgid * 5 + 0] = lds[0];
    partial_buf[wgid * 5 + 1] = lds[1];
    partial_buf[wgid * 5 + 2] = lds[2];
    partial_buf[wgid * 5 + 3] = lds[3];
    partial_buf[wgid * 5 + 4] = lds[4];
  }
}


/* §7.8c: aggregate partials → dark_ref[0..3].  Single WI, FP64 throughout.
 * Reads partial_buf as double (= preserves §7.8b's FP64 accumulation),
 * sums in double, divides in double, casts final dark_ref to float for
 * params (consumed by §6.1 + §6.15 as float, which is fine since the
 * critical precision is in the IRLS sum/divide path). */
kernel void galosh_o32_dark_ref_finalize_mwg(
    global const double *restrict partial_buf,
    const int n_wg,
    global       float *restrict params)
{
  if(get_global_id(0) != 0) return;
  double sw = 0.0, sw0 = 0.0, sw1 = 0.0, sw2 = 0.0, sw3 = 0.0;
  for(int i = 0; i < n_wg; i++)
  {
    sw  += partial_buf[i * 5 + 0];
    sw0 += partial_buf[i * 5 + 1];
    sw1 += partial_buf[i * 5 + 2];
    sw2 += partial_buf[i * 5 + 3];
    sw3 += partial_buf[i * 5 + 4];
  }
  const double inv_sw = 1.0 / fmax(sw, 1e-20);
  params[P_DARK_REF0 + 0] = (float)(sw0 * inv_sw);
  params[P_DARK_REF0 + 1] = (float)(sw1 * inv_sw);
  params[P_DARK_REF0 + 2] = (float)(sw2 * inv_sw);
  params[P_DARK_REF0 + 3] = (float)(sw3 * inv_sw);
}


/* §7.8d: per-WI scan blocks for residual std, WG reduce, write partials.
 * FP64 throughout to mirror CPU + preserve precision through to §7.8e. */
kernel void galosh_o32_dark_resid_reduce_mwg(
    global const float *restrict in_gat_full,
    global const float *restrict raw,
    const int width,
    const int height,
    global const float *restrict params,
    global       double *restrict partial_resid_buf)  /* [n_wg * 2] doubles */
{
  const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  const int wgid = get_group_id(0);
  const int wgsize = get_local_size(0);
  const int total_wis = get_global_size(0);

  /* s_scale and inv_s in double (= mirror CPU galosh_raw_cpu.c line 4696
   * `const double inv_s = 1.0 / fmax(s_scale, 1e-20);`).
   * dr* read as float, cast to double for residual computation. */
  const double s_scale = (double)params[P_S_SCALE];
  const double inv_s = 1.0 / fmax(s_scale, 1e-20);
  const double dr0_d = (double)params[P_DARK_REF0 + 0];
  const double dr1_d = (double)params[P_DARK_REF0 + 1];
  const double dr2_d = (double)params[P_DARK_REF0 + 2];
  const double dr3_d = (double)params[P_DARK_REF0 + 3];

  /* BUG FIX 2026-05-09: was `(height-1)/2` / `(width-1)/2` which
   * undercounts by 1 row/col for even dims; same fix as §7.8b. */
  const int n_block_rows = height / 2;
  const int n_block_cols = width  / 2;
  const int total_blocks = n_block_rows * n_block_cols;

  /* Per-WI accumulators in double (= mirror CPU). */
  double my_swr = 0.0;
  double my_sww = 0.0;

  for(int bi = gid; bi < total_blocks; bi += total_wis)
  {
    const int br_idx = bi / n_block_cols;
    const int bc_idx = bi - br_idx * n_block_cols;
    const int br = 2 * br_idx;
    const int bc = 2 * bc_idx;

    const float g0 = in_gat_full[(size_t)br       * width + bc      ];
    const float g1 = in_gat_full[(size_t)(br + 1) * width + bc      ];
    const float g2 = in_gat_full[(size_t)br       * width + (bc + 1)];
    const float g3 = in_gat_full[(size_t)(br + 1) * width + (bc + 1)];
    const float ch_max = fmax(fmax(g0, g1), fmax(g2, g3));
    const float ch_min = fmin(fmin(g0, g1), fmin(g2, g3));
    if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE_O32) continue;

    const float iv0 = raw[(size_t)br       * width + bc      ];
    const float iv1 = raw[(size_t)(br + 1) * width + bc      ];
    const float iv2 = raw[(size_t)br       * width + (bc + 1)];
    const float iv3 = raw[(size_t)(br + 1) * width + (bc + 1)];
    const double L_raw = ((double)iv0 + (double)iv1 + (double)iv2 + (double)iv3) * 0.25;
    const double r  = L_raw * inv_s;
    const double r2 = r * r;
    const double w  = 1.0 / (1.0 + r2 * r2);
    const double d0 = (double)g0 - dr0_d, d1 = (double)g1 - dr1_d;
    const double d2 = (double)g2 - dr2_d, d3 = (double)g3 - dr3_d;
    const double resid2 = d0*d0 + d1*d1 + d2*d2 + d3*d3;
    my_sww += w;
    my_swr += w * resid2 * 0.25;
  }

  /* WG-level tree reduction in double LDS.  LDS: 2 doubles × WGSIZE × 8B
   * = 256 × 2 × 8 = 4 KB at WGSIZE=256, well under 32 KB ISP budget. */
  local double lds[O32_DR_MWG_WGSIZE * 2];
  lds[lid * 2 + 0] = my_swr;
  lds[lid * 2 + 1] = my_sww;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int s = wgsize / 2; s > 0; s >>= 1)
  {
    if(lid < s)
    {
      lds[lid * 2 + 0] += lds[(lid + s) * 2 + 0];
      lds[lid * 2 + 1] += lds[(lid + s) * 2 + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(lid == 0)
  {
    /* Write as double to preserve FP64 precision through to §7.8e finalize. */
    partial_resid_buf[wgid * 2 + 0] = lds[0];
    partial_resid_buf[wgid * 2 + 1] = lds[1];
  }
}


/* §7.8e: aggregate residual partials, update s_scale.  Single WI, FP64 throughout.
 * Mirrors CPU galosh_raw_cpu.c lines 4769-4774 (= double measured_std,
 * double ratio, double s_scale update). */
kernel void galosh_o32_dark_resid_finalize_mwg(
    global const double *restrict partial_resid_buf,
    const int n_wg,
    const float s_min,
    const float s_max,
    global       float *restrict params)
{
  if(get_global_id(0) != 0) return;
  double swr = 0.0, sww = 0.0;
  for(int i = 0; i < n_wg; i++)
  {
    swr += partial_resid_buf[i * 2 + 0];
    sww += partial_resid_buf[i * 2 + 1];
  }
  const double inv_sw2 = 1.0 / fmax(sww, 1e-20);
  const double measured_std = sqrt(fmax(swr * inv_sw2, 1e-20));
  const double ratio = 1.0 / measured_std;
  double new_s = (double)params[P_S_SCALE] * sqrt(ratio);
  if(new_s < (double)s_min) new_s = (double)s_min;
  if(new_s > (double)s_max) new_s = (double)s_max;
  params[P_S_SCALE] = (float)new_s;
}


/* ================================================================
 * §7.9 galosh_o32_dark_sub_full — Phase 2 (2/2): per-pixel CFA-aware
 *   dark_ref subtraction on in_gat_full + ch0..3.  Mirrors CPU lines
 *   4782-4791.
 *
 *   For each pixel (fy, fx):
 *     slot = (fy & 1) | ((fx & 1) << 1)
 *     in_gat_full[fy, fx] -= params[P_DARK_REF0 + slot]
 *     ch[slot][hp]        -= same dark_ref (for downstream §6.* compat)
 *
 * Dispatch: 2D over (width, height).  LDS: none.
 *
 * (日) Phase 2 2/2: per-pixel CFA slot ごとに dark_ref を引く。 in_gat_full
 *   に加えて ch0..3 (= G K7-K10 互換用 half-res view) も同じ dark_ref で
 *   引く (= 後段 §6.x が ch0..3 経由で参照する場合のため)。
 * ================================================================ */
kernel void galosh_o32_dark_sub_full(
    global       float *restrict in_gat_full,
    global       float *restrict ch0,
    global       float *restrict ch1,
    global       float *restrict ch2,
    global       float *restrict ch3,
    global const float *restrict params,
    const int width,
    const int height)
{
  const int fx = get_global_id(0);
  const int fy = get_global_id(1);
  if(fx >= width || fy >= height) return;

  const int slot_r = fy & 1;
  const int slot_c = fx & 1;
  const int slot   = slot_r | (slot_c << 1);

  const float dr = params[P_DARK_REF0 + slot];

  in_gat_full[(size_t)fy * width + fx] -= dr;

  const int hw = width >> 1;
  const int hp = (fy >> 1) * hw + (fx >> 1);
  if(slot == 0)      ch0[hp] -= dr;
  else if(slot == 1) ch1[hp] -= dr;
  else if(slot == 2) ch2[hp] -= dr;
  else               ch3[hp] -= dr;
}


/* End of §7 GALOSH_RAW_O32 Phase 0-2 kernels (= full Phase 0-2 ported). */


/* End of galosh.cl */
