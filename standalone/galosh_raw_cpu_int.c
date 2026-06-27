/* galosh_raw_cpu_int.c — GALOSH RAW: INT32 fixed-point (Q11.20) CPU reference.
 *
 * Bit-faithful integer port of the FP32 pipeline (galosh_raw_cpu.c): no float,
 * no FP64 — 64-bit intermediates use paired-int32 emulation (fxp_acc).  This is
 * the canonical INT reference; the GPU INT16 port (galosh_int_*.cl) is bit-exact
 * to it.  Q11.20 storage (range +-2048).  See README_RAW_V2.md.
 *
 * Usage: galosh_raw_cpu_int in.bin out.bin W H galosh <strength> <luma> <chroma> <alpha> <sigma_sq>
 *   I/O = raw float32 Bayer in [0,1].  alpha=sigma=0 -> blind.  GALOSH_VERBOSE=1 -> progress.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "galosh_cpu_int.h"

static int g_verbose = 0;  /* GALOSH_VERBOSE env */

/* External LUT storage (declared in galosh_cpu_int.h).  Initialised once at
 * startup via INT-only Taylor / Mercator computations.
 *
 * Note (2026-05-13): fxp_exp_lut[] and fxp_log_lut[] removed as dead code
 * after Bug 6/8 fixes — both fxp_log and fxp_exp now compute directly via
 * Taylor/Mercator + pow2 table.  Only fxp_ln2_q20 and exp_neg_pow2_table
 * remain as cached constants. */
int   fxp_exp_lut_initialized = 0;
fxp32 fxp_exp_neg_pow2_table[17];
int   fxp_log_lut_initialized = 0;
fxp32 fxp_ln2_q20 = 0;
fxp32 fxp_kaiser_2d[64];
int   fxp_kaiser_initialized = 0;
fxp32 fxp_factorial_log[FXP_FACTORIAL_LOG_SIZE];
int   fxp_factorial_log_initialized = 0;

/* Foi-exact inverse GAT lookup table.  Built once per image in Phase 0 after
 * α / σ² are estimated.  Replaces fxp_gat_inverse_algebraic in Phase 10. */
static fxp_gat_inv_table_t g_gat_inv_table = { .valid = 0 };

/* Build Foi-exact inverse GAT LUT via Poisson × Gauss-Hermite quadrature.
 * Mirrors gat_build_inverse_table (galosh_cpu.h:1180) in INT.  Cost: ~50ms.
 * Caller must call after fxp_gat_precompute with the same params.
 *
 * Skips build if α < 1e-4 (= Phase 0 likely floored / failed to estimate),
 * leaving table.valid=0 → fxp_gat_inverse_exact falls back to algebraic.
 * For super-dark scenes (s14/s29/s34, mean ~0.003) where Phase 0 hits the
 * adaptive bin range floor, the LUT with bogus params produced wildly wrong
 * d ranges (e.g., [1155, 786] reversed) → catastrophic Phase 10 output. */
static void fxp_gat_build_inverse_table(const fxp_gat_params *gat_p) {
  if(gat_p->alpha < fxp_from_float(1e-4f)) {
    g_gat_inv_table.valid = 0;
    if(g_verbose) fprintf(stderr, "  Foi LUT skipped (α=%.6f below threshold, will use algebraic)\n",
            fxp_to_float(gat_p->alpha));
    return;
  }
  fxp_factorial_log_init();

  /* sqrt(2) × σ_raw — pre-computed factor for Gauss-Hermite noise. */
  fxp32 sqrt2_sigma = fxp_mul(FXP_SQRT2_Q20, gat_p->sigma_raw);

  /* Precompute z_table[g] = sqrt(2) × σ × node[g] for g=1..8 (= 8-point GH,
   * skipping the negligible g=0 and g=9 tails).  z does not depend on k,
   * only on σ and the GH nodes (= image constants).  Saves 1 fxp_mul per
   * (k × g) iteration in the inner loop. */
  fxp32 z_table[8];
  for(int g = 0; g < 8; g++) {
    z_table[g] = fxp_mul(sqrt2_sigma, fxp_gh_nodes_q20[g + 1]);
  }

  for(int i = 0; i < FXP_GAT_INV_TABLE_SIZE; i++) {
    /* x_val = i / (N-1) in Q11.20.  Compute via fxp_acc to avoid
     * intermediate i × FXP_ONE overflow at i ≈ 4095. */
    fxp_acc xtmp = fxp_acc_zero();
    fxp_acc_madd(&xtmp, i, FXP_ONE);
    fxp32 x_val = fxp_acc_div_i32(&xtmp, FXP_GAT_INV_TABLE_SIZE - 1);

    /* λ = x / α.  Direct division avoids fxp_recip(α) overflow for small α. */
    fxp32 alpha_safe = (gat_p->alpha < 1) ? 1 : gat_p->alpha;
    fxp32 lambda = fxp_div_q20(x_val, alpha_safe);

    /* k_max ≈ λ + 6·sqrt(λ) + 10.  6σ Gaussian tail covers > 99.9999% of
     * probability mass; remaining tail contributes < 1e-9 to expected_gat.
     * Old 8σ bound was overly conservative (= 25% more k iterations).
     * Note: no cap at FACTORIAL_LOG_SIZE because fxp_poisson_log_pdf uses
     * Gauss CLT for λ > 100 (no log(k!) lookup needed for those k). */
    int lambda_int = lambda >> FXP_FRAC;
    fxp32 sqrt_lambda = (lambda > 0) ? fxp_sqrt(lambda) : 0;
    int sqrt_lambda_int = sqrt_lambda >> FXP_FRAC;
    int k_max = lambda_int + 6 * sqrt_lambda_int + 10;

    /* Sum over Poisson distribution: expected_gat = Σ_k P(k|λ) × E[T(noisy)|k].
     * Uses fxp_poisson_log_pdf which hybrid-switches between exact Poisson
     * (λ ≤ 30) and Gauss CLT (λ > 30) to avoid log(k!) overflow / fxp_log
     * accumulation error. */
    fxp_acc expected_gat = fxp_acc_zero();
    const fxp32 break_thresh = fxp_from_float(-16.0f);  /* break when log_prob < -16 */

    for(int k = 0; k <= k_max; k++) {
      fxp32 log_prob = fxp_poisson_log_pdf(lambda, k);
      if(log_prob < break_thresh && k > lambda_int + 1) break;
      fxp32 prob = fxp_exp(log_prob);
      if(prob <= 0) continue;

      /* Inner: 8-point Gauss-Hermite over Gaussian noise of stddev σ.
       * Skip g=0 and g=9 (= |node|=3.44, weight=7.6e-6, contributes <1e-5
       * to E[T]).  Uses precomputed z_table to avoid per-k fxp_mul of
       * sqrt(2)·σ·node (= k-independent). */
      fxp_acc eg_acc = fxp_acc_zero();
      fxp32 k_alpha = (fxp32)k * gat_p->alpha;
      for(int g = 0; g < 8; g++) {
        fxp32 noisy_y = k_alpha + z_table[g];
        fxp32 T = fxp_gat_forward(noisy_y, gat_p);
        fxp_acc_madd(&eg_acc, fxp_gh_weights_q20[g + 1], T);
      }
      /* Extract eg as Q11.20 (= shift >>20) and apply 1/sqrt(π) factor. */
      /* Extract eg as Q11.20 (= shift >>20) and apply 1/sqrt(π) factor. */
      fxp32 eg_q = fxp_mul(FXP_INV_SQRT_PI_Q20, fxp_acc_to_fxp32(&eg_acc));

      /* expected_gat += prob × eg. */
      fxp_acc_madd(&expected_gat, prob, eg_q);
    }

    fxp32 expected_gat_q = fxp_acc_to_fxp32(&expected_gat);
    g_gat_inv_table.d[i] = expected_gat_q;
    g_gat_inv_table.x[i] = x_val;
  }
  g_gat_inv_table.d_min = g_gat_inv_table.d[0];
  g_gat_inv_table.d_max = g_gat_inv_table.d[FXP_GAT_INV_TABLE_SIZE - 1];
  g_gat_inv_table.alpha = gat_p->alpha;
  g_gat_inv_table.sigma_raw = gat_p->sigma_raw;
  g_gat_inv_table.y_break = gat_p->y_break;
  g_gat_inv_table.t_break = gat_p->t_break;
  g_gat_inv_table.valid = 1;

  if(g_verbose) fprintf(stderr, "  Phase 10 Foi LUT built: D range [%.4f, %.4f] (%d entries)\n",
          fxp_to_float(g_gat_inv_table.d_min), fxp_to_float(g_gat_inv_table.d_max),
          FXP_GAT_INV_TABLE_SIZE);
}

/* Foi-exact inverse GAT lookup: binary search + linear interp in d[].
 * Below d_min: linear branch (matches algebraic for low D).
 * Above d_max: algebraic (Foi → algebraic in high-signal limit). */
static inline fxp32 fxp_gat_inverse_exact(fxp32 D, const fxp_gat_params *gat_p) {
  if(!g_gat_inv_table.valid) return fxp_gat_inverse_algebraic(D, gat_p);

  /* Below table: use linear-branch inverse (same as algebraic linear branch). */
  if(D <= g_gat_inv_table.d_min) {
    fxp32 dd = D - g_gat_inv_table.t_break;
    return g_gat_inv_table.y_break + fxp_mul(g_gat_inv_table.sigma_raw, dd);
  }
  /* Above table: algebraic (high-signal limit where Foi ≈ algebraic). */
  if(D >= g_gat_inv_table.d_max) {
    return fxp_gat_inverse_algebraic(D, gat_p);
  }
  /* Binary search for D's position in the monotone d[] array. */
  int lo = 0, hi = FXP_GAT_INV_TABLE_SIZE - 1;
  while(lo < hi - 1) {
    int mid = (lo + hi) >> 1;
    if(g_gat_inv_table.d[mid] <= D) lo = mid;
    else hi = mid;
  }
  /* Linear interp between (d[lo], x[lo]) and (d[hi], x[hi]). */
  fxp32 d0 = g_gat_inv_table.d[lo], d1 = g_gat_inv_table.d[hi];
  fxp32 x0 = g_gat_inv_table.x[lo], x1 = g_gat_inv_table.x[hi];
  fxp32 dd_diff = d1 - d0;
  if(dd_diff <= 0) return x0;
  fxp32 t = fxp_div_q20(D - d0, dd_diff);     /* normalized 0..1 in Q11.20 */
  return x0 + fxp_mul(t, x1 - x0);
}

/* Variant selection (single char, mirrors FP32 main). */

/* ============================================================================
 * Phase 0 — noise estimation (Foi-Alenius blind α/σ² for Poisson-Gauss).
 *
 * Mirrors galosh_estimate_noise() in galosh_cpu.h.  Sub-steps:
 *   0a. Block stats: per-channel 8x8 block mean + Laplacian-MAD variance
 *   0b. Bin envelope: 32 mean-bins × 128 var-sub-bins histogram → lower env
 *   0c. Huber WLS 5-iter: robust fit Var = α * mean + σ²
 *   0d. Dark refine: 4096-bin histograms → refine σ² from dark pixels
 *
 * STREAMING design (2026-05-10): refactored to NO per-block storage.
 * Two image passes accumulate mean-bin × var-sub-bin histograms directly
 * (= matches GPU galosh.cl K0b kernel pattern).  Aggregate state:
 *   cnt_bin[32] + msum_bin[32] + vmin/vmax[32]   ~640 B
 *   vhist_bin[32][128]                            16 KB
 *   thresh_hist + lap_hist (Phase 0d, separate)   32 KB (consecutive use)
 * Total instantaneous LDS use ≤ 17 KB → fits 32 KB ISP-streaming budget.
 * No per-block float array (was 520 KB at 4K).
 *
 * NOT YET IMPLEMENTED in this Stage 1c chunk:
 *   - LUT build (Phase 9-10 inverse): TODO
 * ========================================================================== */

#define NE_NBINS         32      /* number of mean-bins */
#define NE_BLOCK_SZ      8       /* block size for stats (per CFA channel) */
#define NE_VAR_BINS      128     /* var sub-bins inside each mean-bin */
#define NE_DARK_HIST_BINS 4096   /* dark refine histogram resolution */
#define NE_DARK_LAP_MAX_F 0.1f   /* Lap range ceiling for dark refine */

#ifdef GALOSH_INT_KEEP_DEPRECATED_ARRAY_PHASE0
/* ----------------------------------------------------------------------------
 * [DEPRECATED 2026-05-10] phase0a/0b array-based block stats.
 *
 * Replaced by streaming pattern (= matches GPU galosh.cl K0b kernel) to
 * eliminate per-block array storage (520 KB at 4K, exceeds 32 KB ISP-LDS
 * budget).  See fxp_estimate_noise() for streaming impl.
 *
 * Kept under ifdef per `feedback_keep_deprecated_variants` policy:
 * codebase is the long-term memory, archived variants stay visible.
 * -------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------
 * Phase 0a — block stats (block_mean, block_var) per CFA channel.
 *
 * Inputs:  in_q20 (full-res bayer in Q11.20), width, height
 * Outputs: blk_mean[], blk_var[] arrays (Q11.20), each of size 4 * n_bx * n_by
 *          where n_bx = halfwidth/8, n_by = halfheight/8.
 *
 * Per-block:
 *   mean = sum(64 pixels) / 64 = sum >> 6
 *     (Q11.20 sum max = 64 × ±2048 = ±131072, fits INT32.)
 *   Laplacian:
 *     L = pix[x] - 2*pix[x+2] + pix[x+4]   (3-tap, stride 2)
 *     |L| values collected for 64-256 samples
 *     median(|L|) → sigma_lap = median / 0.6745
 *     var = sigma_lap² / 6
 *   Stride between adjacent samples on a CFA plane is 2 (full-res), so we
 *   index raw at (2*y + dy0, 2*x + dx0) for offsets[ch] ∈ {(0,0)(0,1)(1,0)(1,1)}.
 * -------------------------------------------------------------------------- */
static int phase0a_block_stats(const fxp32 *in_q20,
                               int width, int height,
                               fxp32 *blk_mean, fxp32 *blk_var) {
  static const int offsets[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
  const int halfwidth  = (width  + 1) / 2;
  const int halfheight = (height + 1) / 2;
  const int n_bx = halfwidth  / NE_BLOCK_SZ;
  const int n_by = halfheight / NE_BLOCK_SZ;
  if(n_bx < 2 || n_by < 2) return 0;

  /* Q11.20 of 1/0.6745 ≈ 1.482. */
  const fxp32 inv_06745 = fxp_from_float(1.0f / 0.6745f);
  /* Variance precision fix (2026-05-10): for MAD ≈ 1e-3, sigma_lap² ≈ 2e-6
   * truncates to 0 in Q11.20.  Scale all variance by 1024× so values stay
   * above LSB.  alpha_est in WLS becomes 1024× scaled too; phase0c unscales.
   *
   * var_scaled = var × 1024 = (sigma_lap² / 6) × 1024 = (sigma_lap × √(1024/6))²
   * Combined factor: MAD × √(1024/6) / 0.6745 = MAD × 19.359
   * So sigma_lap_scaled = fxp_mul(MAD, scale_const) where scale_const = 19.359. */
  const fxp32 var_scale_combined = fxp_from_float(1.482f * 13.064f);  /* ≈ 19.359 */

  /* |Lap| working buffer (max 8*6 + 6*8 = 96 samples per block). */
  fxp32 laps[256];
  int bi = 0;
  for(int ch = 0; ch < 4; ch++) {
    const int dy0 = offsets[ch][0], dx0 = offsets[ch][1];
    for(int by = 0; by < n_by; by++) {
      for(int bx = 0; bx < n_bx; bx++) {
        const int y0 = by * NE_BLOCK_SZ;
        const int x0 = bx * NE_BLOCK_SZ;

        /* Mean over 64 pixels (block stride is 2 in full-res).
         * Plain int32 sum: max 64 × FXP_ONE × ~3 (real max) = ~2e8, fits. */
        int32_t msum = 0;
        int np = 0;
        for(int y = y0; y < y0 + NE_BLOCK_SZ; y++) {
          for(int x = x0; x < x0 + NE_BLOCK_SZ; x++) {
            const int rr = 2*y + dy0;
            const int cc = 2*x + dx0;
            if(rr < height && cc < width) {
              msum += in_q20[(size_t)rr * width + cc];
              np++;
            }
          }
        }
        if(np <= 0) { blk_mean[bi] = 0; blk_var[bi] = FXP_MAX_INT; bi++; continue; }
        fxp32 bm = (np == 64) ? (msum >> 6) : (msum / np);

        /* Laplacian samples (horizontal then vertical). */
        int nl = 0;
        for(int y = y0; y < y0 + NE_BLOCK_SZ; y++) {
          for(int x = x0; x < x0 + NE_BLOCK_SZ - 2; x++) {
            const int r = 2*y + dy0;
            const int c0 = 2*x + dx0;
            const int c1 = 2*(x+1) + dx0;
            const int c2 = 2*(x+2) + dx0;
            if(c2 >= width || r >= height) continue;
            fxp32 v0 = in_q20[(size_t)r * width + c0];
            fxp32 v1 = in_q20[(size_t)r * width + c1];
            fxp32 v2 = in_q20[(size_t)r * width + c2];
            fxp32 lap = v0 - (v1 << 1) + v2;
            laps[nl++] = (lap < 0) ? -lap : lap;
            if(nl >= 256) goto laps_done;
          }
        }
        for(int y = y0; y < y0 + NE_BLOCK_SZ - 2; y++) {
          for(int x = x0; x < x0 + NE_BLOCK_SZ; x++) {
            const int c = 2*x + dx0;
            const int r0 = 2*y + dy0;
            const int r1 = 2*(y+1) + dy0;
            const int r2 = 2*(y+2) + dy0;
            if(r2 >= height || c >= width) continue;
            fxp32 v0 = in_q20[(size_t)r0 * width + c];
            fxp32 v1 = in_q20[(size_t)r1 * width + c];
            fxp32 v2 = in_q20[(size_t)r2 * width + c];
            fxp32 lap = v0 - (v1 << 1) + v2;
            laps[nl++] = (lap < 0) ? -lap : lap;
            if(nl >= 256) goto laps_done;
          }
        }
laps_done:
        if(nl > 10) {
          fxp32 med = fxp_partial_selection_median(laps, nl);
          /* sigma_lap_scaled = MAD × 19.359 (combined 1.482 × √(1024/6)).
           * Then var_scaled = sigma_lap_scaled² (= var × 1024).  This avoids
           * Q11.20 truncation of small variances. */
          fxp32 sigma_lap_scaled = fxp_mul(med, var_scale_combined);
          blk_var[bi] = fxp_mul(sigma_lap_scaled, sigma_lap_scaled);
        } else {
          blk_var[bi]  = FXP_MAX_INT;     /* invalid: sentinel */
        }
        blk_mean[bi] = bm;
        bi++;
      }
    }
  }
  return bi;
}

/* ----------------------------------------------------------------------------
 * Phase 0b — bin envelope.
 *
 * Inputs:  blk_mean[], blk_var[] arrays of size n_total (Q11.20)
 * Outputs: bin_mean_arr[NE_NBINS], bin_var_arr[NE_NBINS],
 *          bin_cnt_arr[NE_NBINS], bin_valid[NE_NBINS], n_valid
 *
 * Algorithm mirrors FP32 lines 695-810:
 *   1. Find global_min/max of blk_mean (filter mean ∈ [0.003, 0.97])
 *   2. Bin width bw = (max - min) / NE_NBINS
 *   3. For each of NE_NBINS bins:
 *      a. Pass 1: count + msum + (vmin, vmax) of blk_var in this bin
 *      b. Pass 2: histogram blk_var into NE_VAR_BINS sub-bins
 *      c. Cumsum to find p5_bin, p20_bin
 *      d. bin-center weighted sum over [p5, p20] = bin_var_arr[b]
 *      e. bin_mean_arr[b] = msum / cnt
 * -------------------------------------------------------------------------- */
static int phase0b_bin_envelope(const fxp32 *blk_mean, const fxp32 *blk_var,
                                int n_total,
                                fxp32 *bin_mean_arr, fxp32 *bin_var_arr,
                                int   *bin_cnt_arr,  int   *bin_valid) {
  const fxp32 mean_lo = fxp_from_float(0.003f);   /* low-end filter */
  const fxp32 mean_hi = fxp_from_float(0.97f);    /* high-end filter */

  /* Find global min/max of blk_mean (within filter). */
  fxp32 gmin = FXP_MAX_INT, gmax = 0;
  for(int i = 0; i < n_total; i++) {
    fxp32 m = blk_mean[i];
    if(m > mean_lo && m < mean_hi) {
      if(m < gmin) gmin = m;
      if(m > gmax) gmax = m;
    }
  }
  if(gmin >= gmax) return 0;
  const fxp32 grange = gmax - gmin;
  const fxp32 bw = grange / NE_NBINS;
  if(bw <= 0) return 0;
  /* Recip of bw for assigning bin index quickly. */
  const fxp32 inv_bw = fxp_recip(bw);

  int n_valid_bins = 0;
  for(int b = 0; b < NE_NBINS; b++) {
    bin_valid[b] = 0;
    fxp32 bin_lo = gmin + (fxp32)b * bw;
    fxp32 bin_hi = bin_lo + bw;

    /* Pass 1: count / msum / vmin / vmax.
     * msum: sum of Q11.20 values, use fxp_acc to avoid INT32 overflow. */
    fxp_acc msum = fxp_acc_zero();
    int cnt = 0;
    fxp32 vmin = FXP_MAX_INT, vmax = 0;
    for(int i = 0; i < n_total; i++) {
      fxp32 m = blk_mean[i];
      fxp32 v = blk_var[i];
      if(m >= bin_lo && m < bin_hi
         && m > mean_lo && m < mean_hi
         && v != FXP_MAX_INT) {
        fxp_acc_add_i32(&msum, m);
        cnt++;
        if(v < vmin) vmin = v;
        if(v > vmax) vmax = v;
      }
    }
    if(cnt < 20) continue;

    /* Pass 2: histogram of var into NE_VAR_BINS bins of [vmin, vmax]. */
    int vhist[NE_VAR_BINS]; for(int k = 0; k < NE_VAR_BINS; k++) vhist[k] = 0;
    fxp32 vrange = vmax - vmin;
    if(vrange < 1) vrange = 1;
    /* vscale = NE_VAR_BINS / vrange.  We compute vbin = (v - vmin) * vscale.
     * To stay in Q11.20 + INT32 range, use fxp_recip(vrange) * NE_VAR_BINS. */
    fxp32 inv_vrange = fxp_recip(vrange);
    /* multiplier = NE_VAR_BINS * inv_vrange in Q11.20. */
    fxp32 nbins_q = (fxp32)NE_VAR_BINS << FXP_FRAC;
    fxp32 vscale_q = fxp_mul(nbins_q, inv_vrange) >> FXP_FRAC;
    /* vscale_q is a raw integer (NOT Q11.20).  vbin = (v-vmin) * vscale_q,
     * where (v-vmin) is Q11.20, vscale_q is integer multiplier. */
    if(vscale_q < 1) vscale_q = 1;

    for(int i = 0; i < n_total; i++) {
      fxp32 m = blk_mean[i];
      fxp32 v = blk_var[i];
      if(m >= bin_lo && m < bin_hi
         && m > mean_lo && m < mean_hi
         && v != FXP_MAX_INT) {
        /* vbin_idx = (v - vmin) * NE_VAR_BINS / vrange.
         * Simplified: frac = (v-vmin) * (1/vrange) is Q11.20 ∈ [0, 1].
         * vbin = frac * 128 = frac >> (FXP_FRAC - 7) = frac >> 13. */
        fxp32 dv = v - vmin;
        fxp32 frac = fxp_mul(dv, inv_vrange);
        int vbin = (frac >= 0) ? ((int)frac >> 13) : 0;
        if(vbin < 0) vbin = 0;
        if(vbin >= NE_VAR_BINS) vbin = NE_VAR_BINS - 1;
        vhist[vbin]++;
      }
    }

    /* Cumsum: find p5 and p20 var bins. */
    int p5_target  = cnt / 20;
    int p20_target = cnt / 5;
    int cum = 0;
    int p5_bin = 0, p20_bin = NE_VAR_BINS - 1;
    int found_p5 = 0;
    for(int i = 0; i < NE_VAR_BINS; i++) {
      cum += vhist[i];
      if(!found_p5 && cum >= p5_target) { p5_bin = i; found_p5 = 1; }
      if(cum >= p20_target) { p20_bin = i; break; }
    }

    /* Lower-envelope mean var: weighted bin-center sum over [p5, p20]. */
    fxp_acc vsum_acc = fxp_acc_zero();
    int vcnt = 0;
    for(int i = p5_bin; i <= p20_bin; i++) {
      /* bin_center = vmin + (i + 0.5) * (vrange / NE_VAR_BINS)
       *            = vmin + (i + 0.5) * vrange >> 7  (since 128 = 2^7)  */
      fxp32 i_q = (fxp32)i << FXP_FRAC;
      fxp32 i_plus_half = i_q + (FXP_ONE >> 1);
      /* (i + 0.5) * vrange / 128 in Q11.20: */
      fxp32 t = fxp_mul(i_plus_half, vrange) >> 7;
      fxp32 bin_center = vmin + t;
      int n = vhist[i];
      for(int k = 0; k < n; k++) fxp_acc_add_i32(&vsum_acc, bin_center);
      vcnt += n;
    }
    if(vcnt > 0) {
      /* vsum_acc was built via fxp_acc_add_i32 (NOT madd) — extract w/o shift. */
      fxp32 vsum_q = fxp_acc_extract_q20(&vsum_acc);
      bin_var_arr[b] = vsum_q / vcnt;
    } else {
      bin_var_arr[b] = vmin;
    }
    bin_mean_arr[b] = fxp_acc_extract_q20(&msum) / cnt;
    bin_cnt_arr[b]  = (vcnt > 0) ? vcnt : 1;
    bin_valid[b]    = 1;
    n_valid_bins++;
    (void)inv_bw;     /* suppress unused warning if pruned */
    (void)vscale_q;
  }
  return n_valid_bins;
}

#endif  /* GALOSH_INT_KEEP_DEPRECATED_ARRAY_PHASE0 */

/* ----------------------------------------------------------------------------
 * Phase 0c — Huber WLS 5-iter slope/intercept fit.
 *
 * Solves  Var = α·mean + σ²  via robust weighted least squares:
 *   1. iter 0: plain WLS (w = bin_count)
 *   2. iter 1-4: Huber-weighted (w *= huber_k / |residual| if |r| > huber_k)
 *
 * Critical numerical concern: catastrophic cancellation in the 2×2 normal
 * equations.  We use multi-precision fxp_acc throughout for sums; for the
 * final 2×2 solve we extract scaled fxp32 values and divide via fxp_recip.
 *
 * Q-format convention here:
 *   - bin_mean ∈ [0.003, 0.97]                       (Q11.20 representable)
 *   - bin_var  ∈ [1e-7, 1e-3]                        (low-bin precision risk)
 *   - alpha ∈ [1e-4, 1e-2]                           (Q11.20: ≥100 LSBs)
 *   - sigma_sq ∈ [1e-8, 1e-4]                        (Q11.20: 0..100 LSBs;
 *                                                     low-precision floor —
 *                                                     accept ~10% relative err
 *                                                     for σ² near 1e-7).
 *
 * For Stage 1c initial version, we omit the Huber re-weighting (= use plain
 * WLS).  Production accuracy requires Huber; will be added in Stage 1d after
 * verifying baseline matches FP32.
 * -------------------------------------------------------------------------- */
/* phase0c helper: one centered-WLS pass with given Huber threshold (or
 * INT_MAX = no clipping for iter 0). */
static void wls_centered_pass(const fxp32 *bin_mean_arr,
                              const fxp32 *bin_var_arr,
                              const int *bin_cnt_arr,
                              const int *bin_valid,
                              int n_total_bins,
                              fxp32 alpha_in, fxp32 sigma_sq_in,
                              fxp32 huber_k_q,
                              fxp32 *alpha_out, fxp32 *sigma_sq_out) {
  /* Sw stored as plain int32 (sum of integer block counts × Huber ratio
   * in Q11.20).  Up to ~16000 across 32 bins, fits int32 easily.
   *
   * Sx, Sy stored as fxp_acc (64-bit equivalent): Σ(w_q11.20 × x_q11.20),
   * conceptually 64-bit before the divide by Sw.  Previous code stored Sw
   * as fxp_acc too, then extracted to int32 (= silent saturation for >2048
   * total weight).  Solution: keep Sw as plain int32, use fxp_acc_div_i32
   * to compute mean = Sx / Sw exactly.
   *
   * Sw を plain int32 で保持し fxp_acc_extract_q20 の saturation を回避。 */
  fxp_acc Sx_acc = fxp_acc_zero();
  fxp_acc Sy_acc = fxp_acc_zero();
  fxp_acc Sw_acc = fxp_acc_zero();    /* Σw_q11.20, may include Huber fractions */

  /* Weight down-scale (root fix 2026-06-22): on a full frame the per-bin block
   * count reaches ~1e5, so w_q = count × FXP_ONE overflows int32 once count >
   * 2047 → garbage weights → Sxx_c (Σ w·xc²) overflows to NEGATIVE → the slope
   * solve below is skipped → α stays at its unscaled init and the later /1024
   * unscale drives it to ~0 → all-black output on bright full-frame scenes
   * (SIDD 23/80, RawNIND 121/1288; 512² crops mostly escaped).  WLS is scale-
   * invariant in the weight, so right-shift every count by a common wshift that
   * keeps the largest weight inside Q11.20 (count >> wshift < 1024); the Huber
   * fractional re-weighting still applies on the scaled w_q, and the var-domain
   * /1024 unscale is independent of the weight so stays correct.
   * full-frame で bin count 大 → w=count×2²⁰ が int32 overflow → Sxx_c 負 →
   * slope skip → α collapse。重みは scale 不変なので共通 shift で縮小。 */
  int wshift = 0;
  {
    int max_cnt = 0;
    for(int b = 0; b < n_total_bins; b++)
      if(bin_valid[b] && bin_cnt_arr[b] > max_cnt) max_cnt = bin_cnt_arr[b];
    while((max_cnt >> wshift) >= 1024) wshift++;
  }

  /* Pass 1: weighted Sw, Sx, Sy. */
  for(int b = 0; b < n_total_bins; b++) {
    if(!bin_valid[b]) continue;
    int w_int = bin_cnt_arr[b] >> wshift;
    if(w_int < 1) w_int = 1;
    fxp32 w_q = (fxp32)w_int * FXP_ONE;          /* Q11.20 of integer weight */
    /* Huber down-weighting (= no-op if huber_k_q = FXP_MAX_INT). */
    if(huber_k_q < FXP_MAX_INT) {
      fxp32 pred = fxp_mul(alpha_in, bin_mean_arr[b]) + sigma_sq_in;
      fxp32 r = bin_var_arr[b] - pred;
      fxp32 abs_r = (r < 0) ? -r : r;
      if(abs_r > huber_k_q) {
        /* w_q *= huber_k / |resid| (fraction ≤ 1). */
        fxp32 ratio = fxp_mul(huber_k_q, fxp_recip(abs_r));
        w_q = fxp_mul(w_q, ratio);
      }
    }
    fxp_acc_add_i32(&Sw_acc, w_q);
    fxp_acc_add_i32(&Sx_acc, fxp_mul(w_q, bin_mean_arr[b]));
    fxp_acc_add_i32(&Sy_acc, fxp_mul(w_q, bin_var_arr[b]));
  }
  /* Sanity check: Sw must be positive.  Sw_acc.hi positive AND lo > 0,
   * or hi == 0 and lo > 0. */
  if(Sw_acc.hi < 0 || (Sw_acc.hi == 0 && Sw_acc.lo == 0)) {
    *alpha_out = alpha_in; *sigma_sq_out = sigma_sq_in;
    return;
  }
  /* mean_x = Sx / Sw, mean_y = Sy / Sw.  Use fxp_acc_div_acc to handle
   * 64-bit numerator and denominator (Σ(w×x) and Σw can both be large). */
  fxp32 mean_x = fxp_acc_div_acc(&Sx_acc, &Sw_acc);
  fxp32 mean_y = fxp_acc_div_acc(&Sy_acc, &Sw_acc);

  /* Pass 2: centered cross + squared-deviation sums. */
  fxp_acc Sxx_c_acc = fxp_acc_zero();
  fxp_acc Sxy_c_acc = fxp_acc_zero();
  for(int b = 0; b < n_total_bins; b++) {
    if(!bin_valid[b]) continue;
    int w_int = bin_cnt_arr[b] >> wshift;
    if(w_int < 1) w_int = 1;
    fxp32 w_q = (fxp32)w_int * FXP_ONE;
    if(huber_k_q < FXP_MAX_INT) {
      fxp32 pred = fxp_mul(alpha_in, bin_mean_arr[b]) + sigma_sq_in;
      fxp32 r = bin_var_arr[b] - pred;
      fxp32 abs_r = (r < 0) ? -r : r;
      if(abs_r > huber_k_q) {
        fxp32 ratio = fxp_mul(huber_k_q, fxp_recip(abs_r));
        w_q = fxp_mul(w_q, ratio);
      }
    }
    fxp32 xc = bin_mean_arr[b] - mean_x;
    fxp32 yc = bin_var_arr[b] - mean_y;
    /* Fix 2026-06-09 (super-clean tiny-α WLS slope): accumulate the centred
     * cross / square products WITHOUT the intermediate Q11.20 truncation.
     * For super-clean content the variance-vs-mean deviation yc is tiny
     * (~5e-4) so the cross-product xc·yc ≈ 5e-6 — fxp_mul(xc,yc) keeps only
     * ~2-3 significant bits, collapsing Sxy_c and halving the regression
     * slope (α came out 6e-6 vs FP32 1.3e-5 on sewingmachine).  fxp_acc_madd
     * keeps the full Q40 raw product; div_acc divides the two Q40 sums (same
     * scale) → correct slope.  This is stats-engine precision (per-image,
     * not the per-pixel MAC) so wider intermediate is ISP-faithful.
     * 超クリーンで分散偏差の積が極小 → fxp_mul truncate で slope 半減のバグ修正。 */
    fxp32 w_xc = fxp_mul(w_q, xc);
    fxp_acc_madd(&Sxx_c_acc, w_xc, xc);   /* (w·xc)·xc — no xc² truncation */
    fxp_acc_madd(&Sxy_c_acc, w_xc, yc);   /* (w·xc)·yc — no xc·yc truncation */
  }

  fxp32 alpha_est = alpha_in, sigma_sq_est = sigma_sq_in;
  /* α = Sxy_c / Sxx_c (slope of linear regression).  Both are fxp_acc
   * (64-bit) so use fxp_acc_div_acc which shifts both equally to fit
   * int32 before dividing — preserves ratio + avoids fxp_recip overflow
   * on small Sxx_c.  See fxp_acc_div_acc doc-comment. */
  if(Sxx_c_acc.hi > 0 || (Sxx_c_acc.hi == 0 && Sxx_c_acc.lo > 0)) {
    fxp32 new_alpha = fxp_acc_div_acc(&Sxy_c_acc, &Sxx_c_acc);
    if(new_alpha > 0) alpha_est = new_alpha;
  }
  fxp32 new_sigma = mean_y - fxp_mul(alpha_est, mean_x);
  if(new_sigma >= 0) sigma_sq_est = new_sigma;
  *alpha_out = alpha_est;
  *sigma_sq_out = sigma_sq_est;
}

static void phase0c_wls_solve(const fxp32 *bin_mean_arr,
                              const fxp32 *bin_var_arr,
                              const int *bin_cnt_arr,
                              const int *bin_valid,
                              int n_total_bins,
                              fxp32 *alpha_out, fxp32 *sigma_sq_out) {
  /* 5 iterations of Huber-weighted centered WLS.  iter 0 = no clipping
   * (= plain WLS); iter 1-4 use huber_k = 1.345 · MAD(residuals) / 0.6745. */
  fxp32 alpha_est = fxp_from_float(1e-4f);
  fxp32 sigma_sq_est = 1;

  /* CENTERED-FORMULATION WLS (avoids cancellation in normal equations) +
   * 5-iter Huber re-weighting (= matches FP32 reference).
   *
   * iter 0: huber_k = INT_MAX (= plain WLS, no re-weighting)
   * iter 1-4: huber_k = 1.345·MAD(resids)/0.6745 (= robust against outliers)
   *
   * Re-weight rule: w_eff = w · min(1, huber_k / |resid|).
   * w_eff stored as Q11.20 to allow fractional weights post-scaling.
   *
   * Pure INT32, no FP, no INT64.  All accumulators are fxp_acc (paired INT32).
   */

  /* huber-k constant: 1.345 / 0.6745 ≈ 1.99407... in Q11.20. */
  const fxp32 huber_factor_q = fxp_from_float(1.345f / 0.6745f);

  for(int iter = 0; iter < 5; iter++) {
    fxp32 huber_k_q = FXP_MAX_INT;     /* iter 0: no clipping */

    if(iter > 0) {
      /* Compute residuals = |y - (alpha·x + sigma_sq)| in Q11.20. */
      fxp32 resids[NE_NBINS];
      int nr = 0;
      for(int b = 0; b < n_total_bins; b++) {
        if(!bin_valid[b]) continue;
        fxp32 pred = fxp_mul(alpha_est, bin_mean_arr[b]) + sigma_sq_est;
        fxp32 r = bin_var_arr[b] - pred;
        resids[nr++] = (r < 0) ? -r : r;
      }
      if(nr < 2) break;
      /* MAD via partial-selection median.  Note: median takes |·| internally
       * so we already pass non-negative values; that's fine. */
      fxp32 mad = fxp_partial_selection_median(resids, nr);
      huber_k_q = fxp_mul(huber_factor_q, mad);
      if(huber_k_q < 1) huber_k_q = 1;     /* floor */
    }

    /* One centered-WLS pass with current huber_k_q. */
    fxp32 new_alpha, new_sigma_sq;
    wls_centered_pass(bin_mean_arr, bin_var_arr, bin_cnt_arr, bin_valid,
                      n_total_bins, alpha_est, sigma_sq_est, huber_k_q,
                      &new_alpha, &new_sigma_sq);
    if(new_alpha > 0) alpha_est = new_alpha;
    if(new_sigma_sq >= 0) sigma_sq_est = new_sigma_sq;
  }

  /* Unscale: bin_var values were 1024× scaled to avoid Q11.20 truncation.
   * Both alpha and sigma_sq scale linearly with var, so unscale by /1024. */
  alpha_est    = alpha_est    >> 10;     /* /1024 */
  sigma_sq_est = sigma_sq_est >> 10;

  /* Floor α at 1 LSB (≈9.5e-7) so downstream divisions don't hit zero. */
  if(alpha_est < 1) alpha_est = 1;
  *alpha_out    = alpha_est;
  *sigma_sq_out = sigma_sq_est;
}

/* ----------------------------------------------------------------------------
 * Phase 0d — dark refine via histogram percentile.
 *
 * Mirrors FP32 lines 891-989.  Two passes:
 *   1. 4096-bin histogram of pixel values (stride=3 per CFA channel × 4)
 *      → cumsum to find 10th percentile = dark_thresh
 *   2. 4096-bin histogram of |Lap| at dark pixels (stride=1, 4 channels, H+V)
 *      → cumsum to find median = MAD
 *      → sigma_lap = MAD / 0.6745, dark_var = sigma_lap²/6
 *      → refined sigma_sq = max(dark_var - α·dark_mean, 0)
 * -------------------------------------------------------------------------- */
static void phase0d_dark_refine(const fxp32 *in_q20,
                                int width, int height,
                                fxp32 alpha_in,
                                fxp32 *sigma_sq_out) {
  static const int offsets[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
  const int halfwidth  = (width  + 1) / 2;
  const int halfheight = (height + 1) / 2;

  /* Pass 1: thresh histogram of half-res values [0, 1] → 4096 bins. */
  static int thresh_hist[NE_DARK_HIST_BINS];
  for(int i = 0; i < NE_DARK_HIST_BINS; i++) thresh_hist[i] = 0;
  int total_thresh = 0;
  for(int ch = 0; ch < 4; ch++) {
    int dy0 = offsets[ch][0], dx0 = offsets[ch][1];
    for(int y = 0; y < halfheight; y += 3) {
      for(int x = 0; x < halfwidth; x += 3) {
        int rr = 2*y + dy0, cc = 2*x + dx0;
        if(rr >= height || cc >= width) continue;
        fxp32 v = in_q20[(size_t)rr * width + cc];
        /* Bin: v * 4096 in [0, 1].  In Q11.20, v ∈ [0, FXP_ONE], so:
         * bin = (v * NE_DARK_HIST_BINS) >> FXP_FRAC = v >> (FXP_FRAC - 12) */
        int b = (int)(v >> (FXP_FRAC - 12));
        if(b < 0) b = 0;
        if(b >= NE_DARK_HIST_BINS) b = NE_DARK_HIST_BINS - 1;
        thresh_hist[b]++;
        total_thresh++;
      }
    }
  }
  /* Cumsum: 10th percentile. */
  int target = total_thresh / 10;
  int cum = 0, dark_bin = 0;
  for(int i = 0; i < NE_DARK_HIST_BINS; i++) {
    cum += thresh_hist[i];
    if(cum >= target) { dark_bin = i; break; }
  }
  /* dark_thresh in Q11.20 = (dark_bin + 0.5) * FXP_ONE / NE_DARK_HIST_BINS. */
  fxp32 dark_thresh = (fxp32)((dark_bin << (FXP_FRAC - 12)) + (1 << (FXP_FRAC - 13)));
  fxp32 dark_max = dark_thresh + fxp_from_float(0.02f);

  /* Pass 2: |Lap| at dark triplets, 4096 bins of [0, NE_DARK_LAP_MAX_F]. */
  static int lap_hist[NE_DARK_HIST_BINS];
  for(int i = 0; i < NE_DARK_HIST_BINS; i++) lap_hist[i] = 0;
  int total_lap = 0;
  /* lap_scale = NE_DARK_HIST_BINS / 0.1 = 40960.  In Q11.20 multiplier:
   * to convert lap (Q11.20, ∈ [0, 0.1]) to bin: bin = lap * (4096/0.1)
   * = lap * 40960.  In raw INT: bin = lap * 40960 / 2^20 = lap * 40960 >> 20. */
  for(int ch = 0; ch < 4; ch++) {
    int dy0 = offsets[ch][0], dx0 = offsets[ch][1];
    /* Horizontal */
    for(int y = 0; y < halfheight; y++) {
      for(int x = 0; x < halfwidth - 2; x++) {
        int r = 2*y + dy0;
        int c0 = 2*x + dx0, c1 = 2*(x+1) + dx0, c2 = 2*(x+2) + dx0;
        if(r >= height || c2 >= width) continue;
        fxp32 v0 = in_q20[(size_t)r * width + c0];
        fxp32 v1 = in_q20[(size_t)r * width + c1];
        fxp32 v2 = in_q20[(size_t)r * width + c2];
        if(v0 > dark_max || v1 > dark_max || v2 > dark_max) continue;
        fxp32 lap = v0 - (v1 << 1) + v2;
        if(lap < 0) lap = -lap;
        /* bin = lap * 40960 >> FXP_FRAC */
        /* bin = lap * 40960 >> 20.  Splitting to avoid INT32 overflow:
         *   40960 = 40 * 1024, so (lap*40960)>>20 = (lap*40)>>10.
         *   For lap_q max ~Q11.20 of 0.1 = 104857: 104857*40 = 4.2e6 fits. */
        int b = (int)((lap * 40) >> 10);
        if(b < 0) b = 0;
        if(b >= NE_DARK_HIST_BINS) b = NE_DARK_HIST_BINS - 1;
        lap_hist[b]++; total_lap++;
      }
    }
    /* Vertical */
    for(int y = 0; y < halfheight - 2; y++) {
      for(int x = 0; x < halfwidth; x++) {
        int c = 2*x + dx0;
        int r0 = 2*y + dy0, r1 = 2*(y+1) + dy0, r2 = 2*(y+2) + dy0;
        if(r2 >= height || c >= width) continue;
        fxp32 v0 = in_q20[(size_t)r0 * width + c];
        fxp32 v1 = in_q20[(size_t)r1 * width + c];
        fxp32 v2 = in_q20[(size_t)r2 * width + c];
        if(v0 > dark_max || v1 > dark_max || v2 > dark_max) continue;
        fxp32 lap = v0 - (v1 << 1) + v2;
        if(lap < 0) lap = -lap;
        /* bin = lap * 40960 >> 20.  Splitting to avoid INT32 overflow:
         *   40960 = 40 * 1024, so (lap*40960)>>20 = (lap*40)>>10.
         *   For lap_q max ~Q11.20 of 0.1 = 104857: 104857*40 = 4.2e6 fits. */
        int b = (int)((lap * 40) >> 10);
        if(b < 0) b = 0;
        if(b >= NE_DARK_HIST_BINS) b = NE_DARK_HIST_BINS - 1;
        lap_hist[b]++; total_lap++;
      }
    }
  }
  if(total_lap < 100) return;     /* keep input sigma_sq if too few darks */

  /* Median: 50th percentile of |Lap|. */
  int target_med = total_lap / 2;
  int cum_l = 0, med_bin = 0;
  for(int i = 0; i < NE_DARK_HIST_BINS; i++) {
    cum_l += lap_hist[i];
    if(cum_l >= target_med) { med_bin = i; break; }
  }
  /* mad_q = (med_bin + 0.5) * NE_DARK_LAP_MAX_F / NE_DARK_HIST_BINS in Q11.20.
   * Step = 0.1 / 4096 ≈ 2.4e-5.  In Q11.20, step = FXP_ONE * 0.1 / 4096
   * = (FXP_ONE / 40960).  Since FXP_ONE = 2^20, step = 2^20 / 40960 = 25.6,
   * round to 26. */
  fxp32 mad_q = ((fxp32)med_bin * (FXP_ONE / 40960)) + (FXP_ONE / 81920);
  /* Use same 1024× variance scaling as phase0a to avoid Q11.20 truncation of
   * small dark_var values (~1e-7).  Compute dark_var_scaled = sigma_lap² * 170,
   * then unscale at end via >>10. */
  const fxp32 var_scale_combined = fxp_from_float(1.482f * 13.064f);
  fxp32 sigma_lap_scaled = fxp_mul(mad_q, var_scale_combined);
  fxp32 dark_var_scaled  = fxp_mul(sigma_lap_scaled, sigma_lap_scaled);
  /* dark_mean = dark_thresh / 2 = dark_thresh >> 1. */
  fxp32 dark_mean = dark_thresh >> 1;
  /* refined sigma_sq = max(dark_var - alpha * dark_mean, 0).
   * In scaled space: dyn_scaled = dark_var_scaled - (alpha << 10) * dark_mean.
   * (alpha is unscaled; multiply by 1024 to bring into scaled domain.) */
  fxp32 alpha_scaled = (alpha_in <= (FXP_MAX_INT >> 10)) ? (alpha_in << 10)
                                                          : FXP_MAX_INT;
  fxp32 dyn_scaled = dark_var_scaled - fxp_mul(alpha_scaled, dark_mean);
  if(dyn_scaled < 0) dyn_scaled = 0;
  *sigma_sq_out = dyn_scaled >> 10;       /* unscale back to Q11.20 */
}

/* ----------------------------------------------------------------------------
 * Helper: compute one block's (mean, var) on demand.
 * Used in two passes of the streaming Phase 0 (no per-block storage).
 *
 * Returns 0 if block invalid (out-of-bounds or insufficient Lap samples).
 * -------------------------------------------------------------------------- */
static int compute_block_stats(const fxp32 *in_q20, int width, int height,
                               int ch, int by, int bx,
                               fxp32 *out_mean, fxp32 *out_var,
                               fxp32 var_scale_combined) {
  static const int offsets[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
  const int dy0 = offsets[ch][0], dx0 = offsets[ch][1];
  const int y0  = by * NE_BLOCK_SZ;
  const int x0  = bx * NE_BLOCK_SZ;

  int32_t msum = 0;
  int np = 0;
  for(int y = y0; y < y0 + NE_BLOCK_SZ; y++) {
    for(int x = x0; x < x0 + NE_BLOCK_SZ; x++) {
      int rr = 2*y + dy0, cc = 2*x + dx0;
      if(rr < height && cc < width) {
        msum += in_q20[(size_t)rr * width + cc];
        np++;
      }
    }
  }
  if(np <= 0) return 0;
  *out_mean = (np == 64) ? (msum >> 6) : (msum / np);

  fxp32 laps[256];
  int nl = 0;
  for(int y = y0; y < y0 + NE_BLOCK_SZ; y++) {
    for(int x = x0; x < x0 + NE_BLOCK_SZ - 2; x++) {
      int r = 2*y + dy0;
      int c0 = 2*x + dx0, c1 = 2*(x+1) + dx0, c2 = 2*(x+2) + dx0;
      if(c2 >= width || r >= height) continue;
      fxp32 v0 = in_q20[(size_t)r * width + c0];
      fxp32 v1 = in_q20[(size_t)r * width + c1];
      fxp32 v2 = in_q20[(size_t)r * width + c2];
      fxp32 lap = v0 - (v1 << 1) + v2;
      laps[nl++] = (lap < 0) ? -lap : lap;
      if(nl >= 256) goto stat_lap_done;
    }
  }
  for(int y = y0; y < y0 + NE_BLOCK_SZ - 2; y++) {
    for(int x = x0; x < x0 + NE_BLOCK_SZ; x++) {
      int c = 2*x + dx0;
      int r0 = 2*y + dy0, r1 = 2*(y+1) + dy0, r2 = 2*(y+2) + dy0;
      if(r2 >= height || c >= width) continue;
      fxp32 v0 = in_q20[(size_t)r0 * width + c];
      fxp32 v1 = in_q20[(size_t)r1 * width + c];
      fxp32 v2 = in_q20[(size_t)r2 * width + c];
      fxp32 lap = v0 - (v1 << 1) + v2;
      laps[nl++] = (lap < 0) ? -lap : lap;
      if(nl >= 256) goto stat_lap_done;
    }
  }
stat_lap_done:
  if(nl <= 10) return 0;
  fxp32 med = fxp_partial_selection_median(laps, nl);
  fxp32 sigma_lap_scaled = fxp_mul(med, var_scale_combined);
  *out_var = fxp_mul(sigma_lap_scaled, sigma_lap_scaled);   /* var × 1024 */
  return 1;
}

/* ----------------------------------------------------------------------------
 * fxp_estimate_noise — Phase 0 orchestrator (STREAMING).
 *
 * No per-block array; iterates blocks twice and accumulates histograms in
 * fixed-size LDS-friendly buffers (~17 KB total).  Mirrors GPU galosh.cl K0b.
 * -------------------------------------------------------------------------- */
static void fxp_estimate_noise(const fxp32 *in_q20,
                               int width, int height,
                               fxp32 *alpha_out, fxp32 *sigma_sq_out) {
  *alpha_out    = fxp_from_float(1e-4f);
  *sigma_sq_out = 1;

  const int halfwidth  = (width  + 1) / 2;
  const int halfheight = (height + 1) / 2;
  const int n_bx = halfwidth  / NE_BLOCK_SZ;
  const int n_by = halfheight / NE_BLOCK_SZ;
  const int total_blocks = 4 * n_bx * n_by;
  if(total_blocks < 100) return;

  /* Variance scaling factor (= same as old phase0a). */
  const fxp32 var_scale_combined = fxp_from_float(1.482f * 13.064f);

  /* ============== Pre-pass: input-adaptive bin range ==============
   *
   * Original hardcoded [0.003, 0.97] fails for narrow-range scenes (e.g.
   * SIDD dark captures with input range [0.01, 0.13]) — most blocks cluster
   * in <4 mean-bins, the n_valid<4 fallback fires, and Phase 0 returns wrong
   * defaults α=1e-4 instead of the true ~1e-3.  Downstream this corrupts
   * Phase 2 dark_ref (becomes ≈ full signal mean → Phase 3+ collapsed to 0).
   *
   * Observed in SIDD val: s03/s11/s25 (dark, max<0.34) all regressed to
   * 24 dB INT vs FP32 56 dB.  v1 was hiding this via WHT 8x8 DC overflow
   * which incidentally short-circuited the bad α downstream.
   *
   * Fix: 256-bin coarse histogram of observed block means over [0, 1], then
   * take 5/95-th percentile as adaptive [mean_lo, mean_hi].  This puts all
   * NE_NBINS=32 fine bins where the data actually lives → ≥4 valid bins
   * even for narrow-range inputs.  Cost: ~16 KB int hist (LDS-friendly) +
   * one extra pass through blocks (same data as Pass 1).
   *
   * 暗い scene で hardcoded [0.003, 0.97] が広すぎて bin が荒くなり、
   * 有効 bin < 4 で defaults fallback していたバグの修正。観測 block mean
   * の 5/95 percentile を NE_NBINS の bin 範囲に採用する。 */
  const int PREPASS_BINS = 256;
  /* m in Q11.20, range [0, 1] (= [0, FXP_ONE]).  bin = m * 256 / FXP_ONE
   * = m >> (FXP_FRAC - 8) = m >> 12.  Step size = FXP_ONE >> 8 = 4096.
   *
   * Filter: exclude blocks with mean ≥ 0.97 (= saturation zone where
   * noise variance is artificially zero due to pixel clipping at 1.0).
   * Without this filter, scenes with >5% saturated pixels (e.g. SIDD s09
   * high-noise capture) have p95 percentile = saturation bin → adaptive
   * mean_hi reaches 1.0 → saturated blocks pollute WLS → α=0 (= negative
   * slope from saturation outliers dominating).  Matches FP32's
   * `bm < 0.97f` filter at galosh_cpu.h:755/773.
   *
   * 飽和域 (mean ≥ 0.97) を除外して percentile 計算する。s09 等の高ノイズ
   * scene で >5% saturated → p95 が飽和 bin に届き mean_hi=1.0 → WLS で
   * α=負の傾き認識する FP32 と同等の処理。 */
  const fxp32 sat_threshold = fxp_from_float(0.97f);
  static int prepass_hist[256];
  for(int i = 0; i < PREPASS_BINS; i++) prepass_hist[i] = 0;
  int prepass_total = 0;
  for(int ch = 0; ch < 4; ch++) {
    for(int by = 0; by < n_by; by++) {
      for(int bx = 0; bx < n_bx; bx++) {
        fxp32 m, v;
        if(!compute_block_stats(in_q20, width, height, ch, by, bx,
                                &m, &v, var_scale_combined)) continue;
        if(m >= sat_threshold) continue;          /* skip saturated blocks */
        int b = (int)(m >> 12);
        if(b < 0) b = 0;
        if(b >= PREPASS_BINS) b = PREPASS_BINS - 1;
        prepass_hist[b]++;
        prepass_total++;
      }
    }
  }
  if(prepass_total < 100) return;

  /* 5/95-th percentile bin boundaries. */
  int p5_target  = prepass_total / 20;
  int p95_target = prepass_total - p5_target;
  int p5_bin = 0, p95_bin = PREPASS_BINS - 1;
  {
    int cum = 0;
    int found_p5 = 0;
    for(int b = 0; b < PREPASS_BINS; b++) {
      cum += prepass_hist[b];
      if(!found_p5 && cum >= p5_target) { p5_bin = b; found_p5 = 1; }
      if(cum >= p95_target) { p95_bin = b; break; }
    }
  }
  /* Convert percentile bins to Q11.20: bin i covers [i*4096, (i+1)*4096). */
  fxp32 mean_lo = (fxp32)p5_bin << 12;
  fxp32 mean_hi = (fxp32)(p95_bin + 1) << 12;
  /* Do NOT floor mean_lo at 0.003 — super-dark SIDD scenes (s14/s29/s34,
   * mean ~0.003) have actual block-mean range entirely below 0.003 and a
   * floor would exclude them all → "0 valid bins" → bad defaults.
   * Ensure only that mean_hi - mean_lo ≥ NE_NBINS LSBs (= bin width ≥ 1
   * raw int) so the integer division in Pass1 stays well-defined. */
  if(mean_hi <= mean_lo + (fxp32)NE_NBINS) mean_hi = mean_lo + (fxp32)NE_NBINS;
  /* Defense in depth: cap mean_hi at saturation threshold even if percentile
   * somehow lands above (e.g. extreme distributions). */
  if(mean_hi > sat_threshold) mean_hi = sat_threshold;
  if(mean_lo < 0) mean_lo = 0;
  const fxp32 mean_bw = (mean_hi - mean_lo) / NE_NBINS;
  if(mean_bw < 1) {
    /* Adaptive range collapsed (single-bin input).  Fall back to defaults. */
    if(g_verbose) fprintf(stderr, "  [phase0] adaptive range collapsed, using defaults\n");
    return;
  }
  /* NOTE: bin index = (m - mean_lo) / mean_bw computed via direct int32
   * division.  Avoid fxp_recip(mean_bw): for narrow adaptive ranges, mean_bw
   * may be small enough that 1/mean_bw exceeds Q11.20 max=2048 and saturates,
   * sending all blocks to bin 0 (= "1 valid bin" Phase 0 failure).
   * Direct integer division of two int32 raw values has no Q-format issue.
   *
   * fxp_recip(mean_bw) は narrow range で Q11.20 overflow するため整数除算に変更。 */

  /* === Pass 1: per-mean-bin cnt + msum + vmin + vmax === */
  int     cnt_bin[NE_NBINS] = {0};
  fxp_acc msum_bin[NE_NBINS];
  fxp32   vmin_bin[NE_NBINS], vmax_bin[NE_NBINS];
  for(int b = 0; b < NE_NBINS; b++) {
    msum_bin[b] = fxp_acc_zero();
    vmin_bin[b] = FXP_MAX_INT;
    vmax_bin[b] = 0;
  }
  for(int ch = 0; ch < 4; ch++) {
    for(int by = 0; by < n_by; by++) {
      for(int bx = 0; bx < n_bx; bx++) {
        fxp32 m, v;
        if(!compute_block_stats(in_q20, width, height, ch, by, bx,
                                &m, &v, var_scale_combined)) continue;
        if(m <= mean_lo || m >= mean_hi) continue;
        /* bin index = (m - mean_lo) / mean_bw, integer. */
        fxp32 dm = m - mean_lo;
        int b = (int)(dm / mean_bw);
        if(b < 0) b = 0;
        if(b >= NE_NBINS) b = NE_NBINS - 1;
        cnt_bin[b]++;
        fxp_acc_add_i32(&msum_bin[b], m);
        if(v < vmin_bin[b]) vmin_bin[b] = v;
        if(v > vmax_bin[b]) vmax_bin[b] = v;
      }
    }
  }

  /* === Pass 2: per-(mean-bin, var-sub-bin) histogram === */
  /* Allocate vhist[NE_NBINS][NE_VAR_BINS] = 32×128 ints = 16 KB.  Fits LDS. */
  static int vhist[NE_NBINS][NE_VAR_BINS];
  for(int b = 0; b < NE_NBINS; b++)
    for(int s = 0; s < NE_VAR_BINS; s++) vhist[b][s] = 0;

  fxp32 inv_vrange_bin[NE_NBINS];
  for(int b = 0; b < NE_NBINS; b++) {
    if(cnt_bin[b] < 20) { inv_vrange_bin[b] = 0; continue; }
    fxp32 vrange = vmax_bin[b] - vmin_bin[b];
    if(vrange < 1) vrange = 1;
    inv_vrange_bin[b] = fxp_recip(vrange);
  }

  for(int ch = 0; ch < 4; ch++) {
    for(int by = 0; by < n_by; by++) {
      for(int bx = 0; bx < n_bx; bx++) {
        fxp32 m, v;
        if(!compute_block_stats(in_q20, width, height, ch, by, bx,
                                &m, &v, var_scale_combined)) continue;
        if(m <= mean_lo || m >= mean_hi) continue;
        fxp32 dm = m - mean_lo;
        int b = (int)(dm / mean_bw);
        if(b < 0) b = 0;
        if(b >= NE_NBINS) b = NE_NBINS - 1;
        if(cnt_bin[b] < 20 || inv_vrange_bin[b] == 0) continue;
        /* var sub-bin index */
        fxp32 dv = v - vmin_bin[b];
        fxp32 frac = fxp_mul(dv, inv_vrange_bin[b]);
        int vbin = (int)(frac >> 13);
        if(vbin < 0) vbin = 0;
        if(vbin >= NE_VAR_BINS) vbin = NE_VAR_BINS - 1;
        vhist[b][vbin]++;
      }
    }
  }

  /* === Pass 3: per-mean-bin cumsum, lower-envelope mean var === */
  fxp32 bin_mean_arr[NE_NBINS], bin_var_arr[NE_NBINS];
  int   bin_cnt_arr[NE_NBINS],  bin_valid[NE_NBINS];
  int n_valid = 0;
  for(int b = 0; b < NE_NBINS; b++) {
    bin_valid[b] = 0;
    if(cnt_bin[b] < 20) continue;

    int p5_target  = cnt_bin[b] / 20;
    int p20_target = cnt_bin[b] / 5;
    int cum = 0;
    int p5_bin = 0, p20_bin = NE_VAR_BINS - 1;
    int found_p5 = 0;
    for(int i = 0; i < NE_VAR_BINS; i++) {
      cum += vhist[b][i];
      if(!found_p5 && cum >= p5_target) { p5_bin = i; found_p5 = 1; }
      if(cum >= p20_target) { p20_bin = i; break; }
    }
    fxp32 vrange = vmax_bin[b] - vmin_bin[b];
    if(vrange < 1) vrange = 1;
    fxp_acc vsum_acc = fxp_acc_zero();
    int vcnt = 0;
    for(int i = p5_bin; i <= p20_bin; i++) {
      fxp32 i_q = (fxp32)i << FXP_FRAC;
      fxp32 i_plus_half = i_q + (FXP_ONE >> 1);
      fxp32 t = fxp_mul(i_plus_half, vrange) >> 7;     /* /128 */
      fxp32 bin_center = vmin_bin[b] + t;
      int n = vhist[b][i];
      for(int k = 0; k < n; k++) fxp_acc_add_i32(&vsum_acc, bin_center);
      vcnt += n;
    }
    if(vcnt > 0) {
      fxp32 vsum_q = fxp_acc_extract_q20(&vsum_acc);
      bin_var_arr[b] = vsum_q / vcnt;
    } else {
      bin_var_arr[b] = vmin_bin[b];
    }
    bin_mean_arr[b] = fxp_acc_extract_q20(&msum_bin[b]) / cnt_bin[b];
    bin_cnt_arr[b]  = (vcnt > 0) ? vcnt : 1;
    bin_valid[b]    = 1;
    n_valid++;
  }

  if(n_valid < 4) {
    if(g_verbose) fprintf(stderr, "  [phase0] too few valid bins (%d), using defaults\n", n_valid);
    return;
  }

  /* Solve the centered WLS on all valid bins first. */
  fxp32 alpha_est, sigma_sq_est;
  phase0c_wls_solve(bin_mean_arr, bin_var_arr, bin_cnt_arr, bin_valid,
                    NE_NBINS, &alpha_est, &sigma_sq_est);

  /* === Poisson-monotonicity fallback (案B — physical prior, collapse-only) ===
   * For a Poisson-Gaussian sensor the per-mean-bin noise variance is monotone
   * non-decreasing in the mean (var = α·mean + σ²).  Bright FLAT regions (low
   * texture, few blocks) can give an anomalously LOW lower-envelope variance at
   * high mean, breaking that monotonicity and dragging the WLS cross-product
   * Sxy_c negative; no usable positive slope survives, so α is floored to 1 LSB
   * (≈9.5e-7), which makes the GAT inv_2sqrt_alpha = 2/√α overflow Q11.20 (±2048)
   * → all-black collapse (the plomberie__ISO100 pathology, 1/1493).  Detect that
   * collapse by the floored α (alpha_est <= 1 LSB — a healthy fit returns α far
   * above the floor, e.g. 5e-3 ≈ 5243 LSB) and ONLY then re-solve on the monotone-
   * increasing variance prefix: walk bins in increasing-mean order (bin index b is
   * the mean quantile, so already sorted) and once the variance sustains a drop
   * below 3/5·running-max over ≥2 consecutive bins, drop that bin and the whole
   * high-mean tail, then re-solve; adopt the result only if it lifts α off the
   * floor (a2 > alpha_est).  Gating on the floored α means every scene with a
   * valid slope — including the many merely mildly non-monotone from bin-variance
   * noise — is left BIT-IDENTICAL (validated: 1409/1493 RawNIND + 80/80 SIDD
   * unchanged; an earlier unconditional-cut variant broke 3 already-healthy scenes
   * by 17-26 dB, which this floored-α gating fixes).
   * Poisson-Gaussian は var が mean に単調非減少。明るい平坦領域の異常低 var が単調性
   * を破り WLS の Sxy_c を負に倒し、使える正 slope が残らず α が 1 LSB に floor →
   * GAT inv_2sqrt_alpha=2/√α が Q11.20 overflow → 崩壊（plomberie__ISO100）。
   * **floor された α (alpha_est<=1 LSB) を検出した時のみ** 単調 prefix で再解し、
   * α が floor を脱した(a2>alpha_est)時だけ採用。健全シーン（軽微な非単調含む）は
   * bit 一致＝zero regression。*/
  if(alpha_est <= 1) {
    int bv_cut[NE_NBINS];
    for(int b = 0; b < NE_NBINS; b++) bv_cut[b] = bin_valid[b];
    fxp32 run_max = 0;
    int viol = 0, cut_from = -1, nv_cut = n_valid;
    for(int b = 0; b < NE_NBINS; b++) {
      if(!bv_cut[b]) continue;
      fxp32 thr = (fxp32)(((int64_t)run_max * 3) / 5);   /* 3/5·running_max */
      if(bin_var_arr[b] < thr) {
        if(viol == 0) cut_from = b;                       /* first of a run */
        if(++viol >= 2) break;                            /* sustained drop */
      } else {
        viol = 0; cut_from = -1;
        if(bin_var_arr[b] > run_max) run_max = bin_var_arr[b];
      }
    }
    if(viol >= 2 && cut_from >= 0) {
      int dropped = 0;
      for(int b = cut_from; b < NE_NBINS; b++)
        if(bv_cut[b]) { bv_cut[b] = 0; nv_cut--; dropped++; }
      if(nv_cut >= 4) {
        fxp32 a2, s2;
        phase0c_wls_solve(bin_mean_arr, bin_var_arr, bin_cnt_arr, bv_cut,
                          NE_NBINS, &a2, &s2);
        if(a2 > alpha_est) {  /* monotone prefix lifts α off the floor → adopt */
          if(g_verbose) fprintf(stderr, "  [phase0] Poisson-monotonicity fallback: "
            "floored α (collapse); dropped %d non-monotone high-mean bins "
            "(from bin %d, n_valid %d→%d) → α recovered\n",
            dropped, cut_from, n_valid, nv_cut);
          alpha_est = a2; sigma_sq_est = s2;
        }
      }
    }
  }

  phase0d_dark_refine(in_q20, width, height, alpha_est, &sigma_sq_est);

  *alpha_out    = alpha_est;
  *sigma_sq_out = sigma_sq_est;
}

/* Phase 0 stub removed 2026-05-13 — was Stage 1b placeholder before
 * fxp_estimate_noise (= full Foi-Alenius blind α/σ² estimation) was ported
 * in Stage 1c.  fxp_estimate_noise above is now the production path. */

/* ============================================================================
 * Phase 2 — dark_ref IRLS + per-pixel CFA-aware subtract.
 *
 * Mirrors gat_galosh_denoise_rawlc_o() Phase 2 (galosh_raw_cpu.c lines 4685-4793).
 * Streaming-friendly: per-2x2-cell pass, only 5 fxp_acc accumulators in LDS
 * (= ~40 B), no per-image working array.
 *
 * Algorithm:
 *   s_init = σ²/α, s_min = 0.05·s_init, s_max = 50·s_init, s_scale = s_init.
 *   For iter ∈ 0..2 (3 iters):
 *     inv_s = 1/s_scale
 *     For each 2x2 CFA cell at (br, bc):
 *       (g0,g1,g2,g3) = GAT-domain values   (in_gat)
 *       If max(g)-min(g) > 4.0 (= achromatic range): skip
 *       L_raw = mean of 4 raw values
 *       w  = 1 / (1 + (L_raw·inv_s)⁴)        (= 4th-power soft on bright)
 *       sum_w  += w;  sum_wk += w·gk
 *     ch_dark_ref[k] = sum_wk / sum_w
 *     If not last iter: measured_std = √(weighted residual MSE),
 *       s_scale ×= √(1/measured_std), clamped [s_min, s_max].
 *   Subtract ch_dark_ref[slot] from in_gat per pixel.
 *
 * Q-format care:
 *   Bright r > ~7 → r⁴ overflows Q11.20.  Saturate denom to FXP_MAX_INT.
 *   Residuals (g-dark_ref) ~small → 32× scale before squaring (= ×1024 var
 *   scaling, same lesson as Phase 0).  Unscale MSE by >>10.
 * ========================================================================== */

#define FXP_GALOSH_ACHROMATIC_RANGE  fxp_from_float(4.0f)

static void phase2_dark_ref(const fxp32 *in_q20, const fxp32 *in_gat_q20,
                            int width, int height,
                            fxp32 alpha_q20, fxp32 sigma_sq_q20,
                            fxp32 ch_dark_ref[4]) {
  /* s_init = σ²/α.  Use fxp_div_q20 directly: 1/α overflows Q11.20 for
   * α < 4.88e-4 (e.g. SIDD s04 with α=4.77e-4), causing fxp_recip to
   * saturate and the product fxp_mul(σ², 1/α) to drop to 0.  σ²/α itself
   * fits Q11.20, computed via paired-int32 long division. */
  fxp32 s_init = fxp_div_q20(sigma_sq_q20, alpha_q20 < 1 ? 1 : alpha_q20);
  if(s_init < 1) s_init = 1;
  fxp32 s_min = fxp_mul(s_init, fxp_from_float(0.05f));
  fxp32 s_max = fxp_mul(s_init, fxp_from_float(50.0f));
  if(s_min < 1) s_min = 1;
  fxp32 s_scale = s_init;

  ch_dark_ref[0] = ch_dark_ref[1] = ch_dark_ref[2] = ch_dark_ref[3] = 0;

  const int n_iter = 2;       /* matches FP32 reference (3 total iters) */
  for(int iter = 0; iter <= n_iter; iter++) {
    fxp32 inv_s = fxp_recip(s_scale < 1 ? 1 : s_scale);
    fxp_acc sw = fxp_acc_zero();
    fxp_acc sw0 = fxp_acc_zero(), sw1 = fxp_acc_zero();
    fxp_acc sw2 = fxp_acc_zero(), sw3 = fxp_acc_zero();

    for(int br = 0; br < height - 1; br += 2) {
      for(int bc = 0; bc < width - 1; bc += 2) {
        fxp32 g0 = in_gat_q20[(size_t)br     * width + bc    ];
        fxp32 g1 = in_gat_q20[(size_t)(br+1) * width + bc    ];
        fxp32 g2 = in_gat_q20[(size_t)br     * width + bc + 1];
        fxp32 g3 = in_gat_q20[(size_t)(br+1) * width + bc + 1];
        fxp32 gmax = g0; if(g1 > gmax) gmax = g1;
        if(g2 > gmax) gmax = g2; if(g3 > gmax) gmax = g3;
        fxp32 gmin = g0; if(g1 < gmin) gmin = g1;
        if(g2 < gmin) gmin = g2; if(g3 < gmin) gmin = g3;
        if(gmax - gmin > FXP_GALOSH_ACHROMATIC_RANGE) continue;
        fxp32 iv0 = in_q20[(size_t)br     * width + bc    ];
        fxp32 iv1 = in_q20[(size_t)(br+1) * width + bc    ];
        fxp32 iv2 = in_q20[(size_t)br     * width + bc + 1];
        fxp32 iv3 = in_q20[(size_t)(br+1) * width + bc + 1];
        fxp32 L_raw = (iv0 + iv1 + iv2 + iv3) >> 2;
        fxp32 r  = fxp_mul(L_raw, inv_s);
        fxp32 r2 = fxp_mul(r, r);
        fxp32 r4 = fxp_mul(r2, r2);
        fxp32 denom = (r4 < FXP_MAX_INT - FXP_ONE) ? (FXP_ONE + r4) : FXP_MAX_INT;
        fxp32 w = fxp_recip(denom);
        if(w <= 0) continue;
        /* Pre-shift each contribution by >>10 to keep extracted Q11.20 sum
         * within INT32 range across up to 4M cells (4K image).  The same
         * shift applies to numerator AND denominator → ratio unchanged. */
        fxp_acc_add_i32(&sw,  w  >> 10);
        fxp_acc_add_i32(&sw0, fxp_mul(w, g0) >> 10);
        fxp_acc_add_i32(&sw1, fxp_mul(w, g1) >> 10);
        fxp_acc_add_i32(&sw2, fxp_mul(w, g2) >> 10);
        fxp_acc_add_i32(&sw3, fxp_mul(w, g3) >> 10);
      }
    }

    fxp32 sw_q = fxp_acc_extract_q20(&sw);
    if(sw_q < 1) sw_q = 1;
    fxp32 inv_sw = fxp_recip(sw_q);
    ch_dark_ref[0] = fxp_mul(fxp_acc_extract_q20(&sw0), inv_sw);
    ch_dark_ref[1] = fxp_mul(fxp_acc_extract_q20(&sw1), inv_sw);
    ch_dark_ref[2] = fxp_mul(fxp_acc_extract_q20(&sw2), inv_sw);
    ch_dark_ref[3] = fxp_mul(fxp_acc_extract_q20(&sw3), inv_sw);

    if(iter == n_iter) break;

    fxp_acc swW  = fxp_acc_zero();
    fxp_acc swR2 = fxp_acc_zero();
    for(int br = 0; br < height - 1; br += 2) {
      for(int bc = 0; bc < width - 1; bc += 2) {
        fxp32 g0 = in_gat_q20[(size_t)br     * width + bc    ];
        fxp32 g1 = in_gat_q20[(size_t)(br+1) * width + bc    ];
        fxp32 g2 = in_gat_q20[(size_t)br     * width + bc + 1];
        fxp32 g3 = in_gat_q20[(size_t)(br+1) * width + bc + 1];
        fxp32 gmax = g0; if(g1 > gmax) gmax = g1;
        if(g2 > gmax) gmax = g2; if(g3 > gmax) gmax = g3;
        fxp32 gmin = g0; if(g1 < gmin) gmin = g1;
        if(g2 < gmin) gmin = g2; if(g3 < gmin) gmin = g3;
        if(gmax - gmin > FXP_GALOSH_ACHROMATIC_RANGE) continue;
        fxp32 iv0 = in_q20[(size_t)br     * width + bc    ];
        fxp32 iv1 = in_q20[(size_t)(br+1) * width + bc    ];
        fxp32 iv2 = in_q20[(size_t)br     * width + bc + 1];
        fxp32 iv3 = in_q20[(size_t)(br+1) * width + bc + 1];
        fxp32 L_raw = (iv0 + iv1 + iv2 + iv3) >> 2;
        fxp32 r  = fxp_mul(L_raw, inv_s);
        fxp32 r2 = fxp_mul(r, r);
        fxp32 r4 = fxp_mul(r2, r2);
        fxp32 denom = (r4 < FXP_MAX_INT - FXP_ONE) ? (FXP_ONE + r4) : FXP_MAX_INT;
        fxp32 w = fxp_recip(denom);
        if(w <= 0) continue;
        fxp32 d0 = g0 - ch_dark_ref[0];
        fxp32 d1 = g1 - ch_dark_ref[1];
        fxp32 d2 = g2 - ch_dark_ref[2];
        fxp32 d3 = g3 - ch_dark_ref[3];
        /* Residual squaring: d may be small (~1 in real after subtract).
         * d² real ~ 1, fits Q11.20 OK without scaling.  No <<5 pre-scale needed
         * here (bigger residuals than Phase 0 σ²). */
        fxp32 d0_sq = fxp_mul(d0, d0);
        fxp32 d1_sq = fxp_mul(d1, d1);
        fxp32 d2_sq = fxp_mul(d2, d2);
        fxp32 d3_sq = fxp_mul(d3, d3);
        fxp32 r2_sum = d0_sq + d1_sq + d2_sq + d3_sq;
        /* wr2 = w · resid² · 0.25  in Q11.20.  Apply same >>10 pre-shift as
         * iter-loop sw to keep accumulator within INT32 range. */
        fxp32 wr2 = (fxp_mul(w, r2_sum) >> 2) >> 10;
        fxp_acc_add_i32(&swW,  w   >> 10);
        fxp_acc_add_i32(&swR2, wr2);
      }
    }
    fxp32 swW_q  = fxp_acc_extract_q20(&swW);
    fxp32 swR2_q = fxp_acc_extract_q20(&swR2);
    if(swW_q < 1) swW_q = 1;
    /* mse = sum_wresid2 / sum_w (= weighted mean of resid²).  Pre-shift
     * cancelled in ratio. */
    fxp32 mse = fxp_mul(swR2_q, fxp_recip(swW_q));
    if(mse < 1) mse = 1;
    fxp32 measured_std = fxp_sqrt(mse);
    if(measured_std < 1) measured_std = 1;
    fxp32 ratio = fxp_recip(measured_std);
    fxp32 sqrt_ratio = fxp_sqrt(ratio);
    s_scale = fxp_mul(s_scale, sqrt_ratio);
    if(s_scale < s_min) s_scale = s_min;
    if(s_scale > s_max) s_scale = s_max;
  }
}

static void phase2_dark_subtract(fxp32 *in_gat_q20, int width, int height,
                                 const fxp32 ch_dark_ref[4]) {
  for(int r = 0; r < height; r++) {
    int r_off = r & 1;
    for(int c = 0; c < width; c++) {
      int c_off = c & 1;
      int slot = r_off | (c_off << 1);
      in_gat_q20[(size_t)r * width + c] -= ch_dark_ref[slot];
    }
  }
}

/* ============================================================================
 * Phase 1 — GAT forward + per-CFA σ + unified_sigma normalization.
 *
 * Mirrors gat_galosh_denoise_rawlc_o() Phase 1 (galosh_raw_cpu.c lines 4642-4682):
 *   1a. in_gat[i] = GAT_forward(in[i])    per pixel
 *   1b. sigma_gat_ch[s] = estimate_gat_sigma_halfres(in_gat half-res CFA s)
 *       (= 4096-bin |Lap| histogram → median MAD → sigma = MAD/1.6521)
 *   1c. unified_sigma = √(¼ · Σ sigma_gat_ch[s]²)   (RMS across 4 CFA)
 *   1d. in_gat *= 1/unified_sigma                    (normalize to σ=1)
 *
 * Streaming-friendly: 1b uses 4 × 4096-bin histograms in LDS (= 64 KB total
 * if all 4 simultaneously, but processed sequentially in 16 KB at a time).
 * No per-channel tmp half-res buffer.
 *
 * Output: out_q20 = normalized in_gat (= σ_unified = 1 in this domain).
 *         unified_sigma_q20 written out for downstream phases.
 * ========================================================================== */

/* Helper: estimate per-CFA sigma in GAT-normalized space via 4096-bin Lap-MAD
 * histogram.  Streaming version (no tmp half-res buffer). */
static fxp32 fxp_estimate_gat_sigma_halfres_stream(const fxp32 *gat_q20,
                                                   int width, int height,
                                                   int dy0, int dx0) {
  const int halfwidth  = (width  - dx0 + 1) / 2;
  const int halfheight = (height - dy0 + 1) / 2;
  if(halfwidth < 4 || halfheight < 4) return FXP_ONE;

  /* |Lap| range cap = 8.0 in real (covers normalized GAT residuals).
   * 4096 bins, scale = 4096/8 = 512 = 2^9.
   * bin = (lap_q20 × 512) >> FXP_FRAC = lap_q20 >> 11 (bit shift only). */
  static int lap_hist[NE_DARK_HIST_BINS];
  for(int i = 0; i < NE_DARK_HIST_BINS; i++) lap_hist[i] = 0;
  int total = 0;

  /* Stride-2 H-Laplacian along CFA channel. */
  for(int y = 0; y < halfheight; y++) {
    int r = 2*y + dy0;
    if(r >= height) continue;
    for(int x = 0; x < halfwidth - 2; x++) {
      int c0 = 2*x + dx0, c1 = 2*(x+1) + dx0, c2 = 2*(x+2) + dx0;
      if(c2 >= width) break;
      fxp32 v0 = gat_q20[(size_t)r * width + c0];
      fxp32 v1 = gat_q20[(size_t)r * width + c1];
      fxp32 v2 = gat_q20[(size_t)r * width + c2];
      fxp32 lap = v0 - (v1 << 1) + v2;
      if(lap < 0) lap = -lap;
      int b = (int)(lap >> 11);
      if(b < 0) b = 0;
      if(b >= NE_DARK_HIST_BINS) b = NE_DARK_HIST_BINS - 1;
      lap_hist[b]++; total++;
    }
  }
  /* V-Laplacian. */
  for(int y = 0; y < halfheight - 2; y++) {
    int r0 = 2*y + dy0, r1 = 2*(y+1) + dy0, r2 = 2*(y+2) + dy0;
    if(r2 >= height) break;
    for(int x = 0; x < halfwidth; x++) {
      int c = 2*x + dx0;
      if(c >= width) continue;
      fxp32 v0 = gat_q20[(size_t)r0 * width + c];
      fxp32 v1 = gat_q20[(size_t)r1 * width + c];
      fxp32 v2 = gat_q20[(size_t)r2 * width + c];
      fxp32 lap = v0 - (v1 << 1) + v2;
      if(lap < 0) lap = -lap;
      int b = (int)(lap >> 11);
      if(b < 0) b = 0;
      if(b >= NE_DARK_HIST_BINS) b = NE_DARK_HIST_BINS - 1;
      lap_hist[b]++; total++;
    }
  }
  if(total < 100) return FXP_ONE;

  /* Median: 50th percentile of |Lap|. */
  int target = total / 2;
  int cum = 0, med_bin = 0;
  for(int i = 0; i < NE_DARK_HIST_BINS; i++) {
    cum += lap_hist[i];
    if(cum >= target) { med_bin = i; break; }
  }
  /* mad_q = (med_bin + 0.5) << 11 in Q11.20 (since bin step = 1/2^11 in real). */
  fxp32 mad_q = ((fxp32)med_bin << 11) + (1 << 10);
  /* sigma = MAD / 1.6521 = MAD × 0.6053. */
  const fxp32 inv_1p6521 = fxp_from_float(1.0f / 1.6521f);
  fxp32 sigma = fxp_mul(mad_q, inv_1p6521);
  if(sigma < 1) sigma = 1;
  return sigma;
}

/* Phase 1 full pipeline. */
static void phase1_gat_full(const fxp32 *in_q20, fxp32 *gat_q20,
                            int width, int height,
                            const fxp_gat_params *gat_p,
                            fxp32 *unified_sigma_out,
                            fxp32 sigma_gat_ch_out[4]) {
  size_t npixels = (size_t)width * height;
  /* 1a. GAT forward. */
  for(size_t i = 0; i < npixels; i++) {
    gat_q20[i] = fxp_gat_forward(in_q20[i], gat_p);
  }
  /* 1b. Per-CFA σ via streaming half-res Lap-MAD histogram. */
  static const int offsets[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
  for(int s = 0; s < 4; s++) {
    sigma_gat_ch_out[s] = fxp_estimate_gat_sigma_halfres_stream(
      gat_q20, width, height, offsets[s][0], offsets[s][1]);
  }
  /* 1c. unified_sigma = √(¼ Σ σ_s²).
   * σ values typically ∈ [0.5, 2.0] in real, σ² ∈ [0.25, 4.0] — safely
   * representable in Q11.20 without scaling.  No pre-scale needed. */
  fxp_acc sum_sq = fxp_acc_zero();
  for(int s = 0; s < 4; s++) {
    fxp_acc_add_i32(&sum_sq, fxp_mul(sigma_gat_ch_out[s], sigma_gat_ch_out[s]));
  }
  fxp32 sum_sq_q = fxp_acc_extract_q20(&sum_sq);          /* Σσ² in Q11.20 */
  fxp32 mean_var = sum_sq_q >> 2;                          /* /4 → mean σ² */
  if(mean_var < 1) mean_var = 1;
  fxp32 unified_sigma = fxp_sqrt(mean_var);
  if(unified_sigma < 1) unified_sigma = 1;
  *unified_sigma_out = unified_sigma;
  /* 1d. in_gat *= 1/unified_sigma. */
  fxp32 inv_sg = fxp_recip(unified_sigma);
  for(size_t i = 0; i < npixels; i++) {
    gat_q20[i] = fxp_mul(gat_q20[i], inv_sg);
  }
}

/* Legacy single-step GAT forward (= Phase 1a only).  Kept for primitives test. */
static void phase1_gat_forward(const fxp32 *in_q20, fxp32 *out_q20,
                               size_t npixels,
                               const fxp_gat_params *p) {
  for(size_t i = 0; i < npixels; i++) {
    out_q20[i] = fxp_gat_forward(in_q20[i], p);
  }
}

/* ============================================================================
 * Phase 3 — forward 2x2 WHT (stride=1, L-only) with mirror padding.
 *
 * Mirrors gat_h_forward_l_only_stride1() in galosh_cpu.h.  For each pixel
 * (r,c), compute the L coefficient of the 2x2 block whose top-left is at
 * (r,c); use mirror reflection at right/bottom edges.
 *
 *   a  = in[r,    c   ]
 *   b  = in[r+1,  c   ]   (mirror at bottom)
 *   cc = in[r,    c+1 ]   (mirror at right)
 *   d  = in[r+1,  c+1 ]
 *   L[r,c] = (a + b + cc + d) / 2
 *
 * Stateless per-pixel: no LDS, streaming-friendly.  Output L is full-res.
 *
 * Q-format: in values ~10 in real after normalize+subtract, sum/2 ~20.
 * Q11.20 representation = 20M LSBs, fits INT32 (max 2K real = 2.1G LSBs).
 * Sum a+b+cc+d range ±80 → ±84M LSBs raw, fits.  Then >>1 = /2.
 * ========================================================================== */

static inline int fxp_h_mirror_idx(int i, int n) {
  if(i < 0)  return -i;
  if(i >= n) return 2 * n - i - 2;
  return i;
}

static void phase3_forward_l_stride1(const fxp32 *in_gat_q20,
                                     fxp32 *L_cs_q20,
                                     int width, int height) {
  for(int r = 0; r < height; r++) {
    int rb = fxp_h_mirror_idx(r + 1, height);
    const fxp32 *row_a = in_gat_q20 + (size_t)r  * width;
    const fxp32 *row_b = in_gat_q20 + (size_t)rb * width;
    fxp32 *out = L_cs_q20 + (size_t)r * width;
    for(int c = 0; c < width; c++) {
      int cb = fxp_h_mirror_idx(c + 1, width);
      fxp32 a  = row_a[c];
      fxp32 b  = row_b[c];
      fxp32 cc = row_a[cb];
      fxp32 d  = row_b[cb];
      /* (a + b + cc + d) / 2  in Q11.20.  Sum bounded ±84M LSBs (fits int32),
       * then arithmetic shift right by 1.  For negative sums: signed >>1
       * floors toward -∞ which matches FP "/2" for our use case. */
      out[c] = (a + b + cc + d) >> 1;
    }
  }
}

/* ============================================================================
 * Phase 7 — multi-scale LOESS chroma pyramid + K16 joint bilateral upsample.
 *
 * Mirrors Phase 7 of gat_galosh_denoise_rawlc_o (galosh_raw_cpu.c lines
 * 4884-5170+).  Constructs 3 LOESS-denoised chroma estimates at half/quarter/
 * eighth res, each guided by L at the matching scale, then upsamples coarse
 * scales to half-res via K16 jinc-windowed-jinc bilateral upsample with L
 * coupling preserved at every scale transition.
 *
 * Components implemented below:
 *   - fxp_box_downsample_2x:   2x2 avg downsample
 *   - fxp_crop_2d_topleft:     stride conversion crop
 *   - fxp_pad_2d_edge:         edge-replicate pad
 *   - fxp_reflect_idx:         mirror reflection at boundary
 *   - fxp_loess_chroma_3ch_r:  R=7 fused 3-channel Y-guided local linear reg
 *   - fxp_k16_jinc_upsample:   K16 EWA-JL3 bilateral upsample (4 sub-pixel
 *                              orientations, 5x5 jinc window, L-guide bilateral)
 *
 * Q-format care:
 *   - Y² and Y·C products in fxp_acc to avoid Q11.20 overflow
 *   - Bilateral exp() via fxp_exp LUT (clamp arg ≤ 0)
 *   - K16 jinc weights precomputed as Q11.20 constants (= no FP at init)
 * ========================================================================== */

#define FXP_LOESS_RADIUS 7
/* inv_2sigma_sq for BW=3.0: 1/(2·9) = 1/18 ≈ 0.0556 in Q11.20 = 58253. */
#define FXP_LOESS_INV_2SIGMA_SQ  58253
/* eps_gat = strength_c² × GALOSH_LOESS_TAU_SQ_INV (=1.0).
 * For strength_c = 1.0, eps_gat = 1.0 (Q11.20 = FXP_ONE). */

/* Precomputed K16 jinc-windowed-jinc weights (Q11.20).  Generated from
 * jinc(r) × jinc(r/3) for 4 sub-pixel offsets × 5x5 window. */
static const fxp32 fxp_k16_jw[4][5][5] = {
  { /* si=0 sub-offset (0.00,0.00) */
    {       0,       0,       0,       0,       0 },
    {       0,   14475,  -38508,   14475,       0 },
    {       0,  -38508, 1048576,  -38508,       0 },
    {       0,   14475,  -38508,   14475,       0 },
    {       0,       0,       0,       0,       0 },
  },
  { /* si=1 sub-offset (0.00,0.50) */
    {       0,       0,       0,       0,       0 },
    {       0,       0,     376,     376,       0 },
    {       0,       0,  165113,  165113,       0 },
    {       0,       0,     376,     376,       0 },
    {       0,       0,       0,       0,       0 },
  },
  { /* si=2 sub-offset (0.50,0.00) */
    {       0,       0,       0,       0,       0 },
    {       0,       0,       0,       0,       0 },
    {       0,     376,  165113,     376,       0 },
    {       0,     376,  165113,     376,       0 },
    {       0,       0,       0,       0,       0 },
  },
  { /* si=3 sub-offset (0.50,0.50) */
    {       0,       0,       0,       0,       0 },
    {       0,       0,       0,       0,       0 },
    {       0,       0,  -76175,  -76175,       0 },
    {       0,       0,  -76175,  -76175,       0 },
    {       0,       0,       0,       0,       0 },
  },
};

static inline int fxp_reflect_idx(int i, int n) {
  if(i < 0)  return -i;
  if(i >= n) return 2 * n - i - 2;
  return i;
}

static void fxp_box_downsample_2x(const fxp32 *src, fxp32 *dst, int sw, int sh) {
  int dw = sw / 2;
  int dh = sh / 2;
  for(int y = 0; y < dh; y++) {
    int sy = 2 * y;
    const fxp32 *row0 = src + (size_t)sy * sw;
    const fxp32 *row1 = src + (size_t)(sy + 1) * sw;
    fxp32 *drow = dst + (size_t)y * dw;
    for(int x = 0; x < dw; x++) {
      int sx = 2 * x;
      /* (row0[sx] + row0[sx+1] + row1[sx] + row1[sx+1]) / 4 = sum >> 2 */
      drow[x] = (row0[sx] + row0[sx+1] + row1[sx] + row1[sx+1]) >> 2;
    }
  }
}

static void fxp_crop_2d_topleft(const fxp32 *src, int sw, int sh,
                                fxp32 *dst, int dw, int dh) {
  (void)sh;
  for(int y = 0; y < dh; y++) {
    memcpy(dst + (size_t)y * dw, src + (size_t)y * sw, (size_t)dw * sizeof(fxp32));
  }
}

static void fxp_pad_2d_edge(const fxp32 *src, int sw, int sh,
                            fxp32 *dst, int dw, int dh) {
  for(int y = 0; y < dh; y++) {
    int sy = (y < sh) ? y : sh - 1;
    const fxp32 *srow = src + (size_t)sy * sw;
    fxp32 *drow = dst + (size_t)y * dw;
    int copy_w = (dw < sw) ? dw : sw;
    memcpy(drow, srow, (size_t)copy_w * sizeof(fxp32));
    if(dw > sw) {
      fxp32 edge = srow[sw - 1];
      for(int x = sw; x < dw; x++) drow[x] = edge;
    }
  }
}

/* LOESS fused 3-channel Y-guided bilateral local linear regression.
 *
 * Per pixel (y, x):
 *   For each sample in R=7 window:
 *     w = exp(-(Y_i - Y_c)² / (2·σ²))
 *   Accumulate weighted sums (sumW, sumY, sumYY, sumC1/2/3, sumYC1/2/3)
 *   meanY = sumY/sumW, meanYY = sumYY/sumW, etc.
 *   var_Y = max(meanYY − meanY², 0)
 *   denom = var_Y + eps_gat
 *   a_k = (meanYC_k − meanY·meanC_k) / denom
 *   b_k = meanC_k − a_k · meanY
 *   Output C_k = a_k · Y_c + b_k
 *
 * Bilateral exp via fxp_exp LUT (clamp arg ≤ 0, ≥ -16).
 * Sums use fxp_acc to avoid INT32 overflow on YY and YC products.
 */
static void fxp_loess_chroma_3ch_r(const fxp32 *y_guide,
                                   const fxp32 *c1_in,
                                   const fxp32 *c2_in,
                                   const fxp32 *c3_in,
                                   fxp32 *c1_out,
                                   fxp32 *c2_out,
                                   fxp32 *c3_out,
                                   int width, int height,
                                   fxp32 strength_c_q20) {
  const int R = FXP_LOESS_RADIUS;
  /* eps_gat = strength_c² × TAU_SQ_INV (=1.0).  Scale by k²=1/256 for the
   * internal 1/16 input scaling (= prevents sumYY/sumYC overflow on 225-px
   * window).  Output formula: out_scaled = a*Y_c_scaled + b_scaled = k*out.
   * We scale the result back by ×16 (= <<4) at the end. */
  fxp32 eps_gat = fxp_mul(strength_c_q20, strength_c_q20);
  if(eps_gat < 1) eps_gat = 1;
  const fxp32 eps_gat_scaled = eps_gat >> 8;     /* eps × 1/256 (= k²) */
  const fxp32 inv_2sigma_sq = FXP_LOESS_INV_2SIGMA_SQ;

  for(int y = 0; y < height; y++) {
    for(int x = 0; x < width; x++) {
      size_t cx = (size_t)y * width + x;
      /* Internal /16 scaling (k=1/16, k²=1/256).  Used to keep
       * sumYY / sumYC sums within Q11.20 representable range for 225-pixel
       * window.  Bilateral dY is computed in ORIGINAL scale for accurate w. */
      fxp32 Y_c_orig = y_guide[cx];
      fxp32 Y_c     = Y_c_orig >> 4;

      /* Two-pass CENTERED guided filter (fix 2026-06-07).  The original
       * single pass used var_Y = E[Y²]−meanY² and cov = E[Y·C]−meanY·meanC,
       * each a difference of large near-equal weighted window sums.  For
       * large GAT-domain luma the products Yi² / Yi·Ci are ~hundreds and the
       * weighted window sums sumYY / sumYC approach/exceed the Q11.20 ±2048
       * cap → fxp_acc_to_fxp32 SATURATES → meanYY too low → var_Y goes
       * negative (clamped 0) and cov is corrupted, so the slope
       * a = cov/(var+eps) blows up (measured a=1336 vs FP32 0.741) and the
       * chroma output saturates at 2048 (foodstuff2-class catastrophe).
       * Fix: PASS 1 computes the means only (Σw·Y, Σw·C stay small for Y/C in
       * 1/16 scale); PASS 2 re-accumulates var/cov from the SMALL centred
       * deviations (Y−meanY),(C−meanC) — these never approach 2048 and have
       * no catastrophic cancellation — matching the FP32 reference.  Weights
       * are cached from pass 1 so the costly fxp_exp is not recomputed.
       * 大GAT輝度で sumYY/sumYC が ±2048 飽和 → var/cov 崩壊 → a 爆発のバグ修正。
       * 1パス目で平均、2パス目で中央化偏差から var/cov を直接累算（飽和・打消し無）。 */
      fxp32 w_buf[256], Y_buf[256], C1_buf[256], C2_buf[256], C3_buf[256];
      int nb = 0;
      fxp_acc sumW = fxp_acc_zero();
      fxp_acc sumY = fxp_acc_zero();
      fxp_acc sumC1 = fxp_acc_zero(), sumC2 = fxp_acc_zero(), sumC3 = fxp_acc_zero();
      /* GALOSH_CHROMA_CLAMP (canonical, = FP32 galosh_loess_chroma_3ch_r): full-
       * window input chroma range (in the 1/16-scaled space, same as the a·Y+b
       * regression) to clamp the degree-1 extrapolation overshoot. */
      fxp32 cmin1 = 0x7fffffff, cmin2 = 0x7fffffff, cmin3 = 0x7fffffff;
      fxp32 cmax1 = -0x7fffffff, cmax2 = -0x7fffffff, cmax3 = -0x7fffffff;

      for(int dy = -R; dy <= R; dy++) {
        int yi = fxp_reflect_idx(y + dy, height);
        size_t row_off = (size_t)yi * width;
        const fxp32 *rowY  = y_guide + row_off;
        const fxp32 *rowC1 = c1_in   + row_off;
        const fxp32 *rowC2 = c2_in   + row_off;
        const fxp32 *rowC3 = c3_in   + row_off;
        for(int dx = -R; dx <= R; dx++) {
          int xi = fxp_reflect_idx(x + dx, width);
          fxp32 Yi_orig  = rowY [xi];
          fxp32 C1i_orig = rowC1[xi];
          fxp32 C2i_orig = rowC2[xi];
          fxp32 C3i_orig = rowC3[xi];
          /* Bilateral weight from ORIGINAL Y values (= no scale, full precision). */
          fxp32 dY = Yi_orig - Y_c_orig;
          fxp_acc dY_sq_acc = fxp_acc_zero();
          fxp_acc_madd(&dY_sq_acc, dY, dY);
          fxp32 dY_sq = fxp_acc_to_fxp32(&dY_sq_acc);
          fxp32 arg = -fxp_mul(dY_sq, inv_2sigma_sq);
          if(arg < -16 * FXP_ONE) arg = -16 * FXP_ONE;
          if(arg > 0) arg = 0;
          fxp32 w = fxp_exp(arg);

          /* Scale Y, C by 1/16 (= shift right 4) for sums. */
          fxp32 Yi  = Yi_orig  >> 4;
          fxp32 C1i = C1i_orig >> 4;
          fxp32 C2i = C2i_orig >> 4;
          fxp32 C3i = C3i_orig >> 4;

          w_buf[nb] = w; Y_buf[nb] = Yi;
          C1_buf[nb] = C1i; C2_buf[nb] = C2i; C3_buf[nb] = C3i; nb++;
          if(C1i < cmin1) cmin1 = C1i; if(C1i > cmax1) cmax1 = C1i;
          if(C2i < cmin2) cmin2 = C2i; if(C2i > cmax2) cmax2 = C2i;
          if(C3i < cmin3) cmin3 = C3i; if(C3i > cmax3) cmax3 = C3i;
          fxp_acc_add_i32(&sumW, w);
          fxp_acc_madd(&sumY,  w, Yi);
          fxp_acc_madd(&sumC1, w, C1i);
          fxp_acc_madd(&sumC2, w, C2i);
          fxp_acc_madd(&sumC3, w, C3i);
        }
      }

      /* PASS 1 result: means (Σw·Y, Σw·C stay < 2048 for Y/C in 1/16 scale). */
      fxp32 sumW_q = fxp_acc_extract_q20(&sumW);
      if(sumW_q < 1) sumW_q = 1;
      fxp32 invW = fxp_recip(sumW_q);
      fxp32 meanY   = fxp_mul(fxp_acc_to_fxp32(&sumY),  invW);
      fxp32 meanC1  = fxp_mul(fxp_acc_to_fxp32(&sumC1), invW);
      fxp32 meanC2  = fxp_mul(fxp_acc_to_fxp32(&sumC2), invW);
      fxp32 meanC3  = fxp_mul(fxp_acc_to_fxp32(&sumC3), invW);

      /* PASS 2: centred variance/covariance from small deviations.
       * var_Y/cov stay in 1/256 scale (= same as the old E[Y²]−meanY² form);
       * a dimensionless, b in 1/16 scale, output ×16 (<<4). */
      fxp_acc sumVarY = fxp_acc_zero();
      fxp_acc sumCov1 = fxp_acc_zero(), sumCov2 = fxp_acc_zero(), sumCov3 = fxp_acc_zero();
      for(int i = 0; i < nb; i++) {
        fxp32 wi = w_buf[i];
        fxp32 dYc = Y_buf[i] - meanY;
        fxp_acc_madd(&sumVarY, wi, fxp_mul(dYc, dYc));
        fxp_acc_madd(&sumCov1, wi, fxp_mul(dYc, C1_buf[i] - meanC1));
        fxp_acc_madd(&sumCov2, wi, fxp_mul(dYc, C2_buf[i] - meanC2));
        fxp_acc_madd(&sumCov3, wi, fxp_mul(dYc, C3_buf[i] - meanC3));
      }
      fxp32 var_Y = fxp_mul(fxp_acc_to_fxp32(&sumVarY), invW);
      if(var_Y < 0) var_Y = 0;
      fxp32 cov1 = fxp_mul(fxp_acc_to_fxp32(&sumCov1), invW);
      fxp32 cov2 = fxp_mul(fxp_acc_to_fxp32(&sumCov2), invW);
      fxp32 cov3 = fxp_mul(fxp_acc_to_fxp32(&sumCov3), invW);

      fxp32 denom = var_Y + eps_gat_scaled;
      if(denom < 1) denom = 1;
      fxp32 inv_denom = fxp_recip(denom);

      fxp32 a_c1 = fxp_mul(cov1, inv_denom);
      fxp32 a_c2 = fxp_mul(cov2, inv_denom);
      fxp32 a_c3 = fxp_mul(cov3, inv_denom);
      fxp32 b_c1 = meanC1 - fxp_mul(a_c1, meanY);
      fxp32 b_c2 = meanC2 - fxp_mul(a_c2, meanY);
      fxp32 b_c3 = meanC3 - fxp_mul(a_c3, meanY);

      /* output_scaled = a · Y_c_scaled + b (= k * actual_output).
       * Clamp the regression to the local input chroma band (scaled space),
       * then unscale: output = output_scaled × 16 (= << 4). */
      fxp32 os1 = fxp_mul(a_c1, Y_c) + b_c1;
      fxp32 os2 = fxp_mul(a_c2, Y_c) + b_c2;
      fxp32 os3 = fxp_mul(a_c3, Y_c) + b_c3;
      if(cmax1 >= cmin1){ if(os1 < cmin1) os1 = cmin1; else if(os1 > cmax1) os1 = cmax1; }
      if(cmax2 >= cmin2){ if(os2 < cmin2) os2 = cmin2; else if(os2 > cmax2) os2 = cmax2; }
      if(cmax3 >= cmin3){ if(os3 < cmin3) os3 = cmin3; else if(os3 > cmax3) os3 = cmax3; }
      c1_out[cx] = os1 << 4;
      c2_out[cx] = os2 << 4;
      c3_out[cx] = os3 << 4;
    }
  }
}

/* K16 joint bilateral upsample (= 4 sub-pixel orientations × 5x5 jinc kernel
 * with L-bilateral weighting).  Output is 2*halfwidth × 2*halfheight. */
static void fxp_k16_jinc_upsample(const fxp32 *c1_h, const fxp32 *c2_h, const fxp32 *c3_h,
                                  const fxp32 *L_pixel,
                                  fxp32 *c1_full, fxp32 *c2_full, fxp32 *c3_full,
                                  int halfwidth, int halfheight,
                                  fxp32 inv_2sigma_sq_q20) {
  const int fw = 2 * halfwidth;
  const int fh = 2 * halfheight;
  const int W = 2;

  for(int hy = 0; hy < halfheight; hy++) {
    for(int hx = 0; hx < halfwidth; hx++) {
      for(int si = 0; si < 4; si++) {
        int sub_dy = si / 2;
        int sub_dx = si % 2;
        int fr = 2 * hy + sub_dy;
        int fc = 2 * hx + sub_dx;
        if(fr >= fh || fc >= fw) continue;

        fxp32 L_c = L_pixel[(size_t)fr * fw + fc];

        fxp_acc sum_w  = fxp_acc_zero();
        fxp_acc sum_c1 = fxp_acc_zero();
        fxp_acc sum_c2 = fxp_acc_zero();
        fxp_acc sum_c3 = fxp_acc_zero();
        /* GALOSH_CHROMA_CLAMP (canonical, = FP32 gat_k16_joint_bilateral_upsample):
         * full-window input chroma range to clamp the jinc-ringing overshoot. */
        fxp32 cmin1 = 0x7fffffff, cmin2 = 0x7fffffff, cmin3 = 0x7fffffff;
        fxp32 cmax1 = -0x7fffffff, cmax2 = -0x7fffffff, cmax3 = -0x7fffffff;

        for(int dy = -W; dy <= W; dy++) {
          int hyi = hy + dy;
          if(hyi < 0) hyi = 0;
          if(hyi >= halfheight) hyi = halfheight - 1;
          for(int dx = -W; dx <= W; dx++) {
            int hxi = hx + dx;
            if(hxi < 0) hxi = 0;
            if(hxi >= halfwidth) hxi = halfwidth - 1;

            int fri = 2 * hyi;
            int fci = 2 * hxi;
            if(fri >= fh) fri = fh - 1;
            if(fci >= fw) fci = fw - 1;
            fxp32 L_i = L_pixel[(size_t)fri * fw + fci];

            fxp32 dL = L_i - L_c;
            fxp_acc dL_sq_acc = fxp_acc_zero();
            fxp_acc_madd(&dL_sq_acc, dL, dL);
            fxp32 dL_sq = fxp_acc_to_fxp32(&dL_sq_acc);
            fxp32 arg = -fxp_mul(dL_sq, inv_2sigma_sq_q20);
            if(arg < -16 * FXP_ONE) arg = -16 * FXP_ONE;
            if(arg > 0) arg = 0;
            fxp32 w_bilat = fxp_exp(arg);

            fxp32 jw = fxp_k16_jw[si][dy + W][dx + W];
            fxp32 w  = fxp_mul(jw, w_bilat);

            size_t hp = (size_t)hyi * halfwidth + hxi;
            fxp_acc_add_i32(&sum_w, w);
            fxp_acc_madd(&sum_c1, w, c1_h[hp]);
            fxp_acc_madd(&sum_c2, w, c2_h[hp]);
            fxp_acc_madd(&sum_c3, w, c3_h[hp]);
            { fxp32 v1=c1_h[hp], v2=c2_h[hp], v3=c3_h[hp];
              if(v1<cmin1)cmin1=v1; if(v1>cmax1)cmax1=v1;
              if(v2<cmin2)cmin2=v2; if(v2>cmax2)cmax2=v2;
              if(v3<cmin3)cmin3=v3; if(v3>cmax3)cmax3=v3; }
          }
        }

        fxp32 sw   = fxp_acc_extract_q20(&sum_w);
        fxp32 sc1  = fxp_acc_to_fxp32(&sum_c1);
        fxp32 sc2  = fxp_acc_to_fxp32(&sum_c2);
        fxp32 sc3  = fxp_acc_to_fxp32(&sum_c3);
        fxp32 abs_sw = (sw < 0) ? -sw : sw;
        if(abs_sw < 1) sw = (sw < 0) ? -1 : 1;
        /* Bug fix 2026-06-07: `fxp_mul(sc, fxp_recip(sw))` saturated the
         * chroma output to ±2048 whenever |sw| (= Σ jinc·bilateral weights)
         * is tiny — the jinc kernel's negative lobes cancel the centre in
         * high-contrast chroma regions, driving Σw toward 0.  Then 1/sw
         * overflows Q11.20 (true 1/sw reaches ~1e6, confirmed by the int64 HP
         * probe) and fxp_recip saturates at 2048, so c_full = sc·2048 garbage.
         * Fix: compute the ratio sc/sw DIRECTLY via fxp_div_q20, skipping the
         * unrepresentable 1/sw intermediate.  The true ratio (= chroma value)
         * fits Q11.20 (≤ ~40), so no saturation.  Same pattern as the SIDD-s04
         * σ²/α fix documented on fxp_div_q20.
         * jinc 負ローブ相殺で Σ重み≈0 → 1/sw が Q11.20 を溢れ recip 飽和して
         * chroma が 2048 に張り付くバグ。比 sc/sw を fxp_div_q20 で直接計算。 */
        size_t fp = (size_t)fr * fw + fc;
        fxp32 oc1 = fxp_div_q20(sc1, sw);
        fxp32 oc2 = fxp_div_q20(sc2, sw);
        fxp32 oc3 = fxp_div_q20(sc3, sw);
        /* clamp jinc-ringing overshoot to the local input chroma band */
        if(cmax1>=cmin1){ if(oc1<cmin1)oc1=cmin1; else if(oc1>cmax1)oc1=cmax1; }
        if(cmax2>=cmin2){ if(oc2<cmin2)oc2=cmin2; else if(oc2>cmax2)oc2=cmax2; }
        if(cmax3>=cmin3){ if(oc3<cmin3)oc3=cmin3; else if(oc3>cmax3)oc3=cmax3; }
        c1_full[fp] = oc1;
        c2_full[fp] = oc2;
        c3_full[fp] = oc3;
      }
    }
  }
}

/* ============================================================================
 * Phase 6 — L_pixel (2x2 overlap-avg) + L_h_den (subsample).
 *
 * Mirrors Phase 6 in galosh_raw_cpu.c (lines 4856-4877).
 *   L_pixel[fr,fc] = avg(L_cs_den[fr,fc], L_cs_den[fr-1,fc], L_cs_den[fr,fc-1],
 *                        L_cs_den[fr-1,fc-1])  with edge clipping
 *   L_h_den[hr,hc] = L_cs_den[2hr, 2hc]
 * ========================================================================== */
static void phase6_l_pixel(const fxp32 *L_cs_den_q20,
                           fxp32 *L_pixel_q20,
                           int width, int height) {
  for(int fr = 0; fr < height; fr++) {
    for(int fc = 0; fc < width; fc++) {
      size_t p_tl = (size_t)fr * width + fc;
      int32_t sum = L_cs_den_q20[p_tl];
      int count = 1;
      if(fr > 0) {
        sum += L_cs_den_q20[(size_t)(fr - 1) * width + fc];
        count++;
      }
      if(fc > 0) {
        sum += L_cs_den_q20[(size_t)fr * width + (fc - 1)];
        count++;
      }
      if(fr > 0 && fc > 0) {
        sum += L_cs_den_q20[(size_t)(fr - 1) * width + (fc - 1)];
        count++;
      }
      L_pixel_q20[p_tl] = (count == 4) ? (sum >> 2) :
                          (count == 2) ? (sum >> 1) :
                          (count == 1) ? sum :
                          (sum / count);     /* count == 3 case (rare) */
    }
  }
}

static void phase6_l_h_den(const fxp32 *L_cs_den_q20,
                           fxp32 *L_h_den_q20,
                           int width, int height,
                           int halfwidth, int halfheight) {
  for(int hr = 0; hr < halfheight; hr++) {
    int fr = 2 * hr;
    if(fr >= height) continue;
    for(int hc = 0; hc < halfwidth; hc++) {
      int fc = 2 * hc;
      if(fc >= width) continue;
      L_h_den_q20[(size_t)hr * halfwidth + hc] = L_cs_den_q20[(size_t)fr * width + fc];
    }
  }
}

/* ============================================================================
 * Phase 5 Pass1 — BayesShrink hard-threshold pilot (MAD-based σ_Y).
 *
 * Mirrors galosh_pass1_blocked() with use_robust_shrink=1, block=8, stride=2.
 *
 * For each block on stride-2 grid:
 *   1. Extract 8x8 from L_cs into local block[64]
 *   2. Forward WHT (unnormalized)
 *   3. sigma_y_sq = ((mad/0.6745)²) / N    where mad = median |AC coefs|
 *      mad/0.6745 = mad × 1.482 (= sqrt(N) × σ_per_pixel in unnormalized scale)
 *   4. sigma_x_sq = max(sigma_y_sq − sigma_strength², 0)
 *   5. lambda (unnormalized): if sigma_x_sq < 1e-10 → kill (1e30);
 *      else lambda = (sigma_sq / sqrt(sigma_x_sq)) × sqrt(N), capped at
 *        lambda_max × sqrt(N) where lambda_max = sigma_strength × sqrt(2 ln N)
 *   6. Hard threshold: zero AC coefs with |coef| < lambda; DC always kept
 *   7. Inverse WHT (with normalization >>6)
 *   8. Overlap-add to {numer, denom} with Kaiser × (1/n_nonzero) weight
 *
 * Q-format care:
 *   - WHT block coefs (unnormalized): up to ~640 in real for typical noisy
 *     normalized data → Q11.20 = 6.7e8, fits.  Worst case overflow on bright
 *     edge content; saturate gracefully.
 *   - lambda computation: lambda² may be large; use sqrt then divide structure.
 *
 * Output: pilot buffer (denoised L estimate, full-res).  No LDS-streaming
 * concerns at single-thread CPU level (per-pixel numer/denom are
 * "phase boundary memory" per `feedback_int_port_streaming_required.md`).
 * ========================================================================== */

static void phase5_pass1_blocked(const fxp32 *input_q20,
                                 fxp32 *pilot_q20,
                                 int width, int height,
                                 fxp32 sigma_strength_q20) {
  const int B = 8;
  const int N = 64;
  const int stride = 2;
  const int rmax = height - B;
  const int cmax = width  - B;
  const size_t npix = (size_t)width * height;

  /* Pre-compute σ² and lambda_max in unnormalized WHT scale.
   * sigma_strength = 1.0 typical → sigma_sq = 1.0, lambda_max = sqrt(2 ln 64)
   * ≈ 2.884. */
  fxp32 sigma_sq = fxp_mul(sigma_strength_q20, sigma_strength_q20);
  /* lambda_max = sigma_strength × sqrt(2 ln 64).  ln 64 ≈ 4.1589, 2 ln 64 ≈ 8.318,
   * sqrt ≈ 2.884.  Pre-quantize the constant: 2.884 × 2^20 = 3,023,536. */
  fxp32 lambda_max_factor = fxp_from_float(2.8838f);
  fxp32 lambda_max = fxp_mul(sigma_strength_q20, lambda_max_factor);
  /* sqrt(N) = sqrt(64) = 8 (exact integer). */
  const fxp32 sqrtN = 8 * FXP_ONE;

  /* Per-pixel numer/denom accumulators.  fxp_acc (= paired INT32, INT64-
   * equivalent) needed because ~16 overlapping blocks contribute per pixel
   * and bright-coef contributions can sum to >INT32_MAX in Q11.20. */
  fxp_acc *numer = (fxp_acc *)calloc(npix, sizeof(fxp_acc));
  fxp_acc *denom = (fxp_acc *)calloc(npix, sizeof(fxp_acc));
  if(!numer || !denom) {
    free(numer); free(denom);
    memcpy(pilot_q20, input_q20, npix * sizeof(fxp32));
    return;
  }

  for(int ref_r = 0; ref_r <= rmax; ref_r += stride) {
    for(int ref_c = 0; ref_c <= cmax; ref_c += stride) {
      /* Extract 8x8 block. */
      fxp32 block[64];
      for(int dy = 0; dy < B; dy++) {
        memcpy(block + dy * B,
               input_q20 + (size_t)(ref_r + dy) * width + ref_c,
               B * sizeof(fxp32));
      }

      /* Subtract block mean BEFORE forward WHT.
       *
       * Why: unnormalized 8x8 WHT computes DC = sum of 64 inputs.  For
       * Q11.20 inputs with mean > ~32, DC = 64 × mean > 2048 = Q11.20 max,
       * causing INT32 overflow in the final col-WHT stage 3 addition.
       * Observed failure: GAT-domain L_cs mean ~77 for SIDD bright scenes
       * → DC raw ≈ 5e9 > INT32_MAX = 2.15e9 → wraps to garbage.
       *
       * Fix: center block around 0 (subtract mean), WHT centered AC only,
       * threshold AC, inverse WHT, re-add mean.  DC is preserved
       * mathematically (BayesShrink semantics: DC always passes through).
       *
       * Sum via fxp_acc (paired-int32) since direct sum also overflows.
       *
       * 8x8 WHT 前に block 平均を引き、AC 成分のみを WHT で処理。DC は
       * 数学的に保存 (元コードも block[0] を threshold 対象外にしていた)。
       * Q11.20 で input mean>32 の DC overflow 回避。
       *
       * Compute mean as Σ(block[i] >> 6).  Each pre-shifted element ≤ ~1.3M
       * raw for typical L_cs ≤ 80, so the 64-term sum fits INT32 (~80M)
       * and avoids fxp_acc_extract_q20 saturation.  Precision loss from the
       * >> 6 is 6 LSBs ≈ 6e-5 in Q11.20, negligible vs DC magnitude. */
      fxp_acc sum_acc = fxp_acc_zero();
      for(int i = 0; i < N; i++) fxp_acc_add_i32(&sum_acc, block[i] >> 6);
      fxp32 block_mean = fxp_acc_extract_q20(&sum_acc);
      for(int i = 0; i < N; i++) block[i] -= block_mean;

      /* Forward WHT (unnormalized). */
      fxp_wht2d_8x8(block, 0);

      /* σ_y² via MAD: median |AC coefs| / 0.6745, squared, divided by N. */
      fxp32 mad = fxp_partial_selection_median(block + 1, N - 1);
      const fxp32 inv_06745 = fxp_from_float(1.0f / 0.6745f);
      fxp32 mad_scaled = fxp_mul(mad, inv_06745);   /* = sqrt(N) σ in unnorm */
      /* sigma_y_sq = mad_scaled² / N.  N=64=2^6, divide via >>6 after square.
       * To avoid overflow on mad_scaled² (could be ~1e10 raw), use fxp_acc. */
      fxp_acc m2_acc = fxp_acc_zero();
      fxp_acc_madd(&m2_acc, mad_scaled, mad_scaled);
      fxp32 mad_sq = fxp_acc_to_fxp32(&m2_acc);
      fxp32 sigma_y_sq = mad_sq >> 6;          /* /N = /64 */

      /* sigma_x_sq = max(sigma_y_sq − sigma_sq, 0). */
      fxp32 sigma_x_sq = sigma_y_sq - sigma_sq;
      if(sigma_x_sq < 0) sigma_x_sq = 0;

      /* lambda computation.  Threshold matches FP32's 1e-10 effectively:
       * sub-LSB sigma_x_sq in Q11.20 is 1 LSB = ~1e-6, which acts as the
       * "flat block kill all AC" condition. */
      fxp32 lambda;
      if(sigma_x_sq < 1) {          /* sub-LSB → flat, kill all AC */
        lambda = FXP_MAX_INT;
      } else {
        fxp32 sigma_x = fxp_sqrt(sigma_x_sq);
        if(sigma_x < 1) sigma_x = 1;
        /* lambda = (sigma_sq / sigma_x) × sqrt(N) in unnormalized scale.
         * sqrt(N) = 8, so lambda = (sigma_sq / sigma_x) × 8.
         * To avoid overflow with large sigma_sq, compute /sigma_x first. */
        fxp32 inv_sigma_x = fxp_recip(sigma_x);
        fxp32 lam_norm = fxp_mul(sigma_sq, inv_sigma_x);
        lambda = fxp_mul(lam_norm, sqrtN);
        /* Cap at lambda_max × sqrt(N). */
        fxp32 lambda_max_unorm = fxp_mul(lambda_max, sqrtN);
        if(lambda > lambda_max_unorm) lambda = lambda_max_unorm;
      }

      /* Hard threshold (DC = block[0] always preserved). */
      int n_nonzero = 1;
      for(int i = 1; i < N; i++) {
        fxp32 v = block[i];
        fxp32 av = (v < 0) ? -v : v;
        if(av < lambda) block[i] = 0;
        else            n_nonzero++;
      }

      /* Inverse WHT (normalized: divide by N=64 = >>6). */
      fxp_wht2d_8x8(block, 1);

      /* Re-add block mean to restore DC (subtracted before forward WHT). */
      for(int i = 0; i < N; i++) block[i] += block_mean;

      /* Overlap-add with Kaiser × (1/n_nonzero) weight.  Add into 64-bit-
       * equivalent fxp_acc accumulator to avoid INT32 overflow at bright
       * coefs × multiple overlapping blocks. */
      fxp32 weight = fxp_recip((fxp32)n_nonzero * FXP_ONE);
      for(int dy = 0; dy < B; dy++) {
        for(int dx = 0; dx < B; dx++) {
          size_t pos = (size_t)(ref_r + dy) * width + (ref_c + dx);
          fxp32 kw = fxp_kaiser_2d[dy * B + dx];
          fxp32 wkw = fxp_mul(weight, kw);
          fxp_acc_add_i32(&numer[pos], fxp_mul(wkw, block[dy * B + dx]));
          fxp_acc_add_i32(&denom[pos], wkw);
        }
      }
    }
  }

  /* Final per-pixel normalization. */
  for(size_t i = 0; i < npix; i++) {
    fxp32 d = fxp_acc_extract_q20(&denom[i]);
    if(d > 1) {
      fxp32 n = fxp_acc_extract_q20(&numer[i]);
      pilot_q20[i] = fxp_mul(n, fxp_recip(d));
    } else {
      pilot_q20[i] = input_q20[i];
    }
  }
  free(numer); free(denom);
}

/* ============================================================================
 * Phase 5 Pass2 — empirical Wiener using pilot.
 *
 * Mirrors galosh_pass2_blocked() with block=8, stride=2.
 *
 * For each block:
 *   1. Extract noisy + pilot 8x8 blocks
 *   2. Forward WHT both (unnormalized)
 *   3. For each coef: w = 1 (DC) or w = pilot²/(pilot² + sigma²·N), floored
 *   4. noisy[i] *= w; track wiener_energy = Σ w²
 *   5. Inverse WHT noisy
 *   6. Overlap-add with Kaiser × (1/wiener_energy) weight
 * ========================================================================== */
static void phase5_pass2_blocked(const fxp32 *noisy_q20,
                                 const fxp32 *pilot_q20,
                                 fxp32 *output_q20,
                                 int width, int height,
                                 fxp32 sigma_strength_q20,
                                 fxp32 wiener_floor_q20) {
  const int B = 8;
  const int N = 64;
  const int stride = 2;
  const int rmax = height - B;
  const int cmax = width  - B;
  const size_t npix = (size_t)width * height;

  fxp32 sigma_sq_unorm = fxp_mul(sigma_strength_q20, sigma_strength_q20);
  /* sigma² × N (= sigma² × 64) in unnormalized WHT scale. */
  sigma_sq_unorm = sigma_sq_unorm * 64;     /* int32: σ²×64 fits if σ ≤ 5.6 */

  /* Per-pixel fxp_acc accumulators (= INT32 overflow avoidance). */
  fxp_acc *numer = (fxp_acc *)calloc(npix, sizeof(fxp_acc));
  fxp_acc *denom = (fxp_acc *)calloc(npix, sizeof(fxp_acc));
  if(!numer || !denom) {
    free(numer); free(denom);
    memcpy(output_q20, noisy_q20, npix * sizeof(fxp32));
    return;
  }

  for(int ref_r = 0; ref_r <= rmax; ref_r += stride) {
    for(int ref_c = 0; ref_c <= cmax; ref_c += stride) {
      fxp32 blk_noisy[64], blk_pilot[64];
      for(int dy = 0; dy < B; dy++) {
        memcpy(blk_noisy + dy * B,
               noisy_q20 + (size_t)(ref_r + dy) * width + ref_c,
               B * sizeof(fxp32));
        memcpy(blk_pilot + dy * B,
               pilot_q20 + (size_t)(ref_r + dy) * width + ref_c,
               B * sizeof(fxp32));
      }

      /* Subtract per-block means BEFORE forward WHT to avoid Q11.20 DC
       * overflow (DC=64×mean exceeds Q11.20 max=2048 for mean>32).  Use
       * NOISY block's mean for both the noisy and the pilot subtraction
       * so that DC removal is consistent (pilot's DC ≈ noisy's DC since
       * Pass1 preserves DC).  After Wiener weighting w_DC=1 the DC is
       * mathematically unchanged, then we add mean back after inverse WHT.
       *
       * Pass1 と同じく block 平均減算で DC overflow 回避。Pass1 が DC
       * を保存しているので pilot/noisy の DC は一致、共通の mean を使う。
       * Pre-shift each element by >>6 before accumulation (see Pass1 note). */
      fxp_acc sum_n = fxp_acc_zero();
      for(int i = 0; i < N; i++) fxp_acc_add_i32(&sum_n, blk_noisy[i] >> 6);
      fxp32 block_mean = fxp_acc_extract_q20(&sum_n);
      for(int i = 0; i < N; i++) {
        blk_noisy[i] -= block_mean;
        blk_pilot[i] -= block_mean;
      }

      fxp_wht2d_8x8(blk_noisy, 0);
      fxp_wht2d_8x8(blk_pilot, 0);

      /* wiener_energy = Σ w², accumulated in fxp_acc to avoid overflow. */
      fxp_acc we_acc = fxp_acc_zero();
      for(int i = 0; i < N; i++) {
        fxp32 w;
        if(i == 0) {
          w = FXP_ONE;          /* DC fully kept */
        } else {
          /* w = pilot² / (pilot² + σ²N).  pilot² may exceed Q11.20 for
           * bright edge content; use fxp_acc for the squared intermediate.
           * If s2 saturates (= bright pilot), denom_v could overflow int32 →
           * wrap negative.  Detect and short-circuit to w=1 (= pilot dominant). */
          fxp_acc s2_acc = fxp_acc_zero();
          fxp_acc_madd(&s2_acc, blk_pilot[i], blk_pilot[i]);
          fxp32 s2 = fxp_acc_to_fxp32(&s2_acc);
          if(s2 >= FXP_MAX_INT - sigma_sq_unorm) {
            /* s2 saturated or s2 + σ²N would overflow → pilot dominates → w = 1. */
            w = FXP_ONE;
          } else {
            fxp32 denom_v = s2 + sigma_sq_unorm;
            if(denom_v <= 0) denom_v = 1;
            w = fxp_mul(s2, fxp_recip(denom_v));
            if(w < wiener_floor_q20) w = wiener_floor_q20;
            if(w > FXP_ONE) w = FXP_ONE;
          }
        }
        blk_noisy[i] = fxp_mul(blk_noisy[i], w);
        fxp_acc_madd(&we_acc, w, w);
      }
      fxp32 wiener_energy = fxp_acc_to_fxp32(&we_acc);
      if(wiener_energy < 1) wiener_energy = 1;

      fxp_wht2d_8x8(blk_noisy, 1);

      /* Re-add block mean to restore DC (subtracted before forward WHT). */
      for(int i = 0; i < N; i++) blk_noisy[i] += block_mean;

      fxp32 weight = fxp_recip(wiener_energy);
      for(int dy = 0; dy < B; dy++) {
        for(int dx = 0; dx < B; dx++) {
          size_t pos = (size_t)(ref_r + dy) * width + (ref_c + dx);
          fxp32 kw = fxp_kaiser_2d[dy * B + dx];
          fxp32 wkw = fxp_mul(weight, kw);
          fxp_acc_add_i32(&numer[pos], fxp_mul(wkw, blk_noisy[dy * B + dx]));
          fxp_acc_add_i32(&denom[pos], wkw);
        }
      }
    }
  }

  for(size_t i = 0; i < npix; i++) {
    fxp32 d = fxp_acc_extract_q20(&denom[i]);
    if(d > 1) {
      fxp32 n = fxp_acc_extract_q20(&numer[i]);
      output_q20[i] = fxp_mul(n, fxp_recip(d));
    } else {
      output_q20[i] = noisy_q20[i];
    }
  }
  free(numer); free(denom);
}

/* ============================================================================
 * Phase 4 — half-res chroma C1/C2/C3 extraction (stride=2 forward 2x2 WHT).
 *
 * Mirrors gat_j_forward_c_halfres() in galosh_cpu.h.  At each half-res
 * position (hr, hc), reads the 2x2 CFA cell at full-res (2hr..+1, 2hc..+1)
 * and computes:
 *   C1[hr,hc] = (a - b + cc - d) / 2
 *   C2[hr,hc] = (a + b - cc - d) / 2
 *   C3[hr,hc] = (a - b - cc + d) / 2
 * Output: 3 half-res buffers of size (halfwidth × halfheight).
 *
 * Streaming-clean: per-2x2-cell, stateless.  No LDS, no extra buffers.
 * ========================================================================== */
static void phase4_chroma_halfres(const fxp32 *in_gat_q20,
                                  fxp32 *C1_h_q20, fxp32 *C2_h_q20, fxp32 *C3_h_q20,
                                  int width, int height,
                                  int halfwidth, int halfheight) {
  for(int hr = 0; hr < halfheight; hr++) {
    int fr0 = 2 * hr;
    int fr1 = fr0 + 1;
    if(fr1 >= height) continue;
    const fxp32 *row_a = in_gat_q20 + (size_t)fr0 * width;
    const fxp32 *row_b = in_gat_q20 + (size_t)fr1 * width;
    size_t hp_off = (size_t)hr * halfwidth;
    for(int hc = 0; hc < halfwidth; hc++) {
      int fc0 = 2 * hc;
      int fc1 = fc0 + 1;
      if(fc1 >= width) continue;
      fxp32 a  = row_a[fc0];
      fxp32 b  = row_b[fc0];
      fxp32 cc = row_a[fc1];
      fxp32 d  = row_b[fc1];
      C1_h_q20[hp_off + hc] = (a - b + cc - d) >> 1;
      C2_h_q20[hp_off + hc] = (a + b - cc - d) >> 1;
      C3_h_q20[hp_off + hc] = (a - b - cc + d) >> 1;
    }
  }
}

/* ============================================================================
 * Smoke-test main — read FP32 raw, convert to Q11.20, GAT forward, convert
 * back to FP32, write.
 *
 * For Phase 1 verification (= no denoising).  Real denoising will be added
 * in Stage 1c.  The output image is meaningful only for smoke checking that
 * GAT_int matches GAT_fp32 within precision.
 * ========================================================================== */
#ifdef GALOSH_OPCOUNT
long long g_n_mac = 0, g_n_sf = 0;   /* op-count instrumentation (see galosh_cpu_int.h) */
#endif
int main(int argc, char **argv) {
  g_verbose = (getenv("GALOSH_VERBOSE") != NULL);
  /* Strip --variant flag from argv. */
  int new_argc = 0;
  char *positional[32];
  for(int i = 0; i < argc; i++) {
    const char *a = argv[i];
    if(strncmp(a, "--variant=", 10) == 0) {
      /* accepted and ignored — this build is the canonical INT32 reference */
    } else if(new_argc < 32) {
      positional[new_argc++] = (char *)a;
    }
  }
  argv = positional;
  argc = new_argc;

  if(argc < 5) {
    fprintf(stderr,
      "Usage: %s input.bin output.bin width height\n"
      "       [method=galosh] [strength=1.0] [luma_str=0.5] [chroma_str=1.0]\n"
      "       [alpha=0] [sigma_sq=0]   (positive = skip Phase 0 estimation)\n"
      "       (raw float32 Bayer in [0,1]; alpha=sigma=0 -> blind estimation)\n",
      argv[0]);
    return 1;
  }
  const char *input_file  = argv[1];
  const char *output_file = argv[2];
  const int   width  = atoi(argv[3]);
  const int   height = atoi(argv[4]);
  /* method ignored for now (= always galosh). */
  const float luma_str    = (argc > 7)  ? (float)atof(argv[7])  : 1.0f;
  /* chroma_str unused until Phase 7-8.  Reserved. */
  const float chroma_str  = (argc > 8)  ? (float)atof(argv[8])  : 1.0f;
  const float alpha_in    = (argc > 9)  ? (float)atof(argv[9])  : 0.0f;
  const float sigma_sq_in = (argc > 10) ? (float)atof(argv[10]) : 0.0f;
  (void)chroma_str;

  if(width <= 0 || height <= 0) {
    fprintf(stderr, "Invalid dimensions: %dx%d\n", width, height);
    return 1;
  }

  if(g_verbose) fprintf(stderr, "GALOSH-RAW INT (Q11.20 fixed-point, blind)\n");
  if(g_verbose) fprintf(stderr, "  Input:  %s (%dx%d)\n", input_file, width, height);
  if(g_verbose) fprintf(stderr, "  Q-format: Q11.20 (range +-2048, precision %.3e)\n",
          1.0f / (float)FXP_ONE);

  /* Initialize transcendental LUTs + Kaiser window. */
  fxp_exp_lut_init();
  fxp_log_lut_init();
  fxp_kaiser_init();

  const size_t npixels = (size_t)width * height;

  /* Allocate FP32 I/O buffers + Q11.20 working buffers. */
  float *in_f32  = (float *)malloc(npixels * sizeof(float));
  float *out_f32 = (float *)malloc(npixels * sizeof(float));
  fxp32 *in_q20  = (fxp32 *)malloc(npixels * sizeof(fxp32));
  fxp32 *out_q20 = (fxp32 *)malloc(npixels * sizeof(fxp32));
  if(!in_f32 || !out_f32 || !in_q20 || !out_q20) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  /* Read input FP32 .bin. */
  FILE *fin = fopen(input_file, "rb");
  if(!fin) { fprintf(stderr, "Cannot open %s\n", input_file); return 1; }
  size_t nread = fread(in_f32, sizeof(float), npixels, fin);
  fclose(fin);
  if(nread != npixels) {
    fprintf(stderr, "Read %zu floats, expected %zu\n", nread, npixels);
    return 1;
  }

  /* Convert FP32 → Q11.20 (saturating). */
  for(size_t i = 0; i < npixels; i++) {
    in_q20[i] = fxp_from_float(in_f32[i]);
  }

  /* ========== Phase 0: blind α/σ² noise estimation (real impl) ========== */
  fxp32 alpha_q20, sigma_sq_q20;
  if(alpha_in > 0.0f && sigma_sq_in >= 0.0f) {
    /* Override via CLI (= bench harness pass-through). */
    alpha_q20    = fxp_from_float(alpha_in);
    sigma_sq_q20 = fxp_from_float(sigma_sq_in);
    if(g_verbose) fprintf(stderr, "  Phase 0 (CLI override): alpha=%.6f sigma_sq=%.10f\n",
            fxp_to_float(alpha_q20), fxp_to_float(sigma_sq_q20));
  } else {
    clock_t tp0 = clock();
    fxp_estimate_noise(in_q20, width, height, &alpha_q20, &sigma_sq_q20);
    double dt0 = (double)(clock() - tp0) / CLOCKS_PER_SEC;
    if(g_verbose) fprintf(stderr, "  Phase 0 (INT estimate, %.3f s): alpha=%.6f sigma_sq=%.10f "
            "(Q11.20: %d, %d)\n",
            dt0, fxp_to_float(alpha_q20), fxp_to_float(sigma_sq_q20),
            alpha_q20, sigma_sq_q20);
  }

  /* ========== Phase 1: GAT forward + per-CFA σ + unified_sigma normalize ========== */
  fxp_gat_params gat_p;
  fxp_gat_precompute(&gat_p, alpha_q20, sigma_sq_q20);
  if(g_verbose) fprintf(stderr, "  GAT params: inv_2sqrt_alpha=%.4f, 3a/8=%.6f, s²/a=%.6f, y_break=%.6f\n",
          fxp_to_float(gat_p.inv_2sqrt_alpha),
          fxp_to_float(gat_p.three_alpha_8),
          fxp_to_float(gat_p.sigma_sq_over_alpha),
          fxp_to_float(gat_p.y_break));
  /* Build Foi-exact inverse GAT LUT for Phase 10.  Replaces algebraic inverse
   * for low-signal patches (s24/s06) where algebraic returns negative output.
   * Build cost ~0.5 s/image. */
  clock_t t_lut0 = clock();
  fxp_gat_build_inverse_table(&gat_p);
  if(g_verbose) fprintf(stderr, "  Foi LUT build time: %.3f s\n",
          (double)(clock() - t_lut0) / CLOCKS_PER_SEC);
  fxp32 unified_sigma_q, sigma_gat_ch[4];
  clock_t t0 = clock();
  phase1_gat_full(in_q20, out_q20, width, height, &gat_p,
                  &unified_sigma_q, sigma_gat_ch);
  double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
  if(g_verbose) fprintf(stderr, "  Phase 1 (%.3f s): unified_sigma=%.4f (per-ch: %.4f %.4f %.4f %.4f)\n",
          elapsed, fxp_to_float(unified_sigma_q),
          fxp_to_float(sigma_gat_ch[0]), fxp_to_float(sigma_gat_ch[1]),
          fxp_to_float(sigma_gat_ch[2]), fxp_to_float(sigma_gat_ch[3]));
  /* Raw Q11.20 dump for bit-exact GPU (i16) per-phase validation.  Guarded by
   * a SEPARATE env var (GALOSH_INT_RAW_DUMP_DIR) from the FP32-divergence float
   * dumps (GALOSH_INT_DUMP_DIR) because GAT-domain values exceed 2^24 and lose
   * bits through fxp_to_float — raw int32 is required for a bit-exact compare. */
  {
    const char *raw_dir = getenv("GALOSH_INT_RAW_DUMP_DIR");
    if(raw_dir) {
      char path[1024];
      snprintf(path, sizeof(path), "%s/p1_ingat.bin", raw_dir);
      FILE *df = fopen(path, "wb");
      if(df) { fwrite(out_q20, sizeof(fxp32), npixels, df); fclose(df); }
      if(g_verbose) fprintf(stderr, "  P1_RAW unified_sigma=%d sigma_ch=%d,%d,%d,%d\n",
              unified_sigma_q, sigma_gat_ch[0], sigma_gat_ch[1],
              sigma_gat_ch[2], sigma_gat_ch[3]);
    }
  }

  /* ========== Phase 2: dark_ref IRLS + per-pixel CFA-aware subtract ========== */
  fxp32 ch_dark_ref[4];
  clock_t tp2 = clock();
  phase2_dark_ref(in_q20, out_q20, width, height,
                  alpha_q20, sigma_sq_q20, ch_dark_ref);
  phase2_dark_subtract(out_q20, width, height, ch_dark_ref);
  double dt2 = (double)(clock() - tp2) / CLOCKS_PER_SEC;
  if(g_verbose) fprintf(stderr, "  Phase 2 dark_ref (%.3f s): [0]=%.4f [1]=%.4f [2]=%.4f [3]=%.4f\n",
          dt2,
          fxp_to_float(ch_dark_ref[0]), fxp_to_float(ch_dark_ref[1]),
          fxp_to_float(ch_dark_ref[2]), fxp_to_float(ch_dark_ref[3]));
  {
    const char *raw_dir = getenv("GALOSH_INT_RAW_DUMP_DIR");
    if(raw_dir) {
      char path[1024];
      snprintf(path, sizeof(path), "%s/p2_ingat.bin", raw_dir);
      FILE *df = fopen(path, "wb");
      if(df) { fwrite(out_q20, sizeof(fxp32), npixels, df); fclose(df); }
      if(g_verbose) fprintf(stderr, "  P2_RAW ch_dark_ref=%d,%d,%d,%d\n",
              ch_dark_ref[0], ch_dark_ref[1], ch_dark_ref[2], ch_dark_ref[3]);
    }
  }

  /* ========== Phase 3: forward L stride=1 cycle-spinning WHT ========== */
  fxp32 *L_cs_q20 = (fxp32 *)malloc(npixels * sizeof(fxp32));
  if(!L_cs_q20) { fprintf(stderr, "alloc L_cs failed\n"); return 1; }
  clock_t tp3 = clock();
  phase3_forward_l_stride1(out_q20, L_cs_q20, width, height);
  double dt3 = (double)(clock() - tp3) / CLOCKS_PER_SEC;
  fxp32 lo = FXP_MAX_INT, hi = FXP_MIN_INT;
  for(size_t i = 0; i < npixels; i++) {
    if(L_cs_q20[i] < lo) lo = L_cs_q20[i];
    if(L_cs_q20[i] > hi) hi = L_cs_q20[i];
  }
  if(g_verbose) fprintf(stderr, "  Phase 3 forward L stride=1 (%.3f s): range [%.4f, %.4f]\n",
          dt3, fxp_to_float(lo), fxp_to_float(hi));
  /* Optional: dump p3_L_cs as FP32 .bin for comparison vs FP32 ref. */
  const char *dump_dir = getenv("GALOSH_INT_DUMP_DIR");
  if(dump_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/p3_L_cs.bin", dump_dir);
    FILE *df = fopen(path, "wb");
    if(df) {
      for(size_t i = 0; i < npixels; i++) {
        float v = fxp_to_float(L_cs_q20[i]);
        fwrite(&v, sizeof(float), 1, df);
      }
      fclose(df);
    }
  }
  {
    const char *raw_dir = getenv("GALOSH_INT_RAW_DUMP_DIR");
    if(raw_dir) {
      char path[1024];
      snprintf(path, sizeof(path), "%s/p3_lcs.bin", raw_dir);
      FILE *df = fopen(path, "wb");
      if(df) { fwrite(L_cs_q20, sizeof(fxp32), npixels, df); fclose(df); }
      if(g_verbose) fprintf(stderr, "  P3_RAW lo=%d hi=%d\n", lo, hi);
    }
  }

  /* ========== Phase 4: half-res chroma extraction ========== */
  const int halfwidth  = (width  + 1) / 2;
  const int halfheight = (height + 1) / 2;
  const size_t chsize = (size_t)halfwidth * halfheight;
  fxp32 *C1_h_q20 = (fxp32 *)malloc(chsize * sizeof(fxp32));
  fxp32 *C2_h_q20 = (fxp32 *)malloc(chsize * sizeof(fxp32));
  fxp32 *C3_h_q20 = (fxp32 *)malloc(chsize * sizeof(fxp32));
  if(!C1_h_q20 || !C2_h_q20 || !C3_h_q20) {
    fprintf(stderr, "alloc C1/2/3_h failed\n"); return 1;
  }
  clock_t tp4 = clock();
  phase4_chroma_halfres(out_q20, C1_h_q20, C2_h_q20, C3_h_q20,
                        width, height, halfwidth, halfheight);
  double dt4 = (double)(clock() - tp4) / CLOCKS_PER_SEC;
  if(g_verbose) fprintf(stderr, "  Phase 4 chroma extract (%.3f s): C1/2/3 half-res %dx%d\n",
          dt4, halfwidth, halfheight);
  if(dump_dir) {
    const char *names[3] = {"p4_C1_h", "p4_C2_h", "p4_C3_h"};
    fxp32 *bufs[3] = { C1_h_q20, C2_h_q20, C3_h_q20 };
    for(int k = 0; k < 3; k++) {
      char path[1024];
      snprintf(path, sizeof(path), "%s/%s.bin", dump_dir, names[k]);
      FILE *df = fopen(path, "wb");
      if(df) {
        for(size_t i = 0; i < chsize; i++) {
          float v = fxp_to_float(bufs[k][i]);
          fwrite(&v, sizeof(float), 1, df);
        }
        fclose(df);
      }
    }
  }
  {
    const char *raw_dir = getenv("GALOSH_INT_RAW_DUMP_DIR");
    if(raw_dir) {
      char path[1024];
      snprintf(path, sizeof(path), "%s/p4_chroma.bin", raw_dir);
      FILE *df = fopen(path, "wb");
      if(df) {
        fwrite(C1_h_q20, sizeof(fxp32), chsize, df);
        fwrite(C2_h_q20, sizeof(fxp32), chsize, df);
        fwrite(C3_h_q20, sizeof(fxp32), chsize, df);
        fclose(df);
      }
      if(g_verbose) fprintf(stderr, "  P4_RAW n=%d\n", (int)chsize);
    }
  }

  /* ========== Phase 5: BayesShrink Pass1 (pilot) + Pass2 (Wiener) ========== */
  fxp32 *L_cs_pilot_q20 = (fxp32 *)malloc(npixels * sizeof(fxp32));
  fxp32 *L_cs_den_q20   = (fxp32 *)malloc(npixels * sizeof(fxp32));
  if(!L_cs_pilot_q20 || !L_cs_den_q20) { fprintf(stderr, "alloc Phase 5 buffers failed\n"); return 1; }
  fxp32 luma_str_q20 = fxp_from_float(luma_str);
  fxp32 wiener_floor_q20 = FXP_ONE / 8;     /* 1/B = 1/8 */
  clock_t tp5 = clock();
  phase5_pass1_blocked(L_cs_q20, L_cs_pilot_q20, width, height, luma_str_q20);
  if(dump_dir) {
    char path[1024]; snprintf(path, sizeof(path), "%s/p5_pilot.bin", dump_dir);
    FILE *df = fopen(path, "wb");
    if(df) {
      for(size_t i = 0; i < npixels; i++) {
        float v = fxp_to_float(L_cs_pilot_q20[i]);
        fwrite(&v, sizeof(float), 1, df);
      }
      fclose(df);
    }
  }
  {
    const char *raw_dir = getenv("GALOSH_INT_RAW_DUMP_DIR");
    if(raw_dir) {
      char path[1024]; snprintf(path, sizeof(path), "%s/p5_pilot.bin", raw_dir);
      FILE *df = fopen(path, "wb");
      if(df) { fwrite(L_cs_pilot_q20, sizeof(fxp32), npixels, df); fclose(df); }
      fxp32 plo = FXP_MAX_INT, phi = FXP_MIN_INT;
      for(size_t i = 0; i < npixels; i++) {
        if(L_cs_pilot_q20[i] < plo) plo = L_cs_pilot_q20[i];
        if(L_cs_pilot_q20[i] > phi) phi = L_cs_pilot_q20[i];
      }
      if(g_verbose) fprintf(stderr, "  P5PILOT_RAW lo=%d hi=%d\n", plo, phi);
    }
  }
  phase5_pass2_blocked(L_cs_q20, L_cs_pilot_q20, L_cs_den_q20,
                       width, height, luma_str_q20, wiener_floor_q20);
  double dt5 = (double)(clock() - tp5) / CLOCKS_PER_SEC;
  fxp32 d_lo = FXP_MAX_INT, d_hi = FXP_MIN_INT;
  for(size_t i = 0; i < npixels; i++) {
    if(L_cs_den_q20[i] < d_lo) d_lo = L_cs_den_q20[i];
    if(L_cs_den_q20[i] > d_hi) d_hi = L_cs_den_q20[i];
  }
  if(g_verbose) fprintf(stderr, "  Phase 5 BayesShrink P1+P2 (%.3f s): L_cs_den range [%.4f, %.4f]\n",
          dt5, fxp_to_float(d_lo), fxp_to_float(d_hi));
  if(dump_dir) {
    char path[1024]; snprintf(path, sizeof(path), "%s/p5_L_cs_den.bin", dump_dir);
    FILE *df = fopen(path, "wb");
    if(df) {
      for(size_t i = 0; i < npixels; i++) {
        float v = fxp_to_float(L_cs_den_q20[i]);
        fwrite(&v, sizeof(float), 1, df);
      }
      fclose(df);
    }
  }
  {
    const char *raw_dir = getenv("GALOSH_INT_RAW_DUMP_DIR");
    if(raw_dir) {
      char path[1024]; snprintf(path, sizeof(path), "%s/p5_den.bin", raw_dir);
      FILE *df = fopen(path, "wb");
      if(df) { fwrite(L_cs_den_q20, sizeof(fxp32), npixels, df); fclose(df); }
      if(g_verbose) fprintf(stderr, "  P5_RAW lo=%d hi=%d\n", d_lo, d_hi);
    }
  }

  /* ========== Phase 7: LOESS chroma pyramid (3-scale) + K16 upsample ========== */
  /* Skip Phase 6 here — Phase 7 only needs C1/2/3_h (already available from
   * Phase 4) and L_h_den (will be available after Phase 6). */

  /* ========== Phase 6: L_pixel (2x2 overlap-avg) + L_h_den (subsample) ========== */
  fxp32 *L_pixel_q20 = (fxp32 *)malloc(npixels * sizeof(fxp32));
  fxp32 *L_h_den_q20 = (fxp32 *)malloc(chsize  * sizeof(fxp32));
  if(!L_pixel_q20 || !L_h_den_q20) { fprintf(stderr, "alloc Phase 6 failed\n"); return 1; }
  clock_t tp6 = clock();
  phase6_l_pixel(L_cs_den_q20, L_pixel_q20, width, height);
  phase6_l_h_den(L_cs_den_q20, L_h_den_q20, width, height, halfwidth, halfheight);
  double dt6 = (double)(clock() - tp6) / CLOCKS_PER_SEC;
  if(g_verbose) fprintf(stderr, "  Phase 6 L_pixel + L_h_den (%.3f s)\n", dt6);
  if(dump_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/p6_L_pixel.bin", dump_dir);
    FILE *df = fopen(path, "wb");
    if(df) {
      for(size_t i = 0; i < npixels; i++) {
        float v = fxp_to_float(L_pixel_q20[i]);
        fwrite(&v, sizeof(float), 1, df);
      }
      fclose(df);
    }
    snprintf(path, sizeof(path), "%s/p6_L_h_den.bin", dump_dir);
    df = fopen(path, "wb");
    if(df) {
      for(size_t i = 0; i < chsize; i++) {
        float v = fxp_to_float(L_h_den_q20[i]);
        fwrite(&v, sizeof(float), 1, df);
      }
      fclose(df);
    }
  }
  {
    const char *raw_dir = getenv("GALOSH_INT_RAW_DUMP_DIR");
    if(raw_dir) {
      char path[1024]; snprintf(path, sizeof(path), "%s/p6.bin", raw_dir);
      FILE *df = fopen(path, "wb");
      if(df) {
        fwrite(L_pixel_q20, sizeof(fxp32), npixels, df);
        fwrite(L_h_den_q20, sizeof(fxp32), chsize, df);
        fclose(df);
      }
      if(g_verbose) fprintf(stderr, "  P6_RAW npix=%d chsize=%d\n", (int)npixels, (int)chsize);
    }
  }

  /* ========== Phase 7 actual: LOESS chroma pyramid + K16 upsample ========== */
  const int cq_w = halfwidth / 2;
  const int cq_h = halfheight / 2;
  const int ce_w = cq_w / 2;
  const int ce_h = cq_h / 2;
  const size_t cqsize = (size_t)cq_w * cq_h;
  const size_t cesize = (size_t)ce_w * ce_h;

  fxp32 *L_q_q20 = (fxp32 *)malloc(cqsize * sizeof(fxp32));
  fxp32 *L_e_q20 = (fxp32 *)malloc(cesize * sizeof(fxp32));
  fxp32 *C1_q = (fxp32 *)malloc(cqsize * sizeof(fxp32));
  fxp32 *C2_q = (fxp32 *)malloc(cqsize * sizeof(fxp32));
  fxp32 *C3_q = (fxp32 *)malloc(cqsize * sizeof(fxp32));
  fxp32 *C1_e = (fxp32 *)malloc(cesize * sizeof(fxp32));
  fxp32 *C2_e = (fxp32 *)malloc(cesize * sizeof(fxp32));
  fxp32 *C3_e = (fxp32 *)malloc(cesize * sizeof(fxp32));
  fxp32 *C1_loess_h = (fxp32 *)malloc(chsize * sizeof(fxp32));
  fxp32 *C2_loess_h = (fxp32 *)malloc(chsize * sizeof(fxp32));
  fxp32 *C3_loess_h = (fxp32 *)malloc(chsize * sizeof(fxp32));
  fxp32 *C1_loess_q = (fxp32 *)malloc(cqsize * sizeof(fxp32));
  fxp32 *C2_loess_q = (fxp32 *)malloc(cqsize * sizeof(fxp32));
  fxp32 *C3_loess_q = (fxp32 *)malloc(cqsize * sizeof(fxp32));
  fxp32 *C1_loess_e = (fxp32 *)malloc(cesize * sizeof(fxp32));
  fxp32 *C2_loess_e = (fxp32 *)malloc(cesize * sizeof(fxp32));
  fxp32 *C3_loess_e = (fxp32 *)malloc(cesize * sizeof(fxp32));
  fxp32 *C1_q_up = (fxp32 *)malloc(chsize * sizeof(fxp32));
  fxp32 *C2_q_up = (fxp32 *)malloc(chsize * sizeof(fxp32));
  fxp32 *C3_q_up = (fxp32 *)malloc(chsize * sizeof(fxp32));
  fxp32 *C1_e_up = (fxp32 *)malloc(chsize * sizeof(fxp32));
  fxp32 *C2_e_up = (fxp32 *)malloc(chsize * sizeof(fxp32));
  fxp32 *C3_e_up = (fxp32 *)malloc(chsize * sizeof(fxp32));

  clock_t tp7 = clock();
  /* Build L pyramid. */
  fxp_box_downsample_2x(L_h_den_q20, L_q_q20, halfwidth, halfheight);
  fxp_box_downsample_2x(L_q_q20,     L_e_q20, cq_w, cq_h);
  /* Build C pyramid. */
  fxp_box_downsample_2x(C1_h_q20, C1_q, halfwidth, halfheight);
  fxp_box_downsample_2x(C2_h_q20, C2_q, halfwidth, halfheight);
  fxp_box_downsample_2x(C3_h_q20, C3_q, halfwidth, halfheight);
  fxp_box_downsample_2x(C1_q, C1_e, cq_w, cq_h);
  fxp_box_downsample_2x(C2_q, C2_e, cq_w, cq_h);
  fxp_box_downsample_2x(C3_q, C3_e, cq_w, cq_h);
  /* LOESS at each scale (strength_c = 1.0 fixed per FP32 ref). */
  fxp32 loess_strength = FXP_ONE;
  fxp_loess_chroma_3ch_r(L_h_den_q20, C1_h_q20, C2_h_q20, C3_h_q20,
                         C1_loess_h, C2_loess_h, C3_loess_h,
                         halfwidth, halfheight, loess_strength);
  fxp_loess_chroma_3ch_r(L_q_q20, C1_q, C2_q, C3_q,
                         C1_loess_q, C2_loess_q, C3_loess_q,
                         cq_w, cq_h, loess_strength);
  fxp_loess_chroma_3ch_r(L_e_q20, C1_e, C2_e, C3_e,
                         C1_loess_e, C2_loess_e, C3_loess_e,
                         ce_w, ce_h, loess_strength);

  /* K16 upsample: quarter→half (L_h_den guide).  BW=1.5 → inv_2sigma_sq=2/9≈0.222 */
  fxp32 inv_2sigma_sq_k16 = fxp_from_float(2.0f/9.0f);
  fxp_k16_jinc_upsample(C1_loess_q, C2_loess_q, C3_loess_q, L_h_den_q20,
                        C1_q_up, C2_q_up, C3_q_up,
                        cq_w, cq_h, inv_2sigma_sq_k16);

  /* eighth→quarter (L_q guide) chained → upsample result to quarter, then K16 to half. */
  fxp32 *C1_e_to_q = (fxp32 *)malloc(cqsize * sizeof(fxp32));
  fxp32 *C2_e_to_q = (fxp32 *)malloc(cqsize * sizeof(fxp32));
  fxp32 *C3_e_to_q = (fxp32 *)malloc(cqsize * sizeof(fxp32));
  fxp_k16_jinc_upsample(C1_loess_e, C2_loess_e, C3_loess_e, L_q_q20,
                        C1_e_to_q, C2_e_to_q, C3_e_to_q,
                        ce_w, ce_h, inv_2sigma_sq_k16);
  /* Final eighth→half (L_h_den guide). */
  fxp_k16_jinc_upsample(C1_e_to_q, C2_e_to_q, C3_e_to_q, L_h_den_q20,
                        C1_e_up, C2_e_up, C3_e_up,
                        cq_w, cq_h, inv_2sigma_sq_k16);
  free(C1_e_to_q); free(C2_e_to_q); free(C3_e_to_q);

  double dt7 = (double)(clock() - tp7) / CLOCKS_PER_SEC;
  if(g_verbose) fprintf(stderr, "  Phase 7 LOESS pyramid (%.3f s): half=%dx%d quarter=%dx%d eighth=%dx%d\n",
          dt7, halfwidth, halfheight, cq_w, cq_h, ce_w, ce_h);
  if(dump_dir) {
    const char *names[3] = {"p7_C1_loess_h", "p7_C2_loess_h", "p7_C3_loess_h"};
    fxp32 *bufs[3] = { C1_loess_h, C2_loess_h, C3_loess_h };
    for(int k = 0; k < 3; k++) {
      char path[1024];
      snprintf(path, sizeof(path), "%s/%s.bin", dump_dir, names[k]);
      FILE *df = fopen(path, "wb");
      if(df) {
        for(size_t i = 0; i < chsize; i++) {
          float v = fxp_to_float(bufs[k][i]);
          fwrite(&v, sizeof(float), 1, df);
        }
        fclose(df);
      }
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/p7_C1_q_up.bin", dump_dir);
    FILE *df = fopen(path, "wb");
    if(df) {
      for(size_t i = 0; i < chsize; i++) {
        float v = fxp_to_float(C1_q_up[i]);
        fwrite(&v, sizeof(float), 1, df);
      }
      fclose(df);
    }
  }
  {
    const char *raw_dir = getenv("GALOSH_INT_RAW_DUMP_DIR");
    if(raw_dir) {
      char path[1024];
      snprintf(path, sizeof(path), "%s/p7_loess_h.bin", raw_dir);
      FILE *df = fopen(path, "wb");
      if(df) {
        fwrite(C1_loess_h, sizeof(fxp32), chsize, df);
        fwrite(C2_loess_h, sizeof(fxp32), chsize, df);
        fwrite(C3_loess_h, sizeof(fxp32), chsize, df);
        fclose(df);
      }
      /* Full P7 output: the 3 half-res chroma estimates that feed P8. */
      snprintf(path, sizeof(path), "%s/p7_full.bin", raw_dir);
      df = fopen(path, "wb");
      if(df) {
        fwrite(C1_loess_h, sizeof(fxp32), chsize, df);
        fwrite(C2_loess_h, sizeof(fxp32), chsize, df);
        fwrite(C3_loess_h, sizeof(fxp32), chsize, df);
        fwrite(C1_q_up, sizeof(fxp32), chsize, df);
        fwrite(C2_q_up, sizeof(fxp32), chsize, df);
        fwrite(C3_q_up, sizeof(fxp32), chsize, df);
        fwrite(C1_e_up, sizeof(fxp32), chsize, df);
        fwrite(C2_e_up, sizeof(fxp32), chsize, df);
        fwrite(C3_e_up, sizeof(fxp32), chsize, df);
        fclose(df);
      }
      if(g_verbose) fprintf(stderr, "  P7_RAW chs=%d\n", (int)chsize);
    }
  }

  /* ========== Phase 8: smoothstep slider walk over 4 chroma anchors ========== */
  fxp32 *C1_h_den_q20 = (fxp32 *)malloc(chsize * sizeof(fxp32));
  fxp32 *C2_h_den_q20 = (fxp32 *)malloc(chsize * sizeof(fxp32));
  fxp32 *C3_h_den_q20 = (fxp32 *)malloc(chsize * sizeof(fxp32));
  if(!C1_h_den_q20 || !C2_h_den_q20 || !C3_h_den_q20) { fprintf(stderr, "alloc Phase 8 failed\n"); return 1; }
  clock_t tp8 = clock();
  {
    /* Determine segment + smoothstep param from chroma_str (default 1.0 → seg 0). */
    fxp32 s_q = fxp_from_float(chroma_str);
    int segment;
    fxp32 t_raw_q;
    fxp32 *A1, *A2, *A3, *B1, *B2, *B3;
    if(s_q <= 0) {
      segment = -1; t_raw_q = 0;
      A1 = A2 = A3 = B1 = B2 = B3 = NULL;
    } else if(s_q <= FXP_ONE) {
      segment = 0; t_raw_q = s_q;
      A1 = C1_h_q20; A2 = C2_h_q20; A3 = C3_h_q20;
      B1 = C1_loess_h; B2 = C2_loess_h; B3 = C3_loess_h;
    } else if(s_q <= 2 * FXP_ONE) {
      segment = 1; t_raw_q = s_q - FXP_ONE;
      A1 = C1_loess_h; A2 = C2_loess_h; A3 = C3_loess_h;
      B1 = C1_q_up;    B2 = C2_q_up;    B3 = C3_q_up;
    } else if(s_q <= 3 * FXP_ONE) {
      segment = 2; t_raw_q = s_q - 2 * FXP_ONE;
      A1 = C1_q_up;    A2 = C2_q_up;    A3 = C3_q_up;
      B1 = C1_e_up;    B2 = C2_e_up;    B3 = C3_e_up;
    } else {
      segment = 3; t_raw_q = FXP_ONE;
      A1 = C1_e_up; A2 = C2_e_up; A3 = C3_e_up;
      B1 = B2 = B3 = NULL;
    }
    if(segment < 0) {
      memcpy(C1_h_den_q20, C1_h_q20, chsize * sizeof(fxp32));
      memcpy(C2_h_den_q20, C2_h_q20, chsize * sizeof(fxp32));
      memcpy(C3_h_den_q20, C3_h_q20, chsize * sizeof(fxp32));
    } else if(segment >= 3 || B1 == NULL) {
      memcpy(C1_h_den_q20, A1, chsize * sizeof(fxp32));
      memcpy(C2_h_den_q20, A2, chsize * sizeof(fxp32));
      memcpy(C3_h_den_q20, A3, chsize * sizeof(fxp32));
    } else {
      /* smoothstep t = t² * (3 − 2t) in Q11.20. */
      fxp32 t_sq = fxp_mul(t_raw_q, t_raw_q);
      fxp32 three_minus_2t = 3 * FXP_ONE - (t_raw_q << 1);
      fxp32 t = fxp_mul(t_sq, three_minus_2t);
      fxp32 oneMt = FXP_ONE - t;
      for(size_t i = 0; i < chsize; i++) {
        C1_h_den_q20[i] = fxp_mul(oneMt, A1[i]) + fxp_mul(t, B1[i]);
        C2_h_den_q20[i] = fxp_mul(oneMt, A2[i]) + fxp_mul(t, B2[i]);
        C3_h_den_q20[i] = fxp_mul(oneMt, A3[i]) + fxp_mul(t, B3[i]);
      }
    }
    double dt8 = (double)(clock() - tp8) / CLOCKS_PER_SEC;
    if(g_verbose) fprintf(stderr, "  Phase 8 smoothstep (%.3f s): chroma_str=%.3f segment=%d\n",
            dt8, chroma_str, segment);
    if(dump_dir) {
      char path[1024];
      snprintf(path, sizeof(path), "%s/p8_C1_h_den.bin", dump_dir);
      FILE *df = fopen(path, "wb");
      if(df) {
        for(size_t i = 0; i < chsize; i++) { float v = fxp_to_float(C1_h_den_q20[i]); fwrite(&v, sizeof(float), 1, df); }
        fclose(df);
      }
    }
  }
  {
    const char *raw_dir = getenv("GALOSH_INT_RAW_DUMP_DIR");
    if(raw_dir) {
      char path[1024]; snprintf(path, sizeof(path), "%s/p8.bin", raw_dir);
      FILE *df = fopen(path, "wb");
      if(df) {
        fwrite(C1_h_den_q20, sizeof(fxp32), chsize, df);
        fwrite(C2_h_den_q20, sizeof(fxp32), chsize, df);
        fwrite(C3_h_den_q20, sizeof(fxp32), chsize, df);
        fclose(df);
      }
      if(g_verbose) fprintf(stderr, "  P8_RAW chs=%d\n", (int)chsize);
    }
  }

  /* ========== Phase 9: K16 joint bilateral upsample → full-res aligned ========== */
  fxp32 *C1_aligned_q20 = (fxp32 *)malloc(npixels * sizeof(fxp32));
  fxp32 *C2_aligned_q20 = (fxp32 *)malloc(npixels * sizeof(fxp32));
  fxp32 *C3_aligned_q20 = (fxp32 *)malloc(npixels * sizeof(fxp32));
  if(!C1_aligned_q20 || !C2_aligned_q20 || !C3_aligned_q20) { fprintf(stderr, "alloc Phase 9 failed\n"); return 1; }
  clock_t tp9 = clock();
  fxp_k16_jinc_upsample(C1_h_den_q20, C2_h_den_q20, C3_h_den_q20, L_pixel_q20,
                        C1_aligned_q20, C2_aligned_q20, C3_aligned_q20,
                        halfwidth, halfheight, inv_2sigma_sq_k16);
  double dt9 = (double)(clock() - tp9) / CLOCKS_PER_SEC;
  if(g_verbose) fprintf(stderr, "  Phase 9 K16 upsample → full-res (%.3f s)\n", dt9);
  if(dump_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/p9_C1_aligned.bin", dump_dir);
    FILE *df = fopen(path, "wb");
    if(df) {
      for(size_t i = 0; i < npixels; i++) { float v = fxp_to_float(C1_aligned_q20[i]); fwrite(&v, sizeof(float), 1, df); }
      fclose(df);
    }
  }
  {
    const char *raw_dir = getenv("GALOSH_INT_RAW_DUMP_DIR");
    if(raw_dir) {
      char path[1024]; snprintf(path, sizeof(path), "%s/p9.bin", raw_dir);
      FILE *df = fopen(path, "wb");
      if(df) {
        fwrite(C1_aligned_q20, sizeof(fxp32), npixels, df);
        fwrite(C2_aligned_q20, sizeof(fxp32), npixels, df);
        fwrite(C3_aligned_q20, sizeof(fxp32), npixels, df);
        fclose(df);
      }
      if(g_verbose) fprintf(stderr, "  P9_RAW npix=%d\n", (int)npixels);
    }
  }

  /* ========== Phase 10: fused inverse WHT + dark restore + denorm + inverse GAT ========== */
  static const int SIGNS[4][3] = {
    { +1, +1, +1 },   /* R  */
    { -1, +1, -1 },   /* Gb */
    { +1, -1, -1 },   /* Gr */
    { -1, -1, +1 },   /* B  */
  };
  clock_t tp10 = clock();
  for(int fr = 0; fr < height; fr++) {
    int r_off = fr & 1;
    for(int fc = 0; fc < width; fc++) {
      int c_off = fc & 1;
      int slot = r_off | (c_off << 1);
      size_t pos = (size_t)fr * width + fc;
      fxp32 c1 = C1_aligned_q20[pos];
      fxp32 c2 = C2_aligned_q20[pos];
      fxp32 c3 = C3_aligned_q20[pos];
      fxp32 sumC = (SIGNS[slot][0] > 0 ? c1 : -c1)
                 + (SIGNS[slot][1] > 0 ? c2 : -c2)
                 + (SIGNS[slot][2] > 0 ? c3 : -c3);
      fxp32 inner = (L_pixel_q20[pos] + sumC) >> 1;   /* ÷2 from WHT inv */
      fxp32 val = inner + ch_dark_ref[slot];
      fxp32 val_denorm = fxp_mul(val, unified_sigma_q);
      /* Foi-exact LUT inverse for Phase 10 (built once per image above).
       * Properly handles low-signal regime where algebraic returns negative. */
      out_q20[pos] = fxp_gat_inverse_exact(val_denorm, &gat_p);
    }
  }
  double dt10 = (double)(clock() - tp10) / CLOCKS_PER_SEC;
  if(g_verbose) fprintf(stderr, "  Phase 10 inverse WHT + denorm + inv-GAT (%.3f s)\n", dt10);
  if(dump_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/p10_output.bin", dump_dir);
    FILE *df = fopen(path, "wb");
    if(df) {
      for(size_t i = 0; i < npixels; i++) { float v = fxp_to_float(out_q20[i]); fwrite(&v, sizeof(float), 1, df); }
      fclose(df);
    }
  }
  {
    const char *raw_dir = getenv("GALOSH_INT_RAW_DUMP_DIR");
    if(raw_dir) {
      char path[1024]; snprintf(path, sizeof(path), "%s/p10.bin", raw_dir);
      FILE *df = fopen(path, "wb");
      if(df) { fwrite(out_q20, sizeof(fxp32), npixels, df); fclose(df); }
      if(g_verbose) fprintf(stderr, "  P10_RAW npix=%d\n", (int)npixels);
    }
  }

  /* ========== Final output: Phase 10 result in raw [0,1] space ========== */
  for(size_t i = 0; i < npixels; i++) {
    float v = fxp_to_float(out_q20[i]);
    if(v < 0.0f) v = 0.0f;
    if(v > 1.0f) v = 1.0f;
    out_f32[i] = v;
  }

  FILE *fout = fopen(output_file, "wb");
  if(!fout) { fprintf(stderr, "Cannot open %s for writing\n", output_file); return 1; }
  fwrite(out_f32, sizeof(float), npixels, fout);
  fclose(fout);
  if(g_verbose) fprintf(stderr, "  Output: %s (Phase 1 GAT-domain, clamped /4 for viewing)\n",
          output_file);

#ifdef GALOSH_OPCOUNT
  { long long px = (long long)width * height;
    fprintf(stderr, "[OPCOUNT] pixels=%lld  MAC=%lld (%.1f/px)  SF=%lld (%.2f/px)\n",
            px, g_n_mac, (double)g_n_mac / (double)px,
            g_n_sf, (double)g_n_sf / (double)px); }
#endif
  free(in_f32); free(out_f32); free(in_q20); free(out_q20);
  return 0;
}
