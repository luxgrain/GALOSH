/* ============================================================================
 *  galosh_int_p0.cl  — Phase 0 kernels (blind noise estimation).
 *
 *  k_p0_block_stats : parallel, one work-item per CFA block -> blk_mean/var/valid.
 *  k_p0_estimate    : single work-item, mirrors fxp_estimate_noise reduction
 *                     (prepass adaptive range -> pass1/2/3 binning ->
 *                      phase0c_wls_solve -> phase0d_dark_refine).
 *
 *  Concatenation order (host): galosh_int.clh + galosh_int_tbl.clh +
 *  galosh_int_p0.clh + galosh_int_p0.cl.
 *
 *  Correctness-first: the estimate kernel is single-threaded and uses __global
 *  scratch for the histograms (prepass_hist[256], vhist[32*128],
 *  thresh_hist[4096], lap_hist[4096]).  Bit-exact vs the r32 CPU reference is
 *  the goal; parallelising the reduction is a later perf pass.
 * ========================================================================== */

__kernel void k_p0_block_stats(__global const int *in_q20, int width, int height,
                               int n_bx, int n_by, int var_scale_combined,
                               __global int *blk_mean, __global int *blk_var,
                               __global int *blk_valid) {
  int bi = get_global_id(0);
  int per_ch = n_by * n_bx;
  if(bi >= 4 * per_ch) return;
  int ch  = bi / per_ch;
  int rem = bi - ch * per_ch;
  int by  = rem / n_bx;
  int bx  = rem - by * n_bx;

  int m = 0, v = 0;
  int valid = compute_block_stats(in_q20, width, height, ch, by, bx,
                                  &m, &v, var_scale_combined);
  blk_mean[bi]  = valid ? m : 0;
  blk_var[bi]   = valid ? v : FXP_MAX_INT;
  blk_valid[bi] = valid;
}

__kernel void k_p0_estimate(__global const int *in_q20, int width, int height,
                            int n_bx, int n_by,
                            __global const int *blk_mean,
                            __global const int *blk_var,
                            __global const int *blk_valid,
                            __constant fxp_p0_consts *C,
                            __global int *prepass_hist,   /* [256] */
                            __global int *vhist,          /* [NE_NBINS*NE_VAR_BINS] */
                            __global int *thresh_hist,    /* [4096] */
                            __global int *lap_hist,       /* [4096] */
                            __global int *out_alpha,
                            __global int *out_sigma) {
  if(get_global_id(0) != 0) return;

  const int var_scale_combined = C->var_scale_combined;
  const int sat_threshold      = C->sat_threshold;

  *out_alpha = C->alpha_init;     /* defaults (= fxp_from_float(1e-4f)) */
  *out_sigma = 1;

  const int per_ch = n_by * n_bx;
  const int total_blocks = 4 * per_ch;
  if(total_blocks < 100) return;

  /* ===== Pre-pass: 256-bin coarse mean histogram -> adaptive [lo,hi] ===== */
  const int PREPASS_BINS = 256;
  for(int i = 0; i < PREPASS_BINS; i++) prepass_hist[i] = 0;
  int prepass_total = 0;
  for(int ch = 0; ch < 4; ch++) {
    for(int by = 0; by < n_by; by++) {
      for(int bx = 0; bx < n_bx; bx++) {
        int bi = ch * per_ch + by * n_bx + bx;
        if(!blk_valid[bi]) continue;
        int m = blk_mean[bi];
        if(m >= sat_threshold) continue;
        int b = (int)(m >> 12);
        if(b < 0) b = 0;
        if(b >= PREPASS_BINS) b = PREPASS_BINS - 1;
        prepass_hist[b]++;
        prepass_total++;
      }
    }
  }
  if(prepass_total < 100) return;

  int p5_target  = prepass_total / 20;
  int p95_target = prepass_total - p5_target;
  int p5_bin = 0, p95_bin = PREPASS_BINS - 1;
  {
    int cum = 0, found_p5 = 0;
    for(int b = 0; b < PREPASS_BINS; b++) {
      cum += prepass_hist[b];
      if(!found_p5 && cum >= p5_target) { p5_bin = b; found_p5 = 1; }
      if(cum >= p95_target) { p95_bin = b; break; }
    }
  }
  int mean_lo = (int)p5_bin << 12;
  int mean_hi = (int)(p95_bin + 1) << 12;
  if(mean_hi <= mean_lo + (int)NE_NBINS) mean_hi = mean_lo + (int)NE_NBINS;
  if(mean_hi > sat_threshold) mean_hi = sat_threshold;
  if(mean_lo < 0) mean_lo = 0;
  const int mean_bw = (mean_hi - mean_lo) / NE_NBINS;
  if(mean_bw < 1) return;

  /* ===== Pass 1: per-mean-bin cnt + msum + vmin + vmax ===== */
  int     cnt_bin[NE_NBINS];
  fxp_acc msum_bin[NE_NBINS];
  int     vmin_bin[NE_NBINS], vmax_bin[NE_NBINS];
  for(int b = 0; b < NE_NBINS; b++) {
    cnt_bin[b] = 0;
    msum_bin[b] = fxp_acc_zero();
    vmin_bin[b] = FXP_MAX_INT;
    vmax_bin[b] = 0;
  }
  for(int ch = 0; ch < 4; ch++) {
    for(int by = 0; by < n_by; by++) {
      for(int bx = 0; bx < n_bx; bx++) {
        int bi = ch * per_ch + by * n_bx + bx;
        if(!blk_valid[bi]) continue;
        int m = blk_mean[bi], v = blk_var[bi];
        if(m <= mean_lo || m >= mean_hi) continue;
        int b = (int)((m - mean_lo) / mean_bw);
        if(b < 0) b = 0;
        if(b >= NE_NBINS) b = NE_NBINS - 1;
        cnt_bin[b]++;
        fxp_acc_add_i32(&msum_bin[b], m);
        if(v < vmin_bin[b]) vmin_bin[b] = v;
        if(v > vmax_bin[b]) vmax_bin[b] = v;
      }
    }
  }

  /* ===== Pass 2: per-(mean-bin, var-sub-bin) histogram ===== */
  for(int i = 0; i < NE_NBINS * NE_VAR_BINS; i++) vhist[i] = 0;
  int inv_vrange_bin[NE_NBINS];
  for(int b = 0; b < NE_NBINS; b++) {
    if(cnt_bin[b] < 20) { inv_vrange_bin[b] = 0; continue; }
    int vrange = vmax_bin[b] - vmin_bin[b];
    if(vrange < 1) vrange = 1;
    inv_vrange_bin[b] = fxp_recip(vrange);
  }
  for(int ch = 0; ch < 4; ch++) {
    for(int by = 0; by < n_by; by++) {
      for(int bx = 0; bx < n_bx; bx++) {
        int bi = ch * per_ch + by * n_bx + bx;
        if(!blk_valid[bi]) continue;
        int m = blk_mean[bi], v = blk_var[bi];
        if(m <= mean_lo || m >= mean_hi) continue;
        int b = (int)((m - mean_lo) / mean_bw);
        if(b < 0) b = 0;
        if(b >= NE_NBINS) b = NE_NBINS - 1;
        if(cnt_bin[b] < 20 || inv_vrange_bin[b] == 0) continue;
        int dv = v - vmin_bin[b];
        int frac = fxp_mul(dv, inv_vrange_bin[b]);
        int vbin = (int)(frac >> 13);
        if(vbin < 0) vbin = 0;
        if(vbin >= NE_VAR_BINS) vbin = NE_VAR_BINS - 1;
        vhist[b * NE_VAR_BINS + vbin]++;
      }
    }
  }

  /* ===== Pass 3: cumsum -> lower-envelope mean var per mean-bin ===== */
  int bin_mean_arr[NE_NBINS], bin_var_arr[NE_NBINS];
  int bin_cnt_arr[NE_NBINS],  bin_valid[NE_NBINS];
  int n_valid = 0;
  for(int b = 0; b < NE_NBINS; b++) {
    bin_valid[b] = 0;
    if(cnt_bin[b] < 20) continue;
    int p5t  = cnt_bin[b] / 20;
    int p20t = cnt_bin[b] / 5;
    int cum = 0, pp5 = 0, pp20 = NE_VAR_BINS - 1, found_p5 = 0;
    for(int i = 0; i < NE_VAR_BINS; i++) {
      cum += vhist[b * NE_VAR_BINS + i];
      if(!found_p5 && cum >= p5t) { pp5 = i; found_p5 = 1; }
      if(cum >= p20t) { pp20 = i; break; }
    }
    int vrange = vmax_bin[b] - vmin_bin[b];
    if(vrange < 1) vrange = 1;
    fxp_acc vsum_acc = fxp_acc_zero();
    int vcnt = 0;
    for(int i = pp5; i <= pp20; i++) {
      int i_q = (int)i << FXP_FRAC;
      int i_plus_half = i_q + (FXP_ONE >> 1);
      int t = fxp_mul(i_plus_half, vrange) >> 7;
      int bin_center = vmin_bin[b] + t;
      int n = vhist[b * NE_VAR_BINS + i];
      for(int k = 0; k < n; k++) fxp_acc_add_i32(&vsum_acc, bin_center);
      vcnt += n;
    }
    if(vcnt > 0) {
      int vsum_q = fxp_acc_extract_q20(&vsum_acc);
      bin_var_arr[b] = vsum_q / vcnt;
    } else {
      bin_var_arr[b] = vmin_bin[b];
    }
    bin_mean_arr[b] = fxp_acc_extract_q20(&msum_bin[b]) / cnt_bin[b];
    bin_cnt_arr[b]  = (vcnt > 0) ? vcnt : 1;
    bin_valid[b]    = 1;
    n_valid++;
  }
  if(n_valid < 4) return;

  int alpha_est, sigma_sq_est;
  phase0c_wls_solve(bin_mean_arr, bin_var_arr, bin_cnt_arr, bin_valid,
                    NE_NBINS, C->alpha_init, C->huber_factor_q,
                    &alpha_est, &sigma_sq_est);
  phase0d_dark_refine(in_q20, width, height, alpha_est, &sigma_sq_est,
                      thresh_hist, lap_hist, var_scale_combined,
                      C->dark_offset_002);

  *out_alpha = alpha_est;
  *out_sigma = sigma_sq_est;
}
