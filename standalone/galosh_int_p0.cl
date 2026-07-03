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

/* ----------------------------------------------------------------------------
 * k_p0_estimate — PARALLEL (single-workgroup).
 *
 *  - thread 0 runs the lightweight block-histogram binning (Pass 1/2/3 over the
 *    ~32k CFA blocks) + the 5-iter Huber-WLS solve.  These are small reductions
 *    (block grid + 32 bins) so staying serial is fine; they set alpha_est /
 *    sigma_sq_est into LDS.
 *  - ALL P0_WG threads then cooperatively build the phase-0d dark-refine
 *    histograms (the dominant cost: ~4M |Laplacian| samples over quarter-res)
 *    via LDS atomic counts.  A histogram COUNT is order-independent, so the
 *    parallel tally is BIT-IDENTICAL to the serial one (k_p0_estimate_serial
 *    below) -> r32 bit-exactness preserved.  thread 0 then finishes the
 *    percentile/median scalar logic.
 *
 *  Launch with global=local=P0_WG (one work-group).  The phase-0d thresh/lap
 *  histograms live in LDS (l_hist[4096]); the block-binning prepass/vhist stay
 *  in the global scratch args (thread-0 serial).
 * -------------------------------------------------------------------------- */
#define P0_WG 256

__kernel __attribute__((reqd_work_group_size(P0_WG, 1, 1)))
void k_p0_estimate(__global const int *in_q20, int width, int height,
                   int n_bx, int n_by,
                   __global const int *blk_mean,
                   __global const int *blk_var,
                   __global const int *blk_valid,
                   __constant fxp_p0_consts *C,
                   __global int *prepass_hist,   /* [256] */
                   __global int *vhist,          /* [NE_NBINS*NE_VAR_BINS] */
                   __global int *thresh_hist,    /* [4096] (unused: LDS path) */
                   __global int *lap_hist,       /* [4096] (unused: LDS path) */
                   __global int *out_alpha,
                   __global int *out_sigma) {
  const int tid = get_local_id(0);
  __local int l_hist[NE_DARK_HIST_BINS];   /* 4096 ints = 16 KB LDS */
  __local int l_done, l_alpha, l_sigma, l_dark_max;

  const int var_scale_combined = C->var_scale_combined;
  const int sat_threshold      = C->sat_threshold;
  const int per_ch = n_by * n_bx;
  const int total_blocks = 4 * per_ch;

  /* ===== thread 0: block binning + WLS (serial; ~32k blocks, 32 bins) ===== */
  if(tid == 0) {
    *out_alpha = C->alpha_init;   /* defaults (= fxp_from_float(1e-4f)) */
    *out_sigma = 1;
    l_done = 0;
    if(total_blocks < 100) { l_done = 1; goto p0_serial_done; }

    const int PREPASS_BINS = 256;
    for(int i = 0; i < PREPASS_BINS; i++) prepass_hist[i] = 0;
    int prepass_total = 0;
    for(int ch = 0; ch < 4; ch++)
      for(int by = 0; by < n_by; by++)
        for(int bx = 0; bx < n_bx; bx++) {
          int bi = ch * per_ch + by * n_bx + bx;
          if(!blk_valid[bi]) continue;
          int m = blk_mean[bi];
          if(m >= sat_threshold) continue;
          int b = (int)(m >> 12);
          if(b < 0) b = 0;
          if(b >= PREPASS_BINS) b = PREPASS_BINS - 1;
          prepass_hist[b]++; prepass_total++;
        }
    if(prepass_total < 100) { l_done = 1; goto p0_serial_done; }

    int p5_target  = prepass_total / 20;
    int p95_target = prepass_total - p5_target;
    int p5_bin = 0, p95_bin = PREPASS_BINS - 1;
    { int cum = 0, found_p5 = 0;
      for(int b = 0; b < PREPASS_BINS; b++) {
        cum += prepass_hist[b];
        if(!found_p5 && cum >= p5_target) { p5_bin = b; found_p5 = 1; }
        if(cum >= p95_target) { p95_bin = b; break; }
      } }
    int mean_lo = (int)p5_bin << 12;
    int mean_hi = (int)(p95_bin + 1) << 12;
    if(mean_hi <= mean_lo + (int)NE_NBINS) mean_hi = mean_lo + (int)NE_NBINS;
    if(mean_hi > sat_threshold) mean_hi = sat_threshold;
    if(mean_lo < 0) mean_lo = 0;
    const int mean_bw = (mean_hi - mean_lo) / NE_NBINS;
    if(mean_bw < 1) { l_done = 1; goto p0_serial_done; }

    int     cnt_bin[NE_NBINS];
    fxp_acc msum_bin[NE_NBINS];
    int     vmin_bin[NE_NBINS], vmax_bin[NE_NBINS];
    for(int b = 0; b < NE_NBINS; b++) {
      cnt_bin[b] = 0; msum_bin[b] = fxp_acc_zero();
      vmin_bin[b] = FXP_MAX_INT; vmax_bin[b] = 0;
    }
    for(int ch = 0; ch < 4; ch++)
      for(int by = 0; by < n_by; by++)
        for(int bx = 0; bx < n_bx; bx++) {
          int bi = ch * per_ch + by * n_bx + bx;
          if(!blk_valid[bi]) continue;
          int m = blk_mean[bi], v = blk_var[bi];
          if(m <= mean_lo || m >= mean_hi) continue;
          int b = (int)((m - mean_lo) / mean_bw);
          if(b < 0) b = 0;
          if(b >= NE_NBINS) b = NE_NBINS - 1;
          cnt_bin[b]++; fxp_acc_add_i32(&msum_bin[b], m);
          if(v < vmin_bin[b]) vmin_bin[b] = v;
          if(v > vmax_bin[b]) vmax_bin[b] = v;
        }

    for(int i = 0; i < NE_NBINS * NE_VAR_BINS; i++) vhist[i] = 0;
    int inv_vrange_bin[NE_NBINS];
    for(int b = 0; b < NE_NBINS; b++) {
      if(cnt_bin[b] < 20) { inv_vrange_bin[b] = 0; continue; }
      int vrange = vmax_bin[b] - vmin_bin[b];
      if(vrange < 1) vrange = 1;
      inv_vrange_bin[b] = fxp_recip(vrange);
    }
    for(int ch = 0; ch < 4; ch++)
      for(int by = 0; by < n_by; by++)
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
      if(vcnt > 0) { int vsum_q = fxp_acc_extract_q20(&vsum_acc); bin_var_arr[b] = vsum_q / vcnt; }
      else         { bin_var_arr[b] = vmin_bin[b]; }
      bin_mean_arr[b] = fxp_acc_extract_q20(&msum_bin[b]) / cnt_bin[b];
      bin_cnt_arr[b]  = (vcnt > 0) ? vcnt : 1;
      bin_valid[b]    = 1; n_valid++;
    }
    if(n_valid < 4) { l_done = 1; goto p0_serial_done; }

    int alpha_est, sigma_sq_est;
    phase0c_wls_solve(bin_mean_arr, bin_var_arr, bin_cnt_arr, bin_valid,
                      NE_NBINS, C->alpha_init, C->huber_factor_q,
                      &alpha_est, &sigma_sq_est);
    /* Poisson-monotonicity fallback (案B) — mirrors galosh_raw_cpu_int.c.  Only
     * when the full-bin solve floors alpha (collapse) do we re-solve on the
     * monotone-increasing variance prefix and adopt it iff it lifts alpha off the
     * floor; gating on the floored alpha keeps healthy scenes bit-identical
     * (plomberie__ISO100 collapse fix).  bin_valid is dead after this point
     * (phase-0d does not read it), so we invalidate the tail IN PLACE rather than
     * a private bv_cut copy — keeps the kernel within its reqd_work_group_size
     * register budget; the adopted alpha is identical to the CPU copy form. */
    if(alpha_est <= 1) {
      int run_max = 0, viol = 0, cut_from = -1, nv_cut = n_valid;
      for(int b = 0; b < NE_NBINS; b++) {
        if(!bin_valid[b]) continue;
        /* no-INT64: split-multiply, EXACT for run_max>=0 (mirrors galosh_raw_cpu_int.c) */
      int thr = (run_max / 5) * 3 + ((run_max % 5) * 3) / 5;
        if(bin_var_arr[b] < thr) { if(viol == 0) cut_from = b; if(++viol >= 2) break; }
        else { viol = 0; cut_from = -1; if(bin_var_arr[b] > run_max) run_max = bin_var_arr[b]; }
      }
      if(viol >= 2 && cut_from >= 0) {
        for(int b = cut_from; b < NE_NBINS; b++) if(bin_valid[b]) { bin_valid[b] = 0; nv_cut--; }
        if(nv_cut >= 4) {
          int a2, s2;
          phase0c_wls_solve(bin_mean_arr, bin_var_arr, bin_cnt_arr, bin_valid,
                            NE_NBINS, C->alpha_init, C->huber_factor_q, &a2, &s2);
          if(a2 > alpha_est) { alpha_est = a2; sigma_sq_est = s2; }
        }
      }
    }
    l_alpha = alpha_est;
    l_sigma = sigma_sq_est;   /* phase-0d may overwrite below */
p0_serial_done: ;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if(l_done) return;          /* defaults already written by thread 0 */

  /* ===== phase 0d (parallel): dark-percentile sigma^2 refinement ===== */
  const int alpha_in    = l_alpha;
  const int halfwidth   = (width  + 1) / 2;
  const int halfheight  = (height + 1) / 2;
  const int dark_offset = C->dark_offset_002;

  /* --- thresh_hist (subsampled by 3) -> dark_thresh / dark_max --- */
  for(int i = tid; i < NE_DARK_HIST_BINS; i += P0_WG) l_hist[i] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  const int nsy = (halfheight + 2) / 3, nsx = (halfwidth + 2) / 3;
  const int nthr = nsy * nsx;
  for(int t = tid; t < 4 * nthr; t += P0_WG) {
    int ch = t / nthr, rem = t - ch * nthr;
    int sy = rem / nsx, sx = rem - sy * nsx;
    int dy0 = (ch >> 1) & 1, dx0 = ch & 1;
    int rr = 2 * (sy * 3) + dy0, cc = 2 * (sx * 3) + dx0;
    if(rr >= height || cc >= width) continue;
    int v = in_q20[(size_t)rr * width + cc];
    int b = (int)(v >> (FXP_FRAC - 12));
    if(b < 0) b = 0;
    if(b >= NE_DARK_HIST_BINS) b = NE_DARK_HIST_BINS - 1;
    atomic_inc(&l_hist[b]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __local int l_dark_thresh;
  if(tid == 0) {
    int total_thresh = 0;
    for(int i = 0; i < NE_DARK_HIST_BINS; i++) total_thresh += l_hist[i];
    int target = total_thresh / 10;
    int cum = 0, dark_bin = 0;
    for(int i = 0; i < NE_DARK_HIST_BINS; i++) {
      cum += l_hist[i];
      if(cum >= target) { dark_bin = i; break; }
    }
    int dark_thresh = (int)((dark_bin << (FXP_FRAC - 12)) + (1 << (FXP_FRAC - 13)));
    l_dark_thresh = dark_thresh;
    l_dark_max = dark_thresh + dark_offset;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int dark_max = l_dark_max;

  /* --- lap_hist (full quarter-res, 2 directions) -> MAD median --- */
  for(int i = tid; i < NE_DARK_HIST_BINS; i += P0_WG) l_hist[i] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  const int nxh = halfwidth - 2;     /* horizontal: x in [0,halfwidth-2) */
  if(nxh > 0) {
    /* no-INT64: flattened index fits int32 (4*hh*nxh <= W*H < 2^31 for any real frame) */
    const int hn_h = halfheight * nxh;
    int tot_h = 4 * hn_h;
    for(int t = tid; t < tot_h; t += P0_WG) {
      int ch = t / hn_h;
      int rem = t - ch * hn_h;
      int y = rem / nxh, x = rem - y * nxh;
      int dy0 = (ch >> 1) & 1, dx0 = ch & 1;
      int r = 2 * y + dy0, c0 = 2 * x + dx0, c1 = 2 * (x+1) + dx0, c2 = 2 * (x+2) + dx0;
      if(r >= height || c2 >= width) continue;
      int v0 = in_q20[(size_t)r * width + c0];
      int v1 = in_q20[(size_t)r * width + c1];
      int v2 = in_q20[(size_t)r * width + c2];
      if(v0 > dark_max || v1 > dark_max || v2 > dark_max) continue;
      int lap = v0 - (v1 << 1) + v2; if(lap < 0) lap = -lap;
      int b = (int)((lap * 40) >> 10);
      if(b < 0) b = 0;
      if(b >= NE_DARK_HIST_BINS) b = NE_DARK_HIST_BINS - 1;
      atomic_inc(&l_hist[b]);
    }
  }
  const int nyv = halfheight - 2;    /* vertical: y in [0,halfheight-2) */
  if(nyv > 0) {
    /* no-INT64: flattened index fits int32 (see horizontal pass note) */
    const int hn_v = nyv * halfwidth;
    int tot_v = 4 * hn_v;
    for(int t = tid; t < tot_v; t += P0_WG) {
      int ch = t / hn_v;
      int rem = t - ch * hn_v;
      int y = rem / halfwidth, x = rem - y * halfwidth;
      int dy0 = (ch >> 1) & 1, dx0 = ch & 1;
      int c = 2 * x + dx0, r0 = 2 * y + dy0, r1 = 2 * (y+1) + dy0, r2 = 2 * (y+2) + dy0;
      if(r2 >= height || c >= width) continue;
      int v0 = in_q20[(size_t)r0 * width + c];
      int v1 = in_q20[(size_t)r1 * width + c];
      int v2 = in_q20[(size_t)r2 * width + c];
      if(v0 > dark_max || v1 > dark_max || v2 > dark_max) continue;
      int lap = v0 - (v1 << 1) + v2; if(lap < 0) lap = -lap;
      int b = (int)((lap * 40) >> 10);
      if(b < 0) b = 0;
      if(b >= NE_DARK_HIST_BINS) b = NE_DARK_HIST_BINS - 1;
      atomic_inc(&l_hist[b]);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if(tid == 0) {
    int total_lap = 0;
    for(int i = 0; i < NE_DARK_HIST_BINS; i++) total_lap += l_hist[i];
    int sigma_sq_est = l_sigma;       /* keep WLS value if too few samples */
    if(total_lap >= 100) {
      int target_med = total_lap / 2;
      int cum_l = 0, med_bin = 0;
      for(int i = 0; i < NE_DARK_HIST_BINS; i++) {
        cum_l += l_hist[i];
        if(cum_l >= target_med) { med_bin = i; break; }
      }
      int mad_q = ((int)med_bin * (FXP_ONE / 40960)) + (FXP_ONE / 81920);
      int sigma_lap_scaled = fxp_mul(mad_q, var_scale_combined);
      int dark_var_scaled  = fxp_mul(sigma_lap_scaled, sigma_lap_scaled);
      int dark_mean = l_dark_thresh >> 1;
      int alpha_scaled = (alpha_in <= (FXP_MAX_INT >> 10)) ? (alpha_in << 10) : FXP_MAX_INT;
      int dyn_scaled = dark_var_scaled - fxp_mul(alpha_scaled, dark_mean);
      if(dyn_scaled < 0) dyn_scaled = 0;
      sigma_sq_est = dyn_scaled >> 10;
    }
    *out_alpha = l_alpha;
    *out_sigma = sigma_sq_est;
  }
}

/* ----------------------------------------------------------------------------
 * [DEPRECATED] k_p0_estimate_serial — original single work-item reference,
 * retained as the bit-exact oracle for the parallel kernel above.
 * -------------------------------------------------------------------------- */
__kernel void k_p0_estimate_serial(__global const int *in_q20, int width, int height,
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
  /* Poisson-monotonicity fallback (案B) — see k_p0_estimate / galosh_raw_cpu_int.c.
   * bin_valid is dead after this point so the tail is invalidated in place. */
  if(alpha_est <= 1) {
    int run_max = 0, viol = 0, cut_from = -1, nv_cut = n_valid;
    for(int b = 0; b < NE_NBINS; b++) {
      if(!bin_valid[b]) continue;
      /* no-INT64: split-multiply, EXACT for run_max>=0 (mirrors galosh_raw_cpu_int.c) */
      int thr = (run_max / 5) * 3 + ((run_max % 5) * 3) / 5;
      if(bin_var_arr[b] < thr) { if(viol == 0) cut_from = b; if(++viol >= 2) break; }
      else { viol = 0; cut_from = -1; if(bin_var_arr[b] > run_max) run_max = bin_var_arr[b]; }
    }
    if(viol >= 2 && cut_from >= 0) {
      for(int b = cut_from; b < NE_NBINS; b++) if(bin_valid[b]) { bin_valid[b] = 0; nv_cut--; }
      if(nv_cut >= 4) {
        int a2, s2;
        phase0c_wls_solve(bin_mean_arr, bin_var_arr, bin_cnt_arr, bin_valid,
                          NE_NBINS, C->alpha_init, C->huber_factor_q, &a2, &s2);
        if(a2 > alpha_est) { alpha_est = a2; sigma_sq_est = s2; }
      }
    }
  }
  phase0d_dark_refine(in_q20, width, height, alpha_est, &sigma_sq_est,
                      thresh_hist, lap_hist, var_scale_combined,
                      C->dark_offset_002);

  *out_alpha = alpha_est;
  *out_sigma = sigma_sq_est;
}
