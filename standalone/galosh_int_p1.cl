/* ============================================================================
 *  galosh_int_p1.cl  — Phase 1 kernels.
 *
 *  k_p1_gat_forward : parallel per-pixel  gat = GAT_forward(in)
 *  k_p1_sigma_ch    : 4 work-items (one per CFA) -> sigma_ch[4]
 *  k_p1_unify       : single work-item -> unified_sigma + 1/unified_sigma
 *  k_p1_normalize   : parallel per-pixel  gat *= 1/unified_sigma
 *
 *  Concatenation order (host): galosh_int.clh + galosh_int_tbl.clh +
 *  galosh_int_p0.clh + galosh_int_p1.clh + (kernels) galosh_int_p0.cl +
 *  galosh_int_p1.cl.
 * ========================================================================== */

__kernel void k_p1_gat_forward(__global const int *in_q20, __global lbuf_t *gat_q20,
                               int npix, __constant fxp_gat_params *P,
                               __constant fxp_tables *T) {
  int i = get_global_id(0);
  if(i >= npix) return;
  fxp_gat_params p = *P;
  STB(gat_q20, i, fxp_gat_forward(in_q20[i], &p, T));
}

/* offsets {0,0},{0,1},{1,0},{1,1} indexed by CFA slot s.
 *
 * PARALLEL: launched as 4 work-groups (global=4*P1_WG, local=P1_WG); group_id =
 * CFA channel s.  Each work-group's P1_WG threads cooperatively tally that
 * channel's 4096-bin |Laplacian| MAD histogram in LDS via atomic counts, then
 * thread 0 of the group does the median scalar logic.  Histogram counts are
 * order-independent -> BIT-IDENTICAL to the serial halfres_stream
 * (k_p1_sigma_ch_serial below) -> r32 bit-exactness preserved. */
#define P1_WG 256

__kernel __attribute__((reqd_work_group_size(P1_WG, 1, 1)))
void k_p1_sigma_ch(__global const lbuf_t *gat_q20, int width, int height,
                   __global int *lap_hist_all /* unused: LDS path */, int inv_1p6521,
                   __global int *sigma_ch) {
  const int s = get_group_id(0);     /* CFA channel 0..3 */
  const int tid = get_local_id(0);
  if(s >= 4) return;
  const int dy0 = (s >> 1) & 1, dx0 = s & 1;
  const int halfwidth  = (width  - dx0 + 1) / 2;
  const int halfheight = (height - dy0 + 1) / 2;
  __local int l_hist[NE_DARK_HIST_BINS];
  if(halfwidth < 4 || halfheight < 4) { if(tid == 0) sigma_ch[s] = FXP_ONE; return; }

  for(int i = tid; i < NE_DARK_HIST_BINS; i += P1_WG) l_hist[i] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  /* horizontal Laplacian: y in [0,halfheight), x in [0,halfwidth-2) */
  const int nxh = halfwidth - 2;
  if(nxh > 0) {
    /* no-INT64: hh*nxh <= W*H/4 < 2^31 for any real frame */
    int tot = halfheight * nxh;
    for(int t = tid; t < tot; t += P1_WG) {
      int y = t / nxh, x = t - y * nxh;
      int r = 2*y + dy0, c0 = 2*x + dx0, c1 = 2*(x+1) + dx0, c2 = 2*(x+2) + dx0;
      if(r >= height || c2 >= width) continue;
      int v0 = LDB(gat_q20, (size_t)r * width + c0);
      int v1 = LDB(gat_q20, (size_t)r * width + c1);
      int v2 = LDB(gat_q20, (size_t)r * width + c2);
      int lap = v0 - (v1 << 1) + v2; if(lap < 0) lap = -lap;
      int b = (int)(lap >> 11);
      if(b < 0) b = 0;
      if(b >= NE_DARK_HIST_BINS) b = NE_DARK_HIST_BINS - 1;
      atomic_inc(&l_hist[b]);
    }
  }
  /* vertical Laplacian: y in [0,halfheight-2), x in [0,halfwidth) */
  const int nyv = halfheight - 2;
  if(nyv > 0) {
    /* no-INT64: nyv*hw <= W*H/4 < 2^31 for any real frame */
    int tot = nyv * halfwidth;
    for(int t = tid; t < tot; t += P1_WG) {
      int y = t / halfwidth, x = t - y * halfwidth;
      int c = 2*x + dx0, r0 = 2*y + dy0, r1 = 2*(y+1) + dy0, r2 = 2*(y+2) + dy0;
      if(r2 >= height || c >= width) continue;
      int v0 = LDB(gat_q20, (size_t)r0 * width + c);
      int v1 = LDB(gat_q20, (size_t)r1 * width + c);
      int v2 = LDB(gat_q20, (size_t)r2 * width + c);
      int lap = v0 - (v1 << 1) + v2; if(lap < 0) lap = -lap;
      int b = (int)(lap >> 11);
      if(b < 0) b = 0;
      if(b >= NE_DARK_HIST_BINS) b = NE_DARK_HIST_BINS - 1;
      atomic_inc(&l_hist[b]);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if(tid == 0) {
    int total = 0;
    for(int i = 0; i < NE_DARK_HIST_BINS; i++) total += l_hist[i];
    if(total < 100) { sigma_ch[s] = FXP_ONE; return; }
    int target = total / 2;
    int cum = 0, med_bin = 0;
    for(int i = 0; i < NE_DARK_HIST_BINS; i++) {
      cum += l_hist[i];
      if(cum >= target) { med_bin = i; break; }
    }
    int mad_q = ((int)med_bin << 11) + (1 << 10);
    int sigma = fxp_mul(mad_q, inv_1p6521);
    if(sigma < 1) sigma = 1;
    sigma_ch[s] = sigma;
  }
}

/* [DEPRECATED] serial oracle (4 work-items, one per channel). */
__kernel void k_p1_sigma_ch_serial(__global const lbuf_t *gat_q20, int width, int height,
                            __global int *lap_hist_all, int inv_1p6521,
                            __global int *sigma_ch) {
  int s = get_global_id(0);
  if(s >= 4) return;
  int dy0 = (s >> 1) & 1;
  int dx0 = s & 1;
  sigma_ch[s] = fxp_estimate_gat_sigma_halfres_stream(
      gat_q20, width, height, dy0, dx0,
      lap_hist_all + s * NE_DARK_HIST_BINS, inv_1p6521);
}

__kernel void k_p1_unify(__global const int *sigma_ch, __global int *unified_out,
                         __global int *inv_sg_out, __constant fxp_tables *T) {
  if(get_global_id(0) != 0) return;
  fxp_acc sum_sq = fxp_acc_zero();
  for(int s = 0; s < 4; s++)
    fxp_acc_add_i32(&sum_sq, fxp_mul(sigma_ch[s], sigma_ch[s]));
  int sum_sq_q = fxp_acc_extract_q20(&sum_sq);
  int mean_var = sum_sq_q >> 2;
  if(mean_var < 1) mean_var = 1;
  int unified = fxp_sqrt(mean_var, T);
  if(unified < 1) unified = 1;
  *unified_out = unified;
  *inv_sg_out  = fxp_recip(unified);
}

__kernel void k_p1_normalize(__global lbuf_t *gat_q20, int npix,
                             __global const int *inv_sg_buf) {
  int i = get_global_id(0);
  if(i >= npix) return;
  STB(gat_q20, i, fxp_mul(LDB(gat_q20, i), inv_sg_buf[0]));
}
