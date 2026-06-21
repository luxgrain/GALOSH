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

/* offsets {0,0},{0,1},{1,0},{1,1} indexed by CFA slot s. */
__kernel void k_p1_sigma_ch(__global const lbuf_t *gat_q20, int width, int height,
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
