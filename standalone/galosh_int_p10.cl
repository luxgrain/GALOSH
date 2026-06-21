/* ============================================================================
 *  galosh_int_p10.cl  — Phase 10 kernel (fused inverse WHT + dark restore +
 *  denorm + Foi-exact inverse GAT).  Parallel per output pixel.
 *  Bit-mirror of the Phase 10 main loop in galosh_raw_cpu_int.c.
 * ========================================================================== */

__kernel void k_p10_reconstruct(__global const lbuf_t *c1a, __global const lbuf_t *c2a,
                                __global const lbuf_t *c3a, __global const lbuf_t *L_pixel,
                                __global lbuf_t *out, int width, int height,
                                int ch0, int ch1, int ch2, int ch3,
                                int unified_sigma, __constant fxp_gat_params *P,
                                __constant fxp_gat_inv_table_t *T) {
  int idx = get_global_id(0);
  if(idx >= width * height) return;
  int fr = idx / width, fc = idx - fr * width;
  int slot = (fr & 1) | ((fc & 1) << 1);
  int c1 = LDB(c1a, idx), c2 = LDB(c2a, idx), c3 = LDB(c3a, idx);
  int s0 = P10_SIGNS[slot][0], s1 = P10_SIGNS[slot][1], s2 = P10_SIGNS[slot][2];
  int sumC = (s0 > 0 ? c1 : -c1) + (s1 > 0 ? c2 : -c2) + (s2 > 0 ? c3 : -c3);
  int inner = (LDB(L_pixel, idx) + sumC) >> 1;
  int dref = (slot == 0) ? ch0 : (slot == 1) ? ch1 : (slot == 2) ? ch2 : ch3;
  int val = inner + dref;
  int val_denorm = fxp_mul(val, unified_sigma);
  fxp_gat_params p = *P;
  STO(out, idx, p10_gat_inverse_exact(val_denorm, &p, T));
}

/* Foi-exact inverse GAT LUT build, one work-item per entry (Poisson x 8-pt
 * Gauss-Hermite).  Bit-mirror of the host build_foi_lut inner loop — replaces
 * the ~300 ms/image host build with a parallel kernel.  z_table[g] =
 * sqrt(2)*sigma_raw*node[g+1] is precomputed on the host (sigma-dependent). */
__kernel void k_build_foi_lut(__global int *out_d, __global int *out_x,
                              __constant fxp_gat_params *P,
                              __constant fxp_tables *T, __constant int *z_table) {
  int i = get_global_id(0);
  if(i >= FXP_GAT_INV_TABLE_SIZE) return;
  fxp_gat_params p = *P;
  fxp_acc xtmp = fxp_acc_zero();
  fxp_acc_madd(&xtmp, i, FXP_ONE);
  int x_val = fxp_acc_div_i32(&xtmp, FXP_GAT_INV_TABLE_SIZE - 1);
  int alpha_safe = (p.alpha < 1) ? 1 : p.alpha;
  int lambda = fxp_div_q20(x_val, alpha_safe);
  int lambda_int = lambda >> FXP_FRAC;
  int sqrt_lambda = (lambda > 0) ? fxp_sqrt(lambda, T) : 0;
  int sqrt_lambda_int = sqrt_lambda >> FXP_FRAC;
  int k_max = lambda_int + 6 * sqrt_lambda_int + 10;
  fxp_acc expected_gat = fxp_acc_zero();
  const int break_thresh = -16 * FXP_ONE;
  for(int k = 0; k <= k_max; k++) {
    int log_prob = fxp_poisson_log_pdf(lambda, k, T);
    if(log_prob < break_thresh && k > lambda_int + 1) break;
    int prob = fxp_exp(log_prob, T);
    if(prob <= 0) continue;
    fxp_acc eg = fxp_acc_zero();
    int k_alpha = k * p.alpha;
    for(int g = 0; g < 8; g++) {
      int noisy_y = k_alpha + z_table[g];
      int Tval = fxp_gat_forward(noisy_y, &p, T);
      fxp_acc_madd(&eg, FXP_GH_W[g + 1], Tval);
    }
    int eg_q = fxp_mul(FXP_INV_SQRT_PI_Q20, fxp_acc_to_fxp32(&eg));
    fxp_acc_madd(&expected_gat, prob, eg_q);
  }
  out_d[i] = fxp_acc_to_fxp32(&expected_gat);
  out_x[i] = x_val;
}
