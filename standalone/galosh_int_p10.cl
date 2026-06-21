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
