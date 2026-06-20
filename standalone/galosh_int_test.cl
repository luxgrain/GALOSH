/* ============================================================================
 *  galosh_int_test.cl  — F2 validation kernels for the INT primitive port.
 *
 *  Each kernel applies ONE galosh_int.clh primitive element-wise so the host
 *  (galosh_int_test.c) can compare GPU output bit-for-bit against the CPU
 *  reference (galosh_cpu_int.h).  The host PREPENDS galosh_int.clh to this
 *  file before clCreateProgramWithSource (no #include needed).
 *
 *  STAGE F1-A coverage: mul, div_q20, recip, add_sat, sub_sat,
 *  acc extract/div_i32/div_acc, median, wht2d.
 * ========================================================================== */

__kernel void k_mul(__global const int *a, __global const int *b,
                    __global int *out) {
  int i = get_global_id(0);
  out[i] = fxp_mul(a[i], b[i]);
}

__kernel void k_div_q20(__global const int *a, __global const int *b,
                        __global int *out) {
  int i = get_global_id(0);
  out[i] = fxp_div_q20(a[i], b[i]);
}

__kernel void k_recip(__global const int *a, __global int *out) {
  int i = get_global_id(0);
  out[i] = fxp_recip(a[i]);
}

__kernel void k_add_sat(__global const int *a, __global const int *b,
                        __global int *out) {
  int i = get_global_id(0);
  out[i] = fxp_add_sat(a[i], b[i]);
}

__kernel void k_sub_sat(__global const int *a, __global const int *b,
                        __global int *out) {
  int i = get_global_id(0);
  out[i] = fxp_sub_sat(a[i], b[i]);
}

/* acc = from_int32(a); acc += b; extract_q20  (no >>FRAC) */
__kernel void k_extract_q20(__global const int *a, __global const int *b,
                            __global int *out) {
  int i = get_global_id(0);
  fxp_acc acc = fxp_acc_from_int32(a[i]);
  fxp_acc_add_i32(&acc, b[i]);
  out[i] = fxp_acc_extract_q20(&acc);
}

/* acc = madd(a,b); out = acc_div_i32(acc, div) */
__kernel void k_div_i32(__global const int *a, __global const int *b,
                        __global const int *dv, __global int *out) {
  int i = get_global_id(0);
  fxp_acc acc = fxp_acc_zero();
  fxp_acc_madd(&acc, a[i], b[i]);
  out[i] = fxp_acc_div_i32(&acc, dv[i]);
}

/* num=madd(a,b); den=madd(c,d); out = acc_div_acc(num,den)  (Q11.20) */
__kernel void k_div_acc(__global const int *a, __global const int *b,
                        __global const int *c, __global const int *d,
                        __global int *out) {
  int i = get_global_id(0);
  fxp_acc n = fxp_acc_zero(); fxp_acc_madd(&n, a[i], b[i]);
  fxp_acc m = fxp_acc_zero(); fxp_acc_madd(&m, c[i], d[i]);
  out[i] = fxp_acc_div_acc(&n, &m);
}

/* one work-item per 64-block; median of first n */
__kernel void k_median(__global const int *in, __global int *out, int n) {
  int g = get_global_id(0);
  int buf[64];
  for(int j = 0; j < 64; j++) buf[j] = in[g * 64 + j];
  out[g] = fxp_partial_selection_median(buf, n);
}

/* one work-item per 64-block; 2D 8x8 WHT */
__kernel void k_wht2d(__global const int *in, __global int *out,
                      int normalize) {
  int g = get_global_id(0);
  int blk[64];
  for(int j = 0; j < 64; j++) blk[j] = in[g * 64 + j];
  fxp_wht2d_8x8(blk, normalize);
  for(int j = 0; j < 64; j++) out[g * 64 + j] = blk[j];
}

/* ======================= Stage F1-B (table-driven) ======================= */

__kernel void k_sqrt(__global const int *a, __global int *out,
                     __constant fxp_tables *T) {
  int i = get_global_id(0);
  out[i] = fxp_sqrt(a[i], T);
}

__kernel void k_log(__global const int *a, __global int *out,
                    __constant fxp_tables *T) {
  int i = get_global_id(0);
  out[i] = fxp_log(a[i], T);
}

__kernel void k_exp(__global const int *a, __global int *out,
                    __constant fxp_tables *T) {
  int i = get_global_id(0);
  out[i] = fxp_exp(a[i], T);
}

__kernel void k_gat_fwd(__global const int *a, __global int *out,
                        __constant fxp_gat_params *P, __constant fxp_tables *T) {
  int i = get_global_id(0);
  fxp_gat_params p = *P;
  out[i] = fxp_gat_forward(a[i], &p, T);
}

__kernel void k_gat_inv(__global const int *a, __global int *out,
                        __constant fxp_gat_params *P) {
  int i = get_global_id(0);
  fxp_gat_params p = *P;
  out[i] = fxp_gat_inverse_algebraic(a[i], &p);
}

__kernel void k_poisson(__global const int *lam, __global const int *kk,
                        __global int *out, __constant fxp_tables *T) {
  int i = get_global_id(0);
  out[i] = fxp_poisson_log_pdf(lam[i], kk[i], T);
}
