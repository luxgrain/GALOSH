/* ============================================================================
 *  galosh_int_p5.cl  — Phase 5 kernels (BayesShrink pilot + empirical Wiener).
 *
 *  Pixel-parallel GATHER (option B): one work-item per output pixel; it
 *  recomputes the <=16 overlapping 8x8 blocks (stride 2) covering the pixel and
 *  accumulates only its own Kaiser-weighted contribution.  Race-free + bit-exact
 *  (exact-integer fxp_acc accumulation is order-independent).
 *
 *  k_p5_pass1 : input (L_cs)            -> pilot
 *  k_p5_pass2 : noisy (L_cs) + pilot    -> output (L_cs_den)
 *
 *  Covering blocks for pixel (pr,pc): even ref_r in [max(0,pr-7), min(rmax,pr)],
 *  even ref_c in [max(0,pc-7), min(cmax,pc)]  (rmax=H-8, cmax=W-8).
 * ========================================================================== */

__kernel void k_p5_pass1(__global const lbuf_t *input, __global lbuf_t *pilot,
                         int width, int height, __constant fxp_tables *T,
                         __constant int *kaiser, __constant fxp_p5_consts *C) {
  int idx = get_global_id(0);
  if(idx >= width * height) return;
  int pr = idx / width, pc = idx - pr * width;
  int rmax = height - 8, cmax = width - 8;
  int rlo = pr - 7; if(rlo < 0) rlo = 0; if(rlo & 1) rlo++;
  int rhi = pr; if(rhi > rmax) rhi = rmax;
  int clo = pc - 7; if(clo < 0) clo = 0; if(clo & 1) clo++;
  int chi = pc; if(chi > cmax) chi = cmax;

  fxp_acc numer = fxp_acc_zero(), denom = fxp_acc_zero();
  int block[64];
  for(int ref_r = rlo; ref_r <= rhi; ref_r += 2) {
    for(int ref_c = clo; ref_c <= chi; ref_c += 2) {
      int nnz = p5_pass1_block(input, width, ref_r, ref_c, block, T, C);
      int weight = fxp_recip(nnz * FXP_ONE);
      int dy = pr - ref_r, dx = pc - ref_c;
      int kw = kaiser[dy * 8 + dx];
      int wkw = fxp_mul(weight, kw);
      fxp_acc_add_i32(&numer, fxp_mul(wkw, block[dy * 8 + dx]));
      fxp_acc_add_i32(&denom, wkw);
    }
  }
  int d = fxp_acc_extract_q20(&denom);
  if(d > 1) {
    int n = fxp_acc_extract_q20(&numer);
    STB(pilot, idx, fxp_mul(n, fxp_recip(d)));
  } else {
    STB(pilot, idx, LDB(input, idx));
  }
}

__kernel void k_p5_pass2(__global const lbuf_t *noisy, __global const lbuf_t *pilot,
                         __global lbuf_t *output, int width, int height,
                         __constant int *kaiser, __constant fxp_p5_consts *C) {
  int idx = get_global_id(0);
  if(idx >= width * height) return;
  int pr = idx / width, pc = idx - pr * width;
  int rmax = height - 8, cmax = width - 8;
  int rlo = pr - 7; if(rlo < 0) rlo = 0; if(rlo & 1) rlo++;
  int rhi = pr; if(rhi > rmax) rhi = rmax;
  int clo = pc - 7; if(clo < 0) clo = 0; if(clo & 1) clo++;
  int chi = pc; if(chi > cmax) chi = cmax;

  fxp_acc numer = fxp_acc_zero(), denom = fxp_acc_zero();
  int block[64];
  for(int ref_r = rlo; ref_r <= rhi; ref_r += 2) {
    for(int ref_c = clo; ref_c <= chi; ref_c += 2) {
      int we = p5_pass2_block(noisy, pilot, width, ref_r, ref_c, block, C);
      int weight = fxp_recip(we);
      int dy = pr - ref_r, dx = pc - ref_c;
      int kw = kaiser[dy * 8 + dx];
      int wkw = fxp_mul(weight, kw);
      fxp_acc_add_i32(&numer, fxp_mul(wkw, block[dy * 8 + dx]));
      fxp_acc_add_i32(&denom, wkw);
    }
  }
  int d = fxp_acc_extract_q20(&denom);
  if(d > 1) {
    int n = fxp_acc_extract_q20(&numer);
    STB(output, idx, fxp_mul(n, fxp_recip(d)));
  } else {
    STB(output, idx, LDB(noisy, idx));
  }
}
