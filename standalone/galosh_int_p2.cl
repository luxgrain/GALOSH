/* ============================================================================
 *  galosh_int_p2.cl  — Phase 2 kernels (dark_ref IRLS + CFA-aware subtract).
 *
 *  k_p2_dark_ref : single work-item, mirrors phase2_dark_ref (3-iter IRLS over
 *                  achromatic 2x2 cells) -> ch_dark_ref[4].
 *  k_p2_subtract : parallel per-pixel  in_gat[i] -= ch_dark_ref[slot].
 *
 *  Float-derived constants (0.05, 50.0, achromatic-range 4.0) computed on host
 *  via fxp_from_float and passed in.  Depends on galosh_int.clh + tbl (sqrt).
 * ========================================================================== */

__kernel void k_p2_dark_ref(__global const int *in_q20, __global const lbuf_t *in_gat,
                            int width, int height, int alpha, int sigma_sq,
                            int c_0p05, int c_50, int achroma,
                            __constant fxp_tables *T, __global int *ch_out) {
  if(get_global_id(0) != 0) return;

  int s_init = fxp_div_q20(sigma_sq, alpha < 1 ? 1 : alpha);
  if(s_init < 1) s_init = 1;
  int s_min = fxp_mul(s_init, c_0p05);
  int s_max = fxp_mul(s_init, c_50);
  if(s_min < 1) s_min = 1;
  int s_scale = s_init;

  int ch[4]; ch[0] = ch[1] = ch[2] = ch[3] = 0;
  const int n_iter = 2;
  for(int iter = 0; iter <= n_iter; iter++) {
    int inv_s = fxp_recip(s_scale < 1 ? 1 : s_scale);
    fxp_acc sw  = fxp_acc_zero();
    fxp_acc sw0 = fxp_acc_zero(), sw1 = fxp_acc_zero();
    fxp_acc sw2 = fxp_acc_zero(), sw3 = fxp_acc_zero();

    for(int br = 0; br < height - 1; br += 2) {
      for(int bc = 0; bc < width - 1; bc += 2) {
        int g0 = LDB(in_gat, (size_t)br     * width + bc    );
        int g1 = LDB(in_gat, (size_t)(br+1) * width + bc    );
        int g2 = LDB(in_gat, (size_t)br     * width + bc + 1);
        int g3 = LDB(in_gat, (size_t)(br+1) * width + bc + 1);
        int gmax = g0; if(g1 > gmax) gmax = g1; if(g2 > gmax) gmax = g2; if(g3 > gmax) gmax = g3;
        int gmin = g0; if(g1 < gmin) gmin = g1; if(g2 < gmin) gmin = g2; if(g3 < gmin) gmin = g3;
        if(gmax - gmin > achroma) continue;
        int iv0 = in_q20[(size_t)br     * width + bc    ];
        int iv1 = in_q20[(size_t)(br+1) * width + bc    ];
        int iv2 = in_q20[(size_t)br     * width + bc + 1];
        int iv3 = in_q20[(size_t)(br+1) * width + bc + 1];
        int L_raw = (iv0 + iv1 + iv2 + iv3) >> 2;
        int r  = fxp_mul(L_raw, inv_s);
        int r2 = fxp_mul(r, r);
        int r4 = fxp_mul(r2, r2);
        int denom = (r4 < FXP_MAX_INT - FXP_ONE) ? (FXP_ONE + r4) : FXP_MAX_INT;
        int w = fxp_recip(denom);
        if(w <= 0) continue;
        fxp_acc_add_i32(&sw,  w  >> 10);
        fxp_acc_add_i32(&sw0, fxp_mul(w, g0) >> 10);
        fxp_acc_add_i32(&sw1, fxp_mul(w, g1) >> 10);
        fxp_acc_add_i32(&sw2, fxp_mul(w, g2) >> 10);
        fxp_acc_add_i32(&sw3, fxp_mul(w, g3) >> 10);
      }
    }

    int sw_q = fxp_acc_extract_q20(&sw);
    if(sw_q < 1) sw_q = 1;
    int inv_sw = fxp_recip(sw_q);
    ch[0] = fxp_mul(fxp_acc_extract_q20(&sw0), inv_sw);
    ch[1] = fxp_mul(fxp_acc_extract_q20(&sw1), inv_sw);
    ch[2] = fxp_mul(fxp_acc_extract_q20(&sw2), inv_sw);
    ch[3] = fxp_mul(fxp_acc_extract_q20(&sw3), inv_sw);

    if(iter == n_iter) break;

    fxp_acc swW  = fxp_acc_zero();
    fxp_acc swR2 = fxp_acc_zero();
    for(int br = 0; br < height - 1; br += 2) {
      for(int bc = 0; bc < width - 1; bc += 2) {
        int g0 = LDB(in_gat, (size_t)br     * width + bc    );
        int g1 = LDB(in_gat, (size_t)(br+1) * width + bc    );
        int g2 = LDB(in_gat, (size_t)br     * width + bc + 1);
        int g3 = LDB(in_gat, (size_t)(br+1) * width + bc + 1);
        int gmax = g0; if(g1 > gmax) gmax = g1; if(g2 > gmax) gmax = g2; if(g3 > gmax) gmax = g3;
        int gmin = g0; if(g1 < gmin) gmin = g1; if(g2 < gmin) gmin = g2; if(g3 < gmin) gmin = g3;
        if(gmax - gmin > achroma) continue;
        int iv0 = in_q20[(size_t)br     * width + bc    ];
        int iv1 = in_q20[(size_t)(br+1) * width + bc    ];
        int iv2 = in_q20[(size_t)br     * width + bc + 1];
        int iv3 = in_q20[(size_t)(br+1) * width + bc + 1];
        int L_raw = (iv0 + iv1 + iv2 + iv3) >> 2;
        int r  = fxp_mul(L_raw, inv_s);
        int r2 = fxp_mul(r, r);
        int r4 = fxp_mul(r2, r2);
        int denom = (r4 < FXP_MAX_INT - FXP_ONE) ? (FXP_ONE + r4) : FXP_MAX_INT;
        int w = fxp_recip(denom);
        if(w <= 0) continue;
        int d0 = g0 - ch[0];
        int d1 = g1 - ch[1];
        int d2 = g2 - ch[2];
        int d3 = g3 - ch[3];
        int r2_sum = fxp_mul(d0, d0) + fxp_mul(d1, d1) + fxp_mul(d2, d2) + fxp_mul(d3, d3);
        int wr2 = (fxp_mul(w, r2_sum) >> 2) >> 10;
        fxp_acc_add_i32(&swW,  w >> 10);
        fxp_acc_add_i32(&swR2, wr2);
      }
    }
    int swW_q  = fxp_acc_extract_q20(&swW);
    int swR2_q = fxp_acc_extract_q20(&swR2);
    if(swW_q < 1) swW_q = 1;
    int mse = fxp_mul(swR2_q, fxp_recip(swW_q));
    if(mse < 1) mse = 1;
    int measured_std = fxp_sqrt(mse, T);
    if(measured_std < 1) measured_std = 1;
    int ratio = fxp_recip(measured_std);
    int sqrt_ratio = fxp_sqrt(ratio, T);
    s_scale = fxp_mul(s_scale, sqrt_ratio);
    if(s_scale < s_min) s_scale = s_min;
    if(s_scale > s_max) s_scale = s_max;
  }

  ch_out[0] = ch[0]; ch_out[1] = ch[1]; ch_out[2] = ch[2]; ch_out[3] = ch[3];
}

__kernel void k_p2_subtract(__global lbuf_t *in_gat, int width, int height,
                            __global const int *ch_ref) {
  int idx = get_global_id(0);
  if(idx >= width * height) return;
  int r = idx / width, c = idx - r * width;
  int slot = (r & 1) | ((c & 1) << 1);
  STB(in_gat, idx, LDB(in_gat, idx) - ch_ref[slot]);
}
