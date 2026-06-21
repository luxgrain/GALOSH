/* ============================================================================
 *  galosh_int_p7.cl  — Phase 7 kernels (per-output-pixel parallel).
 *
 *  k_p7_box_down : 2x2 box downsample (bit-mirror fxp_box_downsample_2x).
 *  k_p7_loess    : R=7 Y-guided 3-channel LOESS (calls p7_loess_pixel).
 *  (k16 jinc upsample lands in a follow-up step.)
 * ========================================================================== */

__kernel void k_p7_box_down(__global const lbuf_t *src, __global lbuf_t *dst,
                            int sw, int sh) {
  int dw = sw / 2, dh = sh / 2;
  int idx = get_global_id(0);
  if(idx >= dw * dh) return;
  int y = idx / dw, x = idx - y * dw;
  int sy = 2 * y, sx = 2 * x;
  STB(dst, idx, (LDB(src, (size_t)sy * sw + sx)     + LDB(src, (size_t)sy * sw + sx + 1) +
                 LDB(src, (size_t)(sy+1) * sw + sx) + LDB(src, (size_t)(sy+1) * sw + sx + 1)) >> 2);
}

__kernel void k_p7_loess(__global const lbuf_t *y_guide,
                         __global const lbuf_t *c1_in, __global const lbuf_t *c2_in,
                         __global const lbuf_t *c3_in,
                         __global lbuf_t *c1_out, __global lbuf_t *c2_out,
                         __global lbuf_t *c3_out,
                         int width, int height, int eps_gat_scaled,
                         int inv_2sigma_sq, __constant fxp_tables *T) {
  int idx = get_global_id(0);
  if(idx >= width * height) return;
  int y = idx / width, x = idx - y * width;
  int o1, o2, o3;
  p7_loess_pixel(y_guide, c1_in, c2_in, c3_in, width, height, x, y,
                 eps_gat_scaled, inv_2sigma_sq, T, &o1, &o2, &o3);
  STB(c1_out, idx, o1); STB(c2_out, idx, o2); STB(c3_out, idx, o3);
}

/* K16 jinc bilateral upsample.  One work-item per full-res output pixel
 * (output grid = 2*halfwidth x 2*halfheight). */
__kernel void k_p7_k16(__global const lbuf_t *c1_h, __global const lbuf_t *c2_h,
                       __global const lbuf_t *c3_h, __global const lbuf_t *L_pixel,
                       __global lbuf_t *c1_full, __global lbuf_t *c2_full,
                       __global lbuf_t *c3_full, int halfwidth, int halfheight,
                       int inv_2sigma_sq, __constant fxp_tables *T) {
  int fw = 2 * halfwidth, fh = 2 * halfheight;
  int fp = get_global_id(0);
  if(fp >= fw * fh) return;
  int fr = fp / fw, fc = fp - fr * fw;
  int o1, o2, o3;
  p7_k16_pixel(c1_h, c2_h, c3_h, L_pixel, halfwidth, halfheight, fr, fc,
               inv_2sigma_sq, T, &o1, &o2, &o3);
  STB(c1_full, fp, o1); STB(c2_full, fp, o2); STB(c3_full, fp, o3);
}
