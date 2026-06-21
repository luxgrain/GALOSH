/* ============================================================================
 *  galosh_int_p7.cl  — Phase 7 kernels (per-output-pixel parallel).
 *
 *  k_p7_box_down : 2x2 box downsample (bit-mirror fxp_box_downsample_2x).
 *  k_p7_loess    : R=7 Y-guided 3-channel LOESS (calls p7_loess_pixel).
 *  (k16 jinc upsample lands in a follow-up step.)
 * ========================================================================== */

__kernel void k_p7_box_down(__global const int *src, __global int *dst,
                            int sw, int sh) {
  int dw = sw / 2, dh = sh / 2;
  int idx = get_global_id(0);
  if(idx >= dw * dh) return;
  int y = idx / dw, x = idx - y * dw;
  int sy = 2 * y, sx = 2 * x;
  dst[idx] = (src[(size_t)sy * sw + sx]     + src[(size_t)sy * sw + sx + 1] +
              src[(size_t)(sy+1) * sw + sx] + src[(size_t)(sy+1) * sw + sx + 1]) >> 2;
}

__kernel void k_p7_loess(__global const int *y_guide,
                         __global const int *c1_in, __global const int *c2_in,
                         __global const int *c3_in,
                         __global int *c1_out, __global int *c2_out,
                         __global int *c3_out,
                         int width, int height, int eps_gat_scaled,
                         int inv_2sigma_sq, __constant fxp_tables *T) {
  int idx = get_global_id(0);
  if(idx >= width * height) return;
  int y = idx / width, x = idx - y * width;
  int o1, o2, o3;
  p7_loess_pixel(y_guide, c1_in, c2_in, c3_in, width, height, x, y,
                 eps_gat_scaled, inv_2sigma_sq, T, &o1, &o2, &o3);
  c1_out[idx] = o1; c2_out[idx] = o2; c3_out[idx] = o3;
}
