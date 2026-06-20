/* ============================================================================
 *  galosh_int_p3.cl  — Phase 3 kernel (forward L, stride-1 cycle-spinning WHT).
 *
 *  k_p3_forward_l : parallel per-pixel.  For each (r,c), L_cs[r,c] =
 *  (a + b + cc + d) / 2 over the 2x2 block at (r,c) with mirror padding at the
 *  right/bottom edges.  Bit-mirror of phase3_forward_l_stride1.
 *
 *  Output L_cs is a SEPARATE full-res luma line buffer (input in_gat is the
 *  P2 dark-subtracted GAT image, unchanged).
 * ========================================================================== */

inline int fxp_h_mirror_idx(int i, int n) {
  if(i < 0)  return -i;
  if(i >= n) return 2 * n - i - 2;
  return i;
}

__kernel void k_p3_forward_l(__global const int *in_gat, __global int *L_cs,
                             int width, int height) {
  int idx = get_global_id(0);
  if(idx >= width * height) return;
  int r = idx / width;
  int c = idx - r * width;
  int rb = fxp_h_mirror_idx(r + 1, height);
  int cb = fxp_h_mirror_idx(c + 1, width);
  int a  = in_gat[(size_t)r  * width + c ];
  int b  = in_gat[(size_t)rb * width + c ];
  int cc = in_gat[(size_t)r  * width + cb];
  int d  = in_gat[(size_t)rb * width + cb];
  L_cs[idx] = (a + b + cc + d) >> 1;
}
