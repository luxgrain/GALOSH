/* ============================================================================
 *  galosh_int_p4.cl  — Phase 4 kernel (half-res chroma extraction).
 *
 *  k_p4_chroma_halfres : parallel per half-res pixel.  From the 2x2 GAT block
 *  (a,b,cc,d) extract the three chroma differences (bit-mirror of
 *  phase4_chroma_halfres):
 *     C1 = (a - b + cc - d) / 2
 *     C2 = (a + b - cc - d) / 2
 *     C3 = (a - b - cc + d) / 2
 *  Input is the P2 dark-subtracted GAT image (in_gat); outputs are 3 half-res
 *  chroma line buffers.  (For even W/H — e.g. SIDD 256x256 — no edge skip.)
 * ========================================================================== */

__kernel void k_p4_chroma_halfres(__global const int *in_gat,
                                  __global int *C1, __global int *C2,
                                  __global int *C3,
                                  int width, int height,
                                  int halfwidth, int halfheight) {
  int idx = get_global_id(0);
  if(idx >= halfwidth * halfheight) return;
  int hr = idx / halfwidth;
  int hc = idx - hr * halfwidth;
  int fr0 = 2 * hr, fr1 = fr0 + 1;
  if(fr1 >= height) return;
  int fc0 = 2 * hc, fc1 = fc0 + 1;
  if(fc1 >= width) return;
  int a  = in_gat[(size_t)fr0 * width + fc0];
  int b  = in_gat[(size_t)fr1 * width + fc0];
  int cc = in_gat[(size_t)fr0 * width + fc1];
  int d  = in_gat[(size_t)fr1 * width + fc1];
  C1[idx] = (a - b + cc - d) >> 1;
  C2[idx] = (a + b - cc - d) >> 1;
  C3[idx] = (a - b - cc + d) >> 1;
}
