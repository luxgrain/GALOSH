/* ============================================================================
 *  galosh_int_p8.cl  — Phase 8 kernel (smoothstep chroma slider).
 *
 *  k_p8_smoothstep : parallel per-pixel lerp  out = (1-t)*A + t*B over the 3
 *  chroma channels.  The segment/anchor choice (A,B) and the smoothstep weight
 *  t are derived from chroma_str on the HOST (per-image), so the kernel is a
 *  plain blend.  Bit-mirror of the Phase 8 inner loop (copy cases map to t=0).
 * ========================================================================== */

__kernel void k_p8_smoothstep(__global const int *a1, __global const int *a2,
                              __global const int *a3, __global const int *b1,
                              __global const int *b2, __global const int *b3,
                              __global int *o1, __global int *o2, __global int *o3,
                              int n, int oneMt, int t) {
  int idx = get_global_id(0);
  if(idx >= n) return;
  o1[idx] = fxp_mul(oneMt, a1[idx]) + fxp_mul(t, b1[idx]);
  o2[idx] = fxp_mul(oneMt, a2[idx]) + fxp_mul(t, b2[idx]);
  o3[idx] = fxp_mul(oneMt, a3[idx]) + fxp_mul(t, b3[idx]);
}
