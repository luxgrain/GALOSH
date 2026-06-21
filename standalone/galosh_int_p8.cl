/* ============================================================================
 *  galosh_int_p8.cl  — Phase 8 kernel (smoothstep chroma slider).
 *
 *  k_p8_smoothstep : parallel per-pixel lerp  out = (1-t)*A + t*B over the 3
 *  chroma channels.  The segment/anchor choice (A,B) and the smoothstep weight
 *  t are derived from chroma_str on the HOST (per-image), so the kernel is a
 *  plain blend.  Bit-mirror of the Phase 8 inner loop (copy cases map to t=0).
 * ========================================================================== */

__kernel void k_p8_smoothstep(__global const lbuf_t *a1, __global const lbuf_t *a2,
                              __global const lbuf_t *a3, __global const lbuf_t *b1,
                              __global const lbuf_t *b2, __global const lbuf_t *b3,
                              __global lbuf_t *o1, __global lbuf_t *o2, __global lbuf_t *o3,
                              int n, int oneMt, int t) {
  int idx = get_global_id(0);
  if(idx >= n) return;
  STB(o1, idx, fxp_mul(oneMt, LDB(a1, idx)) + fxp_mul(t, LDB(b1, idx)));
  STB(o2, idx, fxp_mul(oneMt, LDB(a2, idx)) + fxp_mul(t, LDB(b2, idx)));
  STB(o3, idx, fxp_mul(oneMt, LDB(a3, idx)) + fxp_mul(t, LDB(b3, idx)));
}
