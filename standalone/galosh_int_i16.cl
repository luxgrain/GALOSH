/* ============================================================================
 *  galosh_int_i16.cl  — INT16 line-buffer narrowing (requantize-in-place).
 *
 *  k_requant_i16 rounds a Q11.20 line buffer to the precision of its INT16
 *  storage format and saturates to the INT16 value range, writing the result
 *  back in the Q11.20 container.  This faithfully models a genuine INT16 line
 *  buffer (round-trip = round-to-N-frac + saturate-to-int16) while keeping the
 *  compute kernels INT32 — letting us validate ZERO-LOSS on the real pipeline
 *  before the mechanical storage-type change.
 *
 *  store_shift = 20 - frac:  luma Q10.5 -> 15, chroma Q6.9 -> 11, out Q1.14 -> 6.
 *  INT16 holds [-32768, 32767] in the buffer's own Q-format.
 * ========================================================================== */

__kernel void k_requant_i16(__global int *buf, int n, int store_shift) {
  int idx = get_global_id(0);
  if(idx >= n) return;
  int q = (buf[idx] + (1 << (store_shift - 1))) >> store_shift;
  if(q >  32767) q =  32767;
  if(q < -32768) q = -32768;
  buf[idx] = q << store_shift;
}
