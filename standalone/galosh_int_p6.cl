/* ============================================================================
 *  galosh_int_p6.cl  — Phase 6 kernels (L_pixel + L_h_den from L_cs_den).
 *
 *  k_p6_l_pixel : parallel per-pixel 2x2 overlap-average of L_cs_den over the
 *                 top-left neighborhood {(fr,fc),(fr-1,fc),(fr,fc-1),(fr-1,fc-1)}
 *                 with in-bounds count (shifts for count 1/2/4, divide for 3).
 *  k_p6_l_h_den : parallel half-res subsample  L_h_den[hr,hc] = L_cs_den[2hr,2hc].
 *  Bit-mirror of phase6_l_pixel / phase6_l_h_den.
 * ========================================================================== */

__kernel void k_p6_l_pixel(__global const lbuf_t *lden, __global lbuf_t *lpix,
                           int width, int height) {
  int idx = get_global_id(0);
  if(idx >= width * height) return;
  int fr = idx / width, fc = idx - fr * width;
  int sum = LDB(lden, idx);
  int count = 1;
  if(fr > 0) { sum += LDB(lden, (size_t)(fr - 1) * width + fc); count++; }
  if(fc > 0) { sum += LDB(lden, (size_t)fr * width + (fc - 1)); count++; }
  if(fr > 0 && fc > 0) { sum += LDB(lden, (size_t)(fr - 1) * width + (fc - 1)); count++; }
  STB(lpix, idx, (count == 4) ? (sum >> 2) :
                 (count == 2) ? (sum >> 1) :
                 (count == 1) ? sum : (sum / count));
}

__kernel void k_p6_l_h_den(__global const lbuf_t *lden, __global lbuf_t *lhden,
                           int width, int height, int halfwidth, int halfheight) {
  int idx = get_global_id(0);
  if(idx >= halfwidth * halfheight) return;
  int hr = idx / halfwidth, hc = idx - hr * halfwidth;
  int fr = 2 * hr, fc = 2 * hc;
  if(fr >= height || fc >= width) return;
  STB(lhden, idx, LDB(lden, (size_t)fr * width + fc));
}
