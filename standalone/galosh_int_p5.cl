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

/* ----------------------------------------------------------------------------
 * TILED overlap-add (k_p5_pass1 / k_p5_pass2).
 *
 *  The pixel-parallel gather below recomputes each 8x8 WHT block once per
 *  covering pixel (~64x redundant).  These tiled kernels instead assign ONE
 *  work-group to an 8x8 output tile: its 64 threads first compute the <=64
 *  stride-2 blocks that cover the tile EXACTLY ONCE into LDS (one block per
 *  thread), barrier, then each thread gathers its output pixel's <=16 covering
 *  blocks from LDS.  The per-pixel fxp_acc accumulation is byte-for-byte the
 *  same arithmetic as the gather kernels (only the block VALUES are cached, not
 *  recomputed), so the result is BIT-IDENTICAL to k_p5_passN_gather and to the
 *  r32 CPU reference.  ~10x fewer WHT evaluations.
 *
 *  Launch: global = n_tiles*64, local = 64 (reqd).  Tile (ti,tj) origin
 *  (TR,TC) = (ti,tj)*8; LDS block grid origin (rlo,clo) = even-rounded
 *  max(0,TR-7)/max(0,TC-7) so every tile pixel's covering blocks map into the
 *  8x8 LDS grid: lbi=(ref_r-rlo)/2, lbj=(ref_c-clo)/2, both in [0,8).
 * -------------------------------------------------------------------------- */
#define P5_TS 8       /* output tile = 8x8 pixels (= 64 threads) */
#define P5_NB 8       /* <=8 even stride-2 block positions per axis */

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void k_p5_pass1(__global const lbuf_t *input, __global lbuf_t *pilot,
                int width, int height, __constant fxp_tables *T,
                __constant int *kaiser, __constant fxp_p5_consts *C) {
  const int tid = get_local_id(0);
  const int ntx = (width + P5_TS - 1) / P5_TS;
  const int tile = get_group_id(0);
  const int ti = tile / ntx, tj = tile - ti * ntx;
  const int TR = ti * P5_TS, TC = tj * P5_TS;
  const int rmax = height - 8, cmax = width - 8;
  int rlo = TR - 7; if(rlo < 0) rlo = 0; if(rlo & 1) rlo++;
  int clo = TC - 7; if(clo < 0) clo = 0; if(clo & 1) clo++;
  int rhi = TR + P5_TS - 1; if(rhi > rmax) rhi = rmax;
  int chi = TC + P5_TS - 1; if(chi > cmax) chi = cmax;

  __local int lblk[P5_NB * P5_NB * 64];
  __local int lwt[P5_NB * P5_NB];

  /* phase 1: thread tid computes the block at grid cell (bi,bj). */
  int bi = tid >> 3, bj = tid & 7;
  int ref_r = rlo + 2 * bi, ref_c = clo + 2 * bj;
  if(ref_r <= rhi && ref_c <= chi) {
    int block[64];
    int nnz = p5_pass1_block(input, width, ref_r, ref_c, block, T, C);
    int weight = fxp_recip(nnz * FXP_ONE);
    for(int i = 0; i < 64; i++) lblk[tid * 64 + i] = block[i];
    lwt[tid] = weight;
  } else {
    lwt[tid] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* phase 2: thread tid is output pixel (TR+bi, TC+bj). */
  int pr = TR + bi, pc = TC + bj;
  if(pr >= height || pc >= width) return;
  int prlo = pr - 7; if(prlo < 0) prlo = 0; if(prlo & 1) prlo++;
  int prhi = pr; if(prhi > rmax) prhi = rmax;
  int pclo = pc - 7; if(pclo < 0) pclo = 0; if(pclo & 1) pclo++;
  int pchi = pc; if(pchi > cmax) pchi = cmax;
  fxp_acc numer = fxp_acc_zero(), denom = fxp_acc_zero();
  for(int rr = prlo; rr <= prhi; rr += 2) {
    for(int cc = pclo; cc <= pchi; cc += 2) {
      int li = ((rr - rlo) >> 1) * P5_NB + ((cc - clo) >> 1);
      int weight = lwt[li];
      int dy = pr - rr, dx = pc - cc;
      int kw = kaiser[dy * 8 + dx];
      int wkw = fxp_mul(weight, kw);
      fxp_acc_add_i32(&numer, fxp_mul(wkw, lblk[li * 64 + dy * 8 + dx]));
      fxp_acc_add_i32(&denom, wkw);
    }
  }
  int idx = pr * width + pc;
  int d = fxp_acc_extract_q20(&denom);
  if(d > 1) { int n = fxp_acc_extract_q20(&numer); STB(pilot, idx, fxp_mul(n, fxp_recip(d))); }
  else      { STB(pilot, idx, LDB(input, idx)); }
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void k_p5_pass2(__global const lbuf_t *noisy, __global const lbuf_t *pilot,
                __global lbuf_t *output, int width, int height,
                __constant int *kaiser, __constant fxp_p5_consts *C) {
  const int tid = get_local_id(0);
  const int ntx = (width + P5_TS - 1) / P5_TS;
  const int tile = get_group_id(0);
  const int ti = tile / ntx, tj = tile - ti * ntx;
  const int TR = ti * P5_TS, TC = tj * P5_TS;
  const int rmax = height - 8, cmax = width - 8;
  int rlo = TR - 7; if(rlo < 0) rlo = 0; if(rlo & 1) rlo++;
  int clo = TC - 7; if(clo < 0) clo = 0; if(clo & 1) clo++;
  int rhi = TR + P5_TS - 1; if(rhi > rmax) rhi = rmax;
  int chi = TC + P5_TS - 1; if(chi > cmax) chi = cmax;

  __local int lblk[P5_NB * P5_NB * 64];
  __local int lwt[P5_NB * P5_NB];

  int bi = tid >> 3, bj = tid & 7;
  int ref_r = rlo + 2 * bi, ref_c = clo + 2 * bj;
  if(ref_r <= rhi && ref_c <= chi) {
    int block[64];
    int we = p5_pass2_block(noisy, pilot, width, ref_r, ref_c, block, C);
    int weight = fxp_recip(we);
    for(int i = 0; i < 64; i++) lblk[tid * 64 + i] = block[i];
    lwt[tid] = weight;
  } else {
    lwt[tid] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  int pr = TR + bi, pc = TC + bj;
  if(pr >= height || pc >= width) return;
  int prlo = pr - 7; if(prlo < 0) prlo = 0; if(prlo & 1) prlo++;
  int prhi = pr; if(prhi > rmax) prhi = rmax;
  int pclo = pc - 7; if(pclo < 0) pclo = 0; if(pclo & 1) pclo++;
  int pchi = pc; if(pchi > cmax) pchi = cmax;
  fxp_acc numer = fxp_acc_zero(), denom = fxp_acc_zero();
  for(int rr = prlo; rr <= prhi; rr += 2) {
    for(int cc = pclo; cc <= pchi; cc += 2) {
      int li = ((rr - rlo) >> 1) * P5_NB + ((cc - clo) >> 1);
      int weight = lwt[li];
      int dy = pr - rr, dx = pc - cc;
      int kw = kaiser[dy * 8 + dx];
      int wkw = fxp_mul(weight, kw);
      fxp_acc_add_i32(&numer, fxp_mul(wkw, lblk[li * 64 + dy * 8 + dx]));
      fxp_acc_add_i32(&denom, wkw);
    }
  }
  int idx = pr * width + pc;
  int d = fxp_acc_extract_q20(&denom);
  if(d > 1) { int n = fxp_acc_extract_q20(&numer); STB(output, idx, fxp_mul(n, fxp_recip(d))); }
  else      { STB(output, idx, LDB(noisy, idx)); }
}

/* ----------------------------------------------------------------------------
 * [DEPRECATED] pixel-parallel GATHER kernels — bit-exact oracles for the tiled
 * kernels above (each output pixel recomputes its <=16 covering blocks).
 * -------------------------------------------------------------------------- */
__kernel void k_p5_pass1_gather(__global const lbuf_t *input, __global lbuf_t *pilot,
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

__kernel void k_p5_pass2_gather(__global const lbuf_t *noisy, __global const lbuf_t *pilot,
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
