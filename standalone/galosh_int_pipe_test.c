/* ============================================================================
 *  galosh_int_pipe_test.c  — GPU pipeline driver for per-phase bit-exact
 *  validation (supersedes galosh_int_p1_test.c going forward).
 *
 *  Runs the GPU i16 pipeline P0 -> P1 -> ... up to <phase>, then writes the
 *  working in_gat buffer at that point as RAW int32 to <out.bin> and prints the
 *  phase's scalar summary ("P<phase>_RAW ...").  Compare against the CPU r32
 *  reference run with GALOSH_INT_RAW_DUMP_DIR set (writes p<phase>_ingat.bin).
 *
 *  Build: gcc -O2 -std=c11 galosh_int_pipe_test.c -o galosh_int_pipe_test.exe -lOpenCL
 *  Run:   ./galosh_int_pipe_test.exe <in.bin> <w> <h> <phase> <out.bin> [dev]
 * ========================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

int   fxp_exp_lut_initialized = 0;
int32_t fxp_exp_neg_pow2_table[17];
int   fxp_log_lut_initialized = 0;
int32_t fxp_ln2_q20 = 0;
int32_t fxp_kaiser_2d[64];
int   fxp_kaiser_initialized = 0;
int32_t fxp_factorial_log[400];
int   fxp_factorial_log_initialized = 0;

#include "galosh_cpu_int.h"

typedef struct { int32_t var_scale_combined, sat_threshold, huber_factor_q,
                         dark_offset_002, alpha_init; } host_p0_consts;
typedef struct { int32_t ln2_q20; int32_t exp_neg_pow2[17]; int32_t factorial_log[400]; } host_tables;
typedef struct { int32_t sigma_sq, lambda_max_unorm, inv_06745, sigma_sq_x64, wiener_floor; } host_p5_consts;

static char *load_file(const char *p, size_t *l) {
  FILE *f = fopen(p, "rb"); if(!f) return NULL;
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  char *b = malloc(n + 1); if(!b) { fclose(f); return NULL; }
  size_t rd = fread(b, 1, n, f); b[rd] = 0; fclose(f); if(l) *l = rd; return b;
}
#define CLCHK(e, w) do { if((e) != CL_SUCCESS) { \
  fprintf(stderr, "[CL] %s failed: %d\n", (w), (int)(e)); exit(1); } } while(0)

static cl_context ctx; static cl_command_queue queue; static cl_program prog;
static cl_mem mkbuf(cl_mem_flags f, size_t bytes, void *host) {
  cl_int e; cl_mem b = clCreateBuffer(ctx, f, bytes, host, &e); CLCHK(e, "buf"); return b;
}
static cl_kernel mkkern(const char *n) {
  cl_int e; cl_kernel k = clCreateKernel(prog, n, &e);
  if(e != CL_SUCCESS) { fprintf(stderr, "kernel %s: %d\n", n, e); exit(1); } return k;
}
static void run1(cl_kernel k, size_t gws) {
  cl_int e = clEnqueueNDRangeKernel(queue, k, 1, NULL, &gws, NULL, 0, NULL, NULL);
  CLCHK(e, "ndrange");
}
static void seti(cl_kernel k, int i, int v) { clSetKernelArg(k, i, sizeof(int), &v); }
static void setm(cl_kernel k, int i, cl_mem *m) { clSetKernelArg(k, i, sizeof(cl_mem), m); }

int main(int argc, char **argv) {
  if(argc < 6) { fprintf(stderr, "usage: %s <in.bin> <w> <h> <phase> <out.bin> [dev]\n", argv[0]); return 1; }
  const char *in_path = argv[1];
  int width = atoi(argv[2]), height = atoi(argv[3]);
  int phase = atoi(argv[4]);
  const char *out_path = argv[5];
  int dev_idx = (argc > 6) ? atoi(argv[6]) : 0;
  size_t npix = (size_t)width * height;
  int npix_i = (int)npix;

  size_t flen = 0; float *in_f32 = (float *)load_file(in_path, &flen);
  if(!in_f32 || flen < npix * sizeof(float)) { fprintf(stderr, "bad input\n"); return 1; }
  int32_t *in_q20 = malloc(npix * sizeof(int32_t));
  for(size_t i = 0; i < npix; i++) in_q20[i] = fxp_from_float(in_f32[i]);

  const int n_bx = ((width + 1) / 2) / 8, n_by = ((height + 1) / 2) / 8;
  const int total_blocks = 4 * n_bx * n_by;

  host_p0_consts HC = {
    fxp_from_float(1.482f * 13.064f), fxp_from_float(0.97f),
    fxp_from_float(1.345f / 0.6745f), fxp_from_float(0.02f), fxp_from_float(1e-4f) };
  fxp_log_lut_init(); fxp_exp_lut_init(); fxp_factorial_log_init();
  host_tables HT; HT.ln2_q20 = fxp_ln2_q20;
  memcpy(HT.exp_neg_pow2, fxp_exp_neg_pow2_table, sizeof(HT.exp_neg_pow2));
  memcpy(HT.factorial_log, fxp_factorial_log, sizeof(HT.factorial_log));
  const int32_t inv_1p6521 = fxp_from_float(1.0f / 1.6521f);
  const int32_t c_0p05 = fxp_from_float(0.05f);
  const int32_t c_50   = fxp_from_float(50.0f);
  const int32_t achroma = fxp_from_float(4.0f);

  cl_platform_id pf[8]; cl_uint npf = 0; cl_int err = clGetPlatformIDs(8, pf, &npf); CLCHK(err, "pf");
  cl_device_id dv[32]; int nd = 0;
  for(cl_uint p = 0; p < npf; p++) { cl_device_id d[8]; cl_uint n = 0;
    clGetDeviceIDs(pf[p], CL_DEVICE_TYPE_GPU, 8, d, &n);
    for(cl_uint i = 0; i < n && nd < 32; i++) dv[nd++] = d[i]; }
  if(nd == 0) { fprintf(stderr, "no GPU\n"); return 1; }
  if(dev_idx >= nd) dev_idx = 0;
  cl_device_id device = dv[dev_idx];
  ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CLCHK(err, "ctx");
  queue = clCreateCommandQueue(ctx, device, 0, &err); CLCHK(err, "q");

  const char *files[] = { "galosh_int.clh", "galosh_int_tbl.clh", "galosh_int_p0.clh",
                          "galosh_int_p1.clh", "galosh_int_p5.clh", "galosh_int_p7.clh",
                          "galosh_int_p0.cl", "galosh_int_p1.cl", "galosh_int_p2.cl",
                          "galosh_int_p3.cl", "galosh_int_p4.cl", "galosh_int_p5.cl",
                          "galosh_int_p6.cl", "galosh_int_p7.cl" };
  const int nfiles = 14;
  char dirbuf[1024];
  char *src = malloc(1 << 20); src[0] = 0; size_t pos = 0;
  for(int i = 0; i < nfiles; i++) {
    size_t l = 0; char *part = load_file(files[i], &l);
    if(!part) { snprintf(dirbuf, sizeof dirbuf, "C:/Users/luxgrain/GALOSH/standalone/%s", files[i]);
                part = load_file(dirbuf, &l); }
    if(!part) { fprintf(stderr, "cannot load %s\n", files[i]); return 1; }
    memcpy(src + pos, part, l); pos += l; src[pos++] = '\n'; free(part);
  }
  src[pos] = 0;
  prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err); CLCHK(err, "prog");
  err = clBuildProgram(prog, 1, &device, "", NULL, NULL);
  if(err != CL_SUCCESS) { char log[16384];
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
    fprintf(stderr, "[CL] build failed (%d):\n%s\n", err, log); return 1; }

  cl_mem b_in = mkbuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, npix * 4, in_q20);
  cl_mem b_mean = mkbuf(CL_MEM_READ_WRITE, total_blocks * 4, NULL);
  cl_mem b_var = mkbuf(CL_MEM_READ_WRITE, total_blocks * 4, NULL);
  cl_mem b_valid = mkbuf(CL_MEM_READ_WRITE, total_blocks * 4, NULL);
  cl_mem b_C = mkbuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof HC, &HC);
  cl_mem b_pre = mkbuf(CL_MEM_READ_WRITE, 256 * 4, NULL);
  cl_mem b_vh = mkbuf(CL_MEM_READ_WRITE, 32 * 128 * 4, NULL);
  cl_mem b_th = mkbuf(CL_MEM_READ_WRITE, 4096 * 4, NULL);
  cl_mem b_lh = mkbuf(CL_MEM_READ_WRITE, 4096 * 4, NULL);
  cl_mem b_a = mkbuf(CL_MEM_READ_WRITE, 4, NULL);
  cl_mem b_s = mkbuf(CL_MEM_READ_WRITE, 4, NULL);

  /* ---- P0 ---- */
  cl_kernel kbs = mkkern("k_p0_block_stats");
  setm(kbs,0,&b_in); seti(kbs,1,width); seti(kbs,2,height); seti(kbs,3,n_bx); seti(kbs,4,n_by);
  seti(kbs,5,HC.var_scale_combined); setm(kbs,6,&b_mean); setm(kbs,7,&b_var); setm(kbs,8,&b_valid);
  run1(kbs, total_blocks);
  cl_kernel kes = mkkern("k_p0_estimate");
  setm(kes,0,&b_in); seti(kes,1,width); seti(kes,2,height); seti(kes,3,n_bx); seti(kes,4,n_by);
  setm(kes,5,&b_mean); setm(kes,6,&b_var); setm(kes,7,&b_valid); setm(kes,8,&b_C);
  setm(kes,9,&b_pre); setm(kes,10,&b_vh); setm(kes,11,&b_th); setm(kes,12,&b_lh);
  setm(kes,13,&b_a); setm(kes,14,&b_s);
  run1(kes, 1);
  int32_t alpha = 0, sigma = 0;
  clEnqueueReadBuffer(queue, b_a, CL_TRUE, 0, 4, &alpha, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, b_s, CL_TRUE, 0, 4, &sigma, 0, NULL, NULL);
  clFinish(queue);

  fxp_gat_params gp; fxp_gat_precompute(&gp, alpha, sigma);
  cl_mem b_gp = mkbuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof gp, &gp);
  cl_mem b_T = mkbuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof HT, &HT);

  /* ---- P1 ---- */
  cl_mem b_gat = mkbuf(CL_MEM_READ_WRITE, npix * 4, NULL);
  cl_mem b_lhall = mkbuf(CL_MEM_READ_WRITE, 4 * 4096 * 4, NULL);
  cl_mem b_sig = mkbuf(CL_MEM_READ_WRITE, 4 * 4, NULL);
  cl_mem b_uni = mkbuf(CL_MEM_READ_WRITE, 4, NULL);
  cl_mem b_inv = mkbuf(CL_MEM_READ_WRITE, 4, NULL);

  cl_kernel kf = mkkern("k_p1_gat_forward");
  setm(kf,0,&b_in); setm(kf,1,&b_gat); seti(kf,2,npix_i); setm(kf,3,&b_gp); setm(kf,4,&b_T);
  run1(kf, npix);
  cl_kernel ksig = mkkern("k_p1_sigma_ch");
  setm(ksig,0,&b_gat); seti(ksig,1,width); seti(ksig,2,height); setm(ksig,3,&b_lhall);
  seti(ksig,4,inv_1p6521); setm(ksig,5,&b_sig);
  run1(ksig, 4);
  cl_kernel kuni = mkkern("k_p1_unify");
  setm(kuni,0,&b_sig); setm(kuni,1,&b_uni); setm(kuni,2,&b_inv); setm(kuni,3,&b_T);
  run1(kuni, 1);
  cl_kernel knorm = mkkern("k_p1_normalize");
  setm(knorm,0,&b_gat); seti(knorm,1,npix_i); setm(knorm,2,&b_inv);
  run1(knorm, npix);

  int32_t unified = 0, sig_ch[4] = {0,0,0,0};
  clEnqueueReadBuffer(queue, b_uni, CL_TRUE, 0, 4, &unified, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, b_sig, CL_TRUE, 0, 16, sig_ch, 0, NULL, NULL);
  clFinish(queue);

  int32_t ch_ref[4] = {0,0,0,0};
  cl_mem b_chref = mkbuf(CL_MEM_READ_WRITE, 4 * 4, NULL);

  /* ---- P2 ---- */
  if(phase >= 2) {
    cl_kernel kdr = mkkern("k_p2_dark_ref");
    setm(kdr,0,&b_in); setm(kdr,1,&b_gat); seti(kdr,2,width); seti(kdr,3,height);
    seti(kdr,4,alpha); seti(kdr,5,sigma); seti(kdr,6,c_0p05); seti(kdr,7,c_50);
    seti(kdr,8,achroma); setm(kdr,9,&b_T); setm(kdr,10,&b_chref);
    run1(kdr, 1);
    cl_kernel ksub = mkkern("k_p2_subtract");
    setm(ksub,0,&b_gat); seti(ksub,1,width); seti(ksub,2,height); setm(ksub,3,&b_chref);
    run1(ksub, npix);
    clEnqueueReadBuffer(queue, b_chref, CL_TRUE, 0, 16, ch_ref, 0, NULL, NULL);
    clFinish(queue);
  }

  /* ---- P3: forward L (stride-1 WHT) -> L_cs ---- */
  cl_mem b_lcs = mkbuf(CL_MEM_READ_WRITE, npix * 4, NULL);
  if(phase >= 3) {
    cl_kernel kl = mkkern("k_p3_forward_l");
    setm(kl,0,&b_gat); setm(kl,1,&b_lcs); seti(kl,2,width); seti(kl,3,height);
    run1(kl, npix);
  }

  /* ---- P4: half-res chroma (parallel; consumes P2 in_gat, not L_cs) ---- */
  if(phase == 4) {
    int halfw = (width + 1) / 2, halfh = (height + 1) / 2;
    size_t chsize = (size_t)halfw * halfh;
    cl_mem b_c1 = mkbuf(CL_MEM_READ_WRITE, chsize * 4, NULL);
    cl_mem b_c2 = mkbuf(CL_MEM_READ_WRITE, chsize * 4, NULL);
    cl_mem b_c3 = mkbuf(CL_MEM_READ_WRITE, chsize * 4, NULL);
    cl_kernel kc = mkkern("k_p4_chroma_halfres");
    setm(kc,0,&b_gat); setm(kc,1,&b_c1); setm(kc,2,&b_c2); setm(kc,3,&b_c3);
    seti(kc,4,width); seti(kc,5,height); seti(kc,6,halfw); seti(kc,7,halfh);
    run1(kc, chsize);
    int32_t *cbuf = malloc(chsize * 4 * 3);
    clEnqueueReadBuffer(queue, b_c1, CL_TRUE, 0, chsize*4, cbuf, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, b_c2, CL_TRUE, 0, chsize*4, cbuf + chsize, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, b_c3, CL_TRUE, 0, chsize*4, cbuf + 2*chsize, 0, NULL, NULL);
    clFinish(queue);
    FILE *fo = fopen(out_path, "wb");
    if(fo) { fwrite(cbuf, 4, chsize * 3, fo); fclose(fo); }
    printf("P4_RAW n=%d\n", (int)chsize);
    fprintf(stderr, "[gpu] alpha=%d sigma=%d uni=%d phase=4\n", alpha, sigma, unified);
    return 0;
  }

  /* ---- P5: BayesShrink pilot + Wiener (pixel-parallel gather) ---- */
  cl_mem b_lden = 0;
  if(phase >= 5) {
    fxp_kaiser_init();
    int32_t kaiser[64]; memcpy(kaiser, fxp_kaiser_2d, sizeof kaiser);
    cl_mem b_kai = mkbuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 64 * 4, kaiser);
    int32_t ss = fxp_from_float(1.0f);   /* luma_str = 1.0 (bench argv[7] default) */
    host_p5_consts P5;
    P5.sigma_sq = fxp_mul(ss, ss);
    P5.lambda_max_unorm = fxp_mul(fxp_mul(ss, fxp_from_float(2.8838f)), 8 * FXP_ONE);
    P5.inv_06745 = fxp_from_float(1.0f / 0.6745f);
    P5.sigma_sq_x64 = fxp_mul(ss, ss) * 64;
    P5.wiener_floor = FXP_ONE / 8;
    cl_mem b_P5 = mkbuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof P5, &P5);
    cl_mem b_pilot = mkbuf(CL_MEM_READ_WRITE, npix * 4, NULL);
    b_lden = mkbuf(CL_MEM_READ_WRITE, npix * 4, NULL);
    cl_kernel kp1 = mkkern("k_p5_pass1");
    setm(kp1,0,&b_lcs); setm(kp1,1,&b_pilot); seti(kp1,2,width); seti(kp1,3,height);
    setm(kp1,4,&b_T); setm(kp1,5,&b_kai); setm(kp1,6,&b_P5);
    run1(kp1, npix);
    cl_kernel kp2 = mkkern("k_p5_pass2");
    setm(kp2,0,&b_lcs); setm(kp2,1,&b_pilot); setm(kp2,2,&b_lden);
    seti(kp2,3,width); seti(kp2,4,height); setm(kp2,5,&b_kai); setm(kp2,6,&b_P5);
    run1(kp2, npix);
    if(phase == 5) {
      int32_t *pbuf = malloc(npix * 4);
      clEnqueueReadBuffer(queue, b_lden, CL_TRUE, 0, npix * 4, pbuf, 0, NULL, NULL);
      clFinish(queue);
      FILE *pf = fopen(out_path, "wb"); if(pf) { fwrite(pbuf, 4, npix, pf); fclose(pf); }
      int32_t lo = 0x7FFFFFFF, hi = (int32_t)0x80000000;
      for(size_t i = 0; i < npix; i++) { if(pbuf[i] < lo) lo = pbuf[i]; if(pbuf[i] > hi) hi = pbuf[i]; }
      printf("P5_RAW lo=%d hi=%d\n", lo, hi);
      return 0;
    }
  }

  /* ---- P6: L_pixel (2x2 overlap-avg) + L_h_den (subsample) from L_cs_den ---- */
  if(phase == 6) {
    int halfw = (width + 1) / 2, halfh = (height + 1) / 2;
    size_t chsize = (size_t)halfw * halfh;
    cl_mem b_lpix = mkbuf(CL_MEM_READ_WRITE, npix * 4, NULL);
    cl_mem b_lhden = mkbuf(CL_MEM_READ_WRITE, chsize * 4, NULL);
    cl_kernel k6a = mkkern("k_p6_l_pixel");
    setm(k6a,0,&b_lden); setm(k6a,1,&b_lpix); seti(k6a,2,width); seti(k6a,3,height);
    run1(k6a, npix);
    cl_kernel k6b = mkkern("k_p6_l_h_den");
    setm(k6b,0,&b_lden); setm(k6b,1,&b_lhden); seti(k6b,2,width); seti(k6b,3,height);
    seti(k6b,4,halfw); seti(k6b,5,halfh);
    run1(k6b, chsize);
    int32_t *c6 = malloc((npix + chsize) * 4);
    clEnqueueReadBuffer(queue, b_lpix, CL_TRUE, 0, npix * 4, c6, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, b_lhden, CL_TRUE, 0, chsize * 4, c6 + npix, 0, NULL, NULL);
    clFinish(queue);
    FILE *fo6 = fopen(out_path, "wb"); if(fo6) { fwrite(c6, 4, npix + chsize, fo6); fclose(fo6); }
    printf("P6_RAW npix=%d chsize=%d\n", (int)npix, (int)chsize);
    return 0;
  }

  /* ---- P7 sub-step: LOESS @ half-res (validate fxp_loess_chroma_3ch_r) ----
   * Computes the prerequisites inline (chroma from P2 in_gat, L_h_den from the
   * P5 L_cs_den) so this stays self-contained without a full pipeline rewire. */
  if(phase == 7) {
    int halfw = (width + 1) / 2, halfh = (height + 1) / 2;
    size_t chsize = (size_t)halfw * halfh;
    cl_mem b_c1h = mkbuf(CL_MEM_READ_WRITE, chsize * 4, NULL);
    cl_mem b_c2h = mkbuf(CL_MEM_READ_WRITE, chsize * 4, NULL);
    cl_mem b_c3h = mkbuf(CL_MEM_READ_WRITE, chsize * 4, NULL);
    cl_kernel kc = mkkern("k_p4_chroma_halfres");
    setm(kc,0,&b_gat); setm(kc,1,&b_c1h); setm(kc,2,&b_c2h); setm(kc,3,&b_c3h);
    seti(kc,4,width); seti(kc,5,height); seti(kc,6,halfw); seti(kc,7,halfh);
    run1(kc, chsize);
    cl_mem b_lhden = mkbuf(CL_MEM_READ_WRITE, chsize * 4, NULL);
    cl_kernel k6b = mkkern("k_p6_l_h_den");
    setm(k6b,0,&b_lden); setm(k6b,1,&b_lhden); seti(k6b,2,width); seti(k6b,3,height);
    seti(k6b,4,halfw); seti(k6b,5,halfh);
    run1(k6b, chsize);
    int32_t eps_gat_scaled = fxp_mul(FXP_ONE, FXP_ONE) >> 8;
    int32_t inv_2sigma_sq = 58253;   /* FXP_LOESS_INV_2SIGMA_SQ */
    cl_mem b_l1 = mkbuf(CL_MEM_READ_WRITE, chsize * 4, NULL);
    cl_mem b_l2 = mkbuf(CL_MEM_READ_WRITE, chsize * 4, NULL);
    cl_mem b_l3 = mkbuf(CL_MEM_READ_WRITE, chsize * 4, NULL);
    cl_kernel kl = mkkern("k_p7_loess");
    setm(kl,0,&b_lhden); setm(kl,1,&b_c1h); setm(kl,2,&b_c2h); setm(kl,3,&b_c3h);
    setm(kl,4,&b_l1); setm(kl,5,&b_l2); setm(kl,6,&b_l3);
    seti(kl,7,halfw); seti(kl,8,halfh); seti(kl,9,eps_gat_scaled);
    seti(kl,10,inv_2sigma_sq); setm(kl,11,&b_T);
    run1(kl, chsize);
    int32_t *c7 = malloc(chsize * 4 * 3);
    clEnqueueReadBuffer(queue, b_l1, CL_TRUE, 0, chsize*4, c7, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, b_l2, CL_TRUE, 0, chsize*4, c7 + chsize, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, b_l3, CL_TRUE, 0, chsize*4, c7 + 2*chsize, 0, NULL, NULL);
    clFinish(queue);
    FILE *fo7 = fopen(out_path, "wb"); if(fo7) { fwrite(c7, 4, chsize * 3, fo7); fclose(fo7); }
    printf("P7_RAW loess_h n=%d\n", (int)chsize);
    return 0;
  }

  /* ---- dump the phase-output buffer (L_cs for phase==3, else in_gat) ---- */
  cl_mem dump_target = (phase >= 3) ? b_lcs : b_gat;
  int32_t *buf = malloc(npix * 4);
  clEnqueueReadBuffer(queue, dump_target, CL_TRUE, 0, npix * 4, buf, 0, NULL, NULL);
  clFinish(queue);
  FILE *fo = fopen(out_path, "wb");
  if(fo) { fwrite(buf, 4, npix, fo); fclose(fo); }

  if(phase == 1)
    printf("P1_RAW unified_sigma=%d sigma_ch=%d,%d,%d,%d\n",
           unified, sig_ch[0], sig_ch[1], sig_ch[2], sig_ch[3]);
  else if(phase == 2)
    printf("P2_RAW ch_dark_ref=%d,%d,%d,%d\n", ch_ref[0], ch_ref[1], ch_ref[2], ch_ref[3]);
  else if(phase == 3) {
    int32_t lo = 0x7FFFFFFF, hi = (int32_t)0x80000000;
    for(size_t i = 0; i < npix; i++) { if(buf[i] < lo) lo = buf[i]; if(buf[i] > hi) hi = buf[i]; }
    printf("P3_RAW lo=%d hi=%d\n", lo, hi);
  }
  fprintf(stderr, "[gpu] alpha=%d sigma=%d uni=%d phase=%d\n", alpha, sigma, unified, phase);
  return 0;
}
