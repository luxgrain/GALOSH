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

/* P7 invocation helpers (fresh kernel object per call = safe arg snapshot). */
static cl_kernel mkkern(const char *n);  /* fwd */
static void run1(cl_kernel k, size_t gws);
static void p7_box_down(cl_mem src, cl_mem dst, int sw, int sh, size_t n) {
  cl_kernel k = mkkern("k_p7_box_down");
  setm(k,0,&src); setm(k,1,&dst); seti(k,2,sw); seti(k,3,sh); run1(k,n);
  clReleaseKernel(k);
}
static void p7_loess(cl_mem g, cl_mem i1, cl_mem i2, cl_mem i3,
                     cl_mem o1, cl_mem o2, cl_mem o3, int w, int h,
                     int eps, int inv2s, cl_mem T, size_t n) {
  cl_kernel k = mkkern("k_p7_loess");
  setm(k,0,&g);setm(k,1,&i1);setm(k,2,&i2);setm(k,3,&i3);
  setm(k,4,&o1);setm(k,5,&o2);setm(k,6,&o3);
  seti(k,7,w);seti(k,8,h);seti(k,9,eps);seti(k,10,inv2s);setm(k,11,&T);
  run1(k,n); clReleaseKernel(k);
}
static void p7_k16(cl_mem i1, cl_mem i2, cl_mem i3, cl_mem g,
                   cl_mem o1, cl_mem o2, cl_mem o3, int hw, int hh,
                   int inv2s, cl_mem T, size_t n) {
  cl_kernel k = mkkern("k_p7_k16");
  setm(k,0,&i1);setm(k,1,&i2);setm(k,2,&i3);setm(k,3,&g);
  setm(k,4,&o1);setm(k,5,&o2);setm(k,6,&o3);
  seti(k,7,hw);seti(k,8,hh);seti(k,9,inv2s);setm(k,10,&T);
  run1(k,n); clReleaseKernel(k);
}

/* Foi-exact inverse GAT LUT builder (host) — verbatim port of
 * fxp_gat_build_inverse_table; fills a caller table.  Uploaded to the P10
 * kernel as __constant.  Uses galosh_cpu_int.h primitives. */
static void build_foi_lut(const fxp_gat_params *gat_p, fxp_gat_inv_table_t *tbl) {
  if(gat_p->alpha < fxp_from_float(1e-4f)) { tbl->valid = 0; return; }
  fxp_factorial_log_init();
  fxp32 sqrt2_sigma = fxp_mul(FXP_SQRT2_Q20, gat_p->sigma_raw);
  fxp32 z_table[8];
  for(int g = 0; g < 8; g++) z_table[g] = fxp_mul(sqrt2_sigma, fxp_gh_nodes_q20[g + 1]);
  for(int i = 0; i < FXP_GAT_INV_TABLE_SIZE; i++) {
    fxp_acc xtmp = fxp_acc_zero();
    fxp_acc_madd(&xtmp, i, FXP_ONE);
    fxp32 x_val = fxp_acc_div_i32(&xtmp, FXP_GAT_INV_TABLE_SIZE - 1);
    fxp32 alpha_safe = (gat_p->alpha < 1) ? 1 : gat_p->alpha;
    fxp32 lambda = fxp_div_q20(x_val, alpha_safe);
    int lambda_int = lambda >> FXP_FRAC;
    fxp32 sqrt_lambda = (lambda > 0) ? fxp_sqrt(lambda) : 0;
    int sqrt_lambda_int = sqrt_lambda >> FXP_FRAC;
    int k_max = lambda_int + 6 * sqrt_lambda_int + 10;
    fxp_acc expected_gat = fxp_acc_zero();
    const fxp32 break_thresh = fxp_from_float(-16.0f);
    for(int k = 0; k <= k_max; k++) {
      fxp32 log_prob = fxp_poisson_log_pdf(lambda, k);
      if(log_prob < break_thresh && k > lambda_int + 1) break;
      fxp32 prob = fxp_exp(log_prob);
      if(prob <= 0) continue;
      fxp_acc eg_acc = fxp_acc_zero();
      fxp32 k_alpha = (fxp32)k * gat_p->alpha;
      for(int g = 0; g < 8; g++) {
        fxp32 noisy_y = k_alpha + z_table[g];
        fxp32 T = fxp_gat_forward(noisy_y, gat_p);
        fxp_acc_madd(&eg_acc, fxp_gh_weights_q20[g + 1], T);
      }
      fxp32 eg_q = fxp_mul(FXP_INV_SQRT_PI_Q20, fxp_acc_to_fxp32(&eg_acc));
      fxp_acc_madd(&expected_gat, prob, eg_q);
    }
    tbl->d[i] = fxp_acc_to_fxp32(&expected_gat);
    tbl->x[i] = x_val;
  }
  tbl->d_min = tbl->d[0];
  tbl->d_max = tbl->d[FXP_GAT_INV_TABLE_SIZE - 1];
  tbl->alpha = gat_p->alpha;
  tbl->sigma_raw = gat_p->sigma_raw;
  tbl->y_break = gat_p->y_break;
  tbl->t_break = gat_p->t_break;
  tbl->valid = 1;
}

/* INT16 line-buffer narrowing (requantize a Q11.20 line buffer to its INT16
 * storage precision in place).  luma Q10.5 -> shift 15, chroma Q6.9 -> 11. */
static int g_i16 = 0;       /* requant-sim (INT32 build, rounds in place) */
static int g_genuine = 0;   /* genuine INT16 storage (-DGENUINE_I16 build, short bufs) */
/* CORRECTED i16 line-buffer format (empirical, full-pipeline 2026-06-21):
 * BOTH luma and chroma = Q6.9 (frac 9, +-64).  The earlier design assumed luma
 * Q10.5 (frac 5) on the basis of the 441-max PRE-normalize GAT — but the LINE
 * BUFFERS hold the NORMALISED luma (<=~64), so frac 5 wastes range and is lossy
 * (PSNR(i16,r32) 63-79 dB), while frac 9 is effectively zero-loss (>=~80 dB,
 * mean ~95).  frac 11 (Q4.11 +-16) saturates the luma and is catastrophic. */
static int I16_LUMA_SHIFT = 11;     /* 20 - luma_frac  (Q6.9, frac 9) */
static int I16_CHROMA_SHIFT = 11;   /* 20 - chroma_frac (Q6.9, frac 9) */
static void req(cl_mem buf, size_t n, int store_shift) {
  if(!g_i16 || store_shift <= 0) return;   /* store_shift<=0 (frac>=20) = no narrowing */
  cl_kernel k = mkkern("k_requant_i16");
  setm(k,0,&buf); seti(k,1,(int)n); seti(k,2,store_shift);
  run1(k, n); clReleaseKernel(k);
}

int main(int argc, char **argv) {
  if(argc < 6) { fprintf(stderr, "usage: %s <in.bin> <w> <h> <phase> <out.bin> [dev]\n", argv[0]); return 1; }
  const char *in_path = argv[1];
  int width = atoi(argv[2]), height = atoi(argv[3]);
  int phase = atoi(argv[4]);
  const char *out_path = argv[5];
  int dev_idx = (argc > 6) ? atoi(argv[6]) : 0;
  for(int ai = 1; ai < argc; ai++) {
    if(!strcmp(argv[ai], "i16")) g_i16 = 1;
    else if(!strcmp(argv[ai], "short")) g_genuine = 1;   /* genuine INT16 storage */
    else if(!strncmp(argv[ai], "lf=", 3)) I16_LUMA_SHIFT   = 20 - atoi(argv[ai] + 3);
    else if(!strncmp(argv[ai], "cf=", 3)) I16_CHROMA_SHIFT = 20 - atoi(argv[ai] + 3);
  }
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

  const char *files[] = { "galosh_int.clh", "galosh_int_i16.clh", "galosh_int_tbl.clh",
                          "galosh_int_p0.clh",
                          "galosh_int_p1.clh", "galosh_int_p5.clh", "galosh_int_p7.clh",
                          "galosh_int_p10.clh",
                          "galosh_int_p0.cl", "galosh_int_p1.cl", "galosh_int_p2.cl",
                          "galosh_int_p3.cl", "galosh_int_p4.cl", "galosh_int_p5.cl",
                          "galosh_int_p6.cl", "galosh_int_p7.cl", "galosh_int_p8.cl",
                          "galosh_int_p10.cl", "galosh_int_i16.cl" };
  const int nfiles = 19;
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
  err = clBuildProgram(prog, 1, &device, g_genuine ? "-DGENUINE_I16" : "", NULL, NULL);
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
  req(b_gat, npix, I16_LUMA_SHIFT);    /* in_gat = luma line buffer */

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
    req(b_gat, npix, I16_LUMA_SHIFT);    /* in_gat (post dark-subtract) */
    clEnqueueReadBuffer(queue, b_chref, CL_TRUE, 0, 16, ch_ref, 0, NULL, NULL);
    clFinish(queue);
  }

  /* ---- P3: forward L (stride-1 WHT) -> L_cs ---- */
  cl_mem b_lcs = mkbuf(CL_MEM_READ_WRITE, npix * 4, NULL);
  if(phase >= 3) {
    cl_kernel kl = mkkern("k_p3_forward_l");
    setm(kl,0,&b_gat); setm(kl,1,&b_lcs); seti(kl,2,width); seti(kl,3,height);
    run1(kl, npix);
    req(b_lcs, npix, I16_LUMA_SHIFT);    /* L_cs = luma line buffer */
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
    req(b_pilot, npix, I16_LUMA_SHIFT);    /* pilot = luma line buffer */
    cl_kernel kp2 = mkkern("k_p5_pass2");
    setm(kp2,0,&b_lcs); setm(kp2,1,&b_pilot); setm(kp2,2,&b_lden);
    seti(kp2,3,width); seti(kp2,4,height); setm(kp2,5,&b_kai); setm(kp2,6,&b_P5);
    run1(kp2, npix);
    req(b_lden, npix, I16_LUMA_SHIFT);    /* L_cs_den = luma line buffer */
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

  /* ---- P7..P10 (full chroma tail) ----
   * Prereqs computed inline: chroma @half (P2 in_gat) + L_h_den (P5 L_cs_den).
   * P7 -> 3 half-res estimates; P8 smoothstep; P9 K16 to full-res; P10 inverse. */
  if(phase >= 7) {
    int halfw = (width + 1) / 2, halfh = (height + 1) / 2;
    int cqw = halfw / 2, cqh = halfh / 2, cew = cqw / 2, ceh = cqh / 2;
    size_t chs = (size_t)halfw * halfh, cqs = (size_t)cqw * cqh, ces = (size_t)cew * ceh;
    int eps = fxp_mul(FXP_ONE, FXP_ONE) >> 8;
    int inv2s = 58253;                        /* FXP_LOESS_INV_2SIGMA_SQ */
    int inv2s_k16 = fxp_from_float(2.0f / 9.0f);

    cl_mem c1h = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c2h = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c3h = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL);
    { cl_kernel kc = mkkern("k_p4_chroma_halfres");
      setm(kc,0,&b_gat); setm(kc,1,&c1h); setm(kc,2,&c2h); setm(kc,3,&c3h);
      seti(kc,4,width); seti(kc,5,height); seti(kc,6,halfw); seti(kc,7,halfh);
      run1(kc, chs); clReleaseKernel(kc); }
    req(c1h, chs, I16_CHROMA_SHIFT); req(c2h, chs, I16_CHROMA_SHIFT); req(c3h, chs, I16_CHROMA_SHIFT);
    cl_mem lhden = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL);
    { cl_kernel k6 = mkkern("k_p6_l_h_den");
      setm(k6,0,&b_lden); setm(k6,1,&lhden); seti(k6,2,width); seti(k6,3,height);
      seti(k6,4,halfw); seti(k6,5,halfh); run1(k6, chs); clReleaseKernel(k6); }
    req(lhden, chs, I16_LUMA_SHIFT);

    /* L + C pyramids */
    cl_mem lq = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL), le = mkbuf(CL_MEM_READ_WRITE, ces*4, NULL);
    p7_box_down(lhden, lq, halfw, halfh, cqs);  req(lq, cqs, I16_LUMA_SHIFT);
    p7_box_down(lq, le, cqw, cqh, ces);         req(le, ces, I16_LUMA_SHIFT);
    cl_mem c1q = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL), c2q = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL), c3q = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL);
    p7_box_down(c1h, c1q, halfw, halfh, cqs); p7_box_down(c2h, c2q, halfw, halfh, cqs); p7_box_down(c3h, c3q, halfw, halfh, cqs);
    req(c1q, cqs, I16_CHROMA_SHIFT); req(c2q, cqs, I16_CHROMA_SHIFT); req(c3q, cqs, I16_CHROMA_SHIFT);
    cl_mem c1e = mkbuf(CL_MEM_READ_WRITE, ces*4, NULL), c2e = mkbuf(CL_MEM_READ_WRITE, ces*4, NULL), c3e = mkbuf(CL_MEM_READ_WRITE, ces*4, NULL);
    p7_box_down(c1q, c1e, cqw, cqh, ces); p7_box_down(c2q, c2e, cqw, cqh, ces); p7_box_down(c3q, c3e, cqw, cqh, ces);
    req(c1e, ces, I16_CHROMA_SHIFT); req(c2e, ces, I16_CHROMA_SHIFT); req(c3e, ces, I16_CHROMA_SHIFT);

    /* LOESS @ each scale */
    cl_mem c1lh = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c2lh = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c3lh = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL);
    p7_loess(lhden, c1h, c2h, c3h, c1lh, c2lh, c3lh, halfw, halfh, eps, inv2s, b_T, chs);
    req(c1lh, chs, I16_CHROMA_SHIFT); req(c2lh, chs, I16_CHROMA_SHIFT); req(c3lh, chs, I16_CHROMA_SHIFT);
    cl_mem c1lq = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL), c2lq = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL), c3lq = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL);
    p7_loess(lq, c1q, c2q, c3q, c1lq, c2lq, c3lq, cqw, cqh, eps, inv2s, b_T, cqs);
    req(c1lq, cqs, I16_CHROMA_SHIFT); req(c2lq, cqs, I16_CHROMA_SHIFT); req(c3lq, cqs, I16_CHROMA_SHIFT);
    cl_mem c1le = mkbuf(CL_MEM_READ_WRITE, ces*4, NULL), c2le = mkbuf(CL_MEM_READ_WRITE, ces*4, NULL), c3le = mkbuf(CL_MEM_READ_WRITE, ces*4, NULL);
    p7_loess(le, c1e, c2e, c3e, c1le, c2le, c3le, cew, ceh, eps, inv2s, b_T, ces);
    req(c1le, ces, I16_CHROMA_SHIFT); req(c2le, ces, I16_CHROMA_SHIFT); req(c3le, ces, I16_CHROMA_SHIFT);

    /* K16 upsample: quarter->half, eighth->quarter, (eighth)quarter->half */
    cl_mem c1qup = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c2qup = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c3qup = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL);
    p7_k16(c1lq, c2lq, c3lq, lhden, c1qup, c2qup, c3qup, cqw, cqh, inv2s_k16, b_T, chs);
    req(c1qup, chs, I16_CHROMA_SHIFT); req(c2qup, chs, I16_CHROMA_SHIFT); req(c3qup, chs, I16_CHROMA_SHIFT);
    cl_mem c1eq = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL), c2eq = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL), c3eq = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL);
    p7_k16(c1le, c2le, c3le, lq, c1eq, c2eq, c3eq, cew, ceh, inv2s_k16, b_T, cqs);
    req(c1eq, cqs, I16_CHROMA_SHIFT); req(c2eq, cqs, I16_CHROMA_SHIFT); req(c3eq, cqs, I16_CHROMA_SHIFT);
    cl_mem c1eup = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c2eup = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c3eup = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL);
    p7_k16(c1eq, c2eq, c3eq, lhden, c1eup, c2eup, c3eup, cqw, cqh, inv2s_k16, b_T, chs);
    req(c1eup, chs, I16_CHROMA_SHIFT); req(c2eup, chs, I16_CHROMA_SHIFT); req(c3eup, chs, I16_CHROMA_SHIFT);

    if(phase == 7) {
      cl_mem outs[9] = { c1lh, c2lh, c3lh, c1qup, c2qup, c3qup, c1eup, c2eup, c3eup };
      int32_t *o7 = malloc(chs * 4 * 9);
      for(int k = 0; k < 9; k++)
        clEnqueueReadBuffer(queue, outs[k], CL_TRUE, 0, chs*4, o7 + (size_t)k*chs, 0, NULL, NULL);
      clFinish(queue);
      FILE *fo7 = fopen(out_path, "wb"); if(fo7) { fwrite(o7, 4, chs*9, fo7); fclose(fo7); }
      printf("P7_RAW chs=%d\n", (int)chs);
      return 0;
    }

    /* ---- P8: smoothstep blend (chroma_str=1.0 -> segment 0: A=C_h, B=C_loess_h) ---- */
    int t_raw = fxp_from_float(1.0f);          /* chroma_str = 1.0 -> t_raw = s_q */
    int t_sq = fxp_mul(t_raw, t_raw);
    int t8 = fxp_mul(t_sq, 3 * FXP_ONE - (t_raw << 1));
    int oneMt = FXP_ONE - t8;
    cl_mem h1 = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), h2 = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), h3 = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL);
    { cl_kernel k8 = mkkern("k_p8_smoothstep");
      setm(k8,0,&c1h); setm(k8,1,&c2h); setm(k8,2,&c3h);
      setm(k8,3,&c1lh); setm(k8,4,&c2lh); setm(k8,5,&c3lh);
      setm(k8,6,&h1); setm(k8,7,&h2); setm(k8,8,&h3);
      seti(k8,9,(int)chs); seti(k8,10,oneMt); seti(k8,11,t8);
      run1(k8, chs); clReleaseKernel(k8); }
    req(h1, chs, I16_CHROMA_SHIFT); req(h2, chs, I16_CHROMA_SHIFT); req(h3, chs, I16_CHROMA_SHIFT);
    if(phase == 8) {
      int32_t *o8 = malloc(chs*4*3);
      clEnqueueReadBuffer(queue, h1, CL_TRUE, 0, chs*4, o8, 0, NULL, NULL);
      clEnqueueReadBuffer(queue, h2, CL_TRUE, 0, chs*4, o8 + chs, 0, NULL, NULL);
      clEnqueueReadBuffer(queue, h3, CL_TRUE, 0, chs*4, o8 + 2*chs, 0, NULL, NULL);
      clFinish(queue);
      FILE *fo8 = fopen(out_path, "wb"); if(fo8) { fwrite(o8, 4, chs*3, fo8); fclose(fo8); }
      printf("P8_RAW chs=%d\n", (int)chs);
      return 0;
    }

    /* ---- L_pixel (P6) + P9 K16 upsample to full-res ---- */
    cl_mem lpix = mkbuf(CL_MEM_READ_WRITE, npix*4, NULL);
    { cl_kernel k6a = mkkern("k_p6_l_pixel");
      setm(k6a,0,&b_lden); setm(k6a,1,&lpix); seti(k6a,2,width); seti(k6a,3,height);
      run1(k6a, npix); clReleaseKernel(k6a); }
    req(lpix, npix, I16_LUMA_SHIFT);    /* L_pixel = luma line buffer */
    cl_mem a1 = mkbuf(CL_MEM_READ_WRITE, npix*4, NULL), a2 = mkbuf(CL_MEM_READ_WRITE, npix*4, NULL), a3 = mkbuf(CL_MEM_READ_WRITE, npix*4, NULL);
    p7_k16(h1, h2, h3, lpix, a1, a2, a3, halfw, halfh, inv2s_k16, b_T, npix);
    req(a1, npix, I16_CHROMA_SHIFT); req(a2, npix, I16_CHROMA_SHIFT); req(a3, npix, I16_CHROMA_SHIFT);
    if(phase == 9) {
      int32_t *o9 = malloc(npix*4*3);
      clEnqueueReadBuffer(queue, a1, CL_TRUE, 0, npix*4, o9, 0, NULL, NULL);
      clEnqueueReadBuffer(queue, a2, CL_TRUE, 0, npix*4, o9 + npix, 0, NULL, NULL);
      clEnqueueReadBuffer(queue, a3, CL_TRUE, 0, npix*4, o9 + 2*npix, 0, NULL, NULL);
      clFinish(queue);
      FILE *fo9 = fopen(out_path, "wb"); if(fo9) { fwrite(o9, 4, npix*3, fo9); fclose(fo9); }
      printf("P9_RAW npix=%d\n", (int)npix);
      return 0;
    }

    /* ---- P10: inverse WHT + dark restore + denorm + Foi-exact inverse GAT ---- */
    fxp_gat_inv_table_t lut;
    build_foi_lut(&gp, &lut);
    cl_mem b_lut = mkbuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof lut, &lut);
    int32_t chref[4] = {0,0,0,0};
    clEnqueueReadBuffer(queue, b_chref, CL_TRUE, 0, 16, chref, 0, NULL, NULL);
    int32_t uni = 0;
    clEnqueueReadBuffer(queue, b_uni, CL_TRUE, 0, 4, &uni, 0, NULL, NULL);
    clFinish(queue);
    cl_mem b_out = mkbuf(CL_MEM_READ_WRITE, npix*4, NULL);
    { cl_kernel k10 = mkkern("k_p10_reconstruct");
      setm(k10,0,&a1); setm(k10,1,&a2); setm(k10,2,&a3); setm(k10,3,&lpix); setm(k10,4,&b_out);
      seti(k10,5,width); seti(k10,6,height);
      seti(k10,7,chref[0]); seti(k10,8,chref[1]); seti(k10,9,chref[2]); seti(k10,10,chref[3]);
      seti(k10,11,uni); setm(k10,12,&b_gp); setm(k10,13,&b_lut);
      run1(k10, npix); clReleaseKernel(k10); }
    int32_t *o10 = malloc(npix*4);
    if(g_genuine) {                       /* b_out holds INT16 Q1.14 -> Q11.20 */
      short *s10 = malloc(npix * 2);
      clEnqueueReadBuffer(queue, b_out, CL_TRUE, 0, npix * 2, s10, 0, NULL, NULL);
      clFinish(queue);
      for(size_t i = 0; i < npix; i++) o10[i] = (int)s10[i] << 6;
      free(s10);
    } else {
      clEnqueueReadBuffer(queue, b_out, CL_TRUE, 0, npix*4, o10, 0, NULL, NULL);
      clFinish(queue);
    }
    FILE *fo10 = fopen(out_path, "wb"); if(fo10) { fwrite(o10, 4, npix, fo10); fclose(fo10); }
    printf("P10_RAW npix=%d\n", (int)npix);
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
