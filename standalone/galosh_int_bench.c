/* ============================================================================
 *  galosh_int_bench.c  — end-to-end SPEED benchmark for the i16 GPU pipeline.
 *
 *  Runs the full GALOSH-RAW INT GPU pipeline P0 -> P10 at an arbitrary
 *  resolution and reports per-frame wall-clock timing (H2D + compute + D2H),
 *  plus one-time costs (context init, program build) and a one-shot per-group
 *  breakdown (the single-WI P0-estimate / dark-ref phases are the known
 *  reduction bottleneck — this attributes their cost).
 *
 *  This is the speed twin of galosh_int_pipe_test.c (which is per-phase
 *  bit-exact validation): same kernel sequence, buffers pre-allocated ONCE,
 *  no intermediate readbacks/file dumps.  Output goes nowhere (speed only).
 *
 *  Build (i32 reference):  gcc -O2 -std=c11 galosh_int_bench.c -o galosh_int_bench.exe -lOpenCL
 *  Build (genuine INT16) :  add  -DGENUINE_I16  AND pass `short` at runtime
 *  Run:   ./galosh_int_bench.exe <in_f32.bin> <w> <h> [reps] [dev] [short]
 * ========================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <windows.h>

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

static double now_ms(void) {
  static LARGE_INTEGER f; if(!f.QuadPart) QueryPerformanceFrequency(&f);
  LARGE_INTEGER c; QueryPerformanceCounter(&c);
  return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}

/* per-group breakdown timing (one-shot): BD(slot) closes the prior group with a
 * clFinish and stamps its wall time.  Disabled (no stalls) during headline timing. */
static int    g_bd = 0;
static double g_t[16] = {0};
static double g_last = 0;

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
#define P2_WG_HOST 256   /* must match P2_WG in galosh_int_p2.cl */
#define P0_WG_HOST 256   /* must match P0_WG in galosh_int_p0.cl */
#define P1_WG_HOST 256   /* must match P1_WG in galosh_int_p1.cl */
static void run1l(cl_kernel k, size_t gws, size_t lws) {
  cl_int e = clEnqueueNDRangeKernel(queue, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
  CLCHK(e, "ndrange_l");
}
static void seti(cl_kernel k, int i, int v) { clSetKernelArg(k, i, sizeof(int), &v); }
static void setm(cl_kernel k, int i, cl_mem *m) { clSetKernelArg(k, i, sizeof(cl_mem), m); }

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

int main(int argc, char **argv) {
  if(argc < 4) { fprintf(stderr, "usage: %s <in_f32.bin> <w> <h> [reps] [dev] [short]\n", argv[0]); return 1; }
  const char *in_path = argv[1];
  int width = atoi(argv[2]), height = atoi(argv[3]);
  int reps = (argc > 4) ? atoi(argv[4]) : 20;
  int dev_idx = (argc > 5) ? atoi(argv[5]) : 0;
  int g_genuine = 0;
  for(int ai = 1; ai < argc; ai++) if(!strcmp(argv[ai], "short")) g_genuine = 1;
  if(reps < 1) reps = 1;
  size_t npix = (size_t)width * height;
  int npix_i = (int)npix;

  size_t flen = 0; float *in_f32 = (float *)load_file(in_path, &flen);
  if(!in_f32 || flen < npix * sizeof(float)) { fprintf(stderr, "bad input (need %zu floats, got %zu)\n", npix, flen / 4); return 1; }
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

  /* ---------- one-time: context + queue ---------- */
  double t_ctx0 = now_ms();
  cl_platform_id pf[8]; cl_uint npf = 0; cl_int err = clGetPlatformIDs(8, pf, &npf); CLCHK(err, "pf");
  cl_device_id dv[32]; int nd = 0;
  for(cl_uint p = 0; p < npf; p++) { cl_device_id d[8]; cl_uint n = 0;
    clGetDeviceIDs(pf[p], CL_DEVICE_TYPE_GPU, 8, d, &n);
    for(cl_uint i = 0; i < n && nd < 32; i++) dv[nd++] = d[i]; }
  if(nd == 0) { fprintf(stderr, "no GPU\n"); return 1; }
  if(dev_idx >= nd) dev_idx = 0;
  cl_device_id device = dv[dev_idx];
  char dev_name[256] = {0}; clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof dev_name, dev_name, NULL);
  ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CLCHK(err, "ctx");
  queue = clCreateCommandQueue(ctx, device, 0, &err); CLCHK(err, "q");
  double t_ctx = now_ms() - t_ctx0;

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
    if(!part) { snprintf(dirbuf, sizeof dirbuf, "standalone/%s", files[i]);
                part = load_file(dirbuf, &l); }
    if(!part) { fprintf(stderr, "cannot load %s\n", files[i]); return 1; }
    memcpy(src + pos, part, l); pos += l; src[pos++] = '\n'; free(part);
  }
  src[pos] = 0;

  /* ---------- one-time: program build (binary cache) ---------- */
  double t_build0 = now_ms();
  const char *build_opts = g_genuine ? "-DGENUINE_I16" : "";
  char cache_path[1024];
  snprintf(cache_path, sizeof cache_path,
           "standalone/galosh_int_pipe.%s.dev%d.clbin",
           g_genuine ? "i16" : "i32", dev_idx);
  long src_mtime = 0;
  for(int i = 0; i < nfiles; i++) {
    char ap[1024]; snprintf(ap, sizeof ap, "standalone/%s", files[i]);
    struct stat st; if(stat(ap, &st) == 0 && (long)st.st_mtime > src_mtime) src_mtime = (long)st.st_mtime;
  }
  int built = 0; struct stat cst;
  if(stat(cache_path, &cst) == 0 && (long)cst.st_mtime >= src_mtime) {
    size_t blen = 0; char *bin = load_file(cache_path, &blen);
    if(bin && blen > 0) {
      cl_int bstat = 0;
      prog = clCreateProgramWithBinary(ctx, 1, &device, &blen,
                                       (const unsigned char **)&bin, &bstat, &err);
      free(bin);
      if(err == CL_SUCCESS && bstat == CL_SUCCESS &&
         clBuildProgram(prog, 1, &device, build_opts, NULL, NULL) == CL_SUCCESS)
        built = 1;
      else if(prog) { clReleaseProgram(prog); prog = NULL; }
    }
  }
  if(!built) {
    prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err); CLCHK(err, "prog");
    err = clBuildProgram(prog, 1, &device, build_opts, NULL, NULL);
    if(err != CL_SUCCESS) { char log[16384];
      clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
      fprintf(stderr, "[CL] build failed (%d):\n%s\n", err, log); return 1; }
    size_t bsz = 0;
    clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof bsz, &bsz, NULL);
    if(bsz > 0) {
      unsigned char *binp = (unsigned char *)malloc(bsz);
      unsigned char *bins[1] = { binp };
      if(clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof bins, bins, NULL) == CL_SUCCESS) {
        FILE *cf = fopen(cache_path, "wb"); if(cf) { fwrite(binp, 1, bsz, cf); fclose(cf); }
      }
      free(binp);
    }
  }
  double t_build = now_ms() - t_build0;

  /* ---------- allocate ALL buffers ONCE (persistent ISP resources) ---------- */
  int halfw = (width + 1) / 2, halfh = (height + 1) / 2;
  int cqw = halfw / 2, cqh = halfh / 2, cew = cqw / 2, ceh = cqh / 2;
  size_t chs = (size_t)halfw * halfh, cqs = (size_t)cqw * cqh, ces = (size_t)cew * ceh;

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
  cl_mem b_T = mkbuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof HT, &HT);
  cl_mem b_gp = mkbuf(CL_MEM_READ_ONLY, sizeof(fxp_gat_params), NULL);
  cl_mem b_gat = mkbuf(CL_MEM_READ_WRITE, npix * 4, NULL);
  cl_mem b_lhall = mkbuf(CL_MEM_READ_WRITE, 4 * 4096 * 4, NULL);
  cl_mem b_sig = mkbuf(CL_MEM_READ_WRITE, 4 * 4, NULL);
  cl_mem b_uni = mkbuf(CL_MEM_READ_WRITE, 4, NULL);
  cl_mem b_inv = mkbuf(CL_MEM_READ_WRITE, 4, NULL);
  cl_mem b_chref = mkbuf(CL_MEM_READ_WRITE, 4 * 4, NULL);
  cl_mem b_lcs = mkbuf(CL_MEM_READ_WRITE, npix * 4, NULL);

  /* P5 */
  fxp_kaiser_init();
  int32_t kaiser[64]; memcpy(kaiser, fxp_kaiser_2d, sizeof kaiser);
  cl_mem b_kai = mkbuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 64 * 4, kaiser);
  int32_t ss = fxp_from_float(1.0f);
  host_p5_consts P5;
  P5.sigma_sq = fxp_mul(ss, ss);
  P5.lambda_max_unorm = fxp_mul(fxp_mul(ss, fxp_from_float(2.8838f)), 8 * FXP_ONE);
  P5.inv_06745 = fxp_from_float(1.0f / 0.6745f);
  P5.sigma_sq_x64 = fxp_mul(ss, ss) * 64;
  P5.wiener_floor = FXP_ONE / 8;
  cl_mem b_P5 = mkbuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof P5, &P5);
  cl_mem b_pilot = mkbuf(CL_MEM_READ_WRITE, npix * 4, NULL);
  cl_mem b_lden = mkbuf(CL_MEM_READ_WRITE, npix * 4, NULL);

  /* P7 tail */
  int eps = fxp_mul(FXP_ONE, FXP_ONE) >> 8;
  int inv2s = 58253;
  int inv2s_k16 = fxp_from_float(2.0f / 9.0f);
  cl_mem c1h = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c2h = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c3h = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL);
  cl_mem lhden = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL);
  cl_mem lq = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL), le = mkbuf(CL_MEM_READ_WRITE, ces*4, NULL);
  cl_mem c1q = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL), c2q = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL), c3q = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL);
  cl_mem c1e = mkbuf(CL_MEM_READ_WRITE, ces*4, NULL), c2e = mkbuf(CL_MEM_READ_WRITE, ces*4, NULL), c3e = mkbuf(CL_MEM_READ_WRITE, ces*4, NULL);
  cl_mem c1lh = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c2lh = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c3lh = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL);
  cl_mem c1lq = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL), c2lq = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL), c3lq = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL);
  cl_mem c1le = mkbuf(CL_MEM_READ_WRITE, ces*4, NULL), c2le = mkbuf(CL_MEM_READ_WRITE, ces*4, NULL), c3le = mkbuf(CL_MEM_READ_WRITE, ces*4, NULL);
  cl_mem c1qup = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c2qup = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c3qup = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL);
  cl_mem c1eq = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL), c2eq = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL), c3eq = mkbuf(CL_MEM_READ_WRITE, cqs*4, NULL);
  cl_mem c1eup = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c2eup = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), c3eup = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL);
  cl_mem h1 = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), h2 = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL), h3 = mkbuf(CL_MEM_READ_WRITE, chs*4, NULL);
  cl_mem lpix = mkbuf(CL_MEM_READ_WRITE, npix*4, NULL);
  cl_mem a1 = mkbuf(CL_MEM_READ_WRITE, npix*4, NULL), a2 = mkbuf(CL_MEM_READ_WRITE, npix*4, NULL), a3 = mkbuf(CL_MEM_READ_WRITE, npix*4, NULL);

  /* P10 LUT */
  cl_mem b_z = mkbuf(CL_MEM_READ_ONLY, 8 * 4, NULL);
  cl_mem b_d = mkbuf(CL_MEM_READ_WRITE, 1024 * 4, NULL);
  cl_mem b_x = mkbuf(CL_MEM_READ_WRITE, 1024 * 4, NULL);
  cl_mem b_lut = mkbuf(CL_MEM_READ_ONLY, sizeof(fxp_gat_inv_table_t), NULL);
  cl_mem b_out = mkbuf(CL_MEM_READ_WRITE, npix * 4, NULL);

  int32_t *out_host = malloc(npix * 4);
  short   *out_short = g_genuine ? malloc(npix * 2) : NULL;

  /* persistent kernels for the fixed-size phases */
  cl_kernel kbs = mkkern("k_p0_block_stats");
  cl_kernel kes = mkkern("k_p0_estimate");
  cl_kernel kf  = mkkern("k_p1_gat_forward");
  cl_kernel ksig= mkkern("k_p1_sigma_ch");
  cl_kernel kuni= mkkern("k_p1_unify");
  cl_kernel knorm=mkkern("k_p1_normalize");
  cl_kernel kdr = mkkern("k_p2_dark_ref");
  cl_kernel ksub= mkkern("k_p2_subtract");
  cl_kernel kl  = mkkern("k_p3_forward_l");
  cl_kernel kch = mkkern("k_p4_chroma_halfres");
  cl_kernel kp1 = mkkern("k_p5_pass1");
  cl_kernel kp2 = mkkern("k_p5_pass2");
  cl_kernel k6  = mkkern("k_p6_l_h_den");
  cl_kernel k6a = mkkern("k_p6_l_pixel");
  cl_kernel k8  = mkkern("k_p8_smoothstep");
  cl_kernel klut= mkkern("k_build_foi_lut");
  cl_kernel k10 = mkkern("k_p10_reconstruct");

  /* ---- one frame: enqueue full P0..P10, blocking readback of output ---- */
  /* (host roundtrips: alpha/sigma read -> gat_precompute -> gp upload;
   *  LUT d/x read -> assemble -> lut upload.  All part of per-frame cost.) */
  #define BD(slot) do { if(g_bd){ clFinish(queue); g_t[slot] += now_ms()-g_last; g_last=now_ms(); } } while(0)
  #define FRAME(DO_INPUT_H2D) do { \
    if(g_bd) g_last = now_ms(); \
    if(DO_INPUT_H2D) clEnqueueWriteBuffer(queue, b_in, CL_FALSE, 0, npix*4, in_q20, 0, NULL, NULL); \
    /* P0 */ \
    setm(kbs,0,&b_in); seti(kbs,1,width); seti(kbs,2,height); seti(kbs,3,n_bx); seti(kbs,4,n_by); \
    seti(kbs,5,HC.var_scale_combined); setm(kbs,6,&b_mean); setm(kbs,7,&b_var); setm(kbs,8,&b_valid); \
    run1(kbs, total_blocks); \
    BD(0); /* P0 block_stats (parallel) */ \
    setm(kes,0,&b_in); seti(kes,1,width); seti(kes,2,height); seti(kes,3,n_bx); seti(kes,4,n_by); \
    setm(kes,5,&b_mean); setm(kes,6,&b_var); setm(kes,7,&b_valid); setm(kes,8,&b_C); \
    setm(kes,9,&b_pre); setm(kes,10,&b_vh); setm(kes,11,&b_th); setm(kes,12,&b_lh); \
    setm(kes,13,&b_a); setm(kes,14,&b_s); \
    run1l(kes, P0_WG_HOST, P0_WG_HOST); \
    int32_t alpha=0, sigma=0; \
    clEnqueueReadBuffer(queue, b_a, CL_TRUE, 0, 4, &alpha, 0, NULL, NULL); \
    clEnqueueReadBuffer(queue, b_s, CL_TRUE, 0, 4, &sigma, 0, NULL, NULL); \
    fxp_gat_params gp; fxp_gat_precompute(&gp, alpha, sigma); \
    clEnqueueWriteBuffer(queue, b_gp, CL_FALSE, 0, sizeof gp, &gp, 0, NULL, NULL); \
    BD(1); /* P0 estimate WLS (SERIAL, gws=1) + alpha/sigma roundtrip */ \
    /* P1 */ \
    setm(kf,0,&b_in); setm(kf,1,&b_gat); seti(kf,2,npix_i); setm(kf,3,&b_gp); setm(kf,4,&b_T); \
    run1(kf, npix); \
    setm(ksig,0,&b_gat); seti(ksig,1,width); seti(ksig,2,height); setm(ksig,3,&b_lhall); \
    seti(ksig,4,inv_1p6521); setm(ksig,5,&b_sig); \
    run1l(ksig, 4 * P1_WG_HOST, P1_WG_HOST); \
    setm(kuni,0,&b_sig); setm(kuni,1,&b_uni); setm(kuni,2,&b_inv); setm(kuni,3,&b_T); \
    run1(kuni, 1); \
    setm(knorm,0,&b_gat); seti(knorm,1,npix_i); setm(knorm,2,&b_inv); \
    run1(knorm, npix); \
    BD(2); /* P1 GAT+sigma+normalize (parallel) */ \
    /* P2 */ \
    setm(kdr,0,&b_in); setm(kdr,1,&b_gat); seti(kdr,2,width); seti(kdr,3,height); \
    seti(kdr,4,alpha); seti(kdr,5,sigma); seti(kdr,6,c_0p05); seti(kdr,7,c_50); \
    seti(kdr,8,achroma); setm(kdr,9,&b_T); setm(kdr,10,&b_chref); \
    run1l(kdr, P2_WG_HOST, P2_WG_HOST); \
    BD(3); /* P2 dark_ref IRLS (parallel, single-WG LDS) */ \
    setm(ksub,0,&b_gat); seti(ksub,1,width); seti(ksub,2,height); setm(ksub,3,&b_chref); \
    run1(ksub, npix); \
    BD(4); /* P2 subtract (parallel) */ \
    /* P3 */ \
    setm(kl,0,&b_gat); setm(kl,1,&b_lcs); seti(kl,2,width); seti(kl,3,height); \
    run1(kl, npix); \
    BD(7); /* P3 forward L (WHT stride-1) */ \
    /* P4 chroma halfres */ \
    setm(kch,0,&b_gat); setm(kch,1,&c1h); setm(kch,2,&c2h); setm(kch,3,&c3h); \
    seti(kch,4,width); seti(kch,5,height); seti(kch,6,halfw); seti(kch,7,halfh); \
    run1(kch, chs); \
    BD(8); /* P4 chroma halfres */ \
    /* P5 (tiled overlap-add: n_tiles work-groups of 64) */ \
    size_t p5_tiles = (size_t)((width + 7) / 8) * ((height + 7) / 8); \
    setm(kp1,0,&b_lcs); setm(kp1,1,&b_pilot); seti(kp1,2,width); seti(kp1,3,height); \
    setm(kp1,4,&b_T); setm(kp1,5,&b_kai); setm(kp1,6,&b_P5); \
    run1l(kp1, p5_tiles * 64, 64); \
    setm(kp2,0,&b_lcs); setm(kp2,1,&b_pilot); setm(kp2,2,&b_lden); \
    seti(kp2,3,width); seti(kp2,4,height); setm(kp2,5,&b_kai); setm(kp2,6,&b_P5); \
    run1l(kp2, p5_tiles * 64, 64); \
    BD(9); /* P5 BayesShrink pilot + Wiener (overlap-add gather) */ \
    /* P6 L_h_den + L_pixel */ \
    setm(k6,0,&b_lden); setm(k6,1,&lhden); seti(k6,2,width); seti(k6,3,height); \
    seti(k6,4,halfw); seti(k6,5,halfh); run1(k6, chs); \
    setm(k6a,0,&b_lden); setm(k6a,1,&lpix); seti(k6a,2,width); seti(k6a,3,height); \
    run1(k6a, npix); \
    BD(10); /* P6 L_pixel + L_h_den */ \
    /* P7 pyramids + LOESS + K16 */ \
    p7_box_down(lhden, lq, halfw, halfh, cqs); \
    p7_box_down(lq, le, cqw, cqh, ces); \
    p7_box_down(c1h, c1q, halfw, halfh, cqs); p7_box_down(c2h, c2q, halfw, halfh, cqs); p7_box_down(c3h, c3q, halfw, halfh, cqs); \
    p7_box_down(c1q, c1e, cqw, cqh, ces); p7_box_down(c2q, c2e, cqw, cqh, ces); p7_box_down(c3q, c3e, cqw, cqh, ces); \
    BD(11); /* P7 box_down pyramids */ \
    p7_loess(lhden, c1h, c2h, c3h, c1lh, c2lh, c3lh, halfw, halfh, eps, inv2s, b_T, chs); \
    p7_loess(lq, c1q, c2q, c3q, c1lq, c2lq, c3lq, cqw, cqh, eps, inv2s, b_T, cqs); \
    p7_loess(le, c1e, c2e, c3e, c1le, c2le, c3le, cew, ceh, eps, inv2s, b_T, ces); \
    BD(12); /* P7 LOESS R=7 (3 scales) */ \
    p7_k16(c1lq, c2lq, c3lq, lhden, c1qup, c2qup, c3qup, cqw, cqh, inv2s_k16, b_T, chs); \
    p7_k16(c1le, c2le, c3le, lq, c1eq, c2eq, c3eq, cew, ceh, inv2s_k16, b_T, cqs); \
    p7_k16(c1eq, c2eq, c3eq, lhden, c1eup, c2eup, c3eup, cqw, cqh, inv2s_k16, b_T, chs); \
    BD(13); /* P7 K16 jinc upsample */ \
    /* P8 smoothstep (chroma_str=1.0) */ \
    { int t_raw = fxp_from_float(1.0f); int t_sq = fxp_mul(t_raw, t_raw); \
      int t8 = fxp_mul(t_sq, 3 * FXP_ONE - (t_raw << 1)); int oneMt = FXP_ONE - t8; \
      setm(k8,0,&c1h); setm(k8,1,&c2h); setm(k8,2,&c3h); \
      setm(k8,3,&c1lh); setm(k8,4,&c2lh); setm(k8,5,&c3lh); \
      setm(k8,6,&h1); setm(k8,7,&h2); setm(k8,8,&h3); \
      seti(k8,9,(int)chs); seti(k8,10,oneMt); seti(k8,11,t8); run1(k8, chs); } \
    BD(14); /* P8 smoothstep blend */ \
    /* P9 K16 to full-res */ \
    p7_k16(h1, h2, h3, lpix, a1, a2, a3, halfw, halfh, inv2s_k16, b_T, npix); \
    BD(5); /* P9 K16 to full-res */ \
    /* P10 LUT build (GPU) + reconstruct */ \
    int32_t z_table[8]; int32_t sqrt2_sigma = fxp_mul(FXP_SQRT2_Q20, gp.sigma_raw); \
    for(int g = 0; g < 8; g++) z_table[g] = fxp_mul(sqrt2_sigma, fxp_gh_nodes_q20[g + 1]); \
    clEnqueueWriteBuffer(queue, b_z, CL_FALSE, 0, 8*4, z_table, 0, NULL, NULL); \
    fxp_gat_inv_table_t lut; \
    if(gp.alpha < fxp_from_float(1e-4f)) { lut.valid = 0; \
      clEnqueueWriteBuffer(queue, b_lut, CL_FALSE, 0, sizeof lut, &lut, 0, NULL, NULL); \
    } else { \
      setm(klut,0,&b_d); setm(klut,1,&b_x); setm(klut,2,&b_gp); setm(klut,3,&b_T); setm(klut,4,&b_z); \
      run1(klut, 1024); \
      clEnqueueReadBuffer(queue, b_d, CL_TRUE, 0, 1024*4, lut.d, 0, NULL, NULL); \
      clEnqueueReadBuffer(queue, b_x, CL_TRUE, 0, 1024*4, lut.x, 0, NULL, NULL); \
      lut.d_min = lut.d[0]; lut.d_max = lut.d[FXP_GAT_INV_TABLE_SIZE - 1]; \
      lut.alpha = gp.alpha; lut.sigma_raw = gp.sigma_raw; \
      lut.y_break = gp.y_break; lut.t_break = gp.t_break; lut.valid = 1; \
      clEnqueueWriteBuffer(queue, b_lut, CL_FALSE, 0, sizeof lut, &lut, 0, NULL, NULL); \
    } \
    int32_t chref[4]={0,0,0,0}, uni=0; \
    clEnqueueReadBuffer(queue, b_chref, CL_TRUE, 0, 16, chref, 0, NULL, NULL); \
    clEnqueueReadBuffer(queue, b_uni, CL_TRUE, 0, 4, &uni, 0, NULL, NULL); \
    setm(k10,0,&a1); setm(k10,1,&a2); setm(k10,2,&a3); setm(k10,3,&lpix); setm(k10,4,&b_out); \
    seti(k10,5,width); seti(k10,6,height); \
    seti(k10,7,chref[0]); seti(k10,8,chref[1]); seti(k10,9,chref[2]); seti(k10,10,chref[3]); \
    seti(k10,11,uni); setm(k10,12,&b_gp); setm(k10,13,&b_lut); \
    run1(k10, npix); \
    if(g_genuine) clEnqueueReadBuffer(queue, b_out, CL_TRUE, 0, npix*2, out_short, 0, NULL, NULL); \
    else          clEnqueueReadBuffer(queue, b_out, CL_TRUE, 0, npix*4, out_host, 0, NULL, NULL); \
    clFinish(queue); \
    BD(6); /* P10 LUT(GPU)+reconstruct + output D2H */ \
  } while(0)

  /* warmup (also forces any lazy kernel JIT) */
  FRAME(1); FRAME(1);

  /* timed loop: per-frame wall-clock (H2D input + full compute + D2H output) */
  double *dt = malloc(reps * sizeof(double));
  double tsum = 0, tmin = 1e30;
  for(int r = 0; r < reps; r++) {
    double t0 = now_ms();
    FRAME(1);
    double d = now_ms() - t0;
    dt[r] = d; tsum += d; if(d < tmin) tmin = d;
  }
  /* median */
  for(int i = 0; i < reps; i++) for(int j = i+1; j < reps; j++)
    if(dt[j] < dt[i]) { double t = dt[i]; dt[i] = dt[j]; dt[j] = t; }
  double tmed = dt[reps/2];
  double tavg = tsum / reps;

  printf("=== GALOSH-RAW INT GPU bench ===\n");
  printf("device      : %s (dev %d)\n", dev_name, dev_idx);
  printf("variant      : %s\n", g_genuine ? "genuine INT16 (short line buffers)" : "INT32 reference");
  printf("resolution   : %dx%d  (%.2f MP)\n", width, height, npix / 1e6);
  printf("reps         : %d\n", reps);
  printf("one-time ctx : %7.2f ms (OpenCL context+queue init)\n", t_ctx);
  printf("one-time bld : %7.2f ms (program build, %s)\n", t_build, built ? "from cache" : "from source");
  printf("per-frame    : median %7.3f ms | min %7.3f ms | mean %7.3f ms\n", tmed, tmin, tavg);
  printf("throughput   : %7.2f fps (median)  |  %.2f MP/s\n", 1000.0 / tmed, npix / 1e6 * (1000.0 / tmed));
  printf("realtime     : 4K60 budget=16.67 ms -> %s ;  30fps budget=33.3 ms -> %s\n",
         tmed <= 16.67 ? "PASS" : "miss", tmed <= 33.3 ? "PASS" : "miss");

  /* ---- one-shot per-group breakdown (clFinish-bounded; attributes the two
   *      single-WI serial reductions vs the pixel-parallel body) ---- */
  g_bd = 1; memset(g_t, 0, sizeof g_t);
  FRAME(1);
  const char *gn[7] = {
    "P0 block_stats (parallel)        ",
    "P0 estimate WLS  [SERIAL gws=1]  ",
    "P1 GAT/sigma/normalize (parallel)",
    "P2 dark_ref IRLS [SERIAL gws=1]  ",
    "P2 subtract (parallel)           ",
    "P3-P9 body (pixel-parallel)      ",
    "P10 LUT(GPU)+reconstruct+D2H     " };
  double g_total = 0; for(int i = 0; i < 16; i++) g_total += g_t[i];  /* all slots */
  /* slot 5 now = P9 only; the P3-P8 body lives in slots 7..14 (printed below). */
  double g_body = g_t[5]; for(int i = 7; i < 15; i++) g_body += g_t[i];
  printf("\n--- per-group breakdown (one frame, clFinish-bounded) ---\n");
  for(int i = 0; i < 7; i++) {
    if(i == 5) { printf("  [5] P3-P9 body (pixel-parallel)      %8.3f ms  (%5.1f%%)\n",
                        g_body, 100.0 * g_body / g_total); continue; }
    printf("  [%d] %s %8.3f ms  (%5.1f%%)\n", i, gn[i], g_t[i], 100.0 * g_t[i] / g_total);
  }
  /* body sub-breakdown (slots 7..14 + 5 = the P3..P9 pixel-parallel tail) */
  const char *bn[9] = {
    "P3 forward L (WHT s1)     ",
    "P4 chroma halfres         ",
    "P5 BayesShrink+Wiener gath",
    "P6 L_pixel + L_h_den      ",
    "P7 box_down pyramids      ",
    "P7 LOESS R=7 (3 scales)   ",
    "P7 K16 jinc upsample      ",
    "P8 smoothstep blend       ",
    "P9 K16 to full-res        " };
  int bsl[9] = { 7, 8, 9, 10, 11, 12, 13, 14, 5 };
  printf("    --- body (P5/P7 = overlap-add gather + LOESS, the hot kernels) ---\n");
  for(int i = 0; i < 9; i++)
    printf("    P3-9[%d] %s %8.3f ms  (%5.1f%%)\n", i, bn[i], g_t[bsl[i]], 100.0 * g_t[bsl[i]] / g_total);
  return 0;
}
