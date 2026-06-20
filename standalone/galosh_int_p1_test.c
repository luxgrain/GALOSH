/* ============================================================================
 *  galosh_int_p1_test.c  — GPU P0+P1 driver for bit-exact validation.
 *
 *  Loads a float32 .bin, runs GPU Phase 0 (alpha/sigma), computes the GAT
 *  params on the host (fxp_gat_precompute, identical to the CPU), runs GPU
 *  Phase 1 (GAT forward + per-CFA sigma + unified_sigma normalize), and writes
 *  the normalized in_gat buffer as RAW int32 to <out_p1.bin>, plus prints
 *  "P1_RAW unified_sigma=.. sigma_ch=..".
 *
 *  Compare against the CPU r32 reference run with GALOSH_INT_RAW_DUMP_DIR set
 *  (writes p1_ingat.bin raw int32 + prints the same P1_RAW line).
 *  Driver: int_p1_gpu_compare.py.
 *
 *  Build: gcc -O2 -std=c11 galosh_int_p1_test.c -o galosh_int_p1_test.exe -lOpenCL
 *  Run:   ./galosh_int_p1_test.exe <in.bin> <w> <h> <out_p1.bin> [device_idx]
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
static void run1(cl_kernel k, size_t gws) {
  cl_int e = clEnqueueNDRangeKernel(queue, k, 1, NULL, &gws, NULL, 0, NULL, NULL);
  CLCHK(e, "ndrange");
}

int main(int argc, char **argv) {
  if(argc < 5) { fprintf(stderr, "usage: %s <in.bin> <w> <h> <out_p1.bin> [dev]\n", argv[0]); return 1; }
  const char *in_path = argv[1];
  int width = atoi(argv[2]), height = atoi(argv[3]);
  const char *out_path = argv[4];
  int dev_idx = (argc > 5) ? atoi(argv[5]) : 0;
  size_t npix = (size_t)width * height;

  size_t flen = 0; float *in_f32 = (float *)load_file(in_path, &flen);
  if(!in_f32 || flen < npix * sizeof(float)) { fprintf(stderr, "bad input\n"); return 1; }
  int32_t *in_q20 = malloc(npix * sizeof(int32_t));
  for(size_t i = 0; i < npix; i++) in_q20[i] = fxp_from_float(in_f32[i]);

  const int n_bx = ((width + 1) / 2) / 8, n_by = ((height + 1) / 2) / 8;
  const int total_blocks = 4 * n_bx * n_by;

  /* host consts + tables (exact CPU code) */
  host_p0_consts HC = {
    fxp_from_float(1.482f * 13.064f), fxp_from_float(0.97f),
    fxp_from_float(1.345f / 0.6745f), fxp_from_float(0.02f), fxp_from_float(1e-4f) };
  fxp_log_lut_init(); fxp_exp_lut_init(); fxp_factorial_log_init();
  host_tables HT; HT.ln2_q20 = fxp_ln2_q20;
  memcpy(HT.exp_neg_pow2, fxp_exp_neg_pow2_table, sizeof(HT.exp_neg_pow2));
  memcpy(HT.factorial_log, fxp_factorial_log, sizeof(HT.factorial_log));
  const int32_t inv_1p6521 = fxp_from_float(1.0f / 1.6521f);

  /* OpenCL */
  cl_platform_id pf[8]; cl_uint npf = 0; cl_int err = clGetPlatformIDs(8, pf, &npf); CLCHK(err, "pf");
  cl_device_id dev[32]; int nd = 0;
  for(cl_uint p = 0; p < npf; p++) { cl_device_id d[8]; cl_uint n = 0;
    clGetDeviceIDs(pf[p], CL_DEVICE_TYPE_GPU, 8, d, &n);
    for(cl_uint i = 0; i < n && nd < 32; i++) dev[nd++] = d[i]; }
  if(nd == 0) { fprintf(stderr, "no GPU\n"); return 1; }
  if(dev_idx >= nd) dev_idx = 0;
  cl_device_id device = dev[dev_idx];
  ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CLCHK(err, "ctx");
  queue = clCreateCommandQueue(ctx, device, 0, &err); CLCHK(err, "q");

  const char *files[] = { "galosh_int.clh", "galosh_int_tbl.clh", "galosh_int_p0.clh",
                          "galosh_int_p1.clh", "galosh_int_p0.cl", "galosh_int_p1.cl" };
  char dirbuf[1024];
  char *src = malloc(1 << 20); src[0] = 0; size_t pos = 0;
  for(int i = 0; i < 6; i++) {
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

  /* buffers */
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

  /* ---- Phase 0 ---- */
  cl_kernel kbs = clCreateKernel(prog, "k_p0_block_stats", &err); CLCHK(err, "kbs");
  int ai = 0;
  clSetKernelArg(kbs, ai++, sizeof(cl_mem), &b_in);
  clSetKernelArg(kbs, ai++, sizeof(int), &width);
  clSetKernelArg(kbs, ai++, sizeof(int), &height);
  clSetKernelArg(kbs, ai++, sizeof(int), &n_bx);
  clSetKernelArg(kbs, ai++, sizeof(int), &n_by);
  clSetKernelArg(kbs, ai++, sizeof(int), &HC.var_scale_combined);
  clSetKernelArg(kbs, ai++, sizeof(cl_mem), &b_mean);
  clSetKernelArg(kbs, ai++, sizeof(cl_mem), &b_var);
  clSetKernelArg(kbs, ai++, sizeof(cl_mem), &b_valid);
  run1(kbs, total_blocks);

  cl_kernel kes = clCreateKernel(prog, "k_p0_estimate", &err); CLCHK(err, "kes");
  ai = 0;
  clSetKernelArg(kes, ai++, sizeof(cl_mem), &b_in);
  clSetKernelArg(kes, ai++, sizeof(int), &width);
  clSetKernelArg(kes, ai++, sizeof(int), &height);
  clSetKernelArg(kes, ai++, sizeof(int), &n_bx);
  clSetKernelArg(kes, ai++, sizeof(int), &n_by);
  clSetKernelArg(kes, ai++, sizeof(cl_mem), &b_mean);
  clSetKernelArg(kes, ai++, sizeof(cl_mem), &b_var);
  clSetKernelArg(kes, ai++, sizeof(cl_mem), &b_valid);
  clSetKernelArg(kes, ai++, sizeof(cl_mem), &b_C);
  clSetKernelArg(kes, ai++, sizeof(cl_mem), &b_pre);
  clSetKernelArg(kes, ai++, sizeof(cl_mem), &b_vh);
  clSetKernelArg(kes, ai++, sizeof(cl_mem), &b_th);
  clSetKernelArg(kes, ai++, sizeof(cl_mem), &b_lh);
  clSetKernelArg(kes, ai++, sizeof(cl_mem), &b_a);
  clSetKernelArg(kes, ai++, sizeof(cl_mem), &b_s);
  run1(kes, 1);

  int32_t alpha = 0, sigma = 0;
  clEnqueueReadBuffer(queue, b_a, CL_TRUE, 0, 4, &alpha, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, b_s, CL_TRUE, 0, 4, &sigma, 0, NULL, NULL);
  clFinish(queue);

  /* ---- GAT params on host (identical to CPU) ---- */
  fxp_gat_params gp; fxp_gat_precompute(&gp, alpha, sigma);
  cl_mem b_gp = mkbuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof gp, &gp);
  cl_mem b_T = mkbuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof HT, &HT);

  /* ---- Phase 1 ---- */
  cl_mem b_gat = mkbuf(CL_MEM_READ_WRITE, npix * 4, NULL);
  cl_mem b_lhall = mkbuf(CL_MEM_READ_WRITE, 4 * 4096 * 4, NULL);
  cl_mem b_sig = mkbuf(CL_MEM_READ_WRITE, 4 * 4, NULL);
  cl_mem b_uni = mkbuf(CL_MEM_READ_WRITE, 4, NULL);
  cl_mem b_inv = mkbuf(CL_MEM_READ_WRITE, 4, NULL);
  int npix_i = (int)npix;

  cl_kernel kf = clCreateKernel(prog, "k_p1_gat_forward", &err); CLCHK(err, "kf");
  ai = 0;
  clSetKernelArg(kf, ai++, sizeof(cl_mem), &b_in);
  clSetKernelArg(kf, ai++, sizeof(cl_mem), &b_gat);
  clSetKernelArg(kf, ai++, sizeof(int), &npix_i);
  clSetKernelArg(kf, ai++, sizeof(cl_mem), &b_gp);
  clSetKernelArg(kf, ai++, sizeof(cl_mem), &b_T);
  run1(kf, npix);

  cl_kernel ksig = clCreateKernel(prog, "k_p1_sigma_ch", &err); CLCHK(err, "ksig");
  ai = 0;
  clSetKernelArg(ksig, ai++, sizeof(cl_mem), &b_gat);
  clSetKernelArg(ksig, ai++, sizeof(int), &width);
  clSetKernelArg(ksig, ai++, sizeof(int), &height);
  clSetKernelArg(ksig, ai++, sizeof(cl_mem), &b_lhall);
  clSetKernelArg(ksig, ai++, sizeof(int), &inv_1p6521);
  clSetKernelArg(ksig, ai++, sizeof(cl_mem), &b_sig);
  run1(ksig, 4);

  cl_kernel kuni = clCreateKernel(prog, "k_p1_unify", &err); CLCHK(err, "kuni");
  ai = 0;
  clSetKernelArg(kuni, ai++, sizeof(cl_mem), &b_sig);
  clSetKernelArg(kuni, ai++, sizeof(cl_mem), &b_uni);
  clSetKernelArg(kuni, ai++, sizeof(cl_mem), &b_inv);
  clSetKernelArg(kuni, ai++, sizeof(cl_mem), &b_T);
  run1(kuni, 1);

  cl_kernel knorm = clCreateKernel(prog, "k_p1_normalize", &err); CLCHK(err, "knorm");
  ai = 0;
  clSetKernelArg(knorm, ai++, sizeof(cl_mem), &b_gat);
  clSetKernelArg(knorm, ai++, sizeof(int), &npix_i);
  clSetKernelArg(knorm, ai++, sizeof(cl_mem), &b_inv);
  run1(knorm, npix);

  int32_t *gat = malloc(npix * 4);
  int32_t unified = 0, sig_ch[4] = {0,0,0,0};
  clEnqueueReadBuffer(queue, b_gat, CL_TRUE, 0, npix * 4, gat, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, b_uni, CL_TRUE, 0, 4, &unified, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, b_sig, CL_TRUE, 0, 16, sig_ch, 0, NULL, NULL);
  clFinish(queue);

  FILE *fo = fopen(out_path, "wb");
  if(fo) { fwrite(gat, 4, npix, fo); fclose(fo); }
  printf("P1_RAW unified_sigma=%d sigma_ch=%d,%d,%d,%d\n",
         unified, sig_ch[0], sig_ch[1], sig_ch[2], sig_ch[3]);
  fprintf(stderr, "[gpu] P0 alpha=%d sigma=%d -> P1 unified=%d\n", alpha, sigma, unified);
  return 0;
}
