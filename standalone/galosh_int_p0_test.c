/* ============================================================================
 *  galosh_int_p0_test.c  — GPU Phase-0 driver for bit-exact validation.
 *
 *  Loads a float32 .bin image (same format the bench harness feeds the CPU
 *  exe), converts to Q11.20 with the EXACT CPU fxp_from_float, runs the GPU
 *  Phase-0 kernels (k_p0_block_stats + k_p0_estimate), and prints the raw
 *  Q11.20 alpha / sigma_sq.  Compare against galosh_raw_cpu_int.exe's
 *  "(Q11.20: A, S)" stderr line (see int_p0_gpu_compare.py).
 *
 *  Build: gcc -O2 -std=c11 galosh_int_p0_test.c -o galosh_int_p0_test.exe -lOpenCL
 *  Run:   ./galosh_int_p0_test.exe <input.bin> <width> <height> [device_idx]
 * ========================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

/* extern table globals for galosh_cpu_int.h (only fxp_from_float used here). */
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

static char *load_file(const char *path, size_t *len_out) {
  FILE *f = fopen(path, "rb");
  if(!f) return NULL;
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  char *buf = (char *)malloc(n + 1);
  if(!buf) { fclose(f); return NULL; }
  size_t rd = fread(buf, 1, n, f); buf[rd] = '\0'; fclose(f);
  if(len_out) *len_out = rd;
  return buf;
}

#define CLCHK(err, what) do { if((err) != CL_SUCCESS) { \
  fprintf(stderr, "[CL] %s failed: %d\n", (what), (int)(err)); exit(1); } } while(0)

int main(int argc, char **argv) {
  if(argc < 4) {
    fprintf(stderr, "usage: %s <input.bin> <width> <height> [device_idx]\n", argv[0]);
    return 1;
  }
  const char *in_path = argv[1];
  int width  = atoi(argv[2]);
  int height = atoi(argv[3]);
  int dev_idx = (argc > 4) ? atoi(argv[4]) : 0;
  size_t npix = (size_t)width * height;

  /* --- load float32 input, convert to Q11.20 (exact CPU fxp_from_float) --- */
  size_t flen = 0;
  float *in_f32 = (float *)load_file(in_path, &flen);
  if(!in_f32 || flen < npix * sizeof(float)) {
    fprintf(stderr, "cannot load %s (need %zu floats)\n", in_path, npix); return 1;
  }
  int32_t *in_q20 = (int32_t *)malloc(npix * sizeof(int32_t));
  for(size_t i = 0; i < npix; i++) in_q20[i] = fxp_from_float(in_f32[i]);

  const int halfwidth  = (width  + 1) / 2;
  const int halfheight = (height + 1) / 2;
  const int n_bx = halfwidth  / 8;
  const int n_by = halfheight / 8;
  const int total_blocks = 4 * n_bx * n_by;
  if(total_blocks < 100) {
    printf("P0_RAW alpha=%d sigma_sq=%d\n", fxp_from_float(1e-4f), 1);
    return 0;
  }

  /* --- host-computed float-derived constants (ISP parameter registers) --- */
  host_p0_consts HC;
  HC.var_scale_combined = fxp_from_float(1.482f * 13.064f);
  HC.sat_threshold      = fxp_from_float(0.97f);
  HC.huber_factor_q     = fxp_from_float(1.345f / 0.6745f);
  HC.dark_offset_002    = fxp_from_float(0.02f);
  HC.alpha_init         = fxp_from_float(1e-4f);

  /* --- OpenCL setup --- */
  cl_platform_id platforms[8]; cl_uint n_pl = 0;
  cl_int err = clGetPlatformIDs(8, platforms, &n_pl); CLCHK(err, "getPlatformIDs");
  cl_device_id all_dev[32]; int n_dev = 0;
  for(cl_uint p = 0; p < n_pl; p++) {
    cl_device_id devs[8]; cl_uint nd = 0;
    clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 8, devs, &nd);
    for(cl_uint d = 0; d < nd && n_dev < 32; d++) all_dev[n_dev++] = devs[d];
  }
  if(n_dev == 0) { fprintf(stderr, "[CL] no GPU\n"); return 1; }
  if(dev_idx >= n_dev) dev_idx = 0;
  cl_device_id device = all_dev[dev_idx];
  char dname[256] = {0};
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dname), dname, NULL);
  fprintf(stderr, "[CL] device[%d] = %s\n", dev_idx, dname);

  cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CLCHK(err, "ctx");
  cl_command_queue q = clCreateCommandQueue(ctx, device, 0, &err); CLCHK(err, "queue");

  const char *paths[] = { "galosh_int.clh", "galosh_int_tbl.clh",
                          "galosh_int_p0.clh", "galosh_int_p0.cl" };
  const char *fb[] = {
    "C:/Users/luxgrain/GALOSH/standalone/galosh_int.clh",
    "C:/Users/luxgrain/GALOSH/standalone/galosh_int_tbl.clh",
    "C:/Users/luxgrain/GALOSH/standalone/galosh_int_p0.clh",
    "C:/Users/luxgrain/GALOSH/standalone/galosh_int_p0.cl" };
  size_t cap = 1, lens[4]; char *parts[4];
  for(int i = 0; i < 4; i++) {
    parts[i] = load_file(paths[i], &lens[i]);
    if(!parts[i]) parts[i] = load_file(fb[i], &lens[i]);
    if(!parts[i]) { fprintf(stderr, "cannot load %s\n", paths[i]); return 1; }
    cap += lens[i] + 2;
  }
  char *src = (char *)malloc(cap); src[0] = '\0';
  for(int i = 0; i < 4; i++) { strcat(src, parts[i]); strcat(src, "\n"); }

  cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err);
  CLCHK(err, "createProgram");
  err = clBuildProgram(prog, 1, &device, "", NULL, NULL);
  if(err != CL_SUCCESS) {
    char log[16384];
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
    fprintf(stderr, "[CL] build failed (%d):\n%s\n", err, log);
    return 1;
  }

  /* --- buffers --- */
  cl_mem b_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               npix * sizeof(int32_t), in_q20, &err); CLCHK(err, "b_in");
  cl_mem b_mean  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_blocks * sizeof(int32_t), NULL, &err);
  cl_mem b_var   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_blocks * sizeof(int32_t), NULL, &err);
  cl_mem b_valid = clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_blocks * sizeof(int32_t), NULL, &err);
  cl_mem b_C     = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(HC), &HC, &err);
  cl_mem b_pre   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 256 * sizeof(int32_t), NULL, &err);
  cl_mem b_vh    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 32 * 128 * sizeof(int32_t), NULL, &err);
  cl_mem b_th    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4096 * sizeof(int32_t), NULL, &err);
  cl_mem b_lh    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4096 * sizeof(int32_t), NULL, &err);
  cl_mem b_a     = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int32_t), NULL, &err);
  cl_mem b_s     = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int32_t), NULL, &err);

  /* --- k_p0_block_stats --- */
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
  size_t gws_bs = total_blocks;
  err = clEnqueueNDRangeKernel(q, kbs, 1, NULL, &gws_bs, NULL, 0, NULL, NULL);
  CLCHK(err, "nd block_stats");

  /* --- k_p0_estimate (single workgroup of P0_WG threads — the optimized kernel
   *     has reqd_work_group_size(P0_WG), so launch global=local=P0_WG) --- */
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
  size_t gws_e = 256, lws_e = 256;   /* = P0_WG in galosh_int_p0.cl */
  err = clEnqueueNDRangeKernel(q, kes, 1, NULL, &gws_e, &lws_e, 0, NULL, NULL);
  CLCHK(err, "nd estimate");

  int32_t alpha = 0, sigma = 0;
  clEnqueueReadBuffer(q, b_a, CL_TRUE, 0, sizeof(int32_t), &alpha, 0, NULL, NULL);
  clEnqueueReadBuffer(q, b_s, CL_TRUE, 0, sizeof(int32_t), &sigma, 0, NULL, NULL);
  clFinish(q);

  printf("P0_RAW alpha=%d sigma_sq=%d\n", alpha, sigma);
  return 0;
}
