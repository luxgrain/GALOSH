/* ============================================================================
 *  galosh_int_test.c  — F2 bit-exact validation harness for the INT
 *  primitive port (galosh_int.clh) against the CPU reference
 *  (galosh_cpu_int.h, the r32 reference's primitive layer).
 *
 *  For each primitive it generates a sweep of inputs (edge cases that map to
 *  the known r32 saturation zones + random fill), runs the CPU reference and
 *  the OpenCL kernel, and compares the results BIT-FOR-BIT (int == int).
 *  Any mismatch is a port bug (the math is identical by construction).
 *
 *  Build (ucrt64 gcc; OpenCL headers in C:\msys64\ucrt64\include):
 *    gcc -O2 -std=c11 galosh_int_test.c -o galosh_int_test.exe -lOpenCL
 *  Run:
 *    ./galosh_int_test.exe [device_idx]
 * ========================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

/* --- definitions for the extern table globals declared in galosh_cpu_int.h.
 *     Stage A uses no table-driven functions, but defining these lets the TU
 *     link cleanly regardless of which static-inline bodies the compiler
 *     chooses to emit. --- */
int   fxp_exp_lut_initialized = 0;
int32_t fxp_exp_neg_pow2_table[17];
int   fxp_log_lut_initialized = 0;
int32_t fxp_ln2_q20 = 0;
int32_t fxp_kaiser_2d[64];
int   fxp_kaiser_initialized = 0;
int32_t fxp_factorial_log[400];
int   fxp_factorial_log_initialized = 0;

#include "galosh_cpu_int.h"

/* ------------------------------------------------------------------ */
/* small utilities                                                     */
/* ------------------------------------------------------------------ */
static char *load_file(const char *path, size_t *len_out) {
  FILE *f = fopen(path, "rb");
  if(!f) return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = (char *)malloc(n + 1);
  if(!buf) { fclose(f); return NULL; }
  size_t rd = fread(buf, 1, n, f);
  buf[rd] = '\0';
  fclose(f);
  if(len_out) *len_out = rd;
  return buf;
}

#define CLCHK(err, what) do { \
  if((err) != CL_SUCCESS) { \
    fprintf(stderr, "[CL] %s failed: %d\n", (what), (int)(err)); \
    exit(1); \
  } } while(0)

/* xorshift32 RNG for reproducible inputs */
static uint32_t rng_state = 0x12345678u;
static uint32_t xrnd(void) {
  uint32_t x = rng_state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  rng_state = x;
  return x;
}

/* curated edge values within the multiply-safe range (|raw| < 2^27) that map
 * to the r32 saturation zones. */
static const int32_t EDGES[] = {
  0, 1, -1, 2, -2,
  FXP_ONE, -FXP_ONE, FXP_HALF, -FXP_HALF,
  100, -100,            /* ~9.5e-5 real (recip/div saturation zone) */
  512, -512,            /* ~4.88e-4 real (the fxp_recip / div_q20 boundary) */
  511, 513,
  10000, -10000,        /* ~9.5e-3 real (alpha-ish) */
  1048576*128, -1048576*128,   /* 128 real (large, drives mul saturation) */
  1048576*441,          /* 441 real (GAT-domain max) */
  1 << 26, -(1 << 26),
};
#define N_EDGES ((int)(sizeof(EDGES)/sizeof(EDGES[0])))

/* general values: cycle the edges then random within +/- 2^27 */
static void gen_general(int32_t *a, int n) {
  for(int i = 0; i < n; i++) {
    if(i < N_EDGES) a[i] = EDGES[i];
    else {
      uint32_t r = xrnd();
      int32_t v = (int32_t)(r & 0x0FFFFFFFu);      /* 0 .. 2^28-1 */
      v -= (1 << 27);                               /* +/- 2^27 */
      a[i] = v;
    }
  }
}

/* denominator / recip-x values: biased toward small magnitudes so the
 * saturation boundary (|x| < ~4.88e-4 -> 1/x or a/x overflows Q11.20) is
 * exercised heavily.  Mix of tiny, boundary, normal; both signs; never 0. */
static void gen_small(int32_t *a, int n) {
  static const int32_t small_edges[] = {
    1, -1, 50, -50, 256, -256, 512, -512, 1024, -1024,
    5000, -5000, 100000, FXP_ONE, -FXP_ONE
  };
  const int ne = (int)(sizeof(small_edges)/sizeof(small_edges[0]));
  for(int i = 0; i < n; i++) {
    if(i < ne) { a[i] = small_edges[i]; continue; }
    uint32_t r = xrnd();
    int kind = r & 3;
    int32_t mag;
    if(kind == 0)       mag = 1 + (int32_t)(xrnd() % 2000);        /* tiny: 1..2000 raw */
    else if(kind == 1)  mag = 1 + (int32_t)(xrnd() % 100000);      /* small */
    else                mag = 1 + (int32_t)(xrnd() % (1 << 24));   /* normal */
    a[i] = (r & 4) ? -mag : mag;
  }
}

/* ------------------------------------------------------------------ */
/* OpenCL plumbing                                                     */
/* ------------------------------------------------------------------ */
static cl_context       g_ctx;
static cl_command_queue g_queue;
static cl_program       g_prog;

static cl_mem mk_buf(cl_mem_flags flags, size_t bytes, void *host) {
  cl_int err;
  cl_mem b = clCreateBuffer(g_ctx, flags, bytes, host, &err);
  CLCHK(err, "createBuffer");
  return b;
}

static cl_kernel mk_kernel(const char *name) {
  cl_int err;
  cl_kernel k = clCreateKernel(g_prog, name, &err);
  if(err != CL_SUCCESS) { fprintf(stderr, "[CL] kernel %s: %d\n", name, err); exit(1); }
  return k;
}

/* run an element-wise kernel with `nin` global int input arrays + 1 output,
 * optional trailing scalar int arg (scalar_used). N elements. */
static void run_kernel_ew(const char *name, int32_t **ins, int nin,
                          int32_t *out, int N,
                          int has_scalar, int scalar_val,
                          int elems_per_item /* 1 for ew, 64 for block */) {
  cl_kernel k = mk_kernel(name);
  size_t bytes = (size_t)N * elems_per_item * sizeof(int32_t);
  size_t obytes = (size_t)N * sizeof(int32_t);
  /* for block kernels out is N items (one per block); for ew it's N too */
  if(elems_per_item == 1) obytes = (size_t)N * sizeof(int32_t);

  cl_mem bin[5];
  for(int i = 0; i < nin; i++)
    bin[i] = mk_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, ins[i]);
  cl_mem bout = mk_buf(CL_MEM_WRITE_ONLY, obytes, NULL);

  int ai = 0;
  for(int i = 0; i < nin; i++)
    clSetKernelArg(k, ai++, sizeof(cl_mem), &bin[i]);
  clSetKernelArg(k, ai++, sizeof(cl_mem), &bout);
  if(has_scalar) clSetKernelArg(k, ai++, sizeof(int), &scalar_val);

  size_t gws = N;
  cl_int err = clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &gws, NULL, 0, NULL, NULL);
  CLCHK(err, "enqueueNDRange");
  err = clEnqueueReadBuffer(g_queue, bout, CL_TRUE, 0, obytes, out, 0, NULL, NULL);
  CLCHK(err, "readBuffer");
  clFinish(g_queue);

  for(int i = 0; i < nin; i++) clReleaseMemObject(bin[i]);
  clReleaseMemObject(bout);
  clReleaseKernel(k);
}

/* block kernel: nin block-arrays of N*64, output N*64 (wht) or N (median) */
static void run_kernel_block(const char *name, int32_t *in, int32_t *out,
                             int nblocks, int out_per_block, int scalar_val) {
  cl_kernel k = mk_kernel(name);
  size_t ibytes = (size_t)nblocks * 64 * sizeof(int32_t);
  size_t obytes = (size_t)nblocks * out_per_block * sizeof(int32_t);
  cl_mem bin = mk_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ibytes, in);
  cl_mem bout = mk_buf(CL_MEM_WRITE_ONLY, obytes, NULL);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bin);
  clSetKernelArg(k, 1, sizeof(cl_mem), &bout);
  clSetKernelArg(k, 2, sizeof(int), &scalar_val);
  size_t gws = nblocks;
  cl_int err = clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &gws, NULL, 0, NULL, NULL);
  CLCHK(err, "enqueueNDRange(block)");
  err = clEnqueueReadBuffer(g_queue, bout, CL_TRUE, 0, obytes, out, 0, NULL, NULL);
  CLCHK(err, "readBuffer(block)");
  clFinish(g_queue);
  clReleaseMemObject(bin);
  clReleaseMemObject(bout);
  clReleaseKernel(k);
}

/* runner: 1 global int input + 1 output + 1 __constant buffer (T) */
static void run_ew_T(const char *name, int32_t *in, int32_t *out, int N,
                     cl_mem Tbuf) {
  cl_kernel k = mk_kernel(name);
  size_t bytes = (size_t)N * sizeof(int32_t);
  cl_mem bi = mk_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, in);
  cl_mem bo = mk_buf(CL_MEM_WRITE_ONLY, bytes, NULL);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bi);
  clSetKernelArg(k, 1, sizeof(cl_mem), &bo);
  clSetKernelArg(k, 2, sizeof(cl_mem), &Tbuf);
  size_t gws = N;
  CLCHK(clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &gws, NULL, 0, NULL, NULL), "nd(T)");
  CLCHK(clEnqueueReadBuffer(g_queue, bo, CL_TRUE, 0, bytes, out, 0, NULL, NULL), "rd(T)");
  clFinish(g_queue);
  clReleaseMemObject(bi); clReleaseMemObject(bo); clReleaseKernel(k);
}

/* runner: gat_fwd (in, out, Pbuf, Tbuf) */
static void run_gat_fwd(int32_t *in, int32_t *out, int N, cl_mem Pbuf, cl_mem Tbuf) {
  cl_kernel k = mk_kernel("k_gat_fwd");
  size_t bytes = (size_t)N * sizeof(int32_t);
  cl_mem bi = mk_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, in);
  cl_mem bo = mk_buf(CL_MEM_WRITE_ONLY, bytes, NULL);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bi);
  clSetKernelArg(k, 1, sizeof(cl_mem), &bo);
  clSetKernelArg(k, 2, sizeof(cl_mem), &Pbuf);
  clSetKernelArg(k, 3, sizeof(cl_mem), &Tbuf);
  size_t gws = N;
  CLCHK(clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &gws, NULL, 0, NULL, NULL), "nd(fwd)");
  CLCHK(clEnqueueReadBuffer(g_queue, bo, CL_TRUE, 0, bytes, out, 0, NULL, NULL), "rd(fwd)");
  clFinish(g_queue);
  clReleaseMemObject(bi); clReleaseMemObject(bo); clReleaseKernel(k);
}

/* runner: gat_inv (in, out, Pbuf) */
static void run_gat_inv(int32_t *in, int32_t *out, int N, cl_mem Pbuf) {
  cl_kernel k = mk_kernel("k_gat_inv");
  size_t bytes = (size_t)N * sizeof(int32_t);
  cl_mem bi = mk_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, in);
  cl_mem bo = mk_buf(CL_MEM_WRITE_ONLY, bytes, NULL);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bi);
  clSetKernelArg(k, 1, sizeof(cl_mem), &bo);
  clSetKernelArg(k, 2, sizeof(cl_mem), &Pbuf);
  size_t gws = N;
  CLCHK(clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &gws, NULL, 0, NULL, NULL), "nd(inv)");
  CLCHK(clEnqueueReadBuffer(g_queue, bo, CL_TRUE, 0, bytes, out, 0, NULL, NULL), "rd(inv)");
  clFinish(g_queue);
  clReleaseMemObject(bi); clReleaseMemObject(bo); clReleaseKernel(k);
}

/* runner: poisson (lam, kk, out, Tbuf) */
static void run_poisson(int32_t *lam, int32_t *kk, int32_t *out, int N, cl_mem Tbuf) {
  cl_kernel k = mk_kernel("k_poisson");
  size_t bytes = (size_t)N * sizeof(int32_t);
  cl_mem bl = mk_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, lam);
  cl_mem bk = mk_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, kk);
  cl_mem bo = mk_buf(CL_MEM_WRITE_ONLY, bytes, NULL);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bl);
  clSetKernelArg(k, 1, sizeof(cl_mem), &bk);
  clSetKernelArg(k, 2, sizeof(cl_mem), &bo);
  clSetKernelArg(k, 3, sizeof(cl_mem), &Tbuf);
  size_t gws = N;
  CLCHK(clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &gws, NULL, 0, NULL, NULL), "nd(pois)");
  CLCHK(clEnqueueReadBuffer(g_queue, bo, CL_TRUE, 0, bytes, out, 0, NULL, NULL), "rd(pois)");
  clFinish(g_queue);
  clReleaseMemObject(bl); clReleaseMemObject(bk); clReleaseMemObject(bo); clReleaseKernel(k);
}

/* host mirror of the .clh fxp_tables struct (all int -> identical layout) */
typedef struct { int32_t ln2_q20; int32_t exp_neg_pow2[17]; int32_t factorial_log[400]; } host_fxp_tables;

/* ------------------------------------------------------------------ */
/* comparison + reporting                                              */
/* ------------------------------------------------------------------ */
static int g_total_fail = 0;

static void report(const char *name, int n, const int32_t *cpu,
                   const int32_t *gpu, const int32_t *dbg_a,
                   const int32_t *dbg_b) {
  int nfail = 0, first = -1;
  for(int i = 0; i < n; i++) {
    if(cpu[i] != gpu[i]) { if(first < 0) first = i; nfail++; }
  }
  if(nfail == 0) {
    printf("  [PASS] %-14s  n=%d\n", name, n);
  } else {
    printf("  [FAIL] %-14s  n=%d  mismatch=%d  first@%d", name, n, nfail, first);
    if(dbg_a) printf("  a=%d", dbg_a[first]);
    if(dbg_b) printf("  b=%d", dbg_b[first]);
    printf("  cpu=%d gpu=%d\n", cpu[first], gpu[first]);
    g_total_fail += nfail;
  }
}

int main(int argc, char **argv) {
  int dev_idx = (argc > 1) ? atoi(argv[1]) : 0;

  /* --- enumerate GPU devices across platforms (mirror galosh_raw_gpu.c) --- */
  cl_platform_id platforms[8]; cl_uint n_platforms = 0;
  cl_int err = clGetPlatformIDs(8, platforms, &n_platforms);
  CLCHK(err, "getPlatformIDs");
  cl_device_id all_dev[32]; int n_dev = 0;
  for(cl_uint p = 0; p < n_platforms; p++) {
    cl_device_id devs[8]; cl_uint nd = 0;
    clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 8, devs, &nd);
    for(cl_uint d = 0; d < nd && n_dev < 32; d++) all_dev[n_dev++] = devs[d];
  }
  if(n_dev == 0) { fprintf(stderr, "[CL] no GPU devices\n"); return 1; }
  if(dev_idx >= n_dev) dev_idx = 0;
  cl_device_id device = all_dev[dev_idx];
  char dname[256] = {0};
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dname), dname, NULL);
  printf("[CL] device[%d] = %s\n", dev_idx, dname);

  g_ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CLCHK(err, "createContext");
  g_queue = clCreateCommandQueue(g_ctx, device, 0, &err);
  CLCHK(err, "createQueue");

  /* --- concat galosh_int.clh + galosh_int_tbl.clh + galosh_int_test.cl --- */
  size_t l1 = 0, lt = 0, l2 = 0;
  char *prim = load_file("galosh_int.clh", &l1);
  if(!prim) prim = load_file("standalone/galosh_int.clh", &l1);
  char *tbl = load_file("galosh_int_tbl.clh", &lt);
  if(!tbl) tbl = load_file("standalone/galosh_int_tbl.clh", &lt);
  char *test = load_file("galosh_int_test.cl", &l2);
  if(!test) test = load_file("standalone/galosh_int_test.cl", &l2);
  if(!prim || !tbl || !test) { fprintf(stderr, "cannot load .clh/.cl source\n"); return 1; }
  size_t total = l1 + lt + l2 + 4;
  char *src = (char *)malloc(total);
  snprintf(src, total, "%s\n%s\n%s", prim, tbl, test);

  g_prog = clCreateProgramWithSource(g_ctx, 1, (const char **)&src, NULL, &err);
  CLCHK(err, "createProgram");
  err = clBuildProgram(g_prog, 1, &device, "", NULL, NULL);
  if(err != CL_SUCCESS) {
    char log[16384];
    clGetProgramBuildInfo(g_prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
    fprintf(stderr, "[CL] build failed (%d):\n%s\n", err, log);
    return 1;
  }
  printf("[CL] program built OK\n\n");

  /* --- build CPU tables (exact CPU code) and upload as __constant --- */
  fxp_log_lut_init();        /* fxp_ln2_q20 */
  fxp_exp_lut_init();        /* fxp_exp_neg_pow2_table[] */
  fxp_factorial_log_init();  /* fxp_factorial_log[] (needs ln2) */
  host_fxp_tables HT;
  HT.ln2_q20 = fxp_ln2_q20;
  memcpy(HT.exp_neg_pow2, fxp_exp_neg_pow2_table, sizeof(HT.exp_neg_pow2));
  memcpy(HT.factorial_log, fxp_factorial_log, sizeof(HT.factorial_log));
  cl_mem Tbuf = mk_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(HT), &HT);

  const int N = 8192;
  int32_t *a = malloc(N * sizeof(int32_t));
  int32_t *b = malloc(N * sizeof(int32_t));
  int32_t *c = malloc(N * sizeof(int32_t));
  int32_t *d = malloc(N * sizeof(int32_t));
  int32_t *cpu = malloc(N * sizeof(int32_t));
  int32_t *gpu = malloc(N * sizeof(int32_t));
  int32_t *ins[5];

  printf("=== Stage F1-A primitive bit-exact validation (N=%d) ===\n", N);

  /* fxp_mul */
  rng_state = 0xA1; gen_general(a, N); rng_state = 0xB2; gen_general(b, N);
  for(int i = 0; i < N; i++) cpu[i] = fxp_mul(a[i], b[i]);
  ins[0] = a; ins[1] = b;
  run_kernel_ew("k_mul", ins, 2, gpu, N, 0, 0, 1);
  report("fxp_mul", N, cpu, gpu, a, b);

  /* fxp_div_q20 (small denominators) */
  rng_state = 0xC3; gen_general(a, N); rng_state = 0xD4; gen_small(b, N);
  for(int i = 0; i < N; i++) cpu[i] = fxp_div_q20(a[i], b[i]);
  ins[0] = a; ins[1] = b;
  run_kernel_ew("k_div_q20", ins, 2, gpu, N, 0, 0, 1);
  report("fxp_div_q20", N, cpu, gpu, a, b);

  /* fxp_recip (small x) */
  rng_state = 0xE5; gen_small(a, N);
  for(int i = 0; i < N; i++) cpu[i] = fxp_recip(a[i]);
  ins[0] = a;
  run_kernel_ew("k_recip", ins, 1, gpu, N, 0, 0, 1);
  report("fxp_recip", N, cpu, gpu, a, NULL);

  /* fxp_add_sat */
  rng_state = 0x16; gen_general(a, N); rng_state = 0x27; gen_general(b, N);
  for(int i = 0; i < N; i++) cpu[i] = fxp_add_sat(a[i], b[i]);
  ins[0] = a; ins[1] = b;
  run_kernel_ew("k_add_sat", ins, 2, gpu, N, 0, 0, 1);
  report("fxp_add_sat", N, cpu, gpu, a, b);

  /* fxp_sub_sat */
  rng_state = 0x38; gen_general(a, N); rng_state = 0x49; gen_general(b, N);
  for(int i = 0; i < N; i++) cpu[i] = fxp_sub_sat(a[i], b[i]);
  ins[0] = a; ins[1] = b;
  run_kernel_ew("k_sub_sat", ins, 2, gpu, N, 0, 0, 1);
  report("fxp_sub_sat", N, cpu, gpu, a, b);

  /* fxp_acc_extract_q20: acc=from_int32(a); +=b; extract */
  rng_state = 0x5A; gen_general(a, N); rng_state = 0x6B; gen_general(b, N);
  for(int i = 0; i < N; i++) {
    fxp_acc acc = fxp_acc_from_int32(a[i]);
    fxp_acc_add_i32(&acc, b[i]);
    cpu[i] = fxp_acc_extract_q20(&acc);
  }
  ins[0] = a; ins[1] = b;
  run_kernel_ew("k_extract_q20", ins, 2, gpu, N, 0, 0, 1);
  report("acc_extract_q20", N, cpu, gpu, a, b);

  /* fxp_acc_div_i32: acc=madd(a,b); div_i32(acc,div) */
  rng_state = 0x7C; gen_general(a, N); rng_state = 0x8D; gen_general(b, N);
  rng_state = 0x9E; gen_small(c, N);  /* divisor */
  for(int i = 0; i < N; i++) {
    fxp_acc acc = fxp_acc_zero();
    fxp_acc_madd(&acc, a[i], b[i]);
    cpu[i] = fxp_acc_div_i32(&acc, c[i]);
  }
  ins[0] = a; ins[1] = b; ins[2] = c;
  run_kernel_ew("k_div_i32", ins, 3, gpu, N, 0, 0, 1);
  report("acc_div_i32", N, cpu, gpu, a, c);

  /* fxp_acc_div_acc: n=madd(a,b); m=madd(c,d); div_acc(n,m) */
  rng_state = 0xAF; gen_general(a, N); rng_state = 0xB0; gen_general(b, N);
  rng_state = 0xC1; gen_general(c, N); rng_state = 0xD2; gen_general(d, N);
  for(int i = 0; i < N; i++) {
    fxp_acc n2 = fxp_acc_zero(); fxp_acc_madd(&n2, a[i], b[i]);
    fxp_acc m2 = fxp_acc_zero(); fxp_acc_madd(&m2, c[i], d[i]);
    cpu[i] = fxp_acc_div_acc(&n2, &m2);
  }
  ins[0] = a; ins[1] = b; ins[2] = c; ins[3] = d;
  run_kernel_ew("k_div_acc", ins, 4, gpu, N, 0, 0, 1);
  report("acc_div_acc", N, cpu, gpu, a, b);

  /* fxp_partial_selection_median (blocks of 64, median of first 63) */
  {
    int NB = 512;
    int32_t *blk = malloc(NB * 64 * sizeof(int32_t));
    int32_t *mcpu = malloc(NB * sizeof(int32_t));
    int32_t *mgpu = malloc(NB * sizeof(int32_t));
    rng_state = 0xE3; gen_general(blk, NB * 64);
    for(int g = 0; g < NB; g++)
      mcpu[g] = fxp_partial_selection_median(blk + g * 64, 63);
    run_kernel_block("k_median", blk, mgpu, NB, 1, 63);
    report("median(63)", NB, mcpu, mgpu, NULL, NULL);
    free(blk); free(mcpu); free(mgpu);
  }

  /* fxp_wht2d_8x8 (blocks of 64), normalize=0 and =1 */
  for(int norm = 0; norm <= 1; norm++) {
    int NB = 512;
    int32_t *blk = malloc(NB * 64 * sizeof(int32_t));
    int32_t *wcpu = malloc(NB * 64 * sizeof(int32_t));
    int32_t *wgpu = malloc(NB * 64 * sizeof(int32_t));
    rng_state = 0xF4 + norm; gen_general(blk, NB * 64);
    /* CPU */
    int32_t tmp[64];
    for(int g = 0; g < NB; g++) {
      for(int j = 0; j < 64; j++) tmp[j] = blk[g * 64 + j];
      fxp_wht2d_8x8(tmp, norm);
      for(int j = 0; j < 64; j++) wcpu[g * 64 + j] = tmp[j];
    }
    run_kernel_block("k_wht2d", blk, wgpu, NB, 64, norm);
    char nm[32]; snprintf(nm, sizeof(nm), "wht2d(norm=%d)", norm);
    report(nm, NB * 64, wcpu, wgpu, NULL, NULL);
    free(blk); free(wcpu); free(wgpu);
  }

  /* ============================ Stage F1-B ============================ */
  printf("\n=== Stage F1-B table-driven bit-exact validation ===\n");

  /* fxp_sqrt — small x (exp/log branch) and large x (Newton branch) */
  rng_state = 0x1111; gen_small(a, N);
  for(int i = 0; i < N; i++) cpu[i] = fxp_sqrt(a[i]);
  run_ew_T("k_sqrt", a, gpu, N, Tbuf);
  report("sqrt(small)", N, cpu, gpu, a, NULL);

  rng_state = 0x2222; gen_general(a, N);
  for(int i = 0; i < N; i++) cpu[i] = fxp_sqrt(a[i]);
  run_ew_T("k_sqrt", a, gpu, N, Tbuf);
  report("sqrt(general)", N, cpu, gpu, a, NULL);

  /* fxp_log — small x (tiny + boundary) and general */
  rng_state = 0x3333; gen_small(a, N);
  for(int i = 0; i < N; i++) cpu[i] = fxp_log(a[i]);
  run_ew_T("k_log", a, gpu, N, Tbuf);
  report("log(small)", N, cpu, gpu, a, NULL);

  rng_state = 0x4444; gen_general(a, N);
  for(int i = 0; i < N; i++) cpu[i] = fxp_log(a[i]);
  run_ew_T("k_log", a, gpu, N, Tbuf);
  report("log(general)", N, cpu, gpu, a, NULL);

  /* fxp_exp — domain [-16,0] + edges */
  for(int i = 0; i < N; i++) {
    if(i == 0) a[i] = 0;
    else if(i == 1) a[i] = FXP_EXP_DOMAIN_LO;
    else if(i == 2) a[i] = FXP_EXP_DOMAIN_LO - 1;
    else if(i == 3) a[i] = FXP_ONE;
    else {
      uint32_t r = xrnd();
      int kind = r & 7;
      if(kind == 1)      a[i] =  (int32_t)(xrnd() % (uint32_t)(2 * FXP_ONE));      /* [0,2] */
      else               a[i] = -(int32_t)(xrnd() % (uint32_t)(17 * FXP_ONE));     /* [-17,0] */
    }
  }
  for(int i = 0; i < N; i++) cpu[i] = fxp_exp(a[i]);
  run_ew_T("k_exp", a, gpu, N, Tbuf);
  report("exp(domain)", N, cpu, gpu, a, NULL);

  /* fxp_gat_forward / inverse — two (alpha, sigma^2) parameter sets */
  struct { float al, s2; } gat_sets[2] = { { 1e-3f, 1e-5f }, { 5e-4f, 1e-6f } };
  for(int s = 0; s < 2; s++) {
    fxp_gat_params gp;
    fxp_gat_precompute(&gp, fxp_from_float(gat_sets[s].al), fxp_from_float(gat_sets[s].s2));
    cl_mem Pbuf = mk_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(gp), &gp);

    /* forward: x (signal) in [-0.5, 1.5] real */
    rng_state = 0x5500 + s;
    for(int i = 0; i < N; i++)
      a[i] = (int32_t)(xrnd() % (uint32_t)(2 * FXP_ONE)) - (FXP_ONE / 2);
    for(int i = 0; i < N; i++) cpu[i] = fxp_gat_forward(a[i], &gp);
    run_gat_fwd(a, gpu, N, Pbuf, Tbuf);
    char nm[40]; snprintf(nm, sizeof(nm), "gat_fwd[set%d]", s);
    report(nm, N, cpu, gpu, a, NULL);

    /* inverse: d (GAT domain) in [-10, 450] real */
    rng_state = 0x6600 + s;
    for(int i = 0; i < N; i++)
      a[i] = (int32_t)(xrnd() % (uint32_t)(460 * FXP_ONE)) - (10 * FXP_ONE);
    for(int i = 0; i < N; i++) cpu[i] = fxp_gat_inverse_algebraic(a[i], &gp);
    run_gat_inv(a, gpu, N, Pbuf);
    snprintf(nm, sizeof(nm), "gat_inv[set%d]", s);
    report(nm, N, cpu, gpu, a, NULL);

    clReleaseMemObject(Pbuf);
  }

  /* fxp_poisson_log_pdf — lambda mix (exact + Gauss path), k in valid range */
  rng_state = 0x7777;
  for(int i = 0; i < N; i++) {
    uint32_t r = xrnd();
    if(r & 1) a[i] = (int32_t)(xrnd() % (uint32_t)(100 * FXP_ONE));        /* lambda<100 exact */
    else      a[i] = (int32_t)(100 * FXP_ONE) + (int32_t)(xrnd() % (uint32_t)(1900 * FXP_ONE)); /* >=100 Gauss */
    b[i] = (int32_t)(xrnd() % 250);   /* k in [0,249] */
  }
  for(int i = 0; i < N; i++) cpu[i] = fxp_poisson_log_pdf(a[i], b[i]);
  run_poisson(a, b, gpu, N, Tbuf);
  report("poisson", N, cpu, gpu, a, b);

  printf("\n=== %s ===\n", g_total_fail == 0 ? "ALL PASS" : "FAILURES PRESENT");
  free(a); free(b); free(c); free(d); free(cpu); free(gpu);
  clReleaseMemObject(Tbuf);
  return g_total_fail == 0 ? 0 : 2;
}
