/* galosh_single_gpu.c  --  Single-plane GPU debug driver.
 *
 * Generic single-channel float32 plane denoiser running entirely on GPU.
 * Originally factored out for "GPU YUV via 3 single-plane denoise calls"
 * approach in bench_sidd.py.  Now an archived debug binary -- the proper
 * YUV pipeline is galosh_yuv_gpu.exe and the proper RAW pipeline is
 * galosh_raw_gpu.exe.
 *
 * Pipeline:
 *   K_SP1: sigma histogram (Laplacian MAD, single channel)
 *   K_SP2: sigma finalize (histogram median -> sigma, 1/sigma)
 *   K_SP3: normalize (float32 / sigma -> half)
 *   K13:   fused_pass12 (tiled WHT BayesShrink + Wiener)
 *   K_SP4: denormalize (half x sigma -> float32)
 *
 * Usage: galosh_single_gpu <in.bin> <out.bin> <W> <H> <strength> [cl_dev]
 *
 * Build: gcc -O3 -march=native -o galosh_single_gpu.exe galosh_single_gpu.c \
 *            -lOpenCL -lm
 */

#include "galosh_gpu.h"


/* ================================================================
 * Single-plane mode: denoise a single float32 plane entirely on GPU.
 *
 * All-GPU pipeline (CPU does file I/O only):
 *   GPU K_SP1: σ histogram (Laplacian MAD, single channel)
 *   GPU K_SP2: σ finalize (histogram median → σ, 1/σ)
 *   GPU K_SP3: normalize (float32 / σ → half)
 *   GPU K13:   fused_pass12 (tiled WHT BayesShrink + Wiener)
 *   GPU K_SP4: denormalize (half × σ → float32)
 *
 * Used for YUV GALOSH GPU: Python handles RGB↔YCbCr, calls this per plane.
 * ================================================================ */
static int run_single_plane_gpu(const char *input_file, const char *output_file,
                                int width, int height, float strength,
                                int cl_device_idx)
{
    const size_t npix = (size_t)width * height;
    int ret = 1;

    /* Read input */
    float *plane = alloc_float(npix);
    {
        FILE *f = fopen(input_file, "rb");
        if(!f) { fprintf(stderr, "Cannot open %s\n", input_file); return 1; }
        fread(plane, sizeof(float), npix, f);
        fclose(f);
    }

    double t0 = get_time_ms();

    /* OpenCL setup */
    cl_int err;
    cl_platform_id platforms[8]; cl_uint n_plat = 0;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program prog = NULL;
    cl_kernel k_sigma_hist = NULL, k_sigma_final = NULL;
    cl_kernel k_normalize = NULL, k_denormalize = NULL;
    cl_kernel k_fused = NULL;
    cl_mem plane_buf = NULL, norm_buf = NULL, out_buf = NULL;
    cl_mem hist_buf = NULL, sigma_buf = NULL, result_buf = NULL;

    err = clGetPlatformIDs(8, platforms, &n_plat);
    if(err != CL_SUCCESS) { fprintf(stderr, "[CL] no platforms\n"); goto sp_cleanup; }

    cl_device_id all_devs[32]; int n_dev = 0;
    for(cl_uint p = 0; p < n_plat; p++) {
        cl_device_id devs[8]; cl_uint nd = 0;
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 8, devs, &nd);
        for(cl_uint d = 0; d < nd && n_dev < 32; d++)
            all_devs[n_dev++] = devs[d];
    }
    if(cl_device_idx >= n_dev) { fprintf(stderr, "[CL] device %d not found\n", cl_device_idx); goto sp_cleanup; }

    device = all_devs[cl_device_idx];
    {
        char name[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
        fprintf(stderr, "[SINGLE] device: %s\n", name);
    }
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if(err) { fprintf(stderr, "[CL] context err=%d\n", err); goto sp_cleanup; }
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if(err) { fprintf(stderr, "[CL] queue err=%d\n", err); goto sp_cleanup; }

    /* Load & build kernel source */
    {
        const char *cl_paths[] = {"galosh.cl",
            "C:/Users/luxgrain/GALOSH/standalone/galosh.cl", NULL};
        char *source = NULL; size_t src_len = 0;
        for(int i = 0; cl_paths[i]; i++) {
            source = load_kernel_source(cl_paths[i], &src_len);
            if(source) break;
        }
        if(!source) { fprintf(stderr, "[CL] Cannot find .cl\n"); goto sp_cleanup; }

        char opts[256];
        snprintf(opts, sizeof(opts),
                 "-DGALOSH_STRIDE=%d -DTILE_SIZE=%d -DHIST_BINS=%d -DREDUCE_WG_SIZE=%d",
                 GALOSH_STRIDE, TILE_SIZE, HIST_BINS, REDUCE_WG_SIZE);
        prog = clCreateProgramWithSource(context, 1, (const char **)&source, &src_len, &err);
        free(source);
        if(err) { fprintf(stderr, "[CL] program err=%d\n", err); goto sp_cleanup; }
        err = clBuildProgram(prog, 1, &device, opts, NULL, NULL);
        if(err) {
            char log[4096]; clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
            fprintf(stderr, "[CL] build err:\n%s\n", log); goto sp_cleanup;
        }
    }

    /* Create kernels */
    k_sigma_hist  = clCreateKernel(prog, "galosh_sigma_histogram_single", &err);
    if(err) { fprintf(stderr, "[CL] kernel sigma_hist err=%d\n", err); goto sp_cleanup; }
    k_sigma_final = clCreateKernel(prog, "galosh_sigma_finalize_single", &err);
    if(err) { fprintf(stderr, "[CL] kernel sigma_final err=%d\n", err); goto sp_cleanup; }
    k_normalize   = clCreateKernel(prog, "galosh_normalize_single", &err);
    if(err) { fprintf(stderr, "[CL] kernel normalize err=%d\n", err); goto sp_cleanup; }
    k_denormalize = clCreateKernel(prog, "galosh_denormalize_single", &err);
    if(err) { fprintf(stderr, "[CL] kernel denormalize err=%d\n", err); goto sp_cleanup; }
    k_fused       = clCreateKernel(prog, "galosh_fused_pass12", &err);
    if(err) { fprintf(stderr, "[CL] kernel fused err=%d\n", err); goto sp_cleanup; }

    /* Allocate GPU buffers */
    plane_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, npix * sizeof(float), NULL, &err);
    norm_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, npix * sizeof(cl_half), NULL, &err);
    out_buf    = clCreateBuffer(context, CL_MEM_READ_WRITE, npix * sizeof(cl_half), NULL, &err);
    result_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, npix * sizeof(float), NULL, &err);
    hist_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, HIST_BINS * sizeof(int), NULL, &err);
    sigma_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * sizeof(float), NULL, &err);
    if(err) { fprintf(stderr, "[CL] buffer err=%d\n", err); goto sp_cleanup; }

    /* Upload input */
    clEnqueueWriteBuffer(queue, plane_buf, CL_FALSE, 0,
                         npix * sizeof(float), plane, 0, NULL, NULL);

    /* Zero histogram and output accumulator */
    {
        int izero = 0;
        clEnqueueFillBuffer(queue, hist_buf, &izero, sizeof(int), 0,
                            HIST_BINS * sizeof(int), 0, NULL, NULL);
        cl_half hzero = 0;
        clEnqueueFillBuffer(queue, out_buf, &hzero, sizeof(cl_half), 0,
                            npix * sizeof(cl_half), 0, NULL, NULL);
    }

    double t_gpu_start = get_time_ms();

    /* K_SP1: Sigma histogram */
    {
        const int n_x = (width - 2) / 3 + 1;
        const int n_samp = n_x * height;
        clSetKernelArg(k_sigma_hist, 0, sizeof(cl_mem), &plane_buf);
        clSetKernelArg(k_sigma_hist, 1, sizeof(cl_mem), &hist_buf);
        clSetKernelArg(k_sigma_hist, 2, sizeof(int), &width);
        clSetKernelArg(k_sigma_hist, 3, sizeof(int), &height);
        dispatch_1d_named(queue, k_sigma_hist, align_up(n_samp, 256), 256, "K_SP1 sigma_hist");

        /* K_SP2: Sigma finalize */
        clSetKernelArg(k_sigma_final, 0, sizeof(cl_mem), &hist_buf);
        clSetKernelArg(k_sigma_final, 1, sizeof(cl_mem), &sigma_buf);
        clSetKernelArg(k_sigma_final, 2, sizeof(int), &n_samp);
        dispatch_1d_named(queue, k_sigma_final, 1, 0, "K_SP2 sigma_final");
    }

    /* K_SP3: Normalize float32 → half */
    {
        const int npix_i = (int)npix;
        clSetKernelArg(k_normalize, 0, sizeof(cl_mem), &plane_buf);
        clSetKernelArg(k_normalize, 1, sizeof(cl_mem), &norm_buf);
        clSetKernelArg(k_normalize, 2, sizeof(cl_mem), &sigma_buf);
        clSetKernelArg(k_normalize, 3, sizeof(int), &npix_i);
        dispatch_1d_named(queue, k_normalize, align_up(npix, 256), 256, "K_SP3 normalize");
    }

    /* K13: Fused Pass1+Pass2 */
    clSetKernelArg(k_fused, 0, sizeof(cl_mem), &norm_buf);
    clSetKernelArg(k_fused, 1, sizeof(cl_mem), &out_buf);
    clSetKernelArg(k_fused, 2, sizeof(int), &width);
    clSetKernelArg(k_fused, 3, sizeof(int), &height);
    clSetKernelArg(k_fused, 4, sizeof(float), &strength);
    dispatch_tile_named(queue, k_fused, width, height, "K13 fused_pass12");

    /* K_SP4: Denormalize half → float32 */
    {
        const int npix_i = (int)npix;
        clSetKernelArg(k_denormalize, 0, sizeof(cl_mem), &out_buf);
        clSetKernelArg(k_denormalize, 1, sizeof(cl_mem), &result_buf);
        clSetKernelArg(k_denormalize, 2, sizeof(cl_mem), &sigma_buf);
        clSetKernelArg(k_denormalize, 3, sizeof(int), &npix_i);
        dispatch_1d_named(queue, k_denormalize, align_up(npix, 256), 256, "K_SP4 denormalize");
    }

    clFinish(queue);
    double t_gpu = get_time_ms() - t_gpu_start;

    /* Read back sigma for logging */
    {
        float sp[2];
        clEnqueueReadBuffer(queue, sigma_buf, CL_TRUE, 0, 2 * sizeof(float), sp, 0, NULL, NULL);
        fprintf(stderr, "[SINGLE] sigma=%.6f (GPU Laplacian MAD)\n", sp[0]);
    }

    /* Download denoised float32 output */
    clEnqueueReadBuffer(queue, result_buf, CL_TRUE, 0,
                        npix * sizeof(float), plane, 0, NULL, NULL);

    /* Write output */
    {
        FILE *f = fopen(output_file, "wb");
        if(!f) { fprintf(stderr, "Cannot write %s\n", output_file); goto sp_cleanup; }
        fwrite(plane, sizeof(float), npix, f);
        fclose(f);
    }

    /* Profiling report */
    {
        fprintf(stderr, "\n[SINGLE] ====== PER-KERNEL PROFILING ======\n");
        double total_gpu = 0.0;
        for(int i = 0; i < prof_count; i++) {
            double ms = event_ms(prof_events[i]);
            total_gpu += ms;
            const char *name = prof_names[i] ? prof_names[i] : "(unnamed)";
            fprintf(stderr, "[SINGLE]   %-20s %7.3f ms\n", name, ms);
            clReleaseEvent(prof_events[i]);
        }
        fprintf(stderr, "[SINGLE]   %-20s %7.3f ms\n", "TOTAL (GPU time)", total_gpu);
        fprintf(stderr, "[SINGLE] ==================================\n");
    }

    fprintf(stderr, "[SINGLE] gpu=%.1fms total=%.1fms (%dx%d)\n",
            t_gpu, get_time_ms() - t0, width, height);
    fprintf(stderr, "[GPU_PIPELINE_TIME] %.2f\n", t_gpu);
    ret = 0;

sp_cleanup:
    if(k_sigma_hist)  clReleaseKernel(k_sigma_hist);
    if(k_sigma_final) clReleaseKernel(k_sigma_final);
    if(k_normalize)   clReleaseKernel(k_normalize);
    if(k_denormalize) clReleaseKernel(k_denormalize);
    if(k_fused)       clReleaseKernel(k_fused);
    if(plane_buf)  clReleaseMemObject(plane_buf);
    if(norm_buf)   clReleaseMemObject(norm_buf);
    if(out_buf)    clReleaseMemObject(out_buf);
    if(result_buf) clReleaseMemObject(result_buf);
    if(hist_buf)   clReleaseMemObject(hist_buf);
    if(sigma_buf)  clReleaseMemObject(sigma_buf);
    if(prog)    clReleaseProgram(prog);
    if(queue)   clReleaseCommandQueue(queue);
    if(context) clReleaseContext(context);
    free_aligned(plane);
    return ret;
}



int main(int argc, char **argv)
{
    if(argc < 6) {
        fprintf(stderr,
            "Usage: %s <in.bin> <out.bin> <W> <H> <strength> [cl_dev]\n",
            argv[0]);
        return 1;
    }
    const char *input_file  = argv[1];
    const char *output_file = argv[2];
    const int   width       = atoi(argv[3]);
    const int   height      = atoi(argv[4]);
    const float strength    = (float)atof(argv[5]);
    const int   dev         = (argc > 6) ? atoi(argv[6]) : 0;
    return run_single_plane_gpu(input_file, output_file, width, height,
                                strength, dev);
}
