/* galosh_yuv_gpu.c  --  GALOSH_YUV_G GPU pipeline driver.
 *
 * sRGB float32 input, 3 channels.  Pipeline (all on GPU compute):
 *   sRGB -> linear RGB -> BT.709 YCbCr
 *   Y: GAT forward -> WHT-LOSH (Pass1 BayesShrink + Pass2 Wiener)
 *      -> Makitalo exact-unbiased inverse GAT
 *   Cb, Cr: LOESS guided-chroma (Y_den as guide)
 *   YCbCr -> linear RGB -> sRGB gamma -> output
 *
 * GALOSH_YUV_G structural features (see galosh_yuv_cpu.c):
 *   - MAD-based BayesShrink sigma_Y (Pass1) on Y plane
 *
 * Companion CPU reference: galosh_yuv_cpu.c (same numerical pipeline,
 * OMP-parallel).  Bench validation: PSNR_isp should match within float
 * precision; speed comparison shows GPU vs CPU.
 *
 * Usage: galosh_yuv_gpu <in.bin> <out.bin> <W> <H> <s_y> <s_c> [cl_dev]
 *
 * Build: gcc -O3 -march=native -o galosh_yuv_gpu.exe galosh_yuv_gpu.c  *            -lOpenCL -lm
 */

#include "galosh_gpu.h"


/* ================================================================
 * YUV-GAT mode: linear-domain Y-GAT + Y-driven chroma VST
 *
 * EN: Full GALOSH pipeline for sRGB inputs. Converts sRGB → linear
 *     YCbCr, applies GAT on Y (Poisson-Gauss VST) and linear VST
 *     on Cb/Cr (÷√(α·Y+σ²)), runs LOSH shrinkage via
 *     galosh_fused_pass12 on each stabilised plane, then applies
 *     Makitalo-Foi inverse (Y) / linear inverse (Cb/Cr), bivariate
 *     Wiener coupling on (Cb,Cr), and reconstructs sRGB.
 *
 * JP: sRGB 入力を linear YCbCr に分解し、Y は GAT+LOSH+Makitalo 逆、
 *     Cb/Cr は Y 駆動線形 VST+LOSH+逆 VST + bivariate Wiener で
 *     処理する完全 GPU パイプライン。Foi+Alenius blind estimation で
 *     Y の (α,σ²) を推定し、Cb/Cr パラメータは解析式で導出。
 *
 * 2-source CL build: galosh_fused.cl + galosh_yuv_gat.cl
 * I/O: 3ch sRGB float32 interleaved (H×W×3)
 * ================================================================ */
static int run_yuv_gat_gpu(const char *input_file, const char *output_file,
                           int width, int height,
                           float strength_y, float strength_c,
                           int cl_device_idx)
{
    const size_t npix = (size_t)width * height;
    int ret = 1;
    prof_count = 0;

    /* --- Read 3ch sRGB input --- */
    float *srgb = alloc_float(3 * npix);
    {
        FILE *f = fopen(input_file, "rb");
        if(!f) { fprintf(stderr, "[YUV_GAT] Cannot open %s\n", input_file); return 1; }
        fread(srgb, sizeof(float), 3 * npix, f);
        fclose(f);
    }
    fprintf(stderr, "[YUV_GAT] %dx%d, strength_y=%.3f, strength_c=%.3f\n",
            width, height, strength_y, strength_c);

    double t0 = get_time_ms();

    /* --- OpenCL setup --- */
    cl_int err;
    cl_platform_id platforms[8]; cl_uint n_plat = 0;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program prog = NULL;

    /* Kernel handles (18 kernels) */
    cl_kernel k_srgb2ycbcr = NULL, k_ycbcr2srgb = NULL;
    cl_kernel k_blk_stats = NULL, k_dark_samp = NULL, k_noise_est = NULL;
    cl_kernel k_dark_lap = NULL, k_dark_final = NULL, k_chroma_derive = NULL;
    cl_kernel k_gat_fwd = NULL;
    cl_kernel k_f2h = NULL, k_h2f = NULL;
    cl_kernel k_build_lut = NULL, k_lut_fin = NULL;
    cl_kernel k_makitalo = NULL, k_fused = NULL;
    /* Step 7 (Phase 2a) — Y-guided filter, 4-kernel separable pipeline:
     *   1) moments_x        : 1D x-pass over (Y, Y², Cb, Cr, Y·Cb, Y·Cr)
     *   2) moments_y_and_ab : 1D y-pass + local linear regression → (a, b)
     *   3) apply_x          : 1D x-pass over (a, b) coefficient planes
     *   4) apply_y          : 1D y-pass + output = mean_a·Y + mean_b
     * Reduces per-pixel work from O((2r+1)²)=225 to O(2·(2r+1))=30 at r=7. */
    cl_kernel k_guided_moments_x   = NULL;
    cl_kernel k_guided_moments_yab = NULL;
    cl_kernel k_guided_apply_x     = NULL;
    cl_kernel k_guided_apply_y     = NULL;

    /* Buffer handles */
    cl_mem srgb_buf = NULL, y_buf = NULL, cb_buf = NULL, cr_buf = NULL;
    cl_mem y_stab_buf = NULL, cb_stab_buf = NULL, cr_stab_buf = NULL;
    cl_mem scale_cb_buf = NULL, scale_cr_buf = NULL;
    cl_mem half_in_buf = NULL, half_out_buf = NULL;
    cl_mem cb_biv_buf = NULL, cr_biv_buf = NULL;
    /* Phase 2a separable guided filter: 6 moments_x + 4 apply_x scratch.
     * Float32, npix each — ≈640 MB total at 5326×2998 (desktop-only). */
    cl_mem gf_sum_Y_x   = NULL, gf_sum_YY_x  = NULL;
    cl_mem gf_sum_Cb_x  = NULL, gf_sum_Cr_x  = NULL;
    cl_mem gf_sum_YCb_x = NULL, gf_sum_YCr_x = NULL;
    cl_mem gf_a_cb_x    = NULL, gf_b_cb_x    = NULL;
    cl_mem gf_a_cr_x    = NULL, gf_b_cr_x    = NULL;
    cl_mem blk_mean_buf = NULL, blk_var_buf = NULL;
    cl_mem dark_hist_buf = NULL, lap_hist_buf = NULL;
    cl_mem params_buf = NULL;
    cl_mem lut_d_buf = NULL, lut_x_buf = NULL, lut_params_buf = NULL;

    /* Platform / device enumeration */
    err = clGetPlatformIDs(8, platforms, &n_plat);
    if(err != CL_SUCCESS) { fprintf(stderr, "[YUV_GAT] no platforms\n"); goto yg_cleanup; }

    cl_device_id all_devs[32]; int n_dev = 0;
    for(cl_uint p = 0; p < n_plat; p++) {
        cl_device_id devs[8]; cl_uint nd = 0;
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 8, devs, &nd);
        for(cl_uint d = 0; d < nd && n_dev < 32; d++)
            all_devs[n_dev++] = devs[d];
    }
    if(cl_device_idx >= n_dev) {
        fprintf(stderr, "[YUV_GAT] device %d not found\n", cl_device_idx);
        goto yg_cleanup;
    }
    device = all_devs[cl_device_idx];
    {
        char name[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
        fprintf(stderr, "[YUV_GAT] device: %s\n", name);
    }
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if(err) { fprintf(stderr, "[YUV_GAT] context err=%d\n", err); goto yg_cleanup; }
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if(err) { fprintf(stderr, "[YUV_GAT] queue err=%d\n", err); goto yg_cleanup; }

    /* --- Load & build single-source CL program (galosh.cl) --- */
    {
        const char *cl_paths[] = {"galosh.cl",
            "C:/Users/luxgrain/GALOSH/standalone/galosh.cl", NULL};

        char *source = NULL;
        size_t src_len = 0;
        for(int i = 0; cl_paths[i]; i++) {
            source = load_kernel_source(cl_paths[i], &src_len);
            if(source) break;
        }
        if(!source) {
            fprintf(stderr, "[YUV_GAT] Cannot load galosh.cl\n");
            goto yg_cleanup;
        }

        prog = clCreateProgramWithSource(context, 1, (const char **)&source, &src_len, &err);
        free(source);
        if(err) { fprintf(stderr, "[YUV_GAT] program err=%d\n", err); goto yg_cleanup; }

        char opts[512];
        snprintf(opts, sizeof(opts),
            "-DCL_TARGET_OPENCL_VERSION=120 "
            "-DGALOSH_STRIDE=%d -DTILE_SIZE=%d -DHIST_BINS=%d -DREDUCE_WG_SIZE=%d",
            GALOSH_STRIDE, TILE_SIZE, HIST_BINS, REDUCE_WG_SIZE);
        err = clBuildProgram(prog, 1, &device, opts, NULL, NULL);
        if(err) {
            char log[8192];
            clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
            fprintf(stderr, "[YUV_GAT] build err:\n%s\n", log);
            goto yg_cleanup;
        }
    }

    /* --- Create kernels --- */
#define YG_KERNEL(var, name) do { \
    var = clCreateKernel(prog, name, &err); \
    if(err) { fprintf(stderr, "[YUV_GAT] kernel '%s' err=%d\n", name, err); goto yg_cleanup; } \
} while(0)

    YG_KERNEL(k_srgb2ycbcr,   "galosh_yuv_srgb_to_linear_ycbcr");
    YG_KERNEL(k_ycbcr2srgb,   "galosh_yuv_ycbcr_to_srgb");
    YG_KERNEL(k_blk_stats,    "galosh_yuv_noise_block_stats_Y");
    YG_KERNEL(k_dark_samp,    "galosh_yuv_noise_dark_samp_hist_Y");
    YG_KERNEL(k_noise_est,    "galosh_noise_estimate");       /* from fused.cl */
    YG_KERNEL(k_dark_lap,     "galosh_yuv_noise_dark_lap_hist_Y");
    YG_KERNEL(k_dark_final,   "galosh_yuv_noise_dark_finalize_Y");
    YG_KERNEL(k_chroma_derive,"galosh_yuv_chroma_params_derive");
    YG_KERNEL(k_gat_fwd,      "galosh_yuv_gat_forward_Y");
    YG_KERNEL(k_f2h,          "galosh_yuv_float_to_half");
    YG_KERNEL(k_h2f,          "galosh_yuv_half_to_float");
    YG_KERNEL(k_build_lut,    "galosh_build_inv_lut");        /* from fused.cl */
    YG_KERNEL(k_lut_fin,      "galosh_lut_finalize");         /* from fused.cl */
    YG_KERNEL(k_makitalo,     "galosh_yuv_makitalo_inverse_Y");
    YG_KERNEL(k_fused,        "galosh_fused_pass12");         /* from fused.cl */
    YG_KERNEL(k_guided_moments_x,   "galosh_yuv_guided_moments_x");
    YG_KERNEL(k_guided_moments_yab, "galosh_yuv_guided_moments_y_ab");
    YG_KERNEL(k_guided_apply_x,     "galosh_yuv_guided_apply_x");
    YG_KERNEL(k_guided_apply_y,     "galosh_yuv_guided_apply_y");
#undef YG_KERNEL

    /* --- Allocate GPU buffers --- */
    {
        const size_t fb = npix * sizeof(float);
        const size_t hb = npix * sizeof(cl_half);

        srgb_buf      = clCreateBuffer(context, CL_MEM_READ_WRITE, 3 * fb, NULL, &err);
        y_buf         = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        cb_buf        = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        cr_buf        = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        y_stab_buf    = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        cb_stab_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        cr_stab_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        scale_cb_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        scale_cr_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        half_in_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);
        half_out_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);
        cb_biv_buf    = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        cr_biv_buf    = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        /* Separable guided-filter scratch (6 moments_x + 4 apply_x). */
        gf_sum_Y_x   = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_sum_YY_x  = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_sum_Cb_x  = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_sum_Cr_x  = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_sum_YCb_x = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_sum_YCr_x = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_a_cb_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_b_cb_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_a_cr_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        gf_b_cr_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        if(err) { fprintf(stderr, "[YUV_GAT] alloc plane bufs err=%d\n", err); goto yg_cleanup; }

        const int ne_n_bx = width / 16;
        const int ne_n_by = height / 16;
        const int ne_total = ne_n_bx * ne_n_by;
        blk_mean_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, ne_total * sizeof(float), NULL, &err);
        blk_var_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, ne_total * sizeof(float), NULL, &err);
        dark_hist_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(cl_int), NULL, &err);
        lap_hist_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, 2048 * sizeof(cl_int), NULL, &err);
        if(err) { fprintf(stderr, "[YUV_GAT] alloc noise bufs err=%d\n", err); goto yg_cleanup; }

        params_buf     = clCreateBuffer(context, CL_MEM_READ_WRITE, PARAMS_SIZE * sizeof(float), NULL, &err);
        lut_d_buf      = clCreateBuffer(context, CL_MEM_READ_WRITE, GAT_LUT_SIZE * sizeof(float), NULL, &err);
        lut_x_buf      = clCreateBuffer(context, CL_MEM_READ_WRITE, GAT_LUT_SIZE * sizeof(float), NULL, &err);
        lut_params_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * sizeof(float), NULL, &err);
        if(err) { fprintf(stderr, "[YUV_GAT] alloc lut bufs err=%d\n", err); goto yg_cleanup; }
    }

    /* --- Upload sRGB data + init params --- */
    clEnqueueWriteBuffer(queue, srgb_buf, CL_FALSE, 0,
                         3 * npix * sizeof(float), srgb, 0, NULL, NULL);
    {
        float h_params[PARAMS_SIZE];
        memset(h_params, 0, sizeof(h_params));
        /* P_YG_EPS_BIV retained for parameter-slot compatibility; bivariate
         * Wiener was retired in Step 7 (guided-filter-based chroma path). */
        h_params[P_YG_EPS_BIV] = 1e-3f;
        clEnqueueWriteBuffer(queue, params_buf, CL_FALSE, 0,
                             PARAMS_SIZE * sizeof(float), h_params, 0, NULL, NULL);
    }

    double t_pipe_start = get_time_ms();
    const int npix_i = (int)npix;

    /* ================================================================
     * Phase 1: sRGB → linear YCbCr
     * ================================================================ */
    clSetKernelArg(k_srgb2ycbcr, 0, sizeof(cl_mem), &srgb_buf);
    clSetKernelArg(k_srgb2ycbcr, 1, sizeof(cl_mem), &y_buf);
    clSetKernelArg(k_srgb2ycbcr, 2, sizeof(cl_mem), &cb_buf);
    clSetKernelArg(k_srgb2ycbcr, 3, sizeof(cl_mem), &cr_buf);
    clSetKernelArg(k_srgb2ycbcr, 4, sizeof(int), &npix_i);
    dispatch_1d_named(queue, k_srgb2ycbcr, align_up(npix, 256), 256, "YG1 srgb2ycbcr");

    /* ================================================================
     * Phase 2: Y blind noise estimation (Foi+Alenius 2008)
     * ================================================================ */
    {
        const int ne_n_bx = width / 16;
        const int ne_n_by = height / 16;
        const int ne_total = ne_n_bx * ne_n_by;

        /* 2a: block stats */
        clSetKernelArg(k_blk_stats, 0, sizeof(cl_mem), &y_buf);
        clSetKernelArg(k_blk_stats, 1, sizeof(cl_mem), &blk_mean_buf);
        clSetKernelArg(k_blk_stats, 2, sizeof(cl_mem), &blk_var_buf);
        clSetKernelArg(k_blk_stats, 3, sizeof(int), &width);
        clSetKernelArg(k_blk_stats, 4, sizeof(int), &height);
        clSetKernelArg(k_blk_stats, 5, sizeof(int), &ne_n_bx);
        clSetKernelArg(k_blk_stats, 6, sizeof(int), &ne_n_by);
        dispatch_1d_named(queue, k_blk_stats, align_up(ne_total, 64), 64, "YG2a blk_stats");

        /* 2b: dark sample histogram */
        {
            cl_int zero = 0;
            clEnqueueFillBuffer(queue, dark_hist_buf, &zero, sizeof(cl_int),
                                0, 1024 * sizeof(cl_int), 0, NULL, NULL);
        }
        const int samp_per_row = (width + 2) / 3;
        const int n_rows       = (height + 2) / 3;
        const int n_samples    = n_rows * samp_per_row;

        clSetKernelArg(k_dark_samp, 0, sizeof(cl_mem), &y_buf);
        clSetKernelArg(k_dark_samp, 1, sizeof(cl_mem), &dark_hist_buf);
        clSetKernelArg(k_dark_samp, 2, sizeof(int), &width);
        clSetKernelArg(k_dark_samp, 3, sizeof(int), &height);
        clSetKernelArg(k_dark_samp, 4, sizeof(int), &samp_per_row);
        clSetKernelArg(k_dark_samp, 5, sizeof(int), &n_samples);
        dispatch_1d_named(queue, k_dark_samp, align_up(n_samples, 64), 64, "YG2b dark_samp");

        /* 2c: noise model regression (reused from fused.cl) */
        clSetKernelArg(k_noise_est, 0, sizeof(cl_mem), &blk_mean_buf);
        clSetKernelArg(k_noise_est, 1, sizeof(cl_mem), &blk_var_buf);
        clSetKernelArg(k_noise_est, 2, sizeof(cl_mem), &dark_hist_buf);
        clSetKernelArg(k_noise_est, 3, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_noise_est, 4, sizeof(int), &ne_total);
        clSetKernelArg(k_noise_est, 5, sizeof(int), &n_samples);
        dispatch_1d_named(queue, k_noise_est, 64, 64, "YG2c noise_est");

        /* 2d: dark Laplacian histogram (needs dark_max from 2c) */
        {
            cl_int zero = 0;
            clEnqueueFillBuffer(queue, lap_hist_buf, &zero, sizeof(cl_int),
                                0, 2048 * sizeof(cl_int), 0, NULL, NULL);
        }
        const int pos_per_row_h = height * (width - 2);
        const int pos_per_ch    = pos_per_row_h + (height - 2) * width;

        clSetKernelArg(k_dark_lap, 0, sizeof(cl_mem), &y_buf);
        clSetKernelArg(k_dark_lap, 1, sizeof(cl_mem), &lap_hist_buf);
        clSetKernelArg(k_dark_lap, 2, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_dark_lap, 3, sizeof(int), &width);
        clSetKernelArg(k_dark_lap, 4, sizeof(int), &height);
        clSetKernelArg(k_dark_lap, 5, sizeof(int), &pos_per_row_h);
        clSetKernelArg(k_dark_lap, 6, sizeof(int), &pos_per_ch);
        dispatch_1d_named(queue, k_dark_lap, align_up(pos_per_ch, 64), 64, "YG2d dark_lap");

        /* 2e: dark finalize */
        clSetKernelArg(k_dark_final, 0, sizeof(cl_mem), &lap_hist_buf);
        clSetKernelArg(k_dark_final, 1, sizeof(cl_mem), &params_buf);
        dispatch_1d_named(queue, k_dark_final, 1, 0, "YG2e dark_final");
    }

    /* Readback α/σ² — only 2 scalar floats, needed as explicit kernel args */
    float alpha_y, sigma_sq_y;
    {
        clFinish(queue);
        float est[PARAMS_SIZE];
        clEnqueueReadBuffer(queue, params_buf, CL_TRUE, 0,
                            PARAMS_SIZE * sizeof(float), est, 0, NULL, NULL);
        alpha_y    = est[P_ALPHA];
        sigma_sq_y = est[P_SIGMA_SQ];
        fprintf(stderr, "[YUV_GAT] Blind est: alpha=%.6f sigma_sq=%.8f\n",
                alpha_y, sigma_sq_y);
    }

    /* ================================================================
     * Phase 3: Chroma params derive (analytical from Y)
     * ================================================================ */
    clSetKernelArg(k_chroma_derive, 0, sizeof(cl_mem), &params_buf);
    dispatch_1d_named(queue, k_chroma_derive, 1, 0, "YG3 chroma_derive");

    /* ================================================================
     * Phase 4: Y path — GAT → fused LOSH → Makitalo inverse
     * ================================================================ */
    /* 4a: GAT forward */
    clSetKernelArg(k_gat_fwd, 0, sizeof(cl_mem), &y_buf);
    clSetKernelArg(k_gat_fwd, 1, sizeof(cl_mem), &y_stab_buf);
    clSetKernelArg(k_gat_fwd, 2, sizeof(cl_mem), &params_buf);
    clSetKernelArg(k_gat_fwd, 3, sizeof(int), &npix_i);
    dispatch_1d_named(queue, k_gat_fwd, align_up(npix, 256), 256, "YG4a gat_fwd");

    /* 4b: float → half */
    clSetKernelArg(k_f2h, 0, sizeof(cl_mem), &y_stab_buf);
    clSetKernelArg(k_f2h, 1, sizeof(cl_mem), &half_in_buf);
    clSetKernelArg(k_f2h, 2, sizeof(int), &npix_i);
    dispatch_1d_named(queue, k_f2h, align_up(npix, 256), 256, "YG4b f2h_Y");

    /* 4c: build inverse GAT LUT */
    clSetKernelArg(k_build_lut, 0, sizeof(cl_mem), &lut_d_buf);
    clSetKernelArg(k_build_lut, 1, sizeof(cl_mem), &lut_x_buf);
    clSetKernelArg(k_build_lut, 2, sizeof(float), &alpha_y);
    clSetKernelArg(k_build_lut, 3, sizeof(float), &sigma_sq_y);
    dispatch_1d_named(queue, k_build_lut, GAT_LUT_SIZE, 256, "YG4c build_lut");

    /* 4d: LUT finalize */
    clSetKernelArg(k_lut_fin, 0, sizeof(cl_mem), &lut_d_buf);
    clSetKernelArg(k_lut_fin, 1, sizeof(cl_mem), &lut_params_buf);
    clSetKernelArg(k_lut_fin, 2, sizeof(float), &alpha_y);
    clSetKernelArg(k_lut_fin, 3, sizeof(float), &sigma_sq_y);
    dispatch_1d_named(queue, k_lut_fin, 1, 0, "YG4d lut_fin");

    /* 4e: zero half_out accumulator */
    {
        cl_half hzero = 0;
        clEnqueueFillBuffer(queue, half_out_buf, &hzero, sizeof(cl_half), 0,
                            npix * sizeof(cl_half), 0, NULL, NULL);
    }

    /* 4f: fused pass12 (LOSH shrinkage on Y) */
    clSetKernelArg(k_fused, 0, sizeof(cl_mem), &half_in_buf);
    clSetKernelArg(k_fused, 1, sizeof(cl_mem), &half_out_buf);
    clSetKernelArg(k_fused, 2, sizeof(int), &width);
    clSetKernelArg(k_fused, 3, sizeof(int), &height);
    clSetKernelArg(k_fused, 4, sizeof(float), &strength_y);
    dispatch_tile_named(queue, k_fused, width, height, "YG4f fused_Y");

    /* 4g: half → float */
    clSetKernelArg(k_h2f, 0, sizeof(cl_mem), &half_out_buf);
    clSetKernelArg(k_h2f, 1, sizeof(cl_mem), &y_stab_buf);
    clSetKernelArg(k_h2f, 2, sizeof(int), &npix_i);
    dispatch_1d_named(queue, k_h2f, align_up(npix, 256), 256, "YG4g h2f_Y");

    /* 4h: Makitalo-Foi inverse GAT */
    clSetKernelArg(k_makitalo, 0, sizeof(cl_mem), &y_stab_buf);
    clSetKernelArg(k_makitalo, 1, sizeof(cl_mem), &y_buf);   /* overwrite with denoised Y */
    clSetKernelArg(k_makitalo, 2, sizeof(cl_mem), &lut_d_buf);
    clSetKernelArg(k_makitalo, 3, sizeof(cl_mem), &lut_x_buf);
    clSetKernelArg(k_makitalo, 4, sizeof(cl_mem), &lut_params_buf);
    clSetKernelArg(k_makitalo, 5, sizeof(int), &npix_i);
    dispatch_1d_named(queue, k_makitalo, align_up(npix, 256), 256, "YG4h makitalo");

    /* ================================================================
     * Phase 5: Y-guided filter on (Cb, Cr) (Step 7).
     *
     * Two-kernel guided filter (He 2013):
     *   Kernel 1: compute per-pixel (a_Cb, b_Cb, a_Cr, b_Cr) via local
     *             linear regression C ≈ a·Y + b within window.
     *   Kernel 2: box-filter (a, b) coefficient planes and produce
     *             output = mean(a)·Y + mean(b).
     *
     * ε = strength_c² · (α_C·Y + σ²_C)   (Poisson-Gauss, user-scalable)
     *
     * Coefficient planes reuse existing scratch buffers:
     *   cb_stab_buf  ← a_Cb
     *   scale_cb_buf ← b_Cb
     *   cr_stab_buf  ← a_Cr
     *   scale_cr_buf ← b_Cr
     * Final output lands in cb_biv_buf / cr_biv_buf for ycbcr2srgb.
     * ================================================================ */
    /* Step 1: moments_x — 1D x-pass writes 6 intermediate sum buffers. */
    clSetKernelArg(k_guided_moments_x,  0, sizeof(cl_mem), &y_buf);
    clSetKernelArg(k_guided_moments_x,  1, sizeof(cl_mem), &cb_buf);
    clSetKernelArg(k_guided_moments_x,  2, sizeof(cl_mem), &cr_buf);
    clSetKernelArg(k_guided_moments_x,  3, sizeof(cl_mem), &gf_sum_Y_x);
    clSetKernelArg(k_guided_moments_x,  4, sizeof(cl_mem), &gf_sum_YY_x);
    clSetKernelArg(k_guided_moments_x,  5, sizeof(cl_mem), &gf_sum_Cb_x);
    clSetKernelArg(k_guided_moments_x,  6, sizeof(cl_mem), &gf_sum_Cr_x);
    clSetKernelArg(k_guided_moments_x,  7, sizeof(cl_mem), &gf_sum_YCb_x);
    clSetKernelArg(k_guided_moments_x,  8, sizeof(cl_mem), &gf_sum_YCr_x);
    clSetKernelArg(k_guided_moments_x,  9, sizeof(int),    &width);
    clSetKernelArg(k_guided_moments_x, 10, sizeof(int),    &height);
    dispatch_2d_named(queue, k_guided_moments_x,
                      align_up(width, 16), align_up(height, 16),
                      16, 16, "YG5a moments_x");

    /* Step 2: moments_y + (a, b) — 1D y-pass + local linear regression. */
    clSetKernelArg(k_guided_moments_yab,  0, sizeof(cl_mem), &gf_sum_Y_x);
    clSetKernelArg(k_guided_moments_yab,  1, sizeof(cl_mem), &gf_sum_YY_x);
    clSetKernelArg(k_guided_moments_yab,  2, sizeof(cl_mem), &gf_sum_Cb_x);
    clSetKernelArg(k_guided_moments_yab,  3, sizeof(cl_mem), &gf_sum_Cr_x);
    clSetKernelArg(k_guided_moments_yab,  4, sizeof(cl_mem), &gf_sum_YCb_x);
    clSetKernelArg(k_guided_moments_yab,  5, sizeof(cl_mem), &gf_sum_YCr_x);
    clSetKernelArg(k_guided_moments_yab,  6, sizeof(cl_mem), &y_buf);
    clSetKernelArg(k_guided_moments_yab,  7, sizeof(cl_mem), &params_buf);
    clSetKernelArg(k_guided_moments_yab,  8, sizeof(float),  &strength_c);
    clSetKernelArg(k_guided_moments_yab,  9, sizeof(cl_mem), &cb_stab_buf);   /* a_Cb */
    clSetKernelArg(k_guided_moments_yab, 10, sizeof(cl_mem), &scale_cb_buf);  /* b_Cb */
    clSetKernelArg(k_guided_moments_yab, 11, sizeof(cl_mem), &cr_stab_buf);   /* a_Cr */
    clSetKernelArg(k_guided_moments_yab, 12, sizeof(cl_mem), &scale_cr_buf);  /* b_Cr */
    clSetKernelArg(k_guided_moments_yab, 13, sizeof(int),    &width);
    clSetKernelArg(k_guided_moments_yab, 14, sizeof(int),    &height);
    dispatch_2d_named(queue, k_guided_moments_yab,
                      align_up(width, 16), align_up(height, 16),
                      16, 16, "YG5b moments_y_ab");

    /* Step 3: apply_x — 1D x-pass over (a, b) coefficient planes. */
    clSetKernelArg(k_guided_apply_x, 0, sizeof(cl_mem), &cb_stab_buf);
    clSetKernelArg(k_guided_apply_x, 1, sizeof(cl_mem), &scale_cb_buf);
    clSetKernelArg(k_guided_apply_x, 2, sizeof(cl_mem), &cr_stab_buf);
    clSetKernelArg(k_guided_apply_x, 3, sizeof(cl_mem), &scale_cr_buf);
    clSetKernelArg(k_guided_apply_x, 4, sizeof(cl_mem), &gf_a_cb_x);
    clSetKernelArg(k_guided_apply_x, 5, sizeof(cl_mem), &gf_b_cb_x);
    clSetKernelArg(k_guided_apply_x, 6, sizeof(cl_mem), &gf_a_cr_x);
    clSetKernelArg(k_guided_apply_x, 7, sizeof(cl_mem), &gf_b_cr_x);
    clSetKernelArg(k_guided_apply_x, 8, sizeof(int),    &width);
    clSetKernelArg(k_guided_apply_x, 9, sizeof(int),    &height);
    dispatch_2d_named(queue, k_guided_apply_x,
                      align_up(width, 16), align_up(height, 16),
                      16, 16, "YG5c apply_x");

    /* Step 4: apply_y — 1D y-pass + output = mean_a·Y + mean_b. */
    clSetKernelArg(k_guided_apply_y, 0, sizeof(cl_mem), &gf_a_cb_x);
    clSetKernelArg(k_guided_apply_y, 1, sizeof(cl_mem), &gf_b_cb_x);
    clSetKernelArg(k_guided_apply_y, 2, sizeof(cl_mem), &gf_a_cr_x);
    clSetKernelArg(k_guided_apply_y, 3, sizeof(cl_mem), &gf_b_cr_x);
    clSetKernelArg(k_guided_apply_y, 4, sizeof(cl_mem), &y_buf);
    clSetKernelArg(k_guided_apply_y, 5, sizeof(cl_mem), &cb_biv_buf);
    clSetKernelArg(k_guided_apply_y, 6, sizeof(cl_mem), &cr_biv_buf);
    clSetKernelArg(k_guided_apply_y, 7, sizeof(int),    &width);
    clSetKernelArg(k_guided_apply_y, 8, sizeof(int),    &height);
    dispatch_2d_named(queue, k_guided_apply_y,
                      align_up(width, 16), align_up(height, 16),
                      16, 16, "YG5d apply_y");

    cl_mem cb_final = cb_biv_buf;
    cl_mem cr_final = cr_biv_buf;

    /* ================================================================
     * Phase 9: YCbCr → sRGB
     * ================================================================ */
    clSetKernelArg(k_ycbcr2srgb, 0, sizeof(cl_mem), &y_buf);          /* Y_denoised */
    clSetKernelArg(k_ycbcr2srgb, 1, sizeof(cl_mem), &cb_final);       /* Cb final bilateral pass */
    clSetKernelArg(k_ycbcr2srgb, 2, sizeof(cl_mem), &cr_final);       /* Cr final bilateral pass */
    clSetKernelArg(k_ycbcr2srgb, 3, sizeof(cl_mem), &srgb_buf);
    clSetKernelArg(k_ycbcr2srgb, 4, sizeof(int), &npix_i);
    dispatch_1d_named(queue, k_ycbcr2srgb, align_up(npix, 256), 256, "YG9 ycbcr2srgb");

    /* ================================================================
     * Phase 10: Download + Write output
     * ================================================================ */
    clFinish(queue);
    double t_gpu = get_time_ms() - t_pipe_start;

    clEnqueueReadBuffer(queue, srgb_buf, CL_TRUE, 0,
                        3 * npix * sizeof(float), srgb, 0, NULL, NULL);

    {
        FILE *f = fopen(output_file, "wb");
        if(!f) { fprintf(stderr, "[YUV_GAT] Cannot write %s\n", output_file); goto yg_cleanup; }
        fwrite(srgb, sizeof(float), 3 * npix, f);
        fclose(f);
    }

    /* Profiling report */
    {
        fprintf(stderr, "\n[YUV_GAT] ====== PER-KERNEL PROFILING ======\n");
        double total_gpu = 0.0;
        for(int i = 0; i < prof_count; i++) {
            double ms = event_ms(prof_events[i]);
            total_gpu += ms;
            const char *name = prof_names[i] ? prof_names[i] : "(unnamed)";
            fprintf(stderr, "[YUV_GAT]   %-24s %7.3f ms\n", name, ms);
            clReleaseEvent(prof_events[i]);
        }
        fprintf(stderr, "[YUV_GAT]   %-24s %7.3f ms\n", "TOTAL (GPU time)", total_gpu);
        fprintf(stderr, "[YUV_GAT] ==================================\n");
        prof_count = 0;
    }

    fprintf(stderr, "[YUV_GAT] gpu=%.1fms total=%.1fms (%dx%d)\n",
            t_gpu, get_time_ms() - t0, width, height);
    fprintf(stderr, "[GPU_PIPELINE_TIME] %.2f\n", t_gpu);
    ret = 0;

yg_cleanup:
    if(k_srgb2ycbcr)   clReleaseKernel(k_srgb2ycbcr);
    if(k_ycbcr2srgb)   clReleaseKernel(k_ycbcr2srgb);
    if(k_blk_stats)    clReleaseKernel(k_blk_stats);
    if(k_dark_samp)    clReleaseKernel(k_dark_samp);
    if(k_noise_est)    clReleaseKernel(k_noise_est);
    if(k_dark_lap)     clReleaseKernel(k_dark_lap);
    if(k_dark_final)   clReleaseKernel(k_dark_final);
    if(k_chroma_derive)clReleaseKernel(k_chroma_derive);
    if(k_gat_fwd)      clReleaseKernel(k_gat_fwd);
    if(k_f2h)          clReleaseKernel(k_f2h);
    if(k_h2f)          clReleaseKernel(k_h2f);
    if(k_build_lut)    clReleaseKernel(k_build_lut);
    if(k_lut_fin)      clReleaseKernel(k_lut_fin);
    if(k_makitalo)     clReleaseKernel(k_makitalo);
    if(k_fused)        clReleaseKernel(k_fused);

    if(srgb_buf)       clReleaseMemObject(srgb_buf);
    if(y_buf)          clReleaseMemObject(y_buf);
    if(cb_buf)         clReleaseMemObject(cb_buf);
    if(cr_buf)         clReleaseMemObject(cr_buf);
    if(y_stab_buf)     clReleaseMemObject(y_stab_buf);
    if(cb_stab_buf)    clReleaseMemObject(cb_stab_buf);
    if(cr_stab_buf)    clReleaseMemObject(cr_stab_buf);
    if(scale_cb_buf)   clReleaseMemObject(scale_cb_buf);
    if(scale_cr_buf)   clReleaseMemObject(scale_cr_buf);
    if(half_in_buf)    clReleaseMemObject(half_in_buf);
    if(k_guided_moments_x)   clReleaseKernel(k_guided_moments_x);
    if(k_guided_moments_yab) clReleaseKernel(k_guided_moments_yab);
    if(k_guided_apply_x)     clReleaseKernel(k_guided_apply_x);
    if(k_guided_apply_y)     clReleaseKernel(k_guided_apply_y);
    if(half_out_buf)   clReleaseMemObject(half_out_buf);
    if(cb_biv_buf)     clReleaseMemObject(cb_biv_buf);
    if(cr_biv_buf)     clReleaseMemObject(cr_biv_buf);
    if(gf_sum_Y_x)   clReleaseMemObject(gf_sum_Y_x);
    if(gf_sum_YY_x)  clReleaseMemObject(gf_sum_YY_x);
    if(gf_sum_Cb_x)  clReleaseMemObject(gf_sum_Cb_x);
    if(gf_sum_Cr_x)  clReleaseMemObject(gf_sum_Cr_x);
    if(gf_sum_YCb_x) clReleaseMemObject(gf_sum_YCb_x);
    if(gf_sum_YCr_x) clReleaseMemObject(gf_sum_YCr_x);
    if(gf_a_cb_x)    clReleaseMemObject(gf_a_cb_x);
    if(gf_b_cb_x)    clReleaseMemObject(gf_b_cb_x);
    if(gf_a_cr_x)    clReleaseMemObject(gf_a_cr_x);
    if(gf_b_cr_x)    clReleaseMemObject(gf_b_cr_x);
    if(blk_mean_buf)   clReleaseMemObject(blk_mean_buf);
    if(blk_var_buf)    clReleaseMemObject(blk_var_buf);
    if(dark_hist_buf)  clReleaseMemObject(dark_hist_buf);
    if(lap_hist_buf)   clReleaseMemObject(lap_hist_buf);
    if(params_buf)     clReleaseMemObject(params_buf);
    if(lut_d_buf)      clReleaseMemObject(lut_d_buf);
    if(lut_x_buf)      clReleaseMemObject(lut_x_buf);
    if(lut_params_buf) clReleaseMemObject(lut_params_buf);

    if(prog)    clReleaseProgram(prog);
    if(queue)   clReleaseCommandQueue(queue);
    if(context) clReleaseContext(context);
    free_aligned(srgb);
    return ret;
}



/* ================================================================
 * main: thin CLI wrapper around run_yuv_gat_gpu.
 * ================================================================ */
int main(int argc, char **argv)
{
    if(argc < 7) {
        fprintf(stderr,
            "Usage: %s <in.bin> <out.bin> <W> <H> <s_y> <s_c> [cl_dev]\n",
            argv[0]);
        return 1;
    }
    const char *input_file  = argv[1];
    const char *output_file = argv[2];
    const int   width       = atoi(argv[3]);
    const int   height      = atoi(argv[4]);
    const float sy          = (float)atof(argv[5]);
    const float sc          = (float)atof(argv[6]);
    const int   dev         = (argc > 7) ? atoi(argv[7]) : 0;
    return run_yuv_gat_gpu(input_file, output_file, width, height, sy, sc, dev);
}
