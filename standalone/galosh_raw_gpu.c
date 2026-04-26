/* galosh_raw_gpu.c  --  GALOSH_RAW_G GPU pipeline driver.
 *
 * Bayer RAW float32 input.  Pipeline (all on GPU compute):
 *   GAT estimation -> GAT forward -> dark_ref -> 2x2 WHT decompose
 *   K13 luma WHT-LOSH (Pass1 + Pass2)
 *   K13 chroma LOESS guided by Y
 *   K14 compute_L_fullres (var=1 box) -> K15 full-res L Pass2
 *   K16 inverse 2x2 WHT (per-pixel with EWA-JL3 chroma upsample) -> inv-GAT
 *
 * GALOSH_RAW_G structural features (see galosh_raw_cpu.c):
 *   - K16 chroma full-res EWA-JL3 reconstruction
 *   - Pass1 BayesShrink with MAD-based sigma_Y
 *
 * Companion CPU reference: galosh_raw_cpu.c (same numerical pipeline,
 * OMP-parallel).  Bench validation: PSNR_isp / PSNR_gt2 should match
 * within float precision; speed comparison shows GPU vs CPU.
 *
 * Usage: galosh_raw_gpu <in.bin> <out.bin> <W> <H>
 *                       <strength> <luma_str> <chroma_str>
 *                       <alpha> <sigma_sq> [cl_dev]
 *
 * Build: gcc -O3 -march=native -o galosh_raw_gpu.exe galosh_raw_gpu.c  *            -lOpenCL -lm
 */

#include "galosh_gpu.h"


/* ================================================================
 * GALOSH_RAW_G GPU full-pipeline driver.  Reads input.bin, runs the
 * full Bayer pipeline on GPU, writes output.bin.  alpha / sigma_sq
 * <= 0 trigger blind GPU estimation.
 * ================================================================ */
static int run_galosh_raw_gpu(const char *input_file, const char *output_file,
                              int width, int height, float strength,
                              float luma_str, float chroma_str,
                              float alpha, float sigma_sq, int cl_device_idx)
{

    const int hw = width / 2, hh = height / 2;
    const size_t npix = (size_t)width * height;
    const size_t chsize = (size_t)hw * hh;

    fprintf(stderr, "[GPU] GALOSH full-pipeline: %dx%d, half=%dx%d, stride=%d, tile=%d\n",
            width, height, hw, hh, GALOSH_STRIDE, TILE_SIZE);
    fprintf(stderr, "[GPU] alpha=%.6f sigma_sq=%.6f luma_str=%.3f chroma_str=%.3f\n",
            alpha, sigma_sq, luma_str, chroma_str);

    /* --- List OpenCL devices --- */
    {
        cl_platform_id plats[8]; cl_uint np = 0;
        clGetPlatformIDs(8, plats, &np);
        fprintf(stderr, "[CL] Available GPU devices:\n");
        int idx = 0;
        for(cl_uint p = 0; p < np; p++) {
            cl_device_id devs[8]; cl_uint nd = 0;
            clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 8, devs, &nd);
            for(cl_uint d = 0; d < nd; d++) {
                char name[256]; cl_uint cu; cl_ulong lm;
                clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(name), name, NULL);
                clGetDeviceInfo(devs[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
                clGetDeviceInfo(devs[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lm), &lm, NULL);
                fprintf(stderr, "  [%d] %s (%u CU, %lu KB LDS)%s\n",
                        idx, name, cu, (unsigned long)(lm/1024),
                        (idx == cl_device_idx) ? " <<< SELECTED" : "");
                idx++;
            }
        }
    }

    double t_total_start = get_time_ms();

    /* --- Read input --- */
    float *raw = alloc_float(npix);
    {
        FILE *f = fopen(input_file, "rb");
        if(!f) { fprintf(stderr, "Cannot open %s\n", input_file); return 1; }
        fread(raw, sizeof(float), npix, f);
        fclose(f);
    }

    /* ================================================================
     * OpenCL setup: platform, device, context, queue, program
     * ================================================================ */
    cl_int err;
    cl_platform_id platforms[8];
    cl_uint n_platforms = 0;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;

    /* Kernel objects — 21 kernels for full pipeline (incl. blind noise estimation) */
    cl_kernel k_noise_block_stats = NULL;
    cl_kernel k_noise_dark_samp_hist = NULL;
    cl_kernel k_noise_estimate = NULL;
    cl_kernel k_noise_dark_lap_hist = NULL;
    cl_kernel k_noise_dark_finalize = NULL;
    cl_kernel k_gat_extract = NULL;
    cl_kernel k_build_inv_lut = NULL;
    cl_kernel k_lut_finalize = NULL;
    cl_kernel k_sigma_histogram = NULL;
    cl_kernel k_sigma_finalize = NULL;
    cl_kernel k_normalize = NULL;
    cl_kernel k_dark_ref_reduce = NULL;
    cl_kernel k_dark_ref_finalize = NULL;
    cl_kernel k_dark_ref_resid_reduce = NULL;
    cl_kernel k_dark_ref_resid_finalize = NULL;
    cl_kernel k_dark_ref_subtract = NULL;
    cl_kernel k_wht_decompose = NULL;
    cl_kernel k_pass1_only = NULL;
    cl_kernel k_fused_pass12 = NULL;
    /* RAW mode chroma guided-filter (C1/C2/C3 plane): reuses yuv kernels. */
    cl_kernel k_raw_chroma_derive      = NULL;
    cl_kernel k_raw_h2f                = NULL;  /* half→float for raw chroma guide */
    cl_kernel k_raw_f2h                = NULL;  /* float→half after apply_y      */
    cl_kernel k_raw_guided_moments_x   = NULL;
    cl_kernel k_raw_guided_moments_yab = NULL;
    cl_kernel k_raw_guided_apply_x     = NULL;
    cl_kernel k_raw_guided_apply_y     = NULL;
    cl_kernel k_raw_guided_loess       = NULL;  /* LOESS (bilateral-weighted) guided filter */
    cl_kernel k_raw_guided_loess_3p    = NULL;  /* 3-plane variant: C1+C2+C3 in one launch */
    cl_kernel k_compute_L_fullres = NULL;
    cl_kernel k_pass2_only = NULL;
    cl_kernel k_reconstruct = NULL;
    cl_kernel k_stream_preprocess = NULL;
    cl_kernel k_stream_postprocess = NULL;

    /* GPU buffers */
    cl_mem raw_buf = NULL;
    cl_mem ch0_buf = NULL, ch1_buf = NULL, ch2_buf = NULL, ch3_buf = NULL;
    cl_mem luma_buf = NULL, c1_buf = NULL, c2_buf = NULL, c3_buf = NULL;
    cl_mem l_out_buf = NULL, c1_out_buf = NULL, c2_out_buf = NULL, c3_out_buf = NULL;
    /* Raw-mode chroma guided-filter scratch (6 moments_x + 4 apply_x, each hw*hh). */
    cl_mem rgf_sum_Y_x = NULL, rgf_sum_YY_x = NULL;
    cl_mem rgf_sum_Cb_x = NULL, rgf_sum_Cr_x = NULL;
    cl_mem rgf_sum_YCb_x = NULL, rgf_sum_YCr_x = NULL;
    cl_mem rgf_a_cb_x = NULL, rgf_b_cb_x = NULL;
    cl_mem rgf_a_cr_x = NULL, rgf_b_cr_x = NULL;
    /* Raw mode: luma/c1/c2/c3 are stored as half; guided filter kernels
     * expect float32 — keep 4 converted float buffers here. */
    cl_mem luma_f_buf = NULL, c1_f_buf = NULL, c2_f_buf = NULL, c3_f_buf = NULL;
    cl_mem Lfr_noisy_buf = NULL, Lfr_pilot_buf = NULL, Lfr_den_buf = NULL;
    cl_mem lut_d_buf = NULL, lut_x_buf = NULL, lut_params_buf = NULL;
    cl_mem params_buf = NULL;
    cl_mem hist_buf = NULL;
    cl_mem partial_buf = NULL, partial_resid_buf = NULL;
    cl_mem pilot_tmp_buf = NULL;
    cl_mem blk_mean_buf = NULL, blk_var_buf = NULL;
    cl_mem ne_dark_hist_buf = NULL, ne_lap_hist_buf = NULL;

    double t_cl_setup = get_time_ms();

    err = clGetPlatformIDs(8, platforms, &n_platforms);
    CL_CHECK(err, "getPlatforms");

    /* Enumerate all GPU devices across platforms */
    cl_device_id all_devices[32];
    cl_platform_id device_platform[32];
    int n_devices_total = 0;
    for(cl_uint p = 0; p < n_platforms; p++) {
        cl_device_id devs[8]; cl_uint nd = 0;
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 8, devs, &nd);
        for(cl_uint d = 0; d < nd && n_devices_total < 32; d++) {
            device_platform[n_devices_total] = platforms[p];
            all_devices[n_devices_total] = devs[d];
            n_devices_total++;
        }
    }
    if(n_devices_total == 0) {
        fprintf(stderr, "[CL] No GPU devices found!\n");
        goto cl_cleanup;
    }
    if(cl_device_idx >= n_devices_total) {
        fprintf(stderr, "[CL] Device index %d out of range (max %d)\n",
                cl_device_idx, n_devices_total - 1);
        goto cl_cleanup;
    }

    device = all_devices[cl_device_idx];
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CL_CHECK(err, "createContext");
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err, "createQueue");

    /* Load and build single-source program (galosh.cl = unified RAW + YUV kernels). */
    {
        const char *cl_paths[] = {
            "galosh.cl",
            "C:/Users/luxgrain/GALOSH/standalone/galosh.cl",
            NULL
        };
        char *source = NULL;
        size_t src_len = 0;
        const char *cl_src_path = NULL;
        for(int i = 0; cl_paths[i]; i++) {
            source = load_kernel_source(cl_paths[i], &src_len);
            if(source) { cl_src_path = cl_paths[i]; break; }
        }
        if(!source) {
            fprintf(stderr, "[CL] Cannot find kernel source galosh.cl\n");
            goto cl_cleanup;
        }

        char build_opts[512];
        snprintf(build_opts, sizeof(build_opts),
                 "-DGALOSH_STRIDE=%d -DTILE_SIZE=%d "
                 "-DHIST_BINS=%d -DREDUCE_WG_SIZE=%d",
                 GALOSH_STRIDE, TILE_SIZE, HIST_BINS, REDUCE_WG_SIZE);

        /* Binary cache path — rev'd for single-source merged build so old
         * 2-source caches are invalidated automatically. */
        char cache_path[512];
        snprintf(cache_path, sizeof(cache_path),
                 "%s.unified_s%d_t%d_dev%d.bin",
                 cl_src_path, GALOSH_STRIDE, TILE_SIZE, cl_device_idx);

        int use_cache = 0;
#ifdef _WIN32
        WIN32_FILE_ATTRIBUTE_DATA src_attr, bin_attr;
        if(GetFileAttributesExA(cache_path, GetFileExInfoStandard, &bin_attr) &&
           GetFileAttributesExA(cl_src_path, GetFileExInfoStandard, &src_attr))
        {
            if(CompareFileTime(&bin_attr.ftLastWriteTime, &src_attr.ftLastWriteTime) >= 0)
                use_cache = 1;
        }
#else
        struct stat src_st, bin_st;
        if(stat(cache_path, &bin_st) == 0 && stat(cl_src_path, &src_st) == 0)
            if(bin_st.st_mtime >= src_st.st_mtime) use_cache = 1;
#endif

        double t_build = get_time_ms();

        if(use_cache) {
            size_t bin_len = 0;
            unsigned char *binary = (unsigned char *)load_kernel_source(cache_path, &bin_len);
            if(binary) {
                cl_int bin_status;
                program = clCreateProgramWithBinary(context, 1, &device,
                    &bin_len, (const unsigned char **)&binary, &bin_status, &err);
                free(binary);
                if(err == CL_SUCCESS && bin_status == CL_SUCCESS) {
                    err = clBuildProgram(program, 1, &device, build_opts, NULL, NULL);
                    if(err == CL_SUCCESS) {
                        t_build = get_time_ms() - t_build;
                        fprintf(stderr, "[CL] Loaded cached binary (%.1f ms)\n", t_build);
                        free(source);
                        goto kernel_ready;
                    }
                }
                if(program) { clReleaseProgram(program); program = NULL; }
                fprintf(stderr, "[CL] Cache invalid, recompiling\n");
            }
        }

        /* Compile from source (single-source unified build). */
        program = clCreateProgramWithSource(context, 1, (const char **)&source, &src_len, &err);
        free(source);
        CL_CHECK(err, "createProgram");

        err = clBuildProgram(program, 1, &device, build_opts, NULL, NULL);
        if(err != CL_SUCCESS) {
            char log[8192];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                 sizeof(log), log, NULL);
            fprintf(stderr, "[CL] Build failed:\n%s\n", log);
            goto cl_cleanup;
        }

        t_build = get_time_ms() - t_build;
        fprintf(stderr, "[CL] Compiled from source in %.1f ms\n", t_build);

        /* Save binary cache */
        {
            size_t bin_size = 0;
            clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size, NULL);
            if(bin_size > 0) {
                unsigned char *binary = (unsigned char *)malloc(bin_size);
                clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &binary, NULL);
                FILE *f = fopen(cache_path, "wb");
                if(f) { fwrite(binary, 1, bin_size, f); fclose(f);
                    fprintf(stderr, "[CL] Saved binary cache (%zu bytes)\n", bin_size); }
                free(binary);
            }
        }

    kernel_ready: ;
    }

    /* ================================================================
     * Create all 18 kernels (incl. blind noise estimation)
     * ================================================================ */
    k_noise_block_stats     = clCreateKernel(program, "galosh_noise_block_stats", &err);
    CL_CHECK(err, "kernel noise_block_stats");
    k_noise_dark_samp_hist  = clCreateKernel(program, "galosh_noise_dark_samp_hist", &err);
    CL_CHECK(err, "kernel noise_dark_samp_hist");
    k_noise_estimate        = clCreateKernel(program, "galosh_noise_estimate", &err);
    CL_CHECK(err, "kernel noise_estimate");
    k_noise_dark_lap_hist   = clCreateKernel(program, "galosh_noise_dark_lap_hist", &err);
    CL_CHECK(err, "kernel noise_dark_lap_hist");
    k_noise_dark_finalize   = clCreateKernel(program, "galosh_noise_dark_finalize", &err);
    CL_CHECK(err, "kernel noise_dark_finalize");
    k_gat_extract           = clCreateKernel(program, "galosh_gat_extract", &err);
    CL_CHECK(err, "kernel gat_extract");
    k_build_inv_lut         = clCreateKernel(program, "galosh_build_inv_lut", &err);
    CL_CHECK(err, "kernel build_inv_lut");
    k_lut_finalize          = clCreateKernel(program, "galosh_lut_finalize", &err);
    CL_CHECK(err, "kernel lut_finalize");
    k_sigma_histogram       = clCreateKernel(program, "galosh_sigma_histogram", &err);
    CL_CHECK(err, "kernel sigma_histogram");
    k_sigma_finalize        = clCreateKernel(program, "galosh_sigma_finalize", &err);
    CL_CHECK(err, "kernel sigma_finalize");
    k_normalize             = clCreateKernel(program, "galosh_normalize", &err);
    CL_CHECK(err, "kernel normalize");
    k_dark_ref_reduce       = clCreateKernel(program, "galosh_dark_ref_reduce", &err);
    CL_CHECK(err, "kernel dark_ref_reduce");
    k_dark_ref_finalize     = clCreateKernel(program, "galosh_dark_ref_finalize", &err);
    CL_CHECK(err, "kernel dark_ref_finalize");
    k_dark_ref_resid_reduce = clCreateKernel(program, "galosh_dark_ref_resid_reduce", &err);
    CL_CHECK(err, "kernel dark_ref_resid_reduce");
    k_dark_ref_resid_finalize = clCreateKernel(program, "galosh_dark_ref_resid_finalize", &err);
    CL_CHECK(err, "kernel dark_ref_resid_finalize");
    k_dark_ref_subtract     = clCreateKernel(program, "galosh_dark_ref_subtract", &err);
    CL_CHECK(err, "kernel dark_ref_subtract");
    k_wht_decompose         = clCreateKernel(program, "galosh_wht_decompose", &err);
    CL_CHECK(err, "kernel wht_decompose");
    k_pass1_only            = clCreateKernel(program, "galosh_pass1_only", &err);
    CL_CHECK(err, "kernel pass1_only");
    k_fused_pass12          = clCreateKernel(program, "galosh_fused_pass12", &err);
    /* Raw-mode chroma guided filter kernels (reuse yuv kernels). */
    k_raw_chroma_derive      = clCreateKernel(program, "galosh_yuv_chroma_params_derive", &err);
    k_raw_h2f                = clCreateKernel(program, "galosh_yuv_half_to_float", &err);
    k_raw_f2h                = clCreateKernel(program, "galosh_yuv_float_to_half", &err);
    k_raw_guided_moments_x   = clCreateKernel(program, "galosh_yuv_guided_moments_x", &err);
    k_raw_guided_moments_yab = clCreateKernel(program, "galosh_yuv_guided_moments_y_ab", &err);
    k_raw_guided_apply_x     = clCreateKernel(program, "galosh_yuv_guided_apply_x", &err);
    k_raw_guided_apply_y     = clCreateKernel(program, "galosh_yuv_guided_apply_y", &err);
    k_raw_guided_loess       = clCreateKernel(program, "galosh_yuv_guided_loess", &err);
    k_raw_guided_loess_3p    = clCreateKernel(program, "galosh_raw_guided_loess_3p", &err);
    CL_CHECK(err, "kernel fused_pass12");
    k_compute_L_fullres     = clCreateKernel(program, "galosh_compute_L_fullres", &err);
    CL_CHECK(err, "kernel compute_L_fullres");
    k_pass2_only            = clCreateKernel(program, "galosh_pass2_only", &err);
    CL_CHECK(err, "kernel pass2_only");
    k_reconstruct           = clCreateKernel(program, "galosh_reconstruct", &err);
    CL_CHECK(err, "kernel reconstruct");
    k_stream_preprocess     = clCreateKernel(program, "galosh_stream_preprocess", &err);
    CL_CHECK(err, "kernel stream_preprocess");
    k_stream_postprocess    = clCreateKernel(program, "galosh_stream_postprocess", &err);
    CL_CHECK(err, "kernel stream_postprocess");

    fprintf(stderr, "[CL] All 24 kernels created\n");

    /* ================================================================
     * Allocate GPU buffers
     * ================================================================ */
    {
        const size_t ch_bytes   = chsize * sizeof(float);
        const size_t full_bytes = npix * sizeof(float);
        const size_t ch_bytes_h   = chsize * sizeof(cl_half);   /* FP16 channel buffers */
        const size_t full_bytes_h = npix * sizeof(cl_half);
        const size_t lut_bytes  = GAT_LUT_SIZE * sizeof(float);
        const size_t hist_bytes = 4 * HIST_BINS * sizeof(cl_int);
        const size_t part_bytes = N_REDUCE_WG * 5 * sizeof(float);
        const size_t resid_bytes = N_REDUCE_WG * 2 * sizeof(float);

        raw_buf       = clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes, NULL, &err);
        CL_CHECK(err, "alloc raw_buf");

        ch0_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes, NULL, &err);
        ch1_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes, NULL, &err);
        ch2_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes, NULL, &err);
        ch3_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes, NULL, &err);
        CL_CHECK(err, "alloc ch_bufs");

        luma_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        c1_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        c2_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        c3_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        CL_CHECK(err, "alloc luma/c_bufs (FP16)");

        l_out_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        c1_out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        c2_out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        c3_out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        /* Guided-filter scratch for raw-mode chroma — all float32, so the
         * buffer size is hw*hh*sizeof(float), NOT ch_bytes_h (which is for
         * half precision = ×1/2 the size). */
        const size_t ch_bytes_f = (size_t)hw * (size_t)hh * sizeof(float);
        rgf_sum_Y_x   = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_sum_YY_x  = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_sum_Cb_x  = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_sum_Cr_x  = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_sum_YCb_x = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_sum_YCr_x = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_a_cb_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_b_cb_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_a_cr_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        rgf_b_cr_x    = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        /* Float32 copies of half luma/c1/c2/c3 used by guided filter. */
        luma_f_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        c1_f_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        c2_f_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        c3_f_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
        CL_CHECK(err, "alloc out_bufs (FP16)");

        Lfr_noisy_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes_h, NULL, &err);
        Lfr_pilot_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes_h, NULL, &err);
        Lfr_den_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes_h, NULL, &err);
        CL_CHECK(err, "alloc Lfr_bufs (FP16)");

        pilot_tmp_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_h, NULL, &err);
        CL_CHECK(err, "alloc pilot_tmp_buf (FP16)");

        lut_d_buf      = clCreateBuffer(context, CL_MEM_READ_WRITE, lut_bytes, NULL, &err);
        lut_x_buf      = clCreateBuffer(context, CL_MEM_READ_WRITE, lut_bytes, NULL, &err);
        lut_params_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * sizeof(float), NULL, &err);
        CL_CHECK(err, "alloc lut_bufs");

        params_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, PARAMS_SIZE * sizeof(float), NULL, &err);
        CL_CHECK(err, "alloc params_buf");

        hist_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, hist_bytes, NULL, &err);
        CL_CHECK(err, "alloc hist_buf");

        partial_buf       = clCreateBuffer(context, CL_MEM_READ_WRITE, part_bytes, NULL, &err);
        partial_resid_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, resid_bytes, NULL, &err);
        CL_CHECK(err, "alloc partial_bufs");

        /* Noise estimation buffers */
        const int ne_n_bx = hw / 16;  /* NE_BLOCK_SZ(8) × NE_BLOCK_STEP(2) */
        const int ne_n_by = hh / 16;
        const int ne_total_blocks = 4 * ne_n_bx * ne_n_by;
        const size_t ne_blk_bytes = ne_total_blocks * sizeof(float);
        blk_mean_buf     = clCreateBuffer(context, CL_MEM_READ_WRITE, ne_blk_bytes, NULL, &err);
        blk_var_buf      = clCreateBuffer(context, CL_MEM_READ_WRITE, ne_blk_bytes, NULL, &err);
        ne_dark_hist_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(cl_int), NULL, &err);
        ne_lap_hist_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, 2048 * sizeof(cl_int), NULL, &err);
        CL_CHECK(err, "alloc noise_est_bufs");
    }

    t_cl_setup = get_time_ms() - t_cl_setup;
    fprintf(stderr, "[GPU] CL setup (ctx+build+alloc): %.1f ms\n", t_cl_setup);

    /* ================================================================
     * Upload raw data + initialize params
     * ================================================================ */
    err = clEnqueueWriteBuffer(queue, raw_buf, CL_TRUE, 0,
                               npix * sizeof(float), raw, 0, NULL, NULL);
    CL_CHECK(err, "upload raw");

    /* Initialize params buffer on host — only strengths are known a priori.
     * alpha, sigma_sq, s_scale will be estimated by GPU kernels. */
    {
        float h_params[PARAMS_SIZE];
        memset(h_params, 0, sizeof(h_params));
        h_params[P_LUMA_STR]   = strength * luma_str;
        h_params[P_CHROMA_STR] = strength * chroma_str;
        err = clEnqueueWriteBuffer(queue, params_buf, CL_TRUE, 0,
                                   PARAMS_SIZE * sizeof(float), h_params, 0, NULL, NULL);
        CL_CHECK(err, "upload params");
    }

    /* ================================================================
     * GPU PIPELINE — All processing on GPU
     *
     * CPU does nothing from here until download.
     * Phase 0: blind noise model estimation (Foi 2008)
     * Phase 1: GAT + σ estimation + normalization
     * Phase 2: dark anchor + WHT
     * Phase 3: chroma denoise
     * Phase 4: GALOSH_F (half-res L + full-res L + reconstruct)
     * ================================================================ */
    double t_pipe_start = get_time_ms();

    /* ---- K0a: Block statistics for noise estimation ---- */
    /* ---- K0c: Dark sample histogram (parallel, independent of K0a) ---- */
    {
        const int ne_n_bx = hw / 16;  /* NE_BLOCK_SZ(8) × NE_BLOCK_STEP(2) */
        const int ne_n_by = hh / 16;
        const int ne_total = 4 * ne_n_bx * ne_n_by;

        /* K0a: block stats */
        clSetKernelArg(k_noise_block_stats, 0, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_noise_block_stats, 1, sizeof(cl_mem), &blk_mean_buf);
        clSetKernelArg(k_noise_block_stats, 2, sizeof(cl_mem), &blk_var_buf);
        clSetKernelArg(k_noise_block_stats, 3, sizeof(int), &width);
        clSetKernelArg(k_noise_block_stats, 4, sizeof(int), &height);
        clSetKernelArg(k_noise_block_stats, 5, sizeof(int), &ne_n_bx);
        clSetKernelArg(k_noise_block_stats, 6, sizeof(int), &ne_n_by);
        dispatch_1d_named(queue, k_noise_block_stats, align_up(ne_total, 64), 64, "K0a block_stats");

        /* K0c: dark sample histogram (parallel, stride-3 sampling) */
        const int samp_per_ch_row = (hw + 2) / 3;
        const int samp_per_ch     = ((hh + 2) / 3) * samp_per_ch_row;
        const int n_dark_samples  = 4 * samp_per_ch;

        {
            cl_int zero = 0;
            clEnqueueFillBuffer(queue, ne_dark_hist_buf, &zero, sizeof(cl_int),
                                0, 1024 * sizeof(cl_int), 0, NULL, NULL);
        }
        clSetKernelArg(k_noise_dark_samp_hist, 0, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_noise_dark_samp_hist, 1, sizeof(cl_mem), &ne_dark_hist_buf);
        clSetKernelArg(k_noise_dark_samp_hist, 2, sizeof(int), &width);
        clSetKernelArg(k_noise_dark_samp_hist, 3, sizeof(int), &height);
        clSetKernelArg(k_noise_dark_samp_hist, 4, sizeof(int), &samp_per_ch_row);
        clSetKernelArg(k_noise_dark_samp_hist, 5, sizeof(int), &samp_per_ch);
        clSetKernelArg(k_noise_dark_samp_hist, 6, sizeof(int), &n_dark_samples);
        dispatch_1d_named(queue, k_noise_dark_samp_hist, align_up(n_dark_samples, 64), 64, "K0c dark_samp_hist");

        /* K0b: noise model estimation (needs K0a + K0c results) */
        clSetKernelArg(k_noise_estimate, 0, sizeof(cl_mem), &blk_mean_buf);
        clSetKernelArg(k_noise_estimate, 1, sizeof(cl_mem), &blk_var_buf);
        clSetKernelArg(k_noise_estimate, 2, sizeof(cl_mem), &ne_dark_hist_buf);
        clSetKernelArg(k_noise_estimate, 3, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_noise_estimate, 4, sizeof(int), &ne_total);
        clSetKernelArg(k_noise_estimate, 5, sizeof(int), &n_dark_samples);
        dispatch_1d_named(queue, k_noise_estimate, 64, 64, "K0b noise_estimate");

        /* K0d: dark Laplacian histogram (parallel, needs dark_max from K0b) */
        const int pos_per_ch_h = hh * (hw - 2);
        const int pos_per_ch_v = (hh - 2) * hw;
        const int pos_per_ch   = pos_per_ch_h + pos_per_ch_v;
        const int total_lap_pos = 4 * pos_per_ch;

        {
            cl_int zero = 0;
            clEnqueueFillBuffer(queue, ne_lap_hist_buf, &zero, sizeof(cl_int),
                                0, 2048 * sizeof(cl_int), 0, NULL, NULL);
        }
        clSetKernelArg(k_noise_dark_lap_hist, 0, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_noise_dark_lap_hist, 1, sizeof(cl_mem), &ne_lap_hist_buf);
        clSetKernelArg(k_noise_dark_lap_hist, 2, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_noise_dark_lap_hist, 3, sizeof(int), &width);
        clSetKernelArg(k_noise_dark_lap_hist, 4, sizeof(int), &height);
        clSetKernelArg(k_noise_dark_lap_hist, 5, sizeof(int), &pos_per_ch_h);
        clSetKernelArg(k_noise_dark_lap_hist, 6, sizeof(int), &pos_per_ch);
        dispatch_1d_named(queue, k_noise_dark_lap_hist, align_up(total_lap_pos, 64), 64, "K0d dark_lap_hist");

        /* K0e: dark sigma finalize (single WI) */
        clSetKernelArg(k_noise_dark_finalize, 0, sizeof(cl_mem), &ne_lap_hist_buf);
        clSetKernelArg(k_noise_dark_finalize, 1, sizeof(cl_mem), &params_buf);
        dispatch_1d_named(queue, k_noise_dark_finalize, 1, 0, "K0e dark_finalize");

        /* Readback estimated alpha/sigma_sq for kernel arg setup.
         * NOT CPU processing — just transferring 2 scalars computed on GPU. */
        clFinish(queue);
        float est_params[PARAMS_SIZE];
        clEnqueueReadBuffer(queue, params_buf, CL_TRUE, 0,
                            PARAMS_SIZE * sizeof(float), est_params, 0, NULL, NULL);
        alpha    = est_params[13];
        sigma_sq = est_params[14];
        fprintf(stderr, "[GPU] Blind noise est: alpha=%.6f sigma_sq=%.8f s_scale=%.6f\n",
                alpha, sigma_sq, est_params[10]);
    }

    /* ---- K1: GAT forward + Bayer extraction (uses estimated alpha/sigma_sq) ---- */
    {
        clSetKernelArg(k_gat_extract, 0, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_gat_extract, 1, sizeof(cl_mem), &ch0_buf);
        clSetKernelArg(k_gat_extract, 2, sizeof(cl_mem), &ch1_buf);
        clSetKernelArg(k_gat_extract, 3, sizeof(cl_mem), &ch2_buf);
        clSetKernelArg(k_gat_extract, 4, sizeof(cl_mem), &ch3_buf);
        clSetKernelArg(k_gat_extract, 5, sizeof(int), &width);
        clSetKernelArg(k_gat_extract, 6, sizeof(int), &height);
        clSetKernelArg(k_gat_extract, 7, sizeof(float), &alpha);
        clSetKernelArg(k_gat_extract, 8, sizeof(float), &sigma_sq);
        dispatch_2d_named(queue, k_gat_extract,
                    align_up(hw, 16), align_up(hh, 16), 16, 16, "K1 gat_extract");
    }

    /* ---- K2: Build inverse GAT LUT (uses estimated alpha/sigma_sq) ---- */
    {
        clSetKernelArg(k_build_inv_lut, 0, sizeof(cl_mem), &lut_d_buf);
        clSetKernelArg(k_build_inv_lut, 1, sizeof(cl_mem), &lut_x_buf);
        clSetKernelArg(k_build_inv_lut, 2, sizeof(float), &alpha);
        clSetKernelArg(k_build_inv_lut, 3, sizeof(float), &sigma_sq);
        dispatch_1d_named(queue, k_build_inv_lut, 4096, 256, "K2 build_inv_lut");
    }

    /* ---- K3: LUT finalize ---- */
    {
        clSetKernelArg(k_lut_finalize, 0, sizeof(cl_mem), &lut_d_buf);
        clSetKernelArg(k_lut_finalize, 1, sizeof(cl_mem), &lut_params_buf);
        clSetKernelArg(k_lut_finalize, 2, sizeof(float), &alpha);
        clSetKernelArg(k_lut_finalize, 3, sizeof(float), &sigma_sq);
        dispatch_1d_named(queue, k_lut_finalize, 1, 0, "K3 lut_finalize");
    }

    /* ---- K4: Sigma histogram ---- */
    {
        /* Zero histogram buffer */
        cl_int zero = 0;
        clEnqueueFillBuffer(queue, hist_buf, &zero, sizeof(cl_int), 0,
                            4 * HIST_BINS * sizeof(cl_int), 0, NULL, NULL);

        clSetKernelArg(k_sigma_histogram, 0, sizeof(cl_mem), &ch0_buf);
        clSetKernelArg(k_sigma_histogram, 1, sizeof(cl_mem), &ch1_buf);
        clSetKernelArg(k_sigma_histogram, 2, sizeof(cl_mem), &ch2_buf);
        clSetKernelArg(k_sigma_histogram, 3, sizeof(cl_mem), &ch3_buf);
        clSetKernelArg(k_sigma_histogram, 4, sizeof(cl_mem), &hist_buf);
        clSetKernelArg(k_sigma_histogram, 5, sizeof(int), &hw);
        clSetKernelArg(k_sigma_histogram, 6, sizeof(int), &hh);

        const int n_x_per_row = (hw - 2) / 3 + 1;
        const int n_sigma_samples = n_x_per_row * hh;
        dispatch_1d_named(queue, k_sigma_histogram,
                    align_up(n_sigma_samples, 256), 256, "K4 sigma_hist");

        /* ---- K5: Sigma finalize ---- */
        clSetKernelArg(k_sigma_finalize, 0, sizeof(cl_mem), &hist_buf);
        clSetKernelArg(k_sigma_finalize, 1, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_sigma_finalize, 2, sizeof(int), &n_sigma_samples);
        dispatch_1d_named(queue, k_sigma_finalize, 1, 0, "K5 sigma_finalize");
    }

    /* ---- K6: Normalize channels ---- */
    {
        const int chsize_i = (int)chsize;
        clSetKernelArg(k_normalize, 0, sizeof(cl_mem), &ch0_buf);
        clSetKernelArg(k_normalize, 1, sizeof(cl_mem), &ch1_buf);
        clSetKernelArg(k_normalize, 2, sizeof(cl_mem), &ch2_buf);
        clSetKernelArg(k_normalize, 3, sizeof(cl_mem), &ch3_buf);
        clSetKernelArg(k_normalize, 4, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_normalize, 5, sizeof(int), &chsize_i);
        dispatch_1d_named(queue, k_normalize, align_up(chsize, 256), 256, "K6 normalize");
    }

    /* ---- K7-K10: Dark reference estimation (3 iterations) ----
     *
     * CPU equivalent: 3 iterations of weighted reduction with
     * achromatic filter + scale refinement via residual std.
     * Each iteration: reduce → finalize (→ resid_reduce → resid_finalize)
     * ---- */
    {
        const int n_iter = 1;  /* 2 loops sufficient for IRLS convergence */
        const float s_init = sigma_sq / fmaxf(alpha, 1e-12f);
        const float s_min = 0.05f * s_init;
        const float s_max = 50.0f * s_init;
        const size_t reduce_global = (size_t)N_REDUCE_WG * REDUCE_WG_SIZE;
        const int n_wg = N_REDUCE_WG;

        /* Set static args for dark_ref_reduce */
        clSetKernelArg(k_dark_ref_reduce, 0, sizeof(cl_mem), &ch0_buf);
        clSetKernelArg(k_dark_ref_reduce, 1, sizeof(cl_mem), &ch1_buf);
        clSetKernelArg(k_dark_ref_reduce, 2, sizeof(cl_mem), &ch2_buf);
        clSetKernelArg(k_dark_ref_reduce, 3, sizeof(cl_mem), &ch3_buf);
        clSetKernelArg(k_dark_ref_reduce, 4, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_dark_ref_reduce, 5, sizeof(cl_mem), &partial_buf);
        clSetKernelArg(k_dark_ref_reduce, 6, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_dark_ref_reduce, 7, sizeof(int), &hw);
        clSetKernelArg(k_dark_ref_reduce, 8, sizeof(int), &hh);
        clSetKernelArg(k_dark_ref_reduce, 9, sizeof(int), &width);

        /* Set static args for dark_ref_finalize */
        clSetKernelArg(k_dark_ref_finalize, 0, sizeof(cl_mem), &partial_buf);
        clSetKernelArg(k_dark_ref_finalize, 1, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_dark_ref_finalize, 2, sizeof(int), &n_wg);

        /* Set static args for resid_reduce */
        clSetKernelArg(k_dark_ref_resid_reduce, 0, sizeof(cl_mem), &ch0_buf);
        clSetKernelArg(k_dark_ref_resid_reduce, 1, sizeof(cl_mem), &ch1_buf);
        clSetKernelArg(k_dark_ref_resid_reduce, 2, sizeof(cl_mem), &ch2_buf);
        clSetKernelArg(k_dark_ref_resid_reduce, 3, sizeof(cl_mem), &ch3_buf);
        clSetKernelArg(k_dark_ref_resid_reduce, 4, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_dark_ref_resid_reduce, 5, sizeof(cl_mem), &partial_resid_buf);
        clSetKernelArg(k_dark_ref_resid_reduce, 6, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_dark_ref_resid_reduce, 7, sizeof(int), &hw);
        clSetKernelArg(k_dark_ref_resid_reduce, 8, sizeof(int), &hh);
        clSetKernelArg(k_dark_ref_resid_reduce, 9, sizeof(int), &width);

        /* Set static args for resid_finalize */
        clSetKernelArg(k_dark_ref_resid_finalize, 0, sizeof(cl_mem), &partial_resid_buf);
        clSetKernelArg(k_dark_ref_resid_finalize, 1, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_dark_ref_resid_finalize, 2, sizeof(int), &n_wg);
        clSetKernelArg(k_dark_ref_resid_finalize, 3, sizeof(float), &s_min);
        clSetKernelArg(k_dark_ref_resid_finalize, 4, sizeof(float), &s_max);

        for(int iter = 0; iter <= n_iter; iter++)
        {
            /* Zero partial sums */
            float fzero = 0.0f;
            clEnqueueFillBuffer(queue, partial_buf, &fzero, sizeof(float), 0,
                                N_REDUCE_WG * 5 * sizeof(float), 0, NULL, NULL);

            /* K7: dark_ref_reduce */
            dispatch_1d_named(queue, k_dark_ref_reduce, reduce_global, REDUCE_WG_SIZE, "K7 dark_reduce");

            /* K8: dark_ref_finalize */
            dispatch_1d_named(queue, k_dark_ref_finalize, 1, 0, "K8 dark_finalize");

            if(iter == n_iter) break;

            /* K9: resid_reduce */
            clEnqueueFillBuffer(queue, partial_resid_buf, &fzero, sizeof(float), 0,
                                N_REDUCE_WG * 2 * sizeof(float), 0, NULL, NULL);
            dispatch_1d_named(queue, k_dark_ref_resid_reduce, reduce_global, REDUCE_WG_SIZE, "K9 resid_reduce");

            /* K10: resid_finalize (updates s_scale in params) */
            dispatch_1d_named(queue, k_dark_ref_resid_finalize, 1, 0, "K10 resid_final");
        }
    }

    /* ---- K_SP: Stream Preprocess (fused normalize + dark_sub + WHT) ---- */
    {
        const int chsize_i = (int)chsize;
        clSetKernelArg(k_stream_preprocess, 0, sizeof(cl_mem), &ch0_buf);
        clSetKernelArg(k_stream_preprocess, 1, sizeof(cl_mem), &ch1_buf);
        clSetKernelArg(k_stream_preprocess, 2, sizeof(cl_mem), &ch2_buf);
        clSetKernelArg(k_stream_preprocess, 3, sizeof(cl_mem), &ch3_buf);
        clSetKernelArg(k_stream_preprocess, 4, sizeof(cl_mem), &luma_buf);
        clSetKernelArg(k_stream_preprocess, 5, sizeof(cl_mem), &c1_buf);
        clSetKernelArg(k_stream_preprocess, 6, sizeof(cl_mem), &c2_buf);
        clSetKernelArg(k_stream_preprocess, 7, sizeof(cl_mem), &c3_buf);
        clSetKernelArg(k_stream_preprocess, 8, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_stream_preprocess, 9, sizeof(int), &chsize_i);
        dispatch_1d_named(queue, k_stream_preprocess, align_up(chsize, 256), 256, "K_SP preprocess");
    }

    /* ---- K13: Fused Pass1+Pass2 denoise (half-res, 4 channels) ----
     *
     * Luma: luma_buf → l_out_buf  (luma_strength)
     * C1:   c1_buf   → c1_out_buf (chroma_strength)
     * C2:   c2_buf   → c2_out_buf (chroma_strength)
     * C3:   c3_buf   → c3_out_buf (chroma_strength)
     * ---- */
    {
        const float luma_strength   = strength * luma_str;
        const float chroma_strength = strength * chroma_str;

        /* Luma */
        clSetKernelArg(k_fused_pass12, 0, sizeof(cl_mem), &luma_buf);
        clSetKernelArg(k_fused_pass12, 1, sizeof(cl_mem), &l_out_buf);
        clSetKernelArg(k_fused_pass12, 2, sizeof(int), &hw);
        clSetKernelArg(k_fused_pass12, 3, sizeof(int), &hh);
        clSetKernelArg(k_fused_pass12, 4, sizeof(float), &luma_strength);
        dispatch_tile_named(queue, k_fused_pass12, hw, hh, "K13 fused_L");

        /* ========================================================
         * C1/C2/C3 — Y-guided filter (Step 8: unified L/C split).
         *
         * Luma plane (L = 2×2-WHT DC) stays on WHT LOSH above; the three
         * chroma sub-bands (C1, C2, C3) move to a Y-guided filter that
         * mirrors the galosh_yuv_gpu chroma path:
         *
         *   C_out(x, y) ≈ mean_a(x,y)·L(x,y) + mean_b(x,y)
         *
         * ε_pixel = strength_c² · (α·L + σ²)  with α, σ² = raw blind
         * estimates.  Guide = luma_buf (GAT'd noisy L plane — good enough
         * as cross-guidance; we do not require the denoised L because
         * edges in luminance are already 5-10× SNR over chroma).
         *
         * 2 dispatch runs (joint-pair + singleton):
         *   Run 1: Cb slot=C1, Cr slot=C2 → C1_out, C2_out
         *   Run 2: Cb slot=C3, Cr slot=C3 → C3_out (Cr output = same, discarded)
         *
         * ε α/σ² plumbed through params[16..19] (yuv chroma slots) —
         * pre-write raw α/σ² there so the shared kernel reads valid values.
         * ======================================================== */
        {
            /* Raw-mode pipeline uses half precision for luma/c1/c2/c3; the
             * shared yuv guided-filter kernels operate on fp32.  Convert
             * half → float once into dedicated scratch buffers, run the
             * 4-stage guided filter, then convert the float output back to
             * half to feed the downstream (half-based) reconstruct stages. */
            const int chsize = hw * hh;
            /* half → float for all 4 half buffers. */
            clSetKernelArg(k_raw_h2f, 0, sizeof(cl_mem), &luma_buf);
            clSetKernelArg(k_raw_h2f, 1, sizeof(cl_mem), &luma_f_buf);
            clSetKernelArg(k_raw_h2f, 2, sizeof(int),    &chsize);
            dispatch_1d_named(queue, k_raw_h2f, align_up(chsize, 256), 256, "K13.a h2f_L");
            clSetKernelArg(k_raw_h2f, 0, sizeof(cl_mem), &c1_buf);
            clSetKernelArg(k_raw_h2f, 1, sizeof(cl_mem), &c1_f_buf);
            dispatch_1d_named(queue, k_raw_h2f, align_up(chsize, 256), 256, "K13.a h2f_C1");
            clSetKernelArg(k_raw_h2f, 0, sizeof(cl_mem), &c2_buf);
            clSetKernelArg(k_raw_h2f, 1, sizeof(cl_mem), &c2_f_buf);
            dispatch_1d_named(queue, k_raw_h2f, align_up(chsize, 256), 256, "K13.a h2f_C2");
            clSetKernelArg(k_raw_h2f, 0, sizeof(cl_mem), &c3_buf);
            clSetKernelArg(k_raw_h2f, 1, sizeof(cl_mem), &c3_f_buf);
            dispatch_1d_named(queue, k_raw_h2f, align_up(chsize, 256), 256, "K13.a h2f_C3");

            /* Derive chroma (α, σ²) from params[13,14] into [16..19]. */
            clSetKernelArg(k_raw_chroma_derive, 0, sizeof(cl_mem), &params_buf);
            dispatch_1d_named(queue, k_raw_chroma_derive, 1, 0, "K13.0 chroma_derive");

            /* LOESS (bilateral-weighted guided filter) — single 2D pass that
             * replaces the old 4-kernel separable-guided pipeline.  Replaces
             *   moments_x → moments_y_ab → apply_x → apply_y
             * with one non-separable kernel that re-weights neighbours by
             * exp(-(Y_i - Y_c)² / (2σ²)), excluding specular highlights from
             * silver windows (see yuv_gat.cl docstring for derivation).
             * Outputs to float scratch, then f2h to downstream c*_out_buf. */
            /* Single-launch 3-plane LOESS: galosh_raw_guided_loess_3p
             * processes C1+C2+C3 in one kernel, sharing the per-pixel
             * 15x15 sweep / sumW / sumY / sumYY / exp() across planes.
             * Replaces the previous 2-pass loop (loess(C1,C2) + loess(C3,
             * dummy)).  ~34% faster on chroma denoise (12MP, FP32 path). */
            cl_mem c1_f_out = rgf_a_cb_x;  /* float scratch (reused) */
            cl_mem c2_f_out = rgf_a_cr_x;
            cl_mem c3_f_out = rgf_b_cr_x;  /* previously labelled "unused" */

            clSetKernelArg(k_raw_guided_loess_3p,  0, sizeof(cl_mem), &luma_f_buf);
            clSetKernelArg(k_raw_guided_loess_3p,  1, sizeof(cl_mem), &c1_f_buf);
            clSetKernelArg(k_raw_guided_loess_3p,  2, sizeof(cl_mem), &c2_f_buf);
            clSetKernelArg(k_raw_guided_loess_3p,  3, sizeof(cl_mem), &c3_f_buf);
            clSetKernelArg(k_raw_guided_loess_3p,  4, sizeof(cl_mem), &params_buf);
            clSetKernelArg(k_raw_guided_loess_3p,  5, sizeof(float),  &chroma_strength);
            clSetKernelArg(k_raw_guided_loess_3p,  6, sizeof(cl_mem), &c1_f_out);
            clSetKernelArg(k_raw_guided_loess_3p,  7, sizeof(cl_mem), &c2_f_out);
            clSetKernelArg(k_raw_guided_loess_3p,  8, sizeof(cl_mem), &c3_f_out);
            clSetKernelArg(k_raw_guided_loess_3p,  9, sizeof(int),    &hw);
            clSetKernelArg(k_raw_guided_loess_3p, 10, sizeof(int),    &hh);
            dispatch_2d_named(queue, k_raw_guided_loess_3p,
                              align_up(hw, 16), align_up(hh, 16),
                              16, 16, "K13 raw_gf_loess_3p");

            /* Float -> half conversion for the 3 chroma outputs. */
            clSetKernelArg(k_raw_f2h, 0, sizeof(cl_mem), &c1_f_out);
            clSetKernelArg(k_raw_f2h, 1, sizeof(cl_mem), &c1_out_buf);
            clSetKernelArg(k_raw_f2h, 2, sizeof(int),    &chsize);
            dispatch_1d_named(queue, k_raw_f2h, align_up(chsize, 256), 256, "K13 f2h_c1");
            clSetKernelArg(k_raw_f2h, 0, sizeof(cl_mem), &c2_f_out);
            clSetKernelArg(k_raw_f2h, 1, sizeof(cl_mem), &c2_out_buf);
            dispatch_1d_named(queue, k_raw_f2h, align_up(chsize, 256), 256, "K13 f2h_c2");
            clSetKernelArg(k_raw_f2h, 0, sizeof(cl_mem), &c3_f_out);
            clSetKernelArg(k_raw_f2h, 1, sizeof(cl_mem), &c3_out_buf);
            dispatch_1d_named(queue, k_raw_f2h, align_up(chsize, 256), 256, "K13 f2h_c3");
        }

    }

    /* ---- K14: Compute L fullres (noisy + pilot) ---- */
    {
        clSetKernelArg(k_compute_L_fullres, 0, sizeof(cl_mem), &luma_buf);
        clSetKernelArg(k_compute_L_fullres, 1, sizeof(cl_mem), &c1_buf);
        clSetKernelArg(k_compute_L_fullres, 2, sizeof(cl_mem), &c2_buf);
        clSetKernelArg(k_compute_L_fullres, 3, sizeof(cl_mem), &c3_buf);
        clSetKernelArg(k_compute_L_fullres, 4, sizeof(cl_mem), &Lfr_noisy_buf);
        clSetKernelArg(k_compute_L_fullres, 5, sizeof(int), &hw);
        clSetKernelArg(k_compute_L_fullres, 6, sizeof(int), &hh);
        dispatch_2d_named(queue, k_compute_L_fullres,
                    align_up(hw, 16), align_up(hh, 16), 16, 16, "K14 Lfr_noisy");

        clSetKernelArg(k_compute_L_fullres, 0, sizeof(cl_mem), &l_out_buf);
        clSetKernelArg(k_compute_L_fullres, 1, sizeof(cl_mem), &c1_out_buf);
        clSetKernelArg(k_compute_L_fullres, 2, sizeof(cl_mem), &c2_out_buf);
        clSetKernelArg(k_compute_L_fullres, 3, sizeof(cl_mem), &c3_out_buf);
        clSetKernelArg(k_compute_L_fullres, 4, sizeof(cl_mem), &Lfr_pilot_buf);
        dispatch_2d_named(queue, k_compute_L_fullres,
                    align_up(hw, 16), align_up(hh, 16), 16, 16, "K14 Lfr_pilot");
    }

    /* ---- K15: Pass2-only for full-res L ---- */
    {
        const float luma_strength = strength * luma_str;
        clSetKernelArg(k_pass2_only, 0, sizeof(cl_mem), &Lfr_noisy_buf);
        clSetKernelArg(k_pass2_only, 1, sizeof(cl_mem), &Lfr_pilot_buf);
        clSetKernelArg(k_pass2_only, 2, sizeof(cl_mem), &Lfr_den_buf);
        clSetKernelArg(k_pass2_only, 3, sizeof(int), &width);
        clSetKernelArg(k_pass2_only, 4, sizeof(int), &height);
        clSetKernelArg(k_pass2_only, 5, sizeof(float), &luma_strength);
        dispatch_tile_named(queue, k_pass2_only, width, height, "K15 pass2_Lfr");
    }

    /* ---- K16: Reconstruct ---- */
    {
        clSetKernelArg(k_reconstruct,  0, sizeof(cl_mem), &Lfr_den_buf);
        clSetKernelArg(k_reconstruct,  1, sizeof(cl_mem), &c1_out_buf);
        clSetKernelArg(k_reconstruct,  2, sizeof(cl_mem), &c2_out_buf);
        clSetKernelArg(k_reconstruct,  3, sizeof(cl_mem), &c3_out_buf);
        clSetKernelArg(k_reconstruct,  4, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_reconstruct,  5, sizeof(cl_mem), &lut_d_buf);
        clSetKernelArg(k_reconstruct,  6, sizeof(cl_mem), &lut_x_buf);
        clSetKernelArg(k_reconstruct,  7, sizeof(cl_mem), &lut_params_buf);
        clSetKernelArg(k_reconstruct,  8, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_reconstruct,  9, sizeof(int), &width);
        clSetKernelArg(k_reconstruct, 10, sizeof(int), &height);
        dispatch_2d_named(queue, k_reconstruct,
                    align_up(hw, 16), align_up(hh, 16), 16, 16, "K16 reconstruct");
    }

    /* Wait for all GPU work to complete */
    clFinish(queue);
    double t_pipe = get_time_ms() - t_pipe_start;

    /* Per-kernel profiling report */
    {
        fprintf(stderr, "\n[GPU] ====== PER-KERNEL PROFILING ======\n");
        double total_gpu = 0.0;
        for(int i = 0; i < prof_count; i++) {
            double ms = event_ms(prof_events[i]);
            total_gpu += ms;
            const char *name = prof_names[i] ? prof_names[i] : "(unnamed)";
            fprintf(stderr, "[GPU]   %-20s %7.3f ms\n", name, ms);
            clReleaseEvent(prof_events[i]);
        }
        fprintf(stderr, "[GPU]   %-20s %7.3f ms\n", "TOTAL (GPU time)", total_gpu);
        fprintf(stderr, "[GPU] ==================================\n");
    }

    /* ================================================================
     * Download output + write file
     * ================================================================ */
    err = clEnqueueReadBuffer(queue, raw_buf, CL_TRUE, 0,
                              npix * sizeof(float), raw, 0, NULL, NULL);
    CL_CHECK(err, "download output");

    {
        FILE *f = fopen(output_file, "wb");
        if(!f) { fprintf(stderr, "Cannot write %s\n", output_file); goto cl_cleanup; }
        fwrite(raw, sizeof(float), npix, f);
        fclose(f);
    }

    double t_total = get_time_ms() - t_total_start;

    /* ================================================================
     * Timing summary
     * ================================================================ */
    fprintf(stderr, "\n[GPU] ====== FULL-PIPELINE TIMING ======\n");
    fprintf(stderr, "[GPU]   GPU pipeline:    %7.1f ms  (all 16 kernels)\n", t_pipe);
    fprintf(stderr, "[GPU]   CL setup:        %7.1f ms  (ctx+build+alloc)\n", t_cl_setup);
    fprintf(stderr, "[GPU]   TOTAL:           %7.1f ms  (incl. file I/O)\n", t_total);
    fprintf(stderr, "[GPU] ==================================\n");

    /* Report GPU-only time for benchmarks (parseable) */
    fprintf(stderr, "[GPU_PIPELINE_TIME] %.2f\n", t_pipe);

    /* Read back params for debug info */
    {
        float h_params[PARAMS_SIZE];
        clEnqueueReadBuffer(queue, params_buf, CL_TRUE, 0,
                            PARAMS_SIZE * sizeof(float), h_params, 0, NULL, NULL);
        fprintf(stderr, "[GPU] Sigma: unified=%.4f (ch: %.4f %.4f %.4f %.4f)\n",
                h_params[P_UNIFIED_SIGMA],
                h_params[P_SIGMA_CH0], h_params[P_SIGMA_CH1],
                h_params[P_SIGMA_CH2], h_params[P_SIGMA_CH3]);
        fprintf(stderr, "[GPU] Dark ref: %.4f %.4f %.4f %.4f\n",
                h_params[P_DARK_REF0], h_params[P_DARK_REF1],
                h_params[P_DARK_REF2], h_params[P_DARK_REF3]);

        /* Diagnostic: GPU LUT values and intermediate buffers */
        float lp[5];
        clEnqueueReadBuffer(queue, lut_params_buf, CL_TRUE, 0, 5*sizeof(float), lp, 0, NULL, NULL);
        fprintf(stderr, "[DIAG] lut_params: d_min=%.6f d_max=%.6f y_break=%.6f t_break=%.4f sigma_raw=%.6f\n",
                lp[0], lp[1], lp[2], lp[3], lp[4]);

        int lut_idx[] = {0, 100, 500, 1000, 2000, 4095};
        for(int i = 0; i < 6; i++) {
            float d_val, x_val;
            clEnqueueReadBuffer(queue, lut_d_buf, CL_TRUE, lut_idx[i]*sizeof(float),
                                sizeof(float), &d_val, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, lut_x_buf, CL_TRUE, lut_idx[i]*sizeof(float),
                                sizeof(float), &x_val, 0, NULL, NULL);
            fprintf(stderr, "[DIAG] LUT D[%d]=%.8f x[%d]=%.8f\n",
                    lut_idx[i], d_val, lut_idx[i], x_val);
        }

        /* Read a few intermediate buffer values (FP16 buffers → cl_half) */
        cl_half diag_h;
        clEnqueueReadBuffer(queue, luma_buf, CL_TRUE, 0, sizeof(cl_half), &diag_h, 0, NULL, NULL);
        fprintf(stderr, "[DIAG] luma[0]=%.6f", (double)cl_half_to_float(diag_h));
        clEnqueueReadBuffer(queue, l_out_buf, CL_TRUE, 0, sizeof(cl_half), &diag_h, 0, NULL, NULL);
        fprintf(stderr, " l_out[0]=%.6f", (double)cl_half_to_float(diag_h));
        clEnqueueReadBuffer(queue, Lfr_den_buf, CL_TRUE, 0, sizeof(cl_half), &diag_h, 0, NULL, NULL);
        fprintf(stderr, " Lfr_den[0]=%.6f\n", (double)cl_half_to_float(diag_h));
    }

cl_cleanup:
    /* Release kernels */
    if(k_noise_block_stats) clReleaseKernel(k_noise_block_stats);
    if(k_noise_dark_samp_hist) clReleaseKernel(k_noise_dark_samp_hist);
    if(k_noise_estimate) clReleaseKernel(k_noise_estimate);
    if(k_noise_dark_lap_hist) clReleaseKernel(k_noise_dark_lap_hist);
    if(k_noise_dark_finalize) clReleaseKernel(k_noise_dark_finalize);
    if(k_gat_extract) clReleaseKernel(k_gat_extract);
    if(k_build_inv_lut) clReleaseKernel(k_build_inv_lut);
    if(k_lut_finalize) clReleaseKernel(k_lut_finalize);
    if(k_sigma_histogram) clReleaseKernel(k_sigma_histogram);
    if(k_sigma_finalize) clReleaseKernel(k_sigma_finalize);
    if(k_normalize) clReleaseKernel(k_normalize);
    if(k_dark_ref_reduce) clReleaseKernel(k_dark_ref_reduce);
    if(k_dark_ref_finalize) clReleaseKernel(k_dark_ref_finalize);
    if(k_dark_ref_resid_reduce) clReleaseKernel(k_dark_ref_resid_reduce);
    if(k_dark_ref_resid_finalize) clReleaseKernel(k_dark_ref_resid_finalize);
    if(k_dark_ref_subtract) clReleaseKernel(k_dark_ref_subtract);
    if(k_wht_decompose) clReleaseKernel(k_wht_decompose);
    if(k_pass1_only) clReleaseKernel(k_pass1_only);
    if(k_fused_pass12) clReleaseKernel(k_fused_pass12);
    if(k_raw_chroma_derive)      clReleaseKernel(k_raw_chroma_derive);
    if(k_raw_h2f)                clReleaseKernel(k_raw_h2f);
    if(k_raw_f2h)                clReleaseKernel(k_raw_f2h);
    if(k_raw_guided_moments_x)   clReleaseKernel(k_raw_guided_moments_x);
    if(k_raw_guided_moments_yab) clReleaseKernel(k_raw_guided_moments_yab);
    if(k_raw_guided_apply_x)     clReleaseKernel(k_raw_guided_apply_x);
    if(k_raw_guided_apply_y)     clReleaseKernel(k_raw_guided_apply_y);
    if(k_raw_guided_loess)       clReleaseKernel(k_raw_guided_loess);
    if(k_raw_guided_loess_3p)    clReleaseKernel(k_raw_guided_loess_3p);
    if(k_compute_L_fullres) clReleaseKernel(k_compute_L_fullres);
    if(k_pass2_only) clReleaseKernel(k_pass2_only);
    if(k_reconstruct) clReleaseKernel(k_reconstruct);
    if(k_stream_preprocess) clReleaseKernel(k_stream_preprocess);
    if(k_stream_postprocess) clReleaseKernel(k_stream_postprocess);

    /* Release buffers */
    if(raw_buf) clReleaseMemObject(raw_buf);
    if(ch0_buf) clReleaseMemObject(ch0_buf);
    if(ch1_buf) clReleaseMemObject(ch1_buf);
    if(ch2_buf) clReleaseMemObject(ch2_buf);
    if(ch3_buf) clReleaseMemObject(ch3_buf);
    if(luma_buf) clReleaseMemObject(luma_buf);
    if(c1_buf) clReleaseMemObject(c1_buf);
    if(c2_buf) clReleaseMemObject(c2_buf);
    if(c3_buf) clReleaseMemObject(c3_buf);
    if(l_out_buf) clReleaseMemObject(l_out_buf);
    if(c1_out_buf) clReleaseMemObject(c1_out_buf);
    if(c2_out_buf) clReleaseMemObject(c2_out_buf);
    if(c3_out_buf) clReleaseMemObject(c3_out_buf);
    if(rgf_sum_Y_x)   clReleaseMemObject(rgf_sum_Y_x);
    if(rgf_sum_YY_x)  clReleaseMemObject(rgf_sum_YY_x);
    if(rgf_sum_Cb_x)  clReleaseMemObject(rgf_sum_Cb_x);
    if(rgf_sum_Cr_x)  clReleaseMemObject(rgf_sum_Cr_x);
    if(rgf_sum_YCb_x) clReleaseMemObject(rgf_sum_YCb_x);
    if(rgf_sum_YCr_x) clReleaseMemObject(rgf_sum_YCr_x);
    if(rgf_a_cb_x)    clReleaseMemObject(rgf_a_cb_x);
    if(rgf_b_cb_x)    clReleaseMemObject(rgf_b_cb_x);
    if(rgf_a_cr_x)    clReleaseMemObject(rgf_a_cr_x);
    if(rgf_b_cr_x)    clReleaseMemObject(rgf_b_cr_x);
    if(luma_f_buf)    clReleaseMemObject(luma_f_buf);
    if(c1_f_buf)      clReleaseMemObject(c1_f_buf);
    if(c2_f_buf)      clReleaseMemObject(c2_f_buf);
    if(c3_f_buf)      clReleaseMemObject(c3_f_buf);
    if(Lfr_noisy_buf) clReleaseMemObject(Lfr_noisy_buf);
    if(Lfr_pilot_buf) clReleaseMemObject(Lfr_pilot_buf);
    if(Lfr_den_buf) clReleaseMemObject(Lfr_den_buf);
    if(pilot_tmp_buf) clReleaseMemObject(pilot_tmp_buf);
    if(lut_d_buf) clReleaseMemObject(lut_d_buf);
    if(lut_x_buf) clReleaseMemObject(lut_x_buf);
    if(lut_params_buf) clReleaseMemObject(lut_params_buf);
    if(params_buf) clReleaseMemObject(params_buf);
    if(hist_buf) clReleaseMemObject(hist_buf);
    if(partial_buf) clReleaseMemObject(partial_buf);
    if(partial_resid_buf) clReleaseMemObject(partial_resid_buf);
    if(blk_mean_buf) clReleaseMemObject(blk_mean_buf);
    if(blk_var_buf) clReleaseMemObject(blk_var_buf);
    if(ne_dark_hist_buf) clReleaseMemObject(ne_dark_hist_buf);
    if(ne_lap_hist_buf) clReleaseMemObject(ne_lap_hist_buf);

    /* Release CL objects */
    if(program) clReleaseProgram(program);
    if(queue) clReleaseCommandQueue(queue);
    if(context) clReleaseContext(context);

    free_aligned(raw);
    return 0;
}


/* ================================================================
 * main: thin CLI wrapper around run_galosh_raw_gpu.
 * ================================================================ */
int main(int argc, char **argv)
{
    if(argc < 10) {
        fprintf(stderr,
            "Usage: %s <in.bin> <out.bin> <W> <H>\n"
            "       <strength> <luma_str> <chroma_str>\n"
            "       <alpha> <sigma_sq> [cl_dev]\n",
            argv[0]);
        return 1;
    }
    const char *input_file  = argv[1];
    const char *output_file = argv[2];
    const int   width       = atoi(argv[3]);
    const int   height      = atoi(argv[4]);
    const float strength    = (float)atof(argv[5]);
    const float luma_str    = (float)atof(argv[6]);
    const float chroma_str  = (float)atof(argv[7]);
    const float alpha       = (float)atof(argv[8]);
    const float sigma_sq    = (float)atof(argv[9]);
    const int   dev         = (argc > 10) ? atoi(argv[10]) : 0;
    return run_galosh_raw_gpu(input_file, output_file, width, height,
                              strength, luma_str, chroma_str,
                              alpha, sigma_sq, dev);
}
