/* galosh_raw_gpu.c  --  GALOSH_RAW_G GPU pipeline driver.
 *
 * Bayer RAW float32 input.  Pipeline (all on GPU compute):
 *   K0a-K0e blind noise estimation (Foi-Alenius MAD on raw)
 *   K1 GAT extract / K2-K3 inverse-GAT LUT / K4-K5 sigma / K6 normalize
 *   K7-K10 self-consistent dark anchor (IRLS)
 *   K_SP fused dark_sub + 2x2 WHT decompose -> half-res L/C1/C2/C3
 *   K13 luma  : galosh_fused_pass12   (Pass1 BayesShrink+MAD + Pass2 Wiener)
 *   K13 chroma: galosh_raw_guided_loess_3p (Y-guided bilateral LOESS, 3-plane)
 *   K14 compute_L_fullres (var=1 box, noisy + pilot)
 *   K15 galosh_pass2_only (full-res L Wiener refinement)
 *   K16 galosh_reconstruct_chromaup
 *       — per-pixel inverse 2x2 WHT with EWA Jinc-Lanczos-3 chroma upsample,
 *         then sigma-denormalize + inverse GAT via LUT.  Mirrors CPU
 *         galosh_raw_cpu.c K16 (replaces the block-replicated
 *         galosh_reconstruct kernel; +0.21 dB / -7.7% LPIPS on SIDD Medium).
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
/* variant: 0 = G (= existing GALOSH_RAW_G), 32 = o32 (= full pipeline
 * Phase 0-10, FP32 GPU mirror of CPU --variant=o), 16 = o16 (NYI).
 * o32 verification: per-Phase readback diff vs CPU intermediates. */
static int run_galosh_raw_gpu(const char *input_file, const char *output_file,
                              int width, int height, float strength,
                              float luma_str, float chroma_str,
                              float alpha, float sigma_sq, int cl_device_idx,
                              int variant)
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
    /* O variant: archived 2026-05-09 to archived_o-broke/, replaced by
     * o32 (FP32 mirror of CPU O, Phase 3-4 only as of 2026-05-09). */
    cl_kernel k_o32_assemble_fullres   = NULL;  /* §6.1 */
    cl_kernel k_o32_forward_l_stride1  = NULL;  /* §6.2 */
    cl_kernel k_o32_chroma_extract     = NULL;  /* §6.3 */
    cl_kernel k_o32_pass12             = NULL;  /* §6.5 — Phase 5 WHT-LOSH FP32 (validity-filtered) */
    cl_kernel k_o32_pass1_dump         = NULL;  /* §6.5d — Phase 5 Pass1-only debug pilot extractor */
    cl_kernel k_o32_lpixel_overlap     = NULL;  /* §6.6 — Phase 6 L_pixel (legacy) */
    cl_kernel k_o32_l_h_den_subsample  = NULL;  /* §6.7 — Phase 6 L_h_den (legacy) */
    cl_kernel k_o32_lpixel_lh_den_fused = NULL; /* §6.6+7 fused (production) */
    cl_kernel k_o32_box_down_2x        = NULL;  /* §6.8 — Phase 7 1ch box-down (L pyramid) */
    cl_kernel k_o32_box_down_2x_3p     = NULL;  /* §6.9 — Phase 7 3ch box-down (C pyramid) */
    cl_kernel k_o32_loess_3p           = NULL;  /* §6.10 — Phase 7 LOESS 3ch FP32 (naive) */
    cl_kernel k_o32_loess_3p_tiled     = NULL;  /* §6.10b — LDS-tiled FP32, 16x speedup */
    cl_kernel k_o32_k16_jbu_3p         = NULL;  /* §6.11 — Phase 7+9 K16 3ch FP32 */
    cl_kernel k_o32_pad_2d_edge        = NULL;  /* §6.12 — Phase 7 stride pad */
    cl_kernel k_o32_crop_2d_topleft    = NULL;  /* §6.13 — Phase 7 stride crop */
    cl_kernel k_o32_smoothstep_3p      = NULL;  /* §6.14 — Phase 8 smoothstep walk */
    cl_kernel k_o32_inverse_wht_dark   = NULL;  /* §6.15 — Phase 10 final reconstruct */
    /* Phase 0 noise estimation kernels (o32-specific, replaces G K0a..K0e). */
    cl_kernel k_o32_ne_block_stats     = NULL;  /* §7.1 */
    cl_kernel k_o32_ne_finalize        = NULL;  /* §7.2 */
    cl_kernel k_o32_build_inv_lut      = NULL;  /* §7.3  (replaces G K2) */
    cl_kernel k_o32_lut_finalize       = NULL;  /* §7.3b (replaces G K3) */
    cl_kernel k_o32_ne_dark_thresh_hist     = NULL;  /* §7.2c — histogram-based dark refine */
    cl_kernel k_o32_ne_dark_thresh_finalize = NULL;  /* §7.2d */
    cl_kernel k_o32_ne_dark_lap_hist        = NULL;  /* §7.2e */
    cl_kernel k_o32_ne_dark_finalize        = NULL;  /* §7.2f — writes σ² */
    cl_kernel k_o32_gat_forward_full   = NULL;  /* §7.4  (replaces G K1) */
    cl_kernel k_o32_sigma_per_cfa      = NULL;  /* §7.5  (replaces G K4 sigma_histogram) */
    cl_kernel k_o32_unified_sigma      = NULL;  /* §7.6  (replaces G K5 sigma_finalize) */
    cl_kernel k_o32_normalize_apply    = NULL;  /* §7.7  (replaces G K6 normalize) */
    cl_kernel k_o32_dark_ref_irls      = NULL;  /* §7.8  (replaces G K7-K10, single-WG legacy) */
    cl_kernel k_o32_dr_reduce_mwg      = NULL;  /* §7.8b multi-WG reduce */
    cl_kernel k_o32_dr_finalize_dr_mwg = NULL;  /* §7.8c finalize dark_ref */
    cl_kernel k_o32_dr_resid_reduce_mwg= NULL;  /* §7.8d resid reduce */
    cl_kernel k_o32_dr_resid_finalize_mwg = NULL;  /* §7.8e resid finalize */
    cl_kernel k_o32_dark_sub_full      = NULL;  /* §7.9  (replaces §6.1 + dark_ref_subtract) */

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
    /* O variant buffers archived 2026-05-09 to archived_o-broke/.
     * o32 buffers (FP32, allocated only when variant == 32): */
    cl_mem o32_in_gat_full = NULL;   /* full-res FP32 GAT (post normalize+dark_sub) */
    cl_mem o32_L_cs        = NULL;   /* full-res FP32, Phase 3 output */
    cl_mem o32_L_cs_den    = NULL;   /* full-res FP32, Phase 5 output (= WHT-LOSH denoised L_cs) */
    cl_mem o32_L_pixel     = NULL;   /* full-res FP32, Phase 6 (= 2x2 overlap-avg of L_cs_den) */
    cl_mem o32_L_h_den     = NULL;   /* half-res FP32, Phase 6 (= subsample of L_cs_den) */
    cl_mem o32_C1_h = NULL, o32_C2_h = NULL, o32_C3_h = NULL;  /* half-res FP32, Phase 4 output */
    /* Phase 7 pyramid buffers (cqsize = quarter-res, cesize = eighth-res). */
    cl_mem o32_L_q = NULL, o32_L_e = NULL;     /* L pyramid */
    cl_mem o32_L_for_q = NULL, o32_L_for_e = NULL;  /* L cropped to K16 stride (kqsize, kesize) */
    cl_mem o32_C1_q = NULL, o32_C2_q = NULL, o32_C3_q = NULL;     /* C pyramid quarter input */
    cl_mem o32_C1_e = NULL, o32_C2_e = NULL, o32_C3_e = NULL;     /* C pyramid eighth input */
    cl_mem o32_C1_loess_h = NULL, o32_C2_loess_h = NULL, o32_C3_loess_h = NULL;
    cl_mem o32_C1_loess_q = NULL, o32_C2_loess_q = NULL, o32_C3_loess_q = NULL;
    cl_mem o32_C1_loess_e = NULL, o32_C2_loess_e = NULL, o32_C3_loess_e = NULL;
    cl_mem o32_C1_q_up_raw = NULL, o32_C2_q_up_raw = NULL, o32_C3_q_up_raw = NULL;
    cl_mem o32_C1_q_up = NULL, o32_C2_q_up = NULL, o32_C3_q_up = NULL;
    cl_mem o32_C1_e_to_q_raw = NULL, o32_C2_e_to_q_raw = NULL, o32_C3_e_to_q_raw = NULL;
    cl_mem o32_C1_e_to_q = NULL, o32_C2_e_to_q = NULL, o32_C3_e_to_q = NULL;
    cl_mem o32_C1_e_up_raw = NULL, o32_C2_e_up_raw = NULL, o32_C3_e_up_raw = NULL;
    cl_mem o32_C1_e_up = NULL, o32_C2_e_up = NULL, o32_C3_e_up = NULL;
    /* Phase 8 output (= half-res blended chroma) and Phase 9 output (full-res). */
    cl_mem o32_C1_h_den = NULL, o32_C2_h_den = NULL, o32_C3_h_den = NULL;
    cl_mem o32_C1_aligned = NULL, o32_C2_aligned = NULL, o32_C3_aligned = NULL;
    /* Phase 0 noise est buffers (o32-specific, NE_BLOCK_SZ=8 contiguous,
     * 4× more blocks than G's stride-2 layout). */
    cl_mem o32_blk_mean = NULL, o32_blk_var = NULL;
    /* Phase 0 dark-refine histograms (4096 ints each = 16 KB). */
    cl_mem o32_dark_thresh_hist = NULL;
    cl_mem o32_dark_lap_hist    = NULL;

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
        /* -cl-fp32-correctly-rounded-divide-sqrt forces IEEE-754 strict
         * round-to-nearest for divide and sqrt (default OpenCL allows up
         * to 2.5 ULP error, which can shift WHT-LOSH coefficient
         * magnitudes across BayesShrink's hard threshold and produce
         * spurious GPU/CPU mismatches).  Required for o32 to be a
         * faithful FP32 mirror of CPU O. */
        snprintf(build_opts, sizeof(build_opts),
                 "-cl-fp32-correctly-rounded-divide-sqrt "
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
    /* GALOSH_RAW_G K16: per-pixel inverse 2x2 WHT with EWA Jinc-Lanczos-3
     * chroma upsample (matches CPU galosh_raw_cpu.c).  Replaces the legacy
     * block-replicated `galosh_reconstruct` kernel (which produced visible
     * 2x2 stair-steps on diagonal edges; CPU bench: +0.21 dB / -7.7% LPIPS
     * on SIDD Medium 80-pair). */
    k_reconstruct           = clCreateKernel(program, "galosh_reconstruct_chromaup", &err);
    CL_CHECK(err, "kernel reconstruct_chromaup");
    k_stream_preprocess     = clCreateKernel(program, "galosh_stream_preprocess", &err);
    CL_CHECK(err, "kernel stream_preprocess");
    k_stream_postprocess    = clCreateKernel(program, "galosh_stream_postprocess", &err);
    CL_CHECK(err, "kernel stream_postprocess");

    fprintf(stderr, "[CL] All G-mode kernels created\n");

    /* o32 variant: 3 new kernels for Phase 3-4 (FP32, see galosh.cl §6). */
    if(variant == 32)
    {
        k_o32_assemble_fullres  = clCreateKernel(program, "galosh_o32_assemble_fullres_gat", &err);
        CL_CHECK(err, "kernel o32_assemble_fullres_gat");
        k_o32_forward_l_stride1 = clCreateKernel(program, "galosh_o32_forward_l_stride1", &err);
        CL_CHECK(err, "kernel o32_forward_l_stride1");
        k_o32_chroma_extract    = clCreateKernel(program, "galosh_o32_chroma_extract_halfres", &err);
        CL_CHECK(err, "kernel o32_chroma_extract_halfres");
        k_o32_pass12            = clCreateKernel(program, "galosh_pass12_o32", &err);
        CL_CHECK(err, "kernel pass12_o32");
        k_o32_pass1_dump        = clCreateKernel(program, "galosh_pass1_o32_dump", &err);
        CL_CHECK(err, "kernel pass1_o32_dump");
        k_o32_lpixel_overlap    = clCreateKernel(program, "galosh_o32_lpixel_overlap_avg", &err);
        CL_CHECK(err, "kernel o32_lpixel_overlap_avg");
        k_o32_l_h_den_subsample = clCreateKernel(program, "galosh_o32_l_h_den_subsample", &err);
        CL_CHECK(err, "kernel o32_l_h_den_subsample");
        k_o32_lpixel_lh_den_fused = clCreateKernel(program, "galosh_o32_lpixel_lh_den_fused", &err);
        CL_CHECK(err, "kernel o32_lpixel_lh_den_fused");
        k_o32_box_down_2x       = clCreateKernel(program, "galosh_o32_box_downsample_2x", &err);
        CL_CHECK(err, "kernel o32_box_downsample_2x");
        k_o32_box_down_2x_3p    = clCreateKernel(program, "galosh_o32_box_downsample_2x_3p", &err);
        CL_CHECK(err, "kernel o32_box_downsample_2x_3p");
        k_o32_loess_3p          = clCreateKernel(program, "galosh_o32_loess_chroma_3p", &err);
        CL_CHECK(err, "kernel o32_loess_chroma_3p");
        k_o32_loess_3p_tiled    = clCreateKernel(program, "galosh_o32_loess_chroma_3p_tiled", &err);
        CL_CHECK(err, "kernel o32_loess_chroma_3p_tiled");
        k_o32_k16_jbu_3p        = clCreateKernel(program, "galosh_o32_k16_joint_bilateral_upsample_3p", &err);
        CL_CHECK(err, "kernel o32_k16_jbu_3p");
        k_o32_pad_2d_edge       = clCreateKernel(program, "galosh_o32_pad_2d_edge", &err);
        CL_CHECK(err, "kernel o32_pad_2d_edge");
        k_o32_crop_2d_topleft   = clCreateKernel(program, "galosh_o32_crop_2d_topleft", &err);
        CL_CHECK(err, "kernel o32_crop_2d_topleft");
        k_o32_smoothstep_3p     = clCreateKernel(program, "galosh_o32_smoothstep_blend_3p", &err);
        CL_CHECK(err, "kernel o32_smoothstep_blend_3p");
        k_o32_inverse_wht_dark  = clCreateKernel(program, "galosh_o32_inverse_wht_dark_gat", &err);
        CL_CHECK(err, "kernel o32_inverse_wht_dark_gat");
        /* Phase 0 noise est kernels. */
        k_o32_ne_block_stats    = clCreateKernel(program, "galosh_o32_ne_block_stats", &err);
        CL_CHECK(err, "kernel o32_ne_block_stats");
        k_o32_ne_finalize       = clCreateKernel(program, "galosh_o32_ne_finalize", &err);
        CL_CHECK(err, "kernel o32_ne_finalize");
        k_o32_build_inv_lut     = clCreateKernel(program, "galosh_o32_build_inv_lut", &err);
        CL_CHECK(err, "kernel o32_build_inv_lut");
        k_o32_lut_finalize      = clCreateKernel(program, "galosh_o32_lut_finalize", &err);
        CL_CHECK(err, "kernel o32_lut_finalize");
        /* Phase 0 dark-refine histogram kernels (§7.2c-§7.2f). */
        k_o32_ne_dark_thresh_hist     = clCreateKernel(program, "galosh_o32_ne_dark_thresh_hist", &err);
        CL_CHECK(err, "kernel o32_ne_dark_thresh_hist");
        k_o32_ne_dark_thresh_finalize = clCreateKernel(program, "galosh_o32_ne_dark_thresh_finalize", &err);
        CL_CHECK(err, "kernel o32_ne_dark_thresh_finalize");
        k_o32_ne_dark_lap_hist        = clCreateKernel(program, "galosh_o32_ne_dark_lap_hist", &err);
        CL_CHECK(err, "kernel o32_ne_dark_lap_hist");
        k_o32_ne_dark_finalize        = clCreateKernel(program, "galosh_o32_ne_dark_finalize", &err);
        CL_CHECK(err, "kernel o32_ne_dark_finalize");
        /* Phase 1 kernels. */
        k_o32_gat_forward_full  = clCreateKernel(program, "galosh_o32_gat_forward_full", &err);
        CL_CHECK(err, "kernel o32_gat_forward_full");
        k_o32_sigma_per_cfa     = clCreateKernel(program, "galosh_o32_sigma_per_cfa", &err);
        CL_CHECK(err, "kernel o32_sigma_per_cfa");
        k_o32_unified_sigma     = clCreateKernel(program, "galosh_o32_unified_sigma", &err);
        CL_CHECK(err, "kernel o32_unified_sigma");
        k_o32_normalize_apply   = clCreateKernel(program, "galosh_o32_normalize_apply", &err);
        CL_CHECK(err, "kernel o32_normalize_apply");
        /* Phase 2 kernels. */
        k_o32_dark_ref_irls     = clCreateKernel(program, "galosh_o32_dark_ref_irls", &err);
        CL_CHECK(err, "kernel o32_dark_ref_irls");
        k_o32_dr_reduce_mwg     = clCreateKernel(program, "galosh_o32_dark_ref_reduce_mwg", &err);
        CL_CHECK(err, "kernel o32_dark_ref_reduce_mwg");
        k_o32_dr_finalize_dr_mwg = clCreateKernel(program, "galosh_o32_dark_ref_finalize_mwg", &err);
        CL_CHECK(err, "kernel o32_dark_ref_finalize_mwg");
        k_o32_dr_resid_reduce_mwg = clCreateKernel(program, "galosh_o32_dark_resid_reduce_mwg", &err);
        CL_CHECK(err, "kernel o32_dark_resid_reduce_mwg");
        k_o32_dr_resid_finalize_mwg = clCreateKernel(program, "galosh_o32_dark_resid_finalize_mwg", &err);
        CL_CHECK(err, "kernel o32_dark_resid_finalize_mwg");
        k_o32_dark_sub_full     = clCreateKernel(program, "galosh_o32_dark_sub_full", &err);
        CL_CHECK(err, "kernel o32_dark_sub_full");
        fprintf(stderr, "[CL] +24 o32 kernels created (Phase 0-2 + 3-10 FP32, full pipeline)\n");
    }

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
        /* part_bytes / resid_bytes sized for the LARGEST consumer:
         * - G variant (galosh_dark_ref_reduce): float (= 5 × N_REDUCE_WG × 4B)
         * - o32 variant (galosh_o32_dark_ref_reduce_mwg): double (= 5 × N_REDUCE_WG × 8B)
         * Use sizeof(double) to size for o32 (= safe for both, only ~2.5 KB total). */
        const size_t part_bytes = N_REDUCE_WG * 5 * sizeof(double);
        const size_t resid_bytes = N_REDUCE_WG * 2 * sizeof(double);

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

        /* o32 variant buffer alloc (FP32, allocated only when variant == 32). */
        if(variant == 32)
        {
            const size_t full_bytes_f = npix * sizeof(float);
            const size_t ch_bytes_f   = chsize * sizeof(float);

            o32_in_gat_full = clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes_f, NULL, &err);
            CL_CHECK(err, "alloc o32_in_gat_full");
            o32_L_cs = clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes_f, NULL, &err);
            CL_CHECK(err, "alloc o32_L_cs");
            o32_L_cs_den = clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes_f, NULL, &err);
            CL_CHECK(err, "alloc o32_L_cs_den");
            o32_L_pixel  = clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes_f, NULL, &err);
            CL_CHECK(err, "alloc o32_L_pixel");
            o32_L_h_den  = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            CL_CHECK(err, "alloc o32_L_h_den");
            o32_C1_h = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            o32_C2_h = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            o32_C3_h = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            CL_CHECK(err, "alloc o32_C1/2/3_h");

            /* Phase 7 pyramid buffers — sizes derived from chsize. */
            const int cq_w_a = hw / 2;
            const int cq_h_a = hh / 2;
            const int ce_w_a = cq_w_a / 2;
            const int ce_h_a = cq_h_a / 2;
            const int kq_w_a = 2 * cq_w_a;
            const int kq_h_a = 2 * cq_h_a;
            const int ke_w_a = 2 * ce_w_a;
            const int ke_h_a = 2 * ce_h_a;
            const size_t cqsize_f = (size_t)cq_w_a * cq_h_a * sizeof(float);
            const size_t cesize_f = (size_t)ce_w_a * ce_h_a * sizeof(float);
            const size_t kqsize_f = (size_t)kq_w_a * kq_h_a * sizeof(float);
            const size_t kesize_f = (size_t)ke_w_a * ke_h_a * sizeof(float);

            o32_L_q = clCreateBuffer(context, CL_MEM_READ_WRITE, cqsize_f, NULL, &err);
            o32_L_e = clCreateBuffer(context, CL_MEM_READ_WRITE, cesize_f, NULL, &err);
            o32_L_for_q = clCreateBuffer(context, CL_MEM_READ_WRITE, kqsize_f, NULL, &err);
            o32_L_for_e = clCreateBuffer(context, CL_MEM_READ_WRITE, kesize_f, NULL, &err);
            CL_CHECK(err, "alloc o32 L pyramid + L_for_*");

            o32_C1_q = clCreateBuffer(context, CL_MEM_READ_WRITE, cqsize_f, NULL, &err);
            o32_C2_q = clCreateBuffer(context, CL_MEM_READ_WRITE, cqsize_f, NULL, &err);
            o32_C3_q = clCreateBuffer(context, CL_MEM_READ_WRITE, cqsize_f, NULL, &err);
            o32_C1_e = clCreateBuffer(context, CL_MEM_READ_WRITE, cesize_f, NULL, &err);
            o32_C2_e = clCreateBuffer(context, CL_MEM_READ_WRITE, cesize_f, NULL, &err);
            o32_C3_e = clCreateBuffer(context, CL_MEM_READ_WRITE, cesize_f, NULL, &err);
            CL_CHECK(err, "alloc o32 C pyramid inputs");

            o32_C1_loess_h = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            o32_C2_loess_h = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            o32_C3_loess_h = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            o32_C1_loess_q = clCreateBuffer(context, CL_MEM_READ_WRITE, cqsize_f, NULL, &err);
            o32_C2_loess_q = clCreateBuffer(context, CL_MEM_READ_WRITE, cqsize_f, NULL, &err);
            o32_C3_loess_q = clCreateBuffer(context, CL_MEM_READ_WRITE, cqsize_f, NULL, &err);
            o32_C1_loess_e = clCreateBuffer(context, CL_MEM_READ_WRITE, cesize_f, NULL, &err);
            o32_C2_loess_e = clCreateBuffer(context, CL_MEM_READ_WRITE, cesize_f, NULL, &err);
            o32_C3_loess_e = clCreateBuffer(context, CL_MEM_READ_WRITE, cesize_f, NULL, &err);
            CL_CHECK(err, "alloc o32 C_loess pyramid outputs");

            o32_C1_q_up_raw = clCreateBuffer(context, CL_MEM_READ_WRITE, kqsize_f, NULL, &err);
            o32_C2_q_up_raw = clCreateBuffer(context, CL_MEM_READ_WRITE, kqsize_f, NULL, &err);
            o32_C3_q_up_raw = clCreateBuffer(context, CL_MEM_READ_WRITE, kqsize_f, NULL, &err);
            o32_C1_q_up = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            o32_C2_q_up = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            o32_C3_q_up = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            CL_CHECK(err, "alloc o32 C_q_up");

            o32_C1_e_to_q_raw = clCreateBuffer(context, CL_MEM_READ_WRITE, kesize_f, NULL, &err);
            o32_C2_e_to_q_raw = clCreateBuffer(context, CL_MEM_READ_WRITE, kesize_f, NULL, &err);
            o32_C3_e_to_q_raw = clCreateBuffer(context, CL_MEM_READ_WRITE, kesize_f, NULL, &err);
            o32_C1_e_to_q = clCreateBuffer(context, CL_MEM_READ_WRITE, cqsize_f, NULL, &err);
            o32_C2_e_to_q = clCreateBuffer(context, CL_MEM_READ_WRITE, cqsize_f, NULL, &err);
            o32_C3_e_to_q = clCreateBuffer(context, CL_MEM_READ_WRITE, cqsize_f, NULL, &err);
            CL_CHECK(err, "alloc o32 C_e_to_q");

            o32_C1_e_up_raw = clCreateBuffer(context, CL_MEM_READ_WRITE, kqsize_f, NULL, &err);
            o32_C2_e_up_raw = clCreateBuffer(context, CL_MEM_READ_WRITE, kqsize_f, NULL, &err);
            o32_C3_e_up_raw = clCreateBuffer(context, CL_MEM_READ_WRITE, kqsize_f, NULL, &err);
            o32_C1_e_up = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            o32_C2_e_up = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            o32_C3_e_up = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            CL_CHECK(err, "alloc o32 C_e_up");

            /* Phase 8 + 9 output buffers. */
            o32_C1_h_den = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            o32_C2_h_den = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            o32_C3_h_den = clCreateBuffer(context, CL_MEM_READ_WRITE, ch_bytes_f, NULL, &err);
            CL_CHECK(err, "alloc o32 C_h_den");
            o32_C1_aligned = clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes_f, NULL, &err);
            o32_C2_aligned = clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes_f, NULL, &err);
            o32_C3_aligned = clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes_f, NULL, &err);
            CL_CHECK(err, "alloc o32 C_aligned");

            /* Phase 0 noise estimate buffers (NE_BLOCK_SZ=8, contiguous, no stride). */
            const int o32_ne_n_bx_a = hw / 8;
            const int o32_ne_n_by_a = hh / 8;
            const size_t o32_ne_total_a = (size_t)4 * o32_ne_n_bx_a * o32_ne_n_by_a;
            o32_blk_mean = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          o32_ne_total_a * sizeof(float), NULL, &err);
            o32_blk_var  = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          o32_ne_total_a * sizeof(float), NULL, &err);
            CL_CHECK(err, "alloc o32 blk_mean/blk_var");

            /* §7.2c-§7.2f histogram buffers (4096 ints each = 16 KB). */
            o32_dark_thresh_hist = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                  4096 * sizeof(cl_int), NULL, &err);
            o32_dark_lap_hist    = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                  4096 * sizeof(cl_int), NULL, &err);
            CL_CHECK(err, "alloc o32 dark histograms");

            const size_t o32_total_mb =
                (full_bytes_f * 7 + ch_bytes_f * 7 +
                 cqsize_f * 9 + cesize_f * 6 +
                 kqsize_f * 6 + kesize_f * 3 +
                 cqsize_f * 1 + cesize_f * 1 +
                 ch_bytes_f * 6) / (1024 * 1024);
            fprintf(stderr, "[CL] +%zu MB allocated for o32 variant buffers (Phase 3-10)\n",
                    o32_total_mb);
        }
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

    /* ============================================================
     * Phase 0 — blind α/σ² noise estimation.
     * o32 variant uses §7.1+§7.2 (CPU-mirror, FP32 GPU); G uses K0a..K0e
     * (existing legacy estimator). Both paths write α/σ² to params buf.
     * ============================================================ */
    if(variant == 32)
    {
        /* §7.1 ne_block_stats: per-block mean + Laplacian MAD variance,
         * NE_BLOCK_SZ=8 contiguous (no stride; matches CPU exactly). */
        const int o32_ne_n_bx = hw / 8;
        const int o32_ne_n_by = hh / 8;
        const int o32_ne_per_ch = o32_ne_n_bx * o32_ne_n_by;
        const int o32_ne_total = 4 * o32_ne_per_ch;

        clSetKernelArg(k_o32_ne_block_stats, 0, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_o32_ne_block_stats, 1, sizeof(int), &width);
        clSetKernelArg(k_o32_ne_block_stats, 2, sizeof(int), &height);
        clSetKernelArg(k_o32_ne_block_stats, 3, sizeof(int), &o32_ne_n_bx);
        clSetKernelArg(k_o32_ne_block_stats, 4, sizeof(int), &o32_ne_n_by);
        clSetKernelArg(k_o32_ne_block_stats, 5, sizeof(int), &o32_ne_per_ch);
        clSetKernelArg(k_o32_ne_block_stats, 6, sizeof(cl_mem), &o32_blk_mean);
        clSetKernelArg(k_o32_ne_block_stats, 7, sizeof(cl_mem), &o32_blk_var);
        dispatch_1d_named(queue, k_o32_ne_block_stats,
                          align_up(o32_ne_total, 64), 64, "K_O32_P0a block_stats");

        /* §7.2 ne_finalize: bin envelope + WLS Huber.  Writes params[P_ALPHA].
         * (σ² written by §7.2f below via histogram dark refine.) */
        clSetKernelArg(k_o32_ne_finalize, 0, sizeof(cl_mem), &o32_blk_mean);
        clSetKernelArg(k_o32_ne_finalize, 1, sizeof(cl_mem), &o32_blk_var);
        clSetKernelArg(k_o32_ne_finalize, 2, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_o32_ne_finalize, 3, sizeof(int), &width);
        clSetKernelArg(k_o32_ne_finalize, 4, sizeof(int), &height);
        clSetKernelArg(k_o32_ne_finalize, 5, sizeof(int), &o32_ne_total);
        clSetKernelArg(k_o32_ne_finalize, 6, sizeof(cl_mem), &params_buf);
        dispatch_1d_named(queue, k_o32_ne_finalize, 32, 32, "K_O32_P0b ne_finalize");

        /* §7.2c-§7.2f histogram-based dark refine (multi-WG parallel).
         * Writes σ² to params[P_SIGMA_SQ]; uses params[P_O32_DARK_THRESH]
         * as temp slot for dark_thresh between thresh_finalize and lap_hist. */
        const int p_dark_thresh_slot = 15;   /* free slot in PARAMS_SIZE=32 layout */
        cl_int izero = 0;
        clEnqueueFillBuffer(queue, o32_dark_thresh_hist, &izero, sizeof(cl_int),
                            0, 4096 * sizeof(cl_int), 0, NULL, NULL);

        /* §7.2c thresh_hist: 2D dispatch over (halfwidth/3, halfheight/3). */
        const int hw_a = (width + 1) / 2;
        const int hh_a = (height + 1) / 2;
        const int hw_3 = (hw_a + 2) / 3;
        const int hh_3 = (hh_a + 2) / 3;
        clSetKernelArg(k_o32_ne_dark_thresh_hist, 0, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_o32_ne_dark_thresh_hist, 1, sizeof(int), &width);
        clSetKernelArg(k_o32_ne_dark_thresh_hist, 2, sizeof(int), &height);
        clSetKernelArg(k_o32_ne_dark_thresh_hist, 3, sizeof(cl_mem), &o32_dark_thresh_hist);
        dispatch_2d_named(queue, k_o32_ne_dark_thresh_hist,
                          align_up(hw_3, 16), align_up(hh_3, 16), 16, 16,
                          "K_O32_P0e dark_thresh_hist");

        /* §7.2d thresh_finalize: 1 WI scan. */
        clSetKernelArg(k_o32_ne_dark_thresh_finalize, 0, sizeof(cl_mem), &o32_dark_thresh_hist);
        clSetKernelArg(k_o32_ne_dark_thresh_finalize, 1, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_o32_ne_dark_thresh_finalize, 2, sizeof(int), &p_dark_thresh_slot);
        dispatch_1d_named(queue, k_o32_ne_dark_thresh_finalize, 1, 0, "K_O32_P0f dark_thresh_fin");

        /* §7.2e lap_hist: 2D dispatch over (halfwidth, halfheight). */
        clEnqueueFillBuffer(queue, o32_dark_lap_hist, &izero, sizeof(cl_int),
                            0, 4096 * sizeof(cl_int), 0, NULL, NULL);
        clSetKernelArg(k_o32_ne_dark_lap_hist, 0, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_o32_ne_dark_lap_hist, 1, sizeof(int), &width);
        clSetKernelArg(k_o32_ne_dark_lap_hist, 2, sizeof(int), &height);
        clSetKernelArg(k_o32_ne_dark_lap_hist, 3, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_o32_ne_dark_lap_hist, 4, sizeof(int), &p_dark_thresh_slot);
        clSetKernelArg(k_o32_ne_dark_lap_hist, 5, sizeof(cl_mem), &o32_dark_lap_hist);
        dispatch_2d_named(queue, k_o32_ne_dark_lap_hist,
                          align_up(hw_a, 16), align_up(hh_a, 16), 16, 16,
                          "K_O32_P0g dark_lap_hist");

        /* §7.2f dark_finalize: 1 WI scan + write σ². */
        clSetKernelArg(k_o32_ne_dark_finalize, 0, sizeof(cl_mem), &o32_dark_lap_hist);
        clSetKernelArg(k_o32_ne_dark_finalize, 1, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_o32_ne_dark_finalize, 2, sizeof(int), &p_dark_thresh_slot);
        dispatch_1d_named(queue, k_o32_ne_dark_finalize, 1, 0, "K_O32_P0h dark_finalize");

        /* Readback α/σ² (= 2 scalars, NOT CPU processing per policy). */
        clFinish(queue);
        float est_o32[PARAMS_SIZE];
        clEnqueueReadBuffer(queue, params_buf, CL_TRUE, 0,
                            PARAMS_SIZE * sizeof(float), est_o32, 0, NULL, NULL);
        alpha    = est_o32[P_ALPHA];
        sigma_sq = est_o32[P_SIGMA_SQ];
        fprintf(stderr, "[GPU][o32] Phase 0 noise est: alpha=%.6f sigma_sq=%.8f\n",
                alpha, sigma_sq);
    }
    else
    {
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
    }   /* end else (variant != 32) */

    /* ---- K1: GAT forward + Bayer extraction.
     * o32: §7.4 gat_forward_full (writes both in_gat_full AND ch0..3).
     * G:   legacy galosh_gat_extract (writes ch0..3 only). */
    if(variant == 32)
    {
        clSetKernelArg(k_o32_gat_forward_full, 0, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_o32_gat_forward_full, 1, sizeof(cl_mem), &o32_in_gat_full);
        clSetKernelArg(k_o32_gat_forward_full, 2, sizeof(cl_mem), &ch0_buf);
        clSetKernelArg(k_o32_gat_forward_full, 3, sizeof(cl_mem), &ch1_buf);
        clSetKernelArg(k_o32_gat_forward_full, 4, sizeof(cl_mem), &ch2_buf);
        clSetKernelArg(k_o32_gat_forward_full, 5, sizeof(cl_mem), &ch3_buf);
        clSetKernelArg(k_o32_gat_forward_full, 6, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_o32_gat_forward_full, 7, sizeof(int), &width);
        clSetKernelArg(k_o32_gat_forward_full, 8, sizeof(int), &height);
        dispatch_2d_named(queue, k_o32_gat_forward_full,
                    align_up(width, 16), align_up(height, 16), 16, 16, "K_O32_P1a gat_full");
    }
    else
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

    /* ---- K2: Build inverse GAT LUT (uses estimated alpha/sigma_sq).
     * o32: §7.3 Poisson + Gauss-Hermite (mirrors CPU gat_build_inverse_table).
     * G:   legacy galosh_build_inv_lut. */
    if(variant == 32)
    {
        clSetKernelArg(k_o32_build_inv_lut, 0, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_o32_build_inv_lut, 1, sizeof(cl_mem), &lut_d_buf);
        clSetKernelArg(k_o32_build_inv_lut, 2, sizeof(cl_mem), &lut_x_buf);
        clSetKernelArg(k_o32_build_inv_lut, 3, sizeof(cl_mem), &lut_params_buf);
        dispatch_1d_named(queue, k_o32_build_inv_lut, 4096, 256, "K_O32_P0c build_inv_lut");
    }
    else
    {
        clSetKernelArg(k_build_inv_lut, 0, sizeof(cl_mem), &lut_d_buf);
        clSetKernelArg(k_build_inv_lut, 1, sizeof(cl_mem), &lut_x_buf);
        clSetKernelArg(k_build_inv_lut, 2, sizeof(float), &alpha);
        clSetKernelArg(k_build_inv_lut, 3, sizeof(float), &sigma_sq);
        dispatch_1d_named(queue, k_build_inv_lut, 4096, 256, "K2 build_inv_lut");
    }

    /* ---- K3: LUT finalize ---- */
    if(variant == 32)
    {
        clSetKernelArg(k_o32_lut_finalize, 0, sizeof(cl_mem), &lut_d_buf);
        clSetKernelArg(k_o32_lut_finalize, 1, sizeof(cl_mem), &lut_params_buf);
        dispatch_1d_named(queue, k_o32_lut_finalize, 1, 0, "K_O32_P0d lut_finalize");
    }
    else
    {
        clSetKernelArg(k_lut_finalize, 0, sizeof(cl_mem), &lut_d_buf);
        clSetKernelArg(k_lut_finalize, 1, sizeof(cl_mem), &lut_params_buf);
        clSetKernelArg(k_lut_finalize, 2, sizeof(float), &alpha);
        clSetKernelArg(k_lut_finalize, 3, sizeof(float), &sigma_sq);
        dispatch_1d_named(queue, k_lut_finalize, 1, 0, "K3 lut_finalize");
    }

    /* ---- K4-K6: per-CFA σ → unified_sigma → normalize.
     * o32: §7.5 sigma_per_cfa + §7.6 unified_sigma + §7.7 normalize_apply
     *      (mirrors CPU estimate_gat_sigma_halfres + RMS unified + multiply).
     * G:   legacy K4 sigma_histogram + K5 sigma_finalize + K6 normalize. */
    if(variant == 32)
    {
        /* §7.5 sigma_per_cfa: 4 work-groups (1 per CFA channel),
         * histogram-based MAD on stride=3 horizontal Laplacian samples. */
        clSetKernelArg(k_o32_sigma_per_cfa, 0, sizeof(cl_mem), &o32_in_gat_full);
        clSetKernelArg(k_o32_sigma_per_cfa, 1, sizeof(int), &width);
        clSetKernelArg(k_o32_sigma_per_cfa, 2, sizeof(int), &height);
        clSetKernelArg(k_o32_sigma_per_cfa, 3, sizeof(cl_mem), &params_buf);
        const size_t s_global[1] = { 4 * 64 };  /* 4 WGs × 64 WIs (= O32_SIGMA_WG) */
        const size_t s_local[1]  = { 64 };
        cl_event ev_o32_sig;
        err = clEnqueueNDRangeKernel(queue, k_o32_sigma_per_cfa, 1, NULL,
                                     s_global, s_local, 0, NULL, &ev_o32_sig);
        CL_CHECK(err, "dispatch K_O32_P1b sigma_per_cfa");
        prof_add(ev_o32_sig, "K_O32_P1b sigma_cfa");

        /* §7.6 unified_sigma: single WI scalar compute. */
        clSetKernelArg(k_o32_unified_sigma, 0, sizeof(cl_mem), &params_buf);
        dispatch_1d_named(queue, k_o32_unified_sigma, 1, 0, "K_O32_P1c unified");

        /* §7.7 normalize_apply: in_gat_full + ch0..3 *= 1/unified_sigma. */
        clSetKernelArg(k_o32_normalize_apply, 0, sizeof(cl_mem), &o32_in_gat_full);
        clSetKernelArg(k_o32_normalize_apply, 1, sizeof(cl_mem), &ch0_buf);
        clSetKernelArg(k_o32_normalize_apply, 2, sizeof(cl_mem), &ch1_buf);
        clSetKernelArg(k_o32_normalize_apply, 3, sizeof(cl_mem), &ch2_buf);
        clSetKernelArg(k_o32_normalize_apply, 4, sizeof(cl_mem), &ch3_buf);
        clSetKernelArg(k_o32_normalize_apply, 5, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_o32_normalize_apply, 6, sizeof(int), &width);
        clSetKernelArg(k_o32_normalize_apply, 7, sizeof(int), &height);
        dispatch_2d_named(queue, k_o32_normalize_apply,
                    align_up(width, 16), align_up(height, 16), 16, 16,
                    "K_O32_P1d normalize");
    }
    else
    {
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
    }   /* end else (variant != 32) */

    /* ---- K7-K10: Dark reference estimation.
     * o32: §7.8 dark_ref_irls (3 iter Tukey-bisquare on in_gat_full + raw,
     *      single-WG, mirrors CPU exactly) + §7.9 dark_sub_full.
     * G:   legacy K7-K10 (uses ch0..3 + raw, separate kernels per step). */
    if(variant == 32)
    {
        /* §7.8b-e: multi-WG dark IRLS (3 iterations).  Pattern:
         *   Init params[P_S_SCALE] = sigma_sq / alpha
         *   for iter ∈ {0, 1, 2}:
         *     dispatch reduce_mwg (= 64 WG × 256 WI parallel scan)
         *     dispatch finalize_dr_mwg (= 1 WI aggregate → dark_ref)
         *     if iter == 2 break
         *     dispatch resid_reduce_mwg
         *     dispatch resid_finalize_mwg (= update s_scale, clamp)
         * (= mirrors CPU 3-iter Tukey IRLS, scales O(blocks/WIs) instead
         * of O(blocks) single-WG.) */
        {
            /* Initialize params[P_S_SCALE] = s_init = sigma_sq / alpha. */
            float p_init[2];
            clEnqueueReadBuffer(queue, params_buf, CL_TRUE,
                                P_ALPHA * sizeof(float), 2 * sizeof(float),
                                p_init, 0, NULL, NULL);
            const float a_for_s = fmaxf(p_init[0], 1e-12f);
            const float s_init  = p_init[1] / a_for_s;
            const float s_min   = 0.05f * s_init;
            const float s_max   = 50.0f  * s_init;
            clEnqueueWriteBuffer(queue, params_buf, CL_TRUE,
                                 P_S_SCALE * sizeof(float), sizeof(float),
                                 &s_init, 0, NULL, NULL);

            const int n_wg = 64;   /* O32_DR_MWG_NWG */
            const int wgsize = 256; /* O32_DR_MWG_WGSIZE */
            const size_t dr_global2[1] = { (size_t)n_wg * wgsize };
            const size_t dr_local2[1]  = { (size_t)wgsize };

            for(int dr_iter = 0; dr_iter <= 2; dr_iter++)
            {
                /* §7.8b reduce */
                clSetKernelArg(k_o32_dr_reduce_mwg, 0, sizeof(cl_mem), &o32_in_gat_full);
                clSetKernelArg(k_o32_dr_reduce_mwg, 1, sizeof(cl_mem), &raw_buf);
                clSetKernelArg(k_o32_dr_reduce_mwg, 2, sizeof(int), &width);
                clSetKernelArg(k_o32_dr_reduce_mwg, 3, sizeof(int), &height);
                clSetKernelArg(k_o32_dr_reduce_mwg, 4, sizeof(cl_mem), &params_buf);
                clSetKernelArg(k_o32_dr_reduce_mwg, 5, sizeof(cl_mem), &partial_buf);
                cl_event ev_a;
                err = clEnqueueNDRangeKernel(queue, k_o32_dr_reduce_mwg, 1, NULL,
                                             dr_global2, dr_local2, 0, NULL, &ev_a);
                CL_CHECK(err, "K_O32_P2a_R reduce");
                prof_add(ev_a, dr_iter == 0 ? "K_O32_P2a_R0 dr_reduce" :
                              dr_iter == 1 ? "K_O32_P2a_R1 dr_reduce" :
                                             "K_O32_P2a_R2 dr_reduce");

                /* §7.8c finalize_dr */
                clSetKernelArg(k_o32_dr_finalize_dr_mwg, 0, sizeof(cl_mem), &partial_buf);
                clSetKernelArg(k_o32_dr_finalize_dr_mwg, 1, sizeof(int), &n_wg);
                clSetKernelArg(k_o32_dr_finalize_dr_mwg, 2, sizeof(cl_mem), &params_buf);
                dispatch_1d_named(queue, k_o32_dr_finalize_dr_mwg, 1, 0,
                                  dr_iter == 0 ? "K_O32_P2a_F0 dr_finalize" :
                                  dr_iter == 1 ? "K_O32_P2a_F1 dr_finalize" :
                                                 "K_O32_P2a_F2 dr_finalize");

                if(dr_iter == 2) break;

                /* §7.8d resid reduce */
                clSetKernelArg(k_o32_dr_resid_reduce_mwg, 0, sizeof(cl_mem), &o32_in_gat_full);
                clSetKernelArg(k_o32_dr_resid_reduce_mwg, 1, sizeof(cl_mem), &raw_buf);
                clSetKernelArg(k_o32_dr_resid_reduce_mwg, 2, sizeof(int), &width);
                clSetKernelArg(k_o32_dr_resid_reduce_mwg, 3, sizeof(int), &height);
                clSetKernelArg(k_o32_dr_resid_reduce_mwg, 4, sizeof(cl_mem), &params_buf);
                clSetKernelArg(k_o32_dr_resid_reduce_mwg, 5, sizeof(cl_mem), &partial_resid_buf);
                cl_event ev_b;
                err = clEnqueueNDRangeKernel(queue, k_o32_dr_resid_reduce_mwg, 1, NULL,
                                             dr_global2, dr_local2, 0, NULL, &ev_b);
                CL_CHECK(err, "K_O32_P2a_RR resid_reduce");
                prof_add(ev_b, dr_iter == 0 ? "K_O32_P2a_RR0 resid" :
                                              "K_O32_P2a_RR1 resid");

                /* §7.8e resid finalize → updates s_scale */
                clSetKernelArg(k_o32_dr_resid_finalize_mwg, 0, sizeof(cl_mem), &partial_resid_buf);
                clSetKernelArg(k_o32_dr_resid_finalize_mwg, 1, sizeof(int), &n_wg);
                clSetKernelArg(k_o32_dr_resid_finalize_mwg, 2, sizeof(float), &s_min);
                clSetKernelArg(k_o32_dr_resid_finalize_mwg, 3, sizeof(float), &s_max);
                clSetKernelArg(k_o32_dr_resid_finalize_mwg, 4, sizeof(cl_mem), &params_buf);
                dispatch_1d_named(queue, k_o32_dr_resid_finalize_mwg, 1, 0,
                                  dr_iter == 0 ? "K_O32_P2a_RF0 resid_fin" :
                                                 "K_O32_P2a_RF1 resid_fin");
            }
        }

        /* §7.9 dark_sub_full: in_gat_full (and ch0..3) -= dark_ref. */
        clSetKernelArg(k_o32_dark_sub_full, 0, sizeof(cl_mem), &o32_in_gat_full);
        clSetKernelArg(k_o32_dark_sub_full, 1, sizeof(cl_mem), &ch0_buf);
        clSetKernelArg(k_o32_dark_sub_full, 2, sizeof(cl_mem), &ch1_buf);
        clSetKernelArg(k_o32_dark_sub_full, 3, sizeof(cl_mem), &ch2_buf);
        clSetKernelArg(k_o32_dark_sub_full, 4, sizeof(cl_mem), &ch3_buf);
        clSetKernelArg(k_o32_dark_sub_full, 5, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_o32_dark_sub_full, 6, sizeof(int), &width);
        clSetKernelArg(k_o32_dark_sub_full, 7, sizeof(int), &height);
        dispatch_2d_named(queue, k_o32_dark_sub_full,
                    align_up(width, 16), align_up(height, 16), 16, 16,
                    "K_O32_P2b dark_sub");
    }
    else
    {
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
    }   /* end else (variant != 32) — K7-K10 */

    /* ================================================================
     * o32 variant pipeline branch — Phase 3-10 full FP32 mirror of CPU O.
     * Reuses Phase 0-2 (= K0a..K10 above), then runs:
     *   §6.1 galosh_o32_assemble_fullres_gat  (= reassemble + dark_sub)
     *   §6.2 galosh_o32_forward_l_stride1     (= Phase 3 L_cs)
     *   §6.3 galosh_o32_chroma_extract_halfres (= Phase 4 C1/2/3_h)
     * Phase 5-10 NOT YET IMPLEMENTED — for now copy in_gat_full to raw_buf
     * as the "output" so the pipeline produces a verifiable file (= same
     * numeric domain as CPU O Phase 2 end state, useful for diff vs CPU).
     * ================================================================ */
    if(variant == 32)
    {
        /* §6.1 assemble_fullres SKIPPED for o32: in_gat_full is now produced
         * directly by §7.4 (gat_forward) + §7.7 (normalize) + §7.9 (dark_sub).
         * Kept in galosh.cl as a dead path for now (= will remove after
         * verification harness confirms o32 end-to-end correctness). */

        /* §6.2: in_gat_full -> L_cs (FP32, Phase 3 forward L stride=1) */
        clSetKernelArg(k_o32_forward_l_stride1, 0, sizeof(cl_mem), &o32_in_gat_full);
        clSetKernelArg(k_o32_forward_l_stride1, 1, sizeof(cl_mem), &o32_L_cs);
        clSetKernelArg(k_o32_forward_l_stride1, 2, sizeof(int), &width);
        clSetKernelArg(k_o32_forward_l_stride1, 3, sizeof(int), &height);
        dispatch_2d_named(queue, k_o32_forward_l_stride1,
                          align_up(width, 16), align_up(height, 16), 16, 16,
                          "K_O32_2 fwd_L_cs");

        /* §6.3: in_gat_full -> C1/2/3_h (FP32, Phase 4 stride=2 chroma extract) */
        clSetKernelArg(k_o32_chroma_extract, 0, sizeof(cl_mem), &o32_in_gat_full);
        clSetKernelArg(k_o32_chroma_extract, 1, sizeof(cl_mem), &o32_C1_h);
        clSetKernelArg(k_o32_chroma_extract, 2, sizeof(cl_mem), &o32_C2_h);
        clSetKernelArg(k_o32_chroma_extract, 3, sizeof(cl_mem), &o32_C3_h);
        clSetKernelArg(k_o32_chroma_extract, 4, sizeof(int), &width);
        clSetKernelArg(k_o32_chroma_extract, 5, sizeof(int), &height);
        clSetKernelArg(k_o32_chroma_extract, 6, sizeof(int), &hw);
        clSetKernelArg(k_o32_chroma_extract, 7, sizeof(int), &hh);
        dispatch_2d_named(queue, k_o32_chroma_extract,
                          align_up(hw, 16), align_up(hh, 16), 16, 16,
                          "K_O32_3 chroma_extract");

        /* §6.5: L_cs -> L_cs_den (FP32, Phase 5 single-orient WHT-LOSH on L_cs).
         * Single fused pass12 kernel (no split / no DRAM pilot buffer):
         * pilot stays in tile-LDS, streaming-friendly + 0 extra DRAM.
         * Validity filter inside the block loop matches CPU's
         * iteration range [0, dim - BS] so zero-padded halo blocks are
         * never fed to BayesShrink/Wiener (= prior cause of image-edge
         * Phase 5 PSNR drop on dark crops). */
        {
            const float luma_strength_o32 = strength * luma_str;
            clSetKernelArg(k_o32_pass12, 0, sizeof(cl_mem), &o32_L_cs);
            clSetKernelArg(k_o32_pass12, 1, sizeof(cl_mem), &o32_L_cs_den);
            clSetKernelArg(k_o32_pass12, 2, sizeof(int), &width);
            clSetKernelArg(k_o32_pass12, 3, sizeof(int), &height);
            clSetKernelArg(k_o32_pass12, 4, sizeof(float), &luma_strength_o32);

            const int o32_tile_size = 28;   /* matches galosh.cl O32_TILE_SIZE */
            const int wgd = g_tile_wg_dim;  /* 8 */
            const size_t tiles_x = (width  + o32_tile_size - 1) / o32_tile_size;
            const size_t tiles_y = (height + o32_tile_size - 1) / o32_tile_size;
            const size_t global2[2] = { tiles_x * wgd, tiles_y * wgd };
            const size_t local2[2]  = { wgd, wgd };
            cl_event ev_o32_pass12;
            err = clEnqueueNDRangeKernel(queue, k_o32_pass12, 2, NULL,
                                         global2, local2, 0, NULL, &ev_o32_pass12);
            CL_CHECK(err, "dispatch K_O32_5 pass12");
            prof_add(ev_o32_pass12, "K_O32_5 pass12_L_fr");
        }

        /* §6.6+7 fused: L_cs_den -> L_pixel (full-res) + L_h_den (half-res, only at TL of 2x2). */
        clSetKernelArg(k_o32_lpixel_lh_den_fused, 0, sizeof(cl_mem), &o32_L_cs_den);
        clSetKernelArg(k_o32_lpixel_lh_den_fused, 1, sizeof(cl_mem), &o32_L_pixel);
        clSetKernelArg(k_o32_lpixel_lh_den_fused, 2, sizeof(cl_mem), &o32_L_h_den);
        clSetKernelArg(k_o32_lpixel_lh_den_fused, 3, sizeof(int), &width);
        clSetKernelArg(k_o32_lpixel_lh_den_fused, 4, sizeof(int), &height);
        clSetKernelArg(k_o32_lpixel_lh_den_fused, 5, sizeof(int), &hw);
        dispatch_2d_named(queue, k_o32_lpixel_lh_den_fused,
                          align_up(width, 16), align_up(height, 16), 16, 16,
                          "K_O32_6 L_pixel+L_h_den_fused");

        /* ========== Phase 7 ⭐ Multi-scale LOESS pyramid + K16 chain ⭐ ==========
         * Mirrors CPU galosh_raw_cpu.c gat_galosh_denoise_rawlc_o lines 4854-5097.
         * Builds 3 LOESS-denoised chroma estimates at progressively wider
         * effective receptive fields (~30 / 60 / 120 raw px) via L pyramid +
         * 3-channel LOESS + L-guided K16 EWA-JL3 upsample chain.
         * Anchors produced: C_h (Phase 4), C_loess_h, C_q_up, C_e_up. */
        const int cq_w = hw / 2;
        const int cq_h = hh / 2;
        const int ce_w = cq_w / 2;
        const int ce_h = cq_h / 2;
        const int kq_w = 2 * cq_w;     /* K16 q→h output stride */
        const int kq_h = 2 * cq_h;
        const int ke_w = 2 * ce_w;     /* K16 e→q output stride */
        const int ke_h = 2 * ce_h;

        if(cq_w < 4 || cq_h < 4 || ce_w < 4 || ce_h < 4)
        {
            fprintf(stderr, "[GPU][o32] image too small for 3-level pyramid "
                            "(cq=%dx%d, ce=%dx%d)\n", cq_w, cq_h, ce_w, ce_h);
            err = clEnqueueCopyBuffer(queue, o32_L_pixel, raw_buf, 0, 0,
                                      npix * sizeof(float), 0, NULL, NULL);
            CL_CHECK(err, "o32 stub: copy L_pixel (small image)");
            goto download_phase;
        }

        /* §6.8 box_down_2x: L_h_den -> L_q -> L_e (1ch L pyramid). */
        clSetKernelArg(k_o32_box_down_2x, 0, sizeof(cl_mem), &o32_L_h_den);
        clSetKernelArg(k_o32_box_down_2x, 1, sizeof(cl_mem), &o32_L_q);
        clSetKernelArg(k_o32_box_down_2x, 2, sizeof(int), &hw);
        clSetKernelArg(k_o32_box_down_2x, 3, sizeof(int), &hh);
        dispatch_2d_named(queue, k_o32_box_down_2x,
                          align_up(cq_w, 16), align_up(cq_h, 16), 16, 16,
                          "K_O32_7a L_q");
        clSetKernelArg(k_o32_box_down_2x, 0, sizeof(cl_mem), &o32_L_q);
        clSetKernelArg(k_o32_box_down_2x, 1, sizeof(cl_mem), &o32_L_e);
        clSetKernelArg(k_o32_box_down_2x, 2, sizeof(int), &cq_w);
        clSetKernelArg(k_o32_box_down_2x, 3, sizeof(int), &cq_h);
        dispatch_2d_named(queue, k_o32_box_down_2x,
                          align_up(ce_w, 16), align_up(ce_h, 16), 16, 16,
                          "K_O32_7b L_e");

        /* §6.9 box_down_2x_3p: C_h -> C_q -> C_e (3ch C pyramid). */
        clSetKernelArg(k_o32_box_down_2x_3p, 0, sizeof(cl_mem), &o32_C1_h);
        clSetKernelArg(k_o32_box_down_2x_3p, 1, sizeof(cl_mem), &o32_C2_h);
        clSetKernelArg(k_o32_box_down_2x_3p, 2, sizeof(cl_mem), &o32_C3_h);
        clSetKernelArg(k_o32_box_down_2x_3p, 3, sizeof(cl_mem), &o32_C1_q);
        clSetKernelArg(k_o32_box_down_2x_3p, 4, sizeof(cl_mem), &o32_C2_q);
        clSetKernelArg(k_o32_box_down_2x_3p, 5, sizeof(cl_mem), &o32_C3_q);
        clSetKernelArg(k_o32_box_down_2x_3p, 6, sizeof(int), &hw);
        clSetKernelArg(k_o32_box_down_2x_3p, 7, sizeof(int), &hh);
        dispatch_2d_named(queue, k_o32_box_down_2x_3p,
                          align_up(cq_w, 16), align_up(cq_h, 16), 16, 16,
                          "K_O32_7c C_q");
        clSetKernelArg(k_o32_box_down_2x_3p, 0, sizeof(cl_mem), &o32_C1_q);
        clSetKernelArg(k_o32_box_down_2x_3p, 1, sizeof(cl_mem), &o32_C2_q);
        clSetKernelArg(k_o32_box_down_2x_3p, 2, sizeof(cl_mem), &o32_C3_q);
        clSetKernelArg(k_o32_box_down_2x_3p, 3, sizeof(cl_mem), &o32_C1_e);
        clSetKernelArg(k_o32_box_down_2x_3p, 4, sizeof(cl_mem), &o32_C2_e);
        clSetKernelArg(k_o32_box_down_2x_3p, 5, sizeof(cl_mem), &o32_C3_e);
        clSetKernelArg(k_o32_box_down_2x_3p, 6, sizeof(int), &cq_w);
        clSetKernelArg(k_o32_box_down_2x_3p, 7, sizeof(int), &cq_h);
        dispatch_2d_named(queue, k_o32_box_down_2x_3p,
                          align_up(ce_w, 16), align_up(ce_h, 16), 16, 16,
                          "K_O32_7d C_e");

        /* §6.10b LOESS at 3 scales (LDS-tiled FP32, R=7/BW=3.0/strength_c=1.0).
         * Tile: 16×16 WG with R-halo → 30² LDS = 14.4 KB FP32.  Per-WI 14
         * global reads vs 225 in naive — critical for 4MP+ scaling. */
        const float loess_strength = 1.0f;
        /* LOESS half-res: (L_h_den, C_h) -> C_loess_h */
        clSetKernelArg(k_o32_loess_3p_tiled, 0, sizeof(cl_mem), &o32_L_h_den);
        clSetKernelArg(k_o32_loess_3p_tiled, 1, sizeof(cl_mem), &o32_C1_h);
        clSetKernelArg(k_o32_loess_3p_tiled, 2, sizeof(cl_mem), &o32_C2_h);
        clSetKernelArg(k_o32_loess_3p_tiled, 3, sizeof(cl_mem), &o32_C3_h);
        clSetKernelArg(k_o32_loess_3p_tiled, 4, sizeof(cl_mem), &o32_C1_loess_h);
        clSetKernelArg(k_o32_loess_3p_tiled, 5, sizeof(cl_mem), &o32_C2_loess_h);
        clSetKernelArg(k_o32_loess_3p_tiled, 6, sizeof(cl_mem), &o32_C3_loess_h);
        clSetKernelArg(k_o32_loess_3p_tiled, 7, sizeof(int), &hw);
        clSetKernelArg(k_o32_loess_3p_tiled, 8, sizeof(int), &hh);
        clSetKernelArg(k_o32_loess_3p_tiled, 9, sizeof(float), &loess_strength);
        dispatch_2d_named(queue, k_o32_loess_3p_tiled,
                          align_up(hw, 24), align_up(hh, 24), 24, 24,
                          "K_O32_7e LOESS_h_t");
        /* LOESS quarter-res: (L_q, C_q) -> C_loess_q */
        clSetKernelArg(k_o32_loess_3p_tiled, 0, sizeof(cl_mem), &o32_L_q);
        clSetKernelArg(k_o32_loess_3p_tiled, 1, sizeof(cl_mem), &o32_C1_q);
        clSetKernelArg(k_o32_loess_3p_tiled, 2, sizeof(cl_mem), &o32_C2_q);
        clSetKernelArg(k_o32_loess_3p_tiled, 3, sizeof(cl_mem), &o32_C3_q);
        clSetKernelArg(k_o32_loess_3p_tiled, 4, sizeof(cl_mem), &o32_C1_loess_q);
        clSetKernelArg(k_o32_loess_3p_tiled, 5, sizeof(cl_mem), &o32_C2_loess_q);
        clSetKernelArg(k_o32_loess_3p_tiled, 6, sizeof(cl_mem), &o32_C3_loess_q);
        clSetKernelArg(k_o32_loess_3p_tiled, 7, sizeof(int), &cq_w);
        clSetKernelArg(k_o32_loess_3p_tiled, 8, sizeof(int), &cq_h);
        dispatch_2d_named(queue, k_o32_loess_3p_tiled,
                          align_up(cq_w, 24), align_up(cq_h, 24), 24, 24,
                          "K_O32_7f LOESS_q_t");
        /* LOESS eighth-res: (L_e, C_e) -> C_loess_e */
        clSetKernelArg(k_o32_loess_3p_tiled, 0, sizeof(cl_mem), &o32_L_e);
        clSetKernelArg(k_o32_loess_3p_tiled, 1, sizeof(cl_mem), &o32_C1_e);
        clSetKernelArg(k_o32_loess_3p_tiled, 2, sizeof(cl_mem), &o32_C2_e);
        clSetKernelArg(k_o32_loess_3p_tiled, 3, sizeof(cl_mem), &o32_C3_e);
        clSetKernelArg(k_o32_loess_3p_tiled, 4, sizeof(cl_mem), &o32_C1_loess_e);
        clSetKernelArg(k_o32_loess_3p_tiled, 5, sizeof(cl_mem), &o32_C2_loess_e);
        clSetKernelArg(k_o32_loess_3p_tiled, 6, sizeof(cl_mem), &o32_C3_loess_e);
        clSetKernelArg(k_o32_loess_3p_tiled, 7, sizeof(int), &ce_w);
        clSetKernelArg(k_o32_loess_3p_tiled, 8, sizeof(int), &ce_h);
        dispatch_2d_named(queue, k_o32_loess_3p_tiled,
                          align_up(ce_w, 24), align_up(ce_h, 24), 24, 24,
                          "K_O32_7g LOESS_e_t");

        /* §6.13 crop L_h_den (chsize) -> L_for_q (kq stride) for q→h K16. */
        clSetKernelArg(k_o32_crop_2d_topleft, 0, sizeof(cl_mem), &o32_L_h_den);
        clSetKernelArg(k_o32_crop_2d_topleft, 1, sizeof(cl_mem), &o32_L_for_q);
        clSetKernelArg(k_o32_crop_2d_topleft, 2, sizeof(int), &hw);
        clSetKernelArg(k_o32_crop_2d_topleft, 3, sizeof(int), &hh);
        clSetKernelArg(k_o32_crop_2d_topleft, 4, sizeof(int), &kq_w);
        clSetKernelArg(k_o32_crop_2d_topleft, 5, sizeof(int), &kq_h);
        dispatch_2d_named(queue, k_o32_crop_2d_topleft,
                          align_up(kq_w, 16), align_up(kq_h, 16), 16, 16,
                          "K_O32_7h crop_L_for_q");

        /* §6.11 K16 q→h: (C_loess_q, L_for_q) -> C_q_up_raw (kq stride). */
        const float k16_BW = 1.5f;
        clSetKernelArg(k_o32_k16_jbu_3p, 0, sizeof(cl_mem), &o32_C1_loess_q);
        clSetKernelArg(k_o32_k16_jbu_3p, 1, sizeof(cl_mem), &o32_C2_loess_q);
        clSetKernelArg(k_o32_k16_jbu_3p, 2, sizeof(cl_mem), &o32_C3_loess_q);
        clSetKernelArg(k_o32_k16_jbu_3p, 3, sizeof(cl_mem), &o32_L_for_q);
        clSetKernelArg(k_o32_k16_jbu_3p, 4, sizeof(cl_mem), &o32_C1_q_up_raw);
        clSetKernelArg(k_o32_k16_jbu_3p, 5, sizeof(cl_mem), &o32_C2_q_up_raw);
        clSetKernelArg(k_o32_k16_jbu_3p, 6, sizeof(cl_mem), &o32_C3_q_up_raw);
        clSetKernelArg(k_o32_k16_jbu_3p, 7, sizeof(int), &cq_w);
        clSetKernelArg(k_o32_k16_jbu_3p, 8, sizeof(int), &cq_h);
        clSetKernelArg(k_o32_k16_jbu_3p, 9, sizeof(float), &k16_BW);
        dispatch_2d_named(queue, k_o32_k16_jbu_3p,
                          align_up(kq_w, 16), align_up(kq_h, 16), 16, 16,
                          "K_O32_7i K16_q2h");

        /* §6.12 pad C_q_up_raw (kq stride) -> C_q_up (chsize, edge replicate). */
        for(int p = 0; p < 3; p++)
        {
            cl_mem src = (p == 0) ? o32_C1_q_up_raw : (p == 1) ? o32_C2_q_up_raw : o32_C3_q_up_raw;
            cl_mem dst = (p == 0) ? o32_C1_q_up      : (p == 1) ? o32_C2_q_up      : o32_C3_q_up;
            clSetKernelArg(k_o32_pad_2d_edge, 0, sizeof(cl_mem), &src);
            clSetKernelArg(k_o32_pad_2d_edge, 1, sizeof(cl_mem), &dst);
            clSetKernelArg(k_o32_pad_2d_edge, 2, sizeof(int), &kq_w);
            clSetKernelArg(k_o32_pad_2d_edge, 3, sizeof(int), &kq_h);
            clSetKernelArg(k_o32_pad_2d_edge, 4, sizeof(int), &hw);
            clSetKernelArg(k_o32_pad_2d_edge, 5, sizeof(int), &hh);
            dispatch_2d_named(queue, k_o32_pad_2d_edge,
                              align_up(hw, 16), align_up(hh, 16), 16, 16,
                              "K_O32_7j pad_q_up");
        }

        /* §6.13 crop L_q (cqsize) -> L_for_e (ke stride) for e→q K16. */
        clSetKernelArg(k_o32_crop_2d_topleft, 0, sizeof(cl_mem), &o32_L_q);
        clSetKernelArg(k_o32_crop_2d_topleft, 1, sizeof(cl_mem), &o32_L_for_e);
        clSetKernelArg(k_o32_crop_2d_topleft, 2, sizeof(int), &cq_w);
        clSetKernelArg(k_o32_crop_2d_topleft, 3, sizeof(int), &cq_h);
        clSetKernelArg(k_o32_crop_2d_topleft, 4, sizeof(int), &ke_w);
        clSetKernelArg(k_o32_crop_2d_topleft, 5, sizeof(int), &ke_h);
        dispatch_2d_named(queue, k_o32_crop_2d_topleft,
                          align_up(ke_w, 16), align_up(ke_h, 16), 16, 16,
                          "K_O32_7k crop_L_for_e");

        /* §6.11 K16 e→q: (C_loess_e, L_for_e) -> C_e_to_q_raw (ke stride). */
        clSetKernelArg(k_o32_k16_jbu_3p, 0, sizeof(cl_mem), &o32_C1_loess_e);
        clSetKernelArg(k_o32_k16_jbu_3p, 1, sizeof(cl_mem), &o32_C2_loess_e);
        clSetKernelArg(k_o32_k16_jbu_3p, 2, sizeof(cl_mem), &o32_C3_loess_e);
        clSetKernelArg(k_o32_k16_jbu_3p, 3, sizeof(cl_mem), &o32_L_for_e);
        clSetKernelArg(k_o32_k16_jbu_3p, 4, sizeof(cl_mem), &o32_C1_e_to_q_raw);
        clSetKernelArg(k_o32_k16_jbu_3p, 5, sizeof(cl_mem), &o32_C2_e_to_q_raw);
        clSetKernelArg(k_o32_k16_jbu_3p, 6, sizeof(cl_mem), &o32_C3_e_to_q_raw);
        clSetKernelArg(k_o32_k16_jbu_3p, 7, sizeof(int), &ce_w);
        clSetKernelArg(k_o32_k16_jbu_3p, 8, sizeof(int), &ce_h);
        clSetKernelArg(k_o32_k16_jbu_3p, 9, sizeof(float), &k16_BW);
        dispatch_2d_named(queue, k_o32_k16_jbu_3p,
                          align_up(ke_w, 16), align_up(ke_h, 16), 16, 16,
                          "K_O32_7l K16_e2q");

        /* §6.12 pad C_e_to_q_raw (ke stride) -> C_e_to_q (cqsize). */
        for(int p = 0; p < 3; p++)
        {
            cl_mem src = (p == 0) ? o32_C1_e_to_q_raw : (p == 1) ? o32_C2_e_to_q_raw : o32_C3_e_to_q_raw;
            cl_mem dst = (p == 0) ? o32_C1_e_to_q      : (p == 1) ? o32_C2_e_to_q      : o32_C3_e_to_q;
            clSetKernelArg(k_o32_pad_2d_edge, 0, sizeof(cl_mem), &src);
            clSetKernelArg(k_o32_pad_2d_edge, 1, sizeof(cl_mem), &dst);
            clSetKernelArg(k_o32_pad_2d_edge, 2, sizeof(int), &ke_w);
            clSetKernelArg(k_o32_pad_2d_edge, 3, sizeof(int), &ke_h);
            clSetKernelArg(k_o32_pad_2d_edge, 4, sizeof(int), &cq_w);
            clSetKernelArg(k_o32_pad_2d_edge, 5, sizeof(int), &cq_h);
            dispatch_2d_named(queue, k_o32_pad_2d_edge,
                              align_up(cq_w, 16), align_up(cq_h, 16), 16, 16,
                              "K_O32_7m pad_e_to_q");
        }

        /* §6.11 K16 q→h again on C_e_to_q with L_for_q guide -> C_e_up_raw. */
        clSetKernelArg(k_o32_k16_jbu_3p, 0, sizeof(cl_mem), &o32_C1_e_to_q);
        clSetKernelArg(k_o32_k16_jbu_3p, 1, sizeof(cl_mem), &o32_C2_e_to_q);
        clSetKernelArg(k_o32_k16_jbu_3p, 2, sizeof(cl_mem), &o32_C3_e_to_q);
        clSetKernelArg(k_o32_k16_jbu_3p, 3, sizeof(cl_mem), &o32_L_for_q);
        clSetKernelArg(k_o32_k16_jbu_3p, 4, sizeof(cl_mem), &o32_C1_e_up_raw);
        clSetKernelArg(k_o32_k16_jbu_3p, 5, sizeof(cl_mem), &o32_C2_e_up_raw);
        clSetKernelArg(k_o32_k16_jbu_3p, 6, sizeof(cl_mem), &o32_C3_e_up_raw);
        clSetKernelArg(k_o32_k16_jbu_3p, 7, sizeof(int), &cq_w);
        clSetKernelArg(k_o32_k16_jbu_3p, 8, sizeof(int), &cq_h);
        clSetKernelArg(k_o32_k16_jbu_3p, 9, sizeof(float), &k16_BW);
        dispatch_2d_named(queue, k_o32_k16_jbu_3p,
                          align_up(kq_w, 16), align_up(kq_h, 16), 16, 16,
                          "K_O32_7n K16_q2h_e");

        /* §6.12 pad C_e_up_raw (kq stride) -> C_e_up (chsize). */
        for(int p = 0; p < 3; p++)
        {
            cl_mem src = (p == 0) ? o32_C1_e_up_raw : (p == 1) ? o32_C2_e_up_raw : o32_C3_e_up_raw;
            cl_mem dst = (p == 0) ? o32_C1_e_up      : (p == 1) ? o32_C2_e_up      : o32_C3_e_up;
            clSetKernelArg(k_o32_pad_2d_edge, 0, sizeof(cl_mem), &src);
            clSetKernelArg(k_o32_pad_2d_edge, 1, sizeof(cl_mem), &dst);
            clSetKernelArg(k_o32_pad_2d_edge, 2, sizeof(int), &kq_w);
            clSetKernelArg(k_o32_pad_2d_edge, 3, sizeof(int), &kq_h);
            clSetKernelArg(k_o32_pad_2d_edge, 4, sizeof(int), &hw);
            clSetKernelArg(k_o32_pad_2d_edge, 5, sizeof(int), &hh);
            dispatch_2d_named(queue, k_o32_pad_2d_edge,
                              align_up(hw, 16), align_up(hh, 16), 16, 16,
                              "K_O32_7o pad_e_up");
        }

        /* §6.14 Phase 8 smoothstep slider walk (4 anchors -> C_h_den).
         * slider = strength * chroma_str (= same convention as G K_SP). */
        const float slider = strength * chroma_str;
        clSetKernelArg(k_o32_smoothstep_3p,  0, sizeof(cl_mem), &o32_C1_h);
        clSetKernelArg(k_o32_smoothstep_3p,  1, sizeof(cl_mem), &o32_C2_h);
        clSetKernelArg(k_o32_smoothstep_3p,  2, sizeof(cl_mem), &o32_C3_h);
        clSetKernelArg(k_o32_smoothstep_3p,  3, sizeof(cl_mem), &o32_C1_loess_h);
        clSetKernelArg(k_o32_smoothstep_3p,  4, sizeof(cl_mem), &o32_C2_loess_h);
        clSetKernelArg(k_o32_smoothstep_3p,  5, sizeof(cl_mem), &o32_C3_loess_h);
        clSetKernelArg(k_o32_smoothstep_3p,  6, sizeof(cl_mem), &o32_C1_q_up);
        clSetKernelArg(k_o32_smoothstep_3p,  7, sizeof(cl_mem), &o32_C2_q_up);
        clSetKernelArg(k_o32_smoothstep_3p,  8, sizeof(cl_mem), &o32_C3_q_up);
        clSetKernelArg(k_o32_smoothstep_3p,  9, sizeof(cl_mem), &o32_C1_e_up);
        clSetKernelArg(k_o32_smoothstep_3p, 10, sizeof(cl_mem), &o32_C2_e_up);
        clSetKernelArg(k_o32_smoothstep_3p, 11, sizeof(cl_mem), &o32_C3_e_up);
        clSetKernelArg(k_o32_smoothstep_3p, 12, sizeof(cl_mem), &o32_C1_h_den);
        clSetKernelArg(k_o32_smoothstep_3p, 13, sizeof(cl_mem), &o32_C2_h_den);
        clSetKernelArg(k_o32_smoothstep_3p, 14, sizeof(cl_mem), &o32_C3_h_den);
        clSetKernelArg(k_o32_smoothstep_3p, 15, sizeof(int), &hw);
        clSetKernelArg(k_o32_smoothstep_3p, 16, sizeof(int), &hh);
        clSetKernelArg(k_o32_smoothstep_3p, 17, sizeof(float), &slider);
        dispatch_2d_named(queue, k_o32_smoothstep_3p,
                          align_up(hw, 16), align_up(hh, 16), 16, 16,
                          "K_O32_8 smoothstep");

        /* §6.11 Phase 9 K16 final upsample (C_h_den + L_pixel guide -> C_aligned full-res).
         * K16 output dim = 2*hw × 2*hh = width × height (since width = 2*hw exactly). */
        clSetKernelArg(k_o32_k16_jbu_3p, 0, sizeof(cl_mem), &o32_C1_h_den);
        clSetKernelArg(k_o32_k16_jbu_3p, 1, sizeof(cl_mem), &o32_C2_h_den);
        clSetKernelArg(k_o32_k16_jbu_3p, 2, sizeof(cl_mem), &o32_C3_h_den);
        clSetKernelArg(k_o32_k16_jbu_3p, 3, sizeof(cl_mem), &o32_L_pixel);
        clSetKernelArg(k_o32_k16_jbu_3p, 4, sizeof(cl_mem), &o32_C1_aligned);
        clSetKernelArg(k_o32_k16_jbu_3p, 5, sizeof(cl_mem), &o32_C2_aligned);
        clSetKernelArg(k_o32_k16_jbu_3p, 6, sizeof(cl_mem), &o32_C3_aligned);
        clSetKernelArg(k_o32_k16_jbu_3p, 7, sizeof(int), &hw);
        clSetKernelArg(k_o32_k16_jbu_3p, 8, sizeof(int), &hh);
        clSetKernelArg(k_o32_k16_jbu_3p, 9, sizeof(float), &k16_BW);
        dispatch_2d_named(queue, k_o32_k16_jbu_3p,
                          align_up(width, 16), align_up(height, 16), 16, 16,
                          "K_O32_9 K16_final");

        /* §6.15 Phase 10 inverse WHT + dark restore + ×unified_sigma + inv-GAT
         * -> raw_buf (= final FP32 denoised image). */
        clSetKernelArg(k_o32_inverse_wht_dark, 0, sizeof(cl_mem), &o32_L_pixel);
        clSetKernelArg(k_o32_inverse_wht_dark, 1, sizeof(cl_mem), &o32_C1_aligned);
        clSetKernelArg(k_o32_inverse_wht_dark, 2, sizeof(cl_mem), &o32_C2_aligned);
        clSetKernelArg(k_o32_inverse_wht_dark, 3, sizeof(cl_mem), &o32_C3_aligned);
        clSetKernelArg(k_o32_inverse_wht_dark, 4, sizeof(cl_mem), &raw_buf);
        clSetKernelArg(k_o32_inverse_wht_dark, 5, sizeof(cl_mem), &lut_d_buf);
        clSetKernelArg(k_o32_inverse_wht_dark, 6, sizeof(cl_mem), &lut_x_buf);
        clSetKernelArg(k_o32_inverse_wht_dark, 7, sizeof(cl_mem), &lut_params_buf);
        clSetKernelArg(k_o32_inverse_wht_dark, 8, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_o32_inverse_wht_dark, 9, sizeof(int), &width);
        clSetKernelArg(k_o32_inverse_wht_dark, 10, sizeof(int), &height);
        dispatch_2d_named(queue, k_o32_inverse_wht_dark,
                          align_up(width, 16), align_up(height, 16), 16, 16,
                          "K_O32_10 inverse_final");

        fprintf(stderr, "[GPU][o32] FULL PIPELINE Phase 0-10 done (= CPU O FP32 mirror)\n");

        /* Verification harness: dump all key intermediates if GALOSH_DUMP_DIR set. */
        {
            const char *dump_dir = getenv("GALOSH_DUMP_DIR");
            if(dump_dir)
            {
                clFinish(queue);   /* ensure all kernels completed before readback */
                fprintf(stderr, "[GPU][o32] dumping intermediates to %s\n", dump_dir);
                float *tmpf = alloc_float(npix);
                float *tmph = alloc_float(chsize);
                if(tmpf && tmph)
                {
                    char path[1024];
#define O32_DUMP(name, buf, size_floats) do { \
    clEnqueueReadBuffer(queue, (buf), CL_TRUE, 0, (size_floats) * sizeof(float), \
                        ((size_floats) == npix) ? tmpf : tmph, 0, NULL, NULL); \
    snprintf(path, sizeof(path), "%s/%s.bin", dump_dir, name); \
    FILE *df = fopen(path, "wb"); \
    if(df) { \
        fwrite(((size_floats) == npix) ? tmpf : tmph, sizeof(float), \
               (size_floats), df); \
        fclose(df); \
    } \
} while(0)
                    /* Phase 5 Pass1-only re-run (debug-only): re-runs BayesShrink
                     * on o32_L_cs and writes pilot to a temporary global buffer.
                     * Production pipeline (above, k_o32_pass12) is unchanged. */
                    cl_mem o32_pilot_debug = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                            npix * sizeof(float), NULL, &err);
                    if(err == CL_SUCCESS && o32_pilot_debug)
                    {
                        const float luma_strength_o32 = strength * luma_str;
                        clSetKernelArg(k_o32_pass1_dump, 0, sizeof(cl_mem), &o32_L_cs);
                        clSetKernelArg(k_o32_pass1_dump, 1, sizeof(cl_mem), &o32_pilot_debug);
                        clSetKernelArg(k_o32_pass1_dump, 2, sizeof(int), &width);
                        clSetKernelArg(k_o32_pass1_dump, 3, sizeof(int), &height);
                        clSetKernelArg(k_o32_pass1_dump, 4, sizeof(float), &luma_strength_o32);
                        const int o32_tile_size_dbg = 28;
                        const int wgd_dbg = g_tile_wg_dim;
                        const size_t tx_dbg = (width  + o32_tile_size_dbg - 1) / o32_tile_size_dbg;
                        const size_t ty_dbg = (height + o32_tile_size_dbg - 1) / o32_tile_size_dbg;
                        const size_t g2[2] = { tx_dbg * wgd_dbg, ty_dbg * wgd_dbg };
                        const size_t l2[2] = { wgd_dbg, wgd_dbg };
                        clEnqueueNDRangeKernel(queue, k_o32_pass1_dump, 2, NULL, g2, l2, 0, NULL, NULL);
                        clFinish(queue);
                        O32_DUMP("p5_pilot", o32_pilot_debug, npix);
                        clReleaseMemObject(o32_pilot_debug);
                    }
                    O32_DUMP("p2_in_gat",    o32_in_gat_full, npix);
                    O32_DUMP("p3_L_cs",      o32_L_cs,        npix);
                    O32_DUMP("p4_C1_h",      o32_C1_h,        chsize);
                    O32_DUMP("p4_C2_h",      o32_C2_h,        chsize);
                    O32_DUMP("p4_C3_h",      o32_C3_h,        chsize);
                    O32_DUMP("p5_L_cs_den",  o32_L_cs_den,    npix);
                    O32_DUMP("p6_L_pixel",   o32_L_pixel,     npix);
                    O32_DUMP("p6_L_h_den",   o32_L_h_den,     chsize);
                    O32_DUMP("p7_C1_loess_h", o32_C1_loess_h, chsize);
                    O32_DUMP("p7_C1_q_up",   o32_C1_q_up,     chsize);
                    O32_DUMP("p7_C1_e_up",   o32_C1_e_up,     chsize);
                    O32_DUMP("p8_C1_h_den",  o32_C1_h_den,    chsize);
                    O32_DUMP("p9_C1_aligned", o32_C1_aligned, npix);
                    O32_DUMP("p10_output",   raw_buf,         npix);
#undef O32_DUMP
                    fprintf(stderr, "[GPU][o32] dump complete (14 .bin files)\n");
                }
                free_aligned(tmpf);
                free_aligned(tmph);
            }
        }

        goto download_phase;
    }

    /* O variant pipeline branch archived 2026-05-09 to archived_o-broke/. */


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

    /* ---- K16: Reconstruct (per-pixel inverse 2x2 WHT + EWA-JL3 chroma upsample
     *           + sigma-denormalize + inverse GAT, all fused in
     *           galosh_reconstruct_chromaup).  Mirrors CPU galosh_raw_cpu.c
     *           Phase 4 step (d).  No intermediate full-res chroma buffers
     *           — the kernel reads c1/c2/c3 half-res FP16 with a 5x5 EWA-JL3
     *           neighbourhood per output pixel and does the inverse WHT
     *           in-place.  ---- */
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

download_phase:
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

    /* o32 kernel cleanup */
    if(k_o32_assemble_fullres)  clReleaseKernel(k_o32_assemble_fullres);
    if(k_o32_forward_l_stride1) clReleaseKernel(k_o32_forward_l_stride1);
    if(k_o32_chroma_extract)    clReleaseKernel(k_o32_chroma_extract);
    if(k_o32_pass12)            clReleaseKernel(k_o32_pass12);
    if(k_o32_pass1_dump)        clReleaseKernel(k_o32_pass1_dump);
    if(k_o32_lpixel_overlap)    clReleaseKernel(k_o32_lpixel_overlap);
    if(k_o32_l_h_den_subsample) clReleaseKernel(k_o32_l_h_den_subsample);
    if(k_o32_lpixel_lh_den_fused) clReleaseKernel(k_o32_lpixel_lh_den_fused);
    if(k_o32_box_down_2x)       clReleaseKernel(k_o32_box_down_2x);
    if(k_o32_box_down_2x_3p)    clReleaseKernel(k_o32_box_down_2x_3p);
    if(k_o32_loess_3p)          clReleaseKernel(k_o32_loess_3p);
    if(k_o32_loess_3p_tiled)    clReleaseKernel(k_o32_loess_3p_tiled);
    if(k_o32_k16_jbu_3p)        clReleaseKernel(k_o32_k16_jbu_3p);
    if(k_o32_pad_2d_edge)       clReleaseKernel(k_o32_pad_2d_edge);
    if(k_o32_crop_2d_topleft)   clReleaseKernel(k_o32_crop_2d_topleft);
    if(k_o32_smoothstep_3p)     clReleaseKernel(k_o32_smoothstep_3p);
    if(k_o32_inverse_wht_dark)  clReleaseKernel(k_o32_inverse_wht_dark);
    if(k_o32_ne_block_stats)    clReleaseKernel(k_o32_ne_block_stats);
    if(k_o32_ne_finalize)       clReleaseKernel(k_o32_ne_finalize);
    if(k_o32_build_inv_lut)     clReleaseKernel(k_o32_build_inv_lut);
    if(k_o32_lut_finalize)      clReleaseKernel(k_o32_lut_finalize);
    if(k_o32_ne_dark_thresh_hist)     clReleaseKernel(k_o32_ne_dark_thresh_hist);
    if(k_o32_ne_dark_thresh_finalize) clReleaseKernel(k_o32_ne_dark_thresh_finalize);
    if(k_o32_ne_dark_lap_hist)        clReleaseKernel(k_o32_ne_dark_lap_hist);
    if(k_o32_ne_dark_finalize)        clReleaseKernel(k_o32_ne_dark_finalize);
    if(k_o32_gat_forward_full)  clReleaseKernel(k_o32_gat_forward_full);
    if(k_o32_sigma_per_cfa)     clReleaseKernel(k_o32_sigma_per_cfa);
    if(k_o32_unified_sigma)     clReleaseKernel(k_o32_unified_sigma);
    if(k_o32_normalize_apply)   clReleaseKernel(k_o32_normalize_apply);
    if(k_o32_dark_ref_irls)     clReleaseKernel(k_o32_dark_ref_irls);
    if(k_o32_dr_reduce_mwg)     clReleaseKernel(k_o32_dr_reduce_mwg);
    if(k_o32_dr_finalize_dr_mwg) clReleaseKernel(k_o32_dr_finalize_dr_mwg);
    if(k_o32_dr_resid_reduce_mwg) clReleaseKernel(k_o32_dr_resid_reduce_mwg);
    if(k_o32_dr_resid_finalize_mwg) clReleaseKernel(k_o32_dr_resid_finalize_mwg);
    if(k_o32_dark_sub_full)     clReleaseKernel(k_o32_dark_sub_full);

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

    /* o32 buffer cleanup */
    if(o32_in_gat_full) clReleaseMemObject(o32_in_gat_full);
    if(o32_L_cs)        clReleaseMemObject(o32_L_cs);
    if(o32_L_cs_den)    clReleaseMemObject(o32_L_cs_den);
    if(o32_L_pixel)     clReleaseMemObject(o32_L_pixel);
    if(o32_L_h_den)     clReleaseMemObject(o32_L_h_den);
    if(o32_C1_h)        clReleaseMemObject(o32_C1_h);
    if(o32_C2_h)        clReleaseMemObject(o32_C2_h);
    if(o32_C3_h)        clReleaseMemObject(o32_C3_h);
    /* Phase 7 pyramid buffers */
    if(o32_L_q)         clReleaseMemObject(o32_L_q);
    if(o32_L_e)         clReleaseMemObject(o32_L_e);
    if(o32_L_for_q)     clReleaseMemObject(o32_L_for_q);
    if(o32_L_for_e)     clReleaseMemObject(o32_L_for_e);
    if(o32_C1_q)        clReleaseMemObject(o32_C1_q);
    if(o32_C2_q)        clReleaseMemObject(o32_C2_q);
    if(o32_C3_q)        clReleaseMemObject(o32_C3_q);
    if(o32_C1_e)        clReleaseMemObject(o32_C1_e);
    if(o32_C2_e)        clReleaseMemObject(o32_C2_e);
    if(o32_C3_e)        clReleaseMemObject(o32_C3_e);
    if(o32_C1_loess_h)  clReleaseMemObject(o32_C1_loess_h);
    if(o32_C2_loess_h)  clReleaseMemObject(o32_C2_loess_h);
    if(o32_C3_loess_h)  clReleaseMemObject(o32_C3_loess_h);
    if(o32_C1_loess_q)  clReleaseMemObject(o32_C1_loess_q);
    if(o32_C2_loess_q)  clReleaseMemObject(o32_C2_loess_q);
    if(o32_C3_loess_q)  clReleaseMemObject(o32_C3_loess_q);
    if(o32_C1_loess_e)  clReleaseMemObject(o32_C1_loess_e);
    if(o32_C2_loess_e)  clReleaseMemObject(o32_C2_loess_e);
    if(o32_C3_loess_e)  clReleaseMemObject(o32_C3_loess_e);
    if(o32_C1_q_up_raw) clReleaseMemObject(o32_C1_q_up_raw);
    if(o32_C2_q_up_raw) clReleaseMemObject(o32_C2_q_up_raw);
    if(o32_C3_q_up_raw) clReleaseMemObject(o32_C3_q_up_raw);
    if(o32_C1_q_up)     clReleaseMemObject(o32_C1_q_up);
    if(o32_C2_q_up)     clReleaseMemObject(o32_C2_q_up);
    if(o32_C3_q_up)     clReleaseMemObject(o32_C3_q_up);
    if(o32_C1_e_to_q_raw) clReleaseMemObject(o32_C1_e_to_q_raw);
    if(o32_C2_e_to_q_raw) clReleaseMemObject(o32_C2_e_to_q_raw);
    if(o32_C3_e_to_q_raw) clReleaseMemObject(o32_C3_e_to_q_raw);
    if(o32_C1_e_to_q)   clReleaseMemObject(o32_C1_e_to_q);
    if(o32_C2_e_to_q)   clReleaseMemObject(o32_C2_e_to_q);
    if(o32_C3_e_to_q)   clReleaseMemObject(o32_C3_e_to_q);
    if(o32_C1_e_up_raw) clReleaseMemObject(o32_C1_e_up_raw);
    if(o32_C2_e_up_raw) clReleaseMemObject(o32_C2_e_up_raw);
    if(o32_C3_e_up_raw) clReleaseMemObject(o32_C3_e_up_raw);
    if(o32_C1_e_up)     clReleaseMemObject(o32_C1_e_up);
    if(o32_C2_e_up)     clReleaseMemObject(o32_C2_e_up);
    if(o32_C3_e_up)     clReleaseMemObject(o32_C3_e_up);
    if(o32_C1_h_den)    clReleaseMemObject(o32_C1_h_den);
    if(o32_C2_h_den)    clReleaseMemObject(o32_C2_h_den);
    if(o32_C3_h_den)    clReleaseMemObject(o32_C3_h_den);
    if(o32_C1_aligned)  clReleaseMemObject(o32_C1_aligned);
    if(o32_C2_aligned)  clReleaseMemObject(o32_C2_aligned);
    if(o32_C3_aligned)  clReleaseMemObject(o32_C3_aligned);
    if(o32_blk_mean)    clReleaseMemObject(o32_blk_mean);
    if(o32_blk_var)     clReleaseMemObject(o32_blk_var);
    if(o32_dark_thresh_hist) clReleaseMemObject(o32_dark_thresh_hist);
    if(o32_dark_lap_hist)    clReleaseMemObject(o32_dark_lap_hist);

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
            "       <alpha> <sigma_sq> [cl_dev] [--variant=g|o32]\n"
            "Variants:\n"
            "  g    GALOSH_RAW_G (default, production half-res LOSH + EWA-JL3 chromaup)\n"
            "  o32  GALOSH_RAW_O FP32 GPU port (= mirror of CPU --variant=o)\n"
            "       AS OF 2026-05-09: only Phase 0-4 implemented; output is\n"
            "       in_gat_full (Phase 2 end state) for CPU-diff verification.\n"
            "Note: legacy --variant=o (broken FP16 RAW O) archived to archived_o-broke/.\n",
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

    /* Optional positional [cl_dev] and named [--variant=g|o32]. */
    int dev = 0;
    int variant = 0;   /* 0 = G, 32 = o32, 16 = o16 (NYI) */
    for(int i = 10; i < argc; i++) {
        if(strncmp(argv[i], "--variant=", 10) == 0) {
            const char *v = argv[i] + 10;
            if(strcmp(v, "o32") == 0)      variant = 32;
            else if(strcmp(v, "o16") == 0) {
                fprintf(stderr, "[GPU] ERROR: --variant=o16 not yet implemented "
                                "(o16 = FP16 derivative of o32, port pending)\n");
                return 1;
            }
            else if(strcmp(v, "g") == 0 || strcmp(v, "G") == 0) variant = 0;
            else if(strcmp(v, "o") == 0 || strcmp(v, "O") == 0) {
                fprintf(stderr, "[GPU] ERROR: --variant=o was removed (broken RAW O "
                                "archived to archived_o-broke/).  Use --variant=o32 "
                                "for the in-progress FP32 port.\n");
                return 1;
            }
            else {
                fprintf(stderr, "[GPU] ERROR: unknown --variant=%s "
                                "(valid: g, o32)\n", v);
                return 1;
            }
        } else {
            dev = atoi(argv[i]);
        }
    }
    fprintf(stderr, "[GPU] variant = %s\n",
            variant == 32 ? "o32 (CPU O FP32 port; Phase 0-10 full pipeline as of 2026-05-09)"
                          : "G (GALOSH_RAW_G half-res LOSH + K14/K15/K16 chromaup)");
    return run_galosh_raw_gpu(input_file, output_file, width, height,
                              strength, luma_str, chroma_str,
                              alpha, sigma_sq, dev, variant);
}
