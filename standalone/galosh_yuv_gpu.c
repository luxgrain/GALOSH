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
 * Pipeline-variant selector flag (= mirror of CPU --variant).
 * Forward declared here so run_yuv_gat_gpu() (= defined below) can
 * reference it; main() (= defined later) parses the CLI flag and
 * sets it before calling run_yuv_gat_gpu.
 * ================================================================ */
static int g_galosh_yuv_q_gpu = 0;
/* LOSH luma precision: 0 = FP32 (PC/desktop default — matches the CPU FP32 reference to
 * ~8e-5 via galosh_fused_pass12_f32); 1 = FP16 (galosh_fused_pass12 — the mobile GPU/NPU
 * path, where scalar half is 2× and power-cheaper; on desktop it only adds an FP16 floor). */
static int g_yuv_fp16 = 0;


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
    cl_kernel k_makitalo = NULL, k_fused = NULL, k_fused_f32 = NULL;
    cl_kernel k_pass12_o32 = NULL;  /* PROVEN FP32 LOSH (RAW o32 kernel, 8e-5 vs CPU) */
    /* Step 7 (Phase 2a) — Y-guided filter, 4-kernel separable pipeline:
     *   1) moments_x        : 1D x-pass over (Y, Y², Cb, Cr, Y·Cb, Y·Cr)
     *   2) moments_y_and_ab : 1D y-pass + local linear regression → (a, b)
     *   3) apply_x          : 1D x-pass over (a, b) coefficient planes
     *   4) apply_y          : 1D y-pass + output = mean_a·Y + mean_b
     * Reduces per-pixel work from O((2r+1)²)=225 to O(2·(2r+1))=30 at r=7. */
    /* Step 7 chroma: replaced separable mean-only guided filter with a single
     * non-separable bilateral LOESS kernel that mirrors CPU
     * galosh_loess_chroma — adds exp(-(Y_i-Y_c)²/2σ²) bilateral weighting
     * which excludes specular highlights (e.g. silver windows) from the
     * regression.  CPU bench shows this is the chroma denoise path that
     * GALOSH_YUV_G adopts; the legacy 4-kernel separable pipeline (mean-
     * only, no bilateral) was a GPU shortcut that diverged from CPU. */
    cl_kernel k_guided_loess = NULL;
    /* Legacy 4-kernel separable pipeline kept declared so the existing buffer
     * allocations + cleanup compile unchanged; not registered any more. */
    cl_kernel k_guided_moments_x   = NULL;
    cl_kernel k_guided_moments_yab = NULL;
    cl_kernel k_guided_apply_x     = NULL;
    cl_kernel k_guided_apply_y     = NULL;

    /* Buffer handles */
    cl_mem srgb_buf = NULL, y_buf = NULL, cb_buf = NULL, cr_buf = NULL;
    cl_mem y_stab_buf = NULL, cb_stab_buf = NULL, cr_stab_buf = NULL;
    cl_mem scale_cb_buf = NULL, scale_cr_buf = NULL;
    cl_mem half_in_buf = NULL, half_out_buf = NULL;
    cl_mem y_den_f32_buf = NULL;   /* FP32 LOSH output (PC default path) */
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
    /* [LATEST: GALOSH_YUV_Q] chroma pyramid buffers (FP16 throughout
     * to honor the LDS ≤ 32KB / FP16 strict embedded constraint).
     * Allocated lazily when g_galosh_yuv_q_gpu is set; remain NULL
     * for the [LATEST: GALOSH_YUV_O] path. */
    cl_mem q_y_snap_f = NULL;       /* FP32 snapshot of pre-LOSH Y_stab */
    cl_mem ne_scratch = NULL;       /* |Lap| sample scratch for EXACT-median σ est (matches CPU) */
    cl_mem q_y_h16 = NULL;          /* full-res Y_stab guide, FP16 */
    cl_mem q_cb_h16 = NULL, q_cr_h16 = NULL;       /* full-res input Cb/Cr, FP16 */
    cl_mem q_cb_full_d = NULL, q_cr_full_d = NULL; /* Anchor 0 output, FP16 */
    cl_mem q_dummy_full = NULL, q_dummy_h = NULL, q_dummy_q = NULL;
    cl_mem q_dummy_kfh = NULL, q_dummy_kfq = NULL;
    /* half-res */
    cl_mem q_y_half = NULL, q_cb_half = NULL, q_cr_half = NULL;
    cl_mem q_cb_half_d = NULL, q_cr_half_d = NULL;
    cl_mem q_y_for_h = NULL;
    cl_mem q_cb_h_up_raw = NULL, q_cr_h_up_raw = NULL;
    cl_mem q_cb_h_up = NULL, q_cr_h_up = NULL;
    /* quarter-res */
    cl_mem q_y_qrt = NULL, q_cb_qrt = NULL, q_cr_qrt = NULL;
    cl_mem q_cb_qrt_d = NULL, q_cr_qrt_d = NULL;
    cl_mem q_y_for_q = NULL;
    cl_mem q_cb_q_to_h_raw = NULL, q_cr_q_to_h_raw = NULL;
    cl_mem q_cb_q_to_h = NULL, q_cr_q_to_h = NULL;
    cl_mem q_cb_q_up_raw = NULL, q_cr_q_up_raw = NULL;
    cl_mem q_cb_q_up = NULL, q_cr_q_up = NULL;
    /* smoothstep blend output (FP16) + FP32 conversion buffers */
    cl_mem q_cb_blend = NULL, q_cr_blend = NULL;
    cl_mem q_cb_blend_f = NULL, q_cr_blend_f = NULL;
    /* smoothstep blend kernel handle (= reused from §4) */
    cl_kernel k_o_smoothstep_3p = NULL;
    /* explicit f2h/h2f for chroma planes (= existing k_f2h, k_h2f). */

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
            "standalone/galosh.cl", NULL};

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
            "-DCL_TARGET_OPENCL_VERSION=120 -cl-nv-verbose "
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
    YG_KERNEL(k_fused,        "galosh_fused_pass12");         /* FP16 LOSH (mobile) */
    YG_KERNEL(k_fused_f32,    "galosh_fused_pass12_f32");     /* [DEPRECATED] FP32 LOSH (no validity filter) */
    YG_KERNEL(k_pass12_o32,   "galosh_pass12_o32");           /* FP32 LOSH (PC default) = proven RAW o32 kernel */
    YG_KERNEL(k_guided_loess,       "galosh_yuv_guided_loess");
    /* Separable guided-filter kernels: still present in galosh.cl for
     * archived bench reproduction but no longer used by the production
     * GALOSH_YUV_G pipeline. */

    /* [LATEST: GALOSH_YUV_Q] Q-variant kernels — Laplacian-MAD σ +
     * unified_sigma normalize + chroma pyramid via reused galosh_o_*
     * kernels (= mirror of RAW O Phase 7-9). */
    cl_kernel k_q_lap_mad     = NULL;
    cl_kernel k_q_synth_alpha = NULL;
    cl_kernel k_q_norm        = NULL;
    cl_kernel k_q_denorm      = NULL;
    cl_kernel k_o_box_3p      = NULL;
    cl_kernel k_o_loess_3p    = NULL;
    cl_kernel k_o_k16_3p      = NULL;
    cl_kernel k_o_pad         = NULL;
    cl_kernel k_o_crop        = NULL;
    /* MAD noise-est kernels created UNCONDITIONALLY: the default (GAT) mode now uses
     * the Laplacian-MAD path too (matches canonical CPU YUV_G; the block-mean/var
     * regression floors/underestimates on single-plane Y — see noise-est block). */
    YG_KERNEL(k_q_lap_mad,      "galosh_yuv_q_lap_mad");
    YG_KERNEL(k_q_synth_alpha,  "galosh_yuv_q_synth_alpha_sigma_sq");
    /* σ_gat normalize/denormalize created UNCONDITIONALLY: the default (O)
     * mode now also does the post-GAT unit-variance normalize (4a'/4g'),
     * mirroring the unconditional CPU YUV_O path (was Q-only → ~0.005 gap). */
    YG_KERNEL(k_q_norm,         "galosh_yuv_q_unified_sigma_norm");
    YG_KERNEL(k_q_denorm,       "galosh_yuv_q_unified_sigma_denorm");
    if(g_galosh_yuv_q_gpu)
    {
        YG_KERNEL(k_o_box_3p,       "galosh_o_box_downsample_2x_3p");
        YG_KERNEL(k_o_loess_3p,     "galosh_o_loess_chroma_3p_fp16_tiled");
        YG_KERNEL(k_o_k16_3p,       "galosh_o_k16_joint_bilateral_upsample_3p");
        YG_KERNEL(k_o_pad,          "galosh_o_pad_2d_edge");
        YG_KERNEL(k_o_crop,         "galosh_o_crop_2d_topleft");
        YG_KERNEL(k_o_smoothstep_3p,"galosh_o_smoothstep_blend_3p");
    }
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
        y_den_f32_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        q_y_snap_f    = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);  /* normalized post-GAT NOISY Y snapshot = chroma LOESS guide (BOTH modes; = CPU Y_stab guide) */
        ne_scratch    = clCreateBuffer(context, CL_MEM_READ_WRITE, (200000 + 64) * sizeof(float), NULL, &err);  /* exact-median σ est: holds n_samples=min(W*H/6,200000) |Lap| values */
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

    /* [LATEST: GALOSH_YUV_Q] FP16 chroma pyramid buffer alloc.
     * gat_box_downsample_2x convention: dw = sw/2 (floor); pyramid
     * uses /2 dimensions for half/quarter scales.  K16 stride
     * correction via galosh_o_pad_2d_edge / galosh_o_crop_2d_topleft. */
    if(g_galosh_yuv_q_gpu)
    {
        const int hw = width  / 2;
        const int hh = height / 2;
        const int qw = hw / 2;
        const int qh = hh / 2;
        const int kfw_half = 2 * hw;
        const int kfh_half = 2 * hh;
        const int kfw_q    = 2 * qw;
        const int kfh_q    = 2 * qh;
        const size_t fb = npix * sizeof(float);
        const size_t hb = npix * sizeof(cl_half);
        const size_t hh_h_sz = (size_t)hw * hh * sizeof(cl_half);
        const size_t hh_q_sz = (size_t)qw * qh * sizeof(cl_half);
        const size_t hh_kfh  = (size_t)kfw_half * kfh_half * sizeof(cl_half);
        const size_t hh_kfq  = (size_t)kfw_q    * kfh_q    * sizeof(cl_half);

        /* q_y_snap_f now allocated unconditionally (chroma LOESS guide for BOTH
         * O and Q modes — see the y_den_f32_buf block above). */
        q_y_h16    = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);
        q_cb_h16   = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);
        q_cr_h16   = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);
        q_cb_full_d = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);
        q_cr_full_d = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);
        q_dummy_full = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);
        q_dummy_h    = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_h_sz, NULL, &err);
        q_dummy_q    = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_q_sz, NULL, &err);
        q_dummy_kfh  = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_kfh, NULL, &err);
        q_dummy_kfq  = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_kfq, NULL, &err);

        q_y_half  = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_h_sz, NULL, &err);
        q_cb_half = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_h_sz, NULL, &err);
        q_cr_half = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_h_sz, NULL, &err);
        q_cb_half_d = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_h_sz, NULL, &err);
        q_cr_half_d = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_h_sz, NULL, &err);
        q_y_for_h     = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_kfh, NULL, &err);
        q_cb_h_up_raw = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_kfh, NULL, &err);
        q_cr_h_up_raw = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_kfh, NULL, &err);
        q_cb_h_up     = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);
        q_cr_h_up     = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);

        q_y_qrt     = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_q_sz, NULL, &err);
        q_cb_qrt    = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_q_sz, NULL, &err);
        q_cr_qrt    = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_q_sz, NULL, &err);
        q_cb_qrt_d  = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_q_sz, NULL, &err);
        q_cr_qrt_d  = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_q_sz, NULL, &err);
        q_y_for_q   = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_kfq, NULL, &err);
        q_cb_q_to_h_raw = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_kfq, NULL, &err);
        q_cr_q_to_h_raw = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_kfq, NULL, &err);
        q_cb_q_to_h     = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_h_sz, NULL, &err);
        q_cr_q_to_h     = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_h_sz, NULL, &err);
        q_cb_q_up_raw   = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_kfh, NULL, &err);
        q_cr_q_up_raw   = clCreateBuffer(context, CL_MEM_READ_WRITE, hh_kfh, NULL, &err);
        q_cb_q_up       = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);
        q_cr_q_up       = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);

        q_cb_blend   = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);
        q_cr_blend   = clCreateBuffer(context, CL_MEM_READ_WRITE, hb, NULL, &err);
        q_cb_blend_f = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);
        q_cr_blend_f = clCreateBuffer(context, CL_MEM_READ_WRITE, fb, NULL, &err);

        if(err) { fprintf(stderr, "[YUV_Q] chroma pyramid buf alloc err=%d\n", err); goto yg_cleanup; }

        /* Zero-fill the dummy buffers (= K16/box/LOESS 3p require 3rd channel). */
        cl_half hzero = 0;
        clEnqueueFillBuffer(queue, q_dummy_full, &hzero, sizeof(cl_half), 0, hb,      0, NULL, NULL);
        clEnqueueFillBuffer(queue, q_dummy_h,    &hzero, sizeof(cl_half), 0, hh_h_sz, 0, NULL, NULL);
        clEnqueueFillBuffer(queue, q_dummy_q,    &hzero, sizeof(cl_half), 0, hh_q_sz, 0, NULL, NULL);
        clEnqueueFillBuffer(queue, q_dummy_kfh,  &hzero, sizeof(cl_half), 0, hh_kfh,  0, NULL, NULL);
        clEnqueueFillBuffer(queue, q_dummy_kfq,  &hzero, sizeof(cl_half), 0, hh_kfq,  0, NULL, NULL);
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
     * Phase 2: Y blind noise estimation.
     *
     * [LATEST: GALOSH_YUV_O] (default) — Foi+Alenius 2008 block-based.
     * [LATEST: GALOSH_YUV_Q] (--variant=q) — Laplacian-MAD on Y plane,
     *   matching CPU galosh_yuv_cpu --variant=q.  Single-workgroup
     *   histogram median; output σ_lin → params[P_YG_SIGMA_Y], then
     *   synth_alpha derives α/σ² → params[P_ALPHA] / params[P_SIGMA_SQ].
     * ================================================================ */
    /* Noise est = Laplacian-MAD → synth α/σ², which MATCHES the canonical CPU
     * galosh_yuv_cpu (YUV_G): GPU-MAD α≈0.00177 vs CPU 0.00170 (verified 2026-06-28).
     * The legacy block-mean/var Foi-Alenius regression (else branch below, reusing
     * the RAW-Bayer galosh_noise_estimate kernel) FLOORS / systematically
     * underestimates on the SINGLE-PLANE Y (α=1e-4 floor on small frames, 0.0006 vs
     * CPU 0.0021 on full frames) → GPU under-denoised by ~4.6 dB.  Both default and
     * --variant=q now use the MAD path; the block path is kept [DEPRECATED]. */
    if(1)  /* was: if(g_galosh_yuv_q_gpu) */
    {
        /* Q Stage 1: Laplacian-MAD on linear Y → σ_lin. */
        const int x_stride = 3;
        const float lap_max = 0.5f;       /* clip range for linear-domain Y */
        const int sigma_idx = P_YG_SIGMA_Y;
        clSetKernelArg(k_q_lap_mad, 0, sizeof(cl_mem), &y_buf);
        clSetKernelArg(k_q_lap_mad, 1, sizeof(int), &width);
        clSetKernelArg(k_q_lap_mad, 2, sizeof(int), &height);
        clSetKernelArg(k_q_lap_mad, 3, sizeof(int), &x_stride);
        clSetKernelArg(k_q_lap_mad, 4, sizeof(float), &lap_max);
        clSetKernelArg(k_q_lap_mad, 5, sizeof(int), &sigma_idx);
        clSetKernelArg(k_q_lap_mad, 6, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_q_lap_mad, 7, sizeof(cl_mem), &ne_scratch);
        dispatch_1d_named(queue, k_q_lap_mad, 64, 64, "YGQ2a lap_mad_lin");

        /* Q Stage 1 derive α/σ² from σ_lin (= synthetic, matches CPU). */
        const int alpha_idx    = P_ALPHA;
        const int sigma_sq_idx = P_SIGMA_SQ;
        clSetKernelArg(k_q_synth_alpha, 0, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_q_synth_alpha, 1, sizeof(int), &sigma_idx);
        clSetKernelArg(k_q_synth_alpha, 2, sizeof(int), &alpha_idx);
        clSetKernelArg(k_q_synth_alpha, 3, sizeof(int), &sigma_sq_idx);
        dispatch_1d_named(queue, k_q_synth_alpha, 1, 0, "YGQ2b synth_alpha");
    }
    else
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
        fprintf(stderr, "[YUV_%s] Blind est: alpha=%.6f sigma_sq=%.8f\n",
                g_galosh_yuv_q_gpu ? "Q" : "GAT",
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

    /* 4a': post-GAT σ_gat unit-variance normalize — RUN IN BOTH MODES.
     * Mirrors CPU galosh_yuv_cpu Step 5 (UNCONDITIONAL for both YUV_O and
     * YUV_Q): Laplacian-MAD σ_gat on the post-GAT Y_stab → divide Y_stab by
     * σ_gat for unit variance in GAT space, so the LOSH sees the canonical
     * scale.  BUG FIX 2026-06-28: this was Q-ONLY; the default (O) path
     * skipped it, so the GPU LOSH ran at the wrong noise scale and shrank
     * differently from the CPU → a ~0.005 mean|diff| CPU<->GPU divergence
     * that FP32-LOSH alone could not close (noise-est α already matched to
     * ~0.1%).  The matching de-normalize is "4g'" below. */
    {
        const int x_stride = 3;
        const float lap_max = 8.0f;       /* clip range for GAT-domain σ ≈ 1 */
        const int sigma_gat_idx = P_YG_SIGMA_CB; /* repurposed slot for σ_gat */
        clSetKernelArg(k_q_lap_mad, 0, sizeof(cl_mem), &y_stab_buf);
        clSetKernelArg(k_q_lap_mad, 1, sizeof(int), &width);
        clSetKernelArg(k_q_lap_mad, 2, sizeof(int), &height);
        clSetKernelArg(k_q_lap_mad, 3, sizeof(int), &x_stride);
        clSetKernelArg(k_q_lap_mad, 4, sizeof(float), &lap_max);
        clSetKernelArg(k_q_lap_mad, 5, sizeof(int), &sigma_gat_idx);
        clSetKernelArg(k_q_lap_mad, 6, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_q_lap_mad, 7, sizeof(cl_mem), &ne_scratch);
        dispatch_1d_named(queue, k_q_lap_mad, 64, 64, "YG4a' lap_mad_gat");

        clSetKernelArg(k_q_norm, 0, sizeof(cl_mem), &y_stab_buf);
        clSetKernelArg(k_q_norm, 1, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_q_norm, 2, sizeof(int), &sigma_gat_idx);
        clSetKernelArg(k_q_norm, 3, sizeof(int), &npix_i);
        dispatch_1d_named(queue, k_q_norm, align_up(npix, 256), 256, "YG4a'' unified_sigma_norm");
    }
    /* Snapshot post-σ_gat, pre-LOSH NOISY Y_stab — the chroma LOESS bilateral
     * GUIDE for BOTH modes (mirrors CPU galosh_yuv_cpu, which passes the
     * normalized post-GAT NOISY Y_stab as y_guide; the LOSH below overwrites
     * y_stab_buf, so snapshot first).  BUG FIX 2026-06-28: O-mode chroma
     * previously used the DENOISED LINEAR y_buf as guide — wrong domain (linear
     * vs GAT) and wrong signal (denoised vs noisy), so the bilateral weights
     * went ~uniform and the chroma diverged from the CPU by ~0.0057 (the
     * dominant CPU<->GPU gap, not the luma LOSH/precision). */
    clEnqueueCopyBuffer(queue, y_stab_buf, q_y_snap_f, 0, 0,
                        npix * sizeof(float), 0, NULL, NULL);

    /* 4b: float → half (FP16 LOSH path only; FP32 reads y_stab directly) */
    if(g_yuv_fp16)
    {
        clSetKernelArg(k_f2h, 0, sizeof(cl_mem), &y_stab_buf);
        clSetKernelArg(k_f2h, 1, sizeof(cl_mem), &half_in_buf);
        clSetKernelArg(k_f2h, 2, sizeof(int), &npix_i);
        dispatch_1d_named(queue, k_f2h, align_up(npix, 256), 256, "YG4b f2h_Y");
    }

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

    /* 4e-4g: LOSH (Pass1 BayesShrink + Pass2 Wiener) on Y_stab.
     * FP32 (default, PC): galosh_fused_pass12_f32 on float y_stab → y_den_f32 scratch,
     *   then copy back to y_stab (no f2h/h2f) → matches the CPU FP32 reference (~8e-5).
     * FP16 (--fp16, mobile GPU/NPU): the half galosh_fused_pass12 bracketed by f2h/h2f. */
    if(g_yuv_fp16)
    {
        cl_half hzero = 0;
        clEnqueueFillBuffer(queue, half_out_buf, &hzero, sizeof(cl_half), 0,
                            npix * sizeof(cl_half), 0, NULL, NULL);
        clSetKernelArg(k_fused, 0, sizeof(cl_mem), &half_in_buf);
        clSetKernelArg(k_fused, 1, sizeof(cl_mem), &half_out_buf);
        clSetKernelArg(k_fused, 2, sizeof(int), &width);
        clSetKernelArg(k_fused, 3, sizeof(int), &height);
        clSetKernelArg(k_fused, 4, sizeof(float), &strength_y);
        dispatch_tile_named(queue, k_fused, width, height, "YG4f fused_Y(FP16)");
        clSetKernelArg(k_h2f, 0, sizeof(cl_mem), &half_out_buf);
        clSetKernelArg(k_h2f, 1, sizeof(cl_mem), &y_stab_buf);
        clSetKernelArg(k_h2f, 2, sizeof(int), &npix_i);
        dispatch_1d_named(queue, k_h2f, align_up(npix, 256), 256, "YG4g h2f_Y");
    }
    else
    {
        /* FP32 (PC default) LOSH = the PROVEN galosh_pass12_o32 — the RAW o32
         * kernel that is bit-exact (~8e-5) vs the CPU galosh_pass12_multiorient_
         * blocked.  The YUV CPU LOSH IS that same function (n_orient=1, robust=1,
         * stride=GALOSH_STRIDE=2, block=GALOSH_BS=8), so o32 mirrors it directly,
         * INCLUDING o32's image-space block validity filter (ref ∈ [0, dim-BS])
         * that the hand-ported galosh_fused_pass12_f32 lacked → that omission was
         * the ~half-of-diff border divergence. */
        float fzero = 0.0f;
        clEnqueueFillBuffer(queue, y_den_f32_buf, &fzero, sizeof(float), 0,
                            npix * sizeof(float), 0, NULL, NULL);
        const int o32_phase_stride = 1;   /* 1 = 16 phases = full quality */
        clSetKernelArg(k_pass12_o32, 0, sizeof(cl_mem), &y_stab_buf);
        clSetKernelArg(k_pass12_o32, 1, sizeof(cl_mem), &y_den_f32_buf);
        clSetKernelArg(k_pass12_o32, 2, sizeof(int), &width);
        clSetKernelArg(k_pass12_o32, 3, sizeof(int), &height);
        clSetKernelArg(k_pass12_o32, 4, sizeof(float), &strength_y);
        clSetKernelArg(k_pass12_o32, 5, sizeof(int), &o32_phase_stride);
        {   /* O32_TILE_SIZE=28 (galosh.cl), LDS 25.6KB; wg = g_tile_wg_dim². */
            const int o32_tile = 28;
            const size_t gl[2] = { (size_t)((width  + o32_tile - 1) / o32_tile) * g_tile_wg_dim,
                                   (size_t)((height + o32_tile - 1) / o32_tile) * g_tile_wg_dim };
            const size_t lo[2] = { (size_t)g_tile_wg_dim, (size_t)g_tile_wg_dim };
            clEnqueueNDRangeKernel(queue, k_pass12_o32, 2, NULL, gl, lo, 0, NULL, NULL);
        }
        clEnqueueCopyBuffer(queue, y_den_f32_buf, y_stab_buf, 0, 0,
                            npix * sizeof(float), 0, NULL, NULL);
    }

    /* 4g': [LATEST: GALOSH_YUV_Q] denormalize Y_stab by unified_sigma
     * before Makitalo (= mirrors CPU `gat_inverse_exact(Y_den * unified_sigma)`).
     * Stage 2 normalize divided Y_stab by σ_gat to give unit variance for LOSH;
     * the inverse GAT operates on the original GAT-domain scale, so we
     * multiply by σ_gat here to undo the normalization. */
    /* 4g': de-normalize Y_stab by σ_gat — RUN IN BOTH MODES (undo the 4a'
     * Step-5 normalize; mirrors CPU gat_inverse_exact(Y_den * unified_sigma)
     * which is also unconditional).  Was Q-only — see the 4a' bug-fix note. */
    {
        const int sigma_gat_idx = P_YG_SIGMA_CB;
        clSetKernelArg(k_q_denorm, 0, sizeof(cl_mem), &y_stab_buf);
        clSetKernelArg(k_q_denorm, 1, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_q_denorm, 2, sizeof(int), &sigma_gat_idx);
        clSetKernelArg(k_q_denorm, 3, sizeof(int), &npix_i);
        dispatch_1d_named(queue, k_q_denorm, align_up(npix, 256), 256, "YG4g' denorm");
    }

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
    cl_mem cb_final = NULL;
    cl_mem cr_final = NULL;
    /* Mark legacy separable kernels as intentionally unused. */
    (void)k_guided_moments_x; (void)k_guided_moments_yab;
    (void)k_guided_apply_x;   (void)k_guided_apply_y;

    if(g_galosh_yuv_q_gpu)
    {
        /* ============================================================
         * [LATEST: GALOSH_YUV_Q] Phase 5 — multi-scale chroma pyramid.
         *
         * 3 anchors with smoothstep slider walk (cs ∈ [0, 3]):
         *   anchor 0: noisy  (= cb_buf / cr_buf, raw input, no denoise)
         *   anchor 1: C_full (= LOESS at full-res with Y_stab guide)
         *   anchor 2: C_h_up (= half-res LOESS + EWA Jinc up to full)
         *   anchor 3: C_q_up (= quarter-res LOESS + 2 K16 ups to full)
         *
         * Lazy compute: skip anchors not adjacent to current cs segment.
         * cs=1.0 (= calibrated default) → only anchor 1 (= C_full).
         *
         * FP16 throughout (= LDS ≤ 32KB constraint, mirrors RAW O port).
         * Reuses §4 galosh_o_* kernels (= scale-agnostic, work for both
         * RAW O's half-base pyramid and YUV Q's full-base pyramid).
         * ============================================================ */
        const int hw = width  / 2;
        const int hh = height / 2;
        const int qw = hw / 2;
        const int qh = hh / 2;
        const int kfw_half = 2 * hw;
        const int kfh_half = 2 * hh;
        const int kfw_q    = 2 * qw;
        const int kfh_q    = 2 * qh;

        const float cs = (strength_c < 0.0f) ? 0.0f
                       : (strength_c > 3.0f) ? 3.0f : strength_c;
        const int need_full = (cs <= 2.0f);  /* segments 1, 2 */
        const int need_h_up = (cs >  1.0f);  /* segments 2, 3 */
        const int need_q_up = (cs >  2.0f);  /* segment 3 */

        /* Convert Y_stab snap + Cb + Cr from FP32 to FP16. */
        clSetKernelArg(k_f2h, 0, sizeof(cl_mem), &q_y_snap_f);
        clSetKernelArg(k_f2h, 1, sizeof(cl_mem), &q_y_h16);
        clSetKernelArg(k_f2h, 2, sizeof(int), &npix_i);
        dispatch_1d_named(queue, k_f2h, align_up(npix, 256), 256, "YGQ5a f2h_Y");
        clSetKernelArg(k_f2h, 0, sizeof(cl_mem), &cb_buf);
        clSetKernelArg(k_f2h, 1, sizeof(cl_mem), &q_cb_h16);
        clSetKernelArg(k_f2h, 2, sizeof(int), &npix_i);
        dispatch_1d_named(queue, k_f2h, align_up(npix, 256), 256, "YGQ5b f2h_Cb");
        clSetKernelArg(k_f2h, 0, sizeof(cl_mem), &cr_buf);
        clSetKernelArg(k_f2h, 1, sizeof(cl_mem), &q_cr_h16);
        clSetKernelArg(k_f2h, 2, sizeof(int), &npix_i);
        dispatch_1d_named(queue, k_f2h, align_up(npix, 256), 256, "YGQ5c f2h_Cr");

        /* Anchor 1: full-res LOESS (= equivalent to CPU C_full). */
        if(need_full)
        {
            clSetKernelArg(k_o_loess_3p, 0, sizeof(cl_mem), &q_y_h16);
            clSetKernelArg(k_o_loess_3p, 1, sizeof(cl_mem), &q_cb_h16);
            clSetKernelArg(k_o_loess_3p, 2, sizeof(cl_mem), &q_cr_h16);
            clSetKernelArg(k_o_loess_3p, 3, sizeof(cl_mem), &q_dummy_full);
            clSetKernelArg(k_o_loess_3p, 4, sizeof(cl_mem), &q_cb_full_d);
            clSetKernelArg(k_o_loess_3p, 5, sizeof(cl_mem), &q_cr_full_d);
            clSetKernelArg(k_o_loess_3p, 6, sizeof(cl_mem), &q_dummy_full);
            clSetKernelArg(k_o_loess_3p, 7, sizeof(int),    &width);
            clSetKernelArg(k_o_loess_3p, 8, sizeof(int),    &height);
            const float scs = 1.0f;
            clSetKernelArg(k_o_loess_3p, 9, sizeof(float),  &scs);
            dispatch_2d_named(queue, k_o_loess_3p,
                              align_up(width, 16), align_up(height, 16), 16, 16, "YGQ5d loess_full");
        }

        /* Anchor 2: half-res LOESS + EWA Jinc bilateral up to full. */
        if(need_h_up)
        {
            /* Box-down full → half (3-channel: Y, Cb, Cr). */
            clSetKernelArg(k_o_box_3p, 0, sizeof(cl_mem), &q_y_h16);
            clSetKernelArg(k_o_box_3p, 1, sizeof(cl_mem), &q_cb_h16);
            clSetKernelArg(k_o_box_3p, 2, sizeof(cl_mem), &q_cr_h16);
            clSetKernelArg(k_o_box_3p, 3, sizeof(cl_mem), &q_y_half);
            clSetKernelArg(k_o_box_3p, 4, sizeof(cl_mem), &q_cb_half);
            clSetKernelArg(k_o_box_3p, 5, sizeof(cl_mem), &q_cr_half);
            clSetKernelArg(k_o_box_3p, 6, sizeof(int),    &width);
            clSetKernelArg(k_o_box_3p, 7, sizeof(int),    &height);
            dispatch_2d_named(queue, k_o_box_3p,
                              align_up(hw, 16), align_up(hh, 16), 16, 16, "YGQ5e box_full2half");

            /* LOESS at half-res. */
            clSetKernelArg(k_o_loess_3p, 0, sizeof(cl_mem), &q_y_half);
            clSetKernelArg(k_o_loess_3p, 1, sizeof(cl_mem), &q_cb_half);
            clSetKernelArg(k_o_loess_3p, 2, sizeof(cl_mem), &q_cr_half);
            clSetKernelArg(k_o_loess_3p, 3, sizeof(cl_mem), &q_dummy_h);
            clSetKernelArg(k_o_loess_3p, 4, sizeof(cl_mem), &q_cb_half_d);
            clSetKernelArg(k_o_loess_3p, 5, sizeof(cl_mem), &q_cr_half_d);
            clSetKernelArg(k_o_loess_3p, 6, sizeof(cl_mem), &q_dummy_h);
            clSetKernelArg(k_o_loess_3p, 7, sizeof(int),    &hw);
            clSetKernelArg(k_o_loess_3p, 8, sizeof(int),    &hh);
            const float scs = 1.0f;
            clSetKernelArg(k_o_loess_3p, 9, sizeof(float),  &scs);
            dispatch_2d_named(queue, k_o_loess_3p,
                              align_up(hw, 16), align_up(hh, 16), 16, 16, "YGQ5f loess_half");

            /* Crop full-res Y_stab guide → kfw_half × kfh_half for K16 stride.
             * Kernel sig: (src, dst, sw, sh, dw, dh) -- args 0..5 strict. */
            clSetKernelArg(k_o_crop, 0, sizeof(cl_mem), &q_y_h16);     /* src */
            clSetKernelArg(k_o_crop, 1, sizeof(cl_mem), &q_y_for_h);   /* dst */
            clSetKernelArg(k_o_crop, 2, sizeof(int),    &width);        /* sw */
            clSetKernelArg(k_o_crop, 3, sizeof(int),    &height);       /* sh */
            clSetKernelArg(k_o_crop, 4, sizeof(int),    &kfw_half);     /* dw */
            clSetKernelArg(k_o_crop, 5, sizeof(int),    &kfh_half);     /* dh */
            dispatch_2d_named(queue, k_o_crop,
                              align_up(kfw_half, 16), align_up(kfh_half, 16), 16, 16, "YGQ5g crop_Yh");

            /* K16 EWA Jinc bilateral: half-res → full-res (kfw_half × kfh_half). */
            const float bw = 3.0f;
            clSetKernelArg(k_o_k16_3p, 0, sizeof(cl_mem), &q_cb_half_d);
            clSetKernelArg(k_o_k16_3p, 1, sizeof(cl_mem), &q_cr_half_d);
            clSetKernelArg(k_o_k16_3p, 2, sizeof(cl_mem), &q_dummy_h);
            clSetKernelArg(k_o_k16_3p, 3, sizeof(cl_mem), &q_y_for_h);
            clSetKernelArg(k_o_k16_3p, 4, sizeof(cl_mem), &q_cb_h_up_raw);
            clSetKernelArg(k_o_k16_3p, 5, sizeof(cl_mem), &q_cr_h_up_raw);
            clSetKernelArg(k_o_k16_3p, 6, sizeof(cl_mem), &q_dummy_kfh);
            clSetKernelArg(k_o_k16_3p, 7, sizeof(int),    &hw);
            clSetKernelArg(k_o_k16_3p, 8, sizeof(int),    &hh);
            clSetKernelArg(k_o_k16_3p, 9, sizeof(float),  &bw);
            dispatch_2d_named(queue, k_o_k16_3p,
                              align_up(kfw_half, 16), align_up(kfh_half, 16), 16, 16, "YGQ5h k16_h2full");

            /* Pad raw output to native full-res (width × height).
             * Kernel sig: (src, dst, sw, sh, dw, dh) -- args 0..5 strict. */
            clSetKernelArg(k_o_pad, 0, sizeof(cl_mem), &q_cb_h_up_raw);  /* src */
            clSetKernelArg(k_o_pad, 1, sizeof(cl_mem), &q_cb_h_up);      /* dst */
            clSetKernelArg(k_o_pad, 2, sizeof(int),    &kfw_half);        /* sw */
            clSetKernelArg(k_o_pad, 3, sizeof(int),    &kfh_half);        /* sh */
            clSetKernelArg(k_o_pad, 4, sizeof(int),    &width);           /* dw */
            clSetKernelArg(k_o_pad, 5, sizeof(int),    &height);          /* dh */
            dispatch_2d_named(queue, k_o_pad,
                              align_up(width, 16), align_up(height, 16), 16, 16, "YGQ5i pad_Cb_h");
            clSetKernelArg(k_o_pad, 0, sizeof(cl_mem), &q_cr_h_up_raw);  /* src */
            clSetKernelArg(k_o_pad, 1, sizeof(cl_mem), &q_cr_h_up);      /* dst */
            dispatch_2d_named(queue, k_o_pad,
                              align_up(width, 16), align_up(height, 16), 16, 16, "YGQ5j pad_Cr_h");
        }

        /* Anchor 3: quarter-res LOESS + K16 q→h + K16 h→full. */
        if(need_q_up)
        {
            /* Box-down half → quarter. */
            clSetKernelArg(k_o_box_3p, 0, sizeof(cl_mem), &q_y_half);
            clSetKernelArg(k_o_box_3p, 1, sizeof(cl_mem), &q_cb_half);
            clSetKernelArg(k_o_box_3p, 2, sizeof(cl_mem), &q_cr_half);
            clSetKernelArg(k_o_box_3p, 3, sizeof(cl_mem), &q_y_qrt);
            clSetKernelArg(k_o_box_3p, 4, sizeof(cl_mem), &q_cb_qrt);
            clSetKernelArg(k_o_box_3p, 5, sizeof(cl_mem), &q_cr_qrt);
            clSetKernelArg(k_o_box_3p, 6, sizeof(int),    &hw);
            clSetKernelArg(k_o_box_3p, 7, sizeof(int),    &hh);
            dispatch_2d_named(queue, k_o_box_3p,
                              align_up(qw, 16), align_up(qh, 16), 16, 16, "YGQ5k box_h2q");

            /* LOESS at quarter-res. */
            clSetKernelArg(k_o_loess_3p, 0, sizeof(cl_mem), &q_y_qrt);
            clSetKernelArg(k_o_loess_3p, 1, sizeof(cl_mem), &q_cb_qrt);
            clSetKernelArg(k_o_loess_3p, 2, sizeof(cl_mem), &q_cr_qrt);
            clSetKernelArg(k_o_loess_3p, 3, sizeof(cl_mem), &q_dummy_q);
            clSetKernelArg(k_o_loess_3p, 4, sizeof(cl_mem), &q_cb_qrt_d);
            clSetKernelArg(k_o_loess_3p, 5, sizeof(cl_mem), &q_cr_qrt_d);
            clSetKernelArg(k_o_loess_3p, 6, sizeof(cl_mem), &q_dummy_q);
            clSetKernelArg(k_o_loess_3p, 7, sizeof(int),    &qw);
            clSetKernelArg(k_o_loess_3p, 8, sizeof(int),    &qh);
            const float scs = 1.0f;
            clSetKernelArg(k_o_loess_3p, 9, sizeof(float),  &scs);
            dispatch_2d_named(queue, k_o_loess_3p,
                              align_up(qw, 16), align_up(qh, 16), 16, 16, "YGQ5l loess_qrt");

            /* Crop half-res Y_stab → kfw_q × kfh_q for K16 q→h.
             * Kernel sig: (src, dst, sw, sh, dw, dh) -- args 0..5 strict. */
            clSetKernelArg(k_o_crop, 0, sizeof(cl_mem), &q_y_half);    /* src */
            clSetKernelArg(k_o_crop, 1, sizeof(cl_mem), &q_y_for_q);   /* dst */
            clSetKernelArg(k_o_crop, 2, sizeof(int),    &hw);           /* sw */
            clSetKernelArg(k_o_crop, 3, sizeof(int),    &hh);           /* sh */
            clSetKernelArg(k_o_crop, 4, sizeof(int),    &kfw_q);        /* dw */
            clSetKernelArg(k_o_crop, 5, sizeof(int),    &kfh_q);        /* dh */
            dispatch_2d_named(queue, k_o_crop,
                              align_up(kfw_q, 16), align_up(kfh_q, 16), 16, 16, "YGQ5m crop_Yq");

            /* K16 q→h. */
            const float bw = 3.0f;
            clSetKernelArg(k_o_k16_3p, 0, sizeof(cl_mem), &q_cb_qrt_d);
            clSetKernelArg(k_o_k16_3p, 1, sizeof(cl_mem), &q_cr_qrt_d);
            clSetKernelArg(k_o_k16_3p, 2, sizeof(cl_mem), &q_dummy_q);
            clSetKernelArg(k_o_k16_3p, 3, sizeof(cl_mem), &q_y_for_q);
            clSetKernelArg(k_o_k16_3p, 4, sizeof(cl_mem), &q_cb_q_to_h_raw);
            clSetKernelArg(k_o_k16_3p, 5, sizeof(cl_mem), &q_cr_q_to_h_raw);
            clSetKernelArg(k_o_k16_3p, 6, sizeof(cl_mem), &q_dummy_kfq);
            clSetKernelArg(k_o_k16_3p, 7, sizeof(int),    &qw);
            clSetKernelArg(k_o_k16_3p, 8, sizeof(int),    &qh);
            clSetKernelArg(k_o_k16_3p, 9, sizeof(float),  &bw);
            dispatch_2d_named(queue, k_o_k16_3p,
                              align_up(kfw_q, 16), align_up(kfh_q, 16), 16, 16, "YGQ5n k16_q2h");

            /* Pad raw q→h output (kfw_q × kfh_q) → half-res (hw × hh).
             * Kernel sig: (src, dst, sw, sh, dw, dh) -- args 0..5 strict. */
            clSetKernelArg(k_o_pad, 0, sizeof(cl_mem), &q_cb_q_to_h_raw); /* src */
            clSetKernelArg(k_o_pad, 1, sizeof(cl_mem), &q_cb_q_to_h);     /* dst */
            clSetKernelArg(k_o_pad, 2, sizeof(int),    &kfw_q);            /* sw */
            clSetKernelArg(k_o_pad, 3, sizeof(int),    &kfh_q);            /* sh */
            clSetKernelArg(k_o_pad, 4, sizeof(int),    &hw);               /* dw */
            clSetKernelArg(k_o_pad, 5, sizeof(int),    &hh);               /* dh */
            dispatch_2d_named(queue, k_o_pad,
                              align_up(hw, 16), align_up(hh, 16), 16, 16, "YGQ5o pad_Cb_qh");
            clSetKernelArg(k_o_pad, 0, sizeof(cl_mem), &q_cr_q_to_h_raw); /* src */
            clSetKernelArg(k_o_pad, 1, sizeof(cl_mem), &q_cr_q_to_h);     /* dst */
            dispatch_2d_named(queue, k_o_pad,
                              align_up(hw, 16), align_up(hh, 16), 16, 16, "YGQ5p pad_Cr_qh");

            /* K16 h→full (= reuses q_y_for_h crop computed in Anchor 2). */
            clSetKernelArg(k_o_k16_3p, 0, sizeof(cl_mem), &q_cb_q_to_h);
            clSetKernelArg(k_o_k16_3p, 1, sizeof(cl_mem), &q_cr_q_to_h);
            clSetKernelArg(k_o_k16_3p, 2, sizeof(cl_mem), &q_dummy_h);
            clSetKernelArg(k_o_k16_3p, 3, sizeof(cl_mem), &q_y_for_h);
            clSetKernelArg(k_o_k16_3p, 4, sizeof(cl_mem), &q_cb_q_up_raw);
            clSetKernelArg(k_o_k16_3p, 5, sizeof(cl_mem), &q_cr_q_up_raw);
            clSetKernelArg(k_o_k16_3p, 6, sizeof(cl_mem), &q_dummy_kfh);
            clSetKernelArg(k_o_k16_3p, 7, sizeof(int),    &hw);
            clSetKernelArg(k_o_k16_3p, 8, sizeof(int),    &hh);
            clSetKernelArg(k_o_k16_3p, 9, sizeof(float),  &bw);
            dispatch_2d_named(queue, k_o_k16_3p,
                              align_up(kfw_half, 16), align_up(kfh_half, 16), 16, 16, "YGQ5q k16_q2full");

            /* Pad to full-res.
             * Kernel sig: (src, dst, sw, sh, dw, dh) -- args 0..5 strict. */
            clSetKernelArg(k_o_pad, 0, sizeof(cl_mem), &q_cb_q_up_raw);  /* src */
            clSetKernelArg(k_o_pad, 1, sizeof(cl_mem), &q_cb_q_up);      /* dst */
            clSetKernelArg(k_o_pad, 2, sizeof(int),    &kfw_half);        /* sw */
            clSetKernelArg(k_o_pad, 3, sizeof(int),    &kfh_half);        /* sh */
            clSetKernelArg(k_o_pad, 4, sizeof(int),    &width);           /* dw */
            clSetKernelArg(k_o_pad, 5, sizeof(int),    &height);          /* dh */
            dispatch_2d_named(queue, k_o_pad,
                              align_up(width, 16), align_up(height, 16), 16, 16, "YGQ5r pad_Cb_q");
            clSetKernelArg(k_o_pad, 0, sizeof(cl_mem), &q_cr_q_up_raw);  /* src */
            clSetKernelArg(k_o_pad, 1, sizeof(cl_mem), &q_cr_q_up);      /* dst */
            dispatch_2d_named(queue, k_o_pad,
                              align_up(width, 16), align_up(height, 16), 16, 16, "YGQ5s pad_Cr_q");
        }

        /* Smoothstep walk blend: 4 anchors a/b/c/d ∈ {noisy, C_full, C_h_up, C_q_up}.
         * Unused anchors fall back to existing buffers as no-op (= the kernel
         * always reads all 4 anchor sources, so we point unused ones to
         * computed anchors to avoid undefined behavior). */
        cl_mem a_cb = q_cb_h16;
        cl_mem a_cr = q_cr_h16;
        cl_mem b_cb = need_full ? q_cb_full_d : a_cb;
        cl_mem b_cr = need_full ? q_cr_full_d : a_cr;
        cl_mem c_cb = need_h_up ? q_cb_h_up   : b_cb;
        cl_mem c_cr = need_h_up ? q_cr_h_up   : b_cr;
        cl_mem d_cb = need_q_up ? q_cb_q_up   : c_cb;
        cl_mem d_cr = need_q_up ? q_cr_q_up   : c_cr;

        /* Smoothstep blend (3-channel: Cb / Cr / dummy). */
        clSetKernelArg(k_o_smoothstep_3p,  0, sizeof(cl_mem), &a_cb);
        clSetKernelArg(k_o_smoothstep_3p,  1, sizeof(cl_mem), &a_cr);
        clSetKernelArg(k_o_smoothstep_3p,  2, sizeof(cl_mem), &q_dummy_full);
        clSetKernelArg(k_o_smoothstep_3p,  3, sizeof(cl_mem), &b_cb);
        clSetKernelArg(k_o_smoothstep_3p,  4, sizeof(cl_mem), &b_cr);
        clSetKernelArg(k_o_smoothstep_3p,  5, sizeof(cl_mem), &q_dummy_full);
        clSetKernelArg(k_o_smoothstep_3p,  6, sizeof(cl_mem), &c_cb);
        clSetKernelArg(k_o_smoothstep_3p,  7, sizeof(cl_mem), &c_cr);
        clSetKernelArg(k_o_smoothstep_3p,  8, sizeof(cl_mem), &q_dummy_full);
        clSetKernelArg(k_o_smoothstep_3p,  9, sizeof(cl_mem), &d_cb);
        clSetKernelArg(k_o_smoothstep_3p, 10, sizeof(cl_mem), &d_cr);
        clSetKernelArg(k_o_smoothstep_3p, 11, sizeof(cl_mem), &q_dummy_full);
        clSetKernelArg(k_o_smoothstep_3p, 12, sizeof(cl_mem), &q_cb_blend);
        clSetKernelArg(k_o_smoothstep_3p, 13, sizeof(cl_mem), &q_cr_blend);
        clSetKernelArg(k_o_smoothstep_3p, 14, sizeof(cl_mem), &q_dummy_full);
        clSetKernelArg(k_o_smoothstep_3p, 15, sizeof(int),    &width);
        clSetKernelArg(k_o_smoothstep_3p, 16, sizeof(int),    &height);
        clSetKernelArg(k_o_smoothstep_3p, 17, sizeof(float),  &cs);
        dispatch_2d_named(queue, k_o_smoothstep_3p,
                          align_up(width, 16), align_up(height, 16), 16, 16, "YGQ5t smoothstep");

        /* Convert blend output FP16 → FP32 for ycbcr2srgb consumption. */
        clSetKernelArg(k_h2f, 0, sizeof(cl_mem), &q_cb_blend);
        clSetKernelArg(k_h2f, 1, sizeof(cl_mem), &q_cb_blend_f);
        clSetKernelArg(k_h2f, 2, sizeof(int),    &npix_i);
        dispatch_1d_named(queue, k_h2f, align_up(npix, 256), 256, "YGQ5u h2f_Cb");
        clSetKernelArg(k_h2f, 0, sizeof(cl_mem), &q_cr_blend);
        clSetKernelArg(k_h2f, 1, sizeof(cl_mem), &q_cr_blend_f);
        clSetKernelArg(k_h2f, 2, sizeof(int),    &npix_i);
        dispatch_1d_named(queue, k_h2f, align_up(npix, 256), 256, "YGQ5v h2f_Cr");

        cb_final = q_cb_blend_f;
        cr_final = q_cr_blend_f;
    }
    else
    {
        /* GALOSH_YUV_O chroma: single-pass non-separable bilateral LOESS.
         * GUIDE = q_y_snap_f (normalized post-GAT NOISY Y, = CPU Y_stab guide),
         * NOT the denoised linear y_buf — see the 4a' snapshot bug-fix note. */
        clSetKernelArg(k_guided_loess, 0, sizeof(cl_mem), &q_y_snap_f);
        clSetKernelArg(k_guided_loess, 1, sizeof(cl_mem), &cb_buf);
        clSetKernelArg(k_guided_loess, 2, sizeof(cl_mem), &cr_buf);
        clSetKernelArg(k_guided_loess, 3, sizeof(cl_mem), &params_buf);
        clSetKernelArg(k_guided_loess, 4, sizeof(float),  &strength_c);
        clSetKernelArg(k_guided_loess, 5, sizeof(cl_mem), &cb_biv_buf);
        clSetKernelArg(k_guided_loess, 6, sizeof(cl_mem), &cr_biv_buf);
        clSetKernelArg(k_guided_loess, 7, sizeof(int),    &width);
        clSetKernelArg(k_guided_loess, 8, sizeof(int),    &height);
        dispatch_2d_named(queue, k_guided_loess,
                          align_up(width, 16), align_up(height, 16),
                          16, 16, "YG5 loess_bilateral");
        cb_final = cb_biv_buf;
        cr_final = cr_biv_buf;
    }

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
    if(k_guided_loess)       clReleaseKernel(k_guided_loess);
    /* Separable kernels were not registered in this build; safe no-op. */
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

    /* [LATEST: GALOSH_YUV_Q] release Q-specific kernels + chroma pyramid buffers. */
    if(k_q_lap_mad)       clReleaseKernel(k_q_lap_mad);
    if(k_q_synth_alpha)   clReleaseKernel(k_q_synth_alpha);
    if(k_q_norm)          clReleaseKernel(k_q_norm);
    if(k_q_denorm)        clReleaseKernel(k_q_denorm);
    if(k_o_box_3p)        clReleaseKernel(k_o_box_3p);
    if(k_o_loess_3p)      clReleaseKernel(k_o_loess_3p);
    if(k_o_k16_3p)        clReleaseKernel(k_o_k16_3p);
    if(k_o_pad)           clReleaseKernel(k_o_pad);
    if(k_o_crop)          clReleaseKernel(k_o_crop);
    if(k_o_smoothstep_3p) clReleaseKernel(k_o_smoothstep_3p);
    if(q_y_snap_f)        clReleaseMemObject(q_y_snap_f);
    if(ne_scratch)        clReleaseMemObject(ne_scratch);
    if(q_y_h16)           clReleaseMemObject(q_y_h16);
    if(q_cb_h16)          clReleaseMemObject(q_cb_h16);
    if(q_cr_h16)          clReleaseMemObject(q_cr_h16);
    if(q_cb_full_d)       clReleaseMemObject(q_cb_full_d);
    if(q_cr_full_d)       clReleaseMemObject(q_cr_full_d);
    if(q_dummy_full)      clReleaseMemObject(q_dummy_full);
    if(q_dummy_h)         clReleaseMemObject(q_dummy_h);
    if(q_dummy_q)         clReleaseMemObject(q_dummy_q);
    if(q_dummy_kfh)       clReleaseMemObject(q_dummy_kfh);
    if(q_dummy_kfq)       clReleaseMemObject(q_dummy_kfq);
    if(q_y_half)          clReleaseMemObject(q_y_half);
    if(q_cb_half)         clReleaseMemObject(q_cb_half);
    if(q_cr_half)         clReleaseMemObject(q_cr_half);
    if(q_cb_half_d)       clReleaseMemObject(q_cb_half_d);
    if(q_cr_half_d)       clReleaseMemObject(q_cr_half_d);
    if(q_y_for_h)         clReleaseMemObject(q_y_for_h);
    if(q_cb_h_up_raw)     clReleaseMemObject(q_cb_h_up_raw);
    if(q_cr_h_up_raw)     clReleaseMemObject(q_cr_h_up_raw);
    if(q_cb_h_up)         clReleaseMemObject(q_cb_h_up);
    if(q_cr_h_up)         clReleaseMemObject(q_cr_h_up);
    if(q_y_qrt)           clReleaseMemObject(q_y_qrt);
    if(q_cb_qrt)          clReleaseMemObject(q_cb_qrt);
    if(q_cr_qrt)          clReleaseMemObject(q_cr_qrt);
    if(q_cb_qrt_d)        clReleaseMemObject(q_cb_qrt_d);
    if(q_cr_qrt_d)        clReleaseMemObject(q_cr_qrt_d);
    if(q_y_for_q)         clReleaseMemObject(q_y_for_q);
    if(q_cb_q_to_h_raw)   clReleaseMemObject(q_cb_q_to_h_raw);
    if(q_cr_q_to_h_raw)   clReleaseMemObject(q_cr_q_to_h_raw);
    if(q_cb_q_to_h)       clReleaseMemObject(q_cb_q_to_h);
    if(q_cr_q_to_h)       clReleaseMemObject(q_cr_q_to_h);
    if(q_cb_q_up_raw)     clReleaseMemObject(q_cb_q_up_raw);
    if(q_cr_q_up_raw)     clReleaseMemObject(q_cr_q_up_raw);
    if(q_cb_q_up)         clReleaseMemObject(q_cb_q_up);
    if(q_cr_q_up)         clReleaseMemObject(q_cr_q_up);
    if(q_cb_blend)        clReleaseMemObject(q_cb_blend);
    if(q_cr_blend)        clReleaseMemObject(q_cr_blend);
    if(q_cb_blend_f)      clReleaseMemObject(q_cb_blend_f);
    if(q_cr_blend_f)      clReleaseMemObject(q_cr_blend_f);

    if(prog)    clReleaseProgram(prog);
    if(queue)   clReleaseCommandQueue(queue);
    if(context) clReleaseContext(context);
    free_aligned(srgb);
    return ret;
}



/* ================================================================
 * main: CLI wrapper.  Parses --variant=o|q before positional args.
 *
 * Pipeline-variant flag g_galosh_yuv_q_gpu (= forward-declared near
 * top of file) selects:
 *   0 → [LATEST: GALOSH_YUV_O] = production (Foi-Alenius σ + single-
 *       scale chroma LOESS, current galosh_yuv_guided_loess kernel).
 *   1 → [LATEST: GALOSH_YUV_Q] = production candidate (Laplacian MAD σ
 *       + unified_sigma normalize + 3-anchor multi-scale chroma pyramid
 *       via galosh_o_* kernels with YUV-specific full/half/quarter scale
 *       layout; FP16 + LDS ≤ 32KB inherited from RAW O constraints).
 * ================================================================ */
int main(int argc, char **argv)
{
    /* Strip --variant flag before positional parsing. */
    int new_argc = 0;
    char *positional[32];
    for(int i = 0; i < argc; i++)
    {
        const char *a = argv[i];
        if(strncmp(a, "--variant=", 10) == 0)
        {
            const char ch = (a[10] == 'Q') ? 'q' : (a[10] == 'O') ? 'o' : a[10];
            g_galosh_yuv_q_gpu = (ch == 'q') ? 1 : 0;
        }
        else if(strcmp(a, "--fp16") == 0)
        {
            g_yuv_fp16 = 1;   /* mobile GPU/NPU FP16 LOSH; default is FP32 (matches CPU) */
        }
        else
        {
            if(new_argc < 32) positional[new_argc++] = (char *)a;
        }
    }
    argv = positional;
    argc = new_argc;

    if(argc < 7) {
        fprintf(stderr,
            "Usage: %s <in.bin> <out.bin> <W> <H> <s_y> <s_c> [cl_dev] [--variant=o|q]\n",
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
    fprintf(stderr, "[YUV_GPU_%c] variant=%s\n",
            g_galosh_yuv_q_gpu ? 'Q' : 'O',
            g_galosh_yuv_q_gpu ? "Q (Laplacian-MAD σ + multi-scale chroma)" :
                                 "O (Foi-Alenius σ + single-scale chroma)");
    return run_yuv_gat_gpu(input_file, output_file, width, height, sy, sc, dev);
}
