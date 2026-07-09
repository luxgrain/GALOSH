# GALOSH o32 OpenCL Host Blueprint (for Vulkan mirror)

Source of truth: `standalone\galosh_raw_gpu.c`
(`run_galosh_raw_gpu()`, `variant == 32` path — `main()` hardcodes
`variant = 32`; `--variant=` on the CLI is accepted and ignored) and
`standalone\galosh_gpu.h` (constants + dispatch
helpers). All line numbers below refer to `galosh_raw_gpu.c` at commit-time
of this extraction (file = 2275 lines).

Kernels live in the single-source `standalone/galosh.cl`, built with:

```
-cl-fp32-correctly-rounded-divide-sqrt
-DGALOSH_STRIDE=2 -DTILE_SIZE=48 -DHIST_BINS=4096 -DREDUCE_WG_SIZE=256
```

`-cl-fp32-correctly-rounded-divide-sqrt` is REQUIRED for o32 to be a
faithful FP32 mirror of CPU O (BayesShrink hard-threshold sensitivity —
see comment at lines 297-303). The Vulkan port must ensure correctly
rounded FP32 divide/sqrt in shaders (no `RelaxedPrecision`, no fast-math).

The queue is a single **in-order** `cl_command_queue` with profiling
enabled. Every blocking read (`CL_TRUE`) is therefore an implicit
full-pipeline sync point. In Vulkan: one queue, sequential submission with
memory barriers between dependent dispatches; blocking reads become
fence-waited readbacks.

---

## 0. Global constants (galosh_gpu.h)

| Constant | Value | Notes |
|---|---|---|
| `GALOSH_BS` | 8 | WHT-LOSH block size |
| `GALOSH_BP` | 64 | |
| `GALOSH_STRIDE` | 2 | passed as `-D` to CL build; MUST be 2 (see bug note lines 100-109) |
| `TILE_SIZE` | 48 | G-variant tile kernels only; NOT used by o32 dispatches, but is baked into the CL build opts |
| `HIST_BINS` | 4096 | G-variant sigma histogram; o32 doesn't use `hist_buf` |
| `HIST_MAX` | 16.0f | |
| `REDUCE_WG_SIZE` | 256 | |
| `N_REDUCE_WG` | 64 | sizes `partial_buf` / `partial_resid_buf` |
| `GAT_LUT_SIZE` | 4096 | inverse-GAT LUT length |
| `PARAMS_SIZE` | 32 | params_buf length in floats |
| `TILE_WG_DIM` (`g_tile_wg_dim`) | 8 | WG dim for the o32 pass12 tiled dispatch (8×8 = 64 WIs/WG) |
| `O32_TILE_SIZE` | 28 | host hardcodes `const int o32_tile_size = 28; /* matches galosh.cl O32_TILE_SIZE */` (line 1386) |

### params_buf float-index layout (PARAMS_SIZE = 32)

| idx | name | written by (o32 path) | read by (o32 path) |
|---|---|---|---|
| 0 | `P_SIGMA_CH0` | P1b sigma_per_cfa | P1c unified_sigma; VERBOSE diag |
| 1 | `P_SIGMA_CH1` | P1b | P1c |
| 2 | `P_SIGMA_CH2` | P1b | P1c |
| 3 | `P_SIGMA_CH3` | P1b | P1c |
| 4 | `P_UNIFIED_SIGMA` | P1c unified_sigma | P1d normalize; K_O32_10 (×σ denorm) |
| 5 | `P_INV_SG` | P1c (device) | P1d normalize |
| 6 | `P_DARK_REF0` | P2a_F finalize | P2b dark_sub; K_O32_10 (dark restore) |
| 7 | `P_DARK_REF1` | P2a_F | P2b; K_O32_10 |
| 8 | `P_DARK_REF2` | P2a_F | P2b; K_O32_10 |
| 9 | `P_DARK_REF3` | P2a_F | P2b; K_O32_10 |
| 10 | `P_S_SCALE` | HOST (s_init before IRLS; ext-model `sigma_sq/alpha`); P2a_RF resid_finalize | P2a_R reduce (Tukey scale) |
| 11 | `P_LUMA_STR` | HOST at init (`strength*luma_str`) | (o32 kernels take luma strength as a direct kernel arg instead) |
| 12 | `P_CHROMA_STR` | HOST at init (`strength*chroma_str`) | (o32 uses `slider` kernel arg instead) |
| 13 | `P_ALPHA` | P0b ne_finalize; HOST (ext override) | P0c build_inv_lut, P1a gat_forward, HOST readbacks |
| 14 | `P_SIGMA_SQ` | P0h dark_finalize; HOST (ext override) | P0c, P1a, HOST readbacks |
| 15 | (free slot) `p_dark_thresh_slot` | P0f thresh_finalize (temp dark_thresh) | P0g lap_hist |
| 16-23 | `P_YG_*` (YUV-mode chroma slots) | — not used by o32 Bayer path | — |
| 24-31 | unused | — | — |

The slot index 15 is passed **as an int kernel arg** (`p_dark_thresh_slot = 15`)
to P0f/P0g/P0h — it is not hardcoded inside those kernels.

### Host dispatch helpers (semantics to replicate)

- `align_up(n, a) = ((n + a - 1) / a) * a` — global sizes are padded up to a
  multiple of the local size; kernels bounds-check internally. In Vulkan:
  `groupCount = align_up(n, local)/local`.
- `dispatch_1d_named(q, k, global, local, name)`: 1D NDRange; **`local == 0`
  means local size = NULL (driver-chosen)** — used for all the single-WI
  scalar kernels (`global = 1`). In Vulkan use local_size 1 and dispatch (1,1,1).
- `dispatch_2d_named(q, k, gx, gy, lx, ly, name)`: 2D NDRange, local (lx,ly).
- `dispatch_tile_named` (TILE_SIZE=48 geometry) is NOT used on the o32 path;
  the o32 pass12 tiled dispatch is done inline with O32_TILE_SIZE=28 (see §2, K_O32_5).

---

## 1. BUFFERS

Let (verbatim from code):

```c
const int    hw     = width / 2,  hh = height / 2;       /* width,height even (validated in main) */
const size_t npix   = (size_t)width * height;
const size_t chsize = (size_t)hw * hh;

const size_t full_bytes   = npix   * sizeof(float);      /* = W*H*4        */
const size_t ch_bytes     = chsize * sizeof(float);      /* = (W/2)*(H/2)*4 */
const size_t ch_bytes_h   = chsize * sizeof(cl_half);    /* FP16, G-only    */
const size_t full_bytes_h = npix   * sizeof(cl_half);    /* FP16, G-only    */
const size_t lut_bytes    = GAT_LUT_SIZE * sizeof(float);        /* 4096*4 = 16384 B */
const size_t hist_bytes   = 4 * HIST_BINS * sizeof(cl_int);      /* 4*4096*4 = 64 KB (G-only) */
const size_t part_bytes   = N_REDUCE_WG * 5 * sizeof(double);    /* 64*5*8 = 2560 B  */
const size_t resid_bytes  = N_REDUCE_WG * 2 * sizeof(double);    /* 64*2*8 = 1024 B  */
const size_t ch_bytes_f   = (size_t)hw * (size_t)hh * sizeof(float);  /* == ch_bytes */
```

o32 pyramid geometry (buffer-alloc block, lines 652-663 — note the alloc-time
names carry an `_a` suffix but are numerically identical to the run-time
`cq_w/cq_h/ce_w/ce_h/kq_w/kq_h/ke_w/ke_h` at lines 1416-1423):

```c
const int cq_w = hw / 2;         const int cq_h = hh / 2;      /* quarter-res */
const int ce_w = cq_w / 2;       const int ce_h = cq_h / 2;    /* eighth-res  */
const int kq_w = 2 * cq_w;       const int kq_h = 2 * cq_h;    /* K16 q→h output stride (= hw,hh rounded down to even) */
const int ke_w = 2 * ce_w;       const int ke_h = 2 * ce_h;    /* K16 e→q output stride (= cq rounded down to even)   */
const size_t cqsize_f = (size_t)cq_w * cq_h * sizeof(float);
const size_t cesize_f = (size_t)ce_w * ce_h * sizeof(float);
const size_t kqsize_f = (size_t)kq_w * kq_h * sizeof(float);
const size_t kesize_f = (size_t)ke_w * ke_h * sizeof(float);
```

o32 noise-estimation block counts (lines 725-727 and 795-798):

```c
const int    o32_ne_n_bx    = hw / 8;                 /* NE_BLOCK_SZ = 8, contiguous, no stride */
const int    o32_ne_n_by    = hh / 8;
const int    o32_ne_per_ch  = o32_ne_n_bx * o32_ne_n_by;
const size_t o32_ne_total   = (size_t)4 * o32_ne_n_bx * o32_ne_n_by;   /* 4 CFA channels */
/* o32_blk_mean / o32_blk_var each = o32_ne_total * sizeof(float) */
```

### 1a. Buffers the o32 path ACTUALLY TOUCHES (must exist in Vulkan)

All are `CL_MEM_READ_WRITE` device buffers, no host pointer.

| Buffer | Size (bytes) | Element | Role in o32 |
|---|---|---|---|
| `raw_buf` | `full_bytes = W*H*4` | f32 | INPUT upload target; read by P0a/P0b/P0e/P0g, P1a, P2a reduce kernels; **OVERWRITTEN by K_O32_10 as the final Bayer output**; downloaded at end |
| `ch0_buf`..`ch3_buf` | `ch_bytes = (W/2)*(H/2)*4` each | f32 | per-CFA planes; written by P1a gat_forward, scaled in-place by P1d normalize, dark-subtracted by P2b. (Not read by Phase 3-10 — kept for CPU-mirror parity/debug; safe to keep but they are written by 3 kernels, so the Vulkan kernels need them bound.) |
| `params_buf` | `PARAMS_SIZE*4 = 32*4 = 128` | f32 | scalar parameter block (layout above) |
| `lut_d_buf` | `lut_bytes = 4096*4` | f32 | inverse-GAT LUT (D values) |
| `lut_x_buf` | `lut_bytes = 4096*4` | f32 | inverse-GAT LUT (x values) |
| `lut_params_buf` | `8*sizeof(float) = 32` | f32 | LUT meta: d_min, d_max, y_break, t_break, sigma_raw, … (5 used, 8 allocated) |
| `partial_buf` | `part_bytes = N_REDUCE_WG*5*sizeof(double) = 64*5*8 = 2560` | **f64** | dark-ref IRLS reduce partials (5 doubles per WG × 64 WGs) |
| `partial_resid_buf` | `resid_bytes = N_REDUCE_WG*2*sizeof(double) = 64*2*8 = 1024` | **f64** | resid reduce partials (2 doubles per WG × 64 WGs) |
| `o32_blk_mean` | `o32_ne_total * 4` | f32 | P0 block means |
| `o32_blk_var` | `o32_ne_total * 4` | f32 | P0 block Laplacian-MAD variances |
| `o32_dark_thresh_hist` | `4096 * sizeof(cl_int) = 16384` | i32 | P0e histogram; **zero-filled by host before P0e** |
| `o32_dark_lap_hist` | `4096 * sizeof(cl_int) = 16384` | i32 | P0g histogram; **zero-filled by host before P0g** |
| `o32_in_gat_full` | `full_bytes_f = W*H*4` | f32 | full-res GAT (post normalize + dark_sub) |
| `o32_L_cs` | `full_bytes_f` | f32 | Phase 3 output (forward L stride=1) |
| `o32_L_cs_den` | `full_bytes_f` | f32 | Phase 5 output (WHT-LOSH denoised) |
| `o32_L_pixel` | `full_bytes_f` | f32 | Phase 6 output (2×2 overlap-avg), guide for K_O32_9, input to K_O32_10 |
| `o32_L_h_den` | `ch_bytes_f` | f32 | Phase 6 half-res L |
| `o32_C1_h`,`o32_C2_h`,`o32_C3_h` | `ch_bytes_f` each | f32 | Phase 4 half-res chroma |
| `o32_L_q` | `cqsize_f` | f32 | L pyramid quarter |
| `o32_L_e` | `cesize_f` | f32 | L pyramid eighth |
| `o32_L_for_q` | `kqsize_f` | f32 | L_h_den cropped to kq stride (K16 guide) |
| `o32_L_for_e` | `kesize_f` | f32 | L_q cropped to ke stride (K16 guide) |
| `o32_C{1,2,3}_q` | `cqsize_f` each | f32 | C pyramid quarter |
| `o32_C{1,2,3}_e` | `cesize_f` each | f32 | C pyramid eighth |
| `o32_C{1,2,3}_loess_h` | `ch_bytes_f` each | f32 | LOESS half output |
| `o32_C{1,2,3}_loess_q` | `cqsize_f` each | f32 | LOESS quarter output |
| `o32_C{1,2,3}_loess_e` | `cesize_f` each | f32 | LOESS eighth output |
| `o32_C{1,2,3}_q_up_raw` | `kqsize_f` each | f32 | K16 q→h raw output (kq stride) |
| `o32_C{1,2,3}_q_up` | `ch_bytes_f` each | f32 | padded to chsize |
| `o32_C{1,2,3}_e_to_q_raw` | `kesize_f` each | f32 | K16 e→q raw output (ke stride) |
| `o32_C{1,2,3}_e_to_q` | `cqsize_f` each | f32 | padded to cqsize |
| `o32_C{1,2,3}_e_up_raw` | `kqsize_f` each | f32 | K16 q→h (of e_to_q) raw output (kq stride) |
| `o32_C{1,2,3}_e_up` | `ch_bytes_f` each | f32 | padded to chsize |
| `o32_C{1,2,3}_h_den` | `ch_bytes_f` each | f32 | Phase 8 blended chroma |
| `o32_C{1,2,3}_aligned` | `full_bytes_f` each | f32 | Phase 9 full-res chroma |

Debug-only (allocated on demand inside the `GALOSH_DUMP_DIR` block, line
1722): `o32_pilot_debug` = `npix*4` f32, for `galosh_pass1_o32_dump`.

### 1b. Shared buffers allocated by this host but NOT touched on the o32 path
(the Vulkan port may omit them entirely)

| Buffer | Size | Used only by |
|---|---|---|
| `hist_buf` | `4*HIST_BINS*4 = 64 KB` i32 | G K4 sigma_histogram |
| `blk_mean_buf`, `blk_var_buf` | `(4*(hw/16)*(hh/16))*4` each | G K0a (stride-2 layout, `ne_n_bx = hw/16`, `ne_n_by = hh/16`, `ne_total_blocks = 4*ne_n_bx*ne_n_by`) |
| `ne_dark_hist_buf` | `1024*4` i32 | G K0c |
| `ne_lap_hist_buf` | `2048*4` i32 | G K0d |
| `luma_buf`,`c1_buf`,`c2_buf`,`c3_buf`,`l_out_buf`,`c1_out_buf`,`c2_out_buf`,`c3_out_buf`,`pilot_tmp_buf` | `ch_bytes_h` (FP16) each | G K_SP/K13/K16 |
| `Lfr_noisy_buf`,`Lfr_pilot_buf`,`Lfr_den_buf` | `full_bytes_h` (FP16) each | G K14/K15/K16 |
| `rgf_*` (10 bufs), `luma_f_buf`,`c{1,2,3}_f_buf` | `ch_bytes_f` each | G K13 guided/LOESS chroma |

---

## 2. DISPATCH TABLE — one o32 frame, exact order

Scalar names: `width (W)`, `height (H)`, `hw = W/2`, `hh = H/2`,
`cq_w = hw/2`, `cq_h = hh/2`, `ce_w = cq_w/2`, `ce_h = cq_h/2`,
`kq_w = 2*cq_w`, `kq_h = 2*cq_h`, `ke_w = 2*ce_w`, `ke_h = 2*ce_h`.
All `int` args are 32-bit, all `float` args FP32. "local=0" ⇒ NULL local
(single work-item; use (1,1,1) in Vulkan). Arg indices are the exact
`clSetKernelArg` indices — **preserve order exactly**.

### Pre-pipeline (host)

1. **Upload input**: blocking `clEnqueueWriteBuffer(raw_buf, 0, npix*4, raw)` —
   input file is `W*H` float32 Bayer in [0,1].
2. **Init params**: host builds `float h_params[32]` = all zeros except
   `h_params[P_LUMA_STR] = strength * luma_str;`
   `h_params[P_CHROMA_STR] = strength * chroma_str;`
   then blocking write of all 32 floats to `params_buf`.

### Phase 0 — blind noise estimation

| # | Label | Kernel | Global | Local | Args (index: value) |
|---|---|---|---|---|---|
| 1 | `K_O32_P0a block_stats` | `galosh_o32_ne_block_stats` | 1D `align_up(o32_ne_total, 64)` | 64 | 0:`raw_buf`, 1:`width`, 2:`height`, 3:`o32_ne_n_bx (=hw/8)`, 4:`o32_ne_n_by (=hh/8)`, 5:`o32_ne_per_ch (=n_bx*n_by)`, 6:`o32_blk_mean`, 7:`o32_blk_var` |
| 2 | `K_O32_P0b ne_finalize` | `galosh_o32_ne_finalize` | 1D 256 | 256 | 0:`o32_blk_mean`, 1:`o32_blk_var`, 2:`raw_buf`, 3:`width`, 4:`height`, 5:`o32_ne_total` (int), 6:`params_buf` — writes `params[P_ALPHA]`. Single WG, 256 WIs ("8 thr/bin") |
| — | fill | `clEnqueueFillBuffer(o32_dark_thresh_hist, (cl_int)0, 0, 4096*4)` | | | zero histogram |
| 3 | `K_O32_P0e dark_thresh_hist` | `galosh_o32_ne_dark_thresh_hist` | 2D `align_up(hw_3,16) × align_up(hh_3,16)` where `hw_a=(W+1)/2, hh_a=(H+1)/2, hw_3=(hw_a+2)/3, hh_3=(hh_a+2)/3` | 16×16 | 0:`raw_buf`, 1:`width`, 2:`height`, 3:`o32_dark_thresh_hist` |
| 4 | `K_O32_P0f dark_thresh_fin` | `galosh_o32_ne_dark_thresh_finalize` | 1D 1 | 0 (NULL) | 0:`o32_dark_thresh_hist`, 1:`params_buf`, 2:`p_dark_thresh_slot = 15` (int) — writes dark_thresh into `params[15]` |
| — | fill | `clEnqueueFillBuffer(o32_dark_lap_hist, (cl_int)0, 0, 4096*4)` | | | zero histogram |
| 5 | `K_O32_P0g dark_lap_hist` | `galosh_o32_ne_dark_lap_hist` | 2D `align_up(hw_a,16) × align_up(hh_a,16)` | 16×16 | 0:`raw_buf`, 1:`width`, 2:`height`, 3:`params_buf`, 4:`p_dark_thresh_slot = 15` (int), 5:`o32_dark_lap_hist` |
| 6 | `K_O32_P0h dark_finalize` | `galosh_o32_ne_dark_finalize` | 1D 1 | 0 | 0:`o32_dark_lap_hist`, 1:`params_buf`, 2:`p_dark_thresh_slot = 15` (int) — writes `params[P_SIGMA_SQ]` |

**HOST SYNC #1** (lines 868-898): `clFinish(queue)`; blocking read of the
full `params_buf` (32 floats) → `est_o32[]`.
- If `ext_model` (`alpha_ext > 0 && sigma_sq_ext > 0` from CLI): set host
  `alpha = alpha_ext; sigma_sq = sigma_sq_ext;`
  `sup_s_scale = sigma_sq / fmaxf(alpha, 1e-12f);` then 3 blocking
  single-float writes into `params_buf` at byte offsets `P_ALPHA*4`,
  `P_SIGMA_SQ*4`, `P_S_SCALE*4` with `alpha`, `sigma_sq`, `sup_s_scale`.
  (Content-derived slots, e.g. `params[15]` dark_thresh, keep estimated values.)
- Else (blind): host `alpha = est_o32[P_ALPHA]; sigma_sq = est_o32[P_SIGMA_SQ];`
  (log only — device values already in place; nothing written back).

### Phase 1 — GAT + LUT + σ + normalize
(note the code order: gat_forward FIRST, then LUT build)

| # | Label | Kernel | Global | Local | Args |
|---|---|---|---|---|---|
| 7 | `K_O32_P1a gat_full` | `galosh_o32_gat_forward_full` | 2D `align_up(W,16) × align_up(H,16)` | 16×16 | 0:`raw_buf`, 1:`o32_in_gat_full`, 2:`ch0_buf`, 3:`ch1_buf`, 4:`ch2_buf`, 5:`ch3_buf`, 6:`params_buf`, 7:`width`, 8:`height` — reads α/σ² from params (device), writes full-res GAT and 4 CFA planes |
| 8 | `K_O32_P0c build_inv_lut` | `galosh_o32_build_inv_lut` | 1D 4096 | 256 | 0:`params_buf`, 1:`lut_d_buf`, 2:`lut_x_buf`, 3:`lut_params_buf` — Poisson + Gauss-Hermite exact inverse; reads α/σ² from params (device) |
| 9 | `K_O32_P0d lut_finalize` | `galosh_o32_lut_finalize` | 1D 1 | 0 | 0:`lut_d_buf`, 1:`lut_params_buf` |
| 10 | `K_O32_P1b sigma_cfa` | `galosh_o32_sigma_per_cfa` | 1D `4*256 = 1024` | 256 | 0:`o32_in_gat_full`, 1:`width`, 2:`height`, 3:`params_buf` — 4 WGs, one per CFA channel; writes `params[P_SIGMA_CH0..3]` |
| 11 | `K_O32_P1c unified` | `galosh_o32_unified_sigma` | 1D 1 | 0 | 0:`params_buf` — RMS-unifies σ_ch → `params[P_UNIFIED_SIGMA]` (+`P_INV_SG`) |
| 12 | `K_O32_P1d normalize` | `galosh_o32_normalize_apply` | 2D `align_up(W,16) × align_up(H,16)` | 16×16 | 0:`o32_in_gat_full`, 1:`ch0_buf`, 2:`ch1_buf`, 3:`ch2_buf`, 4:`ch3_buf`, 5:`params_buf`, 6:`width`, 7:`height` — in-place `*= 1/unified_sigma` on all 5 buffers |

### Phase 2 — dark-ref IRLS (3 iterations) + dark subtract

**HOST SYNC #2** (lines 1169-1181): blocking read of **2 floats** from
`params_buf` at byte offset `P_ALPHA * sizeof(float)` (= 52) →
`p_init[0] = params[13] (alpha)`, `p_init[1] = params[14] (sigma_sq)`.
(Blocking read on the in-order queue ⇒ waits for everything enqueued so far,
including P1d.) Host computes:

```c
const float a_for_s = fmaxf(p_init[0], 1e-12f);
const float s_init  = p_init[1] / a_for_s;
const float s_min   = 0.05f * s_init;
const float s_max   = 50.0f * s_init;
```

then blocking single-float write `s_init` → `params_buf` at byte offset
`P_S_SCALE * sizeof(float)` (= 40). `s_min`/`s_max` are kept host-side and
passed as scalar args to resid_finalize.

IRLS constants: `n_wg = 64 /* O32_DR_MWG_NWG */`, `wgsize = 256
/* O32_DR_MWG_WGSIZE */`; reduce dispatch = 1D global `64*256 = 16384`,
local 256. **No zero-fill of `partial_buf`/`partial_resid_buf`** on the o32
path (the mwg reduce kernels overwrite all their WG slots; contrast with
the G path which zero-fills each iteration).

Loop `for(dr_iter = 0; dr_iter <= 2; dr_iter++)` — exact dispatch sequence:
`R0, F0, RR0, RF0, R1, F1, RR1, RF1, R2, F2` (10 dispatches; `break` after
F2 skips RR2/RF2).

| Label (per iter i) | Kernel | Global | Local | Args |
|---|---|---|---|---|
| `K_O32_P2a_R{i} dr_reduce` | `galosh_o32_dark_ref_reduce_mwg` | 1D 16384 | 256 | 0:`o32_in_gat_full`, 1:`raw_buf`, 2:`width`, 3:`height`, 4:`params_buf`, 5:`partial_buf` — Tukey-bisquare weighted partial sums (5 doubles/WG) |
| `K_O32_P2a_F{i} dr_finalize` | `galosh_o32_dark_ref_finalize_mwg` | 1D 1 | 0 | 0:`partial_buf`, 1:`n_wg = 64` (int), 2:`params_buf` — aggregates → `params[P_DARK_REF0..3]` |
| *(skip below when i == 2)* | | | | |
| `K_O32_P2a_RR{i} resid` | `galosh_o32_dark_resid_reduce_mwg` | 1D 16384 | 256 | 0:`o32_in_gat_full`, 1:`raw_buf`, 2:`width`, 3:`height`, 4:`params_buf`, 5:`partial_resid_buf` |
| `K_O32_P2a_RF{i} resid_fin` | `galosh_o32_dark_resid_finalize_mwg` | 1D 1 | 0 | 0:`partial_resid_buf`, 1:`n_wg = 64` (int), 2:`s_min` (float, host), 3:`s_max` (float, host), 4:`params_buf` — updates `params[P_S_SCALE]`, clamped to [s_min, s_max] |

Then:

| # | Label | Kernel | Global | Local | Args |
|---|---|---|---|---|---|
| 23 | `K_O32_P2b dark_sub` | `galosh_o32_dark_sub_full` | 2D `align_up(W,16) × align_up(H,16)` | 16×16 | 0:`o32_in_gat_full`, 1:`ch0_buf`, 2:`ch1_buf`, 3:`ch2_buf`, 4:`ch3_buf`, 5:`params_buf`, 6:`width`, 7:`height` — in-place `-= dark_ref[cfa]` |

### Phase 3-6

| # | Label | Kernel | Global | Local | Args |
|---|---|---|---|---|---|
| 24 | `K_O32_2 fwd_L_cs` | `galosh_o32_forward_l_stride1` | 2D `align_up(W,16) × align_up(H,16)` | 16×16 | 0:`o32_in_gat_full`, 1:`o32_L_cs`, 2:`width`, 3:`height` |
| 25 | `K_O32_3 chroma_extract` | `galosh_o32_chroma_extract_halfres` | 2D `align_up(hw,16) × align_up(hh,16)` | 16×16 | 0:`o32_in_gat_full`, 1:`o32_C1_h`, 2:`o32_C2_h`, 3:`o32_C3_h`, 4:`width`, 5:`height`, 6:`hw`, 7:`hh` |
| 26 | `K_O32_5 pass12_L_fr` | `galosh_pass12_o32` | 2D tiled: `tiles_x = (W + 28 - 1)/28`, `tiles_y = (H + 28 - 1)/28`; global = `(tiles_x * 8, tiles_y * 8)` | 8×8 (`g_tile_wg_dim`) | 0:`o32_L_cs`, 1:`o32_L_cs_den`, 2:`width`, 3:`height`, 4:`luma_strength_o32 = strength * luma_str` (float), 5:`o32_phase_stride` (int; default 1 = all 16 cycle-spin phases; env `GALOSH_O32_PHASE_STRIDE`, clamped ≥ 1) |
| 27 | `K_O32_6 L_pixel+L_h_den_fused` | `galosh_o32_lpixel_lh_den_fused` | 2D `align_up(W,16) × align_up(H,16)` | 16×16 | 0:`o32_L_cs_den`, 1:`o32_L_pixel`, 2:`o32_L_h_den`, 3:`width`, 4:`height`, 5:`hw` |

**Small-image guard** (lines 1425-1433): if `cq_w < 4 || cq_h < 4 || ce_w < 4
|| ce_h < 4` → `clEnqueueCopyBuffer(o32_L_pixel → raw_buf, 0, 0, npix*4)`
and `goto download_phase` (Phases 7-10 skipped entirely).

### Phase 7 — multi-scale LOESS pyramid + K16 chain (7a-7o)

`loess_strength = 1.0f` (literal); `k16_BW = 1.5f` (literal). LOESS tiled
kernel uses local **24×24** (R=7 halo → 30² LDS tile ≈ 14.4 KB FP32 — check
Vulkan shared-memory budget); everything else 16×16.

| # | Label | Kernel | Global | Local | Args |
|---|---|---|---|---|---|
| 28 | `K_O32_7a L_q` | `galosh_o32_box_downsample_2x` | 2D `align_up(cq_w,16) × align_up(cq_h,16)` | 16×16 | 0:`o32_L_h_den`, 1:`o32_L_q`, 2:`hw` (src_w), 3:`hh` (src_h) |
| 29 | `K_O32_7b L_e` | `galosh_o32_box_downsample_2x` | 2D `align_up(ce_w,16) × align_up(ce_h,16)` | 16×16 | 0:`o32_L_q`, 1:`o32_L_e`, 2:`cq_w`, 3:`cq_h` |
| 30 | `K_O32_7c C_q` | `galosh_o32_box_downsample_2x_3p` | 2D `align_up(cq_w,16) × align_up(cq_h,16)` | 16×16 | 0:`o32_C1_h`, 1:`o32_C2_h`, 2:`o32_C3_h`, 3:`o32_C1_q`, 4:`o32_C2_q`, 5:`o32_C3_q`, 6:`hw`, 7:`hh` |
| 31 | `K_O32_7d C_e` | `galosh_o32_box_downsample_2x_3p` | 2D `align_up(ce_w,16) × align_up(ce_h,16)` | 16×16 | 0:`o32_C1_q`, 1:`o32_C2_q`, 2:`o32_C3_q`, 3:`o32_C1_e`, 4:`o32_C2_e`, 5:`o32_C3_e`, 6:`cq_w`, 7:`cq_h` |
| 32 | `K_O32_7e LOESS_h_t` | `galosh_o32_loess_chroma_3p_tiled` | 2D `align_up(hw,24) × align_up(hh,24)` | **24×24** | 0:`o32_L_h_den`, 1:`o32_C1_h`, 2:`o32_C2_h`, 3:`o32_C3_h`, 4:`o32_C1_loess_h`, 5:`o32_C2_loess_h`, 6:`o32_C3_loess_h`, 7:`hw`, 8:`hh`, 9:`loess_strength = 1.0f` (float) |
| 33 | `K_O32_7f LOESS_q_t` | `galosh_o32_loess_chroma_3p_tiled` | 2D `align_up(cq_w,24) × align_up(cq_h,24)` | 24×24 | 0:`o32_L_q`, 1:`o32_C1_q`, 2:`o32_C2_q`, 3:`o32_C3_q`, 4:`o32_C1_loess_q`, 5:`o32_C2_loess_q`, 6:`o32_C3_loess_q`, 7:`cq_w`, 8:`cq_h` — **arg 9 NOT re-set (OpenCL retains 1.0f from 7e; a Vulkan port must pass 1.0f explicitly)** |
| 34 | `K_O32_7g LOESS_e_t` | `galosh_o32_loess_chroma_3p_tiled` | 2D `align_up(ce_w,24) × align_up(ce_h,24)` | 24×24 | 0:`o32_L_e`, 1:`o32_C1_e`, 2:`o32_C2_e`, 3:`o32_C3_e`, 4:`o32_C1_loess_e`, 5:`o32_C2_loess_e`, 6:`o32_C3_loess_e`, 7:`ce_w`, 8:`ce_h` — arg 9 retained (1.0f) |
| 35 | `K_O32_7h crop_L_for_q` | `galosh_o32_crop_2d_topleft` | 2D `align_up(kq_w,16) × align_up(kq_h,16)` | 16×16 | 0:`o32_L_h_den` (src), 1:`o32_L_for_q` (dst), 2:`hw` (src_w), 3:`hh` (src_h), 4:`kq_w` (dst_w), 5:`kq_h` (dst_h) |
| 36 | `K_O32_7i K16_q2h` | `galosh_o32_k16_joint_bilateral_upsample_3p` | 2D `align_up(kq_w,16) × align_up(kq_h,16)` | 16×16 | 0:`o32_C1_loess_q`, 1:`o32_C2_loess_q`, 2:`o32_C3_loess_q`, 3:`o32_L_for_q` (guide), 4:`o32_C1_q_up_raw`, 5:`o32_C2_q_up_raw`, 6:`o32_C3_q_up_raw`, 7:`cq_w` (lowres_w), 8:`cq_h` (lowres_h), 9:`k16_BW = 1.5f` (float) |
| 37-39 | `K_O32_7j pad_q_up` ×3 (p = 0,1,2) | `galosh_o32_pad_2d_edge` | 2D `align_up(hw,16) × align_up(hh,16)` | 16×16 | 0:`o32_C{p+1}_q_up_raw` (src), 1:`o32_C{p+1}_q_up` (dst), 2:`kq_w` (src_w), 3:`kq_h` (src_h), 4:`hw` (dst_w), 5:`hh` (dst_h) — edge replicate |
| 40 | `K_O32_7k crop_L_for_e` | `galosh_o32_crop_2d_topleft` | 2D `align_up(ke_w,16) × align_up(ke_h,16)` | 16×16 | 0:`o32_L_q`, 1:`o32_L_for_e`, 2:`cq_w`, 3:`cq_h`, 4:`ke_w`, 5:`ke_h` |
| 41 | `K_O32_7l K16_e2q` | `galosh_o32_k16_joint_bilateral_upsample_3p` | 2D `align_up(ke_w,16) × align_up(ke_h,16)` | 16×16 | 0:`o32_C1_loess_e`, 1:`o32_C2_loess_e`, 2:`o32_C3_loess_e`, 3:`o32_L_for_e`, 4:`o32_C1_e_to_q_raw`, 5:`o32_C2_e_to_q_raw`, 6:`o32_C3_e_to_q_raw`, 7:`ce_w`, 8:`ce_h`, 9:`k16_BW = 1.5f` |
| 42-44 | `K_O32_7m pad_e_to_q` ×3 | `galosh_o32_pad_2d_edge` | 2D `align_up(cq_w,16) × align_up(cq_h,16)` | 16×16 | 0:`o32_C{p+1}_e_to_q_raw`, 1:`o32_C{p+1}_e_to_q`, 2:`ke_w`, 3:`ke_h`, 4:`cq_w`, 5:`cq_h` |
| 45 | `K_O32_7n K16_q2h_e` | `galosh_o32_k16_joint_bilateral_upsample_3p` | 2D `align_up(kq_w,16) × align_up(kq_h,16)` | 16×16 | 0:`o32_C1_e_to_q`, 1:`o32_C2_e_to_q`, 2:`o32_C3_e_to_q`, 3:`o32_L_for_q`, 4:`o32_C1_e_up_raw`, 5:`o32_C2_e_up_raw`, 6:`o32_C3_e_up_raw`, 7:`cq_w`, 8:`cq_h`, 9:`k16_BW = 1.5f` |
| 46-48 | `K_O32_7o pad_e_up` ×3 | `galosh_o32_pad_2d_edge` | 2D `align_up(hw,16) × align_up(hh,16)` | 16×16 | 0:`o32_C{p+1}_e_up_raw`, 1:`o32_C{p+1}_e_up`, 2:`kq_w`, 3:`kq_h`, 4:`hw`, 5:`hh` |

### Phase 8-10

| # | Label | Kernel | Global | Local | Args |
|---|---|---|---|---|---|
| 49 | `K_O32_8 smoothstep` | `galosh_o32_smoothstep_blend_3p` | 2D `align_up(hw,16) × align_up(hh,16)` | 16×16 | 0:`o32_C1_h`, 1:`o32_C2_h`, 2:`o32_C3_h`, 3:`o32_C1_loess_h`, 4:`o32_C2_loess_h`, 5:`o32_C3_loess_h`, 6:`o32_C1_q_up`, 7:`o32_C2_q_up`, 8:`o32_C3_q_up`, 9:`o32_C1_e_up`, 10:`o32_C2_e_up`, 11:`o32_C3_e_up`, 12:`o32_C1_h_den`, 13:`o32_C2_h_den`, 14:`o32_C3_h_den`, 15:`hw`, 16:`hh`, 17:`slider = strength * chroma_str` (float) |
| 50 | `K_O32_9 K16_final` | `galosh_o32_k16_joint_bilateral_upsample_3p` | 2D `align_up(W,16) × align_up(H,16)` | 16×16 | 0:`o32_C1_h_den`, 1:`o32_C2_h_den`, 2:`o32_C3_h_den`, 3:`o32_L_pixel` (full-res guide), 4:`o32_C1_aligned`, 5:`o32_C2_aligned`, 6:`o32_C3_aligned`, 7:`hw`, 8:`hh`, 9:`k16_BW = 1.5f` — output dim = 2*hw × 2*hh = W × H exactly |
| 51 | `K_O32_10 inverse_final` | `galosh_o32_inverse_wht_dark_gat` | 2D `align_up(W,16) × align_up(H,16)` | 16×16 | 0:`o32_L_pixel`, 1:`o32_C1_aligned`, 2:`o32_C2_aligned`, 3:`o32_C3_aligned`, 4:`raw_buf` (**output — overwrites the input buffer**), 5:`lut_d_buf`, 6:`lut_x_buf`, 7:`lut_params_buf`, 8:`params_buf`, 9:`width`, 10:`height` — inverse 2×2 WHT + dark restore + ×unified_sigma + inverse GAT via LUT |

### Download

`clFinish(queue)` → per-kernel profiling report → blocking
`clEnqueueReadBuffer(raw_buf, 0, npix*4, raw)` → write `raw` to the output
file (`W*H` float32). **The final Bayer output buffer is `raw_buf`.**

Total dispatches per o32 frame (normal-size image, no dump):
6 (P0) + 6 (P1 incl. LUT) + 10 (IRLS) + 1 (P2b) + 4 (P3-6) + 21 (P7a-7o,
pads ×3 each) + 3 (P8-10) = **51 kernel dispatches** + 2 fill-buffer ops.

---

## 3. HOST READBACKS & SYNC POINTS (mid-frame)

| Where | What | Direction | Purpose |
|---|---|---|---|
| After P0h (line 869) | `clFinish` + blocking read `params_buf[0..31]` (128 B) | D→H | get blind α (`[13]`) / σ² (`[14]`) for logging and the ext-model decision |
| ext_model only (lines 882-887) | 3 blocking 4-byte writes at offsets `P_ALPHA*4 = 52`, `P_SIGMA_SQ*4 = 56`, `P_S_SCALE*4 = 40` | H→D | override device model with CLI (α_ext, σ²_ext, σ²_ext/max(α_ext,1e-12)) |
| Before IRLS (lines 1172-1174) | blocking read 8 B at offset `P_ALPHA*4 = 52` (→ α, σ²) | D→H | compute `s_init = σ²/max(α,1e-12)`, `s_min = 0.05*s_init`, `s_max = 50*s_init` |
| Before IRLS (line 1179) | blocking write 4 B (`s_init`) at offset `P_S_SCALE*4 = 40` | H→D | seed IRLS scale |
| IRLS loop | `s_min`, `s_max` passed as **float kernel args** to each `resid_finalize_mwg` | H→D (push const) | Tukey scale clamp bounds |
| End of frame | `clFinish` + blocking read `raw_buf` (npix*4) | D→H | final output |
| `GALOSH_DUMP_DIR` set (debug) | `clFinish` + 14 intermediate buffer reads + 1 extra dispatch (`galosh_pass1_o32_dump`: tiled 28/8×8 geometry; args 0:`o32_L_cs`, 1:`o32_pilot_debug`(npix f32 temp), 2:`width`, 3:`height`, 4:`luma_strength = strength*luma_str`) | D→H | verification dumps (p2_in_gat, p3_L_cs, p4_C{1,2,3}_h, p5_pilot, p5_L_cs_den, p6_L_pixel, p6_L_h_den, p7_C1_loess_h, p7_C1_q_up, p7_C1_e_up, p8_C1_h_den, p9_C1_aligned, p10_output) |
| `GALOSH_VERBOSE` set (post-download) | reads `params_buf`, `lut_params_buf[0..4]`, sample LUT entries, FP16 diag values | D→H | diagnostics only |

Everything else stays on-device; kernel→kernel dependencies rely purely on
the in-order queue (Vulkan: pipeline barriers with
`SHADER_WRITE → SHADER_READ` between consecutive dependent dispatches).

---

## 4. SCALAR PARAMETER FLOW

CLI: `galosh_raw_gpu <in.bin> <out.bin> <W> <H> <strength> <luma_str>
<chroma_str> <alpha> <sigma_sq> [cl_dev]`. Dimensions must be positive and
even (checked in `main`).

- **alpha / sigma_sq (noise model)**:
  - Saved at entry: `alpha_ext = alpha; sigma_sq_ext = sigma_sq;
    ext_model = (alpha > 0.0f && sigma_sq > 0.0f);` (lines 55-56).
  - Phase 0 always runs the blind estimator (writes `params[13]`, `params[14]`
    on device).
  - If `ext_model`: host overwrites `params[P_ALPHA]`, `params[P_SIGMA_SQ]`,
    `params[P_S_SCALE] = sigma_sq_ext / fmaxf(alpha_ext, 1e-12f)` after the
    Phase-0 readback (device estimates for other slots, e.g. dark_thresh
    `params[15]`, are kept). Both positive is required — a single positive
    value alone does NOT trigger the override.
  - Consumers read α/σ² **from params_buf on device**: `P1a gat_forward_full`,
    `P0c build_inv_lut`, and (via P_S_SCALE seed + host s_min/s_max) the P2
    IRLS kernels. No o32 kernel receives α/σ² as a direct kernel argument.
- **strength / luma_str / chroma_str**:
  - `params[P_LUMA_STR] = strength*luma_str`, `params[P_CHROMA_STR] =
    strength*chroma_str` written once at init (used by G kernels; harmless
    for o32 but kept for parity).
  - `luma_strength_o32 = strength * luma_str` → **direct float arg 4** of
    `galosh_pass12_o32` (K_O32_5) (and of debug `galosh_pass1_o32_dump`).
  - `slider = strength * chroma_str` → **direct float arg 17** of
    `galosh_o32_smoothstep_blend_3p` (K_O32_8).
- **unified_sigma**: never touches the host. Computed on device
  (P1b per-CFA σ → P1c RMS unify → `params[P_UNIFIED_SIGMA]`/`P_INV_SG`),
  consumed on device by P1d normalize and K_O32_10 denormalize.
- **dark_ref[0..3]**: device-only (`params[6..9]`), written by IRLS finalize,
  consumed by P2b dark_sub and K_O32_10 dark restore.
- **s_scale (`params[10]`)**: host seeds `s_init` (from device α/σ² readback
  or ext model), device resid_finalize updates it per IRLS iteration within
  host-provided clamp `[0.05*s_init, 50*s_init]`.
- Literal constants passed as kernel args: `loess_strength = 1.0f` (LOESS,
  set once at 7e and retained for 7f/7g), `k16_BW = 1.5f` (all four K16
  dispatches), `o32_phase_stride` (default 1), `p_dark_thresh_slot = 15`,
  `n_wg = 64`.

---

## 5. NOTES / GOTCHAS for the Vulkan port

1. **`align_up(n, a) = ((n + a - 1)/a)*a`** — every 2D global is padded to
   the 16 (or 24 for tiled LOESS) local size; kernels self-guard on bounds.
2. **O32_TILE_SIZE = 28, TILE_WG_DIM = 8** — the pass12 (K_O32_5) dispatch
   is `global = (ceil(W/28)*8, ceil(H/28)*8), local = (8,8)`: each 8×8 WG
   owns one 28×28 output tile (halo handled in-kernel via LDS). This is
   NOT the G-variant `TILE_SIZE = 48` geometry.
3. **`partial_buf` / `partial_resid_buf` hold FP64** (5×64 / 2×64 doubles).
   The o32 mwg reduce/finalize kernels accumulate in double — the Vulkan
   device needs `shaderFloat64` (or a redesigned bit-exact-verified
   compensated-FP32 reduction; do not silently swap without re-verification).
4. **No zero-fill of the IRLS partial buffers** on the o32 path — reduce
   kernels fully overwrite their 64 WG slots each dispatch. The only host
   fills are the two `clEnqueueFillBuffer(…, 0, 4096*sizeof(int))` on
   `o32_dark_thresh_hist` and `o32_dark_lap_hist`.
5. **LOESS arg-retention quirk**: arg 9 (`loess_strength`) is set only on the
   7e dispatch; 7f/7g reuse it via OpenCL's sticky kernel-arg state. In
   Vulkan (push constants rebuilt per dispatch) pass `1.0f` explicitly each
   time.
6. **`raw_buf` is in/out**: input Bayer uploaded into it; K_O32_10 writes the
   final result into it; the small-image fallback copies `o32_L_pixel` into
   it. Download = `raw_buf`, `npix` floats.
7. **`ch0..ch3_buf` are written but never read** by o32 Phases 3-10 (Phase 3+
   consumes only `o32_in_gat_full`). They are still bound to P1a/P1d/P2b, so
   either keep them or (deviation) strip them from those three kernels —
   but any deviation breaks kernel-source parity with `galosh.cl`.
8. **Env vars consulted on the o32 path**:
   - `GALOSH_O32_PHASE_STRIDE` — int arg 5 of pass12 (default 1, clamp ≥ 1);
     production = 1 (all 16 cycle-spin phases).
   - `GALOSH_DUMP_DIR` — enables the 14-file intermediate dump + the extra
     `galosh_pass1_o32_dump` dispatch (debug only).
   - `GALOSH_VERBOSE` — post-download params/LUT diagnostics readback.
9. **Sync semantics**: single in-order queue; the two mid-frame blocking
   reads (Phase-0 params, pre-IRLS α/σ²) are the only points where the CPU
   waits mid-pipeline. Between them, all 51 dispatches can be recorded into
   one command buffer per segment: [P0a..P0h] → fence/readback →
   [P1a..P1d(+LUT)] … the pre-IRLS readback also implicitly waits for P1
   (harmless: it only needs Phase-0's α/σ², which P1 doesn't modify —
   a Vulkan port may read them from the Phase-0 readback it already has,
   or replicate the second readback verbatim for exactness; ext-model case
   must use the overridden values either way).
10. **Small-image guard** (`cq_w<4 || cq_h<4 || ce_w<4 || ce_h<4`): skip
    7a-K_O32_10, buffer-copy `o32_L_pixel → raw_buf` (npix floats), download.
11. **Precision**: build flag `-cl-fp32-correctly-rounded-divide-sqrt` is
    load-bearing (BayesShrink threshold crossings). Match with IEEE-strict
    div/sqrt in the Vulkan shaders.
12. **kq/ke crop-pad rationale**: K16 (`gat_k16_joint_bilateral_upsample`
    family) hardcodes output = 2× input dims (see memory
    `reference_k16_stride_bug`), so for non-2× targets the host crops the
    guide to `2*floor(dim/2)` (`crop_2d_topleft`) and edge-pads the K16
    output back up (`pad_2d_edge`). Chain: h(hw,hh) → q(cq) → e(ce);
    kq = 2*cq (≤ hw,hh), ke = 2*ce (≤ cq,ch).
