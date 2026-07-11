# GALOSH-YUV OpenCL Host Blueprint (for Vulkan mirror)

Source of truth: `standalone\galosh_yuv_gpu.c`
(`run_yuv_gat_gpu_buf()`, the **DEFAULT path**: `g_galosh_yuv_q_gpu == 0`
(variant O) and `g_yuv_fp16 == 0` (FP32 LOSH) — `main()` sets both from
CLI flags `--variant=o|q` / `--fp16`, defaults are 0/0) and
`standalone\galosh_gpu.h` (constants + dispatch helpers, shared with the
RAW host). All line numbers refer to `galosh_yuv_gpu.c` at commit-time of
this extraction (file = 1571 lines) and `galosh.cl` (file = 7886 lines).

Kernels live in the single-source `standalone/galosh.cl`, built with
(lines 213-217):

```
-DCL_TARGET_OPENCL_VERSION=120
-DGALOSH_STRIDE=2 -DTILE_SIZE=48 -DHIST_BINS=4096 -DREDUCE_WG_SIZE=256
```

> **⚠ DISCREPANCY vs the RAW host**: the YUV host does **NOT** pass
> `-cl-fp32-correctly-rounded-divide-sqrt` (the RAW o32 host does, and
> HOST_BLUEPRINT.md calls it load-bearing for `galosh_pass12_o32`
> BayesShrink threshold crossings — the SAME kernel the YUV default path
> dispatches for the Y plane). The YUV acceptance bar vs the CPU FP32
> reference is ~8e-5 mean|diff| (not bit-exact), which is why this was
> tolerated in OpenCL. **The Vulkan port must keep IEEE-strict div/sqrt
> and no fast-math on the sensitive kernels anyway** — the pass12 shaders
> are shared verbatim with the RAW engine, whose parity gates assume it.

## Constraints (state-of-play, read first)

1. **FP32 correctness**: no fast-math / no `RelaxedPrecision` on
   `pass12_o32`, `build_inv_lut`, `lut_finalize`, `makitalo_inverse_Y`.
   `o32 pass12` requires correctly-rounded FP32 divide/sqrt (BayesShrink
   hard-threshold sensitivity — see the RAW `HOST_BLUEPRINT.md` header).
2. **Kahan / FP_CONTRACT OFF regions**: `galosh_kacc` / `galosh_kcombine`
   (galosh.cl lines 28-51, `#pragma OPENCL FP_CONTRACT OFF`) are used by
   `galosh_build_inv_lut`. FMA contraction cancels the TwoSum error terms.
   GLSL equivalent: `precise` on the accumulation expressions — the shipped
   `o32_build_inv_lut.comp` already does this (reuse it, §Reuse map).
3. **Single in-order queue**: one `cl_command_queue` (profiling enabled);
   kernel→kernel dependencies rely purely on queue order. In Vulkan:
   one queue, `SHADER_WRITE → SHADER_READ` pipeline barriers between
   dependent dispatches; the two blocking reads become fence-waited
   readbacks (see §3).
4. **TDR-safe banded dispatch**: the heavy full-frame kernels
   (`pass12_o32`, `yuv_guided_loess`) must go through the Vulkan host's
   rate-learning banding (`galosh_vk.c`: `dispatch_k_banded()` /
   `band_piece()` / `g_band_rate_ms`; band = its OWN queue submission —
   measured on AMD iGPU the Windows TDR preemption unit is the
   submission, not `vkCmdDispatchBase` inside one command buffer).
   `yuv_q_lap_mad` is a single-workgroup dispatch and cannot be banded
   (it is ~ms-scale; acceptable).
5. **FP16 inter-phase storage contract v1** (`galosh_vk.c` line 11,
   `dataflow_spec.md §4`, `shaders/galosh_f16_rne.glsl`): the Vulkan YUV
   FP16 fast version adopts the SAME contract as RAW — inter-phase
   line-buffer stores rounded to IEEE binary16 RNE via the integer-exact
   `galosh_f16_rne()` (vendor-agnostic; AMD hardware f32→f16 is RTZ).
   This is NOT the OpenCL `--fp16` mobile path (`galosh_fused_pass12`
   half kernel bracketed by f2h/h2f) — that path is out of scope here.

---

## 0. Global constants

From `galosh_gpu.h` (shared header):

| Constant | Value | Notes |
|---|---|---|
| `PARAMS_SIZE` | 32 | params_buf length in floats |
| `GAT_LUT_SIZE` | 4096 | inverse-GAT LUT length |
| `TILE_WG_DIM` (`g_tile_wg_dim`) | 8 | WG dim of the pass12 tiled dispatch (8×8 = 64 WIs/WG). Same B7g note as RAW applies: Vulkan quality default = `o32_pass12_sg` (local 512, subgroup 32); 8×8 remains the OpenCL geometry and the `GALOSH_SG=0` bit-twin path. |
| `O32_TILE_SIZE` | 28 | host hardcodes `const int o32_tile = 28` (line 681) |
| `GALOSH_STRIDE` / `GALOSH_BS` / `GALOSH_BP` | 2 / 8 / 64 | pass12 block geometry (build `-D` / galosh.cl 98-115) |
| `TILE_SIZE=48`, `HIST_BINS=4096`, `REDUCE_WG_SIZE=256` | — | baked into the CL build opts; **NOT used by any default-path dispatch** (FP16/G-family only) |

From `galosh.cl`:

| Constant | Value | Used by |
|---|---|---|
| `SRGB_A/ONEP/GAMMA/PHI/THRS/THRS_L` | 0.055 / 1.055 / 2.4 / 12.92 / 0.04045 / 0.0031308 | srgb↔linear (YG1/YG9) |
| `BT709_KR/KG/KB` | 0.2126f / 0.7152f / 0.0722f | YG1/YG9 |
| `BT709_CB_DEN / CR_DEN` | 1.8556f / 1.5748f | = 2(1−KB) / 2(1−KR) |
| `YG_LOESS_RADIUS / YG_LOESS_BW` | 7 / 3.0f | guided_loess window (15×15) / bilateral bandwidth |
| `NE_LAPMAD_WG` | 64 | q_lap_mad single-WG size |
| `NE_LAPMAD_BINS / NE_LAPMAD_MAX_INV` | 4096 / 16.0f | **STALE** — the histogram median was replaced by exact quickselect; the defines and the "16 KB LDS" doc comment above the kernel (galosh.cl 4702-4725) no longer describe the implementation |
| `CHROMA_RATIO_CB / CR` | 0.733f / 0.816f | chroma_derive (dead in O path, see §5.6) |
| `O32_TILE_W / O32_TILE_PIXELS` | 40 / 1600 | pass12 LDS tile (28 + 2·HALO, HALO=6) |

### params_buf float-index layout (YUV usage; PARAMS_SIZE = 32)

galosh.cl YUV kernels use their own alias macros (`YG_P_ALPHA_Y = 13` etc.,
lines 3102-3108) that are numerically identical to the `P_*` host macros.

| idx | name (host / CL alias) | written by (default path) | read by (default path) |
|---|---|---|---|
| 13 | `P_ALPHA` / `YG_P_ALPHA_Y` | YGQ2b synth_alpha | YG4a gat_fwd (device); HOST readback → LUT kernel args |
| 14 | `P_SIGMA_SQ` / `YG_P_SIGMA_SQ_Y` | YGQ2b synth_alpha | YG4a gat_fwd (device); HOST readback → LUT kernel args |
| 16-19 | `P_YG_ALPHA_CB`, `P_YG_SIGMA_SQ_CB`, `P_YG_ALPHA_CR`, `P_YG_SIGMA_SQ_CR` | YG3 chroma_derive | **nothing** (dead on the O path — guided_loess no longer reads params; see §5.6) |
| 20 | `P_YG_SIGMA_Y` | YGQ2a lap_mad (linear-domain σ_lin) | YGQ2b synth_alpha |
| 21 | `P_YG_SIGMA_CB` (**repurposed** = σ_gat) | YG4a' lap_mad (GAT-domain) | YG4a'' norm, YG4g' denorm |
| 23 | `P_YG_EPS_BIV` | HOST at init (1e-3) | **nothing** (bivariate Wiener retired; slot kept for compatibility) |
| others | — | zero-initialised by host | — |

`sigma_idx` / `alpha_idx` / `sigma_sq_idx` / `sigma_gat_idx` are passed
**as int kernel args** — never hardcoded inside `yuv_q_lap_mad`,
`yuv_q_synth_alpha_sigma_sq`, `yuv_q_unified_sigma_norm/denorm`.

### Host dispatch helpers (same semantics as RAW blueprint §0)

- `align_up(n, a) = ((n + a - 1)/a)*a`; kernels bounds-check internally.
- `dispatch_1d_named`: `local == 0` ⇒ NULL local (driver-chosen) — used
  for the 1-WI scalar kernels (`global = 1`). Vulkan: local 1, dispatch (1,1,1).
- `dispatch_2d_named`: **FAIL FAST since 2026-07-11** — a failed dispatch
  `exit(1)`s (a silently-ignored `-54` masked the AMD LOESS bug for two
  months). Vulkan host: `CHECK()` semantics, already in place.
- `dispatch_tile_named` (TILE_SIZE=48 geometry) is used ONLY by the
  `--fp16` mobile branch (K `galosh_fused_pass12`), not by the default path.
- **The pass12 dispatch is a raw `clEnqueueNDRangeKernel`** (lines
  680-686), NOT a `dispatch_*_named` — it therefore has NO profiling
  event and is missing from the per-kernel profile printout (its time
  is only in the wall-clock total). Do not be misled when comparing
  per-kernel timings across backends.

---

## 1. BUFFERS

Let (verbatim from code): `npix = (size_t)W * H`, `fb = npix * sizeof(float)`.
No pyramid geometry on the default path (single scale). All buffers are
`CL_MEM_READ_WRITE`, no host pointer.

### 1a. Buffers the DEFAULT path ACTUALLY TOUCHES (must exist in Vulkan)

| Buffer | Size (bytes) | Element | Role in default path |
|---|---|---|---|
| `srgb_buf` | `3 * fb` | f32 | INPUT upload target (3-ch interleaved sRGB); **overwritten by YG9 as the final output**; downloaded at end |
| `y_buf` | `fb` | f32 | linear Y plane (YG1 out); noise-est input (YGQ2a); GAT input (YG4a); **overwritten by YG4h makitalo with denoised linear Y**; YG9 input |
| `cb_buf`, `cr_buf` | `fb` each | f32 | linear centred Cb/Cr (YG1 out); guided_loess input (noisy) |
| `y_stab_buf` | `fb` | f32 | GAT-domain Y: YG4a out → YG4a'' in-place ÷σ_gat → (snapshot copy) → pass12 in → overwritten by copy from `y_den_f32_buf` → YG4g' in-place ×σ_gat → YG4h in |
| `q_y_snap_f` | `fb` | f32 | snapshot of the normalized post-GAT **NOISY** `y_stab_buf` = chroma LOESS bilateral guide (allocated unconditionally despite the `q_` prefix; = CPU Y_stab guide; 2026-06-28 bug fix — see §5.3) |
| `y_den_f32_buf` | `fb` | f32 | pass12 FP32 LOSH output scratch (zero-filled before pass12, copied back into y_stab_buf) |
| `cb_biv_buf`, `cr_biv_buf` | `fb` each | f32 | guided_loess outputs (name is a bivariate-Wiener legacy); YG9 inputs |
| `ne_scratch` | `(200000 + 64) * 4` | f32 | \|Laplacian\| sample scratch for the exact-median σ estimator (holds `n_samples = min(W*H/6, 200000)` floats); used by BOTH lap_mad dispatches |
| `params_buf` | `32 * 4 = 128` | f32 | scalar parameter block (layout above) |
| `lut_d_buf`, `lut_x_buf` | `4096 * 4` each | f32 | inverse-GAT LUT (D values / x values) |
| `lut_params_buf` | `8 * 4 = 32` | f32 | LUT meta: d_min, d_max, y_break, t_break, sigma_raw, alpha, sigma_sq (7 used, 8 allocated) |

Peak npix-scale footprint = **11 × npix floats** (3 for srgb + 8 planes)
≈ 88 MB at 4K, 352 MB at 8K.

### 1b. Buffers allocated by this host but NOT touched on the default path
(the Vulkan port should omit them entirely)

| Buffer(s) | Size | Used only by |
|---|---|---|
| `cb_stab_buf`, `cr_stab_buf`, `scale_cb_buf`, `scale_cr_buf` | `fb` each | legacy separable guided filter (retired) |
| `half_in_buf`, `half_out_buf` | `npix * 2` (FP16) each | `--fp16` mobile LOSH branch |
| `gf_sum_*` (6), `gf_a/b_*` (4) | `fb` each — **10 × npix floats of dead weight** | legacy separable guided filter (kernels not even created any more) |
| `blk_mean_buf`, `blk_var_buf` | `(W/16)*(H/16)*4` each | [DEPRECATED] Foi-Alenius block-regression noise est (`else` branch, lines 481-546 — statically dead: `if(1)`) |
| `dark_hist_buf` (1024 i32), `lap_hist_buf` (2048 i32) | 4 KB / 8 KB | same deprecated branch |
| `q_*` chroma-pyramid buffers (~40) | various FP16 | Q-variant only (lazily allocated; NULL on default path) except `q_y_snap_f` which is unconditional (§1a) |

---

## 2. DISPATCH TABLE — one default-path frame, exact order

Scalar names: `W`, `H`, `npix = W*H` (passed as `int npix_i`),
`strength_y`, `strength_c` (CLI floats). "local=0" ⇒ NULL local
(single work-item; use (1,1,1) in Vulkan). Arg indices are the exact
`clSetKernelArg` indices — **preserve order exactly**.

### Pre-pipeline (host)

1. **Upload input**: non-blocking `clEnqueueWriteBuffer(srgb_buf, 0,
   3*npix*4, srgb)` — `srgb` is the caller-owned 3-ch interleaved float
   buffer (H×W×3, values in [0,1] gamma domain).
2. **Init params**: `float h_params[32]` = all zeros except
   `h_params[P_YG_EPS_BIV] = 1e-3f` (retired slot, kept for compat);
   non-blocking write of all 32 floats to `params_buf`.

### Phase 1 — colour decompose

| # | Label | Kernel (galosh.cl line) | Global | Local | Args (index: value) |
|---|---|---|---|---|---|
| 1 | `YG1 srgb2ycbcr` | `galosh_yuv_srgb_to_linear_ycbcr` (3153) | 1D `align_up(npix,256)` | 256 | 0:`srgb_buf`, 1:`y_buf`, 2:`cb_buf`, 3:`cr_buf`, 4:`npix` — fused sRGB EOTF⁻¹ → linear RGB → BT.709 YCbCr (centred Cb/Cr, NOT +0.5) |

### Phase 2 — blind noise estimation (Laplacian-MAD; **the ACTUAL default**)

> Since the 2026-06-28 fix (lines 449-456) the default (O) mode routes
> noise estimation through the **Laplacian-MAD kernels**, created
> unconditionally, NOT the old Foi-Alenius block-regression path. The
> guard is literally `if(1)  /* was: if(g_galosh_yuv_q_gpu) */`; the block
> path (`galosh_yuv_noise_block_stats_Y` → `galosh_noise_estimate` →
> `galosh_yuv_noise_dark_*`) is kept [DEPRECATED] in the dead `else`
> (it floors/underestimates α on single-plane Y → GPU was ~4.6 dB
> under-denoised). **Do not port the block path.**

| # | Label | Kernel | Global | Local | Args |
|---|---|---|---|---|---|
| 2 | `YGQ2a lap_mad_lin` | `galosh_yuv_q_lap_mad` (4752) | 1D 64 | 64 (**single WG**) | 0:`y_buf`, 1:`W`, 2:`H`, 3:`x_stride = 3` (int), 4:`lap_max = 0.5f` (**UNUSED** by the kernel — host-ABI relic), 5:`sigma_idx = P_YG_SIGMA_Y = 20` (int), 6:`params_buf`, 7:`ne_scratch` |
| 3 | `YGQ2b synth_alpha` | `galosh_yuv_q_synth_alpha_sigma_sq` (4850) | 1D 1 | 0 (NULL) | 0:`params_buf`, 1:`sigma_lin_idx = 20`, 2:`alpha_idx = P_ALPHA = 13`, 3:`sigma_sq_idx = P_SIGMA_SQ = 14` — α = max(σ_lin·0.1, 1e-5); σ² = max(σ_lin², 1e-8) |

**HOST SYNC #1** (lines 549-560): `clFinish(queue)`; blocking read of the
full `params_buf` (32 floats) → `alpha_y = est[13]`, `sigma_sq_y = est[14]`.
These two floats are REQUIRED on the host: they are passed as **direct
float kernel args** to YG4c/YG4d below (unlike RAW o32 where the LUT
kernels read them from `params_buf` on device — see §5.2 for the Vulkan
alternative). No ext-model override exists on the YUV CLI.

### Phase 3 — chroma params derive (dead compute, kept for parity)

| # | Label | Kernel | Global | Local | Args |
|---|---|---|---|---|---|
| 4 | `YG3 chroma_derive` | `galosh_yuv_chroma_params_derive` (3470) | 1D 1 | 0 | 0:`params_buf` — writes `params[16..19]` = `CHROMA_RATIO_{CB,CR}` × (α_Y, σ²_Y). **Nothing on the O path reads these** (§5.6) |

### Phase 4 — Y path: GAT → σ_gat normalize → FP32 LOSH → denormalize → inverse GAT

| # | Label | Kernel | Global | Local | Args |
|---|---|---|---|---|---|
| 5 | `YG4a gat_fwd` | `galosh_yuv_gat_forward_Y` (3494) | 1D `align_up(npix,256)` | 256 | 0:`y_buf`, 1:`y_stab_buf`, 2:`params_buf`, 3:`npix` — `Y_stab = (2/max(α,1e-12))·sqrt(max(α·Y + 0.375α² + σ², 0))`. NOTE: **no linear branch** (differs from RAW `o32_gat_forward_full`'s piecewise C¹ VST — do not substitute) |
| 6 | `YG4a' lap_mad_gat` | `galosh_yuv_q_lap_mad` (**2nd dispatch, same kernel**) | 1D 64 | 64 | 0:`y_stab_buf`, 1:`W`, 2:`H`, 3:`x_stride = 3`, 4:`lap_max = 8.0f` (unused), 5:`sigma_gat_idx = P_YG_SIGMA_CB = 21` (**repurposed slot**), 6:`params_buf`, 7:`ne_scratch` — σ_gat of post-GAT Y_stab. RUN IN BOTH MODES since 2026-06-28 (was Q-only → ~0.005 CPU↔GPU gap) |
| 7 | `YG4a'' norm` | `galosh_yuv_q_unified_sigma_norm` (4814) | 1D `align_up(npix,256)` | 256 | 0:`y_stab_buf`, 1:`params_buf`, 2:`sigma_idx = 21`, 3:`npix` — in-place `*= 1/max(σ_gat, 1e-6)` (unit variance for LOSH) |
| — | copy | `clEnqueueCopyBuffer(y_stab_buf → q_y_snap_f, 0, 0, npix*4)` | | | snapshot the normalized post-GAT **NOISY** Y_stab = chroma LOESS guide (BOTH modes; LOSH overwrites y_stab). Vulkan: `vkCmdCopyBuffer` + barrier |
| 8 | `YG4c build_lut` | `galosh_build_inv_lut` (1764) | 1D 4096 | 256 | 0:`lut_d_buf`, 1:`lut_x_buf`, 2:`alpha_y` (**float, host readback value**), 3:`sigma_sq_y` (float) — Makitalo exact-unbiased inverse table: Poisson series (log-prob recurrence) × 10-pt Gauss-Hermite; **Kahan-compensated FP32** (2026-07-11 de-FP64, back-port of the Vulkan shader) |
| 9 | `YG4d lut_fin` | `galosh_lut_finalize` (1830) | 1D 1 | 0 | 0:`lut_d_buf`, 1:`lut_params_buf`, 2:`alpha_y` (float), 3:`sigma_sq_y` (float) — writes lut_params[0..6] = d_min, d_max, y_break=−0.375α, t_break=2σ/α, σ_raw, α, σ² |
| — | fill | `clEnqueueFillBuffer(y_den_f32_buf, 0.0f, 0, npix*4)` | | | zero LOSH output scratch (RAW does not zero its pass12 output; keep for parity) |
| 10 | `YG4f pass12(FP32)` | `galosh_pass12_o32` (5129) — **the proven RAW o32 LOSH kernel** | 2D tiled: `(ceil(W/28)*8, ceil(H/28)*8)` | 8×8 (`g_tile_wg_dim`) | 0:`y_stab_buf`, 1:`y_den_f32_buf`, 2:`W`, 3:`H`, 4:`strength_y` (float — the raw CLI value, no `luma_str` product as in RAW), 5:`o32_phase_stride = 1` (int; **hardcoded 1** = all 16 cycle-spin phases; the `GALOSH_O32_PHASE_STRIDE` env of the RAW host is NOT consulted here). LDS = 4 × 1600 × 4 B = 25.6 KB (tile_in/numer/denom/pilot). Dispatched via raw `clEnqueueNDRangeKernel` — no profiling event (§0) |
| — | copy | `clEnqueueCopyBuffer(y_den_f32_buf → y_stab_buf, 0, 0, npix*4)` | | | LOSH result back into y_stab (Vulkan may instead ping-pong bindings — proven-deviation territory, re-verify) |
| 11 | `YG4g' denorm` | `galosh_yuv_q_unified_sigma_denorm` (4832) | 1D `align_up(npix,256)` | 256 | 0:`y_stab_buf`, 1:`params_buf`, 2:`sigma_idx = 21`, 3:`npix` — in-place `*= σ_gat` (undo YG4a''; mirrors CPU `gat_inverse_exact(Y_den * unified_sigma)`) |
| 12 | `YG4h makitalo` | `galosh_yuv_makitalo_inverse_Y` (3517) | 1D `align_up(npix,256)` | 256 | 0:`y_stab_buf`, 1:`y_buf` (**overwrites linear Y with denoised Y**), 2:`lut_d_buf`, 3:`lut_x_buf`, 4:`lut_params_buf`, 5:`npix` — per-pixel: `d<=d_min` → analytical `y_break + σ_raw·(d−t_break)` (NOT a clamp — dark-tail fix); `d>=d_max` → 1.0; interior → 12-iter binary search over lut_d + lerp with `/ fmax(d1−d0, 1e-10)` |

### Phase 5 — chroma: single-pass bilateral LOESS (Y-guided)

| # | Label | Kernel | Global | Local | Args |
|---|---|---|---|---|---|
| 13 | `YG5 loess_bilateral` | `galosh_yuv_guided_loess` (3912) | 2D `align_up(W,16) × align_up(H,16)` | 16×16 | 0:`q_y_snap_f` (**guide = normalized post-GAT NOISY Y_stab**, NOT the denoised linear y_buf — 2026-06-28 fix, §5.3), 1:`cb_buf`, 2:`cr_buf`, 3:`params_buf` (**bound but UNREAD** by the kernel body — legacy arg, §5.6), 4:`strength_c` (float), 5:`cb_biv_buf`, 6:`cr_biv_buf`, 7:`W`, 8:`H` — per pixel: 15×15 window (R=7), mirror-reflect edges (`yi = -yi; yi = 2H−yi−2`), bilateral weight `native_exp(−ΔY²/(2·3²))`, weighted linear regression C≈a·Y+b with `ε = strength_c²·1.0`, output clamped to the full-window input-chroma band (anti-overshoot, = RAW chroma clamp reflected to YUV). No LDS |

### Phase 6 — recompose

| # | Label | Kernel | Global | Local | Args |
|---|---|---|---|---|---|
| 14 | `YG9 ycbcr2srgb` | `galosh_yuv_ycbcr_to_srgb` (3189) | 1D `align_up(npix,256)` | 256 | 0:`y_buf` (denoised), 1:`cb_biv_buf`, 2:`cr_biv_buf`, 3:`srgb_buf` (**output — overwrites the input buffer**), 4:`npix` — BT.709 inverse, linear clip to [0,1], sRGB OETF |

### Download

`clFinish(queue)` → blocking `clEnqueueReadBuffer(srgb_buf, 0, 3*npix*4,
srgb)` → per-kernel profiling report. **The final output buffer is
`srgb_buf`** (result lands back in the caller-owned host buffer; file I/O
lives in the thin `run_yuv_gat_gpu` wrapper, lines 1208-1240).

Total per default frame: **14 kernel dispatches** (13 unique kernels;
`yuv_q_lap_mad` runs twice) + 1 fill-buffer + 2 copy-buffer ops.

### Kernel weight classification

| Class | Kernels |
|---|---|
| Trivial — 1-WI scalar | `yuv_q_synth_alpha_sigma_sq`, `yuv_chroma_params_derive`, `lut_finalize` |
| Trivial — per-pixel 1D map | `yuv_srgb_to_linear_ycbcr`, `yuv_gat_forward_Y`, `yuv_q_unified_sigma_norm`, `yuv_q_unified_sigma_denorm`, `yuv_ycbcr_to_srgb` |
| Medium | `yuv_q_lap_mad` (single WG; **WI-0 serial quickselect over ≤200k global floats** — a deliberate exactness/serialization trade, §5.5), `build_inv_lut` (4096 WIs × Poisson·GH series, Kahan), `yuv_makitalo_inverse_Y` (per-pixel + 12-iter LUT binary search) |
| Heavy — band these (constraint 4) | `pass12_o32` (25.6 KB LDS, 16 cycle-spin phases, WHT+BayesShrink+Wiener; identical to RAW K_O32_5), `yuv_guided_loess` (15×15 non-separable regression, 7 accumulators + min/max, full frame) |

### Q-variant branches (NOT extracted in detail — port later if ever)

`--variant=q` (`g_galosh_yuv_q_gpu=1`) keeps Phases 1-4/6 identical and
replaces Phase 5 with a 3-anchor multi-scale chroma pyramid (lines
741-1026): f2h converts (Y_snap, Cb, Cr) to FP16, then lazily computes
anchors {noisy, full-res LOESS, half-res LOESS + K16 up, quarter-res
LOESS + 2×K16 up} via the **FP16 G-family** kernels
`galosh_o_box_downsample_2x_3p` / `galosh_o_loess_chroma_3p_fp16_tiled` /
`galosh_o_k16_joint_bilateral_upsample_3p` / `galosh_o_pad_2d_edge` /
`galosh_o_crop_2d_topleft` / `galosh_o_smoothstep_blend_3p` (a zeroed
dummy plane fills the 3rd channel), smoothstep-blends by `cs =
clamp(strength_c, 0, 3)`, and h2f-converts back. `--fp16` replaces
step 10 with f2h → `galosh_fused_pass12` (half, TILE_SIZE=48 dispatch)
→ h2f. Neither is the production PC path.

---

## 3. HOST READBACKS & SYNC POINTS (mid-frame)

| Where | What | Direction | Purpose |
|---|---|---|---|
| After YGQ2b (lines 549-560) | `clFinish` + blocking read `params_buf[0..31]` (128 B) | D→H | get α (`[13]`) / σ² (`[14]`) — **required**: they become direct float args of YG4c/YG4d; also logged |
| End of frame (line 1061-1067) | `clFinish` + blocking read `srgb_buf` (3·npix·4) | D→H | final output |

That is ALL — exactly ONE mid-frame sync. Everything else stays
on-device; the two copy-buffers and one fill are queue ops, not syncs.
In Vulkan: record [YG1..YGQ2b] → fence + 128 B readback → record
[YG3..YG9] → fence + output readback. There is no `GALOSH_DUMP_DIR` /
env-var machinery in this host (unlike RAW).

---

## 4. SCALAR PARAMETER FLOW

CLI: `galosh_yuv_gpu <in.bin> <out.bin> <W> <H> <s_y> <s_c> [cl_dev]
[--variant=o|q] [--fp16] [--pix=… --depth=… --range=… --matrix=…
--eotf=… --siting=…] [--selftest-phase]`. W/H need not be even for the
legacy 444 path (evenness is enforced only by the 420/422 front-end).

- **α / σ² (noise model)**: always blind — estimated on device
  (YGQ2a → params[20], YGQ2b → params[13,14]), read back once, then
  consumed BOTH on device (YG4a via params) and as host float args
  (YG4c/YG4d). No CLI override path exists.
- **σ_gat (params[21])**: device-only round trip
  (YG4a' write → YG4a''/YG4g' read). Never touches the host.
- **strength_y** → direct float arg 4 of `galosh_pass12_o32` (contrast
  with RAW, which passes `strength * luma_str`).
- **strength_c** → direct float arg 4 of `galosh_yuv_guided_loess`;
  enters the math only as `ε = strength_c²` (Bayes-MAP prior in GAT
  space, τ²=1).
- Literal constants passed as kernel args: `x_stride = 3` (both lap_mad),
  `lap_max = 0.5f / 8.0f` (both lap_mad — **unused** by the kernel),
  `o32_phase_stride = 1`, the four params-slot index ints
  (20/13/14 at YGQ2a-b, 21 at YG4a'/YG4a''/YG4g').

---

## 5. NOTES / GOTCHAS for the Vulkan port

1. **The default noise estimator is Laplacian-MAD, not Foi-Alenius.**
   Verified in code: the `if(1)` at line 456 makes the MAD branch
   unconditional; the block-regression branch is statically dead and
   [DEPRECATED] (kept per repo policy). The MAD kernels
   (`k_q_lap_mad`, `k_q_synth_alpha`, `k_q_norm`, `k_q_denorm`) are
   created UNCONDITIONALLY (lines 267-276) precisely because the default
   mode needs them. Port ONLY the MAD path.
2. **The one mid-frame readback can be engineered away** (proven-deviation
   option): the values the host reads back (`params[13]`, `params[14]`)
   are already on-device, and the RAW Vulkan shaders
   `o32_build_inv_lut.comp` / `o32_lut_finalize.comp` read α/σ² **from
   the params buffer** instead of push constants. Reusing them verbatim
   (§6) removes the only mid-frame sync → the whole frame becomes one
   submission chain. If you instead mirror the OpenCL host exactly,
   replicate the readback and push α/σ² as constants. Either way the
   numbers are identical (same device values).
3. **Guide-domain trap (2026-06-28 root cause)**: the chroma LOESS guide
   MUST be the **normalized post-GAT NOISY** Y (`q_y_snap_f`, snapshotted
   BEFORE the LOSH overwrites `y_stab_buf`), not the denoised linear Y.
   Wrong domain/signal made the bilateral weights ~uniform → ~0.0057
   chroma divergence (the dominant CPU↔GPU gap of that era). Preserve
   the snapshot copy point exactly: after YG4a'' norm, before pass12.
4. **In-place buffer mutation chain on `y_stab_buf`** (write-after-read
   hazards for Vulkan barriers): YG4a writes it → YG4a' reads → YG4a''
   read-modify-writes → copy reads → pass12 reads → copy overwrites →
   YG4g' read-modify-writes → YG4h reads. Same for `y_buf`: YG1 writes →
   YGQ2a/YG4a read → YG4h overwrites → YG9 reads. `srgb_buf` is in/out.
   Every arrow needs the appropriate barrier on a single queue.
5. **`yuv_q_lap_mad` internals** (galosh.cl 4752-4806): single WG of 64
   WIs writes \|Lap\| = \|p[x] − 2p[x+2] + p[x+4]\| samples (row-major,
   x ∈ [0, W−4) step 3, capped at `n_samples = min(W*H/6, 200000)` — the
   cap samples only the TOP rows, matching CPU) to `ne_scratch` at
   deterministic indices, `barrier(CLK_GLOBAL_MEM_FENCE)` (valid because
   single-WG), then **WI 0 runs a serial iterative Hoare quickselect over
   global memory** for the exact k-th smallest (k = n/2) and writes
   σ = max(MAD/1.6521, 0.01) to `params[result_idx]`. This replaced a
   histogram median that diverged on clean scenes — the EXACTNESS is the
   point (bit-faithful to CPU `quick_select_kth`). GLSL: `barrier()` +
   `memoryBarrierBuffer()`; keep the serial quickselect (do NOT
   parallelize without a parity gate). The doc comment above the kernel
   still describes the old histogram design — trust the code.
   Note the quickselect **mutates `ne_scratch`** (partition swaps), which
   is fine since the buffer is write-once-per-dispatch scratch.
6. **Dead-but-bound artifacts** (keep or strip — stripping is the B7f-style
   proven deviation): `yuv_guided_loess` arg 3 (`params_buf`) is bound but
   never read by the kernel body; `yuv_chroma_params_derive` (YG3) writes
   params[16..19] that nothing on the O path consumes; `P_YG_EPS_BIV`
   (params[23] = 1e-3) is written by the host and never read. Suggested:
   keep YG3 out of the Vulkan default path (1-WI, zero risk either way)
   and drop the params binding from the LOESS shader; byte-compare once.
7. **pass12 zero-fill**: the host zero-fills `y_den_f32_buf` before
   pass12 (the RAW host does NOT zero `o32_L_cs_den`). pass12 writes
   every covered pixel, but keep the fill for belt-and-braces parity with
   this host (it is one cheap `vkCmdFillBuffer`).
8. **No small-image guard** exists on the YUV path (contrast RAW §2):
   single-scale chroma means no pyramid floor. The 420 front-end imposes
   its own evenness checks instead (§7).
9. **`native_exp` in guided_loess** (galosh.cl 3953): vendor fast-exp —
   deliberately NOT reproducible bit-for-bit across devices; CPU parity
   for chroma is statistical. GLSL `exp()` is the correct mirror (this is
   what the RAW LOESS shaders already use). Do not try to bit-match it.
10. **Sub-normal σ floors**: norm divides by `fmax(σ_gat, 1e-6)`, GAT by
    `fmax(α, 1e-12)`, synth_alpha floors α at 1e-5 / σ² at 1e-8, lap_mad
    floors σ at 0.01, LUT lerp by `fmax(Δd, 1e-10)` — all are
    contract-relevant constants; copy them exactly.
11. **Context lifecycle**: `run_yuv_gat_gpu_buf` creates and destroys the
    ENTIRE OpenCL stack (context, queue, program build, 18+ kernels,
    ~40 buffers) per call. Tolerable for one image; the 420 driver calls
    it TWICE per file (§7). The Vulkan host must init once and reuse
    pipelines across runs (it already does for video; same pattern).

---

## 6. REUSE MAP — existing `standalone/vk/shaders/*.comp` vs NEW ports

Layout premise: RAW o32 = 4 half-res Bayer planes + full-res GAT plane,
2D-indexed everywhere; YUV = **one full-res Y plane + 2 full-res chroma
planes, mostly 1D-indexed** (`gid >= npix` guard). The trivial 1D
kernels have no RAW twin because RAW's equivalents are fused into 2D
Bayer kernels (e.g. `o32_gat_forward_full` writes 5 buffers with a
piecewise VST — NOT the YUV single-plane sqrt-only GAT). Where a twin
exists it is because the YUV path literally dispatches the same CL kernel.

| YUV stage | CL kernel | Existing .comp | Verdict |
|---|---|---|---|
| YG4f LOSH | `galosh_pass12_o32` | `o32_pass12.comp` (classic bit-twin) / `o32_pass12_sg.comp` (B7g subgroup quality) / `o32_pass12_wht4.comp` | **REUSE AS-IS** — same kernel, same push constants (in, out, W, H, strength, phase_stride), same 28/8×8 geometry (or SG local-512 variant). Single-plane input already |
| YG4c LUT build | `galosh_build_inv_lut` | `o32_build_inv_lut.comp` | **REUSE AS-IS** — identical Kahan math (the CL kernel is the back-port OF this shader). Interface difference: the vk shader reads α/σ² from `params_buf` (slots 13/14 — exactly where YUV puts them) and writes lut_params[2..6] from WI 0; combined with o32_lut_finalize the net lut_params[0..6] content equals the YUV pair's output. Enables removing the mid-frame readback (§5.2) |
| YG4d LUT finalize | `galosh_lut_finalize` | `o32_lut_finalize.comp` | **REUSE AS-IS** (writes [0..1] = d_min/d_max; [2..6] come from the build shader) |
| YG1 decompose | `galosh_yuv_srgb_to_linear_ycbcr` | — | **NEW** (trivial 1D) |
| YG9 recompose | `galosh_yuv_ycbcr_to_srgb` | — | **NEW** (trivial 1D) |
| YGQ2a/YG4a' σ-MAD | `galosh_yuv_q_lap_mad` | — (`o32_sigma_hist_mwg`/`o32_sigma_fin_mwg` are a DIFFERENT algorithm — per-CFA histogram MAD — not a substitute for the exact quickselect) | **NEW** (medium; port verbatim incl. serial quickselect) |
| YGQ2b synth | `galosh_yuv_q_synth_alpha_sigma_sq` | — | **NEW** (trivial 1-WI) |
| YG3 derive | `galosh_yuv_chroma_params_derive` | — | **NEW or OMIT** (dead on O path, §5.6) |
| YG4a GAT fwd | `galosh_yuv_gat_forward_Y` | (`o32_gat_forward_full` is NOT a twin: Bayer 2D, 5 outputs, piecewise VST) | **NEW** (trivial 1D) |
| YG4a''/YG4g' (de)normalize | `galosh_yuv_q_unified_sigma_norm` / `_denorm` | (`o32_normalize_apply` is NOT a twin: 2D, 5 buffers) | **NEW** ×2 (trivial 1D) |
| YG4h inverse GAT | `galosh_yuv_makitalo_inverse_Y` | — (`o32_inverse_wht_dark_gat` embeds an equivalent LUT lookup but fused with WHT/dark/denorm — cannibalize its `gat_inv` helper code, don't reuse the shader) | **NEW** (medium-trivial 1D) |
| YG5 chroma LOESS | `galosh_yuv_guided_loess` | `o32_loess_chroma_3p_tiled.comp` is the closest TEMPLATE (same LOESS regression + chroma-band clamp) but differs: 3 planes vs 2, LDS-tiled vs direct global + mirror-reflect edges, different ε formula (`loess_strength` semantics vs `strength_c²`), RAW bandwidth vs `YG_LOESS_BW=3.0` | **NEW** (heavy; transcribe the CL kernel literally — do NOT adapt the tiled RAW shader without a parity gate, the edge handling differs) |
| (Q-variant pyramid) | `galosh_o_{box,loess_fp16,k16,pad,crop,smoothstep}*` | `o32_box_downsample_2x_3p` / `o32_loess_chroma_3p_tiled(_g16)` / `o32_k16_jbu_3p(_f16)` / `o32_pad_2d_edge(_3p)` / `o32_crop_2d_topleft(_h16)` / `o32_smoothstep_blend_3p` | out of scope (Q only); note the CL Q path uses the FP16 **G-family** kernels, the vk shaders are the o32/f16-contract family — an eventual Q port is an ADAPT with re-verification, not a reuse |

**Summary: 3 shaders reused as-is (incl. both LUT stages and the single
heavy LOSH kernel), 10 new ports — of which 8 are trivial (5 per-pixel
1D maps + 3 one-WI scalars) and 2 are real work (`yuv_q_lap_mad`,
`yuv_guided_loess`).**

FP16 fast variant (future): per constraint 5, the YUV FP16 version is a
contract-v1 storage build (f16 stores via `galosh_f16_rne.glsl` on the
inter-phase planes Y_stab/y_den/q_y_snap/cb/cr), reusing the same shader
set with `float16_t` storage bindings — mirror how `o32_pass12` /
`*_f16.comp` / `*_h16.comp` split storage vs compute; NOT a port of the
OpenCL `galosh_fused_pass12` mobile kernel.

---

## 7. GALOSH-420 COMPOSITION (planar front-end, 2026-07-11)

Source: `run_yuv420_gpu()` (galosh_yuv_gpu.c lines 1262-1449) +
`standalone/galosh_yuv420.h` (**host-side, shared verbatim with the CPU
reference driver — the Vulkan host must include the SAME header** so the
flag vocabulary and siting phases cannot drift).

Activated by `--pix=420|422|444|400` (any `--pix` sets planar mode).
I/O = planar integer container: Y plane then Cb then Cr, `uint8` for
depth 8 / `uint16` little-endian for depth 9-16 (`wide = depth > 8`).
Chroma lattice: 420 → (W/2)×(H/2) (W,H must be even), 422 → (W/2)×H
(W even), 444 → W×H, 400 → none.

### Composition (the core is called TWICE, unchanged)

```
decode:  codes → float gamma planes        galosh420_dequant_y/_c  (no clamp;
                                           sub-black/super-white preserved)

LUMA (full res):
  gray[i].rgb = eotf_fwd_srgb(clamp01(eotf_inv(Y'[i], eotf)))   /* gray image */
  run_yuv_gat_gpu_buf(gray, W, H, s_y, s_c)                      /* CORE #1 */
  Y'_den[i] = eotf_fwd(clamp01(eotf_inv_srgb(gray[3i+1])), eotf) /* G channel! */

CHROMA (native lattice, skipped for --pix=400):
  guide  = galosh420_down_luma(Y', siting)     /* 420: siting-phased ½-res Y'  */
           (422: Y' as-is + up422_h chroma → full-res run; 444: as-is)
  rgb    = eotf_fwd_srgb(eotf_inv(ncl_inv(guide, Cb, Cr, Kr/Kb), eotf))
  run_yuv_gat_gpu_buf(rgb, pw, ph, s_y, s_c)                     /* CORE #2 */
  (—, Cb_den, Cr_den) = ncl_fwd(eotf_fwd(eotf_inv_srgb(rgb), eotf))
           (Y of the reconstruction is DISCARDED; 422: down422_h back)

encode:  galosh420_requant_y/_c  (lrintf + clamp to [0, 2^depth−1]) → same
         container layout (format-preserving)
```

The sRGB re-encode adapter exists because the core's entry kernel is
`srgb_to_linear_ycbcr`; the round trip is float-exact to ~1e-7. All
transforms between the two GPU runs are **host-side FP32 loops** (EOTF
pairs incl. HLG/PQ, H.273 NCL matrix, siting kernels box/tent — see the
header for exact formulas and the 4-layer siting framework). Port them by
INCLUDING the header, not by re-implementing. `--selftest-phase` runs
`galosh420_phase_selftest()` (affine-field phase proof) — wire it up in
the Vulkan CLI too.

### Vulkan-specific requirements

1. **One device init, two frames**: the OpenCL version pays full
   context + program build twice (§5.11). The Vulkan host runs core #1
   and core #2 as two frames on the SAME pipelines/queue, only
   reallocating the size-dependent buffers (the chroma run is
   (W/2)×(H/2) for 420 — quarter area, so either a second buffer set or
   suballocation).
2. **Luma-only fast path** (gray run, and the entirety of `--pix=400`):
   the gray image has Cb = Cr ≈ 0 (exactly: `(B−Y)·k` where
   Y = (KR+KG+KB)·v — the FP32 constant sum is not exactly 1, so chroma
   is ~1e-7·v, not 0). The clean skip in the §2 sequence:
   - **run unchanged**: YG1, YGQ2a/b, readback, YG4a…YG4h (the whole
     luma chain — it never touches cb/cr);
   - **skip**: YG3 (chroma_derive), the `y_stab → q_y_snap_f` snapshot
     copy after YG4a'' (guide only feeds chroma), and YG5 guided_loess;
   - **rebind**: YG9 takes `cb_buf`/`cr_buf` (the untouched YG1 outputs)
     instead of `cb_biv_buf`/`cr_biv_buf`.
   This is numerically clean but **not bit-identical** to running the
   full core on the gray image (guided_loess of a ~1e-7 chroma plane
   returns ~1e-7, not the identical bits): the residual enters the luma
   decode only through the G channel's −(KR·CR_DEN/KG)·Cr −
   (KB·CB_DEN/KG)·Cb terms, i.e. O(1e-7) linear ≪ one 16-bit code.
   Ship the fast path as default with a `--no-luma-fast` style toggle (or
   env) for A/B parity testing against the OpenCL composition.
3. The banding rate state (`g_band_rate_ms`) learned on the full-res
   luma run over-estimates per-row cost for the quarter-area chroma run —
   harmless (more bands than needed), no action required.
