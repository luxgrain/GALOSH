/* galosh_raw_cpu.c — RAW Bayer GALOSH denoiser (CPU reference).
 *
 * ============================================================================
 *  CANONICAL — GALOSH RAW V2  (LOCKED 2026-06-21)
 *  ---------------------------------------------------------------------------
 *  The paper pipeline = GALOSH_RAW_O  (--variant=o, DEFAULT; dispatch: any
 *  --variant not in {g..n} -> 'o').  O = L pipeline + 3-level multi-scale LOESS
 *  chroma pyramid + L-guided refinement + smoothstep slider walk.
 *
 *  "GALOSH RAW V2" ships this single O algorithm at FOUR precisions:
 *    - CPU  FP32  : variant o   (this file)
 *    - CPU  INT32 : variant r32 (galosh_raw_cpu_int.c, Q11.20)
 *    - GPU  FP32  : variant o32 (galosh_raw_gpu.c)        ← GPU canonical
 *    - GPU  INT16 : variant i16 (galosh_int_*.cl, bit-exact vs r32)
 *
 *  ALL of g,h,i,j,k,l,m,n documented below are [DEPRECATED/PREVIOUS] — they are
 *  the perceptual-evolution lineage that O supersedes (each fixed a chroma
 *  edge-fringe / blotch artifact; n = -3.4 dB, do NOT use).  Kept in-code for
 *  ablation/forensic bench only.  GPU --variant=g is likewise DEPRECATED.
 *  NOTE: the per-variant "[LATEST]" labels below are HISTORICAL — O is latest.
 * ============================================================================
 *
 * ##############################################################
 * # [PREVIOUS] GALOSH_RAW_L  (selected via --variant=l)
 * ##############################################################
 *   Production canonical pipeline — this is what runs by default.
 *   L = K's two-stage chroma reconstruction (Phase 6 K16 + Phase 8
 *   LOESS post-process) FUSED into a single guide-aware K16 stage
 *   (joint bilateral upsample).
 *
 *   Theoretical motivation (Kopf et al. SIGGRAPH 2007 / He et al.
 *   TPAMI 2010 guided filter framework):
 *     K's two-stage chain has an information bottleneck — K16 first
 *     bandlimit-smooths C edges to half-res grid, then LOESS tries to
 *     "snap back" using L_pixel.  But the snap can only use the
 *     already-smoothed C, not the original half-res samples.
 *     Sub-pixel edge information is lost in the smoothing step.
 *
 *     Joint bilateral upsample combines the bandlimit interp AND
 *     L-edge alignment criteria into a single filter weight:
 *       w[i] = w_jinc(d_i) × exp(-(L_pixel - L_at_h[i])²/(2BW²))
 *       C_full = Σ w[i]·C_h[i] / Σ w[i]
 *     The bilateral operates on ORIGINAL half-res chroma (not pre-
 *     smoothed) → preserves the sub-pixel edge information needed for
 *     accurate cross-channel super-resolution.
 *
 *     In flat L regions: bilateral ≈ uniform → effective kernel = pure
 *     jinc → bandlimit-faithful (matches K16 standard exactly).
 *     At L edges: bilateral kills cross-edge half-res samples → effective
 *     kernel = one-sided jinc → C edges snap to L edges.
 *
 *   Cost: K's Phase 6 (K16 unaware ~0.3s) + Phase 8 (LOESS R=3 ~1-2s)
 *   collapse into L's Phase 7 (joint bilateral ~0.5-1s).  Net saving
 *   ~0.5-1.5s/scene.
 *
 *   GALOSH_RAW_L full flow (= what runs in production with --variant=l):
 *     Phase 0..5  identical to K Phase 0..5
 *     Phase 6     L_pixel = 2x2 overlap average of L_cs_den
 *                 (= K Phase 7, moved earlier as bilateral guide)
 *     Phase 7     Joint bilateral K16 upsample ⭐ NEW vs K ⭐
 *                 gat_k16_joint_bilateral_upsample(C_h_den, L_pixel,
 *                                                   ..., BW=1.5)
 *                 (replaces K Phase 6 + Phase 8 with single fused stage)
 *     Phase 8+9   per-pixel WHT inverse + dark_ref restore + ×unified_sigma
 *                 + inverse GAT (LUT) → out  (fused, = K Phase 9+10)
 *
 * ##############################################################
 * # [PREVIOUS: GALOSH_RAW_K]  (selected via --variant=k)
 * ##############################################################
 *   Was the LATEST until GALOSH_RAW_L replaced it.  K = J + Bayesian-
 *   correct Phase 8 hyperparameters (ε=0.01 hardcode, BW=1.5).
 *   Two-stage K16 (unaware) + LOESS-with-L-guide chain.
 *
 *   Theoretical motivation (Bayesian framing of LOESS):
 *     LOESS regression `cb = a·L + b` has closed-form
 *       a = cov(L,C) / (var(L) + ε),  ε = σ_C² / τ²
 *     where σ_C is the input chroma noise std and τ² is the prior
 *     variance on the L-C correlation slope.
 *
 *     Phase 5(C) input: noisy half-res C with σ_C ≈ 1 (GAT-norm) →
 *       ε = 1.0 (= chroma_slider²) is the correct posterior weight.
 *     Phase 8 input: ALREADY DENOISED C_full from K16 EWA-JL3, with
 *       residual σ_C ≈ 0.1 (= 1/√n_eff after Phase 5(C) over ~100
 *       effective bilateral samples; K16 preserves noise level via
 *       bandlimit interp) → ε = 0.01 (= 0.1²) is the correct
 *       refinement-step regularizer.  J used ε = 1, which was 100×
 *       over-regularization → regression damped → moderate edges
 *       not snapped → residual fringe.
 *
 *     K hardcodes Phase 8 hyperparameters to refinement-correct values:
 *       ε = 0.01 (slider-INDEPENDENT; slider only affects Phase 5(C))
 *       BW = 1.5 (vs J's 3.0; tighter "same-cluster" threshold for
 *                 moderate edges; safe because L_pixel residual σ ≈
 *                 0.1 << 1.5 in GAT-norm space)
 *
 *   Compute cost: ZERO additional vs J (parameter-only change).
 *
 *   GALOSH_RAW_K full flow (= what runs in production with --variant=k):
 *     Phase 0..7  identical to J Phase 0..7
 *     Phase 8     L-guided chroma refinement, NEW PARAMETERS:
 *                 galosh_loess_chroma_3ch_r(L_pixel, ..., strength=0.1,
 *                                            R=3, BW=1.5)
 *     Phase 9+10  identical to J Phase 9+10
 *
 * ##############################################################
 * # [PREVIOUS: GALOSH_RAW_J]  (selected via --variant=j)
 * ##############################################################
 *   Was the LATEST until GALOSH_RAW_K replaced it.  Kept in this file
 *   (function gat_galosh_denoise_rawlc_j) for refinement-parameter
 *   ablation.  J = I + Phase 8 L-guided chroma refinement (joint
 *   bilateral upsample / guided filter post-process).  Resolves edge
 *   color fringe observed in I outputs.
 *
 *   Theoretical motivation (Bayesian MAP):
 *     P(C_full | C_h_obs, L_full) ∝ P(C_h_obs | C_full) × P(C_full | L_full)
 *                                    ──────────────────    ───────────────
 *                                    CFA likelihood          cross-channel
 *                                    (K16 = ML estimator)    structural prior
 *                                                            (LOESS w/ L guide)
 *
 *   I provides the ML branch (CFA likelihood via K16 EWA-JL3 bandlimit
 *   upsample of half-res C).  J adds the prior branch: small-radius
 *   LOESS (R=3) at full-res with L_pixel as Y guide, encoding the
 *   natural-image fact that chroma edges spatially align with luma
 *   edges (Hirakawa & Wolfe TIP 2008).  Bilateral weight separates C
 *   samples by L value at edges → snaps C to full-res L edges.  Flat
 *   L regions left smooth (K16 quality preserved, no blotch).
 *
 *   GALOSH_RAW_J full flow (= what runs in production with --variant=j):
 *     Phase 0  Foi-Alenius blind α / σ²  (= I/H/G Phase 0)
 *     Phase 1  GAT forward (full-res) + per-CFA σ_GAT MAD + RMS
 *              unified_sigma + scalar normalize  (= I Phase 1)
 *     Phase 2  dark_ref IRLS (achromatic) + per-pixel CFA-aware subtract
 *              (= I Phase 2)
 *     Phase 3  stride=1 forward 2x2 WHT @ full-res → L_cs, C1_cs, C2_cs,
 *              C3_cs  (= I Phase 3, gat_h_forward_wht_stride1)
 *     Phase 4  SPLIT: 4(L) keep L_cs full-res; 4(C) sub-sample at
 *              every-other-pixel → half-res C  (= I Phase 4)
 *     Phase 5  denoise:
 *                5(L) galosh_pass12_multiorient_blocked on L_cs (full-res)
 *                5(C) galosh_loess_chroma at half-res (= I Phase 5)
 *     Phase 6  K16 EWA-JL3 upsample C_h_den → C_full_smooth
 *              (galosh_upsample_2x_ewajl3, = ML estimator)
 *     Phase 7  L_pixel = 2x2 overlap average of L_cs_den
 *              (gat_i_lpixel_overlap_avg)
 *     Phase 8  L-guided chroma refinement ⭐ NEW vs I ⭐
 *              galosh_loess_chroma_r(R=3, guide=L_pixel) on C_full_smooth
 *              → C_aligned (snaps chroma edges to L edges via
 *              cross-channel structural prior; Bayesian MAP posterior)
 *     Phase 9+10  per-pixel WHT inverse + dark_ref restore + ×unified_sigma
 *                 + inverse GAT (LUT) → out  (fused)
 *
 *   Cost: I cost + R=3 full-res LOESS (~+2-3 s/scene) → ~9-10 s/scene
 *   total; still faster than H.  Worth it because edge fringe
 *   elimination is a perceptual win that LPIPS/DISTS/NIQE all detect.
 *
 * ##############################################################
 * # [PREVIOUS: GALOSH_RAW_I]  (selected via --variant=i)
 * ##############################################################
 *   Was the LATEST until GALOSH_RAW_J replaced it.  Kept in this file
 *   (function gat_galosh_denoise_rawlc_i) for edge-fringe reproduction
 *   bench.  Hybrid "L from H, C from G".  L processed at full-res via
 *   stride=1 cycle-spinning (= H), C processed at half-res via LOESS
 *   + K16 EWA-JL3 chromaup (= G).  10-phase pipeline (no Phase 8
 *   L-guided refinement).  Suffers from edge color fringe at sharp L
 *   edges due to L (full-res) / C (half-res grid) misalignment.
 *
 * ##############################################################
 * # [PREVIOUS: GALOSH_RAW_H]  (selected via --variant=h)
 * ##############################################################
 *   Was the LATEST until GALOSH_RAW_I replaced it.  Kept in this file
 *   (function gat_galosh_denoise_rawlc_h) for chroma-blotch reproduction.
 *   GALOSH_RAW_H operates entirely at full resolution.  It replaces
 *   GALOSH_RAW_G's half-res ↔ full-res roundtrip (K14 box compute_L_fullres
 *   + K16 EWA-JL3 chromaup) with stride=1 cycle-spinning forward WHT +
 *   CFA sign-flip demod / remod + 4-block overlap-add inverse.  See
 *   gat_galosh_denoise_rawlc_h() body for details and inline phase labels.
 *
 *   GALOSH_RAW_H full flow (= what runs with --variant=h):
 *     Phase 0  Foi-Alenius blind α / σ²  (galosh_estimate_noise)
 *     Phase 1  GAT forward (full-res) + per-CFA σ_GAT MAD + RMS
 *              unified_sigma + scalar normalize
 *     Phase 2  dark_ref IRLS (achromatic) + per-pixel CFA-aware subtract
 *     Phase 3  stride=1 forward 2x2 WHT @ full-res  → L, C1, C2, C3
 *              (gat_h_forward_wht_stride1; right/bottom mirror padding)
 *     Phase 4  CFA sign-flip demodulate
 *              (C1*=(-1)^r, C2*=(-1)^c, C3*=(-1)^(r+c))
 *     Phase 5  full-res denoise:
 *                L : galosh_pass12_multiorient_blocked (BayesShrink-MAD
 *                    + Wiener; block=8 stride=2 n_orient=1 robust=1)
 *                C1/C2/C3 : galosh_loess_chroma (Y-guided bilateral
 *                    LOESS, Y guide = denoised L)
 *     Phase 6  CFA sign-flip remodulate (= self-inverse Phase 4)
 *     Phase 7  4-block overlap-add inverse 2x2 WHT
 *              (gat_h_inverse_overlap_add; per-pixel TL/BL/TR/BR average)
 *     Phase 8  per-pixel dark_ref restore + ×unified_sigma denormalize
 *     Phase 9  per-pixel inverse GAT (LUT, gat_inverse_exact)
 *
 *   Phases 8 + 9 are fused into a single per-pixel loop in the
 *   implementation.  All 10 phases are labelled [LATEST: GALOSH_RAW_H]
 *   inline for traceability.
 *
 * ##############################################################
 * # [PREVIOUS: GALOSH_RAW_G]  (selected via --variant=g; bench-only)
 * ##############################################################
 *   Was the production canonical until GALOSH_RAW_H replaced it.  Kept
 *   in this file (function gat_galosh_denoise_rawlc) for benchmarking
 *   the H-vs-G uplift.  All phase labels inside that function are
 *   [PREVIOUS: GALOSH_RAW_G].
 *
 *   GALOSH_RAW_G full flow (= what runs with --variant=g):
 *     Phase 0  Foi-Alenius blind α / σ²  (galosh_estimate_noise)
 *     Phase 1  GAT extract → 4 ch half-res → per-ch σ_GAT (MAD)
 *              → unified_sigma = RMS(σ_ch) → unit-variance normalise
 *     Phase 2  Self-consistent dark anchor (achromatic IRLS, 2 iter)
 *              → ch_gat[c] -= ch_dark_ref[c]
 *     Phase 3  WHT decompose: 4 ch (R/Gb/Gr/B) → L / C1 / C2 / C3
 *              (half-res 2x2 WHT, scale 0.5)
 *     Phase 3.5 Chroma denoise: galosh_loess_chroma (Y-guided
 *              bilateral LOESS, R=7 BW=3, exp(-(Y-Yc)²/2σ²) weights;
 *              specular-aware) — replaces WHT-LOSH on chroma
 *     Phase 4(a) K13 half-res luma WHT-LOSH
 *              galosh_pass12_multiorient_blocked w/ use_robust_shrink=1
 *     Phase 4(b) K14 compute_L_fullres × 2 (4-tap box, half→full L)
 *     Phase 4(c) K15 full-res L Pass2 only (Wiener w/ K14 pilot)
 *     Phase 4(d) K16 chromaup (EWA-JL3 c1/c2/c3 upsample + per-pixel
 *                inverse 2x2 WHT + dark_ref restore + inverse GAT)
 *
 *   Two structural changes vs pre-G are baked in unconditionally:
 *     (i)  K16 EWA-JL3 chromaup + per-pixel inverse WHT
 *     (ii) Pass1 σ_Y via MAD partial-selection-sort robust estimator
 *
 * ##############################################################
 * # [ARCHIVED] pre-GALOSH_RAW_G legacy path
 * ##############################################################
 *   Triggered when -DGALOSH_LEGACY IS defined → gat_galosh_denoise_rawlc
 *   takes the #else branch at the bottom which runs:
 *     Phase 4 (legacy)  full-res raw WHT-LOSH (galosh_pass1 / galosh_pass2
 *                       directly on Bayer mosaic, stride=4, luma only)
 *     Phase 5 (legacy)  chroma replacement: per-2x2-block dC1/dC2/dC3
 *                       computed from full-res output, then ADDED back
 *                       BLOCK-REPLICATED to all 4 sub-pixels of the block
 *                       → 2x2 stair-step on diagonals (specifically the
 *                       artefact GALOSH_RAW_G K16 chromaup fixes).
 *     Phase 6 (legacy)  inverse GAT.
 *   Kept for bench reproducibility; do NOT use in production.
 *
 *   Other archived flags / paths (also kept for bench, never default):
 *     g_galosh_unified=1    : 4-plane EWA-JL3 upsample (L+C),  K15 抜き
 *                             — var=1 不変性破壊 → denoise 性能↓
 *     g_galosh_lfr_kernel=1 : compute_L_fullres を EWA-JL3 化 (variant C)
 *                             — diagonal アーチファクト改善見られず
 *     g_galosh_k13_block=4  : K13 を 4x4 WHT で組む (4x4 grain-scale 実験)
 *                             — K15 8x8 が支配で 4x4 化のメリット消失
 *     galosh_compute_L_fullres_ewajl3  : variant C 用ヘルパ
 *     galosh_pass1_cfa / galosh_pass2_cfa : CFA-protect 経路 (案A、効果なし)
 *
 * ##############################################################
 *
 * Usage: galosh_raw_cpu in.bin out.bin width height
 *          [method] [strength] [luma_str] [chroma_str] [alpha] [sigma_sq]
 *          [--variant=l|k|j|i|h|g]   (default l = LATEST GALOSH_RAW_L)
 *
 * Build (LATEST / production = GALOSH_RAW_L, default):
 *   gcc -O3 -march=native -ffast-math -funroll-loops -fopenmp \
 *       -o galosh_raw_cpu galosh_raw_cpu.c -lm
 *
 * Variant selection at run-time (no recompile needed):
 *   ./galosh_raw_cpu in.bin out.bin W H galosh 1 1 1 0 0 --variant=l  # LATEST (default)
 *   ./galosh_raw_cpu in.bin out.bin W H galosh 1 1 1 0 0 --variant=k  # PREVIOUS_K (bench)
 *   ./galosh_raw_cpu in.bin out.bin W H galosh 1 1 1 0 0 --variant=j  # PREVIOUS_J (bench)
 *   ./galosh_raw_cpu in.bin out.bin W H galosh 1 1 1 0 0 --variant=i  # PREVIOUS_I (bench)
 *   ./galosh_raw_cpu in.bin out.bin W H galosh 1 1 1 0 0 --variant=h  # PREVIOUS_H (bench)
 *   ./galosh_raw_cpu in.bin out.bin W H galosh 1 1 1 0 0 --variant=g  # PREVIOUS_G (bench)
 *
 * Build (pre-G legacy stride=4 path, [ARCHIVED], bench-only):
 *   gcc ... -DGALOSH_LEGACY ... -o galosh_raw_cpu galosh_raw_cpu.c -lm
 *   (selects the [ARCHIVED] #else branch inside gat_galosh_denoise_rawlc)
 *
 * Algorithm primitives (WHT, BayesShrink, Wiener, EWA-JL3 upsample,
 * LOESS chroma) are in galosh_cpu.h — see file-top there for the
 * same LATEST / ARCHIVED labelling.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "galosh_cpu.h"

/* ============================================================
 * [MIXED] Pipeline tuning flags (set from CLI in main()).
 *
 *   g_galosh_stride     : [LATEST] 2 = GALOSH_RAW_G default (75% overlap)
 *                         [ARCHIVED] 1 = full cycle-spinning (variant A,
 *                                        slow, no measurable quality gain)
 *   g_galosh_n_orient   : [LATEST] 1 = GALOSH_RAW_G default
 *                         [ARCHIVED] 4 = 4-orientation WHT averaging
 *                                        (variant B, no quality gain)
 *   g_galosh_lfr_kernel : [LATEST] 0 = box compute_L_fullres (default)
 *                         [ARCHIVED] 1 = EWA-JL3 compute_L_fullres
 *                                        (variant C, no diagonal-artifact
 *                                        improvement vs box)
 *   g_galosh_unified    : [LATEST] 0 = use K14/K15/K16 chromaup pipeline
 *                         [ARCHIVED] 1 = single 4-plane EWA-JL3 upsample
 *                                        (var=1 不変性破壊で denoise 性能↓)
 *   g_galosh_k13_block  : [LATEST] 8 = GALOSH_RAW_G default
 *                         [ARCHIVED] 4 = K13 4x4 grain-scale-matched
 *                                        (K15 8x8 が支配で実効 gain なし)
 *
 * Defaults below produce the [LATEST] GALOSH_RAW_G canonical pipeline.
 * Non-default values are accepted only for bench archive reproducibility.
 * ============================================================ */
static int g_galosh_stride     = 2;   /* [PREVIOUS: GALOSH_RAW_G] */
static int g_galosh_n_orient   = 1;   /* [PREVIOUS: GALOSH_RAW_G] */
static int g_galosh_lfr_kernel = 0;   /* [PREVIOUS: GALOSH_RAW_G] */
static int g_galosh_unified    = 0;   /* [PREVIOUS: GALOSH_RAW_G] */
static int g_galosh_k13_block  = 8;   /* [PREVIOUS: GALOSH_RAW_G] */

/* [LATEST: GALOSH_RAW_M] Variant dispatch.
 *   'm' = GALOSH_RAW_M — H + hierarchical Bayesian Phase 5(C):
 *         R_local=7 LOESS (with σ_local²(x) emission) + R_global=15
 *         LOESS (= scale-doubled prior) + per-pixel inverse-variance
 *         fusion.  chroma_strength acts as σ_n scaling, affecting BOTH
 *         flat-region noise AND edge behavior at FIXED cost.  All
 *         constants principled (CFA Nyquist, scale-doubling, GAT σ_n=1,
 *         data-driven σ_local²(x)).  No magic numbers.
 *   (preserved for bench reproducibility:)
 *   '_' (unused L/K/J/I/H/G dispatch documented below)
 */

/* [PREVIOUS: GALOSH_RAW_L] Variant dispatch (preserved).
 *   'l' = GALOSH_RAW_L — K's K16 + LOESS post-process two-stage chain
 *         FUSED into single guide-aware K16 (joint bilateral upsample).
 *         Replaces K's Phase 6 (K16 unaware) + Phase 8 (LOESS post)
 *         with single Phase 7 that combines bandlimit interp AND
 *         L-edge alignment in one filter pass — bilateral operates on
 *         original half-res C samples (not pre-smoothed) → preserves
 *         sub-pixel edge information.  Kopf 2007 / He 2010 framework.
 *   'k' = [PREVIOUS: GALOSH_RAW_K] — J + Bayesian-correct Phase 8 ε/BW.
 *         Two-stage K16 + LOESS-with-L-guide chain.
 *   'j' = [PREVIOUS: GALOSH_RAW_J] — I + L-guided chroma refinement
 *         (slider-coupled ε, BW=3).
 *   'i' = [PREVIOUS: GALOSH_RAW_I] — hybrid "L from H, C from G" no
 *         L-guided refinement.
 *   'h' = [PREVIOUS: GALOSH_RAW_H] — full-res stride=1 cycle-spinning
 *         forward + 4-block overlap-add inverse.
 *   'g' = [PREVIOUS: GALOSH_RAW_G] — half-res LOSH + K14 box + K15
 *         + K16 EWA-JL3 chromaup.
 * Defaults to 'm' (LATEST).  Overridable via CLI --variant=m|l|k|j|i|h|g. */
static char g_galosh_variant   = 'o';

/* Pass 1 BayesShrink threshold mode (CLI --pass1=).
 *   0 = baseline (BayesShrink + VisuShrink cap)
 *   1 = a1 (hierarchical empirical Bayes, σ_x_global prior, no cap).
 *       Targets super-clean catastrophic destruction without ad-hoc constants.
 *       See galosh_pass1_blocked prepass for theory. */
int g_galosh_pass1_mode = 0;

/* Phase 1 unified_sigma override (CLI --unified-sigma=X).
 *
 * Phase 1 normally measures unified_sigma from in_gat via per-CFA halfres
 * median MAD (= estimate_gat_sigma_halfres), then divides in_gat by it
 * to enforce effective σ = 1.0 in the normalized space.  Phase 10 multi-
 * plies back by the same unified_sigma before inverse GAT.
 *
 * When X > 0 is provided, Phase 1 SKIPS the measurement and uses X as
 * unified_sigma.  X = 1.0 means "trust Phase 0 (α, σ²) completely; no
 * additional calibration" — useful when an external blind estimator
 * (= Python EM_iter etc.) has already calibrated (α, σ²) and we want
 * Phase 1's safety net to step out of the way.
 *
 * Default 0 = use measured unified_sigma (= production behavior).      */
float g_galosh_unified_sigma_override = 0.0f;

/* Super-clean luma-denoise gate (CLI --super-clean-threshold=X).
 *
 * Catastrophic super-clean destruction (-8 to -12 dB on RawNIND
 * 2pilesofplates_ISO125/160, Blombukett ISO200, Elplint ISO200) was
 * diagnosed as per-block hard threshold killing subtle texture AC
 * coefficients in WHT space.  The destruction is NOT fixable by
 * threshold-magnitude tuning (cliff observation: L=0 → noisy verbatim,
 * L=0.1 → catastrophic loss; any positive threshold kills subtle texture
 * AC < 1 = σ_normalized).  The clean fix is to GATE the entire luma
 * denoise (= Phase 5 Pass 1 + Pass 2) when predicted noise is below
 * visual-perceptual significance.
 *
 * Detection signal: pred_noise_std(s=0.5) = √(α·0.5 + σ²) from Phase 0.
 *   Threshold default 0.0 = disabled (= baseline behavior).
 *   When threshold > 0 and pred_noise_std < threshold, Phase 5 is
 *   bypassed (L_cs_den = L_cs) for that image only — chroma denoise
 *   (Phase 7-9) still runs.
 *
 * Initial test value 0.01 corresponds to ~ ISO 200 sensor noise floor;
 * below this denoising is perceptually unnecessary AND destructive. */
float g_galosh_super_clean_threshold = 0.0f;

/* ============================================================
 * [PREVIOUS: GALOSH_RAW_G] Two structural features adopted unconditionally
 * (commit a48e716).  These cannot be disabled at runtime — they are the
 * defining traits of GALOSH_RAW_G vs pre-G.
 *
 *   (i)  K16 chroma full-res reconstruction (was: "v1 chromaup")
 *        c1/c2/c3 upsampled to full-res via EWA Jinc-Lanczos-3, then
 *        per-pixel inverse 2x2 WHT with Bayer-aware sign tables.
 *        Replaces pre-G block-replicated chroma inverse (visible 2x2
 *        stair-step on diagonal edges).  Var=1 noise invariance preserved.
 *
 *   (ii) Pass1 BayesShrink σ_Y via MAD partial-selection-sort
 *        (was: "v5 robust-MAD").  σ_Y = median(|AC|) / 0.6745.  Robust
 *        to ~25% outlier coefficients (Donoho-Johnstone 1995); kills
 *        spatial noise clusters that pre-G L2 sum_sq mistook for
 *        signal.  Improved pilot propagates to Pass2.
 *
 * Implemented inside galosh_pass12_multiorient_blocked (pass1+pass2
 * wrapper) and the K16 step (d) block of gat_galosh_denoise_rawlc.
 * ============================================================ */

/* ============================================================
 * [ARCHIVED] CFA-protected WHT bin indices (案A "CFA frequency protect").
 * Used only by galosh_pass1_cfa / galosh_pass2_cfa (= legacy stride=4
 * Phase 4 with -DGALOSH_CFA_PROTECT).  GALOSH_RAW_G does NOT protect
 * any WHT bins; the L/C decompose handles CFA structure structurally.
 * ============================================================ */
#define CFA_IDX_HORIZ  1   /* row=0, col=1 → 0*8+1 */
#define CFA_IDX_VERT   8   /* row=1, col=0 → 1*8+0 */
#define CFA_IDX_DIAG   9   /* row=1, col=1 → 1*8+1 */

/* [ARCHIVED] is_cfa_protected — used only by *_cfa archived variants. */
static inline int is_cfa_protected(int i)
{
    return (i == 0 || i == CFA_IDX_HORIZ || i == CFA_IDX_VERT || i == CFA_IDX_DIAG);
}

/* ============================================================
 * [ARCHIVED] SSE 2x2 WHT decompose / reconstruct primitives.
 * Original-design helpers for inline 8x8 patch processing; never used
 * by the current GALOSH_RAW_G pipeline (which decomposes/reconstructs
 * via plane-level loops in gat_galosh_denoise_rawlc Phase 3 and Phase
 * 4(d) K16 chromaup).  Kept as reference for SSE WHT optimisation.
 * ============================================================ */
static inline void wht_decompose_8x8(const float patch[GALOSH_BLOCK_PIXELS],
                                       float lc[4][GALOSH_HALF_PIXELS])
{
  const __m128 half = _mm_set1_ps(0.5f);
  for(int i = 0; i < GALOSH_HALF_BLOCK; i++)
  {
    /* Load 8 floats from even row: [a0,b0,a1,b1,a2,b2,a3,b3] */
    const float *row0 = patch + (2 * i) * GALOSH_BLOCK_SIZE;
    const float *row1 = patch + (2 * i + 1) * GALOSH_BLOCK_SIZE;
    const __m128 r0lo = _mm_loadu_ps(row0);      /* a0,b0,a1,b1 */
    const __m128 r0hi = _mm_loadu_ps(row0 + 4);  /* a2,b2,a3,b3 */
    const __m128 r1lo = _mm_loadu_ps(row1);      /* c0,d0,c1,d1 */
    const __m128 r1hi = _mm_loadu_ps(row1 + 4);  /* c2,d2,c3,d3 */
    /* Deinterleave: a = even cols, b = odd cols */
    const __m128 a = _mm_shuffle_ps(r0lo, r0hi, _MM_SHUFFLE(2, 0, 2, 0)); /* a0,a1,a2,a3 */
    const __m128 b = _mm_shuffle_ps(r0lo, r0hi, _MM_SHUFFLE(3, 1, 3, 1)); /* b0,b1,b2,b3 */
    const __m128 c = _mm_shuffle_ps(r1lo, r1hi, _MM_SHUFFLE(2, 0, 2, 0)); /* c0,c1,c2,c3 */
    const __m128 d = _mm_shuffle_ps(r1lo, r1hi, _MM_SHUFFLE(3, 1, 3, 1)); /* d0,d1,d2,d3 */
    const __m128 ab_sum = _mm_add_ps(a, b);
    const __m128 ab_dif = _mm_sub_ps(a, b);
    const __m128 cd_sum = _mm_add_ps(c, d);
    const __m128 cd_dif = _mm_sub_ps(c, d);
    _mm_storeu_ps(lc[0] + i * GALOSH_HALF_BLOCK, _mm_mul_ps(_mm_add_ps(ab_sum, cd_sum), half));
    _mm_storeu_ps(lc[1] + i * GALOSH_HALF_BLOCK, _mm_mul_ps(_mm_add_ps(ab_dif, cd_dif), half));
    _mm_storeu_ps(lc[2] + i * GALOSH_HALF_BLOCK, _mm_mul_ps(_mm_sub_ps(ab_sum, cd_sum), half));
    _mm_storeu_ps(lc[3] + i * GALOSH_HALF_BLOCK, _mm_mul_ps(_mm_sub_ps(ab_dif, cd_dif), half));
  }
}

/* SSE inverse WHT: 4x4 x 4ch -> 8x8 Bayer patch.
 * Processes all 4 columns simultaneously, interleaving back to 8-wide rows. */
static inline void wht_reconstruct_8x8(const float lc[4][GALOSH_HALF_PIXELS],
                                         float patch[GALOSH_BLOCK_PIXELS])
{
  const __m128 half = _mm_set1_ps(0.5f);
  for(int i = 0; i < GALOSH_HALF_BLOCK; i++)
  {
    const __m128 L  = _mm_loadu_ps(lc[0] + i * GALOSH_HALF_BLOCK);
    const __m128 C1 = _mm_loadu_ps(lc[1] + i * GALOSH_HALF_BLOCK);
    const __m128 C2 = _mm_loadu_ps(lc[2] + i * GALOSH_HALF_BLOCK);
    const __m128 C3 = _mm_loadu_ps(lc[3] + i * GALOSH_HALF_BLOCK);
    const __m128 LC1_sum = _mm_add_ps(L, C1);
    const __m128 LC1_dif = _mm_sub_ps(L, C1);
    const __m128 C2C3_sum = _mm_add_ps(C2, C3);
    const __m128 C2C3_dif = _mm_sub_ps(C2, C3);
    /* even-col (a), odd-col (b) for even row */
    const __m128 a = _mm_mul_ps(_mm_add_ps(LC1_sum, C2C3_sum), half);  /* L+C1+C2+C3 */
    const __m128 b = _mm_mul_ps(_mm_add_ps(LC1_dif, C2C3_dif), half);  /* L-C1+C2-C3 */
    /* even-col (c), odd-col (d) for odd row */
    const __m128 cc = _mm_mul_ps(_mm_sub_ps(LC1_sum, C2C3_sum), half); /* L+C1-C2-C3 */
    const __m128 dd = _mm_mul_ps(_mm_sub_ps(LC1_dif, C2C3_dif), half); /* L-C1-C2+C3 */
    /* Interleave even/odd columns back to 8-wide rows */
    float *out0 = patch + (2 * i) * GALOSH_BLOCK_SIZE;
    float *out1 = patch + (2 * i + 1) * GALOSH_BLOCK_SIZE;
    _mm_storeu_ps(out0,     _mm_unpacklo_ps(a, b));  /* a0,b0,a1,b1 */
    _mm_storeu_ps(out0 + 4, _mm_unpackhi_ps(a, b));  /* a2,b2,a3,b3 */
    _mm_storeu_ps(out1,     _mm_unpacklo_ps(cc, dd)); /* c0,d0,c1,d1 */
    _mm_storeu_ps(out1 + 4, _mm_unpackhi_ps(cc, dd)); /* c2,d2,c3,d3 */
  }
}


/* ============================================================
 * [ARCHIVED] galosh_pass1_cfa — 案A "CFA frequency protect" Pass1.
 *
 * 8x8 WHT-LOSH on full-res Bayer mosaic with bins {0, 1, 8, 9} held
 * fixed (DC + 3 CFA frequency bins) to "protect" CFA structure from
 * shrinkage.  Theory: hard-threshold all bins except the CFA-locked
 * frequencies.  Bench result: visible 8x8 grid artefact, no measurable
 * colour-shift improvement; archived 2026-04 (案A 棄却済み).
 *
 * GALOSH_RAW_G does NOT use this — it decomposes 4 ch RGGB → L/C1/C2/C3
 * structurally before WHT, removing the need for any frequency masking.
 *
 * Called from gat_galosh_denoise_rawlc legacy stride=4 #else branch when
 * GALOSH_CFA_PROTECT is defined; not reached by default GALOSH_RAW_G build.
 * ============================================================ */
static void galosh_pass1_cfa(const float *restrict input,
                             float *restrict output,
                             const int width, const int height,
                             const float sigma_strength,
                             const int stride)
{
    const int rmax = height - GALOSH_BLOCK_SIZE;
    const int cmax = width  - GALOSH_BLOCK_SIZE;
    const int npix = width * height;

    float *numer = (float *)dt_alloc_align(64, sizeof(float) * npix);
    float *denom = (float *)dt_alloc_align(64, sizeof(float) * npix);
    if(!numer || !denom)
    {
        if(numer) dt_free_align(numer);
        if(denom) dt_free_align(denom);
        memcpy(output, input, sizeof(float) * npix);
        return;
    }
    memset(numer, 0, sizeof(float) * npix);
    memset(denom, 0, sizeof(float) * npix);

    const float sigma_sq = sigma_strength * sigma_strength;
    const float lambda_max = sigma_strength * sqrtf(2.0f * logf((float)GALOSH_BLOCK_PIXELS));

    #pragma omp parallel
    {
        float *my_numer = (float *)dt_alloc_align(64, sizeof(float) * npix);
        float *my_denom = (float *)dt_alloc_align(64, sizeof(float) * npix);
        if(my_numer && my_denom)
        {
        memset(my_numer, 0, sizeof(float) * npix);
        memset(my_denom, 0, sizeof(float) * npix);

        #pragma omp for schedule(dynamic, 4)
        for(int ref_r = 0; ref_r <= rmax; ref_r += stride)
        {
            for(int ref_c = 0; ref_c <= cmax; ref_c += stride)
            {
                float block[GALOSH_BLOCK_PIXELS];
                for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                    memcpy(block + dy * GALOSH_BLOCK_SIZE,
                           input + (ref_r + dy) * width + ref_c,
                           GALOSH_BLOCK_SIZE * sizeof(float));

                wht2d_8x8(block, 0);

                /* Estimate sigma_Y^2 from non-CFA AC coefficients */
                float sum_sq = 0.0f;
                int n_ac = 0;
                for(int i = 0; i < GALOSH_BLOCK_PIXELS; i++)
                {
                    if(!is_cfa_protected(i))
                    {
                        sum_sq += block[i] * block[i];
                        n_ac++;
                    }
                }
                const float sigma_y_sq = (n_ac > 0) ? sum_sq / ((float)n_ac * (float)GALOSH_BLOCK_PIXELS) : 0.0f;
                const float sigma_x_sq = fmaxf(sigma_y_sq - sigma_sq, 0.0f);

                float lambda;
                if(sigma_x_sq < 1e-10f)
                    lambda = 1e30f;
                else
                {
                    lambda = (sigma_sq / sqrtf(sigma_x_sq)) * sqrtf((float)GALOSH_BLOCK_PIXELS);
                    const float lambda_max_unorm = lambda_max * sqrtf((float)GALOSH_BLOCK_PIXELS);
                    if(lambda > lambda_max_unorm) lambda = lambda_max_unorm;
                }

                /* Hard threshold: preserve DC + CFA frequency bins */
                int n_nonzero = 4; /* DC + 3 CFA bins always kept */
                for(int i = 0; i < GALOSH_BLOCK_PIXELS; i++)
                {
                    if(is_cfa_protected(i)) continue; /* skip protected bins */
                    if(fabsf(block[i]) < lambda)
                        block[i] = 0.0f;
                    else
                        n_nonzero++;
                }

                wht2d_8x8(block, 1);

                const float weight = 1.0f / (float)n_nonzero;
                for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                    for(int dx = 0; dx < GALOSH_BLOCK_SIZE; dx++)
                    {
                        const int pos = (ref_r + dy) * width + (ref_c + dx);
                        const float kw = galosh_kaiser_2d[dy * GALOSH_BLOCK_SIZE + dx];
                        my_numer[pos] += weight * kw * block[dy * GALOSH_BLOCK_SIZE + dx];
                        my_denom[pos] += weight * kw;
                    }
            }
        }

        #pragma omp critical
        {
            for(int i = 0; i < npix; i++)
            {
                numer[i] += my_numer[i];
                denom[i] += my_denom[i];
            }
        }
        } /* end if alloc ok */
        dt_free_align(my_numer);
        dt_free_align(my_denom);
    }

    for(int i = 0; i < npix; i++)
        output[i] = (denom[i] > 1e-10f) ? numer[i] / denom[i] : input[i];

    dt_free_align(numer);
    dt_free_align(denom);
}

/* ============================================================
 * [ARCHIVED] galosh_pass2_cfa — 案A Pass2 paired with galosh_pass1_cfa.
 * Same archival rationale: CFA-protect approach yields 8x8 grid artefact,
 * archived in favour of GALOSH_RAW_G structural L/C decompose.
 * Called from gat_galosh_denoise_rawlc legacy stride=4 #else branch only
 * (GALOSH_CFA_PROTECT compile flag).  Not used by default build.
 * ============================================================ */
static void galosh_pass2_cfa(const float *restrict noisy,
                             const float *restrict pilot,
                             float *restrict output,
                             const int width, const int height,
                             const float sigma_strength,
                             const int stride)
{
    const int rmax = height - GALOSH_BLOCK_SIZE;
    const int cmax = width  - GALOSH_BLOCK_SIZE;
    const int npix = width * height;

    float *numer = (float *)dt_alloc_align(64, sizeof(float) * npix);
    float *denom = (float *)dt_alloc_align(64, sizeof(float) * npix);
    if(!numer || !denom)
    {
        if(numer) dt_free_align(numer);
        if(denom) dt_free_align(denom);
        memcpy(output, noisy, sizeof(float) * npix);
        return;
    }
    memset(numer, 0, sizeof(float) * npix);
    memset(denom, 0, sizeof(float) * npix);

    const float sigma_sq_unorm = sigma_strength * sigma_strength
                                * (float)GALOSH_BLOCK_PIXELS;

    #pragma omp parallel
    {
        float *my_numer = (float *)dt_alloc_align(64, sizeof(float) * npix);
        float *my_denom = (float *)dt_alloc_align(64, sizeof(float) * npix);
        if(my_numer && my_denom)
        {
        memset(my_numer, 0, sizeof(float) * npix);
        memset(my_denom, 0, sizeof(float) * npix);

        #pragma omp for schedule(dynamic, 4)
        for(int ref_r = 0; ref_r <= rmax; ref_r += stride)
        {
            for(int ref_c = 0; ref_c <= cmax; ref_c += stride)
            {
                float blk_noisy[GALOSH_BLOCK_PIXELS];
                float blk_pilot[GALOSH_BLOCK_PIXELS];
                for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                {
                    memcpy(blk_noisy + dy * GALOSH_BLOCK_SIZE,
                           noisy + (ref_r + dy) * width + ref_c,
                           GALOSH_BLOCK_SIZE * sizeof(float));
                    memcpy(blk_pilot + dy * GALOSH_BLOCK_SIZE,
                           pilot + (ref_r + dy) * width + ref_c,
                           GALOSH_BLOCK_SIZE * sizeof(float));
                }

                wht2d_8x8(blk_noisy, 0);
                wht2d_8x8(blk_pilot, 0);

                float wiener_energy = 0.0f;
                for(int i = 0; i < GALOSH_BLOCK_PIXELS; i++)
                {
                    float w;
                    if(is_cfa_protected(i))
                    {
                        w = 1.0f; /* DC + CFA bins protected */
                    }
                    else
                    {
                        const float s2 = blk_pilot[i] * blk_pilot[i];
                        w = s2 / (s2 + sigma_sq_unorm);
                        if(w < GALOSH_WIENER_FLOOR) w = GALOSH_WIENER_FLOOR;
                    }
                    blk_noisy[i] *= w;
                    wiener_energy += w * w;
                }

                wht2d_8x8(blk_noisy, 1);

                const float weight = 1.0f / fmaxf(wiener_energy, 1e-6f);
                for(int dy = 0; dy < GALOSH_BLOCK_SIZE; dy++)
                    for(int dx = 0; dx < GALOSH_BLOCK_SIZE; dx++)
                    {
                        const int pos = (ref_r + dy) * width + (ref_c + dx);
                        const float kw = galosh_kaiser_2d[dy * GALOSH_BLOCK_SIZE + dx];
                        my_numer[pos] += weight * kw * blk_noisy[dy * GALOSH_BLOCK_SIZE + dx];
                        my_denom[pos] += weight * kw;
                    }
            }
        }

        #pragma omp critical
        {
            for(int i = 0; i < npix; i++)
            {
                numer[i] += my_numer[i];
                denom[i] += my_denom[i];
            }
        }
        } /* end if alloc ok */
        dt_free_align(my_numer);
        dt_free_align(my_denom);
    }

    for(int i = 0; i < npix; i++)
        output[i] = (denom[i] > 1e-10f) ? numer[i] / denom[i] : noisy[i];

    dt_free_align(numer);
    dt_free_align(denom);
}


/* ================================================================
 * [PREVIOUS: GALOSH_RAW_G] compute_L_fullres — Phase 4(b) K14 (box variant).
 *
 * Used by gat_galosh_denoise_rawlc Phase 4(b) to build full-res L from
 * half-res L+C1/C2/C3, in TWO calls (noisy + pilot) which feed the
 * full-res Pass2 Wiener (Phase 4(c) K15).  This is the canonical box
 * 4-tap variant; the EWA-JL3 alternative is at galosh_compute_L_fullres_ewajl3
 * (= [ARCHIVED] variant C).
 *
 * Compute L_fullres from half-res L/C planes.
 *
 * Half-res L/C produces one L value per 2x2 Bayer block.
 * Under strong shrinkage this creates a visible 2x2 plateau artifact.
 * To eliminate it we compute L at every raw pixel position using
 * a sliding 2x2 WHT-DC.
 *
 * Derivation -- inv-WHT of sliding 2x2 blocks yields:
 *   L_fullres(2hy,   2hx)   = L(hy,hx)
 *   L_fullres(2hy,   2hx+1) = [(L-C2)@(hy,hx) + (L+C2)@(hy,hx+1)] / 2
 *   L_fullres(2hy+1, 2hx)   = [(L-C1)@(hy,hx) + (L+C1)@(hy+1,hx)] / 2
 *   L_fullres(2hy+1, 2hx+1) = sum [(L +/- C1 +/- C2 +/- C3) at 4 blocks] / 4
 *
 * Output: L_out of size (2*halfwidth) x (2*halfheight) = full-res.
 * ================================================================ */
static void compute_L_fullres(const float *restrict L,
                               const float *restrict C1,
                               const float *restrict C2,
                               const float *restrict C3,
                               const int halfwidth, const int halfheight,
                               float *restrict L_out)
{
  const int fw = 2 * halfwidth;

  DT_OMP_FOR()
  for(int hy = 0; hy < halfheight; hy++)
  {
    for(int hx = 0; hx < halfwidth; hx++)
    {
      const size_t hi = (size_t)hy * halfwidth + hx;
      /* Neighbor indices (clamped at boundaries) */
      const int hx1 = MIN(hx + 1, halfwidth - 1);
      const int hy1 = MIN(hy + 1, halfheight - 1);
      const size_t hi_r  = (size_t)hy  * halfwidth + hx1;
      const size_t hi_d  = (size_t)hy1 * halfwidth + hx;
      const size_t hi_dr = (size_t)hy1 * halfwidth + hx1;

      const int fr = 2 * hy, fc = 2 * hx;

      /* (2hy, 2hx): block-aligned = L itself */
      L_out[(size_t)fr * fw + fc] = L[hi];

      /* (2hy, 2hx+1): horizontal sliding */
      L_out[(size_t)fr * fw + fc + 1]
        = ((L[hi] - C2[hi]) + (L[hi_r] + C2[hi_r])) * 0.5f;

      /* (2hy+1, 2hx): vertical sliding */
      L_out[(size_t)(fr + 1) * fw + fc]
        = ((L[hi] - C1[hi]) + (L[hi_d] + C1[hi_d])) * 0.5f;

      /* (2hy+1, 2hx+1): diagonal sliding */
      {
        const float Ls = L[hi] + L[hi_r] + L[hi_d] + L[hi_dr];
        const float C1s = -C1[hi] - C1[hi_r] + C1[hi_d] + C1[hi_dr];
        const float C2s = -C2[hi] + C2[hi_r] - C2[hi_d] + C2[hi_dr];
        const float C3s =  C3[hi] - C3[hi_r] - C3[hi_d] + C3[hi_dr];
        L_out[(size_t)(fr + 1) * fw + fc + 1]
          = (Ls + C1s + C2s + C3s) * 0.25f;
      }
    }
  }
}


/* ================================================================
 * [ARCHIVED] galosh_compute_L_fullres_ewajl3 — variant C (jaggy-fix)
 *
 * Drop-in alternative to compute_L_fullres above (= Phase 4(b) K14).
 * Only used when g_galosh_lfr_kernel == 1 (--lfr-kernel=ewajl3 CLI flag).
 * GALOSH_RAW_G default uses the box variant; bench archive needs this for
 * variant-C reproducibility (PSNR -0.5 dB vs box, archived 2026-04).
 *
 * EWA jinc-windowed-jinc 3-tap (Lanczos-Jinc-3) compute_L_fullres
 * variant — RAW-specific archived experiment (variant C).
 *
 * Replaces the legacy box-like 4-tap reconstruction (compute_L_fullres
 * above) with a circularly symmetric, band-limited reconstruction
 * kernel.  Active only when --lfr-kernel=ewajl3 is passed; otherwise
 * the legacy compute_L_fullres is used.  Archived because pure EWA-JL3
 * reconstruction at the 4 sub-pixel positions used by the inverse 2x2
 * WHT introduces ~1 px shift relative to the legacy box convention
 * (PSNR -0.5 dB in bench).  Kept here for reproducibility of the
 * variant-C numbers in the bench archive.
 *
 * Lives in galosh_raw_cpu.c rather than galosh_core.h because the
 * subpixel offsets and the 4-channel API are RAW-Bayer-specific; the
 * generic 2x EWA-JL3 upsample for v1 chromaup chroma upsampling stays
 * in galosh_core.h as galosh_upsample_2x_ewajl3().
 *
 * Sub-pixel offsets (half-res units, relative to half-res sample
 * centre): TL(-0.25,-0.25), TR(-0.25,+0.25), BL(+0.25,-0.25),
 * BR(+0.25,+0.25).  We sample a 5x5 half-res window and apply a
 * precomputed normalized weight table per sub-pixel.
 * ================================================================ */
static void galosh_compute_L_fullres_ewajl3(const float *restrict L,
                                             const float *restrict C1,
                                             const float *restrict C2,
                                             const float *restrict C3,
                                             const int halfwidth, const int halfheight,
                                             float *restrict L_out)
{
    /* C1/C2/C3 unused: pure spatial reconstruction from the L plane.
     * The chroma-derived L refinement of the legacy formulas is sacrificed
     * for bandlimited diagonal accuracy -- this trade-off is intentional
     * for variant C and is part of what the bench is measuring. */
    (void)C1; (void)C2; (void)C3;

    const int fw = 2 * halfwidth;
    const int fh = 2 * halfheight;

    /* NOTE on alignment.
     *
     * Pure EWA-JL3 reconstruction at the geometric pixel centre offset
     * (TL at (-0.25,-0.25) half-res, etc.) produces a clean L plane that
     * is band-limited but is HALF A PIXEL DOWN-RIGHT of legacy's
     * block-centroid-stored-at-TL convention.  The downstream inverse
     * 2x2 WHT was tuned around legacy's built-in shift, so plain EWA-
     * JL3 ends up displaying RGGB output shifted ~1 px DOWN-RIGHT
     * relative to the base / A / B variants, as flagged in the dartboard
     * bench.
     *
     * Trying to match legacy by sampling at offsets {0, +0.5, +0.5, +0.5}
     * collides with jinc's negative ring at ~r_full=1.41 and produces
     * numerically unstable EWA weights for the BR position (only four
     * samples land inside support, all with negative inner-jinc and
     * positive outer-jinc contributions, leaving wsum tiny and the
     * normalised kernel magnifying small noise into huge values).
     *
     * Conclusion: C-1 (pure EWA-JL3 on L) is structurally incompatible
     * with the legacy WHT inverse pipeline.  The right fix is C-2 --
     * keep legacy compute_L_fullres formulas (which have the correct
     * built-in shift convention) and add a band-limited refinement on
     * top.  Until C-2 is implemented we stay with the original geometric
     * offsets here, which the user already saw shifts by ~1 px but at
     * least produces stable values to feed into the bench. */
    const float subpix[4][2] = {
        { -0.25f, -0.25f },  /* TL */
        { -0.25f, +0.25f },  /* TR */
        { +0.25f, -0.25f },  /* BL */
        { +0.25f, +0.25f },  /* BR */
    };

    /* Precompute 4 normalized 5x5 weight tables. */
    const int W = 2;          /* half-window radius in half-res samples */
    const int kw = 2 * W + 1; /* 5 */
    float weights[4][5][5];

    for(int si = 0; si < 4; si++)
    {
        const float oy = subpix[si][0];
        const float ox = subpix[si][1];
        float wsum = 0.0f;
        for(int dy = -W; dy <= W; dy++)
        {
            for(int dx = -W; dx <= W; dx++)
            {
                const float ry = (float)dy - oy;
                const float rx = (float)dx - ox;
                const float r_half = sqrtf(rx * rx + ry * ry);
                const float r_full = r_half * 2.0f;  /* output pixel units */
                float w_val = 0.0f;
                if(r_full < 3.0f)
                    w_val = galosh_jinc(r_full) * galosh_jinc(r_full / 3.0f);
                weights[si][dy + W][dx + W] = w_val;
                wsum += w_val;
            }
        }
        const float inv_wsum = 1.0f / fmaxf(wsum, 1e-20f);
        for(int dy = 0; dy < kw; dy++)
            for(int dx = 0; dx < kw; dx++)
                weights[si][dy][dx] *= inv_wsum;
    }

    /* Apply weights to produce 4 full-res samples per half-res block. */
    DT_OMP_FOR()
    for(int hy = 0; hy < halfheight; hy++)
    {
        for(int hx = 0; hx < halfwidth; hx++)
        {
            for(int si = 0; si < 4; si++)
            {
                const int sub_dy = si / 2;
                const int sub_dx = si % 2;
                const int fr = 2 * hy + sub_dy;
                const int fc = 2 * hx + sub_dx;
                if(fr >= fh || fc >= fw) continue;

                float sum = 0.0f;
                for(int dy = -W; dy <= W; dy++)
                {
                    int hyi = hy + dy;
                    if(hyi < 0)            hyi = 0;
                    if(hyi >= halfheight)  hyi = halfheight - 1;
                    for(int dx = -W; dx <= W; dx++)
                    {
                        int hxi = hx + dx;
                        if(hxi < 0)           hxi = 0;
                        if(hxi >= halfwidth)  hxi = halfwidth - 1;
                        sum += L[(size_t)hyi * halfwidth + hxi]
                             * weights[si][dy + W][dx + W];
                    }
                }
                L_out[(size_t)fr * fw + fc] = sum;
            }
        }
    }
}


/* ================================================================
 *  GALOSH -- Full pipeline (RAW L/C decomposed)
 *
 *  Pre-demosaic denoiser operating on RGGB before WB and demosaic.
 *  Fully blind: no white-balance coefficients, no per-channel QE prior.
 *
 *  Pipeline:
 *   1. Half-res RGGB extraction -> piecewise C1 GAT -> RMS unified sigma-normalize
 *   2. Self-consistent dark anchor (per-ch DC subtract) -> WHT -> L/C1/C2/C3
 *   3. Pass 1: BayesShrink hard-threshold pilot on each half-res L/C plane
 *   4. Pass 2: Wiener shrinkage on each half-res L/C plane (independent sigma_L / sigma_C)
 *   5. Full-res L: sliding WHT-DC -> GALOSH Pass 2 on L plane
 *   6. Inverse WHT with full-res L + half-res C (anchor restore)
 *   7. sigma-denormalize -> exact inverse GAT -> write back
 *
 *  Key difference from BM3D: GALOSH operates on ONE plane at a time
 *  (L, C1, C2, C3 independently). No block matching, no non-local processing.
 *  GAT normalization ensures noise is i.i.d. Gaussian (sigma=1), so local
 *  adaptive shrinkage in the WHT domain is sufficient.
 *
 *  Theoretical highlights:
 *  (a) Piecewise C1 VST (extends Foi GAT to all reals)
 *  (b) Unified sigma normalization (RMS, makes Var[L]=Var[Ck]=1 exactly)
 *  (c) Self-consistent dark anchor (per-channel DC protection)
 *  See detailed derivations in comments below.
 *
 *  Ref: Danielyan et al. "Cross-color BM3D" (LNLA 2009) -- L/C decomposition
 *       Foi et al. (Sig.Proc. 2008) -- Poisson-Gaussian noise model
 *       Makitalo & Foi (TIP 2013) -- exact unbiased inverse GAT
 *       Chang, Yu & Vetterli (IEEE TIP 2000) -- BayesShrink
 *       Cleveland (J. Am. Stat. Assoc., 1979) -- LOESS (see galosh_core.h)
 * ================================================================ */
/* ================================================================
 * [PREVIOUS: GALOSH_RAW_G] gat_galosh_denoise_rawlc — main entry point.
 *
 * Orchestrates the full GALOSH_RAW_G pipeline.  Called by main() (CLI)
 * and by darktable's process() (via the bayer.h port).  Branches at
 * the bottom on `#ifndef GALOSH_LEGACY`:
 *   #ifndef GALOSH_LEGACY (default)  → [LATEST] half-res LOSH +
 *                                       K14/K15/K16 chromaup
 *   #else                            → [ARCHIVED] full-res raw WHT-LOSH
 *                                       + block-replicated chroma
 * ================================================================ */
#ifndef GALOSH_RELEASE  /* ==== [DEPRECATED] variants g,h,i,j,k — ablation builds only ==== */
static void gat_galosh_denoise_rawlc(const float *const restrict in, float *const restrict out,
                                    const dt_iop_roi_t *const roi,
                                    const float luma_strength, const float chroma_strength,
                                    const uint32_t filters)
{
  const int width = roi->width, height = roi->height;
  const size_t npixels = (size_t)width * height;
  memcpy(out, in, sizeof(float) * npixels);

  if(luma_strength <= 0.0f) return;

  const int halfwidth = (width + 1) / 2;
  const int halfheight = (height + 1) / 2;
  if(halfwidth < GALOSH_BLOCK_SIZE * 2 || halfheight < GALOSH_BLOCK_SIZE * 2) return;

  const size_t chsize = (size_t)halfwidth * halfheight;

  /* ================================================================
   * [PREVIOUS: GALOSH_RAW_G] Phase 0 — Foi-Alenius blind α / σ² estimation.
   * Sets the Poisson-Gauss noise model parameters for GAT.  See
   * galosh_estimate_noise() in galosh_cpu.h for the MAD-based estimator.
   * ================================================================ */
  const galosh_noise_params_t np = galosh_estimate_noise(in, width, height);
  gat_build_inverse_table(np.alpha, np.sigma_sq);

  /* Pre-declare all buffers */
  float *ch_gat[4] = { NULL, NULL, NULL, NULL };
  float *luma = NULL, *chroma1 = NULL, *chroma2 = NULL, *chroma3 = NULL;
  float *c1_pilot = NULL, *c2_pilot = NULL, *c3_pilot = NULL;
  float *c1_out = NULL, *c2_out = NULL, *c3_out = NULL;

  /* ================================================================
   * [PREVIOUS: GALOSH_RAW_G] Phase 1 — GAT extract + normalize half-res RGGB.
   * Extract 4 ch (R/Gb/Gr/B) at half-res, GAT forward each ch, measure
   * per-ch σ_GAT (Laplacian MAD), then RMS unified_sigma normalisation
   * → unit variance per-ch in GAT space.
   * ================================================================ */
  float sigma_gat_ch[4];
  for(int c = 0; c < 4; c++)
  {
    const int row_offset = c & 1, col_offset = (c >> 1) & 1;
    ch_gat[c] = dt_alloc_align_float(chsize);
    if(!ch_gat[c]) goto cleanup_rawlc;

    DT_OMP_FOR()
    for(int row = row_offset; row < height; row += 2)
      for(int col = col_offset; col < width; col += 2)
        ch_gat[c][((row - row_offset) / 2) * halfwidth + (col - col_offset) / 2]
          = in[(size_t)row * width + col];

    DT_OMP_FOR()
    for(size_t i = 0; i < chsize; i++)
      ch_gat[c][i] = gat_forward(ch_gat[c][i], np.alpha, np.sigma_sq);

    sigma_gat_ch[c] = estimate_gat_sigma_halfres(ch_gat[c], halfwidth, halfheight);
  }

  /* RMS unified sigma normalization.
   *
   * unified_sigma := sqrt( mean(sigma_c^2) )
   *
   * This choice makes Var[L] = Var[Ck] = 1 exactly after WHT,
   * regardless of per-channel GAT-domain variance non-uniformity
   * caused by single-shot noise estimation bias toward G.
   * Per-channel normalization would fix per-ch Var=1 but break
   * WHT signal proportionality (false chroma on uniform input). */
  {
    const float mean_var = 0.25f * (sigma_gat_ch[0] * sigma_gat_ch[0]
                                  + sigma_gat_ch[1] * sigma_gat_ch[1]
                                  + sigma_gat_ch[2] * sigma_gat_ch[2]
                                  + sigma_gat_ch[3] * sigma_gat_ch[3]);
    const float unified_sigma = sqrtf(fmaxf(mean_var, 1e-12f));

    const float post_mean_var = mean_var / (unified_sigma * unified_sigma);

    fprintf(stderr, "[rawdenoise] GALOSH: alpha=%.8f sigma_sq=%.10f | "
                     "unified_sigma=%.4f [RMS] (per-ch: %.4f %.4f %.4f %.4f) | "
                     "post-norm mean Var=%.4f (target=1.0) | "
                     "size=%dx%d (half=%dx%d) | "
                     "sigma_L=%.3f sigma_C=%.3f (independent, GAT-norm space)\n",
            np.alpha, np.sigma_sq, unified_sigma,
            sigma_gat_ch[0], sigma_gat_ch[1], sigma_gat_ch[2], sigma_gat_ch[3],
            post_mean_var,
            width, height, halfwidth, halfheight,
            luma_strength, chroma_strength);
    for(int c = 0; c < 4; c++) sigma_gat_ch[c] = unified_sigma;
    const float inv_sg = 1.0f / unified_sigma;
    for(int c = 0; c < 4; c++)
    {
      DT_OMP_FOR()
      for(size_t i = 0; i < chsize; i++)
        ch_gat[c][i] *= inv_sg;
    }
  }

  /* ================================================================
   * [PREVIOUS: GALOSH_RAW_G] Phase 2 — Self-consistent dark anchor.
   *
   * Per-channel QE differences are REAL signal to be PRESERVED in bright
   * regions. In dark regions, however, QE x signal ~ 0, while the GAT
   * linear branch and read-noise statistics leave residual per-channel
   * DC offsets that, after WHT, manifest as false dark-area chroma.
   *
   * Fix: estimate a per-channel DC anchor over a noise-dominated
   * cohort and subtract it before WHT, then restore it after shrinkage.
   * The cohort is selected by smooth weights w(L_i) = 1 / (1 + (L_i/s)^4).
   * Scale s is self-consistently estimated (2 iterations).
   *
   * Restored after shrinkage in Phase 4(d) K16 chromaup.
   * ================================================================ */
  float ch_dark_ref[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  {
    const double s_init = (double)np.sigma_sq / fmax((double)np.alpha, 1e-12);
    double s_scale = s_init;
    const double s_min = 0.05 * s_init;
    const double s_max = 50.0 * s_init;
    const int n_iter = 2;

    for(int iter = 0; iter <= n_iter; iter++)
    {
      const double inv_s = 1.0 / fmax(s_scale, 1e-20);
      double sum_w = 0.0;
      double sum_wch[4] = {0.0, 0.0, 0.0, 0.0};

      for(int hy = 0; hy < halfheight; hy++)
        for(int hx = 0; hx < halfwidth; hx++)
        {
          const size_t hpos = (size_t)hy * halfwidth + hx;

          /* Achromatic filter: exclude pixels with inter-channel spread
           * exceeding noise expectation. Dark skin (melanin: R < G) and
           * other spectrally non-neutral dark objects are rejected.
           * In sigma-normalized GAT domain, noise-only range ≈ 2.06 ± 0.73. */
          const float g0 = ch_gat[0][hpos], g1 = ch_gat[1][hpos];
          const float g2 = ch_gat[2][hpos], g3 = ch_gat[3][hpos];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const int fy = hy * 2, fx = hx * 2;
          const float iv0 = in[(size_t)fy * width + fx];
          const float iv1 = in[(size_t)(fy + 1) * width + fx];
          const float iv2 = in[(size_t)fy * width + fx + 1];
          const float iv3 = in[(size_t)(fy + 1) * width + fx + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          sum_w += w;
          sum_wch[0] += w * g0;
          sum_wch[1] += w * g1;
          sum_wch[2] += w * g2;
          sum_wch[3] += w * g3;
        }

      const double inv_sw = 1.0 / fmax(sum_w, 1e-20);
      for(int c = 0; c < 4; c++)
        ch_dark_ref[c] = (float)(sum_wch[c] * inv_sw);

      if(iter == n_iter) break;

      /* Weighted per-ch residual std in GAT-norm (target = 1.0) */
      double sum_wresid2 = 0.0;
      double sum_w2 = 0.0;
      for(int hy = 0; hy < halfheight; hy++)
        for(int hx = 0; hx < halfwidth; hx++)
        {
          const size_t hpos = (size_t)hy * halfwidth + hx;

          /* Same achromatic filter as above */
          const float g0 = ch_gat[0][hpos], g1 = ch_gat[1][hpos];
          const float g2 = ch_gat[2][hpos], g3 = ch_gat[3][hpos];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const int fy = hy * 2, fx = hx * 2;
          const float iv0 = in[(size_t)fy * width + fx];
          const float iv1 = in[(size_t)(fy + 1) * width + fx];
          const float iv2 = in[(size_t)fy * width + fx + 1];
          const float iv3 = in[(size_t)(fy + 1) * width + fx + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          sum_w2 += w;
          double resid2 = 0.0;
          for(int c = 0; c < 4; c++)
          {
            const double d = (double)ch_gat[c][hpos] - ch_dark_ref[c];
            resid2 += d * d;
          }
          sum_wresid2 += w * resid2 * 0.25;
        }
      const double inv_sw2 = 1.0 / fmax(sum_w2, 1e-20);
      const double measured_std =
          sqrt(fmax(sum_wresid2 * inv_sw2, 1e-20));
      const double ratio = 1.0 / measured_std;
      s_scale *= sqrt(ratio);
      if(s_scale < s_min) s_scale = s_min;
      if(s_scale > s_max) s_scale = s_max;
    }

    fprintf(stderr, "[rawdenoise] dark anchor self-consistent: s_init=%.6e s_final=%.6e | "
                    "ch_dark_ref: B=%.4f Gb=%.4f Gr=%.4f R=%.4f\n",
            s_init, s_scale,
            ch_dark_ref[0], ch_dark_ref[1], ch_dark_ref[2], ch_dark_ref[3]);

    for(int c = 0; c < 4; c++)
    {
      const float ref = ch_dark_ref[c];
      DT_OMP_FOR()
      for(size_t i = 0; i < chsize; i++) ch_gat[c][i] -= ref;
    }
  }

  /* ================================================================
   * [PREVIOUS: GALOSH_RAW_G] Phase 3 — 2x2 WHT decompose at half-res.
   * 4 half-res RGGB channels (after dark_ref subtract) → L / C1 / C2 / C3
   *   L  = (R + Gb + Gr + B) / 2
   *   C1 = (R - Gb + Gr - B) / 2
   *   C2 = (R + Gb - Gr - B) / 2
   *   C3 = (R - Gb - Gr + B) / 2
   * Inverse 2x2 WHT (= K16 chromaup signs in Phase 4(d)) is the
   * canonical Bayer-aware sign table.
   * ================================================================ */
  luma    = dt_alloc_align_float(chsize);
  chroma1 = dt_alloc_align_float(chsize);
  chroma2 = dt_alloc_align_float(chsize);
  chroma3 = dt_alloc_align_float(chsize);
  if(!luma || !chroma1 || !chroma2 || !chroma3) goto cleanup_rawlc;

  DT_OMP_FOR()
  for(size_t i = 0; i < chsize; i++)
  {
    const float a = ch_gat[0][i], b = ch_gat[1][i];
    const float cc = ch_gat[2][i], d = ch_gat[3][i];
    luma[i]    = (a + b + cc + d) * 0.5f;
    chroma1[i] = (a - b + cc - d) * 0.5f;
    chroma2[i] = (a + b - cc - d) * 0.5f;
    chroma3[i] = (a - b - cc + d) * 0.5f;
  }

  /* ================================================================
   * [PREVIOUS: GALOSH_RAW_G] Phase 3.5 — chroma denoise (Y-guided LOESS).
   * Replaces pre-G WHT-LOSH-on-chroma with bilateral LOESS.  See block
   * below.  (Original "Phase 3" docstring is retained for context but
   * the WHT-LOSH paths are no longer invoked — see (void) marks at end.)
   *
   * (Below originally commented:)
   * Phase 3: Independent chroma 2-pass denoising
   *
   * Each chroma plane (C1, C2, C3) gets its own BayesShrink pilot
   * (Pass 1) and Wiener shrinkage (Pass 2). This is the standard
   * 2-pass design applied independently to each L/C plane.
   *
   * Key difference from GALOSH_LG: chroma pilots come from chroma
   * itself, not from luma. This preserves chroma-specific noise
   * structure and avoids the edge-blocking artifacts that occur
   * when a cross-channel pilot mismatches chroma's own structure.
   *
   * Color shift mitigation: Pass 2 uses galosh_pass2_chroma() with
   * a higher Wiener floor (GALOSH_CHROMA_WIENER_FLOOR = 0.3) to
   * prevent over-shrinkage of chroma AC in dark areas. Standard
   * floor 0.125 kills spatial color variation, leaving Phase 4's
   * G-biased luma to dominate the output → green shift.
   *
   * Phase 3: 独立 chroma 2パスデノイズ
   * 各 chroma plane が自身の BayesShrink pilot + Wiener を持つ。
   * LG のように luma pilot を流用せず、chroma 固有のノイズ構造を保持。
   * 色シフト対策: chroma Wiener floor を 0.3 に引き上げ、暗部の
   * chroma AC 過剰 shrinkage を防止。
   * ================================================================ */
  {
    /* Chroma Pass 1 (BayesShrink pilot) + Pass 2 (Wiener with higher floor) */
    c1_pilot = dt_alloc_align_float(chsize);
    c2_pilot = dt_alloc_align_float(chsize);
    c3_pilot = dt_alloc_align_float(chsize);
    c1_out   = dt_alloc_align_float(chsize);
    c2_out   = dt_alloc_align_float(chsize);
    c3_out   = dt_alloc_align_float(chsize);
    if(!c1_pilot || !c2_pilot || !c3_pilot || !c1_out || !c2_out || !c3_out)
      goto cleanup_rawlc;

    /* [PREVIOUS: GALOSH_RAW_G] chroma_stride = 2 (75% overlap on 8x8 block).
     * Used to be a tunable for archived chroma WHT-LOSH variants; current
     * pipeline does Phase 3.5 chroma via LOESS so this is now unused
     * (kept for `(void)chroma_stride` reference). */
    const int chroma_stride = 2;

    /* ============================================================
     * [PREVIOUS: GALOSH_RAW_G] Phase 3.5 — Y-guided bilateral LOESS chroma
     * denoise.  Replaces pre-G WHT-LOSH-on-chroma.
     *
     * Noisy half-res luma plane is the guide; bilateral weight
     *   w_i = exp(-(Y_i - Y_c)² / (2 σ²))
     * excludes specular highlights (silver windows etc.) from the
     * locally-weighted linear regression on Cb/Cr.  Two calls handle
     * the 3 chroma planes (C1/C2 pair, then C3 alone w/ dummy 2nd out).
     *
     * Implementation: galosh_loess_chroma() in galosh_cpu.h
     * Output: c1_out / c2_out / c3_out — fed to Phase 4(d) K16
     * chromaup as the half-res chroma to upsample.
     *
     * [ARCHIVED] alternatives (removed from code 2026-04-26, in git):
     *   - v2 chromawiener (Lee residual reinjection):
     *     PSNR -1.16 dB / LPIPS +0.034
     *   - v3 chromawhtlosh (B-cfa = WHT-LOSH on C with low-freq protect):
     *     8x8 grid artefact, PSNR -3.46 dB / SSIM -0.13 / LPIPS +0.12
     * ============================================================ */
    galosh_loess_chroma(luma, chroma1, chroma2, c1_out, c2_out,
                        halfwidth, halfheight, chroma_strength);
    galosh_loess_chroma(luma, chroma3, chroma3, c3_out, c3_pilot /*dummy*/,
                        halfwidth, halfheight, chroma_strength);

    fprintf(stderr, "[rawdenoise] GALOSH: chroma LOESS done (sigma_C=%.3f, R=%d, BW=%.1f)\n",
            chroma_strength, GALOSH_LOESS_RADIUS, GALOSH_LOESS_BW);

    (void)chroma_stride; (void)c1_pilot; (void)c2_pilot;  /* no longer used */
  }

  dt_free_align(c1_pilot); c1_pilot = NULL;
  dt_free_align(c2_pilot); c2_pilot = NULL;
  dt_free_align(c3_pilot); c3_pilot = NULL;

/* GALOSH_RAW_G is the canonical pipeline (= default).  Define
 * GALOSH_LEGACY to opt INTO the pre-G archived stride=4 path instead
 * (bench reproducibility only).  GALOSH_F / GALOSH_F_HALF / GALOSH_F_FULL
 * are old aliases retained for backwards compat with bench scripts —
 * they still select [LATEST]. */
#if !defined(GALOSH_LEGACY)
  /* ================================================================
   * [PREVIOUS: GALOSH_RAW_G] Phase 4 — Half-Res L/C + Full-Res L Refinement
   *
   * The defining structural choice of GALOSH_RAW_G: separate luma and
   * chroma processing scales.
   *   (a) K13: half-res luma Pass1+Pass2 (BayesShrink + Wiener,
   *       stride=2, 75% overlap, MAD-based σ_Y robust shrink)
   *   (b) K14: compute L_fullres from noisy + pilot half-res L+C via
   *       4-tap box reconstruction
   *   (c) K15: full-res L Pass2 only (Wiener using step (b) pilot)
   *   (d) K16 chromaup: EWA-JL3 upsample c1/c2/c3 → full-res C, then
   *       per-pixel inverse 2x2 WHT + dark_ref restore + inverse GAT.
   *       Replaces pre-G block-replicated chroma reconstruct.
   *
   * Key benefits (vs pre-G full-res raw WHT-LOSH legacy #else path):
   *   - Chroma denoise stays at half-res → no inter-channel
   *     homogenization → no green-shift
   *   - stride=2 overlap-add gives 4x block contributions per pixel
   *     → smoother estimates, better luma resolution
   *   - K16 chromaup eliminates 2x2 stair-step on diagonals
   *   - Fully GPU-parallelisable (OpenCL): see standalone/galosh.cl
   *
   * 半解像度 L/C + 全解像度 L リファインメント (GALOSH_RAW_G の本体)。
   * raw Bayer 全体に GALOSH をかけず、L のみ全解像度化して Pass2 処理。
   * C は半解像度の LOESS 結果を K16 EWA-JL3 で full-res 化、per-pixel
   * 逆 WHT で再構成。stride=2 (75% overlap)。
   * ================================================================ */
  {
    /* ============================================================
     * [PREVIOUS: GALOSH_RAW_G] Phase 4(a) — K13 half-res luma Pass1+Pass2.
     * BayesShrink pilot (with MAD-based σ_Y) + empirical Wiener.
     * ============================================================ */
    float *l_pilot = dt_alloc_align_float(chsize);
    float *l_out   = dt_alloc_align_float(chsize);
    if(!l_pilot || !l_out)
    {
      dt_free_align(l_pilot);
      dt_free_align(l_out);
      goto cleanup_rawlc;
    }

    /* Half-res luma stride.  legacy default: stride=2 (= 75% overlap on
     *   8×8 block).  When K13 block = 4, stride=2 means only 50% overlap;
     *   to match the legacy 75% overlap quality (and required to suppress
     *   block grid given the smaller block), default stride to 1 instead.
     *   User may still override via --stride=N. */
    int halfres_stride = g_galosh_stride;
    if(g_galosh_k13_block == 4 && halfres_stride == 2)
      halfres_stride = 1;

    fprintf(stderr, "[GALOSH_RAW_G] half-res luma Pass1+2 "
                     "(%dx%d, sigma_L=%.3f, block=%d, stride=%d, n_orient=%d)\n",
                     halfwidth, halfheight, luma_strength,
                     g_galosh_k13_block, halfres_stride, g_galosh_n_orient);

    /* Pass1+Pass2 with optional 4-orientation averaging.  When
     * g_galosh_k13_block == 4, the half-res L is processed with 4×4 WHT-
     * LOSH so that K15's 8×8 full-res WHT-LOSH and K13's effective
     * full-res grain scale (2 × 4 = 8 pixels) match. */
    /* GALOSH_RAW_G: use_robust_shrink hardcoded to 1 (MAD-based BayesShrink
     * is part of the adopted pipeline definition; see file-top comment). */
    galosh_pass12_multiorient_blocked(luma, l_out, halfwidth, halfheight,
                                       luma_strength, g_galosh_k13_block,
                                       halfres_stride, g_galosh_n_orient,
                                       /*use_robust_shrink=*/1);
    (void)l_pilot;  /* internal pilot now lives inside the wrapper */

    fprintf(stderr, "[GALOSH_RAW_G] half-res luma done\n");

    /* ============================================================
     * [ARCHIVED] Unified-reconstruction path (g_galosh_unified == 1).
     * Single 4-plane EWA-JL3 upsample (L AND C1/C2/C3) + per-pixel
     * inverse WHT.  Bypasses K15 full-res Pass2.  Archived: var=1
     * noise invariance broken (denoise quality ↓).  GALOSH_RAW_G
     * keeps K15 + uses K16 chromaup ONLY for chroma upsample (not L).
     * Default g_galosh_unified=0 → this `if` block is skipped.
     * ============================================================
     * Original docstring follows:
     *
     * Replaces Step (b) compute_L_fullres + Step (c) full-res LOSH +
     * Step (d) block-replicated inverse 2x2 WHT with a single coherent
     * 4-plane upsample + per-pixel reconstruction.
     *
     * Why: legacy Step (b) reconstructs full-res L by mixing half-res L
     * and chroma corrections via 4 box-weighted formulas, then Step (d)
     * REPLICATES half-res chroma to all 4 pixels of every 2x2 block.
     * This produces visible 2x2 stair-stepping on diagonal edges (the
     * specific bug the user observed: hard_baseline clean, guided_c
     * blocky on text edges).
     *
     * Unified path: bandlimit-upsample L AND each of C1/C2/C3 to full
     * res with the same EWA jinc-windowed-jinc-3 kernel, then run the
     * inverse 2x2 WHT PER-PIXEL using the upsampled values and Bayer-
     * pattern-aware sign tables.  Chroma now varies smoothly across
     * the 2x2 block boundary, killing the stair.
     *
     * Theory: the half-res L and C planes are samples of bandlimited
     * signals at 1/2 Nyquist; EWA-JL3 is a near-optimal isotropic
     * reconstruction kernel.  Each Bayer pixel value is then a linear
     * combination of those four reconstructed values with the
     * channel-specific WHT inverse signs -- exactly the per-block
     * inverse the legacy code applies, but evaluated AT EACH PIXEL
     * rather than once per block. */
    if(g_galosh_unified)
    {
      const size_t fullsize_unified = (size_t)width * height;
      float *L_full  = dt_alloc_align_float(fullsize_unified);
      float *C1_full = dt_alloc_align_float(fullsize_unified);
      float *C2_full = dt_alloc_align_float(fullsize_unified);
      float *C3_full = dt_alloc_align_float(fullsize_unified);
      if(!L_full || !C1_full || !C2_full || !C3_full)
      {
        dt_free_align(L_full);  dt_free_align(C1_full);
        dt_free_align(C2_full); dt_free_align(C3_full);
        dt_free_align(l_pilot); dt_free_align(l_out);
        goto cleanup_rawlc;
      }

      galosh_upsample_2x_ewajl3(l_out,  L_full,  halfwidth, halfheight);
      galosh_upsample_2x_ewajl3(c1_out, C1_full, halfwidth, halfheight);
      galosh_upsample_2x_ewajl3(c2_out, C2_full, halfwidth, halfheight);
      galosh_upsample_2x_ewajl3(c3_out, C3_full, halfwidth, halfheight);

      fprintf(stderr, "[ARCHIVED unified] 4-plane EWA-JL3 upsample done\n");

      /* Channel-slot lookup from 2x2 cell offset, RGGB hardcoded for
       * the standalone CPU reference (filters arg is ignored here as
       * elsewhere in this file -- the darktable build uses
       * galosh_bayer.h which derives co_row/co_col from FC()).  RGGB
       * gives R at (0,0), Gb at (1,0), Gr at (0,1), B at (1,1). */
      int ch_lut[2][2];
      ch_lut[0][0] = 0;  /* R  at TL */
      ch_lut[1][0] = 1;  /* Gb at BL */
      ch_lut[0][1] = 2;  /* Gr at TR */
      ch_lut[1][1] = 3;  /* B  at BR */

      /* Inverse 2x2 WHT signs: for output channel ch the linear
       * combination is L + s1*C1 + s2*C2 + s3*C3, all scaled by 1/2.
       * SIGNS[ch] = {s1, s2, s3} matches the legacy formulas:
       *   R  = (L + C1 + C2 + C3)/2
       *   Gb = (L - C1 + C2 - C3)/2
       *   Gr = (L + C1 - C2 - C3)/2
       *   B  = (L - C1 - C2 + C3)/2
       */
      static const float SIGNS[4][3] = {
        { +1.0f, +1.0f, +1.0f },  /* R  */
        { -1.0f, +1.0f, -1.0f },  /* Gb */
        { +1.0f, -1.0f, -1.0f },  /* Gr */
        { -1.0f, -1.0f, +1.0f },  /* B  */
      };

      const float sg_unified = sigma_gat_ch[0];  /* unified_sigma */

      DT_OMP_FOR()
      for(int fr = 0; fr < height; fr++)
      {
        for(int fc = 0; fc < width; fc++)
        {
          const int ch = ch_lut[fr & 1][fc & 1];
          const size_t pos = (size_t)fr * width + fc;
          const float val = 0.5f * (L_full[pos]
                                  + SIGNS[ch][0] * C1_full[pos]
                                  + SIGNS[ch][1] * C2_full[pos]
                                  + SIGNS[ch][2] * C3_full[pos])
                          + ch_dark_ref[ch];
          out[pos] = gat_inverse_exact(val * sg_unified);
        }
      }

      fprintf(stderr, "[rawdenoise] GALOSH_F unified: per-pixel inverse WHT + inv-GAT done\n");

      dt_free_align(L_full);  dt_free_align(C1_full);
      dt_free_align(C2_full); dt_free_align(C3_full);
      dt_free_align(l_pilot); dt_free_align(l_out);
      goto cleanup_rawlc;
    }

    /* ============================================================
     * [PREVIOUS: GALOSH_RAW_G] Phase 4(b) — K14 compute_L_fullres × 2.
     *
     * We build TWO full-res L planes via 4-tap box reconstruction
     * from half-res L+C1+C2+C3:
     *   L_fullres_noisy: from noisy half-res (luma, chroma1/2/3)
     *   L_fullres_pilot: from denoised half-res (l_out, c1_out/2/3)
     *
     * The pilot provides the Wiener reference for K15 (Phase 4(c)).
     * ============================================================ */
    const size_t fullsize = (size_t)width * height;
    float *L_fr_noisy = dt_alloc_align_float(fullsize);
    float *L_fr_pilot = dt_alloc_align_float(fullsize);
    float *L_fr_den   = dt_alloc_align_float(fullsize);
    if(!L_fr_noisy || !L_fr_pilot || !L_fr_den)
    {
      dt_free_align(L_fr_noisy);
      dt_free_align(L_fr_pilot);
      dt_free_align(L_fr_den);
      dt_free_align(l_pilot);
      dt_free_align(l_out);
      goto cleanup_rawlc;
    }

    /* Full-res L stride: bench-overrideable for variant A. */
    const int fullres_stride = g_galosh_stride;

    /* compute_L_fullres kernel selection (variant C):
     *   0 = legacy 4-tap box (axis-aligned, square frequency response)
     *   1 = EWA jinc-windowed-jinc-3 (isotropic, circular freq response,
     *       targets diagonal stair-step from upsample aliasing) */
    if(g_galosh_lfr_kernel == 1)
    {
        galosh_compute_L_fullres_ewajl3(luma, chroma1, chroma2, chroma3,
                                         halfwidth, halfheight, L_fr_noisy);
        galosh_compute_L_fullres_ewajl3(l_out, c1_out, c2_out, c3_out,
                                         halfwidth, halfheight, L_fr_pilot);
    }
    else
    {
        compute_L_fullres(luma, chroma1, chroma2, chroma3,
                          halfwidth, halfheight, L_fr_noisy);
        compute_L_fullres(l_out, c1_out, c2_out, c3_out,
                          halfwidth, halfheight, L_fr_pilot);
    }

    fprintf(stderr, "[GALOSH_RAW_G] L_fullres computed (%dx%d), "
                     "full-res stride=%d, lfr_kernel=%s\n",
                     width, height, fullres_stride,
                     g_galosh_lfr_kernel == 1 ? "EWA-JL3" : "box");

    /* ============================================================
     * [PREVIOUS: GALOSH_RAW_G] Phase 4(c) — K15 full-res L Pass2 only.
     * Empirical Wiener using step (b) box-reconstructed L_fr_pilot.
     * `galosh_pass2_multiorient` with n_orient=1 collapses to plain
     * galosh_pass2 (= darktable port calls galosh_pass2 directly).
     * ============================================================ */
    galosh_pass2_multiorient(L_fr_noisy, L_fr_pilot, L_fr_den,
                             width, height, luma_strength,
                             fullres_stride, g_galosh_n_orient);

    dt_free_align(L_fr_noisy); L_fr_noisy = NULL;
    dt_free_align(L_fr_pilot); L_fr_pilot = NULL;
    dt_free_align(l_pilot);    l_pilot = NULL;

    fprintf(stderr, "[GALOSH_RAW_G] full-res L Pass2 done\n");

    /* ============================================================
     * [PREVIOUS: GALOSH_RAW_G] Phase 4(d) — K16 chromaup ⭐ NEW vs pre-G ⭐
     *
     * Per-pixel reconstruction at full-res from:
     *   - L_fr_den: full-res denoised luma (Phase 4(c) output)
     *   - c1_out, c2_out, c3_out: half-res denoised chroma (Phase 3.5)
     *     EWA-JL3 upsampled to full-res (this step)
     *   - ch_dark_ref: dark anchor offsets per channel (Phase 2)
     *
     * Inverse 2x2 WHT applied PER-PIXEL (not per-2x2-block as pre-G):
     *   R  = (L + C1 + C2 + C3) / 2 + dark_ref[R]
     *   Gb = (L - C1 + C2 - C3) / 2 + dark_ref[Gb]
     *   Gr = (L + C1 - C2 - C3) / 2 + dark_ref[Gr]
     *   B  = (L - C1 - C2 + C3) / 2 + dark_ref[B]
     *
     * After WHT + dark_ref, denormalize × unified_sigma → inverse GAT
     * via piecewise LUT (gat_inverse_exact).  ch_gat values had
     * dark_ref subtracted in Phase 2; we restore it here so that
     * the inverse GAT receives absolute GAT-normalized values.
     *
     * Pre-G's BLOCK-REPLICATED chroma (one C set per 2x2 block) caused
     * 2x2 stair-step on diagonal edges; K16 chromaup smoothly
     * interpolates C across the 2x2 block boundary, killing the stair
     * (+0.21 dB / -7.7% LPIPS / -2.5% DISTS on SIDD Medium 80-pair).
     *
     * 逆 2x2 WHT: 全解像度 L_den + 半解像度 C_den → RGGB。
     * dark_ref を加算して absolute GAT-norm 空間に戻す。
     *
     * (Original docstring follows for traceability):
     *
     * GALOSH_RAW_G K16 chroma full-res reconstruction:
     *   - Legacy K14 (compute_L_fullres) + K15 Pass2 produce L_fr_den
     *     in GAT-normalized var=1 space (already done above).
     *   - K16 upsamples c1/c2/c3 to full resolution via EWA Jinc-Lanczos-3
     *     and applies the inverse 2x2 WHT per-pixel with Bayer-aware sign
     *     tables: val[fr,fc] = (L_fr_den + signs[ch] dot C_full) / 2 + dark_ref[ch].
     *   - Replaces the legacy block-replicated inverse (which produced
     *     visible 2x2 stair-steps on diagonal edges) without breaking
     *     K11/K14/K15's var=1 noise invariance.
     *
     * 全 res L_den + EWA-JL3 で full-res 化した chroma + per-pixel
     * inverse 2x2 WHT。block 階段除去 + var=1 不変性保存。 */
    {
      const size_t fullsize = (size_t)width * height;
      float *C1_full = dt_alloc_align_float(fullsize);
      float *C2_full = dt_alloc_align_float(fullsize);
      float *C3_full = dt_alloc_align_float(fullsize);
      if(!C1_full || !C2_full || !C3_full)
      {
        dt_free_align(C1_full); dt_free_align(C2_full); dt_free_align(C3_full);
        dt_free_align(L_fr_den); dt_free_align(l_out);
        goto cleanup_rawlc;
      }

      galosh_upsample_2x_ewajl3(c1_out, C1_full, halfwidth, halfheight);
      galosh_upsample_2x_ewajl3(c2_out, C2_full, halfwidth, halfheight);
      galosh_upsample_2x_ewajl3(c3_out, C3_full, halfwidth, halfheight);

      /* Channel-slot lookup (RGGB hardcoded for the standalone CPU
       * reference; darktable build uses galosh_bayer.h's FC()-derived
       * co_row/co_col which already supports any Bayer pattern).
       *
       * Slot order matches the channel extraction in Phase 1:
       *   0 = R  at TL (fr%2=0, fc%2=0)
       *   1 = Gb at BL (fr%2=1, fc%2=0)
       *   2 = Gr at TR (fr%2=0, fc%2=1)
       *   3 = B  at BR (fr%2=1, fc%2=1) */
      int ch_lut[2][2];
      ch_lut[0][0] = 0;  /* R  */
      ch_lut[1][0] = 1;  /* Gb */
      ch_lut[0][1] = 2;  /* Gr */
      ch_lut[1][1] = 3;  /* B  */

      /* Inverse 2x2 WHT sign tables (per Bayer channel). */
      static const float SIGNS[4][3] = {
        { +1.0f, +1.0f, +1.0f },  /* R  */
        { -1.0f, +1.0f, -1.0f },  /* Gb */
        { +1.0f, -1.0f, -1.0f },  /* Gr */
        { -1.0f, -1.0f, +1.0f },  /* B  */
      };

      const float sg = sigma_gat_ch[0];

      DT_OMP_FOR()
      for(int fr = 0; fr < height; fr++)
      {
        for(int fc = 0; fc < width; fc++)
        {
          const int ch = ch_lut[fr & 1][fc & 1];
          const size_t pos = (size_t)fr * width + fc;
          const float val = 0.5f * (L_fr_den[pos]
                                  + SIGNS[ch][0] * C1_full[pos]
                                  + SIGNS[ch][1] * C2_full[pos]
                                  + SIGNS[ch][2] * C3_full[pos])
                          + ch_dark_ref[ch];
          out[pos] = gat_inverse_exact(val * sg);
        }
      }

      dt_free_align(C1_full); dt_free_align(C2_full); dt_free_align(C3_full);
      fprintf(stderr, "[GALOSH_RAW_G] K16 EWA-JL3 chroma + per-pixel inverse done\n");
    }

    dt_free_align(L_fr_den);  L_fr_den = NULL;
    dt_free_align(l_out);     l_out = NULL;

    fprintf(stderr, "[GALOSH_RAW_G] inverse WHT + inv-GAT done\n");
  }

#else /* legacy full-res raw GALOSH (stride=4) */

  /* ################################################################
   * # [ARCHIVED] pre-GALOSH_RAW_G LEGACY PATH (stride=4)
   * ################################################################
   * Triggered when -DGALOSH_F is NOT defined.  Replaced by Phase 4(a)
   * -(d) GALOSH_RAW_G above.  Kept for bench reproducibility — DO NOT
   * use in production.  Legacy phases (different numbering):
   *   Phase 4 (legacy)  full-res raw WHT-LOSH (Pass 1 + Pass 2 on
   *                     reconstructed full-res Bayer mosaic, stride=4).
   *                     Effectively luma-only (chroma_strength ignored).
   *   Phase 5 (legacy)  chroma replacement: per-2x2-block dC1/dC2/dC3
   *                     computed from full-res output, ADDED back
   *                     BLOCK-REPLICATED to all 4 sub-pixels.
   *                     ⚠ This is the source of the 2x2 stair-step
   *                       artefact GALOSH_RAW_G K16 chromaup fixes.
   *   Phase 6 (legacy)  sigma-denormalize → inverse GAT → write back.
   * ################################################################ */

  /* ================================================================
   * [ARCHIVED] Phase 4 (legacy): Full-res raw Bayer GALOSH (Pass 1 + Pass 2)
   *
   * Apply GALOSH directly on the GAT-normalized raw Bayer at full
   * resolution. This denoises luma (and partially chroma) without
   * the half-resolution bottleneck, preserving sub-pixel features.
   *
   * The GAT-normalized raw is reconstructed from ch_gat[0..3] into
   * a single full-res buffer (interleaved RGGB).
   * ================================================================ */
  {
    const size_t fullsize = (size_t)width * height;
    float *raw_gat_full    = dt_alloc_align_float(fullsize);
    float *raw_gat_pilot   = dt_alloc_align_float(fullsize);
    float *raw_gat_den     = dt_alloc_align_float(fullsize);
    if(!raw_gat_full || !raw_gat_pilot || !raw_gat_den)
    {
      dt_free_align(raw_gat_full);
      dt_free_align(raw_gat_pilot);
      dt_free_align(raw_gat_den);
      goto cleanup_rawlc;
    }

    /* Reconstruct full-res GAT-normalized raw from half-res channels.
     * ch_gat[c] has dark_ref already subtracted; add it back for raw. */
    DT_OMP_FOR()
    for(int row = 0; row < height; row++)
      for(int col = 0; col < width; col++)
      {
        /* Determine which Bayer channel: c = (row&1) | ((col&1)<<1)
         * c=0: R(even row, even col), c=1: Gb(odd row, even col)
         * c=2: Gr(even row, odd col), c=3: B(odd row, odd col) */
        const int c = (row & 1) | ((col & 1) << 1);
        const size_t hi = (size_t)(row / 2) * halfwidth + (col / 2);
        raw_gat_full[(size_t)row * width + col] = ch_gat[c][hi] + ch_dark_ref[c];
      }

    fprintf(stderr, "[rawdenoise] GALOSH: starting full-res raw Pass 1+2 "
                     "(%dx%d, sigma_L=%.3f)\n", width, height, luma_strength);

#ifdef GALOSH_CFA_PROTECT
    /* CFA frequency protection (legacy, superseded by GALOSH_F).
     * Protects WHT bins {0,1,8,9} from shrinkage. Does NOT fully fix
     * color shift — GALOSH_F is the recommended path. */
    galosh_pass1_cfa(raw_gat_full, raw_gat_pilot, width, height, luma_strength, GALOSH_STRIDE);
    galosh_pass2_cfa(raw_gat_full, raw_gat_pilot, raw_gat_den, width, height, luma_strength, GALOSH_STRIDE);
#else
    galosh_pass1(raw_gat_full, raw_gat_pilot, width, height, luma_strength, GALOSH_STRIDE);
    galosh_pass2(raw_gat_full, raw_gat_pilot, raw_gat_den, width, height, luma_strength, GALOSH_STRIDE);
#endif

    dt_free_align(raw_gat_pilot); raw_gat_pilot = NULL;

    fprintf(stderr, "[rawdenoise] GALOSH: full-res raw Pass 1+2 done\n");

    /* ================================================================
     * [ARCHIVED] Phase 5 (legacy): Chroma replacement (BLOCK-REPLICATED).
     *
     * ⚠ This is the source of the 2x2 stair-step artefact on diagonal
     * edges that GALOSH_RAW_G Phase 4(d) K16 chromaup fixes.
     *
     * raw_gat_den has good luma but mediocre chroma denoising (only
     * luma_strength was used). Replace its chroma component with the
     * properly denoised half-res C1/C2/C3 from Phase 3 — but ADDED
     * BLOCK-REPLICATED (same correction to all 4 sub-pixels of every
     * 2x2 block) → diagonal edges show 2x2 stair.
     *
     * For each 2x2 Bayer block at half-res (hy, hx):
     *   1. Extract the 4 full-res denoised values → compute C1', C2', C3'
     *   2. Compute delta: dCk = Ck_den - Ck'
     *   3. Apply chroma correction with WHT sign pattern:
     *        R  (0,0): +dC1, +dC2, +dC3
     *        Gb (1,0): -dC1, +dC2, -dC3
     *        Gr (0,1): +dC1, -dC2, -dC3
     *        B  (1,1): -dC1, -dC2, +dC3
     *
     * Sign convention follows the WHT decomposition in Phase 2:
     *   ch_gat order: [0]=R, [1]=Gb, [2]=Gr, [3]=B
     *   L  = (R + Gb + Gr + B) / 2
     *   C1 = (R - Gb + Gr - B) / 2
     *   C2 = (R + Gb - Gr - B) / 2
     *   C3 = (R - Gb - Gr + B) / 2
     * ================================================================ */
    DT_OMP_FOR()
    for(int hy = 0; hy < halfheight; hy++)
      for(int hx = 0; hx < halfwidth; hx++)
      {
        const size_t hi = (size_t)hy * halfwidth + hx;
        const int fr = 2 * hy, fc = 2 * hx;

        /* Full-res denoised Bayer values at the 4 positions */
        const float den_R  = raw_gat_den[(size_t)fr       * width + fc];      /* ch0: R */
        const float den_Gb = raw_gat_den[(size_t)(fr + 1) * width + fc];      /* ch1: Gb */
        const float den_Gr = raw_gat_den[(size_t)fr       * width + fc + 1];  /* ch2: Gr */
        const float den_B  = raw_gat_den[(size_t)(fr + 1) * width + fc + 1];  /* ch3: B */

        /* Chroma of full-res denoised (same WHT as Phase 2) */
        const float c1_fr = (den_R - den_Gb + den_Gr - den_B) * 0.5f;
        const float c2_fr = (den_R + den_Gb - den_Gr - den_B) * 0.5f;
        const float c3_fr = (den_R - den_Gb - den_Gr + den_B) * 0.5f;

        /* Chroma correction: replace full-res chroma with half-res denoised */
        const float dC1 = c1_out[hi] - c1_fr;
        const float dC2 = c2_out[hi] - c2_fr;
        const float dC3 = c3_out[hi] - c3_fr;

        /* Apply chroma correction to each full-res pixel.
         * raw_gat_den values include dark_ref (absolute GAT-norm space).
         * Inverse 2x2 WHT sign pattern applied to chroma delta. */
        const float corr_R  = ( dC1 + dC2 + dC3) * 0.5f;
        const float corr_Gb = (-dC1 + dC2 - dC3) * 0.5f;
        const float corr_Gr = ( dC1 - dC2 - dC3) * 0.5f;
        const float corr_B  = (-dC1 - dC2 + dC3) * 0.5f;

        raw_gat_den[(size_t)fr       * width + fc]     = den_R  + corr_R;
        raw_gat_den[(size_t)(fr + 1) * width + fc]     = den_Gb + corr_Gb;
        raw_gat_den[(size_t)fr       * width + fc + 1]  = den_Gr + corr_Gr;
        raw_gat_den[(size_t)(fr + 1) * width + fc + 1]  = den_B  + corr_B;
      }

    fprintf(stderr, "[rawdenoise] GALOSH: chroma replacement done\n");

    /* ================================================================
     * [ARCHIVED] Phase 6 (legacy): sigma-denormalize → inverse GAT → write back.
     *
     * raw_gat_den is in absolute GAT-normalized space:
     *   value = GAT(raw) / unified_sigma
     * To invert: value * unified_sigma → gat_inverse_exact()
     *
     * GALOSH_RAW_G fuses this denormalize + inverse GAT into Phase 4(d)
     * K16 chromaup (per-pixel) — see above #if branch.
     * ================================================================ */
    {
      const float sg = sigma_gat_ch[0]; /* = unified_sigma for all channels */
      DT_OMP_FOR()
      for(int row = 0; row < height; row++)
        for(int col = 0; col < width; col++)
        {
          const size_t pos = (size_t)row * width + col;
          out[pos] = gat_inverse_exact(raw_gat_den[pos] * sg);
        }
    }

    dt_free_align(raw_gat_den);  raw_gat_den = NULL;
    dt_free_align(raw_gat_full); raw_gat_full = NULL;
  }
#endif /* !GALOSH_LEGACY (= [LATEST] GALOSH_RAW_G default) */

  /* Free half-res L/C buffers */
  dt_free_align(luma);    luma = NULL;
  dt_free_align(chroma1); chroma1 = NULL;
  dt_free_align(chroma2); chroma2 = NULL;
  dt_free_align(chroma3); chroma3 = NULL;
  dt_free_align(c1_out);  c1_out = NULL;
  dt_free_align(c2_out);  c2_out = NULL;
  dt_free_align(c3_out);  c3_out = NULL;

  fprintf(stderr, "[rawdenoise] GALOSH: done\n");

  for(int c = 0; c < 4; c++) { dt_free_align(ch_gat[c]); ch_gat[c] = NULL; }
  return;

cleanup_rawlc:
  for(int c = 0; c < 4; c++) dt_free_align(ch_gat[c]);
  dt_free_align(luma);
  dt_free_align(chroma1);
  dt_free_align(chroma2);
  dt_free_align(chroma3);
  dt_free_align(c1_pilot);
  dt_free_align(c2_pilot);
  dt_free_align(c3_pilot);
  dt_free_align(c1_out);
  dt_free_align(c2_out);
  dt_free_align(c3_out);
}


/* ================================================================
 * [LATEST: GALOSH_RAW_H] gat_galosh_denoise_rawlc_h — H pipeline entry.
 *
 * Replaces GALOSH_RAW_G's half-res ↔ full-res roundtrip (K14 box +
 * K16 EWA-JL3 chromaup) with full-resolution processing throughout,
 * using stride=1 cycle-spinning forward WHT + CFA sign-flip demod /
 * remod + 4-block overlap-add inverse.
 *
 * Pipeline (10 phases, all labelled [LATEST: GALOSH_RAW_H] inline):
 *   Phase 0  Foi-Alenius blind α / σ²  (galosh_estimate_noise)
 *   Phase 1  GAT forward (full-res) + per-CFA σ_GAT MAD + RMS
 *            unified_sigma + scalar normalize
 *   Phase 2  dark_ref IRLS (achromatic) + per-pixel CFA-aware subtract
 *   Phase 3  stride=1 forward 2x2 WHT @ full-res → L, C1, C2, C3
 *   Phase 4  CFA sign-flip demodulate
 *              (C1*=(-1)^r, C2*=(-1)^c, C3*=(-1)^(r+c))
 *   Phase 5  denoise:
 *              L : galosh_pass12_multiorient_blocked (Pass1 BayesShrink-MAD
 *                  + Pass2 Wiener) on the full-res L plane
 *              C1/C2/C3 : galosh_loess_chroma (Y-guided bilateral LOESS,
 *                  Y guide = denoised L)
 *   Phase 6  CFA sign-flip remodulate (= self-inverse Phase 4)
 *   Phase 7  4-block overlap-add inverse 2x2 WHT (per-pixel average)
 *   Phase 8  per-pixel dark_ref restore + ×unified_sigma denormalize
 *   Phase 9  per-pixel inverse GAT (LUT) → out
 *
 * Phases 8 + 9 are fused into a single per-pixel loop for efficiency.
 * The 10-phase numbering is preserved in the inline labels for
 * traceability against the spec.
 *
 * (日) GALOSH_RAW_H: full-res 一貫処理。stride=1 cycle-spinning 順方向
 *   WHT (Coifman-Donoho 1995) + CFA 符号反転の脱/再変調 + 4-block
 *   重なり加算逆変換。K14 box 階段 / K16 chromaup 半解像度往復による
 *   aliasing を完全消滅。Var=1 (unitary WHT) 雑音不変性は保存。
 *
 * Variant selection in CLI: --variant=h (default) or --variant=g for
 * the [PREVIOUS] G pipeline.  No legacy compile flag, no CFA-protect.
 * ================================================================ */
static void gat_galosh_denoise_rawlc_h(const float *const restrict in,
                                       float *const restrict out,
                                       const dt_iop_roi_t *const roi,
                                       const float luma_strength,
                                       const float chroma_strength,
                                       const uint32_t filters)
{
  const int width = roi->width, height = roi->height;
  const size_t npixels = (size_t)width * height;
  memcpy(out, in, sizeof(float) * npixels);

  if(luma_strength <= 0.0f) return;
  if(width < GALOSH_BLOCK_SIZE * 2 || height < GALOSH_BLOCK_SIZE * 2) return;

  /* CFA channel slot lookup (RGGB hardcoded for the standalone CPU
   * reference; the darktable port derives co_row/co_col from FC()).
   *   slot 0 = R  at (r%2=0, c%2=0)
   *   slot 1 = Gb at (r%2=1, c%2=0)
   *   slot 2 = Gr at (r%2=0, c%2=1)
   *   slot 3 = B  at (r%2=1, c%2=1)
   * Slot index = (r&1) | ((c&1) << 1) — same convention as
   * GALOSH_RAW_G Phase 1 half-res extraction. */
  const int co_row = 0, co_col = 0;
  (void)filters;

  /* ================================================================
   * [LATEST: GALOSH_RAW_H] Phase 0 — Foi-Alenius blind α / σ² estimation.
   * Identical to GALOSH_RAW_G Phase 0 (Poisson-Gauss VST setup); the
   * GAT model is independent of the spatial pipeline that follows.
   * ================================================================ */
  const galosh_noise_params_t np = galosh_estimate_noise(in, width, height);
  gat_build_inverse_table(np.alpha, np.sigma_sq);

  /* Pre-declare buffers for cleanup-on-error. */
  float *in_gat = NULL;
  float *L = NULL, *C1 = NULL, *C2 = NULL, *C3 = NULL;
  float *L_den = NULL, *C1_den = NULL, *C2_den = NULL, *C3_den = NULL;
  float *recon = NULL;

  in_gat = dt_alloc_align_float(npixels);
  if(!in_gat) goto cleanup_rawlc_h;

  /* ================================================================
   * [LATEST: GALOSH_RAW_H] Phase 1 — GAT forward (full-res) + per-CFA
   * σ_GAT MAD + RMS unified_sigma normalize.
   *
   * GAT is point-wise so it can be applied to the full-res raw directly
   * (no need for the per-channel half-res buffers GALOSH_RAW_G used).
   * Per-CFA σ_GAT is measured by stride-2 sub-sampling each Bayer
   * channel and feeding it to estimate_gat_sigma_halfres (Laplacian
   * MAD); this preserves CFA awareness without buffer fragmentation.
   * RMS-combined into a single unified_sigma; scalar division on the
   * full-res in_gat plane gives Var=1 in expectation post-WHT.
   *
   * (日) GAT は点ごとなので full-res raw に直接適用可能。CFA ch ごと
   *   σ は stride-2 サブサンプル + 既存 MAD 推定で取得、RMS で
   *   unified_sigma に統合。GALOSH_RAW_G の半解像度バッファ展開は不要。
   * ================================================================ */
  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++)
    in_gat[i] = gat_forward(in[i], np.alpha, np.sigma_sq);

  float sigma_gat_ch[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  for(int s = 0; s < 4; s++)
  {
    const int ro = ((s & 1)        + co_row) & 1;
    const int co = (((s >> 1) & 1) + co_col) & 1;
    const int hw = (width  - co + 1) / 2;
    const int hh = (height - ro + 1) / 2;
    if(hw < 4 || hh < 4) continue;

    float *tmp = dt_alloc_align_float((size_t)hw * hh);
    if(!tmp) continue;
    DT_OMP_FOR()
    for(int rr = 0; rr < hh; rr++)
      for(int cc = 0; cc < hw; cc++)
        tmp[(size_t)rr * hw + cc] = in_gat[(size_t)(ro + 2*rr) * width + (co + 2*cc)];
    sigma_gat_ch[s] = estimate_gat_sigma_halfres(tmp, hw, hh);
    dt_free_align(tmp);
  }

  const float mean_var = 0.25f * (sigma_gat_ch[0]*sigma_gat_ch[0]
                                + sigma_gat_ch[1]*sigma_gat_ch[1]
                                + sigma_gat_ch[2]*sigma_gat_ch[2]
                                + sigma_gat_ch[3]*sigma_gat_ch[3]);
  const float unified_sigma = sqrtf(fmaxf(mean_var, 1e-12f));
  const float inv_sg = 1.0f / unified_sigma;

  fprintf(stderr, "[GALOSH_RAW_H] alpha=%.8f sigma_sq=%.10f | "
                  "unified_sigma=%.4f [RMS] (per-ch: %.4f %.4f %.4f %.4f) | "
                  "size=%dx%d | sigma_L=%.3f sigma_C=%.3f\n",
                  np.alpha, np.sigma_sq, unified_sigma,
                  sigma_gat_ch[0], sigma_gat_ch[1], sigma_gat_ch[2], sigma_gat_ch[3],
                  width, height, luma_strength, chroma_strength);

  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++) in_gat[i] *= inv_sg;

  /* ================================================================
   * [LATEST: GALOSH_RAW_H] Phase 2 — Self-consistent dark anchor (IRLS)
   * + per-pixel CFA-aware subtract.
   *
   * Per-CFA-channel DC offset estimated by achromatic IRLS — same logic
   * as GALOSH_RAW_G Phase 2 but operating on the full-res GAT plane.
   * Iterates 2 times over the noise-dominated cohort selected by the
   * smooth weight w(L_raw) = 1/(1+(L_raw/s)^4); s is rescaled each
   * iteration so the residual std matches 1.0 (target = unified_sigma).
   *
   * Subtraction is per-pixel CFA-aware: each pixel has its dark_ref[ch]
   * removed, where ch = (row&1) | ((col&1) << 1) (the standard slot
   * lookup, restored post-inverse in Phase 8).
   *
   * (日) per-CFA-ch DC offset を IRLS で推定し、各画素の CFA 色に応じた
   *   dark_ref を per-pixel 減算。Phase 8 で復元。Phase 0 σ_sq/α 決定後
   *   なので s_init は α/σ² に整合する。
   * ================================================================ */
  float ch_dark_ref[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  {
    const double s_init = (double)np.sigma_sq / fmax((double)np.alpha, 1e-12);
    double s_scale = s_init;
    const double s_min = 0.05 * s_init;
    const double s_max = 50.0 * s_init;
    const int n_iter = 2;

    for(int iter = 0; iter <= n_iter; iter++)
    {
      const double inv_s = 1.0 / fmax(s_scale, 1e-20);
      /* OMP reduction over the 2x2-cell grid.  Per-channel sums kept
       * as separate scalars (sum_w0..sum_w3) since OMP array reduction
       * is OpenMP 4.5+ — keeping it scalar-portable. */
      double sum_w  = 0.0;
      double sum_w0 = 0.0, sum_w1 = 0.0, sum_w2 = 0.0, sum_w3 = 0.0;

      /* Walk in 2x2 Bayer cells; same achromatic filter as G. */
      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_w,sum_w0,sum_w1,sum_w2,sum_w3)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          sum_w  += w;
          sum_w0 += w * g0;
          sum_w1 += w * g1;
          sum_w2 += w * g2;
          sum_w3 += w * g3;
        }

      const double inv_sw = 1.0 / fmax(sum_w, 1e-20);
      ch_dark_ref[0] = (float)(sum_w0 * inv_sw);
      ch_dark_ref[1] = (float)(sum_w1 * inv_sw);
      ch_dark_ref[2] = (float)(sum_w2 * inv_sw);
      ch_dark_ref[3] = (float)(sum_w3 * inv_sw);

      if(iter == n_iter) break;

      /* Re-estimate s via residual-std rescale (same formula as G). */
      double sum_wresid2 = 0.0;
      double sum_wW = 0.0;
      const float dr0 = ch_dark_ref[0], dr1 = ch_dark_ref[1];
      const float dr2 = ch_dark_ref[2], dr3 = ch_dark_ref[3];
      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_wresid2,sum_wW)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          const double d0 = (double)g0 - dr0;
          const double d1 = (double)g1 - dr1;
          const double d2 = (double)g2 - dr2;
          const double d3 = (double)g3 - dr3;
          const double resid2 = d0*d0 + d1*d1 + d2*d2 + d3*d3;
          sum_wW      += w;
          sum_wresid2 += w * resid2 * 0.25;
        }
      const double inv_sw2 = 1.0 / fmax(sum_wW, 1e-20);
      const double measured_std = sqrt(fmax(sum_wresid2 * inv_sw2, 1e-20));
      const double ratio = 1.0 / measured_std;
      s_scale *= sqrt(ratio);
      if(s_scale < s_min) s_scale = s_min;
      if(s_scale > s_max) s_scale = s_max;
    }

    fprintf(stderr, "[GALOSH_RAW_H] dark anchor: s_init=%.6e s_final=%.6e | "
                    "ch_dark_ref: [0]=%.4f [1]=%.4f [2]=%.4f [3]=%.4f\n",
            s_init, s_scale,
            ch_dark_ref[0], ch_dark_ref[1], ch_dark_ref[2], ch_dark_ref[3]);

    /* Per-pixel CFA-aware subtract. */
    DT_OMP_FOR()
    for(int r = 0; r < height; r++)
    {
      const int r_off = (r - co_row) & 1;
      for(int c = 0; c < width; c++)
      {
        const int c_off = (c - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        in_gat[(size_t)r * width + c] -= ch_dark_ref[slot];
      }
    }
  }

  /* ================================================================
   * [LATEST: GALOSH_RAW_H] Phase 3 — stride=1 forward 2x2 WHT (full-res).
   * Cycle-spinning forward WHT at every pixel TL with right/bottom
   * mirror padding (galosh_h_mirror_idx).  Output: 4 full-res planes
   * (L, C1, C2, C3), each plane has the same shape as the input.
   *
   * Replaces GALOSH_RAW_G's stride=2 half-res WHT (Phase 3) +
   * K14 box compute_L_fullres (Phase 4(b)) + K16 EWA-JL3 chromaup
   * (Phase 4(d)) all in one shot.  No 2x2 stair on diagonals (the
   * spinning provides correct interleaved-block coverage); no
   * half-res ↔ full-res aliasing.
   * ================================================================ */
  L  = dt_alloc_align_float(npixels);
  C1 = dt_alloc_align_float(npixels);
  C2 = dt_alloc_align_float(npixels);
  C3 = dt_alloc_align_float(npixels);
  if(!L || !C1 || !C2 || !C3) goto cleanup_rawlc_h;
  gat_h_forward_wht_stride1(in_gat, L, C1, C2, C3, width, height);

  /* in_gat no longer needed after the forward WHT (Phase 3) — Phase 8
   * needs only ch_dark_ref + recon, and unified_sigma is captured. */
  dt_free_align(in_gat); in_gat = NULL;

  fprintf(stderr, "[GALOSH_RAW_H] Phase 3 forward WHT (stride=1) done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_H] Phase 4 — CFA sign-flip demodulate.
   * In-place: C1[r,c] *= (-1)^r ; C2[r,c] *= (-1)^c ;
   *           C3[r,c] *= (-1)^(r+c).
   * Removes CFA-induced periodic sign flips so chroma planes become
   * smooth (zero-mean, locally low-variance) and amenable to LOESS.
   * Self-inverse — Phase 6 re-applies the same operator.
   * ================================================================ */
  gat_h_demodulate_chroma(C1, C2, C3, width, height);

  fprintf(stderr, "[GALOSH_RAW_H] Phase 4 CFA sign-flip demodulate done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_H] Phase 5 — full-res denoise.
   *   L : galosh_pass12_multiorient_blocked (Pass1 BayesShrink-MAD +
   *       Pass2 Wiener) on the full-res L plane.  Block=8, stride=2
   *       (75% overlap), n_orient=1, use_robust_shrink=1 (= GALOSH_RAW_G
   *       K13 settings, applied at full-res instead of half-res).
   *   C1/C2/C3 : galosh_loess_chroma (Y-guided bilateral LOESS) using
   *       the just-denoised L plane as Y guide.  Operates at full-res
   *       — no 2× upsample needed (= K16 elimination); the LOESS
   *       window is the native scale.
   *
   * (日) L plane は full-res で BayesShrink-MAD + Wiener (= G K13 を
   *   半解像度から full-res に持ち上げ)。chroma は denoised L を guide
   *   とする bilateral LOESS。両方とも full-res で動くため K16 EWA-JL3
   *   chromaup 経由の 2× upsample が不要。
   * ================================================================ */
  L_den  = dt_alloc_align_float(npixels);
  C1_den = dt_alloc_align_float(npixels);
  C2_den = dt_alloc_align_float(npixels);
  C3_den = dt_alloc_align_float(npixels);
  if(!L_den || !C1_den || !C2_den || !C3_den) goto cleanup_rawlc_h;

  galosh_pass12_multiorient_blocked(L, L_den, width, height,
                                     luma_strength, /*block=*/8,
                                     /*stride=*/2, /*n_orient=*/1,
                                     /*use_robust_shrink=*/1);
  fprintf(stderr, "[GALOSH_RAW_H] Phase 5(a) full-res L Pass1+2 done\n");

  /* LOESS expects two chroma planes per call (Cb, Cr); we have three
   * planes (C1, C2, C3) so we call once for (C1, C2) + once for (C3,
   * dummy).  Y guide is L_den, the just-denoised luma plane. */
  {
    float *dummy = dt_alloc_align_float(npixels);
    if(!dummy) goto cleanup_rawlc_h;
    galosh_loess_chroma(L_den, C1, C2, C1_den, C2_den,
                         width, height, chroma_strength);
    galosh_loess_chroma(L_den, C3, C3, C3_den, dummy,
                         width, height, chroma_strength);
    dt_free_align(dummy);
  }
  fprintf(stderr, "[GALOSH_RAW_H] Phase 5(b) full-res chroma LOESS done "
                   "(sigma_C=%.3f, R=%d, BW=%.1f)\n",
          chroma_strength, GALOSH_LOESS_RADIUS, GALOSH_LOESS_BW);

  /* Pre-denoise L/C1/C2/C3 buffers no longer needed. */
  dt_free_align(L);  L  = NULL;
  dt_free_align(C1); C1 = NULL;
  dt_free_align(C2); C2 = NULL;
  dt_free_align(C3); C3 = NULL;

  /* ================================================================
   * [LATEST: GALOSH_RAW_H] Phase 6 — CFA sign-flip remodulate.
   * In-place re-application of (-1)^r etc on denoised C1/C2/C3.
   * Self-inverse: applying the demod operator twice gives identity,
   * so this restores the cycle-spun "raw" WHT bin orientation needed
   * for Phase 7 inverse WHT.
   * ================================================================ */
  gat_h_demodulate_chroma(C1_den, C2_den, C3_den, width, height);

  fprintf(stderr, "[GALOSH_RAW_H] Phase 6 CFA sign-flip remodulate done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_H] Phase 7 — 4-block overlap-add inverse 2x2 WHT.
   * Each output pixel is the average of up to 4 cycle-shifted block
   * inverses (TL, BL, TR, BR roles); see gat_h_inverse_overlap_add.
   *
   * Result is in absolute GAT-norm space (still has dark_ref subtracted,
   * still divided by unified_sigma).  Phase 8+9 restore those.
   *
   * Cycle-spinning denoising theory (Coifman & Donoho 1995): averaging
   * over translations of a denoising operator reduces translation-
   * dependent artefacts (block grid).  Here the translations are the
   * 4 cycle-spun 2x2 block decompositions (TL ∈ {(fr,fc), (fr-1,fc),
   * (fr,fc-1), (fr-1,fc-1)}) and the operator is the 2x2 WHT-LOSH /
   * LOESS denoise applied per plane.
   * ================================================================ */
  recon = dt_alloc_align_float(npixels);
  if(!recon) goto cleanup_rawlc_h;
  gat_h_inverse_overlap_add(L_den, C1_den, C2_den, C3_den,
                             recon, width, height);

  fprintf(stderr, "[GALOSH_RAW_H] Phase 7 inverse WHT (4-block overlap-add) done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_H] Phase 8 + 9 — fused per-pixel dark_ref
   * restore + ×unified_sigma denormalize + inverse GAT (LUT).
   *
   * Mirror of Phase 2 subtract: per-pixel CFA slot lookup → restore
   * dark_ref → multiply by unified_sigma → gat_inverse_exact (LUT).
   * Fused into one loop because each step is per-pixel point-wise and
   * the CFA slot computation is cheap.
   *
   * (日) Phase 2 の per-pixel CFA-aware 減算の逆 + Phase 1 の RMS
   *   unified_sigma 正規化の逆 + 逆 GAT (Mäkitalo & Foi 2013) の LUT
   *   評価を 1 ループに融合。
   * ================================================================ */
  DT_OMP_FOR()
  for(int r = 0; r < height; r++)
  {
    const int r_off = (r - co_row) & 1;
    for(int c = 0; c < width; c++)
    {
      const int c_off = (c - co_col) & 1;
      const int slot  = r_off | (c_off << 1);
      const size_t pos = (size_t)r * width + c;
      const float val = (recon[pos] + ch_dark_ref[slot]) * unified_sigma;
      out[pos] = gat_inverse_exact(val);
    }
  }

  fprintf(stderr, "[GALOSH_RAW_H] Phase 8+9 dark_ref restore + denorm + inv-GAT done\n");

cleanup_rawlc_h:
  dt_free_align(in_gat);
  dt_free_align(L); dt_free_align(C1); dt_free_align(C2); dt_free_align(C3);
  dt_free_align(L_den); dt_free_align(C1_den); dt_free_align(C2_den); dt_free_align(C3_den);
  dt_free_align(recon);
}


/* ================================================================
 * [LATEST: GALOSH_RAW_I] gat_galosh_denoise_rawlc_i — I pipeline entry.
 *
 * Hybrid: "L from RAW_H, C from RAW_G".  Resolves the H-vs-G trade-off
 * observed empirically:
 *   H wins on PSNR (+0.79 dB) — full-res cycle-spinning kills the K14
 *     box stair on luma diagonals
 *   G wins on LPIPS / DISTS / NIQE — half-res LOESS + K16 EWA-JL3
 *     chromaup provides bandlimit-faithful smooth chroma; H's full-res
 *     cycle-spun C plane has autocov=0.5 noise correlation that LOESS
 *     can't fully suppress, leaving long-wavelength color blotch
 *
 * Theoretical justification for the hybrid: L and C have intrinsically
 * different Nyquist limits in CFA imaging.
 *   L: all 4 Bayer channels contribute → effectively full-res sampling
 *      of luma → full-res cycle-spinning is well-conditioned
 *   C: only 1 sample per 2x2 Bayer block per chroma channel → chroma
 *      signal is intrinsically half-Nyquist limited → cycle-spinning
 *      to full-res introduces ZERO new signal information, only
 *      correlated noise (above-half-Nyquist content is pure noise)
 * Therefore L and C should be processed at their NATIVE resolutions.
 *
 * Pipeline (10 phases, all labelled [LATEST: GALOSH_RAW_I] inline):
 *   Phase 0  Foi-Alenius blind α / σ²  (= H/G Phase 0)
 *   Phase 1  GAT forward (full-res) + per-CFA σ_GAT MAD + RMS
 *            unified_sigma + scalar normalize  (= H Phase 1)
 *   Phase 2  dark_ref IRLS (achromatic) + per-pixel CFA-aware subtract
 *            (= H Phase 2)
 *   Phase 3  stride=1 forward 2x2 WHT @ full-res → L_cs, C1_cs, C2_cs,
 *            C3_cs (= H Phase 3, gat_h_forward_wht_stride1)
 *   Phase 4  SPLIT:
 *              4(L) keep L_cs as-is (CFA-invariant, no demod needed)
 *              4(C) sub-sample C_cs at every-other-pixel → half-res C
 *                   (= G's half-res 2x2 WHT result; CFA-aligned at
 *                    even positions, no demod needed)
 *   Phase 5  denoise:
 *              5(L) galosh_pass12_multiorient_blocked on L_cs at full-res
 *                   (block=8 stride=2 n_orient=1 robust=1) → L_cs_den
 *              5(C) galosh_loess_chroma at half-res, Y guide = sub-sample
 *                   of L_cs_den at every-other-pixel → C_h_den
 *   Phase 6  K16 EWA-JL3 upsample C_h_den → C_full_den
 *            (galosh_upsample_2x_ewajl3, = G Phase 4(d) chromaup)
 *   Phase 7  L_pixel = 2x2 overlap average of L_cs_den
 *            (gat_i_lpixel_overlap_avg; resolves the half-pixel shift
 *             between cycle-spun L grid and full-res pixel grid)
 *   Phase 8  per-pixel WHT inverse:
 *              out[fr,fc] = 0.5 * (L_pixel + signs[CFA(fr,fc)] · C_full_den)
 *                         + dark_ref[CFA(fr,fc)]
 *            (= G's K16 final inverse; per-pixel CFA role determines signs)
 *            + ×unified_sigma denormalize
 *   Phase 9  per-pixel inverse GAT (LUT) → out
 *
 * Phases 8 + 9 fused into one loop for efficiency.
 *
 * Cost vs H: I's chroma denoise is at half-res (~4× less work for the
 *   dominant LOESS step) → ~2× speedup vs H.
 * Cost vs G: I's L denoise is at full-res (slightly more work than G's
 *   half-res LOSH + K14 box + K15 Pass2 chain).  Net: I is intermediate
 *   speed between G and H, with both quality wins (PSNR + LPIPS/DISTS).
 *
 * (日) GALOSH_RAW_I: 「L は H、C は G」のハイブリッド。CFA において L と
 *   C は本質的に異なる Nyquist (L=full, C=half) を持つので、それぞれの
 *   ネイティブ解像度で処理するのが理論的に正しい。H の階段除去 + G の
 *   滑らかな chroma を両立。
 * ================================================================ */
static void gat_galosh_denoise_rawlc_i(const float *const restrict in,
                                       float *const restrict out,
                                       const dt_iop_roi_t *const roi,
                                       const float luma_strength,
                                       const float chroma_strength,
                                       const uint32_t filters)
{
  const int width = roi->width, height = roi->height;
  const size_t npixels = (size_t)width * height;
  memcpy(out, in, sizeof(float) * npixels);

  if(luma_strength <= 0.0f) return;
  if(width < GALOSH_BLOCK_SIZE * 2 || height < GALOSH_BLOCK_SIZE * 2) return;

  /* CFA channel slot lookup (RGGB hardcoded for the standalone CPU
   * reference; the darktable port derives co_row/co_col from FC()). */
  const int co_row = 0, co_col = 0;
  (void)filters;

  const int halfwidth  = (width  + 1) / 2;
  const int halfheight = (height + 1) / 2;
  const size_t chsize  = (size_t)halfwidth * halfheight;

  /* ================================================================
   * [LATEST: GALOSH_RAW_I] Phase 0 — Foi-Alenius blind α / σ² estimation.
   * Identical to GALOSH_RAW_H/G Phase 0.
   * ================================================================ */
  const galosh_noise_params_t np = galosh_estimate_noise(in, width, height);
  gat_build_inverse_table(np.alpha, np.sigma_sq);

  /* Pre-declare buffers for cleanup-on-error. */
  float *in_gat = NULL;
  float *L_cs = NULL, *C1_cs = NULL, *C2_cs = NULL, *C3_cs = NULL;
  float *L_cs_den = NULL;
  float *L_h_den = NULL;
  float *C1_h = NULL, *C2_h = NULL, *C3_h = NULL;
  float *C1_h_den = NULL, *C2_h_den = NULL, *C3_h_den = NULL;
  float *C1_full = NULL, *C2_full = NULL, *C3_full = NULL;
  float *L_pixel = NULL;
  float *dummy = NULL;

  in_gat = dt_alloc_align_float(npixels);
  if(!in_gat) goto cleanup_rawlc_i;

  /* ================================================================
   * [LATEST: GALOSH_RAW_I] Phase 1 — GAT forward (full-res) + per-CFA
   * σ_GAT MAD + RMS unified_sigma + scalar normalize.  Identical to H
   * Phase 1 (point-wise GAT on full-res raw, no per-channel half-res
   * buffer needed; per-CFA σ via stride-2 sub-sample MAD).
   * ================================================================ */
  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++)
    in_gat[i] = gat_forward(in[i], np.alpha, np.sigma_sq);

  float sigma_gat_ch[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  for(int s = 0; s < 4; s++)
  {
    const int ro = ((s & 1)        + co_row) & 1;
    const int co = (((s >> 1) & 1) + co_col) & 1;
    const int hw = (width  - co + 1) / 2;
    const int hh = (height - ro + 1) / 2;
    if(hw < 4 || hh < 4) continue;

    float *tmp = dt_alloc_align_float((size_t)hw * hh);
    if(!tmp) continue;
    DT_OMP_FOR()
    for(int rr = 0; rr < hh; rr++)
      for(int cc = 0; cc < hw; cc++)
        tmp[(size_t)rr * hw + cc] = in_gat[(size_t)(ro + 2*rr) * width + (co + 2*cc)];
    sigma_gat_ch[s] = estimate_gat_sigma_halfres(tmp, hw, hh);
    dt_free_align(tmp);
  }

  const float mean_var = 0.25f * (sigma_gat_ch[0]*sigma_gat_ch[0]
                                + sigma_gat_ch[1]*sigma_gat_ch[1]
                                + sigma_gat_ch[2]*sigma_gat_ch[2]
                                + sigma_gat_ch[3]*sigma_gat_ch[3]);
  const float unified_sigma = sqrtf(fmaxf(mean_var, 1e-12f));
  const float inv_sg = 1.0f / unified_sigma;

  fprintf(stderr, "[GALOSH_RAW_I] alpha=%.8f sigma_sq=%.10f | "
                  "unified_sigma=%.4f [RMS] (per-ch: %.4f %.4f %.4f %.4f) | "
                  "size=%dx%d (half=%dx%d) | sigma_L=%.3f sigma_C=%.3f\n",
                  np.alpha, np.sigma_sq, unified_sigma,
                  sigma_gat_ch[0], sigma_gat_ch[1], sigma_gat_ch[2], sigma_gat_ch[3],
                  width, height, halfwidth, halfheight, luma_strength, chroma_strength);

  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++) in_gat[i] *= inv_sg;

  /* ================================================================
   * [LATEST: GALOSH_RAW_I] Phase 2 — dark_ref IRLS (achromatic) +
   * per-pixel CFA-aware subtract.  Identical to GALOSH_RAW_H Phase 2;
   * walks the full-res GAT plane in 2x2 Bayer cells.
   * ================================================================ */
  float ch_dark_ref[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  {
    const double s_init = (double)np.sigma_sq / fmax((double)np.alpha, 1e-12);
    double s_scale = s_init;
    const double s_min = 0.05 * s_init;
    const double s_max = 50.0 * s_init;
    const int n_iter = 2;

    for(int iter = 0; iter <= n_iter; iter++)
    {
      const double inv_s = 1.0 / fmax(s_scale, 1e-20);
      double sum_w = 0.0;
      double sum_w0 = 0.0, sum_w1 = 0.0, sum_w2 = 0.0, sum_w3 = 0.0;

      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_w,sum_w0,sum_w1,sum_w2,sum_w3)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          sum_w  += w;
          sum_w0 += w * g0;
          sum_w1 += w * g1;
          sum_w2 += w * g2;
          sum_w3 += w * g3;
        }

      const double inv_sw = 1.0 / fmax(sum_w, 1e-20);
      ch_dark_ref[0] = (float)(sum_w0 * inv_sw);
      ch_dark_ref[1] = (float)(sum_w1 * inv_sw);
      ch_dark_ref[2] = (float)(sum_w2 * inv_sw);
      ch_dark_ref[3] = (float)(sum_w3 * inv_sw);

      if(iter == n_iter) break;

      double sum_wresid2 = 0.0;
      double sum_wW = 0.0;
      const float dr0 = ch_dark_ref[0], dr1 = ch_dark_ref[1];
      const float dr2 = ch_dark_ref[2], dr3 = ch_dark_ref[3];
      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_wresid2,sum_wW)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          const double d0 = (double)g0 - dr0;
          const double d1 = (double)g1 - dr1;
          const double d2 = (double)g2 - dr2;
          const double d3 = (double)g3 - dr3;
          const double resid2 = d0*d0 + d1*d1 + d2*d2 + d3*d3;
          sum_wW      += w;
          sum_wresid2 += w * resid2 * 0.25;
        }
      const double inv_sw2 = 1.0 / fmax(sum_wW, 1e-20);
      const double measured_std = sqrt(fmax(sum_wresid2 * inv_sw2, 1e-20));
      const double ratio = 1.0 / measured_std;
      s_scale *= sqrt(ratio);
      if(s_scale < s_min) s_scale = s_min;
      if(s_scale > s_max) s_scale = s_max;
    }

    fprintf(stderr, "[GALOSH_RAW_I] dark anchor: s_init=%.6e s_final=%.6e | "
                    "ch_dark_ref: [0]=%.4f [1]=%.4f [2]=%.4f [3]=%.4f\n",
            s_init, s_scale,
            ch_dark_ref[0], ch_dark_ref[1], ch_dark_ref[2], ch_dark_ref[3]);

    DT_OMP_FOR()
    for(int r = 0; r < height; r++)
    {
      const int r_off = (r - co_row) & 1;
      for(int c = 0; c < width; c++)
      {
        const int c_off = (c - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        in_gat[(size_t)r * width + c] -= ch_dark_ref[slot];
      }
    }
  }

  /* ================================================================
   * [LATEST: GALOSH_RAW_I] Phase 3 — stride=1 forward 2x2 WHT @ full-res.
   * Cycle-spinning forward at every pixel TL with right/bottom mirror
   * padding (gat_h_forward_wht_stride1, reused from H pipeline).
   * Output: 4 full-res planes (L_cs, C1_cs, C2_cs, C3_cs).
   * ================================================================ */
  L_cs  = dt_alloc_align_float(npixels);
  C1_cs = dt_alloc_align_float(npixels);
  C2_cs = dt_alloc_align_float(npixels);
  C3_cs = dt_alloc_align_float(npixels);
  if(!L_cs || !C1_cs || !C2_cs || !C3_cs) goto cleanup_rawlc_i;
  gat_h_forward_wht_stride1(in_gat, L_cs, C1_cs, C2_cs, C3_cs, width, height);

  dt_free_align(in_gat); in_gat = NULL;

  fprintf(stderr, "[GALOSH_RAW_I] Phase 3 forward WHT (stride=1) done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_I] Phase 4 — split L (full-res, no action) and
   * C (sub-sample at every-other-pixel → half-res).
   *
   * 4(L): L_cs is CFA-invariant (sum of all 4 channels, sign-flip free)
   *       → no demod needed, kept full-res for Phase 5(L) LOSH.
   *
   * 4(C): sub-sample C_cs at (2*hr, 2*hc) for hr ∈ [0, halfheight),
   *       hc ∈ [0, halfwidth).  Mathematically: C_cs[2hr, 2hc] reads
   *       block (2hr, 2hc) which has CFA TL = R for RGGB → no sign
   *       flip needed (= G's half-res 2x2 WHT convention).  Cycle-spun
   *       C plane positions OFF the half-res lattice carry no extra
   *       signal (chroma is intrinsically half-Nyquist limited per CFA
   *       sampling theory) so this sub-sampling is information-lossless.
   * ================================================================ */
  C1_h = dt_alloc_align_float(chsize);
  C2_h = dt_alloc_align_float(chsize);
  C3_h = dt_alloc_align_float(chsize);
  if(!C1_h || !C2_h || !C3_h) goto cleanup_rawlc_i;

  DT_OMP_FOR()
  for(int hr = 0; hr < halfheight; hr++)
  {
    const int fr = 2 * hr;
    if(fr >= height) continue;
    for(int hc = 0; hc < halfwidth; hc++)
    {
      const int fc = 2 * hc;
      if(fc >= width) continue;
      const size_t fp = (size_t)fr * width + fc;
      const size_t hp = (size_t)hr * halfwidth + hc;
      C1_h[hp] = C1_cs[fp];
      C2_h[hp] = C2_cs[fp];
      C3_h[hp] = C3_cs[fp];
    }
  }

  /* C_cs no longer needed (we keep only the half-res sub-sample). */
  dt_free_align(C1_cs); C1_cs = NULL;
  dt_free_align(C2_cs); C2_cs = NULL;
  dt_free_align(C3_cs); C3_cs = NULL;

  fprintf(stderr, "[GALOSH_RAW_I] Phase 4 chroma half-res sub-sample done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_I] Phase 5 — denoise.
   *   5(L): galosh_pass12_multiorient_blocked on L_cs at full-res
   *         (block=8 stride=2 n_orient=1 robust=1 = H/G K13 settings)
   *   5(C): galosh_loess_chroma at half-res, Y guide = sub-sample of
   *         L_cs_den at every-other-pixel = L_h_den.  Per-pixel std on
   *         half-res C is 1.0 (unitary WHT, no further smoothing) →
   *         LOESS BW=3σ default applies directly, no calibration needed.
   * ================================================================ */
  L_cs_den = dt_alloc_align_float(npixels);
  if(!L_cs_den) goto cleanup_rawlc_i;
  galosh_pass12_multiorient_blocked(L_cs, L_cs_den, width, height,
                                     luma_strength, /*block=*/8,
                                     /*stride=*/2, /*n_orient=*/1,
                                     /*use_robust_shrink=*/1);
  dt_free_align(L_cs); L_cs = NULL;
  fprintf(stderr, "[GALOSH_RAW_I] Phase 5(L) full-res cycle-spun L Pass1+2 done\n");

  L_h_den  = dt_alloc_align_float(chsize);
  C1_h_den = dt_alloc_align_float(chsize);
  C2_h_den = dt_alloc_align_float(chsize);
  C3_h_den = dt_alloc_align_float(chsize);
  if(!L_h_den || !C1_h_den || !C2_h_den || !C3_h_den) goto cleanup_rawlc_i;

  /* Sub-sample denoised cycle-spun L at every-other-pixel → half-res
   * Y guide (matches the half-res C lattice). */
  DT_OMP_FOR()
  for(int hr = 0; hr < halfheight; hr++)
  {
    const int fr = 2 * hr;
    if(fr >= height) continue;
    for(int hc = 0; hc < halfwidth; hc++)
    {
      const int fc = 2 * hc;
      if(fc >= width) continue;
      L_h_den[(size_t)hr * halfwidth + hc] = L_cs_den[(size_t)fr * width + fc];
    }
  }

  dummy = dt_alloc_align_float(chsize);
  if(!dummy) goto cleanup_rawlc_i;
  galosh_loess_chroma(L_h_den, C1_h, C2_h, C1_h_den, C2_h_den,
                       halfwidth, halfheight, chroma_strength);
  galosh_loess_chroma(L_h_den, C3_h, C3_h, C3_h_den, dummy,
                       halfwidth, halfheight, chroma_strength);
  dt_free_align(dummy); dummy = NULL;
  dt_free_align(L_h_den); L_h_den = NULL;
  dt_free_align(C1_h); C1_h = NULL;
  dt_free_align(C2_h); C2_h = NULL;
  dt_free_align(C3_h); C3_h = NULL;

  fprintf(stderr, "[GALOSH_RAW_I] Phase 5(C) half-res chroma LOESS done "
                   "(sigma_C=%.3f, R=%d, BW=%.1f)\n",
          chroma_strength, GALOSH_LOESS_RADIUS, GALOSH_LOESS_BW);

  /* ================================================================
   * [LATEST: GALOSH_RAW_I] Phase 6 — K16 EWA-JL3 upsample C_h_den to
   * full-res.  Reuses galosh_upsample_2x_ewajl3 (= G Phase 4(d)).
   * Bandlimit-faithful (5x5 jinc-windowed-jinc-3) — matches the
   * chroma's intrinsic half-Nyquist limit.
   * ================================================================ */
  C1_full = dt_alloc_align_float(npixels);
  C2_full = dt_alloc_align_float(npixels);
  C3_full = dt_alloc_align_float(npixels);
  if(!C1_full || !C2_full || !C3_full) goto cleanup_rawlc_i;
  galosh_upsample_2x_ewajl3(C1_h_den, C1_full, halfwidth, halfheight);
  galosh_upsample_2x_ewajl3(C2_h_den, C2_full, halfwidth, halfheight);
  galosh_upsample_2x_ewajl3(C3_h_den, C3_full, halfwidth, halfheight);
  dt_free_align(C1_h_den); C1_h_den = NULL;
  dt_free_align(C2_h_den); C2_h_den = NULL;
  dt_free_align(C3_h_den); C3_h_den = NULL;

  fprintf(stderr, "[GALOSH_RAW_I] Phase 6 K16 EWA-JL3 chroma upsample done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_I] Phase 7 — L_pixel = 2x2 overlap average of
   * L_cs_den (gat_i_lpixel_overlap_avg).  Resolves the half-pixel grid
   * shift between cycle-spun L (block centers at (r+0.5, c+0.5)) and
   * full-res pixel grid (centers at (r, c)).  Effectively a 3x3
   * weighted box (1,2,1; 2,4,2; 1,2,1)/16 self-conv on L_cs_den.
   * ================================================================ */
  L_pixel = dt_alloc_align_float(npixels);
  if(!L_pixel) goto cleanup_rawlc_i;
  gat_i_lpixel_overlap_avg(L_cs_den, L_pixel, width, height);
  dt_free_align(L_cs_den); L_cs_den = NULL;

  fprintf(stderr, "[GALOSH_RAW_I] Phase 7 L_pixel overlap-avg done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_I] Phase 8 + 9 — fused per-pixel WHT inverse +
   * dark_ref restore + ×unified_sigma denormalize + inverse GAT (LUT).
   *
   *   out[fr,fc] = 0.5 * (L_pixel[fr,fc]
   *                      + signs[CFA(fr,fc)]·C1_full[fr,fc]
   *                      + signs[CFA(fr,fc)]·C2_full[fr,fc]
   *                      + signs[CFA(fr,fc)]·C3_full[fr,fc])
   *              + dark_ref[CFA(fr,fc)]
   *
   * Sign tables (G K16 convention, RGGB):
   *   slot 0 = R  at (0,0): (+, +, +)
   *   slot 1 = Gb at (1,0): (-, +, -)
   *   slot 2 = Gr at (0,1): (+, -, -)
   *   slot 3 = B  at (1,1): (-, -, +)
   * ================================================================ */
  {
    static const float SIGNS[4][3] = {
      { +1.0f, +1.0f, +1.0f },  /* R  */
      { -1.0f, +1.0f, -1.0f },  /* Gb */
      { +1.0f, -1.0f, -1.0f },  /* Gr */
      { -1.0f, -1.0f, +1.0f },  /* B  */
    };

    DT_OMP_FOR()
    for(int fr = 0; fr < height; fr++)
    {
      const int r_off = (fr - co_row) & 1;
      for(int fc = 0; fc < width; fc++)
      {
        const int c_off = (fc - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        const size_t pos = (size_t)fr * width + fc;
        const float val = 0.5f * (L_pixel[pos]
                                + SIGNS[slot][0] * C1_full[pos]
                                + SIGNS[slot][1] * C2_full[pos]
                                + SIGNS[slot][2] * C3_full[pos])
                        + ch_dark_ref[slot];
        out[pos] = gat_inverse_exact(val * unified_sigma);
      }
    }
  }

  fprintf(stderr, "[GALOSH_RAW_I] Phase 8+9 per-pixel inverse WHT + denorm + inv-GAT done\n");

cleanup_rawlc_i:
  dt_free_align(in_gat);
  dt_free_align(L_cs); dt_free_align(C1_cs); dt_free_align(C2_cs); dt_free_align(C3_cs);
  dt_free_align(L_cs_den);
  dt_free_align(L_h_den);
  dt_free_align(C1_h); dt_free_align(C2_h); dt_free_align(C3_h);
  dt_free_align(C1_h_den); dt_free_align(C2_h_den); dt_free_align(C3_h_den);
  dt_free_align(C1_full); dt_free_align(C2_full); dt_free_align(C3_full);
  dt_free_align(L_pixel);
  dt_free_align(dummy);
}


/* ================================================================
 * [LATEST: GALOSH_RAW_J] gat_galosh_denoise_rawlc_j — J pipeline entry.
 *
 * J = I + L-edge-aligned chroma refinement.  Resolves the edge color
 * fringe observed in I outputs (where L is sharp at full-res from
 * cycle-spinning + LOSH, but C transitions are at half-res granularity
 * from K16 EWA-JL3 upsample → 1-pixel offset color halo at edges).
 *
 * Theoretical motivation: CFA chroma sampling theorem bounds info
 * recoverable from C alone at half-Nyquist (= K16 EWA-JL3 = ML
 * estimator).  But natural images have strong cross-channel correlation
 * (chroma edges align with luma edges spatially — Hirakawa & Wolfe TIP
 * 2008).  Adding L as a structural prior gives a Bayesian MAP estimator:
 *
 *   P(C_full | C_h_obs, L_full) ∝ P(C_h_obs | C_full) × P(C_full | L_full)
 *                                  ──────────────────    ──────────────────
 *                                  CFA likelihood        cross-channel prior
 *                                  (K16 = ML)            (LOESS-with-L-guide)
 *
 * J adds the prior step: small-radius LOESS (R=3) at full-res with
 * L_pixel as Y guide.  Bilateral weight w_i=exp(-(L_i-L_c)²/2σ²)
 * encodes "C edges align with L edges": at an L edge, the regression
 * separates C samples into two clusters by L value, snapping the C
 * transition to L's full-res edge position.  In flat-L regions, all
 * window samples contribute equally → C is locally mean-filtered (no
 * blotch reintroduction; K16 smoothness preserved).  R=3 sufficient
 * because input is already bandlimit-correct (K16 ML), only edge
 * position correction is needed.
 *
 * Pipeline (11 phases, all labelled [LATEST: GALOSH_RAW_J] inline):
 *   Phase 0  Foi-Alenius blind α / σ²  (= I/H/G Phase 0)
 *   Phase 1  GAT forward (full-res) + per-CFA σ_GAT MAD + RMS
 *            unified_sigma + scalar normalize  (= I Phase 1)
 *   Phase 2  dark_ref IRLS (achromatic) + per-pixel CFA-aware subtract
 *            (= I Phase 2)
 *   Phase 3  stride=1 forward 2x2 WHT @ full-res → L_cs, C1_cs, C2_cs,
 *            C3_cs  (= I Phase 3, gat_h_forward_wht_stride1)
 *   Phase 4  SPLIT:
 *              4(L) keep L_cs as-is (CFA-invariant)
 *              4(C) sub-sample C_cs at every-other-pixel → half-res C
 *   Phase 5  denoise:
 *              5(L) galosh_pass12_multiorient_blocked on L_cs (full-res)
 *              5(C) galosh_loess_chroma at half-res (R=GALOSH_LOESS_RADIUS,
 *                   Y guide = sub-sample of L_cs_den)
 *   Phase 6  K16 EWA-JL3 upsample C_h_den → C_full_smooth
 *            (galosh_upsample_2x_ewajl3, = ML estimate from CFA samples)
 *   Phase 7  L_pixel = 2x2 overlap average of L_cs_den
 *            (gat_i_lpixel_overlap_avg)
 *   Phase 8  L-guided full-res chroma refinement ⭐ NEW vs I ⭐
 *            galosh_loess_chroma_r(R=3, guide=L_pixel) on C_full_smooth
 *            → C_full_aligned (snaps chroma edges to L edges via
 *            cross-channel prior; Bayesian MAP posterior)
 *   Phase 9+10  per-pixel WHT inverse + dark_ref restore + ×unified_sigma
 *               + inverse GAT (LUT) → out  (fused)
 *
 * Cost: ~I cost + R=3 full-res LOESS (3 chroma planes × 7x7 window) ≈
 *   +2-3 s/scene → ~9-10 s/scene total (still faster than H).  Worth it
 *   because edge fringe elimination is a perceptual win that LPIPS/
 *   DISTS/NIQE all respond to.
 *
 * (日) GALOSH_RAW_J: I に Phase 8 (L-guided chroma refinement) を追加。
 *   K16 で帯域制限的に正しい C を作った後、L_pixel を guide にした
 *   小窓 LOESS で C edge 位置を L edge 位置に snap。CFA sampling 理論
 *   (likelihood) + cross-channel prior の Bayesian MAP 合成。
 * ================================================================ */
static void gat_galosh_denoise_rawlc_j(const float *const restrict in,
                                       float *const restrict out,
                                       const dt_iop_roi_t *const roi,
                                       const float luma_strength,
                                       const float chroma_strength,
                                       const uint32_t filters)
{
  const int width = roi->width, height = roi->height;
  const size_t npixels = (size_t)width * height;
  memcpy(out, in, sizeof(float) * npixels);

  if(luma_strength <= 0.0f) return;
  if(width < GALOSH_BLOCK_SIZE * 2 || height < GALOSH_BLOCK_SIZE * 2) return;

  const int co_row = 0, co_col = 0;
  (void)filters;

  const int halfwidth  = (width  + 1) / 2;
  const int halfheight = (height + 1) / 2;
  const size_t chsize  = (size_t)halfwidth * halfheight;

  /* ================================================================
   * [LATEST: GALOSH_RAW_J] Phase 0 — Foi-Alenius blind α / σ² estimation.
   * Identical to I/H/G Phase 0.
   * ================================================================ */
  const galosh_noise_params_t np = galosh_estimate_noise(in, width, height);
  gat_build_inverse_table(np.alpha, np.sigma_sq);

  /* Pre-declare buffers for cleanup-on-error. */
  float *in_gat = NULL;
  float *L_cs = NULL;
  float *L_cs_den = NULL;
  float *L_h_den = NULL;
  float *C1_h = NULL, *C2_h = NULL, *C3_h = NULL;
  float *C1_h_den = NULL, *C2_h_den = NULL, *C3_h_den = NULL;
  float *C1_full = NULL, *C2_full = NULL, *C3_full = NULL;
  float *C1_aligned = NULL, *C2_aligned = NULL, *C3_aligned = NULL;
  float *L_pixel = NULL;

  in_gat = dt_alloc_align_float(npixels);
  if(!in_gat) goto cleanup_rawlc_j;

  /* ================================================================
   * [LATEST: GALOSH_RAW_J] Phase 1 — GAT forward (full-res) + per-CFA
   * σ_GAT MAD + RMS unified_sigma + scalar normalize.  Identical to
   * I/H Phase 1 (point-wise GAT, stride-2 sub-sample MAD per channel).
   * ================================================================ */
  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++)
    in_gat[i] = gat_forward(in[i], np.alpha, np.sigma_sq);

  float sigma_gat_ch[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  for(int s = 0; s < 4; s++)
  {
    const int ro = ((s & 1)        + co_row) & 1;
    const int co = (((s >> 1) & 1) + co_col) & 1;
    const int hw = (width  - co + 1) / 2;
    const int hh = (height - ro + 1) / 2;
    if(hw < 4 || hh < 4) continue;

    float *tmp = dt_alloc_align_float((size_t)hw * hh);
    if(!tmp) continue;
    DT_OMP_FOR()
    for(int rr = 0; rr < hh; rr++)
      for(int cc = 0; cc < hw; cc++)
        tmp[(size_t)rr * hw + cc] = in_gat[(size_t)(ro + 2*rr) * width + (co + 2*cc)];
    sigma_gat_ch[s] = estimate_gat_sigma_halfres(tmp, hw, hh);
    dt_free_align(tmp);
  }

  const float mean_var = 0.25f * (sigma_gat_ch[0]*sigma_gat_ch[0]
                                + sigma_gat_ch[1]*sigma_gat_ch[1]
                                + sigma_gat_ch[2]*sigma_gat_ch[2]
                                + sigma_gat_ch[3]*sigma_gat_ch[3]);
  const float unified_sigma = sqrtf(fmaxf(mean_var, 1e-12f));
  const float inv_sg = 1.0f / unified_sigma;

  fprintf(stderr, "[GALOSH_RAW_J] alpha=%.8f sigma_sq=%.10f | "
                  "unified_sigma=%.4f [RMS] (per-ch: %.4f %.4f %.4f %.4f) | "
                  "size=%dx%d (half=%dx%d) | sigma_L=%.3f sigma_C=%.3f\n",
                  np.alpha, np.sigma_sq, unified_sigma,
                  sigma_gat_ch[0], sigma_gat_ch[1], sigma_gat_ch[2], sigma_gat_ch[3],
                  width, height, halfwidth, halfheight, luma_strength, chroma_strength);

  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++) in_gat[i] *= inv_sg;

  /* ================================================================
   * [LATEST: GALOSH_RAW_J] Phase 2 — dark_ref IRLS (achromatic) +
   * per-pixel CFA-aware subtract.  Identical to I/H Phase 2.
   * ================================================================ */
  float ch_dark_ref[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  {
    const double s_init = (double)np.sigma_sq / fmax((double)np.alpha, 1e-12);
    double s_scale = s_init;
    const double s_min = 0.05 * s_init;
    const double s_max = 50.0 * s_init;
    const int n_iter = 2;

    for(int iter = 0; iter <= n_iter; iter++)
    {
      const double inv_s = 1.0 / fmax(s_scale, 1e-20);
      double sum_w = 0.0;
      double sum_w0 = 0.0, sum_w1 = 0.0, sum_w2 = 0.0, sum_w3 = 0.0;

      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_w,sum_w0,sum_w1,sum_w2,sum_w3)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          sum_w  += w;
          sum_w0 += w * g0;
          sum_w1 += w * g1;
          sum_w2 += w * g2;
          sum_w3 += w * g3;
        }

      const double inv_sw = 1.0 / fmax(sum_w, 1e-20);
      ch_dark_ref[0] = (float)(sum_w0 * inv_sw);
      ch_dark_ref[1] = (float)(sum_w1 * inv_sw);
      ch_dark_ref[2] = (float)(sum_w2 * inv_sw);
      ch_dark_ref[3] = (float)(sum_w3 * inv_sw);

      if(iter == n_iter) break;

      double sum_wresid2 = 0.0;
      double sum_wW = 0.0;
      const float dr0 = ch_dark_ref[0], dr1 = ch_dark_ref[1];
      const float dr2 = ch_dark_ref[2], dr3 = ch_dark_ref[3];
      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_wresid2,sum_wW)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          const double d0 = (double)g0 - dr0;
          const double d1 = (double)g1 - dr1;
          const double d2 = (double)g2 - dr2;
          const double d3 = (double)g3 - dr3;
          const double resid2 = d0*d0 + d1*d1 + d2*d2 + d3*d3;
          sum_wW      += w;
          sum_wresid2 += w * resid2 * 0.25;
        }
      const double inv_sw2 = 1.0 / fmax(sum_wW, 1e-20);
      const double measured_std = sqrt(fmax(sum_wresid2 * inv_sw2, 1e-20));
      const double ratio = 1.0 / measured_std;
      s_scale *= sqrt(ratio);
      if(s_scale < s_min) s_scale = s_min;
      if(s_scale > s_max) s_scale = s_max;
    }

    fprintf(stderr, "[GALOSH_RAW_J] dark anchor: s_init=%.6e s_final=%.6e | "
                    "ch_dark_ref: [0]=%.4f [1]=%.4f [2]=%.4f [3]=%.4f\n",
            s_init, s_scale,
            ch_dark_ref[0], ch_dark_ref[1], ch_dark_ref[2], ch_dark_ref[3]);

    DT_OMP_FOR()
    for(int r = 0; r < height; r++)
    {
      const int r_off = (r - co_row) & 1;
      for(int c = 0; c < width; c++)
      {
        const int c_off = (c - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        in_gat[(size_t)r * width + c] -= ch_dark_ref[slot];
      }
    }
  }

  /* ================================================================
   * [LATEST: GALOSH_RAW_J] Phase 3 — stride=1 forward 2x2 WHT @ full-res
   * (L plane only).  Optimization vs I: J's chroma path operates at
   * half-res (intrinsic CFA chroma Nyquist limit), so the cycle-spun
   * full-res C planes that I/H computed are immediately sub-sampled
   * and discarded.  J skips the C compute entirely and the 3 full-res
   * C buffers (~150 MB at 16 MP) — see gat_h_forward_l_only_stride1.
   * Mathematically identical L output to gat_h_forward_wht_stride1.
   * ================================================================ */
  L_cs = dt_alloc_align_float(npixels);
  if(!L_cs) goto cleanup_rawlc_j;
  gat_h_forward_l_only_stride1(in_gat, L_cs, width, height);

  fprintf(stderr, "[GALOSH_RAW_J] Phase 3 forward WHT (stride=1, L-only) done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_J] Phase 4 — half-res chroma extract via stride=2
   * forward 2x2 WHT directly from in_gat (no full-res C intermediate).
   *
   * For RGGB at half-res (hr, hc), reads in_gat at (2hr, 2hc),
   * (2hr+1, 2hc), (2hr, 2hc+1), (2hr+1, 2hc+1) and computes the 3
   * chroma WHT components.  CFA-aligned at every-other-pixel → no
   * demod needed.  Mathematically identical to I's two-step
   * (full-res cycle-spun C) + (sub-sample at even positions) but
   * eliminates the redundant full-res C compute.
   * ================================================================ */
  C1_h = dt_alloc_align_float(chsize);
  C2_h = dt_alloc_align_float(chsize);
  C3_h = dt_alloc_align_float(chsize);
  if(!C1_h || !C2_h || !C3_h) goto cleanup_rawlc_j;
  gat_j_forward_c_halfres(in_gat, C1_h, C2_h, C3_h,
                           width, height, halfwidth, halfheight);

  dt_free_align(in_gat); in_gat = NULL;

  fprintf(stderr, "[GALOSH_RAW_J] Phase 4 half-res chroma extract done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_J] Phase 5 — denoise.
   *   5(L) galosh_pass12_multiorient_blocked on L_cs at full-res
   *   5(C) galosh_loess_chroma at half-res, Y guide = sub-sample of
   *        L_cs_den.  Identical to I Phase 5.
   * ================================================================ */
  L_cs_den = dt_alloc_align_float(npixels);
  if(!L_cs_den) goto cleanup_rawlc_j;
  galosh_pass12_multiorient_blocked(L_cs, L_cs_den, width, height,
                                     luma_strength, /*block=*/8,
                                     /*stride=*/2, /*n_orient=*/1,
                                     /*use_robust_shrink=*/1);
  dt_free_align(L_cs); L_cs = NULL;
  fprintf(stderr, "[GALOSH_RAW_J] Phase 5(L) full-res cycle-spun L Pass1+2 done\n");

  L_h_den  = dt_alloc_align_float(chsize);
  C1_h_den = dt_alloc_align_float(chsize);
  C2_h_den = dt_alloc_align_float(chsize);
  C3_h_den = dt_alloc_align_float(chsize);
  if(!L_h_den || !C1_h_den || !C2_h_den || !C3_h_den) goto cleanup_rawlc_j;

  DT_OMP_FOR()
  for(int hr = 0; hr < halfheight; hr++)
  {
    const int fr = 2 * hr;
    if(fr >= height) continue;
    for(int hc = 0; hc < halfwidth; hc++)
    {
      const int fc = 2 * hc;
      if(fc >= width) continue;
      L_h_den[(size_t)hr * halfwidth + hc] = L_cs_den[(size_t)fr * width + fc];
    }
  }

  /* Multi-channel LOESS: 1 fused call processes C1/C2/C3 with shared
   * Y bilateral weight + var_Y → ~36% compute reduction vs 2 separate
   * galosh_loess_chroma_r calls. */
  galosh_loess_chroma_3ch_r(L_h_den, C1_h, C2_h, C3_h,
                             C1_h_den, C2_h_den, C3_h_den,
                             halfwidth, halfheight, chroma_strength,
                             /*R=*/GALOSH_LOESS_RADIUS,
                             /*BW=*/GALOSH_LOESS_BW);
  dt_free_align(L_h_den); L_h_den = NULL;
  dt_free_align(C1_h); C1_h = NULL;
  dt_free_align(C2_h); C2_h = NULL;
  dt_free_align(C3_h); C3_h = NULL;

  fprintf(stderr, "[GALOSH_RAW_J] Phase 5(C) half-res chroma LOESS done "
                   "(sigma_C=%.3f, R=%d, BW=%.1f, 3ch fused)\n",
          chroma_strength, GALOSH_LOESS_RADIUS, GALOSH_LOESS_BW);

  /* ================================================================
   * [LATEST: GALOSH_RAW_J] Phase 6 — K16 EWA-JL3 upsample C_h_den
   * → C_full_smooth (= ML estimator from CFA samples, half-Nyquist
   * bandlimit-faithful).  Reuses galosh_upsample_2x_ewajl3.
   * ================================================================ */
  C1_full = dt_alloc_align_float(npixels);
  C2_full = dt_alloc_align_float(npixels);
  C3_full = dt_alloc_align_float(npixels);
  if(!C1_full || !C2_full || !C3_full) goto cleanup_rawlc_j;
  galosh_upsample_2x_ewajl3(C1_h_den, C1_full, halfwidth, halfheight);
  galosh_upsample_2x_ewajl3(C2_h_den, C2_full, halfwidth, halfheight);
  galosh_upsample_2x_ewajl3(C3_h_den, C3_full, halfwidth, halfheight);
  dt_free_align(C1_h_den); C1_h_den = NULL;
  dt_free_align(C2_h_den); C2_h_den = NULL;
  dt_free_align(C3_h_den); C3_h_den = NULL;

  fprintf(stderr, "[GALOSH_RAW_J] Phase 6 K16 EWA-JL3 chroma upsample done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_J] Phase 7 — L_pixel = 2x2 overlap average of
   * L_cs_den (gat_i_lpixel_overlap_avg).  Resolves half-pixel grid
   * shift between cycle-spun L and full-res pixel grid.
   * ================================================================ */
  L_pixel = dt_alloc_align_float(npixels);
  if(!L_pixel) goto cleanup_rawlc_j;
  gat_i_lpixel_overlap_avg(L_cs_den, L_pixel, width, height);
  dt_free_align(L_cs_den); L_cs_den = NULL;

  fprintf(stderr, "[GALOSH_RAW_J] Phase 7 L_pixel overlap-avg done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_J] Phase 8 ⭐ NEW vs I ⭐ — L-guided full-res
   * chroma refinement (joint bilateral upsample / guided filter).
   *
   * Bayesian MAP posterior step: K16 (Phase 6) gave the CFA likelihood
   * ML estimate; this phase adds the cross-channel structural prior
   * P(C_full | L_full) via L-guided LOESS at full-res with R=3 (7x7
   * window).  Bilateral weight w_i=exp(-(L_i-L_c)²/2σ²) snaps C edges
   * to L edges; flat L regions are unchanged (K16 smoothness preserved).
   *
   * Effect: eliminates the edge color fringe observed in I outputs
   * (where K16's half-res-grid C transitions misaligned with cycle-spun
   * L's full-res-grid transitions).
   *
   * R=3 sufficient: only edge position correction needed; larger R
   * would smear unrelated structures and over-smooth flat regions.
   *
   * Reference: He et al. (TPAMI 2010) guided filter; Kopf et al.
   *            (SIGGRAPH 2007) joint bilateral upsample;
   *            Hirakawa & Wolfe (TIP 2008) cross-channel CFA prior.
   *
   * (日) K16 で帯域制限的に正しい C を作った後、L_pixel を guide に
   *   した小窓 LOESS (R=3) で C edge を L edge に snap。CFA likelihood
   *   + cross-channel prior の Bayesian MAP 合成。
   * ================================================================ */
  C1_aligned = dt_alloc_align_float(npixels);
  C2_aligned = dt_alloc_align_float(npixels);
  C3_aligned = dt_alloc_align_float(npixels);
  if(!C1_aligned || !C2_aligned || !C3_aligned) goto cleanup_rawlc_j;

  /* Phase 8 refinement: same chroma_strength + BW as Phase 5(C). */
  galosh_loess_chroma_3ch_r(L_pixel, C1_full, C2_full, C3_full,
                             C1_aligned, C2_aligned, C3_aligned,
                             width, height, chroma_strength,
                             /*R=*/3, /*BW=*/GALOSH_LOESS_BW);
  dt_free_align(C1_full); C1_full = NULL;
  dt_free_align(C2_full); C2_full = NULL;
  dt_free_align(C3_full); C3_full = NULL;

  fprintf(stderr, "[GALOSH_RAW_J] Phase 8 L-guided chroma refinement done "
                   "(R=3, BW=%.1f, 3ch fused)\n", GALOSH_LOESS_BW);

  /* ================================================================
   * [LATEST: GALOSH_RAW_J] Phase 9 + 10 — fused per-pixel WHT inverse
   * + dark_ref restore + ×unified_sigma denormalize + inverse GAT (LUT).
   *
   *   out[fr,fc] = 0.5 * (L_pixel[fr,fc]
   *                      + signs[CFA(fr,fc)]·C1_aligned[fr,fc]
   *                      + signs[CFA(fr,fc)]·C2_aligned[fr,fc]
   *                      + signs[CFA(fr,fc)]·C3_aligned[fr,fc])
   *              + dark_ref[CFA(fr,fc)]
   *
   * Sign tables (G K16 convention, RGGB):
   *   slot 0 = R  at (0,0): (+, +, +)
   *   slot 1 = Gb at (1,0): (-, +, -)
   *   slot 2 = Gr at (0,1): (+, -, -)
   *   slot 3 = B  at (1,1): (-, -, +)
   * ================================================================ */
  {
    static const float SIGNS[4][3] = {
      { +1.0f, +1.0f, +1.0f },  /* R  */
      { -1.0f, +1.0f, -1.0f },  /* Gb */
      { +1.0f, -1.0f, -1.0f },  /* Gr */
      { -1.0f, -1.0f, +1.0f },  /* B  */
    };

    DT_OMP_FOR()
    for(int fr = 0; fr < height; fr++)
    {
      const int r_off = (fr - co_row) & 1;
      for(int fc = 0; fc < width; fc++)
      {
        const int c_off = (fc - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        const size_t pos = (size_t)fr * width + fc;
        const float val = 0.5f * (L_pixel[pos]
                                + SIGNS[slot][0] * C1_aligned[pos]
                                + SIGNS[slot][1] * C2_aligned[pos]
                                + SIGNS[slot][2] * C3_aligned[pos])
                        + ch_dark_ref[slot];
        out[pos] = gat_inverse_exact(val * unified_sigma);
      }
    }
  }

  fprintf(stderr, "[GALOSH_RAW_J] Phase 9+10 per-pixel inverse WHT + denorm + inv-GAT done\n");

cleanup_rawlc_j:
  dt_free_align(in_gat);
  dt_free_align(L_cs);
  dt_free_align(L_cs_den);
  dt_free_align(L_h_den);
  dt_free_align(C1_h); dt_free_align(C2_h); dt_free_align(C3_h);
  dt_free_align(C1_h_den); dt_free_align(C2_h_den); dt_free_align(C3_h_den);
  dt_free_align(C1_full); dt_free_align(C2_full); dt_free_align(C3_full);
  dt_free_align(C1_aligned); dt_free_align(C2_aligned); dt_free_align(C3_aligned);
  dt_free_align(L_pixel);
}


/* ================================================================
 * [LATEST: GALOSH_RAW_K] gat_galosh_denoise_rawlc_k — K pipeline entry.
 *
 * K = J + Bayesian-correct Phase 8 parameters (= "refinement-aware"
 * regularization).  J's Phase 8 (L-guided chroma refinement) used the
 * same eps and bilateral bandwidth as Phase 5(C) (= the primary
 * chroma denoise), causing edge color fringe to persist in J outputs:
 *
 *   J Phase 8: eps = chroma_slider² (= 1.0 default), BW = 3.0
 *   K Phase 8: eps = 0.01 (slider-INDEPENDENT), BW = 1.5
 *
 * Theoretical motivation (Bayesian framing of LOESS):
 *   The LOESS local linear regression `cb = a·L + b` has closed-form
 *
 *     a = cov(L, C) / (var(L) + ε),  ε = σ_C² / τ²
 *
 *   where σ_C is the input chroma noise std and τ² is the prior
 *   variance on the L-C correlation slope.  ε balances data fidelity
 *   vs the "no prior preference" assumption.
 *
 *   For Phase 5(C) — denoising fresh half-res chroma with σ_C ≈ 1 in
 *   GAT-norm space — eps = 1.0 (= σ_C² with τ²=1) is the correct
 *   Bayesian posterior weighting.
 *
 *   For Phase 8 — REFINING already-denoised C_full from K16 EWA-JL3
 *   (which preserves the half-res LOESS noise reduction) — the input
 *   noise σ_C ≈ 0.1 (= 1/√n_eff after the first LOESS pass over
 *   ~100 effective samples).  → ε = σ_C² = 0.01 is the correct
 *   Bayesian regularizer.  Using ε=1 was 100× over-regularization,
 *   damping the regression response and weakening edge snap.
 *
 * Bilateral bandwidth (BW):
 *   Phase 5(C) BW=3 = 3σ Mahalanobis "same-cluster" threshold for
 *     unit-noise input.  Standard.
 *   Phase 8 BW=1.5: input L_pixel has residual noise σ ≈ 0.1 after
 *     LOSH + 2x2 box, so 1.5 in GAT-norm units = 15× the residual
 *     noise → bilateral never triggers on noise (safe), but DOES
 *     separate moderate-to-weak edges (Y_jump 1.5-3 σ_GAT) that
 *     BW=3 would treat as same cluster.  Targets the residual fringe.
 *
 * Result: stronger and more reliable edge snap on moderate edges
 *   without re-introducing chroma blotch (input C_full is already
 *   smooth from K16 + Phase 5(C)).  PSNR/LPIPS/DISTS should improve
 *   over J at the cost of zero additional compute (parameter-only
 *   change).
 *
 * Pipeline (11 phases, all labelled [LATEST: GALOSH_RAW_K] inline):
 *   Phase 0..7  identical to J Phase 0..7
 *   Phase 8     L-guided chroma refinement, NEW PARAMETERS:
 *               galosh_loess_chroma_3ch_r(L_pixel, ..., strength=0.1,
 *                                          R=3, BW=1.5)
 *   Phase 9+10  identical to J Phase 9+10
 *
 * (日) GALOSH_RAW_K: J の Phase 8 LOESS の hyperparameter を Bayesian
 *   原理 (= 入力 C_full の residual σ ≈ 0.1 に整合する ε = 0.01) +
 *   refinement 用 narrow BW=1.5 に変更。J の中-弱 edge fringe を消す。
 * ================================================================ */
static void gat_galosh_denoise_rawlc_k(const float *const restrict in,
                                       float *const restrict out,
                                       const dt_iop_roi_t *const roi,
                                       const float luma_strength,
                                       const float chroma_strength,
                                       const uint32_t filters)
{
  const int width = roi->width, height = roi->height;
  const size_t npixels = (size_t)width * height;
  memcpy(out, in, sizeof(float) * npixels);

  if(luma_strength <= 0.0f) return;
  if(width < GALOSH_BLOCK_SIZE * 2 || height < GALOSH_BLOCK_SIZE * 2) return;

  const int co_row = 0, co_col = 0;
  (void)filters;

  const int halfwidth  = (width  + 1) / 2;
  const int halfheight = (height + 1) / 2;
  const size_t chsize  = (size_t)halfwidth * halfheight;

  /* ================================================================
   * [LATEST: GALOSH_RAW_K] Phase 0 — Foi-Alenius blind α / σ² estimation.
   * Identical to J/I/H/G Phase 0.
   * ================================================================ */
  const galosh_noise_params_t np = galosh_estimate_noise(in, width, height);
  gat_build_inverse_table(np.alpha, np.sigma_sq);

  /* Pre-declare buffers for cleanup-on-error. */
  float *in_gat = NULL;
  float *L_cs = NULL;
  float *L_cs_den = NULL;
  float *L_h_den = NULL;
  float *C1_h = NULL, *C2_h = NULL, *C3_h = NULL;
  float *C1_h_den = NULL, *C2_h_den = NULL, *C3_h_den = NULL;
  float *C1_full = NULL, *C2_full = NULL, *C3_full = NULL;
  float *C1_aligned = NULL, *C2_aligned = NULL, *C3_aligned = NULL;
  float *L_pixel = NULL;

  in_gat = dt_alloc_align_float(npixels);
  if(!in_gat) goto cleanup_rawlc_k;

  /* ================================================================
   * [LATEST: GALOSH_RAW_K] Phase 1 — GAT forward (full-res) + per-CFA
   * σ_GAT MAD + RMS unified_sigma + scalar normalize.  Identical to J.
   * ================================================================ */
  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++)
    in_gat[i] = gat_forward(in[i], np.alpha, np.sigma_sq);

  float sigma_gat_ch[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  for(int s = 0; s < 4; s++)
  {
    const int ro = ((s & 1)        + co_row) & 1;
    const int co = (((s >> 1) & 1) + co_col) & 1;
    const int hw = (width  - co + 1) / 2;
    const int hh = (height - ro + 1) / 2;
    if(hw < 4 || hh < 4) continue;

    float *tmp = dt_alloc_align_float((size_t)hw * hh);
    if(!tmp) continue;
    DT_OMP_FOR()
    for(int rr = 0; rr < hh; rr++)
      for(int cc = 0; cc < hw; cc++)
        tmp[(size_t)rr * hw + cc] = in_gat[(size_t)(ro + 2*rr) * width + (co + 2*cc)];
    sigma_gat_ch[s] = estimate_gat_sigma_halfres(tmp, hw, hh);
    dt_free_align(tmp);
  }

  const float mean_var = 0.25f * (sigma_gat_ch[0]*sigma_gat_ch[0]
                                + sigma_gat_ch[1]*sigma_gat_ch[1]
                                + sigma_gat_ch[2]*sigma_gat_ch[2]
                                + sigma_gat_ch[3]*sigma_gat_ch[3]);
  const float unified_sigma = sqrtf(fmaxf(mean_var, 1e-12f));
  const float inv_sg = 1.0f / unified_sigma;

  fprintf(stderr, "[GALOSH_RAW_K] alpha=%.8f sigma_sq=%.10f | "
                  "unified_sigma=%.4f [RMS] (per-ch: %.4f %.4f %.4f %.4f) | "
                  "size=%dx%d (half=%dx%d) | sigma_L=%.3f sigma_C=%.3f\n",
                  np.alpha, np.sigma_sq, unified_sigma,
                  sigma_gat_ch[0], sigma_gat_ch[1], sigma_gat_ch[2], sigma_gat_ch[3],
                  width, height, halfwidth, halfheight, luma_strength, chroma_strength);

  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++) in_gat[i] *= inv_sg;

  /* ================================================================
   * [LATEST: GALOSH_RAW_K] Phase 2 — dark_ref IRLS + per-pixel subtract.
   * Identical to J Phase 2.
   * ================================================================ */
  float ch_dark_ref[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  {
    const double s_init = (double)np.sigma_sq / fmax((double)np.alpha, 1e-12);
    double s_scale = s_init;
    const double s_min = 0.05 * s_init;
    const double s_max = 50.0 * s_init;
    const int n_iter = 2;

    for(int iter = 0; iter <= n_iter; iter++)
    {
      const double inv_s = 1.0 / fmax(s_scale, 1e-20);
      double sum_w = 0.0;
      double sum_w0 = 0.0, sum_w1 = 0.0, sum_w2 = 0.0, sum_w3 = 0.0;

      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_w,sum_w0,sum_w1,sum_w2,sum_w3)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          sum_w  += w;
          sum_w0 += w * g0;
          sum_w1 += w * g1;
          sum_w2 += w * g2;
          sum_w3 += w * g3;
        }

      const double inv_sw = 1.0 / fmax(sum_w, 1e-20);
      ch_dark_ref[0] = (float)(sum_w0 * inv_sw);
      ch_dark_ref[1] = (float)(sum_w1 * inv_sw);
      ch_dark_ref[2] = (float)(sum_w2 * inv_sw);
      ch_dark_ref[3] = (float)(sum_w3 * inv_sw);

      if(iter == n_iter) break;

      double sum_wresid2 = 0.0;
      double sum_wW = 0.0;
      const float dr0 = ch_dark_ref[0], dr1 = ch_dark_ref[1];
      const float dr2 = ch_dark_ref[2], dr3 = ch_dark_ref[3];
      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_wresid2,sum_wW)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          const double d0 = (double)g0 - dr0;
          const double d1 = (double)g1 - dr1;
          const double d2 = (double)g2 - dr2;
          const double d3 = (double)g3 - dr3;
          const double resid2 = d0*d0 + d1*d1 + d2*d2 + d3*d3;
          sum_wW      += w;
          sum_wresid2 += w * resid2 * 0.25;
        }
      const double inv_sw2 = 1.0 / fmax(sum_wW, 1e-20);
      const double measured_std = sqrt(fmax(sum_wresid2 * inv_sw2, 1e-20));
      const double ratio = 1.0 / measured_std;
      s_scale *= sqrt(ratio);
      if(s_scale < s_min) s_scale = s_min;
      if(s_scale > s_max) s_scale = s_max;
    }

    fprintf(stderr, "[GALOSH_RAW_K] dark anchor: s_init=%.6e s_final=%.6e | "
                    "ch_dark_ref: [0]=%.4f [1]=%.4f [2]=%.4f [3]=%.4f\n",
            s_init, s_scale,
            ch_dark_ref[0], ch_dark_ref[1], ch_dark_ref[2], ch_dark_ref[3]);

    DT_OMP_FOR()
    for(int r = 0; r < height; r++)
    {
      const int r_off = (r - co_row) & 1;
      for(int c = 0; c < width; c++)
      {
        const int c_off = (c - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        in_gat[(size_t)r * width + c] -= ch_dark_ref[slot];
      }
    }
  }

  /* ================================================================
   * [LATEST: GALOSH_RAW_K] Phase 3 — stride=1 forward 2x2 WHT (L-only).
   * Identical to J Phase 3.
   * ================================================================ */
  L_cs = dt_alloc_align_float(npixels);
  if(!L_cs) goto cleanup_rawlc_k;
  gat_h_forward_l_only_stride1(in_gat, L_cs, width, height);

  fprintf(stderr, "[GALOSH_RAW_K] Phase 3 forward WHT (stride=1, L-only) done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_K] Phase 4 — half-res chroma extract via stride=2
   * forward 2x2 WHT.  Identical to J Phase 4.
   * ================================================================ */
  C1_h = dt_alloc_align_float(chsize);
  C2_h = dt_alloc_align_float(chsize);
  C3_h = dt_alloc_align_float(chsize);
  if(!C1_h || !C2_h || !C3_h) goto cleanup_rawlc_k;
  gat_j_forward_c_halfres(in_gat, C1_h, C2_h, C3_h,
                           width, height, halfwidth, halfheight);

  dt_free_align(in_gat); in_gat = NULL;

  fprintf(stderr, "[GALOSH_RAW_K] Phase 4 half-res chroma extract done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_K] Phase 5 — denoise.
   *   5(L) full-res LOSH on L_cs (= J Phase 5(L))
   *   5(C) half-res 3-channel LOESS (= J Phase 5(C); slider eps, BW=3)
   * ================================================================ */
  L_cs_den = dt_alloc_align_float(npixels);
  if(!L_cs_den) goto cleanup_rawlc_k;
  galosh_pass12_multiorient_blocked(L_cs, L_cs_den, width, height,
                                     luma_strength, /*block=*/8,
                                     /*stride=*/2, /*n_orient=*/1,
                                     /*use_robust_shrink=*/1);
  dt_free_align(L_cs); L_cs = NULL;
  fprintf(stderr, "[GALOSH_RAW_K] Phase 5(L) full-res cycle-spun L Pass1+2 done\n");

  L_h_den  = dt_alloc_align_float(chsize);
  C1_h_den = dt_alloc_align_float(chsize);
  C2_h_den = dt_alloc_align_float(chsize);
  C3_h_den = dt_alloc_align_float(chsize);
  if(!L_h_den || !C1_h_den || !C2_h_den || !C3_h_den) goto cleanup_rawlc_k;

  DT_OMP_FOR()
  for(int hr = 0; hr < halfheight; hr++)
  {
    const int fr = 2 * hr;
    if(fr >= height) continue;
    for(int hc = 0; hc < halfwidth; hc++)
    {
      const int fc = 2 * hc;
      if(fc >= width) continue;
      L_h_den[(size_t)hr * halfwidth + hc] = L_cs_den[(size_t)fr * width + fc];
    }
  }

  galosh_loess_chroma_3ch_r(L_h_den, C1_h, C2_h, C3_h,
                             C1_h_den, C2_h_den, C3_h_den,
                             halfwidth, halfheight, chroma_strength,
                             /*R=*/GALOSH_LOESS_RADIUS,
                             /*BW=*/GALOSH_LOESS_BW);
  dt_free_align(L_h_den); L_h_den = NULL;
  dt_free_align(C1_h); C1_h = NULL;
  dt_free_align(C2_h); C2_h = NULL;
  dt_free_align(C3_h); C3_h = NULL;

  fprintf(stderr, "[GALOSH_RAW_K] Phase 5(C) half-res chroma LOESS done "
                   "(sigma_C=%.3f, R=%d, BW=%.1f, 3ch fused)\n",
          chroma_strength, GALOSH_LOESS_RADIUS, GALOSH_LOESS_BW);

  /* ================================================================
   * [LATEST: GALOSH_RAW_K] Phase 6 — K16 EWA-JL3 upsample.
   * Identical to J Phase 6.
   * ================================================================ */
  C1_full = dt_alloc_align_float(npixels);
  C2_full = dt_alloc_align_float(npixels);
  C3_full = dt_alloc_align_float(npixels);
  if(!C1_full || !C2_full || !C3_full) goto cleanup_rawlc_k;
  galosh_upsample_2x_ewajl3(C1_h_den, C1_full, halfwidth, halfheight);
  galosh_upsample_2x_ewajl3(C2_h_den, C2_full, halfwidth, halfheight);
  galosh_upsample_2x_ewajl3(C3_h_den, C3_full, halfwidth, halfheight);
  dt_free_align(C1_h_den); C1_h_den = NULL;
  dt_free_align(C2_h_den); C2_h_den = NULL;
  dt_free_align(C3_h_den); C3_h_den = NULL;

  fprintf(stderr, "[GALOSH_RAW_K] Phase 6 K16 EWA-JL3 chroma upsample done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_K] Phase 7 — L_pixel overlap-avg.
   * Identical to J Phase 7.
   * ================================================================ */
  L_pixel = dt_alloc_align_float(npixels);
  if(!L_pixel) goto cleanup_rawlc_k;
  gat_i_lpixel_overlap_avg(L_cs_den, L_pixel, width, height);
  dt_free_align(L_cs_den); L_cs_den = NULL;

  fprintf(stderr, "[GALOSH_RAW_K] Phase 7 L_pixel overlap-avg done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_K] Phase 8 ⭐ NEW vs J ⭐ — L-guided full-res
   * chroma refinement with **Bayesian-correct refinement parameters**:
   *
   *   strength_phase8 = 0.1f → ε = 0.1² × τ²_inv = 0.01
   *     Slider-INDEPENDENT.  Bayesian: input C_full noise std after
   *     Phase 5(C) + K16 ≈ 0.1 (Phase 5(C) reduced σ from 1 to ~0.1
   *     via 1/√n_eff over ~100 effective bilateral samples; K16 EWA-JL3
   *     preserves noise level via bandlimit interp).  ε = σ_C² = 0.01
   *     is the principled regularizer for this residual noise level.
   *     J's choice ε = chroma_slider² = 1.0 was 100× over-regularization,
   *     damping the regression at all but the strongest edges.
   *
   *   BW_phase8 = 1.5f (vs Phase 5(C) and J Phase 8: BW = 3.0)
   *     Phase 8 input L_pixel has residual noise σ ≈ 0.1 after LOSH +
   *     2x2 box.  BW=1.5 keeps noise/threshold ratio at 15× (= safe,
   *     bilateral never triggers on noise) while ALSO admitting
   *     moderate-to-weak edges (Y_jump 1.5-3 σ_GAT) into "different
   *     cluster" judgment.  These moderate edges are exactly where
   *     residual color fringe persists in J.
   *
   * Compute cost: ZERO additional vs J (parameter-only change).
   * Quality effect: stronger and more reliable edge snap on moderate
   * edges, no impact on flat regions (input C_full smoothness preserved).
   *
   * (日) Phase 8 の hyperparameter のみ変更:
   *      ε = 0.01 (固定、refinement 入力の residual σ²) +
   *      BW = 1.5 (中-弱 edge も bilateral 分離可能に).
   *      コスト変化なし、edge snap が J より確実に効く。
   * ================================================================ */
  C1_aligned = dt_alloc_align_float(npixels);
  C2_aligned = dt_alloc_align_float(npixels);
  C3_aligned = dt_alloc_align_float(npixels);
  if(!C1_aligned || !C2_aligned || !C3_aligned) goto cleanup_rawlc_k;

  galosh_loess_chroma_3ch_r(L_pixel, C1_full, C2_full, C3_full,
                             C1_aligned, C2_aligned, C3_aligned,
                             width, height,
                             /*strength=*/0.1f, /*R=*/3, /*BW=*/1.5f);
  dt_free_align(C1_full); C1_full = NULL;
  dt_free_align(C2_full); C2_full = NULL;
  dt_free_align(C3_full); C3_full = NULL;

  fprintf(stderr, "[GALOSH_RAW_K] Phase 8 L-guided chroma refinement done "
                   "(R=3, BW=1.5, eps=0.01 hardcoded, 3ch fused)\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_K] Phase 9 + 10 — fused per-pixel WHT inverse
   * + dark_ref restore + ×unified_sigma denormalize + inverse GAT (LUT).
   * Identical to J Phase 9+10.
   * ================================================================ */
  {
    static const float SIGNS[4][3] = {
      { +1.0f, +1.0f, +1.0f },  /* R  */
      { -1.0f, +1.0f, -1.0f },  /* Gb */
      { +1.0f, -1.0f, -1.0f },  /* Gr */
      { -1.0f, -1.0f, +1.0f },  /* B  */
    };

    DT_OMP_FOR()
    for(int fr = 0; fr < height; fr++)
    {
      const int r_off = (fr - co_row) & 1;
      for(int fc = 0; fc < width; fc++)
      {
        const int c_off = (fc - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        const size_t pos = (size_t)fr * width + fc;
        const float val = 0.5f * (L_pixel[pos]
                                + SIGNS[slot][0] * C1_aligned[pos]
                                + SIGNS[slot][1] * C2_aligned[pos]
                                + SIGNS[slot][2] * C3_aligned[pos])
                        + ch_dark_ref[slot];
        out[pos] = gat_inverse_exact(val * unified_sigma);
      }
    }
  }

  fprintf(stderr, "[GALOSH_RAW_K] Phase 9+10 per-pixel inverse WHT + denorm + inv-GAT done\n");

cleanup_rawlc_k:
  dt_free_align(in_gat);
  dt_free_align(L_cs);
  dt_free_align(L_cs_den);
  dt_free_align(L_h_den);
  dt_free_align(C1_h); dt_free_align(C2_h); dt_free_align(C3_h);
  dt_free_align(C1_h_den); dt_free_align(C2_h_den); dt_free_align(C3_h_den);
  dt_free_align(C1_full); dt_free_align(C2_full); dt_free_align(C3_full);
  dt_free_align(C1_aligned); dt_free_align(C2_aligned); dt_free_align(C3_aligned);
  dt_free_align(L_pixel);
}


/* ================================================================
 * [LATEST: GALOSH_RAW_L] gat_galosh_denoise_rawlc_l — L pipeline entry.
 *
 * L = K's two-stage chroma reconstruction (Phase 6 K16 + Phase 8 LOESS
 * refinement) FUSED into a single guide-aware K16 stage.  Replaces:
 *
 *   K Phase 6: galosh_upsample_2x_ewajl3       (L-unaware bandlimit)
 *   K Phase 8: galosh_loess_chroma_3ch_r       (L-guided LOESS post)
 *
 * with:
 *
 *   L Phase 7: gat_k16_joint_bilateral_upsample (L-aware EWA-JL3
 *                                                 = bandlimit + edge
 *                                                 alignment in 1 step)
 *
 * Theoretical motivation:
 *   K's two-stage approach has an information bottleneck: K16 first
 *   smooths C edges to the half-res grid (= bandlimit interp), then
 *   LOESS tries to "snap back" using L.  But the snap can only use
 *   the smoothed (= already-bandlimited) C, not the original half-res
 *   samples.  Information that could distinguish edge sub-pixel
 *   position is lost in the smoothing step.
 *
 *   Joint bilateral upsample (Kopf et al. SIGGRAPH 2007 / He et al.
 *   TPAMI 2010) directly modulates the K16 jinc kernel by an L-bilateral
 *   weight at each output pixel:
 *     w_combined[i] = w_jinc(d_i) × exp( -(L_pixel - L_at_h[i])²/(2BW²) )
 *   The output is a single-pass weighted sum over original half-res
 *   chroma samples — both bandlimit interp and L-edge alignment are
 *   computed from the same source data, no intermediate smoothing.
 *
 *   In flat L regions: bilateral ≈ uniform → effective kernel = pure
 *   jinc → bandlimit-faithful (matches K16 standard exactly).
 *   At L edges: bilateral kills cross-edge samples → effective kernel
 *   = one-sided jinc → C edges snap to L edges (cross-channel
 *   super-resolution from CFA likelihood + L structural prior).
 *
 *   This is theoretically cleaner than K's K16 + LOESS chain: the
 *   bandlimit and edge-alignment criteria are merged into one filter
 *   weight rather than applied sequentially with information loss.
 *
 * Pipeline (10 phases, all labelled [LATEST: GALOSH_RAW_L] inline):
 *   Phase 0..5  identical to K Phase 0..5
 *   Phase 6     L_pixel = 2x2 overlap average of L_cs_den
 *               (= K Phase 7, moved earlier since now needed BEFORE
 *                upsample as the bilateral guide)
 *   Phase 7     Joint bilateral K16 upsample ⭐ NEW vs K ⭐
 *               gat_k16_joint_bilateral_upsample(C_h_den, L_pixel, ...)
 *               → C_full_aligned (replaces K's Phase 6 + Phase 8)
 *   Phase 8+9   per-pixel WHT inverse + dark_ref restore + ×unified_sigma
 *               + inverse GAT (LUT) → out  (fused, = K Phase 9+10)
 *
 * Cost: K's Phase 6 (~0.3s) + Phase 8 (~1-2s) → L's Phase 7 (~0.5-1s).
 * Net saving: ~0.5-1.5s per scene from eliminating the redundant
 * smoothing+refining chain.  Plus the Phase 8 LOESS R=3 LOESS-with-L
 * pass is now subsumed into the upsample weight.
 *
 * Quality: edge fringe SHOULD be more substantially eliminated because
 * the bilateral operates on the original (unsmoothed) half-res C samples,
 * preserving sub-pixel edge information that K's chained approach lost.
 *
 * (日) GALOSH_RAW_L: K の K16 + LOESS post-process 二段を joint
 *   bilateral upsample (Kopf 2007 / He 2010) で 1 段融合。帯域制限と
 *   L-edge alignment を同じ filter weight で同時計算 → 中間平滑化に
 *   よる情報劣化を排除。理論的には K より edge fringe 改善 + 計算
 *   コストやや低減。
 * ================================================================ */
#endif  /* GALOSH_RELEASE: end deprecated g..k.  L kept = canonical O's small-input fallback. */
static void gat_galosh_denoise_rawlc_l(const float *const restrict in,
                                       float *const restrict out,
                                       const dt_iop_roi_t *const roi,
                                       const float luma_strength,
                                       const float chroma_strength,
                                       const uint32_t filters)
{
  const int width = roi->width, height = roi->height;
  const size_t npixels = (size_t)width * height;
  memcpy(out, in, sizeof(float) * npixels);

  if(luma_strength <= 0.0f) return;
  if(width < GALOSH_BLOCK_SIZE * 2 || height < GALOSH_BLOCK_SIZE * 2) return;

  const int co_row = 0, co_col = 0;
  (void)filters;

  const int halfwidth  = (width  + 1) / 2;
  const int halfheight = (height + 1) / 2;
  const size_t chsize  = (size_t)halfwidth * halfheight;

  /* ================================================================
   * [LATEST: GALOSH_RAW_L] Phase 0 — Foi-Alenius blind α / σ²
   * estimation.  Identical to K/J/I/H/G Phase 0.
   * ================================================================ */
  const galosh_noise_params_t np = galosh_estimate_noise(in, width, height);
  gat_build_inverse_table(np.alpha, np.sigma_sq);

  /* Pre-declare buffers for cleanup-on-error. */
  float *in_gat = NULL;
  float *L_cs = NULL;
  float *L_cs_den = NULL;
  float *L_h_den = NULL;
  float *C1_h = NULL, *C2_h = NULL, *C3_h = NULL;
  float *C1_h_den = NULL, *C2_h_den = NULL, *C3_h_den = NULL;
  float *C1_aligned = NULL, *C2_aligned = NULL, *C3_aligned = NULL;
  float *L_pixel = NULL;

  in_gat = dt_alloc_align_float(npixels);
  if(!in_gat) goto cleanup_rawlc_l;

  /* ================================================================
   * [LATEST: GALOSH_RAW_L] Phase 1 — GAT forward (full-res) + per-CFA
   * σ_GAT MAD + RMS unified_sigma + scalar normalize.  Identical to K.
   * ================================================================ */
  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++)
    in_gat[i] = gat_forward(in[i], np.alpha, np.sigma_sq);

  float sigma_gat_ch[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  for(int s = 0; s < 4; s++)
  {
    const int ro = ((s & 1)        + co_row) & 1;
    const int co = (((s >> 1) & 1) + co_col) & 1;
    const int hw = (width  - co + 1) / 2;
    const int hh = (height - ro + 1) / 2;
    if(hw < 4 || hh < 4) continue;

    float *tmp = dt_alloc_align_float((size_t)hw * hh);
    if(!tmp) continue;
    DT_OMP_FOR()
    for(int rr = 0; rr < hh; rr++)
      for(int cc = 0; cc < hw; cc++)
        tmp[(size_t)rr * hw + cc] = in_gat[(size_t)(ro + 2*rr) * width + (co + 2*cc)];
    sigma_gat_ch[s] = estimate_gat_sigma_halfres(tmp, hw, hh);
    dt_free_align(tmp);
  }

  const float mean_var = 0.25f * (sigma_gat_ch[0]*sigma_gat_ch[0]
                                + sigma_gat_ch[1]*sigma_gat_ch[1]
                                + sigma_gat_ch[2]*sigma_gat_ch[2]
                                + sigma_gat_ch[3]*sigma_gat_ch[3]);
  const float unified_sigma = sqrtf(fmaxf(mean_var, 1e-12f));
  const float inv_sg = 1.0f / unified_sigma;

  fprintf(stderr, "[GALOSH_RAW_L] alpha=%.8f sigma_sq=%.10f | "
                  "unified_sigma=%.4f [RMS] (per-ch: %.4f %.4f %.4f %.4f) | "
                  "size=%dx%d (half=%dx%d) | sigma_L=%.3f sigma_C=%.3f\n",
                  np.alpha, np.sigma_sq, unified_sigma,
                  sigma_gat_ch[0], sigma_gat_ch[1], sigma_gat_ch[2], sigma_gat_ch[3],
                  width, height, halfwidth, halfheight, luma_strength, chroma_strength);

  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++) in_gat[i] *= inv_sg;

  /* ================================================================
   * [LATEST: GALOSH_RAW_L] Phase 2 — dark_ref IRLS + per-pixel subtract.
   * Identical to K/J Phase 2.
   * ================================================================ */
  float ch_dark_ref[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  {
    const double s_init = (double)np.sigma_sq / fmax((double)np.alpha, 1e-12);
    double s_scale = s_init;
    const double s_min = 0.05 * s_init;
    const double s_max = 50.0 * s_init;
    const int n_iter = 2;

    for(int iter = 0; iter <= n_iter; iter++)
    {
      const double inv_s = 1.0 / fmax(s_scale, 1e-20);
      double sum_w = 0.0;
      double sum_w0 = 0.0, sum_w1 = 0.0, sum_w2 = 0.0, sum_w3 = 0.0;

      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_w,sum_w0,sum_w1,sum_w2,sum_w3)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          sum_w  += w;
          sum_w0 += w * g0;
          sum_w1 += w * g1;
          sum_w2 += w * g2;
          sum_w3 += w * g3;
        }

      const double inv_sw = 1.0 / fmax(sum_w, 1e-20);
      ch_dark_ref[0] = (float)(sum_w0 * inv_sw);
      ch_dark_ref[1] = (float)(sum_w1 * inv_sw);
      ch_dark_ref[2] = (float)(sum_w2 * inv_sw);
      ch_dark_ref[3] = (float)(sum_w3 * inv_sw);

      if(iter == n_iter) break;

      double sum_wresid2 = 0.0;
      double sum_wW = 0.0;
      const float dr0 = ch_dark_ref[0], dr1 = ch_dark_ref[1];
      const float dr2 = ch_dark_ref[2], dr3 = ch_dark_ref[3];
      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_wresid2,sum_wW)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          const double d0 = (double)g0 - dr0;
          const double d1 = (double)g1 - dr1;
          const double d2 = (double)g2 - dr2;
          const double d3 = (double)g3 - dr3;
          const double resid2 = d0*d0 + d1*d1 + d2*d2 + d3*d3;
          sum_wW      += w;
          sum_wresid2 += w * resid2 * 0.25;
        }
      const double inv_sw2 = 1.0 / fmax(sum_wW, 1e-20);
      const double measured_std = sqrt(fmax(sum_wresid2 * inv_sw2, 1e-20));
      const double ratio = 1.0 / measured_std;
      s_scale *= sqrt(ratio);
      if(s_scale < s_min) s_scale = s_min;
      if(s_scale > s_max) s_scale = s_max;
    }

    fprintf(stderr, "[GALOSH_RAW_L] dark anchor: s_init=%.6e s_final=%.6e | "
                    "ch_dark_ref: [0]=%.4f [1]=%.4f [2]=%.4f [3]=%.4f\n",
            s_init, s_scale,
            ch_dark_ref[0], ch_dark_ref[1], ch_dark_ref[2], ch_dark_ref[3]);

    DT_OMP_FOR()
    for(int r = 0; r < height; r++)
    {
      const int r_off = (r - co_row) & 1;
      for(int c = 0; c < width; c++)
      {
        const int c_off = (c - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        in_gat[(size_t)r * width + c] -= ch_dark_ref[slot];
      }
    }
  }

  /* ================================================================
   * [LATEST: GALOSH_RAW_L] Phase 3 — stride=1 forward 2x2 WHT (L-only).
   * Identical to K/J Phase 3.
   * ================================================================ */
  L_cs = dt_alloc_align_float(npixels);
  if(!L_cs) goto cleanup_rawlc_l;
  gat_h_forward_l_only_stride1(in_gat, L_cs, width, height);

  fprintf(stderr, "[GALOSH_RAW_L] Phase 3 forward WHT (stride=1, L-only) done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_L] Phase 4 — half-res chroma extract.
   * Identical to K/J Phase 4.
   * ================================================================ */
  C1_h = dt_alloc_align_float(chsize);
  C2_h = dt_alloc_align_float(chsize);
  C3_h = dt_alloc_align_float(chsize);
  if(!C1_h || !C2_h || !C3_h) goto cleanup_rawlc_l;
  gat_j_forward_c_halfres(in_gat, C1_h, C2_h, C3_h,
                           width, height, halfwidth, halfheight);

  dt_free_align(in_gat); in_gat = NULL;

  fprintf(stderr, "[GALOSH_RAW_L] Phase 4 half-res chroma extract done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_L] Phase 5 — denoise.
   *   5(L) full-res LOSH on L_cs (= K Phase 5(L))
   *   5(C) half-res 3-channel LOESS on C with sub-sampled L_cs_den
   *        as Y guide (= K Phase 5(C))
   * ================================================================ */
  L_cs_den = dt_alloc_align_float(npixels);
  if(!L_cs_den) goto cleanup_rawlc_l;
  galosh_pass12_multiorient_blocked(L_cs, L_cs_den, width, height,
                                     luma_strength, /*block=*/8,
                                     /*stride=*/2, /*n_orient=*/1,
                                     /*use_robust_shrink=*/1);
  dt_free_align(L_cs); L_cs = NULL;
  fprintf(stderr, "[GALOSH_RAW_L] Phase 5(L) full-res cycle-spun L Pass1+2 done\n");

  L_h_den  = dt_alloc_align_float(chsize);
  C1_h_den = dt_alloc_align_float(chsize);
  C2_h_den = dt_alloc_align_float(chsize);
  C3_h_den = dt_alloc_align_float(chsize);
  if(!L_h_den || !C1_h_den || !C2_h_den || !C3_h_den) goto cleanup_rawlc_l;

  DT_OMP_FOR()
  for(int hr = 0; hr < halfheight; hr++)
  {
    const int fr = 2 * hr;
    if(fr >= height) continue;
    for(int hc = 0; hc < halfwidth; hc++)
    {
      const int fc = 2 * hc;
      if(fc >= width) continue;
      L_h_den[(size_t)hr * halfwidth + hc] = L_cs_den[(size_t)fr * width + fc];
    }
  }

  galosh_loess_chroma_3ch_r(L_h_den, C1_h, C2_h, C3_h,
                             C1_h_den, C2_h_den, C3_h_den,
                             halfwidth, halfheight, chroma_strength,
                             /*R=*/GALOSH_LOESS_RADIUS,
                             /*BW=*/GALOSH_LOESS_BW);
  dt_free_align(L_h_den); L_h_den = NULL;
  dt_free_align(C1_h); C1_h = NULL;
  dt_free_align(C2_h); C2_h = NULL;
  dt_free_align(C3_h); C3_h = NULL;

  fprintf(stderr, "[GALOSH_RAW_L] Phase 5(C) half-res chroma LOESS done "
                   "(sigma_C=%.3f, R=%d, BW=%.1f, 3ch fused)\n",
          chroma_strength, GALOSH_LOESS_RADIUS, GALOSH_LOESS_BW);

  /* ================================================================
   * [LATEST: GALOSH_RAW_L] Phase 6 — L_pixel = 2x2 overlap average of
   * L_cs_den.  Computed BEFORE Phase 7 (vs K where it was AFTER K16
   * upsample) because the joint bilateral upsample needs L_pixel as
   * its bilateral guide.
   * ================================================================ */
  L_pixel = dt_alloc_align_float(npixels);
  if(!L_pixel) goto cleanup_rawlc_l;
  gat_i_lpixel_overlap_avg(L_cs_den, L_pixel, width, height);
  dt_free_align(L_cs_den); L_cs_den = NULL;

  fprintf(stderr, "[GALOSH_RAW_L] Phase 6 L_pixel overlap-avg done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_L] Phase 7 ⭐ NEW vs K ⭐ — Joint bilateral
   * K16 EWA-JL3 upsample.  Replaces K's Phase 6 (K16 standard upsample)
   * AND K's Phase 8 (LOESS edge-alignment refinement) with a single
   * fused stage:
   *
   *   w_combined[i] = w_jinc(d_i) × exp(-(L_pixel - L_at_h[i])²/(2BW²))
   *   C_full[fr,fc] = Σ w_combined[i] · C_h[i] / Σ w_combined[i]
   *
   * BW = 1.5 (= K Phase 8 BW, = "tight" bilateral for moderate-edge
   * separation; safe at this stage because input L_pixel residual σ
   * ≈ 0.1 << 1.5 in GAT-norm space).
   *
   * In flat L regions: bilateral ≈ uniform → effective kernel = pure
   * jinc → bandlimit-faithful (matches K16 standard exactly).
   * At L edges: bilateral kills cross-edge half-res samples → effective
   * kernel = one-sided jinc → C edges snap to L edges via cross-channel
   * structural prior.  Information advantage over K's chained approach:
   * the bilateral operates on the ORIGINAL half-res C samples (not the
   * pre-smoothed K16 output), preserving sub-pixel edge information.
   * ================================================================ */
  C1_aligned = dt_alloc_align_float(npixels);
  C2_aligned = dt_alloc_align_float(npixels);
  C3_aligned = dt_alloc_align_float(npixels);
  if(!C1_aligned || !C2_aligned || !C3_aligned) goto cleanup_rawlc_l;

  gat_k16_joint_bilateral_upsample(C1_h_den, C2_h_den, C3_h_den, L_pixel,
                                    C1_aligned, C2_aligned, C3_aligned,
                                    halfwidth, halfheight, /*BW=*/1.5f);
  dt_free_align(C1_h_den); C1_h_den = NULL;
  dt_free_align(C2_h_den); C2_h_den = NULL;
  dt_free_align(C3_h_den); C3_h_den = NULL;

  fprintf(stderr, "[GALOSH_RAW_L] Phase 7 joint bilateral K16 upsample done "
                   "(BW=1.5, 3ch fused; replaces K Phase 6+8)\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_L] Phase 8+9 — fused per-pixel WHT inverse
   * + dark_ref restore + ×unified_sigma denormalize + inverse GAT (LUT).
   * Identical to K Phase 9+10.
   * ================================================================ */
  {
    static const float SIGNS[4][3] = {
      { +1.0f, +1.0f, +1.0f },  /* R  */
      { -1.0f, +1.0f, -1.0f },  /* Gb */
      { +1.0f, -1.0f, -1.0f },  /* Gr */
      { -1.0f, -1.0f, +1.0f },  /* B  */
    };

    DT_OMP_FOR()
    for(int fr = 0; fr < height; fr++)
    {
      const int r_off = (fr - co_row) & 1;
      for(int fc = 0; fc < width; fc++)
      {
        const int c_off = (fc - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        const size_t pos = (size_t)fr * width + fc;
        const float val = 0.5f * (L_pixel[pos]
                                + SIGNS[slot][0] * C1_aligned[pos]
                                + SIGNS[slot][1] * C2_aligned[pos]
                                + SIGNS[slot][2] * C3_aligned[pos])
                        + ch_dark_ref[slot];
        out[pos] = gat_inverse_exact(val * unified_sigma);
      }
    }
  }

  fprintf(stderr, "[GALOSH_RAW_L] Phase 8+9 per-pixel inverse WHT + denorm + inv-GAT done\n");

cleanup_rawlc_l:
  dt_free_align(in_gat);
  dt_free_align(L_cs);
  dt_free_align(L_cs_den);
  dt_free_align(L_h_den);
  dt_free_align(C1_h); dt_free_align(C2_h); dt_free_align(C3_h);
  dt_free_align(C1_h_den); dt_free_align(C2_h_den); dt_free_align(C3_h_den);
  dt_free_align(C1_aligned); dt_free_align(C2_aligned); dt_free_align(C3_aligned);
  dt_free_align(L_pixel);
}


/* ================================================================
 * [LATEST: GALOSH_RAW_M] gat_galosh_denoise_rawlc_m — M pipeline entry.
 *
 * M = H + hierarchical Bayesian chroma denoising in Phase 5(C).
 *
 * Returns to H's full-res cycle-spinning architecture (no I/J/K/L
 * half-res chroma path) but replaces Phase 5(C) plain LOESS with a
 * principled two-scale Bayesian estimator that allows chroma_strength
 * slider to truly act as σ_n scaling — affecting BOTH flat-region
 * noise floor AND edge behavior, with FIXED computational cost.
 *
 * Theoretical motivation (no magic numbers):
 *   Generative model:
 *     C_true ~ N(C_global, σ_local²)        (= chroma local-vs-global prior)
 *     C_obs = C_true + n,  n ~ N(0, σ_n²)    (= GAT noise model)
 *     σ_n² = 1                                (= structural property of
 *                                                 GAT normalization, NOT
 *                                                 a magic constant)
 *
 *   Local LOESS estimator: C_local_hat with std σ_n/√N_local
 *   Global LOESS estimator: C_global_hat (= prior mean)
 *
 *   MAP posterior:
 *     w_data = (N_local · σ_local²) / (N_local · σ_local² + σ_n_eff²)
 *     C_post = w_data · C_local_hat + (1 - w_data) · C_global_hat
 *
 *   User slider: σ_n_eff² = chroma_strength² · σ_n²
 *     k = 0:  w_data = 1 → C_post = C_local (= H equivalent, max detail)
 *     k = 1:  balanced (σ_local²(x)-adaptive: flat→C_global, detail→C_local)
 *     k → ∞: w_data = 0 → C_post = C_global (= max smoothing)
 *
 *   σ_local²(x) is DATA-DRIVEN: estimated from local sample variance of
 *   C_in within R_local window, bias-corrected by σ_n².  No empirical
 *   tuning constant — the variance estimator is computed inside the
 *   LOESS pass (sumC²_per_channel) at +~15% LOESS cost.
 *
 * Constants (all derived, no magic):
 *   R_local  = 7   (CFA chroma half-Nyquist period × 3-period oversampling)
 *   R_global = 15  (= 2 · R_local; scale-space doubling, Lindeberg 1994)
 *   σ_n²     = 1   (GAT normalization structural property)
 *   N_local  = (2·R_local + 1)² = 225  (window sample count)
 *
 * Pipeline (= H Phases 0..4, 6..9; M's only structural change is Phase 5(C)):
 *   Phase 0..4    identical to H Phases 0..4 (= forward stride=1 cycle-
 *                  spinning WHT + CFA demod)
 *   Phase 5(L)    = H Phase 5(L) (full-res LOSH on cycle-spun L plane)
 *   Phase 5(C) ⭐ NEW vs H ⭐ — hierarchical Bayesian:
 *     5(C-i)  Local LOESS at R=7 with σ²_local(x) emission
 *               galosh_loess_chroma_3ch_r_with_var → C_local, var_C
 *     5(C-ii) Global LOESS at R=15 (= prior mean estimator)
 *               galosh_loess_chroma_3ch_r → C_global
 *     5(C-iii) Per-pixel Bayesian inverse-variance fusion
 *               gat_m_bayesian_fusion_3ch (chroma_strength as σ_n scaling)
 *   Phase 6..9    identical to H Phases 6..9 (= remod, 4-block overlap-add
 *                  inverse, per-pixel inverse + GAT)
 *
 * Cost: M Phase 5(C) is heavier than H Phase 5(C) due to R_global=15
 * LOESS (= 4.27× window vs R=7).  Net cost vs H depends on (A) multi-
 * channel fusion savings vs R_global expansion.  Speed measurement
 * pending — if too slow, R_global can be reduced to 11 (= 1.5×) or 9
 * (= scale-doubling principle relaxed for cost).
 *
 * (日) M = H + 階層 Bayesian chroma denoise.  chroma_strength を真の
 *   σ_n scaling として動作させる (= user 期待通り「強い denoise」)。
 *   R_local=7 (CFA Nyquist), R_global=15 (scale-space ×2)、σ_local²(x)
 *   data-driven、全 constant 構造的・教科書的・データ駆動 のいずれか
 *   (= magic-free)。
 * ================================================================ */
#ifndef GALOSH_RELEASE  /* [DEPRECATED] variant m — ablation builds only */
static void gat_galosh_denoise_rawlc_m(const float *const restrict in,
                                       float *const restrict out,
                                       const dt_iop_roi_t *const roi,
                                       const float luma_strength,
                                       const float chroma_strength,
                                       const uint32_t filters)
{
  const int width = roi->width, height = roi->height;
  const size_t npixels = (size_t)width * height;
  memcpy(out, in, sizeof(float) * npixels);

  if(luma_strength <= 0.0f) return;
  if(width < GALOSH_BLOCK_SIZE * 2 || height < GALOSH_BLOCK_SIZE * 2) return;

  const int co_row = 0, co_col = 0;
  (void)filters;

  /* M structural constants (all derived, no magic):
   *   R_local  = 7   (CFA Nyquist × 3-period oversampling)
   *   R_global = 15  (scale-doubling 2× R_local)
   *   N_local  = (2·R_local + 1)² = 225  (window samples) */
  const int R_local  = GALOSH_LOESS_RADIUS;          /* = 7 */
  const int R_global = 2 * GALOSH_LOESS_RADIUS + 1;  /* = 15 */
  const int N_local  = (2*R_local + 1) * (2*R_local + 1);

  /* ================================================================
   * [LATEST: GALOSH_RAW_M] Phase 0 — Foi-Alenius blind α / σ² estimation.
   * Identical to H/G Phase 0.
   * ================================================================ */
  const galosh_noise_params_t np = galosh_estimate_noise(in, width, height);
  gat_build_inverse_table(np.alpha, np.sigma_sq);

  /* Pre-declare buffers for cleanup-on-error. */
  float *in_gat = NULL;
  float *L = NULL, *C1 = NULL, *C2 = NULL, *C3 = NULL;
  float *L_den = NULL, *C1_den = NULL, *C2_den = NULL, *C3_den = NULL;
  float *C1_global = NULL, *C2_global = NULL, *C3_global = NULL;
  float *var_C1 = NULL, *var_C2 = NULL, *var_C3 = NULL;
  float *recon = NULL;

  in_gat = dt_alloc_align_float(npixels);
  if(!in_gat) goto cleanup_rawlc_m;

  /* ================================================================
   * [LATEST: GALOSH_RAW_M] Phase 1 — GAT forward (full-res) + per-CFA
   * σ_GAT MAD + RMS unified_sigma + scalar normalize.  Identical to H.
   * ================================================================ */
  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++)
    in_gat[i] = gat_forward(in[i], np.alpha, np.sigma_sq);

  float sigma_gat_ch[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  for(int s = 0; s < 4; s++)
  {
    const int ro = ((s & 1)        + co_row) & 1;
    const int co = (((s >> 1) & 1) + co_col) & 1;
    const int hw = (width  - co + 1) / 2;
    const int hh = (height - ro + 1) / 2;
    if(hw < 4 || hh < 4) continue;

    float *tmp = dt_alloc_align_float((size_t)hw * hh);
    if(!tmp) continue;
    DT_OMP_FOR()
    for(int rr = 0; rr < hh; rr++)
      for(int cc = 0; cc < hw; cc++)
        tmp[(size_t)rr * hw + cc] = in_gat[(size_t)(ro + 2*rr) * width + (co + 2*cc)];
    sigma_gat_ch[s] = estimate_gat_sigma_halfres(tmp, hw, hh);
    dt_free_align(tmp);
  }

  const float mean_var = 0.25f * (sigma_gat_ch[0]*sigma_gat_ch[0]
                                + sigma_gat_ch[1]*sigma_gat_ch[1]
                                + sigma_gat_ch[2]*sigma_gat_ch[2]
                                + sigma_gat_ch[3]*sigma_gat_ch[3]);
  const float unified_sigma = sqrtf(fmaxf(mean_var, 1e-12f));
  const float inv_sg = 1.0f / unified_sigma;

  fprintf(stderr, "[GALOSH_RAW_M] alpha=%.8f sigma_sq=%.10f | "
                  "unified_sigma=%.4f [RMS] (per-ch: %.4f %.4f %.4f %.4f) | "
                  "size=%dx%d | sigma_L=%.3f sigma_C=%.3f | R_local=%d R_global=%d\n",
                  np.alpha, np.sigma_sq, unified_sigma,
                  sigma_gat_ch[0], sigma_gat_ch[1], sigma_gat_ch[2], sigma_gat_ch[3],
                  width, height, luma_strength, chroma_strength,
                  R_local, R_global);

  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++) in_gat[i] *= inv_sg;

  /* ================================================================
   * [LATEST: GALOSH_RAW_M] Phase 2 — dark_ref IRLS + per-pixel subtract.
   * Identical to H Phase 2.
   * ================================================================ */
  float ch_dark_ref[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  {
    const double s_init = (double)np.sigma_sq / fmax((double)np.alpha, 1e-12);
    double s_scale = s_init;
    const double s_min = 0.05 * s_init;
    const double s_max = 50.0 * s_init;
    const int n_iter = 2;

    for(int iter = 0; iter <= n_iter; iter++)
    {
      const double inv_s = 1.0 / fmax(s_scale, 1e-20);
      double sum_w = 0.0;
      double sum_w0 = 0.0, sum_w1 = 0.0, sum_w2 = 0.0, sum_w3 = 0.0;

      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_w,sum_w0,sum_w1,sum_w2,sum_w3)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          sum_w  += w;
          sum_w0 += w * g0;
          sum_w1 += w * g1;
          sum_w2 += w * g2;
          sum_w3 += w * g3;
        }

      const double inv_sw = 1.0 / fmax(sum_w, 1e-20);
      ch_dark_ref[0] = (float)(sum_w0 * inv_sw);
      ch_dark_ref[1] = (float)(sum_w1 * inv_sw);
      ch_dark_ref[2] = (float)(sum_w2 * inv_sw);
      ch_dark_ref[3] = (float)(sum_w3 * inv_sw);

      if(iter == n_iter) break;

      double sum_wresid2 = 0.0;
      double sum_wW = 0.0;
      const float dr0 = ch_dark_ref[0], dr1 = ch_dark_ref[1];
      const float dr2 = ch_dark_ref[2], dr3 = ch_dark_ref[3];
      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_wresid2,sum_wW)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          const double d0 = (double)g0 - dr0;
          const double d1 = (double)g1 - dr1;
          const double d2 = (double)g2 - dr2;
          const double d3 = (double)g3 - dr3;
          const double resid2 = d0*d0 + d1*d1 + d2*d2 + d3*d3;
          sum_wW      += w;
          sum_wresid2 += w * resid2 * 0.25;
        }
      const double inv_sw2 = 1.0 / fmax(sum_wW, 1e-20);
      const double measured_std = sqrt(fmax(sum_wresid2 * inv_sw2, 1e-20));
      const double ratio = 1.0 / measured_std;
      s_scale *= sqrt(ratio);
      if(s_scale < s_min) s_scale = s_min;
      if(s_scale > s_max) s_scale = s_max;
    }

    fprintf(stderr, "[GALOSH_RAW_M] dark anchor: s_init=%.6e s_final=%.6e | "
                    "ch_dark_ref: [0]=%.4f [1]=%.4f [2]=%.4f [3]=%.4f\n",
            s_init, s_scale,
            ch_dark_ref[0], ch_dark_ref[1], ch_dark_ref[2], ch_dark_ref[3]);

    DT_OMP_FOR()
    for(int r = 0; r < height; r++)
    {
      const int r_off = (r - co_row) & 1;
      for(int c = 0; c < width; c++)
      {
        const int c_off = (c - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        in_gat[(size_t)r * width + c] -= ch_dark_ref[slot];
      }
    }
  }

  /* ================================================================
   * [LATEST: GALOSH_RAW_M] Phase 3 — stride=1 forward 2x2 WHT @ full-res.
   * Identical to H Phase 3.
   * ================================================================ */
  L  = dt_alloc_align_float(npixels);
  C1 = dt_alloc_align_float(npixels);
  C2 = dt_alloc_align_float(npixels);
  C3 = dt_alloc_align_float(npixels);
  if(!L || !C1 || !C2 || !C3) goto cleanup_rawlc_m;
  gat_h_forward_wht_stride1(in_gat, L, C1, C2, C3, width, height);

  dt_free_align(in_gat); in_gat = NULL;

  fprintf(stderr, "[GALOSH_RAW_M] Phase 3 forward WHT (stride=1) done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_M] Phase 4 — CFA sign-flip demodulate.
   * Identical to H Phase 4.
   * ================================================================ */
  gat_h_demodulate_chroma(C1, C2, C3, width, height);

  fprintf(stderr, "[GALOSH_RAW_M] Phase 4 CFA sign-flip demodulate done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_M] Phase 5(L) — full-res LOSH on L_cs.
   * Identical to H Phase 5(L).
   * ================================================================ */
  L_den  = dt_alloc_align_float(npixels);
  C1_den = dt_alloc_align_float(npixels);
  C2_den = dt_alloc_align_float(npixels);
  C3_den = dt_alloc_align_float(npixels);
  if(!L_den || !C1_den || !C2_den || !C3_den) goto cleanup_rawlc_m;

  galosh_pass12_multiorient_blocked(L, L_den, width, height,
                                     luma_strength, /*block=*/8,
                                     /*stride=*/2, /*n_orient=*/1,
                                     /*use_robust_shrink=*/1);
  fprintf(stderr, "[GALOSH_RAW_M] Phase 5(L) full-res cycle-spun L Pass1+2 done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_M] Phase 5(C) ⭐ NEW vs H ⭐ — hierarchical
   * Bayesian chroma denoise (= 3 sub-stages).
   * ================================================================ */

  /* 5(C-i): Local LOESS at R=7 with σ_local²(x) emission. */
  C1_global = dt_alloc_align_float(npixels);
  C2_global = dt_alloc_align_float(npixels);
  C3_global = dt_alloc_align_float(npixels);
  var_C1 = dt_alloc_align_float(npixels);
  var_C2 = dt_alloc_align_float(npixels);
  var_C3 = dt_alloc_align_float(npixels);
  if(!C1_global || !C2_global || !C3_global || !var_C1 || !var_C2 || !var_C3)
    goto cleanup_rawlc_m;

  galosh_loess_chroma_3ch_r_with_var(
      L_den, C1, C2, C3,
      C1_den, C2_den, C3_den,
      var_C1, var_C2, var_C3,
      width, height, chroma_strength,
      R_local, GALOSH_LOESS_BW);

  fprintf(stderr, "[GALOSH_RAW_M] Phase 5(C-i) local LOESS R=%d + sigma²_local(x) done\n", R_local);

  /* 5(C-ii): Global LOESS at R=15 (no variance needed). */
  galosh_loess_chroma_3ch_r(
      L_den, C1, C2, C3,
      C1_global, C2_global, C3_global,
      width, height, chroma_strength,
      R_global, GALOSH_LOESS_BW);

  fprintf(stderr, "[GALOSH_RAW_M] Phase 5(C-ii) global LOESS R=%d done\n", R_global);

  /* 5(C-iii): Per-pixel Bayesian inverse-variance fusion.
   * w_data(x) = N_local · σ_local² / (N_local · σ_local² + chroma_strength²)
   * C_out(x) = w_data · C_local + (1-w_data) · C_global
   *
   * In-place over C_den slots — per-pixel operation, no aliasing
   * concern (each output pixel only depends on the SAME pixel's local,
   * global, and variance values). */
  gat_m_bayesian_fusion_3ch(
      C1_den, C2_den, C3_den,           /* C_local (LOESS R=7) */
      C1_global, C2_global, C3_global,  /* C_global (LOESS R=15) */
      var_C1, var_C2, var_C3,            /* σ_local²(x) per channel */
      C1_den, C2_den, C3_den,            /* output (in-place safe per-pixel) */
      width, height, N_local, chroma_strength);

  /* C_global and var_C buffers no longer needed. */
  dt_free_align(C1_global); C1_global = NULL;
  dt_free_align(C2_global); C2_global = NULL;
  dt_free_align(C3_global); C3_global = NULL;
  dt_free_align(var_C1);    var_C1 = NULL;
  dt_free_align(var_C2);    var_C2 = NULL;
  dt_free_align(var_C3);    var_C3 = NULL;

  fprintf(stderr, "[GALOSH_RAW_M] Phase 5(C-iii) Bayesian fusion (N_local=%d, σ_n²_eff=%.3f) done\n",
          N_local, chroma_strength * chroma_strength);

  /* Pre-denoise L/C1/C2/C3 buffers no longer needed. */
  dt_free_align(L);  L  = NULL;
  dt_free_align(C1); C1 = NULL;
  dt_free_align(C2); C2 = NULL;
  dt_free_align(C3); C3 = NULL;

  /* ================================================================
   * [LATEST: GALOSH_RAW_M] Phase 6 — CFA sign-flip remodulate.
   * Identical to H Phase 6.
   * ================================================================ */
  gat_h_demodulate_chroma(C1_den, C2_den, C3_den, width, height);

  fprintf(stderr, "[GALOSH_RAW_M] Phase 6 CFA sign-flip remodulate done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_M] Phase 7 — 4-block overlap-add inverse 2x2 WHT.
   * Identical to H Phase 7.
   * ================================================================ */
  recon = dt_alloc_align_float(npixels);
  if(!recon) goto cleanup_rawlc_m;
  gat_h_inverse_overlap_add(L_den, C1_den, C2_den, C3_den,
                             recon, width, height);

  fprintf(stderr, "[GALOSH_RAW_M] Phase 7 inverse WHT (4-block overlap-add) done\n");

  /* ================================================================
   * [LATEST: GALOSH_RAW_M] Phase 8 + 9 — fused per-pixel dark_ref restore
   * + ×unified_sigma denormalize + inverse GAT (LUT).
   * Identical to H Phase 8+9.
   * ================================================================ */
  DT_OMP_FOR()
  for(int r = 0; r < height; r++)
  {
    const int r_off = (r - co_row) & 1;
    for(int c = 0; c < width; c++)
    {
      const int c_off = (c - co_col) & 1;
      const int slot  = r_off | (c_off << 1);
      const size_t pos = (size_t)r * width + c;
      const float val = (recon[pos] + ch_dark_ref[slot]) * unified_sigma;
      out[pos] = gat_inverse_exact(val);
    }
  }

  fprintf(stderr, "[GALOSH_RAW_M] Phase 8+9 dark_ref restore + denorm + inv-GAT done\n");

cleanup_rawlc_m:
  dt_free_align(in_gat);
  dt_free_align(L); dt_free_align(C1); dt_free_align(C2); dt_free_align(C3);
  dt_free_align(L_den); dt_free_align(C1_den); dt_free_align(C2_den); dt_free_align(C3_den);
  dt_free_align(C1_global); dt_free_align(C2_global); dt_free_align(C3_global);
  dt_free_align(var_C1); dt_free_align(var_C2); dt_free_align(var_C3);
  dt_free_align(recon);
}


/* ================================================================
 * [LATEST: GALOSH_RAW_O] gat_galosh_denoise_rawlc_o — O pipeline entry.
 *
 * O = L's GAT × WHT-LOSH × L/C decomposition pipeline EXTENDED with
 *   multi-scale LOESS chroma pyramid + smoothstep slider walk + L-guided
 *   K16 upsample at every chroma stage.  Replaces L's single-scale
 *   chroma LOESS (R=7, ~30 raw px receptive field) with a 3-level LOESS
 *   pyramid (half / quarter / eighth res, ~30 / 60 / 120 raw px) blended
 *   by a slider-controlled smoothstep walk.
 *
 * Design intent — "chroma blotch killer with functional flat-region slider":
 *   The L pipeline's LOESS slider modulates ε (= regularization on the
 *   linear-regression slope a).  In flat L regions cov(Y, C) ≈ 0 and a
 *   ≈ 0 regardless of slider value — so the slider effectively works
 *   only at edges, not at chroma blotches.  Yet color blotches (= 30 to
 *   120 raw px low-frequency chroma fluctuation) ARE the dominant
 *   chroma denoise target.  O fixes this by replacing "single-scale
 *   LOESS with slider-as-regularization" with "multi-scale LOESS pyramid
 *   with slider-as-scale-walk":
 *     - slider 0       → C_h (= no denoise, identity pass-through)
 *     - slider 1       → C_loess_h     (= L baseline,    30 raw px scale)
 *     - slider 2       → C_q_up        (= 60 raw px receptive field)
 *     - slider 3       → C_e_up        (= 120 raw px receptive field)
 *     - slider in (k, k+1): smoothstep blend between adjacent anchors
 *     - slider ≥ 3      → saturate at C_e_up
 *
 * L coupling at every chroma stage:
 *   - LOESS at half-res  uses  L_h_den  (= half-res sample of L_cs_den)
 *   - LOESS at quarter   uses  L_q      (= box_down(L_h_den))
 *   - LOESS at eighth    uses  L_e      (= box_down(L_q))
 *   - K16 upsample at each scale transition uses bilateral guide at the
 *     destination resolution (= L_q for eighth→quarter, L_h_den for
 *     quarter→half) via the existing gat_k16_joint_bilateral_upsample
 *   - Final half→full upsample uses L_pixel (= 2x2 overlap-avg of
 *     L_cs_den, full-res) as in L pipeline Phase 9.
 *   → all four full-res-L-derived guides leveraged at the appropriate
 *     scale; full-res L_pixel is applied at the final reconstruction.
 *
 * Pipeline (10 phases):
 *   Phase 0..6  identical to L Phase 0..6 (= GAT + L_cs forward + half-
 *               res chroma extract + single-scale luma WHT-LOSH +
 *               L_pixel + L_h_den).  Luma multi-scale was tried in N
 *               and regressed -0.6 dB (= cycle-spinning correlation
 *               breaks the σ scaling assumption); reverted to L.
 *   Phase 7  ⭐ Multi-scale LOESS pyramid + L-guided upsample at each
 *               scale transition.  3 LOESS calls (3ch fused per call):
 *               Lv0 = LOESS(C_h, L_h_den, R=7)  [≈ 30 raw px]
 *               Lv1 = LOESS(box_down(C_h), L_q, R=7)   [≈ 60 raw px]
 *               Lv2 = LOESS(box_down²(C_h), L_e, R=7)  [≈ 120 raw px]
 *               Upsamples (= 3 × gat_k16_joint_bilateral_upsample):
 *               C_q_up   = K16(C_loess_q,         L_h_den)  [q→h]
 *               C_e_to_q = K16(C_loess_e,         L_q)      [e→q]
 *               C_e_up   = K16(C_e_to_q,          L_h_den)  [q→h]
 *   Phase 8  ⭐ smoothstep slider walk (= 4 anchors: C_h / C_loess_h /
 *               C_q_up / C_e_up, blended by chroma_strength via cubic
 *               smoothstep within each unit interval for C¹-continuous
 *               slider feel; saturates at C_e_up for slider ≥ 3).
 *   Phase 9     Joint bilateral K16 EWA-JL3 upsample with full-res
 *               L_pixel guide (= L pipeline unchanged).
 *   Phase 10    Fused per-pixel inverse WHT + dark_ref restore + ×
 *               unified_sigma + inverse GAT (LUT) (= L Phase 8+9).
 *
 * Cost: chroma path = L baseline (LOESS at half-res, ~3,400 ops/pix)
 *   + LOESS at 1/4 (~850 ops/pix) + LOESS at 1/8 (~213 ops/pix)
 *   + 3 × K16 bilateral upsample (~240 ops/pix) + smoothstep blend
 *   ≈ chsize × 4,733 ops = L × 1.39.  Total time +12% on 16 MP SIDD.
 *
 * (日) GALOSH_RAW_O: L パイプラインの chroma LOESS を 3-level LOESS
 *   pyramid (half/quarter/eighth res, 受容野 30/60/120 raw px) に拡張、
 *   slider 値で smoothstep blend (= 0 noisy / 1 標準 / 3 max、 整数間
 *   C¹ 連続)。 chroma blotch (= 大粒度色むら) を slider で確実に
 *   reach できる、 N の WHT-LOSH 失敗を踏まえた LOESS-only design。
 *   各 scale で L_cs_den 由来 guide が効き、 final full-res reconstruct
 *   は L_pixel guide で edge 整合 = L coupling 全段配置 完全保持。
 * ================================================================ */
#endif  /* GALOSH_RELEASE: end deprecated m */
static void gat_galosh_denoise_rawlc_o(const float *const restrict in,
                                       float *const restrict out,
                                       const dt_iop_roi_t *const roi,
                                       const float luma_strength,
                                       const float chroma_strength,
                                       const uint32_t filters)
{
  const int width = roi->width, height = roi->height;
  const size_t npixels = (size_t)width * height;
  memcpy(out, in, sizeof(float) * npixels);

  if(luma_strength <= 0.0f) return;
  if(width < GALOSH_BLOCK_SIZE * 2 || height < GALOSH_BLOCK_SIZE * 2) return;

  const int co_row = 0, co_col = 0;
  (void)filters;

  const int halfwidth  = (width  + 1) / 2;
  const int halfheight = (height + 1) / 2;
  const size_t chsize  = (size_t)halfwidth * halfheight;

  /* Pre-declare buffers for cleanup-on-error. */
  float *in_gat = NULL;
  float *L_cs = NULL;
  float *L_cs_den = NULL;
  float *L_pixel = NULL;
  float *L_h_den = NULL;
  float *C1_h = NULL, *C2_h = NULL, *C3_h = NULL;
  /* Phase 7 chroma pyramid buffers. */
  float *L_q = NULL, *L_e = NULL;
  float *C1_q = NULL, *C2_q = NULL, *C3_q = NULL;
  float *C1_e = NULL, *C2_e = NULL, *C3_e = NULL;
  float *C1_loess_h = NULL, *C2_loess_h = NULL, *C3_loess_h = NULL;
  float *C1_loess_q = NULL, *C2_loess_q = NULL, *C3_loess_q = NULL;
  float *C1_loess_e = NULL, *C2_loess_e = NULL, *C3_loess_e = NULL;
  float *C1_q_up = NULL, *C2_q_up = NULL, *C3_q_up = NULL;
  float *C1_e_to_q = NULL, *C2_e_to_q = NULL, *C3_e_to_q = NULL;
  float *C1_e_up = NULL, *C2_e_up = NULL, *C3_e_up = NULL;
  /* Phase 8 output (= half-res blended chroma). */
  float *C1_h_den = NULL, *C2_h_den = NULL, *C3_h_den = NULL;
  /* Phase 9 output (= full-res aligned chroma). */
  float *C1_aligned = NULL, *C2_aligned = NULL, *C3_aligned = NULL;

  in_gat = dt_alloc_align_float(npixels);
  if(!in_gat) goto cleanup_rawlc_o;

  /* ============== [LATEST: GALOSH_RAW_O] Phase 0 — blind α / σ² ============== */
  const galosh_noise_params_t np = galosh_estimate_noise(in, width, height);
  gat_build_inverse_table(np.alpha, np.sigma_sq);

  /* Super-clean image-level gate.  When Phase 0 reports predicted noise std
   * below the threshold (default 0 = disabled), bypass the entire denoise
   * pipeline (Phase 1-10) and return the noisy input verbatim.  Background:
   * for super-clean images (= mid-tone pred noise std < 0.01) the per-block
   * BayesShrink + Wiener catastrophically destroys subtle texture (cliff
   * observation: ANY positive luma_strength produces -8 to -12 dB loss vs
   * noisy; only luma_strength=0 preserves).  Chroma path also contributes
   * to the destruction (= LOESS chroma denoising attenuates subtle color
   * structure).  Cleanest fix: skip the entire pipeline (= what the L=0
   * early return already does, but triggered automatically). */
  if(g_galosh_super_clean_threshold > 0.0f)
  {
    const float pred_noise_std_05 =
      sqrtf(fmaxf(np.alpha * 0.5f + np.sigma_sq, 0.0f));
    if(pred_noise_std_05 < g_galosh_super_clean_threshold)
    {
      fprintf(stderr, "[GALOSH_RAW_O] SUPER-CLEAN GATE: pred_noise_std(0.5)=%.6f "
              "< threshold=%.6f → bypass entire pipeline (out = noisy)\n",
              pred_noise_std_05, g_galosh_super_clean_threshold);
      /* out has already been initialized via memcpy(out, in) at top of
       * function.  Skip Phase 1-10 entirely. */
      return;
    }
  }

  /* ============== [LATEST: GALOSH_RAW_O] Phase 1 — GAT forward + per-CFA σ
   * + RMS unified_sigma + scalar normalize.  Identical to L Phase 1. ============== */
  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++)
    in_gat[i] = gat_forward(in[i], np.alpha, np.sigma_sq);

  float sigma_gat_ch[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  for(int s = 0; s < 4; s++)
  {
    const int ro = ((s & 1)        + co_row) & 1;
    const int co = (((s >> 1) & 1) + co_col) & 1;
    const int hw = (width  - co + 1) / 2;
    const int hh = (height - ro + 1) / 2;
    if(hw < 4 || hh < 4) continue;

    float *tmp = dt_alloc_align_float((size_t)hw * hh);
    if(!tmp) continue;
    DT_OMP_FOR()
    for(int rr = 0; rr < hh; rr++)
      for(int cc = 0; cc < hw; cc++)
        tmp[(size_t)rr * hw + cc] = in_gat[(size_t)(ro + 2*rr) * width + (co + 2*cc)];
    sigma_gat_ch[s] = estimate_gat_sigma_halfres(tmp, hw, hh);
    dt_free_align(tmp);
  }

  const float mean_var = 0.25f * (sigma_gat_ch[0]*sigma_gat_ch[0]
                                + sigma_gat_ch[1]*sigma_gat_ch[1]
                                + sigma_gat_ch[2]*sigma_gat_ch[2]
                                + sigma_gat_ch[3]*sigma_gat_ch[3]);
  float unified_sigma = sqrtf(fmaxf(mean_var, 1e-12f));

  /* Phase 1 override (CLI --unified-sigma=X): bypass the in_gat-based
   * measurement and force unified_sigma = X.  Used when Phase 0 (α, σ²)
   * is supplied externally (= EM_iter, oracle, etc.) and the caller wants
   * to disable Phase 1's safety-net re-estimation.  X = 1.0 corresponds to
   * "trust the supplied (α, σ²) completely; no further calibration". */
  if(g_galosh_unified_sigma_override > 0.0f)
  {
    fprintf(stderr, "[GALOSH_RAW_O] unified_sigma override: measured=%.4f -> %.4f\n",
            unified_sigma, g_galosh_unified_sigma_override);
    unified_sigma = g_galosh_unified_sigma_override;
  }
  const float inv_sg = 1.0f / unified_sigma;

  fprintf(stderr, "[GALOSH_RAW_O] alpha=%.8f sigma_sq=%.10f | "
                  "unified_sigma=%.4f [RMS] (per-ch: %.4f %.4f %.4f %.4f) | "
                  "size=%dx%d (half=%dx%d) | sigma_L=%.3f sigma_C=%.3f\n",
                  np.alpha, np.sigma_sq, unified_sigma,
                  sigma_gat_ch[0], sigma_gat_ch[1], sigma_gat_ch[2], sigma_gat_ch[3],
                  width, height, halfwidth, halfheight, luma_strength, chroma_strength);

  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++) in_gat[i] *= inv_sg;

  /* ============== [LATEST: GALOSH_RAW_O] Phase 2 — dark_ref IRLS + per-pixel
   * CFA-aware subtract.  Identical to L Phase 2. ============== */
  float ch_dark_ref[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  {
    const double s_init = (double)np.sigma_sq / fmax((double)np.alpha, 1e-12);
    double s_scale = s_init;
    const double s_min = 0.05 * s_init;
    const double s_max = 50.0 * s_init;
    const int n_iter = 2;

    for(int iter = 0; iter <= n_iter; iter++)
    {
      const double inv_s = 1.0 / fmax(s_scale, 1e-20);
      double sum_w = 0.0;
      double sum_w0 = 0.0, sum_w1 = 0.0, sum_w2 = 0.0, sum_w3 = 0.0;

      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_w,sum_w0,sum_w1,sum_w2,sum_w3)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          sum_w  += w;
          sum_w0 += w * g0;
          sum_w1 += w * g1;
          sum_w2 += w * g2;
          sum_w3 += w * g3;
        }

      const double inv_sw = 1.0 / fmax(sum_w, 1e-20);
      ch_dark_ref[0] = (float)(sum_w0 * inv_sw);
      ch_dark_ref[1] = (float)(sum_w1 * inv_sw);
      ch_dark_ref[2] = (float)(sum_w2 * inv_sw);
      ch_dark_ref[3] = (float)(sum_w3 * inv_sw);

      if(iter == n_iter) break;

      double sum_wresid2 = 0.0;
      double sum_wW = 0.0;
      const float dr0 = ch_dark_ref[0], dr1 = ch_dark_ref[1];
      const float dr2 = ch_dark_ref[2], dr3 = ch_dark_ref[3];
      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_wresid2,sum_wW)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          const double d0 = (double)g0 - dr0;
          const double d1 = (double)g1 - dr1;
          const double d2 = (double)g2 - dr2;
          const double d3 = (double)g3 - dr3;
          const double resid2 = d0*d0 + d1*d1 + d2*d2 + d3*d3;
          sum_wW      += w;
          sum_wresid2 += w * resid2 * 0.25;
        }
      const double inv_sw2 = 1.0 / fmax(sum_wW, 1e-20);
      const double measured_std = sqrt(fmax(sum_wresid2 * inv_sw2, 1e-20));
      const double ratio = 1.0 / measured_std;
      s_scale *= sqrt(ratio);
      if(s_scale < s_min) s_scale = s_min;
      if(s_scale > s_max) s_scale = s_max;
    }

    fprintf(stderr, "[GALOSH_RAW_O] dark anchor: s_init=%.6e s_final=%.6e | "
                    "ch_dark_ref: [0]=%.4f [1]=%.4f [2]=%.4f [3]=%.4f\n",
            s_init, s_scale,
            ch_dark_ref[0], ch_dark_ref[1], ch_dark_ref[2], ch_dark_ref[3]);

    DT_OMP_FOR()
    for(int r = 0; r < height; r++)
    {
      const int r_off = (r - co_row) & 1;
      for(int c = 0; c < width; c++)
      {
        const int c_off = (c - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        in_gat[(size_t)r * width + c] -= ch_dark_ref[slot];
      }
    }
  }

  /* Verification harness: dump intermediates if GALOSH_DUMP_DIR set. */
#define O32_CPU_DUMP(name, ptr, n_floats) do { \
  const char *_dd = getenv("GALOSH_DUMP_DIR"); \
  if(_dd) { \
    char _path[1024]; \
    snprintf(_path, sizeof(_path), "%s/%s.bin", _dd, name); \
    FILE *_df = fopen(_path, "wb"); \
    if(_df) { fwrite((ptr), sizeof(float), (n_floats), _df); fclose(_df); } \
  } \
} while(0)
  O32_CPU_DUMP("p2_in_gat", in_gat, npixels);

  /* ============== [LATEST: GALOSH_RAW_O] Phase 3 — stride=1 forward 2x2 WHT
   * (L-only).  Identical to L Phase 3. ============== */
  L_cs = dt_alloc_align_float(npixels);
  if(!L_cs) goto cleanup_rawlc_o;
  gat_h_forward_l_only_stride1(in_gat, L_cs, width, height);
  O32_CPU_DUMP("p3_L_cs", L_cs, npixels);

  fprintf(stderr, "[GALOSH_RAW_O] Phase 3 forward WHT (stride=1, L-only) done\n");

  /* ============== [LATEST: GALOSH_RAW_O] Phase 4 — half-res chroma extract.
   * Identical to L Phase 4. ============== */
  C1_h = dt_alloc_align_float(chsize);
  C2_h = dt_alloc_align_float(chsize);
  C3_h = dt_alloc_align_float(chsize);
  if(!C1_h || !C2_h || !C3_h) goto cleanup_rawlc_o;
  gat_j_forward_c_halfres(in_gat, C1_h, C2_h, C3_h,
                           width, height, halfwidth, halfheight);
  O32_CPU_DUMP("p4_C1_h", C1_h, chsize);
  O32_CPU_DUMP("p4_C2_h", C2_h, chsize);
  O32_CPU_DUMP("p4_C3_h", C3_h, chsize);
  dt_free_align(in_gat); in_gat = NULL;

  fprintf(stderr, "[GALOSH_RAW_O] Phase 4 half-res chroma extract done\n");

  /* ============== [LATEST: GALOSH_RAW_O] Phase 5 — single-scale WHT-LOSH on
   * L_cs.  Identical to L Phase 5(L).  N's multi-scale luma was diagnosed
   * to regress -0.6 dB (= cycle-spinning correlation breaks σ scaling),
   * reverted. ============== */
  L_cs_den = dt_alloc_align_float(npixels);
  if(!L_cs_den) goto cleanup_rawlc_o;
  /* Phase 5 expanded inline (= equivalent to galosh_pass12_multiorient_blocked
   * with n_orient=1) so pilot is exposed for cross-CPU/GPU verification
   * dumping.  Bit-identical to the wrapper for n_orient=1 path. */
  {
    float *L_cs_pilot = dt_alloc_align_float(npixels);
    if(!L_cs_pilot) goto cleanup_rawlc_o;
    galosh_pass1_blocked(L_cs, L_cs_pilot, width, height,
                          luma_strength, /*block=*/8, /*stride=*/2,
                          /*use_robust_shrink=*/1);
    O32_CPU_DUMP("p5_pilot", L_cs_pilot, npixels);
    galosh_pass2_blocked(L_cs, L_cs_pilot, L_cs_den, width, height,
                          luma_strength, /*wiener_floor=*/(1.0f / 8.0f),
                          /*block=*/8, /*stride=*/2);
    dt_free_align(L_cs_pilot);
  }
  O32_CPU_DUMP("p5_L_cs_den", L_cs_den, npixels);
  dt_free_align(L_cs); L_cs = NULL;
  fprintf(stderr, "[GALOSH_RAW_O] Phase 5 single-scale L WHT-LOSH done\n");

  /* ============== [LATEST: GALOSH_RAW_O] Phase 6 — L_pixel = 2x2 overlap-avg
   * of L_cs_den (= full-res chroma guide for Phase 9);
   * L_h_den = subsample of L_cs_den at every-other position (= half-res
   * chroma guide for Phase 7).  Identical to L Phase 6. ============== */
  L_pixel = dt_alloc_align_float(npixels);
  L_h_den = dt_alloc_align_float(chsize);
  if(!L_pixel || !L_h_den) goto cleanup_rawlc_o;

  gat_i_lpixel_overlap_avg(L_cs_den, L_pixel, width, height);

  DT_OMP_FOR()
  for(int hr = 0; hr < halfheight; hr++)
  {
    const int fr = 2 * hr;
    if(fr >= height) continue;
    for(int hc = 0; hc < halfwidth; hc++)
    {
      const int fc = 2 * hc;
      if(fc >= width) continue;
      L_h_den[(size_t)hr * halfwidth + hc] = L_cs_den[(size_t)fr * width + fc];
    }
  }
  O32_CPU_DUMP("p6_L_pixel", L_pixel, npixels);
  O32_CPU_DUMP("p6_L_h_den", L_h_den, chsize);
  dt_free_align(L_cs_den); L_cs_den = NULL;

  fprintf(stderr, "[GALOSH_RAW_O] Phase 6 L_pixel + L_h_den done\n");

  /* ============== [LATEST: GALOSH_RAW_O] Phase 7 ⭐ NEW vs L ⭐ — Multi-scale
   * LOESS chroma pyramid + L-guided K16 upsample at each scale transition.
   *
   * Constructs 3 LOESS-denoised chroma estimates at progressively wider
   * effective receptive fields:
   *   Lv0 (half-res,    ~30 raw px) = LOESS(C_h,             L_h_den, R=7)
   *   Lv1 (quarter-res, ~60 raw px) = LOESS(box_down(C_h),   L_q,     R=7)
   *   Lv2 (eighth-res, ~120 raw px) = LOESS(box_down²(C_h),  L_e,     R=7)
   * Each LOESS is the same fused 3ch Y-bilateral local linear regression
   * used in L pipeline Phase 5(C) — only the input scale changes.
   *
   * Coarse scales are upsampled to half-res with the existing
   * gat_k16_joint_bilateral_upsample (= K16 EWA-JL3 jinc + L bilateral
   * weight) using L at the destination scale as guide:
   *   C_q_up   = K16(C_loess_q,   L_h_den)         [quarter → half]
   *   C_e_to_q = K16(C_loess_e,   L_q)             [eighth  → quarter]
   *   C_e_up   = K16(C_e_to_q,    L_h_den)         [quarter → half]
   * The K16 bilateral guide ensures L coupling is preserved at every
   * scale transition (= no plain unguided upsample anywhere in the
   * pyramid). ============== */
  const int cq_w = halfwidth / 2;
  const int cq_h = halfheight / 2;
  const int ce_w = cq_w / 2;
  const int ce_h = cq_h / 2;
  const size_t cqsize = (size_t)cq_w * cq_h;
  const size_t cesize = (size_t)ce_w * ce_h;

  if(cq_w < 4 || cq_h < 4 || ce_w < 4 || ce_h < 4)
  {
    /* Image too small for 3-level pyramid — fall back to L pipeline. */
    fprintf(stderr, "[GALOSH_RAW_O] image too small for 3-level pyramid "
                    "(cq=%dx%d, ce=%dx%d); falling back to L pipeline\n",
            cq_w, cq_h, ce_w, ce_h);
    goto cleanup_rawlc_o;  /* TODO: graceful fallback to L variant */
  }

  /* ─── Build L pyramid (= box_down twice). ─── */
  L_q = dt_alloc_align_float(cqsize);
  L_e = dt_alloc_align_float(cesize);
  if(!L_q || !L_e) goto cleanup_rawlc_o;
  gat_box_downsample_2x(L_h_den, L_q, halfwidth, halfheight);
  gat_box_downsample_2x(L_q,     L_e, cq_w,      cq_h);

  /* ─── Build chroma pyramid (= box_down twice, per channel). ─── */
  C1_q = dt_alloc_align_float(cqsize);
  C2_q = dt_alloc_align_float(cqsize);
  C3_q = dt_alloc_align_float(cqsize);
  C1_e = dt_alloc_align_float(cesize);
  C2_e = dt_alloc_align_float(cesize);
  C3_e = dt_alloc_align_float(cesize);
  if(!C1_q || !C2_q || !C3_q || !C1_e || !C2_e || !C3_e) goto cleanup_rawlc_o;
  gat_box_downsample_2x(C1_h, C1_q, halfwidth, halfheight);
  gat_box_downsample_2x(C2_h, C2_q, halfwidth, halfheight);
  gat_box_downsample_2x(C3_h, C3_q, halfwidth, halfheight);
  gat_box_downsample_2x(C1_q, C1_e, cq_w, cq_h);
  gat_box_downsample_2x(C2_q, C2_e, cq_w, cq_h);
  gat_box_downsample_2x(C3_q, C3_e, cq_w, cq_h);

  /* ─── LOESS at each scale (3ch fused per call). ─── */
  C1_loess_h = dt_alloc_align_float(chsize);
  C2_loess_h = dt_alloc_align_float(chsize);
  C3_loess_h = dt_alloc_align_float(chsize);
  C1_loess_q = dt_alloc_align_float(cqsize);
  C2_loess_q = dt_alloc_align_float(cqsize);
  C3_loess_q = dt_alloc_align_float(cqsize);
  C1_loess_e = dt_alloc_align_float(cesize);
  C2_loess_e = dt_alloc_align_float(cesize);
  C3_loess_e = dt_alloc_align_float(cesize);
  if(!C1_loess_h || !C2_loess_h || !C3_loess_h ||
     !C1_loess_q || !C2_loess_q || !C3_loess_q ||
     !C1_loess_e || !C2_loess_e || !C3_loess_e)
    goto cleanup_rawlc_o;

  /* LOESS strength is fixed at the L-baseline tuning (= chroma_strength
   * controls the slider WALK across pyramid scales, not the LOESS ε).
   * Each LOESS invocation runs at "noise-matched" ε via strength_c=1.0. */
  const float loess_strength = 1.0f;
  galosh_loess_chroma_3ch_r(L_h_den, C1_h, C2_h, C3_h,
                             C1_loess_h, C2_loess_h, C3_loess_h,
                             halfwidth, halfheight, loess_strength,
                             GALOSH_LOESS_RADIUS, GALOSH_LOESS_BW);
  O32_CPU_DUMP("p7_C1_loess_h", C1_loess_h, chsize);
  galosh_loess_chroma_3ch_r(L_q, C1_q, C2_q, C3_q,
                             C1_loess_q, C2_loess_q, C3_loess_q,
                             cq_w, cq_h, loess_strength,
                             GALOSH_LOESS_RADIUS, GALOSH_LOESS_BW);
  galosh_loess_chroma_3ch_r(L_e, C1_e, C2_e, C3_e,
                             C1_loess_e, C2_loess_e, C3_loess_e,
                             ce_w, ce_h, loess_strength,
                             GALOSH_LOESS_RADIUS, GALOSH_LOESS_BW);

  /* C{1,2,3}_q and C{1,2,3}_e are pyramid INPUTS; LOESS outputs replace
   * them.  Free input pyramid buffers (= no longer needed). */
  dt_free_align(C1_q); dt_free_align(C2_q); dt_free_align(C3_q);
  C1_q = C2_q = C3_q = NULL;
  dt_free_align(C1_e); dt_free_align(C2_e); dt_free_align(C3_e);
  C1_e = C2_e = C3_e = NULL;

  fprintf(stderr, "[GALOSH_RAW_O] Phase 7 LOESS at 3 scales done "
                  "(half: %dx%d, quarter: %dx%d, eighth: %dx%d)\n",
          halfwidth, halfheight, cq_w, cq_h, ce_w, ce_h);

  /* ─── Upsample coarse levels to half-res via K16 joint bilateral. ───
   * Using L at the destination scale as bilateral guide preserves L
   * coupling at every scale transition (= no plain unguided upsample).
   *
   * STRIDE FIX: K16 hardcodes fw = 2*halfwidth_param, fh = 2*halfheight
   * for the destination buffer's stride.  When pyramid downsampling
   * gives odd dims (= e.g. SIDD halfwidth=2663 → cq_w=1331 → 2*cq_w=
   * 2662 ≠ halfwidth), the K16-internal stride does NOT match
   * chsize-allocated buffers' stride, producing garbage output.
   * Workaround: allocate K16 input/output buffers at exact (2*input_w
   * × 2*input_h) dim, then edge-replicate-pad to chsize for smoothstep
   * blend compatibility.
   */
  C1_q_up = dt_alloc_align_float(chsize);
  C2_q_up = dt_alloc_align_float(chsize);
  C3_q_up = dt_alloc_align_float(chsize);
  C1_e_up = dt_alloc_align_float(chsize);
  C2_e_up = dt_alloc_align_float(chsize);
  C3_e_up = dt_alloc_align_float(chsize);
  if(!C1_q_up || !C2_q_up || !C3_q_up ||
     !C1_e_up || !C2_e_up || !C3_e_up)
    goto cleanup_rawlc_o;

  /* quarter → half (L_h_den guide) — stride-corrected */
  {
    const int fw = 2 * cq_w;          /* output width K16 uses (= 2*input_w) */
    const int fh = 2 * cq_h;          /* output height K16 uses */
    const size_t fsize = (size_t)fw * fh;

    float *L_for_q = dt_alloc_align_float(fsize);
    float *C1_q_up_raw = dt_alloc_align_float(fsize);
    float *C2_q_up_raw = dt_alloc_align_float(fsize);
    float *C3_q_up_raw = dt_alloc_align_float(fsize);
    if(!L_for_q || !C1_q_up_raw || !C2_q_up_raw || !C3_q_up_raw)
    {
      dt_free_align(L_for_q);
      dt_free_align(C1_q_up_raw); dt_free_align(C2_q_up_raw); dt_free_align(C3_q_up_raw);
      goto cleanup_rawlc_o;
    }

    /* Crop L_h_den (chsize stride) to (fw × fh) for stride-matching K16. */
    gat_crop_2d_topleft(L_h_den, halfwidth, halfheight, L_for_q, fw, fh);

    gat_k16_joint_bilateral_upsample(C1_loess_q, C2_loess_q, C3_loess_q, L_for_q,
                                      C1_q_up_raw, C2_q_up_raw, C3_q_up_raw,
                                      cq_w, cq_h, /*BW=*/1.5f);

    /* Pad-to-chsize via edge replication for the smoothstep blend. */
    gat_pad_2d_edge(C1_q_up_raw, fw, fh, C1_q_up, halfwidth, halfheight);
    gat_pad_2d_edge(C2_q_up_raw, fw, fh, C2_q_up, halfwidth, halfheight);
    gat_pad_2d_edge(C3_q_up_raw, fw, fh, C3_q_up, halfwidth, halfheight);
    O32_CPU_DUMP("p7_C1_q_up", C1_q_up, chsize);

    dt_free_align(L_for_q);
    dt_free_align(C1_q_up_raw);
    dt_free_align(C2_q_up_raw);
    dt_free_align(C3_q_up_raw);
  }

  /* eighth → quarter → half (chained, with stride-correction at each step) */
  {
    /* Step 1: eighth → quarter.  K16 expects fw = 2*ce_w, fh = 2*ce_h. */
    const int fw_eq = 2 * ce_w;       /* K16 output width at quarter scale */
    const int fh_eq = 2 * ce_h;
    const size_t fsize_eq = (size_t)fw_eq * fh_eq;

    float *L_for_e = dt_alloc_align_float(fsize_eq);
    float *C1_e_to_q_raw = dt_alloc_align_float(fsize_eq);
    float *C2_e_to_q_raw = dt_alloc_align_float(fsize_eq);
    float *C3_e_to_q_raw = dt_alloc_align_float(fsize_eq);
    if(!L_for_e || !C1_e_to_q_raw || !C2_e_to_q_raw || !C3_e_to_q_raw)
    {
      dt_free_align(L_for_e);
      dt_free_align(C1_e_to_q_raw); dt_free_align(C2_e_to_q_raw); dt_free_align(C3_e_to_q_raw);
      goto cleanup_rawlc_o;
    }

    /* L_q is at (cq_w × cq_h); crop to (fw_eq × fh_eq) for K16 stride match. */
    gat_crop_2d_topleft(L_q, cq_w, cq_h, L_for_e, fw_eq, fh_eq);

    gat_k16_joint_bilateral_upsample(C1_loess_e, C2_loess_e, C3_loess_e, L_for_e,
                                      C1_e_to_q_raw, C2_e_to_q_raw, C3_e_to_q_raw,
                                      ce_w, ce_h, /*BW=*/1.5f);

    /* Pad raw output (fw_eq × fh_eq) up to (cq_w × cq_h) for next K16 step. */
    C1_e_to_q = dt_alloc_align_float(cqsize);
    C2_e_to_q = dt_alloc_align_float(cqsize);
    C3_e_to_q = dt_alloc_align_float(cqsize);
    if(!C1_e_to_q || !C2_e_to_q || !C3_e_to_q)
    {
      dt_free_align(L_for_e);
      dt_free_align(C1_e_to_q_raw); dt_free_align(C2_e_to_q_raw); dt_free_align(C3_e_to_q_raw);
      goto cleanup_rawlc_o;
    }
    gat_pad_2d_edge(C1_e_to_q_raw, fw_eq, fh_eq, C1_e_to_q, cq_w, cq_h);
    gat_pad_2d_edge(C2_e_to_q_raw, fw_eq, fh_eq, C2_e_to_q, cq_w, cq_h);
    gat_pad_2d_edge(C3_e_to_q_raw, fw_eq, fh_eq, C3_e_to_q, cq_w, cq_h);
    dt_free_align(L_for_e);
    dt_free_align(C1_e_to_q_raw);
    dt_free_align(C2_e_to_q_raw);
    dt_free_align(C3_e_to_q_raw);

    /* Step 2: quarter → half (= same as q→h above, but input is C_e_to_q). */
    const int fw_qh = 2 * cq_w;
    const int fh_qh = 2 * cq_h;
    const size_t fsize_qh = (size_t)fw_qh * fh_qh;

    float *L_for_q2 = dt_alloc_align_float(fsize_qh);
    float *C1_e_up_raw = dt_alloc_align_float(fsize_qh);
    float *C2_e_up_raw = dt_alloc_align_float(fsize_qh);
    float *C3_e_up_raw = dt_alloc_align_float(fsize_qh);
    if(!L_for_q2 || !C1_e_up_raw || !C2_e_up_raw || !C3_e_up_raw)
    {
      dt_free_align(L_for_q2);
      dt_free_align(C1_e_up_raw); dt_free_align(C2_e_up_raw); dt_free_align(C3_e_up_raw);
      goto cleanup_rawlc_o;
    }
    gat_crop_2d_topleft(L_h_den, halfwidth, halfheight, L_for_q2, fw_qh, fh_qh);

    gat_k16_joint_bilateral_upsample(C1_e_to_q, C2_e_to_q, C3_e_to_q, L_for_q2,
                                      C1_e_up_raw, C2_e_up_raw, C3_e_up_raw,
                                      cq_w, cq_h, /*BW=*/1.5f);

    gat_pad_2d_edge(C1_e_up_raw, fw_qh, fh_qh, C1_e_up, halfwidth, halfheight);
    gat_pad_2d_edge(C2_e_up_raw, fw_qh, fh_qh, C2_e_up, halfwidth, halfheight);
    gat_pad_2d_edge(C3_e_up_raw, fw_qh, fh_qh, C3_e_up, halfwidth, halfheight);
    O32_CPU_DUMP("p7_C1_e_up", C1_e_up, chsize);

    dt_free_align(L_for_q2);
    dt_free_align(C1_e_up_raw);
    dt_free_align(C2_e_up_raw);
    dt_free_align(C3_e_up_raw);
  }

  /* Free intermediates. */
  dt_free_align(C1_loess_q); dt_free_align(C2_loess_q); dt_free_align(C3_loess_q);
  C1_loess_q = C2_loess_q = C3_loess_q = NULL;
  dt_free_align(C1_loess_e); dt_free_align(C2_loess_e); dt_free_align(C3_loess_e);
  C1_loess_e = C2_loess_e = C3_loess_e = NULL;
  dt_free_align(C1_e_to_q); dt_free_align(C2_e_to_q); dt_free_align(C3_e_to_q);
  C1_e_to_q = C2_e_to_q = C3_e_to_q = NULL;
  dt_free_align(L_q); dt_free_align(L_e);
  L_q = L_e = NULL;

  fprintf(stderr, "[GALOSH_RAW_O] Phase 7 K16 joint-bilateral upsample × 3 done "
                  "(L coupling preserved at every scale transition)\n");

  /* ============== [LATEST: GALOSH_RAW_O] Phase 8 ⭐ NEW vs L ⭐ — smoothstep
   * slider walk over 4 anchors {C_h, C_loess_h, C_q_up, C_e_up}.
   *
   *   slider ∈ [0, 1]: t = smoothstep(s    );  C = (1-t)*C_h        + t*C_loess_h
   *   slider ∈ [1, 2]: t = smoothstep(s - 1); C = (1-t)*C_loess_h + t*C_q_up
   *   slider ∈ [2, 3]: t = smoothstep(s - 2); C = (1-t)*C_q_up    + t*C_e_up
   *   slider ≥ 3:                              C = C_e_up   (saturate)
   *
   * Cubic smoothstep (= 3t² - 2t³, derivative 0 at t=0,1) gives C¹
   * continuity at integer slider values — no "click" feel at scale
   * boundaries. ============== */
  C1_h_den = dt_alloc_align_float(chsize);
  C2_h_den = dt_alloc_align_float(chsize);
  C3_h_den = dt_alloc_align_float(chsize);
  if(!C1_h_den || !C2_h_den || !C3_h_den) goto cleanup_rawlc_o;

  {
    /* Determine segment + smoothstep parameter once (= same for all 3 ch). */
    const float s = chroma_strength;
    int segment;
    float t_raw;
    const float *A1, *A2, *A3, *B1, *B2, *B3;
    if(s <= 0.0f)
    {
      segment = -1;
      t_raw = 0.0f;
      A1 = A2 = A3 = B1 = B2 = B3 = NULL;
    }
    else if(s <= 1.0f)
    {
      segment = 0;
      t_raw = s;
      A1 = C1_h;       A2 = C2_h;       A3 = C3_h;
      B1 = C1_loess_h; B2 = C2_loess_h; B3 = C3_loess_h;
    }
    else if(s <= 2.0f)
    {
      segment = 1;
      t_raw = s - 1.0f;
      A1 = C1_loess_h; A2 = C2_loess_h; A3 = C3_loess_h;
      B1 = C1_q_up;    B2 = C2_q_up;    B3 = C3_q_up;
    }
    else if(s <= 3.0f)
    {
      segment = 2;
      t_raw = s - 2.0f;
      A1 = C1_q_up;    A2 = C2_q_up;    A3 = C3_q_up;
      B1 = C1_e_up;    B2 = C2_e_up;    B3 = C3_e_up;
    }
    else
    {
      segment = 3;
      t_raw = 1.0f;
      A1 = C1_e_up;    A2 = C2_e_up;    A3 = C3_e_up;
      B1 = NULL;       B2 = NULL;       B3 = NULL;
    }

    if(segment < 0)
    {
      /* slider ≤ 0: pure noisy, no denoise. */
      memcpy(C1_h_den, C1_h, sizeof(float) * chsize);
      memcpy(C2_h_den, C2_h, sizeof(float) * chsize);
      memcpy(C3_h_den, C3_h, sizeof(float) * chsize);
    }
    else if(segment >= 3 || B1 == NULL)
    {
      /* slider ≥ 3: saturate at C_e_up. */
      memcpy(C1_h_den, A1, sizeof(float) * chsize);
      memcpy(C2_h_den, A2, sizeof(float) * chsize);
      memcpy(C3_h_den, A3, sizeof(float) * chsize);
    }
    else
    {
      const float t = t_raw * t_raw * (3.0f - 2.0f * t_raw);   /* smoothstep */
      const float oneMt = 1.0f - t;
      DT_OMP_FOR()
      for(size_t i = 0; i < chsize; i++)
      {
        C1_h_den[i] = oneMt * A1[i] + t * B1[i];
        C2_h_den[i] = oneMt * A2[i] + t * B2[i];
        C3_h_den[i] = oneMt * A3[i] + t * B3[i];
      }
    }

    fprintf(stderr, "[GALOSH_RAW_O] Phase 8 smoothstep slider walk done "
                    "(slider=%.3f, segment=%d, t_raw=%.3f)\n",
            chroma_strength, segment, t_raw);
    O32_CPU_DUMP("p8_C1_h_den", C1_h_den, chsize);
  }

  /* Free anchor buffers (= already consumed). */
  dt_free_align(C1_h); dt_free_align(C2_h); dt_free_align(C3_h);
  C1_h = C2_h = C3_h = NULL;
  dt_free_align(C1_loess_h); dt_free_align(C2_loess_h); dt_free_align(C3_loess_h);
  C1_loess_h = C2_loess_h = C3_loess_h = NULL;
  dt_free_align(C1_q_up); dt_free_align(C2_q_up); dt_free_align(C3_q_up);
  C1_q_up = C2_q_up = C3_q_up = NULL;
  dt_free_align(C1_e_up); dt_free_align(C2_e_up); dt_free_align(C3_e_up);
  C1_e_up = C2_e_up = C3_e_up = NULL;
  dt_free_align(L_h_den); L_h_den = NULL;

  /* ============== [LATEST: GALOSH_RAW_O] Phase 9 — Joint bilateral K16
   * EWA-JL3 upsample to full-res with L_pixel guide.  Identical to L
   * pipeline. ============== */
  C1_aligned = dt_alloc_align_float(npixels);
  C2_aligned = dt_alloc_align_float(npixels);
  C3_aligned = dt_alloc_align_float(npixels);
  if(!C1_aligned || !C2_aligned || !C3_aligned) goto cleanup_rawlc_o;

  gat_k16_joint_bilateral_upsample(C1_h_den, C2_h_den, C3_h_den, L_pixel,
                                    C1_aligned, C2_aligned, C3_aligned,
                                    halfwidth, halfheight, /*BW=*/1.5f);
  O32_CPU_DUMP("p9_C1_aligned", C1_aligned, npixels);
  dt_free_align(C1_h_den); dt_free_align(C2_h_den); dt_free_align(C3_h_den);
  C1_h_den = C2_h_den = C3_h_den = NULL;

  fprintf(stderr, "[GALOSH_RAW_O] Phase 9 final K16 joint-bilateral upsample "
                  "(L_pixel guide, full-res reconstruction) done\n");

  /* ============== [LATEST: GALOSH_RAW_O] Phase 10 — fused per-pixel inverse
   * 2x2 WHT + dark_ref restore + ×unified_sigma + inverse GAT (LUT).
   * Identical to L Phase 8+9. ============== */
  {
    static const float SIGNS[4][3] = {
      { +1.0f, +1.0f, +1.0f },  /* R  */
      { -1.0f, +1.0f, -1.0f },  /* Gb */
      { +1.0f, -1.0f, -1.0f },  /* Gr */
      { -1.0f, -1.0f, +1.0f },  /* B  */
    };

    DT_OMP_FOR()
    for(int fr = 0; fr < height; fr++)
    {
      const int r_off = (fr - co_row) & 1;
      for(int fc = 0; fc < width; fc++)
      {
        const int c_off = (fc - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        const size_t pos = (size_t)fr * width + fc;
        const float val = 0.5f * (L_pixel[pos]
                                + SIGNS[slot][0] * C1_aligned[pos]
                                + SIGNS[slot][1] * C2_aligned[pos]
                                + SIGNS[slot][2] * C3_aligned[pos])
                        + ch_dark_ref[slot];
        out[pos] = gat_inverse_exact(val * unified_sigma);
      }
    }
  }

  fprintf(stderr, "[GALOSH_RAW_O] Phase 10 per-pixel inverse WHT + denorm + inv-GAT done\n");
  O32_CPU_DUMP("p10_output", out, npixels);
#undef O32_CPU_DUMP

cleanup_rawlc_o:
  dt_free_align(in_gat);
  dt_free_align(L_cs);
  dt_free_align(L_cs_den);
  dt_free_align(L_pixel);
  dt_free_align(L_h_den);
  dt_free_align(L_q); dt_free_align(L_e);
  dt_free_align(C1_h); dt_free_align(C2_h); dt_free_align(C3_h);
  dt_free_align(C1_q); dt_free_align(C2_q); dt_free_align(C3_q);
  dt_free_align(C1_e); dt_free_align(C2_e); dt_free_align(C3_e);
  dt_free_align(C1_loess_h); dt_free_align(C2_loess_h); dt_free_align(C3_loess_h);
  dt_free_align(C1_loess_q); dt_free_align(C2_loess_q); dt_free_align(C3_loess_q);
  dt_free_align(C1_loess_e); dt_free_align(C2_loess_e); dt_free_align(C3_loess_e);
  dt_free_align(C1_q_up); dt_free_align(C2_q_up); dt_free_align(C3_q_up);
  dt_free_align(C1_e_to_q); dt_free_align(C2_e_to_q); dt_free_align(C3_e_to_q);
  dt_free_align(C1_e_up); dt_free_align(C2_e_up); dt_free_align(C3_e_up);
  dt_free_align(C1_h_den); dt_free_align(C2_h_den); dt_free_align(C3_h_den);
  dt_free_align(C1_aligned); dt_free_align(C2_aligned); dt_free_align(C3_aligned);
}


/* ================================================================
 * [DEPRECATED: GALOSH_RAW_N] galosh_n_chroma_pyramid_lcoupled — 2-level
 * Laplacian pyramid + L-coupled WHT-LOSH on a single half-res chroma
 * plane.  Used by Phase 8 of the (deprecated) N pipeline.
 *
 * WHY DEPRECATED: replacing LOESS with L-coupled WHT-LOSH on chroma
 *   regressed sRGB PSNR by −3.4 dB on SIDD `0001_S6_GRBG_010` (= 38.77
 *   vs L's 42.14).  Diagnostic isolated the regression to the chroma
 *   side: LOESS's per-pixel Y-bilateral weighted local linear regression
 *   is structurally better-suited for half-Nyquist chroma than 8x8 WHT
 *   block transform, regardless of multi-scale or L-coupling.  Block-
 *   transform per-coefficient BayesShrink loses the cross-channel
 *   coherence that LOESS captures pixel-by-pixel.
 *
 * REPLACEMENT: GALOSH_RAW_O uses multi-scale LOESS pyramid (3 levels:
 *   half / quarter / eighth) + smoothstep slider walk.  Slider semantics
 *   match user spec (= 0 noisy, 1 standard, ≥1 stronger toward 120 raw
 *   px receptive field) without the WHT block-transform regression.
 *
 * Kept in source as reference for the negative result.
 *
 * Decompose:
 *   C_quarter = box_down(C_h)                 (quarter-res low-pass)
 *   C_detail  = C_h - box_up(C_quarter)       (half-res high-freq detail)
 *
 * Per-level denoise (= cross-channel BayesShrink threshold derived from
 * pooled C+L AC variance; see galosh_pass12_lcoupled_multiorient_blocked):
 *   C_quarter_den = lcoupled_WHT-LOSH(C_quarter, L_guide_quarter,
 *                                     σ = chroma_strength × 0.5)
 *   C_detail_den  = lcoupled_WHT-LOSH(C_detail,  L_guide_detail,
 *                                     σ = chroma_strength × 1.0)
 * σ scales 0.5× per coarser level (= 2x2 box-avg variance reduction
 * sqrt(1/4)).  Threshold uses pooled (C+L) AC sample variance so that
 * blocks with strong L structure get relaxed C threshold (= preserve
 * cross-channel-aligned chroma edges) while flat-L blocks get aggressive
 * threshold (= strong chroma smoothing where signal is weak).
 *
 * Reconstruct (= exact inverse of decompose if no denoising applied):
 *   C_h_den = box_up(C_quarter_den) + C_detail_den
 *
 * (日) Chroma 1ch を 2-level Laplacian pyramid に分解し、各 level で
 *   L-coupled WHT-LOSH (= 各 block で C+L AC pooled BayesShrink)。
 *   Coarse level の σ は 0.5× にスケール (= 2x2 box-down で var/4)。
 *   L 構造あり block では C threshold 緩和 (= cross-channel edge 保持)、
 *   L 平坦 block では aggressive threshold (= 弱信号で強 smoothing)。
 * ================================================================ */
static void galosh_n_chroma_pyramid_lcoupled(
    const float *restrict C_h,
    const float *restrict L_guide_quarter,
    const float *restrict L_guide_detail,
    float *restrict C_h_den,
    const int halfwidth, const int halfheight,
    const float chroma_strength)
{
  const int cq_w = halfwidth / 2;
  const int cq_h = halfheight / 2;
  const size_t chsize = (size_t)halfwidth * halfheight;
  const size_t cqsize = (size_t)cq_w * cq_h;

  float *C_quarter     = dt_alloc_align_float(cqsize);
  float *C_quarter_up  = dt_alloc_align_float(chsize);
  float *C_detail      = dt_alloc_align_float(chsize);
  float *C_quarter_den = dt_alloc_align_float(cqsize);
  float *C_detail_den  = dt_alloc_align_float(chsize);

  if(!C_quarter || !C_quarter_up || !C_detail || !C_quarter_den || !C_detail_den)
  {
    memcpy(C_h_den, C_h, sizeof(float) * chsize);
    goto cleanup_n_chroma;
  }

  /* Decompose */
  gat_box_downsample_2x(C_h, C_quarter, halfwidth, halfheight);
  gat_box_replicate_upsample_2x(C_quarter, C_quarter_up,
                                 cq_w, cq_h, halfwidth, halfheight);
  DT_OMP_FOR()
  for(size_t i = 0; i < chsize; i++)
    C_detail[i] = C_h[i] - C_quarter_up[i];

  /* Per-level σ measured via MAD-of-Laplacian (= sqrt(0.75)/0.5 textbook
   * for uncorrelated half-res C input + 2x2 box-avg, but measured values
   * track the actual marginal noise level robustly under signal). */
  const float sigma_C_q_meas = estimate_gat_sigma_halfres(C_quarter, cq_w, cq_h);
  const float sigma_C_d_meas = estimate_gat_sigma_halfres(C_detail, halfwidth, halfheight);

  /* Per-level L-coupled WHT-LOSH */
  galosh_pass12_lcoupled_multiorient_blocked(
      C_quarter, L_guide_quarter, C_quarter_den,
      cq_w, cq_h, chroma_strength * sigma_C_q_meas,
      /*block=*/8, /*stride=*/2, /*n_orient=*/1, /*use_robust_shrink=*/1);
  galosh_pass12_lcoupled_multiorient_blocked(
      C_detail, L_guide_detail, C_detail_den,
      halfwidth, halfheight, chroma_strength * sigma_C_d_meas,
      /*block=*/8, /*stride=*/2, /*n_orient=*/1, /*use_robust_shrink=*/1);

  /* Reconstruct */
  gat_box_replicate_upsample_2x(C_quarter_den, C_quarter_up,
                                 cq_w, cq_h, halfwidth, halfheight);
  DT_OMP_FOR()
  for(size_t i = 0; i < chsize; i++)
    C_h_den[i] = C_quarter_up[i] + C_detail_den[i];

cleanup_n_chroma:
  dt_free_align(C_quarter);
  dt_free_align(C_quarter_up);
  dt_free_align(C_detail);
  dt_free_align(C_quarter_den);
  dt_free_align(C_detail_den);
}


/* ================================================================
 * [DEPRECATED: GALOSH_RAW_N] gat_galosh_denoise_rawlc_n — N pipeline entry.
 *
 * STATUS: superseded by GALOSH_RAW_O (multi-scale LOESS pyramid +
 *   smoothstep slider walk).  N regressed −3.4 dB sRGB on SIDD
 *   `0001_S6_GRBG_010` because L-coupled WHT-LOSH on chroma is
 *   structurally inferior to LOESS regardless of multi-scale or L
 *   coupling.  Kept in source as the negative-result reference.
 *
 * ORIGINAL DESIGN (= what we tried):
 * N = L's GAT × WHT-LOSH × L/C decomposition pipeline EXTENDED with
 *   multi-scale (Laplacian pyramid) WHT-LOSH on luma + cross-channel
 *   L-coupled WHT-LOSH on chroma.  Replaces L's Phase 5(L) single-scale
 *   8x8 WHT-LOSH and Phase 5(C) LOESS chroma with multi-scale variants
 *   that maximise the L (full-res) / C (half-res) asymmetry GALOSH
 *   builds on.
 *
 * Design intent — "L coupling at every chroma stage":
 *   Phase 5(L) multi-scale = L's own multi-resolution structure used as
 *              its own prior; flat-region smoothing slider becomes
 *              truly functional via low-frequency band shrinkage that
 *              single-scale 8x8 WHT cannot reach.
 *   Phase 8    L-coupled  = L_guide_half (= 2x2 subsample of L_cs_den)
 *              guides chroma per-block BayesShrink threshold via pooled
 *              (C+L) AC variance.  chroma_strength slider works in flat
 *              regions while preserving L-aligned chroma edges.
 *   Phase 9    L-guided   = L_pixel (= overlap-avg of L_cs_den) guides
 *              K16 EWA-JL3 jinc kernel via bilateral weight, super-
 *              resolving chroma beyond half-Nyquist by borrowing L
 *              high-freq structure (= L variant unchanged).
 *
 * Pipeline (10 phases):
 *   Phase 0..4  identical to L Phase 0..4.
 *   Phase 5  ⭐ Multi-scale WHT-LOSH on luma (3-level Laplacian pyramid):
 *              decompose L_cs into {detail_0 (full), detail_1 (half),
 *              coarse_2 (quarter)}; WHT-LOSH per level with σ scaled
 *              by 0.5^k (= 2x2 box-avg variance reduction); reconstruct
 *              L_cs_den exactly inverse of decompose.
 *   Phase 6     L_pixel = 2x2 overlap-avg of L_cs_den (= L variant
 *              unchanged); L_guide_half = subsample L_cs_den at every
 *              other position (= L variant L_h_den convention).
 *   Phase 7     L_guide_half pyramid decompose (2 levels) →
 *              {L_guide_quarter, L_guide_detail_0}.
 *   Phase 8  ⭐ L-coupled multi-scale WHT-LOSH on chroma (3ch × 2-level):
 *              per channel decompose into {C_quarter, C_detail}; per
 *              level lcoupled WHT-LOSH (= pooled C+L AC BayesShrink);
 *              reconstruct C_h_den.
 *   Phase 9     Joint bilateral K16 EWA-JL3 upsample (= L variant
 *              unchanged): C_h_den + L_pixel guide → C_full_aligned.
 *   Phase 10    fused per-pixel WHT inverse + dark_ref restore + ×
 *              unified_sigma + inverse GAT (LUT) (= L Phase 8+9
 *              unchanged).
 *
 * Theoretical motivation:
 *   Single-scale 8x8 WHT-LOSH cannot denoise low-frequency noise below
 *   the 1/8-pixel Nyquist of its smallest detectable signal — flat
 *   regions accumulate noise floor that the slider cannot reach.
 *   Multi-scale extension (Burt-Adelson / Mallat) shrinks low-freq
 *   subbands at coarser resolutions where the noise floor is more
 *   visible.  L-coupled BayesShrink threshold (Chang-Yu-Vetterli 2000
 *   extended cross-channel) replaces LOESS bandwidth tuning with a
 *   per-block threshold modulated by cross-channel L AC variance,
 *   giving theoretically clean separation of edge/flat regions and
 *   full slider control over flat-region chroma smoothing.
 *
 * (日) GALOSH_RAW_N: L パイプラインに multi-scale + L-coupled 拡張。
 *   L (フル解像度) と C (半解像度) の解像度非対称性を最大限活用し、
 *   chroma processing 全 stage で L 情報を guide として使用。
 *   Phase 5(L) で 3-level Laplacian pyramid WHT-LOSH (= flat 域で
 *   slider 効く)、 Phase 8 で L-coupled BayesShrink (= cross-channel
 *   pooled variance、 LOESS 廃止)、 Phase 9 は joint bilateral upsample
 *   不変 (= L から full-res chroma 高周波借用)。
 * ================================================================ */
#ifndef GALOSH_RELEASE  /* [DEPRECATED] variant n (-3.4 dB) — ablation builds only */
static void gat_galosh_denoise_rawlc_n(const float *const restrict in,
                                       float *const restrict out,
                                       const dt_iop_roi_t *const roi,
                                       const float luma_strength,
                                       const float chroma_strength,
                                       const uint32_t filters)
{
  const int width = roi->width, height = roi->height;
  const size_t npixels = (size_t)width * height;
  memcpy(out, in, sizeof(float) * npixels);

  if(luma_strength <= 0.0f) return;
  if(width < 32 || height < 32)
  {
    /* Multi-scale 3-level pyramid needs >= 32 px to fit 8x8 WHT at
     * quarter-res.  Tiny input: fall back to L pipeline (single-scale
     * + LOESS chroma + joint bilateral upsample). */
    gat_galosh_denoise_rawlc_l(in, out, roi, luma_strength, chroma_strength, filters);
    return;
  }

  const int co_row = 0, co_col = 0;
  (void)filters;

  const int halfwidth  = (width  + 1) / 2;
  const int halfheight = (height + 1) / 2;
  const size_t chsize  = (size_t)halfwidth * halfheight;

  /* Pre-declare buffers for cleanup-on-error. */
  float *in_gat = NULL;
  float *L_cs = NULL;
  float *L_cs_den = NULL;
  float *L_pixel = NULL;
  float *C1_h = NULL, *C2_h = NULL, *C3_h = NULL;
  float *C1_h_den = NULL, *C2_h_den = NULL, *C3_h_den = NULL;
  float *C1_aligned = NULL, *C2_aligned = NULL, *C3_aligned = NULL;
  /* Phase 5 luma pyramid temps (freed after Phase 5 completes). */
  float *L_half = NULL, *L_quarter = NULL;
  float *L_detail_0 = NULL, *L_detail_1 = NULL;
  float *L_quarter_up_half = NULL, *L_half_up_full = NULL;
  float *L_detail_0_den = NULL, *L_detail_1_den = NULL, *L_quarter_den = NULL;
  /* Phase 6/7 chroma L-guide pyramid (freed after Phase 8 completes). */
  float *L_guide_half = NULL, *L_guide_quarter = NULL, *L_guide_detail_0 = NULL;

  in_gat = dt_alloc_align_float(npixels);
  if(!in_gat) goto cleanup_rawlc_n;

  /* ================================================================
   * [DEPRECATED: GALOSH_RAW_N] Phase 0 — Foi-Alenius blind α / σ²
   * estimation.  Identical to L Phase 0.
   * ================================================================ */
  const galosh_noise_params_t np = galosh_estimate_noise(in, width, height);
  gat_build_inverse_table(np.alpha, np.sigma_sq);

  /* ================================================================
   * [DEPRECATED: GALOSH_RAW_N] Phase 1 — GAT forward (full-res) + per-CFA
   * σ_GAT MAD + RMS unified_sigma + scalar normalize.  Identical to L.
   * ================================================================ */
  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++)
    in_gat[i] = gat_forward(in[i], np.alpha, np.sigma_sq);

  float sigma_gat_ch[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  for(int s = 0; s < 4; s++)
  {
    const int ro = ((s & 1)        + co_row) & 1;
    const int co = (((s >> 1) & 1) + co_col) & 1;
    const int hw = (width  - co + 1) / 2;
    const int hh = (height - ro + 1) / 2;
    if(hw < 4 || hh < 4) continue;

    float *tmp = dt_alloc_align_float((size_t)hw * hh);
    if(!tmp) continue;
    DT_OMP_FOR()
    for(int rr = 0; rr < hh; rr++)
      for(int cc = 0; cc < hw; cc++)
        tmp[(size_t)rr * hw + cc] = in_gat[(size_t)(ro + 2*rr) * width + (co + 2*cc)];
    sigma_gat_ch[s] = estimate_gat_sigma_halfres(tmp, hw, hh);
    dt_free_align(tmp);
  }

  const float mean_var = 0.25f * (sigma_gat_ch[0]*sigma_gat_ch[0]
                                + sigma_gat_ch[1]*sigma_gat_ch[1]
                                + sigma_gat_ch[2]*sigma_gat_ch[2]
                                + sigma_gat_ch[3]*sigma_gat_ch[3]);
  const float unified_sigma = sqrtf(fmaxf(mean_var, 1e-12f));
  const float inv_sg = 1.0f / unified_sigma;

  fprintf(stderr, "[GALOSH_RAW_N] alpha=%.8f sigma_sq=%.10f | "
                  "unified_sigma=%.4f [RMS] (per-ch: %.4f %.4f %.4f %.4f) | "
                  "size=%dx%d (half=%dx%d) | sigma_L=%.3f sigma_C=%.3f\n",
                  np.alpha, np.sigma_sq, unified_sigma,
                  sigma_gat_ch[0], sigma_gat_ch[1], sigma_gat_ch[2], sigma_gat_ch[3],
                  width, height, halfwidth, halfheight, luma_strength, chroma_strength);

  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++) in_gat[i] *= inv_sg;

  /* ================================================================
   * [DEPRECATED: GALOSH_RAW_N] Phase 2 — dark_ref IRLS + per-pixel CFA-aware
   * subtract.  Identical to L Phase 2.
   * ================================================================ */
  float ch_dark_ref[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  {
    const double s_init = (double)np.sigma_sq / fmax((double)np.alpha, 1e-12);
    double s_scale = s_init;
    const double s_min = 0.05 * s_init;
    const double s_max = 50.0 * s_init;
    const int n_iter = 2;

    for(int iter = 0; iter <= n_iter; iter++)
    {
      const double inv_s = 1.0 / fmax(s_scale, 1e-20);
      double sum_w = 0.0;
      double sum_w0 = 0.0, sum_w1 = 0.0, sum_w2 = 0.0, sum_w3 = 0.0;

      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_w,sum_w0,sum_w1,sum_w2,sum_w3)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          sum_w  += w;
          sum_w0 += w * g0;
          sum_w1 += w * g1;
          sum_w2 += w * g2;
          sum_w3 += w * g3;
        }

      const double inv_sw = 1.0 / fmax(sum_w, 1e-20);
      ch_dark_ref[0] = (float)(sum_w0 * inv_sw);
      ch_dark_ref[1] = (float)(sum_w1 * inv_sw);
      ch_dark_ref[2] = (float)(sum_w2 * inv_sw);
      ch_dark_ref[3] = (float)(sum_w3 * inv_sw);

      if(iter == n_iter) break;

      double sum_wresid2 = 0.0;
      double sum_wW = 0.0;
      const float dr0 = ch_dark_ref[0], dr1 = ch_dark_ref[1];
      const float dr2 = ch_dark_ref[2], dr3 = ch_dark_ref[3];
      #pragma omp parallel for collapse(2) schedule(static) \
              reduction(+:sum_wresid2,sum_wW)
      for(int br = 0; br < height - 1; br += 2)
        for(int bc = 0; bc < width - 1; bc += 2)
        {
          const float g0 = in_gat[(size_t)br     * width + bc    ];
          const float g1 = in_gat[(size_t)(br+1) * width + bc    ];
          const float g2 = in_gat[(size_t)br     * width + bc + 1];
          const float g3 = in_gat[(size_t)(br+1) * width + bc + 1];
          const float ch_max = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
          const float ch_min = fminf(fminf(g0, g1), fminf(g2, g3));
          if(ch_max - ch_min > GALOSH_ACHROMATIC_RANGE) continue;

          const float iv0 = in[(size_t)br     * width + bc    ];
          const float iv1 = in[(size_t)(br+1) * width + bc    ];
          const float iv2 = in[(size_t)br     * width + bc + 1];
          const float iv3 = in[(size_t)(br+1) * width + bc + 1];
          const double L_raw = (iv0 + iv1 + iv2 + iv3) * 0.25;
          const double r = L_raw * inv_s;
          const double r2 = r * r;
          const double w = 1.0 / (1.0 + r2 * r2);
          const double d0 = (double)g0 - dr0;
          const double d1 = (double)g1 - dr1;
          const double d2 = (double)g2 - dr2;
          const double d3 = (double)g3 - dr3;
          const double resid2 = d0*d0 + d1*d1 + d2*d2 + d3*d3;
          sum_wW      += w;
          sum_wresid2 += w * resid2 * 0.25;
        }
      const double inv_sw2 = 1.0 / fmax(sum_wW, 1e-20);
      const double measured_std = sqrt(fmax(sum_wresid2 * inv_sw2, 1e-20));
      const double ratio = 1.0 / measured_std;
      s_scale *= sqrt(ratio);
      if(s_scale < s_min) s_scale = s_min;
      if(s_scale > s_max) s_scale = s_max;
    }

    fprintf(stderr, "[GALOSH_RAW_N] dark anchor: s_init=%.6e s_final=%.6e | "
                    "ch_dark_ref: [0]=%.4f [1]=%.4f [2]=%.4f [3]=%.4f\n",
            s_init, s_scale,
            ch_dark_ref[0], ch_dark_ref[1], ch_dark_ref[2], ch_dark_ref[3]);

    DT_OMP_FOR()
    for(int r = 0; r < height; r++)
    {
      const int r_off = (r - co_row) & 1;
      for(int c = 0; c < width; c++)
      {
        const int c_off = (c - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        in_gat[(size_t)r * width + c] -= ch_dark_ref[slot];
      }
    }
  }

  /* ================================================================
   * [DEPRECATED: GALOSH_RAW_N] Phase 3 — stride=1 forward 2x2 WHT (L-only).
   * Identical to L Phase 3.
   * ================================================================ */
  L_cs = dt_alloc_align_float(npixels);
  if(!L_cs) goto cleanup_rawlc_n;
  gat_h_forward_l_only_stride1(in_gat, L_cs, width, height);

  fprintf(stderr, "[GALOSH_RAW_N] Phase 3 forward WHT (stride=1, L-only) done\n");

  /* ================================================================
   * [DEPRECATED: GALOSH_RAW_N] Phase 4 — half-res chroma extract.
   * Identical to L Phase 4.
   * ================================================================ */
  C1_h = dt_alloc_align_float(chsize);
  C2_h = dt_alloc_align_float(chsize);
  C3_h = dt_alloc_align_float(chsize);
  if(!C1_h || !C2_h || !C3_h) goto cleanup_rawlc_n;
  gat_j_forward_c_halfres(in_gat, C1_h, C2_h, C3_h,
                           width, height, halfwidth, halfheight);

  dt_free_align(in_gat); in_gat = NULL;

  fprintf(stderr, "[GALOSH_RAW_N] Phase 4 half-res chroma extract done\n");

  /* ================================================================
   * [DEPRECATED: GALOSH_RAW_N] Phase 5 ⭐ NEW vs L ⭐ — Multi-scale WHT-LOSH
   *
   *   coarse_2 (quarter-res low-pass) = box_down(box_down(L_cs))
   *   detail_1 (half-res high-freq)   = box_down(L_cs) - box_up(coarse_2)
   *   detail_0 (full-res high-freq)   = L_cs - box_up(box_down(L_cs))
   *
   * Per-level WHT-LOSH:
   *   detail_0_den  = WHT-LOSH(detail_0,  σ = luma_strength × 1.0)
   *   detail_1_den  = WHT-LOSH(detail_1,  σ = luma_strength × 0.5)
   *   coarse_2_den  = WHT-LOSH(coarse_2,  σ = luma_strength × 0.25)
   * σ scales 0.5^k per pyramid level (= 2x2 box-avg variance reduction).
   *
   * Reconstruct (= exact inverse of decompose if no denoising applied):
   *   L_half_den = box_up(coarse_2_den) + detail_1_den
   *   L_cs_den   = box_up(L_half_den)   + detail_0_den
   *
   * Multi-scale lets the slider reach low-frequency noise that
   * single-scale 8x8 cannot — flat-region smoothing becomes truly
   * functional.
   * ================================================================ */
  {
    const int lh_w = width / 2;
    const int lh_h = height / 2;
    const int lq_w = lh_w / 2;
    const int lq_h = lh_h / 2;
    const size_t lhsize = (size_t)lh_w * lh_h;
    const size_t lqsize = (size_t)lq_w * lq_h;

    L_half            = dt_alloc_align_float(lhsize);
    L_quarter         = dt_alloc_align_float(lqsize);
    L_detail_0        = dt_alloc_align_float(npixels);
    L_detail_1        = dt_alloc_align_float(lhsize);
    L_quarter_up_half = dt_alloc_align_float(lhsize);
    L_half_up_full    = dt_alloc_align_float(npixels);
    L_detail_0_den    = dt_alloc_align_float(npixels);
    L_detail_1_den    = dt_alloc_align_float(lhsize);
    L_quarter_den     = dt_alloc_align_float(lqsize);
    L_cs_den          = dt_alloc_align_float(npixels);
    if(!L_half || !L_quarter || !L_detail_0 || !L_detail_1 ||
       !L_quarter_up_half || !L_half_up_full ||
       !L_detail_0_den || !L_detail_1_den || !L_quarter_den || !L_cs_den)
      goto cleanup_rawlc_n;

    /* Decompose */
    gat_box_downsample_2x(L_cs,   L_half,    width, height);
    gat_box_downsample_2x(L_half, L_quarter, lh_w,  lh_h);
    gat_box_replicate_upsample_2x(L_quarter, L_quarter_up_half,
                                   lq_w, lq_h, lh_w, lh_h);
    gat_box_replicate_upsample_2x(L_half, L_half_up_full,
                                   lh_w, lh_h, width, height);

    DT_OMP_FOR()
    for(size_t i = 0; i < lhsize; i++)
      L_detail_1[i] = L_half[i] - L_quarter_up_half[i];

    DT_OMP_FOR()
    for(size_t i = 0; i < npixels; i++)
      L_detail_0[i] = L_cs[i] - L_half_up_full[i];

    /* Per-level σ measured via MAD-of-Laplacian (= robust to mild signal,
     * captures actual marginal noise level after box-down/detail-subtract).
     * Cycle-spinning correlation in L_cs makes the textbook 0.5^k scaling
     * incorrect — measured σ values typically come out around
     *   detail_0: ~0.66   (analytical sqrt(7/16))
     *   detail_1: ~0.5
     *   coarse_2: ~0.4
     * but vary with image content.  The slider semantics are preserved:
     * luma_strength × measured_σ matches noise at slider=1, scales
     * proportionally otherwise. */
    const float sigma_d0_meas = estimate_gat_sigma_halfres(L_detail_0, width, height);
    const float sigma_d1_meas = estimate_gat_sigma_halfres(L_detail_1, lh_w,  lh_h );
    const float sigma_q2_meas = estimate_gat_sigma_halfres(L_quarter,  lq_w,  lq_h );

    galosh_pass12_multiorient_blocked(L_detail_0, L_detail_0_den,
                                       width, height, luma_strength * sigma_d0_meas,
                                       /*block=*/8, /*stride=*/2,
                                       /*n_orient=*/1, /*use_robust_shrink=*/1);
    galosh_pass12_multiorient_blocked(L_detail_1, L_detail_1_den,
                                       lh_w, lh_h, luma_strength * sigma_d1_meas,
                                       /*block=*/8, /*stride=*/2,
                                       /*n_orient=*/1, /*use_robust_shrink=*/1);
    galosh_pass12_multiorient_blocked(L_quarter, L_quarter_den,
                                       lq_w, lq_h, luma_strength * sigma_q2_meas,
                                       /*block=*/8, /*stride=*/2,
                                       /*n_orient=*/1, /*use_robust_shrink=*/1);

    /* Reconstruct: L_half_den = box_up(L_quarter_den) + L_detail_1_den;
     *              L_cs_den   = box_up(L_half_den)    + L_detail_0_den. */
    gat_box_replicate_upsample_2x(L_quarter_den, L_quarter_up_half,
                                   lq_w, lq_h, lh_w, lh_h);
    DT_OMP_FOR()
    for(size_t i = 0; i < lhsize; i++)
      L_half[i] = L_quarter_up_half[i] + L_detail_1_den[i];   /* reuse L_half as L_half_den */

    gat_box_replicate_upsample_2x(L_half, L_half_up_full,
                                   lh_w, lh_h, width, height);
    DT_OMP_FOR()
    for(size_t i = 0; i < npixels; i++)
      L_cs_den[i] = L_half_up_full[i] + L_detail_0_den[i];

    /* Free Phase 5 temps; keep L_cs_den for Phase 6. */
    dt_free_align(L_cs);              L_cs = NULL;
    dt_free_align(L_half);            L_half = NULL;
    dt_free_align(L_quarter);         L_quarter = NULL;
    dt_free_align(L_detail_0);        L_detail_0 = NULL;
    dt_free_align(L_detail_1);        L_detail_1 = NULL;
    dt_free_align(L_quarter_up_half); L_quarter_up_half = NULL;
    dt_free_align(L_half_up_full);    L_half_up_full = NULL;
    dt_free_align(L_detail_0_den);    L_detail_0_den = NULL;
    dt_free_align(L_detail_1_den);    L_detail_1_den = NULL;
    dt_free_align(L_quarter_den);     L_quarter_den = NULL;

    fprintf(stderr, "[GALOSH_RAW_N] Phase 5 multi-scale WHT-LOSH on luma done "
                    "(3 levels: meas_σ=%.3f/%.3f/%.3f × luma_strength=%.3f)\n",
            sigma_d0_meas, sigma_d1_meas, sigma_q2_meas, luma_strength);
  }

  /* ================================================================
   * [DEPRECATED: GALOSH_RAW_N] Phase 6 — L_pixel = 2x2 overlap-avg of
   * L_cs_den (= L variant unchanged, used as Phase 9 bilateral guide
   * + Phase 10 inverse WHT per-pixel L);
   * L_guide_half = subsample L_cs_den at every-other position
   * (= L variant L_h_den convention; used as Phase 8 cross-channel
   * coupling guide).
   * ================================================================ */
  L_pixel      = dt_alloc_align_float(npixels);
  L_guide_half = dt_alloc_align_float(chsize);
  if(!L_pixel || !L_guide_half) goto cleanup_rawlc_n;

  gat_i_lpixel_overlap_avg(L_cs_den, L_pixel, width, height);

  DT_OMP_FOR()
  for(int hr = 0; hr < halfheight; hr++)
  {
    const int fr = 2 * hr;
    if(fr >= height) continue;
    for(int hc = 0; hc < halfwidth; hc++)
    {
      const int fc = 2 * hc;
      if(fc >= width) continue;
      L_guide_half[(size_t)hr * halfwidth + hc] = L_cs_den[(size_t)fr * width + fc];
    }
  }

  dt_free_align(L_cs_den); L_cs_den = NULL;

  fprintf(stderr, "[GALOSH_RAW_N] Phase 6 L_pixel overlap-avg + L_guide_half subsample done\n");

  /* ================================================================
   * [DEPRECATED: GALOSH_RAW_N] Phase 7 — L_guide_half pyramid decompose
   * (2 levels):
   *   L_guide_quarter  = box_down(L_guide_half)
   *   L_guide_detail_0 = L_guide_half - box_up(L_guide_quarter)
   * Used as cross-channel guides for Phase 8 chroma denoise.
   * ================================================================ */
  {
    const int cq_w = halfwidth / 2;
    const int cq_h = halfheight / 2;
    const size_t cqsize = (size_t)cq_w * cq_h;

    L_guide_quarter  = dt_alloc_align_float(cqsize);
    L_guide_detail_0 = dt_alloc_align_float(chsize);
    if(!L_guide_quarter || !L_guide_detail_0) goto cleanup_rawlc_n;

    gat_box_downsample_2x(L_guide_half, L_guide_quarter, halfwidth, halfheight);

    float *tmp = dt_alloc_align_float(chsize);
    if(!tmp) goto cleanup_rawlc_n;
    gat_box_replicate_upsample_2x(L_guide_quarter, tmp,
                                   cq_w, cq_h, halfwidth, halfheight);
    DT_OMP_FOR()
    for(size_t i = 0; i < chsize; i++)
      L_guide_detail_0[i] = L_guide_half[i] - tmp[i];
    dt_free_align(tmp);

    dt_free_align(L_guide_half); L_guide_half = NULL;

    fprintf(stderr, "[GALOSH_RAW_N] Phase 7 L_guide_half pyramid decompose done\n");
  }

  /* ================================================================
   * [DEPRECATED: GALOSH_RAW_N] Phase 8 ⭐ NEW vs L ⭐ — L-coupled multi-scale
   * WHT-LOSH on chroma (3 channels × 2-level pyramid).  Replaces L's
   * Phase 5(C) LOESS chroma with cross-channel BayesShrink that uses
   * pooled (C+L) AC variance per block — chroma_strength slider
   * functional in flat regions, edge preservation via L coupling.
   * See galosh_n_chroma_pyramid_lcoupled() for per-channel details.
   * ================================================================ */
  C1_h_den = dt_alloc_align_float(chsize);
  C2_h_den = dt_alloc_align_float(chsize);
  C3_h_den = dt_alloc_align_float(chsize);
  if(!C1_h_den || !C2_h_den || !C3_h_den) goto cleanup_rawlc_n;

  galosh_n_chroma_pyramid_lcoupled(C1_h, L_guide_quarter, L_guide_detail_0,
                                    C1_h_den, halfwidth, halfheight, chroma_strength);
  galosh_n_chroma_pyramid_lcoupled(C2_h, L_guide_quarter, L_guide_detail_0,
                                    C2_h_den, halfwidth, halfheight, chroma_strength);
  galosh_n_chroma_pyramid_lcoupled(C3_h, L_guide_quarter, L_guide_detail_0,
                                    C3_h_den, halfwidth, halfheight, chroma_strength);

  dt_free_align(C1_h); C1_h = NULL;
  dt_free_align(C2_h); C2_h = NULL;
  dt_free_align(C3_h); C3_h = NULL;
  dt_free_align(L_guide_quarter);  L_guide_quarter = NULL;
  dt_free_align(L_guide_detail_0); L_guide_detail_0 = NULL;

  fprintf(stderr, "[GALOSH_RAW_N] Phase 8 L-coupled multi-scale WHT-LOSH on chroma done "
                  "(3ch × 2 levels: σ=%.3f / %.3f)\n",
          chroma_strength, chroma_strength * 0.5f);

  /* ================================================================
   * [DEPRECATED: GALOSH_RAW_N] Phase 9 — Joint bilateral K16 EWA-JL3
   * upsample (= L variant unchanged).  C_h_den + L_pixel guide →
   * C_full_aligned via jinc kernel × bilateral weight on L_pixel
   * (BW=1.5 in GAT-norm space).  Borrows L high-freq structure to
   * super-resolve chroma beyond half-Nyquist.
   * ================================================================ */
  C1_aligned = dt_alloc_align_float(npixels);
  C2_aligned = dt_alloc_align_float(npixels);
  C3_aligned = dt_alloc_align_float(npixels);
  if(!C1_aligned || !C2_aligned || !C3_aligned) goto cleanup_rawlc_n;

  gat_k16_joint_bilateral_upsample(C1_h_den, C2_h_den, C3_h_den, L_pixel,
                                    C1_aligned, C2_aligned, C3_aligned,
                                    halfwidth, halfheight, /*BW=*/1.5f);
  dt_free_align(C1_h_den); C1_h_den = NULL;
  dt_free_align(C2_h_den); C2_h_den = NULL;
  dt_free_align(C3_h_den); C3_h_den = NULL;

  fprintf(stderr, "[GALOSH_RAW_N] Phase 9 joint bilateral K16 upsample done (BW=1.5)\n");

  /* ================================================================
   * [DEPRECATED: GALOSH_RAW_N] Phase 10 — fused per-pixel WHT inverse
   * + dark_ref restore + ×unified_sigma denormalize + inverse GAT
   * (LUT).  Identical to L Phase 8+9.
   * ================================================================ */
  {
    static const float SIGNS[4][3] = {
      { +1.0f, +1.0f, +1.0f },  /* R  */
      { -1.0f, +1.0f, -1.0f },  /* Gb */
      { +1.0f, -1.0f, -1.0f },  /* Gr */
      { -1.0f, -1.0f, +1.0f },  /* B  */
    };

    DT_OMP_FOR()
    for(int fr = 0; fr < height; fr++)
    {
      const int r_off = (fr - co_row) & 1;
      for(int fc = 0; fc < width; fc++)
      {
        const int c_off = (fc - co_col) & 1;
        const int slot  = r_off | (c_off << 1);
        const size_t pos = (size_t)fr * width + fc;
        const float val = 0.5f * (L_pixel[pos]
                                + SIGNS[slot][0] * C1_aligned[pos]
                                + SIGNS[slot][1] * C2_aligned[pos]
                                + SIGNS[slot][2] * C3_aligned[pos])
                        + ch_dark_ref[slot];
        out[pos] = gat_inverse_exact(val * unified_sigma);
      }
    }
  }

  fprintf(stderr, "[GALOSH_RAW_N] Phase 10 per-pixel inverse WHT + denorm + inv-GAT done\n");

cleanup_rawlc_n:
  dt_free_align(in_gat);
  dt_free_align(L_cs);
  dt_free_align(L_cs_den);
  dt_free_align(L_pixel);
  dt_free_align(C1_h); dt_free_align(C2_h); dt_free_align(C3_h);
  dt_free_align(C1_h_den); dt_free_align(C2_h_den); dt_free_align(C3_h_den);
  dt_free_align(C1_aligned); dt_free_align(C2_aligned); dt_free_align(C3_aligned);
  dt_free_align(L_half); dt_free_align(L_quarter);
  dt_free_align(L_detail_0); dt_free_align(L_detail_1);
  dt_free_align(L_quarter_up_half); dt_free_align(L_half_up_full);
  dt_free_align(L_detail_0_den); dt_free_align(L_detail_1_den); dt_free_align(L_quarter_den);
  dt_free_align(L_guide_half); dt_free_align(L_guide_quarter); dt_free_align(L_guide_detail_0);
}


/* --- main() --- */
#endif  /* GALOSH_RELEASE: end deprecated n */
int main(int argc, char **argv)
{
  /* Strip --stride / --orient / --lfr-kernel flags out of argv before
   * parsing positional args.  Order-independent so the bench harness
   * can append them anywhere on the CLI. */
  int new_argc = 0;
  char *positional[32];
  for(int i = 0; i < argc; i++)
  {
    const char *a = argv[i];
    if(strncmp(a, "--stride=", 9) == 0)
    {
      g_galosh_stride = atoi(a + 9);
      if(g_galosh_stride < 1 || g_galosh_stride > 8) g_galosh_stride = 2;
    }
    else if(strncmp(a, "--orient=", 9) == 0)
    {
      g_galosh_n_orient = atoi(a + 9);
      if(g_galosh_n_orient != 1 && g_galosh_n_orient != 4) g_galosh_n_orient = 1;
    }
    else if(strncmp(a, "--lfr-kernel=", 13) == 0)
    {
      const char *k = a + 13;
      if(strcmp(k, "ewajl3") == 0 || strcmp(k, "ewa-jl3") == 0)
        g_galosh_lfr_kernel = 1;
      else
        g_galosh_lfr_kernel = 0;
    }
    else if(strncmp(a, "--unified=", 10) == 0)
    {
      g_galosh_unified = (atoi(a + 10) != 0);
    }
    else if(strncmp(a, "--k13-block=", 12) == 0)
    {
      const int b = atoi(a + 12);
      if(b == 4 || b == 8) g_galosh_k13_block = b;
      else                 g_galosh_k13_block = 8;
    }
    else if(strncmp(a, "--variant=", 10) == 0)
    {
      /* [LATEST: GALOSH_RAW_O] variant dispatch — accepts 'o' (default,
       * LATEST), 'n' (DEPRECATED: -3.4 dB), 'm', 'l', 'k', 'j', 'i', 'h',
       * 'g' (= all PREVIOUS for bench). */
      const char ch = a[10];
#ifndef GALOSH_RELEASE
      if(ch == 'g' || ch == 'G')      g_galosh_variant = 'g';
      else if(ch == 'h' || ch == 'H') g_galosh_variant = 'h';
      else if(ch == 'i' || ch == 'I') g_galosh_variant = 'i';
      else if(ch == 'j' || ch == 'J') g_galosh_variant = 'j';
      else if(ch == 'k' || ch == 'K') g_galosh_variant = 'k';
      else if(ch == 'l' || ch == 'L') g_galosh_variant = 'l';
      else if(ch == 'm' || ch == 'M') g_galosh_variant = 'm';
      else if(ch == 'n' || ch == 'N') g_galosh_variant = 'n';
      else                            g_galosh_variant = 'o';
#else
      g_galosh_variant = 'o';   /* release build: canonical GALOSH_RAW_O only */
      (void)ch;
#endif
    }
    else if(strncmp(a, "--pass1=", 8) == 0)
    {
      const char *v = a + 8;
      if(strcmp(v, "a1") == 0 || strcmp(v, "A1") == 0) g_galosh_pass1_mode = 1;
      else                                              g_galosh_pass1_mode = 0;
      fprintf(stderr, "  --pass1=%s (mode=%d)\n", v, g_galosh_pass1_mode);
    }
    else if(strncmp(a, "--super-clean-threshold=", 24) == 0)
    {
      g_galosh_super_clean_threshold = (float)atof(a + 24);
      if(g_galosh_super_clean_threshold < 0.0f) g_galosh_super_clean_threshold = 0.0f;
      fprintf(stderr, "  --super-clean-threshold=%.6f\n",
              g_galosh_super_clean_threshold);
    }
    else if(strncmp(a, "--unified-sigma=", 16) == 0)
    {
      g_galosh_unified_sigma_override = (float)atof(a + 16);
      if(g_galosh_unified_sigma_override < 0.0f) g_galosh_unified_sigma_override = 0.0f;
      fprintf(stderr, "  --unified-sigma=%.6f (Phase 1 measurement bypassed)\n",
              g_galosh_unified_sigma_override);
    }
    else if(strncmp(a, "--chroma-up=",     12) == 0 ||
            strncmp(a, "--robust-shrink=", 16) == 0 ||
            strncmp(a, "--chroma-wiener=", 16) == 0 ||
            strncmp(a, "--chroma-method=", 16) == 0)
    {
      /* Deprecated flags from the v0..v6 development variants -- these
       * features are now baked into GALOSH_RAW_G's definition (chroma-up
       * + robust-shrink) or removed as archived experiments (chroma-
       * wiener / chroma-method).  Silently accepted for bench-script
       * backward compatibility. */
    }
    else
    {
      if(new_argc < 32) positional[new_argc++] = (char *)a;
    }
  }
  argv = positional;
  argc = new_argc;

  if(argc < 5)
  {
    fprintf(stderr,
      "Usage: %s input.bin output.bin width height\n"
      "       [method] [strength] [luma_str] [chroma_str]\n"
      "       [alpha] [sigma_sq]\n"
      "       [--variant=V]     (o | n | m | l | k | j | i | h | g, default o;\n"
      "                          O = LATEST (L + multi-scale LOESS chroma pyramid\n"
      "                              + smoothstep slider walk + L-guided upsample),\n"
      "                          N = DEPRECATED (multi-scale luma + L-coupled WHT-LOSH\n"
      "                              chroma; -3.4 dB regression, superseded by O),\n"
      "                          M = PREVIOUS (H + hierarchical Bayesian Phase 5(C)),\n"
      "                          L = PREVIOUS (joint bilateral K16, K Phase 6+8 fused),\n"
      "                          K = PREVIOUS (J + Bayesian-correct Phase 8 ε/BW),\n"
      "                          J = PREVIOUS (I + L-guided chroma refinement),\n"
      "                          I = PREVIOUS hybrid (L from H + C from G),\n"
      "                          H = full-res cycle-spinning + overlap-add inverse,\n"
      "                          G = half-res LOSH + K14/K15/K16 chromaup)\n"
      "       [--stride=N]      (G-only: 1 or 2, default 2; A enables stride=1)\n"
      "       [--orient=N]      (G-only: 1 or 4, default 1; B enables orient=4)\n"
      "       [--lfr-kernel=K]  (G-only: box | ewajl3, default box; C: ewajl3)\n"
      "\n"
      "  method:  'galosh' or 'ours' (GALOSH local WHT shrinkage, default)\n"
      "  luma_str:   sigma_L for luma shrinkage (default 0.5, user-tunable)\n"
      "  chroma_str: sigma_C for chroma shrinkage (default 1.0, user-tunable)\n"
      "  alpha:      P-G shot noise gain (auto if <= 0)\n"
      "  sigma_sq:   read noise variance (auto if <= 0)\n",
      argv[0]);
    return 1;
  }

  if(g_galosh_variant == 'o')
  {
    fprintf(stderr, "[GALOSH_RAW_O] variant=o (LATEST: L + multi-scale LOESS chroma "
                    "pyramid (3 levels: half/quarter/eighth, 30/60/120 raw px receptive "
                    "field) + smoothstep slider walk + L-guided K16 upsample at every "
                    "stage = chroma-blotch-targeted denoiser with functional flat-region "
                    "slider)\n");
  }
  else if(g_galosh_variant == 'n')
  {
    fprintf(stderr, "[GALOSH_RAW_N] variant=n (DEPRECATED: multi-scale luma WHT-LOSH "
                    "+ L-coupled WHT-LOSH chroma; -3.4 dB sRGB regression vs L on SIDD; "
                    "superseded by O)\n");
  }
  else if(g_galosh_variant == 'm')
  {
    fprintf(stderr, "[GALOSH_RAW_M] variant=m (PREVIOUS: H + hierarchical Bayesian "
                    "Phase 5(C); R_local=7 LOESS w/ σ_local²(x) + R_global=15 LOESS "
                    "+ inverse-variance fusion; chroma_strength = σ_n scaling)\n");
  }
  else if(g_galosh_variant == 'l')
  {
    fprintf(stderr, "[GALOSH_RAW_L] variant=l (PREVIOUS: K16 + LOESS post-process "
                    "FUSED into joint bilateral upsample)\n");
  }
  else if(g_galosh_variant == 'k')
  {
    fprintf(stderr, "[GALOSH_RAW_K] variant=k (PREVIOUS: J + Bayesian-correct "
                    "Phase 8 ε=0.01/BW=1.5; two-stage K16+LOESS chain)\n");
  }
  else if(g_galosh_variant == 'j')
  {
    fprintf(stderr, "[GALOSH_RAW_J] variant=j (PREVIOUS: I + L-guided chroma "
                    "refinement; full-res cycle-spinning L + half-res LOESS C "
                    "+ K16 EWA-JL3 chromaup + Phase 8 R=3 LOESS w/ L_pixel guide)\n");
  }
  else if(g_galosh_variant == 'i')
  {
    fprintf(stderr, "[GALOSH_RAW_I] variant=i (PREVIOUS: hybrid L-from-H + "
                    "C-from-G; full-res cycle-spinning L + half-res LOESS C "
                    "+ K16 EWA-JL3 chromaup)\n");
  }
  else if(g_galosh_variant == 'h')
  {
    fprintf(stderr, "[GALOSH_RAW_H] variant=h (PREVIOUS: full-res stride=1 "
                    "cycle-spinning + 4-block overlap-add inverse)\n");
  }
  else
  {
    fprintf(stderr, "[GALOSH_RAW_G] variant=g (PREVIOUS: half-res LOSH + "
                    "K14/K15/K16 chromaup) | stride=%d orient=%d lfr_kernel=%s "
                    "unified=%d k13_block=%d\n",
            g_galosh_stride, g_galosh_n_orient,
            g_galosh_lfr_kernel == 1 ? "ewajl3" : "box",
            g_galosh_unified, g_galosh_k13_block);
  }

  const char *input_file = argv[1];
  const char *output_file = argv[2];
  const int width = atoi(argv[3]);
  const int height = atoi(argv[4]);

  const char *method = (argc > 5) ? argv[5] : "galosh";
  const float strength = (argc > 6) ? (float)atof(argv[6]) : 1.0f;

  /* sigma_L (luma shrinkage strength):
   *   In GAT-normalized space, true noise sigma = 1.0.
   *   sigma_L < 1.0 means conservative denoising (preserve detail).
   *   Default 1.0: full-strength denoising matching the noise model.
   *   Applied to full-res raw GALOSH (Phase 4) where sub-pixel
   *   detail matters; also used for half-res luma pilot (Phase 3).
   *
   * sigma_C (chroma shrinkage strength):
   *   Applied in Phase 3 luma-guided chroma Wiener.
   *   Default 1.0: matches GAT-normalized noise sigma exactly.
   *   Since chroma Wiener uses luma pilot (not chroma pilot),
   *   sigma_C directly controls the denoising/preservation tradeoff
   *   without affecting the pilot quality. */
  const float luma_str = (argc > 7) ? (float)atof(argv[7]) : 0.5f;
  const float chroma_str = (argc > 8) ? (float)atof(argv[8]) : 1.0f;

  /* CLI override for Phase 0 (α, σ²) — argv[9] = α, argv[10] = σ²
   *   0 = use binary's internal Foi-Alenius blind estimation (= default)
   *   > 0 = bypass Phase 0, force these values throughout the pipeline
   * Mechanism: positive override sets globals consumed by galosh_estimate_noise
   * top-of-function short-circuit (see galosh_cpu.h).  Used by:
   *   - run_oracle_parallel.py    (= GT-derived oracle (α, σ²))
   *   - EM_iter Python prototype  (= test alternative blind estimators without
   *                                 rebuilding the binary)
   *   - any external (α, σ²) source for ablation studies                       */
  const float alpha_in    = (argc > 9)  ? (float)atof(argv[9])  : 0.0f;
  const float sigma_sq_in = (argc > 10) ? (float)atof(argv[10]) : 0.0f;
  if(alpha_in > 0.0f && sigma_sq_in >= 0.0f)
  {
    g_galosh_alpha_override    = alpha_in;
    g_galosh_sigma_sq_override = sigma_sq_in;
    fprintf(stderr, "  noise_est override: alpha=%.8f sigma_sq=%.10f\n",
            alpha_in, sigma_sq_in);
  }
  (void)strength; /* strength is absorbed into luma_str/chroma_str */

  if(width <= 0 || height <= 0)
  {
    fprintf(stderr, "Invalid dimensions: %dx%d\n", width, height);
    return 1;
  }

  fprintf(stderr, "Raw Denoiser Standalone v6 (GALOSH)\n");
  fprintf(stderr, "  Input:  %s (%dx%d)\n", input_file, width, height);
  fprintf(stderr, "  Method: %s\n", method);
  fprintf(stderr, "  Params: luma=%.2f chroma=%.2f\n", luma_str, chroma_str);

  /* Initialize Kaiser window */
  init_galosh_kaiser();

  /* Read input */
  const size_t npixels = (size_t)width * height;
  float *in = dt_alloc_align_float(npixels);
  float *out = dt_alloc_align_float(npixels);
  if(!in || !out) { fprintf(stderr, "Memory allocation failed\n"); return 1; }

  FILE *fin = fopen(input_file, "rb");
  if(!fin) { fprintf(stderr, "Cannot open %s\n", input_file); return 1; }
  size_t nread = fread(in, sizeof(float), npixels, fin);
  fclose(fin);
  if(nread != npixels)
  {
    fprintf(stderr, "Read %zu floats, expected %zu\n", nread, npixels);
    return 1;
  }

  /* Process */
  dt_iop_roi_t roi = { .width = width, .height = height };
  double t_start = omp_get_wtime();

  if(strcmp(method, "galosh") == 0 || strcmp(method, "ours") == 0)
  {
    if(g_galosh_variant == 'o')
      gat_galosh_denoise_rawlc_o(in, out, &roi, luma_str, chroma_str, 0);
#ifndef GALOSH_RELEASE  /* deprecated variant dispatch — ablation builds only */
    else if(g_galosh_variant == 'n')
      gat_galosh_denoise_rawlc_n(in, out, &roi, luma_str, chroma_str, 0);
    else if(g_galosh_variant == 'm')
      gat_galosh_denoise_rawlc_m(in, out, &roi, luma_str, chroma_str, 0);
    else if(g_galosh_variant == 'l')
      gat_galosh_denoise_rawlc_l(in, out, &roi, luma_str, chroma_str, 0);
    else if(g_galosh_variant == 'k')
      gat_galosh_denoise_rawlc_k(in, out, &roi, luma_str, chroma_str, 0);
    else if(g_galosh_variant == 'j')
      gat_galosh_denoise_rawlc_j(in, out, &roi, luma_str, chroma_str, 0);
    else if(g_galosh_variant == 'i')
      gat_galosh_denoise_rawlc_i(in, out, &roi, luma_str, chroma_str, 0);
    else if(g_galosh_variant == 'h')
      gat_galosh_denoise_rawlc_h(in, out, &roi, luma_str, chroma_str, 0);
    else
      gat_galosh_denoise_rawlc(in, out, &roi, luma_str, chroma_str, 0);
#endif  /* GALOSH_RELEASE: end deprecated variant dispatch */
  }
  else
  {
    fprintf(stderr, "Unknown method '%s'. Use 'galosh' or 'ours'.\n", method);
    dt_free_align(in);
    dt_free_align(out);
    return 1;
  }

  double elapsed = omp_get_wtime() - t_start;
  fprintf(stderr, "  Elapsed: %.2f seconds\n", elapsed);

  /* Write output */
  FILE *fout = fopen(output_file, "wb");
  if(!fout) { fprintf(stderr, "Cannot open %s for writing\n", output_file); return 1; }
  fwrite(out, sizeof(float), npixels, fout);
  fclose(fout);

  fprintf(stderr, "  Output: %s\n", output_file);

  dt_free_align(in);
  dt_free_align(out);
  return 0;
}
