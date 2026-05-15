# GALOSH-RAW INT port — variant archive

Production source lives at `standalone/galosh_raw_cpu_int.{c,h}` (build target `galosh_raw_cpu_int.exe`).  This directory archives historical variants for forensic reference; **do not delete** per `feedback_keep_deprecated_variants` policy.

## File map

| variant | size (.c) | source pair | D60-3_ISO400 | status |
|---|---:|---|---:|---|
| `pre_v0_201213` | 118344 | pre-cleanup C, .h shared w/ v0 | (not tested) | earlier WIP, 2026-05-13 20:12 JST |
| `v0_baseline` | 118621 | **.c + .h** | **42.30 dB** ✓ | **canonical**; mirror of `standalone/` |
| `pre_v1_205413` | 119677 | post-v0 build, pre-v1 edit | (not tested) | 2026-05-13 20:54 JST intermediate |
| `v1_bug11_broken` | 120375 | .c + .h | 21.01 dB ✗ | **Bug 11 introduction**, crw catastrophic |
| `v2_bug11_more_broken` | 121356 | .c + .h | 21.01 dB ✗ | Bug 11 elaborated |
| `v3_failed_revert` | 119028 | .c + .h | 21.01 dB ✗ | revert that *failed silently* |

## v0 → v1 regression — root cause

v1 (formerly called "v14" in informal logs) replaced `fxp_gat_precompute` in `galosh_cpu_int.h` with a wrapper:

```c
/* v1 form — silently scales sigma_sq by 1024 */
static inline void fxp_gat_precompute(fxp_gat_params *p,
                                      fxp32 alpha, fxp32 sigma_sq) {
  fxp_gat_precompute_scaled(p, alpha, sigma_sq * 1024);   // ← bug
}
```

v0 (canonical) form computes the params directly from real-domain sigma_sq:

```c
/* v0 form — direct computation, no implicit scaling */
static inline void fxp_gat_precompute(fxp_gat_params *p,
                                      fxp32 alpha, fxp32 sigma_sq) {
  p->alpha = alpha;
  p->sigma_sq = sigma_sq;
  /* ... full algebraic precompute ... */
}
```

Downstream Phase 1+ assumed real-domain sigma_sq, so the v1 wrapper produced an effective σ² 1024× too large.  Most patches survived because the downstream computations clip / floor at safe values, but Canon crw (D60 sensor, mid-ISO clean signal) was catastrophically miscompensated, with 99.5%+ of pixels clamping to 0 after the inverse GAT.

## v3 "failed revert" — what went wrong

v3 attempted to revert Bug 11 by switching `main()` from `fxp_gat_precompute_scaled(...)` back to `fxp_gat_precompute(...)` and changing the σ²-scaled convention back to unscaled at Phase 0 output.  But the `fxp_gat_precompute` *body itself* remained the v1 silently-scaling wrapper.  Net effect: σ² was unscaled at Phase 0 → multiplied by 1024 by the wrapper → same broken state.

The lesson: when reverting a bug, check both the **call sites** and the **function body**.  Diff against the original git/file-history snapshot rather than relying on memory.

## Recovery (2026-05-16)

v0 source was recovered from Windows file history (`(5).c` at 2026-05-13 20:33 JST + `(4).h` at 20:00 JST).  Rebuild is bit-identical to the surviving `galosh_int_v0_baseline.exe`.  Md5 of `D60-3_ISO400` output matches the original binary's output exactly.

## Benchmark numbers (v0 / canonical)

- **SIDD val (1280 patches)**: PSNR 49.10 / SSIM 0.9861 vs FP32 49.47 / 0.9865
- **RawNIND (1493 patches)**: PSNR 29.370 / SSIM 0.7527 vs FP32 30.43 / 0.7907
- Gap: -0.37 dB SIDD, -1.06 dB RawNIND (= essentially FP32-equivalent on SIDD)

## How to extend (future iterations)

1. Snapshot the current `standalone/galosh_raw_cpu_int.{c,h,exe}` into this directory as `*_vN_<tag>.{c,h,exe}` **before** starting any modification.
2. Build + test + commit BEFORE editing further.
3. If a change regresses on `D60-3_ISO400` (canary), revert immediately and bisect against the most recent passing variant.
