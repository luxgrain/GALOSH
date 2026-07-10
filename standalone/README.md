# GALOSH standalone — canonical RAW pipeline

Fully-blind RAW Bayer denoiser (Generalized Anscombe LOcal SHrinkage).
**Canonical algorithm = GALOSH_RAW_O**: GAT-normalized 2×2 WHT mosaic
decomposition + overlapping 8×8 WHT luma shrinkage (BayesShrink-MAD +
empirical Wiener, cycle-spun) + 3-level
multi-scale LOESS chroma pyramid with L-guided joint-bilateral upsample.
No block-matching, no sorting, no training, no noise profile.

## Four precisions

| precision | source | binary | role |
|---|---|---|---|
| **CPU FP32** (o) | `galosh_raw_cpu.c` | `galosh_raw_cpu.exe` | reference / offline quality |
| **CPU INT32** (r32) | `galosh_raw_cpu_int.c` | `galosh_raw_cpu_int.exe` | Q11.20 fixed-point reference (FP-free) |
| **GPU FP32** (o32) | `galosh_raw_gpu.c` + `galosh.cl` | `galosh_raw_gpu.exe` | fast GPU path (measured 11.8 ms @1080p / 39.3 ms @4K on an RTX 4070 Ti; see the GPU-speed table in the top-level README) |
| **GPU INT16** (i16) | `galosh_int_*.{clh,cl}` | `galosh_int_pipe_test.exe` | bit-exact vs r32 at INT32 storage; INT16-narrowed storage = near-lossless (~58-65 dB); fixed-point streaming **by design** (32 KB LDS, no FP) |

Vulkan port (FP32 compute / FP16 inter-phase storage): `vk/` — 43 SPIR-V
shaders + `galosh_vk.exe`, CPU-exe-compatible CLI; see the top-level README
GPU-speed section.

At INT32 storage the GPU pipeline is bit-exact against the r32 CPU reference
end-to-end (verified on SIDD/RawNIND full frames, 2026-07-04). Narrowing the
line buffers to the deployed INT16 storage formats (`i16 lf=5 cf=9`, luma
Q10.5 / chroma Q6.9) leaves the two near-lossless (~58-65 dB PSNR on full
frames — the residual is exactly this storage quantization); both are
~-0.4 dB vs FP32 (essentially equivalent quality). The INT16 design supports
the fixed-point mapping claim (not its GPU speed, which carries paired-int32
emulation overhead).

## Build

```sh
# Clean public/paper build — canonical O only (CPU FP32):
gcc -O2 -fopenmp -std=c11 -DGALOSH_RELEASE galosh_raw_cpu.c -o galosh_raw_cpu.exe -lm
# GPU FP32 (default --variant=o32):
gcc -O2 -std=c11 galosh_raw_gpu.c -o galosh_raw_gpu.exe -lOpenCL -lm
# CPU INT32 reference:
gcc -O2 -std=c11 galosh_raw_cpu_int.c -o galosh_raw_cpu_int.exe -lm
```

`-DGALOSH_RELEASE` excludes the deprecated variant lineage (see below); the
canonical O path is byte-identical with or without it (verified). Omit it for
ablation builds that expose `--variant=g..n`.

## Usage

```sh
./galosh_raw_cpu.exe     in.bin out.bin W H galosh 1.0 1.0 1.0 0 0   # CPU FP32, blind
./galosh_raw_gpu.exe     in.bin out.bin W H 1.0 1.0 1.0 0 0 0        # GPU FP32 (o32 default)
./galosh_raw_cpu_int.exe in.bin out.bin W H galosh 1.0 1.0 1.0 0 0   # CPU INT32 (single pipeline)
```
Input/output = raw float32, row-major, single-channel Bayer in [0,1]. Noise is
estimated blind (per-image Poisson–Gaussian α/σ² fit); pass `0 0` for
alpha/sigma to estimate. On Windows/MSYS2, run with `C:\msys64\ucrt64\bin` on
`PATH` (the OpenMP runtime `libgomp-1.dll` lives there). Set `GALOSH_VERBOSE=1`
for per-phase diagnostics (all exes are quiet by default).

## V2.0 options (CPU FP32 exe + Vulkan exe) — independent, freely combinable

All flags default to the canonical published pipeline (flags-off output is
byte-identical to it — enforced by `tests/v2_identity.py`). Speed/quality
below are MEASURED: speed on 12 MP RawNIND full frame + SIDD full frames
(CPU), quality on the full campaign (RawNIND 1493 crops + SIDD Medium 80
full frames, `benchmark/results_*/_metrics_v2.json`).

| Flag | Default | Speed (CPU) | Quality (measured) | Use for |
|---|---|---|---|---|
| `--upsample=fast` | `jinc` | **−20%** | **neutral on both datasets** (ΔPSNR ≤ +0.02 dB, ΔLPIPS ≤ 0.0006); ring-free by construction | safe speed win, photos included |
| `--wht=4` | `8` | −15% | high noise (SIDD): **−1.21 dB / +0.047 LPIPS**; low-ISO (RawNIND): neutral PSNR, LPIPS slightly better | video / speed-critical only |
| both of the above | — | **−36%** (SIDD 10.3→6.5 s/frame) | = `--wht=4` quality | real-time-leaning pipelines |
| `--noise=every:N` / `hold` / `ema:B` (+`--noise-state=F`) | `fit` (per-frame) | skips re-estimation on held frames (full payoff — LUT reuse — is realized in the Vulkan port) | held frames: byte-identical for a fixed model; `ema` tracks slow drift | video loops, fixed camera/ISO |
| `--f16-storage` | off | none (oracle, slightly slower) | 77–83 dB vs FP32 = invisible | GPU-FP16 parity reference only |
| `luma_str` / `chroma_str` (args 7/8) | 1.0 / 1.0 | — | user taste (grain vs. smoothness) | tuning |

GPU note: on the OpenCL/Vulkan pipeline the upsample stage is ~1% of frame
time (fast upsample buys nothing there), while the 8×8 luma stage is 42–58%
(`--wht=4` buys the most there). The Vulkan exe (`vk/galosh_vk.exe`,
CPU-exe-compatible CLI) accepts the same flags; the OpenCL exe stays V1
(quality mode, fit-every-frame).

## Deprecated lineage (ablation only)

`--variant=g,h,i,j,k,l,m,n` are the PREVIOUS perceptual-evolution lineage that O
supersedes (each fixed a chroma edge-fringe / blotch artifact detectable by
LPIPS/DISTS/NIQE; `n` = −3.4 dB). They are `#ifndef GALOSH_RELEASE`-guarded in
`galosh_raw_cpu.c` — present for forensic/ablation, excluded from release builds.
`l` is the single exception: kept as O's small-input (<32 px) fallback.
GPU `--variant=g` (GALOSH_RAW_G) is likewise DEPRECATED (chroma fringe).
