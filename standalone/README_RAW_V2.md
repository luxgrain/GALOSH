# GALOSH RAW V2 — canonical pipeline

Fully-blind RAW Bayer denoiser (Generalized Anscombe LOcal SHrinkage).
**Canonical algorithm = GALOSH_RAW_O**: GAT-normalized overlapping 2×2 WHT
luma shrinkage (BayesShrink-MAD + empirical Wiener, cycle-spun) + 3-level
multi-scale LOESS chroma pyramid with L-guided joint-bilateral upsample.
No block-matching, no sorting, no training, no noise profile.

## Four precisions (the paper "GALOSH RAW V2")

| precision | source | binary | role |
|---|---|---|---|
| **CPU FP32** (o) | `galosh_raw_cpu.c` | `galosh_raw_cpu.exe` | reference / offline quality |
| **CPU INT32** (r32) | `galosh_raw_cpu_int.c` | `galosh_raw_cpu_int.exe` | Q11.20 fixed-point reference (FP-free) |
| **GPU FP32** (o32) | `galosh_raw_gpu.c` + `galosh.cl` | `galosh_raw_gpu.exe` | real-time (commodity GPU) |
| **GPU INT16** (i16) | `galosh_int_*.{clh,cl}` | `galosh_int_pipe_test.exe` | bit-exact vs r32 at INT32 storage; INT16-narrowed storage = near-lossless (~58-65 dB); fixed-point streaming **by design** (32 KB LDS, no FP) |

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
./galosh_raw_cpu.exe   in.bin out.bin W H galosh 1.0 1.0 1.0 0 0   # CPU FP32, blind
./galosh_raw_gpu.exe   in.bin out.bin W H 1.0 1.0 1.0 0 0 0        # GPU FP32 (o32 default)
./galosh_raw_cpu_int.exe in.bin out.bin W H galosh 1.0 1.0 1.0 0 0 --variant=r32
```
Input/output = raw float32, row-major, single-channel Bayer in [0,1]. Noise is
estimated blind (Foi–Alenius α/σ²); pass `0 0` for alpha/sigma to estimate.

## Deprecated lineage (ablation only)

`--variant=g,h,i,j,k,l,m,n` are the PREVIOUS perceptual-evolution lineage that O
supersedes (each fixed a chroma edge-fringe / blotch artifact detectable by
LPIPS/DISTS/NIQE; `n` = −3.4 dB). They are `#ifndef GALOSH_RELEASE`-guarded in
`galosh_raw_cpu.c` — present for forensic/ablation, excluded from release builds.
`l` is the single exception: kept as O's small-input (<32 px) fallback.
GPU `--variant=g` (GALOSH_RAW_G) is likewise DEPRECATED (chroma fringe).
