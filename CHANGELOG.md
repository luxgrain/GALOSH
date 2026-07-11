# Changelog

## v0.4.0 — GALOSH-420: planar YCbCr containers + Vulkan YUV engine (2026-07-11)

### GALOSH-420 planar front-end (all three backends)
- Native 4:2:0 / 4:2:2 / 4:4:4 / 4:0:0 planar integer containers,
  format-preserving in/out: `--pix --depth=8..16 --range=full|limited
  --matrix=bt601|bt709|bt2020|custom:Kr,Kb
  --eotf=srgb|g22|g24|bt709|hlg|pq|linear --siting=center|left|topleft`
  (DVD / Blu-ray / HDR10 / HLG / JPEG profiles). One shared header
  (`standalone/galosh_yuv420.h`) serves CPU, OpenCL and Vulkan — flag
  vocabulary and siting phases cannot drift between backends.
- Design locked by A/B experiment (SIDD 80 scenes × 3 sitings × 2 matrices,
  `benchmark/scripts/ab_yuv420.py`): chroma denoised at its NATIVE
  half-resolution lattice with a siting-phased downsampled-Y guide beats
  upsample-to-444-first by +0.3–0.5 dB Cb/Cr at ~4× lower chroma cost;
  siting itself does not affect denoise quality; only guide-phase mismatch
  costs (−0.13/−0.19 dB). Spec: `docs/yuv420_frontend_spec.md`; the
  phase-matched guide is machine-verified by a built-in affine-field
  selftest (`--selftest-phase`, all backends).
- Luma is denoised at full resolution and is chroma-independent by
  construction; the legacy sRGB float path is byte-identical with the new
  flags off (identity harness). CPU reproduces the design rig at ±0.000 dB
  (16-bit); validation harness `benchmark/scripts/bench_yuv420_planar.py`.

### Vulkan YUV engine (`standalone/vk/galosh_yuv_vk.exe`, new)
- 10 new SPIR-V kernels + the raw engine's LOSH / inverse-LUT shaders
  reused verbatim (port blueprint: `standalone/vk/YUV_BLUEPRINT.md`).
  FP16 inter-phase storage contract, zero mid-frame readbacks,
  watchdog-safe banded submissions (8K completes on every GPU).
- Parity: 67.3 dB vs the CPU FP32 reference on NVIDIA / Arc / AMD iGPU,
  95 dB cross-GPU agreement.
- Video amortization ported from the raw engine: `--noise=hold|every:N` +
  state file + 32 KB LUT sidecar; a held frame skips both exact-median
  noise estimators and is **bit-identical** to fit on the same frame.
  Measured (RTX 4070 Ti, GPU time): 4:2:0 video steady state
  **1080p 2.6 ms (391 fps), 4K 10.2 ms (98 fps)**; full table in README.
- OpenCL YUV host refactored to a buffer-based core (byte-identical legacy
  CLI) and carries the same `--pix` front-end as the FP32 portable
  reference (NVIDIA 4K 444: OpenCL 79 ms vs Vulkan 50 ms fit / 11 ms hold).

## v0.3.0 — V2 engine: Vulkan compute + video modes + cross-vendor GPU (2026-07-11)

### Vulkan compute port (`standalone/vk/`, new)
- Full raw pipeline as one vendor-agnostic GLSL shader set (43 SPIR-V
  kernels + `galosh_vk.exe`, CPU-exe-compatible CLI). Verified on NVIDIA
  RTX 4070 Ti, Intel Arc A310 and an AMD Radeon iGPU at 69.7–70.6 dB
  against the CPU FP32 reference — one code path, no per-GPU branching.
- FP16 inter-phase storage contract (compute stays FP32; explicit IEEE
  RNE stores — fixes AMD's round-toward-zero f16 conversion), FP64
  eliminated via Neumaier–Kahan compensated FP32 (unlocks Arc).
- Subgroup-cooperative 8×8 WHT shrinkage (one block per subgroup,
  coefficients register-resident): quality mode 4–18× faster than the
  literal transcription; **4K 91.7 fps / 8K 22.5 fps quality, 4K 216 fps
  fast on the 4070 Ti** (see the README speed table for all three GPUs).
- Windows-watchdog-safe execution at any resolution: adaptive banded
  per-submission dispatch with a bounded probe (8K completes even on the
  slowest iGPU; TDR root cause documented in-source).
- Video amortization: `--noise=hold|every:N|ema:B` + state file + 32 KB
  inverse-GAT LUT cache — held frames are bit-identical to fit frames.

### CPU reference — V2 flags (flags-off output byte-identical, enforced
### by the new identity harness `standalone/tests/v2_identity.py`)
- `--noise=fit|hold|every:N|ema:B --noise-state=FILE` — formalized
  noise-model injection (video semantics oracle).
- `--wht=4` fast luma mode (−15% CPU; quality label: video-grade — trades
  PSNR on high noise) and `--upsample=fast` guided bilinear (−20% CPU;
  measured quality-neutral on the full benchmark). Per-flag speed/quality
  labels in `standalone/README.md`; both exposed in the dist wrapper.
- `--f16-storage` FP16-storage oracle (77–83 dB vs FP32 = near-lossless)
  and `docs/dataflow_spec.md`, the precision/dataflow porting reference.

### OpenCL cross-vendor fix (`galosh.cl` + hosts)
- FP64 removed (compensated FP32, same scheme as Vulkan): the kernel
  program now builds and runs on Intel Arc.
- Fixed a silent AMD failure: the tiled LOESS workgroup (24×24 = 576)
  exceeded AMD's max workgroup size 256, the dispatch failed with -54 and
  the pipeline continued on garbage. Tile is now 16×16 (NVIDIA output
  byte-identical); dispatch errors are fatal now instead of ignored.
- All three GPUs at 70.55 dB parity vs the CPU reference; NVIDIA quality
  fit 17.6 → 11.8 ms. OpenCL stays the portable reference (quality mode,
  fit-every-frame); Vulkan is the performance path (2–19× like-for-like).

### Distribution
- `GALOSH_RAW_win64.zip` rebuilt with the fixed OpenCL engine (`--gpu`
  now works on NVIDIA / Intel Arc / AMD) and the `--wht/--upsample`
  wrapper flags; `GALOSH_YUV_win64.zip` unchanged (CPU engine).

## v0.2.0 — distributable Windows builds (2026-07-09)

End-user distribution ZIPs (attached to the GitHub release):
`GALOSH_RAW_win64.zip` (DNG in → denoised DNG out) and
`GALOSH_YUV_win64.zip` (sRGB PNG in → denoised PNG out).

- Drag & drop onto the exe (defaults `l=1 c=1`) or CLI
  `galosh_raw.exe -l 0.8 -c 1.2 [--gpu] file.dng`; output written next to
  the original as `<name>_GALOSH_l<L>_c<C>.dng/.png`, original untouched.
- Fully blind: no noise profile, no training, any 2×2 Bayer CFA DNG
  (X-Trans / monochrome / linear DNG rejected with a clear error).
- Metadata copied completely from the original via bundled ExifTool
  (EXIF / GPS / MakerNotes / XMP / ICC, plus the protected DNG
  calibration tags — ColorMatrix1/2, CalibrationIlluminant1/2,
  AsShotNeutral, ForwardMatrix, lens opcodes — which bulk copy skips).
  Deliberately not copied: `NoiseProfile` (stale after denoising) and
  `LinearizationTable` (output data is already linearized).
- New tracked sources: `tools/dist/galosh_dng.py`, `tools/dist/galosh_png.py`
  (PyInstaller onefile wrappers; UTF-8-safe subprocess handling on CJK
  locales; per-channel black/white normalization; tifffile DNG skeleton).
- SPL letter prior-art hardening (refs [22]–[26]: YOND, Noise2VST,
  Dubois 2005, Hirakawa 2007, Park 2009) — manuscript as submitted to
  IEEE Signal Processing Letters on 2026-07-09.

## v0.1.0 — first public release (2026-07-04)

Initial public snapshot of GALOSH: a blind, training-free, search-free classical
denoiser for raw Bayer and sRGB/YUV images.

### Canonical implementation (standalone CLI, `standalone/`)
- GALOSH-RAW: `galosh_raw_cpu.exe` (FP32 reference), `galosh_raw_cpu_int.exe`
  (INT32 fixed-point reference), `galosh_raw_gpu.exe` (OpenCL FP32 `o32` +
  streaming INT16 `i16`).
- GALOSH-YUV/RGB: `galosh_yuv_cpu.exe` (FP32 reference), `galosh_yuv_gpu.exe`
  (OpenCL FP32).
- Kernel sources resolve relative to the executable (runs from any working
  directory).
- GPU CLI honors an externally supplied `(alpha, sigma^2)` noise model
  (previously the arguments were silently ignored and the blind estimate was
  always used).

### Harness
- `Makefile` + portable `build.sh` (targets: `raw`, `yuv`, `gpu`, `test`,
  `bench-small`, `check-no-int64`).
- Smoke tests (`tests/run_smoke.sh`): constant / random / odd / small /
  high-noise / near-black inputs through the RAW and YUV paths (+ GPU smoke
  with OpenCL device probing).
- `tests/run_bench_small.sh`: seeded synthetic Poisson-Gaussian micro-benchmark
  (CSV/JSON output), dataset-free.
- `check_no_int64.sh`: fails if the INT fixed-point shipping path uses a native
  64-bit (or floating) type; profiling/timing exemptions are marked
  `no64-exempt` in-source. Wide intermediates use int32 split-multiply /
  paired-int32 patterns.

### Benchmarks (reproduction scripts, `benchmark/scripts/`)
- Raw domain: SIDD Medium (80 full frames) + RawNIND (1493 crops), all blind.
- sRGB domain: SIDD Medium sRGB (full frame) + RawNIND-rendered sRGB, all blind.
- Third-party datasets are NOT redistributed; see README for download/prep.
