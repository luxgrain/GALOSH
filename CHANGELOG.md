# Changelog

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

## v0.1.0 — first public release (2026, unreleased)

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
