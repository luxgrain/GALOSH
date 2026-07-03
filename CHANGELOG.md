# Changelog

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
