# GALOSH

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21187219.svg)](https://doi.org/10.5281/zenodo.21187219)
[![License](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)

**Blind, training-free denoising of raw Bayer and sRGB/YUV images by
parallel-friendly local shrinkage.**

Preprint: [PDF (release asset)](https://github.com/luxgrain/GALOSH/releases/download/v0.1.0/galosh_paper.pdf)
· [Zenodo (DOI 10.5281/zenodo.21187219)](https://doi.org/10.5281/zenodo.21187219)
· arXiv: in progress

GALOSH (Generalized Anscombe LOcal SHrinkage) is a **classical** image denoiser
redesigned for modern parallel hardware:

- **Blind** — the Poisson–Gaussian noise model is estimated from the input
  image itself; no calibration data, no per-image noise oracle.
- **Training-free** — no learned weights; generalizes across sensors and
  datasets by construction.
- **Search-free** — unlike BM3D/NLM there is no block matching or neighborhood
  search: every stage is local, regular, and data-independent, so the same
  fixed computation graph runs for every pixel of every image.
- **Multi-domain** — one shared core serves two thin front-ends:
  **GALOSH-RAW** (Bayer mosaic) and **GALOSH-YUV/RGB** (rendered sRGB images).
- **Fast** — the fixed structure parallelizes trivially: 7x-650x faster than
  trained-network baselines on the same GPU at full benchmark size (the ratio
  depends on the domain and baseline), and practical on plain CPUs.

On SIDD Medium and RawNIND (raw and sRGB, all methods blind), GALOSH is
consistently the strongest blind, training-free method — ahead of the
BM3D/NLM family even when those are given an oracle noise level — while
trained networks remain ahead in their own training domain (reported honestly
in the paper).

## Algorithm summary

**GALOSH core** (shared by both domains)
1. *Blind noise estimation* — robust per-image fit of
   `Var[x] = α·E[x] + σ²` from local mean/variance statistics.
2. *Variance stabilization* — generalized Anscombe transform (GAT); exact
   unbiased inverse on output.
3. *Luminance* — two-pass local shrinkage on overlapping 8×8 Walsh–Hadamard
   blocks (robust-MAD BayesShrink pilot → empirical Wiener), cycle-spun,
   windowed overlap-add.
4. *Chrominance* — luminance-guided local linear regression (Y-guided LOESS
   with a bilateral kernel, multi-scale residual pyramid), clamped to the
   local input chroma range.

The luma/chroma split is deliberate and asymmetric: luminance noise reads as
grain and coexists with texture (conservative shrinkage), chroma noise reads
as color blotches (aggressive smoothing, anchored to luma structure). The two
paths expose independent strength controls.

**GALOSH-RAW** adds the raw-only stages: CFA-aware achromatic dark reference,
stride-1 cycle-spun 2×2 WHT decomposition of each Bayer quad into full-res
luma + half-res chroma, and a luminance-guided joint-bilateral EWA-jinc chroma
upsampling (anti-ringed). Input/output = linear raw Bayer float32 in [0,1].

**GALOSH-YUV/RGB** adds the color-image front-end: inverse sRGB gamma →
full-range BT.709 YCbCr; Y takes GAT+shrinkage, Cb/Cr take the guided
regression at full resolution; output clamped to [0,1]. Input = sRGB float32
(HWC, [0,1]).

## Repository layout

| Path | Role |
|---|---|
| `standalone/` | **Canonical reference implementation** (CLI / exe) — the basis for the paper and all benchmarks |
| `standalone/tests/` | Smoke tests + dataset-free micro-benchmark |
| `benchmark/scripts/` | Full benchmark harness (SIDD / RawNIND, raw + sRGB) |
| `benchmark/results_*/` | Benchmark outputs (JSON metrics, PNG artifacts; regenerable, not committed) |
| `docs/paper/` | Manuscript sources (LaTeX, tables, figures) |
| `*_hp.*` | **Diagnostic probes** — see below |

**Canonical vs. archived vs. diagnostic.** The canonical pipeline is the code
in `standalone/` listed below. Superseded experimental variants are kept (not
deleted) under archive directories and clearly marked `[DEPRECATED]` in-source;
they are not part of the release build. Files named `*_hp.*`
(`galosh_raw_cpu_int_hp.c`, `galosh_cpu_int_hp.h`) are **high-precision
diagnostic probes**: they intentionally use int64/`__int128` for numerical
analysis, are excluded from the no-INT64 check, and are never included by the
canonical pipeline.

**darktable**: a darktable integration exists as a demo direction but is
**not** part of this release and is not the canonical implementation; it may
appear later on a separate branch.

## Build

Requirements: a C99/C11 compiler with OpenMP; the GPU targets additionally
need OpenCL headers + `libOpenCL`. On Windows we use MSYS2 ucrt64 gcc
(`pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-opencl-headers
mingw-w64-ucrt-x86_64-opencl-icd`).

```sh
cd standalone
make all          # CPU references: galosh_raw_cpu, galosh_raw_cpu_int, galosh_yuv_cpu
make gpu          # OpenCL: galosh_raw_gpu, galosh_yuv_gpu
make test         # smoke tests (constant/random/odd/small/high-noise/near-black)
make bench-small  # seeded synthetic micro-benchmark -> tests/bench_small_results.csv
make check-no-int64
# make is optional: ./build.sh {all|gpu|test|bench-small|check-no-int64} does the same
```

**Windows (MSYS2), exact commands.** A plain `bash` from PowerShell/cmd may
resolve to the WSL launcher and fail when no distro is installed — invoke the
MSYS2 bash explicitly, put the ucrt64 toolchain on `PATH`, and point the test
harness at a Python that has numpy (the harness reads `PYTHON`, defaulting to
`python`):

```powershell
C:/msys64/usr/bin/bash.exe -lc "export PATH=/usr/bin:/c/msys64/ucrt64/bin:$PATH; export PYTHON='<python-with-numpy>'; cd /c/<path-to>/GALOSH/standalone; bash build.sh all && bash build.sh gpu && bash tests/run_smoke.sh && bash tests/run_bench_small.sh && bash check_no_int64.sh"
```

If MSYS2's own Python lacks numpy, either `pacman -S
mingw-w64-ucrt-x86_64-python-numpy` or set `PYTHON` to any CPython 3.x that
has it.

The `.cl` kernel files must sit next to the executables (they do in-tree);
the loaders resolve them relative to the executable, so the CLI can be invoked
from any working directory.

## CLI usage

Raw Bayer (float32 `.bin`, RGGB, values in [0,1]):

```sh
./galosh_raw_cpu.exe  in.bin out.bin W H galosh 1.0 1.0 1.0 0 0
./galosh_raw_gpu.exe  in.bin out.bin W H 1.0 1.0 1.0 0 0 [cl_device]
#                                       strength luma_str chroma_str alpha sigma_sq
# alpha = sigma_sq = 0  -> fully blind (default);  positive values supply an
# external noise model and are honored on both CPU and GPU.
```

sRGB / YUV (float32 HWC `.bin`, values in [0,1]):

```sh
./galosh_yuv_cpu.exe  in.bin out.bin W H 1.0 1.0
./galosh_yuv_gpu.exe  in.bin out.bin W H 1.0 1.0 [cl_device]
#                                        strength_y strength_c
```

Notes:
- Raw input requires even W and H (Bayer quads); odd sizes are rejected with
  an error. The YUV path accepts odd sizes.
- On multi-GPU machines the OpenCL device order can vary between runs and
  non-capable iGPUs may enumerate first; if the default device fails to build
  the kernels, try `cl_device` = 1..3 (the benchmark harness probes
  automatically).

## Tests and expected output

`make test` runs 20 smoke cases through the RAW (FP32 + INT32) and YUV paths
(+ GPU when available) and ends with `SMOKE: PASS`. `make bench-small` denoises
a seeded synthetic Poisson–Gaussian pair and must report a PSNR gain for every
method (typical: raw FP32 ≈ +15 dB, raw INT32 ≈ +15 dB, YUV ≈ +7 dB on the
synthetic scene), ending with `BENCH-SMALL: PASS`.

## Benchmarks (paper reproduction)

Third-party datasets are **not redistributed**. Download and prepare:

- **SIDD Medium** (raw + sRGB pairs): from the [SIDD site](https://abdokamel.github.io/sidd/).
  The harness consumes per-scene `.npy` pairs (`<tag>_{noisy,gt}_{raw,srgb}.npy`
  + `scenes.json`); see `benchmark/scripts/unzip_sidd_full.py` and
  `bench_sidd_medium.py` headers for the expected layout.
- **RawNIND**: from the [UCLouvain Dataverse](https://doi.org/10.14428/DVN/DEQCIM)
  (`.arw` originals); `benchmark/scripts/render_rawnind.py` produces the raw
  crops and sRGB renders the harness consumes.

Then (paths at the top of each script point at the dataset root):

```sh
python benchmark/scripts/bench_raw_campaign.py --dataset sidd_medium   # raw domain
python benchmark/scripts/bench_raw_campaign.py --dataset rawnind
python benchmark/scripts/bench_yuv_srgb.py --dataset sidd --crop 0        # sRGB, full frame
python benchmark/scripts/bench_yuv_srgb.py --dataset rawnind
```

Results are written as JSON/CSV plus per-method PNG artifacts; the scripts in
`benchmark/scripts/` also regenerate the paper tables and qualitative figures.

## Fixed-point / streaming (design target, not a product)

GALOSH's search-free structure is **designed to map naturally onto fixed-point
and streaming implementations**: the computation graph is fixed, per-pixel cost
is constant (~3.4k MAC/pixel, resolution-independent), and on-chip state is
bounded by line buffers. Three measured facts back this up: at INT32 storage
the GPU streaming implementation is **bit-exact** against the INT32 CPU
reference end-to-end (verified on SIDD and RawNIND full frames); narrowing
the line buffers to the INT16 storage formats (luma Q10.5, chroma Q6.9)
leaves the two near-lossless (~58-65 dB PSNR on full frames — the residual
is exactly this storage quantization); and the shipping INT path contains
**no native 64-bit arithmetic** (`make check-no-int64`; wide intermediates
use paired-int32 / split-multiply patterns; profiling-only exemptions are
marked `no64-exempt` in-source).

We do **not** claim a completed ISP implementation, real-time operation on ISP
silicon, or a fully verified hardware datapath — those are future work.

## Known limitations

- Trained networks remain ahead in their training domain (e.g. SIDD-trained
  models on SIDD sRGB, and at high ISO on rendered sRGB); GALOSH's claim is the
  combination blind × training-free × multi-domain × speed, not SOTA quality.
- The blind estimator under-estimates strongly spatially-correlated rendered
  noise (all high-pass estimators do); the sRGB results absorb this.
- Raw path requires even dimensions; GPU kernels are tuned for NVIDIA
  (OpenCL); other vendors may fail to build the kernels (pick another
  `cl_device`).
- The INT16 fixed-point CPU reference is correctness-first (not speed-optimized);
  its GPU throughput reflects paired-int32 emulation on FP-oriented hardware,
  not ISP-native speed.

## Citing GALOSH / Who uses it?

If GALOSH is useful in your research, please cite it (`CITATION.cff` has
machine-readable metadata; the arXiv identifier will be added on release):

```bibtex
@article{galosh2026,
  author = {Sato, Yoshiro},
  title  = {{GALOSH}: Blind, Training-Free Denoising of Raw Bayer and sRGB
            Images by Parallel-Friendly Local Shrinkage},
  year   = {2026},
  note   = {arXiv preprint, identifier to be added on release}
}
```

If you use GALOSH in a product or pipeline — **or even just drew on it as a
reference for your own design** — we would love to hear about it: open an
issue, or add yourself to [`ADOPTERS.md`](ADOPTERS.md). Entirely voluntary
(Apache-2.0 imposes no reporting obligation), always appreciated.

## License and publication

- **Code:** Apache-2.0 (see `LICENSE` and `NOTICE`).
- **Paper / figures:** CC BY 4.0. Planned submission path: **arXiv → IEEE SPL →
  IPOL** (reproducible-implementation article).
- A U.S. provisional patent application covering the methods has been filed
  (App. No. 64/058,343, May 6 2026).

Copyright 2026 luxgrain.
