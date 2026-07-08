# GALOSH dataflow & precision specification (ISP porting reference)

**GALOSH データフロー・精度仕様（ISP 移植参考の正本）** — V2.0, 2026-07.

This document collects, in one place, the numeric contracts that an
ISP/hardware (or GPU FP16) port of GALOSH-RAW must honor. Everything here is
implemented and verified in the reference code; sources are cited per section.
The pipeline phases (P0 blind fit … P10 inverse) are documented in
`standalone/galosh_raw_cpu.c` and the paper (arXiv:2607.03768).

## 1. Precision ladder / 精度の階層

| Build | Storage | Compute | Role |
|---|---|---|---|
| CPU FP32 (`galosh_raw_cpu.exe`) | f32 | f32 | canonical reference; fully deterministic incl. OpenMP (verified: identity harness hashes) |
| CPU FP32 `--f16-storage` | **binary16 at contract points (§5)** | f32 | oracle for the GPU FP16 port |
| CPU INT32 "r32" (`galosh_raw_cpu_int.c`) | **Q11.20** int32 | int32 (no FP, no INT64) | fixed-point reference; the normative ISP dataflow |
| GPU INT16 "i16" (`galosh_int_*.cl`, frozen) | **INT16 line buffers (§4)**, INT32 elsewhere | int32 | streaming-feasibility evidence: bit-exact vs r32 at INT32 storage (full frames, 2026-07-04) |

Rule of thumb: **storage precision ≪ compute precision**. Both 16-bit storage
contracts (§4, §5) are near-lossless because every accumulation happens in
32-bit (int32 MAC / f32).

## 2. Fixed-point compute format (r32) / 固定小数点演算形式

Source: `standalone/galosh_cpu_int.h`.

- `fxp32` = int32, **Q11.20**: 1 sign + 11 integer + 20 fractional bits.
  `FXP_ONE = 2^20 = 1048576`; representable real range ≈ **±2048**
  (`FXP_MAX_REAL = 2047.999`). GAT-domain signals (D ≈ 7…40 for real frames)
  fit with ~5 bits of headroom.
- Multiplication: `fxp_mul` **split multiply** (hi/lo 16-bit halves), never a
  64-bit intermediate. Accumulators: paired-int32 (`fxp_acc`) where a single
  int32 could overflow.
- **No INT64 / no double anywhere in the shipping path** — enforced by
  `standalone/check_no_int64.sh` (profiling/diagnostic code is `no64-exempt`
  marked in-source). Diagnostic HP probes (`*_hp.{c,h}`) may use int64 and are
  build-excluded from release.
- Known overflow trap (fixed 2026-06-22): Phase-0 WLS weights `count×FXP_ONE`
  overflow int32 on full frames (count ~1e5). The shipping fix down-scales the
  count (`wshift`) before weighting. Any port MUST replicate this scaling; the
  symptom of getting it wrong is a silent α→0 collapse on large frames.
- `fxp_sqrt`: Newton on the reciprocal root with range reduction (see header
  comments); no FP fallback.

## 3. INT16 storage contract (i16) / INT16 ストレージ契約

Deployed narrowing `lf=5 cf=9` (fractional bits kept in 16-bit line buffers):

| Plane | 16-bit format | Rationale |
|---|---|---|
| Luma (L_cs, pilot, L_cs_den) | **Q10.5** | GAT-domain range ±~40 needs 6+ integer bits; 5 frac bits ≈ 1/32 step |
| Chroma (half-res C1/C2/C3) | **Q6.9** | chroma is small-amplitude around 0; 9 frac bits |
| Output plane | **Q1.14** | normalized [0,1] output |

Compute between buffers stays int32 Q11.20. Measured cost of the narrowing:
**~58–65 dB PSNR vs the INT32-storage result on full frames — exactly the
storage quantization** (re-verified 2026-07-04, 16/16 patches + 3/3 full
frames bit-exact at INT32 storage). INT16 quality vs FP32 end-to-end:
−0.1…−0.7 dB (paper Table I).

## 4. FP16 storage contract v1 (Vulkan port) / FP16 ストレージ契約

Source: `--f16-storage` oracle (`GALOSH_F16_STAGE` sites in
`galosh_raw_cpu.c`); rounding = IEEE binary16, round-to-nearest-even,
compute stays f32 ("FP16 storage / FP32 accumulate").

Rounded inter-phase buffers (the GPU FP16 port stores exactly these in
16-bit images/buffers):

1. `L_cs` (P5 input), `L_cs_pilot` (pass1→pass2), `L_cs_den` (P5 output)
2. `L_pixel` (P6 full-res guide), `L_h_den` (P6 half-res guide)
3. `C1/C2/C3_h_den` (P8 blended half-res chroma)
4. `C1/C2/C3_aligned` (P9 full-res chroma)

Pyramid-internal quarter/eighth buffers remain f32 in this contract revision.
Measured storage-quantization cost: **77.3 dB (synthetic 1080p) / 82.6 dB
(RawNIND 12 MP real frame)** — better than the INT16 contract, i.e. visually
transparent. A Vulkan implementation that rounds at exactly these points can
be bit-parity tested against the oracle.

## 5. GAT inverse LUT / GAT 逆変換 LUT

The exact unbiased inverse (Makitalo–Foi) is realized as a piecewise LUT
built from the per-image (α, σ²) fit: verbose log reports
`piecewise GAT inv table: D range [lo, hi] … y_break/t_break`. Two properties
matter for ports:

- The build cost is **resolution-independent** (measured 6.7 ms on GPU
  regardless of 1080p/4K) and depends ONLY on (α, σ²) → under the video
  noise-model amortization (`--noise=hold/every:N/ema`) the LUT is reused
  and its cost disappears from steady-state frames.
- r32 keeps the LUT in Q11.20 with the same break-point structure
  (`galosh_raw_cpu_int.c` jinc/LUT sections mirror the FP32 piecewise
  definition, sign-preserving floors included).

## 6. Streaming / line-buffer model / ストリーミング&ラインバッファ

- On-chip state is bounded by the line buffers of the largest vertical
  kernel: **~200 KB at 1080p in INT16** (paper §IV). The GPU i16 pipeline
  demonstrates the tiled streaming organization within a **32 KB LDS**
  budget per workgroup (tiled overlap-add for P5, LDS reductions for P0–P2).
- Per-pixel cost (instrumented, resolution-independent): ≈3.4k MACs +
  ≈140 LUT/special ops.
- Kernel vertical footprints: luma LOSH 8 rows (stride-2 cycle spin), chroma
  LOESS R=7 half-res window, K16 upsample 5×5 half-res taps (fast mode: ≤2×2).

## 7. Verification anchors / 検証アンカー

| Claim | Evidence |
|---|---|
| r32 ≡ GPU INT32 storage | bit-exact end-to-end, SIDD/RawNIND full frames (2026-06-21, re-verified 07-04) |
| INT16 narrowing cost | 58–65 dB (storage quantization only) |
| FP16 contract cost | 77.3 / 82.6 dB (`--f16-storage` vs canonical) |
| flags-off invariance | `standalone/tests/v2_identity.py` — byte-identical across V2.0 feature additions (8-case hash manifest) |
| FP32 determinism | every identity case hashed equal across repeated runs (OpenMP included) |

## 8. File map / 実装ファイル対応

- FP32 canonical + V2.0 flags: `standalone/galosh_raw_cpu.c`, `standalone/galosh_cpu.h`
- Fixed-point reference (Q11.20): `standalone/galosh_raw_cpu_int.c`, `standalone/galosh_cpu_int.h`
- INT16 streaming GPU (frozen evidence): `standalone/galosh_int_*.{clh,cl}`, `galosh_int_pipe_test.c`
- FP32 GPU OpenCL (frozen, maintenance): `standalone/galosh_raw_gpu.c`, `standalone/galosh.cl`
- Identity harness: `standalone/tests/v2_identity.py` (+ `v2_baseline.json`)
- Diagnostic HP probes (never ship): `standalone/*_hp.{c,h}` (int64-based, build-excluded)
