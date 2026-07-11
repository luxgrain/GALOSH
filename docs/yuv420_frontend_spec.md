# GALOSH-420 front-end specification (draft v0.1, 2026-07-11)

Scope: the 4:2:0 (and 4:2:2 / 4:0:0) input front-end for GALOSH-YUV —
planar YCbCr in, format-preserving out.  Chroma is denoised at its NATIVE
half-resolution lattice with a downsampled-Y guide (design decision locked
by the A/B experiment `benchmark/scripts/ab_yuv420.py`: native beats
upsample-to-444-first by +0.51/+0.54 dB Cb/Cr on the 420 lattice and on
LPIPS on 62/80 SIDD scenes, with ~4x cheaper chroma; the gap grows with
noise level, which is exactly when a denoiser is needed).

## Chroma siting (`--siting`)

`--siting` specifies the chroma sample-center location.

In luma pixel-corner coordinates:

```
  Y[0,0] covers [0,1]x[0,1]
```

Chroma sample centers:

```
  center   = (1.0, 1.0)
  left     = (0.5, 1.0)
  topleft  = (0.5, 0.5)
```

Equivalent 2x2 chroma-cell top-left positions:

```
  center   = ( 0.0,  0.0)
  left     = (-0.5,  0.0)
  topleft  = (-0.5, -0.5)
```

The names center/left/topleft describe sample-center locations,
not chroma-cell anchors.

Phase offset relative to center-sited 420 (luma-pixel units):

```
  center   = ( 0.0,  0.0)
  left     = (-0.5,  0.0)
  topleft  = (-0.5, -0.5)
```

Standards mapping: center = JPEG/JFIF, MPEG-1 (H.273 type 1);
left = MPEG-2/AVC/HEVC default (type 0); topleft = BT.2020/BT.2100
recommendation (type 2).  Progressive only — the interlaced bottom-field
variants (types 3-5) are deliberately out of scope.

External normative anchors (verified 2026-07-11):
- FFmpeg `libavutil/pixfmt.h` AVChromaLocation diagram: TOPLEFT =
  position ON the 1st luma line AT the 1st horizontal luma sample
  (co-sited both axes); LEFT = 1st horizontal luma position, midway
  between luma lines; CENTER = midway both axes.
- Microsoft `MFVideoChromaSubsampling`: "Horizontally_Cosited ... if
  this flag is not set, chroma samples are located 1/2 pixel to the
  right of the corresponding luma sample"; "Vertically_Cosited ... if
  not set, 1/2 pixel down".  MPEG-2 = horizontally co-sited only
  (= left); fully Cosited (= topleft); MPEG-1 = neither (= center).
Equivalent statement with the CENTER position as reference: left is
shifted (-0.5, 0) and topleft (-0.5, -0.5) luma px — the two phrasings
describe the same geometry from different anchors.

**Siting determines sample position/phase ONLY — never the resampling
kernel.**  The guide-construction and reconstruction kernels below are
implementation choices evaluated AT the siting phase, documented
separately:

- Guide construction (Y -> chroma lattice, band-limited): v1 uses the
  minimal kernels per phase — center: 2x2 box; left: vertical 2-tap on
  the co-sited column; topleft: co-sited tap (optionally a small
  symmetric low-pass).  Any phase-correct kernel is admissible.

  **MANDATORY verification (affine-field phase test)**: for every siting,
  feeding an analytic affine field f(x,y) = ax+by+c (evaluated at luma
  pixel centers) through the guide constructor must reproduce
  f(chroma sample-center position) exactly (float rounding only; box and
  tent kernels are affine-exact, so any residual = phase error).  The
  guide and the chroma plane built from the same field must be identical
  arrays (co-location).  Detection power: a deliberately center-phased
  guide against left-sited positions must show mean error = 0.5*|ax|.
  Reference implementation of the test: benchmark/scripts (A/B rig,
  2026-07-11 session — measured 1e-16 / 0.0 / 0.00650 vs theory 0.00650).
  Measured cost of IGNORING guide phase (center-phased guide on left-sited
  chroma, 20 SIDD scenes): Cb -0.13 / Cr -0.19 dB, worse on 18/20 scenes —
  small but systematic; phase-matched guides recover it for free.
  Measured siting-vs-quality (80 SIDD scenes, phase-matched guides):
  center/left/topleft sRGB PSNR 35.77/35.77/35.74, LPIPS
  0.3035/0.3014/0.3000 — siting choice itself has NO effect on denoise
  quality; only guide-phase mismatch costs anything.
- 444 export (chroma lattice -> Y lattice): K16 joint-bilateral EWA-jinc
  with per-siting phase weight tables (generated offline per siting).
  Note: at zero phase (topleft, even/even luma) the windowed-jinc still
  has side lobes — it is NOT an identity tap; only interpolating kernels
  (bilinear/NN) are exact there.

## Transfer function (`--eotf`)

`srgb | g22 | g24 | bt709 | hlg | pq | linear` — externally specified, no
auto-detection.  Linearization to relative [0,1].  NOTE (documented
approximation): NCL luma is linearized directly as EOTF^-1(Y'), which is
not true luminance; the blind PG fit is performed on that domain and is
self-consistent.  Chroma-lattice processing reconstructs half-res R'G'B'
from (guide-downsampled Y', Cb, Cr), linearizes, and reuses the existing
YUV front-end at that scale (the form validated by the A/B rig).

## Matrix (`--matrix`)

`bt601 | bt709 | bt2020 | custom:Kr,Kb` — H.273 matrix coefficients.
Primaries (e.g. Display P3) are NOT a matrix axis: GALOSH performs no
gamut conversion and is primaries-agnostic (tag passthrough).  P3 content
is typically wrapped with bt709 or bt601 coefficients.

## Range (`--range`) and depth (`--depth`)

`limited | full`, externally specified.  Depth 8-16 bit planar integer;
limited-range scaling uses the 8-bit constants x 2^(n-8); full range
divides by 2^n - 1.  Output is re-quantized to the input depth
(rounding; optional dither TBD).

## Formats

- 4:2:0: native path (this spec).
- 4:2:2: provisionally accepted via horizontal upsample to 4:4:4
  (video 4:2:2 is horizontally co-sited by convention; JPEG-style
  centered 4:2:2 unsupported in v1).  Native-422 deferred until A/B
  evidence demands it.
- 4:0:0: Y-only — chroma stage skipped.

## Reference profiles (validation matrix)

| profile | matrix | eotf | range | siting |
|---|---|---|---|---|
| DVD (SD)        | bt601  | g24  | limited | left |
| Blu-ray (HD)    | bt709  | g24  | limited | left |
| UHD HDR10       | bt2020 | pq   | limited | topleft |
| UHD HLG (bcast) | bt2020 | hlg  | limited | topleft |
| JPEG            | bt601  | srgb | full    | center |
