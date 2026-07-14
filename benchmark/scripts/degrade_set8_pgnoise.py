#!/usr/bin/env python3
"""PG-noise track — realistic degradation synthesis (clean sRGB video -> realistic
noisy), 2026-07-12.  The far-more-representative alternative to flat AWGN.

FLOW (physically ordered: sensor noise -> ISP -> compression):
  clean sRGB 8-bit
   |  [Stage 1 UNPROCESS: sRGB -> pseudo-raw Bayer]  (Brooks 2019)
   |   inverse sRGB OETF -> linear RGB
   |   inverse CCM (camera==sRGB => identity here)
   |   inverse white-balance (/ WB gains)  -> sensor-linear RGB
   |   mosaic (GBRG)                        -> pseudo-raw Bayer
   v
  [Stage 2 NOISE: signal-dependent Poisson-Gaussian in raw domain]
   |   var(I) = a*I + b   (a=shot, b=read; CRVD-CALIBRATED per ISO,
   |                       crvd_pg_params.json, real SONY IMX385)
   |   + optional ELD row/banding (per-row Gaussian)  [high-ISO]
   |   clip [0,1]
   v
  [Stage 3 RE-ISP: raw -> sRGB]   <- this is what CORRELATES the noise
   |   Malvar demosaic (GBRG)  -> spatial + cross-channel correlation
   |   white-balance (* WB gains)
   |   CCM (identity) -> clip -> sRGB OETF
   v
  noisy sRGB 8-bit   ("B-core": realistic-noise track, GT = clean sRGB)
   |
   |  [Stage 4 COMPRESSION (B-full only): H.264/H.265 round-trip, CRF sweep]
   v
  compressed noisy sRGB   ("B-full": + dominant real-world video degradation)

NO resize (this is denoise, not SR).  Optional components (banding,
compression, grain) fire per-clip so the bench isn't only worst-case
(avoids the corner-case trap, arXiv 2205.04910).  The demosaic-correlated
signal-dependent noise is exactly GALOSH's GAT/Anscombe regime.

This module = Stages 1-3 (B-core).  Stage 4 compression = separate helper
(needs x264/x265; see compress_h264 below, ffmpeg-gated).
"""
import json
from pathlib import Path

import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004 as _demo

# CRVD IMX385 ISP constants (shared with bench_crvd / crvd_to_dng)
WB = np.array([1.0 / 0.55, 1.0, 1.0 / 0.60])   # AsShotNeutral reciprocals

# ISO-axis calibration: SMARTPHONE trimmed-median curve, 2026-07-14.
# normalized raw var = a*I + b.  See NOISE_CALIBRATION.md.
# [2026-07-14 REPLACES the 10-model all-camera median (a(1600)=0.000641):
#  that median was dragged ~9x low by clean DSLR/mirrorless votes, making
#  "ISO25600" only ~ phone ISO2800 — visibly under-noised.]
# New curve = per-phone-vote median over the SIDD per-image NLFs
# (noise_level_functions.csv, 160 data rows, 5 phones), each row normalized to
# ISO1600 (a~ISO^0.85, b~ISO^1.5), per-phone median, then the TRIMMED
# 3-phone median (drop lowest iPhone7 / highest LG G4 -> GP/S6/N6):
#   a(1600) = 0.005763,  b(1600) = 6.34e-5
# Developed-noise equivalence (measured, tractor, clip conv.): ISO400=AWGN
# sigma10, 6400=sigma30, 12800=sigma40, 25600~=sigma50 -> the 7-ISO sweep is
# level-matched with the AWGN sigma10-50 track.  Caveat: ISO>~10000
# extrapolates the ^0.85 law beyond the SIDD observed range (ISO 50-10000).
PG = {"ISO400":   (0.0017737, 0.0000079),
      "ISO800":   (0.0031983, 0.0000224),
      "ISO1600":  (0.0057630, 0.0000634),
      "ISO3200":  (0.0103878, 0.0001793),
      "ISO6400":  (0.0187240, 0.0005072),
      "ISO12800": (0.0337500, 0.0014346),
      "ISO25600": (0.0608347, 0.0040576)}

# Sensor-diversity ranges (mode 2): measured across 11 real sensors
# (CRVD + SIDD + RawNIND).  a=shot log-uniform, b=read uniform (clamp>=0).
A_RANGE = (1e-4, 1e-2)      # shot; library log-median ~7e-4
B_RANGE = (0.0, 1.5e-3)     # read


def sample_pg(rng):
    """Mode 2: draw (a, b) from the multi-sensor library distribution."""
    a = 10.0 ** rng.uniform(np.log10(A_RANGE[0]), np.log10(A_RANGE[1]))
    b = rng.uniform(*B_RANGE)
    return float(a), float(b)


def srgb_eotf_inv(s):        # sRGB -> linear
    s = np.clip(s, 0, 1)
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def srgb_oetf(l):            # linear -> sRGB
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, 12.92 * l,
                    1.055 * np.power(l, 1 / 2.4) - 0.055)


def mosaic_gbrg(rgb):
    """RGB HxWx3 -> Bayer HxW, GBRG (row0: G B / row1: R G)."""
    h, w = rgb.shape[:2]
    b = np.empty((h, w), rgb.dtype)
    b[0::2, 0::2] = rgb[0::2, 0::2, 1]   # G
    b[0::2, 1::2] = rgb[0::2, 1::2, 2]   # B
    b[1::2, 0::2] = rgb[1::2, 0::2, 0]   # R
    b[1::2, 1::2] = rgb[1::2, 1::2, 1]   # G
    return b


def degrade_bcore(clean_srgb_u8, iso="ISO12800", rng=None, banding=None,
                  pg=None):
    """clean sRGB uint8 HxWx3 -> clean + ISP-correlated PG noise.

    [v3 2026-07-14 NOISE-ONLY INJECTION] The PG noise is generated in the
    raw domain and carried through the ISP as a DIFFERENCE:
        noise = ISP(raw + n) - ISP(raw);      noisy = clean + noise
    so the noise keeps its raw-domain signal dependence (gamma boosts shadow
    noise), demosaic-driven spatial/chroma correlation and WB R/B asymmetry —
    while the CONTENT never rides through the lossy mosaic->demosaic
    roundtrip.  (v2 passed the whole image through the roundtrip: on Set8 —
    already developed/sharpened/compressed video, violating the Brooks
    unprocess assumption — the noise-free roundtrip lost ~27 dB vs the
    original with visible zipper/ringing, and a better demosaicer did not
    help: Malvar 27.35 / Menon2007 26.79 dB.  See _gt_demosaic_check.png.)
    GT = the pristine clean frame itself; pg=(0,0) returns clean exactly.

    iso: key into the phone-median PG table, OR pass pg=(a,b) directly.
    banding: per-row read-noise std (normalized); None -> ISO-scaled default
             at high ISO, 0 to disable.
    """
    if rng is None:
        rng = np.random.default_rng()
    a, b = pg if pg is not None else PG[iso]
    if a == 0.0 and b == 0.0 and (banding is None or banding == 0):
        return clean_srgb_u8.copy()      # GT path: pristine frame

    # unprocess (NOISE path only): sRGB -> sensor-linear -> un-WB -> mosaic
    lin = srgb_eotf_inv(clean_srgb_u8.astype(np.float64) / 255.0)
    lin = np.clip(lin / WB[None, None, :], 0, 1)
    raw = mosaic_gbrg(lin)                          # pseudo-raw Bayer

    # signal-dependent Poisson-Gaussian noise in the raw domain
    var = np.clip(a * raw + b, 1e-12, None)
    rawn = raw + rng.normal(0.0, 1.0, raw.shape) * np.sqrt(var)
    if banding is None:
        banding = 0.3 * np.sqrt(b) if b > 4e-4 else 0.0   # high-ISO only
    if banding > 0:
        rawn = rawn + rng.normal(0.0, banding, (raw.shape[0], 1))  # per-row
    rawn = np.clip(rawn, 0, 1)

    def isp(bayer):        # demosaic (correlates noise) -> WB -> gamma
        rgb = np.clip(_demo(bayer, "GBRG") * WB[None, None, :], 0, 1)
        return srgb_oetf(rgb) * 255.0

    noise = isp(rawn) - isp(raw)         # content cancels exactly
    out = clean_srgb_u8.astype(np.float64) + noise
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# ---- Stage 4 (B-full): H.264/H.265 round-trip; ffmpeg-gated ------------
def compress_h264(frames_u8, crf=23, codec="libx264", fps=25, tmp=None):
    """list of HxWx3 uint8 -> encode/decode round-trip (4:2:0). Requires
    ffmpeg on PATH; raises if absent.  Dominant real-world video degradation.
    """
    import shutil, subprocess, tempfile, os
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH (Stage 4 compression skipped)")
    h, w = frames_u8[0].shape[:2]
    td = Path(tmp or tempfile.mkdtemp())
    raw = td / "in.rgb"; mp4 = td / "c.mp4"; out = td / "out.rgb"
    with open(raw, "wb") as f:
        for fr in frames_u8:
            f.write(np.ascontiguousarray(fr, np.uint8).tobytes())
    subprocess.run(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", str(fps), "-i", str(raw),
                    "-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p",
                    str(mp4)], capture_output=True, check=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(mp4), "-f", "rawvideo",
                    "-pix_fmt", "rgb24", str(out)], capture_output=True,
                   check=True)
    d = np.fromfile(out, np.uint8).reshape(-1, h, w, 3)
    return [d[i] for i in range(d.shape[0])]


if __name__ == "__main__":
    import sys
    from PIL import Image
    # smoke: degrade a clean frame at each ISO, save clean|noisy strips
    src = sys.argv[1] if len(sys.argv) > 1 else \
        r"E:\img_dataset\Set8\test_sequences\tractor\00000.png"
    clean = np.asarray(Image.open(src).convert("RGB"))
    rng = np.random.default_rng(0)
    outdir = Path(__file__).resolve().parents[2] / "benchmark" / \
        "results_set8_pgnoise"
    outdir.mkdir(parents=True, exist_ok=True)
    for iso in ["ISO1600", "ISO6400", "ISO25600"]:
        noisy = degrade_bcore(clean, iso, rng)
        d = noisy.astype(float) - clean.astype(float)
        strip = np.concatenate([clean, noisy], axis=1).astype(np.uint8)
        Image.fromarray(strip).save(outdir / f"_smoke_{iso}.png")
        print(f"{iso}: sRGB noise std={np.std(d):.2f} "
              f"mean|n|={np.mean(np.abs(d)):.2f}  (clean|noisy saved)")
    print("saved smoke strips to", outdir)
