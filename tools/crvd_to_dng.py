#!/usr/bin/env python3
"""CRVD raw TIFF -> darktable-openable DNG (viewing/inspection helper).

CRVD (RViDeNet, CVPR 2020) indoor raws are bare 16-bit single-channel
Bayer mosaics (1080x1920, 12-bit range, SONY IMX385, GBRG, black=240,
white=4095) in TIFF containers with no CFA/DNG tags — darktable rejects
them. This wraps the mosaic into a minimal valid DNG using the SAME tag
set as the validated distribution tool (tools/dist/galosh_dng.py).

ColorMatrix1 uses the camera==sRGB assumption (no IMX385 calibration
available) — colors are approximate; fine for inspection, white-balance
in darktable as needed.

Usage:
  python tools/crvd_to_dng.py "E:/img_dataset/CRVD/indoor_raw_noisy/scene1/ISO25600" [-o outdir]
  python tools/crvd_to_dng.py <file.tiff> [...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

TIFF_ASCII, TIFF_SHORT, TIFF_LONG, TIFF_RATIONAL = 2, 3, 4, 5
TIFF_BYTE, TIFF_SRATIONAL = 1, 10

# CRVD / IMX385 container facts (RViDeNet paper + data inspection)
CFA_GBRG = (1, 2, 0, 1)          # 0=R 1=G 2=B; rows: G B / R G
BLACK, WHITE = 240, 4095
# camera==sRGB assumption: ColorMatrix1 (XYZ->cam) = XYZ->sRGB(D65)
XYZ_TO_SRGB = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660,  1.8760108,  0.0415560],
                        [ 0.0556434, -0.2040259,  1.0572252]])
AS_SHOT = (0.55, 1.0, 0.60)      # plausible indoor neutral; WB in dt


def _rat(values, denom=10000):
    out = []
    for v in values:
        out.extend([int(round(float(v) * denom)), denom])
    return out


def convert(src: Path, dst: Path):
    data = tifffile.imread(str(src)).astype(np.uint16)
    assert data.ndim == 2, f"not a bare mosaic: {src}"
    extratags = [
        (50706, TIFF_BYTE, 4, (1, 4, 0, 0), True),                  # DNGVersion
        (50707, TIFF_BYTE, 4, (1, 1, 0, 0), True),                  # DNGBackwardVersion
        (50708, TIFF_ASCII, None, "CRVD IMX385 (GALOSH wrap)", True),
        (33421, TIFF_SHORT, 2, (2, 2), True),                       # CFARepeatPatternDim
        (33422, TIFF_BYTE, 4, CFA_GBRG, True),                      # CFAPattern
        (50713, TIFF_SHORT, 2, (2, 2), True),                       # BlackLevelRepeatDim
        (50714, TIFF_RATIONAL, 4, _rat([BLACK] * 4, 1), True),      # BlackLevel
        (50717, TIFF_LONG, 1, WHITE, True),                         # WhiteLevel
        (50778, TIFF_SHORT, 1, 21, True),                           # CalibIlluminant1=D65
        (50721, TIFF_SRATIONAL, 9, tuple(_rat(XYZ_TO_SRGB.flatten())), True),
        (50728, TIFF_RATIONAL, 3, tuple(_rat(AS_SHOT)), True),      # AsShotNeutral
    ]
    tifffile.imwrite(str(dst), data, photometric=32803, compression=None,
                     planarconfig=1, rowsperstrip=data.shape[0],
                     extratags=extratags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="tiff files or directories")
    ap.add_argument("-o", "--outdir", default=None,
                    help="output dir (default: alongside, .dng extension)")
    args = ap.parse_args()

    files = []
    for inp in args.inputs:
        p = Path(inp)
        files += sorted(p.rglob("*.tiff")) if p.is_dir() else [p]
    if not files:
        sys.exit("no tiff files found")
    for f in files:
        out = (Path(args.outdir) / f.with_suffix(".dng").name) if args.outdir \
              else f.with_suffix(".dng")
        out.parent.mkdir(parents=True, exist_ok=True)
        convert(f, out)
        print(out)
    print(f"{len(files)} file(s) converted")


if __name__ == "__main__":
    main()
