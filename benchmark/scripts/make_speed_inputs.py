"""Generate float32 Bayer test inputs at 1080p / 4K / 5K for the i16 GPU speed
benchmark (galosh_int_bench.exe).

A real SIDD noisy RAW validation block is tiled (with per-tile flips to avoid
seam periodicity) to the target resolution, preserving realistic Poisson-Gaussian
noise statistics so that P0 noise estimation / the Foi LUT k_max loop see a
representative noise level.  Output = raw float32, row-major, single-channel
Bayer (same layout galosh_int_bench loads).

Artifacts -> benchmark/results/speed/  (no /c/tmp clutter).
"""
import os, sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat

GALOSH = Path(r"C:\Users\luxgrain\GALOSH")
SIDD_VAL = GALOSH / "benchmark" / "SIDD_Validation"
OUT = GALOSH / "benchmark" / "results" / "speed"
OUT.mkdir(parents=True, exist_ok=True)

RES = {"1080p": (1920, 1080), "4k": (3840, 2160), "5k": (5120, 2880)}


def tile_to(block, W, H):
    bh, bw = block.shape
    out = np.empty((H, W), np.float32)
    for y0 in range(0, H, bh):
        for x0 in range(0, W, bw):
            ty, tx = (y0 // bh), (x0 // bw)
            t = block
            if ty & 1: t = t[::-1, :]
            if tx & 1: t = t[:, ::-1]
            yh = min(bh, H - y0); xw = min(bw, W - x0)
            # keep Bayer phase: tile offsets are multiples of bh/bw (even) so
            # the 2x2 CFA pattern is preserved across tiles.
            out[y0:y0+yh, x0:x0+xw] = t[:yh, :xw]
    return out


def main():
    raw = loadmat(str(SIDD_VAL / "ValidationNoisyBlocksRaw.mat"))["ValidationNoisyBlocksRaw"]
    blk = raw[0, 0]
    blk = blk.astype(np.float32) / 65535.0 if blk.dtype == np.uint16 else blk.astype(np.float32)
    bh, bw = blk.shape
    # ensure even block dims (Bayer)
    blk = blk[: bh - (bh & 1), : bw - (bw & 1)]
    print(f"source block {blk.shape} range [{blk.min():.4f},{blk.max():.4f}] mean {blk.mean():.4f}")
    for name, (W, H) in RES.items():
        img = tile_to(blk, W, H)
        p = OUT / f"noisy_{name}.bin"
        img.astype(np.float32).tofile(str(p))
        print(f"{name:6s} {W}x{H}  {W*H/1e6:6.2f} MP  -> {p}  ({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
