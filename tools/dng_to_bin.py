"""DNG → float32 binary raw, normalized [0,1]."""
import rawpy
import numpy as np
import os
import sys

if len(sys.argv) < 3:
    print("Usage: dng_to_bin.py <in.dng> <out.bin>")
    print("       Outputs <out.bin> + prints W H to stdout")
    sys.exit(1)

src, dst = sys.argv[1], sys.argv[2]
raw = rawpy.imread(src)
raw_u16 = raw.raw_image.copy()  # uint16
H, W = raw_u16.shape
black = raw.black_level_per_channel[0]  # all 528
white = raw.white_level
# Linearize: subtract black, divide by (white - black)
raw_f = (raw_u16.astype(np.float32) - black) / (white - black)
raw_f = np.clip(raw_f, 0.0, 1.0)
raw_f.tofile(dst)
print(f"{W} {H}")
print(f"  in:  {src}", file=sys.stderr)
print(f"  out: {dst}", file=sys.stderr)
print(f"  shape: {H}x{W} (HxW)", file=sys.stderr)
print(f"  pattern: {raw.raw_pattern.tolist()} (RGBG indexing)", file=sys.stderr)
print(f"  raw [black, white]: [{black}, {white}]", file=sys.stderr)
print(f"  norm stats: min={raw_f.min():.4f} max={raw_f.max():.4f} mean={raw_f.mean():.4f}", file=sys.stderr)
