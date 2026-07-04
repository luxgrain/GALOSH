import rawpy
import numpy as np
import os
import sys

dng_dir = os.environ.get("GALOSH_DNG_DIR", "dng")
files = sorted(os.listdir(dng_dir))
print(f"DNG files: {files}")

for fn in files:
    if not fn.endswith('.dng'): continue
    path = dng_dir + "/" + fn
    print(f"\n=== {fn} ===")
    print(f"path: {path}, exists: {os.path.exists(path)}")
    print(f"size: {os.path.getsize(path)} bytes")
    try:
        raw = rawpy.imread(path)
        print(f"  raw shape: {raw.raw_image.shape}")
        print(f"  raw dtype: {raw.raw_image.dtype}")
        print(f"  raw_pattern:\n{raw.raw_pattern}")
        print(f"  color_desc: {raw.color_desc}")
        print(f"  white_level: {raw.white_level}")
        print(f"  black_level_per_channel: {raw.black_level_per_channel}")
        print(f"  raw stats: min={raw.raw_image.min()}, max={raw.raw_image.max()}, mean={raw.raw_image.mean():.1f}")
    except Exception as e:
        print(f"  ERROR: {e}")
