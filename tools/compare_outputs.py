"""Compare two raw float32 outputs from standalone CPU.

Shows overall diff stats + saves visualizations.
"""
import numpy as np
import sys

if len(sys.argv) < 5:
    print("Usage: compare_outputs.py <out0.bin> <out4.bin> <W> <H>")
    sys.exit(1)

out0 = np.fromfile(sys.argv[1], dtype=np.float32).reshape(int(sys.argv[4]), int(sys.argv[3]))
out4 = np.fromfile(sys.argv[2], dtype=np.float32).reshape(int(sys.argv[4]), int(sys.argv[3]))

diff = out4 - out0
abs_diff = np.abs(diff)

print(f"out0 stats: min={out0.min():.4f} max={out0.max():.4f} mean={out0.mean():.4f}")
print(f"out4 stats: min={out4.min():.4f} max={out4.max():.4f} mean={out4.mean():.4f}")
print(f"diff (out4 - out0) stats:")
print(f"  min={diff.min():.6f}")
print(f"  max={diff.max():.6f}")
print(f"  mean={diff.mean():.6f}")
print(f"  std={diff.std():.6f}")
print(f"  abs_mean={abs_diff.mean():.6f}")
print(f"  abs_max={abs_diff.max():.6f}")
print()

# Percentage of pixels with diff > threshold
for thresh in [0.0001, 0.001, 0.01, 0.05]:
    pct = (abs_diff > thresh).mean() * 100
    print(f"  pixels with |diff| > {thresh}: {pct:.2f}%")

# If diff is essentially 0, slider has no effect on output binary
if abs_diff.max() < 1e-7:
    print("\n!!! WARNING: diff is essentially ZERO. chroma_strength has NO effect on output. !!!")
elif abs_diff.mean() < 1e-5:
    print("\n!!! WARNING: diff is very small. chroma_strength has minimal effect. !!!")
else:
    print(f"\nOK: chroma_strength affects output. abs_mean diff = {abs_diff.mean():.6f}")
