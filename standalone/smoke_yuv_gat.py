#!/usr/bin/env python3
"""Smoke test for galosh_gpu.exe yuv_gat mode on SIDD Medium scene #0."""
import numpy as np
import subprocess
import sys
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr

BASE = Path(__file__).parent.parent
BENCH_DIR = BASE / "datasets" / "sidd" / "medium_bench"
GPU_EXE = Path(__file__).parent / "galosh_gpu.exe"
BASH_EXE = Path(r"C:\msys64\usr\bin\bash.exe")

# Load first scene
noisy = np.load(str(BENCH_DIR / "0001_S6_GRBG_010_noisy_srgb.npy"))
gt    = np.load(str(BENCH_DIR / "0001_S6_GRBG_010_gt_srgb.npy"))
print(f"[SMOKE] noisy shape={noisy.shape} dtype={noisy.dtype}")
print(f"[SMOKE] gt    shape={gt.shape} dtype={gt.dtype}")

# noisy is uint8 [0,255] or float [0,1]?
if noisy.dtype == np.uint8:
    noisy_f = noisy.astype(np.float32) / 255.0
    gt_f    = gt.astype(np.float32) / 255.0
else:
    noisy_f = noisy.astype(np.float32)
    gt_f    = gt.astype(np.float32)

H, W, C = noisy_f.shape
print(f"[SMOKE] H={H} W={W} C={C} range=[{noisy_f.min():.4f}, {noisy_f.max():.4f}]")

# PSNR noisy vs GT
p_noisy = psnr(gt_f, noisy_f, data_range=1.0)
print(f"[SMOKE] PSNR(noisy vs GT) = {p_noisy:.2f} dB")

# Write input .bin (interleaved H*W*3 float32)
in_path  = Path(__file__).parent / "_smoke_in.bin"
out_path = Path(__file__).parent / "_smoke_out.bin"
noisy_f.tofile(str(in_path))

# Run yuv_gat
cmd = [str(BASH_EXE), "-c",
       f'"{GPU_EXE}" "{in_path}" "{out_path}" {W} {H} yuv_gat 1.0 1.0 0']
print(f"[SMOKE] cmd: {' '.join(cmd)}")
r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
print("[SMOKE] stderr:", r.stderr[:3000] if r.stderr else "(empty)")
print(f"[SMOKE] return code: {r.returncode}")

if r.returncode != 0:
    print("[SMOKE] FAILED")
    sys.exit(1)

# Read output
den = np.fromfile(str(out_path), dtype=np.float32).reshape(H, W, 3)
den = np.clip(den, 0.0, 1.0)

p_den = psnr(gt_f, den, data_range=1.0)
print(f"[SMOKE] PSNR(denoised vs GT) = {p_den:.2f} dB")
print(f"[SMOKE] Delta = {p_den - p_noisy:+.2f} dB")
if p_den > p_noisy:
    print("[SMOKE] SUCCESS: denoising improved PSNR")
else:
    print("[SMOKE] WARNING: denoising did NOT improve PSNR")

# Clean up
in_path.unlink(missing_ok=True)
out_path.unlink(missing_ok=True)
