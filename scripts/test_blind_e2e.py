#!/usr/bin/env python3
"""End-to-end test: blind estimation vs known parameters.
Compares PSNR of denoising with true α,σ² vs auto-estimated (α=-1).
"""
import numpy as np
import subprocess
import os
from pathlib import Path
from skimage.io import imread

MSYS_PATH = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
BASE = Path(__file__).parent.parent
EXE = str(BASE / "standalone" / "rawdenoise_v3.exe")
KODAK = BASE / "datasets" / "kodak"

def inv_srgb(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def synth_noise(clean_raw, iso, rng):
    gain = iso / 100.0
    alpha = 0.001 * gain; sr = 0.002 * gain; sq = sr ** 2
    rate = np.maximum(clean_raw / max(alpha, 1e-10), 0)
    noisy = rng.poisson(rate).astype(np.float32) * alpha
    noisy += rng.standard_normal(clean_raw.shape).astype(np.float32) * sr
    return np.clip(noisy, 0, 1).astype(np.float32), alpha, sq

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0

def run(noisy, w, h, alpha, sq, tag):
    inp = str(BASE / "standalone" / f"tmp_e2e_{tag}.bin")
    out = str(BASE / "standalone" / f"tmp_e2e_{tag}_out.bin")
    noisy.tofile(inp)
    env = os.environ.copy(); env["PATH"] = MSYS_PATH
    result = subprocess.run(
        [EXE, inp, out, str(w), str(h), "ours",
         "1.0", "1.0", "1.0", str(alpha), str(sq), "1", "25"],
        capture_output=True, timeout=300, env=env
    )
    stderr = result.stderr.decode("utf-8", errors="replace")
    return np.fromfile(out, dtype=np.float32).reshape(h, w), stderr


def main():
    images = sorted(KODAK.glob("kodim*.png"))[:8]
    isos = [800, 1600, 3200, 6400, 12800]

    print("End-to-end: blind vs known-parameter denoising")
    print(f"{'ISO':>6}  {'Known PSNR':>11}  {'Blind PSNR':>11}  {'Delta':>7}")
    print("-" * 45)

    for iso in isos:
        gain = iso / 100.0
        true_alpha = 0.001 * gain; true_sq = (0.002 * gain) ** 2

        known_psnrs = []
        blind_psnrs = []

        for idx, img_path in enumerate(images):
            rng = np.random.default_rng(42)
            img = imread(str(img_path)).astype(np.float32) / 255.0
            h, w = img.shape[:2]; h -= h % 2; w -= w % 2; img = img[:h, :w]
            linear = inv_srgb(img).astype(np.float32)
            cr = np.zeros((h, w), dtype=np.float32)
            cr[0::2, 0::2] = linear[0::2, 0::2, 0]
            cr[0::2, 1::2] = linear[0::2, 1::2, 1]
            cr[1::2, 0::2] = linear[1::2, 0::2, 1]
            cr[1::2, 1::2] = linear[1::2, 1::2, 2]
            noisy, _, _ = synth_noise(cr, iso, rng)

            # Known parameters
            den_known, _ = run(noisy, w, h, true_alpha, true_sq, "known")
            p_known = psnr(cr, den_known)
            known_psnrs.append(p_known)

            # Blind estimation (alpha=-1, sigma_sq=-1)
            den_blind, stderr = run(noisy, w, h, -1, -1, "blind")
            p_blind = psnr(cr, den_blind)
            blind_psnrs.append(p_blind)

        avg_known = np.mean(known_psnrs)
        avg_blind = np.mean(blind_psnrs)
        delta = avg_blind - avg_known
        print(f"{iso:>6}  {avg_known:>11.2f}  {avg_blind:>11.2f}  {delta:>+7.2f}")

    print()
    print("Delta > 0: blind is better; < 0: known params is better")


if __name__ == "__main__":
    main()
