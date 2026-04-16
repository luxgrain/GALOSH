#!/usr/bin/env python3
"""
Kodak 24 benchmark: synthetic Poisson-Gaussian noise at multiple ISOs.
Compares Ours (GAT+BM3D+L/C) vs PerChannel BM3D.
Uses MAD-adapted sigma with luma=1.0, chroma=1.0 for fair structural comparison.
"""
import numpy as np
import subprocess
import os
import json
from pathlib import Path
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

BASE = Path(__file__).parent.parent
EXE = str(BASE / "standalone" / "rawdenoise.exe")
KODAK = BASE / "datasets" / "kodak"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


def inv_srgb(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0


def ssim_4ch(a, b):
    vals = []
    for dy in range(2):
        for dx in range(2):
            vals.append(ssim(a[dy::2, dx::2], b[dy::2, dx::2], data_range=1.0))
    return np.mean(vals)


def synth_noise(clean_raw, iso, rng):
    gain = iso / 100.0
    alpha = 0.001 * gain
    sigma_read = 0.002 * gain
    sigma_sq = sigma_read ** 2
    rate = np.maximum(clean_raw / max(alpha, 1e-10), 0.0)
    noisy = rng.poisson(rate).astype(np.float32) * alpha
    noisy += rng.standard_normal(clean_raw.shape).astype(np.float32) * sigma_read
    return np.clip(noisy, 0, 1).astype(np.float32), alpha, sigma_sq


def run_denoiser(inp, out, w, h, method, alpha, sigma_sq, ls=1.0, cs=1.0):
    r = subprocess.run(
        [EXE, inp, out, str(w), str(h), method, "1.0", str(ls), str(cs),
         str(alpha), str(sigma_sq), "1"],
        capture_output=True, text=True, timeout=300)
    return r


def main():
    isos = [400, 800, 1600, 3200, 6400, 12800]
    images = sorted(KODAK.glob("kodim*.png"))[:24]

    if not images:
        print("ERROR: No Kodak images found in", KODAK)
        return

    print(f"Kodak benchmark: {len(images)} images × {len(isos)} ISOs")
    print(f"MAD-adapted σ, luma=1.0, chroma=1.0 (fair structural comparison)")
    print()

    all_results = []

    for iso in isos:
        gains = {"ours_psnr": [], "ours_ssim": [], "pc_psnr": [], "pc_ssim": [],
                 "noisy_psnr": [], "noisy_ssim": []}

        for img_path in images:
            rng = np.random.default_rng(42)

            # Load and convert to linear
            img = imread(str(img_path)).astype(np.float32) / 255.0
            h, w = img.shape[:2]
            h -= h % 2; w -= w % 2
            img = img[:h, :w]
            linear = inv_srgb(img).astype(np.float32)

            # Create Bayer mosaic
            clean_raw = np.zeros((h, w), dtype=np.float32)
            clean_raw[0::2, 0::2] = linear[0::2, 0::2, 0]  # R
            clean_raw[0::2, 1::2] = linear[0::2, 1::2, 1]  # Gr
            clean_raw[1::2, 0::2] = linear[1::2, 0::2, 1]  # Gb
            clean_raw[1::2, 1::2] = linear[1::2, 1::2, 2]  # B

            noisy, alpha, sigma_sq = synth_noise(clean_raw, iso, rng)

            # Save
            inp = str(BASE / "standalone" / "tmp_bench.bin")
            noisy.tofile(inp)

            # Run Ours
            out_ours = str(BASE / "standalone" / "tmp_ours.bin")
            run_denoiser(inp, out_ours, w, h, "ours", alpha, sigma_sq, 1.0, 1.0)
            ours = np.fromfile(out_ours, dtype=np.float32).reshape(h, w)

            # Run PerChannel
            out_pc = str(BASE / "standalone" / "tmp_pc.bin")
            run_denoiser(inp, out_pc, w, h, "perchannel", alpha, sigma_sq)
            pc = np.fromfile(out_pc, dtype=np.float32).reshape(h, w)

            # Metrics
            gains["noisy_psnr"].append(psnr(clean_raw, noisy))
            gains["noisy_ssim"].append(ssim_4ch(clean_raw, noisy))
            gains["ours_psnr"].append(psnr(clean_raw, ours))
            gains["ours_ssim"].append(ssim_4ch(clean_raw, ours))
            gains["pc_psnr"].append(psnr(clean_raw, pc))
            gains["pc_ssim"].append(ssim_4ch(clean_raw, pc))

        # Average
        avg = {k: np.mean(v) for k, v in gains.items()}
        delta_psnr = avg["ours_psnr"] - avg["pc_psnr"]
        delta_ssim = avg["ours_ssim"] - avg["pc_ssim"]

        result = {
            "iso": iso,
            "noisy_psnr": round(avg["noisy_psnr"], 2),
            "pc_psnr": round(avg["pc_psnr"], 2),
            "pc_ssim": round(avg["pc_ssim"], 4),
            "ours_psnr": round(avg["ours_psnr"], 2),
            "ours_ssim": round(avg["ours_ssim"], 4),
            "delta_psnr": round(delta_psnr, 2),
            "delta_ssim": round(delta_ssim, 4),
        }
        all_results.append(result)

        print(f"ISO {iso:>5}: noisy={avg['noisy_psnr']:.2f}  "
              f"PC={avg['pc_psnr']:.2f}/{avg['pc_ssim']:.4f}  "
              f"Ours={avg['ours_psnr']:.2f}/{avg['ours_ssim']:.4f}  "
              f"Δ={delta_psnr:+.2f}/{delta_ssim:+.4f}")

    # Summary table
    print(f"\n{'='*65}")
    print(f"{'ISO':>6} {'Noisy':>8} {'PC PSNR':>9} {'Ours PSNR':>10} {'ΔPSNR':>7} {'ΔSSIM':>8}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['iso']:>6} {r['noisy_psnr']:>8.2f} {r['pc_psnr']:>9.2f} "
              f"{r['ours_psnr']:>10.2f} {r['delta_psnr']:>+7.2f} {r['delta_ssim']:>+8.4f}")

    # Save
    out_path = RESULTS / "kodak_benchmark_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
