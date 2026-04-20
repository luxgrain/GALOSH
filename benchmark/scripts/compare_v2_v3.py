#!/usr/bin/env python3
"""Quick comparison: V2 (old inverse, luma-only match) vs V3 (Foi exact inverse + Mahalanobis)
Tests on Kodak first 8 images, ISO 3200/6400/12800.
Metrics: PSNR, Chroma PSNR, False Color RMS, LPIPS
"""
import numpy as np
import subprocess
import os
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import rgb2lab
from scipy.ndimage import uniform_filter

MSYS_PATH = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
BASE = Path(__file__).parent.parent
EXE_V2 = str(BASE / "standalone" / "rawdenoise.exe")
EXE_V3 = str(BASE / "standalone" / "rawdenoise_v3.exe")
KODAK = BASE / "datasets" / "kodak"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inv_srgb(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(np.maximum(x, 0), 1/2.4) - 0.055)

def demosaic_bilinear(bayer):
    h, w = bayer.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[0::2, 0::2, 0] = bayer[0::2, 0::2]
    rgb[0::2, 1::2, 1] = bayer[0::2, 1::2]
    rgb[1::2, 0::2, 1] = bayer[1::2, 0::2]
    rgb[1::2, 1::2, 2] = bayer[1::2, 1::2]
    for c in range(3):
        mask = np.zeros((h, w), dtype=np.float32)
        if c == 0: mask[0::2, 0::2] = 1
        elif c == 1: mask[0::2, 1::2] = 1; mask[1::2, 0::2] = 1
        else: mask[1::2, 1::2] = 1
        num = uniform_filter(rgb[:, :, c], size=3)
        den = uniform_filter(mask, size=3)
        rgb[:, :, c] = np.where(mask > 0, rgb[:, :, c], num / np.maximum(den, 1e-10))
    return np.clip(rgb, 0, 1)

def synth_noise(clean_raw, iso, rng):
    gain = iso / 100.0
    alpha = 0.001 * gain; sr = 0.002 * gain; sq = sr ** 2
    rate = np.maximum(clean_raw / max(alpha, 1e-10), 0)
    noisy = rng.poisson(rate).astype(np.float32) * alpha
    noisy += rng.standard_normal(clean_raw.shape).astype(np.float32) * sr
    return np.clip(noisy, 0, 1).astype(np.float32), alpha, sq

_lpips_fn = None
def get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(DEVICE)
    return _lpips_fn

def compute_lpips(a, b):
    fn = get_lpips()
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE) * 2 - 1
    tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE) * 2 - 1
    with torch.no_grad():
        return fn(ta, tb).item()

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0

def chroma_psnr(gt_srgb, den_srgb):
    gt_lab = rgb2lab(np.clip(gt_srgb, 0, 1))
    den_lab = rgb2lab(np.clip(den_srgb, 0, 1))
    chroma_mse = np.mean((gt_lab[:,:,1:] - den_lab[:,:,1:]) ** 2)
    return 10 * np.log10(256.0**2 / max(chroma_mse, 1e-10))

def false_color_rms(gt_srgb, den_srgb):
    gt_lab = rgb2lab(np.clip(gt_srgb, 0, 1))
    den_lab = rgb2lab(np.clip(den_srgb, 0, 1))
    a_err = den_lab[:,:,1] - gt_lab[:,:,1]
    b_err = den_lab[:,:,2] - gt_lab[:,:,2]
    return float(np.sqrt(np.mean(a_err**2 + b_err**2)))

def chroma_psnr_masked(gt_srgb, den_srgb, mask):
    gt_lab = rgb2lab(np.clip(gt_srgb, 0, 1))
    den_lab = rgb2lab(np.clip(den_srgb, 0, 1))
    idx = mask > 0.5
    if idx.sum() < 100: return float('nan')
    diff = gt_lab[idx, 1:] - den_lab[idx, 1:]
    chroma_mse = np.mean(diff ** 2)
    return 10 * np.log10(256.0**2 / max(chroma_mse, 1e-10))

def false_color_masked(gt_srgb, den_srgb, mask):
    gt_lab = rgb2lab(np.clip(gt_srgb, 0, 1))
    den_lab = rgb2lab(np.clip(den_srgb, 0, 1))
    idx = mask > 0.5
    if idx.sum() < 100: return float('nan')
    a_err = den_lab[idx, 1] - gt_lab[idx, 1]
    b_err = den_lab[idx, 2] - gt_lab[idx, 2]
    return float(np.sqrt(np.mean(a_err**2 + b_err**2)))

def run_exe(exe, noisy, w, h, method, alpha, sq, sw=25):
    tag = "v2" if "rawdenoise.exe" in exe else "v3"
    inp = str(BASE / "standalone" / f"tmp_cmp_{tag}.bin")
    out = str(BASE / "standalone" / f"tmp_cmp_{tag}_out.bin")
    noisy.tofile(inp)
    env = os.environ.copy(); env["PATH"] = MSYS_PATH
    result = subprocess.run([exe, inp, out, str(w), str(h), method,
                   "1.0", "1.0", "1.0", str(alpha), str(sq), "1", str(sw)],
                  capture_output=True, timeout=300, env=env)
    if result.returncode != 0:
        print(f"  ERROR ({tag}): {result.stderr.decode('utf-8', errors='replace')[:200]}")
    return np.fromfile(out, dtype=np.float32).reshape(h, w)


def main():
    images = sorted(KODAK.glob("kodim*.png"))[:8]  # First 8 for quick test
    isos = [3200, 6400, 12800]
    versions = [("V2", EXE_V2), ("V3", EXE_V3)]

    print("V2 vs V3 comparison: Foi exact inverse + Mahalanobis matching")
    print(f"Images: {len(images)}, ISOs: {isos}")
    print()

    for iso in isos:
        metrics = {v: {"psnr": [], "cpsnr": [], "fc": [], "lpips": [],
                       "dark_cpsnr": [], "dark_fc": []}
                   for v, _ in versions}
        gain = iso / 100.0
        alpha = 0.001 * gain; sr = 0.002 * gain; sq = sr ** 2

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
            gt_srgb = np.clip(img, 0, 1).astype(np.float32)

            # Dark region mask
            lum = 0.2126 * linear[:,:,0] + 0.7152 * linear[:,:,1] + 0.0722 * linear[:,:,2]
            dark_mask = (lum < 0.08).astype(np.float32)

            for v_name, exe in versions:
                den_bayer = run_exe(exe, noisy, w, h, "ours", alpha, sq)
                den_srgb = np.clip(linear_to_srgb(demosaic_bilinear(den_bayer)), 0, 1).astype(np.float32)

                p = psnr(cr, den_bayer)
                cp = chroma_psnr(gt_srgb, den_srgb)
                fc = false_color_rms(gt_srgb, den_srgb)
                lp = compute_lpips(gt_srgb, den_srgb)
                dcp = chroma_psnr_masked(gt_srgb, den_srgb, dark_mask)
                dfc = false_color_masked(gt_srgb, den_srgb, dark_mask)

                metrics[v_name]["psnr"].append(p)
                metrics[v_name]["cpsnr"].append(cp)
                metrics[v_name]["fc"].append(fc)
                metrics[v_name]["lpips"].append(lp)
                metrics[v_name]["dark_cpsnr"].append(dcp)
                metrics[v_name]["dark_fc"].append(dfc)

            print(f"  ISO {iso} img {idx+1}/8: done")

        print(f"\nISO {iso} (8 images):")
        print(f"  {'Version':<6} {'PSNR':>7} {'ChromaPSNR':>10} {'FalseColor':>11} {'LPIPS':>7} | {'Dark_CPSNR':>10} {'Dark_FC':>8}")
        print(f"  {'-'*68}")
        for v_name, _ in versions:
            m = metrics[v_name]
            print(f"  {v_name:<6} {np.mean(m['psnr']):>7.2f} {np.mean(m['cpsnr']):>10.2f} "
                  f"{np.mean(m['fc']):>11.3f} {np.mean(m['lpips']):>7.4f} | "
                  f"{np.nanmean(m['dark_cpsnr']):>10.2f} {np.nanmean(m['dark_fc']):>8.3f}")

        # Delta V3 - V2
        d_psnr = np.mean(metrics["V3"]["psnr"]) - np.mean(metrics["V2"]["psnr"])
        d_cpsnr = np.mean(metrics["V3"]["cpsnr"]) - np.mean(metrics["V2"]["cpsnr"])
        d_fc = np.mean(metrics["V3"]["fc"]) - np.mean(metrics["V2"]["fc"])
        d_lpips = np.mean(metrics["V3"]["lpips"]) - np.mean(metrics["V2"]["lpips"])
        d_dcp = np.nanmean(metrics["V3"]["dark_cpsnr"]) - np.nanmean(metrics["V2"]["dark_cpsnr"])
        d_dfc = np.nanmean(metrics["V3"]["dark_fc"]) - np.nanmean(metrics["V2"]["dark_fc"])
        print(f"  Delta  {d_psnr:>+7.2f} {d_cpsnr:>+10.2f} {d_fc:>+11.3f} {d_lpips:>+7.4f} | "
              f"{d_dcp:>+10.2f} {d_dfc:>+8.3f}")
        print()


if __name__ == "__main__":
    main()
