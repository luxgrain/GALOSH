#!/usr/bin/env python3
"""
Region-specific analysis: dark/mid/bright regions
+ Chroma PSNR (Lab a*b* channels)
+ Per-region LPIPS
Kodak 24, ISO 3200/6400/12800 (where overall LPIPS favors BM3D-CFA)
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
EXE = str(BASE / "standalone" / "rawdenoise.exe")
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

def gat_forward(x, alpha, sigma_sq):
    return (2.0 / alpha) * np.sqrt(np.maximum(alpha * np.maximum(x, 0) + 0.375 * alpha**2 + sigma_sq, 0))

def gat_inverse(D, alpha, sigma_sq):
    D = np.maximum(D, 1e-8)
    D_inv = 1.0 / D
    y = 0.25*D*D + 0.25*1.2247448713916*D_inv - 11.0/8.0*D_inv**2 + 5.0/8.0*1.2247448713916*D_inv**3 - 1.0/8.0
    return np.maximum(alpha * y - sigma_sq / alpha, 0)

_lpips_fn = None
def get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(DEVICE)
    return _lpips_fn

def compute_lpips_masked(gt_srgb, den_srgb, mask):
    """Compute LPIPS on masked region (set non-masked to GT to neutralize)."""
    fn = get_lpips()
    # Blend: where mask=0, use GT (no penalty); where mask=1, use denoised
    blended = gt_srgb * (1 - mask[:,:,np.newaxis]) + den_srgb * mask[:,:,np.newaxis]
    ta = torch.from_numpy(gt_srgb.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE) * 2 - 1
    tb = torch.from_numpy(blended.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE) * 2 - 1
    with torch.no_grad():
        return fn(ta, tb).item()

def chroma_psnr(gt_srgb, den_srgb):
    gt_lab = rgb2lab(np.clip(gt_srgb, 0, 1))
    den_lab = rgb2lab(np.clip(den_srgb, 0, 1))
    chroma_mse = np.mean((gt_lab[:,:,1:] - den_lab[:,:,1:]) ** 2)
    return 10 * np.log10(256.0**2 / max(chroma_mse, 1e-10))

def chroma_psnr_masked(gt_srgb, den_srgb, mask):
    gt_lab = rgb2lab(np.clip(gt_srgb, 0, 1))
    den_lab = rgb2lab(np.clip(den_srgb, 0, 1))
    idx = mask > 0.5
    if idx.sum() < 100:
        return float('nan')
    diff = gt_lab[idx, 1:] - den_lab[idx, 1:]
    chroma_mse = np.mean(diff ** 2)
    return 10 * np.log10(256.0**2 / max(chroma_mse, 1e-10))

def false_color_score(gt_srgb, den_srgb, mask):
    """Measure false color: std of (a*,b*) error in masked region.
    Higher = more false color artifacts."""
    gt_lab = rgb2lab(np.clip(gt_srgb, 0, 1))
    den_lab = rgb2lab(np.clip(den_srgb, 0, 1))
    idx = mask > 0.5
    if idx.sum() < 100:
        return float('nan')
    a_err = den_lab[idx, 1] - gt_lab[idx, 1]
    b_err = den_lab[idx, 2] - gt_lab[idx, 2]
    # RMS of chroma error
    return float(np.sqrt(np.mean(a_err**2 + b_err**2)))

def run_standalone(noisy, w, h, method, alpha, sq, sw=25):
    inp = str(BASE / "standalone" / "tmp_ra.bin")
    out = str(BASE / "standalone" / f"tmp_ra_{method}.bin")
    noisy.tofile(inp)
    env = os.environ.copy(); env["PATH"] = MSYS_PATH
    subprocess.run([EXE, inp, out, str(w), str(h), method,
                   "1.0", "1.0", "1.0", str(alpha), str(sq), "1", str(sw)],
                  capture_output=True, timeout=300, env=env)
    return np.fromfile(out, dtype=np.float32).reshape(h, w)

import bm3d as bm3d_pkg
from scipy.signal import convolve2d

def method_bm3d_cfa(noisy_bayer, alpha, sq):
    h, w = noisy_bayer.shape
    offs = [(0,0),(0,1),(1,0),(1,1)]
    ch_gat = []
    for dy, dx in offs:
        ch = noisy_bayer[dy::2, dx::2].copy()
        ch_gat.append(gat_forward(ch, alpha, sq).astype(np.float64))
    stack = np.stack(ch_gat, axis=2)
    kernel = np.array([1,-2,1], dtype=np.float64).reshape(1,3)
    laps = convolve2d(ch_gat[0], kernel, mode='valid')
    sigma_est = max(np.median(np.abs(laps)) / 0.6745 / np.sqrt(6), 0.1)
    denoised = bm3d_pkg.bm3d(stack, sigma_psd=sigma_est, profile='np')
    output = np.zeros_like(noisy_bayer)
    gat_max = gat_forward(1.0, alpha, sq) * 1.2
    for i, (dy, dx) in enumerate(offs):
        ch = np.clip(denoised[:,:,i], 0, gat_max)
        output[dy::2, dx::2] = np.minimum(gat_inverse(ch, alpha, sq), 1.0).astype(np.float32)
    return output


def main():
    images = sorted(KODAK.glob("kodim*.png"))[:24]
    isos = [3200, 6400, 12800]
    methods = ["BM3D-PC", "BM3D-CFA", "Ours(V2)"]

    print("Region-specific analysis: dark / mid / bright")
    print("Metrics: Chroma PSNR (higher=better), False Color RMS (lower=better), LPIPS (lower=better)")
    print()

    for iso in isos:
        region_metrics = {m: {r: {"cpsnr": [], "fc": [], "lpips": []}
                              for r in ["dark", "mid", "bright", "all"]}
                          for m in methods}

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

            # Luminance for region masks (from GT linear)
            lum = 0.2126 * linear[:,:,0] + 0.7152 * linear[:,:,1] + 0.0722 * linear[:,:,2]
            dark_mask = (lum < 0.08).astype(np.float32)
            mid_mask  = ((lum >= 0.08) & (lum < 0.4)).astype(np.float32)
            bright_mask = (lum >= 0.4).astype(np.float32)
            all_mask = np.ones_like(lum)

            results = {}
            results["BM3D-PC"] = run_standalone(noisy, w, h, "perchannel", alpha, sq)
            results["Ours(V2)"] = run_standalone(noisy, w, h, "ours", alpha, sq, 25)
            try:
                results["BM3D-CFA"] = method_bm3d_cfa(noisy, alpha, sq)
            except:
                results["BM3D-CFA"] = noisy

            for m_name, den_bayer in results.items():
                den_srgb = np.clip(linear_to_srgb(demosaic_bilinear(den_bayer)), 0, 1).astype(np.float32)
                for rname, rmask in [("dark", dark_mask), ("mid", mid_mask),
                                      ("bright", bright_mask), ("all", all_mask)]:
                    cp = chroma_psnr_masked(gt_srgb, den_srgb, rmask)
                    fc = false_color_score(gt_srgb, den_srgb, rmask)
                    lp = compute_lpips_masked(gt_srgb, den_srgb, rmask)
                    region_metrics[m_name][rname]["cpsnr"].append(cp)
                    region_metrics[m_name][rname]["fc"].append(fc)
                    region_metrics[m_name][rname]["lpips"].append(lp)

            if (idx + 1) % 8 == 0:
                print(f"  ISO {iso}: {idx+1}/24")

        print(f"\nISO {iso}:")
        for rname in ["dark", "mid", "bright", "all"]:
            print(f"\n  Region: {rname.upper()}")
            print(f"    {'Method':<12} {'ChromaPSNR':>10} {'FalseColor':>11} {'LPIPS':>7}")
            print(f"    {'-'*42}")
            for m in methods:
                rm = region_metrics[m][rname]
                cp = np.nanmean(rm["cpsnr"])
                fc = np.nanmean(rm["fc"])
                lp = np.mean(rm["lpips"])
                print(f"    {m:<12} {cp:>10.2f} {fc:>11.3f} {lp:>7.4f}")

            # Delta Ours vs CFA
            cp_ours = np.nanmean(region_metrics["Ours(V2)"][rname]["cpsnr"])
            cp_cfa  = np.nanmean(region_metrics["BM3D-CFA"][rname]["cpsnr"])
            fc_ours = np.nanmean(region_metrics["Ours(V2)"][rname]["fc"])
            fc_cfa  = np.nanmean(region_metrics["BM3D-CFA"][rname]["fc"])
            lp_ours = np.mean(region_metrics["Ours(V2)"][rname]["lpips"])
            lp_cfa  = np.mean(region_metrics["BM3D-CFA"][rname]["lpips"])
            print(f"    Ours vs CFA: dChromaPSNR={cp_ours-cp_cfa:+.2f}  dFalseColor={fc_ours-fc_cfa:+.3f}  dLPIPS={lp_ours-lp_cfa:+.4f}")
        print()


if __name__ == "__main__":
    main()
