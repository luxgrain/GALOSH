#!/usr/bin/env python3
"""
Add classical rule-based denoisers to comparison:
  1. NLM (sRGB)         — Non-Local Means, per-channel on sRGB
  2. Wavelet BayesShrink (sRGB) — DWT BayesShrink, per-channel on sRGB
  3. BM3D (sRGB)        — BM3D per-channel on sRGB (not RAW-aware)

Saves images as *_{method}.png into comparison_images_v5_{dataset}/iso{N}/
Saves metrics to results/v6c_{dataset}_results.json
Skips images that already exist.
"""
import numpy as np
import os
import sys
import json
import time
import torch
import argparse
import cv2
import pywt
import bm3d as bm3d_pkg
from pathlib import Path
from skimage.io import imread, imsave
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import uniform_filter

BASE = Path(__file__).parent.parent
RESULTS = BASE / "results"

DATASETS = {
    "kodak":   {"path": BASE / "datasets" / "kodak",   "glob": "kodim*.png", "max": 24},
    "cbsd68":  {"path": BASE / "datasets" / "cbsd68",  "glob": "*.png",      "max": 68},
    "mcmaster":{"path": BASE / "datasets" / "mcmaster", "glob": "*.png",     "max": 18},
}

# ---------- colour helpers ----------
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
        if c == 0:   mask[0::2, 0::2] = 1
        elif c == 1: mask[0::2, 1::2] = 1; mask[1::2, 0::2] = 1
        else:        mask[1::2, 1::2] = 1
        num = uniform_filter(rgb[:, :, c], size=3)
        den = uniform_filter(mask, size=3)
        rgb[:, :, c] = np.where(mask > 0, rgb[:, :, c], num / np.maximum(den, 1e-10))
    return np.clip(rgb, 0, 1)

def synth_noise(clean_raw, iso, rng):
    gain = iso / 100.0
    alpha = 0.001 * gain
    sigma_read = 0.002 * gain
    sigma_sq = sigma_read ** 2
    rate = np.maximum(clean_raw / max(alpha, 1e-10), 0.0)
    noisy = rng.poisson(rate).astype(np.float32) * alpha
    noisy += rng.standard_normal(clean_raw.shape).astype(np.float32) * sigma_read
    return np.clip(noisy, 0, 1).astype(np.float32), alpha, sigma_sq

def raw_to_srgb(bayer):
    return np.clip(linear_to_srgb(demosaic_bilinear(bayer)), 0, 1).astype(np.float32)

# ---------- noise estimation ----------
def estimate_sigma_mad(img_ch):
    """Estimate noise sigma via MAD of finest-scale wavelet coefficients.
    Robust estimator: sigma = median(|d|) / 0.6745
    Uses horizontal detail coefficients from 1-level Haar DWT.
    """
    coeffs = pywt.dwt2(img_ch.astype(np.float64), 'haar')
    _, (_, _, dHH) = coeffs
    sigma = np.median(np.abs(dHH)) / 0.6745
    return max(float(sigma), 1e-4)

def estimate_sigma_rgb(img):
    """Estimate per-image noise sigma (average across channels)."""
    sigmas = [estimate_sigma_mad(img[:, :, c]) for c in range(3)]
    return float(np.mean(sigmas))

# ---------- Method 1: NLM (sRGB, per-channel) ----------
def method_nlm(noisy_srgb, oracle_sigma):
    """Non-Local Means denoising, applied per-channel on sRGB.
    h parameter set to oracle sigma (computed from known noise model).
    templateWindowSize=7, searchWindowSize=21 (OpenCV defaults).
    Oracle sigma used for fair comparison: BM3D-CFA/GALOSH also receive
    true noise parameters (alpha, sigma_sq).
    """
    # Convert to uint8 for OpenCV NLM (works on 0-255)
    noisy_u8 = (np.clip(noisy_srgb, 0, 1) * 255).astype(np.uint8)
    h = oracle_sigma * 255  # h in pixel intensity scale
    out_channels = []
    for c in range(3):
        den = cv2.fastNlMeansDenoising(noisy_u8[:, :, c], None,
                                        h=h, templateWindowSize=7,
                                        searchWindowSize=21)
        out_channels.append(den)
    result = np.stack(out_channels, axis=2).astype(np.float32) / 255.0
    return np.clip(result, 0, 1)

# ---------- Method 2: Wavelet BayesShrink (sRGB, per-channel) ----------
def bayesshrink_channel(ch, sigma_n, wavelet='db4', level=3):
    """BayesShrink on a single channel with known noise sigma.
    Chang, Yu & Vetterli (2000): Adaptive Wavelet Thresholding for
    Image Denoising and Compression.

    Algorithm:
      1. Multi-level DWT decomposition
      2. Use provided noise sigma (oracle from known noise model)
      3. For each subband: sigma_x = sqrt(max(sigma_y^2 - sigma_n^2, 0))
         where sigma_y = std of subband coefficients
      4. Threshold lambda = sigma_n^2 / sigma_x (BayesShrink)
      5. Soft-threshold each subband
      6. Inverse DWT reconstruction
    """
    coeffs = pywt.wavedec2(ch.astype(np.float64), wavelet, level=level)
    sigma_n = max(sigma_n, 1e-6)

    # Threshold each detail subband (skip approximation coeffs[0])
    new_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        new_detail = []
        for d in detail:
            sigma_y = np.sqrt(max(np.mean(d ** 2), 0))
            sigma_x = np.sqrt(max(sigma_y ** 2 - sigma_n ** 2, 0))
            if sigma_x < 1e-8:
                # Pure noise subband — kill it
                new_detail.append(np.zeros_like(d))
            else:
                # BayesShrink threshold
                lam = sigma_n ** 2 / sigma_x
                # Soft thresholding
                new_detail.append(np.sign(d) * np.maximum(np.abs(d) - lam, 0))
        new_coeffs.append(tuple(new_detail))

    return pywt.waverec2(new_coeffs, wavelet).astype(np.float64)

def method_wavelet_bayesshrink(noisy_srgb, oracle_sigma):
    """Wavelet BayesShrink denoising, per-channel on sRGB.
    db4 wavelet, 3-level decomposition, soft thresholding.
    Oracle sigma used for fair comparison with RAW-domain methods.
    """
    h, w = noisy_srgb.shape[:2]
    out_channels = []
    for c in range(3):
        den = bayesshrink_channel(noisy_srgb[:, :, c], sigma_n=oracle_sigma)
        # waverec2 may produce slightly different size due to padding
        den = den[:h, :w]
        out_channels.append(den)
    result = np.stack(out_channels, axis=2).astype(np.float32)
    return np.clip(result, 0, 1)

# ---------- Method 3: BM3D (sRGB, per-channel) ----------
def method_bm3d_srgb(noisy_srgb, oracle_sigma):
    """BM3D denoising applied per-channel on sRGB.
    Uses the bm3d Python package (Mäkinen et al.).
    Not RAW-aware — operates after demosaic on each RGB channel independently.
    Oracle sigma used for fair comparison with RAW-domain methods.
    """
    out_channels = []
    for c in range(3):
        ch = noisy_srgb[:, :, c].astype(np.float64)
        den = bm3d_pkg.bm3d(ch, sigma_psd=oracle_sigma, stage_arg=bm3d_pkg.BM3DStages.ALL_STAGES)
        out_channels.append(den)
    result = np.stack(out_channels, axis=2).astype(np.float32)
    return np.clip(result, 0, 1)

# ---------- Metrics ----------
def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0

def ssim_rgb(a, b):
    return float(ssim(a, b, data_range=1.0, channel_axis=2))

_lpips_fn = None
def compute_lpips(a, b):
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net='alex', verbose=False)
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
    tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
    with torch.no_grad():
        return _lpips_fn(ta, tb).item()

_dists_fn = None
def compute_dists(a, b):
    global _dists_fn
    if _dists_fn is None:
        import pyiqa
        _dists_fn = pyiqa.create_metric('dists', device='cpu')
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float()
    tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        return _dists_fn(ta, tb).item()

_niqe_fn = None
def compute_niqe(a):
    global _niqe_fn
    if _niqe_fn is None:
        import pyiqa
        _niqe_fn = pyiqa.create_metric('niqe', device='cpu')
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        return _niqe_fn(ta).item()


METHODS = {
    "NLM":              method_nlm,
    "Wavelet_BayesShrink": method_wavelet_bayesshrink,
    "BM3D_sRGB":        method_bm3d_srgb,
}

def compute_oracle_sigma(gt_srgb, noisy_srgb):
    """Compute oracle noise sigma as RMSE between GT and noisy in sRGB.
    Fair comparison: BM3D-CFA and GALOSH receive true raw noise parameters
    (alpha, sigma_sq), so sRGB-domain methods should also get accurate sigma.
    """
    return float(np.sqrt(np.mean((gt_srgb.astype(np.float64) - noisy_srgb.astype(np.float64)) ** 2)))


def main():
    parser = argparse.ArgumentParser(description="Add classical rule-based denoisers")
    parser.add_argument("--dataset", "-d", default="kodak",
                        choices=list(DATASETS.keys()))
    parser.add_argument("--methods", "-m", nargs="*", default=None,
                        help="Which methods to run (default: all)")
    args = parser.parse_args()

    ds_name = args.dataset
    ds = DATASETS[ds_name]
    IMGDIR = ds["path"]
    OUTDIR = BASE / f"comparison_images_v5_{ds_name}"
    OUTDIR.mkdir(exist_ok=True)

    methods_to_run = args.methods if args.methods else list(METHODS.keys())
    # Validate
    for m in methods_to_run:
        if m not in METHODS:
            print(f"Unknown method: {m}. Available: {list(METHODS.keys())}")
            sys.exit(1)

    images = sorted(IMGDIR.glob(ds["glob"]))[:ds["max"]]
    n_images = len(images)
    isos = [400, 800, 1600, 3200, 6400, 12800]

    print(f"Classical denoisers benchmark [{ds_name}] ({n_images} images x {len(isos)} ISOs)")
    print(f"Methods: {methods_to_run}")
    print(f"Images saved to: {OUTDIR}")
    print()

    # Preload metrics
    print("Loading LPIPS...", end=" ", flush=True)
    compute_lpips(np.zeros((64,64,3), dtype=np.float32),
                  np.zeros((64,64,3), dtype=np.float32))
    print("OK")
    print("Loading DISTS...", end=" ", flush=True)
    compute_dists(np.zeros((64,64,3), dtype=np.float32),
                  np.zeros((64,64,3), dtype=np.float32))
    print("OK")
    print("Loading NIQE...", end=" ", flush=True)
    try:
        compute_niqe(np.ones((64,64,3), dtype=np.float32) * 0.5)
        print("OK")
    except:
        print("SKIP")
    print()

    all_results = []

    for iso in isos:
        t_iso_start = time.time()
        gain = iso / 100.0
        alpha = 0.001 * gain
        sigma_sq = (0.002 * gain) ** 2

        # Per-method metric accumulators
        metrics_acc = {m: {k: [] for k in ["psnr", "ssim", "lpips", "dists", "niqe"]}
                       for m in methods_to_run}

        for idx, img_path in enumerate(images):
            rng = np.random.default_rng(42 + idx)
            img = imread(str(img_path)).astype(np.float32) / 255.0
            h, w = img.shape[:2]
            h -= h % 2; w -= w % 2
            img = img[:h, :w]
            linear = inv_srgb(img).astype(np.float32)

            clean_raw = np.zeros((h, w), dtype=np.float32)
            clean_raw[0::2, 0::2] = linear[0::2, 0::2, 0]
            clean_raw[0::2, 1::2] = linear[0::2, 1::2, 1]
            clean_raw[1::2, 0::2] = linear[1::2, 0::2, 1]
            clean_raw[1::2, 1::2] = linear[1::2, 1::2, 2]

            noisy_raw, _, _ = synth_noise(clean_raw, iso, rng)
            gt_srgb = raw_to_srgb(clean_raw)
            noisy_srgb = raw_to_srgb(noisy_raw)

            # Oracle sigma: RMSE(gt, noisy) in sRGB — fair comparison since
            # BM3D-CFA/GALOSH receive true raw noise params (alpha, sigma_sq)
            oracle_sigma = compute_oracle_sigma(gt_srgb, noisy_srgb)

            iso_dir = OUTDIR / f"iso{iso}"
            iso_dir.mkdir(exist_ok=True)

            for method_name in methods_to_run:
                out_path = iso_dir / f"{img_path.stem}_{method_name}.png"

                try:
                    den_srgb = METHODS[method_name](noisy_srgb, oracle_sigma)
                    imsave(str(out_path),
                           (np.clip(den_srgb, 0, 1) * 255).astype(np.uint8))
                except Exception as e:
                    print(f"  {method_name} FAIL {img_path.stem} ISO{iso}: {e}")
                    continue

                # Compute metrics
                p = psnr(gt_srgb, den_srgb)
                s = ssim_rgb(gt_srgb, den_srgb)
                lp = compute_lpips(gt_srgb, den_srgb)
                di = compute_dists(gt_srgb, den_srgb)
                nq = compute_niqe(den_srgb)

                metrics_acc[method_name]["psnr"].append(p)
                metrics_acc[method_name]["ssim"].append(s)
                metrics_acc[method_name]["lpips"].append(lp)
                metrics_acc[method_name]["dists"].append(di)
                metrics_acc[method_name]["niqe"].append(nq)

            if (idx + 1) % 6 == 0:
                print(f"  ISO {iso}: {idx+1}/{n_images} done", flush=True)

        # Summarize
        iso_result = {"iso": iso}
        elapsed = time.time() - t_iso_start
        parts = []
        for m in methods_to_run:
            avg = {k: round(float(np.mean(v)), 4 if k != "psnr" else 2)
                   for k, v in metrics_acc[m].items() if len(v) > 0}
            for k, v in avg.items():
                iso_result[f"{m}_{k}"] = v
            if "psnr" in avg:
                parts.append(f"{m}: PSNR={avg['psnr']:.2f} LPIPS={avg.get('lpips', 0):.4f}")

        all_results.append(iso_result)
        print(f"ISO {iso:5d} ({elapsed:.0f}s): {' | '.join(parts)}")

    # Save results
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"v6c_{ds_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
