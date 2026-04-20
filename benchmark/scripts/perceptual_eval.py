#!/usr/bin/env python3
"""
Perceptual quality evaluation for raw denoising.
Metrics:
  1. LPIPS (Learned Perceptual Image Patch Similarity)
  2. Chroma PSNR (color noise removal in Lab space)
  3. Banding detection (gradient histogram entropy in smooth regions)
"""
import numpy as np
import subprocess
import os
import json
from pathlib import Path
from skimage.io import imread
from skimage.color import rgb2lab
import torch

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

BASE = Path(__file__).parent.parent
EXE = str(BASE / "standalone" / "rawdenoise.exe")
KODAK = BASE / "datasets" / "kodak"
RESULTS = BASE / "results"

# Simple bilinear demosaic for converting Bayer to RGB
def demosaic_bilinear(bayer):
    """Minimal bilinear demosaic for RGGB Bayer."""
    from scipy.ndimage import uniform_filter
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
        den = np.maximum(den, 1e-10)
        rgb[:, :, c] = np.where(mask > 0, rgb[:, :, c], num / den)
    return np.clip(rgb, 0, 1)


def linear_to_srgb(img):
    return np.where(img <= 0.0031308, 12.92 * img,
                    1.055 * np.power(np.maximum(img, 0), 1/2.4) - 0.055)


def inv_srgb(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


# --- Metric 1: LPIPS ---
_lpips_fn = None
def get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net='alex', verbose=False)
    return _lpips_fn


def compute_lpips(img_a, img_b):
    """Compute LPIPS between two sRGB images [H,W,3] in [0,1]."""
    fn = get_lpips()
    # LPIPS expects [B,C,H,W] in [-1,1]
    a = torch.from_numpy(img_a.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
    b = torch.from_numpy(img_b.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
    with torch.no_grad():
        d = fn(a, b)
    return d.item()


# --- Metric 2: Chroma PSNR ---
def chroma_psnr(gt_srgb, den_srgb):
    """PSNR on chroma (a,b) channels in CIE Lab space."""
    gt_lab = rgb2lab(np.clip(gt_srgb, 0, 1))
    den_lab = rgb2lab(np.clip(den_srgb, 0, 1))
    # a,b channels range roughly [-128, 128]
    chroma_gt = gt_lab[:, :, 1:]  # [H,W,2]
    chroma_den = den_lab[:, :, 1:]
    mse = np.mean((chroma_gt - chroma_den) ** 2)
    if mse < 1e-10:
        return 100.0
    # Max range for a,b is ~256
    return 10 * np.log10(256.0 ** 2 / mse)


# --- Metric 3: Banding detection ---
def banding_score(srgb_img):
    """Detect banding via gradient histogram entropy in smooth regions.
    Lower entropy = more banding (quantized gradients).
    Higher entropy = smoother gradients (no banding).
    Returns entropy (higher is better)."""
    gray = 0.299 * srgb_img[:, :, 0] + 0.587 * srgb_img[:, :, 1] + 0.114 * srgb_img[:, :, 2]

    # Horizontal gradient
    gx = np.diff(gray, axis=1)

    # Find smooth regions (low variance blocks)
    h, w = gray.shape
    block = 16
    smooth_grads = []
    for by in range(0, h - block, block):
        for bx in range(0, w - block, block):
            patch = gray[by:by+block, bx:bx+block]
            if patch.std() < 0.05:  # smooth region
                g = gx[by:by+block, bx:bx+block-1].ravel()
                smooth_grads.extend(g.tolist())

    if len(smooth_grads) < 100:
        return float('nan')  # Not enough smooth regions

    grads = np.array(smooth_grads)
    # Histogram with fine bins
    hist, _ = np.histogram(grads, bins=256, range=(-0.02, 0.02))
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def main():
    isos = [400, 800, 1600, 3200, 6400]
    images = sorted(KODAK.glob("kodim*.png"))[:12]  # 12 images for speed

    if not images:
        print("ERROR: No Kodak images found")
        return

    print(f"Perceptual evaluation: {len(images)} images x {len(isos)} ISOs")
    print(f"Metrics: LPIPS (lower=better), Chroma PSNR (higher=better), Banding entropy (higher=better)")
    print()

    all_results = []

    for iso in isos:
        metrics = {k: [] for k in [
            "lpips_ours", "lpips_pc", "lpips_noisy",
            "cpsnr_ours", "cpsnr_pc", "cpsnr_noisy",
            "band_ours", "band_pc", "band_noisy", "band_gt"
        ]}

        for img_path in images:
            rng = np.random.default_rng(42)
            gain = iso / 100; alpha = 0.001 * gain; sr = 0.002 * gain; sq = sr ** 2

            img = imread(str(img_path)).astype(np.float32) / 255.0
            h, w = img.shape[:2]; h -= h % 2; w -= w % 2; img = img[:h, :w]
            linear = inv_srgb(img).astype(np.float32)

            cr = np.zeros((h, w), dtype=np.float32)
            cr[0::2, 0::2] = linear[0::2, 0::2, 0]
            cr[0::2, 1::2] = linear[0::2, 1::2, 1]
            cr[1::2, 0::2] = linear[1::2, 0::2, 1]
            cr[1::2, 1::2] = linear[1::2, 1::2, 2]

            rate = np.maximum(cr / max(alpha, 1e-10), 0)
            noisy = rng.poisson(rate).astype(np.float32) * alpha
            noisy += rng.standard_normal(cr.shape).astype(np.float32) * sr
            noisy = np.clip(noisy, 0, 1).astype(np.float32)

            inp = str(BASE / "standalone" / "tmp.bin")
            noisy.tofile(inp)

            subprocess.run([EXE, inp, str(BASE/"standalone"/"tmp_o.bin"), str(w), str(h),
                           "ours", "1.0", "1.0", "1.0", str(alpha), str(sq), "1"],
                          capture_output=True, timeout=300)
            subprocess.run([EXE, inp, str(BASE/"standalone"/"tmp_p.bin"), str(w), str(h),
                           "perchannel", "1.0", "1.0", "1.0", str(alpha), str(sq), "1"],
                          capture_output=True, timeout=300)

            ours = np.fromfile(str(BASE/"standalone"/"tmp_o.bin"), dtype=np.float32).reshape(h, w)
            pc = np.fromfile(str(BASE/"standalone"/"tmp_p.bin"), dtype=np.float32).reshape(h, w)

            # Demosaic to sRGB for perceptual metrics
            gt_srgb = np.clip(img, 0, 1)
            noisy_srgb = np.clip(linear_to_srgb(demosaic_bilinear(noisy)), 0, 1).astype(np.float32)
            ours_srgb = np.clip(linear_to_srgb(demosaic_bilinear(ours)), 0, 1).astype(np.float32)
            pc_srgb = np.clip(linear_to_srgb(demosaic_bilinear(pc)), 0, 1).astype(np.float32)

            # LPIPS
            metrics["lpips_noisy"].append(compute_lpips(gt_srgb, noisy_srgb))
            metrics["lpips_ours"].append(compute_lpips(gt_srgb, ours_srgb))
            metrics["lpips_pc"].append(compute_lpips(gt_srgb, pc_srgb))

            # Chroma PSNR
            metrics["cpsnr_noisy"].append(chroma_psnr(gt_srgb, noisy_srgb))
            metrics["cpsnr_ours"].append(chroma_psnr(gt_srgb, ours_srgb))
            metrics["cpsnr_pc"].append(chroma_psnr(gt_srgb, pc_srgb))

            # Banding
            b_gt = banding_score(gt_srgb)
            b_n = banding_score(noisy_srgb)
            b_o = banding_score(ours_srgb)
            b_p = banding_score(pc_srgb)
            if not np.isnan(b_gt):
                metrics["band_gt"].append(b_gt)
                metrics["band_noisy"].append(b_n)
                metrics["band_ours"].append(b_o)
                metrics["band_pc"].append(b_p)

        avg = {k: np.mean(v) if v else float('nan') for k, v in metrics.items()}

        result = {
            "iso": iso,
            "lpips_ours": round(avg["lpips_ours"], 4),
            "lpips_pc": round(avg["lpips_pc"], 4),
            "lpips_noisy": round(avg["lpips_noisy"], 4),
            "cpsnr_ours": round(avg["cpsnr_ours"], 2),
            "cpsnr_pc": round(avg["cpsnr_pc"], 2),
            "band_ours": round(avg["band_ours"], 3) if not np.isnan(avg["band_ours"]) else None,
            "band_pc": round(avg["band_pc"], 3) if not np.isnan(avg["band_pc"]) else None,
            "band_gt": round(avg["band_gt"], 3) if not np.isnan(avg["band_gt"]) else None,
        }
        all_results.append(result)

        dl = avg["lpips_ours"] - avg["lpips_pc"]
        dc = avg["cpsnr_ours"] - avg["cpsnr_pc"]
        db = avg["band_ours"] - avg["band_pc"] if not np.isnan(avg["band_ours"]) else float('nan')

        print(f"ISO {iso:>5}:")
        print(f"  LPIPS:  Ours={avg['lpips_ours']:.4f}  PC={avg['lpips_pc']:.4f}  Δ={dl:+.4f} {'(Ours better)' if dl < 0 else '(PC better)'}")
        print(f"  ChromaPSNR: Ours={avg['cpsnr_ours']:.2f}  PC={avg['cpsnr_pc']:.2f}  Δ={dc:+.2f}")
        print(f"  Banding: Ours={avg['band_ours']:.3f}  PC={avg['band_pc']:.3f}  Δ={db:+.3f} (higher=less banding)")
        print()

    out_path = RESULTS / "perceptual_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
