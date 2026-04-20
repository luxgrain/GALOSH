#!/usr/bin/env python3
"""
GALOSH_B + chroma color-shift fix benchmark.
Runs two parameter sets: sigma_L=0.5/sigma_C=1.0 and sigma_L=1.0/sigma_C=1.0.
Uses rawdenoise_v7.exe (GALOSH_B base + higher chroma Wiener floor).

Saves images as *_GALOSH_B2_05_10.png and *_GALOSH_B2_10_10.png.
Saves metrics to results/v7_{ls}_{cs}_{dataset}_results.json.
"""
import numpy as np
import subprocess
import os
import sys
import json
import time
import torch
import argparse
from pathlib import Path
from skimage.io import imread, imsave
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import uniform_filter

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

BASE = Path(__file__).parent.parent
EXE = str(BASE / "standalone" / "rawdenoise_v7.exe")
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

DATASETS = {
    "kodak":   {"path": BASE / "datasets" / "kodak",   "glob": "kodim*.png", "max": 24},
    "cbsd68":  {"path": BASE / "datasets" / "cbsd68",  "glob": "*.png",      "max": 68},
    "mcmaster":{"path": BASE / "datasets" / "mcmaster", "glob": "*.png",     "max": 18},
}

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

# Metrics
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


def run_galosh(noisy_raw, w, h, alpha, sigma_sq, ls, cs, uid="0"):
    inp = str(BASE / "standalone" / f"tmp_v7_{uid}.bin")
    out = str(BASE / "standalone" / f"tmp_v7_out_{uid}.bin")
    noisy_raw.tofile(inp)
    cmd = [EXE, inp, out, str(w), str(h), "galosh",
           "1.0", str(ls), str(cs), str(alpha), str(sigma_sq), "1", "25"]
    result = subprocess.run(cmd, capture_output=True, timeout=180)
    try: os.remove(inp)
    except: pass
    if result.returncode != 0:
        raise RuntimeError(f"GALOSH failed: {result.stderr.decode()}")
    den = np.fromfile(out, dtype=np.float32).reshape(h, w)
    try: os.remove(out)
    except: pass
    return den


def run_benchmark(ds_name, ls, cs):
    ds = DATASETS[ds_name]
    IMGDIR = ds["path"]
    OUTDIR = BASE / f"comparison_images_v5_{ds_name}"
    OUTDIR.mkdir(exist_ok=True)

    images = sorted(IMGDIR.glob(ds["glob"]))[:ds["max"]]
    n_images = len(images)
    isos = [400, 800, 1600, 3200, 6400, 12800]

    ls_tag = str(ls).replace(".", "")
    cs_tag = str(cs).replace(".", "")
    method_name = f"GALOSH_B2_{ls_tag}_{cs_tag}"

    print(f"\n{'='*60}")
    print(f"GALOSH_B2 benchmark [{ds_name}] sigma_L={ls} sigma_C={cs}")
    print(f"  ({n_images} images x {len(isos)} ISOs)")
    print(f"  exe: {EXE}")
    print(f"{'='*60}\n")

    all_results = []

    for iso in isos:
        t_iso_start = time.time()
        gain = iso / 100.0
        alpha = 0.001 * gain
        sigma_sq = (0.002 * gain) ** 2

        metrics_acc = {k: [] for k in ["psnr", "ssim", "lpips", "dists", "niqe"]}

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

            uid = f"{ds_name}_{ls_tag}{cs_tag}_iso{iso}_img{idx}"
            den_raw = run_galosh(noisy_raw, w, h, alpha, sigma_sq, ls, cs, uid=uid)
            den_srgb = raw_to_srgb(den_raw)

            p = psnr(gt_srgb, den_srgb)
            s = ssim_rgb(gt_srgb, den_srgb)
            lp = compute_lpips(gt_srgb, den_srgb)
            di = compute_dists(gt_srgb, den_srgb)
            nq = compute_niqe(den_srgb)

            metrics_acc["psnr"].append(p)
            metrics_acc["ssim"].append(s)
            metrics_acc["lpips"].append(lp)
            metrics_acc["dists"].append(di)
            metrics_acc["niqe"].append(nq)

            # Save all images as PNG
            iso_dir = OUTDIR / f"iso{iso}"
            iso_dir.mkdir(exist_ok=True)
            imsave(str(iso_dir / f"{img_path.stem}_{method_name}.png"),
                   (np.clip(den_srgb, 0, 1) * 255).astype(np.uint8))

            # Also save GT and Noisy for reference (only once per ISO)
            if idx == 0 or img_path.stem == "kodim15":
                noisy_srgb = raw_to_srgb(noisy_raw)
                gt_path = iso_dir / f"{img_path.stem}_GT.png"
                noisy_path = iso_dir / f"{img_path.stem}_Noisy.png"
                if not gt_path.exists():
                    imsave(str(gt_path), (np.clip(gt_srgb, 0, 1) * 255).astype(np.uint8))
                if not noisy_path.exists():
                    imsave(str(noisy_path), (np.clip(noisy_srgb, 0, 1) * 255).astype(np.uint8))

            if (idx + 1) % 6 == 0:
                print(f"  ISO {iso}: {idx+1}/{n_images} done", flush=True)

        avg = {k: round(float(np.mean(v)), 4 if k != "psnr" else 2)
               for k, v in metrics_acc.items()}

        elapsed = time.time() - t_iso_start
        print(f"ISO {iso:5d} ({elapsed:.0f}s): PSNR={avg['psnr']:.2f} SSIM={avg['ssim']:.4f} "
              f"LPIPS={avg['lpips']:.4f} DISTS={avg['dists']:.4f} NIQE={avg['niqe']:.4f}")

        all_results.append({"iso": iso, **{f"{method_name}_{k}": v for k, v in avg.items()}})

    # Save
    out_path = RESULTS / f"v7_{ls_tag}_{cs_tag}_{ds_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="GALOSH_B2 (color-shift fix) benchmark")
    parser.add_argument("--dataset", "-d", default="kodak",
                        choices=list(DATASETS.keys()))
    args = parser.parse_args()

    # Preload metrics
    print("Loading metrics...", end=" ", flush=True)
    compute_lpips(np.zeros((64,64,3), dtype=np.float32),
                  np.zeros((64,64,3), dtype=np.float32))
    compute_dists(np.zeros((64,64,3), dtype=np.float32),
                  np.zeros((64,64,3), dtype=np.float32))
    try: compute_niqe(np.ones((64,64,3), dtype=np.float32) * 0.5)
    except: pass
    print("OK")

    # Run both parameter sets
    run_benchmark(args.dataset, ls=0.5, cs=1.0)
    run_benchmark(args.dataset, ls=1.0, cs=1.0)


if __name__ == "__main__":
    main()
