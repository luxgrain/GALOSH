#!/usr/bin/env python3
"""
Add BM3D-CFA (s=0.5) results to existing v5 benchmark JSON files.
Only computes the missing method, reuses existing noise/GT from the same pipeline.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.io import imread, imsave
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import uniform_filter

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

n_cores = os.cpu_count() or 4
torch.set_num_threads(max(n_cores // 2, 2))

BASE = Path(__file__).parent.parent
EXE = str(BASE / "standalone" / "rawdenoise_v4.exe")
RESULTS = BASE / "results"

DATASETS = {
    "kodak":   {"path": BASE / "datasets" / "kodak",   "glob": "kodim*.png", "max": 24},
    "cbsd68":  {"path": BASE / "datasets" / "cbsd68",  "glob": "*.png",      "max": 68},
}

METHOD_KEY = "BM3D-CFA_(s=0.5)"
METHOD_NAME = "BM3D-CFA (s=0.5)"

# ===================== Utilities (same as v5) =====================

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

METRIC_KEYS = ["psnr", "ssim", "lpips", "dists", "niqe"]

def run_bm3dcfa_s05(noisy_raw, w, h, alpha, sigma_sq, uid="0"):
    inp = str(BASE / "standalone" / f"tmp_s05_{uid}.bin")
    out = str(BASE / "standalone" / f"tmp_s05_out_{uid}.bin")
    noisy_raw.tofile(inp)
    cmd = [EXE, inp, out, str(w), str(h), "bm3dcfa",
           "0.5", "1.0", "1.0", str(alpha), str(sigma_sq), "1", "25"]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    try: os.remove(inp)
    except: pass
    if result.returncode != 0:
        raise RuntimeError(f"C standalone failed: {result.stderr.decode()}")
    den = np.fromfile(out, dtype=np.float32).reshape(h, w)
    try: os.remove(out)
    except: pass
    return den


def process_one(img_path, idx, iso, alpha, sigma_sq):
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

    uid = f"iso{iso}_img{idx}_s05"
    den = run_bm3dcfa_s05(noisy_raw, w, h, alpha, sigma_sq, uid=uid)
    den_srgb = raw_to_srgb(den)

    p = psnr(gt_srgb, den_srgb)
    s = ssim_rgb(gt_srgb, den_srgb)
    lp = compute_lpips(gt_srgb, den_srgb)
    di = compute_dists(gt_srgb, den_srgb)
    nq = compute_niqe(den_srgb)

    return {"psnr": p, "ssim": s, "lpips": lp, "dists": di, "niqe": nq}, den_srgb, img_path.stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, choices=list(DATASETS.keys()))
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    ds_name = args.dataset
    IMGDIR = ds["path"]
    OUTDIR = BASE / f"comparison_images_v5_{ds_name}"
    OUTDIR.mkdir(exist_ok=True)

    isos = [400, 800, 1600, 3200, 6400, 12800]
    images = sorted(IMGDIR.glob(ds["glob"]))[:ds["max"]]
    n_images = len(images)

    # Load existing results
    json_path = RESULTS / f"v5_{ds_name}_results.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        # Handle both formats
        if isinstance(data, dict) and "quality" in data:
            all_results = data["quality"]
            extra = {k: v for k, v in data.items() if k != "quality"}
        elif isinstance(data, list):
            all_results = data
            extra = {}
        else:
            all_results = []
            extra = {}
    else:
        print(f"No existing results at {json_path}")
        return

    print(f"Adding BM3D-CFA (s=0.5) to {ds_name}: {n_images} images x {len(isos)} ISOs")

    # Preload metrics
    print("Preloading metrics...", end=" ", flush=True)
    compute_lpips(np.zeros((64,64,3), dtype=np.float32), np.zeros((64,64,3), dtype=np.float32))
    compute_dists(np.zeros((64,64,3), dtype=np.float32), np.zeros((64,64,3), dtype=np.float32))
    try:
        compute_niqe(np.ones((64,64,3), dtype=np.float32) * 0.5)
    except:
        pass
    print("OK")

    N_PARALLEL = 4

    for iso_idx, iso in enumerate(isos):
        t0 = time.time()
        gain = iso / 100.0
        alpha = 0.001 * gain
        sigma_sq = (0.002 * gain) ** 2

        metrics_list = {k: [] for k in METRIC_KEYS}

        with ThreadPoolExecutor(max_workers=N_PARALLEL) as pool:
            futures = {}
            for idx, img_path in enumerate(images):
                fut = pool.submit(process_one, img_path, idx, iso, alpha, sigma_sq)
                futures[fut] = idx

            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    met, den_srgb, stem = fut.result()
                    for k in METRIC_KEYS:
                        metrics_list[k].append(met[k])
                    # Save image
                    iso_dir = OUTDIR / f"iso{iso}"
                    iso_dir.mkdir(exist_ok=True)
                    imsave(str(iso_dir / f"{stem}_BM3D-CFA_(s=0.5).png"),
                           (np.clip(den_srgb, 0, 1) * 255).astype(np.uint8))
                except Exception as e:
                    print(f"  Image {idx} failed ISO {iso}: {e}")

                done = sum(1 for k2 in metrics_list["psnr"])
                if done % 6 == 0:
                    print(f"  ISO {iso}: {done}/{n_images}", flush=True)

        # Compute averages
        avg = {}
        for k in METRIC_KEYS:
            vals = [x for x in metrics_list[k] if not (isinstance(x, float) and x != x)]
            avg[k] = float(np.mean(vals)) if vals else float('nan')

        # Add to existing result row
        if iso_idx < len(all_results):
            row = all_results[iso_idx]
        else:
            row = {"iso": iso}
            all_results.append(row)

        for k in METRIC_KEYS:
            dec = 2 if k == "psnr" else 4
            val = avg[k]
            row[f"{METHOD_KEY}_{k}"] = round(val, dec) if not np.isnan(val) else None

        elapsed = time.time() - t0
        print(f"  ISO {iso} ({elapsed:.0f}s): PSNR={avg['psnr']:.2f} SSIM={avg['ssim']:.4f} "
              f"LPIPS={avg['lpips']:.4f} DISTS={avg['dists']:.4f} NIQE={avg['niqe']:.4f}")

    # Save merged results
    if extra:
        save_data = {"quality": all_results, **extra}
    else:
        save_data = all_results
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nMerged results saved to {json_path}")


if __name__ == "__main__":
    main()
