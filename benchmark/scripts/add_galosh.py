#!/usr/bin/env python3
"""
Add GALOSH results to existing v5 JSON benchmark files.
Runs GALOSH only (no DL methods, no BM3D re-runs).
Merges into v5 JSON -> saves as v6 JSON.
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
EXE_V6 = str(BASE / "standalone" / "rawdenoise_v6.exe")
RESULTS = BASE / "results"

DATASETS = {
    "kodak":   {"path": BASE / "datasets" / "kodak",   "glob": "kodim*.png", "max": 24},
    "cbsd68":  {"path": BASE / "datasets" / "cbsd68",  "glob": "*.png",      "max": 68},
    "mcmaster":{"path": BASE / "datasets" / "mcmaster", "glob": "*.png",     "max": 18},
}

# ===================== Utilities =====================

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

# ===================== Metrics =====================

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

# ===================== GALOSH runner =====================

def run_galosh(noisy_raw, w, h, alpha, sigma_sq, uid="0"):
    inp = str(BASE / "standalone" / f"tmp_galosh_{uid}.bin")
    out = str(BASE / "standalone" / f"tmp_galosh_out_{uid}.bin")
    noisy_raw.tofile(inp)
    cmd = [EXE_V6, inp, out, str(w), str(h), "galosh",
           "1.0", "0.5", "1.0", str(alpha), str(sigma_sq), "1", "25"]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    try: os.remove(inp)
    except: pass
    if result.returncode != 0:
        raise RuntimeError(f"GALOSH failed: {result.stderr.decode()}")
    den = np.fromfile(out, dtype=np.float32).reshape(h, w)
    try: os.remove(out)
    except: pass
    return den


def main():
    parser = argparse.ArgumentParser(description="Add GALOSH to v5 benchmark results")
    parser.add_argument("--dataset", "-d", default="kodak",
                        choices=list(DATASETS.keys()))
    args = parser.parse_args()

    ds_name = args.dataset
    ds = DATASETS[ds_name]
    IMGDIR = ds["path"]

    # Load existing v5 results
    v5_path = RESULTS / f"v5_{ds_name}_results.json"
    if not v5_path.exists():
        print(f"ERROR: {v5_path} not found")
        return
    with open(v5_path) as f:
        v5_data = json.load(f)

    # v5 JSON structure: {"quality": [...], "sequential_timing": {...}}
    v5_quality = v5_data["quality"]
    v5_timing = v5_data.get("sequential_timing", {})

    images = sorted(IMGDIR.glob(ds["glob"]))[:ds["max"]]
    n_images = len(images)
    isos = [400, 800, 1600, 3200, 6400, 12800]

    # Output directory for GALOSH images
    OUTDIR = BASE / f"comparison_images_v6_{ds_name}"
    OUTDIR.mkdir(exist_ok=True)

    print(f"Adding GALOSH to {ds_name} ({n_images} images x {len(isos)} ISOs)")
    print(f"GALOSH exe: {EXE_V6}")
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

    for iso_idx, iso in enumerate(isos):
        t_iso_start = time.time()
        gain = iso / 100.0
        alpha = 0.001 * gain
        sigma_sq = (0.002 * gain) ** 2

        galosh_metrics = {k: [] for k in ["psnr", "ssim", "lpips", "dists", "niqe"]}
        galosh_times = []

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

            uid = f"{ds_name}_iso{iso}_img{idx}"
            t0 = time.time()
            den_raw = run_galosh(noisy_raw, w, h, alpha, sigma_sq, uid=uid)
            elapsed = time.time() - t0
            galosh_times.append(elapsed)

            den_srgb = raw_to_srgb(den_raw)

            # Metrics
            p = psnr(gt_srgb, den_srgb)
            s = ssim_rgb(gt_srgb, den_srgb)
            lp = compute_lpips(gt_srgb, den_srgb)
            di = compute_dists(gt_srgb, den_srgb)
            nq = compute_niqe(den_srgb)

            galosh_metrics["psnr"].append(p)
            galosh_metrics["ssim"].append(s)
            galosh_metrics["lpips"].append(lp)
            galosh_metrics["dists"].append(di)
            galosh_metrics["niqe"].append(nq)

            # Save image
            iso_dir = OUTDIR / f"iso{iso}"
            iso_dir.mkdir(exist_ok=True)
            imsave(str(iso_dir / f"{img_path.stem}_GALOSH.png"),
                   (np.clip(den_srgb, 0, 1) * 255).astype(np.uint8))

            if (idx + 1) % 6 == 0:
                print(f"  ISO {iso}: {idx+1}/{n_images} done", flush=True)

        # Compute averages
        avg = {k: round(float(np.mean(v)), 4 if k != "psnr" else 2)
               for k, v in galosh_metrics.items()}

        elapsed_iso = time.time() - t_iso_start
        print(f"ISO {iso:5d} ({elapsed_iso:.0f}s): PSNR={avg['psnr']:.2f} SSIM={avg['ssim']:.4f} "
              f"LPIPS={avg['lpips']:.4f} DISTS={avg['dists']:.4f} NIQE={avg['niqe']:.4f} "
              f"(avg {np.mean(galosh_times):.3f}s/img)")

        # Merge into v5 quality data
        row = v5_quality[iso_idx]
        row["GALOSH_psnr"] = avg["psnr"]
        row["GALOSH_ssim"] = avg["ssim"]
        row["GALOSH_lpips"] = avg["lpips"]
        row["GALOSH_dists"] = avg["dists"]
        row["GALOSH_niqe"] = avg["niqe"]

    # Sequential timing for GALOSH (3 images, ISO 1600)
    print("\nSequential timing (3 images, ISO 1600)...")
    timing_iso = 1600
    gain = timing_iso / 100.0
    t_alpha = 0.001 * gain
    t_sigma_sq = (0.002 * gain) ** 2
    seq_times = []
    for idx in range(min(3, n_images)):
        rng = np.random.default_rng(42 + idx)
        img = imread(str(images[idx])).astype(np.float32) / 255.0
        h, w = img.shape[:2]; h -= h % 2; w -= w % 2
        img = img[:h, :w]
        linear = inv_srgb(img).astype(np.float32)
        clean_raw = np.zeros((h, w), dtype=np.float32)
        clean_raw[0::2, 0::2] = linear[0::2, 0::2, 0]
        clean_raw[0::2, 1::2] = linear[0::2, 1::2, 1]
        clean_raw[1::2, 0::2] = linear[1::2, 0::2, 1]
        clean_raw[1::2, 1::2] = linear[1::2, 1::2, 2]
        noisy_raw, _, _ = synth_noise(clean_raw, timing_iso, rng)

        t0 = time.perf_counter()
        run_galosh(noisy_raw, w, h, t_alpha, t_sigma_sq, uid=f"timing_{idx}")
        seq_times.append(time.perf_counter() - t0)

    avg_time = float(np.mean(seq_times))
    print(f"  GALOSH avg: {avg_time:.4f}s (runs: {', '.join(f'{t:.3f}' for t in seq_times)})")

    # Save merged results as v6
    v5_timing["GALOSH"] = avg_time
    save_data = {"quality": v5_quality, "sequential_timing": v5_timing}
    out_path = RESULTS / f"v6_{ds_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
