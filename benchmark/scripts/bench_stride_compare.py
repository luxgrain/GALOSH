#!/usr/bin/env python3
"""
Benchmark: stride=2 (GALOSH_F) vs stride=4 (GALOSH_FULLRES_L baseline)

Purpose: Quantify quality delta between GPU-fast target (stride=4)
and CPU-quality reference (stride=2) to validate mobile/ISP story.

Usage:
  python bench_stride_compare.py -d kodak
  python bench_stride_compare.py -d kodak -v S4    # stride=4 only
  python bench_stride_compare.py -d kodak -v S2    # stride=2 only
  python bench_stride_compare.py -d kodak -v all   # both + comparison
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
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

VARIANTS = {
    "S2": {"exe": BASE / "standalone" / "rawdenoise_v8_f.exe",
           "label": "GALOSH_F (stride=2, CPU quality)", "gpu": False},
    "S4": {"exe": BASE / "standalone" / "rawdenoise_v9_s4.exe",
           "label": "GALOSH_FULLRES_L (stride=4, CPU fast)", "gpu": False},
    "GPU": {"exe": BASE / "standalone" / "galosh_raw_gpu.exe",
            "label": "GALOSH fused tile (stride=4, FP32, GPU gfx1036)", "gpu": True,
            "cl_device": 2},
}

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


def run_denoise(exe, noisy_raw, w, h, alpha, sigma_sq, ls, cs, uid="0",
                gpu=False, cl_device=0):
    inp = str(BASE / "standalone" / f"tmp_sc_{uid}.bin")
    out = str(BASE / "standalone" / f"tmp_sc_out_{uid}.bin")
    noisy_raw.tofile(inp)
    if gpu:
        cmd = [str(exe), inp, out, str(w), str(h), "galosh",
               "1.0", str(ls), str(cs), str(alpha), str(sigma_sq),
               str(cl_device)]
    else:
        cmd = [str(exe), inp, out, str(w), str(h), "galosh",
               "1.0", str(ls), str(cs), str(alpha), str(sigma_sq), "1", "25"]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    try: os.remove(inp)
    except: pass
    if result.returncode != 0:
        raise RuntimeError(f"Denoise failed ({exe}): {result.stderr.decode()}")
    den = np.fromfile(out, dtype=np.float32).reshape(h, w)
    try: os.remove(out)
    except: pass
    return den


def run_benchmark(ds_name, variant, ls, cs):
    info = VARIANTS[variant]
    exe = info["exe"]
    if not exe.exists():
        print(f"ERROR: {exe} not found, skipping {variant}")
        return None

    ds = DATASETS[ds_name]
    IMGDIR = ds["path"]
    OUTDIR = BASE / f"comparison_stride_{ds_name}"
    OUTDIR.mkdir(exist_ok=True)

    images = sorted(IMGDIR.glob(ds["glob"]))[:ds["max"]]
    n_images = len(images)
    isos = [400, 800, 1600, 3200, 6400, 12800]

    ls_tag = str(ls).replace(".", "")
    cs_tag = str(cs).replace(".", "")
    method_name = f"GALOSH_{variant}_{ls_tag}_{cs_tag}"

    print(f"\n{'='*60}")
    print(f"{method_name}: {info['label']}")
    print(f"  [{ds_name}] {n_images} images x {len(isos)} ISOs")
    print(f"  exe: {exe}")
    print(f"{'='*60}\n")

    all_results = []
    total_denoise_time = 0.0
    total_denoise_count = 0

    for iso in isos:
        t_iso_start = time.time()
        gain = iso / 100.0
        alpha = 0.001 * gain
        sigma_sq = (0.002 * gain) ** 2

        metrics_acc = {k: [] for k in ["psnr", "ssim", "lpips", "dists", "niqe"]}
        times_acc = []

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

            uid = f"{variant}_{ds_name}_{ls_tag}{cs_tag}_iso{iso}_img{idx}"
            t0 = time.time()
            den_raw = run_denoise(exe, noisy_raw, w, h, alpha, sigma_sq, ls, cs, uid=uid,
                                  gpu=info.get("gpu", False),
                                  cl_device=info.get("cl_device", 0))
            dt = time.time() - t0
            times_acc.append(dt)
            total_denoise_time += dt
            total_denoise_count += 1

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

            # Save images (all as PNG)
            iso_dir = OUTDIR / f"iso{iso}"
            iso_dir.mkdir(exist_ok=True)
            imsave(str(iso_dir / f"{img_path.stem}_{method_name}.png"),
                   (np.clip(den_srgb, 0, 1) * 255).astype(np.uint8))

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
        avg_time = round(float(np.mean(times_acc)), 3)

        elapsed = time.time() - t_iso_start
        print(f"ISO {iso:5d} ({elapsed:.0f}s): PSNR={avg['psnr']:.2f} SSIM={avg['ssim']:.4f} "
              f"LPIPS={avg['lpips']:.4f} DISTS={avg['dists']:.4f} NIQE={avg['niqe']:.4f} "
              f"t={avg_time:.3f}s/img")

        all_results.append({
            "iso": iso,
            "variant": variant,
            "label": info["label"],
            "time_per_img": avg_time,
            **avg
        })

    out_path = RESULTS / f"stride_compare_{variant.lower()}_{ls_tag}_{cs_tag}_{ds_name}.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    if total_denoise_count > 0:
        print(f"Average denoise time: {total_denoise_time/total_denoise_count:.3f}s/img")
    return all_results


def print_comparison(results_s2, results_s4):
    """Print side-by-side comparison table"""
    print(f"\n{'='*80}")
    print("STRIDE COMPARISON: stride=2 (quality) vs stride=4 (fast/mobile)")
    print(f"{'='*80}")
    print(f"{'ISO':>6} | {'PSNR s2':>8} {'s4':>8} {'diff':>7} | "
          f"{'LPIPS s2':>8} {'s4':>8} {'diff':>7} | "
          f"{'DISTS s2':>8} {'s4':>8} {'diff':>7} | "
          f"{'t_s2':>6} {'t_s4':>6} {'speedup':>7}")
    print("-" * 80)

    for r2, r4 in zip(results_s2, results_s4):
        iso = r2["iso"]
        dp = r4["psnr"] - r2["psnr"]
        dl = r4["lpips"] - r2["lpips"]
        dd = r4["dists"] - r2["dists"]
        sp = r2["time_per_img"] / max(r4["time_per_img"], 0.001)
        print(f"{iso:>6} | {r2['psnr']:>8.2f} {r4['psnr']:>8.2f} {dp:>+7.2f} | "
              f"{r2['lpips']:>8.4f} {r4['lpips']:>8.4f} {dl:>+7.4f} | "
              f"{r2['dists']:>8.4f} {r4['dists']:>8.4f} {dd:>+7.4f} | "
              f"{r2['time_per_img']:>6.3f} {r4['time_per_img']:>6.3f} {sp:>6.1f}x")

    # Averages
    avg_dp = np.mean([r4["psnr"] - r2["psnr"] for r2, r4 in zip(results_s2, results_s4)])
    avg_dl = np.mean([r4["lpips"] - r2["lpips"] for r2, r4 in zip(results_s2, results_s4)])
    avg_dd = np.mean([r4["dists"] - r2["dists"] for r2, r4 in zip(results_s2, results_s4)])
    avg_sp = np.mean([r2["time_per_img"] / max(r4["time_per_img"], 0.001)
                       for r2, r4 in zip(results_s2, results_s4)])
    print("-" * 80)
    print(f"{'AVG':>6} | {'':>8} {'':>8} {avg_dp:>+7.2f} | "
          f"{'':>8} {'':>8} {avg_dl:>+7.4f} | "
          f"{'':>8} {'':>8} {avg_dd:>+7.4f} | "
          f"{'':>6} {'':>6} {avg_sp:>6.1f}x")
    print(f"\n  + means stride=4 is WORSE, - means stride=4 is BETTER")
    print(f"  For LPIPS/DISTS: lower is better (+ = quality loss)")
    print(f"  For PSNR: higher is better (- = quality loss)")


def main():
    parser = argparse.ArgumentParser(
        description="Stride comparison: stride=2 (quality) vs stride=4 (fast/mobile)")
    parser.add_argument("--dataset", "-d", default="kodak",
                        choices=list(DATASETS.keys()))
    parser.add_argument("--variant", "-v", default="all",
                        choices=["S2", "S4", "GPU", "all"])
    parser.add_argument("--params", "-p", default="10_10",
                        choices=["10_10"])
    args = parser.parse_args()

    ls, cs = 1.0, 1.0

    # Preload metrics
    print("Loading metrics...", end=" ", flush=True)
    compute_lpips(np.zeros((64,64,3), dtype=np.float32),
                  np.zeros((64,64,3), dtype=np.float32))
    compute_dists(np.zeros((64,64,3), dtype=np.float32),
                  np.zeros((64,64,3), dtype=np.float32))
    try: compute_niqe(np.ones((64,64,3), dtype=np.float32) * 0.5)
    except: pass
    print("OK")

    variants = ["S2", "GPU"] if args.variant == "all" else [args.variant]

    all_results = {}
    for v in variants:
        r = run_benchmark(args.dataset, v, ls, cs)
        if r:
            all_results[v] = r

    # Print comparison if both ran
    if "S2" in all_results and "S4" in all_results:
        print_comparison(all_results["S2"], all_results["S4"])


if __name__ == "__main__":
    main()
