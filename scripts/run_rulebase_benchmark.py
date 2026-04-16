#!/usr/bin/env python3
"""
Rule-based benchmark on multiple datasets: BM3D-PC vs BM3D-CFA vs Ours(V2)
Datasets: Kodak (24), CBSD68 (68)
PSNR / SSIM / LPIPS
"""
import numpy as np
import subprocess
import os
import json
import time
import torch
from pathlib import Path
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import uniform_filter
import bm3d as bm3d_pkg

MSYS_PATH = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

BASE = Path(__file__).parent.parent
EXE = str(BASE / "standalone" / "rawdenoise.exe")
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "Kodak":  (BASE / "datasets" / "kodak",  "kodim*.png", 24),
    "CBSD68": (BASE / "datasets" / "cbsd68", "*.png", 68),
}


def inv_srgb(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(np.maximum(x, 0), 1/2.4) - 0.055)

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0

def ssim_4ch(a, b):
    vals = []
    for dy in range(2):
        for dx in range(2):
            vals.append(ssim(a[dy::2, dx::2], b[dy::2, dx::2], data_range=1.0))
    return float(np.mean(vals))

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

def synth_noise(clean_raw, iso, rng):
    gain = iso / 100.0
    alpha = 0.001 * gain
    sigma_read = 0.002 * gain
    sigma_sq = sigma_read ** 2
    rate = np.maximum(clean_raw / max(alpha, 1e-10), 0.0)
    noisy = rng.poisson(rate).astype(np.float32) * alpha
    noisy += rng.standard_normal(clean_raw.shape).astype(np.float32) * sigma_read
    return np.clip(noisy, 0, 1).astype(np.float32), alpha, sigma_sq

def gat_forward(x, alpha, sigma_sq):
    return (2.0 / alpha) * np.sqrt(np.maximum(alpha * np.maximum(x, 0) + 0.375 * alpha**2 + sigma_sq, 0))

def gat_inverse(D, alpha, sigma_sq):
    D = np.maximum(D, 1e-8)
    D_inv = 1.0 / D
    y = 0.25*D*D + 0.25*1.2247448713916*D_inv - 11.0/8.0*D_inv**2 + 5.0/8.0*1.2247448713916*D_inv**3 - 1.0/8.0
    return np.maximum(alpha * y - sigma_sq / alpha, 0)

def method_c_standalone(noisy, w, h, method_name, alpha, sigma_sq, search_window=25):
    inp = str(BASE / "standalone" / "tmp_rb.bin")
    out = str(BASE / "standalone" / f"tmp_rb_{method_name}.bin")
    noisy.tofile(inp)
    env = os.environ.copy()
    env["PATH"] = MSYS_PATH
    subprocess.run([EXE, inp, out, str(w), str(h), method_name,
                   "1.0", "1.0", "1.0", str(alpha), str(sigma_sq), "1", str(search_window)],
                  capture_output=True, timeout=300, env=env)
    return np.fromfile(out, dtype=np.float32).reshape(h, w)

def method_bm3d_cfa(noisy_bayer, alpha, sigma_sq):
    h, w = noisy_bayer.shape
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
    channels_gat = []
    for dy, dx in offsets:
        ch = noisy_bayer[dy::2, dx::2].copy()
        channels_gat.append(gat_forward(ch, alpha, sigma_sq).astype(np.float64))
    stack = np.stack(channels_gat, axis=2)
    from scipy.signal import convolve2d
    kernel = np.array([1, -2, 1], dtype=np.float64).reshape(1, 3)
    laps = convolve2d(channels_gat[0], kernel, mode='valid')
    sigma_est = np.median(np.abs(laps)) / 0.6745 / np.sqrt(6)
    sigma_est = max(sigma_est, 0.1)
    denoised_stack = bm3d_pkg.bm3d(stack, sigma_psd=sigma_est, profile='np')
    output = np.zeros_like(noisy_bayer)
    gat_max = gat_forward(1.0, alpha, sigma_sq) * 1.2
    for i, (dy, dx) in enumerate(offsets):
        ch = np.clip(denoised_stack[:, :, i], 0, gat_max)
        output[dy::2, dx::2] = np.minimum(gat_inverse(ch, alpha, sigma_sq), 1.0).astype(np.float32)
    return output


def run_dataset(ds_name, ds_path, ds_glob, ds_max, isos, methods):
    images = sorted(ds_path.glob(ds_glob))[:ds_max]
    n_images = len(images)
    if not images:
        print(f"  WARNING: No images found in {ds_path}")
        return []

    print(f"\n{'='*80}")
    print(f"Dataset: {ds_name} ({n_images} images)")
    print(f"{'='*80}")

    ds_results = []

    for iso in isos:
        t0 = time.time()
        gains = {m: {"psnr": [], "ssim": [], "lpips": []} for m in methods}

        for idx, img_path in enumerate(images):
            rng = np.random.default_rng(42)
            gain = iso / 100.0
            alpha = 0.001 * gain
            sigma_read = 0.002 * gain
            sigma_sq = sigma_read ** 2

            img = imread(str(img_path)).astype(np.float32) / 255.0
            if img.ndim == 2:
                img = np.stack([img]*3, axis=2)
            h, w = img.shape[:2]; h -= h % 2; w -= w % 2; img = img[:h, :w]
            linear = inv_srgb(img).astype(np.float32)

            cr = np.zeros((h, w), dtype=np.float32)
            cr[0::2, 0::2] = linear[0::2, 0::2, 0]
            cr[0::2, 1::2] = linear[0::2, 1::2, 1]
            cr[1::2, 0::2] = linear[1::2, 0::2, 1]
            cr[1::2, 1::2] = linear[1::2, 1::2, 2]

            noisy, _, _ = synth_noise(cr, iso, rng)
            gt_srgb = np.clip(img, 0, 1).astype(np.float32)

            results_bayer = {}
            results_bayer["BM3D-PC"] = method_c_standalone(noisy, w, h, "perchannel", alpha, sigma_sq)
            results_bayer["Ours(V2)"] = method_c_standalone(noisy, w, h, "ours", alpha, sigma_sq, search_window=25)
            try:
                results_bayer["BM3D-CFA"] = method_bm3d_cfa(noisy, alpha, sigma_sq)
            except Exception as e:
                print(f"  BM3D-CFA failed on {img_path.name}: {e}")
                results_bayer["BM3D-CFA"] = noisy

            for m_name, den_bayer in results_bayer.items():
                p = psnr(cr, den_bayer)
                s = ssim_4ch(cr, den_bayer)
                den_srgb = np.clip(linear_to_srgb(demosaic_bilinear(den_bayer)), 0, 1).astype(np.float32)
                lp = compute_lpips(gt_srgb, den_srgb)
                gains[m_name]["psnr"].append(float(p))
                gains[m_name]["ssim"].append(float(s))
                gains[m_name]["lpips"].append(float(lp))

            if (idx + 1) % 10 == 0:
                print(f"  ISO {iso}: {idx+1}/{n_images}")

        elapsed = time.time() - t0
        avg = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in gains.items()}

        result = {"dataset": ds_name, "iso": iso, "n_images": n_images}
        for m in methods:
            key = m.replace(" ", "_").replace("(", "").replace(")", "")
            result[f"{key}_psnr"] = round(avg[m]["psnr"], 2)
            result[f"{key}_ssim"] = round(avg[m]["ssim"], 4)
            result[f"{key}_lpips"] = round(avg[m]["lpips"], 4)
        ds_results.append(result)

        print(f"\n  ISO {iso:>5} ({elapsed:.0f}s):")
        print(f"    {'Method':<12} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
        print(f"    {'-'*36}")
        for m in methods:
            a = avg[m]
            print(f"    {m:<12} {a['psnr']:>7.2f} {a['ssim']:>7.4f} {a['lpips']:>7.4f}")
        dp = avg["Ours(V2)"]["psnr"] - avg["BM3D-CFA"]["psnr"]
        dl = avg["Ours(V2)"]["lpips"] - avg["BM3D-CFA"]["lpips"]
        mark = "OK" if dl < 0 else "NG"
        print(f"    Ours vs CFA: dPSNR={dp:+.2f}  dLPIPS={dl:+.4f} {mark}")

    return ds_results


def main():
    isos = [400, 800, 1600, 3200, 6400, 12800]
    methods = ["BM3D-PC", "BM3D-CFA", "Ours(V2)"]

    print(f"Rule-based benchmark: {', '.join(methods)}")
    print(f"Datasets: {', '.join(DATASETS.keys())}")
    print(f"ISOs: {isos}")
    print()

    all_results = []

    for ds_name, (ds_path, ds_glob, ds_max) in DATASETS.items():
        ds_results = run_dataset(ds_name, ds_path, ds_glob, ds_max, isos, methods)
        all_results.extend(ds_results)

    # Combined summary
    print()
    print("=" * 90)
    print("COMBINED SUMMARY (Rule-based methods)")
    print(f"{'Dataset':<8} {'ISO':>6} {'BM3D-PC':>16} {'BM3D-CFA':>16} {'Ours(V2)':>16}  {'dLPIPS':>8}")
    print("-" * 90)
    for r in all_results:
        pc_p = r["BM3D-PC_psnr"]; pc_l = r["BM3D-PC_lpips"]
        cfa_p = r["BM3D-CFA_psnr"]; cfa_l = r["BM3D-CFA_lpips"]
        ours_p = r["OursV2_psnr"]; ours_l = r["OursV2_lpips"]
        dl = ours_l - cfa_l
        print(f"{r['dataset']:<8} {r['iso']:>6}  {pc_p:>6.2f}/{pc_l:.3f}  {cfa_p:>6.2f}/{cfa_l:.3f}  {ours_p:>6.2f}/{ours_l:.3f}  {dl:>+.4f}")

    # Cross-dataset average
    print()
    print("CROSS-DATASET AVERAGE:")
    for iso in isos:
        iso_rows = [r for r in all_results if r["iso"] == iso]
        if not iso_rows:
            continue
        # Weighted average by n_images
        total_n = sum(r["n_images"] for r in iso_rows)
        avg = {}
        for key in ["BM3D-PC_psnr", "BM3D-PC_lpips", "BM3D-CFA_psnr", "BM3D-CFA_lpips", "OursV2_psnr", "OursV2_lpips"]:
            avg[key] = sum(r[key] * r["n_images"] for r in iso_rows) / total_n
        dl = avg["OursV2_lpips"] - avg["BM3D-CFA_lpips"]
        dp = avg["OursV2_psnr"] - avg["BM3D-CFA_psnr"]
        print(f"  ISO {iso:>5}: PC={avg['BM3D-PC_psnr']:.2f}/{avg['BM3D-PC_lpips']:.3f}  "
              f"CFA={avg['BM3D-CFA_psnr']:.2f}/{avg['BM3D-CFA_lpips']:.3f}  "
              f"Ours={avg['OursV2_psnr']:.2f}/{avg['OursV2_lpips']:.3f}  "
              f"dPSNR={dp:+.2f} dLPIPS={dl:+.4f}")

    out_path = RESULTS / "rulebase_multi_dataset.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
