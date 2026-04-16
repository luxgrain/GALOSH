#!/usr/bin/env python3
"""V2 (known params) vs V3 (blind estimation) full comparison.
Tests: Kodak first 8, ISO 800-12800.
Metrics: PSNR, SSIM, LPIPS
"""
import numpy as np
import subprocess
import os
import torch
from pathlib import Path
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
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

def ssim_4ch(a, b):
    vals = []
    for dy in range(2):
        for dx in range(2):
            vals.append(ssim(a[dy::2, dx::2], b[dy::2, dx::2], data_range=1.0))
    return float(np.mean(vals))

def run_exe(exe, noisy, w, h, alpha, sq, tag, sw=25):
    inp = str(BASE / "standalone" / f"tmp_cmp_{tag}.bin")
    out = str(BASE / "standalone" / f"tmp_cmp_{tag}_out.bin")
    noisy.tofile(inp)
    env = os.environ.copy(); env["PATH"] = MSYS_PATH
    result = subprocess.run(
        [exe, inp, out, str(w), str(h), "ours",
         "1.0", "1.0", "1.0", str(alpha), str(sq), "1", str(sw)],
        capture_output=True, timeout=300, env=env
    )
    return np.fromfile(out, dtype=np.float32).reshape(h, w)


def main():
    images = sorted(KODAK.glob("kodim*.png"))[:8]
    isos = [800, 1600, 3200, 6400, 12800]

    configs = [
        ("V2-known", EXE_V2, True),    # V2 with true params
        ("V3-known", EXE_V3, True),     # V3 with true params (shows Foi inv improvement)
        ("V3-blind", EXE_V3, False),    # V3 with blind estimation
    ]

    print("V2(known) vs V3(known) vs V3(blind) comparison")
    print(f"Images: {len(images)}, ISOs: {isos}")
    print()

    for iso in isos:
        gain = iso / 100.0
        true_alpha = 0.001 * gain; true_sq = (0.002 * gain) ** 2
        metrics = {c[0]: {"psnr": [], "ssim": [], "lpips": []} for c in configs}

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

            for name, exe, use_known in configs:
                a = true_alpha if use_known else -1
                s = true_sq if use_known else -1
                den_bayer = run_exe(exe, noisy, w, h, a, s, name.replace("-","_"))
                den_srgb = np.clip(linear_to_srgb(demosaic_bilinear(den_bayer)), 0, 1).astype(np.float32)
                p = psnr(cr, den_bayer)
                ss = ssim_4ch(cr, den_bayer)
                lp = compute_lpips(gt_srgb, den_srgb)
                metrics[name]["psnr"].append(p)
                metrics[name]["ssim"].append(ss)
                metrics[name]["lpips"].append(lp)

        print(f"ISO {iso}:")
        print(f"  {'Config':<12} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
        print(f"  {'-'*36}")
        for name, _, _ in configs:
            m = metrics[name]
            print(f"  {name:<12} {np.mean(m['psnr']):>7.2f} {np.mean(m['ssim']):>7.4f} {np.mean(m['lpips']):>7.4f}")

        # Delta V3-blind vs V2-known
        dp = np.mean(metrics["V3-blind"]["psnr"]) - np.mean(metrics["V2-known"]["psnr"])
        dl = np.mean(metrics["V3-blind"]["lpips"]) - np.mean(metrics["V2-known"]["lpips"])
        print(f"  V3-blind vs V2-known: dPSNR={dp:+.2f} dLPIPS={dl:+.4f}")
        print()


if __name__ == "__main__":
    main()
