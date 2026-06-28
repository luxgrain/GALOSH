#!/usr/bin/env python3
"""
Full benchmark on Kodak 24.
Methods:
  1. BM3D per-channel (C standalone)
  2. BM3D-CFA (Python bm3d, 4ch with luma-based matching)
  3. CBM3D (Python bm3d_rgb, demosaic -> opponent color space)
  4. NAFNet (deep learning, sRGB)
  5. Ours (C standalone, GAT+BM3D+L/C+cross-channel)
"""
import numpy as np
import subprocess
import os
import sys
import json
import time
import torch
from pathlib import Path
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import uniform_filter
import bm3d as bm3d_pkg

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

BASE = Path(__file__).parent.parent
EXE = str(BASE / "standalone" / "rawdenoise.exe")
KODAK = BASE / "datasets" / "kodak"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

NAFNET_DIR = Path(os.path.expanduser(r"~\NAFNet"))


def inv_srgb(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(np.maximum(x, 0), 1/2.4) - 0.055)

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0

def ssim_rgb(a, b):
    return float(ssim(a, b, data_range=1.0, channel_axis=2))

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

# LPIPS
_lpips_fn = None
def get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net='alex', verbose=False)
    return _lpips_fn

def compute_lpips(a, b):
    fn = get_lpips()
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
    tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
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


# ===================== METHOD IMPLEMENTATIONS =====================

def method_c_standalone(noisy, w, h, method_name, alpha, sigma_sq, ls=1.0, cs=1.0, search_window=25):
    """Run C standalone (ours or perchannel)."""
    inp = str(BASE / "standalone" / "tmp_bench.bin")
    out = str(BASE / "standalone" / f"tmp_{method_name}.bin")
    noisy.tofile(inp)
    cmd = [EXE, inp, out, str(w), str(h), method_name,
           "1.0", str(ls), str(cs), str(alpha), str(sigma_sq), "1", str(search_window)]
    subprocess.run(cmd, capture_output=True, timeout=300)
    return np.fromfile(out, dtype=np.float32).reshape(h, w)


def method_bm3d_cfa(noisy_bayer, alpha, sigma_sq):
    """BM3D-CFA: GAT -> stack 4 Bayer channels -> BM3D multichannel -> inverse GAT."""
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


def method_cbm3d(noisy_bayer, alpha, sigma_sq):
    """CBM3D: GAT -> demosaic -> bm3d_rgb in opponent color space -> re-mosaic."""
    h, w = noisy_bayer.shape
    noisy_gat = gat_forward(noisy_bayer, alpha, sigma_sq).astype(np.float64)
    rgb_gat = demosaic_bilinear(noisy_gat.astype(np.float32)).astype(np.float64)

    from scipy.signal import convolve2d
    kernel = np.array([1, -2, 1], dtype=np.float64).reshape(1, 3)
    laps = convolve2d(noisy_gat[0::2, 0::2], kernel, mode='valid')
    sigma_est = np.median(np.abs(laps)) / 0.6745 / np.sqrt(6)
    sigma_est = max(sigma_est, 0.1)

    denoised_rgb = bm3d_pkg.bm3d_rgb(rgb_gat, sigma_psd=sigma_est, profile='np', colorspace='opp')
    denoised_rgb = np.clip(denoised_rgb, 0, None)

    gat_max = gat_forward(1.0, alpha, sigma_sq) * 1.2
    output = np.zeros((h, w), dtype=np.float32)
    for dy, dx, c in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 2)]:
        ch = np.clip(denoised_rgb[dy::2, dx::2, c], 0, gat_max)
        output[dy::2, dx::2] = np.minimum(gat_inverse(ch, alpha, sigma_sq), 1.0).astype(np.float32)

    return output


# ===================== NAFNet =====================

_nafnet_model = None

def get_nafnet():
    global _nafnet_model
    if _nafnet_model is not None:
        return _nafnet_model

    sys.path.insert(0, str(NAFNET_DIR))
    from basicsr.models.archs.NAFNet_arch import NAFNet

    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    ckpt = torch.load(
        str(NAFNET_DIR / "experiments" / "pretrained_models" / "NAFNet-SIDD-width64.pth"),
        map_location='cpu', weights_only=True)
    if 'params' in ckpt:
        model.load_state_dict(ckpt['params'])
    elif 'params_ema' in ckpt:
        model.load_state_dict(ckpt['params_ema'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    _nafnet_model = model
    return model


def method_nafnet(noisy_srgb):
    """NAFNet denoising on sRGB image. Input/output: [H,W,3] float32 in [0,1]."""
    model = get_nafnet()
    h, w = noisy_srgb.shape[:2]

    # Pad to multiple of 64 for NAFNet
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    if pad_h or pad_w:
        noisy_srgb = np.pad(noisy_srgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    inp = torch.from_numpy(noisy_srgb.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        out = model(inp)
    result = out.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)

    # Remove padding
    if pad_h or pad_w:
        result = result[:h, :w]

    return result.astype(np.float32)


# ===================== MAIN BENCHMARK =====================

def main():
    isos = [400, 800, 1600, 3200, 6400, 12800]
    images = sorted(KODAK.glob("kodim*.png"))[:24]
    n_images = len(images)

    if not images:
        print("ERROR: No Kodak images found")
        return

    # Methods that operate on Bayer domain (output: Bayer)
    bayer_methods = ["Noisy", "① BM3D-PC", "② BM3D-CFA", "③ CBM3D", "⑥ Ours"]
    # Methods that operate on sRGB domain (output: sRGB)
    srgb_methods = ["⑤ NAFNet"]
    all_methods = bayer_methods + srgb_methods

    print(f"Full benchmark: {n_images} images x {len(isos)} ISOs x {len(all_methods)-1} methods")
    print(f"Methods: {', '.join(all_methods[1:])}")
    print()

    # Preload NAFNet
    print("Loading NAFNet model...")
    try:
        get_nafnet()
        print("NAFNet loaded OK")
        nafnet_ok = True
    except Exception as e:
        print(f"NAFNet load failed: {e}")
        nafnet_ok = False

    all_results = []

    for iso in isos:
        t0 = time.time()
        gains = {m: {"psnr": [], "ssim": [], "lpips": []} for m in all_methods}

        for idx, img_path in enumerate(images):
            rng = np.random.default_rng(42)
            gain = iso / 100.0
            alpha = 0.001 * gain
            sigma_read = 0.002 * gain
            sigma_sq = sigma_read ** 2

            img = imread(str(img_path)).astype(np.float32) / 255.0
            h, w = img.shape[:2]; h -= h % 2; w -= w % 2; img = img[:h, :w]
            linear = inv_srgb(img).astype(np.float32)

            cr = np.zeros((h, w), dtype=np.float32)
            cr[0::2, 0::2] = linear[0::2, 0::2, 0]
            cr[0::2, 1::2] = linear[0::2, 1::2, 1]
            cr[1::2, 0::2] = linear[1::2, 0::2, 1]
            cr[1::2, 1::2] = linear[1::2, 1::2, 2]

            noisy, _, _ = synth_noise(cr, iso, rng)

            # GT sRGB for LPIPS / NAFNet comparison
            gt_srgb = np.clip(img, 0, 1).astype(np.float32)

            # Noisy sRGB (shared for NAFNet input and noisy metrics)
            noisy_srgb = np.clip(linear_to_srgb(demosaic_bilinear(noisy)), 0, 1).astype(np.float32)

            # === Bayer-domain methods ===
            results_bayer = {}
            results_bayer["Noisy"] = noisy
            results_bayer["① BM3D-PC"] = method_c_standalone(noisy, w, h, "perchannel", alpha, sigma_sq)
            results_bayer["⑥ Ours"] = method_c_standalone(noisy, w, h, "ours", alpha, sigma_sq)

            try:
                results_bayer["② BM3D-CFA"] = method_bm3d_cfa(noisy, alpha, sigma_sq)
            except Exception as e:
                print(f"  BM3D-CFA failed on {img_path.name}: {e}")
                results_bayer["② BM3D-CFA"] = noisy

            try:
                results_bayer["③ CBM3D"] = method_cbm3d(noisy, alpha, sigma_sq)
            except Exception as e:
                print(f"  CBM3D failed on {img_path.name}: {e}")
                results_bayer["③ CBM3D"] = noisy

            # Compute Bayer-domain metrics (PSNR/SSIM on raw Bayer, LPIPS on demosaiced sRGB)
            for m_name, den_bayer in results_bayer.items():
                p = psnr(cr, den_bayer)
                s = ssim_4ch(cr, den_bayer)
                den_srgb = np.clip(linear_to_srgb(demosaic_bilinear(den_bayer)), 0, 1).astype(np.float32)
                lp = compute_lpips(gt_srgb, den_srgb)
                gains[m_name]["psnr"].append(p)
                gains[m_name]["ssim"].append(s)
                gains[m_name]["lpips"].append(lp)

            # === sRGB-domain methods ===
            if nafnet_ok:
                try:
                    nafnet_srgb = method_nafnet(noisy_srgb)
                    # For NAFNet: compute sRGB PSNR/SSIM (since it operates in sRGB domain)
                    p_naf = psnr(gt_srgb, nafnet_srgb)
                    s_naf = ssim_rgb(gt_srgb, nafnet_srgb)
                    lp_naf = compute_lpips(gt_srgb, nafnet_srgb)
                    gains["⑤ NAFNet"]["psnr"].append(p_naf)
                    gains["⑤ NAFNet"]["ssim"].append(s_naf)
                    gains["⑤ NAFNet"]["lpips"].append(lp_naf)
                except Exception as e:
                    print(f"  NAFNet failed on {img_path.name}: {e}")
                    gains["⑤ NAFNet"]["psnr"].append(float('nan'))
                    gains["⑤ NAFNet"]["ssim"].append(float('nan'))
                    gains["⑤ NAFNet"]["lpips"].append(float('nan'))

            if (idx + 1) % 6 == 0:
                print(f"  ISO {iso}: {idx+1}/{n_images} images done...")

        elapsed = time.time() - t0
        avg = {}
        for m, d in gains.items():
            avg[m] = {}
            for k, v in d.items():
                vals = [x for x in v if not (isinstance(x, float) and x != x)]
                avg[m][k] = float(np.mean(vals)) if vals else float('nan')

        result = {"iso": iso}
        for m in all_methods:
            key = (m.replace(" ", "_").replace("①", "1").replace("②", "2")
                    .replace("③", "3").replace("⑤", "5").replace("⑥", "6"))
            result[f"{key}_psnr"] = round(avg[m]["psnr"], 2) if not np.isnan(avg[m]["psnr"]) else None
            result[f"{key}_ssim"] = round(avg[m]["ssim"], 4) if not np.isnan(avg[m]["ssim"]) else None
            result[f"{key}_lpips"] = round(avg[m]["lpips"], 4) if not np.isnan(avg[m]["lpips"]) else None
        all_results.append(result)

        # Print table row
        print(f"\nISO {iso:>5} ({elapsed:.0f}s):")
        print(f"  {'Method':<15} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
        print(f"  {'-'*40}")
        for m in all_methods:
            a = avg[m]
            p_str = f"{a['psnr']:>7.2f}" if not np.isnan(a['psnr']) else "    N/A"
            s_str = f"{a['ssim']:>7.4f}" if not np.isnan(a['ssim']) else "    N/A"
            l_str = f"{a['lpips']:>7.4f}" if not np.isnan(a['lpips']) else "    N/A"
            print(f"  {m:<15} {p_str} {s_str} {l_str}")
        print()

    # Summary
    print("=" * 85)
    print("SUMMARY (PSNR / SSIM / LPIPS)")
    hdr = f"{'ISO':>6}"
    for m in all_methods[1:]:
        hdr += f"  {m:>16}"
    print(hdr)
    print("-" * 85)
    for r in all_results:
        line = f"{r['iso']:>6}"
        for m in all_methods[1:]:
            key = (m.replace(" ", "_").replace("①", "1").replace("②", "2")
                    .replace("③", "3").replace("⑤", "5").replace("⑥", "6"))
            p = r.get(f"{key}_psnr")
            l = r.get(f"{key}_lpips")
            if p is not None and l is not None:
                line += f"  {p:>7.2f}/{l:.3f}"
            else:
                line += f"       N/A     "
        print(line)

    # Note on metric domains
    print("\nNote: BM3D methods use raw-domain PSNR/SSIM; NAFNet uses sRGB-domain PSNR/SSIM.")
    print("LPIPS is always computed in sRGB domain (perceptual similarity).")

    out_path = RESULTS / "full_benchmark_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
