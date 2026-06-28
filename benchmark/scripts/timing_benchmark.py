#!/usr/bin/env python3
"""
Dedicated timing benchmark -- sequential, no parallel contention.
3 image sizes x 5 methods (+ BM3D-CFA s=0.5 = 6 total), 1 image each.
ISO 1600 fixed.
"""
import numpy as np
import subprocess
import os
import sys
import time
import torch
from pathlib import Path
from skimage.io import imread
from scipy.ndimage import uniform_filter

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

# Fix thread count for reproducibility
torch.set_num_threads(1)  # Single-threaded torch for fair comparison
os.environ["OMP_NUM_THREADS"] = "1"  # Single-threaded OpenMP for C exe

BASE = Path(__file__).parent.parent
EXE = str(BASE / "standalone" / "rawdenoise_v4.exe")
KAIR_DIR = Path(os.path.expanduser(r"~\KAIR"))

DATASETS = {
    "CBSD68 (481x321)":  (BASE / "datasets" / "cbsd68", "101085.png"),
    "McMaster (500x500)": (BASE / "datasets" / "mcmaster", "1.png"),
    "Kodak (768x512)":   (BASE / "datasets" / "kodak", "kodim01.png"),
}

ISO = 1600
N_REPEAT = 3  # Average over 3 runs for stability

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

# ===================== Methods =====================

def run_c(noisy, w, h, method, strength, alpha, sigma_sq, ls=1.0, cs=1.0):
    inp = str(BASE / "standalone" / "tmp_timing_in.bin")
    out = str(BASE / "standalone" / "tmp_timing_out.bin")
    noisy.tofile(inp)
    cmd = [EXE, inp, out, str(w), str(h), method,
           str(strength), str(ls), str(cs), str(alpha), str(sigma_sq), "1", "25"]
    subprocess.run(cmd, capture_output=True, timeout=600)
    try: os.remove(inp)
    except: pass
    try: os.remove(out)
    except: pass

_dncnn = None
def get_dncnn():
    global _dncnn
    if _dncnn: return _dncnn
    sys.path.insert(0, str(KAIR_DIR))
    from models.network_dncnn import DnCNN
    m = DnCNN(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='R')
    s = torch.load(str(KAIR_DIR / "model_zoo" / "dncnn_gray_blind.pth"),
                   map_location='cpu', weights_only=True)
    m.load_state_dict(s, strict=True); m.eval()
    _dncnn = m; return m

_drunet = None
def get_drunet():
    global _drunet
    if _drunet: return _drunet
    sys.path.insert(0, str(KAIR_DIR))
    from models.network_unet import UNetRes
    m = UNetRes(in_nc=2, out_nc=1, nc=[64,128,256,512],
                nb=4, act_mode='R', downsample_mode='strideconv',
                upsample_mode='convtranspose', bias=False)
    s = torch.load(str(KAIR_DIR / "model_zoo" / "drunet_gray.pth"),
                   map_location='cpu', weights_only=True)
    m.load_state_dict(s, strict=True); m.eval()
    _drunet = m; return m

def run_dncnn(srgb):
    model = get_dncnn()
    for c in range(3):
        inp = torch.from_numpy(srgb[:,:,c]).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            model(inp)

def run_drunet(srgb, sigma):
    model = get_drunet()
    h, w = srgb.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    for c in range(3):
        ch = srgb[:,:,c]
        if pad_h or pad_w:
            ch = np.pad(ch, ((0,pad_h),(0,pad_w)), mode='reflect')
        sm = np.full_like(ch, sigma)
        inp = torch.from_numpy(np.stack([ch, sm], axis=0)).unsqueeze(0).float()
        with torch.no_grad():
            model(inp)

def estimate_sigma(srgb):
    gray = np.mean(srgb, axis=2)
    from scipy.signal import convolve2d
    k = np.array([1,-2,1], dtype=np.float64).reshape(1,3)
    lap = convolve2d(gray.astype(np.float64), k, mode='valid')
    return max(float(np.median(np.abs(lap)) / 0.6745 / np.sqrt(6)), 0.001)


def main():
    print("=" * 80)
    print("TIMING BENCHMARK -- Sequential, single-threaded")
    print(f"  OMP_NUM_THREADS=1, torch.num_threads=1")
    print(f"  ISO={ISO}, N_REPEAT={N_REPEAT}")
    print(f"  3 image sizes x 6 methods")
    print("=" * 80)
    print()

    # Preload models (warmup, not timed)
    print("Preloading models...", flush=True)
    get_dncnn()
    get_drunet()
    print("Done.\n")

    methods_order = [
        "BM3D-CFA (s=1.0)",
        "BM3D-CFA (s=0.5)",
        "RAW L/C BM3D",
        "DnCNN-B",
        "DRUNet",
    ]

    all_times = {}

    for ds_name, (ds_path, img_name) in DATASETS.items():
        img_path = ds_path / img_name
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

        rng = np.random.default_rng(42)
        noisy_raw, alpha, sigma_sq = synth_noise(clean_raw, ISO, rng)
        noisy_srgb = raw_to_srgb(noisy_raw)
        sigma_est = estimate_sigma(noisy_srgb)

        print(f"--- {ds_name} ({w}x{h}, {w*h:,} pixels) ---")
        times = {}

        for method in methods_order:
            elapsed_list = []
            for rep in range(N_REPEAT):
                if method == "BM3D-CFA (s=1.0)":
                    t0 = time.perf_counter()
                    run_c(noisy_raw, w, h, "bm3dcfa", 1.0, alpha, sigma_sq)
                    elapsed_list.append(time.perf_counter() - t0)
                elif method == "BM3D-CFA (s=0.5)":
                    t0 = time.perf_counter()
                    run_c(noisy_raw, w, h, "bm3dcfa", 0.5, alpha, sigma_sq)
                    elapsed_list.append(time.perf_counter() - t0)
                elif method == "RAW L/C BM3D":
                    t0 = time.perf_counter()
                    run_c(noisy_raw, w, h, "ours", 1.0, alpha, sigma_sq, ls=0.5, cs=1.0)
                    elapsed_list.append(time.perf_counter() - t0)
                elif method == "DnCNN-B":
                    t0 = time.perf_counter()
                    run_dncnn(noisy_srgb)
                    elapsed_list.append(time.perf_counter() - t0)
                elif method == "DRUNet":
                    t0 = time.perf_counter()
                    run_drunet(noisy_srgb, sigma_est)
                    elapsed_list.append(time.perf_counter() - t0)

            avg_t = np.mean(elapsed_list)
            times[method] = avg_t
            print(f"  {method:<20} {avg_t:>8.3f}s  (runs: {', '.join(f'{t:.3f}' for t in elapsed_list)})")

        all_times[ds_name] = times
        print()

    # Summary table
    print("=" * 80)
    print("SUMMARY (seconds, single-threaded, avg of 3 runs)")
    print(f"{'Method':<20}", end="")
    for ds_name in DATASETS:
        short = ds_name.split("(")[1].rstrip(")")
        print(f"  {short:>12}", end="")
    print()
    print("-" * 60)

    base_method = "RAW L/C BM3D"
    for method in methods_order:
        print(f"{method:<20}", end="")
        for ds_name in DATASETS:
            t = all_times[ds_name][method]
            print(f"  {t:>11.3f}s", end="")
        print()

    # Relative to RAW L/C BM3D
    print()
    print("Relative to RAW L/C BM3D:")
    print(f"{'Method':<20}", end="")
    for ds_name in DATASETS:
        short = ds_name.split("(")[1].rstrip(")")
        print(f"  {short:>12}", end="")
    print()
    print("-" * 60)
    for method in methods_order:
        print(f"{method:<20}", end="")
        for ds_name in DATASETS:
            t = all_times[ds_name][method]
            base = all_times[ds_name][base_method]
            print(f"  {t/base:>11.2f}x", end="")
        print()

    # Pixels/second
    print()
    print("Throughput (pixels/second):")
    print(f"{'Method':<20}", end="")
    for ds_name in DATASETS:
        short = ds_name.split("(")[1].rstrip(")")
        print(f"  {short:>12}", end="")
    print()
    print("-" * 60)
    for method in methods_order:
        print(f"{method:<20}", end="")
        for ds_name, (ds_path, img_name) in DATASETS.items():
            img = imread(str(ds_path / img_name))
            h, w = img.shape[:2]
            h -= h % 2; w -= w % 2
            npix = h * w
            t = all_times[ds_name][method]
            pps = npix / t
            if pps > 1e6:
                print(f"  {pps/1e6:>9.2f} MP/s", end="")
            else:
                print(f"  {pps/1e3:>9.1f} KP/s", end="")
        print()

    print()


if __name__ == "__main__":
    main()
