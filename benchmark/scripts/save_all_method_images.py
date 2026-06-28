#!/usr/bin/env python3
"""
Generate and save comparison images for ALL methods.
Saves GT, Noisy, BM3D-CFA, BM3D-CFA(s=0.5), RAW L/C BM3D, DnCNN-B, DRUNet
into comparison_images_v5_{dataset}/iso{N}/{stem}_{method}.png
Skips images that already exist.
"""
import numpy as np
import subprocess
import os
import sys
import time
import torch
import argparse
from pathlib import Path
from skimage.io import imread, imsave
from scipy.ndimage import uniform_filter

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

BASE = Path(__file__).parent.parent
EXE = str(BASE / "standalone" / "rawdenoise_v4.exe")
KAIR_DIR = Path(os.path.expanduser(r"~\KAIR"))

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

def save_png(arr, path):
    imsave(str(path), (np.clip(arr, 0, 1) * 255).astype(np.uint8))

def run_c_exe(noisy, w, h, method, alpha, sigma_sq, strength=1.0, ls=1.0, cs=1.0, uid="0"):
    inp = str(BASE / "standalone" / f"tmp_img_{uid}_{method}.bin")
    out = str(BASE / "standalone" / f"tmp_imgout_{uid}_{method}.bin")
    noisy.tofile(inp)
    cmd = [EXE, inp, out, str(w), str(h), method,
           str(strength), str(ls), str(cs), str(alpha), str(sigma_sq), "1", "25"]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    try: os.remove(inp)
    except: pass
    if result.returncode != 0:
        raise RuntimeError(f"C exe failed: {result.stderr.decode()}")
    den = np.fromfile(out, dtype=np.float32).reshape(h, w)
    try: os.remove(out)
    except: pass
    return den

# DnCNN-B
_dncnn = None
def get_dncnn():
    global _dncnn
    if _dncnn is not None: return _dncnn
    sys.path.insert(0, str(KAIR_DIR))
    from models.network_dncnn import DnCNN
    model = DnCNN(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='R')
    state = torch.load(str(KAIR_DIR / "model_zoo" / "dncnn_gray_blind.pth"),
                       map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _dncnn = model
    return model

def method_dncnn_b(noisy_srgb):
    model = get_dncnn()
    out_channels = []
    for c in range(3):
        ch = noisy_srgb[:, :, c]
        inp = torch.from_numpy(ch).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            res = model(inp)
        out_channels.append(res.squeeze().clamp(0, 1).numpy())
    return np.stack(out_channels, axis=2).astype(np.float32)

# DRUNet
_drunet = None
def get_drunet():
    global _drunet
    if _drunet is not None: return _drunet
    sys.path.insert(0, str(KAIR_DIR))
    from models.network_unet import UNetRes
    model = UNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512],
                    nb=4, act_mode='R', downsample_mode='strideconv',
                    upsample_mode='convtranspose', bias=False)
    state = torch.load(str(KAIR_DIR / "model_zoo" / "drunet_gray.pth"),
                       map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _drunet = model
    return model

def method_drunet(noisy_srgb, sigma_est):
    model = get_drunet()
    h, w = noisy_srgb.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    out_channels = []
    for c in range(3):
        ch = noisy_srgb[:, :, c]
        if pad_h or pad_w:
            ch = np.pad(ch, ((0, pad_h), (0, pad_w)), mode='reflect')
        sigma_map = np.full_like(ch, sigma_est)
        inp = np.stack([ch, sigma_map], axis=0)
        inp_t = torch.from_numpy(inp).unsqueeze(0).float()
        with torch.no_grad():
            res = model(inp_t)
        out_ch = res.squeeze().clamp(0, 1).numpy()
        if pad_h or pad_w:
            out_ch = out_ch[:h, :w]
        out_channels.append(out_ch)
    return np.stack(out_channels, axis=2).astype(np.float32)

def estimate_srgb_sigma(noisy_srgb):
    gray = np.mean(noisy_srgb, axis=2)
    from scipy.signal import convolve2d
    kernel = np.array([1, -2, 1], dtype=np.float64).reshape(1, 3)
    lap = convolve2d(gray.astype(np.float64), kernel, mode='valid')
    sigma = np.median(np.abs(lap)) / 0.6745 / np.sqrt(6)
    return max(float(sigma), 0.001)


def main():
    parser = argparse.ArgumentParser(description="Save all method comparison images")
    parser.add_argument("--dataset", "-d", required=True, choices=list(DATASETS.keys()))
    args = parser.parse_args()

    ds_name = args.dataset
    ds = DATASETS[ds_name]
    IMGDIR = ds["path"]
    OUTDIR = BASE / f"comparison_images_v5_{ds_name}"
    OUTDIR.mkdir(exist_ok=True)

    images = sorted(IMGDIR.glob(ds["glob"]))[:ds["max"]]
    n_images = len(images)
    isos = [400, 800, 1600, 3200, 6400, 12800]

    c_ok = os.path.isfile(EXE)
    print(f"Generating all method images for {ds_name} ({n_images} images x {len(isos)} ISOs)")
    print(f"C exe: {EXE} ({'OK' if c_ok else 'MISSING'})")
    print(f"Output: {OUTDIR}")

    # Preload models
    print("Loading DnCNN-B...", end=" ", flush=True)
    try: get_dncnn(); print("OK"); dncnn_ok = True
    except Exception as e: print(f"FAIL: {e}"); dncnn_ok = False

    print("Loading DRUNet...", end=" ", flush=True)
    try: get_drunet(); print("OK"); drunet_ok = True
    except Exception as e: print(f"FAIL: {e}"); drunet_ok = False
    print()

    for iso in isos:
        t0 = time.time()
        gain = iso / 100.0
        alpha = 0.001 * gain
        sigma_sq = (0.002 * gain) ** 2

        iso_dir = OUTDIR / f"iso{iso}"
        iso_dir.mkdir(exist_ok=True)

        for idx, img_path in enumerate(images):
            stem = img_path.stem
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
            uid = f"{ds_name}_iso{iso}_img{idx}"

            # GT and Noisy
            gt_path = iso_dir / f"{stem}_gt.png"
            if not gt_path.exists():
                save_png(gt_srgb, gt_path)
            noisy_path = iso_dir / f"{stem}_Noisy.png"
            if not noisy_path.exists():
                save_png(noisy_srgb, noisy_path)
            nd_path = iso_dir / f"{stem}_No-denoise.png"
            if not nd_path.exists():
                save_png(noisy_srgb, nd_path)

            # BM3D-CFA (s=1.0)
            p = iso_dir / f"{stem}_BM3D-CFA.png"
            if not p.exists() and c_ok:
                try:
                    den = run_c_exe(noisy_raw, w, h, "bm3dcfa", alpha, sigma_sq, strength=1.0, uid=uid)
                    save_png(raw_to_srgb(den), p)
                except Exception as e:
                    print(f"  BM3D-CFA fail {stem} ISO{iso}: {e}")

            # RAW L/C BM3D
            p = iso_dir / f"{stem}_RAW_L-C_BM3D.png"
            if not p.exists() and c_ok:
                try:
                    den = run_c_exe(noisy_raw, w, h, "ours", alpha, sigma_sq, ls=0.5, cs=1.0, uid=uid)
                    save_png(raw_to_srgb(den), p)
                except Exception as e:
                    print(f"  RAW L/C BM3D fail {stem} ISO{iso}: {e}")

            # DnCNN-B
            p = iso_dir / f"{stem}_DnCNN-B.png"
            if not p.exists() and dncnn_ok:
                try:
                    save_png(method_dncnn_b(noisy_srgb), p)
                except Exception as e:
                    print(f"  DnCNN-B fail {stem} ISO{iso}: {e}")

            # DRUNet
            p = iso_dir / f"{stem}_DRUNet.png"
            if not p.exists() and drunet_ok:
                try:
                    sigma_est = estimate_srgb_sigma(noisy_srgb)
                    save_png(method_drunet(noisy_srgb, sigma_est), p)
                except Exception as e:
                    print(f"  DRUNet fail {stem} ISO{iso}: {e}")

            if (idx + 1) % 6 == 0:
                print(f"  ISO {iso}: {idx+1}/{n_images}", flush=True)

        elapsed = time.time() - t0
        print(f"ISO {iso} done ({elapsed:.0f}s)")

    print(f"\nAll images saved to {OUTDIR}")


if __name__ == "__main__":
    main()
