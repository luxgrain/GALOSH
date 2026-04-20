#!/usr/bin/env python3
"""Per-image timing comparison: all methods on kodim01 at ISO 1600."""
import numpy as np
import subprocess
import os
import sys
import time
import torch
from pathlib import Path
from skimage.io import imread

MSYS_PATH = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
BASE = Path(__file__).parent.parent
EXE = str(BASE / "standalone" / "rawdenoise.exe")
KAIR_DIR = Path(r"C:\Users\luxgrain\KAIR")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inv_srgb(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def gat_forward(x, alpha, sigma_sq):
    return (2.0 / alpha) * np.sqrt(np.maximum(alpha * np.maximum(x, 0) + 0.375 * alpha**2 + sigma_sq, 0))

def gat_inverse(D, alpha, sigma_sq):
    D = np.maximum(D, 1e-8)
    D_inv = 1.0 / D
    y = 0.25*D*D + 0.25*1.2247448713916*D_inv - 11.0/8.0*D_inv**2 + 5.0/8.0*1.2247448713916*D_inv**3 - 1.0/8.0
    return np.maximum(alpha * y - sigma_sq / alpha, 0)

# Load image
img = imread(str(BASE / "datasets" / "kodak" / "kodim01.png")).astype(np.float32) / 255.0
h, w = img.shape[:2]; h -= h % 2; w -= w % 2; img = img[:h, :w]
linear = inv_srgb(img).astype(np.float32)
cr = np.zeros((h, w), dtype=np.float32)
cr[0::2, 0::2] = linear[0::2, 0::2, 0]
cr[0::2, 1::2] = linear[0::2, 1::2, 1]
cr[1::2, 0::2] = linear[1::2, 0::2, 1]
cr[1::2, 1::2] = linear[1::2, 1::2, 2]

iso = 1600
gain = iso / 100.0
alpha = 0.001 * gain
sigma_read = 0.002 * gain
sigma_sq = sigma_read ** 2

rng = np.random.default_rng(42)
rate = np.maximum(cr / max(alpha, 1e-10), 0)
noisy = rng.poisson(rate).astype(np.float32) * alpha
noisy += rng.standard_normal(cr.shape).astype(np.float32) * sigma_read
noisy = np.clip(noisy, 0, 1).astype(np.float32)

print(f"Image: kodim01 ({w}x{h}), ISO {iso}, Device: {DEVICE}")
print(f"{'Method':<20} {'Time (s)':>10} {'PSNR (dB)':>10} {'Params':>12}")
print("-" * 55)

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0

def run_c(method, sw=25):
    inp = str(BASE / "standalone" / "tmp_t.bin")
    out = str(BASE / "standalone" / f"tmp_t_{method}.bin")
    noisy.tofile(inp)
    env = os.environ.copy()
    env["PATH"] = MSYS_PATH
    t0 = time.perf_counter()
    subprocess.run([EXE, inp, out, str(w), str(h), method,
                   "1.0", "1.0", "1.0", str(alpha), str(sigma_sq), "1", str(sw)],
                  capture_output=True, timeout=300, env=env)
    elapsed = time.perf_counter() - t0
    d = np.fromfile(out, dtype=np.float32).reshape(h, w)
    return elapsed, psnr(cr, d)

# BM3D-PC
t, p = run_c("perchannel")
print(f"{'BM3D-PC (C)':<20} {t:>10.3f} {p:>10.2f} {'N/A':>12}")

# Ours(V2)
t, p = run_c("ours", 25)
print(f"{'Ours(V2) s=25 (C)':<20} {t:>10.3f} {p:>10.2f} {'N/A':>12}")

# BM3D-CFA (Python)
import bm3d as bm3d_pkg
from scipy.signal import convolve2d
t0 = time.perf_counter()
offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
channels_gat = []
for dy, dx in offsets:
    ch = noisy[dy::2, dx::2].copy()
    channels_gat.append(gat_forward(ch, alpha, sigma_sq).astype(np.float64))
stack = np.stack(channels_gat, axis=2)
kernel = np.array([1, -2, 1], dtype=np.float64).reshape(1, 3)
laps = convolve2d(channels_gat[0], kernel, mode='valid')
sigma_est = np.median(np.abs(laps)) / 0.6745 / np.sqrt(6)
sigma_est = max(sigma_est, 0.1)
denoised_stack = bm3d_pkg.bm3d(stack, sigma_psd=sigma_est, profile='np')
output_cfa = np.zeros_like(noisy)
gat_max = gat_forward(1.0, alpha, sigma_sq) * 1.2
for i, (dy, dx) in enumerate(offsets):
    ch = np.clip(denoised_stack[:, :, i], 0, gat_max)
    output_cfa[dy::2, dx::2] = np.minimum(gat_inverse(ch, alpha, sigma_sq), 1.0).astype(np.float32)
t_cfa = time.perf_counter() - t0
p_cfa = psnr(cr, output_cfa)
print(f"{'BM3D-CFA (Python)':<20} {t_cfa:>10.3f} {p_cfa:>10.2f} {'N/A':>12}")

# DL models - load
sys.path.insert(0, str(KAIR_DIR))
from models.network_dncnn import DnCNN
from models.network_unet import UNetRes

def _pad8(x):
    h, w = x.shape[-2:]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode='reflect')
    return x, h, w

def dl_denoise(noisy_bayer, model, model_name, alpha, sigma_sq):
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
    output = np.zeros_like(noisy_bayer)
    gat_max = gat_forward(1.0, alpha, sigma_sq) * 1.2
    gat_sigma = 1.0

    for dy, dx in offsets:
        ch = noisy_bayer[dy::2, dx::2].copy()
        ch_gat = gat_forward(ch, alpha, sigma_sq).astype(np.float32)
        ch_norm = np.clip(ch_gat / gat_max, 0, 1)
        sigma_norm = gat_sigma / gat_max

        with torch.no_grad():
            t_img = torch.from_numpy(ch_norm[np.newaxis, np.newaxis]).float().to(DEVICE)
            if model_name == "drunet":
                noise_map = torch.full_like(t_img, sigma_norm)
                t_in = torch.cat([t_img, noise_map], dim=1)
            else:
                t_in = t_img
            t_in_pad, oh, ow = _pad8(t_in)
            t_out = model(t_in_pad)
            t_out = t_out[:, :, :oh, :ow]
            den_norm = t_out.squeeze().cpu().numpy()

        den_gat = np.clip(den_norm * gat_max, 0, gat_max)
        den = np.minimum(gat_inverse(den_gat, alpha, sigma_sq), 1.0).astype(np.float32)
        output[dy::2, dx::2] = den
    return output

# DnCNN-B
dncnn = DnCNN(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='R')
dncnn.load_state_dict(torch.load(str(KAIR_DIR / "model_zoo" / "dncnn_gray_blind.pth"), map_location='cpu'))
dncnn.eval().to(DEVICE)
n_dncnn = sum(p.numel() for p in dncnn.parameters())

# warmup
_ = dl_denoise(noisy, dncnn, "dncnn", alpha, sigma_sq)

t0 = time.perf_counter()
d = dl_denoise(noisy, dncnn, "dncnn", alpha, sigma_sq)
t_dncnn = time.perf_counter() - t0
p_dncnn = psnr(cr, d)
print(f"{'DnCNN-B (CPU)':<20} {t_dncnn:>10.3f} {p_dncnn:>10.2f} {n_dncnn:>12,}")

# DRUNet
drunet = UNetRes(in_nc=2, out_nc=1, nc=[64,128,256,512], nb=4, act_mode='R', bias=False)
drunet.load_state_dict(torch.load(str(KAIR_DIR / "model_zoo" / "drunet_gray.pth"), map_location='cpu'))
drunet.eval().to(DEVICE)
n_drunet = sum(p.numel() for p in drunet.parameters())

# warmup
_ = dl_denoise(noisy, drunet, "drunet", alpha, sigma_sq)

t0 = time.perf_counter()
d = dl_denoise(noisy, drunet, "drunet", alpha, sigma_sq)
t_drunet = time.perf_counter() - t0
p_drunet = psnr(cr, d)
print(f"{'DRUNet (CPU)':<20} {t_drunet:>10.3f} {p_drunet:>10.2f} {n_drunet:>12,}")

print()
print("Note: C methods include process spawn overhead (~0.1s)")
print("Note: DL methods run on CPU (no CUDA available)")
