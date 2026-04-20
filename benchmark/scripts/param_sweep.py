#!/usr/bin/env python3
"""Parameter sweep for Ours method on RawNIND scene."""
import numpy as np
import rawpy
import subprocess
import os

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

def load_arw(path):
    with rawpy.imread(path) as raw:
        b = raw.raw_image_visible.copy().astype(np.float32)
        black = float(raw.black_level_per_channel[0])
        white = float(raw.white_level)
        b = np.clip((b - black) / (white - black), 0, 1)
    return b

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0

def estimate_noise(noisy, gt):
    h, w = noisy.shape
    np.random.seed(42)
    gts, vs = [], []
    for _ in range(500):
        y, x = np.random.randint(0, h - 32), np.random.randint(0, w - 32)
        gp = gt[y:y+32:2, x:x+32:2]
        np_ = noisy[y:y+32:2, x:x+32:2]
        m, v = gp.mean(), (np_ - gp).var()
        if 0.01 < m < 0.95:
            gts.append(m)
            vs.append(v)
    A = np.vstack([gts, np.ones(len(gts))]).T
    r = np.linalg.lstsq(A, vs, rcond=None)
    return max(r[0][0], 1e-6), max(r[0][1], 1e-10)

def run(method, w, h, alpha, sigma_sq, luma_s=1.0, chroma_s=1.0):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exe = os.path.join(base, "standalone", "rawdenoise.exe")
    out = os.path.join(base, f"standalone/tmp_{method}.bin")
    inp = os.path.join(base, "standalone/tmp_sweep.bin")
    subprocess.run([
        exe, inp, out,
        str(w), str(h), method, "1.0", str(luma_s), str(chroma_s),
        str(alpha), str(sigma_sq), "1"
    ], capture_output=True, timeout=120)
    return np.fromfile(out, dtype=np.float32).reshape(h, w)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load
gt = load_arw(os.path.join(base_dir, "datasets/rawnind/file_28664.arw"))
noisy = load_arw(os.path.join(base_dir, "datasets/rawnind/file_28837.arw"))
h, w = 2048, 2048
cy, cx = gt.shape[0] // 2, gt.shape[1] // 2
gt = gt[cy-1024:cy+1024, cx-1024:cx+1024]
noisy = noisy[cy-1024:cy+1024, cx-1024:cx+1024]
alpha, sigma_sq = estimate_noise(noisy, gt)
noisy.tofile(os.path.join(base_dir, "standalone/tmp_sweep.bin"))

print(f"Scene: 2pilesofplates (ISO 65535)")
print(f"alpha={alpha:.6f} sigma_sq={sigma_sq:.8f}")
print(f"Noisy PSNR: {psnr(gt, noisy):.2f}")

# Perchannel baseline
pc = run("perchannel", w, h, alpha, sigma_sq)
p_pc = psnr(gt, pc)
print(f"PerChannel PSNR: {p_pc:.2f}\n")

print(f"{'luma':>6} {'chroma':>7} {'PSNR':>8} {'delta':>8}")
print("-" * 35)

best_p, best_l, best_c = 0, 0, 0
for luma_s in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
    for chroma_s in [0.8, 1.0, 1.2, 1.5]:
        o = run("ours", w, h, alpha, sigma_sq, luma_s, chroma_s)
        p = psnr(gt, o)
        marker = " *" if p > p_pc else ""
        print(f"{luma_s:>6.1f} {chroma_s:>7.1f} {p:>8.2f} {p-p_pc:>+8.2f}{marker}")
        if p > best_p:
            best_p, best_l, best_c = p, luma_s, chroma_s

print(f"\nBest: luma={best_l} chroma={best_c} PSNR={best_p:.2f} delta={best_p-p_pc:+.2f}")
