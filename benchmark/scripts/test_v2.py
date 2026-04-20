#!/usr/bin/env python3
"""Quick A/B test: V1 (old separate) vs V2 (multichannel) on kodim01."""
import numpy as np
import subprocess
import os
import time
from skimage.io import imread

MSYS_PATH = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
EXE_V1 = r"C:\Users\luxgrain\denoise_eval\standalone\rawdenoise.exe"
EXE_V2 = r"C:\Users\luxgrain\denoise_eval\standalone\rawdenoise_v2.exe"
SDIR = r"C:\Users\luxgrain\denoise_eval\standalone"

img = imread("datasets/kodak/kodim01.png").astype(np.float32) / 255.0
h, w = img.shape[:2]; h -= h % 2; w -= w % 2; img = img[:h, :w]
linear = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4).astype(np.float32)
cr = np.zeros((h, w), dtype=np.float32)
cr[0::2, 0::2] = linear[0::2, 0::2, 0]
cr[0::2, 1::2] = linear[0::2, 1::2, 1]
cr[1::2, 0::2] = linear[1::2, 0::2, 1]
cr[1::2, 1::2] = linear[1::2, 1::2, 2]

inp = os.path.join(SDIR, "tmp_cmp.bin")

def run(exe, tag, method, extra=[]):
    out = os.path.join(SDIR, f"tmp_{tag}.bin")
    t0 = time.time()
    env = os.environ.copy()
    env["PATH"] = MSYS_PATH
    r = subprocess.run(
        [exe, inp, out, str(w), str(h), method,
         "1.0", "1.0", "1.0", str(alpha), str(sq), "1"] + extra,
        capture_output=True, text=True, timeout=300, env=env)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"  FAILED {tag}: {r.stderr[-200:]}")
        return None, elapsed
    d = np.fromfile(out, dtype=np.float32).reshape(h, w)
    mse = float(np.mean((cr - d) ** 2))
    psnr = 10 * np.log10(1 / mse) if mse > 1e-10 else 100
    return psnr, elapsed

for iso in [400, 800, 1600, 3200, 6400]:
    rng = np.random.default_rng(42)
    gain = iso / 100
    alpha = 0.001 * gain
    sr = 0.002 * gain
    sq = sr ** 2
    rate = np.maximum(cr / max(alpha, 1e-10), 0)
    noisy = rng.poisson(rate).astype(np.float32) * alpha
    noisy += rng.standard_normal(cr.shape).astype(np.float32) * sr
    noisy = np.clip(noisy, 0, 1).astype(np.float32)
    noisy.tofile(inp)

    p_pc, t_pc = run(EXE_V2, "pc", "perchannel")
    p_v1, t_v1 = run(EXE_V1, "v1", "ours")
    p_v2_33, t_v2_33 = run(EXE_V2, "v2s33", "ours", ["33"])
    p_v2_39, t_v2_39 = run(EXE_V2, "v2s39", "ours", ["39"])
    p_v2_25, t_v2_25 = run(EXE_V2, "v2s25", "ours", ["25"])

    print(f"ISO {iso}:")
    print(f"  PC:           {p_pc:7.2f} dB  {t_pc:.2f}s")
    print(f"  V1 (old):     {p_v1:7.2f} dB  {t_v1:.2f}s")
    print(f"  V2 mc s=25:   {p_v2_25:7.2f} dB  {t_v2_25:.2f}s")
    print(f"  V2 mc s=33:   {p_v2_33:7.2f} dB  {t_v2_33:.2f}s")
    print(f"  V2 mc s=39:   {p_v2_39:7.2f} dB  {t_v2_39:.2f}s")
    d33 = p_v2_33 - p_v1 if p_v2_33 and p_v1 else float("nan")
    d39 = p_v2_39 - p_v1 if p_v2_39 and p_v1 else float("nan")
    print(f"  delta V2(33)-V1: {d33:+.3f}  V2(39)-V1: {d39:+.3f}")
    print()
