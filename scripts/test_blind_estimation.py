#!/usr/bin/env python3
"""Test blind noise estimation accuracy: estimated vs true alpha, sigma_sq.
Generates synthetic noisy Bayer with known params, runs exe with alpha=-1 sigma_sq=-1.
"""
import numpy as np
import subprocess
import os
import re
from pathlib import Path
from skimage.io import imread

MSYS_PATH = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
BASE = Path(__file__).parent.parent
EXE_V2 = str(BASE / "standalone" / "rawdenoise.exe")
EXE_V3 = str(BASE / "standalone" / "rawdenoise_v3.exe")
KODAK = BASE / "datasets" / "kodak"

def inv_srgb(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def synth_noise(clean_raw, iso, rng):
    gain = iso / 100.0
    alpha = 0.001 * gain
    sr = 0.002 * gain
    sq = sr ** 2
    rate = np.maximum(clean_raw / max(alpha, 1e-10), 0)
    noisy = rng.poisson(rate).astype(np.float32) * alpha
    noisy += rng.standard_normal(clean_raw.shape).astype(np.float32) * sr
    return np.clip(noisy, 0, 1).astype(np.float32), alpha, sq

def run_blind(exe, noisy, w, h, tag):
    """Run exe with alpha=-1, sigma_sq=-1 to trigger blind estimation. Return estimated alpha, sigma_sq."""
    inp = str(BASE / "standalone" / f"tmp_blind_{tag}.bin")
    out = str(BASE / "standalone" / f"tmp_blind_{tag}_out.bin")
    noisy.tofile(inp)
    env = os.environ.copy()
    env["PATH"] = MSYS_PATH
    # Pass -1 for alpha and sigma_sq to trigger auto-estimation
    result = subprocess.run(
        [exe, inp, out, str(w), str(h), "ours",
         "1.0", "1.0", "1.0", "-1", "-1", "1", "25"],
        capture_output=True, timeout=300, env=env
    )
    stderr = result.stderr.decode("utf-8", errors="replace")
    # Parse "Auto-estimated: alpha=X sigma_sq=Y"
    m = re.search(r"Auto-estimated:\s*alpha=([\d.eE+-]+)\s+sigma_sq=([\d.eE+-]+)", stderr)
    if m:
        return float(m.group(1)), float(m.group(2))
    else:
        print(f"  PARSE ERROR: {stderr[:300]}")
        return None, None


def main():
    images = sorted(KODAK.glob("kodim*.png"))[:8]
    isos = [400, 800, 1600, 3200, 6400, 12800]

    print("Blind Noise Estimation Accuracy Test")
    print(f"Images: {len(images)}, ISOs: {isos}")
    print()

    for exe_name, exe_path in [("V2", EXE_V2), ("V3", EXE_V3)]:
        print(f"{'='*70}")
        print(f"  {exe_name}: {exe_path}")
        print(f"{'='*70}")

        for iso in isos:
            gain = iso / 100.0
            true_alpha = 0.001 * gain
            true_sq = (0.002 * gain) ** 2

            alpha_errs = []
            sq_errs = []

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

                est_alpha, est_sq = run_blind(exe_path, noisy, w, h, exe_name.lower())
                if est_alpha is not None:
                    a_err = (est_alpha - true_alpha) / true_alpha * 100
                    s_err = (est_sq - true_sq) / true_sq * 100
                    alpha_errs.append(a_err)
                    sq_errs.append(s_err)
                    if idx == 0:
                        print(f"  ISO {iso}: true_alpha={true_alpha:.6f} true_sq={true_sq:.8f}")
                        print(f"    img1: est_alpha={est_alpha:.6f} ({a_err:+.1f}%) est_sq={est_sq:.8f} ({s_err:+.1f}%)")

            if alpha_errs:
                print(f"    Mean alpha error: {np.mean(alpha_errs):+.1f}% (std {np.std(alpha_errs):.1f}%)")
                print(f"    Mean sigma_sq error: {np.mean(sq_errs):+.1f}% (std {np.std(sq_errs):.1f}%)")
                print(f"    |alpha err| range: {np.min(np.abs(alpha_errs)):.1f}% - {np.max(np.abs(alpha_errs)):.1f}%")
                print(f"    |sq err| range: {np.min(np.abs(sq_errs)):.1f}% - {np.max(np.abs(sq_errs)):.1f}%")
            print()


if __name__ == "__main__":
    main()
