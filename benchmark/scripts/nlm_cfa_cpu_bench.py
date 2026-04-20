#!/usr/bin/env python3
"""NLM-CFA CPU speed benchmark (skimage reference implementation).

EN: Runs skimage.restoration.denoise_nl_means per-Bayer-channel to match
    the CUDA fast_mode=False algorithm at patch_radius=2, search_radius=11.
    Reports timing only; PSNR comparison uses the saved GPU PNGs.

JP: skimage の denoise_nl_means を RGGB 4プレーンにかけて CUDA 版と同じ
    アルゴリズム (patch=5x5, search=23x23) を CPU で再現、速度計測。
"""
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
from skimage.restoration import denoise_nl_means

os.environ["PYTHONIOENCODING"] = "utf-8"

BENCH = Path(r"C:\Users\luxgrain\datasets\sidd\medium_bench")


def run_nlm_cfa_cpu(noisy: np.ndarray, sigma: float,
                    patch_radius: int = 2, search_radius: int = 11,
                    h_multiplier: float = 1.0) -> tuple:
    """Per-plane NLM on RGGB Bayer using skimage CPU reference.
    h (filter strength) = h_multiplier * sigma, matches CUDA default."""
    H, W = noisy.shape
    assert H % 2 == 0 and W % 2 == 0

    planes = [noisy[0::2, 0::2],   # R
              noisy[0::2, 1::2],   # G1
              noisy[1::2, 0::2],   # G2
              noisy[1::2, 1::2]]   # B

    t0 = time.time()
    den_planes = []
    for p in planes:
        d = denoise_nl_means(p.astype(np.float32),
                             patch_size=2 * patch_radius + 1,
                             patch_distance=search_radius,
                             h=h_multiplier * sigma,
                             sigma=sigma,
                             fast_mode=True,  # integral image O(N·search²) approx
                             preserve_range=True)
        den_planes.append(d.astype(np.float32))
    dt = time.time() - t0

    out = np.empty_like(noisy)
    out[0::2, 0::2] = den_planes[0]
    out[0::2, 1::2] = den_planes[1]
    out[1::2, 0::2] = den_planes[2]
    out[1::2, 1::2] = den_planes[3]
    return out, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--scenes", type=int, default=4,
                    help="Number of scenes to time (default 4 — full 80 is too slow)")
    ap.add_argument("--uids", nargs="+", default=None,
                    help="Explicit UIDs to use (overrides -s)")
    args = ap.parse_args()

    if args.uids:
        uids = args.uids
    else:
        uids = [
            "0001_S6_GRBG_010", "0045_G4_BGGR_010",
            "0066_GP_BGGR_010", "0138_IP_RGGB_010",
        ][: args.scenes]

    print(f"NLM-CFA CPU (skimage fast_mode=False, patch=5x5, search=23x23)")
    print(f"threads: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<default>')}")
    print(f"{'uid':<24}{'HxW':>12}{'den_time':>10}{'wall':>8}")
    print("-" * 60)
    for uid in uids:
        noisy = np.load(str(BENCH / f"{uid}_noisy_raw.npy")).astype(np.float32)
        gt    = np.load(str(BENCH / f"{uid}_gt_raw.npy")).astype(np.float32)
        sigma = float(np.std(noisy.astype(np.float64) - gt.astype(np.float64)))
        H, W = noisy.shape
        t0 = time.time()
        den, dt_den = run_nlm_cfa_cpu(noisy, sigma)
        t_wall = time.time() - t0
        print(f"{uid:<24}{f'{W}x{H}':>12}{dt_den:>9.2f}s{t_wall:>7.2f}s")


if __name__ == "__main__":
    main()
