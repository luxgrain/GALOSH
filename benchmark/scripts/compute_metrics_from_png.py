#!/usr/bin/env python3
"""Compute SSIM/LPIPS/DISTS/NIQE for already-denoised PNG sets against GT2.

EN: Bench run without rerunning denoisers.  Reads method sRGB PNGs from
    sidd_medium/<method>/*_den.png, compares against sidd_medium/__gt2__/
    *_gt2.png using the same perceptual metric functions as the main bench.

JP: 既存 PNG から perceptual metric のみ再計算（デノイザは再実行しない）。
    bench と同じ perceptual 関数を再利用して一貫性を担保。
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim

os.environ["PYTHONIOENCODING"] = "utf-8"
BASE    = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

# Reuse bench's perceptual metric functions.
from bench_sidd_medium import (
    compute_lpips_patched,
    compute_dists_patched,
    compute_niqe,
)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--methods", nargs="+", required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--gt", choices=["gt2", "gt1"], default="gt2",
                    help="Reference GT: gt2 = calibrated pre-demosaic pipeline, "
                         "gt1 = camera ISP sRGB (for after-demosaic methods).")
    ap.add_argument("--no-perceptual", action="store_true",
                    help="PSNR + SSIM only (skip LPIPS/DISTS/NIQE)")
    args = ap.parse_args()

    sidd = BASE / "sidd_medium"
    if args.gt == "gt2":
        gt_dir = sidd / "__gt2__"
        gt_suffix = "_gt2.png"
    else:  # gt1 — ISP sRGB ground truth for after-demosaic methods
        gt_dir = sidd / "__gt__"
        gt_suffix = "_gt_srgb.png"
    gt_files = sorted(glob.glob(str(gt_dir / f"*{gt_suffix}")))
    print(f"N={len(gt_files)} GT files (ref={args.gt})")

    out = {}
    for m in args.methods:
        den_dir = sidd / m
        if not den_dir.exists():
            print(f"{m}: skip (dir missing)")
            continue
        print(f"\n--- {m} ---")
        per_image = {}
        for gf in gt_files:
            base = Path(gf).stem.replace("_gt2", "").replace("_gt_srgb", "")
            dp = den_dir / f"{base}_den.png"
            if not dp.exists():
                continue
            gt2 = imread(gf).astype(np.float32) / 255.0
            den = imread(str(dp)).astype(np.float32) / 255.0
            t0 = time.time()
            p  = psnr(den, gt2)
            s  = float(ssim(gt2, den, channel_axis=-1, data_range=1.0))
            rec = {"psnr": p, "ssim": s}
            if not args.no_perceptual:
                rec["lpips"] = float(compute_lpips_patched(den, gt2))
                rec["dists"] = float(compute_dists_patched(den, gt2))
                rec["niqe"]  = float(compute_niqe(den))
            per_image[base] = rec
            elapsed = time.time() - t0
            print(f"  {base}: PSNR={p:.2f} SSIM={s:.4f}" +
                  (f" LPIPS={rec['lpips']:.4f} DISTS={rec['dists']:.4f} NIQE={rec['niqe']:.2f}"
                   if not args.no_perceptual else "") +
                  f"  ({elapsed:.1f}s)")
        # Aggregate
        keys = list(per_image.values())[0].keys() if per_image else []
        agg = {k: float(np.mean([v[k] for v in per_image.values()])) for k in keys}
        print(f"  AGG: {agg}")
        out[m] = {"per_image": per_image, "aggregate": agg}

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
