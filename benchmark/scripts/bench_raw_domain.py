#!/usr/bin/env python3
"""SIDD Medium - RAW-domain comparison (pre-demosaic).

EN: Compares three Bayer-domain denoisers directly in the RAW plane,
    bypassing demosaic+ISP: PSNR(den_raw, gt_raw) per scene, then mean.
    Same dataset as bench_sidd_medium.py (80 images pre-cropped RGGB).

JP: 80 枚の SIDD Medium フル画像を対象に、現像を挟まずに
    RAW プレーンのまま PSNR(denoised_raw, gt_raw) を算出する。
    bench_sidd_medium.py と同じデノイザ関数を直接呼ぶ。

Methods: galosh_raw_gpu (guided C), nlm_cfa_oracle, bm3d_cfa.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
os.environ["PYTHONIOENCODING"] = "utf-8"

SCRIPTS = Path(__file__).parent
BASE    = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "methods"))

from bench_sidd_medium import (
    run_galosh_gpu, run_nlm_cfa_cuda, run_bm3d_cfa_wrapper, BENCH_DIR,
)

RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


def psnr_raw(den, gt):
    mse = float(np.mean((den.astype(np.float64) - gt.astype(np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--scenes", type=int, default=0, help="Limit to first N scenes (0 = all)")
    ap.add_argument("-m", "--methods", nargs="+",
                    default=["galosh_raw_gpu", "nlm_cfa_oracle", "bm3d_cfa"])
    ap.add_argument("--out", type=str, default=str(RESULTS / "sidd_medium_raw_domain.json"))
    args = ap.parse_args()

    pairs = sorted(BENCH_DIR.glob("*_noisy_raw.npy"))
    if args.scenes:
        # scenes = image pairs (010/011 count as 2 images / scene_image)
        pairs = pairs[: args.scenes]
    print(f"N={len(pairs)} images, methods={args.methods}")

    results = {m: {"per_image": {}, "psnr_mean": None, "time_mean": None} for m in args.methods}

    for i, noisy_path in enumerate(pairs):
        uid = noisy_path.stem.replace("_noisy_raw", "")
        gt_path = noisy_path.with_name(noisy_path.name.replace("_noisy_raw", "_gt_raw"))
        noisy = np.load(noisy_path).astype(np.float32)
        gt    = np.load(gt_path).astype(np.float32)
        H, W  = noisy.shape
        psnr_noisy = psnr_raw(noisy, gt)

        print(f"\n[{i+1}/{len(pairs)}] {uid}  {H}x{W}  noisy PSNR={psnr_noisy:.3f} dB")

        for mid in args.methods:
            t0 = time.time()
            if mid == "galosh_raw_gpu":
                den, dt = run_galosh_gpu(noisy, W, H, uid)
            elif mid == "nlm_cfa_oracle":
                sigma = float(np.std(noisy.astype(np.float64) - gt.astype(np.float64)))
                den, dt = run_nlm_cfa_cuda(noisy, sigma)
            elif mid == "bm3d_cfa":
                try:
                    den, dt = run_bm3d_cfa_wrapper(noisy)
                except Exception as e:
                    print(f"    bm3d_cfa FAILED: {type(e).__name__}: {e}")
                    den, dt = None, 0.0
            elif mid == "bm3d_cfa_oracle":
                # Oracle sigma from (noisy - gt) std — matches nlm_cfa_oracle.
                try:
                    from bm3d_cfa import run_bm3d_cfa
                    sigma = float(np.std(noisy.astype(np.float64) - gt.astype(np.float64)))
                    den, dt = run_bm3d_cfa(noisy, sigma=sigma)
                except Exception as e:
                    print(f"    bm3d_cfa_oracle FAILED: {type(e).__name__}: {e}")
                    den, dt = None, 0.0
            elif mid == "b2u":
                # Blind2Unblind pretrained rawRGB_112rf20_beta19.4 + tiling.
                try:
                    from b2u import run_b2u
                    den, dt = run_b2u(noisy)
                except Exception as e:
                    print(f"    b2u FAILED: {type(e).__name__}: {e}")
                    den, dt = None, 0.0
            else:
                print(f"    unknown method: {mid}"); continue

            total_dt = time.time() - t0
            if den is None:
                print(f"    {mid:<16} FAILED")
                continue
            p = psnr_raw(den, gt)
            results[mid]["per_image"][uid] = {"psnr": p, "time": float(dt), "wall": float(total_dt)}
            print(f"    {mid:<16} PSNR={p:.3f} dB  den_time={dt:.2f}s wall={total_dt:.2f}s")

            # Always save RAW denoise as .npy (cheap) for downstream sRGB metric
            # reconstruction without re-running the slow CPU methods.  Tiny
            # compared to GPU bench but crucial for BM3D/NLM oracle pipelines
            # where re-run cost is 45-50 s/image.  User requested: always
            # persist denoised artefacts for visual / metric reuse.
            try:
                raw_dir = BASE / "sidd_medium" / f"{mid}_raw_npy"
                raw_dir.mkdir(parents=True, exist_ok=True)
                np.save(str(raw_dir / f"{uid}_den_raw.npy"), den.astype(np.float32))
            except Exception as e:
                print(f"    [warn] could not save {mid} {uid} .npy: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY - RAW-domain PSNR (denoised_raw vs gt_raw)")
    print("=" * 60)
    print(f"{'method':<20}{'PSNR_mean':>10}{'time_mean':>12}{'N':>6}")
    print("-" * 60)
    for mid in args.methods:
        entries = list(results[mid]["per_image"].values())
        if not entries:
            print(f"{mid:<20}      -          -     0")
            continue
        pm = float(np.mean([e["psnr"] for e in entries]))
        tm = float(np.mean([e["time"] for e in entries]))
        results[mid]["psnr_mean"] = pm
        results[mid]["time_mean"] = tm
        print(f"{mid:<20}{pm:>10.3f}{tm:>12.2f}s{len(entries):>6}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
