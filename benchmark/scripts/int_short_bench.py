"""INT r32 short bench — quick regression check on 100 SIDD val + 100 RawNIND patches.

Goal: fast feedback loop (~5-10 min) for r32 improvement work.
Stable seeded random subset → reproducible across iterations.

Usage:
  python benchmark/scripts/int_short_bench.py
  python benchmark/scripts/int_short_bench.py --variant r16
  python benchmark/scripts/int_short_bench.py --n_sidd 200 --n_rawnind 200
"""
import argparse
import os
import sys
import subprocess
import time
import random
from pathlib import Path
import numpy as np
from scipy.io import loadmat

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

GALOSH = Path(r"C:\Users\luxgrain\GALOSH")
EXE = GALOSH / "standalone" / "galosh_raw_cpu_int.exe"
SIDD_VAL = GALOSH / "benchmark" / "SIDD_Validation"
RAWNIND = Path(r"E:\rawnind_bench")
TMP = GALOSH / "benchmark" / "results" / "int_short_bench_tmp"
TMP.mkdir(parents=True, exist_ok=True)

# v12 canonical baseline (from project_int_port_progress memory)
BASELINE = {
    "SIDD_val_full_1280": 49.10,
    "RawNIND_full_1493": 29.37,
}

SEED = 42


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def run_int_patch(noisy, variant, uid):
    h, w = noisy.shape
    in_p = TMP / f"_{uid}_in.bin"
    out_p = TMP / f"_{uid}_out.bin"
    noisy.astype(np.float32).tofile(str(in_p))
    cmd = [str(EXE), str(in_p), str(out_p), str(w), str(h),
           "galosh", "1.0", "1.0", "1.0", "0", "0", f"--variant={variant}"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True)
        return None, time.time() - t0
    dt = time.time() - t0
    in_p.unlink(missing_ok=True)
    if r.returncode != 0 or not out_p.exists():
        out_p.unlink(missing_ok=True)
        return None, dt
    out = np.fromfile(str(out_p), dtype=np.float32).reshape(h, w)
    out_p.unlink(missing_ok=True)
    return out, dt


def bench_sidd(n, variant, rng):
    print(f"\n=== SIDD val short bench (n={n}, variant={variant}) ===")
    n_raw = loadmat(str(SIDD_VAL / "ValidationNoisyBlocksRaw.mat"))["ValidationNoisyBlocksRaw"]
    g_raw = loadmat(str(SIDD_VAL / "ValidationGtBlocksRaw.mat"))["ValidationGtBlocksRaw"]
    # shape (40, 32, 256, 256), total 1280 patches
    n_scenes, n_patches = n_raw.shape[:2]
    total = n_scenes * n_patches
    idxs = rng.sample(range(total), n)
    psnrs = []; times = []; fails = 0
    t_start = time.time()
    for k, gi in enumerate(idxs):
        si = gi // n_patches; pi = gi % n_patches
        noisy = n_raw[si, pi].astype(np.float32) / 65535.0 if n_raw.dtype == np.uint16 else n_raw[si, pi].astype(np.float32)
        gt = g_raw[si, pi].astype(np.float32) / 65535.0 if g_raw.dtype == np.uint16 else g_raw[si, pi].astype(np.float32)
        den, dt = run_int_patch(noisy, variant, f"sidd_{si}_{pi}")
        if den is None:
            fails += 1; continue
        psnrs.append(psnr(gt, den)); times.append(dt)
        if (k + 1) % 25 == 0:
            elapsed = time.time() - t_start
            print(f"  [{k+1}/{n}] mean PSNR={np.mean(psnrs):.3f}  elapsed={elapsed:.1f}s")
    p_mean = float(np.mean(psnrs)) if psnrs else 0.0
    p_med = float(np.median(psnrs)) if psnrs else 0.0
    t_total = time.time() - t_start
    print(f"  n_ok={len(psnrs)}, n_fail={fails}, total={t_total:.1f}s, mean_dt={np.mean(times):.2f}s")
    print(f"  RESULT: mean PSNR = {p_mean:.3f} dB,  median PSNR = {p_med:.3f} dB")
    print(f"  v12 full 1280 baseline: {BASELINE['SIDD_val_full_1280']:.3f} dB (= reference for trends)")
    return p_mean, p_med


def bench_rawnind(n, variant, rng):
    print(f"\n=== RawNIND short bench (n={n}, variant={variant}) ===")
    noisy_dir = RAWNIND / "__noisy_raw__"
    gt_dir = RAWNIND / "__gt_raw__"
    tags = sorted(p.stem for p in noisy_dir.glob("*.npy"))
    if not tags:
        print(f"  ERR: no noisy patches in {noisy_dir}")
        return 0, 0
    n = min(n, len(tags))
    sampled = rng.sample(tags, n)
    psnrs = []; times = []; fails = 0
    t_start = time.time()
    for k, tag in enumerate(sampled):
        scene = tag.rsplit("__ISO", 1)[0]
        try:
            noisy = np.load(noisy_dir / f"{tag}.npy").astype(np.float32)
            gt = np.load(gt_dir / f"{scene}.npy").astype(np.float32)
        except Exception:
            fails += 1; continue
        den, dt = run_int_patch(noisy, variant, f"rn_{k}")
        if den is None:
            fails += 1; continue
        psnrs.append(psnr(gt, den)); times.append(dt)
        if (k + 1) % 25 == 0:
            elapsed = time.time() - t_start
            print(f"  [{k+1}/{n}] mean PSNR={np.mean(psnrs):.3f}  elapsed={elapsed:.1f}s")
    p_mean = float(np.mean(psnrs)) if psnrs else 0.0
    p_med = float(np.median(psnrs)) if psnrs else 0.0
    t_total = time.time() - t_start
    print(f"  n_ok={len(psnrs)}, n_fail={fails}, total={t_total:.1f}s, mean_dt={np.mean(times):.2f}s")
    print(f"  RESULT: mean PSNR = {p_mean:.3f} dB,  median PSNR = {p_med:.3f} dB")
    print(f"  v12 full 1493 baseline: {BASELINE['RawNIND_full_1493']:.3f} dB (= reference for trends)")
    return p_mean, p_med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="r32", choices=["r32", "r16"])
    ap.add_argument("--n_sidd", type=int, default=100)
    ap.add_argument("--n_rawnind", type=int, default=100)
    ap.add_argument("--skip_sidd", action="store_true")
    ap.add_argument("--skip_rawnind", action="store_true")
    args = ap.parse_args()

    rng = random.Random(SEED)
    print(f"INT short bench — variant={args.variant}, seed={SEED}")

    sidd_mean, sidd_med = 0, 0
    rn_mean, rn_med = 0, 0
    if not args.skip_sidd:
        rng_sidd = random.Random(SEED)
        sidd_mean, sidd_med = bench_sidd(args.n_sidd, args.variant, rng_sidd)
    if not args.skip_rawnind:
        rng_rn = random.Random(SEED + 1)
        rn_mean, rn_med = bench_rawnind(args.n_rawnind, args.variant, rng_rn)

    print(f"\n{'=' * 60}")
    print(f"  Summary (variant={args.variant}, seeded subsets):")
    print(f"  SIDD val   n={args.n_sidd:>4d}  mean={sidd_mean:.3f}  med={sidd_med:.3f}  (full bench baseline: {BASELINE['SIDD_val_full_1280']:.2f})")
    print(f"  RawNIND    n={args.n_rawnind:>4d}  mean={rn_mean:.3f}  med={rn_med:.3f}  (full bench baseline: {BASELINE['RawNIND_full_1493']:.2f})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
