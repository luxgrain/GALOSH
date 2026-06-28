"""INT r32 vs FP32 GT-PSNR gap, split by ISO bucket.

Hypothesis under test (2026-06-06): the RawNIND -1.06 dB INT gap is NOT
uniform.  It is concentrated in low-ISO (clean) patches where the Q11.20
sigma^2 floor (= deliberate minimum-noise feature) costs dB at very high
PSNR, while high-ISO (heavy-noise = GALOSH's main regime) patches already
match FP32.

If confirmed: chasing the gap via sigma^2 precision is wrong (attacks the
clean regime + is the v14-v17 trap zone).

Runs galosh_raw_cpu.exe (FP32 -o) and galosh_raw_cpu_int.exe (-r32) on the
same patches, records GT-PSNR for each, aggregates gap per ISO bucket.

Usage:
  python benchmark/scripts/int_gap_by_iso.py --n_per_bucket 15
"""
import argparse
import os
import sys
import subprocess
import pickle
from pathlib import Path
import numpy as np

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

GALOSH = Path(os.path.expanduser(r"~\GALOSH"))
EXE_FP = GALOSH / "standalone" / "galosh_raw_cpu.exe"
EXE_INT = GALOSH / "standalone" / "galosh_raw_cpu_int.exe"
RAWNIND = Path(r"E:\rawnind_bench")
WORK = GALOSH / "benchmark" / "results" / "int_gap_by_iso"
WORK.mkdir(parents=True, exist_ok=True)

BUCKETS = ["<=400", "400-1600", "1600-6400", ">6400"]


def iso_of(tag):
    try:
        return int(tag.rsplit("__ISO", 1)[1].split("_")[0])
    except Exception:
        return None


def bucket_of(iso):
    if iso is None: return None
    if iso <= 400: return "<=400"
    if iso <= 1600: return "400-1600"
    if iso <= 6400: return "1600-6400"
    return ">6400"


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def run_exe(exe, noisy, w, h, variant, uid):
    in_p = WORK / f"_{uid}_in.bin"
    out_p = WORK / f"_{uid}_out.bin"
    noisy.astype(np.float32).tofile(str(in_p))
    cmd = [str(exe), str(in_p), str(out_p), str(w), str(h),
           "galosh", "1.0", "1.0", "1.0", "0", "0", f"--variant={variant}"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True)
        return None
    in_p.unlink(missing_ok=True)
    if r.returncode != 0 or not out_p.exists():
        out_p.unlink(missing_ok=True)
        return None
    out = np.fromfile(str(out_p), dtype=np.float32).reshape(h, w)
    out_p.unlink(missing_ok=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_bucket", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    noisy_dir = RAWNIND / "__noisy_raw__"
    gt_dir = RAWNIND / "__gt_raw__"
    tags = sorted(p.stem for p in noisy_dir.glob("*.npy"))

    by_bucket = {b: [] for b in BUCKETS}
    for t in tags:
        b = bucket_of(iso_of(t))
        if b: by_bucket[b].append(t)

    import random
    rng = random.Random(args.seed)
    picked = []
    for b in BUCKETS:
        lst = by_bucket[b]
        picked += [(b, t) for t in rng.sample(lst, min(args.n_per_bucket, len(lst)))]

    print(f"INT r32 vs FP32 GT-PSNR gap by ISO bucket — {len(picked)} patches\n")
    results = []
    for i, (b, tag) in enumerate(picked):
        scene = tag.rsplit("__ISO", 1)[0]
        try:
            noisy = np.load(noisy_dir / f"{tag}.npy").astype(np.float32)
            gt = np.load(gt_dir / f"{scene}.npy").astype(np.float32)
        except Exception:
            continue
        h, w = noisy.shape
        out_fp = run_exe(EXE_FP, noisy, w, h, "o", f"fp_{i}")
        out_int = run_exe(EXE_INT, noisy, w, h, "r32", f"int_{i}")
        if out_fp is None or out_int is None:
            print(f"  [{i+1}/{len(picked)}] {tag}: FAIL")
            continue
        pf = psnr(gt, out_fp); pi = psnr(gt, out_int)
        results.append({"bucket": b, "tag": tag, "iso": iso_of(tag),
                        "psnr_fp": pf, "psnr_int": pi, "gap": pi - pf})
        print(f"  [{i+1}/{len(picked)}] {b:<10} {tag[:38]:<38} FP={pf:6.2f} INT={pi:6.2f} gap={pi-pf:+.3f}")

    out_pkl = WORK / f"gap_by_iso_n{args.n_per_bucket}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)

    print(f"\n{'='*64}")
    print(f"AGGREGATE — mean GT-PSNR gap (INT - FP32) by ISO bucket")
    print(f"  {'bucket':<12} {'n':>3} {'FP32':>8} {'INT':>8} {'gap':>9} {'worst':>9}")
    allgaps = []
    for b in BUCKETS:
        rows = [r for r in results if r["bucket"] == b]
        if not rows: continue
        gaps = [r["gap"] for r in rows]; allgaps += gaps
        fp = np.mean([r["psnr_fp"] for r in rows])
        it = np.mean([r["psnr_int"] for r in rows])
        print(f"  {b:<12} {len(rows):>3} {fp:>8.2f} {it:>8.2f} {np.mean(gaps):>+9.3f} {min(gaps):>+9.3f}")
    print(f"  {'-'*52}")
    print(f"  {'OVERALL':<12} {len(allgaps):>3} {'':>8} {'':>8} {np.mean(allgaps):>+9.3f} {min(allgaps):>+9.3f}")
    print(f"{'='*64}")
    print(f"  saved: {out_pkl}")


if __name__ == "__main__":
    main()
