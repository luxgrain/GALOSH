"""INT r32 vs FP32 per-phase divergence localizer.

Goal: find WHERE the INT r32 pipeline diverges from the FP32 reference,
to localize the -1.06 dB RawNIND quality gap to a specific Phase BEFORE
attempting any precision fix (= bisect to root cause, not shotgun).

Mechanism:
  - galosh_raw_cpu.exe     (FP32 --variant=o)  dumps via GALOSH_DUMP_DIR
  - galosh_raw_cpu_int.exe (INT  --variant=r32) dumps via GALOSH_INT_DUMP_DIR
  Both emit the same named phase buffers as float32 .bin.

For each common phase buffer, divergence is reported as a scale-invariant
"match dB" = 20*log10( ||fp32|| / ||int - fp32|| ).  Higher = closer.
The phase where match-dB DROPS sharply is the divergence source.

Also reports GT-PSNR at p10 for both INT and FP32 (= the visible gap).

Usage:
  python benchmark/scripts/int_phase_divergence.py
  python benchmark/scripts/int_phase_divergence.py --tags D60-3__ISO400 ...
"""
import argparse
import os
import sys
import subprocess
import shutil
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
WORK = GALOSH / "benchmark" / "results" / "int_phase_divergence"
WORK.mkdir(parents=True, exist_ok=True)

# Phase buffers in pipeline order (= names both exes dump).
PHASES = [
    "p3_L_cs",        # Phase 3: luma after cross-shrink (GAT domain)
    "p5_pilot",       # Phase 5 Pass1: BayesShrink pilot
    "p5_L_cs_den",    # Phase 5 Pass2: Wiener output
    "p6_L_pixel",     # Phase 6: luma back to pixel domain
    "p6_L_h_den",     # Phase 6: luma half-res denoised
    "p7_C1_q_up",     # Phase 7: chroma C1 quadratic upsample
    "p8_C1_h_den",    # Phase 8: chroma C1 half-res denoised
    "p9_C1_aligned",  # Phase 9: chroma C1 aligned to full-res
    "p10_output",     # Phase 10: final output
]


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def match_db(fp, it):
    """Scale-invariant divergence: 20*log10(||fp|| / ||it-fp||)."""
    fp = fp.astype(np.float64); it = it.astype(np.float64)
    sig = np.sqrt(np.sum(fp * fp))
    err = np.sqrt(np.sum((it - fp) ** 2))
    if err < 1e-30:
        return 999.0
    if sig < 1e-30:
        return -999.0
    return float(20.0 * np.log10(sig / err))


def run_exe(exe, noisy, w, h, variant, dump_env, dump_dir, uid):
    in_p = WORK / f"_{uid}_in.bin"
    out_p = WORK / f"_{uid}_out.bin"
    noisy.astype(np.float32).tofile(str(in_p))
    env = dict(os.environ)
    env[dump_env] = str(dump_dir)
    cmd = [str(exe), str(in_p), str(out_p), str(w), str(h),
           "galosh", "1.0", "1.0", "1.0", "0", "0", f"--variant={variant}"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300, env=env)
    except subprocess.TimeoutExpired:
        in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True)
        return None
    in_p.unlink(missing_ok=True)
    if r.returncode != 0 or not out_p.exists():
        err = r.stderr.decode("utf-8", errors="replace")[:400] if r.stderr else ""
        print(f"  exe FAIL ({variant}): rc={r.returncode} {err}")
        out_p.unlink(missing_ok=True)
        return None
    out = np.fromfile(str(out_p), dtype=np.float32).reshape(h, w)
    out_p.unlink(missing_ok=True)
    return out


def load_phase(dump_dir, name):
    p = dump_dir / f"{name}.bin"
    if not p.exists():
        return None
    return np.fromfile(str(p), dtype=np.float32)


def process_tag(tag):
    scene = tag.rsplit("__ISO", 1)[0]
    noisy_p = RAWNIND / "__noisy_raw__" / f"{tag}.npy"
    gt_p = RAWNIND / "__gt_raw__" / f"{scene}.npy"
    if not noisy_p.exists() or not gt_p.exists():
        print(f"[{tag}] missing npy, skip")
        return None
    noisy = np.load(noisy_p).astype(np.float32)
    gt = np.load(gt_p).astype(np.float32)
    h, w = noisy.shape

    dd_fp = WORK / "dump_fp"; dd_int = WORK / "dump_int"
    for d in (dd_fp, dd_int):
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True)

    out_fp = run_exe(EXE_FP, noisy, w, h, "o", "GALOSH_DUMP_DIR", dd_fp, f"fp_{tag}")
    out_int = run_exe(EXE_INT, noisy, w, h, "r32", "GALOSH_INT_DUMP_DIR", dd_int, f"int_{tag}")
    if out_fp is None or out_int is None:
        return None

    gt_psnr_fp = psnr(gt, out_fp)
    gt_psnr_int = psnr(gt, out_int)

    rows = []
    prev_db = None
    for name in PHASES:
        a = load_phase(dd_fp, name)
        b = load_phase(dd_int, name)
        if a is None or b is None or a.shape != b.shape:
            rows.append((name, None, None))
            continue
        mdb = match_db(a, b)
        drop = (prev_db - mdb) if prev_db is not None else 0.0
        rows.append((name, mdb, drop))
        prev_db = mdb

    shutil.rmtree(dd_fp, ignore_errors=True)
    shutil.rmtree(dd_int, ignore_errors=True)

    return {
        "tag": tag, "shape": (h, w),
        "gt_psnr_fp": gt_psnr_fp, "gt_psnr_int": gt_psnr_int,
        "gap": gt_psnr_int - gt_psnr_fp, "rows": rows,
    }


def pick_default_tags(n_per_bucket=2):
    """Spread across ISO buckets, weighting high-noise (= where gap lives)."""
    noisy_dir = RAWNIND / "__noisy_raw__"
    tags = sorted(p.stem for p in noisy_dir.glob("*.npy"))
    buckets = {"<=400": [], "400-1600": [], "1600-6400": [], ">6400": []}
    for t in tags:
        if "__ISO" not in t: continue
        try: iso = int(t.rsplit("__ISO", 1)[1].split("_")[0])
        except: continue
        if iso <= 400: buckets["<=400"].append(t)
        elif iso <= 1600: buckets["400-1600"].append(t)
        elif iso <= 6400: buckets["1600-6400"].append(t)
        else: buckets[">6400"].append(t)
    import random
    rng = random.Random(42)
    picked = []
    for b, lst in buckets.items():
        if lst:
            picked.extend(rng.sample(lst, min(n_per_bucket, len(lst))))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--n_per_bucket", type=int, default=2)
    args = ap.parse_args()

    tags = args.tags if args.tags else pick_default_tags(args.n_per_bucket)
    print(f"INT vs FP32 per-phase divergence — {len(tags)} patches")
    print(f"match-dB = 20*log10(||fp32|| / ||int-fp32||); higher=closer; big DROP=divergence source\n")

    results = []
    for tag in tags:
        print(f"[{tag}] running both exes ...")
        r = process_tag(tag)
        if r is None:
            print(f"  skip/fail\n"); continue
        results.append(r)
        h, w = r["shape"]
        print(f"  shape={h}x{w}  GT-PSNR: FP32={r['gt_psnr_fp']:.3f}  INT={r['gt_psnr_int']:.3f}  gap={r['gap']:+.3f} dB")
        print(f"  {'phase':<16} {'match-dB':>10} {'drop':>8}")
        for name, mdb, drop in r["rows"]:
            if mdb is None:
                print(f"  {name:<16} {'(n/a)':>10}")
            else:
                flag = "  <== big drop" if (drop is not None and drop > 6.0) else ""
                print(f"  {name:<16} {mdb:>10.2f} {drop:>8.2f}{flag}")
        print()

    if not results:
        print("no results")
        return

    # Aggregate: mean drop per phase across patches.
    print("=" * 64)
    print("AGGREGATE — mean match-dB and mean drop per phase")
    print(f"  {'phase':<16} {'mean match-dB':>14} {'mean drop':>10}")
    for i, name in enumerate(PHASES):
        mdbs = [r["rows"][i][1] for r in results if r["rows"][i][1] is not None]
        drops = [r["rows"][i][2] for r in results if r["rows"][i][2] is not None]
        if not mdbs:
            print(f"  {name:<16} {'(n/a)':>14}")
            continue
        mm = float(np.mean(mdbs)); md = float(np.mean(drops))
        flag = "  <== divergence source" if md > 6.0 else ""
        print(f"  {name:<16} {mm:>14.2f} {md:>10.2f}{flag}")
    gaps = [r["gap"] for r in results]
    print(f"\n  mean GT-PSNR gap (INT - FP32): {np.mean(gaps):+.3f} dB over {len(results)} patches")
    print("=" * 64)


if __name__ == "__main__":
    main()
