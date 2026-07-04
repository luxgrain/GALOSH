"""P0 GPU vs CPU bit-exact comparison.

Runs the r32 CPU reference (galosh_raw_cpu_int.exe) and the GPU Phase-0
harness (galosh_int_p0_test.exe) on the SAME SIDD val patches and compares the
raw Q11.20 alpha / sigma_sq.  Both consume the identical float32 .bin and
convert with the identical fxp_from_float, so a correct GPU port is BIT-EXACT.

Usage:
  python benchmark/scripts/int_p0_gpu_compare.py            # 20 patches
  python benchmark/scripts/int_p0_gpu_compare.py --n 60
"""
import argparse, os, re, subprocess, sys, random
from pathlib import Path
import numpy as np
from scipy.io import loadmat

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

GALOSH   = Path(os.path.expanduser(r"~\GALOSH"))
CPU_EXE  = GALOSH / "standalone" / "galosh_raw_cpu_int.exe"
GPU_EXE  = GALOSH / "standalone" / "galosh_int_p0_test.exe"
SIDD_VAL = Path(os.environ.get("GALOSH_SIDD_VAL", "benchmark/datasets/SIDD_Validation"))
TMP      = GALOSH / "benchmark" / "results" / "int_p0_gpu_tmp"
TMP.mkdir(parents=True, exist_ok=True)

CPU_RE = re.compile(r"Q11\.20:\s*(-?\d+),\s*(-?\d+)")
GPU_RE = re.compile(r"P0_RAW\s+alpha=(-?\d+)\s+sigma_sq=(-?\d+)")
SEED = 42


def cpu_ab(in_p, out_p, w, h):
    cmd = [str(CPU_EXE), str(in_p), str(out_p), str(w), str(h),
           "galosh", "1.0", "1.0", "1.0", "0", "0", "--variant=r32"]
    r = subprocess.run(cmd, capture_output=True, timeout=180)
    err = r.stderr.decode("utf-8", "replace")
    m = CPU_RE.search(err)
    return (int(m.group(1)), int(m.group(2))) if m else None


def gpu_ab(in_p, w, h):
    cmd = [str(GPU_EXE), str(in_p), str(w), str(h)]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    out = r.stdout.decode("utf-8", "replace")
    m = GPU_RE.search(out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    n_raw = loadmat(str(SIDD_VAL / "ValidationNoisyBlocksRaw.mat"))["ValidationNoisyBlocksRaw"]
    n_scenes, n_patches = n_raw.shape[:2]
    total = n_scenes * n_patches
    rng = random.Random(SEED)
    idxs = rng.sample(range(total), min(args.n, total))

    print(f"P0 GPU vs CPU bit-exact (n={len(idxs)})")
    print(f"{'patch':<12} {'CPU a,s':>18} {'GPU a,s':>18}  match")
    n_ok = n_bad = n_fail = 0
    bad = []
    for gi in idxs:
        si, pi = gi // n_patches, gi % n_patches
        noisy = n_raw[si, pi]
        noisy = noisy.astype(np.float32) / 65535.0 if noisy.dtype == np.uint16 else noisy.astype(np.float32)
        h, w = noisy.shape
        in_p = TMP / f"s{si}_p{pi}.bin"
        out_p = TMP / f"s{si}_p{pi}_out.bin"
        noisy.astype(np.float32).tofile(str(in_p))
        c = cpu_ab(in_p, out_p, w, h)
        g = gpu_ab(in_p, w, h)
        in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True)
        if c is None or g is None:
            n_fail += 1
            print(f"s{si}_p{pi:<8} {'PARSE FAIL':>18} {str(g):>18}")
            continue
        match = (c == g)
        if match: n_ok += 1
        else:
            n_bad += 1
            bad.append((si, pi, c, g))
        print(f"s{si}_p{pi:<8} {str(c):>18} {str(g):>18}  {'OK' if match else 'MISMATCH'}")

    print(f"\n=== {n_ok} bit-exact / {n_bad} mismatch / {n_fail} fail (of {len(idxs)}) ===")
    if bad:
        print("First mismatches (CPU vs GPU raw Q11.20 alpha,sigma):")
        for si, pi, c, g in bad[:8]:
            da, ds = g[0] - c[0], g[1] - c[1]
            print(f"  s{si}_p{pi}: CPU={c} GPU={g}  d_alpha={da} d_sigma={ds}")
    return 0 if (n_bad == 0 and n_fail == 0) else 2


if __name__ == "__main__":
    sys.exit(main())
