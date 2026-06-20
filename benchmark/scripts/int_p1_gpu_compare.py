"""P1 (GAT forward + per-CFA sigma + unified_sigma) GPU vs CPU bit-exact compare.

CPU r32 reference dumps the normalized in_gat buffer as raw int32 (p1_ingat.bin)
via GALOSH_INT_RAW_DUMP_DIR, and prints "P1_RAW unified_sigma=.. sigma_ch=..".
The GPU harness (galosh_int_p1_test.exe) writes the same buffer raw + prints the
same line.  A correct port is BIT-EXACT on the full GAT-domain buffer.

Usage:
  python benchmark/scripts/int_p1_gpu_compare.py --n 20
"""
import argparse, os, re, subprocess, sys, random
from pathlib import Path
import numpy as np
from scipy.io import loadmat

os.environ["PYTHONIOENCODING"] = "utf-8"
SUBENV = dict(os.environ)
SUBENV["PATH"] = r"C:\msys64\ucrt64\bin;" + SUBENV.get("PATH", "")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

GALOSH   = Path(r"C:\Users\luxgrain\GALOSH")
CPU_EXE  = GALOSH / "standalone" / "galosh_raw_cpu_int.exe"
GPU_EXE  = GALOSH / "standalone" / "galosh_int_p1_test.exe"
SIDD_VAL = GALOSH / "benchmark" / "SIDD_Validation"
TMP      = GALOSH / "benchmark" / "results" / "int_p1_gpu_tmp"
TMP.mkdir(parents=True, exist_ok=True)

P1_RE = re.compile(r"P1_RAW\s+unified_sigma=(-?\d+)\s+sigma_ch=(-?\d+),(-?\d+),(-?\d+),(-?\d+)")
SEED = 42


def run_cpu(in_p, out_p, w, h, dump_dir):
    env = dict(SUBENV); env["GALOSH_INT_RAW_DUMP_DIR"] = str(dump_dir)
    cmd = [str(CPU_EXE), str(in_p), str(out_p), str(w), str(h),
           "galosh", "1.0", "1.0", "1.0", "0", "0", "--variant=r32"]
    r = subprocess.run(cmd, capture_output=True, timeout=240, env=env)
    m = P1_RE.search(r.stderr.decode("utf-8", "replace"))
    return tuple(int(x) for x in m.groups()) if m else None


def run_gpu(in_p, w, h, out_p1):
    cmd = [str(GPU_EXE), str(in_p), str(w), str(h), str(out_p1)]
    r = subprocess.run(cmd, capture_output=True, timeout=120, env=SUBENV)
    m = P1_RE.search(r.stdout.decode("utf-8", "replace"))
    return tuple(int(x) for x in m.groups()) if m else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    n_raw = loadmat(str(SIDD_VAL / "ValidationNoisyBlocksRaw.mat"))["ValidationNoisyBlocksRaw"]
    ns, npp = n_raw.shape[:2]
    rng = random.Random(SEED)
    idxs = rng.sample(range(ns * npp), min(args.n, ns * npp))

    print(f"P1 GPU vs CPU bit-exact (n={len(idxs)})")
    n_ok = n_bad = n_fail = 0
    bad = []
    for gi in idxs:
        si, pi = gi // npp, gi % npp
        noisy = n_raw[si, pi]
        noisy = noisy.astype(np.float32) / 65535.0 if noisy.dtype == np.uint16 else noisy.astype(np.float32)
        h, w = noisy.shape
        in_p = TMP / f"s{si}_p{pi}.bin"; out_p = TMP / f"s{si}_p{pi}_o.bin"
        gpu_p1 = TMP / f"s{si}_p{pi}_gpu.bin"
        noisy.astype(np.float32).tofile(str(in_p))
        c = run_cpu(in_p, out_p, w, h, TMP)
        cpu_p1 = TMP / "p1_ingat.bin"
        g = run_gpu(in_p, w, h, gpu_p1)
        if c is None or g is None or not cpu_p1.exists() or not gpu_p1.exists():
            n_fail += 1
            print(f"  s{si}_p{pi}: FAIL (cpu={c} gpu={g})")
            for p in (in_p, out_p, gpu_p1): p.unlink(missing_ok=True)
            continue
        ca = np.fromfile(str(cpu_p1), dtype=np.int32)
        ga = np.fromfile(str(gpu_p1), dtype=np.int32)
        buf_match = ca.shape == ga.shape and np.array_equal(ca, ga)
        scal_match = (c == g)
        ndiff = int(np.count_nonzero(ca != ga)) if ca.shape == ga.shape else -1
        ok = buf_match and scal_match
        if ok: n_ok += 1
        else:
            n_bad += 1
            bad.append((si, pi, c, g, ndiff))
        tag = "OK" if ok else f"MISMATCH(buf_ndiff={ndiff}, scal={'ok' if scal_match else 'BAD'})"
        print(f"  s{si}_p{pi:<3} uni_cpu={c[0]:<7} uni_gpu={g[0]:<7}  {tag}")
        for p in (in_p, out_p, gpu_p1): p.unlink(missing_ok=True)

    print(f"\n=== {n_ok} bit-exact / {n_bad} mismatch / {n_fail} fail (of {len(idxs)}) ===")
    for si, pi, c, g, nd in bad[:8]:
        print(f"  s{si}_p{pi}: CPU={c} GPU={g} buf_ndiff={nd}")
    return 0 if (n_bad == 0 and n_fail == 0) else 2


if __name__ == "__main__":
    sys.exit(main())
