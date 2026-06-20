"""Per-phase GPU vs CPU bit-exact comparison for the i16 GPU port.

For phase N, the CPU r32 reference (run with GALOSH_INT_RAW_DUMP_DIR) dumps the
working in_gat buffer as raw int32 (pN_ingat.bin) and prints "PN_RAW ...".  The
GPU pipeline harness (galosh_int_pipe_test.exe) runs P0..PN, writes the same
buffer raw, and prints the same line.  A correct port is BIT-EXACT.

Usage:
  python benchmark/scripts/int_phase_gpu_compare.py --phase 2 --n 20
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
GPU_EXE  = GALOSH / "standalone" / "galosh_int_pipe_test.exe"
SIDD_VAL = GALOSH / "benchmark" / "SIDD_Validation"
TMP      = GALOSH / "benchmark" / "results" / "int_phase_gpu_tmp"
TMP.mkdir(parents=True, exist_ok=True)
SEED = 42


def ints_after(text, phase):
    m = re.search(rf"P{phase}_RAW[^\n]*", text)
    return [int(x) for x in re.findall(r"-?\d+", m.group(0).split("RAW", 1)[1])] if m else None


def run_cpu(in_p, out_p, w, h, phase):
    env = dict(SUBENV); env["GALOSH_INT_RAW_DUMP_DIR"] = str(TMP)
    cmd = [str(CPU_EXE), str(in_p), str(out_p), str(w), str(h),
           "galosh", "1.0", "1.0", "1.0", "0", "0", "--variant=r32"]
    r = subprocess.run(cmd, capture_output=True, timeout=240, env=env)
    return ints_after(r.stderr.decode("utf-8", "replace"), phase)


def run_gpu(in_p, w, h, phase, out_p1):
    cmd = [str(GPU_EXE), str(in_p), str(w), str(h), str(phase), str(out_p1)]
    r = subprocess.run(cmd, capture_output=True, timeout=120, env=SUBENV)
    return ints_after(r.stdout.decode("utf-8", "replace"), phase)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    ph = args.phase

    n_raw = loadmat(str(SIDD_VAL / "ValidationNoisyBlocksRaw.mat"))["ValidationNoisyBlocksRaw"]
    ns, npp = n_raw.shape[:2]
    rng = random.Random(SEED)
    idxs = rng.sample(range(ns * npp), min(args.n, ns * npp))

    print(f"Phase {ph} GPU vs CPU bit-exact (n={len(idxs)})")
    n_ok = n_bad = n_fail = 0; bad = []
    for gi in idxs:
        si, pi = gi // npp, gi % npp
        noisy = n_raw[si, pi]
        noisy = noisy.astype(np.float32) / 65535.0 if noisy.dtype == np.uint16 else noisy.astype(np.float32)
        h, w = noisy.shape
        in_p = TMP / f"s{si}_p{pi}.bin"; out_p = TMP / f"s{si}_p{pi}_o.bin"
        gpu_b = TMP / f"s{si}_p{pi}_gpu.bin"
        noisy.astype(np.float32).tofile(str(in_p))
        c = run_cpu(in_p, out_p, w, h, ph)
        cpu_b = TMP / f"p{ph}_ingat.bin"
        g = run_gpu(in_p, w, h, ph, gpu_b)
        if c is None or g is None or not cpu_b.exists() or not gpu_b.exists():
            n_fail += 1; print(f"  s{si}_p{pi}: FAIL (cpu={c} gpu={g})")
            for p in (in_p, out_p, gpu_b): p.unlink(missing_ok=True)
            continue
        ca = np.fromfile(str(cpu_b), dtype=np.int32); ga = np.fromfile(str(gpu_b), dtype=np.int32)
        ndiff = int(np.count_nonzero(ca != ga)) if ca.shape == ga.shape else -1
        ok = (ndiff == 0) and (c == g)
        if ok: n_ok += 1
        else: n_bad += 1; bad.append((si, pi, c, g, ndiff))
        print(f"  s{si}_p{pi:<3} cpu={c} gpu={g}  {'OK' if ok else f'MISMATCH(ndiff={ndiff})'}")
        for p in (in_p, out_p, gpu_b): p.unlink(missing_ok=True)

    print(f"\n=== phase {ph}: {n_ok} bit-exact / {n_bad} mismatch / {n_fail} fail (of {len(idxs)}) ===")
    for si, pi, c, g, nd in bad[:8]:
        print(f"  s{si}_p{pi}: CPU={c} GPU={g} buf_ndiff={nd}")
    return 0 if (n_bad == 0 and n_fail == 0) else 2


if __name__ == "__main__":
    sys.exit(main())
