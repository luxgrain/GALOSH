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

PHASE_FILE = {1: "p1_ingat.bin", 2: "p2_ingat.bin", 3: "p3_lcs.bin", 4: "p4_chroma.bin",
              5: "p5_den.bin", 6: "p6.bin", 7: "p7_full.bin",
              8: "p8.bin", 9: "p9.bin", 10: "p10.bin"}

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


def run_gpu(in_p, w, h, phase, out_p1, i16=False, lf=None, cf=None):
    cmd = [str(GPU_EXE), str(in_p), str(w), str(h), str(phase), str(out_p1)]
    if i16:
        cmd += ["0", "i16"]
        if lf is not None: cmd.append(f"lf={lf}")
        if cf is not None: cmd.append(f"cf={cf}")
    r = subprocess.run(cmd, capture_output=True, timeout=120, env=SUBENV)
    return ints_after(r.stdout.decode("utf-8", "replace"), phase)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--i16", action="store_true",
                    help="narrow GPU line buffers to INT16 precision (zero-loss check)")
    ap.add_argument("--lf", type=int, default=None, help="luma fractional bits (default 5)")
    ap.add_argument("--cf", type=int, default=None, help="chroma fractional bits (default 9)")
    args = ap.parse_args()
    ph = args.phase

    n_raw = loadmat(str(SIDD_VAL / "ValidationNoisyBlocksRaw.mat"))["ValidationNoisyBlocksRaw"]
    ns, npp = n_raw.shape[:2]
    rng = random.Random(SEED)
    idxs = rng.sample(range(ns * npp), min(args.n, ns * npp))

    print(f"Phase {ph} GPU vs CPU {'INT16-narrow zero-loss' if args.i16 else 'bit-exact'} (n={len(idxs)})")
    n_ok = n_bad = n_fail = 0; bad = []; psnrs = []
    for gi in idxs:
        si, pi = gi // npp, gi % npp
        noisy = n_raw[si, pi]
        noisy = noisy.astype(np.float32) / 65535.0 if noisy.dtype == np.uint16 else noisy.astype(np.float32)
        h, w = noisy.shape
        in_p = TMP / f"s{si}_p{pi}.bin"; out_p = TMP / f"s{si}_p{pi}_o.bin"
        gpu_b = TMP / f"s{si}_p{pi}_gpu.bin"
        noisy.astype(np.float32).tofile(str(in_p))
        c = run_cpu(in_p, out_p, w, h, ph)
        cpu_b = TMP / PHASE_FILE.get(ph, f"p{ph}_ingat.bin")
        g = run_gpu(in_p, w, h, ph, gpu_b, i16=args.i16, lf=args.lf, cf=args.cf)
        if c is None or g is None or not cpu_b.exists() or not gpu_b.exists():
            n_fail += 1; print(f"  s{si}_p{pi}: FAIL (cpu={c} gpu={g})")
            for p in (in_p, out_p, gpu_b): p.unlink(missing_ok=True)
            continue
        ca = np.fromfile(str(cpu_b), dtype=np.int32); ga = np.fromfile(str(gpu_b), dtype=np.int32)
        ndiff = int(np.count_nonzero(ca != ga)) if ca.shape == ga.shape else -1
        if args.i16:
            cf = ca.astype(np.float64) / 2**20; gf = ga.astype(np.float64) / 2**20
            mse = float(np.mean((cf - gf) ** 2))
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
            maxd = float(np.max(np.abs(cf - gf)))
            psnrs.append(psnr)
            ok = (ndiff == 0) or (psnr >= 80.0)   # >=80 dB vs r32 = effectively zero-loss
            if ok: n_ok += 1
            else: n_bad += 1; bad.append((si, pi, c, g, ndiff))
            print(f"  s{si}_p{pi:<3} PSNR(i16,r32)={psnr:6.2f} dB  maxΔ={maxd:.2e}  ndiff={ndiff}  {'OK' if ok else 'LOSSY'}")
        else:
            ok = (ndiff == 0)
            if ok: n_ok += 1
            else: n_bad += 1; bad.append((si, pi, c, g, ndiff))
            print(f"  s{si}_p{pi:<3} ndiff={ndiff}  {'OK' if ok else 'MISMATCH'}")
        for p in (in_p, out_p, gpu_b): p.unlink(missing_ok=True)

    if args.i16 and psnrs:
        import statistics
        print(f"\n=== phase {ph} INT16-narrow: {n_ok} zero-loss(>=80dB) / {n_bad} lossy / {n_fail} fail "
              f"(of {len(idxs)});  PSNR(i16,r32) mean={statistics.mean(psnrs):.2f} min={min(psnrs):.2f} dB ===")
        return 0 if n_bad == 0 and n_fail == 0 else 2
    print(f"\n=== phase {ph}: {n_ok} bit-exact / {n_bad} mismatch / {n_fail} fail (of {len(idxs)}) ===")
    for si, pi, c, g, nd in bad[:8]:
        print(f"  s{si}_p{pi}: CPU={c} GPU={g} buf_ndiff={nd}")
    return 0 if (n_bad == 0 and n_fail == 0) else 2


if __name__ == "__main__":
    sys.exit(main())
