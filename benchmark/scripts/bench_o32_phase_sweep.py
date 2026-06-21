"""o32 cycle-spin (N_PHASES) quality-vs-speed tradeoff sweep on SIDD val.

Runs GPU o32 at GALOSH_O32_PHASE_STRIDE in {1,2,4} (= N_PHASES 16/4/1) over a
deterministic subset (default 384) of SIDD val RAW; PSNR/SSIM vs gt_raw (same
metric as bench_sidd_validation, comparable to the canonical o 49.47).

Cycle-spinning removes 8x8 blocking; fewer phases = faster pass12 but potential
blocking (a perceptual artifact PSNR/SSIM only partly capture — flag for visual).

  python bench_o32_phase_sweep.py [N]     # N patches (default 384; 0 = full 1280)
"""
import os, sys, time, subprocess
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim_fn

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

GALOSH = Path(r"C:\Users\luxgrain\GALOSH")
VAL = GALOSH / "benchmark" / "SIDD_Validation"
OUTROOT = GALOSH / "benchmark" / "sidd_validation"
TMP = GALOSH / "benchmark" / "results" / "speed" / "_o32sweep_tmp"
TMP.mkdir(parents=True, exist_ok=True)
EXE = GALOSH / "standalone" / "galosh_raw_gpu.exe"
STRIDES = [1, 2, 4]   # N_PHASES 16 / 4 / 1
NPH = {1: 16, 2: 4, 4: 1}


def psnr(ref, test):
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def run_o32(noisy, w, h, uid, stride):
    in_p = TMP / f"{uid}_in.bin"; out_p = TMP / f"{uid}_out.bin"
    noisy.astype(np.float32).tofile(str(in_p))
    env = dict(os.environ); env["GALOSH_O32_PHASE_STRIDE"] = str(stride)
    cmd = [str(EXE), str(in_p), str(out_p), str(w), str(h),
           "1.0", "1.0", "1.0", "0", "0", "0", "--variant=o32"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120, env=env)
    except subprocess.TimeoutExpired:
        in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True); return None
    in_p.unlink(missing_ok=True)
    if r.returncode != 0 or not out_p.exists():
        sys.stderr.write(r.stderr.decode("utf-8", "replace")[-300:] + "\n"); return None
    den = np.fromfile(str(out_p), dtype=np.float32).reshape(h, w)
    out_p.unlink(missing_ok=True)
    return den


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 384
    N_RAW = loadmat(str(VAL / "ValidationNoisyBlocksRaw.mat"))["ValidationNoisyBlocksRaw"]
    G_RAW = loadmat(str(VAL / "ValidationGtBlocksRaw.mat"))["ValidationGtBlocksRaw"]
    ns, npp = N_RAW.shape[:2]; tot = ns * npp
    if N <= 0 or N > tot: N = tot
    idxs = list(range(0, tot, max(1, tot // N)))[:N]
    print(f"o32 N_PHASES sweep: {len(idxs)} patches (of {tot}); metric = RAW PSNR/SSIM vs gt (cf. canonical o 49.47)")

    res = {}
    for st in STRIDES:
        outd = OUTROOT / f"o32_ps{st}"; outd.mkdir(parents=True, exist_ok=True)
        pss = []; sss = []; t0 = time.time(); fails = 0
        for k, gi in enumerate(idxs):
            s, p = gi // npp, gi % npp
            nr = np.ascontiguousarray(N_RAW[s, p]).astype(np.float32)
            gr = np.ascontiguousarray(G_RAW[s, p]).astype(np.float32)
            h, w = nr.shape
            onpy = outd / f"s{s:02d}_p{p:02d}.npy"
            den = np.load(onpy) if onpy.exists() else run_o32(nr, w, h, f"{st}_{gi}", st)
            if den is None: fails += 1; continue
            if not onpy.exists(): np.save(onpy, den.astype(np.float32))
            pss.append(psnr(gr, den)); sss.append(float(ssim_fn(gr, den, data_range=1.0)))
        res[st] = (np.mean(pss), np.mean(sss), time.time() - t0, fails)
        print(f"  stride={st} N_PHASES={NPH[st]:>2}: PSNR={res[st][0]:.3f} SSIM={res[st][1]:.4f} "
              f"({len(pss)} ok, {fails} fail, {res[st][2]/60:.1f}min)")

    print("\n=== o32 cycle-spin tradeoff (SIDD val) ===")
    p16 = res[1][0]; s16 = res[1][1]
    for st in STRIDES:
        ps, ss, _, _ = res[st]
        print(f"  N_PHASES={NPH[st]:>2} (stride{st}): PSNR={ps:.3f} ({ps-p16:+.3f})  SSIM={ss:.4f} ({ss-s16:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
