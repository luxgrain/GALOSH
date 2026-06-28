"""SIDD Validation 1280-patch quality bench for GALOSH_RAW_G (variant g).

Purpose: the published SIDD RAW numbers (PSNR 49.47 / SSIM 0.9865) are from the
canonical heavy `o` algorithm; the *production / 4K60* pipeline is `g`, whose
quality was never measured in this harness.  This script fills that gap.

Metric = SAME as bench_sidd_validation.py: RAW-plane CFA-agnostic PSNR + SSIM
vs gt_raw (data_range 1.0), so the result is directly comparable to o's 49.47.

Flow:
  1. VERIFY (N=24): run CPU g AND GPU g; report PSNR(cpu_g, gpu_g) agreement.
     If they agree (>=45 dB ~ FP rounding only), proceed with the FAST GPU g.
  2. FULL (1280): GPU g on every patch; mean PSNR/SSIM; save .npy per patch.

Artifacts -> benchmark/sidd_validation/galosh_raw_gpu_g/  (npy, per persistence rule).
"""
import os, sys, time, subprocess
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim_fn

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
os.environ["PYTHONIOENCODING"] = "utf-8"
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

GALOSH = Path(os.path.expanduser(r"~\GALOSH"))
VAL    = GALOSH / "benchmark" / "SIDD_Validation"
OUT    = GALOSH / "benchmark" / "sidd_validation" / "galosh_raw_gpu_g"
OUT.mkdir(parents=True, exist_ok=True)
TMP    = GALOSH / "benchmark" / "results" / "speed" / "_g_sidd_tmp"
TMP.mkdir(parents=True, exist_ok=True)
GPU_EXE = GALOSH / "standalone" / "galosh_raw_gpu.exe"
CPU_EXE = GALOSH / "standalone" / "galosh_raw_cpu.exe"


def psnr(ref, test):
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))

def ssim_raw(ref, test):
    return float(ssim_fn(ref, test, data_range=1.0))


def run_g(exe, noisy_raw, w, h, uid, gpu):
    in_p = TMP / f"{uid}_in.bin"; out_p = TMP / f"{uid}_out.bin"
    noisy_raw.astype(np.float32).tofile(str(in_p))
    if gpu:   # in out W H strength luma chroma alpha sigma [dev] --variant=g
        cmd = [str(exe), str(in_p), str(out_p), str(w), str(h),
               "1.0", "1.0", "1.0", "0", "0", "0", "--variant=g"]
    else:     # in out W H mode strength luma chroma alpha sigma --variant=g
        cmd = [str(exe), str(in_p), str(out_p), str(w), str(h),
               "galosh", "1.0", "1.0", "1.0", "0", "0", "--variant=g"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True)
        return None, 0.0
    dt = time.time() - t0
    in_p.unlink(missing_ok=True)
    if r.returncode != 0 or not out_p.exists():
        sys.stderr.write(r.stderr.decode("utf-8", "replace")[-400:] + "\n")
        return None, dt
    den = np.fromfile(str(out_p), dtype=np.float32).reshape(h, w)
    out_p.unlink(missing_ok=True)
    return den, dt


def main():
    # engine: "auto" (verify CPU==GPU then GPU), "cpu" (force CPU g), "gpu" (force GPU g)
    engine = sys.argv[1] if len(sys.argv) > 1 else "auto"
    N_RAW = loadmat(str(VAL / "ValidationNoisyBlocksRaw.mat"))["ValidationNoisyBlocksRaw"]
    G_RAW = loadmat(str(VAL / "ValidationGtBlocksRaw.mat"))["ValidationGtBlocksRaw"]
    ns, npp = N_RAW.shape[:2]
    print(f"SIDD val RAW: {N_RAW.shape} dtype={N_RAW.dtype}  engine={engine}  (data_range=1.0)")

    def get(s, p):
        nr = np.ascontiguousarray(N_RAW[s, p]).astype(np.float32)
        gr = np.ascontiguousarray(G_RAW[s, p]).astype(np.float32)
        return nr, gr

    # ---- Phase 1: VERIFY CPU g == GPU g (auto mode only) -> pick engine ----
    use_gpu = (engine == "gpu")
    if engine == "auto":
        VN = 24
        print(f"\n=== VERIFY CPU g vs GPU g (n={VN}) ===")
        agree = []; cpu_ps = []; gpu_ps = []
        import random
        idxs = random.Random(0).sample(range(ns * npp), VN)
        for gi in idxs:
            s, p = gi // npp, gi % npp
            nr, gr = get(s, p); h, w = nr.shape
            dc, _ = run_g(CPU_EXE, nr, w, h, f"c{gi}", gpu=False)
            dg, _ = run_g(GPU_EXE, nr, w, h, f"g{gi}", gpu=True)
            if dc is None or dg is None:
                print(f"  s{s:02d}p{p:02d}: FAIL cpu={dc is not None} gpu={dg is not None}"); continue
            a = psnr(dc, dg); agree.append(a)
            cps, gps = psnr(gr, dc), psnr(gr, dg); cpu_ps.append(cps); gpu_ps.append(gps)
            print(f"  s{s:02d}p{p:02d}: PSNR(cpu,gpu)={a:6.2f}dB | vs_gt cpu={cps:.2f} gpu={gps:.2f}")
        if not agree:
            print("VERIFY failed: no successful pairs"); return 2
        mn = float(np.min(agree))
        print(f"\nagreement PSNR(cpu_g,gpu_g): mean={np.mean(agree):.2f} min={mn:.2f} dB | "
              f"vs_gt cpu={np.mean(cpu_ps):.3f} gpu={np.mean(gpu_ps):.3f}")
        if mn < 45.0:
            print(f"!! CPU g and GPU g DIFFER (min {mn:.1f} dB < 45) — using CPU g (correct) for full run.")
            use_gpu = False
        else:
            print(f"OK: CPU g ~= GPU g (min {mn:.1f} dB) -> FAST GPU g.")
            use_gpu = True

    # ---- Phase 2: FULL run on chosen engine ----
    eng = "GPU" if use_gpu else "CPU"
    exe = GPU_EXE if use_gpu else CPU_EXE
    print(f"\n=== FULL {eng} g on {ns*npp} patches ===")
    pss = []; sss = []; t0 = time.time(); fails = 0
    for s in range(ns):
        for p in range(npp):
            tag = f"s{s:02d}_p{p:02d}"; out_npy = OUT / f"{eng.lower()}_{tag}.npy"
            nr, gr = get(s, p); h, w = nr.shape
            if out_npy.exists():
                den = np.load(out_npy)
            else:
                den, _ = run_g(exe, nr, w, h, f"f{s}_{p}", gpu=use_gpu)
                if den is None: fails += 1; continue
                np.save(out_npy, den.astype(np.float32))
            pss.append(psnr(gr, den)); sss.append(ssim_raw(gr, den))
        done = (s + 1) * npp
        if (s + 1) % 5 == 0 or s == ns - 1:
            el = time.time() - t0
            print(f"  [{done}/{ns*npp}] mean PSNR={np.mean(pss):.3f} SSIM={np.mean(sss):.4f} "
                  f"elapsed={el/60:.1f}min fails={fails}")

    print("\n" + "=" * 56)
    print(f"GALOSH_RAW_G (variant g, {eng} FP32) — SIDD val {len(pss)} patches")
    print(f"  PSNR = {np.mean(pss):.3f} dB     SSIM = {np.mean(sss):.4f}")
    print(f"  (canonical o:  PSNR 49.47  SSIM 0.9865)")
    print(f"  delta vs o:    PSNR {np.mean(pss)-49.47:+.3f}  SSIM {np.mean(sss)-0.9865:+.4f}")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())
