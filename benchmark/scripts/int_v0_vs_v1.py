"""Run archived v0 (canonical) vs v1 (bug11 scaled-sigma2 attempt) on key patches.

Diagnostic: did the v1 scaled-sigma^2 approach actually FIX the catastrophic
low-ISO patches (proving the direction is right), and what exactly did it break?

Patches:
  - catastrophic low-ISO (foodstuff2, sewingmachine, 7D-7): does v1 fix them?
  - canary D60-3_ISO400: v1 broke this to 21.01 per NOTES; confirm
  - a few SIDD-like mid/high already-good ones for regression check
"""
import os, sys, subprocess
from pathlib import Path
import numpy as np

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

GALOSH = Path(os.path.expanduser(r"~\GALOSH"))
ARCH = GALOSH / "standalone" / "archive_int_iter"
EXE_V0 = ARCH / "galosh_raw_cpu_int_v0_baseline.exe"
EXE_V1 = ARCH / "galosh_raw_cpu_int_v1_bug11_broken.exe"
EXE_FP = GALOSH / "standalone" / "galosh_raw_cpu.exe"
RAWNIND = Path(os.environ.get("GALOSH_RAWNIND_BENCH", "benchmark/datasets/rawnind_bench"))
WORK = GALOSH / "benchmark" / "results" / "int_gap_by_iso"
WORK.mkdir(parents=True, exist_ok=True)

TAGS = sys.argv[1:] if len(sys.argv) > 1 else [
    "foodstuff2__ISO400", "sewingmachine__ISO200", "7D-7__ISO400",
    "MuseeL-AraBleu-A7C__ISO250", "7D-6__ISO200",   # catastrophic low-ISO
    "D60-3__ISO400",                                  # the canary
    "couch__ISO100",                                  # already-fine low-ISO
    "MuseeL-ram-A7C__ISO64000", "fan__ISO32000",      # high-ISO (must not regress)
]


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def run(exe, noisy, w, h, variant, uid):
    in_p = WORK / f"_{uid}_in.bin"; out_p = WORK / f"_{uid}_out.bin"
    noisy.astype(np.float32).tofile(str(in_p))
    cmd = [str(exe), str(in_p), str(out_p), str(w), str(h),
           "galosh", "1.0", "1.0", "1.0", "0", "0", f"--variant={variant}"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True); return None
    in_p.unlink(missing_ok=True)
    if r.returncode != 0 or not out_p.exists():
        out_p.unlink(missing_ok=True); return None
    out = np.fromfile(str(out_p), dtype=np.float32).reshape(h, w)
    out_p.unlink(missing_ok=True)
    return out


print(f"{'tag':<34} {'FP32':>7} {'v0':>7} {'v1':>7} {'v1-v0':>8}  verdict")
print("-" * 78)
for tag in TAGS:
    scene = tag.rsplit("__ISO", 1)[0]
    np_p = RAWNIND / "__noisy_raw__" / f"{tag}.npy"
    gt_p = RAWNIND / "__gt_raw__" / f"{scene}.npy"
    if not np_p.exists() or not gt_p.exists():
        print(f"{tag:<34} (missing)"); continue
    noisy = np.load(np_p).astype(np.float32); gt = np.load(gt_p).astype(np.float32)
    h, w = noisy.shape
    ofp = run(EXE_FP, noisy, w, h, "o", f"fp_{tag}")
    o0 = run(EXE_V0, noisy, w, h, "r32", f"v0_{tag}")
    o1 = run(EXE_V1, noisy, w, h, "r32", f"v1_{tag}")
    pfp = psnr(gt, ofp) if ofp is not None else float("nan")
    p0 = psnr(gt, o0) if o0 is not None else float("nan")
    p1 = psnr(gt, o1) if o1 is not None else float("nan")
    d = p1 - p0
    verdict = ("FIX" if d > 0.5 else ("BREAK" if d < -0.5 else "same"))
    print(f"{tag:<34} {pfp:>7.2f} {p0:>7.2f} {p1:>7.2f} {d:>+8.2f}  {verdict}")
