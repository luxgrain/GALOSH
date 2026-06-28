"""Decisive test: is the INT low-ISO gap from Phase 0 WLS accuracy or downstream?

For each patch:
  1. FP32 (-o): target PSNR + its estimated alpha/sigma_sq (parsed from stderr)
  2. INT blind (-r32): current INT, its own Phase 0 estimate
  3. INT override (-r32 with FP32's alpha/sigma_sq injected via CLI args 9,10)

If (3) ~ (1): the gap is the INT Phase 0 WLS estimate (alpha/sigma_sq wrong),
              NOT representation (FP32's values are Q11.20-representable) and
              NOT downstream arithmetic -> SOLVABLE via better WLS precision.
If (3) ~ (2): better alpha/sigma_sq doesn't help -> gap is downstream GAT/chroma.
"""
import os, sys, subprocess, re
from pathlib import Path
import numpy as np

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

GALOSH = Path(os.path.expanduser(r"~\GALOSH"))
EXE_FP = GALOSH / "standalone" / "galosh_raw_cpu.exe"
EXE_INT = GALOSH / "standalone" / "galosh_raw_cpu_int.exe"
RAWNIND = Path(r"E:\rawnind_bench")
WORK = GALOSH / "benchmark" / "results" / "int_gap_by_iso"
WORK.mkdir(parents=True, exist_ok=True)

TAGS = sys.argv[1:] if len(sys.argv) > 1 else [
    "sewingmachine__ISO200", "7D-7__ISO400", "7D-6__ISO200",
    "D60-3__ISO400", "MuseeL-AraBleu-A7C__ISO250", "foodstuff2__ISO400",
    "couch__ISO100",
]

FP_AS = re.compile(r"\[GALOSH_RAW_O\]\s*alpha=([0-9.eE+-]+)\s*sigma_sq=([0-9.eE+-]+)")


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def run(exe, noisy, w, h, variant, uid, alpha=None, sigma=None):
    in_p = WORK / f"_{uid}_in.bin"; out_p = WORK / f"_{uid}_out.bin"
    noisy.astype(np.float32).tofile(str(in_p))
    cmd = [str(exe), str(in_p), str(out_p), str(w), str(h),
           "galosh", "1.0", "1.0", "1.0",
           ("%.10g" % alpha) if alpha else "0",
           ("%.12g" % sigma) if sigma is not None else "0",
           f"--variant={variant}"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True); return None, ""
    in_p.unlink(missing_ok=True)
    err = r.stderr.decode("utf-8", errors="replace") if r.stderr else ""
    if r.returncode != 0 or not out_p.exists():
        out_p.unlink(missing_ok=True); return None, err
    out = np.fromfile(str(out_p), dtype=np.float32).reshape(h, w)
    out_p.unlink(missing_ok=True)
    return out, err


print(f"{'tag':<30} {'ISO':>6} {'FP32':>7} {'INTbl':>7} {'INTov':>7} {'ov-bl':>7}  {'verdict':<18}")
print("-" * 92)
for tag in TAGS:
    scene = tag.rsplit("__ISO", 1)[0]
    np_p = RAWNIND / "__noisy_raw__" / f"{tag}.npy"
    gt_p = RAWNIND / "__gt_raw__" / f"{scene}.npy"
    if not np_p.exists() or not gt_p.exists():
        print(f"{tag:<30} (missing)"); continue
    iso = tag.rsplit("__ISO", 1)[1]
    noisy = np.load(np_p).astype(np.float32); gt = np.load(gt_p).astype(np.float32)
    h, w = noisy.shape

    ofp, efp = run(EXE_FP, noisy, w, h, "o", f"fp_{tag}")
    m = FP_AS.search(efp)
    if not m:
        print(f"{tag:<30} (no FP32 alpha/sigma parsed)"); continue
    fa, fs = float(m.group(1)), float(m.group(2))

    obl, _ = run(EXE_INT, noisy, w, h, "r32", f"bl_{tag}")
    oov, _ = run(EXE_INT, noisy, w, h, "r32", f"ov_{tag}", alpha=fa, sigma=fs)

    pfp = psnr(gt, ofp) if ofp is not None else float("nan")
    pbl = psnr(gt, obl) if obl is not None else float("nan")
    pov = psnr(gt, oov) if oov is not None else float("nan")
    d = pov - pbl
    if pov >= pfp - 0.5:
        verdict = "WLS-accuracy (FIX)"
    elif d > 0.5:
        verdict = "partial WLS"
    else:
        verdict = "downstream (not WLS)"
    print(f"{tag[:30]:<30} {iso:>6} {pfp:>7.2f} {pbl:>7.2f} {pov:>7.2f} {d:>+7.2f}  {verdict:<18}  (FPa={fa:.2e} s2={fs:.2e})")
