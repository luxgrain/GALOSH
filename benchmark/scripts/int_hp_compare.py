"""HP precision probe: FP32 vs current INT (Q11.20) vs HP INT (Q27.36 int64).

Validation logic:
  - HIGH-ISO patches: HP must ~ current INT ~ FP32 (sanity; if HP is garbage
    here it has an arithmetic bug -> discard results).
  - CLEAN low-ISO patches (the actual test):
      HP -> FP32  => the gap IS Q11.20 working precision (solvable via bits)
      HP ~ INT    => the gap is discrete/algorithmic (more bits won't fix)
"""
import os, sys, subprocess
from pathlib import Path
import numpy as np

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

GALOSH = Path(r"C:\Users\luxgrain\GALOSH")
EXE_FP = GALOSH / "standalone" / "galosh_raw_cpu.exe"
EXE_INT = GALOSH / "standalone" / "galosh_raw_cpu_int.exe"
EXE_HP = GALOSH / "standalone" / "galosh_raw_cpu_int_hp.exe"
RAWNIND = Path(r"E:\rawnind_bench")
WORK = GALOSH / "benchmark" / "results" / "int_gap_by_iso"
WORK.mkdir(parents=True, exist_ok=True)

# (tag, role)
PATCHES = [
    ("MuseeL-ram-A7C__ISO64000", "hiISO-validate"),
    ("fan__ISO32000",            "hiISO-validate"),
    ("dustpan__ISO51200",        "hiISO-validate"),
    ("couch__ISO100",            "clean-fine"),
    ("D60-3__ISO400",            "clean-gap"),
    ("7D-6__ISO200",             "clean-gap"),
    ("7D-7__ISO400",             "clean-gap"),
    ("sewingmachine__ISO200",    "clean-catastrophe"),
    ("foodstuff2__ISO400",       "clean-chroma"),
]


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def run(exe, noisy, w, h, uid):
    in_p = WORK / f"_{uid}_in.bin"; out_p = WORK / f"_{uid}_out.bin"
    noisy.astype(np.float32).tofile(str(in_p))
    cmd = [str(exe), str(in_p), str(out_p), str(w), str(h),
           "galosh", "1.0", "1.0", "1.0", "0", "0", "--variant=r32"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=400)
    except subprocess.TimeoutExpired:
        in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True); return None
    in_p.unlink(missing_ok=True)
    if r.returncode != 0 or not out_p.exists():
        out_p.unlink(missing_ok=True); return None
    out = np.fromfile(str(out_p), dtype=np.float32).reshape(h, w)
    out_p.unlink(missing_ok=True)
    return out


print(f"{'tag':<28}{'role':<18}{'FP32':>7}{'INT':>7}{'HP':>8}{'HP-INT':>8}{'HP-FP':>8}")
print("-" * 84)
for tag, role in PATCHES:
    scene = tag.rsplit("__ISO", 1)[0]
    np_p = RAWNIND / "__noisy_raw__" / f"{tag}.npy"
    gt_p = RAWNIND / "__gt_raw__" / f"{scene}.npy"
    if not np_p.exists() or not gt_p.exists():
        print(f"{tag:<28}{role:<18}(missing)"); continue
    noisy = np.load(np_p).astype(np.float32); gt = np.load(gt_p).astype(np.float32)
    h, w = noisy.shape
    ofp = run(EXE_FP, noisy, w, h, f"fp_{tag}")
    oin = run(EXE_INT, noisy, w, h, f"in_{tag}")
    ohp = run(EXE_HP, noisy, w, h, f"hp_{tag}")
    pf = psnr(gt, ofp) if ofp is not None else float('nan')
    pi = psnr(gt, oin) if oin is not None else float('nan')
    ph = psnr(gt, ohp) if ohp is not None else float('nan')
    print(f"{tag[:28]:<28}{role:<18}{pf:>7.2f}{pi:>7.2f}{ph:>8.2f}{ph-pi:>+8.2f}{ph-pf:>+8.2f}")
