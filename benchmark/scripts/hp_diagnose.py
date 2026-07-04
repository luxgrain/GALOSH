"""Diagnose the int32 Q11.20 gap: is it (A) fractional precision floor, or
(B) integer-range saturation at +-2048?

(A) Downward frac sweep: FRAC in {12,14,16,18,20} in int64 (range stays huge).
    If PSNR holds near FP32 down to low FRAC -> few frac bits suffice.
(B) Range-clamp test: FRAC=20 int64 but _sat128 clamped to +-{2048,8192,32768}.
    If clamp=2048 reproduces the current INT gap -> integer-range saturation
    is the cause (=> fix via more INT bits, e.g. Q16.16, stays 32-bit).
"""
import os, sys, subprocess
from pathlib import Path
import numpy as np

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

GALOSH = Path(os.path.expanduser(r"~\GALOSH"))
STD = GALOSH / "standalone"
EXE_FP = STD / "galosh_raw_cpu.exe"
EXE_INT = STD / "galosh_raw_cpu_int.exe"
RAWNIND = Path(os.environ.get("GALOSH_RAWNIND_BENCH", "benchmark/datasets/rawnind_bench"))
WORK = GALOSH / "benchmark" / "results" / "int_gap_by_iso"
SRC = str(STD / "galosh_raw_cpu_int_hp.c")

PATCHES = [
    ("MuseeL-ram-A7C__ISO64000", "hiISO"),
    ("D60-3__ISO400",            "clean-gap"),
    ("7D-7__ISO400",             "clean-gap"),
    ("sewingmachine__ISO200",    "catastrophe"),
    ("foodstuff2__ISO400",       "chroma"),
]


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def build(defs, name):
    exe = STD / f"galosh_hp_{name}.exe"
    cmd = ["gcc", "-O2", "-std=c11"] + defs + ["-o", str(exe), SRC, "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  build {name} FAILED: {r.stderr[:400]}"); return None
    return exe


def run(exe, noisy, w, h, uid):
    in_p = WORK / f"_{uid}_in.bin"; out_p = WORK / f"_{uid}_out.bin"
    noisy.astype(np.float32).tofile(str(in_p))
    cmd = [str(exe), str(in_p), str(out_p), str(w), str(h),
           "galosh", "1.0", "1.0", "1.0", "0", "0", "--variant=r32"]
    try: r = subprocess.run(cmd, capture_output=True, timeout=400)
    except subprocess.TimeoutExpired:
        in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True); return None
    in_p.unlink(missing_ok=True)
    if r.returncode != 0 or not out_p.exists():
        out_p.unlink(missing_ok=True); return None
    out = np.fromfile(str(out_p), dtype=np.float32).reshape(h, w)
    out_p.unlink(missing_ok=True)
    return out


data = {}
for tag, role in PATCHES:
    scene = tag.rsplit("__ISO", 1)[0]
    noisy = np.load(RAWNIND / "__noisy_raw__" / f"{tag}.npy").astype(np.float32)
    gt = np.load(RAWNIND / "__gt_raw__" / f"{scene}.npy").astype(np.float32)
    h, w = noisy.shape
    ofp = run(EXE_FP, noisy, w, h, f"fp_{tag}")
    oin = run(EXE_INT, noisy, w, h, f"in_{tag}")
    data[tag] = (noisy, gt, h, w, psnr(gt, ofp), psnr(gt, oin), role)

cols = " ".join(f"{t.split('__')[0][:10]:>11}" for t, _ in PATCHES)
def row(label, exe):
    cells = []
    for tag, _ in PATCHES:
        noisy, gt, h, w, pf, pi, role = data[tag]
        o = run(exe, noisy, w, h, f"{label}_{tag}")
        cells.append(f"{psnr(gt,o):>11.2f}" if o is not None else f"{'FAIL':>11}")
    print(f"{label:>16} " + " ".join(cells))

print(f"{'variant':>16} {cols}")
print("-" * 90)
print(f"{'FP32':>16} " + " ".join(f"{data[t][4]:>11.2f}" for t,_ in PATCHES))
print(f"{'INT int32 Q11.20':>16} " + " ".join(f"{data[t][5]:>11.2f}" for t,_ in PATCHES))
print("--- (A) downward frac sweep (int64, range huge) ---")
for frac in [12, 14, 16, 18, 20]:
    e = build([f"-DFXP_FRAC={frac}"], f"f{frac}")
    if e: row(f"FRAC={frac}", e); e.unlink(missing_ok=True)
print("--- (B) range-clamp test (FRAC=20 int64, clamp to +-N real) ---")
for clamp in [2048, 8192, 32768]:
    e = build(["-DFXP_FRAC=20", f"-DHP_CLAMP_REAL={clamp}"], f"clamp{clamp}")
    if e: row(f"clamp=±{clamp}", e); e.unlink(missing_ok=True)
print("-" * 90)
