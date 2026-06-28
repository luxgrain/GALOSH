"""Fractional-bit sweep: build HP probe at FXP_FRAC = {20,24,26,28,30,36},
measure clean-gap patches + a high-ISO validator.

Finds the MINIMUM global fractional bits that closes the Q11.20 gap.
FRAC=20 should reproduce the current INT (Q11.20) gap (sanity).
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
RAWNIND = Path(r"E:\rawnind_bench")
WORK = GALOSH / "benchmark" / "results" / "int_gap_by_iso"

FRACS = [20, 24, 26, 28, 30, 36]
PATCHES = [
    ("MuseeL-ram-A7C__ISO64000", "hiISO"),
    ("couch__ISO100",            "clean-ok"),
    ("D60-3__ISO400",            "clean-gap"),
    ("7D-7__ISO400",             "clean-gap"),
    ("sewingmachine__ISO200",    "catastrophe"),
    ("foodstuff2__ISO400",       "chroma"),
]


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def build(frac):
    exe = STD / f"galosh_hp_f{frac}.exe"
    cmd = ["gcc", "-O2", "-std=c11", f"-DFXP_FRAC={frac}",
           "-o", str(exe), str(STD / "galosh_raw_cpu_int_hp.c"), "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  build FRAC={frac} FAILED:\n{r.stderr[:500]}")
        return None
    return exe


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


# preload patches + FP32 reference
data = {}
for tag, role in PATCHES:
    scene = tag.rsplit("__ISO", 1)[0]
    noisy = np.load(RAWNIND / "__noisy_raw__" / f"{tag}.npy").astype(np.float32)
    gt = np.load(RAWNIND / "__gt_raw__" / f"{scene}.npy").astype(np.float32)
    h, w = noisy.shape
    ofp = run(EXE_FP, noisy, w, h, f"fp_{tag}")
    data[tag] = (noisy, gt, h, w, psnr(gt, ofp) if ofp is not None else float('nan'), role)

# header
hdr = f"{'FRAC':>5} " + " ".join(f"{tag.split('__')[0][:10]:>11}" for tag, _ in PATCHES)
print("PSNR by FXP_FRAC (FP32 ref in first row):")
print(f"{'FP32':>5} " + " ".join(f"{data[tag][4]:>11.2f}" for tag, _ in PATCHES))
print("-" * len(hdr))
print(hdr)
print("-" * len(hdr))
for frac in FRACS:
    exe = build(frac)
    if exe is None: continue
    row = [f"{frac:>5}"]
    for tag, _ in PATCHES:
        noisy, gt, h, w, pf, role = data[tag]
        o = run(exe, noisy, w, h, f"f{frac}_{tag}")
        row.append(f"{psnr(gt, o):>11.2f}" if o is not None else f"{'FAIL':>11}")
    print(" ".join(row))
    try: exe.unlink()
    except Exception: pass
print("-" * len(hdr))
print("(FRAC=20 ~ current Q11.20 INT; find smallest FRAC matching FP32 row)")
