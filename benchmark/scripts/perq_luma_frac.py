"""Per-variable Q design — Phase B: luma line-buffer precision requirement.

Builds the HP probe with the luma line buffers (L_cs, pilot, L_den, L_pixel,
L_h_den) rounded to N fractional bits (= INT16 Q(int).N data path), compute
stays int64.  Sweep N; the N where PSNR converges to HP-full = the luma frac
bits the data path actually needs.  INT16 luma allows ~6 frac (9 int bits).
"""
import os, sys, subprocess
from pathlib import Path
import numpy as np

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

GALOSH = Path(os.path.expanduser(r"~\GALOSH")); STD = GALOSH / "standalone"
SRC = str(STD / "galosh_raw_cpu_int_hp.c")
EXE_FP = STD / "galosh_raw_cpu.exe"
RAWNIND = Path(os.environ.get("GALOSH_RAWNIND_BENCH", "benchmark/datasets/rawnind_bench"))
WORK = GALOSH / "benchmark" / "results" / "perq_range_survey"; WORK.mkdir(parents=True, exist_ok=True)

PATCHES = ["D60-3__ISO400", "sewingmachine__ISO200", "7D-7__ISO400",
           "foodstuff2__ISO400", "couch__ISO100", "MuseeL-ram-A7C__ISO64000"]
FRACS = [3, 4, 5, 6, 7]   # INT16 luma allows ~6 (int bits ~9); find the floor


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def build(defs, name):
    exe = STD / f"galosh_hp_{name}.exe"
    cmd = ["gcc", "-O2", "-std=c11"] + defs + ["-o", str(exe), SRC, "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  build {name} FAIL: {r.stderr[:300]}"); return None
    return exe


def run(exe, n, w, h, v):
    n.astype(np.float32).tofile(str(WORK / "_in.bin"))
    cmd = [str(exe), str(WORK / "_in.bin"), str(WORK / "_out.bin"), str(w), str(h),
           "galosh", "1.0", "1.0", "1.0", "0", "0", f"--variant={v}"]
    try: subprocess.run(cmd, capture_output=True, timeout=400)
    except subprocess.TimeoutExpired: return None
    return np.fromfile(str(WORK / "_out.bin"), dtype=np.float32).reshape(h, w)


data = {}
for t in PATCHES:
    sc = t.rsplit("__ISO", 1)[0]
    n = np.load(RAWNIND / "__noisy_raw__" / f"{t}.npy").astype(np.float32)
    g = np.load(RAWNIND / "__gt_raw__" / f"{sc}.npy").astype(np.float32)
    h, w = n.shape
    data[t] = (n, g, h, w, psnr(g, run(EXE_FP, n, w, h, "o")))

exe_full = build([], "lumafull")
cols = " ".join(f"{t.split('__')[0][:9]:>10}" for t in PATCHES)
print(f"{'variant':>16} {cols}")
print(f"{'FP32':>16} " + " ".join(f"{data[t][4]:>10.2f}" for t in PATCHES))
print(f"{'HP full(36)':>16} " + " ".join(
    f"{psnr(data[t][1], run(exe_full, data[t][0], data[t][3], data[t][2], 'r32')):>10.2f}" for t in PATCHES))
print("--- luma line buffer rounded to N frac bits (INT16 luma ~ N=6) ---")
for frac in FRACS:
    e = build([f"-DLUMA_FRAC={frac}"], f"luma{frac}")
    if not e: continue
    row = []
    for t in PATCHES:
        n, g, h, w, pf = data[t]
        row.append(f"{psnr(g, run(e, n, w, h, 'r32')):>10.2f}")
    print(f"{'luma frac='+str(frac):>16} " + " ".join(row))
    e.unlink(missing_ok=True)
exe_full.unlink(missing_ok=True)
print("\n(find smallest luma frac that matches HP full → the luma data-path frac requirement)")
