"""Per-variable Q-format design — Phase A: value-range survey.

Runs the HP probe (int64 = true unsaturated values) on a diverse RawNIND patch
set (super-clean low-ISO .. heavy-noise high-ISO, dark .. bright) and reports,
per phase-boundary buffer, the max |real value| across the set.  This gives the
INTEGER-bit requirement per buffer for the per-variable Q design (the basis for
which buffers fit INT16 and which need wider int / scaling).

Transient intermediates (WHT coefs, lambda, WLS sums, etc.) are surveyed
separately via HP_MAXTRACK instrumentation.
"""
import os, sys, subprocess, shutil, math
from pathlib import Path
import numpy as np

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

GALOSH = Path(os.path.expanduser(r"~\GALOSH"))
EXE_HP = GALOSH / "standalone" / "galosh_raw_cpu_int_hp.exe"
RAWNIND = Path(os.environ.get("GALOSH_RAWNIND_BENCH", "benchmark/datasets/rawnind_bench"))
WORK = GALOSH / "benchmark" / "results" / "perq_range_survey"
WORK.mkdir(parents=True, exist_ok=True)

# diverse set: clean/dark .. noisy/bright, low..high ISO
PATCHES = [
    "sewingmachine__ISO200", "D60-3__ISO400", "couch__ISO100", "foodstuff2__ISO400",
    "7D-7__ISO400", "7D-6__ISO200", "MuseeL-skull-A7C__ISO1250",
    "MuseeL-ram-A7C__ISO64000", "fan__ISO32000", "dustpan__ISO51200",
    "big_bookshelf_1__ISO51200", "logpile_closeup__ISO65535",
]
# all phase-boundary dumps the HP probe emits (real units)
BUFS = ["p3_L_cs", "p5_pilot", "p5_L_cs_den", "p6_L_pixel", "p6_L_h_den",
        "p7_C1_q_up", "p8_C1_h_den", "p9_C1_aligned", "p10_output"]


def run_dump(noisy, w, h, dd, uid):
    if dd.exists(): shutil.rmtree(dd)
    dd.mkdir(parents=True)
    in_p = WORK / f"_{uid}_in.bin"; out_p = WORK / f"_{uid}_out.bin"
    noisy.astype(np.float32).tofile(str(in_p))
    env = dict(os.environ); env["GALOSH_INT_DUMP_DIR"] = str(dd)
    cmd = [str(EXE_HP), str(in_p), str(out_p), str(w), str(h),
           "galosh", "1.0", "1.0", "1.0", "0", "0", "--variant=r32"]
    try: subprocess.run(cmd, capture_output=True, timeout=400, env=env)
    except subprocess.TimeoutExpired: pass
    in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True)


def maxabs(dd, name):
    p = dd / f"{name}.bin"
    if not p.exists(): return None
    a = np.fromfile(str(p), dtype=np.float32)
    return float(np.max(np.abs(a))) if a.size else None


agg = {b: 0.0 for b in BUFS}
per = {}
for tag in PATCHES:
    np_p = RAWNIND / "__noisy_raw__" / f"{tag}.npy"
    if not np_p.exists():
        print(f"  (missing {tag})"); continue
    noisy = np.load(np_p).astype(np.float32); h, w = noisy.shape
    dd = WORK / "dd"
    run_dump(noisy, w, h, dd, tag)
    row = {}
    for b in BUFS:
        m = maxabs(dd, b)
        row[b] = m
        if m is not None: agg[b] = max(agg[b], m)
    per[tag] = row
    shutil.rmtree(dd, ignore_errors=True)


def intbits(v):
    """signed integer bits needed to hold ±v (excl sign): ceil(log2(v))."""
    if v is None or v <= 0: return 0
    return max(0, int(math.ceil(math.log2(max(v, 1e-30)))))


print("=== Phase A: phase-boundary value ranges (HP int64 true values) ===\n")
print(f"{'buffer':<16}{'max|val|':>12}{'int bits':>10}  {'INT32 frac':>11}  {'INT16 frac':>11}")
print("-" * 64)
for b in BUFS:
    v = agg[b]; ib = intbits(v)
    f32 = 31 - ib            # frac bits available in INT32 (1 sign)
    f16 = 15 - ib            # frac bits available in INT16 (1 sign)
    f16s = f"{f16}" if f16 >= 0 else "OVERFLOW"
    print(f"{b:<16}{v:>12.2f}{ib:>10}  {f32:>11}  {f16s:>11}")

print("\n=== per-patch max (which content drives each buffer's range) ===")
print(f"{'buffer':<16}" + "".join(f"{t.split('__')[0][:8]:>9}" for t in PATCHES))
for b in BUFS:
    print(f"{b:<16}" + "".join(f"{(per.get(t,{}).get(b) or 0):>9.0f}" for t in PATCHES))
print("\n(int bits = ceil(log2 max); INT16 frac = 15 - int bits; <0 => needs INT32 or scaling)")
