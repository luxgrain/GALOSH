"""Saturation map: per phase-boundary buffer, max |real value| for
HP (int64, true unsaturated values) vs INT (int32 Q11.20, saturates at +-2048).

A buffer where HP_max > 2048 AND INT_max ~ 2048 (capped) is a saturation
victim in the shipping r32 build -> candidate for a wider Q-format (Q15.16).

Both exes emit the same named phase buffers as float32 .bin (real units) via
GALOSH_INT_DUMP_DIR.
"""
import os, sys, subprocess, shutil
from pathlib import Path
import numpy as np

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

GALOSH = Path(r"C:\Users\luxgrain\GALOSH")
EXE_INT = GALOSH / "standalone" / "galosh_raw_cpu_int.exe"
EXE_HP = GALOSH / "standalone" / "galosh_raw_cpu_int_hp.exe"
RAWNIND = Path(r"E:\rawnind_bench")
WORK = GALOSH / "benchmark" / "results" / "int_saturation_map"
WORK.mkdir(parents=True, exist_ok=True)

PHASES = ["p3_L_cs", "p5_pilot", "p5_L_cs_den", "p6_L_pixel", "p6_L_h_den",
          "p7_C1_q_up", "p8_C1_h_den", "p9_C1_aligned", "p10_output"]
PATCHES = ["D60-3__ISO400", "foodstuff2__ISO400", "sewingmachine__ISO200",
           "7D-7__ISO400", "couch__ISO100", "MuseeL-ram-A7C__ISO64000"]
Q11_20_MAX = 2048.0


def run_dump(exe, noisy, w, h, dumpdir, uid):
    if dumpdir.exists(): shutil.rmtree(dumpdir)
    dumpdir.mkdir(parents=True)
    in_p = WORK / f"_{uid}_in.bin"; out_p = WORK / f"_{uid}_out.bin"
    noisy.astype(np.float32).tofile(str(in_p))
    env = dict(os.environ); env["GALOSH_INT_DUMP_DIR"] = str(dumpdir)
    cmd = [str(exe), str(in_p), str(out_p), str(w), str(h),
           "galosh", "1.0", "1.0", "1.0", "0", "0", "--variant=r32"]
    try: subprocess.run(cmd, capture_output=True, timeout=400, env=env)
    except subprocess.TimeoutExpired: pass
    in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True)


def maxabs(dumpdir, name):
    p = dumpdir / f"{name}.bin"
    if not p.exists(): return None
    a = np.fromfile(str(p), dtype=np.float32)
    if a.size == 0: return None
    return float(np.max(np.abs(a)))


# aggregate max over all patches, per phase, for HP and INT
agg_hp = {ph: 0.0 for ph in PHASES}
agg_int = {ph: 0.0 for ph in PHASES}
per_patch = {}
for tag in PATCHES:
    noisy = np.load(RAWNIND / "__noisy_raw__" / f"{tag}.npy").astype(np.float32)
    h, w = noisy.shape
    dd_hp = WORK / "dump_hp"; dd_int = WORK / "dump_int"
    run_dump(EXE_HP, noisy, w, h, dd_hp, f"hp_{tag}")
    run_dump(EXE_INT, noisy, w, h, dd_int, f"int_{tag}")
    row = {}
    for ph in PHASES:
        mh = maxabs(dd_hp, ph); mi = maxabs(dd_int, ph)
        row[ph] = (mh, mi)
        if mh is not None: agg_hp[ph] = max(agg_hp[ph], mh)
        if mi is not None: agg_int[ph] = max(agg_int[ph], mi)
    per_patch[tag] = row
    shutil.rmtree(dd_hp, ignore_errors=True); shutil.rmtree(dd_int, ignore_errors=True)

print("=== max |real value| per phase buffer  (HP=int64 true / INT=int32 saturating) ===")
print(f"Q11.20 integer range = ±{Q11_20_MAX:.0f}.  HP>2048 & INT capped => saturation victim.\n")
print(f"{'buffer':<16}{'HP max':>12}{'INT max':>12}   verdict")
print("-" * 60)
for ph in PHASES:
    mh, mi = agg_hp[ph], agg_int[ph]
    if mh > Q11_20_MAX and mi <= Q11_20_MAX * 1.02:
        verdict = f"SATURATES (HP {mh:.0f} > 2048, INT capped)"
    elif mh > Q11_20_MAX:
        verdict = f"HP>2048 (INT {mi:.0f} also high?)"
    else:
        verdict = "ok (< 2048)"
    print(f"{ph:<16}{mh:>12.1f}{mi:>12.1f}   {verdict}")

print("\n=== per-patch HP max (to see which content drives saturation) ===")
print(f"{'buffer':<16}" + "".join(f"{t.split('__')[0][:9]:>10}" for t in PATCHES))
for ph in PHASES:
    print(f"{ph:<16}" + "".join(
        f"{(per_patch[t][ph][0] or 0):>10.0f}" for t in PATCHES))
