"""Compare Phase 0 alpha/sigma_sq estimates INT vs FP32 on given tags.

Root-cause probe: do the catastrophic low-ISO INT failures originate in
Phase 0 noise estimation (alpha/sigma_sq diverge) or downstream arithmetic?

Captures stderr from both exes, extracts the Phase 0 + GAT param lines.
"""
import os, sys, subprocess, re
from pathlib import Path
import numpy as np

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

GALOSH = Path(r"C:\Users\luxgrain\GALOSH")
EXE_FP = GALOSH / "standalone" / "galosh_raw_cpu.exe"
EXE_INT = GALOSH / "standalone" / "galosh_raw_cpu_int.exe"
RAWNIND = Path(r"E:\rawnind_bench")
WORK = GALOSH / "benchmark" / "results" / "int_gap_by_iso"
WORK.mkdir(parents=True, exist_ok=True)

TAGS = sys.argv[1:] if len(sys.argv) > 1 else [
    "foodstuff2__ISO400", "sewingmachine__ISO200", "7D-7__ISO400",
    "couch__ISO100", "MuseeL-ram-A7C__ISO64000",
]

PAT = re.compile(r"(Phase 0[^\n]*|GAT params[^\n]*|Foi LUT[^\n]*|alpha=[^\n]*)")


def run(exe, noisy, w, h, variant, uid):
    in_p = WORK / f"_{uid}_in.bin"; out_p = WORK / f"_{uid}_out.bin"
    noisy.astype(np.float32).tofile(str(in_p))
    cmd = [str(exe), str(in_p), str(out_p), str(w), str(h),
           "galosh", "1.0", "1.0", "1.0", "0", "0", f"--variant={variant}"]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True)
    err = r.stderr.decode("utf-8", errors="replace") if r.stderr else ""
    lines = [l.strip() for l in err.splitlines() if PAT.search(l)]
    return lines


for tag in TAGS:
    scene = tag.rsplit("__ISO", 1)[0]
    np_p = RAWNIND / "__noisy_raw__" / f"{tag}.npy"
    if not np_p.exists():
        print(f"[{tag}] missing\n"); continue
    noisy = np.load(np_p).astype(np.float32)
    h, w = noisy.shape
    print(f"=== {tag}  ({h}x{w})  mean={noisy.mean():.4f} max={noisy.max():.4f} ===")
    print("  -- FP32 --")
    for l in run(EXE_FP, noisy, w, h, "o", f"fp_{tag}"): print(f"    {l}")
    print("  -- INT r32 --")
    for l in run(EXE_INT, noisy, w, h, "r32", f"int_{tag}"): print(f"    {l}")
    print()
