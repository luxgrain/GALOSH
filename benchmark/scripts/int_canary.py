"""INT r32 canary test — D60-3_ISO400 single patch quick check.

Runs galosh_raw_cpu_int.exe --variant=r32 on the canary patch.
Expected PSNR: 42.30 dB (= v0/v12 canonical, per archive_int_iter/NOTES.md).
PSNR < 40 dB indicates a Phase 0 / inverse GAT regression (likely the
v14-v17 silent ×1024 wrapper trap reappearing).

Usage:
  python benchmark/scripts/int_canary.py
  python benchmark/scripts/int_canary.py --variant r16    (= future r16 test)
"""
import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
import numpy as np

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

GALOSH = Path(os.path.expanduser(r"~\GALOSH"))
EXE = GALOSH / "standalone" / "galosh_raw_cpu_int.exe"
BENCH = Path(os.environ.get("GALOSH_RAWNIND_BENCH", "benchmark/datasets/rawnind_bench"))
TAG = "D60-3__ISO400"
EXPECTED_PSNR = 42.73  # 2026-06-07: was 42.30 (v12); +chroma-recip +guided-filter 2-pass fixes
THRESHOLD = 40.0  # below = regression

TMP = Path(os.path.expanduser(r"~\GALOSH\benchmark\results\int_canary_tmp"))
TMP.mkdir(parents=True, exist_ok=True)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def run_int(noisy, variant):
    h, w = noisy.shape
    in_p = TMP / "canary_in.bin"
    out_p = TMP / "canary_out.bin"
    noisy.astype(np.float32).tofile(str(in_p))
    cmd = [str(EXE), str(in_p), str(out_p), str(w), str(h),
           "galosh", "1.0", "1.0", "1.0", "0", "0", f"--variant={variant}"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True)
        return None, time.time() - t0, "timeout", ""
    dt = time.time() - t0
    in_p.unlink(missing_ok=True)
    if r.returncode != 0 or not out_p.exists():
        out_p.unlink(missing_ok=True)
        err = r.stderr.decode("utf-8", errors="replace")[:300] if r.stderr else ""
        return None, dt, f"rc={r.returncode}", err
    out = np.fromfile(str(out_p), dtype=np.float32).reshape(h, w)
    out_p.unlink(missing_ok=True)
    return out, dt, "ok", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="r32", choices=["r32", "r16"])
    args = ap.parse_args()

    noisy_path = BENCH / "__noisy_raw__" / f"{TAG}.npy"
    gt_path = BENCH / "__gt_raw__" / f"{TAG.rsplit('__ISO', 1)[0]}.npy"
    if not noisy_path.exists() or not gt_path.exists():
        print(f"ERR: missing {noisy_path} or {gt_path}")
        sys.exit(2)

    noisy = np.load(noisy_path).astype(np.float32)
    gt = np.load(gt_path).astype(np.float32)

    print(f"=== INT {args.variant} canary: {TAG} ===")
    print(f"  shape: {noisy.shape}, expected PSNR: {EXPECTED_PSNR:.2f} dB (= v0/v12 canonical)")

    den, dt, status, err = run_int(noisy, args.variant)
    if den is None:
        print(f"  FAIL: status={status}")
        if err:
            print(f"  stderr: {err}")
        sys.exit(3)

    p = psnr(gt, den)
    delta = p - EXPECTED_PSNR
    print(f"  PSNR: {p:.3f} dB  (delta from canonical: {delta:+.3f})  time={dt:.2f}s")

    if p < THRESHOLD:
        print(f"\n  ❌ REGRESSION: PSNR {p:.2f} < threshold {THRESHOLD:.2f}")
        print(f"  Likely cause: GAT precompute wrapper bug or σ²/α floor break")
        print(f"  ACTION: revert recent changes, bisect against last good archive variant")
        sys.exit(1)
    elif p < EXPECTED_PSNR - 0.5:
        print(f"\n  ⚠  Mild regression: PSNR {p:.2f} below canonical {EXPECTED_PSNR:.2f} by > 0.5 dB")
        sys.exit(1)
    else:
        print(f"\n  ✓ canary passed (PSNR >= {EXPECTED_PSNR - 0.5:.2f})")
        sys.exit(0)


if __name__ == "__main__":
    main()
