#!/usr/bin/env bash
# ============================================================================
# GALOSH bench-small — a fixed-seed synthetic Poisson-Gaussian pair (256^2)
# through the canonical RAW (FP32 + INT32) and YUV/RGB CPU paths. Records
# runtime + PSNR-vs-clean before/after into tests/bench_small_results.csv
# (and .json), and FAILS unless every method improves PSNR over the noisy
# input. Reproducible (seeded), dataset-free.               (Apache-2.0)
# ============================================================================
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
T=tests/_smoke_tmp; mkdir -p "$T"
CSV=tests/bench_small_results.csv
JSON=tests/bench_small_results.json

"$PY" - <<'PYEOF'
import json, subprocess, time
import numpy as np

rng = np.random.default_rng(2026)
W = H = 256
yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
clean = (0.15 + 0.6 * (xx + yy) / 2 + 0.15 * np.sin(14 * xx) * np.sin(11 * yy)).astype(np.float32)
alpha, sig = 0.01, 0.02
noisy = np.clip(clean + rng.normal(0, 1, clean.shape) *
                np.sqrt(alpha * clean + sig * sig), 0, 1).astype(np.float32)
clean3 = np.stack([clean, np.roll(clean, 40, 0), clean[::-1]], 2).astype(np.float32)
noisy3 = np.clip(clean3 + rng.normal(0, 1, clean3.shape) *
                 np.sqrt(alpha * clean3 + sig * sig), 0, 1).astype(np.float32)

def psnr(a, b):
    return float(-10 * np.log10(np.mean((a - b) ** 2) + 1e-12))

def run(cmd, op, ref, ch):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    dt = time.perf_counter() - t0
    assert r.returncode == 0, r.stderr.decode()[-200:]
    out = np.fromfile(op, dtype=np.float32).reshape(ref.shape)
    return psnr(out, ref), dt

noisy.tofile("tests/_smoke_tmp/bs_raw_in.bin")
noisy3.tofile("tests/_smoke_tmp/bs_yuv_in.bin")
rows = []
base_raw = psnr(noisy, clean)
base_yuv = psnr(noisy3, clean3)
rows.append({"method": "noisy_raw", "psnr": round(base_raw, 2), "time_s": 0.0})
rows.append({"method": "noisy_yuv", "psnr": round(base_yuv, 2), "time_s": 0.0})
for name, cmd, op, ref in [
    ("galosh_raw_fp32", ["./galosh_raw_cpu.exe", "tests/_smoke_tmp/bs_raw_in.bin",
                         "tests/_smoke_tmp/bs_raw_fp32.bin", "256", "256",
                         "galosh", "1.0", "1.0", "1.0", "0", "0"],
     "tests/_smoke_tmp/bs_raw_fp32.bin", clean),
    ("galosh_raw_int32", ["./galosh_raw_cpu_int.exe", "tests/_smoke_tmp/bs_raw_in.bin",
                          "tests/_smoke_tmp/bs_raw_r32.bin", "256", "256",
                          "galosh", "1.0", "1.0", "1.0", "0", "0"],
     "tests/_smoke_tmp/bs_raw_r32.bin", clean),
    ("galosh_yuv_fp32", ["./galosh_yuv_cpu.exe", "tests/_smoke_tmp/bs_yuv_in.bin",
                         "tests/_smoke_tmp/bs_yuv.bin", "256", "256", "1.0", "1.0"],
     "tests/_smoke_tmp/bs_yuv.bin", clean3),
]:
    base = base_raw if ref.ndim == 2 else base_yuv
    p, dt = run(cmd, op, ref, 1)
    ok = p > base
    rows.append({"method": name, "psnr": round(p, 2), "time_s": round(dt, 3),
                 "gain_db": round(p - base, 2), "improves": ok})
    print(f"  {name:<18} PSNR {p:6.2f} dB  (noisy {base:.2f}, gain {p-base:+.2f})  {dt:.2f}s"
          f"  {'OK' if ok else 'FAIL'}")

with open("tests/bench_small_results.csv", "w") as f:
    keys = ["method", "psnr", "gain_db", "time_s", "improves"]
    f.write(",".join(keys) + "\n")
    for r in rows:
        f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
json.dump(rows, open("tests/bench_small_results.json", "w"), indent=1)
bad = [r for r in rows if r.get("improves") is False]
print("BENCH-SMALL:", "FAIL " + str([r["method"] for r in bad]) if bad else "PASS")
raise SystemExit(1 if bad else 0)
PYEOF
rc=$?
echo "results: $CSV / $JSON"
exit $rc
