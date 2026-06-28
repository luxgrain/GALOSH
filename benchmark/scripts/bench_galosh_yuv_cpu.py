"""GALOSH_YUV_O CPU bench.

SIDD Medium 80-pair benchmark for galosh_yuv_cpu.exe.

GALOSH_YUV_O is the production canonical pipeline (= former
"v5 robust-MAD" / "GALOSH_YUV_G", now finalised under the _O naming
mirror of GALOSH_RAW_O).  MAD-based BayesShrink and bilateral LOESS
chroma are unconditionally on; no flag toggles them any more.

Legacy mean-based BayesShrink (--robust-shrink=0) and the pre-G
separable mean-only chroma filter remain in archived bench dirs only.

Output PNGs land at:
    benchmark/sidd_medium/galosh_yuv_cpu_o/<tag>_den.png

Per-image pipeline:
    noisy_srgb.npy (float32 [0,1])
        ↓ float32 binary, HxWx3 row-major
    galosh_yuv_cpu.exe (sRGB → linear RGB → BT.709 YCbCr → Y GAT+LOSH,
                        Cb/Cr LOESS guided by Y → BT.709 inv → sRGB gamma)
        ↓ float32 binary
    sRGB float32 → uint8 PNG

Metrics:
    psnr_isp  : sRGB output vs gt_srgb (camera ISP ground truth)
    No gt_raw / gt2 path — YUV CPU operates entirely in sRGB space.

Parallelism: multiprocessing.Pool of N_WORKERS, each spawns
galosh_yuv_cpu.exe with OMP_NUM_THREADS=OMP_PER_WORKER.

Bench acts as the reference for:
  - Stage 3 GPU YUV port correctness (PSNR should match within float precision)
  - CPU vs GPU speed comparison (CPU mean_ms here is the baseline)
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

GALOSH_ROOT = Path(__file__).parent.parent.parent
EXE         = GALOSH_ROOT / "standalone" / "galosh_yuv_cpu.exe"
BENCH_DIR   = Path(os.path.expanduser(r"~\datasets\sidd\medium_bench"))
OUT_ROOT    = GALOSH_ROOT / "benchmark" / "sidd_medium"
TMP_DIR     = GALOSH_ROOT / "benchmark" / "sidd_medium" / "_tmp_yuvcpubench"
TMP_DIR.mkdir(parents=True, exist_ok=True)

UCRT_BIN    = r"C:\msys64\ucrt64\bin"

VARIANTS = [
    # method_dir,                          robust_shrink
    ("galosh_yuv_cpu",                     0),
    ("galosh_yuv_cpu_robust_mad",          1),
]


def run_galosh_yuv_cpu(noisy_srgb_f32, w, h, uid, robust_shrink, omp_threads):
    """Spawn galosh_yuv_cpu.exe.  Returns denoised sRGB float32 (h, w, 3)."""
    in_path  = TMP_DIR / f"_{uid}_in.bin"
    out_path = TMP_DIR / f"_{uid}_out.bin"
    noisy_srgb_f32.astype(np.float32).tofile(str(in_path))

    env = os.environ.copy()
    if UCRT_BIN not in env.get("PATH", ""):
        env["PATH"] = UCRT_BIN + os.pathsep + env.get("PATH", "")
    env["OMP_NUM_THREADS"] = str(omp_threads)

    cmd = [
        str(EXE), str(in_path), str(out_path), str(w), str(h),
        "1.0", "1.0",  # strength_y, strength_c
        f"--robust-shrink={robust_shrink}",
    ]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True)
    dt = time.perf_counter() - t0

    in_path.unlink(missing_ok=True)
    if p.returncode != 0 or not out_path.exists():
        out_path.unlink(missing_ok=True)
        return None, dt, p.stderr.decode("utf-8", errors="replace")

    den = np.fromfile(str(out_path), dtype=np.float32).reshape(h, w, 3)
    out_path.unlink(missing_ok=True)
    return den, dt, ""


def process_one(args):
    scene_meta, omp_threads = args
    tag = scene_meta["tag"]

    noisy_srgb = np.load(BENCH_DIR / f"{tag}_noisy_srgb.npy").astype(np.float32)
    gt_srgb    = np.load(BENCH_DIR / f"{tag}_gt_srgb.npy"   ).astype(np.float32)

    if noisy_srgb.max() > 1.5: noisy_srgb /= 255.0
    if gt_srgb.max()    > 1.5: gt_srgb    /= 255.0

    h, w = noisy_srgb.shape[:2]

    rows = []
    for method_dir, robust_shrink in VARIANTS:
        out_dir = OUT_ROOT / method_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{tag}_den.png"

        den, dt_ms, err = run_galosh_yuv_cpu(
            noisy_srgb, w, h,
            uid=f"{tag}_{method_dir}",
            robust_shrink=robust_shrink,
            omp_threads=omp_threads,
        )
        if den is None:
            rows.append({"tag": tag, "method": method_dir,
                         "ok": False, "err": err[-300:]})
            continue

        den_clip = np.clip(den, 0.0, 1.0)
        Image.fromarray((den_clip * 255 + 0.5).astype(np.uint8), mode="RGB").save(out_png)

        psnr_isp = -10.0 * np.log10(np.mean((den_clip - gt_srgb) ** 2) + 1e-12)
        rows.append({
            "tag": tag, "method": method_dir, "ok": True,
            "ms":       float(round(dt_ms * 1000, 1)),
            "psnr_isp": float(round(psnr_isp, 3)),
        })
        print(f"  [{tag} {method_dir:<33s}] {dt_ms*1000:7.1f}ms PSNR_isp={psnr_isp:.3f}",
              flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--omp",     type=int, default=0)
    parser.add_argument("--scenes",  type=int, default=0)
    args = parser.parse_args()

    if not EXE.exists():
        sys.exit(f"missing exe: {EXE}")
    if not BENCH_DIR.exists():
        sys.exit(f"missing dataset: {BENCH_DIR}")

    n_cores = os.cpu_count() or 8
    n_workers = args.workers or max(1, n_cores // 4)
    omp_threads = args.omp or max(1, n_cores // n_workers)

    with open(BENCH_DIR / "scenes.json") as f:
        scenes = json.load(f)
    if args.scenes:
        scenes = scenes[:args.scenes]

    print(f"=== GALOSH YUV CPU bench (legacy + v5 robust-MAD) ===")
    print(f"  exe       : {EXE}")
    print(f"  dataset   : {BENCH_DIR}")
    print(f"  scenes    : {len(scenes)}")
    print(f"  methods   : {[v[0] for v in VARIANTS]}")
    print(f"  workers   : {n_workers} x OMP_NUM_THREADS={omp_threads}  (cpu_count={n_cores})")
    print()

    t0 = time.time()
    if n_workers <= 1:
        all_rows = []
        for s in scenes:
            all_rows.extend(process_one((s, omp_threads)))
    else:
        from multiprocessing import Pool
        pool_args = [(s, omp_threads) for s in scenes]
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_one, pool_args)
        all_rows = [row for batch in results for row in batch]
    dt_total = time.time() - t0

    print()
    print(f"=== Aggregate (total {dt_total:.1f}s) ===")
    by_method = {}
    for r in all_rows:
        if not r.get("ok"): continue
        by_method.setdefault(r["method"], []).append(r)
    for method, rs in by_method.items():
        ms = np.mean([r["ms"] for r in rs])
        isp = np.mean([r["psnr_isp"] for r in rs])
        print(f"  {method:<35s} N={len(rs):2d}  mean_ms={ms:7.1f}  PSNR_isp={isp:.3f}")

    metrics_path = OUT_ROOT / "_metrics_yuv_cpu_bench.json"
    metrics_path.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
    print(f"\nWrote {metrics_path}")


if __name__ == "__main__":
    main()
