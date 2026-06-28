"""GALOSH RAW CPU v2 chromawiener bench.

v2 = v1 chromaup (--chroma-up=1) + Lee local-Wiener residual reinjection on
chroma planes (--chroma-wiener=1).  Runs galosh_raw_cpu.exe across the
80-pair SIDD Medium subset and saves calibrated sRGB PNGs to:

    benchmark/sidd_medium/galosh_raw_cpu_v2_chromawiener/<tag>_den.png

v0 base and v1 chromaup PNGs from bench_galosh_raw_cpu_v1.py are kept
intact.  Identical pipeline (Menon demosaic + per-scene 3x4 affine + sRGB
gamma) and identical CLI parameters (luma_str=1.0, chroma_str=1.0); only
the new --chroma-wiener=1 flag differs.

Per-image pipeline:
    noisy_raw.npy
        ↓ galosh_raw_cpu.exe --chroma-up=1 --chroma-wiener=1
    denoised raw .bin
        ↓ Menon (DDFAPD) demosaic
    linear RGB
        ↓ per-scene 3x4 affine (estimated once from gt_raw + gt_srgb)
    sRGB linear → sRGB gamma → uint8 PNG

Parallelism:
    multiprocessing.Pool of N_WORKERS, each worker spawns a
    galosh_raw_cpu.exe subprocess with OMP_NUM_THREADS=OMP_PER_WORKER.
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
EXE         = GALOSH_ROOT / "standalone" / "galosh_raw_cpu.exe"
BENCH_DIR   = Path(os.path.expanduser(r"~\datasets\sidd\medium_bench"))
OUT_ROOT    = GALOSH_ROOT / "benchmark" / "sidd_medium"
TMP_DIR     = GALOSH_ROOT / "benchmark" / "sidd_medium" / "_tmp_v2bench"
TMP_DIR.mkdir(parents=True, exist_ok=True)

UCRT_BIN    = r"C:\msys64\ucrt64\bin"

VARIANTS = [
    # method_dir,                              chroma_up, chroma_wiener
    ("galosh_raw_cpu_v2_chromawiener",         1,         1),
]


def linear_to_srgb(x):
    return np.where(x <= 0.0031308,
                    12.92 * x,
                    1.055 * np.power(np.maximum(x, 0.0), 1.0/2.4) - 0.055)


def srgb_to_linear(x):
    return np.where(x <= 0.04045,
                    x / 12.92,
                    np.power((np.maximum(x, 0.0) + 0.055) / 1.055, 2.4))


def demosaic_menon(bayer):
    from colour_demosaicing import demosaicing_CFA_Bayer_DDFAPD
    rgb = demosaicing_CFA_Bayer_DDFAPD(bayer.astype(np.float64), pattern='RGGB')
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def estimate_affine(gt_raw, gt_srgb):
    rgb_menon = demosaic_menon(gt_raw)
    rgb_isp_lin = srgb_to_linear(gt_srgb.astype(np.float64))
    step = 4
    A = rgb_menon[::step, ::step].reshape(-1, 3).astype(np.float64)
    B = rgb_isp_lin[::step, ::step].reshape(-1, 3).astype(np.float64)
    A_aug = np.hstack([A, np.ones((A.shape[0], 1))])
    M, _, _, _ = np.linalg.lstsq(A_aug, B, rcond=None)
    return M.T


def raw_to_srgb_calibrated(bayer, affine):
    rgb_lin = demosaic_menon(bayer).astype(np.float64)
    h, w = rgb_lin.shape[:2]
    flat = rgb_lin.reshape(-1, 3)
    out = flat @ affine[:, :3].T + affine[:, 3]
    out = np.clip(out.reshape(h, w, 3), 0.0, 1.0)
    return np.clip(linear_to_srgb(out), 0.0, 1.0).astype(np.float32)


def run_galosh_raw_cpu(noisy_raw, w, h, uid, chroma_up, chroma_wiener,
                        omp_threads):
    in_path  = TMP_DIR / f"_{uid}_in.bin"
    out_path = TMP_DIR / f"_{uid}_out.bin"
    noisy_raw.astype(np.float32).tofile(str(in_path))

    env = os.environ.copy()
    if UCRT_BIN not in env.get("PATH", ""):
        env["PATH"] = UCRT_BIN + os.pathsep + env.get("PATH", "")
    env["OMP_NUM_THREADS"] = str(omp_threads)

    cmd = [
        str(EXE), str(in_path), str(out_path), str(w), str(h),
        "galosh", "1.0", "1.0", "1.0", "0", "0",
        f"--chroma-up={chroma_up}",
        f"--chroma-wiener={chroma_wiener}",
    ]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True)
    dt = time.perf_counter() - t0

    in_path.unlink(missing_ok=True)
    if p.returncode != 0 or not out_path.exists():
        out_path.unlink(missing_ok=True)
        return None, dt, p.stderr.decode("utf-8", errors="replace")

    den = np.fromfile(str(out_path), dtype=np.float32).reshape(h, w)
    out_path.unlink(missing_ok=True)
    return den, dt, ""


def process_one(args):
    scene_meta, omp_threads = args
    tag = scene_meta["tag"]

    gt_raw    = np.load(BENCH_DIR / f"{tag}_gt_raw.npy"   ).astype(np.float32)
    gt_srgb   = np.load(BENCH_DIR / f"{tag}_gt_srgb.npy"  )
    noisy_raw = np.load(BENCH_DIR / f"{tag}_noisy_raw.npy").astype(np.float32)
    h, w = gt_raw.shape

    if isinstance(gt_srgb.dtype, np.dtype) and gt_srgb.dtype.kind == 'u':
        gt_srgb = gt_srgb.astype(np.float32) / 255.0

    aff = estimate_affine(gt_raw, gt_srgb)
    gt2_srgb = raw_to_srgb_calibrated(gt_raw, aff)

    rows = []
    for method_dir, chroma_up, chroma_wiener in VARIANTS:
        out_dir = OUT_ROOT / method_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{tag}_den.png"

        den_raw, dt_ms, err = run_galosh_raw_cpu(
            noisy_raw, w, h,
            uid=f"{tag}_{method_dir}",
            chroma_up=chroma_up,
            chroma_wiener=chroma_wiener,
            omp_threads=omp_threads,
        )
        if den_raw is None:
            rows.append({"tag": tag, "method": method_dir,
                         "ok": False, "err": err[-300:]})
            continue

        den_srgb_f = raw_to_srgb_calibrated(den_raw, aff)
        Image.fromarray((den_srgb_f * 255 + 0.5).astype(np.uint8), mode="RGB").save(out_png)

        psnr_gt2 = -10.0 * np.log10(np.mean((den_srgb_f - gt2_srgb) ** 2) + 1e-12)
        psnr_isp = -10.0 * np.log10(np.mean((den_srgb_f - gt_srgb)  ** 2) + 1e-12)
        rows.append({
            "tag": tag, "method": method_dir, "ok": True,
            "ms":       float(round(dt_ms * 1000, 1)),
            "psnr_gt2": float(round(psnr_gt2, 3)),
            "psnr_isp": float(round(psnr_isp, 3)),
        })
        print(f"  [{tag} {method_dir:<35s}] {dt_ms*1000:7.1f}ms PSNR_gt2={psnr_gt2:.3f} PSNR_isp={psnr_isp:.3f}",
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

    print(f"=== GALOSH RAW CPU v2 chromawiener bench ===")
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
        gt2 = np.mean([r["psnr_gt2"] for r in rs])
        isp = np.mean([r["psnr_isp"] for r in rs])
        print(f"  {method:<37s} N={len(rs):2d}  mean_ms={ms:7.1f}  PSNR_gt2={gt2:.3f}  PSNR_isp={isp:.3f}")

    metrics_path = OUT_ROOT / "_metrics_v2bench.json"
    metrics_path.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
    print(f"\nWrote {metrics_path}")


if __name__ == "__main__":
    main()
