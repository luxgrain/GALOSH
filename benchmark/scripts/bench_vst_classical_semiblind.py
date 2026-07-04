"""Classical VST + BM3D-CFA / VST + NLM-CFA bench with DND-style semi-blind sigma.

Faithful reproduction of DND VST baseline protocol:
  1. Compute per-CFA sigma_ch from std(noisy - gt) per channel (= DND-provided
     sigma_raw[bb, yy, xx] semantics)
  2. Fit single (alpha, sigma_sq) via LSQ from 4 (mean_ch, sigma_ch^2) points
     (= matches DND nlf.a, nlf.b semantics: shared Poisson-Gauss model)
  3. Per CFA channel:
       gat = (2/sqrt(alpha)) * sqrt(noisy_ch + 3*alpha/8 + sigma_sq/alpha)
       denoised_gat = BM3D(gat, sigma=1.0)   # GAT normalizes noise to ~N(0,1)
       denoised_ch = (alpha/4) * denoised_gat^2 - 3*alpha/8 - sigma_sq/alpha
  4. Reassemble Bayer output

Stored as:
  galosh_raw_cpu_bm3dcfa_vst_semiblind   (= VST + BM3D-CFA)
  galosh_raw_cpu_nlmcfa_vst_semiblind    (= VST + NLM-CFA)
"""
import os
import sys
import json
import time
import argparse
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "methods"))

import bench_sidd_medium as smb  # for run_nlm_cfa_cuda
from bm3d_cfa import run_bm3d_cfa

BENCH_OUT = Path(os.environ.get("GALOSH_RAWNIND_BENCH", "benchmark/datasets/rawnind_bench"))
METRICS_JSON = BENCH_OUT / "_metrics.json"


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def ssim_raw(a, b):
    from skimage.metrics import structural_similarity as ssim_fn
    return float(ssim_fn(a, b, data_range=1.0))


def dnd_style_alpha_sigma(noisy, gt):
    """Per-CFA std(noisy-gt) -> LSQ fit (alpha, sigma_sq) under Poisson-Gauss model."""
    diff = noisy.astype(np.float64) - gt.astype(np.float64)
    means, vars_ = [], []
    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        means.append(float(noisy[dy::2, dx::2].mean()))
        vars_.append(float(diff[dy::2, dx::2].var()))
    A = np.column_stack([means, np.ones(4)])
    sol, *_ = np.linalg.lstsq(A, np.array(vars_), rcond=None)
    alpha = max(float(sol[0]), 1e-8)
    sigma_sq = max(float(sol[1]), 1e-12)
    return alpha, sigma_sq


def gat_forward(x, alpha, sigma_sq):
    """GAT: D = (2/sqrt(alpha)) * sqrt(x + 3*alpha/8 + sigma_sq/alpha)."""
    a = max(float(alpha), 1e-12)
    arg = np.maximum(x.astype(np.float64) + 3.0 * a / 8.0 + sigma_sq / a, 0.0)
    return (2.0 / np.sqrt(a)) * np.sqrt(arg)


def gat_inverse_algebraic(D, alpha, sigma_sq):
    """Algebraic inverse GAT: x = (alpha/4)*D^2 - 3*alpha/8 - sigma_sq/alpha."""
    a = max(float(alpha), 1e-12)
    return (a / 4.0) * D.astype(np.float64) ** 2 - 3.0 * a / 8.0 - sigma_sq / a


def vst_bm3d_cfa(noisy, alpha, sigma_sq):
    """VST + BM3D-CFA: forward GAT on full Bayer, BM3D-CFA with sigma=1, inverse GAT."""
    gat = gat_forward(noisy, alpha, sigma_sq).astype(np.float32)
    den_gat, _ = run_bm3d_cfa(gat, sigma=[1.0, 1.0, 1.0, 1.0])
    den = gat_inverse_algebraic(den_gat, alpha, sigma_sq)
    return np.clip(den, 0.0, 1.0).astype(np.float32)


def vst_nlm_cfa(noisy, alpha, sigma_sq):
    """VST + NLM-CFA: forward GAT, normalize to [0, 1] for NLM-cuda, inverse GAT.

    NLM-CFA-cuda internally clips inputs to [0, 1]. GAT output is in roughly
    [1, 60+] for typical raw data, so we must rescale before NLM.
    """
    gat = gat_forward(noisy, alpha, sigma_sq).astype(np.float32)
    gat_max = float(max(gat.max(), 1e-6))
    gat_norm = (gat / gat_max).astype(np.float32)
    sigma_norm = 1.0 / gat_max  # GAT noise std is ~1, scaled by 1/gat_max
    den_norm, _ = smb.run_nlm_cfa_cuda(gat_norm, sigma=sigma_norm)
    den_gat = den_norm * gat_max
    den = gat_inverse_algebraic(den_gat, alpha, sigma_sq)
    return np.clip(den, 0.0, 1.0).astype(np.float32)


METHODS = {
    "galosh_raw_cpu_bm3dcfa_vst_semiblind": vst_bm3d_cfa,
    "galosh_raw_cpu_nlmcfa_vst_semiblind":  vst_nlm_cfa,
}


def run_one(tag, method_name, runner):
    scene = tag.rsplit("__ISO", 1)[0]
    noisy_path = BENCH_OUT / "__noisy_raw__" / f"{tag}.npy"
    gt_path = BENCH_OUT / "__gt_raw__" / f"{scene}.npy"
    output_npy = BENCH_OUT / method_name / f"{tag}.npy"

    if output_npy.exists():
        try:
            denoised = np.load(output_npy).astype(np.float32)
            gt = np.load(gt_path).astype(np.float32)
            return (tag, psnr(gt, denoised), ssim_raw(gt, denoised), 0.0, "cached")
        except Exception:
            pass

    try:
        noisy = np.load(noisy_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
    except Exception as e:
        return (tag, None, None, 0.0, f"load_err {e}")

    alpha, sigma_sq = dnd_style_alpha_sigma(noisy, gt)

    t0 = time.time()
    try:
        denoised = runner(noisy, alpha, sigma_sq)
    except Exception as e:
        return (tag, None, None, time.time() - t0, f"run_err {e}")
    dt = time.time() - t0

    if denoised.shape != noisy.shape:
        return (tag, None, None, dt, "shape_mismatch")

    np.save(output_npy, denoised)
    return (tag, psnr(gt, denoised), ssim_raw(gt, denoised), dt, "ok")


def bench_method(method_name, runner, workers):
    (BENCH_OUT / method_name).mkdir(parents=True, exist_ok=True)
    tags = sorted(p.stem for p in (BENCH_OUT / "__noisy_raw__").glob("*.npy"))
    print(f"\n[{method_name}] n={len(tags)} workers={workers}")

    metrics = json.load(open(METRICS_JSON)) if METRICS_JSON.exists() else {}
    if method_name not in metrics:
        metrics[method_name] = {"per_image": {}}
    per_image = metrics[method_name]["per_image"]

    start = time.time()
    n_done = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(run_one, tag, method_name, runner): tag
            for tag in tags if tag not in per_image
        }
        n_cached = len(tags) - len(futures)
        print(f"  {n_cached} cached, {len(futures)} to run")
        for fut in as_completed(futures):
            tag, ps, ss, dt, status = fut.result()
            if ps is not None:
                per_image[tag] = {"psnr": ps, "ssim": ss, "dt": dt}
                n_done += 1
            else:
                n_fail += 1
                if status != "shape_mismatch":
                    print(f"  FAIL {tag}: {status}")
            if (n_done % 100) == 0 and n_done > 0:
                elapsed = time.time() - start
                print(f"  [{n_done}/{len(futures)}] {elapsed/60:.1f}min")
                with open(METRICS_JSON, "w") as fp:
                    json.dump(metrics, fp, indent=2)

    with open(METRICS_JSON, "w") as fp:
        json.dump(metrics, fp, indent=2)
    elapsed = (time.time() - start) / 60
    print(f"[{method_name} done] {n_done} processed, {n_fail} failed, {elapsed:.1f} min")
    psnr_all = [v["psnr"] for v in per_image.values() if v.get("psnr")]
    ssim_all = [v["ssim"] for v in per_image.values() if v.get("ssim")]
    if psnr_all:
        print(f"  aggregate: PSNR={np.mean(psnr_all):.3f}  SSIM={np.mean(ssim_all):.4f}  n={len(per_image)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--methods", nargs="*", default=list(METHODS.keys()))
    args = parser.parse_args()

    for method_name in args.methods:
        if method_name not in METHODS:
            print(f"  unknown method: {method_name}, skip")
            continue
        bench_method(method_name, METHODS[method_name], args.workers)


if __name__ == "__main__":
    main()
