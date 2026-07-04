"""Fill the result-table CPU-time holes the archives don't cover: GALOSH FP32 (blind)
and NLM-CFA (skimage reference, same patch5/search23 as the CUDA impl) at the result-table
sizes (SIDD 16MP full-frame subset / RawNIND 512^2 subset). Single-threaded methods on a
32-core box, so a light concurrent bench does not skew them.

  python _fill_cpu_times.py --sidd-n 12 --rawnind-n 100
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
from skimage.restoration import denoise_nl_means

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_raw_campaign as C

GALOSH = Path(__file__).resolve().parents[2]
CPU_EXE = GALOSH / "standalone" / "galosh_raw_cpu.exe"
TMP = Path(r"C:\tmp")

def smad(nr):
    hh = nr[:, 1:] - nr[:, :-1]
    return float(np.median(np.abs(hh)) / 0.6745)

def galosh_cpu(nr):
    h, w = nr.shape
    ip, op = TMP / "_fill_in.raw", TMP / "_fill_out.raw"
    nr.astype(np.float32).tofile(ip)
    t0 = time.time()
    r = subprocess.run([str(CPU_EXE), str(ip), str(op), str(w), str(h),
                        "galosh", "1.0", "1.0", "1.0", "0", "0"],
                       capture_output=True, timeout=1800, cwd=str(GALOSH))
    dt = time.time() - t0
    assert r.returncode == 0, r.stderr.decode()[-200:]
    return dt

def nlm_cfa_cpu(nr):
    sigma = smad(nr)
    t0 = time.time()
    for p in (nr[0::2, 0::2], nr[0::2, 1::2], nr[1::2, 0::2], nr[1::2, 1::2]):
        denoise_nl_means(p.astype(np.float32), patch_size=5, patch_distance=11,
                         h=sigma, sigma=sigma, fast_mode=True, preserve_range=True)
    return time.time() - t0

def run(dataset, n):
    items, _ = (C.load_sidd_medium() if dataset == "sidd_medium" else C.load_rawnind())
    items = items[::max(1, len(items)//n)][:n]
    out = {"galosh_fp32_cpu": [], "nlm_cfa_cpu": []}
    for i, (stem, nf, gr, _gs) in enumerate(items):
        nr = np.load(nf).astype(np.float32)
        out["galosh_fp32_cpu"].append(galosh_cpu(nr))
        out["nlm_cfa_cpu"].append(nlm_cfa_cpu(nr))
        print(f"  [{dataset} {i+1}/{len(items)}] {stem}: galosh {out['galosh_fp32_cpu'][-1]:.2f}s"
              f"  nlm {out['nlm_cfa_cpu'][-1]:.2f}s", flush=True)
    return {k: {"n": len(v), "mean_s": float(np.mean(v)), "median_s": float(np.median(v))}
            for k, v in out.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidd-n", type=int, default=12)
    ap.add_argument("--rawnind-n", type=int, default=100)
    a = ap.parse_args()
    res = {"sidd_medium": run("sidd_medium", a.sidd_n),
           "rawnind": run("rawnind", a.rawnind_n),
           "note": "single-threaded; galosh = galosh_raw_cpu.exe blind incl file I/O; "
                   "nlm = skimage per-plane fast_mode patch5/search23 (matches CUDA params)"}
    print(json.dumps(res, indent=1))
    op = GALOSH / "benchmark" / "results_raw" / "_cpu_time_fill.json"
    json.dump(res, open(op, "w"), indent=1)
    print(f"saved {op}")

if __name__ == "__main__":
    main()
