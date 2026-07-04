"""Re-measure the VST+GALOSH-core ablation TIME on the GPU exe (alpha/sigma supplied
externally = 'no native estimation'), so the paper's Time column compares the same
platform as the headline GALOSH FP32 (GPU) row. Quality stays from the original CPU
run (CPU<->GPU reconciled to ~8e-5); we recompute raw-PSNR here only as a sanity check.

  python _revst_gpu_time.py --dataset sidd_medium|rawnind [--limit N]
"""
import argparse, json, sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_raw_v2_campaign as C
import bench_sidd_medium as smb

def psnr(a, b):
    return float(-10 * np.log10(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2) + 1e-12))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["sidd_medium", "rawnind"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items, _ = (C.load_sidd_medium() if a.dataset == "sidd_medium" else C.load_rawnind())
    if a.limit: items = items[:a.limit]
    print(f"[{a.dataset}] {len(items)} scenes", flush=True)

    dev = None                      # probe: this box has AMD/Intel iGPUs that can't run galosh
    res = {"ext": {"t": [], "p": []}, "blind": {"t": [], "p": []}}
    fail = 0
    for i, (stem, nf, gr, _gs) in enumerate(items):
        noisy = np.load(nf).astype(np.float32)
        gt = np.load(gr).astype(np.float32)
        al, sg = C.blind_pg(noisy)
        h, w = noisy.shape
        row = {}
        for mode, (aa, ss) in [("ext", (al, sg)), ("blind", (0.0, 0.0))]:
            den, dt = None, None
            cand = [dev] if dev is not None else [0, 1, 2, 3]
            for d in cand:
                try:
                    den, dt = smb.run_galosh_gpu(noisy, w, h, f"revst_{i}_{mode}",
                                                 alpha=aa, sigma_sq=ss, cl_device=d)
                except Exception:
                    den = None   # runner raises when the exe fails (iGPU device) - probe next
                if den is not None:
                    dev = d; break
            if den is None:
                dev = None; row = None; break
            res[mode]["t"].append(dt); res[mode]["p"].append(psnr(np.clip(den, 0, 1), gt))
            row[mode] = (dt, res[mode]["p"][-1])
        if row is None:
            fail += 1; print(f"  [{i+1}/{len(items)}] {stem}: FAIL", flush=True); continue
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(items)}] {stem}: ext {row['ext'][0]:.2f}s/{row['ext'][1]:.2f}dB"
                  f"  blind {row['blind'][0]:.2f}s/{row['blind'][1]:.2f}dB (dev {dev})", flush=True)

    out = {"dataset": a.dataset, "n": len(res["ext"]["t"]), "fail": fail}
    for mode in ("ext", "blind"):
        out[mode] = {"time_mean_s": float(np.mean(res[mode]["t"])),
                     "time_median_s": float(np.median(res[mode]["t"])),
                     "raw_psnr_mean": float(np.mean(res[mode]["p"]))}
    print(json.dumps(out, indent=1), flush=True)
    op = Path(__file__).resolve().parents[1] / "results_raw" / f"_revst_gpu_time_{a.dataset}.json"
    json.dump(out, open(op, "w"), indent=1)
    print(f"saved {op}", flush=True)

if __name__ == "__main__":
    main()
