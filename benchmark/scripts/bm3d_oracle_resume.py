#!/usr/bin/env python3
"""Resume bm3d_cfa_oracle for the remaining 16 UIDs and merge with the
existing 64 from the earlier partial log."""
import sys, os, re, json, time
import numpy as np
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "methods"))
from bench_sidd_medium import BENCH_DIR
from bm3d_cfa import run_bm3d_cfa

LOG_PATH = r"C:\Users\luxgrain\AppData\Local\Temp\claude\c--Users-luxgrain-GALOSH\7cb74090-d4e6-415a-8cac-1d59ba9abaaf\tasks\bqo6yvno7.output"

def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def parse_existing(path):
    log = open(path).read()
    pairs = re.findall(r'\[\d+/80\]\s+(\S+)\s+\d+x\d+.*\n\s+bm3d_cfa_oracle\s+PSNR=([0-9.]+)\s+den_time=([0-9.]+)s\s+wall=([0-9.]+)s', log)
    return {uid: {"psnr": float(p), "time": float(t), "wall": float(w)} for uid, p, t, w in pairs}


def main():
    existing = parse_existing(LOG_PATH)
    print(f"Parsed existing: N={len(existing)}")
    all_ids = sorted([Path(f).stem.replace("_noisy_raw", "")
                      for f in BENCH_DIR.glob("*_noisy_raw.npy")])
    todo = [u for u in all_ids if u not in existing]
    print(f"Remaining: {len(todo)}")

    per_image = dict(existing)
    for i, uid in enumerate(todo):
        noisy = np.load(str(BENCH_DIR / f"{uid}_noisy_raw.npy")).astype(np.float32)
        gt    = np.load(str(BENCH_DIR / f"{uid}_gt_raw.npy")).astype(np.float32)
        sigma = float(np.std(noisy.astype(np.float64) - gt.astype(np.float64)))
        t0 = time.time()
        den, dt = run_bm3d_cfa(noisy, sigma=sigma)
        wall = time.time() - t0
        p = psnr(den, gt)
        per_image[uid] = {"psnr": p, "time": float(dt), "wall": float(wall)}
        print(f"[{i+1}/{len(todo)}] {uid}: PSNR={p:.3f} den_time={dt:.2f}s wall={wall:.2f}s")

    # Aggregate + save
    psnrs = [v["psnr"] for v in per_image.values()]
    times = [v["time"] for v in per_image.values()]
    print(f"\nbm3d_cfa_oracle FULL 80: mean PSNR = {np.mean(psnrs):.3f}, mean time = {np.mean(times):.2f}s")

    out = {"bm3d_cfa_oracle": {"per_image": per_image,
                                "psnr_mean": float(np.mean(psnrs)),
                                "time_mean": float(np.mean(times))}}
    out_path = SCRIPTS.parent / "results" / "sidd_medium_raw_domain_bm3d_oracle.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
