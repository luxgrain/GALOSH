"""Unzip SIDD Full NOISY_RAW + GT_RAW from all 28 scenes.

Each scene has 150 .MAT noisy frames + 150 .MAT GT frames.
Total ~340 GB unzipped.

Set GALOSH_SIDD_FULL to the downloaded SIDD_Full directory and
GALOSH_SIDD_EXTRACT to the (large) extraction target.
Parallel: 4 scenes at a time (= I/O-bound, more workers don't help).
"""
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["PYTHONIOENCODING"] = "utf-8"

SIDD_FULL = Path(os.environ.get("GALOSH_SIDD_FULL", "benchmark/datasets/SIDD_Full"))
EXTRACT_DIR = Path(os.environ.get("GALOSH_SIDD_EXTRACT", "benchmark/datasets/SIDD_extracted"))
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)


def unzip_one(scene_dir):
    """Unzip NOISY_RAW + GT_RAW for one scene."""
    scene_name = scene_dir.name
    out_dir = EXTRACT_DIR / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for kind in ["NOISY_RAW", "GT_RAW"]:
        zip_path = None
        for f in scene_dir.iterdir():
            if f.suffix == ".zip" and kind in f.name:
                zip_path = f
                break
        if zip_path is None:
            results.append((kind, "no zip"))
            continue
        # Check if already extracted
        target_dir = out_dir / f"{scene_name.split('_')[0]}_{kind}"
        if target_dir.exists() and len(list(target_dir.glob("*.MAT"))) >= 150:
            results.append((kind, "cached"))
            continue
        t0 = time.time()
        try:
            cmd = ["unzip", "-q", "-o", str(zip_path), "-d", str(out_dir)]
            r = subprocess.run(cmd, capture_output=True, timeout=3600)
            if r.returncode == 0:
                dt = time.time() - t0
                results.append((kind, f"ok {dt:.0f}s"))
            else:
                results.append((kind, f"err rc={r.returncode}"))
        except Exception as e:
            results.append((kind, f"exc {e}"))
    return (scene_name, results)


def main():
    scenes = sorted([d for d in SIDD_FULL.iterdir() if d.is_dir()])
    print(f"Found {len(scenes)} SIDD Full scenes")
    print(f"Unzipping to: {EXTRACT_DIR}")
    print(f"Estimated total unzipped: ~340 GB")

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(unzip_one, s): s.name for s in scenes}
        done = 0
        for fut in as_completed(futures):
            scene_name, results = fut.result()
            done += 1
            r_str = ", ".join(f"{k}={v}" for k, v in results)
            elapsed = (time.time() - t_start) / 60
            print(f"[{done}/{len(scenes)}] {elapsed:.1f}min {scene_name}: {r_str}")

    print(f"\nDone in {(time.time() - t_start) / 60:.1f}min")


if __name__ == "__main__":
    main()
