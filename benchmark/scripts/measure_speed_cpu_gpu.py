"""Measure per-method denoise wall-clock at 1080p (1920x1080 Bayer) for the
CPU+GPU speed table.  Uses the campaign runners (which report each method's
internal compute dt).  Warm-up run discarded; 2nd run recorded.

  python measure_speed_cpu_gpu.py            # natural platform (bm3d=CPU, nlm=CUDA, b2u=CUDA)
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import bench_raw_campaign as camp

SP = Path(__file__).resolve().parents[1] / "results_raw" / "speed"
nr = np.fromfile(SP / "noisy_1080p.bin", dtype=np.float32).reshape(1080, 1920)
print(f"input 1920x1080 ({nr.size/1e6:.2f} MP)", flush=True)
R = camp.make_runners()

def timeit(m, reps=2):
    dt = None
    for _ in range(reps):
        den, dt = R[m](nr, nr)
    return dt

for m in ["bm3d_cfa", "nlm_cfa", "b2u"]:
    if m not in R:
        print(f"  {m}: <no runner>", flush=True); continue
    try:
        print(f"  {m}: {timeit(m):.3f} s", flush=True)
    except Exception as e:
        print(f"  {m}: ERR {str(e)[:140]}", flush=True)
