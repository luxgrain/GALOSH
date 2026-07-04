"""Bit-exact check: r32 CPU exe before vs after the no-INT64 split-multiply fix
(galosh_raw_cpu_int.c run_max*3/5). Mathematically exact for run_max>=0; this verifies it
on 2 real SIDD frames (blind path exercises the alpha_est<=1 rescue branch or not — either
way outputs must be byte-identical)."""
import numpy as np, subprocess, os, json
from pathlib import Path

ROOT = Path(os.environ.get("GALOSH_ROOT", str(Path(__file__).resolve().parents[2])))

B = Path(os.environ.get("GALOSH_SIDD_BENCH", "benchmark/datasets/sidd_medium_bench"))
scenes = json.load(open(B / "scenes.json"))
outs = []
for tag in [scenes[0]["tag"], scenes[40]["tag"]]:
    n = np.load(B / f"{tag}_noisy_raw.npy").astype(np.float32)
    h, w = n.shape
    n.tofile(r"C:\tmp\_ne_in.raw")
    pair = []
    for exe, t in ((r"C:\tmp\_r32_prefix.exe", "pre"),
                   (str(ROOT / "standalone" / "galosh_raw_cpu_int.exe"), "post")):
        r = subprocess.run([exe, r"C:\tmp\_ne_in.raw", rf"C:\tmp\_ne_out_{t}.raw", str(w), str(h),
                            "galosh", "1.0", "1.0", "1.0", "0", "0"],
                           capture_output=True, timeout=1800, cwd=str(ROOT))
        assert r.returncode == 0, r.stderr.decode()[-150:]
        pair.append(np.fromfile(rf"C:\tmp\_ne_out_{t}.raw", dtype=np.float32))
    same = np.array_equal(pair[0], pair[1])
    print(f"{tag}: bit-exact pre==post? {same}  (maxdiff {np.abs(pair[0] - pair[1]).max():.2e})", flush=True)
    outs.append(same)
print("ALL BIT-EXACT:", all(outs))
