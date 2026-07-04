"""Fresh-process retry of galosh on the RawNIND scenes where it crashed (CUDA<->OpenCL
contention accumulates in a long run; a fresh process resets it). Recovers + merges
into _metrics.json + the galosh PNG dir. Re-runnable (only re-tries still-missing tags)."""
import os
import sys, json
import numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_yuv_srgb as B

R = Path(__file__).resolve().parents[2] / "benchmark" / "results_srgb_rawnind"
ND = Path(os.environ.get("GALOSH_RAWNIND_BENCH", "benchmark/datasets/rawnind_bench")) / "__noisy_raw_render__"
GD = Path(os.environ.get("GALOSH_RAWNIND_BENCH", "benchmark/datasets/rawnind_bench")) / "__gt_raw_render__"
B.TMP.mkdir(parents=True, exist_ok=True)

rows = json.load(open(R / "_metrics.json"))
need = [x["tag"] for x in rows if x.get("method") == "galosh_yuv_gpu_fp32" and x.get("ok") is False]
print(f"retrying {len(need)} failed galosh scenes")
new = {}
for i, tag in enumerate(need):
    scene = tag.split("__ISO")[0]
    n = np.asarray(Image.open(ND / f"{tag}.png").convert("RGB"), np.float32) / 255.0
    g = np.asarray(Image.open(GD / f"{scene}.png").convert("RGB"), np.float32) / 255.0
    try:
        den, dt = B.m_galosh(n, f"retry_{i}")
    except Exception as e:
        print(f"  {tag}: STILL FAIL {str(e)[-90:]}"); continue
    (R / "galosh_yuv_gpu_fp32").mkdir(exist_ok=True)
    Image.fromarray((den * 255 + 0.5).astype(np.uint8)).save(R / "galosh_yuv_gpu_fp32" / f"{tag}.png")
    new[tag] = {"tag": tag, "method": "galosh_yuv_gpu_fp32", "ok": True, "ms": round(dt * 1000, 1),
                "psnr": B.psnr(den, g), "ssim": B.ssim_rgb(den, g), "lpips": B.lpips_m(den, g),
                "dists": B.dists_m(den, g), "niqe": B.niqe_m(den)}
    print(f"  [{i+1}/{len(need)}] {tag}: OK PSNR={new[tag]['psnr']:.2f}", flush=True)

out = []
for x in rows:
    if x.get("method") == "galosh_yuv_gpu_fp32" and x.get("tag") in new:
        out.append(new.pop(x["tag"]))
    else:
        out.append(x)
json.dump(out, open(R / "_metrics.json", "w"), indent=1)
ok = len([x for x in out if x.get("method") == "galosh_yuv_gpu_fp32" and x.get("ok")])
print(f"recovered this pass: {len(need) - len([x for x in out if x.get('method')=='galosh_yuv_gpu_fp32' and x.get('ok') is False])}; galosh OK now: {ok}/1493")
