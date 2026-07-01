"""Merge the galosh-only 1493 RawNIND run into the DL+classical 1493 run, and
regenerate _table.txt over all methods on the full 1493."""
import json
import numpy as np
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "benchmark" / "rawnind_srgb_results"
dl = json.load(open(R / "_metrics_dl1493.json"))   # all methods (galosh only 261) + _noisy
gal = json.load(open(R / "_metrics.json"))          # galosh-only 1493 + _noisy

merged = [r for r in dl if r.get("method") != "galosh_yuv_gpu_fp32"]
merged += [r for r in gal if r.get("method") == "galosh_yuv_gpu_fp32"]
(R / "_metrics.json").write_text(json.dumps(merged, indent=1))

agg = {}
for r in merged:
    if r.get("ok") is False:
        continue
    agg.setdefault(r["method"], []).append(r)
order = ["_noisy", "galosh_yuv_gpu_fp32", "cbm3d", "color_nlm", "guided",
         "nafnet", "scunet", "restormer"]
hdr = f"{'method':<22}{'N':>5}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}{'DISTS':>8}{'NIQE':>7}{'ms':>9}"
note = ("RawNIND-rendered sRGB, 512x512 FULL 1493 | ALL methods BLIND, sRGB colour | "
        "CBM3D/Color-NLM/Guided sigma=estimate_sigma; GALOSH/NAFNet-SIDD/SCUNet-real/Restormer-SIDD self-contained")
lines = [note, "", hdr, "-" * len(hdr)]
for m in order:
    rs = agg.get(m, [])
    if not rs:
        continue
    def mean(k):
        v = [x[k] for x in rs if k in x]
        return np.mean(v) if v else float("nan")
    ms = [x["ms"] for x in rs if "ms" in x]
    lines.append(f"{m:<22}{len(rs):>5}{mean('psnr'):>8.2f}{mean('ssim'):>8.4f}"
                 f"{mean('lpips'):>8.4f}{mean('dists'):>8.4f}{mean('niqe'):>7.2f}"
                 f"{(np.mean(ms) if ms else float('nan')):>9.1f}")
table = "\n".join(lines)
(R / "_table.txt").write_text(table)
print(table)
print("\nmerged rows:", len(merged), "| galosh N:", len(agg.get("galosh_yuv_gpu_fp32", [])))
