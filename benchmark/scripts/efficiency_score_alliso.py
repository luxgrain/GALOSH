#!/usr/bin/env python3
"""
Compute composite quality-efficiency scores for all ISOs.
Uses Kodak 24 full benchmark results.
"""
import numpy as np
import json
from pathlib import Path

BASE = Path(__file__).parent.parent
with open(BASE / "results" / "v2_benchmark_results.json") as f:
    results = json.load(f)

methods = ["BM3D-PC", "BM3D-CFA", "DnCNN-B", "DRUNet", "Ours(V2)"]
key_map = {
    "BM3D-PC":  "1_BM3D-PC",
    "BM3D-CFA": "2_BM3D-CFA",
    "DnCNN-B":  "3_DnCNN-B",
    "DRUNet":   "4_DRUNet",
    "Ours(V2)": "6_OursV2",
}
gflops = {"BM3D-PC": 14, "BM3D-CFA": 8, "DnCNN-B": 524, "DRUNet": 1632, "Ours(V2)": 7}

def norm(v, higher_better=True):
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-10:
        return np.ones_like(v)
    n = (v - lo) / (hi - lo)
    return n if higher_better else (1.0 - n)


# ── Per-ISO tables ──
print("=" * 100)
print("QUALITY-EFFICIENCY SCORES BY ISO (Kodak 24, higher = better)")
print("=" * 100)

all_scores = []

for r in results:
    iso = r["iso"]
    psnr_v = np.array([r[f"{key_map[m]}_psnr"] for m in methods])
    ssim_v = np.array([r[f"{key_map[m]}_ssim"] for m in methods])
    lpips_v = np.array([r[f"{key_map[m]}_lpips"] for m in methods])
    gf = np.array([gflops[m] for m in methods], dtype=float)

    pn = norm(psnr_v, True)
    sn = norm(ssim_v, True)
    ln = norm(lpips_v, False)  # lower better
    cn = norm(gf, False)       # lower better

    quality = (pn + sn + ln) / 3.0
    eff = 0.7 * quality + 0.3 * cn  # 70% quality + 30% cost

    print(f"\nISO {iso}:")
    print(f"  {'Method':<12} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'GFLOPs':>7} | {'Q_psnr':>6} {'Q_ssim':>6} {'Q_lpips':>7} {'Q_cost':>6} | {'Quality':>7} {'70Q+30C':>7}")
    print(f"  {'-'*95}")
    for i, m in enumerate(methods):
        print(f"  {m:<12} {psnr_v[i]:>7.2f} {ssim_v[i]:>7.4f} {lpips_v[i]:>7.4f} {gf[i]:>7.0f} |"
              f" {pn[i]:>6.3f} {sn[i]:>6.3f} {ln[i]:>7.3f} {cn[i]:>6.3f} |"
              f" {quality[i]:>7.3f} {eff[i]:>7.3f}")

    # Rankings
    q_rank = {m: int(np.sum(quality > quality[i]) + 1) for i, m in enumerate(methods)}
    e_rank = {m: int(np.sum(eff > eff[i]) + 1) for i, m in enumerate(methods)}

    row = {"iso": iso}
    for i, m in enumerate(methods):
        row[f"{m}_quality"] = round(float(quality[i]), 3)
        row[f"{m}_eff"] = round(float(eff[i]), 3)
        row[f"{m}_q_rank"] = q_rank[m]
        row[f"{m}_e_rank"] = e_rank[m]
    all_scores.append(row)


# ── Summary ranking table ──
print()
print("=" * 100)
print("RANKING SUMMARY")
print(f"{'ISO':>6}", end="")
for m in methods:
    print(f"  {m:>12}", end="")
print()

print("\nQuality rank (pure PSNR+SSIM+LPIPS):")
for row in all_scores:
    print(f"{row['iso']:>6}", end="")
    for m in methods:
        r = row[f"{m}_q_rank"]
        s = row[f"{m}_quality"]
        print(f"  #{r} ({s:.3f})  ", end="")
    print()

print("\nEfficiency rank (70% quality + 30% cost):")
for row in all_scores:
    print(f"{row['iso']:>6}", end="")
    for m in methods:
        r = row[f"{m}_e_rank"]
        s = row[f"{m}_eff"]
        print(f"  #{r} ({s:.3f})  ", end="")
    print()

# ── Win count ──
print()
print("=" * 100)
print("WIN COUNT (across 6 ISOs)")
print(f"  {'Method':<12} {'Quality #1':>10} {'Eff #1':>10} {'Quality top2':>12} {'Eff top2':>12}")
print(f"  {'-'*60}")
for m in methods:
    q1 = sum(1 for row in all_scores if row[f"{m}_q_rank"] == 1)
    e1 = sum(1 for row in all_scores if row[f"{m}_e_rank"] == 1)
    q2 = sum(1 for row in all_scores if row[f"{m}_q_rank"] <= 2)
    e2 = sum(1 for row in all_scores if row[f"{m}_e_rank"] <= 2)
    print(f"  {m:<12} {q1:>10} {e1:>10} {q2:>12} {e2:>12}")

# ── Average scores across all ISOs ──
print()
print("=" * 100)
print("AVERAGE ACROSS ALL ISOs")
print(f"  {'Method':<12} {'Avg Quality':>12} {'Avg Efficiency':>15} {'GFLOPs':>8}")
print(f"  {'-'*50}")
for m in methods:
    avg_q = np.mean([row[f"{m}_quality"] for row in all_scores])
    avg_e = np.mean([row[f"{m}_eff"] for row in all_scores])
    print(f"  {m:<12} {avg_q:>12.3f} {avg_e:>15.3f} {gflops[m]:>8}")
