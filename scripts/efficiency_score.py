#!/usr/bin/env python3
"""
Compute composite quality-efficiency scores for all denoising methods.
Uses ISO 400 Kodak-24 averages + FLOP estimates.
"""
import numpy as np

# ISO 400, Kodak 24 average (from benchmark)
methods = {
    "BM3D-PC":   {"psnr": 32.35, "ssim": 0.8874, "lpips": 0.3400, "gflops": 14},
    "BM3D-CFA":  {"psnr": 33.65, "ssim": 0.9055, "lpips": 0.3052, "gflops": 8},
    "DnCNN-B":   {"psnr": 32.82, "ssim": 0.9061, "lpips": 0.2951, "gflops": 524},
    "DRUNet":    {"psnr": 35.27, "ssim": 0.9431, "lpips": 0.2175, "gflops": 1632},
    "Ours(V2)":  {"psnr": 33.16, "ssim": 0.9088, "lpips": 0.2927, "gflops": 7},
}

names = list(methods.keys())

# Extract raw values
psnr_vals  = np.array([methods[n]["psnr"]   for n in names])
ssim_vals  = np.array([methods[n]["ssim"]   for n in names])
lpips_vals = np.array([methods[n]["lpips"]  for n in names])
gflops     = np.array([methods[n]["gflops"] for n in names])

# ── Normalize to [0, 1] (min-max within these methods) ──
def norm(v, higher_better=True):
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-10:
        return np.ones_like(v)
    n = (v - lo) / (hi - lo)
    return n if higher_better else (1.0 - n)

psnr_n  = norm(psnr_vals, higher_better=True)
ssim_n  = norm(ssim_vals, higher_better=True)
lpips_n = norm(lpips_vals, higher_better=False)  # lower is better
cost_n  = norm(gflops,    higher_better=False)   # lower is better

print("=" * 85)
print("NORMALIZED SCORES [0-1] (higher = better)")
print(f"{'Method':<12} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'Cost':>8}")
print("-" * 85)
for i, n in enumerate(names):
    print(f"{n:<12} {psnr_n[i]:>8.3f} {ssim_n[i]:>8.3f} {lpips_n[i]:>8.3f} {cost_n[i]:>8.3f}")

# ── Score 1: Pure Quality (equal weight PSNR/SSIM/LPIPS) ──
quality = (psnr_n + ssim_n + lpips_n) / 3.0

# ── Score 2: Quality-Efficiency (quality weighted by cost) ──
# Method A: quality * cost_normalized  (simple product)
qe_product = quality * cost_n

# Method B: quality / log2(GFLOPs)  (log-scale cost penalty)
qe_log = quality / np.log2(gflops)

# Method C: Weighted composite (quality 70% + cost 30%)
composite_70_30 = 0.7 * quality + 0.3 * cost_n

# Method D: quality / GFLOPs^0.2  (mild cost penalty)
qe_mild = quality / (gflops ** 0.2)

print()
print("=" * 85)
print("COMPOSITE SCORES (higher = better)")
print(f"{'Method':<12} {'Quality':>8} {'Q*Cost':>8} {'Q/log2C':>8} {'70Q+30C':>8} {'Q/C^0.2':>8}")
print("-" * 85)
for i, n in enumerate(names):
    print(f"{n:<12} {quality[i]:>8.3f} {qe_product[i]:>8.3f} {qe_log[i]:>8.3f} {composite_70_30[i]:>8.3f} {qe_mild[i]:>8.3f}")

# ── Rank each score ──
print()
print("=" * 85)
print("RANKINGS (1 = best)")
scores = {
    "Quality":  quality,
    "Q*Cost":   qe_product,
    "Q/log2C":  qe_log,
    "70Q+30C":  composite_70_30,
    "Q/C^0.2":  qe_mild,
}
print(f"{'Method':<12}", end="")
for sname in scores:
    print(f" {sname:>8}", end="")
print()
print("-" * 85)
for i, n in enumerate(names):
    print(f"{n:<12}", end="")
    for sname, sv in scores.items():
        rank = int(np.sum(sv > sv[i]) + 1)
        print(f" {rank:>8}", end="")
    print()

# ── All-ISO analysis (from available data) ──
print()
print("=" * 85)
print("MULTI-ISO ANALYSIS (Quality score per ISO)")
print()

# Full data: ISO 400 only has all methods. Let me also include kodim01 single-image data
# for multiple ISOs from timing_test
all_iso_data = {
    400: {
        "BM3D-PC":  {"psnr": 32.35, "ssim": 0.8874, "lpips": 0.3400},
        "BM3D-CFA": {"psnr": 33.65, "ssim": 0.9055, "lpips": 0.3052},
        "DnCNN-B":  {"psnr": 32.82, "ssim": 0.9061, "lpips": 0.2951},
        "DRUNet":   {"psnr": 35.27, "ssim": 0.9431, "lpips": 0.2175},
        "Ours(V2)": {"psnr": 33.16, "ssim": 0.9088, "lpips": 0.2927},
    },
}

for iso, idata in all_iso_data.items():
    ns = list(idata.keys())
    p = np.array([idata[n]["psnr"] for n in ns])
    s = np.array([idata[n]["ssim"] for n in ns])
    l = np.array([idata[n]["lpips"] for n in ns])
    g = np.array([methods[n]["gflops"] for n in ns])

    pn = norm(p, True); sn = norm(s, True); ln = norm(l, False); cn = norm(g, False)
    q = (pn + sn + ln) / 3.0
    eff = 0.7 * q + 0.3 * cn

    print(f"ISO {iso}:")
    print(f"  {'Method':<12} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'Quality':>8} {'70Q+30C':>8}")
    for i, n in enumerate(ns):
        print(f"  {n:<12} {p[i]:>7.2f} {s[i]:>7.4f} {l[i]:>7.4f} {q[i]:>8.3f} {eff[i]:>8.3f}")
    print()

# ── Final verdict ──
print("=" * 85)
print("VERDICT (ISO 400, Kodak-24)")
print()
best_q = names[np.argmax(quality)]
best_eff = names[np.argmax(composite_70_30)]
best_bang = names[np.argmax(qe_product)]

print(f"  Best Quality:              {best_q:<12} (score: {quality.max():.3f})")
print(f"  Best Quality+Efficiency:   {best_eff:<12} (score: {composite_70_30.max():.3f})")
print(f"  Best Bang-for-Buck:        {best_bang:<12} (score: {qe_product.max():.3f})")
print()

# Show Ours specifically
idx_ours = names.index("Ours(V2)")
print(f"  Ours(V2) position:")
for sname, sv in scores.items():
    rank = int(np.sum(sv > sv[idx_ours]) + 1)
    print(f"    {sname:<12}: #{rank}/5 (score: {sv[idx_ours]:.3f})")
