"""Paper qualitative figure for the sRGB (GALOSH-YUV) benchmark — same style as
make_qualitative_figure.py (RAW): pick the noisiest scenes (input LPIPS), auto-crop a
high-texture region, lay out method columns with per-image LPIPS labels.

Selection is by input noise only (no per-method cherry-pick); scenes where NAFNet-SIDD
diverges on near-black input (PSNR<10, documented in the table footnote) are excluded so
the figure shows the representative behaviour, not the failure cases.

  python make_yuv_qualitative.py --dataset sidd --n 3
  python make_yuv_qualitative.py --dataset rawnind --n 3
"""
import argparse, json
from pathlib import Path
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / "benchmark"
DIRS = {"sidd": ROOT / "yuv_srgb_results_full",       # full-frame run (paper table source)
        "sidd_crop": ROOT / "yuv_srgb_results",       # legacy central-1024^2 run
        "rawnind": ROOT / "rawnind_srgb_results"}
OUT  = ROOT / "raw_v2_results"   # paper directory (figures live next to the .tex)

COLS = [("noisy", "Noisy"), ("cbm3d", "CBM3D"), ("color_nlm", "Color-NLM"),
        ("guided", "Guided"), ("nafnet", "NAFNet-SIDD"), ("scunet", "SCUNet"),
        ("restormer", "Restormer-SIDD"), ("galosh_yuv_gpu_fp32", "GALOSH (ours)"),
        ("gt", "GT")]

def metrics_by_tag(d):
    """tag -> method -> {lpips, psnr} from the flat rows of _metrics.json."""
    out = {}
    for x in json.load(open(d / "_metrics.json")):
        if x.get("ok") is False or x.get("lpips") is None: continue
        out.setdefault(x["tag"], {})[x["method"]] = {"lpips": x["lpips"], "psnr": x.get("psnr")}
    return out

def base_scene(ds, tag):
    return tag.split("__ISO")[0] if ds == "rawnind" else tag.split("_")[0]

def best_crop(gt_gray, cs):
    H, W = gt_gray.shape
    cs = min(cs, H, W)
    gx = cv2.Sobel(gt_gray, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gt_gray, cv2.CV_32F, 0, 1)
    g = np.abs(gx) + np.abs(gy)
    best, by, bx = -1, 0, 0
    for y in range(0, H - cs + 1, max(1, (H - cs) // 12 or 1)):
        for x in range(0, W - cs + 1, max(1, (W - cs) // 12 or 1)):
            s = g[y:y+cs, x:x+cs].mean()
            if s > best: best, by, bx = s, y, x
    return by, bx, cs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DIRS))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--cs", type=int, default=0, help="crop (0=auto: 180 rawnind / 340 sidd)")
    ap.add_argument("--scenes", default="", help="comma list of tags to force")
    ap.add_argument("--iso-min", type=int, default=0)
    ap.add_argument("--iso-max", type=int, default=10**9,
                    help="rawnind: cap ISO so the pick is the realistic high-ISO band, "
                         "not the ISO-65535 sensor-ceiling tail where every method fails")
    a = ap.parse_args()
    d = DIRS[a.dataset]; cs = a.cs or (180 if a.dataset == "rawnind" else 340)
    M = metrics_by_tag(d)

    if a.scenes:
        picks = [s for s in a.scenes.split(",") if s]
    else:
        # noisiest first (input LPIPS), one per base scene, all methods present,
        # skip NAFNet near-black divergence cases (footnoted in the table instead)
        cand, seen = [], set()
        for tag, mm in sorted(M.items(), key=lambda kv: -(kv[1].get("_noisy", {}).get("lpips") or 0)):
            b = base_scene(a.dataset, tag)
            if b in seen: continue
            if "__ISO" in tag:
                iso = int(tag.split("__ISO")[1])
                if not (a.iso_min <= iso <= a.iso_max): continue
            if not all(c in mm for c, _ in COLS if c not in ("noisy", "gt")): continue
            if (mm.get("nafnet", {}).get("psnr") or 99) < 10: continue
            seen.add(b); cand.append(tag)
            if len(cand) >= a.n: break
        picks = cand
    print(f"[{a.dataset}] scenes: {picks}")

    nr, nc = len(picks), len(COLS)
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 2.05, nr * 2.35))
    axes = np.atleast_2d(axes)
    for r, tag in enumerate(picks):
        gt = cv2.imread(str(d / "gt" / f"{tag}.png"))[:, :, ::-1]
        by, bx, c = best_crop(cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).astype(np.float32), cs)
        for cidx, (stem, label) in enumerate(COLS):
            ax = axes[r, cidx]
            im = cv2.imread(str(d / stem / f"{tag}.png"))
            ax.imshow(im[by:by+c, bx:bx+c, ::-1]); ax.set_xticks([]); ax.set_yticks([])
            key = "_noisy" if stem == "noisy" else stem
            lp = M[tag].get(key, {}).get("lpips")
            sub = "" if stem == "gt" else (f"\nLPIPS {lp:.3f}" if lp is not None else "")
            if r == 0: ax.set_title(label, fontsize=9, fontweight="bold")
            ax.set_xlabel(sub.strip(), fontsize=7.5)
            for s in ax.spines.values():
                s.set_color("#444"); s.set_linewidth(0.6)
        iso = tag.split("__ISO")[1] if "__ISO" in tag else ""
        axes[r, 0].set_ylabel((base_scene(a.dataset, tag)[:14] + (f"\nISO {iso}" if iso else "")),
                              fontsize=7.5)
    fig.tight_layout(pad=0.4)
    out = OUT / f"qualitative_srgb_{a.dataset}.png"
    fig.savefig(out, dpi=210, bbox_inches="tight")
    print(f"saved {out}")

if __name__ == "__main__":
    main()
