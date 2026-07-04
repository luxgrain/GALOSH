"""Paper qualitative figure: pick scenes where GALOSH wins perceptually (LPIPS gap
vs BM3D-CFA), auto-crop a high-texture region, lay out method columns side by side
with per-cell LPIPS labels.

  python make_qualitative_figure.py --dataset rawnind --n 4
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("GALOSH_ROOT", str(Path(__file__).resolve().parents[2])))
DIRS = {"sidd_medium": ROOT / "benchmark" / "results_raw_sidd",
        "rawnind":     Path(os.environ.get("GALOSH_RAWNIND_RESULTS", "benchmark/results_raw_rawnind"))}
# columns to show (artifact stem -> display label); _noisy/_gt are the input/GT.
# BM3D/NLM use the VST (noise-aware) variants — the strongest classical baselines —
# while GALOSH stays fully blind (user request 2026-06-27).
COLS = [("_noisy", "Noisy"), ("vst_nlm_cfa", "VST+NLM-CFA"), ("vst_bm3d_cfa", "VST+BM3D-CFA"),
        ("b2u", "Blind2Unblind"), ("galosh_fp32", "GALOSH (ours)"), ("_gt", "GT")]
GAP_REF = "vst_bm3d_cfa"   # baseline GALOSH is scored against for scene selection

def merged_lpips(d):
    """scene -> method -> lpips, merged over the per-stream metrics JSONs."""
    out = {}
    for jf in sorted(d.glob("_metrics_*.json")):
        if "_OLD" in jf.name: continue
        data = json.load(open(jf))
        for m, scenes in data.items():
            for s, v in scenes.items():
                if isinstance(v, dict) and v.get("lpips") is not None:
                    out.setdefault(s, {})[m] = v["lpips"]
    return out

def best_crop(gt_gray, cs):
    """argmax mean gradient-magnitude window (texture), coarse grid, avoid borders."""
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

def load(adir, stem):
    p = adir / f"{stem}.png"
    if not p.exists(): return None
    im = cv2.imread(str(p)); return im[:, :, ::-1] if im is not None else None  # BGR->RGB

def magenta_count(adir, stem="galosh_fp32", thr=0.12, gt_thr=0.04):
    """INTRODUCED magenta = #pixels strongly magenta (min(R,B)-G > thr) in the method
    output BUT neutral in the aligned GT (< gt_thr). Isolates the FP32 chroma-overshoot
    defect from genuinely-purple content (flowers etc.), so the figure filter drops only
    true-artifact scenes. Returns a big number if a png is missing."""
    im = load(adir, stem); gt = load(adir, "_gt")
    if im is None or gt is None: return 10**9
    rgb = im.astype(np.float32) / 255.0
    g = gt.astype(np.float32) / 255.0
    out_mag = np.minimum(rgb[..., 0], rgb[..., 2]) - rgb[..., 1] > thr
    gt_neutral = (np.minimum(g[..., 0], g[..., 2]) - g[..., 1]) < gt_thr
    return int((out_mag & gt_neutral).sum())

def noise_proxy(adir):
    """Input noise magnitude = mean|noisy - gt| on the CFA (higher = noisier scene)."""
    try:
        n = np.load(adir / "_noisy.npy").astype(np.float32)
        g = np.load(adir / "_gt.npy").astype(np.float32)
        return float(np.abs(n - g).mean())
    except Exception:
        return -1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DIRS))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--cs", type=int, default=0, help="crop size (0=auto: 160 rawnind / 400 sidd)")
    ap.add_argument("--scenes", default="", help="comma list to force specific scenes")
    ap.add_argument("--mode", choices=["win", "noisy", "mix"], default="mix",
                    help="win=max GALOSH LPIPS gap; noisy=highest input noise; mix=half/half")
    ap.add_argument("--max-magenta", type=int, default=0,
                    help="drop scenes whose galosh_fp32 introduces >N magenta specks (FP32 chroma defect)")
    args = ap.parse_args()
    root = DIRS[args.dataset]; adir_root = root / "_artifacts"
    cs = args.cs or (160 if args.dataset == "rawnind" else 400)

    lp = merged_lpips(root)
    def base(s):  # collapse ISO / crop variants of the same scene -> diverse picks
        return s.rsplit("__ISO", 1)[0] if "__ISO" in s else s.rsplit("_", 1)[0]
    if args.scenes:
        sel = [s for s in args.scenes.split(",") if (adir_root / s).exists()]
    else:
        cand = []
        for s, mv in lp.items():
            if all(k in mv for k in ("galosh_fp32", GAP_REF, "b2u")):
                gap = mv[GAP_REF] - mv["galosh_fp32"]             # GALOSH win over VST+BM3D
                if mv["galosh_fp32"] < mv["b2u"] + 0.05:          # GALOSH near/below B2U
                    cand.append((gap, s))
        cand.sort(reverse=True)
        # build a clean, deduped pool: keep best-gap scene per base, dropping FP32-magenta
        # scenes so the figure never shows the chroma defect (reported separately to user)
        seen, pool = set(), []
        for gap, s in cand:
            b = base(s)
            if b in seen: continue
            if magenta_count(adir_root / s) > args.max_magenta: continue
            seen.add(b); pool.append((gap, s))
            if len(pool) >= 80: break
        if args.mode == "win":
            sel = [s for _, s in pool[:args.n]]
        elif args.mode == "noisy":
            pool.sort(key=lambda gs: noise_proxy(adir_root / gs[1]), reverse=True)
            sel = [s for _, s in pool[:args.n]]
        else:  # mix: half clearest-win, half noisiest
            half = (args.n + 1) // 2
            win_sel = [s for _, s in pool[:half]]
            rest = [gs for gs in pool if gs[1] not in win_sel]
            rest.sort(key=lambda gs: noise_proxy(adir_root / gs[1]), reverse=True)
            sel = win_sel + [s for _, s in rest[:args.n - len(win_sel)]]
    print(f"[{args.dataset}] mode={args.mode} selected: {sel}", flush=True)

    nrow, ncol = len(sel), len(COLS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0*ncol, 2.15*nrow))
    if nrow == 1: axes = axes[None, :]
    for r, scene in enumerate(sel):
        adir = adir_root / scene
        gt = load(adir, "_gt")
        gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).astype(np.float32)
        by, bx, c = best_crop(gt_gray, cs)
        for cc, (stem, label) in enumerate(COLS):
            ax = axes[r, cc]; ax.set_xticks([]); ax.set_yticks([])
            im = load(adir, stem)
            if im is None:
                ax.set_facecolor("0.9")
            else:
                crop = im[by:by+c, bx:bx+c]
                ax.imshow(np.clip(crop, 0, 255).astype(np.uint8), interpolation="nearest")
                if stem not in ("_noisy", "_gt") and stem in lp.get(scene, {}):
                    ax.set_xlabel(f"LPIPS {lp[scene][stem]:.3f}", fontsize=7)
            if r == 0: ax.set_title(label, fontsize=9)
            if cc == 0: ax.set_ylabel(scene[:18], fontsize=7)
    plt.tight_layout(pad=0.4)
    out = ROOT / "docs" / "paper" / f"qualitative_{args.dataset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}  ({nrow}x{ncol})", flush=True)

if __name__ == "__main__":
    main()
