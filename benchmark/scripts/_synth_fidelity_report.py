#!/usr/bin/env python3
"""Generator-fidelity 3-way comparison (2026-07-17).

real (_metrics_crvd) vs imx385-synth (_metrics_crvdsynth_*) vs sigma-matched
generic phone curve (_metrics_crvdsynthphone_*), same content, same methods,
same measurement pipeline.  Emits per-(method, ISO) PSNR & LPIPS side-by-side
+ per-method fidelity deltas + per-ISO method-rank correlation (Spearman).
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "benchmark" / "results_crvd"
ISOS = ["ISO1600", "ISO3200", "ISO6400", "ISO12800", "ISO25600"]
METHODS = ["noisy", "galosh-cpu-fit", "galosh-cpu-hold", "galosh-vk-fit",
           "galosh-vk-hold", "bm3d1", "bm3d1b", "vbm3d", "vbm3db", "knl",
           "hqdn3d"]


def merge(stem):
    m = {}
    files = sorted(OUT.glob(f"{stem}_*.json"))
    if (OUT / f"{stem}.json").exists():
        files = [OUT / f"{stem}.json"] + files
    for f in files:
        # exclude longer stems (e.g. crvdsynth* when merging crvd_*)
        for scene, isos in json.loads(f.read_text()).items():
            if scene == "_env":
                continue
            dst = m.setdefault(scene, {})
            for iso, meth in isos.items():
                if isinstance(meth, dict) and isinstance(dst.get(iso), dict):
                    dst[iso].update(meth)
                else:
                    dst[iso] = meth
    return m


def cell(m, meth, iso, key):
    v = [m[s][iso][meth][key] for s in m
         if iso in m[s] and meth in m[s][iso] and key in m[s][iso][meth]]
    return float(np.mean(v)) if v else None


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    real = json.loads((OUT / "_metrics_crvd.json").read_text())
    real = {k: v for k, v in real.items() if k != "_env"}
    syn = merge("_metrics_crvdsynth")
    # merge("_metrics_crvdsynth") glob also catches synthphone files; filter:
    syn = {}
    for f in sorted(OUT.glob("_metrics_crvdsynth_c*.json")):
        for scene, isos in json.loads(f.read_text()).items():
            if scene != "_env":
                syn.setdefault(scene, {}).update(isos)
    pho = {}
    for f in sorted(OUT.glob("_metrics_crvdsynthphone_p*.json")):
        for scene, isos in json.loads(f.read_text()).items():
            if scene != "_env":
                pho.setdefault(scene, {}).update(isos)

    for key, prec, worse in (("psnr", 2, -1), ("lpips", 3, +1)):
        print(f"\n## {key.upper()} - real / imx385-synth / phone-match "
              f"(delta vs real)")
        print("| method | " + " | ".join(i.replace('ISO', '') for i in ISOS)
              + " |")
        print("|---|" + "---|" * len(ISOS))
        for meth in METHODS:
            cells = []
            for iso in ISOS:
                r = cell(real, meth, iso, key)
                s = cell(syn, meth, iso, key)
                p = cell(pho, meth, iso, key)
                if r is None:
                    cells.append("--")
                else:
                    st = f"{r:.{prec}f}"
                    st += f" / {s:.{prec}f}" if s is not None else " / --"
                    st += f" / {p:.{prec}f}" if p is not None else " / --"
                    cells.append(st)
            print(f"| {meth} | " + " | ".join(cells) + " |")

    print("\n## fidelity summary (mean |delta| vs real over methods x ISOs)")
    for name, m in (("imx385-synth", syn), ("phone-match", pho)):
        dp, dl = [], []
        for meth in METHODS:
            for iso in ISOS:
                r = cell(real, meth, iso, "psnr")
                s = cell(m, meth, iso, "psnr")
                if r is not None and s is not None:
                    dp.append(abs(r - s))
                r = cell(real, meth, iso, "lpips")
                s = cell(m, meth, iso, "lpips")
                if r is not None and s is not None:
                    dl.append(abs(r - s))
        print(f"{name:14s} mean|dPSNR|={np.mean(dp):.2f}dB "
              f"mean|dLPIPS|={np.mean(dl):.3f} (n={len(dp)})")

    print("\n## per-ISO method-rank Spearman (PSNR, vs real)")
    for name, m in (("imx385-synth", syn), ("phone-match", pho)):
        rs = []
        for iso in ISOS:
            rr = [cell(real, meth, iso, "psnr") for meth in METHODS]
            ss = [cell(m, meth, iso, "psnr") for meth in METHODS]
            ok = [i for i, (a, b) in enumerate(zip(rr, ss))
                  if a is not None and b is not None]
            if len(ok) >= 5:
                rs.append(spearman([rr[i] for i in ok], [ss[i] for i in ok]))
        print(f"{name:14s} " + " ".join(f"{r:.3f}" for r in rs))

    print("\n## GALOSH-vs-oracle-BM3D gap (galosh-cpu-fit - bm3d1, PSNR)")
    for name, m in (("real", real), ("imx385-synth", syn),
                    ("phone-match", pho)):
        gaps = []
        for iso in ISOS:
            g = cell(m, "galosh-cpu-fit", iso, "psnr")
            b = cell(m, "bm3d1", iso, "psnr")
            gaps.append(f"{g - b:+.2f}" if g is not None and b is not None
                        else "--")
        print(f"{name:14s} " + " ".join(gaps))


if __name__ == "__main__":
    main()
