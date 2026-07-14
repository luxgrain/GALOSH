#!/usr/bin/env python3
"""Merge the tagged CRVD bench shards and emit the ISO-binned summary.

Reads results_crvd/_metrics_crvd_{a,b,c,d}.json (+ any untagged), merges
by scene, aggregates each method across all scenes per ISO, and writes
results_crvd/_metrics_crvd.json + SUMMARY_crvd.md.

CRVD = REAL SONY IMX385 sensor noise (signal-dependent, ISP-correlated),
the regime GALOSH's GAT is designed for — contrast with the Set8 AWGN
track where the sigma-known baselines win.  ISO is the real noise axis.
NOTE: CRVD motion is stop-and-motion, so motion-compensated temporal
methods (SMDegrain, V-BM3D) are handicapped vs continuous video.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "benchmark" / "results_crvd"
ISOS = ["ISO1600", "ISO3200", "ISO6400", "ISO12800", "ISO25600"]
METHODS = ["noisy", "galosh-cpu-fit", "galosh-cpu-hold", "galosh-vk-fit",
           "galosh-vk-hold", "galosh444", "bm3d1", "bm3d1b", "vbm3d",
           "vbm3db", "knl", "smdegrain", "hqdn3d",
           # legacy names (pre-2026-07-14 EXE runs), shown when present:
           "galosh420"]
LABEL = {"noisy": "noisy (developed input)",
         "galosh-cpu-fit": "GALOSH cpu-fit (blind, single-frame)",
         "galosh-cpu-hold": "GALOSH cpu-hold (blind, single-frame)",
         "galosh-vk-fit": "GALOSH vk-fit (blind, single-frame)",
         "galosh-vk-hold": "GALOSH vk-hold (blind, single-frame)",
         "galosh420": "GALOSH-fix 420 (blind, single-frame; legacy exe)",
         "galosh444": "GALOSH-fix 444 (blind, single-frame)",
         "bm3d1": "BM3D (sigma-ORACLE, single-frame)",
         "bm3d1b": "BM3D (sigma-BLIND MAD, single-frame)",
         "vbm3d": "V-BM3D r=2 (sigma-ORACLE, temporal*)",
         "vbm3db": "V-BM3D r=2 (sigma-BLIND MAD, temporal*)",
         "knl": "KNL d=1 (temporal*)",
         "smdegrain": "SMDegrain tr=3 (temporal*)",
         "hqdn3d": "hqdn3d (untuned default)"}


def merge():
    merged = {}
    for f in sorted(OUT.glob("_metrics_crvd_*.json")) + \
            ([OUT / "_metrics_crvd.json"] if (OUT / "_metrics_crvd.json").exists()
             else []):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for scene, isos in d.items():
            merged.setdefault(scene, {}).update(isos)
    return merged


def main():
    m = merge()
    scenes = sorted(m.keys(), key=lambda s: int(s.replace("scene", "")))
    (OUT / "_metrics_crvd.json").write_text(json.dumps(m, indent=1))

    def cell(method, iso, key):
        vals = [m[s][iso][method][key] for s in scenes
                if iso in m[s] and method in m[s][iso]
                and key in m[s][iso][method]]
        return float(np.mean(vals)) if vals else None

    L = ["# CRVD real-noise video denoising — summary\n",
         "Real SONY IMX385 sensor noise (signal-dependent, ISP-correlated) "
         "on 7-frame sequences, 11 indoor scenes x 5 ISO.  This is the "
         "regime GALOSH's GAT/Anscombe design targets — contrast with the "
         "Set8 AWGN track where sigma-known baselines win.  *Temporal "
         "methods are handicapped: CRVD motion is stop-and-motion, breaking "
         "motion compensation.  BM3D (single-frame) is the clean baseline.\n",
         "Mean over scenes present, PSNR / LPIPS:\n",
         "| method | " + " | ".join(ISOS) + " |",
         "|---|" + "---|" * len(ISOS)]
    for method in METHODS:
        cells = []
        for iso in ISOS:
            p, l = cell(method, iso, "psnr"), cell(method, iso, "lpips")
            cells.append("--" if p is None else f"{p:.2f} / {l:.3f}")
        L.append(f"| {LABEL[method]} | " + " | ".join(cells) + " |")
    L.append("")
    # full 5-metric per ISO
    for iso in ISOS:
        L.append(f"## {iso} (all metrics, scene-mean)\n")
        L.append("| method | PSNR | SSIM | LPIPS | DISTS | NIQE | Cr plane |")
        L.append("|---|---|---|---|---|---|---|")
        for method in METHODS:
            c = [cell(method, iso, k) for k in
                 ("psnr", "ssim", "lpips", "dists", "niqe", "cr_psnr")]
            fmt = ["--" if x is None else (f"{x:.2f}" if i in (0, 5)
                   else f"{x:.4f}") for i, x in enumerate(c)]
            L.append(f"| {LABEL[method]} | " + " | ".join(fmt) + " |")
        L.append("")
    n_present = sum(1 for s in scenes for iso in ISOS if iso in m[s])
    L.append(f"\n_scenes: {len(scenes)}, (scene,ISO) cells: {n_present}/55_")
    (OUT / "SUMMARY_crvd.md").write_text("\n".join(L), encoding="utf-8")
    print("saved:", OUT / "SUMMARY_crvd.md", "| scenes:", len(scenes),
          "| cells:", n_present)


if __name__ == "__main__":
    main()
