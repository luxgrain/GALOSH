#!/usr/bin/env python3
"""Consolidated Set8 report: merge _metrics_444 / _metrics_420 /
_metrics_420_baselines / _speed_set8 into results_set8_awgn/SUMMARY.md.

Honest 4-axis framing (nothing is compared out of its class):
  A) blind single-frame       — GALOSH (no sigma given, no temporal)
  B) sigma-known single-frame — BM3D (measured per-plane sigma)
  C) sigma-known / calibrated temporal — V-BM3D, KNLMeansCL d=1,
     SMDegrain (frozen DAVIS-calibrated knob tables)
  D) DL published values      — V-BM4D/VNLB/DVDnet/FastDVDnet from the
     FastDVDnet paper (arXiv 1907.01361 Table 1; their own protocol)

Published-value caveat: axis D numbers are quoted, not re-run; our 444
track replicates their noise protocol (seeded AWGN in [0,255] RGB) but
carries frames through a YUV444P16 bt709 full-range round trip.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmark" / "results_set8_awgn"
SIGMAS = [10, 20, 30, 40, 50]

# FastDVDnet paper, arXiv 1907.01361, Table 1 (Set8 mean PSNR, sigma-known)
PUBLISHED = {  # name -> (per-sigma PSNR, axis-D sub-label)
    "V-BM4D":     ({10: 36.05, 20: 32.19, 30: 30.00, 40: 28.48, 50: 27.33},
                   "classical temporal"),
    "VNLB":       ({10: 37.26, 20: 33.72, 30: 31.74, 40: 30.39, 50: 29.24},
                   "classical temporal"),
    "DVDnet":     ({10: 36.08, 20: 33.49, 30: 31.79, 40: 30.55, 50: 29.56},
                   "temporal DL"),
    "FastDVDnet": ({10: 36.44, 20: 33.43, 30: 31.68, 40: 30.46, 50: 29.53},
                   "temporal DL"),
}

GALOSH_VARIANTS = ["galosh-cpu-fit", "galosh-cpu-hold",
                   "galosh-vk-fit", "galosh-vk-hold"]
BASELINES = ["bm3d1", "bm3d1b", "vbm3d", "vbm3db", "hqdn3d", "knl",
             "smdegrain"]
# [T] marks TEMPORAL (multi-frame) methods; unmarked = single-frame spatial.
# Label policy (2026-07-16): say exactly where sigma comes from — no "blind"
# shorthand ("MAD estimate" = Donoho MAD on the noisy input fed to a
# sigma-required filter).  GALOSH takes no sigma input (built-in estimation).
AXIS = {  # display name, axis label
    "galosh-cpu-fit":  ("GALOSH cpu-fit",  "A no-sigma-input (built-in est.) single-frame"),
    "galosh-cpu-hold": ("GALOSH cpu-hold", "A no-sigma-input (built-in est.) single-frame"),
    "galosh-vk-fit":   ("GALOSH vk-fit",   "A no-sigma-input (built-in est.) single-frame"),
    "galosh-vk-hold":  ("GALOSH vk-hold",  "A no-sigma-input (built-in est.) single-frame"),
    "bm3d1":     ("BM3D (sigma: measured oracle)", "B sigma measured single-frame"),
    "bm3d1b":    ("BM3D (sigma: MAD estimate)", "B' sigma MAD-estimated single-frame"),
    "bm3d1bg":   ("BM3D (sigma: MAD estimate + 0.5 floor guard)",
                  "B' sigma MAD-estimated single-frame"),
    "vbm3d":     ("[T] V-BM3D r=2 (sigma: measured oracle)",  "C sigma measured temporal"),
    "vbm3db":    ("[T] V-BM3D r=2 (sigma: MAD estimate)", "C' sigma MAD-estimated temporal"),
    "vbm3dbg":   ("[T] V-BM3D r=2 (sigma: MAD estimate + 0.5 floor guard)",
                  "C' sigma MAD-estimated temporal"),
    "knl":       ("[T] KNL d=1 (h: frozen DAVIS table)", "C calibrated temporal"),
    "smdegrain": ("[T] SMDegrain (thSAD: frozen DAVIS table)", "C calibrated temporal"),
    "hqdn3d":    ("[T] hqdn3d default", "untuned reference"),
}
METRICS = ["psnr", "ssim", "lpips", "dists", "niqe"]


def seqs_of(d):
    return sorted(s for s in d if isinstance(d[s], dict) and "n_frames" in d[s])


def mean_of(d, variant, sigma, key):
    vals = [d[s][f"s{sigma}"][variant][key] for s in seqs_of(d)
            if variant in d[s].get(f"s{sigma}", {})]
    return float(np.mean(vals)) if vals else None


def table_5metric(sources, sigma):
    """sources = list of (json_dict, variant_key); full 5-metric rows."""
    lines = ["| method (axis) | PSNR | SSIM | LPIPS | DISTS | NIQE |",
             "|---|---|---|---|---|---|"]
    for d, v in sources:
        name, ax = AXIS[v]
        cells = []
        for m in METRICS:
            x = mean_of(d, v, sigma, m)
            cells.append("--" if x is None else
                         (f"{x:.2f}" if m == "psnr" else f"{x:.4f}"))
        lines.append(f"| {name} ({ax}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    m444 = json.loads((OUT / "_metrics_444.json").read_text())
    m420 = json.loads((OUT / "_metrics_420.json").read_text())
    mbl = json.loads((OUT / "_metrics_420_baselines.json").read_text())
    f4 = OUT / "_metrics_444_baselines.json"
    mbl4 = json.loads(f4.read_text()) if f4.exists() else None

    def merge_shard(base, path):     # fold add-on method shards (e.g. blind
        if base is None or not path.exists():   # BM3D twins) into the base
            return base
        for s, cells in json.loads(path.read_text()).items():
            for k, v in cells.items():
                if k == "n_frames":
                    continue
                base.setdefault(s, {}).setdefault(k, {}).update(v)
        return base
    mbl = merge_shard(mbl, OUT / "_metrics_420_baselines_blind.json")
    mbl4 = merge_shard(mbl4, OUT / "_metrics_444_baselines_blind.json")
    # [2026-07-16 unification] add-on shards: vbm3d-family/hqdn3d on 444,
    # bg guard-twin copies on both lanes (sharded -> glob)
    for f in sorted(OUT.glob("_metrics_420_baselines_bg*.json")):
        mbl = merge_shard(mbl, f)
    for f in sorted(OUT.glob("_metrics_444_baselines_vb*.json")):
        mbl4 = merge_shard(mbl4, f)
    for f in sorted(OUT.glob("_metrics_444_baselines_bg*.json")):
        mbl4 = merge_shard(mbl4, f)
    speed = None
    sp = OUT / "_speed_set8.json"
    if sp.exists():
        speed = json.loads(sp.read_text())

    L = []
    L.append("# Set8 video benchmark — consolidated summary\n")
    L.append("Positioning: practical validation of GALOSH-frameserver + "
             "future Doom9 material (not a paper claim).  8 sequences "
             "(4 derf capped at 85 frames + 4 GoPro), seeded AWGN "
             "sigma in {10..50} added in [0,255] RGB.  GALOSH runs "
             "**fully blind** (no sigma given); axis B/C methods receive "
             "measured sigma or frozen DAVIS-calibrated knob tables "
             "(never tuned on Set8).\n")

    # ---- panel 1: 420 track ----
    L.append("## 420 track (YUV420P8 bt709 limited left; reference = "
             "GT-420 via the same upsampler)\n")
    L.append("PSNR / LPIPS, mean over 8 sequences:\n")
    hdr = "| method (axis) | " + " | ".join(f"s{s}" for s in SIGMAS) + " |"
    L.append(hdr)
    L.append("|---|" + "---|" * len(SIGMAS))
    # [2026-07-17 fairness policy] BM3D family shown as sigma-oracle (upper
    # bound) + MAD-estimate-with-guard (realistic deployment).  Unguarded
    # MAD twins + smdegrain fully archived (dropped_methods_20260717).
    order = ([(m420, v) for v in GALOSH_VARIANTS] +
             [(mbl, "bm3d1"), (mbl, "bm3d1bg"),
              (mbl, "vbm3d"), (mbl, "vbm3dbg"),
              (mbl, "knl"), (mbl, "hqdn3d")])
    for d, v in order:
        name, ax = AXIS[v]
        cells = []
        for s in SIGMAS:
            p = mean_of(d, v, s, "psnr")
            l = mean_of(d, v, s, "lpips")
            cells.append("--" if p is None else f"{p:.2f} / {l:.3f}")
        L.append(f"| {name} ({ax}) | " + " | ".join(cells) + " |")
    L.append("")
    for s in SIGMAS:
        L.append(f"### 420 track, all 5 metrics, sigma {s}\n")
        L.append(table_5metric(order, s))
        L.append("")

    # ---- panel 2: 444 track (+ measured baselines) + published ----
    L.append("## 444 track (YUV444P16 full; FastDVDnet-comparable protocol)"
             "\n")
    L.append("PSNR / LPIPS, mean over 8 sequences.  The BM3D family, "
             "V-BM3D family (2026-07-16 unification), knl and hqdn3d are "
             "run locally on the identical 444 inputs (sigma = measured "
             "per-plane std on the 444P16 lattice; knl h = frozen table); "
             "axis D rows are quoted from the FastDVDnet paper "
             "(arXiv 1907.01361 Table 1), not re-run:\n")
    L.append(hdr)
    L.append("|---|" + "---|" * len(SIGMAS))
    order444 = [(m444, v) for v in GALOSH_VARIANTS]
    if mbl4:
        order444 += [(mbl4, "bm3d1"), (mbl4, "bm3d1bg"),
                     (mbl4, "vbm3d"), (mbl4, "vbm3dbg"),
                     (mbl4, "knl"), (mbl4, "hqdn3d")]
    for d, v in order444:
        name, ax = AXIS[v]
        cells = []
        for s in SIGMAS:
            p = mean_of(d, v, s, "psnr")
            l = mean_of(d, v, s, "lpips")
            cells.append("--" if p is None else f"{p:.2f} / {l:.3f}")
        L.append(f"| {name} ({ax}) | " + " | ".join(cells) + " |")
    for name, (tab, sub) in PUBLISHED.items():
        cells = [f"{tab[s]:.2f}" for s in SIGMAS]
        L.append(f"| {name} (D published, sigma-known {sub}) | "
                 + " | ".join(cells) + " |")
    L.append("")
    if mbl4:
        for s in SIGMAS:
            L.append(f"### 444 track, all 5 metrics, sigma {s}\n")
            L.append(table_5metric(order444, s))
            L.append("")
        L.append("### Track-relative gap (GALOSH cpu-fit minus baseline, "
                 "PSNR; the 420 column sinking further = the 420-native "
                 "chroma-path deficit)\n")
        L.append("| sigma | vs BM3D @444 | vs BM3D @420 | extra loss @420 "
                 "| vs KNL @444 | vs KNL @420 | extra loss @420 |")
        L.append("|---|---|---|---|---|---|---|")
        for s in SIGMAS:
            g4 = mean_of(m444, "galosh-cpu-fit", s, "psnr")
            g2 = mean_of(m420, "galosh-cpu-fit", s, "psnr")
            gb4 = g4 - mean_of(mbl4, "bm3d1", s, "psnr")
            gb2 = g2 - mean_of(mbl, "bm3d1", s, "psnr")
            gk4 = g4 - mean_of(mbl4, "knl", s, "psnr")
            gk2 = g2 - mean_of(mbl, "knl", s, "psnr")
            L.append(f"| {s} | {gb4:+.2f} | {gb2:+.2f} | {gb2-gb4:+.2f} "
                     f"| {gk4:+.2f} | {gk2:+.2f} | {gk2-gk4:+.2f} |")
        L.append("")

    # ---- panel 3: speed ----
    if speed:
        L.append("## Speed (serial in-order pulls, net of source cost, "
                 "RTX 4070 Ti; ms/frame)\n")
        L.append("| method / lane | 540p | 1080p | 2160p |")
        L.append("|---|---|---|---|")
        vsrows = ["cpu-fit", "cpu-hold", "vk-fit", "vk-hold"]  # speed json keys (unchanged)
        for v in vsrows:
            cells = []
            for res in ["540p", "1080p", "2160p"]:
                x = speed.get(res, {}).get("vs", {}).get(v)
                cells.append("--" if x is None else f"{x:.1f}")
            L.append(f"| GALOSH {v} (VapourSynth) | " + " | ".join(cells)
                     + " |")
        avsrows = ["galosh cpu-fit", "galosh vk-fit", "galosh vk-hold",
                   "bm3d1", "vbm3d", "hqdn3d", "knl"]
        for v in avsrows:
            cells = []
            for res in ["540p", "1080p", "2160p"]:
                x = speed.get(res, {}).get("avs", {}).get(v)
                cells.append("--" if x is None else f"{x:.1f}")
            L.append(f"| {v} (AviSynth) | " + " | ".join(cells) + " |")
        L.append("")

    L.append("Knob provenance: BM3D/V-BM3D sigma = measured per-plane "
             "std(noisy-GT) on the actual 420 planes; KNL h and SMDegrain "
             "thSAD from frozen DAVIS calibrations (_knl_h.json, "
             "_smdegrain_thsad.json); hqdn3d untouched defaults.\n")
    (OUT / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print("saved:", OUT / "SUMMARY.md")


if __name__ == "__main__":
    main()
