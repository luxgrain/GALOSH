#!/usr/bin/env python3
"""Merge PG-noise track shards and emit the ISO-binned summary.

Reads results_set8_pgnoise/_metrics_pg_{420,444}_*.json, merges by seq, aggregates
each method across seqs per ISO, writes _metrics_pg_{420,444}.json +
SUMMARY.md.  PG-noise track = realistic signal-dependent noise on Set8 (the
middle ground between the flat-AWGN Set8 track and the CRVD real-noise track).
"""
import json
import re
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "benchmark" / "results_set8_pgnoise"
ISOS = ["ISO400", "ISO800", "ISO1600", "ISO3200", "ISO6400", "ISO12800",
        "ISO25600"]
GAL = ["galosh-cpu-fit", "galosh-cpu-hold", "galosh-vk-fit", "galosh-vk-hold"]
M420 = ["noisy"] + GAL + ["bm3d1", "bm3d1b", "bm3d1bg", "vbm3d", "vbm3db",
                          "vbm3dbg", "knl", "smdegrain", "hqdn3d"]
M444 = ["noisy"] + GAL + ["bm3d1", "bm3d1b", "bm3d1bg", "knl"]
LAB = {"noisy": "noisy (developed)",
       "galosh-cpu-fit": "GALOSH cpu-fit", "galosh-cpu-hold": "GALOSH cpu-hold",
       "galosh-vk-fit": "GALOSH vk-fit", "galosh-vk-hold": "GALOSH vk-hold",
       "bm3d1": "BM3D (sigma-oracle)", "bm3d1b": "BM3D (sigma-blind MAD)",
       "bm3d1bg": "BM3D (sigma-blind MAD, 0.5 floor)",
       "vbm3d": "V-BM3D (sigma-oracle temporal)",
       "vbm3db": "V-BM3D (sigma-blind MAD temporal)",
       "vbm3dbg": "V-BM3D (sigma-blind MAD, 0.5 floor, temporal)",
       "knl": "KNL d=1 (temporal)", "smdegrain": "SMDegrain (temporal)",
       "hqdn3d": "hqdn3d (untuned default)"}


# variant -> (tag matcher, merged-output suffix).  The campaign shards are
# tagged c{mode}_{seq} (core) and m{mode}_{seq} (cmp), e.g. c420_tractor,
# m444_snowboard.  Match that EXACT scheme so stale/dev shards (c, cmpchk,
# m420all, ...) and this script's own merged outputs (_metrics_pg_420_cmp
# whose tag "cmp" also starts with 'c') are never re-ingested.
VARIANTS = {
    "core": (lambda t: re.match(r"^c(420|444)_.+", t), ""),
    "cmp":  (lambda t: re.match(r"^m(420|444)_.+", t), "_cmp"),
}


def merge(mode, variant="base"):
    match, suffix = VARIANTS[variant]
    m = {}
    for f in sorted(OUT.glob(f"_metrics_pg_{mode}_*.json")):
        tag = f.stem.split(f"_metrics_pg_{mode}_", 1)[1]
        if not match(tag):
            continue
        for seq, isos in json.loads(f.read_text()).items():
            m.setdefault(seq, {}).update({k: v for k, v in isos.items()
                                          if k != "n_frames"})
            if "n_frames" in isos:
                m[seq]["n_frames"] = isos["n_frames"]
    if m:
        (OUT / f"_metrics_pg_{mode}{suffix}.json").write_text(
            json.dumps(m, indent=1))
    return m


def cell(m, seqs, method, iso, key):
    v = [m[s][iso][method][key] for s in seqs
         if iso in m[s] and method in m[s].get(iso, {})
         and key in m[s][iso][method]]
    return float(np.mean(v)) if v else None


def main():
    L = ["# PG-noise track — realistic signal-dependent noise on Set8 (summary)\n",
         "[v3 2026-07-15] Noise = SMARTPHONE trimmed-median Poisson-Gaussian "
         "curve (SIDD per-image NLFs, 160 data rows, per-phone vote, trimmed "
         "median of GP/S6/N6; a=0.005763 b=6.34e-5 @ISO1600; ISO400..25600 ≈ "
         "AWGN σ10..50), injected as an ISP-passed DIFFERENCE "
         "(noisy = clean + [ISP(raw+n) − ISP(raw)]) so the noise carries "
         "demosaic/WB correlation while **GT = the pristine Set8 frame** "
         "(no roundtrip).  GALOSH via the frameserver DLL (420 chroma-R fix "
         "in cpu AND vk); 4 variants engine{cpu,vulkan} x noise{fit,hold}.  "
         "bm3d1b/vbm3db = sigma-BLIND twins (Donoho MAD on the noisy planes, "
         "UNGUARDED — on H.264 low-ISO material the chroma MAD can hit 0 and "
         "the V-BM3D pipeline breaks catastrophically; that is an estimator+"
         "pipeline failure, not a general blind-BM3D property).  Per-ISO "
         "GALOSH-vs-blind-BM3D PSNR wins are consistent (6/8 seqs) but n=8 "
         "means 95% CIs cross 0 on the 420 track; the 444 track is larger "
         "and more consistent.  bm3d1bg/vbm3dbg = the GUARDED blind twins "
         "(MAD clamped to a 0.5 per-plane floor): re-run where the floor "
         "engages (any zero-MAD plane — all cmp-444 cells, since H.264's "
         "4:2:0 chroma upsampling leaves no finest-scale Haar energy, and 11 "
         "low-ISO cmp-420 cells); elsewhere the floor is inactive so the "
         "sigma input is IDENTICAL to the unguarded twin and entries are "
         "copied, not re-run.  Seed disclosure: the noise seed is "
         "7000+ISO, shared by all sequences at a given ISO (content-"
         "independent draws repeat across seqs; realizations differ only "
         "through the signal-dependent variance).\n"]
    VLAB = {"core": "core (phone-median PG noise, no compression)",
            "cmp": "cmp (core noisy + H.264 CRF23; GT uncompressed)"}
    for variant in ("core", "cmp"):
        for mode, methods in (("420", M420), ("444", M444)):
            m = merge(mode, variant)
            if not m:
                continue
            seqs = sorted(s for s in m if "n_frames" in m[s])
            L.append(f"## [{VLAB[variant]}] {mode} track — PSNR / LPIPS, "
                     f"mean over {len(seqs)} seqs\n")
            L.append("| method | " + " | ".join(ISOS) + " |")
            L.append("|---|" + "---|" * len(ISOS))
            for meth in methods:
                cells = []
                for iso in ISOS:
                    p = cell(m, seqs, meth, iso, "psnr")
                    lp = cell(m, seqs, meth, iso, "lpips")
                    cells.append("--" if p is None else f"{p:.2f} / {lp:.3f}")
                L.append(f"| {LAB[meth]} | " + " | ".join(cells) + " |")
            L.append("")
            if mode == "420":
                for iso in ISOS:
                    L.append(f"### [{variant}] {iso} — all metrics\n")
                    L.append("| method | PSNR | SSIM | LPIPS | DISTS | NIQE "
                             "| Cr |")
                    L.append("|---|---|---|---|---|---|---|")
                    for meth in methods:
                        c = [cell(m, seqs, meth, iso, k) for k in
                             ("psnr", "ssim", "lpips", "dists", "niqe",
                              "cr_psnr")]
                        fmt = ["--" if x is None else (f"{x:.2f}" if i in (0, 5)
                               else f"{x:.4f}") for i, x in enumerate(c)]
                        L.append(f"| {LAB[meth]} | " + " | ".join(fmt) + " |")
                    L.append("")
    (OUT / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print("saved:", OUT / "SUMMARY.md")


if __name__ == "__main__":
    main()
