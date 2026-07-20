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
# [2026-07-16 整理] galosh420 legacy-exe row removed (0 cells, never run in
# the current campaign) and galosh444 EXE side-check ARCHIVED (data + PNGs in
# _ARCHIVE/results_crvd_galosh444; stripped from the live JSONs).
# [2026-07-17 fairness policy] the fair BM3D-family comparison is
# sigma-oracle (upper bound) vs MAD-estimate + guard (realistic deployment).
# UNGUARDED MAD twins (bm3d1b/vbm3db) and smdegrain are FULLY ARCHIVED
# (_ARCHIVE/dropped_methods_20260717; collapse evidence preserved in notes).
METHODS = ["noisy", "galosh-cpu-fit", "galosh-cpu-hold", "galosh-vk-fit",
           "galosh-vk-hold", "bm3d1", "bm3d1bg", "vbm3d",
           "vbm3dbg", "knl", "hqdn3d"]
# [T] marks TEMPORAL (multi-frame) methods; unmarked = single-frame spatial.
# Label policy (2026-07-16): say exactly where sigma comes from — no "blind"
# shorthand.  "sigma: MAD estimate" = Donoho MAD run on the noisy input and
# fed to a sigma-required filter; "+ 0.5 floor guard" = the estimate is
# clamped to >= 0.5 per plane.  GALOSH takes no sigma input at all (noise
# estimation is built into the method itself).
LABEL = {"noisy": "noisy (developed input)",
         "galosh-cpu-fit": "GALOSH cpu-fit (built-in noise est.)",
         "galosh-cpu-hold": "GALOSH cpu-hold (built-in noise est.)",
         "galosh-vk-fit": "GALOSH vk-fit (built-in noise est.)",
         "galosh-vk-hold": "GALOSH vk-hold (built-in noise est.)",
         "bm3d1": "BM3D (sigma: measured oracle)",
         "bm3d1b": "BM3D (sigma: MAD estimate)",
         "bm3d1bg": "BM3D (sigma: MAD estimate + 0.5 floor guard)",
         "vbm3d": "[T] V-BM3D r=2 (sigma: measured oracle)",
         "vbm3db": "[T] V-BM3D r=2 (sigma: MAD estimate)",
         "vbm3dbg": "[T] V-BM3D r=2 (sigma: MAD estimate + 0.5 floor guard)",
         "knl": "[T] KNL d=1 (h: frozen DAVIS table)",
         "hqdn3d": "[T] hqdn3d (untuned default)"}


def merge(stem):
    """stem = '_metrics_crvd' (420 lane) or '_metrics_crvd444' (444 lane).
    Method-level DEEP merge so add-method shards (e.g. the bg-twin copies)
    never replace a whole (scene, ISO) entry; '_env' provenance skipped."""
    merged = {}
    files = sorted(OUT.glob(f"{stem}_*.json"))
    if (OUT / f"{stem}.json").exists():
        files = [OUT / f"{stem}.json"] + files
    for f in files:
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for scene, isos in d.items():
            if scene == "_env":
                continue
            dst = merged.setdefault(scene, {})
            for iso, methods in isos.items():
                if isinstance(methods, dict) and isinstance(dst.get(iso), dict):
                    dst[iso].update(methods)
                else:
                    dst[iso] = methods
    return merged


def main():
    m = merge("_metrics_crvd")
    m444 = merge("_metrics_crvd444")
    scenes = sorted(m.keys(), key=lambda s: int(s.replace("scene", "")))
    (OUT / "_metrics_crvd.json").write_text(json.dumps(m, indent=1))
    if m444:
        (OUT / "_metrics_crvd444.json").write_text(json.dumps(m444, indent=1))

    def cell(method, iso, key, src=None):
        src = m if src is None else src
        sc = sorted(src.keys(), key=lambda s: int(s.replace("scene", "")))
        vals = [src[s][iso][method][key] for s in sc
                if iso in src[s] and method in src[s][iso]
                and key in src[s][iso][method]]
        return float(np.mean(vals)) if vals else None

    L = ["# CRVD real-noise video denoising — summary\n",
         "Real SONY IMX385 sensor noise (signal-dependent, ISP-correlated) "
         "on 7-frame sequences, 11 indoor scenes x 5 ISO.  This is the "
         "regime GALOSH's GAT/Anscombe design targets — contrast with the "
         "Set8 AWGN track where sigma-known baselines win.  [T] marks "
         "TEMPORAL (multi-frame) methods; they are handicapped here: CRVD "
         "motion is stop-and-motion, breaking motion compensation.  BM3D "
         "(single-frame) is the clean baseline.\n",
         "KNOWN CAVEATS (2026-07-16 audit; archival 2026-07-17):\n"
         "- Fairness policy: BM3D family = sigma-oracle (upper bound) + "
         "MAD-estimate-with-guard (realistic deployment).  Unguarded MAD "
         "twins (bm3d1b/vbm3db) fully archived; their reference numbers "
         "(e.g. 420 ISO25600: bm3d1b 26.23 / vbm3db 26.01 vs guarded "
         "identical here — floor never engages on CRVD) live in "
         "_ARCHIVE/dropped_methods_20260717.\n"
         "- [T] SMDegrain was FULLY ARCHIVED (2026-07-17; also: real-world "
         "use on sources this noisy presumes a prefilter, which this bench "
         "did not provide): "
         "at ISO12800 and ISO25600 it was a NO-OP — the frozen "
         "DAVIS thSAD table (calibrated for AWGN sigma>=10, indexed by "
         "measured sigma_Y 3.2-12.4 -> clamped to the sigma=10 entry, "
         "thSAD=432) sits below the real-noise SAD floor, so MDegrain "
         "rejects every block and the output is BIT-IDENTICAL to noisy "
         "(verified 11/11 scenes at both ISOs).  Those rows measure the "
         "calibration-transfer failure, not denoising.\n"
         "- [T] KNL: all 55 cells clamp to the sigma=10 edge of the frozen "
         "DAVIS h-table (CRVD sigma is below the calibration range), i.e. "
         "h=3.0 everywhere — untuned for this regime by policy, but the "
         "clamp means the knob never adapts across the ISO axis.\n"
         "- CRVD developed-domain noise is mild in absolute terms: "
         "sigma_sRGB ~= 4.5 / 6.2 / 8.6 / 12.1 / 17.6 at ISO 1600..25600 "
         "(~= AWGN sigma 5..18), far below the smartphone-curve levels "
         "with the same ISO labels.\n",
         ]
    lanes = [("420 lane (production video path)", m)]
    if m444:
        lanes.append(("444 lane (Set8-444 protocol; 2026-07-16 unification)",
                      m444))
    for lane_name, src in lanes:
        L.append(f"# {lane_name}\n")
        L.append("Mean over scenes present, PSNR / LPIPS:\n")
        L.append("| method | " + " | ".join(ISOS) + " |")
        L.append("|---|" + "---|" * len(ISOS))
        for method in METHODS:
            cells = []
            for iso in ISOS:
                p = cell(method, iso, "psnr", src)
                l = cell(method, iso, "lpips", src)
                cells.append("--" if p is None else f"{p:.2f} / {l:.3f}")
            L.append(f"| {LABEL[method]} | " + " | ".join(cells) + " |")
        L.append("")
        # full 5-metric per ISO
        for iso in ISOS:
            L.append(f"## [{lane_name.split()[0]}] {iso} "
                     f"(all metrics, scene-mean)\n")
            L.append("| method | PSNR | SSIM | LPIPS | DISTS | NIQE "
                     "| Cr plane |")
            L.append("|---|---|---|---|---|---|---|")
            for method in METHODS:
                c = [cell(method, iso, k, src) for k in
                     ("psnr", "ssim", "lpips", "dists", "niqe", "cr_psnr")]
                fmt = ["--" if x is None else (f"{x:.2f}" if i in (0, 5)
                       else f"{x:.4f}") for i, x in enumerate(c)]
                L.append(f"| {LABEL[method]} | " + " | ".join(fmt) + " |")
            L.append("")
    n_present = sum(1 for s in scenes for iso in ISOS if iso in m[s])
    n444 = sum(1 for s in m444 for iso in ISOS if iso in m444[s])
    L.append(f"\n_scenes: {len(scenes)}, (scene,ISO) cells: 420 "
             f"{n_present}/55, 444 {n444}/55_")
    (OUT / "SUMMARY_crvd.md").write_text("\n".join(L), encoding="utf-8")
    print("saved:", OUT / "SUMMARY_crvd.md", "| scenes:", len(scenes),
          "| cells:", n_present)


if __name__ == "__main__":
    main()
