#!/usr/bin/env python3
"""Luma-strength A/B (2026-07-20): l in {1.0, 0.7, 0.5} x c=1.0.

User observation after the envelope rerun: default l1c1 is detail-aggressive
(the envelope estimator raised the effective sigma vs the old MAD
under-estimates, so the same strength now denoises harder).  This rig
reproduces the bench 420-lane cells exactly (bench_set8_video conversion
chain + DLL cpu-fit) at three luma strengths and reports metrics5 + PNGs.

Cells: Set8 tractor s20/s40 (AWGN) + CRVD scene7 ISO12800/25600 (real).
Output: benchmark/_strength_ab_20260720/ (json + png).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bench_set8_video as v
import vapoursynth as vs

core = vs.core
core.std.LoadPlugin(path=str(v.DLL))

OUT = Path(__file__).resolve().parents[1] / "_strength_ab_20260720"
NFR = 3
ARMS = [(1.0, 1.0), (0.7, 1.0), (0.5, 1.0)]
KW420 = dict(matrix="bt709", range="limited", eotf="srgb", siting="left",
             engine="cpu", noise="fit")


def run_cell(name, clean_u8, noisy_f):
    src = v.to_420(v.rgb_clip_from(noisy_f, raw_rgb48=True))
    gt420 = v.to_420(v.rgb48_clip_from_u8(clean_u8))
    gt_ref = list(v.frames_rgb(v.from_420(gt420)))
    noisy_rec = list(v.frames_rgb(v.from_420(src)))
    for i, g in enumerate(gt_ref):
        v.save_png(g, OUT / name / "gt" / f"{i:04d}.png")
    for i, nn in enumerate(noisy_rec):
        v.save_png(nn, OUT / name / "noisy" / f"{i:04d}.png")
    res = {"noisy": v.agg([v.metrics5(nn, g)
                           for nn, g in zip(noisy_rec, gt_ref)])}
    for l, c in ARMS:
        den = core.galosh.Denoise(src, luma=l, chroma=c, **KW420)
        out = list(v.frames_rgb(v.from_420(den)))
        tag = f"l{l:g}c{c:g}"
        res[tag] = v.agg([v.metrics5(o, g) for o, g in zip(out, gt_ref)])
        for i, o in enumerate(out):
            v.save_png(o, OUT / name / tag / f"{i:04d}.png")
        m = res[tag]
        print(f"[{name}] {tag}: PSNR {m['psnr']:.2f} SSIM {m['ssim']:.4f} "
              f"LPIPS {m['lpips']:.4f} DISTS {m['dists']:.4f} "
              f"NIQE {m['niqe']:.3f}", flush=True)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    for sigma in (20, 40):                       # Set8 tractor (AWGN lane)
        files = v.seq_frames("tractor")[:NFR]
        clean = v.load_clean(files)
        noisy_f = v.make_noisy(clean, sigma, seed=1000 + sigma)
        results[f"tractor_s{sigma}"] = run_cell(f"tractor_s{sigma}",
                                                clean, noisy_f)

    import bench_crvd as bc                      # CRVD scene7 (real noise)
    for iso in ("ISO12800", "ISO25600"):
        noisy, clean = bc.load_seq(7, iso)
        noisy, clean = noisy[:NFR], clean[:NFR]
        clean_u8 = [np.clip(np.round(c), 0, 255).astype(np.uint8)
                    for c in clean]
        results[f"crvd_s7_{iso}"] = run_cell(f"crvd_s7_{iso}",
                                             clean_u8, noisy)

    (OUT / "_metrics_strength_ab.json").write_text(
        json.dumps(results, indent=1))
    print(f"saved: {OUT / '_metrics_strength_ab.json'}")


if __name__ == "__main__":
    main()
