#!/usr/bin/env python3
"""Generate the noisy + gt_ref PNGs the CRVD bench didn't save, so the CRVD
viewer can show noisy -> GALOSH/SMDegrain -> clean side by side.  Develops
each (scene, ISO) with the same ISP as bench_crvd (identical for noisy/GT).
CPU only (no denoise), light."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from bench_crvd import load_seq, OUT, planes_to_420clip
from bench_set8_video import (rgb_clip_from, to_420, from_420, frames_planes,
                              frames_rgb, save_png)
from bench_set8_baselines import planes_to_420clip as _p  # noqa
import vapoursynth as vs
core = vs.core

ISOS = ["ISO1600", "ISO3200", "ISO6400", "ISO12800", "ISO25600"]
PNG = OUT / "png"


def main():
    scenes = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1
                               else range(1, 12))] if len(sys.argv) > 1 \
        else list(range(1, 12))
    for scene in scenes:
        for iso in ISOS:
            try:
                noisy_f, clean_f = load_seq(scene, iso)
            except Exception as e:
                print(f"scene{scene} {iso}: skip ({str(e)[-60:]})"); continue
            h_, w_ = noisy_f[0].shape[:2]
            gt420 = to_420(rgb_clip_from([c for c in clean_f], raw_rgb48=True))
            gt_ref = list(frames_rgb(from_420(gt420)))
            noisy420 = to_420(rgb_clip_from(noisy_f, raw_rgb48=True))
            noisy_rec = list(frames_rgb(from_420(noisy420)))
            for i, (nn, gg) in enumerate(zip(noisy_rec, gt_ref)):
                save_png(nn, PNG / f"scene{scene}" / iso / "noisy"
                         / f"{i:02d}.png")
                save_png(gg, PNG / f"scene{scene}" / iso / "gt_ref"
                         / f"{i:02d}.png")
            print(f"scene{scene} {iso}: noisy+gt_ref saved ({len(gt_ref)}f)",
                  flush=True)
    print("CRVD_PNG_GEN_DONE")


if __name__ == "__main__":
    main()
