#!/usr/bin/env python3
"""Arm-2 prep: find the GENERIC phone-curve setting whose injected noise
matches each CRVD ISO's real developed sigma (2026-07-17).

For each CRVD ISO, binary-search the continuous phone trimmed-median curve
(tools/pgnoise.iso_to_pg) for the equivalent-ISO t such that the injected
noise on CRVD clean content develops to the same 420-lane Y sigma as the
REAL noise (target = scene-mean measured_sigma_yuv[0] from the real bench).
This is the "sigma-matched generic generator" test: no target-sensor
measurement, only a sigma match — the generality claim of pgnoise.

Writes results_crvd/_synth_phone_match.json:
  {CRVD_ISO: {phone_iso_equiv, a, b, target_sigmaY, achieved_sigmaY}}
"""
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parents[1] / "tools"))
import pgnoise as pgn
import bench_crvd
from bench_set8_video import rgb_clip_from, to_420, frames_planes

OUT = SCRIPTS.parents[1] / "benchmark" / "results_crvd"
SCENES = ["1", "4", "7"]
NFR = 2          # frames per scene for the matcher (speed; sigma is stable)


def sigma420_y(clean_f, noisy_f):
    gt = to_420(rgb_clip_from([c for c in clean_f], raw_rgb48=True))
    nz = to_420(rgb_clip_from([n for n in noisy_f], raw_rgb48=True))
    gp = list(frames_planes(gt))
    np_ = list(frames_planes(nz))
    return float(np.mean([np.std(a[0].astype(np.float64)
                                 - b[0].astype(np.float64))
                          for a, b in zip(np_, gp)]))


def injected_sigma(packs, t):
    a, b = pgn.iso_to_pg(t)
    s = []
    for cu8, cf in packs:
        noisy = [pgn.add_pg_noise(f, a, b, seed_shot=311, seed_read=313,
                                  frame_idx=i).astype(np.float64)
                 for i, f in enumerate(cu8)]
        s.append(sigma420_y(cf, noisy))
    return float(np.mean(s))


def main():
    real = json.loads((OUT / "_metrics_crvd.json").read_text())
    mapping = {}
    for iso in bench_crvd.ISOS:
        target = float(np.mean([real[s][iso]["measured_sigma_yuv"][0]
                                for s in real if s != "_env"
                                and iso in real[s]]))
        packs = []
        for sc in SCENES:
            _, clean_f = bench_crvd.load_seq(sc, iso)
            clean_f = clean_f[:NFR]
            cu8 = [np.clip(np.round(c), 0, 255).astype(np.uint8)
                   for c in clean_f]
            packs.append((cu8, clean_f))
        lo, hi = np.log(50.0), np.log(60000.0)
        for _ in range(11):
            mid = (lo + hi) / 2
            s = injected_sigma(packs, float(np.exp(mid)))
            if s < target:
                lo = mid
            else:
                hi = mid
        t = float(np.exp((lo + hi) / 2))
        a, b = pgn.iso_to_pg(t)
        ach = injected_sigma(packs, t)
        mapping[iso] = {"phone_iso_equiv": round(t, 1),
                        "a": round(a, 8), "b": round(b, 8),
                        "target_sigmaY_420": round(target, 3),
                        "achieved_sigmaY_420": round(ach, 3)}
        print(f"[match {iso}] target sY={target:.2f} -> phone-ISO~{t:.0f} "
              f"(a={a:.6f} b={b:.6f}) achieved sY={ach:.2f}", flush=True)
    (OUT / "_synth_phone_match.json").write_text(json.dumps(mapping, indent=1))
    print("saved:", OUT / "_synth_phone_match.json")


if __name__ == "__main__":
    main()
