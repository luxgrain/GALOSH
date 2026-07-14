#!/usr/bin/env python3
"""KNLMeansCL h calibration (frozen per-sigma table) — same design as the
SMDegrain thSAD calibration: DAVIS bear+bmx-bumps, sweep
h = sigma * {0.3, 0.5, 0.7, 1.0, 1.4}, argmax mean Y-PSNR, freeze.
Recipe: KNLMeansCL(d=1, a=2, h=table[sigma], channels="YUV").
Output: _knl_h.json
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_set8_video import (load_clean, make_noisy, rgb_clip_from,
                              rgb48_clip_from_u8, to_420, frames_planes,
                              plane_psnr, OUT)
from calib_smdegrain import (write_y4m, run_avs, WORK, DAVIS, CALIB_SEQS,
                             N_FRAMES, SIGMAS)

FACTORS = [0.3, 0.5, 0.7, 1.0, 1.4]


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    table, detail = {}, {}
    for sigma in SIGMAS:
        scores = {}
        for seq in CALIB_SEQS:
            files = sorted((DAVIS / seq).glob("*.jpg"))[:N_FRAMES]
            clean = load_clean(files)
            gt_planes = list(frames_planes(to_420(rgb48_clip_from_u8(clean))))
            noisy_f = make_noisy(clean, sigma, seed=1000 + sigma)
            noisy_planes = list(frames_planes(
                to_420(rgb_clip_from(noisy_f, raw_rgb48=True))))
            y4m = WORK / f"knl_{seq}_s{sigma}.y4m"
            h_, w_ = clean[0].shape[:2]
            write_y4m(y4m, noisy_planes, w_, h_)
            for fac in FACTORS:
                hh = round(sigma * fac, 2)
                script = (f'FFVideoSource("{y4m.as_posix()}")\n'
                          'ConvertToYUV444()\n'
                          f'KNLMeansCL(d=1, a=2, h={hh}, channels="YUV")\n'
                          'ConvertToYUV420()\n')
                den = run_avs(script, f"knl_{seq}_s{sigma}_h{hh}", N_FRAMES)
                yp = float(np.mean([plane_psnr(d[0], g[0], 255)
                                    for d, g in zip(den, gt_planes)]))
                scores.setdefault(fac, []).append(yp)
                print(f"[knl s{sigma} {seq} h={hh} (x{fac})] Y-PSNR {yp:.2f}",
                      flush=True)
        mean_by = {f: float(np.mean(v)) for f, v in scores.items()}
        best = max(mean_by, key=mean_by.get)
        table[str(sigma)] = round(sigma * best, 2)
        detail[str(sigma)] = {"mean_y_psnr_by_factor": mean_by,
                              "best_factor": best, "h": table[str(sigma)]}
        print(f"== s{sigma}: h = {table[str(sigma)]} (x{best}) ==", flush=True)
    out = {"recipe": 'KNLMeansCL(d=1, a=2, h=table[sigma], channels="YUV")',
           "calibrated_on": {"seqs": CALIB_SEQS, "frames": N_FRAMES,
                             "grid": "sigma*" + str(FACTORS)},
           "table": table, "detail": detail}
    (OUT / "_knl_h.json").write_text(json.dumps(out, indent=1))
    print("saved:", OUT / "_knl_h.json")


if __name__ == "__main__":
    main()
