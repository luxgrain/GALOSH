#!/usr/bin/env python3
"""SMDegrain thSAD calibration (frozen per-sigma table for the Set8 bench).

Fairness design (user-approved recipe): SMDegrain cannot take sigma
directly, so the sigma-equivalent information is delivered as a FROZEN
thSAD(sigma) table calibrated OUTSIDE Set8 (two DAVIS 480p sequences),
then applied unchanged to Set8.  Everything else fixed:
    SMDegrain(tr=3, thSAD=<calibrated>, prefilter=2, contrasharp=false)

Protocol per (seq, sigma): RGB AWGN (same seeds as the Set8 harness) ->
YUV420P8 bt709 limited left (same conversion as the bench 420 track) ->
y4m -> AviSynth SMDegrain sweep -> Y-plane PSNR vs clean-420 -> argmax.
Sweep grid: thSAD = round(72*sigma * {0.6, 0.8, 1.0, 1.3, 1.6, 2.0})
(72*sigma = the 8x8 SAD noise floor of a sigma-noisy frame pair).

Output: benchmark/results_set8_awgn/_smdegrain_thsad.json
"""
import json, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_set8_video import (make_noisy, rgb_clip_from, rgb48_clip_from_u8,
                              to_420, frames_planes, plane_psnr, load_clean)
import vapoursynth as vs
core = vs.core

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmark" / "results_set8_awgn"
WORK = OUT / "_calib_work"
DAVIS = Path("E:/img_dataset/DAVIS2017/DAVIS/JPEGImages/480p")
AVS_DUMP = ROOT / "tools" / "avs_dump.exe"

CALIB_SEQS = ["bear", "bmx-bumps"]     # static-ish + fast motion
N_FRAMES = 40
SIGMAS = [10, 20, 30, 40, 50]
FACTORS = [0.6, 0.8, 1.0, 1.3, 1.6, 2.0]


def write_y4m(path, planes_list, w, h):
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F25:1 Ip A1:1 C420mpeg2\n".encode())
        for pl in planes_list:
            f.write(b"FRAME\n")
            for p in pl:
                f.write(np.ascontiguousarray(p).tobytes())


def run_avs(script_text, tag, n_frames):
    avs = WORK / f"{tag}.avs"
    raw = WORK / f"{tag}.raw"
    avs.write_text(script_text)
    r = subprocess.run([str(AVS_DUMP), str(avs), str(raw), str(n_frames)],
                       capture_output=True, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace")[-400:])
    w, h, cw, ch, n = map(int, r.stdout.split())
    data = np.fromfile(raw, dtype=np.uint8)
    fsz = w * h + 2 * cw * ch
    frames = []
    for i in range(n):
        o = i * fsz
        y = data[o:o + w * h].reshape(h, w)
        u = data[o + w * h:o + w * h + cw * ch].reshape(ch, cw)
        v = data[o + w * h + cw * ch:o + fsz].reshape(ch, cw)
        frames.append([y, u, v])
    raw.unlink()
    return frames


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    table = {}
    detail = {}
    for sigma in SIGMAS:
        scores = {}
        for seq in CALIB_SEQS:
            files = sorted((DAVIS / seq).glob("*.jpg"))[:N_FRAMES]
            clean = load_clean(files)
            # clean-420 reference (identical conversion to the bench)
            gt_planes = list(frames_planes(to_420(rgb48_clip_from_u8(clean))))
            noisy_f = make_noisy(clean, sigma, seed=1000 + sigma)
            noisy_planes = list(frames_planes(
                to_420(rgb_clip_from(noisy_f, raw_rgb48=True))))
            y4m = WORK / f"{seq}_s{sigma}.y4m"
            h, w = clean[0].shape[:2]
            write_y4m(y4m, noisy_planes, w, h)

            for fac in FACTORS:
                thsad = int(round(72 * sigma * fac))
                deps = (WORK / "deps" / "ExTools.avsi").as_posix()
                script = (f'Import("{deps}")\n'
                          f'FFVideoSource("{y4m.as_posix()}")\n'
                          f'SMDegrain(tr=3, thSAD={thsad}, prefilter=2, '
                          f'contrasharp=false)\n')
                den = run_avs(script, f"{seq}_s{sigma}_t{thsad}", N_FRAMES)
                yp = float(np.mean([plane_psnr(d[0], g[0], 255)
                                    for d, g in zip(den, gt_planes)]))
                scores.setdefault(fac, []).append(yp)
                print(f"[calib σ{sigma} {seq} thSAD={thsad} (x{fac})] "
                      f"Y-PSNR {yp:.2f}", flush=True)
        mean_by_fac = {fac: float(np.mean(v)) for fac, v in scores.items()}
        best_fac = max(mean_by_fac, key=mean_by_fac.get)
        table[str(sigma)] = int(round(72 * sigma * best_fac))
        detail[str(sigma)] = {"mean_y_psnr_by_factor": mean_by_fac,
                              "best_factor": best_fac,
                              "thSAD": table[str(sigma)]}
        print(f"== σ{sigma}: thSAD = {table[str(sigma)]} "
              f"(factor {best_fac}) ==", flush=True)
    out = {"recipe": "SMDegrain(tr=3, thSAD=table[sigma], prefilter=2, "
                     "contrasharp=false)",
           "calibrated_on": {"seqs": CALIB_SEQS, "frames": N_FRAMES,
                             "grid": "72*sigma*" + str(FACTORS)},
           "table": table, "detail": detail}
    (OUT / "_smdegrain_thsad.json").write_text(json.dumps(out, indent=1))
    print("saved:", OUT / "_smdegrain_thsad.json")


if __name__ == "__main__":
    main()
