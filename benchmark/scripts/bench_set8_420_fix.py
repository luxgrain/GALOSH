#!/usr/bin/env python3
"""420-track A/B for the 2026-07-12 half-res chroma-radius fix.

Drives the FIXED galosh_yuv_cpu.exe (planar 4:2:0, frame by frame =
cpu-fit semantics) twice per frame:
  - galosh-old : GALOSH_YUV420_CHROMA_R=7 forced  -> byte-identical to the
                 pre-fix behaviour (galosh_loess_chroma, R=7)
  - galosh-fix : env unset -> noise-adaptive R (sigma_lin<0.027 -> R=2,
                 else R=3), the fix
Same 420 machinery as bench_set8_video (identical seeds, to_420/from_420,
GT-420 same-upsampler reference, 5 metrics + per-plane Y/Cb/Cr PSNR).
Writes results_set8_awgn/_metrics_420_fix.json and PNGs under
png420/<seq>/s<sigma>/{galosh-old,galosh-fix}/ so the viewer can show the
controlled A/B beside the existing baselines.

Usage: python bench_set8_420_fix.py [--seqs ...] [--sigmas ...]
         [--limit-frames N] [--no-png]
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_set8_video import (DERF, GOPRO, OUT, seq_frames, load_clean,
                              make_noisy, rgb_clip_from, rgb48_clip_from_u8,
                              to_420, from_420, frames_planes, frames_rgb,
                              plane_psnr, metrics5, agg, save_png)
import vapoursynth as vs
core = vs.core

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "standalone" / "galosh_yuv_cpu.exe"
UCRT = r"C:\msys64\ucrt64\bin"
FLAGS = ["--pix=420", "--depth=8", "--range=limited", "--matrix=bt709",
         "--eotf=srgb", "--siting=left"]
VARIANTS = {"galosh-old": "7", "galosh-fix": None}   # env force value
WORK = OUT / "_fix_work"


def planes_to_420clip(planes_list, w, h):
    fmt = core.query_video_format(vs.YUV, vs.INTEGER, 8, 1, 1)
    c = core.std.BlankClip(width=w, height=h, format=fmt,
                           length=len(planes_list))

    def put(n, f, pl=planes_list):
        fo = f.copy()
        for p in range(3):
            np.asarray(fo[p])[:] = pl[n][p]
        return fo
    return core.std.ModifyFrame(c, c, put)


def run_exe_seq(frame_planes, w, h, force_R):
    """frame_planes: list of [Y,Cb,Cr]; returns denoised list, same layout."""
    env = dict(os.environ)
    env["PATH"] = env.get("PATH", "") + os.pathsep + UCRT
    if force_R:
        env["GALOSH_YUV420_CHROMA_R"] = force_R
    else:
        env.pop("GALOSH_YUV420_CHROMA_R", None)
    WORK.mkdir(parents=True, exist_ok=True)
    inp = WORK / "in.bin"; out = WORK / "out.bin"
    res = []
    for pl in frame_planes:
        inp.write_bytes(b"".join(np.ascontiguousarray(p).tobytes()
                                 for p in pl))
        r = subprocess.run([str(EXE), str(inp), str(out), str(w), str(h),
                            "1.0", "1.0", "0", "0"] + FLAGS,
                           capture_output=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode("utf-8", "replace")[-300:])
        d = np.fromfile(out, np.uint8)
        res.append([d[:w * h].reshape(h, w),
                    d[w * h:w * h + w * h // 4].reshape(h // 2, w // 2),
                    d[w * h + w * h // 4:].reshape(h // 2, w // 2)])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default=",".join(DERF + GOPRO))
    ap.add_argument("--sigmas", default="10,20,30,40,50")
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()
    sigmas = [int(s) for s in args.sigmas.split(",")]
    pngroot = OUT / "png420"
    results = {}
    for seq in args.seqs.split(","):
        files = seq_frames(seq)
        if args.limit_frames:
            files = files[:args.limit_frames]
        clean = load_clean(files)
        n = len(clean)
        h_, w_ = clean[0].shape[:2]
        results[seq] = {"n_frames": n}

        gt420 = to_420(rgb48_clip_from_u8(clean))
        gt_planes = list(frames_planes(gt420))
        gt_ref = list(frames_rgb(from_420(gt420)))

        for sigma in sigmas:
            noisy_f = make_noisy(clean, sigma, seed=1000 + sigma)
            noisy_planes = list(frames_planes(
                to_420(rgb_clip_from(noisy_f, raw_rgb48=True))))
            ent = {}
            for vname, forceR in VARIANTS.items():
                den_planes = run_exe_seq(noisy_planes, w_, h_, forceR)
                den_rgb = list(frames_rgb(from_420(
                    planes_to_420clip(den_planes, w_, h_))))
                rows = [metrics5(o, r) for o, r in zip(den_rgb, gt_ref)]
                e = agg(rows)
                for pi, pn in enumerate(["y_psnr", "cb_psnr", "cr_psnr"]):
                    e[pn] = float(np.mean(
                        [plane_psnr(d[pi], g[pi], 255)
                         for d, g in zip(den_planes, gt_planes)]))
                ent[vname] = e
                if not args.no_png:
                    for i, o in enumerate(den_rgb):
                        save_png(o, pngroot / seq / f"s{sigma}" / vname
                                 / f"{i:04d}.png")
                print(f"[fix {seq} s{sigma} {vname}] PSNR {e['psnr']:.2f} "
                      f"Cb {e['cb_psnr']:.2f} Cr {e['cr_psnr']:.2f} "
                      f"LPIPS {e['lpips']:.4f}", flush=True)
            results[seq][f"s{sigma}"] = ent
            (OUT / "_metrics_420_fix.json").write_text(
                json.dumps(results, indent=1))
    print("saved:", OUT / "_metrics_420_fix.json")


if __name__ == "__main__":
    main()
