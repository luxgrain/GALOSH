#!/usr/bin/env python3
"""Set8 baselines on the 420 track (same inputs/references as the GALOSH
420 run — identical seeds, identical bt709/limited/left conversion,
identical GT-420 same-upsampler reference).

Fairness contract:
  - sigma-parameterized methods (BM3D / V-BM3D) receive the MEASURED
    per-plane sigma: std(noisy - GT) on the actual 420 8-bit planes
    (the honest "oracle sigma" — the RGB-injected value changes under
    the matrix/range conversion);
  - free-knob methods use FROZEN DAVIS-calibrated tables:
    SMDegrain(tr=3, thSAD=table, prefilter=2, contrasharp=false)
    [_smdegrain_thsad.json] and KNLMeansCL(d=1, a=2, h=table,
    channels="YUV") [_knl_h.json];
  - hqdn3d runs at defaults (the untuned-lightweight reality check).

Baselines run through AviSynth (the tools' native host) via
tools/avs_dump.exe; metrics = the same 5-metric set + per-plane PSNR.

Usage: python bench_set8_baselines.py [--seqs ...] [--sigmas ...]
         [--methods bm3d1,vbm3d,hqdn3d,knl,smdegrain] [--limit-frames N]
         [--no-png]
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_set8_video import (DERF, GOPRO, OUT, seq_frames, load_clean,
                              make_noisy, rgb_clip_from, rgb48_clip_from_u8,
                              to_420, from_420, to_rgb24, frames_planes,
                              frames_rgb, plane_psnr, metrics5, agg, save_png)
from calib_smdegrain import write_y4m, run_avs, WORK, AVS_DUMP
import vapoursynth as vs
core = vs.core

# sigma-BLIND twins of the BM3D family: identical filter chain, sigma from
# mad_sigma(noisy) instead of the measured-vs-GT oracle.  The *bg twins are
# GUARDED: MAD clamped to SIGMA_FLOOR per plane (a zero MAD — H.264-smoothed
# or 420-upsampled chroma — otherwise feeds sigma=0 into the pipeline and
# V-BM3D breaks catastrophically).
BLIND = {"bm3d1b": "bm3d1", "vbm3db": "vbm3d",
         "bm3d1bg": "bm3d1", "vbm3dbg": "vbm3d"}
GUARDED = {"bm3d1bg", "vbm3dbg"}
SIGMA_FLOOR = 0.5   # smallest nonzero 8-bit Haar-MAD step is ~0.74


def blind_sig(m, bsig, meas):
    """sigma input for method m: guarded / raw MAD / oracle."""
    if m in GUARDED:
        return [max(s, SIGMA_FLOOR) for s in bsig]
    return bsig if m in BLIND else meas


def mad_sigma(planes_list):
    """BLIND per-plane sigma from the NOISY frames only (no GT): single-level
    Haar HH detail, sigma-hat = MAD/0.6745 (Donoho & Johnstone 1994), median
    across frames.  DUPLICATED from bench_set8_pgnoise.mad_sigma (importing it
    would be a circular import: pgnoise imports planes_to_420clip from here);
    keep the two in sync / consolidate into a shared module post-campaign."""
    est = []
    for pl in planes_list:
        s = []
        for p in pl:
            q = p.astype(np.float64)
            hh = (q[0::2, 0::2] - q[0::2, 1::2]
                  - q[1::2, 0::2] + q[1::2, 1::2]) * 0.5
            s.append(float(np.median(np.abs(hh - np.median(hh))) / 0.6745))
        est.append(s)
    return [float(np.median([e[p] for e in est])) for p in range(3)]

EXTOOLS = (WORK / "deps" / "ExTools.avsi").as_posix()


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


# ---- 444/16-bit lane (methods that natively do YUV444P16: bm3d1, knl) ----

def planes_to_444p16clip(planes_list, w, h):
    fmt = core.query_video_format(vs.YUV, vs.INTEGER, 16, 0, 0)
    c = core.std.BlankClip(width=w, height=h, format=fmt,
                           length=len(planes_list))

    def put(n, f, pl=planes_list):
        fo = f.copy()
        for p in range(3):
            np.asarray(fo[p])[:] = pl[n][p]
        return fo
    return core.std.ModifyFrame(c, c, put)


def write_y4m_444p16(path, planes_list, w, h):
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F25:1 Ip A1:1 C444p16\n".encode())
        for pl in planes_list:
            f.write(b"FRAME\n")
            for p in pl:
                f.write(np.ascontiguousarray(p, dtype="<u2").tobytes())


def run_avs_444p16(script_text, tag, n_frames):
    """avs_dump reader for YUV444P16 output (row sizes are in bytes)."""
    avs = WORK / f"{tag}.avs"
    raw = WORK / f"{tag}.raw"
    avs.write_text(script_text)
    r = subprocess.run([str(AVS_DUMP), str(avs), str(raw), str(n_frames)],
                       capture_output=True, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace")[-400:])
    w, h, cw, ch, n = map(int, r.stdout.split())
    if cw != 2 * w or ch != h:
        raise RuntimeError(f"{tag}: expected 444p16 dump, got cw={cw} ch={ch}"
                           f" for w={w} h={h}")
    data = np.fromfile(raw, dtype="<u2")
    fsz = 3 * w * h
    frames = []
    for i in range(n):
        o = i * fsz
        frames.append([data[o + p * w * h:o + (p + 1) * w * h].reshape(h, w)
                       for p in range(3)])
    raw.unlink()
    return frames


def method_script_444(method, y4m_path, meas_sigma, tables):
    """meas_sigma already on the 0-255 scale (16-bit std / 257)."""
    method = BLIND.get(method, method)   # blind twin -> same chain
    sy, su, sv = (round(s, 2) for s in meas_sigma)
    src = f'FFVideoSource("{y4m_path}")\n'
    if method == "bm3d1":
        return (src + 'ConvertBits(32)\n'
                f'BM3D_CPU(sigma=[{sy},{su},{sv}])\n'
                'ConvertBits(16)\n')
    if method == "knl":
        h = tables["knl"]
        return src + f'KNLMeansCL(d=1, a=2, h={h}, channels="YUV")\n'
    raise ValueError(method)


def method_script(method, y4m_path, meas_sigma, tables):
    method = BLIND.get(method, method)   # blind twin -> same chain
    sy, su, sv = (round(s, 2) for s in meas_sigma)
    src = f'FFVideoSource("{y4m_path}")\n'
    if method == "bm3d1":
        return (src + 'ConvertBits(32)\nConvertToYUV444()\n'
                f'BM3D_CPU(sigma=[{sy},{su},{sv}])\n'
                'ConvertToYUV420()\nConvertBits(8)\n')
    if method == "vbm3d":
        return (src + 'ConvertBits(32)\nConvertToYUV444()\n'
                f'BM3D_CPU(sigma=[{sy},{su},{sv}], radius=2)\n'
                'BM3D_VAggregate(radius=2)\n'
                'ConvertToYUV420()\nConvertBits(8)\n')
    if method == "hqdn3d":
        return src + 'hqdn3d()\n'
    if method == "knl":
        h = tables["knl"]
        return (src + 'ConvertToYUV444()\n'
                f'KNLMeansCL(d=1, a=2, h={h}, channels="YUV")\n'
                'ConvertToYUV420()\n')
    if method == "smdegrain":
        t = tables["smdegrain"]
        return (f'Import("{EXTOOLS}")\n' + src +
                f'SMDegrain(tr=3, thSAD={t}, prefilter=2, '
                f'contrasharp=false)\n')
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="420", choices=["420", "444"])
    ap.add_argument("--seqs", default=",".join(DERF + GOPRO))
    ap.add_argument("--sigmas", default="10,20,30,40,50")
    ap.add_argument("--methods", default="")
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if not args.methods:
        args.methods = ("bm3d1,vbm3d,hqdn3d,knl,smdegrain"
                        if args.mode == "420" else "bm3d1,knl")
    methods = args.methods.split(",")
    sigmas = [int(s) for s in args.sigmas.split(",")]
    WORK.mkdir(parents=True, exist_ok=True)
    if args.mode == "444":
        return main_444(args, methods, sigmas)

    # frozen knob tables (skip dependent methods when absent)
    thsad_tab = {}
    knl_tab = {}
    f1 = OUT / "_smdegrain_thsad.json"
    f2 = OUT / "_knl_h.json"
    if f1.exists():
        thsad_tab = json.loads(f1.read_text())["table"]
    if f2.exists():
        knl_tab = json.loads(f2.read_text())["table"]

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

        gt420_clip = to_420(rgb48_clip_from_u8(clean))
        gt_planes = list(frames_planes(gt420_clip))
        gt_ref = list(frames_rgb(from_420(gt420_clip)))

        for sigma in sigmas:
            noisy_f = make_noisy(clean, sigma, seed=1000 + sigma)
            noisy_planes = list(frames_planes(
                to_420(rgb_clip_from(noisy_f, raw_rgb48=True))))
            meas = [float(np.mean([np.std(a[p].astype(np.float64)
                                          - b[p].astype(np.float64))
                                   for a, b in zip(noisy_planes, gt_planes)]))
                    for p in range(3)]
            y4m = WORK / f"bl_{seq}_s{sigma}.y4m"
            write_y4m(y4m, noisy_planes, w_, h_)
            bsig = mad_sigma(noisy_planes)     # blind (no GT), for *b twins
            ent = {"measured_sigma_yuv": [round(m, 3) for m in meas],
                   "blind_sigma_yuv": [round(s, 3) for s in bsig]}

            for m in methods:
                tables = {}
                if m == "smdegrain":
                    if str(sigma) not in thsad_tab:
                        print(f"[skip] smdegrain (no thSAD table σ{sigma})")
                        continue
                    tables["smdegrain"] = thsad_tab[str(sigma)]
                if m == "knl":
                    if str(sigma) not in knl_tab:
                        print(f"[skip] knl (no h table σ{sigma})")
                        continue
                    tables["knl"] = knl_tab[str(sigma)]
                sig_in = blind_sig(m, bsig, meas)
                script = method_script(m, y4m.as_posix(), sig_in, tables)
                den_planes = run_avs(script, f"bl_{seq}_s{sigma}_{m}", n)
                den_clip = planes_to_420clip(den_planes, w_, h_)
                den_rgb = list(frames_rgb(from_420(den_clip)))
                rows = [metrics5(o, r) for o, r in zip(den_rgb, gt_ref)]
                e = agg(rows)
                for pi, pname in enumerate(["y_psnr", "cb_psnr", "cr_psnr"]):
                    e[pname] = float(np.mean(
                        [plane_psnr(d[pi], g[pi], 255)
                         for d, g in zip(den_planes, gt_planes)]))
                if not args.no_png:
                    for i, o in enumerate(den_rgb):
                        save_png(o, pngroot / seq / f"s{sigma}" / m
                                 / f"{i:04d}.png")
                ent[m] = e
                print(f"[bl {seq} s{sigma} {m}] PSNR {e['psnr']:.2f} "
                      f"SSIM {e['ssim']:.4f} LPIPS {e['lpips']:.4f}",
                      flush=True)
            y4m.unlink()
            results[seq][f"s{sigma}"] = ent
            outf = OUT / (f"_metrics_420_baselines"
                          f"{('_' + args.tag) if args.tag else ''}.json")
            outf.write_text(json.dumps(results, indent=1))
    print("saved:", outf)


def main_444(args, methods, sigmas):
    """444 lane: methods that natively run on YUV444P16 (bm3d1, knl).
    Protocol mirrors the GALOSH 444 track exactly: noisy RGB float ->
    YUV444P16 bt709 full (rgb_clip_from) -> filter -> RGB24 -> metrics
    vs the clean dataset frames.  sigma for BM3D = measured per-plane
    std on the 444P16 lattice / 257 (0-255 scale); KNL h = frozen table."""
    knl_tab = {}
    f2 = OUT / "_knl_h.json"
    if f2.exists():
        knl_tab = json.loads(f2.read_text())["table"]

    pngroot = OUT / "png"
    results = {}
    for seq in args.seqs.split(","):
        files = seq_frames(seq)
        if args.limit_frames:
            files = files[:args.limit_frames]
        clean = load_clean(files)
        n = len(clean)
        h_, w_ = clean[0].shape[:2]
        results[seq] = {"n_frames": n}

        gt_planes = list(frames_planes(
            rgb_clip_from([f.astype(np.float64) for f in clean])))

        for sigma in sigmas:
            noisy_f = make_noisy(clean, sigma, seed=1000 + sigma)
            noisy_planes = list(frames_planes(rgb_clip_from(noisy_f)))
            meas = [float(np.mean([np.std(a[p].astype(np.float64)
                                          - b[p].astype(np.float64))
                                   for a, b in zip(noisy_planes, gt_planes)]))
                    / 257.0 for p in range(3)]
            y4m = WORK / f"bl4_{seq}_s{sigma}.y4m"
            write_y4m_444p16(y4m, noisy_planes, w_, h_)
            bsig = [s / 257.0 for s in mad_sigma(noisy_planes)]  # 0-255 scale
            ent = {"measured_sigma_yuv_8bit_scale":
                   [round(m, 3) for m in meas],
                   "blind_sigma_yuv_8bit_scale":
                   [round(s, 3) for s in bsig]}

            for m in methods:
                tables = {}
                if m == "knl":
                    if str(sigma) not in knl_tab:
                        print(f"[skip] knl (no h table s{sigma})")
                        continue
                    tables["knl"] = knl_tab[str(sigma)]
                sig_in = blind_sig(m, bsig, meas)
                script = method_script_444(m, y4m.as_posix(), sig_in, tables)
                den_planes = run_avs_444p16(script, f"bl4_{seq}_s{sigma}_{m}",
                                            n)
                den_rgb = list(frames_rgb(to_rgb24(
                    planes_to_444p16clip(den_planes, w_, h_))))
                rows = [metrics5(o, r) for o, r in zip(den_rgb, clean)]
                e = agg(rows)
                if not args.no_png:
                    for i, o in enumerate(den_rgb):
                        save_png(o, pngroot / seq / f"s{sigma}" / m
                                 / f"{i:04d}.png")
                ent[m] = e
                print(f"[bl444 {seq} s{sigma} {m}] PSNR {e['psnr']:.2f} "
                      f"SSIM {e['ssim']:.4f} LPIPS {e['lpips']:.4f}",
                      flush=True)
            y4m.unlink()
            results[seq][f"s{sigma}"] = ent
            outf = OUT / (f"_metrics_444_baselines"
                          f"{('_' + args.tag) if args.tag else ''}.json")
            outf.write_text(json.dumps(results, indent=1))
    print("saved:", outf)


if __name__ == "__main__":
    main()
