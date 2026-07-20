#!/usr/bin/env python3
"""YUV noise-estimator ablation A/B/D (2026-07-17 audit follow-up).

A = current global Laplacian MAD + synthetic alpha  (canonical since ff95bc0)
B = single-plane Foi-style lower-envelope (alpha, sigma^2) fit
    (GALOSH_YUV_NOISE_EST=envelope; the RAW Phase-0 counterpart, restoring
    the paper's shared-core claim)
D = measured oracle: (alpha, sigma^2) LSQ-fit of var(noisyY-gtY) vs gtY in
    LINEAR Y from the noisy/GT pair, passed via argv[7]/argv[8]

Lanes (subset): AWGN / PG-core / PG-cmp (Set8: tractor, rafting) and CRVD
(scenes 1,4,7), each in 420 and 444 container protocol, NFR frames per cell.
Per cell x arm: estimated (alpha, sigma^2), full 5-metric set vs the lane GT,
runtime; PNGs persisted for every arm (persist-everything).

CPU-only (the exe); GPU parity is a later step for the chosen winner.
Usage: python _noiseest_ablation.py [--jobs N] [--quick]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parents[1] / "tools"))
from bench_set8_video import (seq_frames, load_clean, make_noisy,
                              rgb_clip_from, to_420, from_420, to_rgb24,
                              frames_planes, frames_rgb, metrics5, agg)
from degrade_set8_pgnoise import degrade_bcore, compress_h264
import bench_crvd
from bench_crvd import load_seq, to444
import vapoursynth as vs
core = vs.core

ROOT = SCRIPTS.parents[1]
EXE = ROOT / "standalone" / "galosh_yuv_cpu.exe"
OUT = ROOT / "benchmark" / "results_noiseest_ablation"
UCRT = r"C:\msys64\ucrt64\bin"
G420 = ["--pix=420", "--depth=8", "--range=limited", "--matrix=bt709",
        "--eotf=srgb", "--siting=left"]
G444 = ["--pix=444", "--depth=16", "--range=full", "--matrix=bt709",
        "--eotf=srgb"]
NFR = 5
SET8_SEQS = ["tractor", "rafting"]
SIGMAS = [10, 30, 50]
ISOS = ["ISO1600", "ISO6400", "ISO25600"]
CRVD_SCENES = ["1", "4", "7"]
EST_RE = re.compile(
    r"(?:using α=|envelope\): α=|alpha=)([0-9.e+-]+)"
    r".*?(?:σ²=|sigma_sq=)([0-9.e+-]+)")


def run_exe_est(planes, w, h, flags, arm, oracle=None, tag="x"):
    """arm: 'A' (mad) / 'B' (envelope) / 'D' (oracle=(alpha,sigma_sq))."""
    env = dict(os.environ)
    env["PATH"] = env.get("PATH", "") + os.pathsep + UCRT
    env.pop("GALOSH_YUV_NOISE_EST", None)
    # [2026-07-19] canonical default flipped to envelope — arms are now
    # EXPLICIT so the rig's semantics survive the flip: A=mad, B=envelope.
    if arm == "A":
        env["GALOSH_YUV_NOISE_EST"] = "mad"
    elif arm == "B":
        env["GALOSH_YUV_NOISE_EST"] = "envelope"
    a_cli, s_cli = ("0", "0")
    if arm == "D":
        a_cli, s_cli = (f"{oracle[0]:.8g}", f"{oracle[1]:.10g}")
    wd = OUT / f"_work_{os.getpid()}"
    wd.mkdir(parents=True, exist_ok=True)
    inp, outp = wd / "in.bin", wd / "out.bin"
    is444 = "--pix=444" in flags
    res, ests = [], []
    for pl in planes:
        inp.write_bytes(b"".join(np.ascontiguousarray(p).tobytes()
                                 for p in pl))
        r = subprocess.run([str(EXE), str(inp), str(outp), str(w), str(h),
                            "1.0", "1.0", a_cli, s_cli] + flags,
                           capture_output=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode("utf-8", "replace")[-300:])
        err = r.stderr.decode("utf-8", "replace")
        m = EST_RE.search(err)
        ests.append((float(m.group(1)), float(m.group(2))) if m else
                    ((float(a_cli), float(s_cli)) if arm == "D" else None))
        d = np.fromfile(outp, np.uint16 if is444 else np.uint8)
        if is444:
            res.append([d[i * w * h:(i + 1) * w * h].reshape(h, w)
                        for i in range(3)])
        else:
            res.append([d[:w * h].reshape(h, w),
                        d[w * h:w * h + w * h // 4].reshape(h // 2, w // 2),
                        d[w * h + w * h // 4:].reshape(h // 2, w // 2)])
    return res, ests


def srgb_to_linear_np(s):
    s = np.clip(s, 0.0, 1.0)
    return np.where(s <= 0.04045, s / 12.92,
                    ((s + 0.055) / 1.055) ** 2.4)


def linear_y(rgb_u8):
    lin = srgb_to_linear_np(rgb_u8.astype(np.float64) / 255.0)
    return (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1]
            + 0.0722 * lin[..., 2])


def oracle_fit(noisy_rgb, gt_rgb, nbins=32):
    """LSQ fit var(nY-gY | gY) = alpha*gY + sigma^2 in linear Y."""
    d, g = [], []
    for n, gt in zip(noisy_rgb, gt_rgb):
        ny, gy = linear_y(n), linear_y(gt)
        d.append((ny - gy).ravel())
        g.append(gy.ravel())
    d, g = np.concatenate(d), np.concatenate(g)
    lo, hi = np.quantile(g, 0.01), np.quantile(g, 0.99)
    edges = np.linspace(lo, hi, nbins + 1)
    xs, ys = [], []
    for i in range(nbins):
        m = (g >= edges[i]) & (g < edges[i + 1])
        if m.sum() > 2000:
            xs.append(float(g[m].mean()))
            ys.append(float(d[m].var()))
    if len(xs) < 4:
        return 1e-5, float(np.var(d))
    A = np.column_stack([xs, np.ones(len(xs))])
    sol, *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    # clamp BOTH strictly positive: the exe treats alpha<=0 OR sigma<=0 as
    # "not supplied" and would silently fall back to blind estimation
    return max(float(sol[0]), 1e-6), max(float(sol[1]), 1e-9)


def prep_cell(lane, material, level):
    """-> (planes_in, w, h, flags, gt_ref, noisy_rgb_u8_for_oracle)."""
    mode = lane.split("-")[-1]                      # '420' or '444'
    kind = lane.rsplit("-", 1)[0]                   # awgn / pgcore / pgcmp / crvd
    if kind == "crvd":
        noisy_f, clean_f = load_seq(material, level)
        noisy_f, clean_f = noisy_f[:NFR], clean_f[:NFR]
        noisy_u8 = [np.clip(np.round(f), 0, 255).astype(np.uint8)
                    for f in noisy_f]
        clean_u8 = [np.clip(np.round(f), 0, 255).astype(np.uint8)
                    for f in clean_f]
    else:
        clean = load_clean(seq_frames(material)[:NFR])
        if kind == "awgn":
            noisy = make_noisy(clean, level, seed=1000 + level)
            noisy_u8 = [np.clip(np.round(f), 0, 255).astype(np.uint8)
                        for f in noisy]
        else:
            from degrade_set8_pgnoise import PG
            rng = np.random.default_rng(
                7000 + int(level.replace("ISO", "")))
            noisy_u8 = [degrade_bcore(c, pg=PG[level], rng=rng)
                        for c in clean]
            if kind == "pgcmp":
                env0 = os.environ.get("PATH", "")
                os.environ["PATH"] = env0 + os.pathsep + UCRT
                h264_tmp = OUT / f"_h264_{os.getpid()}"
                h264_tmp.mkdir(parents=True, exist_ok=True)
                noisy_u8 = compress_h264(
                    [np.ascontiguousarray(f, np.uint8) for f in noisy_u8],
                    crf=23, tmp=str(h264_tmp))
        clean_u8 = clean
    cf = [c.astype(np.float64) for c in clean_u8]
    nf = [n.astype(np.float64) for n in noisy_u8]
    h, w = cf[0].shape[:2]
    if mode == "420":
        gt420 = to_420(rgb_clip_from(cf, raw_rgb48=True))
        gt_ref = list(frames_rgb(from_420(gt420)))
        nz = to_420(rgb_clip_from(nf, raw_rgb48=True))
        planes = list(frames_planes(nz))
        flags = G420
    else:
        gtc = to444(cf, 16)
        gt_ref = list(frames_rgb(to_rgb24(gtc)))
        planes = list(frames_planes(to444(nf, 16)))
        flags = G444
    return planes, w, h, flags, gt_ref, noisy_u8, clean_u8


def recon_rgb(planes_out, w, h, flags):
    from bench_set8_baselines import planes_to_420clip
    if "--pix=444" in flags:
        fmt = core.query_video_format(vs.YUV, vs.INTEGER, 16, 0, 0)
        c = core.std.BlankClip(width=w, height=h, format=fmt,
                               length=len(planes_out))

        def put(n, f, pl=planes_out):
            fo = f.copy()
            for p in range(3):
                np.asarray(fo[p])[:] = pl[n][p]
            return fo
        return list(frames_rgb(to_rgb24(core.std.ModifyFrame(c, c, put))))
    return list(frames_rgb(from_420(planes_to_420clip(planes_out, w, h))))


def run_cell(job):
    lane, material, level = job
    try:
        return _run_cell_inner(job)
    except Exception as ex:
        print(f"[abl {lane} {material} {level}] FAILED: {str(ex)[-160:]}",
              flush=True)
        return (f"{lane}|{material}|{level}", {"error": str(ex)[-300:]})


def _run_cell_inner(job):
    lane, material, level = job
    t0 = time.time()
    planes, w, h, flags, gt_ref, noisy_u8, clean_u8 = prep_cell(
        lane, material, level)
    orc = oracle_fit(noisy_u8, clean_u8)
    out = {"oracle_fit": [round(orc[0], 7), round(orc[1], 9)]}
    for arm in ("A", "B", "D"):
        dp, ests = run_exe_est(planes, w, h, flags, arm,
                               oracle=orc, tag=f"{lane}_{material}_{level}")
        drgb = recon_rgb(dp, w, h, flags)
        e = agg([metrics5(o, r) for o, r in zip(drgb, gt_ref)])
        est_ok = [x for x in ests if x]
        e["est_alpha"] = round(float(np.mean([x[0] for x in est_ok])), 7) \
            if est_ok else None
        e["est_sigma_sq"] = round(float(np.mean([x[1] for x in est_ok])), 9) \
            if est_ok else None
        out[arm] = e
        from bench_set8_video import save_png
        for i, o in enumerate(drgb):
            save_png(o, OUT / "png" / lane / f"{material}_{level}" / arm
                     / f"{i:02d}.png")
    out["sec"] = round(time.time() - t0, 1)
    print(f"[abl {lane} {material} {level}] "
          + " ".join(f"{a}={out[a]['psnr']:.2f}/{out[a]['lpips']:.3f}"
                     for a in ("A", "B", "D"))
          + f" oracle=({orc[0]:.5f},{orc[1]:.2e})", flush=True)
    return (f"{lane}|{material}|{level}", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for mode in ("420", "444"):
        for seq in SET8_SEQS:
            for s in SIGMAS:
                jobs.append((f"awgn-{mode}", seq, s))
            for iso in ISOS:
                jobs.append((f"pgcore-{mode}", seq, iso))
                jobs.append((f"pgcmp-{mode}", seq, iso))
        for sc in CRVD_SCENES:
            for iso in ISOS:
                jobs.append((f"crvd-{mode}", sc, iso))
    if args.quick:
        jobs = jobs[:2]
    # resume: skip cells already completed (non-error) in an existing JSON
    results = {}
    mf = OUT / "_metrics_ablation.json"
    if mf.exists():
        results = {k: v for k, v in json.loads(mf.read_text()).items()
                   if "error" not in v}
    jobs = [j for j in jobs if f"{j[0]}|{j[1]}|{j[2]}" not in results]
    print(f"[abl] {len(jobs)} cells to run x 3 arms, NFR={NFR} "
          f"(resume-skipped {len(results)})")
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for key, val in ex.map(run_cell, jobs):
            results[key] = val
            mf.write_text(json.dumps(results, indent=1))
    print("saved:", mf)


if __name__ == "__main__":
    main()
