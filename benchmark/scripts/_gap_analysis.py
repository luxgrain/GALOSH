#!/usr/bin/env python3
"""CRVD-vs-PG GALOSH-gap factor analysis (2026-07-16, diagnostic probe).

Question: at MATCHED developed sigma (CRVD ISO25600 ~ sigma17.6 vs phone-PG
ISO1600 ~ sigma18), why does GALOSH beat oracle BM3D by +2.6 dB on CRVD but
lose by 0.4 dB on the PG track?

Design: 2x2 cross (content x noise-model), 420 protocol of the PG track,
PSNR-only (diagnostic; full metrics live in the main benches):
  cell A (ref)   Set8 content + phone-PG ISO1600 params    (= PG track cell)
  cell B         Set8 content + CRVD-measured ISO25600 params
  cell C         CRVD clean content + phone-PG ISO1600 params
  cell D (ref)   CRVD content + real noise (existing bench numbers)
If B ~ D-gap -> the (a,b) noise composition explains it; if C ~ D-gap ->
content explains it.  banding=0 in synthetic cells (the old 0.3*sqrt(b)
heuristic is ~20x over-strong and would contaminate B).

Plus direct noise statistics at matched sigma: variance-vs-intensity curve,
lag-1 spatial autocorrelation, kurtosis, per-plane sigma (Y/Cb/Cr).
[2026-07-16b] cells report the FULL 5-metric set (initially PSNR-only —
user caught the violation of the full-metrics default; LPIPS-first).
"""
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from bench_set8_video import (seq_frames, load_clean, rgb_clip_from, to_420,
                              from_420, frames_planes, frames_rgb,
                              metrics5, agg)
from bench_set8_baselines import planes_to_420clip
from bench_set8_pgnoise import GVAR, DLL, avs_method
from calib_smdegrain import write_y4m, run_avs, WORK as AVS_WORK
from degrade_set8_pgnoise import degrade_bcore
import bench_crvd
import vapoursynth as vs
core = vs.core

OUT = SCRIPTS.parents[1] / "benchmark" / "results_gap_analysis"
PHONE1600 = (0.0057630, 0.0000634)      # phone trimmed-median ISO1600
CRVD25600 = (0.00811, 0.00160828)       # CRVD_IMX385 measured ISO25600
SET8_SEQS = ["tractor", "sunflower", "hypersmooth"]
CRVD_SCENES = ["1", "4", "7"]
NFR = 7          # frames per cell (CRVD seq length; Set8 capped to match)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)))


def degrade(frames_u8, pg, seed):
    rng = np.random.default_rng(seed)
    return [degrade_bcore(f, pg=pg, rng=rng, banding=0.0) for f in frames_u8]


def run_cell(tag, clean_u8, pg):
    """PG-track 420 protocol: GT = clean->420 roundtrip; galosh DLL + BM3D
    oracle sigma; returns per-method PSNR (mean over frames)."""
    noisy = degrade(clean_u8, pg, seed=1234)
    h_, w_ = clean_u8[0].shape[:2]
    gt420 = to_420(rgb_clip_from([c.astype(np.float64) for c in clean_u8],
                                 raw_rgb48=True))
    gt_planes = list(frames_planes(gt420))
    gt_ref = list(frames_rgb(from_420(gt420)))
    nz = to_420(rgb_clip_from([c.astype(np.float64) for c in noisy],
                              raw_rgb48=True))
    noisy_planes = list(frames_planes(nz))
    noisy_rec = list(frames_rgb(from_420(nz)))
    meas = [float(np.mean([np.std(a[p].astype(np.float64)
                                  - b[p].astype(np.float64))
                           for a, b in zip(noisy_planes, gt_planes)]))
            for p in range(3)]
    # FULL 5-metric set (PSNR/SSIM/LPIPS/DISTS/NIQE) per the bench default —
    # never PSNR-only, LPIPS-first is the house rule.
    r = {"sigma_yuv": [round(m, 2) for m in meas],
         "noisy": agg([metrics5(n, g) for n, g in zip(noisy_rec, gt_ref)])}
    den = core.galosh.Denoise(nz, luma=1.0, chroma=1.0, matrix="bt709",
                              eotf="srgb", range="limited", siting="left",
                              engine="cpu", noise="fit")
    drgb = list(frames_rgb(from_420(den)))
    r["galosh"] = agg([metrics5(d, g) for d, g in zip(drgb, gt_ref)])
    y4m = AVS_WORK / f"gap_{tag}.y4m"
    AVS_WORK.mkdir(parents=True, exist_ok=True)
    write_y4m(y4m, noisy_planes, w_, h_)
    sc = avs_method("bm3d1", y4m.as_posix(), meas, {}, "420")
    dp = run_avs(sc, f"gap_{tag}_bm3d1", len(clean_u8))
    brgb = list(frames_rgb(from_420(planes_to_420clip(dp, w_, h_))))
    r["bm3d1"] = agg([metrics5(d, g) for d, g in zip(brgb, gt_ref)])
    y4m.unlink(missing_ok=True)
    print(f"[cell {tag}] sigma={r['sigma_yuv']} "
          f"noisy={r['noisy']['psnr']:.2f}/{r['noisy']['lpips']:.3f} "
          f"galosh={r['galosh']['psnr']:.2f}/{r['galosh']['lpips']:.3f} "
          f"bm3d1={r['bm3d1']['psnr']:.2f}/{r['bm3d1']['lpips']:.3f} "
          f"gapPSNR={r['galosh']['psnr'] - r['bm3d1']['psnr']:+.2f} "
          f"gapLPIPS={r['bm3d1']['lpips'] - r['galosh']['lpips']:+.3f}",
          flush=True)
    return r


def noise_stats(name, noise_y, gt_y):
    """noise/gt as float [0,255] Y-plane lists."""
    n = np.concatenate([x.ravel() for x in noise_y])
    g = np.concatenate([x.ravel() for x in gt_y])
    bins = np.linspace(0, 255, 9)
    var_curve = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (g >= lo) & (g < hi)
        var_curve.append(round(float(n[m].std()), 2) if m.sum() > 5000
                         else None)
    ac = []
    for x in noise_y[:3]:
        x0 = x - x.mean()
        ac.append(float((x0[:, :-1] * x0[:, 1:]).mean() / max(x0.var(), 1e-9)))
    from scipy.stats import kurtosis
    st = {"sigma_Y": round(float(n.std()), 2),
          "lag1_autocorr": round(float(np.mean(ac)), 3),
          "excess_kurtosis": round(float(kurtosis(n[::7])), 2),
          "var_vs_intensity(8bins)": var_curve}
    print(f"[stats {name}] {st}", flush=True)
    return st


def to_y(rgb):
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    core.std.LoadPlugin(path=str(DLL))
    res = {"cells": {}, "stats": {}}

    # ---- noise statistics at matched sigma ----
    # real CRVD ISO25600 (developed, pre-420)
    ny, gy = [], []
    for s in CRVD_SCENES:
        noisy_f, clean_f = bench_crvd.load_seq(s, "ISO25600")
        ny += [to_y(n) - to_y(c) for n, c in zip(noisy_f[:3], clean_f[:3])]
        gy += [to_y(c) for c in clean_f[:3]]
    res["stats"]["crvd_real_ISO25600"] = noise_stats("crvd_real", ny, gy)
    # synthetic on the same CRVD clean content: phone1600 + crvd25600 params
    for pname, pg in (("phone1600", PHONE1600), ("crvdparams25600", CRVD25600)):
        ny, gy = [], []
        for s in CRVD_SCENES:
            _, clean_f = bench_crvd.load_seq(s, "ISO25600")
            cu8 = [np.clip(np.round(c), 0, 255).astype(np.uint8)
                   for c in clean_f[:3]]
            nz = degrade(cu8, pg, seed=99)
            ny += [to_y(n).astype(np.float64) - to_y(c) for n, c in
                   zip(nz, cu8)]
            gy += [to_y(c) for c in cu8]
        res["stats"][f"synth_{pname}_on_crvd"] = noise_stats(
            f"synth_{pname}", ny, gy)

    # ---- 2x2 cross cells ----
    for seq in SET8_SEQS:
        clean = load_clean(seq_frames(seq)[:NFR])
        res["cells"][f"A_set8-{seq}_phone1600"] = run_cell(
            f"A_{seq}", clean, PHONE1600)
        res["cells"][f"B_set8-{seq}_crvdparams"] = run_cell(
            f"B_{seq}", clean, CRVD25600)
    for s in CRVD_SCENES:
        _, clean_f = bench_crvd.load_seq(s, "ISO25600")
        cu8 = [np.clip(np.round(c), 0, 255).astype(np.uint8) for c in clean_f]
        res["cells"][f"C_crvd-s{s}_phone1600"] = run_cell(
            f"C_s{s}", cu8, PHONE1600)
        res["cells"][f"C2_crvd-s{s}_crvdparams"] = run_cell(
            f"C2_s{s}", cu8, CRVD25600)

    (OUT / "_gap_analysis.json").write_text(json.dumps(res, indent=1))
    print("saved:", OUT / "_gap_analysis.json")


if __name__ == "__main__":
    main()
