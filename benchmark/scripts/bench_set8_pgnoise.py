#!/usr/bin/env python3
"""PG-noise track bench — realistic signal-dependent noise on Set8 (2026-07-12).

The realism-track replacement for flat AWGN: clean Set8 sRGB -> PG-noise track
degradation (unprocess -> ISO-calibrated Poisson-Gaussian raw noise ->
re-ISP; NO compression in the main bench) -> the two standard tracks
(444 full / 420 limited left), GALOSH-fix + baselines, 5 metrics.

FAIR GT (matches CRVD / Set8 convention): the NOISE-FREE degradation
(roundtripped clean) put through each track's YUV format, so the demosaic
softening AND the chroma subsampling both cancel and only noise removal
is measured:
    444 GT = degrade(clean, noise=0) -> YUV444P16 full -> RGB
    420 GT = degrade(clean, noise=0) -> to_420 -> from_420 -> RGB
Noise level axis = ISO (CRVD IMX385 calibrated curve, the canonical axis).
Seeded per (seq, ISO) for reproducibility.

Usage: python bench_set8_pgnoise.py [--mode 444|420] [--seqs ...]
         [--isos ISO1600,...] [--methods ...] [--tag X] [--no-png]
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_set8_video import (DERF, GOPRO, seq_frames, load_clean,
                              rgb_clip_from, to_420, from_420, to_rgb24,
                              frames_planes, frames_rgb, plane_psnr,
                              metrics5, agg, save_png)
from bench_set8_baselines import planes_to_420clip
from calib_smdegrain import write_y4m, run_avs, WORK as AVS_WORK
from degrade_set8_pgnoise import (degrade_bcore, compress_h264, sample_pg, PG,
                            A_RANGE, B_RANGE)
import vapoursynth as vs
core = vs.core

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmark" / "results_set8_pgnoise"
EXE = ROOT / "standalone" / "galosh_yuv_cpu.exe"
DLL = Path(os.environ.get("GALOSH_FS_DLL",
      str(ROOT.parent / "GALOSH-frameserver" / "galosh_frameserver.dll")))
UCRT = r"C:\msys64\ucrt64\bin"
EXTOOLS = (AVS_WORK / "deps" / "ExTools.avsi").as_posix()
ISOS = list(PG.keys())
# GALOSH via the frameserver DLL (has the 420 chroma-R fix in cpu AND vk):
# 4 variants = engine {cpu, vulkan} x noise {fit, hold}.
GVAR = {"galosh-cpu-fit":  dict(engine="cpu",    noise="fit"),
        "galosh-cpu-hold": dict(engine="cpu",    noise="hold"),
        "galosh-vk-fit":   dict(engine="vulkan", noise="fit"),
        "galosh-vk-hold":  dict(engine="vulkan", noise="hold")}
G420 = ["--pix=420", "--depth=8", "--range=limited", "--matrix=bt709",
        "--eotf=srgb", "--siting=left"]
G444 = ["--pix=444", "--depth=16", "--range=full", "--matrix=bt709",
        "--eotf=srgb"]
# per-plane floor for the GUARDED blind twins (bm3d1bg/vbm3dbg): the smallest
# nonzero step of the 8-bit Haar-MAD estimator is ~0.74, so 0.5 only engages
# on exact-zero planes.
SIGMA_FLOOR = 0.5


def run_env():
    """Provenance for the shard JSONs: binary hashes + library versions."""
    import hashlib
    def sha(p):
        try:
            return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
        except OSError:
            return None
    return {"dll_sha256_16": sha(DLL), "exe_sha256_16": sha(EXE),
            "vapoursynth": getattr(vs.core, "version_number", lambda: None)(),
            "numpy": np.__version__,
            "python": sys.version.split()[0]}


def degrade_seq(clean, seed, noise=True, pg=None):
    """list of clean sRGB u8 -> degraded (or noise-free roundtrip) u8.
    pg = explicit (a, b): B-core passes PG[iso]; B-div passes a library
    sample from sample_pg (a~log-uniform 1e-4..1e-2, b~uniform 0..1.5e-3)."""
    rng = np.random.default_rng(seed)
    if not noise:
        return [degrade_bcore(c, pg=(0.0, 0.0), rng=rng, banding=0.0)
                for c in clean]
    return [degrade_bcore(c, pg=pg, rng=rng) for c in clean]


def run_exe(planes, w, h, flags):
    env = dict(os.environ); env["PATH"] = env.get("PATH", "") + os.pathsep + UCRT
    wd = OUT / f"_work_{os.getpid()}"; wd.mkdir(parents=True, exist_ok=True)
    inp = wd / "in.bin"; out = wd / "out.bin"; is444 = "--pix=444" in flags
    res = []
    for pl in planes:
        inp.write_bytes(b"".join(np.ascontiguousarray(p).tobytes() for p in pl))
        r = subprocess.run([str(EXE), str(inp), str(out), str(w), str(h),
                            "1.0", "1.0", "0", "0"] + flags,
                           capture_output=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode("utf-8", "replace")[-300:])
        d = np.fromfile(out, np.uint16 if is444 else np.uint8)
        if is444:
            res.append([d[i * w * h:(i + 1) * w * h].reshape(h, w)
                        for i in range(3)])
        else:
            res.append([d[:w * h].reshape(h, w),
                        d[w * h:w * h + w * h // 4].reshape(h // 2, w // 2),
                        d[w * h + w * h // 4:].reshape(h // 2, w // 2)])
    return res


def clip444p16(planes, w, h):
    fmt = core.query_video_format(vs.YUV, vs.INTEGER, 16, 0, 0)
    c = core.std.BlankClip(width=w, height=h, format=fmt, length=len(planes))

    def put(n, f, pl=planes):
        fo = f.copy()
        for p in range(3):
            np.asarray(fo[p])[:] = pl[n][p]
        return fo
    return core.std.ModifyFrame(c, c, put)


def to_444p16(frames_u8):
    return core.resize.Bicubic(rgb_clip_from([f.astype(np.float64)
                                              for f in frames_u8],
                                             raw_rgb48=True),
                               format=vs.YUV444P16, matrix_s="709",
                               range_s="full", range_in_s="full")


def to_444p8(frames_u8):
    return core.resize.Bicubic(rgb_clip_from([f.astype(np.float64)
                                              for f in frames_u8],
                                             raw_rgb48=True),
                               format=vs.YUV444P8, matrix_s="709",
                               range_s="full", range_in_s="full")


def write_y4m_444(path, planes_list, w, h):
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F25:1 Ip A1:1 C444\n".encode())
        for pl in planes_list:
            f.write(b"FRAME\n")
            for p in pl:
                f.write(np.ascontiguousarray(p).tobytes())


def mad_sigma(planes_list):
    """BLIND per-plane sigma from the NOISY frames only (no GT): single-level
    Haar HH detail, sigma-hat = MAD/0.6745 (Donoho & Johnstone 1994), median
    across frames.  This is what a real user without sigma would run; on
    signal-dependent PG noise it returns one global effective sigma (the
    misspecification is the point of the blind axis)."""
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


def avs_method(m, y4m, sig, tables, mode="420"):
    src = f'FFVideoSource("{Path(y4m).as_posix()}")\n'
    sy, su, sv = (round(s, 2) for s in sig)
    # 420 y4m needs ->444 for BM3D/KNL then back to 420; a 444 y4m is already
    # 4:4:4 so no chroma resample.
    to44 = "" if mode == "444" else "ConvertToYUV444()\n"
    back = "" if mode == "444" else "ConvertToYUV420()\n"
    # bm3d1b / vbm3db = sigma-BLIND twins of bm3d1 / vbm3d: identical filter,
    # sigma comes from mad_sigma(noisy) instead of the measured-vs-GT oracle
    # (caller passes the blind estimate via `sig`).  bm3d1bg / vbm3dbg =
    # GUARDED blind twins: same MAD estimate clamped to a per-plane floor
    # (SIGMA_FLOOR) so a zero MAD (H.264-smoothed / 420-upsampled chroma has
    # no finest-scale Haar energy) cannot feed sigma=0 into the pipeline —
    # unguarded sigma=0 breaks the V-BM3D chain catastrophically.
    if m in ("bm3d1", "bm3d1b", "bm3d1bg"):
        return (src + f'ConvertBits(32)\n{to44}'
                f'BM3D_CPU(sigma=[{sy},{su},{sv}])\n{back}ConvertBits(8)\n')
    if m in ("vbm3d", "vbm3db", "vbm3dbg"):
        return (src + f'ConvertBits(32)\n{to44}'
                f'BM3D_CPU(sigma=[{sy},{su},{sv}], radius=2)\n'
                f'BM3D_VAggregate(radius=2)\n{back}ConvertBits(8)\n')
    if m == "knl":
        return (src + to44 +
                f'KNLMeansCL(d=1, a=2, h={tables["knl"]}, channels="YUV")\n'
                + back)
    if m == "smdegrain":
        return (f'Import("{EXTOOLS}")\n' + src +
                f'SMDegrain(tr=3, thSAD={tables["smdegrain"]}, prefilter=2, '
                'contrasharp=false)\n')
    if m == "hqdn3d":
        return src + "hqdn3d()\n"      # untuned-default reference
    raise ValueError(m)


def clip444p8(planes, w, h):
    fmt = core.query_video_format(vs.YUV, vs.INTEGER, 8, 0, 0)
    c = core.std.BlankClip(width=w, height=h, format=fmt, length=len(planes))

    def put(n, f, pl=planes):
        fo = f.copy()
        for p in range(3):
            np.asarray(fo[p])[:] = pl[n][p]
        return fo
    return core.std.ModifyFrame(c, c, put)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="420", choices=["420", "444"])
    ap.add_argument("--seqs", default=",".join(DERF + GOPRO))
    ap.add_argument("--isos", default=",".join(ISOS))
    ap.add_argument("--methods", default="")
    ap.add_argument("--tag", default="")
    # PNG persistence is ON BY DEFAULT (persist-everything rule: all methods +
    # GT + Noisy).  --no-png discards ALL denoise artifacts and must ONLY be
    # passed when the user explicitly asks for a metrics-only run.
    ap.add_argument("--no-png", action="store_true",
                    help="DISCARD all denoise artifacts (user-explicit only; "
                         "violates the persist-everything rule otherwise)")
    ap.add_argument("--diversity", action="store_true",
                    help="per-seq sensor gain (mode 2): scale (a,b) by a "
                         "log-uniform(0.3,3.0) factor per sequence")
    ap.add_argument("--compress-crf", type=int, default=0,
                    help="B-full: H.264 round-trip on the noisy video at this "
                         "CRF (0=off; GT stays uncompressed)")
    ap.add_argument("--limit-frames", type=int, default=0,
                    help="cap frames per sequence (0 = full)")
    args = ap.parse_args()
    if args.no_png:
        print("WARNING: --no-png set — discarding ALL denoise artifacts "
              "(metrics-only run).", file=sys.stderr, flush=True)
    mode = args.mode
    SEQ_ORDER = DERF + GOPRO   # stable index for reproducible per-seq gain
    if not args.methods:
        gal = "galosh-cpu-fit,galosh-cpu-hold,galosh-vk-fit,galosh-vk-hold"
        # [2026-07-16 unification] identical default set on 420 and 444;
        # smdegrain dropped (frozen-thSAD pass-through defect — the
        # avs_method branch is kept for legacy-shard reproduction only).
        args.methods = f"{gal},bm3d1,bm3d1b,vbm3d,vbm3db,knl,hqdn3d"
    core.std.LoadPlugin(path=str(DLL))
    methods = args.methods.split(",")
    thsad = json.loads((ROOT / "benchmark" / "results_set8_awgn" /
                        "_smdegrain_thsad.json").read_text())["table"]
    knl = json.loads((ROOT / "benchmark" / "results_set8_awgn" /
                      "_knl_h.json").read_text())["table"]

    def keyclamp(s):
        return str(min([10, 20, 30, 40, 50], key=lambda k: abs(k - s)))

    OUT.mkdir(parents=True, exist_ok=True)
    # variant PNGs go to a namespaced dir so they persist without clobbering
    # the base track (e.g. png420_div, png420_cmp23).  Base = png/png420.
    variant = ("_div" if args.diversity else "") + \
              (f"_cmp{args.compress_crf}" if args.compress_crf > 0 else "")
    pngroot = OUT / (("png" if mode == "444" else "png420") + variant)
    results = {"_env": run_env()}   # provenance: binary hashes + versions
    for seq in args.seqs.split(","):
        files = seq_frames(seq)
        if args.limit_frames:
            files = files[:args.limit_frames]
        clean = load_clean(files)
        n = len(clean); h_, w_ = clean[0].shape[:2]
        results[seq] = {"n_frames": n}
        # noise levels: B-core/B-comp sweep the ISO axis (IMX385 curve);
        # B-div draws (a,b) from the multi-sensor library distribution per
        # slot — the AGREED sensor-diversity design: a ~ log-uniform
        # (1e-4,1e-2) [log-median ~7e-4], b ~ uniform(0,1.5e-3).  7 draws/seq.
        gi = SEQ_ORDER.index(seq) if seq in SEQ_ORDER else 0
        if args.diversity:
            levels = [(f"draw{di}",
                       sample_pg(np.random.default_rng(80000 + gi * 100 + di)),
                       8000 + gi * 100 + di) for di in range(7)]
        else:
            levels = [(iso, PG[iso], 7000 + int(iso.replace("ISO", "")))
                      for iso in args.isos.split(",")]
        for key, pg, seed in levels:
            noisy_f = degrade_seq(clean, seed, noise=True, pg=pg)
            if args.compress_crf > 0:                # B-full: compress noisy
                noisy_f = compress_h264(
                    [np.ascontiguousarray(c, np.uint8) for c in noisy_f],
                    crf=args.compress_crf)
            cleanrt = degrade_seq(clean, seed, noise=False)   # fair base
            ent = {"pg": [round(pg[0], 8), round(pg[1], 8)]}
            if mode == "420":
                gt420 = to_420(rgb_clip_from([c.astype(np.float64)
                                              for c in cleanrt], raw_rgb48=True))
                gt_planes = list(frames_planes(gt420))
                gt_ref = list(frames_rgb(from_420(gt420)))
                nz = to_420(rgb_clip_from([c.astype(np.float64)
                                           for c in noisy_f], raw_rgb48=True))
                noisy_planes = list(frames_planes(nz))
                noisy_rec = list(frames_rgb(from_420(nz)))
                meas = [float(np.mean([np.std(a[p].astype(np.float64)
                                             - b[p].astype(np.float64))
                                      for a, b in zip(noisy_planes, gt_planes)]))
                        for p in range(3)]
                ent["measured_sigma_yuv"] = [round(m, 2) for m in meas]
                ent["noisy"] = agg([metrics5(nn, rr)
                                    for nn, rr in zip(noisy_rec, gt_ref)])
                ent["noisy"]["cr_psnr"] = float(np.mean(
                    [plane_psnr(a[2], b[2], 255)
                     for a, b in zip(noisy_planes, gt_planes)]))
                # tag-namespaced so parallel core/cmp jobs don't collide
                y4m = AVS_WORK / f"pg_{args.tag}_{seq}_{key}.y4m"
                AVS_WORK.mkdir(parents=True, exist_ok=True)
                write_y4m(y4m, noisy_planes, w_, h_)
                nzclip = nz          # 420 noisy clip for the DLL
            else:
                gtc = to_444p16([c.astype(np.uint8) for c in cleanrt])
                gt_ref = list(frames_rgb(to_rgb24(gtc)))
                gt_planes = list(frames_planes(gtc))
                nzclip = to_444p16([c.astype(np.uint8) for c in noisy_f])
                noisy_planes = list(frames_planes(nzclip))
                # 444 baselines (bm3d1/knl): 8-bit 4:4:4 y4m + measured sigma
                np8 = list(frames_planes(
                    to_444p8([c.astype(np.uint8) for c in noisy_f])))
                gp8 = list(frames_planes(
                    to_444p8([c.astype(np.uint8) for c in cleanrt])))
                meas = [float(np.mean([np.std(a[p].astype(np.float64)
                                              - b[p].astype(np.float64))
                                       for a, b in zip(np8, gp8)]))
                        for p in range(3)]
                ent["measured_sigma_yuv"] = [round(m, 2) for m in meas]
                y4m = AVS_WORK / f"pg_{args.tag}_{seq}_{key}.y4m"
                AVS_WORK.mkdir(parents=True, exist_ok=True)
                write_y4m_444(y4m, np8, w_, h_)
                noisy_rec = list(frames_rgb(to_rgb24(clip444p16(
                    noisy_planes, w_, h_))))
                ent["noisy"] = agg([metrics5(nn, rr)
                                    for nn, rr in zip(noisy_rec, gt_ref)])

            # BLIND sigma from the same 8-bit planes the y4m holds (no GT):
            # feeds the bm3d1b/vbm3db sigma-blind twins.
            bsig = mad_sigma(noisy_planes if mode == "420" else np8)
            ent["blind_sigma_yuv"] = [round(s, 2) for s in bsig]
            if not args.no_png:      # viewer refs: noisy + GT, once per cell
                for i, o in enumerate(noisy_rec):
                    save_png(o, pngroot / seq / key / "noisy" / f"{i:04d}.png")
                for i, o in enumerate(gt_ref):
                    save_png(o, pngroot / seq / key / "gt_ref" / f"{i:04d}.png")
            for m in methods:
                try:
                    if m in GVAR:
                        kw = dict(luma=1.0, chroma=1.0, matrix="bt709",
                                  eotf="srgb", **GVAR[m])
                        if mode == "420":
                            den_c = core.galosh.Denoise(
                                nzclip, range="limited", siting="left", **kw)
                            drgb = list(frames_rgb(from_420(den_c)))
                        else:
                            den_c = core.galosh.Denoise(
                                nzclip, range="full", **kw)
                            drgb = list(frames_rgb(to_rgb24(den_c)))
                        dp = list(frames_planes(den_c))
                    else:
                        tb = {}
                        if m == "smdegrain":
                            tb["smdegrain"] = thsad[keyclamp(meas[0])]
                        if m == "knl":
                            tb["knl"] = knl[keyclamp(meas[0])]
                        # blind twins get the MAD estimate (guarded twins the
                        # floor-clamped one); everything else the oracle
                        if m in ("bm3d1bg", "vbm3dbg"):
                            sig_in = [max(s, SIGMA_FLOOR) for s in bsig]
                        elif m in ("bm3d1b", "vbm3db"):
                            sig_in = bsig
                        else:
                            sig_in = meas
                        sc = avs_method(m, y4m.as_posix(), sig_in, tb, mode)
                        dp = run_avs(sc, f"pg_{args.tag}_{seq}_{key}_{m}", n)
                        if mode == "420":
                            drgb = list(frames_rgb(from_420(
                                planes_to_420clip(dp, w_, h_))))
                        else:
                            drgb = list(frames_rgb(to_rgb24(
                                clip444p8(dp, w_, h_))))
                    e = agg([metrics5(o, r) for o, r in zip(drgb, gt_ref)])
                    if mode == "420" or m in GVAR:
                        e["cr_psnr"] = float(np.mean(
                            [plane_psnr(a[2], b[2], 255)
                             for a, b in zip(dp, gt_planes)]))
                    ent[m] = e
                    # persist-everything rule: save EVERY method (+ GT/Noisy
                    # above), not a curated subset.  PNG is the default; only
                    # an explicit user-requested --no-png skips it.
                    if not args.no_png:
                        for i, o in enumerate(drgb):
                            save_png(o, pngroot / seq / key / m
                                     / f"{i:04d}.png")
                    print(f"[pg {mode} {seq} {key} {m}] PSNR {e['psnr']:.2f} "
                          f"SSIM {e['ssim']:.4f} LPIPS {e['lpips']:.4f}",
                          flush=True)
                except Exception as ex:
                    print(f"[tb {mode} {seq} {key} {m}] FAILED: "
                          f"{str(ex)[-160:]}", flush=True)
            y4m.unlink(missing_ok=True)
            results[seq][key] = ent
            outf = OUT / (f"_metrics_pg_{mode}"
                          f"{('_' + args.tag) if args.tag else ''}.json")
            outf.write_text(json.dumps(results, indent=1))
    print("saved:", outf)


if __name__ == "__main__":
    main()
