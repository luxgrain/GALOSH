#!/usr/bin/env python3
"""CRVD real-noise video denoising benchmark (Track C, 2026-07-12).

The realism track the AWGN Set8 bench can't be: real SONY IMX385 sensor
noise (signal-dependent, ISP-correlated), on 7-frame stop-motion
sequences, 11 indoor scenes x 5 ISO {1600,3200,6400,12800,25600}.  ISO is
the noise-level axis (real, not synthetic sigma).

Pipeline: raw Bayer noisy0 / clean tiff -> IDENTICAL minimal ISP
(black-sub -> normalize -> Malvar demosaic GBRG -> AsShotNeutral WB ->
camera==sRGB -> sRGB OETF) -> developed sRGB RGB video.  The demosaic +
WB + gamma give the noise its real spatial/chroma correlation and
signal dependence (= GALOSH's GAT design regime).

Methods on the developed video (7-frame seq)  [2026-07-14 full-method set]:
  - GALOSH 4 variants via the frameserver DLL (engine{cpu,vulkan} x
    noise{fit,hold}, 420 chroma-R fix; blind) + galosh444 EXE side-check
  - baselines via AVS: bm3d1/vbm3d with ORACLE sigma AND bm3d1b/vbm3db
    with BLIND MAD-estimated sigma, knl d=1, smdegrain tr=3,
    hqdn3d untuned default
  - noisy (developed input) as the floor
Metrics: the 5-metric set + per-plane Y/Cb/Cr PSNR vs developed clean GT.

sigma for the sigma-parameterized baselines = MEASURED per-plane
std(noisy_dev - gt_dev) on the developed 420 planes (honest oracle for
real noise); SMDegrain/KNL frozen tables are indexed by that measured
sigma clamped to the calibrated key range (low-ISO is below calibration
-> flagged).

Usage: python bench_crvd.py [--scenes 1,2,...] [--isos ISO1600,...]
         [--methods galosh420,galosh444,bm3d1,vbm3d,knl,smdegrain]
         [--tag X] [--no-png]
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np
import tifffile
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004 as _demo

sys.path.insert(0, str(Path(__file__).parent))
from bench_set8_video import (rgb_clip_from, to_420, from_420, to_rgb24,
                              frames_planes, frames_rgb, plane_psnr,
                              metrics5, agg, save_png, psnr)
from bench_set8_baselines import (planes_to_420clip, mad_sigma, BLIND,
                                  blind_sig)
from calib_smdegrain import write_y4m, run_avs, WORK as AVS_WORK
import vapoursynth as vs
core = vs.core

CRVD = Path("E:/img_dataset/CRVD/indoor_raw_noisy")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmark" / "results_crvd"
EXE = ROOT / "standalone" / "galosh_yuv_cpu.exe"
DLL = Path(os.environ.get("GALOSH_FS_DLL",
      str(ROOT.parent / "GALOSH-frameserver" / "galosh_frameserver.dll")))
# GALOSH 4 variants via the frameserver DLL (420 chroma-R fix, cpu AND vk) —
# same lanes as the AWGN / PG-noise tracks.  galosh444 stays EXE (444 check).
GVAR = {"galosh-cpu-fit":  dict(engine="cpu",    noise="fit"),
        "galosh-cpu-hold": dict(engine="cpu",    noise="hold"),
        "galosh-vk-fit":   dict(engine="vulkan", noise="fit"),
        "galosh-vk-hold":  dict(engine="vulkan", noise="hold")}
UCRT = r"C:\msys64\ucrt64\bin"
EXTOOLS = (AVS_WORK / "deps" / "ExTools.avsi").as_posix()
BLACK, WHITE = 240, 4095
WB = np.array([1.0 / 0.55, 1.0, 1.0 / 0.60])   # AsShotNeutral reciprocals
ISOS = ["ISO1600", "ISO3200", "ISO6400", "ISO12800", "ISO25600"]
NFR = 7
G420 = ["--pix=420", "--depth=8", "--range=limited", "--matrix=bt709",
        "--eotf=srgb", "--siting=left"]
G444 = ["--pix=444", "--depth=16", "--range=full", "--matrix=bt709",
        "--eotf=srgb"]   # YUV444P16 full-range (matches Set8 444 track)


def srgb_oetf(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, 12.92 * l,
                    1.055 * np.power(l, 1 / 2.4) - 0.055)


def develop(raw):
    """raw uint16 Bayer -> sRGB float [0,255] (identical for noisy & GT)."""
    x = (raw.astype(np.float64) - BLACK) / (WHITE - BLACK)
    x = np.clip(x, 0, None)
    rgb = _demo(x, "GBRG") * WB[None, None, :]
    return srgb_oetf(np.clip(rgb, 0, 1)) * 255.0


def load_seq(scene, iso):
    d = CRVD / f"scene{scene}" / iso
    noisy = [develop(tifffile.imread(str(d / f"frame{f}_noisy0.tiff")))
             for f in range(1, NFR + 1)]
    clean = [develop(tifffile.imread(str(d / f"frame{f}_clean.tiff")))
             for f in range(1, NFR + 1)]
    return noisy, clean


def run_exe(planes, w, h, flags):
    env = dict(os.environ); env["PATH"] = env.get("PATH", "") + os.pathsep + UCRT
    wd = OUT / f"_work_{os.getpid()}"   # pid-namespaced for parallel jobs
    wd.mkdir(parents=True, exist_ok=True)
    inp = wd / "in.bin"; out = wd / "out.bin"
    is444 = "--pix=444" in flags
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


# ---- 444 lane helpers (2026-07-16 unification; protocol mirrors the Set8
# PG-noise 444 track exactly: GALOSH on YUV444P16 full via the DLL,
# baselines on an 8-bit 4:4:4 y4m) ----

def clip444p8(planes, w, h):
    fmt = core.query_video_format(vs.YUV, vs.INTEGER, 8, 0, 0)
    c = core.std.BlankClip(width=w, height=h, format=fmt, length=len(planes))

    def put(n, f, pl=planes):
        fo = f.copy()
        for p in range(3):
            np.asarray(fo[p])[:] = pl[n][p]
        return fo
    return core.std.ModifyFrame(c, c, put)


def to444(frames_f64, depth):
    fmt = vs.YUV444P16 if depth == 16 else vs.YUV444P8
    return core.resize.Bicubic(rgb_clip_from(frames_f64, raw_rgb48=True),
                               format=fmt, matrix_s="709",
                               range_s="full", range_in_s="full")


def write_y4m_444(path, planes_list, w, h):
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F25:1 Ip A1:1 C444\n".encode())
        for pl in planes_list:
            f.write(b"FRAME\n")
            for p in pl:
                f.write(np.ascontiguousarray(p).tobytes())


def avs_method(method, y4m, sig, tables, mode="420"):
    method = BLIND.get(method, method)   # bm3d1b/vbm3db -> same chain,
    src = f'FFVideoSource("{Path(y4m).as_posix()}")\n'   # blind sig via caller
    sy, su, sv = (round(s, 2) for s in sig)
    # 420 y4m -> 444 for BM3D/KNL then back; a 444 y4m needs no resample
    to44 = "" if mode == "444" else "ConvertToYUV444()\n"
    back = "" if mode == "444" else "ConvertToYUV420()\n"
    if method == "bm3d1":
        return (src + f'ConvertBits(32)\n{to44}'
                f'BM3D_CPU(sigma=[{sy},{su},{sv}])\n'
                f'{back}ConvertBits(8)\n')
    if method == "vbm3d":
        return (src + f'ConvertBits(32)\n{to44}'
                f'BM3D_CPU(sigma=[{sy},{su},{sv}], radius=2)\n'
                f'BM3D_VAggregate(radius=2)\n{back}ConvertBits(8)\n')
    if method == "knl":
        return (src + to44 +
                f'KNLMeansCL(d=1, a=2, h={tables["knl"]}, channels="YUV")\n'
                + back)
    if method == "smdegrain":
        # [DEPRECATED 2026-07-16] dropped from the unified set (frozen thSAD
        # pass-through defect at CRVD/high-ISO); kept for legacy shards only.
        return (f'Import("{EXTOOLS}")\n' + src +
                f'SMDegrain(tr=3, thSAD={tables["smdegrain"]}, prefilter=2, '
                'contrasharp=false)\n')
    if method == "hqdn3d":
        return src + 'hqdn3d()\n'      # untuned-default reference
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="420", choices=["420", "444"])
    ap.add_argument("--scenes", default=",".join(str(i) for i in range(1, 12)))
    ap.add_argument("--isos", default=",".join(ISOS))
    # [2026-07-16 unification] galosh444 side-check + smdegrain dropped from
    # the default set (galosh444 archived; smdegrain thSAD pass-through).
    ap.add_argument("--methods",
                    default="galosh-cpu-fit,galosh-cpu-hold,galosh-vk-fit,"
                            "galosh-vk-hold,bm3d1,bm3d1b,vbm3d,"
                            "vbm3db,knl,hqdn3d")
    ap.add_argument("--tag", default="")
    # [2026-07-17 generator-fidelity track] inject SYNTHETIC noise on the
    # developed clean instead of loading the real noisy frames:
    #   --synth imx385  -> CRVD-measured per-ISO (a,b) (sensor_pg_library),
    #                      injected via tools/pgnoise v2 (measured banding)
    #   --synth phone   -> phone trimmed-median curve at the same ISO label
    # Everything downstream (sigma measurement, methods, metrics) is
    # IDENTICAL to the real-noise track, so real-vs-synth tables compare
    # generator fidelity method-by-method.  GT stays the developed clean.
    #   --synth phone-match -> GENERIC phone curve, sigma-matched per ISO
    #                      (equiv-ISO from _synth_phone_match.json; the
    #                      generality test: no target-sensor measurement)
    ap.add_argument("--synth", default="",
                    choices=["", "imx385", "phone", "phone-match"])
    # PNG persistence ON BY DEFAULT (persist-everything); --no-png is
    # user-explicit only.
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()
    methods = args.methods.split(",")
    mode = args.mode
    if args.synth:
        sys.path.insert(0, str(ROOT / "tools"))
        import pgnoise as pgn
        pg_tab = json.loads((ROOT / "benchmark" / "results_set8_pgnoise" /
                             "sensor_pg_library.json").read_text())
        pm_tab = None
        if args.synth == "phone-match":
            pm_tab = json.loads((OUT / "_synth_phone_match.json").read_text())

        def synth_ab(iso):
            if args.synth == "imx385":
                e = pg_tab["CRVD_IMX385"][iso]
                return e["a"], e["b"]
            if args.synth == "phone-match":
                return pm_tab[iso]["a"], pm_tab[iso]["b"]
            return pgn.iso_to_pg(float(iso.replace("ISO", "")))
    if any(m in GVAR for m in methods):
        core.std.LoadPlugin(path=str(DLL))
    OUT.mkdir(parents=True, exist_ok=True)
    thsad = json.loads((ROOT / "benchmark" / "results_set8_awgn" /
                        "_smdegrain_thsad.json").read_text())["table"]
    knl = json.loads((ROOT / "benchmark" / "results_set8_awgn" /
                      "_knl_h.json").read_text())["table"]

    def keyclamp(s):     # measured sigma -> nearest calibrated table key
        return str(min([10, 20, 30, 40, 50], key=lambda k: abs(k - s)))

    def run_env():
        """Provenance for the shard JSONs (mirrors bench_set8_pgnoise)."""
        import hashlib

        def sha(p):
            try:
                return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
            except OSError:
                return None
        return {"dll_sha256_16": sha(DLL), "exe_sha256_16": sha(EXE),
                "vapoursynth": getattr(vs.core, "version_number",
                                       lambda: None)(),
                "numpy": np.__version__, "python": sys.version.split()[0]}

    results = {"_env": run_env()}
    synth_sfx = ("" if not args.synth else
                 ("_synth" if args.synth == "imx385" else "_synth_phone"))
    pngroot = OUT / (("png" if mode == "420" else "png444") + synth_sfx)
    for scene in args.scenes.split(","):
        results[f"scene{scene}"] = {}
        for iso in args.isos.split(","):
            noisy_f, clean_f = load_seq(scene, iso)
            if args.synth:
                # generator-fidelity: same developed clean, synthetic noise
                # via tools/pgnoise v2 (measured banding, split seeds)
                a_s, b_s = synth_ab(iso)
                cu8 = [np.clip(np.round(c), 0, 255).astype(np.uint8)
                       for c in clean_f]
                noisy_f = [pgn.add_pg_noise(
                               f, a_s, b_s, seed_shot=311, seed_read=313,
                               frame_idx=t).astype(np.float64)
                           for t, f in enumerate(cu8)]
            h_, w_ = noisy_f[0].shape[:2]
            AVS_WORK.mkdir(parents=True, exist_ok=True)
            if mode == "420":
                gt420 = to_420(rgb_clip_from([c for c in clean_f],
                                             raw_rgb48=True))
                gt_planes = list(frames_planes(gt420))
                gt_ref = list(frames_rgb(from_420(gt420)))
                nzclip = to_420(rgb_clip_from(noisy_f, raw_rgb48=True))
                noisy_planes = list(frames_planes(nzclip))
                noisy_rec = list(frames_rgb(from_420(nzclip)))
                meas = [float(np.mean([np.std(a[p].astype(np.float64)
                                              - b[p].astype(np.float64))
                                       for a, b in zip(noisy_planes,
                                                       gt_planes)]))
                        for p in range(3)]
                bsig = mad_sigma(noisy_planes)   # blind (no GT), for *b twins
                y4m = AVS_WORK / f"crvd_{args.tag}_s{scene}_{iso}.y4m"
                write_y4m(y4m, noisy_planes, w_, h_)
            else:
                # 444 lane (unification): GALOSH on YUV444P16 full via DLL,
                # baselines on 8-bit 4:4:4 y4m — the Set8 PG-444 protocol.
                gtc = to444(clean_f, 16)
                gt_planes = list(frames_planes(gtc))
                gt_ref = list(frames_rgb(to_rgb24(gtc)))
                nzclip = to444(noisy_f, 16)
                noisy_rec = list(frames_rgb(to_rgb24(nzclip)))
                np8 = list(frames_planes(to444(noisy_f, 8)))
                gp8 = list(frames_planes(to444(clean_f, 8)))
                meas = [float(np.mean([np.std(a[p].astype(np.float64)
                                              - b[p].astype(np.float64))
                                       for a, b in zip(np8, gp8)]))
                        for p in range(3)]
                bsig = mad_sigma(np8)
                y4m = AVS_WORK / f"crvd444_{args.tag}_s{scene}_{iso}.y4m"
                write_y4m_444(y4m, np8, w_, h_)
            ent = {"measured_sigma_yuv": [round(m, 2) for m in meas],
                   "blind_sigma_yuv": [round(s, 2) for s in bsig]}
            if args.synth:
                ent["synth_pg"] = [args.synth, round(a_s, 8), round(b_s, 8)]
            ent["noisy"] = agg([metrics5(nn, rr)
                                for nn, rr in zip(noisy_rec, gt_ref)])
            if mode == "420":
                ent["noisy"]["cr_psnr"] = float(np.mean(
                    [plane_psnr(a[2], b[2], 255)
                     for a, b in zip(noisy_planes, gt_planes)]))
            if not args.no_png:     # viewer refs: noisy + GT, once per cell
                for i, o in enumerate(noisy_rec):
                    save_png(o, pngroot / f"scene{scene}" / iso / "noisy"
                             / f"{i:02d}.png")
                for i, o in enumerate(gt_ref):
                    save_png(o, pngroot / f"scene{scene}" / iso / "gt_ref"
                             / f"{i:02d}.png")

            for m in methods:
                try:
                    if m in GVAR:
                        if mode == "420":
                            den_c = core.galosh.Denoise(
                                nzclip, luma=1.0, chroma=1.0, matrix="bt709",
                                eotf="srgb", range="limited", siting="left",
                                **GVAR[m])
                            drgb = list(frames_rgb(from_420(den_c)))
                        else:
                            den_c = core.galosh.Denoise(
                                nzclip, luma=1.0, chroma=1.0, matrix="bt709",
                                eotf="srgb", range="full", **GVAR[m])
                            drgb = list(frames_rgb(to_rgb24(den_c)))
                        dp = list(frames_planes(den_c))
                    elif m == "galosh444":
                        # [ARCHIVED 2026-07-16] legacy EXE side-check; kept
                        # runnable for reproducing the archived row only.
                        n444 = list(frames_planes(core.resize.Bicubic(
                            rgb_clip_from(noisy_f, raw_rgb48=True),
                            format=vs.YUV444P16, matrix_s="709",
                            range_s="full", range_in_s="full")))
                        den = run_exe(n444, w_, h_, G444)
                        drgb = list(frames_rgb(to_rgb24(
                            clip444p16(den, w_, h_))))
                        dp = None
                    else:
                        tb = {}
                        if m == "smdegrain":
                            tb["smdegrain"] = thsad[keyclamp(meas[0])]
                        if m == "knl":
                            tb["knl"] = knl[keyclamp(meas[0])]
                        # blind twins: raw / guarded MAD; others oracle
                        sig_in = blind_sig(m, bsig, meas)
                        sc = avs_method(m, y4m.as_posix(), sig_in, tb, mode)
                        dp = run_avs(sc, f"crvd{mode}_{args.tag}_s{scene}"
                                     f"_{iso}_{m}", NFR)
                        if mode == "420":
                            drgb = list(frames_rgb(from_420(
                                planes_to_420clip(dp, w_, h_))))
                        else:
                            drgb = list(frames_rgb(to_rgb24(
                                clip444p8(dp, w_, h_))))
                    e = agg([metrics5(o, r) for o, r in zip(drgb, gt_ref)])
                    if dp is not None and (mode == "420" or m in GVAR):
                        e["cr_psnr"] = float(np.mean(
                            [plane_psnr(a[2], b[2], 255)
                             for a, b in zip(dp, gt_planes)]))
                    ent[m] = e
                    # persist-everything: save EVERY method's frames
                    if not args.no_png:
                        for i, o in enumerate(drgb):
                            save_png(o, pngroot / f"scene{scene}" / iso / m
                                     / f"{i:02d}.png")
                    print(f"[crvd s{scene} {iso} {m}] PSNR {e['psnr']:.2f} "
                          f"SSIM {e['ssim']:.4f} LPIPS {e['lpips']:.4f}",
                          flush=True)
                except Exception as ex:
                    print(f"[crvd s{scene} {iso} {m}] FAILED: "
                          f"{str(ex)[-160:]}", flush=True)
            y4m.unlink()
            results[f"scene{scene}"][iso] = ent
            stem = ("_metrics_crvd"
                    + ("" if not args.synth else
                       ("synth" if args.synth == "imx385" else "synthphone"))
                    + ("444" if mode == "444" else ""))
            outf = OUT / (f"{stem}{('_' + args.tag) if args.tag else ''}"
                          ".json")
            outf.write_text(json.dumps(results, indent=1))
    print("saved:", outf)


if __name__ == "__main__":
    main()
