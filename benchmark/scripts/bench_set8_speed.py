#!/usr/bin/env python3
"""Set8 speed benchmark — clean run AFTER all quality benches (GPU idle).

What is measured (wall clock, serial in-order frame pulls = the hold-
semantics-correct way a frameserver actually serves):
  - VS lane : GALOSH-frameserver 4 variants (engine cpu/vulkan x noise
    fit/hold) through VapourSynth; source = RAM-preloaded planes via
    ModifyFrame; reported per-frame ms is NET of the measured
    source-only pull cost (same clip, no filter).
  - AVS lane: GALOSH (same DLL, AviSynth host) + baselines
    (BM3D_CPU / V-BM3D / KNLMeansCL / SMDegrain / hqdn3d) through
    tools/avs_dump.exe, output to NUL; per-frame ms is NET of a
    source-only (FFVideoSource, no filter) run of the same script.
    FFIndex is pre-warmed so indexing never lands in a timed run.

Inputs: 540p = Set8 tractor PNGs + seeded sigma20 AWGN -> YUV420P8 y4m
        (the bench-track conversion); 1080p = real derf tractor y4m
        (C420, content irrelevant for speed); 2160p = 2x2 tile of the
        1080p planes (GALOSH VS lane only).
Baseline knobs at sigma20 frozen tables: thSAD=864, KNL h=6, BM3D
sigma=[5,5,5] (sigma value does not change BM3D's speed).

Output: benchmark/results_set8_awgn/_speed_set8.json
"""
import json, subprocess, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_set8_video import (DLL, OUT, seq_frames, load_clean, make_noisy,
                              rgb_clip_from, to_420, frames_planes)
from calib_smdegrain import write_y4m, WORK, AVS_DUMP
import vapoursynth as vs
core = vs.core

EXTOOLS = (WORK / "deps" / "ExTools.avsi").as_posix()
DERF_Y4M = Path("E:/img_dataset/Set8/derf_y4m/tractor_1080p25.y4m")
WARMUP = 5


def read_y4m_420(path, n_max):
    with open(path, "rb") as f:
        hdr = f.readline().split()
        w = int([x for x in hdr if x[:1] == b"W"][0][1:])
        h = int([x for x in hdr if x[:1] == b"H"][0][1:])
        fr = []
        while len(fr) < n_max:
            m = f.readline()
            if not m.startswith(b"FRAME"):
                break
            y = np.frombuffer(f.read(w * h), np.uint8).reshape(h, w)
            u = np.frombuffer(f.read(w * h // 4), np.uint8).reshape(h // 2, w // 2)
            v = np.frombuffer(f.read(w * h // 4), np.uint8).reshape(h // 2, w // 2)
            fr.append([y, u, v])
    return fr, w, h


def clip_from_planes(planes_list, w, h):
    fmt = core.query_video_format(vs.YUV, vs.INTEGER, 8, 1, 1)
    c = core.std.BlankClip(width=w, height=h, format=fmt,
                           length=len(planes_list))

    def put(n, f, pl=planes_list):
        fo = f.copy()
        for p in range(3):
            np.asarray(fo[p])[:] = pl[n][p]
        return fo
    return core.std.ModifyFrame(c, c, put)


def pull_ms(clip, n):
    for i in range(min(WARMUP, n)):        # warmup / hold state build
        clip.get_frame(i)
    t0 = time.perf_counter()
    for i in range(WARMUP, n):
        clip.get_frame(i)
    return (time.perf_counter() - t0) * 1000.0 / (n - WARMUP)


def vs_lane(src_clip, n, variants):
    res = {}
    src_ms = pull_ms(src_clip, n)
    res["_source_ms"] = round(src_ms, 2)
    for name, kw in variants.items():
        den = core.galosh.Denoise(src_clip, luma=1.0, chroma=1.0,
                                  matrix="bt709", range="limited",
                                  eotf="srgb", siting="left", **kw)
        ms = pull_ms(den, n) - src_ms
        res[name] = round(ms, 2)
        print(f"  [vs {name}] {ms:.1f} ms/f ({1000.0/ms:.1f} fps)",
              flush=True)
    return res


def avs_wall(script_text, tag, n):
    avs = WORK / f"spd_{tag}.avs"
    avs.write_text(script_text)
    t0 = time.perf_counter()
    r = subprocess.run([str(AVS_DUMP), str(avs), "NUL", str(n)],
                       capture_output=True, timeout=3600)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(tag + ": " +
                           r.stderr.decode("utf-8", "replace")[-300:])
    return dt


def avs_lane(y4m_path, n, sigma_knobs):
    src = f'FFVideoSource("{Path(y4m_path).as_posix()}")\n'
    dll = DLL.as_posix()
    thsad, knl_h = sigma_knobs
    scripts = {
        "galosh cpu-fit": (f'LoadPlugin("{dll}")\n' + src +
                           'galosh_Denoise(engine="cpu", noise="fit", '
                           'matrix="bt709", range="limited", eotf="srgb", '
                           'siting="left")\n'),
        "galosh vk-fit": (f'LoadPlugin("{dll}")\n' + src +
                          'galosh_Denoise(engine="vulkan", noise="fit", '
                          'matrix="bt709", range="limited", eotf="srgb", '
                          'siting="left")\n'),
        "galosh vk-hold": (f'LoadPlugin("{dll}")\n' + src +
                           'galosh_Denoise(engine="vulkan", noise="hold", '
                           'matrix="bt709", range="limited", eotf="srgb", '
                           'siting="left")\n'),
        "bm3d1": (src + 'ConvertBits(32)\nConvertToYUV444()\n'
                  'BM3D_CPU(sigma=[5,5,5])\n'
                  'ConvertToYUV420()\nConvertBits(8)\n'),
        "vbm3d": (src + 'ConvertBits(32)\nConvertToYUV444()\n'
                  'BM3D_CPU(sigma=[5,5,5], radius=2)\n'
                  'BM3D_VAggregate(radius=2)\n'
                  'ConvertToYUV420()\nConvertBits(8)\n'),
        "hqdn3d": src + 'hqdn3d()\n',
        "knl": (src + 'ConvertToYUV444()\n'
                f'KNLMeansCL(d=1, a=2, h={knl_h}, channels="YUV")\n'
                'ConvertToYUV420()\n'),
        "smdegrain": (f'Import("{EXTOOLS}")\n' + src +
                      f'SMDegrain(tr=3, thSAD={thsad}, prefilter=2, '
                      'contrasharp=false)\n'),
    }
    avs_wall(src, "warm_index", 2)          # pre-warm .ffindex
    base = avs_wall(src, "srconly", n)      # source-only wall
    res = {"_source_wall_s": round(base, 2)}
    for name, sc in scripts.items():
        try:
            dt = avs_wall(sc, name.replace(" ", "_"), n)
        except RuntimeError as e:
            print(f"  [avs {name}] FAILED: {e}", flush=True)
            res[name] = None
            continue
        ms = max(dt - base, 1e-6) * 1000.0 / n
        res[name] = round(ms, 2)
        print(f"  [avs {name}] {ms:.1f} ms/f ({1000.0/ms:.1f} fps)",
              flush=True)
    return res


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    core.std.LoadPlugin(path=str(DLL))
    variants = {
        "cpu-fit":  dict(engine="cpu",    noise="fit"),
        "cpu-hold": dict(engine="cpu",    noise="hold"),
        "vk-fit":   dict(engine="vulkan", noise="fit"),
        "vk-hold":  dict(engine="vulkan", noise="hold"),
    }
    out = {"protocol": "serial in-order frame pulls, net of source cost, "
                       f"warmup {WARMUP} frames excluded",
           "knobs_sigma20": {"thSAD": 864, "knl_h": 6.0,
                             "bm3d_sigma": "[5,5,5]"}}

    # ---- 540p: Set8 tractor + sigma20 AWGN -> bench-track 420 ----
    print("== 540p (Set8 tractor, sigma20, YUV420P8) ==", flush=True)
    files = seq_frames("tractor")
    clean = load_clean(files)
    n540 = len(clean)
    noisy = make_noisy(clean, 20, seed=1020)
    pl540 = list(frames_planes(to_420(rgb_clip_from(noisy, raw_rgb48=True))))
    h540, w540 = clean[0].shape[:2]
    y4m540 = WORK / "spd_540p.y4m"
    write_y4m(y4m540, pl540, w540, h540)
    out["540p"] = {"n_frames": n540,
                   "vs": vs_lane(clip_from_planes(pl540, w540, h540),
                                 n540, variants),
                   "avs": avs_lane(y4m540, n540, (864, 6.0))}

    # ---- 1080p: real derf tractor y4m ----
    print("== 1080p (derf tractor y4m, C420) ==", flush=True)
    n1080 = 60
    pl1080, w1080, h1080 = read_y4m_420(DERF_Y4M, n1080)
    out["1080p"] = {"n_frames": n1080,
                    "vs": vs_lane(clip_from_planes(pl1080, w1080, h1080),
                                  n1080, variants),
                    "avs": avs_lane(DERF_Y4M, n1080, (864, 6.0))}

    # ---- 2160p: 2x2 tile of the 1080p planes, GALOSH VS lane only ----
    print("== 2160p (2x2 tile, GALOSH only) ==", flush=True)
    n4k = 30
    pl4k = [[np.tile(p, (2, 2)) for p in fr] for fr in pl1080[:n4k]]
    out["2160p"] = {"n_frames": n4k,
                    "vs": vs_lane(clip_from_planes(pl4k, w1080 * 2,
                                                   h1080 * 2), n4k, variants)}

    y4m540.unlink()
    outf = OUT / "_speed_set8.json"
    outf.write_text(json.dumps(out, indent=1))
    print("saved:", outf)


if __name__ == "__main__":
    main()
