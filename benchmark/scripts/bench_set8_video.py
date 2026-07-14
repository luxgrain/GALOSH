#!/usr/bin/env python3
"""Set8 video benchmark for GALOSH-frameserver (2026-07-11).

Positioning (user-confirmed): PRACTICAL VALIDATION of the frameserver +
future Doom9 material — not a paper-claim campaign.

Protocol (FastDVDnet-comparable):
  - sources: Set8 official PNG sequences (derf 4 capped at 85 frames,
    GoPro 4 full length), 960x540 RGB
  - seeded AWGN sigma in {10,20,30,40,50} added in [0,255] RGB float,
    clipped; carried to the filter as RGB48 -> YUV444P16 (bt709 full)
  - GALOSH-frameserver variants: engine {cpu, vulkan} x noise {fit, hold}
    (one filter instance per sequence, frames in order -> real hold
    semantics); back to RGB24 for metrics
  - metrics: per-frame PSNR (RGB, vs clean) -> sequence mean, plus
    SSIM/LPIPS/DISTS via the SIDD bench stack; runtimes recorded
  - persistence: PNG sequences for noisy + cpu-fit + vulkan-hold
    (representative ends; other variants metrics-only), GT = dataset path

Modes:
  --mode 444 : YUV444P16 full-range (FastDVDnet-comparable, default)
  --mode 420 : real-video track — GT/noisy converted to YUV420P8
               bt709/limited/siting=left; the metric reference is GT-420
               reconstructed to RGB with the SAME upsampler as the output
               (measures denoising, not subsampling; ab_yuv420 convention).
               Adds per-plane Y/Cb/Cr PSNR on the 420 lattice.

Usage:
  python bench_set8_video.py [--mode 444|420] [--seqs tractor,...]
                             [--sigmas 10,20,30,40,50]
                             [--variants galosh-cpu-fit,...]
                             [--limit-frames N] [--no-png]
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bench_sidd_medium as smb   # LPIPS/DISTS/SSIM stack (GPU torch)

import vapoursynth as vs
core = vs.core

ROOT = Path(__file__).resolve().parents[2]
SET8 = Path("E:/img_dataset/Set8/test_sequences")
OUT = ROOT / "benchmark" / "results_set8_awgn"
DLL = Path(os.environ.get("GALOSH_FS_DLL",
      str(ROOT.parent / "GALOSH-frameserver" / "galosh_frameserver.dll")))

DERF = ["tractor", "touchdown", "park_joy", "sunflower"]
GOPRO = ["hypersmooth", "motorbike", "rafting", "snowboard"]
DERF_CAP = 85     # FastDVDnet convention

# [2026-07-14] unified naming with the pgnoise/CRVD tracks: galosh-<eng>-<noise>
VARIANTS = {
    "galosh-cpu-fit":  dict(engine="cpu",    noise="fit"),
    "galosh-cpu-hold": dict(engine="cpu",    noise="hold"),
    "galosh-vk-fit":   dict(engine="vulkan", noise="fit"),
    "galosh-vk-hold":  dict(engine="vulkan", noise="hold"),
}
# persist-everything rule: save ALL variants (was a 2-variant subset)
SAVE_PNG_FOR = set(VARIANTS)


def seq_frames(name):
    d = SET8 / ("gopro_540p" if name in GOPRO else "") / name
    fs = sorted(d.glob("*.png"))
    if name in DERF:
        fs = fs[:DERF_CAP]
    return fs


def load_clean(files):
    from PIL import Image
    return [np.asarray(Image.open(f).convert("RGB")) for f in files]


def make_noisy(clean, sigma, seed):
    rng = np.random.default_rng(seed)
    out = []
    for img in clean:
        n = img.astype(np.float64) + rng.normal(0.0, sigma, img.shape)
        out.append(np.clip(n, 0, 255))
    return out


def rgb_clip_from(noisy_f, raw_rgb48=False):
    """list of float RGB [0,255] -> RGB48 clip [-> YUV444P16 bt709 full]."""
    h, w = noisy_f[0].shape[:2]
    fmt = core.query_video_format(vs.RGB, vs.INTEGER, 16, 0, 0)
    c = core.std.BlankClip(width=w, height=h, format=fmt, length=len(noisy_f))
    arrs = [np.round(x * 257.0).astype(np.uint16) for x in noisy_f]

    def put(n, f, arrs=arrs):
        fo = f.copy()
        for p in range(3):
            np.asarray(fo[p])[:] = arrs[n][:, :, p]
        return fo
    c = core.std.ModifyFrame(c, c, put)
    if raw_rgb48:
        return c
    return core.resize.Bicubic(c, format=vs.YUV444P16,
                               matrix_s="709", range_s="full",
                               range_in_s="full")


def to_rgb24(clip):
    return core.resize.Bicubic(clip, format=vs.RGB24,
                               matrix_in_s="709", range_in_s="full",
                               range_s="full")


def to_420(clip_rgb48):
    """RGB48 -> YUV420P8 bt709 limited, left-sited (video convention)."""
    return core.resize.Bicubic(clip_rgb48, format=vs.YUV420P8,
                               matrix_s="709", range_s="limited",
                               range_in_s="full", chromaloc_s="left")


def from_420(clip420):
    """YUV420P8 -> RGB24; the SAME upsampler for GT / noisy / outputs."""
    return core.resize.Bicubic(clip420, format=vs.RGB24,
                               matrix_in_s="709", range_in_s="limited",
                               range_s="full", chromaloc_in_s="left")


def rgb48_clip_from_u8(frames_u8):
    return rgb_clip_from([f.astype(np.float64) for f in frames_u8],
                         raw_rgb48=True)


def frames_planes(clip):
    for f in clip.frames():
        yield [np.asarray(f[p]).copy() for p in range(3)]


def plane_psnr(a, b, peak):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(peak * peak / mse)


def frames_rgb(clip):
    for f in clip.frames():
        yield np.stack([np.asarray(f[p]) for p in range(3)], axis=-1)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def save_png(arr_u8, path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8).save(path)


def metrics5(out_u8, ref_u8):
    """the standard 5-metric set (feedback rule: always all five)."""
    o = out_u8.astype(np.float32) / 255.0
    r = ref_u8.astype(np.float32) / 255.0
    return dict(
        psnr=psnr(out_u8, ref_u8),
        ssim=smb.ssim_rgb(r.astype(np.float64), o.astype(np.float64)),
        lpips=smb.compute_lpips_patched(o, r),
        dists=smb.compute_dists_patched(o, r),
        niqe=smb.compute_niqe(o),
    )


def agg(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="444", choices=["444", "420"])
    ap.add_argument("--seqs", default=",".join(DERF + GOPRO))
    ap.add_argument("--sigmas", default="10,20,30,40,50")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    core.std.LoadPlugin(path=str(DLL))
    seqs = args.seqs.split(",")
    sigmas = [int(s) for s in args.sigmas.split(",")]
    variants = args.variants.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    mode = args.mode
    pngroot = OUT / ("png" if mode == "444" else "png420")

    results = {}
    for seq in seqs:
        files = seq_frames(seq)
        if args.limit_frames:
            files = files[:args.limit_frames]
        clean = load_clean(files)
        n = len(clean)
        results[seq] = {"n_frames": n, "src": str(files[0].parent)}

        if mode == "420":
            # GT-420 (per seq, sigma-independent): planes + SAME-upsampler
            # RGB reconstruction = the metric reference
            gt420_clip = to_420(rgb48_clip_from_u8(clean))
            gt_planes = list(frames_planes(gt420_clip))
            gt_ref = list(frames_rgb(from_420(gt420_clip)))
            if not args.no_png:
                for i, g in enumerate(gt_ref):
                    save_png(g, pngroot / seq / "gt_ref" / f"{i:04d}.png")

        for sigma in sigmas:
            noisy_f = make_noisy(clean, sigma, seed=1000 + sigma)
            ent = {}

            if mode == "444":
                noisy_u8 = [np.clip(np.round(x), 0, 255).astype(np.uint8)
                            for x in noisy_f]
                ent["noisy"] = agg([metrics5(nn, cc)
                                    for nn, cc in zip(noisy_u8, clean)])
                if not args.no_png:
                    for i, nn in enumerate(noisy_u8):
                        save_png(nn, pngroot / seq / f"s{sigma}" / "noisy"
                                 / f"{i:04d}.png")
                src_clip = rgb_clip_from(noisy_f)
                galosh_kw = dict(matrix="bt709", range="full", eotf="srgb")
                ref = clean
            else:
                src_clip = to_420(rgb_clip_from(noisy_f, raw_rgb48=True))
                noisy_rec = list(frames_rgb(from_420(src_clip)))
                noisy_planes = list(frames_planes(src_clip))
                ent["noisy"] = agg([metrics5(nn, rr)
                                    for nn, rr in zip(noisy_rec, gt_ref)])
                ent["noisy"]["y_psnr"] = float(np.mean(
                    [plane_psnr(a[0], b[0], 255) for a, b in
                     zip(noisy_planes, gt_planes)]))
                if not args.no_png:
                    for i, nn in enumerate(noisy_rec):
                        save_png(nn, pngroot / seq / f"s{sigma}" / "noisy"
                                 / f"{i:04d}.png")
                galosh_kw = dict(matrix="bt709", range="limited",
                                 eotf="srgb", siting="left")
                ref = gt_ref

            for var in variants:
                kw = dict(galosh_kw); kw.update(VARIANTS[var])
                den = core.galosh.Denoise(src_clip, luma=1.0, chroma=1.0, **kw)
                rgb = to_rgb24(den) if mode == "444" else from_420(den)
                t0 = time.perf_counter()
                rows, yps, cbps, crps = [], [], [], []
                for i in range(n):
                    f_rgb = rgb.get_frame(i)
                    out = np.stack([np.asarray(f_rgb[p]) for p in range(3)],
                                   axis=-1)
                    rows.append(metrics5(out, ref[i]))
                    if mode == "420":
                        f_den = den.get_frame(i)   # cache-warm from rgb pull
                        pl = [np.asarray(f_den[p]) for p in range(3)]
                        yps.append(plane_psnr(pl[0], gt_planes[i][0], 255))
                        cbps.append(plane_psnr(pl[1], gt_planes[i][1], 255))
                        crps.append(plane_psnr(pl[2], gt_planes[i][2], 255))
                    if not args.no_png and var in SAVE_PNG_FOR:
                        save_png(out, pngroot / seq / f"s{sigma}" / var
                                 / f"{i:04d}.png")
                dt = time.perf_counter() - t0
                ent[var] = agg(rows)
                ent[var]["wall_s_per_frame_incl_metrics"] = dt / n
                if mode == "420":
                    ent[var]["y_psnr"] = float(np.mean(yps))
                    ent[var]["cb_psnr"] = float(np.mean(cbps))
                    ent[var]["cr_psnr"] = float(np.mean(crps))
                print(f"[{mode} {seq} s{sigma} {var}] "
                      f"PSNR {ent[var]['psnr']:.2f} "
                      f"SSIM {ent[var]['ssim']:.4f} "
                      f"LPIPS {ent[var]['lpips']:.4f}", flush=True)
            results[seq][f"s{sigma}"] = ent
            outf = OUT / (f"_metrics_{mode}"
                          f"{('_' + args.tag) if args.tag else ''}.json")
            outf.write_text(json.dumps(results, indent=1))
    print("saved:", outf)


if __name__ == "__main__":
    main()
