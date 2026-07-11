"""ab_yuv420.py — design A/B for GALOSH-420: native half-res chroma (A) vs
upsample-to-444-first (B).

Question (2026-07-11 design session): for 4:2:0 input, should chroma be
denoised at its NATIVE half-res lattice with a downsampled-Y guide (A), or
upsampled to 4:4:4 first and pushed through the existing GALOSH-YUV
pipeline (B)?  Theory says A (denoise before interpolation; no noise
correlation, no noisy-guide injection, 1/4 chroma cost) — this rig measures
it with ZERO new C code, composing both options from the existing exe:

  B: 420 -> phase-correct bilinear 444 -> galosh_yuv_cpu (full-res run)
     -> back to 420 by the reference box filter.
  A: chroma = a HALF-RES image (box-down Y, native Cb/Cr) through the SAME
     exe = half-res LOESS with a box-downsampled noisy-Y guide (the YUV
     pipeline's own guide convention).  Y = taken from the full-res run
     (the Y path is chroma-independent, so A and B share it by design).

Test material: SIDD Medium sRGB pairs -> gamma-domain BT.709 full-range
YCbCr, chroma 2x2 box to 420 (center siting / JPEG convention; the GT
chroma reference uses the same box, so the subsampling itself costs both
options nothing).

Rig approximations (documented, fine for a design decision):
  - The half-res run re-fits the noise model on the box-downsampled Y
    (sigma halves); a production A would inherit the full-res fit.
  - Chroma guide is the NOISY stabilized Y at each scale (existing YUV
    convention) — the denoised-guide variant is a separate ablation.

Metrics: decision metric = Cb/Cr PSNR on the 420 lattice vs GT-420.
Context: Y-PSNR (shared), and full perceptual set (PSNR/SSIM/LPIPS/DISTS/
NIQE) on 444-reconstructed sRGB using the SAME bilinear upsampler for A
and B (so reconstruction cost cancels).  Noisy baseline: PSNR only.

Usage:  python ab_yuv420.py [--limit N]
Env:    GALOSH_SIDD_BENCH (dataset dir with *_noisy_srgb.npy/*_gt_srgb.npy
        — falls back to the raw-pair dir layout used by bench_raw_campaign)
Output: benchmark/results_yuv420_ab/_ab_metrics.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GALOSH = HERE.parent.parent
EXE = GALOSH / "standalone" / "galosh_yuv_cpu.exe"
OUTROOT = GALOSH / "benchmark" / "results_yuv420_ab"
sys.path.insert(0, str(HERE))
import bench_sidd_medium as smb  # LPIPS/DISTS/NIQE metric helpers

# ---------- full-range, gamma-domain YCbCr (H.273 matrix coefficients) ----
# Kr/Kb per matrix; Kg = 1-Kr-Kb; Cb=(B'-Y')/(2(1-Kb)); Cr=(R'-Y')/(2(1-Kr)).
MATRICES = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114)}
KR, KB = MATRICES["bt709"]   # set in main() from --matrix

def rgb2ycc(rgb):
    kg = 1.0 - KR - KB
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = KR * r + kg * g + KB * b
    cb = (b - y) / (2.0 * (1.0 - KB))
    cr = (r - y) / (2.0 * (1.0 - KR))
    return y, cb, cr

def ycc2rgb(y, cb, cr):
    kg = 1.0 - KR - KB
    r = y + 2.0 * (1.0 - KR) * cr
    b = y + 2.0 * (1.0 - KB) * cb
    g = (y - KR * r - KB * b) / kg
    return np.stack([r, g, b], axis=-1)

def box2(p):
    """444 -> 420 chroma generation used by this rig.

    Siting terminology (4 layers, kept separate on purpose):
      1. chroma sample-center position: luma pixel-corner coords, luma
         pixel k covers [k,k+1]; this filter's support is [2j, 2j+2] so
         the produced sample CENTER sits at 2j+1 -> (1.0, 1.0) for j=0 =
         CENTER-sited 420 (JPEG/MPEG-1 convention).
      2. equivalent 2x2 chroma-cell top-left: (0.0, 0.0).
      3. phase offset vs center-sited: (0, 0) by construction.
      4. filter kernel: 2x2 box — an IMPLEMENTATION CHOICE for this rig
         (siting specifies position/phase only, never the kernel)."""
    h, w = p.shape
    return p.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

def up2_center(p):
    """420 -> 444 reconstruction for CENTER-sited samples, phase-exact
    bilinear (kernel choice of this rig; siting only fixes the phase).
    Corner coords: full-res target = luma centers k+0.5; chroma sample
    centers at 2j+1.  Even k=2m: own sample d=0.5, left d=1.5 -> weights
    .75/.25; odd k=2m+1: own d=0.5, right d=1.5 -> .75/.25.  Verified
    phase-exact (reproduces a linear ramp with zero interior error)."""
    def up1(a, axis):
        a = np.moveaxis(a, axis, 0)
        left = np.concatenate([a[:1], a[:-1]], axis=0)     # clamp
        right = np.concatenate([a[1:], a[-1:]], axis=0)
        even = 0.75 * a + 0.25 * left
        odd = 0.75 * a + 0.25 * right
        out = np.empty((a.shape[0] * 2,) + a.shape[1:], a.dtype)
        out[0::2] = even
        out[1::2] = odd
        return np.moveaxis(out, 0, axis)
    return up1(up1(p, 0), 1)

def down420(p, siting):
    """444 -> 420 generation.  center: 2x2 box (sample centers (1,1)).
    left: horizontal 3-tap tent [.25,.5,.25] sampled at even column
    centers (x=2j+0.5, horizontally co-sited) + vertical 2-tap box
    (y=2j+1, midway) -> sample centers (0.5, 1.0).  Kernels are rig
    choices; siting fixes only the phase."""
    if siting == "center":
        return box2(p)
    lp = np.pad(p, ((0, 0), (1, 1)), mode="edge")
    horiz = 0.25 * lp[:, :-2] + 0.5 * lp[:, 1:-1] + 0.25 * lp[:, 2:]
    if siting == "left":
        hs = horiz[:, 0::2]
        return 0.5 * (hs[0::2, :] + hs[1::2, :])
    # topleft: 3-tap tent in BOTH axes, sampled at even rows/cols
    vp = np.pad(horiz, ((1, 1), (0, 0)), mode="edge")
    vert = 0.25 * vp[:-2, :] + 0.5 * vp[1:-1, :] + 0.25 * vp[2:, :]
    return vert[0::2, 0::2]

def up420(p, siting):
    """420 -> 444 phase-exact bilinear per siting.  left: horizontal
    co-sited (even col weight 1.0; odd col 0.5/0.5), vertical 0.75/0.25."""
    if siting == "center":
        return up2_center(p)
    def upH(a):   # horizontally co-sited: even col exact, odd col midway
        right = np.concatenate([a[:, 1:], a[:, -1:]], axis=1)
        out = np.empty((a.shape[0], a.shape[1] * 2), a.dtype)
        out[:, 0::2] = a
        out[:, 1::2] = 0.5 * (a + right)
        return out
    def upV_mid(a):   # vertically midway: 0.75/0.25 phase
        up = np.concatenate([a[:1], a[:-1]], axis=0)
        dn = np.concatenate([a[1:], a[-1:]], axis=0)
        out = np.empty((a.shape[0] * 2, a.shape[1]), a.dtype)
        out[0::2] = 0.75 * a + 0.25 * up
        out[1::2] = 0.75 * a + 0.25 * dn
        return out
    def upV_cosited(a):   # vertically co-sited: even row exact, odd midway
        dn = np.concatenate([a[1:], a[-1:]], axis=0)
        out = np.empty((a.shape[0] * 2, a.shape[1]), a.dtype)
        out[0::2] = a
        out[1::2] = 0.5 * (a + dn)
        return out
    return (upV_mid if siting == "left" else upV_cosited)(upH(p))

SITING = "center"   # set in main() from --siting

def run_exe(rgb, uid):
    h, w = rgb.shape[:2]
    ip = OUTROOT / f"_tmp_{uid}_in.bin"
    op = OUTROOT / f"_tmp_{uid}_out.bin"
    rgb.astype(np.float32).tofile(ip)
    t0 = time.time()
    r = subprocess.run([str(EXE), str(ip), str(op), str(w), str(h), "1.0", "1.0"],
                       capture_output=True, timeout=900)
    dt = time.time() - t0
    ip.unlink(missing_ok=True)
    if r.returncode != 0 or not op.exists():
        raise RuntimeError(r.stderr.decode("utf-8", "replace")[-300:])
    out = np.fromfile(op, dtype=np.float32).reshape(h, w, 3)
    op.unlink(missing_ok=True)
    return out, dt

def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else float(10 * np.log10(1.0 / mse))

def load_sidd():
    base = Path(os.environ.get("GALOSH_SIDD_BENCH", "benchmark/datasets/sidd_medium_bench"))
    items = []
    for nf in sorted(base.glob("*_noisy_srgb.npy")):
        stem = nf.name.replace("_noisy_srgb.npy", "")
        gf = base / f"{stem}_gt_srgb.npy"
        if gf.exists():
            items.append((stem, nf, gf))
    return items

def main():
    global KR, KB, SITING
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--matrix", default="bt709", choices=sorted(MATRICES))
    ap.add_argument("--siting", default="center", choices=["center", "left", "topleft"])
    ap.add_argument("--guide-phase", default="match", choices=["match", "center"],
                    help="A-arm guide construction: 'match' = resample Y at the "
                         "chroma siting phase (correct); 'center' = naive box "
                         "regardless of siting (deliberate 0.5px-mismatch probe)")
    args = ap.parse_args()
    KR, KB = MATRICES[args.matrix]
    SITING = args.siting
    OUTROOT.mkdir(parents=True, exist_ok=True)
    suffix = ("" if args.matrix == "bt709" else f"_{args.matrix}") + \
             ("" if args.siting == "center" else f"_{args.siting}") + \
             ("" if args.guide_phase == "match" else "_gmis")
    res_path = OUTROOT / f"_ab_metrics{suffix}.json"
    metrics = json.load(open(res_path)) if res_path.exists() else {}

    items = load_sidd()
    if args.limit:
        items = items[: args.limit]
    print(f"[ab_yuv420] {len(items)} scenes", flush=True)

    t_start = time.time()
    for ii, (stem, nf, gf) in enumerate(items):
        if stem in metrics:
            continue
        noisy = np.load(nf).astype(np.float32)
        gt = np.load(gf).astype(np.float32)
        # even-crop for the 2x2 box
        H, W = (noisy.shape[0] // 2) * 2, (noisy.shape[1] // 2) * 2
        noisy, gt = noisy[:H, :W], gt[:H, :W]

        # ---- construct gamma-domain 420 input + GT-420 reference ----
        Yn, Cbn, Crn = rgb2ycc(noisy)
        Cbn_h, Crn_h = down420(Cbn, SITING), down420(Crn, SITING)
        Yg, Cbg, Crg = rgb2ycc(gt)
        Cbg_h, Crg_h = down420(Cbg, SITING), down420(Crg, SITING)

        # ---- B: 444-first (full-res run; also donates the shared Y) ----
        rgbB_in = np.clip(ycc2rgb(Yn, up420(Cbn_h, SITING), up420(Crn_h, SITING)), 0, 1)
        rgbB_out, dtB = run_exe(rgbB_in, f"{stem}_B")
        YB, CbB, CrB = rgb2ycc(rgbB_out)
        CbB_h, CrB_h = down420(CbB, SITING), down420(CrB, SITING)

        # ---- A: native half-res chroma run ----
        guide_siting = SITING if args.guide_phase == "match" else "center"
        rgbA_in_h = np.clip(ycc2rgb(down420(Yn, guide_siting), Cbn_h, Crn_h), 0, 1)
        rgbA_out_h, dtA_h = run_exe(rgbA_in_h, f"{stem}_A")
        _, CbA_h, CrA_h = rgb2ycc(rgbA_out_h)

        rec = {
            "y_psnr": psnr(YB, Yg),                       # shared by A and B
            "noisy": {"cb": psnr(Cbn_h, Cbg_h), "cr": psnr(Crn_h, Crg_h)},
            "A": {"cb": psnr(CbA_h, Cbg_h), "cr": psnr(CrA_h, Crg_h),
                  "dt_chroma": dtA_h},
            "B": {"cb": psnr(CbB_h, Cbg_h), "cr": psnr(CrB_h, Crg_h),
                  "dt_full": dtB},
        }

        # ---- 444-reconstructed sRGB perceptual metrics (same upsampler) ----
        for tag, (cbh, crh) in (("A", (CbA_h, CrA_h)), ("B", (CbB_h, CrB_h))):
            rgb = np.clip(ycc2rgb(YB, up420(cbh, SITING), up420(crh, SITING)), 0, 1)
            rec[tag]["srgb_psnr"] = psnr(rgb, gt)
            try:
                rec[tag]["lpips"] = smb.compute_lpips_patched(rgb, gt)
                rec[tag]["dists"] = smb.compute_dists_patched(rgb, gt)
                rec[tag]["niqe"] = smb.compute_niqe(rgb)
            except Exception as e:
                sys.stderr.write(f"  {stem} {tag} perceptual fail: {str(e)[:120]}\n")

        metrics[stem] = rec
        json.dump(metrics, open(res_path, "w"), indent=1)
        el = (time.time() - t_start) / 60
        print(f"  [{ii+1}/{len(items)}] {stem} {el:.1f}min | "
              f"Cb A {rec['A']['cb']:.2f} vs B {rec['B']['cb']:.2f} | "
              f"Cr A {rec['A']['cr']:.2f} vs B {rec['B']['cr']:.2f}", flush=True)

    json.dump(metrics, open(res_path, "w"), indent=1)
    print(f"DONE in {(time.time()-t_start)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
