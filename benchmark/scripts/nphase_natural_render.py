"""Natural sRGB render of o32 N_PHASES=16 vs 4 for human visual judgment.

N_PHASES only affects LUMA (overlapping-WHT cycle-spin); chroma path (LOESS) is
identical, so N16 vs N4 differ in luma detail only.  This renders a natural
demosaiced sRGB image so the luma difference can be judged in context.

Pipeline per patch:
  - Bayer pattern auto-detected by best correlation of demosaic(gt_raw) vs gt_srgb
  - Menon2007 demosaic -> linear RGB
  - per-channel WB/exposure gain fit from GT (demosaic_gt_raw -> linear gt_srgb)
  - sRGB encode
Montage per patch: [Noisy | GT | N16 | N4]; plus individual N16/N4 full-res.
Output -> benchmark/sidd_validation/_nphase_visual/natural/
"""
import os, sys, subprocess
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from skimage.io import imsave
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007 as demo

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
GALOSH = Path(r"C:\Users\luxgrain\GALOSH")
VAL = GALOSH / "benchmark" / "SIDD_Validation"
OUTD = GALOSH / "benchmark" / "sidd_validation" / "_nphase_visual" / "natural"
OUTD.mkdir(parents=True, exist_ok=True)
TMP = GALOSH / "benchmark" / "results" / "speed" / "_nphnat_tmp"
TMP.mkdir(parents=True, exist_ok=True)
EXE = GALOSH / "standalone" / "galosh_raw_gpu.exe"
PATS = ["RGGB", "BGGR", "GRBG", "GBRG"]


def run_o32(noisy, w, h, uid, stride):
    in_p = TMP / f"{uid}_in.bin"; out_p = TMP / f"{uid}_out.bin"
    noisy.astype(np.float32).tofile(str(in_p))
    env = dict(os.environ); env["GALOSH_O32_PHASE_STRIDE"] = str(stride)
    subprocess.run([str(EXE), str(in_p), str(out_p), str(w), str(h),
                    "1.0", "1.0", "1.0", "0", "0", "0", "--variant=o32"],
                   capture_output=True, timeout=120, env=env)
    den = np.fromfile(str(out_p), dtype=np.float32).reshape(h, w)
    in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True)
    return den


def srgb_dec(s):
    s = np.clip(s, 0, 1)
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def srgb_enc(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055)


def corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def up(img, k=2):
    return np.repeat(np.repeat(img, k, 0), k, 1)


def main():
    N_RAW = loadmat(str(VAL / "ValidationNoisyBlocksRaw.mat"))["ValidationNoisyBlocksRaw"]
    G_RAW = loadmat(str(VAL / "ValidationGtBlocksRaw.mat"))["ValidationGtBlocksRaw"]
    G_SR = loadmat(str(VAL / "ValidationGtBlocksSrgb.mat"))["ValidationGtBlocksSrgb"]
    ns, npp = N_RAW.shape[:2]

    # rank by gt luma std; take a spread of textured/mid (skip near-black flats)
    def lstd(s, p):
        g = np.asarray(G_RAW[s, p], np.float32)
        return float((0.25 * (g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2])).std())
    ranked = sorted(((lstd(s, p), s, p) for s in range(ns) for p in range(npp)), reverse=True)
    picks = [ranked[i] for i in (0, 3, 8, 20, 60, 150)]   # high -> mid texture

    for sd, s, p in picks:
        nr = np.asarray(N_RAW[s, p], np.float32); gr = np.asarray(G_RAW[s, p], np.float32)
        gs = np.asarray(G_SR[s, p], np.float32) / 255.0
        h, w = nr.shape
        gt_lin = srgb_dec(gs)
        # detect pattern by best demosaic(gt_raw) vs gt_srgb-linear correlation
        best, bc = "RGGB", -9
        for pat in PATS:
            dm = demo(gr, pat)
            c = corr(dm[..., 0], gt_lin[..., 0]) + corr(dm[..., 1], gt_lin[..., 1]) + corr(dm[..., 2], gt_lin[..., 2])
            if c > bc: bc, best = c, pat
        # WB/exposure gains from GT
        dm_gt = np.clip(demo(gr, best), 0, None)
        gain = np.array([gt_lin[..., k].mean() / max(dm_gt[..., k].mean(), 1e-6) for k in range(3)])

        def render(bayer):
            rgb = np.clip(demo(bayer, best), 0, None) * gain
            return (np.clip(srgb_enc(rgb), 0, 1) * 255).astype(np.uint8)

        n16 = run_o32(nr, w, h, f"{s}_{p}_16", 1)
        n4 = run_o32(nr, w, h, f"{s}_{p}_4", 2)
        imgs = {"noisy": render(nr), "gt": render(gr), "N16": render(n16), "N4": render(n4)}
        tag = f"s{s:02d}p{p:02d}_gtstd{sd:.3f}_{best}"
        montage = up(np.concatenate([imgs["noisy"], imgs["gt"], imgs["N16"], imgs["N4"]], axis=1), 2)
        imsave(str(OUTD / f"{tag}__noisy_gt_N16_N4.png"), montage, check_contrast=False)
        imsave(str(OUTD / f"{tag}__N16.png"), up(imgs["N16"], 2), check_contrast=False)
        imsave(str(OUTD / f"{tag}__N4.png"), up(imgs["N4"], 2), check_contrast=False)
        print(f"s{s:02d}p{p:02d} gtstd={sd:.3f} pattern={best} -> {tag}")
    print(f"\nmontage = [Noisy | GT | N16 | N4]  (natural sRGB, 2x).  saved -> {OUTD}")


if __name__ == "__main__":
    main()
