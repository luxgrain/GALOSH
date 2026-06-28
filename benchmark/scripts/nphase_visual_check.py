"""Visual blocking check for o32 N_PHASES=4 vs 16 (cycle-spin reduction).

Reduced cycle-spinning can introduce 8x8 BLOCKING (a perceptual artifact PSNR
understates).  To see it cleanly we 2x2-box-bin the Bayer output -> CFA-free
luma, contrast-stretch, and render N16 | N4 | amplified diff(N16-N4).  If
N_PHASES=4 blocks, the diff shows an 8x8 grid; if not, it's smooth low-noise.

Picks the smoothest gt patches (blocking shows most on flat regions) + a couple
textured controls.  Outputs montage PNGs -> benchmark/sidd_validation/_nphase_visual/.
"""
import os, sys, subprocess
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from skimage.io import imsave

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
GALOSH = Path(os.path.expanduser(r"~\GALOSH"))
VAL = GALOSH / "benchmark" / "SIDD_Validation"
OUTD = GALOSH / "benchmark" / "sidd_validation" / "_nphase_visual"
OUTD.mkdir(parents=True, exist_ok=True)
TMP = GALOSH / "benchmark" / "results" / "speed" / "_nphase_tmp"
TMP.mkdir(parents=True, exist_ok=True)
EXE = GALOSH / "standalone" / "galosh_raw_gpu.exe"


def run_o32(noisy, w, h, uid, stride):
    in_p = TMP / f"{uid}_in.bin"; out_p = TMP / f"{uid}_out.bin"
    noisy.astype(np.float32).tofile(str(in_p))
    env = dict(os.environ); env["GALOSH_O32_PHASE_STRIDE"] = str(stride)
    cmd = [str(EXE), str(in_p), str(out_p), str(w), str(h), "1.0", "1.0", "1.0", "0", "0", "0", "--variant=o32"]
    subprocess.run(cmd, capture_output=True, timeout=120, env=env)
    den = np.fromfile(str(out_p), dtype=np.float32).reshape(h, w)
    in_p.unlink(missing_ok=True); out_p.unlink(missing_ok=True)
    return den


def luma(bayer):
    """2x2 box bin -> CFA-free half-res luma."""
    return 0.25 * (bayer[0::2, 0::2] + bayer[0::2, 1::2] + bayer[1::2, 0::2] + bayer[1::2, 1::2])


def stretch(img, lo=2, hi=98):
    a, b = np.percentile(img, [lo, hi])
    return np.clip((img - a) / max(b - a, 1e-6), 0, 1)


def up(img, k=3):
    return np.kron(img, np.ones((k, k)))   # nearest upscale, keeps grid sharp


def main():
    N_RAW = loadmat(str(VAL / "ValidationNoisyBlocksRaw.mat"))["ValidationNoisyBlocksRaw"]
    G_RAW = loadmat(str(VAL / "ValidationGtBlocksRaw.mat"))["ValidationGtBlocksRaw"]
    ns, npp = N_RAW.shape[:2]
    # rank patches by gt luma std (texture proxy)
    stds = []
    for s in range(ns):
        for p in range(npp):
            stds.append((float(luma(np.asarray(G_RAW[s, p], np.float32)).std()), s, p))
    stds.sort()
    picks = stds[:4] + stds[len(stds)//2:len(stds)//2+1] + stds[-1:]   # 4 smoothest + 1 mid + 1 textured
    labels = ["smooth", "smooth", "smooth", "smooth", "mid", "textured"]

    for (sd, s, p), lab in zip(picks, labels):
        nr = np.asarray(N_RAW[s, p], np.float32); gr = np.asarray(G_RAW[s, p], np.float32)
        h, w = nr.shape
        n16 = run_o32(nr, w, h, f"{s}_{p}_16", 1)
        n4  = run_o32(nr, w, h, f"{s}_{p}_4", 2)
        Ln, Lg = luma(nr), luma(gr)
        L16, L4 = luma(n16), luma(n4)
        diff = L16 - L4
        # amplified diff centered at 0.5 grey (reveals 8x8 grid if blocking)
        amp = 30.0
        dvis = np.clip(0.5 + diff * amp, 0, 1)
        # common stretch for the 4 luma panels (same window for fair compare)
        a, b = np.percentile(Lg, [2, 98]); win = lambda x: np.clip((x - a) / max(b - a, 1e-6), 0, 1)
        panels = [win(Ln), win(Lg), win(L16), win(L4), dvis]
        row = np.concatenate([up(x, 3) for x in panels], axis=1)
        diff_rms = float(np.sqrt(np.mean(diff**2))); diff_max = float(np.abs(diff).max())
        name = f"{lab}_s{s:02d}p{p:02d}_gtstd{sd:.3f}_diffRMS{diff_rms:.4f}.png"
        imsave(str(OUTD / name), (np.clip(row, 0, 1) * 255).astype(np.uint8), check_contrast=False)
        print(f"{lab:9s} s{s:02d}p{p:02d} gtstd={sd:.4f}  diff(N16-N4) RMS={diff_rms:.5f} max={diff_max:.4f}  -> {name}")
    print(f"\npanels per row: [Noisy | GT | N16 | N4 | diff(N16-N4)x{int(30)}+0.5]   (luma, 3x upscaled)")
    print(f"saved -> {OUTD}")


if __name__ == "__main__":
    main()
