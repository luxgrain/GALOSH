"""
BM3D-CFA (Pre-demosaic Bayer-domain BM3D).

EN: Approximates the Danielyan 2009 "BM3D-CFA" by packing a RGGB Bayer image
    into 4 half-resolution planes (R, G1, G2, B) and running the standard
    grayscale BM3D on each. This is the common "BM3D at CFA" baseline used
    across the denoising literature and is ~2-3 dB stronger than per-channel
    BM3D on the demosaiced image for real sensor noise, because Bayer-domain
    denoising avoids amplifying noise through the demosaic step.

JP: Danielyan 2009 の BM3D-CFA を RGGB Bayer を 4 ハーフ解像度プレーン
    (R, G1, G2, B) にパックして各プレーンで BM3D グレースケールを走らせる形で近似。
    真の BM3D-CFA (Bayer grid 上での joint block matching) ではないが、
    文献で広く使われる "CFA BM3D" ベースライン相当。
    Demosaic 後にかけるより 2-3 dB 強い（ノイズがデモザイクで拡散しないため）。

Signature:
    run_bm3d_cfa(noisy_bayer, sigma=None) -> (denoised_bayer, elapsed_s)
      noisy_bayer: (H, W) float32 RGGB pattern, H and W even
      sigma: noise std scalar; if None, per-plane MAD estimation
"""
import time
import threading
import numpy as np

import bm3d
from skimage.restoration import estimate_sigma

# bm3d is not thread-safe; share across CBM3D too by importing the same lock
_bm3d_lock = threading.Lock()


def _estimate_sigma_plane(plane: np.ndarray) -> float:
    """Per-plane sigma via skimage robust wavelet-based estimator."""
    try:
        s = float(estimate_sigma(plane, channel_axis=None))
        return max(s, 1e-4)
    except Exception:
        # Fallback: MAD on Laplacian high-pass
        hp = plane[1:-1, 1:-1] - 0.25 * (plane[:-2, 1:-1] + plane[2:, 1:-1]
                                         + plane[1:-1, :-2] + plane[1:-1, 2:])
        return float(np.median(np.abs(hp)) / 0.6745 / 2.0) or 1e-4


def run_bm3d_cfa(noisy_bayer: np.ndarray, sigma=None):
    """Run BM3D on packed RGGB Bayer planes.

    Args:
        noisy_bayer: (H, W) float32 Bayer image in RGGB pattern, [0, 1].
        sigma: noise std. Accepts:
            - None: per-plane wavelet MAD estimation (= "blind").
            - scalar float: same value applied to all 4 sub-channels.
            - list/array of 4 floats: per-channel oracle sigma (= "oracle"
              when computed from true noise = std(noisy - gt) per sub-channel).

    Returns:
        denoised_bayer: (H, W) float32 same shape/pattern.
        elapsed_s: wall-clock time including sigma estimation and all 4 BM3D
                   calls.
    """
    assert noisy_bayer.ndim == 2, "Expected 2D Bayer image"
    H, W = noisy_bayer.shape
    assert H % 2 == 0 and W % 2 == 0, f"Expected even dims, got {H}x{W}"

    t0 = time.time()

    # Unpack RGGB -> 4 half-resolution planes (H/2, W/2)
    planes = [
        noisy_bayer[0::2, 0::2],   # R
        noisy_bayer[0::2, 1::2],   # G1
        noisy_bayer[1::2, 0::2],   # G2
        noisy_bayer[1::2, 1::2],   # B
    ]

    # Resolve per-plane sigma list
    if sigma is None:
        sigmas = [None, None, None, None]
    elif np.isscalar(sigma):
        sigmas = [float(sigma)] * 4
    else:
        sigma_arr = np.asarray(sigma).flatten()
        assert len(sigma_arr) == 4, f"sigma list must have 4 entries, got {len(sigma_arr)}"
        sigmas = [float(s) for s in sigma_arr]

    den_planes = []
    with _bm3d_lock:
        for p, s in zip(planes, sigmas):
            if s is None:
                s = _estimate_sigma_plane(p)
            # bm3d returns float64; cast back to float32 for memory
            d = bm3d.bm3d(p.astype(np.float32),
                          sigma_psd=s,
                          stage_arg=bm3d.BM3DStages.ALL_STAGES)
            den_planes.append(d.astype(np.float32))

    # Repack planes back to full-resolution Bayer
    den = np.empty_like(noisy_bayer, dtype=np.float32)
    den[0::2, 0::2] = den_planes[0]
    den[0::2, 1::2] = den_planes[1]
    den[1::2, 0::2] = den_planes[2]
    den[1::2, 1::2] = den_planes[3]

    dt = time.time() - t0
    return den, dt


if __name__ == "__main__":
    # Smoke test on SIDD Medium sample
    from pathlib import Path
    BASE = Path(__file__).resolve().parents[3]   # self-test only; repo root
    DBENCH = BASE / "datasets" / "sidd" / "medium_bench"
    tag = "0042_IP_RGGB_010"

    noisy = np.load(str(DBENCH / f"{tag}_noisy_raw.npy"))
    gt    = np.load(str(DBENCH / f"{tag}_gt_raw.npy"))
    print(f"Full Bayer shape: {noisy.shape}, dtype={noisy.dtype}, range=[{noisy.min():.3f},{noisy.max():.3f}]")

    # 512x512 center crop (must be even-even aligned to preserve RGGB)
    H, W = noisy.shape
    y0 = (H // 2 - 256) & ~1
    x0 = (W // 2 - 256) & ~1
    nc = noisy[y0:y0+512, x0:x0+512].astype(np.float32)
    gc = gt[y0:y0+512, x0:x0+512].astype(np.float32)

    den, dt = run_bm3d_cfa(nc, sigma=None)
    mse = float(np.mean((den - gc) ** 2))
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
    print(f"BM3D-CFA: 512x512 elapsed={dt:.2f}s, PSNR(raw)={psnr:.2f} dB")
