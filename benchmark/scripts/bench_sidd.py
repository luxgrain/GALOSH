#!/usr/bin/env python3
"""
SIDD Real RAW benchmark — two-group comparison with per-scene WB×CCM calibration.

Dataset:
  ValidationNoisyBlocksRaw.mat   (40, 32, 256, 256) float32 [0,1]
  ValidationGtBlocksRaw.mat      (40, 32, 256, 256) float32 [0,1]
  ValidationNoisyBlocksSrgb.mat  (40, 32, 256, 256, 3) uint8
  ValidationGtBlocksSrgb.mat     (40, 32, 256, 256, 3) uint8
  40 scenes × 32 patches = 1280 real smartphone RAW/sRGB pairs.

Two evaluation groups:

  Pre-demosaic:
    RAW_noisy → [DENOISE] → Menon → per-scene affine WB×CCM → sRGB gamma
    Methods: GALOSH GPU, BM3D-CFA, NLM-CFA
    GT②  = GT_RAW → same calibrated pipeline  (pure denoise quality)
    GT①  = ISP sRGB                           (cross-group comparison vs NAFNet)

  After-demosaic:
    ISP_sRGB_noisy → [DENOISE] → eval
    Methods: GALOSH-YUV GPU, NAFNet, DnCNN-B, DRUNet, BM3D, NLM
    GT   = ISP sRGB

  Per-scene affine (3×4 matrix) estimated via least-squares from paired
  Menon(GT_RAW) ↔ inv_gamma(GT_sRGB), approximating camera ISP WB+CCM+tone.
  Speed measured on [DENOISE] block only.

Usage:
  python bench_sidd.py                    # default methods (both groups)
  python bench_sidd.py -m all             # every method
  python bench_sidd.py -m pre             # pre-demosaic only
  python bench_sidd.py -m post            # after-demosaic only
  python bench_sidd.py -m galosh_gpu nafnet  # pick specific
  python bench_sidd.py --no-perceptual    # skip LPIPS/DISTS
  python bench_sidd.py --scenes 5         # first N scenes (quick test)
"""
import numpy as np
import subprocess
import os
import sys
import json
import time
import argparse
import threading
import concurrent.futures
from pathlib import Path
from skimage.io import imsave
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import uniform_filter

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

BASE     = Path(__file__).parent.parent
SIDD     = BASE / "datasets" / "sidd"
OUTDIR   = BASE / "comparison_images_sidd"
RESULTS  = BASE / "results"
RESULTS.mkdir(exist_ok=True)
OUTDIR.mkdir(exist_ok=True)
KAIR_DIR    = Path(r"C:\Users\luxgrain\GALOSH\benchmark\external\KAIR")
NAFNET_DIR  = Path(r"C:\Users\luxgrain\GALOSH\benchmark\external\NAFNet")
BASH_EXE    = Path(r"C:\msys64\usr\bin\bash.exe")

# bm3d and skimage NLM use internal threading — must not be called concurrently
_bm3d_lock = threading.Lock()
_nlm_lock  = threading.Lock()

# SIDD mat file keys and dimensions
# shape: (40 scenes, 32 patches, 256, 256) float32 [0, 1]
SIDD_NOISY_MAT      = SIDD / "ValidationNoisyBlocksRaw.mat"
SIDD_GT_MAT         = SIDD / "ValidationGtBlocksRaw.mat"
SIDD_NOISY_SRGB_MAT = SIDD / "ValidationNoisyBlocksSrgb.mat"
SIDD_GT_SRGB_MAT    = SIDD / "ValidationGtBlocksSrgb.mat"
SIDD_N_SCENES  = 40
SIDD_N_PATCHES = 32   # patches per scene

# ---------------------------------------------------------------------------
# Shared pipeline — identical for ALL methods
# NOTE: changing these functions affects every method equally.
# ---------------------------------------------------------------------------

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Linear light → sRGB gamma (IEC 61966-2-1)."""
    return np.where(x <= 0.0031308,
                    12.92 * x,
                    1.055 * np.power(np.maximum(x, 0.0), 1.0 / 2.4) - 0.055)

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB gamma → linear light (IEC 61966-2-1 inverse)."""
    return np.where(x <= 0.04045,
                    x / 12.92,
                    np.power((np.maximum(x, 0.0) + 0.055) / 1.055, 2.4))

def demosaic_bilinear(bayer: np.ndarray) -> np.ndarray:
    """Bilinear demosaic for RGGB Bayer (fallback)."""
    h, w = bayer.shape
    rgb  = np.zeros((h, w, 3), dtype=np.float32)
    rgb[0::2, 0::2, 0] = bayer[0::2, 0::2]   # R
    rgb[0::2, 1::2, 1] = bayer[0::2, 1::2]   # Gr
    rgb[1::2, 0::2, 1] = bayer[1::2, 0::2]   # Gb
    rgb[1::2, 1::2, 2] = bayer[1::2, 1::2]   # B
    mask_r = np.zeros((h, w), np.float32); mask_r[0::2, 0::2] = 1
    mask_g = np.zeros((h, w), np.float32); mask_g[0::2, 1::2] = 1; mask_g[1::2, 0::2] = 1
    mask_b = np.zeros((h, w), np.float32); mask_b[1::2, 1::2] = 1
    for c, mask in enumerate([mask_r, mask_g, mask_b]):
        num = uniform_filter(rgb[:, :, c], size=3)
        den = uniform_filter(mask, size=3)
        rgb[:, :, c] = np.where(mask > 0, rgb[:, :, c],
                                 num / np.maximum(den, 1e-10))
    return np.clip(rgb, 0.0, 1.0)

def detect_bayer_pattern(bayer: np.ndarray, srgb_gt: np.ndarray = None) -> str:
    """Auto-detect Bayer pattern from sub-channel statistics.

    Step 1: Green channels must have similar means.
      - If (0,0)≈(1,1): green on main diagonal → GRBG or GBRG
      - If (0,1)≈(1,0): green on anti-diagonal → RGGB or BGGR
    Step 2: Disambiguate R/B using sRGB GT channel ratio (if available),
      otherwise default to RGGB/GRBG.
    """
    s00 = bayer[0::2, 0::2].mean()
    s01 = bayer[0::2, 1::2].mean()
    s10 = bayer[1::2, 0::2].mean()
    s11 = bayer[1::2, 1::2].mean()

    diag_main = abs(s00 - s11)   # (0,0) vs (1,1)
    diag_anti = abs(s01 - s10)   # (0,1) vs (1,0)

    if diag_main < diag_anti:
        # Green at (0,0) and (1,1) → GRBG or GBRG
        # (0,1) and (1,0) are R and B — need to figure out which
        if srgb_gt is not None:
            r_mean = srgb_gt[:, :, 0].mean()
            b_mean = srgb_gt[:, :, 2].mean()
            if (r_mean >= b_mean) == (s01 >= s10):
                return 'GRBG'   # (0,1)=R, (1,0)=B
            else:
                return 'GBRG'   # (0,1)=B, (1,0)=R
        return 'GRBG'  # default
    else:
        # Green at (0,1) and (1,0) → RGGB or BGGR
        # (0,0) and (1,1) are R and B
        if srgb_gt is not None:
            r_mean = srgb_gt[:, :, 0].mean()
            b_mean = srgb_gt[:, :, 2].mean()
            if (r_mean >= b_mean) == (s00 >= s11):
                return 'RGGB'   # (0,0)=R, (1,1)=B
            else:
                return 'BGGR'   # (0,0)=B, (1,1)=R
        return 'RGGB'  # default

def crop_to_rggb(arr: np.ndarray, pattern: str) -> np.ndarray:
    """Crop array so Bayer grid starts at R position → RGGB pattern.

    Shifts the 2×2 grid origin to align R at top-left by cropping
    the leading row/column. Spatial relationships between pixels are
    fully preserved — no pixel rearrangement.

    Works for 2D (H,W) Bayer and 3D (H,W,C) sRGB arrays.
    Returns array with even dimensions.

    Offsets: RGGB(0,0) GRBG(0,1) BGGR(1,1) GBRG(1,0)
    """
    offsets = {'RGGB': (0, 0), 'GRBG': (0, 1), 'BGGR': (1, 1), 'GBRG': (1, 0)}
    ry, rx = offsets[pattern]
    cropped = arr[ry:, rx:]
    h, w = cropped.shape[:2]
    return cropped[:h - h % 2, :w - w % 2]

def demosaic_menon(bayer: np.ndarray) -> np.ndarray:
    """Menon (2007) DDFAPD demosaic — noise-robust, edge-adaptive.
    Input must be RGGB (use crop_to_rggb at load time)."""
    from colour_demosaicing import demosaicing_CFA_Bayer_DDFAPD
    rgb = demosaicing_CFA_Bayer_DDFAPD(bayer.astype(np.float64), pattern='RGGB')
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)

def demosaic_lmmse(bayer: np.ndarray) -> np.ndarray:
    """LMMSE demosaic (Zhang & Wu 2005) — noise-robust directional interpolation.
    Port of darktable/librtprocess implementation to numpy.
    RGGB Bayer pattern. Input: float32 [0,1]. Output: float32 (H,W,3) [0,1]."""
    from scipy.ndimage import convolve1d
    h, w = bayer.shape
    cfa = bayer.astype(np.float64)

    # RGGB masks: (0,0)=R, (0,1)=Gr, (1,0)=Gb, (1,1)=B
    # FC(r,c): 0=R, 1=G, 2=B
    fc = np.zeros((h, w), dtype=np.int32)
    fc[0::2, 1::2] = 1  # Gr
    fc[1::2, 0::2] = 1  # Gb
    fc[1::2, 1::2] = 2  # B
    is_rb = (fc != 1)    # True at R/B locations
    is_g  = (fc == 1)    # True at G locations

    # Step 1: G-R(B) at R/B locations using 5-pixel cross pattern
    # hdiff = G_est_h - CFA, vdiff = G_est_v - CFA
    hdiff = np.zeros_like(cfa)
    vdiff = np.zeros_like(cfa)

    # At R/B locations: estimate G horizontally and vertically
    for r in range(2, h-2):
        for c in range(2, w-2):
            if not is_rb[r, c]:
                continue
            v = cfa
            # horizontal G estimate
            gh = -0.25*(v[r,c-2]+v[r,c+2]) + 0.5*(v[r,c-1]+v[r,c]+v[r,c+1])
            # check for overshooting
            v0 = 0.0625*(v[r-1,c-1]+v[r-1,c+1]+v[r+1,c-1]+v[r+1,c+1]) + 0.25*v[r,c]
            y0 = v0 + 0.5*gh
            if v[r,c] > 1.75*y0:
                gh = np.median([gh, v[r,c-1], v[r,c+1]])
            else:
                gh = np.clip(gh, 0.0, 1.0)
            hdiff[r,c] = gh - v[r,c]
            # vertical G estimate
            gv = -0.25*(v[r-2,c]+v[r+2,c]) + 0.5*(v[r-1,c]+v[r,c]+v[r+1,c])
            y1 = v0 + 0.5*gv
            if v[r,c] > 1.75*y1:
                gv = np.median([gv, v[r-1,c], v[r+1,c]])
            else:
                gv = np.clip(gv, 0.0, 1.0)
            vdiff[r,c] = gv - v[r,c]

    # At G locations: G-R(B) = known G - estimated R/B
    for r in range(2, h-2):
        for c in range(2, w-2):
            if not is_g[r, c]:
                continue
            v = cfa
            hdiff[r,c] = np.clip(0.25*(v[r,c-2]+v[r,c+2]) - 0.5*(v[r,c-1]+v[r,c]+v[r,c+1]),
                                 -1.0, 0.0) + v[r,c]
            vdiff[r,c] = np.clip(0.25*(v[r-2,c]+v[r+2,c]) - 0.5*(v[r-1,c]+v[r,c]+v[r+1,c]),
                                 -1.0, 0.0) + v[r,c]

    # Step 2: Gaussian low-pass filter on differences
    h0 = 1.0
    h1 = np.exp(-1.0/8.0)
    h2 = np.exp(-4.0/8.0)
    h3 = np.exp(-9.0/8.0)
    h4 = np.exp(-16.0/8.0)
    hs = h0 + 2*(h1+h2+h3+h4)
    kernel = np.array([h4, h3, h2, h1, h0, h1, h2, h3, h4]) / hs

    hlp = convolve1d(hdiff, kernel, axis=1, mode='reflect')
    vlp = convolve1d(vdiff, kernel, axis=0, mode='reflect')

    # Step 3: LMMSE fusion at R/B locations
    interp = np.zeros_like(cfa)
    for r in range(8, h-8):
        for c in range(8, w-8):
            if not is_rb[r, c]:
                continue
            # horizontal variance estimation (9-tap window)
            hvals = hlp[r, c-4:c+5]
            mu_h = np.mean(hvals)
            vx_h = 1e-7 + np.sum((hvals - mu_h)**2)
            noise_h = hdiff[r, c-4:c+5]
            vn_h = 1e-7 + np.sum((hvals - noise_h)**2)
            xh = (hdiff[r,c]*vx_h + hlp[r,c]*vn_h) / (vx_h + vn_h)
            vh = vx_h * vn_h / (vx_h + vn_h)

            # vertical variance estimation
            vvals = vlp[r-4:r+5, c]
            mu_v = np.mean(vvals)
            vx_v = 1e-7 + np.sum((vvals - mu_v)**2)
            noise_v = vdiff[r-4:r+5, c]
            vn_v = 1e-7 + np.sum((vvals - noise_v)**2)
            xv = (vdiff[r,c]*vx_v + vlp[r,c]*vn_v) / (vx_v + vn_v)
            vv = vx_v * vn_v / (vx_v + vn_v)

            # fuse
            interp[r,c] = (xh*vv + xv*vh) / (vh + vv)

    # Step 4: Reconstruct RGB
    rgb = np.zeros((h, w, 3), dtype=np.float64)

    # Fill native CFA values
    rgb[0::2, 0::2, 0] = cfa[0::2, 0::2]   # R at R
    rgb[0::2, 1::2, 1] = cfa[0::2, 1::2]   # G at Gr
    rgb[1::2, 0::2, 1] = cfa[1::2, 0::2]   # G at Gb
    rgb[1::2, 1::2, 2] = cfa[1::2, 1::2]   # B at B

    # G at R/B locations from LMMSE interpolated difference
    for r in range(h):
        for c in range(w):
            if is_rb[r, c]:
                rgb[r, c, 1] = cfa[r, c] + interp[r, c]

    # R/B at G locations: guided interpolation
    for r in range(1, h-1):
        for c in range(1, w-1):
            if not is_g[r, c]:
                continue
            g = rgb[r, c, 1]
            if r % 2 == 0:  # Gr row: R neighbors are horizontal, B neighbors vertical
                rgb[r,c,0] = g + 0.5*((rgb[r,c-1,0]-rgb[r,c-1,1]) + (rgb[r,c+1,0]-rgb[r,c+1,1]))
                rgb[r,c,2] = g + 0.5*((rgb[r-1,c,2]-rgb[r-1,c,1]) + (rgb[r+1,c,2]-rgb[r+1,c,1]))
            else:           # Gb row: B neighbors are horizontal, R neighbors vertical
                rgb[r,c,2] = g + 0.5*((rgb[r,c-1,2]-rgb[r,c-1,1]) + (rgb[r,c+1,2]-rgb[r,c+1,1]))
                rgb[r,c,0] = g + 0.5*((rgb[r-1,c,0]-rgb[r-1,c,1]) + (rgb[r+1,c,0]-rgb[r+1,c,1]))

    # R at B / B at R: 4-neighbor guided interpolation
    for r in range(1, h-1):
        for c in range(1, w-1):
            if not is_rb[r, c]:
                continue
            g = rgb[r, c, 1]
            if fc[r, c] == 0:  # R location: interpolate B
                rgb[r,c,2] = g + 0.25*((rgb[r-1,c-1,2]-rgb[r-1,c-1,1]) +
                                        (rgb[r-1,c+1,2]-rgb[r-1,c+1,1]) +
                                        (rgb[r+1,c-1,2]-rgb[r+1,c-1,1]) +
                                        (rgb[r+1,c+1,2]-rgb[r+1,c+1,1]))
            else:              # B location: interpolate R
                rgb[r,c,0] = g + 0.25*((rgb[r-1,c-1,0]-rgb[r-1,c-1,1]) +
                                        (rgb[r-1,c+1,0]-rgb[r-1,c+1,1]) +
                                        (rgb[r+1,c-1,0]-rgb[r+1,c-1,1]) +
                                        (rgb[r+1,c+1,0]-rgb[r+1,c+1,1]))

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)

def raw_to_srgb(bayer: np.ndarray) -> np.ndarray:
    """Bayer RAW → sRGB float32 [0,1]. Menon demosaic + sRGB gamma.
    Input must be RGGB (use crop_to_rggb at load time)."""
    return np.clip(linear_to_srgb(demosaic_menon(bayer)), 0.0, 1.0).astype(np.float32)

# ---------------------------------------------------------------------------
# Per-scene affine WB×CCM calibration (Pre-demosaic → ISP approximation)
# ---------------------------------------------------------------------------
# SIDD provides paired GT_RAW and GT_sRGB (ISP-processed) for each scene.
# By fitting a per-scene 3×4 affine from Menon(GT_RAW) to inv_gamma(GT_sRGB),
# we approximate the camera ISP's WB + CCM + tone mapping in a single matrix.
# Each of 40 scenes gets its own matrix (different AWB gains per scene).
# ---------------------------------------------------------------------------

def estimate_scene_affines(gt_raw_all: np.ndarray, gt_srgb_all: np.ndarray,
                           n_scenes: int, n_patches: int,
                           patches_per_scene: int = 8) -> list:
    """Estimate per-scene 3×4 affine (WB×CCM+offset) from paired GT_RAW ↔ GT_sRGB.

    Input data must already be cropped to RGGB via crop_to_rggb().
    Menon(GT_RAW) → linear_rgb  vs  inv_gamma(GT_sRGB) → linear_rgb.
    Least-squares: linear_isp ≈ [R_menon, G_menon, B_menon, 1] @ M.

    Returns: list of n_scenes (3, 4) float64 affine matrices.
    """
    affines = []
    for s in range(n_scenes):
        A_pixels = []  # source: Menon(GT_RAW) linear
        B_pixels = []  # target: inv_gamma(GT_sRGB) linear
        indices = np.linspace(0, n_patches - 1, patches_per_scene, dtype=int)
        for p in indices:
            idx = s * n_patches + p
            rgb_menon = demosaic_menon(gt_raw_all[idx])
            rgb_isp_lin = srgb_to_linear(gt_srgb_all[idx].astype(np.float64))
            A_pixels.append(rgb_menon.reshape(-1, 3))
            B_pixels.append(rgb_isp_lin.reshape(-1, 3).astype(np.float32))

        A = np.vstack(A_pixels).astype(np.float64)
        B = np.vstack(B_pixels).astype(np.float64)
        A_aug = np.hstack([A, np.ones((A.shape[0], 1))])
        M, _, _, _ = np.linalg.lstsq(A_aug, B, rcond=None)
        affine = M.T  # (3, 4)
        affines.append(affine)
        print(f"    Scene {s:2d}: diag=[{affine[0,0]:.3f}, {affine[1,1]:.3f}, {affine[2,2]:.3f}]"
              f"  offset=[{affine[0,3]:.4f}, {affine[1,3]:.4f}, {affine[2,3]:.4f}]")

    return affines

def raw_to_srgb_calibrated(bayer: np.ndarray, affine_3x4: np.ndarray) -> np.ndarray:
    """Bayer RAW → calibrated sRGB: Menon demosaic → per-scene affine → sRGB gamma.
    Input must be RGGB (cropped at load time)."""
    rgb_linear = demosaic_menon(bayer).astype(np.float64)
    h, w = rgb_linear.shape[:2]
    flat = rgb_linear.reshape(-1, 3)
    out = flat @ affine_3x4[:, :3].T + affine_3x4[:, 3]
    out = np.clip(out.reshape(h, w, 3), 0.0, 1.0)
    return np.clip(linear_to_srgb(out), 0.0, 1.0).astype(np.float32)

# ---------------------------------------------------------------------------
# Sigma estimation helpers
# ---------------------------------------------------------------------------

def sigma_oracle_raw(gt: np.ndarray, noisy: np.ndarray) -> float:
    """True noise sigma in RAW domain. Only valid for synthetic data with known GT.
    Uses std of the actual noise realization (noisy − gt)."""
    return float(np.std(noisy.astype(np.float64) - gt.astype(np.float64)))

def sigma_oracle_srgb(gt: np.ndarray, noisy: np.ndarray) -> float:
    """True noise sigma in sRGB domain (after shared demosaic + gamma)."""
    gt_srgb    = raw_to_srgb(gt)
    noisy_srgb = raw_to_srgb(noisy)
    return float(np.std(noisy_srgb.astype(np.float64) - gt_srgb.astype(np.float64)))

def sigma_mad_raw(noisy: np.ndarray) -> float:
    """Blind sigma estimate in RAW domain via wavelet MAD (finest-scale HH band)."""
    import pywt
    _, (_, _, hh) = pywt.dwt2(noisy.astype(np.float64), 'haar')
    return float(np.median(np.abs(hh)) / 0.6745)

def sigma_mad_srgb(noisy_srgb: np.ndarray) -> float:
    """Blind sigma estimate in sRGB domain via wavelet MAD on luminance."""
    import pywt
    gray = np.mean(noisy_srgb, axis=2)
    _, (_, _, hh) = pywt.dwt2(gray.astype(np.float64), 'haar')
    return float(np.median(np.abs(hh)) / 0.6745)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10.0 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0

def ssim_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(ssim(a, b, data_range=1.0))

def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    return float(ssim(a, b, data_range=1.0, channel_axis=2))

_lpips_fn = None
def compute_lpips(a: np.ndarray, b: np.ndarray) -> float:
    global _lpips_fn
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(dev)
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float().to(dev) * 2 - 1
    tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).float().to(dev) * 2 - 1
    with torch.no_grad():
        return float(_lpips_fn(ta, tb).item())

_dists_fn = None
def compute_dists(a: np.ndarray, b: np.ndarray) -> float:
    global _dists_fn
    import torch
    dev_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    if _dists_fn is None:
        import pyiqa
        _dists_fn = pyiqa.create_metric('dists', device=dev_str)
    dev = torch.device(dev_str)
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
    tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        return float(_dists_fn(ta, tb).item())

# ---------------------------------------------------------------------------
# RAW-domain exe runner (GALOSH, BM3D-CFA blind)
# Pipeline: noisy_raw → exe (fully blind) → denoised_raw → raw_to_srgb
# ---------------------------------------------------------------------------

def run_raw_exe(exe_path: Path, noisy: np.ndarray, w: int, h: int,
                method: str, strength: float, luma_str: float,
                chroma_str: float, uid: str) -> np.ndarray:
    """Call rawdenoise_vX.exe. Returns denoised RAW float32 (H,W).
    Exes are fully blind — any sigma argument is estimated internally."""
    inp = str(BASE / "standalone" / f"tmp_sidd_{uid}.bin")
    out = str(BASE / "standalone" / f"tmp_sidd_out_{uid}.bin")
    noisy.astype(np.float32).tofile(inp)
    cmd = [str(exe_path), inp, out, str(w), str(h),
           method, str(strength), str(luma_str), str(chroma_str)]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    try: os.remove(inp)
    except: pass
    if result.returncode != 0:
        raise RuntimeError(f"{exe_path.name} failed:\n{result.stderr.decode()[:500]}")
    den = np.fromfile(out, dtype=np.float32).reshape(h, w)
    try: os.remove(out)
    except: pass
    return den

def run_raw_gpu_exe(noisy: np.ndarray, w: int, h: int,
                    strength: float, luma_str: float, chroma_str: float,
                    uid: str, cl_device: int = 0) -> np.ndarray:
    """Call galosh_gpu.exe (OpenCL full-pipeline). Returns denoised RAW float32 (H,W).
    Fully blind — alpha/sigma_sq set to 0 for blind estimation on GPU."""
    exe_path = BASE / "standalone" / "galosh_raw_gpu.exe"
    inp = str(BASE / "standalone" / f"tmp_sidd_{uid}.bin")
    out = str(BASE / "standalone" / f"tmp_sidd_out_{uid}.bin")
    noisy.astype(np.float32).tofile(inp)
    cmd = [str(BASH_EXE), "-c",
           f'"{exe_path}" "{inp}" "{out}" {w} {h} 1 '
           f'{strength} {luma_str} {chroma_str} 0.0 0.0 {cl_device}']
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    try: os.remove(inp)
    except: pass
    if result.returncode != 0:
        raise RuntimeError(f"galosh_gpu.exe failed:\n{result.stderr.decode()[:500]}")
    den = np.fromfile(out, dtype=np.float32).reshape(h, w)
    try: os.remove(out)
    except: pass
    return den

def method_galosh_gpu_after(noisy_srgb: np.ndarray, uid: str,
                            cl_device: int = 0) -> np.ndarray:
    """GALOSH GPU after-demosaic: run GPU exe on each RGB channel independently.
    Each channel (H,W) is treated as a 'Bayer' input — the CFA L/C decomposition
    acts as a multi-scale analysis even on single-channel sRGB data."""
    h, w = noisy_srgb.shape[:2]
    out_channels = []
    for c in range(3):
        ch = noisy_srgb[:, :, c]
        den = run_raw_gpu_exe(ch, w, h, 1.0, 1.0, 1.0,
                              f"{uid}_ch{c}", cl_device=cl_device)
        out_channels.append(np.clip(den, 0.0, 1.0))
    return np.stack(out_channels, axis=2).astype(np.float32)


def run_single_plane_gpu(plane: np.ndarray, w: int, h: int,
                         strength: float, uid: str,
                         cl_device: int = 0) -> np.ndarray:
    """Call galosh_single_gpu.exe (all-GPU single-plane pipeline).
    Input/output: single float32 plane (H,W)."""
    exe_path = BASE / "standalone" / "galosh_single_gpu.exe"
    inp = str(BASE / "standalone" / f"tmp_sp_{uid}.bin")
    out = str(BASE / "standalone" / f"tmp_sp_out_{uid}.bin")
    plane.astype(np.float32).tofile(inp)
    cmd = [str(BASH_EXE), "-c",
           f'"{exe_path}" "{inp}" "{out}" {w} {h} '
           f'{strength} {cl_device}']
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    try: os.remove(inp)
    except: pass
    if result.returncode != 0:
        raise RuntimeError(f"galosh_gpu single failed:\n{result.stderr.decode()[:500]}")
    den = np.fromfile(out, dtype=np.float32).reshape(h, w)
    try: os.remove(out)
    except: pass
    return den


def method_galosh_yuv_gpu(noisy_srgb: np.ndarray, uid: str,
                          strength_y: float = 1.0, strength_c: float = 1.0,
                          cl_device: int = 0) -> np.ndarray:
    """GALOSH YUV GPU: RGB→YCbCr (BT.709) → per-plane GPU denoise → RGB.
    All signal processing on GPU (single-plane mode). CPU does color space only."""
    H, W = noisy_srgb.shape[:2]
    R, G, B = noisy_srgb[..., 0], noisy_srgb[..., 1], noisy_srgb[..., 2]

    # BT.709 forward
    Y  =  0.2126 * R + 0.7152 * G + 0.0722 * B
    Cb = -0.1146 * R - 0.3854 * G + 0.5000 * B
    Cr =  0.5000 * R - 0.4542 * G - 0.0458 * B

    # Denoise each plane on GPU
    Y_d  = run_single_plane_gpu(Y,  W, H, strength_y, f"{uid}_Y",  cl_device)
    Cb_d = run_single_plane_gpu(Cb, W, H, strength_c, f"{uid}_Cb", cl_device)
    Cr_d = run_single_plane_gpu(Cr, W, H, strength_c, f"{uid}_Cr", cl_device)

    # BT.709 inverse
    R_d = np.clip(Y_d + 1.5748 * Cr_d,                  0.0, 1.0)
    G_d = np.clip(Y_d - 0.1873 * Cb_d - 0.4681 * Cr_d, 0.0, 1.0)
    B_d = np.clip(Y_d + 1.8556 * Cb_d,                  0.0, 1.0)
    return np.stack([R_d, G_d, B_d], axis=2).astype(np.float32)

# ---------------------------------------------------------------------------
# BM3D-CFA oracle — Python bm3d package with true sigma in RAW domain
# Pipeline: noisy_raw → bm3d(sigma=oracle) → denoised_raw → raw_to_srgb
# ---------------------------------------------------------------------------

def run_bm3d_raw(noisy: np.ndarray, sigma: float) -> np.ndarray:
    """BM3D on RAW Bayer (treated as grayscale) with explicit sigma.
    Used for oracle comparison where true sigma is known from GT.
    Serialized with _bm3d_lock — bm3d uses internal threads, not re-entrant."""
    import bm3d as bm3d_pkg
    with _bm3d_lock:
        den = bm3d_pkg.bm3d(noisy.astype(np.float64), sigma_psd=sigma)
    return np.clip(den, 0.0, 1.0).astype(np.float32)

# ---------------------------------------------------------------------------
# NLM — sRGB domain with explicit or estimated sigma
# Pipeline: noisy_raw → raw_to_srgb → nlm(sigma) → denoised_srgb
# ---------------------------------------------------------------------------

def run_nlm_srgb(noisy_srgb: np.ndarray, sigma: float) -> np.ndarray:
    """Non-Local Means per-channel in sRGB with given sigma.
    Serialized with _nlm_lock — skimage NLM uses Cython-level parallelism."""
    from skimage.restoration import denoise_nl_means
    out_channels = []
    for c in range(3):
        ch = noisy_srgb[:, :, c].astype(np.float64)
        with _nlm_lock:
            den = denoise_nl_means(ch, h=1.15 * sigma, sigma=sigma,
                                   fast_mode=True, patch_size=5, patch_distance=6)
        out_channels.append(np.clip(den, 0.0, 1.0).astype(np.float32))
    return np.stack(out_channels, axis=2)

def method_nlm_cfa(noisy_bayer: np.ndarray, sigma: float) -> np.ndarray:
    """NLM on CFA: split RGGB Bayer into 4 quarter-res planes, NLM each, reassemble.
    Pre-demosaic counterpart of sRGB-domain NLM."""
    from skimage.restoration import denoise_nl_means
    h, w = noisy_bayer.shape
    denoised = np.zeros_like(noisy_bayer)
    for r0, c0 in [(0, 0), (0, 1), (1, 0), (1, 1)]:  # R, Gr, Gb, B
        plane = noisy_bayer[r0::2, c0::2].astype(np.float64)
        with _nlm_lock:
            den = denoise_nl_means(plane, h=1.15 * sigma, sigma=sigma,
                                   fast_mode=True, patch_size=5, patch_distance=6)
        denoised[r0::2, c0::2] = np.clip(den, 0.0, 1.0).astype(np.float32)
    return denoised

def run_bm3d_srgb(noisy_srgb: np.ndarray, sigma: float) -> np.ndarray:
    """BM3D per-channel on sRGB with given sigma. After-demosaic counterpart."""
    import bm3d as bm3d_pkg
    out_channels = []
    for c in range(3):
        with _bm3d_lock:
            den = bm3d_pkg.bm3d(noisy_srgb[:, :, c].astype(np.float64), sigma_psd=sigma)
        out_channels.append(np.clip(den, 0.0, 1.0).astype(np.float32))
    return np.stack(out_channels, axis=2)

# ---------------------------------------------------------------------------
# DL models — sRGB domain (KAIR)
# Pipeline: noisy_raw → raw_to_srgb → DL → denoised_srgb
# ---------------------------------------------------------------------------

_dncnn_model = None
def get_dncnn():
    global _dncnn_model
    if _dncnn_model is not None:
        return _dncnn_model
    sys.path.insert(0, str(KAIR_DIR))
    from models.network_dncnn import DnCNN
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DnCNN(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='R')
    state = torch.load(str(KAIR_DIR / "model_zoo" / "dncnn_gray_blind.pth"),
                       map_location=dev, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(dev)
    _dncnn_model = model
    return model

def method_dncnn_b(noisy_srgb: np.ndarray) -> np.ndarray:
    import torch
    model = get_dncnn()
    dev = next(model.parameters()).device
    out_channels = []
    for c in range(3):
        inp = torch.from_numpy(noisy_srgb[:, :, c]).unsqueeze(0).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            res = model(inp)
        out_channels.append(res.squeeze().clamp(0, 1).cpu().numpy())
    return np.stack(out_channels, axis=2).astype(np.float32)

# ---------------------------------------------------------------------------
# GALOSH YUV444 — sRGB-domain denoiser via yuv_galosh.exe
# Pipeline: noisy_sRGB → BT.709 YCbCr (444) → exe → YCbCr → sRGB
# Uses ISP-processed sRGB as input/GT — same evaluation domain as NAFNet.
# ---------------------------------------------------------------------------

def run_galosh_yuv_444(noisy_srgb: np.ndarray, uid: str) -> np.ndarray:
    """Run yuv_galosh.exe in YUV444 mode on an sRGB float32 [0,1] patch (H,W,3).
    BT.709 RGB↔YCbCr conversion (same as bench_compare.py).
    Returns denoised sRGB float32 [0,1] (H,W,3)."""
    H, W = noisy_srgb.shape[:2]
    R, G, B = noisy_srgb[..., 0], noisy_srgb[..., 1], noisy_srgb[..., 2]
    # BT.709 forward transform
    Y  =  0.2126 * R + 0.7152 * G + 0.0722 * B
    Cb = -0.1146 * R - 0.3854 * G + 0.5000 * B
    Cr =  0.5000 * R - 0.4542 * G - 0.0458 * B

    yuv_dir = BASE / "standalone" / "yuv_bench"
    yuv_dir.mkdir(exist_ok=True)
    in_path  = str(yuv_dir / f"sidd_{uid}_in.yuv")
    out_path = str(yuv_dir / f"sidd_{uid}_out.yuv")

    # Write Y, Cb, Cr as flat float32 (444: all planes same size H×W)
    with open(in_path, "wb") as f:
        Y.astype(np.float32).tofile(f)
        Cb.astype(np.float32).tofile(f)
        Cr.astype(np.float32).tofile(f)

    def win2bash(p: str) -> str:
        return p.replace("C:", "/c").replace("\\", "/")

    exe_path = BASE / "standalone" / "yuv_galosh.exe"
    cmd = (f"{win2bash(str(exe_path))} {win2bash(in_path)} {win2bash(out_path)}"
           f" {W} {H} 444 1.0 1.0 2 2")
    env = dict(os.environ)
    env["PATH"] = "C:/msys64/ucrt64/bin;C:/msys64/usr/bin;" + env.get("PATH", "")
    r = subprocess.run([str(BASH_EXE), "-lc", cmd], capture_output=True,
                       text=True, encoding="utf-8", errors="replace",
                       env=env, timeout=60)
    try:
        os.remove(in_path)
    except OSError:
        pass
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"yuv_galosh failed (rc={r.returncode}): {r.stderr[:200]}")

    d = np.fromfile(out_path, dtype=np.float32)
    try:
        os.remove(out_path)
    except OSError:
        pass

    Y_d  = d[0        : H * W    ].reshape(H, W)
    Cb_d = d[H * W    : 2 * H * W].reshape(H, W)
    Cr_d = d[2 * H * W:           ].reshape(H, W)

    # BT.709 inverse transform
    R_d = np.clip(Y_d + 1.5748 * Cr_d,                  0.0, 1.0)
    G_d = np.clip(Y_d - 0.1873 * Cb_d - 0.4681 * Cr_d, 0.0, 1.0)
    B_d = np.clip(Y_d + 1.8556 * Cb_d,                  0.0, 1.0)
    return np.stack([R_d, G_d, B_d], axis=2).astype(np.float32)


def run_galosh_yuv_444_linear(noisy_srgb: np.ndarray, uid: str) -> np.ndarray:
    """GALOSH YUV444 with exe-side gamma-aware alpha estimation (gamma=2.2).

    The exe's variance-mean regression (Var=alpha*x+sigma^2) assumes linear domain.
    Passing gamma=2.2 activates the built-in compensation:
      x_lin = x_sRGB^2.2,  jacob = (1/2.2)*x_sRGB^(-1.2)
      Var_lin = Var_sRGB / jacob^2 = Var_sRGB * 4.84 * x_sRGB^2.4
    This lets the exe correctly detect the Poisson alpha in ISP sRGB input.

    Input/output: ISP sRGB float32 [0,1] HWC.
    GT reference: ISP sRGB (same as NAFNet) -- fair apples-to-apples comparison."""
    H, W = noisy_srgb.shape[:2]
    R, G, B = noisy_srgb[..., 0], noisy_srgb[..., 1], noisy_srgb[..., 2]
    Y  =  0.2126 * R + 0.7152 * G + 0.0722 * B
    Cb = -0.1146 * R - 0.3854 * G + 0.5000 * B
    Cr =  0.5000 * R - 0.4542 * G - 0.0458 * B

    yuv_dir = BASE / "standalone" / "yuv_bench"
    yuv_dir.mkdir(exist_ok=True)
    in_path  = str(yuv_dir / f"sidd_{uid}_g22in.yuv")
    out_path = str(yuv_dir / f"sidd_{uid}_g22out.yuv")

    with open(in_path, "wb") as f:
        Y.astype(np.float32).tofile(f)
        Cb.astype(np.float32).tofile(f)
        Cr.astype(np.float32).tofile(f)

    def win2bash(p: str) -> str:
        return p.replace("C:", "/c").replace("\\", "/")

    exe_path = BASE / "standalone" / "yuv_galosh.exe"
    # gamma=2.2: exe compensates variance-mean regression to linear domain internally
    cmd = (f"{win2bash(str(exe_path))} {win2bash(in_path)} {win2bash(out_path)}"
           f" {W} {H} 444 1.0 1.0 2 2 2.2")
    env = dict(os.environ)
    env["PATH"] = "C:/msys64/ucrt64/bin;C:/msys64/usr/bin;" + env.get("PATH", "")
    r = subprocess.run([str(BASH_EXE), "-lc", cmd], capture_output=True,
                       text=True, encoding="utf-8", errors="replace",
                       env=env, timeout=60)
    try:
        os.remove(in_path)
    except OSError:
        pass
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"yuv_galosh gamma=2.2 failed (rc={r.returncode}): {r.stderr[:200]}")

    d = np.fromfile(out_path, dtype=np.float32)
    try:
        os.remove(out_path)
    except OSError:
        pass

    Y_d  = d[0        : H * W    ].reshape(H, W)
    Cb_d = d[H * W    : 2 * H * W].reshape(H, W)
    Cr_d = d[2 * H * W:           ].reshape(H, W)

    R_d = np.clip(Y_d + 1.5748 * Cr_d,                  0.0, 1.0)
    G_d = np.clip(Y_d - 0.1873 * Cb_d - 0.4681 * Cr_d, 0.0, 1.0)
    B_d = np.clip(Y_d + 1.8556 * Cb_d,                  0.0, 1.0)
    return np.stack([R_d, G_d, B_d], axis=2).astype(np.float32)

_nafnet_model = None
def get_nafnet():
    """Load NAFNet-SIDD-width64 using the NAFNet-local basicsr (avoids torchvision compat issue)."""
    global _nafnet_model
    if _nafnet_model is not None:
        return _nafnet_model
    # Insert NAFNet local path first so it shadows any installed (incompatible) basicsr
    nafnet_str = str(NAFNET_DIR)
    if nafnet_str not in sys.path:
        sys.path.insert(0, nafnet_str)
    from basicsr.models.archs.NAFNet_arch import NAFNet
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    ckpt_path = NAFNET_DIR / "experiments" / "pretrained_models" / "NAFNet-SIDD-width64.pth"
    state = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    model.load_state_dict(state["params"], strict=True)
    model.eval().to(dev)
    _nafnet_model = model
    return model

def method_nafnet(noisy_srgb: np.ndarray) -> np.ndarray:
    """NAFNet denoising on sRGB [0,1] float32 (H,W,3).
    Trained on SIDD sRGB training data — proper apples-to-apples comparison.
    Input/output: float32 [0,1] HWC → HWC."""
    import torch
    model = get_nafnet()
    dev = next(model.parameters()).device
    # (H, W, 3) → (1, 3, H, W) float32
    inp = torch.from_numpy(noisy_srgb.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        out = model(inp)
    return out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy().astype(np.float32)

_drunet_model = None
def get_drunet():
    global _drunet_model
    if _drunet_model is not None:
        return _drunet_model
    sys.path.insert(0, str(KAIR_DIR))
    from models.network_unet import UNetRes
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512],
                    nb=4, act_mode='R', downsample_mode='strideconv',
                    upsample_mode='convtranspose', bias=False)
    state = torch.load(str(KAIR_DIR / "model_zoo" / "drunet_gray.pth"),
                       map_location=dev, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(dev)
    _drunet_model = model
    return model

def method_drunet(noisy_srgb: np.ndarray) -> np.ndarray:
    """DRUNet with MAD sigma estimate from sRGB."""
    import torch
    model  = get_drunet()
    dev    = next(model.parameters()).device
    sigma  = sigma_mad_srgb(noisy_srgb)
    h, w   = noisy_srgb.shape[:2]
    pad_h  = (8 - h % 8) % 8
    pad_w  = (8 - w % 8) % 8
    out_channels = []
    for c in range(3):
        ch = noisy_srgb[:, :, c]
        if pad_h or pad_w:
            ch = np.pad(ch, ((0, pad_h), (0, pad_w)), mode='reflect')
        sigma_map = np.full_like(ch, sigma)
        inp = torch.from_numpy(np.stack([ch, sigma_map])).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            res = model(inp)
        out_ch = res.squeeze().clamp(0, 1).cpu().numpy()
        if pad_h or pad_w:
            out_ch = out_ch[:h, :w]
        out_channels.append(out_ch)
    return np.stack(out_channels, axis=2).astype(np.float32)

# ---------------------------------------------------------------------------
# Method registry
# fn signature: (noisy_raw, gt_raw, w, h, uid) -> (den_raw | None, den_srgb)
#   - gt_raw is provided for oracle methods; blind methods must ignore it
#   - den_raw: denoised RAW float32 (H,W) or None for sRGB-domain methods
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    # --- Pre-demosaic: RAW → [denoise] → Menon → affine WB×CCM → sRGB gamma ---
    "galosh_gpu":         "GALOSH GPU",
    "bm3dcfa_blind":      "BM3D-CFA (blind)",
    "bm3dcfa_oracle":     "BM3D-CFA (oracle)",
    "nlm_cfa_blind":      "NLM-CFA (blind)",
    "nlm_cfa_oracle":     "NLM-CFA (oracle)",
    # --- After-demosaic: ISP sRGB → [denoise] → eval vs ISP sRGB GT ---
    "galosh_yuv_gpu":     "GALOSH-YUV GPU",
    "nafnet":             "NAFNet-SIDD",
    "dncnn_b_srgb":       "DnCNN-B",
    "drunet_srgb":        "DRUNet",
    "bm3d_srgb":          "BM3D",
    "nlm_srgb":           "NLM",
}

def build_method_registry() -> dict:
    exe_v4  = BASE / "standalone" / "rawdenoise_v4.exe"

    # ---------------------------------------------------------------
    # Pre-demosaic: RAW → [denoise] → Menon → affine WB×CCM → sRGB γ
    # gt_source="raw_calibrated" — dual GT evaluation (GT② + GT①)
    # fn signature: (noisy, gt, w, h, uid, affine=None, **kw)
    # ---------------------------------------------------------------

    def raw_exe_cal(exe_path, method="bm3dcfa", strength=0.5, ls=1.0, cs=1.0):
        """Pre-demosaic via C exe, calibrated sRGB output."""
        def fn(noisy, gt, w, h, uid, affine=None, **kw):
            den_raw = run_raw_exe(exe_path, noisy, w, h,
                                  method, strength, ls, cs, uid=uid)
            den_srgb = (raw_to_srgb_calibrated(den_raw, affine)
                        if affine is not None else raw_to_srgb(den_raw))
            return den_raw, den_srgb
        return {"fn": fn, "domain": "RAW", "exe_path": exe_path,
                "gt_source": "raw_calibrated"}

    def raw_gpu_cal(strength=1.0, ls=1.0, cs=1.0, cl_device=0):
        """GALOSH GPU (OpenCL) pre-demosaic, calibrated sRGB output."""
        gpu_exe = BASE / "standalone" / "galosh_raw_gpu.exe"
        def fn(noisy, gt, w, h, uid, affine=None, **kw):
            den_raw = run_raw_gpu_exe(noisy, w, h, strength, ls, cs,
                                      uid=uid, cl_device=cl_device)
            den_srgb = (raw_to_srgb_calibrated(den_raw, affine)
                        if affine is not None else raw_to_srgb(den_raw))
            return den_raw, den_srgb
        return {"fn": fn, "domain": "RAW", "exe_path": gpu_exe,
                "gt_source": "raw_calibrated"}

    def bm3d_oracle_cal():
        """BM3D-CFA oracle (Python bm3d, true sigma from GT)."""
        def fn(noisy, gt, w, h, uid, affine=None, **kw):
            sigma = sigma_oracle_raw(gt, noisy)
            den_raw = run_bm3d_raw(noisy, sigma)
            den_srgb = (raw_to_srgb_calibrated(den_raw, affine)
                        if affine is not None else raw_to_srgb(den_raw))
            return den_raw, den_srgb
        return {"fn": fn, "domain": "RAW", "exe_path": None,
                "gt_source": "raw_calibrated"}

    def nlm_cfa(use_oracle: bool):
        """NLM-CFA: per-Bayer-channel NLM, calibrated sRGB output."""
        def fn(noisy, gt, w, h, uid, affine=None, **kw):
            sigma = sigma_oracle_raw(gt, noisy) if use_oracle else sigma_mad_raw(noisy)
            den_raw = method_nlm_cfa(noisy, sigma)
            den_srgb = (raw_to_srgb_calibrated(den_raw, affine)
                        if affine is not None else raw_to_srgb(den_raw))
            return den_raw, den_srgb
        return {"fn": fn, "domain": "RAW", "exe_path": None,
                "gt_source": "raw_calibrated"}

    # ---------------------------------------------------------------
    # After-demosaic: ISP sRGB → [denoise] → eval vs ISP sRGB GT
    # gt_source="srgb" — uses ValidationNoisyBlocksSrgb / GtBlocksSrgb
    # fn signature: (noisy, gt, w, h, uid, noisy_srgb_ref=None,
    #                gt_srgb_ref=None)
    # ---------------------------------------------------------------

    def dl_from_srgb(denoise_fn):
        """DL model on ISP sRGB input."""
        def fn(noisy, gt, w, h, uid, noisy_srgb_ref=None, gt_srgb_ref=None):
            if noisy_srgb_ref is None:
                raise RuntimeError("sRGB mat not loaded")
            den_srgb = denoise_fn(noisy_srgb_ref)
            return None, den_srgb
        return {"fn": fn, "domain": "sRGB(ISP)", "exe_path": None,
                "gt_source": "srgb"}

    def yuv_gpu_srgb(strength_y=1.0, strength_c=1.0, cl_device=0):
        """GALOSH YUV GPU: ISP sRGB → BT.709 YCbCr → per-plane GPU → RGB."""
        gpu_exe = BASE / "standalone" / "galosh_raw_gpu.exe"
        def fn(noisy, gt, w, h, uid, noisy_srgb_ref=None, gt_srgb_ref=None):
            if noisy_srgb_ref is None:
                raise RuntimeError("sRGB mat not loaded")
            den_srgb = method_galosh_yuv_gpu(noisy_srgb_ref, uid,
                                             strength_y, strength_c, cl_device)
            return None, den_srgb
        return {"fn": fn, "domain": "sRGB(ISP)", "exe_path": gpu_exe,
                "gt_source": "srgb"}

    def bm3d_from_srgb():
        """BM3D per-channel on ISP sRGB (blind, wavelet MAD sigma)."""
        def fn(noisy, gt, w, h, uid, noisy_srgb_ref=None, gt_srgb_ref=None):
            if noisy_srgb_ref is None:
                raise RuntimeError("sRGB mat not loaded")
            sigma = sigma_mad_srgb(noisy_srgb_ref)
            den_srgb = run_bm3d_srgb(noisy_srgb_ref, sigma)
            return None, den_srgb
        return {"fn": fn, "domain": "sRGB(ISP)", "exe_path": None,
                "gt_source": "srgb"}

    def nlm_from_srgb():
        """NLM per-channel on ISP sRGB (blind, wavelet MAD sigma)."""
        def fn(noisy, gt, w, h, uid, noisy_srgb_ref=None, gt_srgb_ref=None):
            if noisy_srgb_ref is None:
                raise RuntimeError("sRGB mat not loaded")
            sigma = sigma_mad_srgb(noisy_srgb_ref)
            den_srgb = run_nlm_srgb(noisy_srgb_ref, sigma)
            return None, den_srgb
        return {"fn": fn, "domain": "sRGB(ISP)", "exe_path": None,
                "gt_source": "srgb"}

    return {
        # Pre-demosaic group
        "galosh_gpu":      raw_gpu_cal(strength=1.0, ls=1.0, cs=1.0),
        "bm3dcfa_blind":   raw_exe_cal(exe_v4, "bm3dcfa", 0.5, 1.0, 1.0),
        "bm3dcfa_oracle":  bm3d_oracle_cal(),
        "nlm_cfa_blind":   nlm_cfa(use_oracle=False),
        "nlm_cfa_oracle":  nlm_cfa(use_oracle=True),
        # After-demosaic group
        "galosh_yuv_gpu":  yuv_gpu_srgb(strength_y=1.0, strength_c=1.0),
        "nafnet":          dl_from_srgb(method_nafnet),
        "dncnn_b_srgb":    dl_from_srgb(method_dncnn_b),
        "drunet_srgb":     dl_from_srgb(method_drunet),
        "bm3d_srgb":       bm3d_from_srgb(),
        "nlm_srgb":        nlm_from_srgb(),
    }

# ---------------------------------------------------------------------------
# Per-patch worker (runs in thread pool)
# ---------------------------------------------------------------------------

def process_patch(args: tuple) -> dict:
    """Process one patch for one method.

    args tuple:
      i, method_id, method_info, gt_raw, noisy_raw, W, H, use_perceptual,
      noisy_srgb_ref, gt_srgb_ref, affine, gt_srgb_cal

    All data is already cropped to RGGB at load time — no pattern needed.

    GT evaluation:
      - After-demosaic (gt_source="srgb"): GT = ISP sRGB (gt_srgb_ref)
      - Pre-demosaic  (gt_source="raw_calibrated"):
          GT② (primary) = GT_RAW through calibrated pipeline (gt_srgb_cal)
          GT① (cross)   = ISP sRGB (gt_srgb_ref) for NAFNet comparison
    """
    i, method_id, method_info, gt, noisy, W, H, use_perceptual, \
        noisy_srgb_ref, gt_srgb_ref, affine, gt_srgb_cal = args
    fn  = method_info["fn"]
    uid = f"{method_id}_{i}"
    gt_source = method_info.get("gt_source", "raw")

    # --- Run denoiser (timed) ---
    t0 = time.time()
    if gt_source == "raw_calibrated":
        den_raw, den_srgb = fn(noisy, gt, W, H, uid, affine=affine)
    else:
        den_raw, den_srgb = fn(noisy, gt, W, H, uid, noisy_srgb_ref, gt_srgb_ref)
    dt = time.time() - t0

    # --- Select primary GT ---
    if gt_source == "raw_calibrated" and gt_srgb_cal is not None:
        gt_primary = gt_srgb_cal   # GT②: pure denoise quality
    elif gt_source == "srgb" and gt_srgb_ref is not None:
        gt_primary = gt_srgb_ref   # ISP sRGB for after-demosaic
    else:
        gt_primary = raw_to_srgb(gt)

    result = {
        "patch_idx": i,
        "time_s":    dt,
        "den_srgb":  den_srgb,
        "den_raw":   den_raw,
        "rgb_psnr":  psnr(gt_primary, den_srgb),
        "rgb_ssim":  ssim_rgb(gt_primary, den_srgb),
    }

    # Pre-demosaic cross-comparison vs ISP GT (GT①)
    if gt_source == "raw_calibrated" and gt_srgb_ref is not None:
        result["isp_psnr"] = psnr(gt_srgb_ref, den_srgb)
        result["isp_ssim"] = ssim_rgb(gt_srgb_ref, den_srgb)

    if den_raw is not None:
        result["raw_psnr"] = psnr(gt, den_raw)
        result["raw_ssim"] = ssim_2d(gt, den_raw)
    if use_perceptual:
        result["lpips"] = compute_lpips(gt_primary, den_srgb)
        result["dists"] = compute_dists(gt_primary, den_srgb)
        if gt_source == "raw_calibrated" and gt_srgb_ref is not None:
            result["isp_lpips"] = compute_lpips(gt_srgb_ref, den_srgb)
            result["isp_dists"] = compute_dists(gt_srgb_ref, den_srgb)
    return result

# ---------------------------------------------------------------------------
# Comparison grid builder — one image: rows=patches, cols=methods+ref
# ---------------------------------------------------------------------------

def build_grid_image(gt_raw: np.ndarray, noisy_raw: np.ndarray,
                     method_den_map: dict, method_order: list) -> np.ndarray:
    N, H, W  = gt_raw.shape
    col_keys = ["__noisy__", "__gt__"] + method_order
    col_labels = ["Noisy", "GT"] + [DISPLAY_NAMES.get(m, m) for m in method_order]
    n_cols = len(col_keys)

    label_h = 20
    gap     = 2
    total_h = label_h + N * H + (N - 1) * gap
    total_w = n_cols * W + (n_cols - 1) * gap
    grid = np.ones((total_h, total_w, 3), dtype=np.uint8) * 180

    for row in range(N):
        gt_u8    = (np.clip(raw_to_srgb(gt_raw[row]),    0, 1) * 255).astype(np.uint8)
        noisy_u8 = (np.clip(raw_to_srgb(noisy_raw[row]), 0, 1) * 255).astype(np.uint8)

        for col, key in enumerate(col_keys):
            y = label_h + row * (H + gap)
            x = col * (W + gap)
            if key == "__noisy__":
                patch = noisy_u8
            elif key == "__gt__":
                patch = gt_u8
            else:
                d = method_den_map[key][row]
                patch = (np.clip(d["den_srgb"] if d else np.full((H,W,3),0.5), 0,1) * 255
                         ).astype(np.uint8)
            grid[y:y+H, x:x+W] = patch

    # Column header labels
    try:
        from PIL import Image, ImageDraw
        img  = Image.fromarray(grid)
        draw = ImageDraw.Draw(img)
        for col, label in enumerate(col_labels):
            x = col * (W + gap)
            draw.rectangle([x, 0, x + W - 1, label_h - 1], fill=(40, 40, 40))
            draw.text((x + 2, 3), label[:20], fill=(255, 255, 255))
        grid = np.array(img)
    except ImportError:
        pass  # no Pillow, save without text labels

    return grid

# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(methods: list, use_perceptual: bool, n_workers: int,
                  max_scenes: int = 0):
    registry = build_method_registry()

    # Validate exe availability
    active = []
    for mid in methods:
        info = registry[mid]
        exe  = info.get("exe_path")
        if exe is not None and not exe.exists():
            print(f"  SKIP {mid}: {exe.name} not found")
            continue
        active.append(mid)
    if not active:
        print("No methods available."); return

    # Load SIDD RAW validation data
    print("Loading SIDD RAW validation data...", end=" ", flush=True)
    import scipy.io as sio
    gt_mat    = sio.loadmat(str(SIDD_GT_MAT))["ValidationGtBlocksRaw"]
    noisy_mat = sio.loadmat(str(SIDD_NOISY_MAT))["ValidationNoisyBlocksRaw"]
    n_scenes, n_pp, H_orig, W_orig = gt_mat.shape
    print(f"OK  ({n_scenes}×{n_pp}, {W_orig}×{H_orig})")

    # Always load sRGB (after-demosaic GT + pre-demosaic affine/cross-comparison)
    print("Loading SIDD sRGB validation data...", end=" ", flush=True)
    gt_srgb_mat    = sio.loadmat(str(SIDD_GT_SRGB_MAT))["ValidationGtBlocksSrgb"]
    noisy_srgb_mat = sio.loadmat(str(SIDD_NOISY_SRGB_MAT))["ValidationNoisyBlocksSrgb"]
    print("OK")

    # --- Detect Bayer pattern per scene and crop ALL data to RGGB ---
    # After crop, different patterns yield different sizes (256→254 or 256).
    # Unify to smallest common size (254×254) so all patches are equal.
    print("Detecting Bayer patterns and cropping to RGGB alignment...")
    H, W = 254, 254  # minimum after crop_to_rggb on any pattern
    N = n_scenes * n_pp
    gt_raw         = np.zeros((N, H, W), dtype=np.float32)
    noisy_raw      = np.zeros((N, H, W), dtype=np.float32)
    gt_srgb_all    = np.zeros((N, H, W, 3), dtype=np.float32)
    noisy_srgb_all = np.zeros((N, H, W, 3), dtype=np.float32)

    for s in range(n_scenes):
        # Detect pattern from first GT patch of this scene
        srgb_ref = gt_srgb_mat[s, 0].astype(np.float32) / 255.0
        pat = detect_bayer_pattern(gt_mat[s, 0].astype(np.float32), srgb_ref)
        print(f"    Scene {s:2d}: {pat}")
        for p in range(n_pp):
            idx = s * n_pp + p
            # Crop RAW (2D) to RGGB, then trim to common H×W
            g  = crop_to_rggb(gt_mat[s, p].astype(np.float32), pat)
            n  = crop_to_rggb(noisy_mat[s, p].astype(np.float32), pat)
            gt_raw[idx]    = g[:H, :W]
            noisy_raw[idx] = n[:H, :W]
            # Crop sRGB (3D) to match spatial alignment, then trim
            gs = crop_to_rggb(gt_srgb_mat[s, p].astype(np.float32) / 255.0, pat)
            ns = crop_to_rggb(noisy_srgb_mat[s, p].astype(np.float32) / 255.0, pat)
            gt_srgb_all[idx]    = gs[:H, :W]
            noisy_srgb_all[idx] = ns[:H, :W]

    print(f"  All data cropped to RGGB: {W}×{H}, {N} patches")

    # Optional scene limit
    if max_scenes and max_scenes < n_scenes:
        N = max_scenes * n_pp
        gt_raw         = gt_raw[:N]
        noisy_raw      = noisy_raw[:N]
        gt_srgb_all    = gt_srgb_all[:N]
        noisy_srgb_all = noisy_srgb_all[:N]
        n_scenes       = max_scenes
        print(f"  (limited to first {max_scenes} scenes = {N} patches)")

    noisy_ref_psnrs = [psnr(gt_raw[i], noisy_raw[i]) for i in range(N)]
    print(f"Noisy baseline: RAW PSNR = {np.mean(noisy_ref_psnrs):.2f} dB")

    # Per-scene affine WB×CCM estimation for pre-demosaic calibrated pipeline
    needs_cal = any(registry[m].get("gt_source") == "raw_calibrated" for m in active)
    affines = None
    gt_srgb_cal = None
    if needs_cal:
        print("Estimating per-scene affine WB×CCM from paired GT...")
        affines = estimate_scene_affines(gt_raw, gt_srgb_all, n_scenes, n_pp)

        # Precompute GT② = GT_RAW → Menon → affine → sRGB gamma
        print("Precomputing calibrated GT② (GT_RAW → Menon → affine → sRGB)...",
              flush=True)
        gt_srgb_cal = np.zeros((N, H, W, 3), dtype=np.float32)
        for i in range(N):
            scene_idx = i // n_pp
            gt_srgb_cal[i] = raw_to_srgb_calibrated(
                gt_raw[i], affines[scene_idx])
            if (i + 1) % 100 == 0 or i + 1 == N:
                print(f"    [{i+1}/{N}]", flush=True)

    if use_perceptual:
        print("Pre-loading LPIPS/DISTS...", end=" ", flush=True)
        dummy = np.zeros((H, W, 3), dtype=np.float32)
        compute_lpips(dummy, dummy)
        compute_dists(dummy, dummy)
        print("OK")

    print(f"\n{'='*70}")
    print(f"Methods  : {', '.join(active)}")
    print(f"Patches  : {N}  |  Size: {W}×{H}  |  Workers: {n_workers}")
    print(f"{'='*70}\n")

    all_results    = []
    method_den_map = {}

    for mid in active:
        info = registry[mid]
        name = DISPLAY_NAMES.get(mid, mid)
        gt_source = info.get("gt_source", "raw")
        gt_label = "calibrated pipeline" if gt_source == "raw_calibrated" else "ISP sRGB"
        print(f"[{name}]  (GT: {gt_label})", flush=True)

        patch_args = []
        for i in range(N):
            scene_idx = i // n_pp
            aff  = affines[scene_idx] if affines is not None else None
            gcal = gt_srgb_cal[i] if gt_srgb_cal is not None else None
            ns_ref = noisy_srgb_all[i] if gt_source == "srgb" else None
            gs_ref = gt_srgb_all[i]  # always pass ISP GT for cross-comparison

            patch_args.append(
                (i, mid, info,
                 gt_raw[i], noisy_raw[i],
                 W, H, use_perceptual,
                 ns_ref, gs_ref, aff, gcal)
            )

        t_method      = time.time()
        patch_results = [None] * N
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(process_patch, args): args[0]
                       for args in patch_args}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                try:
                    patch_results[idx] = fut.result()
                except Exception as e:
                    print(f"  patch {idx} FAILED: {e}")
                done += 1
                if done % 50 == 0 or done == N:
                    ok_so_far = [r for r in patch_results if r]
                    if ok_so_far:
                        msg = f"  [{done:4d}/{N}] PSNR ~{np.mean([r['rgb_psnr'] for r in ok_so_far]):.2f}"
                        if ok_so_far[0].get("isp_psnr") is not None:
                            msg += f"  ISP_PSNR ~{np.mean([r['isp_psnr'] for r in ok_so_far if 'isp_psnr' in r]):.2f}"
                        print(msg, flush=True)
        elapsed = time.time() - t_method

        ok = [r for r in patch_results if r is not None]
        if not ok:
            print(f"  All patches failed.\n"); continue

        method_den_map[mid] = patch_results

        def avg(key):
            vals = [r[key] for r in ok if key in r]
            return float(np.mean(vals)) if vals else None

        res = {
            "method": name, "method_id": mid,
            "dataset": "sidd", "n_patches": len(ok),
            "patch_size": f"{W}x{H}", "domain": info["domain"],
            "gt_source": gt_source,
            "avg_time_s":   round(float(np.mean([r["time_s"] for r in ok])), 3),
            "total_time_s": round(elapsed, 1),
            "rgb_psnr":     round(avg("rgb_psnr"), 2),
            "rgb_ssim":     round(avg("rgb_ssim"), 4),
        }
        for key in ["raw_psnr", "raw_ssim", "lpips", "dists",
                     "isp_psnr", "isp_ssim", "isp_lpips", "isp_dists"]:
            v = avg(key)
            if v is not None:
                res[key] = round(v, 2 if "psnr" in key else 4)

        parts = [f"PSNR={res['rgb_psnr']:.2f}"]
        if "isp_psnr" in res:
            parts.append(f"ISP_PSNR={res['isp_psnr']:.2f}")
        parts.append(f"SSIM={res['rgb_ssim']:.4f}")
        if "lpips" in res:
            parts.append(f"LPIPS={res['lpips']:.4f}")
        if "dists" in res:
            parts.append(f"DISTS={res['dists']:.4f}")
        parts.append(f"{res['avg_time_s']:.3f}s/patch")
        print(f"  => {' | '.join(parts)}\n")

        all_results.append(res)
        with open(RESULTS / f"sidd_{mid}_results.json", "w") as f:
            json.dump(res, f, indent=2)

    if not all_results:
        print("No results."); return

    # --- Summary tables (separate for each group) ---
    pre_results  = [r for r in all_results if r["gt_source"] == "raw_calibrated"]
    post_results = [r for r in all_results if r["gt_source"] == "srgb"]

    def print_table(title, results, show_isp=False):
        if not results:
            return
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")
        cols = ["Method", "PSNR", "SSIM", "LPIPS", "DISTS"]
        if show_isp:
            cols += ["ISP_PSNR", "ISP_LPIPS"]
        cols.append("t/patch")
        hdr = f"{'Method':<22} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'DISTS':>7}"
        if show_isp:
            hdr += f" {'ISP_PSNR':>9} {'ISP_LPIPS':>9}"
        hdr += f" {'t/patch':>8}"
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            lp = f"{r['lpips']:.4f}" if "lpips" in r else "  N/A  "
            di = f"{r['dists']:.4f}" if "dists" in r else "  N/A  "
            line = (f"{r['method']:<22} {r['rgb_psnr']:>7.2f} {r['rgb_ssim']:>7.4f} "
                    f"{lp:>7} {di:>7}")
            if show_isp:
                ip = f"{r['isp_psnr']:.2f}" if "isp_psnr" in r else "  N/A  "
                il = f"{r['isp_lpips']:.4f}" if "isp_lpips" in r else "  N/A  "
                line += f" {ip:>9} {il:>9}"
            line += f" {r['avg_time_s']:>7.3f}s"
            print(line)

    print_table("Pre-demosaic (GT② = calibrated pipeline)", pre_results, show_isp=True)
    print_table("After-demosaic (GT = ISP sRGB)", post_results, show_isp=False)

    # Grid image
    print("\nBuilding comparison grid image...", end=" ", flush=True)
    grid_methods = [m for m in active if m in method_den_map]
    grid = build_grid_image(gt_raw, noisy_raw, method_den_map, grid_methods)
    grid_path = OUTDIR / "sidd_comparison_grid.png"
    imsave(str(grid_path), grid)
    print(f"OK → {grid_path}")

    # Save per-patch PNG images
    print("Saving per-method PNG samples...", end=" ", flush=True)
    sample_indices = list(range(0, min(N, 320), 32))  # first patch of each scene
    for mid in grid_methods:
        mid_dir = OUTDIR / mid
        mid_dir.mkdir(exist_ok=True)
        for si in sample_indices:
            r = method_den_map[mid][si]
            if r is None:
                continue
            den_u8 = (np.clip(r["den_srgb"], 0, 1) * 255).astype(np.uint8)
            imsave(str(mid_dir / f"patch_{si:04d}_den.png"), den_u8)
            # GT and noisy (save once)
            gt_dir = OUTDIR / "__gt__"
            gt_dir.mkdir(exist_ok=True)
            noisy_dir = OUTDIR / "__noisy__"
            noisy_dir.mkdir(exist_ok=True)
            gt_path = gt_dir / f"patch_{si:04d}_gt.png"
            if not gt_path.exists():
                gt_u8 = (np.clip(gt_srgb_all[si], 0, 1) * 255).astype(np.uint8)
                imsave(str(gt_path), gt_u8)
                noisy_u8 = (np.clip(noisy_srgb_all[si], 0, 1) * 255).astype(np.uint8)
                imsave(str(noisy_dir / f"patch_{si:04d}_noisy.png"), noisy_u8)
    print("OK")

    combined = {
        "dataset": "sidd", "n_patches": int(N), "patch_size": f"{W}x{H}",
        "noisy_raw_psnr": round(float(np.mean(noisy_ref_psnrs)), 2),
        "methods": all_results,
    }
    combined_path = RESULTS / "sidd_combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nCombined JSON → {combined_path}")
    print(f"Grid image    → {grid_path}")


def main():
    parser = argparse.ArgumentParser(description="SIDD benchmark: GALOSH vs baselines")
    all_ids = list(DISPLAY_NAMES.keys())
    # Default: all pre-demosaic + key after-demosaic methods
    default_methods = [
        # Pre-demosaic
        "galosh_gpu", "bm3dcfa_oracle", "nlm_cfa_blind",
        # After-demosaic
        "galosh_yuv_gpu", "nafnet", "bm3d_srgb", "nlm_srgb",
    ]
    parser.add_argument("--methods", "-m", nargs="+", default=default_methods,
                        choices=all_ids + ["all",
                                           "pre",   # all pre-demosaic
                                           "post"], # all after-demosaic
                        help="Methods to benchmark")
    parser.add_argument("--no-perceptual", action="store_true",
                        help="Skip LPIPS/DISTS (much faster)")
    parser.add_argument("--workers", "-j", type=int,
                        default=min(os.cpu_count() or 4, 8),
                        help="Parallel workers (default: min(cpu_count, 8))")
    parser.add_argument("--scenes", "-s", type=int, default=0,
                        help="Limit to first N scenes (0 = all 40 scenes)")
    args = parser.parse_args()

    PRE_IDS  = ["galosh_gpu", "bm3dcfa_blind", "bm3dcfa_oracle",
                "nlm_cfa_blind", "nlm_cfa_oracle"]
    POST_IDS = ["galosh_yuv_gpu", "nafnet", "dncnn_b_srgb", "drunet_srgb",
                "bm3d_srgb", "nlm_srgb"]

    if "all" in args.methods:
        methods = all_ids
    elif "pre" in args.methods:
        methods = PRE_IDS
    elif "post" in args.methods:
        methods = POST_IDS
    else:
        methods = args.methods

    # Preload DL models before threads (not thread-safe to load in parallel)
    needs_dncnn  = "dncnn_b_srgb" in methods
    needs_drunet = "drunet_srgb" in methods
    needs_nafnet = "nafnet" in methods
    for need, loader, label, mids in [
        (needs_dncnn,  get_dncnn,  "DnCNN-B",        ("dncnn_b_srgb",)),
        (needs_drunet, get_drunet, "DRUNet",          ("drunet_srgb",)),
        (needs_nafnet, get_nafnet, "NAFNet-SIDD-w64", ("nafnet",)),
    ]:
        if need:
            print(f"Loading {label}...", end=" ", flush=True)
            try:
                loader(); print("OK")
            except Exception as e:
                print(f"FAILED ({e}) -- skip")
                methods = [m for m in methods if m not in mids]

    run_benchmark(methods, not args.no_perceptual, args.workers, args.scenes)


if __name__ == "__main__":
    main()
