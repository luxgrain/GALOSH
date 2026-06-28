#!/usr/bin/env python3
"""
Raw Bayer Denoising Evaluation Framework
=========================================
Compares: BM3D-perchannel, CBM3D, our GAT+BM3D+L/C method
Datasets: SIDD, Kodak(synthetic), RawNIND
Metrics:  PSNR, SSIM on raw; CPSNR, SSIM, LPIPS on sRGB
"""
import os, sys, json, time, glob
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

# ─── Paths ──────────────────────────────────────────────
EVAL_ROOT = Path(os.path.expanduser(r"~\denoise_eval"))
DATASET_DIR = EVAL_ROOT / "datasets"
RESULT_DIR  = EVAL_ROOT / "results"
COMPARE_DIR = EVAL_ROOT / "comparison_images"

# ─── GAT (Generalized Anscombe Transform) ──────────────
def gat_forward(x: np.ndarray, alpha: float, sigma_sq: float) -> np.ndarray:
    """Variance-stabilizing transform for Poisson-Gaussian noise."""
    return (2.0 / alpha) * np.sqrt(np.maximum(
        alpha * np.maximum(x, 0.0) + 0.375 * alpha * alpha + sigma_sq, 0.0))

def gat_inverse(D: np.ndarray, alpha: float, sigma_sq: float) -> np.ndarray:
    """Exact unbiased inverse of GAT (Mäkitalo & Foi)."""
    D = np.maximum(D, 1e-8)
    D_inv = 1.0 / D
    sqrt3 = 1.2247448713916
    y = (0.25 * D * D
         + 0.25 * sqrt3 * D_inv
         - 11.0/8.0 * D_inv * D_inv
         + 5.0/8.0 * sqrt3 * D_inv**3
         - 1.0/8.0)
    return np.maximum(alpha * y - sigma_sq / alpha, 0.0)

def estimate_gat_sigma(data: np.ndarray) -> float:
    """Estimate σ of GAT-transformed data using Laplacian MAD."""
    h, w = data.shape
    # Laplacian along rows
    lap = data[:, :-2] - 2.0 * data[:, 1:-1] + data[:, 2:]
    abs_lap = np.abs(lap.ravel())
    # subsample if too large
    if len(abs_lap) > 200000:
        idx = np.random.choice(len(abs_lap), 200000, replace=False)
        abs_lap = abs_lap[idx]
    mad = np.median(abs_lap)
    sigma = mad / 1.6521
    return max(sigma, 0.01)

# ─── Bayer channel extraction/insertion ─────────────────
def extract_bayer_channels(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract R, Gr, Gb, B from RGGB Bayer pattern."""
    R  = raw[0::2, 0::2].astype(np.float32)
    Gr = raw[0::2, 1::2].astype(np.float32)
    Gb = raw[1::2, 0::2].astype(np.float32)
    B  = raw[1::2, 1::2].astype(np.float32)
    return R, Gr, Gb, B

def insert_bayer_channels(R, Gr, Gb, B, shape) -> np.ndarray:
    """Reconstruct Bayer mosaic from 4 channels."""
    out = np.zeros(shape, dtype=np.float32)
    out[0::2, 0::2] = R
    out[0::2, 1::2] = Gr
    out[1::2, 0::2] = Gb
    out[1::2, 1::2] = B
    return out

# ─── L/C Transform ─────────────────────────────────────
SIGMA_L  = 0.5
SIGMA_C1 = 0.866
SIGMA_C2 = 0.866
SIGMA_C3 = 0.707

def lc_forward(R, Gr, Gb, B):
    """Forward L/C transform on 4 Bayer channels."""
    L  = (R + Gr + Gb + B) * 0.25
    C1 = R - L
    C2 = B - L
    C3 = (Gr - Gb) * 0.5
    return L, C1, C2, C3

def lc_inverse(L, C1, C2, C3):
    """Inverse L/C transform."""
    R  = L + C1
    Gr = L - 0.5 * C1 - 0.5 * C2 + C3
    Gb = L - 0.5 * C1 - 0.5 * C2 - C3
    B  = L + C2
    # NaN guard
    for arr, fallback in [(R, L), (Gr, L), (Gb, L), (B, L)]:
        mask = ~np.isfinite(arr)
        if mask.any():
            arr[mask] = fallback[mask]
    return R, Gr, Gb, B

# ─── Our method: GAT + BM3D + L/C separation ───────────
def denoise_ours(raw: np.ndarray, strength: float = 1.0,
                 luma_strength: float = 0.7, chroma_strength: float = 1.5,
                 alpha: float = 1.0, sigma_sq: float = 0.0,
                 noise_sigma: float = 0.0, use_gat: bool = True) -> np.ndarray:
    """
    Our GAT + BM3D + L/C separation + cross-channel matching.
    Uses bm3d pip package for the core BM3D filtering.

    For synthetic Gaussian noise: set use_gat=False, noise_sigma=actual_sigma.
    For real Poisson-Gaussian noise: set use_gat=True, alpha/sigma_sq from camera.
    """
    import bm3d as bm3d_lib

    R, Gr, Gb, B = extract_bayer_channels(raw)
    h, w = R.shape

    if use_gat:
        # Phase 1: GAT per channel
        ch_gat = [gat_forward(ch, alpha, sigma_sq) for ch in [R, Gr, Gb, B]]
        # After GAT, noise σ ≈ 1 per channel
        base_sigma = 1.0
    else:
        # Skip GAT for synthetic Gaussian noise
        ch_gat = [R, Gr, Gb, B]
        base_sigma = noise_sigma if noise_sigma > 0 else estimate_gat_sigma(R)

    # Phase 2: L/C forward
    L, C1, C2, C3 = lc_forward(*ch_gat)

    # Phase 3: Normalize to σ=1 regime
    # After L/C, noise σ on each component is base_sigma * sigma_component
    # Dividing by sigma_component normalizes to σ = base_sigma
    L_norm  = L  / SIGMA_L
    C1_norm = C1 / SIGMA_C1
    C2_norm = C2 / SIGMA_C2
    C3_norm = C3 / SIGMA_C3

    # BM3D sigma: the noise σ in normalized space is base_sigma
    # Apply strength controls
    sigma_l = base_sigma * strength * luma_strength
    sigma_c = base_sigma * strength * chroma_strength

    # BM3D on all components
    L_denoised  = bm3d_lib.bm3d(L_norm, sigma_psd=sigma_l)
    C1_denoised = bm3d_lib.bm3d(C1_norm, sigma_psd=sigma_c)
    C2_denoised = bm3d_lib.bm3d(C2_norm, sigma_psd=sigma_c)
    C3_denoised = bm3d_lib.bm3d(C3_norm, sigma_psd=sigma_c)

    # Denormalize
    L_out  = L_denoised  * SIGMA_L
    C1_out = C1_denoised * SIGMA_C1
    C2_out = C2_denoised * SIGMA_C2
    C3_out = C3_denoised * SIGMA_C3

    # Phase 4: Inverse L/C
    R_out, Gr_out, Gb_out, B_out = lc_inverse(L_out, C1_out, C2_out, C3_out)

    if use_gat:
        # Phase 5: Inverse GAT per channel
        result_channels = [gat_inverse(ch, alpha, sigma_sq)
                          for ch in [R_out, Gr_out, Gb_out, B_out]]
    else:
        result_channels = [np.clip(ch, 0, None)
                          for ch in [R_out, Gr_out, Gb_out, B_out]]

    return insert_bayer_channels(*result_channels, raw.shape)

# ─── Comparison: BM3D per-channel ───────────────────────
def denoise_bm3d_perchannel(raw: np.ndarray, noise_sigma: float = 0.0,
                             use_gat: bool = True,
                             alpha: float = 1.0, sigma_sq: float = 0.0) -> np.ndarray:
    """Naive baseline: BM3D independently on each Bayer channel."""
    import bm3d as bm3d_lib

    R, Gr, Gb, B = extract_bayer_channels(raw)
    channels = [R, Gr, Gb, B]

    if use_gat:
        ch_proc = [gat_forward(ch, alpha, sigma_sq) for ch in channels]
    else:
        ch_proc = list(channels)

    # Denoise each channel independently
    denoised = []
    for ch in ch_proc:
        sig = noise_sigma if noise_sigma > 0 else estimate_gat_sigma(ch)
        d = bm3d_lib.bm3d(ch, sigma_psd=sig)
        denoised.append(d)

    if use_gat:
        result = [gat_inverse(d, alpha, sigma_sq) for d in denoised]
    else:
        result = [np.clip(d, 0, None) for d in denoised]
    return insert_bayer_channels(*result, raw.shape)

# ─── Comparison: CBM3D (demosaic first) ─────────────────
def denoise_cbm3d(raw: np.ndarray, sigma: float = 25.0/255.0) -> np.ndarray:
    """CBM3D: demosaic first, then color BM3D, then re-mosaic."""
    import bm3d as bm3d_lib

    # Simple bilinear demosaic for input
    rgb = simple_demosaic(raw)

    # Normalize to [0, 1] if needed
    maxval = rgb.max()
    if maxval > 1.0:
        rgb = rgb / maxval

    # Color BM3D
    denoised_rgb = bm3d_lib.bm3d_rgb(rgb, sigma_psd=sigma)

    if maxval > 1.0:
        denoised_rgb = denoised_rgb * maxval

    # Re-mosaic (take back to Bayer)
    out = np.zeros_like(raw)
    out[0::2, 0::2] = denoised_rgb[0::2, 0::2, 0]  # R
    out[0::2, 1::2] = denoised_rgb[0::2, 1::2, 1]  # G
    out[1::2, 0::2] = denoised_rgb[1::2, 0::2, 1]  # G
    out[1::2, 1::2] = denoised_rgb[1::2, 1::2, 2]  # B
    return out

# ─── Demosaicing helpers ────────────────────────────────
def simple_demosaic(raw: np.ndarray) -> np.ndarray:
    """Bilinear demosaic for CBM3D input (RGGB)."""
    from scipy.ndimage import convolve
    h, w = raw.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Create channel masks
    R_mask = np.zeros((h, w), dtype=np.float32)
    G_mask = np.zeros((h, w), dtype=np.float32)
    B_mask = np.zeros((h, w), dtype=np.float32)
    R_mask[0::2, 0::2] = 1
    G_mask[0::2, 1::2] = 1
    G_mask[1::2, 0::2] = 1
    B_mask[1::2, 1::2] = 1

    # Interpolation kernels
    kernel_rb = np.array([[0.25, 0.5, 0.25],
                          [0.5,  1.0, 0.5],
                          [0.25, 0.5, 0.25]], dtype=np.float32)
    kernel_g = np.array([[0.0, 0.25, 0.0],
                         [0.25, 1.0, 0.25],
                         [0.0, 0.25, 0.0]], dtype=np.float32)

    rgb[:,:,0] = convolve(raw * R_mask, kernel_rb, mode='mirror')
    rgb[:,:,1] = convolve(raw * G_mask, kernel_g, mode='mirror')
    rgb[:,:,2] = convolve(raw * B_mask, kernel_rb, mode='mirror')
    return rgb

def demosaic_for_display(raw: np.ndarray, method: str = 'bilinear') -> np.ndarray:
    """Demosaic raw Bayer for sRGB metric evaluation.
    Uses colour-demosaicing if available (amaze-like), else bilinear."""
    try:
        from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007 as demosaic_menon
        return demosaic_menon(raw, pattern='RGGB').astype(np.float32)
    except ImportError:
        try:
            from colour_demosaicing import demosaicing_CFA_Bayer_DDFAPD as demosaic_ddfapd
            return demosaic_ddfapd(raw, pattern='RGGB').astype(np.float32)
        except ImportError:
            return simple_demosaic(raw)

def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Linear RGB to sRGB gamma."""
    img = np.clip(img, 0, 1)
    return np.where(img <= 0.0031308,
                    12.92 * img,
                    1.055 * np.power(img, 1.0/2.4) - 0.055)

# ─── Metrics ────────────────────────────────────────────
def psnr(clean: np.ndarray, denoised: np.ndarray, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((clean.astype(np.float64) - denoised.astype(np.float64))**2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(max_val**2 / mse)

def ssim(clean: np.ndarray, denoised: np.ndarray, max_val: float = 1.0) -> float:
    """Structural Similarity Index."""
    from skimage.metrics import structural_similarity
    return structural_similarity(clean, denoised, data_range=max_val)

def psnr_4ch(clean_raw: np.ndarray, denoised_raw: np.ndarray, max_val: float = 1.0) -> float:
    """PSNR computed on raw RGGB (all 4 channels)."""
    return psnr(clean_raw, denoised_raw, max_val)

def ssim_4ch(clean_raw: np.ndarray, denoised_raw: np.ndarray, max_val: float = 1.0) -> float:
    """Average SSIM across 4 Bayer channels."""
    c_R, c_Gr, c_Gb, c_B = extract_bayer_channels(clean_raw)
    d_R, d_Gr, d_Gb, d_B = extract_bayer_channels(denoised_raw)
    vals = []
    for c, d in [(c_R, d_R), (c_Gr, d_Gr), (c_Gb, d_Gb), (c_B, d_B)]:
        vals.append(ssim(c, d, max_val))
    return np.mean(vals)

def compute_srgb_metrics(clean_raw, denoised_raw, max_val=1.0):
    """Compute CPSNR and SSIM on sRGB-converted images."""
    clean_rgb = demosaic_for_display(clean_raw)
    denoised_rgb = demosaic_for_display(denoised_raw)

    # Normalize to [0,1]
    if max_val != 1.0:
        clean_rgb = clean_rgb / max_val
        denoised_rgb = denoised_rgb / max_val

    clean_rgb = np.clip(clean_rgb, 0, 1)
    denoised_rgb = np.clip(denoised_rgb, 0, 1)

    # sRGB gamma
    clean_srgb = linear_to_srgb(clean_rgb)
    denoised_srgb = linear_to_srgb(denoised_rgb)

    cpsnr = psnr(clean_srgb, denoised_srgb)
    cssim = ssim(clean_srgb[:,:,0], denoised_srgb[:,:,0])  # luma-ish

    return cpsnr, cssim

def compute_lpips_score(clean_raw, denoised_raw, max_val=1.0):
    """LPIPS on sRGB images (requires torch)."""
    try:
        import torch
        import lpips
        loss_fn = lpips.LPIPS(net='alex')

        clean_rgb = demosaic_for_display(clean_raw)
        denoised_rgb = demosaic_for_display(denoised_raw)

        if max_val != 1.0:
            clean_rgb = clean_rgb / max_val
            denoised_rgb = denoised_rgb / max_val

        clean_srgb = linear_to_srgb(np.clip(clean_rgb, 0, 1))
        denoised_srgb = linear_to_srgb(np.clip(denoised_rgb, 0, 1))

        # LPIPS expects [B, C, H, W] in [-1, 1]
        def to_tensor(img):
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            return t * 2.0 - 1.0

        with torch.no_grad():
            score = loss_fn(to_tensor(clean_srgb), to_tensor(denoised_srgb))
        return score.item()
    except Exception as e:
        print(f"  LPIPS failed: {e}")
        return -1.0

# ─── Noise parameter estimation for real raw data ──────
def estimate_noise_params(raw: np.ndarray) -> Tuple[float, float]:
    """Estimate alpha (gain) and sigma_sq (read noise variance) from raw.
    Simple: assume alpha=1.0 and estimate from dark regions."""
    # For normalized [0,1] data, alpha=1.0 is reasonable
    # sigma_sq estimated from darkest 5% of pixels
    flat = raw.ravel()
    dark_thresh = np.percentile(flat[flat > 0], 5)
    dark_pixels = flat[(flat > 0) & (flat < dark_thresh)]
    if len(dark_pixels) < 100:
        return 1.0, 0.0
    sigma_sq = np.var(dark_pixels)
    return 1.0, sigma_sq

# ─── Synthetic noise for Kodak/McMaster ─────────────────
def synthesize_bayer_noise(clean_rgb: np.ndarray, iso: int = 1600
                           ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Synthesize noisy raw Bayer with realistic Poisson-Gaussian noise model.
    Models: y = Poisson(x / alpha) * alpha + N(0, sigma_read^2)

    ISO-based noise parameters (approximate sensor model):
      alpha (analog gain) = ISO / 100
      sigma_read (read noise std in DN) = 3.0 * (ISO / 100) / 65535 * scale

    Returns (noisy_raw, clean_raw, alpha, sigma_sq) all in [0,1].
    """
    h, w = clean_rgb.shape[:2]
    h = h - (h % 2)
    w = w - (w % 2)
    clean_rgb = clean_rgb[:h, :w].astype(np.float32)

    if clean_rgb.max() > 1.0:
        clean_rgb = clean_rgb / 255.0

    # Inverse sRGB gamma to linear
    linear = np.where(clean_rgb <= 0.04045,
                      clean_rgb / 12.92,
                      ((clean_rgb + 0.055) / 1.055) ** 2.4)

    # Create Bayer mosaic from linear RGB
    clean_raw = np.zeros((h, w), dtype=np.float32)
    clean_raw[0::2, 0::2] = linear[0::2, 0::2, 0]  # R
    clean_raw[0::2, 1::2] = linear[0::2, 1::2, 1]  # G
    clean_raw[1::2, 0::2] = linear[1::2, 0::2, 1]  # G
    clean_raw[1::2, 1::2] = linear[1::2, 1::2, 2]  # B

    # Realistic Poisson-Gaussian noise model
    # alpha = gain (photons-to-DN conversion), proportional to ISO
    # For data in [0,1], alpha controls shot noise severity
    gain = iso / 100.0
    alpha = 0.001 * gain  # shot noise coefficient: Var_shot = alpha * x
    sigma_read = 0.002 * gain  # read noise std, scales with ISO
    sigma_sq = sigma_read ** 2

    # Poisson component: generate Poisson with rate x/alpha, then scale by alpha
    # This gives E[y] = x, Var[y] = alpha * x
    rate = np.maximum(clean_raw / max(alpha, 1e-10), 0.0)
    poisson_out = np.random.poisson(rate).astype(np.float32) * alpha

    # Read noise (Gaussian)
    read_noise = np.random.randn(h, w).astype(np.float32) * sigma_read

    noisy_raw = np.clip(poisson_out + read_noise, 0, 1)

    return noisy_raw, clean_raw, alpha, sigma_sq

# ─── Dataset Loaders ────────────────────────────────────
def load_kodak_images():
    """Load Kodak 24 test images."""
    kodak_dir = DATASET_DIR / "kodak"
    images = []
    for f in sorted(kodak_dir.glob("kodim*.png")):
        from skimage.io import imread
        img = imread(str(f)).astype(np.float32) / 255.0
        images.append((f.stem, img))
    return images

def load_sidd_benchmark():
    """Load SIDD benchmark patches (noisy + GT)."""
    sidd_dir = DATASET_DIR / "sidd"
    # SIDD benchmark .mat format
    results = []

    # Try .mat files first
    mat_noisy = sidd_dir / "BenchmarkNoisyBlocksRaw.mat"
    mat_gt = sidd_dir / "BenchmarkGtBlocksRaw.mat"

    if mat_noisy.exists() and mat_gt.exists():
        from scipy.io import loadmat
        print("Loading SIDD Raw benchmark...")
        noisy_data = loadmat(str(mat_noisy))
        gt_data = loadmat(str(mat_gt))

        noisy_blocks = noisy_data.get('BenchmarkNoisyBlocksRaw',
                       noisy_data.get('ValidationNoisyBlocksRaw'))
        gt_blocks = gt_data.get('BenchmarkGtBlocksRaw',
                    gt_data.get('ValidationGtBlocksRaw'))

        if noisy_blocks is not None and gt_blocks is not None:
            n_images, n_blocks = noisy_blocks.shape[:2]
            for i in range(n_images):
                for j in range(n_blocks):
                    noisy = noisy_blocks[i, j].astype(np.float32)
                    gt = gt_blocks[i, j].astype(np.float32)
                    results.append((f"sidd_{i:03d}_{j:02d}", noisy, gt))
    else:
        # Try synthetic numpy arrays (created by download_datasets.py)
        noisy_f = sidd_dir / "synthetic_noisy_raw.npy"
        gt_f = sidd_dir / "synthetic_gt_raw.npy"
        if noisy_f.exists() and gt_f.exists():
            noisy_arr = np.load(str(noisy_f))
            gt_arr = np.load(str(gt_f))
            for i in range(len(noisy_arr)):
                results.append((f"synth_{i:04d}", noisy_arr[i], gt_arr[i]))
        else:
            # Try individual numpy files
            for nf in sorted(sidd_dir.glob("*noisy*.npy")):
                gf = sidd_dir / nf.name.replace("noisy", "gt")
                if gf.exists():
                    noisy = np.load(str(nf)).astype(np.float32)
                    gt = np.load(str(gf)).astype(np.float32)
                    results.append((nf.stem, noisy, gt))

    print(f"  Loaded {len(results)} SIDD patches")
    return results

def load_rawnind_high_iso(min_iso: int = 3200, max_images: int = 50):
    """Load RawNIND high-ISO image pairs."""
    rawnind_dir = DATASET_DIR / "rawnind"
    results = []

    # Look for metadata/index file
    index_file = rawnind_dir / "index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        # Filter by ISO
        for entry in index:
            if entry.get('iso', 0) >= min_iso:
                noisy_path = rawnind_dir / entry['noisy']
                clean_path = rawnind_dir / entry['clean']
                if noisy_path.exists() and clean_path.exists():
                    results.append((entry, str(noisy_path), str(clean_path)))
                    if len(results) >= max_images:
                        break
    else:
        # Try to find DNG/ARW pairs
        import rawpy
        for noisy_f in sorted(rawnind_dir.glob("**/*noisy*")):
            # Match with clean counterpart
            clean_f = str(noisy_f).replace("noisy", "clean").replace("Noisy", "Clean")
            if os.path.exists(clean_f):
                results.append((noisy_f.stem, str(noisy_f), clean_f))
                if len(results) >= max_images:
                    break

    print(f"  Found {len(results)} RawNIND high-ISO pairs")
    return results

def load_raw_file(path: str) -> Tuple[np.ndarray, dict]:
    """Load a raw file and return (raw_bayer, metadata)."""
    import rawpy
    with rawpy.imread(path) as raw:
        bayer = raw.raw_image_visible.astype(np.float32).copy()
        black = raw.black_level_per_channel
        white = raw.white_level
        pattern = raw.raw_pattern.tolist()
        meta = {
            'black_levels': black,
            'white_level': white,
            'pattern': pattern,
            'iso': raw.camera_whitebalance if hasattr(raw, 'camera_whitebalance') else None,
        }
        # Normalize to [0, 1]
        bl = np.mean(black)
        bayer = (bayer - bl) / (white - bl)
        bayer = np.clip(bayer, 0, 1)
    return bayer, meta

# ─── Main evaluation runner ─────────────────────────────
@dataclass
class EvalResult:
    method: str
    dataset: str
    image_name: str
    psnr_raw: float = 0.0
    ssim_raw: float = 0.0
    cpsnr_srgb: float = 0.0
    ssim_srgb: float = 0.0
    lpips: float = -1.0
    time_sec: float = 0.0
    iso: int = 0
    sigma: float = 0.0

def run_method(method_name: str, method_fn, noisy_raw: np.ndarray,
               clean_raw: np.ndarray, max_val: float = 1.0,
               dataset: str = '', image_name: str = '',
               compute_srgb: bool = True, compute_lpips: bool = False) -> EvalResult:
    """Run a single method and compute all metrics."""
    t0 = time.time()
    denoised = method_fn(noisy_raw)
    elapsed = time.time() - t0

    denoised = np.clip(denoised, 0, max_val).astype(np.float32)

    result = EvalResult(
        method=method_name,
        dataset=dataset,
        image_name=image_name,
        psnr_raw=psnr_4ch(clean_raw, denoised, max_val),
        ssim_raw=ssim_4ch(clean_raw, denoised, max_val),
        time_sec=elapsed,
    )

    # sRGB metrics (slower, optional)
    if compute_srgb:
        result.cpsnr_srgb, result.ssim_srgb = compute_srgb_metrics(
            clean_raw, denoised, max_val)

    # LPIPS (slowest, optional)
    if compute_lpips:
        result.lpips = compute_lpips_score(clean_raw, denoised, max_val)

    return result, denoised

# ─── Evaluation pipelines ──────────────────────────────
def eval_kodak_synthetic(iso_levels=[800, 1600, 3200, 6400, 12800]):
    """Kodak + synthetic Poisson-Gaussian noise evaluation."""
    print("="*60)
    print("KODAK SYNTHETIC POISSON-GAUSSIAN NOISE EVALUATION")
    print("="*60)

    images = load_kodak_images()
    if not images:
        print("  No Kodak images found. Skipping.")
        return []

    all_results = []
    for iso in iso_levels:
        print(f"\n--- ISO {iso} ---")
        method_names = ['BM3D-perchannel', 'CBM3D', 'Ours (GAT+BM3D+L/C)']
        iso_results = {m: [] for m in method_names}

        for img_name, img_rgb in images[:6]:  # first 6 for speed
            noisy_raw, clean_raw, alpha, sigma_sq = synthesize_bayer_noise(img_rgb, iso)

            # Build method functions with params baked in
            def _make_methods(_alpha, _sigma_sq):
                return [
                    ('BM3D-perchannel',
                     lambda raw: denoise_bm3d_perchannel(
                         raw, use_gat=True, alpha=_alpha, sigma_sq=_sigma_sq)),
                    ('CBM3D',
                     lambda raw: denoise_cbm3d(
                         raw, sigma=estimate_gat_sigma(
                             extract_bayer_channels(raw)[0]))),
                    ('Ours (GAT+BM3D+L/C)',
                     lambda raw: denoise_ours(
                         raw, strength=1.0, luma_strength=0.7, chroma_strength=1.5,
                         alpha=_alpha, sigma_sq=_sigma_sq, use_gat=True)),
                ]
            methods = _make_methods(alpha, sigma_sq)

            for method_name, method_fn in methods:
                try:
                    result, denoised = run_method(
                        method_name, method_fn, noisy_raw, clean_raw, 1.0,
                        dataset='kodak', image_name=img_name,
                        compute_srgb=True, compute_lpips=False)
                    result.iso = iso
                    iso_results[method_name].append(result)
                    all_results.append(result)
                    print(f"  {img_name} | {method_name:25s} | "
                          f"PSNR={result.psnr_raw:.2f} SSIM={result.ssim_raw:.4f} "
                          f"CPSNR={result.cpsnr_srgb:.2f} | {result.time_sec:.1f}s")
                except Exception as e:
                    print(f"  {img_name} | {method_name:25s} | ERROR: {e}")
                    import traceback; traceback.print_exc()

        # Print averages
        print(f"\n  Average at ISO {iso}:")
        for method_name in method_names:
            results = iso_results[method_name]
            if results:
                avg_psnr = np.mean([r.psnr_raw for r in results])
                avg_ssim = np.mean([r.ssim_raw for r in results])
                avg_cpsnr = np.mean([r.cpsnr_srgb for r in results])
                avg_time = np.mean([r.time_sec for r in results])
                print(f"    {method_name:25s} | PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} "
                      f"CPSNR={avg_cpsnr:.2f} | {avg_time:.1f}s")

    return all_results

def eval_sidd():
    """SIDD Raw benchmark evaluation."""
    print("="*60)
    print("SIDD RAW BENCHMARK EVALUATION")
    print("="*60)

    patches = load_sidd_benchmark()
    if not patches:
        print("  No SIDD data found. Skipping.")
        return []

    # Normalize based on data range
    sample = patches[0][1]
    if sample.max() > 1.0:
        max_val = 65535.0 if sample.max() > 4095 else 4095.0
    else:
        max_val = 1.0

    all_results = []
    methods = [
        ('BM3D-perchannel',
         lambda raw: denoise_bm3d_perchannel(raw, use_gat=True)),
        ('Ours (GAT+BM3D+L/C)',
         lambda raw: denoise_ours(raw, strength=1.0, luma_strength=0.7,
                                   chroma_strength=1.5, use_gat=True)),
    ]

    for method_name, method_fn in methods:
        results = []
        for name, noisy, gt in patches[:100]:  # first 100 for speed
            try:
                noisy_norm = noisy / max_val if max_val != 1.0 else noisy
                gt_norm = gt / max_val if max_val != 1.0 else gt

                result, _ = run_method(
                    method_name, method_fn, noisy_norm, gt_norm, 1.0,
                    dataset='sidd', image_name=name,
                    compute_srgb=False, compute_lpips=False)
                results.append(result)
                all_results.append(result)
            except Exception as e:
                print(f"  {name} | {method_name} | ERROR: {e}")

        if results:
            avg_psnr = np.mean([r.psnr_raw for r in results])
            avg_ssim = np.mean([r.ssim_raw for r in results])
            print(f"  {method_name:25s} | PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} "
                  f"| n={len(results)}")

    return all_results

def eval_rawnind_high_iso():
    """RawNIND high-ISO evaluation."""
    print("="*60)
    print("RawNIND HIGH-ISO EVALUATION")
    print("="*60)

    pairs = load_rawnind_high_iso(min_iso=3200, max_images=30)
    if not pairs:
        print("  No RawNIND data found. Skipping.")
        return []

    all_results = []
    for entry, noisy_path, clean_path in pairs:
        try:
            noisy_raw, noisy_meta = load_raw_file(noisy_path)
            clean_raw, clean_meta = load_raw_file(clean_path)

            # Ensure same size
            h = min(noisy_raw.shape[0], clean_raw.shape[0])
            w = min(noisy_raw.shape[1], clean_raw.shape[1])
            h = h - (h % 2)
            w = w - (w % 2)
            noisy_raw = noisy_raw[:h, :w]
            clean_raw = clean_raw[:h, :w]

            iso = entry.get('iso', 0) if isinstance(entry, dict) else 0
            name = entry.get('name', str(entry)) if isinstance(entry, dict) else str(entry)

            for method_name, method_fn, kwargs in [
                ('BM3D-perchannel', denoise_bm3d_perchannel,
                 {'alpha': 1.0, 'sigma_sq': 0.0}),
                ('Ours (GAT+BM3D+L/C)', denoise_ours,
                 {'strength': 1.0, 'luma_strength': 0.7, 'chroma_strength': 1.5,
                  'alpha': 1.0, 'sigma_sq': 0.0}),
            ]:
                kwargs.update({'dataset': 'rawnind', 'image_name': name,
                               'compute_srgb': True, 'compute_lpips': False})
                result, denoised = run_method(method_name, method_fn,
                                               noisy_raw, clean_raw, 1.0, **kwargs)
                result.iso = iso
                all_results.append(result)
                print(f"  ISO {iso:6d} | {name:20s} | {method_name:25s} | "
                      f"PSNR={result.psnr_raw:.2f} | {result.time_sec:.1f}s")
        except Exception as e:
            print(f"  ERROR loading {noisy_path}: {e}")

    return all_results

# ─── Save comparison images ─────────────────────────────
def save_comparison_image(noisy_raw, clean_raw, denoised_dict, name, max_val=1.0):
    """Save visual comparison crops."""
    from skimage.io import imsave

    h, w = noisy_raw.shape
    # Find an interesting crop (high detail area)
    crop_size = min(256, h//2, w//2)
    # Use center crop
    cy, cx = h//2, w//2
    y0, x0 = cy - crop_size//2, cx - crop_size//2
    y0 = y0 - (y0 % 2)
    x0 = x0 - (x0 % 2)

    def raw_to_srgb_crop(raw, y0, x0, crop_size):
        crop = raw[y0:y0+crop_size, x0:x0+crop_size]
        rgb = demosaic_for_display(crop)
        if max_val != 1.0:
            rgb = rgb / max_val
        srgb = linear_to_srgb(np.clip(rgb, 0, 1))
        return (np.clip(srgb, 0, 1) * 255).astype(np.uint8)

    # Save individual crops
    out_dir = COMPARE_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    imsave(str(out_dir / "noisy.png"), raw_to_srgb_crop(noisy_raw, y0, x0, crop_size))
    imsave(str(out_dir / "clean.png"), raw_to_srgb_crop(clean_raw, y0, x0, crop_size))

    for method_name, denoised in denoised_dict.items():
        safe_name = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
        imsave(str(out_dir / f"{safe_name}.png"),
               raw_to_srgb_crop(denoised, y0, x0, crop_size))

    print(f"  Saved comparison images to {out_dir}")

# ─── Results summary ────────────────────────────────────
def save_results(all_results: List[EvalResult], filename: str = "results.json"):
    """Save all results to JSON."""
    data = []
    for r in all_results:
        data.append({
            'method': r.method,
            'dataset': r.dataset,
            'image': r.image_name,
            'psnr_raw': round(r.psnr_raw, 3),
            'ssim_raw': round(r.ssim_raw, 5),
            'cpsnr_srgb': round(r.cpsnr_srgb, 3),
            'ssim_srgb': round(r.ssim_srgb, 5),
            'lpips': round(r.lpips, 5),
            'time_sec': round(r.time_sec, 2),
            'iso': r.iso,
            'sigma': r.sigma,
        })

    out_path = RESULT_DIR / filename
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {out_path}")

def print_summary_table(results: List[EvalResult]):
    """Print LaTeX-ready summary table."""
    from collections import defaultdict

    # Group by dataset and method
    groups = defaultdict(lambda: defaultdict(list))
    for r in results:
        groups[r.dataset][r.method].append(r)

    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    for dataset, methods in groups.items():
        print(f"\n--- {dataset.upper()} ---")
        print(f"{'Method':30s} | {'PSNR':>7s} | {'SSIM':>7s} | {'CPSNR':>7s} | {'Time':>6s}")
        print("-"*70)
        for method, results in sorted(methods.items()):
            avg_psnr = np.mean([r.psnr_raw for r in results])
            avg_ssim = np.mean([r.ssim_raw for r in results])
            avg_cpsnr = np.mean([r.cpsnr_srgb for r in results])
            avg_time = np.mean([r.time_sec for r in results])
            print(f"{method:30s} | {avg_psnr:7.2f} | {avg_ssim:7.4f} | "
                  f"{avg_cpsnr:7.2f} | {avg_time:5.1f}s")

# ─── Main ───────────────────────────────────────────────
def main():
    print("Raw Bayer Denoising Evaluation Framework")
    print(f"Results dir: {RESULT_DIR}")
    print()

    all_results = []

    # 1. Kodak synthetic (Poisson-Gaussian noise, ISO sweep)
    try:
        kodak_results = eval_kodak_synthetic(iso_levels=[800, 3200, 12800])
        all_results.extend(kodak_results)
    except Exception as e:
        print(f"Kodak eval failed: {e}")
        import traceback; traceback.print_exc()

    # 2. SIDD
    try:
        sidd_results = eval_sidd()
        all_results.extend(sidd_results)
    except Exception as e:
        print(f"SIDD eval failed: {e}")
        import traceback; traceback.print_exc()

    # 3. RawNIND high ISO
    try:
        rawnind_results = eval_rawnind_high_iso()
        all_results.extend(rawnind_results)
    except Exception as e:
        print(f"RawNIND eval failed: {e}")
        import traceback; traceback.print_exc()

    # Save and summarize
    if all_results:
        save_results(all_results)
        print_summary_table(all_results)

    print("\nDone!")

if __name__ == "__main__":
    main()
