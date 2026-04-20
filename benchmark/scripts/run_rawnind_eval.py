#!/usr/bin/env python3
"""
RawNIND real camera data evaluation.
Reads Sony ARW files, extracts raw Bayer, runs C standalone denoiser,
computes PSNR/SSIM against ground truth.
"""

import numpy as np
import rawpy
import subprocess
import json
import os
import sys
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

STANDALONE = str(Path(__file__).parent.parent / "standalone" / "rawdenoise.exe")
DATASET_DIR = Path(__file__).parent.parent / "datasets" / "rawnind"
RESULTS_DIR = Path(__file__).parent.parent / "results"
COMPARISON_DIR = Path(__file__).parent.parent / "comparison_images_rawnind"

# MSYS2 path for OpenMP runtime
os.environ["PATH"] = "C:\\msys64\\ucrt64\\bin;" + os.environ.get("PATH", "")

# Crop size for evaluation (full image is ~4024x6024, too slow for BM3D)
# Use 2048x2048 center crop for good balance of speed and coverage
EVAL_CROP_SIZE = 2048

RESULTS_DIR.mkdir(exist_ok=True)
COMPARISON_DIR.mkdir(exist_ok=True)


def load_arw_raw(path):
    """Load ARW file, return raw Bayer as float32 [0,1] and CFA pattern."""
    with rawpy.imread(str(path)) as raw:
        # Get raw image (no processing)
        bayer = raw.raw_image_visible.copy().astype(np.float32)
        black = float(raw.black_level_per_channel[0])
        white = float(raw.white_level)
        pattern = raw.raw_pattern.copy()  # e.g. [[0,1],[3,2]] for RGGB

        # Normalize to [0,1]
        bayer = (bayer - black) / (white - black)
        bayer = np.clip(bayer, 0.0, 1.0)

        print(f"  Loaded {path.name}: {bayer.shape}, black={black}, white={white}")
        print(f"  CFA pattern: {pattern}")
        print(f"  Range: [{bayer.min():.4f}, {bayer.max():.4f}], mean={bayer.mean():.4f}")

    return bayer, pattern


def estimate_noise_params(noisy, gt):
    """Estimate Poisson-Gaussian noise parameters from noisy/GT pair.
    Model: noisy = poisson(gt/alpha)*alpha + N(0, sigma_read)
    Var(noisy) = alpha * gt + sigma_read^2
    Fit linear: var vs gt -> slope=alpha, intercept=sigma_read^2
    """
    # Sample random patches to estimate variance
    h, w = noisy.shape
    patch_size = 32
    n_patches = 500

    np.random.seed(42)
    gts = []
    variances = []

    for _ in range(n_patches):
        y = np.random.randint(0, h - patch_size)
        x = np.random.randint(0, w - patch_size)
        # Use same-color pixels only (every 2nd pixel in each direction)
        gt_patch = gt[y:y+patch_size:2, x:x+patch_size:2]
        noisy_patch = noisy[y:y+patch_size:2, x:x+patch_size:2]
        residual = noisy_patch - gt_patch

        mean_gt = gt_patch.mean()
        var_noise = residual.var()

        if mean_gt > 0.01 and mean_gt < 0.95:  # Avoid clipped regions
            gts.append(mean_gt)
            variances.append(var_noise)

    gts = np.array(gts)
    variances = np.array(variances)

    # Linear fit: var = alpha * gt + sigma_read^2
    A = np.vstack([gts, np.ones(len(gts))]).T
    result = np.linalg.lstsq(A, variances, rcond=None)
    alpha = max(result[0][0], 1e-6)
    sigma_sq = max(result[0][1], 1e-10)

    print(f"  Estimated noise params: alpha={alpha:.6f}, sigma_sq={sigma_sq:.8f}")
    print(f"  (sigma_read={np.sqrt(sigma_sq):.6f})")

    return alpha, sigma_sq


def ensure_rggb(bayer, pattern):
    """Rearrange Bayer so it's RGGB (top-left = R)."""
    # rawpy pattern: 0=R, 1=G, 2=B, 3=G
    # Find where R is
    r_pos = np.argwhere(pattern == 0)
    if len(r_pos) == 0:
        print("  WARNING: No R channel found in CFA pattern, assuming RGGB")
        return bayer

    r_row, r_col = r_pos[0]

    if r_row == 0 and r_col == 0:
        return bayer  # Already RGGB

    # Shift to put R at (0,0)
    shifted = bayer[r_row:, r_col:]
    # Make even dimensions
    h, w = shifted.shape
    h -= h % 2
    w -= w % 2
    print(f"  Shifted CFA by ({r_row},{r_col}) to RGGB, size={h}x{w}")
    return shifted[:h, :w]


def psnr_raw(clean, denoised):
    """PSNR on raw Bayer data."""
    mse = np.mean((clean.astype(np.float64) - denoised.astype(np.float64))**2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def ssim_4ch(clean, denoised):
    """SSIM computed per Bayer channel then averaged."""
    vals = []
    for dy in range(2):
        for dx in range(2):
            c = clean[dy::2, dx::2]
            d = denoised[dy::2, dx::2]
            vals.append(ssim(c, d, data_range=1.0))
    return np.mean(vals)


def run_denoiser(input_bin, output_bin, w, h, method, alpha, sigma_sq,
                 strength=1.0, luma_str=0.7, chroma_str=1.5, iterations=1):
    """Run C standalone denoiser."""
    cmd = [
        STANDALONE, input_bin, output_bin, str(w), str(h),
        method, str(strength), str(luma_str), str(chroma_str),
        str(alpha), str(sigma_sq), str(iterations)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr}")
    return r


def demosaic_simple(bayer):
    """Simple bilinear demosaicing for visualization."""
    h, w = bayer.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    # RGGB
    rgb[0::2, 0::2, 0] = bayer[0::2, 0::2]  # R
    rgb[0::2, 1::2, 1] = bayer[0::2, 1::2]  # Gr
    rgb[1::2, 0::2, 1] = bayer[1::2, 0::2]  # Gb
    rgb[1::2, 1::2, 2] = bayer[1::2, 1::2]  # B

    # Simple interpolation for missing channels
    from scipy.ndimage import uniform_filter
    for c in range(3):
        mask = np.zeros((h, w), dtype=np.float32)
        if c == 0:  # R at (0,0)
            mask[0::2, 0::2] = 1
        elif c == 1:  # G at (0,1) and (1,0)
            mask[0::2, 1::2] = 1
            mask[1::2, 0::2] = 1
        else:  # B at (1,1)
            mask[1::2, 1::2] = 1

        # Weighted average
        num = uniform_filter(rgb[:,:,c], size=3)
        den = uniform_filter(mask, size=3)
        den = np.maximum(den, 1e-10)
        interp = num / den
        rgb[:,:,c] = np.where(mask > 0, rgb[:,:,c], interp)

    return np.clip(rgb, 0, 1)


def linear_to_srgb(img):
    """Linear to sRGB gamma."""
    return np.where(img <= 0.0031308,
                    12.92 * img,
                    1.055 * np.power(np.maximum(img, 0), 1/2.4) - 0.055)


def save_comparison_crop(gt_bayer, noisy_bayer, ours_bayer, pc_bayer,
                         scene_name, crop_y, crop_x, crop_size=256):
    """Save comparison crops demosaiced to sRGB."""
    from PIL import Image

    crops = {
        'GT': gt_bayer[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size],
        'Noisy': noisy_bayer[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size],
        'Ours': ours_bayer[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size],
        'PerChannel': pc_bayer[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size],
    }

    for name, crop in crops.items():
        rgb = demosaic_simple(crop)
        srgb = linear_to_srgb(rgb)
        srgb = np.clip(srgb * 255, 0, 255).astype(np.uint8)

        out_path = COMPARISON_DIR / f"{scene_name}_{name}.png"
        Image.fromarray(srgb).save(str(out_path))

    # Side-by-side
    images = []
    for name in ['Noisy', 'Ours', 'PerChannel', 'GT']:
        p = COMPARISON_DIR / f"{scene_name}_{name}.png"
        images.append(np.array(Image.open(str(p))))

    combined = np.concatenate(images, axis=1)
    out_path = COMPARISON_DIR / f"{scene_name}_comparison.png"
    Image.fromarray(combined).save(str(out_path))
    print(f"  Saved comparison: {out_path}")


def main():
    with open(DATASET_DIR / "download_list.json") as f:
        pairs = json.load(f)

    results = []

    for pair in pairs:
        scene = pair["scene"]
        gt_file = DATASET_DIR / f"file_{pair['gt_id']}.arw"
        noisy_file = DATASET_DIR / f"file_{pair['noisy_id']}.arw"

        if not gt_file.exists() or not noisy_file.exists():
            print(f"\nSkipping {scene}: files not downloaded")
            continue

        print(f"\n{'='*60}")
        print(f"Scene: {scene} (ISO {pair['iso']})")
        print(f"{'='*60}")

        # Load raw data
        print("Loading GT...")
        gt_bayer, gt_pattern = load_arw_raw(gt_file)
        print("Loading Noisy...")
        noisy_bayer, noisy_pattern = load_arw_raw(noisy_file)

        # Ensure same CFA and RGGB
        gt_bayer = ensure_rggb(gt_bayer, gt_pattern)
        noisy_bayer = ensure_rggb(noisy_bayer, noisy_pattern)

        # Ensure same size
        h = min(gt_bayer.shape[0], noisy_bayer.shape[0])
        w = min(gt_bayer.shape[1], noisy_bayer.shape[1])
        h -= h % 2; w -= w % 2
        gt_bayer = gt_bayer[:h, :w]
        noisy_bayer = noisy_bayer[:h, :w]

        # Center crop for speed
        if EVAL_CROP_SIZE > 0 and (h > EVAL_CROP_SIZE or w > EVAL_CROP_SIZE):
            cs = EVAL_CROP_SIZE
            cy, cx = h // 2, w // 2
            y0 = (cy - cs // 2) & ~1  # align to 2
            x0 = (cx - cs // 2) & ~1
            gt_bayer = gt_bayer[y0:y0+cs, x0:x0+cs]
            noisy_bayer = noisy_bayer[y0:y0+cs, x0:x0+cs]
            h, w = gt_bayer.shape

        print(f"  Final size: {h}x{w}")

        # Estimate noise parameters
        print("Estimating noise parameters...")
        alpha, sigma_sq = estimate_noise_params(noisy_bayer, gt_bayer)

        # Save to binary
        tmp_noisy = str(Path(__file__).parent.parent / "standalone" / "tmp_rawnind_noisy.bin")
        tmp_ours = str(Path(__file__).parent.parent / "standalone" / "tmp_rawnind_ours.bin")
        tmp_pc = str(Path(__file__).parent.parent / "standalone" / "tmp_rawnind_pc.bin")

        noisy_bayer.tofile(tmp_noisy)

        # Run Ours (GAT+BM3D+L/C)
        print("Running Ours (GAT+BM3D+L/C)...")
        r = run_denoiser(tmp_noisy, tmp_ours, w, h, "ours",
                         alpha, sigma_sq, strength=1.0, luma_str=0.7, chroma_str=1.5)
        print(f"  Ours stderr: {r.stderr.strip()}")
        if not os.path.exists(tmp_ours):
            print(f"  ERROR: output not produced, skipping")
            continue
        ours_bayer = np.fromfile(tmp_ours, dtype=np.float32).reshape(h, w)

        # Run PerChannel BM3D
        print("Running PerChannel BM3D...")
        r = run_denoiser(tmp_noisy, tmp_pc, w, h, "perchannel", alpha, sigma_sq)
        print(f"  PC stderr: {r.stderr.strip()}")
        if not os.path.exists(tmp_pc):
            print(f"  ERROR: output not produced, skipping")
            continue
        pc_bayer = np.fromfile(tmp_pc, dtype=np.float32).reshape(h, w)

        # Compute metrics
        p_noisy = psnr_raw(gt_bayer, noisy_bayer)
        p_ours = psnr_raw(gt_bayer, ours_bayer)
        p_pc = psnr_raw(gt_bayer, pc_bayer)

        s_noisy = ssim_4ch(gt_bayer, noisy_bayer)
        s_ours = ssim_4ch(gt_bayer, ours_bayer)
        s_pc = ssim_4ch(gt_bayer, pc_bayer)

        print(f"\n  Results for {scene} (ISO {pair['iso']}):")
        print(f"  {'Method':<15} {'PSNR':>8} {'SSIM':>8}")
        print(f"  {'Noisy':<15} {p_noisy:>8.2f} {s_noisy:>8.4f}")
        print(f"  {'PerChannel':<15} {p_pc:>8.2f} {s_pc:>8.4f}")
        print(f"  {'Ours':<15} {p_ours:>8.2f} {s_ours:>8.4f}")
        print(f"  {'Delta(Ours-PC)':<15} {p_ours-p_pc:>+8.2f} {s_ours-s_pc:>+8.4f}")

        result = {
            "scene": scene,
            "iso": pair["iso"],
            "alpha": alpha,
            "sigma_sq": sigma_sq,
            "noisy_psnr": round(p_noisy, 2),
            "noisy_ssim": round(s_noisy, 4),
            "perchannel_psnr": round(p_pc, 2),
            "perchannel_ssim": round(s_pc, 4),
            "ours_psnr": round(p_ours, 2),
            "ours_ssim": round(s_ours, 4),
            "delta_psnr": round(p_ours - p_pc, 2),
            "delta_ssim": round(s_ours - s_pc, 4),
        }
        results.append(result)

        # Save comparison crops (center region)
        crop_size = min(512, h//2, w//2)
        crop_size -= crop_size % 2
        crop_y = (h - crop_size) // 2
        crop_x = (w - crop_size) // 2
        crop_y -= crop_y % 2
        crop_x -= crop_x % 2

        try:
            save_comparison_crop(gt_bayer, noisy_bayer, ours_bayer, pc_bayer,
                                 scene, crop_y, crop_x, crop_size)
        except Exception as e:
            print(f"  Warning: comparison image failed: {e}")

        # Cleanup temp files
        for f in [tmp_noisy, tmp_ours, tmp_pc]:
            try:
                os.remove(f)
            except:
                pass

    # Summary
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY: RawNIND Real Camera Data (ISO 65535)")
        print(f"{'='*60}")
        print(f"{'Scene':<30} {'dPSNR':>8} {'dSSIM':>8}")
        print("-" * 50)

        avg_dpsnr = 0
        avg_dssim = 0
        for r in results:
            print(f"{r['scene']:<30} {r['delta_psnr']:>+8.2f} {r['delta_ssim']:>+8.4f}")
            avg_dpsnr += r['delta_psnr']
            avg_dssim += r['delta_ssim']

        avg_dpsnr /= len(results)
        avg_dssim /= len(results)
        print("-" * 50)
        print(f"{'Average':<30} {avg_dpsnr:>+8.2f} {avg_dssim:>+8.4f}")

        # Save results
        out_path = RESULTS_DIR / "rawnind_results.json"
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
