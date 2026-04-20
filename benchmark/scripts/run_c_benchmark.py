#!/usr/bin/env python3
"""
Full benchmark using C standalone denoiser.
Runs both 'ours' and 'perchannel' methods, computes metrics.
"""
import os, sys, subprocess, time, json
import numpy as np
from pathlib import Path
from skimage.io import imread, imsave
from skimage.metrics import structural_similarity

EVAL_ROOT = Path(r"C:\Users\luxgrain\denoise_eval")
STANDALONE = EVAL_ROOT / "standalone" / "rawdenoise.exe"
KODAK_DIR = EVAL_ROOT / "datasets" / "kodak"
RESULT_DIR = EVAL_ROOT / "results"
COMPARE_DIR = EVAL_ROOT / "comparison_images_c"
WORK_DIR = EVAL_ROOT / "standalone"

# Ensure PATH includes MSYS2 for DLL dependencies
os.environ['PATH'] = r"C:\msys64\ucrt64\bin;" + os.environ.get('PATH', '')

def psnr(a, b, max_val=1.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    if mse < 1e-10: return 100.0
    return 10 * np.log10(max_val**2 / mse)

def ssim_4ch(a, b, max_val=1.0):
    vals = []
    for dy, dx in [(0,0),(0,1),(1,0),(1,1)]:
        vals.append(structural_similarity(
            a[dy::2, dx::2], b[dy::2, dx::2], data_range=max_val))
    return np.mean(vals)

def inv_srgb(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0/2.4) - 0.055)

def demosaic_simple(raw):
    """Simple bilinear demosaic for visualization."""
    from scipy.ndimage import convolve
    h, w = raw.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    R_mask = np.zeros((h, w), dtype=np.float32); R_mask[0::2, 0::2] = 1
    G_mask = np.zeros((h, w), dtype=np.float32); G_mask[0::2, 1::2] = 1; G_mask[1::2, 0::2] = 1
    B_mask = np.zeros((h, w), dtype=np.float32); B_mask[1::2, 1::2] = 1
    k_rb = np.array([[.25,.5,.25],[.5,1,.5],[.25,.5,.25]])
    k_g = np.array([[0,.25,0],[.25,1,.25],[0,.25,0]])
    rgb[:,:,0] = convolve(raw * R_mask, k_rb, mode='mirror')
    rgb[:,:,1] = convolve(raw * G_mask, k_g, mode='mirror')
    rgb[:,:,2] = convolve(raw * B_mask, k_rb, mode='mirror')
    return rgb

def synth_poisson_gaussian(clean_rgb, iso):
    """Create noisy raw Bayer with Poisson-Gaussian noise."""
    h, w = clean_rgb.shape[:2]
    h -= h % 2; w -= w % 2
    clean_rgb = clean_rgb[:h, :w].astype(np.float32)
    if clean_rgb.max() > 1.0: clean_rgb /= 255.0
    linear = inv_srgb(clean_rgb).astype(np.float32)

    clean_raw = np.zeros((h, w), dtype=np.float32)
    clean_raw[0::2, 0::2] = linear[0::2, 0::2, 0]
    clean_raw[0::2, 1::2] = linear[0::2, 1::2, 1]
    clean_raw[1::2, 0::2] = linear[1::2, 0::2, 1]
    clean_raw[1::2, 1::2] = linear[1::2, 1::2, 2]

    gain = iso / 100.0
    alpha = 0.001 * gain
    sigma_read = 0.002 * gain
    sigma_sq = sigma_read ** 2

    rate = np.maximum(clean_raw / max(alpha, 1e-10), 0.0)
    noisy = np.random.poisson(rate).astype(np.float32) * alpha
    noisy += np.random.randn(h, w).astype(np.float32) * sigma_read
    noisy = np.clip(noisy, 0, 1).astype(np.float32)

    return noisy, clean_raw, alpha, sigma_sq

def run_c_method(input_bin, output_bin, w, h, method, strength=1.0,
                  luma=0.7, chroma=1.5, alpha=1.0, sigma_sq=0.0, iters=1):
    """Run C standalone denoiser."""
    cmd = [str(STANDALONE), str(input_bin), str(output_bin),
           str(w), str(h), method,
           str(strength), str(luma), str(chroma),
           str(alpha), str(sigma_sq), str(iters)]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
    return elapsed

def main():
    np.random.seed(42)  # Reproducibility

    print("="*70)
    print("  C STANDALONE BENCHMARK - GAT+BM3D+L/C vs BM3D-perchannel")
    print("="*70)

    # Load Kodak
    kodak_files = sorted(KODAK_DIR.glob("kodim*.png"))
    print(f"Found {len(kodak_files)} Kodak images")

    iso_levels = [400, 800, 1600, 3200, 6400, 12800]
    all_results = []
    comparison_data = []

    for iso in iso_levels:
        print(f"\n{'='*50}")
        print(f"  ISO {iso}")
        print(f"{'='*50}")

        iso_psnr_ours = []
        iso_ssim_ours = []
        iso_psnr_pc = []
        iso_ssim_pc = []

        for kodak_f in kodak_files:
            img = imread(str(kodak_f)).astype(np.float32) / 255.0
            name = kodak_f.stem
            noisy, clean, alpha, sigma_sq = synth_poisson_gaussian(img, iso)
            h, w = noisy.shape

            # Save input
            noisy_bin = WORK_DIR / "tmp_noisy.bin"
            ours_bin = WORK_DIR / "tmp_ours.bin"
            pc_bin = WORK_DIR / "tmp_pc.bin"
            noisy.tofile(str(noisy_bin))

            # Run Ours
            t_ours = run_c_method(noisy_bin, ours_bin, w, h, "ours",
                                   1.0, 0.7, 1.5, alpha, sigma_sq, 1)
            ours_out = np.fromfile(str(ours_bin), dtype=np.float32).reshape(h, w)

            # Run Perchannel
            t_pc = run_c_method(noisy_bin, pc_bin, w, h, "perchannel",
                                 1.0, 0.7, 1.5, alpha, sigma_sq, 1)
            pc_out = np.fromfile(str(pc_bin), dtype=np.float32).reshape(h, w)

            # Metrics
            noisy_psnr = psnr(clean, noisy)
            p_ours = psnr(clean, ours_out)
            s_ours = ssim_4ch(clean, ours_out)
            p_pc = psnr(clean, pc_out)
            s_pc = ssim_4ch(clean, pc_out)

            iso_psnr_ours.append(p_ours)
            iso_ssim_ours.append(s_ours)
            iso_psnr_pc.append(p_pc)
            iso_ssim_pc.append(s_pc)

            delta_p = p_ours - p_pc
            delta_s = s_ours - s_pc
            marker = " ***" if delta_p > 0.3 else ""

            print(f"  {name} | noisy={noisy_psnr:.1f} | "
                  f"pc={p_pc:.2f}/{s_pc:.3f} ours={p_ours:.2f}/{s_ours:.3f} "
                  f"(dp={delta_p:+.2f} ds={delta_s:+.3f}){marker}")

            all_results.append({
                'image': name, 'iso': iso,
                'psnr_pc': round(p_pc, 3), 'ssim_pc': round(s_pc, 4),
                'psnr_ours': round(p_ours, 3), 'ssim_ours': round(s_ours, 4),
                'psnr_noisy': round(noisy_psnr, 3),
                'time_ours': round(t_ours, 2), 'time_pc': round(t_pc, 2),
            })

            # Save comparison image data for best cases
            comparison_data.append({
                'name': f'{name}_ISO{iso}', 'iso': iso,
                'delta_psnr': delta_p, 'delta_ssim': delta_s,
                'noisy': noisy, 'clean': clean, 'ours': ours_out, 'pc': pc_out,
            })

        # ISO averages
        avg_p_ours = np.mean(iso_psnr_ours)
        avg_s_ours = np.mean(iso_ssim_ours)
        avg_p_pc = np.mean(iso_psnr_pc)
        avg_s_pc = np.mean(iso_ssim_pc)
        print(f"\n  ISO {iso} AVERAGES:")
        print(f"    BM3D-perchannel: PSNR={avg_p_pc:.2f}  SSIM={avg_s_pc:.4f}")
        print(f"    Ours (GAT+L/C):  PSNR={avg_p_ours:.2f}  SSIM={avg_s_ours:.4f}")
        print(f"    Delta:           PSNR={avg_p_ours-avg_p_pc:+.2f}  "
              f"SSIM={avg_s_ours-avg_s_pc:+.4f}")

    # Save results JSON
    RESULT_DIR.mkdir(exist_ok=True)
    with open(RESULT_DIR / "c_benchmark_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate comparison images for best cases
    print("\n" + "="*70)
    print("  GENERATING COMPARISON IMAGES")
    print("="*70)

    comparison_data.sort(key=lambda x: x['delta_ssim'], reverse=True)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    # Pick 5 diverse comparison images
    picked = set()
    selected = []

    # Best SSIM wins at different ISOs
    for target_iso in [12800, 6400, 3200, 1600]:
        for c in comparison_data:
            if c['iso'] == target_iso and c['name'] not in picked:
                selected.append(c)
                picked.add(c['name'])
                break

    # Best overall
    for c in comparison_data:
        if c['name'] not in picked:
            selected.append(c)
            picked.add(c['name'])
            if len(selected) >= 5:
                break

    for c in selected:
        out_dir = COMPARE_DIR / c['name']
        out_dir.mkdir(parents=True, exist_ok=True)

        # Center crop
        h, w = c['clean'].shape
        cs = min(256, h//2, w//2)
        cs -= cs % 2
        y0 = h//2 - cs//2; y0 -= y0 % 2
        x0 = w//2 - cs//2; x0 -= x0 % 2

        for label, data in [('clean', c['clean']), ('noisy', c['noisy']),
                            ('bm3d_perchannel', c['pc']), ('ours_gat_lc', c['ours'])]:
            crop = data[y0:y0+cs, x0:x0+cs]
            rgb = demosaic_simple(crop)
            srgb = (to_srgb(np.clip(rgb, 0, 1)) * 255).astype(np.uint8)
            imsave(str(out_dir / f"{label}.png"), srgb)

        print(f"  {c['name']}: dp={c['delta_psnr']:+.2f} ds={c['delta_ssim']:+.4f}")

    # Final summary table
    print("\n" + "="*70)
    print("  SUMMARY TABLE (C Standalone)")
    print("="*70)
    print(f"{'ISO':>6s} | {'BM3D-pc PSNR':>12s} | {'BM3D-pc SSIM':>12s} | "
          f"{'Ours PSNR':>10s} | {'Ours SSIM':>10s} | {'dPSNR':>6s} | {'dSSIM':>7s}")
    print("-"*80)

    for iso in iso_levels:
        iso_r = [r for r in all_results if r['iso'] == iso]
        avg_pp = np.mean([r['psnr_pc'] for r in iso_r])
        avg_sp = np.mean([r['ssim_pc'] for r in iso_r])
        avg_po = np.mean([r['psnr_ours'] for r in iso_r])
        avg_so = np.mean([r['ssim_ours'] for r in iso_r])
        print(f"{iso:6d} | {avg_pp:12.2f} | {avg_sp:12.4f} | "
              f"{avg_po:10.2f} | {avg_so:10.4f} | {avg_po-avg_pp:+6.2f} | {avg_so-avg_sp:+7.4f}")

    print("\nDone!")

if __name__ == "__main__":
    main()
