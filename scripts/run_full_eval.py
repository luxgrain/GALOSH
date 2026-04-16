#!/usr/bin/env python3
"""
Full evaluation run with comparison image generation.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
from pathlib import Path
from eval_framework import *

COMPARE_DIR = EVAL_ROOT / "comparison_images"

def save_comparison_crops(noisy_raw, clean_raw, denoised_dict, name,
                          alpha=1.0, sigma_sq=0.0, max_val=1.0):
    """Save visual comparison crops with multiple ROIs."""
    from skimage.io import imsave

    h, w = noisy_raw.shape
    crop_size = min(256, h//2, w//2)
    crop_size = crop_size - (crop_size % 2)

    # Pick 3 crops: center, top-left detail, bottom-right
    crops = []
    cy, cx = h//2, w//2
    crops.append(('center', cy - crop_size//2, cx - crop_size//2))
    crops.append(('topleft', crop_size//4, crop_size//4))
    crops.append(('botright', h - crop_size - crop_size//4, w - crop_size - crop_size//4))

    for crop_name, y0, x0 in crops:
        y0 = max(0, y0 - (y0 % 2))
        x0 = max(0, x0 - (x0 % 2))
        if y0 + crop_size > h or x0 + crop_size > w:
            continue

        def raw_to_srgb(raw_data):
            crop = raw_data[y0:y0+crop_size, x0:x0+crop_size]
            rgb = demosaic_for_display(crop)
            if max_val != 1.0:
                rgb = rgb / max_val
            srgb = linear_to_srgb(np.clip(rgb, 0, 1))
            return (np.clip(srgb, 0, 1) * 255).astype(np.uint8)

        out_dir = COMPARE_DIR / name / crop_name
        out_dir.mkdir(parents=True, exist_ok=True)

        imsave(str(out_dir / "noisy.png"), raw_to_srgb(noisy_raw))
        imsave(str(out_dir / "clean.png"), raw_to_srgb(clean_raw))

        for method_name, denoised in denoised_dict.items():
            safe = method_name.replace(' ', '_').replace('(', '').replace(')', '') \
                              .replace('+', '_').replace('/', '_')
            imsave(str(out_dir / f"{safe}.png"), raw_to_srgb(denoised))

    print(f"    Saved comparison crops -> {COMPARE_DIR / name}")


def run_full_kodak_eval():
    """Full Kodak evaluation with all ISO levels and comparison images."""
    print("="*70)
    print("  FULL KODAK EVALUATION - Poisson-Gaussian Noise")
    print("="*70)

    images = load_kodak_images()
    if not images:
        print("No Kodak images!")
        return []

    iso_levels = [400, 800, 1600, 3200, 6400, 12800]
    all_results = []

    # Track best images for comparison
    comparison_candidates = []

    for iso in iso_levels:
        print(f"\n{'='*50}")
        print(f"  ISO {iso}")
        print(f"{'='*50}")
        method_names = ['BM3D-perchannel', 'CBM3D', 'Ours (GAT+BM3D+L/C)']
        iso_results = {m: [] for m in method_names}

        for img_name, img_rgb in images:  # All 24 images
            noisy_raw, clean_raw, alpha, sigma_sq = synthesize_bayer_noise(img_rgb, iso)
            noisy_psnr = psnr_4ch(clean_raw, noisy_raw, 1.0)

            denoised_dict = {}

            # BM3D per-channel
            t0 = time.time()
            d_pc = denoise_bm3d_perchannel(noisy_raw, use_gat=True,
                                            alpha=alpha, sigma_sq=sigma_sq)
            t_pc = time.time() - t0
            p_pc = psnr_4ch(clean_raw, d_pc, 1.0)
            s_pc = ssim_4ch(clean_raw, d_pc, 1.0)
            denoised_dict['BM3D-perchannel'] = d_pc

            # CBM3D (estimate sigma from GAT domain)
            t0 = time.time()
            R_ch = extract_bayer_channels(noisy_raw)[0]
            sig_est = estimate_gat_sigma(gat_forward(R_ch, alpha, sigma_sq))
            d_cb = denoise_cbm3d(noisy_raw, sigma=sig_est * 0.3)
            t_cb = time.time() - t0
            p_cb = psnr_4ch(clean_raw, d_cb, 1.0)
            s_cb = ssim_4ch(clean_raw, d_cb, 1.0)
            denoised_dict['CBM3D'] = d_cb

            # Ours
            t0 = time.time()
            d_ours = denoise_ours(noisy_raw, strength=1.0,
                                   luma_strength=0.7, chroma_strength=1.5,
                                   alpha=alpha, sigma_sq=sigma_sq, use_gat=True)
            t_ours = time.time() - t0
            p_ours = psnr_4ch(clean_raw, d_ours, 1.0)
            s_ours = ssim_4ch(clean_raw, d_ours, 1.0)
            denoised_dict['Ours (GAT+BM3D+L/C)'] = d_ours

            # Store results
            for mname, p, s, t in [
                ('BM3D-perchannel', p_pc, s_pc, t_pc),
                ('CBM3D', p_cb, s_cb, t_cb),
                ('Ours (GAT+BM3D+L/C)', p_ours, s_ours, t_ours),
            ]:
                r = EvalResult(method=mname, dataset='kodak', image_name=img_name,
                               psnr_raw=p, ssim_raw=s, time_sec=t, iso=iso)
                # sRGB metrics
                try:
                    r.cpsnr_srgb, r.ssim_srgb = compute_srgb_metrics(
                        clean_raw, denoised_dict[mname], 1.0)
                except:
                    pass
                iso_results[mname].append(r)
                all_results.append(r)

            # Print per-image
            delta = p_ours - p_pc
            marker = " ***" if delta > 0.5 else ""
            print(f"  {img_name} | noisy={noisy_psnr:.1f} | "
                  f"pc={p_pc:.2f} cb={p_cb:.2f} ours={p_ours:.2f} "
                  f"(+{delta:.2f}){marker} | "
                  f"ssim: pc={s_pc:.3f} ours={s_ours:.3f}")

            # Track for comparison image selection
            comparison_candidates.append({
                'name': f"{img_name}_ISO{iso}",
                'iso': iso,
                'img_name': img_name,
                'delta_psnr': p_ours - p_pc,
                'delta_ssim': s_ours - s_pc,
                'noisy_raw': noisy_raw,
                'clean_raw': clean_raw,
                'denoised_dict': denoised_dict,
                'alpha': alpha,
                'sigma_sq': sigma_sq,
            })

        # Print ISO averages
        print(f"\n  --- ISO {iso} AVERAGES ---")
        for mname in method_names:
            results = iso_results[mname]
            if results:
                avg_p = np.mean([r.psnr_raw for r in results])
                avg_s = np.mean([r.ssim_raw for r in results])
                avg_cp = np.mean([r.cpsnr_srgb for r in results])
                avg_t = np.mean([r.time_sec for r in results])
                print(f"    {mname:25s} | PSNR={avg_p:.2f} SSIM={avg_s:.4f} "
                      f"CPSNR={avg_cp:.2f} | {avg_t:.1f}s")

    # Pick best comparison images
    print("\n" + "="*70)
    print("  GENERATING COMPARISON IMAGES")
    print("="*70)

    # Sort by delta_ssim (where our method wins most)
    comparison_candidates.sort(key=lambda x: x['delta_ssim'], reverse=True)

    # Pick diverse: best overall, best at high ISO, interesting scene
    picked = set()
    selected = []

    # Best SSIM improvement
    for c in comparison_candidates:
        if c['name'] not in picked:
            selected.append(c)
            picked.add(c['name'])
            if len(selected) >= 2:
                break

    # Best at highest ISO
    high_iso = [c for c in comparison_candidates if c['iso'] >= 6400 and c['name'] not in picked]
    for c in high_iso[:2]:
        selected.append(c)
        picked.add(c['name'])

    # One medium ISO
    mid_iso = [c for c in comparison_candidates if c['iso'] in [1600, 3200] and c['name'] not in picked]
    for c in mid_iso[:1]:
        selected.append(c)
        picked.add(c['name'])

    for c in selected:
        print(f"\n  Generating: {c['name']} (delta_PSNR={c['delta_psnr']:+.2f}, "
              f"delta_SSIM={c['delta_ssim']:+.4f})")
        save_comparison_crops(c['noisy_raw'], c['clean_raw'], c['denoised_dict'],
                              c['name'], c['alpha'], c['sigma_sq'])

    return all_results


def main():
    print("="*70)
    print("  RAW BAYER DENOISING BENCHMARK")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    all_results = []

    # 1. Full Kodak eval
    t0 = time.time()
    kodak_results = run_full_kodak_eval()
    all_results.extend(kodak_results)
    print(f"\nKodak eval completed in {time.time()-t0:.0f}s")

    # 2. SIDD synthetic patches
    try:
        t0 = time.time()
        sidd_results = eval_sidd()
        all_results.extend(sidd_results)
        print(f"SIDD eval completed in {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"SIDD eval failed: {e}")
        import traceback; traceback.print_exc()

    # Save everything
    save_results(all_results)
    print_summary_table(all_results)

    # Print LaTeX table
    print("\n\n--- LaTeX Table ---")
    print_latex_table(all_results)

def print_latex_table(results):
    """Generate LaTeX-ready results table."""
    from collections import defaultdict

    # Group by ISO and method
    by_iso = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.dataset == 'kodak':
            by_iso[r.iso][r.method].append(r)

    print("\\begin{tabular}{l" + "cc" * len(by_iso) + "}")
    print("\\toprule")

    isos = sorted(by_iso.keys())
    header = "Method"
    for iso in isos:
        header += f" & \\multicolumn{{2}}{{c}}{{ISO {iso}}}"
    header += " \\\\"
    print(header)

    subheader = ""
    for iso in isos:
        subheader += " & PSNR & SSIM"
    subheader += " \\\\"
    print(subheader)
    print("\\midrule")

    method_order = ['BM3D-perchannel', 'CBM3D', 'Ours (GAT+BM3D+L/C)']
    for method in method_order:
        row = method.replace('_', '\\_')
        for iso in isos:
            rs = by_iso[iso].get(method, [])
            if rs:
                avg_p = np.mean([r.psnr_raw for r in rs])
                avg_s = np.mean([r.ssim_raw for r in rs])
                row += f" & {avg_p:.2f} & {avg_s:.4f}"
            else:
                row += " & - & -"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
