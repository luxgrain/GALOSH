"""
Export SIDD comparison images + manifest.json for viewer.html

Usage:
  python export_comparison.py [--scenes N] [--patches N] [--out DIR]

Reads:
  datasets/sidd/ValidationGtBlocksSrgb.mat
  datasets/sidd/ValidationNoisyBlocksSrgb.mat
  + denoised npy files passed via --method

Example:
  python export_comparison.py \
    --method "NAFNet:results/nafnet.npy" \
    --method "GALOSH:results/galosh.npy" \
    --scenes 5 --patches 10

Output structure (viewer.html compatible):
  comparison_viewer/
    __gt__/patchSsPp_gt.png
    __noisy__/patchSsPp_noisy.png
    {method_a}/patchSsPp_{method}.png
    {method_b}/patchSsPp_{method}.png
    manifest.json
"""

import sys, os, argparse, json
import numpy as np
import scipy.io as sio
from pathlib import Path
from PIL import Image

BASE = Path(__file__).parent
SIDD = BASE / 'datasets' / 'sidd'

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255**2 / mse) if mse > 1e-10 else 100.0

def save_png(arr_uint8, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_uint8).save(str(path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes',  type=int, default=5,  help='Number of scenes (max 40)')
    parser.add_argument('--patches', type=int, default=32, help='Patches per scene (max 32)')
    parser.add_argument('--out',     default='comparison_viewer', help='Output directory')
    parser.add_argument('--method',  action='append', default=[],
                        metavar='LABEL:path.npy',
                        help='Denoised result. Can specify up to 2. '
                             'Array shape must match GT: (scenes, patches, H, W, C) or (N, H, W, C)')
    args = parser.parse_args()

    out_dir = BASE / args.out
    print(f'Loading SIDD GT/Noisy...', end='', flush=True)
    gt_mat    = sio.loadmat(str(SIDD / 'ValidationGtBlocksSrgb.mat'))['ValidationGtBlocksSrgb']
    noisy_mat = sio.loadmat(str(SIDD / 'ValidationNoisyBlocksSrgb.mat'))['ValidationNoisyBlocksSrgb']
    print(' done.')

    n_scenes  = min(args.scenes,  gt_mat.shape[0])
    n_patches = min(args.patches, gt_mat.shape[1])
    print(f'Scenes: {n_scenes}, Patches/scene: {n_patches}, Total: {n_scenes*n_patches}')

    # Load method arrays
    methods = []  # [(label, flat_array)]  flat_array: (N, H, W, C) uint8
    for spec in args.method[:2]:
        if ':' not in spec:
            print(f'Warning: skip {spec!r} (expected LABEL:path)')
            continue
        label, path = spec.split(':', 1)
        arr = np.load(path)
        print(f'Loaded {label}: shape={arr.shape} dtype={arr.dtype}')
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8) if arr.max() <= 1.01 else arr.astype(np.uint8)
        methods.append((label, arr))

    manifest = {'labels': {}, 'patches': {}}
    if len(methods) >= 1: manifest['labels']['a'] = methods[0][0]
    if len(methods) >= 2: manifest['labels']['b'] = methods[1][0]

    idx = 0
    for s in range(n_scenes):
        for p in range(n_patches):
            pid = f'patch{s:02d}p{p:02d}'  # e.g. patch00p03

            gt    = gt_mat[s, p]    # uint8 H×W×3
            noisy = noisy_mat[s, p]

            save_png(gt,    out_dir / '__gt__'    / f'{pid}_gt.png')
            save_png(noisy, out_dir / '__noisy__' / f'{pid}_noisy.png')

            patch_scores = {}

            for mi, (label, arr) in enumerate(methods):
                slot = 'ab'[mi]
                # Support both (S, P, H, W, C) and flat (N, H, W, C)
                if arr.ndim == 5:
                    denoised = arr[s, p]
                else:
                    denoised = arr[idx]
                method_key = label.lower().replace(' ', '_')
                save_png(denoised, out_dir / method_key / f'{pid}_{method_key}.png')
                psnr_val = psnr(gt, denoised)
                patch_scores[slot] = {'psnr': round(psnr_val, 4)}

            manifest['patches'][pid] = patch_scores

            if (idx + 1) % 20 == 0:
                print(f'  {idx+1}/{n_scenes*n_patches} patches done')
            idx += 1

    manifest_path = out_dir / 'manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(manifest_path), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\nDone! Output: {out_dir}')
    print(f'  Open viewer.html in browser, then "Open Folder" → select: {out_dir.name}/')
    print(f'  Then "Load Manifest JSON" → select: {out_dir.name}/manifest.json')

if __name__ == '__main__':
    main()
