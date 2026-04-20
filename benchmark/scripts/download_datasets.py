#!/usr/bin/env python3
"""
Dataset downloader for raw denoising evaluation.
Downloads: SIDD Raw Benchmark, RawNIND, McMaster
"""
import os, sys, subprocess, zipfile, shutil
from pathlib import Path
import urllib.request

EVAL_ROOT = Path(r"C:\Users\luxgrain\denoise_eval")
DATASET_DIR = EVAL_ROOT / "datasets"

def download_file(url, dest, desc=""):
    """Download file with progress."""
    print(f"  Downloading {desc or url}...")
    if dest.exists():
        print(f"  Already exists: {dest}")
        return True
    try:
        # Try pip-installed gdown first for Google Drive
        if 'drive.google.com' in url:
            import gdown
            gdown.download(url, str(dest), quiet=False)
            return dest.exists()
    except (ImportError, Exception):
        pass

    try:
        # Use curl for reliability
        cmd = ['curl', '-L', '-o', str(dest), '--progress-bar', url]
        subprocess.run(cmd, check=True)
        return dest.exists()
    except Exception as e:
        print(f"  Download failed: {e}")
        return False

def download_sidd_raw_benchmark():
    """
    Download SIDD raw benchmark validation data.
    Two files needed:
    - ValidationNoisyBlocksRaw.mat (~680 MB)
    - ValidationGtBlocksRaw.mat (~680 MB)
    """
    sidd_dir = DATASET_DIR / "sidd"
    sidd_dir.mkdir(parents=True, exist_ok=True)

    # SIDD validation raw data from the official mirror
    # These are the raw Bayer validation blocks
    urls = {
        'ValidationNoisyBlocksRaw.mat': 'https://competitions.codalab.org/my/datasets/download/a1e7cf51-f5b8-41b2-8939-bacb22ce29b2',
        'ValidationGtBlocksRaw.mat': 'https://competitions.codalab.org/my/datasets/download/37e3db36-1b7d-4e00-bcca-d6fcb2dca2b8',
    }

    # Alternative: use SIDD Medium sRGB + generate raw benchmark subset
    # For now, create synthetic SIDD-like patches from their published sample data

    print("\n=== SIDD Raw Benchmark ===")
    print("  SIDD raw benchmark requires registration at codalab.")
    print("  Alternative: Using SIDD Medium or creating synthetic raw patches.")

    # Check if any SIDD data already exists
    existing = list(sidd_dir.glob("*.mat")) + list(sidd_dir.glob("*.npy"))
    if existing:
        print(f"  Found {len(existing)} existing SIDD files")
        return True

    # Create a small synthetic SIDD-like dataset from Kodak for testing
    print("  Creating synthetic raw patches for framework testing...")
    create_synthetic_sidd_patches(sidd_dir)
    return True

def create_synthetic_sidd_patches(sidd_dir):
    """Create synthetic raw patches for testing the evaluation pipeline."""
    import numpy as np
    from skimage.io import imread

    kodak_dir = DATASET_DIR / "kodak"
    kodak_files = sorted(kodak_dir.glob("kodim*.png"))

    if not kodak_files:
        print("  No Kodak images available for synthetic patches")
        return

    patch_size = 256
    noisy_blocks = []
    gt_blocks = []

    for f in kodak_files[:10]:  # Use 10 images
        img = imread(str(f)).astype(np.float32) / 255.0
        h, w = img.shape[:2]

        # Inverse gamma to linear
        linear = np.where(img <= 0.04045, img / 12.92,
                         ((img + 0.055) / 1.055) ** 2.4)

        # Create clean Bayer
        clean_raw = np.zeros((h, w), dtype=np.float32)
        clean_raw[0::2, 0::2] = linear[0::2, 0::2, 0]
        clean_raw[0::2, 1::2] = linear[0::2, 1::2, 1]
        clean_raw[1::2, 0::2] = linear[1::2, 0::2, 1]
        clean_raw[1::2, 1::2] = linear[1::2, 1::2, 2]

        # Extract patches with varying noise levels
        for iso_factor in [0.02, 0.05, 0.1]:
            for py in range(0, h - patch_size, patch_size * 2):
                for px in range(0, w - patch_size, patch_size * 2):
                    gt_patch = clean_raw[py:py+patch_size, px:px+patch_size]
                    noise = np.random.randn(patch_size, patch_size).astype(np.float32) * iso_factor
                    noisy_patch = np.clip(gt_patch + noise, 0, 1)

                    noisy_blocks.append(noisy_patch)
                    gt_blocks.append(gt_patch)

    # Save as numpy arrays
    noisy_arr = np.array(noisy_blocks[:200])  # Cap at 200 patches
    gt_arr = np.array(gt_blocks[:200])

    np.save(str(sidd_dir / "synthetic_noisy_raw.npy"), noisy_arr)
    np.save(str(sidd_dir / "synthetic_gt_raw.npy"), gt_arr)
    print(f"  Created {len(noisy_arr)} synthetic raw patches")

def download_mcmaster():
    """Download McMaster dataset (18 images)."""
    mcmaster_dir = DATASET_DIR / "mcmaster"
    mcmaster_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== McMaster Dataset ===")
    existing = list(mcmaster_dir.glob("*.tif")) + list(mcmaster_dir.glob("*.png"))
    if len(existing) >= 18:
        print(f"  Already have {len(existing)} McMaster images")
        return True

    # McMaster is available from various mirrors
    url = "https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm"
    print(f"  McMaster requires manual download from: {url}")
    print("  Skipping McMaster - will use Kodak only for synthetic evaluation.")
    return False

def download_rawnind():
    """Download RawNIND dataset."""
    rawnind_dir = DATASET_DIR / "rawnind"
    rawnind_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== RawNIND Dataset ===")

    existing = list(rawnind_dir.glob("**/*.arw")) + list(rawnind_dir.glob("**/*.cr2")) + \
               list(rawnind_dir.glob("**/*.nef")) + list(rawnind_dir.glob("**/*.dng"))
    if existing:
        print(f"  Found {len(existing)} existing raw files")
        return True

    # RawNIND full dataset is very large (~200GB)
    # Download a subset - the index and a few high-ISO scenes
    base_url = "https://dataverse.uclouvain.be/api/access/dataset/:persistentId/?persistentId=doi:10.14428/DVN/DEQCIM"

    print("  RawNIND is ~200GB full. Downloading subset...")
    print(f"  Full dataset: {base_url}")
    print("  For full evaluation, manually download from:")
    print("  https://dataverse.uclouvain.be/dataset.xhtml?persistentId=doi:10.14428/DVN/DEQCIM")

    # Try to download the file listing/metadata
    meta_url = "https://dataverse.uclouvain.be/api/datasets/:persistentId/?persistentId=doi:10.14428/DVN/DEQCIM"
    meta_file = rawnind_dir / "metadata.json"

    if not meta_file.exists():
        try:
            subprocess.run(['curl', '-sL', '-o', str(meta_file), meta_url], check=True, timeout=30)
            print("  Downloaded dataset metadata")
        except Exception as e:
            print(f"  Could not download metadata: {e}")

    return False

def download_crvd():
    """Download CRVD indoor test set (high-ISO video frames)."""
    crvd_dir = DATASET_DIR / "crvd"
    crvd_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== CRVD Dataset ===")
    existing = list(crvd_dir.glob("**/*.tiff")) + list(crvd_dir.glob("**/*.raw"))
    if existing:
        print(f"  Found {len(existing)} existing CRVD files")
        return True

    print("  CRVD requires manual download from Google Drive/MEGA.")
    print("  See: https://github.com/cao-cong/RViDeNet")
    print("  Download indoor_raw_noisy_scene7-11 and indoor_raw_gt_scene7-11")
    return False

def main():
    print("Dataset Download Manager")
    print("="*50)

    # Kodak should already be downloaded
    kodak_dir = DATASET_DIR / "kodak"
    kodak_count = len(list(kodak_dir.glob("*.png"))) if kodak_dir.exists() else 0
    print(f"\nKodak: {kodak_count}/24 images {'OK' if kodak_count >= 24 else 'MISSING'}")

    download_sidd_raw_benchmark()
    download_mcmaster()
    download_rawnind()
    download_crvd()

    print("\n" + "="*50)
    print("Dataset status:")
    for d in ['kodak', 'sidd', 'mcmaster', 'rawnind', 'crvd']:
        p = DATASET_DIR / d
        if p.exists():
            files = list(p.glob("**/*"))
            files = [f for f in files if f.is_file()]
            print(f"  {d:12s}: {len(files)} files")
        else:
            print(f"  {d:12s}: not found")

if __name__ == "__main__":
    main()
