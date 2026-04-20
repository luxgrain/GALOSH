#!/usr/bin/env python3
"""
Prepare SIDD Medium full-image benchmark data.

Extracts 20 scenes (5 cameras × 4 ISO/lighting) from SIDD_Medium_Raw.zip
and SIDD_Medium_Srgb.zip, crops to RGGB alignment, saves as .npy.

Each scene has 2 images (010, 011) → 40 full images total.

Output directory: datasets/sidd/medium_bench/
  {sceneID}_{cam}_{pattern}_{imgID}_gt_raw.npy      (H, W) float32
  {sceneID}_{cam}_{pattern}_{imgID}_noisy_raw.npy    (H, W) float32
  {sceneID}_{cam}_{pattern}_{imgID}_gt_srgb.npy      (H, W, 3) float32
  {sceneID}_{cam}_{pattern}_{imgID}_noisy_srgb.npy   (H, W, 3) float32
  scenes.json — metadata for all scenes
"""
import zipfile, io, json, h5py
import numpy as np
from PIL import Image
from pathlib import Path

BASE = Path(__file__).parent.parent
RAW_ZIP  = BASE / "datasets" / "sidd" / "SIDD_Medium_Raw.zip"
SRGB_ZIP = BASE / "datasets" / "sidd" / "SIDD_Medium_Srgb.zip"
OUTDIR   = BASE / "datasets" / "sidd" / "medium_bench"

# 20 selected scenes: 4 per camera, spread across ISO/lighting
SELECTED_SCENES = [
    # G4 (BGGR) — ISO 100-800, CT 3200-5500K
    {"dir": "0045_002_G4_00100_00060_3200_L", "id": "0045", "cam": "G4", "pat": "BGGR", "iso": 100, "ct": 3200, "light": "L"},
    {"dir": "0077_004_G4_00100_00025_3200_N", "id": "0077", "cam": "G4", "pat": "BGGR", "iso": 100, "ct": 3200, "light": "N"},
    {"dir": "0099_005_G4_00400_00200_3200_N", "id": "0099", "cam": "G4", "pat": "BGGR", "iso": 400, "ct": 3200, "light": "N"},
    {"dir": "0059_003_G4_00800_01000_5500_L", "id": "0059", "cam": "G4", "pat": "BGGR", "iso": 800, "ct": 5500, "light": "L"},
    # GP (BGGR) — ISO 100-6400
    {"dir": "0066_003_GP_00100_00200_3200_L", "id": "0066", "cam": "GP", "pat": "BGGR", "iso": 100, "ct": 3200, "light": "L"},
    {"dir": "0020_001_GP_00800_00350_5500_N", "id": "0020", "cam": "GP", "pat": "BGGR", "iso": 800, "ct": 5500, "light": "N"},
    {"dir": "0064_003_GP_01600_01600_4400_N", "id": "0064", "cam": "GP", "pat": "BGGR", "iso": 1600, "ct": 4400, "light": "N"},
    {"dir": "0036_002_GP_06400_03200_3200_N", "id": "0036", "cam": "GP", "pat": "BGGR", "iso": 6400, "ct": 3200, "light": "N"},
    # IP (RGGB) — ISO 100-1600
    {"dir": "0138_006_IP_00100_00100_3200_L", "id": "0138", "cam": "IP", "pat": "RGGB", "iso": 100, "ct": 3200, "light": "L"},
    {"dir": "0115_005_IP_00400_00750_5500_N", "id": "0115", "cam": "IP", "pat": "RGGB", "iso": 400, "ct": 5500, "light": "N"},
    {"dir": "0165_007_IP_00800_00800_3200_N", "id": "0165", "cam": "IP", "pat": "RGGB", "iso": 800, "ct": 3200, "light": "N"},
    {"dir": "0042_002_IP_01600_03100_5500_N", "id": "0042", "cam": "IP", "pat": "RGGB", "iso": 1600, "ct": 5500, "light": "N"},
    # N6 (BGGR) — ISO 100-3200
    {"dir": "0048_002_N6_00100_00100_5500_L", "id": "0048", "cam": "N6", "pat": "BGGR", "iso": 100, "ct": 5500, "light": "L"},
    {"dir": "0168_008_N6_00400_00200_4400_L", "id": "0168", "cam": "N6", "pat": "BGGR", "iso": 400, "ct": 4400, "light": "L"},
    {"dir": "0075_004_N6_00800_00080_3200_L", "id": "0075", "cam": "N6", "pat": "BGGR", "iso": 800, "ct": 3200, "light": "L"},
    {"dir": "0145_007_N6_03200_03200_4400_N", "id": "0145", "cam": "N6", "pat": "BGGR", "iso": 3200, "ct": 4400, "light": "N"},
    # S6 (GRBG) — ISO 100-3200
    {"dir": "0001_001_S6_00100_00060_3200_L", "id": "0001", "cam": "S6", "pat": "GRBG", "iso": 100, "ct": 3200, "light": "L"},
    {"dir": "0154_007_S6_00400_00400_5500_L", "id": "0154", "cam": "S6", "pat": "GRBG", "iso": 400, "ct": 5500, "light": "L"},
    {"dir": "0010_001_S6_00800_00350_3200_N", "id": "0010", "cam": "S6", "pat": "GRBG", "iso": 800, "ct": 3200, "light": "N"},
    {"dir": "0015_001_S6_03200_01600_5500_L", "id": "0015", "cam": "S6", "pat": "GRBG", "iso": 3200, "ct": 5500, "light": "L"},
]

# Crop functions: all produce RGGB alignment, trim 2px total per axis for uniform sizing
def crop_rggb(arr):
    """RGGB: already aligned, trim 2px from bottom-right to match others."""
    return arr[:-2, :-2] if arr.ndim == 2 else arr[:-2, :-2, :]

def crop_grbg(arr):
    """GRBG: shift 1 col right → R at (0,0). Trim 2px bottom, 1px each side."""
    return arr[:-2, 1:-1] if arr.ndim == 2 else arr[:-2, 1:-1, :]

def crop_bggr(arr):
    """BGGR: shift 1 row + 1 col → R at (0,0). Trim 1px each side."""
    return arr[1:-1, 1:-1] if arr.ndim == 2 else arr[1:-1, 1:-1, :]

CROP_FN = {"RGGB": crop_rggb, "GRBG": crop_grbg, "BGGR": crop_bggr}


def load_raw_from_zip(zf, scene_dir, scene_id, suffix, img_id):
    """Load RAW .MAT (HDF5 v7.3) from zip."""
    fname = f"SIDD_Medium_Raw/Data/{scene_dir}/{scene_id}_{suffix}_{img_id}.MAT"
    with zf.open(fname) as f:
        data = f.read()
    h = h5py.File(io.BytesIO(data), "r")
    arr = np.array(h["x"], dtype=np.float32).T  # stored transposed
    h.close()
    return arr


def load_srgb_from_zip(zf, scene_dir, scene_id, suffix, img_id):
    """Load sRGB PNG from zip."""
    fname = f"mnt/d/SIDD_Medium_Srgb/Data/{scene_dir}/{scene_id}_{suffix}_{img_id}.PNG"
    with zf.open(fname) as f:
        return np.array(Image.open(f), dtype=np.float32) / 255.0


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"Opening zip files...")
    raw_zip = zipfile.ZipFile(str(RAW_ZIP))
    srgb_zip = zipfile.ZipFile(str(SRGB_ZIP))

    scene_meta = []

    for si, scene in enumerate(SELECTED_SCENES):
        pat = scene["pat"]
        crop = CROP_FN[pat]
        tag_base = f"{scene['id']}_{scene['cam']}_{pat}"
        print(f"\n[{si+1:2d}/20] {scene['dir']}  ({pat}, ISO {scene['iso']})")

        for img_id in ["010", "011"]:
            tag = f"{tag_base}_{img_id}"

            # Load
            gt_raw = load_raw_from_zip(raw_zip, scene["dir"], scene["id"], "GT_RAW", img_id)
            noisy_raw = load_raw_from_zip(raw_zip, scene["dir"], scene["id"], "NOISY_RAW", img_id)
            gt_srgb = load_srgb_from_zip(srgb_zip, scene["dir"], scene["id"], "GT_SRGB", img_id)
            noisy_srgb = load_srgb_from_zip(srgb_zip, scene["dir"], scene["id"], "NOISY_SRGB", img_id)

            # Crop to RGGB
            gt_raw_c = crop(gt_raw)
            noisy_raw_c = crop(noisy_raw)
            gt_srgb_c = crop(gt_srgb)
            noisy_srgb_c = crop(noisy_srgb)

            # Verify RGGB: green pair on anti-diagonal
            s01 = gt_raw_c[0::2, 1::2].mean()
            s10 = gt_raw_c[1::2, 0::2].mean()
            s00 = gt_raw_c[0::2, 0::2].mean()
            s11 = gt_raw_c[1::2, 1::2].mean()
            assert abs(s01 - s10) < abs(s00 - s11), f"RGGB check failed for {tag}"

            # Save
            np.save(str(OUTDIR / f"{tag}_gt_raw.npy"), gt_raw_c)
            np.save(str(OUTDIR / f"{tag}_noisy_raw.npy"), noisy_raw_c)
            np.save(str(OUTDIR / f"{tag}_gt_srgb.npy"), gt_srgb_c)
            np.save(str(OUTDIR / f"{tag}_noisy_srgb.npy"), noisy_srgb_c)

            H, W = gt_raw_c.shape
            print(f"    {img_id}: {W}×{H} → {tag}")

            meta = {
                "tag": tag,
                "scene_dir": scene["dir"],
                "scene_id": scene["id"],
                "cam": scene["cam"],
                "pattern": pat,
                "iso": scene["iso"],
                "ct": scene["ct"],
                "light": scene["light"],
                "img_id": img_id,
                "width": W,
                "height": H,
            }
            scene_meta.append(meta)

    raw_zip.close()
    srgb_zip.close()

    # Save metadata
    meta_path = OUTDIR / "scenes.json"
    with open(str(meta_path), "w") as f:
        json.dump(scene_meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Done! {len(scene_meta)} images saved to {OUTDIR}/")
    print(f"Metadata → {meta_path}")

    # Summary by camera
    from collections import Counter
    cam_counts = Counter(m["cam"] for m in scene_meta)
    for cam, n in sorted(cam_counts.items()):
        print(f"  {cam}: {n} images")


if __name__ == "__main__":
    main()
