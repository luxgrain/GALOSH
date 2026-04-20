#!/usr/bin/env python3
"""
Prepare additional 20 SIDD Medium scenes with maximum diversity.

Prioritizes: different scene instances, rare ISOs (50/200/640/1000/2000/10000),
varied color temperatures, and H lighting.

Appends to existing datasets/sidd/medium_bench/ and updates scenes.json.
"""
import zipfile, io, json, h5py
import numpy as np
from PIL import Image
from pathlib import Path

BASE = Path(__file__).parent.parent
RAW_ZIP  = BASE / "datasets" / "sidd" / "SIDD_Medium_Raw.zip"
SRGB_ZIP = BASE / "datasets" / "sidd" / "SIDD_Medium_Srgb.zip"
OUTDIR   = BASE / "datasets" / "sidd" / "medium_bench"

# Additional 20 scenes: maximally diverse from original 20
ADDITIONAL_SCENES = [
    # G4 — instances 006,007,008 (original used 001-005). ISO 200/400/800 + CT 4400
    {"dir": "0078_004_G4_00200_00050_3200_N", "id": "0078", "cam": "G4", "pat": "BGGR", "iso": 200,   "ct": 3200, "light": "N"},
    {"dir": "0147_007_G4_00100_00100_4400_L", "id": "0147", "cam": "G4", "pat": "BGGR", "iso": 100,   "ct": 4400, "light": "L"},
    {"dir": "0149_007_G4_00800_00800_4400_L", "id": "0149", "cam": "G4", "pat": "BGGR", "iso": 800,   "ct": 4400, "light": "L"},
    {"dir": "0123_006_G4_00400_00160_3200_N", "id": "0123", "cam": "G4", "pat": "BGGR", "iso": 400,   "ct": 3200, "light": "N"},

    # GP — ISO 50(!!), 3200, 6400, 10000(!!) — extreme range
    {"dir": "0083_004_GP_00050_00020_4400_N", "id": "0083", "cam": "GP", "pat": "BGGR", "iso": 50,    "ct": 4400, "light": "N"},
    {"dir": "0182_008_GP_03200_03200_5500_N", "id": "0182", "cam": "GP", "pat": "BGGR", "iso": 3200,  "ct": 5500, "light": "N"},
    {"dir": "0108_005_GP_06400_06400_4400_N", "id": "0108", "cam": "GP", "pat": "BGGR", "iso": 6400,  "ct": 4400, "light": "N"},
    {"dir": "0065_003_GP_10000_08460_4400_N", "id": "0065", "cam": "GP", "pat": "BGGR", "iso": 10000, "ct": 4400, "light": "N"},

    # IP — ISO 640, 1000, 1600, 2000 — rare ISOs from different instances
    {"dir": "0092_004_IP_00640_00125_3200_L", "id": "0092", "cam": "IP", "pat": "RGGB", "iso": 640,   "ct": 3200, "light": "L"},
    {"dir": "0069_003_IP_01000_02000_3200_N", "id": "0069", "cam": "IP", "pat": "RGGB", "iso": 1000,  "ct": 3200, "light": "N"},
    {"dir": "0191_008_IP_01600_01600_3200_N", "id": "0191", "cam": "IP", "pat": "RGGB", "iso": 1600,  "ct": 3200, "light": "N"},
    {"dir": "0070_003_IP_02000_04000_3200_N", "id": "0070", "cam": "IP", "pat": "RGGB", "iso": 2000,  "ct": 3200, "light": "N"},

    # N6 — different instances (001,003,005,006), ISO 100-3200
    {"dir": "0023_001_N6_00800_00350_5500_N", "id": "0023", "cam": "N6", "pat": "BGGR", "iso": 800,   "ct": 5500, "light": "N"},
    {"dir": "0054_003_N6_00100_00160_5500_N", "id": "0054", "cam": "N6", "pat": "BGGR", "iso": 100,   "ct": 5500, "light": "N"},
    {"dir": "0097_005_N6_03200_02000_3200_L", "id": "0097", "cam": "N6", "pat": "BGGR", "iso": 3200,  "ct": 3200, "light": "L"},
    {"dir": "0120_006_N6_01600_00400_3200_L", "id": "0120", "cam": "N6", "pat": "BGGR", "iso": 1600,  "ct": 3200, "light": "L"},

    # S6 — instances 002,003,004,008, ISO 800-3200 + CT 4400
    {"dir": "0052_002_S6_01600_01000_5500_N", "id": "0052", "cam": "S6", "pat": "GRBG", "iso": 1600,  "ct": 5500, "light": "N"},
    {"dir": "0062_003_S6_03200_02500_4400_L", "id": "0062", "cam": "S6", "pat": "GRBG", "iso": 3200,  "ct": 4400, "light": "L"},
    {"dir": "0081_004_S6_00800_00160_4400_L", "id": "0081", "cam": "S6", "pat": "GRBG", "iso": 800,   "ct": 4400, "light": "L"},
    {"dir": "0179_008_S6_03200_00800_5500_L", "id": "0179", "cam": "S6", "pat": "GRBG", "iso": 3200,  "ct": 5500, "light": "L"},
]

# Same crop functions as prepare_sidd_medium.py
def crop_rggb(arr):
    return arr[:-2, :-2] if arr.ndim == 2 else arr[:-2, :-2, :]

def crop_grbg(arr):
    return arr[:-2, 1:-1] if arr.ndim == 2 else arr[:-2, 1:-1, :]

def crop_bggr(arr):
    return arr[1:-1, 1:-1] if arr.ndim == 2 else arr[1:-1, 1:-1, :]

CROP_FN = {"RGGB": crop_rggb, "GRBG": crop_grbg, "BGGR": crop_bggr}


def load_raw_from_zip(zf, scene_dir, scene_id, suffix, img_id):
    fname = f"SIDD_Medium_Raw/Data/{scene_dir}/{scene_id}_{suffix}_{img_id}.MAT"
    with zf.open(fname) as f:
        data = f.read()
    h = h5py.File(io.BytesIO(data), "r")
    arr = np.array(h["x"], dtype=np.float32).T
    h.close()
    return arr


def load_srgb_from_zip(zf, scene_dir, scene_id, suffix, img_id):
    fname = f"mnt/d/SIDD_Medium_Srgb/Data/{scene_dir}/{scene_id}_{suffix}_{img_id}.PNG"
    with zf.open(fname) as f:
        return np.array(Image.open(f), dtype=np.float32) / 255.0


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Load existing metadata
    meta_path = OUTDIR / "scenes.json"
    existing_meta = []
    if meta_path.exists():
        with open(str(meta_path)) as f:
            existing_meta = json.load(f)
    existing_tags = {m["tag"] for m in existing_meta}

    print(f"Opening zip files...")
    raw_zip = zipfile.ZipFile(str(RAW_ZIP))
    srgb_zip = zipfile.ZipFile(str(SRGB_ZIP))

    new_meta = []

    for si, scene in enumerate(ADDITIONAL_SCENES):
        pat = scene["pat"]
        crop = CROP_FN[pat]
        tag_base = f"{scene['id']}_{scene['cam']}_{pat}"
        print(f"\n[{si+1:2d}/20] {scene['dir']}  ({pat}, ISO {scene['iso']})")

        for img_id in ["010", "011"]:
            tag = f"{tag_base}_{img_id}"
            if tag in existing_tags:
                print(f"    {img_id}: SKIP (already exists)")
                continue

            gt_raw = load_raw_from_zip(raw_zip, scene["dir"], scene["id"], "GT_RAW", img_id)
            noisy_raw = load_raw_from_zip(raw_zip, scene["dir"], scene["id"], "NOISY_RAW", img_id)
            gt_srgb = load_srgb_from_zip(srgb_zip, scene["dir"], scene["id"], "GT_SRGB", img_id)
            noisy_srgb = load_srgb_from_zip(srgb_zip, scene["dir"], scene["id"], "NOISY_SRGB", img_id)

            gt_raw_c = crop(gt_raw)
            noisy_raw_c = crop(noisy_raw)
            gt_srgb_c = crop(gt_srgb)
            noisy_srgb_c = crop(noisy_srgb)

            # Verify RGGB
            s01 = gt_raw_c[0::2, 1::2].mean()
            s10 = gt_raw_c[1::2, 0::2].mean()
            s00 = gt_raw_c[0::2, 0::2].mean()
            s11 = gt_raw_c[1::2, 1::2].mean()
            assert abs(s01 - s10) < abs(s00 - s11), f"RGGB check failed for {tag}"

            np.save(str(OUTDIR / f"{tag}_gt_raw.npy"), gt_raw_c)
            np.save(str(OUTDIR / f"{tag}_noisy_raw.npy"), noisy_raw_c)
            np.save(str(OUTDIR / f"{tag}_gt_srgb.npy"), gt_srgb_c)
            np.save(str(OUTDIR / f"{tag}_noisy_srgb.npy"), noisy_srgb_c)

            H, W = gt_raw_c.shape
            print(f"    {img_id}: {W}x{H} -> {tag}")

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
            new_meta.append(meta)

    raw_zip.close()
    srgb_zip.close()

    # Merge and save metadata
    all_meta = existing_meta + new_meta
    with open(str(meta_path), "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Added {len(new_meta)} images (total now: {len(all_meta)})")
    print(f"Metadata -> {meta_path}")

    from collections import Counter
    cam_counts = Counter(m["cam"] for m in all_meta)
    for cam, n in sorted(cam_counts.items()):
        print(f"  {cam}: {n} images")


if __name__ == "__main__":
    main()
