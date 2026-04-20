#!/usr/bin/env python3
"""
Reorganize legacy Kodak/McMaster/CBSD68 benchmark data into per-method layout.

Target layout:
  benchmark/<dataset>/<method>/metrics.json
  benchmark/<dataset>/<method>/images/iso{ISO}/{img_id}.png
  benchmark/<dataset>/__gt__/iso{ISO}/{img_id}.png
  benchmark/<dataset>/__noisy__/iso{ISO}/{img_id}.png

Sources:
  benchmark/<dataset>/galosh_f/metrics.json    (current, multi-method wide)
  benchmark/<dataset>/images/iso*/             (current, multi-method PNG)
  benchmark/archive/results_legacy/v*.json     (legacy versioned JSONs)
  benchmark/archive/images_legacy/comparison_images_v{5,6,7}_<ds>/iso*/

GALOSH evolution is preserved as galosh_v{4,5,6,6b,6c,6e,6lg05_10,6lg10_10,
7a_05_10,7a_10_10,7b_05_10,7b_10_10,f,f_05_10,f_full,f_half} so each
algorithmic generation can be compared.
"""
from pathlib import Path
import json
import shutil
import argparse

BASE = Path(r"C:\Users\luxgrain\GALOSH\benchmark")
ARCHIVE_J = BASE / "archive" / "results_legacy"
ARCHIVE_I = BASE / "archive" / "images_legacy"

DATASETS = ["kodak", "mcmaster", "cbsd68"]

# Map source method name (from JSON keys / PNG filenames) → canonical dir name
METHOD_MAP = {
    "DnCNN-B":             "dncnn_b",
    "DRUNet":              "drunet",
    "BM3D-CFA":            "bm3d_cfa",
    "BM3D-PC":             "bm3d_pc",
    "RAW_L-C_BM3D":        "raw_lc_bm3d",
    "CBM3D":               "cbm3d",
    "NAFNet":              "nafnet",
    "OursV2":              "galosh_v2",
    "Ours":                "galosh_v4",
    "GALOSH":              "galosh_v6",
    "GALOSH_B":            "galosh_v6b",
    "BM3D-CFA_(s=0.5)":    "bm3d_cfa_s05",
    "No-denoise":          "__noisy__",
    "Noisy":               "__noisy__",
    "GALOSH_F_10_10":      "galosh_f",
    "GALOSH_F_05_10":      "galosh_f_05_10",
    "GALOSH_F_FULL_10_10": "galosh_f_full",
    "GALOSH_F_HALF_10_10": "galosh_f_half",
    "GALOSH_A_05_10":      "galosh_v7a_05_10",
    "GALOSH_A_10_10":      "galosh_v7a_10_10",
    "GALOSH_B_05_10":      "galosh_v7b_05_10",
    "GALOSH_B_10_10":      "galosh_v7b_10_10",
    # numbered prefix variants from v2
    "1_BM3D-PC":           "bm3d_pc",
    "2_BM3D-CFA":          "bm3d_cfa",
    "3_DnCNN-B":           "dncnn_b",
    "3_CBM3D":             "cbm3d",
    "4_DRUNet":            "drunet",
    "5_NAFNet":            "nafnet",
    "6_Ours":              "galosh_v4",
    "6_OursV2":            "galosh_v2",
}

# Special tags for PNG filename suffixes — case variants seen across versions.
GT_TAGS = ("GT", "gt")
NOISY_TAGS = ("Noisy", "noisy")

# v_<dataset>_results.json files contain a single GALOSH variant whose
# name matches the file stem prefix.  Map JSON file → variant.
GALOSH_VARIANT_FROM_FILE = {
    # v5/v6 are multi-method wide JSONs (No-denoise + BM3D-CFA + DnCNN-B
    # + DRUNet + GALOSH).  Handled by wide-split fallback; do NOT list them
    # here, otherwise the single-variant route mislabels their metrics.
    "v6b":        "galosh_v6b",
    "v6c":        "galosh_v6c",
    "v6e":        "galosh_v6e",
    "v6lg05_10":  "galosh_v6lg05_10",
    "v6lg10_10":  "galosh_v6lg10_10",
    "v7a_05_10":  "galosh_v7a_05_10",
    "v7a_10_10":  "galosh_v7a_10_10",
    "v7b_05_10":  "galosh_v7b_05_10",
    "v7b_10_10":  "galosh_v7b_10_10",
    "v8_f_05_10": "galosh_f_05_10",
}

# JSON files at archive root that are kodak-only (verified by metric match)
KODAK_GLOBAL_JSONS = ["v4_benchmark_results.json"]

METRIC_KEYS = ("psnr", "ssim", "lpips", "dists", "niqe")


def split_wide_json(json_path: Path, ds_dir: Path, dry_run: bool):
    """Split a wide-format JSON (one row per iso, columns = method_metric).

    Returns list of (source_method, canonical_dir, n_rows).
    """
    if not json_path.exists():
        return []
    try:
        data = json.load(open(json_path))
    except Exception as e:
        print(f"    !! parse error {json_path.name}: {e}")
        return []
    # Some legacy JSONs are dict-of-list, e.g. {"quality": [...], "speed": [...]}
    # — extract the per-iso row list ("quality" by convention).
    if isinstance(data, dict):
        data = data.get("quality") or next(
            (v for v in data.values() if isinstance(v, list)), None)
    if not isinstance(data, list) or not data:
        return []

    # Discover methods present.  Sort suffix candidates longest-first so
    # method names like "GALOSH_F_FULL_10_10" don't get truncated by a
    # shorter prefix like "GALOSH_F".
    methods_seen = {}
    metric_set = set()
    for row in data:
        if not isinstance(row, dict):
            continue
        for k in row:
            if k in ("iso", "time_per_img", "dataset", "n_images",
                     "alpha_err_pct", "sq_err_pct"):
                continue
            for m in METRIC_KEYS:
                if k.endswith(f"_{m}"):
                    name = k[:-(len(m) + 1)]
                    methods_seen.setdefault(name, set()).add(m)
                    metric_set.add(m)
                    break

    written = []
    for src_name in methods_seen:
        canonical = METHOD_MAP.get(src_name)
        if canonical is None:
            # fallback: lowercase, '-' → '_'
            canonical = src_name.lower().replace("-", "_").replace(" ", "_")
        if canonical.startswith("__"):
            # GT / noisy don't have metrics, skip writing JSON
            continue
        out_dir = ds_dir / canonical
        out_path = out_dir / "metrics.json"
        rows = []
        for row in data:
            entry = {"iso": row.get("iso")}
            if "time_per_img" in row:
                entry["time_per_img"] = row["time_per_img"]
            for m in METRIC_KEYS:
                k = f"{src_name}_{m}"
                if k in row:
                    entry[m] = row[k]
            rows.append(entry)
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(rows, f, indent=2)
        written.append((src_name, canonical, len(rows)))
    return written


def split_images(src_iso_dirs, ds_dir: Path, dry_run: bool, allow_overwrite: bool):
    """Parse {img_id}_{method}.png → ds/<method>/images/iso{ISO}/{img_id}.png

    src_iso_dirs is iterated in order; later sources OVERWRITE earlier when
    allow_overwrite=True (so latest data wins).
    """
    counts = {}
    # Sort method keys longest-first so suffix matching prefers e.g.
    # GALOSH_F_FULL_10_10 over GALOSH_F.
    suffix_keys = sorted(list(METHOD_MAP.keys()) + list(GT_TAGS) + list(NOISY_TAGS),
                         key=len, reverse=True)
    for iso_dir in src_iso_dirs:
        if not iso_dir.exists() or not iso_dir.is_dir():
            continue
        iso_name = iso_dir.name
        for png in iso_dir.glob("*.png"):
            stem = png.stem
            method_part = None
            img_id = None
            for sk in suffix_keys:
                if stem.endswith(f"_{sk}"):
                    method_part = sk
                    img_id = stem[:-(len(sk) + 1)]
                    break
            if method_part is None:
                # Skip unknown filenames silently for now
                continue
            if method_part in GT_TAGS:
                tgt_method = "__gt__"
            elif method_part in NOISY_TAGS:
                tgt_method = "__noisy__"
            else:
                tgt_method = METHOD_MAP.get(method_part, method_part.lower())
            tgt_iso_dir = ds_dir / tgt_method / "images" / iso_name
            tgt_path = tgt_iso_dir / f"{img_id}.png"
            if (not allow_overwrite) and tgt_path.exists():
                continue
            if not dry_run:
                tgt_iso_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(png, tgt_path)
            counts.setdefault(tgt_method, 0)
            counts[tgt_method] += 1
    return counts


def reorganize_dataset(dataset: str, dry_run: bool):
    ds_dir = BASE / dataset
    print(f"\n=== {dataset} ===")

    # Step 1: split legacy v* JSONs first (so latest sources can overwrite).
    # Use a set to dedupe — both glob patterns can match the same file.
    legacy_files = sorted(set(ARCHIVE_J.glob(f"v*_{dataset}_results.json"))
                          | set(ARCHIVE_J.glob(f"v*_*_*_{dataset}_results.json")))
    for vfile in legacy_files:
        # Extract version tag: e.g. v7a_10_10_kodak_results.json -> v7a_10_10
        stem = vfile.stem  # without .json
        # Remove trailing _<dataset>_results
        prefix = stem.rsplit(f"_{dataset}_results", 1)[0]
        canonical = GALOSH_VARIANT_FROM_FILE.get(prefix)
        if canonical is None:
            print(f"  ?? unknown variant tag for {vfile.name}, fallback to JSON keys")
            results = split_wide_json(vfile, ds_dir, dry_run)
            for src, can, n in results:
                print(f"    {src} -> {can}/metrics.json ({n} iso)")
        else:
            # File contains a single variant; rename whatever method key is
            # present to the canonical galosh_<variant> dir.
            try:
                data = json.load(open(vfile))
            except Exception as e:
                print(f"    !! {vfile.name} parse: {e}")
                continue
            # v5/v6 wrap rows under "quality" key.
            if isinstance(data, dict):
                data = data.get("quality") or next(
                    (v for v in data.values() if isinstance(v, list)), [])
            if not isinstance(data, list):
                data = []
            rows = []
            for row in data:
                if not isinstance(row, dict):
                    continue
                entry = {"iso": row.get("iso")}
                if "time_per_img" in row:
                    entry["time_per_img"] = row["time_per_img"]
                # Find any metric key
                for k, v in row.items():
                    if k in ("iso", "time_per_img"):
                        continue
                    for m in METRIC_KEYS:
                        if k.endswith(f"_{m}"):
                            entry[m] = v
                            break
                rows.append(entry)
            out_dir = ds_dir / canonical
            out_path = out_dir / "metrics.json"
            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(rows, f, indent=2)
            print(f"  {vfile.name} -> {canonical}/metrics.json ({len(rows)} iso)")

    # Step 1b: kodak-only global JSON (v4)
    if dataset == "kodak":
        for fname in KODAK_GLOBAL_JSONS:
            jp = ARCHIVE_J / fname
            if jp.exists():
                results = split_wide_json(jp, ds_dir, dry_run)
                print(f"  {fname}: {len(results)} methods")
                for src, can, n in results:
                    print(f"    {src} -> {can}/metrics.json ({n} iso)")

    # Step 2: split current galosh_f/metrics.json (multi-method wide)
    cur_metrics = ds_dir / "galosh_f" / "metrics.json"
    if cur_metrics.exists():
        results = split_wide_json(cur_metrics, ds_dir, dry_run)
        print(f"  current {cur_metrics.relative_to(BASE)}: {len(results)} methods")
        for src, can, n in results:
            print(f"    {src} -> {can}/metrics.json ({n} iso)")

    # Step 3: split images.  Order: legacy v5/v6/v7 → current (latest wins).
    src_iso_dirs = []
    for v in ["v5", "v6", "v7"]:
        arch = ARCHIVE_I / f"comparison_images_{v}_{dataset}"
        if arch.exists():
            src_iso_dirs.extend(sorted(arch.glob("iso*")))
    cur_images = ds_dir / "images"
    if cur_images.exists():
        src_iso_dirs.extend(sorted(cur_images.glob("iso*")))
    counts = split_images(src_iso_dirs, ds_dir, dry_run, allow_overwrite=True)
    print(f"  images split: {len(counts)} methods")
    for m, n in sorted(counts.items()):
        print(f"    {m}: {n} png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    args = ap.parse_args()

    for ds in args.datasets:
        reorganize_dataset(ds, dry_run=args.dry_run)

    print("\nDone." + (" (dry-run, no files written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
