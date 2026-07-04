"""Verify the SPL draft's Table I / II cells against the benchmark JSONs.
Set SPL_TEX=galosh_spl_ja.tex to check the Japanese review copy.

Result semantics mirror verify_table_numbers.py: PASS (exit 0) only when the
benchmark JSONs are present and every cell matches; on a clean checkout
(result JSONs are not shipped) this reports INCOMPLETE with exit 2."""
import json
import os
import re
import sys
from pathlib import Path
import numpy as np

GALOSH = Path(os.environ.get("GALOSH_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(GALOSH / "benchmark" / "scripts"))
import verify_table_numbers as V

_tex = os.environ.get("SPL_TEX", "galosh_spl.tex")
src = (GALOSH / "docs" / "paper" / "spl" / _tex).read_text(encoding="utf-8")
blocks = re.split(r"\\begin\{table\}", src)[1:]

# ---- Table I: raw, two datasets side by side ----
RAW_MAP = [
    ("GALOSH FP32", "_metrics_antiring.json", "galosh_fp32"),
    ("GALOSH INT16", "_metrics_antiring.json", "galosh_int16"),
    ("VST+GALOSH", "_metrics_revst_gpu.json", "vst_galosh"),
    ("VST+BM3D-CFA", "_metrics_bm3d.json", "vst_bm3d_cfa"),
    ("VST+NLM-CFA", "_metrics_fast.json", "vst_nlm_cfa"),
    ("BM3D-CFA", "_metrics_bm3d.json", "bm3d_cfa"),
    ("NLM-CFA", "_metrics_fast.json", "nlm_cfa"),
    ("Blind2Unblind", "_metrics_fast.json", "b2u"),
]
DIRS = {"sidd": GALOSH / "benchmark" / "results_raw_sidd",
        "rawnind": GALOSH / "benchmark" / "results_raw_rawnind"}
bad = checked = 0
skipped = []
for line in blocks[0].splitlines():
    if "&" not in line or line.strip().startswith(("%", "\\", "Method", " &")):
        continue
    cells = [re.sub(r"\\(textbf|underline)\{([^}]*)\}", r"\2", c).strip()
             for c in line.rstrip("\\").split("&")]
    label = re.sub(r"\s", "", cells[0])
    hit = next(((d, f, k) for d, f, k in RAW_MAP if re.sub(r"\s", "", d) == label), None)
    if not hit:
        continue
    _, fname, key = hit
    nums = [re.search(r"-?\d+\.\d+", c).group(0) for c in cells[1:7]]
    for half, cols in (("sidd", nums[:3]), ("rawnind", nums[3:])):
        means = V.raw_json_means(DIRS[half], fname, key)
        if means is None:
            skipped.append(f"TableI/{half}:{hit[0]}")
            continue
        for col, cell in zip(("psnr", "ssim", "lpips"), cols):
            tol = 0.5 * 10 ** -len(cell.split(".")[1])
            checked += 1
            if abs(means[col] - float(cell)) > tol + 1e-9:
                print(f"[TableI/{half}] {hit[0]} {col}: tex={cell} json={means[col]:.4f} MISMATCH")
                bad += 1

# ---- Table II: sRGB SIDD full ----
SRGB_MAP = {"Noisy input": "_noisy", "GALOSH-YUV": "galosh_yuv_gpu_fp32",
            "CBM3D": "cbm3d", "Color-NLM": "color_nlm", "Guided filter": "guided",
            "NAFNet-SIDD": "nafnet", "Restormer-SIDD": "restormer", "SCUNet-real": "scunet"}
_srgb_json = GALOSH / "benchmark" / "results_srgb_sidd_full" / "_metrics.json"
if not _srgb_json.exists():
    print(f"[TableII] SKIP - {_srgb_json} not present")
    skipped.append("TableII (all)")
else:
    means2 = V.json_means(_srgb_json)
    for line in blocks[1].splitlines():
        if "&" not in line or line.strip().startswith(("%", "\\", "Method")):
            continue
        cells = [re.sub(r"\\(textbf|underline)\{([^}]*)\}", r"\2", c).strip()
                 for c in line.rstrip("\\").split("&")]
        key = SRGB_MAP.get(cells[0])
        if not key:
            continue
        nums = [re.search(r"-?\d+\.\d+", c).group(0) for c in cells[1:4]]
        for col, cell in zip(("psnr", "ssim", "lpips"), nums):
            tol = 0.5 * 10 ** -len(cell.split(".")[1])
            checked += 1
            if abs(means2[key][col] - float(cell)) > tol + 1e-9:
                print(f"[TableII] {cells[0]} {col}: tex={cell} json={means2[key][col]:.4f} MISMATCH")
                bad += 1

if bad:
    print(f"RESULT: FAIL - {bad} mismatched cell(s) of {checked} verified")
    sys.exit(1)
if checked == 0 or skipped:
    print(f"RESULT: INCOMPLETE - {checked} cell(s) verified; skipped: "
          + (", ".join(skipped) if skipped else "(none)")
          + " (expected on a clean checkout: result JSONs are not shipped)")
    sys.exit(2)
print(f"RESULT: PASS - all {checked} cells match the benchmark JSONs")
sys.exit(0)
