"""Verify the SPL draft's Table I / II cells against the benchmark JSONs.
Set SPL_TEX=galosh_spl_ja.tex to check the Japanese review copy."""
import json
import re
import sys
from pathlib import Path
import numpy as np

GALOSH = Path(r"C:\Users\luxgrain\GALOSH")
sys.path.insert(0, str(GALOSH / "benchmark" / "scripts"))
import verify_table_numbers as V

import os as _os
_tex = _os.environ.get("SPL_TEX", "galosh_spl.tex")
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
means2 = V.json_means(GALOSH / "benchmark" / "results_srgb_sidd_full" / "_metrics.json")
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

print(f"RESULT: {'FAIL' if bad else 'PASS'} - {checked} cells checked, {bad} mismatches")
sys.exit(1 if bad else 0)
