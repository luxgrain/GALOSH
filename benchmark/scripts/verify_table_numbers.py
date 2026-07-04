"""NUMERIC verification of the paper's sRGB result tables against the benchmark
JSONs (complements check_paper_consistency.py, which is structural only).

Recomputes per-method means from benchmark/results_srgb_sidd_full/_metrics.json
and benchmark/results_srgb_rawnind/_metrics.json and compares them with the
values printed in docs/paper/galosh_srgb_tables.tex (PSNR/SSIM/LPIPS/DISTS/NIQE,
tolerance = rounding of the table's decimals). Exits 1 on any mismatch.

  python benchmark/scripts/verify_table_numbers.py
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("GALOSH_ROOT", str(Path(__file__).resolve().parents[2])))
TEX = ROOT / "docs" / "paper" / "galosh_srgb_tables.tex"

# tex row label -> metrics.json method key
METHODS = {
    "Noisy input": "_noisy",
    "GALOSH-YUV (ours)": "galosh_yuv_gpu_fp32",
    "CBM3D": "cbm3d",
    "Color-NLM": "color_nlm",
    "Guided filter": "guided",
    "NAFNet-SIDD": "nafnet",
    "Restormer-SIDD": "restormer",
    "SCUNet-real": "scunet",
}
TABLES = [
    ("tab:sidd_srgb", ROOT / "benchmark" / "results_srgb_sidd_full" / "_metrics.json"),
    ("tab:rawnind_srgb", ROOT / "benchmark" / "results_srgb_rawnind" / "_metrics.json"),
]
COLS = ["psnr", "ssim", "lpips", "dists", "niqe"]


def json_means(path):
    rows = json.load(open(path))
    agg = {}
    for r in rows:
        if r.get("ok") is False:
            continue
        agg.setdefault(r["method"], []).append(r)
    out = {}
    for m, rs in agg.items():
        out[m] = {k: float(np.mean([x[k] for x in rs if x.get(k) is not None])) for k in COLS}
    return out


def tex_tables(src):
    """label -> {row label -> [5 numeric strings]} for each table env."""
    out = {}
    for block in re.split(r"\\begin\{table\}", src)[1:]:
        mlab = re.search(r"\\label\{([^}]+)\}", block)
        if not mlab:
            continue
        rows = {}
        for line in block.splitlines():
            if "&" not in line or not line.strip().startswith(("\\quad", "Noisy")):
                continue
            cells = [re.sub(r"\\(textbf|underline)\{([^}]*)\}", r"\2", c) for c in line.split("&")]
            label = re.sub(r"\\(quad|textbf)\s*\{?", "", cells[0]).replace("}", "").strip()
            nums = []
            for c in cells[1:6]:
                mnum = re.search(r"-?\d+\.\d+", c)
                nums.append(mnum.group(0) if mnum else None)
            rows[label] = nums
        out[mlab.group(1)] = rows
    return out


def main():
    src = TEX.read_text(encoding="utf-8")
    tables = tex_tables(src)
    bad = 0
    for label, jpath in TABLES:
        if not jpath.exists():
            print(f"[{label}] SKIP - {jpath} not present (results not on this machine)")
            continue
        means = json_means(jpath)
        rows = tables.get(label, {})
        for texname, key in METHODS.items():
            if texname not in rows or key not in means:
                continue
            for col, cell in zip(COLS, rows[texname]):
                if cell is None:
                    continue
                # tolerance = half a unit in the table's last printed decimal
                tol = 0.5 * 10 ** -len(cell.split(".")[1])
                got = means[key][col]
                if abs(got - float(cell)) > tol + 1e-9:
                    print(f"[{label}] {texname} {col}: tex={cell} json={got:.4f}  MISMATCH")
                    bad += 1
    print("RESULT:", "FAIL" if bad else "PASS - table numbers match the benchmark JSONs")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
