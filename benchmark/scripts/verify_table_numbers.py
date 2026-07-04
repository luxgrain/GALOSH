"""NUMERIC verification of the paper's result tables against the benchmark
JSONs (complements check_paper_consistency.py, which is structural only).

sRGB tables: recomputes per-method means from
benchmark/results_srgb_sidd_full/_metrics.json and
benchmark/results_srgb_rawnind/_metrics.json and compares them with
docs/paper/galosh_srgb_tables.tex.

RAW tables: recomputes per-method means from the raw campaign JSONs
(benchmark/results_raw_sidd/ for SIDD; RawNIND results dir from the
GALOSH_RAWNIND_RESULTS env var, else benchmark/results_raw_rawnind/) and
compares them with docs/paper/galosh_raw_tables.tex AND _ja.tex.

Tolerance = half a unit in the table's last printed decimal.

Result semantics (exit code):
  PASS (0)        every expected table had its JSONs present and ALL cells match.
  FAIL (1)        at least one cell mismatches its JSON mean.
  INCOMPLETE (2)  nothing (or only part) could be verified because benchmark
                  JSONs are missing on this machine -- e.g. a clean checkout or
                  the public tarball, which does not ship result JSONs. The
                  skipped tables are listed; INCOMPLETE is NOT a pass.

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


# ---------------------------------------------------------------- RAW tables
# raw campaign JSONs: {method: {scene: {psnr, ssim, lpips, dists, niqe, dt}}}
# row-label substring -> (metrics file, method key); file choice mirrors the
# provenance comments in galosh_raw_tables.tex.
RAW_METHODS = [
    ("GALOSH FP32",      "_metrics_antiring.json", "galosh_fp32"),
    ("GALOSH INT16",     "_metrics_antiring.json", "galosh_int16"),
    ("VST+GALOSH core",  "_metrics_revst_gpu.json", "vst_galosh"),
    ("VST+BM3D-CFA",     "_metrics_bm3d.json",     "vst_bm3d_cfa"),
    ("VST+NLM-CFA",      "_metrics_fast.json",     "vst_nlm_cfa"),
    ("BM3D-CFA",         "_metrics_bm3d.json",     "bm3d_cfa"),
    ("NLM-CFA",          "_metrics_fast.json",     "nlm_cfa"),
    ("Blind2Unblind",    "_metrics_fast.json",     "b2u"),
]
RAW_TEXES = [ROOT / "docs" / "paper" / "galosh_raw_tables.tex",
             ROOT / "docs" / "paper" / "galosh_raw_tables_ja.tex"]
RAW_DIRS = [
    ("raw_sidd", ROOT / "benchmark" / "results_raw_sidd"),
    ("raw_rawnind", Path(os.environ.get("GALOSH_RAWNIND_RESULTS",
                         str(ROOT / "benchmark" / "results_raw_rawnind")))),
]


def raw_json_means(jdir, fname, key):
    p = jdir / fname
    if not p.exists():
        return None
    d = json.load(open(p)).get(key)
    if not d:
        return None
    rows = [v for v in d.values() if isinstance(v, dict) and v.get("psnr") is not None]
    return {k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in COLS}


def raw_rows(block):
    """row-label -> [5 numeric strings] from one tabular block (\\quad rows)."""
    rows = {}
    for line in block.splitlines():
        if "&" not in line or not line.strip().startswith("\\quad"):
            continue
        cells = [re.sub(r"\\(textbf|underline)\{([^}]*)\}", r"\2", c) for c in line.split("&")]
        # normalize label: drop \quad, math/dagger decorations, thin spaces
        label = re.sub(r"\\quad|\\,|\$[^$]*\$|\\|[{}~]|\s", "", cells[0])
        nums = []
        for c in cells[1:6]:
            mnum = re.search(r"-?\d+\.\d+", c)
            nums.append(mnum.group(0) if mnum else None)
        rows[label] = nums
    return rows


def verify_raw():
    bad = checked = 0
    skipped = []
    for tex in RAW_TEXES:
        src = tex.read_text(encoding="utf-8")
        blocks = re.split(r"\\begin\{table\}", src)[1:]
        if len(blocks) < 2:
            print(f"[{tex.name}] SKIP - expected >=2 table envs")
            skipped.append(f"{tex.name} (unparseable)")
            continue
        # table order in both files: SIDD Medium first, RawNIND second
        for (tag, jdir), block in zip(RAW_DIRS, blocks[:2]):
            if not jdir.exists():
                print(f"[{tex.name}/{tag}] SKIP - {jdir} not present")
                skipped.append(f"{tex.name}/{tag}")
                continue
            rows = raw_rows(block)
            for disp, fname, key in RAW_METHODS:
                want = re.sub(r"\s", "", disp)   # exact match on normalized label
                hit = rows.get(want)
                if hit is None:
                    continue
                means = raw_json_means(jdir, fname, key)
                if means is None:
                    print(f"[{tex.name}/{tag}] {disp}: SKIP - {fname}:{key} missing")
                    skipped.append(f"{tex.name}/{tag}:{disp}")
                    continue
                for col, cell in zip(COLS, hit):
                    if cell is None:
                        continue
                    tol = 0.5 * 10 ** -len(cell.split(".")[1])
                    got = means[col]
                    checked += 1
                    if abs(got - float(cell)) > tol + 1e-9:
                        print(f"[{tex.name}/{tag}] {disp} {col}: tex={cell} json={got:.4f}  MISMATCH")
                        bad += 1
    return bad, checked, skipped


def main():
    src = TEX.read_text(encoding="utf-8")
    tables = tex_tables(src)
    bad = checked = 0
    skipped = []
    for label, jpath in TABLES:
        if not jpath.exists():
            print(f"[{label}] SKIP - {jpath} not present (results not on this machine)")
            skipped.append(label)
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
                checked += 1
                if abs(got - float(cell)) > tol + 1e-9:
                    print(f"[{label}] {texname} {col}: tex={cell} json={got:.4f}  MISMATCH")
                    bad += 1
    rbad, rchecked, rskipped = verify_raw()
    bad += rbad; checked += rchecked; skipped += rskipped

    # PASS only when every expected table verified with zero mismatches.
    if bad:
        print(f"RESULT: FAIL - {bad} mismatched cell(s) of {checked} verified")
        sys.exit(1)
    if checked == 0:
        print("RESULT: INCOMPLETE - no benchmark JSONs found; 0 cells verified "
              "(expected on a clean checkout: result JSONs are not shipped)")
        sys.exit(2)
    if skipped:
        print(f"RESULT: INCOMPLETE - {checked} cell(s) verified OK, but skipped: "
              + ", ".join(skipped))
        sys.exit(2)
    print(f"RESULT: PASS - all {checked} table cells match the benchmark JSONs")
    sys.exit(0)


if __name__ == "__main__":
    main()
