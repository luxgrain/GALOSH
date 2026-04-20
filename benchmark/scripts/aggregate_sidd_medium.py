#!/usr/bin/env python3
"""
Aggregate 16-method SIDD Medium full-image benchmark results.

EN: Reads every results/sidd_medium_<mid>.json and produces:
    1. Unified 16-method table (PSNR/SSIM/LPIPS/DISTS/NIQE/time).
    2. Category rankings (rule-based / sensor-agnostic DL / sensor-specialised DL).
    3. Per-camera breakdown.
    4. Claim verification:
         - "blind non-learning strongest":
             GALOSH GPU PSNR/LPIPS/NIQE ≥ best rule-based competitor.
         - "sensor-agnostic strongest":
             GALOSH GPU ≥ min over sensor-non-specialised DL.

JP: results/sidd_medium_<mid>.json を全部読み、16 手法を 3 カテゴリに分けて
    ランキングと主張検証を行う。

Usage:
    python scripts/aggregate_sidd_medium.py
    python scripts/aggregate_sidd_medium.py --markdown report.md
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from statistics import mean, stdev

# EN: Windows default cp932 can't print em-dashes / Greek; force UTF-8.
# JP: Windows の cp932 は em-dash / Δ を出せないので stdout を UTF-8 に。
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   line_buffering=True)

BASE = Path(__file__).parent.parent
SIDD_MEDIUM_DIR = BASE / "sidd_medium"

# Mirror bench_sidd_medium.py's DISPLAY_NAMES and categories.
DISPLAY_NAMES = {
    # Pre-demosaic
    "galosh_raw_gpu":       "GALOSH-RAW",
    "nlm_cfa_oracle":       "NLM-CFA (oracle)",
    "bm3d_cfa":             "BM3D-CFA (Bayer)",
    # After-demosaic — GALOSH family (ablation: nogat vs gat)
    "galosh_yuv_gpu":       "GALOSH-YUV",           # NEW: linear Y-GAT + chroma VST
    "galosh_yuv_gpu_old":   "GALOSH-YUV (old)",      # OLD: sRGB YCbCr plain Wiener
    # After-demosaic — DL & classical
    "nafnet":               "NAFNet-SIDD",
    "dncnn_b":              "DnCNN-B",
    "drunet":               "DRUNet",
    "nlm_srgb":             "NLM sRGB CUDA",
    "guided_gpu":           "Guided (kornia GPU)",
    "ffdnet":               "FFDNet",
    "swinir_dn":            "SwinIR-Gaussian",
    "scunet":               "SCUNet",
    "restormer_dn":         "Restormer-Gaussian",
    "bm3d_srgb":            "BM3D sRGB (CPU)",
    "nlm_srgb_cpu":         "NLM sRGB (CPU)",
    "cbm3d":                "CBM3D joint (CPU)",
}

# Category: rule-based (no learning), sensor-agnostic DL (AWGN/practical
# blind), sensor-specialised DL (SIDD-trained NAFNet).
RULE_IDS = ["galosh_raw_gpu", "galosh_yuv_gpu", "galosh_yuv_gpu_old",
            "nlm_cfa_oracle", "bm3d_cfa",
            "cbm3d", "bm3d_srgb", "nlm_srgb", "nlm_srgb_cpu", "guided_gpu"]
DL_AGNOSTIC_IDS = ["dncnn_b", "drunet", "ffdnet", "swinir_dn", "scunet",
                   "restormer_dn"]
DL_SPECIALISED_IDS = ["nafnet"]

# Oracle methods (use GT to estimate sigma) - flagged in the table but
# excluded from the "blind" claim.
ORACLE_IDS = {"nlm_cfa_oracle"}

# Primary PSNR key: GT2 for pre-demosaic methods, GT1 for post.
PRE_IDS = {"galosh_raw_gpu", "nlm_cfa_oracle", "bm3d_cfa"}


def load_method(mid: str) -> list[dict]:
    # New layout: benchmark/sidd_medium/<method>/metrics.json
    p = SIDD_MEDIUM_DIR / mid / "metrics.json"
    if not p.exists():
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return []


def safe_mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def summarise(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for k in ("psnr", "ssim", "lpips", "dists", "niqe", "time_s",
             "isp_psnr", "isp_lpips"):
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        out[k] = mean(vals) if vals else None
    return out


def fmt(v, spec: str = ".3f") -> str:
    if v is None:
        return "-"
    return format(v, spec)


def print_table(title: str, rows: list[tuple], columns: list[tuple[str, str]]):
    """rows: list of tuples matching columns; columns: list of (header, fmt)."""
    print(f"\n### {title}\n")
    widths = [max(len(h), max((len(fmt(r[i], s)) for r in rows), default=0))
              for i, (h, s) in enumerate(columns)]
    header = "  ".join(h.ljust(w) for (h, _), w in zip(columns, widths))
    print(header)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(fmt(r[i], s).ljust(w)
              for i, ((_, s), w) in enumerate(zip(columns, widths))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", type=str, default=None,
                    help="Optional output .md path")
    args = ap.parse_args()

    summaries: dict[str, dict] = {}
    for mid in DISPLAY_NAMES:
        rows = load_method(mid)
        if rows:
            summaries[mid] = summarise(rows)

    # --- 16-method unified table ---
    all_ids = [m for m in DISPLAY_NAMES if m in summaries]
    table = []
    for mid in all_ids:
        s = summaries[mid]
        label = DISPLAY_NAMES[mid]
        if mid in ORACLE_IDS:
            label += " *"
        table.append((label, s["n"], s["psnr"], s["ssim"],
                      s["lpips"], s["dists"], s["niqe"], s["time_s"]))
    cols = [("Method", "<"), ("N", "d"),
            ("PSNR", ".2f"), ("SSIM", ".4f"),
            ("LPIPS", ".4f"), ("DISTS", ".4f"),
            ("NIQE", ".3f"), ("Time(s)", ".2f")]
    print_table("SIDD Medium 16-method unified table", table, cols)
    print("  * = oracle (sigma from GT; not blind)")

    # --- Category rankings ---
    for cat_name, cat_ids in [
        ("Rule-based (no learning)", RULE_IDS),
        ("DL sensor-agnostic (AWGN / practical blind)", DL_AGNOSTIC_IDS),
        ("DL sensor-specialised (SIDD-trained)", DL_SPECIALISED_IDS),
    ]:
        present = [m for m in cat_ids if m in summaries]
        if not present:
            continue
        # Rank by LPIPS asc (primary perceptual), ties by PSNR desc.
        ranked = sorted(
            present,
            key=lambda m: (summaries[m]["lpips"] if summaries[m]["lpips"] is not None
                           else 9.99,
                           -(summaries[m]["psnr"] or -999)),
        )
        rows = []
        for mid in ranked:
            s = summaries[mid]
            label = DISPLAY_NAMES[mid]
            if mid in ORACLE_IDS:
                label += " *"
            rows.append((label, s["n"], s["psnr"], s["lpips"],
                         s["niqe"], s["time_s"]))
        print_table(
            f"Category: {cat_name}  (sorted by LPIPS asc)",
            rows,
            [("Method", "<"), ("N", "d"), ("PSNR", ".2f"),
             ("LPIPS", ".4f"), ("NIQE", ".3f"), ("Time(s)", ".2f")],
        )

    # --- Claim verification ---
    print("\n### Claim verification\n")

    # Use best GALOSH variant for claim verification.
    # Priority: galosh_yuv_gpu (new GAT) > galosh_raw_gpu > galosh_yuv_gpu_old
    g = None
    g_name = None
    for gid in ["galosh_yuv_gpu", "galosh_raw_gpu", "galosh_yuv_gpu_old"]:
        if gid in summaries:
            g = summaries[gid]
            g_name = DISPLAY_NAMES[gid]
            break
    if not g:
        print("  No GALOSH results found - cannot verify claims.")
        return 0
    print(f"  Using {g_name} for claim verification.\n")

    # GALOSH family ablation (if multiple variants present)
    galosh_ids = [gid for gid in ["galosh_raw_gpu", "galosh_yuv_gpu", "galosh_yuv_gpu_old"]
                  if gid in summaries]
    if len(galosh_ids) > 1:
        print("  [GALOSH ablation]")
        for gid in galosh_ids:
            s = summaries[gid]
            print(f"    {DISPLAY_NAMES[gid]:<22} "
                  f"PSNR={s['psnr']:.2f}  LPIPS={s['lpips']:.4f}  "
                  f"NIQE={s['niqe']:.3f}  Time={s['time_s']:.2f}s")
        print()

    # Claim A: blind non-learning strongest - GALOSH vs every other rule-based
    # (excluding oracle) on PSNR AND LPIPS.
    competitors_rule = [m for m in RULE_IDS
                        if m not in galosh_ids and m not in ORACLE_IDS
                        and m in summaries]
    print(f"  [A] \"Blind non-learning strongest\" - {g_name} vs rule-based:")
    for mid in competitors_rule:
        s = summaries[mid]
        better_psnr  = (g["psnr"] or 0) >= (s["psnr"] or 0)
        better_lpips = (g["lpips"] or 9.99) <= (s["lpips"] or 9.99)
        tag = "OK" if (better_psnr and better_lpips) else "MIXED"
        print(f"    vs {DISPLAY_NAMES[mid]:<22} "
              f"ΔPSNR={(g['psnr'] or 0)-(s['psnr'] or 0):+6.2f} dB  "
              f"ΔLPIPS={(g['lpips'] or 9.99)-(s['lpips'] or 9.99):+.4f}  "
              f"[{tag}]")

    # Claim B: sensor-agnostic DL strongest - GALOSH vs DL_AGNOSTIC.
    competitors_dl = [m for m in DL_AGNOSTIC_IDS if m in summaries]
    print(f"\n  [B] \"Sensor-agnostic strongest incl. DL\" - {g_name} vs DL non-specialised:")
    for mid in competitors_dl:
        s = summaries[mid]
        better_psnr  = (g["psnr"] or 0) >= (s["psnr"] or 0)
        better_lpips = (g["lpips"] or 9.99) <= (s["lpips"] or 9.99)
        tag = "OK" if (better_psnr and better_lpips) else "MIXED"
        print(f"    vs {DISPLAY_NAMES[mid]:<22} "
              f"ΔPSNR={(g['psnr'] or 0)-(s['psnr'] or 0):+6.2f} dB  "
              f"ΔLPIPS={(g['lpips'] or 9.99)-(s['lpips'] or 9.99):+.4f}  "
              f"[{tag}]")

    # NAFNet as the SIDD-trained ceiling reference.
    if "nafnet" in summaries:
        n = summaries["nafnet"]
        print(f"\n  [Ceiling] SIDD-trained NAFNet: "
              f"PSNR={n['psnr']:.2f}  LPIPS={n['lpips']:.4f}  "
              f"Δ{g_name} PSNR={(g['psnr'] or 0)-(n['psnr'] or 0):+.2f} dB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
