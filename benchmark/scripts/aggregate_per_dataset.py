#!/usr/bin/env python3
"""
Aggregate per-method metrics across kodak/mcmaster/cbsd68.

Reads benchmark/<dataset>/<method>/metrics.json (per-iso rows) and emits
one summary table per dataset, sorted by LPIPS ascending (primary perceptual
metric).  Special dirs (__gt__, __noisy__) are skipped.
"""
from pathlib import Path
import io
import json
import sys
import argparse

# EN: Windows cp932 can't print Greek / em-dashes.  Force UTF-8.
# JP: Windows の cp932 では一部文字が出せないので UTF-8 に固定。
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   line_buffering=True)

BASE = Path(__file__).parent.parent
DATASETS = ["kodak", "mcmaster", "cbsd68"]
METRIC_KEYS = ("psnr", "ssim", "lpips", "dists", "niqe", "time_per_img")


def aggregate_method(method_dir: Path):
    p = method_dir / "metrics.json"
    if not p.exists():
        return None
    try:
        data = json.load(open(p))
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    out = {"n": len(data)}
    for k in METRIC_KEYS:
        vals = [r.get(k) for r in data if isinstance(r, dict)
                and r.get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    return out


def fmt(v, spec: str = ".3f") -> str:
    return format(v, spec) if v is not None else "-"


def print_dataset(ds: str):
    ds_dir = BASE / ds
    if not ds_dir.is_dir():
        print(f"\n### {ds.upper()} — not found, skip\n")
        return

    rows = []
    for m_dir in sorted(ds_dir.iterdir()):
        if not m_dir.is_dir():
            continue
        if m_dir.name.startswith("__"):
            continue
        agg = aggregate_method(m_dir)
        if agg is None:
            continue
        rows.append((m_dir.name, agg))

    rows.sort(key=lambda x: (x[1].get("lpips") if x[1].get("lpips") is not None
                             else 9.99,
                             -(x[1].get("psnr") or -999)))

    print(f"\n### {ds.upper()}  (sorted by LPIPS asc, ties by PSNR desc)\n")
    hdr = f"{'method':<22}  {'N':>3}  {'PSNR':>6}  {'SSIM':>7}  {'LPIPS':>7}  {'DISTS':>7}  {'NIQE':>7}  {'time(s)':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, agg in rows:
        print(f"{name:<22}  {agg['n']:>3}  "
              f"{fmt(agg['psnr'], '.2f'):>6}  "
              f"{fmt(agg['ssim'], '.4f'):>7}  "
              f"{fmt(agg['lpips'], '.4f'):>7}  "
              f"{fmt(agg['dists'], '.4f'):>7}  "
              f"{fmt(agg['niqe'], '.3f'):>7}  "
              f"{fmt(agg['time_per_img'], '.3f'):>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    args = ap.parse_args()
    for ds in args.datasets:
        print_dataset(ds)


if __name__ == "__main__":
    main()
