#!/usr/bin/env python3
"""Envelope-rerun delta report (2026-07-19).

Compares the GALOSH rows (cpu/vk x fit/hold) BEFORE the estimator switch
(snapshot: benchmark/_ARCHIVE/pre_envelope_20260719/) against the freshly
merged results, per track x lane.  Old = MAD-canonical DLL (+ vk clip bug);
new = envelope-canonical DLL (+ vk clip fix).

Output: benchmark/_envelope_delta_20260719.md — one consolidated table
(track x method, mean delta over all cells common to both) + per-level
PSNR breakdown per track for cpu-fit.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "benchmark"
OLD = B / "_ARCHIVE" / "pre_envelope_20260719"
GAL = ["galosh-cpu-fit", "galosh-cpu-hold", "galosh-vk-fit", "galosh-vk-hold"]
METRICS = ["psnr", "ssim", "lpips", "dists", "niqe"]

# track label -> (old file, new file)
TRACKS = {
    "awgn-420":    ("_metrics_420.json",        B / "results_set8_awgn" / "_metrics_420.json"),
    "awgn-444":    ("_metrics_444.json",        B / "results_set8_awgn" / "_metrics_444.json"),
    "pg-core-420": ("_metrics_pg_420.json",     B / "results_set8_pgnoise" / "_metrics_pg_420.json"),
    "pg-cmp-420":  ("_metrics_pg_420_cmp.json", B / "results_set8_pgnoise" / "_metrics_pg_420_cmp.json"),
    "pg-core-444": ("_metrics_pg_444.json",     B / "results_set8_pgnoise" / "_metrics_pg_444.json"),
    "pg-cmp-444":  ("_metrics_pg_444_cmp.json", B / "results_set8_pgnoise" / "_metrics_pg_444_cmp.json"),
    "crvd-420":    ("_metrics_crvd.json",       B / "results_crvd" / "_metrics_crvd.json"),
    "crvd-444":    ("_metrics_crvd444.json",    B / "results_crvd" / "_metrics_crvd444.json"),
}


def cells(d):
    """yield (seq, level, method_dict) skipping meta keys."""
    for seq, levels in d.items():
        if seq == "_env" or not isinstance(levels, dict):
            continue
        for lvl, methods in levels.items():
            if lvl in ("n_frames", "src") or not isinstance(methods, dict):
                continue
            yield seq, lvl, methods


def main():
    lines = ["# Envelope rerun delta (new - old), 2026-07-19",
             "",
             "old = MAD-canonical frameserver DLL (+ vk 420 clip bug); "
             "new = envelope-canonical (+ clip fix).  Mean over all cells "
             "present in BOTH runs.",
             ""]
    lines.append("| track | method | dPSNR | dSSIM | dLPIPS | dDISTS | dNIQE | cells |")
    lines.append("|---|---|---|---|---|---|---|---|")
    per_level = {}
    for track, (oldname, newpath) in TRACKS.items():
        oldp = OLD / oldname
        if not oldp.exists() or not newpath.exists():
            lines.append(f"| {track} | (missing file) | - | - | - | - | - | - |")
            continue
        old = json.loads(oldp.read_text())
        new = json.loads(newpath.read_text())
        oldmap = {(s, l): m for s, l, m in cells(old)}
        for meth in GAL:
            deltas = {k: [] for k in METRICS}
            n = 0
            for s, l, m in cells(new):
                om = oldmap.get((s, l), {})
                if meth not in m or meth not in om:
                    continue
                ok = True
                for k in METRICS:
                    if k not in m[meth] or k not in om[meth]:
                        ok = False
                if not ok:
                    continue
                n += 1
                for k in METRICS:
                    deltas[k].append(m[meth][k] - om[meth][k])
                if meth == "galosh-cpu-fit":
                    per_level.setdefault(track, {}).setdefault(l, []).append(
                        m[meth]["psnr"] - om[meth]["psnr"])
            if n:
                d = {k: float(np.mean(v)) for k, v in deltas.items()}
                lines.append(
                    f"| {track} | {meth} | {d['psnr']:+.3f} | {d['ssim']:+.4f} "
                    f"| {d['lpips']:+.4f} | {d['dists']:+.4f} "
                    f"| {d['niqe']:+.3f} | {n} |")
            else:
                lines.append(f"| {track} | {meth} | (no common cells) | | | | | 0 |")

    lines += ["", "## per-level dPSNR (galosh-cpu-fit)", ""]
    for track, lvls in per_level.items():
        def key(x):
            t = x.lstrip("sISO")
            return int(t) if t.isdigit() else 0
        parts = [f"{l}: {np.mean(v):+.2f}" for l, v in
                 sorted(lvls.items(), key=lambda kv: key(kv[0]))]
        lines.append(f"- **{track}**: " + " / ".join(parts))

    out = B / "_envelope_delta_20260719.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
