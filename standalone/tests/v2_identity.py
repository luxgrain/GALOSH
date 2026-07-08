#!/usr/bin/env python3
"""V2.0 identity harness — flags-off outputs must stay byte-identical.

V2.0 開発ゲート: 新機能フラグ OFF のとき、CPU FP32 / CPU INT32 (r32) の
出力が Phase-A 開始時点 (ベースライン) と byte-identical であることを
出力ハッシュで検証する。公開済みベンチ数値を守るための回帰ハーネス。

Inputs / 入力:
  - Deterministic synthetic Poisson-Gaussian Bayer frames (seeded, generated
    in-script — no binaries in the repo).
  - Optionally real DNGs via env GALOSH_V2_DNGS (";"-separated paths);
    converted with rawpy using per-channel black/white normalization
    (same as tools/dist/galosh_dng.py). Real frames exercise the blind-fit
    path on real sensor statistics.

Usage / 使い方:
  python v2_identity.py --capture     # write baseline manifest (v2_baseline.json)
  python v2_identity.py --verify      # compare current exes against baseline
Exit codes: 0 = OK, 1 = MISMATCH (regression), 2 = baseline missing/incomplete.

Determinism note / 決定性の注意:
  Each case is run twice; if the two runs differ the case is recorded as
  nondeterministic (max|diff| stored instead of a hash) and compared with a
  tolerance instead. OpenMP reductions could in principle reorder float sums;
  this harness measures rather than assumes.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STANDALONE = HERE.parent
MANIFEST = HERE / "v2_baseline.json"

EXES = {
    "cpu_fp32": STANDALONE / "galosh_raw_cpu.exe",
    "cpu_r32": STANDALONE / "galosh_raw_cpu_int.exe",
}
# canonical blind invocation: method=galosh strength=1 luma=1 chroma=1 blind
ARGS = ["galosh", "1.0", "1.0", "1.0", "0", "0"]


def synth_cases():
    """Deterministic synthetic frames (same recipe as the GPU profiling set)."""
    rng = np.random.default_rng(42)
    # NOTE dimensions must be even (Bayer quads); 258x182 is even but not a
    # multiple of the 8x8 block, exercising the boundary paths.
    # 偶数必須（Bayer）。258x182 は 8 の倍数でない偶数 = 境界処理を踏む。
    for name, W, H in (("synth_1080p", 1920, 1080),
                       ("synth_small_edge", 258, 182),
                       ("synth_dark", 640, 480)):
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        base = 0.35 + 0.25 * np.sin(x / 97.0) * np.cos(y / 71.0) \
             + 0.15 * np.sin((x + y) / 31.0)
        if name == "synth_dark":
            base = base * 0.08 + 0.01   # near-black regime / 暗部レジーム
        gain = np.ones((H, W), np.float32)
        gain[0::2, 0::2] = 0.9
        gain[0::2, 1::2] = 1.1
        gain[1::2, 0::2] = 1.1
        gain[1::2, 1::2] = 0.8
        clean = np.clip(base * gain, 0.002, 0.98).astype(np.float32)
        alpha, sigma = 0.004, 0.01
        noise = rng.normal(0, 1, clean.shape).astype(np.float32)
        noisy = np.clip(clean + noise * np.sqrt(alpha * clean + sigma ** 2),
                        0.0, 1.0).astype(np.float32)
        yield name, W, H, noisy


def dng_cases():
    """Real frames from GALOSH_V2_DNGS (optional, recommended)."""
    paths = [p for p in os.environ.get("GALOSH_V2_DNGS", "").split(";") if p]
    if not paths:
        return
    import rawpy
    for p in paths:
        src = Path(p)
        raw = rawpy.imread(str(src))
        img = raw.raw_image
        H, W = img.shape
        blacks = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        black_map = blacks[raw.raw_colors]
        scale = np.maximum(float(raw.white_level) - black_map, 1.0)
        f32 = ((img.astype(np.float32) - black_map) / scale).clip(0.0, 1.0)
        He, We = H & ~1, W & ~1
        yield f"dng_{src.stem[:40]}", We, He, np.ascontiguousarray(f32[:He, :We])
        raw.close()


def run_case(exe: Path, arr: np.ndarray, W: int, H: int, td: Path, tag: str):
    tin = td / f"{tag}_in.bin"
    tout = td / f"{tag}_out.bin"
    arr.tofile(tin)
    cmd = [str(exe), str(tin), str(tout), str(W), str(H), *ARGS]
    r = subprocess.run(cmd, cwd=str(exe.parent), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not tout.is_file():
        sys.stderr.write(r.stderr or "")
        raise RuntimeError(f"{exe.name} failed on {tag} (exit {r.returncode})")
    data = tout.read_bytes()
    tout.unlink()
    return data


def collect():
    """Run every (exe, case) twice; return manifest entries."""
    entries = {}
    cases = list(synth_cases()) + list(dng_cases())
    with tempfile.TemporaryDirectory(prefix="galoshv2_") as td_s:
        td = Path(td_s)
        for exe_key, exe in EXES.items():
            if not exe.is_file():
                raise SystemExit(f"missing exe: {exe} (build first)")
            for name, W, H, arr in cases:
                key = f"{exe_key}/{name}"
                print(f"  {key} ({W}x{H}) ...", flush=True)
                d1 = run_case(exe, arr, W, H, td, "a")
                d2 = run_case(exe, arr, W, H, td, "b")
                if d1 == d2:
                    entries[key] = {"deterministic": True,
                                    "sha256": hashlib.sha256(d1).hexdigest()}
                else:
                    a = np.frombuffer(d1, np.float32)
                    b = np.frombuffer(d2, np.float32)
                    md = float(np.abs(a - b).max())
                    entries[key] = {"deterministic": False,
                                    "run_max_diff": md,
                                    "mean": float(a.mean())}
                    print(f"    NOTE nondeterministic, run-to-run max|d|={md:.3e}")
    return entries


def main():
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--capture", action="store_true")
    m.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    entries = collect()
    if args.capture:
        MANIFEST.write_text(json.dumps(entries, indent=1), encoding="utf-8")
        print(f"baseline captured: {len(entries)} cases -> {MANIFEST.name}")
        return 0

    if not MANIFEST.is_file():
        print("no baseline manifest — run --capture first")
        return 2
    base = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = 0
    for key, cur in entries.items():
        ref = base.get(key)
        if ref is None:
            print(f"  NEW (not in baseline): {key}")
            continue
        if ref.get("deterministic") and cur.get("deterministic"):
            ok = ref["sha256"] == cur["sha256"]
        else:
            # nondeterministic case: compare means loosely (weak but honest)
            ok = abs(ref.get("mean", 0) - cur.get("mean", 0)) < 1e-6
        print(f"  {'OK  ' if ok else 'FAIL'} {key}")
        bad += (not ok)
    missing = [k for k in base if k not in entries]
    for k in missing:
        print(f"  MISSING vs baseline: {k}")
    if bad or missing:
        print(f"IDENTITY: FAIL ({bad} mismatch, {len(missing)} missing)")
        return 1
    print(f"IDENTITY: PASS ({len(entries)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
