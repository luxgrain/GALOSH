"""
Download Restormer Gaussian color denoising weights.

Tries the blind variant first; if that fails, tries sigma15/25/50.
Run from any directory:

    python download_restormer_weights.py
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path


WEIGHTS_DIR = (Path(__file__).resolve().parents[2] / "external" / "Restormer"
               / "Denoising" / "pretrained_models")   # where the sRGB bench loads it from
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Candidate URLs -- try multiple hosts because release layouts change.
BLIND_URLS = [
    # Common GitHub release asset naming
    "https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_blind.pth",
    # Mirror via raw repo (some forks host them there)
    "https://github.com/swz30/Restormer/releases/download/v1.0.0/gaussian_color_denoising_blind.pth",
]

SIGMA_URLS = {
    15: [
        "https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_sigma15.pth",
        "https://github.com/swz30/Restormer/releases/download/v1.0.0/gaussian_color_denoising_sigma15.pth",
    ],
    25: [
        "https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_sigma25.pth",
        "https://github.com/swz30/Restormer/releases/download/v1.0.0/gaussian_color_denoising_sigma25.pth",
    ],
    50: [
        "https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_sigma50.pth",
        "https://github.com/swz30/Restormer/releases/download/v1.0.0/gaussian_color_denoising_sigma50.pth",
    ],
}


def _download(url: str, dest: Path) -> bool:
    """Return True on success."""
    print(f"  GET {url}", flush=True)
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 restormer-downloader"}
    )
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
            size = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                size += len(chunk)
                if size % (10 << 20) == 0:
                    print(f"    ...{size >> 20} MiB", flush=True)
        tmp.rename(dest)
        dt = time.time() - t0
        mb = dest.stat().st_size / (1 << 20)
        print(f"  OK  {dest.name}  {mb:.1f} MiB in {dt:.1f}s", flush=True)
        return True
    except Exception as e:
        print(f"  FAIL {e}", flush=True)
        if tmp.exists():
            tmp.unlink()
        return False


def try_download(name: str, urls: list[str]) -> bool:
    dest = WEIGHTS_DIR / name
    if dest.exists():
        mb = dest.stat().st_size / (1 << 20)
        print(f"[skip] {name} already present ({mb:.1f} MiB)")
        return True
    for url in urls:
        if _download(url, dest):
            return True
    return False


def main() -> int:
    print(f"Downloading Restormer Gaussian color weights to: {WEIGHTS_DIR}")
    print()

    # 1) Blind variant
    print("== Trying BLIND variant ==")
    got_blind = try_download("gaussian_color_denoising_blind.pth", BLIND_URLS)
    print()

    if got_blind:
        print("Blind weights are sufficient; exiting.")
        return 0

    # 2) Fall back to specific-sigma variants
    print("Blind variant unavailable. Trying specific-sigma variants...")
    got_any = False
    for s in (15, 25, 50):
        fname = f"gaussian_color_denoising_sigma{s}.pth"
        print(f"== sigma={s} ==")
        if try_download(fname, SIGMA_URLS[s]):
            got_any = True
        print()

    if not got_any:
        print("ERROR: Could not download any Restormer weights.")
        print("Manual download hints:")
        print("  https://github.com/swz30/Restormer/tree/main/Denoising/pretrained_models")
        print("  https://github.com/swz30/Restormer/releases")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
