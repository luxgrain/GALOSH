"""Synthetic input generator + output checker for the GALOSH smoke/bench harness.
Requires: python3 + numpy (the only harness dependency).            (Apache-2.0)

  gen:   python _synth.py gen  <case> <W> <H> <C> <out.bin>
  check: python _synth.py check <file.bin> <W> <H> <C>   -> prints stats, exits 1 on NaN/size
  psnr:  python _synth.py psnr <a.bin> <b.bin>           -> prints PSNR(a,b)
Cases: constant | random | high-noise | near-black | gradient-pg (Poisson-Gaussian on a gradient)
"""
import sys
import numpy as np


def gen(case, w, h, c, path):
    rng = np.random.default_rng(12345)          # fixed seed: reproducible harness
    shape = (h, w) if c == 1 else (h, w, c)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    base = (0.15 + 0.7 * (xx + yy) / 2).astype(np.float32)
    if c == 3:
        base = np.stack([base, np.roll(base, h // 4, 0), base[::-1]], axis=2)
    if case == "constant":
        img = np.full(shape, 0.5, np.float32)
    elif case == "random":
        img = rng.uniform(0, 1, shape).astype(np.float32)
    elif case == "high-noise":
        img = np.clip(base + rng.normal(0, 0.25, shape), 0, 1).astype(np.float32)
    elif case == "near-black":
        img = np.clip(0.002 * base + rng.normal(0, 0.004, shape), 0, 1).astype(np.float32)
    elif case == "gradient-pg":                  # Poisson-Gaussian, the model GALOSH assumes
        alpha, sig = 0.01, 0.02
        img = np.clip(base + rng.normal(0, 1, shape) * np.sqrt(alpha * base + sig * sig),
                      0, 1).astype(np.float32)
    else:
        raise SystemExit(f"unknown case {case}")
    img.astype(np.float32).tofile(path)
    print(f"gen {case} {w}x{h}x{c} -> {path}")


def check(path, w, h, c):
    a = np.fromfile(path, dtype=np.float32)
    want = w * h * c
    ok = True
    if a.size != want:
        print(f"  SIZE MISMATCH: {a.size} vs {want}"); ok = False
    if a.size and not np.isfinite(a).all():
        print(f"  NON-FINITE values: {np.count_nonzero(~np.isfinite(a))}"); ok = False
    if a.size and (a.min() < -0.25 or a.max() > 1.25):
        print(f"  RANGE SUSPECT: [{a.min():.3f}, {a.max():.3f}]"); ok = False
    if a.size:
        print(f"  stats: min {a.min():.4f} max {a.max():.4f} mean {a.mean():.4f} std {a.std():.4f}")
    sys.exit(0 if ok else 1)


def psnr(pa, pb):
    a = np.fromfile(pa, dtype=np.float32); b = np.fromfile(pb, dtype=np.float32)
    assert a.size == b.size, (a.size, b.size)
    print(f"{-10 * np.log10(np.mean((a - b) ** 2) + 1e-12):.2f}")


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "gen":
        gen(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    elif m == "check":
        check(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif m == "psnr":
        psnr(sys.argv[2], sys.argv[3])
