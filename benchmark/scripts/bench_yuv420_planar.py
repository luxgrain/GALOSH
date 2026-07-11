#!/usr/bin/env python3
"""GALOSH-420 planar-container validation harness (2026-07-11).

Validates the NATIVE 4:2:0 implementation in galosh_yuv_cpu.exe --pix=420
against the A/B design rig (ab_yuv420.py, arm A = the validated form):

  1. Simulate a JPEG-profile container (matrix bt601, eotf srgb, full range,
     center siting, 8-bit planar) from SIDD noisy sRGB.
  2. Run the exe end-to-end (planar in -> planar out, format-preserving).
  3. Score Cb/Cr PSNR at the 420 lattice vs float GT-420 (same reference
     construction as the rig) + full-res Y PSNR.
  4. Compare per-scene vs the rig's arm-A results (_ab_metrics_bt601.json).

Also: --profiles runs the 5 reference profiles + 422/400/16-bit smoke test
on one scene (functional, not scored against the rig).

Usage:
  python bench_yuv420_planar.py [--limit N] [--siting center|left|topleft]
                                [--exe path] [--profiles]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ab_yuv420 as rig   # down420 / rgb2ycc / ycc2rgb + MATRICES (verbatim kernels)

ROOT = Path(__file__).resolve().parents[2]
OUTROOT = ROOT / "benchmark" / "results_yuv420_ab"
SIDD = Path(os.environ.get("GALOSH_SIDD_BENCH", "benchmark/datasets/sidd_medium_bench"))


def to_planar_420(y, cb, cr, siting, depth=8):
    """gamma-domain float planes -> full-range planar 420 bytes."""
    hi = (1 << depth) - 1; half = 1 << (depth - 1)
    dt = np.uint8 if depth <= 8 else np.uint16
    cbh = rig.down420(cb, siting)
    crh = rig.down420(cr, siting)
    yq = np.clip(np.rint(y * hi), 0, hi).astype(dt)
    cbq = np.clip(np.rint(cbh * hi + half), 0, hi).astype(dt)
    crq = np.clip(np.rint(crh * hi + half), 0, hi).astype(dt)
    return yq.tobytes() + cbq.tobytes() + crq.tobytes(), cbh.shape


def from_planar_420(buf, h, w, depth=8):
    ch, cw = h // 2, w // 2
    ysz, csz = h * w, ch * cw
    hi = (1 << depth) - 1; half = 1 << (depth - 1)
    a = np.frombuffer(buf, dtype=np.uint8 if depth <= 8 else np.uint16)
    y = a[:ysz].reshape(h, w).astype(np.float64) / hi
    cb = (a[ysz:ysz + csz].reshape(ch, cw).astype(np.float64) - half) / hi
    cr = (a[ysz + csz:].reshape(ch, cw).astype(np.float64) - half) / hi
    return y, cb, cr


def psnr(a, b):
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return 99.0 if mse == 0 else float(10 * np.log10(1.0 / mse))


def run_exe(exe, blob, w, h, extra):
    ip = OUTROOT / "_tmp_planar_in.yuv"
    op = OUTROOT / "_tmp_planar_out.yuv"
    ip.write_bytes(blob)
    t0 = time.time()
    # strengths positional (required by the GPU CLI; same defaults as CPU)
    r = subprocess.run([str(exe), str(ip), str(op), str(w), str(h),
                        "1.0", "1.0"] + extra,
                       capture_output=True, timeout=1800,
                       cwd=str(Path(exe).parent))
    dt = time.time() - t0
    if r.returncode != 0 or not op.exists():
        raise RuntimeError(r.stderr.decode("utf-8", "replace")[-500:])
    out = op.read_bytes()
    ip.unlink(missing_ok=True); op.unlink(missing_ok=True)
    return out, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--siting", default="center",
                    choices=["center", "left", "topleft"])
    ap.add_argument("--matrix", default="bt601", choices=sorted(rig.MATRICES))
    ap.add_argument("--exe", default=str(ROOT / "standalone" / "galosh_yuv_cpu.exe"))
    ap.add_argument("--profiles", action="store_true",
                    help="5-reference-profile + 422/400/16bit smoke test")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rig.KR, rig.KB = rig.MATRICES[args.matrix]
    scenes = rig.load_sidd() if SIDD is None else _load(SIDD)
    if args.limit: scenes = scenes[:args.limit]
    exe = Path(args.exe)
    OUTROOT.mkdir(parents=True, exist_ok=True)

    if args.profiles:
        return smoke_profiles(exe, scenes[0])

    # rig arm-A reference (same matrix, center siting only)
    rig_file = OUTROOT / ("_ab_metrics_bt601.json" if args.matrix == "bt601"
                          else "_ab_metrics.json")
    rig_ref = json.loads(rig_file.read_text()) if rig_file.exists() else {}

    flags = [f"--pix=420", f"--depth={args.depth}", f"--range=full",
             f"--matrix={args.matrix}", f"--eotf=srgb", f"--siting={args.siting}"]
    res = {}
    for k, (stem, nf, gf) in enumerate(scenes):
        noisy = np.load(nf).astype(np.float64)
        gt = np.load(gf).astype(np.float64)
        h, w = noisy.shape[:2]
        h -= h % 2; w -= w % 2
        noisy, gt = noisy[:h, :w], gt[:h, :w]

        yn, cbn, crn = rig.rgb2ycc(noisy)
        yg, cbg, crg = rig.rgb2ycc(gt)
        blob, _ = to_planar_420(yn, cbn, crn, args.siting, args.depth)
        gt_cbh = rig.down420(cbg, args.siting)
        gt_crh = rig.down420(crg, args.siting)
        noisy_cbh = rig.down420(cbn, args.siting)
        noisy_crh = rig.down420(crn, args.siting)

        out, dt = run_exe(exe, blob, w, h, flags)
        yd, cbd, crd = from_planar_420(out, h, w, args.depth)

        e = {
            "y_psnr": psnr(yd, yg),
            "cb": psnr(cbd, gt_cbh), "cr": psnr(crd, gt_crh),
            "noisy_cb": psnr(noisy_cbh, gt_cbh),
            "noisy_cr": psnr(noisy_crh, gt_crh),
            "dt": dt,
        }
        ra = rig_ref.get(stem, {}).get("A", {})
        if ra and args.siting == "center":
            e["rig_A_cb"] = ra["cb"]; e["rig_A_cr"] = ra["cr"]
            e["d_cb"] = e["cb"] - ra["cb"]; e["d_cr"] = e["cr"] - ra["cr"]
        res[stem] = e
        extra = (f" | rig A {ra['cb']:.2f}/{ra['cr']:.2f} d {e['d_cb']:+.2f}/{e['d_cr']:+.2f}"
                 if "d_cb" in e else "")
        print(f"[{k+1}/{len(scenes)}] {stem} Y {e['y_psnr']:.2f} | "
              f"Cb {e['cb']:.2f} Cr {e['cr']:.2f} ({dt:.1f}s){extra}", flush=True)

    tag = args.tag or f"{args.matrix}_{args.siting}"
    outf = OUTROOT / f"_planar_val_{tag}.json"
    outf.write_text(json.dumps(res, indent=1))
    n = len(res)
    if n:
        m = lambda k2: float(np.mean([v[k2] for v in res.values() if k2 in v]))
        print(f"\nMEAN (n={n}): Y {m('y_psnr'):.2f} | Cb {m('cb'):.2f} Cr {m('cr'):.2f}"
              f" | noisy Cb {m('noisy_cb'):.2f} Cr {m('noisy_cr'):.2f}")
        if any("d_cb" in v for v in res.values()):
            print(f"vs rig arm A: d_cb {m('d_cb'):+.3f} d_cr {m('d_cr'):+.3f}")
    print("saved:", outf)


def _load(base):
    items = []
    for nf in sorted(base.glob("*_noisy_srgb.npy")):
        stem = nf.name.replace("_noisy_srgb.npy", "")
        gf = base / f"{stem}_gt_srgb.npy"
        if gf.exists(): items.append((stem, nf, gf))
    return items


def smoke_profiles(exe, scene):
    """5 reference profiles + 422/400/16-bit — functional smoke test."""
    stem, nf, gf = scene
    noisy = np.load(nf).astype(np.float64)
    gt = np.load(gf).astype(np.float64)
    h, w = noisy.shape[:2]; h -= h % 2; w -= w % 2
    noisy, gt = noisy[:h, :w][:1024, :1024], gt[:h, :w][:1024, :1024]
    h, w = noisy.shape[:2]
    profiles = [
        ("DVD",    dict(matrix="bt601",  eotf="g24",  range="limited", siting="left",    pix="420", depth=8)),
        ("BluRay", dict(matrix="bt709",  eotf="g24",  range="limited", siting="left",    pix="420", depth=8)),
        ("HDR10",  dict(matrix="bt2020", eotf="pq",   range="limited", siting="topleft", pix="420", depth=10)),
        ("HLG",    dict(matrix="bt2020", eotf="hlg",  range="limited", siting="topleft", pix="420", depth=10)),
        ("JPEG",   dict(matrix="bt601",  eotf="srgb", range="full",    siting="center",  pix="420", depth=8)),
        ("422",    dict(matrix="bt709",  eotf="srgb", range="full",    siting="left",    pix="422", depth=8)),
        ("400",    dict(matrix="bt709",  eotf="srgb", range="full",    siting="center",  pix="400", depth=8)),
        ("16bit",  dict(matrix="bt709",  eotf="srgb", range="full",    siting="center",  pix="420", depth=16)),
    ]
    for name, p in profiles:
        rig.KR, rig.KB = rig.MATRICES.get(p["matrix"], (0.2627, 0.0593))
        if p["matrix"] == "bt2020": rig.KR, rig.KB = 0.2627, 0.0593
        # container sim: encode noisy/gt into the profile's gamma domain.
        # For non-sRGB EOTFs treat the sRGB floats as DISPLAY-LINEARISED
        # source: linear = srgb_eotf_inv(pixels), then re-encode with the
        # profile EOTF.  Functional smoke only (not a quality bench).
        lin_n = np.where(noisy <= 0.04045, noisy / 12.92, ((noisy + 0.055) / 1.055) ** 2.4)
        lin_g = np.where(gt <= 0.04045, gt / 12.92, ((gt + 0.055) / 1.055) ** 2.4)
        enc = _eotf_fwd(p["eotf"])
        y_n, cb_n, cr_n = rig.rgb2ycc(enc(lin_n))
        y_g, cb_g, cr_g = rig.rgb2ycc(enc(lin_g))
        blob, w2, ch2 = _pack(y_n, cb_n, cr_n, p, w, h)
        flags = [f"--pix={p['pix']}", f"--depth={p['depth']}", f"--range={p['range']}",
                 f"--matrix={p['matrix']}", f"--eotf={p['eotf']}", f"--siting={p['siting']}"]
        try:
            out, dt = run_exe(exe, blob, w, h, flags)
            yd = _unpack_y(out, p, w, h)
            e_in = psnr(_q_y(y_n, p), _q_y(y_g, p))
            e_out = psnr(yd, _q_y(y_g, p))
            print(f"[{name:6s}] OK {dt:5.1f}s  Y in {e_in:5.2f} -> out {e_out:5.2f} dB"
                  f"  ({'+' if e_out > e_in else '!'})", flush=True)
        except Exception as ex:
            print(f"[{name:6s}] FAIL: {ex}", flush=True)
    return 0


def _eotf_fwd(name):
    def srgb(l): return np.where(l <= 0.0031308, 12.92 * l,
                                 1.055 * np.clip(l, 0, None) ** (1 / 2.4) - 0.055)
    def g24(l): return np.clip(l, 0, None) ** (1 / 2.4)
    def hlg(l):
        a, b, c = 0.17883277, 0.28466892, 0.55991073
        l = np.clip(l, 0, None)
        return np.where(l <= 1 / 12, np.sqrt(3 * l), a * np.log(np.maximum(12 * l - b, 1e-9)) + c)
    def pq(l):
        m1, m2 = 2610 / 16384, 2523 / 4096 * 128
        c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
        lp = np.clip(l, 0, 1) ** m1
        return ((c1 + c2 * lp) / (1 + c3 * lp)) ** m2
    return {"srgb": srgb, "g24": g24, "hlg": hlg, "pq": pq}[name]


def _q_y(y, p):
    d = p["depth"]; hi = (1 << d) - 1
    if p["range"] == "limited":
        s = 1 << (d - 8)
        c = np.clip(np.rint(y * 219 * s + 16 * s), 0, hi)
        return (c - 16 * s) / (219 * s)
    c = np.clip(np.rint(y * hi), 0, hi)
    return c / hi


def _q_c(c_, p):
    d = p["depth"]; hi = (1 << d) - 1
    if p["range"] == "limited":
        s = 1 << (d - 8)
        c = np.clip(np.rint(c_ * 224 * s + 128 * s), 0, hi)
        return c
    return np.clip(np.rint(c_ * hi + (1 << (d - 1))), 0, hi)


def _pack(y, cb, cr, p, w, h):
    d = p["depth"]; hi = (1 << d) - 1
    dt = np.uint8 if d <= 8 else np.uint16
    if p["range"] == "limited":
        s = 1 << (d - 8)
        yq = np.clip(np.rint(y * 219 * s + 16 * s), 0, hi).astype(dt)
    else:
        yq = np.clip(np.rint(y * hi), 0, hi).astype(dt)
    if p["pix"] == "400":
        return yq.tobytes(), 0, 0
    if p["pix"] == "420":
        cbs = rig.down420(cb, p["siting"]); crs = rig.down420(cr, p["siting"])
    elif p["pix"] == "422":
        # horizontally co-sited 422: tent at even columns per axis helper
        lp = np.pad(cb, ((0, 0), (1, 1)), mode="edge")
        cbs = (0.25 * lp[:, :-2] + 0.5 * lp[:, 1:-1] + 0.25 * lp[:, 2:])[:, 0::2]
        lp = np.pad(cr, ((0, 0), (1, 1)), mode="edge")
        crs = (0.25 * lp[:, :-2] + 0.5 * lp[:, 1:-1] + 0.25 * lp[:, 2:])[:, 0::2]
    else:
        cbs, crs = cb, cr
    cbq = _q_c(cbs, p).astype(dt); crq = _q_c(crs, p).astype(dt)
    return yq.tobytes() + cbq.tobytes() + crq.tobytes(), cbs.shape[1], cbs.shape[0]


def _unpack_y(buf, p, w, h):
    d = p["depth"]
    a = np.frombuffer(buf, dtype=np.uint8 if d <= 8 else np.uint16)
    y = a[:h * w].reshape(h, w).astype(np.float64)
    if p["range"] == "limited":
        s = 1 << (d - 8)
        return (y - 16 * s) / (219 * s)
    return y / ((1 << d) - 1)


if __name__ == "__main__":
    sys.exit(main() or 0)
