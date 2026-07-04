"""GALOSH RAW V2 overnight benchmark campaign (SIDD Medium / RawNIND, full-frame).

Runs the confirmed RAW comparison set on full-frame images with 5 metrics
(PSNR/SSIM on RAW Bayer; LPIPS/DISTS/NIQE on calibrated sRGB render) + per-method
runtime.  Resume-able (per-image metrics JSON), storage-safe (metrics always;
denoised .npy only if --save-npy).  Algorithm code = clean V2 (galosh-public).

Methods (all blind / training-free except B2U/AP-BSN which are self-supervised DL):
  galosh_fp32     GALOSH GPU FP32 (o32, blind)
  galosh_int16    GALOSH INT16 (GPU i16 storage; near-lossless to the r32 CPU reference, blind)
  bm3d_cfa        BM3D-CFA (blind MAD sigma)
  vst_bm3d_cfa    VST(gen.Anscombe)+BM3D-CFA (blind alpha/sigma)
  nlm_cfa         NLM-CFA  (blind MAD sigma)
  vst_nlm_cfa     VST+NLM-CFA (blind alpha/sigma)
  vst_galosh      VST + GALOSH shrinkage core (GALOSH fed external blind alpha/sigma
                  = "no GALOSH estimation"; isolates GALOSH's blind-estimation gain)
  b2u             Blind2Unblind (self-supervised RAW DL)

  python bench_raw_v2_campaign.py --dataset sidd_medium [--methods a,b] [--save-npy] [--limit N]
  python bench_raw_v2_campaign.py --dataset rawnind   ...
"""
from __future__ import annotations
import os, sys, json, time, argparse, traceback, glob
from pathlib import Path
import numpy as np

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
os.environ["PYTHONIOENCODING"] = "utf-8"
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

GALOSH = Path(os.environ.get("GALOSH_ROOT", str(Path(__file__).resolve().parents[2])))
SCRIPTS = GALOSH / "benchmark" / "scripts"
sys.path.insert(0, str(SCRIPTS)); sys.path.insert(0, str(SCRIPTS / "methods"))
import bench_sidd_medium as smb   # render + 5-metric + runners (run_galosh_gpu, run_nlm_cfa_cuda, ...)
import cv2
def _demosaic_fast(bayer):
    """Edge-aware (cv2, C++) demosaic for the perceptual render — 128x faster than the
    pure-python DDFAPD (0.12s vs 15.4s on a full SIDD frame).  The per-scene affine
    (estimate_affine) corrects all colour, so only spatial demosaic quality matters;
    render-vs-true-GT LPIPS = 0.018 (faithful).  SIDD path only (RawNIND uses render_rawnind)."""
    b16 = np.clip(bayer.astype(np.float32) * 65535.0, 0, 65535).astype(np.uint16)
    return cv2.cvtColor(b16, cv2.COLOR_BayerRG2RGB_EA).astype(np.float32) / 65535.0
smb.demosaic_menon = _demosaic_fast   # patches estimate_affine + raw_to_srgb_calibrated consistently
from skimage.metrics import structural_similarity as _ssim
def ssim_raw(ref, test):
    return float(_ssim(ref.astype(np.float64), test.astype(np.float64), data_range=1.0))

def save_artifact(adir, name, raw, srgb):
    """Persist one result: raw float32 .npy + sRGB .png (per feedback_png_persistence)."""
    adir.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        np.save(adir / f"{name}.npy", raw.astype(np.float32))
    if srgb is not None:
        img = (np.clip(srgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        cv2.imwrite(str(adir / f"{name}.png"), img[:, :, ::-1])  # RGB -> BGR

CPU_FP32 = GALOSH / "standalone" / "galosh_raw_cpu.exe"
CPU_INT  = GALOSH / "standalone" / "galosh_raw_cpu_int.exe"
BASH = r"C:\msys64\usr\bin\bash.exe"

_RND = None
def build_render(dataset, stem, gr, gsf):
    """Per-image RAW->sRGB render for perceptual metrics.
    SIDD: per-scene affine (WB x CCM) fit from GT raw/srgb.  RawNIND: per-scene
    DNG-metadata ISP (render_rawnind).  Returns (render_fn(den)->srgb, gt_srgb)."""
    global _RND
    try:
        if dataset == "sidd_medium":
            gs = np.load(gsf).astype(np.float32)
            if gs.max() > 1.5: gs = gs / 255.0
            affine = smb.estimate_affine(gr, gs)
            return (lambda d: smb.raw_to_srgb_calibrated(d, affine),
                    smb.raw_to_srgb_calibrated(gr, affine))
        scene = stem.rsplit("__ISO", 1)[0]
        meta = json.load(open(rf"E:\rawnind_bench\__metadata__\{scene}.json"))
        if _RND is None:
            import render_rawnind as _r; _RND = _r
        return (lambda d: _RND.render(d, meta), _RND.render(gr, meta))
    except Exception as e:
        sys.stderr.write(f"render setup {stem}: {str(e)[:120]}\n"); return None, None


# ---------- blind Poisson-Gaussian alpha/sigma (noisy-only, lower-envelope fit) ----------
def blind_pg(noisy, bs=16):
    h, w = noisy.shape
    hh, ww = (h // bs) * bs, (w // bs) * bs
    x = noisy[:hh, :ww].reshape(hh // bs, bs, ww // bs, bs)
    m = x.mean(axis=(1, 3)).ravel()
    v = x.var(axis=(1, 3)).ravel()
    ok = (m > 0.003) & (m < 0.97)
    m, v = m[ok], v[ok]
    if m.size < 50: return 0.01, 1e-4
    # lower-envelope: per mean-bin 10th percentile var, then robust linear fit v = a*m + s
    bins = np.linspace(m.min(), m.max(), 24)
    mm, vv = [], []
    for i in range(len(bins) - 1):
        sel = (m >= bins[i]) & (m < bins[i + 1])
        if sel.sum() >= 8:
            mm.append(0.5 * (bins[i] + bins[i + 1])); vv.append(np.percentile(v[sel], 10))
    if len(mm) < 4: return 0.01, 1e-4
    mm, vv = np.array(mm), np.array(vv)
    A = np.vstack([mm, np.ones_like(mm)]).T
    a, s = np.linalg.lstsq(A, vv, rcond=None)[0]
    return float(max(a, 1e-5)), float(max(s, 1e-8))


# ---------- method runners: each (noisy_raw, gt_raw) -> (denoised_raw, dt) ----------
def run_cpu_exe(exe, noisy, w, h, uid, alpha=0.0, sigma=0.0):
    tmp = smb.OUTDIR
    uid = f"{uid}_{os.getpid()}"   # PID-unique temp: parallel processes must NOT share _tmp files
    ip, op = tmp / f"_tmp_{uid}_in.raw", tmp / f"_tmp_{uid}_out.raw"
    noisy.astype(np.float32).tofile(str(ip))
    cmd = [BASH, "-c", f'"{exe}" "{ip}" "{op}" {w} {h} galosh 1.0 1.0 1.0 {alpha} {sigma}']
    import subprocess
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    dt = time.time() - t0
    ip.unlink(missing_ok=True)
    if r.returncode != 0 or not op.exists():
        sys.stderr.write(r.stderr.decode("utf-8", "replace")[-300:]); return None, dt
    den = np.fromfile(str(op), dtype=np.float32).reshape(h, w); op.unlink(missing_ok=True)
    return den, dt


def run_galosh_i16_gpu(noisy, w, h, uid):
    """GPU INT16 (i16) = the DEPLOYED INT16 product (CPU r32 is dev/verify only;
    near-lossless to this, ~60-90 dB end-to-end -- the residual is the INT16
    storage quantization below; at INT32 storage the same GPU pipeline is
    bit-exact vs r32, verified on full frames 2026-07-04).  Full P0..P10 on GPU
    via galosh_int_pipe_test phase 10
    with i16 storage (lf=5 cf=9); output is Q11.20 int32 -> /2^20 raw float [0,1].
    Matches the paper's claim that the shipped pipeline is the GPU variant for BOTH
    precisions (the speed table is GPU-only)."""
    import subprocess
    GPU_I16 = str(GALOSH / "standalone" / "galosh_int_pipe_test.exe")
    tmp = smb.OUTDIR
    uid = f"{uid}_{os.getpid()}"
    ip, op = tmp / f"_tmp_{uid}_in.raw", tmp / f"_tmp_{uid}_out.bin"
    noisy.astype(np.float32).tofile(str(ip))
    env = dict(os.environ); env["PATH"] = r"C:\msys64\ucrt64\bin;" + env.get("PATH", "")
    # lf=5 cf=9 = the production per-q design (luma Q10.5 / chroma Q6.9); lf=9 overflows
    # the luma integer part on bright scenes (verified: 2pilesofplates 26.7 vs 46.5 dB).
    cmd = [GPU_I16, str(ip), str(w), str(h), "10", str(op), "0", "i16", "lf=5", "cf=9"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, timeout=600, env=env)
    dt = time.time() - t0
    ip.unlink(missing_ok=True)
    if r.returncode != 0 or not op.exists():
        sys.stderr.write(r.stderr.decode("utf-8", "replace")[-300:]); return None, dt
    den = np.fromfile(str(op), dtype=np.int32).reshape(h, w).astype(np.float32) / (2.0**20)
    op.unlink(missing_ok=True)
    return den, dt


def make_runners():
    from bench_vst_classical_semiblind import vst_bm3d_cfa, vst_nlm_cfa
    from bm3d_cfa import run_bm3d_cfa
    import pywt

    def smad(nr):
        _, (_, _, hh) = pywt.dwt2(nr.astype(np.float64), "haar")
        return float(np.median(np.abs(hh)) / 0.6745)

    R = {}
    R["galosh_fp32"]  = lambda nr, gr: smb.run_galosh_gpu(nr, nr.shape[1], nr.shape[0], "gfp32")
    R["galosh_int16"] = lambda nr, gr: run_galosh_i16_gpu(nr, nr.shape[1], nr.shape[0], "gi16")  # GPU i16 (deployed), not CPU r32
    R["bm3d_cfa"]     = lambda nr, gr: (lambda t0: (run_bm3d_cfa(nr.astype(np.float32), sigma=[smad(nr)]*4)[0], time.time()-t0))(time.time())
    R["nlm_cfa"]      = lambda nr, gr: smb.run_nlm_cfa_cuda(nr, smad(nr))
    R["vst_bm3d_cfa"] = lambda nr, gr: (lambda a, t0: (vst_bm3d_cfa(nr, a[0], a[1]), time.time()-t0))(blind_pg(nr), time.time())
    R["vst_nlm_cfa"]  = lambda nr, gr: (lambda a, t0: (vst_nlm_cfa(nr, a[0], a[1]), time.time()-t0))(blind_pg(nr), time.time())
    # vst_galosh = GALOSH core fed an EXTERNAL blind (alpha, sigma) fit (no native Phase-0
    # estimation) — isolates GALOSH's blind-estimation gain.  GPU exe since 2026-07-02
    # (galosh_raw_gpu.c now honors supplied alpha/sigma; formerly it silently ignored them,
    # so this ablation had to run the CPU exe -> the stale 15.3 s / pre-anti-ringing numbers).
    R["vst_galosh"]   = lambda nr, gr: (lambda a: smb.run_galosh_gpu(nr, nr.shape[1], nr.shape[0], "vstg", alpha=a[0], sigma_sq=a[1]))(blind_pg(nr))
    # [DEPRECATED, kept for reference] CPU-exe variant of the same ablation:
    # R["vst_galosh"] = lambda nr, gr: (lambda a: run_cpu_exe(CPU_FP32, nr, nr.shape[1], nr.shape[0], "vstg", alpha=a[0], sigma=a[1]))(blind_pg(nr))
    try:
        from b2u import run_b2u
        R["b2u"] = lambda nr, gr: run_b2u(nr.astype(np.float32), tile=512, overlap=32)
    except Exception as e:
        sys.stderr.write(f"b2u unavailable: {e}\n")
    return R


# ---------- datasets ----------
def load_sidd_medium():
    base = Path(r"E:\img_dataset\sidd\medium_bench")
    items = []
    for nf in sorted(base.glob("*noisy_raw.npy")):
        stem = nf.name.replace("_noisy_raw.npy", "")
        gr, gs = base / f"{stem}_gt_raw.npy", base / f"{stem}_gt_srgb.npy"
        if gr.exists() and gs.exists(): items.append((stem, nf, gr, gs))
    return items, GALOSH / "benchmark" / "results_raw_sidd"


def load_rawnind():
    # noisy "scene__ISO###.npy" -> gt "scene.npy"; precomputed sRGB GT render for perceptual
    base = Path(r"E:\img_dataset\rawnind_bench")
    nd, gd, grr = base / "__noisy_raw__", base / "__gt_raw__", base / "__gt_raw_render__"
    items = []
    if nd.exists():
        for nf in sorted(nd.glob("*.npy")):
            scene = nf.stem.split("__")[0]
            gr = gd / f"{scene}.npy"
            if not gr.exists(): continue
            gs = grr / f"{scene}.npy"
            items.append((nf.stem, nf, gr, gs if gs.exists() else None))
    return items, Path(r"E:\rawnind_bench_v2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["sidd_medium", "rawnind"])
    ap.add_argument("--methods", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-npy", action="store_true")
    ap.add_argument("--save-artifacts", action="store_true",
                    help="persist sRGB PNG + raw npy for every method + GT + Noisy (feedback_png_persistence)")
    ap.add_argument("--tag", default="")   # per-process metrics JSON (avoid parallel write race)
    args = ap.parse_args()

    items, outroot = (load_sidd_medium() if args.dataset == "sidd_medium" else load_rawnind())
    outroot.mkdir(parents=True, exist_ok=True)
    if args.limit: items = items[:args.limit]
    runners = make_runners()
    sel = args.methods.split(",") if args.methods else list(runners.keys())
    res_path = outroot / (f"_metrics_{args.tag}.json" if args.tag else "_metrics.json")
    metrics = json.load(open(res_path)) if res_path.exists() else {}
    print(f"[{args.dataset}] {len(items)} images x {len(sel)} methods -> {outroot}", flush=True)

    t_start = time.time()
    for ii, (stem, nf, grf, gsf) in enumerate(items):
        nr = np.load(nf).astype(np.float32); gr = np.load(grf).astype(np.float32)
        h, w = nr.shape
        render_fn, gt_srgb = build_render(args.dataset, stem, gr, gsf)
        adir = outroot / "_artifacts" / stem
        if args.save_artifacts and not (adir / "_gt.png").exists():
            save_artifact(adir, "_gt", gr, gt_srgb)
            save_artifact(adir, "_noisy", nr, render_fn(nr) if render_fn is not None else None)
        for name in sel:
            if name not in runners: continue
            metrics.setdefault(name, {})
            if stem in metrics[name]: continue
            try:
                den, dt = runners[name](nr, gr)
                if den is None: metrics[name][stem] = {"err": "None"}; continue
                rec = {"psnr": smb.psnr(gr, den), "ssim": ssim_raw(gr, den), "dt": dt}
                ds = None
                if render_fn is not None:
                    ds = render_fn(den)
                    # per-metric guard: one flaky perceptual metric (e.g. NIQE svd) must not drop the others
                    for mn, mf in (("lpips", lambda: smb.compute_lpips_patched(ds, gt_srgb)),
                                   ("dists", lambda: smb.compute_dists_patched(ds, gt_srgb)),
                                   ("niqe",  lambda: smb.compute_niqe(ds))):
                        try: rec[mn] = mf()
                        except Exception as me:
                            rec[mn] = None; sys.stderr.write(f"  {name} {stem} {mn} fail: {str(me)[:120]}\n")
                metrics[name][stem] = rec
                if args.save_artifacts:
                    save_artifact(adir, name, den, ds)
                if args.save_npy:
                    md = outroot / name; md.mkdir(exist_ok=True)
                    np.save(md / f"{stem}.npy", den.astype(np.float32))
            except Exception as e:
                metrics[name][stem] = {"err": str(e)[:200]}
                sys.stderr.write(f"  {name} {stem}: {e}\n{traceback.format_exc()[-400:]}\n")
        if (ii + 1) % 2 == 0 or ii == len(items) - 1:
            json.dump(metrics, open(res_path, "w"), indent=1, default=str)
            el = (time.time() - t_start) / 60
            done = {n: sum(1 for v in metrics.get(n, {}).values() if "psnr" in v) for n in sel}
            print(f"  [{ii+1}/{len(items)}] {el:.1f}min | {done}", flush=True)
    json.dump(metrics, open(res_path, "w"), indent=1, default=str)
    print(f"DONE {args.dataset} in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
