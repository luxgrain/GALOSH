"""GALOSH-YUV-GPU FP32 sRGB denoising benchmark (SIDD Medium, 80 sRGB pairs).

Compares the canonical GALOSH-YUV-GPU FP32 pipeline (galosh_yuv_gpu.exe, the
reconciled sRGB->linear->YCbCr->GAT+LOSH / chroma-LOESS->sRGB product) against:
  classical (blind):  BM3D (per-channel), NLM (per-channel), guided filter
  DL reference:       DnCNN-B (KAIR gray-blind, per-channel)   [if weights present]
                      NAFNet (SIDD)                            [if wired/present]

Metrics (feedback_full_metrics): PSNR, SSIM, LPIPS(alex), DISTS, NIQE.
Artifacts (feedback_png_persistence): per-method PNG + .npy, plus GT + noisy.
Output:  benchmark/results_srgb_sidd_1024|results_srgb_sidd_full|results_srgb_rawnind/
         {<method>/<tag>.png, _metrics.json, _table.txt}

Blind sigma for BM3D/NLM = skimage.restoration.estimate_sigma (matches the
"all methods blind" framing; GALOSH estimates its own alpha/sigma internally).
"""
from __future__ import annotations
import sys, os, json, time, argparse, subprocess, threading
from pathlib import Path
import numpy as np
from PIL import Image

GALOSH     = Path(__file__).resolve().parents[2]
EXE_GPU    = GALOSH / "standalone" / "galosh_yuv_gpu.exe"
BENCH_SIDD = Path(os.environ.get("GALOSH_SIDD_BENCH", "benchmark/datasets/sidd_medium_bench"))
RAWNIND_BASE = Path(os.environ.get("GALOSH_RAWNIND_BENCH", "benchmark/datasets/rawnind_bench"))
RAWNIND_N  = RAWNIND_BASE / "__noisy_raw_render__"
RAWNIND_G  = RAWNIND_BASE / "__gt_raw_render__"
OUT_NAME   = {"sidd": "results_srgb_sidd_1024", "rawnind": "results_srgb_rawnind"}
TMP        = GALOSH / "benchmark" / "_tmp"
UCRT_BIN   = r"C:\msys64\ucrt64\bin"
DEVICE     = "cuda"

def iter_scenes(dataset, limit):
    """Yield (tag, noisy_srgb_f32[0,1], gt_srgb_f32[0,1]) for the dataset."""
    if dataset == "sidd":
        scenes = json.load(open(BENCH_SIDD / "scenes.json"))
        if limit: scenes = scenes[:limit]
        for sc in scenes:
            tag = sc["tag"]
            n = np.load(BENCH_SIDD / f"{tag}_noisy_srgb.npy").astype(np.float32)
            g = np.load(BENCH_SIDD / f"{tag}_gt_srgb.npy").astype(np.float32)
            if n.max() > 1.5: n /= 255.0
            if g.max() > 1.5: g /= 255.0
            yield tag, n, g
    elif dataset == "rawnind":
        from PIL import Image as _I
        noisy = sorted(RAWNIND_N.glob("*.png"))
        if limit and limit < len(noisy):  # even sample across scenes/ISOs, not first-N
            step = len(noisy) / limit
            noisy = [noisy[int(i * step)] for i in range(limit)]
        for npth in noisy:
            scene = npth.stem.split("__ISO")[0]
            gp = RAWNIND_G / f"{scene}.png"
            if not gp.exists(): continue
            n = np.asarray(_I.open(npth).convert("RGB"), np.float32) / 255.0
            g = np.asarray(_I.open(gp).convert("RGB"), np.float32) / 255.0
            if n.shape != g.shape: continue
            yield npth.stem, n, g

def scene_count(dataset, limit):
    if dataset == "sidd":
        n = len(json.load(open(BENCH_SIDD / "scenes.json")))
    else:
        n = len(list(RAWNIND_N.glob("*.png")))
    return min(n, limit) if limit else n

_bm3d_lock = threading.Lock()
_nlm_lock  = threading.Lock()

# ----------------------------------------------------------------- metrics
def psnr(a, b):
    return float(-10.0 * np.log10(np.mean((a - b) ** 2) + 1e-12))

from skimage.metrics import structural_similarity as _ssim
def ssim_rgb(a, b):
    return float(_ssim(a, b, data_range=1.0, channel_axis=2))

from skimage.restoration import estimate_sigma
def est_sigma(img):
    s = estimate_sigma(img, channel_axis=-1, average_sigmas=True)
    return float(max(s, 1e-3))

# Full-frame images (SIDD sRGB = 15.8MP) blow up GPU VRAM for whole-image
# perceptual metrics -> above this pixel count, LPIPS/DISTS are the mean over a
# non-overlapping 1024^2 tile grid (standard full-frame protocol).
_METRIC_TILE_PX = 3_000_000
def _tile_grid(H, W, ts=1024):
    ys = list(range(0, max(H - ts, 0) + 1, ts)) or [0]
    xs = list(range(0, max(W - ts, 0) + 1, ts)) or [0]
    if ys[-1] + ts < H: ys.append(H - ts)
    if xs[-1] + ts < W: xs.append(W - ts)
    return [(y, x, min(ts, H), min(ts, W)) for y in ys for x in xs]

def _tiled_pair_metric(fn, a, b, ts=1024):
    H, W = a.shape[:2]
    vals = [fn(np.ascontiguousarray(a[y:y+h, x:x+w]), np.ascontiguousarray(b[y:y+h, x:x+w]))
            for (y, x, h, w) in _tile_grid(H, W, ts)]
    return float(np.mean(vals))

_lpips_fn = None
def lpips_m(a, b):
    global _lpips_fn
    import torch, lpips
    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(DEVICE)
    if a.shape[0] * a.shape[1] > _METRIC_TILE_PX:
        return _tiled_pair_metric(lpips_m, a, b)
    def t(x):  # HWC [0,1] -> NCHW [-1,1]
        return (torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) * 2 - 1)
    with torch.no_grad():
        return float(_lpips_fn(t(a), t(b)).item())

_dists_fn = None
def dists_m(a, b):
    global _dists_fn
    import torch
    from DISTS_pytorch import DISTS
    if _dists_fn is None:
        _dists_fn = DISTS().to(DEVICE)
    if a.shape[0] * a.shape[1] > _METRIC_TILE_PX:
        return _tiled_pair_metric(dists_m, a, b)
    def t(x):
        return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        return float(_dists_fn(t(a), t(b)).item())

_niqe_fn = None
def niqe_m(a):
    global _niqe_fn
    import torch, pyiqa
    if _niqe_fn is None:
        _niqe_fn = pyiqa.create_metric('niqe', device=DEVICE)
    def t(x):
        return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    try:
        with torch.no_grad():
            return float(_niqe_fn(t(a).clamp(0, 1)).item())
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        H, W = a.shape[:2]
        vals = [float(_niqe_fn(t(np.ascontiguousarray(a[y:y+h, x:x+w])).clamp(0, 1)).item())
                for (y, x, h, w) in _tile_grid(H, W, 2048)]
        return float(np.mean(vals))

# ----------------------------------------------------------------- methods
_galosh_dev = None
def m_galosh(noisy, uid, dev=None):
    # This box has 3 OpenCL devices (AMD gfx1036 iGPU, Intel Arc A310, NVIDIA 4070Ti). Only
    # NVIDIA builds galosh.cl; the others crash. Under CUDA load the OpenCL enumeration order
    # shuffles, so the fixed dev=0 can land on the iGPU. -> PROBE for the working (NVIDIA)
    # device and cache it; re-probe if the cached one later fails.
    global _galosh_dev
    H, W = noisy.shape[:2]
    ip = TMP / f"_{uid}_in.bin"; op = TMP / f"_{uid}_out.bin"
    noisy.astype(np.float32).tofile(str(ip))
    env = os.environ.copy()
    if UCRT_BIN not in env.get("PATH", ""):
        env["PATH"] = UCRT_BIN + os.pathsep + env.get("PATH", "")
    cand = [_galosh_dev] if _galosh_dev is not None else [0, 1, 2, 3]
    err = ""
    for _pass in range(2):
        for d in cand:
            if d is None: continue
            t0 = time.perf_counter()
            p = subprocess.run([str(EXE_GPU), str(ip), str(op), str(W), str(H), "1.0", "1.0", str(d)],
                               env=env, capture_output=True, timeout=600)  # full-frame 15.8MP: 190MB I/O each way
            dt = time.perf_counter() - t0
            if p.returncode == 0 and op.exists():
                _galosh_dev = d
                den = np.fromfile(str(op), dtype=np.float32).reshape(H, W, 3)
                op.unlink(missing_ok=True); ip.unlink(missing_ok=True)
                return np.clip(den, 0, 1), dt
            err = p.stderr.decode("utf-8", "replace")[-140:]
            op.unlink(missing_ok=True)
        cand = [0, 1, 2, 3]; _galosh_dev = None   # cached device failed -> full re-probe
    ip.unlink(missing_ok=True)
    raise RuntimeError(f"galosh_yuv_gpu failed on all devices: {err}")

# ---- classical, proper COLOR (not per-channel), BLIND sigma=estimate_sigma ----
def m_cbm3d(noisy, sigma):              # CBM3D = colour BM3D (opponent transform, joint)
    import bm3d as B
    t0 = time.perf_counter()
    with _bm3d_lock:
        den = B.bm3d_rgb(noisy.astype(np.float64), sigma_psd=sigma)
    return np.clip(den, 0, 1).astype(np.float32), time.perf_counter() - t0

def m_color_nlm(noisy, sigma):          # cv2 colour Non-Local Means (luma+chroma h)
    import cv2
    t0 = time.perf_counter()
    u8 = (np.clip(noisy, 0, 1) * 255 + 0.5).astype(np.uint8)
    h = float(max(sigma * 255.0, 1.0))
    with _nlm_lock:
        d = cv2.fastNlMeansDenoisingColored(u8, None, h, h, 7, 21)
    return (d.astype(np.float32) / 255.0), time.perf_counter() - t0

def m_guided(noisy, sigma):             # self-guided colour filter (He et al.), per-channel guide
    import cv2
    t0 = time.perf_counter()
    g = noisy.astype(np.float32); eps = float(max(sigma, 1e-3)) ** 2; r = 4
    out = []
    for c in range(3):
        I = g[..., c]
        mean = cv2.boxFilter(I, -1, (2*r+1, 2*r+1)); corr = cv2.boxFilter(I*I, -1, (2*r+1, 2*r+1))
        var = corr - mean*mean; a = var/(var+eps); b = mean - a*mean
        ma = cv2.boxFilter(a, -1, (2*r+1, 2*r+1)); mb = cv2.boxFilter(b, -1, (2*r+1, 2*r+1))
        out.append(ma*I + mb)
    return np.clip(np.stack(out, 2), 0, 1).astype(np.float32), time.perf_counter() - t0

# ---- DL references: sRGB COLOUR, BLIND (no sigma). NAFNet-SIDD / SCUNet-real / Restormer-SIDD ----
_nafnet = _scunet = _restormer = None
def _dl_run(model, noisy, mult=1):
    import torch, torch.nn.functional as F
    H, W = noisy.shape[:2]
    if H * W > _DL_TILE_PX:                 # full-frame (15.8MP) -> tiled inference (VRAM)
        return _dl_run_tiled(model, noisy, mult)
    x = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    if mult > 1:
        ph = (mult - H % mult) % mult; pw = (mult - W % mult) % mult
        if ph or pw: x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    with torch.no_grad():
        y = model(x)
    y = y[:, :, :H, :W]
    return y.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy().astype(np.float32)

# Whole-image DL inference OOMs on 15.8MP; standard overlapped-tile protocol with
# feathered blending (each tile is still >= the nets' 256^2 train patch).
_DL_TILE_PX = 2_500_000
def _dl_run_tiled(model, noisy, mult=1, ts=1024, ov=96):
    H, W = noisy.shape[:2]
    out = np.zeros((H, W, 3), np.float64); wsum = np.zeros((H, W, 1), np.float64)
    step = ts - ov
    ys = list(range(0, max(H - ts, 0) + 1, step)); xs = list(range(0, max(W - ts, 0) + 1, step))
    if not ys or ys[-1] + ts < H: ys.append(max(H - ts, 0))
    if not xs or xs[-1] + ts < W: xs.append(max(W - ts, 0))
    ramp = np.minimum(np.arange(1, ts + 1), np.arange(ts, 0, -1)).astype(np.float64)
    ramp = np.minimum(ramp / max(ov, 1), 1.0)
    win2d = np.outer(ramp, ramp)[:, :, None]
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            h, w = min(ts, H - y), min(ts, W - x)
            tile = np.ascontiguousarray(noisy[y:y+h, x:x+w])
            d = _dl_run(model, tile, mult)          # tile <= _DL_TILE_PX -> direct path
            wv = win2d[:h, :w]
            out[y:y+h, x:x+w] += d.astype(np.float64) * wv
            wsum[y:y+h, x:x+w] += wv
    return (out / np.maximum(wsum, 1e-12)).astype(np.float32)

def m_nafnet(noisy):
    global _nafnet
    if _nafnet is None:
        sys.path.insert(0, str(GALOSH / "benchmark" / "scripts"))
        from nafnet_standalone import load_nafnet_sidd
        _nafnet = load_nafnet_sidd(GALOSH / "benchmark/external/NAFNet/experiments/pretrained_models/NAFNet-SIDD-width64.pth", DEVICE)
    t0 = time.perf_counter(); d = _dl_run(_nafnet, noisy, mult=16); return d, time.perf_counter() - t0

def m_scunet(noisy):
    global _scunet
    import torch
    if _scunet is None:
        sys.path.insert(0, str(GALOSH / "benchmark/external/SCUNet"))
        from models.network_scunet import SCUNet
        m = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
        sd = torch.load(str(GALOSH / "benchmark/external/SCUNet/model_zoo/scunet_color_real_psnr.pth"),
                        map_location="cpu", weights_only=False)
        m.load_state_dict(sd, strict=True); _scunet = m.eval().to(DEVICE)
    t0 = time.perf_counter(); d = _dl_run(_scunet, noisy, mult=64); return d, time.perf_counter() - t0

def m_restormer(noisy):
    global _restormer
    import torch
    if _restormer is None:
        sys.path.insert(0, str(GALOSH / "benchmark/external/Restormer/basicsr/models/archs"))
        import restormer_arch
        m = restormer_arch.Restormer(LayerNorm_type='BiasFree')  # real-denoising config
        sd = torch.load(str(GALOSH / "benchmark/external/Restormer/Denoising/pretrained_models/real_denoising.pth"),
                        map_location="cpu", weights_only=False)
        m.load_state_dict(sd.get("params", sd), strict=True); _restormer = m.eval().to(DEVICE)
    t0 = time.perf_counter(); d = _dl_run(_restormer, noisy, mult=8); return d, time.perf_counter() - t0

def build_methods(want):
    """ALL BLIND, sRGB COLOUR. Classical sigma = skimage estimate_sigma (blind)."""
    G = GALOSH / "benchmark/external"
    M = {}
    M["galosh_yuv_gpu_fp32"] = ("GALOSH-YUV GPU FP32", lambda n, s, u: m_galosh(n, u))
    M["cbm3d"]     = ("CBM3D",        lambda n, s, u: m_cbm3d(n, s))
    M["color_nlm"] = ("Color-NLM",    lambda n, s, u: m_color_nlm(n, s))
    M["guided"]    = ("Guided",       lambda n, s, u: m_guided(n, s))
    if (G / "NAFNet/experiments/pretrained_models/NAFNet-SIDD-width64.pth").exists():
        M["nafnet"]    = ("NAFNet-SIDD",   lambda n, s, u: m_nafnet(n))
    if (G / "SCUNet/model_zoo/scunet_color_real_psnr.pth").exists():
        M["scunet"]    = ("SCUNet-real",   lambda n, s, u: m_scunet(n))
    if (G / "Restormer/Denoising/pretrained_models/real_denoising.pth").exists():
        M["restormer"] = ("Restormer-SIDD", lambda n, s, u: m_restormer(n))
    return {k: v for k, v in M.items() if (not want or k in want)}

# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sidd", "rawnind"], default="sidd")
    ap.add_argument("--scenes", type=int, default=0, help="limit scene count (0=all)")
    ap.add_argument("--methods", nargs="*", default=[])
    ap.add_argument("--no-perceptual", action="store_true")
    ap.add_argument("--crop", type=int, default=1024,
                    help="central NxN crop (0=full frame). SIDD-sRGB is 15.8MP full; a crop "
                         "keeps the classical baselines (BM3D/NLM) + GPU perceptual metrics "
                         "tractable. NOT the RAW full-frame convention — flagged in _table.txt.")
    args = ap.parse_args()

    def crop_c(im, n):
        if not n: return im
        H, W = im.shape[:2]
        if H <= n and W <= n: return im
        y0 = max(0, (H - n) // 2); x0 = max(0, (W - n) // 2)
        return np.ascontiguousarray(im[y0:y0 + min(n, H), x0:x0 + min(n, W)])

    OUT = GALOSH / "benchmark" / ("results_srgb_sidd_full" if (args.dataset == "sidd" and not args.crop)
                              else OUT_NAME[args.dataset])
    TMP.mkdir(parents=True, exist_ok=True)
    methods = build_methods(set(args.methods))
    total = scene_count(args.dataset, args.scenes)

    print(f"=== GALOSH-YUV sRGB bench ({args.dataset}) ===")
    print(f"  exe     : {EXE_GPU}")
    print(f"  scenes  : {total}    out: {OUT}")
    print(f"  methods : {list(methods)}")
    print(f"  metrics : PSNR SSIM" + ("" if args.no_perceptual else " LPIPS DISTS NIQE"), flush=True)

    rows = []
    for si, (tag, noisy, gt) in enumerate(iter_scenes(args.dataset, args.scenes)):
        noisy = crop_c(noisy, args.crop); gt = crop_c(gt, args.crop)
        sigma = est_sigma(noisy)               # BLIND sigma for classical (CBM3D/Color-NLM/Guided)
        # persist GT + noisy once
        for nm, im in (("_gt", gt), ("_noisy", noisy)):
            d = OUT / nm[1:]; d.mkdir(parents=True, exist_ok=True)
            Image.fromarray((np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8)).save(d / f"{tag}.png")
        # noisy baseline metrics
        nr = {"tag": tag, "method": "_noisy", "psnr": psnr(noisy, gt), "ssim": ssim_rgb(noisy, gt)}
        if not args.no_perceptual:
            nr.update(lpips=lpips_m(noisy, gt), dists=dists_m(noisy, gt), niqe=niqe_m(noisy))
        rows.append(nr)
        for key, (label, fn) in methods.items():
            try:
                den, dt = fn(noisy, sigma, f"{tag}_{key}")
            except Exception as e:
                print(f"  [{tag}] {key}: FAIL {str(e)[-160:]}", flush=True)
                rows.append({"tag": tag, "method": key, "ok": False, "err": str(e)[-200:]})
                continue
            md = OUT / key; md.mkdir(parents=True, exist_ok=True)
            Image.fromarray((den * 255 + 0.5).astype(np.uint8)).save(md / f"{tag}.png")  # sRGB -> PNG (feedback_png_persistence)
            r = {"tag": tag, "method": key, "ok": True, "ms": round(dt * 1000, 1),
                 "psnr": psnr(den, gt), "ssim": ssim_rgb(den, gt)}
            if not args.no_perceptual:
                r.update(lpips=lpips_m(den, gt), dists=dists_m(den, gt), niqe=niqe_m(den))
            rows.append(r)
            pm = f"L={r['lpips']:.4f} D={r['dists']:.4f} NIQE={r['niqe']:.2f}" if not args.no_perceptual else ""
            print(f"  [{si+1}/{total} {tag} {key:<20s}] {r['ms']:7.1f}ms "
                  f"PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f} {pm}", flush=True)
        try:
            import torch; torch.cuda.empty_cache()  # free torch GPU cache so galosh OpenCL can allocate
        except Exception: pass
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "_metrics.json").write_text(json.dumps(rows, indent=1))  # checkpoint each scene

    # ---- aggregate consolidated table ----
    agg = {}
    for r in rows:
        if r.get("ok") is False: continue
        agg.setdefault(r["method"], []).append(r)
    keys = ["psnr", "ssim", "lpips", "dists", "niqe"]
    hdr = f"{'method':<22}{'N':>4}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}{'DISTS':>8}{'NIQE':>7}{'ms':>9}"
    dsdesc = {"sidd": (f"SIDD-Medium sRGB, central {args.crop}x{args.crop} crop (full=15.8MP)"
                       if args.crop else
                       "SIDD-Medium sRGB, FULL FRAME (~15.8MP; DL tiled 1024/ov96; LPIPS/DISTS = 1024-tile mean)"),
              "rawnind": "RawNIND-rendered sRGB, 512x512 full"}[args.dataset]
    note = (f"{dsdesc} | N={total} | ALL methods BLIND, sRGB colour | CBM3D/Color-NLM/Guided "
            f"sigma=estimate_sigma; GALOSH/NAFNet-SIDD/SCUNet-real/Restormer-SIDD self-contained")
    lines = [note, "", hdr, "-" * len(hdr)]
    order = ["_noisy"] + list(methods)
    for m in order:
        rs = agg.get(m, [])
        if not rs: continue
        def mean(k):
            v = [x[k] for x in rs if k in x]; return np.mean(v) if v else float("nan")
        ms = [x["ms"] for x in rs if "ms" in x]
        lines.append(f"{m:<22}{len(rs):>4}{mean('psnr'):>8.2f}{mean('ssim'):>8.4f}"
                     f"{mean('lpips'):>8.4f}{mean('dists'):>8.4f}{mean('niqe'):>7.2f}"
                     f"{(np.mean(ms) if ms else float('nan')):>9.1f}")
    table = "\n".join(lines)
    print("\n=== Aggregate (lower LPIPS/DISTS/NIQE better; higher PSNR/SSIM better) ===")
    print(table)
    (OUT / "_table.txt").write_text(table)
    (OUT / "_metrics.json").write_text(json.dumps(rows, indent=1))
    print(f"\nWrote {OUT/'_metrics.json'} and {OUT/'_table.txt'}")

if __name__ == "__main__":
    main()
