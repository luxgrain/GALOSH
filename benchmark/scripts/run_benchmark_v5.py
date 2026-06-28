#!/usr/bin/env python3
"""
RAW L/C BM3D -- Unified Pipeline Benchmark v5
IEEE SPL / IPOL submission benchmark.

v5 changes vs v4:
  - Added DISTS (Deep Image Structure and Texture Similarity) metric
  - Added NIQE (No-Reference Natural Image Quality Evaluator) metric
  - Fixed Unicode output for Windows cp932 console

Methods:
  1. No-denoise (baseline)
  2. BM3D-CFA (blind GAT, darktable rawdenoise.c)
  3. RAW L/C BM3D (blind GAT, darktable rawdenoise.c)
  4. DnCNN-B (blind, sRGB domain)
  5. DRUNet (non-blind sigma-input, sRGB domain)

Unified pipeline:
  Pre-demosaic methods:  RAW -> [denoise] -> demosaic -> sRGB
  Post-demosaic methods: RAW -> demosaic -> sRGB -> [denoise]
  Same demosaic for all. Metrics on final sRGB vs GT sRGB.

Dataset: Kodak 24 with synthetic Poisson-Gaussian noise.

Parallelized: images processed concurrently (N_PARALLEL workers).
"""
import numpy as np
import subprocess
import os
import sys
import json
import time
import torch
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.io import imread, imsave
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import uniform_filter

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

# Use half of available cores for torch, leave rest for C exe (OpenMP)
n_cores = os.cpu_count() or 4
torch.set_num_threads(max(n_cores // 2, 2))

BASE = Path(__file__).parent.parent
EXE = str(BASE / "standalone" / "rawdenoise_v4.exe")
EXE_V6 = str(BASE / "standalone" / "rawdenoise_v6.exe")
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

DATASETS = {
    "kodak":   {"path": BASE / "datasets" / "kodak",   "glob": "kodim*.png", "max": 24},
    "cbsd68":  {"path": BASE / "datasets" / "cbsd68",  "glob": "*.png",      "max": 68},
    "mcmaster":{"path": BASE / "datasets" / "mcmaster", "glob": "*.png",     "max": 18},
}

KAIR_DIR = Path(os.path.expanduser(r"~\KAIR"))

# ===================== COLOR / NOISE UTILITIES =====================

def inv_srgb(x):
    """sRGB -> linear."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    """linear -> sRGB."""
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(np.maximum(x, 0), 1/2.4) - 0.055)

def demosaic_bilinear(bayer):
    """Simple bilinear demosaic (RGGB). Same for all methods."""
    h, w = bayer.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[0::2, 0::2, 0] = bayer[0::2, 0::2]       # R
    rgb[0::2, 1::2, 1] = bayer[0::2, 1::2]       # Gr
    rgb[1::2, 0::2, 1] = bayer[1::2, 0::2]       # Gb
    rgb[1::2, 1::2, 2] = bayer[1::2, 1::2]       # B
    for c in range(3):
        mask = np.zeros((h, w), dtype=np.float32)
        if c == 0:   mask[0::2, 0::2] = 1
        elif c == 1: mask[0::2, 1::2] = 1; mask[1::2, 0::2] = 1
        else:        mask[1::2, 1::2] = 1
        num = uniform_filter(rgb[:, :, c], size=3)
        den = uniform_filter(mask, size=3)
        rgb[:, :, c] = np.where(mask > 0, rgb[:, :, c], num / np.maximum(den, 1e-10))
    return np.clip(rgb, 0, 1)

def synth_noise(clean_raw, iso, rng):
    """Synthesise Poisson-Gaussian noise at given ISO."""
    gain = iso / 100.0
    alpha = 0.001 * gain
    sigma_read = 0.002 * gain
    sigma_sq = sigma_read ** 2
    rate = np.maximum(clean_raw / max(alpha, 1e-10), 0.0)
    noisy = rng.poisson(rate).astype(np.float32) * alpha
    noisy += rng.standard_normal(clean_raw.shape).astype(np.float32) * sigma_read
    return np.clip(noisy, 0, 1).astype(np.float32), alpha, sigma_sq

def raw_to_srgb(bayer):
    """Unified pipeline: demosaic -> clip -> sRGB gamma."""
    return np.clip(linear_to_srgb(demosaic_bilinear(bayer)), 0, 1).astype(np.float32)

# ===================== METRICS =====================

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0

def ssim_rgb(a, b):
    return float(ssim(a, b, data_range=1.0, channel_axis=2))

_lpips_fn = None
def compute_lpips(a, b):
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net='alex', verbose=False)
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
    tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
    with torch.no_grad():
        return _lpips_fn(ta, tb).item()

_dists_fn = None
def compute_dists(a, b):
    global _dists_fn
    if _dists_fn is None:
        import pyiqa
        _dists_fn = pyiqa.create_metric('dists', device='cpu')
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float()
    tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        return _dists_fn(ta, tb).item()

_niqe_fn = None
def compute_niqe(a):
    global _niqe_fn
    if _niqe_fn is None:
        import pyiqa
        _niqe_fn = pyiqa.create_metric('niqe', device='cpu')
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        return _niqe_fn(ta).item()

# ===================== METHOD: C STANDALONE (ours / bm3d-cfa) =====================

def method_c_standalone(noisy, w, h, method_name, alpha, sigma_sq,
                        strength=1.0, ls=1.0, cs=1.0, uid="0", exe=None):
    """Run C standalone exe (blind GAT estimation). Returns denoised Bayer.
    strength: BM3D-CFA overall strength (argv[6]).
    ls/cs: luma/chroma strength for RAW L/C BM3D.
    uid: unique id to avoid tmp file collisions in parallel.
    exe: override executable path (default: EXE = rawdenoise_v4)."""
    if exe is None:
        exe = EXE
    inp = str(BASE / "standalone" / f"tmp_bench_{uid}_{method_name}.bin")
    out = str(BASE / "standalone" / f"tmp_out_{uid}_{method_name}.bin")
    noisy.tofile(inp)
    cmd = [exe, inp, out, str(w), str(h), method_name,
           str(strength), str(ls), str(cs), str(alpha), str(sigma_sq), "1", "25"]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    # Cleanup input file
    try: os.remove(inp)
    except: pass
    if result.returncode != 0:
        raise RuntimeError(f"C standalone failed: {result.stderr.decode()}")
    den = np.fromfile(out, dtype=np.float32).reshape(h, w)
    try: os.remove(out)
    except: pass
    return den

# ===================== METHOD: DnCNN-B (blind) =====================

_dncnn_model = None
def get_dncnn():
    global _dncnn_model
    if _dncnn_model is not None:
        return _dncnn_model
    sys.path.insert(0, str(KAIR_DIR))
    from models.network_dncnn import DnCNN
    model = DnCNN(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='R')
    state = torch.load(str(KAIR_DIR / "model_zoo" / "dncnn_gray_blind.pth"),
                       map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _dncnn_model = model
    return model

def method_dncnn_b(noisy_srgb):
    """DnCNN-B: blind denoising on grayscale, applied per-channel."""
    model = get_dncnn()
    out_channels = []
    for c in range(3):
        ch = noisy_srgb[:, :, c]
        inp = torch.from_numpy(ch).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            res = model(inp)
        out_channels.append(res.squeeze().clamp(0, 1).numpy())
    return np.stack(out_channels, axis=2).astype(np.float32)

# ===================== METHOD: DRUNet (non-blind, sigma input) =====================

_drunet_model = None
def get_drunet():
    global _drunet_model
    if _drunet_model is not None:
        return _drunet_model
    sys.path.insert(0, str(KAIR_DIR))
    from models.network_unet import UNetRes
    model = UNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512],
                    nb=4, act_mode='R', downsample_mode='strideconv',
                    upsample_mode='convtranspose', bias=False)
    state = torch.load(str(KAIR_DIR / "model_zoo" / "drunet_gray.pth"),
                       map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _drunet_model = model
    return model

def method_drunet(noisy_srgb, sigma_est):
    """DRUNet: non-blind denoising with sigma map input. Per-channel on sRGB."""
    model = get_drunet()
    h, w = noisy_srgb.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    out_channels = []
    for c in range(3):
        ch = noisy_srgb[:, :, c]
        if pad_h or pad_w:
            ch = np.pad(ch, ((0, pad_h), (0, pad_w)), mode='reflect')
        sigma_map = np.full_like(ch, sigma_est)
        inp = np.stack([ch, sigma_map], axis=0)
        inp_t = torch.from_numpy(inp).unsqueeze(0).float()
        with torch.no_grad():
            res = model(inp_t)
        out_ch = res.squeeze().clamp(0, 1).numpy()
        if pad_h or pad_w:
            out_ch = out_ch[:h, :w]
        out_channels.append(out_ch)
    return np.stack(out_channels, axis=2).astype(np.float32)

def estimate_srgb_sigma(noisy_srgb):
    """Estimate noise sigma in sRGB domain using MAD of Laplacian."""
    gray = np.mean(noisy_srgb, axis=2)
    from scipy.signal import convolve2d
    kernel = np.array([1, -2, 1], dtype=np.float64).reshape(1, 3)
    lap = convolve2d(gray.astype(np.float64), kernel, mode='valid')
    sigma = np.median(np.abs(lap)) / 0.6745 / np.sqrt(6)
    return max(float(sigma), 0.001)


# ===================== PER-IMAGE WORKER =====================

METRIC_KEYS = ["psnr", "ssim", "lpips", "dists", "niqe"]

def process_one_image(img_path, idx, iso, alpha, sigma_sq, c_ok, dncnn_ok, drunet_ok, galosh_ok=False):
    """Process a single image for all methods. Returns dict of results.
    Methods run sequentially within each image; parallelism is at image level."""
    rng = np.random.default_rng(42 + idx)

    img = imread(str(img_path)).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    h -= h % 2; w -= w % 2
    img = img[:h, :w]
    linear = inv_srgb(img).astype(np.float32)

    clean_raw = np.zeros((h, w), dtype=np.float32)
    clean_raw[0::2, 0::2] = linear[0::2, 0::2, 0]
    clean_raw[0::2, 1::2] = linear[0::2, 1::2, 1]
    clean_raw[1::2, 0::2] = linear[1::2, 0::2, 1]
    clean_raw[1::2, 1::2] = linear[1::2, 1::2, 2]

    noisy_raw, _, _ = synth_noise(clean_raw, iso, rng)
    gt_srgb = raw_to_srgb(clean_raw)
    noisy_srgb = raw_to_srgb(noisy_raw)

    results_srgb = {}
    timings = {}
    uid = f"iso{iso}_img{idx}"

    # No-denoise
    results_srgb["No-denoise"] = noisy_srgb

    # BM3D-CFA (s=1.0)
    if c_ok:
        try:
            t0 = time.time()
            den = method_c_standalone(noisy_raw, w, h, "bm3dcfa",
                                      alpha, sigma_sq, strength=1.0, uid=uid)
            timings["BM3D-CFA"] = time.time() - t0
            results_srgb["BM3D-CFA"] = raw_to_srgb(den)
        except Exception as e:
            print(f"  BM3D-CFA failed on {img_path.name} ISO{iso}: {e}")
            results_srgb["BM3D-CFA"] = noisy_srgb

    # BM3D-CFA (s=0.5)
    if c_ok:
        try:
            t0 = time.time()
            den = method_c_standalone(noisy_raw, w, h, "bm3dcfa",
                                      alpha, sigma_sq, strength=0.5,
                                      uid=f"{uid}_s05")
            timings["BM3D-CFA (s=0.5)"] = time.time() - t0
            results_srgb["BM3D-CFA (s=0.5)"] = raw_to_srgb(den)
        except Exception as e:
            print(f"  BM3D-CFA (s=0.5) failed on {img_path.name} ISO{iso}: {e}")
            results_srgb["BM3D-CFA (s=0.5)"] = noisy_srgb

    # RAW L/C BM3D
    if c_ok:
        try:
            t0 = time.time()
            den = method_c_standalone(noisy_raw, w, h, "ours",
                                      alpha, sigma_sq, ls=0.5, cs=1.0,
                                      uid=uid)
            timings["RAW L/C BM3D"] = time.time() - t0
            results_srgb["RAW L/C BM3D"] = raw_to_srgb(den)
        except Exception as e:
            print(f"  RAW L/C BM3D failed on {img_path.name} ISO{iso}: {e}")
            results_srgb["RAW L/C BM3D"] = noisy_srgb

    # GALOSH
    if galosh_ok:
        try:
            t0 = time.time()
            den = method_c_standalone(noisy_raw, w, h, "galosh",
                                      alpha, sigma_sq, ls=0.5, cs=1.0,
                                      uid=uid, exe=EXE_V6)
            timings["GALOSH"] = time.time() - t0
            results_srgb["GALOSH"] = raw_to_srgb(den)
        except Exception as e:
            print(f"  GALOSH failed on {img_path.name} ISO{iso}: {e}")
            results_srgb["GALOSH"] = noisy_srgb

    # DnCNN-B
    if dncnn_ok:
        try:
            t0 = time.time()
            results_srgb["DnCNN-B"] = method_dncnn_b(noisy_srgb)
            timings["DnCNN-B"] = time.time() - t0
        except Exception as e:
            print(f"  DnCNN-B failed on {img_path.name} ISO{iso}: {e}")
            results_srgb["DnCNN-B"] = noisy_srgb

    # DRUNet
    if drunet_ok:
        try:
            t0 = time.time()
            sigma_est = estimate_srgb_sigma(noisy_srgb)
            results_srgb["DRUNet"] = method_drunet(noisy_srgb, sigma_est)
            timings["DRUNet"] = time.time() - t0
        except Exception as e:
            print(f"  DRUNet failed on {img_path.name} ISO{iso}: {e}")
            results_srgb["DRUNet"] = noisy_srgb

    # Compute metrics
    metrics = {}
    for m_name, den_srgb in results_srgb.items():
        p = psnr(gt_srgb, den_srgb)
        s = ssim_rgb(gt_srgb, den_srgb)
        lp = compute_lpips(gt_srgb, den_srgb)
        di = compute_dists(gt_srgb, den_srgb)
        nq = compute_niqe(den_srgb)
        metrics[m_name] = {"psnr": p, "ssim": s, "lpips": lp, "dists": di, "niqe": nq}

    return {
        "gt_srgb": gt_srgb,
        "results_srgb": results_srgb,
        "metrics": metrics,
        "timings": timings,
    }


# ===================== MAIN BENCHMARK =====================

def _fmt(val, width=7, decimals=4):
    """Format a metric value, handling NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return " " * (width - 3) + "N/A"
    return f"{val:>{width}.{decimals}f}"


def main():
    parser = argparse.ArgumentParser(description="RAW L/C BM3D Benchmark v5")
    parser.add_argument("--dataset", "-d", default="kodak",
                        choices=list(DATASETS.keys()),
                        help="Dataset to benchmark (default: kodak)")
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    ds_name = args.dataset
    IMGDIR = ds["path"]
    OUTDIR = BASE / f"comparison_images_v6_{ds_name}"
    OUTDIR.mkdir(exist_ok=True)

    isos = [400, 800, 1600, 3200, 6400, 12800]
    images = sorted(IMGDIR.glob(ds["glob"]))[:ds["max"]]
    n_images = len(images)

    if not images:
        print(f"ERROR: No images found in {IMGDIR}")
        return

    methods = [
        "No-denoise",
        "BM3D-CFA",
        "BM3D-CFA (s=0.5)",
        "RAW L/C BM3D",
        "GALOSH",
        "DnCNN-B",
        "DRUNet",
    ]

    print(f"Benchmark v5 [{ds_name}] (unified pipeline + DISTS/NIQE): {n_images} images x {len(isos)} ISOs")
    print(f"Methods: {', '.join(methods)}")
    print(f"Metrics: PSNR, SSIM, LPIPS, DISTS, NIQE (all in sRGB domain)")
    print(f"Parallelization: {n_cores} cores, image-level parallelism")
    print(f"Torch threads: {torch.get_num_threads()}")
    print()

    # Preload models
    print("Loading DnCNN-B model...", end=" ", flush=True)
    try:
        get_dncnn(); print("OK")
        dncnn_ok = True
    except Exception as e:
        print(f"FAILED: {e}"); dncnn_ok = False

    print("Loading DRUNet model...", end=" ", flush=True)
    try:
        get_drunet(); print("OK")
        drunet_ok = True
    except Exception as e:
        print(f"FAILED: {e}"); drunet_ok = False

    # Preload LPIPS
    print("Loading LPIPS (AlexNet)...", end=" ", flush=True)
    try:
        compute_lpips(np.zeros((64,64,3), dtype=np.float32),
                      np.zeros((64,64,3), dtype=np.float32))
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")

    # Preload DISTS
    print("Loading DISTS...", end=" ", flush=True)
    try:
        compute_dists(np.zeros((64,64,3), dtype=np.float32),
                      np.zeros((64,64,3), dtype=np.float32))
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")

    # Preload NIQE
    print("Loading NIQE...", end=" ", flush=True)
    try:
        compute_niqe(np.ones((64,64,3), dtype=np.float32) * 0.5)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")

    print(f"C standalone (v4): {EXE}", end=" ")
    c_ok = os.path.isfile(EXE)
    print("OK" if c_ok else "NOT FOUND")

    print(f"C standalone (v6 GALOSH): {EXE_V6}", end=" ")
    galosh_ok = os.path.isfile(EXE_V6)
    print("OK" if galosh_ok else "NOT FOUND")
    print(flush=True)

    N_PARALLEL = 4  # Process 4 images concurrently (DRUNet is memory-heavy on CPU)

    all_results = []
    all_timing = {m: [] for m in methods}

    for iso in isos:
        t_iso_start = time.time()
        gain = iso / 100.0
        alpha = 0.001 * gain
        sigma_sq = (0.002 * gain) ** 2

        iso_metrics = {m: {k: [] for k in METRIC_KEYS} for m in methods}

        # Submit all images for this ISO in parallel (N_PARALLEL at a time)
        image_results = [None] * n_images
        with ThreadPoolExecutor(max_workers=N_PARALLEL) as img_pool:
            img_futures = {}
            for idx, img_path in enumerate(images):
                fut = img_pool.submit(process_one_image, img_path, idx, iso,
                                       alpha, sigma_sq, c_ok, dncnn_ok, drunet_ok,
                                       galosh_ok)
                img_futures[fut] = idx

            for fut in as_completed(img_futures):
                idx = img_futures[fut]
                try:
                    image_results[idx] = fut.result()
                except Exception as e:
                    print(f"  Image {idx} failed at ISO {iso}: {e}", flush=True)
                done_count = sum(1 for r in image_results if r is not None)
                if done_count % 6 == 0:
                    print(f"  ISO {iso}: {done_count}/{n_images} done...", flush=True)

        # Collect results in order
        for idx, res in enumerate(image_results):
            if res is None:
                continue

            for m_name, met in res["metrics"].items():
                for k in METRIC_KEYS:
                    iso_metrics[m_name][k].append(met[k])

            for m_name, t in res["timings"].items():
                all_timing[m_name].append(t)

            # Save comparison images (all images, all methods)
            img_stem = images[idx].stem
            iso_dir = OUTDIR / f"iso{iso}"
            iso_dir.mkdir(exist_ok=True)
            imsave(str(iso_dir / f"{img_stem}_gt.png"),
                   (np.clip(res["gt_srgb"], 0, 1) * 255).astype(np.uint8))
            imsave(str(iso_dir / f"{img_stem}_Noisy.png"),
                   (np.clip(res["results_srgb"]["No-denoise"], 0, 1) * 255).astype(np.uint8))
            for m_name, den_srgb in res["results_srgb"].items():
                mfname = m_name.replace(" ", "_").replace("/", "-")
                imsave(str(iso_dir / f"{img_stem}_{mfname}.png"),
                       (np.clip(den_srgb, 0, 1) * 255).astype(np.uint8))

        # Aggregate
        elapsed = time.time() - t_iso_start
        avg = {}
        for m, d in iso_metrics.items():
            avg[m] = {}
            for k, v in d.items():
                vals = [x for x in v if not (isinstance(x, float) and x != x)]
                avg[m][k] = float(np.mean(vals)) if vals else float('nan')

        result_row = {"iso": iso}
        for m in methods:
            key = m.replace(" ", "_").replace("/", "-")
            for k in METRIC_KEYS:
                dec = 2 if k == "psnr" else 4
                val = avg[m].get(k, float('nan'))
                result_row[f"{key}_{k}"] = round(val, dec) if m in avg and not np.isnan(val) else None
        all_results.append(result_row)

        # Print table
        print(f"\nISO {iso:>5} ({elapsed:.0f}s):")
        print(f"  {'Method':<16} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'DISTS':>7} {'NIQE':>7}")
        print(f"  {'-'*58}")
        for m in methods:
            if m not in avg: continue
            a = avg[m]
            print(f"  {m:<16} {_fmt(a.get('psnr'), 7, 2)} {_fmt(a.get('ssim'))} {_fmt(a.get('lpips'))} {_fmt(a.get('dists'))} {_fmt(a.get('niqe'))} ")
        print(flush=True)

        # Save intermediate results
        out_path = RESULTS / f"v6_{ds_name}_results.json"
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # ===================== SUMMARY TABLE =====================
    print("=" * 100)
    print(f"SUMMARY -- Unified Pipeline Benchmark v5 [{ds_name}]")
    print("All metrics computed in sRGB domain. Same demosaic for all methods.")
    print()

    # Per-ISO PSNR/LPIPS summary
    hdr = f"{'ISO':>6}"
    for m in methods:
        hdr += f"  {m:>16}"
    print(hdr)
    print("-" * 100)
    for r in all_results:
        line = f"{r['iso']:>6}"
        for m in methods:
            key = m.replace(" ", "_").replace("/", "-")
            p = r.get(f"{key}_psnr")
            l = r.get(f"{key}_lpips")
            if p is not None and l is not None:
                line += f"  {p:>6.2f}/{l:.4f}"
            else:
                line += f"       N/A      "
        print(line)
    print()

    # Full average table
    print("Average across all ISOs:")
    print(f"  {'Method':<16} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'DISTS':>7} {'NIQE':>7}  {'Time(s)':>8}")
    print(f"  {'-'*70}")
    for m in methods:
        key = m.replace(" ", "_").replace("/", "-")
        avgs = {}
        for k in METRIC_KEYS:
            vals = [r[f"{key}_{k}"] for r in all_results if r.get(f"{key}_{k}") is not None]
            avgs[k] = np.mean(vals) if vals else float('nan')
        avg_time = np.mean(all_timing[m]) if all_timing[m] else float('nan')
        print(f"  {m:<16} {_fmt(avgs['psnr'], 7, 2)} {_fmt(avgs['ssim'])} {_fmt(avgs['lpips'])} {_fmt(avgs['dists'])} {_fmt(avgs['niqe'])}  {avg_time:>8.2f}")
    print()

    # ===================== SEQUENTIAL TIMING (for paper) =====================
    # Run 3 images sequentially at ISO 1600 for fair timing comparison
    print("=" * 100)
    print("SEQUENTIAL TIMING (for paper -- no parallel contention)")
    timing_iso = 1600
    gain = timing_iso / 100.0
    t_alpha = 0.001 * gain
    t_sigma_sq = (0.002 * gain) ** 2
    n_timing = min(3, n_images)
    seq_timing = {m: [] for m in methods if m != "No-denoise"}

    for idx in range(n_timing):
        img_path = images[idx]
        rng = np.random.default_rng(42 + idx)
        img = imread(str(img_path)).astype(np.float32) / 255.0
        h, w = img.shape[:2]
        h -= h % 2; w -= w % 2
        img = img[:h, :w]
        linear = inv_srgb(img).astype(np.float32)
        clean_raw = np.zeros((h, w), dtype=np.float32)
        clean_raw[0::2, 0::2] = linear[0::2, 0::2, 0]
        clean_raw[0::2, 1::2] = linear[0::2, 1::2, 1]
        clean_raw[1::2, 0::2] = linear[1::2, 0::2, 1]
        clean_raw[1::2, 1::2] = linear[1::2, 1::2, 2]
        noisy_raw, _, _ = synth_noise(clean_raw, timing_iso, rng)
        noisy_srgb = raw_to_srgb(noisy_raw)
        uid = f"timing_{idx}"

        if c_ok:
            t0 = time.time()
            method_c_standalone(noisy_raw, w, h, "bm3dcfa", t_alpha, t_sigma_sq,
                                strength=1.0, uid=uid)
            seq_timing["BM3D-CFA"].append(time.time() - t0)

            t0 = time.time()
            method_c_standalone(noisy_raw, w, h, "bm3dcfa", t_alpha, t_sigma_sq,
                                strength=0.5, uid=f"{uid}_s05")
            seq_timing["BM3D-CFA (s=0.5)"].append(time.time() - t0)

            t0 = time.time()
            method_c_standalone(noisy_raw, w, h, "ours", t_alpha, t_sigma_sq,
                                ls=0.5, cs=1.0, uid=uid)
            seq_timing["RAW L/C BM3D"].append(time.time() - t0)

        if galosh_ok:
            t0 = time.time()
            method_c_standalone(noisy_raw, w, h, "galosh", t_alpha, t_sigma_sq,
                                ls=0.5, cs=1.0, uid=uid, exe=EXE_V6)
            seq_timing["GALOSH"].append(time.time() - t0)

        if dncnn_ok:
            t0 = time.time()
            method_dncnn_b(noisy_srgb)
            seq_timing["DnCNN-B"].append(time.time() - t0)

        if drunet_ok:
            t0 = time.time()
            sigma_est = estimate_srgb_sigma(noisy_srgb)
            method_drunet(noisy_srgb, sigma_est)
            seq_timing["DRUNet"].append(time.time() - t0)

        print(f"  Timing image {idx+1}/{n_timing} done", flush=True)

    print(f"\nSequential timing (avg of {n_timing} images, ISO {timing_iso}, {images[0].name} size):")
    print(f"  {'Method':<16} {'Time(s)':>8}  {'Relative':>8}")
    print(f"  {'-'*38}")
    base_time = np.mean(seq_timing.get("RAW L/C BM3D", [1.0])) or 1.0
    for m in ["BM3D-CFA", "BM3D-CFA (s=0.5)", "RAW L/C BM3D", "GALOSH", "DnCNN-B", "DRUNet"]:
        if seq_timing.get(m):
            avg_t = np.mean(seq_timing[m])
            rel = avg_t / base_time
            print(f"  {m:<16} {avg_t:>8.3f}  {rel:>7.1f}x")
    print()

    # Save all
    out_path = RESULTS / f"v6_{ds_name}_results.json"
    save_data = {
        "quality": all_results,
        "sequential_timing": {m: float(np.mean(v)) for m, v in seq_timing.items() if v},
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {out_path}")
    print(f"Comparison images saved to {OUTDIR}")


if __name__ == "__main__":
    main()
