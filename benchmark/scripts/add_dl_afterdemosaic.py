#!/usr/bin/env python3
"""
Add DnCNN-B / DRUNet (after-demosaic, sRGB domain) results to v8 GALOSH JSON files.

Pipeline for DL methods (fair comparison):
  RAW (noisy) -> demosaic (bilinear) -> sRGB gamma -> [DL denoise per-channel] -> evaluate

This is the correct evaluation pipeline for sRGB-trained models (DnCNN-B, DRUNet).
Pre-demosaic methods (GALOSH, BM3D-CFA) denoise in RAW domain before demosaic.

Noise synthesis uses the same seed (42+idx) as all other benchmark scripts,
ensuring pixel-identical noisy images for fair metric comparison.

Usage:
  python add_dl_afterdemosaic.py -d kodak
  python add_dl_afterdemosaic.py -d cbsd68
  python add_dl_afterdemosaic.py -d mcmaster
"""
import numpy as np
import os
import sys
import json
import time
import torch
import argparse
from pathlib import Path
from skimage.io import imread, imsave
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import uniform_filter

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

n_cores = os.cpu_count() or 4
torch.set_num_threads(max(n_cores // 2, 2))

BASE = Path(__file__).parent.parent
RESULTS = BASE / "results"
KAIR_DIR = Path(os.path.expanduser(r"~\KAIR"))

DATASETS = {
    "kodak":    {"path": BASE / "datasets" / "kodak",    "glob": "kodim*.png", "max": 24},
    "cbsd68":   {"path": BASE / "datasets" / "cbsd68",   "glob": "*.png",      "max": 68},
    "mcmaster": {"path": BASE / "datasets" / "mcmaster", "glob": "*.png",      "max": 18},
}

# ===================== Color / Noise Utilities =====================

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
    rgb[0::2, 0::2, 0] = bayer[0::2, 0::2]
    rgb[0::2, 1::2, 1] = bayer[0::2, 1::2]
    rgb[1::2, 0::2, 1] = bayer[1::2, 0::2]
    rgb[1::2, 1::2, 2] = bayer[1::2, 1::2]
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

# ===================== Metrics =====================

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

# ===================== DnCNN-B (blind, sRGB after demosaic) =====================

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
    """DnCNN-B: blind denoising per-channel on sRGB (after demosaic).
    This is the correct pipeline for sRGB-trained DnCNN-B."""
    model = get_dncnn()
    out_channels = []
    for c in range(3):
        ch = noisy_srgb[:, :, c]
        inp = torch.from_numpy(ch).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            res = model(inp)
        out_channels.append(res.squeeze().clamp(0, 1).numpy())
    return np.stack(out_channels, axis=2).astype(np.float32)

# ===================== DRUNet (non-blind, sigma input, sRGB after demosaic) =====================

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
    """DRUNet: non-blind denoising with sigma map input. Per-channel on sRGB.
    This is the correct pipeline for sRGB-trained DRUNet."""
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


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(
        description="Add DnCNN-B / DRUNet (after-demosaic sRGB) to v8 GALOSH results")
    parser.add_argument("--dataset", "-d", default="kodak",
                        choices=list(DATASETS.keys()))
    args = parser.parse_args()

    ds_name = args.dataset
    ds = DATASETS[ds_name]
    IMGDIR = ds["path"]

    # Load existing v8 GALOSH results
    v8_path = RESULTS / f"v8_f_10_10_{ds_name}_results.json"
    if not v8_path.exists():
        print(f"ERROR: {v8_path} not found")
        return
    with open(v8_path) as f:
        v8_data = json.load(f)

    images = sorted(IMGDIR.glob(ds["glob"]))[:ds["max"]]
    n_images = len(images)
    isos = [400, 800, 1600, 3200, 6400, 12800]

    # Output directory for DL images
    OUTDIR = BASE / f"comparison_images_v8_{ds_name}"
    OUTDIR.mkdir(exist_ok=True)

    print(f"=" * 70)
    print(f"Adding DnCNN-B / DRUNet (after-demosaic) to v8 [{ds_name}]")
    print(f"Pipeline: RAW -> demosaic -> sRGB -> [DL denoise] -> evaluate")
    print(f"Images: {n_images}, ISOs: {len(isos)}")
    print(f"=" * 70)
    print()

    # Preload models
    print("Loading DnCNN-B model...", end=" ", flush=True)
    try:
        get_dncnn()
        print("OK")
        dncnn_ok = True
    except Exception as e:
        print(f"FAILED: {e}")
        dncnn_ok = False

    print("Loading DRUNet model...", end=" ", flush=True)
    try:
        get_drunet()
        print("OK")
        drunet_ok = True
    except Exception as e:
        print(f"FAILED: {e}")
        drunet_ok = False

    if not dncnn_ok and not drunet_ok:
        print("ERROR: No DL models available")
        return

    # Preload metrics
    print("Loading LPIPS...", end=" ", flush=True)
    compute_lpips(np.zeros((64,64,3), dtype=np.float32),
                  np.zeros((64,64,3), dtype=np.float32))
    print("OK")
    print("Loading DISTS...", end=" ", flush=True)
    compute_dists(np.zeros((64,64,3), dtype=np.float32),
                  np.zeros((64,64,3), dtype=np.float32))
    print("OK")
    print("Loading NIQE...", end=" ", flush=True)
    try:
        compute_niqe(np.ones((64,64,3), dtype=np.float32) * 0.5)
        print("OK")
    except:
        print("SKIP")
    print()

    METRIC_KEYS = ["psnr", "ssim", "lpips", "dists", "niqe"]

    for iso_idx, iso in enumerate(isos):
        t_iso_start = time.time()
        gain = iso / 100.0
        alpha = 0.001 * gain
        sigma_sq = (0.002 * gain) ** 2

        dl_metrics = {}
        if dncnn_ok:
            dl_metrics["DnCNN-B"] = {k: [] for k in METRIC_KEYS}
        if drunet_ok:
            dl_metrics["DRUNet"] = {k: [] for k in METRIC_KEYS}

        for idx, img_path in enumerate(images):
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

            iso_dir = OUTDIR / f"iso{iso}"
            iso_dir.mkdir(exist_ok=True)

            # DnCNN-B (after demosaic)
            if dncnn_ok:
                den_srgb = method_dncnn_b(noisy_srgb)
                p = psnr(gt_srgb, den_srgb)
                s = ssim_rgb(gt_srgb, den_srgb)
                lp = compute_lpips(gt_srgb, den_srgb)
                di = compute_dists(gt_srgb, den_srgb)
                nq = compute_niqe(den_srgb)
                dl_metrics["DnCNN-B"]["psnr"].append(p)
                dl_metrics["DnCNN-B"]["ssim"].append(s)
                dl_metrics["DnCNN-B"]["lpips"].append(lp)
                dl_metrics["DnCNN-B"]["dists"].append(di)
                dl_metrics["DnCNN-B"]["niqe"].append(nq)
                imsave(str(iso_dir / f"{img_path.stem}_DnCNN-B.png"),
                       (np.clip(den_srgb, 0, 1) * 255).astype(np.uint8))

            # DRUNet (after demosaic)
            if drunet_ok:
                sigma_est = estimate_srgb_sigma(noisy_srgb)
                den_srgb = method_drunet(noisy_srgb, sigma_est)
                p = psnr(gt_srgb, den_srgb)
                s = ssim_rgb(gt_srgb, den_srgb)
                lp = compute_lpips(gt_srgb, den_srgb)
                di = compute_dists(gt_srgb, den_srgb)
                nq = compute_niqe(den_srgb)
                dl_metrics["DRUNet"]["psnr"].append(p)
                dl_metrics["DRUNet"]["ssim"].append(s)
                dl_metrics["DRUNet"]["lpips"].append(lp)
                dl_metrics["DRUNet"]["dists"].append(di)
                dl_metrics["DRUNet"]["niqe"].append(nq)
                imsave(str(iso_dir / f"{img_path.stem}_DRUNet.png"),
                       (np.clip(den_srgb, 0, 1) * 255).astype(np.uint8))

            # Save GT + Noisy
            imsave(str(iso_dir / f"{img_path.stem}_gt.png"),
                   (np.clip(gt_srgb, 0, 1) * 255).astype(np.uint8))
            imsave(str(iso_dir / f"{img_path.stem}_Noisy.png"),
                   (np.clip(noisy_srgb, 0, 1) * 255).astype(np.uint8))

            if (idx + 1) % 6 == 0:
                print(f"  ISO {iso}: {idx+1}/{n_images} done", flush=True)

        # Compute averages and merge into v8 data
        elapsed_iso = time.time() - t_iso_start
        row = v8_data[iso_idx]

        print(f"\nISO {iso:5d} ({elapsed_iso:.0f}s):")
        for method_name, metrics in dl_metrics.items():
            avg = {k: round(float(np.mean(v)), 4 if k != "psnr" else 2)
                   for k, v in metrics.items()}
            prefix = method_name.replace("-", "")  # DnCNNB, DRUNet
            for k, v in avg.items():
                row[f"{method_name}_{k}"] = v
            print(f"  {method_name:<10} PSNR={avg['psnr']:.2f} SSIM={avg['ssim']:.4f} "
                  f"LPIPS={avg['lpips']:.4f} DISTS={avg['dists']:.4f} NIQE={avg['niqe']:.4f}")
        print()

        # Save intermediate results
        out_path = RESULTS / f"v8_f_10_10_{ds_name}_results.json"
        with open(out_path, 'w') as f:
            json.dump(v8_data, f, indent=2)

    # ===================== Summary =====================
    print("=" * 70)
    print(f"SUMMARY — DL after-demosaic [{ds_name}]")
    print(f"Pipeline: RAW -> demosaic -> sRGB -> [DL denoise per-ch] -> evaluate")
    print()
    print(f"  {'ISO':>6}  {'Method':<10} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'DISTS':>7} {'NIQE':>7}")
    print(f"  {'-' * 60}")
    for row in v8_data:
        iso = row.get("iso", "?")
        for method_name in ["DnCNN-B", "DRUNet", "GALOSH_F_10_10"]:
            p = row.get(f"{method_name}_psnr")
            if p is None:
                continue
            s = row.get(f"{method_name}_ssim", 0)
            lp = row.get(f"{method_name}_lpips", 0)
            di = row.get(f"{method_name}_dists", 0)
            nq = row.get(f"{method_name}_niqe", 0)
            label = method_name.replace("_F_10_10", "")
            print(f"  {iso:>6}  {label:<10} {p:>7.2f} {s:>7.4f} {lp:>7.4f} {di:>7.4f} {nq:>7.4f}")
        print()

    out_path = RESULTS / f"v8_f_10_10_{ds_name}_results.json"
    print(f"Results saved to {out_path}")
    print(f"Comparison images saved to {OUTDIR}")


if __name__ == "__main__":
    main()
