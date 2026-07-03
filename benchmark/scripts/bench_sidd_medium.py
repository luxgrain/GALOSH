#!/usr/bin/env python3
"""
SIDD Medium full-image GPU benchmark — Pre-demosaic vs After-demosaic.

Dataset: 20 scenes (5 cameras x 4 ISO/lighting), 2 images each = 40 full images.
         Pre-cropped to RGGB alignment, stored as .npy in datasets/sidd/medium_bench/.

Two evaluation groups:

  Pre-demosaic:
    RAW_noisy -> [DENOISE] -> Menon -> per-scene affine WB x CCM -> sRGB gamma
    Methods: GALOSH GPU (blind), NLM-CFA CUDA (oracle)
    GT2  = GT_RAW -> same calibrated pipeline  (pure denoise quality)
    GT1  = ISP sRGB                            (cross-group comparison)

  After-demosaic:
    ISP_sRGB_noisy -> [DENOISE] -> eval
    Methods: GALOSH-YUV GPU, NAFNet, DnCNN-B, DRUNet, NLM sRGB CUDA
    GT = ISP sRGB

Metrics: PSNR, SSIM, LPIPS, DISTS, NIQE (patch-based GPU for perceptual).
All denoising runs on GPU for fair speed comparison.

Usage:
  python bench_sidd_medium.py                     # all methods
  python bench_sidd_medium.py -m pre              # pre-demosaic only
  python bench_sidd_medium.py -m post             # after-demosaic only
  python bench_sidd_medium.py --no-perceptual     # skip LPIPS/DISTS/NIQE
  python bench_sidd_medium.py --scenes 2          # first N scenes (quick test)
"""
import numpy as np
import subprocess
import os
import sys
import json
import time
import argparse
from pathlib import Path
from skimage.io import imsave
from skimage.metrics import structural_similarity as ssim

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

BASE       = Path(__file__).parent.parent
BENCH_DIR  = Path(r"E:\img_dataset\sidd\medium_bench")
OUTDIR     = BASE / "sidd_medium"
RESULTS    = BASE / "results"
RESULTS.mkdir(exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

GPU_RAW_EXE    = Path(os.path.expanduser(r"~\GALOSH\standalone\galosh_raw_gpu.exe"))
GPU_YUV_EXE    = Path(os.path.expanduser(r"~\GALOSH\standalone\galosh_yuv_gpu.exe"))
GPU_SINGLE_EXE = Path(os.path.expanduser(r"~\GALOSH\standalone\galosh_single_gpu.exe"))
# Backward-compat alias (some legacy bench helpers reference GPU_EXE).
GPU_EXE        = GPU_RAW_EXE
KAIR_DIR   = Path(os.path.expanduser(r"~\GALOSH\benchmark\external\KAIR"))
NAFNET_DIR = Path(os.path.expanduser(r"~\GALOSH\benchmark\external\NAFNet"))
BASH_EXE   = Path(r"C:\msys64\usr\bin\bash.exe")

# EN: Make per-method wrappers under scripts/methods/ importable.
# JP: scripts/methods/ 配下のラッパー群を import 可能にする。
_METHODS_DIR = Path(__file__).parent / "methods"
if str(_METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(_METHODS_DIR))

# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------

def linear_to_srgb(x):
    return np.where(x <= 0.0031308,
                    12.92 * x,
                    1.055 * np.power(np.maximum(x, 0.0), 1.0/2.4) - 0.055)

def srgb_to_linear(x):
    return np.where(x <= 0.04045,
                    x / 12.92,
                    np.power((np.maximum(x, 0.0) + 0.055) / 1.055, 2.4))

def demosaic_menon(bayer):
    """Menon (2007) DDFAPD demosaic on RGGB."""
    from colour_demosaicing import demosaicing_CFA_Bayer_DDFAPD
    rgb = demosaicing_CFA_Bayer_DDFAPD(bayer.astype(np.float64), pattern='RGGB')
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)

def estimate_affine(gt_raw, gt_srgb):
    """Per-scene 3x4 affine from paired GT: Menon(gt_raw) linear -> inv_gamma(gt_srgb) linear."""
    rgb_menon = demosaic_menon(gt_raw)
    rgb_isp_lin = srgb_to_linear(gt_srgb.astype(np.float64))
    step = 4  # subsample for speed on full images
    A = rgb_menon[::step, ::step].reshape(-1, 3).astype(np.float64)
    B = rgb_isp_lin[::step, ::step].reshape(-1, 3).astype(np.float64)
    A_aug = np.hstack([A, np.ones((A.shape[0], 1))])
    M, _, _, _ = np.linalg.lstsq(A_aug, B, rcond=None)
    return M.T  # (3, 4)

def raw_to_srgb_calibrated(bayer, affine):
    """Bayer -> Menon -> affine -> sRGB gamma."""
    rgb_lin = demosaic_menon(bayer).astype(np.float64)
    h, w = rgb_lin.shape[:2]
    flat = rgb_lin.reshape(-1, 3)
    out = flat @ affine[:, :3].T + affine[:, 3]
    out = np.clip(out.reshape(h, w, 3), 0.0, 1.0)
    return np.clip(linear_to_srgb(out), 0.0, 1.0).astype(np.float32)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(ref, test):
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    return 10.0 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0

def ssim_rgb(ref, test):
    return float(ssim(ref, test, channel_axis=2, data_range=1.0))

_lpips_fn = None
def compute_lpips_patched(a, b, tile=1024):
    """Patch-based LPIPS to avoid OOM on full images."""
    global _lpips_fn
    import torch
    dev = torch.device('cuda')
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(dev)
    H, W = a.shape[:2]
    vals, weights = [], []
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1, x1 = min(y0+tile, H), min(x0+tile, W)
            pa = a[y0:y1, x0:x1]
            pb = b[y0:y1, x0:x1]
            ta = torch.from_numpy(pa.transpose(2,0,1)).unsqueeze(0).float().to(dev) * 2 - 1
            tb = torch.from_numpy(pb.transpose(2,0,1)).unsqueeze(0).float().to(dev) * 2 - 1
            with torch.no_grad():
                v = float(_lpips_fn(ta, tb).item())
            npix = (y1-y0) * (x1-x0)
            vals.append(v * npix)
            weights.append(npix)
    return sum(vals) / sum(weights)

_dists_fn = None
def compute_dists_patched(a, b, tile=1024):
    """Patch-based DISTS to avoid OOM."""
    global _dists_fn
    import torch
    dev = torch.device('cuda')
    if _dists_fn is None:
        import pyiqa
        _dists_fn = pyiqa.create_metric('dists', device='cuda')
    H, W = a.shape[:2]
    vals, weights = [], []
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1, x1 = min(y0+tile, H), min(x0+tile, W)
            pa = a[y0:y1, x0:x1]
            pb = b[y0:y1, x0:x1]
            ta = torch.from_numpy(pa.transpose(2,0,1)).unsqueeze(0).float().to(dev)
            tb = torch.from_numpy(pb.transpose(2,0,1)).unsqueeze(0).float().to(dev)
            with torch.no_grad():
                v = float(_dists_fn(ta, tb).item())
            npix = (y1-y0) * (x1-x0)
            vals.append(v * npix)
            weights.append(npix)
    return sum(vals) / sum(weights)

_niqe_fn = None
def compute_niqe(srgb):
    """NIQE no-reference quality via pyiqa (lower = better). Tiled for large images."""
    global _niqe_fn
    import torch
    if _niqe_fn is None:
        import pyiqa
        _niqe_fn = pyiqa.create_metric('niqe', device='cuda')
    H, W = srgb.shape[:2]
    tile = 1024
    vals, weights = [], []
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1, x1 = min(y0+tile, H), min(x0+tile, W)
            if (y1-y0) < 96 or (x1-x0) < 96:
                continue  # skip tiny edge tiles
            patch = srgb[y0:y1, x0:x1]
            t = torch.from_numpy(patch.transpose(2,0,1)).unsqueeze(0).float().cuda()
            with torch.no_grad():
                v = float(_niqe_fn(t).item())
            npix = (y1-y0) * (x1-x0)
            vals.append(v * npix)
            weights.append(npix)
    return sum(vals) / sum(weights) if weights else 0.0

# ---------------------------------------------------------------------------
# Denoisers — Pre-demosaic (RAW domain, GPU)
# ---------------------------------------------------------------------------

_galosh_cl_dev = None   # probed working (NVIDIA) OpenCL device; enumeration shuffles under CUDA load
def run_galosh_gpu(noisy, w, h, uid, strength=1.0, ls=1.0, cs=1.0,
                   alpha=0.0, sigma_sq=0.0, cl_device=None):
    """GALOSH GPU (OpenCL) on full RAW. Fully blind (or external alpha/sigma when > 0).
    cl_device=None -> PROBE devices 0..3 for the working (NVIDIA) one and cache it:
    this box has AMD/Intel iGPUs that cannot run galosh.cl, and under CUDA load the
    OpenCL enumeration order shuffles, so a fixed dev index is unreliable."""
    global _galosh_cl_dev
    uid = f"{uid}_{os.getpid()}"   # PID-unique temp: parallel processes must NOT share _tmp files
    in_path  = OUTDIR / f"_tmp_{uid}_in.raw"
    out_path = OUTDIR / f"_tmp_{uid}_out.raw"
    noisy.astype(np.float32).tofile(str(in_path))
    if cl_device is not None:
        cands = [cl_device]
    else:
        cands = [_galosh_cl_dev] if _galosh_cl_dev is not None else [0, 1, 2, 3]
    for attempt in range(2):
        for d in cands:
            cmd = [str(BASH_EXE), "-c",
                   f'"{GPU_RAW_EXE}" "{in_path}" "{out_path}" {w} {h} '
                   f'{strength} {ls} {cs} {alpha} {sigma_sq} {d}']
            t0 = time.time()
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=300)
            except subprocess.TimeoutExpired:
                out_path.unlink(missing_ok=True)
                print(f"    GALOSH GPU TIMEOUT (dev {d})")
                continue
            dt = time.time() - t0
            if r.returncode == 0 and out_path.exists():
                if cl_device is None: _galosh_cl_dev = d
                den = np.fromfile(str(out_path), dtype=np.float32).reshape(h, w)
                out_path.unlink(missing_ok=True); in_path.unlink(missing_ok=True)
                return den, dt
            out_path.unlink(missing_ok=True)
        if cl_device is not None: break
        cands = [0, 1, 2, 3]; _galosh_cl_dev = None   # cached device went stale -> full re-probe
    in_path.unlink(missing_ok=True)
    print(f"    GALOSH GPU FAILED on all devices")
    return None, 0.0

def run_nlm_cfa_cuda(noisy, sigma, patch_radius=2, search_radius=11):
    """NLM-CFA on RGGB Bayer via PyTorch CUDA."""
    from nlm_cuda import nlm_cfa_cuda
    t0 = time.time()
    den = nlm_cfa_cuda(noisy, sigma, patch_radius, search_radius)
    dt = time.time() - t0
    return den, dt

# ---------------------------------------------------------------------------
# Denoisers — After-demosaic (sRGB domain, GPU)
# ---------------------------------------------------------------------------

def run_galosh_yuv_gat_gpu(noisy_srgb, uid, strength_y=1.0, strength_c=1.0, cl_device=0):
    """GALOSH-YUV (new): sRGB -> OpenCL linear Y-GAT + chroma VST -> sRGB.

    EN: Calls galosh_gpu.exe with yuv_gat mode. All processing on GPU:
        sRGB -> linear YCbCr, Y blind noise estimation (Foi+Alenius),
        Y-GAT + LOSH + Makitalo inverse, Cb/Cr linear VST + LOSH + inverse,
        bivariate Wiener coupling, YCbCr -> sRGB.
    JP: yuv_gat モードで galosh_gpu.exe を呼び出す。全処理 GPU 上。
    """
    H, W = noisy_srgb.shape[:2]
    in_path  = OUTDIR / f"_tmp_{uid}_yuvgat_in.bin"
    out_path = OUTDIR / f"_tmp_{uid}_yuvgat_out.bin"
    noisy_srgb.astype(np.float32).tofile(str(in_path))
    cmd = [str(BASH_EXE), "-c",
           f'"{GPU_YUV_EXE}" "{in_path}" "{out_path}" {W} {H} '
           f'{strength_y} {strength_c} {cl_device}']
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
        print(f"    GALOSH-YUV (GAT) TIMEOUT")
        return None, time.time() - t0
    dt = time.time() - t0
    in_path.unlink(missing_ok=True)
    if r.returncode != 0:
        print(f"    GALOSH-YUV (GAT) FAILED: {r.stderr.decode()[:500]}")
        out_path.unlink(missing_ok=True)
        return None, dt
    den = np.fromfile(str(out_path), dtype=np.float32).reshape(H, W, 3)
    out_path.unlink(missing_ok=True)
    return np.clip(den, 0.0, 1.0).astype(np.float32), dt


def run_galosh_yuv_gpu_old(noisy_srgb, uid, strength_y=1.0, strength_c=1.0, cl_device=0):
    """GALOSH YUV GPU (old, nogat): sRGB -> BT.709 YCbCr -> per-plane GPU single -> RGB."""
    H, W = noisy_srgb.shape[:2]
    R, G, B = noisy_srgb[...,0], noisy_srgb[...,1], noisy_srgb[...,2]
    Y  =  0.2126*R + 0.7152*G + 0.0722*B
    Cb = -0.1146*R - 0.3854*G + 0.5000*B
    Cr =  0.5000*R - 0.4542*G - 0.0458*B

    t0 = time.time()
    planes_den = []
    for name, plane, strength in [("Y", Y, strength_y), ("Cb", Cb, strength_c), ("Cr", Cr, strength_c)]:
        in_path  = OUTDIR / f"_tmp_{uid}_{name}_in.raw"
        out_path = OUTDIR / f"_tmp_{uid}_{name}_out.raw"
        plane.astype(np.float32).tofile(str(in_path))
        cmd = [str(BASH_EXE), "-c",
               f'"{GPU_SINGLE_EXE}" "{in_path}" "{out_path}" {W} {H} '
               f'{strength} {cl_device}']
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=120)
        except subprocess.TimeoutExpired:
            in_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
            print(f"    GALOSH YUV {name} TIMEOUT")
            return None, time.time() - t0
        in_path.unlink(missing_ok=True)
        if r.returncode != 0:
            print(f"    GALOSH YUV {name} FAILED: {r.stderr.decode()[:300]}")
            out_path.unlink(missing_ok=True)
            return None, time.time() - t0
        d = np.fromfile(str(out_path), dtype=np.float32).reshape(H, W)
        out_path.unlink(missing_ok=True)
        planes_den.append(d)
    dt = time.time() - t0

    Y_d, Cb_d, Cr_d = planes_den
    R_d = np.clip(Y_d + 1.5748*Cr_d,                  0.0, 1.0)
    G_d = np.clip(Y_d - 0.1873*Cb_d - 0.4681*Cr_d, 0.0, 1.0)
    B_d = np.clip(Y_d + 1.8556*Cb_d,                  0.0, 1.0)
    return np.stack([R_d, G_d, B_d], axis=2).astype(np.float32), dt

_nafnet_model = None
def get_nafnet():
    global _nafnet_model
    if _nafnet_model is not None:
        return _nafnet_model
    nafnet_str = str(NAFNET_DIR)
    if nafnet_str not in sys.path:
        sys.path.insert(0, nafnet_str)
    from basicsr.models.archs.NAFNet_arch import NAFNet
    import torch
    dev = torch.device('cuda')
    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    ckpt = NAFNET_DIR / "experiments" / "pretrained_models" / "NAFNet-SIDD-width64.pth"
    state = torch.load(str(ckpt), map_location=dev, weights_only=False)
    model.load_state_dict(state["params"], strict=True)
    model.eval().to(dev)
    _nafnet_model = model
    return model

def run_nafnet(noisy_srgb, tile=512):
    """NAFNet on sRGB. Always tile-based for full images (OOM on >1024x1024).
    Overlap of 32px between tiles to reduce border artifacts."""
    import torch
    model = get_nafnet()
    dev = next(model.parameters()).device
    H, W = noisy_srgb.shape[:2]
    overlap = 32

    t0 = time.time()
    result = np.zeros_like(noisy_srgb)
    weight = np.zeros((H, W, 1), dtype=np.float32)

    step = tile - overlap
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            # Pad to multiple of 64
            ph = (64 - (y1-y0) % 64) % 64
            pw = (64 - (x1-x0) % 64) % 64
            patch = np.pad(noisy_srgb[y0:y1, x0:x1], ((0,ph),(0,pw),(0,0)), mode='reflect')
            pi = torch.from_numpy(patch.transpose(2,0,1)).unsqueeze(0).float().to(dev)
            with torch.no_grad():
                po = model(pi)
            out = po.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy()[:y1-y0,:x1-x0]
            result[y0:y1, x0:x1] += out
            weight[y0:y1, x0:x1] += 1.0
            del pi, po
    torch.cuda.empty_cache()

    result = (result / np.maximum(weight, 1.0)).astype(np.float32)
    dt = time.time() - t0
    return result, dt

_dncnn_model = None
def get_dncnn():
    global _dncnn_model
    if _dncnn_model is not None:
        return _dncnn_model
    sys.path.insert(0, str(KAIR_DIR))
    from models.network_dncnn import DnCNN
    import torch
    dev = torch.device('cuda')
    model = DnCNN(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='R')
    state = torch.load(str(KAIR_DIR / "model_zoo" / "dncnn_gray_blind.pth"),
                       map_location=dev, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(dev)
    _dncnn_model = model
    return model

def run_dncnn_b(noisy_srgb):
    """DnCNN-B per-channel on sRGB (GPU)."""
    import torch
    model = get_dncnn()
    dev = next(model.parameters()).device
    t0 = time.time()
    out_ch = []
    for c in range(3):
        inp = torch.from_numpy(noisy_srgb[:,:,c]).unsqueeze(0).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            res = model(inp)
        out_ch.append(res.squeeze().clamp(0,1).cpu().numpy())
        del inp, res
    dt = time.time() - t0
    torch.cuda.empty_cache()
    return np.stack(out_ch, axis=2).astype(np.float32), dt

_drunet_model = None
def get_drunet():
    global _drunet_model
    if _drunet_model is not None:
        return _drunet_model
    sys.path.insert(0, str(KAIR_DIR))
    from models.network_unet import UNetRes
    import torch
    dev = torch.device('cuda')
    model = UNetRes(in_nc=2, out_nc=1, nc=[64,128,256,512],
                    nb=4, act_mode='R', downsample_mode='strideconv',
                    upsample_mode='convtranspose', bias=False)
    state = torch.load(str(KAIR_DIR / "model_zoo" / "drunet_gray.pth"),
                       map_location=dev, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(dev)
    _drunet_model = model
    return model

def sigma_mad_srgb(noisy_srgb):
    """Blind sigma from sRGB luminance via wavelet MAD."""
    import pywt
    gray = np.mean(noisy_srgb, axis=2)
    _, (_, _, hh) = pywt.dwt2(gray.astype(np.float64), 'haar')
    return float(np.median(np.abs(hh)) / 0.6745)

def run_drunet(noisy_srgb):
    """DRUNet per-channel with MAD sigma (GPU)."""
    import torch
    model = get_drunet()
    dev = next(model.parameters()).device
    sigma = sigma_mad_srgb(noisy_srgb)
    H, W = noisy_srgb.shape[:2]
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8

    t0 = time.time()
    out_ch = []
    for c in range(3):
        ch = noisy_srgb[:,:,c]
        if pad_h or pad_w:
            ch = np.pad(ch, ((0,pad_h),(0,pad_w)), mode='reflect')
        sigma_map = np.full_like(ch, sigma)
        inp = torch.from_numpy(np.stack([ch, sigma_map])).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            res = model(inp)
        out = res.squeeze().clamp(0,1).cpu().numpy()
        if pad_h or pad_w:
            out = out[:H, :W]
        out_ch.append(out)
        del inp, res
    dt = time.time() - t0
    torch.cuda.empty_cache()
    return np.stack(out_ch, axis=2).astype(np.float32), dt

def run_nlm_srgb_cuda(noisy_srgb, sigma, patch_radius=2, search_radius=11):
    """NLM per-channel on sRGB via PyTorch CUDA."""
    from nlm_cuda import nlm_srgb_cuda
    t0 = time.time()
    den = nlm_srgb_cuda(noisy_srgb, sigma, patch_radius, search_radius)
    dt = time.time() - t0
    return den, dt

# ---------------------------------------------------------------------------
# CPU denoisers — BM3D sRGB, NLM sRGB (skimage)
# ---------------------------------------------------------------------------

import threading
_bm3d_lock = threading.Lock()
_nlm_lock  = threading.Lock()

def run_bm3d_srgb_cpu(noisy_srgb, sigma):
    """BM3D per-channel on sRGB (CPU). Sigma from MAD."""
    import bm3d as bm3d_pkg
    t0 = time.time()
    out_ch = []
    for c in range(3):
        with _bm3d_lock:
            den = bm3d_pkg.bm3d(noisy_srgb[:,:,c].astype(np.float64), sigma_psd=sigma)
        out_ch.append(np.clip(den, 0.0, 1.0).astype(np.float32))
    dt = time.time() - t0
    return np.stack(out_ch, axis=2), dt

def run_nlm_srgb_cpu(noisy_srgb, sigma):
    """NLM per-channel on sRGB (CPU, skimage)."""
    from skimage.restoration import denoise_nl_means
    t0 = time.time()
    out_ch = []
    for c in range(3):
        ch = noisy_srgb[:,:,c].astype(np.float64)
        with _nlm_lock:
            den = denoise_nl_means(ch, h=1.15*sigma, sigma=sigma,
                                   fast_mode=True, patch_size=5, patch_distance=6)
        out_ch.append(np.clip(den, 0.0, 1.0).astype(np.float32))
    dt = time.time() - t0
    return np.stack(out_ch, axis=2), dt

# ---------------------------------------------------------------------------
# Extra denoisers — rule-based + sensor-agnostic DL (v2 expansion)
# ---------------------------------------------------------------------------
# EN: Added for the "blind non-learning strongest" and "sensor-agnostic DL
#     strongest" narrative. Each wrapper lives under scripts/methods/ and is
#     imported lazily so that failures in one method don't block the others.
# JP: "blind 非学習最強" "sensor 非特化 DL 最強" の主張を確定させるため追加。
#     各 wrapper は scripts/methods/ 配下にあり、遅延 import で単独故障を局所化。

def run_bm3d_cfa_wrapper(noisy_raw):
    """BM3D-CFA (Bayer 4-plane, CPU). Blind sigma via MAD per plane inside wrapper."""
    from bm3d_cfa import run_bm3d_cfa
    return run_bm3d_cfa(noisy_raw, sigma=None)

def run_cbm3d_wrapper(noisy_srgb):
    """CBM3D joint 3-channel on sRGB (CPU). Blind sigma via MAD inside wrapper."""
    from cbm3d import run_cbm3d
    return run_cbm3d(noisy_srgb, sigma=None)

def run_guided_gpu_wrapper(noisy_srgb):
    """Guided Filter (self-guide) on sRGB via kornia (GPU). eps auto from MAD sigma."""
    from guided_filter import run_guided_filter
    return run_guided_filter(noisy_srgb, sigma=None, device="cuda")

def run_ffdnet_wrapper(noisy_srgb):
    """FFDNet color (AWGN-trained, KAIR). sigma_map from MAD on sRGB."""
    from ffdnet import run_ffdnet
    return run_ffdnet(noisy_srgb, sigma=None, device="cuda")

def run_swinir_wrapper(noisy_srgb):
    """SwinIR color Gaussian (AWGN σ ∈ {15,25,50}/255). Nearest-σ auto selection.

    EN: Returns a 3-tuple (denoised, elapsed, picked_sigma); we drop the
        picked_sigma here to keep the common (den, dt) contract.
    JP: SwinIR は (den, dt, picked σ) の 3 要素を返す。ここでは picked を捨てる。
    """
    from swinir import run_swinir_gaussian
    den, dt, _picked = run_swinir_gaussian(noisy_srgb, sigma=None, device="cuda")
    return den, dt

def run_scunet_wrapper(noisy_srgb):
    """SCUNet color practical-blind (Swin-Conv UNet). Tile-default 512+32 overlap."""
    from scunet import run_scunet
    return run_scunet(noisy_srgb, device="cuda")

def run_restormer_wrapper(noisy_srgb):
    """Restormer Gaussian color blind. BiasFree LayerNorm, AMP fp16 tile=512+32."""
    from restormer import run_restormer_gaussian
    return run_restormer_gaussian(noisy_srgb, sigma=None, device="cuda")


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    # Pre-demosaic
    "galosh_raw_gpu":       "GALOSH-RAW",
    "nlm_cfa_oracle":       "NLM-CFA (oracle)",
    "bm3d_cfa":             "BM3D-CFA (blind)",
    "bm3d_cfa_oracle":      "BM3D-CFA (oracle)",
    "b2u":                  "Blind2Unblind (raw)",
    # After-demosaic (GPU) — GALOSH family
    "galosh_yuv_gpu":       "GALOSH-YUV",          # NEW: linear Y-GAT + chroma VST
    "galosh_yuv_gpu_old":   "GALOSH-YUV (old)",     # OLD: sRGB YCbCr plain Wiener (nogat)
    # After-demosaic (GPU) — DL & classical
    "nafnet":               "NAFNet-SIDD",
    "dncnn_b":              "DnCNN-B",
    "drunet":               "DRUNet",
    "nlm_srgb":             "NLM sRGB CUDA",
    "guided_gpu":           "Guided (kornia GPU)",
    "ffdnet":               "FFDNet",
    "swinir_dn":            "SwinIR-Gaussian",
    "scunet":               "SCUNet",
    "restormer_dn":         "Restormer-Gaussian",
    # After-demosaic (CPU reference)
    "bm3d_srgb":            "BM3D sRGB (CPU)",
    "nlm_srgb_cpu":         "NLM sRGB (CPU)",
    "cbm3d":                "CBM3D joint (CPU)",
}

PRE_IDS  = ["galosh_raw_gpu", "nlm_cfa_oracle", "bm3d_cfa", "bm3d_cfa_oracle", "b2u"]
POST_IDS = ["galosh_yuv_gpu", "galosh_yuv_gpu_old",
            "nafnet", "dncnn_b", "drunet", "nlm_srgb",
            "guided_gpu", "ffdnet", "swinir_dn", "scunet", "restormer_dn",
            "bm3d_srgb", "nlm_srgb_cpu", "cbm3d"]

# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SIDD Medium full-image GPU benchmark")
    parser.add_argument("--methods", "-m", nargs="+",
                        default=["all"],
                        choices=list(DISPLAY_NAMES.keys()) + ["all", "pre", "post"],
                        help="Methods to benchmark")
    parser.add_argument("--no-perceptual", action="store_true",
                        help="Skip LPIPS/DISTS/NIQE")
    parser.add_argument("--scenes", "-s", type=int, default=0,
                        help="Limit to first N scenes (0 = all)")
    args = parser.parse_args()

    if "all" in args.methods:
        methods = PRE_IDS + POST_IDS
    elif "pre" in args.methods:
        methods = PRE_IDS
    elif "post" in args.methods:
        methods = POST_IDS
    else:
        methods = args.methods

    use_perceptual = not args.no_perceptual

    # --- Load existing results for skip logic ---
    existing_results = {}  # method_id -> {tag: result_dict}
    for mid in methods:
        json_path = RESULTS / f"sidd_medium_{mid}.json"
        if json_path.exists():
            with open(str(json_path)) as f:
                prev = json.load(f)
            existing_results[mid] = {r["tag"]: r for r in prev}
            print(f"  Loaded {len(prev)} existing results for {mid}")
        else:
            existing_results[mid] = {}

    # --- Load scene metadata ---
    with open(str(BENCH_DIR / "scenes.json")) as f:
        all_scenes = json.load(f)

    # Group by scene (2 images per scene)
    scene_groups = {}
    for s in all_scenes:
        key = f"{s['scene_id']}_{s['cam']}_{s['pattern']}"
        scene_groups.setdefault(key, []).append(s)

    scene_keys = list(scene_groups.keys())
    if args.scenes:
        scene_keys = scene_keys[:args.scenes]

    n_scenes = len(scene_keys)
    n_images = sum(len(scene_groups[k]) for k in scene_keys)
    print(f"=== SIDD Medium Full-Image GPU Benchmark ===")
    print(f"Scenes: {n_scenes}, Images: {n_images}")
    print(f"Methods: {', '.join(DISPLAY_NAMES[m] for m in methods)}")
    print(f"Perceptual: {'Yes' if use_perceptual else 'No'}\n")

    # --- Preload DL models ---
    for mid, loader, label in [
        ("nafnet", get_nafnet, "NAFNet"),
        ("dncnn_b", get_dncnn, "DnCNN-B"),
        ("drunet", get_drunet, "DRUNet"),
    ]:
        if mid in methods:
            print(f"Loading {label}...", end=" ", flush=True)
            try:
                loader()
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")
                methods = [m for m in methods if m != mid]

    if use_perceptual:
        print("Pre-loading LPIPS/DISTS...", end=" ", flush=True)
        dummy = np.zeros((64, 64, 3), dtype=np.float32)
        compute_lpips_patched(dummy, dummy)
        compute_dists_patched(dummy, dummy)
        print("OK")

    # --- Phase 1: Affine estimation + GT2 precomputation ---
    needs_pre = any(m in PRE_IDS for m in methods)
    affines = {}
    gt2_cache = {}

    if needs_pre:
        print("\n--- Phase 1: Per-scene affine WB x CCM estimation ---")
        for sk in scene_keys:
            imgs = scene_groups[sk]
            s0 = imgs[0]
            tag0 = s0["tag"]
            gt_raw = np.load(str(BENCH_DIR / f"{tag0}_gt_raw.npy"))
            gt_srgb = np.load(str(BENCH_DIR / f"{tag0}_gt_srgb.npy"))
            print(f"  {sk}: estimating affine from {tag0}...", end=" ", flush=True)
            aff = estimate_affine(gt_raw, gt_srgb)
            affines[sk] = aff
            print(f"diag=[{aff[0,0]:.3f}, {aff[1,1]:.3f}, {aff[2,2]:.3f}]")

        print("\n--- Phase 1b: GT2 precomputation ---")
        # GT2 is deterministic (gt_raw + per-scene affine, both fixed).
        # Reuse the PNG cache from __gt2__/ if present to skip the ~7-8 s per
        # image Menon demosaic + affine (saves ~10 min across 80 images).
        from skimage.io import imread as _imread
        gt2_cache_dir = OUTDIR / "__gt2__"
        for sk in scene_keys:
            aff = affines[sk]
            for s in scene_groups[sk]:
                tag = s["tag"]
                cached_path = gt2_cache_dir / f"{tag}_gt2.png"
                t0 = time.time()
                if cached_path.exists():
                    gt2 = _imread(str(cached_path)).astype(np.float32) / 255.0
                    gt2_cache[tag] = gt2
                    print(f"  GT2: {tag} (cached, {time.time()-t0:.2f}s)", flush=True)
                else:
                    print(f"  GT2: {tag}...", end=" ", flush=True)
                    gt_raw = np.load(str(BENCH_DIR / f"{tag}_gt_raw.npy"))
                    gt2 = raw_to_srgb_calibrated(gt_raw, aff)
                    gt2_cache[tag] = gt2
                    del gt_raw
                    print(f"({time.time()-t0:.1f}s)")

    # --- Phase 2: Run denoisers + metrics ---
    print(f"\n{'='*70}")
    print(f"--- Phase 2: Denoise + Metrics ---")
    print(f"{'='*70}\n")

    all_results = {}  # method_id -> list of per-image results

    for mid in methods:
        name = DISPLAY_NAMES[mid]
        is_pre = mid in PRE_IDS
        gt_label = "GT2 (calibrated)" if is_pre else "GT1 (ISP sRGB)"
        print(f"\n[{name}] (GT: {gt_label})")
        print("-" * 50)

        method_results = []

        for sk in scene_keys:
            for s in scene_groups[sk]:
                tag = s["tag"]
                H, W = s["height"], s["width"]
                uid = f"{mid}_{tag}"

                # --- Skip if already computed ---
                if tag in existing_results.get(mid, {}):
                    prev = existing_results[mid][tag]
                    method_results.append(prev)
                    print(f"  {tag} ... SKIP (cached PSNR={prev['psnr']:.2f})")
                    continue

                print(f"  {tag} ({W}x{H}, ISO {s['iso']})...", end=" ", flush=True)

                # Load data
                gt_raw    = np.load(str(BENCH_DIR / f"{tag}_gt_raw.npy"))
                noisy_raw = np.load(str(BENCH_DIR / f"{tag}_noisy_raw.npy"))
                gt_srgb   = np.load(str(BENCH_DIR / f"{tag}_gt_srgb.npy"))
                noisy_srgb = np.load(str(BENCH_DIR / f"{tag}_noisy_srgb.npy"))

                # --- Run denoiser ---
                den_srgb = None
                den_raw = None
                dt = 0

                if mid == "galosh_raw_gpu":
                    den_raw, dt = run_galosh_gpu(noisy_raw, W, H, uid)
                    if den_raw is not None:
                        aff = affines[sk]
                        den_srgb = raw_to_srgb_calibrated(den_raw, aff)

                elif mid == "nlm_cfa_oracle":
                    sigma = float(np.std(noisy_raw.astype(np.float64) - gt_raw.astype(np.float64)))
                    den_raw, dt = run_nlm_cfa_cuda(noisy_raw, sigma)
                    aff = affines[sk]
                    den_srgb = raw_to_srgb_calibrated(den_raw, aff)

                elif mid == "bm3d_cfa":
                    # Pre-demosaic: denoise Bayer 4-plane -> calibrated sRGB.
                    try:
                        den_raw, dt = run_bm3d_cfa_wrapper(noisy_raw)
                    except Exception as e:
                        print(f"    BM3D-CFA FAILED: {type(e).__name__}: {str(e)[:120]}")
                        den_raw, dt = None, 0.0
                    if den_raw is not None:
                        aff = affines[sk]
                        den_srgb = raw_to_srgb_calibrated(den_raw, aff)

                elif mid == "bm3d_cfa_oracle":
                    # Same as bm3d_cfa but sigma from (noisy - gt) std (oracle).
                    try:
                        from bm3d_cfa import run_bm3d_cfa
                        sigma = float(np.std(noisy_raw.astype(np.float64) - gt_raw.astype(np.float64)))
                        den_raw, dt = run_bm3d_cfa(noisy_raw, sigma=sigma)
                    except Exception as e:
                        print(f"    BM3D-CFA-oracle FAILED: {type(e).__name__}: {str(e)[:120]}")
                        den_raw, dt = None, 0.0
                    if den_raw is not None:
                        aff = affines[sk]
                        den_srgb = raw_to_srgb_calibrated(den_raw, aff)

                elif mid == "b2u":
                    # Pre-demosaic: pretrained Blind2Unblind (rawRGB_112rf20),
                    # tile-inferred over full-res Bayer, then same calibrated
                    # demosaic + ISP pipeline as other pre-demosaic methods.
                    try:
                        from b2u import run_b2u
                        den_raw, dt = run_b2u(noisy_raw)
                    except Exception as e:
                        print(f"    B2U FAILED: {type(e).__name__}: {str(e)[:120]}")
                        den_raw, dt = None, 0.0
                    if den_raw is not None:
                        aff = affines[sk]
                        den_srgb = raw_to_srgb_calibrated(den_raw, aff)

                elif mid == "galosh_yuv_gpu":
                    den_srgb, dt = run_galosh_yuv_gat_gpu(noisy_srgb, uid)

                elif mid == "galosh_yuv_gpu_old":
                    den_srgb, dt = run_galosh_yuv_gpu_old(noisy_srgb, uid)

                elif mid == "nafnet":
                    den_srgb, dt = run_nafnet(noisy_srgb)

                elif mid == "dncnn_b":
                    den_srgb, dt = run_dncnn_b(noisy_srgb)

                elif mid == "drunet":
                    den_srgb, dt = run_drunet(noisy_srgb)

                elif mid == "nlm_srgb":
                    sigma_s = sigma_mad_srgb(noisy_srgb)
                    den_srgb, dt = run_nlm_srgb_cuda(noisy_srgb, sigma_s)

                elif mid == "bm3d_srgb":
                    sigma_s = sigma_mad_srgb(noisy_srgb)
                    den_srgb, dt = run_bm3d_srgb_cpu(noisy_srgb, sigma_s)

                elif mid == "nlm_srgb_cpu":
                    sigma_s = sigma_mad_srgb(noisy_srgb)
                    den_srgb, dt = run_nlm_srgb_cpu(noisy_srgb, sigma_s)

                elif mid == "guided_gpu":
                    try:
                        den_srgb, dt = run_guided_gpu_wrapper(noisy_srgb)
                    except Exception as e:
                        print(f"    Guided GPU FAILED: {type(e).__name__}: {str(e)[:120]}")
                        den_srgb, dt = None, 0.0

                elif mid == "ffdnet":
                    try:
                        den_srgb, dt = run_ffdnet_wrapper(noisy_srgb)
                    except Exception as e:
                        print(f"    FFDNet FAILED: {type(e).__name__}: {str(e)[:120]}")
                        den_srgb, dt = None, 0.0

                elif mid == "swinir_dn":
                    try:
                        den_srgb, dt = run_swinir_wrapper(noisy_srgb)
                    except Exception as e:
                        print(f"    SwinIR FAILED: {type(e).__name__}: {str(e)[:120]}")
                        den_srgb, dt = None, 0.0

                elif mid == "scunet":
                    try:
                        den_srgb, dt = run_scunet_wrapper(noisy_srgb)
                    except Exception as e:
                        print(f"    SCUNet FAILED: {type(e).__name__}: {str(e)[:120]}")
                        den_srgb, dt = None, 0.0

                elif mid == "restormer_dn":
                    try:
                        den_srgb, dt = run_restormer_wrapper(noisy_srgb)
                    except Exception as e:
                        print(f"    Restormer FAILED: {type(e).__name__}: {str(e)[:120]}")
                        den_srgb, dt = None, 0.0

                elif mid == "cbm3d":
                    try:
                        den_srgb, dt = run_cbm3d_wrapper(noisy_srgb)
                    except Exception as e:
                        print(f"    CBM3D FAILED: {type(e).__name__}: {str(e)[:120]}")
                        den_srgb, dt = None, 0.0

                if den_srgb is None:
                    print("FAILED")
                    continue

                # --- Select GT ---
                if is_pre:
                    gt_primary = gt2_cache[tag]  # GT2
                else:
                    gt_primary = gt_srgb         # GT1 = ISP sRGB

                # --- Compute metrics ---
                res = {
                    "tag": tag, "scene": sk,
                    "cam": s["cam"], "iso": s["iso"],
                    "ct": s["ct"], "light": s["light"],
                    "width": W, "height": H,
                    "time_s": round(dt, 3),
                    "psnr": round(psnr(gt_primary, den_srgb), 2),
                    "ssim": round(ssim_rgb(gt_primary, den_srgb), 4),
                }

                # Cross-comparison for pre-demosaic: vs GT1
                if is_pre:
                    res["isp_psnr"] = round(psnr(gt_srgb, den_srgb), 2)
                    res["isp_ssim"] = round(ssim_rgb(gt_srgb, den_srgb), 4)

                if use_perceptual:
                    res["lpips"] = round(compute_lpips_patched(gt_primary, den_srgb), 4)
                    res["dists"] = round(compute_dists_patched(gt_primary, den_srgb), 4)
                    res["niqe"]  = round(compute_niqe(den_srgb), 3)
                    if is_pre:
                        res["isp_lpips"] = round(compute_lpips_patched(gt_srgb, den_srgb), 4)

                parts = [f"PSNR={res['psnr']:.2f}"]
                if "isp_psnr" in res:
                    parts.append(f"ISP={res['isp_psnr']:.2f}")
                parts.append(f"SSIM={res['ssim']:.4f}")
                if "lpips" in res:
                    parts.append(f"LPIPS={res['lpips']:.4f}")
                if "niqe" in res:
                    parts.append(f"NIQE={res['niqe']:.3f}")
                parts.append(f"{dt:.2f}s")
                print(f"{' | '.join(parts)}")

                method_results.append(res)

                # --- Save PNG ---
                mid_dir = OUTDIR / mid
                mid_dir.mkdir(exist_ok=True)
                imsave(str(mid_dir / f"{tag}_den.png"),
                       (np.clip(den_srgb, 0, 1) * 255).astype(np.uint8))

                # Save GT/Noisy once
                gt_dir = OUTDIR / "__gt__"
                gt_dir.mkdir(exist_ok=True)
                noisy_dir = OUTDIR / "__noisy__"
                noisy_dir.mkdir(exist_ok=True)
                gt_path = gt_dir / f"{tag}_gt_srgb.png"
                if not gt_path.exists():
                    imsave(str(gt_path),
                           (np.clip(gt_srgb, 0, 1) * 255).astype(np.uint8))
                    imsave(str(noisy_dir / f"{tag}_noisy_srgb.png"),
                           (np.clip(noisy_srgb, 0, 1) * 255).astype(np.uint8))
                    if is_pre and tag in gt2_cache:
                        gt2_dir = OUTDIR / "__gt2__"
                        gt2_dir.mkdir(exist_ok=True)
                        imsave(str(gt2_dir / f"{tag}_gt2.png"),
                               (np.clip(gt2_cache[tag], 0, 1) * 255).astype(np.uint8))

                # Free memory
                del gt_raw, noisy_raw, gt_srgb, noisy_srgb, den_srgb
                if den_raw is not None:
                    del den_raw

        all_results[mid] = method_results

        # Per-method summary
        if method_results:
            avg_psnr = np.mean([r["psnr"] for r in method_results])
            avg_ssim = np.mean([r["ssim"] for r in method_results])
            avg_time = np.mean([r["time_s"] for r in method_results])
            summary = f"  => Avg PSNR={avg_psnr:.2f} | SSIM={avg_ssim:.4f} | {avg_time:.2f}s/img"
            if "lpips" in method_results[0]:
                avg_lpips = np.mean([r["lpips"] for r in method_results])
                avg_dists = np.mean([r["dists"] for r in method_results])
                avg_niqe  = np.mean([r["niqe"] for r in method_results])
                summary += f" | LPIPS={avg_lpips:.4f} | DISTS={avg_dists:.4f} | NIQE={avg_niqe:.3f}"
            print(summary)

        # Save per-method JSON
        with open(RESULTS / f"sidd_medium_{mid}.json", "w") as f:
            json.dump(method_results, f, indent=2)

    # --- Phase 3: Summary tables ---
    print(f"\n\n{'='*90}")
    print(f"  SIDD Medium Full-Image Benchmark Results ({n_images} images)")
    print(f"{'='*90}")

    for group_name, group_ids, show_isp in [
        ("Pre-demosaic (GT2 = calibrated pipeline)", PRE_IDS, True),
        ("After-demosaic (GT = ISP sRGB)", POST_IDS, False),
    ]:
        group_methods = [m for m in group_ids if m in all_results and all_results[m]]
        if not group_methods:
            continue

        print(f"\n  {group_name}")
        hdr = f"  {'Method':<20} {'PSNR':>7} {'SSIM':>7}"
        if use_perceptual:
            hdr += f" {'LPIPS':>7} {'DISTS':>7} {'NIQE':>7}"
        if show_isp:
            hdr += f" {'ISP_PSNR':>9}"
        hdr += f" {'Time':>8}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for mid in group_methods:
            mr = all_results[mid]
            name = DISPLAY_NAMES[mid]
            avg = lambda k: np.mean([r[k] for r in mr if k in r])
            line = f"  {name:<20} {avg('psnr'):>7.2f} {avg('ssim'):>7.4f}"
            if use_perceptual:
                line += f" {avg('lpips'):>7.4f} {avg('dists'):>7.4f} {avg('niqe'):>7.3f}"
            if show_isp and "isp_psnr" in mr[0]:
                line += f" {avg('isp_psnr'):>9.2f}"
            line += f" {avg('time_s'):>7.2f}s"
            print(line)

    # --- Per-camera breakdown ---
    cameras = sorted(set(s["cam"] for sk in scene_keys for s in scene_groups[sk]))
    for cam in cameras:
        print(f"\n  --- {cam} ---")
        hdr = f"  {'Method':<20} {'PSNR':>7} {'SSIM':>7}"
        if use_perceptual:
            hdr += f" {'LPIPS':>7} {'NIQE':>7}"
        hdr += f" {'Time':>8}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for mid in methods:
            if mid not in all_results:
                continue
            mr = [r for r in all_results[mid] if r["cam"] == cam]
            if not mr:
                continue
            name = DISPLAY_NAMES[mid]
            avg = lambda k: np.mean([r[k] for r in mr if k in r])
            line = f"  {name:<20} {avg('psnr'):>7.2f} {avg('ssim'):>7.4f}"
            if use_perceptual:
                line += f" {avg('lpips'):>7.4f} {avg('niqe'):>7.3f}"
            line += f" {avg('time_s'):>7.2f}s"
            print(line)

    # --- Also compute GT2/Noisy baselines ---
    print(f"\n  --- Baselines ---")
    if needs_pre:
        noisy_psnrs = []
        gt2_psnrs = []
        gt2_niqes = []
        for sk in scene_keys:
            aff = affines[sk]
            for s in scene_groups[sk]:
                tag = s["tag"]
                gt_raw = np.load(str(BENCH_DIR / f"{tag}_gt_raw.npy"))
                noisy_raw = np.load(str(BENCH_DIR / f"{tag}_noisy_raw.npy"))
                gt_srgb = np.load(str(BENCH_DIR / f"{tag}_gt_srgb.npy"))
                noisy_psnrs.append(psnr(gt_raw, noisy_raw))
                gt2 = gt2_cache.get(tag)
                if gt2 is not None:
                    gt2_psnrs.append(psnr(gt_srgb, gt2))
                    if use_perceptual:
                        gt2_niqes.append(compute_niqe(gt2))
                del gt_raw, noisy_raw, gt_srgb
        print(f"  Noisy RAW PSNR:     {np.mean(noisy_psnrs):.2f} dB")
        print(f"  GT2 vs GT1 PSNR:    {np.mean(gt2_psnrs):.2f} dB")
        if gt2_niqes:
            print(f"  GT2 NIQE:           {np.mean(gt2_niqes):.3f}")

    # --- Save combined results ---
    combined = {
        "dataset": "sidd_medium",
        "n_scenes": n_scenes, "n_images": n_images,
        "methods": {mid: all_results[mid] for mid in methods if mid in all_results},
    }
    combined_path = RESULTS / "sidd_medium_combined.json"
    with open(str(combined_path), "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n  Combined JSON -> {combined_path}")
    print(f"  PNG outputs   -> {OUTDIR}/")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
