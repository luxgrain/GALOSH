"""
bench_compare.py  ---  YUV GALOSH vs BM3D vs NLM on Kodak
         + LPIPS / DISTS perceptual metrics

Methods compared (all on YUV420, Y plane as primary metric):
  1. GALOSH-YUV   : our method, fully blind (sigma auto-estimated per plane)
  2. BM3D-oracle  : bm3d on Y, sigma known (upper bound for sigma-dependent methods)
  3. BM3D-blind   : bm3d on Y, sigma estimated via Laplacian MAD (same as GALOSH)
  4. NLM-oracle   : cv2 fastNlMeansDenoising on Y, h tuned from true sigma
  5. Wiener-freq  : scipy Wiener filter baseline

Chroma (U/V) for all non-GALOSH methods: Gaussian shrinkage (sigma_c estimated).

Usage:
  python bench_compare.py [noise_sigma] [galosh_sy] [galosh_sc]
"""

import os
import sys, os, subprocess, time, math
import numpy as np
import bm3d
import cv2
import torch
import lpips
from DISTS_pytorch import DISTS
from scipy.ndimage import gaussian_filter
from PIL import Image
from skimage.metrics import structural_similarity

# -- Config -------------------------------------------------------------------
KODAK_DIR  = os.path.expanduser(r"~\denoise_eval\datasets\kodak")
EXE        = os.path.expanduser(r"~\denoise_eval\standalone\yuv_galosh.exe")
BASH       = r"C:\msys64\usr\bin\bash.exe"
WORK_DIR   = os.path.expanduser(r"~\denoise_eval\standalone\yuv_bench")

noise_sigma  = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
galosh_sy    = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
galosh_sc    = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
galosh_gamma = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0  # 1.0=linear, 2.2=sRGB
STRIDE_Y     = 2
STRIDE_C     = 2

os.makedirs(WORK_DIR, exist_ok=True)

# -- Perceptual metric models (load once) -------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
lpips_fn.eval()
dists_fn = DISTS().to(DEVICE)
dists_fn.eval()

def to_tensor(rgb_f32):
    """HxWx3 float32 [0,1] -> 1x3xHxW tensor on DEVICE."""
    t = torch.from_numpy(rgb_f32.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    return t

@torch.no_grad()
def calc_lpips(ref_rgb, test_rgb):
    """LPIPS Alex (lower = better). Normalise to [-1,1] as lpips expects."""
    r = to_tensor(ref_rgb) * 2 - 1
    t = to_tensor(test_rgb) * 2 - 1
    return float(lpips_fn(r, t).item())

@torch.no_grad()
def calc_dists(ref_rgb, test_rgb):
    """DISTS (lower = better). Expects [0,1] input."""
    r = to_tensor(ref_rgb)
    t = to_tensor(test_rgb)
    return float(dists_fn(r, t).item())

# -- YUV helpers --------------------------------------------------------------
def rgb_to_yuv(rgb):
    R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
    Y  =  0.2126*R + 0.7152*G + 0.0722*B
    Cb = -0.1146*R - 0.3854*G + 0.5000*B
    Cr =  0.5000*R - 0.4542*G - 0.0458*B
    return Y, Cb, Cr

def yuv_to_rgb(Y, Cb, Cr):
    R = np.clip(Y + 1.5748*Cr,             0, 1)
    G = np.clip(Y - 0.1873*Cb - 0.4681*Cr, 0, 1)
    B = np.clip(Y + 1.8556*Cb,             0, 1)
    return np.stack([R, G, B], -1)

def subsample420(C):
    return 0.25*(C[0::2,0::2]+C[1::2,0::2]+C[0::2,1::2]+C[1::2,1::2])

def upsample420(C, H, W):
    return np.repeat(np.repeat(C, 2, 0), 2, 1)[:H, :W]

def write_yuv420(path, Y, U, V):
    with open(path, 'wb') as f:
        Y.astype(np.float32).tofile(f)
        U.astype(np.float32).tofile(f)
        V.astype(np.float32).tofile(f)

def read_yuv420(path, H, W):
    d = np.fromfile(path, dtype=np.float32)
    cH, cW = H//2, W//2
    Y = d[:H*W].reshape(H,W)
    U = d[H*W:H*W+cH*cW].reshape(cH,cW)
    V = d[H*W+cH*cW:].reshape(cH,cW)
    return Y, U, V

# -- Noise estimation (Laplacian MAD) ----------------------------------------
def estimate_sigma_laplacian(plane):
    row = plane[:, 1:-1]
    lap = np.abs(plane[:, :-2] - 2*row + plane[:, 2:]).ravel()
    if len(lap) < 10: return 1e-4
    return float(max(np.median(lap) / 1.6521, 1e-5))

# -- Chroma denoising (non-GALOSH) -------------------------------------------
def denoise_chroma_gauss(C_noisy, sigma_c):
    kernel_sigma = max(sigma_c * 6.0, 0.5)
    return gaussian_filter(C_noisy.astype(np.float64),
                           sigma=kernel_sigma).astype(np.float32)

# -- GALOSH subprocess --------------------------------------------------------
def run_galosh(in_yuv, out_yuv, W, H, sy, sc, gamma=1.0):
    def win2bash(p): return p.replace('\\','/').replace('C:','/c')
    cmd = (f'{win2bash(EXE)} {win2bash(in_yuv)} {win2bash(out_yuv)} '
           f'{W} {H} 420 {sy} {sc} {STRIDE_Y} {STRIDE_C} {gamma}')
    env = dict(os.environ)
    env['PATH'] = 'C:/msys64/ucrt64/bin;C:/msys64/usr/bin;' + env.get('PATH','')
    subprocess.run([BASH, '-lc', cmd], capture_output=True,
                   text=True, encoding='utf-8', errors='replace', env=env)
    return os.path.exists(out_yuv)

# -- Scalar metrics -----------------------------------------------------------
def psnr(ref, test):
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64))**2)
    return 100.0 if mse < 1e-20 else 10*math.log10(1.0/mse)

def ssim_y(ref, test):
    return float(structural_similarity(
        ref.astype(np.float64), test.astype(np.float64), data_range=1.0))

# -- Method list --------------------------------------------------------------
METHODS = ['GALOSH-YUV', 'BM3D-oracle', 'BM3D-blind', 'NLM-oracle', 'Wiener-freq']

# -- Accumulator --------------------------------------------------------------
keys = ['psnr_y', 'ssim_y', 'psnr_rgb', 'lpips', 'dists', 't']
accumulator = {m: {k: [] for k in keys} for m in METHODS}
accumulator['noisy'] = {k: [] for k in keys}

# -- Main loop ----------------------------------------------------------------
rng = np.random.default_rng(0)
print(f"\nnoise sigma={noise_sigma}  galosh: sy={galosh_sy} sc={galosh_sc}  "
      f"stride={STRIDE_Y}/{STRIDE_C}  device={DEVICE}\n")
print(f"{'image':<12} {'method':<14} {'PSNR-Y':>7} {'SSIM-Y':>7} "
      f"{'PSNR-RGB':>9} {'LPIPS':>7} {'DISTS':>7}")
print("-" * 70)

for idx in range(1, 25):
    fname = f"kodim{idx:02d}.png"
    fpath = os.path.join(KODAK_DIR, fname)
    if not os.path.exists(fpath): continue

    rgb_clean = np.asarray(Image.open(fpath).convert("RGB"),
                           dtype=np.float32) / 255.0
    H, W = (rgb_clean.shape[0]//2)*2, (rgb_clean.shape[1]//2)*2
    rgb_clean = rgb_clean[:H, :W]

    Y_c, Cb_c, Cr_c = rgb_to_yuv(rgb_clean)
    U_c = subsample420(Cb_c)
    V_c = subsample420(Cr_c)

    Y_n = (Y_c + rng.normal(0, noise_sigma,      Y_c.shape)).astype(np.float32)
    U_n = (U_c + rng.normal(0, noise_sigma*0.5,  U_c.shape)).astype(np.float32)
    V_n = (V_c + rng.normal(0, noise_sigma*0.5,  V_c.shape)).astype(np.float32)

    # Noisy RGB for perceptual metrics
    rgb_n = yuv_to_rgb(Y_n, upsample420(U_n, H, W), upsample420(V_n, H, W))
    n_py  = psnr(Y_c, Y_n)
    n_sy  = ssim_y(Y_c, Y_n)
    n_pr  = psnr(rgb_clean, rgb_n)
    n_lp  = calc_lpips(rgb_clean, np.clip(rgb_n, 0, 1))
    n_di  = calc_dists(rgb_clean, np.clip(rgb_n, 0, 1))
    for k, v in zip(keys[:-1], [n_py, n_sy, n_pr, n_lp, n_di]):
        accumulator['noisy'][k].append(v)

    # Shared chroma for non-GALOSH
    sigma_c_est  = estimate_sigma_laplacian(U_n)
    U_den_gauss  = denoise_chroma_gauss(U_n, sigma_c_est)
    V_den_gauss  = denoise_chroma_gauss(V_n, sigma_c_est)

    results_y = {}

    # 1. GALOSH-YUV
    in_yuv  = os.path.join(WORK_DIR, f'cmp_in_{idx:02d}.yuv')
    out_yuv = os.path.join(WORK_DIR, f'cmp_out_{idx:02d}.yuv')
    write_yuv420(in_yuv, Y_n, U_n, V_n)
    t0 = time.time()
    ok = run_galosh(in_yuv, out_yuv, W, H, galosh_sy, galosh_sc)
    t_g = time.time() - t0
    if ok:
        Y_g, U_g, V_g = read_yuv420(out_yuv, H, W)
    else:
        Y_g, U_g, V_g = Y_n.copy(), U_n.copy(), V_n.copy()
    results_y['GALOSH-YUV'] = (Y_g, U_g, V_g, t_g)

    # 2. BM3D oracle
    t0 = time.time()
    Y_bm3d_ora = bm3d.bm3d(Y_n.astype(np.float64), sigma_psd=noise_sigma).astype(np.float32)
    results_y['BM3D-oracle'] = (Y_bm3d_ora, U_den_gauss, V_den_gauss, time.time()-t0)

    # 3. BM3D blind
    sigma_y_est = estimate_sigma_laplacian(Y_n)
    t0 = time.time()
    Y_bm3d_bl = bm3d.bm3d(Y_n.astype(np.float64), sigma_psd=sigma_y_est).astype(np.float32)
    results_y['BM3D-blind'] = (Y_bm3d_bl, U_den_gauss, V_den_gauss, time.time()-t0)

    # 4. NLM oracle
    Y_u8  = np.clip(Y_n * 255, 0, 255).astype(np.uint8)
    h_nlm = max(1, int(noise_sigma * 255 * 0.8))
    t0 = time.time()
    Y_nlm_u8 = cv2.fastNlMeansDenoising(Y_u8, None, h=h_nlm,
                                          templateWindowSize=7, searchWindowSize=21)
    Y_nlm = Y_nlm_u8.astype(np.float32) / 255.0
    results_y['NLM-oracle'] = (Y_nlm, U_den_gauss, V_den_gauss, time.time()-t0)

    # 5. Wiener freq
    t0 = time.time()
    F = np.fft.rfft2(Y_n.astype(np.float64))
    S_noisy = np.abs(F)**2 / Y_n.size
    S_noise = noise_sigma**2
    S_sig   = np.maximum(S_noisy - S_noise, 0)
    G       = S_sig / (S_sig + S_noise + 1e-30)
    Y_wien  = np.fft.irfft2(F * G, s=Y_n.shape).astype(np.float32)
    results_y['Wiener-freq'] = (Y_wien, U_den_gauss, V_den_gauss, time.time()-t0)

    # Per-image print: noisy first
    print(f"{fname:<12} {'Noisy':<14} {n_py:>7.3f} {n_sy:>7.4f} "
          f"{n_pr:>9.3f} {n_lp:>7.4f} {n_di:>7.4f}")

    for M in METHODS:
        Y_d, U_d, V_d, t_m = results_y[M]
        py = psnr(Y_c, Y_d)
        sy = ssim_y(Y_c, Y_d)
        Cb_up = upsample420(U_d, H, W)
        Cr_up = upsample420(V_d, H, W)
        rgb_d = np.clip(yuv_to_rgb(Y_d, Cb_up, Cr_up), 0, 1)
        pr  = psnr(rgb_clean, rgb_d)
        lp  = calc_lpips(rgb_clean, rgb_d)
        di  = calc_dists(rgb_clean, rgb_d)
        for k, v in zip(keys, [py, sy, pr, lp, di, t_m]):
            accumulator[M][k].append(v)
        print(f"{'':12} {M:<14} {py:>7.3f} {sy:>7.4f} "
              f"{pr:>9.3f} {lp:>7.4f} {di:>7.4f}")
    print()

# -- Summary ------------------------------------------------------------------
print("=" * 70)
print(f"  SUMMARY  noise sigma={noise_sigma}  galosh: sy={galosh_sy} sc={galosh_sc}")
print("=" * 70)
print(f"{'Method':<14} {'PSNR-Y':>7} {'SSIM-Y':>7} {'PSNR-RGB':>9} "
      f"{'LPIPS':>7} {'DISTS':>7} {'Time/img':>9}")
print("-" * 70)

n_py = np.mean(accumulator['noisy']['psnr_y'])
n_lp = np.mean(accumulator['noisy']['lpips'])
n_di = np.mean(accumulator['noisy']['dists'])
print(f"{'Noisy':<14} {n_py:>7.3f} "
      f"{np.mean(accumulator['noisy']['ssim_y']):>7.4f} "
      f"{np.mean(accumulator['noisy']['psnr_rgb']):>9.3f} "
      f"{n_lp:>7.4f} {n_di:>7.4f}  {'---':>8}")

for M in METHODS:
    a = accumulator[M]
    print(f"{M:<14} {np.mean(a['psnr_y']):>7.3f} "
          f"{np.mean(a['ssim_y']):>7.4f} "
          f"{np.mean(a['psnr_rgb']):>9.3f} "
          f"{np.mean(a['lpips']):>7.4f} "
          f"{np.mean(a['dists']):>7.4f} "
          f"{np.mean(a['t']):>8.2f}s")

print()
print("LPIPS/DISTS: lower is better. LPIPS=AlexNet, DISTS=texture-aware.")
print("GALOSH-YUV chroma: own UV denoiser. Others: Gaussian chroma shrinkage.")