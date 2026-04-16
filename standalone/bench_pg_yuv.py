"""
bench_pg_yuv.py  ---  Poisson-Gaussian YUV420 benchmark on Kodak

Noise model (linear domain, per-pixel):
    x_noisy = Poisson(x_linear / alpha) * alpha + N(0, sigma^2)
    Var(x_noisy) = alpha * x_linear + sigma^2

Pipeline for each image:
    PNG (sRGB 8-bit)
    -> inv_sRGB_gamma -> linear [0,1]
    -> add PG noise in linear domain
    -> sRGB_gamma
    -> YUV420 conversion
    -> GALOSH (gamma=2.2) / BM3D / NLM
    -> metric in sRGB domain vs clean sRGB

ISO presets:   (alpha, sigma)
    ISO 800:   (0.002, 0.010)
    ISO 3200:  (0.008, 0.020)
    ISO 12800: (0.030, 0.040)

Usage:
    python bench_pg_yuv.py [iso_preset]
    iso_preset: 800 | 3200 | 12800  (default: 3200)
"""

import sys, os, subprocess, time, math
import numpy as np
import bm3d
import cv2
import torch
import lpips
from DISTS_pytorch import DISTS
from PIL import Image
from skimage.metrics import structural_similarity

# ── Config ────────────────────────────────────────────────────────────
KODAK_DIR = r"C:\Users\luxgrain\denoise_eval\datasets\kodak"
EXE       = r"C:\Users\luxgrain\denoise_eval\standalone\yuv_galosh.exe"
BASH      = r"C:\msys64\usr\bin\bash.exe"
WORK_DIR  = r"C:\Users\luxgrain\denoise_eval\standalone\yuv_bench"
STRIDE_Y  = 2
STRIDE_C  = 2
GAMMA_EXE = 2.2       # tell galosh the input is sRGB-gamma-encoded

ISO_PRESETS = {
    800:   (0.002, 0.010),
    3200:  (0.008, 0.020),
    12800: (0.030, 0.040),
}

iso_preset = int(sys.argv[1]) if len(sys.argv) > 1 else 3200
if iso_preset not in ISO_PRESETS:
    print(f"Unknown ISO preset {iso_preset}. Choose from {list(ISO_PRESETS)}"); sys.exit(1)
PG_ALPHA, PG_SIGMA = ISO_PRESETS[iso_preset]

os.makedirs(WORK_DIR, exist_ok=True)

# ── Perceptual metrics ─────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_fn = lpips.LPIPS(net='alex').to(DEVICE); lpips_fn.eval()
dists_fn = DISTS().to(DEVICE);                 dists_fn.eval()

def to_tensor(rgb):
    return torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)

@torch.no_grad()
def calc_lpips(ref, test):
    return float(lpips_fn(to_tensor(ref)*2-1, to_tensor(test)*2-1).item())

@torch.no_grad()
def calc_dists(ref, test):
    return float(dists_fn(to_tensor(ref), to_tensor(test)).item())

# ── Gamma helpers ──────────────────────────────────────────────────────
def srgb_to_linear(x):
    """IEC 61966-2-1 piecewise (exact, not power-law approx)."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0/2.4) - 0.055)

# ── Poisson-Gaussian noise ─────────────────────────────────────────────
def add_pg_noise(lin, alpha, sigma, rng):
    """
    Add Poisson-Gaussian noise to a linear-domain plane in [0,1].
    Var(out) = alpha * lin + sigma^2
    """
    if alpha < 1e-8:
        return lin + rng.normal(0.0, sigma, lin.shape).astype(np.float32)
    # Scale to photon counts, sample Poisson, rescale back
    scale   = 1.0 / alpha
    photons = rng.poisson(np.maximum(lin * scale, 0.0)).astype(np.float64)
    noisy   = photons / scale + rng.normal(0.0, sigma, lin.shape)
    return noisy.astype(np.float32)

# ── YUV helpers ────────────────────────────────────────────────────────
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
    return (d[:H*W].reshape(H,W),
            d[H*W:H*W+cH*cW].reshape(cH,cW),
            d[H*W+cH*cW:].reshape(cH,cW))

# ── Run GALOSH ─────────────────────────────────────────────────────────
def run_galosh(in_yuv, out_yuv, W, H, sy=1.0, sc=1.0, gamma=2.2):
    def b(p): return p.replace('\\','/').replace('C:','/c')
    cmd = (f'{b(EXE)} {b(in_yuv)} {b(out_yuv)} '
           f'{W} {H} 420 {sy} {sc} {STRIDE_Y} {STRIDE_C} {gamma}')
    env = dict(os.environ)
    env['PATH'] = 'C:/msys64/ucrt64/bin;C:/msys64/usr/bin;' + env.get('PATH','')
    subprocess.run([BASH, '-lc', cmd], capture_output=True,
                   text=True, encoding='utf-8', errors='replace', env=env)
    return os.path.exists(out_yuv)

# ── Metrics ────────────────────────────────────────────────────────────
def psnr(ref, test):
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64))**2)
    return 100.0 if mse < 1e-20 else 10*math.log10(1.0/mse)

def ssim_y(ref, test):
    return float(structural_similarity(
        ref.astype(np.float64), test.astype(np.float64), data_range=1.0))

# ── BM3D effective sigma (sigma_psd) for PG model ─────────────────────
def bm3d_sigma_psd(alpha, sigma, mean_signal=0.18):
    """
    For BM3D-oracle: effective Gaussian sigma that matches PG variance
    at the expected signal level (assume mean ~18% grey in linear).
    sigma_eff = sqrt(alpha * mean + sigma^2)
    """
    return math.sqrt(alpha * mean_signal + sigma**2)

# ── Chroma Gaussian denoising (shared by non-GALOSH methods) ──────────
def estimate_sigma_lap(plane):
    row = plane[:, 1:-1]
    lap = np.abs(plane[:,:-2] - 2*row + plane[:,2:]).ravel()
    if len(lap) < 10: return 1e-4
    return float(max(np.median(lap) / 1.6521, 1e-5))

from scipy.ndimage import gaussian_filter
def denoise_chroma_gauss(C, sigma_c):
    return gaussian_filter(C.astype(np.float64), sigma=max(sigma_c*6, 0.5)).astype(np.float32)

# ── Main ───────────────────────────────────────────────────────────────
METHODS = ['GALOSH-YUV', 'BM3D-oracle', 'BM3D-blind', 'NLM-oracle']
keys    = ['psnr_y', 'ssim_y', 'psnr_rgb', 'lpips', 'dists', 't']
acc     = {m: {k: [] for k in keys} for m in METHODS}
acc['noisy'] = {k: [] for k in keys}

rng = np.random.default_rng(0)

print(f"\nPoisson-Gaussian YUV420 Benchmark  ISO~{iso_preset}")
print(f"  alpha={PG_ALPHA}  sigma={PG_SIGMA}  gamma={GAMMA_EXE}  stride={STRIDE_Y}/{STRIDE_C}")
print(f"  device={DEVICE}\n")
print(f"{'image':<12} {'method':<14} {'PSNR-Y':>7} {'SSIM-Y':>7} "
      f"{'PSNR-RGB':>9} {'LPIPS':>7} {'DISTS':>7}")
print("-" * 70)

for idx in range(1, 25):
    fname = f"kodim{idx:02d}.png"
    fpath = os.path.join(KODAK_DIR, fname)
    if not os.path.exists(fpath): continue

    # Load clean sRGB
    rgb_srgb_clean = np.asarray(Image.open(fpath).convert("RGB"),
                                dtype=np.float32) / 255.0
    H, W = (rgb_srgb_clean.shape[0]//2)*2, (rgb_srgb_clean.shape[1]//2)*2
    rgb_srgb_clean = rgb_srgb_clean[:H, :W]

    # Linearise
    rgb_lin_clean = srgb_to_linear(rgb_srgb_clean)

    # Add PG noise independently per channel in linear domain
    R_n = add_pg_noise(rgb_lin_clean[...,0], PG_ALPHA, PG_SIGMA, rng)
    G_n = add_pg_noise(rgb_lin_clean[...,1], PG_ALPHA, PG_SIGMA, rng)
    B_n = add_pg_noise(rgb_lin_clean[...,2], PG_ALPHA, PG_SIGMA, rng)
    rgb_lin_noisy = np.stack([R_n, G_n, B_n], -1).astype(np.float32)

    # Re-apply sRGB gamma
    rgb_srgb_noisy = linear_to_srgb(rgb_lin_noisy).astype(np.float32)

    # Convert noisy to YUV420 (in sRGB domain, as camera/encoder would)
    Y_n_yuv, Cb_n, Cr_n = rgb_to_yuv(rgb_srgb_noisy)
    U_n = subsample420(Cb_n).astype(np.float32)
    V_n = subsample420(Cr_n).astype(np.float32)
    Y_n = Y_n_yuv.astype(np.float32)

    # Noisy sRGB metrics
    rgb_srgb_noisy_c = np.clip(rgb_srgb_noisy, 0, 1)
    n_py = psnr(rgb_to_yuv(rgb_srgb_clean)[0], Y_n)
    n_sy = ssim_y(rgb_to_yuv(rgb_srgb_clean)[0], Y_n)
    n_pr = psnr(rgb_srgb_clean, rgb_srgb_noisy_c)
    n_lp = calc_lpips(rgb_srgb_clean, rgb_srgb_noisy_c)
    n_di = calc_dists(rgb_srgb_clean, rgb_srgb_noisy_c)
    for k,v in zip(keys[:-1],[n_py,n_sy,n_pr,n_lp,n_di]):
        acc['noisy'][k].append(v)

    # Shared Gaussian chroma for non-GALOSH methods
    sigma_c_est  = estimate_sigma_lap(U_n)
    U_den_gauss  = denoise_chroma_gauss(U_n, sigma_c_est)
    V_den_gauss  = denoise_chroma_gauss(V_n, sigma_c_est)

    results_y = {}

    # 1. GALOSH-YUV (gamma=2.2, full internal linearise pipeline)
    in_yuv  = os.path.join(WORK_DIR, f'pg_in_{idx:02d}.yuv')
    out_yuv = os.path.join(WORK_DIR, f'pg_out_{idx:02d}.yuv')
    write_yuv420(in_yuv, Y_n, U_n, V_n)
    t0 = time.time()
    ok = run_galosh(in_yuv, out_yuv, W, H, gamma=GAMMA_EXE)
    t_g = time.time() - t0
    if ok:
        Y_g, U_g, V_g = read_yuv420(out_yuv, H, W)
    else:
        Y_g, U_g, V_g = Y_n.copy(), U_n.copy(), V_n.copy()
    results_y['GALOSH-YUV'] = (Y_g, U_g, V_g, t_g)

    # 2. BM3D oracle — estimate effective sigma for PG model
    # BM3D operates on sRGB Y (gamma-domain), sigma_eff estimated from oracle params
    sigma_eff = bm3d_sigma_psd(PG_ALPHA, PG_SIGMA)
    t0 = time.time()
    Y_bm3d_ora = bm3d.bm3d(Y_n.astype(np.float64),
                             sigma_psd=sigma_eff).astype(np.float32)
    results_y['BM3D-oracle'] = (Y_bm3d_ora, U_den_gauss, V_den_gauss, time.time()-t0)

    # 3. BM3D blind (Laplacian MAD estimate in sRGB Y domain)
    sigma_y_est = estimate_sigma_lap(Y_n)
    t0 = time.time()
    Y_bm3d_bl = bm3d.bm3d(Y_n.astype(np.float64),
                            sigma_psd=sigma_y_est).astype(np.float32)
    results_y['BM3D-blind'] = (Y_bm3d_bl, U_den_gauss, V_den_gauss, time.time()-t0)

    # 4. NLM oracle — h from effective sigma
    Y_u8  = np.clip(Y_n * 255, 0, 255).astype(np.uint8)
    h_nlm = max(1, int(sigma_eff * 255 * 0.8))
    t0 = time.time()
    Y_nlm_u8 = cv2.fastNlMeansDenoising(Y_u8, None, h=h_nlm,
                                          templateWindowSize=7, searchWindowSize=21)
    Y_nlm = Y_nlm_u8.astype(np.float32) / 255.0
    results_y['NLM-oracle'] = (Y_nlm, U_den_gauss, V_den_gauss, time.time()-t0)

    # Metrics
    Y_c_srgb = rgb_to_yuv(rgb_srgb_clean)[0]  # clean Y in sRGB domain

    print(f"{fname:<12} {'Noisy':<14} {n_py:>7.3f} {n_sy:>7.4f} "
          f"{n_pr:>9.3f} {n_lp:>7.4f} {n_di:>7.4f}")

    for M in METHODS:
        Y_d, U_d, V_d, t_m = results_y[M]
        py = psnr(Y_c_srgb, Y_d)
        sy = ssim_y(Y_c_srgb, Y_d)
        Cb_up = upsample420(U_d, H, W)
        Cr_up = upsample420(V_d, H, W)
        rgb_d = np.clip(yuv_to_rgb(Y_d, Cb_up, Cr_up), 0, 1)
        pr = psnr(rgb_srgb_clean, rgb_d)
        lp = calc_lpips(rgb_srgb_clean, rgb_d)
        di = calc_dists(rgb_srgb_clean, rgb_d)
        for k,v in zip(keys,[py,sy,pr,lp,di,t_m]):
            acc[M][k].append(v)
        print(f"{'':12} {M:<14} {py:>7.3f} {sy:>7.4f} "
              f"{pr:>9.3f} {lp:>7.4f} {di:>7.4f}")
    print()

# ── Summary ────────────────────────────────────────────────────────────
print("=" * 70)
print(f"  SUMMARY  ISO~{iso_preset}  alpha={PG_ALPHA}  sigma={PG_SIGMA}")
print("=" * 70)
print(f"{'Method':<14} {'PSNR-Y':>7} {'SSIM-Y':>7} {'PSNR-RGB':>9} "
      f"{'LPIPS':>7} {'DISTS':>7} {'Time/img':>9}")
print("-" * 70)

n_py_m = np.mean(acc['noisy']['psnr_y'])
print(f"{'Noisy':<14} {n_py_m:>7.3f} "
      f"{np.mean(acc['noisy']['ssim_y']):>7.4f} "
      f"{np.mean(acc['noisy']['psnr_rgb']):>9.3f} "
      f"{np.mean(acc['noisy']['lpips']):>7.4f} "
      f"{np.mean(acc['noisy']['dists']):>7.4f}  {'---':>8}")

for M in METHODS:
    a = acc[M]
    print(f"{M:<14} {np.mean(a['psnr_y']):>7.3f} "
          f"{np.mean(a['ssim_y']):>7.4f} "
          f"{np.mean(a['psnr_rgb']):>9.3f} "
          f"{np.mean(a['lpips']):>7.4f} "
          f"{np.mean(a['dists']):>7.4f} "
          f"{np.mean(a['t']):>8.2f}s")

print()
print("Notes:")
print(f"  PG noise added in linear domain: alpha={PG_ALPHA}, sigma={PG_SIGMA}")
print(f"  GALOSH: sRGB input with gamma={GAMMA_EXE} (internal linearise pipeline)")
print(f"  BM3D-oracle: sigma_eff=sqrt(alpha*0.18+sigma^2)={bm3d_sigma_psd(PG_ALPHA,PG_SIGMA):.4f} (sRGB Y domain)")
print(f"  PSNR-RGB: GALOSH uses own UV denoiser; others use Gaussian chroma.")