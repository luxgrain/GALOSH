"""
bench_yuv_galosh.py
Kodak YUV GALOSH benchmark
  PNG (clean) → add Gaussian noise → denoise with yuv_galosh.exe → PSNR / SSIM

Usage:
  python bench_yuv_galosh.py [sigma] [sigma_y] [sigma_c]
  sigma    : noise level added to Y (default 0.05 = ISO~800 equivalent)
  sigma_y  : galosh Y strength   (default 1.0)
  sigma_c  : galosh chroma strength (default 1.0)
"""

import sys, os, struct, subprocess, math, time
import numpy as np
from PIL import Image

# ── config ──────────────────────────────────────────────────────────
KODAK_DIR = r"C:\Users\luxgrain\denoise_eval\datasets\kodak"
EXE       = r"C:\Users\luxgrain\denoise_eval\standalone\yuv_galosh.exe"
BASH      = r"C:\msys64\usr\bin\bash.exe"
WORK_DIR  = r"C:\Users\luxgrain\denoise_eval\standalone\yuv_bench"
N_IMAGES  = 24          # all 24 Kodak images
STRIDE_Y  = 2
STRIDE_C  = 2
YUV_FMT   = 420

noise_sigma   = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
user_sigma_y  = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
user_sigma_c  = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

os.makedirs(WORK_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────
def rgb_to_yuv420(rgb_f32):
    """RGB float [0,1] H×W×3 → Y [H×W], U [H/2×W/2], V [H/2×W/2] float."""
    R, G, B = rgb_f32[...,0], rgb_f32[...,1], rgb_f32[...,2]
    Y  =  0.2126*R + 0.7152*G + 0.0722*B
    Cb = -0.1146*R - 0.3854*G + 0.5000*B
    Cr =  0.5000*R - 0.4542*G - 0.0458*B
    # 420 subsample: simple 2x2 average
    Cb420 = 0.25*(Cb[0::2,0::2]+Cb[1::2,0::2]+Cb[0::2,1::2]+Cb[1::2,1::2])
    Cr420 = 0.25*(Cr[0::2,0::2]+Cr[1::2,0::2]+Cr[0::2,1::2]+Cr[1::2,1::2])
    return Y.astype(np.float32), Cb420.astype(np.float32), Cr420.astype(np.float32)

def yuv420_to_rgb(Y, Cb420, Cr420):
    """Reconstruct RGB from YUV420 (nearest-neighbour upsampling for PSNR)."""
    H, W = Y.shape
    Cb = np.repeat(np.repeat(Cb420, 2, axis=0), 2, axis=1)[:H,:W]
    Cr = np.repeat(np.repeat(Cr420, 2, axis=0), 2, axis=1)[:H,:W]
    R = Y + 1.5748*Cr
    G = Y - 0.1873*Cb - 0.4681*Cr
    B = Y + 1.8556*Cb
    return np.stack([R,G,B], axis=-1)

def psnr(ref, test, peak=1.0):
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64))**2)
    if mse < 1e-20: return 100.0
    return 10.0 * math.log10(peak*peak / mse)

def write_yuv420(path, Y, U, V):
    with open(path, 'wb') as f:
        Y.tofile(f); U.tofile(f); V.tofile(f)

def read_yuv420(path, H, W):
    cH, cW = H//2, W//2
    data = np.fromfile(path, dtype=np.float32)
    Y = data[:H*W].reshape(H, W)
    U = data[H*W:H*W+cH*cW].reshape(cH, cW)
    V = data[H*W+cH*cW:].reshape(cH, cW)
    return Y, U, V

def run_galosh(in_yuv, out_yuv, W, H, sy, sc):
    in_p  = in_yuv .replace('\\','/').replace('C:','/c')
    out_p = out_yuv.replace('\\','/').replace('C:','/c')
    cmd = (f'{in_p} {out_p} {W} {H} {YUV_FMT} {sy} {sc} '
           f'{STRIDE_Y} {STRIDE_C}')
    exe_p = EXE.replace('\\','/').replace('C:','/c')
    env = dict(os.environ)
    env['PATH'] = 'C:/msys64/ucrt64/bin;C:/msys64/usr/bin;' + env.get('PATH','')
    r = subprocess.run([BASH, '-lc', f'{exe_p} {cmd}'],
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace', env=env)
    return r

# ── main loop ────────────────────────────────────────────────────────
rng = np.random.default_rng(0)

results = []
print(f"{'img':<12} {'PSNR_noisy':>11} {'PSNR_den':>10} {'gain':>6}  "
      f"{'PSNR_Y_nz':>10} {'PSNR_Y_dn':>10}  time(s)")
print("-"*80)

for idx in range(1, N_IMAGES+1):
    fname = f"kodim{idx:02d}.png"
    fpath = os.path.join(KODAK_DIR, fname)
    if not os.path.exists(fpath):
        continue

    img = Image.open(fpath).convert("RGB")
    rgb_clean = np.asarray(img, dtype=np.float32) / 255.0
    H, W = rgb_clean.shape[:2]
    # Kodak images are 768×512 or 512×768 — ensure even dims for YUV420
    H = (H // 2) * 2
    W = (W // 2) * 2
    rgb_clean = rgb_clean[:H, :W]

    # Clean YUV
    Y_clean, U_clean, V_clean = rgb_to_yuv420(rgb_clean)

    # Add Gaussian noise (same sigma to Y; chroma noise ~ sigma/4 after 420 avg)
    Y_noisy  = Y_clean + rng.normal(0, noise_sigma, Y_clean.shape).astype(np.float32)
    U_noisy  = U_clean + rng.normal(0, noise_sigma*0.5, U_clean.shape).astype(np.float32)
    V_noisy  = V_clean + rng.normal(0, noise_sigma*0.5, V_clean.shape).astype(np.float32)

    # Write noisy YUV to disk
    in_yuv  = os.path.join(WORK_DIR, f"noisy_{idx:02d}.yuv")
    out_yuv = os.path.join(WORK_DIR, f"denoised_{idx:02d}.yuv")
    write_yuv420(in_yuv, Y_noisy, U_noisy, V_noisy)

    # Run galosh
    t0 = time.time()
    r  = run_galosh(in_yuv, out_yuv, W, H, user_sigma_y, user_sigma_c)
    elapsed = time.time() - t0

    if not os.path.exists(out_yuv):
        print(f"{fname:<12}  FAILED: {r.stdout[-200:]}")
        continue

    Y_den, U_den, V_den = read_yuv420(out_yuv, H, W)

    # PSNR on Y only (most meaningful for luma)
    psnr_y_noisy = psnr(Y_clean, Y_noisy)
    psnr_y_den   = psnr(Y_clean, Y_den)

    # PSNR on full RGB (reconstruct from YUV)
    rgb_noisy = yuv420_to_rgb(Y_noisy, U_noisy, V_noisy).clip(0,1)
    rgb_den   = yuv420_to_rgb(Y_den,   U_den,   V_den  ).clip(0,1)
    psnr_noisy_rgb = psnr(rgb_clean, rgb_noisy)
    psnr_den_rgb   = psnr(rgb_clean, rgb_den)

    gain = psnr_den_rgb - psnr_noisy_rgb
    results.append((psnr_noisy_rgb, psnr_den_rgb, psnr_y_noisy, psnr_y_den))

    print(f"{fname:<12} {psnr_noisy_rgb:>11.3f} {psnr_den_rgb:>10.3f} "
          f"{gain:>+6.3f}  {psnr_y_noisy:>10.3f} {psnr_y_den:>10.3f}  {elapsed:.1f}s")

if results:
    arr = np.array(results)
    print("-"*80)
    print(f"{'MEAN':<12} {arr[:,0].mean():>11.3f} {arr[:,1].mean():>10.3f} "
          f"{(arr[:,1]-arr[:,0]).mean():>+6.3f}  "
          f"{arr[:,2].mean():>10.3f} {arr[:,3].mean():>10.3f}")
    print()
    print(f"Config: noise_sigma={noise_sigma}  sigma_y={user_sigma_y}  "
          f"sigma_c={user_sigma_c}  stride={STRIDE_Y}/{STRIDE_C}  fmt=YUV{YUV_FMT}")