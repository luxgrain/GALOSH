"""
Debug GALOSH YUV444 on real SIDD ISP sRGB patches.
Shows exe's internal sigma/alpha estimates and saves input/output images.
"""
import numpy as np
import subprocess, os, sys
from pathlib import Path
import scipy.io as sio
from skimage.io import imsave

BASE    = Path(__file__).parent.parent
SIDD    = BASE / "datasets" / "sidd"
EXE     = BASE / "standalone" / "yuv_galosh.exe"
BASH    = Path(r"C:\msys64\usr\bin\bash.exe")
OUTDIR  = BASE / "debug_yuv444"
OUTDIR.mkdir(exist_ok=True)

def win2bash(p):
    return str(p).replace("C:", "/c").replace("\\", "/")

def run_galosh_yuv444_debug(Y, Cb, Cr, uid):
    H, W = Y.shape
    tmpdir = BASE / "standalone" / "yuv_bench"
    tmpdir.mkdir(exist_ok=True)
    in_path  = str(tmpdir / f"dbg_{uid}_in.yuv")
    out_path = str(tmpdir / f"dbg_{uid}_out.yuv")
    with open(in_path, "wb") as f:
        Y.astype(np.float32).tofile(f)
        Cb.astype(np.float32).tofile(f)
        Cr.astype(np.float32).tofile(f)
    cmd = f"{win2bash(EXE)} {win2bash(in_path)} {win2bash(out_path)} {W} {H} 444 1.0 1.0 2 2"
    env = dict(os.environ)
    env["PATH"] = "C:/msys64/ucrt64/bin;C:/msys64/usr/bin;" + env.get("PATH", "")
    r = subprocess.run([str(BASH), "-lc", cmd], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env, timeout=60)
    print(f"  [stderr] {r.stderr.strip()}")
    if not os.path.exists(out_path):
        print("  OUTPUT MISSING"); return None, None, None
    d = np.fromfile(out_path, dtype=np.float32)
    os.remove(in_path); os.remove(out_path)
    Y_d  = d[0:H*W].reshape(H, W)
    Cb_d = d[H*W:2*H*W].reshape(H, W)
    Cr_d = d[2*H*W:].reshape(H, W)
    return Y_d, Cb_d, Cr_d

# Load SIDD sRGB mats
print("Loading SIDD sRGB mats...")
gt_mat    = sio.loadmat(str(SIDD / "ValidationGtBlocksSrgb.mat"))["ValidationGtBlocksSrgb"]
noisy_mat = sio.loadmat(str(SIDD / "ValidationNoisyBlocksSrgb.mat"))["ValidationNoisyBlocksSrgb"]
# (40, 32, 256, 256, 3) uint8
print(f"Shape: {gt_mat.shape}, dtype: {gt_mat.dtype}")

# Test on first 5 patches of scene 0
n_test = 5
psnr_list = []

for pi in range(n_test):
    gt_u8    = gt_mat[0, pi]         # (256, 256, 3) uint8
    noisy_u8 = noisy_mat[0, pi]

    gt_f    = gt_u8.astype(np.float32) / 255.0
    noisy_f = noisy_u8.astype(np.float32) / 255.0
    H, W    = gt_f.shape[:2]

    R, G, B = noisy_f[...,0], noisy_f[...,1], noisy_f[...,2]
    Y  =  0.2126*R + 0.7152*G + 0.0722*B
    Cb = -0.1146*R - 0.3854*G + 0.5000*B
    Cr =  0.5000*R - 0.4542*G - 0.0458*B

    print(f"\nPatch {pi}:")
    print(f"  noisy Y   range [{Y.min():.4f}, {Y.max():.4f}]  mean={Y.mean():.4f}  std={Y.std():.4f}")
    print(f"  noisy Cb  std={Cb.std():.4f}")
    print(f"  noisy Cr  std={Cr.std():.4f}")

    Y_d, Cb_d, Cr_d = run_galosh_yuv444_debug(Y, Cb, Cr, f"s0p{pi}")
    if Y_d is None:
        continue

    R_d = np.clip(Y_d + 1.5748*Cr_d,                0, 1)
    G_d = np.clip(Y_d - 0.1873*Cb_d - 0.4681*Cr_d, 0, 1)
    B_d = np.clip(Y_d + 1.8556*Cb_d,                0, 1)
    den_f = np.stack([R_d, G_d, B_d], axis=2).astype(np.float32)

    mse = np.mean((gt_f.astype(np.float64) - den_f.astype(np.float64))**2)
    psnr = 10*np.log10(1.0/mse) if mse > 1e-10 else 100.0
    mse_n = np.mean((gt_f.astype(np.float64) - noisy_f.astype(np.float64))**2)
    psnr_noisy = 10*np.log10(1.0/mse_n) if mse_n > 1e-10 else 100.0
    psnr_list.append(psnr)

    print(f"  denoised Y range [{Y_d.min():.4f}, {Y_d.max():.4f}]")
    print(f"  PSNR: noisy={psnr_noisy:.2f} dB  denoised={psnr:.2f} dB")

    # Save comparison (noisy | gt | denoised)
    gap = np.ones((H, 4, 3), dtype=np.uint8) * 128
    comp = np.concatenate([
        (noisy_f * 255).astype(np.uint8), gap,
        (gt_f    * 255).astype(np.uint8), gap,
        (den_f   * 255).astype(np.uint8),
    ], axis=1)
    imsave(str(OUTDIR / f"s0p{pi}_comparison.png"), comp)

if psnr_list:
    print(f"\nAvg PSNR: {np.mean(psnr_list):.2f} dB over {len(psnr_list)} patches")
