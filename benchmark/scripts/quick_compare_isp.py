"""
Quick comparison: noisy ISP sRGB vs GALOSH YUV (gamma=1.0 and 2.2)
Measures PSNR against ISP GT to show actual improvement.
"""
import os
import numpy as np, scipy.io as sio, subprocess, os
from pathlib import Path

BASE  = Path(os.path.expanduser(r'~\denoise_eval'))
SIDD  = BASE / 'datasets' / 'sidd'
BASH  = Path(r'C:\msys64\usr\bin\bash.exe')
EXE   = BASE / 'standalone' / 'yuv_galosh.exe'
YUBD  = BASE / 'standalone' / 'yuv_bench'
YUBD.mkdir(exist_ok=True)

gt_mat    = sio.loadmat(str(SIDD / 'ValidationGtBlocksSrgb.mat'))['ValidationGtBlocksSrgb']
noisy_mat = sio.loadmat(str(SIDD / 'ValidationNoisyBlocksSrgb.mat'))['ValidationNoisyBlocksSrgb']

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return 10*np.log10(1/mse) if mse > 1e-10 else 100.0

def w2b(p):
    return str(p).replace('C:', '/c').replace('\\', '/')

def run_galosh_yuv(noisy_srgb, uid, gamma=1.0):
    H, W = noisy_srgb.shape[:2]
    R, G, B = noisy_srgb[...,0], noisy_srgb[...,1], noisy_srgb[...,2]
    Y  =  0.2126*R + 0.7152*G + 0.0722*B
    Cb = -0.1146*R - 0.3854*G + 0.5000*B
    Cr =  0.5000*R - 0.4542*G - 0.0458*B
    ip = YUBD / f'q_{uid}_in.yuv'
    op = YUBD / f'q_{uid}_out.yuv'
    with open(str(ip), 'wb') as f:
        Y.astype(np.float32).tofile(f)
        Cb.astype(np.float32).tofile(f)
        Cr.astype(np.float32).tofile(f)
    cmd = f'{w2b(EXE)} {w2b(ip)} {w2b(op)} {W} {H} 444 1.0 1.0 2 2 {gamma}'
    env = dict(os.environ)
    env['PATH'] = 'C:/msys64/ucrt64/bin;C:/msys64/usr/bin;' + env.get('PATH', '')
    subprocess.run([str(BASH), '-lc', cmd], capture_output=True, env=env, timeout=60)
    if not op.exists():
        return None
    raw = np.fromfile(str(op), dtype=np.float32)
    os.remove(str(ip)); os.remove(str(op))
    Yd  = raw[      : H*W    ].reshape(H, W)
    Cbd = raw[H*W   : 2*H*W  ].reshape(H, W)
    Crd = raw[2*H*W :         ].reshape(H, W)
    R_d = np.clip(Yd + 1.5748*Crd,                  0, 1)
    G_d = np.clip(Yd - 0.1873*Cbd - 0.4681*Crd,     0, 1)
    B_d = np.clip(Yd + 1.8556*Cbd,                  0, 1)
    return np.stack([R_d, G_d, B_d], 2).astype(np.float32)

# Test on scene 0, patches 0-9
print(f"\n{'Patch':>5} | {'Noisy':>7} | {'GALOSH g=1':>10} | {'GALOSH g=2.2':>12} | {'delta(g=2.2)':>12}")
print('-' * 60)

n_list, g1_list, g2_list = [], [], []
for pi in range(10):
    gt_f    = gt_mat[0, pi].astype(np.float32) / 255.0
    noisy_f = noisy_mat[0, pi].astype(np.float32) / 255.0
    pn = psnr(gt_f, noisy_f)
    d1 = run_galosh_yuv(noisy_f, f's0p{pi}g1',  gamma=1.0)
    d2 = run_galosh_yuv(noisy_f, f's0p{pi}g22', gamma=2.2)
    p1 = psnr(gt_f, d1) if d1 is not None else float('nan')
    p2 = psnr(gt_f, d2) if d2 is not None else float('nan')
    n_list.append(pn); g1_list.append(p1); g2_list.append(p2)
    print(f"{pi:>5} | {pn:>7.2f} | {p1:>10.2f} | {p2:>12.2f} | {p2-pn:>+12.2f}")

print('-' * 60)
print(f"{'avg':>5} | {np.mean(n_list):>7.2f} | {np.mean(g1_list):>10.2f} | "
      f"{np.mean(g2_list):>12.2f} | {np.mean(g2_list)-np.mean(n_list):>+12.2f}")
print()
print("Note: ISP sRGB GT is a harder target than bilinear demosaic GT.")
print("GALOSH_F 38.33 dB was measured against bilinear GT (lower bar).")
