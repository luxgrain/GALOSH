#!/usr/bin/env python3
"""
V2 Multichannel benchmark: BM3D-PC vs BM3D-CFA vs Ours(V2) vs DRUNet vs DnCNN-B
Kodak 24 x 6 ISOs, PSNR/SSIM/LPIPS
"""
import numpy as np
import subprocess
import os
import sys
import json
import time
import torch
from pathlib import Path
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import uniform_filter
import bm3d as bm3d_pkg

MSYS_PATH = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

BASE = Path(__file__).parent.parent
EXE = str(BASE / "standalone" / "rawdenoise.exe")
KODAK = BASE / "datasets" / "kodak"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

KAIR_DIR = Path(os.path.expanduser(r"~\KAIR"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inv_srgb(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(np.maximum(x, 0), 1/2.4) - 0.055)

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0

def ssim_4ch(a, b):
    vals = []
    for dy in range(2):
        for dx in range(2):
            vals.append(ssim(a[dy::2, dx::2], b[dy::2, dx::2], data_range=1.0))
    return float(np.mean(vals))

def demosaic_bilinear(bayer):
    h, w = bayer.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[0::2, 0::2, 0] = bayer[0::2, 0::2]
    rgb[0::2, 1::2, 1] = bayer[0::2, 1::2]
    rgb[1::2, 0::2, 1] = bayer[1::2, 0::2]
    rgb[1::2, 1::2, 2] = bayer[1::2, 1::2]
    for c in range(3):
        mask = np.zeros((h, w), dtype=np.float32)
        if c == 0: mask[0::2, 0::2] = 1
        elif c == 1: mask[0::2, 1::2] = 1; mask[1::2, 0::2] = 1
        else: mask[1::2, 1::2] = 1
        num = uniform_filter(rgb[:, :, c], size=3)
        den = uniform_filter(mask, size=3)
        rgb[:, :, c] = np.where(mask > 0, rgb[:, :, c], num / np.maximum(den, 1e-10))
    return np.clip(rgb, 0, 1)

_lpips_fn = None
def get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(DEVICE)
    return _lpips_fn

def compute_lpips(a, b):
    fn = get_lpips()
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE) * 2 - 1
    tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE) * 2 - 1
    with torch.no_grad():
        return fn(ta, tb).item()

def synth_noise(clean_raw, iso, rng):
    gain = iso / 100.0
    alpha = 0.001 * gain
    sigma_read = 0.002 * gain
    sigma_sq = sigma_read ** 2
    rate = np.maximum(clean_raw / max(alpha, 1e-10), 0.0)
    noisy = rng.poisson(rate).astype(np.float32) * alpha
    noisy += rng.standard_normal(clean_raw.shape).astype(np.float32) * sigma_read
    return np.clip(noisy, 0, 1).astype(np.float32), alpha, sigma_sq

def gat_forward(x, alpha, sigma_sq):
    return (2.0 / alpha) * np.sqrt(np.maximum(alpha * np.maximum(x, 0) + 0.375 * alpha**2 + sigma_sq, 0))

def gat_inverse(D, alpha, sigma_sq):
    D = np.maximum(D, 1e-8)
    D_inv = 1.0 / D
    y = 0.25*D*D + 0.25*1.2247448713916*D_inv - 11.0/8.0*D_inv**2 + 5.0/8.0*1.2247448713916*D_inv**3 - 1.0/8.0
    return np.maximum(alpha * y - sigma_sq / alpha, 0)


# ── C standalone methods ──

def method_c_standalone(noisy, w, h, method_name, alpha, sigma_sq, search_window=25):
    inp = str(BASE / "standalone" / "tmp_bench.bin")
    out = str(BASE / "standalone" / f"tmp_{method_name}.bin")
    noisy.tofile(inp)
    env = os.environ.copy()
    env["PATH"] = MSYS_PATH
    subprocess.run([EXE, inp, out, str(w), str(h), method_name,
                   "1.0", "1.0", "1.0", str(alpha), str(sigma_sq), "1", str(search_window)],
                  capture_output=True, timeout=300, env=env)
    return np.fromfile(out, dtype=np.float32).reshape(h, w)


def method_bm3d_cfa(noisy_bayer, alpha, sigma_sq):
    h, w = noisy_bayer.shape
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
    channels_gat = []
    for dy, dx in offsets:
        ch = noisy_bayer[dy::2, dx::2].copy()
        channels_gat.append(gat_forward(ch, alpha, sigma_sq).astype(np.float64))
    stack = np.stack(channels_gat, axis=2)

    from scipy.signal import convolve2d
    kernel = np.array([1, -2, 1], dtype=np.float64).reshape(1, 3)
    laps = convolve2d(channels_gat[0], kernel, mode='valid')
    sigma_est = np.median(np.abs(laps)) / 0.6745 / np.sqrt(6)
    sigma_est = max(sigma_est, 0.1)

    denoised_stack = bm3d_pkg.bm3d(stack, sigma_psd=sigma_est, profile='np')

    output = np.zeros_like(noisy_bayer)
    gat_max = gat_forward(1.0, alpha, sigma_sq) * 1.2
    for i, (dy, dx) in enumerate(offsets):
        ch = np.clip(denoised_stack[:, :, i], 0, gat_max)
        output[dy::2, dx::2] = np.minimum(gat_inverse(ch, alpha, sigma_sq), 1.0).astype(np.float32)
    return output


# ── Deep learning methods ──

_drunet_model = None
def get_drunet():
    global _drunet_model
    if _drunet_model is None:
        sys.path.insert(0, str(KAIR_DIR))
        from models.network_unet import UNetRes
        model = UNetRes(in_nc=2, out_nc=1, nc=[64,128,256,512], nb=4, act_mode='R', bias=False)
        state = torch.load(str(KAIR_DIR / "model_zoo" / "drunet_gray.pth"), map_location='cpu')
        model.load_state_dict(state, strict=True)
        model.eval().to(DEVICE)
        _drunet_model = model
        print(f"  [DRUNet loaded on {DEVICE}]")
    return _drunet_model

_dncnn_model = None
def get_dncnn():
    global _dncnn_model
    if _dncnn_model is None:
        sys.path.insert(0, str(KAIR_DIR))
        from models.network_dncnn import DnCNN
        model = DnCNN(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='R')
        state = torch.load(str(KAIR_DIR / "model_zoo" / "dncnn_gray_blind.pth"), map_location='cpu')
        model.load_state_dict(state, strict=True)
        model.eval().to(DEVICE)
        _dncnn_model = model
        print(f"  [DnCNN-B loaded on {DEVICE}]")
    return _dncnn_model


def _pad8(x):
    """Pad to multiple of 8 for UNet."""
    h, w = x.shape[-2:]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode='reflect')
    return x, h, w

def _dl_denoise_bayer(noisy_bayer, alpha, sigma_sq, model_fn, model_name):
    """GAT → split 4 Bayer channels → DL denoise each → reassemble → GAT inverse."""
    h, w = noisy_bayer.shape
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
    output = np.zeros_like(noisy_bayer)

    # GAT stabilized sigma ≈ 1.0 (theoretical for Anscombe-like transform)
    gat_sigma = 1.0
    gat_max = gat_forward(1.0, alpha, sigma_sq) * 1.2

    model = model_fn()

    for dy, dx in offsets:
        ch = noisy_bayer[dy::2, dx::2].copy()
        ch_gat = gat_forward(ch, alpha, sigma_sq).astype(np.float32)

        # Normalize to [0,1] range for DL model
        ch_min = 0.0
        ch_max = float(gat_max)
        ch_norm = np.clip(ch_gat / ch_max, 0, 1)

        # sigma in normalized scale
        sigma_norm = gat_sigma / ch_max

        with torch.no_grad():
            t_img = torch.from_numpy(ch_norm[np.newaxis, np.newaxis]).float().to(DEVICE)

            if model_name == "drunet":
                # DRUNet takes [img, noise_map] as 2-channel input
                noise_map = torch.full_like(t_img, sigma_norm)
                t_in = torch.cat([t_img, noise_map], dim=1)
            else:
                # DnCNN-B: single channel, blind
                t_in = t_img

            t_in_pad, oh, ow = _pad8(t_in)
            t_out = model(t_in_pad)
            t_out = t_out[:, :, :oh, :ow]
            den_norm = t_out.squeeze().cpu().numpy()

        # Denormalize back to GAT domain
        den_gat = np.clip(den_norm * ch_max, 0, gat_max)
        # GAT inverse
        den = np.minimum(gat_inverse(den_gat, alpha, sigma_sq), 1.0).astype(np.float32)
        output[dy::2, dx::2] = den

    return output


def method_drunet(noisy_bayer, alpha, sigma_sq):
    return _dl_denoise_bayer(noisy_bayer, alpha, sigma_sq, get_drunet, "drunet")

def method_dncnn(noisy_bayer, alpha, sigma_sq):
    return _dl_denoise_bayer(noisy_bayer, alpha, sigma_sq, get_dncnn, "dncnn")


# ── Main benchmark ──

def main():
    isos = [400, 800, 1600, 3200, 6400, 12800]
    images = sorted(KODAK.glob("kodim*.png"))[:24]
    n_images = len(images)

    if not images:
        print("ERROR: No Kodak images found")
        return

    methods = ["Noisy", "① BM3D-PC", "② BM3D-CFA", "③ DnCNN-B", "④ DRUNet", "⑥ Ours(V2)"]
    method_key_map = {
        "Noisy": "Noisy",
        "① BM3D-PC": "1_BM3D-PC",
        "② BM3D-CFA": "2_BM3D-CFA",
        "③ DnCNN-B": "3_DnCNN-B",
        "④ DRUNet": "4_DRUNet",
        "⑥ Ours(V2)": "6_OursV2",
    }

    print(f"V2 Multichannel benchmark: {n_images} images x {len(isos)} ISOs")
    print(f"Methods: BM3D-PC, BM3D-CFA, DnCNN-B, DRUNet, Ours(V2 multichannel s=25)")
    print(f"Device: {DEVICE}")
    print()

    all_results = []

    for iso in isos:
        t0 = time.time()
        gains = {m: {"psnr": [], "ssim": [], "lpips": []} for m in methods}

        for idx, img_path in enumerate(images):
            rng = np.random.default_rng(42)
            gain = iso / 100.0
            alpha = 0.001 * gain
            sigma_read = 0.002 * gain
            sigma_sq = sigma_read ** 2

            img = imread(str(img_path)).astype(np.float32) / 255.0
            h, w = img.shape[:2]; h -= h % 2; w -= w % 2; img = img[:h, :w]
            linear = inv_srgb(img).astype(np.float32)

            cr = np.zeros((h, w), dtype=np.float32)
            cr[0::2, 0::2] = linear[0::2, 0::2, 0]
            cr[0::2, 1::2] = linear[0::2, 1::2, 1]
            cr[1::2, 0::2] = linear[1::2, 0::2, 1]
            cr[1::2, 1::2] = linear[1::2, 1::2, 2]

            noisy, _, _ = synth_noise(cr, iso, rng)
            gt_srgb = np.clip(img, 0, 1).astype(np.float32)

            results_bayer = {}
            results_bayer["Noisy"] = noisy
            results_bayer["① BM3D-PC"] = method_c_standalone(noisy, w, h, "perchannel", alpha, sigma_sq)
            results_bayer["⑥ Ours(V2)"] = method_c_standalone(noisy, w, h, "ours", alpha, sigma_sq, search_window=25)

            try:
                results_bayer["② BM3D-CFA"] = method_bm3d_cfa(noisy, alpha, sigma_sq)
            except Exception as e:
                print(f"  BM3D-CFA failed: {e}")
                results_bayer["② BM3D-CFA"] = noisy

            try:
                results_bayer["③ DnCNN-B"] = method_dncnn(noisy, alpha, sigma_sq)
            except Exception as e:
                print(f"  DnCNN-B failed: {e}")
                results_bayer["③ DnCNN-B"] = noisy

            try:
                results_bayer["④ DRUNet"] = method_drunet(noisy, alpha, sigma_sq)
            except Exception as e:
                print(f"  DRUNet failed: {e}")
                results_bayer["④ DRUNet"] = noisy

            for m_name, den_bayer in results_bayer.items():
                p = psnr(cr, den_bayer)
                s = ssim_4ch(cr, den_bayer)
                den_srgb = np.clip(linear_to_srgb(demosaic_bilinear(den_bayer)), 0, 1).astype(np.float32)
                lp = compute_lpips(gt_srgb, den_srgb)
                gains[m_name]["psnr"].append(float(p))
                gains[m_name]["ssim"].append(float(s))
                gains[m_name]["lpips"].append(float(lp))

            if (idx + 1) % 6 == 0:
                print(f"  ISO {iso}: {idx+1}/{n_images}")

        elapsed = time.time() - t0
        avg = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in gains.items()}

        result = {"iso": iso}
        for m in methods:
            key = method_key_map[m]
            result[f"{key}_psnr"] = round(avg[m]["psnr"], 2)
            result[f"{key}_ssim"] = round(avg[m]["ssim"], 4)
            result[f"{key}_lpips"] = round(avg[m]["lpips"], 4)
        all_results.append(result)

        print(f"\nISO {iso:>5} ({elapsed:.0f}s):")
        print(f"  {'Method':<15} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
        print(f"  {'-'*40}")
        for m in methods:
            a = avg[m]
            print(f"  {m:<15} {a['psnr']:>7.2f} {a['ssim']:>7.4f} {a['lpips']:>7.4f}")

        # Delta vs BM3D-CFA
        dp = avg["⑥ Ours(V2)"]["psnr"] - avg["② BM3D-CFA"]["psnr"]
        dl = avg["⑥ Ours(V2)"]["lpips"] - avg["② BM3D-CFA"]["lpips"]
        mark = "OK" if dl < 0 else "NG"
        print(f"  Ours vs CFA: dPSNR={dp:+.2f}  dLPIPS={dl:+.4f} {mark}")
        print()

    # Summary
    print("=" * 110)
    print("SUMMARY (PSNR / LPIPS)")
    print(f"{'ISO':>6} {'BM3D-PC':>16} {'BM3D-CFA':>16} {'DnCNN-B':>16} {'DRUNet':>16} {'Ours(V2)':>16}  {'vs CFA':>8}")
    print("-" * 110)
    for r in all_results:
        pc_p = r["1_BM3D-PC_psnr"]; pc_l = r["1_BM3D-PC_lpips"]
        cfa_p = r["2_BM3D-CFA_psnr"]; cfa_l = r["2_BM3D-CFA_lpips"]
        dn_p = r["3_DnCNN-B_psnr"]; dn_l = r["3_DnCNN-B_lpips"]
        dr_p = r["4_DRUNet_psnr"]; dr_l = r["4_DRUNet_lpips"]
        ours_p = r["6_OursV2_psnr"]; ours_l = r["6_OursV2_lpips"]
        dl_cfa = ours_l - cfa_l
        print(f"{r['iso']:>6}  {pc_p:>6.2f}/{pc_l:.3f}  {cfa_p:>6.2f}/{cfa_l:.3f}  {dn_p:>6.2f}/{dn_l:.3f}  {dr_p:>6.2f}/{dr_l:.3f}  {ours_p:>6.2f}/{ours_l:.3f}  {dl_cfa:>+.4f}")

    # Best method per ISO
    print()
    print("BEST METHOD PER ISO:")
    for r in all_results:
        methods_data = {
            "BM3D-PC": (r["1_BM3D-PC_psnr"], r["1_BM3D-PC_lpips"]),
            "BM3D-CFA": (r["2_BM3D-CFA_psnr"], r["2_BM3D-CFA_lpips"]),
            "DnCNN-B": (r["3_DnCNN-B_psnr"], r["3_DnCNN-B_lpips"]),
            "DRUNet": (r["4_DRUNet_psnr"], r["4_DRUNet_lpips"]),
            "Ours(V2)": (r["6_OursV2_psnr"], r["6_OursV2_lpips"]),
        }
        best_psnr = max(methods_data.items(), key=lambda x: x[1][0])
        best_lpips = min(methods_data.items(), key=lambda x: x[1][1])
        print(f"  ISO {r['iso']:>5}: Best PSNR = {best_psnr[0]:<10} ({best_psnr[1][0]:.2f} dB)  Best LPIPS = {best_lpips[0]:<10} ({best_lpips[1][1]:.4f})")

    out_path = RESULTS / "v2_benchmark_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
