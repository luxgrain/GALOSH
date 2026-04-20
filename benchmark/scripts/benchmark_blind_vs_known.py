#!/usr/bin/env python3
"""
Benchmark: 5 methods × 2 conditions (known GAT vs blind GAT)
Methods: BM3D-PC, BM3D-CFA, DnCNN-B, DRUNet, Ours
Metrics: Luma PSNR, Chroma PSNR, LPIPS

Optimization: Known-condition results loaded from v2_benchmark_results.json.
Only blind condition is computed fresh.
V3 blind estimation runs ONCE per image → estimated params shared by all methods.
"""
import numpy as np
import subprocess
import os
import sys
import json
import time
import re
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from skimage.io import imread
from skimage.color import rgb2lab
from scipy.ndimage import uniform_filter
import bm3d as bm3d_pkg

MSYS_PATH = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

BASE = Path(__file__).parent.parent
EXE_V2 = str(BASE / "standalone" / "rawdenoise.exe")
EXE_V3 = str(BASE / "standalone" / "rawdenoise_v3.exe")
KODAK = BASE / "datasets" / "kodak"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

KAIR_DIR = Path(r"C:\Users\luxgrain\KAIR")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Utilities ──

def inv_srgb(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(np.maximum(x, 0), 1/2.4) - 0.055)

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

def synth_noise(clean_raw, iso, rng):
    gain = iso / 100.0
    alpha = 0.001 * gain
    sigma_read = 0.002 * gain
    sigma_sq = sigma_read ** 2
    rate = np.maximum(clean_raw / max(alpha, 1e-10), 0.0)
    noisy = rng.poisson(rate).astype(np.float32) * alpha
    noisy += rng.standard_normal(clean_raw.shape).astype(np.float32) * sigma_read
    return np.clip(noisy, 0, 1).astype(np.float32), alpha, sigma_sq


# ── Metrics ──

def luma_psnr(gt_bayer, den_bayer):
    mse = np.mean((gt_bayer.astype(np.float64) - den_bayer.astype(np.float64)) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0

def chroma_psnr(gt_srgb, den_srgb):
    gt_lab = rgb2lab(np.clip(gt_srgb, 0, 1))
    den_lab = rgb2lab(np.clip(den_srgb, 0, 1))
    chroma_mse = np.mean((gt_lab[:,:,1:] - den_lab[:,:,1:]) ** 2)
    return 10 * np.log10(256.0**2 / max(chroma_mse, 1e-10))

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

def compute_all_metrics(gt_bayer, gt_srgb, den_bayer):
    lp = luma_psnr(gt_bayer, den_bayer)
    den_srgb = np.clip(linear_to_srgb(demosaic_bilinear(den_bayer)), 0, 1).astype(np.float32)
    cp = chroma_psnr(gt_srgb, den_srgb)
    lpips_val = compute_lpips(gt_srgb, den_srgb)
    return {"lpsnr": lp, "cpsnr": cp, "lpips": lpips_val}


# ── GAT ──

def gat_forward(x, alpha, sigma_sq):
    return (2.0 / alpha) * np.sqrt(np.maximum(alpha * np.maximum(x, 0) + 0.375 * alpha**2 + sigma_sq, 0))

def gat_inverse(D, alpha, sigma_sq):
    D = np.maximum(D, 1e-8)
    D_inv = 1.0 / D
    y = 0.25*D*D + 0.25*1.2247448713916*D_inv - 11.0/8.0*D_inv**2 + 5.0/8.0*1.2247448713916*D_inv**3 - 1.0/8.0
    return np.maximum(alpha * y - sigma_sq / alpha, 0)


# ── C standalone ──

def run_c_exe(exe, noisy, w, h, method_name, alpha, sigma_sq, tag="tmp", search_window=25):
    inp = str(BASE / "standalone" / f"tmp_{tag}.bin")
    out = str(BASE / "standalone" / f"tmp_{tag}_out.bin")
    noisy.tofile(inp)
    env = os.environ.copy()
    env["PATH"] = MSYS_PATH
    result = subprocess.run(
        [exe, inp, out, str(w), str(h), method_name,
         "1.0", "1.0", "1.0", str(alpha), str(sigma_sq), "1", str(search_window)],
        capture_output=True, timeout=300, env=env
    )
    stderr = result.stderr.decode('utf-8', errors='replace')
    den = np.fromfile(out, dtype=np.float32).reshape(h, w)
    return den, stderr

def extract_estimated_params(stderr_text):
    m = re.search(r'Auto-estimated:\s+alpha=([0-9.e+-]+)\s+sigma_sq=([0-9.e+-]+)', stderr_text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


# ── BM3D-CFA ──

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


# ── Deep learning ──

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
    return _dncnn_model

def _pad8(x):
    h, w = x.shape[-2:]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode='reflect')
    return x, h, w

def _dl_denoise_bayer(noisy_bayer, alpha, sigma_sq, model_fn, model_name):
    h, w = noisy_bayer.shape
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
    output = np.zeros_like(noisy_bayer)
    gat_sigma = 1.0
    gat_max = gat_forward(1.0, alpha, sigma_sq) * 1.2
    model = model_fn()

    for dy, dx in offsets:
        ch = noisy_bayer[dy::2, dx::2].copy()
        ch_gat = gat_forward(ch, alpha, sigma_sq).astype(np.float32)
        ch_max = float(gat_max)
        ch_norm = np.clip(ch_gat / ch_max, 0, 1)
        sigma_norm = gat_sigma / ch_max

        with torch.no_grad():
            t_img = torch.from_numpy(ch_norm[np.newaxis, np.newaxis]).float().to(DEVICE)
            if model_name == "drunet":
                noise_map = torch.full_like(t_img, sigma_norm)
                t_in = torch.cat([t_img, noise_map], dim=1)
            else:
                t_in = t_img
            t_in_pad, oh, ow = _pad8(t_in)
            t_out = model(t_in_pad)
            t_out = t_out[:, :, :oh, :ow]
            den_norm = t_out.squeeze().cpu().numpy()

        den_gat = np.clip(den_norm * ch_max, 0, gat_max)
        den = np.minimum(gat_inverse(den_gat, alpha, sigma_sq), 1.0).astype(np.float32)
        output[dy::2, dx::2] = den
    return output


# ── Main ──

def main():
    isos = [400, 800, 1600, 3200, 6400, 12800]
    images = sorted(KODAK.glob("kodim*.png"))[:24]
    n_images = len(images)

    if not images:
        print("ERROR: No Kodak images found")
        return

    # Load existing known-condition results from v2_benchmark
    v2_path = RESULTS / "v2_benchmark_results.json"
    known_data = {}
    if v2_path.exists():
        with open(v2_path) as f:
            v2_results = json.load(f)
        for r in v2_results:
            known_data[r["iso"]] = r
        print(f"Loaded known-condition results from {v2_path}")
    else:
        print("WARNING: v2_benchmark_results.json not found, will compute known too")

    # Key mapping: v2 results use different naming
    v2_key_map = {
        "BM3D-PC":  "1_BM3D-PC",
        "BM3D-CFA": "2_BM3D-CFA",
        "DnCNN-B":  "3_DnCNN-B",
        "DRUNet":   "4_DRUNet",
        "Ours":     "6_OursV2",
    }

    print(f"Benchmark: Blind GAT estimation × 5 methods")
    print(f"Images: {n_images}, ISOs: {isos}")
    print(f"Known results: from v2_benchmark (pre-computed)")
    print(f"Blind results: fresh computation with V3 estimator")
    print(f"Device: {DEVICE}")
    print()

    # Preload DL models
    print("Loading DL models...", flush=True)
    dl_ok = True
    try:
        get_drunet()
        get_dncnn()
        print("  DL models loaded OK")
    except Exception as e:
        print(f"  DL model load failed: {e}")
        dl_ok = False

    get_lpips()
    print("  LPIPS loaded OK", flush=True)

    methods = ["BM3D-PC", "BM3D-CFA", "DnCNN-B", "DRUNet", "Ours"]
    all_results = []

    for iso in isos:
        t0 = time.time()
        gain = iso / 100.0
        true_alpha = 0.001 * gain
        true_sq = (0.002 * gain) ** 2

        blind_metrics = {m: {"lpsnr": [], "cpsnr": [], "lpips": []} for m in methods}
        est_errors = {"alpha_err": [], "sq_err": []}

        for idx, img_path in enumerate(images):
            rng = np.random.default_rng(42)
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

            # Phase 1: V3 blind estimation ONCE → get params + Ours blind output
            den_ours_blind, stderr = run_c_exe(
                EXE_V3, noisy, w, h, "ours", -1, -1, tag=f"v3b_{idx}")
            est_alpha, est_sq = extract_estimated_params(stderr)
            if est_alpha is None:
                print(f"  WARN: parse failed {img_path.name} ISO {iso}")
                est_alpha, est_sq = true_alpha, true_sq

            est_errors["alpha_err"].append(abs(est_alpha - true_alpha) / true_alpha * 100)
            est_errors["sq_err"].append(abs(est_sq - true_sq) / true_sq * 100)

            # Phase 2: Run blind methods
            results_blind = {}
            results_blind["Ours"] = den_ours_blind  # Reuse!

            # C exes parallel
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_pc = pool.submit(run_c_exe, EXE_V2, noisy, w, h, "perchannel",
                                   est_alpha, est_sq, f"pcb_{idx}")
                results_blind["BM3D-PC"] = f_pc.result()[0]

            # BM3D-CFA blind (sequential)
            try:
                results_blind["BM3D-CFA"] = method_bm3d_cfa(noisy, est_alpha, est_sq)
            except Exception as e:
                print(f"  BM3D-CFA blind failed: {e}")
                results_blind["BM3D-CFA"] = noisy

            # DL blind
            if dl_ok:
                try:
                    results_blind["DnCNN-B"] = _dl_denoise_bayer(
                        noisy, est_alpha, est_sq, get_dncnn, "dncnn")
                    results_blind["DRUNet"] = _dl_denoise_bayer(
                        noisy, est_alpha, est_sq, get_drunet, "drunet")
                except Exception as e:
                    print(f"  DL blind failed: {e}")
                    results_blind.setdefault("DnCNN-B", noisy)
                    results_blind.setdefault("DRUNet", noisy)
            else:
                results_blind["DnCNN-B"] = noisy
                results_blind["DRUNet"] = noisy

            # Phase 3: Metrics for blind
            for m in methods:
                met = compute_all_metrics(cr, gt_srgb, results_blind[m])
                blind_metrics[m]["lpsnr"].append(met["lpsnr"])
                blind_metrics[m]["cpsnr"].append(met["cpsnr"])
                blind_metrics[m]["lpips"].append(met["lpips"])

            if (idx + 1) % 4 == 0 or idx == 0:
                print(f"  ISO {iso}: {idx+1}/{n_images} done", flush=True)

        elapsed = time.time() - t0

        # Average blind metrics
        avg_blind = {m: {k: float(np.mean(v)) for k, v in blind_metrics[m].items()}
                     for m in methods}
        mean_alpha_err = float(np.mean(est_errors["alpha_err"]))
        mean_sq_err = float(np.mean(est_errors["sq_err"]))

        # Get known metrics from v2 results (PSNR = Luma PSNR in v2)
        # v2 has PSNR and LPIPS but not Chroma PSNR - we'll use PSNR as Luma
        avg_known = {}
        if iso in known_data:
            kd = known_data[iso]
            for m in methods:
                vk = v2_key_map[m]
                avg_known[m] = {
                    "lpsnr": kd[f"{vk}_psnr"],
                    "lpips": kd[f"{vk}_lpips"],
                    "cpsnr": None,  # Not in v2 results
                }
        else:
            for m in methods:
                avg_known[m] = {"lpsnr": None, "lpips": None, "cpsnr": None}

        # Print table
        print(f"\nISO {iso:>5} ({elapsed:.0f}s) | Est error: alpha={mean_alpha_err:.1f}% sq={mean_sq_err:.1f}%")
        print(f"  {'Method':<12} | {'Known(v2)':^18} | {'Blind(new)':^27} | {'Delta':^18}")
        print(f"  {'':12} | {'Luma':>7} {'LPIPS':>7} | {'Luma':>7} {'Chroma':>7} {'LPIPS':>7} | {'dLuma':>6} {'dLPIPS':>7}")
        print(f"  {'-'*88}")
        for m in methods:
            k = avg_known[m]; b = avg_blind[m]
            kl = f"{k['lpsnr']:>7.2f}" if k['lpsnr'] else "    N/A"
            klp = f"{k['lpips']:>7.4f}" if k['lpips'] else "    N/A"
            dl = f"{b['lpsnr'] - k['lpsnr']:>+6.2f}" if k['lpsnr'] else "   N/A"
            dlp = f"{b['lpips'] - k['lpips']:>+7.4f}" if k['lpips'] else "    N/A"
            print(f"  {m:<12} | {kl} {klp} | {b['lpsnr']:>7.2f} {b['cpsnr']:>7.2f} {b['lpips']:>7.4f} | {dl} {dlp}")
        print(flush=True)

        # Store
        result = {"iso": iso, "alpha_err_pct": round(mean_alpha_err, 1),
                  "sq_err_pct": round(mean_sq_err, 1)}
        for m in methods:
            mk = m.replace("-", "").replace(" ", "")
            if avg_known[m]["lpsnr"]:
                result[f"{mk}_known_lpsnr"] = avg_known[m]["lpsnr"]
                result[f"{mk}_known_lpips"] = avg_known[m]["lpips"]
            result[f"{mk}_blind_lpsnr"] = round(avg_blind[m]["lpsnr"], 2)
            result[f"{mk}_blind_cpsnr"] = round(avg_blind[m]["cpsnr"], 2)
            result[f"{mk}_blind_lpips"] = round(avg_blind[m]["lpips"], 4)
        all_results.append(result)

    # ── Final Summary ──
    print("\n" + "=" * 130)
    print("FINAL SUMMARY: Known (v2) vs Blind (V3 estimator)")
    print()

    # Table 1: Known PSNR/LPIPS
    print("Known GAT (from v2_benchmark, Luma PSNR / LPIPS):")
    print(f"{'ISO':>6}", end="")
    for m in methods:
        print(f"  {m:>16}", end="")
    print()
    print("-" * 100)
    for r in all_results:
        print(f"{r['iso']:>6}", end="")
        for m in methods:
            mk = m.replace("-", "").replace(" ", "")
            lp = r.get(f"{mk}_known_lpsnr")
            li = r.get(f"{mk}_known_lpips")
            if lp and li:
                print(f"  {lp:>6.2f}/{li:.3f}", end="")
            else:
                print(f"       N/A      ", end="")
        print()

    # Table 2: Blind PSNR/ChromaPSNR/LPIPS
    print()
    print("Blind GAT (V3 estimator, Luma / Chroma / LPIPS):")
    print(f"{'ISO':>6} {'aErr':>5}", end="")
    for m in methods:
        print(f"  {m:>22}", end="")
    print()
    print("-" * 130)
    for r in all_results:
        print(f"{r['iso']:>6} {r['alpha_err_pct']:>4.0f}%", end="")
        for m in methods:
            mk = m.replace("-", "").replace(" ", "")
            lp = r[f"{mk}_blind_lpsnr"]; cp = r[f"{mk}_blind_cpsnr"]; li = r[f"{mk}_blind_lpips"]
            print(f"  {lp:>5.1f}/{cp:>5.1f}/{li:.3f}", end="")
        print()

    # Table 3: Delta (Blind - Known) Luma PSNR
    print()
    print("Delta Luma PSNR (Blind - Known):")
    print(f"{'ISO':>6}", end="")
    for m in methods:
        print(f"  {m:>10}", end="")
    print()
    print("-" * 70)
    for r in all_results:
        print(f"{r['iso']:>6}", end="")
        for m in methods:
            mk = m.replace("-", "").replace(" ", "")
            kl = r.get(f"{mk}_known_lpsnr")
            bl = r[f"{mk}_blind_lpsnr"]
            if kl:
                print(f"  {bl - kl:>+10.2f}", end="")
            else:
                print(f"       N/A", end="")
        print()

    out_path = RESULTS / "blind_vs_known_benchmark.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
