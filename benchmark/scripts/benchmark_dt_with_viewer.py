#!/usr/bin/env python3
"""
Full benchmark with all methods + HTML viewer.
Saves PNG images for every method + GT + Noisy.
Computes LPIPS / SSIM / Luma PSNR / Chroma PSNR from output images.

Methods: BM3D-PC, BM3D-CFA, DnCNN-B, DRUNet, Ours (darktable culmination)
Dataset: Kodak 24 × 6 ISOs (400-12800)
"""
import numpy as np
import subprocess
import os
import sys
import json
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from skimage.io import imread, imsave
from skimage.metrics import structural_similarity as ssim_fn
from skimage.color import rgb2lab
from scipy.ndimage import uniform_filter
import warnings
warnings.filterwarnings("ignore")

MSYS_PATH = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

BASE = Path(__file__).parent.parent
EXE_DT  = str(BASE / "standalone" / "rawdenoise_dt.exe")   # Ours (darktable culmination)
EXE_V2  = str(BASE / "standalone" / "rawdenoise.exe")       # BM3D-PC (per-channel)
KODAK   = BASE / "datasets" / "kodak"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

IMG_OUT = BASE / "viewer_images"
IMG_OUT.mkdir(exist_ok=True)

KAIR_DIR = Path(r"C:\Users\luxgrain\KAIR")

ISOS = [400, 800, 1600, 3200, 6400, 12800]
CONDITIONS = ["known", "blind"]
METHODS = ["BM3D-PC", "BM3D-CFA", "DnCNN-B", "DRUNet", "Ours"]
METHOD_LABELS = {
    "BM3D-PC":  "BM3D Per-Channel",
    "BM3D-CFA": "BM3D-CFA (Danielyan)",
    "DnCNN-B":  "DnCNN Blind",
    "DRUNet":   "DRUNet",
    "Ours":     "Ours (darktable)",
}

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

def bayer_from_srgb(img_u8):
    linear = inv_srgb(img_u8.astype(np.float32) / 255.0).astype(np.float32)
    h, w = linear.shape[:2]
    bayer = np.zeros((h, w), dtype=np.float32)
    bayer[0::2, 0::2] = linear[0::2, 0::2, 0]
    bayer[0::2, 1::2] = linear[0::2, 1::2, 1]
    bayer[1::2, 0::2] = linear[1::2, 0::2, 1]
    bayer[1::2, 1::2] = linear[1::2, 1::2, 2]
    return bayer

def bayer_to_srgb_f32(bayer):
    """Bayer float → sRGB float32 [0,1]"""
    return np.clip(linear_to_srgb(demosaic_bilinear(bayer)), 0, 1).astype(np.float32)

def save_srgb_png(srgb_f32, path):
    imsave(str(path), (np.clip(srgb_f32, 0, 1) * 255).astype(np.uint8))


# ── Metrics (computed from saved PNG for reproducibility) ──

def load_png_f32(path):
    """Load PNG as float32 [0,1]"""
    return imread(str(path)).astype(np.float32) / 255.0

def luma_psnr_from_srgb(gt_srgb, den_srgb):
    """Luma PSNR in linear domain (inverse sRGB → linear → grayscale)"""
    gt_lin = inv_srgb(gt_srgb).astype(np.float64)
    den_lin = inv_srgb(den_srgb).astype(np.float64)
    gt_y  = 0.2126 * gt_lin[:,:,0]  + 0.7152 * gt_lin[:,:,1]  + 0.0722 * gt_lin[:,:,2]
    den_y = 0.2126 * den_lin[:,:,0] + 0.7152 * den_lin[:,:,1] + 0.0722 * den_lin[:,:,2]
    mse = np.mean((gt_y - den_y) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0

def luma_psnr_bayer(gt_bayer, den_bayer):
    """Luma PSNR directly on Bayer (original method for backward compat)"""
    mse = np.mean((gt_bayer.astype(np.float64) - den_bayer.astype(np.float64)) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0

def chroma_psnr_srgb(gt_srgb, den_srgb):
    gt_lab = rgb2lab(np.clip(gt_srgb, 0, 1))
    den_lab = rgb2lab(np.clip(den_srgb, 0, 1))
    chroma_mse = np.mean((gt_lab[:,:,1:] - den_lab[:,:,1:]) ** 2)
    return 10 * np.log10(256.0**2 / max(chroma_mse, 1e-10))

def compute_ssim_srgb(gt_srgb, den_srgb):
    return ssim_fn(gt_srgb, den_srgb, channel_axis=2, data_range=1.0)

_lpips_fn = None
_device = None
def get_lpips():
    global _lpips_fn, _device
    if _lpips_fn is None:
        import torch
        import lpips
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(_device)
    return _lpips_fn, _device

def compute_lpips_srgb(a, b):
    import torch
    fn, device = get_lpips()
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float().to(device) * 2 - 1
    tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).float().to(device) * 2 - 1
    with torch.no_grad():
        return fn(ta, tb).item()

def compute_all_metrics_from_srgb(gt_srgb, den_srgb):
    """Compute all 4 metrics from sRGB float32 images."""
    lpips_val = compute_lpips_srgb(gt_srgb, den_srgb)
    ssim_val  = compute_ssim_srgb(gt_srgb, den_srgb)
    lpsnr     = luma_psnr_from_srgb(gt_srgb, den_srgb)
    cpsnr     = chroma_psnr_srgb(gt_srgb, den_srgb)
    return {
        "lpips": round(lpips_val, 4),
        "ssim":  round(ssim_val, 4),
        "lpsnr": round(lpsnr, 2),
        "cpsnr": round(cpsnr, 2),
    }


# ── GAT ──

def gat_forward(x, alpha, sigma_sq):
    return (2.0 / alpha) * np.sqrt(np.maximum(alpha * np.maximum(x, 0) + 0.375 * alpha**2 + sigma_sq, 0))

def gat_inverse(D, alpha, sigma_sq):
    D = np.maximum(D, 1e-8)
    D_inv = 1.0 / D
    y = 0.25*D*D + 0.25*1.2247448713916*D_inv - 11.0/8.0*D_inv**2 + 5.0/8.0*1.2247448713916*D_inv**3 - 1.0/8.0
    return np.maximum(alpha * y - sigma_sq / alpha, 0)


# ── C standalone runners ──

def run_c_exe(exe, noisy, w, h, method_name, alpha, sigma_sq, tag="tmp", extra_args=None):
    inp = str(BASE / "standalone" / f"tmp_{tag}.bin")
    out = str(BASE / "standalone" / f"tmp_{tag}_out.bin")
    noisy.tofile(inp)
    env = os.environ.copy()
    env["PATH"] = MSYS_PATH
    cmd = [exe, inp, out, str(w), str(h), method_name,
           "1.0", "1.0", "1.0", str(alpha), str(sigma_sq), "1", "16"]
    if extra_args:
        cmd = [exe, inp, out, str(w), str(h)] + extra_args
    result = subprocess.run(cmd, capture_output=True, timeout=600, env=env)
    stderr = result.stderr.decode('utf-8', errors='replace')
    den = np.fromfile(out, dtype=np.float32).reshape(h, w)
    return den, stderr

def run_dt_exe(noisy, w, h, tag="tmp", alpha=-1, sigma_sq=-1):
    """Run rawdenoise_dt.exe. alpha/sigma_sq > 0 = known, -1 = blind."""
    inp = str(BASE / "standalone" / f"tmp_dt_{tag}.bin")
    out = str(BASE / "standalone" / f"tmp_dt_{tag}_out.bin")
    noisy.tofile(inp)
    env = os.environ.copy()
    env["PATH"] = MSYS_PATH
    result = subprocess.run(
        [EXE_DT, inp, out, str(w), str(h), "ours",
         "1.0", "0.25", "1.5", str(alpha), str(sigma_sq), "1", "16"],
        capture_output=True, timeout=600, env=env
    )
    stderr = result.stderr.decode('utf-8', errors='replace')
    den = np.fromfile(out, dtype=np.float32).reshape(h, w)
    return den, stderr

def extract_estimated_params(stderr_text):
    m = re.search(r'alpha=([0-9.e+-]+)\s+sigma_sq=([0-9.e+-]+)', stderr_text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


# ── BM3D-CFA ──

def method_bm3d_cfa(noisy_bayer, alpha, sigma_sq):
    import bm3d as bm3d_pkg
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

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def dl_denoise_bayer(noisy_bayer, alpha, sigma_sq, model_fn, model_name):
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


# ── Utility: numpy → native Python for JSON ──

def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ── HTML generation ──

def generate_html(all_results, output_path):
    methods_js = json.dumps(METHODS)
    labels_js = json.dumps(METHOD_LABELS)
    results_js = json.dumps(_sanitize(all_results), indent=None)
    isos_js = json.dumps(ISOS)
    conditions_js = json.dumps(CONDITIONS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Denoiser Benchmark Viewer</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 6px; color: #e94560; font-size: 24px; }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 20px; font-size: 13px; }}
  .controls {{ text-align: center; margin: 15px 0; }}
  .controls select, .controls button {{
    padding: 8px 16px; margin: 4px; border-radius: 6px; border: 1px solid #444;
    background: #16213e; color: #eee; font-size: 14px; cursor: pointer;
  }}
  .controls select:hover, .controls button:hover {{ background: #0f3460; }}
  .controls button.active {{ background: #e94560; color: #fff; border-color: #e94560; }}
  table.summary {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 13px; }}
  table.summary th, table.summary td {{ padding: 6px 10px; text-align: center; border: 1px solid #333; }}
  table.summary th {{ background: #16213e; color: #e94560; font-size: 12px; }}
  table.summary tr:nth-child(even) {{ background: rgba(22,33,62,0.5); }}
  table.summary tr:hover {{ background: #0f3460; }}
  .best {{ color: #00ff88; font-weight: bold; }}
  .worse {{ color: #ff6b6b; }}
  .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 8px; margin: 10px 0; }}
  .image-card {{ background: #16213e; border-radius: 6px; padding: 8px; }}
  .image-card h3 {{ text-align: center; margin-bottom: 4px; font-size: 12px; color: #aaa; }}
  .image-card img {{ width: 100%; border-radius: 3px; cursor: zoom-in; }}
  .image-card .metrics {{ font-size: 10px; color: #888; text-align: center; margin-top: 3px; line-height: 1.5; }}
  .per-image {{ margin: 25px 0; border: 1px solid #333; border-radius: 8px; padding: 12px; background: #16213e; }}
  .per-image h2 {{ color: #e94560; margin-bottom: 8px; font-size: 16px; }}
  .est-info {{ font-size: 11px; color: #888; margin: 4px 0; }}
  .zoom-overlay {{
    display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.92); z-index: 1000; cursor: zoom-out;
    justify-content: center; align-items: center;
  }}
  .zoom-overlay.active {{ display: flex; }}
  .zoom-overlay img {{ max-width: 95%; max-height: 95%; object-fit: contain; }}
  .section-title {{ color: #e94560; text-align: center; margin: 30px 0 10px; font-size: 18px; }}
</style>
</head>
<body>
<h1>Denoiser Benchmark Viewer</h1>
<p class="subtitle">Known vs Blind | All methods | Kodak 24 | LPIPS / SSIM / Luma PSNR / Chroma PSNR</p>

<div class="controls">
  <label>Condition:
    <button id="btn-known" class="active" onclick="setCond('known')">Known</button>
    <button id="btn-blind" onclick="setCond('blind')">Blind</button>
    <button id="btn-both" onclick="setCond('both')">Both</button>
  </label>
  <label>ISO: <select id="iso-select" onchange="render()">
    <option value="all">All ISOs</option>
  </select></label>
  <label>Image: <select id="img-select" onchange="render()">
    <option value="all">All Images</option>
  </select></label>
</div>

<h2 class="section-title">Summary (ISO averages)</h2>
<div id="summary"></div>

<h2 class="section-title">Per-Image Results</h2>
<div id="per-image"></div>

<div class="zoom-overlay" id="zoom" onclick="this.classList.remove('active')">
  <img id="zoom-img" src="">
</div>

<script>
const METHODS = {methods_js};
const LABELS = {labels_js};
const ISOS = {isos_js};
const CONDS = {conditions_js};
const R = {results_js};
let curCond = 'known';

function setCond(c) {{
  curCond = c;
  document.querySelectorAll('.controls button').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-' + c).classList.add('active');
  render();
}}

const isoSel = document.getElementById('iso-select');
ISOS.forEach(iso => {{ const o = document.createElement('option'); o.value = iso; o.text = 'ISO ' + iso; isoSel.add(o); }});
const imgSel = document.getElementById('img-select');
[...new Set(R.map(r => r.image))].forEach(img => {{ const o = document.createElement('option'); o.value = img; o.text = img; imgSel.add(o); }});

function zoomImg(src) {{
  document.getElementById('zoom-img').src = src;
  document.getElementById('zoom').classList.add('active');
}}

function fmt(v, type) {{
  if (v === null || v === undefined) return '-';
  if (type === 'lpips' || type === 'ssim') return v.toFixed(4);
  return v.toFixed(2);
}}

function bestIdx(vals, lower) {{
  let best = null, bi = -1;
  vals.forEach((v, i) => {{
    if (v === null || v === undefined) return;
    if (best === null || (lower ? v < best : v > best)) {{ best = v; bi = i; }}
  }});
  return bi;
}}

function render() {{
  const isoF = isoSel.value;
  const imgF = imgSel.value;
  let filtered = R;
  if (curCond !== 'both') filtered = filtered.filter(r => r.condition === curCond);
  if (isoF !== 'all') filtered = filtered.filter(r => r.iso == +isoF);
  if (imgF !== 'all') filtered = filtered.filter(r => r.image === imgF);

  // Summary by ISO + condition
  const groups = {{}};
  filtered.forEach(r => {{
    const key = r.iso + '_' + r.condition;
    if (!groups[key]) groups[key] = {{ iso: r.iso, cond: r.condition, rows: [] }};
    groups[key].rows.push(r);
  }});

  let sh = '';
  Object.values(groups).sort((a,b) => a.iso - b.iso || a.cond.localeCompare(b.cond)).forEach(g => {{
    const {{ iso, cond, rows }} = g;
    const n = rows.length;
    const metrics = ['lpips', 'ssim', 'lpsnr', 'cpsnr'];
    const labels = ['LPIPS ↓', 'SSIM ↑', 'Luma PSNR ↑', 'Chroma PSNR ↑'];
    const lowerBetter = [true, false, false, false];
    const condLabel = curCond === 'both' ? ` [${{cond}}]` : '';

    sh += `<h3 style="color:#aaa;margin:20px 0 6px;">ISO ${{iso}}${{condLabel}} (${{n}} images)</h3>`;
    sh += '<table class="summary"><tr><th>Metric</th>';
    METHODS.forEach(m => {{ sh += `<th>${{LABELS[m]}}</th>`; }});
    sh += '</tr>';

    metrics.forEach((met, mi) => {{
      sh += `<tr><td>${{labels[mi]}}</td>`;
      const avgs = METHODS.map(m => {{
        const vals = rows.map(r => r[m + '_' + met]).filter(v => v !== null && v !== undefined);
        return vals.length > 0 ? vals.reduce((a,b) => a+b, 0) / vals.length : null;
      }});
      const bi = bestIdx(avgs, lowerBetter[mi]);
      avgs.forEach((v, i) => {{
        const cls = (i === bi) ? ' class="best"' : '';
        sh += `<td${{cls}}>${{fmt(v, met)}}</td>`;
      }});
      sh += '</tr>';
    }});
    sh += '</table>';
  }});
  document.getElementById('summary').innerHTML = sh;

  // Per-image
  let ph = '';
  filtered.forEach(r => {{
    const condTag = curCond === 'both' ? ` [${{r.condition}}]` : '';
    ph += `<div class="per-image">`;
    ph += `<h2>${{r.image}} — ISO ${{r.iso}}${{condTag}}</h2>`;
    if (r.condition === 'blind' && r.alpha_err_pct !== undefined) {{
      ph += `<p class="est-info">Estimation error: α=${{r.alpha_err_pct.toFixed(1)}}% σ²=${{r.sq_err_pct.toFixed(1)}}%</p>`;
    }}

    ph += '<table class="summary"><tr><th></th><th>LPIPS ↓</th><th>SSIM ↑</th><th>Luma PSNR ↑</th><th>Chroma PSNR ↑</th></tr>';
    ph += `<tr><td>Noisy</td><td>${{fmt(r.Noisy_lpips,'lpips')}}</td><td>${{fmt(r.Noisy_ssim,'ssim')}}</td><td>${{fmt(r.Noisy_lpsnr,'psnr')}}</td><td>${{fmt(r.Noisy_cpsnr,'psnr')}}</td></tr>`;

    const mets = ['lpips','ssim','lpsnr','cpsnr'];
    const lb = [true, false, false, false];
    const bests = {{}};
    mets.forEach((met, mi) => {{
      const vals = METHODS.map(m => r[m + '_' + met]);
      bests[met] = bestIdx(vals, lb[mi]);
    }});

    METHODS.forEach((m, mi) => {{
      ph += `<tr><td><b>${{LABELS[m]}}</b></td>`;
      mets.forEach((met) => {{
        const v = r[m + '_' + met];
        const cls = (bests[met] === mi) ? ' class="best"' : '';
        ph += `<td${{cls}}>${{fmt(v, met)}}</td>`;
      }});
      ph += '</tr>';
    }});
    ph += '</table>';

    ph += '<div class="image-grid">';
    if (r.img_gt) {{
      ph += `<div class="image-card"><h3>Ground Truth</h3><img src="${{r.img_gt}}" onclick="zoomImg(this.src)"></div>`;
    }}
    if (r.img_noisy) {{
      ph += `<div class="image-card"><h3>Noisy</h3><img src="${{r.img_noisy}}" onclick="zoomImg(this.src)">`;
      ph += `<div class="metrics">LPIPS:${{fmt(r.Noisy_lpips,'lpips')}} SSIM:${{fmt(r.Noisy_ssim,'ssim')}}</div></div>`;
    }}
    METHODS.forEach(m => {{
      const key = 'img_' + m.replace('-','_').replace(' ','_');
      if (r[key]) {{
        ph += `<div class="image-card"><h3>${{LABELS[m]}}</h3><img src="${{r[key]}}" onclick="zoomImg(this.src)">`;
        ph += `<div class="metrics">LPIPS:${{fmt(r[m+'_lpips'],'lpips')}} SSIM:${{fmt(r[m+'_ssim'],'ssim')}} L:${{fmt(r[m+'_lpsnr'],'psnr')}}dB C:${{fmt(r[m+'_cpsnr'],'psnr')}}dB</div></div>`;
      }}
    }});
    ph += '</div></div>';
  }});
  document.getElementById('per-image').innerHTML = ph;
}}

render();
</script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML viewer: {output_path}")


# ── Main ──

def main():
    images = sorted(KODAK.glob("kodim*.png"))[:24]
    n_images = len(images)

    if not images:
        print("ERROR: No Kodak images found"); return
    if not os.path.exists(EXE_DT):
        print(f"ERROR: {EXE_DT} not found"); return

    print(f"Full Benchmark: {len(METHODS)} methods × {len(ISOS)} ISOs × {n_images} images")
    print(f"Methods: {', '.join(METHODS)}")
    print(f"Output: {IMG_OUT}")
    print()

    # Load models
    print("Loading models...", flush=True)
    dl_ok = True
    try:
        get_drunet(); get_dncnn()
        print("  DL models OK")
    except Exception as e:
        print(f"  DL models failed: {e}")
        dl_ok = False

    try:
        get_lpips()
        print("  LPIPS OK")
    except Exception as e:
        print(f"  LPIPS failed: {e}"); return

    all_results = []
    total_t0 = time.time()

    for iso in ISOS:
        print(f"\n{'='*60}")
        print(f"  ISO {iso}")
        print(f"{'='*60}")

        gain = iso / 100.0
        true_alpha = 0.001 * gain
        true_sq = (0.002 * gain) ** 2

        for idx, img_path in enumerate(images):
            img_name = img_path.stem
            print(f"  {img_name} ({idx+1}/{n_images})", flush=True)

            # Read image → Bayer
            img_u8 = imread(str(img_path))
            h, w = img_u8.shape[:2]
            h -= h % 2; w -= w % 2
            img_u8 = img_u8[:h, :w]
            clean_raw = bayer_from_srgb(img_u8)

            # Synthesize noise (same seed for both conditions)
            rng = np.random.default_rng(42)
            noisy_raw, _, _ = synth_noise(clean_raw, iso, rng)

            # GT and Noisy sRGB
            gt_srgb = bayer_to_srgb_f32(clean_raw)
            noisy_srgb = bayer_to_srgb_f32(noisy_raw)

            for cond in CONDITIONS:
                is_known = (cond == "known")
                cond_dir = IMG_OUT / cond / f"ISO{iso}" / img_name
                cond_dir.mkdir(parents=True, exist_ok=True)

                # Save GT and Noisy (shared, but save in both dirs for self-contained HTML)
                save_srgb_png(gt_srgb, cond_dir / "gt.png")
                save_srgb_png(noisy_srgb, cond_dir / "noisy.png")

                # Choose noise params for this condition
                use_alpha = true_alpha if is_known else -1
                use_sq    = true_sq    if is_known else -1

                den_bayers = {}

                # 1. Ours (darktable culmination)
                t0 = time.time()
                try:
                    den_ours, stderr_ours = run_dt_exe(noisy_raw, w, h,
                        tag=f"{cond}_{img_name}_{iso}", alpha=use_alpha, sigma_sq=use_sq)
                    if is_known:
                        est_alpha, est_sq = true_alpha, true_sq
                    else:
                        est_alpha, est_sq = extract_estimated_params(stderr_ours)
                        if est_alpha is None:
                            est_alpha, est_sq = true_alpha, true_sq
                    den_bayers["Ours"] = den_ours
                except Exception as e:
                    print(f"    Ours {cond} FAILED: {e}")
                    den_bayers["Ours"] = noisy_raw
                    est_alpha, est_sq = true_alpha, true_sq
                ours_time = time.time() - t0

                # For known: all methods get true params
                # For blind: all methods get estimated params from Ours
                m_alpha = true_alpha if is_known else est_alpha
                m_sq    = true_sq    if is_known else est_sq

                # 2. BM3D-PC
                try:
                    den_pc, _ = run_c_exe(EXE_V2, noisy_raw, w, h, "perchannel",
                                           m_alpha, m_sq, tag=f"pc_{cond}_{img_name}_{iso}")
                    den_bayers["BM3D-PC"] = den_pc
                except Exception as e:
                    print(f"    PC {cond} fail: {e}")
                    den_bayers["BM3D-PC"] = noisy_raw

                # 3. BM3D-CFA
                try:
                    den_bayers["BM3D-CFA"] = method_bm3d_cfa(noisy_raw, m_alpha, m_sq)
                except Exception as e:
                    print(f"    CFA {cond} fail: {e}")
                    den_bayers["BM3D-CFA"] = noisy_raw

                # 4. DnCNN-B
                if dl_ok:
                    try:
                        den_bayers["DnCNN-B"] = dl_denoise_bayer(noisy_raw, m_alpha, m_sq, get_dncnn, "dncnn")
                    except:
                        den_bayers["DnCNN-B"] = noisy_raw
                else:
                    den_bayers["DnCNN-B"] = noisy_raw

                # 5. DRUNet
                if dl_ok:
                    try:
                        den_bayers["DRUNet"] = dl_denoise_bayer(noisy_raw, m_alpha, m_sq, get_drunet, "drunet")
                    except:
                        den_bayers["DRUNet"] = noisy_raw
                else:
                    den_bayers["DRUNet"] = noisy_raw

                # ── Save all method outputs as sRGB PNG ──
                method_srgbs = {}
                method_fnames = {
                    "BM3D-PC":  "bm3d_pc.png",
                    "BM3D-CFA": "bm3d_cfa.png",
                    "DnCNN-B":  "dncnn_b.png",
                    "DRUNet":   "drunet.png",
                    "Ours":     "ours_dt.png",
                }
                for m in METHODS:
                    srgb = bayer_to_srgb_f32(den_bayers[m])
                    save_srgb_png(srgb, cond_dir / method_fnames[m])
                    method_srgbs[m] = srgb

                # ── Compute metrics ──
                result = {
                    "image": img_name,
                    "iso": iso,
                    "condition": cond,
                    "true_alpha": round(true_alpha, 8),
                    "true_sigma_sq": round(true_sq, 10),
                    "used_alpha": round(m_alpha, 8),
                    "used_sigma_sq": round(m_sq, 10),
                }
                if not is_known:
                    result["est_alpha"] = round(est_alpha, 8)
                    result["est_sigma_sq"] = round(est_sq, 10)
                    result["alpha_err_pct"] = round(100 * abs(est_alpha - true_alpha) / max(true_alpha, 1e-10), 1)
                    result["sq_err_pct"] = round(100 * abs(est_sq - true_sq) / max(true_sq, 1e-10), 1)

                # Noisy metrics
                noisy_met = compute_all_metrics_from_srgb(gt_srgb, noisy_srgb)
                for k, v in noisy_met.items():
                    result[f"Noisy_{k}"] = v

                # Method metrics
                for m in METHODS:
                    met = compute_all_metrics_from_srgb(gt_srgb, method_srgbs[m])
                    for k, v in met.items():
                        result[f"{m}_{k}"] = v

                # Image paths (relative for HTML)
                rel = f"viewer_images/{cond}/ISO{iso}/{img_name}"
                result["img_gt"]    = f"{rel}/gt.png"
                result["img_noisy"] = f"{rel}/noisy.png"
                for m in METHODS:
                    key = "img_" + m.replace("-", "_").replace(" ", "_")
                    result[key] = f"{rel}/{method_fnames[m]}"

                result["ours_time"] = round(ours_time, 2)
                all_results.append(result)

                # Print summary line
                ours_lpips = result.get("Ours_lpips", 0)
                ours_ssim  = result.get("Ours_ssim", 0)
                ours_lpsnr = result.get("Ours_lpsnr", 0)
                print(f"    [{cond:5s}] LPIPS={ours_lpips:.4f} SSIM={ours_ssim:.4f} L={ours_lpsnr:.2f}dB ({ours_time:.1f}s)")

        # ISO summary
        for cond in CONDITIONS:
            iso_rows = [r for r in all_results if r["iso"] == iso and r["condition"] == cond]
            if iso_rows:
                print(f"\n  ISO {iso} [{cond}] averages:")
                for m in METHODS:
                    avg_lpips = np.mean([r[f"{m}_lpips"] for r in iso_rows if r.get(f"{m}_lpips") is not None])
                    avg_ssim  = np.mean([r[f"{m}_ssim"]  for r in iso_rows])
                    avg_lpsnr = np.mean([r[f"{m}_lpsnr"] for r in iso_rows])
                    avg_cpsnr = np.mean([r[f"{m}_cpsnr"] for r in iso_rows])
                    print(f"    {m:12s}: LPIPS={avg_lpips:.4f} SSIM={avg_ssim:.4f} L={avg_lpsnr:.2f} C={avg_cpsnr:.2f}")

    total_elapsed = time.time() - total_t0
    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # Save JSON
    json_path = RESULTS / "dt_full_benchmark.json"

    json_path = RESULTS / "dt_full_benchmark.json"
    with open(str(json_path), 'w') as f:
        json.dump(_sanitize(all_results), f, indent=2)
    print(f"Results: {json_path}")

    # Generate HTML
    html_path = BASE / "benchmark_viewer.html"
    generate_html(all_results, str(html_path))

    print(f"\nOpen in browser: {html_path}")


if __name__ == "__main__":
    main()
