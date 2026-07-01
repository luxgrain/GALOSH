"""Fairness probe: feed BM3D/Color-NLM the noise sigma from 3 sources
(skimage estimate_sigma [current blind] / GALOSH Laplacian-MAD [GALOSH's blind estimator] /
TRUE std(noisy-gt) [oracle upper bound]) on a high-ISO RawNIND subset, to see whether the
classical methods are handicapped by sigma under-estimation (correlated sRGB noise) rather than
by the denoising algorithm. GALOSH (training-free) shown for reference on the same scenes."""
import sys, json, numpy as np, cv2, bm3d
from pathlib import Path
from PIL import Image
from skimage.restoration import estimate_sigma
sys.path.insert(0, "benchmark/scripts")
import bench_yuv_srgb as B

ND = Path("E:/rawnind_bench/__noisy_raw_render__"); GD = Path("E:/rawnind_bench/__gt_raw_render__")
R = Path("benchmark/rawnind_srgb_results"); r = json.load(open(R / "_metrics.json"))
gal = {x['tag']: x for x in r if x['method'] == 'galosh_yuv_gpu_fp32' and x.get('ok')}

def iso(t):
    try: return int(t.split("__ISO")[1])
    except: return 0

def galosh_mad(img):
    s = []
    for c in range(3):
        p = img[:, :, c]; lap = np.abs(p[:, :-4] - 2 * p[:, 2:-2] + p[:, 4:]); s.append(np.median(lap) / 1.6521)
    return float(np.mean(s))

hi = sorted([t for t in gal if iso(t) >= 6400])
hi = hi[::max(1, len(hi) // 30)][:30]
keys = ['noisy', 'galosh', 'cbm3d_sk', 'cbm3d_gal', 'cbm3d_true', 'nlm_sk', 'nlm_gal', 'nlm_true']
acc = {k: {'psnr': [], 'lpips': []} for k in keys}
for j, t in enumerate(hi):
    sc = t.split("__ISO")[0]
    n = np.asarray(Image.open(ND / f"{t}.png").convert("RGB"), np.float32) / 255
    g = np.asarray(Image.open(GD / f"{sc}.png").convert("RGB"), np.float32) / 255
    sk = float(estimate_sigma(n, channel_axis=-1, average_sigmas=True)); gm = galosh_mad(n); tr = float(np.std(n - g))
    acc['noisy']['psnr'].append(B.psnr(n, g)); acc['noisy']['lpips'].append(B.lpips_m(n, g))
    acc['galosh']['psnr'].append(gal[t]['psnr']); acc['galosh']['lpips'].append(gal[t]['lpips'])
    u8 = (np.clip(n, 0, 1) * 255).astype(np.uint8)
    for nm, s in [('sk', sk), ('gal', gm), ('true', tr)]:
        b = np.clip(bm3d.bm3d_rgb(n.astype(np.float64), sigma_psd=s), 0, 1).astype(np.float32)
        acc[f'cbm3d_{nm}']['psnr'].append(B.psnr(b, g)); acc[f'cbm3d_{nm}']['lpips'].append(B.lpips_m(b, g))
        d = cv2.fastNlMeansDenoisingColored(u8, None, s * 255, s * 255, 7, 21).astype(np.float32) / 255
        acc[f'nlm_{nm}']['psnr'].append(B.psnr(d, g)); acc[f'nlm_{nm}']['lpips'].append(B.lpips_m(d, g))
    print(f"  [{j+1}/{len(hi)}] {t[:30]:<30} sk{sk:.3f} gal{gm:.3f} true{tr:.3f}", flush=True)

out = {k: {'psnr': float(np.mean(v['psnr'])), 'lpips': float(np.mean(v['lpips'])), 'n': len(v['psnr'])} for k, v in acc.items()}
json.dump(out, open(R / "_sigma_source_exp.json", "w"), indent=1)
print("\n=== high-ISO (ISO>=6400, n={}) mean: sigma-source effect ===".format(len(hi)))
print(f"  {'method':<16}{'PSNR':>8}{'LPIPS':>8}")
for k in keys:
    print(f"  {k:<16}{out[k]['psnr']:>8.2f}{out[k]['lpips']:>8.3f}")
