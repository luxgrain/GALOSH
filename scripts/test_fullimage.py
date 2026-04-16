#!/usr/bin/env python3
"""
Quick full-image pre-demosaic test on SIDD Medium cropped data.
Tests GALOSH GPU, BM3D-CFA, NLM-CFA on a single scene.

Usage:
  python test_fullimage.py                      # default: S6 GRBG
  python test_fullimage.py --scene 0138_IP_RGGB # pick scene
"""
import numpy as np
import subprocess, os, sys, time, argparse, tempfile, struct
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")

BASE   = Path(__file__).parent.parent
CROP   = BASE / "datasets" / "sidd" / "test_crop"
OUTDIR = BASE / "comparison_images_sidd" / "fullimage_test"
OUTDIR.mkdir(parents=True, exist_ok=True)

GPU_EXE = BASE / "standalone" / "rawdenoise_gpu.exe"

# ── Pipeline helpers ──────────────────────────────────────────────

def linear_to_srgb(x):
    return np.where(x <= 0.0031308,
                    12.92 * x,
                    1.055 * np.power(np.maximum(x, 0.0), 1.0/2.4) - 0.055)

def srgb_to_linear(x):
    return np.where(x <= 0.04045,
                    x / 12.92,
                    np.power((np.maximum(x, 0.0) + 0.055) / 1.055, 2.4))

def demosaic_menon(bayer):
    """Menon (2007) demosaic — input must be RGGB."""
    from colour_demosaicing import demosaicing_CFA_Bayer_DDFAPD
    rgb = demosaicing_CFA_Bayer_DDFAPD(bayer.astype(np.float64), pattern='RGGB')
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)

def estimate_affine(gt_raw, gt_srgb):
    """Estimate 3×4 affine from paired GT: Menon(gt_raw) linear → inv_gamma(gt_srgb) linear."""
    print("  Estimating affine WB×CCM from paired GT...", end=" ", flush=True)
    rgb_menon = demosaic_menon(gt_raw)
    rgb_isp_lin = srgb_to_linear(gt_srgb.astype(np.float64))
    # Subsample for speed (full image is huge)
    step = 4
    A = rgb_menon[::step, ::step].reshape(-1, 3).astype(np.float64)
    B = rgb_isp_lin[::step, ::step].reshape(-1, 3).astype(np.float64)
    A_aug = np.hstack([A, np.ones((A.shape[0], 1))])
    M, _, _, _ = np.linalg.lstsq(A_aug, B, rcond=None)
    affine = M.T  # (3, 4)
    print(f"diag=[{affine[0,0]:.3f}, {affine[1,1]:.3f}, {affine[2,2]:.3f}]")
    return affine

def raw_to_srgb_calibrated(bayer, affine):
    """Bayer → Menon → affine → sRGB gamma."""
    rgb_lin = demosaic_menon(bayer).astype(np.float64)
    h, w = rgb_lin.shape[:2]
    flat = rgb_lin.reshape(-1, 3)
    out = flat @ affine[:, :3].T + affine[:, 3]
    out = np.clip(out.reshape(h, w, 3), 0.0, 1.0)
    return np.clip(linear_to_srgb(out), 0.0, 1.0).astype(np.float32)

# ── Metrics ───────────────────────────────────────────────────────

def psnr(ref, test):
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse < 1e-15: return 99.0
    return float(10 * np.log10(1.0 / mse))

def ssim_rgb(ref, test):
    return float(ssim(ref, test, channel_axis=2, data_range=1.0))

# ── Denoisers ─────────────────────────────────────────────────────

def sigma_mad_raw(noisy):
    """Blind sigma via wavelet MAD."""
    import pywt
    _, (_, _, hh) = pywt.dwt2(noisy.astype(np.float64), 'haar')
    return float(np.median(np.abs(hh)) / 0.6745)

def run_galosh_gpu(noisy, w, h, strength=1.0, ls=1.0, cs=1.0,
                   alpha=1.0, sigma_sq=0.0):
    """Run GALOSH GPU on full RAW image.
    Args: galosh mode: input output W H galosh strength luma_str chroma_str alpha sigma_sq [cl_device]
    """
    uid = "fulltest"
    in_path  = OUTDIR / f"_tmp_{uid}_in.raw"
    out_path = OUTDIR / f"_tmp_{uid}_out.raw"
    noisy.astype(np.float32).tofile(str(in_path))
    cmd = [str(GPU_EXE),
           str(in_path), str(out_path),
           str(w), str(h),
           "galosh",
           str(strength), str(ls), str(cs),
           str(alpha), str(sigma_sq),
           "0"]  # cl_device
    print(f"    cmd: {' '.join(cmd[:4])}... w={w} h={h}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"    GPU FAILED: {r.stderr.decode()[:500]}")
        return None, dt
    den = np.fromfile(str(out_path), dtype=np.float32).reshape(h, w)
    in_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)
    return den, dt

def run_bm3d_cfa(noisy, sigma):
    """BM3D on raw Bayer (single-channel)."""
    import bm3d
    den = bm3d.bm3d(noisy.astype(np.float64), sigma_psd=sigma,
                     stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return np.clip(den, 0, 1).astype(np.float32)

def run_nlm_cfa(noisy, sigma):
    """NLM per Bayer channel: split RGGB → 4 quarter-res planes → NLM each → reassemble."""
    from skimage.restoration import denoise_nl_means, estimate_sigma
    h_nlm = 1.2 * sigma
    den = noisy.copy()
    # RGGB: (0,0)=R, (0,1)=Gr, (1,0)=Gb, (1,1)=B
    for dy, dx in [(0,0), (0,1), (1,0), (1,1)]:
        plane = noisy[dy::2, dx::2]
        plane_den = denoise_nl_means(plane, h=h_nlm, sigma=sigma,
                                      fast_mode=True, patch_size=5, patch_distance=11)
        den[dy::2, dx::2] = plane_den
    return np.clip(den, 0, 1).astype(np.float32)

# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="0001_S6_GRBG",
                        help="Scene tag (e.g. 0001_S6_GRBG)")
    args = parser.parse_args()
    tag = args.scene

    print(f"=== Full-image pre-demosaic test: {tag} ===\n")

    # Load cropped data
    gt_raw     = np.load(CROP / f"{tag}_gt_raw.npy")
    noisy_raw  = np.load(CROP / f"{tag}_noisy_raw.npy")
    gt_srgb    = np.load(CROP / f"{tag}_gt_srgb.npy")
    noisy_srgb = np.load(CROP / f"{tag}_noisy_srgb.npy")
    H, W = gt_raw.shape
    print(f"Image size: {W}×{H} (RGGB aligned)")
    print(f"Noisy RAW PSNR: {psnr(gt_raw, noisy_raw):.2f} dB\n")

    # Estimate affine
    affine = estimate_affine(gt_raw, gt_srgb)

    # GT② = GT_RAW through calibrated pipeline
    print("  Computing GT② (GT_RAW → Menon → affine → sRGB)...", flush=True)
    t0 = time.time()
    gt2_srgb = raw_to_srgb_calibrated(gt_raw, affine)
    print(f"  GT② done ({time.time()-t0:.1f}s)")
    gt2_vs_isp = psnr(gt_srgb, gt2_srgb)
    print(f"  GT② vs ISP sRGB (GT①): PSNR={gt2_vs_isp:.2f} dB\n")

    # Blind sigma
    sigma = sigma_mad_raw(noisy_raw)
    print(f"Blind sigma (MAD): {sigma:.6f}\n")

    # Save GT images
    from skimage.io import imsave
    imsave(str(OUTDIR / f"{tag}_gt_srgb_isp.png"),
           (np.clip(gt_srgb, 0, 1) * 255).astype(np.uint8))
    imsave(str(OUTDIR / f"{tag}_gt2_calibrated.png"),
           (np.clip(gt2_srgb, 0, 1) * 255).astype(np.uint8))
    imsave(str(OUTDIR / f"{tag}_noisy_srgb.png"),
           (np.clip(noisy_srgb, 0, 1) * 255).astype(np.uint8))
    print(f"Saved GT①, GT②, Noisy → {OUTDIR}/\n")

    # ── Run denoisers ──
    results = []

    # 1. GALOSH GPU
    print("[GALOSH GPU]")
    den_raw, dt = run_galosh_gpu(noisy_raw, W, H)
    if den_raw is not None:
        den_srgb = raw_to_srgb_calibrated(den_raw, affine)
        p = psnr(gt2_srgb, den_srgb)
        s = ssim_rgb(gt2_srgb, den_srgb)
        ip = psnr(gt_srgb, den_srgb)
        results.append(("GALOSH GPU", p, s, ip, dt))
        print(f"  PSNR={p:.2f}  SSIM={s:.4f}  ISP_PSNR={ip:.2f}  time={dt:.2f}s")
        imsave(str(OUTDIR / f"{tag}_galosh_gpu.png"),
               (np.clip(den_srgb, 0, 1) * 255).astype(np.uint8))

    # 2. BM3D-CFA (blind)
    print("\n[BM3D-CFA blind]")
    t0 = time.time()
    den_raw_bm3d = run_bm3d_cfa(noisy_raw, sigma)
    dt_bm3d = time.time() - t0
    den_srgb_bm3d = raw_to_srgb_calibrated(den_raw_bm3d, affine)
    p = psnr(gt2_srgb, den_srgb_bm3d)
    s = ssim_rgb(gt2_srgb, den_srgb_bm3d)
    ip = psnr(gt_srgb, den_srgb_bm3d)
    results.append(("BM3D-CFA blind", p, s, ip, dt_bm3d))
    print(f"  PSNR={p:.2f}  SSIM={s:.4f}  ISP_PSNR={ip:.2f}  time={dt_bm3d:.2f}s")
    imsave(str(OUTDIR / f"{tag}_bm3d_cfa.png"),
           (np.clip(den_srgb_bm3d, 0, 1) * 255).astype(np.uint8))

    # 3. NLM-CFA (blind)
    print("\n[NLM-CFA blind]")
    t0 = time.time()
    den_raw_nlm = run_nlm_cfa(noisy_raw, sigma)
    dt_nlm = time.time() - t0
    den_srgb_nlm = raw_to_srgb_calibrated(den_raw_nlm, affine)
    p = psnr(gt2_srgb, den_srgb_nlm)
    s = ssim_rgb(gt2_srgb, den_srgb_nlm)
    ip = psnr(gt_srgb, den_srgb_nlm)
    results.append(("NLM-CFA blind", p, s, ip, dt_nlm))
    print(f"  PSNR={p:.2f}  SSIM={s:.4f}  ISP_PSNR={ip:.2f}  time={dt_nlm:.2f}s")
    imsave(str(OUTDIR / f"{tag}_nlm_cfa.png"),
           (np.clip(den_srgb_nlm, 0, 1) * 255).astype(np.uint8))

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"  Full-image pre-demosaic results: {tag} ({W}×{H})")
    print(f"{'='*65}")
    print(f"{'Method':<18} {'PSNR':>7} {'SSIM':>7} {'ISP_PSNR':>9} {'Time':>8}")
    print("-" * 55)
    for name, p, s, ip, dt in results:
        print(f"{name:<18} {p:>7.2f} {s:>7.4f} {ip:>9.2f} {dt:>7.2f}s")
    print(f"\nPNG outputs → {OUTDIR}/")

if __name__ == "__main__":
    main()
