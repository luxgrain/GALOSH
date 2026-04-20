"""
NLM (Non-Local Means) denoiser on PyTorch CUDA.

Matches skimage denoise_nl_means(fast_mode=False) behavior:
  weight = exp(-max(msd - 2*sigma^2, 0) / h^2)
  where msd = mean squared difference over patch

Supports:
  - nlm_cfa_cuda(bayer, sigma)  : per-Bayer-channel NLM on RGGB raw
  - nlm_srgb_cuda(srgb, sigma)  : per-channel NLM on sRGB
"""
import torch
import torch.nn.functional as F
import numpy as np


def _nlm_plane_cuda(plane: torch.Tensor, h: float, sigma: float = 0.0,
                    patch_radius: int = 2, search_radius: int = 11) -> torch.Tensor:
    """NLM on a single 2D float32 CUDA tensor (H, W).

    Uses vectorized shift-and-compare: for each offset in search window,
    compute patch SSD via convolution, convert to weight, accumulate.
    """
    H, W = plane.shape
    pr = patch_radius
    sr = search_radius
    ps = 2 * pr + 1
    ps_sq = float(ps * ps)
    pad = sr + pr

    padded = F.pad(plane.unsqueeze(0).unsqueeze(0),
                   [pad, pad, pad, pad], mode='reflect')[0, 0]  # (H+2*pad, W+2*pad)

    h_sq = h * h
    sigma_bias = 2.0 * sigma * sigma

    # Uniform kernel for patch SSD → MSD
    kernel = torch.ones(1, 1, ps, ps, device=plane.device, dtype=torch.float32) / ps_sq

    weight_sum = torch.zeros(H, W, device=plane.device, dtype=torch.float32)
    result = torch.zeros(H, W, device=plane.device, dtype=torch.float32)

    # Reference block: patches centered on each pixel
    # For conv2d input we need (H + 2*pr, W + 2*pr) region starting at sr
    ref = padded[sr:sr + H + 2 * pr, sr:sr + W + 2 * pr]  # (H+2*pr, W+2*pr)

    for dy in range(-sr, sr + 1):
        for dx in range(-sr, sr + 1):
            # Shifted block for offset (dy, dx)
            sy = sr + dy
            sx = sr + dx
            shifted = padded[sy:sy + H + 2 * pr, sx:sx + W + 2 * pr]

            # Mean squared difference over patch via convolution
            diff_sq = (shifted - ref) ** 2  # (H+2*pr, W+2*pr)
            msd = F.conv2d(diff_sq.unsqueeze(0).unsqueeze(0),
                           kernel, padding=0)[0, 0]  # (H, W)

            # Weight with bias correction (skimage convention)
            d = torch.clamp(msd - sigma_bias, min=0.0)
            w = torch.exp(-d / h_sq)

            # Pixel value at offset position
            val = padded[pad + dy:pad + dy + H, pad + dx:pad + dx + W]

            weight_sum += w
            result += w * val

    return result / weight_sum.clamp(min=1e-10)


def nlm_cfa_cuda(bayer: np.ndarray, sigma: float,
                 patch_radius: int = 2, search_radius: int = 11) -> np.ndarray:
    """NLM-CFA: split RGGB Bayer into 4 quarter-res planes, NLM each on GPU.

    Args:
        bayer: (H, W) float32 RGGB Bayer array [0, 1]
        sigma: noise standard deviation
    Returns:
        (H, W) float32 denoised Bayer
    """
    h_param = 1.2 * sigma
    device = torch.device('cuda')
    den = bayer.copy()

    for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        plane = torch.from_numpy(bayer[dy::2, dx::2].astype(np.float32)).to(device)
        denoised = _nlm_plane_cuda(plane, h_param, sigma, patch_radius, search_radius)
        den[dy::2, dx::2] = denoised.cpu().numpy()

    return np.clip(den, 0, 1).astype(np.float32)


def nlm_srgb_cuda(srgb: np.ndarray, sigma: float,
                  patch_radius: int = 2, search_radius: int = 11) -> np.ndarray:
    """NLM per-channel on sRGB image, GPU accelerated.

    Args:
        srgb: (H, W, 3) float32 sRGB [0, 1]
        sigma: noise standard deviation
    Returns:
        (H, W, 3) float32 denoised sRGB
    """
    h_param = 1.2 * sigma
    device = torch.device('cuda')
    den = np.zeros_like(srgb)

    for c in range(3):
        plane = torch.from_numpy(srgb[:, :, c].astype(np.float32)).to(device)
        denoised = _nlm_plane_cuda(plane, h_param, sigma, patch_radius, search_radius)
        den[:, :, c] = denoised.cpu().numpy()

    return np.clip(den, 0, 1).astype(np.float32)


if __name__ == "__main__":
    import time
    from pathlib import Path

    print("=== NLM CUDA vs skimage comparison ===")

    CROP = Path(__file__).parent.parent / "datasets" / "sidd" / "test_crop"

    for tag, label in [("0001_S6_GRBG", "S6 ISO100"), ("0036_GP_BGGR", "GP ISO6400")]:
        gt = np.load(CROP / f"{tag}_gt_raw.npy")
        noisy = np.load(CROP / f"{tag}_noisy_raw.npy")
        sigma = float(np.std(noisy.astype(np.float64) - gt.astype(np.float64)))
        psnr_n = 10 * np.log10(1 / np.mean((gt - noisy) ** 2))

        print(f"\n[{label}] shape={gt.shape} sigma={sigma:.6f}")

        t0 = time.time()
        den_cuda = nlm_cfa_cuda(noisy, sigma)
        dt_cuda = time.time() - t0
        psnr_cuda = 10 * np.log10(1 / np.mean((gt.astype(np.float64) - den_cuda) ** 2))

        from skimage.restoration import denoise_nl_means
        t0 = time.time()
        den_cpu = noisy.copy()
        h_nlm = 1.2 * sigma
        for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            p = noisy[dy::2, dx::2]
            den_cpu[dy::2, dx::2] = denoise_nl_means(
                p, h=h_nlm, sigma=sigma, fast_mode=False,
                patch_size=5, patch_distance=11)
        dt_cpu = time.time() - t0
        psnr_cpu = 10 * np.log10(1 / np.mean(
            (gt.astype(np.float64) - np.clip(den_cpu, 0, 1)) ** 2))

        print(f"  CUDA:   PSNR {psnr_n:.2f} → {psnr_cuda:.2f}  ({dt_cuda:.2f}s)")
        print(f"  CPU:    PSNR {psnr_n:.2f} → {psnr_cpu:.2f}  ({dt_cpu:.2f}s)")
        print(f"  Δ PSNR: {abs(psnr_cuda - psnr_cpu):.2f} dB, Speedup: {dt_cpu/dt_cuda:.1f}x")
