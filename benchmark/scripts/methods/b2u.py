"""Blind2Unblind (Wang et al., CVPR 2022) — pre-demosaic RAW-RGB denoising.

EN: Wraps the pretrained `rawRGB_112rf20_beta19.4` checkpoint from the
    Blind2Unblind repo for blind Bayer denoising. The repo's test script
    runs on 256-crop SIDD validation blocks; we tile full-resolution SIDD
    Medium images (12MP) with overlap since the masker replicates each
    input 16× (4×4 mask grid), and full-res packed tensors would OOM.

JP: B2U 事前学習 (Poisson-Gauss 合成ノイズで自己教師あり学習) 済の raw-RGB
    checkpoint を使って盲検 Bayer デノイズ。公式 test は 256 パッチ前提
    だが、我々の SIDD Medium 12MP を 1 パッチで回すと Masker が 16 複製
    を張るため VRAM が足りない。そこで raw 側 1024×1024 タイル、
    overlap 64 で reflect pad しつつ走査、重複部は単純平均で合成。
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2] / "external" / "Blind2Unblind"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from arch_unet import UNet  # type: ignore  # noqa: E402

_CKPT = _REPO / "pretrained_models" / "rawRGB_112rf20_beta19.4.pth"
_WF = 48
_N_CH = 4  # packed RGGB
_MASK_WIDTH = 4
_DEPTH_POOLS = 5  # network pools 5 times -> packed tile must be /32
_BETA = 20.0     # paper default (--beta in benchmark_sidd_b2u.py argparse)

_model: UNet | None = None


def _load() -> UNet:
    """Load pretrained B2U UNet (strip any 'module.' prefix)."""
    global _model
    if _model is not None:
        return _model
    net = UNet(in_channels=_N_CH, out_channels=_N_CH, wf=_WF).cuda()
    state = torch.load(str(_CKPT), map_location="cuda")
    clean = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    net.load_state_dict(clean, strict=True)
    net.eval()
    _model = net
    return net


def _generate_mask(n: int, c: int, h: int, w: int, idx: int,
                   width: int, device) -> torch.Tensor:
    """Reconstruct B2U fix-index mask at packed resolution (h, w). Width=4."""
    # The fix mask places a 1 at position `idx` (in the width×width grid)
    # of every width×width block in each (H,W) plane.  Applied per-sample/
    # per-channel uniformly (same spatial pattern).  Matches generate_mask
    # in test_sidd_b2u.py with mask_type='fix_{idx}'.
    i_row = idx // width
    i_col = idx %  width
    mask = torch.zeros((h, w), dtype=torch.float32, device=device)
    mask[i_row::width, i_col::width] = 1.0
    return mask.view(1, 1, h, w).expand(n, 1, h, w).contiguous()


def _interpolate_mask(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace masked-out pixels with 3×3 cross-neighbour interpolation.
    Mirrors `interpolate_mask` in test_sidd_b2u.py."""
    n, c, h, w = tensor.shape
    kernel = torch.tensor(
        [[0.5, 1.0, 0.5],
         [1.0, 0.0, 1.0],
         [0.5, 1.0, 0.5]],
        dtype=tensor.dtype, device=tensor.device,
    ).view(1, 1, 3, 3)
    kernel = kernel / kernel.sum()
    filtered = torch.nn.functional.conv2d(
        tensor.view(n * c, 1, h, w), kernel, padding=1
    ).view(n, c, h, w)
    mask_inv = 1.0 - mask
    return filtered * mask + tensor * mask_inv


def _masker_train(img: torch.Tensor, width: int = _MASK_WIDTH
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (width²) masked variants for B2U aggregation. Returns (16N,C,h,w)."""
    n, c, h, w = img.shape
    k = width * width
    tensors = torch.zeros((n, k, c, h, w), device=img.device, dtype=img.dtype)
    masks   = torch.zeros((n, k, 1, h, w), device=img.device, dtype=img.dtype)
    for i in range(k):
        m = _generate_mask(n, c, h, w, i, width, img.device)  # (n,1,h,w)
        interp = _interpolate_mask(img, m.expand(-1, c, -1, -1))
        tensors[:, i] = interp
        masks[:, i]   = m
    return tensors.view(-1, c, h, w), masks.view(-1, 1, h, w)


def _infer_patch(net: UNet, packed: torch.Tensor, beta: float) -> torch.Tensor:
    """Run full B2U inference on one packed patch (1, 4, hp, wp).
    Returns denoised packed tensor (1, 4, hp, wp). hp/wp must be /32."""
    n, c, hp, wp = packed.shape
    net_input, mask = _masker_train(packed)
    # (16*n, 4, hp, wp) → network → mask-aware aggregate
    dn_out = (net(net_input) * mask).view(n, -1, c, hp, wp).sum(dim=1)
    exp_out = net(packed)
    return (dn_out + beta * exp_out) / (1.0 + beta)


def run_b2u(noisy_raw: np.ndarray, beta: float = _BETA,
            tile: int = 1024, overlap: int = 64) -> Tuple[np.ndarray, float]:
    """Denoise a full-resolution RGGB Bayer via tiled B2U.

    Args:
        noisy_raw: (H, W) float32 RGGB in [0, 1].
        tile:    raw-domain tile size (must be even, multiple of 64).
        overlap: raw-domain overlap between tiles (even, ≥ 2× receptive field).
    Returns:
        (denoised_raw (H, W), elapsed_seconds).
    """
    assert noisy_raw.ndim == 2 and noisy_raw.dtype == np.float32
    net = _load()
    H, W = noisy_raw.shape
    # Pad raw so H,W are multiples of 2 (for pixel_unshuffle) and tile covers cleanly.
    # Tile stride:  tile - overlap.  We over-cover and blend via accumulator weights.
    assert tile % 2 == 0 and overlap % 2 == 0 and tile > overlap

    t0 = time.time()

    with torch.no_grad():
        # Full reflect-pad so each axis is multiple of 2; packed → divisible by 32.
        # We tile on the RAW grid; within each tile we pad to /32 before network.
        raw = torch.from_numpy(noisy_raw).float().unsqueeze(0).unsqueeze(0).cuda()  # (1,1,H,W)

        out_accum  = torch.zeros_like(raw)
        wgt_accum  = torch.zeros_like(raw)
        # Precompute 2D blend.  For the single-tile case (tile >= H and
        # tile >= W) skip the boundary ramp entirely — the original ramp set
        # blend_1d[0] = blend_1d[-1] = 0 via linspace(0, 1, overlap), which
        # zeroed the 1-pixel border of every tile and capped PSNR at ~42 dB
        # on 256×256 SIDD Validation patches (1 tile = 256, ramp zeroed
        # the entire perimeter).
        # For multi-tile case, ramp eases the boundary between tiles; the
        # zeroed extreme edges still get covered by the OUTER tile that
        # extends beyond, so the artifact only manifested when tile size
        # matched image size exactly.
        # JP: 旧ロジックは linspace(0, 1, overlap) で blend_1d 両端を 0 に
        # していたため、tile == image size の単一タイル時に出力外周 1 pixel
        # が必ず 0 化され PSNR が ~42 dB で頭打ちになっていた（SIDD
        # Validation 256² 評価の B2U 値が 44 vs 公式 51 でズレていた直因）。
        # 単一タイル時は ramp スキップ、複数タイル時は外側タイルが extreme
        # edge をカバーするので問題なし。
        single_tile = (tile >= H) and (tile >= W)
        if single_tile:
            blend_2d = torch.ones((1, 1, tile, tile), device=raw.device)
        else:
            blend_1d = torch.ones(tile, device=raw.device)
            ramp = torch.linspace(0, 1, steps=overlap, device=raw.device)
            blend_1d[:overlap]        = ramp
            blend_1d[-overlap:]       = ramp.flip(0)
            blend_2d = blend_1d[:, None] * blend_1d[None, :]
            blend_2d = blend_2d.view(1, 1, tile, tile)

        stride = tile - overlap
        ys = list(range(0, max(H - tile, 0) + 1, stride))
        xs = list(range(0, max(W - tile, 0) + 1, stride))
        # Always include a final tile that hits the bottom/right edge.
        if not ys or ys[-1] + tile < H:
            ys.append(max(0, H - tile))
        if not xs or xs[-1] + tile < W:
            xs.append(max(0, W - tile))

        for y in ys:
            for x in xs:
                y0, x0 = y, x
                y1, x1 = min(H, y + tile), min(W, x + tile)
                th, tw = y1 - y0, x1 - x0
                patch = raw[:, :, y0:y1, x0:x1]
                # Pad this patch to (tile, tile) if it lies at edge.
                pad_h = tile - th
                pad_w = tile - tw
                if pad_h or pad_w:
                    patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')

                # Pack to 4ch packed RGGB, run B2U, unpack.
                packed = torch.nn.functional.pixel_unshuffle(patch, 2)  # (1,4,tile/2,tile/2)
                den_packed = _infer_patch(net, packed, beta)             # (1,4,tile/2,tile/2)
                den = torch.nn.functional.pixel_shuffle(den_packed, 2)   # (1,1,tile,tile)

                # Blend into accumulator on the valid region only.
                w2d = blend_2d[:, :, :th, :tw]
                out_accum[:, :, y0:y1, x0:x1] += den[:, :, :th, :tw] * w2d
                wgt_accum[:, :, y0:y1, x0:x1] += w2d

        out = (out_accum / wgt_accum.clamp_min(1e-8)).clamp(0.0, 1.0)
    dt = time.time() - t0
    return out.squeeze().cpu().numpy().astype(np.float32), dt
