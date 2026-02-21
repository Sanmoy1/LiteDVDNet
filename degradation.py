"""
Complex degradation pipeline for LiteDVDNet training.

Implements a RealESRGAN-style composite degradation model that applies
combinations of Gaussian noise, Poisson noise, Gaussian blur, and JPEG
compression artifacts to simulate real-world camera degradation.

All degradation parameters are chosen ONCE per clip and applied identically
to all frames to maintain temporal consistency (critical for video models).

Input:  img  -- torch.Tensor [N, num_frames*C, H, W] in [0., 1.], on GPU
Output: imgn -- torch.Tensor [N, num_frames*C, H, W] in [0., 1.], on GPU
        stdn -- torch.Tensor [N, 1, 1, 1] effective noise std (Gaussian component)
"""

import io
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Gaussian Noise
# ---------------------------------------------------------------------------

def add_gaussian_noise(x: torch.Tensor, sigma_range: list) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Adds AWGN (Additive White Gaussian Noise) to a batch of frames.

    Args:
        x:            [N, F*C, H, W] float tensor on GPU, range [0, 1]
        sigma_range:  [low, high] noise std in [0, 1] scale

    Returns:
        noisy tensor (clamped to [0,1]), stdn tensor [N,1,1,1]
    """
    N = x.size(0)
    low, high = sigma_range[0], sigma_range[1]
    stdn = torch.empty((N, 1, 1, 1), device=x.device).uniform_(low, high)
    noise = torch.normal(mean=torch.zeros_like(x), std=stdn.expand_as(x))
    return torch.clamp(x + noise, 0., 1.), stdn


# ---------------------------------------------------------------------------
# Poisson Noise
# ---------------------------------------------------------------------------

def add_poisson_noise(x: torch.Tensor, scale_range: list) -> torch.Tensor:
    """
    Adds Poisson (shot) noise. Poisson noise is signal-dependent — brighter
    regions have more variance, matching real photon-counting sensor behaviour.

    We use the Gaussian approximation: var(X) ≈ λ, so std ≈ sqrt(x / scale).

    Args:
        x:            [N, F*C, H, W] float tensor on GPU, range [0, 1]
        scale_range:  [low, high] scale factor; higher = less noise

    Returns:
        noisy tensor (clamped to [0,1])
    """
    N = x.size(0)
    low, high = scale_range[0], scale_range[1]
    scale = torch.empty((N, 1, 1, 1), device=x.device).uniform_(low, high)

    # std of Poisson noise is proportional to sqrt(signal)
    std = torch.sqrt(x.clamp(min=1e-8) / scale)
    noise = torch.normal(mean=torch.zeros_like(x), std=std)
    return torch.clamp(x + noise, 0., 1.)


# ---------------------------------------------------------------------------
# Gaussian Blur
# ---------------------------------------------------------------------------

def _gaussian_kernel_2d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Creates a 2D Gaussian kernel [1, 1, kH, kW]."""
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g[:, None] * g[None, :]      # [kH, kW]
    return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, kH, kW]


def add_gaussian_blur(x: torch.Tensor, kernel_sizes: list, sigma_range: list) -> torch.Tensor:
    """
    Applies a separable Gaussian blur to all frames in the clip.
    Same kernel is applied to every frame (temporal consistency).

    Args:
        x:            [N, F*C, H, W] float tensor on GPU, range [0, 1]
        kernel_sizes: list of candidate odd kernel sizes, e.g. [3, 5, 7]
        sigma_range:  [low, high] for Gaussian sigma

    Returns:
        blurred tensor (same shape, same range)
    """
    ks = random.choice(kernel_sizes)
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    pad = ks // 2

    N, FC, H, W = x.shape
    kernel = _gaussian_kernel_2d(ks, sigma, x.device)  # [1, 1, kH, kW]
    kernel = kernel.expand(FC, 1, ks, ks)              # [FC, 1, kH, kW]

    # Depthwise convolution: each channel gets its own copy of the kernel
    out = F.conv2d(x, kernel, padding=pad, groups=FC)
    return out.clamp(0., 1.)


# ---------------------------------------------------------------------------
# JPEG Compression Artifacts
# ---------------------------------------------------------------------------

def add_jpeg_artifacts(x: torch.Tensor, quality_range: list) -> torch.Tensor:
    """
    Simulates JPEG / video-codec compression blocking artifacts.
    Quality is randomly chosen from [quality_range[0], quality_range[1]].

    Since PIL operates on CPU/numpy, we move to CPU, compress, then move back.
    Applied per-image in the batch with the same quality factor.

    Args:
        x:             [N, F*C, H, W] float tensor on GPU, range [0, 1]
        quality_range: [low, high] JPEG quality factor (1=worst, 95=best)

    Returns:
        compressed tensor (clamped to [0,1])
    """
    N, FC, H, W = x.shape
    num_frames = FC // 3   # assumes RGB (3 channels per frame)

    quality = random.randint(int(quality_range[0]), int(quality_range[1]))

    out = x.clone().cpu()

    for n in range(N):
        for f in range(num_frames):
            ch_start = f * 3
            frame = out[n, ch_start:ch_start+3, :, :]  # [3, H, W]

            # Convert to uint8 PIL image
            frame_np = (frame.permute(1, 2, 0).numpy() * 255.).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(frame_np, mode='RGB')

            # JPEG encode then decode in memory
            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            pil_decoded = Image.open(buf).convert('RGB')

            frame_decoded = torch.from_numpy(
                np.array(pil_decoded, dtype=np.float32) / 255.
            ).permute(2, 0, 1)  # [3, H, W]

            out[n, ch_start:ch_start+3, :, :] = frame_decoded

    return out.to(x.device).clamp(0., 1.)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def apply_complex_degradation(
    img_train: torch.Tensor,
    cfg: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a composite real-world degradation pipeline to a batch of
    multi-frame clips. Each degradation type is applied with a configurable
    probability. Degradation parameters are sampled ONCE per batch (same for
    all clips in the batch, different calls will vary).

    Args:
        img_train:  [N, F*C, H, W] float tensor on GPU, range [0, 1]
        cfg:        dict read from YAML 'degradation:' section

    Returns:
        imgn_train: degraded tensor [N, F*C, H, W], range [0, 1], clamped
        stdn:       effective Gaussian noise std [N, 1, 1, 1] for noise_map

    Example cfg:
        degradation:
          gaussian_noise:
            enabled: true
            sigma_range: [0.0196, 0.2157]   # [5/255, 55/255]
          poisson_noise:
            enabled: true
            scale_range: [0.05, 0.5]
            prob: 0.5
          blur:
            enabled: true
            kernel_sizes: [3, 5, 7]
            sigma_range: [0.5, 3.0]
            prob: 0.4
          compression:
            enabled: true
            quality_range: [20, 95]
            prob: 0.4
    """
    x = img_train

    # ---- 1. Blur (apply before noise, mirrors camera optics order) --------
    blur_cfg = cfg.get('blur', {})
    if blur_cfg.get('enabled', False) and random.random() < blur_cfg.get('prob', 0.4):
        x = add_gaussian_blur(
            x,
            kernel_sizes=blur_cfg.get('kernel_sizes', [3, 5, 7]),
            sigma_range=blur_cfg.get('sigma_range', [0.5, 3.0])
        )

    # ---- 2. Poisson noise (signal-dependent, models photon shot noise) ----
    poisson_cfg = cfg.get('poisson_noise', {})
    if poisson_cfg.get('enabled', False) and random.random() < poisson_cfg.get('prob', 0.5):
        x = add_poisson_noise(
            x,
            scale_range=poisson_cfg.get('scale_range', [0.05, 0.5])
        )

    # ---- 3. Gaussian noise (always applied — mirrors thermal sensor noise) -
    gauss_cfg = cfg.get('gaussian_noise', {})
    stdn = None
    if gauss_cfg.get('enabled', True):
        x, stdn = add_gaussian_noise(
            x,
            sigma_range=gauss_cfg.get('sigma_range', [5/255., 55/255.])
        )
    else:
        # If Gaussian is disabled, create a dummy stdn of 0
        stdn = torch.zeros((img_train.size(0), 1, 1, 1), device=img_train.device)

    # ---- 4. JPEG compression (applied last — codec artifacts are post-sensor)
    comp_cfg = cfg.get('compression', {})
    if comp_cfg.get('enabled', False) and random.random() < comp_cfg.get('prob', 0.4):
        x = add_jpeg_artifacts(
            x,
            quality_range=comp_cfg.get('quality_range', [20, 95])
        )

    return x.clamp(0., 1.), stdn
