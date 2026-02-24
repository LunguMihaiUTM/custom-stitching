"""Multi-band blending for panorama stitching.

Laplacian pyramid blending: smoothly blends brightness (low freq)
while keeping details (high freq) sharp from each source.
"""

import cv2 as cv
import numpy as np


def _build_gaussian_pyramid(img, levels):
    """Build Gaussian pyramid (list of progressively smaller images)."""
    pyramid = [img.astype(np.float32)]
    for _ in range(levels):
        img = cv.pyrDown(pyramid[-1])
        pyramid.append(img)
    return pyramid


def _build_laplacian_pyramid(img, levels):
    """Build Laplacian pyramid from an image."""
    gauss = _build_gaussian_pyramid(img, levels)
    lap = []
    for i in range(levels):
        h, w = gauss[i].shape[:2]
        up = cv.pyrUp(gauss[i + 1], dstsize=(w, h))
        lap.append(gauss[i] - up)
    lap.append(gauss[-1])
    return lap


def multiband_blend(img_a, img_b, mask, levels=4):
    """Blend img_a and img_b using multi-band blending.

    Where mask=1, use img_b. Where mask=0, use img_a.
    The transition is smoothed across frequency bands.

    Args:
        img_a: background image (H, W, 3) uint8
        img_b: foreground image (H, W, 3) uint8
        mask: blend mask (H, W) float32 in [0, 1]
        levels: number of pyramid levels

    Returns:
        blended image (H, W, 3) uint8
    """
    # Ensure consistent sizes (pad to even dimensions for pyrDown)
    h, w = img_a.shape[:2]
    # Limit levels based on image size
    max_levels = int(np.log2(min(h, w))) - 1
    levels = min(levels, max(max_levels, 1))

    # Build Laplacian pyramids for both images
    lap_a = _build_laplacian_pyramid(img_a, levels)
    lap_b = _build_laplacian_pyramid(img_b, levels)

    # Build Gaussian pyramid for the mask
    mask_3ch = np.stack([mask] * 3, axis=-1).astype(np.float32)
    gauss_mask = _build_gaussian_pyramid(mask_3ch, levels)

    # Blend each level
    lap_blend = []
    for la, lb, gm in zip(lap_a, lap_b, gauss_mask):
        blended = la * (1 - gm) + lb * gm
        lap_blend.append(blended)

    # Reconstruct from blended Laplacian pyramid
    result = lap_blend[-1]
    for i in range(levels - 1, -1, -1):
        h, w = lap_blend[i].shape[:2]
        result = cv.pyrUp(result, dstsize=(w, h)) + lap_blend[i]

    return np.clip(result, 0, 255).astype(np.uint8)