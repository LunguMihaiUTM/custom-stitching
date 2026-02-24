"""Exposure normalization for panorama stitching.

Computes per-frame gain factors to match a global target brightness.
"""

import cv2 as cv
import numpy as np


def compute_brightness(frame):
    """Average brightness of a BGR frame (via grayscale)."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return float(gray.mean())


def compute_gains(frames, target=None):
    """Compute per-frame gain factors to normalize exposure.

    Args:
        frames: list of BGR frames (np arrays)
        target: target brightness (default: median of all frames)

    Returns:
        list of (gain_b, gain_g, gain_r) tuples per frame
    """
    # Per-channel averages for each frame
    channel_avgs = []
    for frame in frames:
        b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        channel_avgs.append((float(b.mean()), float(g.mean()), float(r.mean())))

    # Global target: median per channel across all frames
    if target is None:
        all_b = [a[0] for a in channel_avgs]
        all_g = [a[1] for a in channel_avgs]
        all_r = [a[2] for a in channel_avgs]
        target_b = float(np.median(all_b))
        target_g = float(np.median(all_g))
        target_r = float(np.median(all_r))
    else:
        target_b = target_g = target_r = float(target)

    gains = []
    for avg_b, avg_g, avg_r in channel_avgs:
        gb = target_b / max(avg_b, 1.0)
        gg = target_g / max(avg_g, 1.0)
        gr = target_r / max(avg_r, 1.0)
        # Clamp individual channel gains
        gb = np.clip(gb, 0.7, 1.4)
        gg = np.clip(gg, 0.7, 1.4)
        gr = np.clip(gr, 0.7, 1.4)
        # Also limit the average gain to prevent overall over-brightening
        avg_gain = (gb + gg + gr) / 3.0
        if avg_gain > 1.3:
            scale = 1.3 / avg_gain
            gb *= scale
            gg *= scale
            gr *= scale
        gains.append((gb, gg, gr))

    return gains


def apply_gain(pixels_float, gain):
    """Apply per-channel gain to float pixels (N, 3) BGR.

    Args:
        pixels_float: (N, 3) float32 array in BGR order
        gain: (gain_b, gain_g, gain_r) tuple

    Returns:
        corrected pixels (N, 3) float32, NOT clipped to 255 yet
    """
    gb, gg, gr = gain
    corrected = pixels_float.copy()
    corrected[:, 0] *= gb
    corrected[:, 1] *= gg
    corrected[:, 2] *= gr
    return corrected