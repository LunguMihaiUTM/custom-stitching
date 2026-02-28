"""Android IMU panorama stitcher - projects frames onto equirectangular canvas
using quaternion orientation data from Android TYPE_GAME_ROTATION_VECTOR.

Pure rotation-only stitching (no position/parallax correction) since Android
IMU data provides orientation but not absolute camera position.

Usage:
    python stitch_android.py --video video.mp4 --transforms imu_data.json -o panorama.png
    python stitch_android.py --video video.mp4 --transforms imu_data.json -n 50
"""

import argparse
import json
import os
import sys
import time

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from exposure import compute_gains, apply_gain, equalize_brightness
from blending import multiband_blend


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_imu_data(path):
    with open(path) as f:
        return json.load(f)


def extract_frame(cap, frame_idx):
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


# ---------------------------------------------------------------------------
# Coordinate conventions
#
# Android device frame (portrait): X=right, Y=up, Z=out-of-screen
# OpenCV camera frame:             X=right, Y=down, Z=forward
#
# The quaternion from TYPE_GAME_ROTATION_VECTOR gives R_world_to_device.
# We need R_world_to_camera_opencv.
#
# Chain: R_w2c = AXIS_FLIP @ R_sensor_rot @ R_world_to_device
#   where R_sensor_rot rotates from device portrait to camera sensor frame
#   and AXIS_FLIP converts camera sensor (Y-up, Z-out) to OpenCV (Y-down, Z-in)
# ---------------------------------------------------------------------------

AXIS_FLIP = np.diag([1.0, -1.0, -1.0])

# equirect_to_ray uses Y-up convention: X=East, Y=Up, Z=North
# Android ENU world frame:             X=East, Y=North, Z=Up
# This matrix converts Y-up vectors to ENU vectors (swap Y<->Z)
YUP_TO_ENU = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)


def sensor_rotation_matrix(degrees):
    """Rotation from device portrait frame to camera sensor frame.

    sensorRotationDegrees tells us how many degrees CW the camera sensor
    is rotated relative to the device's portrait orientation (looking at screen).
    """
    rad = np.radians(-degrees)  # CW = negative angle in math convention
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float64)


def build_device_to_opencv(sensor_rotation_deg):
    """Pre-combine sensor rotation + axis flip into a single matrix.

    For sensorRotationDegrees=90 this gives:
        [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
    Meaning: OpenCV X = device Y (landscape right)
             OpenCV Y = device X (landscape down)
             OpenCV Z = -device Z (into screen = forward)
    """
    R_sensor = sensor_rotation_matrix(sensor_rotation_deg)
    return AXIS_FLIP @ R_sensor


def quat_to_R_world_to_device(q):
    """Convert Android quaternion [x,y,z,w] to world-to-device rotation matrix.

    TYPE_GAME_ROTATION_VECTOR gives the rotation FROM device TO world frame,
    so we transpose to get world-to-device.
    """
    r = Rotation.from_quat([q['x'], q['y'], q['z'], q['w']])
    return r.as_matrix().T


def get_camera_rotation(sample, device_to_opencv):
    """Get R_world_to_camera (OpenCV convention) from an IMU sample.

    Returns R_w2c in Y-up world convention (matching equirect_to_ray):
      R_w2c_yup = R_w2c_enu @ YUP_TO_ENU
    so that cam_coords = R_w2c_yup @ ray_yup works directly.
    """
    R_w2d = quat_to_R_world_to_device(sample['quaternion'])
    R_w2c_enu = device_to_opencv @ R_w2d
    return R_w2c_enu @ YUP_TO_ENU


def get_camera_forward_world(sample, device_to_opencv):
    """Get the camera's forward direction in world space."""
    R_w2c = get_camera_rotation(sample, device_to_opencv)
    # OpenCV forward = [0, 0, 1] in camera frame
    # World forward = R_c2w @ [0, 0, 1] = R_w2c.T @ [0, 0, 1] = 3rd row of R_w2c.T
    return R_w2c.T @ np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

def build_intrinsics(imu_data, video_w, video_h):
    """Build the 3x3 intrinsics matrix scaled from sensor to video resolution.

    The JSON intrinsics are for the full sensor resolution (e.g. 4624x3468).
    The video is typically 16:9 (1920x1080), center-cropped from the 4:3 sensor
    then downscaled uniformly.
    """
    intr = imu_data['cameraIntrinsics']
    fx, fy = intr['fx'], intr['fy']
    cx, cy = intr['cx'], intr['cy']

    # Infer full sensor resolution from principal point (center of sensor)
    sensor_w = int(round(cx * 2))
    sensor_h = int(round(cy * 2))

    # Uniform scale factor (width-based)
    scale = video_w / sensor_w

    # Height crop: sensor is 4:3, video is 16:9
    crop_h = sensor_w * video_h / video_w  # cropped height in sensor pixels
    cy_offset = (sensor_h - crop_h) / 2.0  # pixels removed from top

    K = np.array([
        [fx * scale, 0,          cx * scale],
        [0,          fy * scale, (cy - cy_offset) * scale],
        [0,          0,          1]
    ], dtype=np.float64)

    print(f"Intrinsics: sensor {sensor_w}x{sensor_h} -> video {video_w}x{video_h}")
    print(f"  scale={scale:.4f}, cy_offset={cy_offset:.1f}px")
    print(f"  K_video: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
          f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    return K


def build_intrinsics_portrait(imu_data, port_w, port_h, sensor_rot_deg):
    """Build intrinsics for a portrait frame (OpenCV auto-rotated).

    The JSON intrinsics are for the landscape sensor. When OpenCV applies the
    90° rotation, the frame becomes portrait and we need to remap intrinsics.
    For 90° CW rotation: portrait_u = raw_h - 1 - raw_v, portrait_v = raw_u
    """
    intr = imu_data['cameraIntrinsics']
    fx_s, fy_s = intr['fx'], intr['fy']
    cx_s, cy_s = intr['cx'], intr['cy']

    # Landscape sensor dimensions
    sensor_w = int(round(cx_s * 2))
    sensor_h = int(round(cy_s * 2))

    # First compute landscape video intrinsics (before rotation)
    # Landscape video would be port_h x port_w (e.g., 1920x1080)
    land_w, land_h = port_h, port_w
    scale = land_w / sensor_w
    crop_h = sensor_w * land_h / land_w
    cy_offset = (sensor_h - crop_h) / 2.0

    fx_land = fx_s * scale
    fy_land = fy_s * scale
    cx_land = cx_s * scale
    cy_land = (cy_s - cy_offset) * scale

    # After 90° CW rotation: u_port = land_h - 1 - v_land, v_port = u_land
    # fx_port corresponds to the fy_land direction, fy_port to fx_land
    fx_port = fy_land
    fy_port = fx_land
    cx_port = (land_h - 1) - cy_land
    cy_port = cx_land

    K = np.array([
        [fx_port, 0,       cx_port],
        [0,       fy_port, cy_port],
        [0,       0,       1]
    ], dtype=np.float64)

    print(f"Intrinsics (portrait): sensor {sensor_w}x{sensor_h} -> "
          f"portrait {port_w}x{port_h}")
    print(f"  K_port: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
          f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    return K


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def select_frames(samples, num_frames, device_to_opencv):
    """Select frames evenly distributed across the yaw range."""
    valid = []
    for sample in samples:
        fwd = get_camera_forward_world(sample, device_to_opencv)
        yaw = np.arctan2(fwd[0], fwd[2])
        valid.append((sample['frameIndex'], yaw))

    if len(valid) <= num_frames:
        return [i for i, _ in valid]

    yaw_arr = np.array([y for _, y in valid])
    idx_arr = np.array([i for i, _ in valid])

    bin_edges = np.linspace(-np.pi, np.pi, num_frames + 1)
    selected = []
    for b in range(num_frames):
        mask = (yaw_arr >= bin_edges[b]) & (yaw_arr < bin_edges[b + 1])
        candidates = idx_arr[mask]
        if len(candidates) > 0:
            selected.append(int(candidates[len(candidates) // 2]))

    if len(selected) < num_frames:
        step = max(1, len(valid) // num_frames)
        used = set(selected)
        for j in range(0, len(valid), step):
            if valid[j][0] not in used:
                selected.append(valid[j][0])
                used.add(valid[j][0])
            if len(selected) >= num_frames:
                break

    return sorted(set(selected))


# ---------------------------------------------------------------------------
# Equirectangular helpers
# ---------------------------------------------------------------------------

def equirect_to_ray(lon, lat):
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    return np.stack([x, y, z], axis=-1)


def find_seam_dp(img_a, img_b, mask_a, mask_b):
    """Find minimum-cost vertical seam through the overlap of two images.

    Uses dynamic programming to find a vertical path (one pixel per row)
    that cuts through where the two images are most similar, minimizing
    visible ghosting artifacts.

    Args:
        img_a: existing panorama patch (H, W, 3) float32
        img_b: new strip patch (H, W, 3) float32
        mask_a: where panorama has valid pixels (H, W) bool
        mask_b: where strip has valid pixels (H, W) bool

    Returns:
        seam_mask: (H, W) float32, 0=use img_a, 1=use img_b
    """
    h, w = img_a.shape[:2]
    overlap = mask_a & mask_b

    if not np.any(overlap):
        # No overlap — use strip wherever it has content
        return mask_b.astype(np.float32)

    # Cost = color difference magnitude in overlap
    diff = np.sum((img_a - img_b) ** 2, axis=2)

    # Penalize cutting outside overlap
    cost = np.full((h, w), 1e8, dtype=np.float32)
    cost[overlap] = diff[overlap]

    # DP: accumulate min cost top-to-bottom
    dp = np.full((h, w), 1e8, dtype=np.float32)
    dp[0] = cost[0]
    back = np.zeros((h, w), dtype=np.int32)

    for y in range(1, h):
        # Three candidates: from (y-1, x-1), (y-1, x), (y-1, x+1)
        left = np.empty(w, dtype=np.float32)
        left[0] = 1e8
        left[1:] = dp[y - 1, :-1]

        center = dp[y - 1]

        right = np.empty(w, dtype=np.float32)
        right[-1] = 1e8
        right[:-1] = dp[y - 1, 1:]

        candidates = np.stack([left, center, right], axis=0)  # (3, W)
        best = np.argmin(candidates, axis=0)
        dp[y] = cost[y] + candidates[best, np.arange(w)]
        back[y] = best - 1  # -1=from left, 0=center, 1=from right

    # Trace back from bottom row (prefer overlap columns)
    valid_costs = dp[-1].copy()
    bottom_overlap_cols = np.where(overlap[-1])[0]
    if len(bottom_overlap_cols) > 0:
        non_overlap = np.ones(w, dtype=bool)
        non_overlap[bottom_overlap_cols] = False
        valid_costs[non_overlap] = 1e8

    x = int(np.argmin(valid_costs))

    seam_x = np.zeros(h, dtype=np.int32)
    seam_x[-1] = x
    for y in range(h - 2, -1, -1):
        seam_x[y] = int(np.clip(seam_x[y + 1] - back[y + 1, seam_x[y + 1]], 0, w - 1))

    # Binary mask: left of seam = 0 (panorama), right of seam = 1 (strip)
    col_indices = np.arange(w)[np.newaxis, :]
    seam_mask = (col_indices >= seam_x[:, np.newaxis]).astype(np.float32)

    return seam_mask


def frame_canvas_bounds(R_c2w, K, img_w, img_h, canvas_w, canvas_h):
    """Compute the canvas pixel bounding box where a frame projects."""
    border_u = [0, img_w/2, img_w-1, img_w-1, img_w-1, img_w/2, 0, 0]
    border_v = [0, 0, 0, img_h/2, img_h-1, img_h-1, img_h-1, img_h/2]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    lons = []
    lats = []
    for bu, bv in zip(border_u, border_v):
        cam_ray = np.array([(bu - cx) / fx, (bv - cy) / fy, 1.0])
        cam_ray /= np.linalg.norm(cam_ray)
        world_ray = R_c2w @ cam_ray
        lon = np.arctan2(world_ray[0], world_ray[2])
        lat = np.arcsin(np.clip(world_ray[1], -1, 1))
        lons.append(lon)
        lats.append(lat)

    lons = np.array(lons)
    lats = np.array(lats)

    lon_min, lon_max = lons.min(), lons.max()
    wraps = (lon_max - lon_min) > np.pi
    margin = 0.05

    if wraps:
        pos_lons = lons[lons >= 0]
        neg_lons = lons[lons < 0]
        lon_start = float(pos_lons.min() - margin) if len(pos_lons) > 0 else np.pi - margin
        lon_end = float(neg_lons.max() + margin) if len(neg_lons) > 0 else -np.pi + margin
    else:
        lon_start = float(lon_min - margin)
        lon_end = float(lon_max + margin)

    lat_min = float(lats.min() - margin)
    lat_max = float(lats.max() + margin)

    def lon_to_col(l):
        return int(np.clip((l / (2 * np.pi) + 0.5) * canvas_w, 0, canvas_w - 1))

    def lat_to_row(l):
        return int(np.clip((0.5 - l / np.pi) * canvas_h, 0, canvas_h - 1))

    row_start = lat_to_row(lat_max)
    row_end = lat_to_row(lat_min)

    if wraps:
        col_start = lon_to_col(lon_start)
        col_end = lon_to_col(lon_end)
        return col_start, col_end, row_start, row_end, True
    else:
        col_start = lon_to_col(lon_start)
        col_end = lon_to_col(lon_end)
        return col_start, col_end, row_start, row_end, False


# ---------------------------------------------------------------------------
# Layered stitching (rotation-only projection)
# ---------------------------------------------------------------------------

def stitch_layered(cap, samples, K, frame_indices, device_to_opencv,
                   canvas_w=None, canvas_h=None, reverse=False):
    """Project frames onto equirectangular canvas using rotation-only projection.

    Unlike the ARKit version, there is no position data, so no pivot estimation
    or parallax correction. Each canvas ray is simply rotated into the camera
    frame and projected with intrinsics.
    """
    t0 = time.time()
    n_frames = len(frame_indices)

    # Build a lookup from frameIndex -> sample
    sample_lookup = {s['frameIndex']: s for s in samples}

    # Gather camera rotations for selected frames
    R_w2cs = []
    R_c2ws = []

    for idx in frame_indices:
        sample = sample_lookup[idx]
        R_w2c = get_camera_rotation(sample, device_to_opencv)
        R_w2cs.append(R_w2c)
        R_c2ws.append(R_w2c.T)

    # Canvas size from focal length
    avg_focal = (K[0, 0] + K[1, 1]) / 2.0
    if canvas_w is None:
        canvas_w = int(round(2 * np.pi * avg_focal))
    if canvas_h is None:
        canvas_h = canvas_w // 2

    print(f"Canvas: {canvas_w}x{canvas_h}, focal={avg_focal:.1f}")

    # Get frame dimensions
    test_frame = extract_frame(cap, frame_indices[0])
    img_h, img_w = test_frame.shape[:2]
    print(f"Frame size: {img_w}x{img_h}")

    # Exposure gains (computed but not applied)
    print("Computing exposure gains...")
    all_frames = []
    for idx in frame_indices:
        f = extract_frame(cap, idx)
        if f is not None:
            all_frames.append(f)
    exposure_gains = compute_gains(all_frames)
    del all_frames
    gain_values = [sum(g)/3 for g in exposure_gains]
    print(f"Exposure gains: min={min(gain_values):.3f}, max={max(gain_values):.3f}, "
          f"spread={max(gain_values)-min(gain_values):.3f}")

    # Precompute lon/lat arrays for the full canvas
    xs = np.arange(canvas_w, dtype=np.float64)
    ys = np.arange(canvas_h, dtype=np.float64)
    lon_full = (xs / canvas_w - 0.5) * 2 * np.pi
    lat_full = (0.5 - ys / canvas_h) * np.pi

    panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    painted_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    # Compute yaw for each frame
    yaws = []
    for fi in range(n_frames):
        fwd = get_camera_forward_world(sample_lookup[frame_indices[fi]],
                                        device_to_opencv)
        yaws.append(np.degrees(np.arctan2(fwd[0], fwd[2])))
    yaws = np.array(yaws)
    print(f"Yaw range: {yaws.min():.1f}\u00b0 to {yaws.max():.1f}\u00b0 "
          f"(sweep: {yaws.max() - yaws.min():.1f}\u00b0)")

    # Sort by yaw so seam lands at ±180° boundary
    order = list(np.argsort(yaws))
    if reverse:
        order = order[::-1]

    avg_yaw_step = 360.0 / n_frames
    strip_half_deg = avg_yaw_step * 1.2  # tighter than stitch_layer.py to reduce ghosting

    print(f"Painting {n_frames} frames sorted by yaw "
          f"(strip width {2*strip_half_deg:.1f}\u00b0)...")

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # ---- Pass 1: Full FOV paint ----

    def paint_frame(fi, use_strip=False):
        idx = frame_indices[fi]
        frame = extract_frame(cap, idx)
        if frame is None:
            return

        bounds = frame_canvas_bounds(R_c2ws[fi], K, img_w, img_h,
                                     canvas_w, canvas_h)
        if bounds is None:
            return

        col_start, col_end, row_start, row_end, wraps = bounds

        if wraps:
            cols = np.concatenate([np.arange(col_start, canvas_w),
                                   np.arange(0, col_end + 1)])
        else:
            cols = np.arange(col_start, col_end + 1)

        if use_strip:
            yaw_rad = np.radians(yaws[fi])
            strip_half_rad = np.radians(strip_half_deg)
            lon_lo = yaw_rad - strip_half_rad
            lon_hi = yaw_rad + strip_half_rad
            col_lo = int(np.clip((lon_lo / (2 * np.pi) + 0.5) * canvas_w, 0, canvas_w - 1))
            col_hi = int(np.clip((lon_hi / (2 * np.pi) + 0.5) * canvas_w, 0, canvas_w - 1))
            if col_lo <= col_hi:
                strip_set = set(range(col_lo, col_hi + 1))
            else:
                strip_set = set(range(col_lo, canvas_w)) | set(range(0, col_hi + 1))
            cols = np.array([c for c in cols if c in strip_set])
            if len(cols) == 0:
                return

        rows = np.arange(row_start, row_end + 1)
        if len(cols) == 0 or len(rows) == 0:
            return

        sub_lon = lon_full[cols]
        sub_lat = lat_full[rows]
        sub_lon_grid, sub_lat_grid = np.meshgrid(sub_lon, sub_lat)

        sub_rays = equirect_to_ray(sub_lon_grid, sub_lat_grid)
        sub_h, sub_w = sub_rays.shape[:2]
        rays_flat = sub_rays.reshape(-1, 3).astype(np.float32)

        # Pure rotation projection (no parallax)
        cam_coords = R_w2cs[fi] @ rays_flat.T

        z = cam_coords[2, :]
        valid = z > 0

        u = np.full(len(rays_flat), -1.0, dtype=np.float32)
        v = np.full(len(rays_flat), -1.0, dtype=np.float32)

        u[valid] = cam_coords[0, valid] / z[valid] * fx + cx
        v[valid] = cam_coords[1, valid] / z[valid] * fy + cy

        in_bounds = valid & (u >= 0) & (u < img_w - 1) & (v >= 0) & (v < img_h - 1)
        if not np.any(in_bounds):
            return

        # Bilinear sample
        u_valid = u[in_bounds]
        v_valid = v[in_bounds]
        u0 = np.floor(u_valid).astype(np.int32)
        v0 = np.floor(v_valid).astype(np.int32)
        u1 = np.clip(u0 + 1, 0, img_w - 1)
        v1 = np.clip(v0 + 1, 0, img_h - 1)
        u0 = np.clip(u0, 0, img_w - 1)
        v0 = np.clip(v0, 0, img_h - 1)

        du = (u_valid - u0).astype(np.float32)[:, np.newaxis]
        dv = (v_valid - v0).astype(np.float32)[:, np.newaxis]

        p00 = frame[v0, u0].astype(np.float32)
        p01 = frame[v0, u1].astype(np.float32)
        p10 = frame[v1, u0].astype(np.float32)
        p11 = frame[v1, u1].astype(np.float32)

        interp = (p00 * (1 - du) * (1 - dv) +
                  p01 * du * (1 - dv) +
                  p10 * (1 - du) * dv +
                  p11 * du * dv)

        flat_idx = np.where(in_bounds)[0]
        sub_row = flat_idx // sub_w
        sub_col = flat_idx % sub_w

        canvas_row = rows[sub_row]
        canvas_col = cols[sub_col]

        pixels = np.clip(interp, 0, 255).astype(np.uint8)
        panorama[canvas_row, canvas_col] = pixels
        painted_mask[canvas_row, canvas_col] = 255

    for count, fi in enumerate(order):
        paint_frame(fi, use_strip=False)
        if (count + 1) % 10 == 0:
            print(f"  Pass 1: {count + 1}/{n_frames} frames")
    print(f"  Pass 1 done: {n_frames} frames (full FOV)")

    # ---- Pass 2: cv.detail.MultiBandBlender ----
    # OpenCV's industry-standard multi-band blender processes ALL frames
    # simultaneously in a single Laplacian pyramid decomposition.
    # Low frequencies (brightness) blend over wide zones → no banding.
    # High frequencies (detail) blend over narrow zones → no ghosting.
    print(f"  Pass 2: cv.detail.MultiBandBlender ({n_frames} frames, 5 bands)...")

    # Determine pixel ownership: each column belongs to the frame with
    # nearest yaw. This creates non-overlapping seam masks (Voronoi strips).
    yaw_rad_ordered = np.array([np.radians(yaws[order[i]])
                                 for i in range(len(order))])
    ownership = np.zeros(canvas_w, dtype=np.int32)
    for c in range(canvas_w):
        lon = lon_full[c]
        dists = np.abs(np.arctan2(np.sin(lon - yaw_rad_ordered),
                                   np.cos(lon - yaw_rad_ordered)))
        ownership[c] = int(np.argmin(dists))

    blender = cv.detail.MultiBandBlender(0, 5)
    blender.prepare((0, 0, canvas_w, canvas_h))

    for oi, fi in enumerate(order):
        idx = frame_indices[fi]
        frame = extract_frame(cap, idx)
        if frame is None:
            continue

        bounds = frame_canvas_bounds(R_c2ws[fi], K, img_w, img_h,
                                     canvas_w, canvas_h)
        if bounds is None:
            continue
        col_start, col_end, row_start, row_end, wraps = bounds

        if wraps:
            cols = np.concatenate([np.arange(col_start, canvas_w),
                                   np.arange(0, col_end + 1)])
        else:
            cols = np.arange(col_start, col_end + 1)
        rows = np.arange(row_start, row_end + 1)

        if len(cols) == 0 or len(rows) == 0:
            continue

        sub_lon = lon_full[cols]
        sub_lat = lat_full[rows]
        sub_lon_grid, sub_lat_grid = np.meshgrid(sub_lon, sub_lat)
        sub_rays = equirect_to_ray(sub_lon_grid, sub_lat_grid)
        sub_h, sub_w = sub_rays.shape[:2]
        rays_flat = sub_rays.reshape(-1, 3).astype(np.float32)

        cam_coords = R_w2cs[fi] @ rays_flat.T
        z = cam_coords[2, :]
        valid = z > 0

        u = np.full(len(rays_flat), -1.0, dtype=np.float32)
        v = np.full(len(rays_flat), -1.0, dtype=np.float32)
        u[valid] = cam_coords[0, valid] / z[valid] * fx + cx
        v[valid] = cam_coords[1, valid] / z[valid] * fy + cy
        in_bounds = valid & (u >= 0) & (u < img_w - 1) & (v >= 0) & (v < img_h - 1)

        if not np.any(in_bounds):
            continue

        u_valid = u[in_bounds]
        v_valid = v[in_bounds]
        u0 = np.floor(u_valid).astype(np.int32)
        v0 = np.floor(v_valid).astype(np.int32)
        u1 = np.clip(u0 + 1, 0, img_w - 1)
        v1 = np.clip(v0 + 1, 0, img_h - 1)
        u0 = np.clip(u0, 0, img_w - 1)
        v0 = np.clip(v0, 0, img_h - 1)
        du = (u_valid - u0).astype(np.float32)[:, np.newaxis]
        dv = (v_valid - v0).astype(np.float32)[:, np.newaxis]
        p00 = frame[v0, u0].astype(np.float32)
        p01 = frame[v0, u1].astype(np.float32)
        p10 = frame[v1, u0].astype(np.float32)
        p11 = frame[v1, u1].astype(np.float32)
        interp = (p00 * (1 - du) * (1 - dv) +
                  p01 * du * (1 - dv) +
                  p10 * (1 - du) * dv +
                  p11 * du * dv)

        flat_idx = np.where(in_bounds)[0]
        sr = flat_idx // sub_w
        sc = flat_idx % sub_w

        # For wrapping frames, build contiguous sub-images per segment
        if wraps:
            # Split into right segment (col_start..canvas_w-1)
            # and left segment (0..col_end)
            right_len = canvas_w - col_start
            left_len = col_end + 1

            for seg_start, seg_cols in [(col_start, np.arange(col_start, canvas_w)),
                                         (0, np.arange(0, col_end + 1))]:
                seg_len = len(seg_cols)
                if seg_len == 0:
                    continue
                # Map sub_col indices to this segment
                if seg_start == col_start:
                    col_offset = 0
                else:
                    col_offset = right_len
                seg_img = np.zeros((sub_h, seg_len, 3), dtype=np.int16)
                seg_mask = np.zeros((sub_h, seg_len), dtype=np.uint8)

                # Which pixels fall in this segment
                local_sc = sc - col_offset
                in_seg = (local_sc >= 0) & (local_sc < seg_len)
                if in_seg.any():
                    seg_img[sr[in_seg], local_sc[in_seg]] = \
                        np.clip(interp[in_seg], 0, 255).astype(np.int16)
                    seg_mask[sr[in_seg], local_sc[in_seg]] = 255

                # Apply ownership
                owned = ownership[seg_cols] == oi
                seg_mask[:, ~owned] = 0

                if seg_mask.any():
                    blender.feed(seg_img, seg_mask, (seg_start, row_start))
        else:
            sub_img = np.zeros((sub_h, sub_w, 3), dtype=np.int16)
            sub_mask = np.zeros((sub_h, sub_w), dtype=np.uint8)
            sub_img[sr, sc] = np.clip(interp, 0, 255).astype(np.int16)
            sub_mask[sr, sc] = 255

            # Apply ownership: only pixels owned by this frame
            owned = ownership[cols] == oi
            sub_mask[:, ~owned] = 0

            if sub_mask.any():
                blender.feed(sub_img, sub_mask, (col_start, row_start))

        if (oi + 1) % 10 == 0:
            print(f"    {oi + 1}/{n_frames} frames fed")

    print("  Blending pyramids...")
    result, result_mask = blender.blend(None, None)
    blended_pano = np.clip(result, 0, 255).astype(np.uint8)

    # Overlay blended result onto Pass 1 panorama (Pass 1 fills gaps)
    blend_valid = result_mask > 0
    panorama[blend_valid] = blended_pano[blend_valid]
    painted_mask[blend_valid] = 255
    del blended_pano, result, result_mask
    print(f"  Pass 2 done: {n_frames} frames (cv.detail.MultiBandBlender)")

    # Save painted mask for boundary diffusion later
    _debug_painted = painted_mask.copy()

    # ---- Center panorama content vertically ----
    painted_rows = np.any(painted_mask > 0, axis=1)
    row_indices_painted = np.where(painted_rows)[0]
    if len(row_indices_painted) > 0:
        content_center = (row_indices_painted[0] + row_indices_painted[-1]) / 2.0
        canvas_center = canvas_h / 2.0
        shift_px = int(round(canvas_center - content_center))
        if abs(shift_px) > 1:
            print(f"Centering panorama: shifting {shift_px}px vertically")
            panorama = np.roll(panorama, shift_px, axis=0)
            painted_mask = np.roll(painted_mask, shift_px, axis=0)
            _debug_painted = np.roll(_debug_painted, shift_px, axis=0)
            if shift_px > 0:
                panorama[:shift_px] = 0
                painted_mask[:shift_px] = 0
                _debug_painted[:shift_px] = 0
            else:
                panorama[shift_px:] = 0
                painted_mask[shift_px:] = 0
                _debug_painted[shift_px:] = 0

    # ---- Blur gap filling ----
    print("Filling gaps with blur padding...")
    painted = painted_mask > 0

    if not painted.all():
        erode_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        eroded_mask = cv.erode(painted_mask, erode_k, iterations=3)
        safe_pano = panorama.copy()
        safe_pano[eroded_mask == 0] = 0

        # Edge-color propagation at reduced scale
        scale = 8
        sh, sw = canvas_h // scale, canvas_w // scale
        small_pano = cv.resize(safe_pano, (sw, sh), interpolation=cv.INTER_AREA)
        small_mask = cv.resize(eroded_mask, (sw, sh),
                               interpolation=cv.INTER_NEAREST) > 0

        # Top-down propagation
        filled_down = small_pano.copy()
        mask_down = small_mask.copy()
        for y in range(1, sh):
            unpainted = ~mask_down[y]
            if unpainted.any():
                filled_down[y][unpainted] = filled_down[y - 1][unpainted]
                mask_down[y] |= mask_down[y - 1]

        # Bottom-up propagation
        filled_up_dir = small_pano.copy()
        mask_up = small_mask.copy()
        for y in range(sh - 2, -1, -1):
            unpainted = ~mask_up[y]
            if unpainted.any():
                filled_up_dir[y][unpainted] = filled_up_dir[y + 1][unpainted]
                mask_up[y] |= mask_up[y + 1]

        # Blend by vertical distance
        dist_down = np.full((sh, sw), np.inf, dtype=np.float32)
        dist_up = np.full((sh, sw), np.inf, dtype=np.float32)
        for y in range(sh):
            p = small_mask[y]
            dist_down[y][p] = 0
            if y > 0:
                dist_down[y][~p] = dist_down[y - 1][~p] + 1
        for y in range(sh - 1, -1, -1):
            p = small_mask[y]
            dist_up[y][p] = 0
            if y < sh - 1:
                dist_up[y][~p] = dist_up[y + 1][~p] + 1

        total = dist_down + dist_up
        both_inf = np.isinf(dist_down) & np.isinf(dist_up)
        total[total == 0] = 1
        total[both_inf] = 1
        with np.errstate(invalid='ignore'):
            w_down = dist_up / total
            w_up_w = dist_down / total
        only_down = np.isinf(dist_up) & ~np.isinf(dist_down)
        only_up = np.isinf(dist_down) & ~np.isinf(dist_up)
        w_down[only_down] = 1.0; w_up_w[only_down] = 0.0
        w_up_w[only_up] = 1.0; w_down[only_up] = 0.0
        w_down[both_inf] = 0.5; w_up_w[both_inf] = 0.5

        filled_blend = (filled_down.astype(np.float32) * w_down[:, :, np.newaxis] +
                        filled_up_dir.astype(np.float32) * w_up_w[:, :, np.newaxis])
        filled_blend = np.clip(filled_blend, 0, 255).astype(np.uint8)
        filled_blend[small_mask] = small_pano[small_mask]

        # Left-right propagation for side gaps
        mask_lr = small_mask | (w_down + w_up_w > 0.01)
        for x in range(1, sw):
            u = ~mask_lr[:, x]
            if u.any():
                filled_blend[:, x][u] = filled_blend[:, x - 1][u]
                mask_lr[:, x] |= mask_lr[:, x - 1]
        for x in range(sw - 2, -1, -1):
            u = ~mask_lr[:, x]
            if u.any():
                filled_blend[:, x][u] = filled_blend[:, x + 1][u]
                mask_lr[:, x] |= mask_lr[:, x + 1]

        # Horizontal smoothing
        kw = max(sw // 8, 3) | 1
        smoothed = cv.GaussianBlur(filled_blend.astype(np.float32), (kw, 3), 0)
        smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
        gap_px = ~small_mask
        for c in range(3):
            filled_blend[:, :, c][gap_px] = smoothed[:, :, c][gap_px]

        # Upscale and progressive blur (RAM-optimized: one level at a time)
        filled_full = cv.resize(filled_blend, (canvas_w, canvas_h),
                                interpolation=cv.INTER_LINEAR)
        del filled_blend
        eroded_bool = eroded_mask > 0
        filled_full[eroded_bool] = safe_pano[eroded_bool]
        del safe_pano

        gap_full = (~eroded_bool).astype(np.uint8)
        dist_full = cv.distanceTransform(gap_full, cv.DIST_L2, 5)
        del gap_full

        bg = filled_full.astype(np.float32)

        # Light blur (0-15px from edge)
        t = np.clip(dist_full / 15.0, 0, 1).astype(np.float32)[:, :, np.newaxis]
        blur = cv.GaussianBlur(filled_full, (31, 31), 0)
        bg = bg * (1 - t) + blur.astype(np.float32) * t
        del blur, t

        # Medium blur (15-60px)
        t = np.clip((dist_full - 15) / 45.0, 0, 1).astype(np.float32)[:, :, np.newaxis]
        blur_s = cv.resize(filled_full, (canvas_w // 4, canvas_h // 4),
                           interpolation=cv.INTER_AREA)
        blur_s = cv.GaussianBlur(blur_s, (41, 41), 0)
        blur = cv.resize(blur_s, (canvas_w, canvas_h),
                         interpolation=cv.INTER_LINEAR)
        del blur_s
        bg = bg * (1 - t) + blur.astype(np.float32) * t
        del blur, t

        # Heavy blur (60px+)
        t = np.clip((dist_full - 60) / 90.0, 0, 1).astype(np.float32)[:, :, np.newaxis]
        blur_s = cv.resize(filled_full, (canvas_w // 8, canvas_h // 8),
                           interpolation=cv.INTER_AREA)
        blur_s = cv.GaussianBlur(blur_s, (61, 61), 0)
        blur = cv.resize(blur_s, (canvas_w, canvas_h),
                         interpolation=cv.INTER_LINEAR)
        del blur_s
        bg = bg * (1 - t) + blur.astype(np.float32) * t
        del blur, t
        del filled_full, dist_full

        # Feathered composite
        edge_band = painted & ~eroded_bool
        bg_u8 = np.clip(bg, 0, 255).astype(np.uint8)
        panorama[edge_band] = bg_u8[edge_band]
        del bg_u8, edge_band

        alpha = eroded_mask.astype(np.float32) / 255.0
        alpha = cv.GaussianBlur(alpha, (71, 71), 0)
        alpha3 = alpha[:, :, np.newaxis]
        del alpha
        result = panorama.astype(np.float32) * alpha3 + bg * (1.0 - alpha3)
        del bg, alpha3
        panorama = np.clip(result, 0, 255).astype(np.uint8)
        del result

        gap_pct = 100.0 * (~painted).sum() / painted.size
        print(f"  Filled {gap_pct:.1f}% gap pixels with blur padding")

    # ---- Boundary diffusion ----
    _orig_painted = _debug_painted > 0
    _bnd_inner = cv.erode(_debug_painted, np.ones((3, 3), np.uint8), iterations=10) > 0
    _bnd_outer = cv.dilate(_debug_painted, np.ones((3, 3), np.uint8), iterations=10) > 0
    _bnd_band = _bnd_outer & ~_bnd_inner
    shift = 20
    ys_bnd, xs_bnd = np.where(_bnd_band)
    if len(ys_bnd) > 0:
        check_above = np.clip(ys_bnd - 30, 0, canvas_h - 1)
        check_below = np.clip(ys_bnd + 30, 0, canvas_h - 1)
        content_above = _orig_painted[check_above, xs_bnd]
        content_below = _orig_painted[check_below, xs_bnd]
        src_y = ys_bnd.copy()
        src_y[content_above] = np.clip(ys_bnd[content_above] - shift, 0, canvas_h - 1)
        src_y[content_below & ~content_above] = np.clip(
            ys_bnd[content_below & ~content_above] + shift, 0, canvas_h - 1)
        panorama[ys_bnd, xs_bnd] = panorama[src_y, xs_bnd]
        blurred_pano = cv.GaussianBlur(panorama, (31, 31), 0)
        panorama[_bnd_band] = blurred_pano[_bnd_band]

    # Flip horizontally for inside-sphere view
    panorama = panorama[:, ::-1].copy()

    # ---- Panorama transform (rotation-only, centered at origin) ----
    forward = np.array([0.0, 0.0, 1.0])
    z_axis = -forward
    y_axis = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(y_axis, z_axis)

    pano_transform = np.eye(4, dtype=np.float64)
    pano_transform[:3, 0] = x_axis
    pano_transform[:3, 1] = y_axis
    pano_transform[:3, 2] = z_axis
    # No position data — leave translation at origin

    transform_data = {
        "panoramaTransform": pano_transform.flatten(order="F").tolist(),
        "panoramaSize": [canvas_w, canvas_h],
        "subset_indices": [int(i) for i in frame_indices],
        "num_input_images": len(samples),
        "num_stitched_images": n_frames,
    }

    print(f"Total: {canvas_w}x{canvas_h} panorama [{fmt_time(time.time() - t0)}]")
    return panorama, transform_data


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def find_file_by_prefix(directory, prefix, ext):
    """Find a file in directory matching prefix*.ext (e.g. imu-*.json)."""
    import glob
    pattern = os.path.join(directory, f"{prefix}*{ext}")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def process_session(session_dir, output_dir, num_frames=50, reverse=False):
    """Process a single session directory containing video-*.mp4 and imu-*.json."""
    session_name = os.path.basename(os.path.normpath(session_dir))
    print(f"\n{'='*60}")
    print(f"Session: {session_name}")
    print(f"{'='*60}")

    video_path = find_file_by_prefix(session_dir, "video", ".mp4")
    imu_path = find_file_by_prefix(session_dir, "imu", ".json")

    if not video_path:
        print(f"Error: no video-*.mp4 found in {session_dir}")
        return False
    if not imu_path:
        print(f"Error: no imu-*.json found in {session_dir}")
        return False

    print(f"Video: {os.path.basename(video_path)}")
    print(f"IMU:   {os.path.basename(imu_path)}")

    imu_data = load_imu_data(imu_path)
    samples = imu_data['samples']
    sensor_rot_deg = imu_data.get('sensorRotationDegrees', 90)
    print(f"IMU data: {len(samples)} samples, sensorRotation={sensor_rot_deg}\u00b0")

    res_str = imu_data.get('videoResolution', '')
    if 'x' in res_str:
        video_w, video_h = map(int, res_str.split('x'))
    else:
        video_w, video_h = None, None

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return False

    n_video = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)

    if video_w is None:
        video_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Check actual frame dimensions from OpenCV (may auto-apply rotation on Windows)
    ret, test_frame = cap.read()
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    actual_w, actual_h = test_frame.shape[1], test_frame.shape[0]

    if actual_w < actual_h and sensor_rot_deg != 0:
        # OpenCV already rotated frames to portrait — don't apply sensor rotation
        print(f"Detected portrait frames ({actual_w}x{actual_h}): "
              f"OpenCV applied rotation, skipping sensor rotation in matrix")
        device_to_opencv = build_device_to_opencv(0)
        # Intrinsics: swap fx/fy and cx/cy for the rotated frame
        K = build_intrinsics_portrait(imu_data, actual_w, actual_h, sensor_rot_deg)
    else:
        print(f"Video: {n_video} frames, {fps:.1f} fps, {actual_w}x{actual_h}")
        device_to_opencv = build_device_to_opencv(sensor_rot_deg)
        K = build_intrinsics(imu_data, actual_w, actual_h)

    indices = select_frames(samples, num_frames, device_to_opencv)
    print(f"Selected {len(indices)} frames")

    os.makedirs(output_dir, exist_ok=True)
    suffix = "_android_rev" if reverse else "_android"
    output_path = os.path.join(output_dir, f"{session_name}{suffix}.png")

    result = stitch_layered(cap, samples, K, indices, device_to_opencv,
                            reverse=reverse)
    cap.release()

    if result is not None:
        panorama, transform_data = result
        cv.imwrite(output_path, panorama, [cv.IMWRITE_PNG_COMPRESSION, 3])
        print(f"Saved: {output_path}")

        transform_path = os.path.splitext(output_path)[0] + "_transform.json"
        with open(transform_path, "w") as f:
            json.dump(transform_data, f, indent=2)
        print(f"Saved: {transform_path}")
        return True

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Android IMU panorama stitcher (rotation-only projection)"
    )
    parser.add_argument("input", help="Session directory or parent of multiple sessions")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory (default: <input>/output)")
    parser.add_argument("-n", "--num-frames", type=int, default=50,
                        help="Number of frames to select (default: 50)")
    parser.add_argument("--all", action="store_true",
                        help="Process all session subdirectories")
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse painting order")

    args = parser.parse_args()
    input_dir = os.path.normpath(args.input)
    output_dir = args.output or os.path.join(input_dir, "output")

    if args.all:
        sessions = []
        for name in sorted(os.listdir(input_dir)):
            sub = os.path.join(input_dir, name)
            if os.path.isdir(sub) and find_file_by_prefix(sub, "imu", ".json"):
                sessions.append(sub)

        if not sessions:
            print(f"No Android sessions found in {input_dir}")
            sys.exit(1)

        print(f"Found {len(sessions)} sessions")
        for session_dir in sessions:
            process_session(session_dir, output_dir, args.num_frames, args.reverse)
    else:
        if not os.path.isdir(input_dir):
            print(f"Error: {input_dir} is not a directory")
            sys.exit(1)
        process_session(input_dir, output_dir, args.num_frames, args.reverse)


if __name__ == "__main__":
    main()
