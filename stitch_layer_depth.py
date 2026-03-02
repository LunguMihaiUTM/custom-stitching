"""Layered panorama stitcher - projects frames onto equirectangular canvas
in sequence, each frame painting over the previous ones.

Usage:
    python stitch_layer.py <session_dir> [-o output_dir] [-n num_frames]
    python stitch_layer.py C:/Users/lungu/Desktop/sessions/bathroom
    python stitch_layer.py C:/Users/lungu/Desktop/sessions --all
    python stitch_layer.py C:/Users/lungu/Desktop/sessions/bathroom --reverse
"""

import argparse
import json
import os
import sys
import time

import cv2 as cv
import numpy as np
import onnxruntime as ort

from exposure import compute_gains, apply_gain, equalize_brightness
from blending import multiband_blend

# ---------------------------------------------------------------------------
# Depth estimation (Depth Anything V2 Small, ONNX)
# ---------------------------------------------------------------------------
_depth_session = None
_depth_input_name = None
_depth_output_name = None
_DEPTH_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DEPTH_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "models", "depth_anything_v2_vits.onnx")

def _init_depth_model():
    global _depth_session, _depth_input_name, _depth_output_name
    if _depth_session is not None:
        return
    print(f"Loading depth model: {MODEL_PATH}")
    _depth_session = ort.InferenceSession(
        MODEL_PATH, providers=["CPUExecutionProvider"])
    _depth_input_name = _depth_session.get_inputs()[0].name
    _depth_output_name = _depth_session.get_outputs()[0].name

def estimate_depth(bgr_frame):
    """Returns float32 (H,W) in [0,1], larger=closer."""
    _init_depth_model()
    orig_h, orig_w = bgr_frame.shape[:2]
    rgb = cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB)
    resized = cv.resize(rgb, (518, 518), interpolation=cv.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    img = (img - _DEPTH_MEAN) / _DEPTH_STD
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    depth = _depth_session.run(
        [_depth_output_name], {_depth_input_name: img})[0].squeeze()
    depth = cv.resize(depth, (orig_w, orig_h), interpolation=cv.INTER_LINEAR)
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    return depth


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ---------------------------------------------------------------------------
# Data loading (shared with stitch.py)
# ---------------------------------------------------------------------------

def load_ar_data(path):
    with open(path) as f:
        return json.load(f)


def arkit_to_matrix(camera_transform):
    return np.array(camera_transform, dtype=np.float64).reshape(4, 4, order="F")


def arkit_intrinsics(intrinsics):
    return np.array(intrinsics, dtype=np.float64).reshape(3, 3, order="F")


def extract_frame(cap, frame_idx):
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


# ---------------------------------------------------------------------------
# Coordinate conventions
# ---------------------------------------------------------------------------

AXIS_FLIP = np.diag([1.0, -1.0, -1.0])


def get_camera_rotation(ar_entry):
    T = arkit_to_matrix(ar_entry["cameraTransform"])
    R_arkit = T[:3, :3]
    R_world_to_cam = AXIS_FLIP @ R_arkit.T
    return R_world_to_cam


def get_camera_forward_world(ar_entry):
    T = arkit_to_matrix(ar_entry["cameraTransform"])
    R = T[:3, :3]
    return -R[:, 2]


def estimate_pivot(cam_positions):
    px = cam_positions[:, 0]
    pz = cam_positions[:, 2]
    cy = cam_positions[:, 1].mean()

    A = np.column_stack([2 * px, 2 * pz, np.ones(len(px))])
    b = px**2 + pz**2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cz, c = result
    orbit_radius = float(np.sqrt(c + cx**2 + cz**2))

    pivot = np.array([cx, cy, cz])

    radii = np.sqrt((px - cx)**2 + (pz - cz)**2)
    std_dev = float(np.std(radii))
    print(f"Circle fit: radius={orbit_radius:.3f}m, std={std_dev:.3f}m "
          f"(deviation {std_dev/orbit_radius*100:.1f}%)")

    return pivot, orbit_radius


def rotation_to_ypr(R):
    """Decompose a 3x3 rotation matrix into yaw, pitch, roll (radians).
    Convention: R = Ry(yaw) @ Rx(pitch) @ Rz(roll)
    """
    pitch = np.arcsin(np.clip(-R[1, 2], -1, 1))
    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) > 1e-6:
        yaw = np.arctan2(R[0, 2], R[2, 2])
        roll = np.arctan2(R[1, 0], R[1, 1])
    else:
        yaw = np.arctan2(-R[2, 0], R[0, 0])
        roll = 0.0
    return yaw, pitch, roll


def ypr_to_rotation(yaw, pitch, roll):
    """Build rotation matrix from yaw, pitch, roll (radians).
    R = Ry(yaw) @ Rx(pitch) @ Rz(roll)
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def select_frames(ar_data, num_frames=50):
    valid = []
    for i, entry in enumerate(ar_data):
        if entry.get("trackingState") != "normal":
            continue
        fwd = get_camera_forward_world(entry)
        yaw = np.arctan2(fwd[0], fwd[2])
        valid.append((i, yaw))

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


# ---------------------------------------------------------------------------
# Layered stitching
# ---------------------------------------------------------------------------

def frame_canvas_bounds(R_c2w, K, img_w, img_h, canvas_w, canvas_h):
    """Compute the canvas pixel bounding box where a frame projects.

    Projects the 4 image corners + edge midpoints back to world rays,
    converts to lon/lat, and returns the pixel range on the canvas.
    Returns (col_start, col_end, row_start, row_end) or None if degenerate.
    Also returns whether the frame wraps around the lon=±pi boundary.
    """
    # Sample points along frame edges (corners + midpoints for better coverage)
    border_u = [0, img_w/2, img_w-1, img_w-1, img_w-1, img_w/2, 0, 0]
    border_v = [0, 0, 0, img_h/2, img_h-1, img_h-1, img_h-1, img_h/2]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    lons = []
    lats = []
    for bu, bv in zip(border_u, border_v):
        # Pixel -> camera ray (OpenCV convention)
        cam_ray = np.array([(bu - cx) / fx, (bv - cy) / fy, 1.0])
        cam_ray /= np.linalg.norm(cam_ray)
        # Camera ray -> world ray: R_c2w = R_w2c.T
        world_ray = R_c2w @ cam_ray
        lon = np.arctan2(world_ray[0], world_ray[2])
        lat = np.arcsin(np.clip(world_ray[1], -1, 1))
        lons.append(lon)
        lats.append(lat)

    # Check for wrapping around ±pi
    lons = np.array(lons)
    lats = np.array(lats)

    # If lon range spans more than pi, it likely wraps
    lon_min, lon_max = lons.min(), lons.max()
    wraps = (lon_max - lon_min) > np.pi

    margin = 0.05  # ~3 degrees extra margin

    if wraps:
        # Frame straddles the ±pi boundary
        # Split into positive and negative lons
        pos_lons = lons[lons >= 0]
        neg_lons = lons[lons < 0]
        lon_start = float(pos_lons.min() - margin) if len(pos_lons) > 0 else np.pi - margin
        lon_end = float(neg_lons.max() + margin) if len(neg_lons) > 0 else -np.pi + margin
    else:
        lon_start = float(lon_min - margin)
        lon_end = float(lon_max + margin)

    lat_min = float(lats.min() - margin)
    lat_max = float(lats.max() + margin)

    # Convert to canvas pixel coordinates
    # lon -> col: col = (lon / (2*pi) + 0.5) * canvas_w
    # lat -> row: row = (0.5 - lat / pi) * canvas_h
    def lon_to_col(l):
        return int(np.clip((l / (2 * np.pi) + 0.5) * canvas_w, 0, canvas_w - 1))

    def lat_to_row(l):
        return int(np.clip((0.5 - l / np.pi) * canvas_h, 0, canvas_h - 1))

    row_start = lat_to_row(lat_max)  # higher lat = smaller row (top)
    row_end = lat_to_row(lat_min)

    if wraps:
        col_start = lon_to_col(lon_start)  # left chunk starts here -> end of canvas
        col_end = lon_to_col(lon_end)       # right chunk 0 -> here
        return col_start, col_end, row_start, row_end, True
    else:
        col_start = lon_to_col(lon_start)
        col_end = lon_to_col(lon_end)
        return col_start, col_end, row_start, row_end, False


def stitch_layered(cap, ar_data, frame_indices, canvas_w=None, canvas_h=None,
                   scene_distance=2.0, reverse=False, no_pivot=False):
    """Project frames onto equirectangular canvas in sequence.

    Each frame is painted on top of the previous ones. Later frames
    overwrite earlier ones in overlapping regions. Only processes the
    canvas region where each frame can project (FOV bounding box).
    """
    t0 = time.time()
    n_frames = len(frame_indices)

    # Gather camera data
    Ks = []
    R_w2cs = []
    R_c2ws = []
    cam_positions = []

    for idx in frame_indices:
        entry = ar_data[idx]
        K = arkit_intrinsics(entry["intrinsics"])
        R_w2c = get_camera_rotation(entry)
        T = arkit_to_matrix(entry["cameraTransform"])
        pos = T[:3, 3]

        Ks.append(K)
        R_w2cs.append(R_w2c)
        R_c2ws.append(R_w2c.T)
        cam_positions.append(pos)

    cam_positions = np.array(cam_positions)

    # --- Raw frame data (before pivot correction) ---
    print(f"\n  === RAW CAMERA DATA (before circle fit) ===")
    print(f"  {'Frame':>5}  {'Yaw':>7}  {'Pitch':>7}  {'fx':>7}  {'fy':>7}  "
          f"{'Pos X':>7}  {'Pos Y':>7}  {'Pos Z':>7}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}")
    for fi, idx in enumerate(frame_indices):
        fwd = get_camera_forward_world(ar_data[idx])
        yaw_d = np.degrees(np.arctan2(fwd[0], fwd[2]))
        pitch_d = np.degrees(np.arcsin(np.clip(fwd[1], -1, 1)))
        pos = cam_positions[fi]
        K = Ks[fi]
        print(f"  {idx:5d}  {yaw_d:7.1f}  {pitch_d:7.1f}  {K[0,0]:7.1f}  {K[1,1]:7.1f}  "
              f"{pos[0]:7.3f}  {pos[1]:7.3f}  {pos[2]:7.3f}")
    print()

    # Estimate pivot
    if no_pivot:
        pivot = np.zeros(3, dtype=np.float64)
        orbit_radius = 0.0
        cam_offsets = np.zeros_like(cam_positions)
        print("Pivot correction DISABLED (rotation-only mode)")
    else:
        pivot, orbit_radius = estimate_pivot(cam_positions)
        cam_offsets = cam_positions - pivot

    # Canvas size from average focal length
    avg_focal = np.mean([(K[0, 0] + K[1, 1]) / 2 for K in Ks])
    if canvas_w is None:
        canvas_w = int(round(2 * np.pi * avg_focal))
    if canvas_h is None:
        canvas_h = canvas_w // 2

    print(f"Pivot: ({pivot[0]:.2f}, {pivot[1]:.2f}, {pivot[2]:.2f}), "
          f"orbit radius: {orbit_radius:.3f}m")
    print(f"Canvas: {canvas_w}x{canvas_h}, avg focal={avg_focal:.1f}")

    # --- After pivot correction ---
    print(f"\n  === AFTER CIRCLE FIT (offsets from pivot) ===")
    print(f"  {'Frame':>5}  {'Yaw':>7}  {'Pitch':>7}  "
          f"{'Off X':>7}  {'Off Y':>7}  {'Off Z':>7}  {'Dist':>6}  {'Dev%':>5}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*5}")
    for fi, idx in enumerate(frame_indices):
        fwd = get_camera_forward_world(ar_data[idx])
        yaw_d = np.degrees(np.arctan2(fwd[0], fwd[2]))
        pitch_d = np.degrees(np.arcsin(np.clip(fwd[1], -1, 1)))
        off = cam_offsets[fi]
        dist = np.linalg.norm(off)
        dev_pct = abs(dist - orbit_radius) / orbit_radius * 100 if orbit_radius > 0 else 0
        print(f"  {idx:5d}  {yaw_d:7.1f}  {pitch_d:7.1f}  "
              f"{off[0]:7.3f}  {off[1]:7.3f}  {off[2]:7.3f}  {dist:6.3f}  {dev_pct:5.1f}")
    print()

    # Get frame dimensions and compute exposure gains
    test_frame = extract_frame(cap, frame_indices[0])
    img_h, img_w = test_frame.shape[:2]
    print(f"Frame size: {img_w}x{img_h}")

    print("Computing exposure gains...")
    all_frames = []
    for idx in frame_indices:
        f = extract_frame(cap, idx)
        if f is not None:
            all_frames.append(f)
    exposure_gains = compute_gains(all_frames)
    del all_frames  # free memory
    gain_values = [sum(g)/3 for g in exposure_gains]
    print(f"Exposure gains: min={min(gain_values):.3f}, max={max(gain_values):.3f}, "
          f"spread={max(gain_values)-min(gain_values):.3f}")

    # Compute per-frame scene distance from depth maps
    print(f"Computing depth maps for {n_frames} frames...")
    t_depth = time.time()
    frame_scene_dist = []
    for fi, idx in enumerate(frame_indices):
        frame = extract_frame(cap, idx)
        if frame is not None:
            depth = estimate_depth(frame)
            med = float(np.median(depth))
        else:
            med = 0.5
        # Convert: depth_rel=1 (close) -> small dist, depth_rel=0 (far) -> large dist
        min_d = scene_distance * 0.5
        max_d = scene_distance * 3.0
        frame_scene_dist.append(max_d - med * (max_d - min_d))
        if (fi + 1) % 10 == 0 or fi == 0:
            print(f"  Depth: {fi + 1}/{n_frames}")
    print(f"  Depth done [{fmt_time(time.time() - t_depth)}]")
    print(f"  Per-frame distance: min={min(frame_scene_dist):.2f}m, "
          f"max={max(frame_scene_dist):.2f}m, "
          f"median={np.median(frame_scene_dist):.2f}m")

    # Precompute lon/lat arrays for the full canvas
    xs = np.arange(canvas_w, dtype=np.float64)
    ys = np.arange(canvas_h, dtype=np.float64)
    lon_full = (xs / canvas_w - 0.5) * 2 * np.pi   # [-pi, pi]
    lat_full = (0.5 - ys / canvas_h) * np.pi        # [pi/2, -pi/2]

    panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    painted_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)  # track painted pixels

    # Compute yaw for each frame
    yaws = []
    for fi in range(n_frames):
        fwd = get_camera_forward_world(ar_data[frame_indices[fi]])
        yaws.append(np.degrees(np.arctan2(fwd[0], fwd[2])))
    yaws = np.array(yaws)
    print(f"Yaw range: {yaws.min():.1f}° to {yaws.max():.1f}° "
          f"(sweep: {yaws.max() - yaws.min():.1f}°)")

    # Sort frames by yaw (lowest to highest) for painting order.
    # This way the seam lands at the ±180° boundary (panorama edge),
    # not in the middle of the image.
    order = list(np.argsort(yaws))
    if reverse:
        order = order[::-1]

    # Average yaw spacing between frames — used for strip width
    avg_yaw_step = 360.0 / n_frames
    strip_half_deg = avg_yaw_step * 1.5  # strip = ~3x average spacing

    print(f"Painting {n_frames} frames sorted by yaw "
          f"(last 5 use {2*strip_half_deg:.1f}° strip)...")

    def paint_frame(fi, count_label, use_strip=False, gap_fill_only=False):
        idx = frame_indices[fi]
        frame = extract_frame(cap, idx)
        if frame is None:
            return

        # Compute bounding box on canvas for this frame's FOV
        bounds = frame_canvas_bounds(R_c2ws[fi], Ks[fi], img_w, img_h,
                                     canvas_w, canvas_h)
        if bounds is None:
            return

        col_start, col_end, row_start, row_end, wraps = bounds

        # Build the column indices for this frame
        if wraps:
            cols = np.concatenate([np.arange(col_start, canvas_w),
                                   np.arange(0, col_end + 1)])
        else:
            cols = np.arange(col_start, col_end + 1)

        # For the last few frames, clip to a narrow strip centered on yaw
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

        # Build sub-grid of lon/lat for this region
        sub_lon = lon_full[cols]
        sub_lat = lat_full[rows]
        sub_lon_grid, sub_lat_grid = np.meshgrid(sub_lon, sub_lat)

        # Convert to rays
        sub_rays = equirect_to_ray(sub_lon_grid, sub_lat_grid)
        sub_h, sub_w = sub_rays.shape[:2]
        rays_flat = sub_rays.reshape(-1, 3).astype(np.float32)

        # Project into this frame's camera
        offset = cam_offsets[fi]
        scene_pts = frame_scene_dist[fi] * rays_flat.T - offset[:, np.newaxis]
        cam_coords = R_w2cs[fi] @ scene_pts

        z = cam_coords[2, :]
        valid = z > 0

        u = np.full(len(rays_flat), -1.0, dtype=np.float32)
        v = np.full(len(rays_flat), -1.0, dtype=np.float32)

        u[valid] = cam_coords[0, valid] / z[valid] * Ks[fi][0, 0] + Ks[fi][0, 2]
        v[valid] = cam_coords[1, valid] / z[valid] * Ks[fi][1, 1] + Ks[fi][1, 2]

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

        # # Apply exposure correction (disabled for testing)
        # interp = apply_gain(interp, exposure_gains[fi])
        pixels = np.clip(interp, 0, 255).astype(np.uint8)

        panorama[canvas_row, canvas_col] = pixels
        painted_mask[canvas_row, canvas_col] = 255

    # Pass 1: all frames with full FOV (good geometry base)
    for count, fi in enumerate(order):
        paint_frame(fi, count_label=count, use_strip=False, gap_fill_only=False)
        if (count + 1) % 10 == 0:
            print(f"  Pass 1: {count + 1}/{n_frames} frames")
    print(f"  Pass 1 done: {n_frames} frames (full FOV)")

    # Pass 2: blend narrow strips onto panorama using multi-band blending.
    # For each frame, render its strip into a patch, then blend it with
    # the existing panorama content using Laplacian pyramid blending.
    # This smooths brightness transitions while keeping details sharp.
    print(f"  Pass 2: blending {n_frames} strips (multi-band)...")
    for count, fi in enumerate(order):
        idx = frame_indices[fi]
        frame = extract_frame(cap, idx)
        if frame is None:
            continue

        # Compute strip column range
        yaw_rad = np.radians(yaws[fi])
        strip_half_rad = np.radians(strip_half_deg)
        lon_lo = yaw_rad - strip_half_rad
        lon_hi = yaw_rad + strip_half_rad
        col_lo = int(np.clip((lon_lo / (2 * np.pi) + 0.5) * canvas_w, 0, canvas_w - 1))
        col_hi = int(np.clip((lon_hi / (2 * np.pi) + 0.5) * canvas_w, 0, canvas_w - 1))

        # Also need row range from FOV bounds
        bounds = frame_canvas_bounds(R_c2ws[fi], Ks[fi], img_w, img_h,
                                     canvas_w, canvas_h)
        if bounds is None:
            continue
        _, _, row_start, row_end, _ = bounds

        # Build column indices for strip
        if col_lo <= col_hi:
            cols = np.arange(col_lo, col_hi + 1)
        else:
            cols = np.concatenate([np.arange(col_lo, canvas_w),
                                   np.arange(0, col_hi + 1)])
        rows = np.arange(row_start, row_end + 1)

        if len(cols) == 0 or len(rows) == 0:
            continue

        # Render this frame's strip into a small patch
        sub_lon = lon_full[cols]
        sub_lat = lat_full[rows]
        sub_lon_grid, sub_lat_grid = np.meshgrid(sub_lon, sub_lat)
        sub_rays = equirect_to_ray(sub_lon_grid, sub_lat_grid)
        sub_h, sub_w = sub_rays.shape[:2]
        rays_flat = sub_rays.reshape(-1, 3).astype(np.float32)

        offset = cam_offsets[fi]
        scene_pts = frame_scene_dist[fi] * rays_flat.T - offset[:, np.newaxis]
        cam_coords = R_w2cs[fi] @ scene_pts
        z = cam_coords[2, :]
        valid = z > 0

        u = np.full(len(rays_flat), -1.0, dtype=np.float32)
        v = np.full(len(rays_flat), -1.0, dtype=np.float32)
        u[valid] = cam_coords[0, valid] / z[valid] * Ks[fi][0, 0] + Ks[fi][0, 2]
        v[valid] = cam_coords[1, valid] / z[valid] * Ks[fi][1, 1] + Ks[fi][1, 2]
        in_bounds = valid & (u >= 0) & (u < img_w - 1) & (v >= 0) & (v < img_h - 1)

        if not np.any(in_bounds):
            continue

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

        # # Apply exposure correction (disabled for testing)
        # interp = apply_gain(interp, exposure_gains[fi])

        # Build the strip patch (new frame content) and mask
        strip_patch = np.zeros((sub_h, sub_w, 3), dtype=np.uint8)
        strip_mask = np.zeros((sub_h, sub_w), dtype=np.float32)

        flat_idx = np.where(in_bounds)[0]
        sr = flat_idx // sub_w
        sc = flat_idx % sub_w
        pixels = np.clip(interp, 0, 255).astype(np.uint8)
        strip_patch[sr, sc] = pixels
        strip_mask[sr, sc] = 1.0

        # Extract the corresponding panorama region
        row_ix = rows[:, np.newaxis]   # (sub_h, 1)
        col_ix = cols[np.newaxis, :]   # (1, sub_w)
        pano_patch = panorama[row_ix, col_ix]  # (sub_h, sub_w, 3)

        # Create a feathered blend mask: full blend in center, fade at edges
        # Horizontal feather based on distance from strip center
        n_cols = len(cols)
        feather_width = max(n_cols // 6, 4)
        feather = np.ones(n_cols, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, feather_width)
        feather[:feather_width] = ramp
        feather[-feather_width:] = ramp[::-1]
        blend_mask = strip_mask * feather[np.newaxis, :]

        # Only blend where both panorama and strip have content
        pano_has_content = np.any(pano_patch > 0, axis=2).astype(np.float32)
        # Where pano is empty, just paste the strip directly
        direct_paste = (blend_mask > 0) & (pano_has_content < 0.5)

        # Multi-band blend where both have content
        if np.any(blend_mask * pano_has_content > 0):
            # Ensure patch dimensions are at least 8x8 for pyramid blending
            if sub_h >= 8 and sub_w >= 8:
                blended = multiband_blend(pano_patch, strip_patch,
                                          blend_mask, levels=4)
            else:
                # Too small for pyramid, do simple alpha blend
                m3 = blend_mask[:, :, np.newaxis]
                blended = (pano_patch.astype(np.float32) * (1 - m3) +
                           strip_patch.astype(np.float32) * m3)
                blended = np.clip(blended, 0, 255).astype(np.uint8)

            # Write blended result back where mask > 0
            update = blend_mask > 0
            for c in range(3):
                ch = panorama[row_ix, col_ix, c]
                ch[update] = blended[:, :, c][update]
                panorama[row_ix, col_ix, c] = ch
            pm = painted_mask[row_ix, col_ix]
            pm[update] = 255
            painted_mask[row_ix, col_ix] = pm

        # Direct paste where panorama was empty
        if np.any(direct_paste):
            for c in range(3):
                ch = panorama[row_ix, col_ix, c]
                ch[direct_paste] = strip_patch[:, :, c][direct_paste]
                panorama[row_ix, col_ix, c] = ch
            pm = painted_mask[row_ix, col_ix]
            pm[direct_paste] = 255
            painted_mask[row_ix, col_ix] = pm

        if (count + 1) % 10 == 0:
            print(f"    {count + 1}/{n_frames} strips blended")
    print(f"  Pass 2 done: {n_frames} strips (multi-band blended)")

    # Save original painted_mask boundary for debug drawing after padding
    _debug_painted = painted_mask.copy()

    # Post-process: fill black gaps with blurred color padding.
    # Uses edge-color propagation so each gap pixel inherits colors from
    # its nearest content boundary (top gets ceiling, bottom gets floor).
    print("Filling gaps with blur padding...")
    painted = painted_mask > 0

    if not painted.all():
        # Strategy: erode the painted mask inward so unreliable dark edge
        # pixels get excluded from content. The fill covers them instead.
        # Then use a wide feather for seamless blending.

        # Step 0: Erode painted mask — cut off dark boundary pixels
        erode_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        eroded_mask = cv.erode(painted_mask, erode_k, iterations=3)
        # The "safe" panorama: only pixels well inside the content boundary
        safe_pano = panorama.copy()
        safe_pano[eroded_mask == 0] = 0  # black out unreliable edge pixels

        # Step 1: Edge-color propagation at reduced scale
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

        # Blend top-down and bottom-up by vertical distance
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
        total[both_inf] = 1  # avoid inf/inf = nan
        w_down = dist_up / total
        w_up_w = dist_down / total
        # Handle edges where only one direction reaches (or neither)
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

        # Horizontal smoothing to remove per-column stripe artifacts
        kw = max(sw // 8, 3) | 1
        smoothed = cv.GaussianBlur(filled_blend.astype(np.float32), (kw, 3), 0)
        smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
        gap_px = ~small_mask
        for c in range(3):
            filled_blend[:, :, c][gap_px] = smoothed[:, :, c][gap_px]

        # Step 2: Upscale filled to full resolution
        filled_full = cv.resize(filled_blend, (canvas_w, canvas_h),
                                interpolation=cv.INTER_LINEAR)
        # Restore safe panorama pixels
        eroded_bool = eroded_mask > 0
        filled_full[eroded_bool] = safe_pano[eroded_bool]

        # Step 3: Distance-based progressive blur
        gap_full = (~eroded_bool).astype(np.uint8)
        dist_full = cv.distanceTransform(gap_full, cv.DIST_L2, 5)

        blur_light = cv.GaussianBlur(filled_full, (31, 31), 0)
        blur_med_s = cv.resize(filled_full, (canvas_w // 4, canvas_h // 4),
                               interpolation=cv.INTER_AREA)
        blur_med_s = cv.GaussianBlur(blur_med_s, (41, 41), 0)
        blur_med = cv.resize(blur_med_s, (canvas_w, canvas_h),
                             interpolation=cv.INTER_LINEAR)
        blur_heavy_s = cv.resize(filled_full, (canvas_w // 8, canvas_h // 8),
                                 interpolation=cv.INTER_AREA)
        blur_heavy_s = cv.GaussianBlur(blur_heavy_s, (61, 61), 0)
        blur_heavy = cv.resize(blur_heavy_s, (canvas_w, canvas_h),
                               interpolation=cv.INTER_LINEAR)

        t1 = np.clip(dist_full / 15.0, 0, 1).astype(np.float32)[:, :, np.newaxis]
        t2 = np.clip((dist_full - 15) / 45.0, 0, 1).astype(np.float32)[:, :, np.newaxis]
        t3 = np.clip((dist_full - 60) / 90.0, 0, 1).astype(np.float32)[:, :, np.newaxis]

        bg = filled_full.astype(np.float32)
        bg = bg * (1 - t1) + blur_light.astype(np.float32) * t1
        bg = bg * (1 - t2) + blur_med.astype(np.float32) * t2
        bg = bg * (1 - t3) + blur_heavy.astype(np.float32) * t3

        # Step 4: Wide feathered alpha composite
        # First, replace dark edge pixels (between eroded and original boundary)
        # in the panorama with fill colors so the feather never hits dark pixels.
        edge_band = painted & ~eroded_bool
        bg_u8 = np.clip(bg, 0, 255).astype(np.uint8)
        panorama[edge_band] = bg_u8[edge_band]

        alpha = eroded_mask.astype(np.float32) / 255.0
        alpha = cv.GaussianBlur(alpha, (71, 71), 0)
        alpha3 = alpha[:, :, np.newaxis]
        result = panorama.astype(np.float32) * alpha3 + bg * (1.0 - alpha3)
        panorama = np.clip(result, 0, 255).astype(np.uint8)

        gap_pct = 100.0 * (~painted).sum() / painted.size
        print(f"  Filled {gap_pct:.1f}% gap pixels with blur padding")

    # Fix boundary: copy content from 20px inside the panorama over the
    # boundary band, then diffuse it.
    _orig_painted = _debug_painted > 0
    _bnd_inner = cv.erode(_debug_painted, np.ones((3, 3), np.uint8), iterations=10) > 0
    _bnd_outer = cv.dilate(_debug_painted, np.ones((3, 3), np.uint8), iterations=10) > 0
    _bnd_band = _bnd_outer & ~_bnd_inner
    shift = 20
    ys, xs = np.where(_bnd_band)
    if len(ys) > 0:
        # For each boundary pixel, check if content is above or below
        check_above = np.clip(ys - 30, 0, canvas_h - 1)
        check_below = np.clip(ys + 30, 0, canvas_h - 1)
        content_above = _orig_painted[check_above, xs]
        content_below = _orig_painted[check_below, xs]
        # Source row: copy from 20px toward the content side
        src_y = ys.copy()
        # Bottom boundary (content above) → copy from 20px above
        src_y[content_above] = np.clip(ys[content_above] - shift, 0, canvas_h - 1)
        # Top boundary (content below) → copy from 20px below
        src_y[content_below & ~content_above] = np.clip(
            ys[content_below & ~content_above] + shift, 0, canvas_h - 1)
        # Overwrite boundary pixels with content from inside
        panorama[ys, xs] = panorama[src_y, xs]
        # Now diffuse the band
        blurred_pano = cv.GaussianBlur(panorama, (31, 31), 0)
        panorama[_bnd_band] = blurred_pano[_bnd_band]

    # Flip horizontally for inside-sphere view (panorama viewer convention)
    panorama = panorama[:, ::-1].copy()

    # Compute panorama transform (4x4 ARKit-convention matrix).
    # Our equirectangular center (lon=0) points at world +Z.
    # ARKit convention: -Z_local = forward, so the rotation is 180° around Y.
    forward = np.array([0.0, 0.0, 1.0])   # panorama center direction
    z_axis = -forward                       # (0, 0, -1) = back direction
    y_axis = np.array([0.0, 1.0, 0.0])     # up
    x_axis = np.cross(y_axis, z_axis)       # (-1, 0, 0) = right in local frame

    centroid = cam_positions.mean(axis=0)

    pano_transform = np.eye(4, dtype=np.float64)
    pano_transform[:3, 0] = x_axis
    pano_transform[:3, 1] = y_axis
    pano_transform[:3, 2] = z_axis
    pano_transform[:3, 3] = centroid

    transform_data = {
        "panoramaTransform": pano_transform.flatten(order="F").tolist(),
        "panoramaSize": [canvas_w, canvas_h],
        "subset_indices": [int(i) for i in frame_indices],
        "num_input_images": len(ar_data),
        "num_stitched_images": n_frames,
    }

    print(f"Total: {canvas_w}x{canvas_h} panorama [{fmt_time(time.time() - t0)}]")
    return panorama, transform_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_session(session_dir, output_dir, num_frames=50, scene_distance=2.0,
                    reverse=False, no_pivot=False):
    session_name = os.path.basename(os.path.normpath(session_dir))
    print(f"\n{'='*60}")
    print(f"Session: {session_name}")
    print(f"{'='*60}")

    # Support three directory layouts:
    #   Flat:     session_dir/ar_data.json + session_dir/video.mp4
    #   Nested:   session_dir/ardata/ardata-*.json + session_dir/media/video-*.mp4
    #   Direct:   session_dir/ardata-*.json + session_dir/video-*.mp4
    import glob
    ar_path = os.path.join(session_dir, "ar_data.json")
    video_path = os.path.join(session_dir, "video.mp4")

    if not os.path.exists(ar_path):
        # Try nested layout: ardata/ardata-*.json
        matches = glob.glob(os.path.join(session_dir, "ardata", "ardata-*.json"))
        if matches:
            ar_path = matches[0]
    if not os.path.exists(ar_path):
        # Try direct layout: ardata-*.json in session dir
        matches = glob.glob(os.path.join(session_dir, "ardata-*.json"))
        if matches:
            ar_path = matches[0]
    if not os.path.exists(video_path):
        # Try nested layout: media/video-*.mp4
        matches = glob.glob(os.path.join(session_dir, "media", "video-*.mp4"))
        if matches:
            video_path = matches[0]
    if not os.path.exists(video_path):
        # Try direct layout: video-*.mp4 in session dir
        matches = glob.glob(os.path.join(session_dir, "video-*.mp4"))
        if matches:
            video_path = matches[0]

    if not os.path.exists(ar_path):
        print(f"Error: no AR data found in {session_dir}")
        return False
    if not os.path.exists(video_path):
        print(f"Error: no video found in {session_dir}")
        return False

    ar_data = load_ar_data(ar_path)
    print(f"AR data: {len(ar_data)} frames")

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return False

    n_video = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)
    print(f"Video: {n_video} frames, {fps:.1f} fps")

    indices = select_frames(ar_data, num_frames)
    print(f"Selected {len(indices)} frames")

    os.makedirs(output_dir, exist_ok=True)
    suffix = "_depth_rev" if reverse else "_depth"
    output_path = os.path.join(output_dir, f"{session_name}{suffix}.jpg")

    result = stitch_layered(cap, ar_data, indices, scene_distance=scene_distance,
                            reverse=reverse, no_pivot=no_pivot)
    cap.release()

    if result is not None:
        panorama, transform_data = result
        cv.imwrite(output_path, panorama, [cv.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved: {output_path}")

        # Save panorama transform JSON alongside the image
        transform_path = os.path.splitext(output_path)[0] + "_transform.json"
        with open(transform_path, "w") as f:
            json.dump(transform_data, f, indent=2)
        print(f"Saved: {transform_path}")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Layered panorama stitcher - frames painted on top of each other."
    )
    parser.add_argument("input", help="Session directory or parent of multiple sessions")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory (default: <input>/output)")
    parser.add_argument("-n", "--num-frames", type=int, default=50,
                        help="Number of frames to select (default: 50)")
    parser.add_argument("--all", action="store_true",
                        help="Process all session subdirectories")
    parser.add_argument("-d", "--scene-distance", type=float, default=2.0,
                        help="Estimated scene distance in meters (default: 2.0)")
    parser.add_argument("--reverse", action="store_true",
                        help="Paint last frame first (first frame ends up on top)")
    parser.add_argument("--no-pivot", action="store_true",
                        help="Disable pivot correction (rotation-only, like Android IMU)")

    args = parser.parse_args()
    input_dir = os.path.normpath(args.input)
    output_dir = args.output or os.path.join(input_dir, "output")

    if args.all:
        sessions = []
        for name in sorted(os.listdir(input_dir)):
            sub = os.path.join(input_dir, name)
            if os.path.isdir(sub) and (
                os.path.exists(os.path.join(sub, "ar_data.json")) or
                os.path.isdir(os.path.join(sub, "ardata")) or
                any(f.startswith("ardata-") and f.endswith(".json")
                    for f in os.listdir(sub))):
                sessions.append(sub)

        if not sessions:
            print(f"No sessions found in {input_dir}")
            sys.exit(1)

        print(f"Found {len(sessions)} sessions")
        ok = 0
        for session_dir in sessions:
            if process_session(session_dir, output_dir, args.num_frames,
                               args.scene_distance, args.reverse, args.no_pivot):
                ok += 1
        print(f"\nDone: {ok}/{len(sessions)} sessions processed")
    else:
        if not os.path.exists(os.path.join(input_dir, "ar_data.json")) and \
           not os.path.isdir(os.path.join(input_dir, "ardata")):
            print(f"Error: {input_dir} does not contain ar_data.json or ardata/")
            sys.exit(1)
        process_session(input_dir, output_dir, args.num_frames,
                        args.scene_distance, args.reverse, args.no_pivot)


if __name__ == "__main__":
    main()
