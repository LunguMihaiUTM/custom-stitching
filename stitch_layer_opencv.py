"""OpenCV detail pipeline stitcher — uses pre-known camera rotations.

Skips feature detection, camera estimation, and bundle adjustment.
Uses OpenCV's spherical warper, DpColorGrad seam finder, exposure
compensation, and MultiBand blender.

Supports --no-pivot (rotation-only, default) and pivot correction
(parallax-corrected projection using estimated orbit center).

Usage:
    python stitch_layer_opencv.py <session_dir> [-o output_dir] [-n num_frames]
    python stitch_layer_opencv.py C:/Users/lungu/Desktop/sessions/bathroom
    python stitch_layer_opencv.py C:/Users/lungu/Desktop/sessions --all
    python stitch_layer_opencv.py C:/Users/lungu/Desktop/sessions/bathroom --pivot
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2 as cv
import numpy as np


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ---------------------------------------------------------------------------
# Data loading
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


# ---------------------------------------------------------------------------
# Equirectangular helpers (for pivot mode)
# ---------------------------------------------------------------------------

def equirect_to_ray(lon, lat):
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    return np.stack([x, y, z], axis=-1)


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


def _project_col_range(frame, R_w2c, K, cam_offset, scene_distance,
                       cols, rows, lon_full, lat_full):
    """Project frame onto a contiguous column range. Returns (warped, mask) or (None, None)."""
    img_h, img_w = frame.shape[:2]

    sub_lon = lon_full[cols]
    sub_lat = lat_full[rows]
    sub_lon_grid, sub_lat_grid = np.meshgrid(sub_lon, sub_lat)
    sub_rays = equirect_to_ray(sub_lon_grid, sub_lat_grid)
    sub_h, sub_w = sub_rays.shape[:2]
    rays_flat = sub_rays.reshape(-1, 3).astype(np.float32)

    # Parallax-corrected projection
    scene_pts = scene_distance * rays_flat.T - cam_offset[:, np.newaxis]
    cam_coords = R_w2c @ scene_pts

    z = cam_coords[2, :]
    valid = z > 0

    u = np.full(len(rays_flat), -1.0, dtype=np.float32)
    v = np.full(len(rays_flat), -1.0, dtype=np.float32)

    u[valid] = cam_coords[0, valid] / z[valid] * K[0, 0] + K[0, 2]
    v[valid] = cam_coords[1, valid] / z[valid] * K[1, 1] + K[1, 2]

    in_bounds = valid & (u >= 0) & (u < img_w - 1) & (v >= 0) & (v < img_h - 1)

    if not np.any(in_bounds):
        return None, None

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

    pixels = np.clip(interp, 0, 255).astype(np.uint8)

    warped = np.zeros((sub_h, sub_w, 3), dtype=np.uint8)
    mask = np.zeros((sub_h, sub_w), dtype=np.uint8)

    flat_idx = np.where(in_bounds)[0]
    sr = flat_idx // sub_w
    sc = flat_idx % sub_w
    warped[sr, sc] = pixels
    mask[sr, sc] = 255

    return warped, mask


def project_frame_to_equirect(frame, R_w2c, R_c2w, K, cam_offset,
                               scene_distance, canvas_w, canvas_h,
                               lon_full, lat_full):
    """Project a single frame onto equirectangular canvas with parallax correction.

    Returns list of (warped_image, mask, corner) tuples.
    Wrapping frames are split into two patches (right edge + left edge).
    """
    img_h, img_w = frame.shape[:2]

    bounds = frame_canvas_bounds(R_c2w, K, img_w, img_h, canvas_w, canvas_h)
    if bounds is None:
        return []

    col_start, col_end, row_start, row_end, wraps = bounds
    rows = np.arange(row_start, row_end + 1)

    if len(rows) == 0:
        return []

    results = []

    if wraps:
        # Split into two contiguous patches
        # Right patch: col_start -> end of canvas
        cols_right = np.arange(col_start, canvas_w)
        if len(cols_right) > 0:
            w, m = _project_col_range(frame, R_w2c, K, cam_offset,
                                       scene_distance, cols_right, rows,
                                       lon_full, lat_full)
            if w is not None:
                results.append((w, m, (int(col_start), row_start)))

        # Left patch: 0 -> col_end
        cols_left = np.arange(0, col_end + 1)
        if len(cols_left) > 0:
            w, m = _project_col_range(frame, R_w2c, K, cam_offset,
                                       scene_distance, cols_left, rows,
                                       lon_full, lat_full)
            if w is not None:
                results.append((w, m, (0, row_start)))
    else:
        cols = np.arange(col_start, col_end + 1)
        if len(cols) > 0:
            w, m = _project_col_range(frame, R_w2c, K, cam_offset,
                                       scene_distance, cols, rows,
                                       lon_full, lat_full)
            if w is not None:
                results.append((w, m, (int(col_start), row_start)))

    return results


# ---------------------------------------------------------------------------
# Frame selection (same as stitch_layer.py)
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
# OpenCV detail pipeline stitcher
# ---------------------------------------------------------------------------

def stitch_opencv(cap, ar_data, frame_indices, seam_scale=None,
                  blend_strength=5, use_pivot=False, scene_distance=2.0):
    """Stitch using OpenCV detail pipeline with pre-known cameras.

    Pipeline: warp → exposure compensate → seam find → blend
    Skips: feature detection, matching, camera estimation, bundle adjustment.

    When use_pivot=True, uses custom parallax-corrected equirectangular
    projection instead of OpenCV's rotation-only spherical warper.
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

    # Get frame dimensions
    test_frame = extract_frame(cap, frame_indices[0])
    img_h, img_w = test_frame.shape[:2]
    print(f"Frame size: {img_w}x{img_h}")

    # Extract all frames
    print(f"Extracting {n_frames} frames...")
    frames = []
    for idx in frame_indices:
        f = extract_frame(cap, idx)
        if f is not None:
            frames.append(f)
        else:
            frames.append(np.zeros((img_h, img_w, 3), dtype=np.uint8))

    # Pivot estimation
    if use_pivot:
        pivot, orbit_radius = estimate_pivot(cam_positions)
        cam_offsets = cam_positions - pivot
        print(f"Pivot: ({pivot[0]:.2f}, {pivot[1]:.2f}, {pivot[2]:.2f}), "
              f"orbit radius: {orbit_radius:.3f}m, "
              f"scene distance: {scene_distance}m")
    else:
        cam_offsets = np.zeros_like(cam_positions)
        print("Pivot correction DISABLED (rotation-only mode)")

    if use_pivot:
        # === PIVOT MODE: Custom equirectangular projection ===
        panorama = _stitch_pivot_mode(
            frames, Ks, R_w2cs, R_c2ws, cam_offsets,
            scene_distance, img_w, img_h, n_frames,
            seam_scale, blend_strength)
    else:
        # === ROTATION-ONLY MODE: OpenCV spherical warper ===
        panorama = _stitch_rotation_mode(
            frames, Ks, R_w2cs, img_w, img_h, n_frames,
            seam_scale, blend_strength)

    if panorama is None:
        return None

    # Build transform data
    centroid = cam_positions.mean(axis=0)
    forward = np.array([0.0, 0.0, 1.0])
    z_axis = -forward
    y_axis = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(y_axis, z_axis)

    pano_transform = np.eye(4, dtype=np.float64)
    pano_transform[:3, 0] = x_axis
    pano_transform[:3, 1] = y_axis
    pano_transform[:3, 2] = z_axis
    pano_transform[:3, 3] = centroid

    transform_data = {
        "panoramaTransform": pano_transform.flatten(order="F").tolist(),
        "panoramaSize": [panorama.shape[1], panorama.shape[0]],
        "subset_indices": [int(i) for i in frame_indices],
        "num_input_images": len(ar_data),
        "num_stitched_images": n_frames,
    }

    print(f"Total: {panorama.shape[1]}x{panorama.shape[0]} panorama "
          f"[{fmt_time(time.time() - t0)}]")
    return panorama, transform_data


def _stitch_pivot_mode(frames, Ks, R_w2cs, R_c2ws, cam_offsets,
                       scene_distance, img_w, img_h, n_frames,
                       seam_scale, blend_strength):
    """Pivot-corrected stitching: custom projection + OpenCV seam/blend."""
    # Canvas size from average focal length
    avg_focal = np.mean([(K[0, 0] + K[1, 1]) / 2 for K in Ks])
    canvas_w = int(round(2 * np.pi * avg_focal))
    canvas_h = canvas_w // 2
    print(f"Canvas: {canvas_w}x{canvas_h}, avg focal={avg_focal:.1f}")

    # Precompute lon/lat arrays
    xs = np.arange(canvas_w, dtype=np.float64)
    ys = np.arange(canvas_h, dtype=np.float64)
    lon_full = (xs / canvas_w - 0.5) * 2 * np.pi
    lat_full = (0.5 - ys / canvas_h) * np.pi

    # Auto seam_scale
    if seam_scale is None:
        seam_scale = min(1.0, np.sqrt(0.5e6 / (img_h * img_w)))
        seam_scale = max(seam_scale, 0.2)
    print(f"Seam scale: {seam_scale:.3f}")

    # ------------------------------------------------------------------
    # Step 1: Project all frames at seam scale
    # ------------------------------------------------------------------
    print(f"Projecting {n_frames} frames at seam scale (pivot mode)...")
    corners_seam = []
    sizes_seam = []
    images_warped_seam = []
    masks_warped_seam = []
    # Map from patch index back to frame index (for seam mask reuse)
    patch_to_frame = []

    # Seam-scale canvas
    s_canvas_w = max(1, int(canvas_w * seam_scale))
    s_canvas_h = max(1, int(canvas_h * seam_scale))
    s_lon_full = (np.arange(s_canvas_w, dtype=np.float64) / s_canvas_w - 0.5) * 2 * np.pi
    s_lat_full = (0.5 - np.arange(s_canvas_h, dtype=np.float64) / s_canvas_h) * np.pi

    for i in range(n_frames):
        frame_small = cv.resize(frames[i], None, fx=seam_scale, fy=seam_scale,
                                interpolation=cv.INTER_LINEAR_EXACT)

        # Scale intrinsics for seam resolution
        K_seam = Ks[i].copy()
        K_seam[0, 0] *= seam_scale
        K_seam[0, 2] *= seam_scale
        K_seam[1, 1] *= seam_scale
        K_seam[1, 2] *= seam_scale

        patches = project_frame_to_equirect(
            frame_small, R_w2cs[i], R_c2ws[i], K_seam, cam_offsets[i],
            scene_distance, s_canvas_w, s_canvas_h, s_lon_full, s_lat_full)

        if not patches:
            # Empty frame — create a tiny dummy
            corners_seam.append((0, 0))
            sizes_seam.append((1, 1))
            images_warped_seam.append(np.zeros((1, 1, 3), dtype=np.uint8))
            masks_warped_seam.append(np.zeros((1, 1), dtype=np.uint8))
            patch_to_frame.append(i)
        else:
            for warped, mask, corner in patches:
                corners_seam.append(corner)
                sizes_seam.append((warped.shape[1], warped.shape[0]))
                images_warped_seam.append(warped)
                masks_warped_seam.append(mask)
                patch_to_frame.append(i)

        if (i + 1) % 10 == 0:
            print(f"  Projected (seam): {i + 1}/{n_frames}")

    n_patches = len(corners_seam)
    print(f"  {n_patches} patches from {n_frames} frames "
          f"({n_patches - n_frames} from wrapping)")

    # ------------------------------------------------------------------
    # Step 2: Exposure compensation
    # ------------------------------------------------------------------
    print("Exposure compensation: DISABLED (testing)")
    compensator = cv.detail.ExposureCompensator_createDefault(
        cv.detail.ExposureCompensator_NO)
    compensator.feed(corners=corners_seam,
                     images=images_warped_seam,
                     masks=masks_warped_seam)

    # ------------------------------------------------------------------
    # Step 3: Seam finding
    # ------------------------------------------------------------------
    print("Finding seams (DpColorGrad)...")
    t_seam = time.time()
    images_warped_f = [img.astype(np.float32) for img in images_warped_seam]

    seam_finder = cv.detail_DpSeamFinder("COLOR_GRAD")
    masks_warped_seam = list(seam_finder.find(
        images_warped_f, corners_seam, masks_warped_seam))

    for i in range(len(masks_warped_seam)):
        if not isinstance(masks_warped_seam[i], np.ndarray):
            masks_warped_seam[i] = masks_warped_seam[i].get()

    print(f"  Seams found [{fmt_time(time.time() - t_seam)}]")

    del images_warped_seam, images_warped_f

    # ------------------------------------------------------------------
    # Step 4: Project full-res, apply seam masks, blend
    # ------------------------------------------------------------------
    print(f"Projecting {n_frames} frames at full resolution (pivot mode)...")

    # First pass: compute layout for blender
    all_full_patches = []  # list of lists of (warped, mask, corner)
    corners_full = []
    sizes_full = []

    for i in range(n_frames):
        patches = project_frame_to_equirect(
            frames[i], R_w2cs[i], R_c2ws[i], Ks[i], cam_offsets[i],
            scene_distance, canvas_w, canvas_h, lon_full, lat_full)
        all_full_patches.append(patches)
        if not patches:
            corners_full.append((0, 0))
            sizes_full.append((1, 1))
        else:
            for warped, mask, corner in patches:
                corners_full.append(corner)
                sizes_full.append((warped.shape[1], warped.shape[0]))

    dst_roi = cv.detail.resultRoi(corners=corners_full, sizes=sizes_full)
    print(f"Panorama canvas: {dst_roi[2]}x{dst_roi[3]} "
          f"(offset: {dst_roi[0]}, {dst_roi[1]})")

    blend_width = np.sqrt(dst_roi[2] * dst_roi[3]) * blend_strength / 100.0
    num_bands = max(1, int(np.log(blend_width) / np.log(2.0)))
    print(f"MultiBand blender: {num_bands} bands, blend_width={blend_width:.1f}")

    blender = cv.detail_MultiBandBlender(0)
    blender.setNumBands(num_bands)
    blender.prepare(dst_roi)

    # Feed patches to blender, matching seam masks by patch index
    patch_idx = 0
    for i in range(n_frames):
        patches = all_full_patches[i]
        if not patches:
            patch_idx += 1  # skip the dummy
            continue

        for warped, mask, corner in patches:
            # Upscale seam mask to full-res patch size
            if patch_idx < len(masks_warped_seam):
                seam_mask = cv.resize(
                    masks_warped_seam[patch_idx],
                    (mask.shape[1], mask.shape[0]),
                    interpolation=cv.INTER_LINEAR_EXACT)
                mask = cv.bitwise_and(mask, seam_mask)

            compensator.apply(patch_idx, corner, warped, mask)

            image_warped_s = warped.astype(np.int16)
            blender.feed(cv.UMat(image_warped_s), mask, corner)
            patch_idx += 1

        if (i + 1) % 10 == 0:
            print(f"  Composited: {i + 1}/{n_frames}")

    # ------------------------------------------------------------------
    # Step 5: Blend
    # ------------------------------------------------------------------
    print("Blending...")
    result, result_mask = blender.blend(None, None)
    if not isinstance(result, np.ndarray):
        result = result.get()

    panorama = np.clip(result, 0, 255).astype(np.uint8)

    # Flip horizontally for inside-sphere view (panorama viewer convention)
    panorama = panorama[:, ::-1].copy()

    return panorama


def _stitch_rotation_mode(frames, Ks, R_w2cs, img_w, img_h, n_frames,
                          seam_scale, blend_strength):
    """Rotation-only stitching using OpenCV's spherical warper."""
    # Build CameraParams
    print("Building camera params...")
    cameras = []
    for K, R_w2c in zip(Ks, R_w2cs):
        cam = cv.detail_CameraParams()
        cam.focal = float(K[0, 0])
        cam.ppx = float(K[0, 2])
        cam.ppy = float(K[1, 2])
        cam.aspect = float(K[1, 1] / K[0, 0])
        cam.R = R_w2c.T.astype(np.float32)
        cam.t = np.zeros((3, 1), dtype=np.float64)
        cameras.append(cam)

    # Median focal for warper scale
    focals = sorted(cam.focal for cam in cameras)
    mid = len(focals) // 2
    if len(focals) % 2 == 1:
        warped_image_scale = focals[mid]
    else:
        warped_image_scale = (focals[mid] + focals[mid - 1]) / 2.0
    print(f"Warped image scale (focal): {warped_image_scale:.1f}")

    if seam_scale is None:
        seam_scale = min(1.0, np.sqrt(0.5e6 / (img_h * img_w)))
        seam_scale = max(seam_scale, 0.2)
    print(f"Seam scale: {seam_scale:.3f}")

    seam_work_aspect = seam_scale

    # Warp at seam scale
    print(f"Warping {n_frames} frames at seam scale...")
    warper_seam = cv.PyRotationWarper("spherical",
                                      warped_image_scale * seam_work_aspect)

    corners_seam = []
    sizes_seam = []
    images_warped_seam = []
    masks_warped_seam = []

    for i, (frame, cam) in enumerate(zip(frames, cameras)):
        img = cv.resize(frame, None, fx=seam_scale, fy=seam_scale,
                        interpolation=cv.INTER_LINEAR_EXACT)

        K = cam.K().astype(np.float32)
        K[0, 0] *= seam_work_aspect
        K[0, 2] *= seam_work_aspect
        K[1, 1] *= seam_work_aspect
        K[1, 2] *= seam_work_aspect
        R = cam.R.astype(np.float32)

        corner, img_wp = warper_seam.warp(
            img, K, R, cv.INTER_LINEAR, cv.BORDER_REFLECT)

        mask_src = np.full((img.shape[0], img.shape[1]), 255, np.uint8)
        _, mask_wp = warper_seam.warp(
            mask_src, K, R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        if not isinstance(mask_wp, np.ndarray):
            mask_wp = mask_wp.get()

        corners_seam.append(corner)
        sizes_seam.append((img_wp.shape[1], img_wp.shape[0]))
        images_warped_seam.append(img_wp)
        masks_warped_seam.append(mask_wp)

        if (i + 1) % 10 == 0:
            print(f"  Warped (seam): {i + 1}/{n_frames}")

    # Exposure compensation
    print("Exposure compensation: DISABLED (testing)")
    compensator = cv.detail.ExposureCompensator_createDefault(
        cv.detail.ExposureCompensator_NO)
    compensator.feed(corners=corners_seam,
                     images=images_warped_seam,
                     masks=masks_warped_seam)

    # Seam finding
    print("Finding seams (DpColorGrad)...")
    t_seam = time.time()
    images_warped_f = [img.astype(np.float32) for img in images_warped_seam]

    seam_finder = cv.detail_DpSeamFinder("COLOR_GRAD")
    masks_warped_seam = list(seam_finder.find(
        images_warped_f, corners_seam, masks_warped_seam))

    for i in range(len(masks_warped_seam)):
        if not isinstance(masks_warped_seam[i], np.ndarray):
            masks_warped_seam[i] = masks_warped_seam[i].get()

    print(f"  Seams found [{fmt_time(time.time() - t_seam)}]")
    del images_warped_seam, images_warped_f

    # Full-res layout
    print("Computing full-res layout...")
    compose_warper = cv.PyRotationWarper("spherical", warped_image_scale)

    corners_full = []
    sizes_full = []
    for cam in cameras:
        K = cam.K().astype(np.float32)
        R = cam.R.astype(np.float32)
        roi = compose_warper.warpRoi((img_w, img_h), K, R)
        corners_full.append((roi[0], roi[1]))
        sizes_full.append((roi[2], roi[3]))

    dst_roi = cv.detail.resultRoi(corners=corners_full, sizes=sizes_full)
    print(f"Panorama canvas: {dst_roi[2]}x{dst_roi[3]} "
          f"(offset: {dst_roi[0]}, {dst_roi[1]})")

    blend_width = np.sqrt(dst_roi[2] * dst_roi[3]) * blend_strength / 100.0
    num_bands = max(1, int(np.log(blend_width) / np.log(2.0)))
    print(f"MultiBand blender: {num_bands} bands, blend_width={blend_width:.1f}")

    blender = cv.detail_MultiBandBlender(0)
    blender.setNumBands(num_bands)
    blender.prepare(dst_roi)

    # Warp full-res, compensate, feed blender
    print(f"Compositing {n_frames} frames at full resolution...")
    for i, (frame, cam) in enumerate(zip(frames, cameras)):
        K = cam.K().astype(np.float32)
        R = cam.R.astype(np.float32)

        corner, image_warped = compose_warper.warp(
            frame, K, R, cv.INTER_CUBIC, cv.BORDER_REFLECT)

        mask_src = np.full((img_h, img_w), 255, np.uint8)
        _, mask_warped = compose_warper.warp(
            mask_src, K, R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        if not isinstance(mask_warped, np.ndarray):
            mask_warped = mask_warped.get()

        seam_mask = cv.resize(
            masks_warped_seam[i],
            (mask_warped.shape[1], mask_warped.shape[0]),
            interpolation=cv.INTER_LINEAR_EXACT)
        mask_warped = cv.bitwise_and(mask_warped, seam_mask)

        compensator.apply(i, corner, image_warped, mask_warped)

        image_warped_s = image_warped.astype(np.int16)
        blender.feed(cv.UMat(image_warped_s), mask_warped, corner)

        if (i + 1) % 10 == 0:
            print(f"  Composited: {i + 1}/{n_frames}")

    # Blend
    print("Blending...")
    result, result_mask = blender.blend(None, None)
    if not isinstance(result, np.ndarray):
        result = result.get()

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_session(session_dir, output_dir, num_frames=50,
                    use_pivot=False, scene_distance=2.0):
    session_name = os.path.basename(os.path.normpath(session_dir))
    print(f"\n{'='*60}")
    print(f"Session: {session_name}")
    print(f"{'='*60}")

    # Support three directory layouts
    ar_path = os.path.join(session_dir, "ar_data.json")
    video_path = os.path.join(session_dir, "video.mp4")

    if not os.path.exists(ar_path):
        matches = glob.glob(os.path.join(session_dir, "ardata", "ardata-*.json"))
        if matches:
            ar_path = matches[0]
    if not os.path.exists(ar_path):
        matches = glob.glob(os.path.join(session_dir, "ardata-*.json"))
        if matches:
            ar_path = matches[0]
    if not os.path.exists(video_path):
        matches = glob.glob(os.path.join(session_dir, "media", "video-*.mp4"))
        if matches:
            video_path = matches[0]
    if not os.path.exists(video_path):
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
    output_path = os.path.join(output_dir, f"{session_name}_opencv.jpg")

    result = stitch_opencv(cap, ar_data, indices,
                           use_pivot=use_pivot,
                           scene_distance=scene_distance)
    cap.release()

    if result is not None:
        panorama, transform_data = result
        cv.imwrite(output_path, panorama, [cv.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved: {output_path}")

        transform_path = os.path.splitext(output_path)[0] + "_transform.json"
        with open(transform_path, "w") as f:
            json.dump(transform_data, f, indent=2)
        print(f"Saved: {transform_path}")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="OpenCV detail pipeline stitcher with pre-known cameras."
    )
    parser.add_argument("input", help="Session directory or parent of multiple sessions")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory (default: <input>/output)")
    parser.add_argument("-n", "--num-frames", type=int, default=50,
                        help="Number of frames to select (default: 50)")
    parser.add_argument("--all", action="store_true",
                        help="Process all session subdirectories")
    parser.add_argument("--pivot", action="store_true",
                        help="Enable pivot correction (parallax-corrected projection)")
    parser.add_argument("-d", "--scene-distance", type=float, default=2.0,
                        help="Estimated scene distance in meters (default: 2.0, used with --pivot)")

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
                               args.pivot, args.scene_distance):
                ok += 1
        print(f"\nDone: {ok}/{len(sessions)} sessions processed")
    else:
        if not os.path.exists(os.path.join(input_dir, "ar_data.json")) and \
           not os.path.isdir(os.path.join(input_dir, "ardata")):
            print(f"Error: {input_dir} does not contain ar_data.json or ardata/")
            sys.exit(1)
        process_session(input_dir, output_dir, args.num_frames,
                        args.pivot, args.scene_distance)


if __name__ == "__main__":
    main()
