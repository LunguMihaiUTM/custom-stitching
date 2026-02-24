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
                   scene_distance=2.0, reverse=False):
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

    # Estimate pivot
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

    # Get frame dimensions
    test_frame = extract_frame(cap, frame_indices[0])
    img_h, img_w = test_frame.shape[:2]
    print(f"Frame size: {img_w}x{img_h}")

    # Precompute lon/lat arrays for the full canvas
    xs = np.arange(canvas_w, dtype=np.float64)
    ys = np.arange(canvas_h, dtype=np.float64)
    lon_full = (xs / canvas_w - 0.5) * 2 * np.pi   # [-pi, pi]
    lat_full = (0.5 - ys / canvas_h) * np.pi        # [pi/2, -pi/2]

    panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    painted = np.zeros((canvas_h, canvas_w), dtype=bool)

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
        scene_pts = scene_distance * rays_flat.T - offset[:, np.newaxis]
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

        pixels = np.clip(interp, 0, 255).astype(np.uint8)

        if gap_fill_only:
            not_painted = ~painted[canvas_row, canvas_col]
            if np.any(not_painted):
                panorama[canvas_row[not_painted], canvas_col[not_painted]] = pixels[not_painted]
                painted[canvas_row[not_painted], canvas_col[not_painted]] = True
        else:
            panorama[canvas_row, canvas_col] = pixels
            painted[canvas_row, canvas_col] = True

    # Pass 1: all frames with full FOV (good geometry base)
    for count, fi in enumerate(order):
        paint_frame(fi, count_label=count, use_strip=False, gap_fill_only=False)
        if (count + 1) % 10 == 0:
            print(f"  Pass 1: {count + 1}/{n_frames} frames")
    print(f"  Pass 1 done: {n_frames} frames (full FOV)")

    # Pass 2: all frames again as narrow strips (overwrites with strip-aligned content)
    for count, fi in enumerate(order):
        paint_frame(fi, count_label=count, use_strip=True, gap_fill_only=False)
        if (count + 1) % 10 == 0:
            print(f"  Pass 2: {count + 1}/{n_frames} frames")
    print(f"  Pass 2 done: {n_frames} frames (strip)")

    # Flip horizontally for inside-sphere view (panorama viewer convention)
    panorama = panorama[:, ::-1].copy()

    print(f"Total: {canvas_w}x{canvas_h} panorama [{fmt_time(time.time() - t0)}]")
    return panorama


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_session(session_dir, output_dir, num_frames=50, scene_distance=2.0,
                    reverse=False):
    session_name = os.path.basename(os.path.normpath(session_dir))
    print(f"\n{'='*60}")
    print(f"Session: {session_name}")
    print(f"{'='*60}")

    ar_path = os.path.join(session_dir, "ar_data.json")
    video_path = os.path.join(session_dir, "video.mp4")

    if not os.path.exists(ar_path):
        print(f"Error: {ar_path} not found")
        return False
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found")
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
    suffix = "_layered_rev" if reverse else "_layered"
    output_path = os.path.join(output_dir, f"{session_name}{suffix}.jpg")

    panorama = stitch_layered(cap, ar_data, indices, scene_distance=scene_distance,
                              reverse=reverse)
    cap.release()

    if panorama is not None:
        cv.imwrite(output_path, panorama, [cv.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved: {output_path}")
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

    args = parser.parse_args()
    input_dir = os.path.normpath(args.input)
    output_dir = args.output or os.path.join(input_dir, "output")

    if args.all:
        sessions = []
        for name in sorted(os.listdir(input_dir)):
            sub = os.path.join(input_dir, name)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "ar_data.json")):
                sessions.append(sub)

        if not sessions:
            print(f"No sessions found in {input_dir}")
            sys.exit(1)

        print(f"Found {len(sessions)} sessions")
        ok = 0
        for session_dir in sessions:
            if process_session(session_dir, output_dir, args.num_frames,
                               args.scene_distance, args.reverse):
                ok += 1
        print(f"\nDone: {ok}/{len(sessions)} sessions processed")
    else:
        if not os.path.exists(os.path.join(input_dir, "ar_data.json")):
            print(f"Error: {input_dir} does not contain ar_data.json")
            sys.exit(1)
        process_session(input_dir, output_dir, args.num_frames,
                        args.scene_distance, args.reverse)


if __name__ == "__main__":
    main()
