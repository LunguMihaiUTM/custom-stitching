"""Custom panorama stitcher using known ARKit camera transforms.

For each output pixel on the equirectangular sphere, determines which
frame owns it (whose optical axis is closest), reprojects to that frame's
pixel coordinates, and samples it. This creates clean vertical-cut seams
between frames with no ghosting or blending artifacts.

Usage:
    python stitch.py <session_dir> [-o output_dir] [--num-frames 50]
    python stitch.py C:/Users/lungu/Desktop/sessions/balcon
    python stitch.py C:/Users/lungu/Desktop/sessions --all
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
# Data loading
# ---------------------------------------------------------------------------

def load_ar_data(path):
    with open(path) as f:
        return json.load(f)


def arkit_to_matrix(camera_transform):
    """Column-major 16 floats -> 4x4 numpy matrix."""
    return np.array(camera_transform, dtype=np.float64).reshape(4, 4, order="F")


def arkit_intrinsics(intrinsics):
    """ARKit column-major 3x3 intrinsics -> standard K matrix."""
    return np.array(intrinsics, dtype=np.float64).reshape(3, 3, order="F")


def extract_frame(cap, frame_idx):
    """Extract a single frame from video capture."""
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


# ---------------------------------------------------------------------------
# Coordinate conventions
# ---------------------------------------------------------------------------

# ARKit camera: X=right, Y=up, Z=out-of-screen (looks at -Z)
# OpenCV camera: X=right, Y=down, Z=into-screen (looks at +Z)
# Conversion: flip Y and Z
AXIS_FLIP = np.diag([1.0, -1.0, -1.0])


def get_camera_rotation(ar_entry):
    """Get world-to-camera rotation matrix in OpenCV convention.

    ARKit gives camera-to-world T. R_arkit maps camera-local -> world.
    We want world -> OpenCV-camera = FLIP @ R_arkit^T
    """
    T = arkit_to_matrix(ar_entry["cameraTransform"])
    R_arkit = T[:3, :3]
    R_world_to_cam = AXIS_FLIP @ R_arkit.T
    return R_world_to_cam


def get_camera_forward_world(ar_entry):
    """Get camera forward direction in world coordinates."""
    T = arkit_to_matrix(ar_entry["cameraTransform"])
    R = T[:3, :3]
    return -R[:, 2]  # camera looks at -Z in ARKit


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def select_frames(ar_data, num_frames=50):
    """Select frames with good angular coverage + sharpness."""
    # Compute yaw for all valid frames
    valid = []
    for i, entry in enumerate(ar_data):
        if entry.get("trackingState") != "normal":
            continue
        fwd = get_camera_forward_world(entry)
        yaw = np.arctan2(fwd[0], fwd[2])
        valid.append((i, yaw))

    if len(valid) <= num_frames:
        return [i for i, _ in valid]

    # Bin by yaw, pick middle frame per bin
    yaw_arr = np.array([y for _, y in valid])
    idx_arr = np.array([i for i, _ in valid])

    bin_edges = np.linspace(-np.pi, np.pi, num_frames + 1)
    selected = []
    for b in range(num_frames):
        mask = (yaw_arr >= bin_edges[b]) & (yaw_arr < bin_edges[b + 1])
        candidates = idx_arr[mask]
        if len(candidates) > 0:
            selected.append(int(candidates[len(candidates) // 2]))

    # Fill remaining slots with temporal picks if needed
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
# Inverse-warp panorama stitching (cut-based, no blending)
# ---------------------------------------------------------------------------

def equirect_to_ray(lon, lat):
    """Convert equirectangular (longitude, latitude) to 3D unit ray.

    lon in [-pi, pi], lat in [-pi/2, pi/2]
    Returns ray in ARKit world coordinates (Y-up, right-handed).
    Top of image = +Y (up), bottom = -Y (down).
    """
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)   # +lat = up = +Y
    z = np.cos(lat) * np.cos(lon)
    return np.stack([x, y, z], axis=-1)


def estimate_pivot(cam_positions):
    """Estimate the rotation pivot point (body center) from camera orbit.

    Uses least-squares circle fitting in the X-Z (horizontal) plane to
    find the center and radius of the camera's orbital path around the body.
    """
    px = cam_positions[:, 0]
    pz = cam_positions[:, 2]
    cy = cam_positions[:, 1].mean()

    # Least-squares circle fit in X-Z plane (Kasa method)
    # Minimize algebraic distance: (x-cx)^2 + (z-cz)^2 = R^2
    # Rewrite as: x^2 + z^2 = 2*cx*x + 2*cz*z + (R^2 - cx^2 - cz^2)
    # Linear system: A @ [cx, cz, c] = b  where c = R^2 - cx^2 - cz^2
    A = np.column_stack([2 * px, 2 * pz, np.ones(len(px))])
    b = px**2 + pz**2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cz, c = result
    orbit_radius = float(np.sqrt(c + cx**2 + cz**2))

    pivot = np.array([cx, cy, cz])

    # Report fit quality
    radii = np.sqrt((px - cx)**2 + (pz - cz)**2)
    std_dev = float(np.std(radii))
    print(f"Circle fit: radius={orbit_radius:.3f}m, std={std_dev:.3f}m "
          f"(deviation {std_dev/orbit_radius*100:.1f}%)")

    return pivot, orbit_radius


def stitch_panorama(cap, ar_data, frame_indices, canvas_w=None, canvas_h=None,
                    scene_distance=2.0):
    """Build panorama by inverse-warping: for each output pixel, sample the best frame.

    For each pixel on the equirectangular canvas:
    1. Convert to a 3D world ray from the estimated pivot point (body center)
    2. Find which selected frame's optical axis is closest (ownership)
    3. Project ray into that frame's image, accounting for the camera's
       actual position (orbiting around the pivot)
    4. Sample the pixel (bilinear interpolation)

    The pivot-aware correction models the camera as orbiting around the
    person's body, giving correct reprojection for both the orbital motion
    and residual hand sway.
    """
    t0 = time.time()

    n_frames = len(frame_indices)

    # Gather camera data for all selected frames
    Ks = []
    R_w2cs = []  # world-to-camera rotation
    cam_positions = []  # camera position in world space
    forwards = []  # camera forward in world space (for ownership)

    for idx in frame_indices:
        entry = ar_data[idx]
        K = arkit_intrinsics(entry["intrinsics"])
        R_w2c = get_camera_rotation(entry)
        fwd = get_camera_forward_world(entry)

        T = arkit_to_matrix(entry["cameraTransform"])
        pos = T[:3, 3]  # camera position in world

        Ks.append(K)
        R_w2cs.append(R_w2c)
        cam_positions.append(pos)
        forwards.append(fwd / np.linalg.norm(fwd))

    forwards = np.array(forwards)  # (N, 3)
    cam_positions = np.array(cam_positions)  # (N, 3)

    # Estimate pivot (body center) and orbit radius
    pivot, orbit_radius = estimate_pivot(cam_positions)
    cam_offsets = cam_positions - pivot  # (N, 3)

    print(f"Pivot: ({pivot[0]:.2f}, {pivot[1]:.2f}, {pivot[2]:.2f}), "
          f"orbit radius: {orbit_radius:.3f}m")

    # Determine canvas size from average focal length
    avg_focal = np.mean([(K[0, 0] + K[1, 1]) / 2 for K in Ks])
    if canvas_w is None:
        canvas_w = int(round(2 * np.pi * avg_focal))
    if canvas_h is None:
        canvas_h = canvas_w // 2

    print(f"Canvas: {canvas_w}x{canvas_h}, avg focal={avg_focal:.1f}")

    # Read the first frame to get image dimensions
    test_frame = extract_frame(cap, frame_indices[0])
    img_h, img_w = test_frame.shape[:2]
    print(f"Frame size: {img_w}x{img_h}")

    # Pre-load all selected frames into memory
    print(f"Loading {n_frames} frames...")
    frames = []
    for idx in frame_indices:
        frame = extract_frame(cap, idx)
        if frame is None:
            frame = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        frames.append(frame)
    print(f"Frames loaded [{fmt_time(time.time() - t0)}]")

    # Build pixel coordinate grids for the equirectangular canvas
    t1 = time.time()
    print("Computing pixel assignments...")

    # Pixel (x, y) -> (lon, lat) -> 3D ray
    xs = np.arange(canvas_w, dtype=np.float64)
    ys = np.arange(canvas_h, dtype=np.float64)
    lon = (xs / canvas_w - 0.5) * 2 * np.pi   # [-pi, pi]
    lat = (0.5 - ys / canvas_h) * np.pi        # [pi/2, -pi/2] (top=north)

    lon_grid, lat_grid = np.meshgrid(lon, lat)  # (H, W)

    # 3D world rays for every pixel
    rays = equirect_to_ray(lon_grid, lat_grid)  # (H, W, 3)
    rays_flat = rays.reshape(-1, 3).astype(np.float32)  # (H*W, 3)

    # Compute ownership in chunks to limit memory
    chunk_size = 500_000
    n_total = rays_flat.shape[0]
    owner = np.empty(n_total, dtype=np.int32)
    fwd_f32 = forwards.astype(np.float32)

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        dots = rays_flat[start:end] @ fwd_f32.T  # (chunk, N)
        owner[start:end] = np.argmax(dots, axis=1)

    print(f"Ownership computed [{fmt_time(time.time() - t1)}]")

    # For each frame, compute the pixel coordinates for all pixels it owns
    t2 = time.time()
    panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for fi in range(n_frames):
        mask = (owner == fi)
        n_pixels = np.sum(mask)
        if n_pixels == 0:
            continue

        # Get rays owned by this frame
        fi_rays = rays_flat[mask]  # (M, 3)

        # Place scene points at estimated distance along each ray from
        # the panorama origin, then reproject from the camera's actual position.
        # world_point = pano_origin + scene_distance * ray_dir
        # cam_coords = R_w2c @ (world_point - cam_position)
        #            = R_w2c @ (scene_distance * ray - cam_offset)
        offset = cam_offsets[fi]  # (3,) offset from panorama origin
        scene_pts = scene_distance * fi_rays.T - offset[:, np.newaxis]  # (3, M)
        cam_coords = R_w2cs[fi] @ scene_pts  # (3, M)

        # Check if point is in front of camera (z > 0 in OpenCV convention)
        z = cam_coords[2, :]
        valid = z > 0
        if not np.any(valid):
            continue

        # Perspective divide
        u = cam_coords[0, :] / z * Ks[fi][0, 0] + Ks[fi][0, 2]
        v = cam_coords[1, :] / z * Ks[fi][1, 1] + Ks[fi][1, 2]

        # Check bounds (with 0.5px margin)
        in_bounds = valid & (u >= 0) & (u < img_w - 1) & (v >= 0) & (v < img_h - 1)

        if not np.any(in_bounds):
            continue

        # Bilinear interpolation
        u_valid = u[in_bounds]
        v_valid = v[in_bounds]

        u0 = np.floor(u_valid).astype(np.int32)
        v0 = np.floor(v_valid).astype(np.int32)
        u1 = u0 + 1
        v1 = v0 + 1

        # Clamp
        u0 = np.clip(u0, 0, img_w - 1)
        u1 = np.clip(u1, 0, img_w - 1)
        v0 = np.clip(v0, 0, img_h - 1)
        v1 = np.clip(v1, 0, img_h - 1)

        du = (u_valid - u0).astype(np.float32)
        dv = (v_valid - v0).astype(np.float32)

        img = frames[fi]
        # (M_valid, 3) for each corner
        p00 = img[v0, u0].astype(np.float32)
        p01 = img[v0, u1].astype(np.float32)
        p10 = img[v1, u0].astype(np.float32)
        p11 = img[v1, u1].astype(np.float32)

        du3 = du[:, np.newaxis]
        dv3 = dv[:, np.newaxis]
        interp = (p00 * (1 - du3) * (1 - dv3) +
                  p01 * du3 * (1 - dv3) +
                  p10 * (1 - du3) * dv3 +
                  p11 * du3 * dv3)

        # Write to panorama
        # Convert flat mask indices back to (y, x)
        flat_indices = np.where(mask)[0]
        valid_flat = flat_indices[in_bounds]
        py = valid_flat // canvas_w
        px = valid_flat % canvas_w

        panorama[py, px] = np.clip(interp, 0, 255).astype(np.uint8)

        if (fi + 1) % 10 == 0:
            print(f"  Rendered {fi + 1}/{n_frames} frames")

    print(f"Rendering [{fmt_time(time.time() - t2)}]")

    # --- Exposure equalization ---
    t3 = time.time()
    panorama = equalize_exposure(panorama, owner, canvas_w, canvas_h, n_frames)
    print(f"Exposure equalized [{fmt_time(time.time() - t3)}]")

    # Flip horizontally for inside-sphere view (panorama viewer convention)
    panorama = panorama[:, ::-1].copy()

    print(f"Total: {canvas_w}x{canvas_h} panorama [{fmt_time(time.time() - t0)}]")

    return panorama


def equalize_exposure(panorama, owner, canvas_w, canvas_h, n_frames):
    """Equalize exposure by computing per-frame gains from overlap borders.

    Finds border pixels where two adjacent frame regions meet, compares
    brightness, and adjusts per-frame gain to minimize seam visibility.
    """
    owner_2d = owner.reshape(canvas_h, canvas_w)
    gray = cv.cvtColor(panorama, cv.COLOR_BGR2GRAY).astype(np.float64)

    # Build graph of overlapping frame pairs and their brightness ratios
    # Look at 1-pixel-wide border between adjacent owned regions
    pairs = {}  # (fi, fj) -> list of (brightness_fi, brightness_fj)

    # Horizontal borders
    diff_h = owner_2d[:, :-1] != owner_2d[:, 1:]
    ys, xs = np.where(diff_h)
    for y, x in zip(ys[::10], xs[::10]):  # subsample for speed
        fi, fj = int(owner_2d[y, x]), int(owner_2d[y, x + 1])
        bi, bj = gray[y, x], gray[y, x + 1]
        if bi > 5 and bj > 5:  # skip black pixels
            key = (min(fi, fj), max(fi, fj))
            if key not in pairs:
                pairs[key] = ([], [])
            if fi < fj:
                pairs[key][0].append(bi)
                pairs[key][1].append(bj)
            else:
                pairs[key][0].append(bj)
                pairs[key][1].append(bi)

    if not pairs:
        return panorama

    # Compute per-frame gain using iterative approach
    gains = np.ones(n_frames, dtype=np.float64)
    for _ in range(10):  # iterate to converge
        new_gains = np.ones(n_frames, dtype=np.float64)
        counts = np.zeros(n_frames, dtype=np.float64)
        for (fi, fj), (bi_list, bj_list) in pairs.items():
            if len(bi_list) < 10:
                continue
            mean_i = np.mean(bi_list) * gains[fi]
            mean_j = np.mean(bj_list) * gains[fj]
            target = (mean_i + mean_j) / 2
            if mean_i > 1:
                new_gains[fi] += target / mean_i
                counts[fi] += 1
            if mean_j > 1:
                new_gains[fj] += target / mean_j
                counts[fj] += 1
        for fi in range(n_frames):
            if counts[fi] > 0:
                gains[fi] *= new_gains[fi] / (counts[fi] + 1)
        gains = np.clip(gains, 0.5, 2.0)

    # Apply gains
    for fi in range(n_frames):
        if abs(gains[fi] - 1.0) < 0.01:
            continue
        mask = (owner_2d == fi)
        if not np.any(mask):
            continue
        panorama[mask] = np.clip(
            panorama[mask].astype(np.float32) * gains[fi], 0, 255
        ).astype(np.uint8)

    return panorama


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_session(session_dir, output_dir, num_frames=50, scene_distance=2.0):
    """Process a single session: load data, select frames, stitch."""
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

    if n_video != len(ar_data):
        print(f"Warning: video frames ({n_video}) != ar_data entries ({len(ar_data)})")

    # Select frames
    indices = select_frames(ar_data, num_frames)
    print(f"Selected {len(indices)} frames")

    # Stitch
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{session_name}_panorama.jpg")

    panorama = stitch_panorama(cap, ar_data, indices, scene_distance=scene_distance)
    cap.release()

    if panorama is not None:
        cv.imwrite(output_path, panorama, [cv.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved: {output_path}")

        meta_path = os.path.join(output_dir, f"{session_name}_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "session": session_name,
                "num_frames_selected": len(indices),
                "frame_indices": indices,
                "panorama_size": [panorama.shape[1], panorama.shape[0]],
            }, f, indent=2)
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Custom panorama stitcher using ARKit camera transforms."
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
                               args.scene_distance):
                ok += 1
        print(f"\nDone: {ok}/{len(sessions)} sessions processed")
    else:
        if not os.path.exists(os.path.join(input_dir, "ar_data.json")):
            print(f"Error: {input_dir} does not contain ar_data.json")
            sys.exit(1)
        process_session(input_dir, output_dir, args.num_frames,
                        args.scene_distance)


if __name__ == "__main__":
    main()
