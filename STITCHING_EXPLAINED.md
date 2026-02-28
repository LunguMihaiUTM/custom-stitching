# Panorama Stitching Pipeline: Deep Technical Explanation

This document provides a comprehensive, step-by-step explanation of the two panorama stitching pipelines implemented in `stitch_layer.py` (ARKit/iOS) and `stitch_android.py` (Android IMU), along with the shared `blending.py` and `exposure.py` modules.

---

## 1. Overview

Both stitchers take a video file and per-frame orientation data, then project selected video frames onto an **equirectangular canvas** -- the standard 2:1 panorama format where longitude maps to the horizontal axis and latitude maps to the vertical axis. The equirectangular image can be directly loaded into a Three.js sphere for immersive viewing.

The pipeline is:

```
Video + Orientation Data
    |
    v
Parse input data (transforms / IMU quaternions)
    |
    v
Build camera intrinsics matrix (sensor -> video resolution)
    |
    v
Select ~50 frames distributed evenly by yaw angle
    |
    v
Pass 1: Paint all frames onto equirectangular canvas (full FOV, later overwrites earlier)
    |
    v
Pass 2: Re-paint narrow strips with multi-band Laplacian blending for smooth seams
    |
    v
Vertical centering
    |
    v
Blur gap filling (edge propagation + progressive blur)
    |
    v
Boundary diffusion (smooth content edges)
    |
    v
Horizontal flip (for inside-sphere viewing convention)
    |
    v
Output panorama image + transform JSON
```

**Key difference**: `stitch_layer.py` has camera **position** data from ARKit and performs **parallax correction** (pivot estimation + scene distance projection). `stitch_android.py` only has orientation (quaternions) and performs **pure rotation-only projection** -- no parallax correction is possible.

---

## 2. Input Data

### ARKit (stitch_layer.py)

Input file: `ar_data.json` -- a JSON array where each element is a per-frame record containing:

| Field | Type | Description |
|-------|------|-------------|
| `cameraTransform` | float[16] | 4x4 column-major homogeneous transformation matrix (ARKit world -> camera pose) |
| `intrinsics` | float[9] | 3x3 column-major camera intrinsics matrix |
| `trackingState` | string | `"normal"`, `"limited"`, etc. Only `"normal"` frames are used |

The `cameraTransform` is a 4x4 matrix in **ARKit world coordinates**:
- **X** = right
- **Y** = up
- **Z** = backward (out of screen toward user)

This is a right-handed Y-up system. The matrix contains both **rotation** (upper-left 3x3) and **translation** (4th column) -- giving full 6-DOF camera pose.

### Android IMU (stitch_android.py)

Input file: `imu_data.json` -- a JSON object containing:

| Field | Type | Description |
|-------|------|-------------|
| `samples` | array | Per-frame IMU records |
| `cameraIntrinsics` | object | `{fx, fy, cx, cy}` for the full sensor resolution |
| `sensorRotationDegrees` | int | How many degrees CW the camera sensor is rotated relative to device portrait (typically 90) |
| `videoResolution` | string | e.g. `"1920x1080"` |

Each sample in `samples` contains:

| Field | Type | Description |
|-------|------|-------------|
| `frameIndex` | int | Video frame number |
| `quaternion` | object | `{x, y, z, w}` from Android `TYPE_GAME_ROTATION_VECTOR` |

The quaternion represents the rotation **from device frame to world frame** (ENU convention):
- **E** (East) = X
- **N** (North) = Y
- **U** (Up) = Z

The Android device portrait frame is:
- **X** = right (when holding phone portrait)
- **Y** = up
- **Z** = out of screen

**Critical difference**: Android provides orientation only. No position. No parallax correction possible.

---

## 3. Step-by-Step Pipeline

### 3.1 Data Loading & Parsing

#### ARKit (`load_ar_data`, `arkit_to_matrix`, `arkit_intrinsics`)

```python
def arkit_to_matrix(camera_transform):
    return np.array(camera_transform, dtype=np.float64).reshape(4, 4, order="F")
```

The flat 16-element array is reshaped with `order="F"` (Fortran/column-major) because ARKit stores matrices in column-major order (like OpenGL). This produces:

```
| R00  R01  R02  tx |
| R10  R11  R12  ty |
| R20  R21  R22  tz |
|  0    0    0    1 |
```

The upper-left 3x3 is the rotation `R_arkit`, and `T[:3, 3]` is the camera position in world coordinates.

Intrinsics are similarly reshaped from a flat 9-element column-major array:

```python
def arkit_intrinsics(intrinsics):
    return np.array(intrinsics, dtype=np.float64).reshape(3, 3, order="F")
```

Producing:

```
| fx   0   cx |
|  0  fy   cy |
|  0   0    1 |
```

ARKit provides intrinsics already scaled to the video resolution, so no additional scaling is needed.

#### Android (`load_imu_data`)

Simple JSON load. The quaternion is accessed as `sample['quaternion']` with fields `x, y, z, w`. The intrinsics are in `imu_data['cameraIntrinsics']` as explicit `fx, fy, cx, cy` values at **full sensor resolution** (not video resolution), which requires scaling (see Section 3.3).

---

### 3.2 Coordinate System Conventions

This is the most subtle and critical part of the pipeline. Three coordinate systems are in play:

#### World Frame

```
ARKit world (Y-up):          Android ENU world:

      Y (up)                       Z (up)
      |                            |
      |                            |
      +--- X (right/east)          +--- Y (north)
     /                            /
    Z (backward/south)           X (east)
```

The equirectangular projection in both stitchers uses a **Y-up convention** (matching ARKit):
- X = East (lon > 0 direction)
- Y = Up (lat > 0 direction)
- Z = North (lon = 0, the center of the panorama)

For the Android stitcher, a conversion matrix bridges ENU to Y-up:

```python
YUP_TO_ENU = np.array([[1, 0, 0],   # ENU_x = Yup_x (both East)
                        [0, 0, 1],   # ENU_y = Yup_z (North)
                        [0, 1, 0]])  # ENU_z = Yup_y (Up)
```

This swaps the Y and Z axes: `vec_enu = YUP_TO_ENU @ vec_yup`.

#### Camera Frame (OpenCV Convention)

Both stitchers use the **OpenCV camera convention**:

```
        Z (forward, into scene)
       /
      /
     +--- X (right)
     |
     Y (down)
```

This is a right-handed system where Z points forward into the scene, X is right, and Y is **down** (not up). This matches what the pinhole projection model expects.

#### ARKit Camera Frame vs OpenCV Camera Frame

ARKit's camera frame has Y-up and Z-backward (out of screen). To convert to OpenCV convention:

```python
AXIS_FLIP = np.diag([1.0, -1.0, -1.0])
```

This negates Y (up -> down) and Z (backward -> forward):

```
ARKit camera:     OpenCV camera:
  Y (up)            X (right)
  |                 |
  +-- X (right)     +-- Z (forward)
 /                 /
Z (out of screen) Y (down)

Conversion: X_cv = X_ar, Y_cv = -Y_ar, Z_cv = -Z_ar
```

#### Android Device Frame vs OpenCV Camera Frame

The Android device in portrait has X=right, Y=up, Z=out-of-screen. But the camera sensor is physically rotated by `sensorRotationDegrees` (typically 90 degrees CW) relative to the device's portrait orientation.

The conversion is a two-step process:

1. **Sensor rotation** (`sensor_rotation_matrix`): Rotate from device portrait frame to camera sensor frame
2. **Axis flip** (`AXIS_FLIP`): Convert from sensor frame (Y-up, Z-out) to OpenCV (Y-down, Z-forward)

```python
def build_device_to_opencv(sensor_rotation_deg):
    R_sensor = sensor_rotation_matrix(sensor_rotation_deg)
    return AXIS_FLIP @ R_sensor
```

For `sensorRotationDegrees=90`, this produces:

```
device_to_opencv = [[0, 1, 0],    # OpenCV X = device Y (landscape right)
                     [1, 0, 0],    # OpenCV Y = device X (landscape down)
                     [0, 0, -1]]   # OpenCV Z = -device Z (into screen)
```

#### Complete Rotation Chain (Android)

To get `R_world_to_camera` in the Y-up world convention:

```python
def get_camera_rotation(sample, device_to_opencv):
    R_w2d = quat_to_R_world_to_device(sample['quaternion'])  # world(ENU) -> device
    R_w2c_enu = device_to_opencv @ R_w2d                      # world(ENU) -> camera(OpenCV)
    return R_w2c_enu @ YUP_TO_ENU                             # world(Y-up) -> camera(OpenCV)
```

The chain is: `R_w2c_yup = device_to_opencv @ R_w2d @ YUP_TO_ENU`

Conceptually:
1. `YUP_TO_ENU` converts the Y-up world ray into ENU coordinates
2. `R_w2d` rotates from ENU world frame to device frame
3. `device_to_opencv` converts from device frame to OpenCV camera frame

#### Complete Rotation Chain (ARKit)

```python
def get_camera_rotation(ar_entry):
    T = arkit_to_matrix(ar_entry["cameraTransform"])
    R_arkit = T[:3, :3]           # Rotation part: camera -> world (ARKit convention)
    R_world_to_cam = AXIS_FLIP @ R_arkit.T   # world -> camera (OpenCV convention)
    return R_world_to_cam
```

`R_arkit.T` gives world-to-camera in ARKit convention. `AXIS_FLIP` then converts to OpenCV convention.

Since ARKit already uses Y-up world coordinates (matching the equirectangular convention), no `YUP_TO_ENU` conversion is needed.

---

### 3.3 Building the Intrinsics Matrix

#### ARKit

ARKit provides intrinsics already scaled to the video resolution. Direct reshape:

```python
K = arkit_intrinsics(entry["intrinsics"])  # 3x3 column-major reshape
```

Each frame can have slightly different intrinsics (e.g., autofocus changes focal length), so ARKit stores per-frame `Ks[fi]`.

#### Android

The JSON provides intrinsics for the **full sensor resolution** (e.g., 4624x3468 at 4:3). The video is typically 1920x1080 (16:9), which is center-cropped from the sensor and then downscaled.

**Numerical example**: For a sensor of 4624x3468 with video 1920x1080:

```
sensor_w = 2 * cx = 4624   (cx ~ 2312)
sensor_h = 2 * cy = 3468   (cy ~ 1734)

scale = video_w / sensor_w = 1920 / 4624 = 0.4153

crop_h = sensor_w * video_h / video_w = 4624 * 1080 / 1920 = 2601.0
cy_offset = (sensor_h - crop_h) / 2 = (3468 - 2601) / 2 = 433.5 pixels

K_video:
  fx_video = fx * 0.4153
  fy_video = fy * 0.4153
  cx_video = cx * 0.4153
  cy_video = (cy - 433.5) * 0.4153
```

The `cy_offset` accounts for the fact that the 4:3 sensor is cropped to 16:9 by removing equal strips from top and bottom. The horizontal axis is not cropped (scale is width-based), so `cx` is simply scaled.

```python
def build_intrinsics(imu_data, video_w, video_h):
    sensor_w = int(round(cx * 2))
    sensor_h = int(round(cy * 2))
    scale = video_w / sensor_w
    crop_h = sensor_w * video_h / video_w
    cy_offset = (sensor_h - crop_h) / 2.0
    K = np.array([
        [fx * scale, 0,          cx * scale],
        [0,          fy * scale, (cy - cy_offset) * scale],
        [0,          0,          1]
    ])
```

#### Portrait Frame Handling (Android on Windows)

On Windows, OpenCV sometimes auto-applies the video's rotation metadata, delivering portrait frames (e.g., 1080x1920) instead of landscape (1920x1080). When `actual_w < actual_h` is detected, the code uses `build_intrinsics_portrait`:

```python
if actual_w < actual_h and sensor_rot_deg != 0:
    device_to_opencv = build_device_to_opencv(0)  # skip sensor rotation
    K = build_intrinsics_portrait(imu_data, actual_w, actual_h, sensor_rot_deg)
```

The logic for portrait intrinsics:
1. Compute landscape video intrinsics as normal (land_w = port_h, land_h = port_w)
2. Apply the 90-degree CW rotation transform to fx/fy/cx/cy:
   - `fx_port = fy_land`, `fy_port = fx_land`
   - `cx_port = (land_h - 1) - cy_land`, `cy_port = cx_land`

The sensor rotation is set to 0 for `device_to_opencv` because OpenCV already applied it.

---

### 3.4 Frame Selection (Yaw-Based Binning)

Both stitchers select `num_frames` (default 50) frames evenly distributed across the yaw range, ensuring uniform angular coverage.

**Algorithm** (`select_frames`):

1. For each frame, compute the camera forward direction in world space
2. Compute yaw as `atan2(forward_x, forward_z)` -- the angle in the XZ (horizontal) plane
3. Divide the full yaw range [-pi, pi] into `num_frames` equal bins
4. From each bin, select the **median** frame (middle of the candidates list)
5. If some bins are empty (camera didn't point in that direction), fill remaining slots with evenly-spaced frames from the full list

**Why median, not random?** The median candidate is likely to be temporally centered within the time the camera pointed in that direction, avoiding edge cases at the beginning/end of a sweep.

**ARKit-specific**: Only frames with `trackingState == "normal"` are considered. Android has no tracking state filter.

**Camera forward direction**:
- ARKit: `get_camera_forward_world(entry)` returns `-R[:, 2]` (the negated third column of the rotation matrix, because ARKit Z is backward)
- Android: `get_camera_forward_world(sample, device_to_opencv)` returns `R_w2c.T @ [0, 0, 1]` (the third row of R_w2c transposed, because OpenCV forward is +Z)

---

### 3.5 Canvas Size Computation

The equirectangular canvas dimensions are computed from the average focal length:

```python
avg_focal = (K[0, 0] + K[1, 1]) / 2.0    # average of fx and fy
canvas_w = int(round(2 * pi * avg_focal))  # full 360 degrees
canvas_h = canvas_w // 2                   # full 180 degrees
```

**Why this formula?** In an equirectangular projection, one pixel corresponds to the same angular increment everywhere. The full horizontal span is 2*pi radians. If we want each pixel to correspond to the same angle as one pixel in the original camera image (at the center, where the pinhole model is most accurate), then:

```
pixels_per_radian = focal_length (in pixels)
canvas_width = 2 * pi * focal_length
```

For `avg_focal = 1500`, this gives `canvas_w = 9425`, `canvas_h = 4712`.

---

### 3.6 Equirectangular Projection (lon/lat <-> 3D rays)

The equirectangular canvas maps each pixel (col, row) to a (longitude, latitude) pair:

```python
lon_full = (xs / canvas_w - 0.5) * 2 * pi    # col 0 -> lon=-pi, col W-1 -> lon=+pi
lat_full = (0.5 - ys / canvas_h) * pi         # row 0 -> lat=+pi/2 (top), row H-1 -> lat=-pi/2
```

**Longitude**: Leftmost pixel = -pi (180 degrees W), center = 0 (facing +Z/North), rightmost = +pi (180 degrees E).

**Latitude**: Top pixel = +pi/2 (straight up), center = 0 (horizon), bottom = -pi/2 (straight down).

A (lon, lat) pair maps to a 3D unit ray in Y-up world coordinates:

```python
def equirect_to_ray(lon, lat):
    x = cos(lat) * sin(lon)   # East component
    y = sin(lat)               # Up component
    z = cos(lat) * cos(lon)    # North component
    return [x, y, z]
```

**Derivation**: This is the standard spherical-to-Cartesian conversion where longitude is measured from +Z toward +X, and latitude is elevation from the XZ plane.

**Examples**:
- `(lon=0, lat=0)` -> `(0, 0, 1)` = looking North (+Z)
- `(lon=pi/2, lat=0)` -> `(1, 0, 0)` = looking East (+X)
- `(lon=0, lat=pi/2)` -> `(0, 1, 0)` = looking Up (+Y)

---

### 3.7 Camera Rotation Computation

See Section 3.2 for the full rotation chain derivation. The key outputs are:

- `R_w2c` (world-to-camera): Used to project world rays into camera coordinates
- `R_c2w = R_w2c.T` (camera-to-world): Used to back-project image border pixels into world rays (for bounding box computation)

---

### 3.8 Ray Projection (World Rays -> Pixel Coordinates)

This is the core of the stitching: for each canvas pixel, shoot a ray into the world, project it into a camera frame, and sample the frame's color.

#### Rotation-Only (Android)

```python
cam_coords = R_w2cs[fi] @ rays_flat.T     # (3, N) camera-space coordinates
z = cam_coords[2, :]                       # depth
u = cam_coords[0, :] / z * fx + cx         # horizontal pixel coordinate
v = cam_coords[1, :] / z * fy + cy         # vertical pixel coordinate
```

The ray is a direction (unit vector), so after rotation the Z component gives the "virtual depth." The pinhole projection divides by Z and scales by focal length.

**Validity check**: `z > 0` (ray must be in front of camera) and `0 <= u < img_w - 1`, `0 <= v < img_h - 1` (projected point must land on the image).

#### With Parallax Correction (ARKit)

```python
offset = cam_offsets[fi]                                    # camera position relative to pivot
scene_pts = scene_distance * rays_flat.T - offset[:, None]  # parallax-corrected points
cam_coords = R_w2cs[fi] @ scene_pts                         # project into camera
```

The key difference is `scene_pts = scene_distance * rays_flat.T - offset`. Instead of projecting infinite-distance rays (pure rotation), we:

1. Place a 3D point on each ray at distance `scene_distance` from the pivot
2. Subtract the camera's offset from the pivot to get the vector from the camera to that scene point
3. Project this vector into camera coordinates

**Pivot estimation** (`estimate_pivot`): Fits a circle to the camera positions in the XZ plane using least squares:

```python
A = [[2*px_0, 2*pz_0, 1],    b = [px_0^2 + pz_0^2,
     [2*px_1, 2*pz_1, 1],         px_1^2 + pz_1^2,
     ...]                         ...]
[cx, cz, c] = lstsq(A, b)
orbit_radius = sqrt(c + cx^2 + cz^2)
```

This solves for the center `(cx, cz)` and radius of the circle in the horizontal plane that best fits the camera trajectory. The Y coordinate is just the average `cy = mean(cam_positions[:, 1])`.

**Why parallax matters**: When the camera translates (orbits), nearby objects shift relative to distant ones. Without correction, overlapping frames would show misalignment on nearby objects. The `scene_distance` parameter (default 2.0 meters) controls where the "sharp alignment" plane is -- objects at this distance from the pivot will stitch perfectly; objects closer or farther will have some residual parallax.

---

### 3.9 Bilinear Sampling

Once `(u, v)` pixel coordinates are computed in the source frame, bilinear interpolation fetches sub-pixel-accurate colors:

```python
u0, v0 = floor(u), floor(v)          # top-left integer pixel
u1, v1 = u0 + 1, v0 + 1              # bottom-right
du, dv = u - u0, v - v0              # fractional offsets in [0, 1)

interp = p00 * (1-du) * (1-dv)       # top-left weight
       + p01 * du     * (1-dv)       # top-right weight
       + p10 * (1-du) * dv           # bottom-left weight
       + p11 * du     * dv           # bottom-right weight
```

Where `p00 = frame[v0, u0]`, etc. (Note: image indexing is `[row, col]` = `[v, u]`.)

This avoids jagged edges and provides smooth color transitions between pixels.

---

### 3.10 Frame Canvas Bounds

Before projecting an entire frame, `frame_canvas_bounds` computes the bounding box on the equirectangular canvas where this frame can possibly contribute pixels. This avoids processing the entire canvas for each frame.

**Algorithm**:

1. Sample 8 points along the frame border (4 corners + 4 edge midpoints)
2. Back-project each border point to a world ray: `world_ray = R_c2w @ cam_ray`
3. Convert each world ray to (lon, lat)
4. Find the lon/lat extremes
5. Convert to canvas pixel ranges

**Wrap-around detection**: If `lon_max - lon_min > pi`, the frame straddles the +/-pi boundary. In this case, the column range is split into two segments (one at the right edge, one at the left edge of the canvas).

```python
wraps = (lon_max - lon_min) > np.pi
```

A small margin of 0.05 radians (~3 degrees) is added to avoid missing edge pixels.

---

### 3.11 Pass 1: Full FOV Paint

In the first pass, frames are sorted by yaw and painted in order. Each frame's full field of view is projected, and pixels are simply overwritten (no blending).

```python
order = list(np.argsort(yaws))
for count, fi in enumerate(order):
    paint_frame(fi, use_strip=False)
```

**Why sort by yaw?** When frames are painted in yaw order, the seam (where the last-painted frame meets the first) falls at the +/-180-degree boundary (the left/right edge of the panorama). This places the seam at the least-visible location.

**Result**: A complete panorama with sharp frame boundaries visible wherever frames overlap. This provides the geometric base for Pass 2.

---

### 3.12 Pass 2: Strip-Based Multi-Band Blending

Pass 2 re-renders each frame but only within a **narrow yaw strip** centered on the frame's viewing direction, then blends this strip onto the panorama using multi-band blending.

#### Strip Width

```python
avg_yaw_step = 360.0 / n_frames        # e.g., 360/50 = 7.2 degrees
strip_half_deg = avg_yaw_step * 1.5     # e.g., 10.8 degrees
# Total strip width = 2 * strip_half_deg = ~21.6 degrees
```

The strip is ~3x the average angular spacing between frames. This means adjacent strips overlap by roughly one strip width, ensuring smooth blending coverage.

#### Feathering

The strip has a horizontal feather (fade-in / fade-out) at its left and right edges:

```python
feather_width = max(n_cols // 6, 4)
feather = np.ones(n_cols, dtype=np.float32)
ramp = np.linspace(0.0, 1.0, feather_width)
feather[:feather_width] = ramp        # left fade-in: 0 -> 1
feather[-feather_width:] = ramp[::-1] # right fade-out: 1 -> 0
blend_mask = strip_mask * feather[np.newaxis, :]
```

The feather is 1/6 of the strip width on each side. The `blend_mask` combines the feather with the valid-pixel mask (`strip_mask`), which is 1.0 where the frame successfully projected and 0.0 elsewhere.

#### Multi-Band Blending

Where both the panorama and the strip have content, `multiband_blend` is called:

```python
blended = multiband_blend(pano_patch, strip_patch, blend_mask, levels=4)
```

This uses Laplacian pyramid blending (see Section 5.4) to smoothly merge brightness (low frequencies) while preserving sharp details (high frequencies) from each source.

Where the panorama has no content yet (first time painting this region), the strip is directly pasted:

```python
direct_paste = (blend_mask > 0) & (pano_has_content < 0.5)
```

---

### 3.13 Exposure Compensation

The `exposure.py` module provides two mechanisms:

#### Per-Frame Gain (`compute_gains`)

1. For each frame, compute the **trimmed mean** brightness per channel (B, G, R), ignoring pixels below 10 and above 245 to exclude clipped highlights and deep shadows
2. Compute a **target brightness** as the median per-channel value across all frames
3. Compute per-channel gain: `gain = target / frame_avg`
4. Clamp each channel gain to [0.7, 1.4] and limit average gain to 1.3

```python
gains = []
for avg_b, avg_g, avg_r in channel_avgs:
    gb = np.clip(target_b / max(avg_b, 1.0), 0.7, 1.4)
    gg = np.clip(target_g / max(avg_g, 1.0), 0.7, 1.4)
    gr = np.clip(target_r / max(avg_r, 1.0), 0.7, 1.4)
```

**Note**: In the current code, `apply_gain` is **computed but not applied** (the call is commented out in both stitchers). The gain values are logged for diagnostic purposes only.

#### Post-Process CLAHE (`equalize_brightness`)

Applies **Contrast Limited Adaptive Histogram Equalization** on the L channel in LAB color space. This flattens local brightness differences without shifting color balance. Currently imported but not called in the main pipeline.

---

### 3.14 Vertical Centering

After stitching, the content may not be vertically centered on the canvas (e.g., if the camera was tilted up during capture). The code shifts the panorama vertically to center the content band:

```python
painted_rows = np.any(painted_mask > 0, axis=1)
row_indices_painted = np.where(painted_rows)[0]
content_center = (row_indices_painted[0] + row_indices_painted[-1]) / 2.0
canvas_center = canvas_h / 2.0
shift_px = int(round(canvas_center - content_center))
panorama = np.roll(panorama, shift_px, axis=0)
```

Pixels shifted beyond the canvas edges are zeroed out. This centering only appears in `stitch_android.py`. The ARKit stitcher does not perform this step.

---

### 3.15 Blur Gap Filling

The top and bottom of the panorama (ceiling and floor) are typically not covered by the camera. These gaps are filled with a blurred, color-propagated fill to avoid harsh black edges.

#### Step 0: Erode the Painted Mask

```python
erode_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
eroded_mask = cv.erode(painted_mask, erode_k, iterations=3)
```

Eroding by 3 iterations with a 5x5 elliptical kernel shrinks the "trusted" content region by about 15 pixels. This excludes unreliable dark pixels at the content boundary (often caused by partial frame coverage or vignetting).

#### Step 1: Edge Color Propagation (at 1/8 scale)

For memory efficiency, propagation runs on a downscaled version:

```python
scale = 8
sh, sw = canvas_h // scale, canvas_w // scale
```

**Top-down propagation**: For each row from top to bottom, if a pixel is unpainted, copy the color from the pixel directly above it. This extends the top edge of the content downward.

```python
for y in range(1, sh):
    unpainted = ~mask_down[y]
    filled_down[y][unpainted] = filled_down[y - 1][unpainted]
```

**Bottom-up propagation**: Same idea, but propagating upward from the bottom edge.

**Distance-weighted blending**: Combine top-down and bottom-up fills using inverse-distance weights:

```
w_down = dist_up / (dist_down + dist_up)    # farther from bottom -> more weight on top-down fill
w_up   = dist_down / (dist_down + dist_up)  # farther from top -> more weight on bottom-up fill
```

This creates a smooth transition. For example, a gap pixel that is 10 rows below the top content and 30 rows above the bottom content will be:

```
w_down = 30 / (10 + 30) = 0.75  (mostly top-down fill color)
w_up   = 10 / (10 + 30) = 0.25  (some bottom-up fill color)
```

**Left-right propagation**: Handles side gaps (columns with no vertical content nearby) by copying from adjacent columns.

**Horizontal smoothing**: A Gaussian blur with kernel width `max(sw//8, 3)` x 3 removes per-column color stripe artifacts in the gap regions.

#### Step 2: Upscale and Restore

The filled 1/8-scale image is upscaled to full resolution. Pixels within the eroded mask are restored from the original panorama to prevent the fill from contaminating real content.

#### Step 3: Progressive Blur

A distance-based progressive blur makes the fill increasingly blurry the farther it is from real content:

```python
dist_full = cv.distanceTransform(gap_full, cv.DIST_L2, 5)
```

Three blur levels are applied:

| Distance from edge | Blur level | Method |
|---|---|---|
| 0-15 px | Light | GaussianBlur(31, 31) at full res |
| 15-60 px | Medium | GaussianBlur(41, 41) at 1/4 res, upscaled |
| 60+ px | Heavy | GaussianBlur(61, 61) at 1/8 res, upscaled |

Each level is blended in using a linear ramp:

```python
t = np.clip(dist_full / 15.0, 0, 1)       # 0 at edge, 1 at 15px
bg = bg * (1 - t) + blur * t               # linear interpolation
```

The medium and heavy blurs are computed at reduced resolution for efficiency, then upscaled.

#### Step 4: Feathered Alpha Composite

The fill is composited with the panorama using a soft alpha mask:

```python
alpha = eroded_mask.astype(np.float32) / 255.0
alpha = cv.GaussianBlur(alpha, (71, 71), 0)    # 71px Gaussian feather
result = panorama * alpha + bg * (1 - alpha)    # blend
```

The 71x71 Gaussian blur on the binary alpha mask creates a ~35-pixel soft transition zone between real content and the fill, eliminating visible seams.

---

### 3.16 Boundary Diffusion

After gap filling, the boundary between real content and fill may still show artifacts. Boundary diffusion smooths this transition:

1. Compute the **boundary band**: pixels within 10 morphological iterations of the original painted mask edge (both inward and outward):

```python
_bnd_inner = cv.erode(_debug_painted, kernel, iterations=10)
_bnd_outer = cv.dilate(_debug_painted, kernel, iterations=10)
_bnd_band = _bnd_outer & ~_bnd_inner    # ~30px wide band around content edge
```

2. For each boundary pixel, determine whether real content is above or below (by checking 30 pixels in each direction)

3. **Copy from 20px inside** the content toward the boundary:

```python
shift = 20
src_y[content_above] = ys[content_above] - shift    # copy from above
src_y[content_below] = ys[content_below] + shift     # copy from below
panorama[ys, xs] = panorama[src_y, xs]
```

4. Apply Gaussian blur to the entire boundary band:

```python
blurred_pano = cv.GaussianBlur(panorama, (31, 31), 0)
panorama[_bnd_band] = blurred_pano[_bnd_band]
```

This replaces the boundary with content from slightly inside, then diffuses it, creating a smooth fade from content to fill.

---

### 3.17 Horizontal Flip for Inside-Sphere View

```python
panorama = panorama[:, ::-1].copy()
```

Equirectangular panoramas are typically viewed from **inside** a sphere in a 3D viewer (Three.js `SphereGeometry` with `material.side = BackSide`). When viewed from inside, the image appears mirrored compared to the "outside-looking-in" projection used during stitching. The horizontal flip corrects this so text and spatial relationships appear correct to the viewer.

---

### 3.18 Panorama Transform Output

Both stitchers output a JSON transform describing the panorama's orientation in world space.

#### ARKit

```python
forward = np.array([0.0, 0.0, 1.0])    # panorama center = +Z
z_axis = -forward                        # ARKit convention: -Z = forward
y_axis = np.array([0.0, 1.0, 0.0])
x_axis = np.cross(y_axis, z_axis)        # = [-1, 0, 0]
centroid = cam_positions.mean(axis=0)     # average camera position

pano_transform[:3, 0] = x_axis    # [-1, 0, 0]
pano_transform[:3, 1] = y_axis    # [0, 1, 0]
pano_transform[:3, 2] = z_axis    # [0, 0, -1]
pano_transform[:3, 3] = centroid   # [cx, cy, cz]
```

This 4x4 matrix is stored column-major (`flatten(order="F")`) and encodes both the orientation (which direction the panorama center points) and the position (centroid of all camera positions).

#### Android

Identical rotation, but `pano_transform[:3, 3]` is left as `[0, 0, 0]` since there is no position data.

---

## 4. Key Differences: ARKit vs Android IMU

### Comparison Table

| Aspect | ARKit (`stitch_layer.py`) | Android IMU (`stitch_android.py`) |
|--------|--------------------------|-----------------------------------|
| **Orientation source** | 4x4 camera transform matrix | Quaternion from `TYPE_GAME_ROTATION_VECTOR` |
| **Position data** | Yes (from ARKit SLAM) | No (IMU only gives orientation) |
| **World coordinate system** | Y-up (X=right, Y=up, Z=back) | ENU (X=east, Y=north, Z=up), converted to Y-up |
| **Parallax correction** | Yes (`estimate_pivot` + `scene_distance`) | No (pure rotation-only projection) |
| **Intrinsics** | Per-frame 3x3 matrix (ARKit provides at video resolution) | Single set of (fx, fy, cx, cy) at sensor resolution, manually scaled |
| **Tracking state filter** | Yes (`trackingState == "normal"`) | No filter |
| **Sensor rotation handling** | Not needed (ARKit handles it) | `sensorRotationDegrees` + portrait auto-detection |
| **Vertical centering** | Not performed | Performed (shifts content to canvas center) |
| **Transform output position** | Centroid of camera positions | Origin `[0, 0, 0]` |
| **Output format** | JPEG (quality 95) | PNG (compression 3) |

### Why ARKit Produces Better Alignment

1. **Position data enables parallax correction**: When the camera orbits a scene, nearby objects shift relative to distant ones. ARKit provides the camera's 3D position, allowing the stitcher to model this parallax. The pivot estimation fits a circle to the camera trajectory and uses `scene_distance` to place virtual scene points on each ray. This dramatically reduces ghosting in overlapping regions where nearby objects would otherwise appear in slightly different positions from adjacent frames.

2. **Per-frame intrinsics**: If autofocus changes the focal length between frames, ARKit captures the exact intrinsics per frame. Android uses a single fixed intrinsics set, which may be slightly wrong for some frames.

3. **Tracking state filtering**: ARKit flags frames where visual-inertial tracking is degraded. Skipping these avoids projecting with inaccurate poses.

4. **Higher accuracy rotation**: ARKit fuses visual features (point cloud) with IMU data for drift-corrected pose estimation. Android `TYPE_GAME_ROTATION_VECTOR` fuses accelerometer + gyroscope only (no visual anchoring), so drift accumulates over time. A 360-degree sweep may have several degrees of accumulated yaw drift.

### Parallax Correction Detail

The ARKit stitcher projects rays not as infinite-distance directions (pure rotation), but as **scene points at finite distance**:

```
Android (rotation only):    cam_coords = R_w2c @ ray
ARKit (with parallax):      scene_pt = scene_distance * ray - cam_offset
                            cam_coords = R_w2c @ scene_pt
```

`cam_offset` is the vector from the estimated pivot (center of the orbit) to the camera position. Subtracting it shifts the virtual scene point from the pivot-centered coordinate system to the camera-centered one.

```
          scene_pt (at scene_distance from pivot)
              *
             /|\
            / | \
           /  |  \
  cam_A --/   |   \-- cam_B
              |
            pivot
```

Both cameras see the same scene point, but from different positions. The `- offset` term accounts for this difference.

---

## 5. Mathematical Details

### 5.1 Equirectangular <-> 3D Ray Conversion

**Canvas pixel to (lon, lat)**:

```
lon = (col / canvas_w - 0.5) * 2 * pi        range: [-pi, +pi]
lat = (0.5 - row / canvas_h) * pi            range: [+pi/2, -pi/2]
```

**(lon, lat) to 3D unit ray** (Y-up, right-handed):

```
x = cos(lat) * sin(lon)
y = sin(lat)
z = cos(lat) * cos(lon)
```

**3D unit ray to (lon, lat)**:

```
lon = atan2(x, z)
lat = arcsin(y)
```

**(lon, lat) to canvas pixel**:

```
col = (lon / (2*pi) + 0.5) * canvas_w
row = (0.5 - lat / pi) * canvas_h
```

### 5.2 Quaternion to Rotation Matrix

Android provides quaternions in `[x, y, z, w]` format. `scipy.spatial.transform.Rotation.from_quat([x, y, z, w])` converts this to a 3x3 rotation matrix `R_device_to_world`.

The rotation matrix from quaternion `q = (x, y, z, w)` is:

```
R = | 1-2(y^2+z^2)   2(xy-wz)      2(xz+wy)   |
    | 2(xy+wz)        1-2(x^2+z^2)  2(yz-wx)   |
    | 2(xz-wy)        2(yz+wx)      1-2(x^2+y^2)|
```

Since the Android quaternion represents device-to-world, we need `R_world_to_device = R_device_to_world.T`:

```python
r = Rotation.from_quat([q['x'], q['y'], q['z'], q['w']])
R_w2d = r.as_matrix().T
```

### 5.3 Pinhole Projection Model

The pinhole camera model projects a 3D point `(X, Y, Z)` in camera coordinates to 2D pixel coordinates `(u, v)`:

```
u = X / Z * fx + cx
v = Y / Z * fy + cy
```

Or in matrix form:

```
s * [u]   [fx  0  cx] [X]
    [v] = [ 0 fy  cy] [Y]
    [1]   [ 0  0   1] [Z]
```

Where `s = Z` is the depth (projective scale factor).

- `fx, fy`: focal length in pixels (horizontal and vertical, may differ slightly due to non-square pixels)
- `cx, cy`: principal point (optical center, ideally at image center)

**Validity**: The projection is only valid when `Z > 0` (point is in front of the camera) and `0 <= u < W, 0 <= v < H` (point lands within the image).

### 5.4 Multi-Band (Laplacian Pyramid) Blending

Multi-band blending smooths brightness transitions while preserving sharp detail. It operates on the frequency decomposition of both images.

#### Laplacian Pyramid Construction

The **Gaussian pyramid** is a sequence of progressively downsampled images:

```
G_0 = original image
G_1 = pyrDown(G_0)        # half resolution
G_2 = pyrDown(G_1)        # quarter resolution
...
```

The **Laplacian pyramid** captures the detail lost at each downsampling step:

```
L_i = G_i - pyrUp(G_{i+1})    # difference = high-frequency detail at level i
L_n = G_n                      # lowest resolution = residual (low-frequency content)
```

#### Blending Process

1. Build Laplacian pyramids for both images `A` and `B`
2. Build Gaussian pyramid for the blend mask `M`
3. At each level, blend using the corresponding mask:

```
L_blend_i = L_A_i * (1 - M_i) + L_B_i * M_i
```

4. Reconstruct the result by collapsing the blended Laplacian pyramid:

```
result = L_blend_n
for i = n-1 down to 0:
    result = pyrUp(result) + L_blend_i
```

**Why it works**: At high levels (low resolution), the mask is heavily blurred, so brightness transitions between images are smooth and gradual. At low levels (high resolution), the mask is sharp, so fine details come from whichever source is designated by the mask. The result: seamless brightness with sharp detail.

**Level count**: `levels = min(4, int(log2(min(h, w))) - 1)` ensures we don't downsample below 1 pixel.

### 5.5 Parallax Correction (ARKit Only)

The parallax model assumes the scene consists of surfaces at distance `d` from the pivot point. For each canvas ray (from the pivot):

```
scene_point = d * ray_direction
```

The camera sees this point from position `cam_pos`. The vector from camera to scene point (in pivot-centered coordinates) is:

```
view_vector = scene_point - (cam_pos - pivot) = d * ray - cam_offset
```

Projecting `view_vector` into the camera frame:

```
cam_coords = R_w2c @ view_vector
u = cam_coords[0] / cam_coords[2] * fx + cx
v = cam_coords[1] / cam_coords[2] * fy + cy
```

**Effect of scene_distance**:
- `d = infinity` -> pure rotation (parallax vanishes, since `cam_offset / d -> 0`)
- `d = orbit_radius` -> maximum parallax correction (object at the pivot itself)
- `d = 2.0m` (default) -> good for typical indoor rooms

The pivot estimation uses least-squares circle fitting in the XZ plane:

```
Minimize sum_i || [px_i, pz_i] - [cx, cz] ||^2 = r^2

Linearized: 2*px*cx + 2*pz*cz + (r^2 - cx^2 - cz^2) = px^2 + pz^2
```

This is solved as a standard `Ax = b` linear system.

---

## 6. Code File Reference

| File | Purpose |
|------|---------|
| `stitch_layer.py` | ARKit stitcher with parallax correction and 6-DOF pose |
| `stitch_android.py` | Android IMU stitcher with rotation-only (3-DOF) projection |
| `blending.py` | Multi-band Laplacian pyramid blending |
| `exposure.py` | Per-frame exposure gain computation and CLAHE equalization |

### Key Functions by File

**stitch_layer.py**:
- `arkit_to_matrix()` / `arkit_intrinsics()` -- parse column-major ARKit data
- `get_camera_rotation()` -- `AXIS_FLIP @ R_arkit.T` for world-to-OpenCV
- `estimate_pivot()` -- least-squares circle fit for camera orbit
- `select_frames()` -- yaw-based binning with tracking state filter
- `frame_canvas_bounds()` -- bounding box for frame's canvas footprint
- `stitch_layered()` -- main 2-pass stitching with parallax correction

**stitch_android.py**:
- `sensor_rotation_matrix()` / `build_device_to_opencv()` -- device-to-OpenCV chain
- `quat_to_R_world_to_device()` -- quaternion to rotation matrix (transposed)
- `get_camera_rotation()` -- full chain `device_to_opencv @ R_w2d @ YUP_TO_ENU`
- `build_intrinsics()` / `build_intrinsics_portrait()` -- sensor-to-video intrinsics scaling
- `select_frames()` -- yaw-based binning (no tracking state filter)
- `stitch_layered()` -- main 2-pass stitching, rotation-only

**blending.py**:
- `_build_gaussian_pyramid()` -- `pyrDown` cascade
- `_build_laplacian_pyramid()` -- difference-of-Gaussians decomposition
- `multiband_blend()` -- per-level blending and pyramid reconstruction

**exposure.py**:
- `compute_gains()` -- trimmed-mean per-channel gain calculation
- `apply_gain()` -- per-channel multiplication (currently unused in pipeline)
- `equalize_brightness()` -- CLAHE on LAB L-channel (available but not called)