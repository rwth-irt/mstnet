"""
Utilities for reading SeePerSea dataset files.

The SeePerSea aggregated dataset uses:
- PCD ASCII files for point clouds  (x, y, z, intensity)
- KITTI-format label files          (class x1 y1 x2 y2 h w l x_cam y_cam z_cam rot_y)
- Per-frame calibration files       (P2, R0_rect, Tr_velo_to_cam)

Labels are in camera coordinates (standard KITTI convention).
Point clouds are in LiDAR/velodyne coordinates.
This module provides helpers to bridge the two.
"""

import numpy as np


# ── Point cloud reading ───────────────────────────────────────────────────────

# Divisor for the intensity channel, the 99th percentile over the 2951 training frames
# read from the raw PCD files (p50 53, p90 291, p99 821, max 7256).
#
# The previous value was 255, which is not a property of this sensor and clipped
# 12.8 percent of all points to exactly 1.0, so one point in eight carried no
# intensity information at all. It also put the detection data on a scale roughly 120
# times below the self-supervised data, which is converted by a different script, and a
# control run showed that mismatch alone costs about 5 average precision in three
# dimensions.
#
# The same rule is applied to every dataset, each with its own divisor taken from its
# own training split, so all four share a comparable distribution while no test
# statistic enters the pipeline.
SEEPERSEA_INTENSITY_P99 = 821.0


def read_pcd(path: str) -> np.ndarray:
    """
    Read a PCD v0.7 file (ASCII) and return a [N, 4] float32 array (x, y, z, intensity).

    Intensity is normalised to [0, 1] by dividing by the 99th percentile of the
    training split and clipping.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_start = 0
    data_type = "ascii"
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("DATA"):
            data_type = stripped.split()[1].lower()
            data_start = i + 1
            break

    if data_type != "ascii":
        raise ValueError(
            f"read_pcd: only ASCII PCD is supported, got '{data_type}' in {path}"
        )

    points = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        vals = line.split()
        if len(vals) < 4:
            continue
        points.append([float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])])

    if not points:
        return np.zeros((0, 4), dtype=np.float32)

    arr = np.array(points, dtype=np.float32)
    arr[:, 3] = np.clip(arr[:, 3] / SEEPERSEA_INTENSITY_P99, 0.0, 1.0)
    return arr


# ── Calibration reading ───────────────────────────────────────────────────────

def read_kitti_calib(path: str) -> dict[str, np.ndarray]:
    """
    Parse a KITTI-style calibration file.

    Returns a dict with keys like 'P2', 'R0_rect', 'Tr_velo_to_cam',
    each mapping to a flat float64 numpy array.
    """
    calib: dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, values = line.partition(":")
            calib[key.strip()] = np.array(
                [float(v) for v in values.strip().split()], dtype=np.float64
            )
    return calib


def get_tr_velo_to_cam(calib: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract rotation matrix R (3×3) and translation vector t (3,) from Tr_velo_to_cam.

    Tr_velo_to_cam is a 3×4 matrix [R | t] stored as 12 values row-major.
    """
    tr = calib["Tr_velo_to_cam"].reshape(3, 4)
    R = tr[:, :3]   # (3, 3)
    t = tr[:, 3]    # (3,)
    return R, t


# ── Coordinate transformation ─────────────────────────────────────────────────

def transform_boxes_cam_to_lidar(
    boxes_cam: list[list[float]],
    calib: dict[str, np.ndarray],
) -> list[list[float]]:
    """
    Transform 3-D boxes from camera coordinates to LiDAR coordinates.

    Each box is [x_cam, y_cam, z_cam, l, w, h, rotation_y].
    Returns boxes in format [x_lidar, y_lidar, z_lidar, l, w, h, theta_lidar].

    Coordinate systems (standard KITTI convention):
        Camera:  x right, y down, z forward (depth)
        LiDAR:   x forward, y left, z up

    The transformation is:
        p_lidar = R^T @ (p_cam - t)
    where R and t come from Tr_velo_to_cam (LiDAR → camera).

    Heading angle:
        An object with rotation_y = 0 faces camera +z.
        The heading is transformed via the forward-vector method:
        forward_lidar = R^T @ (sin θ, 0, cos θ), then
        theta_lidar = arctan2(forward_y, forward_x) - π/2
        (the -π/2 aligns with the codebase convention where theta=0
        means the length axis extends along the y-axis).
    """
    R, t = get_tr_velo_to_cam(calib)
    R_inv = R.T  # R is (approximately) orthogonal

    boxes_lidar = []
    for box in boxes_cam:
        x_c, y_c, z_c, l, w, h, rotation_y = box

        # 1. Transform center
        p_cam = np.array([x_c, y_c, z_c], dtype=np.float64)
        p_lidar = R_inv @ (p_cam - t)

        # 2. Transform heading
        #    Object forward unit vector in camera frame: (sin θ, 0, cos θ)
        #    Map to LiDAR frame and extract yaw around z.
        #
        #    Codebase box convention: at theta=0 the length axis (index 3)
        #    extends along the y-axis and width (index 4) along x.  The
        #    forward angle from arctan2 measures from the x-axis, so we
        #    subtract π/2 to align with the codebase reference direction.
        forward_cam = np.array(
            [np.sin(rotation_y), 0.0, np.cos(rotation_y)], dtype=np.float64
        )
        forward_lidar = R_inv @ forward_cam
        theta_lidar = (
            float(np.arctan2(forward_lidar[1], forward_lidar[0])) - np.pi / 2
        )

        boxes_lidar.append(
            [
                float(p_lidar[0]),
                float(p_lidar[1]),
                float(p_lidar[2]),
                l,
                w,
                h,
                theta_lidar,
            ]
        )
    return boxes_lidar
