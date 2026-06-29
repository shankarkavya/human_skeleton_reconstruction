"""
Multi-view triangulation using Direct Linear Transform (DLT).
Takes 2D keypoints from N exo cameras + their projection matrices → 3D points.
"""

import numpy as np
from typing import List, Optional, Tuple
from calibration import Camera


def triangulate_dlt(points_2d: List[Optional[np.ndarray]],
                    cameras: List[Camera]) -> np.ndarray:
    """
    Triangulate a single 3D point from N views using DLT.

    Args:
        points_2d: List of (2,) arrays [u, v] or None if not visible
        cameras:   Corresponding Camera objects

    Returns:
        (3,) array — 3D point in world coordinates
    """
    A_rows = []
    for pt, cam in zip(points_2d, cameras):
        if pt is None:
            continue
        u, v = pt
        P = cam.P  # 3x4
        A_rows.append(u * P[2] - P[0])
        A_rows.append(v * P[2] - P[1])

    if len(A_rows) < 4:
        return np.full(3, np.nan)

    A = np.stack(A_rows, axis=0)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3])


def triangulate_pose(keypoints_per_cam: List[Optional[np.ndarray]],
                     cameras: List[Camera],
                     conf_thresh: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate a full pose skeleton from multiple views.

    Args:
        keypoints_per_cam: List of (N_joints, 3) arrays [x, y, conf] per camera,
                           or None if cam has no detection
        cameras:           Camera list
        conf_thresh:       Min confidence to use a detection

    Returns:
        keypoints_3d: (N_joints, 3) in world coords
        valid_mask:   (N_joints,) bool — True if triangulated successfully
    """
    # Determine joint count from first valid detection
    n_joints = None
    for kps in keypoints_per_cam:
        if kps is not None:
            n_joints = kps.shape[0]
            break
    if n_joints is None:
        return np.zeros((0, 3)), np.zeros(0, dtype=bool)

    keypoints_3d = np.zeros((n_joints, 3))
    valid_mask = np.zeros(n_joints, dtype=bool)

    for j in range(n_joints):
        pts_2d = []
        cams_j = []
        for kps, cam in zip(keypoints_per_cam, cameras):
            if kps is None:
                continue
            x, y, conf = kps[j]
            if conf >= conf_thresh:
                pts_2d.append(np.array([x, y]))
                cams_j.append(cam)

        if len(cams_j) >= 2:
            p3d = triangulate_dlt(pts_2d, cams_j)
            if not np.any(np.isnan(p3d)):
                keypoints_3d[j] = p3d
                valid_mask[j] = True

    return keypoints_3d, valid_mask


def reprojection_error(point_3d: np.ndarray,
                       point_2d: np.ndarray,
                       camera: Camera) -> float:
    """Compute reprojection error for a single point."""
    p = camera.P @ np.append(point_3d, 1.0)
    p = p[:2] / p[2]
    return float(np.linalg.norm(p - point_2d))
