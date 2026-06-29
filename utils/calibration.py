"""
Camera calibration utilities.
Handles both real Ego-Exo4D calibration JSON and synthetic demo cameras.
"""

import numpy as np
import json
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Camera:
    name: str
    K: np.ndarray        # 3x3 intrinsic matrix
    R: np.ndarray        # 3x3 rotation (world -> camera)
    t: np.ndarray        # 3x1 translation (world -> camera)
    dist: np.ndarray     # distortion coefficients
    width: int
    height: int

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix"""
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    @property
    def center(self) -> np.ndarray:
        """Camera center in world coords"""
        return (-self.R.T @ self.t).flatten()


def load_egoexo4d_calibration(calib_json_path: str) -> List[Camera]:
    """
    Parse Ego-Exo4D camera calibration JSON.
    Format: aria + exo gopro cameras with intrinsics/extrinsics.
    """
    with open(calib_json_path) as f:
        data = json.load(f)

    cameras = []
    for cam_data in data.get("cameras", []):
        name = cam_data["name"]
        # Ego-Exo4D uses column-major matrices
        intrinsics = cam_data.get("intrinsics", {})
        fx = intrinsics.get("fx", 500)
        fy = intrinsics.get("fy", 500)
        cx = intrinsics.get("cx", 640)
        cy = intrinsics.get("cy", 360)
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)

        extrinsics = cam_data.get("extrinsics", {})
        # T_world_camera -> we need T_camera_world = inv
        T = np.array(extrinsics.get("T_world_camera", np.eye(4).tolist()))
        T_cw = np.linalg.inv(T)
        R = T_cw[:3, :3]
        t = T_cw[:3, 3]

        dist = np.zeros(5)
        w = cam_data.get("width", 1280)
        h = cam_data.get("height", 720)

        cameras.append(Camera(name=name, K=K, R=R, t=t, dist=dist, width=w, height=h))

    return cameras


def make_synthetic_cameras(n_exo: int = 4, radius: float = 3.0,
                            height: float = 1.5, img_w: int = 1280,
                            img_h: int = 720) -> List[Camera]:
    """
    Generate synthetic exo cameras arranged in a circle around the scene origin.
    Used for demo mode.
    """
    cameras = []
    fx = fy = 800.0
    cx, cy = img_w / 2, img_h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    for i in range(n_exo):
        angle = 2 * np.pi * i / n_exo
        cam_pos = np.array([radius * np.cos(angle),
                            radius * np.sin(angle),
                            height])
        # Look toward origin
        forward = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0, 0, 1.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        R = np.stack([right, -up, forward], axis=0)  # camera axes as rows
        t = -R @ cam_pos

        cameras.append(Camera(
            name=f"exo_cam_{i:02d}",
            K=K.copy(), R=R, t=t,
            dist=np.zeros(5),
            width=img_w, height=img_h
        ))

    return cameras


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion → 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


def load_gopro_calibration(take_dir: str) -> List[Camera]:
    """
    Load exo GoPro camera calibrations from trajectory/gopro_calibs.csv.

    The CSV stores T_world_cam as translation + quaternion.
    Intrinsics are KANNALABRANDTK3: fx, fy, cx, cy, k1..k4.
    """
    csv_path = Path(take_dir) / "trajectory" / "gopro_calibs.csv"
    cameras = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            name = row["cam_uid"]
            fx = float(row["intrinsics_0"])
            fy = float(row["intrinsics_1"])
            cx = float(row["intrinsics_2"])
            cy = float(row["intrinsics_3"])
            K  = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            dist = np.array([float(row[f"intrinsics_{i}"]) for i in range(4, 8)])

            # T_world_cam → invert to get T_cam_world (what we need)
            t_wc = np.array([float(row["tx_world_cam"]),
                              float(row["ty_world_cam"]),
                              float(row["tz_world_cam"])])
            R_wc = _quat_to_rot(float(row["qx_world_cam"]),
                                 float(row["qy_world_cam"]),
                                 float(row["qz_world_cam"]),
                                 float(row["qw_world_cam"]))
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc

            w = int(row["image_width"])
            h = int(row["image_height"])
            cameras.append(Camera(name=name, K=K, R=R_cw, t=t_cw,
                                  dist=dist, width=w, height=h))
    return cameras


class AriaPoseInterpolator:
    """
    Per-frame ego camera pose from closed_loop_trajectory.csv.

    The trajectory is at 1kHz; frames are extracted at ~1fps.
    For frame i of N total frames, we linearly map into the trajectory
    and return the nearest pose as a Camera object (same K/dist as the
    static ego camera, but updated R/t).
    """

    def __init__(self, take_dir: str, ego_cam_template: Camera):
        """
        Args:
            take_dir:         path to the take directory
            ego_cam_template: Camera with correct K/dist/width/height
                              (extrinsics will be overwritten per frame)
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Loading trajectory (this reads ~50 MB — one-time cost)...")

        traj_path = Path(take_dir) / "trajectory" / "closed_loop_trajectory.csv"

        # Load T_device_camera fixed rotation from online_calibration.jsonl
        jsonl_path = Path(take_dir) / "trajectory" / "online_calibration.jsonl"
        with open(jsonl_path) as f:
            calib = json.loads(f.readline())
        cam_calib = next(
            c for c in calib["CameraCalibrations"] if "rgb" in c["Label"].lower()
        )
        t_dc = np.array(cam_calib["T_Device_Camera"]["Translation"])
        q_scalar, q_xyz = cam_calib["T_Device_Camera"]["UnitQuaternion"]
        self._R_dc = _quat_to_rot(q_xyz[0], q_xyz[1], q_xyz[2], q_scalar)
        self._t_dc = t_dc

        # Read trajectory: timestamps + device poses
        timestamps, txs, tys, tzs = [], [], [], []
        qxs, qys, qzs, qws = [], [], [], []
        with open(traj_path) as f:
            for row in csv.DictReader(f):
                timestamps.append(int(row["tracking_timestamp_us"]))
                txs.append(float(row["tx_world_device"]))
                tys.append(float(row["ty_world_device"]))
                tzs.append(float(row["tz_world_device"]))
                qxs.append(float(row["qx_world_device"]))
                qys.append(float(row["qy_world_device"]))
                qzs.append(float(row["qz_world_device"]))
                qws.append(float(row["qw_world_device"]))

        self._timestamps = np.array(timestamps, dtype=np.int64)
        self._t_wd = np.stack([txs, tys, tzs], axis=1)   # (N, 3)
        self._quats = np.stack([qxs, qys, qzs, qws], axis=1)  # (N, 4)
        self._template = ego_cam_template
        logger.info(f"Loaded {len(timestamps)} trajectory rows "
                    f"spanning {(timestamps[-1]-timestamps[0])/1e6:.1f}s")

    def camera_at_frame(self, frame_idx: int, n_frames: int) -> Camera:
        """Return ego Camera with pose interpolated to frame_idx."""
        # Map frame index linearly to trajectory index
        traj_idx = int(round(frame_idx * (len(self._timestamps) - 1) / max(n_frames - 1, 1)))
        traj_idx = np.clip(traj_idx, 0, len(self._timestamps) - 1)

        t_wd = self._t_wd[traj_idx]
        qx, qy, qz, qw = self._quats[traj_idx]
        R_wd = _quat_to_rot(qx, qy, qz, qw)

        # T_world_cam = T_world_device @ T_device_camera
        R_wc = R_wd @ self._R_dc
        t_wc = R_wd @ self._t_dc + t_wd

        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc

        cam = self._template
        return Camera(name=cam.name, K=cam.K.copy(), R=R_cw, t=t_cw,
                      dist=cam.dist.copy(), width=cam.width, height=cam.height)


def load_aria_ego_camera(take_dir: str):
    """
    Load Aria ego camera intrinsics + static pose (first trajectory row)
    and return both a static Camera and an AriaPoseInterpolator for per-frame use.

    Returns:
        (Camera, AriaPoseInterpolator)
    """
    take_dir = Path(take_dir)

    # ── Intrinsics + T_Device_Camera ────────────────────────────────────────
    jsonl_path = take_dir / "trajectory" / "online_calibration.jsonl"
    with open(jsonl_path) as f:
        calib = json.loads(f.readline())

    cam_calib = next(
        c for c in calib["CameraCalibrations"] if "rgb" in c["Label"].lower()
    )
    params = cam_calib["Projection"]["Params"]
    # FisheyeRadTanThinPrism: [f, cx, cy, k1, ...] — single focal length,
    # calibrated at native 2880x2880. Scale to extracted frame size (1408x1408).
    ARIA_RGB_NATIVE  = 2880.0
    ARIA_RGB_EXTRACT = 1408.0
    scale = ARIA_RGB_EXTRACT / ARIA_RGB_NATIVE
    f  = params[0] * scale
    cx = params[1] * scale
    cy = params[2] * scale
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    t_dc = np.array(cam_calib["T_Device_Camera"]["Translation"])
    q_scalar, q_xyz = cam_calib["T_Device_Camera"]["UnitQuaternion"]
    R_dc = _quat_to_rot(q_xyz[0], q_xyz[1], q_xyz[2], q_scalar)

    # ── Static pose from first trajectory row ────────────────────────────────
    traj_path = take_dir / "trajectory" / "closed_loop_trajectory.csv"
    with open(traj_path) as f:
        row = next(csv.DictReader(f))
    t_wd = np.array([float(row["tx_world_device"]),
                     float(row["ty_world_device"]),
                     float(row["tz_world_device"])])
    R_wd = _quat_to_rot(float(row["qx_world_device"]),
                         float(row["qy_world_device"]),
                         float(row["qz_world_device"]),
                         float(row["qw_world_device"]))

    R_wc = R_wd @ R_dc
    t_wc = R_wd @ t_dc + t_wd
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc

    static_cam = Camera(name="aria_rgb", K=K, R=R_cw, t=t_cw,
                        dist=np.zeros(5), width=1408, height=1408)

    interpolator = AriaPoseInterpolator(str(take_dir), static_cam)
    return static_cam, interpolator


def make_ego_camera(img_w: int = 1408, img_h: int = 1408) -> Camera:
    """
    Synthetic ego (fisheye-ish) camera at head position looking forward.
    """
    fx = fy = 300.0  # wide FOV
    cx, cy = img_w / 2, img_h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    # Placed at head height, looking forward slightly downward
    cam_pos = np.array([0.0, 0.0, 1.7])
    R = np.array([[1, 0, 0],
                  [0, 0, -1],
                  [0, 1, 0]], dtype=np.float64)  # tilted down ~15 deg
    t = -R @ cam_pos
    return Camera(name="ego_cam", K=K, R=R, t=t,
                  dist=np.zeros(5), width=img_w, height=img_h)
