"""
Exo-only 3D scene reconstruction via multi-view triangulation.

Flow per frame:
  1. Load all 4 GoPro (exo) frames
  2. RTMPose wholebody → 2D keypoints per camera
  3. DLT triangulation across cameras → 3D body + hand skeleton
  4. Log cameras + live frames + 3D skeleton to Rerun

Usage:
  python src/reconstruct_scene.py --take /path/to/take_dir [--frames 60] [--out output/exo.rrd]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

# Allow imports from parent (calibration.py, triangulate.py, pose_exo.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration import load_gopro_calibration
from triangulate import triangulate_pose
from pose_exo import ExoPoseEstimator, BODY_SKELETON, HAND_SKELETON


# ── Rerun setup ───────────────────────────────────────────────────────────────

BODY_COLOR       = [0, 220, 255]
LEFT_HAND_COLOR  = [255, 140,  20]
RIGHT_HAND_COLOR = [80,  80, 255]


def init_rerun(cameras, out_rrd):
    rr.init("exo_scene_reconstruction")
    Path(out_rrd).parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out_rrd))

    cam_views = [rrb.Spatial2DView(name=c.name, origin=f"cameras/{c.name}")
                 for c in cameras]
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D Scene", origin="/"),
            rrb.Vertical(*cam_views),
            column_shares=[3, 1],
        ),
        collapse_panels=True,
    ))

    # Static: world axes + ground grid + camera frustums
    rr.log("world/axes", rr.Arrows3D(
        origins=[[0,0,0]]*3,
        vectors=[[0.3,0,0],[0,0.3,0],[0,0,0.3]],
        colors=[[255,0,0],[0,255,0],[0,0,255]],
    ), static=True)

    lines = []
    for v in np.arange(-3, 3.5, 0.5):
        lines += [[[-3, v, 0], [3, v, 0]], [[v, -3, 0], [v, 3, 0]]]
    rr.log("world/ground",
           rr.LineStrips3D(lines, colors=[[40,40,40]]*len(lines), radii=0.003),
           static=True)

    for cam in cameras:
        R_wc = cam.R.T
        t_wc = -(R_wc @ cam.t)
        rr.log(f"cameras/{cam.name}",
               rr.Transform3D(mat3x3=R_wc, translation=t_wc), static=True)
        rr.log(f"cameras/{cam.name}",
               rr.Pinhole(image_from_camera=cam.K,
                          width=cam.width, height=cam.height), static=True)


def log_frame(fi, exo_frames, body_3d, valid, kps_per_cam, cameras, hand_3d=None):
    rr.set_time("frame", sequence=fi)

    # Camera images
    for cam, frame in zip(cameras, exo_frames):
        if frame is None:
            continue
        rr.log(f"cameras/{cam.name}", rr.Image(frame[..., ::-1]))

    # 3D body joints
    valid_pts = body_3d[valid]
    if len(valid_pts):
        rr.log("pose/body/joints",
               rr.Points3D(valid_pts, radii=0.03,
                           colors=[BODY_COLOR]*len(valid_pts)))

    # 3D body skeleton
    lines = []
    for i, j in BODY_SKELETON:
        if i < len(body_3d) and j < len(body_3d) and valid[i] and valid[j]:
            lines.append([body_3d[i], body_3d[j]])
    if lines:
        rr.log("pose/body/skeleton",
               rr.LineStrips3D(lines, colors=[BODY_COLOR]*len(lines), radii=0.01))

    # 2D keypoints overlaid on each camera
    for cam, frame, kps in zip(cameras, exo_frames, kps_per_cam):
        if frame is None or kps is None:
            continue
        vis = frame.copy()
        for x, y, c in kps[:17]:
            if c > 0.3:
                cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 0), -1)
        for i, j in BODY_SKELETON:
            if i < len(kps) and j < len(kps) and kps[i,2] > 0.3 and kps[j,2] > 0.3:
                cv2.line(vis, (int(kps[i,0]), int(kps[i,1])),
                              (int(kps[j,0]), int(kps[j,1])), (0, 200, 255), 2)
        rr.log(f"cameras/{cam.name}", rr.Image(vis[..., ::-1]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--take",
        default="/home/nfs/datasets/external/egoexo4d/dataset/takes/indiana_cooking_21_4")
    parser.add_argument("--frames", type=int, default=60,
                        help="Number of frames to process")
    parser.add_argument("--start", type=int, default=250,
                        help="Start frame index")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    take_dir = Path(args.take)
    if not take_dir.exists():
        sys.exit(f"Take dir not found: {take_dir}")

    out_rrd = Path(args.out) if args.out else \
              Path(__file__).parent.parent / "output" / "exo_scene.rrd"

    # ── Calibration ───────────────────────────────────────────────────────────
    print("[1/3] Loading GoPro calibrations ...")
    cameras = load_gopro_calibration(str(take_dir))
    print(f"      {len(cameras)} cameras: {[c.name for c in cameras]}")

    # ── Frame paths ───────────────────────────────────────────────────────────
    frames_root = take_dir / "frames_from_videos"
    frame_lists = {}
    for cam in cameras:
        files = sorted((frames_root / cam.name).glob("*.png"))
        frame_lists[cam.name] = files
        print(f"      {cam.name}: {len(files)} frames available")

    n_frames = min(len(v) for v in frame_lists.values())
    end      = min(args.start + args.frames, n_frames)
    indices  = list(range(args.start, end))
    print(f"      Processing frames {args.start}..{end-1} ({len(indices)} total)")

    # ── Pose model ────────────────────────────────────────────────────────────
    print("[2/3] Loading RTMPose wholebody model ...")
    pose_model = ExoPoseEstimator()

    # ── Rerun ─────────────────────────────────────────────────────────────────
    init_rerun(cameras, out_rrd)

    # ── Per-frame loop ────────────────────────────────────────────────────────
    print(f"[3/3] Triangulating {len(indices)} frames ...")
    t0 = time.time()

    for fi_rel, fi_abs in enumerate(indices):
        # Load frames
        exo_frames = []
        for cam in cameras:
            files = frame_lists[cam.name]
            img = cv2.imread(str(files[fi_abs])) if fi_abs < len(files) else None
            exo_frames.append(img)

        # 2D pose per camera
        kps_per_cam = []
        for cam, frame in zip(cameras, exo_frames):
            kps = pose_model.estimate(frame) if frame is not None else None
            kps_per_cam.append(kps)

        # DLT triangulation → 3D body
        body_3d, valid = triangulate_pose(kps_per_cam, cameras, conf_thresh=0.3)

        # Log
        log_frame(fi_rel, exo_frames, body_3d, valid, kps_per_cam, cameras)

        if fi_rel % 10 == 0:
            elapsed = time.time() - t0
            fps = (fi_rel + 1) / max(elapsed, 1e-3)
            n_valid = valid.sum() if len(valid) else 0
            print(f"      frame {fi_rel+1}/{len(indices)}  "
                  f"({fps:.1f} fps)  {n_valid} joints triangulated")

    elapsed = time.time() - t0
    print(f"\n      Done: {len(indices)} frames in {elapsed:.1f}s "
          f"({len(indices)/elapsed:.1f} fps)")
    print(f"      Open with:  rerun {out_rrd}")


if __name__ == "__main__":
    main()
