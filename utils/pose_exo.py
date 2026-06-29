"""
2D whole-body pose estimation on exo (third-person) frames using RTMPose via rtmlib.
RTMPose is fast (~30fps on CPU, >100fps on GPU) and accurate for whole body.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# COCO-Wholebody joint names (133 keypoints)
WHOLEBODY_KEYPOINT_NAMES = [
    # Body (17)
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    # Foot (6)
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    # Face (68)
    *[f"face_{i}" for i in range(68)],
    # Left hand (21)
    *[f"left_hand_{i}" for i in range(21)],
    # Right hand (21)
    *[f"right_hand_{i}" for i in range(21)],
]

# Body skeleton connections (subset of COCO 17 for visualization)
BODY_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 11), (6, 12), (11, 12),                  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),      # legs
]

# Hand skeleton connections (21 landmarks, wrist at 0)
HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
]


def load_rtmpose_wholebody():
    """
    Load RTMPose Wholebody model via rtmlib.
    Falls back to body-only if wholebody unavailable.
    """
    try:
        from rtmlib import Wholebody
        model = Wholebody(
            to_openpose=False,
            mode='balanced',      # 'lightweight' < 'balanced' < 'performance'
            backend='onnxruntime',
            device='cpu',         # change to 'cuda' if GPU available
        )
        logger.info("Loaded RTMPose Wholebody (rtmlib)")
        return model, 'rtmlib'
    except ImportError:
        logger.warning("rtmlib not available, trying mmpose...")

    try:
        from mmpose.apis import init_model, inference_topdown
        from mmpose.utils import adapt_mmdet_pipeline
        # RTMPose-x wholebody config
        cfg = 'td-hm_RTMPose-x_8xb32-270e_coco-wholebody-384x288'
        ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-coco-wholebody_pt-body7_270e-384x288-401dfc90_20230629.pth'
        model = init_model(cfg, ckpt, device='cpu')
        logger.info("Loaded RTMPose via mmpose")
        return model, 'mmpose'
    except ImportError:
        logger.warning("mmpose not available, using MediaPipe fallback")
        return None, 'mediapipe'


class ExoPoseEstimator:
    def __init__(self):
        self.model, self.backend = load_rtmpose_wholebody()
        if self.backend == 'mediapipe':
            self._init_mediapipe()

    def _init_mediapipe(self):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.pose_model = self.mp_pose.Pose(
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        logger.info("Using MediaPipe Pose as fallback (33 body joints, no hands/face)")

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run pose estimation on one frame.

        Returns:
            keypoints: (N_joints, 3) array [x, y, confidence], or None
        """
        if self.backend == 'rtmlib':
            return self._estimate_rtmlib(frame)
        elif self.backend == 'mmpose':
            return self._estimate_mmpose(frame)
        else:
            return self._estimate_mediapipe(frame)

    def _estimate_rtmlib(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            keypoints, scores = self.model(frame)
            if keypoints is None or len(keypoints) == 0:
                return None
            # Take first person (highest confidence)
            kps = keypoints[0]     # (N, 2)
            sc = scores[0]         # (N,)
            return np.hstack([kps, sc.reshape(-1, 1)]).astype(np.float32)
        except Exception as e:
            logger.warning(f"rtmlib estimate failed: {e}")
            return None

    def _estimate_mmpose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            from mmpose.apis import inference_topdown
            # Dummy bbox covering full image
            h, w = frame.shape[:2]
            bbox = np.array([[0, 0, w, h, 1.0]])
            results = inference_topdown(self.model, frame, bbox)
            if not results:
                return None
            kps = results[0].pred_instances.keypoints[0]    # (N, 2)
            sc = results[0].pred_instances.keypoint_scores[0]  # (N,)
            return np.hstack([kps, sc.reshape(-1, 1)]).astype(np.float32)
        except Exception as e:
            logger.warning(f"mmpose estimate failed: {e}")
            return None

    def _estimate_mediapipe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose_model.process(rgb)
        if not result.pose_landmarks:
            return None
        h, w = frame.shape[:2]
        kps = []
        for lm in result.pose_landmarks.landmark:
            kps.append([lm.x * w, lm.y * h, lm.visibility])
        return np.array(kps, dtype=np.float32)


def draw_pose_2d(frame: np.ndarray, keypoints: np.ndarray,
                 conf_thresh: float = 0.3) -> np.ndarray:
    """Draw skeleton on frame for debug visualization."""
    vis = frame.copy()
    # Draw body joints
    for idx, (x, y, c) in enumerate(keypoints[:17]):
        if c > conf_thresh:
            cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)
    # Draw body skeleton
    for i, j in BODY_SKELETON:
        if i < len(keypoints) and j < len(keypoints):
            if keypoints[i, 2] > conf_thresh and keypoints[j, 2] > conf_thresh:
                p1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                p2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                cv2.line(vis, p1, p2, (0, 200, 255), 2)
    return vis
