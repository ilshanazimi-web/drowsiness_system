"""
face_landmarks.py
thin wrapper around mediapipe facemesh
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "mediapipe missing - pip install mediapipe==0.10.13"
    ) from exc

from utils.config import CONFIG


@dataclass(frozen=True)
class LandmarkIndices:
    LEFT_EYE =(362, 385, 387, 263, 373, 380)
    RIGHT_EYE = (33, 160, 158, 133, 153, 144)
    MOUTH =(78, 81, 13, 311, 308, 402, 14, 178)
    HEAD_POSE_2D = (1, 152, 33, 263, 61, 291)
    LEFT_IRIS= (468, 469, 470, 471, 472)
    RIGHT_IRIS =(473, 474, 475, 476, 477)
    LEFT_EYE_CORNERS =(362, 263)
    RIGHT_EYE_CORNERS= (33, 133)


LANDMARKS = LandmarkIndices()


class FaceLandmarkExtractor:
    def __init__(self,config=None):
        self._config =config or CONFIG.mediapipe
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=self._config.max_num_faces,
            refine_landmarks=self._config.refine_landmarks,
            min_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> List["FaceLandmarkResult"]:
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return []

        h, w = frame_bgr.shape[:2]
        output: List["FaceLandmarkResult"] = []

        for face_landmarks in results.multi_face_landmarks:
            points_norm = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark],
                dtype=np.float32,
            )
            points_px = points_norm.copy()
            points_px[:, 0] = points_px[:, 0] * w
            points_px[:, 1] = points_px[:, 1] * h
            output.append(FaceLandmarkResult(
                points_normalized=points_norm,
                points_pixel=points_px[:, :2],
                frame_width=w,
                frame_height=h,
            ))

        return output
        
    def close(self) -> None:
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val,exc_tb):
        self.close()


@dataclass
class FaceLandmarkResult:
    points_normalized: np.ndarray
    points_pixel: np.ndarray
    frame_width: int
    frame_height:int

    def get_points(self, indices: List[int]) -> np.ndarray:
        return self.points_pixel[list(indices)]

    def get_center(self) -> tuple[float, float]:
        pts = self.points_pixel
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        return cx, cy

    def get_bbox(self) -> tuple[int, int, int, int]:
        pts = self.points_pixel
        x_min = int(np.min(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))
        x_max = int(np.max(pts[:, 0]))
        y_max = int(np.max(pts[:, 1]))
        return x_min, y_min, x_max, y_max
