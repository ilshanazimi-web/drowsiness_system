"""
feature_extractor.py - computes features from landmarks

EAR, MAR, head pose, gaze direction
"""

from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

from core.face_landmarks import FaceLandmarkResult, LANDMARKS


@dataclass
class FrameFeatures:
    ear:float
    mar: float
    head_pitch: float
    head_yaw: float
    head_roll: float
    gaze_x: float
    gaze_y:float
    face_detected: bool = True

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.ear,
                self.mar,
                self.head_pitch,
                self.head_yaw,
                self.head_roll,
                self.gaze_x,
                self.gaze_y],
            
            dtype=np.float32)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def feature_names():
        return [
            "ear", "mar", "head_pitch", "head_yaw",
            "head_roll", "gaze_x", "gaze_y"]
        

    @staticmethod
    def empty() -> "FrameFeatures":
        return FrameFeatures(
            ear=0.0, mar=0.0, head_pitch= 0.0, head_yaw=0.0,
            head_roll= 0.0, gaze_x=0.0, gaze_y=0.0, face_detected=False)
        
class FeatureExtractor:
    _MODEL_POINTS_3D = np.array(
        [
            (0.0, 0.0,0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0,170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0,-150.0, -125.0),
        ],
        dtype=np.float64)

    @staticmethod
    def _eye_aspect_ratio(eye_points: np.ndarray) -> float:
        p1, p2,p3, p4, p5,p6 = eye_points
        vertical_1 = np.linalg.norm(p2 -p6)
        vertical_2 = np.linalg.norm(p3- p5)
        horizontal =np.linalg.norm(p1 - p4)
        if horizontal == 0:
            return 0.0
        return float((vertical_1 + vertical_2) /(2.0 *horizontal))

    @staticmethod
    def _mouth_aspect_ratio(mouth_points: np.ndarray) -> float:
        left,top1, top2,right, _r2, bottom2,bottom1, *_ = mouth_points
        vertical = np.linalg.norm(top1 - bottom1) + np.linalg.norm(top2 - bottom2)
        horizontal = np.linalg.norm(left -right)
        if horizontal ==0:
            return 0.0
        return float(vertical/(2.0 *horizontal))

    def _head_pose(self, landmark_result: FaceLandmarkResult):
        image_points = landmark_result.get_points(LANDMARKS.HEAD_POSE_2D).astype(np.float64)

        h, w = landmark_result.frame_height, landmark_result.frame_width
        focal_length =w
        center = (w /2.0, h/2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length,center[1]],
                [0,0, 1] ],
            
            dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success,rotation_vec, _translation_vec = cv2.solvePnP(
            self._MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0,0.0, 0.0

        rotation_mat, _ =cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat,np.zeros((3,1))))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch, yaw, roll = (float(a) for a in euler_angles.flatten())
        return pitch, yaw,roll

    @staticmethod
    def _gaze_direction(landmark_result: FaceLandmarkResult):
        try:
            left_iris =landmark_result.get_points(LANDMARKS.LEFT_IRIS).mean(axis=0)
            right_iris = landmark_result.get_points(LANDMARKS.RIGHT_IRIS).mean(axis=0)

            l_corner_a, l_corner_b = landmark_result.get_points(LANDMARKS.LEFT_EYE_CORNERS)
            r_corner_a, r_corner_b = landmark_result.get_points(LANDMARKS.RIGHT_EYE_CORNERS)

            def relative_pos(iris,corner_a,corner_b):
                eye_min = np.minimum(corner_a, corner_b)
                eye_max = np.maximum(corner_a,corner_b)
                span = eye_max -eye_min
                span[span ==0] = 1e-6
                rel = (iris - eye_min) / span
                return rel

            left_rel = relative_pos(left_iris,l_corner_a, l_corner_b)
            right_rel = relative_pos(right_iris,r_corner_a,r_corner_b)
            gaze = (left_rel + right_rel)/2.0
            return float(gaze[0]),float(gaze[1])
        except (IndexError, ValueError):
            return 0.5, 0.5

    def extract(self, landmark_result: Optional[FaceLandmarkResult]) -> FrameFeatures:
        if landmark_result is None:
            return FrameFeatures.empty()

        left_eye = landmark_result.get_points(LANDMARKS.LEFT_EYE)
        right_eye = landmark_result.get_points(LANDMARKS.RIGHT_EYE)
        ear = (self._eye_aspect_ratio(left_eye) + self._eye_aspect_ratio(right_eye)) / 2.0

        mouth = landmark_result.get_points(LANDMARKS.MOUTH)
        mar = self._mouth_aspect_ratio(mouth)

        pitch, yaw, roll= self._head_pose(landmark_result)
        gaze_x,gaze_y =self._gaze_direction(landmark_result)

        return FrameFeatures(
            ear=ear,
            mar=mar,
            head_pitch=pitch,
            head_yaw=yaw,
            head_roll=roll,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            face_detected=True)
        
