"""
dataset_builder.py - build labeled dataset from video
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from core.face_landmarks import FaceLandmarkExtractor
from core.feature_extractor import FeatureExtractor, FrameFeatures
from core.temporal_buffer import TemporalBuffer
from models.rule_based import RuleBasedClassifier
from utils.config import CONFIG, OUTPUT_DIR
from utils.logger import make_logger


@dataclass
class DatasetSample:
    features: np.ndarray
    label: str
    frame_index: int


class VideoDatasetBuilder:
    def __init__(self,video_path: Path | str):
        self._video_path = Path(video_path)
        if not self._video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self._video_path}\n"
                "put a video next to the project")
            

    def build(
        self,
        max_frames: int | None =None,
        show_progress : bool = True,
        log_path: Union[str, Path] | None = None,
        log_format:str = "csv",
    ) -> List[DatasetSample]:
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {self._video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames =min(total_frames, max_frames)

        video_fps =cap.get(cv2.CAP_PROP_FPS) or CONFIG.buffer.fps_assumed
        samples: List[DatasetSample] = []
        logger = None
        if log_path:
            logger = make_logger(log_path, fmt=log_format, fps=video_fps)

        with FaceLandmarkExtractor() as landmark_extractor:
            feature_extractor = FeatureExtractor()
            buffer =TemporalBuffer()
            rule_based = RuleBasedClassifier()
            frame_idx = 0
            iterator =tqdm(total=total_frames, disable=not show_progress, desc="extracting")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and frame_idx >= max_frames:
                    break

                all_faces = landmark_extractor.process(frame)
                driver_face = all_faces[0] if all_faces else None
                features = feature_extractor.extract(driver_face)
                buffer.push(features)
                label = rule_based.predict(features, buffer)

                samples.append(
                    DatasetSample(
                        features=features.to_vector(),
                        label=label,
                        frame_index=frame_idx))

                if logger is not None:
                    blink_stats = buffer.compute_blink_stats(fps=video_fps)
                    logger.log(features, label, stable_label=label, blink_stats=blink_stats)

                frame_idx +=1
                iterator.update(1)

            iterator.close()

        cap.release()
        if logger is not None:
            logger.close()
        return samples

    @staticmethod
    def to_arrays(samples: List[DatasetSample]) -> Tuple[np.ndarray,np.ndarray]:
        X = np.stack([s.features for s in samples])
        y = np.array([s.label for s in samples])
        return X, y
