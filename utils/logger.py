"""
logger.py - per-frame logging utility
writes csv or json lines for each processed frame
"""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

from core.feature_extractor import FrameFeatures
from core.temporal_buffer import TemporalBuffer, BlinkStats
from utils.config import CONFIG


class FrameLogger:
    _CSV_FIELDS = [
        "frame_index", "timestamp", "face_detected", "ear", "mar",
        "head_pitch", "head_yaw", "head_roll", "gaze_x", "gaze_y",
        "raw_label", "stable_label", "blink_count", "blink_rate_per_min",
        "consecutive_closed_frames"]

    def __init__(self, path: Union[str, Path], fps: float =None):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fps = fps or CONFIG.buffer.fps_assumed
        self._file = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._CSV_FIELDS)
        self._writer.writeheader()
        self._frame_index= 0

    def log(
        self,
        features: FrameFeatures,
        raw_label: str,
        stable_label: Optional[str] =None,
        blink_stats: Optional[BlinkStats] = None,
    ) -> None:
        row = {
            "frame_index": self._frame_index,
            "timestamp": self._frame_index /self._fps,
            "face_detected": features.face_detected,
            "ear": features.ear,
            "mar": features.mar,
            "head_pitch": features.head_pitch,
            "head_yaw": features.head_yaw,
            "head_roll": features.head_roll,
            "gaze_x": features.gaze_x,
            "gaze_y": features.gaze_y,
            "raw_label": raw_label,
            "stable_label": stable_label if stable_label is not None else raw_label,
            "blink_count": blink_stats.blink_count if blink_stats else 0,
            "blink_rate_per_min": blink_stats.blink_rate_per_min if blink_stats else 0.0,
            "consecutive_closed_frames": blink_stats.consecutive_closed_frames if blink_stats else 0 }
        
        self._writer.writerow(row)
        self._frame_index +=1

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val,exc_tb):
        self.close()


class JSONLogger:
    def __init__(self, path: Union[str, Path], fps: float = None):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fps = fps or CONFIG.buffer.fps_assumed
        self._file = open(self._path, "w", encoding="utf-8")
        self._frame_index =0

    def log(
        self,
        features: FrameFeatures,
        raw_label: str,
        stable_label: Optional[str] =None,
        blink_stats: Optional[BlinkStats] = None,
    ) -> None:
        record = {
            "frame_index": self._frame_index,
            "timestamp": self._frame_index / self._fps,
            "features": asdict(features),
            "raw_label": raw_label,
            "stable_label": stable_label if stable_label is not None else raw_label,
            "blink_count": blink_stats.blink_count if blink_stats else 0,
            "blink_rate_per_min": blink_stats.blink_rate_per_min if blink_stats else 0.0,
            "consecutive_closed_frames": blink_stats.consecutive_closed_frames if blink_stats else 0}
        
        self._file.write(json.dumps(record, default=str) + "\n")
        self._frame_index += 1

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def make_logger(
    path: Union[str, Path],
    fmt: str = "csv",
    fps: float = None,
) -> Union[FrameLogger, JSONLogger]:
    fmt = fmt.lower().strip().lstrip(".")
    if fmt == "json":
        return JSONLogger(path,fps=fps)
    return FrameLogger(path, fps=fps)
