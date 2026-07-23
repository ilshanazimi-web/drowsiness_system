"""
temporal_buffer.py

holds recent frame features. used for blink counting, consecutive
closed-eye detection, and future sequence models.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List

from core.feature_extractor import FrameFeatures
from utils.config import CONFIG


@dataclass
class BlinkStats:
    blink_count: int
    blink_rate_per_min:float
    consecutive_closed_frames: int


class TemporalBuffer:
    def __init__(self, config=None):
        self._config =config or CONFIG.buffer
        self._thresholds= CONFIG.thresholds
        self._buffer:Deque[FrameFeatures] = deque(maxlen=self._config.buffer_size)
        self._eye_closed_history: Deque[bool] =deque(maxlen=self._config.buffer_size)
        self._total_frames_seen=0
        self._consecutive_closed=0

    def push(self,features: FrameFeatures) -> None:
        self._buffer.append(features)
        is_closed = features.face_detected and features.ear < self._thresholds.ear_threshold
        self._eye_closed_history.append(is_closed)
        self._consecutive_closed = self._consecutive_closed + 1 if is_closed else 0
        self._total_frames_seen +=1

    def get_recent_features(self, n:int=None) -> List[FrameFeatures]:
        if n is None or n >= len(self._buffer):
            return list(self._buffer)
        return list(self._buffer)[-n:]

    def consecutive_closed_frames(self) -> int:
        return self._consecutive_closed

    def compute_blink_stats(self,fps:float =None) -> BlinkStats:
        fps = fps or self._config.fps_assumed
        history= list(self._eye_closed_history)
        blink_count =0
        i = 0
        while i < len(history):
            if history[i]:
                start =i
                while i < len(history) and history[i]:
                    i += 1
                closed_duration = i -start
                if closed_duration < self._thresholds.ear_microsleep_frames:
                    blink_count +=1
            else:
                i += 1

        duration_sec = max(len(history)/ fps, 1e-6)
        blink_rate_per_min =(blink_count /duration_sec) * 60.0

        return BlinkStats(
            blink_count=blink_count,
            blink_rate_per_min=blink_rate_per_min,
            consecutive_closed_frames=self._consecutive_closed,
        )

    def is_full(self) -> bool:
        return len(self._buffer) == self._config.buffer_size

    def __len__(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
        self._eye_closed_history.clear()
        self._total_frames_seen =0
        self._consecutive_closed=0
