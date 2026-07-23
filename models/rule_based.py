"""
rule_based.py
rule-based classifier using ear/mar/head-pose thresholds
"""

from __future__ import annotations

from core.feature_extractor import FrameFeatures
from core.temporal_buffer import TemporalBuffer
from utils.config import CONFIG


class RuleBasedClassifier:
    def __init__(self, config: type[CONFIG.thresholds] | None = None) -> None:
        self._thresholds = config or CONFIG.thresholds
        self._states = CONFIG.states
        self._yawn_counter = 0
        self._distraction_counter =0

    def predict(self, features: FrameFeatures, buffer: TemporalBuffer) -> str:
        if not features.face_detected:
            self._yawn_counter = 0
            self._distraction_counter =0
            return self._states.FNV

        consecutive_closed = buffer.consecutive_closed_frames()

        if consecutive_closed >= self._thresholds.ear_microsleep_frames:
            return self._states.MICROSLEEP

        if consecutive_closed >= self._thresholds.ear_drowsy_frames:
            return self._states.DROWSY

        if features.mar >= self._thresholds.mar_yawn_threshold:
            self._yawn_counter += 1
        else:
            self._yawn_counter = 0

        if self._yawn_counter >= self._thresholds.yawn_consec_frames:
            return self._states.DROWSY

        head_deviated = (
            abs(features.head_yaw) >= self._thresholds.head_yaw_distraction_deg
            or abs(features.head_pitch) >= self._thresholds.head_pitch_distraction_deg)
        
        gaze_deviation = abs(features.gaze_x - 0.5)
        gaze_deviated = gaze_deviation >= self._thresholds.gaze_off_center_threshold

        if head_deviated or gaze_deviated:
            self._distraction_counter +=1
        else:
            self._distraction_counter = 0

        if self._distraction_counter >= self._thresholds.distraction_consec_frames:
            return self._states.DISTRACTED

        return self._states.ALERT

    def reset(self)-> None:
        self._yawn_counter = 0
        self._distraction_counter = 0
