"""
config.py - settings and thresholds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

DEFAULT_TEST_VIDEO = BASE_DIR / "test_video.mp4"
SVM_MODEL_PATH = MODELS_DIR / "svm_model.joblib"
RF_MODEL_PATH = MODELS_DIR / "rf_model.joblib"
SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"


@dataclass
class MediaPipeConfig:
    max_num_faces: int = 1
    refine_landmarks:bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class FeatureThresholds:
    ear_threshold: float = 0.21
    ear_microsleep_frames: int = 30
    ear_drowsy_frames: int = 15
    mar_yawn_threshold: float = 0.6
    yawn_consec_frames:int = 15
    blink_rate_window_sec: float = 60.0
    head_yaw_distraction_deg: float = 25.0
    head_pitch_distraction_deg: float = 20.0
    distraction_consec_frames: int = 10
    gaze_off_center_threshold:float = 0.35


@dataclass
class TemporalBufferConfig:
    buffer_size: int = 90
    fps_assumed:int = 30


@dataclass
class SmoothingConfig:
    window_size:int = 9
    min_state_persistence: int = 5


@dataclass
class MLConfig:
    random_state: int = 42
    test_size: float = 0.2
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    svm_gamma: str = "scale"
    svm_class_weight:str = "balanced"
    rf_n_estimators: int = 200
    rf_max_depth:int = 12
    rf_class_weight: str = "balanced"


@dataclass
class DriverStates:
    ALERT:str = "Alert"
    DROWSY: str = "Drowsy"
    MICROSLEEP:str = "Microsleep"
    DISTRACTED: str = "Distracted"
    FNV: str ="FNV"


@dataclass
class AppConfig:
    mediapipe: MediaPipeConfig =field(default_factory=MediaPipeConfig)
    thresholds: FeatureThresholds = field(default_factory=FeatureThresholds)
    buffer: TemporalBufferConfig = field(default_factory=TemporalBufferConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    states: DriverStates = field(default_factory=DriverStates)


CONFIG =AppConfig()
