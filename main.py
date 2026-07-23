# -*- coding: utf-8 -*-
"""
main.py - runs the full detection pipeline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import cv2.typing
import absl.logging

os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")
absl.logging.set_verbosity(absl.logging.ERROR)

warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype\\(\\) is deprecated",
    category=UserWarning,
)

from core.face_landmarks import FaceLandmarkExtractor
from core.feature_extractor import FeatureExtractor
from core.temporal_buffer import TemporalBuffer
from models.rule_based import RuleBasedClassifier
from models.model_factory import create_model
from utils.logger import make_logger
from utils.smoothing import TemporalSmoother
from utils.config import CONFIG, DEFAULT_TEST_VIDEO, OUTPUT_DIR

_BG_COLOR = (30, 30, 30)
_BG_HEIGHT = 70
_TEXT_PRIMARY = (0, 200, 0)
_TEXT_SECONDARY = (200, 200, 200)
_TEXT_X = 15
_TEXT_Y1 = 30
_TEXT_Y2 = 58
_BORDER_WARN = (0, 0, 255)
_BORDER_WARN_THICK = 6

STATE_COLORS: dict[str, tuple[int, int, int]] = {
    CONFIG.states.ALERT: (0, 200, 0),
    CONFIG.states.DROWSY: (0, 165, 255),
    CONFIG.states.MICROSLEEP: (0, 0, 255),
    CONFIG.states.DISTRACTED: (255, 0, 255),
    CONFIG.states.FNV: (128, 128, 128),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Driver drowsiness and impaired-alertness detection"
    )
    parser.add_argument(
        "--video", type=str, default=str(DEFAULT_TEST_VIDEO),
        help="Path to the input video, or 0 for webcam",
    )
    parser.add_argument(
        "--algorithm", type=str, default="rule_based",
        choices=["rule_based", "svm", "random_forest"],
        help="algorithm to determine driver state",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="disable the video preview window",
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="save annotated output video under output/",
    )
    parser.add_argument(
        "--log", nargs="?", const=str(OUTPUT_DIR / "session_log.csv"),
        help="Save per-frame log to CSV or JSONL. Omit path for default.",
    )
    parser.add_argument(
        "--log-format", type=str, default="csv", choices=["csv", "json"],
        help="format for per-frame log file",
    )
    return parser.parse_args()


def build_classifier(algorithm: str) -> tuple[RuleBasedClassifier | None, object | None]:
    if algorithm == "rule_based":
        return RuleBasedClassifier(), None

    model = create_model(algorithm)
    try:
        model.load()
    except FileNotFoundError as exc:
        print(
            f"Error: {exc}\n"
            f"Train '{algorithm}' first by running train.py:\n"
            f"    python train.py --video {DEFAULT_TEST_VIDEO}",
            file=sys.stderr,
        )
        sys.exit(1)
    return None, model


def _build_video_writer(output_path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)


def draw_overlay(frame: cv2.typing.MatLike, state: str, fps: float, algorithm: str) -> cv2.typing.MatLike:
    color = STATE_COLORS.get(state, (255, 255, 255))
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, _BG_HEIGHT), _BG_COLOR, thickness=-1)
    cv2.putText(frame, f"Status: {state}", (_TEXT_X, _TEXT_Y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Algorithm: {algorithm} | FPS: {fps:.1f}", (_TEXT_X, _TEXT_Y2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _TEXT_SECONDARY, 1, cv2.LINE_AA)
    if state in (CONFIG.states.MICROSLEEP, CONFIG.states.DROWSY):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), _BORDER_WARN, thickness=_BORDER_WARN_THICK)
    return frame


def run_pipeline(args: argparse.Namespace) -> None:
    video_source: int | str = 0 if args.video == "0" else args.video
    if isinstance(video_source, str) and not Path(video_source).exists():
        print(f"Error: input video not found: {video_source}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: could not open video source: {video_source}", file=sys.stderr)
        sys.exit(1)

    rule_based_classifier, ml_model = build_classifier(args.algorithm)

    feature_extractor = FeatureExtractor()
    buffer = TemporalBuffer()
    smoother = TemporalSmoother()

    writer: cv2.VideoWriter | None = None
    if args.save_video:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"result_{args.algorithm}.mp4"
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = _build_video_writer(out_path, fps_in, (w, h))
        print(f"Output video will be saved to: {out_path}")

    print(f"Starting processing with algorithm: {args.algorithm}")
    print("Press 'q' in the preview window to quit.")

    frame_count = 0
    prev_time = time.perf_counter()

    video_fps = cap.get(cv2.CAP_PROP_FPS) or CONFIG.buffer.fps_assumed

    logger = None
    if args.log:
        logger = make_logger(args.log, fmt=args.log_format, fps=video_fps)
        print(f"Per-frame log enabled: {args.log} ({args.log_format})")

    with FaceLandmarkExtractor() as landmark_extractor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmark_result = next(iter(landmark_extractor.process(frame)), None)
            features = feature_extractor.extract(landmark_result)
            buffer.push(features)

            if args.algorithm == "rule_based":
                raw_label = rule_based_classifier.predict(features, buffer)
            else:
                if not features.face_detected:
                    raw_label = CONFIG.states.FNV
                else:
                    raw_label = ml_model.predict(features.to_vector().reshape(1, -1))[0]

            stable_state = smoother.update(raw_label)

            if logger is not None:
                blink_stats = buffer.compute_blink_stats(fps=video_fps)
                logger.log(features, raw_label, stable_state, blink_stats=blink_stats)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            frame = draw_overlay(frame, stable_state, fps, args.algorithm)

            if writer is not None:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Driver Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"    Processed frame {frame_count} | current state: {stable_state}")

    cap.release()
    if writer is not None:
        writer.release()
    if logger is not None:
        logger.close()
    if not args.no_display:
        cv2.destroyAllWindows()

    print(f"Processing finished. Total frames processed: {frame_count}")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
