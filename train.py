"""
train.py - train and compare SVM/Random Forest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from models.model_factory import create_model, available_models
from utils.config import CONFIG, OUTPUT_DIR, DEFAULT_TEST_VIDEO
from utils.dataset_builder import VideoDatasetBuilder
from utils.logger import make_logger


def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser(
        description="train and compare models"
    )
    parser.add_argument(
        "--video",type=str,default=str(DEFAULT_TEST_VIDEO),
        help="video path")
    
    parser.add_argument(
        "--max-frames",type=int,default=None,
        help = "limit frames for quick test",
    )
    parser.add_argument(
        "--models",nargs="+",default=available_models(),
        choices=available_models(),
        help ="models to train",
    )
    parser.add_argument(
        "--log",nargs="?",const=str(OUTPUT_DIR/"dataset_log.csv"),
        help ="save per-frame log. omit path for default.")
    
    parser.add_argument(
        "--log-format",type=str,default="csv",choices=["csv","json"],
        help= "log format")
    
    return parser.parse_args()


def build_dataset(
    video_path:Path,
    max_frames:int | None,
    log_path:str|None = None,
    log_format:str = "csv",
)->tuple[np.ndarray,np.ndarray]:
    print(f"[1/4] extracting features from: {video_path}")
    builder=VideoDatasetBuilder(video_path)
    samples = builder.build(
        max_frames= max_frames,
        log_path =log_path,
        log_format = log_format)

    if len(samples)== 0:
        print("Error: no frames processed.",file=sys.stderr)
        sys.exit(1)

    X,y = builder.to_arrays(samples)
    print(f"  Total samples: {len(y)}")

    unique,counts=np.unique(y,return_counts=True)
    print(" Label distribution (Rule-Based labels):")
    for label,count in zip(unique,counts):
        print(f" {label}: {count} ({count/len(y)*100:.1f}%)")

    if len(unique)<2:
        print(
            "\nWarning: only one state in video. need at least two classes\n"
            "Use a video with mixed states or tune thresholds in utils/config.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    dominant_idx=int(np.argmax(counts))
    dominant_ratio=counts[dominant_idx]/len(y)
    if dominant_ratio>0.85:
        print(
            f"\nWarning: label '{unique[dominant_idx]}' is {dominant_ratio*100:.1f}% of data.\n"
            "Dataset is imbalanced. check per-class precision/recall.",
            file=sys.stderr,
        )

    return X,y


def split_dataset(X:np.ndarray,y:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    print("[2/4] splitting train/test")
    cfg=CONFIG.ml

    unique,counts=np.unique(y,return_counts=True)
    can_stratify=np.all(counts>=2) and len(unique)>1

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y if can_stratify else None,
    )
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")
    return X_train, X_test, y_train,y_test

def train_and_evaluate(
    model_names:list[str],
    X_train: np.ndarray,
    y_train : np.ndarray,
    X_test: np.ndarray,
    y_test : np.ndarray,
)->dict[str,dict]:
    print("[3/4] training and evaluating")
    results= {}

    for model_name in model_names:
        print(f"\n    >>> {model_name}")
        model=create_model(model_name)
        model.fit(X_train,y_train)
        model.save()

        report=model.evaluate(X_test,y_test)
        results[model_name]=report

        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Avg latency: {report['avg_latency_ms']:.4f} ms "
              f"(+/- {report['std_latency_ms']:.4f}, {report['latency_samples_used']} samples)")

    return results

def _build_model_report_lines(report:dict)->list[str]:
    lines=[
        f"Model: {report['model_name']}",
        f"  Accuracy: {report['accuracy']:.4f}",
        f"  Avg latency: {report['avg_latency_ms']:.4f} ms",
        f" Std: {report['std_latency_ms']:.4f} ms",
        f" Samples: {report['latency_samples_used']}",
        " Report:",
        report["classification_report"],
        f"  Labels: {report['confusion_matrix_labels']}",
        "  Matrix:",
        str(report["confusion_matrix"]),
    ]
    return lines


def write_comparison_report(results:dict)->Path:
    print("[4/4] writing report")
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    report_path=OUTPUT_DIR/"comparison_report.txt"

    lines=["Driver State Detection Comparison Report","="*60,""]
    for report in results.values():
        lines.extend(_build_model_report_lines(report))
        lines.append("-"*30)

    report_path.write_text("\n".join(lines),encoding="utf-8")
    print(f"    Report saved to: {report_path}")
    return report_path


def main()->None:
    args=parse_args()
    video_path=Path(args.video)

    X,y=build_dataset(video_path,args.max_frames,log_path=args.log,log_format=args.log_format)
    X_train,X_test,y_train,y_test=split_dataset(X,y)
    results=train_and_evaluate(args.models,X_train,y_train,X_test,y_test)
    write_comparison_report(results)

    print("\nDone.")


if __name__=="__main__":
    main()
