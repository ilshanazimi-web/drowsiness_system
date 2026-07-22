# Driver Drowsiness and Impaired-Alertness Detection System

Modular implementation covering the software stages (hardware excluded).
Rule-Based, SVM, and Random Forest are implemented. MLP / CNN / LSTM / TCN
can be added later without changing the pipeline.

## Project Structure

```
drowsiness_system/
├── core/
│   ├── face_landmarks.py
│   ├── feature_extractor.py
│   └── temporal_buffer.py
├── models/
│   ├── rule_based.py
│   ├── ml_classifier.py
│   ├── svm_model.py
│   ├── random_forest_model.py
│   └── model_factory.py
├── utils/
│   ├── config.py
│   ├── dataset_builder.py
│   ├── logger.py
│   └── smoothing.py
├── data/
├── output/
├── main.py
├── train.py
└── requirements.txt
```

## Installation

Required Python: 3.9 to 3.12
mediapipe==0.10.13 has no wheel for 3.13+ or 3.8-.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train and compare

```bash
python train.py --video test_video.mp4
python train.py --video test_video.mp4 --max-frames 3000
python train.py --video test_video.mp4 --models svm
```

Output: `models/svm_model.joblib`, `models/rf_model.joblib`,
`models/feature_scaler.joblib`, `output/comparison_report.txt`

### Run the pipeline

```bash
python main.py --video test_video.mp4 --algorithm rule_based
python main.py --video test_video.mp4 --algorithm svm --no-display --save-video
python main.py --video 0 --algorithm rule_based
```

## Pipeline

Video -> FaceLandmarkExtractor -> FeatureExtractor -> TemporalBuffer
-> algorithm -> TemporalSmoother -> Display/Save

## Future Work

1. MLP: add class under `models/` inheriting from `BaseMLClassifier`
2. CNN/LSTM/TCN: use `TemporalBuffer.get_recent_features()`
3. Dashboard: `main.py` already produces needed per-frame data

## Troubleshooting

If one label makes up >85% of samples, head-pose thresholds in
`utils/config.py` are likely miscalibrated. Try raising
`head_yaw_distraction_deg` / `head_pitch_distraction_deg` and re-run.
