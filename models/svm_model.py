"""
svm_model.py - svm classifier
"""

from pathlib import Path

from sklearn.svm import SVC

from models.ml_classifier import BaseMLClassifier
from utils.config import CONFIG, SVM_MODEL_PATH, SCALER_PATH


class SVMClassifier(BaseMLClassifier):
    name = "SVM"

    def __init__(self, model_path: Path = SVM_MODEL_PATH, scaler_path: Path = SCALER_PATH, config=None):
        super().__init__(model_path=model_path, scaler_path=scaler_path, config=config)

    def _build_model(self) -> SVC:
        cfg = self._config
        return SVC(
            kernel=cfg.svm_kernel,
            C =cfg.svm_C,
            gamma=cfg.svm_gamma,
            class_weight=cfg.svm_class_weight,
            probability=True,
            random_state= cfg.random_state)
        
