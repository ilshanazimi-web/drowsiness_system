"""
random_forest_model.py - random forest classifier
"""

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from models.ml_classifier import BaseMLClassifier
from utils.config import CONFIG, RF_MODEL_PATH, SCALER_PATH


class RandomForestModel(BaseMLClassifier):
    name = "RandomForest"

    def __init__(
        self,
        model_path: Path =RF_MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        config=None,
    ):
        super().__init__(
            model_path=model_path,
            scaler_path=scaler_path,
            config= config,
        )

    def _build_model(self) -> RandomForestClassifier:
        cfg = self._config
        return RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            class_weight= cfg.rf_class_weight,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
