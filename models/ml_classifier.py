"""
ml_classifier.py - base class for svm/random forest

shared fit/predict/evaluate/save/load
"""

from abc import ABC, abstractmethod
from pathlib import Path
from time import perf_counter
from typing import Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

from utils.config import CONFIG


class BaseMLClassifier(ABC):
    name:str ="base"

    def __init__(self,model_path: Path, scaler_path: Path = None, config=None):
        self._config = config or CONFIG.ml
        self._model_path = Path(model_path)
        self._scaler_path = Path(scaler_path) if scaler_path else None
        self._model =self._build_model()
        self._scaler: StandardScaler =StandardScaler()
        self._is_fitted =False

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    def fit(self,X: np.ndarray, y:np.ndarray) -> None:
        X_scaled= self._scaler.fit_transform(X)
        self._model.fit(X_scaled,y)
        self._is_fitted= True

    def predict(self,X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                f"Model '{self.name}' not trained. Call fit() or load()."
            )
        X_scaled = self._scaler.transform(np.atleast_2d(X))
        return self._model.predict(X_scaled)

    def predict_single(self, feature_vector: np.ndarray) -> Tuple[str,float]:
        start = perf_counter()
        label = self.predict(feature_vector.reshape(1, -1))[0]
        latency_ms = (perf_counter() - start) *1000.0
        return label,latency_ms

    def evaluate(self,X_test: np.ndarray, y_test: np.ndarray, n_latency_samples: int = 200) -> dict:
        predictions =self.predict(X_test)

        rng = np.random.default_rng(CONFIG.ml.random_state)
        n_samples = min(n_latency_samples,len(X_test))
        sample_idx = rng.choice(len(X_test),size=n_samples, replace=False)

        latencies = []
        for idx in sample_idx:
            start = perf_counter()
            _ = self.predict(X_test[idx].reshape(1,-1))
            latencies.append((perf_counter()- start) * 1000.0)

        return {
            "model_name":self.name,
            "accuracy": accuracy_score(y_test,predictions),
            "classification_report":classification_report(
                y_test, predictions,zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "confusion_matrix_labels": sorted(np.unique(y_test).tolist()),
            "avg_latency_ms": float(np.mean(latencies)),
            "std_latency_ms":float(np.std(latencies)),
            "latency_samples_used": int(n_samples),
        }

    def save(self) -> None:
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model,self._model_path)
        if self._scaler_path:
            joblib.dump(self._scaler, self._scaler_path)

    def load(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")
        self._model = joblib.load(self._model_path)
        if self._scaler_path and self._scaler_path.exists():
            self._scaler = joblib.load(self._scaler_path)
        self._is_fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
