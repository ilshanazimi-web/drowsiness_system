"""
model_factory.py
build a model from a string name
"""

from typing import Dict, Type

from models.ml_classifier import BaseMLClassifier
from models.svm_model import SVMClassifier
from models.random_forest_model import RandomForestModel


_REGISTRY: Dict[str, Type[BaseMLClassifier]] = {
    "svm": SVMClassifier,
    "random_forest": RandomForestModel,
    "rf": RandomForestModel,
}


def create_model(model_name: str, **kwargs) -> BaseMLClassifier:
    key = model_name.strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(set(_REGISTRY.keys())))
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available}"
        )
    return _REGISTRY[key](**kwargs)


def available_models() -> list:
    return list(dict.fromkeys(["svm", "random_forest"]))
