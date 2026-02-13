"""Neural sequence component wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models.neural import NeuralSequenceModel


class NeuralComponent:
    """Wrapper for NeuralSequenceModel as a component."""

    def __init__(self, seed: int, **params: Any) -> None:
        self.model = NeuralSequenceModel(seed=seed, **params)

    def fit(
        self,
        features: np.ndarray,
        documents: list[str],
        labels: np.ndarray,
    ) -> "NeuralComponent":
        """Fit the component."""
        self.model.fit(features, documents, labels)
        return self

    def predict_proba(self, features: np.ndarray, documents: list[str]) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(features, documents)
