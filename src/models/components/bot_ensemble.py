"""Bot ensemble component wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models.ensemble import BotEnsembleModel


class BotEnsembleComponent:
    """Wrapper for BotEnsembleModel as a component."""

    def __init__(self, seed: int, **params: Any) -> None:
        self.model = BotEnsembleModel(seed=seed, **params)

    def fit(
        self,
        features: np.ndarray,
        _: list[str],
        labels: np.ndarray,
    ) -> "BotEnsembleComponent":
        """Fit the component."""
        self.model.fit(features, labels)
        return self

    def predict_proba(self, features: np.ndarray, _: list[str]) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(features)
