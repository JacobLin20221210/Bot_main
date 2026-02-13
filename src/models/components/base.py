"""Base component interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ModelComponent(ABC):
    """Abstract base class for model components."""

    @abstractmethod
    def fit(
        self,
        features: np.ndarray,
        documents: list[str],
        labels: np.ndarray,
    ) -> "ModelComponent":
        """Fit the component."""
        ...

    @abstractmethod
    def predict_proba(self, features: np.ndarray, documents: list[str]) -> np.ndarray:
        """Predict probabilities."""
        ...
