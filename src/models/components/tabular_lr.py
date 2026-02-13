"""Tabular logistic regression component."""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TabularLogisticComponent:
    """Logistic regression on tabular features."""

    def __init__(self, seed: int, c: float, bot_weight: float) -> None:
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c,
                        class_weight={0: 1.0, 1: bot_weight},
                        max_iter=1200,
                        random_state=seed,
                    ),
                ),
            ]
        )

    def fit(
        self,
        features: np.ndarray,
        _: list[str],
        labels: np.ndarray,
    ) -> "TabularLogisticComponent":
        """Fit the component."""
        self.pipeline.fit(features, labels)
        return self

    def predict_proba(self, features: np.ndarray, _: list[str]) -> np.ndarray:
        """Predict probabilities."""
        return self.pipeline.predict_proba(features)[:, 1]
