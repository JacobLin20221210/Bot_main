"""Embedding-based logistic regression component."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.features.embeddings import EmbeddingEncoder


class EmbeddingLinearComponent:
    """Logistic regression on sentence embeddings."""

    def __init__(
        self,
        seed: int,
        c: float,
        bot_weight: float,
        embedding_model_name: str = "intfloat/multilingual-e5-small",
    ) -> None:
        self.encoder = EmbeddingEncoder(
            cache_dir="output/cache/embeddings/train",
            model_name=embedding_model_name,
        )
        self.classifier = LogisticRegression(
            C=c,
            class_weight={0: 1.0, 1: bot_weight},
            max_iter=1400,
            random_state=seed,
        )

    def fit(
        self,
        _: np.ndarray,
        documents: list[str],
        labels: np.ndarray,
    ) -> "EmbeddingLinearComponent":
        """Fit the component."""
        train_x = self.encoder.encode(documents)
        self.classifier.fit(train_x, labels)
        return self

    def predict_proba(self, _: np.ndarray, documents: list[str]) -> np.ndarray:
        """Predict probabilities."""
        test_x = self.encoder.encode(documents)
        return self.classifier.predict_proba(test_x)[:, 1]
