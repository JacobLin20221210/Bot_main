"""Text-based logistic regression component."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TextLinearComponent:
    """Logistic regression on TF-IDF text features."""

    def __init__(
        self,
        seed: int,
        c: float,
        bot_weight: float,
        word_max_features: int,
        char_max_features: int,
    ) -> None:
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=word_max_features,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=char_max_features,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        self.classifier = LogisticRegression(
            C=c,
            class_weight={0: 1.0, 1: bot_weight},
            max_iter=1600,
            random_state=seed,
        )

    def _matrix(self, documents: list[str], fit: bool) -> Any:
        if fit:
            word = self.word_vectorizer.fit_transform(documents)
            char = self.char_vectorizer.fit_transform(documents)
        else:
            word = self.word_vectorizer.transform(documents)
            char = self.char_vectorizer.transform(documents)
        return hstack([word, char], format="csr")

    def fit(
        self,
        _: np.ndarray,
        documents: list[str],
        labels: np.ndarray,
    ) -> "TextLinearComponent":
        """Fit the component."""
        train_x = self._matrix(documents, fit=True)
        self.classifier.fit(train_x, labels)
        return self

    def predict_proba(self, _: np.ndarray, documents: list[str]) -> np.ndarray:
        """Predict probabilities."""
        test_x = self._matrix(documents, fit=False)
        return self.classifier.predict_proba(test_x)[:, 1]
