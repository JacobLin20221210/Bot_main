"""Neural sequence model for bot detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class NeuralSequenceModel:
    """Neural network model combining tabular and text features."""

    seed: int = 42
    hidden_layer_sizes: tuple[int, ...] = (256, 128)
    alpha: float = 1e-4
    max_iter: int = 350
    text_svd_components: int = 128
    word_max_features: int = 12000
    char_max_features: int = 18000

    def __post_init__(self) -> None:
        self.tab_imputer = SimpleImputer(strategy="median")
        self.tab_scaler = StandardScaler()
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=self.word_max_features,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=self.char_max_features,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        self.text_svd: TruncatedSVD | None = None
        self.mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            learning_rate="adaptive",
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=self.seed,
        )

    def _fit_text_features(self, documents: list[str]) -> np.ndarray:
        word = self.word_vectorizer.fit_transform(documents)
        char = self.char_vectorizer.fit_transform(documents)
        merged = hstack([word, char], format="csr")

        n_samples, n_features = merged.shape
        max_components = min(self.text_svd_components, n_samples - 1, n_features - 1)
        if max_components >= 2:
            self.text_svd = TruncatedSVD(n_components=max_components, random_state=self.seed)
            return self.text_svd.fit_transform(merged)

        self.text_svd = None
        return merged.toarray()

    def _transform_text_features(self, documents: list[str]) -> np.ndarray:
        word = self.word_vectorizer.transform(documents)
        char = self.char_vectorizer.transform(documents)
        merged = hstack([word, char], format="csr")

        if self.text_svd is not None:
            return self.text_svd.transform(merged)
        return merged.toarray()

    def _prepare_features(
        self,
        tabular_features: np.ndarray,
        documents: list[str],
        fit: bool,
    ) -> np.ndarray:
        if fit:
            tab = self.tab_scaler.fit_transform(self.tab_imputer.fit_transform(tabular_features))
            text = self._fit_text_features(documents)
        else:
            tab = self.tab_scaler.transform(self.tab_imputer.transform(tabular_features))
            text = self._transform_text_features(documents)

        return np.hstack([tab, text]).astype(float)

    def fit(
        self,
        tabular_features: np.ndarray,
        documents: list[str],
        labels: np.ndarray,
    ) -> "NeuralSequenceModel":
        """Fit the model."""
        features = self._prepare_features(tabular_features, documents, fit=True)
        self.mlp.fit(features, labels)
        return self

    def predict_proba(
        self,
        tabular_features: np.ndarray,
        documents: list[str],
    ) -> np.ndarray:
        """Predict probabilities."""
        features = self._prepare_features(tabular_features, documents, fit=False)
        return self.mlp.predict_proba(features)[:, 1]
