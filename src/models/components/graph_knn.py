"""Relationship-oblivious graph component based on kNN smoothing."""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.embeddings import EmbeddingEncoder


class GraphKNNComponent:
    """Graph-inspired component using feature-similarity neighborhoods."""

    def __init__(
        self,
        seed: int,
        c: float,
        bot_weight: float,
        k_neighbors: int,
        graph_weight: float,
        tabular_weight: float,
        embedding_weight: float,
        temperature: float,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.encoder = EmbeddingEncoder(
            cache_dir="output/cache/embeddings/graph_knn",
            model_name=embedding_model_name,
        )
        self.classifier = LogisticRegression(
            C=c,
            class_weight={0: 1.0, 1: bot_weight},
            max_iter=1400,
            random_state=seed,
        )

        self.k_neighbors = max(3, int(k_neighbors))
        self.graph_weight = float(np.clip(graph_weight, 0.0, 1.0))
        self.tabular_weight = float(max(1e-6, tabular_weight))
        self.embedding_weight = float(max(1e-6, embedding_weight))
        self.temperature = float(max(1e-3, temperature))

        self.train_repr_normalized: np.ndarray | None = None
        self.train_labels: np.ndarray | None = None

    def _build_representation(self, features: np.ndarray, documents: list[str], fit: bool) -> np.ndarray:
        tabular_raw = self.imputer.fit_transform(features) if fit else self.imputer.transform(features)
        tabular = self.scaler.fit_transform(tabular_raw) if fit else self.scaler.transform(tabular_raw)
        embeddings = self.encoder.encode(documents)

        combined = np.hstack(
            [
                self.tabular_weight * tabular,
                self.embedding_weight * embeddings,
            ]
        ).astype(float)
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        return combined / np.maximum(norms, 1e-8)

    def fit(self, features: np.ndarray, documents: list[str], labels: np.ndarray) -> "GraphKNNComponent":
        train_repr = self._build_representation(features, documents, fit=True)
        self.classifier.fit(train_repr, labels)
        self.train_repr_normalized = train_repr
        self.train_labels = labels.astype(float)
        return self

    def _neighbor_probabilities(self, query_repr: np.ndarray) -> np.ndarray:
        if self.train_repr_normalized is None or self.train_labels is None:
            return np.zeros(query_repr.shape[0], dtype=float)

        similarities = np.matmul(query_repr, self.train_repr_normalized.T)
        k = min(self.k_neighbors, similarities.shape[1])
        if k <= 0:
            return np.zeros(query_repr.shape[0], dtype=float)

        top_idx = np.argpartition(similarities, kth=similarities.shape[1] - k, axis=1)[:, -k:]
        top_sim = np.take_along_axis(similarities, top_idx, axis=1)
        top_labels = np.take(self.train_labels, top_idx)

        scaled = np.exp(np.clip(top_sim, -1.0, 1.0) / self.temperature)
        weighted_sum = np.sum(scaled * top_labels, axis=1)
        norm = np.maximum(np.sum(scaled, axis=1), 1e-8)
        return weighted_sum / norm

    def predict_proba(self, features: np.ndarray, documents: list[str]) -> np.ndarray:
        query_repr = self._build_representation(features, documents, fit=False)
        base_prob = self.classifier.predict_proba(query_repr)[:, 1]
        neighbor_prob = self._neighbor_probabilities(query_repr)
        return ((1.0 - self.graph_weight) * base_prob) + (self.graph_weight * neighbor_prob)
