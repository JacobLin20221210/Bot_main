"""LLM-augmented semantic component using prompted embedding views."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.embeddings import EmbeddingEncoder


class LLMSemanticLinearComponent:
    """Embedding fusion component with prompted semantic views."""

    def __init__(
        self,
        seed: int,
        c: float,
        bot_weight: float,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        self.encoder = EmbeddingEncoder(
            cache_dir="output/cache/embeddings/llm_semantic",
            model_name=embedding_model_name,
        )
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c,
                        class_weight={0: 1.0, 1: bot_weight},
                        max_iter=1600,
                        random_state=seed,
                    ),
                ),
            ]
        )

    @staticmethod
    def _prompted_view(document: str) -> str:
        lowered = document.lower()
        post_count = lowered.count("__post_sep__") + (1 if lowered.strip() else 0)
        signal_line = (
            f"signals: urls={lowered.count('__url__')} mentions={lowered.count('__mention__')} "
            f"hashtags={lowered.count('__hashtag__')} posts={post_count} chars={len(document)}"
        )
        prompt = (
            "Task: infer if this multilingual social account behaves like automation or human communication. "
            "Focus on repetitive templates, promotional cues, cadence regularity, and profile/timeline consistency.\n"
            f"{signal_line}\n"
            f"account_document: {document[:3200]}"
        )
        return prompt

    def _matrix(self, documents: list[str]) -> np.ndarray:
        prompted = [self._prompted_view(document) for document in documents]
        base_emb = self.encoder.encode(documents)
        prompt_emb = self.encoder.encode(prompted)
        delta = np.abs(prompt_emb - base_emb)
        return np.hstack([base_emb, prompt_emb, delta]).astype(np.float32)

    def fit(self, _: np.ndarray, documents: list[str], labels: np.ndarray) -> "LLMSemanticLinearComponent":
        train_x = self._matrix(documents)
        self.pipeline.fit(train_x, labels)
        return self

    def predict_proba(self, _: np.ndarray, documents: list[str]) -> np.ndarray:
        test_x = self._matrix(documents)
        return self.pipeline.predict_proba(test_x)[:, 1]
