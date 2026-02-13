"""Embedding-based feature extraction."""

from __future__ import annotations

import hashlib
import os
from threading import Lock
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv


def load_env() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path, override=False)


def _get_hf_token() -> str | None:
    load_env()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


EMBEDDING_FEATURE_NAMES = [
    "emb_profile_timeline_cosine",
    "emb_early_late_drift",
    "emb_segment_dispersion",
    "emb_segment_norm_mean",
    "emb_segment_norm_std",
]


class EmbeddingEncoder:
    _SHARED_BACKENDS: dict[str, Any] = {}
    _SHARED_FAILED_MODELS: set[str] = set()
    _SHARED_LOCK = Lock()

    def __init__(
        self,
        cache_dir: str | Path = "output/cache/embeddings",
        model_name: str = "intfloat/multilingual-e5-small",
        fallback_dim: int = 128,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.fallback_dim = fallback_dim

    def _cache_key(self, text: str) -> str:
        payload = f"{self.model_name}::{text}".encode("utf-8", errors="ignore")
        return hashlib.sha1(payload).hexdigest()

    def _cache_path(self, text: str) -> Path:
        return self.cache_dir / f"{self._cache_key(text)}.npy"

    def _fallback_vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.fallback_dim, dtype=np.float32)
        lowered = (text or "").lower().strip()
        if not lowered:
            return vector
        for token in lowered.split():
            idx = int(hashlib.sha1(token.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) % self.fallback_dim
            vector[idx] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def _load_backend(self) -> Any | None:
        shared_backend = self._SHARED_BACKENDS.get(self.model_name)
        if shared_backend is not None:
            return shared_backend
        if self.model_name in self._SHARED_FAILED_MODELS:
            return None

        with self._SHARED_LOCK:
            shared_backend = self._SHARED_BACKENDS.get(self.model_name)
            if shared_backend is not None:
                return shared_backend
            if self.model_name in self._SHARED_FAILED_MODELS:
                return None

            try:
                from sentence_transformers import SentenceTransformer

                token = _get_hf_token()
                shared_backend = SentenceTransformer(self.model_name, token=token)
                self._SHARED_BACKENDS[self.model_name] = shared_backend
                return shared_backend
            except Exception:
                self._SHARED_FAILED_MODELS.add(self.model_name)
                return None

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, self.fallback_dim), dtype=np.float32)

        vectors: list[np.ndarray | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for idx, text in enumerate(texts):
            cache_path = self._cache_path(text)
            if cache_path.exists():
                vectors[idx] = np.load(cache_path)
                continue
            missing_indices.append(idx)
            missing_texts.append(text)

        backend = self._load_backend()
        if missing_texts:
            if backend is not None:
                encoded = backend.encode(
                    missing_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                encoded = np.asarray(encoded, dtype=np.float32)
            else:
                encoded = np.vstack([self._fallback_vector(text) for text in missing_texts]).astype(np.float32)

            for idx, vector in zip(missing_indices, encoded, strict=False):
                vectors[idx] = vector
                np.save(self._cache_path(texts[idx]), vector)

        first = vectors[0]
        if first is None:
            return np.empty((0, self.fallback_dim), dtype=np.float32)
        matrix = np.vstack([vector if vector is not None else np.zeros_like(first) for vector in vectors]).astype(np.float32)
        return matrix


def _safe_cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm + 1e-12))


def _segment_texts(post_texts: list[str]) -> tuple[str, str, str, str]:
    cleaned = [text for text in post_texts if text]
    if not cleaned:
        return "", "", "", ""
    timeline = " __post_sep__ ".join(cleaned)
    n_posts = len(cleaned)
    first_cut = max(1, n_posts // 3)
    second_cut = max(first_cut + 1, (2 * n_posts) // 3)
    early = " __post_sep__ ".join(cleaned[:first_cut])
    middle = " __post_sep__ ".join(cleaned[first_cut:second_cut])
    late = " __post_sep__ ".join(cleaned[second_cut:])
    return timeline, early, middle, late


def build_user_embedding_feature_rows(
    users: list[dict[str, Any]],
    posts_by_author: dict[str, list[dict[str, Any]]],
    cache_dir: str | Path = "output/cache/embeddings",
    model_name: str = "intfloat/multilingual-e5-small",
) -> np.ndarray:
    encoder = EmbeddingEncoder(cache_dir=cache_dir, model_name=model_name)

    profile_texts: list[str] = []
    timeline_texts: list[str] = []
    early_texts: list[str] = []
    middle_texts: list[str] = []
    late_texts: list[str] = []

    for user in users:
        user_id = str(user.get("id") or "")
        posts = posts_by_author.get(user_id, [])
        post_texts = [str(post.get("text") or "").strip() for post in posts]
        timeline, early, middle, late = _segment_texts(post_texts)

        profile = " ".join(
            [
                str(user.get("name") or "").strip(),
                str(user.get("description") or "").strip(),
                str(user.get("location") or "").strip(),
            ]
        ).strip()

        profile_texts.append(profile)
        timeline_texts.append(timeline)
        early_texts.append(early)
        middle_texts.append(middle)
        late_texts.append(late)

    profile_embeddings = encoder.encode(profile_texts)
    timeline_embeddings = encoder.encode(timeline_texts)
    early_embeddings = encoder.encode(early_texts)
    middle_embeddings = encoder.encode(middle_texts)
    late_embeddings = encoder.encode(late_texts)

    rows: list[np.ndarray] = []
    for idx in range(len(users)):
        profile = profile_embeddings[idx]
        timeline = timeline_embeddings[idx]
        early = early_embeddings[idx]
        middle = middle_embeddings[idx]
        late = late_embeddings[idx]

        profile_timeline_cos = _safe_cosine_similarity(profile, timeline)
        early_late_drift = 1.0 - _safe_cosine_similarity(early, late)
        pairwise_distances = [
            1.0 - _safe_cosine_similarity(early, middle),
            1.0 - _safe_cosine_similarity(middle, late),
            1.0 - _safe_cosine_similarity(early, late),
        ]
        segment_dispersion = float(np.mean(pairwise_distances))

        norms = np.array(
            [
                np.linalg.norm(early),
                np.linalg.norm(middle),
                np.linalg.norm(late),
                np.linalg.norm(timeline),
            ],
            dtype=float,
        )

        rows.append(
            np.array(
                [
                    float(profile_timeline_cos),
                    float(early_late_drift),
                    float(segment_dispersion),
                    float(np.mean(norms)),
                    float(np.std(norms)),
                ],
                dtype=float,
            )
        )

    if not rows:
        return np.empty((0, len(EMBEDDING_FEATURE_NAMES)), dtype=float)

    return np.vstack(rows)
