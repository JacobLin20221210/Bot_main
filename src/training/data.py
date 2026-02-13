"""Training data loading and preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data.loader import (
    discover_training_pairs,
    group_posts_by_author,
    load_dataset_bundle,
)
from src.features.matrix import build_feature_matrix, build_sequence_documents


def load_training_rows(
    dataset_dir: str | Path,
    feature_embedding_model_name: str = "intfloat/multilingual-e5-small",
) -> list[dict[str, object]]:
    """Load training data rows from all datasets in directory."""
    rows: list[dict[str, object]] = []

    for dataset_path, bots_path in discover_training_pairs(dataset_dir):
        bundle = load_dataset_bundle(dataset_path, bots_path)
        posts_by_author = group_posts_by_author(bundle["posts"])
        user_ids, features = build_feature_matrix(
            bundle["users"],
            posts_by_author,
            embedding_model_name=feature_embedding_model_name,
        )
        doc_user_ids, documents = build_sequence_documents(bundle["users"], posts_by_author)

        if user_ids != doc_user_ids:
            raise ValueError(
                f"Feature/document user order mismatch in dataset {bundle['dataset_id']}"
            )

        labels = np.array(
            [1 if user_id in bundle["bot_ids"] else 0 for user_id in user_ids],
            dtype=int,
        )

        rows.append(
            {
                "dataset_id": str(bundle["dataset_id"]),
                "language": str(bundle["language"]),
                "user_ids": user_ids,
                "features": features,
                "documents": documents,
                "labels": labels,
            }
        )

    return rows
