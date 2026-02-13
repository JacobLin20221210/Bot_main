"""Core training functionality."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.utils.config import BEST_LANGUAGE_CONFIGS, LANGUAGE_STRICT_HOLDOUTS, TRANSFER_MODEL_LIBRARY
from src.data.loader import discover_training_pairs, group_posts_by_author, load_dataset_bundle
from src.evaluation.metrics import evaluate_predictions
from src.features.matrix import FEATURE_NAMES, build_feature_matrix, build_sequence_documents
from src.models.components.factory import build_component
from src.models.ensemble import BotEnsembleModel
from src.models.neural import NeuralSequenceModel
from src.training.oof import (
    compute_oof_probabilities_component,
    compute_oof_probabilities_neural,
    compute_oof_probabilities_tabular,
)
from src.training.selection import (
    select_blend_and_threshold,
    select_soft_cascade_with_threshold_robust,
    select_threshold_with_margin_grid_robust,
)
from src.utils.io import append_jsonl, save_json, save_pickle
from src.utils.logger import get_git_commit_hash, get_logger

# Legacy candidate configurations
TABULAR_CANDIDATES: list[dict[str, object]] = [
    {
        "name": "balanced_default",
        "params": {
            "rf_estimators": 900,
            "et_estimators": 1200,
            "min_samples_leaf": 2,
            "rf_bot_weight": 1.2,
            "et_bot_weight": 1.3,
            "calibration_cv": 3,
        },
    },
    {
        "name": "precision_tilted",
        "params": {
            "rf_estimators": 1000,
            "et_estimators": 1400,
            "min_samples_leaf": 3,
            "rf_bot_weight": 1.15,
            "et_bot_weight": 1.2,
            "calibration_cv": 3,
        },
    },
    {
        "name": "recall_tilted",
        "params": {
            "rf_estimators": 900,
            "et_estimators": 1200,
            "min_samples_leaf": 2,
            "rf_bot_weight": 1.35,
            "et_bot_weight": 1.45,
            "calibration_cv": 3,
        },
    },
]

NEURAL_CANDIDATES: list[dict[str, object]] = [
    {
        "name": "neural_wide",
        "params": {
            "hidden_layer_sizes": (256, 128),
            "alpha": 1e-4,
            "max_iter": 350,
            "text_svd_components": 128,
        },
    },
    {
        "name": "neural_compact",
        "params": {
            "hidden_layer_sizes": (192, 96),
            "alpha": 2e-4,
            "max_iter": 320,
            "text_svd_components": 96,
        },
    },
]

BLEND_WEIGHTS = [0.65, 0.75, 0.85]


def _sanitize_run_name(name: str | None) -> str:
    """Sanitize run name for filesystem safety."""
    if not name:
        return ""
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip())
    return normalized.strip("-._")[:80]


def _build_run_identity(args: Any) -> tuple[str, str, dict[str, object], str]:
    """Build run identity with signature."""
    timestamp = datetime.now(timezone.utc)
    timestamp_str = timestamp.strftime("%Y%m%d-%H%M%S")

    signature_payload: dict[str, object] = {
        "dataset_dir": str(args.dataset_dir),
        "cv_folds": int(args.cv_folds),
        "min_precision": float(args.min_precision),
        "threshold_step": float(args.threshold_step),
        "margin": float(args.margin),
        "seed": int(args.seed),
        "training_mode": str(args.training_mode),
        "tabular_candidates": TABULAR_CANDIDATES,
        "neural_candidates": NEURAL_CANDIDATES,
        "blend_weights": BLEND_WEIGHTS,
        "best_language_configs": BEST_LANGUAGE_CONFIGS,
    }

    digest = hashlib.sha1(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]

    name_suffix = _sanitize_run_name(args.run_name)
    run_id = f"{timestamp_str}-{digest}" if not name_suffix else f"{timestamp_str}-{digest}-{name_suffix}"

    return run_id, digest, signature_payload, timestamp.isoformat()


def _load_training_rows(dataset_dir: str | Path) -> list[dict[str, object]]:
    """Load training data rows from all datasets."""
    rows: list[dict[str, object]] = []

    for dataset_path, bots_path in discover_training_pairs(dataset_dir):
        bundle = load_dataset_bundle(dataset_path, bots_path)
        posts_by_author = group_posts_by_author(bundle["posts"])
        user_ids, features = build_feature_matrix(bundle["users"], posts_by_author)
        doc_user_ids, documents = build_sequence_documents(bundle["users"], posts_by_author)

        if user_ids != doc_user_ids:
            raise ValueError(f"Feature/document user order mismatch in dataset {bundle['dataset_id']}")

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


def _build_seed_grid(base_seed: int, count: int, stride: int) -> list[int]:
    """Build a grid of seeds."""
    usable_count = max(1, int(count))
    usable_stride = max(1, int(stride))
    return [int(base_seed + (idx * usable_stride)) for idx in range(usable_count)]


def _resolve_model_from_library(model_name: str) -> dict[str, object]:
    """Resolve a model specification from the library."""
    model_spec = TRANSFER_MODEL_LIBRARY.get(model_name)
    if model_spec is None:
        raise ValueError(f"Unknown model in config: {model_name}")
    return {
        "model": model_name,
        "kind": str(model_spec["kind"]),
        "params": dict(model_spec["params"]),
    }


def _resolve_best_components(language: str) -> list[dict[str, object]]:
    """Resolve best components for a language."""
    language_config = BEST_LANGUAGE_CONFIGS.get(language)
    if language_config is None:
        raise ValueError(f"No best config found for language: {language}")

    resolved: list[dict[str, object]] = []
    for component in language_config["components"]:
        model_name = str(component["model"])
        resolved_model = _resolve_model_from_library(model_name)
        resolved.append(
            {
                "model": resolved_model["model"],
                "weight": float(component["weight"]),
                "kind": resolved_model["kind"],
                "params": resolved_model["params"],
            }
        )
    return resolved


def _resolve_cascade_shortlist_component(
    language: str, resolved_components: list[dict[str, object]]
) -> dict[str, object]:
    """Resolve cascade shortlist component configuration."""
    language_config = BEST_LANGUAGE_CONFIGS.get(language, {})
    cascade_cfg = dict(language_config.get("cascade", {}))
    shortlist_model_name = str(
        cascade_cfg.get("shortlist_model", resolved_components[0]["model"])
    )
    shortlist_component = _resolve_model_from_library(shortlist_model_name)

    return {
        "enabled": bool(cascade_cfg.get("enabled", True)),
        "model": shortlist_component["model"],
        "kind": shortlist_component["kind"],
        "params": shortlist_component["params"],
        "shortlist_threshold_grid": [
            float(x) for x in cascade_cfg.get("shortlist_threshold_grid", [0.25, 0.35, 0.45, 0.55, 0.65])
        ],
        "stage2_weight_grid": [
            float(x) for x in cascade_cfg.get("stage2_weight_grid", [0.7, 0.85, 0.95])
        ],
    }


def _blend_component_probabilities(
    probabilities: list[np.ndarray], weights: list[float]
) -> np.ndarray:
    """Blend component probabilities with weights."""
    if not probabilities:
        raise ValueError("No component probabilities provided")
    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("Total component weight must be positive")

    blended = np.zeros_like(probabilities[0], dtype=float)
    for probs, weight in zip(probabilities, weights):
        blended += float(weight) * probs
    return blended / total_weight


def _summarize_fold_report(report: list[dict[str, object]]) -> dict[str, float]:
    """Summarize fold report metrics."""
    if not report:
        return {
            "mean_competition_score": float("-inf"),
            "mean_accuracy": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_f1": 0.0,
            "mean_fp": 0.0,
            "mean_fn": 0.0,
            "pooled_tp": 0.0,
            "pooled_tn": 0.0,
            "pooled_fp": 0.0,
            "pooled_fn": 0.0,
            "pooled_accuracy": 0.0,
            "pooled_precision": 0.0,
            "pooled_recall": 0.0,
            "pooled_f1": 0.0,
            "pooled_competition_score": float("-inf"),
        }

    pooled_tp = int(sum(int(round(float(fold["tp"]))) for fold in report))
    pooled_tn = int(sum(int(round(float(fold["tn"]))) for fold in report))
    pooled_fp = int(sum(int(round(float(fold["fp"]))) for fold in report))
    pooled_fn = int(sum(int(round(float(fold["fn"]))) for fold in report))
    pooled_total = max(1, pooled_tp + pooled_tn + pooled_fp + pooled_fn)
    pooled_accuracy = (pooled_tp + pooled_tn) / pooled_total
    pooled_precision = pooled_tp / max(1, pooled_tp + pooled_fp)
    pooled_recall = pooled_tp / max(1, pooled_tp + pooled_fn)
    pooled_f1 = (
        0.0
        if (pooled_precision + pooled_recall) == 0
        else 2 * pooled_precision * pooled_recall / (pooled_precision + pooled_recall)
    )

    return {
        "mean_competition_score": float(np.mean([float(fold["competition_score"]) for fold in report])),
        "mean_accuracy": float(np.mean([float(fold["accuracy"]) for fold in report])),
        "mean_precision": float(np.mean([float(fold["precision"]) for fold in report])),
        "mean_recall": float(np.mean([float(fold["recall"]) for fold in report])),
        "mean_f1": float(np.mean([float(fold["f1"]) for fold in report])),
        "mean_fp": float(np.mean([float(fold["fp"]) for fold in report])),
        "mean_fn": float(np.mean([float(fold["fn"]) for fold in report])),
        "pooled_tp": float(pooled_tp),
        "pooled_tn": float(pooled_tn),
        "pooled_fp": float(pooled_fp),
        "pooled_fn": float(pooled_fn),
        "pooled_accuracy": float(pooled_accuracy),
        "pooled_precision": float(pooled_precision),
        "pooled_recall": float(pooled_recall),
        "pooled_f1": float(pooled_f1),
        "pooled_competition_score": float((4 * pooled_tp) - pooled_fn - (2 * pooled_fp)),
    }
