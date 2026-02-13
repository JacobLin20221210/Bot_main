"""Main training pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.models.components.factory import build_component
from src.models.ensemble import BotEnsembleModel
from src.models.neural import NeuralSequenceModel
from src.training.config import BLEND_WEIGHTS, NEURAL_CANDIDATES, TABULAR_CANDIDATES
from src.training.data import load_training_rows
from src.training.identity import build_run_identity
from src.training.oof import (
    compute_oof_probabilities_component,
    compute_oof_probabilities_neural,
    compute_oof_probabilities_tabular,
)
from src.training.selection import (
    select_blend_and_threshold,
    select_threshold_with_margin_grid,
)
from src.utils.config import BEST_LANGUAGE_CONFIGS, LANGUAGE_STRICT_HOLDOUTS, TRANSFER_MODEL_LIBRARY
from src.utils.io import append_jsonl, save_json, save_pickle
from src.utils.logger import get_git_commit_hash, get_logger


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
            "pooled_competition_score": float("-inf"),
        }

    pooled_tp = sum(int(round(float(fold["tp"]))) for fold in report)
    pooled_fp = sum(int(round(float(fold["fp"]))) for fold in report)
    pooled_fn = sum(int(round(float(fold["fn"]))) for fold in report)

    return {
        "mean_competition_score": float(
            np.mean([float(fold["competition_score"]) for fold in report])
        ),
        "mean_accuracy": float(np.mean([float(fold["accuracy"]) for fold in report])),
        "mean_precision": float(np.mean([float(fold["precision"]) for fold in report])),
        "mean_recall": float(np.mean([float(fold["recall"]) for fold in report])),
        "mean_f1": float(np.mean([float(fold["f1"]) for fold in report])),
        "pooled_tp": float(pooled_tp),
        "pooled_fp": float(pooled_fp),
        "pooled_fn": float(pooled_fn),
        "pooled_competition_score": float((4 * pooled_tp) - pooled_fn - (2 * pooled_fp)),
    }


def run_training_pipeline(args: Any) -> None:
    """Run the complete training pipeline."""
    logger = get_logger()
    run_id, digest, signature_payload, created_at_utc = build_run_identity(args)

    logger.info("Run ID: %s", run_id)
    logger.info("Git commit: %s", get_git_commit_hash())

    # Create run directory
    run_dir = Path(args.archive_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    logger.info("Loading training data from %s", args.dataset_dir)
    training_rows = load_training_rows(args.dataset_dir)

    # Group by language
    by_language: dict[str, list[dict[str, object]]] = {}
    for row in training_rows:
        lang = str(row["language"])
        by_language.setdefault(lang, []).append(row)

    logger.info("Found languages: %s", list(by_language.keys()))

    # Train models for each language
    for language, datasets in by_language.items():
        logger.info("Training %s model with %d datasets", language, len(datasets))
        _train_language_model(language, datasets, args, run_dir, run_id, created_at_utc)

    logger.info("Training complete. Run ID: %s", run_id)


def _train_language_model(
    language: str,
    datasets: list[dict[str, object]],
    args: Any,
    run_dir: Path,
    run_id: str,
    created_at_utc: str,
) -> None:
    """Train a model for a specific language."""
    logger = get_logger()

    if args.training_mode == "best_config":
        _train_best_config(language, datasets, args, run_dir)
    else:
        _train_search_mode(language, datasets, args, run_dir)


def _train_best_config(
    language: str,
    datasets: list[dict[str, object]],
    args: Any,
    run_dir: Path,
) -> None:
    """Train using best config mode."""
    logger = get_logger()
    logger.info("Using best config for %s", language)

    language_config = BEST_LANGUAGE_CONFIGS.get(language)
    if language_config is None:
        raise ValueError(f"No best config found for language: {language}")

    # Resolve components
    resolved_components = _resolve_best_components(language)
    logger.info("Resolved %d components", len(resolved_components))

    # Combine all data
    all_features = np.vstack([d["features"] for d in datasets])
    all_documents = [doc for d in datasets for doc in d["documents"]]
    all_labels = np.concatenate([d["labels"] for d in datasets])

    # Train final models
    trained_components = []
    for component in resolved_components:
        model = build_component(
            kind=str(component["kind"]),
            seed=args.seed,
            params=dict(component["params"]),
        )
        model.fit(all_features, all_documents, all_labels)
        trained_components.append(
            {
                "model": component["model"],
                "kind": component["kind"],
                "weight": component["weight"],
                "params": component["params"],
                "model_object": model,
            }
        )

    # Compute OOF probabilities for threshold selection
    weights = [c["weight"] for c in resolved_components]
    oof_probs = compute_oof_probabilities_component(
        all_features,
        all_documents,
        all_labels,
        folds=args.cv_folds,
        seed=args.seed,
        kind=str(resolved_components[0]["kind"]),
        params=dict(resolved_components[0]["params"]),
    )

    # Select threshold
    threshold_info = select_threshold_with_margin_grid(
        oof_probs,
        all_labels,
        min_precision_grid=[args.min_precision],
        margin_grid=[args.margin],
        threshold_step=args.threshold_step,
    )

    logger.info(
        "Selected threshold: %.3f, score: %.1f",
        threshold_info["threshold"],
        threshold_info["score"],
    )

    # Save model
    model_dir = run_dir / "models" / language
    model_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "language": language,
        "components": trained_components,
        "threshold": threshold_info["threshold"],
        "margin": args.margin,
        "feature_names": [],  # Will be populated from features module
    }

    save_pickle(model_dir / "model.pkl", artifact)
    save_json(model_dir / "metrics.json", threshold_info)

    logger.info("Saved model to %s", model_dir)


def _train_search_mode(
    language: str,
    datasets: list[dict[str, object]],
    args: Any,
    run_dir: Path,
) -> None:
    """Train using search mode (legacy candidate search)."""
    logger = get_logger()
    logger.info("Using search mode for %s", language)

    # Combine all data
    all_features = np.vstack([d["features"] for d in datasets])
    all_documents = [doc for d in datasets for doc in d["documents"]]
    all_labels = np.concatenate([d["labels"] for d in datasets])

    best_candidate = None
    best_score = float("-inf")

    # Search through tabular candidates
    for tabular_candidate in TABULAR_CANDIDATES:
        # Compute OOF probabilities
        oof_probs = compute_oof_probabilities_tabular(
            all_features,
            all_labels,
            folds=args.cv_folds,
            seed=args.seed,
            model_params=tabular_candidate["params"],
        )

        # Select threshold
        threshold_info = select_threshold_with_margin_grid(
            oof_probs,
            all_labels,
            min_precision_grid=[args.min_precision],
            margin_grid=[args.margin],
            threshold_step=args.threshold_step,
        )

        if threshold_info["score"] > best_score:
            best_score = threshold_info["score"]
            best_candidate = {
                "tabular": tabular_candidate,
                "threshold": threshold_info["threshold"],
                "score": threshold_info["score"],
            }

    logger.info(
        "Best candidate: %s, score: %.1f",
        best_candidate["tabular"]["name"] if best_candidate else "none",
        best_score,
    )

    # Train final model with best candidate
    if best_candidate:
        final_model = BotEnsembleModel(
            seed=args.seed,
            **best_candidate["tabular"]["params"],
        )
        final_model.fit(all_features, all_labels)

        # Save model
        model_dir = run_dir / "models" / language
        model_dir.mkdir(parents=True, exist_ok=True)

        artifact = {
            "language": language,
            "model": final_model,
            "threshold": best_candidate["threshold"],
            "margin": args.margin,
        }

        save_pickle(model_dir / "model.pkl", artifact)
        logger.info("Saved model to %s", model_dir)
