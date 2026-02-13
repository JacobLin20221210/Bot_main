"""Holdout evaluation and cross-validation logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import evaluate_predictions
from src.features.matrix import FEATURE_NAMES
from src.models.calibration import HardNegativeCalibrator
from src.models.components.factory import build_component
from src.models.ensemble import BotEnsembleModel
from src.models.neural import NeuralSequenceModel
from src.training.candidates import select_best_candidate_from_training_rows
from src.training.oof import compute_oof_probabilities_component
from src.training.selection import (
    select_soft_cascade_with_threshold_robust,
    select_threshold_with_margin_grid_robust,
)
from src.utils.config import BEST_LANGUAGE_CONFIGS, LANGUAGE_STRICT_HOLDOUTS
from src.utils.io import save_json, save_pickle
from src.utils.logger import get_logger


def _build_seed_grid(base_seed: int, count: int, stride: int) -> list[int]:
    """Build a grid of seeds."""
    usable_count = max(1, int(count))
    usable_stride = max(1, int(stride))
    return [int(base_seed + (idx * usable_stride)) for idx in range(usable_count)]


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


def summarize_fold_report(report: list[dict[str, object]]) -> dict[str, float]:
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
        "mean_competition_score": float(
            np.mean([float(fold["competition_score"]) for fold in report])
        ),
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


def leave_one_dataset_out_report(
    datasets: list[dict[str, object]],
    folds: int,
    seed: int,
    min_precision: float,
    threshold_step: float,
    margin: float,
    language: str,
    persist_dir: Path | None = None,
) -> list[dict[str, object]]:
    """Generate leave-one-dataset-out cross-validation report."""
    logger = get_logger()
    report: list[dict[str, object]] = []

    for idx, test_dataset in enumerate(datasets):
        train_parts = [dataset for j, dataset in enumerate(datasets) if j != idx]
        if not train_parts:
            continue

        train_x = np.vstack([part["features"] for part in train_parts])
        train_docs = [doc for part in train_parts for doc in part["documents"]]
        train_y = np.concatenate([part["labels"] for part in train_parts])
        test_x = test_dataset["features"]
        test_docs = test_dataset["documents"]
        test_y = test_dataset["labels"]

        inner_seed = seed + (idx * 1000)
        candidate_rankings = select_best_candidate_from_training_rows(
            train_x, train_docs, train_y, folds=folds, seed=inner_seed,
            min_precision=min_precision, threshold_step=threshold_step
        )[1]
        selected_candidate = candidate_rankings[0]
        selected_tabular_params = selected_candidate["tabular"]["params"]
        selected_neural_params = selected_candidate["neural"]["params"]
        selected_threshold = float(selected_candidate["threshold"])
        selected_blend_weight = float(selected_candidate["blend_weight"])

        tabular_model = BotEnsembleModel(
            seed=seed + idx, threshold=selected_threshold, margin=margin,
            **selected_tabular_params
        )
        tabular_model.fit(train_x, train_y)

        neural_model = NeuralSequenceModel(seed=seed + idx, **selected_neural_params)
        neural_model.fit(train_x, train_docs, train_y)

        tabular_test_proba = tabular_model.predict_proba(test_x)
        neural_test_proba = neural_model.predict_proba(test_x, test_docs)
        blended_test_proba = (
            selected_blend_weight * tabular_test_proba
            + (1.0 - selected_blend_weight) * neural_test_proba
        )

        predictions = (blended_test_proba >= (selected_threshold + margin)).astype(int)
        metrics = evaluate_predictions(test_y, predictions)
        metrics["threshold"] = selected_threshold
        metrics["blend_weight"] = selected_blend_weight
        train_dataset_ids = sorted(str(part["dataset_id"]) for part in train_parts)
        metrics["train_dataset_ids"] = ",".join(train_dataset_ids)
        metrics["heldout_dataset_id"] = str(test_dataset["dataset_id"])
        metrics["selected_candidate_name"] = str(selected_candidate["name"])

        if persist_dir is not None:
            test_dataset_id = str(test_dataset["dataset_id"])
            fold_dir = persist_dir / f"train_{'_'.join(train_dataset_ids)}__test_{test_dataset_id}"
            fold_model_path = fold_dir / "model.pkl"
            fold_metrics_path = fold_dir / "metrics.json"

            fold_artifact = {
                "language": language,
                "feature_names": FEATURE_NAMES,
                "model_name": str(selected_candidate["name"]),
                "tabular_model": tabular_model.pipeline,
                "neural_model": neural_model,
                "threshold": selected_threshold,
                "margin": float(margin),
            }
            save_pickle(fold_model_path, fold_artifact)
            save_json(fold_metrics_path, metrics)
            metrics["model_path"] = str(fold_model_path)
            metrics["metrics_path"] = str(fold_metrics_path)

        report.append(metrics)

    return report


def evaluate_best_config_holdouts(
    language: str,
    datasets: list[dict[str, object]],
    args: Any,
    resolved_components: list[dict[str, object]],
    shortlist_component: dict[str, object],
    persist_dir: Path | None = None,
) -> list[dict[str, object]]:
    """Evaluate holdout splits for best config mode."""
    from datetime import datetime, timezone
    from src.models.cascade import soft_cascade_blend_probabilities

    logger = get_logger()
    language_config = BEST_LANGUAGE_CONFIGS[language]
    selection = language_config.get("selection", {})
    min_precision_grid = [float(x) for x in selection.get("min_precision_grid", [args.min_precision])]
    margin_grid = [float(x) for x in selection.get("margin_grid", [args.margin])]
    threshold_step = float(selection.get("threshold_step", args.threshold_step))
    robust_seed_count = int(selection.get("robust_seed_count", 8))
    robust_seed_stride = int(selection.get("robust_seed_stride", 101))
    max_fp_rate = float(selection.get("max_fp_rate", -1.0))
    use_contrastive_calibration = bool(selection.get("use_contrastive_calibration", True))
    contrastive_cfg = selection.get("contrastive_calibration", {})
    contrastive_hard_fraction = float(contrastive_cfg.get("hard_fraction", 0.14))
    contrastive_hard_weight = float(contrastive_cfg.get("hard_weight", 3.25))
    contrastive_c = float(contrastive_cfg.get("c", 1.2))
    contrastive_min_samples = int(contrastive_cfg.get("min_samples", 24))

    dataset_map = {str(dataset["dataset_id"]): dataset for dataset in datasets}
    holdouts = LANGUAGE_STRICT_HOLDOUTS.get(language)
    if holdouts is None:
        raise ValueError(f"No strict holdout pairs for language: {language}")

    report: list[dict[str, object]] = []
    for fold_index, (train_dataset_id, test_dataset_id) in enumerate(holdouts):
        logger.info(f"[{language.upper()}] Fold {fold_index + 1}/{len(holdouts)}: Train on {train_dataset_id}, Test on {test_dataset_id}")

        train_part = dataset_map[train_dataset_id]
        test_part = dataset_map[test_dataset_id]
        train_x = train_part["features"]
        train_docs = train_part["documents"]
        train_y = train_part["labels"]
        test_x = test_part["features"]
        test_docs = test_part["documents"]
        test_y = test_part["labels"]

        logger.info(f"  Train samples: {len(train_y)} (bots: {int(train_y.sum())}, humans: {len(train_y) - int(train_y.sum())})")
        logger.info(f"  Test samples:  {len(test_y)} (bots: {int(test_y.sum())}, humans: {len(test_y) - int(test_y.sum())})")

        weights = [float(component["weight"]) for component in resolved_components]
        robust_seed_values = _build_seed_grid(
            base_seed=int(args.seed + (fold_index * 1000)),
            count=robust_seed_count,
            stride=robust_seed_stride,
        )

        blended_train_oof_by_seed: list[np.ndarray] = []
        stage1_train_oof_by_seed: list[np.ndarray] = []

        logger.info(f"  Computing OOF probabilities with {len(robust_seed_values)} seeds...")
        for robust_seed in robust_seed_values:
            stage1_seed = int(robust_seed + 791)
            stage1_train_oof_by_seed.append(
                compute_oof_probabilities_component(
                    train_x, train_docs, train_y, folds=args.cv_folds, seed=stage1_seed,
                    kind=str(shortlist_component["kind"]), params=dict(shortlist_component["params"]),
                )
            )

            seed_component_oof: list[np.ndarray] = []
            for component_index, component in enumerate(resolved_components):
                component_seed = int(robust_seed + (component_index * 37))
                seed_component_oof.append(
                    compute_oof_probabilities_component(
                        train_x, train_docs, train_y, folds=args.cv_folds, seed=component_seed,
                        kind=str(component["kind"]), params=dict(component["params"]),
                    )
                )
            blended_train_oof_by_seed.append(_blend_component_probabilities(seed_component_oof, weights))

        test_probabilities: list[np.ndarray] = []
        trained_components: list[dict[str, object]] = []

        stage1_seed = int(args.seed + (fold_index * 1000) + 791)
        fitted_shortlist_component = build_component(
            kind=str(shortlist_component["kind"]), seed=stage1_seed,
            params=dict(shortlist_component["params"]),
        )
        fitted_shortlist_component.fit(train_x, train_docs, train_y)
        stage1_test = fitted_shortlist_component.predict_proba(test_x, test_docs)

        for component_index, component in enumerate(resolved_components):
            component_seed = int(args.seed + (fold_index * 1000) + (component_index * 37))
            fitted_component = build_component(
                kind=str(component["kind"]), seed=component_seed,
                params=dict(component["params"]),
            )
            fitted_component.fit(train_x, train_docs, train_y)
            test_probabilities.append(fitted_component.predict_proba(test_x, test_docs))
            trained_components.append({
                "model": str(component["model"]),
                "kind": str(component["kind"]),
                "weight": float(component["weight"]),
                "params": dict(component["params"]),
                "model_object": fitted_component,
            })

        blended_test = _blend_component_probabilities(test_probabilities, weights)

        stage2_calibrator = HardNegativeCalibrator(
            hard_fraction=contrastive_hard_fraction,
            hard_weight=contrastive_hard_weight,
            c=contrastive_c,
            min_samples=contrastive_min_samples,
        )
        if use_contrastive_calibration:
            mean_stage2_train_oof = np.mean(np.vstack(blended_train_oof_by_seed), axis=0)
            stage2_calibrator.fit(mean_stage2_train_oof, train_y)
        calibrated_stage2_train_oof_by_seed = [
            stage2_calibrator.transform(probabilities)
            if use_contrastive_calibration
            else probabilities
            for probabilities in blended_train_oof_by_seed
        ]
        calibrated_stage2_test = stage2_calibrator.transform(blended_test) if use_contrastive_calibration else blended_test

        cascade_enabled = bool(shortlist_component.get("enabled", False))
        if cascade_enabled:
            cascade_selection = select_soft_cascade_with_threshold_robust(
                stage1_probabilities_by_seed=stage1_train_oof_by_seed,
                stage2_probabilities_by_seed=calibrated_stage2_train_oof_by_seed,
                labels=train_y,
                shortlist_threshold_grid=[float(x) for x in shortlist_component.get("shortlist_threshold_grid", [0.4])],
                stage2_weight_grid=[float(x) for x in shortlist_component.get("stage2_weight_grid", [0.85])],
                min_precision_grid=min_precision_grid,
                margin_grid=margin_grid,
                threshold_step=threshold_step,
                max_fp_rate=max_fp_rate,
            )
            selection_summary = dict(cascade_selection["selection"])
            selected_shortlist_threshold = float(cascade_selection["shortlist_threshold"])
            selected_stage2_weight = float(cascade_selection["stage2_weight"])
            selected_shortlist_coverage = float(cascade_selection["shortlist_coverage_mean"])
            blended_test_for_decision = soft_cascade_blend_probabilities(
                stage1_probabilities=stage1_test,
                stage2_probabilities=calibrated_stage2_test,
                shortlist_threshold=selected_shortlist_threshold,
                stage2_weight=selected_stage2_weight,
            )
        else:
            selection_summary = select_threshold_with_margin_grid_robust(
                calibrated_stage2_train_oof_by_seed, train_y,
                min_precision_grid=min_precision_grid,
                margin_grid=margin_grid,
                threshold_step=threshold_step,
                max_fp_rate=max_fp_rate,
            )
            selected_shortlist_threshold = 0.0
            selected_stage2_weight = 1.0
            selected_shortlist_coverage = 1.0
            blended_test_for_decision = calibrated_stage2_test

        selected_threshold = float(selection_summary["threshold"])
        selected_margin = float(selection_summary["margin"])

        logger.info(f"  Selected threshold: {selected_threshold:.4f}, margin: {selected_margin:.4f}")

        predictions = (blended_test_for_decision >= (selected_threshold + selected_margin)).astype(int)
        metrics = evaluate_predictions(test_y, predictions)

        logger.info(f"  Results: Score={metrics.get('competition_score', 0):.1f}, "
                    f"Prec={metrics.get('precision', 0):.3f}, Recall={metrics.get('recall', 0):.3f}, "
                    f"TP={int(metrics.get('tp', 0))}, FP={int(metrics.get('fp', 0))}, FN={int(metrics.get('fn', 0))}")
        logger.info("")
        metrics["threshold"] = selected_threshold
        metrics["margin"] = selected_margin
        metrics["shortlist_threshold"] = selected_shortlist_threshold
        metrics["cascade_stage2_weight"] = selected_stage2_weight
        metrics["shortlist_coverage_train_mean"] = selected_shortlist_coverage
        metrics["train_dataset_ids"] = train_dataset_id
        metrics["heldout_dataset_id"] = test_dataset_id
        metrics["selected_candidate_name"] = str(language_config["config_id"])
        metrics["cascade_enabled"] = bool(cascade_enabled)
        metrics["contrastive_calibration_enabled"] = bool(use_contrastive_calibration and stage2_calibrator.enabled)
        metrics["contrastive_hard_negative_center"] = float(stage2_calibrator.hard_negative_center)
        metrics["contrastive_hard_positive_center"] = float(stage2_calibrator.hard_positive_center)
        metrics["selected_components"] = [
            {"model": str(c["model"]), "kind": str(c["kind"]), "weight": float(c["weight"]), "params": dict(c["params"])}
            for c in resolved_components
        ]

        if persist_dir is not None:
            fold_name = f"train_{train_dataset_id}__test_{test_dataset_id}"
            fold_dir = persist_dir / fold_name
            fold_model_path = fold_dir / "model.pkl"
            fold_metrics_path = fold_dir / "metrics.json"

            fold_artifact = {
                "language": language,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "feature_names": FEATURE_NAMES,
                "model_name": str(language_config["config_id"]),
                "model_mode": "component_blend_cascade_soft" if cascade_enabled else "component_blend",
                "cascade": {
                    "enabled": bool(cascade_enabled),
                    "shortlist_threshold": selected_shortlist_threshold,
                    "stage2_weight": selected_stage2_weight,
                    "shortlist_model": str(shortlist_component["model"]),
                    "shortlist_model_object": fitted_shortlist_component,
                },
                "contrastive_calibration": {
                    "enabled": bool(use_contrastive_calibration and stage2_calibrator.enabled),
                    "calibrator_object": stage2_calibrator,
                },
                "components": trained_components,
                "threshold": selected_threshold,
                "margin": selected_margin,
            }
            save_pickle(fold_model_path, fold_artifact)
            save_json(fold_metrics_path, {k: v for k, v in metrics.items() if k not in {"model_path", "metrics_path"}})
            metrics["model_path"] = str(fold_model_path)
            metrics["metrics_path"] = str(fold_metrics_path)

        report.append(metrics)

    return report
