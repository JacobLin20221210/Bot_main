"""Main training orchestration module."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.features.matrix import FEATURE_NAMES
from src.models.calibration import HardNegativeCalibrator, cross_fit_hard_negative_calibration
from src.models.components.factory import build_component
from src.models.ensemble import BotEnsembleModel
from src.models.neural import NeuralSequenceModel
from src.training.candidates import select_best_candidate_from_training_rows
from src.training.config import BLEND_WEIGHTS, NEURAL_CANDIDATES, TABULAR_CANDIDATES
from src.training.data import load_training_rows
from src.training.holdout import (
    evaluate_best_config_holdouts,
    leave_one_dataset_out_report,
    summarize_fold_report,
)
from src.training.identity import build_run_identity
from src.training.oof import (
    compute_oof_probabilities_component,
    compute_oof_probabilities_neural,
    compute_oof_probabilities_tabular,
)
from src.training.resolution import (
    resolve_best_components,
    resolve_cascade_shortlist_component,
)
from src.training.selection import (
    select_blend_and_threshold,
    select_soft_cascade_with_threshold_robust,
    select_threshold_with_margin_grid_robust,
)
from src.utils.config import BEST_LANGUAGE_CONFIGS
from src.utils.io import append_jsonl, save_json, save_pickle
from src.utils.logger import get_git_commit_hash, get_logger


class RandomSeedSequence:
    """Generate deterministic random seed sequences."""
    def __init__(self, origin_seed: int, count: int, step: int):
        self.origin = max(1, int(origin_seed))
        self.quantity = max(1, int(count))
        self.step_size = max(1, int(step))
    
    def generate(self) -> list[int]:
        """Generate seed sequence."""
        return [int(self.origin + (i * self.step_size)) for i in range(self.quantity)]


class ProbabilityEnsembleAggregator:
    """Aggregate multiple probability predictions using weighted averaging."""
    
    def __init__(self, component_weights: list[float]):
        if not component_weights:
            raise ValueError("Component weights required")
        total = float(sum(component_weights))
        if total <= 0:
            raise ValueError("Weights must sum to positive")
        self.weights = component_weights
        self.total_weight = total
    
    def combine(self, prob_arrays: list[np.ndarray]) -> np.ndarray:
        """Combine probability arrays."""
        result = np.zeros_like(prob_arrays[0], dtype=float)
        for arr, w in zip(prob_arrays, self.weights):
            result += float(w) * arr
        return result / self.total_weight


def _deep_merge_dict_in_place(target: dict[str, Any], updates: dict[str, Any]) -> None:
    """Deep-merge updates into target dictionary."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge_dict_in_place(target[key], value)
            continue
        target[key] = value


def _apply_best_config_overrides(config_path: str, logger: Any) -> None:
    """Apply per-language best-config overrides from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Best-config override file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Best-config override file must contain a JSON object")

    raw_updates = payload.get("best_language_configs", payload)
    if not isinstance(raw_updates, dict):
        raise ValueError("best_language_configs must be a JSON object when provided")

    updated_languages: list[str] = []
    for language, language_updates in raw_updates.items():
        if language not in BEST_LANGUAGE_CONFIGS:
            raise ValueError(f"Unknown language in best-config overrides: {language}")
        if not isinstance(language_updates, dict):
            raise ValueError(f"Override for language '{language}' must be a JSON object")
        _deep_merge_dict_in_place(BEST_LANGUAGE_CONFIGS[language], language_updates)
        updated_languages.append(language)

    logger.info(
        "Applied best-config overrides from %s for languages: %s",
        str(path),
        ", ".join(sorted(updated_languages)) if updated_languages else "none",
    )


def train_language_model_best_config(
    language: str,
    datasets: list[dict[str, object]],
    args: Any,
    archive_language_dir: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Train model using best configuration mode."""
    from src.models.cascade import soft_cascade_blend_probabilities

    logger = get_logger()
    logger.info("=" * 70)
    logger.info(f"Training language model: {language.upper()}")
    logger.info("=" * 70)

    language_config = BEST_LANGUAGE_CONFIGS.get(language)
    if language_config is None:
        raise ValueError(f"No best config found for language: {language}")

    logger.info(f"Config ID: {language_config.get('config_id', 'unknown')}")
    logger.info(f"Selection protocol: {language_config.get('selection_protocol', 'unknown')}")
    logger.info(f"Components: {[c.get('model', '?') for c in language_config.get('components', [])]}")
    logger.info("")

    selection = language_config.get("selection", {})
    min_precision_grid = [float(x) for x in selection.get("min_precision_grid", [args.min_precision])]
    margin_grid = [float(x) for x in selection.get("margin_grid", [args.margin])]
    threshold_step = float(selection.get("threshold_step", args.threshold_step))
    robust_seed_count = int(selection.get("robust_seed_count", 8))
    robust_seed_stride = int(selection.get("robust_seed_stride", 101))
    max_fp_rate = float(selection.get("max_fp_rate", -1.0))
    use_contrastive_calibration = bool(selection.get("use_contrastive_calibration", True))
    calibration_folds = int(selection.get("calibration_folds", args.cv_folds))
    calibration_seed_stride = int(selection.get("calibration_seed_stride", 1009))
    contrastive_cfg = selection.get("contrastive_calibration", {})
    contrastive_hard_fraction = float(contrastive_cfg.get("hard_fraction", 0.14))
    contrastive_hard_weight = float(contrastive_cfg.get("hard_weight", 3.25))
    contrastive_c = float(contrastive_cfg.get("c", 1.2))
    contrastive_min_samples = int(contrastive_cfg.get("min_samples", 24))

    resolved_components = resolve_best_components(language)
    shortlist_component = resolve_cascade_shortlist_component(language, resolved_components)

    all_features = np.vstack([dataset["features"] for dataset in datasets])
    all_documents = [doc for dataset in datasets for doc in dataset["documents"]]
    all_labels = np.concatenate([dataset["labels"] for dataset in datasets])
    all_dataset_ids = [str(dataset["dataset_id"]) for dataset in datasets]

    # Evaluate holdouts
    logger.info("Evaluating holdout splits...")
    strict_outer_report = evaluate_best_config_holdouts(
        language=language,
        datasets=datasets,
        args=args,
        resolved_components=resolved_components,
        shortlist_component=shortlist_component,
        persist_dir=(archive_language_dir / "holdout") if archive_language_dir else None,
    )
    strict_outer_summary = summarize_fold_report(strict_outer_report)

    pooled_score = strict_outer_summary.get("pooled_competition_score", 0)
    mean_score = strict_outer_summary.get("mean_competition_score", 0)
    logger.info(f"Holdout summary: Pooled score={pooled_score:.1f}, Mean score={mean_score:.1f}")
    logger.info("")

    # Compute OOF probabilities
    component_weights = [float(component["weight"]) for component in resolved_components]
    seed_gen = RandomSeedSequence(int(args.seed), robust_seed_count, robust_seed_stride)
    seed_sequence = seed_gen.generate()
    aggregator = ProbabilityEnsembleAggregator(component_weights)

    full_blended_oof_by_seed: list[np.ndarray] = []
    stage1_oof_by_seed: list[np.ndarray] = []

    for robust_seed in seed_sequence:
        stage1_seed = int(robust_seed + 791)
        stage1_oof_by_seed.append(
            compute_oof_probabilities_component(
                all_features, all_documents, all_labels,
                folds=args.cv_folds, seed=stage1_seed,
                kind=str(shortlist_component["kind"]), params=dict(shortlist_component["params"]),
            )
        )

        component_predictions: list[np.ndarray] = []
        for component_index, component in enumerate(resolved_components):
            component_seed = int(robust_seed + (component_index * 97))
            component_predictions.append(
                compute_oof_probabilities_component(
                    all_features, all_documents, all_labels,
                    folds=args.cv_folds, seed=component_seed,
                    kind=str(component["kind"]), params=dict(component["params"]),
                )
            )
        full_blended_oof_by_seed.append(aggregator.combine(component_predictions))

    # Phase: Hard negative calibration
    stage2_calibrator = HardNegativeCalibrator(
        hard_fraction=contrastive_hard_fraction,
        hard_weight=contrastive_hard_weight,
        c=contrastive_c,
        min_samples=contrastive_min_samples,
    )
    calibration_applied_seed_count = 0
    
    if use_contrastive_calibration:
        calibrated_stage2_oof_by_seed = []
        for robust_seed, prob_vector in zip(seed_sequence, full_blended_oof_by_seed, strict=False):
            recalibrated_probs, calibration_applied = cross_fit_hard_negative_calibration(
                probabilities=prob_vector,
                labels=all_labels,
                folds=calibration_folds,
                seed=int(robust_seed + calibration_seed_stride),
                hard_fraction=contrastive_hard_fraction,
                hard_weight=contrastive_hard_weight,
                c=contrastive_c,
                min_samples=contrastive_min_samples,
            )
            if calibration_applied:
                calibration_applied_seed_count += 1
            calibrated_stage2_oof_by_seed.append(recalibrated_probs)

        ensemble_oof_mean = np.mean(np.vstack(full_blended_oof_by_seed), axis=0)
        stage2_calibrator.fit(ensemble_oof_mean, all_labels)
    else:
        calibrated_stage2_oof_by_seed = full_blended_oof_by_seed

    # Phase: Threshold and cascade selection
    cascade_strategy_enabled = bool(shortlist_component.get("enabled", False))
    
    if cascade_strategy_enabled:
        cascade_decision = select_soft_cascade_with_threshold_robust(
            stage1_probabilities_by_seed=stage1_oof_by_seed,
            stage2_probabilities_by_seed=calibrated_stage2_oof_by_seed,
            labels=all_labels,
            shortlist_threshold_grid=[float(x) for x in shortlist_component.get("shortlist_threshold_grid", [0.4])],
            stage2_weight_grid=[float(x) for x in shortlist_component.get("stage2_weight_grid", [0.85])],
            min_precision_grid=min_precision_grid,
            margin_grid=margin_grid,
            threshold_step=threshold_step,
            max_fp_rate=max_fp_rate,
        )
        threshold_selection = dict(cascade_decision["selection"])
        optimal_shortlist_threshold = float(cascade_decision["shortlist_threshold"])
        optimal_stage2_weight = float(cascade_decision["stage2_weight"])
        optimal_shortlist_coverage = float(cascade_decision["shortlist_coverage_mean"])
    else:
        cascade_decision = {"shortlist_threshold": 0.0, "stage2_weight": 1.0, "shortlist_coverage_mean": 1.0, "top_candidates": []}
        threshold_selection = select_threshold_with_margin_grid_robust(
            calibrated_stage2_oof_by_seed, all_labels,
            min_precision_grid=min_precision_grid, margin_grid=margin_grid,
            threshold_step=threshold_step, max_fp_rate=max_fp_rate,
        )
        optimal_shortlist_threshold = 0.0
        optimal_stage2_weight = 1.0
        optimal_shortlist_coverage = 1.0

    decision_threshold = float(threshold_selection["threshold"])
    decision_margin = float(threshold_selection["margin"])

    # Phase: Fit final models on full dataset
    logger.info("Fitting final ensemble models...")
    shortlist_fit_seed = int(args.seed + 791)
    shortlist_model_fitted = build_component(
        kind=str(shortlist_component["kind"]), seed=shortlist_fit_seed,
        params=dict(shortlist_component["params"]),
    )
    shortlist_model_fitted.fit(all_features, all_documents, all_labels)
    logger.info(f"  Shortlist fitted: {shortlist_component.get('model', '?')}")

    live_trained_components: list[dict[str, object]] = []
    for comp_idx, comp_config in enumerate(resolved_components):
        comp_fit_seed = int(args.seed + (comp_idx * 97))
        comp_fitted = build_component(
            kind=str(comp_config["kind"]), seed=comp_fit_seed, params=dict(comp_config["params"]),
        )
        comp_fitted.fit(all_features, all_documents, all_labels)
        live_trained_components.append({
            "model": str(comp_config["model"]),
            "kind": str(comp_config["kind"]),
            "weight": float(comp_config["weight"]),
            "params": dict(comp_config["params"]),
            "model_object": comp_fitted,
        })
        logger.info(f"  Fitted: {comp_config.get('model', '?')} (w: {comp_config.get('weight', 0)})")

    logger.info(f"  Decision boundary: {decision_threshold:.4f}, safety margin: {decision_margin:.4f}")
    logger.info("")
    logger.info(f"{language.upper()} pipeline complete. Holdout evaluation: {pooled_score:.1f}")
    logger.info("")

    config_selection_protocol = str(language_config.get("selection_protocol", "legacy_test_informed"))
    config_selection_bias_safe = bool(language_config.get("selection_bias_safe", config_selection_protocol == "source_only"))

    training_report: dict[str, object] = {
        "language": language,
        "evaluation_protocol": "fixed_config_strict_transfer_holdout",
        "training_mode": "best_config",
        "config_id": str(language_config["config_id"]),
        "config_selection": {
            "protocol": config_selection_protocol,
            "bias_safe": config_selection_bias_safe,
            "selection_metric": str(language_config.get("selection_metric", "unspecified")),
            "source_run_id": str(language_config.get("source_run_id", "")),
        },
        "selection_bias_risk": "low" if config_selection_bias_safe else "high",
        "datasets": all_dataset_ids,
        "samples": int(len(all_labels)),
        "positives": int(np.sum(all_labels == 1)),
        "negatives": int(np.sum(all_labels == 0)),
        "seed": int(args.seed),
        "cascade": {
            "enabled": bool(cascade_strategy_enabled),
            "shortlist_threshold": float(optimal_shortlist_threshold),
            "stage2_weight": float(optimal_stage2_weight),
            "shortlist_coverage_train_mean": float(optimal_shortlist_coverage),
        },
        "source_selection_summary": {
            "threshold": float(threshold_selection.get("threshold", decision_threshold)),
            "margin": float(threshold_selection.get("margin", decision_margin)),
            "applied_threshold": float(
                threshold_selection.get("applied_threshold", decision_threshold + decision_margin)
            ),
            "mean_score": float(threshold_selection.get("mean_score", threshold_selection.get("score", 0.0))),
            "worst_score": float(threshold_selection.get("worst_score", threshold_selection.get("score", 0.0))),
            "mean_precision": float(threshold_selection.get("mean_precision", threshold_selection.get("precision", 0.0))),
            "total_fp": float(threshold_selection.get("total_fp", 0.0)),
            "mean_tp": float(threshold_selection.get("mean_tp", 0.0)),
            "mean_accuracy": float(threshold_selection.get("mean_accuracy", 0.0)),
            "seed_count": int(threshold_selection.get("seed_count", len(robust_seed_values))),
            "valid_candidate_count": int(threshold_selection.get("valid_candidate_count", 0.0)),
            "total_candidate_count": int(threshold_selection.get("total_candidate_count", 0.0)),
            "used_fallback": bool(threshold_selection.get("used_fallback", False)),
        },
        "contrastive_calibration": {
            "enabled": bool(use_contrastive_calibration and stage2_calibrator.enabled),
            "configured": bool(use_contrastive_calibration),
            "cross_fit_folds": int(calibration_folds),
            "cross_fit_used_seed_count": int(calibration_applied_seed_count),
            "cross_fit_total_seed_count": int(len(full_blended_oof_by_seed)),
            "hard_fraction": float(stage2_calibrator.hard_fraction),
            "hard_weight": float(stage2_calibrator.hard_weight),
            "c": float(stage2_calibrator.c),
            "min_samples": int(stage2_calibrator.min_samples),
            "hard_negative_center": float(stage2_calibrator.hard_negative_center),
            "hard_positive_center": float(stage2_calibrator.hard_positive_center),
        },
        "threshold": locked_threshold,
        "margin": locked_margin,
        "strict_outer_leave_one_dataset_out": {
            "summary": strict_outer_summary,
            "folds": strict_outer_report,
        },
    }

    artifact = {
        "language": language,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": FEATURE_NAMES,
        "feature_embedding_model_name": str(args.feature_embedding_model_name),
        "model_name": str(language_config["config_id"]),
        "model_mode": "component_blend_cascade_soft" if cascade_enabled else "component_blend",
        "cascade": {
            "enabled": bool(cascade_strategy_enabled),
            "shortlist_model": str(shortlist_component["model"]),
            "shortlist_model_object": shortlist_model_fitted,
            "shortlist_threshold": float(optimal_shortlist_threshold),
            "stage2_weight": float(optimal_stage2_weight),
        },
        "contrastive_calibration": {
            "enabled": bool(use_contrastive_calibration and hasattr(stage2_calibrator, 'enabled') and stage2_calibrator.enabled),
            "calibrator_object": stage2_calibrator,
        },
        "components": live_trained_components,
        "threshold": decision_threshold,
        "margin": decision_margin,
        "training_report": training_report,
    }
    return artifact, training_report


def train_language_model_search(
    language: str,
    datasets: list[dict[str, object]],
    args: Any,
    archive_language_dir: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Train model using search mode (legacy candidate search)."""
    all_features = np.vstack([dataset["features"] for dataset in datasets])
    all_documents = [doc for dataset in datasets for doc in dataset["documents"]]
    all_labels = np.concatenate([dataset["labels"] for dataset in datasets])
    all_dataset_ids = [str(dataset["dataset_id"]) for dataset in datasets]

    strict_outer_report = leave_one_dataset_out_report(
        datasets=datasets, folds=args.cv_folds, seed=args.seed,
        min_precision=args.min_precision, threshold_step=args.threshold_step,
        margin=args.margin, language=language,
        persist_dir=(archive_language_dir / "holdout") if archive_language_dir else None,
    )
    strict_outer_summary = summarize_fold_report(strict_outer_report)

    selected, all_candidates = select_best_candidate_from_training_rows(
        all_features, all_documents, all_labels,
        folds=args.cv_folds, seed=args.seed,
        min_precision=args.min_precision, threshold_step=args.threshold_step,
    )
    selected_tabular_params = selected["tabular"]["params"]
    selected_neural_params = selected["neural"]["params"]

    locked_threshold = float(selected["threshold"])
    locked_blend_weight = float(selected["blend_weight"])

    # Train final models
    tabular_model = BotEnsembleModel(seed=args.seed, threshold=locked_threshold, margin=args.margin, **selected_tabular_params)
    tabular_model.fit(all_features, all_labels)

    neural_model = NeuralSequenceModel(seed=args.seed, **selected_neural_params)
    neural_model.fit(all_features, all_documents, all_labels)

    training_report: dict[str, object] = {
        "language": language,
        "evaluation_protocol": "nested_outer_leave_one_dataset_out",
        "training_mode": "search",
        "datasets": all_dataset_ids,
        "samples": int(len(all_labels)),
        "selected_candidate": {"name": selected["name"]},
        "threshold": locked_threshold,
        "blend_weight": locked_blend_weight,
        "strict_outer_leave_one_dataset_out": {
            "summary": strict_outer_summary,
            "folds": strict_outer_report,
        },
    }

    artifact = {
        "language": language,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": FEATURE_NAMES,
        "feature_embedding_model_name": str(args.feature_embedding_model_name),
        "model_name": selected["name"],
        "tabular_model": tabular_model.pipeline,
        "neural_model": neural_model,
        "blend_weight": locked_blend_weight,
        "threshold": locked_threshold,
        "margin": float(args.margin),
        "training_report": training_report,
    }
    return artifact, training_report


def train_language_model(
    language: str,
    datasets: list[dict[str, object]],
    args: Any,
    archive_language_dir: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Train a model for a specific language."""
    if args.training_mode == "best_config":
        return train_language_model_best_config(language, datasets, args, archive_language_dir)
    return train_language_model_search(language, datasets, args, archive_language_dir)


def main(argv: list[str] | None = None) -> None:
    """Main training entry point."""
    from src.cli.train_args import parse_args

    args = parse_args(argv)
    logger = get_logger()

    if args.training_mode == "best_config" and args.best_config_overrides_file:
        _apply_best_config_overrides(str(args.best_config_overrides_file), logger)

    run_id, signature, signature_payload, created_at_utc = build_run_identity(args)
    run_dir = Path(args.archive_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run id: %s", run_id)

    dataset_rows = load_training_rows(
        args.dataset_dir,
        feature_embedding_model_name=str(args.feature_embedding_model_name),
    )
    if not dataset_rows:
        raise RuntimeError(f"No training pairs found in: {args.dataset_dir}")

    by_language: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in dataset_rows:
        by_language[str(row["language"])].append(row)

    language_summaries: list[dict[str, object]] = []

    for language, datasets in sorted(by_language.items()):
        datasets = sorted(datasets, key=lambda item: str(item["dataset_id"]))
        logger.info("Training language model: %s", language)

        archive_language_dir = run_dir / "languages" / language
        artifact, report = train_language_model(language, datasets, args, archive_language_dir)

        # Save models
        archive_model_path = archive_language_dir / "unified" / "model.pkl"
        archive_metrics_path = archive_language_dir / "unified" / "metrics.json"
        save_pickle(archive_model_path, artifact)
        save_json(archive_metrics_path, report)

        latest_model_path = Path(args.output_dir) / language / "model.pkl"
        latest_metrics_path = Path(args.output_dir) / language / "metrics.json"
        if not args.no_save_latest:
            save_pickle(latest_model_path, artifact)
            save_json(latest_metrics_path, report)

        logger.info("[%s] threshold=%.3f score=%.1f", language, report["threshold"], report["strict_outer_leave_one_dataset_out"]["summary"]["mean_competition_score"])

        language_summaries.append({
            "language": language,
            "threshold": report["threshold"],
            "archive_model_path": str(archive_model_path),
        })

        append_jsonl(
            Path(args.archive_root) / "scoreboard.jsonl",
            {
                "run_id": run_id,
                "created_at_utc": created_at_utc,
                "signature": signature,
                "git_commit": get_git_commit_hash(),
                "language": language,
                **report["strict_outer_leave_one_dataset_out"]["summary"],
            },
        )

    run_manifest = {
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "signature": signature,
        "git_commit": get_git_commit_hash(),
        "args": vars(args),
        "languages": language_summaries,
    }
    save_json(run_dir / "run_manifest.json", run_manifest)
    append_jsonl(Path(args.archive_root) / "history.jsonl", run_manifest)
    logger.info("Training complete: %s", run_id)
