"""Threshold and model selection utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.metrics import evaluate_predictions
from src.models.cascade import soft_cascade_blend_probabilities
from src.models.threshold import select_threshold_and_margin, sweep_thresholds


def select_threshold_with_margin_grid(
    probabilities: np.ndarray,
    labels: np.ndarray,
    min_precision_grid: list[float],
    margin_grid: list[float],
    threshold_step: float,
) -> dict[str, float]:
    """Select threshold using grid search over precision and margin."""
    best: dict[str, float] | None = None

    for min_precision in min_precision_grid:
        threshold_info = sweep_thresholds(
            probabilities,
            labels,
            min_precision=min_precision,
            step=threshold_step,
        )
        base_threshold = float(threshold_info["threshold"])

        for margin in margin_grid:
            applied_threshold = base_threshold + float(margin)
            predictions = (probabilities >= applied_threshold).astype(int)
            metrics = evaluate_predictions(labels, predictions)

            candidate = {
                "threshold": base_threshold,
                "margin": float(margin),
                "applied_threshold": float(applied_threshold),
                "min_precision": float(min_precision),
                "score": float(metrics["competition_score"]),
                "tp": float(metrics["tp"]),
                "fp": float(metrics["fp"]),
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
            }

            if best is None:
                best = candidate
                continue
            if candidate["score"] > best["score"]:
                best = candidate
                continue
            if candidate["score"] == best["score"] and candidate["tp"] > best["tp"]:
                best = candidate
                continue
            if (
                candidate["score"] == best["score"]
                and candidate["tp"] == best["tp"]
                and candidate["accuracy"] > best["accuracy"]
            ):
                best = candidate
                continue
            if (
                candidate["score"] == best["score"]
                and candidate["tp"] == best["tp"]
                and candidate["accuracy"] == best["accuracy"]
                and candidate["precision"] > best["precision"]
            ):
                best = candidate

    if best is None:
        return {
            "threshold": 0.5,
            "margin": 0.0,
            "applied_threshold": 0.5,
            "min_precision": 0.0,
            "score": float("-inf"),
            "tp": 0.0,
            "fp": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
        }
    return best


def select_threshold_with_margin_grid_robust(
    probabilities_by_seed: list[np.ndarray],
    labels: np.ndarray,
    min_precision_grid: list[float],
    margin_grid: list[float],
    threshold_step: float,
    max_fp_rate: float,
) -> dict[str, Any]:
    """Robust threshold selection across multiple seeds."""
    if not probabilities_by_seed:
        raise ValueError("No OOF probability sets provided for robust threshold selection")

    min_precision_values = [float(x) for x in min_precision_grid] if min_precision_grid else [0.0]
    margin_values = [float(x) for x in margin_grid] if margin_grid else [0.0]
    threshold_values = np.arange(
        0.0, 1.0 + max(1e-6, float(threshold_step) * 0.5), float(threshold_step)
    )

    best_valid: dict[str, float] | None = None
    best_fallback: dict[str, float] | None = None
    valid_candidates: list[dict[str, float]] = []
    all_candidates: list[dict[str, float]] = []

    def _is_better(candidate: dict[str, float], current: dict[str, float] | None) -> bool:
        if current is None:
            return True
        candidate_key = (
            candidate["mean_score"],
            candidate["worst_score"],
            -candidate["total_fp"],
            candidate["mean_precision"],
            candidate["applied_threshold"],
        )
        current_key = (
            current["mean_score"],
            current["worst_score"],
            -current["total_fp"],
            current["mean_precision"],
            current["applied_threshold"],
        )
        return candidate_key > current_key

    def _candidate_rank_key(candidate: dict[str, float]) -> tuple[float, float, float, float, float]:
        return (
            candidate["mean_score"],
            candidate["worst_score"],
            -candidate["total_fp"],
            candidate["mean_precision"],
            candidate["applied_threshold"],
        )

    def _candidate_view(candidate: dict[str, float], rank: int) -> dict[str, Any]:
        return {
            "rank": int(rank),
            "threshold": float(candidate["threshold"]),
            "margin": float(candidate["margin"]),
            "applied_threshold": float(candidate["applied_threshold"]),
            "min_precision": float(candidate["min_precision"]),
            "mean_score": float(candidate["mean_score"]),
            "worst_score": float(candidate["worst_score"]),
            "total_fp": float(candidate["total_fp"]),
            "mean_precision": float(candidate["mean_precision"]),
            "mean_tp": float(candidate["mean_tp"]),
            "mean_accuracy": float(candidate["mean_accuracy"]),
        }

    for min_precision in min_precision_values:
        for base_threshold in threshold_values:
            for margin in margin_values:
                applied_threshold = float(base_threshold + margin)
                if applied_threshold > 1.0:
                    continue

                seed_scores: list[float] = []
                seed_fps: list[float] = []
                seed_precisions: list[float] = []
                seed_tps: list[float] = []
                seed_accuracies: list[float] = []
                violates_constraint = False

                for probabilities in probabilities_by_seed:
                    predictions = (probabilities >= applied_threshold).astype(int)
                    metrics = evaluate_predictions(labels, predictions)
                    fp = float(metrics["fp"])
                    tn = float(metrics["tn"])
                    fp_rate = fp / max(1.0, fp + tn)

                    precision = float(metrics["precision"])
                    if precision < min_precision:
                        violates_constraint = True
                    if max_fp_rate >= 0 and fp_rate > max_fp_rate:
                        violates_constraint = True

                    seed_scores.append(float(metrics["competition_score"]))
                    seed_fps.append(fp)
                    seed_precisions.append(precision)
                    seed_tps.append(float(metrics["tp"]))
                    seed_accuracies.append(float(metrics["accuracy"]))

                candidate = {
                    "threshold": float(base_threshold),
                    "margin": float(margin),
                    "applied_threshold": float(applied_threshold),
                    "min_precision": float(min_precision),
                    "worst_score": float(min(seed_scores)),
                    "mean_score": float(np.mean(seed_scores)),
                    "total_fp": float(np.sum(seed_fps)),
                    "mean_precision": float(np.mean(seed_precisions)),
                    "mean_tp": float(np.mean(seed_tps)),
                    "mean_accuracy": float(np.mean(seed_accuracies)),
                    "seed_count": float(len(probabilities_by_seed)),
                    "max_fp_rate": float(max_fp_rate),
                }

                all_candidates.append(candidate)
                if not violates_constraint:
                    valid_candidates.append(candidate)
                if not violates_constraint and _is_better(candidate, best_valid):
                    best_valid = candidate.copy()
                if _is_better(candidate, best_fallback):
                    best_fallback = candidate.copy()

    top_valid = [
        _candidate_view(candidate, rank + 1)
        for rank, candidate in enumerate(
            sorted(valid_candidates, key=_candidate_rank_key, reverse=True)[:5]
        )
    ]
    top_all = [
        _candidate_view(candidate, rank + 1)
        for rank, candidate in enumerate(
            sorted(all_candidates, key=_candidate_rank_key, reverse=True)[:5]
        )
    ]

    selected = best_valid if best_valid is not None else best_fallback
    if selected is None:
        return {
            "threshold": 0.5,
            "margin": 0.0,
            "applied_threshold": 0.5,
            "min_precision": 0.0,
            "worst_score": float("-inf"),
            "mean_score": float("-inf"),
            "score": float("-inf"),
            "total_fp": 0.0,
            "mean_precision": 0.0,
            "mean_tp": 0.0,
            "mean_accuracy": 0.0,
            "seed_count": float(len(probabilities_by_seed)),
            "max_fp_rate": float(max_fp_rate),
            "precision": 0.0,
            "valid_candidate_count": 0.0,
            "total_candidate_count": 0.0,
            "top_valid_candidates": [],
            "top_all_candidates": [],
            "used_fallback": True,
        }

    used_fallback = best_valid is None
    return {
        **selected,
        "score": float(selected["mean_score"]),
        "precision": float(selected["mean_precision"]),
        "valid_candidate_count": float(len(valid_candidates)),
        "total_candidate_count": float(len(all_candidates)),
        "top_valid_candidates": top_valid,
        "top_all_candidates": top_all,
        "used_fallback": used_fallback,
    }


def select_soft_cascade_with_threshold_robust(
    stage1_probabilities_by_seed: list[np.ndarray],
    stage2_probabilities_by_seed: list[np.ndarray],
    labels: np.ndarray,
    shortlist_threshold_grid: list[float],
    stage2_weight_grid: list[float],
    min_precision_grid: list[float],
    margin_grid: list[float],
    threshold_step: float,
    max_fp_rate: float,
) -> dict[str, Any]:
    """Select cascade parameters with robust threshold selection."""
    if not stage1_probabilities_by_seed or not stage2_probabilities_by_seed:
        raise ValueError("Cascade selection requires both stage1 and stage2 probability seeds")
    if len(stage1_probabilities_by_seed) != len(stage2_probabilities_by_seed):
        raise ValueError("Cascade selection requires equal stage1/stage2 seed counts")

    shortlist_values = [float(x) for x in shortlist_threshold_grid] if shortlist_threshold_grid else [0.5]
    stage2_weight_values = [float(x) for x in stage2_weight_grid] if stage2_weight_grid else [0.85]

    best: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []

    for shortlist_threshold in shortlist_values:
        for stage2_weight in stage2_weight_values:
            blended_by_seed: list[np.ndarray] = []
            coverage_values: list[float] = []

            for stage1_probs, stage2_probs in zip(
                stage1_probabilities_by_seed, stage2_probabilities_by_seed, strict=False
            ):
                shortlist_mask = stage1_probs >= shortlist_threshold
                coverage_values.append(float(np.mean(shortlist_mask)))
                blended_by_seed.append(
                    soft_cascade_blend_probabilities(
                        stage1_probabilities=stage1_probs,
                        stage2_probabilities=stage2_probs,
                        shortlist_threshold=shortlist_threshold,
                        stage2_weight=stage2_weight,
                    )
                )

            selection = select_threshold_with_margin_grid_robust(
                blended_by_seed,
                labels,
                min_precision_grid=min_precision_grid,
                margin_grid=margin_grid,
                threshold_step=threshold_step,
                max_fp_rate=max_fp_rate,
            )

            candidate: dict[str, Any] = {
                "shortlist_threshold": float(shortlist_threshold),
                "stage2_weight": float(stage2_weight),
                "shortlist_coverage_mean": float(np.mean(coverage_values)) if coverage_values else 0.0,
                "shortlist_coverage_min": float(min(coverage_values)) if coverage_values else 0.0,
                "selection": selection,
            }
            candidates.append(candidate)

            if best is None:
                best = candidate
                continue

            best_selection = best["selection"]
            candidate_rank = (
                float(selection["score"]),
                float(selection["worst_score"]),
                float(selection["precision"]),
                -float(selection["total_fp"]),
                -float(candidate["shortlist_coverage_mean"]),
            )
            best_rank = (
                float(best_selection["score"]),
                float(best_selection["worst_score"]),
                float(best_selection["precision"]),
                -float(best_selection["total_fp"]),
                -float(best["shortlist_coverage_mean"]),
            )
            if candidate_rank > best_rank:
                best = candidate

    if best is None:
        return {
            "shortlist_threshold": 0.5,
            "stage2_weight": 0.85,
            "shortlist_coverage_mean": 0.0,
            "shortlist_coverage_min": 0.0,
            "selection": {
                "threshold": 0.5,
                "margin": 0.0,
                "applied_threshold": 0.5,
                "min_precision": 0.0,
                "worst_score": float("-inf"),
                "mean_score": float("-inf"),
                "score": float("-inf"),
                "total_fp": 0.0,
                "mean_precision": 0.0,
                "mean_tp": 0.0,
                "mean_accuracy": 0.0,
                "seed_count": float(len(stage1_probabilities_by_seed)),
                "max_fp_rate": float(max_fp_rate),
                "precision": 0.0,
                "valid_candidate_count": 0.0,
                "total_candidate_count": 0.0,
                "top_valid_candidates": [],
                "top_all_candidates": [],
                "used_fallback": True,
            },
            "top_candidates": [],
        }

    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            float(item["selection"]["score"]),
            float(item["selection"]["worst_score"]),
            float(item["selection"]["precision"]),
            -float(item["selection"]["total_fp"]),
            -float(item["shortlist_coverage_mean"]),
        ),
        reverse=True,
    )
    top_candidates = [
        {
            "rank": idx + 1,
            "shortlist_threshold": float(item["shortlist_threshold"]),
            "stage2_weight": float(item["stage2_weight"]),
            "shortlist_coverage_mean": float(item["shortlist_coverage_mean"]),
            "score": float(item["selection"]["score"]),
            "worst_score": float(item["selection"]["worst_score"]),
            "precision": float(item["selection"]["precision"]),
            "total_fp": float(item["selection"]["total_fp"]),
            "applied_threshold": float(item["selection"]["applied_threshold"]),
        }
        for idx, item in enumerate(sorted_candidates[:5])
    ]

    return {
        "shortlist_threshold": float(best["shortlist_threshold"]),
        "stage2_weight": float(best["stage2_weight"]),
        "shortlist_coverage_mean": float(best["shortlist_coverage_mean"]),
        "shortlist_coverage_min": float(best["shortlist_coverage_min"]),
        "selection": dict(best["selection"]),
        "top_candidates": top_candidates,
    }


def select_blend_and_threshold(
    tabular_probabilities: np.ndarray,
    neural_probabilities: np.ndarray,
    labels: np.ndarray,
    blend_weights: list[float],
    min_precision: float,
    threshold_step: float,
) -> dict[str, float]:
    """Select best blend weight and threshold."""
    best: dict[str, float] | None = None

    for blend_weight in blend_weights:
        blended = (blend_weight * tabular_probabilities) + (
            (1.0 - blend_weight) * neural_probabilities
        )
        threshold_info = sweep_thresholds(
            blended,
            labels,
            min_precision=min_precision,
            step=threshold_step,
        )
        candidate = {**threshold_info, "blend_weight": float(blend_weight)}

        if best is None:
            best = candidate
            continue
        if candidate["score"] > best["score"]:
            best = candidate
            continue
        if candidate["score"] == best["score"] and candidate["precision"] > best["precision"]:
            best = candidate

    if best is None:
        return {
            "threshold": 0.5,
            "score": float("-inf"),
            "tp": 0.0,
            "fn": 0.0,
            "fp": 0.0,
            "precision": 0.0,
            "blend_weight": 0.8,
        }
    return best
