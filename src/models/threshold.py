"""Threshold selection utilities."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import evaluate_predictions


def score_classification_result(true_positives: int, false_negatives: int, false_positives: int) -> int:
    """Calculate competition score: +4 for TP, -1 for FN, -2 for FP."""
    return (4 * true_positives) - false_negatives - (2 * false_positives)


def optimize_threshold_grid(
    prob_scores: np.ndarray,
    actual_labels: np.ndarray,
    min_precision: float = 0.0,
    start: float = 0.05,
    stop: float = 0.99,
    step: float = 0.01,
) -> dict[str, float]:
    """Optimize threshold across grid to find best decision boundary."""
    optimal: dict[str, float] | None = None
    threshold_candidates = np.arange(start, stop + 1e-9, step)

    for candidate_threshold in threshold_candidates:
        pred_labels = (prob_scores >= candidate_threshold).astype(int)
        num_tp = int(np.sum((pred_labels == 1) & (actual_labels == 1)))
        num_fn = int(np.sum((pred_labels == 0) & (actual_labels == 1)))
        num_fp = int(np.sum((pred_labels == 1) & (actual_labels == 0)))
        precision_val = num_tp / max(1, num_tp + num_fp)

        if precision_val < min_precision:
            continue

        result_score = score_classification_result(num_tp, num_fn, num_fp)
        option = {
            "threshold": float(candidate_threshold),
            "score": float(result_score),
            "tp": float(num_tp),
            "fn": float(num_fn),
            "fp": float(num_fp),
            "precision": float(precision_val),
        }

        if optimal is None:
            optimal = option
            continue
        if option["score"] > optimal["score"]:
            optimal = option
            continue
        if option["score"] == optimal["score"] and option["precision"] > optimal["precision"]:
            optimal = option

    if optimal is None:
        return {
            "threshold": 0.5,
            "score": float("-inf"),
            "tp": 0.0,
            "fn": 0.0,
            "fp": 0.0,
            "precision": 0.0,
        }
    return optimal


def _calculate_binary_metrics(actual_labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Calculate binary classification metrics."""
    true_pos = int(np.sum((predictions == 1) & (actual_labels == 1)))
    true_neg = int(np.sum((predictions == 0) & (actual_labels == 0)))
    false_pos = int(np.sum((predictions == 1) & (actual_labels == 0)))
    false_neg = int(np.sum((predictions == 0) & (actual_labels == 1)))
    sample_count = max(1, true_pos + true_neg + false_pos + false_neg)
    acc = (true_pos + true_neg) / sample_count
    prec = true_pos / max(1, true_pos + false_pos)
    rec = true_pos / max(1, true_pos + false_neg)
    return {
        "tp": float(true_pos),
        "tn": float(true_neg),
        "fp": float(false_pos),
        "fn": float(false_neg),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "competition_score": float(score_classification_result(true_pos, false_neg, false_pos)),
    }


# Backward compatibility aliases
competition_score = score_classification_result
sweep_thresholds = optimize_threshold_grid


def select_threshold_and_margin(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold_step: float,
    margins: list[float],
    min_precision_grid: list[float],
    max_fp_rate: float = -1.0,
    start: float = 0.05,
    stop: float = 0.99,
) -> dict[str, float]:
    """Select optimal threshold and margin."""
    best: dict[str, float] | None = None
    thresholds = np.arange(start, stop + 1e-9, threshold_step)

    for min_precision in min_precision_grid:
        for margin in margins:
            for threshold in thresholds:
                applied_threshold = float(threshold + margin)
                if applied_threshold >= 1.0:
                    continue

                predictions = (probabilities >= applied_threshold).astype(int)
                metrics = _binary_metrics(labels, predictions)

                if metrics["precision"] < float(min_precision):
                    continue

                fp_rate = float(metrics["fp"] / max(1.0, metrics["fp"] + metrics["tn"]))
                if max_fp_rate >= 0 and fp_rate > max_fp_rate:
                    continue

                candidate = {
                    "threshold": float(threshold),
                    "margin": float(margin),
                    "applied_threshold": float(applied_threshold),
                    "min_precision": float(min_precision),
                    "score": float(metrics["competition_score"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "accuracy": float(metrics["accuracy"]),
                    "fp": float(metrics["fp"]),
                    "fn": float(metrics["fn"]),
                    "tp": float(metrics["tp"]),
                    "tn": float(metrics["tn"]),
                    "fp_rate": float(fp_rate),
                }

                if best is None:
                    best = candidate
                    continue

                candidate_rank = (
                    candidate["score"],
                    candidate["tp"],
                    candidate["precision"],
                    candidate["accuracy"],
                    -candidate["fp"],
                    -candidate["fn"],
                )
                best_rank = (
                    best["score"],
                    best["tp"],
                    best["precision"],
                    best["accuracy"],
                    -best["fp"],
                    -best["fn"],
                )
                if candidate_rank > best_rank:
                    best = candidate

    if best is not None:
        best["used_fallback"] = 0.0
        return best

    # Fallback
    fallback_predictions = (probabilities >= 0.5).astype(int)
    fallback_metrics = _binary_metrics(labels, fallback_predictions)
    fallback_fp_rate = float(
        fallback_metrics["fp"] / max(1.0, fallback_metrics["fp"] + fallback_metrics["tn"])
    )
    return {
        "threshold": 0.5,
        "margin": 0.0,
        "applied_threshold": 0.5,
        "min_precision": 0.0,
        "score": float(fallback_metrics["competition_score"]),
        "precision": float(fallback_metrics["precision"]),
        "recall": float(fallback_metrics["recall"]),
        "accuracy": float(fallback_metrics["accuracy"]),
        "fp": float(fallback_metrics["fp"]),
        "fn": float(fallback_metrics["fn"]),
        "tp": float(fallback_metrics["tp"]),
        "tn": float(fallback_metrics["tn"]),
        "fp_rate": float(fallback_fp_rate),
        "used_fallback": 1.0,
    }
