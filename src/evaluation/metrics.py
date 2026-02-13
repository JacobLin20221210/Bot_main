"""Evaluation metrics for bot detection."""

from __future__ import annotations

import numpy as np


def competition_score(tp: int, fn: int, fp: int) -> int:
    """Calculate competition score: +4 for TP, -1 for FN, -2 for FP."""
    return (4 * tp) - fn - (2 * fp)


def evaluate_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))

    total = max(1, len(labels))
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "competition_score": float(competition_score(tp, fn, fp)),
    }
