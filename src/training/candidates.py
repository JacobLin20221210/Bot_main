"""Candidate model configurations and selection logic."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models.ensemble import BotEnsembleModel
from src.models.neural import NeuralSequenceModel
from src.training.config import BLEND_WEIGHTS, NEURAL_CANDIDATES, TABULAR_CANDIDATES
from src.training.oof import (
    compute_oof_probabilities_neural,
    compute_oof_probabilities_tabular,
)
from src.training.selection import select_blend_and_threshold


def select_best_candidate_from_training_rows(
    features: np.ndarray,
    documents: list[str],
    labels: np.ndarray,
    folds: int,
    seed: int,
    min_precision: float,
    threshold_step: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Select best candidate from training rows using nested CV."""
    candidate_reports: list[dict[str, object]] = []
    candidate_index = 0

    for tabular_candidate in TABULAR_CANDIDATES:
        for neural_candidate in NEURAL_CANDIDATES:
            candidate_seed = seed + (100 * candidate_index)
            tabular_oof = compute_oof_probabilities_tabular(
                features,
                labels,
                folds=folds,
                seed=candidate_seed,
                model_params=tabular_candidate["params"],
            )
            neural_oof = compute_oof_probabilities_neural(
                features,
                documents,
                labels,
                folds=folds,
                seed=candidate_seed,
                neural_params=neural_candidate["params"],
            )
            threshold_info = select_blend_and_threshold(
                tabular_oof,
                neural_oof,
                labels,
                min_precision=min_precision,
                threshold_step=threshold_step,
            )
            blended_oof = threshold_info["blend_weight"] * tabular_oof + (
                1.0 - threshold_info["blend_weight"]
            ) * neural_oof

            from src.evaluation.metrics import evaluate_predictions

            oof_predictions = (blended_oof >= threshold_info["threshold"]).astype(int)
            oof_metrics = evaluate_predictions(labels, oof_predictions)

            candidate_reports.append(
                {
                    "name": f"{tabular_candidate['name']}+{neural_candidate['name']}",
                    "seed": int(candidate_seed),
                    "tabular": tabular_candidate,
                    "neural": neural_candidate,
                    "threshold": float(threshold_info["threshold"]),
                    "blend_weight": float(threshold_info["blend_weight"]),
                    "selection_score": float(threshold_info["score"]),
                    "selection_precision": float(threshold_info["precision"]),
                    "selection_fp": float(threshold_info["fp"]),
                    "selection_tp": float(threshold_info["tp"]),
                    "selection_fn": float(threshold_info["fn"]),
                    "oof_metrics": oof_metrics,
                }
            )
            candidate_index += 1

    candidate_reports.sort(
        key=lambda item: (
            item["selection_score"],
            item["selection_precision"],
            -item["selection_fp"],
        ),
        reverse=True,
    )
    return candidate_reports[0], candidate_reports
