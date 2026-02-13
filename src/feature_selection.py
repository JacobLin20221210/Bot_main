from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold

from src.data import discover_training_pairs, group_posts_by_author, load_dataset_bundle
from src.evaluate import evaluate_predictions
from src.features import FEATURE_NAMES, build_feature_matrix
from src.io_utils import save_json
from src.logger import get_logger
from src.model import BotEnsembleModel, sweep_thresholds


ANALYSIS_MODEL_PARAMS = {
    "rf_estimators": 260,
    "et_estimators": 360,
    "min_samples_leaf": 2,
    "rf_bot_weight": 1.2,
    "et_bot_weight": 1.3,
    "calibration_cv": 3,
}


@dataclass
class FoldModel:
    model: BotEnsembleModel
    valid_indices: np.ndarray
    local_seed: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid feature selection for bot detection.")
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing practice datasets")
    parser.add_argument("--output-dir", default="output/analysis", help="Directory for analysis outputs")
    parser.add_argument("--cv-folds", type=int, default=3, help="CV folds for OOF scoring")
    parser.add_argument("--threshold-step", type=float, default=0.01, help="Threshold sweep step for scoring")
    parser.add_argument("--min-precision", type=float, default=0.0, help="Optional precision floor")
    parser.add_argument("--corr-threshold", type=float, default=0.92, help="Redundancy pruning threshold")
    parser.add_argument("--min-features", type=int, default=6, help="Minimum features to consider before early stop")
    parser.add_argument("--max-features", type=int, default=18, help="Maximum features to consider in subset search")
    parser.add_argument("--min-improvement", type=float, default=1.0, help="Minimum score gain considered meaningful")
    parser.add_argument("--plateau-patience", type=int, default=3, help="Stop after this many non-improving steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(argv)


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    corr = spearmanr(a, b).correlation
    if corr is None or np.isnan(corr):
        return 0.0
    return float(corr)


def _usable_folds(labels: np.ndarray, requested_folds: int) -> int:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives <= 1 or negatives <= 1:
        return 1
    return max(2, min(requested_folds, positives, negatives))


def _fit_fold_models(
    features: np.ndarray,
    labels: np.ndarray,
    folds: int,
    seed: int,
) -> list[FoldModel]:
    usable_folds = _usable_folds(labels, folds)
    if usable_folds <= 1:
        model = BotEnsembleModel(seed=seed, **ANALYSIS_MODEL_PARAMS)
        model.fit(features, labels)
        return [FoldModel(model=model, valid_indices=np.arange(len(labels)), local_seed=seed)]

    splitter = StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=seed)
    fold_models: list[FoldModel] = []
    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(features, labels)):
        local_seed = seed + fold_id
        model = BotEnsembleModel(seed=local_seed, **ANALYSIS_MODEL_PARAMS)
        model.fit(features[train_idx], labels[train_idx])
        fold_models.append(FoldModel(model=model, valid_indices=valid_idx, local_seed=local_seed))
    return fold_models


def _oof_probabilities_with_models(features: np.ndarray, fold_models: list[FoldModel]) -> np.ndarray:
    oof = np.zeros(features.shape[0], dtype=float)
    for fold in fold_models:
        oof[fold.valid_indices] = fold.model.predict_proba(features[fold.valid_indices])
    return oof


def _score_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold_step: float,
    min_precision: float,
    fixed_threshold: float | None = None,
) -> dict[str, float]:
    if fixed_threshold is None:
        threshold_info = sweep_thresholds(
            probabilities,
            labels,
            min_precision=min_precision,
            step=threshold_step,
        )
        threshold = float(threshold_info["threshold"])
    else:
        threshold = float(fixed_threshold)

    predictions = (probabilities >= threshold).astype(int)
    metrics = evaluate_predictions(labels, predictions)
    metrics["threshold"] = threshold
    return metrics


def _evaluate_subset(
    full_features: np.ndarray,
    labels: np.ndarray,
    feature_indices: list[int],
    folds: int,
    seed: int,
    threshold_step: float,
    min_precision: float,
) -> dict[str, float]:
    subset = full_features[:, feature_indices]
    fold_models = _fit_fold_models(subset, labels, folds=folds, seed=seed)
    probabilities = _oof_probabilities_with_models(subset, fold_models)
    return _score_probabilities(
        probabilities,
        labels,
        threshold_step=threshold_step,
        min_precision=min_precision,
    )


def _compute_univariate_relevance(features: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pearsons: list[float] = []
    spearmans: list[float] = []
    for col in range(features.shape[1]):
        x = features[:, col]
        pearsons.append(abs(_safe_corr(x, labels)))
        spearmans.append(abs(_safe_spearman(x, labels)))

    n_neighbors = 3 if len(labels) > 8 else 2
    mutual_info = mutual_info_classif(
        features,
        labels,
        discrete_features=False,
        random_state=seed,
        n_neighbors=n_neighbors,
    )
    mutual_info = np.nan_to_num(mutual_info, nan=0.0, posinf=0.0, neginf=0.0)
    return np.array(pearsons, dtype=float), np.array(spearmans, dtype=float), np.array(mutual_info, dtype=float)


def _compute_permutation_importance(
    features: np.ndarray,
    labels: np.ndarray,
    folds: int,
    seed: int,
    threshold_step: float,
    min_precision: float,
) -> tuple[np.ndarray, dict[str, float]]:
    fold_models = _fit_fold_models(features, labels, folds=folds, seed=seed)
    baseline_probabilities = _oof_probabilities_with_models(features, fold_models)
    baseline_metrics = _score_probabilities(
        baseline_probabilities,
        labels,
        threshold_step=threshold_step,
        min_precision=min_precision,
    )
    baseline_threshold = float(baseline_metrics["threshold"])
    baseline_score = float(baseline_metrics["competition_score"])

    drops = np.zeros(features.shape[1], dtype=float)
    for col in range(features.shape[1]):
        perm_probabilities = np.zeros(features.shape[0], dtype=float)
        for fold in fold_models:
            valid_idx = fold.valid_indices
            local = features[valid_idx].copy()
            rng = np.random.default_rng(seed + (col * 97) + fold.local_seed)
            shuffled = local[:, col].copy()
            rng.shuffle(shuffled)
            local[:, col] = shuffled
            perm_probabilities[valid_idx] = fold.model.predict_proba(local)

        perm_metrics = _score_probabilities(
            perm_probabilities,
            labels,
            threshold_step=threshold_step,
            min_precision=min_precision,
            fixed_threshold=baseline_threshold,
        )
        drops[col] = baseline_score - float(perm_metrics["competition_score"])
    return drops, baseline_metrics


def _redundancy_prune(
    features: np.ndarray,
    ranked_indices: list[int],
    feature_names: list[str],
    corr_threshold: float,
) -> tuple[list[int], list[dict[str, object]]]:
    corr_matrix = np.zeros((features.shape[1], features.shape[1]), dtype=float)
    for i in range(features.shape[1]):
        corr_matrix[i, i] = 1.0
        for j in range(i + 1, features.shape[1]):
            corr = abs(_safe_spearman(features[:, i], features[:, j]))
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    kept: list[int] = []
    dropped: list[dict[str, object]] = []
    for idx in ranked_indices:
        if not kept:
            kept.append(idx)
            continue

        max_corr = -1.0
        max_partner = kept[0]
        for existing in kept:
            corr = float(corr_matrix[idx, existing])
            if corr > max_corr:
                max_corr = corr
                max_partner = existing

        if max_corr >= corr_threshold:
            dropped.append(
                {
                    "feature": feature_names[idx],
                    "reason": "redundant",
                    "correlated_with": feature_names[max_partner],
                    "abs_spearman": max_corr,
                }
            )
        else:
            kept.append(idx)

    return kept, dropped


def _run_language_selection(
    language: str,
    features: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    logger = get_logger()
    logger.info("Feature analysis language=%s samples=%s positives=%s", language, len(labels), int(np.sum(labels == 1)))

    pearson_abs, spearman_abs, mutual_info = _compute_univariate_relevance(features, labels, args.seed)
    permutation_drop, baseline_metrics = _compute_permutation_importance(
        features,
        labels,
        folds=args.cv_folds,
        seed=args.seed,
        threshold_step=args.threshold_step,
        min_precision=args.min_precision,
    )

    univariate_score = (
        _normalize(pearson_abs) + _normalize(spearman_abs) + _normalize(mutual_info)
    ) / 3.0
    hybrid_score = (0.5 * univariate_score) + (0.5 * _normalize(permutation_drop))

    ranked_indices = list(np.argsort(-hybrid_score))
    pruned_indices, dropped_redundant = _redundancy_prune(
        features,
        ranked_indices,
        FEATURE_NAMES,
        corr_threshold=args.corr_threshold,
    )

    ranking_rows: list[dict[str, object]] = []
    for rank, idx in enumerate(ranked_indices, start=1):
        ranking_rows.append(
            {
                "rank": rank,
                "feature": FEATURE_NAMES[idx],
                "pearson_abs": float(pearson_abs[idx]),
                "spearman_abs": float(spearman_abs[idx]),
                "mutual_info": float(mutual_info[idx]),
                "univariate_score": float(univariate_score[idx]),
                "permutation_score_drop": float(permutation_drop[idx]),
                "hybrid_score": float(hybrid_score[idx]),
                "kept_after_redundancy_prune": bool(idx in pruned_indices),
            }
        )

    max_k = min(args.max_features, len(pruned_indices))
    min_k = min(args.min_features, max_k)

    full_feature_metrics = _evaluate_subset(
        features,
        labels,
        feature_indices=list(range(features.shape[1])),
        folds=args.cv_folds,
        seed=args.seed,
        threshold_step=args.threshold_step,
        min_precision=args.min_precision,
    )

    curve: list[dict[str, object]] = []
    best_k = 1
    best_metrics: dict[str, float] | None = None
    non_improving_steps = 0

    for k in range(1, max_k + 1):
        subset_indices = pruned_indices[:k]
        metrics = _evaluate_subset(
            features,
            labels,
            feature_indices=subset_indices,
            folds=args.cv_folds,
            seed=args.seed,
            threshold_step=args.threshold_step,
            min_precision=args.min_precision,
        )

        if best_metrics is None:
            best_metrics = metrics
            best_k = k
            improved = True
        else:
            improved = False
            score_gain = float(metrics["competition_score"] - best_metrics["competition_score"])
            if score_gain >= args.min_improvement:
                improved = True
            elif score_gain > 0 and k <= min_k:
                improved = True
            elif score_gain == 0:
                if float(metrics["precision"]) > float(best_metrics["precision"]):
                    improved = True
                elif float(metrics["precision"]) == float(best_metrics["precision"]) and float(metrics["fp"]) < float(best_metrics["fp"]):
                    improved = True

            if improved:
                best_metrics = metrics
                best_k = k

        if improved:
            non_improving_steps = 0
        else:
            non_improving_steps += 1

        curve.append(
            {
                "k": k,
                "features": [FEATURE_NAMES[idx] for idx in subset_indices],
                "metrics": metrics,
                "is_current_best": k == best_k,
            }
        )

        if k >= min_k and non_improving_steps >= args.plateau_patience:
            break

    selected_indices = pruned_indices[:best_k]
    selected_features = [FEATURE_NAMES[idx] for idx in selected_indices]

    result = {
        "language": language,
        "samples": int(len(labels)),
        "positives": int(np.sum(labels == 1)),
        "negatives": int(np.sum(labels == 0)),
        "analysis_model_params": ANALYSIS_MODEL_PARAMS,
        "baseline_full_feature_metrics": baseline_metrics,
        "baseline_full_feature_subset_cv_metrics": full_feature_metrics,
        "ranking": ranking_rows,
        "redundancy_pruned_features": [FEATURE_NAMES[idx] for idx in pruned_indices],
        "redundancy_dropped": dropped_redundant,
        "selection_curve": curve,
        "selected_feature_count": int(len(selected_features)),
        "selected_features": selected_features,
        "selected_metrics": best_metrics,
        "full_feature_metrics": full_feature_metrics,
        "score_delta_vs_full": None
        if best_metrics is None
        else float(best_metrics["competition_score"] - full_feature_metrics["competition_score"]),
    }
    return result


def _load_language_matrix(dataset_dir: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    by_language_x: dict[str, list[np.ndarray]] = {}
    by_language_y: dict[str, list[np.ndarray]] = {}

    for dataset_path, bots_path in discover_training_pairs(dataset_dir):
        bundle = load_dataset_bundle(dataset_path, bots_path)
        language = str(bundle["language"])
        posts_by_author = group_posts_by_author(bundle["posts"])
        user_ids, features = build_feature_matrix(bundle["users"], posts_by_author)
        labels = np.array([1 if user_id in bundle["bot_ids"] else 0 for user_id in user_ids], dtype=int)

        by_language_x.setdefault(language, []).append(features)
        by_language_y.setdefault(language, []).append(labels)

    merged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for language in sorted(by_language_x):
        merged[language] = (
            np.vstack(by_language_x[language]),
            np.concatenate(by_language_y[language]),
        )
    return merged


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = get_logger()

    language_data = _load_language_matrix(args.dataset_dir)
    if not language_data:
        raise RuntimeError(f"No datasets found in {args.dataset_dir}")

    output_root = Path(args.output_dir)
    summary: dict[str, object] = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(output_root),
        "feature_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "languages": {},
    }

    for language, (features, labels) in language_data.items():
        result = _run_language_selection(language, features, labels, args)
        save_json(output_root / f"feature_selection.{language}.json", result)
        summary["languages"][language] = {
            "selected_feature_count": result["selected_feature_count"],
            "selected_features": result["selected_features"],
            "selected_metrics": result["selected_metrics"],
            "full_feature_metrics": result["full_feature_metrics"],
            "score_delta_vs_full": result["score_delta_vs_full"],
        }
        logger.info(
            "[%s] selected=%s score=%.1f full=%.1f delta=%.1f",
            language,
            result["selected_feature_count"],
            result["selected_metrics"]["competition_score"],
            result["full_feature_metrics"]["competition_score"],
            result["score_delta_vs_full"],
        )

    save_json(output_root / "feature_selection.summary.json", summary)
    logger.info("Saved feature selection analysis to %s", output_root)


if __name__ == "__main__":
    main()
