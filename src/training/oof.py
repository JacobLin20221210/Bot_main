"""Out-of-fold probability computation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.models.components.factory import build_component
from src.models.ensemble import BotEnsembleModel
from src.models.neural import NeuralSequenceModel


class FoldValidator:
    """Determine valid fold count from label distribution."""
    
    @staticmethod
    def get_valid_folds(labels: np.ndarray, requested_folds: int) -> int:
        """Compute safe fold count."""
        pos_count = int(np.sum(labels == 1))
        neg_count = int(np.sum(labels == 0))
        if pos_count < 2 or neg_count < 2:
            return 0
        return max(2, min(requested_folds, pos_count, neg_count))


def compute_oof_probabilities_tabular(
    features: np.ndarray,
    labels: np.ndarray,
    folds: int,
    seed: int,
    model_params: dict[str, object],
) -> np.ndarray:
    """Compute out-of-fold probabilities using tabular model."""
    valid_fold_count = FoldValidator.get_valid_folds(labels, folds)

    if valid_fold_count < 2:
        model_instance = BotEnsembleModel(seed=seed, **model_params)
        model_instance.fit(features, labels)
        return model_instance.predict_proba(features)

    kfold = StratifiedKFold(n_splits=valid_fold_count, shuffle=True, random_state=seed)
    pred_oof = np.zeros(len(labels), dtype=float)

    for fold_idx, (idx_train, idx_valid) in enumerate(kfold.split(features, labels)):
        tabular_model = BotEnsembleModel(seed=seed + fold_idx, **model_params)
        tabular_model.fit(features[idx_train], labels[idx_train])
        pred_oof[idx_valid] = tabular_model.predict_proba(features[idx_valid])

    return pred_oof


def compute_oof_probabilities_neural(
    features: np.ndarray,
    documents: list[str],
    labels: np.ndarray,
    folds: int,
    seed: int,
    neural_params: dict[str, object],
) -> np.ndarray:
    """Compute out-of-fold probabilities using neural model."""
    valid_fold_count = FoldValidator.get_valid_folds(labels, folds)

    if valid_fold_count < 2:
        neural_instance = NeuralSequenceModel(seed=seed, **neural_params)
        neural_instance.fit(features, documents, labels)
        return neural_instance.predict_proba(features, documents)

    kfold = StratifiedKFold(n_splits=valid_fold_count, shuffle=True, random_state=seed)
    pred_oof = np.zeros(len(labels), dtype=float)

    for fold_idx, (idx_train, idx_valid) in enumerate(kfold.split(features, labels)):
        train_documents = [documents[idx] for idx in idx_train]
        valid_documents = [documents[idx] for idx in idx_valid]

        neural_model = NeuralSequenceModel(seed=seed + fold_idx, **neural_params)
        neural_model.fit(features[idx_train], train_documents, labels[idx_train])
        pred_oof[idx_valid] = neural_model.predict_proba(features[idx_valid], valid_documents)

    return pred_oof


def compute_oof_probabilities_component(
    features: np.ndarray,
    documents: list[str],
    labels: np.ndarray,
    folds: int,
    seed: int,
    kind: str,
    params: dict[str, Any],
) -> np.ndarray:
    """Compute out-of-fold probabilities using a generic component."""
    valid_fold_count = FoldValidator.get_valid_folds(labels, folds)

    if valid_fold_count < 2:
        component_instance = build_component(kind=kind, seed=seed, params=params)
        component_instance.fit(features, documents, labels)
        return component_instance.predict_proba(features, documents)

    kfold = StratifiedKFold(n_splits=valid_fold_count, shuffle=True, random_state=seed)
    pred_oof = np.zeros(len(labels), dtype=float)

    for fold_idx, (idx_train, idx_valid) in enumerate(kfold.split(features, labels)):
        component_model = build_component(kind=kind, seed=seed + fold_idx, params=params)
        train_documents = [documents[idx] for idx in idx_train]
        valid_documents = [documents[idx] for idx in idx_valid]
        component_model.fit(features[idx_train], train_documents, labels[idx_train])
        pred_oof[idx_valid] = component_model.predict_proba(features[idx_valid], valid_documents)

    return pred_oof
