"""Hard-negative contrastive probability calibration."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


class HardNegativeCalibrator:
    """Probability calibrator that emphasizes hard positives and hard negatives."""

    def __init__(
        self,
        hard_fraction: float = 0.12,
        hard_weight: float = 3.0,
        c: float = 1.0,
        min_samples: int = 24,
    ) -> None:
        self.hard_fraction = float(np.clip(hard_fraction, 0.02, 0.4))
        self.hard_weight = float(max(1.0, hard_weight))
        self.c = float(max(1e-3, c))
        self.min_samples = int(max(8, min_samples))

        self.model: LogisticRegression | None = None
        self.enabled = False
        self.hard_negative_center = 0.8
        self.hard_positive_center = 0.2
        self.hard_negative_width = 0.15
        self.hard_positive_width = 0.15

    @staticmethod
    def _gaussian_kernel_transform(values: np.ndarray, center: float, width: float) -> np.ndarray:
        """Apply Gaussian kernel transformation."""
        denom = max(width, 1e-3)
        return np.exp(-((values - center) ** 2) / (2.0 * (denom**2)))

    def _extract_features(self, probabilities: np.ndarray) -> np.ndarray:
        """Extract feature matrix from probabilities."""
        prob_clipped = np.clip(probabilities.astype(float), 1e-6, 1.0 - 1e-6)
        log_odds = np.log(prob_clipped / (1.0 - prob_clipped))
        hard_neg_kernel = self._gaussian_kernel_transform(prob_clipped, self.hard_negative_center, self.hard_negative_width)
        hard_pos_kernel = self._gaussian_kernel_transform(prob_clipped, self.hard_positive_center, self.hard_positive_width)
        return np.column_stack([log_odds, prob_clipped, hard_neg_kernel, hard_pos_kernel])

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "HardNegativeCalibrator":
        prob_vec = np.asarray(probabilities, dtype=float)
        label_vec = np.asarray(labels, dtype=int)

        if prob_vec.size < self.min_samples:
            self.enabled = False
            return self
        if np.sum(label_vec == 1) < 2 or np.sum(label_vec == 0) < 2:
            self.enabled = False
            return self

        neg_prob_vals = prob_vec[label_vec == 0]
        pos_prob_vals = prob_vec[label_vec == 1]

        hard_neg_thresh = float(np.quantile(neg_prob_vals, 1.0 - self.hard_fraction))
        hard_pos_thresh = float(np.quantile(pos_prob_vals, self.hard_fraction))

        hard_negs = neg_prob_vals[neg_prob_vals >= hard_neg_thresh]
        hard_poss = pos_prob_vals[pos_prob_vals <= hard_pos_thresh]

        if hard_negs.size > 0:
            self.hard_negative_center = float(np.mean(hard_negs))
            self.hard_negative_width = float(max(np.std(hard_negs), 0.04))
        if hard_poss.size > 0:
            self.hard_positive_center = float(np.mean(hard_poss))
            self.hard_positive_width = float(max(np.std(hard_poss), 0.04))

        x_features = self._extract_features(prob_vec)
        weight_vec = np.ones_like(prob_vec, dtype=float)
        weight_vec[(label_vec == 0) & (prob_vec >= hard_neg_thresh)] *= self.hard_weight
        weight_vec[(label_vec == 1) & (prob_vec <= hard_pos_thresh)] *= self.hard_weight

        calibration_model = LogisticRegression(C=self.c, max_iter=1200)
        calibration_model.fit(x_features, label_vec, sample_weight=weight_vec)

        self.model = calibration_model
        self.enabled = True
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=float)
        if not self.enabled or self.model is None:
            return np.clip(probs, 0.0, 1.0)
        x = self._feature_matrix(probs)
        return self.model.predict_proba(x)[:, 1]


def _usable_folds(labels: np.ndarray, requested_folds: int) -> int:
    """Determine usable number of calibration folds given label distribution."""
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives < 2 or negatives < 2:
        return 0
    return max(2, min(int(requested_folds), positives, negatives))


def cross_fit_hard_negative_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
    folds: int,
    seed: int,
    hard_fraction: float,
    hard_weight: float,
    c: float,
    min_samples: int,
) -> tuple[np.ndarray, bool]:
    """Cross-fit hard-negative calibration to reduce threshold-selection overfit.

    Returns calibrated probabilities and whether cross-fitting was actually used.
    """
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    calibrated = np.clip(probs, 0.0, 1.0).copy()

    usable_folds = _usable_folds(y, folds)
    if usable_folds < 2 or probs.size < int(max(8, min_samples)):
        calibrator = HardNegativeCalibrator(
            hard_fraction=hard_fraction,
            hard_weight=hard_weight,
            c=c,
            min_samples=min_samples,
        )
        calibrator.fit(probs, y)
        return calibrator.transform(probs), False

    splitter = StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=int(seed))
    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(probs.reshape(-1, 1), y)):
        calibrator = HardNegativeCalibrator(
            hard_fraction=hard_fraction,
            hard_weight=hard_weight,
            c=c,
            min_samples=min_samples,
        )
        calibrator.fit(probs[train_idx], y[train_idx])

        if calibrator.enabled:
            calibrated[valid_idx] = calibrator.transform(probs[valid_idx])
        else:
            fallback_model = LogisticRegression(C=max(1e-3, float(c)), max_iter=1200)
            fallback_model.fit(probs[train_idx].reshape(-1, 1), y[train_idx])
            calibrated[valid_idx] = fallback_model.predict_proba(probs[valid_idx].reshape(-1, 1))[:, 1]

    return np.clip(calibrated, 0.0, 1.0), True
