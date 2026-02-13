"""Bot ensemble model using RandomForest and ExtraTrees."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class EnsembleComponentFactory:
    """Factory for building ensemble components."""
    
    @staticmethod
    def create_rf_classifier(seed: int, n_estimators: int, min_samples_leaf: int, bot_weight: float) -> RandomForestClassifier:
        """Create randomized decision forest classifier."""
        return RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            class_weight={0: 1.0, 1: bot_weight},
            n_jobs=-1,
        )
    
    @staticmethod
    def create_et_classifier(seed: int, n_estimators: int, min_samples_leaf: int, bot_weight: float) -> ExtraTreesClassifier:
        """Create extra trees classifier."""
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            class_weight={0: 1.0, 1: bot_weight},
            n_jobs=-1,
        )
    
    @staticmethod
    def create_voter(rf: RandomForestClassifier, et: ExtraTreesClassifier) -> VotingClassifier:
        """Create soft voting ensemble."""
        return VotingClassifier(
            estimators=[("rf", rf), ("et", et)],
            voting="soft",
            n_jobs=-1,
        )
    
    @staticmethod
    def create_calibrated_ensemble(voter: VotingClassifier, cv: int) -> CalibratedClassifierCV:
        """Create calibrated ensemble using sigmoid method."""
        return CalibratedClassifierCV(
            estimator=voter,
            method="sigmoid",
            cv=cv,
        )


@dataclass
class BotEnsembleModel:
    """Ensemble of RandomForest and ExtraTrees with calibration."""

    seed: int = 42
    threshold: float = 0.5
    margin: float = 0.0
    rf_estimators: int = 900
    et_estimators: int = 1200
    min_samples_leaf: int = 2
    rf_bot_weight: float = 1.2
    et_bot_weight: float = 1.3
    calibration_cv: int = 3

    def __post_init__(self) -> None:
        self.pipeline = self._construct_pipeline(
            seed=self.seed,
            rf_estimators=self.rf_estimators,
            et_estimators=self.et_estimators,
            min_samples_leaf=self.min_samples_leaf,
            rf_bot_weight=self.rf_bot_weight,
            et_bot_weight=self.et_bot_weight,
            calibration_cv=self.calibration_cv,
        )

    @staticmethod
    def _construct_pipeline(
        seed: int,
        rf_estimators: int,
        et_estimators: int,
        min_samples_leaf: int,
        rf_bot_weight: float,
        et_bot_weight: float,
        calibration_cv: int,
    ) -> Pipeline:
        factory = EnsembleComponentFactory()
        rf = factory.create_rf_classifier(seed, rf_estimators, min_samples_leaf, rf_bot_weight)
        et = factory.create_et_classifier(seed, et_estimators, min_samples_leaf, et_bot_weight)
        voter = factory.create_voter(rf, et)
        calibrated_voter = factory.create_calibrated_ensemble(voter, calibration_cv)
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", calibrated_voter),
            ]
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "BotEnsembleModel":
        """Fit the model."""
        self.pipeline.fit(features, labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.pipeline.predict_proba(features)[:, 1]

    def predict(
        self,
        features: np.ndarray,
        threshold: float | None = None,
        margin: float | None = None,
    ) -> np.ndarray:
        """Make binary predictions."""
        applied_threshold = self.threshold if threshold is None else threshold
        applied_margin = self.margin if margin is None else margin
        return (self.predict_proba(features) >= (applied_threshold + applied_margin)).astype(
            int
        )
