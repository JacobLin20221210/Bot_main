"""Training configuration constants and candidate definitions."""

from __future__ import annotations

# Tabular model candidates for hyperparameter search
TABULAR_CANDIDATES: list[dict[str, object]] = [
    {
        "name": "balanced_default",
        "params": {
            "rf_estimators": 900,
            "et_estimators": 1200,
            "min_samples_leaf": 2,
            "rf_bot_weight": 1.2,
            "et_bot_weight": 1.3,
            "calibration_cv": 3,
        },
    },
    {
        "name": "precision_tilted",
        "params": {
            "rf_estimators": 1000,
            "et_estimators": 1400,
            "min_samples_leaf": 3,
            "rf_bot_weight": 1.15,
            "et_bot_weight": 1.2,
            "calibration_cv": 3,
        },
    },
    {
        "name": "recall_tilted",
        "params": {
            "rf_estimators": 900,
            "et_estimators": 1200,
            "min_samples_leaf": 2,
            "rf_bot_weight": 1.35,
            "et_bot_weight": 1.45,
            "calibration_cv": 3,
        },
    },
]

# Neural model candidates for hyperparameter search
NEURAL_CANDIDATES: list[dict[str, object]] = [
    {
        "name": "neural_wide",
        "params": {
            "hidden_layer_sizes": (256, 128),
            "alpha": 1e-4,
            "max_iter": 350,
            "text_svd_components": 128,
        },
    },
    {
        "name": "neural_compact",
        "params": {
            "hidden_layer_sizes": (192, 96),
            "alpha": 2e-4,
            "max_iter": 320,
            "text_svd_components": 96,
        },
    },
]

# Blend weights for ensemble
BLEND_WEIGHTS = [0.65, 0.75, 0.85]
