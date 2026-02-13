"""Training module for bot detection models."""

from src.training.oof import (
    compute_oof_probabilities_component,
    compute_oof_probabilities_neural,
    compute_oof_probabilities_tabular,
)
from src.training.selection import (
    select_blend_and_threshold,
    select_soft_cascade_with_threshold_robust,
    select_threshold_with_margin_grid,
    select_threshold_with_margin_grid_robust,
)

__all__ = [
    "compute_oof_probabilities_tabular",
    "compute_oof_probabilities_neural",
    "compute_oof_probabilities_component",
    "select_threshold_with_margin_grid",
    "select_threshold_with_margin_grid_robust",
    "select_soft_cascade_with_threshold_robust",
    "select_blend_and_threshold",
]
