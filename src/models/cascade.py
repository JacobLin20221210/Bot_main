"""Cascade blending utilities for two-stage models."""

from __future__ import annotations

import numpy as np


def soft_cascade_blend_probabilities(
    stage1_probabilities: np.ndarray,
    stage2_probabilities: np.ndarray,
    shortlist_threshold: float,
    stage2_weight: float,
) -> np.ndarray:
    """Blend stage1 and stage2 probabilities using soft cascade.

    For samples above shortlist_threshold, blend stage1 and stage2.
    For samples below, keep stage1 probability.
    """
    if stage1_probabilities.shape != stage2_probabilities.shape:
        raise ValueError("Cascade stage probability shapes must match")

    shortlist_mask = stage1_probabilities >= float(shortlist_threshold)
    stage2_mix = (
        float(stage2_weight) * stage2_probabilities
        + (1.0 - float(stage2_weight)) * stage1_probabilities
    )
    return np.where(shortlist_mask, stage2_mix, stage1_probabilities)
