"""Argument parsing for research.py."""

from __future__ import annotations

import argparse
import re


SELECTION_PROTOCOL_SOURCE_ONLY = "source_only"
SELECTION_PROTOCOL_LEGACY_TEST_INFORMED = "legacy_test_informed"


def _sanitize_name(value: str | None) -> str:
    """Sanitize run name for filesystem safety."""
    if not value:
        return ""
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    return normalized.strip("-._")[:80]


def _parse_float_grid(raw: str) -> list[float]:
    """Parse comma-separated float values."""
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    deduped = sorted(set(values))
    if not deduped:
        raise ValueError("Float grid cannot be empty")
    return deduped


def _parse_method_grid(raw: str) -> list[str]:
    """Parse comma-separated method names."""
    methods = [token.strip().lower() for token in raw.split(",") if token.strip()]
    if not methods:
        return ["sigmoid"]
    deduped: list[str] = []
    for method in methods:
        if method not in deduped:
            deduped.append(method)
    return deduped


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for research."""
    parser = argparse.ArgumentParser(
        description="Language-focused heavy transfer benchmark runner"
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument(
        "--archive-root", default="output/experiments/transfer_research"
    )
    parser.add_argument("--language", choices=("en", "fr"), default="fr")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--threshold-step", type=float, default=0.0025)
    parser.add_argument(
        "--min-precision-grid", default="0.72,0.78,0.84,0.90,0.94"
    )
    parser.add_argument("--margin-grid", default="0.0,0.005,0.01,0.015,0.02,0.03")
    parser.add_argument("--blend-weights", default="0.15,0.25,0.35,0.5,0.65,0.75,0.85")
    parser.add_argument(
        "--cascade-shortlist-thresholds", default="0.25,0.35,0.45,0.55,0.65"
    )
    parser.add_argument("--cascade-stage2-weights", default="0.7,0.85,0.95")
    parser.add_argument("--cascade-top-k", type=int, default=4)
    parser.add_argument("--blend-top-k", type=int, default=8)
    parser.add_argument("--meta-top-k", type=int, default=6)
    parser.add_argument("--meta-calibration-methods", default="sigmoid,isotonic")
    parser.add_argument("--max-fp-rate", type=float, default=-1.0)
    parser.add_argument(
        "--selection-protocol",
        choices=(SELECTION_PROTOCOL_SOURCE_ONLY, SELECTION_PROTOCOL_LEGACY_TEST_INFORMED),
        default=SELECTION_PROTOCOL_SOURCE_ONLY,
    )
    parser.add_argument("--simplicity-epsilon", type=float, default=1.0)
    parser.add_argument("--max-experiments", type=int, default=0)
    parser.add_argument("--print-status", action="store_true")
    parser.add_argument("--include-boosters", action="store_true")
    parser.add_argument(
        "--feature-embedding-model-name",
        default="intfloat/multilingual-e5-small",
    )

    args = parser.parse_args(argv)

    # Parse grid arguments
    args.min_precision_grid = _parse_float_grid(args.min_precision_grid)
    args.margin_grid = _parse_float_grid(args.margin_grid)
    args.blend_weights = _parse_float_grid(args.blend_weights)
    args.cascade_shortlist_thresholds = _parse_float_grid(args.cascade_shortlist_thresholds)
    args.cascade_stage2_weights = _parse_float_grid(args.cascade_stage2_weights)
    args.meta_calibration_methods = _parse_method_grid(args.meta_calibration_methods)
    args.run_name = _sanitize_name(args.run_name)

    return args
