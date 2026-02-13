"""Argument parsing for train.py."""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="Train language-specific bot detectors.")
    parser.add_argument(
        "--dataset-dir", default="dataset", help="Directory containing practice datasets"
    )
    parser.add_argument(
        "--output-dir",
        default="output/models",
        help="Directory for latest models (overwritten)",
    )
    parser.add_argument(
        "--archive-root",
        default="output/experiments",
        help="Directory for timestamped immutable runs",
    )
    parser.add_argument(
        "--run-name", default=None, help="Optional suffix for run folder naming"
    )
    parser.add_argument(
        "--no-save-latest", action="store_true", help="Skip writing to --output-dir latest path"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5, help="Number of CV folds for OOF calibration"
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.0,
        help="Optional precision floor for threshold search",
    )
    parser.add_argument(
        "--threshold-step", type=float, default=0.01, help="Threshold sweep step size"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Extra confidence margin at prediction time",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--training-mode",
        choices=("best_config", "search"),
        default="best_config",
        help="Use centralized best per-language config or legacy nested search",
    )
    parser.add_argument(
        "--feature-embedding-model-name",
        default="intfloat/multilingual-e5-small",
        help="Embedding model used for embedding-derived tabular feature columns",
    )
    parser.add_argument(
        "--best-config-overrides-file",
        default=None,
        help="Optional JSON file with per-language best-config overrides (best_config mode only)",
    )
    return parser.parse_args(argv)
