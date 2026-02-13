"""Argument parsing for predict.py."""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for prediction."""
    parser = argparse.ArgumentParser(description="Predict bot accounts for a dataset.")
    parser.add_argument("dataset_path", help="Path to dataset.posts&users.<id>.json")
    parser.add_argument(
        "--models-dir", default="output/models", help="Directory containing language models"
    )
    parser.add_argument("--output-path", default=None, help="Optional output file path")
    parser.add_argument("--threshold", type=float, default=None, help="Optional threshold override")
    parser.add_argument("--margin", type=float, default=None, help="Optional margin override")
    parser.add_argument(
        "--blind",
        action="store_true",
        help="Enable ID-only blind heuristic (predict bot if user_id matches UUIDv4)",
    )
    return parser.parse_args(argv)
