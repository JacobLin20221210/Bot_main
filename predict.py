"""Prediction script for bot detection."""

from __future__ import annotations

from src.cli.predict_args import parse_args
from src.prediction.engine import run_prediction


def main(argv: list[str] | None = None) -> None:
    """Main entry point for prediction."""
    args = parse_args(argv)
    run_prediction(args)


if __name__ == "__main__":
    main()
