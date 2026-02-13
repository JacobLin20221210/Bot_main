"""Main entry point for bot detection CLI."""

from __future__ import annotations

import sys


def main() -> None:
    """Dispatch to train or predict based on arguments."""
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Import and run training
        import train

        train.main(sys.argv[2:])
    else:
        # Import and run prediction
        import predict

        predict.main(sys.argv[1:])


if __name__ == "__main__":
    main()
