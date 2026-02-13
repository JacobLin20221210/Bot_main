"""Research script for transfer learning experiments."""

from __future__ import annotations

from src.cli.research_args import parse_args
from src.research.main import run_research


def main(argv: list[str] | None = None) -> None:
    """Main entry point for research."""
    args = parse_args(argv)
    run_research(args)


if __name__ == "__main__":
    main()
