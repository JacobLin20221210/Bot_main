"""CLI argument parsing modules."""

from src.cli.train_args import parse_args as parse_train_args
from src.cli.predict_args import parse_args as parse_predict_args
from src.cli.research_args import parse_args as parse_research_args

__all__ = ["parse_train_args", "parse_predict_args", "parse_research_args"]
