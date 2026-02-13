"""Factory for building model components."""

from __future__ import annotations

from typing import Any

from src.models.components.bot_ensemble import BotEnsembleComponent
from src.models.components.embedding_lr import EmbeddingLinearComponent
from src.models.components.graph_knn import GraphKNNComponent
from src.models.components.llm_semantic import LLMSemanticLinearComponent
from src.models.components.neural import NeuralComponent
from src.models.components.tabular_lr import TabularLogisticComponent
from src.models.components.text_lr import TextLinearComponent


def build_component(kind: str, seed: int, params: dict[str, Any]) -> Any:
    """Build a model component by kind."""
    if kind == "bot_ensemble":
        return BotEnsembleComponent(seed=seed, **params)
    if kind == "tab_lr":
        return TabularLogisticComponent(seed=seed, **params)
    if kind == "text_lr":
        return TextLinearComponent(seed=seed, **params)
    if kind == "embedding_lr":
        return EmbeddingLinearComponent(seed=seed, **params)
    if kind == "graph_knn":
        return GraphKNNComponent(seed=seed, **params)
    if kind == "llm_semantic_lr":
        return LLMSemanticLinearComponent(seed=seed, **params)
    if kind == "neural":
        return NeuralComponent(seed=seed, **params)
    raise ValueError(f"Unsupported component kind: {kind}")
