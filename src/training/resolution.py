"""Model resolution from configuration library."""

from __future__ import annotations

from src.utils.config import BEST_LANGUAGE_CONFIGS, TRANSFER_MODEL_LIBRARY


def resolve_model_from_library(model_name: str) -> dict[str, object]:
    """Resolve a model specification from the library."""
    model_spec = TRANSFER_MODEL_LIBRARY.get(model_name)
    if model_spec is None:
        raise ValueError(f"Unknown model in config: {model_name}")
    return {
        "model": model_name,
        "kind": str(model_spec["kind"]),
        "params": dict(model_spec["params"]),
    }


def resolve_best_components(language: str) -> list[dict[str, object]]:
    """Resolve best components for a language."""
    language_config = BEST_LANGUAGE_CONFIGS.get(language)
    if language_config is None:
        raise ValueError(f"No best config found for language: {language}")

    resolved: list[dict[str, object]] = []
    for component in language_config["components"]:
        model_name = str(component["model"])
        resolved_model = resolve_model_from_library(model_name)
        resolved.append({
            "model": resolved_model["model"],
            "weight": float(component["weight"]),
            "kind": resolved_model["kind"],
            "params": resolved_model["params"],
        })
    return resolved


def resolve_cascade_shortlist_component(
    language: str, resolved_components: list[dict[str, object]]
) -> dict[str, object]:
    """Resolve cascade shortlist component configuration."""
    language_config = BEST_LANGUAGE_CONFIGS.get(language, {})
    cascade_cfg = dict(language_config.get("cascade", {}))
    shortlist_model_name = str(cascade_cfg.get("shortlist_model", resolved_components[0]["model"]))
    shortlist_component = resolve_model_from_library(shortlist_model_name)
    return {
        "enabled": bool(cascade_cfg.get("enabled", True)),
        "model": shortlist_component["model"],
        "kind": shortlist_component["kind"],
        "params": shortlist_component["params"],
        "shortlist_threshold_grid": [
            float(x) for x in cascade_cfg.get("shortlist_threshold_grid", [0.25, 0.35, 0.45, 0.55, 0.65])
        ],
        "stage2_weight_grid": [
            float(x) for x in cascade_cfg.get("stage2_weight_grid", [0.7, 0.85, 0.95])
        ],
    }
