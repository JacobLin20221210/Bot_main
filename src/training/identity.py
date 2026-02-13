"""Run identity and tracking utilities."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.training.config import BLEND_WEIGHTS, NEURAL_CANDIDATES, TABULAR_CANDIDATES
from src.utils.config import BEST_LANGUAGE_CONFIGS


def sanitize_run_name(name: str | None) -> str:
    """Sanitize run name for filesystem safety."""
    if not name:
        return ""
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip())
    return normalized.strip("-._")[:80]


def build_run_identity(
    args: Any,
) -> tuple[str, str, dict[str, object], str]:
    """Build run identity with signature.
    
    Returns:
        Tuple of (run_id, digest, signature_payload, created_at_iso)
    """
    timestamp = datetime.now(timezone.utc)
    timestamp_str = timestamp.strftime("%Y%m%d-%H%M%S")

    signature_payload: dict[str, object] = {
        "dataset_dir": str(args.dataset_dir),
        "cv_folds": int(args.cv_folds),
        "min_precision": float(args.min_precision),
        "threshold_step": float(args.threshold_step),
        "margin": float(args.margin),
        "seed": int(args.seed),
        "training_mode": str(args.training_mode),
        "tabular_candidates": TABULAR_CANDIDATES,
        "neural_candidates": NEURAL_CANDIDATES,
        "blend_weights": BLEND_WEIGHTS,
        "best_language_configs": BEST_LANGUAGE_CONFIGS,
    }

    digest = hashlib.sha1(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]

    name_suffix = sanitize_run_name(args.run_name)
    run_id = (
        f"{timestamp_str}-{digest}"
        if not name_suffix
        else f"{timestamp_str}-{digest}-{name_suffix}"
    )

    return run_id, digest, signature_payload, timestamp.isoformat()
