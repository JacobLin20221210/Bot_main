from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def save_pickle(path: str | Path, payload: Any) -> None:
    target = ensure_parent(path)
    with target.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = ensure_parent(path)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_detection_file(path: str | Path, user_ids: list[str]) -> None:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        for user_id in user_ids:
            handle.write(f"{user_id}\n")
