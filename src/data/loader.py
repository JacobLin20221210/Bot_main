"""Data loading utilities."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DATASET_PATTERN = re.compile(r"dataset\.posts&users\.(\d+)\.json$")


def discover_training_pairs(dataset_dir: str | Path) -> list[tuple[Path, Path]]:
    base = Path(dataset_dir)
    pairs: list[tuple[Path, Path]] = []
    for dataset_path in sorted(base.glob("dataset.posts&users.*.json")):
        match = DATASET_PATTERN.search(dataset_path.name)
        if not match:
            continue
        dataset_id = match.group(1)
        bots_path = base / f"dataset.bots.{dataset_id}.txt"
        if bots_path.exists():
            pairs.append((dataset_path, bots_path))
    return pairs


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_bot_ids(path: str | Path) -> set[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def extract_dataset_id(dataset_path: str | Path) -> str:
    match = DATASET_PATTERN.search(Path(dataset_path).name)
    if not match:
        raise ValueError(f"Could not extract dataset id from: {dataset_path}")
    return match.group(1)


def infer_language(posts: list[dict[str, Any]]) -> str:
    counts = Counter((post.get("lang") or "").lower() for post in posts)
    top_lang = counts.most_common(1)[0][0] if counts else ""
    if top_lang.startswith("fr"):
        return "fr"
    return "en"


def group_posts_by_author(posts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for post in posts:
        author_id = str(post.get("author_id", ""))
        grouped.setdefault(author_id, []).append(post)
    return grouped


def load_dataset_bundle(
    dataset_path: str | Path,
    bots_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = load_json(dataset_path)
    posts = payload.get("posts", [])
    users = payload.get("users", [])
    dataset_id = extract_dataset_id(dataset_path)
    bot_ids = load_bot_ids(bots_path) if bots_path else set()
    return {
        "dataset_id": dataset_id,
        "language": infer_language(posts),
        "posts": posts,
        "users": users,
        "bot_ids": bot_ids,
    }
