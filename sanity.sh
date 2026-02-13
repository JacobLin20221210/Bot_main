#!/usr/bin/env bash

set -euo pipefail

EXTERNAL_DIR="dataset_external"
FINAL_MODELS_DIR="output/models"
ARCHIVE_ROOT="output/experiments"
EXP_DIR=""
RESULT_ROOT="output/sanity"
MAX_DATASETS="0"
LANG_FILTER="all"
DATASET_KEYS=""
SORT_BY="size"
MAX_POSTS_PER_USER="0"

while [ "$#" -gt 0 ]; do
    if [ "$1" != "${1#-}" ]; then
        :
    elif [ -z "$DATASET_KEYS" ] && [ "$EXTERNAL_DIR" = "dataset_external" ]; then
        EXTERNAL_DIR="$1"
        shift
        continue
    else
        echo "Unknown argument: $1" >&2
        exit 1
    fi

    case "$1" in
        --external-dir)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --external-dir" >&2
                exit 1
            fi
            EXTERNAL_DIR="$2"
            shift 2
            ;;
        --final-models-dir)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --final-models-dir" >&2
                exit 1
            fi
            FINAL_MODELS_DIR="$2"
            shift 2
            ;;
        --archive-root)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --archive-root" >&2
                exit 1
            fi
            ARCHIVE_ROOT="$2"
            shift 2
            ;;
        --exp-dir)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --exp-dir" >&2
                exit 1
            fi
            EXP_DIR="$2"
            shift 2
            ;;
        --result-root)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --result-root" >&2
                exit 1
            fi
            RESULT_ROOT="$2"
            shift 2
            ;;
        --max-datasets)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --max-datasets" >&2
                exit 1
            fi
            MAX_DATASETS="$2"
            shift 2
            ;;
        --lang)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --lang" >&2
                exit 1
            fi
            LANG_FILTER="$2"
            shift 2
            ;;
        --datasets)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --datasets" >&2
                exit 1
            fi
            DATASET_KEYS="$2"
            shift 2
            ;;
        --sort-by)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --sort-by" >&2
                exit 1
            fi
            SORT_BY="$2"
            shift 2
            ;;
        --max-posts-per-user)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --max-posts-per-user" >&2
                exit 1
            fi
            MAX_POSTS_PER_USER="$2"
            shift 2
            ;;
        -h|--help)
            cat <<'USAGE'
Usage: sh sanity.sh [options]

Options:
  --external-dir PATH       External dataset directory (default: dataset_external)
  --final-models-dir PATH   Final model directory (default: output/models)
  --archive-root PATH       Experiment archive root for auto lookup (default: output/experiments)
  --exp-dir PATH            Specific experiment dir containing fold models
  --result-root PATH        Result root directory (default: output/sanity)
  --max-datasets N          Limit number of datasets (0 = all)
  --lang VALUE              all | en | fr (default: all)
  --datasets CSV            Comma-separated dataset keys (example: FSF,TWT)
  --sort-by VALUE           size | name (default: size)
  --max-posts-per-user N    Cap posts per user for faster sanity runs (0 = all)
USAGE
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

uv run python - "$EXTERNAL_DIR" "$FINAL_MODELS_DIR" "$ARCHIVE_ROOT" "$EXP_DIR" "$RESULT_ROOT" "$MAX_DATASETS" "$LANG_FILTER" "$DATASET_KEYS" "$SORT_BY" "$MAX_POSTS_PER_USER" <<'PY'
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.data.loader import group_posts_by_author, infer_language, load_bot_ids, load_json
from src.evaluation.metrics import competition_score
from src.features.matrix import FEATURE_NAMES, build_feature_matrix, build_sequence_documents
from src.prediction.engine import _compute_probabilities
from src.utils.io import load_pickle, save_json, write_detection_file


def find_latest_experiment(archive_root: Path) -> Path:
    candidates = sorted(
        [
            item
            for item in archive_root.iterdir()
            if item.is_dir() and (item / "languages").exists()
        ],
        key=lambda item: item.name,
        reverse=True,
    )
    for candidate in candidates:
        if any((candidate / "languages" / lang / "holdout").exists() for lang in ("en", "fr")):
            return candidate
    raise FileNotFoundError(f"No experiment with holdout models found under: {archive_root}")


def extract_dataset_key(name: str) -> str | None:
    match = re.match(r"dataset\.posts&users\.(.+)\.json$", name)
    return match.group(1) if match else None


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def load_variant(
    path: Path,
    language: str,
    label: str,
    fold_name: str | None,
    fallback_embedding_model_name: str,
    fallback_source: str,
) -> dict[str, object]:
    artifact = load_pickle(path)
    trained_features = artifact.get("feature_names")
    if trained_features != FEATURE_NAMES:
        raise ValueError(f"Feature mismatch for model: {path}")
    raw_embedding_model_name = str(artifact.get("feature_embedding_model_name", "")).strip()
    if raw_embedding_model_name:
        embedding_model_name = raw_embedding_model_name
        embedding_model_source = "artifact"
    else:
        embedding_model_name = str(fallback_embedding_model_name)
        embedding_model_source = str(fallback_source)
    return {
        "language": language,
        "label": label,
        "fold_name": fold_name,
        "path": str(path),
        "threshold": float(artifact["threshold"]),
        "margin": float(artifact.get("margin", 0.0)),
        "embedding_model_name": embedding_model_name,
        "embedding_model_source": embedding_model_source,
        "artifact": artifact,
    }


def collect_model_variants(exp_dir: Path, final_models_dir: Path) -> dict[str, list[dict[str, object]]]:
    variants: dict[str, list[dict[str, object]]] = {"en": [], "fr": []}

    language_defaults = {
        "en": "intfloat/multilingual-e5-small",
        "fr": "intfloat/multilingual-e5-small",
    }
    language_default_sources = {
        "en": "default",
        "fr": "default",
    }

    manifest_path = exp_dir / "run_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest_embedding = str(manifest.get("args", {}).get("feature_embedding_model_name", "")).strip()
        if manifest_embedding:
            for language in ("en", "fr"):
                language_defaults[language] = manifest_embedding
                language_default_sources[language] = "run_manifest"

    for language in ("en", "fr"):
        unified_model_path = exp_dir / "languages" / language / "unified" / "model.pkl"
        if unified_model_path.exists():
            unified_artifact = load_pickle(unified_model_path)
            unified_embedding = str(unified_artifact.get("feature_embedding_model_name", "")).strip()
            if unified_embedding:
                language_defaults[language] = unified_embedding
                language_default_sources[language] = "unified_artifact"

    for language in ("en", "fr"):
        holdout_root = exp_dir / "languages" / language / "holdout"
        fold_dirs = sorted(
            [
                item
                for item in holdout_root.glob("train_*__test_*")
                if item.is_dir() and (item / "model.pkl").exists()
            ],
            key=lambda item: item.name,
        )
        for index, fold_dir in enumerate(fold_dirs, start=1):
            variants[language].append(
                load_variant(
                    path=fold_dir / "model.pkl",
                    language=language,
                    label=f"fold{index}",
                    fold_name=fold_dir.name,
                    fallback_embedding_model_name=language_defaults[language],
                    fallback_source=language_default_sources[language],
                )
            )

        final_model_path = final_models_dir / language / "model.pkl"
        if final_model_path.exists():
            variants[language].append(
                load_variant(
                    path=final_model_path,
                    language=language,
                    label="final",
                    fold_name=None,
                    fallback_embedding_model_name=language_defaults[language],
                    fallback_source=language_default_sources[language],
                )
            )
    return variants


def pooled_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "tp": 0.0,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "datasets": 0.0,
        }
    )
    for row in rows:
        key = (str(row["language"]), str(row["model_variant"]))
        grouped[key]["tp"] += float(row["tp"])
        grouped[key]["tn"] += float(row["tn"])
        grouped[key]["fp"] += float(row["fp"])
        grouped[key]["fn"] += float(row["fn"])
        grouped[key]["datasets"] += 1.0

    summary: list[dict[str, object]] = []
    for (language, model_variant), stats in sorted(grouped.items()):
        tp = int(stats["tp"])
        tn = int(stats["tn"])
        fp = int(stats["fp"])
        fn = int(stats["fn"])
        total = max(1, tp + tn + fp + fn)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        summary.append(
            {
                "language": language,
                "model_variant": model_variant,
                "datasets": int(stats["datasets"]),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": (tp + tn) / total,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "competition_score": float(competition_score(tp=tp, fn=fn, fp=fp)),
            }
        )
    return summary


def main() -> None:
    if len(sys.argv) != 11:
        raise ValueError("Expected 10 arguments")

    external_dir = Path(sys.argv[1]).resolve()
    final_models_dir = Path(sys.argv[2]).resolve()
    archive_root = Path(sys.argv[3]).resolve()
    exp_dir_arg = sys.argv[4].strip()
    result_root = Path(sys.argv[5]).resolve()
    max_datasets = int(sys.argv[6])
    lang_filter = sys.argv[7].strip().lower()
    dataset_keys_csv = sys.argv[8].strip()
    sort_by = sys.argv[9].strip().lower()
    max_posts_per_user = int(sys.argv[10])

    if lang_filter not in {"all", "en", "fr"}:
        raise ValueError("--lang must be one of: all, en, fr")
    if max_datasets < 0:
        raise ValueError("--max-datasets must be >= 0")
    if sort_by not in {"size", "name"}:
        raise ValueError("--sort-by must be one of: size, name")
    if max_posts_per_user < 0:
        raise ValueError("--max-posts-per-user must be >= 0")
    if not external_dir.exists():
        raise FileNotFoundError(f"External dataset dir not found: {external_dir}")
    if not final_models_dir.exists():
        raise FileNotFoundError(f"Final model dir not found: {final_models_dir}")

    exp_dir = Path(exp_dir_arg).resolve() if exp_dir_arg else find_latest_experiment(archive_root)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = result_root / f"{timestamp}-external-sanity"
    run_dir.mkdir(parents=True, exist_ok=True)

    variants = collect_model_variants(exp_dir=exp_dir, final_models_dir=final_models_dir)

    selected_keys = {
        key.strip()
        for key in dataset_keys_csv.split(",")
        if key.strip()
    }

    dataset_paths = [path for path in external_dir.glob("dataset.posts&users.*.json")]
    if selected_keys:
        dataset_paths = [
            path
            for path in dataset_paths
            if (extract_dataset_key(path.name) or "") in selected_keys
        ]

    if sort_by == "size":
        dataset_paths = sorted(dataset_paths, key=lambda item: (item.stat().st_size, item.name))
    else:
        dataset_paths = sorted(dataset_paths, key=lambda item: item.name)

    if max_datasets > 0:
        dataset_paths = dataset_paths[:max_datasets]

    if not dataset_paths:
        raise FileNotFoundError(f"No external datasets found in: {external_dir}")

    print(f"Using experiment: {exp_dir}")
    print(f"Output dir: {run_dir}")
    print(f"Dataset order ({len(dataset_paths)}):")
    for idx, path in enumerate(dataset_paths, start=1):
        key = extract_dataset_key(path.name) or path.name
        print(f"  {idx:>2}. {key:<32} {format_size(path.stat().st_size):>9}")

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for dataset_path in dataset_paths:
        dataset_key = extract_dataset_key(dataset_path.name)
        if dataset_key is None:
            skipped.append({"dataset": str(dataset_path.name), "reason": "invalid_name"})
            continue

        bots_path = external_dir / f"dataset.bots.{dataset_key}.txt"
        if not bots_path.exists():
            skipped.append({"dataset": str(dataset_path.name), "reason": "missing_bots_file"})
            continue

        payload = load_json(dataset_path)
        posts = payload.get("posts", [])
        users = payload.get("users", [])
        language = infer_language(posts)
        if lang_filter != "all" and language != lang_filter:
            continue

        language_variants = variants.get(language, [])
        if not language_variants:
            skipped.append({"dataset": str(dataset_path.name), "reason": f"no_models_for_{language}"})
            continue

        print(
            f"\nDataset: {dataset_key} | language={language} | users={len(users)} | "
            f"posts={len(posts)} | file={format_size(dataset_path.stat().st_size)}",
            flush=True,
        )

        posts_by_author = group_posts_by_author(posts)
        if max_posts_per_user > 0:
            truncated_authors = 0
            truncated_posts = 0
            capped_posts_by_author: dict[str, list[dict[str, object]]] = {}
            for author_id, author_posts in posts_by_author.items():
                post_count = len(author_posts)
                if post_count > max_posts_per_user:
                    capped_posts_by_author[author_id] = list(author_posts[-max_posts_per_user:])
                    truncated_authors += 1
                    truncated_posts += (post_count - max_posts_per_user)
                else:
                    capped_posts_by_author[author_id] = list(author_posts)
            posts_by_author = capped_posts_by_author
            if truncated_authors > 0:
                print(
                    f"  Applied post cap: {max_posts_per_user} per user "
                    f"(trimmed {truncated_posts} posts across {truncated_authors} users)",
                    flush=True,
                )
        bot_ids = load_bot_ids(bots_path)

        per_embedding: dict[str, dict[str, object]] = {}
        unique_embeddings = sorted({str(variant["embedding_model_name"]) for variant in language_variants})
        for embedding_index, embedding_model_name in enumerate(unique_embeddings, start=1):
            print(
                f"  Building features [{embedding_index}/{len(unique_embeddings)}] for embedding: "
                f"{embedding_model_name}",
                flush=True,
            )
            if embedding_model_name in per_embedding:
                continue
            user_ids, features = build_feature_matrix(
                users,
                posts_by_author,
                embedding_model_name=embedding_model_name,
            )
            doc_user_ids, documents = build_sequence_documents(users, posts_by_author)
            if user_ids != doc_user_ids:
                raise ValueError(f"Feature/document mismatch for dataset: {dataset_path}")
            labels = np.array([1 if user_id in bot_ids else 0 for user_id in user_ids], dtype=int)
            per_embedding[embedding_model_name] = {
                "user_ids": user_ids,
                "features": features,
                "documents": documents,
                "labels": labels,
            }

        source_map = ", ".join(
            sorted(
                {
                    f"{str(variant['label'])}:{str(variant.get('embedding_model_source', 'unknown'))}"
                    for variant in language_variants
                }
            )
        )
        print(f"  Embedding source map: {source_map}", flush=True)

        print(f"  Ground-truth bots: {len(bot_ids)}", flush=True)
        print(f"{'Variant':<8} {'Score':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
        print("-" * 72)

        for variant in language_variants:
            embedding_model_name = str(variant["embedding_model_name"])
            cache = per_embedding[embedding_model_name]
            user_ids = list(cache["user_ids"])
            features = cache["features"]
            documents = list(cache["documents"])
            labels = np.array(cache["labels"], dtype=int)

            probabilities = _compute_probabilities(
                artifact=dict(variant["artifact"]),
                features=features,
                documents=documents,
            )
            threshold = float(variant["threshold"])
            margin = float(variant["margin"])
            applied_threshold = threshold + margin
            predictions = (probabilities >= applied_threshold).astype(int)

            tp = int(np.sum((predictions == 1) & (labels == 1)))
            tn = int(np.sum((predictions == 0) & (labels == 0)))
            fp = int(np.sum((predictions == 1) & (labels == 0)))
            fn = int(np.sum((predictions == 0) & (labels == 1)))
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
            score = float(competition_score(tp=tp, fn=fn, fp=fp))

            predicted_ids = [user_id for user_id, pred in zip(user_ids, predictions) if pred == 1]
            detections_dir = run_dir / "detections" / language / str(variant["label"])
            detections_path = detections_dir / f"detections.{language}.{dataset_key}.txt"
            write_detection_file(detections_path, predicted_ids)

            row = {
                "dataset": dataset_key,
                "language": language,
                "users": len(user_ids),
                "gt_bots": len(bot_ids),
                "model_variant": str(variant["label"]),
                "fold_name": variant["fold_name"],
                "model_path": variant["path"],
                "embedding_model_name": embedding_model_name,
                "embedding_model_source": str(variant.get("embedding_model_source", "unknown")),
                "threshold": threshold,
                "margin": margin,
                "applied_threshold": applied_threshold,
                "tp": float(tp),
                "tn": float(tn),
                "fp": float(fp),
                "fn": float(fn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "competition_score": score,
                "predicted_bots": len(predicted_ids),
                "detections_path": str(detections_path),
            }
            rows.append(row)
            print(
                f"{str(variant['label']):<8} {score:>8.1f} {precision:>8.3f} {recall:>8.3f} {f1:>8.3f} "
                f"{tp:>6d} {fp:>6d} {fn:>6d}"
            )

    if not rows:
        raise RuntimeError("No results produced. Check language filter, models, and dataset files.")

    summary_rows = pooled_metrics(rows)

    print("\n" + "=" * 72)
    print("Pooled summary across datasets")
    print("=" * 72)
    print(f"{'Lang':<6} {'Variant':<8} {'Datasets':>8} {'Score':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'FP':>6} {'FN':>6}")
    print("-" * 72)
    for item in summary_rows:
        print(
            f"{str(item['language']):<6} {str(item['model_variant']):<8} {int(item['datasets']):>8d} "
            f"{float(item['competition_score']):>8.1f} {float(item['precision']):>8.3f} "
            f"{float(item['recall']):>8.3f} {float(item['f1']):>8.3f} {int(item['fp']):>6d} {int(item['fn']):>6d}"
        )

    csv_path = run_dir / "external_sanity_rows.csv"
    json_path = run_dir / "external_sanity_rows.json"
    summary_json_path = run_dir / "external_sanity_summary.json"
    skipped_json_path = run_dir / "external_sanity_skipped.json"

    fieldnames = [
        "dataset",
        "language",
        "users",
        "gt_bots",
        "model_variant",
        "fold_name",
        "model_path",
        "embedding_model_name",
        "embedding_model_source",
        "threshold",
        "margin",
        "applied_threshold",
        "tp",
        "tn",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "competition_score",
        "predicted_bots",
        "detections_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    save_json(json_path, {"rows": rows})
    save_json(summary_json_path, {"summary": summary_rows})
    save_json(
        skipped_json_path,
        {
            "skipped": skipped,
            "external_dir": str(external_dir),
            "experiment_dir": str(exp_dir),
            "final_models_dir": str(final_models_dir),
            "lang_filter": lang_filter,
            "max_datasets": max_datasets,
            "sort_by": sort_by,
            "selected_dataset_keys": sorted(selected_keys),
            "max_posts_per_user": max_posts_per_user,
        },
    )

    print(f"\nResults written to: {run_dir}")
    print(f"  Rows CSV: {csv_path}")
    print(f"  Rows JSON: {json_path}")
    print(f"  Summary: {summary_json_path}")
    print(f"  Skipped: {skipped_json_path}")


if __name__ == "__main__":
    main()
PY
