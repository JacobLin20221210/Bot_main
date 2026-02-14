# Bot or Not (bon)

This repository contains code and experiments for the "Bot or Not" (bon) challenge: an author-level bot detection system for social media posts. The project trains models using engineered features and embeddings, runs predictions on held-out/test sets, and provides evaluation utilities and configuration overrides used during research experiments.

## Quick overview

- **Train models:** feature extraction, candidate generation, and training pipelines under `src/training`.
- **Predict:** a prediction engine that loads trained models and runs batched inference (`predict.py`, `src/prediction/engine.py`).
- **Evaluate:** utilities to compute metrics and calibration (`src/evaluation/metrics.py`, `calc_scores.py`, `validate_predictions.py`).

## Repository layout

- `dataset/` — input datasets (posts, users, bot labels) used for experiments.
- `src/` — core implementation:
  - `src/features/` — feature and embedding extraction.
  - `src/training/` — training pipeline, candidate selection, holdout/OOF logic.
  - `src/models/` — model interfaces, calibration, cascade/ensemble wrappers.
  - `src/prediction/` — prediction engine and wrappers for model inference.
  - `src/evaluation/` — metrics and scoring code.
  - `src/data/loader.py` — dataset loading and preprocessing helpers.
- `output/` — experiment outputs and saved detections, models, and caches.
- `overrides/` — experiment-specific configuration sweeps and overrides.
- CLI entrypoints: `train.py`, `predict.py`, `main.py`, `research.py` for orchestrating experiments.

## Installation

This project uses the Python tooling described in `pyproject.toml`. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .                # installs package in editable mode using pyproject.toml
```

If you use Poetry or another tool, follow the standard steps for that tool instead.

## Quickstart: training, prediction, evaluation

- Train a model (example):

```bash
python train.py --config path/to/config.yaml
```

- Run predictions on a dataset:

```bash
python predict.py --input dataset/posts&users.31.json --output output/detections.31.txt --model-dir output/models/
```

- Score predictions:

```bash
python calc_scores.py --pred output/detections.31.txt --gold dataset/dataset.bots.31.txt
python validate_predictions.py --pred output/detections.31.txt
```

See `src/cli/` for argument parsers used by the command-line scripts.

## How the code works (high level)

- Feature extraction: `src/features` computes hand-crafted features and loads precomputed embeddings from `output/cache/embeddings/`.
- Training pipeline: `src/training/core.py` and `src/training/pipeline.py` orchestrate data folds, candidate selection (`candidates.py`), and model fitting. Holdout and OOF (out-of-fold) predictions are supported for robust evaluation.
- Models: `src/models` exposes abstractions for classifiers, calibration components (`calibration.py`), cascades and ensembles that combine multiple model outputs.
- Prediction engine: `src/prediction/engine.py` loads a model or ensemble and applies it to a stream of author-level features, batching and caching embeddings for efficiency.
- Evaluation: `src/evaluation/metrics.py` contains metric implementations used to score submissions; scripts compute and write human-readable reports to `output/`.

## Configs and overrides

Experiment configuration is kept in YAML/JSON configs loaded by the training and research entrypoints. Use the `overrides/` directory to reproduce particular sweeps or calibration runs. The code supports runtime overrides so you can change thresholds, model paths, and feature flags without changing code.

## Data format

- `dataset/posts&users.*.json` — JSON files combining posts and user metadata; loaders in `src/data/loader.py` parse these into the in-memory structures used by pipelines.
- `dataset/dataset.bots.*.txt` — ground-truth labels used for scoring.

Note: File naming conventions used in `output/` mirror the dataset versions (e.g., `.31.` suffix).

## Outputs

- Model artifacts and metrics saved to `output/models/` and `output/*metrics.json`.
- Prediction outputs (detections) are placed in `output/` with filenames like `detections.31.txt` or `detections.en.30.txt`.

## Development notes

- To trace experiments recreate the `overrides/` used for a given `output/` run.
- The codebase aims to separate feature computation from model logic so features can be reused across model types.
- If you add new features, update `src/features/__init__.py` and ensure embeddings are cached to `output/cache/embeddings/` for repeatable runs.

## Running tests and static checks

There are no formal test suites in the repository root, but you can run quick checks by executing small scripts and verifying outputs. Consider adding `pytest` tests under `tests/` for future CI.

## Where to look next

- `src/training` — understand how experiments are wired together.
- `src/prediction/engine.py` — follow the prediction path used at inference time.
- `calc_scores.py` and `validate_predictions.py` — how outputs are scored and validated.

If you'd like, I can:
- add runnable examples for a single-mini experiment;
- create a `requirements.txt` or lockfile for reproducible installs;
- add a CONTRIBUTING section and developer setup instructions.

---

Updated README to explain project purpose, layout, and how to run core actions.
