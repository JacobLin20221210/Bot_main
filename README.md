# Bot or Not (bon)

Bot or Not is an author-level bot detector for social media posts.

You give it a dataset (posts + user metadata).  
It extracts features (hand-crafted + embeddings).  
It trains one or more models.  
It outputs a `detections.*.txt` file you can score against gold labels.

---

## What you can do with this repo

1. Train a model (and save artifacts)
2. Predict bots on a dataset
3. Score predictions against ground truth

---

## Repo layout

- `dataset/`  
  Input datasets (posts, users, bot labels)

- `src/`  
  Core code
  - `src/features/` — feature + embedding extraction
  - `src/training/` — folds, holdout/OOF logic, candidate selection, training pipeline
  - `src/models/` — model wrappers, calibration, cascades/ensembles
  - `src/prediction/` — prediction engine (loads models and runs batched inference)
  - `src/evaluation/` — metrics and scoring helpers
  - `src/data/loader.py` — dataset loading/parsing

- `output/`  
  Everything produced by runs (models, metrics, caches, detections)

- `overrides/`  
  Experiment configs and sweeps used for research runs

Entry scripts:
- `train.py` — training pipeline  
- `predict.py` — inference/prediction  
- `calc_scores.py` — scoring  
- `validate_predictions.py` — output sanity checks  
- `main.py`, `research.py` — orchestration / experiments  

---

## Setup

Create a virtual environment and install in editable mode:

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

If you use Poetry or another tool, follow the normal steps for that tool.

---

## Quickstart

### 1) Train

```bash
python train.py --config path/to/config.yaml
```

This writes:
- model artifacts under `output/models/`
- metrics reports under `output/`

### 2) Predict

```bash
python predict.py \
  --input dataset/posts&users.31.json \
  --output output/detections.31.txt \
  --model-dir output/models/
```

### 3) Score + validate

```bash
python calc_scores.py --pred output/detections.31.txt --gold dataset/dataset.bots.31.txt
python validate_predictions.py --pred output/detections.31.txt
```

---

## How it works (high level)

1. **Load dataset**  
   `src/data/loader.py` parses `posts&users.*.json` into the internal format.

2. **Build features**  
   `src/features/` computes engineered features and loads embeddings from cache.  
   Embeddings are typically stored under `output/cache/embeddings/`.

3. **Train**  
   `src/training/core.py` and `src/training/pipeline.py` orchestrate:
   - folds
   - candidate selection (`candidates.py`)
   - model fitting
   - holdout and OOF predictions for robust evaluation

4. **Model composition**  
   `src/models/` contains:
   - base model interfaces
   - calibration (`calibration.py`)
   - cascades / ensembles that blend multiple model outputs

5. **Predict**  
   `src/prediction/engine.py` loads the trained artifacts and runs batched inference.  
   It also handles embedding caching for speed.

6. **Evaluate**  
   `src/evaluation/metrics.py` implements the metrics used by scoring scripts.

---

## Data format

- `dataset/posts&users.*.json`  
  JSON combining posts + user metadata for a dataset version.

- `dataset/dataset.bots.*.txt`  
  Gold labels used for scoring.

Naming convention: dataset versions usually show up as a suffix like `.31.`.

---

## Outputs

- `output/models/`  
  Saved model artifacts.

- `output/*metrics.json` (and similar)  
  Run summaries and evaluation outputs.

- `output/detections.*.txt`  
  Final prediction files (what you score / submit).

---

## Reproducing experiments

Most runs are driven by YAML/JSON configs.  
Look in `overrides/` for known settings and sweep configs.

Tip: if you’re trying to reproduce an old run, start from the override used for it,  
then rerun training and prediction with the same dataset version.

---

## Extending the project

Adding new features:
- implement them in `src/features/`
- wire them into `src/features/__init__.py`
- cache heavy embedding work under `output/cache/embeddings/` so reruns stay fast

---

## Where to look next

If you’re new to the codebase:

1. `src/training/` — how experiments are wired
2. `src/prediction/engine.py` — inference path
3. `calc_scores.py` and `validate_predictions.py` — scoring rules + format checks
