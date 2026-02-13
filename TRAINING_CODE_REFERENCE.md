# Training Algorithm - Code Reference

Quick reference to key code snippets implementing each phase.

---

## Phase 1: Feature Extraction

**File:** `src/features/matrix.py`

```python
# Core features extracted
FEATURE_NAMES = [
    # Basic User Stats
    "tweet_count", "z_score", "observed_post_count",
    "text_len_mean", "text_len_std", "text_len_p90",
    
    # Text Content
    "unique_text_ratio", "unique_token_ratio", "token_entropy",
    "punct_rate", "uppercase_rate", "digit_rate",
    "url_rate", "mention_rate", "hashtag_rate",
    
    # Temporal Patterns
    "gap_mean_seconds", "gap_std_seconds", "gap_cv",
    "burst_rate_2m", "burst_rate_10m", "hourly_entropy",
    "night_posting_rate", "cadence_autocorr_24h",
    
    # Behavioral
    "session_count", "session_mean_len", "template_repeat_rate",
    "mention_target_entropy", "url_domain_entropy",
    
    # + 384 E5 embedding features
]

# Build feature matrix
def build_feature_matrix(users, posts_by_author, embedding_model_name):
    user_ids = []
    features = []
    for user in users:
        uid = user["id"]
        user_ids.append(uid)
        
        posts = posts_by_author.get(uid, [])
        feat_row = _extract_user_features(user, posts)
        features.append(feat_row)
    
    return user_ids, np.array(features)

# Build sequence documents
def build_sequence_documents(users, posts_by_author):
    user_ids = []
    documents = []
    for user in users:
        uid = user["id"]
        user_ids.append(uid)
        
        # Combine profile + posts
        doc = f"{user.get('description', '')} _PROFILE_GROUP_MARK_ "
        doc += f"{user.get('name', '')} __profile_sep__ "
        doc += f"{user.get('location', '')} _PROFILE_END_MARK_"
        
        posts = posts_by_author.get(uid, [])
        for post in posts:
            text = post.get("text", "")
            text = re.sub(r"https?://\S+", "__url__", text)
            text = re.sub(r"@\w+", "__mention__", text)
            text = re.sub(r"#\w+", "__hashtag__", text)
            doc += f"\n{text} __post_sep__"
        
        documents.append(doc)
    
    return user_ids, documents
```

---

## Phase 2: Data Loading

**File:** `src/training/data.py`

```python
def load_training_rows(dataset_dir, languages=None):
    """Load training data from practice datasets."""
    from src.data.loader import load_dataset_bundle, discover_training_pairs
    
    rows = []
    for dataset_path, bots_path in discover_training_pairs(dataset_dir):
        bundle = load_dataset_bundle(dataset_path, bots_path)
        
        # Extract features
        posts_by_author = group_posts_by_author(bundle["posts"])
        user_ids, features = build_feature_matrix(
            bundle["users"], posts_by_author,
            embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        doc_user_ids, documents = build_sequence_documents(
            bundle["users"], posts_by_author
        )
        
        # Build labels
        bot_ids = set(bundle["bots"])
        labels = np.array([1 if uid in bot_ids else 0 for uid in user_ids])
        
        rows.append({
            "bundle": bundle,
            "dataset_id": bundle["dataset_id"],
            "features": features,
            "documents": documents,
            "labels": labels,
        })
    
    return rows
```

---

## Phase 3: Out-of-Fold Computation

**File:** `src/training/oof.py`

```python
def compute_oof_probabilities_component(
    features: np.ndarray,
    documents: list[str],
    labels: np.ndarray,
    folds: int,
    seed: int,
    kind: str,
    params: dict[str, Any],
) -> np.ndarray:
    """Compute OOF probabilities using k-fold CV."""
    usable_folds = _usable_folds(labels, folds)
    
    if usable_folds < 2:
        # Fallback if not enough samples
        fallback = build_component(kind=kind, seed=seed, params=params)
        fallback.fit(features, documents, labels)
        return fallback.predict_proba(features, documents)
    
    # K-fold stratified cross-validation
    splitter = StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(labels), dtype=float)
    
    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(features, labels)):
        # Train on fold
        model = build_component(kind=kind, seed=seed + fold_id, params=params)
        train_docs = [documents[idx] for idx in train_idx]
        valid_docs = [documents[idx] for idx in valid_idx]
        
        model.fit(features[train_idx], train_docs, labels[train_idx])
        
        # Predict on held-out fold
        oof[valid_idx] = model.predict_proba(features[valid_idx], valid_docs)
    
    return oof
```

---

## Phase 4: Component Blending

**File:** `src/training/main.py` (lines ~177-186)

```python
def _blend_component_probabilities(
    probabilities: list[np.ndarray], 
    weights: list[float]
) -> np.ndarray:
    """Blend component probabilities."""
    total_weight = float(sum(weights))
    blended = np.zeros_like(probabilities[0], dtype=float)
    
    for probs, weight in zip(probabilities, weights):
        blended += float(weight) * probs
    
    return blended / total_weight

# Usage in training
weights = [0.75, 0.05, 0.20]  # RF+ET, Tabular, LLM
seed_component_oof = []

for component_index, component in enumerate(resolved_components):
    component_seed = int(robust_seed + (component_index * 97))
    seed_component_oof.append(
        compute_oof_probabilities_component(
            all_features, all_documents, all_labels,
            folds=5, seed=component_seed,
            kind=str(component["kind"]), 
            params=dict(component["params"]),
        )
    )

full_blended_oof = _blend_component_probabilities(seed_component_oof, weights)
```

---

## Phase 5: Hard Negative Calibration

**File:** `src/models/calibration.py`

```python
class HardNegativeCalibrator:
    """Calibrate probabilities with hard negative weighting."""
    
    def __init__(self, hard_fraction=0.14, hard_weight=3.25, c=1.2, min_samples=24):
        self.hard_fraction = hard_fraction
        self.hard_weight = hard_weight
        self.c = c
        self.min_samples = min_samples
        self.classifier = None
        self.hard_negative_center = 0.5
        self.hard_positive_center = 0.5
    
    def fit(self, probabilities, labels):
        """Fit calibrator on probabilities."""
        # Identify hard negatives (false positives at high probability)
        negative_mask = labels == 0
        negative_probs = probabilities[negative_mask]
        
        # Sort and identify top hard_fraction
        sort_idx = np.argsort(-negative_probs)
        hard_count = max(self.min_samples, int(len(sort_idx) * self.hard_fraction))
        hard_idx = sort_idx[:hard_count]
        
        # Create sample weights
        sample_weight = np.ones(len(probabilities))
        sample_weight[negative_mask] = 1.0
        sample_weight[negative_mask][hard_idx] = self.hard_weight
        
        # Fit logistic regression with weights
        self.classifier = LogisticRegression(C=self.c, max_iter=1000)
        self.classifier.fit(probabilities.reshape(-1, 1), labels, sample_weight=sample_weight)

def cross_fit_hard_negative_calibration(
    probabilities, labels, folds, seed,
    hard_fraction, hard_weight, c, min_samples
):
    """Cross-fit calibration to avoid overfitting."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    calibrated = np.zeros_like(probabilities)
    
    for train_idx, valid_idx in splitter.split(probabilities, labels):
        calibrator = HardNegativeCalibrator(
            hard_fraction=hard_fraction,
            hard_weight=hard_weight,
            c=c,
            min_samples=min_samples,
        )
        calibrator.fit(probabilities[train_idx], labels[train_idx])
        calibrated[valid_idx] = calibrator.predict_proba(probabilities[valid_idx])
    
    return calibrated
```

---

## Phase 6: Threshold Selection

**File:** `src/models/threshold.py`

```python
def select_threshold_with_margin_grid_robust(
    probabilities_by_seed: list[np.ndarray],
    labels: np.ndarray,
    min_precision_grid: list[float],
    margin_grid: list[float],
    threshold_step: float,
    max_fp_rate: float = -1.0,
    start: float = 0.05,
    stop: float = 0.99,
):
    """Find optimal threshold via grid search."""
    all_candidates = []
    
    # Try all combinations
    thresholds = np.arange(start, stop + 1e-9, threshold_step)
    
    for min_precision in min_precision_grid:
        for margin in margin_grid:
            for threshold in thresholds:
                applied_threshold = threshold + margin
                if applied_threshold >= 1.0:
                    continue
                
                # Test on all seeds and take worst/mean
                scores_per_seed = []
                for probs in probabilities_by_seed:
                    preds = (probs >= applied_threshold).astype(int)
                    metrics = _binary_metrics(labels, preds)
                    
                    if metrics["precision"] < min_precision:
                        continue
                    
                    scores_per_seed.append(metrics["competition_score"])
                
                if not scores_per_seed:
                    continue
                
                candidate = {
                    "threshold": float(threshold),
                    "margin": float(margin),
                    "score": float(np.mean(scores_per_seed)),
                    "worst_score": float(np.min(scores_per_seed)),
                    "seed_count": len(scores_per_seed),
                }
                all_candidates.append(candidate)
    
    # Rank candidates
    best = max(all_candidates, 
               key=lambda c: (c["worst_score"], c["score"], c["seed_count"]))
    
    return best

# Usage
threshold_selection = select_threshold_with_margin_grid_robust(
    calibrated_stage2_oof_by_seed,  # 7 OOF arrays
    all_labels,
    min_precision_grid=[0.72],
    margin_grid=[0.0, 0.0025, 0.005, 0.01],
    threshold_step=0.0025,
)

locked_threshold = threshold_selection["threshold"]  # ≈ 0.2675
locked_margin = threshold_selection["margin"]        # ≈ 0.0
```

---

## Phase 7: Final Model Training

**File:** `src/training/main.py` (lines ~242-268)

```python
# Train final models on all data
logger.info("Training final models on all data...")

# Stage 1: Shortlist component
stage1_seed = int(args.seed + 791)
fitted_shortlist_component = build_component(
    kind=str(shortlist_component["kind"]),
    seed=stage1_seed,
    params=dict(shortlist_component["params"]),
)
fitted_shortlist_component.fit(all_features, all_documents, all_labels)

# Stage 2: Ensemble components
trained_components: list[dict[str, object]] = []
for component_index, component in enumerate(resolved_components):
    component_seed = int(args.seed + (component_index * 97))
    fitted_component = build_component(
        kind=str(component["kind"]),
        seed=component_seed,
        params=dict(component["params"]),
    )
    fitted_component.fit(all_features, all_documents, all_labels)
    
    trained_components.append({
        "model": str(component["model"]),
        "kind": str(component["kind"]),
        "weight": float(component["weight"]),
        "params": dict(component["params"]),
        "model_object": fitted_component,
    })
    
    logger.info(f"  Component trained: {component.get('model')} (weight: {component.get('weight')})")

logger.info(f"  Final threshold: {locked_threshold:.4f}, margin: {locked_margin:.4f}")
```

---

## Phase 8: Save Artifacts

**File:** `src/training/main.py` (lines ~300+)

```python
# Save the trained model
artifact = {
    "trained_components": trained_components,
    "fitted_shortlist_component": fitted_shortlist_component,
    "feature_names": FEATURE_NAMES,
    "feature_embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "threshold": locked_threshold,
    "margin": locked_margin,
    "cascade_enabled": cascade_enabled,
    "calibrator": stage2_calibrator,
    
    # Metadata
    "language": language,
    "training_mode": "best_config",
    "config_id": language_config["config_id"],
    "holdout_score": pooled_score,
    
    # Configuration
    "component_weights": {c["model"]: c["weight"] for c in resolved_components},
}

# Save to disk
save_pickle(model_path / f"{language}/model.pkl", artifact)

# Also save metrics
metrics = {
    "language": language,
    "competition_score": pooled_score,
    "precision": strict_outer_summary.get("pooled_precision"),
    "recall": strict_outer_summary.get("pooled_recall"),
    "f1": strict_outer_summary.get("pooled_f1"),
}
save_json(model_path / f"{language}/metrics.json", metrics)
```

---

## Component Implementations

### BotEnsembleComponent

**File:** `src/models/ensemble.py`

```python
@dataclass
class BotEnsembleModel:
    """Ensemble of RandomForest and ExtraTrees."""
    seed: int = 42
    threshold: float = 0.5
    rf_estimators: int = 900
    et_estimators: int = 1200
    min_samples_leaf: int = 2
    rf_bot_weight: float = 1.2
    et_bot_weight: float = 1.3
    
    def __post_init__(self) -> None:
        self.pipeline = self._build_pipeline(
            seed=self.seed,
            rf_estimators=self.rf_estimators,
            et_estimators=self.et_estimators,
            min_samples_leaf=self.min_samples_leaf,
            rf_bot_weight=self.rf_bot_weight,
            et_bot_weight=self.et_bot_weight,
            calibration_cv=3,
        )
    
    @staticmethod
    def _build_pipeline(seed, rf_estimators, et_estimators, min_samples_leaf,
                       rf_bot_weight, et_bot_weight, calibration_cv):
        rf = RandomForestClassifier(
            n_estimators=rf_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            class_weight={0: 1.0, 1: rf_bot_weight},
            n_jobs=-1,
        )
        et = ExtraTreesClassifier(
            n_estimators=et_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            class_weight={0: 1.0, 1: et_bot_weight},
            n_jobs=-1,
        )
        voter = VotingClassifier(
            estimators=[("rf", rf), ("et", et)],
            voting="soft",
            n_jobs=-1,
        )
        calibrated = CalibratedClassifierCV(
            estimator=voter,
            method="sigmoid",
            cv=calibration_cv,
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", calibrated),
            ]
        )
    
    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.pipeline.fit(features, labels)
        return self
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(features)[:, 1]
```

---

## Configuration

**File:** `src/utils/config.py`

```python
BEST_LANGUAGE_CONFIGS = {
    "en": {
        "config_id": "transfer_en_best_v5_llm_semantic_tuned",
        "selection_protocol": "source_only",
        "components": [
            {"model": "rfet_balanced", "weight": 0.75},
            {"model": "tab_lr_c1", "weight": 0.05},
            {"model": "llm_semantic_balanced", "weight": 0.20},
        ],
        "inference": {
            "mode": "static",
            "threshold": 0.2675,
            "margin": 0.0,
        },
        "selection": {
            "threshold_step": 0.0025,
            "min_precision_grid": [0.72],
            "robust_seed_count": 7,
            "robust_seed_stride": 53,
            "use_contrastive_calibration": True,
            "contrastive_calibration": {
                "hard_fraction": 0.14,
                "hard_weight": 3.25,
                "c": 1.2,
                "min_samples": 24,
            },
        },
    },
    "fr": {
        "config_id": "transfer_fr_best_v1",
        "components": [
            {"model": "rfet_recall", "weight": 0.15},
            {"model": "text_lr_recall", "weight": 0.85},
        ],
        "inference": {
            "mode": "static",
            "threshold": 0.20375,
            "margin": 0.0,
        },
        # ... similar selection config
    },
}
```

---

## Summary: Key Code Paths

```
Main Entry: train.py
  └─> src/training/main.py:main()
       ├─ train_language_model_best_config() for each language
       │  ├─ load_training_rows() [load data + extract features]
       │  ├─ evaluate_best_config_holdouts() [holdout evaluation]
       │  ├─ compute_oof_probabilities_component() × 7 seeds × 3 components
       │  │  └─ uses build_component() from factory
       │  ├─ blend via _blend_component_probabilities()
       │  ├─ cross_fit_hard_negative_calibration()
       │  ├─ select_threshold_with_margin_grid_robust()
       │  ├─ Final fit with build_component()
       │  └─ save_pickle() to output/models/{lang}/model.pkl
       └─ Repeat for next language

Prediction: predict.py
  └─> src/prediction/engine.py:run_prediction()
       ├─ load_pickle(model.pkl) [load trained artifact]
       ├─ build_feature_matrix() [extract features]
       ├─ _compute_probabilities() [blend components]
       ├─ threshold application [make predictions]
       └─ write_detection_file() [output results]
```

