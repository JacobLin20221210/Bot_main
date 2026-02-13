# Training Algorithm - Quick Visual Summary

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          INPUT DATA                                   │
│  dataset.posts&users.{30,31,32,33}.json                             │
│  ├─ ~600 users per language                                         │
│  ├─ ~130 bots (labels: ground truth)                                │
│  ├─ ~1200 posts total                                               │
│  └─ Multilingual (English + French)                                 │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
                    ┌────────────────▼─────────────────┐
                    │   FEATURE EXTRACTION             │
                    │   (src/features/matrix.py)       │
                    │                                  │
                    │  70 Engineered Features:         │
                    │  ├─ User Stats (4)               │
                    │  ├─ Text Features (20)           │
                    │  ├─ Temporal Features (13)       │
                    │  ├─ Behavioral Patterns (15)     │
                    │  └─ Embedding Features (384)     │
                    │                                  │
                    │  Total: 454 features per user    │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼──────────────────┐
                    │    DOCUMENT SEQUENCING            │
                    │  (src/features/matrix.py)         │
                    │                                   │
                    │   Profile + Timeline              │
                    │   with special markers:           │
                    │   __mention__, __url__, __hashtag__
                    └────────────────┬──────────────────┘
                                     │
                ┌────────────────────┴───────────────────┐
                │                                         │
    ┌───────────▼──────────────┐         ┌───────────────▼─────────┐
    │   HOLDOUT EVALUATION     │         │  OOF COMPUTATION (MAIN) │
    │   (source-only)          │         │  (src/training/oof.py)  │
    │                          │         │                         │
    │ Train(30,32)→Test(31,33) │         │ For each Seed {1-7}:    │
    │ Train(31,33)→Test(30,32) │         │   For each Component:   │
    │ Train(30)→Test(32)       │         │     5-fold CV:          │
    │ Train(32)→Test(30)       │         │       fit: train fold   │
    │ Train(31)→Test(33)       │         │       pred: valid fold  │
    │ Train(33)→Test(31)       │         │                         │
    │                          │         │   Blend: 0.75*RF +      │
    │ ✅ Ground truth scores   │         │           0.05*TAB +    │
    │                          │         │           0.20*LLM      │
    └───────────┬──────────────┘         └──────────────┬──────────┘
                │                                       │
                │                   ┌───────────────────┘
                │                   │
                │        ┌──────────▼────────────┐
                │        │  HARD NEGATIVE        │
                │        │  CALIBRATION          │
                │        │                       │
                │        │  Identify hard        │
                │        │  negatives (FP)       │
                │        │  Weight: 3.25x        │
                │        │  Cross-fit CV         │
                │        │                       │
                │        │  Output: Calibrated   │
                │        │  OOF probabilities    │
                │        └──────────┬────────────┘
                │                   │
                │        ┌──────────▼────────────┐
                │        │  THRESHOLD SELECTION  │
                │        │  (src/models/        │
                │        │   threshold.py)       │
                │        │                       │
                │        │  Grid Search:         │
                │        │  threshold: [0.05...0.99]
                │        │  margin: [0, 0.0025, ...]
                │        │  min_prec: [0.72]    │
                │        │                       │
                │        │  Metric:              │
                │        │  score = 4*TP -      │
                │        │          2*FP - FN   │
                │        │                       │
                │        │  Result:              │
                │        │  threshold ≈ 0.2675  │
                │        │  margin = 0.0        │
                │        └──────────┬────────────┘
                │                   │
                │                   │
                └───────────┬───────┘
                            │
                ┌───────────▼──────────────────────┐
                │  FINAL MODEL TRAINING            │
                │  (on all data, no CV)            │
                │                                  │
                │  ├─ Fit RF+ET on all data       │
                │  ├─ Fit Tabular LR on all data  │
                │  ├─ Fit LLM Semantic on all     │
                │  └─ Store models + threshold    │
                └───────────┬──────────────────────┘
                            │
                ┌───────────▼──────────────────────┐
                │  SAVE ARTIFACTS                  │
                │  output/models/en/               │
                │                                  │
                │  ├─ model.pkl (trained models)  │
                │  ├─ metrics.json (holdout)      │
                │  └─ run_manifest.json           │
                └─────────────────────────────────┘
```

---

## Component Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      FINAL PREDICTION                           │
│             (weighted ensemble of 3 components)                 │
│                                                                 │
│  prob = 0.75 * rf_et_prob +                                    │
│          0.05 * tabular_prob +                                 │
│          0.20 * llm_semantic_prob                              │
│                                                                 │
│  final_decision = prob >= 0.2675  (threshold)                  │
└────────────────────────────────────────────────────────────────┘
        ▲                     ▲                ▲
        │                     │                │
   ┌────┴──────┐         ┌────┴──────┐    ┌───┴──────┐
   │ COMPONENT  │         │ COMPONENT  │    │COMPONENT │
   │    1       │         │     2      │    │    3     │
   │ (0.75)     │         │  (0.05)    │    │ (0.20)   │
   └────┬───────┘         └────┬───────┘    └───┬──────┘
        │                      │                │
        │         ┌────────────┼────────────┐   │
        │         │            │            │   │
    ┌───▼──┐  ┌───▼──┐  ┌──────▼──┐   ┌───▼──┐
    │RF    │  │ET    │  │  LR     │   │E5    │
    │900   │  │1200  │  │ Tabular │   │Embed │
    │trees │  │trees │  │         │   │-ding │
    │      │  │      │  │Logistic │   │      │
    └───┬──┘  └───┬──┘  └──────┬──┘   └───┬──┘
        │         │            │          │
        │    ┌────┴────┐       │          │
        │    │VotingCLS│       │          │
        │    │(soft avg)│       │          │
        │    └────┬────┘       │          │
        │         │            │          │
        └────┬────┴────────┬───┴──────────┘
             │             │
          ┌──┴─────────────┴──┐
          │ Calibrated on    │
          │ hard negatives   │
          │ (3.25x weight)   │
          └──────┬───────────┘
                 │
            OOF Probs × 7 seeds
```

---

## Training Pipeline Phases

```
PHASE 1: FEATURE ENGINEERING
  Input: Posts & Users JSON
  Process: Extract 70 features + 384 embeddings
  Output: Feature matrix (600 × 454)
  Time: ~10-15 minutes

PHASE 2: HOLDOUT EVALUATION [REFERENCE BENCHMARK]
  Input: Feature matrix + labels
  Process: Leave-one-dataset-out CV (6 configurations)
  Output: Holdout scores (truth for model performance)
  Time: ~30 minutes
  Metrics: This is our ground truth evaluation

PHASE 3: OUT-OF-FOLD COMPUTATION [MAIN TRAINING]
  Input: Feature matrix + labels
  Process:
    ├─ 7 random seeds (robustness)
    ├─ 5-fold stratified CV per seed
    ├─ Train 3 components per fold
    ├─ Blend predictions
    └─ Collect OOF array
  Output: 7 × OOF arrays (each 600-element probability vector)
  Time: ~45 minutes
  Result: OOF[i] = predicted bot probability for sample i

PHASE 4: CALIBRATION
  Input: 7 × OOF arrays
  Process: Hard negative calibration (cross-fit)
  Output: Calibrated OOF arrays
  Time: ~15 minutes
  Goal: Reduce false positives while maintaining recall

PHASE 5: THRESHOLD TUNING
  Input: Calibrated OOF + labels
  Process: Grid search (250+ threshold combinations)
  Output: Optimal threshold (0.2675)
  Time: ~5 minutes
  Metric: Maximize 4*TP - 2*FP - FN

PHASE 6: FINAL TRAINING
  Input: All data (no CV)
  Process: Train 3 components on full dataset
  Output: Production models
  Time: ~10 minutes
  Result: Models saved to disk

TOTAL TIME: ~1-2 hours per language
```

---

## Key Configuration Parameters

```
# From src/utils/config.py

COMPONENT WEIGHTS:
├─ rfet_balanced:       0.75  (RandomForest + ExtraTrees)
├─ tab_lr_c1:          0.05  (Tabular Logistic Regression)
└─ llm_semantic_balanced: 0.20 (LLM Semantic embeddings)

CROSS-VALIDATION:
├─ folds: 5 (k-fold stratified)
├─ seed: 42 (base random seed)
├─ robust_seed_count: 7 (multiple seeds for robustness)
└─ robust_seed_stride: 53 (spacing between seeds)

THRESHOLD SELECTION:
├─ min_precision_grid: [0.72] (enforce minimum precision)
├─ margin_grid: [0.0, 0.0025, 0.005, 0.01] (adjustment margins)
├─ threshold_step: 0.0025 (fine-grained grid search)
└─ max_fp_rate: -1.0 (no constraint on FP rate)

HARD NEGATIVE CALIBRATION:
├─ enabled: True
├─ hard_fraction: 0.14 (weight top 14% negatives)
├─ hard_weight: 3.25 (3.25x weight for hard negatives)
├─ c: 1.2 (L2 regularization strength)
└─ min_samples: 24 (minimum samples required)

INFERENCE:
├─ mode: "static" (not regime-based)
├─ threshold: 0.2675 (selected via grid search)
└─ margin: 0.0 (no adjustment)
```

---

## Ensemble Component Details

### Component 1: RandomForest + ExtraTrees (75%)
```
PURPOSE: Capture tabular feature patterns
├─ RandomForestClassifier (900 trees)
│  ├─ 70 engineered features
│  ├─ Bootstrap sampling
│  ├─ Random feature search
│  ├─ Bot weight: 1.2 (class imbalance)
│  └─ min_samples_leaf: 2 (shallow trees)
│
├─ ExtraTreesClassifier (1200 trees)
│  ├─ Same features
│  ├─ No bootstrap
│  ├─ Random split thresholds
│  ├─ Bot weight: 1.3
│  └─ min_samples_leaf: 2
│
├─ VotingClassifier
│  ├─ Method: soft voting (average probabilities)
│  └─ Combines RF + ET
│
└─ CalibratedClassifierCV
   ├─ Method: sigmoid calibration
   ├─ CV folds: 3
   └─ Goal: Well-calibrated probabilities

STRENGTH: Captures non-linear patterns in behavioral features
```

### Component 2: Tabular Logistic Regression (5%)
```
PURPOSE: Linear approximation on engineered features
├─ sklearn.linear_model.LogisticRegression
├─ Features: 70 engineered (temporal, behavioral, user stats)
├─ Regularization C: 1.0
├─ Bot weight: 1.2
├─ max_iter: 1600 (convergence)
└─ Solver: lbfgs (good for small datasets)

STRENGTH: Interpretable, captures linear relationships
```

### Component 3: LLM Semantic (20%)
```
PURPOSE: Semantic understanding of account behavior
├─ E5 Embeddings (multilingual-large-instruct)
├─ 384-dim embeddings per document
├─ Input: Profile + Timeline text
│  └─ Marked with __mention__, __url__, __hashtag__
│
├─ Prompted view: Add signal summary
│  ├─ URL count
│  ├─ Mention count
│  ├─ Hashtag count
│  ├─ Post count
│  └─ Character count
│
├─ LinearComponent (Logistic Regression)
│  ├─ C: 1.1 (regularization)
│  ├─ Bot weight: 1.2
│  └─ max_iter: 1600
│
└─ Caching: Embeddings cached to disk (~150 MB)

STRENGTH: Captures semantic patterns (promotional templates, etc.)
```

---

## Probability Calibration Deep-Dive

```
BEFORE CALIBRATION:
├─ Many false positives with high probability (~0.8+)
├─ Poorly calibrated predictions
└─ Gap between predicted and actual probability

HARD NEGATIVE SELECTION:
├─ Identify negative samples (label=0) with high bot probability
├─ Sort by probability descending
├─ Select top 14% (hard_fraction=0.14)
└─ These are the FP risk samples

WEIGHTED TRAINING:
├─ Standard samples: weight = 1.0
├─ Hard negative samples: weight = 3.25
├─ Fit logistic regression with weights
└─ Loss = mean(weight * cross_entropy)

EFFECT:
├─ Model learns to be more conservative
├─ Predictions for hard negatives get pushed down
├─ False positive rate decreases
└─ Recall stays high (true bots still detected)

CROSS-FIT PROCEDURE:
├─ Split data into k folds
├─ For each fold:
│  ├─ Train calibrator on k-1 folds
│  ├─ Apply to held-out fold
│  └─ Avoid overfitting to specific hard negatives
└─ Stack predictions from all folds
```

---

## Performance Metrics Tracking

```
DURING TRAINING:
├─ OOF-based metrics (estimated performance)
│  ├─ Precision: TP / (TP + FP)
│  ├─ Recall: TP / (TP + FN)
│  ├─ Competition Score: 4*TP - 2*FP - FN
│  └─ F1 Score: 2 * (Prec * Rec) / (Prec + Rec)
│
└─ Reported per fold, per seed, and averaged

AFTER TRAINING:
├─ Holdout Metrics (independent validation)
│  ├─ Train on (30,32) → Test on (31,33)
│  ├─ Train on (31,33) → Test on (30,32)
│  ├─ Train on (30) → Test on (32)
│  ├─ Train on (32) → Test on (30)
│  ├─ Train on (31) → Test on (33)
│  └─ Train on (33) → Test on (31)
│
└─ Reports both per-fold and pooled metrics

FINAL REPORTED SCORE:
├─ Holdout competition score: 730/1000
├─ Precision: 98.4%
├─ Recall: 100%
├─ False Positives: 3 total
└─ False Negatives: 0 total
```

---

## Summary: Why This Algorithm Works

✅ **Ensemble Diversity**
- Trees capture non-linear patterns
- Linear models capture global trends  
- Embeddings capture semantic patterns

✅ **Robust Validation**
- 5-fold CV: reduces variance from specific splits
- 7 random seeds: reduces variance from initialization
- Holdout evaluation: independent truth test

✅ **Smart Calibration**
- Hard negatives get special attention
- Reduces false positives (cost = -2 each)
- Maintains recall (cost = -1 per false negative)

✅ **Direct Optimization**
- Threshold tuned on exact competition metric
- 250+ combinations tested
- Best one selected based on competition scoring

✅ **Configuration-Driven**
- Easy to experiment
- No code changes needed
- Reproducibility built-in
- Version control friendly

**Result: 730/1000 score (98.4% precision, 100% recall)**
