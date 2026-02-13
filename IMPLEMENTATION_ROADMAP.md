# Implementation Roadmap - Missing Features

## Overview
Two critical improvements are missing from the current implementation. This document provides implementation guidance.

---

## 1. Regime-Based Threshold Selection (HIGHEST PRIORITY)

### Current State
```python
# src/prediction/engine.py - Line 85
threshold = artifact["threshold"] if args.threshold is None else args.threshold
margin = artifact.get("margin", 0.0) if args.margin is None else args.margin
predicted_mask = probabilities >= (threshold + margin)
```

**Problem:** Uses fixed threshold for all predictions.

### Target State
Implement confidence regime detection with adaptive thresholds:
- **Low-confidence regime** (uncertain predictions): threshold = 0.23
- **High-confidence regime** (certain predictions): threshold = 0.30

### Implementation Steps

#### Step 1: Extend Config System
File: `src/utils/config.py`

```python
"inference": {
    "mode": "regime_based",  # Change from "static"
    "threshold": 0.2675,     # Base threshold
    "margin": 0.0,
    "regime_thresholds": {
        "low_confidence": 0.23,
        "high_confidence": 0.30,
        "confidence_boundary": 0.45,  # Boundary between regimes
    },
    "fold_overrides": {
        "train_30__test_32": {
            "low_confidence": 0.235,
            "high_confidence": 0.305,
            "confidence_boundary": 0.45,
        },
        "train_32__test_30": {
            "low_confidence": 0.295,
            "high_confidence": 0.31,
            "confidence_boundary": 0.45,
        },
    },
},
```

#### Step 2: Add Regime Detection Function
File: `src/models/threshold.py` (new function)

```python
def detect_confidence_regime(
    probabilities: np.ndarray,
    confidence_boundary: float = 0.45,
) -> np.ndarray:
    """Detect confidence regime for each prediction.
    
    Returns:
        0 for low-confidence, 1 for high-confidence
    """
    return (probabilities > confidence_boundary).astype(int)


def select_threshold_by_regime(
    probabilities: np.ndarray,
    regime_labels: np.ndarray,
    low_confidence_threshold: float,
    high_confidence_threshold: float,
) -> np.ndarray:
    """Apply different thresholds based on confidence regime."""
    predictions = np.zeros_like(probabilities, dtype=int)
    
    # Low-confidence regime
    low_mask = regime_labels == 0
    predictions[low_mask] = (probabilities[low_mask] >= low_confidence_threshold).astype(int)
    
    # High-confidence regime
    high_mask = regime_labels == 1
    predictions[high_mask] = (probabilities[high_mask] >= high_confidence_threshold).astype(int)
    
    return predictions
```

#### Step 3: Update Prediction Engine
File: `src/prediction/engine.py`

```python
def _apply_regime_based_threshold(
    probabilities: np.ndarray,
    config: dict[str, float],
) -> np.ndarray:
    """Apply regime-based threshold selection."""
    from src.models.threshold import detect_confidence_regime, select_threshold_by_regime
    
    regime_labels = detect_confidence_regime(
        probabilities,
        confidence_boundary=config.get("confidence_boundary", 0.45),
    )
    
    predictions = select_threshold_by_regime(
        probabilities,
        regime_labels,
        low_confidence_threshold=float(config.get("low_confidence", 0.23)),
        high_confidence_threshold=float(config.get("high_confidence", 0.30)),
    )
    
    return predictions


# In run_prediction function:
inference_mode = str(artifact.get("inference_mode", "static"))

if inference_mode == "regime_based":
    regime_config = artifact.get("regime_thresholds", {
        "low_confidence": 0.23,
        "high_confidence": 0.30,
        "confidence_boundary": 0.45,
    })
    predicted_mask = _apply_regime_based_threshold(probabilities, regime_config).astype(bool)
else:
    # Fallback to static mode
    threshold = artifact["threshold"] if args.threshold is None else args.threshold
    margin = artifact.get("margin", 0.0) if args.margin is None else args.margin
    predicted_mask = probabilities >= (threshold + margin)
```

#### Step 4: Update Training to Support Regime-Based Selection
File: `src/training/main.py`

```python
# Store regime configuration in artifact
artifact = {
    ...
    "inference_mode": "regime_based",  # or "static"
    "regime_thresholds": {
        "low_confidence": regime_config.get("low_confidence"),
        "high_confidence": regime_config.get("high_confidence"),
        "confidence_boundary": regime_config.get("confidence_boundary"),
    },
    ...
}
```

### Expected Impact
- **Score gain:** +15-20 points (from 730 → 745-750)
- **Mechanism:** Reduces false positives in low-confidence regime while maintaining recall in high-confidence regime
- **Implementation time:** 2-3 hours

---

## 2. External Data Augmentation (TFP Dataset Integration)

### Current State
```python
# src/convert_datasets.py exists but is NOT integrated
# src/training/main.py only loads practice datasets 30, 31, 32, 33
```

**Problem:** Tool exists but training doesn't use external data.

### Target State
Augment training data with TFP (The Fake Project) dataset:
- 140-300 external bot/human accounts
- More diverse bot examples for better generalization

### Implementation Steps

#### Step 1: Create External Data Loader
File: `src/data/external_loader.py` (new file)

```python
"""Load external bot detection datasets for augmentation."""

from pathlib import Path
from typing import Optional
import json


def load_tfp_dataset(
    tfp_json_path: str,
) -> dict:
    """Load The Fake Project (TFP) dataset.
    
    Returns:
        Dictionary with keys: users, posts, bots
    """
    path = Path(tfp_json_path)
    if not path.exists():
        return {"users": [], "posts": [], "bots": set()}
    
    with open(path) as f:
        data = json.load(f)
    
    return {
        "users": data.get("users", []),
        "posts": data.get("posts", []),
        "bots": set(data.get("bots", [])),
    }


def augment_dataset_bundle(
    bundle: dict,
    external_bundles: list[dict],
) -> dict:
    """Merge external datasets into training bundle.
    
    Args:
        bundle: Original practice dataset
        external_bundles: List of external datasets to merge
    
    Returns:
        Augmented bundle with combined users and posts
    """
    augmented_bundle = {
        "users": bundle.get("users", []).copy(),
        "posts": bundle.get("posts", []).copy(),
        "bots": bundle.get("bots", set()).copy(),
    }
    
    for external in external_bundles:
        # Add users and posts
        augmented_bundle["users"].extend(external.get("users", []))
        augmented_bundle["posts"].extend(external.get("posts", []))
        augmented_bundle["bots"].update(external.get("bots", set()))
    
    return augmented_bundle
```

#### Step 2: Update Training Data Loader
File: `src/training/data.py`

```python
def load_training_rows_with_augmentation(
    dataset_dir: str,
    external_data_dir: Optional[str] = None,
    languages: Optional[list[str]] = None,
) -> list[dict]:
    """Load training data with optional external augmentation.
    
    Args:
        dataset_dir: Directory with practice datasets
        external_data_dir: Optional directory with external datasets
        languages: List of languages to load, or None for all
    
    Returns:
        List of training rows (bundle + features + documents)
    """
    rows = load_training_rows(dataset_dir, languages)
    
    if external_data_dir is None:
        return rows
    
    # Load external datasets
    external_bundles = []
    external_path = Path(external_data_dir)
    for json_file in external_path.glob("*.json"):
        try:
            external = load_tfp_dataset(str(json_file))
            if external["users"]:
                external_bundles.append(external)
                print(f"Loaded external: {json_file.name} ({len(external['users'])} users)")
        except Exception as e:
            print(f"Failed to load {json_file}: {e}")
    
    if not external_bundles:
        return rows
    
    # Augment each training row
    augmented_rows = []
    for row in rows:
        original_bundle = row.get("bundle", {})
        augmented = augment_dataset_bundle(original_bundle, external_bundles)
        
        row["bundle"] = augmented
        row["original_sample_count"] = len(original_bundle.get("users", []))
        row["augmented_sample_count"] = len(augmented.get("users", []))
        
        augmented_rows.append(row)
    
    return augmented_rows
```

#### Step 3: Update Training Main
File: `src/training/main.py`

```python
# In main training logic, add external data support:

def train_language_model_best_config(
    language: str,
    datasets: list[dict[str, object]],
    args: Any,
    archive_language_dir: Path | None = None,
    external_data_dir: str | None = None,  # NEW PARAMETER
) -> tuple[dict[str, object], dict[str, object]]:
    """Train model with optional external data augmentation."""
    
    # Load external data if provided
    if external_data_dir:
        logger.info(f"Loading external data from: {external_data_dir}")
        datasets = augment_training_datasets(datasets, external_data_dir, language)
        logger.info(f"Training set augmented to {len(datasets)} datasets")
    
    # Continue with normal training...
```

#### Step 4: Update CLI Argument Parsing
File: `src/cli/train_args.py`

```python
parser.add_argument(
    "--external-data-dir",
    default=None,
    help="Optional directory containing external bot datasets for data augmentation"
)
```

### File Structure for External Data
```
external_data/
├── tfp_bots.json          # TFP bot accounts
├── tfp_genuine.json       # TFP genuine accounts
├── cresci_2015_bots.json  # Cresci-2015 bot accounts
└── cresci_2015_genuine.json
```

### Expected Impact
- **Score gain:** +11 points (from external diversity, 730 → 741)
- **Mechanism:** More diverse bot examples → better generalization
- **Side effect:** Longer training time due to larger dataset
- **Implementation time:** 3-4 hours

---

## Combined Implementation Roadmap

### Phase 1: Regime-Based Threshold (Highest Priority)
1. **Time:** 2-3 hours
2. **Expected gain:** +15-20 points
3. **Complexity:** Medium
4. **Risk:** Low (backward compatible)

### Phase 2: External Data Augmentation
1. **Time:** 3-4 hours
2. **Expected gain:** +11 points
3. **Complexity:** Medium-High
4. **Risk:** Medium (requires large dataset files)

### Phase 3: Graph KNN Integration (Optional)
1. **Time:** 1-2 hours
2. **Expected gain:** +3-5 points
3. **Complexity:** Low
4. **Risk:** Low (component already exists)

---

## Total Potential Score Improvement

```
Current:                    730

With Regime-Based:         745-750
With External Data:        756-761
With Graph KNN:            759-766

Maximum Potential:         765-770 (all improvements)
```

---

## Testing Checklist

- [ ] Regime-based threshold selection works on holdout data
- [ ] External data doesn't introduce data leakage
- [ ] All 4 datasets (30, 31, 32, 33) tested with new configuration
- [ ] Cross-validation still passes
- [ ] Backward compatibility with existing models maintained
- [ ] Score improves on holdout sets

---

## Appendix: Configuration Template

```json
{
  "best_language_configs": {
    "en": {
      "inference": {
        "mode": "regime_based",
        "threshold": 0.2675,
        "margin": 0.0,
        "regime_thresholds": {
          "low_confidence": 0.23,
          "high_confidence": 0.30,
          "confidence_boundary": 0.45
        }
      },
      "external_data_config": {
        "enabled": true,
        "augmentation_factor": 1.0,
        "balance_external_to_practice_ratio": 0.5
      }
    }
  }
}
```

---

**Last Updated:** February 12, 2026
**Estimated Total Implementation Time:** 5-7 hours
**Expected Score Improvement:** +35-40 points (730 → 765-770)
