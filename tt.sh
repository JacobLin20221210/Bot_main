#!/bin/sh
# Run trainX/testY holdout evaluation using current best config

set -e

DATASET_DIR="${DATASET_DIR:-dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-output/models}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-output/experiments}"
RUN_NAME="${RUN_NAME:-holdout-eval}"

echo "================================================================================"
echo "Running trainX/testY holdout evaluation..."
echo "================================================================================"
echo ""

# Run training
uv run python train.py \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --archive-root "$ARCHIVE_ROOT" \
    --run-name "$RUN_NAME" \
    --training-mode best_config

# Find the latest experiment directory
LATEST_EXP=$(ls -1t "$ARCHIVE_ROOT" | grep -E '^[0-9]{8}-[0-9]{6}-' | head -1)

if [ -z "$LATEST_EXP" ]; then
    echo "Error: Could not find experiment directory"
    exit 1
fi

EXP_DIR="$ARCHIVE_ROOT/$LATEST_EXP"
echo ""
echo "Run ID: $LATEST_EXP"
echo ""

# Display holdout scores
echo "================================================================================"
echo "HOLDOUT RESULTS"
echo "================================================================================"
echo ""

TOTAL_SCORE=0

for LANG in en fr; do
    LANG_DIR="$EXP_DIR/languages/$LANG/holdout"
    
    if [ ! -d "$LANG_DIR" ]; then
        continue
    fi
    
    LANG_UPPER=$(printf '%s' "$LANG" | tr '[:lower:]' '[:upper:]')
    echo "================================================================================"
    echo "Language: $LANG_UPPER"
    echo "================================================================================"
    printf "%-15s %8s %8s %8s %8s %4s %4s %4s\n" "Train->Test" "Score" "Prec" "Recall" "F1" "TP" "FP" "FN"
    echo "--------------------------------------------------------------------------------"
    
    LANG_TOTAL=0
    
    for METRICS_FILE in "$LANG_DIR"/*/metrics.json; do
        if [ ! -f "$METRICS_FILE" ]; then
            continue
        fi
        
        FOLD_NAME=$(basename "$(dirname "$METRICS_FILE")")
        
        # Parse metrics using Python
        METRICS=$(uv run python -c "
import json
import sys
with open('$METRICS_FILE') as f:
    d = json.load(f)
print(d.get('competition_score', 0), d.get('precision', 0), d.get('recall', 0), d.get('f1', 0), d.get('tp', 0), d.get('fp', 0), d.get('fn', 0))
")
        set -- $METRICS
        SCORE=${1:-0}
        PRECISION=${2:-0}
        RECALL=${3:-0}
        F1=${4:-0}
        TP=${5:-0}
        FP=${6:-0}
        FN=${7:-0}
        
        LANG_TOTAL=$(echo "$LANG_TOTAL + $SCORE" | bc)
        
        # Parse train/test from fold name
        TRAIN_TEST=$(echo "$FOLD_NAME" | sed 's/train_\([0-9]*\)__test_\([0-9]*\)/\1->\2/')
        
        printf "%-15s %8.1f %8.3f %8.3f %8.3f %4.0f %4.0f %4.0f\n" "$TRAIN_TEST" "$SCORE" "$PRECISION" "$RECALL" "$F1" "$TP" "$FP" "$FN"
    done
    
    echo "--------------------------------------------------------------------------------"
    printf "%-15s %8.1f\n" "Subtotal:" "$LANG_TOTAL"
    echo ""
    
    TOTAL_SCORE=$(echo "$TOTAL_SCORE + $LANG_TOTAL" | bc)
done

echo "================================================================================"
printf "TOTAL SCORE: %.1f\n" "$TOTAL_SCORE"
echo "================================================================================"
echo ""

# Show config info
MANIFEST_FILE="$EXP_DIR/run_manifest.json"
if [ -f "$MANIFEST_FILE" ]; then
    echo "Config Info:"
    uv run python -c "
import json
with open('$MANIFEST_FILE') as f:
    m = json.load(f)
print(f\"  Git commit: {m.get('git_commit', 'unknown')}\")
for lang_info in m.get('languages', []):
    lang = lang_info.get('language', '?')
    print(f\"  {lang.upper()} threshold: {lang_info.get('threshold', 'unknown')}\")
"
fi
