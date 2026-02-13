#!/usr/bin/env python3
"""Calculate competition scores from predictions and ground truth."""

from pathlib import Path


def calculate_scores():
    """Calculate competition scores for all datasets."""
    
    datasets = [
        ("dataset/dataset.bots.30.txt", "output/detections.30.txt", "30 (EN)"),
        ("dataset/dataset.bots.31.txt", "output/detections.31.txt", "31 (FR)"),
        ("dataset/dataset.bots.32.txt", "output/detections.32.txt", "32 (EN)"),
        ("dataset/dataset.bots.33.txt", "output/detections.33.txt", "33 (FR)"),
    ]
    
    total_score = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    print("=" * 70)
    print("Competition Score Results")
    print("=" * 70)
    print()
    
    for ground_truth_path, predictions_path, label in datasets:
        # Load ground truth
        gt_path = Path(ground_truth_path)
        if not gt_path.exists():
            print(f"Ground truth file not found: {ground_truth_path}")
            continue
        
        ground_truth = set(line.strip() for line in gt_path.read_text().strip().split('\n') if line.strip())
        
        # Load predictions
        pred_path = Path(predictions_path)
        if not pred_path.exists():
            print(f"Predictions file not found: {predictions_path}")
            continue
        
        predictions = set(line.strip() for line in pred_path.read_text().strip().split('\n') if line.strip())
        
        # Calculate metrics
        tp = len(ground_truth & predictions)  # True positives
        fp = len(predictions - ground_truth)   # False positives
        fn = len(ground_truth - predictions)   # False negatives
        
        score = (4 * tp) - (2 * fp) - fn
        
        print(f"Dataset {label}:")
        print(f"  Ground Truth Bots: {len(ground_truth)}")
        print(f"  Detected Bots: {len(predictions)}")
        print(f"  True Positives (TP): {tp}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  Precision: {tp / max(1, tp + fp):.4f}")
        print(f"  Recall: {tp / max(1, tp + fn):.4f}")
        print(f"  Score: {score} (formula: 4*TP - 2*FP - FN = 4*{tp} - 2*{fp} - {fn})")
        print()
        
        total_score += score
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    print("=" * 70)
    print(f"TOTAL SCORE: {total_score}")
    print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}")
    print(f"Overall Precision: {total_tp / max(1, total_tp + total_fp):.4f}")
    print(f"Overall Recall: {total_tp / max(1, total_tp + total_fn):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    calculate_scores()
