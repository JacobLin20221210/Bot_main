import sys
from pathlib import Path

def read_ids(path):
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines

datasets = [30,31,32,33]
total_TP = total_FP = total_FN = 0

for d in datasets:
    pred_path = Path('output')/f'detections.{d}.txt'
    truth_path = Path('dataset')/f'dataset.bots.{d}.txt'
    preds = read_ids(pred_path)
    truth = read_ids(truth_path)
    if preds is None:
        print(f'MISSING prediction file: {pred_path}')
        continue
    if truth is None:
        print(f'MISSING ground-truth file: {truth_path}')
        continue
    pred_set = set(preds)
    truth_set = set(truth)
    TP = len(pred_set & truth_set)
    FP = len(pred_set - truth_set)
    FN = len(truth_set - pred_set)
    score = 4*TP - 2*FP - FN
    total_TP += TP; total_FP += FP; total_FN += FN
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    rec = TP/(TP+FN) if (TP+FN)>0 else 0.0
    print(f'Dataset {d}: TP={TP}, FP={FP}, FN={FN}, Score={score}, Prec={prec:.3f}, Rec={rec:.3f}, preds={len(preds)}, truth={len(truth)}')

total_score = 4*total_TP - 2*total_FP - total_FN
prec_all = total_TP/(total_TP+total_FP) if (total_TP+total_FP)>0 else 0.0
rec_all = total_TP/(total_TP+total_FN) if (total_TP+total_FN)>0 else 0.0
print('\nTOTAL: TP={0}, FP={1}, FN={2}, Score={3}, Prec={4:.3f}, Rec={5:.3f}'.format(total_TP,total_FP,total_FN,total_score,prec_all,rec_all))

# Additional checks
print('\nAdditional checks per prediction file: duplicates and non-numeric lines (first 5 examples)')
for d in datasets:
    p = Path('output')/f'detections.{d}.txt'
    if not p.exists():
        continue
    lines = read_ids(p)
    dup = len(lines) - len(set(lines))
    nonnum = [x for x in lines if not x.isdigit()]
    print(f'{p.name}: lines={len(lines)}, duplicates={dup}, examples_non_numeric={nonnum[:5]}')

sys.exit(0)
