"""Fit post-hoc calibration on calibration-split predictions, report test metrics before and after (PLAN.md M3).

    python scripts/zero_shot_eval.py --split calibration --dump-preds calib.json ...   # held-out splits
    python scripts/zero_shot_eval.py --dump-preds test.json ...                          # test splits
    python scripts/calibrate_eval.py calib.json test.json [--per-dataset]

Default: one fit per (type, K-bucket) pooled over datasets, the deployable setting where an unseen task gets its
type's calibration. --per-dataset fits each dataset separately (an upper bound: needs labelled data per task).
"""

import argparse
import json

import numpy as np

from system_one.calibrate import apply, fit
from system_one.metrics import binary_metrics, categorical_metrics, ordinal_metrics

METRICS = {"noul": binary_metrics, "choice": categorical_metrics, "score": ordinal_metrics}


def metrics(rows, params=None):
    kind = rows[0]["type"]
    probs = [apply(params, kind, r["probs"]) if params else np.asarray(r["probs"]) for r in rows]
    gold = [r["gold"] for r in rows]
    return METRICS[kind]([p[1] for p in probs] if kind == "noul" else probs, gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration")
    ap.add_argument("test")
    ap.add_argument("--per-dataset", action="store_true")
    ap.add_argument("--json", help="write fitted parameters here")
    args = ap.parse_args()
    calib, test = json.load(open(args.calibration)), json.load(open(args.test))
    scorers = sorted({s for d in test.values() for s in d})
    fitted = {}
    print("| dataset | scorer | acc | NLL | ECE-15 | F1 / MAE | → calibrated acc | NLL | ECE-15 | F1 / MAE |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for scorer in scorers:
        pooled = fit([r for d in calib.values() for r in d.get(scorer, [])])
        fitted[scorer] = pooled
        for name, by_scorer in test.items():
            if scorer not in by_scorer:
                continue
            params = fit(calib[name][scorer]) if args.per_dataset else pooled
            before, after = metrics(by_scorer[scorer]), metrics(by_scorer[scorer], params)
            extra = lambda m: f"{m['f1']:.3f}" if "f1" in m and name in ("go_emotions",) else f"{m['mae']:.3f}" if "mae" in m else "–"
            print(f"| {name} | {scorer} | {before['acc']:.3f} | {before['nll']:.3f} | {before['ece15']:.3f} | {extra(before)} "
                  f"| {after['acc']:.3f} | {after['nll']:.3f} | {after['ece15']:.3f} | {extra(after)} |")
    print("\nfitted (pooled per type / K-bucket):", json.dumps(fitted, indent=1))
    if args.json:
        json.dump(fitted, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
