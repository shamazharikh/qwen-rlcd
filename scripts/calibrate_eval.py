"""Fit post-hoc calibration on calibration-split predictions, report test metrics before and after (PLAN.md M3).

    python scripts/zero_shot_eval.py --split calibration --dump-preds calib.json ...   # held-out splits
    python scripts/zero_shot_eval.py --dump-preds test.json ...                          # test splits
    python scripts/calibrate_eval.py calib.json test.json [--per-dataset]

Default: one fit per (type, K-bucket) pooled over datasets with each dataset weighted equally, the deployable
setting where an unseen task gets its type's calibration. --per-dataset fits each dataset separately (needs
labelled data per task). For yes/no datasets the table also reports F1 at the threshold that maximises F1 on the
calibration split ("tuned F1"), since a 0.5 cut is wrong for rare labels.
"""

import argparse
import json

import numpy as np

from system_one.calibrate import apply, best_f1_threshold, fit
from system_one.metrics import binary_metrics, categorical_metrics, ordinal_metrics

METRICS = {"noul": binary_metrics, "choice": categorical_metrics, "score": ordinal_metrics}


def calibrated(rows, params=None):
    kind = rows[0]["type"]
    return [apply(params, kind, r["probs"]) if params else np.asarray(r["probs"]) for r in rows]


def metrics(rows, params=None, calib_rows=None):
    kind, probs, gold = rows[0]["type"], calibrated(rows, params), [r["gold"] for r in rows]
    out = METRICS[kind]([p[1] for p in probs] if kind == "noul" else probs, gold)
    if kind == "noul" and calib_rows:  # F1 at the threshold tuned on the calibration split
        t = best_f1_threshold([p[1] for p in calibrated(calib_rows, params)], [r["gold"] for r in calib_rows])
        pred, g = np.array([p[1] for p in probs]) >= t, np.array(gold) == 1
        out["tuned_f1"], out["threshold"] = 2 * (pred & g).sum() / max(pred.sum() + g.sum(), 1), t
    return out


def weighted(rows):
    """Weight each row by 1 / rows in its dataset, so every dataset counts equally in a pooled fit."""
    return [{**r, "weight": 1.0 / len(rows)} for r in rows]


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
    print("| dataset | scorer | acc | NLL | ECE-15 | F1 / MAE | → calibrated acc | NLL | ECE-15 | F1 / MAE | tuned F1 (raw → calibrated) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for scorer in scorers:
        pooled = fit([r for d in calib.values() for r in weighted(d.get(scorer, []))])
        fitted[scorer] = pooled
        for name, by_scorer in test.items():
            if scorer not in by_scorer:
                continue
            params = fit(calib[name][scorer]) if args.per_dataset else pooled
            cal_rows = calib[name][scorer]
            before, after = metrics(by_scorer[scorer], None, cal_rows), metrics(by_scorer[scorer], params, cal_rows)
            extra = lambda m: f"{m['f1']:.3f}" if "f1" in m else f"{m['mae']:.3f}" if "mae" in m else "–"
            tuned = f"{before['tuned_f1']:.3f} → {after['tuned_f1']:.3f}" if "tuned_f1" in before else "–"
            print(f"| {name} | {scorer} | {before['acc']:.3f} | {before['nll']:.3f} | {before['ece15']:.3f} | {extra(before)} "
                  f"| {after['acc']:.3f} | {after['nll']:.3f} | {after['ece15']:.3f} | {extra(after)} | {tuned} |")
    print("\nfitted (pooled per type / K-bucket):", json.dumps(fitted, indent=1))
    if args.json:
        json.dump(fitted, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
