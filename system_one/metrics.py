"""Calibration and accuracy metrics (README §3, PLAN.md M3)."""

from __future__ import annotations

import math

import numpy as np


def ece(confidences, correct, bins: int = 15) -> float:
    """Expected calibration error over equal-width confidence bins."""
    conf, corr = np.asarray(confidences, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(conf[mask].mean() - corr[mask].mean())
    return float(total)


def auroc(scores, positive) -> float:
    """Area under the ROC curve of `scores` for separating positives (ties count half). NaN if one class is absent."""
    s, y = np.asarray(scores, float), np.asarray(positive, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def categorical_metrics(probs: list[np.ndarray], gold: list[int]) -> dict[str, float]:
    """Accuracy, NLL, multi-class Brier, ECE-15 and AUROC (both on top-1 probability) for per-example distributions."""
    top = [int(np.argmax(p)) for p in probs]
    correct = [t == g for t, g in zip(top, gold)]
    nll = [-math.log(max(p[g], 1e-12)) for p, g in zip(probs, gold)]
    brier = [float(((p - np.eye(len(p))[g]) ** 2).sum()) for p, g in zip(probs, gold)]
    top_p = [p[t] for p, t in zip(probs, top)]
    return {
        "n": len(gold),
        "acc": float(np.mean(correct)),
        "nll": float(np.mean(nll)),
        "brier": float(np.mean(brier)),
        "ece15": ece(top_p, correct),
        "auroc": auroc(top_p, correct),
    }


def binary_metrics(p_yes: list[float], gold: list[int]) -> dict[str, float]:
    return categorical_metrics([np.array([1 - p, p]) for p in p_yes], gold)


def ordinal_metrics(probs: list[np.ndarray], gold: list[int]) -> dict[str, float]:
    """Categorical metrics plus MAE of the expected level."""
    out = categorical_metrics(probs, gold)
    out["mae"] = float(np.mean([abs(float(p @ np.arange(len(p))) - g) for p, g in zip(probs, gold)]))
    return out
