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


def categorical_metrics(probs: list[np.ndarray], gold: list[int]) -> dict[str, float]:
    """Accuracy, NLL, multi-class Brier and ECE-15 (on top-1 probability) for per-example distributions."""
    top = [int(np.argmax(p)) for p in probs]
    # A tie for the top probability (e.g. a uniform fallback for an unparseable answer) counts as wrong.
    correct = [t == g and int((p == p[t]).sum()) == 1 for p, t, g in zip(probs, top, gold)]
    nll = [-math.log(max(p[g], 1e-12)) for p, g in zip(probs, gold)]
    brier = [float(((p - np.eye(len(p))[g]) ** 2).sum()) for p, g in zip(probs, gold)]
    return {
        "n": len(gold),
        "acc": float(np.mean(correct)),
        "nll": float(np.mean(nll)),
        "brier": float(np.mean(brier)),
        "ece15": ece([p[t] for p, t in zip(probs, top)], correct),
    }


def binary_metrics(p_yes: list[float], gold: list[int]) -> dict[str, float]:
    """Categorical metrics plus F1 of the positive class (decision at P(yes) > 0.5), for imbalanced multi-label sets."""
    out = categorical_metrics([np.array([1 - p, p]) for p in p_yes], gold)
    pred, g = np.asarray(p_yes) > 0.5, np.asarray(gold) == 1
    tp = float((pred & g).sum())
    out["f1"] = 2 * tp / (pred.sum() + g.sum()) if pred.sum() + g.sum() else float("nan")
    return out


def ordinal_metrics(probs: list[np.ndarray], gold: list[int]) -> dict[str, float]:
    """Categorical metrics plus MAE of the expected level."""
    out = categorical_metrics(probs, gold)
    out["mae"] = float(np.mean([abs(float(p @ np.arange(len(p))) - g) for p, g in zip(probs, gold)]))
    return out
