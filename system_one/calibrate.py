"""Post-hoc calibration of zero-shot scorers (PLAN.md M3), fitted on a held-out calibration split.

- choice / score: one temperature per (type, K-bucket), minimising NLL. Accuracy is unchanged (argmax kept).
- noul: Platt scaling, p = sigmoid(a * logit(p_raw) + b) per bucket. The bias `b` removes a constant yes/no
  prior (zero-shot 9B says "yes" to ~45% of GoEmotions emotions against a 4% base rate), which a temperature
  alone can't fix.

Works on probabilities, so it applies to any scorer that returns them (fork, vLLM, Gemini Decisions).
Fitted parameters are a small JSON dict usable as `predict(..., temperature=...)` for choice/score and via
`apply` for everything.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def k_bucket(k: int) -> str:
    return "<=5" if k <= 5 else "6-20" if k <= 20 else ">20"


def bucket(qtype: str, k: int) -> str:
    return "noul" if qtype == "noul" else f"{qtype}:{k_bucket(k)}"


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def fit_temperature(probs: list[np.ndarray], gold: list[int], steps: int = 300) -> float:
    """T minimising NLL of softmax(log p / T). Log-probs are logits up to a per-row constant."""
    logp = [torch.tensor(np.log(np.clip(p, 1e-12, 1.0)), dtype=torch.float64) for p in probs]
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = sum(-torch.log_softmax(lp / log_t.exp(), -1)[g] for lp, g in zip(logp, gold)) / len(gold)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def fit_platt(p_yes: list[float], gold: list[int], steps: int = 300) -> tuple[float, float]:
    """(a, b) minimising NLL of sigmoid(a * logit(p) + b)."""
    x = torch.tensor(_logit(p_yes), dtype=torch.float64)
    y = torch.tensor(gold, dtype=torch.float64)
    ab = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([ab], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(ab[0] * x + ab[1], y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(ab[0].detach()), float(ab[1].detach())


def fit(preds: list[dict]) -> dict[str, dict]:
    """`preds`: dicts with type, k, probs (list; noul: [P(no), P(yes)]) and gold (index; noul: 1 = yes).
    Returns {bucket: {"temperature": T} | {"a": a, "b": b}, ...} with the sample count per bucket."""
    groups: dict[str, list[dict]] = {}
    for p in preds:
        groups.setdefault(bucket(p["type"], p["k"]), []).append(p)
    params = {}
    for name, rows in groups.items():
        if name == "noul":
            a, b = fit_platt([r["probs"][1] for r in rows], [r["gold"] for r in rows])
            params[name] = {"a": a, "b": b, "n": len(rows)}
        else:
            params[name] = {"temperature": fit_temperature([np.asarray(r["probs"]) for r in rows], [r["gold"] for r in rows]), "n": len(rows)}
    return params


def apply(params: dict[str, dict], qtype: str, probs) -> np.ndarray:
    """Calibrated probabilities for one question (noul: [P(no), P(yes)]). Unknown buckets pass through."""
    probs = np.asarray(probs, float)
    p = params.get(bucket(qtype, len(probs) if qtype != "noul" else 2))
    if not p:
        return probs
    if qtype == "noul":
        y = 1 / (1 + math.exp(-(p["a"] * _logit(probs[1]) + p["b"])))
        return np.array([1 - y, y])
    z = np.log(np.clip(probs, 1e-12, 1.0)) / p["temperature"]
    z = np.exp(z - z.max())
    return z / z.sum()
