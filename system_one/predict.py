"""Typed answers from per-question logits (README §2.3–2.4)."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from system_one.schema import Answer, ChoiceAnswer, NoulAnswer, Request, ScoreAnswer
from system_one.templates import option_keys

Scorer = Callable[[Request], dict[str, torch.Tensor]]


def confidence(p: torch.Tensor) -> float:
    """Normalised entropy confidence, 1 - H(p) / log K, comparable across option counts."""
    k = p.numel()
    if k < 2:
        return 1.0
    entropy = -(p * p.clamp_min(1e-12).log()).sum().item()
    return max(0.0, 1.0 - entropy / math.log(k))


def to_answer(request: Request, qid: str, logits: torch.Tensor, temperature: float = 1.0) -> Answer:
    q = request.questions[qid]
    if q.type == "noul":
        return NoulAnswer(noul=torch.sigmoid(logits.float() / temperature).item())
    p = torch.softmax(logits.float() / temperature, dim=-1)
    if q.type == "choice":
        keys = option_keys(q)
        probs = dict(zip(keys, p.tolist()))
        return ChoiceAnswer(choice=keys[int(p.argmax())], probabilities=probs, confidence=confidence(p))
    levels = torch.arange(p.numel(), dtype=p.dtype, device=p.device)
    return ScoreAnswer(score=(p * levels).sum().item(), probabilities=p.tolist(), confidence=confidence(p))


def predict(
    request: Request | dict,
    scorer: Scorer,
    temperature: Callable[[str, int], float] | None = None,
) -> dict[str, Answer]:
    """Answer every question in `request`. `temperature(type, K)` defaults to 1 (no calibration)."""
    if isinstance(request, dict):
        request = Request.from_dict(request)
    logits = scorer(request)
    answers = {}
    for qid, q in request.questions.items():
        t = temperature(q.type, q.num_options) if temperature else 1.0
        answers[qid] = to_answer(request, qid, logits[qid], t)
    return answers
