"""Temperature calibration (PLAN.md M3).

Temperatures are fitted by minimising NLL on held-out (logits, gold) pairs and returned as a
`temperature(type, K)` callable for `predict`. Two parameterisations:

- `TemperatureTable`: one T per (question type, K-bucket), buckets K <= 5 / 6-20 / > 20.
- `LogKTemperature`: T = exp(a + b·log K) for choice/score questions, one T for noul.

A record is (type, logits, gold): logits as returned by a scorer ([K] tensor for choice/score in
`option_keys` order, scalar P(yes) logit for noul), and gold as the option index (noul: 1 = yes).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

K_BUCKETS = ((5, "k<=5"), (20, "k6-20"), (math.inf, "k>20"))

Record = tuple[str, torch.Tensor, int]


def k_bucket(k: int) -> str:
    return next(name for upper, name in K_BUCKETS if k <= upper)


def _key(kind: str, k: int) -> str:
    return "noul" if kind == "noul" else f"{kind}/{k_bucket(k)}"


def _nll(records: list[Record], log_t: torch.Tensor) -> torch.Tensor:
    """Mean NLL of `records` at per-record temperatures exp(log_t) ([N] or scalar)."""
    log_t = log_t.expand(len(records))
    losses = []
    for (kind, logits, gold), lt in zip(records, log_t):
        z = logits.detach().float().cpu() / lt.exp()
        if kind == "noul":
            losses.append(F.binary_cross_entropy_with_logits(z, torch.tensor(float(gold))))
        else:
            losses.append(F.cross_entropy(z[None], torch.tensor([gold])))
    return torch.stack(losses).mean()


def _minimise(loss_fn, params: list[torch.Tensor], steps: int = 100) -> None:
    opt = torch.optim.LBFGS(params, lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    opt.step(closure)


def fit_temperature(records: list[Record]) -> float:
    """Single temperature minimising the mean NLL of `records`."""
    log_t = torch.zeros((), requires_grad=True)
    _minimise(lambda: _nll(records, log_t), [log_t])
    return float(log_t.detach().clamp(-5, 5).exp())


@dataclass
class TemperatureTable:
    temperatures: dict[str, float] = field(default_factory=dict)  # "choice/k<=5" / "score/k6-20" / "noul" -> T
    counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, records: list[Record], min_count: int = 20) -> TemperatureTable:
        """Fit one T per (type, K-bucket) that has at least `min_count` records; others fall back to T = 1."""
        groups: dict[str, list[Record]] = defaultdict(list)
        for r in records:
            groups[_key(r[0], r[1].numel())].append(r)
        table = cls(counts={k: len(v) for k, v in groups.items()})
        for key, group in groups.items():
            if len(group) >= min_count:
                table.temperatures[key] = fit_temperature(group)
        return table

    def __call__(self, kind: str, k: int) -> float:
        return self.temperatures.get(_key(kind, k), 1.0)

    def to_json(self) -> str:
        return json.dumps({"kind": "table", "temperatures": self.temperatures, "counts": self.counts})


@dataclass
class LogKTemperature:
    a: float = 0.0
    b: float = 0.0
    noul: float = 1.0

    @classmethod
    def fit(cls, records: list[Record]) -> LogKTemperature:
        options = [r for r in records if r[0] != "noul"]
        binary = [r for r in records if r[0] == "noul"]
        out = cls(noul=fit_temperature(binary) if binary else 1.0)
        if options:
            a, b = torch.zeros((), requires_grad=True), torch.zeros((), requires_grad=True)
            log_k = torch.tensor([math.log(r[1].numel()) for r in options])
            _minimise(lambda: _nll(options, (a + b * log_k).clamp(-5, 5)), [a, b])
            out.a, out.b = a.item(), b.item()
        return out

    def __call__(self, kind: str, k: int) -> float:
        if kind == "noul":
            return self.noul
        return math.exp(min(5.0, max(-5.0, self.a + self.b * math.log(k))))

    def to_json(self) -> str:
        return json.dumps({"kind": "logk", "a": self.a, "b": self.b, "noul": self.noul})


def load_temperature(text: str) -> TemperatureTable | LogKTemperature:
    data = json.loads(text)
    if data.pop("kind") == "table":
        return TemperatureTable(**data)
    return LogKTemperature(**data)
