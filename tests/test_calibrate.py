"""M3: temperature fitting recovers known temperatures on synthetic data."""

import math

import pytest
import torch

from system_one.calibrate import LogKTemperature, TemperatureTable, fit_temperature, k_bucket, load_temperature


def synthetic(kind: str, k: int, true_t: float, n: int = 3000, seed: int = 0):
    """Records whose labels are sampled from softmax(logits / true_t), so true_t is the NLL-optimal T."""
    g = torch.Generator().manual_seed(seed)
    records = []
    for _ in range(n):
        if kind == "noul":
            z = torch.randn((), generator=g) * 3
            gold = int(torch.rand((), generator=g) < torch.sigmoid(z / true_t))
        else:
            z = torch.randn(k, generator=g) * 3
            gold = int(torch.multinomial(torch.softmax(z / true_t, -1), 1, generator=g))
        records.append((kind, z, gold))
    return records


def test_k_bucket():
    assert [k_bucket(k) for k in (2, 5, 6, 20, 21, 200)] == ["k<=5"] * 2 + ["k6-20"] * 2 + ["k>20"] * 2


@pytest.mark.parametrize("kind,k,true_t", [("choice", 4, 2.0), ("score", 5, 0.5), ("noul", 2, 3.0)])
def test_fit_temperature_recovers_truth(kind, k, true_t):
    assert fit_temperature(synthetic(kind, k, true_t)) == pytest.approx(true_t, rel=0.15)


def test_table_fits_per_bucket_and_falls_back():
    records = synthetic("choice", 4, 2.0) + synthetic("choice", 10, 0.5, seed=1) + synthetic("noul", 2, 1.5, n=10)
    table = TemperatureTable.fit(records)
    assert table("choice", 3) == pytest.approx(2.0, rel=0.15)
    assert table("choice", 12) == pytest.approx(0.5, rel=0.15)
    assert table("noul", 2) == 1.0  # too few records: uncalibrated
    assert table("score", 4) == 1.0  # never seen
    assert load_temperature(table.to_json()) == table


def test_logk_fits_trend():
    # T grows with K: T = exp(0.1 + 0.4 log K)
    records = [r for k in (2, 4, 8, 16) for r in synthetic("choice", k, math.exp(0.1 + 0.4 * math.log(k)), 1500, k)]
    records += synthetic("noul", 2, 2.0, 1500)
    fit = LogKTemperature.fit(records)
    assert fit.a == pytest.approx(0.1, abs=0.15) and fit.b == pytest.approx(0.4, abs=0.1)
    assert fit("noul", 2) == pytest.approx(2.0, rel=0.15)
    assert load_temperature(fit.to_json()) == fit
