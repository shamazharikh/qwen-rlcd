"""Post-hoc calibration: temperature recovers a known sharpening, Platt removes a constant yes-bias."""
import numpy as np

from system_one.calibrate import apply, fit


def test_temperature_and_platt_recover_known_distortions():
    rng = np.random.default_rng(0)
    preds = []
    for _ in range(2000):  # choice: true logits z, model reports softmax(3 z) (over-confident by T = 3)
        z = rng.normal(size=4)
        g = rng.choice(4, p=np.exp(z) / np.exp(z).sum())
        p = np.exp(3 * z) / np.exp(3 * z).sum()
        preds.append({"type": "choice", "k": 4, "probs": p.tolist(), "gold": int(g)})
    for _ in range(4000):  # noul: true logit x, model reports sigmoid(x + 3) (constant yes-bias)
        x = rng.normal(-3, 1.5)
        y = int(rng.random() < 1 / (1 + np.exp(-x)))
        q = 1 / (1 + np.exp(-(x + 3)))
        preds.append({"type": "noul", "k": 2, "probs": [1 - q, q], "gold": y})
    params = fit(preds)
    assert abs(params["choice:<=5"]["temperature"] - 3) < 0.4
    assert abs(params["noul"]["b"] + 3) < 0.4 and abs(params["noul"]["a"] - 1) < 0.15
    fixed = apply(params, "noul", [0.5, 0.5])[1]  # raw 0.5 = true logit -3
    assert 0.03 < fixed < 0.08
    assert np.allclose(apply({}, "choice", [0.2, 0.8]), [0.2, 0.8])
