"""Zero-shot baselines (README §7.7 #1–2) through the typed API, with temperature calibration (PLAN.md M3).

    python scripts/zero_shot_eval.py --limit 1000 --dump runs/zs.pt       # CUDA if available
    python scripts/zero_shot_eval.py --from-dump runs/zs.pt               # refit calibration, no model
    python scripts/zero_shot_eval.py --limit 20 --device cpu              # quick CPU check

Each example becomes one single-question request (context in the state), so the numbers measure the
scorers, not batching. Per dataset, the sampled examples are split in half: even indices are the
calibration slice and odd indices the test slice. For each scorer, temperatures are fitted on the
pooled calibration slices of all datasets (per type and K-bucket, and T = exp(a + b·log K)), and test
metrics are reported uncalibrated (T=1) and with each fit.
"""

import argparse
import json
import random
import time
from collections import defaultdict

import numpy as np
import torch

from system_one.calibrate import LogKTemperature, TemperatureTable, k_bucket
from system_one.metrics import binary_metrics, categorical_metrics, ordinal_metrics
from system_one.schema import Request
from system_one.templates import LETTERS, option_keys

SST5_LEVELS = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]
AG_NEWS = {"World": "World news and politics", "Sports": "Sports", "Business": "Business and economy",
           "SciTech": "Science and technology"}  # fmt: skip
TREC = {"ABBR": "An abbreviation or its expansion", "ENTY": "An entity (thing, object, event, ...)",
        "DESC": "A description or definition", "HUM": "A person or group", "LOC": "A location",
        "NUM": "A number, date or quantity"}  # fmt: skip


def _load(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _mcq(ex, question, labels, texts, answer):
    q = {"type": "choice", "instructions": "Which option correctly answers the question?",
         "criteria": dict(zip(labels, texts))}  # fmt: skip
    return {"state": question, "questions": {"q": q}}, answer


def arc(subset):
    def load(split="test"):
        for ex in _load("allenai/ai2_arc", subset, split=split):
            yield _mcq(ex, ex["question"], ex["choices"]["label"], ex["choices"]["text"], ex["answerKey"])

    return load


def csqa(split="validation"):
    for ex in _load("tau/commonsense_qa", split=split):
        yield _mcq(ex, ex["question"], ex["choices"]["label"], ex["choices"]["text"], ex["answerKey"])


def boolq(split="validation"):
    for ex in _load("google/boolq", split=split):
        instructions = ex["question"][:1].upper() + ex["question"][1:] + "?"
        yield {"state": ex["passage"], "questions": {"q": {"type": "noul", "instructions": instructions}}}, int(ex["answer"])


def sst5(split="test"):
    for ex in _load("SetFit/sst5", split=split):
        q = {"type": "score", "instructions": "How positive is this movie review?", "criteria": SST5_LEVELS}
        yield {"state": ex["text"], "questions": {"q": q}}, ex["label"]


def ag_news(split="test"):
    keys = list(AG_NEWS)
    for ex in _load("fancyzhx/ag_news", split=split):
        q = {"type": "choice", "instructions": "What is the topic of this news article?", "criteria": AG_NEWS}
        yield {"state": ex["text"], "questions": {"q": q}}, keys[ex["label"]]


def trec(split="test"):
    ds = _load("CogComp/trec", split=split)
    names = ds.features["coarse_label"].names
    for ex in ds:
        q = {"type": "choice", "instructions": "What kind of answer does this question ask for?", "criteria": TREC}
        yield {"state": ex["text"], "questions": {"q": q}}, names[ex["coarse_label"]]


def banking77(split="test"):
    ds = _load("PolyAI/banking77", split=split)
    names = ds.features["label"].names
    criteria = {n: n.replace("_", " ").capitalize() for n in names}
    for ex in ds:
        q = {"type": "choice", "instructions": "What does the customer want help with?", "criteria": criteria}
        yield {"state": ex["text"], "questions": {"q": q}}, names[ex["label"]]


DATASETS = {"arc_challenge": arc("ARC-Challenge"), "arc_easy": arc("ARC-Easy"), "csqa": csqa, "boolq": boolq,
            "sst5": sst5, "ag_news": ag_news, "trec": trec, "banking77": banking77}  # fmt: skip


def collect(examples, scorer) -> list[tuple[str, torch.Tensor, int]]:
    """Raw scorer logits per example as calibration records (type, logits, gold index)."""
    records = []
    for request, label in examples:
        request = Request.from_dict(request)
        q = request.questions["q"]
        gold = label if q.type in ("noul", "score") else option_keys(q).index(label)
        records.append((q.type, scorer(request)["q"].detach().float().cpu(), gold))
    return records


def evaluate(records, temperature=None) -> dict:
    """Metrics for `records` after dividing logits by `temperature(type, K)`."""
    probs, gold = [], []
    kind = records[0][0]
    for k, logits, g in records:
        t = temperature(k, logits.numel()) if temperature else 1.0
        z = logits / t
        probs.append(torch.sigmoid(z).item() if k == "noul" else torch.softmax(z, -1).numpy())
        gold.append(g)
    return {"noul": binary_metrics, "choice": categorical_metrics, "score": ordinal_metrics}[kind](probs, gold)


def run_model(args) -> dict[tuple[str, str], list]:
    from system_one.scorers import Backbone, LetterScorer, LikelihoodScorer

    torch.set_grad_enabled(False)
    backbone = Backbone.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device=args.device)
    scorers = {
        "letter": lambda: LetterScorer(backbone),
        "sum": lambda: LikelihoodScorer(backbone, "sum"),
        "mean": lambda: LikelihoodScorer(backbone, "mean"),
        "sum-pmi": lambda: LikelihoodScorer(backbone, "sum", pmi=True),
        "mean-pmi": lambda: LikelihoodScorer(backbone, "mean", pmi=True),
    }
    out = {}
    for name in args.datasets.split(","):
        examples = list(DATASETS[name]())
        examples = random.Random(args.seed).sample(examples, min(args.limit, len(examples)))
        max_k = max(Request.from_dict(r).questions["q"].num_options for r, _ in examples)
        for scorer_name in args.scorers.split(","):
            if scorer_name == "letter" and max_k > len(LETTERS):
                continue
            start = time.perf_counter()
            out[name, scorer_name] = collect(examples, scorers[scorer_name]())
            sec = (time.perf_counter() - start) / len(examples)
            print(f"collected {name} / {scorer_name}: {len(examples)} examples, {sec:.2f} s/example", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--scorers", default="letter,sum,mean,sum-pmi,mean-pmi")
    parser.add_argument("--limit", type=int, default=200, help="random examples per dataset (half calibrate, half test)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump", help="save raw logits here (torch.save)")
    parser.add_argument("--from-dump", help="load raw logits instead of running the model")
    parser.add_argument("--json", help="also write results here")
    args = parser.parse_args()

    if args.from_dump:
        data = torch.load(args.from_dump)
    else:
        data = run_model(args)
        if args.dump:
            torch.save(data, args.dump)

    calib, test = defaultdict(list), {}
    for (dataset, scorer), records in data.items():
        calib[scorer] += records[0::2]
        test[dataset, scorer] = records[1::2]
    fits = {s: {"table": TemperatureTable.fit(r), "logk": LogKTemperature.fit(r)} for s, r in calib.items()}

    print("\nFitted temperatures (calibration halves, pooled over datasets):\n")
    for scorer, fit in fits.items():
        table = ", ".join(f"{k}={v:.2f} (n={fit['table'].counts[k]})" for k, v in sorted(fit["table"].temperatures.items()))
        logk = fit["logk"]
        print(f"- {scorer}: {table}; log-K fit a={logk.a:.2f} b={logk.b:.2f} noul={logk.noul:.2f}")

    print("\nTest halves. Cells are uncalibrated → per-bucket T (→ log-K T for NLL and ECE).\n")
    print("| dataset | K | scorer | n | acc | NLL | ECE-15 | Brier | AUROC | MAE |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    results = []
    for (dataset, scorer), records in test.items():
        raw, table, logk = (evaluate(records, t) for t in (None, fits[scorer]["table"], fits[scorer]["logk"]))
        k = records[0][1].numel() if records[0][0] != "noul" else 2
        results.append({"dataset": dataset, "scorer": scorer, "k": k, "k_bucket": k_bucket(k),
                        "raw": raw, "table": table, "logk": logk})  # fmt: skip
        mae = f"{raw['mae']:.2f} → {table['mae']:.2f}" if "mae" in raw else "–"
        print(
            f"| {dataset} | {k} | {scorer} | {raw['n']} | {raw['acc']:.3f} "
            f"| {raw['nll']:.3f} → {table['nll']:.3f} → {logk['nll']:.3f} "
            f"| {raw['ece15']:.3f} → {table['ece15']:.3f} → {logk['ece15']:.3f} "
            f"| {raw['brier']:.3f} → {table['brier']:.3f} | {raw['auroc']:.3f} | {mae} |",
            flush=True,
        )
    if args.json:
        payload = {"results": results, "fits": {s: {n: json.loads(f.to_json()) for n, f in v.items()} for s, v in fits.items()}}
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
