"""Zero-shot baselines (README §7.7 #1–2) through the typed API, on a few benchmarks.

    python scripts/zero_shot_eval.py --limit 200                  # CUDA if available
    python scripts/zero_shot_eval.py --limit 20 --device cpu      # quick CPU check

Each example becomes one single-question request (context in the state), so the numbers measure the
scorers, not batching. Prints a markdown table per dataset and scorer.
"""

import argparse
import json
import random
import time

import numpy as np
import torch
from datasets import load_dataset

from system_one.metrics import binary_metrics, categorical_metrics, ordinal_metrics
from system_one.predict import predict
from system_one.schema import Request
from system_one.scorers import Backbone, LetterScorer, LikelihoodScorer
from system_one.templates import option_keys

SST5_LEVELS = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]


def arc(split="test"):
    for ex in load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split):
        criteria = dict(zip(ex["choices"]["label"], ex["choices"]["text"]))
        q = {"type": "choice", "instructions": "Which option correctly answers the question?", "criteria": criteria}
        yield {"state": ex["question"], "questions": {"q": q}}, ex["answerKey"]


def boolq(split="validation"):
    for ex in load_dataset("google/boolq", split=split):
        instructions = ex["question"][:1].upper() + ex["question"][1:] + "?"
        yield {"state": ex["passage"], "questions": {"q": {"type": "noul", "instructions": instructions}}}, int(ex["answer"])


def sst5(split="test"):
    for ex in load_dataset("SetFit/sst5", split=split):
        q = {"type": "score", "instructions": "How positive is this movie review?", "criteria": SST5_LEVELS}
        yield {"state": ex["text"], "questions": {"q": q}}, ex["label"]


DATASETS = {"arc_challenge": arc, "boolq": boolq, "sst5": sst5}


def evaluate(examples, scorer):
    probs, gold, kind = [], [], None
    for request, label in examples:
        request = Request.from_dict(request)
        q = request.questions["q"]
        kind = q.type
        answer = predict(request, scorer)["q"]
        if kind == "noul":
            probs.append(answer.noul)
            gold.append(label)
        elif kind == "choice":
            probs.append(np.array([answer.probabilities[k] for k in option_keys(q)]))
            gold.append(option_keys(q).index(label))
        else:
            probs.append(np.array(answer.probabilities))
            gold.append(label)
    return {"noul": binary_metrics, "choice": categorical_metrics, "score": ordinal_metrics}[kind](probs, gold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--scorers", default="letter,sum,mean,sum-pmi,mean-pmi")
    parser.add_argument("--limit", type=int, default=200, help="random examples per dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="also write results here")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    backbone = Backbone.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device=args.device)
    scorers = {
        "letter": lambda: LetterScorer(backbone),
        "sum": lambda: LikelihoodScorer(backbone, "sum"),
        "mean": lambda: LikelihoodScorer(backbone, "mean"),
        "sum-pmi": lambda: LikelihoodScorer(backbone, "sum", pmi=True),
        "mean-pmi": lambda: LikelihoodScorer(backbone, "mean", pmi=True),
    }
    results = []
    print(f"\n{args.model} on {args.device}, {args.dtype}, {args.limit} examples per dataset (seed {args.seed})\n")
    print("| dataset | scorer | n | acc | NLL | Brier | ECE-15 | MAE | s/example |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in args.datasets.split(","):
        examples = list(DATASETS[name]())
        examples = random.Random(args.seed).sample(examples, min(args.limit, len(examples)))
        for scorer_name in args.scorers.split(","):
            start = time.perf_counter()
            m = evaluate(examples, scorers[scorer_name]())
            m["sec_per_example"] = (time.perf_counter() - start) / len(examples)
            results.append({"dataset": name, "scorer": scorer_name, **m})
            mae = f"{m['mae']:.3f}" if "mae" in m else "–"
            print(
                f"| {name} | {scorer_name} | {m['n']} | {m['acc']:.3f} | {m['nll']:.3f} | {m['brier']:.3f} "
                f"| {m['ece15']:.3f} | {mae} | {m['sec_per_example']:.2f} |",
                flush=True,
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
