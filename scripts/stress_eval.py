"""Stress suite for the scorers (README §7.6, PLAN.md M3). Order and fan-out invariance are exact unit
tests (tests/test_predict.py); this measures the behaviours that aren't pass/fail:

- K scaling: Banking77 with the gold intent plus K-1 random distractors, K from 2 to 77. How accuracy,
  NLL, ECE and mean confidence move with K shows whether temperature must depend on K.
- Empty-state prior: the same questions with an empty state. A well-behaved scorer is near uniform
  (low KL to uniform) and has no favourite option (top-pick share near 1/K). PMI scorers are skipped:
  they subtract exactly this empty-state score, so they are uniform by construction.
- Length bias: on ARC, how often the argmax is the longest option, against how often gold is.

    python scripts/stress_eval.py --limit 200
"""

import argparse
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from zero_shot_eval import DATASETS, collect, evaluate  # noqa: E402

from system_one.schema import Request  # noqa: E402
from system_one.scorers import Backbone, LetterScorer, LikelihoodScorer  # noqa: E402
from system_one.templates import LETTERS, option_keys  # noqa: E402

K_VALUES = (2, 5, 10, 20, 40, 77)


def subsample_options(examples, k: int, seed: int):
    """Keep the gold option plus k-1 random distractors per example."""
    rng = random.Random(seed)
    out = []
    for request, label in examples:
        q = request["questions"]["q"]
        others = [key for key in q["criteria"] if key != label]
        keep = [label] + rng.sample(others, k - 1)
        q = {**q, "criteria": {key: q["criteria"][key] for key in keep}}
        out.append(({"state": request["state"], "questions": {"q": q}}, label))
    return out


def prior_stats(records) -> dict:
    """Mean KL(p || uniform) and the share of the single most-picked option, with the state emptied."""
    kl, picks = [], Counter()
    for kind, logits, _ in records:
        p = torch.sigmoid(logits).item() if kind == "noul" else None
        probs = np.array([p, 1 - p]) if kind == "noul" else torch.softmax(logits, -1).numpy()
        kl.append(float((probs * np.log(np.clip(probs * len(probs), 1e-12, None))).sum()))
        picks[int(probs.argmax())] += 1
    k = records[0][1].numel() if records[0][0] != "noul" else 2
    return {"k": k, "kl_uniform": float(np.mean(kl)), "top_pick_share": picks.most_common(1)[0][1] / len(records)}


def length_bias(examples, records, backbone) -> dict:
    """Share of argmax picks that are the longest option (in tokens), against the gold rate."""
    picked_longest = gold_longest = 0
    for (request, _), (_, logits, gold) in zip(examples, records):
        q = Request.from_dict(request).questions["q"]
        lengths = [len(backbone.encode(q.options[key])) for key in option_keys(q)]
        longest = int(np.argmax(lengths))
        picked_longest += int(logits.argmax()) == longest
        gold_longest += gold == longest
    return {"argmax_longest": picked_longest / len(records), "gold_longest": gold_longest / len(records)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--scorers", default="letter,sum,mean,sum-pmi,mean-pmi")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--prior-datasets", default="arc_challenge,boolq,sst5,ag_news,trec")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    backbone = Backbone.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device=args.device)
    scorers = {
        "letter": LetterScorer(backbone),
        "sum": LikelihoodScorer(backbone, "sum"),
        "mean": LikelihoodScorer(backbone, "mean"),
        "sum-pmi": LikelihoodScorer(backbone, "sum", pmi=True),
        "mean-pmi": LikelihoodScorer(backbone, "mean", pmi=True),
    }
    scorers = {name: scorers[name] for name in args.scorers.split(",")}

    def sample(name):
        examples = list(DATASETS[name]())
        return random.Random(args.seed).sample(examples, min(args.limit, len(examples)))

    print("\n### K scaling (Banking77, gold + K-1 random distractors, uncalibrated)\n")
    print("| scorer | K | acc | chance | NLL | log K | ECE-15 | mean conf |")
    print("|---|---|---|---|---|---|---|---|")
    banking = sample("banking77")
    for k in K_VALUES:
        examples = subsample_options(banking, k, args.seed + k)
        for name, scorer in scorers.items():
            if name == "letter" and k > len(LETTERS):
                continue
            records = collect(examples, scorer)
            m = evaluate(records)
            conf = np.mean([torch.softmax(z, -1).max().item() for _, z, _ in records])
            print(f"| {name} | {k} | {m['acc']:.3f} | {1 / k:.3f} | {m['nll']:.3f} | {math.log(k):.3f} "
                  f"| {m['ece15']:.3f} | {conf:.3f} |", flush=True)  # fmt: skip

    print("\n### Empty-state prior (state replaced by \"\"; PMI scorers are uniform here by construction)\n")
    print("| dataset | K | scorer | KL to uniform | top-pick share | 1/K |")
    print("|---|---|---|---|---|---|")
    for dataset in args.prior_datasets.split(","):
        empty = [({**request, "state": ""}, label) for request, label in sample(dataset)]
        for name, scorer in scorers.items():
            if getattr(scorer, "pmi", False):
                continue  # PMI subtracts the empty-state score, so it is exactly uniform here by construction
            s = prior_stats(collect(empty, scorer))
            print(f"| {dataset} | {s['k']} | {name} | {s['kl_uniform']:.3f} | {s['top_pick_share']:.3f} "
                  f"| {1 / s['k']:.3f} |", flush=True)  # fmt: skip

    print("\n### Length bias (ARC-Challenge)\n")
    print("| scorer | argmax is longest option | gold is longest option |")
    print("|---|---|---|")
    arc = [ex for ex in sample("arc_challenge") if len(Request.from_dict(ex[0]).questions["q"].options) == 4]
    for name, scorer in scorers.items():
        b = length_bias(arc, collect(arc, scorer), backbone)
        print(f"| {name} | {b['argmax_longest']:.3f} | {b['gold_longest']:.3f} |", flush=True)


if __name__ == "__main__":
    main()
