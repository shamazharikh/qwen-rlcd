"""Zero-shot baselines (README §7.7 #1–2) through the typed API, on a few benchmarks.

    python scripts/zero_shot_eval.py --limit 200                  # CUDA if available
    python scripts/zero_shot_eval.py --limit 20 --device cpu      # quick CPU check
    GEMINI_API_KEY=... python scripts/zero_shot_eval.py --scorers gemini   # Google Decisions API reference
    python scripts/zero_shot_eval.py --scorers vllm --model 9b            # vLLM /v1/systemone serving the same model
    GEMINI_API_KEY=... python scripts/zero_shot_eval.py --scorers chat:gemini-2.5-flash-lite,chat:gemini-3.1-flash-lite

Single-question datasets turn each example into a one-question request (context in the state). Multi-question
datasets put several questions on one state, which is where the fork (one prefill, many branches), Gemini
Decisions (up to 1,000 questions per call), vLLM /v1/systemone (64) and chat baselines (one call per question)
differ in latency and cost:

- `race`: an exam passage with 3-6 multiple-choice questions (long state, few questions).
- `go_emotions`: a comment with 28 yes/no emotion questions (short state, wide fan-out; reports binary F1).

    python scripts/zero_shot_eval.py --datasets race,go_emotions --limit 200 --gpu-usd-per-hour 0.9776

Metrics pool every question in a dataset. Prints a markdown table per dataset and scorer.
"""

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from datasets import load_dataset

from system_one.chat import ChatCompletionScorer
from system_one.remote import GeminiDecisions, VllmSystemOne
from system_one.metrics import binary_metrics, categorical_metrics, ordinal_metrics
from system_one.predict import predict, predict_batch
from system_one.schema import Request
from system_one.models import resolve
from system_one.scorers import Backbone, LetterScorer, LikelihoodScorer
from system_one.templates import option_keys

REMOTE = {"gemini", "vllm"}  # scorers served over HTTP (plus any "chat:<model>"); no local model needed
# $ per 1M (input, output) tokens, Google list prices (2026-11): cost per call for API scorers.
PRICES = {"gemini-decision-1": (0.10, 0.0), "gemini-2.5-flash-lite": (0.10, 0.40), "gemini-3.1-flash-lite": (0.25, 1.50)}


def is_remote(name):
    return name in REMOTE or name.startswith("chat:")
SST5_LEVELS = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]


def _mcq(question, labels, texts, answer):
    q = {"type": "choice", "instructions": "Which option correctly answers the question?", "criteria": dict(zip(labels, texts))}
    return {"state": question, "questions": {"q": q}}, answer


def arc(split="test", subset="ARC-Challenge"):
    for ex in load_dataset("allenai/ai2_arc", subset, split=split):
        yield _mcq(ex["question"], ex["choices"]["label"], ex["choices"]["text"], ex["answerKey"])


def arc_easy(split="test"):
    yield from arc(split, "ARC-Easy")


def commonsense_qa(split="validation"):  # test split has no labels
    for ex in load_dataset("tau/commonsense_qa", split=split):
        yield _mcq(ex["question"], ex["choices"]["label"], ex["choices"]["text"], ex["answerKey"])


AG_NEWS_TOPICS = {
    "world": "World news and international affairs",
    "sports": "Sports",
    "business": "Business and economy",
    "scitech": "Science and technology",
}


def ag_news(split="test"):
    keys = list(AG_NEWS_TOPICS)  # dataset label order: World, Sports, Business, Sci/Tech
    for ex in load_dataset("fancyzhx/ag_news", split=split):
        q = {"type": "choice", "instructions": "What is the topic of this news article?", "criteria": AG_NEWS_TOPICS}
        yield {"state": ex["text"], "questions": {"q": q}}, keys[ex["label"]]


def boolq(split="validation"):
    for ex in load_dataset("google/boolq", split=split):
        instructions = ex["question"][:1].upper() + ex["question"][1:] + "?"
        yield {"state": ex["passage"], "questions": {"q": {"type": "noul", "instructions": instructions}}}, int(ex["answer"])


def sst5(split="test"):
    for ex in load_dataset("SetFit/sst5", split=split):
        q = {"type": "score", "instructions": "How positive is this movie review?", "criteria": SST5_LEVELS}
        yield {"state": ex["text"], "questions": {"q": q}}, ex["label"]


def race(split="test"):
    """RACE: questions grouped by passage, one request per passage."""
    passages = {}
    for ex in load_dataset("ehovy/race", "all", split=split):
        passages.setdefault(ex["example_id"], (ex["article"], []))[1].append(ex)
    for article, rows in passages.values():
        questions, labels = {}, {}
        for i, ex in enumerate(rows):
            qid = f"q{i}"
            instructions = ex["question"] if "_" not in ex["question"] else f"Which option best completes: {ex['question']}"
            questions[qid] = {"type": "choice", "instructions": instructions, "criteria": dict(zip("ABCD", ex["options"]))}
            labels[qid] = ex["answer"]
        yield {"state": article, "questions": questions}, labels


def go_emotions(split="test"):
    """GoEmotions (simplified): 28 yes/no questions per comment, one per emotion label (multi-label gold)."""
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
    names = ds.features["labels"].feature.names
    def text(n):
        return "The text is emotionally neutral." if n == "neutral" else f"The text expresses {n}."
    for ex in ds:
        questions = {n: {"type": "noul", "instructions": text(n)} for n in names}
        yield {"state": ex["text"], "questions": questions}, {n: int(i in ex["labels"]) for i, n in enumerate(names)}


MULTI_QUESTION = {"race", "go_emotions"}
# Held-out split per dataset for fitting calibration (never the split the test numbers come from).
CALIBRATION_SPLITS = {
    "arc_challenge": "validation", "arc_easy": "validation", "commonsense_qa": "train", "ag_news": "train",
    "boolq": "train", "sst5": "validation", "race": "validation", "go_emotions": "validation",
}  # fmt: skip
DATASETS = {
    "arc_challenge": arc,
    "arc_easy": arc_easy,
    "commonsense_qa": commonsense_qa,
    "ag_news": ag_news,
    "boolq": boolq,
    "sst5": sst5,
    "race": race,
    "go_emotions": go_emotions,
}


def evaluate(examples, scorer, workers=1, batch_requests=1):
    """`scorer` is a local scorer (run through `predict`) or anything with `.predict(request)`, e.g. GeminiDecisions.
    Labels are a scalar for single-question requests (question id "q") or a {qid: label} dict. Metrics pool all
    questions; a dataset must use one question type."""
    requests = [Request.from_dict(r) for r, _ in examples]
    if not hasattr(scorer, "predict") and batch_requests > 1:  # local fork scorer: several requests per forward
        all_answers = []
        for i in range(0, len(requests), batch_requests):
            all_answers += predict_batch(requests[i : i + batch_requests], scorer)
    else:
        run = scorer.predict if hasattr(scorer, "predict") else (lambda r: predict(r, scorer))
        with ThreadPoolExecutor(workers) as pool:
            all_answers = list(pool.map(run, requests))
    probs, gold, kinds = [], [], set()
    pairs = []
    for request, (_, labels), answers in zip(requests, examples, all_answers):
        labels = labels if isinstance(labels, dict) else {"q": labels}
        pairs += [(request.questions[qid], label, answers[qid]) for qid, label in labels.items()]
    for q, label, answer in pairs:
        kinds.add(q.type)
        kind = q.type
        if kind == "noul":
            probs.append(answer.noul)
            gold.append(label)
        elif kind == "choice":
            probs.append(np.array([answer.probabilities[k] for k in option_keys(q)]))
            gold.append(option_keys(q).index(label))
        else:
            probs.append(np.array(answer.probabilities))
            gold.append(label)
    if len(kinds) != 1:
        raise ValueError(f"mixed question types in one dataset: {kinds}")
    metrics = {"noul": binary_metrics, "choice": categorical_metrics, "score": ordinal_metrics}[kind](probs, gold)
    # Per-question predictions for post-hoc calibration (noul as [P(no), P(yes)]).
    metrics["_preds"] = [
        {"type": kind, "k": 2 if kind == "noul" else len(p), "probs": [1 - p, p] if kind == "noul" else [float(x) for x in p], "gold": int(g)}
        for p, g in zip(probs, gold)
    ]
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="0.8b", help="alias (0.8b, 9b, 35b-a3b) or HF repo id")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default=None, choices=["float32", "bfloat16", "float16"], help="default: fp32 for 0.8B, bf16 for larger")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--scorers", default="letter,sum,mean,sum-pmi,mean-pmi")
    parser.add_argument("--limit", type=int, default=200, help="random examples per dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="also write results here")
    parser.add_argument("--device-map", help='e.g. "auto" to shard a model across GPUs')
    parser.add_argument("--quantization", default="default", choices=["default", "fp8", "none"], help="default: fp8 for 35b-a3b")
    parser.add_argument("--remote-workers", type=int, default=8, help="concurrent requests for gemini / vllm / chat:*")
    parser.add_argument("--reasoning-effort", default=None, help="chat:* scorers: OpenAI-style reasoning_effort (default: model default)")
    parser.add_argument("--split", default="test", choices=["test", "calibration"], help="calibration: held-out split per dataset (CALIBRATION_SPLITS)")
    parser.add_argument("--dump-preds", help="write per-question probabilities and gold labels here (for scripts/calibrate_eval.py)")
    parser.add_argument("--batch-requests", type=int, default=1, help="local scorers: requests prefilled and branched together")
    parser.add_argument("--chunk-size", type=int, default=None, help="local scorers: max branch rows per forward (default 32)")
    parser.add_argument("--gpu-usd-per-hour", type=float, default=None, help="local / vllm: GPU price, for $ per 1M requests (L4 g6.2xlarge on-demand: 0.9776)")
    parser.add_argument("--vllm-url", default=None, help="vLLM server for --scorers vllm (default $VLLM_BASE_URL or localhost:8000)")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    local = [s for s in args.scorers.split(",") if not is_remote(s)]
    backbone = Backbone.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype) if args.dtype else None, device=args.device, device_map=args.device_map,
        quantization=args.quantization,
    ) if local else None
    if backbone and args.chunk_size:
        backbone.chunk_size = args.chunk_size
    scorers = {
        "gemini": lambda: GeminiDecisions(),  # hosted reference model; needs GEMINI_API_KEY
        "vllm": lambda: VllmSystemOne(resolve(args.model), args.vllm_url),  # vLLM /v1/systemone serving --model
        "letter": lambda: LetterScorer(backbone),
        "sum": lambda: LikelihoodScorer(backbone, "sum"),
        "mean": lambda: LikelihoodScorer(backbone, "mean"),
        "sum-pmi": lambda: LikelihoodScorer(backbone, "sum", pmi=True),
        "mean-pmi": lambda: LikelihoodScorer(backbone, "mean", pmi=True),
    }
    results, dumps = [], {}
    dtype = backbone.text_model.dtype if backbone else "-"
    print(f"\n{args.model} on {args.device}, {dtype}, {args.limit} examples per dataset (seed {args.seed})\n")
    print("| dataset | scorer | requests | questions | acc | NLL | Brier | ECE-15 | MAE / F1 | s/request | ms/question | $ per 1M requests |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in args.datasets.split(","):
        examples = list(DATASETS[name](CALIBRATION_SPLITS[name]) if args.split == "calibration" else DATASETS[name]())
        examples = random.Random(args.seed).sample(examples, min(args.limit, len(examples)))
        for scorer_name in args.scorers.split(","):
            start = time.perf_counter()
            workers = args.remote_workers if is_remote(scorer_name) else 1
            if scorer_name.startswith("chat:"):
                scorer = ChatCompletionScorer(scorer_name[5:], reasoning_effort=args.reasoning_effort)
            else:
                scorer = scorers[scorer_name]()
            m = evaluate(examples, scorer, workers, args.batch_requests)
            preds = m.pop("_preds")
            if args.dump_preds:
                dumps.setdefault(name, {})[scorer_name] = preds
                with open(args.dump_preds, "w") as f:
                    json.dump(dumps, f)
            usage = getattr(scorer, "usage", None)
            if usage:  # chat scorer: billed tokens, hard-answer share, cost per call
                m.update({k: int(v) for k, v in usage.items()})
                m["hard"] = usage["hard_answers"] > usage["calls"] / 2
                p_in, p_out = PRICES.get(scorer_name[5:], (float("nan"), float("nan")))
                m["usd_per_request"] = (usage["prompt_tokens"] * p_in + usage["completion_tokens"] * p_out) / 1e6 / len(examples)
            elif scorer_name == "gemini":
                m["usd_per_request"] = scorer.usage_tokens * PRICES["gemini-decision-1"][0] / 1e6 / len(examples)
            m["requests"] = len(examples)
            m["sec_per_example"] = (time.perf_counter() - start) / len(examples)  # wall time per request
            if "usd_per_request" not in m and args.gpu_usd_per_hour:
                m["usd_per_request"] = args.gpu_usd_per_hour / 3600 * m["sec_per_example"]
            results.append({"dataset": name, "scorer": scorer_name, **m})
            mae = f"{m['mae']:.3f}" if "mae" in m else f"F1 {m['f1']:.3f}" if name in MULTI_QUESTION and "f1" in m else "–"
            # Hard (one-hot) answers have no meaningful NLL / Brier / ECE.
            cal = "– | – | –" if m.get("hard") else f"{m['nll']:.3f} | {m['brier']:.3f} | {m['ece15']:.3f}"
            cost = f"{m['usd_per_request'] * 1e6:.2f}" if "usd_per_request" in m else "–"
            ms_q = 1000 * m["sec_per_example"] * len(examples) / m["n"]
            print(
                f"| {name} | {scorer_name} | {len(examples)} | {m['n']} | {m['acc']:.3f} | {cal} | {mae} "
                f"| {m['sec_per_example']:.3f} | {ms_q:.1f} | {cost} |",
                flush=True,
            )
            if args.json:  # rewrite after every row so a crash keeps finished results
                with open(args.json, "w") as f:
                    json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
