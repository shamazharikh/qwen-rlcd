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

Image and short-video datasets (PLAN.md M5) put the media in the state and every question about it in one request:

- `pope`: COCO images, ~18 yes/no object-presence questions each (random/popular/adversarial sampling).
- `mmstar`: one multiple-choice question per image, checked to need the image.
- `tempcompass_mc` / `tempcompass_yn`: ~10-30 s clips with temporal multiple-choice + caption-matching / yes-no questions.
- `nextqa`: NExT-QA multiple choice (5 options) on clips of at most 60 s, ~8 questions per clip.

Videos are transcoded once to <= 448 px (ffmpeg, cached) and images capped at 1 MP before any scorer sees them,
so the fork and vLLM read identical pixels. `--video-pruning-rate` turns on EVS in the fork; start vLLM with the
same `--video-pruning-rate` and `--media-io-kwargs '{"video": {"fps": 2, "num_frames": 32}}'`.

    python scripts/zero_shot_eval.py --model 9b --scorers letter --datasets pope,mmstar,tempcompass_mc --limit 200

    python scripts/zero_shot_eval.py --datasets race,go_emotions --limit 200 --gpu-usd-per-hour 0.9776

Metrics pool every question in a dataset. Prints a markdown table per dataset and scorer.
"""

import argparse
import glob
import json
import os
import random
import re
import subprocess
import time
import zipfile
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


# ---- images and short videos (M5) ----------------------------------------------------------------------------

MEDIA_ROOT = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "qwen_rlcd_media")
IMAGE_MAX_PIXELS = 1024 * 1024
VIDEO_MAX_SIDE = 448


def _cap_image(img):
    img = img.convert("RGB")
    w, h = img.size
    scale = (IMAGE_MAX_PIXELS / (w * h)) ** 0.5
    return img if scale >= 1 else img.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample=3)


def _video_index(repo: str, zip_name: str) -> dict[str, str]:
    """Download and extract a dataset's video zip once; map file stem -> path."""
    from huggingface_hub import hf_hub_download

    out = os.path.join(MEDIA_ROOT, repo.replace("/", "__"))
    if not os.path.exists(os.path.join(out, ".done")):
        with zipfile.ZipFile(hf_hub_download(repo, zip_name, repo_type="dataset")) as z:
            z.extractall(out)
        open(os.path.join(out, ".done"), "w").close()
    paths = [p for ext in ("mp4", "webm", "mkv", "avi") for p in glob.glob(os.path.join(out, "**", f"*.{ext}"), recursive=True)]
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}


def prepare_video(path: str) -> str:
    """Transcode to H.264 with the longer side <= VIDEO_MAX_SIDE (cached), so every scorer reads the same pixels."""
    out = os.path.join(MEDIA_ROOT, f"transcoded_{VIDEO_MAX_SIDE}", os.path.splitext(os.path.basename(path))[0] + ".mp4")
    if not os.path.exists(out):
        import imageio_ffmpeg

        os.makedirs(os.path.dirname(out), exist_ok=True)
        side = VIDEO_MAX_SIDE
        vf = f"scale='min({side},iw)':'min({side},ih)':force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2"
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", path, "-vf", vf, "-an",
               "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", out + ".tmp.mp4"]  # fmt: skip
        subprocess.run(cmd, check=True)
        os.replace(out + ".tmp.mp4", out)
    return out


def prepare_media(example):
    request, labels = example
    if request.get("videos"):
        request = {**request, "videos": [prepare_video(v) for v in request["videos"]]}
    return request, labels


def _options_from_lines(lines):
    """'A. text' / 'Option 1: text' / 'Caption A: text' lines -> {key: text}."""
    opts = {}
    for line in lines:
        m = re.match(r"^(?:Option|Sentence|Caption)?\s*([A-Z0-9])[.:]\s*(.+)$", line.strip())
        if m:
            opts[m.group(1)] = m.group(2).strip()
    return opts


def pope(split="test"):
    """POPE: yes/no object-presence questions, one request per COCO image with all its questions."""
    groups = {}
    for ex in load_dataset("lmms-lab-encoder/POPE", split=split):
        img, qs, labels = groups.setdefault(ex["image_source"], [None, {}, {}])
        if img is None:
            groups[ex["image_source"]][0] = _cap_image(ex["image"])
        qid = f"{ex['category']}_{ex['question_id']}"
        qs[qid] = {"type": "noul", "instructions": ex["question"]}
        labels[qid] = int(ex["answer"].strip().lower() == "yes")
    for img, qs, labels in groups.values():
        yield {"state": "Look at the image.", "images": [img], "questions": qs}, labels


def _mmstar_question(text: str):
    """(instructions, {letter: option}) from MMStar's two formats: 'Q\nOptions: A: x, B: y' and
    'Hint: ...\nQuestion: Q\nChoices:\n(A) x\n(B) y' (MathVista-style)."""
    m = re.search(r"\nChoices:\s*\n(.*)$", text, re.S)
    if m:
        opts = dict(re.findall(r"^\(([A-F])\)\s*(.+?)\s*$", m.group(1), re.M))
        q = re.search(r"Question:\s*(.*)$", text[: m.start()], re.S)
        return (q.group(1) if q else text[: m.start()]).strip(), opts
    m = re.search(r"\nOptions:\s*(.*)$", text, re.S)
    if not m:
        return text.strip(), {}
    parts = re.split(r"(?:^|,\s*)([A-F]):\s*", m.group(1).strip())
    return text[: m.start()].strip(), {k: v.strip() for k, v in zip(parts[1::2], parts[2::2])}


def mmstar(split="val"):
    """MMStar: one multiple-choice question per image."""
    skipped = 0
    for ex in load_dataset("Lin-Chen/MMStar", split=split):
        instructions, opts = _mmstar_question(ex["question"])
        if len(opts) < 2 or ex["answer"] not in opts:
            skipped += 1
            continue
        q = {"type": "choice", "instructions": instructions, "criteria": opts}
        yield {"state": "Look at the image.", "images": [_cap_image(ex["image"])], "questions": {"q": q}}, ex["answer"]
    if skipped:
        print(f"mmstar: skipped {skipped} questions whose options could not be parsed")


def _tempcompass(configs, split="test"):
    repo = "lmms-eval/TempCompass"
    videos = _video_index(repo, "tempcompass_videos.zip")
    groups = {}
    for config in configs:
        for i, ex in enumerate(load_dataset(repo, config, split=split)):
            qs, labels = groups.setdefault(ex["video_id"], ({}, {}))
            qid = f"{config}_{i}"
            if config == "yes_no":
                qs[qid] = {"type": "noul", "instructions": ex["question"]}
                labels[qid] = int(ex["answer"].strip().lower().startswith("yes"))
                continue
            lines = ex["question"].split("\n")
            opts = _options_from_lines(lines[1:])
            key = next(iter(_options_from_lines([ex["answer"]])), None)
            if key not in opts:  # answer given as bare option text
                key = next((k for k, v in opts.items() if v == ex["answer"].strip()), None)
            if len(opts) < 2 or key is None:
                continue
            qs[qid] = {"type": "choice", "instructions": lines[0].strip(), "criteria": opts}
            labels[qid] = key
    for vid, (qs, labels) in groups.items():
        if vid in videos and qs:
            yield {"state": "Watch the video.", "videos": [videos[vid]], "questions": qs}, labels


def tempcompass_mc(split="test"):
    """TempCompass: temporal multiple-choice and caption-matching questions, one request per clip."""
    yield from _tempcompass(["multi-choice", "caption_matching"], split)


def tempcompass_yn(split="test"):
    """TempCompass: temporal yes/no questions, one request per clip."""
    yield from _tempcompass(["yes_no"], split)


def nextqa(split="test", max_frames=1800):
    """NExT-QA multiple choice, one request per clip; only clips of at most `max_frames` frames (~60 s at 30 fps)."""
    repo = "lmms-eval/NExTQA"
    videos = _video_index(repo, "videos.zip")
    groups = {}
    for ex in load_dataset(repo, "MC", split=split):
        if ex["frame_count"] > max_frames:
            continue
        qs, labels = groups.setdefault(str(ex["video"]), ({}, {}))
        qid = f"q{ex['qid']}"
        q = ex["question"].strip()
        qs[qid] = {"type": "choice", "instructions": q[:1].upper() + q[1:] + "?",
                   "criteria": {k: ex[f"a{i}"] for i, k in enumerate("ABCDE")}}  # fmt: skip
        labels[qid] = "ABCDE"[ex["answer"]]
    for vid, (qs, labels) in groups.items():
        if vid in videos:
            yield {"state": "Watch the video.", "videos": [videos[vid]], "questions": qs}, labels


MEDIA = {"pope", "mmstar", "tempcompass_mc", "tempcompass_yn", "nextqa"}
MULTI_QUESTION = {"race", "go_emotions", "pope", "tempcompass_mc", "tempcompass_yn", "nextqa"}
# Held-out split per dataset for fitting calibration (never the split the test numbers come from).
CALIBRATION_SPLITS = {
    "arc_challenge": "validation", "arc_easy": "validation", "commonsense_qa": "train", "ag_news": "train",
    "boolq": "train", "sst5": "validation", "race": "validation", "go_emotions": "validation",
    # The media sets have no labelled held-out split here: calibrate on a disjoint seed's sample instead.
    "pope": "test", "mmstar": "val", "tempcompass_mc": "test", "tempcompass_yn": "test", "nextqa": "test",
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
    "pope": pope,
    "mmstar": mmstar,
    "tempcompass_mc": tempcompass_mc,
    "tempcompass_yn": tempcompass_yn,
    "nextqa": nextqa,
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
    parser.add_argument("--warmup", type=int, default=0, help="run this many requests untimed first (kernel compile/autotune)")
    parser.add_argument("--batch-requests", type=int, default=1, help="local scorers: requests prefilled and branched together")
    parser.add_argument("--no-single-pass", action="store_true", help="local scorers: fork even for one-question requests")
    parser.add_argument("--cuda-graphs", action="store_true", help="local scorers: replay branch forwards as CUDA graphs")
    parser.add_argument("--video-pruning-rate", type=float, default=0.0, help="local scorers: EVS q in [0, 1) for video tokens")
    parser.add_argument("--video-fps", type=float, default=2.0, help="video frames sampled per second (max --video-max-frames)")
    parser.add_argument("--video-max-frames", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=None, help="local scorers: max branch rows per forward (default 32)")
    parser.add_argument("--gpu-usd-per-hour", type=float, default=None, help="local / vllm: GPU price, for $ per 1M requests (L4 g6.2xlarge on-demand: 0.9776)")
    parser.add_argument("--vllm-url", default=None, help="vLLM server for --scorers vllm (default $VLLM_BASE_URL or localhost:8000)")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    local = [s for s in args.scorers.split(",") if not is_remote(s)]
    vision = any(d in MEDIA for d in args.datasets.split(","))
    backbone = Backbone.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype) if args.dtype else None, device=args.device, device_map=args.device_map,
        quantization=args.quantization, vision=vision,
    ) if local else None
    if backbone and args.chunk_size:
        backbone.chunk_size = args.chunk_size
    if backbone:
        backbone.cuda_graphs = args.cuda_graphs
        backbone.single_pass = not args.no_single_pass
        backbone.media.video_pruning_rate = args.video_pruning_rate
        backbone.media.fps, backbone.media.max_frames = args.video_fps, args.video_max_frames
        backbone.media.image_max_pixels, backbone.media.video_max_side = IMAGE_MAX_PIXELS, VIDEO_MAX_SIDE
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
        if name in MEDIA:
            examples = [prepare_media(e) for e in examples]
        for scorer_name in args.scorers.split(","):
            if args.warmup and not is_remote(scorer_name):  # untimed: Triton/fla compile and autotune per new shape
                evaluate(examples[: args.warmup], scorers[scorer_name](), 1, args.batch_requests)
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
