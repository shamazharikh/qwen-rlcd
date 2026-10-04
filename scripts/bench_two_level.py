"""Two-level fork (state → question cache → answers) vs. one-level (question prefix inside every option).

Prints a markdown table of median latency and branch-token counts per (prefix length, #options).

    python scripts/bench_two_level.py --device cpu --tiny
    python scripts/bench_two_level.py --state-len 512 --option-counts 2,5,20,100
"""

import argparse
import statistics
import sys
import time

import torch

from scripts.bench_fork import load, measure, sync  # noqa: F401  (sync re-exported for symmetry)
from system_one.fork import forward_branches_all, forward_question_tree, prefill_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--state-len", type=int, default=512)
    parser.add_argument("--prefix-lens", default="16,64,256", help="question prefix lengths in tokens")
    parser.add_argument("--option-counts", default="2,5,20,100")
    parser.add_argument("--answer-len", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.dtype is None:
        ampere = args.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        args.dtype = "bfloat16" if ampere else "float32"
    args.dtype = getattr(torch, args.dtype)
    if args.device == "cpu":
        sys.modules.setdefault("fla", None)  # fla is Triton/CUDA-only; see tests/conftest.py

    model, vocab = load(args)
    gen = torch.Generator().manual_seed(0)
    hardware = torch.cuda.get_device_name() if args.device == "cuda" else args.device
    state = torch.randint(1, vocab, (1, args.state_len), generator=gen).to(args.device)
    print(f"\n{'tiny' if args.tiny else args.model} on {hardware}, {args.dtype}, state={args.state_len}, "
          f"answer_len={args.answer_len}, chunk_size={args.chunk_size}\n")
    print("| prefix tokens | options | one-level ms | two-level ms | speedup | branch tokens 1L → 2L |")
    print("|---|---|---|---|---|---|")
    for prefix_len in map(int, args.prefix_lens.split(",")):
        prefix = torch.randint(1, vocab, (prefix_len,), generator=gen).tolist()
        for k in map(int, args.option_counts.split(",")):
            answers = torch.randint(1, vocab, (k, args.answer_len), generator=gen).tolist()
            flat = [prefix + a for a in answers]
            with torch.no_grad():
                cache = prefill_state(model, state)  # shared by both paths: we time the branches only
                one = measure(
                    lambda: forward_branches_all(model, cache, args.state_len, flat, 0, args.chunk_size),
                    args.device, args.repeats,
                )[0]
                two = measure(
                    lambda: forward_question_tree(
                        model, cache, args.state_len, [(prefix, answers)], 0, args.chunk_size
                    ),
                    args.device, args.repeats,
                )[0]
            tokens_one, tokens_two = sum(map(len, flat)), prefix_len + k * args.answer_len
            print(f"| {prefix_len} | {k} | {one:.0f} | {two:.0f} | {one / two:.1f}× | "
                  f"{tokens_one} → {tokens_two} ({tokens_one / tokens_two:.1f}×) |", flush=True)


if __name__ == "__main__":
    main()
