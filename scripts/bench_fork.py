"""M0 benchmark: fork (prefill once + batched branches) vs. one full forward per branch.

Prints a markdown table of median latency and peak memory per (state length, #branches).

    python scripts/bench_fork.py                       # Qwen3.5-0.8B-Base, CUDA
    python scripts/bench_fork.py --tiny --device cpu   # smoke test without downloading weights
"""

import argparse
import statistics
import time

import torch

from system_one.fork import fork_forward, sequential_reads


def load(args):
    if args.tiny:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        config = Qwen3_5TextConfig(
            vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, linear_num_key_heads=4, linear_num_value_heads=4,
            linear_key_head_dim=16, linear_value_head_dim=16,
        )  # fmt: skip
        return Qwen3_5TextModel(config).eval().to(args.device, args.dtype), 512
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=args.dtype).eval().to(args.device)
    return model.model.language_model, model.config.text_config.vocab_size


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def measure(fn, device, repeats):
    fn()  # warmup: kernel autotuning, allocator growth
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
    return statistics.median(times) * 1000, peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default=None, choices=["float32", "bfloat16"], help="default: bf16 on Ampere+")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--state-lens", default="512,2048")
    parser.add_argument("--branch-counts", default="1,10,50,100")
    parser.add_argument("--branch-len", type=int, default=24)
    parser.add_argument("--max-sequential", type=int, default=50, help="skip the sequential baseline above this")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.dtype is None:
        ampere = args.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        args.dtype = "bfloat16" if ampere else "float32"
    args.dtype = getattr(torch, args.dtype)

    model, vocab = load(args)
    gen = torch.Generator().manual_seed(0)
    hardware = torch.cuda.get_device_name() if args.device == "cuda" else args.device
    print(f"\n{'tiny' if args.tiny else args.model} on {hardware}, {args.dtype}, branch_len={args.branch_len}\n")
    print("| state tokens | branches | fork ms | fork peak GiB | sequential ms | sequential peak GiB | speedup |")
    print("|---|---|---|---|---|---|---|")
    for state_len in map(int, args.state_lens.split(",")):
        state = torch.randint(1, vocab, (1, state_len), generator=gen).to(args.device)
        for n in map(int, args.branch_counts.split(",")):
            branches = torch.randint(1, vocab, (n, args.branch_len), generator=gen).tolist()
            with torch.no_grad():
                fork_ms, fork_mem = measure(lambda: fork_forward(model, state, branches, 0), args.device, args.repeats)
                if n <= args.max_sequential:
                    seq_ms, seq_mem = measure(lambda: sequential_reads(model, state, branches), args.device, args.repeats)
                    seq = f"{seq_ms:.1f} | {seq_mem:.2f} | {seq_ms / fork_ms:.1f}x"
                else:
                    seq = "– | – | –"
            print(f"| {state_len} | {n} | {fork_ms:.1f} | {fork_mem:.2f} | {seq} |", flush=True)


if __name__ == "__main__":
    main()
