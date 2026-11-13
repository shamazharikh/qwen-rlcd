# System One Decision Model — Prototype Brief (text + multimodal)

A prototype of a "System One" decision model, inspired by TypeSafe AI's **Jev**. The model takes a **state** (context) and a set of **typed questions**. In a single forward pass it returns **calibrated probability distributions** over a fixed set of answers, with no text generation.

> **Status of claims in this doc:** Anything under "Public facts" comes from TypeSafe's docs and launch coverage. Everything else is **our hypothesis** of how such a model could work. TypeSafe has not published Jev's architecture.

---

## Project status (2026-09-24)

The implementation targets **`Qwen/Qwen3.5-0.8B-Base`**. See [`PLAN.md`](PLAN.md) for milestones. Current scope is **inference only**; training and fine-tuning (PLAN.md M2) are on hold.

| Milestone | Status |
|---|---|
| **M0** backbone spike (prefix fork) | ✅ Done. Fork = full forward on real weights (CPU, CUDA + fla, fp32/fp16); 8–19× faster than one forward per branch; chunked branches bound memory. Gradient-through-cache check deferred until training resumes. |
| **M1** templates, heads, typed API, zero-shot baselines | 🟡 Inference slice done: `predict()` with typed answers, letter / likelihood / PMI / `<|read|>`-head scorers, order and fan-out invariance tests, zero-shot numbers on 6 datasets (L4, n = 500), with Google's Gemini Decisions API as a hosted reference. Remaining: two-level fork. |
| **M2** training (LoRA + heads) | ⏸ On hold |
| **M3** calibration and eval | Metrics (`metrics.py`: acc, NLL, Brier, ECE-15, MAE) in place; temperature fitting and stress suite not started |
| **M4** serving / **M5** multimodal | Not started |

**Environment:** the dev box is 2× RTX 2080 Ti (11 GB, sm75). GPU 0 fell off the PCIe bus during a dual-GPU benchmark on 2026-09-24, and CUDA init fails on both cards until a reboot or root reset. Run one GPU job at a time on this box. GPU runs since 2026-11-13 use an NVIDIA L4 (24 GB, sm89) in a Kubernetes dev pod: `pip install uv && uv venv .venv && uv pip install -e '.[dev,eval,cuda]'`.

### Key design change: prefix-fork instead of a tree mask
Qwen3.5-0.8B is a hybrid model: 24 layers = **18 Gated DeltaNet (linear attention) + 6 gated full attention**. A tree attention mask (§2.2, §5.4 below) cannot isolate sibling branches inside recurrent layers, so this repo uses **prefix-fork execution** instead:

1. Prefill the state once and keep its cache (KV for attention layers, conv + recurrent state for DeltaNet layers).
2. Copy that cache along the batch dimension, one row per branch.
3. Run every `question + answer` branch as one right-padded batch at the same position offset, and read the hidden state at each branch's last real token.

Branches are isolated in every layer type by construction, and results can't depend on question or option order.

### What's built (M0)
| Path | What it does |
|---|---|
| `system_one/fork.py` | `prefill_state`, `expand_cache`, `forward_branches`, `fork_forward` (optional `chunk_size` to bound memory), plus a `sequential_reads` reference (one uncached full forward per branch) |
| `tests/test_fork_equivalence.py` | Fork vs. sequential equivalence on a tiny random hybrid model and on real 0.8B weights; covers padding and sibling independence, conv-kernel and delta-rule chunk edge cases, source-cache immutability, and sensitivity checks that must fail when DeltaNet state isn't forked. Runs on CUDA when available. |
| `scripts/bench_fork.py` | Latency and peak memory: fork vs. one forward per branch, across state lengths and branch counts |
| `notebooks/m0_gpu_checks.ipynb` | Colab notebook: installs `flash-linear-attention`, runs the tests on GPU, then the benchmark |

### M0 results: Mac CPU (fp32, torch fallback kernels)
| Check | Max abs diff |
|---|---|
| Tiny model, fork vs. sequential | ~7e-7 (tolerance 1e-5) |
| **Qwen3.5-0.8B-Base, fork vs. sequential** | **4.8e-5** (tolerance 1e-3) |
| 0.8B with forked recurrent state zeroed (negative control) | 6.3 |
| 0.8B with forked conv state zeroed (negative control) | 3.1 |

### M0 results: GPU (2026-09-24, RTX 2080 Ti 11 GB, sm75, fla 0.5.2, torch 2.14, no `causal-conv1d`)
All fast tests pass on CUDA with both the torch fallback and the fla DeltaNet kernels (fla is picked up automatically once installed).

| Check | Result |
|---|---|
| **Qwen3.5-0.8B-Base, fork vs. sequential, fp32, fla** | max abs diff **2.3e-5** (tolerance 1e-3), min cos 1.000000 |
| Same, fp16 (Turing has no bf16) | max abs diff 4.7e-2, min cos 0.999998 |

Benchmark (`scripts/bench_fork.py --chunk-size 25`, 24-token branches, median of 5). Sequential = one full forward per branch, measured up to 10 branches.

| state tokens | branches | fork ms fp32 | fork ms fp16 | fork peak GiB fp32 | sequential ms fp32 | speedup (fp32) |
|---|---|---|---|---|---|---|
| 512 | 1 | 118 | 108 | 2.91 | 108 | 0.9× |
| 512 | 10 | 135 | 126 | 3.25 | 1137 | 8.4× |
| 512 | 50 | 394 | 361 | 3.87 | (7101 unchunked) | 19× |
| 512 | 200 | 1378 | 1256 | 3.87 | – | – |
| 2048 | 10 | 535 | 298 | 3.87 | 4618 | 8.6× |
| 2048 | 50 | 777 | 555 | 5.38 | – | – |
| 2048 | 200 | 1963 | 1545 | 5.38 | – | – |
| 4096 | 10 | 895 | 553 | 4.70 | 9976 | 11.2× |
| 4096 | 200 | 3196 | – | 7.39 | – | – |

### M0 results: GPU (2026-11-13, NVIDIA L4 24 GB, sm89, fla 0.6.0, torch 2.15 / CUDA 13.2, no `causal-conv1d`)
All tests pass, fast and real-weight, once fla's TF32 is turned off for fp32 (`system_one/kernels.py`, see findings).

| Check | Result |
|---|---|
| **Qwen3.5-0.8B-Base, fork vs. sequential, fp32, fla (IEEE)** | max abs diff **2.1e-5** (tolerance 1e-3), min cos 1.000000 |
| Same, fla with its default TF32 | max abs diff 2.0e-2 (fails) |
| Same, bf16 | min cos 0.999887. bf16 fork and bf16 sequential are each 0.99993–0.99994 from fp32, so the fork adds no error beyond bf16 rounding |

Benchmark (`scripts/bench_fork.py --chunk-size 25`, 24-token branches). The fp32 column uses Triton's default TF32, since the script loads the model directly rather than through `Backbone`.

| state tokens | branches | fork ms bf16 | fork ms fp32 | fork peak GiB bf16 | sequential ms bf16 | speedup (bf16) |
|---|---|---|---|---|---|---|
| 512 | 1 | 100 | 118 | 1.48 | 46 | 0.5× |
| 512 | 10 | 97 | 135 | 1.74 | 457 | 4.7× |
| 512 | 50 | 150 | 343 | 2.21 | 2332 | 15.6× |
| 512 | 100 | 256 | 617 | 2.21 | – | – |
| 2048 | 1 | 137 | 363 | 1.60 | 95 | 0.7× |
| 2048 | 10 | 139 | 402 | 2.05 | 958 | 6.9× |
| 2048 | 50 | 247 | 695 | 2.96 | 4805 | 19.4× |
| 2048 | 100 | 403 | 1099 | 2.96 | – | – |

Unlike Turing, bf16 here is 2–3× faster than fp32 for 50+ branches. With 1 branch the fork is slower than a plain forward, because it pays for a separate prefill.

### M0 findings
- **On Ampere+ (sm ≥ 8), fla's "fp32" is really TF32.** Triton runs fp32 `tl.dot` as TF32 by default, and fla 0.6 also hard-codes `input_precision='tf32'` in the gated-delta-rule `solve_tril` kernel (`SOLVE_TRIL_DOT_PRECISION`), so `TRITON_F32_DEFAULT=ieee` alone isn't enough. Together they put fp32 fork and sequential reads 2e-2 apart on an L4. `system_one.kernels.disable_fla_tf32()` sets the env vars and patches that constant. `Backbone.from_pretrained` calls it for fp32 on CUDA, and so does `tests/conftest.py`. The constant has to be patched after `fla.ops` is imported, because faking an sm < 8 capability at import makes Triton build kernels for the wrong arch. Turing never showed this, since it has no TF32.
- **fla 0.5.2 doesn't import on Triton 3.9** (autotune key `BT` error). transformers then silently falls back to the torch kernels, so a passing fp32 test can mean fla never ran. Check `fla.__version__` imports cleanly before trusting GPU numbers.
- **fla 0.6.0's `fused_recurrent_gated_delta_rule` with an `initial_state` from `chunk_gated_delta_rule` returned NaN** in a standalone kernel check (S ≥ 64). The cause hasn't been looked into; it may be a state-layout mismatch, since 0.6 kernels have a `STATE_V_FIRST` flag. transformers only calls it for single-token decode, which the fork never does, but keep it in mind before using `fused_recurrent` for short branches.
- **`DynamicCache.batch_repeat_interleave` can't fork this model** (transformers 5.17). `LinearAttentionLayer` has no such method, and `LinearAttentionAndFullAttentionLayer` inherits `DynamicLayer`'s, which repeats only keys and values. Hence the custom `expand_cache`.
- **The cache updates DeltaNet state in place** (`copy_`), so backward through a forked cache fails. This only matters for training, which is out of scope for now.
- **Tiny random models barely use DeltaNet state.** Zeroing it moves reads by only ~1e-4, so tiny-model tests use a 1e-5 tolerance to stay sensitive.
- **Colab (T4):** compiling `causal-conv1d` ran the VM out of memory and killed the runtime twice, so the notebook now installs only `flash-linear-attention`. A third attempt couldn't connect to a runtime at all.
- **Unchunked forks run out of memory on long states.** Every branch row gets its own copy of the attention-layer KV (and SDPA's `repeat_kv` copies it again), so memory grows with #branches × state length: 2048 tokens × 50 branches OOMs on 11 GB. `forward_branches(..., chunk_size=N)` caps the rows per forward; with `chunk_size=25`, peak memory is flat in the branch count (table above). `expand` instead of copy wouldn't help, because `DynamicLayer.update` concatenates and materialises the KV anyway.
- **On Turing, fla's DeltaNet kernel is the bottleneck, not the GEMMs.** Profiling a 2048-token state × 50 branches in fp16: `ChunkGatedDeltaRule` is 64% of GPU time (mostly `chunk_fwd_kernel_o`), linear layers 13%, attention 4%. That's why fp16 is barely faster than fp32 for short states. For 24-token branches the torch fallback was actually faster (512 × 50: 275 ms vs 361 ms fla), probably because the chunk kernel pads each short branch to a 64-token chunk. Recheck on Ampere+ before choosing a kernel per phase (fla for the state prefill, and possibly `fused_recurrent` or torch for short branches).
- **With fla installed, the model can't run on CPU.** transformers binds DeltaNet to fla at import time whatever the device, and Triton then fails with "0 active drivers". `tests/conftest.py` hides fla when CUDA is unavailable. Do the same (`sys.modules["fla"] = None` before importing transformers) in any CPU script.
- **Hardware incident:** running benchmarks on both 2080 Tis at once made GPU 0 fall off the bus (`nvidia-smi`: "Unable to determine the device handle ... Unknown Error"). After that, CUDA init failed on both GPUs until a reset. Run one GPU job at a time on this box.

### What's built (M1, inference only)
| Path | What it does |
|---|---|
| `system_one/schema.py` | Parses and validates the README §5.2 request (`Request.from_dict`). Typed `ChoiceAnswer` / `ScoreAnswer` / `NoulAnswer` |
| `system_one/templates.py` | State, question-prefix and answer-branch text, with the compact option list inside every branch (§4). Choice options are always rendered in sorted key order, so outputs don't depend on the caller's order. Letter prompts for the baseline |
| `system_one/scorers.py` | `Backbone` (model + tokenizer + fork) and three scorers, all running every branch of every question through one prefix fork: `LetterScorer` (letter-logit reading, §7.7 #1), `LikelihoodScorer` (sum/mean answer log-likelihood, optional PMI against an empty state, §7.7 #2), `HeadScorer` (linear heads at a `<|read|>` token; heads untrained until M2) |
| `system_one/predict.py` | `predict(request, scorer, temperature=None)` → typed answers: softmax / sigmoid, confidence = 1 − H(p)/log K, score = Σ p_i·i (0-based) |
| `system_one/metrics.py` | Accuracy, NLL, Brier, ECE-15, and MAE for Score |
| `tests/test_predict.py` | Schema validation, rendering independent of option order, answer well-formedness, **question/option-order invariance and fan-out = single-question answers for every scorer** (< 1e-5), fork likelihood = unforked likelihood, PMI arithmetic, `<|read|>` fits in the embedding |
| `scripts/zero_shot_eval.py` | Zero-shot baselines through `predict` on ARC-Challenge/Easy, CommonsenseQA, AG News (choice), BoolQ (noul), SST-5 (score). `--scorers gemini` adds Google's hosted Decisions API as a reference |
| `system_one/remote.py` | `GeminiDecisions` (Google Decisions API, `gemini-decision-1`, needs `GEMINI_API_KEY`) and `VllmSystemOne` (vLLM `/v1/systemone`). Same request shape as §5.2; both return this repo's typed answers |

### M1 results: zero-shot baselines
`scripts/zero_shot_eval.py --limit 500` (2026-11-13): Qwen3.5-0.8B-Base, fp32 (IEEE) on an L4, plus `--scorers gemini`. n = 500 random examples per dataset (seed 0), the same examples for every scorer. The 95% CI on accuracy is about ±0.04. Bold = best local scorer per column. Gemini Decision 1 is a hosted model of unknown size, included as a ceiling reference, not a like-for-like baseline. Its yes/no outputs are clamped to [0.01, 0.99].

| dataset (type, K) | scorer | acc | NLL | Brier | ECE-15 | MAE |
|---|---|---|---|---|---|---|
| ARC-Challenge (choice, 4–5) | **letter** | **0.612** | **0.949** | **0.504** | **0.053** | – |
| | sum / mean | 0.328 / 0.362 | 3.839 / 1.384 | 1.062 / 0.747 | 0.454 / 0.076 | – |
| | sum-pmi / mean-pmi | 0.384 / 0.376 | 2.247 / 1.330 | 0.916 / 0.722 | 0.364 / 0.067 | – |
| | *Gemini Decision 1* | *0.968* | *0.128* | *0.053* | *0.038* | – |
| ARC-Easy (choice, 4–5) | **letter** | **0.756** | **0.625** | **0.320** | **0.097** | – |
| | sum / mean | 0.652 / 0.642 | 1.702 / 1.024 | 0.537 / 0.552 | 0.167 / 0.197 | – |
| | sum-pmi / mean-pmi | 0.598 / 0.584 | 1.308 / 1.089 | 0.579 / 0.590 | 0.183 / 0.177 | – |
| | *Gemini Decision 1* | *0.990* | *0.058* | *0.018* | *0.035* | – |
| CommonsenseQA (choice, 5) | letter | **0.496** | **1.245** | **0.625** | **0.056** | – |
| | sum / mean | 0.382 / 0.440 | 2.151 / 1.369 | 0.894 / 0.695 | 0.332 / **0.056** | – |
| | sum-pmi / mean-pmi | 0.484 / **0.496** | 1.539 / 1.326 | 0.712 / 0.675 | 0.203 / 0.127 | – |
| | *Gemini Decision 1* | *0.860* | *0.387* | *0.195* | *0.093* | – |
| AG News (choice, 4) | **letter** | **0.654** | **0.859** | **0.466** | 0.136 | – |
| | sum / mean | 0.562 / 0.376 | 2.157 / 1.282 | 0.661 / 0.695 | 0.226 / 0.161 | – |
| | sum-pmi / mean-pmi | 0.566 / 0.572 | 1.601 / 1.001 | 0.721 / 0.560 | 0.321 / **0.096** | – |
| | *Gemini Decision 1* | *0.890* | *0.321* | *0.167* | *0.020* | – |
| BoolQ (noul) | letter (yes/no logits) | **0.770** | **0.540** | **0.359** | 0.107 | – |
| | sum = mean | 0.758 | 0.541 | 0.360 | **0.093** | – |
| | sum-pmi = mean-pmi | 0.760 | 0.546 | 0.365 | 0.099 | – |
| | *Gemini Decision 1* | *0.926* | *0.220* | *0.125* | *0.021* | – |
| SST-5 (score, 5) | letter | 0.202 | 2.215 | 1.012 | 0.392 | 1.437 |
| | sum / mean | 0.272 / 0.186 | 1.640 / 1.655 | 0.797 / 0.819 | **0.066** / 0.089 | **1.084** / 1.121 |
| | sum-pmi / mean-pmi | **0.306** / 0.304 | **1.568** / 1.585 | 0.791 / **0.790** | 0.138 / 0.082 | 1.102 / 1.122 |
| | *Gemini Decision 1* | *0.542* | *0.952* | *0.567* | *0.091* | *0.501* |

- **Letter-logit is the best local scorer on every choice dataset**, and on BoolQ too, now that n = 500 tightens the CI. At n = 100, PMI looked best on BoolQ (0.79), but at n = 500 all scorers are within 0.012 there.
- **PMI helps only where the answer prior is skewed.** On SST-5 it is the best local scorer (0.31 acc). On ARC it recovers summed likelihood's length bias (NLL 3.84 → 2.25) but still trails letter-logit by 0.23.
- **The gap to Gemini is large on knowledge tasks** (ARC-Challenge 0.61 vs 0.97) and smaller on BoolQ (0.77 vs 0.93). SST-5 is hard for both (0.31 vs 0.54), so trained Score heads (M2) should be judged against that ceiling, not against 1.0.
- **Gemini is well calibrated out of the box** (ECE ≤ 0.04 on 4 of 6). The local scorers' ECE ranges from 0.05 to 0.39, which is the case for the M3 temperature fitting.
- Local cost on the L4 in fp32: 0.09–0.12 s/example (0.20–0.23 s for PMI, which runs two forks). Gemini: 0.09 s/example wall time with 8 concurrent requests.
- The earlier CPU n = 100 numbers are in git history (`8270d63`).

### Larger backbones, vLLM and Gemini (n = 500, 2026-11-13)
Same 500 examples per dataset (seed 0) for every column. Cells are accuracy / NLL / ECE-15. Our models use the `letter` scorer (the best or within 0.01 of the best local scorer, except SST-5 and BoolQ, noted below). 9B ran in bf16 on an L4, and 35B-A3B in FP8 (`quantization="fp8"`, the default) on an L40S. vLLM 0.32 `POST /v1/systemone` served the same Base checkpoints (bf16, L4), one request at a time.

| dataset | 0.8B | 9B | 35B-A3B (FP8) | vLLM 0.8B | vLLM 9B | Gemini Decision 1 |
|---|---|---|---|---|---|---|
| ARC-Challenge | 0.612 / 0.95 / 0.053 | 0.928 / 0.22 / 0.040 | **0.948** / 0.20 / 0.028 | 0.462 / 1.29 / 0.164 | 0.928 / 0.21 / 0.029 | *0.968 / 0.13 / 0.038* |
| ARC-Easy | 0.756 / 0.62 / 0.097 | 0.982 / 0.09 / 0.048 | 0.982 / 0.09 / 0.043 | 0.626 / 0.92 / 0.067 | **0.984** / 0.06 / 0.019 | *0.990 / 0.06 / 0.035* |
| CommonsenseQA | 0.496 / 1.25 / 0.056 | 0.816 / 0.53 / 0.059 | 0.812 / 0.54 / 0.042 | 0.416 / 1.45 / 0.105 | **0.822** / 0.49 / 0.054 | *0.860 / 0.39 / 0.093* |
| AG News | 0.654 / 0.86 / 0.136 | **0.870** / 0.42 / 0.064 | 0.822 / 0.49 / 0.073 | 0.738 / 0.72 / 0.136 | 0.860 / 0.52 / 0.101 | *0.890 / 0.32 / 0.020* |
| BoolQ | 0.770 / 0.54 / 0.107 | 0.886 / 0.27 / 0.028 | **0.894** / 0.26 / 0.035 | 0.762 / 0.48 / 0.044 | 0.886 / 0.29 / 0.024 | *0.926 / 0.22 / 0.021* |
| SST-5 (MAE) | 0.202 / 2.22 / 0.392 (1.44) | 0.480 / 1.19 / 0.153 (0.57) | 0.418 / 1.45 / 0.267 (0.62) | 0.238 / 1.91 / 0.264 (1.24) | **0.500** / 1.11 / 0.075 (0.58) | *0.542 / 0.95 / 0.091 (0.50)* |
| s/example | 0.09–0.10 (fp32) | 0.17–0.19 | 0.35–0.57 | 0.03–0.04 | 0.09–0.11 | 0.09 (8 concurrent) |

Other local scorers that beat `letter`: SST-5 sum-pmi 0.306 (0.8B) / 0.484 (9B), sum 0.500 (A3B); BoolQ sum 0.894 (9B) / 0.898 (A3B).

- **9B gets within 0.01–0.06 of Gemini zero-shot** and is mostly well calibrated (ECE 0.03–0.06, except SST-5). It's the practical pick: 35B-A3B (FP8) beats it only on ARC-Challenge (+0.02, inside the ±0.04 CI), trails on AG News (−0.05) and SST-5 letter, and runs 2–3× slower here (eager FP8 MoE in transformers, no fused kernels; 40 layers vs 32).
- **vLLM `/v1/systemone` is the same idea as `letter`** (one read of a single label token, softmax over labels) with a different prompt: chat template with thinking off, options as `A: name - description`. At 9B the two are tied on every dataset (±0.02), with vLLM slightly better calibrated. At 0.8B ours wins on knowledge MC by 0.08–0.15 and vLLM wins on AG News by 0.08: the small model is prompt-sensitive, the 9B isn't.
- **vLLM is ~2× faster per single-question request** (9B: 0.1 s vs 0.18 s, both bf16): CUDA graphs and fused kernels vs our eager path. These runs have one question per state, so they don't exercise the fork's advantage (one state prefill shared by many questions), nor vLLM's limits (64 questions, 26 options, 10 levels, one read per question). A many-questions-per-state benchmark is the next comparison.
- Benchmarks (`bench_fork.py --chunk-size 10`, 24-token branches): 9B bf16 on L4, 2048 × 50 branches in 1.47 s at 16.4 GiB, 8.7× faster than sequential at 10 branches. 35B-A3B FP8 on L40S, 2048 × 50 in 1.02 s at 33.7 GiB, 5.1× at 10 branches.

**FP8 and MoE make reads depend on batch shape.** On real 35B-A3B weights (FP8), fork vs sequential reads differ by 1 − cos = 6.1e-3, against ~1e-4 for the dense models in bf16. A diagnostic (router top-8 at each read token, L40S) shows the fork is not the cause:

| comparison (A3B, FP8) | 1 − cos | layers with an expert flip (of 40) |
|---|---|---|
| batched fork vs sequential | 2.5e-3 – 6.1e-3 | 11–14 |
| single-branch fork vs sequential (cache only) | 1.0e-3 – 3.5e-3 | 6–16 |
| batched vs single-branch fork (batch shape only) | 2.1e-3 – 4.2e-3 | 12–18 |
| sequential vs sequential | 0 | 0 |

Changing only the batch shape moves reads as much as the fork does, and identical shapes are bit-exact. The source is FP8: dense models in FP8 already drift 17–18× more than in bf16 (fork vs sequential 1 − cos: 0.8B 2.7e-3, 9B 1.7e-3, vs 1.5e-4 / 1.0e-4 in bf16), because quantizing activations to coarse steps turns rounding-level differences into step-sized jumps. MoE then flips about one of the top-8 experts in a third of the layers. Next-token distributions move by 0.04–0.09 total variation, enough to flip near-tied answers. The real-weight test therefore checks MoE models against their own batch-shape noise (fork within 3× of batched-vs-single-branch fork) instead of a fixed cosine. For batch-invariant FP8/MoE answers, a deterministic serving mode would be needed (vLLM has `VLLM_BATCH_INVARIANT=1`).

### 9B vs Gemini chat-completion baselines, with cost per call (n = 500, 2026-11-14)
Same examples as above. Gemini 2.5 Flash-Lite and 3.1 Flash-Lite run through Gemini's OpenAI-compatible `/chat/completions` (`--scorers chat:<model>`, `system_one/chat.py`): lettered options, "answer with the letter only", temperature 0, no thinking tokens used. That endpoint rejects `logprobs`, so the chat models return hard labels only, with no probabilities or calibration. API scorers ran with 16 concurrent requests.

| dataset | 9B `letter` (fork) | vLLM 9B (16 concurrent) | Gemini 2.5 Flash-Lite (chat) | Gemini 3.1 Flash-Lite (chat) | Gemini Decision 1 |
|---|---|---|---|---|---|
| ARC-Challenge | 0.928 | 0.928 | 0.932 | 0.964 | **0.968** |
| ARC-Easy | 0.982 | 0.982 | 0.972 | **0.992** | 0.990 |
| CommonsenseQA | 0.816 | 0.814 | 0.810 | **0.870** | 0.860 |
| AG News | 0.870 | 0.854 | 0.806 | 0.872 | **0.890** |
| BoolQ | 0.886 | 0.878 | 0.892 | 0.900 | **0.926** |
| SST-5 (MAE) | 0.480 (0.57) | 0.490 (0.58) | 0.528 (0.54) | **0.574 (0.46)** | 0.542 (0.50) |
| probabilities (ECE-15) | yes (0.03–0.15) | yes (0.02–0.09) | no | no | yes (0.02–0.09) |
| **$ per 1M calls** | 45–52 (1 at a time) | **8–15** (mean 10) | **9–18** | 23–45 | 13–22 |

Cost per call: self-hosted = on-demand price ÷ 3600 × wall-clock seconds per call (L4 g6.2xlarge $0.9776/h). APIs = billed prompt and completion tokens × Google list price (2.5 Flash-Lite $0.10 / $0.40 per M, 3.1 Flash-Lite $0.25 / $1.50, Decision 1 $0.10 input only). Per-dataset spread comes mostly from input length (BoolQ passages are longest).

- **Concurrency matters for self-hosting.** vLLM 9B goes from $24.5 per 1M calls one at a time to $15.8 with 4 concurrent and $10.0 with 16 (0.03–0.06 s per call), with unchanged accuracy (±0.01). That's on par with 2.5 Flash-Lite and below Decision 1, on an on-demand L4 (spot or reserved is 20–37% cheaper). Our fork still serves one request at a time; batching requests is the obvious next step (prefill states together, then run all requests' branches in one batch).
- **3.1 Flash-Lite is the most accurate chat baseline** (best on SST-5, CommonsenseQA and ARC-Easy) but costs 2–3× more than 2.5 Flash-Lite or Decision 1, and gives no confidence to threshold on.
- **9B (zero-shot) ≈ 2.5 Flash-Lite on accuracy** (−0.05 to +0.06 per dataset) and adds calibrated probabilities. Gemini Decision 1 stays the best accuracy-per-dollar API with probabilities.
- Real Decision 1 billing is about 2× a token estimate made with the Qwen tokenizer: 125–220 billed tokens per call vs ~70 counted.
- One SST-5 review per chat model was safety-blocked (`finish_reason` not `stop`) and is scored wrong.

M1 design notes:
- `forward_branches_all` returns every branch token's hidden state (the likelihood scorers need them). `forward_branches` reads the last one.
- `<|read|>` gets id 248077. The tokenizer uses 248,077 ids but the embedding has 248,320 rows, so no resize is needed.
- The invariance tests are sensitive: rendering options in caller order makes all four scorers fail them (1e-1 to 3e-4 diffs).
- In the ARC and SST-5 converters, the question or review text is the state, so PMI's empty-state baseline is meaningful.

### Next steps
- [x] GPU zero-shot run at n = 500 on 6 datasets (L4), plus the Gemini Decisions API reference
- [ ] Large-K choice datasets (Banking77 / DBpedia-14 / CLINC150) for the K-scaling test, local, vLLM and Gemini
- [x] Qwen3.5-9B and 35B-A3B (FP8) backbones, and vLLM `/v1/systemone` comparison (n = 500)
- [ ] Many-questions-per-state benchmark: fork vs vLLM `/v1/systemone` vs Gemini, latency vs #questions (README §7.8)
- [ ] bf16 inference speed for the local scorers (eval runs used fp32 for 0.8B); fused MoE kernels for 35B-A3B
- [ ] Two-level fork (state → question cache → answers), so long question prefixes aren't recomputed per option (PLAN.md §6)
- [ ] Kernel choice per phase: fla for the state prefill, and torch or `fused_recurrent` for short branches (recheck on Ampere+)
- [ ] M3: temperature fitting per (type, K-bucket) on the zero-shot scorers, plus the stress suite (permutation, K scaling, fan-out, empty state)
- [ ] M2 (on hold): train `DecisionHeads` + LoRA so `HeadScorer` becomes the real model. Needs gradients through the forked cache, which in-place cache updates currently block (PLAN.md fallback A: per-row `[state, branch]` concatenation)

### Quickstart
```bash
uv venv --python 3.12 .venv && uv pip install -e '.[dev]'
.venv/bin/python -m pytest -q                                   # fast tests (tiny model)
QWEN_RLCD_SLOW=1 .venv/bin/python -m pytest -q -s -k qwen35     # real Qwen3.5-0.8B-Base weights
uv pip install -e '.[cuda]'                                      # CUDA only: fla DeltaNet kernels
.venv/bin/python scripts/bench_fork.py --chunk-size 25           # benchmark (use a CUDA GPU)
uv pip install -e '.[eval]' && .venv/bin/python scripts/zero_shot_eval.py --limit 200   # zero-shot baselines
```
```python
from system_one.predict import predict
from system_one.scorers import Backbone, LikelihoodScorer
scorer = LikelihoodScorer(Backbone.from_pretrained(), normalize="sum", pmi=True)
predict({"state": "...", "questions": {"is_urgent": {"type": "noul", "instructions": "The message conveys urgency."}}}, scorer)
```
GPU: [open the notebook in Colab](https://colab.research.google.com/github/shamazharikh/qwen-rlcd/blob/main/notebooks/m0_gpu_checks.ipynb), choose a GPU runtime, and click Run all.

---

## 1. Public facts about Jev (TypeSafe AI)

- It is a non-chat decision model. It returns typed answers and probabilities, not generated text.
- It has three question types:
  - **Choice** picks one option from a set. It returns `choice`, `probabilities`, and `confidence`.
  - **Score** picks a position on ordered levels. It returns `score` (which can fall *between* levels), `probabilities`, and `confidence`.
  - **Noul** answers a yes/no question. It returns `noul`, P(yes) in [0, 1], with no separate confidence.
- All questions in a request are evaluated **in parallel and in isolation** against the same state.
  - Adding questions barely increases latency.
  - Batching 13 questions into one call gave identical answers to 13 separate calls, at ~11.5x lower cost and ~9.6x faster.
- Question IDs are **not** sent to the model.
- A single Choice can hold hundreds of options (their cookbook scores 218 line IDs in one Choice).
- The token budget is ~32k tokens, shared by state and questions.
- Reported latency is ~150 ms per call. Pricing is ~$42 per billion input tokens, and **output is free**.
- Training uses **RLCD** (Reinforcement Learning for Calibrated Decisions). The objective is decisions plus calibrated probabilities. TypeSafe argues that RLHF hurts calibration through sycophancy and mode dropping.
- Confidence summarizes how peaked the probability distribution is.

Docs: https://docs.typesafe.ai/introduction · index: https://docs.typesafe.ai/llms.txt

---

## 2. Working hypothesis (architecture)

### 2.1 Prefill-only
Free output and per-input-token pricing imply **no decode loop**. A pretrained decoder LLM backbone runs one forward pass, and answers are read from hidden states at specific token positions.

Operationally this behaves like a cross-encoder or reranker:
- compute-bound,
- no KV cache held across decode steps,
- easy to batch.

### 2.2 Packed tree-structured attention
Everything is packed into one sequence:

```
[ state ........ ][ q1 ][ a1_1 ][ a1_2 ][ a1_3 ] ... [ q2 ][ a2_1 ][ a2_2 ] ...
```

| Token belongs to | Attends to |
|---|---|
| state | state (causal) |
| question `qi` | state + itself (causal) |
| answer `ai_k` | state + its parent `qi` + itself (causal) |

Siblings never see each other. This applies both to questions and to answers under the same question.

**Position ids:**
- state: `0 .. S-1`
- every question: starts at `S`
- every answer under `qi`: starts at `S + len(qi)`

Sibling branches therefore sit at **identical positions**, so their results cannot depend on order.

This is the prefix-shared form of the classic BERT/RoBERTa multiple-choice setup, where each (context, option) pair is encoded separately and a softmax runs across options.

Serving alternative: prefill the state once and serve the branches from the cached prefix (FlashInfer cascade attention, SGLang radix-tree prefix sharing, or vLLM prefix caching).

### 2.3 Answer heads
- **Choice:**
  - A linear head reads the hidden state at the **last token of each answer branch** and outputs logit `s_k`.
  - `p = softmax(s / T)` is taken **within each question group**.
- **Score:**
  - Same as Choice, over the ordered levels.
  - The returned value is the expected level, `score = Σ p_i · i`, which is why it can land between levels.
- **Noul:** either option works.
  - A sigmoid head on the question's last token (cheaper).
  - A 2-option Choice with `[yes]` and `[no]` branches (reuses the Choice path).
- **Do not** score branches by summed token log-likelihood. That reintroduces length bias and prior bias. If likelihood must be used, apply a PMI correction: subtract the branch's log-prob under an empty state.

### 2.4 Confidence
Normalized entropy: `confidence = 1 - H(p) / log K`.

This keeps confidence comparable across different option counts K. Top-1/top-2 margin is an alternative.

### 2.5 Training (hypothesis for RLCD)
- **Reward:** a strictly proper scoring rule (log score or Brier) on questions with known answers. Proper scoring rules are maximized only by reporting true beliefs, which is what produces calibration.
- **Relation to supervised learning:** with a softmax head and a log-score reward, this is close to supervised listwise cross-entropy. The RL framing likely matters for:
  - judge-ensemble labels,
  - synthetic question generation,
  - distribution-level rewards.
- **Starting point:** a **base** model, not an RLHF'd chat model.

---

## 3. Calibration notes

- **When raw probabilities work:** they are usable when the answer space is closed and each answer maps to a single read position.
- **Free-form sequence probabilities are poorly calibrated** because of:
  - length bias,
  - surface-form competition (probability split across paraphrases),
  - prior bias (answers likely regardless of input),
  - tokenization artifacts.
- **Post-hoc fix:** temperature scaling (one parameter `T`, fit by minimizing NLL on held-out data). Use vector or Dirichlet scaling if there is class-specific bias.
- **Fit temperature per `(question_type, K-bucket)`.** The softmax behaves differently as K grows, so a `T` fit on 3-option questions will not transfer to 200-option questions. As an alternative, condition `T` on `log K`.
- **Free-form answers:** use self-consistency plus semantic clustering (sum probability within meaning-equivalent clusters, as in semantic entropy), or reformulate as a verification Noul.
- **Evaluation metrics:**
  - ECE and reliability diagrams,
  - **plus** a proper score (NLL and Brier),
  - **plus** discrimination (accuracy and AUROC).
  - Calibration does not improve ranking, and a useless model that always predicts the base rate can still have good ECE.

---

## 4. Known risks and mitigations

| Risk | Why | Mitigation |
|---|---|---|
| Options can't see their alternatives | Each answer is judged in absolute terms; relative options ("partially" vs "mostly", "other", "none of the above") suffer | Put a compact list of all option labels **inside the question branch**; shuffle that list during training |
| Near-duplicate options (red bus / blue bus) | Independent logits mean duplicates split and double-count mass | Require mutually exclusive options; train on sets containing paraphrases; optionally add a small set-attention layer over the K answer embeddings before the softmax |
| Calibration drifts with K | Softmax behavior depends on K | Temperature per K-bucket or conditioned on `log K`; entropy normalized by `log K` |
| Non-monotone Score distributions | Independent level branches can produce bumpy distributions, e.g. `[0.4, 0.05, 0.4, 0.15]` | Ordinal-aware loss (CE + distance penalty, or neighbor label smoothing) or a cumulative-link head (one scalar + learned thresholds) |
| Position bias | Options listed inline get position effects | Tree mask with shared position ids (Section 2.2) |

---

## 5. Implementation guide

### 5.1 Suggested stack
- PyTorch ≥ 2.5 with **FlexAttention** (`torch.nn.attention.flex_attention`)
- Hugging Face `transformers`, with a small base decoder model (e.g. a Qwen3 base in the 0.6B–8B range)
- Optional for serving: vLLM pooling/classification mode with prefix caching, or SGLang for prefix sharing

### 5.2 Request / sample format
```json
{
  "state": "Our API started returning 500s 20 minutes ago; we can't process orders.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payment or subscription issues",
        "technical": "Bugs or integration problems",
        "sales": "Pricing or account questions"
      }
    },
    "is_urgent": { "type": "noul", "instructions": "The message conveys urgency." },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated does the customer appear?",
      "criteria": ["Calm", "Frustrated but civil", "Very angry"]
    }
  },
  "labels": { "department": "technical", "is_urgent": 1, "frustration": 1 }
}
```

### 5.3 Packing
For each sample, the packer produces per-token tensors:
- `input_ids`
- `position_ids` (per Section 2.2)
- `seg_id`:
  - `0` for state tokens,
  - a unique int > 0 for each question branch,
  - a unique int > 0 for each answer branch.
- `parent_id`:
  - `0` for state and question tokens,
  - the question's `seg_id` for answer tokens.
- `read_idx`: the index of the last token of each answer branch (Choice/Score) or question branch (Noul), plus a mapping to `(question_id, option_key)`.

### 5.4 Mask
```python
from torch.nn.attention.flex_attention import create_block_mask

# seg_id, parent_id: LongTensor [B, L]
def mask_mod(b, h, q_idx, kv_idx):
    causal    = kv_idx <= q_idx
    to_state  = seg_id[b, kv_idx] == 0
    to_self   = seg_id[b, kv_idx] == seg_id[b, q_idx]
    to_parent = seg_id[b, kv_idx] == parent_id[b, q_idx]
    return causal & (to_state | to_self | to_parent)

block_mask = create_block_mask(mask_mod, B=B, H=None, Q_LEN=L, KV_LEN=L)
```
Causality in packed order is valid because the state and each parent question always come before their children. For small sequences or debugging, an equivalent dense 4D mask with SDPA is fine.

### 5.5 Heads and losses
- **Choice:** `logits = head(h[read_idx])`, grouped per question. Loss is **listwise cross-entropy** within the group.
- **Score:** listwise cross-entropy plus an ordinal term, or a cumulative-link head.
- **Noul:** BCE on the sigmoid of the head output at the question's last token.
- **Total loss:** sum over questions, optionally weighted by type.
- **Fine-tuning:** start with LoRA on the backbone plus a fully trained head.

### 5.6 Calibration step
After training, fit `T` per `(type, K-bucket)` on a held-out split by minimizing NLL. Store the temperatures with the checkpoint.

---

## 6. Build plan and acceptance tests

1. **Packer + mask**
   - ✅ *Equivalence test:* the head outputs from a packed forward pass match running each `(state, q, a)` branch as a separate sequence (max abs diff < 1e-3 in fp32).
   - ✅ *Permutation test:* shuffling the question order and the option order leaves every answer's probability unchanged.
2. **Heads + training loop** on the train splits of a subset of the benchmarks in Section 7, converted to the format above. Keep the rest fully held out.
   - ✅ Loss decreases, and accuracy beats a zero-shot logit-reading baseline.
3. **Calibration** (benchmarks in Section 7)
   - ✅ Reliability diagrams and ECE (15 bins), NLL, Brier, accuracy, and AUROC, reported **separately for K ≤ 5, 6–20, and > 20**.
   - ✅ Temperature scaling reduces NLL on the test split without changing accuracy.
4. **Risk experiments (Section 4)** using the stress tests in Section 7.6
   - ✅ Option list inside the question vs. not, on "other" and "none of the above" tasks.
   - ✅ Near-duplicate option sets: measure mass inflation with and without paraphrase-augmented training.
   - ✅ Score monotonicity rate with and without the ordinal loss.
5. **Latency and cost benchmark**
   - ✅ Latency vs. number of questions at fixed state length should be near-flat.
   - ✅ Throughput in tokens/s per GPU.

### Suggested layout
```
system_one/
  data/        # format converters, synthetic generation
  packing.py   # packer -> input_ids, position_ids, seg_id, parent_id, read_idx
  masks.py     # FlexAttention mask_mod + block mask builder
  model.py     # backbone wrapper + Choice/Score/Noul heads
  losses.py    # listwise CE, ordinal loss, BCE
  calibrate.py # temperature fitting per (type, K-bucket)
  metrics.py   # ECE, reliability diagrams, NLL, Brier, AUROC
  serve.py     # simple batched inference API returning typed answers + confidence
evals/
  converters/  # one converter per benchmark -> Section 5.2 format (text + multimodal)
  stress/      # synthetic stress-test builders (Sections 7.6, 8.9)
  baselines/   # letter-logit, sequence-likelihood, API, cross-encoder
  run_eval.py  # runs a benchmark suite, writes metrics + reliability plots
tests/
  test_equivalence.py
  test_permutation.py
```

---

## 7. Evaluation benchmarks

Every benchmark below is converted to the sample format in Section 5.2 by a converter in `evals/converters/`. Each converter records:
- the question type (Choice / Score / Noul),
- K (the number of options),
- whether the task has a catch-all option ("other", "none", "not enough info").

Hugging Face dataset IDs are given as a starting point. **Verify them before use**: some datasets have moved namespaces, and some (e.g. GPQA) are gated.

### 7.0 Ground rules
- **Keep training and evaluation separate.** Keep a set of benchmarks **fully held out** (never in the training mix) for zero-shot evaluation. For the rest, train only on the official train split and report on the test split (or validation, where the test split is unlabeled).
- **Watch for contamination.** Popular benchmarks such as MMLU, HellaSwag, and ARC are likely in the backbone's pretraining data. Treat absolute accuracy on them with suspicion; calibration and permutation results are more informative there.
- **Use the same calibration split everywhere.** Fit temperatures on a calibration slice taken from each dataset's train or validation split, never from the test split.
- **Always report together:**
  - accuracy (or macro-F1 for imbalanced classes),
  - NLL,
  - Brier,
  - ECE (15 bins),
  - AUROC of confidence vs. correctness,
  - per-K-bucket breakdown (K ≤ 5, 6–20, > 20).

### 7.1 Multiple-choice QA (Choice, small K)
Core tests of listwise option scoring. Each answer choice becomes an answer branch.

| Benchmark | HF ID (verify) | K | What it tests |
|---|---|---|---|
| MMLU | `cais/mmlu` | 4 | Broad knowledge across 57 subjects; standard calibration benchmark |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | up to 10 | Harder, more options → K-dependence of calibration |
| ARC-Easy / ARC-Challenge | `allenai/ai2_arc` | mostly 4 | Grade-school science reasoning |
| CommonsenseQA | `tau/commonsense_qa` | 5 | Commonsense; options are often semantically close |
| OpenBookQA | `allenai/openbookqa` | 4 | Science facts + commonsense |
| HellaSwag | `Rowan/hellaswag` | 4 | Long answer branches (sentence endings) → length-bias check |
| PIQA | `ybisk/piqa` | 2 | Physical commonsense, long solutions |
| WinoGrande | `allenai/winogrande` | 2 | Minimal-pair options; hard for isolated branches |
| RACE | `ehovy/race` | 4 | Long state (reading passage) + short questions → prefix sharing |
| SciQ | `allenai/sciq` | 4 | Easy science; sanity check |
| TruthfulQA (MC1) | `truthfulqa/truthful_qa` | varies | Plausible-but-false options; tests overconfidence |
| GPQA | `Idavidrein/gpqa` (gated) | 4 | Very hard; checks that confidence drops when the model is out of its depth |
| MedQA / MedMCQA | `GBaker/MedQA-USMLE-4-options`, `openlifescienceai/medmcqa` | 4 | Domain transfer |

**RACE is a good fan-out test.** It has one long passage with several questions, so pack them all into one request against a shared state and check that the results match running each question separately.

### 7.2 Large-K classification (Choice, K ≫ 5)
These test the "hundreds of options" regime and calibration drift with K.

| Benchmark | HF ID (verify) | K | What it tests |
|---|---|---|---|
| AG News | `fancyzhx/ag_news` | 4 | Baseline topic classification |
| TREC (coarse / fine) | `CogComp/trec` | 6 / 50 | Same task at two values of K → direct K-drift comparison |
| DBpedia-14 | `fancyzhx/dbpedia_14` | 14 | Mid-K ontology classification |
| 20 Newsgroups | `SetFit/20_newsgroups` | 20 | Overlapping topics (near-duplicate-ish classes) |
| Banking77 | `PolyAI/banking77` | 77 | Fine-grained, confusable intents |
| CLINC150 | `clinc/clinc_oos` | 150 + out-of-scope | Very large K **plus** a catch-all class → tests "other" handling |
| Hierarchical (e.g. patents, WOS) | various | tree | Beam search over nested Choices (see TypeSafe's hierarchical classification cookbook) |

**CLINC150 is the key one** for risk #1 in Section 4 (options can't see their alternatives). Compare out-of-scope recall with and without the option list inside the question branch.

### 7.3 Yes/no and verification (Noul)

| Benchmark | HF ID (verify) | What it tests |
|---|---|---|
| BoolQ | `google/boolq` | Passage-grounded yes/no |
| SQuAD 2.0 (answerable?) | `rajpurkar/squad_v2` | Recast as "Does the passage contain the answer to the question?" → abstention |
| FEVER / VitaminC | `fever/fever`, `tals/vitaminc` | Claim verification against evidence; VitaminC uses contrastive evidence |
| HaluEval | `pminervini/HaluEval` | Hallucination detection on QA/dialogue/summaries |
| MS MARCO / BEIR (relevance) | `microsoft/ms_marco`, `BeIR/*` | Recast as "Is this passage relevant?" → reranking via P(yes); report nDCG@10 and MRR as well |

### 7.4 NLI (Choice, K = 3, relational)

| Benchmark | HF ID (verify) | What it tests |
|---|---|---|
| MNLI | `nyu-mll/glue` (`mnli`) | Entailment / neutral / contradiction; matched vs. mismatched = domain shift |
| SNLI | `stanfordnlp/snli` | Easier NLI; sanity check |
| ANLI (R1–R3) | `facebook/anli` | Adversarial; calibration under hard distribution shift |

The "neutral" label is defined relative to the other two, so NLI is also a probe for risk #1.

### 7.5 Ordinal / rubric scoring (Score)
These test expected-level scoring and monotonicity.

| Benchmark | HF ID (verify) | Levels | What it tests |
|---|---|---|---|
| SST-5 | `SetFit/sst5` | 5 | Sentiment on an ordered scale |
| Yelp Review Full | `Yelp/yelp_review_full` | 5 | Star ratings; noisy adjacent levels |
| Amazon Reviews (star rating) | `mteb/amazon_reviews_multi` | 5 | Multilingual ordinal ratings |
| STS-B (binned) | `nyu-mll/glue` (`stsb`) | bin 0–5 into 6 levels | Pairwise similarity as a rubric |
| Essay scoring (ASAP-style) | check license | rubric-dependent | Human rubric levels with long state |

**Score-specific metrics** (in addition to the standard set):
- MAE of the expected score vs. the gold level,
- Quadratic Weighted Kappa,
- **monotonicity rate**: the fraction of predictions whose level distribution is unimodal.

### 7.6 Stress tests (synthetic, built from the sets above)
These target the specific failure modes of the tree-mask design.

| Test | How to build | Pass condition |
|---|---|---|
| **Permutation invariance** | Shuffle option order and question order on MMLU and CLINC150 | Per-option probabilities unchanged (max abs diff < 1e-3); compare against a baseline that lists options inline, which should show sensitivity |
| **Near-duplicate injection** | Add a paraphrase of one option (correct or distractor) to MMLU and Banking77 items | Report the mass inflation: `P(meaning) with duplicate − P(meaning) without` |
| **Catch-all removal** | Remove the gold option and add "none of the above" | P(none of the above) should rise; measure how often it becomes top-1 |
| **K scaling** | Take an easy item and pad with 5, 20, 100, 200 random distractors from other items | Accuracy degrades gracefully; ECE stays stable after per-K temperature scaling |
| **Length bias** | On HellaSwag, pad wrong endings with neutral filler | Predictions should not shift toward shorter or longer answers |
| **Fan-out equivalence** | Pack 1, 5, 13, 50 questions against the same state (RACE, BoolQ passages) | Answers identical to single-question runs |
| **Empty-state prior** | Run every question with an empty state | Probabilities should be near uniform; strong skews reveal prior bias |
| **Abstention** | Replace the state with an unrelated passage | Confidence should drop; NLL on "can't tell" labels should improve after training |

### 7.7 Baselines to compare against
1. **Same backbone, zero-shot letter-logit reading.** List options as A/B/C… in the prompt and renormalize over the letter tokens.
2. **Same backbone, summed and length-normalized sequence log-likelihood** per option, with and without a PMI correction.
3. **A general LLM via API**, prompted to output a single label, using token logprobs where available.
4. **A fine-tuned cross-encoder** (e.g. DeBERTa-v3) per task, as a strong specialized reference.

### 7.8 Efficiency benchmarks
- Latency p50/p95 vs. number of questions, at fixed state lengths (512, 4k, 16k, 32k tokens).
- Latency vs. total answer tokens, for K from 2 to 500.
- Throughput (input tokens/s per GPU) under concurrent load.
- Packed single pass vs. prefix-cached branch serving, at equal accuracy.

---

## 8. Multimodal extension and benchmarks

### 8.1 Architecture notes for a vision-language backbone
The state can contain images or video frames as well as text. The tree mask is unchanged, because vision tokens are simply part of the `state` segment.

- **Prefix sharing matters more here.** Vision tokens usually dominate the sequence: a single image can be hundreds to thousands of tokens, and a video clip far more. The vision encoder and the state prefill run **once**, and every question and answer branch reuses them. Asking many questions per image or frame is close to free, which suits fan-out triage (e.g. "person present?", "vehicle present?", "loitering?", "camera obstructed?" on every clip).
- **Position ids with multimodal RoPE.** VLMs such as Qwen2.5-VL / Qwen3-VL use M-RoPE (separate temporal, height, and width components for vision tokens).
  - Compute state positions exactly as the model's processor does.
  - Every branch should then start at the **same** text position offset: the processor's "next position" after the state.
  - The equivalence test (Section 6) catches mistakes here.
- **Backbone caveat.** Base (non-instruct) VLM checkpoints are rarer than text base models, so you may have to start from an instruct checkpoint. Expect worse initial calibration, which makes post-hoc temperature scaling and the calibration metrics more important.
- **Options can be images too.** For image-text matching or "which image shows X?" questions, an answer branch can hold image tokens. That is heavier, but the listwise head works the same way.
- **Video.** The state holds sampled frames (optionally with timestamps). Record the sampling rate and number of frames per benchmark, since both affect accuracy and calibration.

### 8.2 Image multiple-choice QA (Choice)

| Benchmark | HF ID (verify) | K | What it tests |
|---|---|---|---|
| MMMU | `MMMU/MMMU` | mostly 4 (some open) | College-level multi-discipline reasoning with images; use the MC subset |
| MMBench | `lmms-lab/MMBench` | 2–4 | Broad perception + reasoning; its original protocol already uses circular option shifting, which pairs well with the permutation test |
| MMStar | `Lin-Chen/MMStar` | 4 | Filtered so questions **require** the image; good for the image-blind test |
| SEED-Bench | `lmms-lab/SEED-Bench` | 4 | Large, broad image understanding |
| ScienceQA (image subset) | `derek-thomas/ScienceQA` | 2–5 | Science diagrams; variable K |
| A-OKVQA (MC) | `HuggingFaceM4/A-OKVQA` | 4 | Commonsense/world knowledge about images |
| AI2D | `lmms-lab/ai2d` | 4 | Diagram understanding |
| RealWorldQA | `xai-org/RealworldQA` | varies | Real-world spatial understanding (partly MC) |
| BLINK | `BLINK-Benchmark/BLINK` | 2–4 | Low-level perception (depth, correspondence, counting); some options are image regions |
| CV-Bench | `nyu-visionx/CV-Bench` | 2–6 | 2D/3D spatial relations, counting, depth |
| MathVista (MC subset) | `AI4Math/MathVista` | varies | Visual math; charts, plots, figures |

### 8.3 Visual yes/no, hallucination, and entailment (Noul / small-K Choice)

| Benchmark | HF ID (verify) | Type | What it tests |
|---|---|---|---|
| POPE | `lmms-lab/POPE` | Noul | Object hallucination ("Is there a X in the image?"), random/popular/adversarial splits |
| MME | `lmms-lab/MME` | Noul | Paired yes/no questions per image across perception and cognition |
| HallusionBench | `lmms-lab/HallusionBench` | Noul | Visual illusions and knowledge-vs-image conflicts; tests overconfidence |
| AMBER | check source | Noul | Hallucination across existence, attributes, relations |
| NLVR2 | `lil-lab/nlvr` | Noul | Statement true/false over a pair of images |
| SNLI-VE | check source | Choice, K = 3 | Visual entailment (image premise, text hypothesis) |
| Winoground | `facebook/winoground` (gated) | Choice, K = 2 | Minimal-pair image-caption matching; very hard for isolated branches |
| Hateful Memes | check license | Noul | Multimodal classification where neither modality alone suffices |

Report **POPE by split**. Adversarial POPE is where calibration usually breaks.

### 8.4 Image classification and retrieval (large-K Choice)
Class names or captions become answer branches.

| Benchmark | Source (verify) | K | What it tests |
|---|---|---|---|
| CIFAR-100 | `uoft-cs/cifar100` | 100 | Mid-K sanity check |
| Food-101 | `ethz/food101` | 101 | Fine-grained, visually similar classes |
| ImageNet-1k (or a 100-class subset) | `ILSVRC/imagenet-1k` (gated) | 100 / 1000 | Very large K; compare against CLIP zero-shot |
| ImageNet-R / -A / -Sketch, ObjectNet | various | 200 / 1000 | Calibration under distribution shift |
| iNaturalist (subset) | check license | hundreds+ | Extreme fine-grained + long tail |
| RVL-CDIP | `aharley/rvl_cdip` | 16 | Document-type classification from scanned pages |
| Flickr30k / MS-COCO (image→caption) | various | 5–100 captions | Retrieval recast as Choice over candidate captions; report R@1/R@5 too |

**Baseline for this group:** CLIP/SigLIP zero-shot with temperature-scaled similarity. It is cheap, strong, and well studied for calibration.

### 8.5 Visual rubric scoring (Score)

| Benchmark | Source (verify) | Levels | What it tests |
|---|---|---|---|
| KonIQ-10k | check source | bin MOS into 5 | Perceptual image quality |
| AVA (aesthetics) | check license | bin 1–10 | Aesthetic rating; noisy adjacent levels |
| Q-Bench | check source | MC + ratings | Low-level visual quality perception and description |

These use the Score metrics from Section 7.5: MAE, QWK, and monotonicity rate.

### 8.6 Video QA (Choice)

| Benchmark | HF ID (verify) | K | What it tests |
|---|---|---|---|
| Video-MME | `lmms-lab/Video-MME` | 4 | Short/medium/long videos; report by duration bucket |
| MVBench | `OpenGVLab/MVBench` | 2–5 | 20 temporal task types (action order, counting, state change) |
| NExT-QA (MC) | `lmms-lab/NExTQA` | 5 | Causal and temporal reasoning |
| EgoSchema | `lmms-lab/egoschema` | 5 | Very long egocentric clips |
| Perception Test (MC) | check source | 3 | Memory, physics, abstraction over video |
| TempCompass | check source | varies | Temporal direction, speed, order; pairs well with the frame-shuffle test |

### 8.7 Video classification and surveillance-style tasks
These are closest to real deployment triage: many Noul/Choice questions per clip, where confidence gates escalation to a human.

| Benchmark | Source (verify) | Type | What it tests |
|---|---|---|---|
| Kinetics-400 / 700 | check license (YouTube) | Choice, K = 400/700 | Large-K action recognition |
| Something-Something v2 | check license | Choice, K = 174 | Fine-grained temporal actions; direction matters |
| Charades | check source | Noul per class (multi-label) | Indoor multi-label activities |
| UCF-Crime | check license | Choice (13 anomaly types + normal), Noul (anomalous?) | Real surveillance anomalies; long untrimmed video |
| XD-Violence | check license | Choice (6 violence types + normal), Noul | Violence detection; weakly labeled |
| ShanghaiTech / UBnormal | check license | Noul per segment | Campus/synthetic anomaly detection with frame-level labels |

**Surveillance-specific metrics:**
- frame- or segment-level ROC-AUC and AP for "is anomalous",
- **false-alarm rate at a fixed recall** (the operational metric for triage),
- the share of clips that can be **auto-resolved** above a confidence threshold at a target precision.

### 8.8 Audio (optional)

| Benchmark | Source (verify) | Type | What it tests |
|---|---|---|---|
| ESC-50 | check source | Choice, K = 50 | Environmental sound classification |
| AudioSet (eval subset) | check license | Noul per class | Multi-label sound events |
| MMAU | check source | Choice | Audio understanding and reasoning MC |

### 8.9 Multimodal stress tests

| Test | How to build | Pass condition |
|---|---|---|
| **Image-blind** | Replace the image with a blank or noise image (MMStar, POPE, MMMU) | Accuracy near chance and **confidence drops**; a confident answer means the model is using text priors |
| **Image swap** | Pair each question with an image from a different item | Confidence drops; "none/cannot tell" rises if offered |
| **Caption leakage** | Add a misleading caption to the state that contradicts the image | Measure how often text overrides vision, and whether confidence reflects the conflict |
| **Resolution scaling** | Evaluate at several input resolutions / token budgets | Accuracy–cost curve; calibration stable across resolutions |
| **Frame shuffle / reverse** | Shuffle or reverse frame order (MVBench, TempCompass, SSv2) | Temporal questions should fail or lose confidence; static questions should be unaffected |
| **Frame count scaling** | 4, 8, 16, 32, 64 frames | Accuracy and latency curves; check the fan-out cost stays flat |
| **Fan-out per frame** | 1, 10, 50, 100 questions against one image or clip | Answers identical to single-question runs; latency near-flat |
| **Multi-camera state** | Pack clips from several cameras into one state and reference them by path (`cameras[2]`) | Answers match single-camera runs for camera-specific questions |

### 8.10 Multimodal baselines
1. The same VLM with zero-shot letter-logit reading (options listed inline).
2. The same VLM generating a label, with token logprobs.
3. CLIP/SigLIP zero-shot (classification, retrieval, and yes/no via prompt pairs).
4. Task-specific models where they exist (e.g. a trained video anomaly detector for UCF-Crime / XD-Violence).

**Licensing note:** Kinetics, UCF-Crime, XD-Violence, ImageNet, Hateful Memes, AVA, and several others have access or usage restrictions. Check terms before downloading or training on them.

---

## 9. Open questions
- Does a small set-attention layer over answer embeddings fix near-duplicates without reintroducing order effects?
- Should the temperature be a learned function of `log K` instead of being bucketed?
- Is RL (vs. plain listwise CE) actually needed, and for which data sources (judge ensembles, synthetic labels)?
- How small can the backbone get before calibration on hard questions degrades?
- For VLMs, does starting from an instruct checkpoint limit achievable calibration, and how much does RLCD-style training recover?
- Can the vision encoder output be cached across overlapping video windows to cut cost further for continuous streams?
- What is the best frame-sampling policy when many questions share one clip but need different temporal resolution?

---

## 10. References
- TypeSafe docs: https://docs.typesafe.ai/introduction · primitives: https://docs.typesafe.ai/primitives.md · AI primer: https://docs.typesafe.ai/introduction/machine-learning-primer.md
- Guo et al., 2017, *On Calibration of Modern Neural Networks* (temperature scaling)
- Kadavath et al., 2022, *Language Models (Mostly) Know What They Know*
- Zhao et al., 2021, *Calibrate Before Use* (contextual calibration)
- Holtzman et al., 2021, *Surface Form Competition*
- Kuhn et al., 2023, *Semantic Uncertainty* (semantic entropy)
- OpenAI, 2023, *GPT-4 Technical Report* (calibration before vs. after RLHF)
- Hendrycks et al., 2021, *Measuring Massive Multitask Language Understanding* (MMLU)
- Wang et al., 2024, *MMLU-Pro*
- Zheng et al., 2024, *Large Language Models Are Not Robust Multiple Choice Selectors* (option-order bias)
- Larson et al., 2019, *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction* (CLINC150)
- Casanueva et al., 2020, *Efficient Intent Detection with Dual Sentence Encoders* (Banking77)
- Thakur et al., 2021, *BEIR*
- Yue et al., 2024, *MMMU*
- Liu et al., 2024, *MMBench*
- Chen et al., 2024, *Are We on the Right Way for Evaluating Large Vision-Language Models?* (MMStar)
- Li et al., 2023, *Evaluating Object Hallucination in Large Vision-Language Models* (POPE)
- Fu et al., 2024, *BLINK*
- Thrush et al., 2022, *Winoground*
- Fu et al., 2024, *Video-MME*
- Li et al., 2024, *MVBench*
- Xiao et al., 2021, *NExT-QA*
- Mangalam et al., 2023, *EgoSchema*
- Sultani et al., 2018, *Real-world Anomaly Detection in Surveillance Videos* (UCF-Crime)
- Wu et al., 2020, *Not only Look, but also Listen* (XD-Violence)
- Radford et al., 2021, *CLIP*; Zhai et al., 2023, *SigLIP*
