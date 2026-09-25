# System One Decision Model — Prototype Brief (text + multimodal)

A prototype of a "System One" decision model, inspired by TypeSafe AI's **Jev**. The model takes a **state** (context) and a set of **typed questions**. In a single forward pass it returns **calibrated probability distributions** over a fixed set of answers, with no text generation.

> **Status of claims in this doc:** Anything under "Public facts" comes from TypeSafe's docs and launch coverage. Everything else is **our hypothesis** of how such a model could work. TypeSafe has not published Jev's architecture.

---

## Project status (2026-09-25)

The implementation targets **`Qwen/Qwen3.5-0.8B-Base`**. See [`PLAN.md`](PLAN.md) for milestones. Current scope is **inference only**; training and fine-tuning (PLAN.md M2) are on hold.

| Milestone | Status |
|---|---|
| **M0** backbone spike (prefix fork) | ✅ Done. Fork = full forward on real weights (CPU, CUDA + fla, fp32/fp16); 8–19× faster than one forward per branch; chunked branches bound memory. Gradient-through-cache check deferred until training resumes. |
| **M1** templates, heads, typed API, zero-shot baselines | 🟡 Inference slice done: `predict()` with typed answers, letter / likelihood / PMI / `<|read|>`-head scorers, order and fan-out invariance tests, two-level fork for likelihood scoring, zero-shot numbers on 8 datasets (GPU, 500 test examples each). Remaining: trained heads (M2). |
| **M2** training (LoRA + heads) | ⏸ On hold |
| **M3** calibration and eval | 🟡 On the zero-shot scorers: temperature per (type, K-bucket) and T(log K) (`calibrate.py`), calibrated eval with AUROC, stress suite (K scaling, empty-state prior, length bias). Reliability plots and the trained model's numbers remain. |
| **M4** serving / **M5** multimodal | Not started |

**Environment:** runs in Docker (`Dockerfile`, `scripts/docker_run.sh`) on 2× RTX A4000 (16 GB, sm86, driver 550 / CUDA 12.4): torch 2.14 cu126, transformers 5.17, fla 0.5.2. The HF hub cache and the Triton kernel/autotune cache live on `/big/mazhar/qwen-rlcd` (mounted at `/cache`). The earlier M0 numbers come from a 2× RTX 2080 Ti box (one of its GPUs fell off the PCIe bus during a dual-GPU benchmark). On the A4000s, two GPU jobs at once have run for ~2 h without trouble.

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

### M0 results: GPU (2026-09-25, RTX A4000 16 GB, sm86, fla 0.5.2, torch 2.14 cu126, Docker)
All 48 tests pass, including real weights: fork vs. sequential max abs diff **2.9e-5 to 5.7e-5** in fp32 (with IEEE fp32 kernels, see below). In bf16 the min cos is 0.99986 (max abs diff 0.38).

### M0 findings
- **`DynamicCache.batch_repeat_interleave` can't fork this model** (transformers 5.17). `LinearAttentionLayer` has no such method, and `LinearAttentionAndFullAttentionLayer` inherits `DynamicLayer`'s, which repeats only keys and values. Hence the custom `expand_cache`.
- **The cache updates DeltaNet state in place** (`copy_`), so backward through a forked cache fails. This only matters for training, which is out of scope for now.
- **Tiny random models barely use DeltaNet state.** Zeroing it moves reads by only ~1e-4, so tiny-model tests use a 1e-5 tolerance to stay sensitive.
- **Colab (T4):** compiling `causal-conv1d` ran the VM out of memory and killed the runtime twice, so the notebook now installs only `flash-linear-attention`. A third attempt couldn't connect to a runtime at all.
- **Unchunked forks run out of memory on long states.** Every branch row gets its own copy of the attention-layer KV (and SDPA's `repeat_kv` copies it again), so memory grows with #branches × state length: 2048 tokens × 50 branches OOMs on 11 GB. `forward_branches(..., chunk_size=N)` caps the rows per forward; with `chunk_size=25`, peak memory is flat in the branch count (table above). `expand` instead of copy wouldn't help, because `DynamicLayer.update` concatenates and materialises the KV anyway.
- **On Turing, fla's DeltaNet kernel is the bottleneck, not the GEMMs.** Profiling a 2048-token state × 50 branches in fp16: `ChunkGatedDeltaRule` is 64% of GPU time (mostly `chunk_fwd_kernel_o`), linear layers 13%, attention 4%. That's why fp16 is barely faster than fp32 for short states. For 24-token branches the torch fallback was actually faster (512 × 50: 275 ms vs 361 ms fla), probably because the chunk kernel pads each short branch to a 64-token chunk. Recheck on Ampere+ before choosing a kernel per phase (fla for the state prefill, and possibly `fused_recurrent` or torch for short branches).
- **With fla installed, the model can't run on CPU.** transformers binds DeltaNet to fla at import time whatever the device, and Triton then fails with "0 active drivers". `tests/conftest.py` hides fla when CUDA is unavailable. Do the same (`sys.modules["fla"] = None` before importing transformers) in any CPU script.
- **On Ampere+, fla computes fp32 dots in TF32.** fla sets `TRITON_F32_DEFAULT=ieee` only on pre-Ampere cards, and it hardcodes TF32 for the fused triangular solve in `gated_delta_rule/chunk_fwd.py`. Under TF32, fork and sequential reads differ by ~1e-3 relative (2e-2 abs on 0.8B), because the two paths chunk the sequence differently. `fork.force_ieee_fp32()` sets the env var and patches that constant, which brings the A4000 back to 2.9e-5. The tests call it (`tests/conftest.py`). Scoring runs keep TF32, since 1e-3 relative is harmless there.
- **Docker image notes:** the default PyPI torch wheels need a CUDA 13 driver, so the image installs torch from the cu126 index, which runs on the 550 driver. Triton compiles a C launcher at runtime, so the slim image needs `gcc`. The first fla run compiles and autotunes for ~75–110 s; with the Triton cache persisted, a new container starts in ~2.5 s.
- **Hardware incident (2080 Ti box):** running benchmarks on both 2080 Tis at once made GPU 0 fall off the bus (`nvidia-smi`: "Unable to determine the device handle ... Unknown Error"). After that, CUDA init failed on both GPUs until a reset. Run one GPU job at a time on this box.

### What's built (M1, inference only)
| Path | What it does |
|---|---|
| `system_one/schema.py` | Parses and validates the README §5.2 request (`Request.from_dict`). Typed `ChoiceAnswer` / `ScoreAnswer` / `NoulAnswer` |
| `system_one/templates.py` | State, question-prefix and answer-branch text, with the compact option list inside every branch (§4). Choice options are always rendered in sorted key order, so outputs don't depend on the caller's order. Letter prompts for the baseline |
| `system_one/scorers.py` | `Backbone` (model + tokenizer + fork) and three scorers, all running every branch of every question through one prefix fork: `LetterScorer` (letter-logit reading, §7.7 #1), `LikelihoodScorer` (sum/mean answer log-likelihood, optional PMI against an empty state, §7.7 #2), `HeadScorer` (linear heads at a `<|read|>` token; heads untrained until M2) |
| `system_one/predict.py` | `predict(request, scorer, temperature=None)` → typed answers: softmax / sigmoid, confidence = 1 − H(p)/log K, score = Σ p_i·i (0-based) |
| `system_one/metrics.py` | Accuracy, NLL, Brier, ECE-15, and MAE for Score |
| `tests/test_predict.py` | Schema validation, rendering independent of option order, answer well-formedness, **question/option-order invariance and fan-out = single-question answers for every scorer** (< 1e-5), fork likelihood = unforked likelihood, PMI arithmetic, `<|read|>` fits in the embedding |
| `system_one/fork.py` (M1 addition) | `extend_cache` for **two-level forks**: state → one cache per question → its answer branches, so the question text (and its option list) runs once per question, not once per option. `force_ieee_fp32()` for exact comparisons on Ampere+ |
| `system_one/scorers.py` (M1 addition) | `Backbone.run_grouped`; `LikelihoodScorer` uses the two-level fork (Banking77, K = 77: 5.5 → 0.61 s/example on the A4000) |
| `system_one/calibrate.py` | `TemperatureTable` (one T per type × K-bucket ≤5 / 6–20 / >20, NLL-fitted with L-BFGS, T = 1 below 20 records) and `LogKTemperature` (T = exp(a + b·log K), separate noul T). Both are `temperature(type, K)` callables for `predict` and serialise to JSON |
| `system_one/metrics.py` | Adds AUROC of top-1 confidence vs. correctness |
| `scripts/zero_shot_eval.py` | Zero-shot baselines through the typed API on 8 datasets. Dumps raw logits, splits each dataset in half (calibration / test), fits both temperature models on the pooled calibration halves per scorer, and reports test metrics before and after. `--from-dump a.pt,b.pt` refits without the model |
| `scripts/stress_eval.py` | K scaling (Banking77 with K = 2…77), empty-state prior, length bias |
| `tests/test_calibrate.py` | Temperature fits recover known temperatures on synthetic data; per-bucket fallback; JSON round trip |

### M1/M3 results: zero-shot baselines with temperature calibration
`scripts/docker_run.sh python scripts/zero_shot_eval.py --limit 1000` (two runs split by dataset across the two GPUs, merged with `--from-dump`): Qwen3.5-0.8B-Base, fp32, seed 0. There are 1000 random examples per dataset (TREC: all 500), split into even-index calibration and odd-index test halves. Numbers are on the **test halves (n = 500, TREC 250)**, so the 95% CI on accuracy is about ±0.04. Letter prompts support at most 26 options, so Banking77 has no letter row.

**Accuracy** (temperature never changes the argmax):

| dataset (type, K) | letter | sum | mean | sum-pmi | mean-pmi | chance |
|---|---|---|---|---|---|---|
| ARC-Challenge (choice, 4) | **0.612** | 0.344 | 0.376 | 0.400 | 0.372 | 0.25 |
| ARC-Easy (choice, 4) | **0.800** | 0.688 | 0.628 | 0.602 | 0.576 | 0.25 |
| CommonsenseQA (choice, 5) | **0.516** | 0.368 | 0.432 | 0.486 | 0.480 | 0.20 |
| AG News (choice, 4) | **0.676** | 0.312 | 0.356 | 0.548 | 0.546 | 0.25 |
| TREC coarse (choice, 6) | 0.548 | 0.332 | 0.208 | 0.548 | **0.584** | 0.17 |
| Banking77 (choice, 77) | – | 0.078 | 0.022 | **0.234** | 0.208 | 0.013 |
| BoolQ (noul) | **0.770** | 0.758 | 0.758 | 0.754 | 0.754 | 0.62 (majority) |
| SST-5 (score, 5) | 0.214 | 0.256 | 0.178 | 0.304 | **0.306** | 0.20 |

**Calibration:** test NLL and ECE-15, uncalibrated → per-bucket T fitted on the pooled calibration halves:

| dataset | scorer | NLL | ECE-15 | AUROC |
|---|---|---|---|---|
| ARC-Challenge | letter | 0.945 → 0.954 | 0.071 → 0.071 | 0.73 |
| | sum | 3.645 → **1.382** | 0.442 → **0.048** | 0.56 |
| ARC-Easy | letter | 0.557 → 0.585 | 0.101 → 0.126 | 0.88 |
| | sum | 1.415 → 1.123 | 0.135 → 0.309 | 0.68 |
| CommonsenseQA | letter | 1.246 → 1.246 | 0.072 → 0.049 | 0.73 |
| | sum-pmi | 1.458 → 1.284 | 0.193 → 0.096 | 0.68 |
| AG News | letter | 0.873 → 0.835 | 0.150 → 0.124 | 0.74 |
| | sum | 5.157 → **1.298** | 0.636 → 0.329 | 0.87 |
| | sum-pmi | 1.876 → 1.069 | 0.340 → 0.081 | 0.72 |
| TREC | sum | 6.892 → **1.794** | 0.374 → 0.095 | 0.73 |
| | mean-pmi | 1.587 → 1.269 | 0.358 → 0.084 | 0.68 |
| Banking77 | sum-pmi | 3.706 → 3.658 | 0.143 → 0.087 | 0.82 |
| BoolQ | letter | 0.532 → 0.499 | 0.108 → **0.029** | 0.68 |
| SST-5 | letter | 2.219 → 1.592 | 0.373 → 0.098 | 0.58 |
| | mean-pmi | 1.586 → 1.541 | 0.083 → **0.025** | 0.59 |

Fitted temperatures (pooled over datasets): letter choice/K≤5 1.12, K 6–20 1.08, score 6.34, noul 0.56. sum choice/K≤5 9.05, K 6–20 18.5, K>20 4.88. sum-pmi choice/K≤5 3.24, K>20 0.74. mean-pmi choice/K≤5 0.61, K>20 0.12. Every scorer gets noul T ≈ 0.57, meaning the yes/no logits are under-confident. The full table (every scorer, plus the log-K fit) is printed by the script.

- **No zero-shot scorer wins everywhere.** Letter-logit is best on knowledge and topic MC and on BoolQ. PMI is best where label priors dominate (SST-5, TREC) and at large K (Banking77: 0.23, 18× chance).
- **Temperature fixes badly scaled scorers:** summed likelihood drops from NLL 3.6–6.9 to 1.3–1.8, and BoolQ ECE drops from ~0.10 to 0.03 for every scorer. That meets the M3 acceptance on those cells (NLL down, accuracy unchanged, ECE < 0.05 for K ≤ 5).
- **But one T per (type, K-bucket) is not dataset-agnostic for zero-shot scorers.** The choice/K≤5 bucket pools ARC, CSQA and AG News, whose best temperatures differ, so some cells get worse (letter on ARC-Easy: NLL +0.03; sum on ARC-Easy: ECE 0.14 → 0.31). Zero-shot score scales depend on the task's wording, not just K. Trained heads (M2) are what should make one temperature table transfer.
- **T(log K) adds nothing** over the three buckets (NLL within ±0.02 on most cells, worse on TREC, the only K 6–20 data). Fitted slopes disagree in sign across scorers, which is the dataset confound again.
- Throughput on one A4000 (fp32, one question per request): letter 0.10 s/example, likelihood 0.16–0.18, PMI 0.31–0.36; Banking77 0.64 / 1.26 (PMI).

### M3 stress suite (zero-shot scorers)
`scripts/stress_eval.py --limit 300`. Question and option-order invariance and fan-out are exact unit tests; these measure the rest.

**K scaling:** Banking77 with the gold intent plus K−1 random distractors (accuracy / mean confidence, uncalibrated):

| scorer | K = 2 | 5 | 10 | 20 | 40 | 77 |
|---|---|---|---|---|---|---|
| letter | **0.87** / 0.73 | **0.75** / 0.58 | **0.65** / 0.42 | 0.26 / 0.29 | – | – |
| sum | 0.66 / 0.84 | 0.45 / 0.66 | 0.36 / 0.51 | 0.20 / 0.38 | 0.13 / 0.25 | 0.07 / 0.18 |
| mean | 0.55 / 0.68 | 0.25 / 0.39 | 0.15 / 0.25 | 0.04 / 0.16 | 0.03 / 0.10 | 0.02 / 0.06 |
| sum-pmi | 0.83 / 0.77 | 0.58 / 0.55 | 0.53 / 0.41 | **0.39** / 0.28 | **0.28** / 0.17 | **0.25** / 0.09 |
| mean-pmi | 0.82 / 0.55 | 0.58 / 0.25 | 0.51 / 0.13 | 0.36 / 0.07 | 0.26 / 0.03 | 0.22 / 0.02 |

Letter-logit collapses between K = 10 and 20 (long lettered lists), while sum-PMI degrades gracefully. Mean-length likelihood falls to chance from K = 20 on. Confidence falls with K for every scorer, and mean-PMI is badly under-confident at large K (0.02 confidence at 0.22 accuracy).

**Empty-state prior** (state replaced by `""`; KL to uniform, lower is better; PMI is uniform by construction):

| dataset | letter | sum | mean |
|---|---|---|---|
| ARC-Challenge | 0.13 | 0.83 | 0.11 |
| BoolQ | 0.03 | 0.03 | 0.03 |
| SST-5 | 0.90 | 0.21 | 0.03 |
| AG News | 0.91 | 1.37 | 0.23 |
| TREC | 0.34 | 1.13 | 0.12 |

Letter and summed likelihood carry strong label priors on fixed-label tasks (on SST-5, AG News and TREC the empty-state input is the same for every example, so these are single distributions). Under an empty state BoolQ leans to "yes" on 73–74% of questions.

**Length bias** (ARC-Challenge, 4 options): the argmax is the longest option 30% of the time for letter and 29% for sum-PMI, against 29% for gold (unbiased). Sum picks the longest 19% of the time (prefers short answers) and mean 38% (prefers long ones).

M1 design notes:
- `forward_branches_all` returns every branch token's hidden state (the likelihood scorers need them). `forward_branches` reads the last one.
- `<|read|>` gets id 248077. The tokenizer uses 248,077 ids but the embedding has 248,320 rows, so no resize is needed.
- The invariance tests are sensitive: rendering options in caller order makes all four scorers fail them (1e-1 to 3e-4 diffs).
- In the ARC and SST-5 converters, the question or review text is the state, so PMI's empty-state baseline is meaningful.

### Next steps
- [x] ~~Recover the GPU~~ (moved to the A4000 box, Docker) and rerun `zero_shot_eval.py` at n ≥ 500
- [x] Two-level fork (state → question cache → answers) for likelihood scoring
- [x] M3 on the zero-shot scorers: temperature per (type, K-bucket) and T(log K), stress suite
- [ ] Rerun `bench_fork.py` on the A4000 (Ampere, bf16), and choose the kernel per phase: fla for the state prefill, torch or `fused_recurrent` for short branches
- [ ] M3: reliability plots; a per-dataset temperature as an oracle, to measure how much the pooled table loses
- [ ] Use the two-level fork in `HeadScorer` too (the `<|read|>` branches still repeat the question prefix)
- [ ] M2 (on hold): train `DecisionHeads` + LoRA so `HeadScorer` becomes the real model. Needs gradients through the forked cache, which in-place cache updates currently block (PLAN.md fallback A: per-row `[state, branch]` concatenation)

### Quickstart
Docker (CUDA box; the repo is mounted, so edits need no rebuild):
```bash
docker build -t qwen-rlcd .
scripts/docker_run.sh pytest -q                                              # fast tests (tiny model)
scripts/docker_run.sh env QWEN_RLCD_SLOW=1 pytest -q -s -k qwen35            # real Qwen3.5-0.8B-Base weights
scripts/docker_run.sh python scripts/zero_shot_eval.py --limit 1000 --dump runs/zs.pt   # baselines + calibration
scripts/docker_run.sh python scripts/stress_eval.py --limit 300              # stress suite
GPU=1 scripts/docker_run.sh python scripts/bench_fork.py --chunk-size 25      # second GPU; GPU=none for CPU
```
Local venv (Mac/CPU):
```bash
uv venv --python 3.12 .venv && uv pip install -e '.[dev,eval]'
.venv/bin/python -m pytest -q
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
