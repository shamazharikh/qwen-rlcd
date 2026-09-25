# Implementation Plan — Jev-style decision model on Qwen3.5-0.8B

Companion to `README.md` (the design brief). This plan turns that brief into a buildable project on **`Qwen/Qwen3.5-0.8B-Base`**, and records where the backbone forces changes to the brief.

> **Status (2026-09-25):** M0 ✅ done (gradient-through-cache deferred with training). M1 🟡 inference slice done: typed `predict()`, zero-shot and `<|read|>`-head scorers, invariance tests, two-level fork, zero-shot baselines on 8 datasets (GPU, n = 500 test). M2 ⏸ on hold (inference-only scope). M3 🟡 on the zero-shot scorers: temperature fitting, calibrated eval, stress suite. M4 and M5 not started. Dev env is Docker on 2× RTX A4000. Details and numbers are in README "Project status"; next steps are at the end of this file.

---

## 0. Backbone facts that change the design

Verified from the Hugging Face `config.json` and `transformers/models/qwen3_5/modeling_qwen3_5.py` (2026-09-17):

| Fact | Value | Consequence |
|---|---|---|
| Checkpoints | `Qwen3.5-0.8B` and **`Qwen3.5-0.8B-Base`** | Use **Base**, as the brief recommends (no RLHF calibration damage) |
| Layers | 24 = **18 Gated DeltaNet (linear attention)** + **6 gated full attention** (every 4th layer) | ⚠️ **The tree mask in README §2.2 / §5.4 does not work.** A mask can't isolate siblings inside a recurrent layer: packed branch B would see branch A through the DeltaNet state and the short conv. |
| Hidden size | 1024, tied embeddings, vocab 248k | Heads are small; LoRA plus full heads is cheap |
| Positions | M-RoPE (interleaved, `mrope_section [11,11,10]`, partial rotary 0.25) | Branch position ids must be 3-D; use the model's `get_rope_index` for the state |
| Vision | Built-in ViT (12 layers, patch 16, merge 2) in the same checkpoint | The multimodal path (README §8) needs no second model |
| Cache | Per-layer `recurrent_states`, `conv_states`, and KV. Multi-token continuation from a cache is supported (`initial_state=`, conv-state prepend) | ✅ Enables **prefix-fork** execution (below) |
| Kernels | `fla` (flash-linear-attention) and `causal-conv1d` on CUDA; pure-torch fallback elsewhere | Develop on Mac with the torch fallback; train and benchmark on a CUDA GPU (e.g. RunPod) |

### Replacement for the tree mask: prefix-fork execution

```
state ──prefill once──► cache C_s  (KV for 6 attn layers + recurrent/conv state for 18 DeltaNet layers)
                          │  expand along batch dim (repeat_interleave, no copy for KV via expand)
          ┌───────────────┼────────────────┬──────────────┐
   [q1 + a1_1]      [q1 + a1_2]      [q2 + a2_1]     [q3 (noul)]      ← one padded batch, run once from C_s
        ▲ read h at last real token of each branch
```

- **Isolation is exact by construction:** every branch is its own batch row, so siblings can't interact in any layer type.
- **Order invariance is free:** every branch starts from the same cache and the same position offset `S`.
- **v1 flattens `question + answer` into one branch.** The question tokens are recomputed per option, which is cheap because questions are short.
- **v2 (optional, §6):** two-level fork (state → question caches → answer branches) for questions with very long instructions or K in the hundreds.
- **Right-pad branches and read at `last_real_idx`.** Padding after the read position can't affect the read, because the model is causal and recurrent. The final cache is discarded.
- **Cost profile matches Jev:** the state (which dominates the tokens) is prefilled once. Branch compute ≈ Σ branch tokens.

This is also how the brief's "serving alternative" (README §2.2) already works, so training and serving share one code path.

---

## 1. Environment and repo setup (day 0–1)

- `git init`; Python 3.11; `uv` or `pip` with `torch>=2.5`, `transformers` (a version that includes `qwen3_5`), `peft`, `datasets`, `accelerate`, `scikit-learn`, `matplotlib`.
- CUDA box: `flash-linear-attention`, `causal-conv1d`, `flash-attn` (optional).
- Mac: torch fallback kernels on CPU or MPS, for unit tests with tiny inputs only.
- Layout (adapted from README §6):

```
system_one/
  schema.py      # Request/Question/Label dataclasses; JSON (README §5.2) validation
  templates.py   # text rendering of state / question / option branches
  packing.py     # tokenize → state ids + list of Branch(ids, qid, option_key, type)
  fork.py        # prefill state, expand cache, batched branch forward, gather reads   ← replaces masks.py
  model.py       # backbone wrapper (LoRA) + heads
  losses.py      # listwise CE, ordinal term, BCE
  calibrate.py   # temperature per (type, K-bucket) or T(log K)
  metrics.py     # ECE-15, reliability plots, NLL, Brier, AUROC, QWK, monotonicity
  serve.py       # FastAPI: typed answers + probabilities + confidence
  train.py
evals/ converters/ stress/ baselines/ run_eval.py
tests/ test_fork_equivalence.py test_permutation.py test_padding.py test_cache_grad.py
```

**As built (2026-09-24):** `fork.py`, `schema.py`, `templates.py`, `scorers.py` (backbone wrapper, zero-shot scorers, `DecisionHeads`; this takes the roles planned for `packing.py` and the inference half of `model.py`), `predict.py` (the typed API that `serve.py` will wrap), and `metrics.py`. Tests are `test_fork_equivalence.py` (padding and sibling cases included) and `test_predict.py` (permutation and fan-out invariance). Scripts are `bench_fork.py` and `zero_shot_eval.py` (dataset converters inline for now). Extras: `[cuda]` = flash-linear-attention, `[eval]` = datasets. **Added 2026-09-25:** `calibrate.py`, `fork.extend_cache` (two-level fork, used by `LikelihoodScorer`), `tests/test_calibrate.py`, `scripts/stress_eval.py`, and `Dockerfile` + `scripts/docker_run.sh`. `zero_shot_eval.py` also does the calibrate/test split and plays the role planned for `run_eval.py`. `causal-conv1d` is not used (its build OOM-killed Colab; the torch conv path is cheap).

## 2. Milestone M0 — backbone spike (days 1–3) ⛳ go/no-go

Retire the architectural risk before writing anything else.

1. Load `Qwen3.5-0.8B-Base` text-only (`Qwen3_5ForConditionalGeneration`, or text submodel only).
2. **Fork equivalence:** for `state + branch`, compare the last-token hidden state of
   (a) one full forward pass, and (b) prefill the state, expand the cache ×N, then forward the N branches as a padded batch.
   Pass: max abs diff < 1e-3 in fp32, on both torch-fallback and fla kernels.
3. **Padding test:** the same branch padded to lengths +0/+7/+50 gives an identical read.
4. **Gradient through cache:** loss on branch reads backpropagates into LoRA params in the *state* segment (check that `A_log`/`in_proj` LoRA grads are non-zero on state-only layers). Confirm the fla `chunk_gated_delta_rule` backward supports `initial_state` grads. If not, run the state and branches in a single differentiable pass by concatenating `[state, branch_i]` per row (fallback A), or detach the state cache (fallback B, weaker).
5. Measure memory and throughput for state 2k tokens × 50 branches on a 24 GB GPU.

**Deliverable:** `fork.py` + the 3 tests green, plus a short note on which gradient path works.

**Outcome (✅ go):**
- Steps 1–3: done. Fork vs. full forward on real weights: 4.8e-5 (Mac CPU fallback) and 2.3e-5 (RTX 2080 Ti, fla, fp32); fp16 min cos 0.999998. Negative controls (zeroed recurrent or conv state) diverge by 3–6.
- Step 5 (on 11 GB instead of 24 GB): 2k-token state × 50 branches takes 0.78 s fp32 / 0.56 s fp16 with `chunk_size=25`, at 5.4 GiB. The fork is 8–19× faster than one forward per branch. Unchunked it OOMs, because every branch row copies the attention KV.
- Step 4: **deferred.** The transformers cache updates DeltaNet state in place (`copy_`), so backward through a forked cache fails. Resolve it when M2 resumes, via fallback A (per-row `[state, branch]` concatenation) or an out-of-place cache.

## 3. Milestone M1 — packing, heads, inference API (week 1)

**Templates** (base model, no chat template):
```
<state>\n{state}\n</state>\n
Question: {instructions}\nOptions: {shuffled list of option labels}\nAnswer: {option label}: {option description}<|read|>
```
- Include the option list inside the branch (README §4 risk 1). Shuffle it per training sample; at eval use a canonical sort so outputs stay deterministic.
- Add one special read token `<|read|>` (new embedding row, trained). Read the hidden state there instead of at a variable last subword.
- Noul branch: `Question: {instructions}\nAnswer (yes/no):<|read|>`.
- Score branch: same as Choice, plus the level index (`Level 2 of 5`).

**Heads** (on the final-norm hidden state, d=1024):
- `choice_head: Linear(1024→1)` shared with Score → logits grouped per question → `softmax(s / T)`.
- `noul_head: Linear(1024→1)` → sigmoid.
- Score v1 = listwise head; v2 = cumulative-link head (scalar + K−1 learned thresholds) behind a flag.
- Confidence = `1 − H(p)/log K` (Choice/Score). None for Noul.
- Score value = `Σ p_i · i`.

**Zero-shot baseline in the same harness:** letter-logit reading and PMI-corrected sequence log-likelihood (README §7.7 #1–2) on the same backbone.

**Deliverable:** `serve.predict(request) → typed response` matching TypeSafe's shapes. Unit tests for permutation invariance (question order *and* option order) pass with max diff < 1e-3.

**Outcome (🟡 inference slice done):**
- `predict(request, scorer, temperature)` in `predict.py` returns `ChoiceAnswer` / `ScoreAnswer` / `NoulAnswer`. Question ids never reach the model.
- Templates follow the plan. Options are listed by key in canonical sorted order (no training-time shuffle yet). Score answers read `Level i of K: {desc}`. Noul reads at the end of `Answer (yes/no):` (head) or compares ` yes`/` no` (zero-shot).
- `<|read|>` = id 248077, which fits in the checkpoint's spare embedding rows (no resize). `DecisionHeads` = `option_head` (shared by Choice/Score) + `noul_head`. They are untrained, and the cumulative-link Score head is not built.
- Zero-shot scorers: letter-logit, sum/mean likelihood, and PMI against an empty state. Invariance (question order, option order, fan-out) holds to < 1e-5 for every scorer, and the tests fail if options are rendered in caller order.
- Zero-shot numbers (CPU, n = 100): letter-logit is best on ARC-Challenge (0.60), PMI is best on BoolQ (0.79), and every scorer is weak on SST-5 (≤ 0.33). See README.
- Remaining: GPU numbers at n ≥ 500; a FastAPI wrapper (M4).

## 4. Milestone M2 — training (weeks 2–3)

**Data mix v1** (train splits only; converters in `evals/converters/`):
- Choice small-K: ARC, OpenBookQA, CommonsenseQA, SciQ, RACE (fan-out), HellaSwag
- Choice large-K: AG News, DBpedia-14, TREC-fine, Banking77
- Noul: BoolQ, FEVER, SQuAD2-answerable
- NLI: MNLI, SNLI
- Score: SST-5, Yelp-5
- **Held out, never trained on:** MMLU, MMLU-Pro, CLINC150, ANLI, TruthfulQA, GPQA, 20NG, MedMCQA
- Augmentations: option shuffle, random K subsampling / distractor padding (K ∈ {2..200}), paraphrased-duplicate options, "none of the above" with the gold option removed, empty/unrelated state with a uniform or "can't tell" target.

**Recipe:**
- LoRA r=32 on all linear layers of both attention types (q/k/v/o, DeltaNet `in_proj_*`/`out_proj`), MLP too. Heads and the `<|read|>` embedding fully trained. The 0.8B model is small enough to try full fine-tuning as an ablation.
- Batch = requests; each request expands to ≤ ~256 branches. Use gradient checkpointing per layer and bucket requests by state length.
- Loss = Σ_q w_type · { listwise CE (Choice) | CE + λ·EMD ordinal (Score) | BCE (Noul) }, with λ≈0.1.
- AdamW, lr 1e-4 (LoRA) / 1e-3 (heads), cosine, 1–2 epochs, bf16. Keep the DeltaNet state in fp32 (`mamba_ssm_dtype: float32`).
- Rough compute: a single 24–48 GB GPU (A6000/L40S/A100) is enough. Expect ≈ 1 GPU-day for v1.

**"RLCD" stage (v2, optional, after M3 numbers exist):**
- With a softmax head, the log-score reward's policy gradient equals CE, so start with CE only (README §2.5).
- Add an RL-style stage only for **soft/ensemble labels** (judge distributions): minimize KL(judge ‖ model) or the Brier score over distributions, and compare against CE on hard labels. Treat it as an experiment, not a dependency.

## 5. Milestone M3 — calibration and evaluation (week 3–4)

- `calibrate.py`: fit `T` per `(type, K-bucket ≤5 / 6–20 / >20)` on a calibration slice of the train/val splits. Also test `T = exp(a + b·log K)`. Store the temperatures in the checkpoint.
- `run_eval.py`: accuracy / macro-F1, NLL, Brier, ECE-15 plus reliability plots, AUROC(confidence vs. correct), per-K-bucket. For Score also MAE, QWK, and monotonicity rate.
- Stress suite (README §7.6): permutation, near-duplicate, catch-all removal, K scaling (5/20/100/200), length bias, fan-out (1/5/13/50), empty-state prior, abstention.
- Report against the §7.7 baselines. Add a DeBERTa-v3 cross-encoder on 2–3 tasks as a specialized reference.

**Acceptance (v1 targets, adjust after the first run):**
- Beats letter-logit zero-shot on held-out accuracy **and** NLL.
- Temperature scaling lowers NLL without changing accuracy.
- ECE < 0.05 on the K ≤ 5 bucket after scaling.
- Fan-out and permutation tests are exact (< 1e-3).

**Outcome on the zero-shot scorers (🟡, 2026-09-25):** see README "M1/M3 results" and "M3 stress suite".
- `calibrate.py` fits `TemperatureTable` (type × K-bucket) and `LogKTemperature` by NLL, fitted on the calibration halves of 8 datasets pooled per scorer. It is not yet stored in a checkpoint, since there is no trained checkpoint.
- Temperature scaling lowers NLL on most cells without changing accuracy, dramatically for summed likelihood (3.6–6.9 → 1.3–1.8). ECE < 0.05 for K ≤ 5 is met on BoolQ, SST-5 (PMI) and ARC-C (sum), but not generally.
- **Finding:** zero-shot score scales depend on the dataset's wording, not just K, so a pooled per-bucket T hurts some datasets (letter on ARC-Easy). T(log K) doesn't help. This is evidence that trained heads are needed for one temperature table to transfer.
- Stress: K scaling (2–77), empty-state prior and length bias are measured. Permutation and fan-out are exact unit tests. Not done: near-duplicate options, catch-all removal, abstention, K ≥ 100, reliability plots, QWK/monotonicity for Score, the DeBERTa reference.

## 6. Milestone M4 — serving and efficiency (week 4–5)

- `serve.py`: FastAPI with `POST /decide` using the README §5.2 schema and TypeSafe-shaped responses. Question IDs are never tokenized.
- Batching: group requests, prefill states in a batch, then run all branches in one padded batch (sort by branch length to reduce padding).
- Two-level fork for long questions or large K: prefill `state+question` once per question, then expand to its answers.
- Benchmarks (README §7.8): p50/p95 latency vs. #questions at states of 512 / 4k / 16k / 32k, latency vs. K from 2 to 500, and tokens/s. Target: near-flat latency vs. #questions. Note that the 0.8B hybrid model's constant-size DeltaNet state makes long states cheap for 18/24 layers.
- Later: check whether vLLM/SGLang support prefix caching for Qwen3.5 hybrid models in pooling mode. If not, keep the custom server.

## 7. Milestone M5 — multimodal (after text v1 is solid)

- Feed images/video through the built-in vision tower into the **state** only. Get state positions from `get_rope_index`. Branch positions continue at the processor's next text position for all 3 M-RoPE axes.
- Re-run the fork-equivalence test with an image in the state (it catches M-RoPE offset bugs).
- Add POPE, MMStar, MMBench, and ScienceQA-img to the mix. Hold out MMMU and HallusionBench. Run the README §8.9 stress tests (image-blind, image-swap, frame shuffle, fan-out per frame).

---

## Risks specific to this backbone

| Risk | Mitigation |
|---|---|
| fla backward doesn't support grads through `initial_state` | Fallback A: per-row `[state, branch]` concatenation (exact, costs O(branches × state)) for training only. Serving still forks. |
| Expanded caches blow memory at K=200 with long states | **Confirmed** (2k × 50 OOMs on 11 GB). `expand` doesn't help, because the cache update concatenates and materialises the KV. **Mitigated** by `chunk_size`, which keeps memory flat in K. A two-level fork would also cut per-row cost. |
| fla uses TF32 for fp32 dots on Ampere+ (and hardcodes it in the fused triangular solve) | **Confirmed** on the A4000: fork vs. sequential drifts to ~1e-3 relative. `fork.force_ieee_fp32()` restores 2.9e-5 for tests. Scoring keeps TF32 |
| fla's DeltaNet kernel is slow on pre-Ampere GPUs | Measured on Turing: 64% of GPU time, and slower than the torch path for short branches. Choose the kernel per phase, and benchmark on Ampere+ before optimising. |
| fla installed but running on CPU | transformers binds fla at import time for any device, and Triton then fails. Hide fla on CPU (`tests/conftest.py`, `Backbone.from_pretrained`). |
| 0.8B too weak for hard knowledge tasks (MMLU/GPQA) | Expected. Judge it on calibration (confidence drops when out of depth), not raw accuracy. The same code scales to Qwen3.5-2B/4B/9B. |
| Tokenizer has no clean "last token" for option text | The `<|read|>` sentinel token |
| Kernels unavailable on Mac | Keep the pure-torch path for tests; do real runs on CUDA |
| Base-model prior bias toward certain labels | Empty-state stress test; contextual calibration as a diagnostic |

## Timeline summary

| Week | Milestone | Exit criterion |
|---|---|---|
| 0–1 | M0 spike | Fork equivalence + gradient path proven |
| 1 | M1 | Typed inference API, invariance tests green, zero-shot baselines |
| 2–3 | M2 | Trained LoRA + heads beat baselines on held-out data |
| 3–4 | M3 | Calibration report with per-K breakdown + stress suite |
| 4–5 | M4 | Server + latency curves |
| 6+ | M5 / RLCD v2 | Multimodal, soft-label training experiments |

## Immediate next steps (updated 2026-09-25)
1. Benchmark on the A4000: `bench_fork.py` in fp32/bf16, and the per-phase kernel choice (fla prefill, torch or `fused_recurrent` for short branches).
2. Finish M3 on the zero-shot scorers: reliability plots, a per-dataset-T oracle to size the pooling loss, and the remaining stress tests (near-duplicates, catch-all removal, K ≥ 100).
3. Two-level fork in `HeadScorer`; batch several requests' states into one prefill (M4 groundwork).
4. When training resumes (M2): pick the gradient path (fallback A or an out-of-place cache), then train `DecisionHeads` + LoRA, and rerun the calibrated eval to check whether one temperature table transfers across datasets.
