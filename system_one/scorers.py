"""Scorers turn a request into per-question logits, running every branch through one prefix fork.

Each scorer returns {question_id: logits}: a [K] tensor in `templates.option_keys` order for
choice/score questions, and a scalar logit of P(yes) for noul questions.

- `LetterScorer`: zero-shot letter-logit reading (README §7.7 baseline 1).
- `LikelihoodScorer`: summed or mean answer log-likelihood per option branch, optionally
  PMI-corrected by subtracting the same score under an empty state (README §7.7 baseline 2).
- `HeadScorer`: linear heads on the hidden state at a `<|read|>` token (the trained path, PLAN.md M1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from system_one import templates
from system_one.fork import forward_branches_all, forward_branches_batched, forward_sequences, prefill_state, prefill_states
from system_one.models import load_text_model
from system_one.schema import Request


@dataclass
class Backbone:
    text_model: nn.Module  # Qwen3.5 text model returning final-norm hidden states
    lm_head: nn.Module
    tokenizer: object
    chunk_size: int | None = 32
    cuda_graphs: bool = False  # replay branch forwards through captured CUDA graphs (system_one/graphs.py)
    single_pass: bool = True  # requests with one branch run `state + branch` in one forward instead of forking
    _runner: object = field(default=None, init=False, repr=False)

    @classmethod
    def from_pretrained(cls, repo: str = "0.8b", dtype=None, device=None, device_map=None, quantization="default", **kwargs):
        """`repo`: an alias from `system_one.models.MODELS` (`0.8b`, `9b`, `35b-a3b`) or an HF repo id.
        `dtype=None` picks fp32 for 0.8B and bf16 (fp16 pre-Ampere) for the larger models."""
        text_model, lm_head, tokenizer = load_text_model(repo, dtype, device, device_map, quantization)
        return cls(text_model, lm_head, tokenizer, **kwargs)

    @property
    def pad_id(self) -> int:
        tok = self.tokenizer
        return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @property
    def device(self) -> torch.device:
        return self.text_model.embed_tokens.weight.device

    @property
    def runner(self):
        """Lazily built `GraphBranchRunner`; its largest row bucket is `chunk_size`."""
        chunk = self.chunk_size or 32
        if self._runner is None or self._runner.row_buckets[-1] != chunk:
            from system_one.graphs import GraphBranchRunner

            buckets = tuple(sorted({b for b in (1, 2, 4, 8, 16, 32, 64, 128) if b < chunk} | {chunk}))
            self._runner = GraphBranchRunner(self.text_model, self.pad_id, row_buckets=buckets)
        return self._runner

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def run(self, state_text: str, branches: list[list[int]]) -> list[torch.Tensor]:
        """Prefill `state_text` once and return each branch's hidden states, a list of [len_i, d]."""
        return self.run_batch([state_text], [branches])[0]

    def run_batch(self, state_texts: list[str], branches: list[list[list[int]]]) -> list[list[torch.Tensor]]:
        """Several requests at once. Returns per request a list of [len_i, d] branch hidden states.

        With `single_pass`, requests with exactly one branch run as `state + branch` sequences, `chunk_size` rows
        per forward. The rest fork: states prefilled together (left-padded), then every request's branches in
        shared batches of `chunk_size` rows."""
        states = [self.encode(t) for t in state_texts]
        out: list[list[torch.Tensor] | None] = [None] * len(states)
        singles = [i for i, bs in enumerate(branches) if len(bs) == 1] if self.single_pass else []
        step = self.chunk_size or len(singles) or 1
        for start in range(0, len(singles), step):
            idx = singles[start : start + step]
            hidden = forward_sequences(self.text_model, [states[i] + branches[i][0] for i in idx], self.pad_id)
            for i, h in zip(idx, hidden):
                out[i] = [h[-len(branches[i][0]) :]]
        forked = [i for i in range(len(states)) if out[i] is None]
        if forked:
            for i, h in zip(forked, self._fork([states[i] for i in forked], [branches[i] for i in forked])):
                out[i] = h
        return out

    def _fork(self, states: list[list[int]], branches: list[list[list[int]]]) -> list[list[torch.Tensor]]:
        if len(states) == 1:
            state_ids = torch.tensor([states[0]], device=self.device)
            cache = prefill_state(self.text_model, state_ids)
            if self.cuda_graphs:
                return [self.runner(cache, torch.ones_like(state_ids), branches[0], [0] * len(branches[0]))]
            return [forward_branches_all(self.text_model, cache, state_ids.shape[1], branches[0], self.pad_id, self.chunk_size)]
        cache, mask = prefill_states(self.text_model, states, self.pad_id)
        flat = [b for bs in branches for b in bs]
        rows = [r for r, bs in enumerate(branches) for _ in bs]
        if self.cuda_graphs:
            hidden = self.runner(cache, mask, flat, rows)
        else:
            hidden = forward_branches_batched(self.text_model, cache, mask, flat, rows, self.pad_id, self.chunk_size)
        out, i = [], 0
        for bs in branches:
            out.append(hidden[i : i + len(bs)])
            i += len(bs)
        return out

    def token_id(self, text: str) -> int:
        ids = self.encode(text)
        if len(ids) != 1:
            raise ValueError(f"{text!r} is not a single token: {ids}")
        return ids[0]


def _last_token_logprobs(backbone: Backbone, hidden: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(backbone.lm_head(hidden[-1]).float(), dim=-1)


class LetterScorer:
    """Letter-logit reading: one prompt per question, renormalised over the option letters."""

    def __init__(self, backbone: Backbone):
        self.backbone = backbone
        self.letter_ids = [backbone.token_id(f" {c}") for c in templates.LETTERS]
        self.yes_ids = [backbone.token_id(t) for t in (" yes", " Yes")]
        self.no_ids = [backbone.token_id(t) for t in (" no", " No")]

    @torch.no_grad()
    def __call__(self, request: Request) -> dict[str, torch.Tensor]:
        return self.batch([request])[0]

    @torch.no_grad()
    def batch(self, requests: list[Request]) -> list[dict[str, torch.Tensor]]:
        branches = [[self.backbone.encode(templates.letter_prompt(q)) for q in r.questions.values()] for r in requests]
        hidden = self.backbone.run_batch([templates.render_state(r.state) for r in requests], branches)
        return [self._logits(r, h) for r, h in zip(requests, hidden)]

    def _logits(self, request: Request, hidden: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for qid, h in zip(request.questions, hidden):
            q, logprobs = request.questions[qid], _last_token_logprobs(self.backbone, h)
            if q.type == "noul":
                out[qid] = logprobs[self.yes_ids].logsumexp(0) - logprobs[self.no_ids].logsumexp(0)
            else:
                out[qid] = logprobs[self.letter_ids[: q.num_options]]
        return out


class LikelihoodScorer:
    """Answer log-likelihood per option branch, optionally PMI-corrected against an empty state."""

    def __init__(self, backbone: Backbone, normalize: str = "sum", pmi: bool = False):
        if normalize not in ("sum", "mean"):
            raise ValueError("normalize must be 'sum' or 'mean'")
        self.backbone, self.normalize, self.pmi = backbone, normalize, pmi

    def _branches(self, request: Request):
        """Flattened (qid, prefix_len, ids) for every answer branch, in question then option order."""
        items = []
        for qid, q in request.questions.items():
            prefix = self.backbone.encode(templates.branch_prefix(q))
            for answer in templates.branch_answers(q):
                items.append((qid, len(prefix), prefix + self.backbone.encode(answer)))
        return items

    def _answer_scores(self, state: str, items) -> torch.Tensor:
        return self._scores(items, self.backbone.run(templates.render_state(state), [ids for _, _, ids in items]))

    def _scores(self, items, hidden) -> torch.Tensor:
        scores = []
        for (_, prefix_len, ids), h in zip(items, hidden):
            # Hidden state at t predicts token t + 1; the answer span is ids[prefix_len:].
            logprobs = torch.log_softmax(self.backbone.lm_head(h[prefix_len - 1 : -1]).float(), dim=-1)
            targets = torch.tensor(ids[prefix_len:], device=logprobs.device)
            token_lp = logprobs.gather(1, targets[:, None]).squeeze(1)
            scores.append(token_lp.sum() if self.normalize == "sum" else token_lp.mean())
        return torch.stack(scores)

    @torch.no_grad()
    def __call__(self, request: Request) -> dict[str, torch.Tensor]:
        return self.batch([request])[0]

    @torch.no_grad()
    def batch(self, requests: list[Request]) -> list[dict[str, torch.Tensor]]:
        items = [self._branches(r) for r in requests]
        branches = [[ids for _, _, ids in it] for it in items]
        hidden = self.backbone.run_batch([templates.render_state(r.state) for r in requests], branches)
        scores = [self._scores(it, h) for it, h in zip(items, hidden)]
        if self.pmi:  # same branches under an empty state
            empty = self.backbone.run_batch([templates.render_state("")] * len(requests), branches)
            scores = [s - self._scores(it, h) for s, it, h in zip(scores, items, empty)]
        return [_group(r, [qid for qid, _, _ in it], s) for r, it, s in zip(requests, items, scores)]


class DecisionHeads(nn.Module):
    """Choice and Score share `option_head` (one logit per option branch); Noul uses `noul_head`."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.option_head = nn.Linear(hidden_size, 1)
        self.noul_head = nn.Linear(hidden_size, 1)


def add_read_token(tokenizer, embedding_rows: int) -> int:
    """Register `<|read|>` as a special token. It must fit in the existing embedding (Qwen3.5 has spare rows)."""
    tokenizer.add_special_tokens({"additional_special_tokens": [templates.READ_TOKEN]})
    read_id = tokenizer.convert_tokens_to_ids(templates.READ_TOKEN)
    if read_id >= embedding_rows:
        raise ValueError(f"{templates.READ_TOKEN} got id {read_id}, beyond the {embedding_rows}-row embedding")
    return read_id


class HeadScorer:
    """Reads linear heads at a `<|read|>` token appended to each branch. Needs trained heads to be useful."""

    def __init__(self, backbone: Backbone, heads: DecisionHeads):
        self.backbone, self.heads = backbone, heads
        rows = backbone.text_model.embed_tokens.num_embeddings
        self.read_id = add_read_token(backbone.tokenizer, rows)

    @torch.no_grad()
    def __call__(self, request: Request) -> dict[str, torch.Tensor]:
        qids, branches = [], []
        for qid, q in request.questions.items():
            prefix = self.backbone.encode(templates.branch_prefix(q))
            answers = [""] if q.type == "noul" else templates.branch_answers(q)
            for answer in answers:
                qids.append(qid)
                branches.append(prefix + (self.backbone.encode(answer) if answer else []) + [self.read_id])
        hidden = self.backbone.run(templates.render_state(request.state), branches)
        reads = torch.stack([h[-1] for h in hidden]).to(self.heads.option_head.weight.dtype)
        option_logits = self.heads.option_head(reads).squeeze(-1).float()
        noul_logits = self.heads.noul_head(reads).squeeze(-1).float()
        out = _group(request, qids, option_logits)
        for qid, q in request.questions.items():
            if q.type == "noul":
                out[qid] = noul_logits[qids.index(qid)]
        return out


def _group(request: Request, qids: list[str], scores: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split flat per-branch scores back into per-question vectors (noul: yes minus no)."""
    out = {}
    for qid, q in request.questions.items():
        mask = torch.tensor([x == qid for x in qids], device=scores.device)
        group = scores[mask]
        out[qid] = group[0] - group[1] if q.type == "noul" and len(group) == 2 else group
    return out
