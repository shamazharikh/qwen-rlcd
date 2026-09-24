"""Scorers turn a request into per-question logits, running every branch through one prefix fork.

Each scorer returns {question_id: logits}: a [K] tensor in `templates.option_keys` order for
choice/score questions, and a scalar logit of P(yes) for noul questions.

- `LetterScorer`: zero-shot letter-logit reading (README §7.7 baseline 1).
- `LikelihoodScorer`: summed or mean answer log-likelihood per option branch, optionally
  PMI-corrected by subtracting the same score under an empty state (README §7.7 baseline 2).
- `HeadScorer`: linear heads on the hidden state at a `<|read|>` token (the trained path, PLAN.md M1).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch
from torch import nn

from system_one import templates
from system_one.fork import forward_branches_all, prefill_state
from system_one.schema import Request


@dataclass
class Backbone:
    text_model: nn.Module  # Qwen3.5 text model returning final-norm hidden states
    lm_head: nn.Module
    tokenizer: object
    chunk_size: int | None = 32

    @classmethod
    def from_pretrained(cls, repo: str = "Qwen/Qwen3.5-0.8B-Base", dtype=torch.float32, device=None, **kwargs):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(device) == "cpu" and "transformers.models.qwen3_5.modeling_qwen3_5" not in sys.modules:
            # transformers binds the DeltaNet kernel at import and picks fla (CUDA-only) whenever installed.
            sys.modules.setdefault("fla", None)
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        model = AutoModelForImageTextToText.from_pretrained(repo, dtype=dtype).eval().to(device)
        return cls(model.model.language_model, model.lm_head, AutoTokenizer.from_pretrained(repo), **kwargs)

    @property
    def pad_id(self) -> int:
        tok = self.tokenizer
        return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @property
    def device(self) -> torch.device:
        return self.text_model.embed_tokens.weight.device

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def run(self, state_text: str, branches: list[list[int]]) -> list[torch.Tensor]:
        """Prefill `state_text` once and return each branch's hidden states, a list of [len_i, d]."""
        state_ids = torch.tensor([self.encode(state_text)], device=self.device)
        cache = prefill_state(self.text_model, state_ids)
        return forward_branches_all(self.text_model, cache, state_ids.shape[1], branches, self.pad_id, self.chunk_size)

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
        qids = list(request.questions)
        branches = [self.backbone.encode(templates.letter_prompt(request.questions[q])) for q in qids]
        hidden = self.backbone.run(templates.render_state(request.state), branches)
        out = {}
        for qid, h in zip(qids, hidden):
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
        hidden = self.backbone.run(templates.render_state(state), [ids for _, _, ids in items])
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
        items = self._branches(request)
        scores = self._answer_scores(request.state, items)
        if self.pmi:
            scores = scores - self._answer_scores("", items)
        return _group(request, [qid for qid, _, _ in items], scores)


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
