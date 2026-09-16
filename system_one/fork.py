"""Prefix-fork execution for hybrid (Gated DeltaNet + attention) backbones.

A tree attention mask cannot isolate sibling branches inside recurrent layers, so instead we
prefill the shared state once, copy its cache along the batch dimension, and run every branch as
an independent batch row continuing from that cache (see PLAN.md §0).
"""

from __future__ import annotations

import copy

import torch
from transformers.cache_utils import DynamicCache


def prefill_state(text_model, state_ids: torch.LongTensor) -> DynamicCache:
    """Run the state once and return its cache. `state_ids` is [1, S]."""
    if state_ids.ndim != 2 or state_ids.shape[0] != 1:
        raise ValueError(f"state_ids must be [1, S], got {tuple(state_ids.shape)}")
    cache = DynamicCache(config=text_model.config)
    text_model(input_ids=state_ids, past_key_values=cache, use_cache=True)
    return cache


def _repeat(value, n: int, batch_size: int):
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
        return value.repeat_interleave(n, dim=0)
    if isinstance(value, dict):
        return {k: _repeat(v, n, batch_size) for k, v in value.items()}
    return value


def expand_cache(cache: DynamicCache, n: int, batch_size: int = 1) -> DynamicCache:
    """Return a new cache whose every tensor state is repeated `n` times along the batch dim.

    `DynamicCache.batch_repeat_interleave` is not usable here: in transformers 5.17 the hybrid
    layers either lack it or inherit `DynamicLayer`'s, which repeats only keys/values and leaves the
    conv/recurrent states at batch size 1. The source cache is left untouched, because forward
    passes update cache states in place.
    """
    new = copy.copy(cache)
    new.layers = []
    for layer in cache.layers:
        clone = copy.copy(layer)
        for name, value in vars(layer).items():
            # Copy mutable bookkeeping dicts (e.g. has_previous_state) so the source is not mutated.
            setattr(clone, name, _repeat(value, n, batch_size) if isinstance(value, (torch.Tensor, dict)) else value)
        new.layers.append(clone)
    return new


def forward_branches(
    text_model,
    state_cache: DynamicCache,
    state_len: int,
    branches: list[list[int]],
    pad_id: int,
) -> torch.Tensor:
    """Run `branches` as one right-padded batch continuing from `state_cache`.

    Returns final-norm hidden states at each branch's last real token, shape [N, d]. Right padding
    is safe without a mask: everything is causal, so pad tokens after the read position cannot
    influence it (they only pollute the per-branch cache, which is discarded).
    """
    if any(len(b) == 0 for b in branches):
        raise ValueError("branches must be non-empty")
    device = text_model.embed_tokens.weight.device
    n, max_len = len(branches), max(len(b) for b in branches)
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long, device=device)
    for i, b in enumerate(branches):
        input_ids[i, : len(b)] = torch.tensor(b, dtype=torch.long, device=device)
    # Every branch starts at the same position offset, so siblings are position-identical.
    position_ids = torch.arange(state_len, state_len + max_len, device=device).expand(n, -1)
    cache = expand_cache(state_cache, n)
    out = text_model(input_ids=input_ids, position_ids=position_ids, past_key_values=cache, use_cache=True)
    last_idx = torch.tensor([len(b) - 1 for b in branches], device=device)
    return out.last_hidden_state[torch.arange(n, device=device), last_idx]


def fork_forward(text_model, state_ids: torch.LongTensor, branches: list[list[int]], pad_id: int) -> torch.Tensor:
    """Prefill `state_ids` ([1, S]) once, then read every branch. Returns [N, d]."""
    cache = prefill_state(text_model, state_ids)
    return forward_branches(text_model, cache, state_ids.shape[1], branches, pad_id)


def sequential_reads(text_model, state_ids: torch.LongTensor, branches: list[list[int]]) -> torch.Tensor:
    """Reference: one uncached full forward per `state + branch`, reading the last token. Returns [N, d]."""
    reads = []
    for b in branches:
        branch = torch.tensor([b], dtype=torch.long, device=state_ids.device)
        out = text_model(input_ids=torch.cat([state_ids, branch], dim=1), use_cache=False)
        reads.append(out.last_hidden_state[0, -1])
    return torch.stack(reads)
