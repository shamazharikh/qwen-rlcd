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


def select_cache_rows(cache: DynamicCache, rows: torch.LongTensor, batch_size: int) -> DynamicCache:
    """New cache whose row i is source row `rows[i]` (every batched tensor state is gathered along dim 0)."""

    def take(value):
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            return value.index_select(0, rows.to(value.device))
        if isinstance(value, dict):
            return {k: take(v) for k, v in value.items()}
        return value

    new = copy.copy(cache)
    new.layers = []
    for layer in cache.layers:
        clone = copy.copy(layer)
        for name, value in vars(layer).items():
            setattr(clone, name, take(value) if isinstance(value, (torch.Tensor, dict)) else value)
        new.layers.append(clone)
    return new


def _left_pad(seqs: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[B, S_max] left-padded ids, padding mask (1 = real token) and positions (real tokens at 0..len-1)."""
    if any(len(s) == 0 for s in seqs):
        raise ValueError("sequences must be non-empty")
    b, s_max = len(seqs), max(len(s) for s in seqs)
    input_ids = torch.full((b, s_max), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((b, s_max), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        input_ids[i, s_max - len(s) :] = torch.tensor(s, dtype=torch.long, device=device)
        mask[i, s_max - len(s) :] = 1
    return input_ids, mask, (mask.cumsum(-1) - 1).clamp_min(0)


def prefill_states(text_model, states: list[list[int]], pad_id: int) -> tuple[DynamicCache, torch.Tensor]:
    """Prefill several states in one batch, left-padded to the longest. Returns the cache and the [B, S_max]
    padding mask (1 = real token). Real tokens get positions 0..len-1, as in an unpadded prefill; DeltaNet
    layers zero the padded positions, so their recurrent and conv states start exactly as for an unpadded row."""
    input_ids, mask, position_ids = _left_pad(states, pad_id, text_model.embed_tokens.weight.device)
    cache = DynamicCache(config=text_model.config)
    text_model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids, past_key_values=cache, use_cache=True)
    return cache, mask


def forward_sequences(text_model, seqs: list[list[int]], pad_id: int) -> list[torch.Tensor]:
    """One uncached, left-padded forward over full sequences. Returns each sequence's final-norm hidden states
    over its real tokens, a list of [len_i, d].

    For a request with a single branch, running `state + branch` this way reads the weights once, where the
    fork reads them twice (state prefill, then the branch forward). On 9B that second pass costs ~70 ms."""
    input_ids, mask, position_ids = _left_pad(seqs, pad_id, text_model.embed_tokens.weight.device)
    out = text_model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids, use_cache=False)
    return [out.last_hidden_state[i, input_ids.shape[1] - len(s) :] for i, s in enumerate(seqs)]


def forward_branches_batched(
    text_model,
    state_cache: DynamicCache,
    state_mask: torch.Tensor,
    branches: list[list[int]],
    rows: list[int],
    pad_id: int,
    chunk_size: int | None = None,
) -> list[torch.Tensor]:
    """Run `branches` (branch i continues state row `rows[i]` of a `prefill_states` cache) as right-padded
    batches. Returns each branch's final-norm hidden states over its real tokens, a list of [len_i, d]."""
    if len(rows) != len(branches) or any(len(b) == 0 for b in branches):
        raise ValueError("need one state row per branch, and non-empty branches")
    step = chunk_size or len(branches)
    device, batch_size = text_model.embed_tokens.weight.device, state_mask.shape[0]
    out_all = []
    for start in range(0, len(branches), step):
        chunk, idx = branches[start : start + step], torch.tensor(rows[start : start + step], device=device)
        n, max_len = len(chunk), max(len(b) for b in chunk)
        input_ids = torch.full((n, max_len), pad_id, dtype=torch.long, device=device)
        for i, b in enumerate(chunk):
            input_ids[i, : len(b)] = torch.tensor(b, dtype=torch.long, device=device)
        smask = state_mask.index_select(0, idx)
        position_ids = smask.sum(-1, keepdim=True) + torch.arange(max_len, device=device)  # continue at each row's own length
        attention_mask = torch.cat([smask, torch.ones((n, max_len), dtype=smask.dtype, device=device)], dim=1)
        cache = select_cache_rows(state_cache, idx, batch_size)
        out = text_model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=cache, use_cache=True
        )
        out_all += [out.last_hidden_state[i, : len(b)] for i, b in enumerate(chunk)]
    return out_all


def forward_branches_all(
    text_model,
    state_cache: DynamicCache,
    state_len: int,
    branches: list[list[int]],
    pad_id: int,
    chunk_size: int | None = None,
) -> list[torch.Tensor]:
    """Run `branches` as right-padded batches continuing from `state_cache`.

    Returns each branch's final-norm hidden states over its real tokens, a list of [len_i, d]. Right
    padding is safe without a mask: everything is causal, so pad tokens after a branch's last real
    token cannot influence it (they only pollute the per-branch cache, which is discarded).

    Each branch row holds its own copy of the attention-layer KV, so memory grows with
    #branches × state length. `chunk_size` caps the rows per forward; outputs don't depend on it.
    """
    if any(len(b) == 0 for b in branches):
        raise ValueError("branches must be non-empty")
    if chunk_size is not None and chunk_size < len(branches):
        return [
            h
            for i in range(0, len(branches), chunk_size)
            for h in forward_branches_all(text_model, state_cache, state_len, branches[i : i + chunk_size], pad_id)
        ]
    device = text_model.embed_tokens.weight.device
    n, max_len = len(branches), max(len(b) for b in branches)
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long, device=device)
    for i, b in enumerate(branches):
        input_ids[i, : len(b)] = torch.tensor(b, dtype=torch.long, device=device)
    # Every branch starts at the same position offset, so siblings are position-identical.
    position_ids = torch.arange(state_len, state_len + max_len, device=device).expand(n, -1)
    cache = expand_cache(state_cache, n)
    out = text_model(input_ids=input_ids, position_ids=position_ids, past_key_values=cache, use_cache=True)
    return [out.last_hidden_state[i, : len(b)] for i, b in enumerate(branches)]


def forward_branches(
    text_model,
    state_cache: DynamicCache,
    state_len: int,
    branches: list[list[int]],
    pad_id: int,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Final-norm hidden states at each branch's last real token, shape [N, d] (see `forward_branches_all`)."""
    hidden = forward_branches_all(text_model, state_cache, state_len, branches, pad_id, chunk_size)
    return torch.stack([h[-1] for h in hidden])


def fork_forward(
    text_model,
    state_ids: torch.LongTensor,
    branches: list[list[int]],
    pad_id: int,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Prefill `state_ids` ([1, S]) once, then read every branch. Returns [N, d]."""
    cache = prefill_state(text_model, state_ids)
    return forward_branches(text_model, cache, state_ids.shape[1], branches, pad_id, chunk_size)


def sequential_reads(text_model, state_ids: torch.LongTensor, branches: list[list[int]]) -> torch.Tensor:
    """Reference: one uncached full forward per `state + branch`, reading the last token. Returns [N, d]."""
    reads = []
    for b in branches:
        branch = torch.tensor([b], dtype=torch.long, device=state_ids.device)
        out = text_model(input_ids=torch.cat([state_ids, branch], dim=1), use_cache=False)
        reads.append(out.last_hidden_state[0, -1])
    return torch.stack(reads)
