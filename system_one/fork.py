"""Prefix-fork execution for hybrid (Gated DeltaNet + attention) backbones.

A tree attention mask cannot isolate sibling branches inside recurrent layers, so instead we
prefill the shared state once, copy its cache along the batch dimension, and run every branch as
an independent batch row continuing from that cache (see PLAN.md §0).
"""

from __future__ import annotations

import copy
import os

import torch
from transformers.cache_utils import DynamicCache


def force_ieee_fp32() -> None:
    """Make fla's Triton kernels compute fp32 dots in IEEE precision instead of TF32 (Ampere+ default).

    Fork and sequential passes chunk the sequence differently, so under TF32 their reads differ by
    ~1e-3 relative (2e-2 abs on 0.8B). Exact comparisons need IEEE; scoring doesn't. fla hardcodes TF32
    for its fused triangular solve, so that constant is patched too. Call before the first forward.
    """
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    try:
        import triton.language as tl
        from fla.ops.gated_delta_rule import chunk_fwd
    except ImportError:
        return
    chunk_fwd.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")


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
