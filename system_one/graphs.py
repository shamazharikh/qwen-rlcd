"""CUDA-graph replay for the fork's branch phase.

Branch forwards are short (tens of tokens) and launch hundreds of small kernels per layer stack, so on a
9B model they are launch-bound in eager mode. `GraphBranchRunner` captures one CUDA graph per shape bucket
(branch rows, branch length, state length) and replays it with the request copied into static buffers.

What makes the forward capturable:
- Masks are precomputed outside the graph and passed as `{"full_attention": 4D bool, "linear_attention": None}`,
  which skips transformers' mask builders (they call `torch.all(...)` on the host).
- Attention KV lives in preallocated buffers laid out as `[pad | state | branch]`; a static cache layer
  writes the branch keys/values into the tail in place instead of `torch.cat`, and padding is masked out.
- DeltaNet conv/recurrent states are copied into fixed buffers; their in-place updates during replay are
  harmless because every call refills them.

The same static path runs eagerly when graphs are off (CPU, or a shape that failed to capture), so its
layout and masking are testable without a GPU. Outputs match `fork.forward_branches_batched`.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import Cache, DynamicLayer, LinearAttentionCacheLayerMixin

log = logging.getLogger(__name__)


def _bucket(x: int, buckets: tuple[int, ...]) -> int:
    for b in buckets:
        if x <= b:
            return b
    raise ValueError(f"{x} exceeds the largest bucket {buckets[-1]}")


class _StaticKVLayer(DynamicLayer):
    """Attention cache layer over preallocated `[rows, heads, S + L, d]` views: `update` writes the new keys and
    values into the last L positions in place and returns the whole view."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        super().__init__()
        self.keys, self.values = keys, values
        self.dtype, self.device, self.is_initialized = keys.dtype, keys.device, True

    def update(self, key_states, value_states, *args, **kwargs):
        n = key_states.shape[-2]
        self.keys[:, :, -n:].copy_(key_states)
        self.values[:, :, -n:].copy_(value_states)
        return self.keys, self.values


@dataclass
class _Graph:
    rows: int
    length: int
    state: int  # state-length bucket
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    mask: torch.Tensor  # [rows, 1, length, state + length] bool
    cache: Cache
    graph: torch.cuda.CUDAGraph | None = None
    output: torch.Tensor | None = None  # graph-owned last_hidden_state
    eager: bool = False  # capture failed or graphs are off: run the static path eagerly
    replays: int = field(default=0)


class GraphBranchRunner:
    """Run fork branches through captured CUDA graphs (same contract as `fork.forward_branches_batched`).

    `row_buckets`: allowed batch sizes per forward; the largest is the chunk size. `length_buckets`: allowed
    branch lengths (longer branches fall back to eager). States are padded up to a multiple of
    `state_multiple` tokens. Buffers are sized for the largest state seen so far and regrown (dropping
    captured graphs) when a longer one arrives.
    """

    def __init__(
        self,
        text_model,
        pad_id: int,
        row_buckets: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        length_buckets: tuple[int, ...] = (8, 16, 32, 64, 128),
        state_multiple: int = 256,
        use_graphs: bool | None = None,
    ):
        self.model, self.pad_id = text_model, pad_id
        self.row_buckets, self.length_buckets, self.state_multiple = tuple(sorted(row_buckets)), tuple(sorted(length_buckets)), state_multiple
        self.device = text_model.embed_tokens.weight.device
        self.use_graphs = self.device.type == "cuda" if use_graphs is None else use_graphs
        self.graphs: dict[tuple[int, int, int], _Graph] = {}
        self.capacity = 0  # state positions reserved in the KV buffers
        self.buffers: list[dict] | None = None
        self.pool = None
        self.captures = 0

    # ---- buffers -------------------------------------------------------------------------------------

    def _allocate(self, template: Cache, capacity: int):
        """(Re)allocate per-layer buffers for `max(row_buckets)` rows and `capacity` state positions."""
        rows, lmax = self.row_buckets[-1], self.length_buckets[-1]
        self.graphs.clear()
        self.buffers = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        buffers = []
        for layer in template.layers:
            buf = {}
            if isinstance(layer, LinearAttentionCacheLayerMixin):
                conv, rec = layer.conv_states[0], layer.recurrent_states[0]
                buf["conv"] = torch.zeros((rows, *conv.shape[1:]), dtype=conv.dtype, device=conv.device)
                buf["rec"] = torch.zeros((rows, *rec.shape[1:]), dtype=rec.dtype, device=rec.device)
            if getattr(layer, "keys", None) is not None and layer.keys.numel():
                k, v = layer.keys, layer.values
                buf["k"] = torch.zeros((rows, k.shape[1], capacity + lmax, k.shape[3]), dtype=k.dtype, device=k.device)
                buf["v"] = torch.zeros((rows, v.shape[1], capacity + lmax, v.shape[3]), dtype=v.dtype, device=v.device)
            buffers.append(buf)
        self.buffers, self.capacity = buffers, capacity
        self.pool = torch.cuda.graph_pool_handle() if self.use_graphs else None

    def _cache_view(self, template: Cache, rows: int, length: int, state: int) -> Cache:
        """A cache whose tensors are views of the shared buffers, sized for this bucket."""
        cache = copy.copy(template)
        cache.layers = []
        lo, hi = self.capacity - state, self.capacity + length
        for layer, buf in zip(template.layers, self.buffers):
            if "k" in buf:
                cache.layers.append(_StaticKVLayer(buf["k"][:rows, :, lo:hi], buf["v"][:rows, :, lo:hi]))
            else:
                clone = copy.copy(layer)
                clone.conv_states = {0: buf["conv"][:rows]}
                clone.recurrent_states = {0: buf["rec"][:rows]}
                clone.has_previous_state = {0: True}
                cache.layers.append(clone)
        return cache

    def _fill_cache(self, src: Cache, idx: torch.Tensor, rows: int, src_len: int, state: int):
        """Copy source rows `idx` ([rows]) into the buffers: KV right-aligned at the state boundary, left pad zeroed."""
        for layer, buf in zip(src.layers, self.buffers):
            if "k" in buf:
                start = self.capacity - src_len
                if src_len < state:
                    buf["k"][:rows, :, self.capacity - state : start].zero_()
                    buf["v"][:rows, :, self.capacity - state : start].zero_()
                buf["k"][:rows, :, start : self.capacity].copy_(layer.keys.index_select(0, idx))
                buf["v"][:rows, :, start : self.capacity].copy_(layer.values.index_select(0, idx))
            else:
                buf["conv"][:rows].copy_(layer.conv_states[0].index_select(0, idx))
                buf["rec"][:rows].copy_(layer.recurrent_states[0].index_select(0, idx))

    # ---- graphs --------------------------------------------------------------------------------------

    def _get(self, template: Cache, rows: int, length: int, state: int) -> _Graph:
        key = (rows, length, state)
        if key in self.graphs:
            return self.graphs[key]
        g = _Graph(
            rows, length, state,
            input_ids=torch.full((rows, length), self.pad_id, dtype=torch.long, device=self.device),
            position_ids=torch.zeros((rows, length), dtype=torch.long, device=self.device),
            mask=torch.zeros((rows, 1, length, state + length), dtype=torch.bool, device=self.device),
            cache=self._cache_view(template, rows, length, state),
        )  # fmt: skip
        g.eager = not self.use_graphs
        self.graphs[key] = g
        return g

    def _forward(self, g: _Graph) -> torch.Tensor:
        masks = {"full_attention": g.mask, "linear_attention": None}
        out = self.model(input_ids=g.input_ids, attention_mask=masks, position_ids=g.position_ids, past_key_values=g.cache, use_cache=True)
        return out.last_hidden_state

    def _capture(self, g: _Graph):
        """Warm up (Triton autotune, cuBLAS handles) on a side stream, then capture. Buffers already hold real data."""
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._forward(g)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                g.output = self._forward(g)
            g.graph = graph
            self.captures += 1
        except Exception as e:  # e.g. host syncs in MoE routing: keep the static path, run it eagerly
            log.warning("CUDA graph capture failed for rows=%d len=%d state=%d (%s); running eagerly", g.rows, g.length, g.state, e)
            g.graph, g.output, g.eager = None, None, True
            torch.cuda.synchronize()

    # ---- entry point ---------------------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, state_cache: Cache, state_mask: torch.Tensor, branches: list[list[int]], rows: list[int]) -> list[torch.Tensor]:
        """Branch i continues state row `rows[i]` of `state_cache` (left-padded per `state_mask`, [B, S]).
        Returns each branch's final-norm hidden states over its real tokens, a list of [len_i, d]."""
        if len(rows) != len(branches) or any(len(b) == 0 for b in branches):
            raise ValueError("need one state row per branch, and non-empty branches")
        src_len = state_mask.shape[1]
        state = math.ceil(src_len / self.state_multiple) * self.state_multiple
        if self.buffers is None or state > self.capacity:
            self._allocate(state_cache, max(state, self.capacity))
        step = self.row_buckets[-1]
        out: list[torch.Tensor] = []
        for start in range(0, len(branches), step):
            out += self._run_chunk(state_cache, state_mask, branches[start : start + step], rows[start : start + step], state)
        return out

    def _run_chunk(self, src: Cache, state_mask, chunk, chunk_rows, state) -> list[torch.Tensor]:
        n, max_len = len(chunk), max(len(b) for b in chunk)
        if max_len > self.length_buckets[-1]:
            from system_one.fork import forward_branches_batched

            return forward_branches_batched(self.model, src, state_mask, chunk, chunk_rows, self.pad_id)
        rows, length = _bucket(n, self.row_buckets), _bucket(max_len, self.length_buckets)
        g = self._get(src, rows, length, state)
        dev, src_len = self.device, state_mask.shape[1]

        # Dummy rows (n..rows) repeat the first branch's state row and read pad tokens; their outputs are dropped.
        idx = torch.tensor(chunk_rows + [chunk_rows[0]] * (rows - n), device=dev)
        ids = torch.full((rows, length), self.pad_id, dtype=torch.long)
        for i, b in enumerate(chunk):
            ids[i, : len(b)] = torch.tensor(b, dtype=torch.long)
        g.input_ids.copy_(ids.to(dev, non_blocking=True))
        smask = state_mask.index_select(0, idx).bool()  # [rows, src_len]
        g.position_ids.copy_(smask.sum(-1, keepdim=True) + torch.arange(length, device=dev))
        g.mask.zero_()
        g.mask[:, 0, :, state - src_len : state] = smask[:, None, :]
        g.mask[:, 0, :, state:] = torch.ones((length, length), dtype=torch.bool, device=dev).tril()
        self._fill_cache(src, idx, rows, src_len, state)

        if not g.eager and g.graph is None:
            self._capture(g)
            self._fill_cache(src, idx, rows, src_len, state)  # warmup (and a failed capture) mutated the states
        if g.eager:
            hidden = self._forward(g)  # in-place state updates only touch the buffers, refilled every call
        else:
            g.graph.replay()
            g.replays += 1
            hidden = g.output
        return [hidden[i, : len(b)].clone() for i, b in enumerate(chunk)]
