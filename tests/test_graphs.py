"""`GraphBranchRunner` must give the same branch hidden states as the eager fork.

On CPU this checks the static path (bucket padding, KV layout, precomputed masks) run eagerly; on CUDA it
also captures and replays graphs, including reuse of a captured graph with new inputs.
"""

import random

import pytest
import torch

from system_one.fork import forward_branches_batched, prefill_state, prefill_states
from system_one.graphs import GraphBranchRunner
from tests.test_fork_equivalence import DEVICE, PAD_ID, TINY_TOL, _rand_ids, tiny_model  # noqa: F401


def _request(seed, state_lens, branch_counts, max_branch=12):
    random.seed(seed)
    states = [_rand_ids(n) for n in state_lens]
    branches, rows = [], []
    for r, k in enumerate(branch_counts):
        for _ in range(k):
            branches.append(_rand_ids(random.randint(1, max_branch)))
            rows.append(r)
    return states, branches, rows


def _check(model, runner, states, branches, rows):
    with torch.no_grad():
        cache, mask = prefill_states(model, states, PAD_ID)
        ref = forward_branches_batched(model, cache, mask, branches, rows, PAD_ID)
        got = runner(cache, mask, branches, rows)
    assert len(got) == len(ref)
    for i, (a, b) in enumerate(zip(got, ref)):
        assert a.shape == b.shape, i
        assert torch.isfinite(a).all()
        diff = (a - b).abs().max().item()
        assert diff < TINY_TOL, (i, diff)


@pytest.mark.parametrize("state_multiple", [1, 16, 64])
def test_runner_matches_batched_fork(tiny_model, state_multiple):
    runner = GraphBranchRunner(tiny_model, PAD_ID, row_buckets=(2, 4, 8), length_buckets=(4, 16), state_multiple=state_multiple)
    _check(tiny_model, runner, *_request(3, (7, 25, 1, 40), (3, 1, 2, 4)))  # 10 branches: chunks of 8 + 2


def test_runner_reuses_shapes_and_grows_capacity(tiny_model):
    runner = GraphBranchRunner(tiny_model, PAD_ID, row_buckets=(4,), length_buckets=(16,), state_multiple=32)
    _check(tiny_model, runner, *_request(0, (10, 20), (2, 2)))
    _check(tiny_model, runner, *_request(1, (5, 30), (1, 3)))  # same buckets, new data: replay
    assert len(runner.graphs) == 1
    _check(tiny_model, runner, *_request(2, (70,), (3,)))  # longer state: buffers regrow
    assert runner.capacity == 96
    if runner.use_graphs:
        assert runner.captures == 2 and runner.graphs[(4, 16, 96)].graph is not None


def test_runner_single_state_and_long_branch_fallback(tiny_model):
    runner = GraphBranchRunner(tiny_model, PAD_ID, row_buckets=(4,), length_buckets=(8,), state_multiple=16)
    random.seed(5)
    state = _rand_ids(13)
    branches = [_rand_ids(n) for n in (3, 8, 20)]  # 20 > largest length bucket: eager fallback for that chunk
    with torch.no_grad():
        cache = prefill_state(tiny_model, torch.tensor([state], device=DEVICE))
        mask = torch.ones((1, len(state)), dtype=torch.long, device=DEVICE)
        ref = forward_branches_batched(tiny_model, cache, mask, branches, [0] * 3, PAD_ID)
        got = runner(cache, mask, branches, [0] * 3)
    for a, b in zip(got, ref):
        assert (a - b).abs().max().item() < TINY_TOL
