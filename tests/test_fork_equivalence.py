"""M0: prefix-fork reads must match independent full forward passes (PLAN.md §2).

Fast tests use a tiny randomly initialised Qwen3.5 hybrid text model. The real-weights test runs
`Qwen/Qwen3.5-0.8B-Base` and is enabled with `QWEN_RLCD_SLOW=1`. Tests run on CUDA when available,
which exercises the fla / causal-conv1d kernels if they are installed.
"""

import os
import random

import pytest
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from system_one.fork import expand_cache, fork_forward, forward_branches, prefill_state, sequential_reads

TOL = 1e-3  # PLAN.md acceptance threshold for real weights (fp32)
# Random tiny weights make the DeltaNet contribution small (zeroing its state moves reads ~1e-4), so the
# tiny-model tests need a much tighter bound to be able to catch a broken fork at all.
TINY_TOL = 1e-5
CONV_KERNEL = 4
PAD_ID = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=CONV_KERNEL,
    )
    return Qwen3_5TextModel(config).eval().float().to(DEVICE)


def _rand_ids(n, vocab=512):
    return [random.randrange(1, vocab) for _ in range(n)]


def assert_fork_matches(model, state, branches, tol=TINY_TOL):
    state_ids = torch.tensor([state], device=DEVICE)
    with torch.no_grad():
        forked = fork_forward(model, state_ids, branches, PAD_ID)
        reference = sequential_reads(model, state_ids, branches)
    diff = (forked - reference).abs().max().item()
    assert diff < tol, f"max abs diff {diff:.2e} >= {tol}"
    return forked


@pytest.mark.parametrize(
    "state_len, branch_lens",
    [
        (20, [5, 5, 5]),  # equal lengths, no padding
        (20, [1, 2, 3, 9]),  # branches shorter than the conv kernel
        (2, [4, 7]),  # state shorter than the conv kernel
        (100, [3, 70, 150]),  # branches crossing the 64-token delta-rule chunk boundary
    ],
)
def test_fork_matches_sequential(tiny_model, state_len, branch_lens):
    random.seed(state_len + sum(branch_lens))
    assert_fork_matches(tiny_model, _rand_ids(state_len), [_rand_ids(n) for n in branch_lens])


def test_read_independent_of_padding_and_siblings(tiny_model):
    random.seed(1)
    state, target = _rand_ids(30), _rand_ids(6)
    state_ids = torch.tensor([state], device=DEVICE)
    with torch.no_grad():
        alone = fork_forward(tiny_model, state_ids, [target], PAD_ID)[0]
        for extra in (7, 50):  # the sibling forces target to be right-padded by `extra` tokens
            batched = fork_forward(tiny_model, state_ids, [_rand_ids(6 + extra), target], PAD_ID)[1]
            assert (batched - alone).abs().max().item() < TINY_TOL


def test_expand_cache_forks_recurrent_state_and_leaves_source_intact(tiny_model):
    random.seed(2)
    state_ids = torch.tensor([_rand_ids(12)], device=DEVICE)
    branches = [_rand_ids(4), _rand_ids(8)]
    with torch.no_grad():
        cache = prefill_state(tiny_model, state_ids)
        expanded = expand_cache(cache, 3)
        for layer, src in zip(expanded.layers, cache.layers):
            for states in ("conv_states", "recurrent_states"):
                for k, t in getattr(src, states, {}).items():
                    if t is not None:
                        assert getattr(layer, states)[k].shape[0] == 3 and t.shape[0] == 1
        first = forward_branches(tiny_model, cache, 12, branches, PAD_ID)
        second = forward_branches(tiny_model, cache, 12, branches, PAD_ID)  # reuse: cache must be unmutated
    assert torch.equal(first, second)


@pytest.mark.parametrize("states", ["recurrent_states", "conv_states"])
def test_detects_unforked_linear_attention_state(tiny_model, monkeypatch, states):
    """Sensitivity check: if a DeltaNet state is not carried into the branches, the test must fail."""
    random.seed(3)
    state, branches = _rand_ids(20), [_rand_ids(5), _rand_ids(9)]

    def broken_expand(cache, n, batch_size=1):
        expanded = expand_cache(cache, n, batch_size)
        for layer in expanded.layers:
            for t in getattr(layer, states, {}).values():
                if t is not None:
                    t.zero_()
        return expanded

    monkeypatch.setattr("system_one.fork.expand_cache", broken_expand)
    with pytest.raises(AssertionError):
        assert_fork_matches(tiny_model, state, branches)


REAL_DTYPES = [torch.float32]
if DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
    REAL_DTYPES.append(torch.bfloat16)


@pytest.mark.skipif(os.environ.get("QWEN_RLCD_SLOW") != "1", reason="set QWEN_RLCD_SLOW=1 to run real weights")
@pytest.mark.parametrize("dtype", REAL_DTYPES, ids=str)
def test_fork_matches_sequential_qwen35_08b_base(dtype):
    repo = "Qwen/Qwen3.5-0.8B-Base"
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForImageTextToText.from_pretrained(repo, dtype=dtype).eval().to(DEVICE)
    text_model = model.model.language_model

    state = tok("<state>\nOur API started returning 500s 20 minutes ago; we can't process orders.\n</state>\n").input_ids
    question = "Question: Which team should handle this?\nOptions: billing, technical, sales\nAnswer: "
    options = [
        "billing: Payment or subscription issues",
        "technical: Bugs or integration problems",
        "sales: Pricing or account questions",
    ]
    branches = [tok(question + o).input_ids for o in options]
    branches.append(tok("Question: The message conveys urgency.\nAnswer (yes/no):").input_ids)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    state_ids = torch.tensor([state], device=DEVICE)
    with torch.no_grad():
        forked = fork_forward(text_model, state_ids, branches, pad_id).float()
        reference = sequential_reads(text_model, state_ids, branches).float()
    diff = (forked - reference).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(forked, reference, dim=-1).min().item()
    print(f"\nQwen3.5-0.8B-Base [{DEVICE}, {dtype}] fork vs sequential: max abs diff {diff:.2e}, min cos {cos:.6f}")
    if dtype == torch.float32:
        assert diff < TOL
    else:  # bf16: kernels pick different chunkings for different shapes, so compare direction only
        assert cos > 0.9999
