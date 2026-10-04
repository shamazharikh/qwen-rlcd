"""Two-level fork (state → question cache → answers) must match the one-level fork (PLAN.md §6).

Fast tests use the tiny random hybrid model from `test_fork_equivalence`; the real-weights test runs
`Qwen/Qwen3.5-0.8B-Base` with `QWEN_RLCD_SLOW=1`.
"""

import os
import random

import pytest
import torch

from system_one import fork
from system_one.fork import extend_cache, forward_branches_all, forward_question_tree, prefill_state
from tests.test_fork_equivalence import DEVICE, PAD_ID, TINY_TOL, TOL, _rand_ids, tiny_model  # noqa: F401


def _two_level_vs_one_level(model, state, questions, tol, chunk_size=None, min_prefix_saving=0):
    """Compare `cat(prefix_hidden, answer_hidden)` against the flat `prefix + answer` branch."""
    state_ids = torch.tensor([state], device=DEVICE)
    flat = [prefix + answer for prefix, answers in questions for answer in answers]
    with torch.no_grad():
        cache = prefill_state(model, state_ids)
        tree = forward_question_tree(model, cache, len(state), questions, PAD_ID, chunk_size, min_prefix_saving)
        one_level = forward_branches_all(model, cache, len(state), flat, PAD_ID)
    worst, i = 0.0, 0
    for prefix_hidden, answer_hidden in tree:
        for hidden in answer_hidden:
            joined = torch.cat([prefix_hidden, hidden])
            assert joined.shape == one_level[i].shape
            worst = max(worst, (joined - one_level[i]).abs().max().item())
            i += 1
    assert i == len(flat)
    assert worst < tol, f"max abs diff {worst:.2e} >= {tol}"
    return worst


@pytest.mark.parametrize(
    "state_len, shapes",
    [
        (20, [(6, [4, 4, 4])]),  # one question, equal-length answers
        (20, [(6, [1, 3, 9]), (11, [2, 2])]),  # several questions, answers shorter than the conv kernel
        (2, [(3, [5, 5])]),  # state shorter than the conv kernel
        (30, [(70, [3, 80])]),  # prefix and answer crossing the 64-token delta-rule chunk boundary
        (20, [(0, [5, 7])]),  # empty prefix: the question cache is just the state
    ],
)
@pytest.mark.parametrize("min_prefix_saving", [0, 10**9], ids=["two-level", "one-level-fallback"])
def test_two_level_matches_one_level(tiny_model, state_len, shapes, min_prefix_saving):
    """Both paths must return the same values; `min_prefix_saving` only decides which one runs."""
    random.seed(state_len + sum(p for p, _ in shapes))
    questions = [(_rand_ids(p), [_rand_ids(a) for a in answers]) for p, answers in shapes]
    _two_level_vs_one_level(tiny_model, _rand_ids(state_len), questions, TINY_TOL, min_prefix_saving=min_prefix_saving)


def test_saving_threshold_picks_the_path(tiny_model, monkeypatch):
    """A question below the threshold must not pay for the extra prefix forward."""
    random.seed(14)
    calls = []
    real_extend = fork.extend_cache
    monkeypatch.setattr("system_one.fork.extend_cache", lambda *a: (calls.append(1), real_extend(*a))[1])
    state_ids = torch.tensor([_rand_ids(20)], device=DEVICE)
    with torch.no_grad():
        cache = prefill_state(tiny_model, state_ids)
        # (K - 1) * prefix = 2 * 10 = 20 tokens saved
        fork.forward_question_tree(tiny_model, cache, 20, [(_rand_ids(10), [_rand_ids(4)] * 3)], PAD_ID,
                                   min_prefix_saving=21)
        assert calls == []
        fork.forward_question_tree(tiny_model, cache, 20, [(_rand_ids(10), [_rand_ids(4)] * 3)], PAD_ID,
                                   min_prefix_saving=20)
    assert len(calls) == 1


def test_chunking_answers_does_not_change_reads(tiny_model):
    random.seed(11)
    questions = [(_rand_ids(5), [_rand_ids(4) for _ in range(7)])]
    _two_level_vs_one_level(tiny_model, _rand_ids(25), questions, TINY_TOL, chunk_size=3)


def test_state_cache_is_reusable_across_questions(tiny_model):
    """Each question extends the state cache; the source must survive for the next question."""
    random.seed(12)
    state = _rand_ids(18)
    questions = [(_rand_ids(4), [_rand_ids(3), _rand_ids(5)]) for _ in range(3)]
    state_ids = torch.tensor([state], device=DEVICE)
    with torch.no_grad():
        cache = prefill_state(tiny_model, state_ids)
        first = forward_question_tree(tiny_model, cache, len(state), questions, PAD_ID, min_prefix_saving=0)
        second = forward_question_tree(tiny_model, cache, len(state), questions, PAD_ID, min_prefix_saving=0)
    for (p1, a1), (p2, a2) in zip(first, second):
        assert torch.equal(p1, p2) and all(torch.equal(x, y) for x, y in zip(a1, a2))


def test_detects_question_cache_that_drops_the_prefix(tiny_model, monkeypatch):
    """Sensitivity: if the answers fork from the state cache instead of the question cache, it must fail."""
    random.seed(13)
    questions = [(_rand_ids(6), [_rand_ids(4), _rand_ids(4)])]

    def broken_extend(model, cache, prefix_len, ids):
        _, hidden = extend_cache(model, cache, prefix_len, ids)
        return cache, hidden  # returns the un-extended state cache

    monkeypatch.setattr("system_one.fork.extend_cache", broken_extend)
    with pytest.raises(AssertionError):
        _two_level_vs_one_level(tiny_model, _rand_ids(20), questions, TINY_TOL)


@pytest.mark.skipif(os.environ.get("QWEN_RLCD_SLOW") != "1", reason="set QWEN_RLCD_SLOW=1 to run real weights")
def test_two_level_matches_one_level_qwen35_08b_base():
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    repo = "Qwen/Qwen3.5-0.8B-Base"
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.float32).eval().to(DEVICE)
    text_model = model.model.language_model

    state = tok("<state>\nOur API started returning 500s 20 minutes ago; we can't process orders.\n</state>\n").input_ids
    questions = [
        (
            tok("Question: Which team should handle this?\nOptions: billing, technical, sales\nAnswer: ").input_ids,
            [tok(o).input_ids for o in ("billing: Payment issues", "technical: Bugs", "sales: Pricing questions")],
        ),
        (
            tok("Question: The message conveys urgency.\nAnswer (yes/no):").input_ids,
            [tok(o).input_ids for o in (" yes", " no")],
        ),
    ]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    state_ids = torch.tensor([state], device=DEVICE)
    flat = [prefix + answer for prefix, answers in questions for answer in answers]
    with torch.no_grad():
        cache = prefill_state(text_model, state_ids)
        tree = forward_question_tree(text_model, cache, len(state), questions, pad_id, min_prefix_saving=0)
        one_level = forward_branches_all(text_model, cache, len(state), flat, pad_id)
    worst = max(
        (torch.cat([prefix_hidden, hidden]) - one_level[i]).abs().max().item()
        for i, (prefix_hidden, hidden) in enumerate(
            (prefix_hidden, hidden) for prefix_hidden, answers in tree for hidden in answers
        )
    )
    tokens_one, tokens_two = sum(map(len, flat)), sum(len(p) + sum(map(len, a)) for p, a in questions)
    print(f"\ntwo-level vs one-level: max abs diff {worst:.2e}; branch tokens {tokens_one} → {tokens_two}")
    assert worst < TOL
