"""M1: request schema, templates, scorers and typed answers (PLAN.md §3).

Uses a tiny random Qwen3.5 hybrid text model with the real Qwen3.5 tokenizer, so the answers are
meaningless but every code path, and the invariance properties, are exercised.
"""

import math
import random

import pytest
import torch
from torch import nn
from transformers import AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from system_one import templates
from system_one.predict import confidence, predict
from system_one.schema import ChoiceAnswer, NoulAnswer, Request, ScoreAnswer
from system_one.scorers import Backbone, DecisionHeads, HeadScorer, LetterScorer, LikelihoodScorer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOL = 1e-5
VOCAB = 248320  # Qwen3.5 embedding rows (the tokenizer uses 248077 of them)

REQUEST = {
    "state": "Our API started returning 500s 20 minutes ago; we can't process orders.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "technical": "Bugs or integration problems",
                "billing": "Payment or subscription issues",
                "sales": "Pricing or account questions",
            },
        },
        "is_urgent": {"type": "noul", "instructions": "The message conveys urgency."},
        "frustration": {
            "type": "score",
            "instructions": "How frustrated does the customer appear?",
            "criteria": ["Calm", "Frustrated but civil", "Very angry"],
        },
    },
}


@pytest.fixture(scope="module")
def backbone():
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_num_key_heads=4, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16,
    )  # fmt: skip
    model = Qwen3_5TextModel(config).eval().to(DEVICE)
    lm_head = nn.Linear(64, VOCAB, bias=False).to(DEVICE)
    lm_head.weight = model.embed_tokens.weight  # tied, like the real checkpoint
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
    return Backbone(model, lm_head, tok, chunk_size=4)  # small chunks so chunking is exercised


@pytest.fixture(scope="module")
def scorers(backbone):
    torch.manual_seed(1)
    return {
        "letter": LetterScorer(backbone),
        "sum": LikelihoodScorer(backbone),
        "mean-pmi": LikelihoodScorer(backbone, normalize="mean", pmi=True),
        "head": HeadScorer(backbone, DecisionHeads(64).to(DEVICE)),
    }


def _shuffled(request: dict, seed: int) -> dict:
    rng = random.Random(seed)
    questions = list(request["questions"].items())
    rng.shuffle(questions)
    out = {}
    for qid, q in questions:
        q = dict(q)
        if isinstance(q.get("criteria"), dict):
            items = list(q["criteria"].items())
            rng.shuffle(items)
            q["criteria"] = dict(items)
        out[qid] = q
    return {"state": request["state"], "questions": out}


def _flat(answer) -> list[float]:
    if isinstance(answer, NoulAnswer):
        return [answer.noul]
    probs = answer.probabilities
    return [probs[k] for k in sorted(probs)] if isinstance(probs, dict) else list(probs)


# --- schema / templates -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"questions": {"q": {"type": "noul", "instructions": "x"}}},
        {"state": "s", "questions": {}},
        {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "only one"}}}},
        {"state": "s", "questions": {"q": {"type": "score", "instructions": "x", "criteria": {"a": "b"}}}},
        {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x", "criteria": ["a", "b"]}}},
        {"state": "s", "questions": {"q": {"type": "rank", "instructions": "x"}}},
        {"state": "s", "questions": {"q": {"type": "noul", "instructions": " "}}},
    ],
)
def test_request_validation(bad):
    with pytest.raises(ValueError):
        Request.from_dict(bad)


def test_rendered_text_is_independent_of_option_order():
    a = Request.from_dict(REQUEST).questions["department"]
    b = Request.from_dict(_shuffled(REQUEST, 0)).questions["department"]
    assert list(a.options) != list(b.options)
    assert templates.option_keys(a) == ["billing", "sales", "technical"]
    for render in (templates.branch_prefix, templates.branch_answers, templates.letter_prompt):
        assert render(a) == render(b)


def test_letter_prompt_rejects_too_many_options():
    q = Request.from_dict(
        {"state": "", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {f"o{i}": "" for i in range(27)}}}}
    ).questions["q"]
    with pytest.raises(ValueError):
        templates.letter_prompt(q)


def test_confidence_bounds():
    assert confidence(torch.tensor([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert confidence(torch.full((7,), 1 / 7)) == pytest.approx(0.0, abs=1e-6)


# --- scorers ------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["letter", "sum", "mean-pmi", "head"])
def test_answers_are_well_formed(scorers, name):
    answers = predict(REQUEST, scorers[name])
    assert set(answers) == set(REQUEST["questions"])
    dep, urgent, frus = answers["department"], answers["is_urgent"], answers["frustration"]
    assert isinstance(dep, ChoiceAnswer) and isinstance(urgent, NoulAnswer) and isinstance(frus, ScoreAnswer)
    assert set(dep.probabilities) == {"billing", "sales", "technical"}
    assert dep.choice == max(dep.probabilities, key=dep.probabilities.get)
    for probs in (list(dep.probabilities.values()), frus.probabilities):
        assert math.isclose(sum(probs), 1.0, abs_tol=1e-5)
    assert 0.0 <= dep.confidence <= 1.0 and 0.0 <= frus.confidence <= 1.0
    assert 0.0 <= frus.score <= 2.0 and 0.0 <= urgent.noul <= 1.0


@pytest.mark.parametrize("name", ["letter", "sum", "mean-pmi", "head"])
def test_invariant_to_question_and_option_order(scorers, name):
    base = predict(REQUEST, scorers[name])
    for seed in range(3):
        shuffled = predict(_shuffled(REQUEST, seed), scorers[name])
        for qid in base:
            diff = max(abs(x - y) for x, y in zip(_flat(base[qid]), _flat(shuffled[qid])))
            assert diff < TOL, f"{qid}: {diff:.2e}"


@pytest.mark.parametrize("name", ["letter", "sum", "mean-pmi", "head"])
def test_fan_out_matches_single_question_requests(scorers, name):
    """Asking all questions at once gives the same answers as asking each alone (README §1)."""
    together = predict(REQUEST, scorers[name])
    for qid, q in REQUEST["questions"].items():
        alone = predict({"state": REQUEST["state"], "questions": {qid: q}}, scorers[name])[qid]
        diff = max(abs(x - y) for x, y in zip(_flat(together[qid]), _flat(alone)))
        assert diff < TOL, f"{qid}: {diff:.2e}"


def test_likelihood_matches_unforked_computation(backbone, scorers):
    """Summed answer log-likelihood from the fork equals a plain full forward over state + branch."""
    request = Request.from_dict(REQUEST)
    logits = scorers["sum"](request)["department"]
    q = request.questions["department"]
    state = backbone.encode(templates.render_state(request.state))
    prefix = backbone.encode(templates.branch_prefix(q))
    expected = []
    with torch.no_grad():
        for answer in templates.branch_answers(q):
            ans = backbone.encode(answer)
            ids = torch.tensor([state + prefix + ans], device=DEVICE)
            hidden = backbone.text_model(input_ids=ids, use_cache=False).last_hidden_state[0]
            logprobs = torch.log_softmax(backbone.lm_head(hidden).float(), dim=-1)
            start = len(state) + len(prefix)
            expected.append(sum(logprobs[start + i - 1, t].item() for i, t in enumerate(ans)))
    assert torch.allclose(logits.cpu(), torch.tensor(expected), atol=1e-3)


def test_pmi_subtracts_empty_state_score(backbone):
    request = Request.from_dict(REQUEST)
    plain = LikelihoodScorer(backbone)(request)
    pmi = LikelihoodScorer(backbone, pmi=True)(request)
    empty = LikelihoodScorer(backbone)(Request(state="", questions=request.questions))
    for qid in plain:
        assert torch.allclose(pmi[qid], plain[qid] - empty[qid], atol=1e-4)


def test_read_token_fits_in_existing_embedding(scorers):
    assert scorers["head"].read_id == 248077


def test_temperature_flattens_distribution(scorers):
    sharp = predict(REQUEST, scorers["sum"])["department"]
    flat = predict(REQUEST, scorers["sum"], temperature=lambda kind, k: 100.0)["department"]
    assert flat.confidence < sharp.confidence
    assert flat.choice == sharp.choice
