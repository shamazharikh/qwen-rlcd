"""GeminiDecisions / VllmSystemOne request/response mapping, with the HTTP call mocked (no network, no key needed)."""
from system_one.remote import GeminiDecisions, VllmSystemOne
from system_one.schema import ChoiceAnswer, NoulAnswer, ScoreAnswer

REQUEST = {
    "state": "Our API is down and orders are failing.",
    "questions": {
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "tech": "Bugs"}},
        "urgent": {"type": "noul", "instructions": "The message conveys urgency."},
        "severity": {"type": "score", "instructions": "How severe?", "criteria": ["Low", "Medium", "High"]},
    },
}

RESPONSE = {
    "answers": {
        "team": {"type": "choice", "choice": "tech", "probabilities": {"billing": 0.1, "tech": 0.9}, "confidence": 0.9},
        "urgent": {"type": "noul", "noul": 0.97, "confidence": 0.97},
        "severity": {"type": "score", "score": 1.8, "probabilities": {"0": 0.05, "1": 0.1, "2": 0.85}, "confidence": 0.85},
    },
    "usage": {"total_tokens": 42},
}


def test_request_and_answers(monkeypatch):
    sent = {}

    def fake_post(self, body):
        sent.update(body)
        return RESPONSE

    monkeypatch.setattr(GeminiDecisions, "_post", fake_post)
    answers = GeminiDecisions(api_key="test").predict(REQUEST)

    assert sent["model"] == "gemini-decision-1"
    assert sent["input"] == REQUEST["state"]
    assert sent["questions"]["team"] == REQUEST["questions"]["team"]
    assert sent["questions"]["severity"]["criteria"] == ["Low", "Medium", "High"]
    assert "criteria" not in sent["questions"]["urgent"]

    assert answers["team"] == ChoiceAnswer("tech", {"billing": 0.1, "tech": 0.9}, 0.9)
    assert answers["urgent"] == NoulAnswer(0.97)
    assert answers["severity"] == ScoreAnswer(1.8, [0.05, 0.1, 0.85], 0.85)


def test_vllm_request_sorted_options_and_answers(monkeypatch):
    sent = {}

    def fake_post(self, body):
        sent.update(body)
        resp = {k: dict(v) for k, v in RESPONSE["answers"].items()}
        resp["urgent"]["probabilities"] = {"yes": 0.97, "no": 0.03}
        resp["severity"]["probabilities"] = {0: 0.05, 1: 0.1, 2: 0.85}  # int keys also accepted
        return {"answers": resp, "usage": {"input_tokens": 42, "output_tokens": 0}}

    monkeypatch.setattr(VllmSystemOne, "_post", fake_post)
    request = {**REQUEST, "questions": {**REQUEST["questions"]}}
    request["questions"]["team"] = {**REQUEST["questions"]["team"], "criteria": {"tech": "Bugs", "billing": "Payments"}}
    client = VllmSystemOne("Qwen/Qwen3.5-9B-Base", base_url="http://vllm:8000/")
    answers = client.predict(request)

    assert client.url == "http://vllm:8000/v1/systemone"
    assert sent["state"] == REQUEST["state"] and sent["model"] == "Qwen/Qwen3.5-9B-Base"
    assert list(sent["questions"]["team"]["criteria"]) == ["billing", "tech"]  # A/B labels follow sorted keys
    assert sent["questions"]["urgent"]["criteria"] is None
    assert answers["severity"] == ScoreAnswer(1.8, [0.05, 0.1, 0.85], 0.85)
    assert answers["urgent"] == NoulAnswer(0.97)
    assert client.usage_tokens == 42
