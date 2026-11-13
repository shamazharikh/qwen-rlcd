"""GeminiDecisions request/response mapping, with the HTTP call mocked (no network, no key needed)."""
from system_one.gemini import GeminiDecisions
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
