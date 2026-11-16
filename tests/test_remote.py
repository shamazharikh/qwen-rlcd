"""GeminiDecisions / VllmSystemOne request/response mapping, with the HTTP call mocked (no network, no key needed)."""
from system_one.remote import GeminiDecisions, VllmSystemOne
from system_one.schema import ChoiceAnswer, NoulAnswer, Request, ScoreAnswer

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


def test_chat_scorer_logprobs_and_hard_fallback(monkeypatch):
    from system_one.chat import ChatCompletionScorer, parse, render

    replies = iter([
        # choice with logprobs: letters in sorted key order (billing=A, tech=B)
        {"choices": [{"message": {"content": "B"}, "logprobs": {"content": [{"top_logprobs": [
            {"token": "B", "logprob": -0.1}, {"token": "A", "logprob": -2.4}, {"token": " B", "logprob": -5.0}]}]}}],
         "usage": {"prompt_tokens": 50, "completion_tokens": 1}},
        {"choices": [{"message": {"content": "Yes."}}], "usage": {"prompt_tokens": 30, "completion_tokens": 2}},  # no logprobs
        {"choices": [{"message": {"content": "I think it's high"}}], "usage": {"prompt_tokens": 40, "completion_tokens": 5}},  # unparseable
    ])  # fmt: skip
    monkeypatch.setattr(ChatCompletionScorer, "_post", lambda self, body: next(replies))
    scorer = ChatCompletionScorer("gemini-2.5-flash-lite", api_key="test")
    answers = scorer.predict(REQUEST)

    assert answers["team"].choice == "tech" and 0.9 < answers["team"].probabilities["tech"] < 0.95
    assert answers["urgent"] == NoulAnswer(1.0)
    assert answers["severity"].probabilities == [1 / 3] * 3
    assert dict(scorer.usage) == {"prompt_tokens": 120, "completion_tokens": 8, "calls": 3, "hard_answers": 2, "invalid": 1}
    assert render("s", Request.from_dict(REQUEST).questions["team"])[1] == ["A", "B"]
    assert parse("Answer: no", ["yes", "no"]) == "no" and parse("C) maybe", ["A", "B"]) is None


def test_vllm_body_sends_media_urls(tmp_path):
    from PIL import Image

    from system_one.remote import VllmSystemOne

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    req = Request.from_dict({
        "state": "Look.", "images": [Image.new("RGB", (4, 4)), "https://example.com/a.png"], "videos": [str(video)],
        "questions": {"q": {"type": "noul", "instructions": "Is it red?"}},
    })  # fmt: skip
    body = VllmSystemOne("m", "http://localhost:1").body(req)
    assert body["images"][0].startswith("data:image/png;base64,") and body["images"][1] == "https://example.com/a.png"
    assert body["videos"] == ["file://" + str(video)]


def test_chat_scorer_sends_media_and_reads_letter_logprobs(monkeypatch):
    from PIL import Image

    from system_one.chat import ChatCompletionScorer

    sent = {}

    def fake_post(self, body):
        sent.update(body)
        top = [{"token": "B", "logprob": -0.1}, {"token": "A", "logprob": -2.5}]
        return {"choices": [{"message": {"content": "B"}, "logprobs": {"content": [{"top_logprobs": top}]}}], "usage": {}}

    monkeypatch.setattr(ChatCompletionScorer, "_post", fake_post)
    scorer = ChatCompletionScorer("m", "http://localhost:8000/v1", max_tokens=1, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    req = Request.from_dict({"state": "Look.", "images": [Image.new("RGB", (4, 4))], "videos": ["/tmp/v.mp4"],
                             "questions": {"q": {"type": "choice", "instructions": "Colour?", "criteria": {"a": "", "b": ""}}}})  # fmt: skip
    ans = scorer.predict(req)["q"]
    parts = sent["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["image_url", "video_url", "text"]
    assert parts[1]["video_url"]["url"] == "file:///tmp/v.mp4"
    assert sent["chat_template_kwargs"] == {"enable_thinking": False} and sent["max_tokens"] == 1
    assert ans.choice == "b" and ans.probabilities["b"] > 0.9


def test_gemini_decisions_body_inlines_media(tmp_path):
    from PIL import Image

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x01")
    req = Request.from_dict({"state": "Look.", "images": [Image.new("RGB", (4, 4))], "videos": [str(video)],
                             "questions": {"q": {"type": "noul", "instructions": "Is it red?"}}})  # fmt: skip
    body = GeminiDecisions(api_key="k", resolution="low").body(req)
    text, image, vid = body["input"]
    assert text == {"type": "text", "text": "Look."}
    assert image["type"] == "image" and image["mime_type"] == "image/png" and image["resolution"] == "low"
    assert vid == {"type": "video", "data": "AAE=", "mime_type": "video/mp4", "resolution": "low"}


def test_gemini_chat_uses_native_api_for_media(monkeypatch, tmp_path):
    from system_one.chat import ChatCompletionScorer

    calls = []

    def fake_post(self, body, url=None, headers=None):
        calls.append((url, body))
        return {"candidates": [{"content": {"parts": [{"text": "yes"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 1}}  # fmt: skip

    monkeypatch.setattr(ChatCompletionScorer, "_post", fake_post)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    scorer = ChatCompletionScorer("gemini-2.5-flash-lite", api_key="k")
    req = Request.from_dict({"state": "Watch.", "videos": [str(video)], "questions": {"q": {"type": "noul", "instructions": "Fire?"}}})
    ans = scorer.predict(req)["q"]
    url, body = calls[0]
    assert url.endswith("/v1beta/models/gemini-2.5-flash-lite:generateContent")
    assert body["contents"][0]["parts"][0] == {"inline_data": {"data": "AA==", "mime_type": "video/mp4"}}
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert ans.noul == 1.0 and scorer.usage["prompt_tokens"] == 300
