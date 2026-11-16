"""Hosted decision APIs that take the System One request shape (README §5.2) and return typed answers.

- `GeminiDecisions`: Google's Gemini Decisions API (`gemini-decision-1`). Docs: https://aistudio.google.com/docs/decisions-api
- `VllmSystemOne`: vLLM's `POST /v1/systemone` (vLLM >= 0.32) serving any supported checkpoint, including the
  Qwen3.5 models here. It applies the chat template, labels options A-Z / yes,no / 0-9, and softmaxes the
  first reply token's label logprobs: the same idea as `LetterScorer`, but with one read per question.
  Launch: `vllm serve Qwen/Qwen3.5-0.8B-Base --logprobs-mode raw_logprobs --enable-prefix-caching`.

Both expose `predict(request) -> {qid: typed answer}`, mirroring `system_one.predict.predict`, so the
zero-shot harness can score them on the same examples as the local scorers.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from system_one.schema import ChoiceAnswer, NoulAnswer, Question, Request, ScoreAnswer


def _criteria(q: Question):
    if q.type == "choice":
        return dict(q.options)
    if q.type == "score":
        return list(q.options)
    return None


def _answer(q: Question, a: dict):
    if q.type == "noul":
        return NoulAnswer(noul=float(a["noul"]))
    probs = {str(k): float(v) for k, v in a["probabilities"].items()}
    if q.type == "choice":
        return ChoiceAnswer(
            choice=a["choice"], probabilities={k: probs.get(k, 0.0) for k in q.options}, confidence=float(a["confidence"])
        )
    levels = [probs.get(str(i), 0.0) for i in range(len(q.options))]
    return ScoreAnswer(score=float(a["score"]), probabilities=levels, confidence=float(a["confidence"]))


class RemoteDecisions:
    """POST a decision request as JSON, retrying 429/5xx and connection errors with backoff."""

    url: str

    def __init__(self, model: str, headers: dict | None = None, retries: int = 5, timeout: float = 60):
        self.model, self.headers, self.retries, self.timeout = model, headers or {}, retries, timeout
        self.usage_tokens = 0

    def body(self, request: Request) -> dict:
        raise NotImplementedError

    def predict(self, request: Request | dict) -> dict:
        if isinstance(request, dict):
            request = Request.from_dict(request)
        data = self._post(self.body(request))
        usage = data.get("usage") or {}
        self.usage_tokens += usage.get("total_tokens", usage.get("input_tokens", 0))
        return {qid: _answer(q, data["answers"][qid]) for qid, q in request.questions.items()}

    def _post(self, body: dict) -> dict:
        payload = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", **self.headers}
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(f"{type(self).__name__} HTTP {e.code}: {detail}") from None
            except urllib.error.URLError:
                if attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                raise


class GeminiDecisions(RemoteDecisions):
    """Gemini Decisions API. Needs GEMINI_API_KEY. Noul outputs are clamped to [0.01, 0.99] by the API."""

    url = "https://generativelanguage.googleapis.com/v1beta/decisions"

    def __init__(self, model: str = "gemini-decision-1", api_key: str | None = None, **kwargs):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("set GEMINI_API_KEY")
        super().__init__(model, {"x-goog-api-key": api_key}, **kwargs)

    def body(self, request: Request) -> dict:
        questions = {}
        for qid, q in request.questions.items():
            questions[qid] = {"type": q.type, "instructions": q.instructions}
            if q.type != "noul":
                questions[qid]["criteria"] = _criteria(q)
        return {"model": self.model, "input": request.state, "questions": questions}


class VllmSystemOne(RemoteDecisions):
    """vLLM `POST /v1/systemone`. Options are labeled A-Z in the order sent, so they are sent sorted by key
    (this repo's canonical order) to keep answers independent of the caller's order. Limits: 64 questions,
    26 options, 10 score levels."""

    def __init__(self, model: str = "Qwen/Qwen3.5-0.8B-Base", base_url: str | None = None, **kwargs):
        base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
        self.url = base_url.rstrip("/") + "/v1/systemone"
        super().__init__(model, **kwargs)

    def body(self, request: Request) -> dict:
        questions = {}
        for qid, q in request.questions.items():
            criteria = _criteria(q)
            if q.type == "choice":
                criteria = dict(sorted(criteria.items()))
            questions[qid] = {"type": q.type, "instructions": q.instructions, "criteria": criteria}
        body = {"model": self.model, "state": request.state, "questions": questions}
        if request.images:
            body["images"] = [_media_url(i, "image") for i in request.images]
        if request.videos:
            body["videos"] = [_media_url(v, "video") for v in request.videos]
        return body


def _media_url(src, kind: str) -> str:
    """URL for vLLM: http(s)/data URLs pass through, local paths become file:// (start vLLM with
    --allowed-local-media-path), PIL images become PNG data URLs."""
    if isinstance(src, str):
        return src if src.startswith(("http://", "https://", "data:", "file://")) else "file://" + os.path.abspath(src)
    if kind == "image":
        import base64
        import io

        buf = io.BytesIO()
        src.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    raise TypeError(f"can't send {type(src).__name__} as a {kind} URL")
