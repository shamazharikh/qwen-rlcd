"""Google's hosted Gemini Decisions API (`gemini-decision-1`) as a reference decision model.

Same request shape as this repo (README §5.2: noul / choice / score with instructions and criteria), so a
`Request` maps over directly. Answers come back as this repo's typed answers, letting the zero-shot harness
score Gemini next to the local scorers. Docs: https://aistudio.google.com/docs/decisions-api

Needs GEMINI_API_KEY. Note the API clamps noul probabilities to [0.01, 0.99], which caps NLL on confident misses.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from system_one.schema import ChoiceAnswer, NoulAnswer, Question, Request, ScoreAnswer

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/decisions"


def _question_json(q: Question) -> dict:
    out = {"type": q.type, "instructions": q.instructions}
    if q.type == "choice":
        out["criteria"] = dict(q.options)
    elif q.type == "score":
        out["criteria"] = list(q.options)
    return out


def _answer(q: Question, a: dict):
    if q.type == "noul":
        return NoulAnswer(noul=float(a["noul"]))
    if q.type == "choice":
        probs = {k: float(a["probabilities"].get(k, 0.0)) for k in q.options}
        return ChoiceAnswer(choice=a["choice"], probabilities=probs, confidence=float(a["confidence"]))
    probs = [float(a["probabilities"].get(str(i), 0.0)) for i in range(len(q.options))]
    return ScoreAnswer(score=float(a["score"]), probabilities=probs, confidence=float(a["confidence"]))


class GeminiDecisions:
    """`predict(request) -> {qid: typed answer}`, mirroring `system_one.predict.predict`."""

    def __init__(self, model: str = "gemini-decision-1", api_key: str | None = None, retries: int = 5, timeout: float = 60):
        self.model, self.retries, self.timeout = model, retries, timeout
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("set GEMINI_API_KEY")
        self.usage_tokens = 0

    def predict(self, request: Request | dict) -> dict:
        if isinstance(request, dict):
            request = Request.from_dict(request)
        body = {
            "model": self.model,
            "input": request.state,
            "questions": {qid: _question_json(q) for qid, q in request.questions.items()},
        }
        data = self._post(body)
        self.usage_tokens += data.get("usage", {}).get("total_tokens", 0)
        return {qid: _answer(q, data["answers"][qid]) for qid, q in request.questions.items()}

    def _post(self, body: dict) -> dict:
        payload = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(ENDPOINT, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(f"Decisions API HTTP {e.code}: {detail}") from None
            except urllib.error.URLError:
                if attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                raise
