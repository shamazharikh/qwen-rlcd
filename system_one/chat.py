"""Generative LLM baseline: ask a chat-completion model for the label, one question per call.

Works with any OpenAI-compatible `/chat/completions` endpoint. The default is Gemini's
(`https://generativelanguage.googleapis.com/v1beta/openai`, key in GEMINI_API_KEY), e.g. `gemini-2.5-flash-lite`
or `gemini-3.1-flash-lite`. Options are lettered in canonical (sorted-key) order, as for `LetterScorer`.

Probabilities come from the first answer token's `top_logprobs` when the endpoint returns them. Gemini's
OpenAI-compatible endpoint generally doesn't, so answers are then hard (one-hot): `hard_answers` counts those
and the eval reports accuracy without calibration. Billed tokens (prompt, completion incl. thinking) are summed
in `usage` for cost per call.

Requests with media send them as content parts ahead of the text (`image_url` / `video_url`, as vLLM's
OpenAI server accepts), so the same scorer is the vLLM image/video baseline: `--scorers vllm-chat` points it at
a local vLLM server, with thinking off (`chat_template_kwargs`) and one answer token. Gemini's OpenAI-compatible
endpoint takes no video, so for Gemini models a request with media goes to the native `generateContent` API
instead, with images and videos inline and thinking off.
"""

from __future__ import annotations

import json
import math
import os
import re
import string
import time
import urllib.error
import urllib.request
from collections import Counter

from system_one.schema import ChoiceAnswer, NoulAnswer, Question, Request, ScoreAnswer
from system_one.templates import option_keys

LETTERS = string.ascii_uppercase
SYSTEM = "You are a precise classifier. Reply with only the requested label and nothing else."


def render(state: str, q: Question) -> tuple[str, list[str]]:
    """User message and the label set, in canonical option order."""
    if q.type == "noul":
        return f"{state}\n\nStatement: {q.instructions}\nIs the statement true? Answer yes or no.", ["yes", "no"]
    if q.type == "choice":
        keys = option_keys(q)
        labels = list(LETTERS[: len(keys)])
        body = "\n".join(f"{l}. {k}: {q.options[k]}" if q.options[k] else f"{l}. {k}" for l, k in zip(labels, keys))
        return f"{state}\n\nQuestion: {q.instructions}\nOptions:\n{body}\nAnswer with the letter only.", labels
    labels = [str(i) for i in range(len(q.options))]
    body = "\n".join(f"{i}: {desc}" for i, desc in enumerate(q.options))
    return f"{state}\n\nQuestion: {q.instructions}\nLevels (lowest to highest):\n{body}\nAnswer with the level number only.", labels


def parse(text: str, labels: list[str]) -> str | None:
    """First label that appears as a standalone token in the reply (case-insensitive for yes/no)."""
    for tok in re.findall(r"[A-Za-z]+|\d+", text or ""):
        for label in labels:
            if tok == label or (label in ("yes", "no") and tok.lower() == label):
                return label
    return None


def label_probs(top_logprobs: list[dict], labels: list[str]) -> list[float] | None:
    mass = dict.fromkeys(labels, 0.0)
    for entry in top_logprobs or []:
        label = parse(entry.get("token", "").strip(), labels)
        if label is not None:
            mass[label] += math.exp(entry["logprob"])
    total = sum(mass.values())
    return [mass[l] / total for l in labels] if total > 0 else None


class ChatCompletionScorer:
    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        logprobs: bool = True,
        max_tokens: int = 16,
        retries: int = 6,
        timeout: float = 180,
        extra_body: dict | None = None,
        top_logprobs: int = 10,
    ):
        self.model, self.url = model, base_url.rstrip("/") + "/chat/completions"
        local = base_url.startswith(("http://localhost", "http://127.", "http://0.0.0.0"))
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ("EMPTY" if local else None)
        if not self.api_key:
            raise RuntimeError("set GEMINI_API_KEY (or OPENAI_API_KEY)")
        self.reasoning_effort, self.logprobs, self.max_tokens = reasoning_effort, logprobs, max_tokens
        self.extra_body, self.top_logprobs = extra_body or {}, top_logprobs
        self.retries, self.timeout = retries, timeout
        self.usage = Counter()  # prompt_tokens, completion_tokens, calls, hard_answers, invalid, blocked

    def predict(self, request: Request | dict) -> dict:
        if isinstance(request, dict):
            request = Request.from_dict(request)
        if request.has_media and "generativelanguage.googleapis.com" in self.url:
            parts = _gemini_parts(request, self.api_key)
            return {qid: self._ask_gemini_native(request.state, q, parts) for qid, q in request.questions.items()}
        media = _media_parts(request)
        return {qid: self._ask(request.state, q, media) for qid, q in request.questions.items()}

    def _ask_gemini_native(self, state: str, q: Question, media_parts: list):
        """Gemini `generateContent` with inline media (hard answers: no logprobs requested)."""
        text, labels = render(state, q)
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [*media_parts, {"text": text}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": self.max_tokens, "thinkingConfig": {"thinkingBudget": 0}},
        }
        base = self.url.rsplit("/openai/", 1)[0]
        data = self._post(body, url=f"{base}/models/{self.model}:generateContent", headers={"x-goog-api-key": self.api_key})
        usage = data.get("usageMetadata") or {}
        self.usage.update(prompt_tokens=usage.get("promptTokenCount", 0), completion_tokens=usage.get("candidatesTokenCount", 0)
                          + usage.get("thoughtsTokenCount", 0), calls=1, hard_answers=1)  # fmt: skip
        cand = (data.get("candidates") or [{}])[0]
        reply = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []))
        if cand.get("finishReason") not in (None, "STOP", "MAX_TOKENS"):
            self.usage["blocked"] += 1
        picked = parse(reply, labels)
        if picked is None:
            self.usage["invalid"] += 1
            probs = [1.0 / len(labels)] * len(labels)
        else:
            probs = [1.0 if l == picked else 0.0 for l in labels]
        return _typed(q, probs)

    def _ask(self, state: str, q: Question, media: list | None = None):
        text, labels = render(state, q)
        content = [*media, {"type": "text", "text": text}] if media else text
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.logprobs:
            body["logprobs"], body["top_logprobs"] = True, self.top_logprobs
        data = self._post(body)
        usage = data.get("usage") or {}
        self.usage.update(prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0), calls=1)
        choice = data["choices"][0]
        probs = None
        content_lp = ((choice.get("logprobs") or {}).get("content")) or []
        if content_lp:
            probs = label_probs(content_lp[0].get("top_logprobs"), labels)
        if probs is None:
            self.usage["hard_answers"] += 1
            # A safety-blocked or empty reply has no message/content: counted as invalid (scored wrong).
            if choice.get("finish_reason") not in (None, "stop", "length"):
                self.usage["blocked"] += 1
            picked = parse((choice.get("message") or {}).get("content"), labels)
            if picked is None:
                self.usage["invalid"] += 1
                probs = [1.0 / len(labels)] * len(labels)  # unparseable: uniform (scored as wrong by the eval)
            else:
                probs = [1.0 if l == picked else 0.0 for l in labels]
        return _typed(q, probs)

    def _post(self, body: dict, url: str | None = None, headers: dict | None = None) -> dict:
        payload = json.dumps(body)
        headers = headers or {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                url or self.url, data=payload.encode(), method="POST", headers={"Content-Type": "application/json", **headers},
            )  # fmt: skip
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code == 400 and '"logprobs"' in payload and "logprob" in detail.lower():
                    self.logprobs = False  # endpoint rejects logprobs: fall back to hard answers for the rest
                    body = {k: v for k, v in json.loads(payload).items() if k not in ("logprobs", "top_logprobs")}
                    payload = json.dumps(body)
                    continue
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(min(2**attempt, 30))
                    continue
                raise RuntimeError(f"chat completion HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 30))
                    continue
                raise


def _media_parts(request: Request) -> list:
    from system_one.remote import _media_url

    parts = [{"type": "image_url", "image_url": {"url": _media_url(i, "image")}} for i in request.images]
    return parts + [{"type": "video_url", "video_url": {"url": _media_url(v, "video")}} for v in request.videos]


def _gemini_parts(request: Request, api_key: str) -> list:
    """Images inline; local videos uploaded once to the Files API and referenced by URI in every call."""
    from system_one.gemini_files import upload
    from system_one.remote import MIME, inline_media

    parts = [{"inline_data": inline_media(i, "image")} for i in request.images]
    for v in request.videos:
        if isinstance(v, str) and not v.startswith(("data:", "http://", "https://")):
            path = v[7:] if v.startswith("file://") else v
            mime = MIME[os.path.splitext(path)[1].lower()]
            parts.append({"file_data": {"file_uri": upload(path, mime, api_key), "mime_type": mime}})
        else:
            parts.append({"inline_data": inline_media(v, "video")})
    return parts


def _typed(q: Question, probs: list[float]):
    if q.type == "noul":
        return NoulAnswer(noul=probs[0])
    if q.type == "choice":
        keys = option_keys(q)
        best = max(range(len(keys)), key=probs.__getitem__)
        return ChoiceAnswer(choice=keys[best], probabilities=dict(zip(keys, probs)), confidence=probs[best])
    return ScoreAnswer(score=sum(i * p for i, p in enumerate(probs)), probabilities=probs, confidence=max(probs))
