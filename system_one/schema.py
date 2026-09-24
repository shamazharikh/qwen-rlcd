"""Request and answer types (README §1, §5.2).

A request is a state plus named, typed questions:

    {"state": "...",
     "questions": {
        "department":  {"type": "choice", "instructions": "...", "criteria": {"billing": "...", ...}},
        "is_urgent":   {"type": "noul",   "instructions": "..."},
        "frustration": {"type": "score",  "instructions": "...", "criteria": ["Calm", ...]}}}

Question ids are never shown to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

QuestionType = Literal["choice", "score", "noul"]


@dataclass(frozen=True)
class Question:
    type: QuestionType
    instructions: str
    # choice: option key -> description; score: ordered level descriptions; noul: empty
    options: dict[str, str] | tuple[str, ...] = field(default_factory=dict)

    @property
    def num_options(self) -> int:
        return 2 if self.type == "noul" else len(self.options)


@dataclass(frozen=True)
class Request:
    state: str
    questions: dict[str, Question]

    @classmethod
    def from_dict(cls, data: dict) -> Request:
        if not isinstance(data.get("state"), str):
            raise ValueError("'state' must be a string")
        raw = data.get("questions")
        if not isinstance(raw, dict) or not raw:
            raise ValueError("'questions' must be a non-empty object keyed by question id")
        return cls(state=data["state"], questions={qid: _parse_question(qid, q) for qid, q in raw.items()})


def _parse_question(qid: str, q: dict) -> Question:
    kind, instructions, criteria = q.get("type"), q.get("instructions"), q.get("criteria")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(f"{qid}: 'instructions' must be a non-empty string")
    if kind == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError(f"{qid}: choice 'criteria' must map at least 2 option keys to descriptions")
        if not all(isinstance(k, str) and k and isinstance(v, str) for k, v in criteria.items()):
            raise ValueError(f"{qid}: choice option keys and descriptions must be strings")
        return Question("choice", instructions, dict(criteria))
    if kind == "score":
        if not isinstance(criteria, list) or len(criteria) < 2 or not all(isinstance(c, str) for c in criteria):
            raise ValueError(f"{qid}: score 'criteria' must be a list of at least 2 level descriptions")
        return Question("score", instructions, tuple(criteria))
    if kind == "noul":
        if criteria is not None:
            raise ValueError(f"{qid}: noul questions take no 'criteria'")
        return Question("noul", instructions)
    raise ValueError(f"{qid}: unknown question type {kind!r}")


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    score: float  # expected level index (0-based), may fall between levels
    probabilities: list[float]
    confidence: float


@dataclass(frozen=True)
class NoulAnswer:
    noul: float  # P(yes)


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer
