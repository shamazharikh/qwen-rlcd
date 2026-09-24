"""Text rendering for the state, question branches and zero-shot prompts (PLAN.md M1).

Branches are built as (prefix, answer) text pairs and tokenized separately by the scorers, so the
answer span is known exactly. Choice options are always rendered in sorted key order, so the text
the model sees does not depend on the order the caller listed them in. Score levels keep their
given order, which is part of their meaning.
"""

from __future__ import annotations

import string

from system_one.schema import Question

READ_TOKEN = "<|read|>"
LETTERS = string.ascii_uppercase
NOUL_ANSWERS = (" yes", " no")


def render_state(state: str) -> str:
    return f"<state>\n{state}\n</state>\n"


def option_keys(q: Question) -> list[str]:
    """Canonical option order; the order of every logit / probability vector for `q`."""
    if q.type == "choice":
        return sorted(q.options)
    if q.type == "score":
        return [str(i) for i in range(len(q.options))]
    return ["yes", "no"]


def _choice_lines(q: Question) -> list[tuple[str, str]]:
    """(label, description) per option in canonical order."""
    if q.type == "choice":
        return [(k, q.options[k]) for k in option_keys(q)]
    return [(f"Level {i + 1} of {len(q.options)}", desc) for i, desc in enumerate(q.options)]


def branch_prefix(q: Question) -> str:
    """Question text shared by every answer branch of `q`, including the compact option list (README §4)."""
    if q.type == "noul":
        return f"Question: {q.instructions}\nAnswer (yes/no):"
    if q.type == "choice":
        listing = "Options: " + ", ".join(option_keys(q))
    else:
        listing = "Levels: " + " | ".join(q.options)
    return f"Question: {q.instructions}\n{listing}\nAnswer:"


def branch_answers(q: Question) -> list[str]:
    """Answer text per option in canonical order; appended to `branch_prefix(q)`."""
    if q.type == "noul":
        return list(NOUL_ANSWERS)
    return [f" {label}: {desc}" for label, desc in _choice_lines(q)]


def letter_prompt(q: Question) -> str:
    """Zero-shot multiple-choice prompt read at its last token (README §7.7 baseline 1)."""
    if q.type == "noul":
        return branch_prefix(q)
    lines = _choice_lines(q)
    if len(lines) > len(LETTERS):
        raise ValueError(f"letter prompts support at most {len(LETTERS)} options, got {len(lines)}")
    body = "\n".join(f"{LETTERS[i]}. {label}: {desc}" for i, (label, desc) in enumerate(lines))
    return f"Question: {q.instructions}\nOptions:\n{body}\nAnswer:"
