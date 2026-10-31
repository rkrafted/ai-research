"""Needle and multi-hop scoring for the long-context set.

Facts are fictional (``@example.com``, ``555-01xx``).  A probe passes only when
every expected string is present in the model's line for that question.
"""
from __future__ import annotations

import re

from . import corpus


def load(path=None) -> dict:
    return corpus.load_needles_spec(path)


def fact_blocks(spec: dict, names: list[str]) -> list[dict]:
    return [spec["facts"][name] for name in names]


def prompt_for(spec: dict, kind: str, document: str) -> tuple[str, list[dict]]:
    """Insert the prompt's facts into ``document`` and return (prompt, questions)."""
    probe = spec["prompts"][kind]
    with_facts = corpus.insert_facts(document, fact_blocks(spec, probe["facts"]))
    return corpus.build_prompt(with_facts, probe["questions"]), probe["questions"]


def normalise(text: str) -> str:
    text = text.lower().replace("\u00a0", " ")
    text = re.sub(r"[^\w\u00c0-\u024f.@-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_line(answer: str, index: int) -> str:
    """Pull ``index: ...`` out of a multi-line answer; fall back to the whole text."""
    pattern = re.compile(rf"^\s*{index}\s*[:.)]\s*(.+)$", re.MULTILINE)
    m = pattern.search(answer)
    if m:
        return m.group(1)
    lines = [ln for ln in answer.splitlines() if ln.strip()]
    return lines[index - 1] if len(lines) >= index else answer


def score(answer: str, questions: list[dict]) -> dict:
    """Score each question; a question passes when all expected strings appear.

    A single question is scored against the whole answer: models often split a
    two-fact answer over two numbered lines ("1: Marta Ozola" / "2: 2026-12-04"),
    which is correct but does not fit the per-question line rule. With several
    questions, the numbered line for that question is the unit, because scanning
    the whole answer would let one line satisfy another question's facts.
    """
    hits, misses = 0, []
    details = []
    whole = normalise(answer)
    for i, q in enumerate(questions, 1):
        line = whole if len(questions) == 1 else normalise(answer_line(answer, i))
        missing = [a for a in q["answers"] if normalise(a) not in line]
        ok = not missing
        hits += int(ok)
        if not ok:
            misses.append({"question": q["q"], "missing": missing, "line": line[:200]})
        details.append({"question": q["q"], "ok": ok})
    return {"questions": len(questions), "hits": hits,
            "score": round(hits / len(questions), 3) if questions else 0.0,
            "misses": misses, "details": details}


def agreement(answer: str, baseline: str) -> bool:
    """Greedy answer identical to the full-precision (f16) answer."""
    return bool(normalise(answer)) and normalise(answer) == normalise(baseline)
