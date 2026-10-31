"""Offline tests for caveman-terse.

Stdlib only: no transformers, no network, no transcript access. The committed
artifacts (docs, prompt, savings summary, rewrites) are what the runner checks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "caveman"))

import extract_turns as E  # noqa: E402
import measure as M  # noqa: E402

PROMPT = REPO / "prompts" / "caveman-system.txt"
DOC = REPO / "docs" / "caveman.md"


def _artifact(primary: Path, fallback: Path) -> Path:
    """Generated artifacts are gitignored; the local cache keeps the tests runnable."""
    if primary.is_file():
        return primary
    if fallback.is_file():
        return fallback
    return primary


SAVINGS = _artifact(REPO / "state" / "caveman-savings.json",
                    REPO / "data" / "generated" / "state" / "caveman-savings.json")
REWRITES = _artifact(REPO / "tools" / "caveman" / "rewrites.json",
                     REPO / "data" / "generated" / "tools" / "caveman" / "rewrites.json")


def _require(path: Path) -> Path:
    if not path.is_file():
        pytest.skip(f"{path.name} is a generated artifact, not committed")
    return path


def test_savings_bounds_and_consistency():
    r = json.loads(_require(SAVINGS).read_text(encoding="utf-8"))
    assert r["n_turns"] >= 30
    assert 0 < r["saving_pct"] < 100
    assert r["tokens_before"] > r["tokens_after"] > 0
    assert len(r["turns"]) == r["n_turns"]
    before = sum(t["before"] for t in r["turns"])
    after = sum(t["after"] for t in r["turns"])
    assert (before, after) == (r["tokens_before"], r["tokens_after"])
    assert abs(100.0 * (before - after) / before - r["saving_pct"]) < 0.01


def test_savings_has_no_absolute_home_paths():
    text = _require(SAVINGS).read_text(encoding="utf-8")
    assert ("/" + "home" + "/") not in text
    assert "~/" in text


def test_prompt_protects_payload():
    text = PROMPT.read_text(encoding="utf-8")
    assert text.strip()
    low = text.lower()
    for needle in ("tool-call arguments", "code", "verbatim", "prose and reasoning"):
        assert needle in low, needle


def test_doc_covers_required_sections():
    text = DOC.read_text(encoding="utf-8")
    assert text.strip()
    low = text.lower()
    for needle in ("licence", "risk", "saving", "skill"):
        assert needle in low, needle
    assert "JuliusBrussee/caveman" in text
    assert ("/" + "home" + "/") not in text


def test_rewrites_shape():
    art = json.loads(_require(REWRITES).read_text(encoding="utf-8"))
    savings = json.loads(_require(SAVINGS).read_text(encoding="utf-8"))
    assert art["n"] == len(art["rewrites"]) == savings["n_turns"] >= 30
    ids = [r["id"] for r in art["rewrites"]]
    assert len(ids) == len(set(ids))
    assert {t["id"] for t in savings["turns"]} == set(ids)
    for r in art["rewrites"]:
        assert r["prose_after"] and all(isinstance(s, str) for s in r["prose_after"])


def test_segments_keep_fences_byte_exact():
    text = "before\n```py\nx = 1  # drop nothing\ny = 2\n```\nafter prose"
    segs = E.segments(text)
    assert [s["kind"] for s in segs] == ["text", "code", "text"]
    assert segs[1]["text"] == "```py\nx = 1  # drop nothing\ny = 2\n```"


def test_assemble_rewrites_prose_only():
    turn = {"id": "t", "segments": [
        {"kind": "text", "text": "The quick brown fox."},
        {"kind": "code", "text": "```\nrun --flag\n```"},
        {"kind": "text", "text": " It was very fast."},
    ]}
    out = M.assemble(turn, ["Fox fast.", " Was fast."])
    assert out == "Fox fast.```\nrun --flag\n``` Was fast."


def test_assemble_rejects_segment_mismatch():
    turn = {"id": "t", "segments": [{"kind": "text", "text": "a"}, {"kind": "text", "text": "b"}]}
    try:
        M.assemble(turn, ["only one"])
    except ValueError:
        return
    raise AssertionError("expected ValueError for segment/rewrite mismatch")


def test_tokenizer_label():
    p = "/x/models--Qwen--Qwen3-1.7B/snapshots/abc"
    assert M.tokenizer_label(p) == "Qwen/Qwen3-1.7B"
    assert M.tildify(str(Path.home() / "x")).startswith("~")
