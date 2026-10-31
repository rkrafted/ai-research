"""Offline tests for the R2 ingest, prompt building and SQLite KB (no network)."""

from __future__ import annotations

import json
from datetime import date

from airesearch import kb, prompts
from airesearch.ingest import window_start

VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.000
LoRA fine tuning
LoRA fine tuning

00:00:02.000 --> 00:00:05.000
on a single 24 GB GPU

00:00:05.000 --> 00:00:07.000
see arxiv 2305.14314
"""


def test_window_start_is_24_months_back():
    assert window_start(24, date(2026, 10, 2)) == date(2024, 10, 1)
    assert window_start(12, date(2026, 1, 15)) == date(2025, 1, 1)


def test_vtt_segments_strips_tags_and_repeats():
    segs = prompts.vtt_segments(VTT)
    assert len(segs) == 1                      # short file -> one timestamped chunk
    sec, text = segs[0]
    assert sec == 0
    assert "LoRA fine tuning on a single 24 GB GPU" in text
    assert "2305.14314" in text
    assert text.count("LoRA fine tuning") == 1  # the repeated line is dropped


def test_vtt_segments_merges_scrolling_overlap():
    vtt = (
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n"
        "Much of the progress in large language\n\n"
        "00:00:02.000 --> 00:00:04.000\n"
        "Much of the progress in large language models\n\n"
        "00:00:04.000 --> 00:00:06.000\n"
        "models comes from loops\n"
    )
    text = " ".join(t for _, t in prompts.vtt_segments(vtt))
    assert text.count("Much of the progress") == 1
    assert text.endswith("comes from loops")


def test_build_prompt_has_metadata_and_schema():
    row = {
        "id": "abc", "channel": "Test Channel", "title": "LoRA on 24 GB",
        "upload_date": "20260101", "duration": 600, "view_count": 12,
        "url": "https://www.youtube.com/watch?v=abc", "description": "A test.",
    }
    text = prompts.build_prompt(row, transcript="[00:10] hello")
    assert text.startswith("You extract structured research notes")
    assert "Test Channel" in text and "LoRA on 24 GB" in text
    assert "[00:10] hello" in text
    assert len(text) <= prompts.MAX_PROMPT


def test_write_prompts_jsonl(tmp_path):
    row = {"id": "v1", "channel": "c", "title": "t", "upload_date": "20260101",
           "duration": 1, "view_count": 1, "url": "u", "description": "d"}
    path = tmp_path / "p.jsonl"
    assert prompts.write_prompts([row], path) == 1
    rec = json.loads(path.read_text().splitlines()[0])
    assert rec["id"] == "v1" and rec["prompt"]


def test_arxiv_normalisation_and_validation():
    assert kb.normalize_arxiv("arxiv.org/abs/2305.14314v2") == "2305.14314"
    assert kb.normalize_arxiv("2401.12345") == "2401.12345"
    assert kb.normalize_arxiv("9999.99999") is None      # impossible year/month
    assert kb.normalize_arxiv("not a paper") is None
    assert kb.arxiv_ids("See 2401.12345 and https://arxiv.org/abs/1706.03762") == {
        "2401.12345", "1706.03762"}


def _write_extracts(path):
    rows = [
        {"id": "a", "response": json.dumps({
            "methods": ["QLoRA", "quantization"],
            "claims": [{"claim": "fits in 8 GB", "numbers": "8 GB", "timestamp": "[01:20]"}],
            "papers": [{"arxiv_id": "2305.14314", "title": "QLoRA"}],
            "repos": ["https://github.com/artidoro/qlora"],
            "hardware": ["RTX 3090"], "audience": "builders"})},
        {"id": "b", "response": "```json\n" + json.dumps({
            "methods": [], "claims": [{"claim": "x", "numbers": "", "timestamp": None}],
            "papers": [{"arxiv_id": "2401.12345"}], "repos": [],
            "hardware": [], "audience": ""}) + "\n```"},
        {"id": "c", "response": "the model rambled with no json"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _write_corpus(path):
    rows = [{"id": v, "channel": "chan", "title": f"title {v}", "upload_date": "20260101",
             "duration": 60, "view_count": 5, "url": f"https://x/{v}"}
            for v in ("a", "b", "c")]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_kb_build_from_batch_output(tmp_path):
    extracts = tmp_path / "extracts.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    db = tmp_path / "kb.sqlite"
    _write_extracts(extracts)
    _write_corpus(corpus)

    kb.build(extracts, db, corpus)
    s = kb.stats(db)
    assert s["videos"] == 3
    assert s["claim_videos"] == 2          # c had no parseable JSON
    assert s["claims"] == 2
    assert s["arxiv_ids"] == 2             # 2305.14314 + 2401.12345
    assert s["methods"] == 2
    assert s["links"] >= 1


def test_kb_keeps_timestamp_and_video_id(tmp_path):
    extracts = tmp_path / "extracts.jsonl"
    _write_extracts(extracts)
    db = tmp_path / "kb.sqlite"
    kb.build(extracts, db, tmp_path / "missing-corpus.jsonl")
    import sqlite3
    con = sqlite3.connect(db)
    vid, ts = con.execute(
        "SELECT video_id, timestamp FROM claims WHERE timestamp IS NOT NULL").fetchone()
    assert vid == "a" and ts == "[01:20]"
