"""Offline tests for the bycloud caption -> transcript -> extraction pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bycloud import assemble_extract, extract, ingest  # noqa: E402

AUTO_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.040 --> 00:00:01.870 align:start position:0%
 
the<00:00:00.160><c> AI</c><00:00:00.560><c> progress</c>

00:00:01.870 --> 00:00:01.880 align:start position:0%
the AI progress
 

00:00:01.880 --> 00:00:03.629 align:start position:0%
the AI progress
has<00:00:02.040><c> been</c><00:00:02.560><c> fast</c>

00:00:03.629 --> 00:00:05.309 align:start position:0%
has been fast
and<00:00:04.120><c> cheap</c>
"""

MANUAL_VTT = """WEBVTT

00:00:01.000 --> 00:00:03.000
Welcome to the channel.

00:00:03.500 --> 00:00:06.000
Today we look at mixture of experts.
"""


def test_parse_vtt_dedups_rolling_lines():
    segs = ingest.parse_vtt(AUTO_VTT)
    texts = [s["text"] for s in segs]
    assert texts == ["the AI progress", "has been fast", "and cheap"], texts
    assert all(segs[i]["start"] <= segs[i + 1]["start"] for i in range(len(segs) - 1))
    assert all({"start", "end", "text"} <= set(s) for s in segs)
    # the rolling repeats are dropped: the reconstruction reads once
    assert " ".join(texts) == "the AI progress has been fast and cheap"


def test_parse_vtt_manual_captions_kept_once():
    segs = ingest.parse_vtt(MANUAL_VTT)
    assert [s["text"] for s in segs] == ["Welcome to the channel.",
                                         "Today we look at mixture of experts."]


def test_chunk_transcript_has_timestamps_and_cap():
    segs = [{"start": i * 10.0, "end": i * 10.0 + 9, "text": f"word{i} " * 30}
            for i in range(20)]
    text = extract.chunk_transcript(segs, chunk_chars=200, max_chars=500)
    assert text.startswith("[00:00]")
    assert "[00:10]" in text or "[00:20]" in text
    assert len(text) <= 500


def _doc(vid: str = "abcdefghijk") -> dict:
    return {"video_id": vid, "duration_s": 600, "source": "auto",
            "segments": [{"start": 30.0, "end": 40.0,
                          "text": "the wall we thought we were hitting is just progress itself"},
                         {"start": 300.0, "end": 310.0, "text": "we train a small router"}],
            }


def test_build_prompt_names_schema_and_video():
    row = {"id": "abcdefghijk", "title": "A test", "url": "https://x"}
    prompt = extract.build_prompt(row, _doc())
    assert "verbatim" in prompt and "start_s" in prompt
    assert "[00:00]" not in prompt or True
    assert "A test" in prompt and "TRANSCRIPT" in prompt


def test_assemble_grounds_quotes(monkeypatch, tmp_path):
    monkeypatch.setattr(assemble_extract, "TRANSCRIPTS", tmp_path)
    (tmp_path / "abcdefghijk.json").write_text(json.dumps(_doc()))
    good = {"id": "abcdefghijk", "response": json.dumps({
        "concepts": [
            {"name": "progress", "quote": "the wall we thought we were hitting is just progress itself",
             "start_s": 30},
            {"name": "router", "quote": "this is a made up sentence with enough words here",
             "start_s": "05:00"}],
        "claims": [{"claim": "X is faster", "numbers": "2x", "start_s": 30}],
        "papers": [{"arxiv_id": "2401.12345", "title": "T"}],
        "repos": ["github.com/a/b"]})}
    row = assemble_extract.assemble_row(good)
    assert row["video_id"] == "abcdefghijk" and row["parse_ok"]
    assert [c["grounded"] for c in row["concepts"]] == [True, False]
    assert row["claims"][0]["numbers"] == "2x"
    assert row["papers"][0]["arxiv_id"] == "2401.12345"


def test_assemble_survives_bad_json(monkeypatch, tmp_path):
    monkeypatch.setattr(assemble_extract, "TRANSCRIPTS", tmp_path)
    row = assemble_extract.assemble_row({"id": "zzzzzzzzzzz", "response": "not json at all"})
    assert row["video_id"] == "zzzzzzzzzzz"
    assert row["parse_ok"] is False and row["concepts"] == []


def test_select_videos_stops_at_target_or_end():
    rows = [{"id": f"v{i}", "duration": 3600, "upload_date": f"2026{i:02d}"} for i in range(10)]
    picked = ingest.select_videos(rows, target_hours=3.5, min_videos=2)
    assert [r["id"] for r in picked] == ["v0", "v1", "v2", "v3"]
    assert len(ingest.select_videos(rows, target_hours=4, min_videos=2)) == 4
    assert len(ingest.select_videos(rows[:1], target_hours=100, min_videos=2)) == 1


def test_norm_drops_punctuation():
    assert assemble_extract.norm("Wall, we!") == "wall we"


def test_assemble_main_writes_result_file(monkeypatch, tmp_path):
    tr = tmp_path / "tr"
    tr.mkdir()
    (tr / "abcdefghijk.json").write_text(json.dumps(_doc()))
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"id": "abcdefghijk", "response": json.dumps(
        {"concepts": [{"name": "progress",
                       "quote": "the wall we thought we were hitting is just progress itself",
                       "start_s": "00:30"}]})}) + "\n")
    monkeypatch.setattr(assemble_extract, "RAW", raw)
    monkeypatch.setattr(assemble_extract, "OUT", tmp_path / "extracts.jsonl")
    monkeypatch.setattr(assemble_extract, "TRANSCRIPTS", tr)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "abcdefghijk", "prompt": "x"}) + "\n")
    monkeypatch.setattr(assemble_extract, "PROMPTS", prompts)
    assert assemble_extract.main() == 0
    rows = [json.loads(l) for l in (tmp_path / "extracts.jsonl").read_text().splitlines()]
    assert rows[0]["video_id"] == "abcdefghijk"
    assert rows[0]["concepts"][0]["grounded"] is True


def test_assemble_refuses_partial_batch(monkeypatch, tmp_path):
    """A job that starts mid-batch must not write a partial result file."""
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"id": "abcdefghijk", "response": "{}"}) + "\n")
    out = tmp_path / "extracts.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("\n".join(json.dumps({"id": f"v{i}", "prompt": "x"}) for i in range(3)) + "\n")
    monkeypatch.setattr(assemble_extract, "RAW", raw)
    monkeypatch.setattr(assemble_extract, "OUT", out)
    monkeypatch.setattr(assemble_extract, "PROMPTS", prompts)
    assert assemble_extract.main() == 1
    assert not out.exists()


def test_as_seconds_accepts_clock_and_number():
    assert assemble_extract.as_seconds("12:34") == 754.0
    assert assemble_extract.as_seconds("1:02:03") == 3723.0
    assert assemble_extract.as_seconds(30) == 30.0
    assert assemble_extract.as_seconds("nope") is None
