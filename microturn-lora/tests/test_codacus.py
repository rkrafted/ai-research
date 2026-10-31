"""Offline tests for the Codacus ingest/extract/assemble pipeline."""

import json
from pathlib import Path

import pytest

from codacus import assemble, extract, ingest
from airesearch import ingest as ai

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "codacus"


def _videos_path() -> Path:
    """`videos.json` is a generated index (gitignored); use the local cache if present."""
    primary = OUT / "videos.json"
    if primary.is_file():
        return primary
    fallback = REPO / "data" / "generated" / "codacus" / "videos.json"
    if fallback.is_file():
        return fallback
    pytest.skip("data/codacus/videos.json is generated and not committed")
    raise AssertionError("unreachable")


def _videos() -> list[dict]:
    return json.loads(_videos_path().read_text(encoding="utf-8"))

VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.000
we run a 35B model on

00:00:02.000 --> 00:00:04.000
we run a 35B model on
one RTX 3060 with &gt;&gt; 12 GB

00:00:04.000 --> 00:00:06.000
one RTX 3060 with &gt;&gt; 12 GB
and a 4 bit cache
"""


def test_vtt_text_dedups_and_drops_timestamps(tmp_path, monkeypatch):
    monkeypatch.setattr(ai, "CAPTIONDIR", tmp_path)
    (tmp_path / "abc.en.vtt").write_text(VTT, encoding="utf-8")
    text = ingest.vtt_text("abc")
    assert "00:00" not in text
    assert "&gt;" not in text and ">>" not in text
    # The rolling repeat appears once, not three times.
    assert text.count("we run a 35B model on") == 1
    assert text.count("one RTX 3060") == 1
    assert "4 bit cache" in text


def test_videos_json_shape():
    videos = _videos()
    assert isinstance(videos, list) and len(videos) >= 30
    for row in videos:
        assert row["id"] and row["title"]
        assert row["upload_date"] and isinstance(row["duration"], int)
    ids = {v["id"] for v in videos}
    assert {"pIN-2oVJpyU", "bcGO7xre46o"} <= ids


def test_prompt_has_schema_and_caps_transcript():
    row = {"id": "x", "title": "t", "upload_date": "20260101", "duration": 60,
           "url": "https://youtube.com/watch?v=x", "channel": "Codacus"}
    prompt = extract.build_prompt(row, "hello world " * 5000)
    assert '"mechanism"' in prompt and '"goals"' in prompt
    assert "speed, context, model-size-on-24GB, hallucination, forgetting/memory, other" in prompt
    assert len(prompt) <= extract.MAX_PROMPT
    assert "/" "home/" not in prompt


@pytest.mark.parametrize("raw", [
    '```json\n{"system": "s", "technique": "t", "mechanism": ["a"], "goals": ["speed"]}\n```',
    'Sure. {"system": "s", "mechanism": ["a"], "goals": []} done',
])
def test_parse_answer_tolerates_fences_and_prose(raw):
    obj = assemble.parse_answer(raw)
    assert obj and obj["mechanism"] == ["a"]


def test_parse_answer_rejects_garbage():
    assert assemble.parse_answer("no json here") is None
    assert assemble.parse_answer('{"broken": ') is None
    assert assemble.parse_answer(None) is None


def test_clean_marks_claims_and_filters_goals():
    out = assemble.clean({
        "system": "Colibri", "technique": "MoE offload",
        "mechanism": [f"step {i}" for i in range(9)],
        "claims": [{"claim": "2x faster", "numbers": "2x"},
                   {"claim": "fits", "numbers": "12 GB"}, "plain"],
        "links": ["https://github.com/a/b", "not-a-url"],
        "hardware": ["RTX 3060"],
        "goals": ["speed", "vibes", "context"],
    })
    assert out["mechanism"] == [f"step {i}" for i in range(6)]
    assert len(out["claims"]) == 3 and all(c["claimed"] for c in out["claims"])
    assert out["links"] == ["https://github.com/a/b"]
    assert out["goals"] == ["speed", "context"]
    assert assemble.clean({})["goals"] == ["other"]


def test_ground_links_drops_invented_urls_and_adds_description():
    out = assemble.ground_links(
        ["https://github.com/invented/repo"],
        transcript="no urls here",
        description="code: https://github.com/real/repo\nmore https://example.com/x.",
        video_url="https://www.youtube.com/watch?v=x",
    )
    assert "https://github.com/invented/repo" not in out
    assert "https://github.com/real/repo" in out
    assert "https://example.com/x" in out
    assert "https://www.youtube.com/watch?v=x" in out


def _fake_batch(tmp_path, n=26):
    videos = _videos()
    (tmp_path / "videos.json").write_text(json.dumps(videos), encoding="utf-8")
    raw = tmp_path / "extracts.jsonl"
    with raw.open("w", encoding="utf-8") as fh:
        for v in videos[:n]:
            answer = {"system": "sys", "technique": "tech",
                      "mechanism": ["load", "offload", "decode"],
                      "claims": [{"claim": "faster", "numbers": "2x"}],
                      "links": [], "hardware": ["RTX 3090"],
                      "goals": ["speed"], "buildable": "reuse the scheduler"}
            fh.write(json.dumps({"id": v["id"], "response": json.dumps(answer)}) + "\n")
    return raw, videos


def test_assemble_writes_one_json_per_video(tmp_path, monkeypatch):
    raw, videos = _fake_batch(tmp_path)
    monkeypatch.setattr(assemble, "OUT", tmp_path)
    monkeypatch.setattr(assemble, "VIDEOS", tmp_path / "videos.json")
    assert assemble.main(["--raw", str(raw)]) == 0
    files = [p for p in sorted(tmp_path.glob("*.json")) if p.name != "videos.json"]
    assert len(files) == 26
    row = json.loads((tmp_path / f"{videos[0]['id']}.json").read_text(encoding="utf-8"))
    assert row["id"] == videos[0]["id"] and row["date"] == videos[0]["upload_date"]
    assert row["claims"][0]["claimed"] is True


def test_assemble_refuses_short_batch(tmp_path, monkeypatch):
    raw, _ = _fake_batch(tmp_path, n=5)
    monkeypatch.setattr(assemble, "OUT", tmp_path)
    monkeypatch.setattr(assemble, "VIDEOS", tmp_path / "videos.json")
    assert assemble.main(["--raw", str(raw)]) == 1
    assert not [p for p in tmp_path.glob("*.json") if p.name != "videos.json"]
