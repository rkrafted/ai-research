"""Registry build test on a synthetic cache (offline, no real channels)."""

import json
import os

from airesearch import registry
from airesearch.fetch import slug

RICH = (
    "LoRA vs QLoRA: a 7B model on a single 24 GB GPU. Ablation: 3.5x less VRAM, "
    "+8.2% over baseline. Paper https://arxiv.org/abs/2305.14314 code on GitHub."
)


def video(i, rich=True):
    return {
        "id": f"v{i}",
        "title": "LoRA QLoRA distillation" if rich else "hello world",
        "description": RICH if rich else "a short update",
        "captions_auto": ["en"],
        "captions_manual": [],
        "duration": 600,
        "upload_date": "20260101",
        "view_count": 1000,
    }


def write_cache(cache_dir, url, n, rich=True):
    os.makedirs(cache_dir, exist_ok=True)
    payload = {"url": url, "videos": [video(i, rich) for i in range(n)]}
    with open(os.path.join(cache_dir, slug(url) + ".json"), "w") as fh:
        json.dump(payload, fh)


def test_build_registry(tmp_path, monkeypatch):
    cache = str(tmp_path / "channels")
    sources = [
        ("Seed A", "https://www.youtube.com/@SeedA", "seed"),
        ("Seed B", "https://www.youtube.com/@SeedB", "seed"),
        ("Good Candidate", "https://www.youtube.com/@Good", "candidate"),
        ("Weak Candidate", "https://www.youtube.com/@Weak", "candidate"),
        ("Tiny Candidate", "https://www.youtube.com/@Tiny", "candidate"),
    ]
    for name, url, prov in sources:
        if name == "Weak Candidate":
            write_cache(cache, url, 20, rich=False)
        elif name == "Tiny Candidate":
            write_cache(cache, url, 5, rich=True)
        else:
            write_cache(cache, url, 20, rich=True)
    monkeypatch.setattr(registry, "all_sources", lambda: sources)

    result = registry.build(cache)
    by_name = {r["name"]: r for r in result["channels"]}

    assert len(result["channels"]) == 4
    assert result["seed_count"] == 2
    assert by_name["Good Candidate"]["kept"] is True
    assert by_name["Weak Candidate"]["kept"] is False
    assert any(s["name"] == "Tiny Candidate" for s in result["skipped"])

    row = registry.channel_json(by_name["Good Candidate"])
    assert set(row) >= {"name", "url", "videos_sampled", "score", "kept", "reason", "seed"}
    assert row["videos_sampled"] == 20


def test_write_outputs(tmp_path, monkeypatch):
    cache = str(tmp_path / "channels")
    sources = [("Seed A", "https://www.youtube.com/@SeedA", "seed"),
               ("Good", "https://www.youtube.com/@Good", "candidate")]
    write_cache(cache, sources[0][1], 20)
    write_cache(cache, sources[1][1], 20)
    monkeypatch.setattr(registry, "all_sources", lambda: sources)

    result = registry.build(cache)
    jpath = str(tmp_path / "channels.json")
    mpath = str(tmp_path / "channels.md")
    registry.write_json(result, jpath)
    registry.write_md(result, mpath)

    data = json.load(open(jpath))
    assert all({"name", "url", "videos_sampled", "score", "kept", "reason", "seed"} <= set(d) for d in data)
    md = open(mpath).read()
    assert "youtube.com/" in md
