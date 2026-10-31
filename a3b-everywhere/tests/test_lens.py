"""Offline tests for the R3 lens ranker, taxonomy and digest (no network)."""

from __future__ import annotations

import sqlite3

from airesearch import kb, lens
from airesearch.taxonomy import FAMILY, PROFILES


def _db(rows):
    """In-memory KB from rows: (video, channel, title, method, ts)."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(kb.SCHEMA)
    for i, (vid, ch, title, method, ts) in enumerate(rows):
        conn.execute(
            "INSERT OR IGNORE INTO videos (id,channel,title,published,url) "
            "VALUES (?,?,?,'20260101',?)",
            (vid, ch, title, f"https://www.youtube.com/watch?v={vid}"),
        )
        conn.execute(
            "INSERT INTO methods (video_id, method) VALUES (?,?)", (vid, method)
        )
        if ts is not None:
            conn.execute(
                "INSERT INTO claims (video_id, claim, numbers, timestamp) "
                "VALUES (?,?,?,?)",
                (vid, f"{method} improves things", "2x", ts),
            )
        conn.execute(
            "INSERT INTO papers (video_id, arxiv_id, title) VALUES (?,?,?)",
            (vid, "2501.00001", f"{method} paper"),
        )
        conn.execute(
            "INSERT INTO links (video_id, url, kind) VALUES (?,?,?)",
            (vid, f"https://github.com/example/{i}", "github"),
        )
    conn.commit()
    return conn


# three videos per family so the >=3-video eligibility bar is met
ROWS = [
    ("v1", "Chan A", "Quantization of 7B", "Quantization", "00:30"),
    ("v2", "Chan B", "4-bit quantization", "4-bit quantization", None),
    ("v3", "Chan C", "GPTQ", "GPTQ", "01:00"),
    ("v4", "Chan A", "QLoRA on one GPU", "QLoRA", None),
    ("v5", "Chan B", "QLoRA fine-tuning", "QLoRA", "00:10"),
    ("v6", "Chan C", "LoRA adapters", "LoRA", None),
    ("v7", "Chan A", "Retrieval augmented generation", "RAG", None),
    ("v8", "Chan B", "RAG pipelines", "RAG", None),
    ("v9", "Chan C", "Retrieval augmented generation", "RAG", None),
]


def test_taxonomy_profiles_are_well_formed():
    for key, entry in PROFILES.items():
        assert len(entry) == 8, key
        param, flop, ttc, data, cost, theme, why, areas = entry
        for axis in (param, flop, ttc, data):
            assert 0 <= axis <= 100, key
        assert cost >= 0.25, key
        assert theme and why, key
        assert areas, key
    for surface, fam in FAMILY.items():
        assert fam in PROFILES, f"{surface} -> {fam} has no profile"


def test_canonical_folds_surface_forms():
    assert lens.canonical("4-bit quantization") == "4-bit quantization"
    assert lens.canonical("GPTQ") == "gptq"
    assert lens.canonical("Mixture of Experts (MoE)") == "mixture of experts"
    assert lens.canonical("Retrieval-Augmented Generation") == "rag"
    # umbrella terms are canonicalised, then dropped by the stop list
    assert lens.canonical("LLMs") in lens.STOP


def test_families_merge_into_one_cluster():
    conn = _db(ROWS)
    clusters = {c.key: c for c in lens.load_clusters(conn)}
    # Quantization, 4-bit quantization and GPTQ are one family
    assert "quantization" in clusters
    assert "4 bit quantization" not in clusters
    assert clusters["quantization"].videos == {"v1", "v2", "v3"}
    assert {"qlora", "lora", "rag"} <= set(clusters)


def test_score_components_are_bounded_and_deterministic():
    conn = _db(ROWS)
    rows = lens.rank(conn)
    assert rows, "expected at least one eligible cluster"
    by_key = {r["key"]: r for r in rows}
    q = by_key["quantization"]
    c = q["components"]
    assert 0 <= c["capability_per_parameter"] <= 100
    assert 0 <= c["capability_per_flop_vram"] <= 100
    assert 0 <= c["evidence_quality"] <= 100
    assert c["cost_to_try"] > 0
    assert q["priority"] > 0
    # PROFILES give quantization a 95/98 param/flop profile
    assert c["capability_per_parameter"] == 95
    assert c["capability_per_flop_vram"] == 98
    # deterministic: same KB -> identical payload
    assert lens.rank(_db(ROWS)) == rows


def test_eligibility_needs_three_videos():
    one = _db(ROWS[:1])
    assert lens.rank(one) == [] or all(
        r["videos"] >= 3 for r in lens.rank(one)
    )


def test_render_digest_has_items_and_arxiv_links():
    conn = _db(ROWS)
    rows = lens.rank(conn)
    clusters = {c.key: c for c in lens.load_clusters(conn)}
    md = lens.render_digest(rows, clusters, top=len(rows))
    assert md.count("- **") == len(rows)
    assert "arxiv.org/abs/" in md
    assert "## Component table" in md
    assert "youtube.com/watch" in md
