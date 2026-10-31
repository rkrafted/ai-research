"""Offline tests for the R5 upgrade harness.

Nothing here touches `state/upgrades/` -- those files may only be written by a
runner GPU job -- so every write test uses a tmp dir.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "upgrades"))

import upg_common as C  # noqa: E402
import shortlist  # noqa: E402
from candidates import CANDIDATES  # noqa: E402


def test_change_pct_directions():
    assert C.change_pct({"t": 10.0}, {"t": 12.5}, "t") == 25.0
    # lower is better -> a drop is a positive improvement
    assert C.change_pct({"v": 100.0}, {"v": 40.0}, "v", higher_is_better=False) == 60.0
    assert C.change_pct({"v": 0.0}, {"v": 5.0}, "v") == 0.0
    assert C.change_pct({"a": 1.0}, {"b": 2.0}, "v") is None


def test_write_upgrade_schema(tmp_path):
    p = C.write_upgrade("demo", "area", "workload", {"tokens_per_s": 10.0},
                        {"tokens_per_s": 15.0}, True, "flags",
                        change_key="tokens_per_s", state_dir=tmp_path)
    r = json.loads(p.read_text(encoding="utf-8"))
    assert set(r) >= {"name", "area", "workload", "before", "after", "change_pct",
                      "adopt", "how"}
    assert r["change_pct"] == 50.0 and r["adopt"] is True
    assert r["before"] and r["after"] and isinstance(r["adopt"], bool)


def test_run_trial_writes_even_when_fn_fails(tmp_path):
    def boom():
        raise RuntimeError("no GPU")
    out = C.run_trial("broken", "area", "workload", "how", boom,
                      change_key="tokens_per_s", state_dir=tmp_path)
    r = json.loads((tmp_path / "broken.json").read_text(encoding="utf-8"))
    assert out["adopt"] is False
    assert r["before"] and r["after"]           # done_when requires non-empty
    assert "error" in r["after"]


def test_fallback_ingest_prompts_are_deterministic():
    a = C._fallback_ingest_prompts(5)
    b = C._fallback_ingest_prompts(5)
    assert a == b and len(a) == 5 and all("JSON:" in p for p in a)


def _mini_kb() -> sqlite3.Cursor:
    con = sqlite3.connect(":memory:")
    con.executescript(
        """create table videos(id text primary key, title text, url text, views int);
           create table claims(id int, video_id text, claim text, timestamp text);
           create table papers(id int, video_id text, arxiv_id text, title text);""")
    con.execute("insert into videos values('v1','Flash Attention from scratch',"
                "'https://www.youtube.com/watch?v=abc', 10)")
    con.execute("insert into claims values(1,'v1','Flash attention speeds up generation',"
                "'[01:23]')")
    con.execute("insert into papers values(1,'v1','2205.14135','Flash Attention')")
    return con.cursor()


def test_shortlist_renders_one_line_per_candidate():
    text = shortlist.render(_mini_kb())
    bullets = [l for l in text.splitlines() if l.startswith("- ")]
    assert len(bullets) >= 12
    assert len([l for l in bullets if l.startswith("- **")]) == len(CANDIDATES)
    assert "Flash Attention from scratch" in text
    assert "[01:23]" in text and "arXiv 2205.14135" in text
    assert ("/" "home/") not in text           # no absolute host paths in the doc


def test_committed_shortlist_is_current():
    con = sqlite3.connect(shortlist.KB)
    try:
        expected = shortlist.render(con.cursor())
    finally:
        con.close()
    assert shortlist.OUT.read_text(encoding="utf-8") == expected


def test_report_renders_job_rows(tmp_path):
    import report
    (tmp_path / "kv.json").write_text(json.dumps({
        "name": "kv-cache-quant", "area": "fit-bigger", "workload": "27B ingest",
        "before": {"kv_cache_mb": 100.0}, "after": {"kv_cache_mb": 50.0},
        "change_pct": 50.0, "adopt": True, "how": "q8_0"}), encoding="utf-8")
    rows = report.load(tmp_path)
    assert len(rows) == 1
    text = report.render(rows)
    assert "kv-cache-quant" in text and "true" in text and "| 50.0 |" in text
    assert report.load(tmp_path / "missing").__class__ is list


def test_pack_keeps_only_supervised_chunks():
    import trials
    examples = [{"ids": [1, 2, 3], "labels": [-100, -100, 4]},
                {"ids": [5, 6], "labels": [7, 8]}]
    ids, labels = trials._pack(examples, max_len=4)
    assert len(ids) == len(labels) == 2
    assert all(any(l != -100 for l in lab) for lab in labels)
    assert [len(i) for i in ids] == [4, 3]


def test_parse_vram_mb_sums_server_buffers(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "llama_kv_cache_init: CUDA0 KV buffer size = 1024.00 MiB\n"
        "llama_kv_cache_init: CUDA1 KV buffer size = 512.00 MiB\n"
        "llama_context: CUDA0 compute buffer size = 200.00 MiB\n", encoding="utf-8")
    m = C.parse_vram_mb(log)
    assert m["kv_cache_mb"] == 1536.0
    assert m["compute_buffer_mb"] == 200.0


def test_llama_flag_selection_prefers_long_names(monkeypatch):
    monkeypatch.setattr(C, "_HELP",
                        "-fa, --flash-attn [on|off|auto]\n"
                        "-ctk, --cache-type-k TYPE\n-ctv, --cache-type-v TYPE\n"
                        "-np, --parallel N\n")
    assert C.flash_flags() == ["--flash-attn", "on"]
    assert C.context_flags() == ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    assert C.parallel_flags(2) == ["--parallel", "2"]
    monkeypatch.setattr(C, "_HELP", "-fa\n-ctk DIR\n-ctv DIR\n-np N\n")
    assert C.flash_flags() == ["-fa"]
    assert C.context_flags() == ["-ctk", "q8_0", "-ctv", "q8_0"]
    assert C.parallel_flags(2) == ["-np", "2"]
