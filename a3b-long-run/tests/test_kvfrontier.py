"""Offline checks for the KV-frontier eval harness. No GPU, no network."""
from __future__ import annotations

import json
import re
from pathlib import Path

from kvfrontier import corpus, needles, report

ROOT = Path(__file__).resolve().parent.parent


def test_kv_bytes_per_token_matches_a3b_metadata():
    # 10 full-attention layers (40 blocks, interval 4), 2 KV heads, head_dim 256
    assert report.kv_bytes_per_token("f16", "f16") == 20480
    assert report.kv_bytes_per_token("q8_0", "q8_0") == 10880
    assert report.kv_bytes_per_token("q5_1", "q5_1") == 7680
    assert report.kv_bytes_per_token("q4_0", "q4_0") == 5760
    assert report.kv_bytes_per_token("q8_0", "q4_0") < report.kv_bytes_per_token("q8_0", "q8_0")
    assert report.kv_bytes_per_token("f16", "f16") == 10 * 2 * 256 * 4


def test_concat_prefers_whole_docs_and_cuts_only_when_none_fit():
    docs = [corpus.Doc("big", "x" * 100), corpus.Doc("ok", "y" * 40), corpus.Doc("tail", "z" * 30)]
    text, ids = corpus.concat_within(docs, 60)
    assert ids == ["ok"] and len(text) == 40          # big skipped, ok kept whole
    text, ids = corpus.concat_within(docs, 30)
    assert ids == ["tail"] and len(text) == 30         # a later whole doc beats cutting
    text, ids = corpus.concat_within([docs[0]], 30)
    assert ids == ["big"] and len(text) == 30          # nothing fits -> cut the first


def test_insert_facts_lands_at_both_ends():
    base = "a" * 1000
    facts = [{"text": "EARLY", "at": 0.08}, {"text": "LATE", "at": 0.9}]
    out = corpus.insert_facts(base, facts)
    assert out.index("EARLY") < out.index("LATE")
    assert out.index("EARLY") < 150
    assert out.index("LATE") > 850


def test_prompt_numbers_every_question():
    qs = [{"q": "one?", "answers": ["a"]}, {"q": "two?", "answers": ["b"]}]
    prompt = corpus.build_prompt("DOC", qs)
    assert "1. one?" in prompt and "2. two?" in prompt
    assert "<document>\nDOC\n</document>" in prompt


def test_score_requires_every_expected_string():
    qs = [{"q": "invoice?", "answers": ["KV-7741", "Dana Rieksta"]},
          {"q": "code?", "answers": ["ZIRGA-19", "2026-11-30"]}]
    good = "1: KV-7741, approved by Dana Rieksta\n2: ZIRGA-19 until 2026-11-30"
    assert needles.score(good, qs)["score"] == 1.0
    partial = "1: KV-7741\n2: ZIRGA-19 until 2026-11-30"
    result = needles.score(partial, qs)
    assert result["score"] == 0.5 and result["misses"][0]["missing"] == ["Dana Rieksta"]


def test_single_multihop_question_scores_a_two_line_answer():
    # the 2026-10-18 GPU run: the model answered both facts, but on numbered
    # lines, and the per-question line rule scored it 0.0.
    qs = [{"q": "audit lead and deadline?", "answers": ["Marta Ozola", "2026-12-04"]}]
    answer = "1: Marta Ozola\n2: 2026-12-04"
    result = needles.score(answer, qs)
    assert result["score"] == 1.0, result
    assert needles.score("1: Marta Ozola", qs)["score"] == 0.0


def test_probe_rejects_a_server_that_dies_on_first_decode(monkeypatch, tmp_path):
    # /health passes after the KV buffer loads; the compute buffer is allocated
    # on the first prompt. The 2026-10-18 job called such a config "fits".
    from kvfrontier import measure

    def boom(*a, **k):
        raise RuntimeError("oom")

    srv = measure.Server("t", "q8_0", "q8_0", 81920)
    srv.log_path = tmp_path / "t.log"
    srv.log_path.write_text("failed to allocate compute pp buffers\n")
    monkeypatch.setattr(measure, "http_json", boom)
    assert srv.probe() is False
    srv.log_path.write_text("model loaded\n")
    monkeypatch.setattr(measure, "http_json", lambda *a, **k: {"choices": []})
    assert srv.probe() is True


def test_failed_cells_are_retried(tmp_path):
    from kvfrontier import measure
    path = tmp_path / "results.jsonl"
    path.write_text(
        '{"config":"q8_0","context":"mix-32k","kind":"needle","error":"oom"}\n'
        '{"config":"f16","context":"mix-32k","kind":"needle","score":1.0}\n')
    done = measure.load_results(path)
    assert ("f16", "mix-32k", "needle") in done
    assert ("q8_0", "mix-32k", "needle") not in done


def test_answer_line_falls_back_to_position():
    assert needles.answer_line("1: alpha\n2: beta", 2) == "beta"
    assert needles.answer_line("alpha\nbeta", 2) == "beta"
    assert "alpha" in needles.answer_line("alpha", 1)


def test_needle_fixtures_are_fictional_and_safe():
    spec = json.loads((ROOT / "eval" / "needles.json").read_text(encoding="utf-8"))
    text = json.dumps(spec)
    emails = set(re.findall(r"[\w.+-]+@[\w.-]+", text))
    assert emails and all(e.endswith("@example.com") for e in emails), emails
    phones = set(re.findall(r"\b555-\d{4}\b", text))
    assert phones and all(p.startswith("555-01") for p in phones), phones
    assert "/" + "home/" not in text


def test_committed_config_has_no_home_paths_or_raw_data():
    home = "/" + "home/"
    for rel in ("eval/contexts.json", "eval/needles.json", "gpu_jobs/kv-measure.job.yaml",
                "state/a3b_server.args.kv-proposed"):
        path = ROOT / rel
        if path.is_file():
            assert home not in path.read_text(encoding="utf-8"), rel


def test_context_spec_covers_real_sources_and_lengths():
    spec = corpus.load_contexts_spec()
    assert set(spec["sources"]) == {"calls", "talks", "repos"}
    assert [c["tokens"] for c in spec["contexts"]] == [32768, 65536, 131072]
    for src in spec["sources"].values():
        assert src.get("path") or src.get("glob") or src.get("work_dir")


def test_missing_sources_build_empty_without_crashing(tmp_path):
    spec = {
        "chars_per_token": 2.6,
        "sources": {"gone": {"kind": "realspeech", "path": str(tmp_path / "nope.jsonl")}},
        "contexts": [{"id": "x", "tokens": 100, "mix": {"gone": 1.0}}],
    }
    built = corpus.build_all(spec)
    assert built["x"]["text"] == "" and built["x"]["chars"] == 0


def test_measure_configs_cover_the_baselines():
    from kvfrontier.measure import CONFIGS
    names = {c[0] for c in CONFIGS}
    assert {"f16", "q8_0", "q5_1", "q4_0"} <= names
    assert all(c[1] in report.BLOCK_BYTES and c[2] in report.BLOCK_BYTES for c in CONFIGS)
