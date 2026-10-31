"""Offline tests for the R4 experiment harness (tools/exp/common.py)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "tools" / "exp"
sys.path.insert(0, str(EXP))

import common as C  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------

def test_extraction_fixture_is_deterministic_and_fictional():
    a = C.extraction_fixture(12)
    b = C.extraction_fixture(12)
    assert a == b
    assert all(set(r["gold"]) == set(C.FIELDS) for r in a)
    joined = " ".join(r["text"] for r in a)
    assert "@example.com" in joined
    assert "555-01" in joined
    assert "gmail" not in joined


def test_qa_fixture_and_edit_facts():
    fx = C.qa_fixture(40)
    assert len(fx["passages"]) == 40 and len(fx["pairs"]) == 40
    for pair in fx["pairs"]:
        assert pair["question"] and pair["answer"]
        assert pair["passage_id"] in range(40)
    changed = [fx["pairs"][1]["id"], fx["pairs"][2]["id"]]
    edited = C.edit_facts(fx, changed)
    old = {p["id"]: p["answer"] for p in fx["pairs"]}
    new = {p["id"]: p["answer"] for p in edited["pairs"]}
    assert all(old[i] != new[i] for i in changed)
    # the passage text actually carries the new value -> retrieval can find it
    for i in changed:
        pair = {p["id"]: p for p in edited["pairs"]}[i]
        doc = edited["passages"][pair["passage_id"]]["text"]
        assert C.norm_value(pair["answer"]) in C.norm_value(doc)
    # the untouched pairs keep their answers
    untouched = [i for i in old if i not in changed]
    assert all(old[i] == new[i] for i in untouched)


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------

def test_micro_f1_perfect_partial_zero():
    gold = [{"a": "x", "b": "y"}]
    assert C.micro_f1(gold, gold, ["a", "b"]) == pytest.approx(100.0)
    half = C.micro_f1([{"a": "x", "b": "z"}], gold, ["a", "b"])
    assert half == pytest.approx(50.0)
    assert C.micro_f1([{}], gold, ["a", "b"]) == 0.0


def test_norm_and_canonical_fields():
    assert C.norm_value(" $12,000. ") == "12000"
    got = C.canonical_fields({"Name": "Dana", "budget": "$5,000",
                              "Next Step": "call", "junk": 1})
    assert got["contact_name"] == "Dana"
    assert got["budget_usd"] == "$5,000"
    assert got["next_step"] == "call"
    assert "junk" not in got


def test_majority_prefers_most_common_then_first():
    assert C.majority(["a", "b", "a"]) == "a"
    assert C.majority(["", "b", "b"]) == "b"
    assert C.majority(["", ""]) == ""


def test_extract_json_handles_fences_and_trailing_text():
    assert C.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert C.extract_json('reasoning... {"a": 1} then {"b": 2}') == {"b": 2}
    assert C.extract_json("no object here") is None
    assert C.extract_json('{"s": "a } brace"}') == {"s": "a } brace"}


def test_exact_match_normalises():
    assert C.exact_match(" $12,000 ", "12000")
    assert not C.exact_match("", "12000")


# ---------------------------------------------------------------------------
# result plumbing
# ---------------------------------------------------------------------------

def test_arm_and_result_schema_is_done_when_ready(tmp_path):
    arms = [C.arm("a", 10.0, 5.0, 3.0), C.arm("b", 11.0, 6.0, 3.0)]
    res = C.base_result("01-x", "h", "m", 60, arms, True)
    res["gpu_minutes"] = 1.0
    assert len(res["arms"]) >= 2 and res["n"] >= 50 and res["gpu_minutes"] > 0
    assert isinstance(res["pass"], bool)
    p = tmp_path / "r.json"
    C.write_result(str(p), res)
    loaded = json.loads(p.read_text())
    assert loaded["arms"][0]["tokens_per_s"] == 5.0


def test_guard_writes_a_result_even_on_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def boom():
        raise RuntimeError("gpu fell over")
    res = C.guard("99-fail", "h", "m", "state/exp/99-fail.json", boom)
    assert res["pass"] is False and res["arms"] == []
    assert "gpu fell over" in res["error"]
    assert (tmp_path / "state" / "exp" / "99-fail.json").is_file()


# ---------------------------------------------------------------------------
# queue files: every job points at a real script inside the repo
# ---------------------------------------------------------------------------

def test_job_files_reference_existing_scripts():
    import yaml
    jobs = sorted((ROOT / "gpu_jobs").glob("*.job.yaml"))
    assert len(jobs) >= 5, jobs
    for jf in jobs:
        job = yaml.safe_load(jf.read_text())
        script = ROOT / job["script"]
        assert script.is_file(), (jf.name, script)
        assert int(job.get("timeout_minutes", 90)) <= 240


def test_card_scripts_import_without_touching_the_gpu():
    for name in ("card1_cot_self_consistency", "card2_rag_vs_qlora",
                 "card3_quant_formats", "card4_speculative", "card5_compiled_prompts"):
        mod = _load(name)
        assert hasattr(mod, "main") and hasattr(mod, "OUT")
        assert mod.OUT.startswith("state/exp/")


def test_fetch_requests_are_within_the_size_cap():
    import yaml
    reqs = sorted((ROOT / "fetch").glob("*.yaml"))
    assert reqs, "no fetch specs"
    for f in reqs:
        req = yaml.safe_load(f.read_text())
        assert req.get("hf") or req.get("gguf") or req.get("ollama"), f.name
        assert float(req["size_gb"]) <= 80, f.name
    draft = yaml.safe_load((ROOT / "fetch" / "qwen3-0.6b-gguf.yaml").read_text())
    assert draft["hf"] == "Qwen/Qwen3-0.6B-GGUF"
    assert float(draft["size_gb"]) <= 1


# ---------------------------------------------------------------------------
# hard task (cards 1/5 rerun): sales-brain calls, adversarial fallback
# ---------------------------------------------------------------------------

def test_hard_fixture_adversarial_fallback_is_deterministic(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "SALES_BRAIN_DIR", tmp_path / "nope")
    a, src = C.hard_extraction_fixture(6)
    b, _ = C.hard_extraction_fixture(6)
    assert src.startswith("adversarial-fallback")
    assert a == b and len(a) == 6
    assert all(set(r["gold"]) == set(C.HARD_FIELDS) for r in a)


def test_hard_fixture_reads_sales_brain_calls(monkeypatch, tmp_path):
    d = tmp_path / "call_0000"
    d.mkdir()
    (d / "call.json").write_text(json.dumps({
        "id": "call_0000", "lang": "lv",
        "turns": [{"speaker": "prospect", "t": 3, "text": "Summa ir 50 000 EUR."}]}))
    (d / "gold.json").write_text(json.dumps({
        "deal": {"amount": 50000, "term_months": 24, "industry": "transports"}}))
    monkeypatch.setattr(C, "SALES_BRAIN_DIR", tmp_path)
    rows, src = C.hard_extraction_fixture(1)
    assert src == "sales-brain-synthetic"
    assert rows[0]["gold"]["amount"] == "50000"
    assert rows[0]["gold"]["term_months"] == "24"
    assert rows[0]["gold"]["industry"] == "transports"
    assert "50 000 EUR" in rows[0]["text"]


def test_canonical_hard_and_hard_norm():
    got = C.hard_fields({"Sum": "50 000 EUR", "term": "24 months",
                         "nozare": "Transports", "junk": 1})
    assert got["amount"] == "50000"
    assert got["term_months"] == "24"
    assert got["industry"] == "transports"
    assert "junk" not in got
    assert C.hard_norm("amount", None) == ""
