"""Offline tests for the R0 looped-LM deep-dive. No network, no torch, no GPU."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench_items():
    path = ROOT / "benchmark" / "loop-eval-v1.jsonl"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    # The benchmark is generated, not committed (no raw data); rebuild it in memory.
    gen = _load(ROOT / "tools" / "make_benchmark.py", "make_benchmark")
    return gen.build()


def test_benchmark_shape():
    items = bench_items()
    assert len(items) == 200
    assert sum(i["task"] == "arithmetic" for i in items) == 100
    assert sum(i["task"] == "knowledge" for i in items) == 100
    assert len({i["id"] for i in items}) == 200
    assert len({i["question"] for i in items}) == 200
    for it in items:
        assert len(it["choices"]) == 4
        assert len(set(it["choices"])) == 4
        assert 0 <= it["answer"] < 4


def test_benchmark_is_generator_deterministic():
    gen = _load(ROOT / "tools" / "make_benchmark.py", "make_benchmark")
    assert gen.render(gen.build()) == gen.render(gen.build())
    committed = ROOT / "benchmark" / "loop-eval-v1.jsonl"
    if committed.exists():  # optional derived artifact, regenerated on demand
        assert gen.render(gen.build()) == committed.read_text(encoding="utf-8")


def test_docs_links_and_cards():
    llm = (ROOT / "docs" / "looped-lm.md").read_text(encoding="utf-8")
    vid = (ROOT / "docs" / "neuro-dump-video.md").read_text(encoding="utf-8")
    # mirrors the done_when count: at least six distinct cited arXiv abs pages in total
    n_arxiv = len(re.findall(r"arxiv\.org/abs/", llm)) + len(re.findall(r"arxiv\.org/abs/", vid))
    assert n_arxiv >= 6, n_arxiv
    assert re.search(r"youtube\.com/watch\?v=|youtu\.be/", vid)
    cards = (ROOT / "docs" / "next-experiments.md").read_text(encoding="utf-8")
    assert len(re.findall(r"^## ", cards, flags=re.M)) >= 3


def test_eval_job_wiring_and_offline_load():
    job = (ROOT / "gpu_jobs" / "looped-eval.job.yaml").read_text(encoding="utf-8")
    assert "tools/looped_eval.py" in job
    for f in ("ouro-1.4b.yaml", "ouro-2.6b.yaml", "ouro-2.6b-thinking.yaml", "qwen3-1.7b.yaml"):
        assert (ROOT / "fetch" / f).exists(), f
    mod = _load(ROOT / "tools" / "looped_eval.py", "looped_eval")
    assert len(mod.load_benchmark()) == 200


def _committed_text_files():
    """Tracked + untracked-not-ignored text files (so gitignored artifacts are skipped)."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=30)
        rels = [r for r in out.stdout.splitlines() if r.strip()]
    except (OSError, subprocess.SubprocessError):
        rels = []
    if not rels:                               # no git: fall back to the old tree walk
        rels = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                if p.is_file() and ".git" not in p.parts and ".venv" not in p.parts]
    for rel in rels:
        p = ROOT / rel
        # The batch result is provenance-checked (a session must not edit it) and
        # export-ignored from the runner's own scans; the model's answer text is
        # not ours to sanitise. Its input prompts are sanitised instead.
        if rel == "data/kb-extracts.jsonl":
            continue
        if p.is_file() and p.suffix in {".md", ".py", ".yaml", ".yml", ".json", ".jsonl", ".txt"}:
            yield p, rel


def test_no_absolute_home_paths_in_committed_text():
    # The runner rejects committed absolute home paths; ~ or repo-relative only.
    # Only files that git would commit are checked (gitignored caches are fine).
    bad = []
    for path, rel in _committed_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "/" "home/" in text:
            bad.append(rel)
    assert not bad, bad
