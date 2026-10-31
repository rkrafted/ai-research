"""Offline tests for the R7 small-vs-big harness.

Nothing here builds a real workspace, downloads a model or writes a result file
under `state/small-vs-big/`.  The episode test uses a fake backend and a tmp
workspace; `aggregate` is pure.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from smallvsbig import arms, backends, config, run
from smallvsbig import tasks as tasklib
from smallvsbig.distill import agent as A


def test_spec_union_is_58_unique():
    specs = tasklib.load_specs()
    assert len(specs) == 58
    assert len({t["id"] for t in specs}) == 58
    by_source = {}
    for t in specs:
        by_source[t["source"]] = by_source.get(t["source"], 0) + 1
    assert by_source == {"agent-distill": 44, "shootout": 14}
    assert all(t.get("test_cmd") for t in specs)
    assert all(t.get("issue") for t in specs)


def test_every_spec_has_a_buildable_origin():
    for t in tasklib.load_specs():
        if t.get("origin") == "injected":
            assert t.get("base_commit") and t.get("target_file") and t.get("mutation")
        else:
            assert t.get("parent") and t.get("commit")


def test_arm_e_router_by_repo():
    assert arms.route_key({"repo": "job-radar"}) == "job-radar"
    assert arms.route_key({"repo": "prior-art"}) == "prior-art"
    assert arms.route_key({"repo": "sponsor-check"}) == "job-radar"


def test_arm_inventory_is_a_to_g():
    assert sorted(arms.ARMS) == list("ABCDEFG")
    assert arms.ARMS["B"]["terse"] and not arms.ARMS["A"]["terse"]


def test_terse_addon_is_payload_safe():
    text = config.terse_addon()
    assert "NEVER compress" in text
    assert not text.startswith("#")
    assert "tool-call arguments" in text


def test_backends_import_without_a_gpu():
    assert hasattr(backends, "LayerRepeatBackend")
    assert hasattr(backends, "RoutedBackend")
    assert backends.ollama_base().startswith("http")


def test_aggregate_math():
    recs = [
        {"passed": True, "wall_time_s": 60.0, "tokens_out": 100, "peak_vram_gb": 5.0, "escalations": 0},
        {"passed": False, "wall_time_s": 120.0, "tokens_out": 300, "peak_vram_gb": 6.5, "escalations": 1},
    ]
    out = run.aggregate(recs, "F")
    assert out["arm"] == "F" and out["n"] == 2 and out["passed"] == 1
    assert out["pass_rate"] == 0.5
    assert out["gpu_minutes"] == 3.0
    assert out["tokens_out_per_task"] == 200.0
    assert out["peak_vram_gb"] == 6.5
    assert out["escalations"] == 1
    assert out["passes_per_gpu_hour"] == 20.0
    for key in ("n", "passed", "pass_rate", "median_wall_s", "tokens_out_per_task",
                "peak_vram_gb", "gpu_minutes", "passes_per_gpu_hour", "escalations"):
        assert key in out


def test_aggregate_empty_is_zero_safe():
    out = run.aggregate([], "A")
    assert out["n"] == 0 and out["passes_per_gpu_hour"] == 0.0 and out["gpu_minutes"] == 0.0


def test_episode_loop_with_fake_backend(tmp_path, monkeypatch):
    workspace = tmp_path / "work" / "t1" / "workspace"
    workspace.mkdir(parents=True)
    monkeypatch.setattr(config, "WORK", tmp_path / "work")
    monkeypatch.setattr(config, "WS", tmp_path / "ws")

    class Fake:
        name = "fake"
        model_name = "fake"

        def complete(self, messages, deadline):
            call = {"function": {"name": "finish", "arguments": "{}"}}
            return ({"role": "assistant", "content": "", "tool_calls": [call]},
                    {"input": 3, "output": 7, "cache_read": 0}, "fake")

    task = {"id": "t1", "issue": "noop", "test_cmd": "true", "kind": "bugfix"}
    rec = run.run_episode("A", task, Fake(), terse=False)
    assert rec["passed"] is True
    assert rec["tokens_out"] == 7
    assert rec["arm"] == "A"
    assert rec["failure_mode"] == "pass"


def test_queued_jobs_are_valid():
    yamls = sorted((config.REPO_ROOT / "gpu_jobs").glob("r7-*.job.yaml"))
    assert yamls, "no r7 gpu jobs queued"
    for path in yamls:
        text = path.read_text()
        assert "script:" in text
        for line in text.splitlines():
            if line.strip().startswith("script:"):
                rel = line.split(":", 1)[1].strip()
                assert (config.REPO_ROOT / rel).is_file(), f"{path.name}: missing {rel}"
            if line.strip().startswith("timeout_minutes:"):
                assert int(float(line.split(":", 1)[1])) <= 240
