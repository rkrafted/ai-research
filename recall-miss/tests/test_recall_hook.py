"""Offline checks for the proposed recall hook (fictional fixture, no ~/.claude)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import recall_hook  # noqa: E402


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    orch = tmp_path / "orchestrator"
    (orch / "state").mkdir(parents=True)
    units = tmp_path / "units"
    units.mkdir()
    (units / "queue-watchdog.service").write_text(
        "[Service]\nDescription=agent-watchdog queue check for stuck jobs\n")
    (units / "queue-watchdog.timer").write_text(
        "[Timer]\nDescription=Run the queue check every 5 minutes\nOnUnitActiveSec=5min\n")
    (units / "orchestrator-health.service").write_text(
        "[Service]\nDescription=Orchestrator health: alerts for failed or idle work\n")
    (units / "orchestrator-health.timer").write_text(
        "[Timer]\nOnUnitActiveSec=5min\n")
    now = 1_800_000_000
    rows = [
        {"ts": now - 60, "key": "gpu_idle", "msg": "GPU idle while 12 GPU task(s) are ready"},
        {"ts": now - 120, "key": "unit:demo", "msg": "demo failed"},
        {"ts": now - 30, "key": "unit:demo", "msg": "demo failed again"},
    ]
    (orch / "state/alerts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    (orch / "state/health.json").write_text(json.dumps({"unit:demo": 1}))
    return orch, units


def test_ops_prompt_names_both_systems(tmp_path):
    orch, units = fixture(tmp_path)
    ctx = recall_hook.build_context(orch, units, "user_prompt",
                                    "why is the gpu idle and the queue stuck",
                                    now=1_800_000_000)
    assert "queue-watchdog" in ctx
    assert "health.py" in ctx
    assert "alerts.jsonl" in ctx


def test_alerts_are_deduped_by_key(tmp_path):
    orch, units = fixture(tmp_path)
    ctx = recall_hook.build_context(orch, units, "user_prompt",
                                    "gpu idle", now=1_800_000_000)
    assert ctx.count("demo failed") == 1
    assert "GPU idle" in ctx


def test_unrelated_prompt_pushes_nothing(tmp_path):
    orch, units = fixture(tmp_path)
    ctx = recall_hook.build_context(orch, units, "user_prompt",
                                    "write a poem about trees", now=1_800_000_000)
    assert ctx == ""


def test_session_start_pushes_the_live_list(tmp_path):
    orch, units = fixture(tmp_path)
    ctx = recall_hook.build_context(orch, units, "session_start", now=1_800_000_000)
    assert "Systems we run" in ctx
    assert "queue-watchdog" in ctx and "orchestrator-health" in ctx


def test_hit_classifier_is_exact():
    import recall_replay
    assert recall_replay.hits("queue-watchdog and health.py") == (True, True)
    assert recall_replay.hits("watchdog-waste is a different tool") == (False, False)
