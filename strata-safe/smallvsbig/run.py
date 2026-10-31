#!/usr/bin/env python3
"""Run one R7 arm (or finalize it) on the 58-task union.

    # one shard of arm A on the 3090 (via gpu_jobs/r7-*.job.yaml)
    python3 smallvsbig/run.py --arm A --shard 0/3 --workers 1

    # write state/small-vs-big/F.json from the per-task records (a separate job)
    python3 smallvsbig/run.py --finalize F

Per-task records land under ``state/small-vs-big/runs/<arm>/<task>.json``
(gitignored, resumable).  The final per-arm file is written only by
``--finalize``, which the queued aggregate job runs on the GPU.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallvsbig import arms, backends, config, tasks as tasklib  # noqa: E402
from smallvsbig.distill import agent as A  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ harness glue
def configure_harness(terse: bool) -> None:
    A.MAX_TURNS = config.MAX_TURNS
    A.NUM_CTX = config.NUM_CTX
    A.EPISODE_TIMEOUT_S = config.EPISODE_TIMEOUT_S
    if terse:
        A.SYSTEM_PROMPT = A.SYSTEM_PROMPT + "\n\n" + config.terse_addon()
    else:
        A.SYSTEM_PROMPT = A.SYSTEM_PROMPT.split("\n\nTERSE OUTPUT (caveman style)")[0]


def copy_workspace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, symlinks=True)


def restore_tests(pristine: Path, ws: Path, test_files: list[str]) -> None:
    rels = list(test_files or [])
    if not rels:
        for pat in ("test_*.py", "*_test.py", "conftest.py"):
            for p in pristine.rglob(pat):
                rels.append(str(p.relative_to(pristine)))
    for rel in rels:
        src = pristine / rel
        dst = ws / rel
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def peak_vram_gb() -> float:
    """Max VRAM seen: torch for in-process models, ollama /api/ps otherwise."""
    best = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            best = torch.cuda.max_memory_allocated() / 1024 ** 3
    except Exception:  # noqa: BLE001
        pass
    try:
        import urllib.request
        with urllib.request.urlopen(backends.ollama_base().rstrip("/") + "/api/ps", timeout=5) as r:
            data = json.loads(r.read().decode())
        for m in data.get("models", []):
            best = max(best, (m.get("size_vram") or 0) / 1e9)
    except Exception:  # noqa: BLE001
        pass
    return round(best, 2)


def unload_ollama(model: str) -> None:
    import urllib.request
    try:
        req = urllib.request.Request(
            backends.ollama_base().rstrip("/") + "/api/generate",
            data=json.dumps({"model": model, "keep_alive": 0}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:  # noqa: BLE001
        pass


def free_model(backend) -> None:
    """Drop an in-process model and its CUDA cache (used before the 27B escalation)."""
    if getattr(backend, "model", None) is None:
        return
    try:
        backend.model = None
    except Exception:  # noqa: BLE001
        pass
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def gate_ok() -> tuple[bool, str]:
    """Honor the spec note: the fixed agent-distill harness gate must have passed.

    Missing gate file -> (True, 'missing') so a fresh checkout still runs; a gate
    that exists and fails -> (False, reason) and the arm job does nothing.
    """
    path = config.SIBLING / "state" / "harness-gate.json"
    if not path.is_file():
        return True, "gate file absent"
    try:
        g = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:  # noqa: BLE001
        return False, f"gate unreadable: {e}"
    if int(g.get("teacher_solved") or 0) < 3:
        return False, f"teacher_solved={g.get('teacher_solved')}"
    if not g.get("base_ran_tests_in_repo"):
        return False, "base_ran_tests_in_repo is false"
    return True, "ok"


def run_episode(arm: str, task: dict, backend, terse: bool, retry_context: str = "") -> dict:
    pristine = config.WORK / task["id"] / "workspace"
    if not pristine.is_dir():
        raise FileNotFoundError(f"task workspace missing: {pristine}; run smallvsbig.tasks first")
    ws = config.WS / arm / task["id"] / "workspace"
    copy_workspace(pristine, ws)
    task2 = dict(task)
    if retry_context:
        task2["issue"] = (task.get("issue") or "") + retry_context
    try:
        rec = A.run_episode(
            backend, task2, ws, max_turns=config.MAX_TURNS,
            test_cmd=task.get("test_cmd", "python3 -m pytest -q"),
            restore_tests=lambda: restore_tests(pristine, ws, task.get("test_files") or []),
        )
    finally:
        shutil.rmtree(ws.parent, ignore_errors=True)
    rec["arm"] = arm
    rec["task"] = task["id"]
    rec["task_kind"] = task.get("kind")
    rec["source"] = task.get("source")
    rec["tokens_out"] = int((rec.get("usage") or {}).get("output") or 0)
    rec["peak_vram_gb"] = peak_vram_gb()
    return rec


def cascade(arm: str, task: dict, state: dict) -> dict:
    """Student attempt -> retry with the failure output -> escalate to the 27B."""
    student = state.setdefault("student", backends.TransformersBackend(config.SERA, adapter=config.STUDENT))
    attempts: list[dict] = []
    rec = run_episode(arm, task, student, terse=False)
    attempts.append(rec)
    if not rec.get("passed"):
        fail = (rec.get("test_output") or "")[-1500:]
        context = ("\n\n# Previous attempt failed\nYour last fix did not pass. "
                   "Test output:\n```\n" + fail + "\n```\nFix the remaining failure.")
        rec = run_episode(arm, task, student, terse=False, retry_context=context)
        attempts.append(rec)
    escalations = 0
    if not rec.get("passed"):
        escalations = 1
        free_model(student)
        qwen = state.setdefault("qwen", backends.make_ollama(config.QWEN27))
        rec = run_episode(arm, task, qwen, terse=False)
        attempts.append(rec)
        unload_ollama(config.QWEN27)
    out = dict(rec)
    out.update({
        "attempts": [{"passed": a.get("passed"), "failure_mode": a.get("failure_mode"),
                      "turns": a.get("turns"), "tokens_out": a.get("tokens_out"),
                      "wall_time_s": a.get("wall_time_s")} for a in attempts],
        "n_attempts": len(attempts),
        "escalations": escalations,
        "tokens_out": sum(a.get("tokens_out", 0) for a in attempts),
        "wall_time_s": round(sum(a.get("wall_time_s", 0) for a in attempts), 2),
        "peak_vram_gb": max(a.get("peak_vram_gb", 0) for a in attempts),
    })
    return out


# ------------------------------------------------------------------ shard runner
def run_shard(arm: str, selected: list[dict], force: bool = False) -> dict:
    if arm not in arms.ARMS:
        raise SystemExit(f"unknown arm {arm!r}; choose from {sorted(arms.ARMS)}")
    outdir = config.RUNS / arm
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in selected if force or not (outdir / f"{t['id']}.json").is_file()]
    log(f"arm={arm} selected={len(selected)} todo={len(todo)}")
    backend = state = None
    terse = arms.ARMS[arm].get("terse", False)
    configure_harness(terse)
    if arm != "F":
        backend = arms.make_backend(arm)
    done = passed = 0
    for task in todo:
        if arm == "F":
            state = state if state is not None else {}
            rec = cascade(arm, task, state)
        else:
            if arms.ARMS[arm]["kind"] == "routed":
                backend.current = arms.route_key(task)
            rec = run_episode(arm, task, backend, terse)
        rec = config.sanitize(rec)
        (outdir / f"{task['id']}.json").write_text(json.dumps(rec, indent=2))
        done += 1
        passed += bool(rec.get("passed"))
        log(f"  {arm} {task['id']}: passed={rec.get('passed')} turns={rec.get('turns')} "
            f"{rec.get('wall_time_s')}s mode={rec.get('failure_mode')}")
    summary = summarize(outdir)
    log(f"arm={arm} shard finished: ran={done} passed={passed} total_n={summary['n']}")
    return summary


def summarize(run_dir: Path) -> dict:
    import statistics
    recs = []
    for f in sorted(run_dir.glob("*.json")):
        try:
            recs.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue
    n = len(recs)
    passed = sum(1 for r in recs if r.get("passed"))
    walls = [r.get("wall_time_s", 0) for r in recs]
    toks = [r.get("tokens_out", 0) for r in recs]
    return {"n": n, "passed": passed,
            "median_wall_s": round(statistics.median(walls), 2) if walls else 0.0,
            "tokens_out_per_task": round(sum(toks) / n, 1) if n else 0.0}


def aggregate(recs: list[dict], arm: str) -> dict:
    """Pure reduction of per-task records to the schema the done-check reads."""
    import statistics
    n = len(recs)
    passed = sum(1 for r in recs if r.get("passed"))
    walls = [r.get("wall_time_s", 0) for r in recs]
    toks = [r.get("tokens_out", 0) for r in recs]
    vram = max([r.get("peak_vram_gb", 0) for r in recs] or [0])
    gpu_minutes = round(sum(walls) / 60.0, 2)
    return {
        "arm": arm,
        "n": n,
        "passed": passed,
        "pass_rate": round(passed / n, 4) if n else 0.0,
        "median_wall_s": round(statistics.median(walls), 2) if walls else 0.0,
        "tokens_out_per_task": round(sum(toks) / n, 1) if n else 0.0,
        "peak_vram_gb": vram,
        "gpu_minutes": gpu_minutes,
        "passes_per_gpu_hour": round(passed / (gpu_minutes / 60.0), 2) if gpu_minutes else 0.0,
        "escalations": sum(int(r.get("escalations") or 0) for r in recs),
        "model": arms.ARMS[arm].get("model") or arms.ARMS[arm].get("base"),
        "terse": arms.ARMS[arm].get("terse", False),
    }


def finalize(arm: str) -> dict:
    """Aggregate the per-task records and write the final per-arm file."""
    run_dir = config.RUNS / arm
    recs = []
    for f in sorted(run_dir.glob("*.json")) if run_dir.is_dir() else []:
        try:
            recs.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue
    out = aggregate(recs, arm)
    config.STATE.mkdir(parents=True, exist_ok=True)
    (config.STATE / f"{arm}.json").write_text(json.dumps(config.sanitize(out), indent=2))
    log(f"finalized {arm}: n={out['n']} passed={out['passed']} gpu_minutes={out['gpu_minutes']}")
    return out


# ------------------------------------------------------------------ cli
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R7 small-vs-big arm runner")
    ap.add_argument("--arm", default=None, choices=sorted(arms.ARMS))
    ap.add_argument("--finalize", default=None, help="arm letter or 'all'")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None, help="K/N: tasks with index %% N == K")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if a.finalize:
        names = sorted(arms.ARMS) if a.finalize == "all" else [a.finalize]
        for name in names:
            finalize(name)
        return 0

    if not a.arm:
        ap.error("--arm or --finalize is required")

    manifest = tasklib.load_manifest()
    if not manifest and not a.skip_build:
        log("building task workspaces (first run) ...")
        manifest = tasklib.build_all()
    selected = [t for t in manifest if "error" not in t]
    if a.tasks:
        want = set(a.tasks)
        selected = [t for t in selected if t["id"] in want]
    if a.limit:
        selected = selected[: a.limit]
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        selected = [t for i, t in enumerate(selected) if i % n == k]
    log(f"arm={a.arm} tasks={len(selected)}")
    if a.dry_run:
        for t in selected:
            print(t["id"], t.get("kind"), t.get("source"))
        return 0
    if not selected:
        return 2
    if not a.skip_build:
        missing = [t["id"] for t in selected
                   if not (config.WORK / t["id"] / "workspace").is_dir()]
        if missing:
            tasklib.build_all(ids=missing)
    ok, why = gate_ok()
    if not ok:
        log(f"harness gate not passed ({why}); arm {a.arm} skipped")
        return 0
    run_shard(a.arm, selected, force=a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
