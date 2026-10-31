#!/usr/bin/env python3
"""Proposed recall hook: push the live systems list and deduplicated open alerts.

Not installed in ~/.claude.  It is a runnable prototype for docs/RECALL_MISS.md and
for tools/recall_replay.py, which compares it with ~/.claude/patterns/recall.py.

Why it is push-first:
  * the "systems we run" list is small (user services + timers) and changes rarely,
    so SessionStart pushes it instead of hoping a keyword search finds it later;
  * alerts are deduplicated by health key, capped, and only recent/open ones are
    shown, instead of replaying the 450-line alerts.jsonl;
  * a UserPromptSubmit call adds only the 1-3 systems that match the prompt, and
    expands operational words (bottleneck/slow/stuck/idle) to the watchdog and
    health systems even when the user never says "watchdog".

Usage:
  recall_hook.py session-start [--now EPOCH]
  recall_hook.py user-prompt   [--now EPOCH]   # prompt JSON on stdin
  recall_hook.py compact       [--now EPOCH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HOME = Path.home()
DEFAULT_ORCH = HOME / "theprojects/orchestrator"
DEFAULT_UNITS = HOME / ".config/systemd/user"

# Units we own.  Same family health.py watches; xerj/laya are local services too.
OUR_UNITS = (
    "orchestrator-", "queue-watchdog", "stallwatch", "gpu-queue", "laya-",
    "xerj", "ollama", "automaton", "dr-watcher", "ai-memory",
)
OPS = re.compile(
    r"watchdog|monitor|alert|health|stall|stuck|frozen|hung|slow|bottleneck|"
    r"idle|gpu|queue|runner|fail|crash|timer|service|efficien|waste", re.I)
RUN = re.compile(r"watchdog|alert|health|stall|monitor|every \d+ ?min|running", re.I)
CORE = ("queue-watchdog", "orchestrator-health")
STOP = set("""a an the and or of to in on at by for with from as is are was were be
it its this that these those i you we they me my our your do does did not no yes so
very can could should would will just than also into out up down over about after
before what which who how why when where all any some more most other such only own
same few each both make made get got use used using need needs want wants like one
two new now via per let lets please""".split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", text.lower()) if t not in STOP]


def _unit_files(unit_dir: Path) -> dict[str, dict]:
    """Parse user unit files.  systemctl often has no bus in a sandbox; files work."""
    units: dict[str, dict] = {}
    if not unit_dir.is_dir():
        return units
    for p in sorted(unit_dir.glob("*.service")):
        if not p.name.startswith(OUR_UNITS):
            continue
        body = p.read_text(errors="ignore")
        desc = next((l.split("=", 1)[1].strip() for l in body.splitlines()
                     if l.startswith("Description=")), p.stem)
        units[p.stem] = {"name": p.stem, "kind": "service", "purpose": desc,
                         "timer": "", "path": ""}
    for p in sorted(unit_dir.glob("*.timer")):
        if not p.name.startswith(OUR_UNITS):
            continue
        body = p.read_text(errors="ignore")
        desc = next((l.split("=", 1)[1].strip() for l in body.splitlines()
                     if l.startswith("Description=")), p.stem)
        every = next((l.split("=", 1)[1].strip() for l in body.splitlines()
                      if l.startswith(("OnUnitActiveSec=", "OnCalendar="))), "")
        stem = p.stem
        target = stem if stem in units else stem.replace("-timer", "")
        if target in units:
            units[target]["timer"] = every or desc
        else:
            units[stem] = {"name": stem, "kind": "timer", "purpose": desc,
                           "timer": every, "path": ""}
    return units


def _queue_systems(orch: Path) -> list[dict]:
    """Queue specs whose repo has a deploy/*.timer, plus orchestrator tools."""
    out = []
    queue_dir = orch / "queue"
    if queue_dir.is_dir():
        for p in sorted(queue_dir.glob("*.yaml")):
            text = p.read_text(errors="ignore")
            title = re.search(r"^title:\s*(.+)$", text, re.M)
            repo = re.search(r"^repo:\s*(.+)$", text, re.M)
            status = re.search(r"^status:\s*(.+)$", text, re.M)
            if not title:
                continue
            repo_dir = HOME / "theprojects/queue" / (repo.group(1).strip() if repo else "")
            if not list((repo_dir / "deploy").glob("*.timer")):
                continue
            out.append({"name": p.stem, "kind": "queue", "purpose": title.group(1).strip(),
                        "timer": "timer", "path": str(p)})
    # bin/*.py tools with an operational docstring: health.py is not a queue item.
    bin_dir = orch / "bin"
    if bin_dir.is_dir():
        for p in sorted(bin_dir.glob("*.py")):
            try:
                head = p.read_text(errors="ignore").split('"""', 2)
                doc = head[1].strip().splitlines()[0] if len(head) > 2 and head[1].strip() else ""
            except OSError:
                continue
            if doc and RUN.search(p.stem + " " + doc):
                out.append({"name": p.stem, "kind": "tool", "purpose": doc,
                            "timer": "", "path": str(p)})
    return out


def systems(orch: Path, unit_dir: Path) -> list[dict]:
    by_name = {u["name"]: u for u in _unit_files(unit_dir).values()}
    for s in _queue_systems(orch):
        by_name.setdefault(s["name"], s)
    return list(by_name.values())


def _score(prompt: str, s: dict) -> int:
    qt = set(tokens(prompt))
    text = " ".join([s["name"], s.get("purpose", ""), s.get("timer", "")])
    st = set(tokens(text))
    score = 8 * len(qt & set(tokens(s["name"]))) + len(qt & st)
    if score and s.get("kind") == "service":
        score += 2
    if OPS.search(prompt) and OPS.search(text):
        score += 1
    return score


def rank(prompt: str, systems_: list[dict], limit: int = 4) -> list[dict]:
    if not prompt:
        return systems_[:limit]
    scored = [(_score(prompt, s), s) for s in systems_]
    scored = [(n, s) for n, s in scored if n > 0]
    scored.sort(key=lambda t: (-t[0], t[1]["name"]))
    picked = [s for _, s in scored]
    if OPS.search(prompt):
        # The ops pair is invariant: never let a wording change hide it again.
        core = [s for s in systems_ if s["name"] in CORE]
        picked = core + [s for s in picked if s["name"] not in CORE]
    return picked[:limit]


def _open_keys(orch: Path) -> set[str]:
    try:
        data = json.loads((orch / "state/health.json").read_text())
    except (OSError, ValueError):
        return set()
    return {k for k in data if not k.startswith("_")}


def alerts(orch: Path, now: float, since_h: float = 24, limit: int = 3,
           os_query: str = "") -> list[dict]:
    """One row per health key: newest message, recent, open keys first."""
    path = orch / "state/alerts.jsonl"
    latest: dict[str, dict] = {}
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        ts = int(r.get("ts", 0))
        if ts > now or now - ts > since_h * 3600:
            continue
        key = r.get("key", "")
        if not key:
            continue
        if key not in latest or ts > latest[key]["ts"]:
            latest[key] = {"ts": ts, "key": key, "msg": r.get("msg", "")}
    opened = _open_keys(orch)
    rows = list(latest.values())
    def alert_score(r: dict) -> tuple:
        qt = set(tokens(r["msg"]))
        relevant = len(qt & set(tokens(os_query))) + (3 if "gpu_idle" in r["key"] and
                   re.search(r"gpu|idle|free", os_query, re.I) else 0)
        return (relevant, r["key"] in opened, r["ts"])
    rows.sort(key=alert_score, reverse=True)
    return rows[:limit]


def _line(s: dict) -> str:
    when = f" [{s['timer']}]" if s.get("timer") else ""
    purpose = re.sub(r"\s+", " ", s.get("purpose", ""))[:150]
    if "health" in s["name"]:
        purpose += " (bin/health.py) -> state/alerts.jsonl"
    return f"- {s['name']}{when}: {purpose}"


def build_context(orch: Path, unit_dir: Path, event: str, prompt: str = "",
                  now: float | None = None, max_alerts: int = 3) -> str:
    import time
    now = time.time() if now is None else now
    ss = systems(orch, unit_dir)
    if event in ("session_start", "compact"):
        picked = [s for s in ss if OPS.search(s["name"] + " " + s.get("purpose", ""))][:8]
    else:
        picked = rank(prompt, ss, 4)
    lines: list[str] = []
    if picked:
        where = "Systems we run (live; reuse before building anything new)" if event in ("session_start", "compact") \
            else "Ops context: existing systems that match this request (reuse, do not rebuild)"
        lines.append(where + ":")
        lines += [_line(s) for s in picked]
    rows = alerts(orch, now, limit=max_alerts, os_query=prompt) \
        if event != "user_prompt" or OPS.search(prompt) else []
    if rows:
        lines.append("Recent open alerts (deduplicated by key; do not repeat these as new):")
        for r in rows:
            lines.append(f"- {r['msg'][:180]} (key: {r['key']})")
    if event in ("session_start", "compact") and not rows:
        lines.append(f"No new alerts in the last 24h. Live alerts: {orch / 'state/alerts.jsonl'}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("event", choices=["session-start", "user-prompt", "compact"])
    ap.add_argument("--root", type=Path, default=DEFAULT_ORCH)
    ap.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--now", type=float, default=None)
    ap.add_argument("--max-alerts", type=int, default=3)
    ap.add_argument("--no-alerts", action="store_true")
    a = ap.parse_args()
    event = a.event.replace("-", "_")
    prompt = a.prompt
    if prompt is None and event == "user_prompt":
        try:
            prompt = json.load(sys.stdin).get("prompt", "")
        except ValueError:
            prompt = ""
    ctx = build_context(a.root, a.units, event, prompt or "", a.now,
                        0 if a.no_alerts else a.max_alerts)
    if not ctx:
        return 0
    name = "UserPromptSubmit" if event == "user_prompt" else "SessionStart"
    print(json.dumps({"hookSpecificOutput": {"hookEventName": name,
                                             "additionalContext": ctx}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
