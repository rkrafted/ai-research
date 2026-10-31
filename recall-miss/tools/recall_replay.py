#!/usr/bin/env python3
"""Replay one real session's prompts through two hooks and count recall hits.

Read-only for ~/.claude.  It never prints prompt bodies: only prompt indexes and
whether the output named the queue watchdog or bin/health.py.  The session id is a
UUID, not personal data.

Current hook: ~/.claude/patterns/recall.py hook, exactly as installed.
Proposed hook: tools/recall_hook.py, a SessionStart push plus a per-prompt rank.

The transcript cited by the task (~/.claude/projects/-home-neo/<id>.jsonl) was gone
in this snapshot (the projects directory is empty), so the prompt list falls back to
~/.claude/history.jsonl, which keeps {display, timestamp, sessionId} for every prompt.

Usage:
  python3 tools/recall_replay.py --session 620f5a88-f819-4b1b-a5de-f180bc3ed532
  python3 tools/recall_replay.py --session ... --json-out state/recall-replay.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
HISTORY = HOME / ".claude/history.jsonl"
CURRENT = HOME / ".claude/patterns/recall.py"
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import recall_hook  # noqa: E402

HIT_WD = re.compile(r"queue-watchdog|agent[_-]watchdog", re.I)
HIT_HEALTH = re.compile(r"health\.py|orchestrator-health|alerts\.jsonl", re.I)


def short(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(HOME))
    except ValueError:
        return str(path)


def prompts(path: Path, session: str) -> list[dict]:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("sessionId") != session or not (d.get("display") or "").strip():
            continue
        rows.append({"ts": int(d.get("timestamp", 0)) / 1000,
                     "prompt": d["display"]})
    return rows


def current_hook(prompt: str) -> str:
    if not CURRENT.exists():
        return ""
    r = subprocess.run([sys.executable, str(CURRENT), "hook"],
                       input=json.dumps({"prompt": prompt}),
                       capture_output=True, text=True, timeout=30)
    try:
        return json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]
    except (ValueError, KeyError):
        return ""


def hits(text: str) -> tuple[bool, bool]:
    return bool(HIT_WD.search(text)), bool(HIT_HEALTH.search(text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--history", type=Path, default=HISTORY)
    ap.add_argument("--root", type=Path, default=recall_hook.DEFAULT_ORCH)
    ap.add_argument("--units", type=Path, default=recall_hook.DEFAULT_UNITS)
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()

    rows = prompts(a.history, a.session)
    if not rows:
        print(f"no prompts for session {a.session} in {a.history}", file=sys.stderr)
        return 2

    ss = recall_hook.build_context(a.root, a.units, "session_start",
                                   now=rows[0]["ts"])
    ss_wd, ss_health = hits(ss)
    per = []
    for i, row in enumerate(rows):
        cur = current_hook(row["prompt"])
        prop = recall_hook.build_context(a.root, a.units, "user_prompt",
                                         row["prompt"], row["ts"])
        cwd, chealth = hits(cur)
        pwd, phealth = hits(prop)
        per.append({"idx": i, "ts": row["ts"],
                    "current": [cwd, chealth], "proposed": [pwd, phealth],
                    "current_ctx": bool(cur), "proposed_ctx": bool(prop)})

    cur_any = sum(1 for r in per if any(r["current"]))
    prop_any = sum(1 for r in per if any(r["proposed"]))
    print(f"session prompts: {len(rows)}  (source: {short(a.history)}, read-only)")
    print(f"SessionStart push: queue-watchdog={ss_wd} health.py={ss_health}")
    print(f"UserPromptSubmit: current hook any-hit {cur_any}/{len(per)}; "
          f"proposed any-hit {prop_any}/{len(per)}")
    print(f"  current returned some context on {sum(r['current_ctx'] for r in per)}/{len(per)} "
          f"prompts, but named neither system")
    print(f"  current  queue-watchdog {sum(r['current'][0] for r in per)}/{len(per)}, "
          f"health.py {sum(r['current'][1] for r in per)}/{len(per)}")
    print(f"  proposed queue-watchdog {sum(r['proposed'][0] for r in per)}/{len(per)}, "
          f"health.py {sum(r['proposed'][1] for r in per)}/{len(per)}")

    # The real case: the bottleneck question and the follow-up "you did not see watchdog".
    focus = [r for r in per if re.search(
        r"biggest bottleneck|find all bottlenecks|did not see watchdog", rows[r["idx"]]["prompt"], re.I)]
    for r in focus:
        label = ("bottleneck" if "bottleneck" in rows[r["idx"]]["prompt"].lower()
                 else "watchdog")
        print(f"  #{r['idx']} {label}: current={r['current']} proposed={r['proposed']}")

    # Show the proposed context for the bottleneck prompt (systems/alerts only, no user text).
    for r in focus:
        if "bottleneck" in rows[r["idx"]]["prompt"].lower():
            print("\nProposed context for the bottleneck prompt:")
            print(recall_hook.build_context(a.root, a.units, "user_prompt",
                                            rows[r["idx"]]["prompt"], rows[r["idx"]]["ts"]))
            break

    out = {"session": a.session, "prompts": len(rows),
           "session_start": {"queue_watchdog": ss_wd, "health_py": ss_health},
           "totals": {"current": {"queue_watchdog": sum(r["current"][0] for r in per),
                                   "health_py": sum(r["current"][1] for r in per),
                                   "any_context": sum(r["current_ctx"] for r in per)},
                      "proposed": {"queue_watchdog": sum(r["proposed"][0] for r in per),
                                    "health_py": sum(r["proposed"][1] for r in per)}},
           "focus": focus}
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
