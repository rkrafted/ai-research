"""R7 small-vs-big configuration, paths and the arm inventory.

Everything a job needs is derived from the repo root and the read-only source
repos under ``~/theprojects/queue`` (the same repos the two task builders use).
Environment overrides exist so one job can vary the model without editing code.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUEUE = Path.home() / "theprojects" / "queue"

STATE = REPO_ROOT / "state" / "small-vs-big"
WORK = STATE / "work"            # pristine built workspaces (gitignored)
RUNS = STATE / "runs"            # per-task episode records (gitignored)
WS = STATE / "ws"                # per-episode scratch copies (gitignored)

TASKS = REPO_ROOT / "tasks"
ADAPTERS = REPO_ROOT / "artifacts"

def _first_existing(candidates: list, default):
    for c in candidates:
        if c.exists():
            return str(c)
    return default


# ---- models on disk ---------------------------------------------------------
QWEN27 = os.environ.get("R7_QWEN27", "qwen3.8-27b-uncensored:latest")
SERA = os.environ.get("R7_SERA", "allenai/SERA-8B")
# The trained SERA-8B student (agent-distill).  Prefer our gitignored copy so a
# fresh worktree still finds it; fall back to the sibling project's adapter dir.
STUDENT = os.environ.get("R7_STUDENT") or _first_existing(
    [ADAPTERS / "sera-8b-student",
     Path.home() / "theprojects/queue/.worktrees/agent-distill/agent-distill/state/adapters/sera-8b-r1"],
    str(ADAPTERS / "sera-8b-student"))
# Two specialists on the same SERA base (arm E); produced by smallvsbig/train_expert.py.
# The split axis is the task family (repo): the two repos that supply every held-in
# teacher trajectory, job-radar and prior-art.  Other repos route to job-radar.
EXPERT_JOBRADAR = os.environ.get("R7_EXPERT_JOBRADAR", str(ADAPTERS / "expert-job-radar"))
EXPERT_PRIORART = os.environ.get("R7_EXPERT_PRIORART", str(ADAPTERS / "expert-prior-art"))
# Where the proven trainer and its trajectories live (agent-distill, completed).
SIBLING = Path(os.environ.get("R7_SIBLING", "~/theprojects/queue/agent-distill")).expanduser()

# ---- harness ----------------------------------------------------------------
MAX_TURNS = int(os.environ.get("R7_MAX_TURNS", "30"))
NUM_CTX = int(os.environ.get("R7_NUM_CTX", "32768"))
EPISODE_TIMEOUT_S = int(os.environ.get("R7_EPISODE_TIMEOUT", "1200"))
TERSE_PROMPT = REPO_ROOT / "prompts" / "caveman-system.txt"

_HOME_RE = re.compile(r"/" r"home/[A-Za-z0-9_.-]*")


def sanitize(value):
    """Replace absolute home-directory paths with ~ before anything is written."""
    if isinstance(value, str):
        return _HOME_RE.sub("~", value)
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    return value


def terse_addon() -> str:
    """The caveman rules without their leading comment lines (prose/reasoning only)."""
    text = TERSE_PROMPT.read_text(encoding="utf-8")
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("#")).strip()
