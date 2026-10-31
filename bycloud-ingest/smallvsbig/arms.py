"""Arm inventory and backend construction for R7."""
from __future__ import annotations

from . import backends, config

# Which arm buys which backend.  ``terse`` turns on the caveman system add-on
# (prose/reasoning only).  F is a two-stage cascade handled in run.py.
ARMS: dict[str, dict] = {
    "A": {"kind": "ollama", "model": config.QWEN27, "terse": False},
    "B": {"kind": "ollama", "model": config.QWEN27, "terse": True},
    "C": {"kind": "transformers", "base": config.SERA, "adapter": config.STUDENT, "terse": False},
    "D": {"kind": "transformers", "base": config.SERA, "adapter": config.STUDENT, "terse": True},
    "E": {"kind": "routed", "base": config.SERA,
          "adapters": {"job-radar": config.EXPERT_JOBRADAR,
                       "prior-art": config.EXPERT_PRIORART},
          "terse": False},
    "F": {"kind": "cascade", "terse": False},
    "G": {"kind": "layerrepeat", "base": config.SERA, "repeat": 1, "terse": False},
}


def route_key(task: dict) -> str:
    """Arm E's router: pick the family expert by repo, fall back to job-radar.

    The two held-in teacher families are job-radar and prior-art; every other
    held-out repo is out of distribution and goes to the larger expert.
    """
    repo = task.get("repo", "")
    return repo if repo in ("job-radar", "prior-art") else "job-radar"


def make_backend(arm: str):
    """Construct the arm's backend (models load lazily on first call)."""
    spec = ARMS[arm]
    kind = spec["kind"]
    if kind == "ollama":
        return backends.make_ollama(spec["model"])
    if kind == "transformers":
        return backends.TransformersBackend(spec["base"], adapter=spec.get("adapter"))
    if kind == "layerrepeat":
        return backends.LayerRepeatBackend(spec["base"], repeat=spec.get("repeat", 1))
    if kind == "routed":
        return backends.RoutedBackend(spec["base"], spec["adapters"])
    raise SystemExit(f"arm {arm} has no single backend (kind={kind})")
