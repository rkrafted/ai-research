#!/usr/bin/env python3
"""Pick one of N options for a text: the cascade measured in cascade_bench.py, as a library + CLI.

  1. shortlist   options carry examples -> embed (nomic via Ollama), score = mean top-3 cosine
  2. out-of-scope  top score minus that option's typical score (leave-one-out over its own examples, for
                 options with >= 4) < t_s, or JEV "is it one of these?" < t_in -> refused (when allow_none)
  3. System 1    JEV-27B (hub-jev) picks 1 of k, options shown with their 3 nearest examples;
                 asked in rank order, reversed order, and with "none of these" added
  4. gate        embedding top-1 = both JEV picks and none not chosen -> accepted (~99% right)
  5. otherwise   "none of these" chosen -> refused; else JEV's own pick -> escalated (flag it for review).
                 Saluki thinking over the same options measured no better on CLINC150, so it is not run.

Defaults (k, t_s, t_in) come from state/experts/bench/cascade/report4.json (CLINC150, val-fitted).
Thresholds do not transfer between datasets: refit them on the caller's own held-out set.

  decide.py "text" options.json [--k 3] [--t-s X] [--t-in Y] [--no-none]
  options.json: [{"label": "...", "examples": ["...", ...]}, ...]
Served by `experts.py proxy` as POST /v1/decide {"text", "options", ...same keys}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None  # the plain-Python path is fine for tens of options with tens of examples

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experts  # noqa: E402
import jev_decide  # noqa: E402

OLLAMA = "http://127.0.0.1:11434/api/embed"
EMB_MODEL = "nomic-embed-text"
REPORT = experts.STATE / "bench" / "cascade" / "report4.json"
NONE = "none of these"
QUESTION = "Which intent does this user message express?"  # the wording the thresholds were fitted with
IN_SCOPE_Q = "Does this user message express one of the listed intents?"
_cache: dict[str, list[float]] = {}
_opts: dict[str, tuple] = {}


def defaults() -> dict:
    rep = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    return {"k": rep.get("k_from_val", 3), "t_s": rep.get("t_s_from_val", 0.0), "t_in": rep.get("t_in_from_val", 0.0)}


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def embed(texts: list[str]) -> list[list[float]]:
    """Unit vectors; option examples repeat across calls, so they are cached by content."""
    keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    todo = [t for t, k in zip(texts, keys) if k not in _cache]
    for i in range(0, len(todo), 256):
        body = {"model": EMB_MODEL, "input": ["classification: " + t for t in todo[i:i + 256]]}
        req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            for t, v in zip(todo[i:i + 256], json.loads(r.read())["embeddings"]):
                _cache[hashlib.sha256(t.encode()).hexdigest()] = _unit(v)
    return [_cache[k] for k in keys]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _typical(vs) -> float:
    """Mean over an option's examples of their mean top-3 cosine to the option's other examples."""
    if len(vs) < 4:
        return 0.0
    if np is not None:
        sim = vs @ vs.T
        np.fill_diagonal(sim, -1)
        return float(np.sort(sim, axis=1)[:, -3:].mean())
    tot = 0.0
    for i, a in enumerate(vs):
        tot += sum(sorted((_dot(a, b) for j, b in enumerate(vs) if j != i), reverse=True)[:3]) / 3
    return tot / len(vs)


def _option(ex: list[str]) -> tuple:
    """(example vectors, typical score), cached: an option's examples do not change between calls."""
    key = hashlib.sha256("\0".join(ex).encode()).hexdigest()
    if key not in _opts:
        vs = embed(ex)
        vs = np.array(vs, dtype=np.float32) if np is not None else vs
        _opts[key] = (vs, _typical(vs))
    return _opts[key]


def shortlist(text: str, options: list[dict]) -> tuple[list[int], list[float], dict[int, list[str]]]:
    """(option indices best first by raw score, their scores minus typical, 3 nearest examples per option)."""
    if not any(o.get("examples") for o in options):
        return list(range(len(options))), [1.0] * len(options), {}
    q = embed([text])[0]
    qv = np.array(q, dtype=np.float32) if np is not None else q
    raw, typ, near = [], [], {}
    for i, o in enumerate(options):
        ex = o.get("examples") or []
        if not ex:
            raw.append(-1.0)
            typ.append(0.0)
            continue
        vs, mu = _option(ex)
        sims = (vs @ qv).tolist() if np is not None else [_dot(q, v) for v in vs]
        top = sorted(zip(sims, ex), key=lambda x: -x[0])[:3]
        raw.append(sum(s for s, _ in top) / len(top))
        typ.append(mu)
        near[i] = [t for _, t in top]
    order = sorted(range(len(options)), key=lambda i: -raw[i])
    return order, [raw[i] - typ[i] for i in order], near


def _show(o: dict, near: list[str] | None) -> str:
    return f'{o["label"]} (like: {"; ".join(chr(34) + t + chr(34) for t in near)})' if near else o["label"]


def decide(text: str, options: list[dict], k: int | None = None, t_s: float | None = None,
           t_in: float | None = None, allow_none: bool = True) -> dict:
    d = defaults()
    k = min(k or d["k"], len(options), 15)
    t_s = d["t_s"] if t_s is None else t_s
    t_in = d["t_in"] if t_in is None else t_in
    order, scores, near = shortlist(text, options)
    cand = order[:k]
    shown = [_show(options[i], near.get(i)) for i in cand]
    res = {"candidates": [options[i]["label"] for i in cand], "scores": [round(s, 4) for s in scores[:k]]}
    jev = jev_decide._lora("hub-jev")
    if allow_none:
        listing = "\n".join(f"- {s}" for s in shown)
        p_in = jev_decide.decide("noul", f"{text}\n\n[listed intents]\n{listing}", IN_SCOPE_Q, [], jev)["probabilities"][1]
        res["p_in"] = round(p_in, 4)
        if (near and scores[0] < t_s) or p_in < t_in:
            return {**res, "choice": None, "status": "refused"}
    if len(cand) == 1:
        return {**res, "choice": res["candidates"][0], "status": "accepted"}
    one = jev_decide.decide("choice", text, QUESTION, shown, jev)
    rev = jev_decide.decide("choice", text, QUESTION, shown[::-1], jev)
    pick, pick_rev = one["choice_index"], len(cand) - 1 - rev["choice_index"]
    none_ok = True
    if allow_none:
        nn = jev_decide.decide("choice", text, QUESTION, shown + [NONE], jev)
        none_ok = nn["choice_index"] != len(cand)
    res["p"] = max(one["probabilities"])
    if pick == pick_rev == 0 and none_ok and near:
        return {**res, "choice": res["candidates"][0], "status": "accepted"}
    if not none_ok:
        return {**res, "choice": None, "status": "refused", "by": "none-arm"}
    return {**res, "choice": res["candidates"][pick], "status": "escalated"}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("options", help="JSON file: [{label, examples}]")
    ap.add_argument("--k", type=int)
    ap.add_argument("--t-s", type=float)
    ap.add_argument("--t-in", type=float)
    ap.add_argument("--no-none", action="store_true")
    a = ap.parse_args(argv)
    opts = json.loads(Path(a.options).read_text())
    print(json.dumps(decide(a.text, opts, a.k, a.t_s, a.t_in, not a.no_none)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
