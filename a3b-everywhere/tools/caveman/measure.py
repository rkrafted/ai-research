#!/usr/bin/env python3
"""Measure caveman prose compression on real assistant turns (offline, CPU).

For each turn in the extracted corpus:
  before = the raw assistant message (payload included)
  after  = the same message with only the prose segments replaced by the
           DeepSeek caveman rewrite; fenced code stays byte-exact
and both are tokenized with the Qwen3 tokenizer.

The result is ``state/caveman-savings.json``::

    {"n_turns", "tokens_before", "tokens_after", "saving_pct", ...}

Requires the raw corpus (``state/caveman-turns.raw.json``, gitignored) produced
by ``extract_turns.py`` and a local Qwen3 tokenizer (transformers +
``~/.cache/huggingface``). Run it with any Python that has transformers, e.g.::

    ~/theprojects/gpu/.venv/bin/python tools/caveman/measure.py

The committed JSON is the artifact; the raw transcript text never leaves
``state/`` (gitignored) and is never written back out.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

DEFAULT_RAW = os.path.join("state", "caveman-turns.raw.json")
DEFAULT_REWRITES = os.path.join("tools", "caveman", "rewrites.json")
DEFAULT_OUT = os.path.join("state", "caveman-savings.json")


def tildify(path: str) -> str:
    home = os.path.expanduser("~")
    return path.replace(home, "~") if home and path.startswith(home) else path


def tokenizer_label(snapshot: str) -> str:
    """``.../models--Qwen--Qwen3-1.7B/snapshots/<sha>`` -> ``Qwen/Qwen3-1.7B``."""
    for part in snapshot.split(os.sep):
        if part.startswith("models--"):
            bits = part.split("--")[1:]
            if len(bits) >= 2:
                return f"{bits[0]}/{bits[1]}"
    return "Qwen3"


def find_qwen3_tokenizer(explicit: str | None) -> str:
    """Return a local snapshot dir holding a Qwen3 tokenizer.json."""
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("CAVEMAN_TOKENIZER")
    if env:
        candidates.append(env)
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    for name in ("models--Qwen--Qwen3-1.7B", "models--Qwen--Qwen3-0.6B"):
        candidates += sorted(glob.glob(os.path.join(hub, name, "snapshots", "*")))
    candidates += sorted(glob.glob(os.path.join(hub, "models--Qwen--Qwen3*", "snapshots", "*")))
    for c in candidates:
        if os.path.isfile(os.path.join(c, "tokenizer.json")):
            return c
    raise SystemExit(
        "no local Qwen3 tokenizer found; set CAVEMAN_TOKENIZER or --tokenizer to a "
        "snapshot dir containing tokenizer.json"
    )


def assemble(turn: dict, prose_after: list[str]) -> str:
    """Reassemble the message: code segments verbatim, prose segments rewritten."""
    parts: list[str] = []
    pi = 0
    for seg in turn["segments"]:
        if seg["kind"] == "code":
            parts.append(seg["text"])
        else:
            if pi >= len(prose_after):
                raise ValueError(f"{turn['id']}: more text segments than rewrites")
            parts.append(prose_after[pi])
            pi += 1
    if pi != len(prose_after):
        raise ValueError(f"{turn['id']}: {pi} text segments vs {len(prose_after)} rewrites")
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--rewrites", default=DEFAULT_REWRITES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args(argv)

    if not os.path.isfile(args.raw):
        raise SystemExit(f"raw corpus missing: {args.raw} (run tools/caveman/extract_turns.py)")
    raw = json.load(open(args.raw, encoding="utf-8"))
    rew = json.load(open(args.rewrites, encoding="utf-8"))
    by_id = {r["id"]: r["prose_after"] for r in rew["rewrites"]}

    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit("transformers not importable; run with a venv that has it")

    tok_dir = find_qwen3_tokenizer(args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)

    def ntok(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    turns_out = []
    before_total = after_total = 0
    for turn in raw["turns"]:
        tid = turn["id"]
        if tid not in by_id:
            raise SystemExit(f"no rewrite for turn {tid}")
        after_text = assemble(turn, by_id[tid])
        b, a = ntok(turn["content"]), ntok(after_text)
        before_total += b
        after_total += a
        turns_out.append({"id": tid, "before": b, "after": a})

    n = len(turns_out)
    saving = round(100.0 * (before_total - after_total) / before_total, 2) if before_total else 0.0
    result = {
        "n_turns": n,
        "tokens_before": before_total,
        "tokens_after": after_total,
        "saving_pct": saving,
        "tokenizer": tokenizer_label(tok_dir),
        "tokenizer_path": tildify(tok_dir),
        "scope": "prose segments only; fenced code byte-exact; no reasoning tokens",
        "source": tildify(raw.get("traj_dir", "")),
        "turns": turns_out,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print(f"n_turns={n} before={before_total} after={after_total} saving={saving}% -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
