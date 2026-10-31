#!/usr/bin/env python3
"""Extract real assistant prose turns from saved orchestrator transcripts.

Read-only over the transcript directory. This tool only *collects* candidates and
splits each turn into prose vs. payload (fenced code) segments; the caveman rewrite
itself is done by DeepSeek and stored in ``tools/caveman/rewrites.jsonl``.

Usage::

    python3 tools/caveman/extract_turns.py [--limit 40] [--min-chars 120] [--out FILE]

The default transcript dir is ``~/theprojects/orchestrator/state/trajectories``
(override with ``CAVEMAN_TRAJ_DIR``). The raw output is written under ``state/``
which is gitignored, so transcript text never lands in a commit.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

DEFAULT_TRAJ = os.path.expanduser("~/theprojects/orchestrator/state/trajectories")
FENCE = re.compile(r"```.*?```", re.S)


def segments(text: str) -> list[dict]:
    """Split a message into {'kind': 'text'|'code', 'text': ...} segments.

    Fenced code blocks (```...```) are payload and must stay byte-exact.
    """
    out: list[dict] = []
    pos = 0
    for m in FENCE.finditer(text):
        if m.start() > pos:
            out.append({"kind": "text", "text": text[pos:m.start()]})
        out.append({"kind": "code", "text": m.group(0)})
        pos = m.end()
    if pos < len(text):
        out.append({"kind": "text", "text": text[pos:]})
    return out


def prose_only(text: str) -> str:
    return FENCE.sub(" ", text).strip()


def load_candidates(traj_dir: str, min_chars: int, max_chars: int) -> list[dict]:
    """Return candidate turns, spread round-robin over sessions for diversity.

    Structured payloads (a whole assistant message that is JSON) are skipped:
    they must stay byte-exact, so they are not prose to rewrite.
    """
    per_file: dict[str, list[dict]] = {}
    files = sorted(glob.glob(os.path.join(traj_dir, "*.jsonl")))
    for path in files:
        session = os.path.basename(path)[:-6]
        try:
            fh = open(path, encoding="utf-8")
        except OSError:
            continue
        with fh:
            for idx, line in enumerate(fh):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict) or obj.get("type") == "meta":
                    continue
                if obj.get("role") != "assistant":
                    continue
                content = obj.get("content")
                if not isinstance(content, str):
                    continue
                n = len(prose_only(content))
                if n < min_chars or n > max_chars:
                    continue
                stripped = content.strip()
                if stripped[:1] in "{[":
                    try:
                        json.loads(stripped)
                        continue  # structured output, not prose
                    except json.JSONDecodeError:
                        pass
                per_file.setdefault(session, []).append(
                    {"id": f"{session}:{idx}", "session": session, "line": idx,
                     "content": content, "segments": segments(content)}
                )

    picked: list[dict] = []
    order = sorted(per_file)
    round_no = 0
    while True:
        added = False
        for session in order:
            bucket = per_file[session]
            if round_no < len(bucket):
                picked.append(bucket[round_no])
                added = True
        if not added:
            break
        round_no += 1
    return picked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj-dir", default=os.environ.get("CAVEMAN_TRAJ_DIR", DEFAULT_TRAJ))
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--max-chars", type=int, default=1100)
    ap.add_argument("--out", default=os.path.join("state", "caveman-turns.raw.json"))
    args = ap.parse_args(argv)

    if not os.path.isdir(args.traj_dir):
        print(f"transcript dir not found: {args.traj_dir}", file=sys.stderr)
        return 1
    picked = load_candidates(args.traj_dir, args.min_chars, args.max_chars)[: args.limit]
    if not picked:
        print("no candidate turns found", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"traj_dir": args.traj_dir, "n": len(picked), "turns": picked},
                  fh, ensure_ascii=False, indent=1)
    code = sum(1 for t in picked if any(s["kind"] == "code" for s in t["segments"]))
    print(f"wrote {len(picked)} turns to {args.out} ({code} with fenced code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
