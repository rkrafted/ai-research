#!/usr/bin/env python3
"""Runner GPU job: turn the local batch's raw answers into the verified result file.

The local batch writes ``data/bycloud-extracts-raw.jsonl`` rows as
``{"id","response","model","tokens","seconds"}``; the queue spec's done check needs
``video_id`` plus the parsed concepts/claims/papers/repos. This job parses each
response, re-grounds every concept quote against the transcript (>= 8 words,
normalised, inside +/-90 s of the cited start), and writes
``data/bycloud-extracts.jsonl``. It is a runner job on purpose: a session must
never write a results file.

Usage (the runner does this): python3 bycloud/assemble_extract.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "bycloud-extracts-raw.jsonl"
OUT = REPO / "data" / "bycloud-extracts.jsonl"
PROMPTS = REPO / "local_batch" / "bycloud-prompts.jsonl"
TRANSCRIPTS = REPO / "data" / "private" / "bycloud" / "transcripts"

_WORD = re.compile(r"[a-z0-9]+")
_CLOCK = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})$")


def norm(s: str) -> str:
    return " ".join(_WORD.findall((s or "").lower()))


def as_seconds(value) -> float | None:
    """Accept 123, 123.4, "123", "12:34" and "1:02:03"; else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        m = _CLOCK.match(s)
        if m:
            h, mi, sec = (int(g) if g else 0 for g in m.groups())
            return float(h * 3600 + mi * 60 + sec)
        try:
            return float(s)
        except ValueError:
            return None
    return None


def parse_response(text: str) -> dict:
    """The model's JSON object, tolerating fences or a wrapper around it."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            obj = json.loads(s[i:j + 1])
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def grounded(quote: str, start_s, segments: list[dict]) -> bool:
    q = norm(quote)
    if len(q.split()) < 8:
        return False
    t = as_seconds(start_s)
    if t is None:
        return False
    window = norm(" ".join(str(s.get("text") or "") for s in segments
                           if abs(float(s.get("start") or 0) - t) <= 90))
    return q in window


def load_segments(vid: str) -> list[dict]:
    p = TRANSCRIPTS / f"{vid}.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("segments") or []
    except (OSError, json.JSONDecodeError):
        return []


def as_list(value) -> list:
    return value if isinstance(value, list) else []


def assemble_row(row: dict) -> dict:
    vid = str(row.get("id") or "")
    obj = parse_response(row.get("response") or "")
    segs = load_segments(vid)
    concepts = []
    for c in as_list(obj.get("concepts")):
        if not isinstance(c, dict) or not c.get("name"):
            continue
        quote = str(c.get("quote") or "")
        start = as_seconds(c.get("start_s"))
        concepts.append({"name": str(c["name"]), "quote": quote, "start_s": start,
                         "grounded": grounded(quote, start, segs)})
    claims = [{"claim": str(c.get("claim")), "numbers": str(c.get("numbers") or ""),
               "start_s": as_seconds(c.get("start_s"))}
              for c in as_list(obj.get("claims")) if isinstance(c, dict) and c.get("claim")]
    papers = [{"arxiv_id": str(p.get("arxiv_id")), "title": str(p.get("title") or "")}
              for p in as_list(obj.get("papers")) if isinstance(p, dict) and p.get("arxiv_id")]
    repos = [str(r) for r in as_list(obj.get("repos")) if r]
    return {"video_id": vid, "parse_ok": bool(obj), "concepts": concepts,
            "claims": claims, "papers": papers, "repos": repos}


def _rel(p: Path) -> str:
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


def main() -> int:
    if not RAW.exists():
        print(f"no {_rel(RAW)}; the local batch has not run")
        return 1
    rows = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    # A GPU job once ran while the local batch was still writing this file and produced a
    # 9-video result (2026-10-16). Refuse to write anything until every prompt has an answer.
    if PROMPTS.exists():
        want = sum(1 for l in PROMPTS.read_text(encoding="utf-8").splitlines() if l.strip())
        if len(rows) < want:
            print(f"incomplete: {len(rows)}/{want} raw rows; the local batch is not done")
            return 1
    out = [assemble_row(r) for r in rows]
    n_grounded = sum(1 for r in out for c in r["concepts"] if c["grounded"])
    n_concepts = sum(len(r["concepts"]) for r in out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"assemble: {len(out)} videos, {n_concepts} concepts "
          f"({n_grounded} grounded) -> {_rel(OUT)}")
    return 0 if out else 1


if __name__ == "__main__":
    raise SystemExit(main())
