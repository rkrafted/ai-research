"""Split the local batch answers into one JSON per video.

Input:  ``data/codacus/extracts.jsonl`` (runner-written local batch output,
        ``{id, response, model, tokens, seconds}``).
Output: ``data/codacus/<id>.json`` (the committed deliverable) plus a short
        report on stderr. Refuses to write if fewer than 25 answers parse, so a
        half-finished batch never replaces a good set.

Usage:
    python3 -m codacus.assemble
    python3 -m codacus.assemble --raw data/codacus/extracts.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from airesearch.ingest import load_info

from .extract import GOALS, OUT

RAW = OUT / "extracts.jsonl"
VIDEOS = OUT / "videos.json"
URL = re.compile(r"https?://[^\s)\]\"'<>]+")


def parse_answer(text: str):
    """First JSON object in a model answer (tolerates fences and prose)."""
    if not isinstance(text, str):
        return None
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _strs(value, cap=12) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in out:
            out.append(text[:400])
    return out[:cap]


def _claims(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value[:8]:
        if isinstance(item, str):
            claim, numbers = item, ""
        elif isinstance(item, dict):
            claim = str(item.get("claim") or item.get("text") or "").strip()
            numbers = str(item.get("numbers") or "").strip()
        else:
            continue
        if not claim:
            continue
        out.append({"claim": claim[:500], "numbers": numbers[:120], "claimed": True})
    return out


def clean(obj: dict) -> dict:
    goals = [g for g in _strs(obj.get("goals"), cap=6) if g in GOALS]
    return {
        "system": str(obj.get("system") or "").strip()[:300],
        "technique": str(obj.get("technique") or "").strip()[:300],
        "mechanism": _strs(obj.get("mechanism"), cap=6),
        "claims": _claims(obj.get("claims")),
        "links": [u for u in _strs(obj.get("links")) if u.startswith("http")][:20],
        "hardware": _strs(obj.get("hardware"), cap=8),
        "goals": goals or ["other"],
        "buildable": str(obj.get("buildable") or "").strip()[:600],
    }


def urls_in(text: str) -> list[str]:
    """URLs in a transcript or description, in source order, no placeholders."""
    out = []
    for raw in URL.findall(text or ""):
        url = raw.rstrip(".,;:")
        if "..." in url or len(url) < 12 or url.rstrip("/").endswith("/abs"):
            continue
        if url not in out:
            out.append(url)
    return out


def ground_links(links, transcript: str = "", description: str = "",
                 video_url: str = "") -> list[str]:
    """Keep only links the video actually carries.

    The 9B invents plausible repo URLs. A link survives when it appears in the
    transcript or description, plus the video's own URL. Real project links are
    then added from the description (intro/outro links live there, not in the
    captions). Never invents one.
    """
    allowed = urls_in(transcript) + urls_in(description)
    if video_url:
        allowed.append(video_url)
    allowed = list(dict.fromkeys(allowed))
    out = []
    for url in list(links or []) + allowed:
        url = str(url or "").strip().rstrip(".,;:")
        if url in allowed and url not in out:
            out.append(url)
    return out[:20]


def load_answers(raw: Path) -> dict[str, dict]:
    answers = {}
    for line in raw.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        vid = row.get("id")
        parsed = parse_answer(row.get("response") or "")
        if vid and parsed:
            answers[vid] = clean(parsed)
    return answers


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="codacus.assemble")
    ap.add_argument("--raw", type=Path, default=RAW)
    args = ap.parse_args(argv)
    if not args.raw.exists():
        print(f"missing {args.raw}; the local batch has not run yet")
        return 1
    videos = {v["id"]: v for v in json.loads(VIDEOS.read_text(encoding="utf-8"))}
    answers = load_answers(args.raw)
    valid = {vid: a for vid, a in answers.items() if vid in videos and a["mechanism"]}
    if len(valid) < 25:
        print(f"only {len(valid)} valid answers (need >=25); not writing")
        return 1
    for vid, answer in valid.items():
        v = videos[vid]
        transcript_path = OUT / f"{vid}.txt"
        transcript = (transcript_path.read_text(encoding="utf-8")
                      if transcript_path.exists() else "")
        description = (load_info(vid) or {}).get("description") or ""
        row = {
            "id": vid,
            "title": v.get("title"),
            "date": v.get("upload_date"),
            "duration": v.get("duration"),
            "url": v.get("url"),
            **answer,
            "links": ground_links(answer["links"], transcript, description,
                                  v.get("url") or ""),
        }
        (OUT / f"{vid}.json").write_text(
            json.dumps(row, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    missing = sorted(videos[vid]["id"] for vid in videos if vid not in valid)
    print(f"wrote {len(valid)}/{len(videos)} video JSONs -> data/codacus/")
    if missing:
        print("no valid answer: " + " ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
