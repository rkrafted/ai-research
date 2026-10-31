"""Extraction prompts for the local batch: one video -> one JSON note.

Reads the transcripts (``data/private/bycloud/transcripts/*.json``) and writes
``local_batch/bycloud-prompts.jsonl`` (``{"id": video_id, "prompt": ...}``).
With ``--queue`` it also writes ``local_batch/bycloud-extract.yaml``, so the
runner serves the 27B into ``data/bycloud-extracts-raw.jsonl`` and the
``bycloud-assemble`` GPU job turns that into ``data/bycloud-extracts.jsonl``
(the verified result file; a session never writes it).

The prompt asks for a verbatim quote (>= 8 words) and segment start for every
concept, which is exactly what the grounded-report check re-verifies later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .ingest import REPO, TRANSCRIPTS

PROMPTS = REPO / "local_batch" / "bycloud-prompts.jsonl"
BATCH_YAML = REPO / "local_batch" / "bycloud-extract.yaml"
RAW_OUT = "data/bycloud-extracts-raw.jsonl"
DEFAULT_MODEL = "qwen3.8-27b-uncensored:latest"
MAX_TRANSCRIPT = 6000
CHUNK_CHARS = 500

INSTRUCTION = """You extract structured research notes from ONE YouTube video by the AI channel bycloud.
Return ONLY one compact JSON object. No prose, no markdown, no code fences. Schema:
{"concepts":[{"name":str,"quote":str,"start_s":number}],"claims":[{"claim":str,"numbers":str,"start_s":number|null}],"papers":[{"arxiv_id":str,"title":str}],"repos":[str]}
Rules:
- concepts: up to 8 techniques, methods or ideas the video actually discusses.
- For EVERY concept, "quote" MUST be a verbatim span of at least 8 consecutive words copied exactly from the transcript, and "start_s" is the whole-second start of the [MM:SS] line that quote is taken from. Never paraphrase a quote.
- claims: up to 5 concrete, checkable statements. "numbers" is the exact figure(s) as written, else "". "start_s" is the supporting line's start, else null.
- papers: every arXiv id present in the text, arxiv_id exactly like 2401.12345. Never invent an id.
- repos: code or data URLs actually cited (github.com/... or huggingface.co/...).
- Use [] or "" when absent. Never invent quotes, numbers, ids or urls that are not in the transcript.
"""


def _stamp(sec: float) -> str:
    s = int(sec)
    return f"[{s // 60:02d}:{s % 60:02d}]"


def chunk_transcript(segments: list[dict], chunk_chars: int = CHUNK_CHARS,
                     max_chars: int = MAX_TRANSCRIPT) -> str:
    """``[MM:SS] paragraph`` lines built from adjacent segments, capped at max_chars."""
    blocks: list[tuple[float, str]] = []
    cur: list[str] = []
    cur_len = 0
    start = float(segments[0].get("start") or 0) if segments else 0.0
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        if cur and cur_len + len(text) > chunk_chars:
            blocks.append((start, " ".join(cur)))
            cur, cur_len = [], 0
        if not cur:
            start = float(seg.get("start") or 0)
        cur.append(text)
        cur_len += len(text) + 1
    if cur:
        blocks.append((start, " ".join(cur)))

    lines: list[str] = []
    used = 0
    for st, text in blocks:
        line = f"{_stamp(st)} {text}"
        if used + len(line) > max_chars and lines:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def build_prompt(row: dict, doc: dict) -> str:
    meta = (f"channel: bycloud\n"
            f"title: {row.get('title')}\n"
            f"published: {row.get('upload_date')}  duration_s: {doc.get('duration_s')}  "
            f"views: {row.get('view_count')}\n"
            f"url: {row.get('url')}")
    body = [INSTRUCTION, "VIDEO", meta, "TRANSCRIPT",
            chunk_transcript(doc.get("segments") or [])]
    return "\n".join(body)


def iter_transcripts(limit: int = 0):
    rows = {v["id"]: v for v in _videos()}
    paths = sorted(TRANSCRIPTS.glob("*.json")) if TRANSCRIPTS.exists() else []
    for i, path in enumerate(paths):
        if limit and i >= limit:
            break
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        vid = doc.get("video_id") or path.stem
        row = rows.get(vid) or {"id": vid, "title": "", "url": f"https://www.youtube.com/watch?v={vid}"}
        yield row, doc


def _videos() -> list[dict]:
    from .ingest import load_videos
    return load_videos()


def write_prompts(limit: int = 0, path: Path = PROMPTS) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row, doc in iter_transcripts(limit):
            fh.write(json.dumps({"id": row["id"], "prompt": build_prompt(row, doc)},
                                ensure_ascii=False) + "\n")
            n += 1
    return n


def write_yaml(path: Path = BATCH_YAML) -> None:
    path.write_text(
        "# bycloud extraction: one prompt per transcript, served to the local 27B.\n"
        f"# Answers land in {RAW_OUT}; the bycloud-assemble GPU job writes\n"
        "# data/bycloud-extracts.jsonl (the verified result file).\n"
        f"model: {DEFAULT_MODEL}\n"
        f"prompts: {PROMPTS.relative_to(REPO)}\n"
        f"out: {RAW_OUT}\n"
        "options:\n"
        "  temperature: 0.0\n"
        "  num_ctx: 8192\n"
        "  format: json\n",
        encoding="utf-8",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bycloud.extract")
    ap.add_argument("--queue", action="store_true",
                    help="also write local_batch/bycloud-extract.yaml")
    ap.add_argument("--limit", type=int, default=0, help="max prompts (0 = all)")
    args = ap.parse_args(argv)
    n = write_prompts(args.limit)
    if not n:
        print("no transcripts; run `python3 -m bycloud captions` first")
        return 1
    if args.queue:
        write_yaml()
    print(f"prompts: {n} -> {PROMPTS.relative_to(REPO)}"
          + (f" and {BATCH_YAML.relative_to(REPO)}" if args.queue else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
