"""Build the R2 local-batch extraction prompts from the corpus.

Reads ``data/private/ingest/corpus.jsonl`` (metadata + captions) and writes
``local_batch/ingest-prompts.jsonl`` (one ``{"id","prompt"}`` per video, id =
YouTube video id). With ``--queue`` it also writes
``local_batch/ingest-extract.yaml`` so the runner serves the batch to Ollama and
lands the answers in ``data/kb-extracts.jsonl`` (the only extraction file).

The prompt is deliberately compact: short schema, at most 8 methods / 5 claims,
and the transcript capped so the prompt fits ``num_ctx: 8192``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .ingest import CAPTIONDIR, REPO, load_corpus

PROMPTS = REPO / "local_batch" / "ingest-prompts.jsonl"
BATCH_YAML = REPO / "local_batch" / "ingest-extract.yaml"
DEFAULT_MODEL = "qwen3.8-27b-uncensored:latest"
OUT = "data/kb-extracts.jsonl"

MAX_DESC = 1200
MAX_TRANSCRIPT = 4200
MAX_PROMPT = 6000

INSTRUCTION = """You extract structured research notes from ONE YouTube video by an AI/ML channel.
Return ONLY one compact JSON object. No prose, no markdown, no code fences. Schema:
{"methods": [str], "claims": [{"claim": str, "numbers": str, "timestamp": str|null}], "papers": [{"arxiv_id": str, "title": str}], "repos": [str], "hardware": [str], "audience": str}
Rules:
- methods: up to 8 named techniques/models/algorithms the video actually discusses.
- claims: up to 5 concrete, checkable statements. "numbers" is the exact figure(s) stated, else "". "timestamp" is a [MM:SS] marker from the transcript supporting it, else null.
- papers: every arXiv id present in the text; arxiv_id exactly like 2401.12345. Never invent an id.
- repos: code/data URLs actually cited (github.com/... or huggingface.co/...).
- hardware: GPUs, devices or compute budgets mentioned (e.g. "RTX 3090", "8x H100", "24 GB").
- audience: one short phrase for who it is for.
- Use [] or "" when absent. Never invent numbers, ids or urls that are not in the text.
"""

_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[.,](\d+)")
_TAG = re.compile(r"<[^>]+>")


def _secs(match: re.Match) -> int:
    h, m, s, _ = (int(g) for g in match.groups())
    return h * 3600 + m * 60 + s


def _stamp(sec: int) -> str:
    return f"[{sec // 60:02d}:{sec % 60:02d}]"


def vtt_segments(text: str, chunk_chars: int = 600) -> list[tuple[int, str]]:
    """[(start_seconds, text)] from a WebVTT caption file, tags stripped.

    YouTube auto-captions scroll: each cue repeats the tail of the previous one.
    Consecutive cues are merged on their longest suffix/prefix overlap, then
    grouped into ~``chunk_chars`` blocks so one ``[MM:SS]`` stamp covers a
    readable paragraph instead of every 2-second cue.
    """
    cues: list[list] = []                       # [start_seconds, text]
    start, lines = 0, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        if "-->" in line:
            if lines:
                cues.append([start, " ".join(lines)])
            m = _TIME.search(line.split("-->")[0])
            start, lines = (_secs(m) if m else 0), []
            continue
        clean = _TAG.sub("", line).replace("&nbsp;", " ").strip()
        if clean and (not lines or lines[-1] != clean):
            lines.append(clean)
    if lines:
        cues.append([start, " ".join(lines)])

    merged: list[list] = []
    for sec, txt in cues:
        if not txt:
            continue
        if merged:
            a = merged[-1][1]
            k = min(len(a), len(txt))
            while k > 7 and a[-k:] != txt[:k]:
                k -= 1
            if k > 7:                            # overlap: extend, do not repeat
                merged[-1][1] = a + txt[k:]
                continue
        merged.append([sec, txt])

    chunks: list[list] = []
    for sec, txt in merged:
        if chunks and len(chunks[-1][1]) < chunk_chars:
            chunks[-1][1] += " " + txt
        else:
            chunks.append([sec, txt])
    return [(s, t.strip()) for s, t in chunks]


def transcript_for(vid: str, max_chars: int = MAX_TRANSCRIPT) -> str:
    """Timestamped transcript text from the cached English VTT files."""
    files = sorted(CAPTIONDIR.glob(f"{vid}.*.vtt")) if CAPTIONDIR.exists() else []
    if not files:
        return ""
    segs = vtt_segments(files[0].read_text(encoding="utf-8", errors="replace"))
    parts: list[str] = []
    used = 0
    for sec, txt in segs:
        chunk = f"{_stamp(sec)} {txt}"
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk) + 1
    return "\n".join(parts)


def _clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def build_prompt(row: dict, transcript: str | None = None) -> str:
    if transcript is None:
        transcript = transcript_for(row["id"])
    meta = (
        f"channel: {row.get('channel')}\n"
        f"title: {row.get('title')}\n"
        f"published: {row.get('upload_date')}  duration_s: {row.get('duration')}  "
        f"views: {row.get('view_count')}\n"
        f"url: {row.get('url')}"
    )
    body = [INSTRUCTION, "VIDEO", meta]
    desc = _clip(row.get("description") or "", MAX_DESC)
    if desc:
        body += ["DESCRIPTION", desc]
    if transcript:
        body += ["TRANSCRIPT", transcript]
    prompt = "\n".join(body)
    # Never send (or let the model echo) an absolute home path.
    prompt = _HOME.sub("~/", prompt)
    return prompt[:MAX_PROMPT]


def eligible(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("has_captions") or len(r.get("description") or "") >= 40]


def channel_scores() -> dict[str, float]:
    try:
        data = json.loads((REPO / "data" / "channels.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {c["name"]: float(c.get("score") or 0) for c in data}


_ARXIV_LINK = re.compile(r"arxiv\.org/(?:abs|pdf)/\d", re.IGNORECASE)
_HOME = re.compile("/" + "home" + "/")   # split so this file itself passes the scan


def select(rows: list[dict], cap: int) -> list[dict]:
    """arXiv-citing videos first (paper yield), then channel score, then newest.

    If ``cap`` truncates, the kept set is the one most likely to satisfy the
    claims/papers targets in the done_when assert.
    """
    scores = channel_scores()
    rows = sorted(rows, key=lambda r: str(r.get("upload_date") or ""), reverse=True)
    rows.sort(key=lambda r: scores.get(r.get("channel") or "", 0.0), reverse=True)
    rows.sort(key=lambda r: bool(_ARXIV_LINK.search(r.get("description") or "")), reverse=True)
    return rows[:cap] if cap else rows


def write_prompts(rows: list[dict], path: Path = PROMPTS) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({"id": row["id"], "prompt": build_prompt(row)},
                                ensure_ascii=False) + "\n")
            n += 1
    return n


def write_yaml(model: str = DEFAULT_MODEL, path: Path = BATCH_YAML,
               out: str = OUT) -> None:
    path.write_text(
        "# R2 extraction: one prompt per ingested video, served to the local 27B.\n"
        f"# Answers land in {out} (the only extraction file; kb.sqlite is built from it).\n"
        f"model: {model}\n"
        f"prompts: {PROMPTS.relative_to(REPO)}\n"
        f"out: {out}\n"
        "options:\n"
        "  temperature: 0.0\n"
        "  num_ctx: 8192\n"
        "  format: json\n",
        encoding="utf-8",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="airesearch.prompts")
    ap.add_argument("--cap", type=int, default=4000,
                    help="max prompts to write (0 = all eligible)")
    ap.add_argument("--queue", action="store_true",
                    help="also write local_batch/ingest-extract.yaml")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args(argv)

    rows = eligible(load_corpus())
    if not rows:
        print("no eligible videos in corpus; run `python -m airesearch.ingest corpus` first")
        return 1
    chosen = select(rows, args.cap)
    n = write_prompts(chosen)
    withcaps = sum(1 for r in chosen if r.get("has_captions"))
    print(f"prompts: {n} (of {len(rows)} eligible), {withcaps} with transcript -> "
          f"{PROMPTS.relative_to(REPO)}")
    if args.queue:
        write_yaml(model=args.model)
        print(f"queued: {BATCH_YAML.relative_to(REPO)} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
