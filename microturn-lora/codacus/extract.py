"""Build the Codacus local-batch extraction prompts.

One prompt per transcript in ``data/codacus/``. The 9B answers with the
mechanism, claimed numbers, links, target hardware and the goal(s) it touches;
``codacus.assemble`` then splits the answers into ``data/codacus/<id>.json``.

The local batch is the house tool: the runner serves these prompts to Ollama
after the session and writes ``data/codacus/extracts.jsonl``. Nothing here runs
a model.

Usage:
    python -m codacus.extract            # prompts + batch yaml
    python -m codacus.extract --check    # report only, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from airesearch.ingest import load_info

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "codacus"
PROMPTS = REPO / "local_batch" / "codacus-prompts.jsonl"
BATCH = REPO / "local_batch" / "codacus-extract.yaml"
EXTRACTS = "data/codacus/extracts.jsonl"
MODEL = "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"

GOALS = ("speed", "context", "model-size-on-24GB", "hallucination",
         "forgetting/memory", "other")

MAX_TRANSCRIPT = 5200
MAX_DESC = 900
MAX_PROMPT = 7000

INSTRUCTION = """You reverse-engineer ONE AI video by the Codacus YouTube channel.
Return ONLY one JSON object, no prose, no markdown fences. Schema:
{"system": str, "technique": str, "mechanism": [str], "claims": [{"claim": str, "numbers": str}], "links": [str], "hardware": [str], "goals": [str], "buildable": str}
Rules:
- system: the project/model/engine the video is about, as named in the text (not "AI").
- technique: the one named method that makes it work (e.g. "MoE offload", "Q4_K_M quantization", "RPC pipeline split").
- mechanism: 3 to 6 short steps that say HOW it actually works. Name the data structures, algorithms, memory layout, kernels or protocol where the text does. No marketing verbs, no "revolutionary".
- claims: up to 6 concrete statements. "numbers" is the exact figure(s) stated (speed, VRAM, context, size), else "". A number the video only asserts is still a claim; do not add numbers that are not in the text.
- links: URLs actually present in the text, ideally github.com or huggingface.co or arxiv.org. Never invent a URL.
- hardware: devices and budgets named (e.g. "RTX 3090", "24 GB VRAM", "M1 MacBook", "CPU only").
- goals: 1 or more of exactly: speed, context, model-size-on-24GB, hallucination, forgetting/memory, other. Choose the goal(s) the mechanism actually moves.
- buildable: one sentence: what part of this could be reused in a local llama.cpp/Ollama stack?
- Use [] or "" when absent. Never invent numbers, ids, URLs or hardware.
"""

_HOME = re.compile("/" + "home" + "/")


def _clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def build_prompt(row: dict, transcript: str) -> str:
    meta = (f"channel: {row.get('channel', 'Codacus')}\n"
            f"title: {row.get('title')}\n"
            f"published: {row.get('upload_date')}  duration_s: {row.get('duration')}\n"
            f"url: {row.get('url')}")
    body = [INSTRUCTION, "VIDEO", meta]
    desc = _clip(row.get("description") or "", MAX_DESC)
    if desc:
        body += ["DESCRIPTION", desc]
    body += ["TRANSCRIPT", _clip(transcript, MAX_TRANSCRIPT)]
    return _HOME.sub("~/", "\n".join(body))[:MAX_PROMPT]


def rows_with_transcripts() -> list[dict]:
    videos = json.loads((OUT / "videos.json").read_text(encoding="utf-8"))
    rows = []
    for row in videos:
        path = OUT / f"{row['id']}.txt"
        if path.exists() and path.stat().st_size > 200:
            info = load_info(row["id"]) or {}
            rows.append({**row,
                         "description": info.get("description") or "",
                         "transcript": path.read_text(encoding="utf-8")})
    return rows


def write_prompts(rows: list[dict]) -> int:
    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    with PROMPTS.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({"id": row["id"],
                                 "prompt": build_prompt(row, row["transcript"])},
                                ensure_ascii=False) + "\n")
    return len(rows)


def write_yaml() -> None:
    BATCH.write_text(
        "# Codacus extraction: one prompt per video, served to the local 9B.\n"
        f"# Answers land in {EXTRACTS}; `python3 -m codacus.assemble` splits them into\n"
        "# data/codacus/<id>.json (the committed deliverable).\n"
        f"model: {MODEL}\n"
        f"prompts: {PROMPTS.relative_to(REPO)}\n"
        f"out: {EXTRACTS}\n"
        "options:\n"
        "  temperature: 0.0\n"
        "  num_ctx: 8192\n"
        "  format: json\n",
        encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="codacus.extract")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    rows = rows_with_transcripts()
    if len(rows) < 25:
        print(f"only {len(rows)} transcripts in data/codacus; need >=25")
        return 1
    if args.check:
        sizes = [len(build_prompt(r, r["transcript"])) for r in rows]
        print(f"{len(rows)} prompts, max {max(sizes)} chars")
        return 0
    n = write_prompts(rows)
    write_yaml()
    print(f"prompts: {n} -> {PROMPTS.relative_to(REPO)}")
    print(f"queued: {BATCH.relative_to(REPO)} -> {EXTRACTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
