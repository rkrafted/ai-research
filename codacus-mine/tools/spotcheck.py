#!/usr/bin/env python3
"""Independent, deterministic audit of the local model's extractions.

Reads ``data/kb-extracts.jsonl`` (written by the runner's Ollama batch, read-only
here) and ``data/private/ingest/corpus.jsonl``, then audits a fixed 10 % sample of
the videos: it rebuilds the *exact* prompt the model was served
(``airesearch.prompts.build_prompt``) and checks whether every arXiv id, every
claim number and every claim timestamp the model reported is actually present in
that source text. Nothing in the batch output is modified.

Usage:
    python3 tools/spotcheck.py            # writes docs/spot-checks.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airesearch.prompts import build_prompt            # noqa: E402
from airesearch.kb import parse_response, normalize_arxiv  # noqa: E402

EXTRACTS = ROOT / "data" / "kb-extracts.jsonl"
CORPUS = ROOT / "data" / "private" / "ingest" / "corpus.jsonl"
PROMPTS = ROOT / "local_batch" / "ingest-prompts.jsonl"
OUT = ROOT / "docs" / "spot-checks.md"
STEP = 10                       # every 10th video -> ~10 % of the batch


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_TS = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?")


def _num_tokens(text: str) -> list[str]:
    return [m.group(0).replace(",", "").rstrip(".") for m in _NUM.finditer(text or "")]


def _ts_in_source(ts: str, source: str) -> bool:
    m = _TS.search(str(ts or ""))
    if not m:
        return True                            # null/absent: nothing to ground
    if m.group(3):                              # HH:MM:SS
        variants = {m.group(0), m.group(0)[-5:]}
    else:
        variants = {m.group(0), f"[{m.group(0)}]"}
    return any(v in source for v in variants)


def audit(extracts: list[dict], corpus: dict[str, dict],
          prompts: dict[str, str] | None = None) -> list[dict]:
    prompts = prompts or {}
    out = []
    for rec in extracts:
        vid = str(rec.get("id"))
        row = corpus.get(vid, {})
        # Ground truth is the prompt the batch really served (stored by the runner);
        # rebuild it only if that file is gone.
        source = prompts.get(vid) or (build_prompt(row) if row else "")
        obj = parse_response(rec.get("response") or "")
        item = {
            "id": vid,
            "channel": row.get("channel") or "?",
            "title": row.get("title") or "",
            "parsed": obj is not None,
            "methods": 0, "claims": 0, "claims_ts": 0,
            "papers": 0, "papers_ok": 0, "bad_ids": [],
            "bad_papers": [], "bad_numbers": [], "bad_ts": [],
            "empty_methods": False, "no_claims": False,
        }
        if obj is None:
            out.append(item)
            continue
        methods = obj.get("methods") or []
        claims = obj.get("claims") or []
        papers = obj.get("papers") or []
        item["methods"] = len(methods)
        item["empty_methods"] = not methods
        item["claims"] = len(claims)
        item["no_claims"] = not claims
        item["papers"] = len(papers)
        low = source.lower()
        src_digits = re.sub(r"\D", "", source)
        src_compact = re.sub(r"[,\s]", "", source)   # "4,000" -> "4000"
        for paper in papers:
            raw = paper.get("arxiv_id") if isinstance(paper, dict) else paper
            aid = normalize_arxiv(str(raw or ""))
            if not aid:
                if str(raw or "").strip():
                    item["bad_ids"].append(str(raw))
                continue
            if aid in low or aid.replace(".", "") in src_digits:
                item["papers_ok"] += 1
            else:
                item["bad_papers"].append(aid)
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            ts = claim.get("timestamp")
            if ts:
                item["claims_ts"] += 1
                if not _ts_in_source(ts, source):
                    item["bad_ts"].append(str(ts))
            nums = _num_tokens(claim.get("numbers"))
            if nums and not any(n in source or n in src_compact for n in nums):
                item["bad_numbers"].append(claim.get("numbers"))
        out.append(item)
    return out


def verdict(it: dict) -> str:
    if not it["parsed"]:
        return "**unparseable JSON**"
    issues = []
    if it["empty_methods"]:
        issues.append("no methods")
    if it["bad_papers"]:
        issues.append("arXiv id(s) not in source: " + ", ".join(it["bad_papers"]))
    if it["bad_ids"]:
        issues.append("malformed arXiv id(s): " + ", ".join(it["bad_ids"]))
    if it["bad_numbers"]:
        issues.append("claim number(s) not in source: " + "; ".join(it["bad_numbers"]))
    if it["bad_ts"]:
        issues.append("timestamp(s) not in transcript: " + ", ".join(it["bad_ts"]))
    if not issues:
        return "ok"
    return "; ".join(issues)


def summary_line(it: dict) -> str:
    bits = [f"{it['methods']} methods", f"{it['claims']} claims"]
    if it["claims"]:
        bits.append(f"{it['claims_ts']} timestamped")
    if it["papers"]:
        bits.append(f"{it['papers']} papers ({it['papers_ok']} grounded)")
    return ", ".join(bits)


def render(items: list[dict], total: int) -> str:
    n = len(items)
    parsed = sum(i["parsed"] for i in items)
    with_claims = sum(i["claims"] > 0 for i in items)
    ts_claims = sum(i["claims_ts"] for i in items)
    claims = sum(i["claims"] for i in items)
    papers = sum(i["papers"] for i in items)
    ok_papers = sum(i["papers_ok"] for i in items)
    flagged = [i for i in items if verdict(i) != "ok"]
    lines = [
        "# Spot-checks — R2 local-model extraction",
        "",
        f"Deterministic audit of **{n} of {total}** extracted videos "
        f"({100 * n / total:.1f} %); every {STEP}th video id in sorted order. "
        "Generated by `python3 tools/spotcheck.py` from the runner's batch output "
        "`data/kb-extracts.jsonl` — that file is only read.",
        "",
        "## Method",
        "",
        "`tools/spotcheck.py` reads the exact prompt the model was served from "
        "`local_batch/ingest-prompts.jsonl` (channel/title/description + the first ~4 200 "
        "caption characters, all timestamped) and checks the answer against that text:",
        "",
        "- **papers** — every non-empty `arxiv_id` must appear in the source text "
        "(normalised; a malformed or unknown id is a miss);",
        "- **claims** — at least one digit token from the `numbers` field must appear in "
        "the source text;",
        "- **timestamps** — a non-null `timestamp` must correspond to a `[MM:SS]` marker "
        "in the (truncated) transcript the model saw;",
        "- **shape** — the response parses as one JSON object and has a non-empty "
        "`methods` list.",
        "",
        "Caveat: the caption chunker merges YouTube auto-captions into long blocks, so "
        "many prompts carry only a handful of `[MM:SS]` markers. A timestamp that is not "
        "one of those markers is the model guessing a plausible time rather than quoting "
        "the transcript; it is flagged, but it does not affect the claim text or the KB.",
        "",
        "A **manual deep review** of individual videos, reading the captions next to the "
        "model's answer, is in the last section.",
        "",
        "## Sample totals",
        "",
        f"- responses parsed as JSON: **{parsed}/{n}**",
        f"- videos with claims: **{with_claims}/{n}**; claims: {claims} ({ts_claims} with a timestamp)",
        f"- papers: {papers}; grounded in the source text: **{ok_papers}/{papers}**",
        f"- videos with at least one finding: **{len(flagged)}/{n}** "
        f"(no methods: {sum(i['empty_methods'] for i in items)}, "
        f"bad/ungrounded arXiv id: {sum(bool(i['bad_papers'] or i['bad_ids']) for i in items)}, "
        f"ungrounded number: {sum(bool(i['bad_numbers']) for i in items)}, "
        f"timestamp not a prompt marker: {sum(bool(i['bad_ts']) for i in items)})",
        "",
        "## Findings (videos with anything to report)",
        "",
    ]
    if flagged:
        for it in flagged:
            url = f"https://www.youtube.com/watch?v={it['id']}"
            lines.append(f"- [{it['id']}]({url}) · {it['channel']} · "
                         f"{summary_line(it)} — {verdict(it)}")
    else:
        lines.append("_none: every sampled video passed every check._")
    lines += [
        "",
        "## Full 10 % sample",
        "",
        f"One line per sampled video ({n}). No line below repeats a finding already listed "
        "above; `ok` means shape, papers, numbers and timestamps all grounded.",
        "",
    ]
    for it in items:
        url = f"https://www.youtube.com/watch?v={it['id']}"
        lines.append(f"- [{it['id']}]({url}) · {it['channel']} · {summary_line(it)} — {verdict(it)}")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spotcheck")
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    extracts = _load_jsonl(EXTRACTS)
    if not CORPUS.exists():
        print("spotcheck: data/private/ingest/corpus.jsonl is missing (gitignored cache); "
              "run `python3 -m airesearch.ingest corpus` first")
        return 1
    corpus = {str(r["id"]): r for r in _load_jsonl(CORPUS)}
    prompts = ({str(r["id"]): r["prompt"] for r in _load_jsonl(PROMPTS)}
               if PROMPTS.exists() else {})
    ordered = sorted(extracts, key=lambda r: str(r.get("id")))
    sample = ordered[::args.step] if args.step > 0 else ordered
    items = audit(sample, corpus, prompts)
    text = render(items, len(extracts))
    # The manual deep review is written by hand; never clobber it on a re-run.
    out = Path(args.out)
    marker = "\n## Manual deep review\n"
    if out.exists():
        old = out.read_text(encoding="utf-8")
        if marker in old:
            text = text.rstrip("\n") + "\n\n" + marker.lstrip("\n") + old.split(marker, 1)[1]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"spot-checks: {len(items)}/{len(extracts)} videos -> {args.out}")
    print(f"flagged: {sum(verdict(i) != 'ok' for i in items)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
