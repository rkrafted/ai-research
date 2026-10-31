"""Audit the local 9B's per-video extraction against the captions.

Read-only: at least one numeric token in every claim must occur in the transcript the model
saw, every link must occur in the transcript, the description or be the video's
own URL, and the mechanism must have 3-6 steps. Writes a markdown report;
text after the ``<!-- manual -->`` marker is kept across runs.

Usage:
    python3 -m codacus.spotcheck
    python3 -m codacus.spotcheck --out docs/spot-checks-codacus.md
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

from airesearch.ingest import load_info

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "codacus"
REPORT = REPO / "docs" / "spot-checks-codacus.md"
MANUAL = "<!-- manual -->"
NUM = re.compile(r"\d[\d.,]*")


def numbers(text: str) -> set[str]:
    return {t.rstrip(".,").replace(",", "") for t in NUM.findall(text or "")}


def audit(row: dict, transcript: str, description: str) -> list[str]:
    bad = []
    source = numbers(transcript)
    for claim in row.get("claims") or []:
        found = numbers((claim.get("numbers") or "") + " " + (claim.get("claim") or ""))
        if found and not (found & source):
            bad.append("claim number not in transcript: " + ", ".join(sorted(found)))
    own = row.get("url") or ""
    for url in row.get("links") or []:
        if url != own and url not in transcript and url not in description:
            bad.append("link not in transcript/description: " + url)
    if not 3 <= len(row.get("mechanism") or []) <= 6:
        bad.append(f"mechanism has {len(row.get('mechanism') or [])} steps")
    if not row.get("goals"):
        bad.append("no goal")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="codacus.spotcheck")
    ap.add_argument("--out", type=Path, default=REPORT)
    args = ap.parse_args(argv)
    manual = ""
    if args.out.exists():
        text = args.out.read_text(encoding="utf-8")
        if MANUAL in text:
            manual = text.split(MANUAL, 1)[1]

    rows, flagged = [], 0
    for path in sorted(glob.glob(str(OUT / "*.json"))):
        if path.endswith("videos.json"):
            continue
        row = json.loads(Path(path).read_text(encoding="utf-8"))
        vid = row["id"]
        transcript = (OUT / f"{vid}.txt").read_text(encoding="utf-8")
        description = (load_info(vid) or {}).get("description") or ""
        bad = audit(row, transcript, description)
        flagged += bool(bad)
        rows.append((vid, len(row.get("claims") or []),
                     len(row.get("links") or []), len(row.get("mechanism") or []),
                     bad))

    total_claims = sum(r[1] for r in rows)
    total_links = sum(r[2] for r in rows)
    lines = [
        "# Codacus spot-check — local 9B extraction",
        "",
        f"Deterministic audit of all **{len(rows)}** videos (100 %, the batch is small). "
        "`python3 -m codacus.spotcheck` reads `data/codacus/<id>.json`, the transcript "
        "and the video description; it never writes the JSON.",
        "",
        "Checks (cheap screen): at least one numeric token in a claim appears in the transcript; every link "
        "appears in the transcript/description or is the video URL; the mechanism has "
        "3-6 steps; at least one goal. A claim whose numbers do not ground at all is "
        "flagged; one grounded number can mask one invented number, so the manual "
        "review below is the backstop.",
        "",
        f"- videos: **{len(rows)}**, claims: **{total_claims}**, links: **{total_links}**",
        f"- videos with at least one finding: **{flagged}/{len(rows)}**",
        "",
        "| video | claims | links | mechanism | findings |",
        "|---|---:|---:|---:|---|",
    ]
    for vid, claims, links, mech, bad in rows:
        note = "; ".join(bad) if bad else "—"
        lines.append(f"| [{vid}](https://www.youtube.com/watch?v={vid}) | {claims} "
                     f"| {links} | {mech} | {note} |")
    lines += ["", MANUAL + manual]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"{len(rows)} videos, {total_claims} claims, {total_links} links, "
          f"{flagged} with findings -> {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
