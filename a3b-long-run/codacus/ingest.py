"""Codacus channel ingest: last 30 uploads + English captions.

Reuses the R2 house tool ``airesearch.ingest`` for the per-video caption fetch
(``--skip-download``, ``--write-subs --write-auto-subs``, one call per video)
and ``airesearch.prompts.vtt_segments`` for the rolling auto-caption
de-duplication. New here: the channel-specific list, a plain-text transcript per
video (timestamps dropped) and the ``data/codacus/videos.json`` shape.

Network output lands in the gitignored ``data/private/codacus/`` (flat list) and
``data/private/captions/`` (VTT + info json). The committed deliverables are
``data/codacus/videos.json`` and ``data/codacus/<id>.txt``.

Captions only, no media, no login. YouTube answered HTTP 429 to two quick
requests on 2026-10-17, so the default gap is 5 s and a failed fetch backs off.

Usage:
    python -m codacus.ingest list
    python -m codacus.ingest captions --sleep 5
    python -m codacus.ingest status
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from airesearch import ingest as ai
from airesearch.prompts import vtt_segments

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "codacus"
PRIVATE = REPO / "data" / "private" / "codacus"

CHANNEL = "Codacus"
CHANNEL_URL = "https://www.youtube.com/channel/UCsRvxZErBo0ByyWUX_aVuvg"
FLAT = PRIVATE / "flat.jsonl"
VIDEOS = OUT / "videos.json"
LIMIT = 30
# The two links named in the project brief. Both are inside the last 30 now; the
# code keeps them anyway so a later upload cannot push one out of the sample.
EXTRA = ("pIN-2oVJpyU", "bcGO7xre46o")


def _flat(force: bool = False, timeout: int = 600) -> list[dict]:
    """Newest-first flat listing with approximate upload dates (one request)."""
    if FLAT.exists() and not force:
        rows = [json.loads(line) for line in FLAT.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if rows:
            return rows
    PRIVATE.mkdir(parents=True, exist_ok=True)
    cmd = [
        ai._ytdlp_bin(), "--flat-playlist", "--dump-json", "--no-warnings",
        "--ignore-errors", "--retries", "2",
        "--extractor-args", "youtubetab:approximate_date",
        "--playlist-end", "60", ai.videos_tab(CHANNEL_URL),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise SystemExit(f"enumerate failed: {(proc.stderr or '').strip()[-300:]}")
    FLAT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [json.loads(line) for line in lines]


def _row(d: dict) -> dict:
    vid = d["id"]
    day = d.get("upload_date")
    if not day and d.get("timestamp"):
        day = time.strftime("%Y%m%d", time.gmtime(d["timestamp"]))
    return {
        "id": vid,
        "title": d.get("title"),
        "upload_date": day,
        "duration": d.get("duration"),
        "url": d.get("url") or f"https://www.youtube.com/watch?v={vid}",
        "channel": CHANNEL,
    }


def build_videos(force: bool = False) -> list[dict]:
    """The last 30 uploads plus the two brief links, newest first."""
    rows = [_row(d) for d in _flat(force=force)]
    seen = {r["id"] for r in rows}
    chosen = rows[:LIMIT]
    for vid in EXTRA:
        if vid in {r["id"] for r in chosen}:
            continue
        if vid in seen:
            chosen.append(next(r for r in rows if r["id"] == vid))
            continue
        # Older than the flat window: one metadata call, then a row from info.
        ai.fetch_video(vid)
        info = ai.load_info(vid) or {}
        if info:
            chosen.append(_row(info))
    OUT.mkdir(parents=True, exist_ok=True)
    VIDEOS.write_text(json.dumps(chosen, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return chosen


def vtt_text(vid: str) -> str:
    """Plain transcript for one video (timestamps dropped, rolling cues merged)."""
    files = sorted(ai.CAPTIONDIR.glob(f"{vid}.*.vtt")) if ai.CAPTIONDIR.exists() else []
    if not files:
        return ""
    preferred = [p for p in files if p.name.endswith(".en.vtt")] or files
    segs = vtt_segments(preferred[0].read_text(encoding="utf-8", errors="replace"),
                        chunk_chars=600)
    text = " ".join(t for _, t in segs)
    text = html.unescape(text)
    text = re.sub(r"(?:>>\s*)+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("<c>", "").replace("</c>", "")
    return text


def captions(rows: list[dict], sleep: float = 5.0, retries: int = 2) -> dict:
    """Fetch captions one video at a time, write ``data/codacus/<id>.txt``."""
    OUT.mkdir(parents=True, exist_ok=True)
    have, missing, cached = 0, [], 0
    for i, row in enumerate(rows, 1):
        vid = row["id"]
        dest = OUT / f"{vid}.txt"
        if dest.exists() and dest.stat().st_size > 200:
            cached += 1
            continue
        if (ai.CAPTIONDIR / f"{vid}.missing").exists():
            missing.append(vid)
            continue
        ok = False
        for attempt in range(retries + 1):
            if i > 1 or attempt:
                time.sleep(sleep if attempt == 0 else max(60.0, sleep * 12))
            ai.fetch_video(vid)
            text = vtt_text(vid)
            if text:
                dest.write_text(text + "\n", encoding="utf-8")
                ok = True
                break
            # A failed fetch may mean 429, not "no captions": drop the marker so
            # a later session (or this retry) can try again.
            marker = ai.CAPTIONDIR / f"{vid}.missing"
            if marker.exists():
                marker.unlink()
            print(f"[{i}/{len(rows)}] {vid}: retry {attempt + 1}/{retries}", file=sys.stderr)
        if ok:
            have += 1
            print(f"[{i}/{len(rows)}] {vid}: {len(vtt_text(vid))} chars")
        else:
            missing.append(vid)
            print(f"[{i}/{len(rows)}] {vid}: NO CAPTIONS")
    return {"have": have, "cached": cached, "missing": missing, "total": len(rows)}


def cmd_list(args) -> int:
    rows = build_videos(force=args.force)
    for r in rows:
        print(f"{r['id']}  {r['upload_date']}  {str(r['duration']):>5}s  {r['title']}")
    print(f"videos.json: {len(rows)} rows -> {VIDEOS.relative_to(REPO)}")
    return 0


def cmd_captions(args) -> int:
    rows = json.loads(VIDEOS.read_text(encoding="utf-8")) if VIDEOS.exists() else build_videos()
    res = captions(rows, sleep=args.sleep, retries=args.retries)
    print(f"captions: {res['have']} new, {res['cached']} cached, "
          f"{len(res['missing'])} missing of {res['total']}")
    if res["missing"]:
        print("missing: " + " ".join(res["missing"]))
    return 0 if res["have"] + res["cached"] >= 25 else 1


def cmd_status(args) -> int:
    rows = json.loads(VIDEOS.read_text(encoding="utf-8")) if VIDEOS.exists() else []
    have = sum(1 for r in rows if (OUT / f"{r['id']}.txt").stat().st_size > 200
               if (OUT / f"{r['id']}.txt").exists())
    chars = sorted((OUT / f"{r['id']}.txt").stat().st_size for r in rows
                   if (OUT / f"{r['id']}.txt").exists())
    print(f"videos={len(rows)} transcripts={have}")
    if chars:
        print(f"chars min={chars[0]} median={chars[len(chars) // 2]} max={chars[-1]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="codacus.ingest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("captions")
    p.add_argument("--sleep", type=float, default=5.0)
    p.add_argument("--retries", type=int, default=2)
    p.set_defaults(func=cmd_captions)
    p = sub.add_parser("status")
    p.set_defaults(func=cmd_status)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
