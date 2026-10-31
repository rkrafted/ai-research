"""R2 ingest: metadata + captions for every video of the kept channels.

Two steps, both resumable and network-only (raw output stays in the gitignored
``data/private/``):

1. ``enumerate`` — a fast ``--flat-playlist`` pass per kept channel with the
   ``youtubetab:approximate_date`` extractor arg, filtered to the last
   ``--months`` months (default 24). Gives id, title, date, duration, views.
2. ``captions`` — one yt-dlp call per video (``--skip-download``): writes the
   per-video ``.info.json`` (full description + links) and the English
   auto/manual ``.vtt`` into ``data/private/captions/``. No media is downloaded.

``corpus`` merges both into ``data/private/ingest/corpus.jsonl`` (one row per
video, with description, links and caption availability). ``airesearch.prompts``
turns that into the local-batch prompts; ``airesearch.kb`` builds the SQLite
knowledge base from the batch output.

Usage:
    python -m airesearch.ingest enumerate --months 24
    python -m airesearch.ingest captions --workers 6
    python -m airesearch.ingest corpus
    python -m airesearch.ingest status
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

from .fetch import _ytdlp_bin, slug, videos_tab

REPO = Path(__file__).resolve().parent.parent
PRIVATE = REPO / "data" / "private"
INGEST = PRIVATE / "ingest"
CHAN_DIR = INGEST / "channels"
CAPTIONDIR = PRIVATE / "captions"
CORPUS = INGEST / "corpus.jsonl"

LINK_RE = re.compile(r"https?://[^\s)\]\"'<>]+")

# Fields kept from the full per-video info json.
INFO_KEEP = (
    "id",
    "title",
    "description",
    "duration",
    "view_count",
    "upload_date",
    "timestamp",
    "channel",
    "channel_id",
    "webpage_url",
)


def kept_channels(registry: Path | None = None) -> list[dict]:
    """The KEPT channels from data/channels.json, in registry order."""
    path = Path(registry) if registry else REPO / "data" / "channels.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [c for c in data if c.get("kept")]


def channel_rank() -> dict[str, float]:
    """channel name -> R1 lens score (0.0 when unknown), for download priority."""
    try:
        data = json.loads((REPO / "data" / "channels.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {c["name"]: float(c.get("score") or 0.0) for c in data}


def window_start(months: int, today: date | None = None) -> date:
    """Calendar date ``months`` months before ``today`` (no dateutil needed)."""
    today = today or date.today()
    y, m = today.year, today.month - months
    while m <= 0:
        m += 12
        y -= 1
    return date(y, m, 1)


def _date_from(rec: dict) -> str | None:
    up = rec.get("upload_date")
    if up:
        return str(up)
    ts = rec.get("timestamp") or rec.get("release_timestamp")
    if isinstance(ts, (int, float)) and ts > 0:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d")
    return None


def enumerate_channel(name: str, url: str, months: int, *, timeout: int = 600,
                      force: bool = False) -> dict:
    """Flat, date-filtered video list for one channel. Cached per channel slug."""
    url = videos_tab(url)
    CHAN_DIR.mkdir(parents=True, exist_ok=True)
    path = CHAN_DIR / (slug(url) + ".json")
    start = window_start(months)
    if path.exists() and not force:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("window_start") == start.isoformat() and cached.get("videos"):
                return {"ok": True, "videos": cached["videos"], "cached": True}
        except (json.JSONDecodeError, OSError):
            pass

    cmd = [
        _ytdlp_bin(),
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
        "--ignore-errors",
        "--retries", "2",
        "--extractor-args", "youtubetab:approximate_date",
        "--dateafter", start.strftime("%Y%m%d"),
        "--playlist-end", "5000",
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "videos": [], "error": "timeout"}

    videos: list[dict] = []
    seen: set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        vid = d.get("id")
        if not vid or vid in seen:
            continue
        day = _date_from(d)
        if day and day < start.strftime("%Y%m%d"):
            continue
        seen.add(vid)
        videos.append({
            "id": vid,
            "title": d.get("title"),
            "upload_date": day,
            "duration": d.get("duration"),
            "view_count": d.get("view_count"),
            "url": d.get("url") or f"https://www.youtube.com/watch?v={vid}",
        })

    if not videos:
        err = (proc.stderr or "").strip().splitlines()
        return {"ok": False, "videos": [], "error": err[-1] if err else "no videos"}

    payload = {
        "name": name,
        "url": url,
        "window_start": start.isoformat(),
        "months": months,
        "videos": videos,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "videos": videos}


def caption_files(vid: str) -> list[Path]:
    return sorted(CAPTIONDIR.glob(f"{vid}.*.vtt")) if CAPTIONDIR.exists() else []


def info_path(vid: str) -> Path:
    return CAPTIONDIR / f"{vid}.info.json"


def load_info(vid: str) -> dict | None:
    p = info_path(vid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def fetch_video(vid: str, *, timeout: int = 240, force: bool = False) -> dict:
    """Per-video metadata + English captions (no media). Cached by info.json."""
    if info_path(vid).exists() and not force:
        return {"ok": True, "cached": True}
    CAPTIONDIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ytdlp_bin(),
        "--skip-download",
        "--no-warnings",
        "--ignore-errors",
        "--retries", "2",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en,en-orig",
        "--sub-format", "vtt",
        "-o", str(CAPTIONDIR / "%(id)s.%(ext)s"),
        f"https://www.youtube.com/watch?v={vid}",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    if info_path(vid).exists():
        return {"ok": True}
    # Unavailable/removed/age-gated: remember so we never retry it.
    (CAPTIONDIR / f"{vid}.missing").write_text("no info\n", encoding="utf-8")
    return {"ok": False, "error": "no info"}


def _enumerated() -> list[dict]:
    """All enumerated rows with their channel name attached."""
    rows: list[dict] = []
    if not CHAN_DIR.exists():
        return rows
    for path in sorted(CHAN_DIR.glob("*.json")):
        try:
            ch = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for v in ch.get("videos", []):
            rows.append({**v, "channel": ch.get("name") or path.stem})
    return rows


def build_corpus() -> list[dict]:
    """Merge enumeration + per-video info into data/private/ingest/corpus.jsonl."""
    rows: list[dict] = []
    for v in _enumerated():
        info = load_info(v["id"]) or {}
        desc = info.get("description") or ""
        caps = caption_files(v["id"])
        rows.append({
            "id": v["id"],
            "channel": v.get("channel"),
            "title": info.get("title") or v.get("title"),
            "upload_date": info.get("upload_date") or v.get("upload_date"),
            "duration": info.get("duration") or v.get("duration"),
            "view_count": info.get("view_count") or v.get("view_count"),
            "url": info.get("webpage_url") or v.get("url")
                   or f"https://www.youtube.com/watch?v={v['id']}",
            "description": desc,
            "links": sorted(set(LINK_RE.findall(desc)))[:60],
            "captions": [p.name for p in caps],
            "has_captions": bool(caps),
        })
    INGEST.mkdir(parents=True, exist_ok=True)
    with CORPUS.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def load_corpus() -> list[dict]:
    if not CORPUS.exists():
        return []
    out = []
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def cmd_enumerate(args) -> int:
    todo = kept_channels()
    print(f"enumerating {len(todo)} kept channels, months={args.months}, "
          f"start={window_start(args.months).isoformat()}")
    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(enumerate_channel, c["name"], c["url"], args.months,
                          force=args.force): c for c in todo}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:                       # noqa: BLE001
                print(f"  FAIL {c['name']:32s} {type(exc).__name__}: {exc}", flush=True)
                continue
            if r["ok"]:
                total += len(r["videos"])
                print(f"  OK   {c['name']:32s} {len(r['videos']):4d} videos"
                      + (" (cached)" if r.get("cached") else ""), flush=True)
            else:
                print(f"  FAIL {c['name']:32s} {r.get('error')}", flush=True)
    print(f"total videos in window: {total}")
    return 0 if total else 1


def cmd_captions(args) -> int:
    rows = _enumerated()
    if not rows:
        print("nothing enumerated; run `enumerate` first")
        return 1
    # High-scoring channels first, newest video first: a partial download still
    # covers the videos most likely to make the extraction cut.
    rank = channel_rank()
    rows.sort(key=lambda r: str(r.get("upload_date") or ""), reverse=True)
    rows.sort(key=lambda r: rank.get(r.get("channel") or "", 0.0), reverse=True)
    todo = [r for r in rows if not info_path(r["id"]).exists()
            and not (CAPTIONDIR / f"{r['id']}.missing").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"captions: {len(todo)} missing of {len(rows)} enumerated, workers={args.workers}", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_video, r["id"], force=args.force): r for r in todo}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:                       # noqa: BLE001
                res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if res["ok"]:
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(todo)}", flush=True)
            else:
                print(f"  miss {r['id']} {res.get('error')}", flush=True)
    print(f"captions: {done} fetched, {len(todo) - done} unavailable")
    return 0


def cmd_corpus(args) -> int:
    rows = build_corpus()
    withcaps = sum(1 for r in rows if r["has_captions"])
    withdesc = sum(1 for r in rows if r["description"])
    print(f"corpus: {len(rows)} videos, {withdesc} with description, {withcaps} with captions "
          f"-> {CORPUS.relative_to(REPO)}")
    return 0 if rows else 1


def cmd_status(args) -> int:
    rows = _enumerated()
    have = sum(1 for r in rows if info_path(r["id"]).exists())
    caps = sum(1 for r in rows if caption_files(r["id"]))
    misses = len(list(CAPTIONDIR.glob("*.missing"))) if CAPTIONDIR.exists() else 0
    print(f"enumerated={len(rows)} info={have} captions={caps} missing={misses}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="airesearch.ingest")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("enumerate")
    p1.add_argument("--months", type=int, default=24)
    p1.add_argument("--workers", type=int, default=4)
    p1.add_argument("--force", action="store_true")
    p1.set_defaults(func=cmd_enumerate)

    p2 = sub.add_parser("captions")
    p2.add_argument("--workers", type=int, default=6)
    p2.add_argument("--limit", type=int, default=0)
    p2.add_argument("--force", action="store_true")
    p2.set_defaults(func=cmd_captions)

    p3 = sub.add_parser("corpus")
    p3.set_defaults(func=cmd_corpus)

    p4 = sub.add_parser("status")
    p4.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
