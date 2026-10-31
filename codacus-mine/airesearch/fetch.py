"""Fetch YouTube channel metadata with yt-dlp (flat, no video downloads).

One yt-dlp process per channel, ``--playlist-end N --dump-json --skip-download``.
For each video we keep only the fields the scorer needs and drop the format
lists (which are large). Raw output lands in data/private/ (gitignored); the
committed registry is built separately by ``airesearch.registry``.

Usage:
    python -m airesearch.fetch            # all sources, workers=4
    python -m airesearch.fetch --limit 20
    python -m airesearch.fetch --only seed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from .sources import all_sources

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, "data", "private", "channels")
CAPTIONDIR = os.path.join(REPO, "data", "private", "captions")

KEEP = (
    "id",
    "title",
    "description",
    "duration",
    "view_count",
    "upload_date",
    "channel",
    "channel_id",
    "channel_follower_count",
    "availability",
)


def slug(url: str) -> str:
    """Stable per-channel filename; ignores a trailing tab segment."""
    url = re.sub(r"/(videos|shorts|streams)/?$", "", url.rstrip("/"))
    tail = url.rstrip("/").split("/")[-1].lstrip("@")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", tail)


def videos_tab(url: str) -> str:
    """Channel URLs must point at the /videos tab or yt-dlp reads the home tab."""
    if "youtube.com" in url and not re.search(r"/(videos|shorts|streams)/?$", url):
        return url.rstrip("/") + "/videos"
    return url


def _ytdlp_bin() -> str:
    venv = os.path.join(REPO, ".venv", "bin", "yt-dlp")
    return venv if os.path.exists(venv) else "yt-dlp"


def fetch_channel(url: str, limit: int, timeout: int = 300, force: bool = False,
                  captions: bool = False) -> dict:
    """Return {ok, videos, error}. Cached per-URL unless force/captions.

    With ``captions=True`` the English auto/manual subtitle files are also
    written to data/private/captions/ (no media is downloaded).
    """
    url = videos_tab(url)
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, slug(url) + ".json")
    if os.path.exists(path) and not force and not captions:
        try:
            data = json.load(open(path))
            if data.get("videos"):
                return {"ok": True, "videos": data["videos"], "cached": True}
        except (json.JSONDecodeError, OSError):
            pass

    cmd = [
        _ytdlp_bin(),
        "--skip-download",
        "--no-warnings",
        "--ignore-errors",
        "--retries", "2",
        "--playlist-end", str(limit),
        "--dump-json",
        url,
    ]
    if captions:
        os.makedirs(CAPTIONDIR, exist_ok=True)
        cmd[-1:-1] = [
            "--write-subs", "--write-auto-subs",
            "--sub-langs", "en.*,en", "--sub-format", "vtt",
            "-o", os.path.join(CAPTIONDIR, "%(id)s.%(ext)s"),
        ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "videos": [], "error": "timeout"}

    videos: list = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        rec = {k: d.get(k) for k in KEEP}
        rec["captions_auto"] = sorted((d.get("automatic_captions") or {}).keys())[:5]
        rec["captions_manual"] = sorted((d.get("subtitles") or {}).keys())
        videos.append(rec)

    videos = videos[:limit]
    if not videos:
        err = (proc.stderr or "").strip().splitlines()
        return {"ok": False, "videos": [], "error": err[-1] if err else "no videos"}

    with open(path, "w") as fh:
        json.dump({"url": url, "videos": videos}, fh)
    return {"ok": True, "videos": videos}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", choices=["seed", "candidate", "expansion"], default=None)
    ap.add_argument("--match", default=None, help="substring filter on channel name/url")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--captions", action="store_true",
                    help="also write English VTT captions to data/private/captions/")
    args = ap.parse_args(argv)

    todo = [
        (n, u, p)
        for n, u, p in all_sources()
        if (args.only is None or p == args.only)
        and (args.match is None or args.match.lower() in (n + u).lower())
    ]
    print(f"fetching {len(todo)} channels, limit={args.limit}, workers={args.workers}")

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(fetch_channel, u, args.limit, force=args.force,
                      captions=args.captions): (n, u, p)
            for n, u, p in todo
        }
        for fut in as_completed(futs):
            n, u, p = futs[fut]
            r = fut.result()
            results[u] = (n, p, r)
            if r["ok"]:
                print(f"  OK   {n:35s} {len(r['videos']):3d} videos"
                      + (" (cached)" if r.get("cached") else ""), flush=True)
            else:
                print(f"  FAIL {n:35s} {r.get('error')}", flush=True)

    ok = sum(1 for _, _, r in results.values() if r["ok"])
    print(f"\n{ok}/{len(todo)} channels ok")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
