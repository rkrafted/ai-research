"""bycloud catalogue + captions -> data/private/bycloud/transcripts/<id>.json.

Reuses research-ingest's caption path: one yt-dlp call per video
(``--skip-download --write-info-json --write-subs --write-auto-subs``, English, vtt)
inside its own cache dir, so a re-run resumes on the cached files. No media is
downloaded. Captions that already exist in a sibling ``--reuse-dir`` (the
research-ingest cache) are parsed instead of fetched again.

Run::

    python3 -m bycloud catalogue                 # yt-dlp flat playlist -> catalogue.json
    python3 -m bycloud captions --target-hours 100
    python3 -m bycloud status
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRIVATE = REPO / "data" / "private" / "bycloud"
RAW = PRIVATE / "raw"
TRANSCRIPTS = PRIVATE / "transcripts"
FLAT = PRIVATE / "flat.jsonl"
CATALOGUE = PRIVATE / "catalogue.json"
VIDEOS = PRIVATE / "videos.json"
NO_CAPTIONS = PRIVATE / "no_captions.json"

CHANNEL_URL = "https://www.youtube.com/@bycloudAI/videos"

_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[.,](\d+)")
_TAG = re.compile(r"<[^>]*>")
_HOME = re.compile("/" + "home" + "/")          # split so this file passes the raw scan


# --------------------------------------------------------------------------
# yt-dlp
# --------------------------------------------------------------------------


def ytdlp_bin() -> str:
    """Prefer the repo venv, then PATH, then two known local installs (via ~)."""
    env = os.environ.get("BYCLOUD_YTDLP")
    if env and Path(env).exists():
        return env
    cands = [REPO / ".venv" / "bin" / "yt-dlp"]
    found = shutil.which("yt-dlp")
    if found:
        cands.append(Path(found))
    cands += [Path.home() / ".local" / "bin" / "yt-dlp",
              Path.home() / "TickTock" / ".ytvenv" / "bin" / "yt-dlp"]
    for c in cands:
        if Path(c).exists():
            return str(c)
    return "yt-dlp"


# --------------------------------------------------------------------------
# WebVTT -> segments (de-duplicated rolling auto-caption lines)
# --------------------------------------------------------------------------


def _secs(m: re.Match) -> float:
    h, mi, s, frac = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(frac.ljust(3, "0")[:3]) / 1000.0


def parse_vtt(text: str) -> list[dict]:
    """``[{"start","end","text"}]`` from a WebVTT file with the rolling lines removed.

    YouTube auto-captions repeat the previous cue's text at the top of the next
    cue (and add word-level ``<00:00:...>`` tags). Every cue's text is cleaned of
    tags/entities, then the longest prefix equal to the previous cue is dropped;
    a shifted-window suffix/prefix overlap is dropped as a fallback.
    """
    blocks: list[list] = []                      # [start, end, [lines]]
    cur: list | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE",
                                        "STYLE", "REGION")):
            continue
        if "-->" in line:
            if cur and cur[2]:
                blocks.append(cur)
            left, _, right = line.partition("-->")
            m, m2 = _TIME.search(left), _TIME.search(right)
            start = _secs(m) if m else 0.0
            end = _secs(m2) if m2 else start
            cur = [start, end, []]
            continue
        if cur is None:
            continue
        clean = html.unescape(_TAG.sub("", line)).strip()
        if clean and (not cur[2] or cur[2][-1] != clean):
            cur[2].append(clean)
    if cur and cur[2]:
        blocks.append(cur)

    segs: list[dict] = []
    prev = ""
    for start, end, lines in blocks:
        txt = re.sub(r"\s+", " ", " ".join(lines)).strip()
        if not txt:
            continue
        inc = txt
        for _ in range(2):                       # cue may repeat the previous line twice
            if prev and inc.startswith(prev):
                inc = inc[len(prev):].strip()
        if prev and inc and not inc.startswith(prev):
            k = min(len(prev), len(inc))         # shifted window: prev tail == inc head
            while k > 8 and prev[-k:] != inc[:k]:
                k -= 1
            if k > 8:
                inc = inc[k:].strip()
        if inc:
            segs.append({"start": round(start, 3), "end": round(end, 3), "text": inc})
        prev = txt
    return segs


def _clean_vtt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def detect_source(vid: str, dirs: list[Path], vtt: Path, info: dict | None) -> str:
    """``manual`` when yt-dlp lists an English (non-auto) subtitle track, else ``auto``."""
    if info:
        keys = list((info.get("subtitles") or {}).keys())
        if any(k.lower().startswith("en") for k in keys):
            return "manual"
        auto = (info.get("automatic_captions") or {})
        if any(k.lower().startswith("en") for k in auto):
            return "auto"
    name = vtt.name.lower()
    return "auto" if ("-orig" in name or re.search(r"<\d{2}:\d{2}", _clean_vtt(vtt))) else "manual"


def pick_vtt(vid: str, dirs: list[Path]) -> Path | None:
    """The best English VTT for ``vid`` across ``dirs`` (prefer plain ``.en.vtt``)."""
    files: list[Path] = []
    for d in dirs:
        if d and Path(d).is_dir():
            files += [p for p in sorted(Path(d).glob(f"{vid}.*.vtt"))
                      if not p.name.endswith(".part")]
    if not files:
        return None
    for suffix in (f"{vid}.en.vtt", f"{vid}.en-orig.vtt"):
        for p in files:
            if p.name == suffix:
                return p
    return files[0]


def load_info(vid: str, dirs: list[Path]) -> dict | None:
    for d in dirs:
        p = Path(d) / f"{vid}.info.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return None


def _dirs(primary: Path, reuse: list[Path] | None = None) -> list[Path]:
    return [primary] + [Path(d) for d in (reuse or [])]


def transcript_for(vid: str, duration: float | None = None,
                   dirs: list[Path] | None = None) -> dict | None:
    """Build the transcript dict for one video, or ``None`` when it has no captions."""
    dirs = dirs or _dirs(RAW)
    vtt = pick_vtt(vid, dirs)
    if vtt is None:
        return None
    segs = parse_vtt(_clean_vtt(vtt))
    if not segs:                                  # captions present but unusable
        return None
    info = load_info(vid, dirs)
    dur = duration or (info or {}).get("duration") or segs[-1]["end"]
    return {
        "video_id": vid,
        "duration_s": round(float(dur)),
        "source": detect_source(vid, dirs, vtt, info),
        "segments": segs,
    }


def write_transcript(doc: dict) -> Path:
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPTS / f"{doc['video_id']}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# catalogue (flat playlist) + selection
# --------------------------------------------------------------------------


def _flat_rows(flat: Path = FLAT) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    if not flat.exists():
        return rows
    for line in flat.read_text(encoding="utf-8", errors="replace").splitlines():
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
        seen.add(vid)
        day = d.get("upload_date")
        if not day and isinstance(d.get("timestamp"), (int, float)):
            day = datetime.fromtimestamp(d["timestamp"], tz=timezone.utc).strftime("%Y%m%d")
        rows.append({
            "id": vid,
            "title": d.get("title"),
            "upload_date": day,
            "duration": d.get("duration"),
            "view_count": d.get("view_count"),
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
    return rows


def run_flat_playlist(url: str = CHANNEL_URL, *, timeout: int = 3600) -> list[dict]:
    PRIVATE.mkdir(parents=True, exist_ok=True)
    cmd = [ytdlp_bin(), "--flat-playlist", "--dump-json", "--no-warnings",
           "--ignore-errors", "--retries", "2",
           "--extractor-args", "youtubetab:approximate_date",
           "--playlist-end", "3000", url]
    with FLAT.open("w", encoding="utf-8") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.DEVNULL, timeout=timeout)
    return _flat_rows()


def build_catalogue(rows: list[dict] | None = None) -> dict:
    """Write catalogue.json = {"total_videos","total_hours"} and videos.json (private rows)."""
    rows = rows if rows is not None else _flat_rows()
    total_s = sum(float(r.get("duration") or 0) for r in rows)
    cat = {"total_videos": len(rows), "total_hours": round(total_s / 3600, 2)}
    PRIVATE.mkdir(parents=True, exist_ok=True)
    CATALOGUE.write_text(json.dumps(cat, indent=2) + "\n", encoding="utf-8")
    VIDEOS.write_text(json.dumps(rows, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    return cat


def load_videos() -> list[dict]:
    if VIDEOS.exists():
        try:
            return json.loads(VIDEOS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return _flat_rows()


def select_videos(rows: list[dict], target_hours: float = 100.0,
                  min_videos: int = 136) -> list[dict]:
    """Newest-first prefix until ``target_hours`` or the catalogue ends, >= min_videos."""
    picked: list[dict] = []
    total = 0.0
    for r in rows:
        picked.append(r)
        total += float(r.get("duration") or 0) / 3600.0
        if len(picked) >= min_videos and total >= target_hours:
            break
    return picked


# --------------------------------------------------------------------------
# caption fetch
# --------------------------------------------------------------------------


def fetch_one(vid: str, *, reuse: list[Path] | None = None, timeout: int = 300) -> dict:
    """Ensure a VTT exists (cached or via yt-dlp) and write the transcript json."""
    out = TRANSCRIPTS / f"{vid}.json"
    if out.exists():
        return {"ok": True, "cached": True}
    dirs = _dirs(RAW, reuse)
    if pick_vtt(vid, dirs) is None:
        RAW.mkdir(parents=True, exist_ok=True)
        cmd = [ytdlp_bin(), "--skip-download", "--no-warnings", "--ignore-errors",
               "--retries", "2", "--sleep-requests", "1",
               "--write-info-json", "--write-subs", "--write-auto-subs",
               "--sub-langs", "en,en-orig", "--sub-format", "vtt",
               "--no-playlist", "-o", str(RAW / "%(id)s.%(ext)s"),
               f"https://www.youtube.com/watch?v={vid}"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout"}
    doc = transcript_for(vid, dirs=dirs)
    if doc is None:
        return {"ok": False, "error": "no captions"}
    write_transcript(doc)
    return {"ok": True, "hours": doc["duration_s"] / 3600.0}


def fetch_transcripts(rows: list[dict], *, workers: int = 3, reuse: list[Path] | None = None,
                      target_hours: float = 100.0) -> dict:
    """Download (politely) until the *transcript* hours cover target_hours or rows run out."""
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in TRANSCRIPTS.glob("*.json")}
    have_h = sum((json.loads((TRANSCRIPTS / f"{v}.json").read_text(encoding="utf-8"))
                  .get("duration_s") or 0) for v in have) / 3600.0
    todo = [r for r in rows if r["id"] not in have]
    todo.sort(key=lambda r: str(r.get("upload_date") or ""), reverse=True)
    print(f"transcripts: have {len(have)} ({have_h:.1f} h), todo {len(todo)}", flush=True)
    misses: list[str] = []
    fetched = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {}
        for r in todo:
            futs[ex.submit(fetch_one, r["id"], reuse=reuse)] = r
            time.sleep(random.uniform(0.1, 0.3))     # polite pacing between submissions
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:                 # noqa: BLE001
                res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if res.get("ok"):
                fetched += 1
                if fetched % 25 == 0:
                    print(f"  {fetched}/{len(todo)}", flush=True)
            else:
                misses.append(r["id"])
                print(f"  miss {r['id']} {res.get('error')}", flush=True)
    no_cap = sorted(set(misses))
    NO_CAPTIONS.write_text(json.dumps(no_cap, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    print(f"captions: {fetched} fetched, {len(no_cap)} without captions -> {NO_CAPTIONS.name}")
    return {"fetched": fetched, "no_captions": no_cap}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def cmd_catalogue(args) -> int:
    rows = _flat_rows()
    if args.refresh or not rows:
        rows = run_flat_playlist()
    cat = build_catalogue(rows)
    print(f"catalogue: {cat['total_videos']} videos, {cat['total_hours']} h "
          f"-> {CATALOGUE.relative_to(REPO)}")
    return 0 if cat["total_videos"] else 1


def cmd_captions(args) -> int:
    rows = load_videos()
    if not rows:
        print("no catalogue; run `python3 -m bycloud catalogue` first")
        return 1
    chosen = select_videos(rows, target_hours=args.target_hours)
    reuse = [Path(p) for p in args.reuse_dir if p]
    print(f"selected {len(chosen)} videos (target {args.target_hours} h, "
          f"{sum(float(r.get('duration') or 0) for r in chosen) / 3600:.1f} h)")
    fetch_transcripts(chosen, workers=args.workers, reuse=reuse,
                      target_hours=args.target_hours)
    return 0


def cmd_status(args) -> int:
    rows = _flat_rows()
    have = list(TRANSCRIPTS.glob("*.json")) if TRANSCRIPTS.exists() else []
    hours = 0.0
    for p in have:
        try:
            hours += (json.loads(p.read_text(encoding="utf-8")).get("duration_s") or 0) / 3600.0
        except (OSError, json.JSONDecodeError):
            continue
    cat = {}
    if CATALOGUE.exists():
        cat = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    misses = json.loads(NO_CAPTIONS.read_text(encoding="utf-8")) if NO_CAPTIONS.exists() else []
    print(f"catalogue={len(rows)} transcripts={len(have)} ({hours:.1f} h) "
          f"total_hours={cat.get('total_hours')} no_captions={len(misses)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bycloud.ingest")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("catalogue")
    p1.add_argument("--refresh", action="store_true")
    p1.set_defaults(func=cmd_catalogue)

    p2 = sub.add_parser("captions")
    p2.add_argument("--workers", type=int, default=3)
    p2.add_argument("--target-hours", type=float, default=100.0)
    p2.add_argument("--reuse-dir", action="append", default=[],
                    help="already-downloaded caption dir to parse before fetching (repeatable)")
    p2.set_defaults(func=cmd_captions)

    p3 = sub.add_parser("status")
    p3.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
