"""Build the R2 knowledge base ``data/kb.sqlite`` from the local-batch output.

Input is ``data/kb-extracts.jsonl`` — written by the runner's Ollama batch from
``local_batch/ingest-extract.yaml`` — with one ``{id, response, ...}`` per video.
No claim, paper or method is ever typed by hand here: everything comes from the
model's JSON response. The corpus (our own yt-dlp metadata) only supplies the
video title/channel/url that key the rows.

Tables: videos, claims, papers, methods, links.

Usage:
    python -m airesearch.kb build
    python -m airesearch.kb stats
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

from .ingest import CORPUS, REPO

EXTRACTS = REPO / "data" / "kb-extracts.jsonl"
DB = REPO / "data" / "kb.sqlite"

SCHEMA = """
CREATE TABLE videos (
  id TEXT PRIMARY KEY, channel TEXT, title TEXT, published TEXT,
  duration INTEGER, views INTEGER, url TEXT, audience TEXT
);
CREATE TABLE claims (
  id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT NOT NULL,
  claim TEXT NOT NULL, numbers TEXT, timestamp TEXT
);
CREATE TABLE papers (
  id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT NOT NULL,
  arxiv_id TEXT NOT NULL, title TEXT
);
CREATE TABLE methods (
  id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT NOT NULL, method TEXT NOT NULL
);
CREATE TABLE links (
  id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT NOT NULL,
  url TEXT NOT NULL, kind TEXT
);
CREATE UNIQUE INDEX ux_papers ON papers(video_id, arxiv_id);
CREATE INDEX ix_claims_video ON claims(video_id);
CREATE INDEX ix_papers_arxiv ON papers(arxiv_id);
CREATE INDEX ix_methods_video ON methods(video_id);
"""

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
_ARXIV_NEW = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
_ARXIV_OLD = re.compile(r"arxiv\.org/(?:abs|pdf)/([a-z.-]+/\d{7})", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s)\]\"'<>]+")


def parse_response(text: str) -> dict | None:
    """The model's JSON object, tolerating code fences and surrounding prose."""
    if not isinstance(text, str):
        return None
    s = _FENCE.sub("", text.strip()).strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def valid_arxiv(a: str) -> bool:
    """arXiv new-style id (YYMM.NNNNN) with a plausible year/month."""
    m = re.fullmatch(r"(\d{2})(\d{2})\.(\d{4,5})", a)
    if not m:
        return False
    yy, mm = int(m.group(1)), int(m.group(2))
    return 7 <= yy <= 26 and 1 <= mm <= 12


def normalize_arxiv(raw: str) -> str | None:
    s = str(raw or "").strip().replace("arXiv:", "").replace("arxiv:", "")
    m = _ARXIV_OLD.search(s)
    if m:
        return m.group(1).lower()
    s = s.rstrip(".,;")
    m = _ARXIV_NEW.search(s)
    if m and valid_arxiv(m.group(1)):
        return m.group(1)
    return None


def arxiv_ids(text: str) -> set[str]:
    out = {m.group(1).lower() for m in _ARXIV_OLD.finditer(text or "")}
    out |= {m.group(1) for m in _ARXIV_NEW.finditer(text or "") if valid_arxiv(m.group(1))}
    return out


def _as_list(x) -> list:
    if x is None or x == "":
        return []
    return x if isinstance(x, list) else [x]


def _s(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x)


def _link_kind(url: str) -> str:
    u = url.lower()
    if "github.com" in u:
        return "github"
    if "huggingface.co" in u:
        return "huggingface"
    if "arxiv.org" in u:
        return "arxiv"
    return "other"


def load_extracts(path: Path) -> dict[str, dict]:
    """id -> last extract row (a resumed batch must not duplicate a video)."""
    records: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and rec.get("id"):
            records[str(rec["id"])] = rec
    return records


def build(extracts: Path = EXTRACTS, db: Path = DB,
          corpus_path: Path = CORPUS) -> dict:
    records = load_extracts(extracts)
    if not records:
        raise SystemExit(f"no extracts in {extracts}")
    meta: dict[str, dict] = {}
    if Path(corpus_path).exists():
        for line in Path(corpus_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                meta[str(r["id"])] = r

    Path(db).parent.mkdir(parents=True, exist_ok=True)
    if Path(db).exists():
        Path(db).unlink()
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)

    counts = {"videos": 0, "claims": 0, "papers": 0, "methods": 0, "links": 0}
    for vid, rec in records.items():
        raw = rec.get("response") or ""
        obj = parse_response(raw)
        m = meta.get(vid, {})
        con.execute(
            "INSERT OR REPLACE INTO videos VALUES (?,?,?,?,?,?,?,?)",
            (vid, m.get("channel"), m.get("title"), m.get("upload_date"),
             m.get("duration"), m.get("view_count"),
             m.get("url") or f"https://www.youtube.com/watch?v={vid}",
             _s(obj.get("audience")) if obj else ""),
        )
        counts["videos"] += 1
        if obj:
            for method in _as_list(obj.get("methods"))[:8]:
                if _s(method):
                    con.execute("INSERT INTO methods (video_id, method) VALUES (?,?)",
                                (vid, _s(method)[:200]))
                    counts["methods"] += 1
            for claim in _as_list(obj.get("claims"))[:10]:
                if isinstance(claim, dict):
                    text, numbers = _s(claim.get("claim")), _s(claim.get("numbers"))
                    ts = claim.get("timestamp")
                else:
                    text, numbers, ts = _s(claim), "", None
                if text:
                    con.execute(
                        "INSERT INTO claims (video_id, claim, numbers, timestamp) "
                        "VALUES (?,?,?,?)",
                        (vid, text[:1000], numbers[:300], _s(ts) or None))
                    counts["claims"] += 1
            for paper in _as_list(obj.get("papers")):
                if isinstance(paper, dict):
                    raw_id, title = paper.get("arxiv_id"), _s(paper.get("title"))
                else:
                    raw_id, title = paper, ""
                aid = normalize_arxiv(_s(raw_id))
                if aid:
                    cur = con.execute(
                        "INSERT OR IGNORE INTO papers (video_id, arxiv_id, title) "
                        "VALUES (?,?,?)", (vid, aid, title[:300]))
                    counts["papers"] += cur.rowcount
            for url in _as_list(obj.get("repos")):
                url = _s(url)
                if url.startswith("http"):
                    con.execute("INSERT INTO links (video_id, url, kind) VALUES (?,?,?)",
                                (vid, url[:500], _link_kind(url)))
                    counts["links"] += 1
        # Recall: any arXiv id anywhere in the model's answer (claim text included).
        for aid in arxiv_ids(raw):
            cur = con.execute(
                "INSERT OR IGNORE INTO papers (video_id, arxiv_id, title) VALUES (?,?,?)",
                (vid, aid, ""))
            counts["papers"] += cur.rowcount
        for url in _URL.findall(raw):
            if _link_kind(url) in ("github", "huggingface"):
                con.execute("INSERT INTO links (video_id, url, kind) VALUES (?,?,?)",
                            (vid, url[:500], _link_kind(url)))
                counts["links"] += 1

    con.commit()
    con.execute("PRAGMA optimize")
    con.commit()
    con.close()
    return counts


def stats(db: Path = DB) -> dict:
    con = sqlite3.connect(db)
    q = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
    out = {
        "videos": q("SELECT count(*) FROM videos"),
        "claims": q("SELECT count(*) FROM claims"),
        "claim_videos": q("SELECT count(DISTINCT video_id) FROM claims"),
        "papers": q("SELECT count(*) FROM papers"),
        "arxiv_ids": q("SELECT count(DISTINCT arxiv_id) FROM papers"),
        "methods": q("SELECT count(*) FROM methods"),
        "links": q("SELECT count(*) FROM links"),
    }
    con.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="airesearch.kb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--extracts", default=str(EXTRACTS))
    b.add_argument("--db", default=str(DB))
    s = sub.add_parser("stats")
    s.add_argument("--db", default=str(DB))
    args = ap.parse_args(argv)

    if args.cmd == "build":
        counts = build(Path(args.extracts), Path(args.db))
        print("built " + " ".join(f"{k}={v}" for k, v in counts.items()))
        print(" ".join(f"{k}={v}" for k, v in stats(Path(args.db)).items()))
    else:
        print(" ".join(f"{k}={v}" for k, v in stats(Path(args.db)).items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
