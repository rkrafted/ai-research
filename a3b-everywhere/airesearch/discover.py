"""PLAN sec 4.3 topic sweeps: find channels that cover the same topics.

Runs one yt-dlp flat search per topic term, counts how often each channel
appears, and writes the ranked discovery to data/private/discovery.json
(gitignored). Channels already in ``sources.py`` are marked so the next fetch
pass can add only the new ones.

Usage:
    python -m airesearch.discover --per-term 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter

from .fetch import _ytdlp_bin
from .sources import all_sources

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "private", "discovery.json")

TOPICS = [
    "quantization", "knowledge distillation", "mixture of experts",
    "looped transformer recurrent depth", "test-time compute",
    "speculative decoding", "LoRA QLoRA fine-tuning", "model merging",
    "synthetic data LLM", "small language model", "AI agents",
    "RAG retrieval augmented generation", "speech recognition",
    "vision language model",
]


def sweep(term: str, per_term: int, timeout: int = 120) -> list[dict]:
    cmd = [_ytdlp_bin(), "--flat-playlist", "--no-warnings", "--ignore-errors",
           "--playlist-end", str(per_term), "--dump-json", f"ytsearch{per_term}:{term}"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return []
    out = []
    for line in proc.stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("channel_id"):
            out.append({"channel_id": d["channel_id"], "channel": d.get("channel")})
    return out


def _known_channel_ids() -> set[str]:
    """Channel IDs already covered, read from the fetched cache and sources."""
    known = set()
    for _, url, _ in all_sources():
        tail = url.rstrip("/").split("/")[-1].lstrip("@")
        if tail.startswith("UC"):
            known.add(tail)
    cache = os.path.join(REPO, "data", "private", "channels")
    if os.path.isdir(cache):
        for name in os.listdir(cache):
            if not name.endswith(".json"):
                continue
            try:
                vids = json.load(open(os.path.join(cache, name))).get("videos") or []
            except (json.JSONDecodeError, OSError):
                continue
            for v in vids[:1]:
                if v.get("channel_id"):
                    known.add(v["channel_id"])
    return known


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-term", type=int, default=20)
    ap.add_argument("--terms", nargs="*", default=TOPICS)
    args = ap.parse_args(argv)

    counts: Counter = Counter()
    names: dict[str, str] = {}
    for term in args.terms:
        found = sweep(term, args.per_term)
        for f in found:
            counts[f["channel_id"]] += 1
            names.setdefault(f["channel_id"], f["channel"])
        print(f"{term:45s} {len(found)} hits", flush=True)

    known = _known_channel_ids()
    ranked = []
    for cid, n in counts.most_common():
        ranked.append({
            "channel_id": cid,
            "channel": names.get(cid),
            "hits": n,
            "already_known": cid in known,
            "url": f"https://www.youtube.com/channel/{cid}",
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"terms": list(args.terms), "channels": ranked}, fh, indent=1)

    new = [r for r in ranked if not r["already_known"]]
    print(f"\n{len(ranked)} unique channels, {len(new)} not already in sources -> {OUT}")
    for r in new[:15]:
        print(f"  {r['hits']:2d}  {r['channel']}  {r['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
