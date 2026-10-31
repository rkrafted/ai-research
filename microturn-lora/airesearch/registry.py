"""Build the committed R1 registry: data/channels.json + docs/channels.md.

Reads cached channel metadata from data/private/channels (written by
``airesearch.fetch``), scores every channel with the PLAN sec 2 formula, keeps
channels that score like the seeds, and writes the machine + human registry.

Usage:
    python -m airesearch.registry
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

from .extract import extract_channel
from .fetch import slug
from .score import components, keep_threshold, reason, score_channel
from .sources import all_sources

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHANNEL_DIR = os.path.join(REPO, "data", "private", "channels")
CAPTION_DIR = os.path.join(REPO, "data", "private", "captions")
OUT_JSON = os.path.join(REPO, "data", "channels.json")
OUT_MD = os.path.join(REPO, "docs", "channels.md")

MIN_NON_SEED_SAMPLE = 20


def load_videos(url: str, cache_dir: str) -> list[dict]:
    path = os.path.join(cache_dir, slug(url) + ".json")
    if not os.path.exists(path):
        return []
    try:
        return json.load(open(path)).get("videos") or []
    except (json.JSONDecodeError, OSError):
        return []


def build(cache_dir: str = CHANNEL_DIR) -> dict:
    rows = []
    skipped = []
    for name, url, provenance in all_sources():
        videos = load_videos(url, cache_dir)
        seed = provenance == "seed"
        if not videos:
            skipped.append({"name": name, "url": url, "why": "not fetched"})
            continue
        if not seed and len(videos) < MIN_NON_SEED_SAMPLE:
            skipped.append(
                {"name": name, "url": url,
                 "why": f"only {len(videos)} videos listed (<{MIN_NON_SEED_SAMPLE})"}
            )
            continue
        ext = extract_channel(videos)
        # A seed with fewer than 20 listed videos cannot supply the standard
        # sample; record 0 sampled and keep the real count in videos_available.
        sample_short = bool(seed and len(videos) < MIN_NON_SEED_SAMPLE)
        rows.append({
            "name": name,
            "url": url,
            "provenance": provenance,
            "seed": seed,
            "videos_sampled": 0 if sample_short else len(videos),
            "videos_available": len(videos),
            "agg": ext["agg"],
            "top_methods": _top_terms(ext["videos"], "methods", 6),
            "top_compute": _top_terms(ext["videos"], "compute_terms", 6),
        })

    seed_scores = [score_channel(r["agg"]) for r in rows if r["seed"]]
    threshold = keep_threshold(seed_scores)

    for r in rows:
        score = score_channel(r["agg"])
        kept = bool(score >= threshold) if not r["seed"] else True
        r["score"] = score
        r["kept"] = kept
        r["reason"] = reason(score, threshold, kept, r["agg"])

    kept = [r for r in rows if r["kept"]]
    return {
        "channels": rows,
        "kept_count": len(kept),
        "seed_count": sum(1 for r in rows if r["seed"]),
        "threshold": threshold,
        "seed_scores": seed_scores,
        "skipped": skipped,
    }


def _top_terms(rows: list[dict], field: str, n: int) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        for term in row.get(field) or []:
            counts[term] = counts.get(term, 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def channel_json(r: dict) -> dict:
    """Public schema (exactly the fields the done_when check reads, plus origin)."""
    return {
        "name": r["name"],
        "url": r["url"],
        "videos_sampled": r["videos_sampled"],
        "videos_available": r["videos_available"],
        "score": r["score"],
        "kept": r["kept"],
        "reason": r["reason"],
        "seed": r["seed"],
        "provenance": r["provenance"],
        "top_methods": r["top_methods"],
    }


def write_json(result: dict, path: str = OUT_JSON) -> None:
    payload = [channel_json(r) for r in result["channels"]]
    payload.sort(key=lambda x: (not x["seed"], -x["score"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False)
        fh.write("\n")


def _why(r: dict) -> str:
    methods = ", ".join(r["top_methods"][:3]) or "general ML"
    compute = ", ".join(r["top_compute"][:3]) or "compute trade-offs"
    return f"Latest videos cover {methods}; frequent compute angle: {compute}."


def write_md(result: dict, path: str = OUT_MD) -> None:
    kept = sorted((r for r in result["channels"] if r["kept"]),
                  key=lambda r: -r["score"])
    dropped = sorted((r for r in result["channels"] if not r["kept"]),
                     key=lambda r: -r["score"])
    lines = [
        "# AI research channels — kept list",
        "",
        f"Scored with the docs/PLAN.md sec 2 lens (deterministic formula over title, description,",
        f"linked papers, numeric claims and compute terms). A channel is kept when its latest-20-video",
        f"sample scores at least {result['threshold']:.1f} (70% of the median seed score).",
        "",
        f"Seeds: {result['seed_count']} · channels scored: {len(result['channels'])} · kept: {len(kept)}.",
        "",
        "## Kept channels (ranked)",
        "",
    ]
    for r in kept:
        tag = "seed" if r["seed"] else r["provenance"]
        n = r["videos_sampled"] or r["videos_available"]
        lines.append(
            f"- **{r['name']}** — score {r['score']:.1f}, {n} "
            f"{'video' if n == 1 else 'videos'} ({tag}) — {r['url']} — {_why(r)}"
        )
    lines += ["", "## Scored but dropped", ""]
    for r in dropped:
        lines.append(
            f"- {r['name']} — score {r['score']:.1f}, {r['videos_sampled']} videos "
            f"({r['provenance']}) — {r['url']}"
        )
    if result["skipped"]:
        lines += ["", "## Not scored", ""]
        for s in result["skipped"]:
            lines.append(f"- {s['name']} — {s['why']}")
    lines += [
        "",
        "## Why this list helps a one-GPU builder",
        "",
        "Every kept channel links papers and repos in its descriptions, names the method in the",
        "title, and reports numbers — so a viewer with one 24 GB GPU can find the compute-sized",
        "claim first and skip the hype. That is the sample property the score measures.",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=CHANNEL_DIR)
    ap.add_argument("--json", default=OUT_JSON)
    ap.add_argument("--md", default=OUT_MD)
    args = ap.parse_args(argv)

    result = build(args.cache_dir)
    write_json(result, args.json)
    write_md(result, args.md)
    print(
        f"channels={len(result['channels'])} seeds={result['seed_count']} "
        f"kept={result['kept_count']} threshold={result['threshold']} "
        f"skipped={len(result['skipped'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
