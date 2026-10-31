"""The PLAN sec 2 lens as a deterministic formula.

score = weighted sum of six extracted components, each capped, total 0..100.
No LLM, no randomness: ``score_channel(extract_channel(videos)['agg'])`` is a
pure function of the metadata. The keep decision is made against the seed
channels' own scores (a channel is kept if it "scores like the seeds").
"""

from __future__ import annotations

import statistics

WEIGHTS = {
    "methods": 25.0,
    "numbers": 20.0,
    "papers": 20.0,
    "compute": 15.0,
    "evidence": 10.0,
    "captions": 10.0,
}

# cap used to normalise each component; values at/above the cap score full marks
CAPS = {
    "methods": 8.0,
    "numbers": 6.0,
    "papers": 0.5,
    "compute": 5.0,
    "evidence": 5.0,
}


def _pct(value: float, cap: float) -> float:
    return max(0.0, min(value / cap, 1.0))


def components(agg: dict) -> dict:
    """Per-component 0..100 contribution before weighting."""
    return {
        "methods": 100.0 * _pct(agg.get("mean_methods", 0.0), CAPS["methods"]),
        "numbers": 100.0 * _pct(agg.get("mean_numbers", 0.0), CAPS["numbers"]),
        "papers": 100.0 * _pct(agg.get("mean_papers", 0.0), CAPS["papers"]),
        "compute": 100.0 * _pct(agg.get("mean_compute", 0.0), CAPS["compute"]),
        "evidence": 100.0 * _pct(agg.get("mean_evidence", 0.0), CAPS["evidence"]),
        "captions": 100.0 * max(0.0, min(agg.get("caption_ratio", 0.0), 1.0)),
    }


def score_channel(agg: dict) -> float:
    comp = components(agg)
    total = sum(WEIGHTS[k] * comp[k] / 100.0 for k in WEIGHTS)
    return round(total, 2)


def keep_threshold(seed_scores: list[float]) -> float:
    """A channel is kept if its score is close to the seed quality bar.

    70% of the median seed score. A small floor (5) only guards against a
    degenerate all-zero seed set; it is deliberately not a quality floor.
    """
    if not seed_scores:
        return 30.0
    return round(max(5.0, 0.70 * statistics.median(seed_scores)), 2)


def reason(score: float, threshold: float, kept: bool, agg: dict) -> str:
    lead = "kept" if kept else "dropped"
    return (
        f"{lead}: score {score:.1f} vs seed bar {threshold:.1f}; "
        f"~{agg.get('mean_methods', 0):.1f} methods, "
        f"~{agg.get('mean_numbers', 0):.1f} numbers, "
        f"~{agg.get('mean_papers', 0):.2f} paper links/video, "
        f"captions {agg.get('caption_ratio', 0) * 100:.0f}%."
    )
