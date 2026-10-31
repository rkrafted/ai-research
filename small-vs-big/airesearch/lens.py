"""The PLAN sec 2 lens ("gains under limits") as a deterministic ranker.

Reads ``data/kb.sqlite`` (R2) and produces, with **no LLM and no randomness**:

* a score per method/claim cluster, broken into seven visible components
  (capability per parameter, per FLOP/VRAM, test-time compute, data
  efficiency, cost to try, evidence quality, transfer); and
* ``docs/digest.md`` plus the machine-readable ``data/lens.json``.

Clustering is a canonicalisation of the model-extracted ``methods.method``
strings: lower-case, strip parentheticals/punctuation, apply a small synonym
map, drop umbrella/product terms. A cluster's evidence is every claim, paper
link and repo link of its member videos.

The score is the PLAN formula made explicit::

    gain      = 0.30*param + 0.30*flop + 0.25*test_time + 0.15*data_eff
    transfer  = 0..100 from how many of our work areas the cluster touches
    cost      = estimated 3090 GPU-hours for one small reproduction
    evidence  = 0..100 from papers, repos, numbers and channel diversity
    priority  = gain * (0.4 + 0.6*transfer/100) / max(cost, 0.25)
                * (0.55 + 0.45*evidence/100)

Every sub-component is returned in ``score()`` and written to ``lens.json``,
so a reviewer can recompute the rank with a pencil.

Usage:
    python -m airesearch.lens build      # -> data/lens.json + docs/digest.md
    python -m airesearch.lens top 50     # print the ranked list
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .kb import DB
from .taxonomy import FAMILY, PROFILES

REPO = Path(__file__).resolve().parent.parent
LENS_JSON = REPO / "data" / "lens.json"
DIGEST = REPO / "docs" / "digest.md"

# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------

# umbrella terms, product names and single-company models: recall, not a
# technique you can try on a 3090. Dropped before scoring.
STOP = {
    "llm", "llms", "large language models", "large language model",
    "language model", "language models", "ai", "artificial intelligence",
    "agi", "gen ai", "generative ai", "machine learning", "deep learning",
    "neural networks", "neural network", "gpt", "gpt 3", "gpt 4", "gpt 4o",
    "gpt 4 turbo", "gpt 5", "chatgpt", "o1", "o3", "claude",
    "claude 3 5 sonnet", "claude 3 7 sonnet", "claude 4", "gemini",
    "gemini 1 5 pro", "gemini 2 0", "sora", "deepseek", "deepseek r1",
    "llama", "llama 3", "mistral", "qwen", "nvidia nim", "nvidia",
    "google", "openai", "anthropic", "meta", "microsoft", "cohere",
    "runway", "midjourney", "vibe coding", "deep research", "ai safety",
    "ai news", "roadmap", "research", "futurism", "prompt to image",
    "gpt 3", "gpt 3 5", "gpt 3.5", "gpt-3", "gpt-4.5", "o3", "o4",
    "kimi k2 thinking", "sam audio", "flux", "devin ai", "cursor ai",
    "dots", "ucp", "nvidia brev launchable", "rapids cudf", "llmperf",
}

# many surface forms for one technique; applied after normalisation
ALIASES = {
    "retrieval augmented generation": "rag",
    "retrieval-augmented generation": "rag",
    "large language models": "llm",
    "transformers": "transformer",
    "chain of thought": "chain of thought",
    "chain-of-thought": "chain of thought",
    "cot": "chain of thought",
    "mixture of experts moe": "mixture of experts",
    "moe": "mixture of experts",
    "reinforcement learning from human feedback": "rlhf",
    "supervised fine tuning": "sft",
    "fine tuning": "fine-tuning",
    "finetuning": "fine-tuning",
    "quantisation": "quantization",
    "quantized": "quantization",
    "4 bit quantization": "quantization",
    "gguf quantization": "quantization",
    "low rank adaptation": "lora",
    "parameter efficient fine tuning": "peft",
    "text to speech": "tts",
    "speech synthesis": "tts",
    "voice cloning": "tts",
    "diffusion models": "diffusion",
    "stable diffusion": "diffusion",
    "test time compute": "test-time compute",
    "test-time": "test-time compute",
    "self consistency": "self-consistency",
    "context engineering": "context engineering",
    "tool calling": "tool use",
    "function calling": "tool use",
    "ai agents": "agents",
    "agent": "agents",
    "coding agents": "agents",
    "mechanistic interpretability": "interpretability",
    "sparse autoencoders": "interpretability",
    "linear attention": "linear attention",
    "state space models": "ssm",
    "mamba": "ssm",
    "world models": "world models",
    "continual learning": "continual learning",
}


def canonical(method: str) -> str:
    """Normalise one extracted method string to a cluster key."""
    m = method.lower().strip()
    m = re.sub(r"\(.*?\)", " ", m)
    m = re.sub(r"[^a-z0-9+#/\- ]", " ", m)
    m = re.sub(r"\s+", " ", m).strip(" -")
    if not m:
        return ""
    if m in ALIASES:
        return ALIASES[m]
    # strip a trailing plural only when the singular is a known technique
    if m.endswith("s") and m[:-1] in ALIASES:
        return ALIASES[m[:-1]]
    return m


# ---------------------------------------------------------------------------
# Deterministic signals (substring match on the lower-cased cluster text)
# ---------------------------------------------------------------------------

SIGNALS = {
    "param": [
        "distill", "quantiz", "quantis", "qlora", "lora", "peft",
        "pruning", "prune", "parameter-efficient", "parameter efficient",
        "weight sharing", "weight tying", "recurrent", "looped", "loop",
        "mixture of experts", "moe", "sparse", "merging", "merge",
        "small model", "small language model", "slm", "1b", "2b", "3b",
        "7b", "8b", "4b", "fewer parameters", "fixed size", "depth reuse",
        "speculative decoding", "heads", "vocabulary",
        "post-training", "post training", "fine-tuning", "finetuning",
        "sft", "dpo", "grpo", "rlhf", "reinforcement learning",
        "reward model", "preference", "adapter", "compression", "quantized",
        "tokeniz", "embedding", "self-training", "self training",
    ],
    "flop": [
        "quantiz", "quantis", "4-bit", "4 bit", "8-bit", "8 bit", "int4",
        "int8", "gguf", "qlora", "lora", "offload", "kv cache", "kv-cache",
        "flash attention", "speculative decoding", "speculative", "prune",
        "pruning", "distill", "mixture of experts", "moe", "sparse",
        "gradient checkpointing", "unsloth", "vllm", "batched",
        "context compression", "long context", "linear attention", "ssm",
        "parallel", "throughput", "latency", "memory", "24 gb", "gpu",
        "rag", "retrieval", "vector search", "faiss", "semantic search",
        "embedding", "knowledge graph", "caching", "cache", "paged",
        "disaggregat", "prefill", "kernel", "fp16", "fp8", "attention",
        "quantization-aware", "token", "batch size", "memory management",
        "compression", "pruning", "distillation", "merging",
    ],
    "test_time": [
        "test-time", "test time", "self-consistency", "self consistency",
        "verifier", "verification", "best-of-n", "best of n",
        "chain of thought", "reasoning", "latent reasoning", "looped",
        "loop", "reflection", "search", "tree of thought", "mcts",
        "majority vote", "thinking", "inference", "sampling", "speculat",
        "self-refine", "self refine", "revision", "critique",
        "process reward", "reward model", "scaling", "compute",
        "gradient descent", "activation", "probing", "steering",
    ],
    "data_eff": [
        "synthetic data", "synthetic", "rejection sampling", "distill",
        "curriculum", "self-play", "self play", "dagger",
        "imitation learning", "few-shot", "few shot", "data efficien",
        "data curation", "data selection", "active learning", "student",
        "teacher", "continual learning", "synthetic data generation",
        "preference", "rlhf", "dpo", "grpo", "reward", "self-training",
        "self training", "unlabeled", "annotation", "labeling", "labelling",
        "knowledge distillation", "data augmentation", "augmentation",
    ],
}

# our own projects (PLAN sec 1): a cluster that touches more of them transfers more
TRANSFER_AREAS = {
    "distill": [
        "distill", "quantiz", "qlora", "lora", "prune", "small model",
        "speculative", "merging", "peft",
    ],
    "extraction": [
        "extraction", "structured output", "json", "tool use",
        "function calling", "information extraction", "rag",
        "retrieval", "rerank", "embeddings",
    ],
    "agents": [
        "agents", "tool use", "planning", "mcp", "context engineering",
        "react", "workflow",
    ],
    "speech": [
        "speech", "voice", "tts", "asr", "audio", "whisper",
    ],
    "vision": [
        "vision", "image", "video", "diffusion", "vit", "segmentation",
        "multimodal",
    ],
    "eval": [
        "evaluation", "benchmark", "eval", "interpretability",
        "mechanistic", "red team", "prompt injection",
    ],
    "training": [
        "fine-tuning", "training", "post-training", "rlhf", "dpo",
        "grpo", "sft", "pre-training", "pretraining", "reinforcement learning",
        "curriculum",
    ],
}

# base 3090 GPU-hours to reproduce one small claim, by technique family
COST_BASE = [
    (12.0, ["pre-training", "pretraining", "pretrain", "scaling laws",
            "from scratch", "world models"]),
    (6.0, ["rlhf", "dpo", "grpo", "reinforcement learning", "ppo", "rl "]),
    (3.0, ["fine-tuning", "finetuning", "sft", "lora", "qlora", "peft",
           "distill", "quantization", "quantis"]),
    (1.5, ["pruning", "merging", "merge", "quantization", "gguf",
           "speculative", "vllm", "flash attention", "ssm",
           "linear attention"]),
    (0.5, ["rag", "retrieval", "prompt", "context engineering", "embeddings",
           "vector search", "tool use", "agents", "self-consistency",
           "prompt injection", "interpretability"]),
]

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")
_GB_RE = re.compile(r"(\d+)\s*GB", re.IGNORECASE)


def _key_text(cl: "Cluster") -> str:
    """The technique's own surface forms — where the axis signals are counted.

    Claims are deliberately excluded here: the R2 extractor attaches a video's
    whole transcript to every method it named, so counting signals in claims
    turned "Attention" into a keyword soup (see PROGRESS). Claims still supply
    the evidence counts (numbers, papers, repos, channels) and the links.
    """
    return " ".join(cl.methods.keys()).lower()


def _context_text(cl: "Cluster") -> str:
    """Key surface forms + video titles; used only for transfer and cost."""
    return (_key_text(cl) + " " + " ".join(cl.titles.values())).lower()


def _hits(text: str, terms: list[str]) -> int:
    return sum(1 for t in terms if t in text)


def _saturate(hits: int, cap: int) -> float:
    """0..100, saturating at ``cap`` distinct hits (sub-linear after half)."""
    if hits <= 0:
        return 0.0
    if hits >= cap:
        return 100.0
    return round(100.0 * math.sqrt(hits / cap), 2)


def estimate_hours(cl: "Cluster") -> float:
    """Deterministic 3090 GPU-hour estimate for one small reproduction.

    A taxonomy family declares its own cost. Unknown families fall back to the
    technique key + titles (never every claim), family base scaled by the
    largest model size mentioned.
    """
    entry = PROFILES.get(cl.key)
    if entry:
        return entry[4]
    text = _context_text(cl)
    base = 2.0
    for hours, terms in COST_BASE:
        if any(t in text for t in terms):
            base = hours
            break
    params = [float(x) for x in _SIZE_RE.findall(text)]
    biggest = max(params) if params else 0.0
    if biggest <= 3:
        size = 1.0
    elif biggest <= 9:
        size = 1.6
    elif biggest <= 40:
        size = 3.0
    elif biggest > 40:
        size = 5.0
    else:
        size = 1.2
    return round(base * size, 2)


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    key: str
    methods: Counter = field(default_factory=Counter)   # surface form -> count
    videos: set = field(default_factory=set)
    channels: set = field(default_factory=set)
    titles: dict = field(default_factory=dict)          # video id -> title
    claims: list = field(default_factory=list)          # {video_id, claim, numbers, timestamp}
    papers: dict = field(default_factory=dict)          # arxiv id -> title
    links: set = field(default_factory=set)             # (url, kind)

    @property
    def label(self) -> str:
        """Prettiest surface form: the most common original spelling."""
        return self.methods.most_common(1)[0][0]


def load_clusters(conn: sqlite3.Connection) -> list[Cluster]:
    """Group the KB's methods (and their videos' evidence) into clusters."""
    clusters: dict[str, Cluster] = {}
    vid_title: dict[str, str] = {}
    vid_channel: dict[str, str] = {}
    for vid, ch, title in conn.execute("SELECT id, channel, title FROM videos"):
        vid_title[vid] = title or vid
        vid_channel[vid] = ch or ""

    for vid, method in conn.execute("SELECT video_id, method FROM methods"):
        key = canonical(method)
        if not key or key in STOP:
            continue
        fam = FAMILY.get(key, key)
        cl = clusters.setdefault(fam, Cluster(key=fam))
        cl.methods[method.strip()] += 1
        cl.videos.add(vid)
        if vid in vid_channel and vid_channel[vid]:
            cl.channels.add(vid_channel[vid])
        cl.titles[vid] = vid_title.get(vid, vid)

    if not clusters:
        return []

    # reverse index video -> clusters, so the evidence scans stay linear
    vid_clusters: dict[str, list[Cluster]] = defaultdict(list)
    for cl in clusters.values():
        for vid in cl.videos:
            vid_clusters[vid].append(cl)

    for vid, claim, numbers, ts in conn.execute(
        "SELECT video_id, claim, numbers, timestamp FROM claims"
    ):
        for cl in vid_clusters.get(vid, ()):
            cl.claims.append({
                "video_id": vid, "claim": claim or "",
                "numbers": numbers or "", "timestamp": ts or "",
            })

    for vid, arxiv, title in conn.execute(
        "SELECT video_id, arxiv_id, title FROM papers"
    ):
        for cl in vid_clusters.get(vid, ()):
            cl.papers.setdefault(arxiv, title or "")

    for vid, url, kind in conn.execute(
        "SELECT video_id, url, kind FROM links"
    ):
        for cl in vid_clusters.get(vid, ()):
            cl.links.add((url, kind or "other"))

    return list(clusters.values())


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(cl: Cluster) -> dict:
    """Return the seven components and the combined priority for one cluster.

    A taxonomy family supplies the four gain axes and the cost directly; an
    unknown family falls back to keyword signals over the technique key.
    Evidence is always counted from the KB, so two families with the same
    profile still differ by how well the corpus supports them.
    """
    entry = PROFILES.get(cl.key)
    if entry:
        param, flop, test_time, data_eff, hours, _theme, _why, areas = entry
        areas = list(areas)
    else:
        key_text = _key_text(cl)
        param = _saturate(_hits(key_text, SIGNALS["param"]), 8)
        flop = _saturate(_hits(key_text, SIGNALS["flop"]), 8)
        test_time = _saturate(_hits(key_text, SIGNALS["test_time"]), 7)
        data_eff = _saturate(_hits(key_text, SIGNALS["data_eff"]), 6)
        context = _context_text(cl)
        areas = [name for name, terms in TRANSFER_AREAS.items()
                 if any(t in context for t in terms)]
        hours = estimate_hours(cl)

    gain = round(0.30 * param + 0.30 * flop + 0.25 * test_time + 0.15 * data_eff, 2)

    transfer = round(100.0 * (0.6 * min(len(areas) / 3.0, 1.0)
                              + 0.4 * min(len(cl.videos) / 8.0, 1.0)), 2)

    n_papers = len(cl.papers)
    n_repos = len({u for u, k in cl.links if k in ("github", "huggingface")})
    n_numbers = sum(1 for c in cl.claims if c["numbers"])
    n_channels = len(cl.channels)
    evidence = round(
        100.0 * (0.35 * min(n_papers / 3.0, 1.0)
                 + 0.25 * min(n_repos / 3.0, 1.0)
                 + 0.20 * min(n_numbers / 15.0, 1.0)
                 + 0.20 * min(n_channels / 8.0, 1.0)), 2)

    priority = round(
        gain * (0.40 + 0.60 * transfer / 100.0)
        / math.sqrt(max(hours, 0.25))
        * (0.55 + 0.45 * evidence / 100.0) * 10.0, 3)

    return {
        "key": cl.key, "label": cl.label, "profiled": entry is not None,
        "videos": len(cl.videos), "channels": n_channels,
        "claims": len(cl.claims), "papers": n_papers, "repos": n_repos,
        "areas": sorted(areas),
        "components": {
            "capability_per_parameter": param,
            "capability_per_flop_vram": flop,
            "test_time_compute": test_time,
            "data_efficiency": data_eff,
            "evidence_quality": evidence,
            "transfer": transfer,
            "cost_to_try": hours,
        },
        "gain": gain, "priority": priority,
    }


def eligible(s: dict) -> bool:
    """A cluster must be actionable and at least weakly evidenced."""
    c = s["components"]
    return (
        s["videos"] >= 3
        and (s["profiled"] or s["gain"] >= 50.0)
        and s["gain"] >= 25.0
        and (s["papers"] >= 1 or s["repos"] >= 2 or s["claims"] >= 5)
        and max(c["capability_per_parameter"], c["capability_per_flop_vram"],
                c["test_time_compute"], c["data_efficiency"]) >= 50.0
    )


def rank(conn: sqlite3.Connection) -> list[dict]:
    rows = [score(cl) for cl in load_clusters(conn)]
    rows = [r for r in rows if eligible(r)]
    rows.sort(key=lambda r: (-r["priority"], r["key"]))
    return rows


# ---------------------------------------------------------------------------
# Digest rendering
# ---------------------------------------------------------------------------

def _ts_seconds(ts: str) -> int:
    ts = ts.strip().strip("[]")
    if not ts:
        return 0
    parts = [p for p in ts.split(":") if p != ""]
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return 0
    sec = 0
    for n in nums:
        sec = sec * 60 + n
    return sec


def _evidence_links(cl: Cluster, s: dict) -> tuple[str, list[str]]:
    """(youtube link or '', arxiv urls) best evidence for one cluster."""
    best = None
    for c in sorted(cl.claims, key=lambda c: (not c["timestamp"], c["video_id"])):
        if c["timestamp"]:
            sec = _ts_seconds(c["timestamp"])
            url = f"https://www.youtube.com/watch?v={c['video_id']}"
            if sec:
                url += f"&t={sec}s"
            best = url
            break
    if best is None:
        for vid in sorted(cl.videos):
            best = f"https://www.youtube.com/watch?v={vid}"
            break
    arxiv = [f"https://arxiv.org/abs/{a}" for a in sorted(cl.papers)]
    return (best or ""), arxiv


def render_digest(rows: list[dict], clusters: dict[str, Cluster], top: int = 50) -> str:
    """Markdown digest: top-N items grouped by theme + the paper index."""
    def theme_of(key: str) -> str:
        entry = PROFILES.get(key)
        return entry[5] if entry else "Other techniques"

    def why_of(key: str) -> str:
        entry = PROFILES.get(key)
        return entry[6] if entry else ""

    shown = rows[:top]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in shown:
        grouped[theme_of(r["key"])].append(r)

    lines = [
        "# Gains under limits — top 50 for one RTX 3090",
        "",
        f"Deterministic lens (PLAN §2) over `data/kb.sqlite`: "
        f"**{len(rows)} eligible method clusters** from "
        f"{sum(r['videos'] for r in rows)} method–video links; no LLM and no "
        f"randomness in the score. Full component breakdown: `data/lens.json`.",
        "",
        "Every item is one canonical method cluster with its evidence links "
        "(YouTube timestamp + arXiv). The score is "
        "`gain × transfer ÷ cost × evidence`; the axis values are in the "
        "component table at the end.",
        "",
    ]
    order = []
    for r in shown:
        theme = theme_of(r["key"])
        if theme not in order:
            order.append(theme)

    for theme in order:
        lines.append(f"## {theme}")
        lines.append("")
        for r in grouped[theme]:
            cl = clusters[r["key"]]
            why = why_of(r["key"]) or _fallback_why(r)
            yt, arxiv = _evidence_links(cl, r)
            link_bits = []
            if yt:
                link_bits.append(f"[video]({yt})")
            link_bits += [f"[arXiv {u.rsplit('/', 1)[-1]}]({u})" for u in arxiv[:2]]
            links = " · ".join(link_bits) if link_bits else "KB entry only"
            lines.append(
                f"- **{r['label']}** — {why} "
                f"`priority {r['priority']:.1f}` · evidence: {links}"
            )
        lines.append("")

    # paper index: every distinct arXiv id attached to the shown clusters
    papers: dict[str, str] = {}
    for r in shown:
        for a, t in clusters[r["key"]].papers.items():
            papers.setdefault(a, t)
    lines += ["## Paper index (cited by the items above)", ""]
    for a in sorted(papers):
        title = papers[a] or a
        lines.append(f"- [{title}](https://arxiv.org/abs/{a}) — arXiv:{a}")
    lines.append("")

    # every component, visible, for the ranked items
    lines += [
        "## Component table",
        "",
        "`param` = capability per parameter, `flop` = per FLOP/VRAM, "
        "`ttc` = test-time compute, `data` = data efficiency, "
        "`ev` = evidence quality, `tr` = transfer, `cost` = 3090 GPU-hours.",
        "",
        "| # | cluster | param | flop | ttc | data | ev | tr | cost | gain | priority |",
        "|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for i, r in enumerate(shown, 1):
        c = r["components"]
        lines.append(
            f"| {i} | {r['label']} | {c['capability_per_parameter']:.0f} "
            f"| {c['capability_per_flop_vram']:.0f} "
            f"| {c['test_time_compute']:.0f} | {c['data_efficiency']:.0f} "
            f"| {c['evidence_quality']:.0f} | {c['transfer']:.0f} "
            f"| {c['cost_to_try']:.2f} | {r['gain']:.1f} | {r['priority']:.1f} |"
        )
    lines.append("")

    lines += [
        "## How the score is computed",
        "",
        "`gain = 0.30·parameter + 0.30·FLOP/VRAM + 0.25·test-time + 0.15·data`;",
        "`priority = gain × (0.40 + 0.60·transfer/100) ÷ √max(cost_hours, 0.25) "
        "× (0.55 + 0.45·evidence/100) × 10`.",
        "",
        "The square root keeps cheap-to-try techniques ahead without letting a "
        "0.25 h prompt tweak outrank a 2 h QLoRA run by 24x.",
        "",
        "Each profiled technique gets its four gain axes and its cost from the "
        "visible taxonomy in `airesearch/taxonomy.py` (change one integer and "
        "rebuild); an unknown family falls back to counting curated signals in "
        "its method strings, saturated at a declared cap. Evidence is always "
        "counted from the KB: arXiv papers, GitHub/Hugging Face links, numeric "
        "claims and channel diversity.",
        "",
        "Caveat: the R2 extractor is noisy (see `docs/spot-checks.md`); "
        "several claims attached to a cluster are only co-located with it. "
        "The digest is a recall-first reading list, not a verified result.",
        "",
    ]
    return "\n".join(lines)


def _fallback_why(r: dict) -> str:
    c = r["components"]
    axes = [
        ("fixed-size capability", c["capability_per_parameter"]),
        ("FLOP/VRAM efficiency", c["capability_per_flop_vram"]),
        ("test-time compute", c["test_time_compute"]),
        ("data efficiency", c["data_efficiency"]),
    ]
    top = ", ".join(n for n, _ in sorted(axes, key=lambda x: -x[1])[:2])
    return (
        f"touches {top}; ~{c['cost_to_try']:.1f} GPU-h to try on the 3090, "
        f"evidence {c['evidence_quality']:.0f}/100 across {r['videos']} videos"
    )


def build(db: Path = DB, out_json: Path = LENS_JSON, out_md: Path = DIGEST) -> dict:
    conn = sqlite3.connect(db)
    try:
        rows = rank(conn)
        clusters = {c.key: c for c in load_clusters(conn)}
    finally:
        conn.close()
    top = rows[:50]
    payload = {
        "generated_by": "airesearch.lens",
        "n_eligible": len(rows),
        "top": [
            {
                **r,
                "youtube": _evidence_links(clusters[r["key"]], r)[0],
                "arxiv": _evidence_links(clusters[r["key"]], r)[1],
            }
            for r in top
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    out_md.write_text(render_digest(rows, clusters, top=50))
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PLAN sec 2 lens ranker + digest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p = sub.add_parser("top")
    p.add_argument("n", type=int, default=50, nargs="?")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        payload = build()
        print(f"eligible clusters: {payload['n_eligible']}")
        for i, r in enumerate(payload["top"], 1):
            print(f"{i:2d}. {r['priority']:7.2f}  {r['label']}")
        return 0
    conn = sqlite3.connect(DB)
    try:
        rows = rank(conn)
    finally:
        conn.close()
    for i, r in enumerate(rows[: args.n], 1):
        print(f"{i:2d}. {r['priority']:7.2f}  {r['label']}  "
              f"(videos={r['videos']}, papers={r['papers']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
