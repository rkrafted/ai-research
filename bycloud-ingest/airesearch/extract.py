"""Deterministic field extraction from video metadata (title + description).

No LLM here: the same text always yields the same fields, so the R1 score is
reproducible offline. Captions, when downloaded to data/private/captions, are
appended to the text by ``text_for`` so the same extractor sees them.

The lexicon is intentionally small and concrete (methods, compute terms,
evidence terms) rather than a general NLP model.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Lexicons (lowercase; matched with word boundaries)
# --------------------------------------------------------------------------
METHODS = [
    "quantization", "quantize", "quantized", "gptq", "awq", "gguf", "bitnet",
    "qlora", "lora", "peft", "adapter", "low-rank",
    "distillation", "distill", "teacher", "student model",
    "mixture of experts", "mixture-of-experts", "moe", "expert routing",
    "speculative decoding", "speculative sampling", "draft model",
    "kv cache", "kv-cache", "flash attention", "attention", "transformer",
    "state space", "state-space", "ssm", "mamba", "recurrent", "looped", "loop",
    "rope", "positional encoding", "positional embedding", "embedding",
    "reinforcement learning", "rlhf", "dpo", "ppo", "grpo", "reward model",
    "chain of thought", "chain-of-thought", "test-time compute", "test time compute",
    "self-consistency", "verifier", "reasoning",
    "synthetic data", "rejection sampling", "data curation", "data mixture",
    "fine-tuning", "finetuning", "fine tuning", "pretraining", "pre-training",
    "instruction tuning", "sft", "continued pretraining", "curriculum learning",
    "model merging", "merging", "task arithmetic", "model soup",
    "pruning", "sparsity", "sparse", "offload", "offloading", "cpu offload",
    "retrieval augmented", "retrieval-augmented", "rag", "vector database",
    "agents", "agentic", "tool use", "function calling",
    "diffusion", "gan", "vae", "latent", "multimodal", "vision language", "vlm",
    "speech", "tts", "whisper", "audio", "vision",
    "tokenizer", "tokenization", "context window", "long context",
    "scaling law", "scaling laws", "gradient checkpointing", "activation checkpointing",
    "deepspeed", "fsdp", "tensor parallel", "data parallel", "vllm", "llama.cpp",
    "gradient", "backprop", "optimizer", "adamw", "learning rate",
]

COMPUTE = [
    "gpu", "vram", "rtx", "3090", "4090", "a100", "h100", "consumer hardware",
    "consumer gpu", "single gpu", "local", "on-device", "on device", "edge",
    "laptop", "memory", "efficient", "efficiency", "small model", "small models",
    "tiny", "parameter", "parameters", "billion", "flops", "throughput",
    "latency", "tokens per second", "cost", "cheap", "budget", "low-resource",
    "low resource", "24gb", "24 gb", "16gb", "8gb", "quantiz",
]

EVIDENCE = [
    "paper", "arxiv", "benchmark", "benchmarks", "ablation", "we measure",
    "results", "github", "code", "reproduce", "experiment", "evaluation",
    "baseline", "state of the art", "sota", "open source", "checkpoint",
]

CAPTION_QUALITY = {"manual": 1.0, "auto": 0.6, "none": 0.0}

_NUM = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|[bmkBMK]\b|billion|million|thousand|"
                  r"gb|tb|tokens?|params?|parameters?|gpus?|hours?|x\b)", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_BARE_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
_ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")
_ARXIV_OLD = re.compile(r"arxiv\.org/(?:abs|pdf)/([a-z.-]+/\d{7})")
_ARXIV_BARE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
_PWC = re.compile(r"paperswithcode\.com/paper/([\w.\-]+)")
_HF = re.compile(r"huggingface\.co/papers/([\w.\-]+)")
_DOI = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")


def _compile(terms):
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b")


_METHODS_RE = _compile(METHODS)
_COMPUTE_RE = _compile(COMPUTE)
_EVIDENCE_RE = _compile(EVIDENCE)


def _match_terms(regex, text):
    return sorted({m.group(1).lower() for m in regex.finditer(text)})


def extract_papers(text: str) -> list[str]:
    """arXiv ids, Papers-with-Code and HF paper slugs, DOIs — deduped, ordered."""
    out: list[str] = []
    for rx in (_ARXIV, _ARXIV_OLD, _ARXIV_BARE, _PWC, _HF, _DOI):
        for m in rx.finditer(text):
            v = m.group(1)
            if v not in out:
                out.append(v)
    return out


def extract_numbers(text: str) -> list[str]:
    """Quantitative tokens, with 4-digit years removed."""
    cleaned = _YEAR.sub(" ", text)
    strong = _NUM.findall(cleaned)
    return [s.strip() for s in strong]


def extract_video(video: dict, caption_text: str | None = None) -> dict:
    title = video.get("title") or ""
    desc = video.get("description") or ""
    text = f"{title}\n{desc}"
    if caption_text:
        text = f"{text}\n{caption_text}"
    low = text.lower()

    methods = _match_terms(_METHODS_RE, low)
    compute = _match_terms(_COMPUTE_RE, low)
    evidence = _match_terms(_EVIDENCE_RE, low)
    numbers = extract_numbers(text)
    papers = extract_papers(text)
    words = max(len(low.split()), 1)
    tech_hits = len(methods) + len(compute) + len(evidence)

    if video.get("captions_manual"):
        captions = "manual"
    elif video.get("captions_auto"):
        captions = "auto"
    else:
        captions = "none"

    return {
        "id": video.get("id"),
        "title": title,
        "methods": methods,
        "compute_terms": compute,
        "evidence_terms": evidence,
        "papers": papers,
        "n_numbers": len(numbers),
        "n_methods": len(methods),
        "n_papers": len(papers),
        "n_compute": len(compute),
        "n_evidence": len(evidence),
        "words": words,
        "tech_density": round(tech_hits / words * 1000, 2),
        "captions": captions,
        "duration": video.get("duration"),
        "upload_date": video.get("upload_date"),
        "view_count": video.get("view_count"),
    }


def extract_channel(videos: list[dict], captions: dict | None = None) -> dict:
    """Extract all sampled videos and aggregate (means, totals)."""
    captions = captions or {}
    rows = [extract_video(v, captions.get(v.get("id"))) for v in videos]
    n = len(rows) or 1
    agg = {
        "n_videos": len(rows),
        "mean_methods": sum(r["n_methods"] for r in rows) / n,
        "mean_numbers": sum(r["n_numbers"] for r in rows) / n,
        "mean_papers": sum(r["n_papers"] for r in rows) / n,
        "mean_compute": sum(r["n_compute"] for r in rows) / n,
        "mean_evidence": sum(r["n_evidence"] for r in rows) / n,
        "mean_density": sum(r["tech_density"] for r in rows) / n,
        "caption_ratio": sum(CAPTION_QUALITY[r["captions"]] for r in rows) / n,
    }
    return {"videos": rows, "agg": agg}


def text_for(video: dict, caption_text: str | None = None) -> str:
    parts = [video.get("title") or "", video.get("description") or ""]
    if caption_text:
        parts.append(caption_text)
    return "\n".join(parts)
