"""Offline tests for the deterministic extraction and scoring (no network)."""

from airesearch.extract import extract_channel, extract_papers, extract_video
from airesearch.score import components, keep_threshold, score_channel


def vid(i, title="", description="", auto=True, manual=False):
    return {
        "id": f"v{i}",
        "title": title,
        "description": description,
        "captions_auto": ["en"] if auto else [],
        "captions_manual": ["en"] if manual else [],
        "duration": 600,
        "upload_date": "20260101",
        "view_count": 1000,
    }


RICH = (
    "LoRA vs QLoRA: we fine-tune a 7B model on a single 24 GB GPU. "
    "Our ablation cuts VRAM by 3.5x and beats the baseline by 8.2%. "
    "Paper: https://arxiv.org/abs/2305.14314 code on GitHub."
)


def test_extract_papers_arxiv():
    assert extract_papers("see https://arxiv.org/abs/2305.14314v2") == ["2305.14314"]


def test_extract_papers_multiple_sources():
    text = ("arxiv.org/pdf/2401.00001 and paperswithcode.com/paper/loRA "
            "and huggingface.co/papers/2310.00002")
    got = extract_papers(text)
    assert "2401.00001" in got and "2310.00002" in got and "loRA" in got


def test_numbers_ignore_years():
    row = extract_video(vid(0, "In 2024 we ran 2 GPUs and gained 12%"))
    assert row["n_numbers"] == 2  # 2 GPUs and 12%, not 2024


def test_extract_video_rich_fields():
    row = extract_video(vid(0, "LoRA vs QLoRA on 24 GB", RICH))
    assert row["n_methods"] >= 2
    assert {"lora", "qlora"} <= set(row["methods"])
    assert row["n_papers"] == 1
    assert row["n_compute"] >= 2
    assert row["captions"] == "auto"
    assert row["tech_density"] > 0


def test_score_monotonic():
    rich = extract_channel([vid(i, "LoRA QLoRA distillation", RICH) for i in range(20)])["agg"]
    empty = extract_channel([vid(i, "a video") for i in range(20)])["agg"]
    assert score_channel(rich) > score_channel(empty)


def test_score_bounds_and_components():
    agg = extract_channel([vid(i, "LoRA on GPU", RICH) for i in range(20)])["agg"]
    s = score_channel(agg)
    assert 0.0 <= s <= 100.0
    assert set(components(agg)) == {"methods", "numbers", "papers", "compute", "evidence", "captions"}


def test_keep_threshold_relative():
    assert keep_threshold([]) == 30.0
    assert keep_threshold([10.0, 12.0]) == 7.7
    assert keep_threshold([80.0, 90.0]) == 59.5
