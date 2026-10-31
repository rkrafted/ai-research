#!/usr/bin/env python3
"""Train one arm-E specialist (a repo-family LoRA) on the teacher trajectories.

The proven QLoRA trainer lives in the completed agent-distill project; this
wrapper filters *its* teacher.jsonl down to one task family and calls its
``train()`` with absolute paths, writing the adapter into our gitignored
``artifacts/``.  No trainer code is copied, and the sibling repo is only read.

    python3 smallvsbig/train_expert.py --repo job-radar --minutes 90
    python3 smallvsbig/train_expert.py --repo prior-art --minutes 90

Run on the GPU via gpu_jobs/r7-01-train-*.job.yaml.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallvsbig import config  # noqa: E402

DEFAULT_OUT = {"job-radar": "artifacts/expert-job-radar",
               "prior-art": "artifacts/expert-prior-art"}


def filter_records(repo: str, teacher: Path, tasks: Path, out: Path) -> int:
    kinds = {}
    for line in tasks.read_text().splitlines():
        if line.strip():
            t = json.loads(line)
            kinds[t["id"]] = t.get("repo")
    if not teacher.is_file():
        raise SystemExit(f"teacher trajectories missing: {teacher}")
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out.open("w") as fh:
        for line in teacher.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if kinds.get(rec.get("task")) == repo:
                fh.write(line.rstrip("\n") + "\n")
                kept += 1
    return kept


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, choices=["job-radar", "prior-art"])
    ap.add_argument("--minutes", type=float, default=90.0)
    ap.add_argument("--max-length", type=int, default=12288)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    teacher = config.SIBLING / "state" / "teacher.jsonl"
    tasks = config.SIBLING / "state" / "tasks.jsonl"
    filtered = config.STATE / "train" / f"{a.repo}.jsonl"
    kept = filter_records(a.repo, teacher, tasks, filtered)
    print(f"filtered {kept} teacher trajectories for {a.repo} -> {filtered}")
    if kept == 0:
        return 2

    sys.path.insert(0, str(config.SIBLING))
    from distill import train as trainer  # noqa: E402  (GPU stack loaded lazily)

    out = a.out or DEFAULT_OUT[a.repo]
    args = [
        "--base", config.SERA,
        "--records", str(filtered),
        "--out", str(config.REPO_ROOT / out),
        "--max-length", str(a.max_length),
        "--epochs", str(a.epochs),
        "--minutes", str(a.minutes),
        "--lora-r", str(a.lora_r),
        "--lora-alpha", str(a.lora_alpha),
        "--lr", str(a.lr),
    ]
    if a.dry_run:
        args.append("--dry-run")
    return trainer.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
