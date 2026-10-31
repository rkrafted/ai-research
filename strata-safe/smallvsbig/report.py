#!/usr/bin/env python3
"""Render the R7 arm table from state/small-vs-big/*.json.

    python3 smallvsbig/report.py            # markdown to stdout
    python3 smallvsbig/report.py --out docs/small-vs-big.md --section

A data dump has no prose; the session that reads it writes the analysis.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallvsbig import config  # noqa: E402

COLS = ["arm", "n", "passed", "pass_rate", "median_wall_s", "tokens_out_per_task",
        "peak_vram_gb", "gpu_minutes", "passes_per_gpu_hour", "escalations"]


def load() -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(config.STATE / "*.json"))):
        try:
            out.append(json.load(open(f)))
        except json.JSONDecodeError:
            continue
    return out


def render(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(COLS) + " |",
             "|" + "|".join("---" for _ in COLS) + "|"]
    for r in sorted(rows, key=lambda x: str(x.get("arm"))):
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in COLS) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--section", action="store_true", help="append to the file, not overwrite")
    a = ap.parse_args(argv)
    table = render(load())
    if not a.out:
        print(table)
        return 0
    path = Path(a.out)
    if a.section:
        text = path.read_text() if path.exists() else ""
        marker = "## Results table"
        head = text.split(marker)[0].rstrip()
        path.write_text(head + "\n\n" + marker + "\n\n" + table + "\n")
    else:
        path.write_text("# small-vs-big results\n\n" + table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
