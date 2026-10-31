#!/usr/bin/env python3
"""Summarise the R5 trial results.

    python3 tools/upgrades/report.py            # markdown table on stdout
    python3 tools/upgrades/report.py --json     # raw list

Reads only `state/upgrades/*.json` (job-written). Prints nothing but the header
when the GPU jobs have not run yet, so it is safe to call at any time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATE = REPO / "state" / "upgrades"


def load(state_dir: Path | str = STATE) -> list[dict]:
    out = []
    for f in sorted(Path(state_dir).glob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def render(rows: list[dict]) -> str:
    lines = ["| trial | area | workload | change | change_pct | adopt |",
             "|---|---|---|---|---:|---|"]
    for r in rows:
        lines.append(
            f"| {r.get('name', '?')} | {r.get('area', '?')} | {r.get('workload', '?')} | "
            f"{r.get('how', '?')} | {r.get('change_pct')} | {str(r.get('adopt')).lower()} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rows = load()
    if a.json:
        print(json.dumps(rows, indent=2))
    else:
        sys.stdout.write(render(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
