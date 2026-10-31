"""Render the measured KV sweep as markdown (read-only).

    python3 -m kvfrontier.report            # tables
    python3 -m kvfrontier.report --json     # raw cells

Reads ``state/kvfrontier/results.jsonl`` and ``measured.json`` produced by the
GPU job; prints "not measured yet" rather than inventing a number.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import needles

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state" / "kvfrontier"

#: A3B GGUF metadata: 40 blocks, full attention every 4th -> 10 KV layers,
#: 2 KV heads, head_dim 256 (key and value length).
A3B_KV_LAYERS = 10
A3B_KV_HEADS = 2
A3B_HEAD_DIM = 256
BLOCK_BYTES = {"f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32,
               "q4_1": 20 / 32, "q4_0": 18 / 32}


def kv_bytes_per_token(kt: str, vt: str) -> int:
    return int(A3B_KV_LAYERS * A3B_KV_HEADS * A3B_HEAD_DIM
               * (BLOCK_BYTES[kt] + BLOCK_BYTES[vt]))


def load_cells() -> list[dict]:
    path = STATE / "results.jsonl"
    if not path.is_file():
        return []
    cells = []
    spec = needles.load()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Re-score stored answers with the current scorer so a fixed scoring
        # rule applies to cells measured before the fix.
        if row.get("answer") and row.get("kind") in spec.get("prompts", {}):
            scored = needles.score(row["answer"], spec["prompts"][row["kind"]]["questions"])
            row.update(score=scored["score"], hits=scored["hits"],
                       questions=scored["questions"], misses=scored["misses"])
        cells.append(row)
    return cells


def measured() -> dict:
    path = STATE / "measured.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> int:
    import sys
    cells = load_cells()
    if "--json" in sys.argv:
        print(json.dumps({"cells": cells, "summary": measured()}, indent=1, ensure_ascii=False))
        return 0

    print("### KV size model (A3B: 10 KV layers, 2 KV heads, head_dim 256)\n")
    print("| KV type | bytes/token | 32K | 64K | 128K | 256K |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, kt, vt in (("f16", "f16", "f16"), ("q8_0", "q8_0", "q8_0"),
                         ("q5_1", "q5_1", "q5_1"), ("q4_0", "q4_0", "q4_0")):
        b = kv_bytes_per_token(kt, vt)
        mib = [b * n / (1024 * 1024) for n in (32768, 65536, 131072, 262144)]
        print(f"| {name} | {b} | " + " | ".join(f"{m:.0f} MiB" for m in mib) + " |")

    summary = measured()
    if not cells and not summary:
        print("\n_No GPU results yet: `state/kvfrontier/results.jsonl` is absent._")
        return 0

    print("\n### Fit beside the DFlash draft at one slot\n")
    print("| config | K | V | max ctx that starts | KV MiB | compute MiB | rot K/V |")
    print("|---|---|---|---:|---:|---:|---|")
    for name, cfg in (summary.get("configs") or {}).items():
        kv_mib = cfg.get("kv_total_mib")
        if kv_mib is None and cfg.get("ctk") in BLOCK_BYTES and cfg.get("ctv") in BLOCK_BYTES:
            kv_mib = round(kv_bytes_per_token(cfg["ctk"], cfg["ctv"]) * int(cfg.get("max_ctx") or 0)
                           / (1024 * 1024), 1)
        print(f"| {name} | {cfg.get('ctk')} | {cfg.get('ctv')} | {cfg.get('max_ctx')} | "
              f"{kv_mib if kv_mib is not None else ''} | {cfg.get('compute_buffer_mib', '')} | "
              f"{cfg.get('n_rot_k', '')}/{cfg.get('n_rot_v', '')} |")

    by = defaultdict(dict)
    for c in cells:
        by[(c["config"], c["context"])][c["kind"]] = c
    print("\n### Accuracy and speed on the real mixed contexts\n")
    base = {k: v["answer"] for k, v in by.items() if k[0] == "f16" and "answer" in v}
    print("| config | context | prompt tok | prefill tok/s | decode tok/s | needle | multihop | f16 agree |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    order = {c["id"]: i for i, c in enumerate(measured().get("contexts", []))}
    for (config, context), kinds in sorted(by.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 0))):
        any_cell = next(iter(kinds.values()))
        agree = [needles.agreement(c["answer"], base[(config, context)]["answer"])
                 for kind, c in kinds.items()
                 if config != "f16" and (config, context) in base and c.get("answer")]
        agree_s = ("—" if config == "f16" or not agree
                   else f"{sum(agree)}/{len(agree)}")
        print(f"| {config} | {context} | {any_cell.get('prompt_tokens', '')} | "
              f"{any_cell.get('prefill_tok_s', '')} | {any_cell.get('decode_tok_s', '')} | "
              f"{kinds.get('needle', {}).get('score', '—')} | "
              f"{kinds.get('multihop', {}).get('score', '—')} | {agree_s} |")
    errors = [c for c in cells if "error" in c]
    if errors:
        print(f"\n_{len(errors)} cell(s) errored; see results.jsonl._")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
