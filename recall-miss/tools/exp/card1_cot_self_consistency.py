#!/usr/bin/env python3
"""Card 1 — chain of thought vs self-consistency at a fixed GPU-second budget.

docs/experiments/01-cot-vs-self-consistency.md. Rerun on the HARD sales-brain
task (2026-10-03): the easy 5-field notes scored 97% and told us nothing, so the
rows are real call transcripts (or an authored adversarial fallback). Runs in the
GPU sandbox (`ollama: true`) and writes state/exp/01-cot-vs-self-consistency.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MODEL = "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"
N = 60
OUT = "state/exp/01-cot-vs-self-consistency.json"
HYPOTHESIS = ("short CoT + n majority-voted samples beats a single long CoT answer and a "
              "direct answer at an equal token budget on a hard extraction task; the gain "
              "is in sampling, not trace length")
METRIC = "JSON field micro-F1 (%) over 7 deal fields, schema-valid rate, F1 per 1000 generated tokens"

SYS = ("You extract deal facts from a sales-call transcript. The call may be in Latvian; "
       "copy numbers as digits and keep enumerated values in English.")
USER = ("Extract exactly these keys: " + ", ".join(C.HARD_FIELDS) +
        ".\nReturn ONE JSON object. Use null or omit a key that the transcript does not "
        "state. Do not invent values.\n\nTranscript:\n{text}")
DIRECT_SUFFIX = "\nReturn the JSON object now, with no other text."
COT_SUFFIX = ("\nThink step by step in at most three sentences, then end with the JSON "
              "object on its own last line.")


def parse(text):
    d = C.extract_json(text)
    return C.hard_fields(d) if d else {}


def run_arm(cli, name, rows, *, mode, samples, max_tokens, temperature):
    preds, golds, per_item_tokens, per_item_seconds, valid = [], [], [], [], 0
    for row in rows:
        suffix = DIRECT_SUFFIX if mode == "direct" else COT_SUFFIX
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": USER.format(text=row["text"]) + suffix}]
        sample_preds, toks, secs = [], 0, 0.0
        for s in range(samples):
            r = cli.chat(MODEL, msgs, num_predict=max_tokens,
                         temperature=temperature, seed=101 + s, num_ctx=4096)
            toks += r["tokens"]
            secs += r["seconds"]
            parsed = parse(r["text"])
            if parsed:
                valid += 1
            sample_preds.append(parsed)
        merged = {f: C.majority([p.get(f, "") for p in sample_preds]) for f in C.HARD_FIELDS}
        preds.append(merged)
        golds.append(row["gold"])
        per_item_tokens.append(toks)
        per_item_seconds.append(secs)
    f1 = C.micro_f1(preds, golds, C.HARD_FIELDS)
    total_tokens = sum(per_item_tokens)
    total_seconds = sum(per_item_seconds)
    tokens_per_item = total_tokens / max(1, len(rows))
    tps = total_tokens / total_seconds if total_seconds else 0.0
    gain_per_1k = (f1 - 0.0) / (tokens_per_item / 1000.0) if tokens_per_item else 0.0
    return C.arm(name, f1, tps, cli.vram_gb(MODEL),
                 schema_valid_rate=round(100.0 * valid / max(1, len(rows) * samples), 1),
                 tokens_per_item=round(tokens_per_item, 1),
                 f1_per_1k_tokens=round(gain_per_1k, 3),
                 latency_p50_s=round(C.pct(per_item_seconds, 0.50), 2),
                 latency_p95_s=round(C.pct(per_item_seconds, 0.95), 2),
                 n_samples=samples, temperature=temperature)


def main():
    cli = C.Ollama()
    rows, source = C.hard_extraction_fixture(N)
    arms = []
    print(f"[card1] model={MODEL} n={N} fixture={source}", flush=True)
    cli.unload(MODEL)
    arms.append(run_arm(cli, "direct-64tok", rows, mode="direct", samples=1,
                        max_tokens=64, temperature=0.0))
    arms.append(run_arm(cli, "cot-256tok", rows, mode="cot", samples=1,
                        max_tokens=256, temperature=0.0))
    arms.append(run_arm(cli, "cot-4-sample-majority", rows, mode="cot", samples=4,
                        max_tokens=256, temperature=0.7))
    arms.append(run_arm(cli, "cot-8-sample-majority", rows, mode="cot", samples=8,
                        max_tokens=256, temperature=0.7))

    a1, a2 = arms[0], arms[1]
    sc = [a for a in arms if a["name"].startswith("cot-4") or a["name"].startswith("cot-8")]
    beats_direct = any(a["score"] >= a1["score"] + 2.0 for a in sc)
    within_budget = any(a["tokens_per_item"] <= a2["tokens_per_item"] for a in sc)
    better_rate = any(a["f1_per_1k_tokens"] > a2["f1_per_1k_tokens"] for a in sc)
    passed = bool(beats_direct and within_budget and better_rate)
    if a2["score"] <= a1["score"] + 1.0 and not beats_direct:
        passed = False
    return N, arms, passed, {
        "model": MODEL,
        "fixture": source,
        "rule": "pass if a 4/8-sample arm >= direct+2 F1 at <= CoT tokens/item AND beats CoT F1/1k",
        "beats_direct": beats_direct, "within_token_budget": within_budget,
        "better_f1_per_1k": better_rate,
        "note": "hard task rerun 2026-10-03: 7-field deal extraction from sales call transcripts",
    }


if __name__ == "__main__":
    C.guard("01-cot-vs-self-consistency", HYPOTHESIS, METRIC, OUT, main)
