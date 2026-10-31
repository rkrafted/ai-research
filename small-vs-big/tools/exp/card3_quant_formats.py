#!/usr/bin/env python3
"""Card 3 — 4-bit format bake-off for a 9B on one 24 GB card.

docs/experiments/03-quantization-format-bakeoff.md. Offline only the same model
is cached in two formats (BF16 and GGUF Q4_K_M), so the bake-off is FP16 vs
4-bit rather than four formats. Runs in the GPU sandbox (`ollama: true`) and
writes state/exp/03-quantization-format-bakeoff.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

N = 250
OUT = "state/exp/03-quantization-format-bakeoff.json"
HYPOTHESIS = ("a 4-bit GGUF keeps >=97% of BF16 field-F1 while decoding >=1.5x faster "
              "on the same 24 GB card; format differences are speed/VRAM, not quality")
METRIC = "field micro-F1 vs BF16, tokens/s, peak VRAM (GB), cold-load seconds"

FORMATS = [
    ("bf16", "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:BF16"),
    ("q4_k_m", "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"),
]
SYS = "You extract fields from short sales notes. Copy values exactly from the note."
USER = ("Extract exactly these five keys: contact_name, company, budget_usd, timeline, "
        "next_step.\nReturn ONE JSON object and nothing else.\n\nNote:\n{text}")


def max_ctx(cli, model):
    try:
        info = cli._req("/api/show", {"model": model})                    # noqa: SLF001
        for k, v in (info.get("model_info") or {}).items():
            if k.endswith("context_length") and isinstance(v, int):
                return v
    except Exception:                                                     # noqa: BLE001
        pass
    return None


def run_format(cli, label, model, rows):
    cli.unload(model)
    load_msgs = [{"role": "user", "content": "Reply with the single word ok."}]
    import time
    t0 = time.time()
    cli.chat(model, load_msgs, num_predict=1, temperature=0.0, num_ctx=2048)
    load_seconds = time.time() - t0
    vram = cli.vram_gb(model)
    preds, golds, toks, secs, valid = [], [], 0, 0.0, 0
    for row in rows:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": USER.format(text=row["text"])}]
        r = cli.chat(model, msgs, num_predict=96, temperature=0.0, num_ctx=2048)
        toks += r["tokens"]
        secs += r["seconds"]
        d = C.extract_json(r["text"])
        if d:
            valid += 1
        preds.append(C.canonical_fields(d) if d else {})
        golds.append(row["gold"])
    f1 = C.micro_f1(preds, golds)
    tps = toks / secs if secs else 0.0
    return C.arm(label, f1, tps, vram,
                 max_ctx=max_ctx(cli, model) or 0,
                 load_seconds=round(load_seconds, 2),
                 schema_valid_rate=round(100.0 * valid / len(rows), 1),
                 model=model)


def main():
    cli = C.Ollama()
    rows = C.extraction_fixture(N)
    arms = [run_format(cli, label, model, rows) for label, model in FORMATS]
    bf16 = next(a for a in arms if a["name"] == "bf16")
    four = next(a for a in arms if a["name"] == "q4_k_m")
    retained = (four["score"] / bf16["score"] * 100.0) if bf16["score"] else 0.0
    speed_ratio = (four["tokens_per_s"] / bf16["tokens_per_s"]) if bf16["tokens_per_s"] else 0.0
    ctx_ratio = (four["max_ctx"] / bf16["max_ctx"]) if bf16["max_ctx"] else 1.0
    quality_ok = retained >= 97.0
    speed_ok = speed_ratio >= 1.5 or ctx_ratio >= 2.0
    passed = bool(quality_ok and speed_ok)
    return N, arms, passed, {
        "rule": "pass if a 4-bit format keeps >=97% of BF16 F1 AND is >=1.5x tok/s or >=2x context",
        "retained_pct": round(retained, 1), "speed_ratio": round(speed_ratio, 2),
        "ctx_ratio": round(ctx_ratio, 2), "quality_ok": quality_ok, "speed_ok": speed_ok,
        "note": "only BF16 and Q4_K_M of the same 9B are cached offline; GPTQ/AWQ/FP8 need "
                "runtimes (auto-gptq/autoawq/vLLM) that are not in the shared GPU venv",
    }


if __name__ == "__main__":
    C.guard("03-quantization-format-bakeoff", HYPOTHESIS, METRIC, OUT, main)
