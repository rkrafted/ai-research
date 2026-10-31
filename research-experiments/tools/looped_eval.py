#!/usr/bin/env python3
"""3090 looped-LM evaluation — the ONLY writer of state/looped-eval.json.

Queued as gpu_jobs/looped-eval.job.yaml. Runs offline against the HF cache populated by
fetch/*.yaml. Scores benchmark/loop-eval-v1.jsonl (200 four-way multiple-choice items) by
length-normalised log-likelihood at several `total_ut_steps` values for the Ouro checkpoints,
plus a same-size non-looped Qwen3-1.7B baseline.

Compatibility: the Ouro repos ship custom `modeling_ouro.py` written for transformers 4.54;
the shared 3090 venv is transformers 5.17. We install three shims before importing the remote
code (verified on CPU): pad_token_id on the config, `rope_scaling={'rope_type':'linear',
'factor':1.0}`, and a `create_causal_mask` wrapper that maps the removed `input_embeds` /
`cache_position` arguments. If a checkpoint still fails to load, we fall back to the documented
"repeat a middle layer" experiment on Qwen3-1.7B and label those runs accordingly.

Usage: python3 tools/looped_eval.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark" / "loop-eval-v1.jsonl"
OUT = ROOT / "state" / "looped-eval.json"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

FEWSHOT = (
    "Question: A shop had 12 apples. It received 8 more, then sold 5. "
    "How many apples are left?\nAnswer: 15\n\n"
    "Question: What is the capital of France?\nAnswer: Paris\n\n"
)

# (repo_id, loop counts) — loop count is `total_ut_steps` for Ouro, always 1 for Qwen.
OURO_RUNS = [
    ("ByteDance/Ouro-2.6B", [1, 2, 4, 8]),
    ("ByteDance/Ouro-1.4B", [1, 4]),
    ("ByteDance/Ouro-2.6B-Thinking", [1, 4]),
]
BASELINE = ("Qwen/Qwen3-1.7B", [1])
FALLBACK_REPO = "Qwen/Qwen3-1.7B"


# ------------------------------------------------------------------------- transformers 5.x shims
def install_shims() -> None:
    import transformers.masking_utils as mu

    orig_causal = mu.create_causal_mask

    def patched_causal(config, input_embeds=None, inputs_embeds=None, attention_mask=None,
                       cache_position=None, past_key_values=None, position_ids=None, **kw):
        if inputs_embeds is None:
            inputs_embeds = input_embeds
        return orig_causal(config=config, inputs_embeds=inputs_embeds,
                           attention_mask=attention_mask, past_key_values=past_key_values,
                           position_ids=position_ids, **kw)

    mu.create_causal_mask = patched_causal
    if hasattr(mu, "create_sliding_window_causal_mask"):
        orig_sliding = mu.create_sliding_window_causal_mask

        def patched_sliding(config, input_embeds=None, inputs_embeds=None, attention_mask=None,
                            cache_position=None, past_key_values=None, position_ids=None, **kw):
            if inputs_embeds is None:
                inputs_embeds = input_embeds
            return orig_sliding(config=config, inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask, past_key_values=past_key_values,
                                position_ids=position_ids, **kw)

        mu.create_sliding_window_causal_mask = patched_sliding


# ---------------------------------------------------------------------------------------- scoring
def _build_benchmark_in_memory() -> list[dict]:
    """Rebuild the benchmark from its committed generator (no raw data in the repo)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_benchmark", ROOT / "tools" / "make_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def load_benchmark() -> list[dict]:
    if BENCH.exists():
        items = [json.loads(line) for line in BENCH.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        items = _build_benchmark_in_memory()
    if len(items) < 200:
        raise RuntimeError(f"benchmark too small: {len(items)} items in {BENCH}")
    for it in items:
        assert len(it["choices"]) == 4 and 0 <= it["answer"] < 4, it
    return items


def load_ouro(repo: str, loops: int, torch):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True, local_files_only=True)
    cfg.pad_token_id = 0
    cfg.rope_scaling = {"rope_type": "linear", "factor": 1.0}
    cfg.total_ut_steps = int(loops)
    cfg.early_exit_threshold = 1.0
    model = AutoModelForCausalLM.from_pretrained(
        repo, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cuda", local_files_only=True,
    )
    return model, cfg


def load_plain(repo: str, torch):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True,
    )
    return model, model.config


def evaluate(model, tok, items, torch) -> dict:
    device = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    def score(body: str, choices: list[str]):
        body_ids = tok(body, add_special_tokens=True).input_ids
        cands = [tok(" " + c, add_special_tokens=False).input_ids for c in choices]
        maxc = max(len(c) for c in cands)
        rows = [body_ids + c + [pad] * (maxc - len(c)) for c in cands]
        mask = [[1] * (len(body_ids) + len(c)) + [0] * (maxc - len(c)) for c in cands]
        input_ids = torch.tensor(rows, device=device)
        attn = torch.tensor(mask, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        scores = []
        for r, c in enumerate(cands):
            if not c:
                scores.append(float("-inf"))
                continue
            s = 0.0
            for i, tokid in enumerate(c):
                pos = len(body_ids) + i
                s += logp[r, pos - 1, tokid].item()
            scores.append(s / len(c))
        return scores, sum(len(body_ids) + len(c) for c in cands)

    correct = {t: 0 for t in ("arithmetic", "knowledge")}
    total = {t: 0 for t in ("arithmetic", "knowledge")}
    tokens = 0
    t0 = time.time()
    for it in items:
        body = FEWSHOT + "Question: " + it["question"] + "\nAnswer:"
        scores, ntok = score(body, it["choices"])
        tokens += ntok
        total[it["task"]] += 1
        if max(range(len(scores)), key=lambda i: scores[i]) == it["answer"]:
            correct[it["task"]] += 1
    elapsed = time.time() - t0
    n = sum(total.values())
    return {
        "accuracy": sum(correct.values()) / n,
        "task_accuracy": {t: (correct[t] / total[t] if total[t] else 0.0) for t in total},
        "tokens_per_s": tokens / elapsed if elapsed > 0 else 0.0,
        "vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        "n": n,
    }


# ------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="only the 2.6B base + baseline")
    args = ap.parse_args()

    install_shims()
    import torch  # noqa: F401  (after shims so remote code is patched)

    try:
        from transformers import AutoTokenizer
    except Exception as e:  # pragma: no cover
        print("FATAL: transformers unavailable:", e, file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device visible to the job", file=sys.stderr)
        return 1

    items = load_benchmark()
    print(f"[bench] {BENCH.name}: {len(items)} items "
          f"({sum(1 for i in items if i['task']=='arithmetic')} arithmetic, "
          f"{sum(1 for i in items if i['task']=='knowledge')} knowledge)")

    runs: list[dict] = []
    failures: list[str] = []
    ouro_runs = OURO_RUNS[:1] if args.quick else OURO_RUNS
    planned = [(r, l, "ouro") for r, ls in ouro_runs for l in ls]
    planned += [(BASELINE[0], 1, "plain")]

    for repo, loops, kind in planned:
        label = f"{repo} (loops={loops})"
        try:
            print(f"[load] {label}", flush=True)
            if kind == "ouro":
                model, cfg = load_ouro(repo, loops, torch)
            else:
                model, cfg = load_plain(repo, torch)
            model.eval()
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=(kind == "ouro"))
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token or tok.unk_token
            torch.cuda.reset_peak_memory_stats()
            res = evaluate(model, tok, items, torch)
            runs.append({"model": repo, "loops": int(loops), "accuracy": round(res["accuracy"], 4),
                         "tokens_per_s": round(res["tokens_per_s"], 1),
                         "vram_gb": round(res["vram_gb"], 2),
                         "task_accuracy": {k: round(v, 4) for k, v in res["task_accuracy"].items()}})
            print(f"[ok]   {label}: acc={res['accuracy']:.3f} "
                  f"tok/s={res['tokens_per_s']:.0f} vram={res['vram_gb']:.2f}GB", flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            failures.append(f"{label}: {type(e).__name__}: {e}")
            print(f"[fail] {label}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    # Fallback: no Ouro run succeeded -> documented middle-layer-repeat experiment on Qwen3-1.7B.
    if not runs:
        print("[fallback] all Ouro loads failed; running middle-layer-repeat on Qwen3-1.7B", flush=True)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            for loops in (1, 2, 4):
                model = AutoModelForCausalLM.from_pretrained(
                    FALLBACK_REPO, torch_dtype=torch.bfloat16, local_files_only=True)
                if loops > 1:
                    layers = model.model.layers
                    mid = len(layers) // 2
                    for _ in range(loops - 1):
                        layers.insert(mid + 1, layers[mid])
                model = model.to("cuda")
                model.eval()
                tok = AutoTokenizer.from_pretrained(FALLBACK_REPO)
                torch.cuda.reset_peak_memory_stats()
                res = evaluate(model, tok, items, torch)
                runs.append({
                    "model": f"layer-repeat:{FALLBACK_REPO}",
                    "loops": loops,
                    "accuracy": round(res["accuracy"], 4),
                    "tokens_per_s": round(res["tokens_per_s"], 1),
                    "vram_gb": round(res["vram_gb"], 2),
                    "task_accuracy": {k: round(v, 4) for k, v in res["task_accuracy"].items()},
                })
                print(f"[fallback-ok] loops={loops}: acc={res['accuracy']:.3f}", flush=True)
                del model
                torch.cuda.empty_cache()
        except Exception as e:
            failures.append(f"fallback: {type(e).__name__}: {e}")
            traceback.print_exc()

    report = {
        "benchmark": ("loop-eval-v1: 100 multi-step arithmetic + 100 factual knowledge, "
                      "4-way multiple choice, length-normalised log-likelihood"),
        "n": len(items),
        "n_items": len(items),
        "generated_by": "gpu_jobs/looped-eval.job.yaml",
        "failures": failures,
        "runs": runs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print(f"[write] {OUT} with {len(runs)} runs")

    distinct_loops = {r["loops"] for r in runs}
    ok = bool(runs) and len(distinct_loops) >= 3 and all(0 <= r["accuracy"] <= 1 for r in runs) \
        and all(r["tokens_per_s"] > 0 for r in runs)
    if not ok:
        print(f"[FAIL] insufficient result: runs={len(runs)} loops={sorted(distinct_loops)} "
              f"failures={failures}", file=sys.stderr)
        return 1
    print("[DONE] valid looped-eval report written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
