#!/usr/bin/env python3
"""R5 upgrade trials -- one `--trial <name>` per GPU job.

Each trial measures the workload BEFORE and AFTER the change and writes
`state/upgrades/<name>.json` (never from a session). Workloads:

  ingest-*   the local 27B bulk-text batch: 50 real R2 ingest prompts
  qlora-*    QLoRA step time / peak VRAM on the cached 9B base
  extract-*  sales-brain deal extraction from call transcripts on the 9B (Ollama)

Usage:  python3 tools/upgrades/trials.py --trial kv-cache-quant
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import upg_common as C  # noqa: E402
import fixtures as F  # noqa: E402

N_PROMPTS = 50
N_PREDICT = 128


# ---------------------------------------------------------------------------
# inference-speed / fit trials on the cached 27B via llama-server
# ---------------------------------------------------------------------------

def _ingest_arms(extra: list[str], label: str, *, ctx: int = 8192) -> dict:
    prompts, source = C.load_ingest_prompts(N_PROMPTS)
    arm = C.run_llama_arm(C.locate_target(), extra, prompts, label, ctx=ctx,
                          n_predict=N_PREDICT)
    arm["prompt_source"] = source
    return arm


def trial_kv_cache_quant():
    note = C.setup_backend()
    flags = C.context_flags()
    base = _ingest_arms([], "kv-base")
    after = _ingest_arms(flags, "kv-q8")
    identity = C.identity(base["contents"], after["contents"])
    before = {**base["metrics"], "identity": 100.0, "flags": "default f16 KV cache"}
    aft = {**after["metrics"], "identity": identity,
           "flags": " ".join(flags) or "(no cache-type flag in this llama-server)"}
    how = aft["flags"]
    return before, aft, how


def trial_flash_attention():
    C.setup_backend()
    flags = C.flash_flags()
    base = _ingest_arms([], "fa-base")
    after = _ingest_arms(flags, "fa-on")
    before = {**base["metrics"], "identity": 100.0, "flags": "no flash attention"}
    aft = {**after["metrics"], "identity": C.identity(base["contents"], after["contents"]),
           "flags": " ".join(flags) or "(no flash-attn flag in this llama-server)"}
    return before, aft, aft["flags"]


def trial_context_length():
    C.setup_backend()
    big = _ingest_arms([], "ctx-8192", ctx=8192)
    small = _ingest_arms([], "ctx-2048", ctx=2048)
    before = {**big["metrics"], "identity": 100.0, "num_ctx": 8192,
              "note": "the current local-batch default (num_ctx 8192)"}
    aft = {**small["metrics"], "identity": C.identity(big["contents"], small["contents"]),
           "num_ctx": 2048, "note": "cap context at what one prompt needs"}
    return before, aft, "serve with -c 2048 (llama-server); OLLAMA default num_ctx for 8192-token prompts"


def _parallel_batch(model: Path, prompts: list[str], parallel: int, label: str) -> dict:
    """Aggregate tokens/s for a fixed 50-prompt list at a given --parallel."""
    ctx = 4096
    log_path = Path("/tmp") / f"upg-{label}.log"
    extra = C.parallel_flags(parallel)
    proc, log = C.start_server(model, extra, log_path, ctx=ctx)
    tok, ms, wall = 0, 0.0, 0.0
    try:
        if not C.wait_ready(proc):
            tail = log_path.read_text(errors="replace")[-1200:]
            raise RuntimeError(f"llama-server ({label}) not ready: ...{tail}")

        def one(p):
            return C.complete(p, n_predict=N_PREDICT)

        t0 = time.time()
        if parallel <= 1:
            outs = [one(p) for p in prompts]
        else:
            outs = []
            with ThreadPoolExecutor(max_workers=parallel) as ex:
                for i in range(0, len(prompts), parallel):
                    outs.extend(ex.map(one, prompts[i:i + parallel]))
        wall = time.time() - t0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            proc.kill()
        log.close()
    tok = sum(o["n"] for o in outs)
    ms = sum(o["ms"] for o in outs)
    metrics = {
        "tokens_per_s": round(tok / (ms / 1000.0), 2) if ms else 0.0,
        "aggregate_tokens_per_s": round(tok / wall, 2) if wall else 0.0,
        "wall_seconds": round(wall, 1), "n_prompts": len(prompts),
        "completion_tokens": tok, "parallel": parallel,
    }
    metrics.update(C.parse_vram_mb(log_path))
    return {"metrics": metrics, "contents": [o["content"] for o in outs]}


def trial_parallel_batch():
    C.setup_backend()
    prompts, source = C.load_ingest_prompts(N_PROMPTS)
    model = C.locate_target()
    one = _parallel_batch(model, prompts, 1, "par-1")
    two = _parallel_batch(model, prompts, 2, "par-2")
    before = {**one["metrics"], "prompt_source": source}
    aft = {**two["metrics"], "prompt_source": source,
           "identity": C.identity(one["contents"], two["contents"])}
    return before, aft, "--parallel 2 with two concurrent requests (llama-server; Ollama: OLLAMA_NUM_PARALLEL=2)"


def _locate_draft() -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf",
                                    local_files_only=True))
    except Exception:  # noqa: BLE001
        base = Path.home() / ".cache" / "huggingface" / "hub"
        hits = sorted((base / "models--Qwen--Qwen3-0.6B-GGUF" / "snapshots").glob(
            "*/Qwen3-0.6B-Q8_0.gguf"))
        return hits[-1] if hits else None


def trial_speculative_draft():
    C.setup_backend()
    prompts, source = C.load_ingest_prompts(N_PROMPTS)
    model = C.locate_target()
    base = C.run_llama_arm(model, [], prompts, "spec-base", ctx=8192, n_predict=N_PREDICT)
    before = {**base["metrics"], "identity": 100.0, "prompt_source": source,
              "arm": "target-only"}
    draft = _locate_draft()
    if draft is not None:
        extra = ["--spec-type", "draft-simple", "-md", str(draft)]
        arm, label = "target+Qwen3-0.6B-Q8_0", "spec-draft"
    else:
        extra, arm, label = ["--spec-type", "ngram-simple"], "target+ngram-simple", "spec-ngram"
    try:
        run = C.run_llama_arm(model, extra, prompts, label, ctx=8192, n_predict=N_PREDICT)
        aft = {**run["metrics"], "identity": C.identity(base["contents"], run["contents"]),
               "prompt_source": source, "arm": arm}
    except Exception as exc:  # noqa: BLE001 - the base arm is still a result
        aft = {**base["metrics"], "identity": 100.0, "prompt_source": source,
               "arm": arm, "error": f"{type(exc).__name__}: {exc}"}
    return before, aft, (f"--spec-type draft-simple -md {draft.name}" if draft
                         else "--spec-type ngram-simple (0 GB draft absent)")


# ---------------------------------------------------------------------------
# fine-tuning efficiency on the cached 9B base (QLoRA)
# ---------------------------------------------------------------------------

def _steps(model, examples, *, batch=1, steps=6, max_len=512, packed=False):
    """Run `steps` training steps, return (supervised_tokens/s, step_seconds, peak_gb)."""
    import torch
    tok = None  # tokenizer not needed for the step loop
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    if packed:
        ids, labels = _pack(examples, max_len)
        batches = [([i], [l]) for i, l in zip(ids, labels)][:steps]
    else:
        batches = [[examples[i % len(examples)] for i in range(b * batch, (b + 1) * batch)]
                   for b in range(steps)]
    torch.cuda.reset_peak_memory_stats()
    sup, times = 0, []
    for batch_items in batches:
        if packed:
            i_list, l_list = batch_items
        else:
            i_list = [x["ids"] for x in batch_items]
            l_list = [x["labels"] for x in batch_items]
        maxlen = max(len(x) for x in i_list)
        input_ids = torch.full((len(i_list), maxlen), 0, dtype=torch.long)
        attn = torch.zeros((len(i_list), maxlen), dtype=torch.long)
        labels = torch.full((len(i_list), maxlen), -100, dtype=torch.long)
        for r, (ids, lab) in enumerate(zip(i_list, l_list)):
            input_ids[r, :len(ids)] = torch.tensor(ids)
            attn[r, :len(ids)] = 1
            labels[r, :len(lab)] = torch.tensor(lab)
        sup += int((labels != -100).sum())
        t = time.time()
        out = model(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device),
                    labels=labels.to(model.device))
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        times.append(time.time() - t)
    peak = torch.cuda.max_memory_allocated() / 1e9
    del opt
    model.zero_grad(set_to_none=True)
    return sup, statistics.fmean(times), peak


def _pack(examples, max_len=512):
    """Concatenate examples, then chunk to max_len; drop all-masked chunks."""
    stream_ids, stream_lab = [], []
    for e in examples:
        stream_ids += list(e["ids"]) + [0]
        stream_lab += list(e["labels"]) + [-100]
    chunks_i, chunks_l = [], []
    for i in range(0, len(stream_ids), max_len):
        ci, cl = stream_ids[i:i + max_len], stream_lab[i:i + max_len]
        if any(x != -100 for x in cl):
            chunks_i.append(ci)
            chunks_l.append(cl)
    return chunks_i, chunks_l


def _load_qlora(grad_checkpoint: bool):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.BASE_9B_HF)
    base = C.load_base_4bit()
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=grad_checkpoint)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, lora)
    return model, tok


def trial_qlora_grad_checkpoint():
    import torch
    examples = None
    model, tok = _load_qlora(False)
    examples = C.qlora_examples(tok, n=24, max_len=512)
    sup, step_s, peak_off = _steps(model, examples)
    before = {"step_seconds": round(step_s, 3), "peak_vram_gb": round(peak_off, 2),
              "supervised_tokens": sup, "gradient_checkpointing": False}
    del model
    torch.cuda.empty_cache()
    model, tok = _load_qlora(True)
    examples = C.qlora_examples(tok, n=24, max_len=512)
    sup, step_s, peak_on = _steps(model, examples)
    after = {"step_seconds": round(step_s, 3), "peak_vram_gb": round(peak_on, 2),
             "supervised_tokens": sup, "gradient_checkpointing": True}
    how = ("prepare_model_for_kbit_training(model, use_gradient_checkpointing=True); "
           "training_args(gradient_checkpointing=True)")
    return before, after, how


def trial_qlora_packing():
    import torch
    model, tok = _load_qlora(True)
    examples = C.qlora_examples(tok, n=24, max_len=512)
    sup, step_s, peak = _steps(model, examples, packed=False)
    before = {"tokens_per_s": round(sup / step_s, 1) if step_s else 0.0,
              "step_seconds": round(step_s, 3), "peak_vram_gb": round(peak, 2),
              "supervised_tokens": sup, "packing": False}
    del model
    torch.cuda.empty_cache()
    model, tok = _load_qlora(True)
    examples = C.qlora_examples(tok, n=24, max_len=512)
    sup, step_s, peak = _steps(model, examples, packed=True)
    after = {"tokens_per_s": round(sup / step_s, 1) if step_s else 0.0,
             "step_seconds": round(step_s, 3), "peak_vram_gb": round(peak, 2),
             "supervised_tokens": sup, "packing": True}
    how = "pack several short examples into one max_seq_length block (no padding); TRL: packing=True"
    return before, after, how


# ---------------------------------------------------------------------------
# data/tool trial: schema-constrained JSON extraction on sales-brain
# ---------------------------------------------------------------------------

def _extract_arm(cli, rows, fmt, n_predict=256):
    preds, golds, secs, valid, toks = [], [], 0.0, 0, 0
    for r in rows:
        user = ("Extract the deal facts from the call transcript. Reply with one JSON "
                "object whose keys are: " + ", ".join(F.HARD_FIELDS) + ".\n\nTranscript:\n"
                + r["text"] + "\nJSON:")
        o = cli.chat(C.TARGET_9B_OLLAMA, [{"role": "user", "content": user}],
                     num_predict=n_predict, num_ctx=8192, fmt=fmt)
        secs += o["seconds"]
        toks += o["tokens"]
        d = F.extract_json(o["text"])
        if isinstance(d, dict) and d:
            valid += 1
        preds.append(F.hard_fields(d or {}))
        golds.append(r["gold"])
    n = len(rows)
    return {
        "field_f1": round(200.0 * _f1(preds, golds), 2),
        "schema_valid_pct": round(100.0 * valid / n, 1) if n else 0.0,
        "seconds_per_item": round(secs / n, 2) if n else 0.0,
        "tokens_per_s": round(toks / secs, 2) if secs else 0.0,
        "n": n,
    }


def _f1(preds, golds):
    fields = F.HARD_FIELDS
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        for f in fields:
            pv, gv = F.norm_value(p.get(f)), F.norm_value(g.get(f))
            if pv == gv and gv != "":
                tp += 1
            else:
                if pv != "":
                    fp += 1
                if gv != "":
                    fn += 1
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return prec * rec / (prec + rec) if (prec + rec) else 0.0


def trial_extraction_json_schema():
    cli = C.Ollama()
    rows, source = C.sales_brain_rows(24)
    before = {**_extract_arm(cli, rows, None), "prompt_source": source, "format": "free text"}
    after = {**_extract_arm(cli, rows, C.HARD_SCHEMA), "prompt_source": source,
             "format": "format=<JSON schema> (all 7 keys as strings)"}
    how = ('Ollama /api/chat: "format": {"type":"object","properties":{...7 keys...},'
           '"required":[...]}; Modelfile equivalent PARAMETER format json')
    return before, after, how


TRIALS = {
    "kv-cache-quant": (trial_kv_cache_quant, "fit-bigger", "27B bulk-text batch",
                       "kv_cache_mb", False, 25.0),
    "flash-attention": (trial_flash_attention, "inference-speed", "27B bulk-text batch",
                        "tokens_per_s", True, 5.0),
    "context-length": (trial_context_length, "fit-bigger", "27B bulk-text batch",
                       "kv_cache_mb", False, 25.0),
    "parallel-batch": (trial_parallel_batch, "inference-speed", "27B bulk-text batch",
                       "aggregate_tokens_per_s", True, 10.0),
    "speculative-draft": (trial_speculative_draft, "inference-speed", "27B bulk-text batch",
                          "tokens_per_s", True, 20.0),
    "qlora-grad-checkpoint": (trial_qlora_grad_checkpoint, "fine-tune-efficiency",
                              "9B QLoRA step", "peak_vram_gb", False, 10.0),
    "qlora-packing": (trial_qlora_packing, "fine-tune-efficiency",
                      "9B QLoRA step", "tokens_per_s", True, 5.0),
    "extraction-json-schema": (trial_extraction_json_schema, "data-tools",
                               "sales-brain extraction", "schema_valid_pct", True, 5.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True, choices=sorted(TRIALS))
    a = ap.parse_args()
    fn, area, workload, key, higher, min_gain = TRIALS[a.trial]
    C.run_trial(a.trial, area, workload, fn.__doc__ or a.trial, fn,
                change_key=key, higher_is_better=higher, min_gain_pct=min_gain)


if __name__ == "__main__":
    main()
