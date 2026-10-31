# Setup changes for the 3090 box — R5 recommendations

**Nothing here is applied to the live system by this project.** This is the exact
config the owner reviews and Claude applies. Each change carries a measured gain; the
per-trial numbers live in `state/upgrades/<name>.json` (written only by the runner GPU
jobs) and are summarised by `python3 tools/upgrades/report.py`.

Status: **all eight R5 trials ran on this box. Two changes are adopted; six are
rejected, each with the measured reason. The two R4-verified changes are kept.**

## Adopted (measured on this box)

### R5-1. Two parallel slots for the 27B bulk-text batch — +18.3% aggregate throughput

Source: `state/upgrades/parallel-batch.json` (n=50 real R2 ingest prompts via
llama-server on the cached 27B Q4_K_M).
Measured: aggregate **22.73 → 26.88 tok/s (+18.26%)** with `--parallel 2` and two
concurrent requests; wall time for the same 50 prompts 116.6 → 112.8 s. Per-request
decode falls 40.63 → 36.81 tok/s because two requests share the GPU, but total work
per second rises.

```sh
# Ollama service
OLLAMA_NUM_PARALLEL=2
# llama-server
llama-server -m <27B>-Q4_K_M.gguf -ngl 99 -c 8192 --parallel 2
```

Caveat: concurrent batching changed 68% of the generated characters vs the sequential
run (`identity` 32.1), so keep `--parallel 1` where byte-exact reproducibility matters.
Each slot needs its own KV cache, so re-check free VRAM before raising the slot count.

### R5-2. Sample packing in the QLoRA step — +44.8% supervised tokens/s

Source: `state/upgrades/qlora-packing.json` (9B NF4 QLoRA step).
Measured: **236.9 → 343.0 tokens/s (+44.79%)**; step time 1.22 → 1.00 s; peak VRAM
flat (14.40 → 14.39 GB) because packing removes padding; supervised tokens per sample
289 → 343.

```python
from trl import SFTConfig
SFTConfig(..., packing=True)   # concatenate short examples, no padding
```

Risk: packing concatenates examples, so an example is no longer cut at its own boundary.
TRL's `packing=True` masks the non-supervised tokens; that is exactly what this step
verified (every packed label chunk keeps at least one supervised token).

### R4-1. Serve the 4-bit GGUF, not BF16

Source: `state/exp/03-quantization-format-bakeoff.json` (R4, n=250).
Measured: **2.13x tokens/s** (44.78 → 95.60) and **5.23 GB vs 16.06 GB** VRAM at
101.2% of BF16 field-F1.

```sh
# Ollama (already how the local 27B and 9B are served)
ollama run hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M
# llama-server
llama-server -m <model>-Q4_K_M.gguf -ngl 99 -c 8192
```

Risk: re-check field-F1 before trusting Q4 on a new model/task; the tie here is
for this 9B and this extraction task.

### R4-2. Metric-driven prompt compilation for extraction

Source: `state/exp/05-compiled-prompts.json` (R4, n=60).
Measured: **+5.71 test field-F1** (87.86 → 93.57) with 24 dev evaluations. The
LLM-judge selector scored 0 gain (every config 100.0 on dev), so use labelled dev
F1, not a judge proxy.

```sh
# recipe, not a flag: docs/results.md (R4) §Card 5
python3 tools/exp/... # search few-shot sets x paraphrases against dev F1
```

## Rejected (measured)

### R5 — q8_0 KV cache: metric not captured and the output changed

Source: `state/upgrades/kv-cache-quant.json` (n=50).
Measured: 40.69 → 40.38 tok/s (-0.8%) and **31.9% greedy-output identity** vs f16.
The server's KV-buffer size line was not captured for this arm (`kv_cache_mb` absent),
so the trial's declared metric has no value and it cannot be adopted. The large output
change alone means it is not a free swap.

### R5 — flash attention: no gain at this context length

Source: `state/upgrades/flash-attention.json` (n=50).
Measured: 40.81 → 40.67 tok/s (**-0.34%**), below the +5% adopt rule, and 64% output
identity. Not adopted.

### R5 — context cap to 2048: no gain, output changed

Source: `state/upgrades/context-length.json` (n=50).
Measured: 41.56 → 40.85 tok/s and wall 126.8 → 119.9 s; `kv_cache_mb` again absent, and
64% output identity. The prompts already fit 8k, so capping bought nothing measurable.
Not adopted.

### R5 — 0.6B speculative draft: -0.6%

Source: `state/upgrades/speculative-draft.json` (n=50).
Measured: 40.86 → 40.61 tok/s (**-0.61%**) with `--spec-type draft-simple -md
Qwen3-0.6B-Q8_0.gguf` vs target-only. Far below the +20% rule. Not adopted on this box.

### R5 — gradient checkpointing on the 9B QLoRA: OOM

Source: `state/upgrades/qlora-grad-checkpoint.json`.
Measured: `use_gradient_checkpointing=True` OOM'd before the first step (a 486 MiB
allocation failed with 23.62 GiB of the 23.68 GiB GPU already in use). Not adopted for
this 4-bit 9B config; reduce batch size or `max_seq_length` first.

### R5 — schema-constrained JSON for extraction: field-F1 collapse

Source: `state/upgrades/extraction-json-schema.json` (sales-brain, n=24).
Measured: field-F1 **72.62 → 18.65** while schema validity stayed 100% (so the trial's
declared `schema_valid_pct` metric showed 0 gain). Forcing all seven keys as strings made
the model emit empty/placeholder values. Keep the free-text prompt plus the parser.

### R4 — speculative decoding, ngram alternative

Source: `state/exp/04-speculative-decoding.json` (R4, n=50).
Measured: **1.02x** tokens/s at **94%** greedy character identity. Not worth the
complexity; the 0.6B draft did not run (fetch failed). The R5 `speculative-draft` trial
re-tested the draft and also rejected it (-0.6%).

### R4 — self-consistency majority vote

Source: `state/exp/01-cot-vs-self-consistency.json` (R4, n=60).
Measured: 4-sample majority bought **+1.26 F1 for 4x the tokens**; F1 per 1k
tokens fell 1006 → 259. Prefer the compiled prompt.

## Proposed Ollama server env (apply only the adopted line)

```sh
# /etc/systemd/system/ollama.service.d/override.conf  (review before applying)
Environment="OLLAMA_NUM_PARALLEL=2"        # R5-1, +18.3% aggregate; adopted
# Environment="OLLAMA_FLASH_ATTENTION=1"  # R5: -0.3%; rejected
# Environment="OLLAMA_KV_CACHE_TYPE=q8_0" # R5: no measured gain, output changed; rejected
# Environment="OLLAMA_KEEP_ALIVE=-1"       # not measured in R5; dedicated box only
```

`OLLAMA_KEEP_ALIVE=-1` holds the 27B resident indefinitely; do not set it if another
project shares the GPU. The QLoRA packing change is a training-script argument
(`SFTConfig(packing=True)`), not an Ollama setting.
