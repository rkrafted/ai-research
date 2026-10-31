# Experiment card 3 — 4-bit format bake-off for a 7B on one 24 GB card

**Source items.** `docs/digest.md` → *Quantization* (rank 16), *KV Cache* (rank 22),
*Flash Attention* (rank 33), *Tokenization* (rank 34). Status: **not yet tested by us.**

**Hypothesis.** At fixed 24 GB, a 4-bit format (GPTQ / AWQ / GGUF-Q4_K_M / FP8) keeps
≥97% of the FP16 field-F1 on our extraction task while raising tokens/s and the
maximum context that fits. The ordering of the formats on *quality* will be within
noise; the real differences are decode speed, load time and context headroom.

**Smallest test on the 3090.**
Same base model in four formats (Qwen2.5-7B-Instruct is the default; substitute the
best 7B already in the local HF cache). One fixed 250-row extraction fixture with the
prompt frozen across arms. For each arm measure:

- field micro-F1 and schema-valid rate (greedy, `max_new_tokens=128`);
- tokens/s at batch 1 and at batch 8;
- peak VRAM and the largest `num_ctx` that still fits with the batch-8 KV cache;
- one cold load time.

Use `llama.cpp`/`vLLM`/`transformers` per format, but record the runner and version.
No network: only checkpoints already in `~/.cache/huggingface` (queue a `fetch/` file
first if one is missing).

**Metric.** F1 retained vs FP16, tokens/s speedup, VRAM (GB), max context at batch 8,
load seconds. Report a small Pareto table, not a single winner.

**GPU-hours.** ~2 h (four formats × 250 items + load/benchmark).

**Pass/fail rule.** Pass if at least one 4-bit format retains ≥97% of FP16 F1 **and**
delivers ≥1.5x tokens/s or ≥2x max context on the same card. Fail if every 4-bit format
loses >3 F1 points — in that case the finding is "quantize only for serving, keep FP16
for extraction", which is itself the decision.

**Evidence.**
- arXiv:2210.17323 (GPTQ), arXiv:2306.00978 (AWQ), arXiv:2208.07339 (LLM.int8),
  arXiv:2211.10438 (SmoothQuant), arXiv:2309.14717 (QA-LoRA).
- KB video evidence: *Quantization*, *KV Cache*, *Flash Attention* entries in
  `docs/digest.md`.
