# Experiment cards — R3 top items not yet tested by us

Five smallest tests that could change a decision, chosen from `docs/digest.md`.
Looped LMs are already covered by `docs/looped-lm.md` and `docs/next-experiments.md`,
so they are excluded here. Every card runs on the single RTX 3090 (24 GB) and writes
its result through a queued `gpu_jobs/*.job.yaml`; a claim becomes a result only in the
provenance ledger, never from these documents.

| # | Card | Digest items | Cost | Decision it unblocks |
|--:|---|---|---:|---|
| 1 | [CoT vs self-consistency](01-cot-vs-self-consistency.md) | CoT, self-consistency, test-time compute | ~1.5 h | spend inference time or a bigger model? |
| 2 | [RAG vs QLoRA](02-rag-vs-qlora.md) | RAG, retrieval, QLoRA, LoRA | ~1.5 h | retrieval or fine-tuning for domain facts? |
| 3 | [4-bit format bake-off](03-quantization-format-bakeoff.md) | quantization, KV cache, flash attention | ~2 h | which format to serve extraction on |
| 4 | [Speculative decoding](04-speculative-decoding.md) | speculative decoding, KV cache | ~1 h | draft model or larger target? |
| 5 | [Compiled prompts](05-compiled-prompts.md) | DSPy, prompt engineering, in-context learning | ~1 h | automate prompt tuning or keep hand prompts? |

Each card states a falsifiable hypothesis, the smallest 3090 test, the metric, the
GPU-hour budget and an explicit pass/fail rule. None of them has been run.
