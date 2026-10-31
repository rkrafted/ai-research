# Experiment card 2 — retrieval vs QLoRA fine-tuning at equal VRAM

**Source items.** `docs/digest.md` → *RAG* (rank 4), *Retrieval* (rank 14),
*QLoRA* (rank 41), *LoRA* (rank 30). Status: **not yet tested by us.**

**Hypothesis.** For a domain with a fixed, changing fact set, retrieval-augmented
generation on a 3B base reaches the same extraction accuracy as QLoRA-tuning the same
base, at lower total GPU-hours, and degrades gracefully when a fact is edited
(re-index only). Fine-tuning should win only on style/format, not on facts.

**Smallest test on the 3090.**
Take the sales-brain knowledge base (fictional fixtures, no real customer data) and
split it into (a) a retrieval corpus of ~500 short passages and (b) 400 Q/A pairs whose
answers require those passages. Three arms, same base model (Qwen3-1.7B or 4B):

1. base + zero-shot prompt;
2. base + top-3 embedding retrieval (local `nomic-embed` or `bge-small`) with passages
   in the prompt;
3. QLoRA (4-bit base + LoRA rank 16) trained on the 400 pairs for 2 epochs.

Then a **held-out edit test**: change 30 facts in the corpus and re-run all arms
without re-training (arm 3 keeps its old weights). Measure the answer accuracy on the
30 edited facts.

**Metric.** Exact-match/field-F1 on the 400 held-out questions; answer accuracy on the
30 edited facts; peak VRAM; total GPU-minutes including training; $0 API spend.

**GPU-hours.** ~1.5 h (retrieval is minutes; QLoRA on 400 pairs at 1.7B is ~25 min;
generation dominates).

**Pass/fail rule.** Pass if retrieval is within 2 F1 points of QLoRA on held-out
questions **and** beats it by ≥20 points on the edited facts, at ≤ half the GPU-minutes.
Fail if QLoRA beats retrieval by >5 F1 points on held-out questions or if retrieval
needs more than 4k prompt tokens to get there.

**Evidence.**
- arXiv:2005.11401 (RAG), arXiv:2305.14314 (QLoRA), arXiv:2106.09685 (LoRA),
  arXiv:2404.10981 (retrieval-augmented generation survey), arXiv:2305.04091
  (plan-and-solve, for the extraction prompt).
- KB video evidence: *RAG*, *Retrieval* and *QLoRA* entries in `docs/digest.md`.
