# Experiment card 5 — compiled prompts (DSPy-style) vs hand-written prompts

**Source items.** `docs/digest.md` → *DSPy* (rank 13), *Prompt Engineering* (rank 2),
*In-Context Learning* (rank 3), *LLM-as-a-Judge* (rank 10). Status: **not yet tested by
us.**

**Hypothesis.** A metric-driven prompt compiler (a small local optimiser over prompt
variants and few-shot demonstrations, the DSPy idea implemented with nothing but our
own extraction harness) finds a prompt that beats our best hand-written prompt by
≥2 F1 points on a held-out extraction split, without any weight update, in under
1 GPU-hour. The optimiser should overfit the dev split, so the held-out gap is the real
test.

**Smallest test on the 3090.**
Task: the sales-brain JSON extraction fixture. Split the labelled rows into
dev (200) / test (300). Arm A: our current hand-written prompt + 3 fixed examples.
Arm B: same base model, prompt compiled by hill-climbing over:

- 8 instruction paraphrases (authored once, not generated at run time);
- 3 few-shot example *sets* selected by dev F1 (nearest-neighbour / worst-first);
- optional field-order and schema-description variants.

Selection uses dev F1 only. Report dev and test separately. A second arm B′ runs the
same search with the LLM-as-judge scoring to see whether a judge can replace exact
labels (cheap-label question). Greedy decoding, frozen base, no fine-tuning.

**Metric.** Test F1 of A vs B vs B′, dev–test gap, number of dev evaluations as the
proxy for cost, and GPU-minutes for the whole search. Also schema-valid rate.

**GPU-hours.** ~1 h (≈400 dev generations + 300 test generations at 3B, batched).

**Pass/fail rule.** Pass if B beats A by ≥2 F1 points on **test** (not just dev) with
≤100 dev evaluations. Fail if the test gain is <1 point or the search needs >300 dev
evaluations; then hand prompts stay the default and the runner only uses compiled
prompts for high-volume tasks.

**Evidence.**
- arXiv:2310.03714 (DSPy), arXiv:2305.03495 (automatic prompt engineer),
  arXiv:2305.04091 (plan-and-solve), arXiv:2401.10020 (self-rewarding / judge-style
  evaluation).
- KB video evidence: *DSPy* and *Prompt Engineering* entries in `docs/digest.md`.
