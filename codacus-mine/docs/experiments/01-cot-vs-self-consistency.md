# Experiment card 1 — chain of thought vs self-consistency at a fixed GPU-second budget

**Source items.** `docs/digest.md` → *Chain of Thought* (rank 1), *Self-Consistency*
(rank 5), *Test-Time Compute* (rank 4). Status: **not yet tested by us.**

**Hypothesis.** On our own extraction task, at an equal wall-clock budget per item a
short chain-of-thought prompt plus `n` majority-voted samples beats a single long
chain-of-thought answer, and both beat a direct answer. The gain should come from
sampling diversity rather than from a longer single trace, so a 3B local model can
close part of the gap to a 7B direct-answer model by spending inference time instead
of VRAM.

**Smallest test on the 3090.**
Fixture: 300 rows from the sales-brain extraction set already used in
`docs/next-experiments.md` (inputs + teacher JSON labels), frozen prompt, greedy
decoding for the direct arm. Four arms on the same Qwen3-4B-Instruct (or the 3B in
`.venv`'s repo if 4B does not fit alongside the KV cache):

1. direct answer, `max_new_tokens=64`;
2. one CoT trace, `max_new_tokens=256`;
3. CoT + 4 samples, majority vote per JSON field, `temperature=0.7`;
4. CoT + 8 samples, majority vote.

Arm 2 gets the same total token budget as the direct arm (~4x), so the comparison is
compute-aware; record actual tokens and seconds. Cache the model between arms, batch
nothing larger than 1 sequence so VRAM peaks are comparable.

**Metric.** JSON field micro-F1 over the five fields, schema-valid rate, and **field-F1
per 1000 generated tokens** and **per peak-VRAM-GB**. Report p50/p95 latency per item.

**GPU-hours.** ~1.5 h (300 items × 4 arms; sampling is decode-bound).

**Pass/fail rule.** Pass if arm 3 or 4 reaches ≥2 F1 points over arm 1 at ≤ the same
tokens-per-item as arm 2 *and* the gain per 1000 tokens is higher than arm 2's.
Fail if no arm beats the direct answer by ≥1 F1 point, or if self-consistency only wins
by spending more tokens than the CoT arm at equal-or-worse tokens-per-point.

**Evidence.**
- arXiv:2201.11903 (chain of thought), arXiv:2203.11171 (self-consistency),
  arXiv:2207.05221 (models mostly know what they know), arXiv:2409.12183 (CoT helps
  mainly on math/symbolic tasks — the honest counterweight), arXiv:2408.03314
  (test-time compute scaling).
- KB video evidence: see the *Chain of Thought* and *Self-Consistency* entries in
  `docs/digest.md` for timestamped links.
