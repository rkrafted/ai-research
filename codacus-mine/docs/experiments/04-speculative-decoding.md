# Experiment card 4 — speculative decoding throughput at identical output

**Source items.** `docs/digest.md` → *Speculative Decoding* (rank 57), *KV Cache*
(rank 22), *Flash Attention* (rank 33). Status: **not yet tested by us.**

**Hypothesis.** A tiny local draft model (0.5B) paired with the 7B used for extraction
produces byte-identical greedy output at ≥1.7x tokens/s on our prompts, and the gain
survives at the long-context end (~8k prompt tokens) where decode dominates. At batch
>1 the gain shrinks; we want to know the crossover.

**Smallest test on the 3090.**
Target: the 7B already used for extraction; draft: a 0.5B instruct model of the same
tokenizer family (both pre-cached). Prompts: 100 frozen extraction prompts bucketed by
length (short <1k, long >6k tokens). Three arms:

1. target alone, greedy;
2. target + draft, `num_speculative_tokens ∈ {3, 5}`;
3. target alone at `temperature=0`, but with n-gram/prompt-lookup self-speculation
   (no second model) where the runner supports it.

Verify equivalence: compare each prompt's greedy continuation token-for-token against
arm 1 (speculative decoding must be lossless on greedy). Measure tokens/s, accepted-
token rate, and peak VRAM (target + draft must both fit).

**Metric.** Tokens/s speedup, acceptance rate, output equality rate, peak VRAM, and
the batch size at which the speedup falls below 1.2x.

**GPU-hours.** ~1 h (100 prompts × 3 arms, short + long).

**Pass/fail rule.** Pass if arm 2 gives ≥1.5x tokens/s on long prompts with a 100%
greedy-match rate and both models inside 24 GB. Fail if acceptance <40% or the speedup
is <1.2x on long prompts; then the decision is to spend the VRAM on a larger target
model instead of a draft.

**Evidence.**
- arXiv:2211.17192 (speculative decoding), arXiv:2302.01318 (fast inference via
  speculative decoding), arXiv:2401.10774 (Medusa multi-head drafting),
  arXiv:2404.15758 (hidden computation).
- KB video evidence: *Speculative Decoding* entry in `docs/digest.md`.
