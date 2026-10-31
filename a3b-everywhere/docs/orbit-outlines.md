# ORBIT episode outlines — R4 verified results

Three episode outlines, each built only from the five experiment cards in
`docs/experiments/` and the numbers their GPU jobs write to `state/exp/`.
Status: cards 1, 3, 4 and 5 are verified on the hard task; card 2's corrected
job is re-queued and its numbers are the one `TBD` below. Nothing here is a claim
until the provenance-checked file exists.

## Episode 1 — Spend tokens or buy VRAM?

- **Hook.** "You have one 24 GB card. Do you get more from sampling the model
  you already have, or from a smaller draft that guesses ahead?"
- **What we ran.** Card 1 (direct vs CoT vs CoT+4/8 majority vote) and Card 4
  (27B target vs 0.6B draft vs 0 GB ngram speculation) on the same hard
  transcript-extraction workload.
- **The numbers that matter.** Card 1, verified hard run: direct 0.0 field-F1
  (64-token cap, 0% parseable JSON), CoT 72.32 at 71.8 tok/item, CoT+4 73.57 at
  284.1, CoT+8 73.33 at 594.2; F1 per 1k tokens 1006 → 259 → 123. Card 4,
  verified: target-only 40.35 tok/s, ngram-simple 41.27 tok/s (1.02x) at 94%
  greedy identity; the 0.6B draft fetch failed, so there is no draft number.
  Card 3's verified tok/s split for context: 44.78 (BF16) vs 95.60 (Q4_K_M).
- **Builders' takeaway.** Sampling bought 1.26 F1 for 4x the tokens and lost on
  F1 per unit compute; the 0 GB speculation bought 2% for 6% different output.
  Neither is the lever. Spend the VRAM on quantization and a better prompt first.
- **Beats.** Cold open on the token-budget arithmetic; one screen per arm; end on
  the decision rule, not the winner. Show the failed draft download as a card —
  "we could not test the interesting arm" is the honest cliffhanger.

## Episode 2 — Retrieval re-indexes, fine-tuning re-trains

- **Hook.** "A fact changed. With RAG you re-embed one paragraph; with QLoRA you
  re-run the job. We measured both."
- **What we ran.** Card 2: 1.7B zero-shot vs top-3 retrieval vs QLoRA r16,
  then 30 training facts edited and all three arms re-run without retraining.
- **The numbers that matter.** Held-out exact-match accuracy per arm,
  edited-fact accuracy per arm, prompt tokens for the retrieved context,
  GPU-minutes: `TBD` (`state/exp/02-rag-vs-qlora.json`). The first two attempts
  are a story in themselves: the job crashed twice on a
  `BatchEncoding`/tensor type change in `apply_chat_template` (0.12 and 0.435
  GPU-minutes, both n=0). The fix is in and the corrected job is queued.
- **Builders' takeaway.** The edit test is the story: a model that wins the
  held-out set but cannot answer a changed fact is the wrong default for a
  moving knowledge base. Report the failed attempts too — two zero-row runs are
  what a silent library-API change looks like.
- **Beats.** Show the edit as a code diff (one passage, one value); show the
  stale-weight arm answering the old value; state the GPU-minute ratio.

## Episode 3 — The 4-bit bargain, and prompt search that pays for itself

- **Hook.** "Quantize 16-bit to 4-bit: what breaks, and does automatic prompt
  search beat a hand-written prompt?"
- **What we ran.** Card 3 (same 9B at BF16 and GGUF Q4_K_M: F1, tokens/s, VRAM,
  load time) and Card 5 (24-config prompt search vs the hand prompt, plus a
  judge-scored arm) on the hard transcripts.
- **The numbers that matter.** Card 3, verified: BF16 96.08 F1 / 44.78 tok/s /
  16.06 GB / 10.88 s load versus Q4_K_M 97.20 F1 / 95.60 tok/s / 5.23 GB /
  2.82 s load — 101.2% of BF16 quality at 2.13x the speed. Card 5, verified hard
  run: hand prompt 87.86 test-F1, compiled 93.57 (+5.71) with 24 dev
  evaluations, judge-compiled 87.86 (+0.00) because the judge scored every dev
  config 100 and chose the hand-prompt config.
- **Builders' takeaway.** Quantization is settled for this model: 4-bit is not a
  trade, it is a straight win. Prompt compilation works — but only the exact
  dev-F1 signal; the LLM-judge proxy collapsed to the baseline and would have
  cost 24 evaluations for nothing.
- **Beats.** Side-by-side Pareto table for the formats (F1, tok/s, VRAM, load);
  one dev-vs-test bar chart for the prompts, with the judge arm drawn as the
  flat line it is; close on the two defaults a one-GPU builder can adopt.

## Production notes

- Source of truth: `state/exp/<card>.json` (provenance-checked). Never read the
  hype; read the file.
- Every episode needs the failure case as prominently as the win: the draft
  download that never finished (card 4), the judge proxy that learned nothing
  (card 5), the two `BatchEncoding` crashes (card 2), and the 1.26-F1 sampling
  gain for 4x the tokens (card 1).
- Machine-readable reference tables belong in the video description, generated
  from `state/exp/*.json`, not hand-copied.
