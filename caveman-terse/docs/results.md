# R4 experiment results — five digest cards, measured on our RTX 3090

Source: `docs/experiments/*.md`. Each card runs as a `gpu_jobs/*.job.yaml` on the
single RTX 3090 and writes `state/exp/<card>.json` from inside that job. A number
is a result only when the runner's provenance ledger recorded the file; a session
never writes these JSON files.

Status legend: **[verified]** the job's `state/exp/*.json` exists and is
provenance-checked; **[rerun queued]** a corrected `.job.yaml` is queued and runs
after the session.

History note. The first run of cards 1 and 5 used easy fictional sales notes and
both hit a ceiling (97.0 / 100.0 field-F1), which cannot rank the arms. Per the
owner's 2026-10-03 call, both were re-run on the hard sales-brain task: 7-field
deal extraction from real call transcripts, where the 9B is far from perfect.
The easy-run tables are kept below as the reason for the rerun.

## Card 1 — CoT vs self-consistency at a fixed GPU-second budget

**Rule.** Pass if a 4/8-sample majority arm reaches ≥2 field-F1 over the direct
answer at ≤ the single-CoT arm's tokens/item and a higher F1 per 1k tokens.

**Setup.** `tools/exp/card1_cot_self_consistency.py`, model
`hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M` via Ollama, 60 sales-brain
transcripts, 7 deal fields, four arms (direct 64 tok; CoT 256 tok; CoT+4;
CoT+8).

**Result, hard task — [verified]**

| arm | field-F1 | tok/item | F1/1k tok | schema-valid |
|---|---:|---:|---:|---:|
| direct-64tok | 0.0 | 64.0 | 0 | 0.0% |
| cot-256tok | 72.32 | 71.8 | 1006 | 100% |
| cot-4-sample-majority | 73.57 | 284.1 | 259 | 100% |
| cot-8-sample-majority | 73.33 | 594.2 | 123 | 97.1% |

**Verdict: fail** (`state/exp/01-cot-vs-self-consistency.json`, n=60, 10.0
GPU-minutes). Majority-of-4 beats single CoT by only 1.26 F1 points (not the
card's 2) while spending 4x the tokens; majority-of-8 is flat and costs 8x. F1
per 1k tokens falls 1006 → 259 → 123. The direct arm's 0.0 is real, not a
scoring artifact: capped at 64 tokens on a hard task it never emitted a parseable
JSON object (0% schema-valid). `beats_direct=true` by the letter of the rule, but
`within_token_budget=false` and `better_f1_per_1k=false`, so the card fails. On
this task test-time sampling does not buy accuracy per unit compute.

**Easy-task first run (superseded).** direct 97.0 F1 / 60.5 tok/item, CoT 96.67 /
54.4, CoT+4 97.0 / 219.0, CoT+8 96.67 / 450.4 — 4–8x the tokens for no gain.

## Card 2 — retrieval vs QLoRA at equal VRAM

**Rule.** Pass if top-3 retrieval is within 2 points of QLoRA on held-out
questions, beats QLoRA by ≥20 points on 30 edited facts, and costs ≤ half the
GPU-minutes at ≤4k prompt tokens.

**Setup.** `tools/exp/card2_rag_vs_qlora.py`, `Qwen/Qwen3-1.7B` (0-shot, top-3
retrieval via the Ollama embed encoder, QLoRA r16 ×2 epochs), 500 fictional
passages, 300 train / 200 held-out questions, then 60 training facts edited and
all three arms re-run without retraining. (The card asks ~400/30; the held-out and
edit sets were enlarged so a perfect retrieval arm cannot sit on a 100.0 ceiling
and tell us nothing — the mistake that made cards 1 and 5 uninformative.)

**Result (both runs): [verified — no result yet]**. n=0 twice. Run 1 died with
`TypeError: ones_like(): argument 'input' must be Tensor, not BatchEncoding`
(0.12 GPU-minutes). Run 2 died in the training-token build with
`TypeError: unsupported operand type(s) for +: 'BatchEncoding' and 'list'`
(0.435 GPU-minutes) — the same `apply_chat_template` return-type change, in a
second place. `gpu_jobs/02-rag-vs-qlora.log` has both. The token path is now
fixed in both places and the corrected `.job.yaml` is re-queued
(**[rerun queued]**); the verified numbers go here next session.

## Card 3 — 4-bit format bake-off on one 24 GB card

**Rule.** Pass if a 4-bit format keeps ≥97% of BF16 field-F1 and gives ≥1.5x
tokens/s or ≥2x context on the same card.

**Setup.** `tools/exp/card3_quant_formats.py`, the same 9B at GGUF BF16 and
Q4_K_M through Ollama, 250 frozen extraction rows. Offline only those two formats
are cached, so the bake-off is FP16 vs 4-bit.

**Result: [verified]**

| arm | field-F1 | tok/s | VRAM | load s |
|---|---:|---:|---:|---:|
| bf16 | 96.08 | 44.78 | 16.06 GB | 10.88 |
| q4_k_m | 97.20 | 95.60 | 5.23 GB | 2.82 |

**Verdict: pass** (`state/exp/03-quantization-format-bakeoff.json`, n=250,
8.5 GPU-minutes): 4-bit kept 101.2% of BF16 F1, decoded 2.13x faster, and used a
third of the VRAM. On one 24 GB card the 4-bit GGUF is strictly the better
serving default *for this model and task*; the quality tie is the useful finding.
Caveat in the file: GPTQ/AWQ/FP8 could not be tested offline (no `auto-gptq`,
`autoawq` or vLLM in the shared venv), so this is FP16 vs 4-bit, not a
four-format bake-off.

## Card 4 — speculative decoding at identical output

**Rule.** Pass if some draft gives ≥1.5x tokens/s on ingest-sized prompts at a
100% greedy-match rate, inside 24 GB.

**Setup (re-scoped 2026-10-03).** No Qwen2.5-7B download. `Qwen3.8-27B` Q4_K_M
(target, the cached bulk-text model) with the Ollama-bundled `llama-server`:
target-only, target + `Qwen3-0.6B` Q8_0 draft (`--model-draft`), and the 0 GB
`--spec-type ngram-simple` alternative. 50 extraction prompts; greedy; output
compared character-for-character to the target-only run; the bottleneck metric
is the local batch's ~8.5 s/prompt. The draft was requested with
`fetch/qwen3-0.6b-gguf.yaml` (0.64 GB).

**Result: [verified]**

| arm | tok/s | speedup | greedy identity | VRAM |
|---|---:|---:|---:|---:|
| target-only | 40.35 | 1.00x | 100% (reference) | 0.01 GB* |
| target-only-rerun | 40.56 | 1.01x | 100.0% | 0.01 GB* |
| target+ngram-simple (0 GB) | 41.27 | 1.02x | 94.0% | 0.01 GB* |

\* the job reads `nvidia-smi memory.used`, which the sandbox's per-job namespace
reports as ~0; the JSON is honest about what was measured, the number is not a
model-size figure.

**Verdict: fail** (`state/exp/04-speculative-decoding.json`, n=50, 7.4
GPU-minutes). Best speculative arm is 1.02x, far below the 1.5x rule, and
ngram-simple changed 6% of the greedy output, so it is not the lossless trade the
card asks for. The draft arm did not run: the 0.64 GB fetch failed mid-download
(peer closed at 466/639 MB, then DNS failures — `fetch/qwen3-0.6b-gguf.log`) and
the job recorded `FileNotFoundError` in `errors.draft` instead of inventing a
number. The draft question is therefore open, not answered. The target-only
rerun reproduced the base run exactly (100% identity), which is what makes the
94% ngram difference a real signal rather than sampling noise.

## Card 5 — compiled prompts vs hand-written prompts

**Rule.** Pass if the compiled prompt beats the hand prompt by ≥2 test-F1 with
≤100 dev evaluations.

**Setup.** `tools/exp/card5_compiled_prompts.py`, the same 9B, hard sales-brain
task: 30 dev / 60 test rows over 7 deal fields, 8 paraphrases × 3 few-shot sets
(24 dev evaluations), plus a judge-scored arm B′.

**Result, hard task — [verified]**

| arm | test F1 | dev F1 | dev evals |
|---|---:|---:|---:|
| A hand-prompt | 87.86 | 89.05 | 1 |
| B compiled (exact dev F1) | 93.57 | 93.81 | 24 |
| B′ compiled (LLM judge) | 87.86 | 89.05 | 24 |

**Verdict: pass** (`state/exp/05-compiled-prompts.json`, n=60, 13.1
GPU-minutes). Arm B gains +5.71 test F1 over the hand prompt with 24 dev
evaluations (≤100) and a dev–test gap of 0.24, so the gain survives held-out
testing. Arm B′ is the honest failure inside the win: the LLM judge scored every
dev config 100.0 (no signal), so it selected paraphrase 0 / example-set 0, which
is exactly the hand prompt — gain 0.00. Metric-driven prompt search paid; the
judge proxy did not.

**Easy-task first run (superseded).** A/B/B′ all 100.0 test-F1, so the +2-point
rule could not be met and the dev–test gap was 0.

## What this means with one 24 GB GPU

- **Verified default now:** serve the 4-bit GGUF. Card 3 is the clean win: same
  F1, 2.13x tokens/s, 5.2 GB instead of 16.1 GB (leaves room for larger context
  or a second model).
- **Test-time compute did not pay here.** Card 1's majority vote bought 1.26 F1
  for 4x the tokens and lost on F1 per 1k tokens. Prefer a better prompt
  (card 5, +5.71 F1 for 24 dev evaluations) over more samples.
- **Speculative decoding is still open.** The 0 GB ngram alternative gave 1.02x
  at 94% output identity — neither fast enough nor lossless. The 0.6B draft never
  downloaded, so the card gives no draft number.
- **Quantization is the only settled serving decision; retrieval vs tuning is
  not measured yet.** Card 2's corrected job is queued; until its file exists the
  default stays "don't fine-tune a 1.7B you can retrieve into".
- **The failure mode to watch:** an easy fixture makes every arm tie at the
  ceiling (cards 1 and 5, first run). The hard sales-brain transcripts are what
  made these cards informative.
