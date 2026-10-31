# Memory & context shortlist (research R8) — the 5 to test first

Picked from `docs/memory-landscape.md`. The baseline is `data/kb.sqlite` (SQLite +
`sqlite-vec` KNN + FTS5) plus a cross-encoder reranker
(`~/homelab/bin/rerank_service.py`, `127.0.0.1:9876`). Every test below is runnable
offline, on our data, and states the metric and the decision rule up front.

Rules for all five tests: no installs on the live system — create a repo-local `.venv`
and list packages in `requirements.txt`, or queue a GPU job in `gpu_jobs/` for anything
that needs the 3090. Any model download > 1.5 GB goes through the cost gate (an
`asks/` card) before fetching. Reuse existing assets first: `rerank_service`,
`embed_semantic` (Qdrant `transcripts_v2`), `data/kb.sqlite`.

## T1 — sqlite-vec + FTS5 hybrid, reranked by FlashRank (baseline hardening)

- **Candidate:** `sqlite-vec` (already in the baseline) + FTS5 + FlashRank
  (`github.com/PrithivirajDamodaran/FlashRank`, Apache-2.0, ~4 MB ONNX).
- **Why:** cheap, offline, and it removes the `rerank_service` HTTP hop. It is the
  control every other test is measured against.
- **Data:** 120 held-out questions written from `data/kb.sqlite` claims (video id +
  claim + the claim text as the gold passage). Build once, keep under `state/` (ignored).
- **Procedure:** (1) as-is baseline: sqlite-vec top-50 + `rerank_service`; (2) FTS5
  top-50; (3) RRF-fuse (2) and (3); (4) rerank the fused top-50 with FlashRank
  `ms-marco-TinyBERT-L-2-v2`. Score each with `recall@10` and `MRR@10`, and record
  wall-clock ms/query (CPU).
- **Metric:** recall@10, MRR@10, p95 latency (ms).
- **Decision rule:** if hybrid+FlashRank is within 2% recall of the baseline *and*
  ≥2x faster, adopt and retire the HTTP reranker. Otherwise keep the baseline and only
  take the RRF fusion.
- **Cost:** ~0 GPU; one afternoon of labelling; FlashRank download < 10 MB.

## T2 — LMCache + vLLM/SGLang prefix caching (KV reuse across turns and workers)

- **Candidate:** LMCache (`github.com/LMCache/LMCache`, Apache-2.0) on vLLM
  (`--enable-prefix-caching`) and SGLang RadixAttention.
- **Why:** the R8 requirement "prefix caching across parallel workers, cache reuse
  between turns" — measure whether our agent's repeated system prompt + growing history
  stops being re-prefilled.
- **Data:** 200 real agent turns (prompts only, sanitised) from the trajectory store;
  three shapes: same long prefix, shared-prefix different suffix, no shared prefix.
- **Procedure:** on the 3090, serve one 8B–9B 4-bit model three ways — vLLM (prefix cache
  off), vLLM (prefix cache on), SGLang — with and without LMCache CPU offload. For each
  request log TTFT, total latency, tokens/s, and the *exact output text*.
- **Metric:** TTFT reduction vs cache-off; tokens/s; **output identity rate** (byte-equal
  greedy outputs) must be 100%.
- **Decision rule:** adopt prefix caching if TTFT drops ≥30% on the shared-prefix shape
  and output identity is 100%; adopt LMCache only if CPU offload lets us keep a second
  prefix resident without OOM. Any identity regression → reject.
- **Cost:** one `gpu_jobs/` run, ≤ 60 min, vram_gb 16. Models must already be in
  `~/.cache/huggingface` (no network in the GPU queue). No new download.

## T3 — LLMLingua-2 context compression before the prompt

- **Candidate:** `microsoft/LLMLingua` (`github.com/microsoft/LLMLingua`, MIT);
  LLMLingua-2 base token classifier (~1.1 GB, under the 1.5 GB gate).
- **Why:** attacks the token budget directly — the same retrieved context with fewer
  tokens. Complements the baseline's `EmbeddingsFilter`-style pruning.
- **Data:** 120 answerable question/context pairs from `data/kb-extracts.jsonl`; context
  = the gold passage plus 4 distractor passages.
- **Procedure:** (1) no compression; (2) embedding-filter (drop chunks with cosine <
  threshold); (3) LLMLingua-2 at 2x, 4x, 8x. Ask a fixed local model to answer, then score
  the answer against the gold claim with an exact-substring/field-F1 check (reuse the
  R4 extractor harness in `tools/exp/`).
- **Metric:** answer accuracy / field-F1, prompt tokens, compression ratio, compression
  latency (ms).
- **Decision rule:** adopt LLMLingua-2 only if accuracy drops < 2 points at ≥4x
  compression and the compressor adds < 100 ms/query on CPU; otherwise adopt only
  embedding-filter.
- **Cost:** one CPU pass + one local-batch run for answers; ~1.1 GB download (gated but
  under 1.5 GB, still write `fetch/llmlingua.yaml` so the runner fetches it).

## T4 — Serena symbol-level context over this repo

- **Candidate:** `oraios/serena` (`github.com/oraios/serena`, MIT).
- **Why:** tests "code-aware context servers (symbol-level retrieval)" on the one codebase
  we know best, and it must beat plain file reads / grep on tokens without losing the
  right answer.
- **Data:** 40 code questions whose gold answer is a symbol (function/class) in this repo,
  e.g. "where is the KB built?", "who calls `build_lens`?", "what does `--check` do?".
  Include 10 negative questions whose answer is "not in this repo".
- **Procedure:** answer each question with (a) `grep`/read as today and (b) Serena's
  symbol tools; record retrieved files/symbols, total prompt tokens, latency, and whether
  the gold symbol is in the retrieved set.
- **Metric:** symbol retrieval precision/recall, prompt tokens per correct answer, latency.
- **Decision rule:** adopt if Serena keeps recall ≥ 0.9 with ≥30% fewer tokens; if it
  only matches grep, skip (grep is already installed).
- **Cost:** repo-local `.venv` with Serena; one language server (Python); no GPU, no
  download > 50 MB.

## T5 — Model2Vec retriever + SetFit "enough context?" classifier

- **Candidate:** Model2Vec (`github.com/MinishLab/model2vec`, MIT) as the first-stage
  retriever; SetFit (`github.com/huggingface/setfit`, Apache-2.0) and fastText
  (`github.com/facebookresearch/fastText`, MIT) as the tiny CPU classifiers.
- **Why:** the two remaining requirements meet here — millisecond CPU retrieval, and a
  local decision on whether the retrieved context is *enough* to answer (routing /
  abstain).
- **Data:** the same 120 labelled pairs as T1, plus 40 questions deliberately made
  unanswerable from the KB. Labels: `enough` (answer is in the top-k passages) /
  `not_enough`.
- **Procedure:** (1) compare Model2Vec retrieval against sqlite-vec for recall@50 and
  ms/query; (2) train SetFit on 24 examples and fastText on the same 24, test on 136;
  (3) also test a zero-training baseline that thresholds the reranker's top score.
- **Metric:** retrieval recall@50 and p95 ms; classifier accuracy, F1 on `not_enough`,
  inference ms.
- **Decision rule:** adopt Model2Vec as stage-1 if recall@50 is within 3 points of
  sqlite-vec and it is ≥5x faster; adopt the classifier only if it beats the
  threshold baseline on `not_enough` F1 at < 20 ms/query. Otherwise keep sqlite-vec and
  the score threshold.
- **Cost:** CPU only; SetFit encoder (~30–110 MB) and Model2Vec (~30–120 MB) downloads
  are under the 1.5 GB gate; label 40 negatives by hand.

## Order and stop rule

Run T1 first (it defines the metric); then T4 (cheap, no GPU) while T2 sits in the GPU
queue; T3 and T5 after T1's labels exist. Stop after each test and record the result in
`PROGRESS.md`; only the tests whose decision rule says "adopt" become dependencies of the
agent. Anything that needs the owner's money or a > 1.5 GB download becomes an `asks/`
card, not a silent fetch.
