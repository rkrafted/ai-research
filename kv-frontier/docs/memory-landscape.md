# Memory, context & KV-cache landscape (research R8)

Date: 2026-10-04. Scope: things that could improve how this project gives a
compute-limited agent useful memory and a useful context window on one 24 GB GPU.
Judged against a **plain baseline** we already run:

> **Baseline:** `data/kb.sqlite` (SQLite; 3,246 videos, claims/papers/methods/links) +
> `sqlite-vec` (KNN over embeddings) + `FTS5` (keyword) + a cross-encoder reranker
> (`~/homelab/bin/rerank_service.py`, `127.0.0.1:9876`). No service beyond SQLite;
> milliseconds per query on CPU; everything reproducible offline.

## Method (in order, per project rules)

1. **Local inventory first** — `python3 -m inventory search` in
   `~/theprojects/queue/inventory`. Only hit: `rerank_service` (cross-encoder HTTP
   service, active) and `embed_semantic` / `build_concept_graph` (Qdrant "transcripts_v2"
   + concept graph under `~/homelab/bin`). The baseline above already wraps this, so R8
   reuses `rerank_service` rather than rebuilding a reranker.
2. **Public repo finder** — `prior_art check` on the R8 idea
   (`~/theprojects/queue/prior-art`, cache `/tmp/prior-art`). Verdict: **not already
   built** — hits were unrelated (SQLite forks, generic cache videos). Recorded as weak
   overlap; no component to fork.
3. **Data KB** — `data/kb.sqlite` queried for `memory`, `KV cache`, `context`,
   `compression`, `retrieval` topics to ground claims (e.g. the *Attention Sink*,
   *Distributed Inference KV-cache routing*, and *TurboQuant 6x* videos).
4. Licence of every GitHub candidate was read from the repo's own `LICENSE` file at
   session time (raw.githubusercontent.com); "unclear" means no root licence file.

## Verdict key

- **adopt** — clearly better than the baseline at our scale; low cost; do it.
- **test** — plausible win; needs a measured trial on our data before adopting.
- **skip** — heavier/slower/riskier than the baseline, or licence/ops block us.

No installs happen on the live system; every *test* is queued or run in a repo-local
`.venv`. Any download > 1.5 GB goes through the cost gate (ask) before fetching.

---

## 1. Agent memory layers (long-term / graph / consolidation)

### mem0 — https://github.com/mem0ai/mem0
- **Licence:** Apache-2.0 · **Needs:** Python; a vector store (can be SQLite) + an LLM for
  fact extraction/update · **Size:** ~5 MB, CPU.
- **What:** long-term memory API. Extracts facts from turns, dedups/updates them across a
  vector store and optional graph store, then retrieves a compact memory block.
- **Claims:** paper reports ~26% relative accuracy gain over full-context on LOCOMO,
  ~90% token reduction, ~91% lower p95 latency vs full context.
- **Verdict:** **test** — closest OSS match to "agent memory". It sits *on top* of our
  SQLite baseline; the cost is the extraction LLM calls.

### Graphiti (Zep) — https://github.com/getzep/graphiti
- **Licence:** Apache-2.0 · **Needs:** Python + Neo4j/FalkorDB (or in-memory driver);
  LLM for entity extraction · **Size:** ~10 MB, CPU; graph DB service.
- **What:** temporally-aware knowledge graph memory. Each fact is an edge with validity
  intervals, so memory can be invalidated/superseded rather than overwritten.
- **Claims:** Zep reports large gains on DMR/LongMemEval vs a RAG baseline; exact numbers
  in the repo README change by release.
- **Verdict:** **test** (only if we need fact invalidation) — otherwise the extra graph
  service is more ops than the baseline. KB signal: KB has several "graph memory"
  talks, but no benchmark we trust yet.

### Letta / MemGPT — https://github.com/letta-ai/letta
- **Licence:** Apache-2.0 · **Needs:** Python; LLM backend; PostgreSQL for production,
  SQLite for local · **Size:** ~20 MB, CPU.
- **What:** agent runtime with explicit memory tiers (core / recall / archival) and
  self-editing memory; "sleep-time compute" does offline memory consolidation.
- **Claims:** MemGPT paper: long-document QA and multi-session chat with a fixed context
  window; no single headline % to reproduce cheaply.
- **Verdict:** **skip as a dependency, adopt the ideas** — too invasive to run our agent
  inside Letta. Worth copying: separating hot core memory from cold archival recall.

### Cognee — https://github.com/topoteretes/cognee
- **Licence:** Apache-2.0 · **Needs:** Python; vector store + graph store (LanceDB/Kuzu
  defaults); LLM for extraction · **Size:** ~15 MB, CPU.
- **What:** ECL (extract–cognify–load) pipeline that turns documents into a graph+vector
  memory, with a search layer over both.
- **Claims:** project benchmarks vs naive RAG on its own datasets; no third-party
  reproduction.
- **Verdict:** **skip** — solves the same problem as Graphiti with more moving parts; our
  KB is already structured.

### Microsoft GraphRAG — https://github.com/microsoft/graphrag
- **Licence:** MIT · **Needs:** Python; LLM for entity/community summarisation (expensive)
  · **Size:** ~10 MB code; index build costs many LLM calls.
- **What:** build a knowledge graph from a corpus, detect communities, summarise them, and
  answer local (entity) and global (theme) queries.
- **Claims:** paper: substantially better comprehensiveness/diversity on global
  sensemaking vs naive RAG; index cost is high.
- **Verdict:** **skip** — index build is LLM-cost-prohibitive for a 3,246-video corpus we
  can already FTS5 + vector search. Keep for a future "themes across all videos" ask.

### Memobase — https://github.com/memodb-io/memobase
- **Licence:** Apache-2.0 · **Needs:** Python; Postgres + vector store; LLM for writing
  profiles · **Size:** ~8 MB, CPU.
- **What:** user-profile memory service: buffers events, then asynchronously builds a
  structured evolving profile per user.
- **Claims:** README claims better persona consistency than raw chat history; no
  independent benchmark.
- **Verdict:** **skip** — multi-user profile memory is not our shape (single owner,
  document KB).

### Basic Memory — https://github.com/basicmachines-co/basic-memory
- **Licence:** AGPL-3.0 · **Needs:** Python; filesystem (Markdown) + SQLite index · **Size:**
  ~5 MB, CPU.
- **What:** agent memory as local Markdown notes with SQLite FTS search; MCP server so an
  agent can read/write notes.
- **Claims:** none quantitative; "local-first, human-readable memory".
- **Verdict:** **skip** — AGPL-3.0 is a licence risk for vendored code, and it is a thinner
  version of our SQLite+FTS baseline.

---

## 2. Code-aware context servers (symbol-level retrieval)

### Serena — https://github.com/oraios/serena
- **Licence:** MIT · **Needs:** Python + a language server per language (LSP); no GPU;
  no network · **Size:** ~10 MB.
- **What:** MCP server that gives an agent *symbol-level* tools (find symbol, find
  references, rename, insert) via LSP, instead of dumping whole files.
- **Claims:** project shows large token reductions vs reading files; no formal benchmark.
- **Verdict:** **test** — exactly the "code-aware context server" requirement, MIT,
  offline, and it fits our own repo first. KB signal: MCP tooling is trending (20+ videos).

### Claude Context (Zilliz) — https://github.com/zilliztech/claude-context
- **Licence:** MIT · **Needs:** Node/Python; an embedding model (local or API) + vector DB
  (Milvus/Zilliz) · **Size:** ~10 MB; embedding compute.
- **What:** MCP server for semantic code search over a repo: AST-aware chunking, hybrid
  search, then retrieve.
- **Claims:** README shows token savings and faster answers vs whole-repo context; no
  peer benchmark.
- **Verdict:** **test** — a semantic complement to Serena's exact-symbol lookup; needs an
  embedding model we already have locally.

### code-graph-rag — https://github.com/SShadowS/code-graph-rag
- **Licence:** MIT · **Needs:** Python; tree-sitter + Memgraph (Docker) + embeddings ·
  **Size:** ~20 MB + graph service.
- **What:** builds a multi-language AST graph of a monorepo and answers structural queries
  ("who calls this", "what breaks if I change this").
- **Claims:** no formal numbers; positions itself as "the ultimate RAG for a monorepo".
- **Verdict:** **skip** — requires a graph database service; Serena + grep covers the same
  queries on our repo size.

### Potpie — https://github.com/potpie-ai/potpie
- **Licence:** Apache-2.0 · **Needs:** Python; Neo4j + LLM + embeddings; Docker-compose ·
  **Size:** ~30 MB + services.
- **What:** turns a repo into a code knowledge graph and runs "agents" (debug, test,
  code-review) over it.
- **Claims:** no reproducible numbers; product-oriented.
- **Verdict:** **skip** — heavyweight multi-service stack for our single-repo needs.

### ast-grep — https://github.com/ast-grep/ast-grep
- **Licence:** MIT · **Needs:** a single Rust binary; tree-sitter grammars · **Size:** ~10 MB,
  CPU, milliseconds.
- **What:** structural search/replace by AST pattern (`sg -p 'fn $A($$$)'`). Deterministic,
  no model, no network.
- **Claims:** not a benchmark product; correctness of AST matching is the feature.
- **Verdict:** **adopt** — cheap, offline, deterministic glue for any code-context server;
  pairs with Serena and with our `grep` tool.

### Repomix — https://github.com/yamadashy/repomix
- **Licence:** MIT · **Needs:** Node; no services · **Size:** ~10 MB.
- **What:** packs a repo into a single LLM-friendly file with token counts and ignore
  rules.
- **Claims:** none; a packaging tool.
- **Verdict:** **skip** — we want retrieval, not "stuff the whole repo"; useful only for a
  one-off audit of a small repo.

---

## 3. Context compression

### Microsoft LLMLingua — https://github.com/microsoft/LLMLingua
- **Licence:** MIT · **Needs:** Python + a small LM (LLMLingua-2 uses an XLM-R/mBERT
  encoder) · **Size:** LLMLingua-2 base encoder ~1.1 GB; runs CPU or GPU.
- **What:** drop low-information tokens from prompts/retrieved context while preserving
  meaning (LLMLingua-2 is a trained token classifier, fast and task-agnostic).
- **Claims:** paper: 2x–20x compression with little performance loss on reasoning/QA.
- **Verdict:** **test** — directly attacks our context budget. Download size crosses the
  1.5 GB gate only for the large variant; prefer LLMLingua-2 base.

### Provence (Naver Labs) — https://github.com/naver/provence
- **Licence:** unclear (no root LICENSE file found) · **Needs:** Python + a DeBERTa-scale
  model · **Size:** ~1.5 GB.
- **What:** context pruning as reranking: jointly selects the sentence spans relevant to
  the question and can drop everything else.
- **Claims:** paper: strong zero-shot QA/AST results with large context reduction.
- **Verdict:** **skip pending licence** — good idea, but we cannot vendor an
  unlicensed repo; revisit if a licence file is added.

### LangChain contextual compression — https://github.com/langchain-ai/langchain
- **Licence:** MIT · **Needs:** Python; the compressor's model (EmbeddingsFilter needs
  only embeddings) · **Size:** package ~30 MB.
- **What:** `ContextualCompressionRetriever` + `EmbeddingsFilter` / `LLMChainExtractor`:
  prune retrieved documents before they enter the prompt.
- **Claims:** not a research claim; an integration pattern.
- **Verdict:** **adopt the EmbeddingsFilter pattern** — we already have embeddings, so
  cosine-filtering retrieved chunks is a few lines and no new service.

---

## 4. KV cache for long context on 24 GB

### NVIDIA kvpress — https://github.com/NVIDIA/kvpress
- **Licence:** Apache-2.0 · **Needs:** Python, HF `transformers` + CUDA · **Size:** library
  ~1 MB (models come from the HF cache).
- **What:** unified framework of KV-cache compression "presses" (SnapKV, PyramidKV,
  H2O-style, expected-attention, etc.), measurable with a built-in benchmark.
- **Claims:** reproduces published compression ratios (often 2x–8x KV reduction) at
  near-equal accuracy; numbers vary per method.
- **Verdict:** **adopt as the test harness** — one place to run the KV methods below on
  our 3090 instead of reimplementing each.

### LMCache — https://github.com/LMCache/LMCache
- **Licence:** Apache-2.0 · **Needs:** Python + vLLM; CPU RAM or disk tier for KV · **Size:**
  ~5 MB library.
- **What:** KV-cache layer that stores/reuses KV blocks across requests, turns and engine
  instances (prefix reuse, offload to CPU/disk).
- **Claims:** project reports multi-x TTFT reduction and higher throughput on repeated
  prefixes; context-dependent.
- **Verdict:** **test** — the direct answer to "prefix caching across parallel workers,
  cache reuse between turns". Same GPU job can measure output identity.

### Mooncake — https://github.com/kvcache-ai/Mooncake
- **Licence:** Apache-2.0 (license file not at repo root at session time) · **Needs:**
  C++/Python + RDMA/ethernet for the transfer engine; KVCache store · **Size:** large
  build.
- **What:** KVCache-centric disaggregated serving (prefill/decode separation) plus a
  transfer engine; from Moonshot AI's Kimi serving stack.
- **Claims:** paper: large TTFT/throughput gains in a disaggregated cluster.
- **Verdict:** **skip** — designed for multi-node clusters; single 3090 has one worker,
  so the transfer engine buys nothing.

### NVIDIA Dynamo — https://github.com/ai-dynamo/dynamo
- **Licence:** MIT · **Needs:** Python/Rust + a serving engine (vLLM/SGLang/TRT-LLM);
  Rust toolchain · **Size:** ~50 MB build.
- **What:** datacenter inference framework with KV-cache-aware routing, disaggregated
  prefill/decode and dynamic GPU scheduling.
- **Claims:** NVIDIA reports multi-x throughput on large deployments.
- **Verdict:** **skip** — cluster-scale scheduler; our KB video on KV-aware routing is
  the concept to borrow, not the software. (KB: *Distributed Inference 101: KV Cache-Aware
  Smart Router*, NVIDIA Developer.)

### SGLang — https://github.com/sgl-project/sglang
- **Licence:** Apache-2.0 · **Needs:** Python + CUDA; GPU · **Size:** package ~50 MB.
- **What:** serving engine whose RadixAttention keeps a radix tree of KV prefixes so
  parallel/sequential requests share cached prefix KV automatically.
- **Claims:** paper: up to ~5x throughput vs vLLM on prefix-heavy workloads (their
  measurement).
- **Verdict:** **test** — the single most practical "prefix caching across parallel
  workers" lever on one GPU; compare against vLLM on our prompts.

### vLLM — https://github.com/vllm-project/vllm
- **Licence:** Apache-2.0 · **Needs:** Python + CUDA · **Size:** package ~100 MB.
- **What:** paged-attention serving engine with Automatic Prefix Caching (`--enable-prefix-caching`).
- **Claims:** PagedAttention paper: near-zero KV waste, 2x–4x throughput vs prior serving.
- **Verdict:** **adopt as the baseline server** — we already use it (23 KB videos); turning
  prefix caching on for repeated system prompts is free throughput.

### StreamingLLM — https://github.com/mit-han-lab/streaming-llm
- **Licence:** MIT · **Needs:** Python + CUDA for the reference impl · **Size:** ~1 MB code.
- **What:** keeps a few "attention sink" tokens plus a sliding window so a model can stream
  indefinitely with bounded KV.
- **Claims:** paper: stable perplexity over 4M+ tokens with a 4K window.
- **Verdict:** **skip for our agent** — quality drops on recall beyond the window. KB
  signal: the *Attention Sink* video explains why removing token 0 breaks it (useful
  background, not a memory solution).

### SnapKV — https://github.com/FasterDecoding/SnapKV
- **Licence:** Apache-2.0 · **Needs:** Python + CUDA; HF model · **Size:** ~1 MB code.
- **What:** selects important KV positions per attention head using the prompt's own
  attention, then keeps a fixed budget.
- **Claims:** paper: ~3.6x generation speedup, 8.2x memory efficiency with minor accuracy
  loss on long-context tasks.
- **Verdict:** **test via kvpress** — deterministic-ish budget reduction; we must verify
  output identity on our own prompts before trusting it.

### KIVI — https://github.com/jy-yuan/KIVI
- **Licence:** MIT · **Needs:** Python + CUDA · **Size:** ~1 MB code.
- **What:** 2-bit per-channel KV quantisation (key: per-channel, value: per-token) with a
  full-precision residual window.
- **Claims:** paper: 2.35x–3.47x less memory, ~2.3x–3.9x higher throughput, near
  full-precision quality on long-context benchmarks.
- **Verdict:** **test via kvpress** — the "quantisation that keeps outputs identical"
  requirement; measure exact-match on our extraction outputs, not just perplexity.

### Prompt Cache — https://github.com/gimmaru/prompt-cache
- **Licence:** unclear (no root LICENSE file found) · **Needs:** Python + CUDA · **Size:**
  ~1 MB code.
- **What:** modular attention reuse: precompute KV for reusable prompt segments and splice
  them for different requests.
- **Claims:** paper: up to ~8x TTFT reduction for prompts sharing segments.
- **Verdict:** **skip pending licence** — conceptually superseded by LMCache/SGLang for
  our case; revisit if licensed.

---

## 5. Small fast rerankers / retrievers (CPU, milliseconds)

### FlashRank — https://github.com/PrithivirajDamodaran/FlashRank
- **Licence:** Apache-2.0 · **Needs:** Python + ONNX Runtime (CPU); no GPU, no network
  after model fetch · **Size:** ~4 MB per tiny model (e.g. `ms-marco-TinyBERT-L-2-v2`).
- **What:** ultra-light cross-encoder reranker: ONNX quantised models with a simple API.
- **Claims:** README: ~1 ms–10 ms per query on CPU, ~4 MB models, quality near the
  `bge-reranker-base` class.
- **Verdict:** **test** — our `rerank_service` cross-encoder is heavier; FlashRank on CPU
  is the "milliseconds" requirement. Candidate replacement for the rerank step.

### rerankers (Answer.AI) — https://github.com/AnswerDotAI/rerankers
- **Licence:** Apache-2.0 · **Needs:** Python; optional ONNX/TFLite backends · **Size:** ~5 MB.
- **What:** one API over cross-encoders, ColBERT late-interaction and ONNX tiny rerankers,
  so rerankers are swappable.
- **Claims:** benchmark table in README across models; not a new method.
- **Verdict:** **adopt as the interface** — lets the shortlist test compare FlashRank vs
  ColBERT vs `bge` without glue code.

### Model2Vec — https://github.com/MinishLab/model2vec
- **Licence:** MIT · **Needs:** Python + NumPy; CPU only · **Size:** distilled models
  8M–30M params (~30–120 MB).
- **What:** distils a sentence-transformer into static token embeddings; retrieval becomes
  a lookup + mean-pool, no transformer forward pass.
- **Claims:** README: ~500x faster on CPU than sentence-transformers, 2x smaller than
  GloVe-scale models, quality close on MTEB retrieval subsets.
- **Verdict:** **test** — candidate first-stage retriever in front of the reranker; if
  recall@50 holds, it makes local search instant and offline.

### bm25s — https://github.com/xhluca/bm25s
- **Licence:** MIT · **Needs:** Python + NumPy/SciPy; CPU · **Size:** ~1 MB.
- **What:** fast sparse BM25 with optional `scipy.sparse` scoring; a drop-in for `rank_bm25`.
- **Claims:** README: 50x–500x faster than `rank_bm25`; memory-efficient.
- **Verdict:** **skip** — SQLite FTS5 already gives us BM25-like keyword search inside the
  baseline; no reason to add a second index.

### Tantivy — https://github.com/quickwit-oss/tantivy
- **Licence:** MIT · **Needs:** Rust (or a Python binding) · **Size:** ~10 MB.
- **What:** Rust full-text search engine (Lucene-class) with BM25 and tokenisers.
- **Claims:** Lucene-competitive indexing/search speed.
- **Verdict:** **skip** — FTS5 is the lazy correct choice inside the baseline; Tantivy only
  wins at corpora far larger than 3,246 docs.

### sqlite-vec — https://github.com/asg017/sqlite-vec
- **Licence:** MIT OR Apache-2.0 (dual; per-file licence files) · **Needs:** a SQLite
  extension, no service · **Size:** < 1 MB.
- **What:** vector search (brute-force KNN + metadata) inside SQLite; runs anywhere SQLite
  does, including WASM.
- **Claims:** README: brute-force KNN is fast enough up to ~1M vectors; zero-dependency.
- **Verdict:** **adopt (already the baseline)** — keep. It removes a whole vector-DB
  service from the stack.

### ColBERT (late interaction) — https://github.com/stanford-futuredata/ColBERT
- **Licence:** MIT · **Needs:** Python + PyTorch; GPU or CPU (slow on CPU) · **Size:** model
  ~400 MB+; index scales with corpus.
- **What:** late-interaction retrieval: per-token embeddings and MaxSim, giving strong
  zero-shot retrieval with cheap re-ranking.
- **Claims:** paper: strong MS MARCO / BEIR results, notably better than single-vector at
  equal index cost.
- **Verdict:** **skip** — index size and CPU latency do not beat sqlite-vec + a tiny
  cross-encoder at our corpus size; revisit if recall fails.

---

## 6. Tiny classifiers for routing / "enough context?"

### fastText — https://github.com/facebookresearch/fastText
- **Licence:** MIT · **Needs:** C++ build + Python wrapper; CPU · **Size:** < 1 MB model.
- **What:** bag-of-ngrams linear text classifier; trains in seconds and predicts in
  microseconds.
- **Claims:** paper: comparable accuracy to deep models on many text-classification tasks
  at a fraction of the cost.
- **Verdict:** **test** — the cheapest possible router / "is this query answerable from the
  retrieved context?" gate; needs a labelled set we can derive from kb.sqlite.

### SetFit — https://github.com/huggingface/setfit
- **Licence:** Apache-2.0 · **Needs:** Python + `sentence-transformers`; CPU train · **Size:**
  encoder 30–110 MB.
- **What:** few-shot text classification by contrastive fine-tuning a sentence encoder then
  fitting a small head; works with tens of examples.
- **Claims:** paper: matches or beats fine-tuned RoBERTa with far fewer labels; strong
  few-shot.
- **Verdict:** **test** — for the "enough context?" decision we have almost no labels, so
  few-shot is the right shape. Compare against fastText once a set exists.

### RouteLLM — https://github.com/lm-sys/RouteLLM
- **Licence:** Apache-2.0 · **Needs:** Python + OpenAI-compatible endpoints + a router model
  · **Size:** router ~100 MB.
- **What:** routes each query to a cheap or strong model by a trained preference-based
  router; includes matrix-factorisation routers.
- **Claims:** paper: ~85% of GPT-4 quality at ~45% of the cost (their setting).
- **Verdict:** **skip** — we are not routing between paid models; the idea of a small
  learned router is reusable, the code is not.

### Semantic Router — https://github.com/aurelio-labs/semantic-router
- **Licence:** MIT · **Needs:** Python + an embedding model (local or API) · **Size:** ~1 MB.
- **What:** intent routing by embedding similarity to utterance examples; no classifier
  training, sub-100 ms decisions on CPU.
- **Claims:** no formal benchmark; latency-oriented.
- **Verdict:** **test** — good fit for "which memory/tool should answer this?"; we already
  generate embeddings, so it is a few lines.

### ModernBERT — https://github.com/AnswerDotAI/ModernBERT
- **Licence:** Apache-2.0 · **Needs:** Python + PyTorch; CPU inference fine · **Size:** base
  ~600 MB, small ~140 MB.
- **What:** modern encoder-only transformer (RoPE, flash-attn, 8K context) that is an
  excellent backbone for classification and reranking.
- **Claims:** paper: state-of-the-art encoder results and ~2x–4x faster inference than
  previous BERT-class models.
- **Verdict:** **test** — strong backbone if a tiny classifier's accuracy plateaus; use the
  `-small` variant to stay CPU-fast.

---

## Summary

| Layer | Candidates | Adopt now | Test | Skip |
|---|---|---|---|---|
| Agent memory | 7 | 0 | 2 | 5 |
| Code context | 6 | 1 (ast-grep) | 2 | 3 |
| Compression | 3 | 1 (EmbeddingsFilter) | 1 | 1 |
| KV cache | 10 | 2 (vLLM, kvpress) | 4 | 4 |
| Rerankers / retrievers | 7 | 2 (sqlite-vec, rerankers) | 2 | 3 |
| Tiny classifiers | 5 | 0 | 4 | 1 |
| **Total** | **38** | **6** | **15** | **17** |

**What is new vs the plain baseline, in one line each:** FlashRank/Model2Vec make
retrieval milliseconds-cheap; LMCache + SGLang/vLLM prefix caching reuse KV across turns
and workers; LLMLingua-2 and embedding-filters cut context before it costs tokens;
kvpress/KIVI/SnapKV test "quantisation that keeps outputs identical"; Serena + ast-grep
give symbol-level code context; SetFit/Semantic Router decide "enough context?" locally.

**Downloads and cost gate:** from this landscape only LLMLingua-2 (~1.1 GB) and
ModernBERT-base (~600 MB) are dependency downloads; both are under the 1.5 GB gate. The
LLMLingua large variant and ColBERT indexes would exceed it and must go through the cost
gate (ask) before fetching. Nothing here is fetched in this session.
