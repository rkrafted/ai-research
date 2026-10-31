# Codacus — 30 videos, buildable ideas for a 3090 + 30 GB RAM

Status: **complete for the 30-video window.** `data/codacus/videos.json` has 30
uploads, `data/codacus/<id>.txt` the captions, `data/codacus/<id>.json` the local
9B extraction, `docs/spot-checks-codacus.md` the audit. This document is the
DeepSeek reverse-engineering pass and the ranked shortlist. No third-party code
was installed or run; repos were read only, and the two video links from the
brief (`pIN-2oVJpyU`, `bcGO7xre46o`) are both in the window.

House tools considered first: the repo's `llama.cpp` pin **f3363593**,
`bin/local_lane.py`, **Laya** (`laya-mcp /ask`, ~60 ms CPU typed decisions) and
**q2** (the runner). The local 9B did the per-video extraction through the
`local_batch` tool; DeepSeek did only this synthesis. Reused prior art:
`airesearch/ingest.py` (caption fetch/rate-limit), `airesearch/kb.py` (SQLite
knowledge base), the `local_batch` house tool, and the local batch schema from
`codacus/extract.py`.

## Method

1. **Evidence first.** Each row cites the video id and, where a number matters,
   the exact transcript figure. Claims are the video's claims unless marked
   measured. Code findings cite a file/line or a merged PR.
2. **Source check.** For every mechanism that touches a goal I asked "is there
   source code?" and used the repo finder (`prior_art check`) plus
   `~/shithub/bin/gh search repos`. Licence, stars and install path were read
   before reading code. No install, no third-party execution.
3. **REA pass.** REA = **`morluto/rea`**, MIT, **305,134 stars**, pushed
   2026-10-17, found by `~/shithub/bin/gh search repos "Reverse Engineer
   Anything"`. Other matches, listed as required: `NabilRFG12/Reverse-Engineer-anything`
   (688, MIT), `aicodeking/leanrea` (325, MIT), `valorisa/Reverse-Engineer-Anything`
   (guide), `REA-Gith/REA` (mirror). The trending one is `morluto/rea`; its
   `install.sh` resolves a GitHub release and needs Node 22+/npm, so it was
   **not installed**. I read its `docs/mcp-prompts.md`: keep observation,
   inference and unknown separate; trace a feature to function-level evidence;
   audit residual unknowns; verify a reconstruction against retained evidence.
   That is the shape of the source table below.
4. **Fits** means: one RTX 3090 (24 GB VRAM) + 30 GB system RAM, llama.cpp /
   Ollama, no network at inference. An idea that needs 64 GB RAM, a second GPU
   or a model we do not run is marked **no** and skipped in one line.

## One row per idea

| # | idea (video) | mechanism (1 line) | goal | fits 3090+30 GB? | effort | expected gain (source) | safe? | recommend |
|---|---|---|---|---|---|---|---|---|
| 1 | REAP-pruned 28B-A3B coder (`0AqpaFm11oI`) | router-weighted expert pruning + Q4 + KV quant lets a 28B-A3B run on 12 GB | model-size, speed | yes (~15 GB Q4) | S download / M prune | video: 12 GB GPU; REAP paper: 50 % experts, 1.9 % mean accuracy drop | yes, with eval | **skip** — our 35B-A3B already fits; pruning needs calibration |
| 2 | DeepSeek V4 CSA/HCA + TurboQuant (`3mX_dc9ZNRA`) | CSA compresses 4x, HCA 128x, Walsh-Hadamard + 1-bit-corrected KV | context | no — needs V4 weights + new attention kernels | L | claimed 4 TB -> 500 MB at 1M; measured 10 % KV, 27 % compute (vLLM, in video) | kernel risk | **skip** — not portable to llama.cpp; port abandoned (see §Source) |
| 3 | Strata single-architecture engine (`6WLBmP-tZ0Q`) | one model only, larger U-batch prefill, prefix cache, expert offload | speed, model-size | no — 125B + 64 GB RAM | L | claimed ~40 tok/s on 12 GB/64 GB | **vet failed** | **skip** — install bootstraps download Python/binaries; needs 64 GB |
| 4 | 35B-A3B on 6 GB: `--n-cpu-moe` (`8F_5pdcD3HY`) | keep attention/KV on GPU, expert FFN in RAM, batch prompt tokens, `--no-mmap` | model-size, speed | yes (24 GB + 30 GB) | S | 3 -> 10 tok/s, +230 % on GTX 1060 6 GB/24 GB (video) | yes, identical output | **build** — top 3 |
| 5 | DFlash speculative decoding (`9vY4-Z-tkHs`) | block-diffusion drafter proposes tokens; target verifies in one pass | speed | yes if target is fully resident | S | 12 -> 40 tok/s, 3.3x on dense 8B; 2.7-3.4x on 27B (DFlash 2) | yes, lossless | **build** — top 1 |
| 6 | Two-GPU layer/tensor split (`FIi2arm_DWM`) | split layers or tensors across two cards | speed | no — one 3090 | S | claimed near 2x | yes | **skip** — no second GPU |
| 7 | Agent identity + per-user ACL (`Gunp20UgDFE`) | separate account, policy page, inbox routing, escalation | other | yes | S | none (workflow) | yes | **skip** — security/workflow, not a model goal |
| 8 | Qwen4-EXP n-gram phrase book (`IH8XmxiwliQ`) | static n-gram table on SSD paged beside the neural net | model-size, speed | no — architecture/model not in our stack | L | claimed 3060-class fit | unknown | **skip** — only helps Qwen4-EXP |
| 9 | understory OKF memory (`IwN-eK1s8og`) | plain-markdown concepts; code (not prompts) enforces frontmatter/index/log; query-first | forgetting/memory | yes | S | no re-embedding/re-chunking; query-first recall (repo + video) | yes, local files | **build** — top 4 |
| 10 | Laguna S 2.1 118B on 12 GB (`NIxCj-3fFNk`) | `NCPU-MoE-46` + IQ4XS + Q8 KV, 8B active | model-size | no — 57.6 GB model, 62 GB RAM | S | 80K context, 1 tok/s class (video) | yes | **skip** — needs 62 GB RAM |
| 11 | TurboQuant 6x KV (`Oq0UdRy1LcM`) | Walsh-Hadamard rotate + beta-distribution quantizer, no calibration | context | partial — needs a working llama.cpp port | L | claimed 6x KV | yes | **skip** — the llama.cpp port is archived and slower (see §Source) |
| 12 | MoE `-ot` routing + Q8 KV (`SsUKTFSQoGM`) | override-tensor per layer; Q8 KV halves cache; Q4 weights | model-size, speed, context | yes | S | video: ~50 % KV; house R5 measured 31.9 % output identity -> reject KV | `-ot` yes, Q8 KV no | **borrow** `-ot` only; Q8 KV already rejected by R5 |
| 13 | Chrome 4 GB model storage (`Tffg78mOgAk`) | automatic model download to a hidden persistent dir; `window.ai` API | other | no relevance | — | none | — | **skip** — policy/legal story, not a build |
| 14 | MoE prefill offload (`VytSYCDhWQ0`) | for prefill, copy a full MoE layer to GPU and batch 2048 tokens; CPU in decode | speed | yes (35B-A3B/12 GB) | M | 1143 -> 1385 tok/s prefill, +21 %, token-identical (video) | yes, verified identical | **build** — useful for ingest, not top 5 |
| 15 | AI subscription pricing (`WkLdJ_1TKdY`) | subsidy, tiers, usage pricing | other | n/a | — | none | — | **skip** — no build |
| 16 | Streaming TTS sentence splitter (`Xuh6PzhqrgQ`) | split tokens at sentence boundaries, synth/play immediately | speed (latency) | yes | S | removes the TTS wait (video) | yes | **skip** — voice stack, not the research pipeline |
| 17 | DiffusionGemma block diffusion (`aYBgCl6ynGk`) | denoise a whole block; revise earlier tokens | speed, hallucination | maybe (26B-A4B Q4) | M | claimed faster/self-correcting; no independent number | unknown | **skip** — model not in our stack, quality unproven |
| 18 | Gemma 4 QAT (`xFEcAGB5kyg`) | 4-bit-aware training, weights stored Q4 | model-size | 31B Q4 ~18 GB | S | BF16 quality at Q4 size (Google claim) | yes | **skip** — only helps Gemma 4, which we do not run |
| 19 | AgentBox Docker sandbox (`akXT0DRvvK4`) | orchestrator + fresh container per tool call + persistent `/workspace` | other | yes | M | isolation, no model gain | yes | **skip** — agent infra, not a goal |
| 20 | Jev mode / parallel decision (`bcGO7xre46o`) | one token per field, score only legal options, KV holds the schema, confidence per field | speed, hallucination | yes | M | 300 ms vs 3.5 s, ~6x on Qwen2.5-1.5B (video); Laya already 60 ms | yes; fields are independent | **borrow** into Laya — top 5 |
| 21 | GGML RPC split (`dypHkOKOLt8`) | sequential pipeline split across Mac + Nvidia over the network | speed, model-size | no — second machine | M | claimed; latency-bound | yes | **skip** — needs a second machine |
| 22 | pi/Hermes local harness (`fpvF4n32lsE`) | local llama.cpp, session cwd, one session file across restarts | speed, hallucination, other | yes | S | workflow, not model | yes | **skip** — local_lane/q2 already own this |
| 23 | clanker_t1000 kernel review (`g_JG1Xbnz0g`) | local LLM reads kernel source, flags bugs, human signs patches | hallucination? | no — 128 GB unified memory | M | 24 bugs claimed | human review | **skip** — hardware and task outside our setup |
| 24 | Expert cache LRU + PCIe cost (`k_LostFpatg`) | pin hot experts in VRAM, LRU evict, account 127 MB/token transfers | speed, model-size | yes | M/L | 127 MB -> 9 MB per token (video) | port risk | **skip** — PR 24524 closed unmerged; `-n-cpu-moe` covers most |
| 25 | Multi-Token Prediction (`kpMLDIj6Ls8`) | model's own MTP head drafts 3 tokens; target verifies | speed | yes if the GGUF carries an MTP head | S | 1.65x dense 27B, 1.23x 35B-A3B (video); PR 22673: ~75 % acceptance, >2x | yes | **build** — top 5 fallback |
| 26 | llama-server + llama-swap + AnythingLLM (`oh50KFF8A_0`) | one OpenAI endpoint, model router, chat UI, Docker | speed, model-size, hallucination | yes | S | integration only | yes | **borrow** as packaging, not research |
| 27 | Colibrì mmap streaming + budget (`pIN-2oVJpyU`) | mmap + page faults, expert budget, hit-rate accounting; llama.cpp needs `-nr` | model-size, speed | yes for >30 GB models | S flags / L engine | 284B on 61 GB at ~1 tok/s; 69 s vs 142 s end-to-end (video) | yes | **borrow** `-nr`/budget; skip engine |
| 28 | Ternary Bonsai 27B (`rBLWDJrXCp0`) | true 1.75-bit trained weights, 5.95 GB GGUF | model-size, speed | yes (24 GB) | S download + fork | 27B in 5.95 GB (`prism-ml/Ternary-Bonsai-2-27B-gguf`); quality unproven | unknown | **skip** — different 27B, quality is the open question |
| 29 | Full-duplex voice micro-turns (`shJLycXjhVY`) | 600 ms slices, no VAD, listen/talk tags | speed, hallucination | yes | M | full-duplex on a 3060 (video) | yes | **skip** — voice pipeline, not text research |
| 30 | Pithagoras voice + MoE offload (`xbedfuqYQYA`) | 35B-A3B offload, 32K context, VAD/Whisper/TTS | speed, model-size | yes | M | overlaps #4 | yes | **skip** — same offload recipe plus voice work |

## Source-code pass (REA method: evidence / inference / unknown)

| system | source found | licence / stars | evidence read | inference for us | unknown |
|---|---|---|---|---|---|
| Colibrì (`pIN-2oVJpyU`) | `JustVugg/colibri` (description; the 9B had guessed a different owner) | Apache-2.0, 42,574 | `c/glm.c`: `COLI_MMAP`, `mmap(MAP_SHARED)`, `madvise(MADV_WILLNEED)`, one coalesced `pread` slab; README: 9.9 GB resident, 258 expert fetches/token, 71 % hit rate | the *engine* is not portable, but the diagnosis is: llama.cpp repacks weights into fresh RAM, so a >RAM model fails unless `-nr`/`--no-repack`; keep the budget/accounting idea | speed on a 3090 with a 30 GB model (not measured) |
| TurboQuant (`3mX_dc9ZNRA`, `Oq0UdRy1LcM`) | `AmesianX/TurboQuant` (llama.cpp port), `domvox/llama.cpp-turboquant-hip` | MIT, 92 (archived) | README: project discontinued; 330 tok/s prefill vs vLLM 1600, 13-19.5 tok/s decode on the same hardware | no llama.cpp path to recommend; the paper (arXiv 2504.19874) is real, the port is not | whether a newer port exists at another owner |
| DFlash / DFlash 2 (`9vY4-Z-tkHs`) | `z-lab/dflash`; `z-lab/Qwen3.8-27B-DFlash2-GGUF`; llama.cpp PR 27342 | MIT, 6,175; model Apache-2.0; PR **merged 2026-08-27** | PR files: `common/speculative.cpp`, `ggml-cuda/top-k.cu`, `conversion/qwen.py`; model card: Q4 draft 1.14 GB, `--spec-type draft-dflash --spec-draft-n-max 7` | this is the strongest speed lever; target 27B Q4 ~17 GB + 1.1 GB draft fits 24 GB | whether pin f3363593 is after 2026-08-27 (check before fetch) |
| MTP (`kpMLDIj6Ls8`) | llama.cpp PR 22673; `unsloth/Qwen3.6-27B-MTP-GGUF` | merged MIT; model Apache-2.0 | PR body: Qwen3.6 27B/35B-A3B, ~75 % acceptance, 3 drafts, >2x; MTP loads from the same GGUF and starts automatically | cheap fallback when a model ships an MTP head; prompt processing takes a hit | whether our Qwen3.8-27B GGUF carries the MTP tensors |
| Jev mode (`bcGO7xre46o`) | `thecodacus/llama.cpp` branch `parallel-decision`, `thecodacus/decision-playground`, HF `harshatheg/Qwen-2.5-1B-RLCD` (all in description) | MIT, 357; playground 26 | branch `parallel-decision` at `ad129b0`; video transcript: score only the allowed-token logits, reuse the schema KV, batch fields, per-field confidence | the *method* is sound and Laya already implements the same shape; port the scoring loop, do not adopt a 357-star llama.cpp fork | whether a multi-field schema beats Laya's one-question API in accuracy |
| Strata (`6WLBmP-tZ0Q`) | `Niko1221/Strata` | MIT, 26,804 | `START-HERE.bat` runs `winget`, downloads `python-3.12.10-amd64.exe` via PowerShell; `UPDATE.bat` downloads a zip | **vet failed** on install scripts that download binaries; read-only verdict, do not run | none needed |
| REAP (`0AqpaFm11oI`) | `CerebrasResearch/reap` | Apache-2.0, 529 | README: ICLR 2026, router-weighted expert pruning, 50 % compression, 1.9 % mean accuracy drop with normalized logits | good paper, but our 35B-A3B already fits; calibration cost > benefit | quality on a 28B-A3B pruned with our data |
| understory (`IwN-eK1s8og`) | `thecodacus/understory` (description) | Apache-2.0, 340 | README: OKF v0.1 markdown, frontmatter validated in code, `index.md`/`log.md` regenerated, paths sandboxed; MCP + web + CLI | borrow the **deterministic conformance layer**, not the Docker image; this is the memory idea | whether a markdown bundle beats `data/kb.sqlite` on recall at 30 videos |
| llm-wiki-okf (prior art for #9) | `hanupratap/llm-wiki-okf` | MIT, 12 | README: query-first discipline, immutable `raw/`, concept pages compiled from raw | confirms the query-first pattern; tiny, readable reference | none |
| Bonsai (`rBLWDJrXCp0`) | HF `prism-ml/Ternary-Bonsai-2-27B-gguf`; `jarkevithwlad/turboquant-prismml-cuda` | model card; fork MIT | GGUF sizes: PTQ1_0 5.95 GB, PQ2_0 7.21 GB, F16 53.8 GB | testable on 24 GB, but it is a different 27B with unproven quality | quality vs Qwen3.8-27B on our extraction task |
| Pithagoras (`xbedfuqYQYA`) | `thecodacus/pithagoras` | Apache-2.0, 281 | repo is a hosted web UI for the pi coding agent, not the voice agent the 9B described | extraction error; do not treat the JSON `system` as authoritative | none |
| Laguna S 2.1 (`NIxCj-3fFNk`) | `poolside.ai` model card (description) | model, no repo | 118B total / 8B active, 57.6 GB IQ4XS | does not fit 30 GB RAM | none |
| AnythingLLM / llama-swap (`oh50KFF8A_0`) | `mozilla/anything-llm`, `mostlygeek/llama-swap` | MIT / MIT | standard OpenAI-compatible stack | packaging, not a research lever | none |
| DiffusionGemma (`aYBgCl6ynGk`) | vLLM PR 57250; Unsloth GGUF | model / PR | block diffusion, whole-block denoise | not on our 27B/9B path | quality on extraction |
| Gemma 4 QAT (`xFEcAGB5kyg`) | Google blog; Unsloth | model | 4-bit-aware training | only helps Gemma 4 | none |

The remaining rows are either a llama.cpp configuration (the source is the
llama.cpp PRs already read), a product/policy story with nothing to reverse, or a
private build; the table names the reason in each case. No source-code question
was skipped silently.

REA's residual unknowns, stated rather than hidden: (a) the exact behaviour of
pin f3363593 vs PR 27342, (b) whether Qwen3.8-27B's GGUF carries MTP tensors,
(c) the real tok/s of any of these on this 3090. None was measured here; the
build sketches below say how to measure each in under an hour.

## Top 5 ranked by gain per effort for OUR setup

### 1. DFlash 2 draft for Qwen3.8-27B — ~3x decode, small effort

**Why first:** the only idea with a published, matching checkpoint (Qwen3.8-27B),
a merged llama.cpp PR, and a claimed **2.7-3.4x** throughput at the same output
(Inco AI, 2026-08-18; mean acceptance length 4.80 vs 4.28 for MTP). Draft Q4 is
**1.14 GB**, so the 27B Q4 (~17 GB) + draft + KV still fits 24 GB. Build sketch:
add `fetch/qwen38-dflash2.yaml` (`hf: z-lab/Qwen3.8-27B-DFlash2-GGUF`,
`include: ["*Q4_K_M.gguf"]`, 1.2 GB); confirm the `llama.cpp` pin contains
PR 27342 (`git log --grep`/`git merge-base --is-ancestor`), otherwise cherry-pick
it and rebuild; start `llama-server` with `-hfd z-lab/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M
--spec-type draft-dflash --spec-draft-n-max 7` from `bin/local_lane.py`; A/B 20
prompts for tok/s and **greedy output equality**; keep it only if equality holds
and the median gain is > 1.5x. Extends the **llama.cpp pin f3363593**;
`bin/local_lane.py` is the launcher.

### 2. MTP as the zero-download fallback — 1.2-2x decode

**Why second:** llama.cpp PR 22673 (merged 2026-05-16) starts MTP automatically
when the GGUF carries the head; the PR reports ~75 % acceptance with 3 drafts and
**>2x** on Qwen3.6 27B/35B-A3B, while the video measured 1.65x dense / 1.23x MoE.
No extra model download, no new dependency. Build sketch: check the 27B and
35B-A3B GGUFs for MTP tensors (`gguf-dump`/tensor name `mtp*`); if present, run
the same A/B as #1 and compare with DFlash 2; if absent, use the official
`unsloth/Qwen3.6-27B-MTP-GGUF` for the 35B-A3B path only. Extends the
**llama.cpp pin f3363593**; q2 schedules the A/B.

### 3. MoE expert-offload recipe for 35B-A3B — run the bigger model well

**Why third:** no code, only flags, and it turns a 6 GB card into a usable 35B
host (3 -> 10 tok/s, +230 %, video `8F_5pdcD3HY`) while the 3090 keeps attention
and KV. For us the win is headroom for 32K context, not raw speed. Build sketch:
sweep `--n-cpu-moe N` (N = 8, 16, 24, 32, 41) with `-ngl 99 --no-mmap -nr
--mlock` on 35B-A3B Q4_K_M; keep the largest N that leaves ~2 GB VRAM for KV;
add `-ot "blk\.\d+\.ffn_(up|down|gate)_exps=CPU"` for layers that still spill;
record tok/s at 4K/16K/32K and a 4K-token prefill. Reuse the video's
`--no-mmap` finding (no page faults during decode) and Colibrì's lesson that
llama.cpp needs `-nr` when the file is larger than RAM. Extends the
**llama.cpp pin f3363593** and the `a3b` wrapper; `bin/local_lane.py` picks the
profile.

### 4. Query-first OKF memory for the lane — the forgetting fix

**Why fourth:** it is the only idea aimed at `forgetting/memory`, it is cheap,
and every later session pays for its absence. understory's design is the one to
copy: markdown concepts with validated frontmatter, `index.md`/`log.md`
regenerated by code, every path sandboxed to the bundle, and a **query-first**
step before the model answers. Build sketch: add `memory/` (OKF bundle) to the
repo; add a stdlib `memory.py` that validates frontmatter (`type` required),
rebuilds `index.md` and appends `log.md` newest-first (the understory rule), and
exposes `memory_query`/`memory_add`; wire `bin/local_lane.py` to call
`memory_query` once per item before model selection, and to `memory_add` after a
check passes. Do **not** run understory's Docker image; borrow the deterministic
layer only. Prior art to reuse: `hanupratap/llm-wiki-okf` (MIT, query-first
index) and the existing `airesearch/kb.py` for the SQLite index. Extends
**bin/local_lane.py**; Laya can classify "answer from memory / ask the model".

### 5. Jev-style typed decisions in Laya — legal values, confidence, one pass

**Why fifth:** the video's mechanism is real and measured (300 ms vs 3.5 s,
~6x, on a 1.5B), and Laya already gives us a 60 ms CPU decision model; the
missing half is **multi-field** decisions with per-field confidence. Build
sketch: extend Laya's `/ask` to accept a schema of N independent fields, tokenise
each `field: value` line, score only the allowed value tokens (read the first
token where options diverge; read one more only when two options share a
prefix), and return `{field: {value, confidence}}`; batch all fields in one
llama.cpp pass; route low-confidence fields to the 27B. Do **not** adopt the
357-star `thecodacus/llama.cpp` fork; port the loop into the Laya server.
Caveat from the video: fields cannot see each other, so keep dependent fields
out. Extends **Laya** (`laya-mcp /ask`) and **bin/local_lane.py** for routing.

## Skipped, one line each

- **Context window:** the only two context videos are TurboQuant/CSA-HCA; the
  llama.cpp TurboQuant port is archived and reports 13-19.5 tok/s decode, and
  house R5 already measured Q8 KV at 31.9 % output identity. No safe build now.
- **REAP (#1), Bonsai (#28):** MoE pruning and a different ternary 27B; both
  need quality evaluation that costs more than the expected gain on a 24 GB card.
- **Strata (#3):** 64 GB RAM and install scripts that download a Python installer
  and binaries — the exact vetting failure named in the brief.
- **Laguna (#10), two-GPU (#6), RPC (#21), 128 GB kernel box (#23):** hardware we
  do not have; skipped in one line in the table.
- **Gemma 4 QAT (#18), DiffusionGemma (#17), Qwen4-EXP (#8):** only help models
  we do not run.
- **Voice (#16, #29, #30), AgentBox (#19), pi harness (#22), pricing (#15),
  Chrome legal (#13), ACL (#7):** real work, but not a model-speed/context/
  size/hallucination/memory lever for this pipeline.
- **Expert cache LRU (#24):** the upstream PR (24524) is closed unmerged; the
  `--n-cpu-moe` recipe in top 3 is the supported 80 % version.

## Exact next step

Run the top-1 A/B as a GPU job: one script that (a) asserts the llama.cpp pin
contains PR 27342, (b) starts the 27B with and without the DFlash 2 Q4 draft on
the same 20 prompts, (c) writes `tok/s`, acceptance length and a byte-for-byte
greedy-equality flag to `state/codacus/dflash2.json`. If the pin predates the
PR, the job must fail fast with that reason rather than rebuild online (the GPU
sandbox has no network).
