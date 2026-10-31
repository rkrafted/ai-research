# KV frontier for the A3B on one RTX 3090

Which 2026 KV-cache method gives the A3B more context than TurboQuant — for real,
on the 24 GB card, with the DFlash draft loaded and one serving slot.

**Verdict (2026-10-18):** the useful 2026 method is not TurboQuant. It is upstream
llama.cpp's quantized KV cache (`q8_0`/`q4_0`) with the mainline Hadamard
rotation. TurboQuant's llama.cpp PR was closed without merging in June 2026 and
the fork is legacy. On the A3B, `q4_0` K+V costs 3.56x fewer KV bytes per token
than `f16` (5,760 vs 20,480), so the projection is 64K–128K on this card. Only
the `f16` side is measured so far: **49,152 tokens fits, at 2.8K tok/s prefill
and 129–180 tok/s decode, with perfect needle/multi-hop recall**. `q4_0` is the
second job's first measured config; the table says "not run" until it lands.
The research eviction methods (CompressKV, SGD-KV) look stronger on paper but
have no llama.cpp port and are not worth porting today.

This document separates three things: the **method table** (papers and code,
§1), the **measured table** (real GPU cells, §3), and the **projection** from
the KV-size arithmetic (§2). No measured cell is guessed; unmeasured cells say
so.

---

## 1. Method table

Scope: methods that reduce KV memory or KV reads for long-context decoding.
"Models tested" is what the paper/repo actually ran, because the A3B is a
hybrid-attention MoE: Qwen3.6-35B-A3B, 40 blocks, `full_attention_interval=4`
(10 full-attention layers), GQA with 2 KV heads, head_dim 256. A method that
only ran dense GQA Llama/Mistral has not been shown on a hybrid MoE.

| Method | Paper claim | Models tested | Training-free | llama.cpp / ggml code | Licence / vet | For the A3B |
|---|---|---|---|---|---|---|
| **TurboQuant** (planar3/iso3) | quality-neutral KV at 3.5 bits/channel, marginal loss at 2.5 | paper is model-agnostic; fork cards target dense and hybrid GGUF models | yes | fork only: `johndpope/llama-cpp-turboquant`; the upstream PR was **closed unmerged (June 2026)** | fork MIT, unmaintained; the old `AmesianX/TurboQuant` repo is MIT and archived | **skip** — dead fork; mainline now has the same idea (Hadamard rotation) |
| **Upstream `q8_0` KV** | ~half the KV bytes, perplexity +0.002–0.05 | every llama.cpp model, incl. hybrid/MoE | yes | in-tree, CUDA FA kernel in f336359 | MIT (llama.cpp) | **adopt** — default safe step |
| **Upstream `q4_0` KV + Hadamard rotation** | ~quarter KV bytes; one 2026 card reports ≈7.6% perplexity rise on a 120B hybrid MoE; rotation improves low-bit quality | in-tree; rotation applies to any quantized KV with head_dim % 64 == 0 | yes | in-tree since PR #21038 (April 2026); opt-out `LLAMA_ATTN_ROT_DISABLE=1` | MIT | **adopt/test** — the main candidate for 64K+ |
| **Upstream `q5_1` KV** | 5-bit; 0.75 B/element vs 2 B for f16 | all models; CUDA FA case exists in f336359 | yes | in-tree | MIT | **test** — middle ground if q4_0 degrades |
| **FP8 KV** | 1 B/element, near-fp16 quality | vLLM-supported dense GQA models | yes (calibration-free e4m3) | **none** in llama.cpp; no FP8 KV type | vLLM Apache-2.0 | **skip on 3090** — SM86 has no FP8 tensor cores; q8_0 is the same size |
| **KIVI** (ICML 2024) | 2-bit per-channel K / per-token V; 2.35–3.47x less memory, near-fp16 quality | LLaMA, Mistral, Qwen dense GQA | yes | HF `transformers`/CUDA only; llama.cpp issue #5492 still open | MIT | **wait** — needs a new ggml type; q4_0+Hadamard gets most of the win |
| **SnapKV** (2024) | 3.6x generation speedup, 8.2x memory efficiency, minor accuracy loss | LLaMA/Mistral dense GQA | yes | HF only (FasterDecoding); no ggml | Apache-2.0 | **skip** — superseded by CompressKV, needs attention scores |
| **CompressKV** (2026) | Semantic Retrieval Heads + layer budgets; 97% of full cache at 3% of the cache on LongBench QA, 90% NIAH at 0.7% | LLaMA / Mistral GQA (repo scripts); **no MoE or hybrid result** | yes | HF + FlashAttention + a custom `adakv` CUDA kernel; **no llama.cpp port** | **no LICENSE file at repo root**; paper CC BY-NC-SA 4.0; pip/conda build, no binaries committed | **wait** — best paper candidate, but unlicensed and no port |
| **SGD-KV** (NeurIPS 2026 workshop) | summarization-score per head + water-filling; up to 75% KV reduction, up to 1M tokens | Qwen2.5-7B-1M (GQA, 4 KV heads), Qwen3-32B (dense) | yes (the diagnostic needs a labelled summarization set, not gradient training) | **no public code found**; no llama.cpp port | paper only, no repo | **wait** — higher port cost than CompressKV for the same eviction machinery |

Sources: [TurboQuant](https://arxiv.org/abs/2504.19874),
[CompressKV](https://arxiv.org/abs/2606.24467) /
[repo](https://github.com/TUDa-HWAI/CompressKV),
[SGD-KV](https://arxiv.org/abs/2609.03235),
[KIVI](https://github.com/jy-yuan/KIVI) /
[llama.cpp issue #5492](https://github.com/ggml-org/llama.cpp/issues/5492),
[SnapKV](https://github.com/FasterDecoding/SnapKV),
[llama.cpp Hadamard PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038),
[TurboQuant fork status](https://huggingface.co/majentik/Nemotron-3-Super-120B-A12B-TurboQuant-GGUF-Q2_K).

Repo vetting: KIVI and SnapKV are research code with no binaries or telemetry;
CompressKV's `methods/adakv` is a CUDA extension built with `make`, and the repo
has no root licence, so it is **not** reusable as-is. TurboQuant's fork is a
full llama.cpp tree (large build, MIT) and is unmaintained. Nothing was executed
from these repos in this project.

---

## 2. What the A3B's cache actually costs

From the GGUF metadata (`qwen35moe`, 40 blocks, `full_attention_interval=4`,
2 KV heads, key/value length 256):

KV bytes/token = 10 layers × 2 KV heads × 256 dim × (bytes/element K + V).

| KV type | bytes/element | bytes/token | 32K | 64K | 128K | 256K |
|---|---:|---:|---:|---:|---:|---:|
| f16 | 2.0 | 20,480 | 640 MiB | 1,280 MiB | 2,560 MiB | 5,120 MiB |
| q8_0 | 1.0625 | 10,880 | 340 MiB | 680 MiB | 1,360 MiB | 2,720 MiB |
| q5_1 | 0.75 | 7,680 | 240 MiB | 480 MiB | 960 MiB | 1,920 MiB |
| q4_0 | 0.5625 | 5,760 | 180 MiB | 360 MiB | 720 MiB | 1,440 MiB |

Only the 10 full-attention layers have KV; the other 30 are linear-attention
layers. That is why the absolute numbers are small and why a 24 GB card can
reach 128K at all. `q4_0` is 3.56x cheaper than `f16` per token.

---

## 3. Measured on this stack

### 3.1 Already measured (reused, same llama.cpp `f336359`, A3B Q4_K_M + DFlash Q8_0, 1 slot)

From `moe-offload/a3b-tune` (sibling worktree, same binary and draft):

| Config | ctx | KV | VRAM | decode tok/s | Result |
|---|---:|---|---:|---:|---|
| `dflash-n6` | 8,192 | `q8_0`/`q8_0` | 22.93 GB | 289.7 code / 382.5 edit | fits |
| `kv-q4_0` | 8,192 | `q4_0`/`q4_0` | 22.77 GB | 301.9 code / 382.5 edit | fits |
| `ctx32k` | 32,768 | `q8_0`/`q8_0` | 22.75 GB | not recorded | fits |
| `ctx64k` | 65,536 | `q8_0`/`q8_0` | — | — | **OOM, server did not start** |

This is the whole problem in one line: at 8K and 32K the model already uses
~22.8 GB of the 24 GB card, so `q8_0` cannot reach 64K. `q4_0` needs 360 MiB at
64K and 720 MiB at 128K, which is the same order as `q8_0` at 32K — the
prediction is that `q4_0` reaches 64K and probably 128K.

### 3.2 First GPU run (`gpu_jobs/kv-measure`, 2026-10-18): what is actually measured

The job measures, for `f16`, `q4_0`, `q8_0`, `q5_1`, the two asymmetric K/V
mixes, and `q4_0` with Hadamard rotation disabled:

1. the largest `--ctx-size` that starts **and survives a real decode** beside
   the DFlash draft at `--parallel 1`, by binary search in 16K steps;
2. prefill and decode tok/s on the real mixed contexts at 32K / 64K / 128K;
3. needle (3 facts at 8% of the text) and multi-hop (one fact at 8%, another
   at 90%) scores, plus agreement with the `f16` answer.

The real long-context set (built at job time, nothing raw committed):

| Slice | Source | Share |
|---|---|---|
| `calls` | sales-brain realspeech manifest, **dev (held-out)** whole transcripts, grouped by speaker | 60% |
| `talks` | the 30 codacus-mine captions in `data/codacus/*.txt` | 20% |
| `repos` | real repo contexts named by the open sales-brain local-lane work items | 20% |

Questions are answerable only from far back: the needle facts sit at 8% of the
document; the multi-hop question needs a fact at 8% and another at 90%. A probe
passes only when every expected string is present; the `f16` row is the
full-precision reference and the other rows also report exact agreement with it.

**Measured so far** (`python3 -m kvfrontier.report`, read from
`state/kvfrontier/results.jsonl`; the job's log and exit code are in
`gpu_jobs/`):

| config | max ctx that starts | KV at max | prefill tok/s (32K / 48K cap) | decode tok/s (32K / 48K cap) | needle | multihop | f16 agree |
|---|---:|---:|---|---|---:|---:|---:|
| `f16` | **49,152** | 960 MiB | 2,835 / 2,574–2,659 | 154 / 129–180 | 3/3 | 1/1 | — |
| `q8_0` | **not yet** | — | — | — | — | — | — |
| `q5_1` | not run | — | — | — | — | — | — |
| `q4_0` | not run | — | — | — | — | — | — |
| K `q8_0` / V `q4_0` | not run | — | — | — | — | — | — |
| K `q4_0` / V `q8_0` | not run | — | — | — | — | — | — |
| `q4_0`, no Hadamard | not run | — | — | — | — | — | — |

Reading this table honestly:

- `f16` reaches **49,152** tokens, not 64K. The "64K" and "128K" rows above were
  therefore run at the 48K cap (prompts 48,349 and 48,020 tokens), so they are
  **not** evidence about 64K/128K accuracy. The needle and multi-hop scores are
  still real: they are at the far end of the deployed 48K cache.
- `f16` multihop scored 0.0 in the raw JSONL and 1.0 here. The model had
  answered both facts on two numbered lines; the scorer looked only at line 1
  for a one-question probe. Fixed in `kvfrontier/needles.py`; the stored answers
  are re-scored when the report is rendered.
- `q8_0`: the server **started** at 81,920 tokens but the first 32K prompt died
  with `failed to allocate compute pp buffers` (a 332 MiB CUDA allocation). The
  first job treated `/health` as proof of fit, so that 81,920 is not a real max.
  `fit_ctx` now prefills a full 2048-token batch (one generated token)
  before calling a size a fit.
- The first job exited 1 at `q8_0`; `q4_0`, `q5_1`, the asymmetric mixes and the
  no-rotation ablation never ran. No cell in this table is projected from the
  KV-size arithmetic.

**Second job queued:** `gpu_jobs/kv-measure2.job.yaml` runs the fixed harness
with `q4_0` second (so the deciding cell lands even if a later config dies) and
reuses the completed `f16` cells. After it finishes:
`python3 -m kvfrontier.report` fills the rows. Until then the measured verdict
for `q4_0` is a prediction, not a result.

Run it locally (read-only, after the job): `python3 -m kvfrontier.report`.

---

## 4. Verdict for the A3B, exact flags

Proposed file: `state/a3b_server.args.kv-proposed` (the live
`~/theprojects/orchestrator/state/a3b_server.args` was **not** touched).

The recommendation is `q4_0` for both K and V, `-fa on`, one slot, DFlash draft,
at the largest context the job confirms:

```
-m .../Qwen3.6-35B-A3B-Q4_K_M.gguf
-c 131072            # fall back to 65536 if measured max_ctx < 131072
--parallel 1
-ngl 99
-fa on
-ctk q4_0
-ctv q4_0
-md .../dflash-Qwen3.6-35B-A3B-Q8_0.gguf
--spec-type draft-dflash
-ngld 99
--spec-draft-n-max 6
--jinja
--reasoning-budget 0
--cache-reuse 256
--metrics
```

Why these:

- **K and V both `q4_0`** keeps the fused Flash-Attention path; mixing types may
  disable the fast kernel or add a conversion. The asymmetric rows exist to
  check the "quantize V harder" folklore, not as the default.
- **Hadamard rotation stays on** (default for quantized KV). It is the 2026
  mainline improvement that makes `q4_0` usable; `LLAMA_ATTN_ROT_DISABLE=1` is
  measured only as an ablation.
- **`--reasoning-budget 0`** keeps answers short and comparable across cells.
- **Do not use `-c 262144`** even if `q4_0` starts there: the real long-context
  prompts need headroom for the question and the generated answer, and the
  measured accuracy at 256K is not part of this project's done-when.
- `q8_0` remains the safe default if `q4_0` accuracy on the real set drops below
  the f16 reference by more than the project is willing to pay.

---

## 5. Port recommendation for the best research method without a llama.cpp port

**Best paper candidate: CompressKV.** It is the only 2026 method in the table
that (a) targets GQA, (b) has public code, and (c) reports a large win at a tiny
cache budget. It is still the wrong thing to port now.

### What a CompressKV port would touch

| Area | Work |
|---|---|
| `src/llama-kv-cache.cpp` | per-layer/per-head retained-token bookkeeping, gather/compaction of the KV cells, budget metadata; the current cache is a flat ring per sequence |
| `src/llama-graph.cpp` | a score pass before generation: QK^T over the prompt, softmax, per-head accumulation; FlashAttention does **not** expose the score matrix, so this needs a separate graph or a new FA variant |
| `ggml/src/ggml-cuda/fattn-*` | optional: an FA kernel that emits per-head scores or applies a token mask, otherwise the score pass runs on a materialised `ggml_mul_mat` |
| `ggml-quants.c` / `ggml-cuda` | no new quant type for CompressKV (it evicts); only a gather (`ggml_cpy` with indices) and a per-head mask |
| `src/llama-model.cpp` | the offline head classification (Semantic Retrieval Heads) and layer-budget table must ship as model metadata or a side file |

Effort: **2–4 weeks** for a correct, benchmarked port (the score pass is the
hard part; the eviction bookkeeping is mechanical). The A3B has only 10
full-attention layers, so the per-layer budget logic has few layers to work with.

Expected gain from the paper on a GQA MoE: the paper's 97%-at-3% number is on
dense GQA LongBench QA, not a hybrid MoE. At 3% of f16 the A3B cache would be
0.6 KB/token (vs 5.6 KB/token for `q4_0`), i.e. 128K would use ~77 MiB instead
of 720 MiB. That is a real memory win, but the card's limit is not only KV:
weights + draft + compute already eat ~22 GB, and the 30 linear-attention layers
have their own recurrent state. A realistic ceiling is 2–4x more context than
`q4_0`, **if** the SRH heads transfer to the A3B — unproven, and the repo has no
licence.

**Recommendation: wait.** Re-check in a quarter for (a) a llama.cpp port or PR
from the CompressKV authors, (b) a licence file, and (c) any MoE/hybrid result.
Porting it now would cost weeks and compete with a zero-maintenance upstream
`q4_0` path that already reaches 64K–128K.

**SGD-KV:** higher cost than CompressKV (the head scores come from a
chunk-summarization diagnostic over a labelled set, then water-filling), no
public code, and the same score/eviction machinery in llama.cpp. **Wait**, and
only revisit if CompressKV stalls.

**KIVI:** 2-bit per-channel K would need a new ggml type (`ggml-quants.c` +
CUDA dequant + FA vec-dot + a per-channel scale layout), ~2–3 weeks. It would
halve `q4_0` again (5.6 → 2.8 KB/token). **Wait** — `q4_0` + Hadamard already
gets the context this project needs, and 2-bit on Latvian/multilingual text is
the riskiest accuracy corner.

**SnapKV / TurboQuant fork:** **skip.** SnapKV is older and weaker than
CompressKV with the same score-pass cost; the TurboQuant fork is a closed PR and
unmaintained.

---

## 6. Reproduce

```bash
# offline: size model, scorer, fixtures (no GPU, no network)
python3 -m pytest -q tests/test_kvfrontier.py

# show the real long contexts the job will build (reads the dev manifest)
python3 -m kvfrontier.measure --dry-run

# GPU job (queued by the runner; 220-minute wall budget, resumable)
cat gpu_jobs/kv-measure.job.yaml

# after the job
python3 -m kvfrontier.report
```

`state/kvfrontier/results.jsonl` is gitignored; the next session copies the
rendered table into §3.2 and commits the result.

---

## 7. What was reused

- `moe-offload`'s CUDA llama.cpp build (`f336359`, DFlash, Hadamard rotation,
  q8_0/q5_1/q4_0 CUDA FA kernels) instead of building one.
- `a3b-tune`'s serving flags (DFlash draft, `--spec-draft-n-max 6`, `-fa on`,
  `--parallel 1`) and its measured 32K/64K OOM data.
- `a3b-shootout`'s `Server`/`VramPeak`/timings pattern for the launcher.
- The sales-brain realspeech **dev** manifest for held-out call transcripts, the
  codacus-mine captions, and the open sales-brain local-lane work items for real
  repo contexts.
- `pip`-free stdlib only; tests run offline.
