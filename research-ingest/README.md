# AI research pipeline — R0 looped-LM deep-dive · R1 channel registry · R2 video ingest

This repo hosts the AI-research pipeline work ("most intelligence per GPU"): **R0**, a
looped/recurrent-depth language-model deep-dive with a real RTX 3090 test, **R1**, a
channel registry, and **R2**, the caption/metadata ingest and knowledge base built on
the R1 kept list. Each project below is self-contained.

# research-ingest — R2 video ingest + knowledge base

Part of the **AI research pipeline** (`docs/PLAN.md` §5 steps 1–3). From the 35 **kept**
channels in `data/channels.json` (built by R1), it collects, for every video of the last
24 months:

1. **metadata** — a yt-dlp flat playlist per channel (id, title, date, duration, views),
   then the per-video info json (full description + links). No video is downloaded.
2. **captions** — English manual/auto VTT, captions only, into `data/private/captions/`
   (gitignored).
3. **extraction** — one prompt per video written by DeepSeek, served by the runner to
   the local `qwen3.8-27b-uncensored:latest` through `local_batch/ingest-extract.yaml`;
   the batch writes `data/kb-extracts.jsonl` — the only extraction file. Per video it
   asks for methods named, claims with the numbers stated, arXiv ids and repos cited,
   hardware/compute mentioned, and the audience.
4. **knowledge base** — `data/kb.sqlite`, built by `airesearch/kb.py` from the batch
   output only: tables `videos`, `claims`, `papers`, `methods`, `links`; every claim
   keeps its video id and a `[MM:SS]` timestamp when the captions give one.

`docs/spot-checks.md` records a ≥10% manual check of the model's answers by video id.

## Install and run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # yt-dlp + pytest

# 1. enumerate every video of the kept channels in the last 24 months (network, resumable)
.venv/bin/python -m airesearch.ingest enumerate --months 24 --workers 6
# 2. per-video description/links + English captions (network, resumable)
.venv/bin/python -m airesearch.ingest captions --workers 10
# 3. merge metadata + captions into the private corpus (offline)
python3 -m airesearch.ingest corpus
# 4. write the local-batch prompts and queue the extraction (offline)
python3 -m airesearch.prompts --cap 4000 --queue
# 5. after the runner serves the batch: build and check the knowledge base (offline)
python3 -m airesearch.kb build
python3 -m airesearch.kb stats
# 6. audit a 10 % sample of the answers and rewrite docs/spot-checks.md (offline)
python3 tools/spotcheck.py

python3 -m pytest -q                        # offline tests
```

## Layout

```
airesearch/ingest.py             yt-dlp enumerate + per-video info/captions + corpus
airesearch/prompts.py            corpus -> local_batch/ingest-prompts.jsonl (+ yaml)
airesearch/kb.py                 data/kb-extracts.jsonl -> data/kb.sqlite
tools/spotcheck.py               grounds a 10 % sample -> docs/spot-checks.md (read-only)
local_batch/ingest-extract.yaml  the runner-served extraction request
data/kb-extracts.jsonl           batch output (provenance-checked; export-ignored)
data/kb.sqlite                   knowledge base (tracked for done_when; export-ignored)
data/private/                    raw metadata + captions cache (gitignored)
docs/spot-checks.md              DeepSeek's spot-check of the model's output
```

`data/kb-extracts.jsonl` counts only because the runner's Ollama batch wrote it
(`provenance.py`); a session never creates or edits it. `data/kb.sqlite` is derived from
that file and is marked `export-ignore`, so it travels into the merged tree but stays out
of published exports.

## Example output

Real output of step 1 (2026-10-02; `data/private/` cache is gitignored):

```
enumerating 35 kept channels, months=24, start=2024-10-01
  OK   neuro-dump                          1 videos
  OK   Machine Learning Street Talk      160 videos
  OK   IBM Technology                    817 videos
  OK   AI Engineer                      1200 videos
  OK   Wes Roth                          722 videos
  ...
total videos in window: 7077
```

Status: steps 1–4 done for 3 246 videos (metadata + captions + corpus + prompts); the
runner served `local_batch/ingest-extract.yaml` to the local 27B and wrote
`data/kb-extracts.jsonl` (provenance-checked, sha256 in the orchestrator ledger). Step 5
(`data/kb.sqlite`) and `docs/spot-checks.md` are complete; `DONE.md` records the run.
Real output of steps 3–5:

```
corpus: 7077 videos, 3242 with description, 3198 with captions -> data/private/ingest/corpus.jsonl
prompts: 3246 (of 3246 eligible), 3198 with transcript -> local_batch/ingest-prompts.jsonl
queued: local_batch/ingest-extract.yaml -> data/kb-extracts.jsonl
built videos=3246 claims=11216 papers=632 methods=18610 links=3024
videos=3246 claims=11216 claim_videos=3224 papers=632 arxiv_ids=488 methods=18610 links=3024
spot-checks: 325/3246 videos -> docs/spot-checks.md
```

`tools/spotcheck.py` grounds a fixed 10 % sample of the batch answers against the exact
prompts stored in `local_batch/ingest-prompts.jsonl`; `docs/spot-checks.md` adds a manual
read of twenty videos and lists the pipeline limitations the audit exposed.

Reused from prior art: R2 does not rebuild ingestion — it extends the R1 `airesearch`
package (the `data/channels.json` kept list, the `data/private/` yt-dlp cache layout) and
reuses the R0 `export-ignore` mechanism so `data/kb.sqlite` and `data/kb-extracts.jsonl`
survive the merge yet stay out of published exports.

# research-looped-lm — R0 deep-dive: looped language models (Ouro)

What this is: a research deep-dive into **looped / recurrent-depth language models** — the
architecture behind ByteDance's **Ouro** checkpoints — with a real test on our RTX 3090. It
answers the project's one question: *what gives the most capability per unit of compute for
someone with one 24 GB GPU?*

Read the two summaries first:

- [`docs/looped-lm.md`](docs/looped-lm.md) — faithful summary of *Scaling Latent Reasoning via
  Looped Language Models* ([arXiv 2510.25741](https://arxiv.org/abs/2510.25741)): method
  (weight-tied loops, entropy-regularized exit gate, two-stage gate training), every results
  table, the knowledge-capacity-vs-manipulation evidence, the honest Appendix D counter-result,
  and what it means for a 24 GB card (capability/parameter, test-time compute, VRAM, latency).
- [`docs/neuro-dump-video.md`](docs/neuro-dump-video.md) — our own timestamped summary of the
  NeuroDump video (<https://www.youtube.com/watch?v=pDsTcrRVNc0>, captions fetched with
  `yt-dlp` into gitignored `data/private/`), plus a claim/evidence/code/compute card for every
  paper linked from its description and captions.
- [`docs/next-experiments.md`](docs/next-experiments.md) — three experiment cards (Distill
  extraction, adaptive-exit agent loops, inference-time layer repetition) sized to the 3090.

## Layout

```
docs/                  paper + video summaries, next experiments
benchmark/             generator output only (loop-eval-v1.jsonl is gitignored, regenerated on demand)
tools/make_benchmark.py   deterministic generator of the 200 authored 4-way MCQ items (100 arithmetic, 100 knowledge); --check verifies determinism
tools/looped_eval.py      the 3090 evaluation; the only writer of state/looped-eval.json
fetch/*.yaml              HF checkpoints the runner downloads before the GPU job
gpu_jobs/looped-eval.job.yaml   queues tools/looped_eval.py on the RTX 3090
tests/                 offline pytest suite
```

## Install and run (offline)

```bash
python3 -m pip install -r requirements.txt   # optional; tests use only pytest
python3 -m pytest -q                         # 5 tests, no network, no GPU
python3 tools/make_benchmark.py --check      # determinism check (also verifies the file if present)
```

Real output of the check and the tests:

```
$ python3 tools/make_benchmark.py --check
OK benchmark/loop-eval-v1.jsonl (200 items, generated on demand)

$ python3 -m pytest -q
.....                                                                    [100%]
```

Both exit 0 (the `-q` in `pytest.ini` suppresses pytest's summary line). The generated `.jsonl` is
not committed: `--check` verifies determinism (and the file if present), while the tests and
`tools/looped_eval.py` rebuild the items in memory when the file is absent. Output paths are
printed repo-relative, so no committed transcript contains an absolute home path.

## The 3090 evaluation (ran; result in `state/looped-eval.json`)

`gpu_jobs/looped-eval.job.yaml` ran on the RTX 3090 after the runner downloaded `fetch/*.yaml`.
`tools/looped_eval.py` scored all 200 `benchmark/loop-eval-v1.jsonl` items by length-normalised
log-likelihood at `total_ut_steps` 1/2/4/8 for `ByteDance/Ouro-2.6B`, 1/4 for `Ouro-1.4B` and
`Ouro-2.6B-Thinking`, plus `Qwen/Qwen3-1.7B` as the non-looped baseline, and wrote
`state/looped-eval.json` (n=200, 9 runs). Accuracy / arithmetic-subset accuracy / throughput:

| model | loops | acc | arith | tok/s | VRAM |
|---|---:|---:|---:|---:|---:|
| Ouro-2.6B | 1 | 0.695 | 0.39 | 5868 | 5.53 GB |
| Ouro-2.6B | 2 | 0.830 | 0.66 | 3396 | 5.53 GB |
| Ouro-2.6B | 4 | 0.915 | 0.83 | 1698 | 5.53 GB |
| Ouro-2.6B | 8 | 0.920 | 0.84 | 844 | 5.53 GB |
| Ouro-1.4B | 1 | 0.675 | 0.41 | 12313 | 3.06 GB |
| Ouro-1.4B | 4 | 0.835 | 0.67 | 3286 | 3.06 GB |
| Ouro-2.6B-Thinking | 1 | 0.685 | 0.39 | 6410 | 5.53 GB |
| Ouro-2.6B-Thinking | 4 | 0.890 | 0.78 | 1661 | 5.53 GB |
| Qwen3-1.7B (non-looped) | 1 | **0.800** | 0.62 | 10243 | 3.98 GB |

Takeaways: extra loops buy a lot on multi-step arithmetic (2.6B base: 0.39 → 0.84 accuracy,
T=1 → T=8) and are nearly free in VRAM; throughput falls ~linearly with loops (5868 → 844 tok/s);
returns are strongly diminishing past T=4 (+0.005 accuracy for 2× compute). The looped 1.4B at
T=4 (0.835) beats the same-size non-looped Qwen3-1.7B (0.800), i.e. local evidence for the
paper's test-time-compute-via-loops claim. Our knowledge half is saturated (0.94–1.0 for every
model) and measures very little; it is the arithmetic half that separates models.

`state/looped-eval.json` may only be written by that GPU job (provenance ledger); nothing in a
CPU session touches it.

The checkpoints ship custom `modeling_ouro.py` written for `transformers==4.54`, while the 3090
queue's shared venv is `transformers==5.17`. We verified on CPU (random-weight tiny config) that
three in-process shims make it run, and `tools/looped_eval.py` applies them automatically:

```
MODEL OK 26479105
FORWARD OK (1, 4, 49152)
GENERATE OK [[1, 2, 3, 4, 30624, 30624, 15455, 27974]]
LOOPS4 OK (1, 4, 49152)
```

## What was reused / prior art

The runner's `PRIOR_ART.md` is for the *channel registry* idea (R1) and reports prior art for
that, not for looped LMs; nothing in it applies to R0, so no code was forked. What we did reuse is
the repo's existing plan (`docs/PLAN.md` §3) and the queue's job conventions (`fetch/*.yaml`,
`gpu_jobs/*.job.yaml`). The arXiv API and Hugging Face model API were used to identify the paper,
the four Ouro checkpoints and the size-matched baseline. No existing Ouro evaluation script was
found in the queue, so `tools/looped_eval.py` and the benchmark generator are new.

## Honesty notes

- The paper does not report GPU model, count, hours or FLOPs anywhere we could find; all
  reproduction-cost estimates are ours and labelled as such.
- Appendix D.1 of the paper is a counter-result: at matched small scale, standard non-looped
  models beat LoopLM and the gap grows with recurrent depth. That is a *training-compute-matched*
  comparison, not an inference-time one; our local sweep of a trained checkpoint at T=1…8 shows
  the opposite direction (more loops → better), which is the paper's own test-time claim.
- Captions and raw datasets are not committed (`data/private/` is gitignored); the benchmark is
  authored by us, not downloaded.

---

# research-channels — R1 channel registry

Part of the **AI research pipeline** ("most intelligence per GPU", `docs/PLAN.md`).
This repo answers one narrow question for the rest of the pipeline:

> Which YouTube channels reliably cover AI models, training tricks, tools and the
> theory behind them in a way that is useful to someone with **one 24 GB GPU**?

It builds a registry of channels with a reproducible score, so the later ingest
stage (R2) knows what to download captions from.

## What it does

1. **Discover** — seeds (`docs/PLAN.md` §4.1), the hand list (§4.2) and an
   expansion list (§4.3) live in `airesearch/sources.py`.
2. **Sample** — for each channel, the latest 20 videos are fetched with
   `yt-dlp` (flat metadata only; **no videos are downloaded**). Raw metadata is
   cached in `data/private/` (gitignored).
3. **Extract** — deterministic fields from title + description: method names
   (LoRA, QLoRA, distillation, MoE, speculative decoding, …), numeric claims,
   linked papers (arXiv / Papers with Code / HF papers / DOI), compute terms
   (GPU, VRAM, 24 GB, consumer hardware, …), evidence terms and caption
   availability. No LLM is involved in the committed score.
4. **Score** — `airesearch/score.py` implements the §2 lens as a fixed weighted
   sum (methods 25, numbers 20, papers 20, compute 15, evidence 10, captions 10).
   A channel is **kept** when its sample scores at least 70% of the median seed
   score ("scores like the seeds").
5. **Write** — `data/channels.json` (machine) and `docs/channels.md` (ranked kept
   list, one line per channel on why it helps a limited-compute builder).

## Install

Python 3 standard library is enough to build the registry and run the tests.
`yt-dlp` is only needed to *fetch* new channels:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
# 1. fetch latest-20 metadata per channel into data/private/ (network + yt-dlp)
.venv/bin/python -m airesearch.fetch --limit 20 --workers 6

#    optional: also write English VTT captions to data/private/captions/
#    (slower; no media is downloaded) -- intended for the R2 ingest pass
.venv/bin/python -m airesearch.fetch --limit 20 --workers 6 --only seed --captions

# 2. build the committed registry (offline, uses the cache)
python3 -m airesearch.registry

# 3. topic-sweep discovery for PLAN 4.3 expansion (network, writes only private data)
.venv/bin/python -m airesearch.discover --per-term 20

# 4. tests (offline, synthetic fixtures only)
python3 -m pytest -q
```

## Layout

```
airesearch/sources.py    seed / candidate / expansion channel lists
airesearch/fetch.py      yt-dlp channel sampler + private cache (+ --captions)
airesearch/discover.py   PLAN §4.3 topic sweeps -> data/private/discovery.json
airesearch/extract.py    deterministic field extraction + lexicons
airesearch/score.py      PLAN §2 lens as a pure function
airesearch/registry.py   builds data/channels.json + docs/channels.md
docs/PLAN.md             the project plan (source of the lens and channel lists)
docs/channels.md         generated kept-channel list
data/channels.json       generated registry (tracked; excluded from exports)
data/private/            raw metadata + captions cache (gitignored)
```

`data/channels.json` is a generated machine-readable dump: it stays tracked so
the offline pipeline works from a checkout, but `.gitattributes` marks it
`export-ignore`, so `git archive` and published tarballs leave it out.

## Prior art

The runner's `PRIOR_ART.md` search found no reusable channel-registry code
(the hits were unrelated high-star repos with weak textual overlap). This repo
builds on the prior art that does fit: `yt-dlp` for YouTube metadata and the
lens/source lists already written in `docs/PLAN.md`.

## Example output

Real output of `python3 -m airesearch.registry` after fetching 85 channels:

```
channels=62 seeds=6 kept=35 threshold=15.97 skipped=23
```

Top of the generated `docs/channels.md`:

```
- **neuro-dump** — score 56.6, 1 video (seed) — https://www.youtube.com/channel/UCMHBJH5rX2dF1S7o-fJuY-A — Latest videos cover latent, looped, reasoning; frequent compute angle: parameters.
- **Machine Learning Street Talk** — score 55.2, 20 videos (candidate) — https://www.youtube.com/@MachineLearningStreetTalk — Latest videos cover agents, diffusion, reasoning; frequent compute angle: cost, edge, efficiency.
- **Yannic Kilcher** — score 47.7, 20 videos (candidate) — https://www.youtube.com/@YannicKilcher — Latest videos cover transformer, reasoning, attention; frequent compute angle: parameters, efficiency, cost.
```

One entry of `data/channels.json`:

```json
{
 "name": "neuro-dump",
 "url": "https://www.youtube.com/channel/UCMHBJH5rX2dF1S7o-fJuY-A",
 "videos_sampled": 0,
 "videos_available": 1,
 "score": 56.62,
 "kept": true,
 "reason": "kept: score 56.6 vs seed bar 16.0; ~5.0 methods, ~0.0 numbers, ~6.00 paper links/video, captions 100%.",
 "seed": true,
 "provenance": "seed",
 "top_methods": ["latent", "looped", "reasoning", "reinforcement learning", "scaling laws"]
}
```

Topic-sweep discovery (`python3 -m airesearch.discover`) found **201 unique channels**,
**178 not already in the source lists** — those are the R2 expansion candidates.

Note: `videos_sampled` is `0` only for the seed `neuro-dump`, which has a single
listed video; `videos_available` records that. It is retained because the owner named
it as the looped-LM source.
