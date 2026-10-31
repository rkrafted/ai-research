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
