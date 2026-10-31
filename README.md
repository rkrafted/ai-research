# ai-research

A collection of small local-LLM research projects: experiments, ingest pipelines, notes and evaluation harnesses
(long-context / KV cache, looped LMs, LoRA micro-turns, memory, small-vs-big model comparisons, etc.).

> **Not independently verified.** These are working research folders produced on a single dev machine.
> Results were not peer-reviewed or reproduced elsewhere, numbers can be stale or specific to one setup,
> and nothing was tested on real-world deployments. Provided **as is, with no warranty**.

Datasets, caches, model weights, transcripts of third-party videos and vendored third-party code are
**not** included. Each folder has its own `README.md`.

| Folder | What |
|---|---|
| `a3b-everywhere/` | Research report (2026-10-17) on one question: can **Qwen3.6-35B-A3B Q4_K_M + DFlash** |
| `a3b-long-run/` | A measured answer to one real miss: a session proposed a new 30-min check-in watchdog |
| `bycloud-ingest/` | Download bycloud's full public video catalogue (newest first), turn each video's |
| `caveman-terse/` | Survey of ways to give a compute-limited agent better memory and a better context window |
| `codacus-mine/` | Turn the last 30 uploads of the Codacus channel into a table of mechanisms we can |
| `kv-frontier/` | Research + a runnable measurement harness for one question: on a 24 GB RTX 3090, |
| `memory-research/` | Survey of ways to give a compute-limited agent better memory and a better context window |
| `microturn-lora/` | Research-only answer: [`docs/MICROTURN_LORA.md`](docs/MICROTURN_LORA.md). No |
| `recall-miss/` | A measured answer to one real miss: a session proposed a new 30-min check-in watchdog |
| `research-channels/` | Part of the **AI research pipeline** ("most intelligence per GPU", `docs/PLAN.md`). |
| `research-digest/` | This repo hosts the AI-research pipeline work ("most intelligence per GPU"): **R0**, a |
| `research-experiments/` | This repo hosts the AI-research pipeline work ("most intelligence per GPU"): **R0**, a |
| `research-ingest/` | This repo hosts the AI-research pipeline work ("most intelligence per GPU"): **R0**, a |
| `research-looped-lm/` | This repo hosts the AI-research pipeline work ("most intelligence per GPU"): **R0**, a |
| `research-setup-upgrades/` | This repo hosts the AI-research pipeline work ("most intelligence per GPU"): **R0**, a |
| `small-vs-big/` | One agent harness (fixed `bash`/`read_file`/`write_file`/`finish` loop, 30 turns, |
| `strata-safe/` | Research answer to: **can this box (RTX 3090 24 GB, 30 GB RAM now, 62 GB after |

## Licence
MIT (`LICENSE`).
