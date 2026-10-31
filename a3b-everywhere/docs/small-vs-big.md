# R7 — small specialists vs one 27B in a coding role

Status: **harness built and the 42 GPU jobs queued (session 1).** No arm has run
yet; every number below the *Method* section arrives from a runner GPU job and is
written to `state/small-vs-big/<arm>.json`. A session never writes those files.

Question: on one 24 GB RTX 3090, does a small specialised stack — terse output
(`docs/caveman.md`), LoRA routing, a test-gated cascade — beat a single 27B at
real repository coding tasks per GPU hour and per output token?

## What is reused (prior art, not rebuilt)

| Reused | Where | What for |
|---|---|---|
| fixed agent loop (`bash`/`read_file`/`write_file`/`finish`, 30 turns, 32K, wall-clock budget) | `agent-distill/distill/agent.py` | vendored verbatim into `smallvsbig/distill/agent.py`; the 2026-10-04 repo-path bug fix is included |
| 4-bit transformers backend + `<tool_call>` parser | `agent-distill/distill/local_backend.py` | vendored; serves SERA-8B and the peft student |
| trained SERA-8B student adapter | `agent-distill` `state/adapters/sera-8b-r1` (95 MB) | copied to gitignored `artifacts/sera-8b-student` for arms C/D/F |
| QLoRA trainer | `agent-distill/distill/train.py` | called (not copied) by `smallvsbig/train_expert.py` for the two arm-E experts |
| task builders | `agent-distill/distill/{build,mutations,workspace}.py` logic | reimplemented in `smallvsbig/tasks.py`; reads source repos read-only |
| terse system prompt | `prompts/caveman-system.txt` | arm B/D add-on, prose/reasoning only |

The prior-art finder was already run by the runner (`PRIOR_ART.md`): only weak,
indirect GitHub hits, nothing to fork.  The two sibling projects (`agent-distill`,
`employee-shootout`) are the real prior art and are used directly.

## Task set — 58 entries, same for every arm

- **44** of agent-distill's held-out tasks (`tasks/agent-distill-heldout.jsonl.txt`):
  30 injected bugs (export `base_commit`, apply the recorded AST mutation) and 14
  mined tasks with the shootout-origin `shot-*` ids.
- **14** employee-shootout tasks (`tasks/shootout.json.txt`), built from `parent`
  with the fix commit's test files overlaid.

**Honest overlap:** the 14 shootout tasks are the same 14 repos/commits already
present inside agent-distill's 44 as `shot-*` records. The spec asks for both sets,
so both are built and run under their own ids (58 entries). 14 of the 58 are
therefore the same start workspace twice; the report keeps that visible rather
than hiding it.

`smallvsbig/tasks.py` materialises each workspace under the gitignored
`state/small-vs-big/work/<id>/workspace` and records the failing tests.  Building
all 58 and probing `fails_before` is one command:

```console
$ python3 -m smallvsbig.tasks --force
built 58/58 tasks -> state/small-vs-big/work
```

All 58 were verified to fail before the fix (recorded `failing` non-empty).

## Arms

| Arm | Backend | Model | Prompt | Notes |
|---|---|---|---|---|
| A | Ollama | `qwen3.8-27b-uncensored:latest` (Q4_K_M, on disk) | normal | the single big model |
| B | Ollama | same 27B | + caveman terse | measures whether terse costs accuracy |
| C | transformers 4-bit | `allenai/SERA-8B` + agent-distill student adapter | normal | the trained 8B student |
| D | transformers 4-bit | same student | + caveman terse | |
| E | transformers 4-bit | one SERA-8B base + **two** peft experts | normal | router picks the expert by task repo (`job-radar` / `prior-art`), others fall back to `job-radar`; one base resident |
| F | cascade | student → 27B | normal | student attempt, retry once with the failure output, escalate to the 27B only if it still fails; `escalations` counted |
| G | transformers 4-bit | SERA-8B with the middle-third decoder block run twice | normal | the cheap looping-depth stand-in; report degradation |

Arm E's two experts are trained by `smallvsbig/train_expert.py` on the two
families of agent-distill teacher trajectories (job-radar 104, prior-art 55); the
two training jobs run before E's evaluation jobs.

## Metrics (per arm)

`state/small-vs-big/<arm>.json` carries exactly:
`arm, n, passed, pass_rate, median_wall_s, tokens_out_per_task, peak_vram_gb,
gpu_minutes, passes_per_gpu_hour, escalations`.

- `n` counts finished episodes (a timeout still counts).
- `gpu_minutes` sums per-episode model wall time; `passes_per_gpu_hour` divides
  solved tasks by it, so a slow cascade is not flattered.
- `tokens_out_per_task` is the mean completion tokens per episode.
- `peak_vram_gb` is the max over episodes (torch allocated for in-process
  models, Ollama `/api/ps` `size_vram` for served models).

## How it runs

Each arm shards the 58 tasks into `gpu_jobs/r7-*.job.yaml` requests (240 min cap,
`ollama: true` where a served model is needed). Per-task records are resumable
under `state/small-vs-big/runs/<arm>/`; one resumable mop-up job per arm catches
tasks a timed-out shard left behind. The final job writes every `<arm>.json`:

```console
$ python3 smallvsbig/run.py --finalize all
[..] finalized A: n=58 passed=.. gpu_minutes=..
```

The jobs honour the spec note by checking agent-distill's harness gate
(`state/harness-gate.json`, teacher 5/5, `base_ran_tests_in_repo=true`) before an
arm runs.

## Results

Pending — the table is generated from the runner's result files:

```console
$ python3 smallvsbig/report.py
| arm | n | passed | pass_rate | median_wall_s | tokens_out_per_task | peak_vram_gb | gpu_minutes | passes_per_gpu_hour | escalations |
| ... |
```

The session that reads the finished `state/small-vs-big/*.json` will:
1. run `python3 smallvsbig/report.py --out docs/small-vs-big.md --section`;
2. write which setup wins per GPU hour and per token, whether caveman costs
   accuracy, whether composition (E/F) beats the 27B, and the recommendation for
   the local coding lane — failures reported as plainly as wins.

## Known risks

- 32K context on the Q4 27B may not fit 24 GB; an OOM is recorded as a failed
  episode, not smoothed over.
- Arm E's router is repo-based because all held-in teacher data is `bugfix`; the
  two experts are family specialists, not planner/coder.
- 14 of the 58 tasks are duplicates (above); their separate run is deliberate.
