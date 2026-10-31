# microturn-lora — can our A3B lane / sales-brain use the MicroTurn Duplex LoRA?

Research-only answer: [`docs/MICROTURN_LORA.md`](docs/MICROTURN_LORA.md). No
adapter or GGUF was downloaded; no third-party code was installed or run. The
only code read is the public 3.4 KB client and the llama.cpp source already on
this machine.

**What it is.** A **15.3 MB GGUF LoRA** (Apache-2.0) for
`Qwen/Qwen3.6-35B-A3B` that teaches a 0.6 s micro-turn voice protocol: each
tick reads a streaming-STT fragment (or `<no voice>`) and replies with up to
three words or `<listen>`. It is a **turn-taking** adapter, not a sales-cue
model, and it is **English-only**.

**Verdict in one line.** Safe to download (no pickle, no `.bin`, no
`.safetensors`, no `trust_remote_code`, no install script, no telemetry); it
should load on our pinned llama.cpp + cached Q4_K_M; **skip for sales-brain
cues** (the 600 ms protocol was already measured there: 46.4 % false cues,
107 s p50 CPU) and **skip for the A3B text lane** (three-word replies). The
smallest decisive test is **15.3 MB and ~10 GPU-minutes**, for Claude to
approve.

**Repo state (session 2).** The inherited worktree had 53 generated data files
tracked (`data/codacus/*.json`, `data/kb*`, `state/*`). They are now untracked:
generated artifacts live in the gitignored `data/generated/` and `state/`, and
`hygiene.py scan --repo .` is clean (2,502 files). The 119-test offline suite
still passes because artifact-dependent tests use the local gitignored cache.

## Install

None. The deliverable is the document. The repo's existing Python harness needs
nothing beyond `requirements.txt` (PyYAML/pytest); no new package, venv or
model is required to read it.

## Run

```console
$ test -s docs/MICROTURN_LORA.md && echo ok
ok
$ grep -c 'https\?://' docs/MICROTURN_LORA.md
10
$ wc -l docs/MICROTURN_LORA.md
511 docs/MICROTURN_LORA.md
```

The document answers the six questions: what it is (§1), the turn protocol,
data and training (§2), vetting (§3), fit for our llama.cpp/3090/lane (§4),
three use-case verdicts (§5) and the smallest test for Claude to approve (§6).

**Reused, not rebuilt:** the already-downloaded captions
(`data/codacus/shJLycXjhVY.txt`), the house triage in
`docs/CODACUS_IDEAS.md` row 29, the measured A3B VRAM/decode in moe-offload,
the pinned llama.cpp source, and the already-run micro-turn experiment in
sales-brain (`docs/VOICE_LATENCY.md`).

---

# recall-miss — why Claude forgot the watchdog, and the structural fix

A measured answer to one real miss: a session proposed a new 30-min check-in watchdog
while `queue-watchdog.timer` and `bin/health.py -> state/alerts.jsonl` were already
running. `docs/RECALL_MISS.md` maps nine causes to the literature and to the session
facts, ranks eight structural fixes (hook / runner / lint, not "remember harder"), and
replays the session's 58 prompts through the current hook and a push-first prototype.

**Result:** the current hook named `queue-watchdog` or `health.py` on **0/58** prompts
(16/58 returned some wrong context); the proposed hook named both at SessionStart and
on 29/58 prompts, including the bottleneck question. Full table and caveats in
`docs/RECALL_MISS.md`.

## Install

Python standard library only; the replay reads `~/.claude/history.jsonl` read-only.
Tests need `pytest` (already in `requirements.txt`). Nothing is installed under
`~/.claude` by this repo; `tools/recall_hook.py` is the prototype to copy if the owner
adopts it.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3 -m pytest
# kv-frontier — which 2026 KV-cache method gives the A3B more context

Research + a runnable measurement harness for one question: on a 24 GB RTX 3090,
with Qwen3.6-35B-A3B (Q4_K_M) and its DFlash draft at one serving slot, which
KV-cache method buys the most real long context — and is any 2026 research
method worth porting to llama.cpp?

Status (session 2, 2026-10-18): **research and harness done; `f16` measured,
`q4_0` queued.** `docs/KV_FRONTIER.md` has the method table (SGD-KV, CompressKV,
KIVI, SnapKV, TurboQuant, q8_0/q5_1/q4_0/FP8), the A3B cache arithmetic, the
measured `f16` cells, and the port recommendation. `state/a3b_server.args.kv-proposed`
holds the proposed serving flags (the live args file was not touched).
`gpu_jobs/kv-measure2.job.yaml` is the fixed second run: the first job called a
server "fits" on `/health` alone and died at the first prompt, so a config now
has to survive a full 2048-token prefill batch. Completed `f16` cells are reused.

Measured so far: `f16` fits **49,152** tokens beside the DFlash draft, 960 MiB
of KV, **2.8K tok/s prefill**, **129–180 tok/s decode**, needle 3/3 and
multi-hop 1/1 on the real mixed contexts (the "64K/128K" prompts were capped at
48K, so they are not 64K accuracy evidence). `q4_0`, `q8_0`, `q5_1` and the
asymmetric mixes are "not run" in the table until the second job lands.

Headline: TurboQuant's llama.cpp PR was closed unmerged (June 2026); upstream
llama.cpp now has Hadamard-rotated `q8_0`/`q4_0` KV. The A3B has only 10
full-attention layers (2 KV heads × 256 dim), so `q4_0` costs 5,760 B/token vs
20,480 B/token for `f16` — the projection is 64K–128K where `f16` stops at 49K.
CompressKV is the best paper candidate but has no licence file, no llama.cpp
port, and no MoE/hybrid result: **wait, do not port**.

## Install

Python standard library only; tests need `pytest`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Run

```bash
# the measurement (read-only; prints only indexes and hit/miss, never prompt bodies)
python3 tools/recall_replay.py --session 620f5a88-f819-4b1b-a5de-f180bc3ed532 \
  --json-out state/recall-replay.json

# the proposed SessionStart push (live unit files + deduped open alerts)
python3 tools/recall_hook.py session-start | python3 -m json.tool

# the proposed UserPromptSubmit path for one prompt
python3 tools/recall_hook.py user-prompt --prompt "why is the gpu idle and the queue stuck"
```

Real output from this repo (2026-10-18):

```console
$ python3 tools/recall_replay.py --session 620f5a88-f819-4b1b-a5de-f180bc3ed532
session prompts: 58  (source: ~/.claude/history.jsonl, read-only)
SessionStart push: queue-watchdog=True health.py=True
UserPromptSubmit: current hook any-hit 0/58; proposed any-hit 29/58
  current returned some context on 16/58 prompts, but named neither system
  current  queue-watchdog 0/58, health.py 0/58
  proposed queue-watchdog 28/58, health.py 21/58
  #54 bottleneck: current=[False, False] proposed=[True, True]
  #56 watchdog: current=[False, False] proposed=[True, True]
  #57 watchdog: current=[False, False] proposed=[True, True]
```

The proposed SessionStart block is eight lines, for example:

```text
Systems we run (live; reuse before building anything new):
- orchestrator-health [5min]: Orchestrator health: alert + safe fixes for failed, hung or idle work (bin/health.py) -> state/alerts.jsonl
- queue-watchdog [5min]: agent-watchdog queue check: rules find stuck queue items and GPU jobs, Laya chooses the action
- gpu-queue: Serial GPU job queue worker (qsubmit)
...
```

Reused rather than rebuilt: the live `~/.config/systemd/user/*.service|*.timer`
descriptions (the authoritative list of what runs), `state/health.json` open keys as the
dedupe key set, `state/alerts.jsonl` for messages, the existing queue specs and `bin/*.py`
docstrings as the systems catalogue, and the local inventory/XERJ for the longer-term
semantic recall fix. No new database, no daemon, no second alert store.

Files: `docs/RECALL_MISS.md` (report), `tools/recall_hook.py` (proposed hook),
`tools/recall_replay.py` (measurement), `tests/test_recall_hook.py` (offline checks).
# offline: KV size model, scorer, fixtures
python3 -m pytest -q tests/test_kvfrontier.py

# see the real long contexts the GPU job builds (dev calls + captions + repo work)
python3 -m kvfrontier.measure --dry-run

# after the GPU job: render the measured tables
python3 -m kvfrontier.report
```

Real output of the dry run (2026-10-18):

```
built mix-32k: 85,196 chars, est 32,767 tokens, docs {'calls': 2, 'talks': 2, 'repos': 1}
built mix-64k: 170,393 chars, est 65,535 tokens, docs {'calls': 3, 'talks': 4, 'repos': 1}
built mix-128k: 340,787 chars, est 131,071 tokens, docs {'calls': 5, 'talks': 6, 'repos': 1}
```

Real output of `python3 -m kvfrontier.report` after the first GPU run (only the
`f16` row is measured; the `q8_0` fit was not proven):

```
| f16 | f16 | f16 | 49152 | 960.0 |  | / |
| f16 | mix-32k | 28195 | 2835.37 | 154.11 | 1.0 | 1.0 | — |
| f16 | mix-64k | 48349 | 2573.57 | 129.43 | 1.0 | 1.0 | — |
| f16 | mix-128k | 48020 | 2658.7 | 179.6 | 1.0 | 1.0 | — |
```

Real output of the size model (`python3 -m kvfrontier.report`):

```
| KV type | bytes/token | 32K | 64K | 128K | 256K |
| f16 | 20480 | 640 MiB | 1280 MiB | 2560 MiB | 5120 MiB |
| q8_0 | 10880 | 340 MiB | 680 MiB | 1360 MiB | 2720 MiB |
| q5_1 | 7680 | 240 MiB | 480 MiB | 960 MiB | 1920 MiB |
| q4_0 | 5760 | 180 MiB | 360 MiB | 720 MiB | 1440 MiB |
```

## Reused

`moe-offload`'s CUDA llama.cpp build (`f336359`) and its 32K/64K OOM data,
`a3b-tune`'s DFlash serving flags, `a3b-shootout`'s server/timings pattern, the
sales-brain realspeech **dev** manifest, the codacus-mine captions, and the open
sales-brain local-lane work items. No raw text, model weights, databases or
absolute home paths are committed.

---

# codacus-mine — last 30 Codacus videos → buildable ideas

Turn the last 30 uploads of the Codacus channel into a table of mechanisms we can
build or borrow for one 3090 + 30 GB RAM. Three stages: captions only (yt-dlp,
no media, no login), a local 9B extraction per video (mechanism / claimed numbers /
links / target hardware / which goal it moves), then a DeepSeek reverse-engineering
pass and `docs/CODACUS_IDEAS.md`.

Status (session 2, 2026-10-17): **done.** All 30 captions, the 9B answered
**30/30**, and `python3 -m codacus.assemble` wrote **30 per-video JSONs**;
`docs/spot-checks-codacus.md` audits all 30 (167 claims, 150 grounded links,
2 numeric findings); `docs/CODACUS_IDEAS.md` has the 30-row idea table, the
source-code pass and the **top 5** ranked build sketches. The links in each JSON
are filtered to URLs the transcript or description actually carries
(`assemble.ground_links`): the 9B had invented one repo URL, and that is now the
real fork from the description. All four `done_when` commands pass.

Reused rather than rebuilt: `airesearch.ingest.fetch_video` (the research-ingest
caption path — `--skip-download --write-info-json --write-subs --write-auto-subs`,
English VTT, cached per video), `airesearch.prompts.vtt_segments` (rolling
auto-caption de-duplication), and the `local_batch` house tool for the 9B pass.
For the ideas, read-only: the llama.cpp PRs (22105 DFlash, 22673 MTP, 27342
DFlash 2), `JustVugg/colibri`, `z-lab/dflash`, `thecodacus/understory` and the
REA (`morluto/rea`) method. Nothing third-party was installed or run.
New: the channel list, the plain-text transcript writer, the extraction
schema (`system / technique / mechanism / claims / links / hardware / goals /
buildable`), link grounding, and the extraction audit (`codacus/spotcheck.py`).

## Install

Python standard library only; tests need `pytest`. Caption fetching needs
`yt-dlp` (repo `.venv/bin`, `PATH`, or a local install):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Run

```bash
python3 -m codacus.ingest list          # last 30 uploads + the two brief links -> data/codacus/videos.json
python3 -m codacus.ingest captions --sleep 5   # one yt-dlp call per video, 5 s apart
python3 -m codacus.ingest status
python3 -m codacus.extract             # -> local_batch/codacus-prompts.jsonl + codacus-extract.yaml
# runner serves the batch, then:
python3 -m codacus.assemble            # data/codacus/extracts.jsonl -> data/codacus/<id>.json
python3 -m codacus.spotcheck           # -> docs/spot-checks-codacus.md
python3 -m pytest
```

YouTube answered HTTP 429 to two quick requests on 2026-10-17; the caption loop
sleeps 5 s between videos and backs off 60 s on a retry, so a rate-limited video
is retried rather than skipped forever. All 30 succeeded.

Real output from this repo:

```console
$ python3 -m codacus.ingest status
videos=30 transcripts=30
chars min=6658 median=14192 max=22599

$ python3 -m codacus.extract --check
30 prompts, max 7000 chars

$ python3 -m codacus.assemble
wrote 30/30 video JSONs -> data/codacus/

$ python3 -m codacus.spotcheck
30 videos, 167 claims, 150 links, 2 with findings -> docs/spot-checks-codacus.md

$ python3 -m pytest
93 passed in 1.23s
```

The 30 transcripts and the per-video extraction JSON are tracked (the merged main
repo re-runs `done_when` there) but `export-ignore`d, so the published tree carries
no raw-caption dump. They are fetched into `data/private/captions/` first and only
the plain text is committed.

## Next step

Top-1 in `docs/CODACUS_IDEAS.md`: run the DFlash 2 A/B on Qwen3.8-27B (check the
pin contains llama.cpp PR 27342, fetch the 1.14 GB Q4 draft, compare 20 prompts
for tok/s and greedy-equality). Top-2/3/4/5 are the MTP fallback, the MoE
offload sweep, the understory-style memory layer and the Jev/Laya decision
endpoint.
# strata-safe — offline, vetted Strata + Qwen3.8-Flash-Next fit for the 3090

Research answer to: **can this box (RTX 3090 24 GB, 30 GB RAM now, 62 GB after
+32 GB) run `DavidAU/Qwen3.8-Flash-Next-...-GSQ-RCO-GGUF` with Strata, and is the
RAM purchase worth it?** Read `docs/STRATA_SAFE.md` for the model card, the fit
tables, the Strata audit and the verdict.

**Status (session 2):** model facts and fit math done; the offline Strata fork is
vendored in `vendor/strata/`, its C++ front end builds under
`bwrap --unshare-net`, and the archived tree passes the runner's secrets,
personal-data and raw-data scans. The CUDA engine build is the one open item: it
needs a host with CMake 3.24+ and CUDA 12.x (`vendor/strata/BUILD-OFFLINE.sh`,
exact command in `RUNNER_TODO.md`). **No model weights were downloaded** and no
`.gguf` is committed. The recommended safe path is not the DavidAU file: it is
SC117/ISTA **IQ3_XXS** + the MTP head + a rebuilt pack, which needs **62 GB** of
RAM.

## What is here

| Path | What |
|---|---|
| `docs/STRATA_SAFE.md` | model card facts, GSQ/RCO explanation, three-tier fit table (VRAM → RAM → NVMe), Strata + fork audit and patch list, converter verdict, llama.cpp/ik_llama.cpp alternatives, Bolt/Colibri comparison, licence + parser-CVE notes, verdict |
| `vendor/strata/` | the reviewer's MIT offline fork (`robertkarl/UnbiasedStrata` @ `b124b59`) with the pinned llama.cpp submodule; no `.gguf`, no prebuilt binary, no network build |
| `vendor/strata/VENDORED.md` | provenance, what was removed, the one build change, run command |
| `vendor/strata/BUILD-OFFLINE.sh` | `bwrap --unshare-net` build (CMake + CUDA), writes `vendor/strata/build/unbiased-strata` |
| `tools/strata_fit.py` | the fit calculator used for the tables (stdlib, opens no model) |
| `tests/test_strata_fit.py` | four offline checks of the formula and the 30/62 GB fit flags |
| `tests/test_vendor_hygiene.py` | offline guard: no weights, no file > 2 MiB, no `cv`/`resume` names, no real email in `vendor/` |

## Install

Nothing to install for the research (Python 3 standard library only). To build
the engine, you need Linux x86-64 with AVX2, CMake ≥ 3.24, a C++20 compiler and
CUDA toolkit 12.x (sm_86 for the RTX 3090):

```sh
cd vendor/strata
./BUILD-OFFLINE.sh 86          # runs make build inside bwrap --unshare-net
./build/unbiased-strata --help
```

The model, MTP head and pack are **not** in this repo and are not downloaded by
this project (cost gate). The commands to make them are in
`docs/STRATA_SAFE.md` §4.5 and `vendor/strata/UPSTREAM.md`.

## Run (real output from this session)

Fit table (excerpt; full table in `docs/STRATA_SAFE.md`):

```console
$ python3 tools/strata_fit.py
| RAM | quant | download GB | experts GB | hit | Strata fits | llama.cpp fits | Strata tok/s | llama.cpp tok/s | Strata prefill tok/s |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| 30 GB | Q2_0 | 66.4 | 34.0 | 92.8% | no | yes | — | 55-65 | — |
| 30 GB | IQ2_XS | 68.0 | 35.5 | 92.4% | no | yes | — | 48-57 | — |
| 30 GB | IQ3_XXS | 75.8 | 42.9 | 90.3% | no | no | — | 33-40 | — |
| 30 GB | IQ3_S | 83.6 | 50.3 | 88.6% | no | no | — | 22-32 | — |
| 62 GB | Q2_0 | 66.4 | 34.0 | 92.8% | yes | yes | 90-105 | 55-65 | 1750 |
| 62 GB | IQ2_XS | 68.0 | 35.5 | 92.4% | yes | yes | 78-91 | 48-57 | 1550 |
| 62 GB | IQ3_XXS | 75.8 | 42.9 | 90.3% | yes | yes | 59-77 | 40-48 | 1400 |
| 62 GB | IQ3_S | 83.6 | 50.3 | 88.6% | yes | yes | 46-62 | 32-38 | 1200 |
```

Measured on this machine (not estimates):

```console
$ free -g | head -2
               total        used        free      shared  buff/cache   available
Mem:              30          13          12           0           5          17
$ cat /sys/bus/pci/devices/0000:01:00.0/current_link_speed
16.0 GT/s PCIe                       # x16 = PCIe 4.0
$ dd if=<5.3GB file> of=/dev/null bs=1M count=2048 iflag=direct
2147483648 bytes (2.1 GB) copied, 0.538897 s, 4.0 GB/s
$ python3 .scratch/nvme_read.py     # 980 PRO, O_DIRECT random reads
4096     MBps 51.8    IOPS 12648.5
65536    MBps 614.1   IOPS 9370.6
1048576  MBps 3971.7  IOPS 3787.7
```

Offline front-end build (the only build this sandbox can do; no CUDA toolkit):

```console
$ bwrap --unshare-net --ro-bind / / --bind "$PWD" "$PWD" --tmpfs /tmp         --dev /dev --proc /proc --chdir "$PWD" make test-frontend
PASS: normal exit and watchdog-style abort reaped and diagnosed
PASS: formatter exceptions and client cancellation drain the old request before the next request
PASS: large responses cannot fill an abandoned pipe or leak into the next request
```

Tests:

```console
$ python3 -m pytest
...
87 passed in 0.19s
```

## Reused rather than rebuilt

- `codacus-mine`'s Strata explainer summary and the 30-video caption set
  (`docs/CODACUS_IDEAS.md`, `data/codacus/6WLBmP-tZ0Q.txt`) supplied the hot-expert
  numbers and the "vet failed" starting point.
- The house **llama.cpp pin `f3363593`** (2026-10-16) is the alternative engine; it
  already contains PR #29887's `--moe-cache-mib` and the `qwen4exp` architecture.
- The reviewer's fork `robertkarl/UnbiasedStrata` is the safe engine; upstream
  `Niko1221/Strata` is read-only evidence for the unsafe install path.
- The **inventory** search could not run in the sandbox (`state/vectors.json` is
  outside the writable tree); `inventory.md` was read instead and had no Strata
  component. The prior-art report (`PRIOR_ART.md`) is for a different project and
  was not used.

---

# bycloud-ingest — deep pass over the bycloud channel

Download bycloud's full public video catalogue (newest first), turn each video's
English captions into a per-video transcript JSON, extract concepts/claims/papers
with the local 27B, and map the concepts onto our own system in
`docs/bycloud-upgrades.md`.

Status (session 3): catalogue **257 videos / 43.19 h** (2020-10 → 2026-10, the
videos tab; streams/VODs excluded), **252 transcripts / 42.69 h** in
`data/private/bycloud/transcripts/`, 5 videos with no captions at all
(`data/private/bycloud/no_captions.json`, ≤ 10 so no Whisper job). The local 27B
batch finished all **252/252** answers; the re-armed `bycloud-assemble` GPU job
wrote **252 rows / 1630 concepts (1439 grounded) / 1018 claims** to
`data/bycloud-extracts.jsonl` (the script refuses a partial raw file, the failure
mode of the first run). `docs/bycloud-upgrades.md` maps the extracted concepts to
forge / agent-distill / lora-expert-bank / rl-extraction / merge-lab / the
orchestrator with 44 grounded `[video_id@mm:ss] "quote"` citations. The 10 %
spot-check (`docs/spot-checks.md`, bycloud deep-pass section) re-read 26 videos:
164/164 concept quotes verbatim, 164/164 start times within 30 s, 97/97 claim
numbers grounded in the transcript or the title. The catalogue ends below 100 h,
so the whole catalogue is taken. The result file is tracked (the merged main repo
must re-run the assert) but `export-ignore`d, like `data/kb-extracts.jsonl`, so
the published tree carries no data dump.

Reused rather than rebuilt: research-ingest's caption path (`yt-dlp
--skip-download --write-info-json --write-subs --write-auto-subs`, English, vtt)
and its already-downloaded caption cache (via `--reuse-dir`); only the
per-video transcript schema and the rolling auto-caption de-duplication are new.

## Install

Python standard library only. Caption fetching needs `yt-dlp` (repo `.venv/bin`,
`PATH`, `BYCLOUD_YTDLP`, or a local install); tests need only `pytest`.

## Run

```bash
python3 -m bycloud catalogue                  # yt-dlp flat playlist -> catalogue.json + videos.json
python3 -m bycloud captions --workers 3 --reuse-dir <caption-cache>
python3 -m bycloud prompts --queue            # -> local_batch/bycloud-extract.yaml
python3 -m bycloud status
python3 -m pytest -q
```

On the GPU the runner serves the local batch, then submits
`gpu_jobs/bycloud-assemble.job.yaml` (`script: bycloud/assemble_extract.py`), which
parses each answer, adds `video_id`, re-grounds every concept quote against the
transcript and writes `data/bycloud-extracts.jsonl`. The report check:

```bash
python3 ~/theprojects/orchestrator/bin/check_grounded_report.py \
  docs/bycloud-upgrades.md data/private/bycloud/transcripts 20 0.9
```

Real output from this repo:

```console
$ python3 -m bycloud catalogue
catalogue: 257 videos, 43.19 h -> data/private/bycloud/catalogue.json

$ python3 -m bycloud status
catalogue=257 transcripts=252 (42.7 h) total_hours=43.19 no_captions=5

$ python3 -m bycloud prompts --queue
prompts: 252 -> local_batch/bycloud-prompts.jsonl and local_batch/bycloud-extract.yaml

$ python3 -c "import json,glob;t=[json.load(open(f)) for f in glob.glob('data/private/bycloud/transcripts/*.json')];h=sum(x['duration_s'] for x in t)/3600;c=json.load(open('data/private/bycloud/catalogue.json'));assert len(t)>=136 and h>=0.95*min(100,c['total_hours']) and all(len(x['segments'])>0 for x in t),(len(t),round(h,1),c['total_hours'])"
(exit 0)

$ python3 -c "import json;r=[json.loads(l) for l in open('data/bycloud-extracts.jsonl')];assert len({x['video_id'] for x in r})>=136,len(r);print(len(r),'rows')"
252 rows

$ python3 ~/theprojects/orchestrator/bin/check_grounded_report.py docs/bycloud-upgrades.md data/private/bycloud/transcripts 20 0.9
grounded citations: 44/44 (1.00); need >= 20 and >= 0.9

$ python3 -m pytest -q
83 passed
```

---

# small-vs-big — R7: small specialists vs one 27B in a coding role

One agent harness (fixed `bash`/`read_file`/`write_file`/`finish` loop, 30 turns,
32K context, per-episode wall-clock budget) runs the same 58 held-out repository
tasks through seven arms on the RTX 3090: A the on-disk 27B, B the 27B with the
caveman terse prompt, C the trained SERA-8B student, D terse student, E two LoRA
experts on one SERA-8B base routed per repo, F the student→27B test-gated cascade,
G SERA-8B with a middle block of decoder layers run twice. Every arm writes
`state/small-vs-big/<arm>.json` from a runner GPU job; `docs/small-vs-big.md` holds
the method and (next session) the table.

Reused rather than rebuilt: agent-distill's fixed agent loop, 4-bit transformers
backend and QLoRA trainer, its trained SERA-8B student adapter, the employee-shootout
task set, and `prompts/caveman-system.txt` (see `docs/caveman.md`).

## Install

Python standard library only. The GPU jobs use the shared training venv
(`~/theprojects/gpu/.venv`) with CUDA torch/transformers/peft; the offline tests
need `pytest` and a local Qwen3 tokenizer snapshot only for the caveman test.

## Run

```bash
python3 -m smallvsbig.tasks --force            # build the 58 task workspaces (offline, ~2 min)
python3 smallvsbig/run.py --arm C --shard 0/4 --dry-run   # check task selection
python3 smallvsbig/run.py --finalize all       # aggregate records -> state/small-vs-big/<arm>.json
python3 smallvsbig/report.py                   # render the arm table
python3 -m pytest -q
```

On the GPU the runner submits `gpu_jobs/r7-*.job.yaml` (training → arm shards →
mop-ups → aggregate). Real offline output from this repo:

```console
$ python3 -m smallvsbig.tasks --force
built 58/58 tasks -> state/small-vs-big/work

$ python3 smallvsbig/run.py --arm A --shard 0/6 --dry-run
[..] arm=A tasks=10
inj-041cde230172 bugfix agent-distill
job-radar-d44fcbe9 feature shootout
...
```

---

# research-memory — R8 memory, context & KV-cache landscape

Survey of ways to give a compute-limited agent better memory and a better context window
on one 24 GB GPU, judged against the plain baseline we already run: `data/kb.sqlite`
(SQLite) + `sqlite-vec` KNN + FTS5 + a cross-encoder reranker
(`~/homelab/bin/rerank_service.py`). Inventory and the public repo finder ran first
(no prior art to fork); the KB was used to ground the KV-cache/attention claims.

- `docs/memory-landscape.md` — 30+ candidates across agent memory layers, code-aware
  context servers, context compression, KV-cache techniques, small CPU
  rerankers/retrievers, and tiny routing/"enough context?" classifiers. Each has a repo
  URL, the licence read from the repo's own file, what it needs, claimed numbers, and an
  **adopt / test / skip** verdict with the reason.
- `docs/memory-shortlist.md` — the 5 to test first, each with a concrete test on our data
  (past conversations, this repo, `kb.sqlite`) and the metric + decision rule.

## How to check

No install, no download and no network needed to read them; the two `done_when`
assertions run offline from the repo root:

```console
$ test $(grep -c 'github.com/' docs/memory-landscape.md) -ge 20
exit=0
$ test $(grep -c '^## ' docs/memory-shortlist.md) -ge 5
exit=0
$ grep -c 'github.com/' docs/memory-landscape.md
38
$ grep -c '^## ' docs/memory-shortlist.md
6
```

The shortlist names the tests to queue next; none is fetched or installed here. Model
downloads > 1.5 GB go through the cost gate before any fetch.

Reused: the local `rerank_service` and `embed_semantic` (from the inventory) as the
reranking/embedding baseline, and the `prior_art` finder as the first check. This
session adds only the two documents.

---

# caveman-terse — terse output prompt + offline token-savings measurement

Adopt the [caveman](https://github.com/JuliusBrussee/caveman) "talk terse" idea for our
small-model runs, adapted to this harness, and size the win on our own transcripts
before spending any GPU time.

- `docs/caveman.md` — what upstream caveman does, its exact prompt rules, claimed
  savings (65% average vs verbose output), licence (Apache-2.0 since 3.0.0; README badge
  still says MIT), and the risks (tool-call JSON, code, test output, negations).
- `prompts/caveman-system.txt` — drop-in system-prompt add-on. Scope is explicit:
  **prose and reasoning only**; tool-call arguments, code, file contents, commands,
  paths, numbers and error text stay byte-exact.
- `tools/caveman/extract_turns.py` — deterministic selection of real assistant turns
  from the saved orchestrator transcripts (read-only, 120–1100 chars of prose,
  structured-JSON messages excluded, round-robin over sessions).
- `tools/caveman/rewrites.json` — the DeepSeek caveman rewrite of each turn's prose
  segments (real names sanitised out; fenced code untouched).
- `tools/caveman/measure.py` — tokenizes before/after with the **Qwen3 tokenizer**
  and writes `state/caveman-savings.json`.
- `tests/test_caveman.py` — 9 offline stdlib tests (artifact shape, prose-only
  reassembly, fence byte-exactness).

## Install / run

No new runtime dependency: the artifacts are committed stdlib/JSON. Rebuilding the
measurement needs `transformers` and a local Qwen3 snapshot (the shared GPU venv has
both; no network, CPU only):

```bash
python3 tools/caveman/extract_turns.py --limit 40          # -> state/caveman-turns.raw.json (gitignored)
~/theprojects/gpu/.venv/bin/python tools/caveman/measure.py
python3 -m pytest -q
```

## Real example

One measured turn (real orchestrator transcript; prose only rewritten):

```
before: The selftest passed, the change is minimal (single-file edit inside
        `bin/bets_due.py`), and it's already committed.  What now works: running
        `bin/bets_due.py selftest` prints `[ok]` and proves the core rule -- an
        unmeasurable bet is recorded as undecidable, never as a loss.
after:  Selftest pass. Change minimal (1 file, `bin/bets_due.py`). Committed.
        Works now: `bin/bets_due.py selftest` prints `[ok]`; unmeasurable bet
        recorded undecidable, never a loss.
```

Result over 40 turns (`state/caveman-savings.json`):

```console
$ ~/theprojects/gpu/.venv/bin/python tools/caveman/measure.py
n_turns=40 before=3768 after=3323 saving=11.81% -> state/caveman-savings.json

$ python3 -c "import json;r=json.load(open('state/caveman-savings.json'));assert r['n_turns']>=30 and 0<r['saving_pct']<100,r;print(r['n_turns'],r['saving_pct'])"
40 11.81
```

**11.81%** incremental saving on top of our already-terse harness style, with all
payload (code, paths, numbers, errors) byte-exact. Upstream's 65% is measured against
verbose output; our baseline already drops most filler, so the remaining headroom is
payload. The on-GPU correctness test is the launch kit's job, not this measurement.

Reused: the prompt rules and always-on snippet from JuliusBrussee/caveman (Apache-2.0),
adapted in `prompts/caveman-system.txt`; the transcript corpus and its extraction are
new here.

---

# AI research pipeline — R0 looped-LM deep-dive · R1 channel registry · R2 video ingest · R3 lens + digest · R4 experiment cards

This repo hosts the AI-research pipeline work ("most intelligence per GPU"): **R0**, a
looped/recurrent-depth language-model deep-dive with a real RTX 3090 test, **R1**, a
channel registry, **R2**, the caption/metadata ingest and knowledge base built on
the R1 kept list, **R3**, the "gains under limits" lens score over that knowledge
base with a top-50 digest and five experiment cards, and **R4**, those five cards
run as provenance-checked GPU jobs. Each project below is
self-contained.

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

# research-digest — R3 "gains under limits" lens + top-50 digest

Part of the **AI research pipeline** (`docs/PLAN.md` §2 lens and §5 steps 4 and 6). It
reads the R2 knowledge base `data/kb.sqlite` and turns it into a ranked reading list for
one RTX 3090. No LLM is in the final score and no randomness is used: rebuilding from the
same KB gives byte-identical output.

| Path | What |
|---|---|
| `airesearch/taxonomy.py` | the visible technique taxonomy: one profile per family with the four gain axes, cost in 3090 GPU-hours, theme and transfer areas |
| `airesearch/lens.py` | canonicalise + cluster the KB's methods, score each cluster, render the digest |
| `data/lens.json` | machine dump: every ranked cluster with all seven components and its evidence links |
| `docs/digest.md` | top 50, grouped by theme, one line each with YouTube-timestamp + arXiv links, a paper index and a component table |
| `docs/experiments/` | 5 experiment cards for the top items not yet tested by us (looped LMs are already in `docs/looped-lm.md`) |
| `tests/test_lens.py` | 6 offline tests (synthetic in-memory KB, no network) |

## Install and run

Python 3 standard library only — no new dependency. `data/kb.sqlite` must exist (R2).

```bash
python3 -m airesearch.lens build     # writes data/lens.json + docs/digest.md
python3 -m airesearch.lens top 20    # print the ranked list
python3 -m pytest -q                 # 28 tests
```

## How the score works

Every model-extracted method is canonicalised, folded onto a taxonomy family, and the
family is joined to every claim, paper, repo link and channel of its member videos.

```
gain     = 0.30·param + 0.30·flop + 0.25·ttc + 0.15·data
transfer = 100·(0.6·areas/3 + 0.4·videos/8)
evidence = 100·(0.35·papers/3 + 0.25·repos/3 + 0.20·numbers/15 + 0.20·channels/8)
priority = gain × (0.40 + 0.60·transfer/100) ÷ √max(cost_hours, 0.25)
           × (0.55 + 0.45·evidence/100) × 10
```

`param`/`flop`/`ttc`/`data` come from the technique's family profile in
`airesearch/taxonomy.py` (a judgement about the technique, written down so it can be
changed and re-run); `cost_hours` is that family's estimated 3090 GPU-hours; evidence is
always counted from the KB. The full numbers are in the component table in
`docs/digest.md` and in `data/lens.json`.

## Example output

```console
$ python3 -m airesearch.lens build
eligible clusters: 100
 1. 1170.00  Chain of Thought
 2. 1130.00  Prompt Engineering
 3. 1061.70  In-Context Learning
 4.  862.67  RAG
 5.  765.11  Tool Use
 6.  731.86  Embeddings
 7.  681.30  context engineering

$ python3 -m airesearch.lens top 5
 1. 1170.00  Chain of Thought  (videos=77, papers=41)
 2. 1130.00  Prompt Engineering  (videos=57, papers=14)
 3. 1061.70  In-Context Learning  (videos=13, papers=14)
 4.  862.67  RAG  (videos=121, papers=18)
 5.  765.11  Tool Use  (videos=58, papers=2)

$ grep -c '^- ' docs/digest.md
261
$ grep -c 'arxiv.org/abs/' docs/digest.md
250
$ ls docs/experiments/*.md | wc -l
6
```

One digest line, verbatim:

```markdown
- **QLoRA** — 4-bit frozen base + LoRA adapters fine-tunes 7B-13B (even 30B) in 24 GB `priority 366.6` · evidence: [video](https://www.youtube.com/watch?v=1N76mJ1pMro&t=231s) · [arXiv 2402.15116](https://arxiv.org/abs/2402.15116)
```

## What was reused / prior art

- `airesearch/score.py` and the R1 §2 lens: the same "deterministic formula over
  extracted fields, no LLM" pattern, promoted from per-channel to per-technique.
- `data/kb.sqlite` and `airesearch/kb.py` (R2): clustering and evidence read only that DB.
- The experiment-card shape of `docs/next-experiments.md` (R0) is reused for the five
  cards, which reference the same 3090 and `local_batch/` fixtures.
- The prior-art check (`~/theprojects/queue/prior-art`, cache `/tmp/prior-art`) returned
  only weak/indirect hits (codebase-memory, OpenWiki, data-lens), nothing to fork.

## Honesty notes

- The axis numbers are a written taxonomy, not a measurement. They encode the PLAN §2
  lens; the KB only supplies evidence. Change one integer and rebuild.
- R2 extraction is noisy: claims are attached to every method the model named for a
  video, so some evidence links are only co-located with a cluster. This is why axis
  signals are counted from the method key, not from the full claim text, and why
  `docs/digest.md` carries a noise caveat.
- A digest item is a claim with sources, never a verified result. Only a GPU job in the
  provenance ledger produces a result.

---

# research-setup-upgrades — R5 measured upgrades for our own box

R5 turns the R2 knowledge base and the R4 verified results into **measured changes for
this machine** (RTX 3090 24 GB, 30 GB RAM, Ollama, QLoRA), not a reading list.

1. `tools/upgrades/candidates.py` + `tools/upgrades/shortlist.py` join a hand-reviewed
   candidate list to a **deterministic query over `data/kb.sqlite`** (a timestamped video
   claim and an arXiv paper per technique) and to the R4 verified results, and write
   `docs/upgrades.md` — one line per candidate with source, expected gain, cost to try
   (GPU-minutes, GB) and risk.
2. The eight candidates that cost ≤30 GPU-min and ≤1.5 GB ran as
   `gpu_jobs/upg-*.job.yaml`. Each trial measured the workload **before** and **after**
   the change and wrote `state/upgrades/<name>.json`; only the runner's GPU jobs may
   write those files. Two changes were adopted (`+18.3%` aggregate throughput from
   parallel slots, `+44.8%` supervised tokens/s from QLoRA packing); six were rejected
   with the measured reason.
3. `docs/setup-changes.md` is the recommendation for the owner: exact Ollama env,
   Modelfile params, `llama-server` flags and training args, each with its measured
   gain, and the rejected changes with the measured reason. **No change is applied to
   the live system by this project.**

## Workloads measured

- the local 27B bulk-text batch: 50 real R2 ingest prompts through the cached
  `qwen3.8-27b-uncensored:latest` GGUF (tokens/s, KV-cache MB from the server log);
- sales-brain deal extraction from call transcripts with the cached 9B (field-F1,
  schema validity, latency);
- one NF4 QLoRA step-time / peak-VRAM loop on the cached 9B base
  `empero-ai/Qwen3.8-9B-Distill`.

## Install and run

```bash
# offline: stdlib only for the shortlist/report; pytest for the tests
python3 tools/upgrades/shortlist.py          # rebuild docs/upgrades.md from data/kb.sqlite
python3 tools/upgrades/shortlist.py --check  # fail if docs/upgrades.md is stale
python3 tools/upgrades/report.py             # markdown table of state/upgrades/*.json
python3 -m pytest -q

# each trial ran as: python3 tools/upgrades/trials.py --trial <name>  (<=30 GPU-min)
ls gpu_jobs/upg-*.job.yaml                   # the 8 requests
ls state/upgrades/*.json                     # the 8 job-written results
```

## Example output

```text
$ python3 tools/upgrades/shortlist.py
wrote .../docs/upgrades.md (52 lines)

$ grep -c '^## \|^- ' docs/upgrades.md
25

$ python3 tools/upgrades/report.py
| trial | area | workload | change | change_pct | adopt |
|---|---|---|---|---:|---|
| parallel-batch | inference-speed | 27B bulk-text batch | --parallel 2 with two concurrent requests ... | 18.26 | true |
| qlora-packing | fine-tune-efficiency | 9B QLoRA step | pack several short examples into one max_seq_length block ... | 44.79 | true |
| flash-attention | inference-speed | 27B bulk-text batch | --flash-attn on | -0.34 | false |
| speculative-draft | inference-speed | 27B bulk-text batch | --spec-type draft-simple -md Qwen3-0.6B-Q8_0.gguf | -0.61 | false |
| ... (8 rows total) |

$ python3 -m pytest
......................................                                   [100%]
53 passed in 0.14s
```

One shortlist line, verbatim (the KB source comes from `data/kb.sqlite`):

```markdown
- **flash-attention** · llama.cpp flash attention (-fa on) · source: KB video "Triton GPU Kernels Lesson #9 | Flash attention (part 1 - forward pass)" [00:00] https://www.youtube.com/watch?v=aDyfrArLPyw — "The video covers the forward pass of Flash Attention, with the backward pass to be covered" + arXiv 2205.14135 Flash Attention · expected: 10-40% faster prompt processing and ~5-20% faster decoding on long prompts · cost: 15 GPU-min, 0.0 GB · risk: low: numerically close to default attention; some quant types fused better than others · how: `llama-server -fa on; Ollama: OLLAMA_FLASH_ATTENTION=1` · **queued**
```

## What was reused / prior art

- `tools/upgrades/fixtures.py` is the R4 experiment harness
  (`research-experiments/tools/exp/common.py`) vendored almost verbatim: the sales-brain
  hard-transcript fixture, the JSON extractor and the field-F1 scoring.
- The R4 verified numbers in `docs/setup-changes.md` come from the sibling R4 results
  (`docs/results.md` / `state/exp/*.json`), reused, not re-measured.
- The `llama-server` control pattern (locate Ollama's GGUF blob + CUDA backend, start the
  server with exact flags) is R4 card 4's, reused for the 27B trials.
- The result-file shape and the "only a runner GPU job writes it" rule follow the R0–R4
  provenance ledger.

## Honesty notes

- `state/upgrades/` holds the eight job-written results. The done checks verify the
  schema; a session that hand-writes one fails the runner's provenance fingerprint.
- `DONE.md` now carries the R5 section at the top; the R0–R3 sections below are
  historical and stay for provenance.
- The KB is noisy (R2 extraction over-merges claims), so a source line is the best
  timestamped KB evidence for a technique, not proof the video is about it. Where the KB
  had no match, the curated fallback source in `candidates.py` is used.
- `nvidia-smi` inside the runner's bubblewrap namespace reports ~0 used VRAM, so the
  llama.cpp trials read the KV/compute buffer sizes the server logs at startup instead.

---

# research-experiments — R4: run the five digest cards on the 3090

`docs/experiments/*.md` (R3) states five smallest 24 GB tests. R4 turns each into a
script that runs **inside the runner's GPU sandbox** and writes its own
`state/exp/<card>.json`; no session ever writes those files. The runner records the
sha256 of each output in the provenance ledger, so a hand-edited result fails
verification.

| Card | Job | Script |
|---|---|---|
| 1 CoT vs self-consistency | `gpu_jobs/01-cot-vs-self-consistency.job.yaml` | `tools/exp/card1_cot_self_consistency.py` |
| 2 RAG vs QLoRA | `gpu_jobs/02-rag-vs-qlora.job.yaml` | `tools/exp/card2_rag_vs_qlora.py` |
| 3 4-bit format bake-off | `gpu_jobs/03-quantization-format-bakeoff.job.yaml` | `tools/exp/card3_quant_formats.py` |
| 4 speculative decoding | `gpu_jobs/04-speculative-decoding.job.yaml` | `tools/exp/card4_speculative.py` |
| 5 compiled prompts | `gpu_jobs/05-compiled-prompts.job.yaml` | `tools/exp/card5_compiled_prompts.py` |

## Install and run

Nothing to install beyond the test runner; the jobs use the shared GPU venv.

```sh
# offline unit tests for the harness (no GPU, no Ollama needed)
python3 -m pytest -q

# the runner submits each job after the session; to inspect what a job would run:
python3 -c "import yaml;print(yaml.safe_load(open('gpu_jobs/01-cot-vs-self-consistency.job.yaml')))"
```

Each card reads its model from the local Ollama store or the HF cache. Card 4's
0.64 GB Qwen3-0.6B draft (`fetch/qwen3-0.6b-gguf.yaml`) failed to download (peer
closed at 466/639 MB, then DNS failures), so card 4 records the missing draft in
`errors` and measures only the 0 GB `ngram-simple` alternative. Cards 1, 3, 4 and
5 are verified; card 2's corrected job is queued (it crashed twice on
`apply_chat_template` returning a `BatchEncoding`).

## Example result (real, verified)

Every card writes the same object (this is the schema the runner's done-check
asserts). This is the verified card 3 output, abridged:

```json
{
  "card": "03-quantization-format-bakeoff",
  "hypothesis": "a 4-bit GGUF keeps >=97% of BF16 field-F1 while decoding >=1.5x faster ...",
  "metric": "field micro-F1 vs BF16, tokens/s, peak VRAM (GB), cold-load seconds",
  "n": 250,
  "arms": [
    {"name": "bf16", "score": 96.08, "tokens_per_s": 44.78, "vram_gb": 16.06},
    {"name": "q4_k_m", "score": 97.2, "tokens_per_s": 95.6, "vram_gb": 5.23}
  ],
  "pass": true,
  "gpu_minutes": 8.5
}
```

The verified numbers and the per-card verdicts are written to `docs/results.md`;
`docs/orbit-outlines.md` turns them into three ORBIT episode outlines.

## What was reused

- The `gpu_jobs/*.job.yaml` + provenance-ledger pattern from R0/R1: a claim is a result
  only when the runner's job wrote the file.
- The easy five-field extraction task from `docs/next-experiments.md`, re-authored here
  as a deterministic fictional generator (`tools/exp/common.py`) so no real data is
  committed.
- The **sales-brain synthetic call corpus** (`~/theprojects/queue/sales-brain/data/synthetic`,
  read at job runtime, never committed) as the hard task cards 1/5/4 are rerun on:
  7 deal fields from Latvian call transcripts, where the 9B is far from perfect.
- The **Ollama-bundled llama-server** (`~/ollama/lib/ollama/llama-server`) for card 4,
  which is the only offline way to get `--model-draft` on the cached 27B GGUF.
- The prior-art check for this idea (`prior_art check`, cache `/tmp/prior-art`) returned
  `already built` on the phrase match but with only weak/indirect hits (top similarity
  0.48), nothing to fork.

## Honesty notes

- The cards ask for models that are not all cached; the scripts substitute the closest
  available ones and say so in each result's `note`/`rule` fields (card 3 has two GGUF
  formats offline, not four).
- Card 4 was re-scoped by the owner (2026-10-03): no Qwen2.5-7B download. It now measures
  the 27B bulk-text bottleneck with a 0.6B draft and, if the vocabularies differ, the
  0 GB `--spec-type ngram-simple` alternative; greedy output identity is the lossless
  check, because llama-server does not report a separate acceptance rate here. The draft
  fetch failed, so the verified card 4 file has only the target-only and ngram arms.
- Cards 1 and 5 hit a 97–100% ceiling on the first easy fixture, which cannot rank the
  arms; both were re-run on the hard sales-brain transcripts and now carry real verdicts.
  Card 2 crashed twice on the same `BatchEncoding` return-type change (n=0 both times);
  its corrected job is re-queued. `docs/results.md` reports every run, wins and failures
  alike.
- `state/exp/*.json` are tracked (so they survive the merge into the main repo) but
  `export-ignore`d, exactly like `state/looped-eval.json`.
