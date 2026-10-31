# Recall miss: why Claude rebuilt a watchdog it already had

**Question:** why did a session on 2026-10-17 propose a new "30-min check-in watchdog" when
`queue-watchdog.timer` and `bin/health.py -> state/alerts.jsonl` already existed — and what
stops the next miss structurally?

**Short answer:** every layer we had was *pull* (an index line, a keyword hook, topic files
read on demand, alerts written to a file nobody reads) and every layer failed for a
different reason. The fix is not a better reminder. It is a **SessionStart push of a live
8-line systems list plus 3 deduplicated open alerts**, a compact re-push, semantic recall
instead of keyword-only recall, and a spec lint that refuses "new tool" work without a
`reuses:` answer from the inventory.

All session facts below are the ones recorded for this item; the transcript file named by
the task was already gone in this snapshot (see [Measurement](#measurement-the-real-case)),
so the replay uses `~/.claude/history.jsonl` read-only.

---

## 1. Causes, evidence, and what we already had

| cause | evidence from literature (link) | did it happen here | what we already have against it | why ours failed | fix |
|---|---|---|---|---|---|
| **Index-only memory.** The index line is loaded; the body that contains the rule is not. | Claude Code loads the first 200 lines / 25 KB of `MEMORY.md`, but **topic files are read on demand**, not at startup: [memory docs](https://code.claude.com/docs/en/memory). | The `MEMORY.md` line for the watchdog *was* in context. The rule "read `state/alerts.jsonl` at the start of every turn" lived only in the memory body, which was never opened. | `MEMORY.md` index + one body per memory. | An index is a pointer, not a trigger. Nothing in the prompt made Claude open the body; after compaction even the pointer was gone. | Push the 8-line systems digest itself at `SessionStart`; keep bodies for detail. |
| **Lexical recall miss / vocabulary mismatch.** The hook searches for the words in the prompt; the user says "bottleneck", the tool is called "watchdog". | BM25 is a strong baseline but lexical matching is brittle across wording; hybrid BM25 + vector retrieval is the robust choice: [BEIR](https://arxiv.org/abs/2104.08663), [XERJ query types](https://xerj.org/docs/queries). Low needle-question similarity degrades recall further as context grows: [Context Rot](https://www.trychroma.com/research/context-rot). | The current hook returned context on 16/58 prompts but named `queue-watchdog`/`health.py` on **0/58**. For the bottleneck question it returned "Queue v2" and no watchdog/health. | `~/.claude/patterns/recall.py` (BM25 over ledger + `COMPLETED.md` + memory), XERJ v1.0.0-rc.96 installed, inventory hybrid search. | The hook needs ≥2 query tokens and a score ≥6.0, its sources exclude `queue/*.yaml`, `bin/*.py` and inventory, and it has no synonym/intent expansion. "bottleneck" has no lexical overlap with "watchdog". | Hybrid recall + an ops-intent expansion. When the prompt is operational, always push the ops core (`queue-watchdog`, `orchestrator-health`) regardless of wording. |
| **Lost in the middle / context rot.** Relevant facts in a long context are used worse than facts at the edges; performance falls as tokens grow. | [Lost in the Middle](https://arxiv.org/abs/2307.03172) (position bias); [Context Rot](https://www.trychroma.com/research/context-rot) (18 models, degradation with input length). | The session logged 58 prompt rows. The watchdog was one line among a large context; after the late-session compaction the summary did not mention it. | Compaction summary, `MEMORY.md`, long context. | The systems list was neither at the start nor at the end of the effective context, and the summary was free-form. | Put the live systems list at the very start of every session and re-inject it after compaction. |
| **Compaction / summary loss.** Compaction is model-selected; aggressive summaries lose context whose importance shows up later. | Anthropic: "overly aggressive compaction can result in the loss of subtle but critical context": [context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). Claude Code re-injects project-root `CLAUDE.md` after `/compact`; conversation-only instructions do not survive: [memory docs](https://code.claude.com/docs/en/memory). | After compaction the summary did not mention the watchdog, the timer, or `health.py`. | The compaction summary itself. | No schema for what a summary must contain, and no post-compaction re-push. The system existed only in conversation. | `SessionStart` matcher `compact` (and `PostCompact` side effect) re-runs the push. Keep the systems block in `CLAUDE.md` too, so it is re-read from disk. |
| **Tool under-use / tool exposure.** An available tool is not called unless it is in the exposed set at the moment of decision. | Tool-rich agents fail when the right tool is recognized but not exposed: [Recognition-Exposure Gap](https://zenodo.org/records/22699321) (52.9% first-attempt reachability). LLMs remain limited at tool use without explicit training/exposure: [ToolLLM](https://arxiv.org/abs/2307.16789). | Inventory search and XERJ existed and were not called. Claude designed a "new 30-min check-in watchdog" instead of running `inventory search "watchdog"`. | `~/theprojects/queue/inventory` hybrid search; XERJ local node. | Neither is triggered by the harness. There is no mandatory "search before you build" step, and the 27B-generated inventory card for `agent-watchdog` describes only v1. | Make the systems list part of the prompt; add a q2 spec lint that forces a `reuses:` answer (fix 3). |
| **Memory staleness.** Stored facts need update/expiry, not just retrieval. | Zep keeps temporal validity on graph edges: [Zep](https://arxiv.org/abs/2501.13956). Mem0 extracts, consolidates and updates: [Mem0](https://arxiv.org/abs/2504.19413). A-MEM evolves notes and links: [A-MEM](https://arxiv.org/abs/2502.12110). | `state/inventory.md` was generated 2026-10-04, four days before `queue-watchdog` was done (2026-10-08). Its `agent-watchdog` card says "not for: not a general-purpose task queue or CI runner" — the opposite of v3 queue mode. `health.py` is not an inventory entry at all. | Inventory, memory `modified` field, queue `COMPLETED.md`. | Inventory is a snapshot, not a projection of current queue state; nothing rescans after a project finishes. | Regenerate inventory cards when a queue item settles; add `verified` timestamps and a stale flag; add `health.py` as a first-class entry. |
| **Alarm fatigue.** Repeated, non-actionable alerts desensitize the reader and hide the new one. | Scoping review of alarm fatigue, definitions/factors/mitigation: [PubMed 40542361](https://pubmed.ncbi.nlm.nih.gov/40542361/). | `state/alerts.jsonl` has **451 rows / 293 unique keys**; one unit key repeats 11 times, another 9. `gpu_idle` repeats 5 times. `health.py` re-writes a key every 6 h while it is still open and never writes a "resolved" event. | `health.py` keyed dedupe, `REMIND_H=6`, hourly Telegram batch, `state/health.json` open keys. | `alerts.jsonl` is append-only history, not an inbox. Nothing projects "open now, newest, max 3". Nothing pushes it into a Claude session. | Read `state/health.json` open keys + latest message per key; cap 3; add resolved events; rotate/compact the log. |
| **Push vs pull.** Pull-only context fails exactly when the user's words do not name the thing. | Anthropic's hybrid: `CLAUDE.md` is dropped in up front, glob/grep is just-in-time: [context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). Cursor's "Apply Intelligently" loads a rule only when the agent decides it is relevant: [Cursor rules](https://cursor.com/docs/rules). Devin retrieves Knowledge only when a trigger description matches: [Devin Knowledge](https://docs.devin.ai/product-guides/knowledge). | The only cross-session mechanism in play was the pull hook. There was **no SessionStart push** of systems or alerts. | `CLAUDE.md`, memory index, recall hook, compaction summary. | All pull. The 8-line list is small enough to push; we made the model ask for it instead. | Push the invariant list; pull only details. This is fix 1 and it is decisive. |
| **No reuse gate at planning time.** "Build a new X" is accepted without proving X is not already in the fleet. | Recognition-Exposure Gap (above) shows the harness, not model intent, decides availability; ToolLLM shows tool use is a trained/exposed capability, not a default. | The session's plan proposed a new watchdog and a new scheduler check; `reuses:` is not a queue-spec field, so nothing forced an inventory lookup. | Inventory search, XERJ, queue specs. | Nothing structural asks "what do we already have?" before a spec is accepted. `done_when` checks output, not reuse. | Add `reuses:` to the spec schema; lint it against inventory (fix 3). |

### Why ours failed, in one paragraph

The memory system was designed as a library: an index, bodies, a search hook, and an alert
file. The session needed a **dashboard**: eight lines that say what is running and three
lines that say what is broken now. The library failed at four separate points —
the index did not trigger a read, the search did not match the wording, compaction dropped
the pointer, and the alerts were never pushed — so no single component was "the bug". The
same shape recurs in the literature: retrieval misses, position bias, summary loss, tool
exposure, stale memory, and alarm fatigue are separate failure modes that all end with the
agent not knowing what the fleet already contains.

**Local tools first.** The inventory search was run before writing this: `inventory search
"watchdog alerts health"` returns `agent-watchdog` first, but its card is the stale v1
description; `inventory search "health check alerts failed units gpu idle"` does not return
`bin/health.py` at all. XERJ v1.0.0-rc.96 is installed and its unit is running, but it has
no index of the fleet and the recall hook never calls it. The runner's prior-art check for
this idea returned only weak/indirect hits, so nothing was rebuilt from it.

A second, quieter cause: **the inventory lied by being old**. `agent-watchdog`'s card was
written before `queue` mode existed and says it is *not* for the task queue. A correct
inventory search would still have pointed at the wrong answer. Freshness is part of recall.

---

## 2. Ranked fixes for this stack

Ranked by "would it have prevented this miss per unit of work". All are structural: a hook,
a check, or a generated projection — not "Claude should remember".

### 1. S — Push a live systems list + open alerts at `SessionStart` and after compaction

**What:** one hook prints 6-8 lines from the user's systemd unit files
(`queue-watchdog.timer`, `orchestrator-health.timer`, `orchestrator-q2.service`,
`gpu-queue.service`, `stallwatch.timer`, `laya-watchdog.service`) and the 3 newest
deduplicated open alerts. Run it on `SessionStart` (`startup|resume|compact`) and as a
`UserPromptSubmit` fallback.
**Files:** new `~/.claude/patterns/systems_push.py` (prototype: `tools/recall_hook.py`);
hook entries in `~/.claude/settings.json`; the live source stays
`~/.config/systemd/user/*.service|*.timer` and `~/theprojects/orchestrator/state/health.json`.
**Why first:** it does not depend on the user's wording. In the replay it is a hit on
1/1 session starts and on the bottleneck prompt; the current pull hook is 0/58 exact.
**Cost:** ~1.2 KB of context per session, well under the 10,000-character hook cap
([hooks docs](https://code.claude.com/docs/en/hooks)).

```json
{
  "hooks": {
    "SessionStart": [{"matcher": "startup|resume|compact",
      "hooks": [{"type": "command", "command": "python3 ~/.claude/patterns/systems_push.py session-start"}]}],
    "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "python3 ~/.claude/patterns/systems_push.py user-prompt"}]}]
  }
}
```

### 2. S — Make `alerts.jsonl` an open-alerts inbox, not a 451-line diary

**What:** project `state/health.json` (open keys) over `alerts.jsonl` (latest message per
key), cap at 3, drop keys that are not open and are older than 6 h, and write a
`{"resolved": true}` line when a key disappears. Add a `--compact` that keeps one row per
key plus resolved markers. The hook reads the projection, never the raw log.
**Files:** `~/theprojects/orchestrator/bin/health.py` (projection + resolved + compact),
`~/.claude/patterns/systems_push.py` (consume it).
**Evidence:** 451 rows, 293 keys, 11 repeats; the 20:32 "GPU idle while 25 GPU task(s) are
ready" was one of five `gpu_idle` rows, so the newest one was not distinguishable from
history. The replay hook picks the 11:17 "GPU idle while 17 GPU task(s) are ready" — 18
minutes before the bottleneck question.

### 3. S/M — Refuse a new tool/spec without `reuses:` filled from inventory

**What:** extend the queue spec schema with a required `reuses:` list. Each entry must
resolve to an inventory name or a `queue/*.yaml` id (or the literal `none — searched:
<terms>`). `load()` rejects a spec whose `reuses:` is missing or unresolvable; the error is
loud and names the closest inventory hit. This turns "search before building" from advice
into a gate.
**Files:** `~/theprojects/orchestrator/bin/projects.py` (`load`, shared by v1 and q2),
`~/theprojects/orchestrator/q2/projects.py` (attempt creation), tests under
`~/theprojects/orchestrator/q2/tests/`.
**Evidence:** this is the Recognition-Exposure Gap applied to planning: expose the existing
tool in the spec template and refuse the spec without an answer. Effort is S for the field
and S/M for the inventory lookup.

### 4. M — Replace keyword-only recall with XERJ/inventory hybrid + ops-intent expansion

**What:** index `queue/*.yaml`, `bin/*.py` docstrings, `COMPLETED.md`, memory bodies and
ledger into XERJ (`/__memory/systems`, BM25 + kNN). The hook asks for top 3; if the prompt
matches the ops vocabulary (`bottleneck|slow|stuck|idle|gpu|queue|watchdog|alert|failed`),
it also adds the invariant ops core. Keep BM25 as the fallback when XERJ is down (it is
currently on `127.0.0.1:9200` with auth; the hook must not depend on a key being present).
**Files:** `~/.claude/patterns/recall.py`, new `~/theprojects/orchestrator/bin/xerj_index.py`,
`~/.config/systemd/user/xerj.service` (already exists).
**Evidence:** BM25 is robust but lexical ([BEIR](https://arxiv.org/abs/2104.08663)); XERJ
gives BM25 + kNN in one store ([agent-memory recipe](https://xerj.org/docs/recipes/agentic-memory)).
**Measured prototype:** a stdlib alias + service-name ranker already takes the bottleneck
prompt from 0/1 to 1/1 without XERJ; XERJ is the durable version.

### 5. M — Make compaction carry the systems block

**What:** keep the generated systems block at `~/.claude/claude-systems.md`, import it from
`~/.claude/CLAUDE.md` (`@~/.claude/claude-systems.md`), and re-run the push on
`SessionStart` matcher `compact`. Add one check to the runner: after a compacted session,
the summary must contain `queue-watchdog` and `health.py`; fail the check if not.
**Files:** `~/.claude/CLAUDE.md`, `~/.claude/settings.json`, `tools/recall_hook.py`
(prototype), `~/theprojects/orchestrator/q2/executors.py` (post-session check).
**Evidence:** Anthropic's own compaction warning; Claude Code docs say project-root
`CLAUDE.md` is re-read after `/compact`, conversation-only instructions are not.

### 6. M — Regenerate inventory cards when a queue item settles

**What:** hook `queue settle` to `python3 -m inventory scan` (or a lighter card refresh) and
add `verified:` timestamps. Mark a card stale after 30 days or after the repo's last commit
is newer than the card. Add `bin/health.py` and `agent_watchdog queue` as explicit entries.
**Files:** `~/theprojects/queue/inventory/inventory/cards.py`, `inventory/__main__.py`,
`~/theprojects/orchestrator/bin/completed.py` or `q2/executors.py`.
**Evidence:** the `agent-watchdog` card is v1-only and says "not for ... task queue"; that
is a false negative for exactly this miss.

### 7. S — Use Laya to route ambiguous ops prompts, not to replace the invariant core

**What:** a `choice` question per prompt: "is this about running/health/queue/gpu, or about
content/research?" If ops, add the core. Laya is ~60 ms on CPU and is already running as
`laya-watchdog.service`; it must never be the only path (if it is down, the regex/core
still runs).
**Files:** `tools/recall_hook.py` / `~/.claude/patterns/systems_push.py`,
`~/theprojects/queue/agent-watchdog`.
**Evidence:** the alias rule is already 1/1 on the real case; Laya only reduces false
positives on ambiguous prompts.

### 8. S — Make `health.py` an inventory entry and a named system

**What:** add a `bin/health.py` tool entry and a `health` queue-system alias so an inventory
search for "alert", "health", "failed unit" or "idle" returns it. Today a search for
`health check alerts failed units gpu idle` returns job-radar, gpu, car-notifier — not
`health.py`.
**Files:** `~/theprojects/queue/inventory/state/inventory.md` (regenerated), `cards.py`,
or a small `inventory/state/cards/health.json`.
**Evidence:** the inventory search run for this report missed it; that is the "local tools
first" check failing at the source.

---

## 3. Measurement: the real case

### Source and method

- The task named `~/.claude/projects/-home-neo/620f5a88-f819-4b1b-a5de-f180bc3ed532.jsonl`.
  In this snapshot `~/.claude/projects/` is empty (the directory was cleaned at 13:04), so
  the replay used `~/.claude/history.jsonl`, filtered by `sessionId`, read-only. That file
  keeps every prompt's text and timestamp for the session: **58 prompts**.
- Current hook: `python3 ~/.claude/patterns/recall.py hook` with each prompt on stdin.
- Proposed hook: `python3 tools/recall_hook.py user-prompt`, a SessionStart push plus a
  per-prompt rank over live unit files, queue specs, `bin/*.py` docstrings, `state/health.json`
  and `state/alerts.jsonl`.
- **Hit** = the output literally names `queue-watchdog`/`agent-watchdog` or
  `health.py`/`orchestrator-health`/`alerts.jsonl`. `watchdog-waste` is a *different* tool
  and is not counted.
- No prompt bodies are printed; only indexes and hit/miss. No personal data is quoted.

Run:

```sh
python3 tools/recall_replay.py --session 620f5a88-f819-4b1b-a5de-f180bc3ed532 \
  --json-out state/recall-replay.json
```

Real output (2026-10-18):

```text
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

For the bottleneck prompt (`#54`, 11:35:55), the current hook returned only "Queue v2 —
built and shadowing v1"; no watchdog, no health, no alerts. The proposed hook returned:

```text
Ops context: existing systems that match this request (reuse, do not rebuild):
- orchestrator-health [5min]: Orchestrator health: alert + safe fixes for failed, hung or idle work (bin/health.py) -> state/alerts.jsonl
- queue-watchdog [5min]: agent-watchdog queue check: rules find stuck queue items and GPU jobs, Laya chooses the action
- gpu-queue: Serial GPU job queue worker (qsubmit)
- laya-watchdog: Laya sidecar for agent-watchdog's queue mode (127.0.0.1:8792, CPU, small typed-decisions model)
Recent open alerts (deduplicated by key; do not repeat these as new):
- GPU idle while 17 GPU task(s) are ready (key: gpu_idle)
- sb-realspeech: repeating_failure — exit -15: stopped from outside twice before it finished ...
- sb-realspeech-check27 frozen: running task sb-realspeech-check27 still holds its slot and its log has not changed for 32 min -> needs_claude
```

### Hit / miss

| case | current hook | proposed hook |
|---|---|---|
| Session start (before any prompt) | **MISS** — no SessionStart hook; only `MEMORY.md`/`CLAUDE.md` | **HIT** — `queue-watchdog` + `orchestrator-health` (health.py) in the first 8 lines |
| Bottleneck question `#54` | **MISS** — returned "Queue v2", not the watchdog or health | **HIT** — both systems + the 11:17 GPU-idle alert (18 min before the question) |
| "you did not see watchdog" `#56`/`#57` | **MISS exact** — returned `watchdog-waste`, a different tool | **HIT** — both systems |
| Whole session, exact named hits | **0/58** (16/58 returned some wrong context) | **29/58** any; 28/58 watchdog, 21/58 health |

The task facts cite the 20:32 `GPU idle while 25 GPU task(s) are ready` line. That is row
408 of `alerts.jsonl`; by 11:17 the same key had a newer message, "GPU idle while 17 GPU
task(s) are ready" (row 445). The proposed hook shows the **newest** one and, because it
deduplicates by key, does not show the older 25-task line as a separate problem. That is
the intended behaviour: one live problem, not five historical copies.

### What this does and does not prove

- It proves the current hook cannot name the system on this session, and that a push-first
  hook does. The SessionStart result is the decisive one: it does not depend on wording.
- It does **not** prove the 29/58 prompt-level rate is all signal. Some prompts are about
  GPU speed and legitimately pull the ops block. The false-positive risk is why the
  invariant core is only two systems and the per-prompt list is capped at four.
- The proposed hook is a **prototype in this repo**, not installed. Nothing under
  `~/.claude` was edited. `tools/recall_hook.py` is the file to copy or point the hook at.

---

## 4. How this maps to the other systems

- **MemGPT/Letta:** core vs archival memory and explicit interrupts ([MemGPT](https://arxiv.org/abs/2310.08560)).
  Our equivalent is the 8-line core (always pushed) vs the inventory/memory bodies (pull).
  We did not need a new memory server; we needed to treat the systems list as core memory.
- **Mem0:** extract/consolidate/update rather than append ([Mem0](https://arxiv.org/abs/2504.19413)).
  The alerts fix is the same move at a smaller scale: consolidate by key, resolve, cap.
- **Zep:** temporal validity ([Zep](https://arxiv.org/abs/2501.13956)). Our alerts and
  inventory cards need `verified`/`resolved` timestamps; today neither has one.
- **A-MEM:** notes link and evolve ([A-MEM](https://arxiv.org/abs/2502.12110)). The queue
  spec's `reuses:` field is the cheap version: force the new note to link to an old one.
- **Reflexion:** keep lessons in an episodic buffer ([Reflexion](https://arxiv.org/abs/2303.11366)).
  The ledger already does this, but a lesson that is not pushed is not reused; the lesson
  here is "push the fleet, not the story".
- **LangGraph:** short-term thread state + long-term namespaced store ([LangGraph memory](https://docs.langchain.com/oss/python/concepts/memory)).
  Same split: thread state is the session; the fleet list is the long-term store.
- **Cursor / Windsurf / Devin:** relevance-gated rules, Cascade memories and trigger-gated
  Knowledge all depend on the gate matching the wording
  ([Cursor rules](https://cursor.com/docs/rules),
  [Windsurf memories](https://docs.windsurf.com/windsurf/cascade/memories),
  [Devin Knowledge](https://docs.devin.ai/product-guides/knowledge)). That is exactly the
  failure mode here; the invariant list must bypass the gate.

---

## 5. Exact next step for the owner (outside this repo)

1. Copy `tools/recall_hook.py` to `~/.claude/patterns/systems_push.py`.
2. Add the two hook entries from fix 1 to `~/.claude/settings.json`.
3. Restart the session (or `/clear`) and check the first screen shows `queue-watchdog` and
   `orchestrator-health` before the first prompt.
4. If that works, add the `reuses:` lint (fix 3); it is the only fix that prevents the next
   "new tool" plan rather than only fixing recall.

No GPU, download or local batch was needed for this report. Web research used ~28
requests (2 multi-query searches, ~26 fetches, several redirected/blocked), inside the
~200 budget.
