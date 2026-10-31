# caveman — terse agent output, read and measured

Research note for `caveman-terse`. Upstream: **[JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman)**
(the canonical repo; `slmingol/caveman`, `amanattar/caveman-claude-skill` and others are
forks/mirrors of it). Read 2026-10-04 from `README.md`, `skills/caveman/SKILL.md`,
`LICENSING.md`, `LICENSE-MIT`. Raw upstream text was **not** copied into this repo;
quotes below are the short prompt lines we need to reuse.

## 1. What it does

A **voice, not a grammar**: a Claude Code / Codex / Gemini CLI skill (and npm/Go
CLI/proxy) that makes an agent answer in terse "caveman" style while keeping every
technical fact. It is a prompt/skill first (a system-prompt fragment plus mode
commands); the rest of the repo is tooling around it (`caveman-commit`,
`caveman-review`, `caveman-compress`, statusline, proxy).

Levels: **lite** (drop filler, keep grammar), **full** (default; drop articles,
fragments), **ultra** (telegraphic), plus a **文言文 (wenyan)** classical-Chinese mode.
Activation is per session (`/caveman`, "talk like caveman"), off on "stop caveman" /
"normal mode".

## 2. Exact prompt text / rules

The always-on snippet the README tells other agents to paste into their system prompt
(verbatim):

```
Terse like caveman. Technical substance exact. Only fluff die.
Drop: articles, filler (just/really/basically), pleasantries, hedging.
Fragments OK. Short synonyms. Code unchanged.
Pattern: [thing] [action] [reason]. [next step].
ACTIVE EVERY RESPONSE. No revert after many turns. No filler drift.
Code/commits/PRs: normal. Off: "stop caveman" / "normal mode".
```

`skills/caveman/SKILL.md` adds the full rule set. The load-bearing rules:

1. **Answer first.** Answer, then reason, then next step.
   Pattern `[thing] [action] [reason]. [next step].`
2. **Kill ceremony.** No greeting, hedging, pleasantries, recap, closer;
   no just/really/basically/actually/simply.
3. **Short word.** "fix" not "implement a solution for". Standard acronyms fine
   (DB, API, HTTP); no invented abbreviations; no arrows.
4. **Articles optional, meaning never.** Fragments fine. Never drop
   not/never/no/only/except. Numbers and units exact.
5. **One idea per sentence.** ASD-STE100 floor: 20 words max, active voice,
   imperative for instructions. Clarity wins over compression.
6. **Payload verbatim.** "Code blocks unchanged. Commands, paths, API names exact.
   Errors quoted exact, shortest decisive line only."
7. **Tool runs: bounded status.** No text between routine calls; one line before a
   multi-step run, one per phase change, one with the result.
8. **User's language.** Compress style, not language; technical terms/errors verbatim.
9. **Never perform caveman.** No "me think", no `Caveman:` prefix, no normal-plus-caveman
   copy, no decorative emoji/tables.

Break the rules (plain prose, then resume): security warning; irreversible action;
step order a fragment could scramble; user confused; **anything persisted outside chat**
(code, comments, commits, docs, issues, PRs, memory files); harness status/confirmation.
Pre-send check: delete a first sentence that announces; delete a last sentence that
recaps; verify every negation and every code span/path/number/error is intact.

## 3. Claimed savings

From upstream `README.md` (2026-10-04):

| Claim | Number |
|---|---|
| Output tokens vs normal verbose Claude | **~75%** (marketing) / **65% average**, range 22–87% |
| `caveman-compress` on memory files (input side) | **46% average** |
| Speed | ~3x faster generation claimed |
| Accuracy | cites arXiv 2604.00025 "Brevity Constraints Reverse Performance Hierarchies" (+26 pp on some benchmarks) |

Upstream caveat, stated explicitly: **caveman affects output tokens only — thinking/
reasoning tokens are untouched.** (Our harness bills/produces both, so that caveat
matters; see our measurement in §5.)

## 4. Licence and risks

**Licence.** Upstream `README.md` badge says MIT, but `LICENSING.md` says: from
**3.0.0** everything in the repo is **Apache-2.0**; pre-3.0.0 MIT parts stay MIT
(`LICENSE-MIT` kept, referenced from `NOTICE`); Copyright 2026 Julius Brussee.
"Caveman" and its logos are trademarks; the licence grants no trademark rights.
Hosted Caveman Cloud is separate commercial software. For us the practical point:
**we may reuse and adapt the prompt under Apache-2.0 with attribution** (we do not
ship the upstream code). The README's "MIT" line is stale versus `LICENSING.md`.

**Risks — what it may break:**
- **Tool-call JSON / arguments.** Rule 6 only covers code blocks and commands; JSON
  tool arguments are not called out. A model told to be terse can drop a JSON field,
  a quote, or a required key. Fix in our adapter: explicitly exempt tool-call
  arguments (see `prompts/caveman-system.txt`).
- **Code / diffs / file contents.** "Code blocks unchanged" is a rule, not a
  guarantee. Any paraphrase inside a fenced block is a silent bug.
- **Test output / errors.** Rule 6 says quote errors exact but "shortest decisive
  line only"; truncating a stack trace or a multi-line failure can hide the cause.
- **Negations.** Rule 4 flags them; a dropped not/never/no is the most expensive
  mistake (upstream says this itself).
- **Step order.** Numbered/ordered instructions can scramble under fragments.
- **Persisted text.** Upstream's own exemption list includes commits/docs/PRs, so
  caveman must not leak into artifacts.
- **Token savings are baseline-dependent.** The 65% is versus *verbose* output. Our
  harness already enforces terse output, so the incremental win is smaller (§5).
- **Output-only.** No saving on input/reasoning; compression for input is a separate
  tool (`caveman-compress`).

## 5. Our measurement (offline, deterministic, no GPU)

`prompts/caveman-system.txt` is our harness adapter (scope: prose + reasoning only;
tool-call arguments, code, file contents stay exact). To size the win on **our** data
instead of upstream's benchmark:

- Source turns: **40 real assistant messages** from the saved orchestrator transcripts
  (`~/theprojects/orchestrator/state/trajectories`, read-only). Deterministic selection:
  `tools/caveman/extract_turns.py` (prose 120–1100 chars, structured-JSON messages
  excluded, round-robin over sessions for diversity, ≥30 required).
- Rewriting: DeepSeek rewrote **only the prose segments**; fenced code blocks were
  passed through byte-exact. Rewrites are in `tools/caveman/rewrites.json` (real names
  sanitised out).
- Counting: `tools/caveman/measure.py` tokenized before/after with the **Qwen3
  tokenizer** (`Qwen/Qwen3-1.7B` snapshot from the local HF cache), no network.
- Result: `state/caveman-savings.json`.

| n_turns | tokens_before | tokens_after | saving_pct |
|---:|---:|---:|---:|
| **40** | **3768** | **3323** | **11.81%** |

Reading: on top of our harness's **already-terse** output style, caveman-style prose
removes a further **11.81%** of Qwen3 tokens across these 40 turns. That is well below
upstream's 65%, which is measured against verbose Claude output — i.e. most of the
caveman win is already banked by our existing no-preamble/no-recap rules. The
remaining token mass is payload (identifiers, paths, numbers, error text) that must stay
verbatim, so the headroom is small. The on-GPU accuracy/correctness test belongs to the
launch kit, not here.

Reproduce (needs a Python with `transformers` and the local Qwen3 snapshot):

```bash
python3 tools/caveman/extract_turns.py --limit 40          # writes state/caveman-turns.raw.json (gitignored)
~/theprojects/gpu/.venv/bin/python tools/caveman/measure.py
python3 -c "import json;r=json.load(open('state/caveman-savings.json'));print(r['n_turns'],r['saving_pct'])"
```

## 6. Reused / adapted

- Reused **the prompt rules** from `JuliusBrussee/caveman` `SKILL.md` and the README
  always-on snippet; adapted for this harness in `prompts/caveman-system.txt`
  (Apache-2.0, attribution here). No upstream code vendored.
- Built on the existing orchestrator transcripts instead of new data collection.
- `tools/caveman/*` is new here; no prior token-savings measurement exists in this repo.
- A prior-art check over public sources (`prior_art check`, 2026-10-04) found no existing
  caveman/token-savings implementation to fork; the nearest hits were generic "save
  tokens" posts, so the measurement here is built from scratch on our transcripts.
