# MicroTurn Duplex LoRA (`codacus/Qwen3.6-35B-A3B-MicroTurn-Duplex-LoRA`) — research note

**Status:** research only, 2026-10-19. No adapter or GGUF was downloaded, no
third-party code was installed or run. The only code read is the public client in
the model repo and the llama.cpp source already checked out on this machine.

**One-line answer.** It is a **15.3 MB GGUF LoRA** for
`Qwen/Qwen3.6-35B-A3B` that teaches a text LLM one narrow behaviour: every
**0.6 s** read the latest speech-to-text fragment and either say **up to three
words** or emit `<listen>`. Licence is **Apache-2.0**; the repo contains **no
pickle, no `.bin`, no `.safetensors`, no `trust_remote_code`, no install script,
no telemetry** — the only executable is a 3.4 KB stdlib client
([S2](#sources), [S4](#sources)). It is **English-only** ([S2](#sources)) and it
solves **turn-taking, not sales cueing**. Our pinned llama.cpp already supports
its architecture and `--lora`, and our cached plain Q4_K_M base measured
**20.95 GB / 24 GB at 8K context** ([S10](#sources)); the adapter adds 15 MB, so
the load should fit with roughly 3 GB to spare. The one real unknown is that it
was trained on **Unsloth's UD-Q4_K_M (22.13 GB)**, not the ggml-org Q4_K_M
(20.42 GB) we have, and the card calls other quants untested
([S2](#sources), [S3](#sources)). Cheapest decisive test: **15.3 MB down, one
server flag, the 18-input demo, under 10 GPU-minutes** (§6).

---

## 1. What it is

| | |
|---|---|
| Author | `codacus` on Hugging Face, the account behind the Codacus YouTube video; no real name on the page ([S1](#sources), [S6](#sources)) |
| Published | repo created **2026-10-12 12:15 UTC**, last modified **2026-10-12 20:13 UTC**; the video is dated **2026-10-13** ([S1](#sources), [S6](#sources)) |
| Licence | **Apache-2.0** (model card and API tag), the same as the base model; the published dataset is also Apache-2.0 ([S1](#sources), [S2](#sources), [S5](#sources), [S7](#sources)) |
| Base model | `Qwen/Qwen3.6-35B-A3B`; current revision `995ad96eacd98c81ed38be0c5b274b04031597b0` (2026-04-24). The adapter pins no revision ([S1](#sources), [S7](#sources)) |
| Training quant | `unsloth/Qwen3.6-35B-A3B-GGUF`, file `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, **22,134,528,992 B = 22.13 GB**; repo revision `a483e9e6cbd595906af30beda3187c2663a1118c` ([S2](#sources), [S3](#sources)) |
| Adapter revision | `f423b2991495a8cb88e3577d6d9515e37d651828` ([S1](#sources)) |
| Format | **GGUF LoRA** (llama.cpp adapter), **not** PEFT and **not** safetensors. The repo has no `adapter_config.json`, no `config.json`, no tokenizer ([S1](#sources), [S2](#sources)) |
| Size / params | **15,339,200 B (15.3 MB)**, **3,833,856 parameters**, **F32**, **rank 32**, **alpha 64** ([S1](#sources), [S2](#sources)) |
| Target modules | the **last four layers (36–39)**: their **attention, delta-net and shared-expert projections**. The routed MoE experts and the router are **not** trained ([S2](#sources)) |
| Downloads / likes | **440 downloads, 49 likes** at 2026-10-19 10:05 UTC ([S1](#sources)) |
| Repo contents | adapter + a 3.4 KB client + the full synthetic training data (`dataset/`), no training code ([S1](#sources), [S2](#sources)) |

**Format arithmetic (verified):** 3,833,856 params × 4 bytes = 15,335,424 B;
the file is 15,339,200 B, i.e. 3,776 B of GGUF header/metadata. That is
consistent with the card's "F32, rank 32, alpha 64" and shows there is no hidden
second blob in the repo. The HF server-side GGUF metadata reports architecture
`qwen35moe`, 3,833,856 total parameters, 15,339,200 B ([S1](#sources)).

**Files and exact sizes** ([S1](#sources)):

| File | Bytes | What |
|---|---:|---|
| `microturn-duplex-lora.gguf` | 15,339,200 | the adapter (LFS) |
| `README.md` | 9,349 | model card |
| `microturn_client.py` | 3,444 | stdlib text client |
| `dataset/calls_train.jsonl` | 29,437,624 | 539 training calls (LFS) |
| `dataset/calls_heldout.jsonl` | 1,631,305 | 28 held-out calls |
| `dataset/dialogues_long.jsonl` | 1,096,604 | 245 long dialogues |
| `dataset/dialogues_short.jsonl` | 621,036 | 295 short dialogues |
| `dataset/README.md` | 10,668 | dataset card |
| `.gitattributes` | 1,644 | LFS rules |

The card's target-module sentence matches the base architecture: the base is
**40 layers** laid out as **10 × (3 Gated DeltaNet → MoE) → (1 Gated Attention →
MoE)**; layers 36–38 are the delta-net layers and 39 is the full-attention layer
([S7](#sources)). Exact per-tensor names are inside the GGUF and were **not**
read (the goal forbids downloading it); the runtime builder's LoRA points are in
§4.1.

---

## 2. How it works

### 2.1 The turn protocol

- **Tick = 0.6 s.** There is **no VAD** and no end-of-utterance detector; the
  model itself decides when to speak ([S2](#sources), [S6](#sources)).
- **Input per tick:** one user message containing the transcript fragment that
  streaming STT produced in the last 0.6 s, or the literal tag `<no voice>` if
  nothing was heard ([S2](#sources), [S4](#sources)).
- **Output per tick:** **up to three new words** to speak, or the literal tag
  `<listen>` to add nothing ([S2](#sources)).
- **History is append-only:** each tick adds one user message and one assistant
  message; nothing already sent is edited. That is deliberate so llama.cpp's
  prompt cache resumes on every request; the video says merging/editing history
  made a 380 ms turn take 617 ms ([S2](#sources), [S6](#sources)).
- **Two tags at inference:** `<listen>` and `<no voice>`. The 13 `tag` values in
  the dataset (`listen_speech`, `speak`, `start`, `hazard_start`, …) are **loss
  weights/labels**, not inputs the model sees ([S5](#sources)).

**Exact system prompt** (byte-for-byte; the client builds it from `TICK = 0.6`,
`LISTEN = "<listen>"`, `NOVOICE = "<no voice>"`; I recomputed it from
[S4](#sources) and it matches the card [S2](#sources)):

```text
You are a voice assistant in a live phone call; both sides can talk at the same time. Every 0.6 seconds you get what the microphone heard in that moment, as a streaming transcript fragment, or <no voice>. You reply with up to three new words to say next, or <listen> to add nothing.
```

**Chat template:** raw ChatML text sent to **`/completion`**, not
`/chat/completions` ([S2](#sources), [S4](#sources)):

```text
<|im_start|>system
{the prompt above}<|im_end|>
<|im_start|>user
Hey I'm cleaning<|im_end|>
<|im_start|>assistant
<listen><|im_end|>
<|im_start|>user
<no voice><|im_end|>
<|im_start|>assistant
```

The change from the base model's template is not a new token scheme: it is the
**absence** of the template's generation prefix. The base tokenizer's ChatML
template appends `<|im_start|>assistant\n` and then `<think>\n` (or an empty
`<think>\n\n</think>\n\n`) when `enable_thinking` is false; the adapter's client
sends `<|im_start|>assistant\n` and stops there, with no thinking block in past
assistant turns ([S7](#sources), [S4](#sources)). Anything that re-inserts
`<think>` (e.g. `/chat/completions`) breaks the trained format.

**Speak-or-listen rule** (this is the part worth stealing):

- Decode at temperature 0, `stop: ["<|im_end|>"]`, `n_predict: 64`,
  `n_probs: 8`, `cache_prompt: true` ([S4](#sources)).
- Read the probability mass of the **first token starting with `<`** — the
  client sums `exp(logprob)` over top tokens whose string starts with `<`. This
  is the card's "P(listen)".
- If the first generated token starts with `<` **and** P ≥ **0.3**, accept
  `<listen>`. If P < 0.3, **ban every `<` token id** with `logit_bias` and
  regenerate, forcing the model to speak ([S2](#sources), [S4](#sources)).
- The card explains why: `<` is one token while "keep talking" is spread over
  many words, so the model can stop mid-sentence at 27 % for `<` even when 73 %
  of the mass wants to continue; 0.3 was the setting where both "stops
  mid-sentence" and "talks over the user" went away ([S2](#sources),
  [S6](#sources)). Note this is **P(first token `<`), not P(exact
  `<listen>`)**: `<no voice>` and any other tag share the prefix. The client's
  code makes that explicit ([S4](#sources)).

**History cost (not mentioned in the card).** One user+assistant pair per 0.6 s
is ~100 pairs/minute. With a ~15–20-token pair, a 30-minute call is ~45–60K
tokens. The card's example serves `-c 32768`; that is roughly 16–20 minutes
before the client hits the context. A long sales call needs `-c 65536` (+~0.7 GB
KV, still inside 24 GB per §4.2) or windowing. Windowing conflicts with the
append-only cache trick. This is a design limit to test before any real call.

### 2.2 How the training data was made — **all claimed** ([S2](#sources), [S5](#sources))

- **Synthetic text, no recordings of real people.** Short dialogues were written
  by **Qwen3.8-Flash**, long ones by the **base Qwen3.6-35B-A3B**; a timing
  script cut them into 0.6 s ticks and imitated a streaming ASR ([S5](#sources)).
- **No TTS in the data.** The only speech model named is the STT whose chunking
  was imitated: **Kyutai `stt-1b`**; the video's live demo uses
  **`stt-1b-en_fr`**. The card says "any text-to-speech" behind the loop and
  does not name the demo TTS ([S2](#sources), [S6](#sources)).
- **Size:** 567 published calls = **539 train + 28 held out**;
  **404,107 ticks**, about **67 hours** of simulated call time; **295 short +
  245 long dialogues** ([S5](#sources)).
- **Tick mix:** **76.7 %** of ticks are `<listen>`; the largest single kind is
  `listen_speech` **38.9 %** ("user is talking, stay silent"); `speak` is
  71,803 ticks at weight 2.0; `start` is 8,176 ticks at weight 6.0; three ticks
  in four are silence, so without loss weights the model learns to stay silent
  always ([S5](#sources)).
- **Timing script:** user pace 2.2–3.2 five-letter words/s, words reach the model
  ~0.5 s after being spoken, long words sometimes split across ticks,
  punctuation sometimes late, letters dropped; assistant speaks three words per
  tick; 330 of 567 calls open with one of six fixed greetings; fillers and
  mid-thought pauses are inserted by rule ([S5](#sources)).
- **Published copy is cleaned, not the training set:** 10 dialogues and the 25
  training calls built from them were removed (~5 % of ticks) because a reply
  approved something dangerous or leaked the writing model's instructions. The
  adapter was trained **before** the audit, so it saw those 25 calls
  ([S2](#sources), [S5](#sources)).
- **Known bad content is documented, not removed:** 9 dialogues where a reply
  approves/suggests something unsafe; 4 e-bike overnight-charging contradictions;
  4 medical/legal errors; 10 user/assistant line swaps; missing replies; runaway
  turns; leaked instructions; factual errors. The card says the list is not
  complete ([S5](#sources)).

### 2.3 Training recipe and hardware — **all claimed** ([S2](#sources), [S6](#sources))

- **Frozen lower 36 layers.** Their outputs for each training call were computed
  once with llama.cpp and cached to disk. Only **layers 36–39** were rebuilt in
  PyTorch from the same GGUF and trained. The reimplementation was checked
  against llama.cpp's layer-39 output at **cosine 0.9996**.
- **LoRA:** rank 32, alpha 64, across attention, delta-net and shared-expert
  projections; **3.83 M parameters**. The LoRA itself was trained on layers
  36–39 only.
- **Run:** **282 steps of 4 calls**, about **39 s/step**, **AdamW at 2e-4**,
  first **8,000 tokens** of each call; **about 3 hours on two RTX 3060 12 GB**
  cards. The video also says early runs were ~90 min each.
- **Hardware:** 2 × RTX 3060 12 GB (24 GB total VRAM), the same total as one
  3090. The 35B base was served by llama.cpp with `--n-cpu-moe` on the 3060s;
  the video claims each turn took ~240 ms on one 3060 ([S2](#sources),
  [S6](#sources)).
- **No training code is published.** The repo has the adapter, the data and the
  client only; the cached-activation trainer is described in prose, not
  released. Reproducing the recipe (e.g. for Latvian) means rebuilding that
  trainer, not running a supplied script ([S1](#sources), [S2](#sources)).

### 2.4 Claimed vs verified

| Claim | Number | Status |
|---|---|---|
| Adapter size / params / dtype | 15.3 MB, 3.83 M, F32 | **verified** by API + arithmetic ([S1](#sources)) |
| Rank / alpha / target families | 32 / 64 / attention+delta-net+shared expert, last 4 layers | **claimed** by card ([S2](#sources)); consistent with 3.83 M params |
| System prompt / tags / threshold | exact string, `<listen>`, `<no voice>`, 0.3 | **verified** in client + card ([S2](#sources), [S4](#sources)) |
| Training time / steps / LR / tokens | ~3 h, 282 × 4, 2e-4, 8K tokens/call | **claimed** ([S2](#sources)) |
| Data size | 567 calls, 404,107 ticks, 67 h | **claimed** ([S5](#sources)); card counts, data not read |
| Behaviour results | 78 %/62 % start timing, 6/10 hazards, 35 %/33 % fillers, 8 %/5 % talk-over, 1 %/0 % mid-sentence stops | **claimed** (closed-loop replay, [S2](#sources)) |
| Per-tick latency | ~230 ms median on one 3060 | **claimed** ([S2](#sources)); our 3090 decode is 157 tok/s measured ([S10](#sources)) |
| English only | `language: en`, card text | **verified** ([S1](#sources), [S2](#sources)) |

---

## 3. Vetting

**File-level.** The repo contains only: `.gitattributes`, `README.md`, four
`.jsonl` dataset files, one `.gguf`, one `.py` client ([S1](#sources)). There is
**no** `.bin`, `.pt`, `.pth`, `.pkl`, `.pickle`, `.safetensors`, no
`adapter_config.json`, no `config.json`, no tokenizer, no `requirements.txt`, no
`.sh`, no notebook ([S1](#sources)). The HF API's `config` field is `{}` ([S1](#sources)).
So the classic pickle/safetensors code-execution question does not apply: the
adapter is a **GGUF tensor container**, loaded by llama.cpp's parser, not by
Python unpickling.

**Client audit.** `microturn_client.py` imports only `json`, `math`, `sys`,
`urllib.request`; it has no `os`, `subprocess`, `eval`, `exec`, `pickle`, no
network call except the URL you pass (default `http://127.0.0.1:8080`), no
telemetry, no file writes, no downloads ([S4](#sources)). It is 3.4 KB and the
logic is ~10 lines of probability/threshold code ([S4](#sources)).

**`trust_remote_code`.** Not applicable and not needed: there is no transformers
repo and no Python model code. Loading is `llama-server --lora`, i.e. our build
reads the binary ([S2](#sources)). Upstream `convert_lora_to_gguf.py` has a
`--trust-remote-code` option, but it is only for loading an HF base config and
is not required for a GGUF adapter ([S9](#sources)).

**Install scripts / binaries.** None in the repo. The only binary is the GGUF
itself; the only execution paths are (a) llama.cpp parsing a GGUF and (b) a
stdlib client talking HTTP ([S1](#sources), [S4](#sources)). No downloader, no
post-install hook, no `curl | sh`.

**Licence.** Apache-2.0 on the adapter, the dataset and the base model; the base
card links the Apache-2.0 `LICENSE` ([S1](#sources), [S2](#sources),
[S7](#sources)). No non-commercial clause, no field-of-use limit, no added
acceptable-use terms in this repo. Apache-2.0 permits commercial use with the
usual notice/attribution and patent terms. The linked DuplexCascade code is MIT,
but **it is not shipped in or needed by the adapter** ([S8](#sources)).

**Data safety (the real caveat).** The dataset is synthetic but not safe
content: the card documents replies that approve dangerous plans, medical/legal
errors, and leaked writing-model instructions, and says the list is incomplete
([S5](#sources)). Downloading the **adapter only** avoids this; downloading
`dataset/` or training on it needs filtering. The model itself is a research
demo; the card explicitly says not to rely on it for safety, medical or legal
advice ([S2](#sources)).

**Verdict: safe to download the adapter for evaluation** — no pickle, no remote
code, no install script, no telemetry, Apache-2.0. Residual risks: (a) the GGUF
is an unread binary (15.3 MB; we have not inspected its header, per the cost
gate), parsed by the same llama.cpp GGUF path as every other model; (b) the card
is explicit that other quantisations are untested; (c) the dataset is unsafe
training material, not a runtime risk.

---

## 4. Fit for us

### 4.1 llama.cpp: our pinned build can load it, no conversion needed

Our build is llama.cpp commit `f3363593f457169cdfbdfe22ea399b1cc139ba18`
(2026-10-16), the one in
`~/theprojects/queue/.worktrees/moe-offload/moe-offload/vendor/llama.cpp`, built with CUDA
([S9](#sources), [S10](#sources)). It has:

- `LLM_ARCH_QWEN35MOE` = `"qwen35moe"` (`src/llama-arch.cpp:42`), instantiated at
  `src/llama-model.cpp:334` ([S9](#sources));
- `--lora FNAME` (`common/arg.cpp:2969`) and the GGUF adapter loader, which
  requires `adapter.type == "lora"` (`src/llama-adapter.cpp:216–218`) and logs
  `loaded N tensors from lora file` (`:426`) ([S9](#sources));
- the Qwen3.5/3.6-MoE graph builder applies LoRA with `build_lora_mm` on
  `wqkv`, `wqkv_gate`, `ssm_beta`, `ssm_alpha`, `ssm_out` and `wo`
  (`src/models/qwen35moe.cpp:243,255,259,351,380,387,483`); the shared expert
  goes through the generic `build_ffn` at `:515–519`, whose up/gate/down use
  `build_lora_mm` (`src/llama-graph.cpp:1909+`); routed experts use
  `build_lora_mm_id` (`src/llama-graph.cpp:2321,2340,2353,2454`) ([S9](#sources)).

So the adapter is **already in the right format**: no `convert_lora_to_gguf`, no
merge, just `--lora microturn-duplex-lora.gguf`. Both files come from the same
`qwen35moe` converter and the adapter is expected to use the same `blk.N.*`
tensor names, so name matching should work; the untested variables are the base
**quantisation** (§4.4) and whether this adapter's build is byte-compatible with
our pinned loader (the GGUF adapter format is stable, but we have not read the
file).

**Does `convert_lora_to_gguf` support LoRA on Qwen3-MoE experts?** In code, yes
in principle. The converter stacks per-expert A/B tensors with
`torch.stack` (`conversion/qwen.py:120–145` and the `LoraTorchTensor`
`__torch_function__` handler at `convert_lora_to_gguf.py:237–245`); the
`Qwen3_5MoeTextModel` (`conversion/qwen.py:740–744`) inherits the
`Qwen3NextModel → Qwen2MoeModel` path, and the runtime applies 3-D expert LoRA
with `build_lora_mm_id` (`src/llama-graph.cpp:1655–1701`) ([S9](#sources)). I
would call it **supported by the code, not exercised by a test**: this adapter
does **not** target routed experts or the router anyway, so it is moot for this
download ([S2](#sources)). It matters only if we later train our own expert
LoRA.

**If a merge were required (it is not):** the BF16 base is two shards,
**49,913,715,424 + 19,462,922,752 = 69,376,638,176 B ≈ 69.38 GB** ([S3](#sources)).
A naive merge loads ~69 GB of weights and writes another ~69 GB, so **~140 GB
of disk/RAM peak**; even a streaming merge needs ~70 GB of storage and several
GB of RAM. This box has **30 GB RAM and 24 GB VRAM**, so a bf16 merge is **not
possible**. The only in-place option, `llama-export-lora` into the Q4_K_M GGUF,
needs ~20.4 GB in + ~20.4 GB out (~41 GB free disk) and re-quantises the merged
weights, which is lossy and unnecessary while runtime `--lora` works
([S9](#sources)). Recommendation: **do not merge**.

### 4.2 VRAM on the 3090

Measured with this exact GGUF and build, 8K context, all experts on the GPU
(M1): **20.95 GB peak VRAM**, 157.16 tok/s decode, 3,772 tok/s prefill
([S10](#sources)). On a 24 GB card that leaves roughly **3 GB** by the same
units. The adapter adds **15 MB** of F32 weights plus small graph buffers, so
8K should fit. At **32K** the KV cache grows from ~168 MB to ~0.67 GB (10
full-attention layers of the 40; the rest are constant-size delta-net state),
i.e. **+~0.5 GB**, putting the server around **21.5 GB** — still inside 24 GB
([S10](#sources)). If that is too close, the card's own escape hatch is
`--n-cpu-moe 24` ([S2](#sources)); our measured all-experts-in-RAM mode (M2)
uses only **4.19 GB VRAM** at **45.48 tok/s** decode ([S10](#sources)). At
157 tok/s a three-word tick is tens of milliseconds of decode; the card's
~230 ms/tick on a 3060 includes its offload and the full loop
([S2](#sources)).

**Latency budget:** 600 ms per tick minus STT fragment time, prefill of the new
fragment, decode of ≤3 words, and HTTP overhead. Our measured decode says the
LLM part is not the bottleneck; the card's own weakness is the **first word
after the user stops (~3 s)**, which it attributes to the STT/TTS loop, not
decode ([S2](#sources)). On a 3090 the LLM side should be well inside the tick;
the loop's real cost is the streaming STT.

### 4.3 Language: English-only, and Latvian is a retrain, not a prompt

The card declares `language: en` and says **"English only"**; the training text
is English synthetic dialogue with English ASR-style fragments ([S1](#sources),
[S2](#sources)). The base model is multilingual — its card reports
`SWE-bench Multilingual` and a 248,320-token vocabulary, though it does not
enumerate Latvian — but that does not transfer the *timing* behaviour
([S7](#sources)).

What Latvian would take, concretely:

1. **A Latvian streaming STT that emits fragments every ~0.6 s.** Sales-brain
   already has `whisper_lv` (whisper large-v3-turbo + an LV adapter), but its
   real 600 ms slices cost seconds per decode on CPU
   (`docs/VOICE_LATENCY.md`: p50 107 s on the loaded box) ([S11](#sources)).
   On a GPU with faster partials this changes; the adapter does not fix it.
2. **Latvian synthetic call data** in the same schema: dialogues (the base/Qwen
   can write them), a timing script that imitates the Latvian STT's chunking,
   and the 13 tick tags. Nothing here is published in Latvian.
3. **A retrain of the same 4-layer LoRA.** The recipe is described but the
   trainer is **not published** ([S1](#sources), [S2](#sources)); we would
   rebuild the activation-cache + 4-layer trainer. The claimed cost is ~3 h on
   2×3060 (24 GB total), which is within one 3090's VRAM class, but expect
   several GPU-hours plus a day of data work, and the tokenizer makes Latvian
   words 2–4× more tokens, so "three words" becomes more decode tokens and the
   P(`<`) calibration must be re-tuned.
4. **Evaluation on real Latvian calls**, not synthetic clips (the owner rule in
   the sales-brain spec) ([S12](#sources)).

Until (1)–(4) exist, the adapter is an **English demo**, not a Latvian
component.

### 4.4 The quant mismatch is the one unknown to test

The card trained against **Unsloth UD-Q4_K_M (22.13 GB)** and says "other
quantisations are untested"; we have the **ggml-org plain Q4_K_M (20.42 GB)**
([S2](#sources), [S3](#sources)). Both are `qwen35moe` with the same tensor
names, and the LoRA is F32 applied at runtime, so it should load. But the base's
last-four-layer activations differ between quant recipes, and the adapter's
decision boundary is a probability threshold — exactly the kind of thing a quant
change can move. This is why §6's smoke test uses **our** quant, not the card's.
If it fails, the fallback is the 22.13 GB UD file; at 8K that is ~22.6 GB with
KV and buffers, i.e. tight but plausible with `--n-cpu-moe` for a few layers
([S3](#sources), [S10](#sources)). That is a second 22 GB download and should
be a separate decision.

---

## 5. Use cases

### (a) Sales-brain live cues while the prospect is still talking — **skip**

**Why:** this adapter is a **turn-taking policy for the assistant**, not a cue
classifier. It outputs "say these three words" or "stay silent"; it has no
objection/price/timing classes and no notion of a sales cue. It is
**English-only**, while sales-brain calls are Latvian. More importantly, the
600 ms idea has **already been measured** in `sb-voice-latency`
(`docs/VOICE_LATENCY.md`): with the existing extractor, text-side micro-turns
cut p50 cue latency to 600 ms but produced **316 false cues out of 681 early
cues (46.4 %)**; the real CPU replay was **107 s p50 / 176 s p95** because every
600 ms slice is a full Whisper decode. The doc's verdict is to keep micro-turn
off by default and, if a fast partial source appears, require two agreeing
slices before a cue ([S11](#sources), [S12](#sources)). The adapter would not
change either number: it does not make Whisper faster and it does not make a
fragment's cue more stable. The current path is **1.5 s p95 after the prospect
stops** ([S13](#sources)); the fix there is streaming partials + a stability
rule, not this LoRA.

### (b) The A3B local lane — **skip for the lane as it exists; try only as a 15 MB voice demo**

**Why:** the A3B lane is a **text** lane (coding, tool dispatch, agentic
fixtures) and already serves this exact Q4_K_M base with llama.cpp
([S14](#sources)). The adapter's output contract is **three words or
`<listen>`**, so putting it on the lane's serving default would truncate coding
answers and tool calls. It is orthogonal to pass@1, tool-name accuracy and
tok/s. If we ever add a voice front-end, the adapter is a **15.3 MB, one-flag
experiment** on the same base — that is a cheap standalone try, not a lane
change.

### (c) The protocol idea alone (tags + 600 ms slices) on our own models — **borrow the idea only**

**Why:** the portable parts are (i) append-only one-user/one-assistant-message
history so the prompt cache resumes, (ii) an explicit `<listen>` tag as the
"do nothing" action, and (iii) the **P(first token `<`) ≥ 0.3 / ban-`<`-below**
trick to stop mid-sentence dropouts. Those apply to any model and cost no
download. The **600 ms cadence has already lost on this box** for sales cues:
46.4 % false cues and CPU STT that cannot keep up ([S11](#sources)). So borrow
the tag/threshold/history pattern; do not re-run the cadence against the
existing extractor unless the STT partials are GPU-fast or provider-fast, and
then only with a two-slice stability rule.

---

## 6. Next step for Claude to approve

**The smallest test that proves or kills it on our hardware: load the adapter on
our cached plain Q4_K_M base and replay the card's demo (18 client inputs;
the card prints 17 output lines).**

- **Download:** the **15,339,200 B (15.3 MB)** `microturn-duplex-lora.gguf`
  only. Do **not** pull `dataset/` (29.4 MB) and do **not** pull the 22.13 GB
  UD base. Total new disk: **~15.3 MB**.
- **GPU time:** one llama-server start (20.42 GB model load, ~20–60 s) plus 18
  inputs; budget **10 GPU-minutes**, **22 GB VRAM**.
- **Commands** (for the runner/operator; **not queued by this session** — this
  is a research report, the cost gate is Claude's):

```bash
# 1. adapter only
hf download codacus/Qwen3.6-35B-A3B-MicroTurn-Duplex-LoRA \
  microturn-duplex-lora.gguf microturn_client.py \
  --local-dir models/microturn

# 2. server on the already-cached base (ggml-org Q4_K_M)
BIN=~/theprojects/queue/.worktrees/moe-offload/moe-offload/vendor/llama.cpp/build/bin/llama-server
BASE=~/.cache/huggingface/hub/models--ggml-org--Qwen3.6-35B-A3B-GGUF/snapshots/*/Qwen3.6-35B-A3B-Q4_K_M.gguf
"$BIN" -m "$BASE" --lora models/microturn/microturn-duplex-lora.gguf \
  -ngl 99 -c 8192 -ub 512 --port 8080

# 3. demo, 18 inputs
python3 models/microturn/microturn_client.py http://127.0.0.1:8080 --demo
```

- **Pass criteria (all four):**
  1. server logs `loaded N tensors from lora file` and stays up;
  2. all 18 demo inputs return, no crash;
  3. the P(listen) trajectory is qualitatively the card's: high (≥0.5) on the
     ten speech ticks, then dropping to ≤0.1 on the first `<no voice>` tick
     after "gets extra strong" (card: 0.37 → 0.09), i.e. the adapter still
     controls the threshold on our quant;
  4. median tick wall time **< 600 ms**.
- **Kill criteria:** load error / tensor mismatch; P(listen) flat at ~1.0 or
  ~0; tick > 2 s. If it fails only on (3), the quant mismatch is the likely
  cause; the next decision is the 22.13 GB UD file, **not** a merge.
- **If A passes, test B (optional, +1.6 MB + ~30 GPU-min):** replay the 28
  held-out calls through the client and compare 2–3 rows of the card's results
  table (start timing, hazard interrupts) with a ±10-point tolerance. Only then
  is there evidence for the *behaviour*, not just the load.
- **Do not** start the Latvian retrain or the sales-brain integration from this
  test. Both are separate decisions; the Latvian work needs the unpublished
  trainer rebuilt and a Latvian streaming STT.

---

## Sources

| Label | Source |
|---|---|
| S1 | [HF API model metadata](https://huggingface.co/api/models/codacus/Qwen3.6-35B-A3B-MicroTurn-Duplex-LoRA) and [file tree with sizes](https://huggingface.co/api/models/codacus/Qwen3.6-35B-A3B-MicroTurn-Duplex-LoRA/tree/main?recursive=true), fetched 2026-10-19 10:05 UTC |
| S2 | [Model card / README](https://huggingface.co/codacus/Qwen3.6-35B-A3B-MicroTurn-Duplex-LoRA/raw/main/README.md) |
| S3 | [Unsloth GGUF repo](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) file tree (`UD-Q4_K_M` 22,134,528,992 B; BF16 shards) |
| S4 | [microturn_client.py](https://huggingface.co/codacus/Qwen3.6-35B-A3B-MicroTurn-Duplex-LoRA/raw/main/microturn_client.py) |
| S5 | [Dataset card](https://huggingface.co/codacus/Qwen3.6-35B-A3B-MicroTurn-Duplex-LoRA/raw/main/dataset/README.md) |
| S6 | [The video](https://www.youtube.com/watch?v=shJLycXjhVY); local captions `data/codacus/shJLycXjhVY.txt` (the per-video extraction JSON was untracked in session 2 as generated data) |
| S7 | [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and `config.json` / `tokenizer_config.json` at revision `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| S8 | [DuplexCascade paper](https://arxiv.org/abs/2603.09180) and [code](https://github.com/sbintuitions/DuplexCascade) (MIT); the adapter does not ship this code |
| S9 | llama.cpp at `f3363593f457169cdfbdfe22ea399b1cc139ba18` in `~/theprojects/queue/.worktrees/moe-offload/moe-offload/vendor/llama.cpp`: `src/llama-arch.cpp:42`, `src/llama-model.cpp:334`, `common/arg.cpp:2969`, `src/llama-adapter.cpp:151–426`, `src/models/qwen35moe.cpp:243–527`, `src/llama-graph.cpp:1655–1701,2321–2454`, `convert_lora_to_gguf.py:237–245,272,410–520`, `conversion/qwen.py:93–145,740–744` |
| S10 | `~/theprojects/queue/.worktrees/moe-offload/moe-offload/state/moe/M1.json` and `M2.json` (measured 2026-10-17: 20.95 GB / 157.16 tok/s; 4.19 GB / 45.48 tok/s at 8K) and `fetch/qwen36-a3b-q4.yaml` |
| S11 | `~/theprojects/queue/sales-brain/docs/VOICE_LATENCY.md` — 46.4 % false cues, 107 s p50 CPU replay, verdict |
| S12 | `~/theprojects/orchestrator/queue/sb-voice-latency.yaml` — the owner rule: real held-out call audio, not synthetic clips |
| S13 | `~/theprojects/queue/inventory/state/inventory.md` (sales-brain entries) — live path cue 1.5 s p95 after the prospect stops |
| S14 | `~/theprojects/queue/.worktrees/a3b-shootout/a3b-shootout/README.md` — the A3B lane is coding/tool-dispatch/agentic text on this same Q4_K_M + pinned llama.cpp |
| S15 | `docs/CODACUS_IDEAS.md` row 29 — prior house triage of this video: "skip — voice pipeline, not text research" |

**What was reused, not rebuilt:** the already-downloaded video captions and
extraction (`data/codacus/shJLycXjhVY.*`), the house triage in
`docs/CODACUS_IDEAS.md`, the measured base VRAM/decode in moe-offload, the
pinned llama.cpp source for the format/support questions, and the measured
micro-turn work already done in sales-brain (`docs/VOICE_LATENCY.md`). No new
tool was built; this note is the deliverable. The repo-finder report in
`PRIOR_ART.md` is the runner's R1 channel-registry check and does not cover this
idea.
