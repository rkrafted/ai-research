# bycloud deep pass — what the channel says, and each idea's upgrade of our system

Source: the whole public bycloud catalogue, newest first (257 videos, 43.19 h;
`data/private/bycloud/catalogue.json`). 252 videos have English captions (249
auto, 3 manual; 42.69 h, 74 797 segments) and became
`data/private/bycloud/transcripts/<video_id>.json`; the 5 with no captions at
all are in `data/private/bycloud/no_captions.json` (0.5 h, below the 10-video
Whisper threshold). Concepts were extracted per video by the local 27B through
`local_batch/` (JSON schema in `bycloud/extract.py:build_prompt`) and assembled
by the runner job into `data/bycloud-extracts.jsonl` (252/252 videos parsed,
1630 concepts, 1018 claims; a raw-file dry run gives 1439/1630 = 88.3 % of
concept quotes grounded by `bycloud/assemble_extract.py:grounded`).

Honesty note repeated from the method: the citations below are the *model's*
concept rows. The quote and start time are what the extractor claimed; the
grounded-run re-checked the quote against the transcript, and every row used
here passed ([video_id@mm:ss] "verbatim quote" appears in that video's
transcript within ±90 s of the cited time). A claim bound to a number can still
misattribute the number — that risk lives in the extraction, not the captions,
which is why the map below stays at the level of *ideas*.

Format: (a) what the channel's concepts say that matters for us; (b) what we
found in our system, checked against the repo (file:line); (c) the table;
(d) the one next step.

## (a) What the channel's concepts say that matters

The channel is AI-news coverage, so most of it is ephemeral. The durable ideas
cluster into six workstreams that each already exist in our tree.

**1. Parameter-efficient adaptation has more shapes than LoRA.** The LoRA
family is presented as a menu of structured variants, not one method: LoRA
learns "specific characteristics"
[-JQDtzSaAuA@01:53] "with the addition of Laura that can learn specific characteristics";
LoCon targets convolutions
[-JQDtzSaAuA@03:29] "the first one is called locon short for Laura for convolutional layer";
LoHa combines two adapters with the Hadamard product
[-JQDtzSaAuA@04:21] "the second one is called loha what it does is that it combines two Lora together"
and [-JQDtzSaAuA@04:21] "using a method called the hadamard product so it theoretically has a better expressiveness",
and the point of the variant is identity preservation
[-JQDtzSaAuA@04:21] "so low coin can be said to have a better identity preservation mechanism".
The training-economics angle recurs: a small model distilled from a teacher's
data [-MKDdpKZTs8A@00:29] "both are pruned from llama 3.28 B and distilled using data synthesized by 45b",
a 3B model fine-tuned for tool use while a 1B sibling was not
[-MKDdpKZTs8A@02:48] "3B seems like it is fine-tuned on tool use and 1B wasn't",
and precision as the memory lever [-MKDdpKZTs8A@02:48] "1B with fp16 requires 2.5 GB of f RAM and quantise with 8 bit integer requires 1.3 GB".

**2. Merging adapters is a first-class way to build a model.** The channel
frames "model mixes" as a genre
[-JQDtzSaAuA@01:30] "we had a nice talk about model mixes where these insane AI models"
and reports emergent behaviour appearing from mixes
[qhYQ20TbtJ8@04:41] "where the reasoning process is emergent, they also observed an increase in thoughts per step".
Smaller models made by distilling/merging recur across the catalogue
[7hMoz9q4zv0@00:00] "researchers make smaller models or even distill them into smaller sizes".

**3. Quantization and KV-cache economics set the serving envelope.** The KV
cache is named as the dominant cost
[SvIJ-BIAPNI@02:57] "there is no other thing that will determine both your speed and your cost than the KV cache"
and [T_GhB7lK2YE@02:51] "Context window is mainly about the growing KV cache";
compression ratios are the reported win
[CsSnSBCVoDU@00:23] "its new architecture also eliminates enough methods to reduce the KV cache by another four times"
and [gC76aeibdFA@04:52] "CSA compresses every four tokens into one KV entry".
Quantization is the everyday lever [7hMoz9q4zv0@01:57] "one uncommon solution is to quantize the models",
[82DyXL0ZXI8@04:25] "It can just serve it at a lower precision to improve speed",
[T_GhB7lK2YE@03:18] "running a quantized version, where some parts of the weights are stored at lower precision",
and the frontier is quantization-aware training down to FP4
[AIRfT41A89s@03:39] "push the indexer QK path, which would typically live in a higher precision FP8 or BF16 down to FP4 through quantization-aware training".
Activation and optimizer memory are called out separately
[T_GhB7lK2YE@00:00] "reduce memory usage by at least two to three times"
and [T_GhB7lK2YE@04:57] "With Adam, the optimizer alone, it keeps two extra copies".

**4. Long context is bought with attention shape, not only VRAM.** Sliding
windows with periodic global layers are one answer
[1BZB4c792-0@02:22] "most of those layers only attend within a 128 token sliding window, while every seventh layer switches to global attention";
RoPE scaling is the cheap retrofit
[9XxTrNoWQzE@04:23] "someone on fortun actually extended llama 3s contact window to 32k using rope";
training for long context is reported as its own stage
[3KwpmSpEplY@00:56] "how they actually trained their model to utilize long context is still really worth to look at";
and a tokenizer change ships with it
[9XxTrNoWQzE@03:51] "llama 3 does use a new tokenizer with a VOC capary size of 128k tokens".
The headline context numbers are large
[90LS4ze4BjQ@02:32] "What's also the highlight of Llama 4 is their use of a new attention technique which enables them to have a context window of up to 10 million tokens"
and attention kernels are the enabling speedup
[48GRiu-TMmg@03:25] "So flash attention improves the speed by three times when you cleverly break down the input into smaller chunks",
with multi-head latent attention compressing the cache
[48GRiu-TMmg@03:01] "which is short for multi head latent Attention".

**5. Agents are the frontier and they fail on their own.** The strongest open
models are sold on agentic ability
[3KwpmSpEplY@01:21] "this is probably one of the greatest agentic LLMs that Open Source Labs have ever achieved",
private agentic benchmarks are named as the real test
[3KwpmSpEplY@01:41] "their extremely strong performance can also be observed on multiple highly difficult private benchmarks like Frontier Suite",
but long autonomous runs are the failure mode
[3KwpmSpEplY@04:07] "Once you want to do an autonomous or complicated agentic workflow, they often fail easily".
The recipe that gets there is teacher trajectories plus a mid-training stage
[1BZB4c792-0@00:24] "adding an agent eccentric mid training stage to follow by a massive reinforcement learning run",
and the tooling is moving to hosted agents with triggers and schedules
[0HotPgUCzyo@01:33] "You get coding agents, and now you can put them on the cloud",
[0HotPgUCzyo@02:46] "One is using it as an event trigger, the other is using it as a scheduler",
[0HotPgUCzyo@03:33] "the scheduler basically lets Oz Agent run at specific cadence".

**6. Extraction is now a named sub-field, and its two failure modes are
hallucination and unvalidated data.** Structured output is table stakes
[_5Xv3kXyBDE@00:23] "now fully support JSON output and function calling";
distilling reasoning into the extractor is the reported quality lever
[_5Xv3kXyBDE@02:02] "they also distilled a train of thought reasoning";
knowledge extraction is described as either teacher-labelled or programmatic
[vyJy-0zBSQ0@03:07] "this process is called knowledge extraction and there are many ways to do this the first way is to have the teacher model label".
The failure modes: the model fabricates facts
[rK8lM4yj9ik@00:32] "model hallucination as the same suggests is where LM would fabricate facts";
retrieval is the workaround
[5Y3a61o0jFQ@00:25] "workaround called rag which stands for retrieval augmented generation in all of the latest research";
and the data itself is the bottleneck
[G4yjpBdki78@00:57] "this is actually a real problem for dataset curation right now"
with pattern-structured sources extracting better
[sdbMbMKTitM@00:57] "perform structural extraction and pattern decomposition a lot better".
An agent's tools are an attack surface
[3pzDw4hAsTs@00:53] "utilizing a method called prompt injection attack prompt injection ction attack basically describes any method".

**7. Mixture-of-experts is the serving architecture the channel keeps returning
to.** The scale split is the point
[90LS4ze4BjQ@00:30] "Meta whipped out 109 billion parameters mixture of experts model with 17 billion active parameters and 16 experts called Llama for Scout",
expert parallelism is the systems answer
[48GRiu-TMmg@04:15] "running the experts for a mixture of experts architecture in parallel"
and [48GRiu-TMmg@04:15] "the second repository they open sourced is called DP, which stands for deep expert parallelism".
A router picks the expert per inference
[PYZIOMvkUF8@00:56] "if something like a router can pick the correct Network for each inference" —
which is also the idea behind our own task-type router.

## (b) What we found in our system

Checked against the repos on this machine, now (not the vendored copy):

**Expert adaptation exists and clears a real bar.**
`~/theprojects/queue/lora-expert-bank/README.md:34-35` is the real bank: two
PEFT LoRA experts over the same 0.90 router threshold — `remote_from_lv` 0.945
on n=110 and `d2-tool-dispatch` 0.9712 on n=453, both
`r=16, alpha=32` on a 4-bit MiMo-9B (`README.md:37`). The router falls back to
the paid model below the threshold, coded at
`~/theprojects/queue/lora-expert-bank/lora_bank/router.py:124,133`.
`forge` proves the recipe is a command: job-post extraction goes 0.3543 → 0.511
on 100 held-out posts (`~/theprojects/queue/forge/README.md:15`) and task routing
0.0 → 1.0 on 136 (`README.md:16`); the training path is 4-bit NF4 + packing at
`forge/forge/gpu_train.py:121,296` with gradient checkpointing at `:270`.
The limit is that post-fields is still below the 0.90 router bar, so that task
keeps paying DeepSeek. The LoHa/LoCon variants above are not in the bank; only
plain LoRA (`ft_gpu.py` recipe, `~/theprojects/orchestrator/bin/ft_gpu.py:9,75`).

**Merging is measured, and one recipe is already a default candidate.**
`~/theprojects/queue/merge-lab/README.md:100` ran the campaign in 34.5 GPU-min;
the published parent was acc=0.9091 / macro-F1 0.6704 while the best merge of
two deliberately halved parents was `magnitude-a-b` acc=0.9364 / macro-F1 0.6764
(`README.md:105,108`) — it matched or beat the full-data reference. TIES *lost*
on the same parents (macro-F1 0.4717, `README.md:107`), so the merge method is
not interchangeable. This is exactly the "model mixes" idea with our own
numbers attached, and `merge-lab`'s from-scratch `peft.utils.merge_utils` port is
already tested on CPU.

**Quantization/KV-cache is where our serving headroom is.** QLoRA is standard
(4-bit NF4), rl-extraction's SFT arm peaks at 7.27 GB
(`~/theprojects/queue/rl-extraction/rl-extraction/README.md:99`), so the 3090's
24 GB is not the binding constraint for training; it is the context and the KV
cache. There is no q8_0 KV cache and no flash-attention flag in any of our job
files. The R5 shortlist in this repo's `docs/upgrades.md` already names exactly
those two (`kv-cache-quant`, `flash-attention`, `context-length` — 8 trials
queued, ≤30 GPU-min each), so the channel is confirming a plan we already had
rather than adding one.

**Long context is the measured wall.** agent-distill's round-1 adapter was
trained at 12K because 16K OOMed on the 3090
(`~/theprojects/queue/.worktrees/agent-distill/agent-distill/README.md:182`),
with packing on and `head_tokens` 3072 (`README.md:161-162`). The channel's RoPE
retrofit and sliding-window answer are not in any of our trainers.

**Agents are measured and improving, but RL has not paid off.** Base SERA-8B
passes 3/44 held-out tasks (0.068); after round-1 QLoRA the student passes 17/44
(0.386), mean turns 22.3 vs 29.3
(`agent-distill/README.md:169,171`). That is the "teacher trajectories + SFT"
recipe the channel describes, minus the mid-training/RL stage. We tried RL on a
different task: on sales-brain extraction, GRPO came in at fact-F1 0.9308 vs SFT
0.9350 — inside the −0.42 pt noise and short of the +2 pt bar, so SFT was kept
(`~/theprojects/queue/rl-extraction/rl-extraction/README.md:98-102`). That is
the honest counterweight to the channel's RL enthusiasm.

**Structured extraction is already the house style, with a grounding filter.**
Every batch of ours uses `format: json` (bycloud's own
`local_batch/bycloud-extract.yaml`), and bycloud's assembler re-grounds each
concept quote against the transcript before a result is accepted
(`bycloud/assemble_extract.py:grounded`). That is the channel's "JSON output"
plus a programmatic guard against hallucinated quotes; our measured rate is
88.3 % of 1630 concept quotes grounded, so 191 rows are *not* trusted. The 1018
claims (numbers, papers, repos) are stored but **not** independently checked yet
— the number-in-source error we saw earlier lives there.

**The orchestrator is the scheduling layer the channel describes for hosted
agents.** The runner already has a GPU queue (`~/theprojects/orchestrator/bin/gpujobs.py`,
including `.reqhash` re-arm and a local-batch hold) and a bulk local-batch lane
(`~/theprojects/orchestrator/bin/local_batch.py`); the FOCUS step convention is
documented at `~/theprojects/orchestrator/README.md:14-17`. What it does not have
is the channel's *trigger* shape (`event trigger` / `scheduler`) exposed as a
user feature; today a session writes the request file and the runner picks it up.

## (c) The table

`now` is a measured number from this machine today; `after` is what the cited
lever is worth on our own held-out sets if it works, not a vendor claim.

| workstream | now (measured) | realistic after | lever | confidence |
|---|---|---|---|---|
| Expert adaptation | 2 experts ≥0.90 (0.945 n=110, 0.9712 n=453); forge post-fields 0.511 < 0.90 | +0.02–0.05 exact-match on a task that is already ≥0.90; gets post-fields toward the bar | train a DoRA / LoRA+ variant beside the plain r=16 adapter on the same split; count it in the bank only at ≥0.90 | medium |
| Adapter merging | best merge acc 0.9364 / F1 0.6764 vs parent 0.9091 / 0.6704; TIES 0.4717 | make magnitude-prune the forge default and test DARE-TIES with tuned density on 2–3 adapters | merge-lab's tested CPU port (`bin/merge_lab.py`); score every merge on the same 110-post key | medium |
| Quantization / KV cache | 4-bit NF4 training; no KV-cache quant, no flash-attention flag; SFT peak 7.27 GB | 1.5–2× context at the same VRAM from q8_0 KV + flash attention | `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_FLASH_ATTENTION=1`; the R5 trials (`kv-cache-quant`, `flash-attention`, `context-length`) | high |
| Long context | 12K is the largest that fits on the 3090; 16K OOMed | 16–32K on the same card via RoPE scaling + KV quant, at some quality cost | add RoPE scaling to the trainer and re-measure held-out pass rate vs the 12K adapter | low–medium |
| Agents | base 3/44 (0.068) → student 17/44 (0.386), 22.3 turns | +5–10 points from more passing trajectories / a mid-training stage; RL only if it clears +2 pt | build more verified tasks and teacher trajectories (agent-distill), keep SFT; re-test GRPO only with a bigger reward signal | medium |
| Extraction & grounding | 252/252 videos parsed, 1630 concepts, 88.3 % quotes grounded, 1018 claims unchecked | ≥95 % grounded rows and a checked-claims rate reported; every claim number re-verified against the transcript | keep `format: json`; extend `assemble_extract.grounded` to claims; store a per-claim `verified` flag | high |
| MoE / capacity | one dense 9B at a time on the 3090 (24 GB); MoE offload on the R5 shortlist, not run | run a 30B-A3B / Mixtral-class MoE at 1–2 experts on GPU | `llama-server --n-cpu-moe` with a cached MoE GGUF | low |
| Orchestration / triggers | runner queue + local-batch lane; request-file triggers only | event/cron triggers for the local lanes without a session | expose the FOCUS step as a scheduled trigger in `bin/projects.py` | medium |

## (d) The one next step

The highest-value, lowest-risk step is the one our own system already has the
harness for: **make the adapter bank the merge default.** `merge-lab` proved a
merge of two halved parents matches the full-data reference (0.9364 vs 0.9091
accuracy, `~/theprojects/queue/merge-lab/README.md:105,108`), and `forge`
already writes both parents and scores them on a held-out split. Add a
`merge-best` step to `forge make` that, when a task type has ≥2 registered
adapters on one base, runs merge-lab's magnitude-prune port and registers the
merged adapter only if it beats every parent on the held-out split. That is a
CPU-first change with one GPU eval, and it turns the channel's "model mixes"
idea into a number on our existing task types.

Second, and only after that: extend `bycloud/assemble_extract.py` so claim
numbers are re-checked against the transcript the same way concept quotes are —
today 1018 claims have no such guard, and that is where the earlier
number-not-in-source error came from.

---

Method/limits: 5 of 257 videos have no English captions (0.5 h) and were left
out; captions are the platform's own auto-captions, so punctuation and numbers
are approximate; the concept extraction is the local 27B's, and only rows whose
verbatim quote re-grounds in the transcript are used above. `python3
~/theprojects/orchestrator/bin/check_grounded_report.py docs/bycloud-upgrades.md
data/private/bycloud/transcripts 20 0.9` is the machine check on the citations.
