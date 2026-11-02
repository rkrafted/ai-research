# What each decision model and trick is good for

A summary of the published JEV/Laya-type models: what to use them for, why, and which of their ideas can be combined.

**How far to trust this.** Outside numbers are the authors' own, and nothing here reproduced them unless the row says "tested here". Many come from small samples or different datasets, so they can't be compared with each other, and gains won't simply add up. The combinations at the end are untested proposals.

**Coverage.** This draws on `research/mentions.md` plus a local model's first pass over 448 repos, model cards and HN posts, of which 249 were relevant. That pass missed 240 of its 566 text chunks and read no arXiv papers directly, so a second pass is running. This file will be updated when it finishes.

## Which approach for which job

| Job | Use | Why | Evidence |
|---|---|---|---|
| Pick one of many intents or tickets | Embedding shortlist, then a decision model that sees examples of each option | Accuracy falls as lists grow, so shrink the list first | **Tested here:** 96.97% in-scope on CLINC150. [JevCascade](https://arxiv.org/abs/2610.07716): 95.4% on a curated 5-option menu vs 80.0% on all 150 labels |
| Choose which documents the LLM reads | One decision over all candidates at once, with a "none" option | Comparing candidates works; scoring them one at a time lost | [Sayora](https://sayora.ai/many-hands/jev-vs-embeddings/): wrong page 33%→3%. [Unblocked](https://getunblocked.com/blog/jev-in-production-vs-cross-encoder): recall 63.8→77.6% |
| Decide when to call a bigger model | Send only the least confident items up | Confidence shows where the small model is weak | [jev-cascade](https://github.com/rlisml/jev-cascade): escalating 25% recovers 96% of the large model's accuracy at 40% of the extra cost, +7.2 pp over random routing |
| Pick a tool | Coarse top-K, then a final choice | Same idea as the cascade, applied to tools | [JevRouter](https://github.com/BillionsBobby/JevRouter): 38→44% hits on ordered routing. Beam selection 28→36% with no extra calls |
| Pick an expert adapter | A small trained router on top of decision-model outputs | Zero-shot routing was at chance | [layaMOE](https://github.com/vishalmysore/layaMOE): 67.3 vs 59.6 on 108 cases. [SeLMRoute](https://arxiv.org/abs/2609.34736): +2.85 pp. **Tested here:** router v1 is already 99.4%, so this waits until adapters overlap |
| Reason only when unsure | Answer directly above a confidence bar, otherwise think | Saves thinking tokens on open-ended questions | [Jeeves](https://github.com/PostHog/jeeves): 0.822 at threshold 0.9 vs 0.818 always thinking. **Tested here:** no help on the cascade's escalated items, which were label boundaries rather than reasoning |
| Plan and act in a loop (browser, robot, game) | A large model plans, code enforces the plan, the small model acts | Small models alone fail at strategy; a plan given only as advice helps less | [Jev_Star](https://github.com/sc2musa/Jev_Star), StarCraft II, ten games each from archived runs: JEV alone 0 wins, plan as context 3, plan as hard constraint 9. [Jev-Mobile](https://arxiv.org/abs/2609.30186): −73% API cost, −5 pp success |
| Trim logs and context for an agent | Ask the model which lines matter to the current goal | Judges relevance better than fixed filters | [mu](https://github.com/qybaihe/mu), held-out goals: cut 46.4%, lost 0 of 19 required lines. Keeping only failures cut 59.4% but lost 6 |
| Screen fetched pages for injected instructions | A decision model before the agent reads the page | Beats pattern matching on known attack shapes | [hermes-jev-skills](https://github.com/kerpopule/hermes-jev-skills): 70 of 79 caught vs 11 for a pattern scan, 0 of 1,520 clean chunks withheld. The authors warn this recall isn't portable |
| Bulk labelling | Decision model first, LLM on low confidence | Costs far less at similar quality | [CSS annotation](https://arxiv.org/abs/2609.24574): matches the LLM at ¼–½ of its cost |
| Latency-critical or no GPU | Small CPU-native models | Fast, but less accurate | [jevos](https://github.com/feder-cr/jev): about 30 ms vs 311 ms for hosted Jev. **Tested here:** Laya lost as a shortlister, 80.0 vs 92.3 for embeddings |

## Where not to use them

| Task | Use instead | Evidence |
|---|---|---|
| Counting, arithmetic, derived values | Compute in code and hand the model the result | [JevGraphBench](https://github.com/VictorYXL/JevGraphBench): 100% on adjacency, 67.80% on degree. [JevNext](https://github.com/Bring-AI/JevNext): error 31.17% from recall, 0% when values were supplied |
| Judging math and code answers | Run the code | [JEV-as-a-Judge](https://arxiv.org/abs/2609.26550): weak where the verdict must be worked out |
| Fixed-label gates with training data | A fine-tuned classifier | [Decision Gates](https://arxiv.org/abs/2610.00346): fine-tuned DeBERTa wins on almost every labelled benchmark |
| Security labels with Laya | Jev, or a trained classifier | [Archestra](https://archestra.ai/blog/we-tested-jev-on-100-real-agent-calls): Laya 48%, below the 79% from always answering the majority label |
| Final authority over actions | Code, with the model as one input | [jevals](https://github.com/openlayer-ai/jevals): the model can say a call looks destructive, but not whether it should run |
| Questions that may have no answer, without an abstain option | Always offer "none" or "insufficient evidence" | [Calibration audit](https://github.com/jujumilk3/jev-calibration-audit): removing abstain took accuracy on unanswerable items from 0.950 to 0.000 |

## Tricks worth borrowing

"Have it" means the published cascade already does this.

### Getting a truer answer from the same model

| Trick | From | Reported gain | Status here |
|---|---|---|---|
| **Read every rotation of the options and average** | [AnyJev](https://github.com/nokia-applied-research/AnyJev), [Bongard](https://github.com/AgentBull/bongard) | Order flips 0.230→0.073, accuracy 0.747→0.803, automatable at ≤5% error 7.7→46.3%, no labels needed (AnyJev, one 300-item test) | **Try first.** We read only two orders |
| Compare to the answer given an empty input (PMI) | [open-jev](https://github.com/daseinlabs/open-jev), [qwen-rlcd](https://github.com/shamazharikh/qwen-rlcd) | Meant to remove a bias towards popular options; no number seen | Try: one extra read per option list, reusable across inputs |
| Agreement between independent readers | [tetrajev](https://github.com/FeiLiuEM/tetrajev); **tested here** | 98.86% accuracy on 93.2% of in-scope items | Have it |
| "None" or "insufficient evidence" option | [jevals](https://github.com/openlayer-ai/jevals), [Sayora](https://sayora.ai/many-hands/jev-vs-embeddings/) | Sayora: correct "not on site" 23→95% | Have it |
| Option scoring that looks at the rival options | [jevbetter](https://github.com/olanotolu/jevbetter) | Top-1 0.873→0.916 and half the ECE on hard menus, on its own data | Later: needs a trained head |

### Writing better options

| Trick | From | Reported gain | Status here |
|---|---|---|---|
| Show examples per option | **Tested here**; [moderation precedents](https://arxiv.org/abs/2610.07953) | Here: 92.9→96.5 at k=3. Moderation: +5 to +22 pts | Have it |
| Describe options, don't just name them | [jevals](https://github.com/openlayer-ai/jevals) recommends it; [synthetic survey](https://github.com/jjd-lab/jev-synthetic-survey) measured almost no change (gap 0.6864→0.6822) | Mixed | Try, as contrastive descriptions that target neighbouring options (`FUTURE.md` step 3) |
| Write options as full sentences | [Verdict](https://github.com/Heman10x-NGU/Verdict-open-jev) | +6.9 pp on its standard tier, +2.1 on easy. Its backbone descends from NLI, so this may not transfer | Try: a one-line prompt change |
| One question per thing you want to know | [jevals](https://github.com/openlayer-ai/jevals) | Guidance | Already the shape of the cascade |

### Trustworthy confidence

| Trick | From | Reported gain | Status here |
|---|---|---|---|
| Learn-then-Test thresholds | [Confidence Is Not a Promise](https://arxiv.org/abs/2610.36311) | Point-estimate thresholds broke in 147 of 190 evaluations; this held in all | **Try** (`FUTURE.md` step 2) |
| Separate allow and block thresholds | [Agent Security Decisions](https://arxiv.org/abs/2609.33401) | Safe automation 7.6→47% | Try, for refusing vs accepting |
| One temperature per question type and per list size | [Verdict](https://github.com/Heman10x-NGU/Verdict-open-jev), [SearchJev](https://github.com/EvoScientist/SearchJev) | Hard-tier ECE 0.298→0.118 | Try only if confidence is used; we rely on agreement instead |
| Thresholds per action, not per model | [jevals](https://github.com/openlayer-ai/jevals) | Guidance: a refund and a lookup shouldn't share one | Use on the first real task |

Small samples mislead here: 20 correct out of 20 still allows a true precision as low as 0.86 ([jev-calibrate](https://github.com/smkrv/jev-calibrate)).

### Speed

| Trick | From | Reported gain | Status here |
|---|---|---|---|
| Share the input's cache across questions and options | [Visual-Jev](https://github.com/guanxuyu-sv/Visual-Jev), [SemIf](https://github.com/TheoLeeCJ/SemIf), [openjev-sglang](https://github.com/ekzhang/openjev-sglang) | 50.7→5.7 ms per question at 32 questions (Visual-Jev) | **Needed for rotations**, which multiply the reads |
| Stop reading rotations once the answer is settled | [AnyJev](https://github.com/nokia-applied-research/AnyJev) | Fewer reads, with a stopping threshold calibrated without labels | Try alongside rotations |
| Reuse answers for repeated inputs | [jev-drone](https://github.com/RomanSlack/jev-drone) | Fewer calls; no accuracy number seen | Later |
| FP8 quantization | [openjev](https://github.com/razorback16/openjev), [jev-novel](https://huggingface.co/aikexue170/jev-novel-3-27b-fp8) | 98.5% same top answer; 1.9% of verdicts flipped | Not needed: we already run a 2-bit base |

### Training your own adapter

| Trick | From | Why | Status here |
|---|---|---|---|
| Soft-label loss plus Brier score | [SearchJev](https://github.com/EvoScientist/SearchJev) | Trains the probabilities, not just the top answer | Later (`FUTURE.md` step 8) |
| Shuffle or rotate options during training | [Jev-Vision](https://huggingface.co/IntelligenceLab/Jev-Vision-0.8B), [PlayJev](https://github.com/OmniJev/PlayJev) | Reduces order bias at the source | Later |
| Teach from a stronger model | [AnyJev](https://github.com/nokia-applied-research/AnyJev), [PlayJev](https://github.com/OmniJev/PlayJev) | Labels without human effort; PlayJev hands its least confident steps back to the teacher | Later |
| Fine-tune for one task | [open-jev](https://github.com/daseinlabs/open-jev) | 84.2 / 66.3% zero-shot → 97.0 / 100% on its cookbook example | Later, for a real task with labels |
| Fit a head to the task on a frozen base | [open-jev](https://github.com/daseinlabs/open-jev), [layaMOE](https://github.com/vishalmysore/layaMOE) | Cheap, and heads can be swapped per domain | Later |

## How they combine

```
input
  → code filters what it can decide by itself
  → embeddings shortlist 5 options, each with examples and a contrastive description
  → decision model reads every rotation of them, plus "none", over one shared cache
  → all rotations agree with the embedding, and pass Learn-then-Test thresholds?
       yes → accept
       says none → refuse
       otherwise → escalate
  → code keeps the final say over any action
```

Each combination below targets something measured as weak here, in the order we'd try them. Expected gains are unknown until tested.

1. **Rotations + "none" + agreement + shared cache**, for out-of-scope messages. 108 of 700 got through in the clean run, our biggest visible weakness. Reading 5 rotations instead of 2 orders should expose more order-dependent picks, and the shared cache keeps the cost down.
2. **Learn-then-Test + separate accept and refuse thresholds + more OOS examples**, for the same weakness. Our thresholds came from only 100 examples.
3. **Examples + contrastive descriptions + full-sentence options**, for the 91 in-scope mistakes. The audit's main real-miss pattern was neighbouring intents.
4. **Cascade + schema check + retry**, for tool calls (BFCL 88.5). The cascade picks the tool, Saluki fills in the arguments, and code validates them. This extends `FUTURE.md` step 5 and is the combination most likely to carry the cascade beyond intent picking.
5. **Decision-model outputs + small trained router + expert adapters**, for adapter choice, once two adapters overlap. Until then router v1 leaves at most 0.6% to gain.
6. **Decision-model log trimming + the verifier**, for long local agent runs. Less irrelevant text in a fixed context window. Unmeasured here.
7. **Soft-label loss + rotation in training + teacher labels + logged decisions**, for our own decision adapter. Only if steps 1–3 show the picker is the limit.

**New data needed.** No clean CLINC test data is left. Tune 1–3 on validation data and score them on a new dataset such as BANKING77 or HWU64, frozen before the first run.
