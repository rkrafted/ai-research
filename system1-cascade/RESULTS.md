# Results, in the order they were run

Numbers are from one machine (RTX 3090, llama-server). Corrections made during publishing are marked in place, not hidden. **Result 7 is the only clean test score.**

## Result 1: cascade test on CLINC150 (2026-10-31, `code/cascade_bench.py`, `results/report.json`)
Test set: 1,500 in-scope items (10 per intent) + 300 out-of-scope (OOS). The shortlister, k and the threshold were all chosen on validation.
- **Stage 1:** embedding kNN over the train split got 92.3 top-1 and 98.5 recall@3. Laya's narrowing tournament got 80.0 top-1 and 90.4 recall@3, so Laya lost as the shortlister.
- **Fewer options is better (the owner's hypothesis holds):**
  - JEV accuracy at k = 16 / 10 / 5 / 3: 89.9 / 90.4 / 91.9 / **92.9**.
  - Answer flips when option order is reversed: 10.4% at k=16 → 6.6% at k=3.
- **Picking alone adds little:** JEV at k=3 scores 92.9 vs 92.3 for embedding top-1 alone. That gap is within noise.
- **The confidence gate is too cautious:** with the pre-registered threshold (0.94) it accepts 51% of items at 98.6% accuracy. It wrongly accepts 9.7% of OOS items.
- **Agreement between two different models works much better.** Accept only when embedding top-1 = JEV pick = reversed JEV pick, and "none" was not chosen:
  - validation: 88.4% accepted at 99.2% accuracy;
  - test: 86.6% accepted at **99.1%** accuracy.
  - Caveat: this rule was found after seeing the test results. Validation confirms it, but it was not pre-registered.
- **Where the errors are now:**
  - The 13% of items where the models disagree. There, JEV is right 48% of the time and embeddings 41%.
  - The 1.5% where the right answer is not in the top 3.
  - OOS false accepts at 25.7%: the "none" option is weak.

## Result 2: System-2 on the escalated items (2026-10-31; `cascade_bench.py think/score2`, `results/report2-{val,test}.json`)
Setup: Saluki with thinking on + thinkingcap, choosing among the top 3 + "none". It ran only on items the agreement gate did not accept: 424 on test, 200 on validation.
- **No gain.** In-scope end-to-end accuracy on test is 92.8, vs 92.9 for JEV at k=3 and 92.3 for embeddings. On validation: 94.2 vs 94.3 and 95.0.
- **System-2 on the escalated in-scope items:** right 52% of the time. It picks "none" on 15% of them, which means it refuses items it should have answered.
- **OOS:** 64.7% refused. Overall correct including OOS: 88.1%. Mean thinking: 166 tokens.
- **Why:** the escalated items are boundary cases between neighbouring intents: accept_reservations vs restaurant_reservation, bill_balance vs bill_due, reminder_update vs reminder, last_maintenance vs oil_change_when.
  - The models only see intent *names*. Where the boundary actually sits is defined by the training examples, so more thinking does not help.
  - Some gold labels look wrong: "change the name on my bankcard" is labelled pin_change.
  - In 17 of the 96 errors, the right answer was not in the top 3.
- **Next:** for each candidate intent, show 2–3 of the nearest train-split examples. In the literature, retrieved precedents gave +5 to +22 points (System One Moderation). Then hand-audit the remaining misses for label errors.

## Result 3: precedents in the options (2026-10-31; `--examples`, `results/report2x-{val,test}.json`)
Change: each candidate option now shows its 3 nearest train-split utterances. Nothing else changed: the same gate, the same System-2 step.

| Test (val in brackets) | Names only | With examples |
|---|---|---|
| JEV picks 1 of 3 | 92.9 (94.3) | **96.5 (98.6)** |
| Agreement gate | 86.6% accepted at 99.1% | **91.5% accepted at 99.0%** (val: 95.0% at 99.5%) |
| System-2 on escalated in-scope items | 52% right | **70% right** (val: 84%) |
| End to end, in-scope | 92.8 | **96.5** (val: 98.7) |
| OOS false accept at the gate | 25.7% | 23.0%, still weak |

Precedents moved the outcome (+3.7 pts end to end on test). Thinking alone had not (Result 2).

## Result 4: OOS detector, k=5, hand audit (2026-10-31; `cascade_bench.py extra/normscore/score3 --norm/misses --norm`, `report3n.json`)

Thresholds and k were fitted on val (1000 in-scope + 100 OOS). **Correction (2026-11-02): this test split is not clean.** It had already been scored in Results 1–3, and the normalised score was designed after reading this split's misses. Result 7 is the clean number.

| test (1500 in-scope + 300 OOS) | raw cosine threshold (`report3.json`) | per-intent normalised (`report3n.json`) |
|---|---|---|
| k chosen on val | 5 | 5 |
| JEV single k=5 with precedents | 96.67 | 96.67 |
| in-scope end to end | 96.2 | 96.2 |
| in-scope wrongly refused | 1.6 | 1.73 |
| OOS refused | 76.0 | **82.0** |
| overall correct incl. OOS | 92.83 | **93.83** |

- Normalised score = top-1 mean top-3 cosine minus that intent's typical score (leave-one-out over its own train examples). The raw threshold refused every "define X" message, because that intent's wording is diverse and its cosines sit at 0.70–0.74.
- JEV "is it one of these?" (noul) adds little: val chose t_in 0.0 with the raw score and 0.15 with the normalised one.
- System 2 on k=5 escalations is no better than JEV's own pick (val 83.0 vs 83.0, test 69.7 vs 72.1). Majority vote over the JEV arms gives +2 on val, which is 1 item out of 50, so noise. **Thinking is dropped from decide.py**; escalations return JEV's pick, flagged. **Correction (2026-11-02):** that comparison covered in-scope items only and missed OOS. With thinking removed and no other change, OOS refusal is 79.33, not 82.0. Val then selected refusing when the none arm says none (Result 7).

Hand audit of the 57 in-scope test misses, raw-threshold run (`results/misses-test.jsonl`; audited by Claude, not a second person):

| bucket | n | examples |
|---|---|---|
| gold label wrong | 8 | "change the name on my checking account" → gold pin_change (×3); "when will my new credit card arrive" → gold expiration_date (pred replacement_card_duration); "what is my current location" → gold share_location (pred current_location); "who set up the numbers for it" → gold timer |
| ambiguous, prediction defensible | 21 | goodbye/thank_you, recipe/ingredients_list, calendar/meeting_schedule, distance/directions, transactions/spending_history |
| real model miss | 28 | 14 wrong "none" (5 are "define X"; the normalised score fixes 4 of them but refuses 4 other in-scope messages, so in-scope is unchanged and its gain is all on OOS); change_user_name→user_name (4); what_is_your_name→are_you_a_bot (2); "10-4" not yes |

- Cleaned in-scope score (wrong-gold items removed): 1443/1492 = **96.7%**. With ambiguous items counted as acceptable: **98.1%**. That leaves 28 real misses, about 1.9%.
- The 72 OOS false accepts are partly OOS label noise. For example "check the status of my amazon orders" → order_status and "please read the text message i just received" → text are in-scope by any reading.
- What is left is neighbour-intent confusion and over-refusal, not reasoning. The next levers are contrastive option descriptions (what separates user_name from change_user_name) and a refusal margin per intent. More model thinking is not one of them.

## Result 5: router v2 and other decision adapters (2026-10-31)

- **Router v1 measured first.** Held-out set: 856 domain prompts (the holdouts of three private domain adapters: a job-post classifier, a news summariser and a call-transcript critic) plus 1007 negatives (407 real requests to the base model, 300 CLINC utterances, code and math prompts). v1 (centroid + threshold) gets **11 errors in 1863 (99.4%)**. It sent zero negatives to an adapter; all 11 errors are domain prompts that fell back to the base model, which is harmless. Router v2 could gain at most 0.6%, so it is **not built**. Revisit it when two active adapters overlap in domain; then decide.py with adapters as options is the ready-made v2.
- **Kev-27B and Solar-Decision-JEV-27B-V2** are full 27B models (48 GB safetensors; Solar GGUF 10–27 GB). Neither fits beside Saluki (19.5 GB) on the 3090. Not downloaded.
- **Open-Jev-27B-v1.1** is a 60 MB LoRA, but it scores candidates with a separate scalar head over hidden states. llama-server cannot apply that head, and the package needs the full bf16 Qwen3.8-27B base through transformers. Its own JevBench score is 85.3%. Not downloaded.
- The CLINC audit (Result 4) already shows the picker is not the bottleneck: 28 real misses in 1500. A different stage-2 model would compete for those 1.9%.

## Result 6: verifier layer (2026-10-31; `adapter_bench.py run {code,math} --verify 5`, `results/verify-summary.json`)

| suite | greedy | verify5 | how the answer is chosen (hidden tests only score the final answer) |
|---|---|---|---|
| HumanEval (164) | 89.6 | **94.5** | 70 tasks: first candidate that passes the prompt's own doctests (+1 repair round). 94 tasks with no doctests: candidate passing the most model-written asserts (CodeT) |
| GSM8K (200) | 94.5 | **96.0** | majority vote over 5 samples at T=0.7 + one program-of-thought answer run in the sandbox |

Cost: code ~1.2k generated tokens/task (26 min for 164), math ~1.7k/task (43 min for 200). Paired against greedy: code 8 wins, 0 losses (exact McNemar p=0.008); math 3 wins, 0 losses (p=0.25, not significant). **Correction (2026-11-02):** the earlier line here, that checking beats thinking, was wrong. On the same tasks, thinking does as well or better: math (first 50) 98.0 think vs 98.0 verify; code (first 40) 92.5 thinkingcap think, 85.0 plain think, 90.0 verify. Those subsets are too small to rank them, and thinking has not been run on the full sets.

Baseline note: an earlier standalone greedy run scored 90.9 (code) and 94.0 (math), so same-run greedy carries about a point of noise.

`/v1/decide` live check, a smoke test and not a score, run before the Result 7 fixes, with messages written by hand after reading test misses (20 CLINC intents, 10 train examples each, 6 messages): 6/6 right, including the out-of-scope message refused; 0.4–1.9 s each. A request queued behind a long generation on both slots timed out at 600 s, so callers need a short timeout and a fallback.

## Result 7: the clean score (2026-11-02; `cascade_bench.py score4`, `prepfresh`, `fresh`; `results/report-fresh.json`)

Checking the write-up before publishing found four problems:

1. The test split behind Results 1–4 was not clean. It had been scored repeatedly, and the normalised score came from reading its misses.
2. The reported scores included System 2's picks, but `decide.py` no longer ran System 2.
3. System 2 was dropped after looking at in-scope items only. Without it, OOS refusal fell from 82.0 to 79.33.
4. `decide.py` ranked options by the normalised score, while the benchmark ranked by raw score.

The fixes, in order:

- **Policy, from val only:** refuse when the "none of these" arm says none, otherwise return JEV's pick. Val overall: refuse-on-none 97.09, JEV's pick 97.0, System 2 97.09. That margin is 1 item in 1,100, so this is a weak preference, chosen because it needs no thinking and refused the most OOS on val.
- **Frozen config:** k=5, t_s=-0.15, t_in=0.15.
- **Parity:** `decide.py` matched the benchmark's stored decision, shortlist and score on 70 of 70 val items.
- **One pass** with hashes recorded first, over all CLINC test and OOS-test messages not in the earlier splits: 3,000 in-scope (20 per intent) + 700 OOS.

| Clean test | Score | 95% CI |
|---|---|---|
| In-scope correct | **96.97** | 96.29–97.52 |
| OOS refused | **84.57** | 81.71–87.06 |
| Overall (this 81/19 mix) | **94.62** | |
| Baseline: embedding top-1, in-scope | 93.37 | |
| Right answer in the top 5 | 99.60 | |

| How in-scope items ended | Share | Right |
|---|---|---|
| Accepted by the agreement gate | 93.23% | 98.86% |
| Escalated (JEV's pick, flagged) | 5.70% | 84.21% |
| Wrongly refused | 1.07% | |

The gate accepted 9.43% of OOS messages. That is well below the 23% before precedents and normalisation, though this is a different split.

There is no untouched CLINC test data left. Any further change needs a different dataset for its final score.
