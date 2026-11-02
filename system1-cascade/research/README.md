# What was read, and the plan it produced

Question: which published JEV/Laya-type ("System 1", decision-only) models exist, what are they used for, and what would help most?

| File | What it is | How far to trust it |
|---|---|---|
| `mentions.md` | About 45 outside mentions, read by one cloud agent on 2026-10-30 | Authors' own claims, none reproduced |
| `CATALOG.md` | Models, uses, results and techniques extracted by a local model. Quotes are checked word for word against the source | Being regenerated, because the 2026-10-30 run's files were lost. The first pass read no arXiv papers and missed 240 of 566 chunks, so a second pass is running. Added in a follow-up commit |
| `local_research.py` | The pipeline behind the catalog: fetch, extract, report | Recovered from the session transcript |

The plan below is the one written before the benchmarks. `../RESULTS.md` shows what survived testing: several items did not, including think-when-unsure and router v2. `../FUTURE.md` has the current order.

All outside numbers below are reported by their own authors and were not reproduced here.

## What these models are (corrections)
- **Laya**: a 421M ModernBERT encoder that **picks among options** and does not generate text. Use it to make a shortlist or as a cheap first pass. Do not use it as an expert, a security gate, or on lists of more than about 20 options.
- **Jev**: hosted and closed. **JEV-27B-VL** (autotrust) = Qwen3.8-27B + LoRA + decision head. It runs here on Saluki-27B, a 2-bit quant of Qwen3.8-27B (`code/jev_decide.py`, adapter name hub-jev).
- **Other 27B decision adapters to bench:**
  - Kev-27B (jaredpalmer/kev, Qwen3.8-27B).
  - Open-Jev-27B-v1.1 (Zefan-Cai).
  - Solar-Decision-JEV-27B-V2.
  - Nimble-v4-max is on Qwen3.5-27B, so it does not fit Saluki.

## Combined system, ranked by expected gain

1. **Calibration + escalate-when-unsure gate.** Everything else depends on it.
   - Fit one temperature per decision type on held-out data. Laya's ECE went 0.466→0.081 and 0.204→0.037 with this.
   - Set thresholds with Learn-then-Test. Thresholds set from point estimates broke 147 of 190 times.
   - Output: each decision is either *accept* or *escalate* (Saluki thinking, then cloud).
2. **Two-stage narrowing cascade.** This was the owner's idea.
   - Stage 1: Laya or embeddings shortlist k. Choose k from measured recall, so the right answer is almost never dropped.
   - Stage 2: JEV-27B picks 1 of the k. Ask twice with the options in a different order, and include a "none of these" option.
   - If the two picks disagree, or "none" wins, escalate.
   - Evidence:
     - JevCascade: 95.4% on a 5-option menu vs 80.0% on all 150 labels.
     - Sayora: wrong page 33%→3%.
     - Laya: 88% at 6 options vs 51% at 42 options.
3. **Think only when unsure, on the same Saluki model.**
   - Answer from the option logprobs when the gate accepts. Otherwise think on the same cached prompt; this combines with thinkingcap.
   - Evidence: ThinkJev on Qwen3.5-27B answered 24.5% of decisions directly at 3.9% error, and early-stopped thinking used 72% of the thought tokens.
4. **Verifier layer for the generative benchmarks.** This is where accuracy goes up, not just cost down.
   - Code: generated tests, best-of-n, self-repair. DFlash speed pays for the extra samples.
   - Math: majority vote over samples and program-of-thought, i.e. compute in code.
   - Tools: validate the schema and AST, retry on failure.
   - Summaries: JEV picks the best of the candidates. Decision models are weak judges for math and code (JEV-as-a-Judge), so those two are checked by running code, never by JEV.
   - Do not cascade into a judge that makes the same mistakes. LLM fallbacks repeated over 85% of the decision model's confident errors.
5. **Router v2.**
   - Feed JEV probabilities on several questions ("needs tools? code? math?") into a small trained router, as SeLMRoute did (+2.85 pp). Zero-shot routing was at chance.
   - Ask all the questions in one prefill (block-causal masking / state reuse); 40 queries cost about 2.7× one query.
6. **Picking context instead of embedding RAG**, for private classifiers and retrieval.
   - Show all candidates at once (Sayora: answers right 65%→90%). Scoring one candidate at a time lost.
7. **Bench the other decision adapters as the stage-2 picker** through the existing gate: Kev-27B, Open-Jev-27B, Solar-V2 vs hub-jev.
   - If one of them wins, train our own decision adapter on the jev-distill corpora (JanerasGate v3, SargeDev).

**Skip:** zero-shot routing with Laya, Laya for security labels (48%, below the 79% constant baseline), and numeric/counting tasks in decision models.

## "100% without cheating": the honest version
- Never train on a test split or select answers using gold labels. Calibrate on held-out data only.
- Score every benchmark on its **original** set *and* on a **cleaned** set (GSM8K-Platinum style for math, EvalPlus tests for code). Some original items have wrong labels or weak tests, so 100% on the original set is not a valid target.
- What counts as reaching the goal: **every remaining miss is either fixed or confirmed by hand as a bad item.**
- Today's starting point: HumanEval 90.9 (92.5 with think), GSM8K 94.0 (98 with think), BFCL 88.5, CNN/DM ROUGE-L 27.4 with its adapter.

## Order of work
1. Cascade test (run on CLINC150 in the end). Compare a single pick, shortlist → pick, and shortlist → pick twice with shuffling + "none". Add calibration (step 1) to the same run.
2. Verifier mode in `adapter_bench.py`: code best-of-n + repair, math majority vote, tool retry.
3. Wire the gate in step 1 and think-when-unsure into the private experts service.
4. Router v2, then bench the other adapters.

