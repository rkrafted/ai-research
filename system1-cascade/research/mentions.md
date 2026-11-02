# C — Mentions of JEV / Laya outside their own repos (as of 2026-10-30)

## Read this first: which models these are

- **Jev** is TypeSafe AI's **hosted, closed** "System One" model, released 2026-09-15 (versions 1.13.0 and 1.14.0). Its size is not published; Red Hat lists the parameter count as "Unknown". It returns typed choice/score/yes-no probabilities from a single forward pass.
- **Laya** (`convaiinnovations/laya`, Apache-2.0) is a **421M ModernBERT-large encoder**, with a 322M mmBERT multilingual variant. It is an open copy of Jev's request/response shape. It is not a generative model.
- **No outside source names a "JEV-27B".** *(Corrected 2026-11-03: this was wrong. A later pass found [autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) and cards that cite it.)* The open 27B Jev-style checkpoints we found are:
  - Zefan-Cai/Open-Jev (2B/9B/27B)
  - getainode/jebadiah (LoRA on Qwen3.8-27B)
  - OmniJev/OneJev (Qwen3.5-0.8B up to Qwen3.8-27B)
  - SII-Decision-27B (arXiv 2610.03935)

  UNVERIFIED which of these the user's "JEV-27B" is. Results below on hosted Jev may not carry over to any of them.
- **How much of this is checked:** almost every number below is self-reported by its author. Nothing has been independently reproduced, and the awesome-jev survey says the same (its F7). Each row's **Check** column says how much we read:
  - **read**: we read the abstract or the post.
  - **headline**: we only saw the one-line headline in the awesome-jev survey (Eurekaleo, v0.3.1, 2026-10-07). These rows are UNVERIFIED.

Sources searched:
- arXiv API, by keyword and by title (Semantic Scholar returned HTTP 429 on every query)
- HN Algolia: about 385 Jev/Laya stories since 2026-09
- the awesome-jev-survey README
- HF Hub cards
- 13 blog/GitHub posts fetched in full
- web search
- reddit-mcp, which returned 0 results for every query, including a plain "jev". Probably broken or blocked.
- the house finder `prior_art`, which ran OK. It pointed to arXiv 2610.02267 and 2609.37647, plus the HF items Luni/laya-jev-benchmark, sumitranjan/Jev-Laya-Guard-Lite (card empty) and morrenhale/decision-benchmark-jev-laya-julia (card empty).

---

## (a) Mentions table

### A1. Routing: LLM, LoRA and model/effort choice (closest to the user's setup)

| Source | Date | Use case | Setup | Result (as reported) | URL | Check |
|---|---|---|---|---|---|---|
| JevSoup (arXiv) | 09-25 | Picks LoRA experts per input, no training | Jev reads the input and the expert descriptions and picks 2 LoRAs. The second LoRA's update is projected orthogonal to the first, and the two are merged with equal weights. Tested on 3 Qwen3 sizes and 14 PorTAL tasks. | Up to **+1.19 pp** task-macro and **+1.21 pp** sample-micro over the best outside baseline | https://arxiv.org/abs/2609.30922 | read |
| SeLMRoute (arXiv) | 09-28 | Routes queries among 20 LLMs | Jev answers interpretable questions about each query (needs reasoning? needs outside knowledge? …). The probabilities feed a small supervised router. | **72.08%** vs **69.23%** for the best single model, on LLMRouterBench (11,481 queries). Cost-aware setting: mean PerfGain 2.66%. | https://arxiv.org/abs/2609.34736 | read |
| Fast Models, Slow Evidence (arXiv) | 10-01 | 11 decision points inside an agent harness: model routing, tool choice, RAG relevance, injection | Paired test of Laya against Jev on byte-identical inputs: 7,283 cases + 6,640 variants | **Neither beats chance on zero-shot model routing.** Jev is better on 9/11 points (+10.8 to +46.0 pp). The two tie on RAG gating. The authors' own audit cut a claimed 23.9% saving to **4.3%**. | https://arxiv.org/abs/2610.02267 | read |
| Harness Tokenomics (arXiv) | 09-24 | Picks the model for coding-agent sessions | Jev labels each prompt against a bring-your-own taxonomy. Routing happens only at session start, side lanes and subagent launch, so the prompt cache is never rebuilt. | Emulated 10,000-seat company: **13–21%** of model spend recovered. This is a simulation, not a live run. | https://arxiv.org/abs/2609.28919 | read |
| astgl.com: Local Laya vs hosted Jev | 09-22 | Sends work to the right specialist model in the author's agent stack | Laya typed-decisions checkpoint on an M3 Ultra. A deterministic filter first narrows to ≤5 eligible models + abstain. On abstain or timeout, a local fallback runs. 300 ms timeout. | **37/40** (92.5%) vs 33/40 for deterministic ordering, on a held-out replay. Laya picked directly 15 times, fell back 17 times, blocked 8. p95 latency **143 ms**, rising to **467 ms** on a busy host. Only 40 cases. | https://astgl.com/p/local-laya-vs-hosted-jev-typed-decisions | read |
| Student-Guided Teacher Distillation (arXiv) | 10-01 | Routes requests across 60 task categories | A ModernBERT student gives a top-k shortlist. A DeBERTa NLI teacher reranks only the shortlist. The paper positions itself against Jev and Laya. | 77.5% teacher agreement on 200 examples. Coverage@16 is 91–100%. The authors say the evaluation is still in progress. | https://arxiv.org/abs/2610.02516 | read |
| s1-llm-auto-router (HF model) | 10 | Classifies a request before an LLM router | mmBERT-small, 7 heads: category, difficulty, stakes, needs_tools/vision/long-context, follow-up. Jev-compatible API. | Latency only: **14 ms p50** on CPU. No accuracy figure on the card. | https://huggingface.co/decisionmodels/s1-llm-auto-router | read |
| REFLEX with Jev (arXiv) | 09-22 | Bounded agent decisions, with escalation | Jev is the fast decision layer. A strong LLM is called only when confidence is low or text has to be generated. | **95%** success with **72.7% fewer** strong-model calls, on a frozen 100-task bench. Little advantage on BFCL/τ-style tasks where cheap routing is already accurate. | https://arxiv.org/abs/2609.26532 | read |
| effort-router; jev-router; Keen Code; Jev → Grok/Codex demo; JevRouter | 09-17 to 10-19 | Picks Claude Code's reasoning effort or the cheapest model per task | Jev classifies each prompt | No numbers found | HN 49880075, 49798407, 50156970, 49747155; github.com/BillionsBobby/JevRouter | UNVERIFIED |

### A2. Verifier / judge / gate (accept when confident, escalate when not)

| Source | Date | Use case | Setup | Result | URL | Check |
|---|---|---|---|---|---|---|
| JEV-as-a-Judge (arXiv) | 09-22 | Judge for pairwise evaluation | Accepts Jev's verdict above a threshold frozen in advance. Otherwise escalates to a reasoning judge. | **+0.9 pts over GPT-6** at **41% of its fee**, on 1,610 held-out pairs. Matches GPT-6 exactly in a pre-specified live test. Weak where the verdict has to be worked out: math, code, logic. | https://arxiv.org/abs/2609.26550 | read |
| Rubric judges (arXiv) | 09-24 | Rubric scoring, including use as a training reward | Jev and Decider vs flash-tier LLM judges, with an LLM fallback cascade | Trails the LLM judges significantly in ≤6 of 27 comparisons, at 1/16–1/357 of the cost. The cascade gains **at most +2.7 pts**, because over 85% of LLM verdicts repeat the decision model's confident errors. | https://arxiv.org/abs/2609.29769 | read |
| CSS annotation (arXiv) | 09-21 | Labelling text for social-science research | Decision model first. Items with low confidence go to an LLM. | Alone it is **−11.6 macro-F1** vs the best LLM, at 44× lower cost. The cascade **matches the LLM at ¼–½ of its cost**. | https://arxiv.org/abs/2609.24574 | read |
| Just Ask Jev (arXiv) | 09-24 | Zero-shot detector of alignment failures (10 types) | One generic question per input | Median **AUROC 0.886**. Beats supervised baselines on most benchmarks, at 1/63 of the cost of an LLM judge. | https://arxiv.org/abs/2609.29429 | read |
| SanSi (arXiv) | 10-06 | Looped typed-decision model, also used as an RL reward judge | Same layers looped 1–8× before the readout | **+13.5 pts** over a non-looped model of the same shape. As a judge without gold answers: generator's correct share **37%→52%**. | https://arxiv.org/abs/2610.07730 | read |
| ThinkJev (arXiv) | 10-27 | Fast/slow decisions from one open hybrid-thinking model | System 1 reads option-letter logprobs. System 2 thinks on the same cached prompt. Acceptance has a statistical guarantee. | Qwen3.5-27B: System 1 alone answers **24.5%** of decisions at **3.9%** error (target 5%). Early-stopped thinking gets 81.6% vs 82.7% for always thinking, using **72% of the thought tokens**. | https://arxiv.org/abs/2610.37732 | read |
| Jeeves, PostHog (GitHub) | 09-29 | Jev-like model that reasons only when unsure | Qwen3.5-9B with LoRA r=16 + pointer head. Thinks when confidence < 0.9. | JevBench-hard **0.820/0.793** vs Jev 0.730. Test set 0.851 reasoning vs 0.812 without. Trails Jev on knowledge: MMLU-Pro 0.770 vs 0.840. p90 latency 4–8 s. | https://github.com/PostHog/jeeves | read |
| Agent Security Decisions (arXiv) | 09-27 | Injection / risk / harmful-request gate | Jev, Laya, Decider, Nimble vs classifiers and LLM judges | Safe automation coverage goes from **7.6% to 47%** with separate allow/block thresholds. Confident "safe" misclassifications cluster inside particular attack groups. | https://arxiv.org/abs/2609.33401 | read |
| Decision Gates Benchmark (arXiv) | 09-29 | Automated branching gates | 8 decision checkpoints vs a fine-tuned DeBERTa vs generative models | A fine-tuned DeBERTa-v3-large wins on almost every labelled benchmark. A 5%-risk threshold still admits **31%** of out-of-scope requests. Swapping yes/no flips **50.5 per 100** of Jev's answers. | https://arxiv.org/abs/2610.00346 | read |
| Laya reproduction (arXiv) | 09-27 | Laya as a probabilistic assessor with selective escalation | Pre-registered study | Card accuracy reproduced: 0.767 vs 0.766. Laya is **under-confident** (gap −0.214). One temperature (0.469) fixes it: ECE 0.204→0.037. The gate misses its 10% error target. No zero-shot OOD transfer (0.617). | https://arxiv.org/abs/2609.33843 | read |
| Confidence Is Not a Promise (arXiv) | 10-26 | Confidence gates, Laya (5 checkpoints) vs Jev | 190 exchangeable evaluations | Thresholds set from point estimates break their promise in **147/190**. Learn-then-Test keeps it in all. A fine-tuned Laya needs **250–8,000 labels** to be non-inferior to Jev. Confidence AUROC for right vs wrong is 0.80–0.96. | https://arxiv.org/abs/2610.36311 | read |
| Red Hat guardrails (blog) | 10-02 | Prompt-injection and toxicity guardrails | Laya, Jev and DiffusionGemma vs DeBERTa and LLM judges | Prompt injection: Laya **85.44%** at 119 ms, Jev 86.35%, DeBERTa-v3 classifier **89.01%** at 54 ms, Qwen3.6-35B 89.31%. Red Hat: decision models were not faster, cheaper or better than LLM-as-judge, except that Laya is small. | https://developers.redhat.com/articles/2026/10/02/benchmarking-ai-decision-models-against-traditional-guardrails | read |
| Archestra (blog) | 09 | Security labels on 100 real Claude Code tool calls | 4 labels per call; 0-shot and 9-shot | Sonnet 5 98%, **Jev 93% (95% 9-shot)**, Nimble-9B 83%. Always answering the majority label scores **79%**. **Laya 48%/46%**: below that constant, with refusal precision 12%. Jev made no errors at confidence ≥0.7. | https://archestra.ai/blog/we-tested-jev-on-100-real-agent-calls | read |
| Pentest decision layers (arXiv) | 09-24 | Jev/Laya to judge findings, recalibrate severity and prune agents in a pentest harness | One NeuroSploit run with Jev, one without | Not statistically tested. Authors say so. | https://arxiv.org/abs/2609.28940 | read |

### A3. Retrieval, reranking, memory, code search

| Source | Date | Use case | Setup | Result | URL | Check |
|---|---|---|---|---|---|---|
| Sayora (blog) | 10-13 | Picks which website page to hand the LLM (RAG) | One Jev choice over every page description + a "none" option. The top 2 pages go to the LLM. | Wrong page **33%→3%** vs chunk embeddings. Answered right end to end: **65%→90%**. Correct "not on site": 23%→95%. p50 0.28 s. About $0.0005 per lookup. | https://sayora.ai/many-hands/jev-vs-embeddings/ | read |
| Unblocked (blog) | 09-22 | Chooses which memory notes the agent sees | Jev shown 20 notes at once ("fan-out") vs the production cross-encoder; 12,927 labels | nDCG@5 **+0.048**. Precision **34.8→46.4%**, recall **63.8→77.6%**. Scoring one note at a time *lost* (−0.033 nDCG@5). Sweeping the threshold mattered more than prompt tuning. | https://getunblocked.com/blog/jev-in-production-vs-cross-encoder | read |
| Oko, tyrpien (blog) | 09-21 | Code search for coding agents | Jev reranks 30 code snippets per query | Over 162 sessions: Claude Code **31% faster**, Codex 18%, OpenCode 15%. **20–42% fewer tokens.** The speed-up only appeared after agents were told when to stop searching. | https://tyrpien.com/blog/oko-agent-search | read |
| kapa.ai (blog) | 10-15 | Reranking on 1,000 company-knowledge cases | Jev Noul and Score modes vs Voyage, Cohere, Qwen3-reranker-8B | Jev 1.14 **0.532** vs Voyage rerank-3 0.535 (Voyage was also cheaper and faster). Jev beat Cohere v4-pro (0.506) and Qwen3-8B. | https://www.kapa.ai/blog/jev-reranker-benchmark | read |
| Mnemon; Jev-Mem; Selection-vs-Extraction memory (arXiv) | 09-21 to 09-28 | Jev judges agent-memory evidence | — | LoCoMo **91.7%** (AUC 0.94). Jev-Mem judge 0.777 (+11.0%). Selection memory ≈ extraction memory at 1/3,061 of the write cost. | arXiv 2609.36059, 2609.23986, 2609.34227 | headline (UNVERIFIED) |
| JEVDB (arXiv) | 10-01 | Semantic SQL joins | Pruning first, then typed decisions | Finishes 540K-pair joins. No quality number in the headline. | https://arxiv.org/abs/2610.02046 | headline (UNVERIFIED) |

### A4. Agent executors and coordinators (big model plans, small model acts)

| Source | Date | Use case | Setup | Result | URL | Check |
|---|---|---|---|---|---|---|
| Jev-Mobile (arXiv) | 09-24 | Android GUI agent | A VLM sets local goals; Jev picks actions from the accessibility tree | AndroidWorld **79%** vs 84% for a step-wise VLM. **−73.4% API cost** and −32.7% time on the successful runs. | https://arxiv.org/abs/2609.30186 | read |
| S1-MAS (arXiv) | 10-06 | Multi-agent coordination: task choice, roles, termination | A System-One controller + a compact reader; an LLM does the open reasoning. The survey says Laya is the controller. | **−44.9% to −97.2%** GPT-4o tokens and −37.8% to −93.0% latency vs AgentVerse/DyLAN/SelfOrg. The abstract says "superior accuracy". The survey adds that random choices trail by only 1.8 pts. | https://arxiv.org/abs/2610.08155 | read |
| Learning to Formalize (arXiv) | 10-27 | Robot tasks (CALVIN-D) with no candidate list | A small trained formalizer generates candidates; a frozen decision model picks one | **11.8%→52.6%** task success with the formalizer. 2.09 s vs 3.67 s for Qwen3.5-27B. | https://arxiv.org/abs/2610.36595 | read |
| JevSpawn (arXiv) | 09-30 | Agent with spawned finite action sets | — | Maze 0.96 vs 0.52. The survey's F4 adds: better on 3 of 8 tasks, 1.4–2.1× slower. | https://arxiv.org/abs/2610.00437 | headline (UNVERIFIED) |
| JEVal / InnerJev (arXiv) | 10-02 | Decision models inside long multi-step agents (τ-bench) | Benchmark of 25 configurations | Steps get faster but **success drops 77.6%→66.1%** as errors compound. Their SII-Decision-27B ≈ Jev on JEVal at about 0.1 s per query. | https://arxiv.org/abs/2610.03935 | read |
| SREGym (blog) | 10-07 | Kubernetes root-cause diagnosis with no LLM agent | Code collects the evidence; Jev picks the root cause and supporting observations | **76.2%** (80/105) pass rate, median 14.6 s, at a fraction of the cost of Claude Code or Codex | https://www.sregym.com/blog/jev-driven-sre-diagnosis | read |

### A5. Other domains (shown so the user sees where it did *not* help)

| Source | Date | Use case | Result | URL | Check |
|---|---|---|---|---|---|
| Benchmarking Jev, 37 datasets (arXiv) | 09-29 | General classification | Ahead of a 27B open LLM on 27/37 datasets. Pooled ECE 0.028. | https://arxiv.org/abs/2609.37647 | headline (UNVERIFIED) |
| GLM-5.3-Flash first-token readout (Privatemode blog) | 09-26 | Any LLM as a Jev-like model, by reading the logprob of the option-index token | 28 datasets: GLM ≈ Jev (median gap 0.7 pp, not significant). **Laya 13–15 pp below both.** TREC 6→42 options: Laya 88.4→51.2%. | https://www.privatemode.ai/blog/system-one-from-glm-flash | read |
| LLM-as-Jev (arXiv) | 10-01 | Frozen LLM read through option IDs | A frozen Qwen3.5-4B already matches community Jev-style models. Fine-tuning helps small models and complex routing. | https://arxiv.org/abs/2610.02076 | read |
| JevCascade (arXiv) | 10-06 | How to spend inference compute | Curating the menu beats a bigger model: **0.8B on a curated 5-option menu 95.4%** vs 4B on the full 150 labels 80.0% (CLINC150). A confidence cascade beats re-asking the same model. | https://arxiv.org/abs/2610.07716 | read |
| Luni/laya-jev-benchmark (HF) | 09-19 | Laya vs Jev on one RTX 5090 | Phishing: raw Laya recall **1.2%**; Platt calibration lifts accuracy 0.505→0.611. typed-decisions: Laya without fine-tuning 0.360, fine-tuned **0.767** (vs Jev 0.727). Probability sums for a question and its negation are 1.73 and 0.09. | https://huggingface.co/datasets/Luni/laya-jev-benchmark | read |
| decision-models-under-pressure (GitHub) | 09-25 | Stress test: more candidates, shuffled order, harder distractors | At 128 candidates: Jev 60%, **Laya 39%**. Laya loses 0.073 per doubling of the list (Jev 0.043). | https://github.com/gazelle93/decision-models-under-pressure | read |
| System One Moderation (arXiv) | 10-06 | Content moderation | Retrieved precedents help Jev by **+5 to +22** pts. **Laya barely moves**, and its confidence is less useful for ranking errors. | https://arxiv.org/abs/2610.07953 | read |
| Network traffic, HAR (sensor activity), Medicine, Graph (arXiv) | 09-27 to 10-08 | Numeric or derived inputs | Traffic 28% vs trees 70%. HAR macro-F1 ≤0.12. Medicine −22 pts on diagnostic cases. Graph degree counted correctly 68% of the time. | arXiv 2610.00376 / 2609.36154 / 2609.34024 / 2610.06354 | headline (UNVERIFIED) |
| Speech-neuroprosthesis rescoring (arXiv) | 09-27 | Rescoring speech hypotheses | WER 7.5% vs 7.8% for 7B LLMs, with no GPU | https://arxiv.org/abs/2609.33538 | headline (UNVERIFIED) |
| JevVibe (arXiv) | 09-28 | CWE triage for secure code generation | Top-1 0.46 vs 0.50 for a frontier model, at 1/56 of the cost | https://arxiv.org/abs/2609.34963 | headline (UNVERIFIED) |
| Jev in the Wild (arXiv) | 09-24 | Survey of 2,170 GitHub projects | Uses: judgment 77%, scoring 52%, action selection 31%. Attention concentrates on routing and interface agents. | https://arxiv.org/abs/2609.30216 | read |

---

## (b) Use cases ranked by measured gain

Ranked by the size of the gain over the authors' own baseline. All figures are self-reported, single-team and unreplicated.

1. **Pick the context (page/note selection) in place of embedding RAG.**
   - Sayora: wrong page 33%→3%, end-to-end correct 65%→90%.
   - Unblocked: precision +11.6 pp, recall +13.8 pp.
   - Common finding: it works when the model sees **all candidates at once** (fan-out). Scoring one candidate at a time lost.
2. **Executor under a big planner, and agent coordinator.** These save cost at roughly the same quality.
   - Jev-Mobile: −73% API cost, −5 pp success.
   - REFLEX: −72.7% strong-model calls at 95% success.
   - S1-MAS: −45% to −97% tokens.
   - Learning-to-Formalize: 11.8→52.6% *only after* adding a candidate generator.
3. **Escalation gate for a judge or labeller.**
   - JEV-as-a-Judge: +0.9 pts over GPT-6 at 41% of the fee.
   - CSS annotation: matches the LLM at ¼–½ of the cost.
   - Rubric judges: ≤+2.7 pts. The gain is mostly cost, because errors are correlated.
4. **Reason only when unsure (System 1→2 on one open model).** This is the closest match to a 27B llama.cpp setup.
   - ThinkJev on Qwen3.5-27B: answers 24.5% directly at 3.9% error; 28% of thought tokens saved.
   - Jeeves (9B LoRA): JevBench-hard 0.82 vs Jev 0.73.
   - SanSi: +13.5 pts from looping; as an RL reward judge, 37→52%.
5. **Code search / reranking for coding agents.** Oko: 15–31% faster and 20–42% fewer tokens. kapa: matches the best reranker within 0.003.
6. **LLM / LoRA / model routing.**
   - SeLMRoute: +2.85 pp over the best single model.
   - JevSoup: LoRA expert routing +1.2 pp.
   - Harness Tokenomics: 13–21% spend, emulated only.
   - astgl: Laya gateway +4 of 40 decisions.
   - **Counter-evidence:** zero-shot model routing at chance for both Jev and Laya (Fast Models, Slow Evidence). The routing wins came from a supervised router or a deterministic pre-filter on top of the decision-model features.
7. **Security / guardrail gates.**
   - Usually no better than a fine-tuned DeBERTa (Red Hat; Decision Gates).
   - Hosted Jev is OK on tool-call labels (93–95%).
   - **Laya fails here** (48%, below the 79% constant baseline).
8. **Numeric, sensor, graph or diagnostic reasoning.** Negative or near-zero gain. Classical models or LLMs win.

---

## (c) Recurring failure modes

1. **Accuracy falls with long or similar option lists.**
   - TREC 6→42 options: Laya 88→51%.
   - At 128 candidates: Laya 39%, Jev 60%.
   - With 50 near-neighbour tools: Laya 31%.
   - Near-identical pages get confused.
   - The fix people report: curate or shortlist the menu (JevCascade 95.4% with 5 options; Sayora embedding shortlist→Jev).
2. **Order and name sensitivity.**
   - Laya changes 30% of its answers when option order is reversed.
   - Swapping yes/no flips Jev on 50.5 per 100.
   - The label wins over the written definition: AUROC .81→.58 after a name–rubric swap.
   - A middle "Neutral" option absorbs 51% of errors.
3. **Thresholds and calibration don't transfer.**
   - Thresholds fitted in-sample: a 5% target became up to 17% held-out misses.
   - Thresholds set from point estimates broke in 147 of 190 evaluations.
   - Rejection thresholds give 69% false rejections after transfer.
   - Laya ships under-confident (needs T≈0.47 or a Platt bias term). Jev is overconfident past its knowledge boundary.
4. **It reads, it doesn't compute.** Arithmetic, counting, multi-update state (53–80% at 16 updates), stale summaries, and long trajectories all fail. τ-bench success fell 77.6%→66.1%. The vendor's own advice is to compute in code.
5. **The fallback makes the same mistakes.** LLM judges repeat more than 85% of the decision model's confident errors, so cascades mostly save cost rather than add accuracy.
6. **Inputs can be manipulated.**
   - Universal adversarial suffixes flip 43% of answers, and 90–100% when moved to other tasks.
   - Natural context flips 61% of correct decisions.
   - In 81.5% of real Jev apps the program's decision could be hijacked.
7. **Probabilities are incoherent.** P(X) + P(not X) ≠ 1: off by 0.064 for Jev, and Laya produced sums of 1.73 and 0.09. Two interfaces asked the same question disagree on 32.8% of pairs.
8. **Hosted Jev is nondeterministic.** Up to 3.5% of answers change between identical runs, with about 0.5% day-to-day drift. The open models reproduce exactly.
9. **Base-rate trap and inflated savings.** A 79%-majority constant beats a 75% classifier. A self-audit cut a claimed 23.9% saving to 4.3%.
10. **Latency under load.** Laya p95 went from 143 to 467 ms on a busy host. Hosted Jev misses 100 ms near-real-time deadlines (warm p50 281 ms).

**For the Saluki-27B setup:** the strongest measured gains come from three patterns:
- a decision model choosing among a **shortlisted** set with all candidates visible at once;
- a frozen-threshold **escalate-when-unsure** gate, validated on held-out data, which fits their existing gate;
- **System-1→2 on the same 27B model** (ThinkJev), which reuses the cached prompt.

Zero-shot LoRA or model routing by Jev or Laya alone has weak or chance-level evidence. Use their probabilities as features for a small trained router, which is what SeLMRoute did. Avoid Laya for security labels, and avoid it on lists longer than about 20 options.
