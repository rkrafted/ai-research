# Looped Language Models (Ouro) — faithful paper summary

Source: **"Scaling Latent Reasoning via Looped Language Models"**, ByteDance Seed and collaborators.
- Paper: <https://arxiv.org/abs/2510.25741> (read from the arXiv HTML full text, v1, 2025-10)
- Project page / model cards: <https://ouro-llm.github.io/>
- HF paper page: <https://huggingface.co/papers/2510.25741>
- Code: announced "coming soon" on the project page at read time; the four checkpoints are public.

Everything below is extracted from the paper text itself; section/table numbers are the paper's.
Numbers in this file are the paper's numbers, not our measurements. Our own measurements live in
`state/looped-eval.json` and are produced only by `gpu_jobs/looped-eval.job.yaml`.

---

## 1. The one-paragraph claim

A standard Transformer spends a fixed amount of compute per token. LoopLM ("looped language model")
instead reuses **one stack of N weight-tied layers** for **t recurrent steps** per token, so compute
depth becomes a separate scaling axis from parameter count. Ouro trains this at 7.7T tokens,
adds an entropy-regularized exit gate (uniform prior over steps) plus a second gate-only training
stage, and releases 1.4B/2.6B base and Thinking checkpoints. Reported result: the 1.4B matches 4B
dense models and the 2.6B matches 8B dense models on many benchmarks (2–3× parameter efficiency),
and the gain comes from **knowledge manipulation**, not from storing more facts. The authors frame
this as a third scaling axis beyond parameters and data, and explicitly ask the community to
validate extrapolation to deeper depths.

---

## 2. Method (paper §3, Fig. 3, Alg. 1)

### 2.1 Architecture (paper §3.1, Eq. 1)

A normal L-layer Transformer applies distinct layers `Tr_{θ1..θL}`. LoopLM applies the *same*
hidden-layer function `H^L` t times before the LM head:

```
F^(t)(·) = lmhead ∘ H^L ∘ H^L ∘ … ∘ H^L ∘ emb(·)        (t copies; Eq. 1)
```

`F^(1)` is the non-looped model. Each copy is trained with the usual causal next-token loss
`L^(t)` (Eq. 2). The number of applications `t` is the "recurrent step" / "recurrent depth".
Note that the layers are *shared*, so the extra depth adds compute, not parameters.

### 2.2 Adaptive exit gate (paper §3.2, Alg. 1)

At each step `t` a learned linear projection + sigmoid gives a per-token exit *hazard*
`λ_t(x) = σ(Gate(F^(t)(x)))`. The distribution over the exit step is

```
q_φ(i|x) = λ_i(x) · Π_{j<i}(1 − λ_j(x)),   CDF(t|x) = Σ_{i≤t} q_φ(i|x).
```

Inference uses a deterministic Q-exit (Ponder-style) rule: exit at the first `t` with
`CDF(t|x) ≥ q`, else force an exit at `T_max`. The threshold `q` is a **deployment-time knob** for
the compute/accuracy trade-off; no retraining is required to change it.

### 2.3 Stage I — entropy-regularized objective (paper §3.3, Eq. 3)

Training does not stop early: all `T_max` steps are run and each step's loss is computed, then

```
L = Σ_t q_φ(t|x) · L^(t)  −  β · H(q_φ(·|x)).
```

This is an expected task loss plus entropy regularization. The paper shows it is equivalent to an
ELBO with a **uniform prior** `π_t = 1/T_max` (dropping the constant `log T_max`). They argue that
PonderNet's geometric prior and Recurrent Depth's Poisson-lognormal prior bake in a
"prefer shallow" bias and under-train deep steps, hence the uniform prior.

### 2.4 Stage II — focused adaptive gate training (paper §3.4, Eq. 4–5)

After the joint stages, only the gates are trained. For each token, measure the actual loss
improvement from step `t−1` to `t`: `I^(t) = max(0, L_stop^(t−1) − L_stop^(t))`, turn it into an
"ideal continuation" target `w^(t) = σ(50·(I^(t) − 0.005))`, and fit the gate with a weighted
cross-entropy `L_adaptive` (Eq. 5) averaged over steps. This penalizes **underthinking** (stop
when improvement is still large) and **overthinking** (continue when improvement is flat). The
paper reports it buys ~2–3 accuracy points over the untrained entropy-regularized gate.

### 2.5 KV cache (paper §5.4.2, Table 14)

Naively each loop stores its own KV cache (4× memory for 4 loops). Findings:
- **Prefill**: all loops need their own cache; reusing the first loop's cache during prefill costs
  >10 points on GSM8K.
- **Decode**: keeping only the **last** loop's cache costs ~0.1 point on GSM8K (78.85 vs 78.92
  full) and 2.0 points on MATH-500 (80.40 vs 82.40) while cutting memory 4×. Averaged cache is
  similar (78.73 / 78.52). First-step-only collapses (18.73 / 8.43).

### 2.6 Safety/faithfulness claims (paper §7)

- On HEx-PHI (330 prompts, GPT-4o harmfulness judge 1–5), harmfulness *decreases* as recurrent
  steps increase, including at extrapolated steps 5–8. Harmful rate at 4 steps: 0.009
  (1.4B-Thinking), 0.003 (2.6B-Thinking), comparable to Qwen3-4B-Thinking (0.009).
- Latent steps are causally coupled to the final answer: linear probes on layer `24(i−1)` cannot
  predict the step-`i` answer, but probes inside step `i` can; step-to-step agreement is only
  36.1% (A[3,4]) and 44.9% (A[2,3]) on 1,000 Quora Question Pairs, versus a Qwen3-4B-Thinking
  probe that predicts its final answer at 0.99 ROC-AUC (i.e. that model's text CoT barely moves
  the decision). The authors call the LoopLM trace "causally faithful" rather than post hoc.
- Deployment extras: intermediate steps act as a built-in draft model for speculative decoding,
  and the expected next-token loss is non-increasing in `t` ("anytime generation").

---

## 3. Models, data and training (paper §4)

### 3.1 Architecture (Table 1)

| Model | Params | Layers | Hidden | Attention | FFN | Pos. | Vocab |
|---|---|---|---|---|---|---|---|
| Ouro 1.4B | 1.4B | 24 | 2048 | MHA | SwiGLU | RoPE | 49,152 |
| Ouro 2.6B | 2.6B | 48 | 2048 | MHA | SwiGLU | RoPE | 49,152 |

Both use sandwich RMSNorm, MHA (not GQA), SwiGLU, RoPE, and the SmolLM2 49,152-token tokenizer
(Latin/code-oriented, no Chinese tokens). The HF `config.json` for Ouro-2.6B confirms
`num_hidden_layers=48`, `hidden_size=2048`, `num_attention_heads=16`, `head_dim=128`,
`total_ut_steps=4`, `max_position_embeddings=65536`. (The HF model card table says 24 layers for
the 2.6B model; the paper's Table 1 and the shipped config both say 48 — the card is wrong.)

### 3.2 Training pipeline (Fig. 4, Table 5)

7.7T tokens total, in five stages, with a common warmup and Stable Training on 3T tokens, then a
fork: "Keep 1.3B" → Ouro-1.4B and "Upcycle 2.6B" → Ouro-2.6B (the 2.6B upcycles by duplicating
non-embedding layers of the 1.4B, then relaxing the weight ties).

| Stage | Tokens | Seq len | Recurrent steps | KL β | RoPE base | Data focus |
|---|---|---|---|---|---|---|
| Stage 1a Pre-train I | 3T | 4K | 8 | 0.1 | 10K | web high |
| Stage 1b Pre-train II | 3T | 4K | 4 | 0.05 | 10K | web high |
| Stage 2 CT Annealing | 1.4T | 16K | (max 4) | — | 40K | math+code high |
| Stage 3 LongCT | 20B | 64K | (max 4) | — | 1M | long context |
| Stage 4 Mid-training | 300B | 32K | (max 4) | — | 1M | SFT-quality |
| Reasoning SFT | 8.3M examples | 32K | (max 4) | — | 1M | math/code/science/chat |

Optimizer AdamW `(0.9, 0.95)`, WSD LR schedule (3e-4 → 3e-5 → 1e-5), batch 4M→8M tokens,
gradient clip 1.0, weight decay 0.1. The paper says "Throughout our training pipeline, we train the
model with a maximum of 4 recurrent steps". Stability tricks: recurrent steps reduced from 8 to 4
after early instability, batch size scaling, and KL coefficient reduction.

### 3.3 Data (Tables 2–4)

Open-source only. Stage 1 mix (6T tokens): Nemotron-CC 73.4%, MAP-CC 13.0%, Ultra-FineWeb-zh 2.0%,
OpenCoder-pretrain 7.5%, MegaMath-web 4.1%. Stage 2 mix (1.4T): Nemotron-CC-high-quality 66.5%,
Nemotron-CC-Math-v1 15.0%, MegaMath-high-quality 4.6%, OpenCoder annealing 0.5%,
Nemotron synthetic code 3.8%, Nemotron SFT code 3.4%, Nemotron SFT general 6.2%. Stage 3 uses
ProLong 64K (20B). Stage 4 mixes >20 SFT datasets, decontaminated, ChatML format; 90B new + 30B
Stage 1 replay + 180B Stage 2 replay = 300B. Chinese data was dropped after Stage 1 because the
Latin/code tokenizer tokenized it inefficiently.

### 3.4 SFT and RL (paper §4.4–4.5, Table 6)

SFT: 8.3M examples — Math 3.5M, Code 3.2M, Science 808K, Chat 767K; 2 epochs, 32K seq, Adam
LR 2e-5, LlamaFactory. RL (DAPO/GRPO on DAPO-17K) did **not** beat the SFT checkpoint: fixed
4-round RL improved normally but did not surpass SFT; adaptive RL failed, attributed to
infrastructure (vLLM could not do variable-depth rollouts). Worth knowing: RL on this
architecture is an open problem, not a solved recipe.

### 3.5 Compute disclosure — what the paper does NOT say

The paper reports token counts, batch sizes, sequence lengths and schedule stages, but **no GPU
model, GPU count, GPU-hours, FLOPs, or dollar cost** anywhere we could find (searched the full
text for H800/H100/A100/GPU-hours/FLOPs). The only hard compute facts are 7.7T training tokens,
up to 8 recurrent steps in Stage 1a, and the SFT/RL descriptions. Any "compute needed to
reproduce" estimate in this repo is therefore our extrapolation, and is labelled as such.

---

## 4. Results — every table, in brief

### 4.1 Base models (Tables 7–8)

**Table 7 (1.4B vs 1–4B dense).** Ouro-1.4B R4: MMLU 67.35, MMLU-Pro 48.62, BBH 71.02, ARC-C
60.92, HellaSwag 74.29, Winogrande 72.30, GSM8K 78.92, MATH500 82.40, HumanEval 74.40,
HumanEval+ 67.40, MBPP 73.00, MBPP+ 62.70. It beats Qwen3-4B on BBH (71.02 vs 70.95), GSM8K
(78.92 vs 72.86) and MATH500 (82.40 vs 59.60), and loses on MMLU-Pro (48.62 vs 51.40), MMLU
(67.35 vs 73.19), ARC-C (60.92 vs 63.65) and HumanEval (74.40 vs 77.40). So "matches 4B" is
task-dependent and strongest on math/reasoning, weakest on broad knowledge and code.

**Table 8 (2.6B vs 3–12B dense).** Ouro-2.6B R4: MMLU 74.60, MMLU-Pro 55.73, BBH 80.46, ARC-C
66.40, HellaSwag 79.69, Winogrande 75.85, GSM8K 81.58, MATH500 90.85, HumanEval 78.70,
HumanEval+ 70.70, MBPP 80.40, MBPP+ 66.60. It beats Qwen3-8B on MMLU-Pro (55.73 vs 53.72), BBH
(80.46 vs 77.65) and MATH500 (90.85 vs 62.30), and loses on MMLU (74.60 vs 76.63), ARC-C,
HumanEval (78.70 vs 84.80) and MBPP (80.40 vs 79.00 — it wins MBPP, loses MBPP+).

### 4.2 Reasoning models (Table 9)

Ouro-1.4B-Thinking-R4 / Ouro-2.6B-Thinking-R4 (pass@1 unless noted): AIME24 65.0/64.7 (pass@10
83.3/90.0), AIME25 46.3/50.3 (pass@10 73.3/76.7), OlympiadBench 71.6/76.4, BeyondAIME 34.0/39.0,
HLE 5.21/5.58, SuperGPQA 47.4/53.7, GPQA 45.5/52.7. Baselines: Qwen3-4B AIME24 61.3,
OlympiadBench 73.2, BeyondAIME 31.0; Qwen3-8B AIME24 73.0, OlympiadBench 75.3, BeyondAIME 38.0.
Protocol: single in-house harness, LLM-as-judge with fixed rubric, temperature 1.0 / top_p 0.7.

### 4.3 Recurrent depth and extrapolation (Tables 10–13)

**Table 10 (1.4B base).** Steps 1→4 improve, peak at 4, then degrade at 5–8 (trained on 4).
Selected: MMLU 41.21 (T1) → 60.43 (T2) → 66.71 (T3) → **67.45 (T4)** → 66.64 (T5) → 64.49 (T8).
ARC-C 37.63 → … → 60.92 (T4) → 58.19 (T8).

**Table 11 (2.6B base).** Same shape: MMLU 51.55 (T1) → 74.60 (T4) → 74.43 (T5) → 72.24 (T8);
ARC-C 47.95 → 66.38 (T4) → 64.76 (T8).

**Table 12 (1.4B-Thinking).** OlympiadBench 2.22 → 59.70 (T2) → 70.67 (T3) → 71.55 (T4) →
72.30 (T5) → 66.81 (T8); AIME24 0.00 → 37.33 → 62.33 → 65.00 (T4) → 60.67 (T5) → 38.67 (T8).

**Table 13 (2.6B-Thinking).** Peaks at T3–T4 for most tasks; e.g. AIME24 3.00 (T1) → 70.33 (T3) →
64.70 (T4) → 39.00 (T8); OlympiadBench 18.96 → 75.56 (T3) → 76.44 (T4) → 39.26 (T8).

Takeaway: **T=1 is almost useless for reasoning** (near-zero on AIME), loops 2→4 give most of the
gain, and beyond the trained depth performance degrades. The paper's own honest caveat: "hard to
make any strong claims" is the video's phrasing; the paper says performance is optimised for the
depth seen during training and asks for future work on deeper extrapolation.

### 4.4 Early exit (Fig. 5)

On MMLU, at average exit round 2.5: specialized trained gate ~66%, standard pretrained gate ~64%,
hidden-state-difference heuristic close behind, static exit lower. The jump from 1→2 rounds is
large (about 40%→60%) while 3→4 is marginal (→67.35), which is exactly why adaptive exit helps.

### 4.5 Knowledge capacity vs manipulation (paper §6, Fig. 6–7, Table 18/§6.2)

- **Capo (storage)**: synthetic biographies (20K–500K people, 5 attributes each); 1M–40M GPT-2
  style models; both looped and non-looped reach ≈2 bits/parameter. Looping gives **no** extra
  knowledge capacity or capacity scaling.
- **Mano (manipulation)**: modular-arithmetic expression trees without CoT. With fixed params,
  looped `k⊗(12/k)` beats non-looped `k⊗1` for k∈{2,3,6}; e.g. `2⊗6` reaches 98.1/96.3/78.0 at
  L=10/16/24 versus `2⊗1` 21.5/8.4/7.5. It can even match the iso-FLOP `12⊗1` baseline.
- **Multi-hop QA**: looped models need fewer unique samples to hit 100% on 3-hop questions than
  iso-parameter non-looped; gains hold for iso-FLOP too, though `2⊗` is weaker than `24⊗1` at
  100K unique samples.
- **Theory (Thm. 1, §6.3)**: one looped layer can decide graph reachability in `O(log D)` steps
  versus `O(n²)` for discrete CoT and `O(D)` for continuous CoT — a parallelism/sample-efficiency
  argument for why loops help manipulation.

**Important counter-result (Appendix D.1, Table 18).** In a controlled 20B-token FineWeb-Edu
scaling study (53M–1.36B params, ARC/ARC-E/HellaSwag/LAMBADA/OpenBookQA/PIQA), the **standard
non-looped model wins** at equal parameter-and-token budget, and the gap *grows* with recurrent
steps (Table 18: at 1.3B, +0.017 at step 2 and +0.025 at step 4) while shrinking with model size.
The 2–3× headline is at the 7.7T-token frontier; at small scale with matched tokens, looping is a
*compute* win, not a free parameter win. This is the single most important caveat for anyone
planning a small reproduction, including us.

### 4.6 Scaling law (Appendix D–E)

Total loss follows a predictable power law in model size N, data D and max recurrent step T_m
(R² = 0.9596); step-wise loss R² = 0.89/0.81/0.80 for T_m = 2/4/8. Generalizes across held-out
model sizes (avg R² 0.954), data fractions (0.94–0.96) and recurrent steps (0.958). The fitted
parameter γ > 0 means step-wise loss decreases with more recurrent steps. The learned average
exit distribution at T_m=4 is `{0.0004, 0.0855, 0.3793, 0.5348}` — i.e. most mass on steps 3–4,
despite the uniform prior during training.

---

## 5. What this means for one 24 GB GPU (RTX 3090, 30 GB RAM, no cluster)

This is the lens the repo cares about; numbers below are our extrapolation from the shipped
configs, not paper claims.

**Capability per parameter.** If the paper's frontier numbers hold, a 2.6B LoopLM delivers roughly
8B-dense reasoning quality at ~1/3 the weights and ~1/3 the working-set memory. That is the whole
appeal for a single-GPU setup: you can serve a "8B-class" model in the memory budget of a 3B one,
and (in principle) tune quality vs latency with the loop count instead of buying a bigger card.

**VRAM (bf16, Ouro-2.6B: 48 layers, 16 KV heads, head_dim 128).**
- Weights: ~5.2 GB (+ rope/norm, negligible) → trivial on 24 GB.
- KV cache per token per loop: `2 × 16 × 128 × 2 B = 8 KB/layer` → 48 layers = **384 KB per token
  per loop**; 4 loops naively = **1.5 MB/token**. At 8K context that is ~12 GB of cache (fits);
  at 64K context it is ~96 GB (does not fit). Last-step-only reuse (Table 14, ~free in quality)
  divides decode cache by 4, so ~24 GB for 64K naively is still too much at full precision, but
  8K–16K contexts are comfortable and 32K needs sharing plus care.
- Ouro-1.4B (24 layers) halves the cache: 4 loops ≈ 0.77 MB/token.
- QLoRA/SFT of the 2.6B on 24 GB is plausible (unique params are only 2.6B; the loop reuses them,
  so optimizer memory does not grow with loops), but the paper's 7.7T-token pretraining is far
  outside reach. Inference + small SFT is the realistic lane.

**Test-time compute via loops.** The loop count is a free inference dial: T=1 for easy tokens,
T=2–4 for reasoning. The paper's own curve shows most of the quality at T≈3–4 and clear
degradation at T≥5 for the trained checkpoints, so "just loop more" is not a free lunch beyond the
trained depth; adaptive exit (`q`) is the principled way to spend compute. A 3090 can therefore
show a *real* quality-vs-depth curve on a fixed benchmark (that is exactly our GPU job).

**Latency.** LoopLM pays forward passes, not output tokens. Each extra recurrent step is roughly
one more pass through the 24/48-layer stack, so throughput scales ≈ 1/t: a 2.6B at 4 loops costs
about like a 10B dense forward per token, while emitting a normal-length sequence (no CoT token
growth, no context blow-up). Decode KV reuse (last-step-only) keeps memory flat while still paying
the loop compute. Practical expectation on a 3090: usable interactive speeds at T≤4 with bf16 and
short contexts, and a measurable tokens/s drop roughly linear in T.

**Bottom line for our work.** Worth testing, but do not assume "more loops = better". Our job
sweeps T ∈ {1,2,4,8} on the released checkpoints and compares to a same-size non-looped baseline;
the Appendix D.1 counter-result is the hypothesis to falsify locally.

---

## 6. Reproduction plan for this repo

- Checkpoints to fetch (see `fetch/`): `ByteDance/Ouro-1.4B`, `ByteDance/Ouro-2.6B`,
  `ByteDance/Ouro-1.4B-Thinking`, `ByteDance/Ouro-2.6B-Thinking`, plus `Qwen/Qwen3-1.7B` as the
  non-looped ~same-size baseline.
- Benchmark: `benchmark/loop-eval-v1.jsonl` (200 authored multiple-choice items, 100 multi-step
  arithmetic + 100 factual knowledge), generated deterministically by `tools/make_benchmark.py`.
  Chosen because the paper's own base-model tables (Tables 10–11) are multiple-choice scored by
  likelihood, and likelihood scoring needs no decoding/answer-extraction heuristics — the only
  honest option in an offline 240-minute GPU job.
- Evaluation: `tools/looped_eval.py`, queued as `gpu_jobs/looped-eval.job.yaml`, writes
  `state/looped-eval.json` with `{benchmark, n, runs:[{model, loops, accuracy, tokens_per_s,
  vram_gb}]}` at loops 1/2/4/8 plus the Qwen3-1.7B baseline. Only that job may write the file.
- Runtime compats: the checkpoints ship custom `modeling_ouro.py` written for
  `transformers==4.54.x`; the 3090 queue's shared venv is `transformers==5.17`. We verified
  on CPU (random small config) that two shims make the custom code run: set
  `rope_scaling={"rope_type":"linear","factor":1.0}` and `pad_token_id=0` on the config, and wrap
  `create_causal_mask` to translate the removed `input_embeds`/`cache_position` kwargs. The job
  script applies these automatically and falls back to a documented layer-repeat experiment only
  if checkpoint load fails.

## 7. Measured on our RTX 3090 (loop-eval-v1, the result of the plan in §6)

The GPU job in `gpu_jobs/looped-eval.job.yaml` completed (`exit 0`); the full run list is in
`state/looped-eval.json` (n = 200, 9 runs). Length-normalised log-likelihood accuracy over the
200 four-way MCQ items (100 multi-step arithmetic + 100 factual knowledge); throughput counts all
scored token positions, so it is a forward-pass proxy, not decode speed.

| model | loops | acc | arith | knowledge | tok/s | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Ouro-2.6B | 1 | 0.695 | 0.39 | 1.00 | 5868 | 5.53 GB |
| Ouro-2.6B | 2 | 0.830 | 0.66 | 1.00 | 3396 | 5.53 GB |
| Ouro-2.6B | 4 | 0.915 | 0.83 | 1.00 | 1698 | 5.53 GB |
| Ouro-2.6B | 8 | 0.920 | 0.84 | 1.00 | 844 | 5.53 GB |
| Ouro-1.4B | 1 | 0.675 | 0.41 | 0.94 | 12313 | 3.06 GB |
| Ouro-1.4B | 4 | 0.835 | 0.67 | 1.00 | 3286 | 3.06 GB |
| Ouro-2.6B-Thinking | 1 | 0.685 | 0.39 | 0.98 | 6410 | 5.53 GB |
| Ouro-2.6B-Thinking | 4 | 0.890 | 0.78 | 1.00 | 1661 | 5.53 GB |
| Qwen3-1.7B (non-looped) | 1 | 0.800 | 0.62 | 0.98 | 10243 | 3.98 GB |

What this says, and what it does not:

- **Test-time loops work here.** The 2.6B base goes 0.39 → 0.84 arithmetic accuracy from T=1 to
  T=8; the 1.4B at T=4 (0.835 overall) already beats the non-looped same-size Qwen3-1.7B (0.800).
  This is the paper's central pitch reproduced on one 24 GB card with no training.
- **Diminishing returns, and they set in early.** T=4 → T=8 costs 2× compute and buys +0.005
  overall (+0.01 arithmetic). T=4 is the practical knee on this benchmark.
- **Throughput is the price.** tok/s falls ≈ proportionally to T (2.6B: 5868 → 844 at T=8), while
  peak VRAM stays flat (5.53 GB) because the same weights are reused and our scoring uses
  `use_cache=False`. So loops trade latency, not memory.
- **All of it fits 24 GB with huge headroom.** Even bf16 2.6B peaks at 5.5 GB; QLoRA/SFT of the
  looped weights is plausible on a 3090, but the paper's 7.7T-token pretraining is not.
- **Caveat — our knowledge half is saturated.** Every model scores 0.94–1.0 on it, so it
  separates nothing; only the arithmetic half is informative here. A harder knowledge task is the
  first thing to fix in a v2 benchmark (card 3 of `docs/next-experiments.md` reuses this file and
  would inherit that limitation).
- **The Appendix D.1 counter-result is not contradicted by this run.** D.1 compares *training*
  compute at matched small scale; our sweep holds the trained checkpoint fixed and varies
  inference-time depth, which is a different axis.

## 8. Source links used

- Ouro paper: <https://arxiv.org/abs/2510.25741>
- Universal Transformers: <https://arxiv.org/abs/1807.03819>
- PonderNet: <https://arxiv.org/abs/2107.05407>
- Scaling Laws for Neural Language Models: <https://arxiv.org/abs/2001.08361>
- Will we run out of data?: <https://arxiv.org/abs/2211.04325>
- Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?: <https://arxiv.org/abs/2504.13837>
