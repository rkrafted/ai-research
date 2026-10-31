# Next experiments — looped / latent reasoning on our own work

Each card is the smallest test that could change a decision. All run on the single RTX 3090
(24 GB) and write results under `state/` through a queued `gpu_jobs/*.job.yaml`; a claim only
becomes a result in the provenance ledger, never from this document.

## 1. Do looped checkpoints improve Distill-style extraction at equal VRAM?

**Hypothesis.** If the Ouro paper is right that loops buy *knowledge manipulation* rather than
storage, a 1.4B/2.6B looped model should extract structured fields from messy sales prose at
least as well as a same-VRAM non-looped model, and the gain should grow with the loop count
up to the trained depth (T≈4) and then flatten/regress.

**Smallest test on the 3090.** Reuse the sales-brain synthetic dialogs (`local_batch/` prompts
and the teacher labels already in the repo): take 300 prospect utterances, ask each model for one
JSON object `{intent, product, budget, objection, next_step}` with a frozen prompt, and compare
field-by-field to the teacher labels. Models: Ouro-1.4B at T∈{1,2,4}, Ouro-2.6B at T∈{1,2,4},
Qwen3-1.7B (non-looped) at the same prompt. Greedy decoding, 128 max new tokens, one prompt at a
time so VRAM is comparable.

**Metric.** JSON field micro-F1 (macro over the five fields), schema-valid rate, accuracy vs
peak VRAM (GB) and tokens/s. Pass if any looped run beats the Qwen3-1.7B baseline by ≥2 F1 points
at ≤ the same peak VRAM; fail if the best looped run is within noise (<1 point) or only wins at
T=1.

**GPU-hours.** ~1.5 h (all prefill/decode is short; the 2.6B at T=4 is the slowest cell).

## 2. Can the learned exit gate make looped models viable for interactive agent loops?

**Hypothesis.** The Q-exit threshold `q` can cut the average recurrent depth well below the
maximum with ≤1 point accuracy loss, so a looped model can serve an interactive agent step
(plan → tool choice → argument) at non-looped latency.

**Smallest test on the 3090.** Take 200 agent-style prompts from the job-radar / agent-distill
fixtures (plan + tool selection + one argument). Run Ouro-2.6B-Thinking twice: (a) static
T∈{1,2,4,8}, (b) the learned gate with `q∈{0.5,0.7,0.9,1.0}`, recording the actual exit step per
prompt. Score exact-match on tool name + argument and record mean exit step, p50/p95 wall latency.

**Metric.** Accuracy vs mean exit step (Pareto frontier): pass if some `q` reaches ≥97% of the
static T=4 accuracy at a mean exit step ≤2.0, with p95 latency ≤ the static T=2 latency. Also
report the fraction of prompts that need exactly one step.

**GPU-hours.** ~2 h (generation, not just scoring; cap `max_new_tokens` at 256).

## 3. Does inference-time layer repetition buy anything on models we already have?

**Hypothesis.** A standard small model can capture part of the loop gain by repeating a middle
transformer block at inference (weight tying without pretraining), and that is a cheaper way for
us to test looped-style test-time compute than waiting for more looped checkpoints.

**Smallest test on the 3090.** Qwen3-1.7B with its middle four decoder layers repeated 1/2/4
times (`tools/looped_eval.py` already contains the fallback path that does this). Evaluate on
`benchmark/loop-eval-v1.jsonl` plus a 3-hop multi-fact extraction set from the sales-brain
fixtures. Compare against a CoT-length-matched baseline at the same measured FLOPs/second so the
comparison is compute-fair, not just accuracy-vs-accuracy.

**Metric.** Accuracy per GPU-second and per estimated TFLOP; pass if repeat-2 or repeat-4 beats
the length-matched CoT baseline on the multi-hop set by ≥2 points per unit compute. Fail if
repetition only adds latency without accuracy, which would also bound what we can expect from
looped checkpoints at our scale.

**GPU-hours.** ~2 h.
