# system1-cascade

Can decision-only "System 1" models (JEV, Laya) pick reliably from many options, and what raises accuracy on reasoning benchmarks without cheating? Measured on one RTX 3090 with local models, October–November 2026.

> **Not independently verified.** One machine, one operator, one auditor. Only the first table below comes from a held-out run. The write-up was written with Claude; see `RESULTS.md` for corrections made along the way.

## Results

**Pick one of 150 intents (CLINC150, clean one-pass test, 3,000 in-scope + 700 out-of-scope):**

| | Score | 95% CI |
|---|---|---|
| In-scope correct | **96.97%** | 96.29–97.52 |
| Out-of-scope refused | **84.57%** | 81.71–87.06 |
| Overall (this mix) | **94.62%** | |
| Baseline: embedding top-1, in-scope | 93.37% | |

When the two models agree, the answer is accepted: that covers 93.2% of in-scope messages, at 98.86% accuracy. The rest are flagged, refused or answered at lower confidence.

**Verifier layer (fixed benchmark subsets, no held-out design split, same-run greedy baseline):**

| Suite | Greedy | Verified | Wins / losses vs greedy |
|---|---|---|---|
| HumanEval (164) | 89.6 | **94.5** | 8 / 0 (p=0.008) |
| GSM8K (200) | 94.5 | **96.0** | 3 / 0 (p=0.25, not significant) |

An earlier standalone greedy run scored 90.9 and 94.0, so the same-run baseline carries about a point of run-to-run noise. Hidden tests and gold answers only score the final pick. Selection uses the prompt's own doctests, model-written tests, votes, and answers computed by running code.

## What we learned

- **Examples beat names.** Showing the 3 nearest training examples per option raised JEV at k=3 from 92.9 to 96.5. Nothing else moved the pick as much.
- **Agreement beats confidence.** Accepting when an embedding and JEV agree in both option orders covers far more than a confidence threshold at similar accuracy.
- **Narrowing helps, but alone adds little:** JEV went from 89.9 at k=16 to 92.9 at k=3, against 92.3 for embeddings alone.
- **Extra thinking did not help the cascade.** Thinking on the disagreements did no better than JEV's own pick, because the hard cases are label boundaries, not reasoning.
- **Laya lost as a shortlister** here: 80.0 top-1 against 92.3 for embeddings.
- **Thinking vs verifying is unsettled.** On the same tasks: math 98.0 each (50 tasks); code 92.5 thinking with an adapter, 85.0 plain thinking, 90.0 verified (40 tasks). Too small to rank them.

Details and corrections: [`RESULTS.md`](RESULTS.md). What to do next: [`FUTURE.md`](FUTURE.md). The research behind the plan: [`research/`](research/README.md).

## How the cascade works (`code/decide.py`)

1. **Shortlist:** embed the text (nomic-embed-text), rank options by mean top-3 cosine to their examples, keep 5.
2. **Out-of-scope check:** refuse if the top score, minus that option's typical score, is under `t_s`, or JEV's "is it one of these?" is under `t_in`.
3. **Pick:** JEV chooses among the 5, shown with their nearest examples, three times: in rank order, reversed, and with "none of these" added.
4. **Decide:** all agree with embedding top-1 → `accepted`. "None" chosen → `refused`. Otherwise JEV's pick → `escalated`, to be reviewed.

Thresholds were fitted on CLINC150 and do not transfer. Refit them on your own held-out data.

## Run it

Needs llama-server with Qwen3.8-27B and the [JEV LoRA](https://huggingface.co/hotdogs/jev_qwen3.8-27B_svd_lora), Ollama with `nomic-embed-text`, and podman for the code sandbox. Then, in a state folder named by `EXPERTS_STATE`:

- `registry.json` and the JEV head + calibration files from `autotrust/JEV-27B-VL`, as listed in `code/experts.py`
- `bench/cascade/report4.json`, copied from `results/`

```bash
python3 code/decide.py "will it be cold tomorrow" options.json   # options: [{"label": "...", "examples": ["...", ...]}, ...]
python3 code/cascade_bench.py prep data_full.json                # CLINC150 from clinc/oos-eval
python3 code/adapter_bench.py run code --verify 5                # after `prepare` on the HumanEval and GSM8K parquet files
```

## What is not here

- **The private serving layer.** `code/experts.py` is a small stand-in with the same calls, and the private service's `POST /v1/decide` route just calls `decide.decide`. `adapter_bench.py` and `jev_decide.py` had private-only parts removed. `decide.py` and `cascade_bench.py` are the exact files that produced the clean score.
- **The router measurement script and its data,** because the prompts are private. Only the numbers are in `RESULTS.md`.
- **Datasets and weights.** Only the few CLINC150 utterances needed to show the misses and audit are included.

CLINC150 is by Larson et al. ([clinc/oos-eval](https://github.com/clinc/oos-eval)), CC BY 3.0.
