# What to do next, for everything

The order follows measured evidence where there is any. "Unknown" means nobody has measured it here.

## Where each thread stands

| Thread | Now | Next | Why this order |
|---|---|---|---|
| Pick-one-of-N cascade | 96.97 in-scope, 84.57 OOS refused (clean) | 1, 2, 3 below | Live as `decide.py`, but only ever tested on CLINC150 |
| Code verifier | HumanEval 94.5 vs 89.6 greedy (8 wins, 0 losses) | 4 | Real gain, but thinking was never run on the full set |
| Math verifier | GSM8K 96.0 vs 94.5 greedy (3 wins, 0 losses, not significant) | 4 | Not proven yet |
| Tool calls | BFCL 88.5; schema check and retry **not built** | 5 | Planned and never started; the lowest of the three accuracy suites |
| Summaries | Best-of-n picked by JEV **not built** | 8 | Unknown gain |
| Router | v1 is 99.4%; v2 not built | Wait | At most 0.6% to gain until two adapters overlap |
| Other decision adapters | Not run | Wait | Kev and Solar need Saluki stopped; Open-Jev needs a custom head |
| Thinking on escalated decisions | Removed | Stop | No better on val or on the reused test split |
| Laya as shortlister | Lost: 80.0 vs 92.3 top-1 | Stop | Embeddings are cheaper and better here |

## Ranked next steps

1. **Run the cascade on a real task of ours.** The original plan said a real classifier, and it ran on CLINC150 instead. Fit thresholds on that task's own validation data, because they do not transfer. This is the only step that shows whether the cascade is useful outside a benchmark.
2. **Lower OOS false accepts.** 108 of 700 OOS messages got through, more than the 91 in-scope mistakes out of 3,000. The thresholds were fitted on only 100 OOS examples, so add CLINC's other OOS examples and use Learn-then-Test thresholds, which were in the plan and never used.
3. **Contrastive option descriptions** for neighbouring intents, such as what separates `user_name` from `change_user_name`. This was the main real-miss pattern in the audit. Tune on CLINC's unused validation messages and score on a new dataset (BANKING77 or HWU64), since no clean CLINC test data remains.
4. **Settle thinking against verifying.** Run thinking, and thinking plus the verifier, on the full 164 and 200 tasks, and compare per task. Today's comparison covers only 40 and 50 tasks. It has thinking ahead on code (92.5 with an adapter vs 90.0) and tied on math (98.0).
5. **Build the tool-call verifier:** validate against the schema, retry on failure, and score on BFCL.
6. **Score on cleaned benchmarks too:** HumanEval+ and GSM8K-Platinum. HumanEval's tests are weak, so some of the verifier's gain may not hold up against stricter tests.
7. **Get a second auditor,** ideally a person. The audit had one reader, and that reader was also the model's operator. Add a sample of the clean run's misses.
8. **Lower priority, unknown gain:** JEV choosing among summaries; picking context instead of embedding retrieval; training our own decision adapter. Start any of these only if steps 1–6 show the picker is the limit.

Serving fix to make alongside step 1: `/v1/decide` needs a short timeout, a fallback and a reserved server slot, because small decisions queue behind long generations.

## About "100%"

100% on these benchmarks is not a valid target, because some labels are wrong. In the audit of a reused split, 8 of 57 misses had wrong labels and 21 more were ambiguous. The honest finish line is: every remaining miss is either fixed or shown by an independent reader to be a bad item. We are not there.

## Rules for all future work

Each of these comes from a mistake made in this project.

1. **Lock away a test set before the first run.** Score it once, with file hashes recorded first. Reading test misses and then rescoring makes that test a tuning set.
2. **Score the code that ships.** Run the shipped code, or show on validation data that it makes the same decisions as the benchmark.
3. **When removing a step, measure every kind of input.** System 2 looked useless on in-scope items, but removing it hurt OOS.
4. **Compare methods on the same tasks.** Different subsets produced a confident wrong conclusion here.
5. **Do not count one-item wins.** Report wins and losses per task, and confidence intervals.
6. **Check an idea's cost before building it.** Measuring router v1 first took minutes and saved building v2.
7. **Never keep results only in a temp folder.** The research catalog was lost that way and had to be regenerated.
8. **Publish corrections in place.** Keep the wrong claim visible, marked as corrected.
