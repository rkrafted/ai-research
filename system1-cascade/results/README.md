# Results files

| File | What |
|---|---|
| `report-fresh.json` | **The clean score**: frozen `decide.py`, one pass over 3,000 + 700 messages not used for anything else |
| `frozen-before-fresh.sha256` | Hashes taken before that run. They match `code/decide.py` and `code/cascade_bench.py` |
| `report4.json` | The val-based choice of policy and thresholds that was frozen |
| `report.json`, `report2*.json`, `report3*.json` | Earlier stages. Their test split was reused, so they are not clean scores |
| `misses-test*.jsonl` | Misses on that reused split. Contains CLINC150 utterances |
| `audit-test-misses.jsonl` | The hand audit of the 57 in-scope misses: `gold_wrong`, `ambiguous` or `model_miss`. One auditor, Claude |
| `verify-summary.json` | Per-task pass/fail for the verifier and thinking runs |
| `raw/` | Per-item model decisions, without message text, enough to re-run `score4` and `freshscore` |

To re-run scoring without a GPU, copy `raw/*` into `$EXPERTS_STATE/bench/cascade/` and run `cascade_bench.py score4` and `freshscore`. Both were checked to reproduce these reports byte for byte.

CLINC150 is by Larson et al. ([clinc/oos-eval](https://github.com/clinc/oos-eval)), CC BY 3.0. Utterances here are unmodified excerpts.
