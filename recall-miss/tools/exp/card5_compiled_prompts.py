#!/usr/bin/env python3
"""Card 5 — compiled prompts (DSPy-style) vs hand-written prompts.

docs/experiments/05-compiled-prompts.md. Rerun on the HARD sales-brain task
(2026-10-03): the easy notes gave 100% to every prompt, so a compiled-vs-hand
comparison there is uninformative. Runs in the GPU sandbox (`ollama: true`) and
writes state/exp/05-compiled-prompts.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MODEL = "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"
N_DEV = 30
N_TEST = 60
OUT = "state/exp/05-compiled-prompts.json"
HYPOTHESIS = ("metric-driven search over 8 instruction paraphrases x 3 few-shot sets finds a "
              "prompt >=2 test-F1 above the hand-written prompt in <=100 dev evaluations on a "
              "hard extraction task, with no weight update; the dev-test gap is the real test")
METRIC = "test field micro-F1 (%) over 7 deal fields, dev-test gap, dev evaluations, GPU-minutes"

PARAPHRASES = [
    "You extract deal facts from a sales-call transcript. The call may be in Latvian; copy "
    "numbers as digits and keep enumerated values in English.",
    "Read the call transcript and pull out the requested deal fields. Copy values from the "
    "text; if a field is not stated, leave it out.",
    "You are a careful CRM data-entry assistant. Read the call and output the deal facts as "
    "they were stated, with no paraphrasing.",
    "Turn the sales call into a small JSON record of the deal facts. Numbers must become "
    "digits even when spoken as words.",
    "Extract structured deal facts from the transcript. Never invent a value that is not "
    "supported by the call.",
    "Given a messy bilingual sales call, output the seven deal fields with their literal "
    "values.",
    "You parse call transcripts for a lender. Return the requested fields, resolving any "
    "mid-call correction in favour of the final value.",
    "Your job: read the transcript and copy the seven deal facts into JSON.",
]

SCHEMA = ("Keys: " + ", ".join(C.HARD_FIELDS) + ". Return ONE JSON object and nothing else; "
          "omit keys the transcript does not state.")


def user_text(row):
    return f"Transcript:\n{row['text']}\n\n{SCHEMA}\nJSON:"


def build_messages(instruction, examples, row):
    msgs = [{"role": "system", "content": instruction}]
    for ex in examples:
        msgs.append({"role": "user", "content": user_text(ex)})
        msgs.append({"role": "assistant",
                     "content": json.dumps(ex["gold"], ensure_ascii=False)})
    msgs.append({"role": "user", "content": user_text(row)})
    return msgs


def gen(cli, msgs, num_predict=96):
    r = cli.chat(MODEL, msgs, num_predict=num_predict, temperature=0.0, num_ctx=8192)
    d = C.extract_json(r["text"])
    return (C.hard_fields(d) if d else {}), r


def example_sets(dev, armA_preds):
    set0 = dev[:3]
    mean_len = sum(len(r["text"]) for r in dev) / max(1, len(dev))
    set1 = sorted(dev, key=lambda r: abs(len(r["text"]) - mean_len))[:3]
    wrong = [r for r in dev if C.micro_f1([armA_preds[r["id"]]], [r["gold"]],
                                          C.HARD_FIELDS) < 99.0]
    pool = wrong or dev
    set2 = sorted(pool, key=lambda r: -len(r["text"]))[:3]
    return [set0, set1, set2]


def judge(cli, row, pred):
    msgs = [{"role": "user", "content":
             "You are grading a JSON extraction from a sales call. The transcript is:\n"
             + row["text"] + "\n\nThe extraction is:\n" + json.dumps(pred, ensure_ascii=False)
             + "\n\nDoes the extraction copy the correct " + ", ".join(C.HARD_FIELDS) +
             "? Resolve mid-call corrections to the final value. Reply with only yes or no."}]
    r = cli.chat(MODEL, msgs, num_predict=4, temperature=0.0, num_ctx=8192)
    return "yes" in C.norm_value(r["text"]), r


def main():
    cli = C.Ollama()
    rows, source = C.hard_extraction_fixture(N_DEV + N_TEST)
    dev, test = rows[:N_DEV], rows[N_DEV:]
    cli.unload(MODEL)

    # arm A on dev (also seeds the worst-first example set)
    tps_all, secs_all = [], 0.0
    a_preds = {}
    for row in dev:
        pred, r = gen(cli, build_messages(PARAPHRASES[0], dev[:3], row))
        a_preds[row["id"]] = pred
        tps_all.append(r["eval_tps"])
        secs_all += r["seconds"]
    sets = example_sets(dev, a_preds)
    configs = [(pi, si) for pi in range(len(PARAPHRASES)) for si in range(len(sets))]

    dev_scores, dev_outputs = {}, {}
    for pi, si in configs:
        preds = []
        for row in dev:
            pred, r = gen(cli, build_messages(PARAPHRASES[pi], sets[si], row))
            preds.append(pred)
            tps_all.append(r["eval_tps"])
            secs_all += r["seconds"]
        f1 = C.micro_f1(preds, [r["gold"] for r in dev], C.HARD_FIELDS)
        dev_scores[(pi, si)] = f1
        dev_outputs[(pi, si)] = preds
    a_dev = dev_scores[(0, 0)]

    best = max(configs, key=lambda c: dev_scores[c])
    test_preds = [gen(cli, build_messages(PARAPHRASES[best[0]], sets[best[1]], row))[0]
                  for row in test]
    b_test = C.micro_f1(test_preds, [r["gold"] for r in test], C.HARD_FIELDS)
    a_test_preds = [gen(cli, build_messages(PARAPHRASES[0], sets[0], row))[0] for row in test]
    a_test = C.micro_f1(a_test_preds, [r["gold"] for r in test], C.HARD_FIELDS)

    # arm B': same 24 dev outputs, scored by an LLM judge on a 20-row subsample
    sub = dev[:20]
    judge_scores = {}
    for c in configs:
        ok = 0
        for j, row in enumerate(sub):
            good, _ = judge(cli, row, dev_outputs[c][j])
            ok += int(good)
        judge_scores[c] = 100.0 * ok / len(sub)
    best_j = max(configs, key=lambda c: judge_scores[c])
    bj_test_preds = [gen(cli, build_messages(PARAPHRASES[best_j[0]], sets[best_j[1]], row))[0]
                     for row in test]
    bj_test = C.micro_f1(bj_test_preds, [r["gold"] for r in test], C.HARD_FIELDS)

    peak = cli.vram_gb(MODEL)
    mean_tps = C.mean(tps_all)
    arms = [
        C.arm("A hand-prompt", a_test, mean_tps, peak, dev_f1=round(a_dev, 2),
              test_f1=round(a_test, 2), dev_evals=1),
        C.arm("B compiled (exact dev F1)", b_test, mean_tps, peak,
              dev_f1=round(dev_scores[best], 2), test_f1=round(b_test, 2),
              dev_evals=len(configs), config={"paraphrase": best[0], "example_set": best[1]}),
        C.arm("B' compiled (LLM judge)", bj_test, mean_tps, peak,
              dev_f1=round(dev_scores[best_j], 2), test_f1=round(bj_test, 2),
              dev_evals=len(configs), judge_dev_score=round(judge_scores[best_j], 1),
              config={"paraphrase": best_j[0], "example_set": best_j[1]}),
    ]
    gain = b_test - a_test
    passed = bool(gain >= 2.0 and len(configs) <= 100)
    return N_TEST, arms, passed, {
        "model": MODEL, "fixture": source, "n_dev": N_DEV, "n_test": N_TEST,
        "n_configs": len(configs),
        "rule": "pass if the compiled prompt beats the hand prompt by >=2 test-F1 with "
                "<=100 dev evaluations",
        "test_gain_pts": round(gain, 2), "dev_test_gap_B": round(dev_scores[best] - b_test, 2),
        "judge_replaced_labels": round(judge_scores[best_j], 1),
        "judge_gain_pts": round(bj_test - a_test, 2),
        "note": "hard task rerun 2026-10-03: 7-field deal extraction from sales call transcripts",
    }


if __name__ == "__main__":
    C.guard("05-compiled-prompts", HYPOTHESIS, METRIC, OUT, main)
