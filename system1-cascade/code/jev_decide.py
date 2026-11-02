#!/usr/bin/env python3
"""JEV System-1 decisions on the experts server: the JEV-27B adapter on Saluki.

JEV-27B (autotrust) is Qwen3.8-27B + one LoRA + a 24-slot decision head that reads the
next-token log-probabilities of fixed label tokens ("false"/"true", "0".."5", "A".."P"),
adds a bias and divides by a per-kind temperature. The base weights are stock, so the
LoRA (hotdogs/jev_qwen3.8-27B_svd_lora, an exact copy) loads on Saluki like any expert.
This reproduces serve_decide.py's System-1 path (protocol jev27-bare-v1) over
llama-server's /completion with n_probs; head + calibration come from
state/experts/fetched-ref/jev (autotrust/JEV-27B-VL).

  jev_decide.py decide KIND STATE QUESTION [OPTION ...] [--adapter hub-jev]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experts  # noqa: E402

REF = experts.STATE / "fetched-ref" / "jev"
HEAD = json.loads((REF / "adapter_vllm" / "decision_head.json").read_text())
TEMPS = json.loads((REF / "calibration.json").read_text())["per_kind"]
N_PROBS = 64


def prompt(kind: str, state: str, question: str, options: list[str]) -> tuple[str, list[str], list[float]]:
    """(bare prompt, label tokens, bias) exactly as serve_decide.s1_pass builds them."""
    lo, hi = HEAD["slots"]["ranges"][kind]
    labels = HEAD["slots"]["verbalizers"][lo:hi]
    bias = HEAD["bias"][lo:hi]
    if kind == "choice":
        if not 2 <= len(options) <= hi - lo:
            raise SystemExit(f"choice takes 2-{hi - lo} options")
        labels, bias = labels[:len(options)], bias[:len(options)]
        lines = [f"{lab}) {o}" for lab, o in zip(labels, options)]
    else:
        lines = labels
    text = f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
    return text, labels, bias


def _top(out: dict) -> dict[str, float]:
    """{token text: logprob} of the first generated position (llama-server, old or new shape)."""
    cp = out["completion_probabilities"][0]
    tops = cp.get("top_logprobs") or cp.get("probs") or []
    res = {}
    for t in tops:
        tok = t.get("token", t.get("tok_str", ""))
        lp = t["logprob"] if "logprob" in t else math.log(max(t.get("prob", 0.0), 1e-30))
        res.setdefault(tok.strip(), lp)  # " A" and "A" both read as A; the first (higher) wins
    return res


def decide(kind: str, state: str, question: str, options: list[str], lora: list[dict]) -> dict:
    text, labels, bias = prompt(kind, state, question, options)
    out = experts._http(f"{experts.SERVER}/completion",
                        {"prompt": text, "n_predict": 1, "n_probs": N_PROBS, "temperature": 0,
                         "lora": lora, "post_sampling_probs": False, "cache_prompt": True}, timeout=600)
    lp = _top(out)
    z = [(lp.get(lab, -30.0) + b) / TEMPS[kind] for lab, b in zip(labels, bias)]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    p = [x / sum(e) for x in e]
    k = max(range(len(p)), key=p.__getitem__)
    return {"probabilities": [round(x, 4) for x in p], "choice_index": k,
            "choice": options[k] if kind == "choice" else labels[k], "seen": sum(lab in lp for lab in labels)}


def _lora(spec: str) -> list[dict]:
    reg = experts.load_registry()
    by_name = {x["name"]: x["gguf"] for x in reg["adapters"]}
    live = experts.live_adapters()
    want = {by_name[n]: s for n, s in experts.parse_adapter_spec(spec).items()}
    if missing := set(want) - {x["path"] for x in live}:  # a parked JEV would silently run as bare Saluki
        raise RuntimeError(f"experts server has not loaded {sorted(missing)}")
    return experts.lora_field(live, want)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("decide")
    p.add_argument("kind", choices=("noul", "score", "choice"))
    p.add_argument("state")
    p.add_argument("question")
    p.add_argument("options", nargs="*")
    p.add_argument("--adapter", default="hub-jev")
    a = ap.parse_args(argv)
    print(json.dumps(decide(a.kind, a.state, a.question, a.options, _lora(a.adapter))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
