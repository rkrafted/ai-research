#!/usr/bin/env python3
"""Two-stage narrowing cascade on CLINC150: shortlist k of 150 intents, then JEV-27B picks 1 of k.

Stage 1 shortlisters (both see only the train split):
  emb   nomic-embed-text kNN over the 15,000 train utterances (mean of the top-3 cosine per intent)
  laya  Laya typed-decisions, a narrowing tournament: 10 groups of 15 -> top-2 each -> one round of 20
Stage 2: jev_decide.decide("choice") with the hub-jev adapter on the experts server, k in K.
Arms per k: single (rank order) · double (rank + reversed order, accept when both agree)
            · none (k options + "none of these").
The shortlister, the headline k and the confidence threshold are all chosen on val, then frozen for test.
Out-of-scope items (oos_val / oos_test) should end in "none" or escalation; a confident pick is a false accept.

  cascade_bench.py prep DATA_FULL_JSON | embed | laya | jev val|test | score
Run in ~/theprojects/gpu/.venv (laya needs torch). Outputs: state/bench/cascade/.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experts  # noqa: E402
import jev_decide  # noqa: E402

OUT = experts.STATE / "bench" / "cascade"
LAYA_DIR = Path.home() / ".cache" / "laya" / "typed-decisions"
LAYA_CODE = Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
OLLAMA = "http://127.0.0.1:11434/api/embed"
EMB_MODEL = "nomic-embed-text"
K = (3, 5, 10, 16)
NONE = "none of these"
QUESTION = "Which intent does this user message express?"
TARGET_ERR = 0.05


def _w(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _r(name: str) -> list[dict]:
    return experts._jsonl(OUT / name)


def label(intent: str) -> str:
    return intent.replace("_", " ")


def cmd_prep(a) -> int:
    d = json.loads(Path(a.data).read_text())
    intents = sorted({x[1] for x in d["train"]})
    rng = random.Random(0)
    val = rng.sample(d["val"], 1000) + rng.sample(d["oos_val"], 100)
    by = {}
    for t, i in d["test"]:
        by.setdefault(i, []).append(t)
    test = [[t, i] for i in intents for t in rng.sample(by[i], 10)] + rng.sample(d["oos_test"], 300)
    _w("intents.jsonl", [{"intent": i} for i in intents])
    _w("train.jsonl", [{"text": t, "intent": i} for t, i in d["train"]])
    for name, rows in (("val", val), ("test", test)):
        _w(f"{name}.jsonl", [{"id": f"{name}-{n}", "text": t, "intent": i} for n, (t, i) in enumerate(rows)])
    print(json.dumps({"intents": len(intents), "train": len(d["train"]), "val": len(val), "test": len(test)}))
    return 0


def _embed(texts: list[str]) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), 256):
        body = {"model": EMB_MODEL, "input": ["classification: " + t for t in texts[i:i + 256]]}
        req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            out += json.loads(r.read())["embeddings"]
    return out


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cmd_embed(a) -> int:
    import numpy as np
    intents = [r["intent"] for r in _r("intents.jsonl")]
    train = _r("train.jsonl")
    tr = np.array([_unit(v) for v in _embed([r["text"] for r in train])], dtype=np.float32)
    lab = np.array([intents.index(r["intent"]) for r in train])
    for split in ("val", "test"):
        rows = _r(f"{split}.jsonl")
        q = np.array([_unit(v) for v in _embed([r["text"] for r in rows])], dtype=np.float32)
        sims = q @ tr.T
        for r, s in zip(rows, sims):
            score = [float(np.sort(s[lab == c])[-3:].mean()) for c in range(len(intents))]
            r["rank"] = [intents[c] for c in np.argsort(score)[::-1][:20]]
        _w(f"{split}.emb.jsonl", rows)
        print(split, "embedded", len(rows))
    return 0


def cmd_examples(a) -> int:
    """Precedents: for each of the top-5 candidate intents, its 3 train utterances nearest to the query."""
    import numpy as np
    train = _r("train.jsonl")
    tr = np.array([_unit(v) for v in _embed([r["text"] for r in train])], dtype=np.float32)
    lab = np.array([r["intent"] for r in train])
    for split in ("val", "test"):
        rows = _r(f"{split}.emb.jsonl")
        q = np.array([_unit(v) for v in _embed([r["text"] for r in rows])], dtype=np.float32)
        out = []
        for r, v in zip(rows, q):
            ex = {}
            for c in r["rank"][:5]:
                idx = np.where(lab == c)[0]
                ex[c] = [train[i]["text"] for i in idx[np.argsort(tr[idx] @ v)[::-1][:3]]]
            out.append({"id": r["id"], "ex": ex})
        _w(f"{split}.ex.jsonl", out)
        print(split, "examples", len(out))
    return 0


def with_ex(intent: str, ex: dict | None) -> str:
    if not ex or intent not in ex:
        return label(intent)
    return f'{label(intent)} (like: {"; ".join(chr(34) + t + chr(34) for t in ex[intent])})'


def cmd_embscore(a) -> int:
    """Embedding scores of the top-5 candidates (mean top-3 cosine), the OOS detector's first feature."""
    import numpy as np
    intents = [r["intent"] for r in _r("intents.jsonl")]
    train = _r("train.jsonl")
    tr = np.array([_unit(v) for v in _embed([r["text"] for r in train])], dtype=np.float32)
    lab = np.array([intents.index(r["intent"]) for r in train])
    for split in ("val", "test"):
        rows = _r(f"{split}.emb.jsonl")
        q = np.array([_unit(v) for v in _embed([r["text"] for r in rows])], dtype=np.float32)
        out = []
        for r, s in zip(rows, q @ tr.T):
            out.append({"id": r["id"], "s": [round(float(np.sort(s[lab == intents.index(c)])[-3:].mean()), 4)
                                             for c in r["rank"][:5]]})
        _w(f"{split}.s.jsonl", out)
        print(split, "embscore", len(out))
    return 0


def cmd_normscore(a) -> int:
    """Per-intent OOS score: s minus that intent's typical score (train only, leave-one-out mean top-3
    cosine of its own examples). Intents with diverse wording ("define X") score low in absolute terms."""
    import numpy as np
    intents = [r["intent"] for r in _r("intents.jsonl")]
    train = _r("train.jsonl")
    tr = np.array([_unit(v) for v in _embed([r["text"] for r in train])], dtype=np.float32)
    lab = np.array([intents.index(r["intent"]) for r in train])
    mu = {}
    for c, name in enumerate(intents):
        x = tr[lab == c]
        if len(x) < 4:
            continue
        sim = x @ x.T
        np.fill_diagonal(sim, -1)
        mu[name] = float(np.sort(sim, axis=1)[:, -3:].mean())
    for split in ("val", "test"):
        rank = {r["id"]: r["rank"] for r in _r(f"{split}.emb.jsonl")}
        out = [{"id": r["id"], "s": [round(v - mu.get(c, 0.0), 4) for v, c in zip(r["s"], rank[r["id"]])]}
               for r in _r(f"{split}.s.jsonl")]
        _w(f"{split}.sn.jsonl", out)
    print("normscore: intent mu range", round(min(mu.values()), 3), "-", round(max(mu.values()), 3))
    return 0


IN_SCOPE_Q = "Does this user message express one of the listed intents?"


def cmd_extra(a) -> int:
    """Adds k=5 arms (with precedents) and a JEV noul 'is it one of these intents?' to {split}.jevx.jsonl."""
    path = OUT / f"{a.split}.jevx.jsonl"
    rows = _r(f"{a.split}.jevx.jsonl")
    exs = {r["id"]: r["ex"] for r in _r(f"{a.split}.ex.jsonl")}
    text = {r["id"]: r["text"] for r in _r(f"{a.split}.jsonl")}
    lora = jev_decide._lora("hub-jev")

    def call(t, opts, ex):
        d = jev_decide.decide("choice", t, QUESTION, [with_ex(o, ex) for o in opts], lora)
        return {"pick": opts[d["choice_index"]], "p": max(d["probabilities"])}

    def one(r):
        t, ex = text[r["id"]], exs[r["id"]]
        if "single5" not in r:
            opts = r["rank"][:5]
            r["single5"], r["rev5"], r["none5"] = call(t, opts, ex), call(t, opts[::-1], ex), call(t, opts + [NONE], ex)
        if "p_in" not in r:
            listing = "\n".join(f"- {with_ex(o, ex)}" for o in r["rank"][:5])
            d = jev_decide.decide("noul", f"{t}\n\n[listed intents]\n{listing}", IN_SCOPE_Q, [], lora)
            r["p_in"] = d["probabilities"][1]
        return r

    with ThreadPoolExecutor(2) as ex:
        rows = list(ex.map(one, rows))
    _w(f"{a.split}.jevx.jsonl", rows)
    print(a.split, "extra", len(rows))
    return 0


def agree(r: dict, k: int) -> bool:
    return r[f"single{k}"]["pick"] == r["rank"][0] == r[f"rev{k}"]["pick"] and r[f"none{k}"]["pick"] != NONE


def pipeline(r: dict, k: int, think: dict, s: dict, t_s: float, t_in: float) -> str | None:
    """Final answer: OOS detector -> refuse; agreement gate -> embedding top-1; else System-2 pick."""
    if s[r["id"]][0] < t_s or r["p_in"] < t_in:
        return NONE
    if agree(r, k):
        return r["rank"][0]
    return think[r["id"]]["pick"] if r["id"] in think else r[f"single{k}"]["pick"]


def _overall(rows, k, think, s, t_s, t_in):
    good = [(pipeline(r, k, think, s, t_s, t_in) or NONE) == (NONE if r["intent"] == "oos" else r["intent"]) for r in rows]
    return sum(good) / len(rows)


def cmd_score3(a) -> int:
    """Pre-registered on val: k in {3,5} by JEV single accuracy; (t_s, t_in) by overall correct incl. OOS.
    System 2 decisions exist for k=3 escalations only; a k=5 escalation without one falls back to JEV's pick."""
    val, test = _r("val.jevx.jsonl"), _r("test.jevx.jsonl")
    sf = "sn" if a.norm else "s"
    sv = {r["id"]: r["s"] for r in _r(f"val.{sf}.jsonl")}
    st = {r["id"]: r["s"] for r in _r(f"test.{sf}.jsonl")}
    thv = {r["id"]: r for r in _r("val.thinkx.jsonl")}
    tht = {r["id"]: r for r in _r("test.thinkx.jsonl")}
    ins_v = [r for r in val if r["intent"] != "oos"]
    k = max((3, 5), key=lambda k: sum(r[f"single{k}"]["pick"] == r["intent"] for r in ins_v))
    grid_s = [x / 100 for x in (range(-30, 11) if a.norm else range(40, 91, 2))]
    grid_in = [x / 100 for x in range(0, 96, 5)]
    t_s, t_in = max(((a_, b_) for a_ in grid_s for b_ in grid_in),
                    key=lambda t: (_overall(val, k, thv, sv, *t), -t[0], -t[1]))
    rep = {"k_from_val": k, "t_s_from_val": t_s, "t_in_from_val": t_in, "score": sf}
    for name, rows, th, s in (("val", val, thv, sv), ("test", test, tht, st)):
        ins = [r for r in rows if r["intent"] != "oos"]
        oos = [r for r in rows if r["intent"] == "oos"]
        rep[name] = {
            "jev_single_k3_%": round(100 * sum(r["single3"]["pick"] == r["intent"] for r in ins) / len(ins), 2),
            "jev_single_k5_%": round(100 * sum(r["single5"]["pick"] == r["intent"] for r in ins) / len(ins), 2),
            "in_scope_end_to_end_%": round(100 * sum(pipeline(r, k, th, s, t_s, t_in) == r["intent"] for r in ins) / len(ins), 2),
            "in_scope_refused_%": round(100 * sum(pipeline(r, k, th, s, t_s, t_in) == NONE for r in ins) / len(ins), 2),
            "oos_refused_%": round(100 * sum(pipeline(r, k, th, s, t_s, t_in) in (NONE, None) for r in oos) / len(oos), 2),
            "overall_correct_incl_oos_%": round(100 * _overall(rows, k, th, s, t_s, t_in), 2)}
    (OUT / f"report3{'n' if a.norm else ''}.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


def _policy(r, s, th, policy, t_s, t_in, k=5):
    """What happens to an item the agreement gate did not accept differs by policy."""
    if s[r["id"]][0] < t_s or r["p_in"] < t_in:
        return NONE
    if agree(r, k):
        return r["rank"][0]
    if policy == "think" and r["id"] in th:
        return th[r["id"]]["pick"]
    if policy == "none-refuses" and r[f"none{k}"]["pick"] == NONE:
        return NONE
    return r[f"single{k}"]["pick"]


def _pstats(rows, s, th, policy, t_s, t_in):
    def right(r):
        return _policy(r, s, th, policy, t_s, t_in) == (NONE if r["intent"] == "oos" else r["intent"])
    ins = [r for r in rows if r["intent"] != "oos"]
    oos = [r for r in rows if r["intent"] == "oos"]
    return {"in_scope_%": round(100 * sum(map(right, ins)) / len(ins), 2),
            "oos_refused_%": round(100 * sum(map(right, oos)) / len(oos), 2),
            "overall_%": round(100 * sum(map(right, rows)) / len(rows), 2)}


def cmd_score4(a) -> int:
    """Policy for items the gate does not accept, fitted on val: System 2 / JEV's pick / refuse on "none".
    Ties in val overall go to the policy that needs no thinking and refuses most OOS."""
    val, test = _r("val.jevx.jsonl"), _r("test.jevx.jsonl")
    sv = {r["id"]: r["s"] for r in _r("val.sn.jsonl")}
    st = {r["id"]: r["s"] for r in _r("test.sn.jsonl")}
    thv = {r["id"]: r for r in _r("val.thinkx.jsonl")}
    tht = {r["id"]: r for r in _r("test.thinkx.jsonl")}
    gs, gi = [x / 100 for x in range(-30, 11)], [x / 100 for x in range(0, 96, 5)]
    rep = {"policies": {}}
    for pol in ("think", "jev-pick", "none-refuses"):
        t = max(((x, y) for x in gs for y in gi), key=lambda t: (_pstats(val, sv, thv, pol, *t)["overall_%"], -t[0], -t[1]))
        rep["policies"][pol] = {"t_s": t[0], "t_in": t[1], "val": _pstats(val, sv, thv, pol, *t), "test": _pstats(test, st, tht, pol, *t)}
    no_think = {p: v for p, v in rep["policies"].items() if p != "think"}
    best = max(no_think, key=lambda p: (no_think[p]["val"]["overall_%"], no_think[p]["val"]["oos_refused_%"]))
    rep.update(policy=best, k_from_val=5, t_s_from_val=no_think[best]["t_s"], t_in_from_val=no_think[best]["t_in"])
    (OUT / "report4.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


def cmd_prepfresh(a) -> int:
    """Everything prep did not sample from CLINC's test + oos_test, minus any text val or test contain."""
    d = json.loads(Path(a.data).read_text())
    used = {r["text"] for r in _r("val.jsonl") + _r("test.jsonl")}
    rows = [[t, i] for t, i in d["test"] + d["oos_test"] if t not in used]
    _w("fresh.jsonl", [{"id": f"fresh-{n}", "text": t, "intent": i} for n, (t, i) in enumerate(rows)])
    print("fresh", len(rows), "oos", sum(i == "oos" for _, i in rows))
    return 0


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p, d = k / n, 1 + z * z / n
    c, w = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - w), 2), round(100 * (c + w), 2)]


def cmd_fresh(a) -> int:
    """One pass of the frozen decide.py over data nothing was tuned on. Finished ids are kept, so a crash
    resumes instead of restarting; config changes must not be followed by another scored run."""
    import threading
    import decide as dc
    train = _r("train.jsonl")
    by = {}
    for r in train:
        by.setdefault(r["intent"], []).append(r["text"])
    opts = [{"label": label(i), "examples": ex, "intent": i} for i, ex in sorted(by.items())]
    path = OUT / "fresh.decide.jsonl"
    done = {r["id"] for r in _r("fresh.decide.jsonl")} if path.exists() else set()
    rows = [r for r in _r("fresh.jsonl") if r["id"] not in done]
    dc.shortlist("warm up the option cache", opts)
    lock, n = threading.Lock(), [0]

    def one(r):
        out = dc.decide(r["text"], opts)
        with lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**r, **out}) + "\n")
            n[0] += 1
            if n[0] % 200 == 0:
                print("fresh", n[0], "of", len(rows), flush=True)

    with ThreadPoolExecutor(2) as ex:
        list(ex.map(one, rows))
    return cmd_freshscore(a)


def cmd_freshscore(a) -> int:
    rows = _r("fresh.decide.jsonl")
    ins = [r for r in rows if r["intent"] != "oos"]
    oos = [r for r in rows if r["intent"] == "oos"]
    gold = lambda r: label(r["intent"])  # noqa: E731
    right = sum(r["choice"] == gold(r) for r in ins)
    ref = sum(r["choice"] is None for r in oos)
    by = {s: [r for r in ins if r["status"] == s] for s in ("accepted", "escalated", "refused")}
    rep = {"n_in_scope": len(ins), "n_oos": len(oos),
           "embedding_top1_%": round(100 * sum(r["candidates"][0] == gold(r) for r in ins) / len(ins), 2),
           "recall_at_k_%": round(100 * sum(gold(r) in r["candidates"] for r in ins) / len(ins), 2),
           "in_scope_%": round(100 * right / len(ins), 2), "in_scope_ci95": _wilson(right, len(ins)),
           "in_scope_refused_%": round(100 * len(by["refused"]) / len(ins), 2),
           "oos_refused_%": round(100 * ref / len(oos), 2), "oos_refused_ci95": _wilson(ref, len(oos)),
           "overall_%": round(100 * (right + ref) / len(rows), 2),
           "accepted_share_of_in_scope_%": round(100 * len(by["accepted"]) / len(ins), 2),
           "accepted_right_%": round(100 * sum(r["choice"] == gold(r) for r in by["accepted"]) / max(1, len(by["accepted"])), 2),
           "escalated_share_of_in_scope_%": round(100 * len(by["escalated"]) / len(ins), 2),
           "escalated_right_%": round(100 * sum(r["choice"] == gold(r) for r in by["escalated"]) / max(1, len(by["escalated"])), 2),
           "oos_accepted_by_gate_%": round(100 * sum(r["status"] == "accepted" for r in oos) / len(oos), 2)}
    (OUT / "report-fresh.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


def cmd_misses(a) -> int:
    """Test misses of the final pipeline, for the hand audit (writes misses-test.jsonl)."""
    n = "n" if a.norm else ""
    rep = json.loads((OUT / f"report3{n}.json").read_text())
    rows = _r("test.jevx.jsonl")
    st = {r["id"]: r["s"] for r in _r(f"test.s{n}.jsonl")}
    th = {r["id"]: r for r in _r("test.thinkx.jsonl")}
    text = {r["id"]: r["text"] for r in _r("test.jsonl")}
    out = []
    for r in rows:
        f = pipeline(r, rep["k_from_val"], th, st, rep["t_s_from_val"], rep["t_in_from_val"]) or NONE
        gold = NONE if r["intent"] == "oos" else r["intent"]
        if f != gold:
            out.append({"id": r["id"], "text": text[r["id"]], "gold": r["intent"], "pred": f, "top5": r["rank"][:5]})
    _w(f"misses-test{n}.jsonl", out)
    print("misses", len(out))
    return 0


def cmd_laya(a) -> int:
    sys.path.insert(0, str(LAYA_CODE))
    from rl_agent_api import RLAgent
    agent = RLAgent(str(LAYA_DIR))
    intents = [r["intent"] for r in _r("intents.jsonl")]
    groups = [intents[i:i + 15] for i in range(0, len(intents), 15)]

    def ask(state, options_by_q):
        qs = {str(n): {"type": "choice", "instructions": QUESTION, "criteria": [label(o) for o in opts]}
              for n, opts in enumerate(options_by_q)}
        ans = agent.system_one(state, qs)["answers"]
        return [{o: ans[str(n)]["probabilities"][label(o)] for o in opts} for n, opts in enumerate(options_by_q)]

    for split in ("val", "test"):
        rows = _r(f"{split}.jsonl")
        for r in rows:
            first = ask(r["text"], groups)
            keep = [o for p in first for o in sorted(p, key=p.get, reverse=True)[:2]]
            final = ask(r["text"], [keep])[0]
            r["rank"] = sorted(final, key=final.get, reverse=True)
        _w(f"{split}.laya.jsonl", rows)
        print(split, "laya", len(rows))
    return 0


def recall(rows: list[dict], k: int) -> float:
    ins = [r for r in rows if r["intent"] != "oos"]
    return sum(r["intent"] in r["rank"][:k] for r in ins) / len(ins)


def pick_shortlister() -> str:
    """Pre-registered: the shortlister with the higher mean val recall over K feeds stage 2."""
    score = {s: sum(recall(_r(f"val.{s}.jsonl"), k) for k in K) / len(K) for s in ("emb", "laya")}
    return max(score, key=score.get)


def cmd_jev(a) -> int:
    src = pick_shortlister()
    rows = _r(f"{a.split}.{src}.jsonl")
    lora = jev_decide._lora("hub-jev")
    sfx = "x" if a.examples else ""  # x = every option carries its 3 nearest train precedents
    exs = {r["id"]: r["ex"] for r in _r(f"{a.split}.ex.jsonl")} if a.examples else {}
    done_path = OUT / f"{a.split}.jev{sfx}.jsonl"
    done = {r["id"]: r for r in experts._jsonl(done_path)} if done_path.exists() else {}

    def call(text, opts, ex=None):
        d = jev_decide.decide("choice", text, QUESTION, [with_ex(o, ex) for o in opts], lora)
        return {"pick": opts[d["choice_index"]], "p": max(d["probabilities"])}

    def one(r):
        if r["id"] in done:
            return done[r["id"]]
        res = {"id": r["id"], "intent": r["intent"], "src": src, "rank": r["rank"][:16]}
        ex = exs.get(r["id"])
        for k in ((3,) if a.examples else K):
            opts = r["rank"][:k]
            res[f"single{k}"] = call(r["text"], opts, ex)
            res[f"rev{k}"] = call(r["text"], opts[::-1], ex)
            if k < 16:
                res[f"none{k}"] = call(r["text"], opts + [NONE], ex)
        with done_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
        return res

    with ThreadPoolExecutor(2) as ex:
        out = list(ex.map(one, rows))
    print(a.split, "jev", len(out), "shortlister", src)
    return 0


def agree3(r: dict) -> bool:
    """The agreement gate (found on test, confirmed on val): embedding top-1 = JEV = JEV reversed, not none."""
    return r["single3"]["pick"] == r["rank"][0] == r["rev3"]["pick"] and r["none3"]["pick"] != NONE


THINK_PROMPT = """A user sent this message to a banking / travel / home / work assistant:

"{text}"

Which intent does the message express? Exactly one option is right.
{options}

Answer with the option letter on the last line as: Answer: X"""


def cmd_think(a) -> int:
    """System 2: Saluki with thinking (+ thinkingcap) decides every item the agreement gate escalated."""
    sfx = "x" if a.examples else ""
    rows = [r for r in _r(f"{a.split}.jev{sfx}.jsonl") if not agree3(r)]
    text = {r["id"]: r["text"] for r in _r(f"{a.split}.jsonl")}
    exs = {r["id"]: r["ex"] for r in _r(f"{a.split}.ex.jsonl")} if a.examples else {}
    lora = jev_decide._lora("hub-thinkingcap")
    done_path = OUT / f"{a.split}.think{sfx}.jsonl"
    done = {r["id"] for r in experts._jsonl(done_path)} if done_path.exists() else set()

    def one(r):
        if r["id"] in done:
            return
        opts = r["rank"][:3] + [NONE]
        letters = "ABCD"
        body = {"messages": [{"role": "user", "content": THINK_PROMPT.format(
                    text=text[r["id"]], options="\n".join(f"{l}) {with_ex(o, exs.get(r['id']))}" for l, o in zip(letters, opts)))}],
                "temperature": 0, "max_tokens": 6000, "lora": lora,
                "chat_template_kwargs": {"enable_thinking": True}}
        out = experts._http(f"{experts.SERVER}/v1/chat/completions", body, timeout=1200)
        msg = out["choices"][0]["message"]["content"] or ""
        m = re.findall(r"Answer:\s*\**\s*([ABCD])", msg.split("</think>")[-1])
        pick = opts[letters.index(m[-1])] if m else None
        res = {"id": r["id"], "intent": r["intent"], "pick": pick, "tokens": out["usage"]["completion_tokens"]}
        with done_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")

    with ThreadPoolExecutor(2) as ex:
        list(ex.map(one, rows))
    print(a.split, "think", len(rows))
    return 0


def cmd_score2(a) -> int:
    """End to end: agreement-accepted -> embedding top-1, escalated -> System-2 pick (none = refuse)."""
    sfx = "x" if a.examples else ""
    rows = _r(f"{a.split}.jev{sfx}.jsonl")
    think = {r["id"]: r for r in _r(f"{a.split}.think{sfx}.jsonl")}
    ins = [r for r in rows if r["intent"] != "oos"]
    oos = [r for r in rows if r["intent"] == "oos"]

    def final(r):
        return r["rank"][0] if agree3(r) else think[r["id"]]["pick"]

    esc_in = [r for r in ins if not agree3(r)]
    rep = {"split": a.split,
           "in_scope_end_to_end_%": round(100 * sum(final(r) == r["intent"] for r in ins) / len(ins), 2),
           "baseline_emb_top1_%": round(100 * sum(r["rank"][0] == r["intent"] for r in ins) / len(ins), 2),
           "baseline_jev_k3_%": round(100 * sum(r["single3"]["pick"] == r["intent"] for r in ins) / len(ins), 2),
           "gate_coverage_%": round(100 * (len(ins) - len(esc_in)) / len(ins), 2),
           "gate_acc_on_accepted_%": round(100 * sum(r["rank"][0] == r["intent"] for r in ins if agree3(r)) / max(1, len(ins) - len(esc_in)), 2),
           "oos_gate_false_accept_%": round(100 * sum(agree3(r) for r in oos) / len(oos), 2),
           "escalated_in_scope": len(esc_in),
           "system2_acc_on_escalated_%": round(100 * sum(think[r["id"]]["pick"] == r["intent"] for r in esc_in) / max(1, len(esc_in)), 2),
           "system2_said_none_on_in_scope_%": round(100 * sum(think[r["id"]]["pick"] == NONE for r in esc_in) / max(1, len(esc_in)), 2),
           "oos_refused_%": round(100 * sum(final(r) in (NONE, None) for r in oos) / len(oos), 2),
           "overall_correct_incl_oos_%": round(100 * (sum(final(r) == r["intent"] for r in ins)
                                                     + sum(final(r) in (NONE, None) for r in oos)) / len(rows), 2),
           "unparsed": sum(t["pick"] is None for t in think.values()),
           "mean_think_tokens": round(sum(t["tokens"] for t in think.values()) / max(1, len(think)))}
    (OUT / f"report2{sfx}-{a.split}.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


def _cp_upper(err: int, n: int, alpha: float = 0.05) -> float:
    """Clopper-Pearson upper bound by bisection on the binomial tail (no scipy)."""
    if n == 0:
        return 1.0
    if err == n:
        return 1.0
    lo, hi = err / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        tail = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(err + 1))
        lo, hi = (mid, hi) if tail > alpha else (lo, mid)
    return hi


def accepted(r: dict, k: int, t: float) -> str | None:
    """The gated cascade: both orders agree, confidence >= t, and the none-arm did not pick none."""
    s, v = r[f"single{k}"], r[f"rev{k}"]
    if s["pick"] != v["pick"] or min(s["p"], v["p"]) < t:
        return None
    if k < 16 and r[f"none{k}"]["pick"] == NONE:
        return None
    return s["pick"]


def fit_threshold(rows: list[dict], k: int) -> float:
    """Learn-then-Test style fixed-sequence scan from high to low: stop at the first t whose
    Clopper-Pearson upper bound on accepted error exceeds TARGET_ERR (oos picks count as errors)."""
    best = 1.01
    for t in [x / 100 for x in range(99, 19, -1)]:
        acc = [(accepted(r, k, t), r["intent"]) for r in rows]
        acc = [(p, g) for p, g in acc if p is not None]
        err = sum(p != g for p, g in acc)
        if _cp_upper(err, len(acc)) > TARGET_ERR:
            break
        best = t
    return best


def cmd_score(a) -> int:
    val, test = _r("val.jev.jsonl"), _r("test.jev.jsonl")
    src = val[0]["src"]
    rep = {"shortlister": src, "stage1": {}, "stage2_test": {}}
    for s in ("emb", "laya"):
        for split in ("val", "test"):
            rows = _r(f"{split}.{s}.jsonl")
            ins = [r for r in rows if r["intent"] != "oos"]
            rep["stage1"][f"{s}.{split}"] = {"top1": round(100 * sum(r["rank"][0] == r["intent"] for r in ins) / len(ins), 2),
                                             **{f"recall@{k}": round(100 * recall(rows, k), 2) for k in K}}

    def acc(rows, key):
        ins = [r for r in rows if r["intent"] != "oos"]
        return round(100 * sum(r[key]["pick"] == r["intent"] for r in ins) / len(ins), 2)

    k_star = max(K, key=lambda k: acc(val, f"single{k}"))
    t = fit_threshold(val, k_star)
    rep["k_star_from_val"], rep["threshold_from_val"] = k_star, t
    for k in K:
        rep["stage2_test"][f"k={k}"] = {"single": acc(test, f"single{k}"), "reversed": acc(test, f"rev{k}"),
                                        "order_flip_%": round(100 * sum(r[f"single{k}"]["pick"] != r[f"rev{k}"]["pick"] for r in test) / len(test), 2)}
    ins = [r for r in test if r["intent"] != "oos"]
    oos = [r for r in test if r["intent"] == "oos"]
    picks = [(accepted(r, k_star, t), r) for r in ins]
    acc_n = [p for p, r in picks if p is not None]
    fallback = [p if p is not None else r["rank"][0] for p, r in picks]  # escalation -> stage-1 top-1
    rep["gated_test"] = {
        "coverage_%": round(100 * len(acc_n) / len(ins), 2),
        "accuracy_on_accepted_%": round(100 * sum(p == r["intent"] for p, r in picks if p is not None) / max(1, len(acc_n)), 2),
        "end_to_end_with_stage1_fallback_%": round(100 * sum(f == r["intent"] for f, r in zip(fallback, ins)) / len(ins), 2),
        "oos_false_accept_%": round(100 * sum(accepted(r, k_star, t) is not None for r in oos) / len(oos), 2),
        "n_in_scope": len(ins), "n_oos": len(oos)}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep").add_argument("data")
    sub.add_parser("embed")
    sub.add_parser("laya")
    sub.add_parser("examples")
    sub.add_parser("embscore")
    sub.add_parser("extra").add_argument("split", choices=("val", "test"))
    sub.add_parser("normscore")
    sub.add_parser("score4")
    sub.add_parser("prepfresh").add_argument("data")
    sub.add_parser("fresh")
    sub.add_parser("freshscore")
    for c in ("score3", "misses"):
        sub.add_parser(c).add_argument("--norm", action="store_true", help="per-intent normalised OOS score")
    for c in ("jev", "think", "score2"):
        s = sub.add_parser(c); s.add_argument("split", choices=("val", "test")); s.add_argument("--examples", action="store_true")
    sub.add_parser("score")
    a = ap.parse_args(argv)
    return globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
