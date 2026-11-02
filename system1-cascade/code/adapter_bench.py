#!/usr/bin/env python3
"""Capability benchmarks on a llama-server base model, plus --verify (test-time verification).

Fixed subsets picked by id hash, temperature 0, the same
request shape for every adapter, so two runs of one spec give the same number.

  code   HumanEval, 164 tasks; the reply's code runs against the tests inside a podman
         container (python:3.12-slim, --network none) -> pass@1
  math   GSM8K test, 200 problems; last "Answer: <number>" vs the gold number -> accuracy
  summ   CNN/DailyMail 3.0.0 test, 100 articles; 3-4 sentence summary vs the highlights ->
         ROUGE-1/2/L F1 (word level, lowercase)

  adapter_bench.py prepare                     parquet -> bench/*.jsonl (run with the GPU venv: pyarrow)
  adapter_bench.py run SUITE [--adapter SPEC] [--n N] [--think] [--variant caveman] [--verify K]

SPEC is the experts spec: none | NAME | NAME:0.5 | a:0.7,b:0.3.

--verify K (code, math): test-time verification that never sees the hidden tests or gold answers.
  code  greedy + K-1 samples (T=0.8); keep the first that passes the doctest examples printed in
        the prompt itself; if none pass, one self-repair round with the failing output -> pass@1
        of the selected program on the hidden tests
  math  majority vote over K samples (T=0.7) + one program-of-thought answer run in the sandbox
"""

from __future__ import annotations

import argparse
import collections
import doctest
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experts  # noqa: E402

BENCH = experts.STATE / "bench"
RESULTS = BENCH / "results"
RESULTS_LOG = BENCH / "results.jsonl"
SANDBOX_IMAGE = "docker.io/library/python:3.12-slim"
SIZES = {"code": 164, "math": 200, "summ": 100}


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def prepare() -> dict:
    import pyarrow.parquet as pq

    def rows(glob: str) -> list[dict]:
        out = []
        for p in sorted(BENCH.glob(glob)):
            out += pq.read_table(p).to_pylist()
        return out

    he = [{"id": r["task_id"], "prompt": r["prompt"], "test": r["test"], "entry": r["entry_point"]}
          for r in rows("humaneval/**/test-*.parquet")]
    gsm = [{"id": f"gsm8k/test/{i}", "question": r["question"],
            "gold": r["answer"].split("####")[-1].strip().replace(",", "")}
           for i, r in enumerate(rows("gsm8k/main/test-*.parquet"))]
    gsm_train = [{"id": f"gsm8k/train/{i}", "question": r["question"], "answer": r["answer"]}
                 for i, r in enumerate(rows("gsm8k/main/train-*.parquet"))]
    cnn = [{"id": r["id"], "article": r["article"], "highlights": r["highlights"]}
           for r in rows("cnndm/3.0.0/test-*.parquet")]
    cnn_train = [{"id": r["id"], "article": r["article"], "highlights": r["highlights"]}
                 for r in rows("cnndm/3.0.0/train-*.parquet")]
    out = {"code": he, "math": gsm, "summ": cnn, "math_train": gsm_train, "summ_train": cnn_train}
    for k, v in out.items():
        v.sort(key=lambda r: _h(r["id"]))
        if k in SIZES:
            v[:] = v[:SIZES[k]]
        elif k == "summ_train":
            v[:] = v[:2000]
        _write_jsonl(BENCH / f"{k}.jsonl", v)
    return {k: len(v) for k, v in out.items()}


def tasks(suite: str, n: int | None = None) -> list[dict]:
    rows = experts._jsonl(BENCH / f"{suite}.jsonl")
    return rows[:n] if n else rows


# --------------------------------------------------------------------------
# prompts and scoring (pure: exercised by the selftest)
# --------------------------------------------------------------------------


def code_messages(t: dict) -> list[dict]:
    return [{"role": "user", "content":
             "Complete this Python function. Reply with the complete function, including any "
             "imports it needs, in one ```python code block.\n\n```python\n" + t["prompt"] + "```"}]


def math_messages(t: dict) -> list[dict]:
    return [{"role": "user", "content":
             t["question"] + "\n\nSolve it step by step, then end with a final line "
             "'Answer: <number>'."}]


def summ_messages(t: dict) -> list[dict]:
    return [{"role": "user", "content":
             "Summarize the following news article in 3 to 4 sentences.\n\n" + t["article"][:6000]}]


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len)
    return text


def code_program(t: dict, reply: str) -> str:
    """The prompt first (its imports and helpers), then the model's function, then the tests.
    A reply that is only a body (no `def entry`) is appended to the prompt as its completion."""
    code = extract_code(reply)
    if re.search(rf"def\s+{re.escape(t['entry'])}\s*\(", code):
        body = t["prompt"] + "\n    pass\n\n" + code
    else:
        body = t["prompt"] + code
    return body + "\n\n" + t["test"] + f"\n\ncheck({t['entry']})\n"


_NUM = r"-?\$?\s*-?[\d,]*\.?\d+"


def math_pred(text: str) -> str | None:
    m = re.findall(r"Answer\s*[:：]\s*\**\s*(" + _NUM + ")", text, flags=re.I)
    if not m:
        m = re.findall(_NUM, text)
    if not m:
        return None
    return re.sub(r"[\s$,]", "", m[-1]).rstrip(".")


def math_ok(pred: str | None, gold: str) -> bool:
    try:
        return pred is not None and abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def _words(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def _f1(overlap: int, a: int, b: int) -> float:
    if not overlap or not a or not b:
        return 0.0
    p, r = overlap / a, overlap / b
    return 2 * p * r / (p + r)


def _ngrams(ws: list[str], n: int) -> dict:
    out: dict = {}
    for i in range(len(ws) - n + 1):
        g = tuple(ws[i:i + n])
        out[g] = out.get(g, 0) + 1
    return out


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge(pred: str, ref: str) -> dict:
    p, r = _words(pred), _words(ref)
    out = {}
    for n in (1, 2):
        pg, rg = _ngrams(p, n), _ngrams(r, n)
        ov = sum(min(c, rg.get(g, 0)) for g, c in pg.items())
        out[f"r{n}"] = _f1(ov, sum(pg.values()), sum(rg.values()))
    out["rL"] = _f1(_lcs(p, r), len(p), len(r))
    return out


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


def _lora(spec: str) -> list[dict]:
    reg = experts.load_registry()
    by_name = {x["name"]: x["gguf"] for x in reg["adapters"]}
    scales = experts.parse_adapter_spec(spec)
    live = experts.live_adapters()
    missing = [n for n in scales if by_name.get(n) not in {a["path"] for a in live}]
    if missing:
        raise SystemExit(f"not loaded on the server: {missing} (start llama-server with it)")
    return experts.lora_field(live, {by_name[n]: s for n, s in scales.items()})


def _chat(messages: list[dict], lora: list[dict], think: bool, max_tokens: int,
          temperature: float = 0, seed: int | None = None) -> dict:
    body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens, "lora": lora,
            "chat_template_kwargs": {"enable_thinking": think}, **({"seed": seed} if seed is not None else {})}
    t0 = time.monotonic()
    try:
        out = experts._http(f"{experts.SERVER}/v1/chat/completions", body, timeout=1800)
        msg = out["choices"][0]["message"]
        return {"text": msg.get("content") or "", "tokens": out.get("usage", {}).get("completion_tokens", 0),
                "ms": round((time.monotonic() - t0) * 1000), "error": None}
    except Exception as e:
        return {"text": "", "tokens": 0, "ms": round((time.monotonic() - t0) * 1000),
                "error": f"{type(e).__name__}: {e}"[:200]}


def sandbox_run(programs: dict[str, str], timeout: int = 10, detail: bool = False) -> dict:
    """Run each program in its own python process inside one throwaway container, no network.
    -> {key: ok} or, with detail, {key: [ok, stdout tail, stderr tail]}."""
    runner = (
        "import json,subprocess,sys\n"
        "progs=json.load(open('/w/progs.json'))\n"
        "out={}\n"
        "for k,src in progs.items():\n"
        "    p='/tmp/p.py'; open(p,'w').write(src)\n"
        "    try:\n"
        f"        r=subprocess.run([sys.executable,p],capture_output=True,text=True,timeout={timeout})\n"
        "        out[k]=[r.returncode==0,r.stdout[-300:],r.stderr[-600:]]\n"
        "    except subprocess.TimeoutExpired:\n"
        "        out[k]=[False,'','timeout']\n"
        "print(json.dumps(out))\n")
    with tempfile.TemporaryDirectory() as d:
        Path(d, "progs.json").write_text(json.dumps(programs), encoding="utf-8")
        Path(d, "run.py").write_text(runner, encoding="utf-8")
        r = subprocess.run(["podman", "run", "--rm", "--network", "none", "--memory", "2g",
                            "--pids-limit", "256", "-v", f"{d}:/w:ro,Z", SANDBOX_IMAGE,
                            "python", "/w/run.py"], capture_output=True, text=True,
                           timeout=timeout * len(programs) + 120)
        if r.returncode:
            raise SystemExit(f"sandbox failed: {r.stderr[-500:]}")
        out = json.loads(r.stdout.strip().splitlines()[-1])
        return out if detail else {k: v[0] for k, v in out.items()}


#: --variant NAME prepends prompts/NAME.txt as a system message (e.g. caveman: terse-style rules)
PROMPTS = BENCH / "prompts"


def run(suite: str, spec: str = "none", n: int | None = None, think: bool = False,
        variant: str | None = None) -> dict:
    rows = tasks(suite, n)
    lora = _lora(spec)
    make = {"code": code_messages, "math": math_messages, "summ": summ_messages}[suite]
    if variant:
        system = (PROMPTS / f"{variant}.txt").read_text(encoding="utf-8")
        make = (lambda f: lambda t: [{"role": "system", "content": system}] + f(t))(make)
    max_tokens = (6144 if think else {"code": 1024, "math": 1024, "summ": 300}[suite])
    workers = experts.load_registry().get("server", {}).get("parallel", 2)
    t0 = time.monotonic()
    with ThreadPoolExecutor(workers) as ex:
        replies = list(ex.map(lambda t: _chat(make(t), lora, think, max_tokens), rows))
    res = []
    if suite == "code":
        verdict = sandbox_run({t["id"]: code_program(t, r["text"]) for t, r in zip(rows, replies)})
        for t, r in zip(rows, replies):
            res.append({"id": t["id"], "ok": verdict[t["id"]], **r})
        score = 100 * sum(x["ok"] for x in res) / len(res)
        extra = {}
    elif suite == "math":
        for t, r in zip(rows, replies):
            p = math_pred(r["text"])
            res.append({"id": t["id"], "ok": math_ok(p, t["gold"]), "pred": p, "gold": t["gold"], **r})
        score = 100 * sum(x["ok"] for x in res) / len(res)
        extra = {}
    else:
        for t, r in zip(rows, replies):
            res.append({"id": t["id"], **rouge(r["text"], t["highlights"]), **r})
        m = {k: 100 * sum(x[k] for x in res) / len(res) for k in ("r1", "r2", "rL")}
        score = m["rL"]
        extra = {k: round(v, 2) for k, v in m.items()}
    doc = {"suite": suite, "adapter": spec, "think": think, **({"variant": variant} if variant else {}),
           "n": len(res), "score": round(score, 1),
           **extra, "errors": sum(bool(x["error"]) for x in res),
           "mean_tokens": round(sum(x["tokens"] for x in res) / len(res)),
           "seconds": round(time.monotonic() - t0, 1)}
    return _finish(doc, res)


# --------------------------------------------------------------------------
# --verify: test-time verification (no hidden tests, no gold answers)
# --------------------------------------------------------------------------


def doctest_examples(prompt: str) -> list[tuple[str, str]]:
    """The >>> examples the HumanEval prompt shows the model (part of the input, not the tests)."""
    try:
        exs = doctest.DocTestParser().get_examples(prompt)
    except ValueError:
        return []
    out = []
    for e in exs:  # a docstring's closing quotes on the next line read as part of the output
        want = "\n".join(w for w in e.want.splitlines() if w.strip() not in ('"""', "'''")).strip()
        if want:
            out.append((e.source.strip(), want))
    return out


def example_program(t: dict, reply: str, examples: list[tuple[str, str]]) -> str:
    """The candidate function, then the prompt's own examples as checks (code_program minus the tests)."""
    prog = code_program(t, reply)
    prog = prog[:prog.rindex(t["test"])]
    return prog + "\n_EX = " + repr(examples) + """
for _s, _w in _EX:
    _got = eval(_s)
    try:
        _ok = _got == eval(_w)
    except Exception:
        _ok = repr(_got).strip() == _w.strip() or str(_got).strip() == _w.strip()
    assert _ok, f"{_s} returned {_got!r}, expected {_w}"
"""


def test_messages(t: dict) -> list[dict]:
    return [{"role": "user", "content": "Write 6 Python assert statements that test this function, based only "
             f"on what its docstring says. Reply with only the assert lines, calling {t['entry']}(...).\n\n"
             "```python\n" + t["prompt"] + "```"}]


def gen_asserts(text: str) -> list[str]:
    return [ln.strip() for ln in extract_code(text).splitlines() if ln.strip().startswith("assert ")][:8]


def assert_program(t: dict, reply: str, asserts: list[str]) -> str:
    """The candidate, then generated asserts; prints how many pass (CodeT-style ranking)."""
    prog = code_program(t, reply)
    prog = prog[:prog.rindex(t["test"])]
    return prog + "\n_A = " + repr(asserts) + """
_n = 0
for _a in _A:
    try:
        exec(_a)
        _n += 1
    except BaseException:
        pass
print(_n)
"""


def vote(preds: list[str | None]) -> str | None:
    """Majority over parsed numbers (as floats); ties keep the earliest answer (greedy first)."""
    keys = []
    for p in preds:
        try:
            keys.append(None if p is None else float(p))
        except ValueError:
            keys.append(None)
    c = collections.Counter(k for k in keys if k is not None)
    if not c:
        return None
    top = max(c.values())
    return next(p for p, k in zip(preds, keys) if k is not None and c[k] == top)


def pot_messages(t: dict) -> list[dict]:
    return [{"role": "user", "content": t["question"] + "\n\nWrite a short Python program that computes the "
             "answer and prints only the final number. Reply with one ```python code block."}]


def run_verify(suite: str, spec: str, k: int, think: bool = False, n: int | None = None) -> dict:
    rows = tasks(suite, n)
    lora = _lora(spec)
    workers = experts.load_registry().get("server", {}).get("parallel", 2)
    mt = 6144 if think else 1024
    t0 = time.monotonic()

    def gen(make, t, i):
        return _chat(make(t), lora, think, mt, temperature=0 if i == 0 else (0.8 if suite == "code" else 0.7),
                     seed=None if i == 0 else 1000 + i)

    jobs = [(t, i) for t in rows for i in range(k)]
    make = code_messages if suite == "code" else math_messages
    with ThreadPoolExecutor(workers) as ex:
        outs = list(ex.map(lambda ti: gen(make, *ti), jobs))
    cand = {t["id"]: [] for t in rows}
    for (t, i), o in zip(jobs, outs):
        cand[t["id"]].append(o)
    res = []
    if suite == "code":
        exs = {t["id"]: doctest_examples(t["prompt"]) for t in rows}
        checks = sandbox_run({f"{t['id']}#{i}": example_program(t, c["text"], exs[t["id"]])
                              for t in rows if exs[t["id"]] for i, c in enumerate(cand[t["id"]])}, detail=True)
        no_ex = [t for t in rows if not exs[t["id"]]]
        with ThreadPoolExecutor(workers) as ex:
            asserts = dict(zip([t["id"] for t in no_ex],
                               ex.map(lambda t: gen_asserts(_chat(test_messages(t), lora, False, 600)["text"]), no_ex)))
        counts = sandbox_run({f"{t['id']}#{i}": assert_program(t, c["text"], asserts[t["id"]])
                              for t in no_ex if asserts[t["id"]] for i, c in enumerate(cand[t["id"]])}, detail=True)
        chosen, repair = {}, {}
        for t in rows:
            if not exs[t["id"]]:
                score = [int((counts.get(f"{t['id']}#{i}", [0, "0"])[1].strip() or "0").splitlines()[-1])
                         for i in range(k)]
                best = max(range(k), key=lambda i: (score[i], -i))
                chosen[t["id"]] = (cand[t["id"]][best]["text"], f"gentests-sample{best}" if best else "gentests-greedy")
                continue
            ok = [i for i in range(k) if checks.get(f"{t['id']}#{i}", [False])[0]]
            if ok:
                chosen[t["id"]] = (cand[t["id"]][ok[0]]["text"], f"sample{ok[0]}")
            else:
                repair[t["id"]] = checks.get(f"{t['id']}#0", [False, "", ""])[2]
        by_id = {t["id"]: t for t in rows}

        def fix(tid):
            t, first = by_id[tid], cand[tid][0]["text"]
            msgs = code_messages(t) + [{"role": "assistant", "content": first},
                                       {"role": "user", "content": "Running the examples from the docstring "
                                        "failed:\n" + repair[tid] + "\nFix the function. Reply with the "
                                        "complete function in one ```python code block."}]
            return tid, _chat(msgs, lora, think, mt)["text"]

        with ThreadPoolExecutor(workers) as ex:
            fixed = dict(ex.map(fix, list(repair)))
        rechk = sandbox_run({tid: example_program(by_id[tid], txt, exs[tid]) for tid, txt in fixed.items()}) if fixed else {}
        for tid, txt in fixed.items():
            chosen[tid] = (txt, "repair") if rechk.get(tid) else (cand[tid][0]["text"], "greedy-fallback")
        final = sandbox_run({t["id"]: code_program(t, chosen[t["id"]][0]) for t in rows})
        greedy = sandbox_run({t["id"]: code_program(t, cand[t["id"]][0]["text"]) for t in rows})
        for t in rows:
            res.append({"id": t["id"], "ok": final[t["id"]], "greedy_ok": greedy[t["id"]], "via": chosen[t["id"]][1],
                        "n_examples": len(exs[t["id"]]), "n_gen_asserts": len(asserts.get(t["id"], []))})
    else:
        with ThreadPoolExecutor(workers) as ex:
            pots = list(ex.map(lambda t: _chat(pot_messages(t), lora, False, 1024), rows))
        ran = sandbox_run({t["id"]: extract_code(p["text"]) for t, p in zip(rows, pots)}, detail=True)
        for t, p in zip(rows, pots):
            preds = [math_pred(c["text"]) for c in cand[t["id"]]]
            pot = math_pred("Answer: " + ran[t["id"]][1].strip().splitlines()[-1]) if ran[t["id"]][0] and ran[t["id"]][1].strip() else None
            v = vote(preds + [pot])
            res.append({"id": t["id"], "ok": math_ok(v, t["gold"]), "greedy_ok": math_ok(preds[0], t["gold"]),
                        "pred": v, "gold": t["gold"], "preds": preds, "pot": pot})
    score = 100 * sum(x["ok"] for x in res) / len(res)
    doc = {"suite": suite, "adapter": spec, "think": think, "variant": f"verify{k}", "n": len(res),
           "score": round(score, 1), "greedy_score": round(100 * sum(x["greedy_ok"] for x in res) / len(res), 1),
           "gen_tokens": sum(o["tokens"] for o in outs), "errors": sum(bool(o["error"]) for o in outs),
           "seconds": round(time.monotonic() - t0, 1)}
    if suite == "code":
        doc["via"] = dict(collections.Counter(x["via"] for x in res))
    return _finish(doc, res)


def _finish(doc: dict, res: list[dict]) -> dict:
    doc["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"{doc['suite']}{'-think' if doc['think'] else ''}{'-' + doc['variant'] if doc.get('variant') else ''}-{doc['adapter'].replace(':', '_').replace(',', '+')}"
    (RESULTS / f"{tag}.json").write_text(json.dumps({**doc, "results": res}, indent=1) + "\n", encoding="utf-8")
    with RESULTS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(doc) + "\n")
    print(" ".join(f"{k}={v}" for k, v in doc.items() if k not in ("by_cat", "ts")), flush=True)
    return doc




def selftest() -> int:
    t = {"id": "x", "prompt": "def add(a, b):\n    \"\"\"sum\"\"\"\n", "test": "def check(f):\n    assert f(1, 2) == 3\n", "entry": "add"}
    full = code_program(t, "Here:\n```python\ndef add(a, b):\n    return a + b\n```")
    body = code_program(t, "```python\n    return a + b\n```")
    ns: dict = {}
    exec(full, ns)
    exec(body, {})
    assert math_pred("so 3 + 4 = 7\nAnswer: $1,250.") == "1250"
    assert math_pred("the total is 18") == "18"
    assert math_ok("18.0", "18") and not math_ok(None, "1")
    r = rouge("the cat sat on the mat", "the cat sat on a mat")
    assert 0.6 < r["rL"] < 1 and r["r1"] > r["r2"]
    p = 'def inc(x):\n    """add one\n    >>> inc(1)\n    2\n    >>> inc(-1)\n    0\n    """\n'
    assert doctest_examples(p) == [("inc(1)", "2"), ("inc(-1)", "0")]
    ti = {"id": "i", "prompt": p, "test": "def check(f):\n    assert f(5) == 6\n", "entry": "inc"}
    exec(example_program(ti, "```python\ndef inc(x):\n    return x + 1\n```", doctest_examples(p)), {})
    try:
        exec(example_program(ti, "```python\ndef inc(x):\n    return x\n```", doctest_examples(p)), {})
        raise RuntimeError("wrong candidate passed its examples")
    except AssertionError:
        pass
    assert vote(["3", "4", "4.0", None]) == "4" and vote(["3", "4"]) == "3" and vote([None]) is None
    print("selftest ok")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("selftest")
    p = sub.add_parser("run")
    p.add_argument("suite", choices=("code", "math", "summ"))
    p.add_argument("--adapter", default="none")
    p.add_argument("--n", type=int)
    p.add_argument("--think", action="store_true")
    p.add_argument("--variant", help="system prompt from bench/prompts/NAME.txt")
    p.add_argument("--verify", type=int, metavar="K", help="test-time verification with K candidates (code, math)")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        print(prepare())
    elif a.cmd == "run" and a.verify:
        run_verify(a.suite, a.adapter, a.verify, a.think, a.n)
    elif a.cmd == "run":
        run(a.suite, a.adapter, a.n, a.think, a.variant)
    else:
        return selftest()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
