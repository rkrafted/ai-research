"""Shared harness for the R4 experiment cards (docs/experiments/*.md).

Runs inside the runner's GPU sandbox: only the repo and the HF cache are
writable, there is no network except a bridged Ollama socket when the job
declares `ollama: true`. Everything here is stdlib + whatever the GPU venv has;
nothing is fetched at run time.

The five cards each write `state/exp/<card>.json` from inside the job, never
from a session. See `docs/experiments/` for the hypotheses and pass rules.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

STATE_DIR = Path("state/exp")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# ---------------------------------------------------------------------------
# fixture: fictional sales-brain notes (no real people, @example.com, 555-01xx)
# ---------------------------------------------------------------------------

FIELDS = ["contact_name", "company", "budget_usd", "timeline", "next_step"]

_FIRST = ["Dana", "Milo", "Rae", "Ivo", "Tess", "Otto", "Nina", "Pax", "Lena", "Cyrus",
          "Marta", "Ravi", "Yuki", "Sofia", "Noor", "Emil", "Freya", "Jonas"]
_LAST = ["Whitfield", "Marchetti", "Okonkwo", "Lindqvist", "Ferreira", "Nakamura", "Bauer",
         "Adeyemi", "Novak", "Petrova", "Haddad", "Kowalski", "Reyes", "Duarte", "Okafor",
         "Lindgren", "Baptiste", "Sorensen"]
_COMPANIES = ["Northwind Analytics", "Helix Foundry", "Paper Kite Labs", "Granite & Co",
              "Blue Harbor Systems", "Cedar Peak Robotics", "Lumen Field Ops", "Kestrel Data",
              "Mistral Bay Tools", "Ironwood Logistics", "Solstice Retail", "Harborlight Health",
              "Vector & Vine", "Quarry Nine", "Alderfin", "Brightloom", "Cobalt Meridian",
              "Fern & Flint"]
_PRODUCTS = ["ledger sync", "route planner", "inbox triage", "compliance digest",
             "field scheduler", "pricing engine", "contract reader", "inventory scout"]
_BUDGETS = ["8000", "12000", "15000", "24000", "36000", "48000", "60000"]
_TIMELINES = ["Q1", "Q2", "Q3", "Q4", "next quarter"]
_NEXT = ["send proposal", "book a demo", "schedule a call", "share pricing", "follow up",
         "send security questionnaire"]

_ALIASES = {
    "contact_name": ("contact_name", "contact", "name", "full_name", "contact person"),
    "company": ("company", "account", "org", "organization", "company_name"),
    "budget_usd": ("budget_usd", "budget", "budget_amount", "amount_usd", "amount"),
    "timeline": ("timeline", "timeframe", "deadline", "when"),
    "next_step": ("next_step", "nextstep", "next step", "action", "next_action"),
}


def _record(i: int) -> dict:
    rng = random.Random(10007 + i)
    first = _FIRST[i % len(_FIRST)]
    last = _LAST[(i * 5 + 3) % len(_LAST)]
    company = _COMPANIES[(i * 7 + 1) % len(_COMPANIES)]
    product = _PRODUCTS[(i * 3) % len(_PRODUCTS)]
    budget = _BUDGETS[(i * 5 + 2) % len(_BUDGETS)]
    timeline = _TIMELINES[(i * 3 + 1) % len(_TIMELINES)]
    step = _NEXT[(i * 7 + 4) % len(_NEXT)]
    rec = {
        "id": i,
        "contact_name": f"{first} {last}",
        "company": company,
        "product": product,
        "budget_usd": budget,
        "timeline": timeline,
        "next_step": step,
        "email": f"{first.lower()}.{last.lower()}@example.com",
        "phone": f"555-01{i % 100:02d}",
    }
    templates = [
        ("{contact} from {company} came in about {product}. Budget is about ${budget} "
         "and they want it by {timeline}. Next: {step}. Reach them at {email} or {phone}."),
        ("Call notes - {company}. The {product} pilot is owned by {contact}. They have "
         "roughly ${budget} set aside; target date {timeline}. Agreed to {step}. {email}."),
        ("{company} / {contact}: looking at {product}. {budget} USD is earmarked for "
         "{timeline}. Follow-up action: {step}. Phone {phone}."),
        ("Inbound from {email} ({contact}) at {company}. They need {product} before "
         "{timeline}; the internal budget is ${budget}. We should {step}."),
        ("{contact}, {company}. Wants {product}. Says money is not the issue - ${budget} "
         "for {timeline}. Let's {step}. {phone}."),
        ("{company} is evaluating {product} with {contact}. Approval window is {timeline}, "
         "budget ${budget}. Action item: {step}. {email}."),
    ]
    text = templates[i % len(templates)].format(contact=rec["contact_name"], company=company,
                                                product=product, budget=budget,
                                                timeline=timeline, step=step,
                                                email=rec["email"], phone=rec["phone"])
    if i % 3 == 2:
        text = text + " " + rng.choice(["Schedule review.", "Second call booked.",
                                        "Champion confirmed.", "Legal loop pending."])
    rec["text"] = text
    return rec


def extraction_fixture(n: int = 300, start: int = 0) -> list[dict]:
    """Deterministic labelled extraction rows: text -> the five FIELDS."""
    rows = []
    for i in range(start, start + n):
        rec = _record(i)
        rows.append({"id": i, "text": rec["text"],
                     "gold": {f: rec[f] for f in FIELDS}})
    return rows


# ---------------------------------------------------------------------------
# hard fixture: sales-brain call transcripts (Latvian) or an adversarial fallback
# ---------------------------------------------------------------------------

# The real task cards 1/5 are rerun on (the easy 5-field notes hit the ceiling).
HARD_FIELDS = ["amount", "term_months", "annual_revenue",
               "years_trading", "employees", "industry", "need"]
# Path is built from Path.home(), never a literal /home path (export rule).
SALES_BRAIN_DIR = Path(os.environ.get(
    "SALES_BRAIN_DIR",
    str(Path.home() / "theprojects" / "queue" / "sales-brain" / "data" / "synthetic")))

_HARD_ALIASES = {
    "amount": ("amount", "sum", "summa", "loan_amount", "deal_amount", "credit_amount"),
    "term_months": ("term_months", "term", "months", "term_month", "duration_months"),
    "annual_revenue": ("annual_revenue", "revenue", "turnover", "apgrozijums",
                       "annual_turnover", "yearly_revenue"),
    "years_trading": ("years_trading", "years", "experience", "years_in_business",
                      "years_in_market"),
    "employees": ("employees", "employee_count", "staff", "headcount", "team_size",
                  "number_of_employees"),
    "industry": ("industry", "sector", "nozare", "field"),
    "need": ("need", "purpose", "financing_need", "loan_purpose", "need_type"),
}
_NUMERIC_HARD = ("amount", "term_months", "annual_revenue", "years_trading", "employees")


def canonical_hard(d: dict) -> dict:
    """Fold model-chosen key spellings onto HARD_FIELDS."""
    if not isinstance(d, dict):
        return {}
    lower = {re.sub(r"[^a-z0-9]+", "_", str(k).lower()).strip("_"): v for k, v in d.items()}
    out = {}
    for field, aliases in _HARD_ALIASES.items():
        for a in aliases:
            key = re.sub(r"[^a-z0-9]+", "_", a.lower()).strip("_")
            if key in lower:
                out[field] = lower[key]
                break
    return out


def hard_norm(field: str, value) -> str:
    """Field-aware normalisation: numbers to bare digits, else the shared norm_value."""
    s = norm_value(value)
    if field in _NUMERIC_HARD:
        digits = re.sub(r"[^0-9]", "", s)
        return digits or s
    return s


def hard_fields(d: dict) -> dict:
    """canonical_hard + hard_norm, ready for micro_f1(..., HARD_FIELDS)."""
    return {f: hard_norm(f, v) for f, v in canonical_hard(d).items()}


def _num_words(n: int) -> str:
    small = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
             7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
             13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
             18: "eighteen", 19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty",
             50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}
    if n in small:
        return small[n]
    if n < 100:
        return small[n // 10 * 10] + "-" + small[n % 10]
    if n < 1000:
        tail = "" if n % 100 == 0 else " " + _num_words(n % 100)
        return small[n // 100] + " hundred" + tail
    if n < 1_000_000:
        tail = "" if n % 1000 == 0 else " " + _num_words(n % 1000)
        return _num_words(n // 1000) + " thousand" + tail
    return str(n)


_HARD_INDUSTRIES = ["transports", "retail", "manufacturing", "construction", "it",
                    "hospitality", "agriculture", "logistics"]
_HARD_NEEDS = ["working_capital", "equipment", "expansion", "refinancing", "inventory"]


def _hard_fallback_row(i: int) -> dict:
    """Authored adversarial note: spelled-out numbers, a mid-call correction, noise."""
    rng = random.Random(90011 + i)
    amount = rng.choice([30000, 50000, 75000, 120000, 200000])
    term = rng.choice([12, 18, 24, 36, 48])
    revenue = amount * rng.choice([2, 3, 4, 5])
    years = rng.choice([2, 3, 5, 8, 12])
    employees = rng.choice([4, 9, 17, 23, 41, 88])
    industry = rng.choice(_HARD_INDUSTRIES)
    need = rng.choice(_HARD_NEEDS)
    corrects = rng.random() < 0.5
    final_amount = amount + 5000 if corrects else amount
    lines = [
        "rep: So what do you need the financing for?",
        f"prospect: {need.replace('_', ' ')}. We are talking about {_num_words(amount)} euros.",
        "rep: And over how long do you want to repay it?",
        f"prospect: {_num_words(term)} months.",
        (f"prospect: Actually, sorry, make the amount {_num_words(final_amount)} euros."
         if corrects else "prospect: The amount stays as I first said."),
        "rep: Turnover last year?",
        f"prospect: Annual revenue was around {_num_words(revenue)} euros.",
        "rep: How long has the company been trading?",
        f"prospect: About {_num_words(years)} years now.",
        "rep: And the team size?",
        f"prospect: We employ {_num_words(employees)} people.",
        "rep: Which industry are you in?",
        f"prospect: {industry}.",
        "rep: Great, thank you. I will note that down.",
    ]
    return {
        "id": f"fallback_{i:04d}",
        "text": "\n".join(lines),
        "lang": "en",
        "gold": {"amount": hard_norm("amount", final_amount),
                 "term_months": hard_norm("term_months", term),
                 "annual_revenue": hard_norm("annual_revenue", revenue),
                 "years_trading": hard_norm("years_trading", years),
                 "employees": hard_norm("employees", employees),
                 "industry": industry, "need": need},
    }


def _sales_brain_rows() -> tuple[list[dict], str]:
    root = Path(SALES_BRAIN_DIR)
    if not root.is_dir():
        return [], f"missing dir {root}"
    rows = []
    for d in sorted(root.glob("call_*")):
        cj, gj = d / "call.json", d / "gold.json"
        if not (cj.is_file() and gj.is_file()):
            continue
        try:
            call = json.loads(cj.read_text(encoding="utf-8"))
            gold = json.loads(gj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        turns = call.get("turns") or []
        text = "\n".join(f"[{int(t.get('t', 0))}s] {t.get('speaker', '?')}: "
                         f"{str(t.get('text', '')).strip()}" for t in turns)
        deal = gold.get("deal") or {}
        rows.append({"id": str(call.get("id", d.name)), "text": text,
                     "lang": str(call.get("lang", "")),
                     "gold": {f: hard_norm(f, deal.get(f)) for f in HARD_FIELDS}})
    return rows, f"{len(rows)} sales-brain calls"


def hard_extraction_fixture(n: int) -> tuple[list[dict], str]:
    """(rows, source). Real sales-brain calls when present, else an authored fallback."""
    rows, why = _sales_brain_rows()
    if len(rows) >= n:
        return rows[:n], "sales-brain-synthetic"
    rows = [_hard_fallback_row(i) for i in range(n)]
    return rows, f"adversarial-fallback ({why})"


def qa_fixture(n: int = 500, seed: int = 7) -> dict:
    """Deterministic passage+question set for card 2.

    Returns {"passages": [...], "pairs": [...]}; each pair asks one attribute of
    one passage. `pairs` is shuffled once with `seed`.
    """
    passages, pairs = [], []
    q_templates = {
        "contact_name": "Who is the contact at {company}?",
        "budget_usd": "What budget did {company} set aside?",
        "timeline": "When does {company} need delivery?",
        "next_step": "What is the agreed next step for {company}?",
        "company": "Which account does {contact} work for?",
    }
    for i in range(n):
        rec = _record(i)
        passages.append({"id": i, "text": rec["text"],
                         "attrs": {k: rec[k] for k in
                                   ("contact_name", "company", "budget_usd", "timeline", "next_step")}})
        field = list(q_templates)[i % len(q_templates)]
        pairs.append({"id": i, "passage_id": i, "field": field,
                      "question": q_templates[field].format(contact=rec["contact_name"],
                                                            company=rec["company"]),
                      "answer": rec[field]})
    random.Random(seed).shuffle(pairs)
    return {"passages": passages, "pairs": pairs}


def edit_facts(fixture: dict, pair_ids: list[int]) -> dict:
    """Return a copy of `fixture` with the named facts changed in the passages.

    The question stays; the gold answer and the passage attribute become the new
    value. Used by card 2's held-out edit test.
    """
    changed = set(pair_ids)
    by_pair = {p["id"]: p for p in fixture["pairs"]}
    new_passages = [dict(p) for p in fixture["passages"]]
    new_pairs = []
    for pid in changed:
        p = by_pair[pid]
        rec = p["passage_id"]
        attrs = dict(new_passages[rec]["attrs"])
        field = p["field"]
        if field == "company":
            new = "Changed Account Ltd"
        elif field == "budget_usd":
            new = str(int(attrs["budget_usd"]) + 1000)
        elif field == "timeline":
            order = ["Q1", "Q2", "Q3", "Q4", "next quarter"]
            new = order[(order.index(attrs["timeline"]) + 1) % len(order)]
        elif field == "next_step":
            new = "escalate to legal"
        else:
            new = "Alex Newperson"
        old = str(p["answer"])
        attrs[field] = new
        text = new_passages[rec]["text"].replace(old, new, 1)
        new_passages[rec] = {**new_passages[rec], "attrs": attrs, "text": text}
    for p in fixture["pairs"]:
        if p["id"] in changed:
            old = str(p["answer"])
            new = new_passages[p["passage_id"]]["attrs"][p["field"]]
            new_pairs.append({**p, "answer": new})
        else:
            new_pairs.append(dict(p))
    return {"passages": new_passages, "pairs": new_pairs}


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def norm_value(v) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = re.sub(r"[$,]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .;:\"'")


def canonical_fields(d: dict) -> dict:
    """Fold model-chosen key spellings onto the five canonical FIELDS."""
    if not isinstance(d, dict):
        return {}
    lower = {re.sub(r"[^a-z0-9]+", "_", str(k).lower()).strip("_"): v for k, v in d.items()}
    out = {}
    for field, aliases in _ALIASES.items():
        for a in aliases:
            key = re.sub(r"[^a-z0-9]+", "_", a.lower()).strip("_")
            if key in lower:
                out[field] = lower[key]
                break
    return out


def micro_f1(preds: list[dict], golds: list[dict], fields=FIELDS) -> float:
    """Field micro-F1 in percent over the listed fields."""
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        for f in fields:
            pv, gv = norm_value(p.get(f)), norm_value(g.get(f))
            if pv == gv and gv != "":
                tp += 1
            else:
                if pv != "":
                    fp += 1
                if gv != "":
                    fn += 1
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return 200.0 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> bool:
    return norm_value(pred) == norm_value(gold) and norm_value(gold) != ""


def majority(values: list[str]) -> str:
    vals = [v for v in values if v]
    if not vals:
        return ""
    counts: dict[str, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: (counts[k], -vals.index(k)))


def extract_json(text: str):
    """Last balanced JSON object in `text`, or None. Survives code fences."""
    if not text:
        return None
    candidates, depth, start, instr, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
    for c in reversed(candidates):
        try:
            d = json.loads(c)
        except Exception:
            continue
        if isinstance(d, dict):
            return d
    return None


# ---------------------------------------------------------------------------
# Ollama (bridged unix socket; only reachable when the job sets ollama: true)
# ---------------------------------------------------------------------------

class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, host: str = OLLAMA_HOST, timeout: float = 900.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _req(self, path: str, payload: dict | None = None, timeout: float | None = None):
        url = self.host + path
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as exc:          # noqa: PERF203 - one flat wrapper
            raise OllamaError(f"{path}: {exc}") from exc

    def chat(self, model: str, messages: list[dict], *, num_predict: int = 128,
             temperature: float = 0.0, num_ctx: int = 2048, seed: int = 1,
             think: bool = False, fmt=None, keep_alive: str | int | None = None) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": temperature,
                        "num_ctx": num_ctx, "seed": seed},
            "think": think,
        }
        if fmt is not None:
            payload["format"] = fmt
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        t0 = time.time()
        try:
            d = self._req("/api/chat", payload)
        except OllamaError:
            if "think" not in payload:                     # older server: retry without it
                raise
            payload.pop("think", None)
            d = self._req("/api/chat", payload)
        wall = time.time() - t0
        msg = d.get("message") or {}
        text = msg.get("content") or ""
        if not text and msg.get("thinking"):
            text = msg["thinking"]
        ec, ed = int(d.get("eval_count") or 0), int(d.get("eval_duration") or 0)
        tps = ec / (ed / 1e9) if ed else 0.0
        return {"text": text, "tokens": ec, "seconds": wall, "eval_tps": tps,
                "prompt_tokens": int(d.get("prompt_eval_count") or 0),
                "total_duration": int(d.get("total_duration") or 0)}

    def unload(self, model: str):
        try:                                                  # empty generate unloads
            self._req("/api/generate", {"model": model, "keep_alive": 0, "prompt": ""})
        except OllamaError:
            pass

    def ps(self) -> list[dict]:
        try:
            return list(self._req("/api/ps").get("models") or [])
        except OllamaError:
            return []

    def vram_gb(self, model: str) -> float:
        for m in self.ps():
            name = m.get("name") or m.get("model") or ""
            if name == model or name.split(":")[0] == model.split(":")[0]:
                return round(int(m.get("size_vram") or 0) / 1e9, 2)
        return 0.0

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        d = self._req("/api/embed", {"model": model, "input": texts})
        return list(d.get("embeddings") or [])


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# result plumbing
# ---------------------------------------------------------------------------

def arm(name: str, score: float, tokens_per_s: float, vram_gb: float, **extra) -> dict:
    a = {"name": name, "score": round(float(score), 4),
         "tokens_per_s": round(float(tokens_per_s), 2), "vram_gb": round(float(vram_gb), 2)}
    a.update(extra)
    return a


def base_result(card: str, hypothesis: str, metric: str, n: int, arms: list[dict],
                passed: bool, **extra) -> dict:
    r = {"card": card, "hypothesis": hypothesis, "metric": metric, "n": int(n),
         "arms": arms, "pass": bool(passed), "gpu_minutes": 0.0}
    r.update(extra)
    return r


def write_result(rel_path: str, result: dict) -> Path:
    p = Path(rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(p)
    print(f"[exp] wrote {p}", flush=True)
    return p


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[k]


def guard(card: str, hypothesis: str, metric: str, out: str, fn):
    """Run fn() -> (n, arms, passed, extra); always write the result file."""
    t0 = time.time()
    n, arms, passed, extra = 0, [], False, {}
    try:
        n, arms, passed, extra = fn()
    except Exception as exc:                                   # noqa: BLE001 - report, don't die
        traceback.print_exc()
        extra = {"error": f"{type(exc).__name__}: {exc}"}
        passed = False
    result = base_result(card, hypothesis, metric, n, arms, passed, **extra)
    result["gpu_minutes"] = max(0.01, round((time.time() - t0) / 60.0, 3))
    write_result(out, result)
    print(f"[exp] {card}: n={n} arms={len(arms)} pass={passed} "
          f"gpu_minutes={result['gpu_minutes']}", flush=True)
    return result


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0
