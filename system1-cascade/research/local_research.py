"""Local JEV/Laya research: plain fetch -> Saluki-27B (:8131) extraction -> quote-verified catalog.

Stages: fetch (stdlib HTTP, cached, 1 req/s per host) | extract (local model, 2 parallel) | report.
Downloaded text lives in ./raw (untrusted data, never executed). No cloud models.
"""
import concurrent.futures as cf, hashlib, json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW, EXT = ROOT / "raw", ROOT / "extract"
LLM = "http://localhost:8131/v1/chat/completions"
UA = {"User-Agent": "jev-research/0.1 (local)", "Accept": "application/json, text/plain, */*"}
CHUNK = 12000
MAX_DOCS = 400
_last = {}


def get(url, raw=False):
    host = urllib.parse.urlparse(url).netloc
    wait = 1.2 - (time.time() - _last.get(host, 0))
    if wait > 0:
        time.sleep(wait)
    _last[host] = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
            body = r.read(2_000_000).decode("utf-8", "replace")
    except Exception as e:
        print("  fail", url, type(e).__name__, file=sys.stderr)
        return None
    return body if raw else json.loads(body)


def save(url, text, kind):
    if not text or len(text.strip()) < 80:
        return
    name = hashlib.sha256(url.encode()).hexdigest()[:16]
    (RAW / f"{name}.json").write_text(json.dumps({"url": url, "kind": kind, "text": text[:60000]}))


def gh_readme(full):
    for br in ("main", "master"):
        t = get(f"https://raw.githubusercontent.com/{full}/{br}/README.md", raw=True)
        if t:
            return f"https://github.com/{full}", t
    return None, None


def fetch():
    RAW.mkdir(exist_ok=True)
    seen_gh, seen_arx = set(), set()
    # 1. awesome list + survey, then one hop over every GitHub/arXiv link they contain
    lists = []
    for path in ("README.md", "paper/survey.md"):
        for br in ("main", "master"):
            u = f"https://raw.githubusercontent.com/Eurekaleo/awesome-jev-survey/{br}/{path}"
            t = get(u, raw=True)
            if t:
                save(u, t, "awesome")
                lists.append(t)
                break
    for q in ("awesome-jev", "awesome laya model", "jev model", "laya model", "jev system-1"):
        d = get("https://api.github.com/search/repositories?per_page=30&q=" + urllib.parse.quote(q)) or {}
        for it in d.get("items", []):
            seen_gh.add(it["full_name"])
    d = get("https://api.github.com/users/he-jev/repos?per_page=100") or []
    for it in d if isinstance(d, list) else []:
        seen_gh.add(it["full_name"])
    blob = "\n".join(lists)
    for m in re.findall(r"github\.com/([\w.-]+/[\w.-]+)", blob):
        seen_gh.add(m.removesuffix(".git").rstrip("."))
    seen_arx |= set(re.findall(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", blob))
    hf_ids = set(re.findall(r"huggingface\.co/([\w.-]+/[\w.-]+)", blob))
    # 2. Hugging Face: search + known authors
    for q in ("search=jev", "search=laya", "author=he-jev", "author=hotdogs"):
        for it in get(f"https://huggingface.co/api/models?limit=100&{q}") or []:
            hf_ids.add(it["id"])
    for q in ("search=jev", "search=laya"):
        for it in get(f"https://huggingface.co/api/datasets?limit=50&{q}") or []:
            save(f"https://huggingface.co/datasets/{it['id']}",
                 get(f"https://huggingface.co/datasets/{it['id']}/raw/main/README.md", raw=True), "hf-dataset")
    # derivatives (finetunes/adapters/quants/merges) of every base found so far
    for base in list(hf_ids)[:40]:
        for it in get("https://huggingface.co/api/models?limit=50&filter=base_model:" + urllib.parse.quote(base)) or []:
            hf_ids.add(it["id"])
    # 3. arXiv + HN mentions
    for q in ('all:"JEV" AND all:model', 'all:"Laya" AND all:model'):
        x = get("http://export.arxiv.org/api/query?max_results=40&search_query=" + urllib.parse.quote(q), raw=True) or ""
        seen_arx |= set(re.findall(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", x))
    for q in ("jev model", "laya model", "awesome-jev"):
        for h in (get("https://hn.algolia.com/api/v1/search?hitsPerPage=30&query=" + urllib.parse.quote(q)) or {}).get("hits", []):
            txt = "\n".join(filter(None, [h.get("title"), h.get("url"), h.get("story_text"), h.get("comment_text")]))
            save(f"https://news.ycombinator.com/item?id={h['objectID']}", txt, "hn")
    print(f"targets: github {len(seen_gh)} hf {len(hf_ids)} arxiv {len(seen_arx)}")
    n = 0
    for full in sorted(seen_gh):
        if n >= MAX_DOCS: break
        u, t = gh_readme(full); save(u, t, "github"); n += 1
    for mid in sorted(hf_ids):
        if n >= MAX_DOCS: break
        save(f"https://huggingface.co/{mid}", get(f"https://huggingface.co/{mid}/raw/main/README.md", raw=True), "hf-model"); n += 1
    for aid in sorted(seen_arx):
        if n >= MAX_DOCS: break
        x = get(f"http://export.arxiv.org/api/query?id_list={aid}", raw=True) or ""
        m = re.search(r"<entry>.*?<title>(.*?)</title>.*?<summary>(.*?)</summary>", x, re.S)
        if m: save(f"https://arxiv.org/abs/{aid}", m.group(1) + "\n\n" + m.group(2), "arxiv")
        n += 1
    print("saved docs:", len(list(RAW.glob("*.json"))))


PROMPT = """You extract facts about fast "System-1" decision models called JEV and Laya (and models of the same type: small fast typed-decision / routing / classification models, their adapters, fine-tunes, quants, merges) from ONE document.
Return ONLY a JSON object:
{"relevant": true|false,   // false if the doc is not about these models or this model type (e.g. a person named Jev/Laya)
 "models": [{"name":"", "base":"", "size":"", "kind":"base|adapter|finetune|quant|merge|router|other", "purpose":"", "hf_or_repo":""}],
 "use_cases": [{"use":"", "setup":"router|adapter|verifier|classifier|ensemble|other", "quote":""}],
 "results": [{"what":"", "benchmark":"", "number":"", "baseline":"", "quote":""}],
 "techniques": [{"name":"", "claimed_gain":"", "quote":""}],
 "contamination_risk": ""}
Every "quote" MUST be copied character-for-character from the document (max 200 chars) and must contain the number it supports. Never invent numbers. Empty lists are fine.

URL: {url}
DOCUMENT:
{text}"""


def llm(prompt):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 1500,
            "response_format": {"type": "json_object"}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(LLM, json.dumps(body).encode(), {"Content-Type": "application/json"})
    # the experts server restarts under other sessions; wait it out instead of recording an error
    for attempt in range(40):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code not in (502, 503) or attempt == 39:
                raise
        except (urllib.error.URLError, ConnectionError):
            if attempt == 39:
                raise
        time.sleep(15)


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def extract_one(p):
    out = EXT / p.name
    if out.exists():
        return
    doc = json.loads(p.read_text())
    text = doc["text"]
    # cheap prefilter: HF/GitHub search returns many "jev*"/"laya*" name collisions
    if doc["kind"] != "awesome" and not re.search(r"\b(jev|laya)\b", text, re.I):
        out.write_text(json.dumps({"url": doc["url"], "kind": doc["kind"], "parts": [], "dropped_unverified": 0}))
        return
    parts = []
    for i in range(0, min(len(text), CHUNK * 3), CHUNK):
        try:
            j = json.loads(llm(PROMPT.replace("{url}", doc["url"]).replace("{text}", text[i:i + CHUNK])))
        except Exception as e:
            j = {"error": type(e).__name__}
        parts.append(j)
    # quote verification: drop any claim whose quote is not verbatim in the source
    hay, dropped = norm(text), 0
    for j in parts:
        for key in ("use_cases", "results", "techniques"):
            keep = []
            for it in j.get(key) or []:
                q = norm(str(it.get("quote", "")))
                if q and q in hay:
                    keep.append(it)
                else:
                    dropped += 1
            j[key] = keep
    out.write_text(json.dumps({"url": doc["url"], "kind": doc["kind"], "parts": parts, "dropped_unverified": dropped}))
    print("done", doc["url"], "dropped", dropped, flush=True)


def extract():
    EXT.mkdir(exist_ok=True)
    with cf.ThreadPoolExecutor(2) as ex:
        list(ex.map(extract_one, sorted(RAW.glob("*.json"))))


def report():
    rows = [json.loads(p.read_text()) for p in sorted(EXT.glob("*.json"))]
    rel = [r for r in rows if any(pt.get("relevant") for pt in r["parts"])]
    L = [f"# JEV/Laya local research catalog\n\ndocs read: {len(rows)} · relevant: {len(rel)} · "
         f"claims dropped (quote not verbatim): {sum(r['dropped_unverified'] for r in rows)}\n"]
    for sec, key in (("Models", "models"), ("Use cases", "use_cases"), ("Results", "results"), ("Techniques", "techniques")):
        L.append(f"\n## {sec}\n")
        for r in rel:
            for pt in r["parts"]:
                for it in pt.get(key) or []:
                    L.append(f"- {json.dumps(it, ensure_ascii=False)} — {r['url']}")
    L.append("\n## Contamination flags\n")
    for r in rel:
        for pt in r["parts"]:
            if (pt.get("contamination_risk") or "").strip():
                L.append(f"- {pt['contamination_risk']} — {r['url']}")
    (ROOT / "CATALOG.md").write_text("\n".join(L))
    print("wrote", ROOT / "CATALOG.md", "relevant", len(rel), "of", len(rows))


if __name__ == "__main__":
    for stage in sys.argv[1:] or ["fetch", "extract", "report"]:
        globals()[stage]()
