"""Build long contexts from inputs this shop already processes.

Sources (see ``eval/contexts.json``):
  calls  - held-out (dev) sales-brain call transcripts from the realspeech
           manifest, whole transcripts grouped by speaker;
  talks  - the 30 codacus-mine captions in ``data/codacus/``;
  repos  - real repo contexts named by open local-lane work items.

No raw text is committed: this module reads the sources at run time and the
job writes its result JSON under ``state/kvfrontier/`` (gitignored).
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval"


@dataclass(frozen=True)
class Doc:
    id: str
    text: str


def _expand(path: str | Path) -> Path:
    return Path(str(path)).expanduser()


def load_contexts_spec(path: Path | None = None) -> dict:
    return json.loads((path or EVAL / "contexts.json").read_text(encoding="utf-8"))


def load_needles_spec(path: Path | None = None) -> dict:
    return json.loads((path or EVAL / "needles.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- sources

def load_realspeech(path: str, split: str = "dev", min_chars: int = 2000) -> list[Doc]:
    """Whole held-out transcripts, one Doc per speaker, ordered longest first."""
    p = _expand(path)
    if not p.is_file():
        return []
    groups: dict[str, list[dict]] = {}
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("split") != split:
                continue
            speaker = row.get("speaker")
            if not speaker:
                continue
            groups.setdefault(speaker, []).append(row)

    docs: list[Doc] = []
    for speaker, rows in groups.items():
        rows.sort(key=lambda r: r.get("start", 0))
        text = "\n".join(r.get("text", "").strip() for r in rows if r.get("text"))
        if len(text) >= min_chars:
            digest = hashlib.sha1(speaker.encode("utf-8")).hexdigest()[:10]
            docs.append(Doc(f"rs-{digest}", text))
    docs.sort(key=lambda d: (-len(d.text), d.id))
    return docs


def load_text_glob(pattern: str) -> list[Doc]:
    paths = sorted(glob.glob(str(ROOT / pattern)))
    docs = []
    for path in paths:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) >= 200:
            docs.append(Doc(p.stem, text))
    return docs


_FILE_LINE = re.compile(r"^\s*-\s+(\S.*)$")


def _files_from_work_item(path: Path) -> list[str]:
    """Read the ``files:`` list from a local-lane work item (line-based, no yaml dep)."""
    files: list[str] = []
    in_files = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.match(r"^files:\s*$", line):
            in_files = True
            continue
        if in_files:
            m = _FILE_LINE.match(line)
            if m:
                files.append(m.group(1).strip().strip("'\""))
            elif line.strip() and not line.startswith((" ", "\t")):
                break
    return files


def load_work_items(work_dir: str, root: str, fallback_glob: str = "") -> list[Doc]:
    """One Doc per open local-lane work item, made of the files it names."""
    work = _expand(work_dir)
    repo = _expand(root)
    docs: list[Doc] = []
    if work.is_dir():
        for item in sorted(work.glob("*.yaml")):
            parts = []
            for rel in _files_from_work_item(item):
                f = repo / rel
                if f.is_file():
                    parts.append(f"// file: {rel}\n" + f.read_text(encoding="utf-8", errors="replace"))
            if parts:
                docs.append(Doc(f"work-{item.stem}", "\n\n".join(parts)))
    if not docs and fallback_glob:
        docs = load_text_glob(fallback_glob)
    docs.sort(key=lambda d: (-len(d.text), d.id))
    return docs


def load_source(spec: dict) -> list[Doc]:
    kind = spec.get("kind")
    if kind == "realspeech":
        return load_realspeech(spec["path"], spec.get("split", "dev"))
    if kind == "text_glob":
        return load_text_glob(spec["glob"])
    if kind == "work_items":
        return load_work_items(spec["work_dir"], spec["root"], spec.get("fallback_glob", ""))
    raise ValueError(f"unknown source kind: {kind}")


# ---------------------------------------------------------------- assembly

def concat_within(docs: list[Doc], char_budget: int) -> tuple[str, list[str]]:
    """Concatenate whole docs until the budget, preferring docs that fit whole.

    Only when no doc fits at all is the first one cut.
    """
    out: list[str] = []
    ids: list[str] = []
    used = 0
    for doc in docs:
        if used >= char_budget:
            break
        room = char_budget - used
        if len(doc.text) <= room:
            out.append(doc.text)
            ids.append(doc.id)
            used += len(doc.text)
    if not out and docs:
        out.append(docs[0].text[:char_budget])
        ids.append(docs[0].id)
    return "\n\n".join(out), ids


def build_mixed(docs_by_source: dict[str, list[Doc]], mix: dict[str, float],
                token_target: int, chars_per_token: float) -> tuple[str, dict]:
    """Mix whole source docs into one context of roughly ``token_target`` tokens.

    Each source gets its fraction of the character budget; if a source runs out
    of docs, the shortfall is topped up from the other sources so the context
    actually reaches the target length.
    """
    char_budget = int(token_target * chars_per_token)
    pieces: list[str] = []
    used: dict[str, list[str]] = {}
    consumed: dict[str, set[str]] = {}
    for source, fraction in mix.items():
        docs = docs_by_source.get(source) or []
        text, ids = concat_within(docs, int(char_budget * fraction))
        if text:
            pieces.append(text)
        used[source] = ids
        consumed[source] = set(ids)

    total = sum(len(p) for p in pieces) + 2 * len(pieces)
    for source in mix:
        if total >= char_budget:
            break
        remaining = [d for d in (docs_by_source.get(source) or []) if d.id not in consumed[source]]
        more, ids = concat_within(remaining, char_budget - total)
        if more:
            pieces.append(more)
            used[source].extend(ids)
            consumed[source].update(ids)
            total += len(more) + 2

    text = "\n\n".join(pieces)
    return text, {"chars": len(text), "docs": used,
                  "estimated_tokens": int(len(text) / chars_per_token)}


def insert_facts(text: str, facts: list[dict]) -> str:
    """Insert ``{"text": ..., "at": fraction}`` blocks, later positions first."""
    for fact in sorted(facts, key=lambda f: f["at"], reverse=True):
        pos = max(0, min(len(text) - 1, int(len(text) * float(fact["at"]))))
        text = text[:pos] + "\n\n" + fact["text"] + "\n\n" + text[pos:]
    return text


def build_prompt(document: str, questions: list[dict]) -> str:
    numbered = "\n".join(f"{i}. {q['q']}" for i, q in enumerate(questions, 1))
    return (
        "The document below is a long internal archive. Read it, then answer the "
        "questions using only facts stated in it.\n\n"
        f"<document>\n{document}\n</document>\n\n"
        f"Questions:\n{numbered}\n\n"
        "Answer each question on its own line as \"1: <answer>\" and so on. "
        "Be exact and brief."
    )


def build_all(spec: dict | None = None) -> dict[str, dict]:
    """Build every context in ``eval/contexts.json`` in memory."""
    spec = spec or load_contexts_spec()
    cpt = float(spec.get("chars_per_token", 3.0))
    docs_by_source = {name: load_source(src) for name, src in spec["sources"].items()}
    out: dict[str, dict] = {}
    for ctx in spec["contexts"]:
        text, meta = build_mixed(docs_by_source, ctx["mix"], int(ctx["tokens"]), cpt)
        out[ctx["id"]] = {"text": text, "tokens": int(ctx["tokens"]), **meta}
    return out
