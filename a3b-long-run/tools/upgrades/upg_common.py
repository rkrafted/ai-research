#!/usr/bin/env python3
"""Shared plumbing for the R5 upgrade trials.

Runs inside the runner's GPU sandbox (repo + HF cache writable, no network,
`ollama: true` bridges only 127.0.0.1:11434). Every trial writes one
`state/upgrades/<name>.json`:

    {"name","area","workload","before":{..},"after":{..},
     "change_pct":..,"adopt":true|false,"how":"exact flags/config"}

Only a runner GPU job may write those files; never a session.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixtures as F  # noqa: E402  (vendored R4 fixtures + scoring)

STATE_DIR = Path("state/upgrades")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# server / model locations -- built from Path.home(), never a literal /home path
OLLAMA_MODELS = Path.home() / ".ollama" / "models"
BLOBS = OLLAMA_MODELS / "blobs"
OLLAMA_LIB = Path.home() / "ollama" / "lib" / "ollama"
LLAMA_SERVER = OLLAMA_LIB / "llama-server"
TARGET_27B = ("hf.co", "JonathanColetti", "Qwen3.8-27B-Uncensored-GGUF", "Q4_K_M")
TARGET_9B_OLLAMA = "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"
BASE_9B_HF = "empero-ai/Qwen3.8-9B-Distill"

# The R2 prompt file is gitignored (raw caption text); a GPU job may read the
# sibling worktree read-only. Relative to this repo, so no absolute path.
SIBLING_PROMPTS = (Path(__file__).resolve().parents[2].parent /
                   "research-ingest" / "local_batch" / "ingest-prompts.jsonl")
SALES_BRAIN_DIR = Path(os.environ.get(
    "SALES_BRAIN_DIR",
    str(Path.home() / "theprojects" / "queue" / "sales-brain" / "data" / "synthetic")))


# ---------------------------------------------------------------------------
# result plumbing (the R5 schema)
# ---------------------------------------------------------------------------

def change_pct(before: dict, after: dict, key: str, higher_is_better: bool = True):
    """Percent change of `key` after vs before, sign-corrected.

    Positive means the change is an improvement, whatever the metric direction.
    Returns None when the key is missing; 0.0 when before is 0.
    """
    if key not in before or key not in after:
        return None
    b, a = float(before[key]), float(after[key])
    if b == 0:
        return 0.0
    pct = (a - b) / abs(b) * 100.0
    return round(pct if higher_is_better else -pct, 2)


def write_upgrade(name: str, area: str, workload: str, before: dict, after: dict,
                  adopt: bool, how: str, *, change_key: str | None = None,
                  higher_is_better: bool = True, state_dir: Path | str | None = None,
                  **extra) -> Path:
    """Write one trial result. `change_pct` is the primary metric's signed gain."""
    if change_key is None:
        change_key = next(iter(before), None)
    pct = change_pct(before, after, change_key, higher_is_better) if change_key else None
    result = {
        "name": name,
        "area": area,
        "workload": workload,
        "before": before,
        "after": after,
        "change_pct": pct,
        "adopt": bool(adopt),
        "how": how,
    }
    result.update(extra)
    d = Path(state_dir) if state_dir else STATE_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(p)
    print(f"[upg] wrote {p}: adopt={adopt} change_pct={pct}", flush=True)
    return p


def run_trial(name: str, area: str, workload: str, how: str, fn, *,
              change_key: str | None = None, higher_is_better: bool = True,
              min_gain_pct: float = 5.0, extra_checks=None,
              state_dir: Path | str | None = None) -> dict:
    """Run fn() -> (before, after, how_effective|None); always write a result.

    A failed trial still writes before/after (the exception text is folded into
    `after` as `error`), so the file is honest rather than empty.
    """
    t0 = time.time()
    before: dict = {"status": "unmeasured"}
    after: dict = {"status": "not reached"}
    how_eff = how
    try:
        before, after, how_eff = fn()
    except Exception as exc:  # noqa: BLE001 - report, never die before writing
        import traceback
        traceback.print_exc()
        after = {**after, "error": f"{type(exc).__name__}: {exc}"}
    gain = change_pct(before, after, change_key or "", higher_is_better) if change_key else None
    adopt = bool(gain is not None and gain >= min_gain_pct)
    if extra_checks is not None:
        try:
            adopt = bool(adopt and extra_checks(before, after))
        except Exception:  # noqa: BLE001
            adopt = False
    write_upgrade(name, area, workload, before, after, adopt, how_eff or how,
                  change_key=change_key, higher_is_better=higher_is_better,
                  gpu_seconds=round(time.time() - t0, 1), state_dir=state_dir)
    return {"before": before, "after": after, "adopt": adopt, "change_pct": gain}


# ---------------------------------------------------------------------------
# workload 1: the local 27B bulk-text batch (ingest prompts)
# ---------------------------------------------------------------------------

def load_ingest_prompts(n: int = 50) -> tuple[list[str], str]:
    """50 real R2 ingest prompts when the sibling worktree has them, else a
    deterministic in-repo stand-in of the same shape (never raw captions)."""
    if SIBLING_PROMPTS.is_file():
        rows = []
        with SIBLING_PROMPTS.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line)["prompt"])
                except (ValueError, KeyError):
                    continue
                if len(rows) >= n:
                    break
        if rows:
            return rows, f"real R2 ingest prompts ({SIBLING_PROMPTS.name})"
    return _fallback_ingest_prompts(n), "authored ingest-shaped fallback (sibling prompts absent)"


def _fallback_ingest_prompts(n: int) -> list[str]:
    base = ("You extract structured research notes from ONE YouTube video by an AI/ML "
            "channel. Return ONLY one compact JSON object. Schema: {\"methods\": [str], "
            "\"claims\": [{\"claim\": str, \"numbers\": str, \"timestamp\": str|null}], "
            "\"papers\": [{\"arxiv_id\": str, \"title\": str}], \"repos\": [str], "
            "\"hardware\": [str], \"audience\": str}\n\nVideo: {title}\nTranscript:\n{body}\nJSON:")
    topics = ["quantization", "KV-cache compression", "speculative decoding", "MoE offload",
              "QLoRA fine-tuning", "flash attention", "packing", "gradient checkpointing"]
    out = []
    for i in range(n):
        t = topics[i % len(topics)]
        title = f"Practical {t} for a single 24 GB GPU (episode {i})"
        body = " ".join(
            f"[{j:02d}:{(j * 7) % 60:02d}] The speaker explains {t} and reports a measured "
            f"speedup of {round(1.0 + (j % 5) * 0.25, 2)}x against the baseline."
            for j in range(40))
        out.append(base.replace("{title}", title).replace("{body}", body))
    return out


# ---------------------------------------------------------------------------
# llama-server (llama.cpp) control -- exact flags, measured on the 27B GGUF
# ---------------------------------------------------------------------------

def locate_target(model: tuple[str, ...] = TARGET_27B) -> Path:
    manifest = OLLAMA_MODELS / "manifests" / Path(*model)
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        models = [l for l in data.get("layers", [])
                  if "image.model" in str(l.get("mediaType", ""))]
        if models:
            layer = max(models, key=lambda x: x.get("size", 0))
            p = BLOBS / str(layer["digest"]).replace(":", "-")
            if p.is_file():
                return p
    cands = sorted(BLOBS.glob("sha256-*"), key=lambda p: p.stat().st_size, reverse=True)
    if cands:
        return cands[0]
    raise FileNotFoundError("no Ollama model blob found")


def _lib_env(variant: Path) -> dict:
    env = os.environ.copy()
    parts = [str(variant), str(OLLAMA_LIB)]
    if env.get("LD_LIBRARY_PATH"):
        parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    env["GGML_BACKEND_PATH"] = str(variant / "libggml-cuda.so")
    return env


def pick_backend() -> tuple[Path | None, str]:
    """Find Ollama's CUDA llama.cpp backend; CPU-only must fail fast."""
    notes = []
    for d in ("cuda_v12", "cuda_v13"):
        variant = OLLAMA_LIB / d
        if not (variant / "libggml-cuda.so").is_file():
            continue
        try:
            r = subprocess.run([str(LLAMA_SERVER), "--list-devices"], capture_output=True,
                               text=True, timeout=120, env=_lib_env(variant))
            out = ((r.stdout or "") + (r.stderr or "")).strip()
            notes.append(f"{d}: {out.splitlines()[-1] if out else 'no output'}")
            if re.search(r"CUDA\d", out):
                return variant, "; ".join(notes)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{d}: {type(exc).__name__}: {exc}")
    return None, "; ".join(notes)


def llama_help() -> str:
    try:
        r = subprocess.run([str(LLAMA_SERVER), "--help"], capture_output=True, text=True,
                           timeout=60, env=_lib_env(_VARIANT or (OLLAMA_LIB / "cuda_v12")))
        return (r.stdout or "") + (r.stderr or "")
    except Exception:  # noqa: BLE001
        return ""


_VARIANT: Path | None = None
_HELP = ""


def flag_for(*names: str) -> str | None:
    for n in names:
        if re.search(rf"(?<![-\w]){re.escape(n)}(?![-\w])", _HELP):
            return n
    return None


def context_flags() -> list[str]:
    k = flag_for("--cache-type-k", "-ctk")
    v = flag_for("--cache-type-v", "-ctv")
    out = []
    if k:
        out += [k, "q8_0"]
    if v:
        out += [v, "q8_0"]
    return out


def flash_flags() -> list[str]:
    f = flag_for("--flash-attn", "-fa")
    if not f:
        return []
    return [f, "on"] if f == "--flash-attn" else [f]


def parallel_flags(n: int) -> list[str]:
    p = flag_for("--parallel", "-np")
    return [p, str(n)] if p else []


def start_server(model: Path, extra: list[str], log_path: Path, *, ctx: int = 8192):
    env = _lib_env(_VARIANT or (OLLAMA_LIB / "cuda_v12"))
    cmd = [str(LLAMA_SERVER), "-m", str(model), "--host", "127.0.0.1", "--port", "8477",
           "-ngl", "99", "-c", str(ctx), "-b", "2048", "-ub", "512"] + extra
    print(f"[upg] server: {' '.join(cmd)}", flush=True)
    log = open(log_path, "wb")  # noqa: SIM115 - closed by caller
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc, log


def wait_ready(proc, timeout: float = 900.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen("http://127.0.0.1:8477/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return False


def complete(prompt: str, n_predict: int = 128) -> dict:
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
                       "top_k": 1, "seed": 0, "cache_prompt": False, "stream": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:8477/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read().decode("utf-8"))
    tim = d.get("timings") or {}
    return {"content": d.get("content") or "",
            "n": int(d.get("tokens_predicted") or tim.get("predicted_n") or 0),
            "ms": float(tim.get("predicted_ms") or 0.0),
            "prompt_n": int(tim.get("prompt_n") or 0),
            "prompt_ms": float(tim.get("prompt_ms") or 0.0)}


def parse_vram_mb(log_path: Path) -> dict:
    """Sum KV + compute buffer sizes llama-server prints at startup (MiB).

    nvidia-smi inside the bubblewrap namespace reports ~0, so the server's own
    allocation log is the honest number for a llama.cpp arm.
    """
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return {}
    kv = [float(x) for x in re.findall(r"KV buffer size\s*=\s*([0-9.]+)\s*MiB", text)]
    comp = [float(x) for x in re.findall(r"compute buffer size\s*=\s*([0-9.]+)\s*MiB", text)]
    out = {}
    if kv:
        out["kv_cache_mb"] = round(sum(kv), 1)
    if comp:
        out["compute_buffer_mb"] = round(sum(comp), 1)
    return out


def run_llama_arm(model: Path, extra: list[str], prompts: list[str], label: str,
                  *, ctx: int = 8192, n_predict: int = 128) -> dict:
    """Start llama-server with `extra` flags, run every prompt, return metrics."""
    log_path = Path("/tmp") / f"upg-{label}.log"
    proc, log = start_server(model, extra, log_path, ctx=ctx)
    tok = {"tokens": 0, "ms": 0.0, "prompt_tokens": 0, "prompt_ms": 0.0}
    contents = []
    t0 = time.time()
    try:
        if not wait_ready(proc):
            tail = log_path.read_text(errors="replace")[-1200:] if log_path.exists() else ""
            raise RuntimeError(f"llama-server ({label}) not ready: ...{tail}")
        for p in prompts:
            o = complete(p, n_predict=n_predict)
            tok["tokens"] += o["n"]
            tok["ms"] += o["ms"]
            tok["prompt_tokens"] += o["prompt_n"]
            tok["prompt_ms"] += o["prompt_ms"]
            contents.append(o["content"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    wall = time.time() - t0
    metrics = {
        "tokens_per_s": round(tok["tokens"] / (tok["ms"] / 1000.0), 2) if tok["ms"] else 0.0,
        "prompt_tokens_per_s": (round(tok["prompt_tokens"] / (tok["prompt_ms"] / 1000.0), 2)
                                if tok["prompt_ms"] else 0.0),
        "wall_seconds": round(wall, 1),
        "n_prompts": len(prompts),
        "completion_tokens": tok["tokens"],
    }
    metrics.update(parse_vram_mb(log_path))
    return {"metrics": metrics, "contents": contents}


def identity(a: list[str], b: list[str]) -> float:
    """Mean character-level agreement between two arm outputs."""
    if not a:
        return 0.0
    scores = []
    for x, y in zip(a, b):
        n = min(len(x), len(y))
        scores.append(100.0 * sum(c == d for c, d in zip(x, y)) / n if n else 0.0)
    return round(sum(scores) / len(scores), 1)


def setup_backend() -> str:
    global _VARIANT, _HELP
    _VARIANT, note = pick_backend()
    if _VARIANT is None:
        raise RuntimeError(f"no CUDA backend for llama-server ({note})")
    _HELP = llama_help()
    print(f"[upg] CUDA backend: {note}", flush=True)
    return note


def gpu_used_gb() -> float:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return round(int(r.stdout.strip().splitlines()[0]) / 1024.0, 2)
    except Exception:  # noqa: BLE001
        return 0.0


# ---------------------------------------------------------------------------
# Ollama client (sales-brain workload; only reachable with `ollama: true`)
# ---------------------------------------------------------------------------

class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, host: str = OLLAMA_HOST, timeout: float = 1800.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _req(self, path: str, payload: dict | None = None, timeout: float | None = None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.host + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(f"{path}: {exc}") from exc

    def chat(self, model: str, messages: list[dict], *, num_predict: int = 256,
             temperature: float = 0.0, num_ctx: int = 8192, seed: int = 1,
             think: bool = False, fmt=None, keep_alive=None) -> dict:
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"num_predict": num_predict, "temperature": temperature,
                               "num_ctx": num_ctx, "seed": seed},
                   "think": think}
        if fmt is not None:
            payload["format"] = fmt
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        t0 = time.time()
        try:
            d = self._req("/api/chat", payload)
        except OllamaError:
            if "think" not in payload:
                raise
            payload.pop("think", None)
            d = self._req("/api/chat", payload)
        wall = time.time() - t0
        msg = d.get("message") or {}
        text = msg.get("content") or msg.get("thinking") or ""
        ec, ed = int(d.get("eval_count") or 0), int(d.get("eval_duration") or 0)
        return {"text": text, "tokens": ec, "seconds": wall,
                "eval_tps": ec / (ed / 1e9) if ed else 0.0,
                "prompt_tokens": int(d.get("prompt_eval_count") or 0)}

    def unload(self, model: str):
        try:
            self._req("/api/generate", {"model": model, "keep_alive": 0, "prompt": ""})
        except OllamaError:
            pass


# ---------------------------------------------------------------------------
# workload 2: sales-brain extraction from call transcripts
# ---------------------------------------------------------------------------

def sales_brain_rows(n: int = 30) -> tuple[list[dict], str]:
    rows, why = F.hard_extraction_fixture(n)
    return rows[:n], why


HARD_SCHEMA = {
    "type": "object",
    "properties": {f: {"type": "string"} for f in F.HARD_FIELDS},
}


# ---------------------------------------------------------------------------
# workload 3 helpers: QLoRA on the 9B base
# ---------------------------------------------------------------------------

def load_base_4bit():
    """Load the cached 9B base in NF4 (shared by the QLoRA trials)."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    return AutoModelForCausalLM.from_pretrained(BASE_9B_HF, quantization_config=bnb,
                                                device_map={"": 0})


def qlora_examples(tokenizer, n: int = 24, max_len: int = 512) -> list[dict]:
    """(input_ids, labels) from the sales-brain transcripts, offline."""
    rows, _ = sales_brain_rows(n)
    out = []
    for r in rows:
        user = ("Extract the deal facts as one JSON object with keys: "
                + ", ".join(F.HARD_FIELDS) + ".\n\nTranscript:\n" + r["text"] + "\nJSON:")
        msgs = [{"role": "user", "content": user}]
        enc = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False)
        prompt_ids = list(enc["input_ids"] if hasattr(enc, "keys") else enc)
        ans = json.dumps(r["gold"], ensure_ascii=False)
        aid = tokenizer(ans, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        ids = (prompt_ids + aid)[:max_len]
        labels = ([-100] * len(prompt_ids) + aid)[:max_len]
        out.append({"ids": ids, "labels": labels})
    return out
