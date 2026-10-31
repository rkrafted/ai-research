#!/usr/bin/env python3
"""Card 4 — speculative decoding on our real bottleneck (the local 27B batches).

Re-scoped 2026-10-03 (the owner): no Qwen2.5-7B download. Use the llama-server
bundled with Ollama (`--model-draft`) on the cached qwen3.8-27B Q4_K_M GGUF,
with a <=1 GB Qwen3 0.6B Q8_0 draft (fetch/qwen3-0.6b-gguf.yaml). If the draft
vocabulary does not match, record it and fall back to the 0 GB alternative,
ngram-simple speculative decoding (`--spec-type ngram-simple`), which needs no
draft model. No network, no Ollama socket: the job starts llama-server itself.

Metric: tokens/s and greedy-output identity vs no-draft on 50 extraction prompts.
Writes state/exp/04-speculative-decoding.json.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TARGET_MODEL = "hf.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF:Q4_K_M"
TARGET_MANIFEST = (Path.home() / ".ollama" / "models" / "manifests" / "hf.co" /
                   "JonathanColetti" / "Qwen3.8-27B-Uncensored-GGUF" / "Q4_K_M")
BLOBS = Path.home() / ".ollama" / "models" / "blobs"
OLLAMA_LIB = Path.home() / "ollama" / "lib" / "ollama"
LLAMA_SERVER = OLLAMA_LIB / "llama-server"
DRAFT_REPO = "Qwen/Qwen3-0.6B-GGUF"
DRAFT_FILE = "Qwen3-0.6B-Q8_0.gguf"
PORT = 8477
N = 50
MAX_NEW = 128
OUT = "state/exp/04-speculative-decoding.json"
HYPOTHESIS = ("a 0.6B Qwen3 draft on the cached 27B gives >=1.5x tokens/s on our ingest-sized "
              "prompts at greedy-identical output; if the vocabularies differ, the 0 GB "
              "ngram-simple alternative is measured instead")
METRIC = "tokens/s speedup vs no-draft, greedy output identity, peak VRAM (GB)"


def locate_target() -> Path:
    if TARGET_MANIFEST.is_file():
        manifest = json.loads(TARGET_MANIFEST.read_text(encoding="utf-8"))
        models = [l for l in manifest.get("layers", [])
                  if "image.model" in str(l.get("mediaType", ""))]
        if models:
            layer = max(models, key=lambda x: x.get("size", 0))
            p = BLOBS / str(layer["digest"]).replace(":", "-")
            if p.is_file():
                return p
    cands = sorted(BLOBS.glob("sha256-*"), key=lambda p: p.stat().st_size, reverse=True)
    if cands:
        return cands[0]
    raise FileNotFoundError("no Ollama model blob found for the 27B target")


def locate_draft() -> Path:
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(DRAFT_REPO, DRAFT_FILE, local_files_only=True))
    except Exception:                                        # noqa: BLE001
        base = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        hits = sorted((base / "hub" / f"models--{DRAFT_REPO.replace('/', '--')}" /
                       "snapshots").glob(f"*/{DRAFT_FILE}"))
        if hits:
            return hits[-1]
        raise FileNotFoundError(f"{DRAFT_REPO}/{DRAFT_FILE} not in the HF cache; fetch it first")


def prompts(n: int) -> tuple[list[str], str]:
    rows, source = C.hard_extraction_fixture(n)
    out = []
    for row in rows:
        out.append("You extract deal facts from a sales-call transcript.\n"
                   "Reply with one JSON object with keys: " + ", ".join(C.HARD_FIELDS) +
                   ".\n\nTranscript:\n" + row["text"] + "\nJSON:")
    return out, source


def gpu_used_gb() -> float:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return round(int(r.stdout.strip().splitlines()[0]) / 1024.0, 2)
    except Exception:                                        # noqa: BLE001
        return 0.0


_VARIANT: Path | None = None             # CUDA backend dir, set by main()


def _lib_env(variant: Path) -> dict:
    env = os.environ.copy()
    parts = [str(variant), str(OLLAMA_LIB)]
    if env.get("LD_LIBRARY_PATH"):
        parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    # ggml_backend_load_all_from_path wants the .so itself, not the directory.
    env["GGML_BACKEND_PATH"] = str(variant / "libggml-cuda.so")
    return env


def pick_backend() -> tuple[Path | None, str]:
    """Find the CUDA backend variant Ollama ships. Returns (variant dir, note).

    llama-server on CPU cannot finish 50 27B prompts inside the timeout, so a
    missing CUDA backend must fail fast instead of silently running on CPU.
    """
    notes = []
    for d in ("cuda_v12", "cuda_v13"):
        variant = OLLAMA_LIB / d
        if not (variant / "libggml-cuda.so").is_file():
            continue
        try:
            r = subprocess.run([str(LLAMA_SERVER), "--list-devices"], capture_output=True,
                               text=True, timeout=120, env=_lib_env(variant))
        except Exception as exc:                             # noqa: BLE001
            notes.append(f"{d}: {type(exc).__name__}: {exc}")
            continue
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        notes.append(f"{d}: {out.splitlines()[-1] if out else 'no output'}")
        if re.search(r"CUDA\d", out):
            return variant, "; ".join(notes)
    return None, "; ".join(notes)


def start_server(model: Path, extra: list[str], log_path: Path):
    env = _lib_env(_VARIANT or (OLLAMA_LIB / "cuda_v12"))
    cmd = [str(LLAMA_SERVER), "-m", str(model), "--host", "127.0.0.1", "--port", str(PORT),
           "-ngl", "99", "-c", "8192", "-b", "2048", "-ub", "512"] + extra
    print(f"[card4] server: {' '.join(cmd)}", flush=True)
    log = open(log_path, "wb")                                # noqa: SIM115 - closed by caller
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc, log


def wait_ready(proc, timeout: float = 900.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:                                    # noqa: BLE001
            time.sleep(2)
    return False


def complete(prompt: str) -> dict:
    body = json.dumps({"prompt": prompt, "n_predict": MAX_NEW, "temperature": 0.0,
                       "top_k": 1, "seed": 0, "cache_prompt": False, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read().decode("utf-8"))
    tim = d.get("timings") or {}
    return {"content": d.get("content") or "",
            "n": int(d.get("tokens_predicted") or tim.get("predicted_n") or 0),
            "ms": float(tim.get("predicted_ms") or 0.0)}


def stats(outs: list[dict]) -> dict:
    tot_n = sum(o["n"] for o in outs)
    tot_ms = sum(o["ms"] for o in outs)
    return {"tps": tot_n / (tot_ms / 1000.0) if tot_ms else 0.0,
            "content": [o["content"] for o in outs]}


def identity(a: dict, b: dict) -> float:
    n = min(len(a["content"]), len(b["content"]))
    if not n:
        return 0.0
    return round(100.0 * sum(x == y for x, y in zip(a["content"], b["content"])) / n, 1)


def run_server(model: Path, extra: list[str], prompt_list: list[str], label: str,
               passes: int = 1) -> list[dict]:
    log_path = Path("/tmp") / f"card4-{label}.log"
    proc, log = start_server(model, extra, log_path)
    try:
        if not wait_ready(proc):
            tail = log_path.read_text(errors="replace")[-900:] if log_path.exists() else ""
            raise RuntimeError(f"llama-server ({label}) not ready: ...{tail}")
        runs = []
        for _ in range(passes):
            runs.append(stats([complete(p) for p in prompt_list]))
        return runs
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def main():
    global _VARIANT
    _VARIANT, backend_note = pick_backend()
    if _VARIANT is None:
        raise RuntimeError(f"no CUDA backend for llama-server ({backend_note})")
    print(f"[card4] CUDA backend: {backend_note}", flush=True)
    prompt_list, source = prompts(N)
    target = locate_target()
    errors: dict[str, str] = {}
    draft = None
    try:
        draft = locate_draft()
    except Exception as exc:                                 # noqa: BLE001
        errors["draft"] = f"{type(exc).__name__}: {exc}"

    base_runs = run_server(target, [], prompt_list, "base", passes=2)
    base = base_runs[0]
    arms = [
        C.arm("target-only", base["tps"], base["tps"], gpu_used_gb(),
              prompt_source=source, equality=100.0),
        C.arm("target-only-rerun", base_runs[1]["tps"], base_runs[1]["tps"], gpu_used_gb(),
              prompt_source=source, equality=identity(base, base_runs[1])),
    ]

    try:
        ng = run_server(target, ["--spec-type", "ngram-simple"], prompt_list, "ngram")[-1]
        arms.append(C.arm("target+ngram-simple (0 GB)", ng["tps"], ng["tps"], gpu_used_gb(),
                          speedup=round(ng["tps"] / base["tps"], 2) if base["tps"] else 0.0,
                          equality=identity(base, ng), prompt_source=source))
    except Exception as exc:                                 # noqa: BLE001
        errors["ngram"] = f"{type(exc).__name__}: {exc}"

    if draft is not None:
        try:
            dr = run_server(target, ["--spec-type", "draft-simple", "-md", str(draft)],
                            prompt_list, "draft")[-1]
            arms.append(C.arm("target+draft-0.6B", dr["tps"], dr["tps"], gpu_used_gb(),
                              speedup=round(dr["tps"] / base["tps"], 2) if base["tps"] else 0.0,
                              equality=identity(base, dr), draft=DRAFT_FILE))
        except Exception as exc:                             # noqa: BLE001
            errors["draft"] = (errors.get("draft", "") + " | " if "draft" in errors else "") \
                + f"{type(exc).__name__}: {exc}"

    spec = [a for a in arms if not a["name"].startswith("target-only")]
    best = max(spec, key=lambda a: a.get("speedup", 0.0)) if spec else None
    passed = bool(best and best.get("speedup", 0.0) >= 1.5 and best.get("equality", 0.0) >= 100.0)
    return N, arms, passed, {
        "target": TARGET_MODEL, "draft": DRAFT_FILE,
        "fixture": source, "cuda_backend": backend_note,
        "rule": "pass if a speculative arm gives >=1.5x tokens/s at 100% greedy output identity",
        "best_arm": best["name"] if best else None,
        "best_speedup": best.get("speedup", 0.0) if best else 0.0,
        "note": "re-scoped 2026-10-03 to the local 27B batch bottleneck; prompts are the shared "
                "hard-transcript extraction workload (local_batch/ingest-prompts.jsonl is not in "
                "the repo). 0 GB alternative = ngram-simple speculative decoding.",
        "errors": errors,
    }


if __name__ == "__main__":
    C.guard("04-speculative-decoding", HYPOTHESIS, METRIC, OUT, main)
