"""GPU job: which KV-cache type buys the A3B more context on one 3090.

For each KV configuration (K and V set separately) this script
  1. binary-searches the largest --ctx-size that starts beside the DFlash draft
     at one slot, and parses the server's own KV/compute buffer sizes;
  2. loads one server at that maximum and measures prefill/decode tok/s on the
     real mixed long contexts from ``kvfrontier.corpus``;
  3. scores needle and multi-hop answers from the far end of the context.

Results append to ``state/kvfrontier/results.jsonl`` (gitignored) and are
printed as one JSON line per cell, so a rerun resumes where it stopped.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                      # run as `python3 kvfrontier/measure.py`
    sys.path.insert(0, str(ROOT))
STATE = ROOT / "state" / "kvfrontier"
HUB = Path.home() / ".cache" / "huggingface" / "hub"

LLAMA_CANDIDATES = [
    Path.home() / "theprojects" / "queue" / ".worktrees" / "moe-offload" / "moe-offload"
    / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server",
    Path.home() / "theprojects" / "queue" / ".worktrees" / "coder27b-q3" / "coder27b-q3"
    / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server",
]
MODEL_GLOB = "models--ggml-org--Qwen3.6-35B-A3B-GGUF/snapshots/*/Qwen3.6-35B-A3B-Q4_K_M.gguf"
DRAFT_GLOB = "models--ggml-org--Qwen3.6-35B-A3B-GGUF/snapshots/*/dflash-Qwen3.6-35B-A3B-Q8_0.gguf"

#: (name, K type, V type, extra env).  f16 is the full-precision baseline.
CONFIGS = [
    ("f16", "f16", "f16", {}),
    # The candidate comes second: if anything still kills the job, the cell that
    # decides the project is already written.
    ("q4_0", "q4_0", "q4_0", {}),
    ("q8_0", "q8_0", "q8_0", {}),
    ("q5_1", "q5_1", "q5_1", {}),
    ("k-q8_0-v-q4_0", "q8_0", "q4_0", {}),
    ("k-q4_0-v-q8_0", "q4_0", "q8_0", {}),
    # Hadamard rotation is on by default for quantized KV; this isolates it.
    ("q4_0-no-rot", "q4_0", "q4_0", {"LLAMA_ATTN_ROT_DISABLE": "1"}),
]
PROMPT_KINDS = ("needle", "multihop")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_json(port: int, path: str, payload: dict | None, timeout: float) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_health(port: int, proc: subprocess.Popen, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            http_json(port, "/health", None, timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


class Server:
    def __init__(self, name: str, ctk: str, ctv: str, ctx: int, env: dict | None = None):
        self.name, self.ctk, self.ctv, self.ctx = name, ctk, ctv, ctx
        self.env = dict(env or {})
        self.port = free_port()
        self.proc: subprocess.Popen | None = None
        self.log_path = STATE / "logs" / f"{name}-{ctx}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def cmd(self) -> list[str]:
        model = sorted(glob.glob(str(HUB / MODEL_GLOB)))[0]
        draft = sorted(glob.glob(str(HUB / DRAFT_GLOB)))[0]
        return [
            str(llama_binary()), "-m", model, "-c", str(self.ctx), "-ngl", "99",
            "-fa", "on", "-ctk", self.ctk, "-ctv", self.ctv,
            "-md", draft, "--spec-type", "draft-dflash", "-ngld", "99",
            "--spec-draft-n-max", "6",
            "--host", "127.0.0.1", "--port", str(self.port),
            "--parallel", "1", "--temp", "0", "--jinja", "--no-warmup",
        ]

    def start(self, timeout_s: float = 180) -> bool:
        env = dict(os.environ)
        env.update(self.env)
        with self.log_path.open("wb") as fh:
            self.proc = subprocess.Popen(self.cmd(), stdout=fh, stderr=fh, env=env)
        ok = wait_health(self.port, self.proc, timeout_s)
        if not ok:
            self.stop()
        return ok

    def probe(self, timeout_s: float = 300) -> bool:
        """Force one real decode at the serving batch size.

        `/health` only proves that the model and KV buffer loaded. The compute
        buffers for prompt processing are allocated on the first prompt, sized
        by the batch (2048 tokens here), not by a one-token hello. So the probe
        prefills a full batch: the 2026-10-18 run called ctx=81920 `fits` for
        q8_0 after a health check and then died on the first 32K prompt with
        "failed to allocate compute pp buffers" (332 MiB).
        """
        try:
            http_json(self.port, "/v1/chat/completions", {
                "messages": [{"role": "user", "content": "probe " * 3000}],
                "max_tokens": 1, "temperature": 0, "stream": False,
                "cache_prompt": False,
            }, timeout=timeout_s)
        except Exception:
            return False
        return not self.memory_info().get("oom", False)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def memory_info(self) -> dict:
        text = self.log_text()
        info: dict = {}
        m = re.search(r"KV buffer size =\s*([\d.]+) MiB", text)
        if m:
            info["kv_buffer_mib"] = float(m.group(1))
        m = re.search(r"size =\s*([\d.]+) MiB.*?K \(([^)]+)\):\s*([\d.]+) MiB, "
                      r"V \(([^)]+)\):\s*([\d.]+) MiB", text)
        if m:
            info.update(kv_total_mib=float(m.group(1)), k_type=m.group(2),
                        k_mib=float(m.group(3)), v_type=m.group(4), v_mib=float(m.group(5)))
        m = re.search(r"compute buffer size =\s*([\d.]+) MiB", text)
        if m:
            info["compute_buffer_mib"] = float(m.group(1))
        m = re.search(r"n_rot_k = (\d+).*?n_rot_v = (\d+)", text)
        if m:
            info["n_rot_k"], info["n_rot_v"] = int(m.group(1)), int(m.group(2))
        info["oom"] = bool(re.search(r"out of memory|CUDA error|failed to allocate", text, re.I))
        return info

    def tokenize(self, text: str, timeout: float = 180) -> list[int]:
        return http_json(self.port, "/tokenize", {"content": text}, timeout)["tokens"]


def llama_binary() -> Path:
    for path in LLAMA_CANDIDATES:
        if path.is_file():
            return path
    raise SystemExit("llama-server not found; expected moe-offload's CUDA build")


def trim_to_tokens(srv: Server, text: str, target: int) -> tuple[str, int]:
    """Cut ``text`` so the model's own tokenizer reports at most ``target`` tokens."""
    n = len(srv.tokenize(text))
    if n <= target:
        return text, n
    cut = max(1, int(len(text) * target / n * 0.98))
    for _ in range(4):
        piece = text[:cut]
        n = len(srv.tokenize(piece))
        if n <= target:
            return piece, n
        cut = max(1, int(cut * target / n * 0.995))
    return text[:cut], n


def fit_ctx(name: str, ctk: str, ctv: str, env: dict, lo: int, hi: int,
            step: int) -> tuple[int, dict]:
    """Largest ctx that starts *and* survives one real decode, on a 16K grid."""
    def attempt(ctx: int) -> tuple[bool, dict]:
        srv = Server(name, ctk, ctv, ctx, env)
        ok = srv.start()
        if ok:
            ok = srv.probe()
        info = srv.memory_info()
        srv.stop()
        return ok, info

    sizes = list(range(lo, hi + 1, step))
    best, best_info = 0, {}
    a, b = 0, len(sizes) - 1
    while a <= b:
        mid = (a + b) // 2
        ok, info = attempt(sizes[mid])
        if ok:
            best, best_info, a = sizes[mid], info, mid + 1
        else:
            b = mid - 1
    return best, best_info


def load_results(path: Path) -> dict[tuple, dict]:
    """Completed cells only: an errored cell must be retried on the next run."""
    done: dict[tuple, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in row:
                continue
            done[(row["config"], row["context"], row["kind"])] = row
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=220, help="wall budget")
    ap.add_argument("--configs", default="", help="comma list; default all")
    ap.add_argument("--contexts", default="", help="comma list of context ids")
    ap.add_argument("--dry-run", action="store_true", help="build contexts, no GPU")
    args = ap.parse_args()

    from kvfrontier import corpus, needles
    from kvfrontier.report import kv_bytes_per_token

    spec = corpus.load_contexts_spec()
    needles_spec = needles.load()
    contexts = corpus.build_all(spec)
    selected = [c for c in spec["contexts"] if not args.contexts or c["id"] in args.contexts.split(",")]
    configs = [c for c in CONFIGS if not args.configs or c[0] in args.configs.split(",")]

    for ctx in selected:
        built = contexts[ctx["id"]]
        log(f"built {ctx['id']}: {built['chars']:,} chars, est {built['estimated_tokens']:,} tokens, "
            f"docs { {k: len(v) for k, v in built['docs'].items()} }")
    if args.dry_run:
        return 0
    if not contexts:
        raise SystemExit("no contexts built (missing sources?)")

    STATE.mkdir(parents=True, exist_ok=True)
    results_path = STATE / "results.jsonl"
    done = load_results(results_path)
    deadline = time.time() + args.minutes * 60
    summary: dict = {"engine": "llama.cpp", "llama": str(llama_binary()),
                     "model": sorted(glob.glob(str(HUB / MODEL_GLOB)))[0],
                     "draft": sorted(glob.glob(str(HUB / DRAFT_GLOB)))[0],
                     "contexts": [c["id"] for c in selected],
                     "configs": {}}

    def save_summary() -> None:
        (STATE / "measured.json").write_text(
            json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    for name, ctk, ctv, env in configs:
        if time.time() > deadline:
            log("budget spent before " + name)
            break
        log(f"fit {name} (K={ctk} V={ctv} env={env})")
        max_ctx, mem = fit_ctx(name, ctk, ctv, env, 32768, 262144, 16384)
        summary["configs"][name] = {"ctk": ctk, "ctv": ctv, "env": env,
                                    "max_ctx": max_ctx,
                                    "kv_model_mib": round(
                                        kv_bytes_per_token(ctk, ctv) * max_ctx / (1024 * 1024), 1),
                                    **mem}
        save_summary()                           # survive a crash in the next config
        log(f"  max_ctx={max_ctx} kv={mem.get('kv_total_mib')} MiB "
            f"compute={mem.get('compute_buffer_mib')} MiB rot={mem.get('n_rot_k')}")
        if max_ctx < 32768:
            continue

        srv = Server(name, ctk, ctv, max_ctx, env)
        if not srv.start():
            log(f"  server failed at max_ctx {max_ctx}: {srv.memory_info()}")
            continue
        try:
            # Trim the shared base once per context with the model's tokenizer.
            for ctx in selected:
                if time.time() > deadline:
                    break
                ctx_id = ctx["id"]
                requested = int(ctx["tokens"])
                # Measure the largest prompt that fits in this config's cache,
                # capped at the context's nominal length (32K/64K/128K).
                target = min(requested, max_ctx - 512)
                if target < 4096:
                    log(f"  skip {ctx_id}: max_ctx {max_ctx} too small")
                    continue
                base = contexts[ctx_id]["text"]
                base, base_tokens = trim_to_tokens(srv, base, target - 512)
                for kind in PROMPT_KINDS:
                    key = (name, ctx_id, kind)
                    if key in done:
                        continue
                    if time.time() > deadline:
                        break
                    prompt, questions = needles.prompt_for(needles_spec, kind, base)
                    tokens = len(srv.tokenize(prompt))
                    if tokens > max_ctx - 128:
                        # Never cut the questions off: shrink the document and rebuild.
                        base, base_tokens = trim_to_tokens(
                            srv, base, max(1024, base_tokens - (tokens - (max_ctx - 128)) - 64))
                        prompt, questions = needles.prompt_for(needles_spec, kind, base)
                        tokens = len(srv.tokenize(prompt))
                    log(f"  {name} {ctx_id} {kind}: prompt {tokens} tok")
                    t0 = time.time()
                    try:
                        resp = http_json(srv.port, "/v1/chat/completions", {
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 128, "temperature": 0, "top_k": 1,
                            "stream": False, "cache_prompt": False,
                            "chat_template_kwargs": {"enable_thinking": False},
                        }, timeout=1800)
                        choice = resp["choices"][0]["message"].get("content") or ""
                        timings = resp.get("timings", {})
                        scored = needles.score(choice, questions)
                        row = {
                            "config": name, "ctk": ctk, "ctv": ctv, "env": env,
                            "context": ctx_id, "target_tokens": requested,
                            "measured_tokens": target,
                            "kind": kind, "prompt_tokens": int(timings.get("prompt_n", tokens)),
                            "max_ctx": max_ctx, "kv_total_mib": mem.get("kv_total_mib"),
                            "rot_k": mem.get("n_rot_k"),
                            "prefill_tok_s": round(float(timings.get("prompt_per_second", 0)), 2),
                            "decode_tok_s": round(float(timings.get("predicted_per_second", 0)), 2),
                            "predicted_n": int(timings.get("predicted_n", 0)),
                            "seconds": round(time.time() - t0, 1),
                            "answer": choice[:2000],
                            "score": scored["score"], "hits": scored["hits"],
                            "questions": scored["questions"], "misses": scored["misses"],
                        }
                    except Exception as exc:                     # noqa: BLE001
                        row = {"config": name, "ctk": ctk, "ctv": ctv, "env": env,
                               "context": ctx_id, "target_tokens": requested,
                               "measured_tokens": target, "kind": kind,
                               "max_ctx": max_ctx, "error": f"{type(exc).__name__}: {exc}",
                               "seconds": round(time.time() - t0, 1)}
                    with results_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done[key] = row
                    log(f"    -> {row.get('score', 'ERR')} {row.get('prefill_tok_s', '')} "
                        f"prefill / {row.get('decode_tok_s', '')} decode")
        finally:
            srv.stop()
        save_summary()

    save_summary()
    log(f"wrote {results_path} ({len(done)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
