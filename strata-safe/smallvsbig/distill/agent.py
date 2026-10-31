#!/usr/bin/env python3
"""The one agent loop used for both the teacher and the student.

The system prompt, the four tools (`bash`, `read_file`, `write_file`, `finish`,
with `submit` graded as `finish`), the 32K context cap and the 30-turn cap are
copied verbatim from `employee-shootout/harness/agent.py`, so a trajectory
produced here is exactly what the shootout harness would have run. The only
difference is the transport: `OllamaBackend` speaks ollama's native `/api/chat`
(as the shootout does), `DeepSeekBackend` speaks the OpenAI-compatible
`/chat/completions` API (the teacher).

Everything is stdlib; the API key is read from the environment or
`~/.dsh/.credentials.yaml` and is never logged or written to a file.
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

_HOME_RE = re.compile(r"/" r"home/[A-Za-z0-9_.-]*")

MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "30"))
NUM_CTX = int(os.environ.get("AGENT_NUM_CTX", "32768"))
MAX_OBS_CHARS = 6000
BASH_TIMEOUT = 60
EPISODE_TIMEOUT_S = int(os.environ.get("AGENT_EPISODE_TIMEOUT", "1500"))

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434")
DS_BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DS_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
# USD per 1M tokens: peak input (cache miss), cache hit, output.
DS_PRICE = {"input": 0.30, "cache_read": 0.006, "output": 1.20}

# `{repo}` is substituted per episode with the absolute workspace path. The
# 2026-10-04 harness bug: the old prompt never said where the repository was, so
# SERA-8B opened with `cd /` (its training convention) and never saw the tests.
# 0/28 measured the harness, not the model. Every eval must state the path.
SYSTEM_PROMPT = """You are an autonomous coding agent working inside a repository.
You are given a task in the user message. Solve it by calling the provided tools.

The repository root is {repo}. That directory is already your current working directory:
every bash command starts there. Never run `cd /`, `cd ~`, `cd /workspace`
or `cd /testbed`; run the commands directly (e.g. `python3 -m pytest -q`).

Rules:
- The working directory is a plain snapshot: there is NO git history. Do not run git commands.
- Inspect only what you need, then make the change; do not cat large files repeatedly.
- Use write_file for files. Keep each write_file short and split a long file into
  several calls, or use a bash heredoc, so tool arguments are never truncated.
- Use bash to run the test command given in the task and iterate until it passes.
- Environment variables and network access are not available; work offline.
- As soon as the tests pass, call the `finish` tool with a short summary.
- Only call the provided tools. Never print a tool call as text."""

# A `cd` to an absolute path or `~`. `/workspace` and `/testbed` do not exist on
# this host, so a command that cds there never reaches the repository either.
_CD_RE = re.compile(r"cd\s+([^\s;&|]+)")


def command_runs_test_in_repo(command: str, test_cmd: str, workspace=None) -> bool:
    """True when a model bash command looks like it ran the test from inside the repo.

    Weak by design: it is the gate's signal that the agent actually reached the
    repository, not a grader. A relative `cd` is fine; a `cd` to `~` or to an
    absolute path outside `workspace` does not count (the 2026-10-04 failure).
    """
    if "pytest" not in command and test_cmd not in command:
        return False
    ws = str(workspace).rstrip("/") if workspace is not None else None
    for m in _CD_RE.finditer(command):
        target = m.group(1).strip("'\"")
        if target.startswith("~"):
            return False
        if target.startswith("/"):
            if ws and (target == ws or target.startswith(ws + "/")):
                continue
            return False
    return True

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the repository root. Returns combined stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The command to run."}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file, relative to the repository root.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a text file, relative to the repository root.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call when the task is complete. Ends the episode.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": [],
            },
        },
    },
]


def sanitize(value):
    """Replace absolute home-directory paths with ~ before anything is committed."""
    if isinstance(value, str):
        return _HOME_RE.sub("~", value)
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    return value


def deepseek_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    path = Path.home() / ".dsh" / ".credentials.yaml"
    try:
        import yaml
        return str(yaml.safe_load(path.read_text())["refs"]["DEEPSEEK_API_KEY"])
    except Exception:  # noqa: BLE001 - missing creds is a state, not a crash
        return ""


def _post(url: str, payload: dict, headers: dict, timeout: int = 900,
          deadline: float | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    last: object = None
    for attempt in range(4):
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("episode wall-clock budget exhausted")
        per = timeout if deadline is None else max(5, min(timeout, deadline - time.time()))
        try:
            with urllib.request.urlopen(req, timeout=per) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            if "invalid tool call arguments" in body:
                raise RuntimeError(f"invalid_tool_arguments: {body}") from e
            if e.code in (401, 403):
                raise RuntimeError(f"auth error HTTP {e.code}") from e
            last = f"HTTP {e.code}: {body}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"request to {url} failed: {last}")


def _safe(workspace: Path, path: str) -> Path:
    p = (workspace / path).resolve()
    if workspace.resolve() not in p.parents and p != workspace.resolve():
        raise ValueError("path escapes the workspace")
    return p


def _truncate(text: str, limit: int = MAX_OBS_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + f"\n...[truncated {len(text) - limit} chars]...\n" + text[-limit // 2 :]


def run_tool(name: str, args: dict, workspace: Path) -> tuple[str, bool]:
    """Return (observation, is_finish)."""
    if name in ("finish", "submit"):
        return (args.get("summary") or args.get("message") or "done", True)
    if name == "bash":
        try:
            r = subprocess.run(
                ["bash", "-lc", str(args.get("command", ""))], cwd=workspace,
                capture_output=True, text=True, timeout=BASH_TIMEOUT,
            )
            return (_truncate(f"exit={r.returncode}\n{r.stdout}{r.stderr}"), False)
        except subprocess.TimeoutExpired:
            return (f"timeout after {BASH_TIMEOUT}s", False)
    if name == "read_file":
        try:
            p = _safe(workspace, args["path"])
            if not p.is_file():
                return (f"error: no such file: {args['path']}", False)
            return (_truncate(p.read_text(errors="replace")), False)
        except Exception as e:  # noqa: BLE001
            return (f"error: {e}", False)
    if name == "write_file":
        try:
            p = _safe(workspace, args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(args.get("content", "")))
            return (f"wrote {len(str(args.get('content', '')))} chars to {args['path']}", False)
        except Exception as e:  # noqa: BLE001
            return (f"error: {e}", False)
    return (f"error: unknown tool {name!r}. Available: bash, read_file, write_file, finish.", False)


def _approx_tokens(messages: list) -> int:
    return sum(len(json.dumps(m)) for m in messages) // 4


def _trim(messages: list, num_ctx: int) -> list:
    """Keep system + first user, drop oldest turns to stay under the context cap."""
    if _approx_tokens(messages) <= num_ctx - 2048:
        return messages
    head = messages[:2]
    tail = messages[2:]
    while tail and _approx_tokens(head + tail) > num_ctx - 2048:
        tail.pop(0)
    return head + tail


class OllamaBackend:
    """ollama native /api/chat, same transport as the employee-shootout."""

    name = "ollama"

    def __init__(self, model: str):
        self.model = model
        self.url = OLLAMA_BASE + "/api/chat"

    @staticmethod
    def prepare_messages(messages: list) -> list:
        """Replay tool calls in ollama's native shape.

        Regression (2026-10-04): the loop appended OpenAI-style assistant
        messages whose `function.arguments` is a JSON *string*. ollama's
        /api/chat parses that as a nested object and rejects turn 2 with
        `Value looks like object, but can't find closing '}' symbol`, so every
        gate episode died as `harness_error` at turn 2. The employee-shootout
        appends the raw response (arguments as an object) and works; do that.
        """
        out = []
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                calls = []
                for tc in m["tool_calls"]:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    calls.append({"function": {"name": fn.get("name", ""), "arguments": args}})
                mm = {"role": "assistant", "content": m.get("content") or "", "tool_calls": calls}
                if m.get("reasoning_content"):
                    mm["thinking"] = m["reasoning_content"]
                out.append(mm)
            elif m.get("role") == "tool":
                out.append({"role": "tool", "content": m.get("content", "")})
            else:
                out.append(m)
        return out

    def complete(self, messages: list, deadline: float) -> tuple[dict, dict, str]:
        resp = _post(self.url, {
            "model": self.model,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
            "keep_alive": "10m",
            "options": {"num_ctx": NUM_CTX, "temperature": 0, "num_predict": 4096},
        }, {}, deadline=deadline)
        msg = resp.get("message") or {}
        usage = {"input": resp.get("prompt_eval_count", 0) or 0,
                 "output": resp.get("eval_count", 0) or 0, "cache_read": 0}
        return msg, usage, ""


class DeepSeekBackend:
    """OpenAI-compatible /chat/completions; the key is never logged."""

    name = "deepseek"

    def __init__(self, model: str = DS_MODEL, key: str | None = None):
        self.model = model
        self.key = key if key is not None else deepseek_key()
        self.url = DS_BASE.rstrip("/") + "/chat/completions"

    def complete(self, messages: list, deadline: float) -> tuple[dict, dict, str]:
        if not self.key:
            raise RuntimeError("DEEPSEEK_API_KEY not found (env or ~/.dsh/.credentials.yaml)")
        resp = _post(self.url, {
            "model": self.model,
            "messages": messages,
            "tools": TOOLS,
            "temperature": 0,
            "max_tokens": 4096,
        }, {"Authorization": f"Bearer {self.key}"}, deadline=deadline)
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        u = resp.get("usage") or {}
        cache = u.get("prompt_cache_hit_tokens") or u.get("prompt_cache_hit") or 0
        usage = {"input": u.get("prompt_tokens", 0) or 0,
                 "output": u.get("completion_tokens", 0) or 0,
                 "cache_read": cache}
        return msg, usage, resp.get("model", self.model)


def price_usd(usage: dict) -> float:
    fresh = max(0, usage.get("input", 0) - usage.get("cache_read", 0))
    return (fresh * DS_PRICE["input"] + usage.get("cache_read", 0) * DS_PRICE["cache_read"]
            + usage.get("output", 0) * DS_PRICE["output"]) / 1e6


def _normalise_tool_calls(raw, turn: int) -> list[dict]:
    out = []
    for i, tc in enumerate(raw or []):
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args or {})
        out.append({"id": tc.get("id") or f"call_{turn}_{i}",
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args}})
    return out


def run_episode(backend, task: dict, workspace: Path, max_turns: int = MAX_TURNS,
                test_cmd: str | None = None, restore_tests=None) -> dict:
    """Run one episode. Returns a sanitized record with the full `messages` list."""
    workspace = workspace.resolve()
    issue = task.get("issue", "")
    test_cmd = test_cmd or task.get("check_cmd", "python3 -m pytest -q")
    repo = str(workspace)
    user_msg = (
        f"# Task\n{issue}\n\n"
        f"# Repository\nThe repository root is {repo} and it is already your current "
        f"working directory. Do not `cd` anywhere else; run commands directly.\n\n"
        f"# Test command\n`{test_cmd}`\n\n"
        "Run the test command first to see it fail, fix the code, then run it again. "
        "Call finish when the task is done."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(repo=repo)},
        {"role": "user", "content": user_msg},
    ]
    usage_total = {"input": 0, "output": 0, "cache_read": 0}
    submit_calls = 0
    finish_called = False
    ran_test = False
    turns = 0
    timed_out = False
    parse_errors: list[str] = []
    started = time.time()
    deadline = started + EPISODE_TIMEOUT_S
    error = ""
    try:
        while turns < max_turns:
            if time.time() >= deadline:
                timed_out = True
                break
            turns += 1
            trimmed = _trim(messages, NUM_CTX)
            prepare = getattr(backend, "prepare_messages", None)
            try:
                msg, usage, _model = backend.complete(
                    prepare(trimmed) if prepare else trimmed, deadline)
            except TimeoutError:
                # `_post` raises TimeoutError for the episode wall clock; a slow
                # local model is a timeout, never a `harness_error` (missing model).
                timed_out = True
                break
            except RuntimeError as e:
                if "invalid_tool_arguments" not in str(e):
                    if time.time() >= deadline:
                        timed_out = True
                        break
                    error = str(e)[:300]
                    break
                obs = (
                    "Your last tool call was rejected: its JSON arguments were truncated "
                    "or invalid (usually because the file content was too long). Write a "
                    "shorter file, or split it into several write_file calls, then continue."
                )
                messages.append({"role": "assistant", "content": "", "tool_calls": []})
                messages.append({"role": "user", "content": obs})
                continue
            for k in usage_total:
                usage_total[k] += usage.get(k, 0) or 0
            tool_calls = _normalise_tool_calls(msg.get("tool_calls"), turns)
            assistant = {"role": "assistant", "content": msg.get("content") or ""}
            if tool_calls:
                assistant["tool_calls"] = tool_calls
                if msg.get("reasoning_content"):
                    assistant["reasoning_content"] = msg["reasoning_content"]
            messages.append(assistant)

            if not tool_calls:
                messages.append({"role": "user",
                                 "content": "You did not call a tool. Continue with a tool call, then call finish."})
                continue

            for tc in tool_calls:
                fn = tc["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                except json.JSONDecodeError:
                    args = {"_raw": fn["arguments"]}
                    parse_errors.append(f"turn {turns}: {name}: invalid JSON arguments")
                if not isinstance(args, dict):
                    args = {"_raw": args}
                if name == "submit":
                    submit_calls += 1
                if name == "bash" and command_runs_test_in_repo(
                        str(args.get("command", "")), test_cmd, workspace):
                    ran_test = True
                obs, is_finish = run_tool(name, args, workspace)
                if is_finish:
                    finish_called = True
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "name": name, "content": obs})
            if finish_called:
                break
    except Exception as e:  # noqa: BLE001 - one bad episode must not kill the batch
        error = f"{type(e).__name__}: {e}"[:300]
    wall = time.time() - started

    if restore_tests is not None:
        try:
            restore_tests()
        except Exception as e:  # noqa: BLE001
            error = error or f"restore_tests: {e}"

    test = subprocess.run(["bash", "-lc", test_cmd], cwd=workspace,
                          capture_output=True, text=True, timeout=900)
    test_out = (test.stdout + test.stderr)[-4000:]
    passed = test.returncode == 0

    if passed:
        failure_mode = "pass"
    elif timed_out:
        failure_mode = "timeout"
    elif error:
        failure_mode = "harness_error"
    elif not finish_called and turns >= max_turns:
        failure_mode = "turn_limit"
    elif submit_calls:
        failure_mode = "test_failure_after_submit"
    else:
        failure_mode = "test_failure"

    return sanitize({
        # `TransformersBackend.model` is the nn.Module; its `model_name` is the
        # string. Prefer the string so the record stays JSON-serializable.
        "model": getattr(backend, "model_name", None) or getattr(backend, "model", backend.name),
        "backend": backend.name,
        "task": task.get("id", "adhoc"),
        "passed": passed,
        "turns": turns,
        "max_turns": max_turns,
        "num_ctx": NUM_CTX,
        "usage": usage_total,
        "cost_usd": round(price_usd(usage_total), 6),
        "wall_time_s": round(wall, 2),
        "submit_calls": submit_calls,
        "finish_called": finish_called,
        "ran_test": ran_test,
        "test_cmd": test_cmd,
        "test_exit": test.returncode,
        "test_output": test_out,
        "failure_mode": failure_mode,
        "error": error,
        "parse_errors": parse_errors,
        "messages": messages,
        "started_at": started,
        "ended_at": time.time(),
    })
