#!/usr/bin/env python3
"""In-process backend for the base model and the LoRA student.

ollama cannot serve this student: its safetensors `ADAPTER` import supports
Llama/Mistral/Gemma only, and a QLoRA (4-bit base) adapter cannot be merged into
the existing Q4 GGUF cleanly. So the held-out eval loads `allenai/SERA-8B` 4-bit
with transformers, optionally attaches the peft adapter, renders the same Qwen3
chat template (with the same `TOOLS`), greedily generates, and parses the model's
`<tool_call>{"name": ..., "arguments": ...}</tool_call>` blocks into the same
OpenAI-style `tool_calls` the agent loop already consumes.

This module is stdlib at import time so CPU tests can exercise the parser; the
GPU stack (torch/transformers/peft) is imported lazily inside `load()`.

    from distill.local_backend import TransformersBackend
    backend = TransformersBackend("allenai/SERA-8B", adapter="state/adapters/sera-8b-r1")
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

try:  # `python -m distill.eval`
    from . import agent as A
except ImportError:  # `python distill/local_backend.py` / direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from distill import agent as A

DEFAULT_BASE = "allenai/SERA-8B"

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_OPEN_TOOL_RE = re.compile(r"<tool_call>\s*(.*)$", re.S)


def _make_call(raw: str, index: int) -> dict:
    """Normalise one `<tool_call>` body into an OpenAI-style tool call."""
    raw = (raw or "").strip()
    obj = None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        obj = None
    if isinstance(obj, dict):
        name = obj.get("name") or obj.get("function") or ""
        if "arguments" in obj:
            args = obj["arguments"]
        else:
            args = {k: v for k, v in obj.items() if k not in ("name", "function")}
        if isinstance(args, str):
            arg_str = args
        else:
            arg_str = json.dumps(args, ensure_ascii=False)
        return {"id": f"call_{index}", "type": "function",
                "function": {"name": str(name), "arguments": arg_str}}
    return {"id": f"call_{index}", "type": "function",
            "function": {"name": "", "arguments": raw}}


def _looks_like_bare_call(text: str) -> bool:
    text = text.strip()
    if not text.startswith("{"):
        return False
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(obj, dict) and ("name" in obj or "arguments" in obj)


def parse_assistant_text(text: str) -> tuple[str, str, list[dict]]:
    """Split a raw generation into (content, reasoning, tool_calls).

    Handles the Qwen3/SERA format: an optional ``<think>...</think>`` block, free
    content, and one or more ``<tool_call>{...}</tool_call>`` blocks. A trailing
    block the model failed to close is kept (its JSON may still be complete), and
    a bare JSON object with `name`/`arguments` is accepted as a single call so the
    base model is not scored as "did not call a tool" for a formatting slip.
    """
    text = text or ""
    reasoning = ""
    m = _THINK_RE.search(text)
    if m:
        reasoning = m.group(1).strip()
    body = _THINK_RE.sub("", text)

    calls: list[dict] = []
    for match in _TOOL_RE.finditer(body):
        calls.append(_make_call(match.group(1), len(calls)))
    body = _TOOL_RE.sub("", body)

    open_match = _OPEN_TOOL_RE.search(body)
    if open_match:
        calls.append(_make_call(open_match.group(1), len(calls)))
        body = body[: open_match.start()]

    body = body.strip()
    if not calls and _looks_like_bare_call(body):
        calls.append(_make_call(body, 0))
        body = ""
    return body, reasoning, calls


class TransformersBackend:
    """Greedy 4-bit generation with the shootout's chat template and tools."""

    name = "transformers"

    def __init__(self, base: str = DEFAULT_BASE, adapter: str | None = None,
                 max_new_tokens: int = 2048, load_in_4bit: bool = True,
                 device: str | None = None):
        self.base = base
        self.adapter = str(adapter) if adapter else None
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_name = Path(base).name

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.base, trust_remote_code=True)
        kwargs: dict = {"dtype": torch.bfloat16}
        if self.load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        kwargs["device_map"] = {"": 0} if self.device == "cuda" else None
        self.model = AutoModelForCausalLM.from_pretrained(self.base, **kwargs)
        if self.adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter)
        self.model.eval()
        if self.adapter:
            self.model_name = f"{Path(self.base).name}+{Path(self.adapter).name}"

    def complete(self, messages: list, deadline: float) -> tuple[dict, dict, str]:
        import torch

        self.load()
        if time.time() >= deadline:
            raise TimeoutError("episode wall-clock budget exhausted")
        text = self.tokenizer.apply_chat_template(
            messages, tools=A.TOOLS, tokenize=False, add_generation_prompt=True,
        )
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)
        attention = enc["attention_mask"].to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new = out[0][input_ids.shape[1]:]
        raw = self.tokenizer.decode(new, skip_special_tokens=True)
        content, reasoning, calls = parse_assistant_text(raw)
        msg: dict = {"role": "assistant", "content": content}
        if calls:
            msg["tool_calls"] = calls
        if reasoning:
            msg["reasoning_content"] = reasoning
        usage = {"input": int(input_ids.shape[1]), "output": int(new.shape[0]),
                 "cache_read": 0}
        return msg, usage, self.model_name


def model_present(adapter: str | None) -> bool:
    """True when the artifact this backend needs exists on disk."""
    if adapter is None:
        return True
    return (Path(adapter) / "adapter_config.json").is_file()
