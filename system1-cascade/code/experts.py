#!/usr/bin/env python3
"""Stand-in for the private experts service: only what these benchmarks call on llama-server.

Point it at your own server with LLAMA_PORT and at a state folder with EXPERTS_STATE. The folder holds:
  registry.json  {"adapters": [{"name": "hub-jev", "gguf": "/same/path/llama-server/was/given.gguf"}],
                  "server": {"parallel": 2}}
  fetched-ref/jev/calibration.json, fetched-ref/jev/adapter_vllm/decision_head.json  from autotrust/JEV-27B-VL
  bench/  data and results
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

STATE = Path(os.environ.get("EXPERTS_STATE", Path(__file__).resolve().parent.parent / "state"))
DATA = STATE / "data"
SERVER = f"http://127.0.0.1:{os.environ.get('LLAMA_PORT', '8131')}"


def load_registry() -> dict:
    return json.loads((STATE / "registry.json").read_text())


def _http(url: str, body: dict | None = None, timeout: float = 300, headers: dict | None = None) -> object:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def live_adapters() -> list[dict]:
    return _http(f"{SERVER}/lora-adapters", timeout=5)


def lora_field(live: list[dict], scales_by_path: dict[str, float]) -> list[dict]:
    """The request's `lora` list: every loaded adapter gets an explicit scale (0 = off)."""
    return [{"id": a["id"], "scale": scales_by_path.get(a["path"], 0.0)} for a in live]


def parse_adapter_spec(spec: str) -> dict[str, float]:
    """'none' -> {}, 'a' -> {a: 1.0}, 'a:0.7,b:0.3' -> {a: 0.7, b: 0.3}."""
    if spec in ("", "none"):
        return {}
    out = {}
    for part in spec.split(","):
        name, _, scale = part.partition(":")
        out[name.strip()] = float(scale) if scale else 1.0
    return out
