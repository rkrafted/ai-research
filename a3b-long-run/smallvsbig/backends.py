"""Backends for the R7 arms.

- OllamaBackend        : the on-disk GGUF models (arms A/B and the F escalation).
- TransformersBackend  : 4-bit SERA-8B with an optional peft adapter (student C/D).
- LayerRepeatBackend   : the same base with a middle block of decoder layers run twice (G).
- RoutedBackend        : two peft adapters on one base, chosen per task kind (E).

The transformers pieces below subclass the vendored
``smallvsbig/distill/local_backend.py`` (agent-distill) so the parse/generate path
is byte-identical to the run that measured the student.
"""
from __future__ import annotations

from pathlib import Path

from .distill import agent as A
from .distill.local_backend import TransformersBackend


def ollama_base() -> str:
    import os
    return os.environ.get("OLLAMA_BASE") or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"


def make_ollama(model: str):
    A.OLLAMA_BASE = ollama_base()
    return A.OllamaBackend(model)


# ------------------------------------------------------------------ layer repeat
def _layers_of(model):
    for path in ("model.layers", "base_model.model.model.layers",
                 "base_model.model.layers", "transformer.h"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__") and len(obj) > 3:
            return path, obj
    raise RuntimeError("could not find the decoder layer list on this model")


def _set_path(model, path: str, value) -> None:
    parts = path.split(".")
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


class LayerRepeatBackend(TransformersBackend):
    """Run a middle block of decoder layers a second time (depth without new weights).

    ``repeat`` copies ``layers[start:end]`` and appends them after ``end``, so the
    same weights run twice.  This is the cheap stand-in for looping depth requested
    by arm G; no parameters are added.
    """

    def __init__(self, base: str, adapter: str | None = None, repeat: int = 1,
                 start_fraction: float = 1 / 3, end_fraction: float = 2 / 3, **kw):
        super().__init__(base, adapter=adapter, **kw)
        self.repeat = max(1, repeat)
        self.start_fraction = start_fraction
        self.end_fraction = end_fraction
        self.repeated_layers = 0

    def load(self) -> None:
        if self.model is not None:
            return
        super().load()
        import torch
        path, layers = _layers_of(self.model)
        n = len(layers)
        start = max(0, int(n * self.start_fraction))
        end = min(n, max(start + 1, int(n * self.end_fraction)))
        order = list(range(end)) + list(range(start, end)) * self.repeat + list(range(end, n))
        new = torch.nn.ModuleList([layers[i] for i in order])
        _set_path(self.model, path, new)
        self.repeated_layers = len(order) - n
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            try:
                cfg.num_hidden_layers = len(new)
            except Exception:  # noqa: BLE001 - some configs are frozen
                pass
            # Qwen3 keeps a per-layer type list; keep it consistent with the new depth.
            if getattr(cfg, "layer_types", None) and len(cfg.layer_types) == n:
                kinds = list(cfg.layer_types)
                cfg.layer_types = [kinds[i] for i in order]
        self.model_name = f"{Path(self.base).name}+repeat{self.repeated_layers}"

    def free(self) -> None:
        import gc
        try:
            import torch
        except Exception:  # noqa: BLE001
            torch = None
        self.model = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


class RoutedBackend(TransformersBackend):
    """One 4-bit base, several peft adapters, swapped per call.

    The runner sets ``current`` to the adapter key for the task; ``complete`` calls
    ``set_adapter`` before generating.  Only one base is resident, so VRAM is the
    cost of one model regardless of the number of experts.
    """

    def __init__(self, base: str, adapters: dict[str, str], current: str | None = None, **kw):
        super().__init__(base, adapter=None, **kw)
        self.adapters = dict(adapters)
        self.current = current or next(iter(self.adapters))
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        super().load()                      # base only
        from peft import PeftModel
        keys = list(self.adapters)
        first = keys[0]
        self.model = PeftModel.from_pretrained(self.model, self.adapters[first],
                                               adapter_name=first)
        for key in keys[1:]:
            self.model.load_adapter(self.adapters[key], adapter_name=key)
        self.model.eval()
        self._loaded = True
        self.model_name = f"{Path(self.base).name}+routed{len(keys)}"

    def complete(self, messages, deadline):
        self.load()
        if self.current not in self.adapters:
            self.current = next(iter(self.adapters))
        self.model.set_adapter(self.current)
        return super().complete(messages, deadline)

    def free(self) -> None:
        import gc
        try:
            import torch
        except Exception:  # noqa: BLE001
            torch = None
        self.model = None
        self._loaded = False
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
