"""Archive-hygiene checks for the vendored engines (offline, stdlib only).

The runner scans a `git archive` of the repo with the publish_check scanners:
no real emails, no file larger than 2 MiB, no `cv`/`resume`-named files and no
model weights.  These checks keep `vendor/` inside those rules.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = [p for p in (ROOT / "vendor" / "strata", ROOT / "vendor" / "colibri") if p.is_dir()]
EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
CV_NAME = re.compile(r"^(cv|resume)[._-]?", re.I)
MAX_BYTES = 2 * 1024 * 1024
WEIGHTS = {".gguf", ".safetensors"}


def _files():
    for root in VENDOR:
        yield from (p for p in root.rglob("*") if p.is_file())


def test_vendor_has_no_model_weights():
    assert not [p for p in _files() if p.suffix.lower() in WEIGHTS]


def test_vendor_files_fit_the_archive_scanner():
    too_big = [(str(p.relative_to(ROOT)), p.stat().st_size) for p in _files() if p.stat().st_size > MAX_BYTES]
    cv_names = [str(p.relative_to(ROOT)) for p in _files() if CV_NAME.match(p.name)]
    assert not too_big, too_big
    assert not cv_names, cv_names


def test_vendor_sources_carry_no_real_email():
    hits = []
    for p in _files():
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for m in EMAIL.finditer(line):
                domain = m.group(0).split("@", 1)[1].lower()
                if domain not in {"example.com", "example.org", "example.net"} and not domain.endswith(
                    (".example", ".invalid", ".test", ".localhost")
                ):
                    hits.append(f"{p.relative_to(ROOT)}:{i}")
    assert not hits, hits[:20]
