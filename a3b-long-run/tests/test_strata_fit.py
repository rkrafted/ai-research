"""Offline checks for the Strata fit math (no model files, no network)."""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("strata_fit", ROOT / "tools" / "strata_fit.py")
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)


def test_expert_bytes_per_token_formula():
    # 10 of 512 routed experts per layer is 1/51.2 of the arena.
    assert abs(sf.expert_bytes_per_token(51.2) - 1.0) < 1e-12
    assert abs(sf.expert_bytes_per_token(34.0) - 34.0 / 51.2) < 1e-12


def test_fit_flags_at_both_ram_sizes():
    # 30 GB: only the two 2-bit quants fit in VRAM cache + RAM, no Strata quant.
    for q in sf.QUANTS:
        assert sf.fit(q, 30)["strata_fits"] is False
    assert sf.fit("Q2_0", 30)["llama_fits"] is True
    assert sf.fit("IQ2_XS", 30)["llama_fits"] is True
    assert sf.fit("IQ3_XXS", 30)["llama_fits"] is False
    assert sf.fit("IQ3_S", 30)["llama_fits"] is False
    # 62 GB: all four fit Strata; IQ3_S is the edge (50.3 <= 54).
    for q in sf.QUANTS:
        assert sf.fit(q, 62)["strata_fits"] is True


def test_hit_rate_and_decode_bounds():
    for q in sf.QUANTS:
        h = sf.fit(q, 62)["hit_rate"]
        assert 0.0 < h < 1.0
        lo, hi = sf.fit(q, 62)["strata_tok_s"]
        assert 0 < lo <= hi
        # more experts in the arena cannot increase the hit rate
    assert sf.fit("Q2_0", 62)["hit_rate"] >= sf.fit("IQ3_S", 62)["hit_rate"]


def test_table_has_62_gb_and_tok_s():
    text = sf.table()
    assert "62 GB" in text
    assert "tok/s" in text
    assert "IQ3_XXS" in text
