#!/usr/bin/env python3
"""Strata / llama.cpp fit math for this box, with the constants used in docs/STRATA_SAFE.md.

No model file is opened. Values are metadata: ISTA/SC117 file sizes, the upstream
Strata RAM table, and measurements from this machine (NVMe, PCIe, RAM).

    python3 tools/strata_fit.py            # markdown table
    python3 tools/strata_fit.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json

VRAM_GB = 24.0
SIDECARS_GB = 1.5          # Laya / xlmr
DENSE_GB = 4.2             # dense weights + KV + session in VRAM
VRAM_USABLE = VRAM_GB - SIDECARS_GB
EXPERT_CACHE_GB = VRAM_USABLE - DENSE_GB
OS_RESERVE_GB = 8.0
RAM_NOW_GB = 30.0
RAM_AFTER_GB = 62.0
RAM_BW_GBPS = 60.0         # DDR5 dual channel, practical
PCIE_GBPS = 55.0           # PCIe 4.0 x16, practical
NVME_GBPS = 3.97           # measured 1 MiB random on the 980 PRO
HIT_EXPONENT = 0.12        # calibrated on the 12 GB Strata run and PR 29887
ROUTED_PER_TOKEN = 10
EXPERTS = 48 * 512

# download GB, expert-arena GB, upstream Strata RAM floor GB, 3090 decode anchor tok/s
QUANTS = {
    "Q2_0":     {"download_gb": 66.4, "experts_gb": 34.0, "ram_floor_gb": 48, "anchor_tok_s": 100.0},
    "IQ2_XS":   {"download_gb": 68.0, "experts_gb": 35.5, "ram_floor_gb": 48, "anchor_tok_s": 87.0},
    "IQ3_XXS":  {"download_gb": 75.8, "experts_gb": 42.9, "ram_floor_gb": 60, "anchor_tok_s": 73.0},
    "IQ3_S":    {"download_gb": 83.6, "experts_gb": 50.3, "ram_floor_gb": 62, "anchor_tok_s": 59.0},
}
STRATA_PREFILL = {"Q2_0": 1750, "IQ2_XS": 1550, "IQ3_XXS": 1400, "IQ3_S": 1200}


def usable_ram(ram_total_gb: float) -> float:
    return max(0.0, ram_total_gb - OS_RESERVE_GB)


def expert_bytes_per_token(experts_gb: float) -> float:
    """GB of expert weights touched by one token (10 of 512 routed per layer)."""
    return experts_gb * ROUTED_PER_TOKEN / 512.0


def hit_rate(experts_gb: float) -> float:
    """VRAM expert-cache hit rate, calibrated power law, clamped."""
    fraction = min(1.0, EXPERT_CACHE_GB / experts_gb)
    return min(0.97, fraction ** HIT_EXPONENT)


def fit(quant: str, ram_total_gb: float) -> dict:
    q = QUANTS[quant]
    E = q["experts_gb"]
    R = usable_ram(ram_total_gb)
    h = hit_rate(E)
    e = expert_bytes_per_token(E)
    ram_miss = 0.75 * (1.0 - h) * e
    pcie_miss = 0.25 * (1.0 - h) * e
    pinned = min(1.0, R / E)
    nvme_miss = (1.0 - h) * max(0.0, 1.0 - pinned) * e

    strata_fits = E <= R                      # experts pinned in RAM
    llama_fits = E <= R + EXPERT_CACHE_GB     # experts split VRAM cache + RAM

    def ceiling(miss_gb: float, bw: float) -> float:
        return float("inf") if miss_gb <= 0 else bw / miss_gb

    forced = None
    if strata_fits:
        scale = min(1.0, R / q["ram_floor_gb"])
        strata_tok_s = (q["anchor_tok_s"] * 0.90 * scale, q["anchor_tok_s"] * 1.05)
        prefill = STRATA_PREFILL[quant]
    else:
        forced = min(ceiling(nvme_miss, NVME_GBPS) * 0.30, q["anchor_tok_s"] * 0.5)
        strata_tok_s = (0.0, 0.0)
        prefill = 0.0

    if llama_fits:
        llama_tok_s = (q["anchor_tok_s"] * 0.55, q["anchor_tok_s"] * 0.65)
    else:
        llama_tok_s = (min(ceiling(nvme_miss, NVME_GBPS) * 0.35, q["anchor_tok_s"] * 0.45),
                       q["anchor_tok_s"] * 0.55)

    return {
        "quant": quant,
        "download_gb": q["download_gb"],
        "experts_gb": E,
        "ram_usable_gb": round(R, 1),
        "cache_fraction": round(EXPERT_CACHE_GB / E, 3),
        "hit_rate": round(h, 3),
        "expert_bytes_per_token_gb": round(e, 3),
        "ram_miss_gb_per_token": round(ram_miss, 4),
        "pcie_miss_gb_per_token": round(pcie_miss, 4),
        "nvme_miss_gb_per_token": round(nvme_miss, 4),
        "nvme_ceiling_tok_s": None if nvme_miss <= 0 else round(NVME_GBPS / nvme_miss, 1),
        "strata_fits": strata_fits,
        "llama_fits": llama_fits,
        "strata_tok_s": [round(x, 1) for x in strata_tok_s],
        "strata_forced_tok_s": None if forced is None else round(forced, 1),
        "llama_tok_s": [round(x, 1) for x in llama_tok_s],
        "strata_prefill_tok_s": round(prefill, 0),
        "llama_prefill_tok_s": 250 if ram_total_gb < 50 else 300,
    }


def table() -> str:
    rows = []
    for ram in (RAM_NOW_GB, RAM_AFTER_GB):
        for name in QUANTS:
            f = fit(name, ram)
            strata_cell = (f"{f['strata_tok_s'][0]:.0f}-{f['strata_tok_s'][1]:.0f}"
                           if f["strata_fits"] else "—")
            prefill_cell = f"{f['strata_prefill_tok_s']:.0f}" if f["strata_fits"] else "—"
            rows.append(
                f"| {ram:.0f} GB | {name} | {f['download_gb']:.1f} | {f['experts_gb']:.1f} "
                f"| {f['hit_rate'] * 100:.1f}% | {'yes' if f['strata_fits'] else 'no'} "
                f"| {'yes' if f['llama_fits'] else 'no'} "
                f"| {strata_cell} "
                f"| {f['llama_tok_s'][0]:.0f}-{f['llama_tok_s'][1]:.0f} "
                f"| {prefill_cell} |"
            )
    head = (
        "| RAM | quant | download GB | experts GB | hit | Strata fits | llama.cpp fits "
        "| Strata tok/s | llama.cpp tok/s | Strata prefill tok/s |\n"
        "|---|---|---:|---:|---:|---|---|---:|---:|---:|"
    )
    return head + "\n" + "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.json:
        print(json.dumps({f"{int(r)}": {q: fit(q, r) for q in QUANTS}
                          for r in (RAM_NOW_GB, RAM_AFTER_GB)}, indent=1))
    else:
        print(table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
