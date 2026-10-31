"""Recompute the committed M4 matrix from compact raw pass durations, without fse/MLX.

This reads evidence only. It never encodes, downloads, profiles or reserves budget.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

DEFAULT = Path(__file__).parent / "evidence/dr147-m4/matrix.json"


def spread(values: list[float]) -> dict:
    return {
        "values": values,
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "width": max(values) - min(values) if len(values) > 1 else None,
    }


def _finite(value) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


def _load(pair: dict, expected: int, limit: float) -> str:
    low, high = pair["load"]
    if (
        pair["guard_count"] != expected
        or not all(_finite(v) for v in (low, high))
        or not 0 <= low <= high < limit
    ):
        raise ValueError("missing or invalid accepted load observations")
    if high < 3.89:
        return "low"
    return "elevated-only" if low >= 3.89 else "mixed"


def _sides(cell: dict) -> list[str]:
    names = ["cpu1", "cpu10", "cpu14", "mlx"]
    if (
        cell["model"] == "static-retrieval-mrl-en-v1"
        and cell["dim"] == 1024
        and cell["corpus"] == "wikitext"
        and cell["dtype"] == "float32"
    ):
        names.append("cpu_default")
    return names + [n + "_aa" for n in names]


def _pair(pair: dict, names: list[str], passes: int, limit: float) -> dict:
    regime = _load(pair, len(names) * (2 + 2 * passes), limit)
    if set(pair["passes"]) != set(names):
        raise ValueError("missing or unexpected paired/A/A side")
    seconds = {}
    for name, values in pair["passes"].items():
        if len(values) != passes or not all(_finite(v) and v > 0 for v in values):
            raise ValueError("invalid pass count or duration")
        seconds[name] = min(values)
    comparisons = [("mlx", n) for n in names if n.startswith("cpu") and not n.endswith("_aa")]
    comparisons += [(n + "_aa", n) for n in names if not n.endswith("_aa")]
    return {
        "regime": regime,
        "load": pair["load"],
        "seconds": seconds,
        "ratios": {f"{a}/{b}": seconds[a] / seconds[b] for a, b in comparisons},
    }


def summarize_cell(cell: dict, limit: float, texts: int) -> dict:
    parity = cell["parity"]
    required = (
        "bit_equal",
        "contract_passed",
        "worker_determinism",
        "corpus_device_launch_observed",
    )
    if (
        any(parity[k] is not True for k in required)
        or cell["restart"] is not True
        or parity["observed_run_fallback_events"]
        or cell["timed_fallback_events"]
        or parity["differing_values"] != 0
    ):
        raise ValueError("incomplete correctness, restart or GPU evidence")
    if len(cell["pairs"]) != 3 or [p["round"] for p in cell["pairs"]] != [0, 1, 2]:
        raise ValueError("acceptance needs exactly three distinct paired rounds")
    names = _sides(cell)
    passes = 3 if cell["corpus"] == "wikitext" else 2
    pairs = [_pair(p, names, passes, limit) for p in cell["pairs"]]
    ratios = {k: spread([p["ratios"][k] for p in pairs]) for k in pairs[0]["ratios"]}
    return {
        "key": cell["key"],
        "pairs": pairs,
        "ratios": ratios,
        "texts_per_s": {n: spread([texts / p["seconds"][n] for p in pairs]) for n in names},
        "regimes": dict(Counter(p["regime"] for p in pairs)),
        "binary_load": "below 3.89"
        if all(p["regime"] == "low" for p in pairs)
        else "exposed >=3.89",
        "by_regime": _groups(pairs),
    }


def _groups(pairs: list[dict]) -> dict:
    return {
        regime: {
            k: spread([p["ratios"][k] for p in pairs if p["regime"] == regime])
            for k in pairs[0]["ratios"]
        }
        for regime in dict.fromkeys(p["regime"] for p in pairs)
    }


def _expected_keys() -> set[str]:
    models = [("static-retrieval-mrl-en-v1", dim) for dim in (1024, 512, 256)] + [
        ("potion-base-8M", 256),
        ("potion-base-32M", 512),
    ]
    return {
        f"{model}-{dim}-{corpus}-{mode}-{dtype}"
        for model, dim in models
        for corpus in ("wikitext", "massive")
        for mode in ("raw", "norm")
        for dtype in ("float32", "float16", "int8")
    }


def _accounting(data: dict) -> None:
    values, settled = {}, set()
    for event in data["accounting"]["ledger"]:
        key, value = event["id"], event["seconds"]
        if not _finite(value) or value <= 0:
            raise ValueError("invalid budget value")
        if event["kind"] == "reserve" and key not in values:
            values[key] = value
        elif event["kind"] == "settle" and key in values and key not in settled:
            values[key] = value
            settled.add(key)
        else:
            raise ValueError("invalid or duplicate budget event")
    total = sum(values.values())
    _totals(data, total)


def _totals(data: dict, total: float) -> None:
    recorded = data["accounting"]
    if not math.isclose(total, recorded["global_spent_s"], abs_tol=1e-9):
        raise ValueError("global ledger total differs")
    if not math.isclose(total, sum(a["seconds"] for a in data["attempts"]), abs_tol=1e-9):
        raise ValueError("attempt charges differ from ledger")
    pilot = next(
        a["seconds"] for a in data["attempts"] if a["root"] == data["protocol"]["pilot_root"]
    )
    matrix = total - pilot
    totals = {
        "matrix_spent_s": matrix,
        "global_remaining_s": 1800 - total,
        "matrix_remaining_s": 900 - matrix,
    }
    if any(not math.isclose(recorded[k], v, abs_tol=1e-9) for k, v in totals.items()):
        raise ValueError("stage or remaining budget differs from ledger")
    if total > 1800 or matrix > 900:
        raise ValueError("measurement cap exceeded")


def _exploratory(data: dict, limits: dict) -> list[dict]:
    cells = {c["key"]: c for c in data["cells"]}
    results = []
    for row in data["exploratory"]:
        pairs = [
            _pair(p, _sides(cells[row["key"]]), row["passes"], limits[row["root"]])
            for p in row["pairs"]
        ]
        results.append(
            {
                "key": row["key"],
                "root": row["root"],
                "reason": row["reason"],
                "matrix_accepted": False,
                "restart": row["restart"],
                "pairs": pairs,
                "by_regime": _groups(pairs),
            }
        )
    return results


def summarize(data: dict) -> dict:
    keys = [c["key"] for c in data["cells"]]
    if data["schema"] != 1 or len(keys) != 60 or set(keys) != _expected_keys():
        raise ValueError("evidence must contain all 60 unique requested cells")
    _accounting(data)
    limits = {a["root"]: a["idle_receipt"]["max_load"] for a in data["attempts"]}
    cells = [
        summarize_cell(c, limits[c["root"]], data["corpora"][c["corpus"]]["texts"])
        for c in data["cells"]
    ]
    return {
        "status": data["status"],
        "cells": cells,
        "pair_regimes": dict(Counter(p["regime"] for c in cells for p in c["pairs"])),
        "cell_load": dict(Counter(c["binary_load"] for c in cells)),
        "exploratory": _exploratory(data, limits),
        "accepted_passes": sum(
            len(v) for c in data["cells"] for p in c["pairs"] for v in p["passes"].values()
        ),
    }


def render(result: dict) -> str:
    lines = [
        "60 complete cells, three valid pairs each. Required profiles: not run. MLX policy: undecided.",
        "",
        "Ratios are MLX time / CPU time, computed per pair before aggregation.",
        "",
        "| Cell | Load group | MLX/CPU1 | MLX/CPU10 | MLX/CPU14 |",
        "|---|---|---|---|---|",
    ]
    for cell in result["cells"]:
        values = []
        for name in ("mlx/cpu1", "mlx/cpu10", "mlx/cpu14"):
            r = cell["ratios"][name]
            values.append(f"{r['median']:.6f} ({r['min']:.6f} to {r['max']:.6f})")
        lines.append(f"| {cell['key']} | {cell['binary_load']} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=DEFAULT)
    parser.add_argument(
        "--json", action="store_true", help="all unrounded ratios, A/A, loads and throughput"
    )
    args = parser.parse_args()
    result = summarize(json.loads(args.path.read_text()))
    print(json.dumps(result, indent=2) if args.json else render(result))


if __name__ == "__main__":
    main()
