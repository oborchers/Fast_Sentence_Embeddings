"""The decision rule of the DR-255 probe, fixed before the run (plan, section 7)."""

from __future__ import annotations

import json
from pathlib import Path

from _quality.binary_inputs import MODELS, OVERSAMPLES

SUPPORT, WEAK = 0.95, 0.90


def _recalls(path):
    """model -> mean recall@10 at the largest oversampling, None for a missing model."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    return {
        m: next(
            (
                r["recall_at_10"]
                for r in rows
                if r["model"] == m and r["oversample"] == max(OVERSAMPLES)
            ),
            None,
        )
        for m in MODELS
    }


def decide(path, cells, finished, failed):
    """failed, inconclusive, or the placement by the recall at 40-fold oversampling."""
    if failed:
        return {"result": "failed", "decision": None}
    status = {r["cell_id"]: r["status"] for r in finished}
    if {status.get(c["id"]) for c in cells} != {"valid"} or {c["model"] for c in cells} != set(
        MODELS
    ):
        return {"result": "inconclusive", "decision": None}
    recalls = _recalls(path)
    if any(v is None for v in recalls.values()):
        return {"result": "inconclusive", "decision": None}
    return {"result": "complete", "recall_at_10_40x": recalls, **_place(recalls)}


def _place(recalls):
    if recalls[MODELS[0]] < WEAK:
        return {"decision": "below 0.90 for POTION: study asymmetric or centered variants first"}
    if all(v >= SUPPORT for v in recalls.values()):
        return {"decision": "supports a symmetric 1-bit draft (at least 0.95 for both models)"}
    return {"decision": "between the limits: no decision, report only"}
