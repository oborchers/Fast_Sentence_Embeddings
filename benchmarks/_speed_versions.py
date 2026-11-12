"""Version-study controls shared by setup, preparation and paired summaries."""

from __future__ import annotations

import _speed_cells as inputs
from _harness.budget import select_billing


def billing_contract(contract, study):
    """Use the same selected profile for setup and every step of an allocation."""
    return select_billing(contract, study=study)


def pair_rounds(cell):
    """Paired observations have a declared integer count, with at least three rounds."""
    rounds = cell.get("rounds", 3)
    if type(rounds) is not int or rounds < 3:
        raise ValueError("rounds must be an integer of at least three")
    if "input" in cell and cell["input"] != "st":
        raise ValueError("the only explicit input profile is 'st'")
    return rounds


def preflight_plan(cell, production, model_bytes, limit):
    """Only a rejected seven-million-row ST-16 plan may fall back to five million."""
    plan = inputs.memory_plan(production, cell["planned_input_bytes"], model_bytes, limit)
    if (
        plan["admitted"]
        or cell["id"] != "version-st-float32-16"
        or production["texts"] != 7_000_000
    ):
        return plan
    rejected = plan
    production.update(texts=5_000_000, planned_input_bytes=681_709_559)
    # Apply the common input to both arms, including later production reconstruction.
    cell["production_texts" if cell.get("functional_only") else "texts"] = 5_000_000
    cell["planned_input_bytes"] = production["planned_input_bytes"]
    plan = inputs.memory_plan(production, cell["planned_input_bytes"], model_bytes, limit)
    return {**plan, "fallback": {"texts": 5_000_000, "rejected_plan": rejected}}


def require_budget(ledger, estimate):
    if not ledger.admit(estimate):
        raise RuntimeError("preparation admission budget or stop signal exhausted")
