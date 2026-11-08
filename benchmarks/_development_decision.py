"""Cell decisions and the common terminal decision writer."""

from _development_heads import nonregression
from _development_heads_multi import practical
from _development_statistics import decide, decision_enabled
from _development_system import LABEL
from _harness.decide import evaluate
from _harness.records import append, atomic


def judge(cell, aa, main, confirmation, functional=False):
    result = practical(cell, nonregression(cell, decide(cell, aa, main, confirmation)))
    if functional:
        result["decision"] = "functional_only"
    gate = result.get("practical")
    if gate and not gate["passed"] and result["decision"] == "keep":
        result["decision"] = "keep_below_practical_saving"  # decision.json must not say keep
    accepted = valid(result, functional)
    return dict(status="valid" if accepted else "rejected", **result)


def valid(result, rehearsal):
    """A decision is valid when it is keep (or descriptive, or functional), and where the
    cell declared them, its noise-limited nonregression and its practical saving hold."""
    valid = result["decision"] in ("keep", "descriptive_only", "functional_only")
    for gate in ("nonregression", "practical"):
        if gate in result and not rehearsal:
            valid = valid and result[gate]["passed"]
    return valid


def finish(ledger):
    """Write after fill_missing and before sealing, including failed preparation."""
    result = {
        **LABEL,
        "functional_only": ledger.contract.get("functional_only", False),
        "cells": {r["cell_id"]: r.get("decision", "not_run") for r in ledger.finished},
    }
    code = 0
    if "decision" in ledger.contract:
        try:
            overall = overall_of(ledger)
        except (ValueError, KeyError, TypeError) as error:
            append(ledger.root / "fatal.jsonl", {**LABEL, "error": repr(error)})
            overall = {"decision": "error", "final": False, "complete": False}
        result["overall"] = overall
        code = int(overall["decision"] not in ("enable", "functional_only"))
    atomic(ledger.root / "decision.json", result)
    return ledger.done(code)


def overall_of(ledger):
    return evaluate(
        ledger.contract["decision"],
        ledger.finished,
        [c["id"] for c in ledger.contract["cells"]],
        ledger.contract.get("functional_only", False),
        ledger.contract["local_phase"],
        getattr(ledger, "stopped_after", None),
    )


def stop_after(ledger, cell):
    rule = ledger.contract.get("decision")
    if rule and rule["stop_early"] and overall_of(ledger)["decision"] == "reject":
        ledger.stopping = True
        ledger.stopped_after = cell["id"]


def screen(cell, aa, main, functional=False):
    """The existing preliminary screen, with no final keep decision."""
    if main["output"] != aa["output"]:
        raise ValueError("screen output differs from its A/A control")
    if functional:
        return dict(status="valid", decision="functional_only", comparison=main)
    if not decision_enabled(cell):
        return dict(status="valid", decision="descriptive_only", comparison=main)
    threshold = max(
        cell["noise_multiplier"] * aa["factor"]["iqr"],
        cell["minimum_saving"],
        cell["practical_saving"] or 0,
    )
    passed = main["time_saving"] >= threshold
    return dict(
        status="inconclusive" if passed else "rejected",
        decision="screen_requires_full_evidence" if passed else "screen_reject",
        comparison=main,
        threshold=threshold,
    )


def replay(results):
    """Re-evaluate recorded complete blocks without writing to the source folder.

    This internal adapter requires the resolved contract in environment.json. It
    does not guess defaults for historical contracts. A historical format adapter
    must supply its documented contract before calling judge or screen directly.
    """
    from _harness.records import document, read_rows

    contract = document(results / "environment.json")["contract"]
    rows = read_rows(results, "cells.jsonl")
    decisions = {}
    for row in rows:
        cell_id = row["cell_id"]
        if cell_id in decisions:
            raise ValueError("duplicate replay cell")
        decisions[cell_id] = _replay_cell(results, contract, row)
    result = {"cells": decisions}
    if "decision" in contract:
        result["overall"] = evaluate(
            contract["decision"],
            rows,
            [c["id"] for c in contract["cells"]],
            contract.get("functional_only", False),
            contract["local_phase"],
            _replay_stop(results, contract, rows),
        )
    return result


def _replay_cell(results, contract, row):
    from _harness.records import document

    if all(isinstance(row.get(k), dict) for k in ("aa", "comparison", "confirmation")):
        return judge(
            row,
            row["aa"],
            row["comparison"],
            row["confirmation"],
            row.get("functional_only", False),
        )
    if contract.get("local_phase") == "screen" and isinstance(row.get("comparison"), dict):
        control = document(results / (row["cell_id"] + "-comparison.json"))["control"]
        return screen(row, control["aa"], row["comparison"], row.get("functional_only", False))
    return {"decision": row.get("decision", "not_run"), "status": row["status"], "replayed": False}


def _replay_stop(results, contract, rows):
    """Preserve the stop address only when the sealed rows prove that first rejection."""
    from _harness.records import document

    marker = document(results / "decision.json")["overall"]["stopped_after"]
    following = [r for r in rows if r.get("stopped_after") is not None]
    if marker is None:
        if following:
            raise ValueError("replay rows contain an unrecorded stop")
        return None
    rule = contract["decision"]
    if not rule["stop_early"] or contract.get("functional_only", False):
        raise ValueError("replay stop is inconsistent with the contract")
    index = _stop_position(rows, marker)
    selected = [c["id"] for c in contract["cells"]]
    before = evaluate(rule, rows[:index], selected, phase=contract["local_phase"])
    after = evaluate(rule, rows[: index + 1], selected, phase=contract["local_phase"])
    if before["decision"] == "reject" or after["decision"] != "reject":
        raise ValueError("replay stop does not address the first final rejection")
    return marker


def _stop_position(rows, marker):
    positions = [i for i, row in enumerate(rows) if row["cell_id"] == marker]
    if len(positions) != 1:
        raise ValueError("replay stop cell is missing or duplicated")
    index = positions[0]
    later = rows[index + 1 :]
    if any(r["status"] != "not_started" or r.get("stopped_after") != marker for r in later):
        raise ValueError("replay contains work after the recorded stop")
    return index
