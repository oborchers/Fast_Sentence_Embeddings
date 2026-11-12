"""Seal the allocation-wide AVX2 decision, without running or changing any experiment."""

from __future__ import annotations

from _harness.decide import evaluate, validate
from _harness.prerequisites import recorded
from _harness.records import atomic, document, seal
from _measurement_setup import outside_checkout
from _prefilter_decision_evidence import float32, profiles, search

STEPS = ("speed-f32", "prefilter", "profile-cpu", "profile-memory")


def evaluate_all(root):
    """Only collected content is consulted. No clock, local code or relocated inputs."""
    result = {
        "format": "prefilter-decision@1",
        "claim_class": "optimization",
        "publishable": False,
        "decision": "inconclusive",
        "final": False,
        "complete": False,
        "inputs": {},
        "reason": None,
    }
    try:
        context, receipts = recorded(root, STEPS)
        result["inputs"] = receipts
        origin = root / "prefilter-results"
        contract = document(origin / "environment.json")["contract"]
        rule = contract["decision"]
        validate(rule, contract["cells"])
        if rule["rule"] != "cells@2" or rule["practical_saving"] != 0.10:
            raise ValueError("AVX2 requires cells@2 with ten percent saving")
        if rule["main"] != ["o1-d256-q8", "o2-d1024-q8"] or rule["edge"]:
            raise ValueError("AVX2 requires exactly both main search cells")
        if contract.get("functional_only") or contract["local_phase"] != "all":
            raise ValueError("full production development study required")
        rows, builds = search(origin, contract)
        float32(root / "speed-f32-results", builds)
        profiles(root, context, origin)
        overall = evaluate(rule, rows, [cell["id"] for cell in contract["cells"]])
        if overall["complete"]:
            result.update(
                decision=overall["decision"], final=True, complete=True, roles=overall["roles"]
            )
        else:
            result["reason"] = "search decision evidence is incomplete"
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        # Stable reason codes make malformed evidence reproducible after collection,
        # without embedding host-dependent exception paths or timestamps.
        result["reason"] = "required evidence is missing, invalid or mismatched"
    return result


def run(root, results):
    for path in (root, results):
        if not path.is_absolute() or path.is_symlink():
            raise ValueError("absolute allocation and result paths without symlinks required")
    outside_checkout(results)
    results = results.resolve()
    protected = [root / (step + "-results") for step in (*STEPS, "allocation")]
    if any(
        results.is_relative_to(p.resolve()) or p.resolve().is_relative_to(results)
        for p in protected
    ):
        raise ValueError("decision output overlaps its evidence")
    results.mkdir(parents=True, exist_ok=False)
    decision = evaluate_all(root)
    atomic(results / "decision.json", decision)
    code = int(decision["decision"] != "enable")
    # Completion belongs to the inputs, not the replaying host's clock. This also
    # makes the manifest and DONE byte-identical for an offline replay.
    atomic(results / "cells.jsonl", decision)
    atomic(results / "progress.json", {"stage": "terminal", "exit_status": code})
    seal(results, code, 1, extra={"completed_at": None})
    return code
