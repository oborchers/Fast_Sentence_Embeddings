"""Which variant and path an arm really ran, observed outside every timer.

The int8 prefilter is entered only inside its domain and gives up for a reason, and a backend
enables a variant or none. A time says nothing about that, so each measured search arm makes
one more, untimed call after its measurement with the debug records of ``fse.store`` captured,
and reports the variant, whether the prefilter was entered and why it fell back. A build
without the prefilter or without the log events reports an empty list and no variant.
"""

from __future__ import annotations

import json
import logging

from _development_prefilter_search import POOL
from _quality.prefilter_inputs import search_enabled

EVENT = "store.prefilter"
FIELDS = ("queries", "k", "rows", "pairs", "fallback", "workers")


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.events = []
        self.call = 0

    def emit(self, record):
        if getattr(record, "fse_event", None) == EVENT:
            fields = {name: getattr(record, "fse_" + name, None) for name in FIELDS}
            self.events.append({"call": self.call, **fields})


def _variant(backend):
    ask = getattr(backend, "prefilter_variant", None)
    return "unavailable" if ask is None else ask()


def observe(operation, job):
    """The evidence of one search arm, or ``None`` for a stage without the prefilter profile."""
    if not search_enabled(job) or getattr(operation, "store", None) is None:
        return None
    calls = max(1, min(job["iterations"], POOL // job["queries"]))
    operation.restart()
    try:
        return capture(lambda _: operation(), calls, operation.store._backend, job["search_mode"])
    finally:
        operation.restart()


def capture(run, calls, backend, mode):
    """Capture a separate, untimed sequence of search calls."""
    logger, capture = logging.getLogger("fse.store"), _Capture()
    level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(capture)
    try:
        for capture.call in range(calls):
            run(capture.call)
    finally:
        logger.removeHandler(capture)
        logger.setLevel(level)
    return {
        "untimed": True,
        "search_mode": mode,
        "backend": getattr(backend, "name", None),
        "isa": getattr(backend, "isa", None),
        "prefilter_variant": _variant(backend),
        "calls": calls,
        "entered": bool(capture.events),
        "entered_calls": len(capture.events),
        "fallbacks": sorted({event["fallback"] for event in capture.events}),
        "events": capture.events,
    }


def per_arm(rows):
    """``{arm: evidence}`` of the rows of a block; a path that changed between rounds is an
    error, never an average."""
    found = {}
    for row in rows:
        evidence = row.get("path_evidence")
        if evidence is None:
            continue
        key = json.dumps(evidence, sort_keys=True)
        if found.setdefault(row["arm"], (key, evidence))[0] != key:
            raise ValueError(f"the path of the {row['arm']} arm changed between rounds")
    return {arm: evidence for arm, (_, evidence) in found.items()}
