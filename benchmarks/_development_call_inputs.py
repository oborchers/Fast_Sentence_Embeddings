"""Pinned external short-call inputs, with the original driver's newline handling."""

from __future__ import annotations

import os
from pathlib import Path

from _development_process import checked
from _measurement_setup import outside_checkout

PINS = {
    "calls-chunks-lt512": {
        "file": "chunks-lt512.txt",
        "sha256": "6729304a1a116461885ee4c75872b565fc0b78d9c184daa9b95a597883dfd2e8",
        "texts": 4242,
    },
    "calls-queries": {
        "file": "queries.txt",
        "sha256": "5e550317552cec5a30c156003961fa4482d67c0758ebec23f44caf102974702c",
        "texts": 65536,
    },
}

# Same production bytes; separate name permits an independent functional reduction.
PINS["calls-chunks-lt512-control"] = PINS["calls-chunks-lt512"]


def corpus(name):
    root = os.environ.get("FSE_CALL_INPUTS")
    if not root:
        raise ValueError("FSE_CALL_INPUTS must name the external pinned input directory")
    path = Path(root).resolve() / PINS[name]["file"]
    outside_checkout(path)
    pin = PINS[name]
    checked(path, pin["sha256"])
    # Exactly pyrunner2.py: split on LF and drop the last element. Require the
    # final LF, rather than silently discarding an incomplete last row.
    raw = path.read_text(encoding="utf-8")
    if not raw.endswith("\n"):
        raise ValueError("short-call input must end with a newline")
    texts = raw.split("\n")[:-1]
    if len(texts) != pin["texts"]:
        raise ValueError("short-call input row count differs from its pin")
    return {
        "texts": texts,
        "timing_texts": texts,
        "score_slices": [],
        "timing_policy": "pyrunner2.py: read UTF-8, split LF, drop final empty element",
        "source": pin,
    }
