"""Local development lifecycle through the existing speed entry."""

from __future__ import annotations

import time

from _development_config import load
from _development_data import prepare
from _development_records import record_result, session


def run(args):
    spec = load(args.development_input, args.cells)
    with session(args, spec) as ledger:
        prepared = prepare(args, spec["cells"], ledger)
        for cell in spec["cells"]:
            record_result(
                ledger,
                cell,
                time.monotonic(),
                status="valid",
                phase=args.local_phase,
                decision="not_run",
                preparation=prepared,
            )
    return ledger.done()
