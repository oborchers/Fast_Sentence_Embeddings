"""Run, rehearse, describe, set up or collect CPU speed studies from data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from _measurement import document, rehearsal_limits
from _measurement_config import CONFIG, contract
from _measurement_setup import setup, utility
from _speed_pipeline import measure_child
from _speed_run import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CONFIG / "speed.json")
    parser.add_argument(
        "--study", choices=("all", "libraries", "headline", "dtypes"), default="libraries"
    )
    parser.add_argument("--workers", type=int, choices=(16, 32))
    parser.add_argument("--reuse-work", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--child-job", type=Path, help=argparse.SUPPRESS)
    mode.add_argument("--setup", type=Path)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--rehearse", action="store_true")
    mode.add_argument("--collect", type=Path)
    mode.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--source-revision")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--results", type=Path, default=Path("/speed-results"))
    parser.add_argument("--work", type=Path, default=Path("/speed-work"))
    parser.add_argument("--reference-python", default="/speed-env/reference/bin/python")
    parser.add_argument("--base-python", help="installed base fse interpreter for fse-base cells")
    parser.add_argument("--allocated-at", type=float)
    args = parser.parse_args()
    if utility(args, "speed"):
        return 0
    if args.child_job:
        measure_child(document(args.child_job))
        return 0
    specification = contract(args.input)
    if (
        sys.platform != "linux"
        or sys.version.split()[0] != specification["machine"]["python_patch"]
    ):
        raise RuntimeError("pinned Linux Python required")
    if args.allocated_at is None:
        parser.error("allocation Unix timestamp required")
    if args.setup:
        if os.getenv("MEASURE_REHEARSAL") == "1":
            rehearsal_limits()
        setup(args.setup, specification, args.allocated_at)
        return 0
    code = run(args)
    if args.rehearse and code == 0:
        from _measurement_rehearse import control_proof

        control_proof(args.results.parent / (args.results.name + "-control"), "speed")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
