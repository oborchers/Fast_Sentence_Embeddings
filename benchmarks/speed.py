"""Run, rehearse, describe, set up or collect CPU speed studies from data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from _development_config import add_arguments, dispatch
from _harness.records import document
from _measurement import rehearsal_limits
from _measurement_config import CONFIG, contract
from _measurement_setup import outside_checkout, setup, utility
from _speed_pipeline import measure_child
from _speed_run import run
from _speed_versions import billing_contract


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CONFIG / "speed.json")
    parser.add_argument(
        "--study",
        choices=(
            "all",
            "readme",
            "libraries",
            "headline",
            "dtypes",
            "bekko",
            "paragraphs",
            "heads",
            "versions",
            "search",
            "search-prefilter",
        ),
        default="libraries",
    )
    parser.add_argument("--workers", type=int, choices=(16, 32))
    parser.add_argument("--reuse-work", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--child-job", type=Path, help=argparse.SUPPRESS)
    mode.add_argument("--profile", type=Path, help="profile an addressed child from profile@1")
    mode.add_argument("--profile-check", type=Path, help="check profiler access from profile@1")
    mode.add_argument("--allocation-profile", choices=("py-spy", "memray"))
    parser.add_argument("--allocation-root", type=Path)
    parser.add_argument("--profile-cell", default="version-encode-float32-16")
    parser.add_argument("--profile-seconds", type=int, default=240)
    mode.add_argument("--profile-probe", type=Path, help=argparse.SUPPRESS)
    mode.add_argument("--setup", type=Path)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--rehearse", action="store_true")
    mode.add_argument("--collect", type=Path)
    mode.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--source-revision")
    parser.add_argument("--allocation", type=Path, default=CONFIG / "allocation.json")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--results", type=Path, default=Path("/speed-results"))
    parser.add_argument("--work", type=Path, default=Path("/speed-work"))
    parser.add_argument("--reference-python", default="/speed-env/reference/bin/python")
    parser.add_argument("--base-python", help="installed base fse interpreter for fse-base cells")
    parser.add_argument("--allocated-at", type=float)
    add_arguments(parser, mode)
    args = parser.parse_args()
    if args.ignore_host_load and (not args.diagnose or args.allocation_root is None):
        parser.error("--ignore-host-load requires --diagnose and --allocation-root")
    return args


def profile_utility(args):
    if args.allocation_profile:
        from _allocation_profile import run as allocation_profile

        return allocation_profile(args)
    if args.profile_probe:
        from _development_profile_probe import child

        child(args.profile_probe)
        return 0
    if args.profile or args.profile_check:
        from _development_profile import run as profile

        return profile(args.profile or args.profile_check, check_only=bool(args.profile_check))
    return None


def child_job(file):
    job = document(file)
    if job.get("kind") == "profile-input":
        from _speed_profile import input_child

        input_child(job)
    else:
        measure_child(job)


def main():
    args = arguments()
    profile_status = profile_utility(args)
    if profile_status is not None:
        return profile_status
    local_status = dispatch(args)
    if local_status is not None:
        return local_status
    if utility(args, "speed"):
        return 0
    if args.child_job:
        child_job(args.child_job)
        return 0
    specification = billing_contract(contract(args.input), args.study)
    if (
        sys.platform != "linux"
        or sys.version.split()[0] != specification["machine"]["python_patch"]
    ):
        raise RuntimeError("pinned Linux Python required")
    if args.allocated_at is None:
        raise ValueError("allocation Unix timestamp required")
    if args.setup:
        if os.getenv("MEASURE_REHEARSAL") == "1":
            rehearsal_limits()
        setup(args.setup, specification, args.allocated_at, "speed", args.allocation)
        return 0
    for path in (args.results, args.work):
        outside_checkout(path)
    code = run(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
