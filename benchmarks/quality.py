"""Evaluate finished encoders, fit SIF offline, rehearse or summarize quality studies."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from _development_config import add_arguments, dispatch
from _harness.ledger import Ledger, watching
from _harness.records import append, atomic
from _measurement import finalize_rehearsal, rehearsal_limits, write_environment
from _measurement_config import CONFIG, contract, validate_cell, validate_prerequisites
from _measurement_setup import outside_checkout, setup, utility


def specification(args):
    spec = contract(args.input)
    if args.gpu_phase:
        from _quality.gpu import select

        spec = select(spec, args.gpu_phase)
    if args.rehearse:
        rehearsal_limits()
        spec["functional_only"] = True
        spec["cells"] = [
            {
                "id": "small-fit",
                "kind": "fit",
                "model": "fse/glove-wiki-gigaword-300",
                "corpus": "wikitext",
                "workers": 1,
                "estimated_seconds": 30,
            },
            {
                "id": "small-sts",
                "kind": "sts",
                "fit": "small-fit",
                "workers": 1,
                "estimated_seconds": 30,
            },
            {
                "id": "small-mteb",
                "kind": "mteb",
                "model": "minishlab/potion-base-8M",
                "backend": "fse",
                "workers": 1,
                "tasks": ["BIOSSES"],
                "estimated_seconds": 60,
            },
        ]
        spec["billing"].update(planning_minutes=180, planning_balance_usd=10)
    if args.rehearse:
        spec["cells"].append(
            {
                "id": "small-recall",
                "kind": "recall",
                "model": "fse/glove-wiki-gigaword-300",
                "backend": "fse",
                "fit": "small-fit",
                "dtype": "int8",
                "workers": 1,
                "estimated_seconds": 30,
            }
        )
    if args.rehearse:
        spec["cells"].append(
            {
                "id": "small-published-fit",
                "kind": "fit",
                "model": "fse/glove-wiki-gigaword-300",
                "corpus": "published-fineweb",
                "output_tokenizer": "fse_words@1",
                "workers": 1,
                "estimated_seconds": 30,
            }
        )
    if args.rehearse:
        spec["cells"].append(
            {
                "id": "small-published-recall",
                "kind": "recall",
                "model": "fse/glove-wiki-gigaword-300",
                "backend": "fse",
                "fit": "small-published-fit",
                "dtype": "float16",
                "workers": 1,
                "estimated_seconds": 30,
            }
        )
    if args.cells:
        spec["cells"] = [c for c in spec["cells"] if c["id"] in args.cells]
        if len(args.cells) != len(spec["cells"]) or set(args.cells) != {
            c["id"] for c in spec["cells"]
        }:
            raise ValueError("unknown or duplicate requested cells")
    validate_prerequisites(spec["cells"])
    from _quality.suites import expand

    spec["cells"] = expand(spec["cells"])
    if not spec["cells"]:
        raise ValueError("empty quality study")
    for cell in spec["cells"]:
        validate_cell(cell)
        cell["functional_only"] = cell.get("functional_only", False) or spec.get(
            "functional_only", False
        )
    return spec


def run(args):
    from _quality.fit import fit, mean, recall, sts
    from _quality.mteb import evaluate
    from _quality.probe import probe
    from _quality.suites import MIND_REASON

    spec = specification(args)
    args.results.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(args.results, spec, args.allocated_at)
    os.environ["MEASURE_WORK"] = str(args.work)
    write_environment(args, spec)
    atomic(args.results / "exclusions.json", {"MindSmallReranking": MIND_REASON})
    context = {
        "results": args.results,
        "work": args.work,
        "cache": args.cache,
        "fixture_repo": args.fixture_repo,
        "fits": {},
        "finished": ledger.finished,
    }
    actions = {
        "fit": fit,
        "mean": mean,
        "sts": sts,
        "mteb": evaluate,
        "recall": recall,
        "probe": probe,
    }
    with watching(ledger, restore=False):
        try:
            args.work.mkdir(parents=True, exist_ok=False)
            from nltk import download

            if not download("punkt_tab", quiet=True):
                raise RuntimeError("pinned NLTK tokenizer resources unavailable")
            for cell in spec["cells"]:
                ledger.stage, ledger.current = cell["kind"], cell["id"]
                started = time.monotonic()
                if not ledger.admit(cell["estimated_seconds"]):
                    ledger.result(
                        cell, started, status="not_started", reason="soft budget or stop signal"
                    )
                    continue
                try:
                    result = actions[cell["kind"]](cell, context)
                    ledger.result(cell, started, **result)
                except Exception as error:
                    ledger.result(cell, started, status="failed", reason=repr(error))
        except Exception as error:
            append(args.results / "fatal.jsonl", {"error": repr(error)})
        finally:
            ledger.fill_missing("preparation failed")
    finalize_rehearsal(ledger, "quality")
    return ledger.done()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--setup", type=Path)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--rehearse", action="store_true")
    mode.add_argument("--summarize", type=Path)
    mode.add_argument("--collect", type=Path)
    mode.add_argument("--bootstrap", action="store_true")
    mode.add_argument("--gpu-task", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, default=CONFIG / "quality.json")
    parser.add_argument("--cells", nargs="+")
    parser.add_argument("--results", type=Path, default=Path("/quality-results"))
    parser.add_argument("--work", type=Path, default=Path("/quality-work"))
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/fse-quality")
    parser.add_argument(
        "--fixture-repo", type=Path, help="clone containing the pinned historical STS Git object"
    )
    parser.add_argument("--allocated-at", type=float)
    parser.add_argument("--gpu-phase", choices=("probe", "measure"))
    parser.add_argument("--allocation-root", type=Path)
    parser.add_argument(
        "--local-run",
        action="store_true",
        help="with --run: MTEB cells on a Unix machine, class correctness, publishable false, no speed value",
    )
    parser.add_argument(
        "--local-budget-seconds", type=float, default=3600, help="wall-time budget of --local-run"
    )
    parser.add_argument(
        "--local-order",
        choices=("cells", "task"),
        default="cells",
        help="--local-run order: as configured, or all cells of one task in a row (loads once)",
    )
    parser.add_argument(
        "--local-deadline-unix",
        type=float,
        help="shared end of a staged study; cuts the budget of this --local-run to what is left",
    )
    parser.add_argument(
        "--local-min-free-gb",
        type=float,
        default=40,
        help="--local-run stops at the next task when any watched file system has less free",
    )
    parser.add_argument(
        "--local-rss-limit-gb",
        type=float,
        default=12,
        help="--local-run aborts at once above this resident set and refuses larger estimates",
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--allocation", type=Path, default=CONFIG / "allocation.json")
    add_arguments(parser, mode, quality=True)
    return parser


def _resolve_paths(args):
    """The result, work and cache paths and the clone of the historical STS fixture."""
    for name in ("results", "work", "cache"):
        setattr(args, name, getattr(args, name).resolve())
    setup_source = Path(sys.executable).parents[2] / "fse-source"
    args.fixture_repo = (
        args.fixture_repo
        or (setup_source if setup_source.is_dir() else Path(__file__).resolve().parents[1])
    ).resolve()


def _local(parser, args):
    """``--run --local-run``: MTEB cells on a Unix machine, class correctness, never publishable."""
    if not args.run:
        parser.error("--local-run goes with --run")
    for path in (args.results, args.work, args.cache):
        outside_checkout(path)
    from _quality.local import run as local_run

    return local_run(args)


def _validate_modes(parser, args):
    if args.gpu_phase and (args.local_run or args.rehearse or not (args.run or args.describe)):
        parser.error("--gpu-phase requires --run or --describe, without local-run or rehearse")
    if args.gpu_phase and args.cells:
        parser.error("GPU quality phases use their complete declared cell selection")


def main():
    parser = _parser()
    args = parser.parse_args()
    _validate_modes(parser, args)
    if args.gpu_task:
        from _quality.gpu import child

        return child(args.gpu_task)
    local_status = dispatch(args, quality=True)
    if local_status is not None:
        return local_status
    _resolve_paths(args)
    if args.summarize:
        from _quality.summary import summarize

        print(json.dumps(summarize(args.summarize), indent=2, allow_nan=False))
        return 0
    if utility(args, "quality"):
        return 0
    if args.local_run:
        return _local(parser, args)
    return _allocated(parser, args)


def _allocated(parser, args):
    spec = contract(args.input)
    if spec.get("billing_profile") == "gpu_quality_billing":
        from _harness.budget import select_billing

        spec = select_billing(spec)
        if args.run and args.gpu_phase is None:
            parser.error("GPU quality execution requires --gpu-phase")
    if sys.platform != "linux" or sys.version.split()[0] != spec["machine"]["python_patch"]:
        raise RuntimeError("pinned Linux Python required")
    if args.allocated_at is None:
        parser.error("allocation Unix timestamp required")
    if args.setup:
        if os.getenv("MEASURE_REHEARSAL") == "1":
            rehearsal_limits()
        setup(args.setup, spec, args.allocated_at, "quality", args.allocation)
        return 0
    for path in (args.results, args.work, args.cache):
        outside_checkout(path)
    if args.gpu_phase:
        from _quality.gpu import run as gpu_run

        return gpu_run(args, specification(args))
    code = run(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
