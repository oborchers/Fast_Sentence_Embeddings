"""Bounded slim Pod 4 entry point; every failed stage has a nonzero exit status."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pod_artifacts import environment  # isort: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["probe", "readme", "classifications"])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("Pod 4 never runs on the local Mac")
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 32:
        raise RuntimeError("pod exposes fewer than 32 assigned CPUs")
    os.sched_setaffinity(0, allowed[:32])
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    args.hardware = {**environment(), "affinity_original": allowed}
    args.deadline = time.monotonic() + args.deadline_epoch - time.time()
    if args.stage in ("probe", "readme"):
        from pod4_readme import QuietPodRequiredError, probe, run

        try:
            (probe if args.stage == "probe" else run)(args)
        except QuietPodRequiredError as error:
            print(str(error), file=sys.stderr, flush=True)
            raise SystemExit(75) from error
    else:
        from pod4_classify import run

        args.output = args.output / "folded-classification"
        args.output.mkdir(exist_ok=True)
        run(args)
    (args.output / f"{args.stage}-complete.json").write_text(
        json.dumps({"stage": args.stage, "status": "completed"}) + "\n"
    )


if __name__ == "__main__":
    main()
