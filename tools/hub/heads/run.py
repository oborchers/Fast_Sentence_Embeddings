"""Prepare, select and evaluate a local Civil Comments classifier. Never uploads."""

from __future__ import annotations

import argparse
import os

from civil_comments import prepare
from evaluation import finalize, probe, verify
from inputs import CONFIG, Budget, external, load, write
from threadpoolctl import threadpool_limits
from training import select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "probe", "select", "verify", "finalize", "preflight")
    )
    parser.add_argument("--work", type=external, required=True)
    parser.add_argument("--config", type=type(CONFIG), default=CONFIG)
    args = parser.parse_args()
    config = load(args.config)
    if args.command == "preflight":
        print("civil-comments-head@1: parser, configuration and external path valid")
        return
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.command == "prepare":
        args.work.mkdir(parents=True, exist_ok=False)
        write(args.work / "config.json", config)
    elif load(args.work / "config.json") != config:
        raise ValueError("run configuration changed")
    budget = Budget(args.work, config)
    budget.check()
    with budget.monitor(), threadpool_limits(limits=1):
        if args.command == "prepare":
            prepare(args.work, config, budget)
        elif args.command == "probe":
            probe(args.work, config, budget)
        elif args.command == "select":
            select(args.work, config, budget)
        elif args.command == "verify":
            verify(args.work, config, budget)
        else:
            finalize(args.work, config, budget)
    print(args.command + ": complete", flush=True)


if __name__ == "__main__":
    main()
