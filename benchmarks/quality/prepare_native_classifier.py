"""Train Model2Vec's own classifier on the exact frozen-POTION dataset splits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_classifier import train  # isort: skip
from prepare_fineweb import REPO, REVISION, load_pages, split  # isort: skip
from prepare_setfit import SOURCES, dataset  # isort: skip
from quality_coordination import task_reservation  # isort: skip


def inputs(suite, cache, selected, *, admission=None):
    if suite == "fineweb":
        yield (
            "fineweb",
            split(*load_pages(cache, download=True, admission=admission)),
            {
                "dataset": REPO,
                "dataset_revision": REVISION,
                "split": "default_rng(0) permutation, last 46743 held out",
                "label": "score >= 3",
                "preprocessing": "whitespace collapsed",
            },
        )
        return
    specifications = json.loads(SOURCES.read_text())["datasets"]
    names = {row["dataset"].removeprefix("SetFit/") for row in specifications}
    if selected and not set(selected) <= names:
        raise ValueError("unknown SetFit dataset requested")
    for spec in specifications:
        name = spec["dataset"].removeprefix("SetFit/")
        if selected and name not in selected:
            continue
        yield (
            name,
            dataset(spec, cache, download=True, admission=admission),
            {
                "dataset": spec["dataset"],
                "dataset_revision": spec["revision"],
                "split": "1000 stratified, seed 0",
                "protocol": "historical 15 displayed rows reconstruction",
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["fineweb", "setfit"], required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--coordination", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["minishlab/potion-base-8M", "minishlab/potion-base-32M"],
        choices=["minishlab/potion-base-8M", "minishlab/potion-base-32M"],
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pod-training-times", action="store_true")
    args = parser.parse_args()
    if args.pod_training_times and sys.platform != "linux":
        raise ValueError("classification timing is restricted to the Linux pod")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("run with nice -n 10")
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with task_reservation(args.coordination), threadpool_limits(limits=args.threads):
        for name, data, provenance in inputs(args.suite, args.cache, args.datasets):
            folder = args.output if args.suite == "fineweb" else args.output / name
            folder.mkdir(parents=True, exist_ok=True)
            for repo in args.models:
                output = folder / (repo.rsplit("/", maxsplit=1)[-1] + "-model2vec-pipeline")
                row = train(data, repo, output, provenance, timed_training=args.pod_training_times)
                with args.results.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(
                    json.dumps({"dataset": name, "model": repo, "macro_f1": row["macro_f1"]}),
                    flush=True,
                )


if __name__ == "__main__":
    main()
