"""Additional Pod 3 classifier rows, preserving every existing Pod 2 comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from folded_classifier_prepare import REPO, prepare, prepare_mlp  # isort: skip
from compare_models import MODELS  # isort: skip
from pod_artifacts import emit, export_jsonl, environment  # isort: skip
from pod_pairs import append, series  # isort: skip
from prepare_native_classifier import inputs  # isort: skip
from prepare_setfit import SOURCES  # isort: skip


def timed_dataset(args, folder, cell, *, mlp=False):
    for workers in (1, 32):
        base = [
            sys.executable,
            str(Path(__file__).with_name("pod_folded_child.py")),
            "--folder",
            str(folder),
            "--workers",
            str(workers),
        ]
        commands = {label: [*base, "--kind", "folded"] for label in ("fse-a", "fse-b")}
        if mlp:
            commands["model2vec-mlp"] = [*base, "--kind", "mlp"]
        else:
            commands["encode-matmul"] = [*base, "--kind", "encode-matmul"]
            commands["model2vec-same-linear"] = [*base, "--kind", "model2vec-linear"]
        series(
            commands,
            workers,
            args.output,
            args.deadline,
            f"folded/{cell}/{workers}/{'mlp' if mlp else 'same-linear-head'}",
        )
        from pod_folded_summary import publish_summary

        publish_summary(args.output)


def workload(args):
    import torch
    from huggingface_hub import snapshot_download

    snapshot_download(
        REPO,
        revision=MODELS[REPO],
        allow_patterns=["*.json", "*.safetensors", "README.md"],
        max_workers=2,
    )
    torch.set_num_threads(16)
    names = [
        row["dataset"].removeprefix("SetFit/")
        for row in json.loads(SOURCES.read_text())["datasets"]
    ]
    # Publish recipe08's AG News comparison first, retaining all fifteen SetFit sets.
    order = [
        ("setfit", ["ag_news", "subj"]),
        ("fineweb", None),
        ("setfit", [name for name in names if name not in ("subj", "ag_news")]),
    ]
    for suite, selected in order:
        for name, data, provenance in inputs(suite, args.cache / suite, selected):
            if time.monotonic() >= args.deadline:
                raise TimeoutError("whole-pod deadline reached")
            folder = args.output / (suite + "-" + name)
            append(
                args.output / "invocations.jsonl",
                {"dataset": name, "suite": suite, "status": "started", **args.hardware},
            )
            try:
                record = prepare(suite, name, data, provenance, folder)
                append(args.output / "classification-quality.jsonl", {**record, **args.hardware})
                emit("prepared-" + suite + "-" + name.replace("/", "_"), record)
                timed_dataset(args, folder, suite + "/" + name)
                record = prepare_mlp(data, provenance, folder, record)
                append(args.output / "classification-quality.jsonl", {**record, **args.hardware})
                timed_dataset(args, folder, suite + "/" + name, mlp=True)
                append(
                    args.output / "invocations.jsonl",
                    {"dataset": name, "suite": suite, "status": "completed", **args.hardware},
                )
            except Exception as error:
                append(
                    args.output / "invocations.jsonl",
                    {
                        "dataset": name,
                        "suite": suite,
                        "status": "failed",
                        "error": repr(error),
                        **args.hardware,
                    },
                )
            del data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["readme", "classifications", "large", "scaling", "showcase"],
        default="classifications",
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("folded classifier timing runs only on the Linux pod")
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 32:
        raise RuntimeError("pod exposes fewer than 32 allowed CPUs")
    os.sched_setaffinity(0, allowed[:32])
    args.output.mkdir(parents=True, exist_ok=True)
    args.hardware = {**environment(), "affinity_original": allowed}
    args.deadline = time.monotonic() + args.deadline_epoch - time.time()
    if args.stage == "readme":
        from pod_readme import run

        run(args)
        return
    if args.stage == "scaling":
        from pod_scaling import run

        run(args)
        return
    if args.stage == "large":
        from pod_large import run

        run(args)
        return
    if args.stage == "showcase":
        from pod_showcase import run

        run(args)
        return
    workload(args)
    export_jsonl("folded-speed-final", args.output / "speed-results.jsonl")
    export_jsonl("folded-quality-final", args.output / "classification-quality.jsonl")


if __name__ == "__main__":
    main()
