"""Linux pod speed stages, ordered for useful partial evidence under four hours."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_models import artifact_path  # isort: skip
from compare_inputs import DATASETS  # isort: skip
from pod_models import HUB, hub_model  # isort: skip
from pod_pairs import append, series  # isort: skip
from pod_artifacts import emit, export_jsonl  # isort: skip
from classic_sources import SOURCES  # isort: skip

ROOT = Path(__file__).resolve().parents[2]
POTION = HUB[0]
STATIC = (*HUB, *(f"fse/{name}" for name in SOURCES))


def ensure_model(repo, dtype, args):
    from preparation_catalog import register
    from prepare_classics import prepare

    from fse import Encoder

    manifest = args.models / "artifact-identities.json"
    rows = json.loads(manifest.read_text())["artifacts"] if manifest.exists() else []
    if any(row["source"] == repo and row["table_dtype"] == dtype for row in rows):
        return
    if dtype == "float32":
        if repo in HUB:
            hub_model(repo, args.models)
        else:
            prepare(repo.removeprefix("fse/"), args.models, args.cache, 16, [dtype], proxy=True)
        return
    ensure_model(repo, "float32", args)
    path = artifact_path(args.models, repo, "float32")
    encoder = Encoder.load(path, mmap=True)
    source = next(
        row for row in json.loads(manifest.read_text())["artifacts"] if row["artifact"] == path.name
    )
    output = path.with_name(path.name.replace("-float32.", f"-{dtype}."))
    encoder.quantize(dtype).save(output)
    register(args.models, output, source["preparation_record"], encoder.fingerprint)


def encode_command(args, repo, dtype, corpus, workers, kind):
    return [
        sys.executable,
        str(ROOT / "benchmarks/bench_compare.py"),
        "child",
        "--kind",
        kind,
        "--model",
        repo,
        "--table-dtype",
        dtype,
        "--corpus",
        corpus,
        "--threads",
        str(workers),
        "--passes",
        "3",
        "--fse-models",
        str(args.models),
        "--canonical-newlines",
    ]


def encode_cell(args, repo, dtype, corpus, workers, against=None, minimum=0):
    ensure_model(repo, dtype, args)
    commands = {
        label: encode_command(args, repo, dtype, corpus, workers, "fse")
        for label in ("fse-a", "fse-b")
    }
    if against:
        commands[against] = encode_command(args, repo, "float32", corpus, workers, against)
    return series(
        commands,
        workers,
        args.output,
        args.deadline,
        f"encode/{repo}/{dtype}/{corpus}/{workers}/{against or 'table'}",
        minimum_seconds=minimum,
    )


def download_corpus(name):
    from huggingface_hub import snapshot_download

    repo, revision = DATASETS[name]
    patterns = (
        ["wikitext-103-raw-v1/validation-*", "wikitext-103-raw-v1/test-*"]
        if name == "wikitext"
        else ["test/*.json.gz"]
    )
    snapshot_download(
        repo, repo_type="dataset", revision=revision, allow_patterns=patterns, max_workers=2
    )


def probe(args):
    download_corpus("massive")
    args.deadline = min(args.deadline, time.monotonic() + 120)
    result = encode_cell(args, POTION, "float32", "massive", 1, minimum=115)
    accepted = result["complete"] and result["aa_max_ratio"] <= 1.1
    emit("host-probe", {**result, "accepted": accepted, "host_load_rejects": False})
    (args.output / "probe.json").write_text(json.dumps({**result, "accepted": accepted}))
    return accepted


def counterparts(args):
    for corpus in ("wikitext", "massive"):
        download_corpus(corpus)
        for workers in (1, 32):
            for kind in ("model2vec", "rust", "fastembed"):
                encode_cell(args, POTION, "float32", corpus, workers, against=kind)


def pipelines(args):
    from pod_pipeline import canonical_inputs

    inputs = args.output / "pipeline-inputs"
    for corpus in ("wikitext", "massive"):
        canonical_inputs(corpus, inputs)
        for workers, durability, source in product((1, 32), ("full", "none"), ("text", "arrow")):
            base = [
                sys.executable,
                str(ROOT / "benchmarks/quality/pod_pipeline.py"),
                "--corpus",
                corpus,
                "--workers",
                str(workers),
                "--durability",
                durability,
                "--models",
                str(args.models),
                "--inputs",
                str(inputs),
            ]
            for kind in ("model2vec", "rust", "fastembed"):
                commands = {
                    label: [*base, "--kind", "fse", "--input-format", source]
                    for label in ("fse-a", "fse-b")
                }
                commands[kind] = [*base, "--kind", kind]
                series(
                    commands,
                    workers,
                    args.output,
                    args.deadline,
                    f"pipeline/{corpus}/{workers}/{source}/{durability}/{kind}",
                )


def tables(args, dtype, corpora):
    for corpus in corpora:
        for repo in STATIC:
            for workers in (1, 32):
                encode_cell(args, repo, dtype, corpus, workers)


def searches(args):
    temporary = args.output / "search-temporary"
    temporary.mkdir()
    for workers, batch, kind in product((1, 32), (1, 64), ("faiss", "hnswlib")):
        base = [
            sys.executable,
            str(ROOT / "benchmarks/bench_search_compare.py"),
            "child",
            "--models",
            str(args.models),
            "--tmp",
            str(temporary),
            "--workers",
            str(workers),
            "--batch",
            str(batch),
            "--passes",
            "3",
        ]
        commands = {label: [*base, "--kind", "fse"] for label in ("fse-a", "fse-b")}
        commands[kind] = [*base, "--kind", kind]
        series(commands, workers, args.output, args.deadline, f"search/{workers}/{batch}/{kind}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "probe",
            "counterparts",
            "pipelines",
            "float32",
            "int8",
            "float16",
            "classification",
            "search",
        ],
    )
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("all speed measurements are restricted to Linux pods")
    original = sorted(os.sched_getaffinity(0))
    if len(original) < 32:
        raise RuntimeError("pod exposes fewer than 32 allowed CPUs")
    os.sched_setaffinity(0, original[:32])
    args.output.mkdir(parents=True, exist_ok=True)
    append(
        args.output / "invocations.jsonl",
        {
            "stage": args.stage,
            "affinity_original": original,
            "affinity_selected": original[:32],
            "status": "started",
        },
    )
    args.deadline = time.monotonic() + args.deadline_epoch - time.time()
    if args.stage == "probe":
        if not probe(args):
            raise SystemExit(10)
    elif args.stage == "counterparts":
        counterparts(args)
    elif args.stage == "pipelines":
        pipelines(args)
    elif args.stage == "classification":
        from pod_classification import run

        run(args)
    elif args.stage == "search":
        searches(args)
    else:
        tables(
            args, args.stage, ("wikitext", "massive") if args.stage == "float32" else ("wikitext",)
        )
    export_jsonl("speed-" + args.stage, args.output / "speed-results.jsonl")
    append(args.output / "invocations.jsonl", {"stage": args.stage, "status": "completed"})


if __name__ == "__main__":
    main()
