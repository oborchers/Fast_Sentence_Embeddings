"""Complete classification datasets, offline training then rotating inference pairs."""

from __future__ import annotations

import json
import os
import sys
import time
from itertools import product
from types import SimpleNamespace

from native_classifier import train
from pod_pairs import append, series
from pod_speed import POTION, ROOT, ensure_model
from prepare_fineweb import load_pages, split
from prepare_setfit import SOURCES, dataset

REPOS = (POTION, "minishlab/potion-base-32M")


def inference(args, suite, name, folder, cache):
    for repo, workers in product(REPOS, (1, 32)):
        kinds = (
            "model2vec",
            "model2vec-classifier",
            "tfidf" if suite == "setfit" else "fasttext",
        )
        base = [
            sys.executable,
            str(ROOT / "benchmarks/bench_classifier_compare.py"),
            "child",
            "--suite",
            suite,
            "--model",
            repo,
            "--models",
            str(args.models),
            "--workers",
            str(workers),
            "--table-dtype",
            "float32",
            "--passes",
            "3",
            "--classifiers",
            str(folder),
            "--cache",
            str(cache),
        ]
        if suite == "setfit":
            base.extend(["--datasets", name])
        for kind in kinds:
            commands = {label: [*base, "--kind", "fse"] for label in ("fse-a", "fse-b")}
            commands[kind] = [*base, "--kind", kind]
            series(
                commands,
                workers,
                args.output,
                args.deadline,
                f"classification/{suite}/{name}/{repo}/{workers}/{kind}",
            )


def native(args, data, folder, provenance):
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(16)
    with threadpool_limits(limits=16):
        for repo in REPOS:
            if time.monotonic() >= args.deadline:
                raise TimeoutError("pod deadline reached before classifier training")
            destination = folder / (repo.rsplit("/", maxsplit=1)[-1] + "-model2vec-pipeline")
            row = train(data, repo, destination, provenance, timed_training=True)
            append(args.output / "classification-quality.jsonl", row)


def run(args):
    import prepare_fineweb
    import prepare_setfit
    from compare_artifacts import seal_preparation
    from compare_inputs import identity
    from threadpoolctl import threadpool_limits

    for repo in REPOS:
        for dtype in ("float32", "float16", "int8"):
            ensure_model(repo, dtype, args)
    root = args.output / "classifiers"
    root.mkdir()
    settings = SimpleNamespace(
        models=args.models,
        coordination=None,
        results=args.output / "classification-quality.jsonl",
        pod_training_times=True,
    )
    # Each complete dataset produces its inference evidence before the next training job.
    specifications = json.loads(SOURCES.read_text())["datasets"]
    specifications.sort(key=lambda spec: spec["dataset"] != "SetFit/subj")
    for spec in specifications:
        name = spec["dataset"].removeprefix("SetFit/")
        settings.output, settings.cache = root / "setfit", args.cache / "setfit"
        settings.output.mkdir(exist_ok=True)
        prepare_setfit.prepare(settings, spec)
        data = dataset(spec, settings.cache, download=False)
        native(
            args,
            data,
            settings.output / name,
            {
                "dataset": spec["dataset"],
                "dataset_revision": spec["revision"],
                "split": "1000 stratified, seed 0",
            },
        )
        inference(args, "setfit", name, settings.output, settings.cache)
    settings.output, settings.cache = root / "fineweb", args.cache / "fineweb"
    settings.output.mkdir()
    data = split(*load_pages(settings.cache, download=True))
    provenance = {
        **identity(),
        "dataset": prepare_fineweb.REPO,
        "dataset_revision": prepare_fineweb.REVISION,
        "train_rows": len(data[0]),
        "evaluation_rows": len(data[2]),
        "train_digest": prepare_fineweb.digest(data[0]),
        "evaluation_digest": prepare_fineweb.digest(data[2]),
        "split": "default_rng(0) permutation, last 46743 held out",
        "label": "score >= 3",
        "preprocessing": "whitespace collapsed",
        "workers": 2,
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
    }
    (settings.output / "manifest.json").write_text(json.dumps(provenance))
    with threadpool_limits(limits=2):
        for repo in REPOS:
            prepare_fineweb.potion(settings, data, provenance, repo)
        prepare_fineweb.fasttext(settings, data, provenance)
        seal_preparation(settings.output)
    native(args, data, settings.output, provenance)
    inference(args, "fineweb", "fineweb", settings.output, settings.cache)
