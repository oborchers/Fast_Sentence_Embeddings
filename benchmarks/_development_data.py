"""Pinned public inputs and identical encoder artifacts, always outside the checkout."""

from __future__ import annotations

import sys
from contextlib import suppress
from pathlib import Path

from _development_budget import check, check_args, stopped
from _development_locks import coordinated
from _development_process import checked, interpreter, launch
from _development_system import LABEL
from _measurement import append, atomic, document, sha256
from _measurement_config import MODEL_PINS, PINS
from _measurement_setup import outside_checkout
from _quality.prefilter_inputs import corpus_names


def prepare(args, cells, ledger):
    check_args(args)
    interpreter(args.base_python)
    interpreter(args.candidate_python)
    if args.reuse_work:
        root = args.reuse_work.resolve()
        receipt = document(root / "prepared.json")
        _verify(receipt, cells, args.wall_deadline)
        atomic(args.results / "inputs.json", receipt)
        return receipt

    def notify(row):
        append(args.results / "coordination.jsonl", row)

    with coordinated(
        "heavy",
        args.worker_name,
        "prepare pinned development inputs",
        notify,
        lambda: stopped(ledger),
    ):
        receipt = launch(
            args,
            {
                "action": "prepare",
                "cells": cells,
                "hf_cache": str(args.hf_cache) if args.hf_cache else None,
            },
            args.base_python,
            "preparation",
        )
    atomic(args.work / "prepared.json", receipt)
    _verify(receipt, cells, args.wall_deadline)
    atomic(args.results / "inputs.json", receipt)
    return receipt


def _verify(receipt, cells, deadline=None):
    if receipt.get("mode") != "local-development" or receipt.get("publishable") is not False:
        raise ValueError("prepared data must identify local development mode")
    for cell in cells:
        check(deadline)
        model = receipt["models"][cell["model"]]
        if model["revision"] != MODEL_PINS[cell["model"]]:
            raise ValueError("prepared model pin differs")
        for record in model["artifacts"].values():
            check(deadline)
            checked(record["path"], record["sha256"])
        for name in corpus_names(cell):
            corpus = receipt["corpora"][name]
            checked(corpus["path"], corpus["sha256"])


def _snapshot(repo, revision, work, existing, dataset=False, patterns=None):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    options = dict(
        repo_type="dataset" if dataset else "model",
        revision=revision,
        token=False,
        allow_patterns=patterns,
    )
    if existing:
        outside_checkout(Path(existing))
        with suppress(LocalEntryNotFoundError):
            return Path(
                snapshot_download(repo, cache_dir=existing, local_files_only=True, **options)
            )
    return Path(snapshot_download(repo, cache_dir=work / "hub", max_workers=2, **options))


def _stsb(deadline=None):
    # Reuse the original labels and checksum-verified Git/download fallback.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from stsb import CHECKSUMS, COMMIT, read_rows

    texts, sections = [], []
    for split in ("train", "dev", "test"):
        check(deadline)
        rows = read_rows(split)
        sections.append(
            dict(split=split, begin=len(texts), pairs=len(rows), labels=[float(r[4]) for r in rows])
        )
        texts.extend(r[5] for r in rows)
        texts.extend(r[6] for r in rows)
    return dict(texts=texts, score_slices=sections, source=dict(commit=COMMIT, checksums=CHECKSUMS))


def _wikitext(work, cache, deadline=None):
    import pyarrow.parquet as pq

    pin = PINS["corpus"]
    snapshot = _snapshot(
        pin["repo"],
        pin["revision"],
        work,
        cache,
        dataset=True,
        patterns=["wikitext-103-raw-v1/test-*.parquet"],
    )
    check(deadline)
    parts = sorted((snapshot / "wikitext-103-raw-v1").glob("test-*.parquet"))
    if not parts:
        raise ValueError("pinned WikiText test split is missing")
    texts = [
        row
        for part in parts
        for batch in pq.ParquetFile(part).iter_batches(columns=["text"])
        for row in batch.column(0).to_pylist()
    ]
    if not texts or any(not isinstance(row, str) for row in texts):
        raise ValueError("WikiText contains no text or a non-string row")
    return dict(
        texts=texts,
        score_slices=[],
        source=dict(
            repo=pin["repo"],
            revision=pin["revision"],
            split="test",
            preprocessing="original rows, including empty rows and headings",
        ),
    )


def _wikitext_train(work, cache, deadline=None):
    """The raw paragraphs of the pinned training split by the rule of hub_classic_corpus
    (strip, not empty, no heading, a letter or digit), whitespace collapsed, not tokenized."""
    import re

    import pyarrow.parquet as pq
    from _quality.prefilter_inputs import TRAIN_ROWS

    pin = PINS["corpus"]
    snapshot = _snapshot(
        pin["repo"],
        pin["revision"],
        work,
        cache,
        dataset=True,
        patterns=["wikitext-103-raw-v1/train-*.parquet"],
    )
    parts = sorted((snapshot / "wikitext-103-raw-v1").glob("train-*.parquet"))
    if not parts:
        raise ValueError("pinned WikiText training split is missing")
    texts = []
    for part in parts:
        for batch in pq.ParquetFile(part).iter_batches(columns=["text"], batch_size=4096):
            check(deadline)
            for row in batch.column(0).to_pylist():
                text = row.strip()
                if text and not text.startswith("=") and re.search("[A-Za-z0-9]", text):
                    texts.append(" ".join(text.split()))
    if len(texts) != TRAIN_ROWS:
        raise ValueError(
            f"WikiText training store has {len(texts)} paragraphs, expected {TRAIN_ROWS}"
        )
    return dict(
        texts=texts,
        score_slices=[],
        source=dict(
            repo=pin["repo"],
            revision=pin["revision"],
            split="train",
            preprocessing="hub_classic_corpus row rule, raw text with collapsed whitespace",
        ),
    )


def _models(cells, work, cache, deadline=None):
    from fse import Encoder, StaticModel

    result = {}
    for repo in sorted({cell["model"] for cell in cells}):
        check(deadline)
        revision = MODEL_PINS[repo]
        snapshot = _snapshot(
            repo,
            revision,
            work,
            cache,
            patterns=["*.json", "*.safetensors", "*/*.json", "*/*.safetensors"],
        )
        check(deadline)
        model = (
            StaticModel.from_pretrained(str(snapshot), normalize=True).encoder
            if repo.startswith("minishlab/")
            else Encoder.from_pretrained(str(snapshot), normalize=True, truncate=False)
        )
        artifacts = {}
        for dtype in ("float32", "float16", "int8"):
            check(deadline)
            path = work / (repo.replace("/", "--") + "-" + dtype + ".safetensors")
            current = model if dtype == "float32" else model.quantize(dtype)
            current.save(path)
            artifacts[dtype] = dict(
                path=str(path), sha256=sha256(path), fingerprint=current.fingerprint
            )
        result[repo] = dict(revision=revision, artifacts=artifacts, dimensions=model.output_dim)
    return result


def _corpus(name, work, cache, deadline):
    if name == "wikitext-train":
        data = _wikitext_train(work, cache, deadline)
        data["timing_texts"] = data["texts"]  # already collapsed; saved once (see prepare_child)
        data["timing_policy"] = "paragraphs of the training split, one-space whitespace"
        return data
    data = _stsb(deadline) if name == "stsb" else _wikitext(work, cache, deadline)
    data["timing_texts"] = [" ".join(text.split()) for text in data["texts"]]
    data["timing_policy"] = "one-space whitespace; preserve row order, empty rows and headings"
    return data


def prepare_child(job):
    deadline = job.get("wall_deadline")
    work = Path(job["work"])
    cache = job.get("hf_cache")
    corpora = {}
    for name in sorted({n for cell in job["cells"] for n in corpus_names(cell)}):
        check(deadline)
        data = _corpus(name, work, cache, deadline)
        timing = data["timing_texts"]
        if timing is data["texts"]:
            del data["timing_texts"]  # the training store texts are saved once
        path = work / (name + ".json")
        atomic(path, data)
        corpora[name] = dict(
            path=str(path),
            sha256=sha256(path),
            texts=len(data["texts"]),
            source=data["source"],
            timing_utf8_bytes=sum(len(text.encode("utf-8")) for text in timing),
            timing_policy=data["timing_policy"],
        )
    return {**LABEL, "corpora": corpora, "models": _models(job["cells"], work, cache, deadline)}
