"""Pinned public inputs and identical encoder artifacts, always outside the checkout."""

from __future__ import annotations

import sys
from contextlib import suppress
from pathlib import Path

from _development_budget import check_args, stopped
from _development_identity import STAGE, prepare_cases, verify
from _development_locks import coordinated
from _development_process import checked, interpreter, launch
from _development_system import LABEL
from _harness.budget import check
from _harness.records import append, atomic, document, sha256
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
        if cell["stage"] == STAGE:
            verify(receipt["corpora"][cell["corpus"]])
            continue
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


def _agnews(work, cache, deadline=None):
    """The corpus and the query pool of the DR-255 probe: PCG64 seed 0 over the pinned AG News
    splits, whitespace collapsed; the pool holds unique test texts that are not in the corpus."""
    import numpy as np
    import pyarrow.parquet as pq
    from _quality.binary_inputs import AGNEWS, CORPUS_ROWS, POOL_QUERIES

    snapshot = _snapshot(
        AGNEWS["repo"], AGNEWS["revision"], work, cache, dataset=True, patterns=["data/*.parquet"]
    )

    def split(name):
        check(deadline)
        table = pq.read_table(next((snapshot / "data").glob(name + "-*.parquet")), columns=["text"])
        return [" ".join(t.split()) for t in table.column(0).to_pylist()]

    train, test = split("train"), split("test")
    rng = np.random.Generator(np.random.PCG64(0))
    corpus = [train[i] for i in np.sort(rng.choice(len(train), CORPUS_ROWS, replace=False))]
    stored, seen, pool = set(corpus), set(), []
    for i in rng.permutation(len(test)):
        text = test[i]
        if text and text not in stored and text not in seen:
            seen.add(text)
            pool.append(text)
    pool = pool[:POOL_QUERIES]
    if len(pool) < POOL_QUERIES:
        raise ValueError("AG News test has too few texts outside the corpus")
    return dict(
        texts=corpus + pool,
        score_slices=[
            dict(split="corpus", begin=0, count=len(corpus)),
            dict(split="queries", begin=len(corpus), count=len(pool)),
        ],
        source=dict(
            **AGNEWS,
            seed=0,
            generator="PCG64",
            preprocessing="collapsed whitespace; corpus sampled from train without replacement; "
            "pool of unique test texts outside the corpus in a seeded order",
        ),
    )


def _models(cells, work, cache, deadline=None):
    from fse import Encoder, StaticModel

    result, loaded = {}, {}
    for repo in sorted({cell["model"] for cell in cells if cell["model"] is not None}):
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
        loaded[repo] = model
        result[repo] = dict(revision=revision, artifacts=artifacts, dimensions=model.output_dim)
    return result, loaded


def _rule(model):
    """The length rule ``[max_tokens, max_chars]`` of a model (null: none)."""
    spec = model._spec.tokenizer
    return [spec.max_tokens, spec.max_chars]


def _length_corpus(name, work, cache, models, deadline):
    """The data cases of the length report (DR-163), built from the rows of the pinned
    WikiText test split by the rule of one model. The statistics of the case, from a report
    outside every timer, are declared with it."""
    from _development_lengths import (
        REPO_OF,
        declared,
        documents,
        documents_at_cut,
        uncut_mask,
    )

    data = _wikitext(work, cache, deadline)
    model = models[REPO_OF[name]]
    rule = model._spec.tokenizer
    normalized = [" ".join(row.split()) for row in data["texts"]]
    if name == "wikitext-lines-uncut":
        keep = uncut_mask(model, normalized)
        data["texts"] = [row for row, ok in zip(data["texts"], keep, strict=True) if ok]
        derived = "the rows of the split that the length rule does not cut, in order"
        stats = declared(model, [t for t, ok in zip(normalized, keep, strict=True) if ok], True)
    elif name == "wikitext-docs-at-cut":
        data["texts"] = documents_at_cut(model, documents(data["texts"], rule.max_chars), deadline)
        derived = (
            "the documents of wikitext-docs-m2v, each cut after the word at which it has "
            "exactly max_tokens ids; documents that never have exactly that many are dropped"
        )
        stats = declared(model, data["texts"])
    else:
        data["texts"] = documents(data["texts"], rule.max_chars)
        derived = (
            f"non-empty rows joined with one space until more than {rule.max_chars} code "
            "points; the incomplete rest is dropped"
        )
        stats = declared(model, data["texts"])
    data["source"] = {**data["source"], "derived": derived, "declared": stats, "rule": _rule(model)}
    return data


def _corpus(name, work, cache, loaded, deadline):
    if name == "wikitext-train":
        data = _wikitext_train(work, cache, deadline)
        data["timing_texts"] = data["texts"]  # already collapsed; saved once (see prepare_child)
        data["timing_policy"] = "paragraphs of the training split, one-space whitespace"
        return data
    if name == "agnews":
        data = _agnews(work, cache, deadline)
    elif name == "stsb":
        data = _stsb(deadline)
    elif name == "wikitext":
        data = _wikitext(work, cache, deadline)
    else:
        data = _length_corpus(name, work, cache, loaded, deadline)
    data["timing_texts"] = [" ".join(text.split()) for text in data["texts"]]
    data["timing_policy"] = "one-space whitespace; preserve row order, empty rows and headings"
    return data


def prepare_child(job):
    deadline = job.get("wall_deadline")
    work = Path(job["work"])
    cache = job.get("hf_cache")
    corpora = {}
    models, loaded = _models(job["cells"], work, cache, deadline)
    corpora.update(prepare_cases(job["cells"], work, cache, deadline))
    names = {n for c in job["cells"] if c["stage"] != STAGE for n in corpus_names(c)}
    for name in sorted(names):
        check(deadline)
        data = _corpus(name, work, cache, loaded, deadline)
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
    return {**LABEL, "corpora": corpora, "models": models}
