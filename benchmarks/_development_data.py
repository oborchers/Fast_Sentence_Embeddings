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
        corpus = receipt["corpora"][cell["corpus"]]
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


def _models(cells, work, cache, deadline=None):
    from fse import Encoder, StaticModel

    result, loaded = {}, {}
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
        loaded[repo] = _with_rule(cells, repo, snapshot, work, artifacts, model, deadline)
        for kind, record in artifacts.items():
            record["rule"] = _rule(loaded[repo] if kind == "float32-rule" else model)
        result[repo] = dict(revision=revision, artifacts=artifacts, dimensions=model.output_dim)
    return result, loaded


def _with_rule(cells, repo, snapshot, work, artifacts, model, deadline):
    """For a sentence-transformers model that a length-report cell reads: the model with its
    own length rule (the standard artifacts read it whole), saved as ``float32-rule``. The
    model the data cases of the repo are built from."""
    from fse import Encoder

    if repo.startswith("minishlab/") or not any(
        c["profile"] == "length-report@1" and c["model"] == repo for c in cells
    ):
        return model
    check(deadline)
    ruled = Encoder.from_pretrained(str(snapshot), normalize=True)
    path = work / (repo.replace("/", "--") + "-float32-rule.safetensors")
    ruled.save(path)
    artifacts["float32-rule"] = dict(
        path=str(path), sha256=sha256(path), fingerprint=ruled.fingerprint
    )
    return ruled


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
        ST_DOCUMENT_CODEPOINTS,
        declared,
        documents,
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
    else:
        limit = rule.max_chars if name == "wikitext-docs-m2v" else ST_DOCUMENT_CODEPOINTS
        if limit is None:
            raise ValueError("the pinned model has no max_chars for its documents")
        data["texts"] = documents(data["texts"], limit)
        derived = (
            f"non-empty rows joined with one space until more than {limit} code points; "
            "the incomplete rest is dropped"
        )
        stats = declared(model, data["texts"])
    data["source"] = {**data["source"], "derived": derived, "declared": stats, "rule": _rule(model)}
    return data


def prepare_child(job):
    deadline = job.get("wall_deadline")
    work = Path(job["work"])
    cache = job.get("hf_cache")
    corpora = {}
    models, loaded = _models(job["cells"], work, cache, deadline)
    for name in sorted({cell["corpus"] for cell in job["cells"]}):
        check(deadline)
        if name == "stsb":
            data = _stsb(deadline)
        elif name == "wikitext":
            data = _wikitext(work, cache, deadline)
        else:
            data = _length_corpus(name, work, cache, loaded, deadline)
        data["timing_texts"] = [" ".join(text.split()) for text in data["texts"]]
        data["timing_policy"] = "one-space whitespace; preserve row order, empty rows and headings"
        path = work / (name + ".json")
        atomic(path, data)
        corpora[name] = dict(
            path=str(path),
            sha256=sha256(path),
            texts=len(data["texts"]),
            source=data["source"],
            timing_utf8_bytes=sum(len(text.encode("utf-8")) for text in data["timing_texts"]),
            timing_policy=data["timing_policy"],
        )
    return {**LABEL, "corpora": corpora, "models": models}
