"""Small untimed adapter probes before expensive local measurement blocks."""

from __future__ import annotations

import json
from pathlib import Path

from _development_budget import check, check_args
from _development_heads_multi import configure as configure_heads
from _development_ids import mode_for
from _development_lengths import mode_for as lengths_for
from _development_pairs import job_for
from _development_process import checked, launch
from _development_search import configure
from _measurement import append, document


def preflight(args, cell, prepared, ledger, owner):
    ledger.current, ledger.stage = cell["id"], "adapter_preflight"
    ledger.progress()
    builds = (
        [("candidate", args.candidate_python)]
        if args.local_phase == "descriptive"
        else [("base", args.base_python), ("candidate", args.candidate_python)]
    )
    proofs, identities = [], []
    for name, python in builds:
        tag = cell["id"] + "-probe-" + name
        job = job_for(cell, prepared, 1, tag, owner)
        configure(job, cell, name, False)
        configure_heads(job, cell, name, False)
        job.update(
            action="probe",
            iterations=1,
            doc_ids=mode_for(cell, candidate=name == "candidate"),
            lengths=lengths_for(cell, candidate=name == "candidate"),
        )
        row = launch(args, job, python, tag)
        model = prepared["models"][cell["model"]]
        if (
            row["encoder_fingerprint"] != model["artifacts"][cell["table_dtype"]]["fingerprint"]
            or row["dimensions"] != model["dimensions"]
        ):
            raise ValueError("preflight model metadata differs from prepared provenance")
        identities.append(row["identity"])
        proofs.append(json.dumps(row["output"], sort_keys=True))
        append(args.results / "probes.jsonl", {"cell_id": cell["id"], "arm": name, **row})
        check_args(args)
    if len(set(proofs)) != 1:
        raise ValueError("preflight adapter outputs differ between builds")
    return identities


def probe(job):
    from _development_encode import operation, output_identity
    from _development_repeat import repeat
    from _development_stages import tokenizer
    from _development_token_reference import reference_batch

    from fse import Encoder

    check(job.get("wall_deadline"))
    model = Encoder.load(checked(job["artifact"]["path"], job["artifact"]["sha256"]), mmap=True)
    check(job.get("wall_deadline"))
    data = document(checked(job["corpus"]["path"], job["corpus"]["sha256"]))
    if len(data.get("timing_texts", data["texts"])) != job["corpus"]["texts"]:
        raise ValueError("preflight corpus row count differs from prepared provenance")
    texts = data.get("timing_texts", data["texts"])[:16]
    if not texts:
        raise ValueError("preflight requires nonempty source rows")
    job["texts"] = len(texts)
    requested_fraction = job.get("filter_fraction")
    check(job.get("wall_deadline"))
    batch = tokenizer(model, texts, job)()
    check(job.get("wall_deadline"))
    expected_batch = reference_batch(model, texts)
    if output_identity(batch) != output_identity(expected_batch):
        raise ValueError("preflight tokenizer differs from independent reference")
    check(job.get("wall_deadline"))
    if job["stage"] in ("store-open", "store-search"):
        from _development_probe_store import probe_store

        proof = probe_store(model, texts, job, _store)
    elif job["stage"] == "tokenizer":
        proof = output_identity(batch)
    elif job["stage"] == "pooling":
        proof = _pool(model, texts, job, expected_batch)
    elif job["stage"] in ("head-folded", "head-with-vectors", "head-labels"):
        from _development_heads import probe as head_probe

        proof = head_probe(job, texts)
    elif job["stage"] in ("head-group-folded", "head-group-mixed", "head-group-score-sink"):
        from _development_heads_multi import probe as group_probe

        proof = group_probe(job, texts)
    else:
        expected = model.encode(texts, dtype=job["output_dtype"], workers=1, backend="numpy")
        check(job.get("wall_deadline"))
        _, got, _ = repeat(operation(model, texts, job), 1)
        if job["stage"] == "encode-sink":
            _store(got["folder"], expected)
            proof = output_identity(got)
            (Path(job["work"]) / (job["tag"] + "-timing.txt")).unlink()
        else:
            proof = _equal(got, expected)
    from _development_lengths import check_reference

    return {
        "output": proof,
        "lengths": check_reference(model, texts, job.get("lengths", "none")),
        "source_rows": len(texts),
        "probe_filter_count": job.get("filter_count"),
        "requested_filter_fraction": requested_fraction,
        "probe_filter_fraction": job.get("filter_fraction"),
        "document_ids": job.get("doc_ids", "none"),
        "encoder_fingerprint": model.fingerprint,
        "dimensions": model.output_dim,
        "reference": "independent tokenizer and numpy backend",
        "valid_for_comparison": False,
        "functional_only": True,
        "artifact": job["artifact"],
        "corpus": job["corpus"],
    }


def _pool(model, texts, job, expected_batch):
    import numpy as np
    from _development_stages import pooling, prepare_batch
    from _measurement import atomic, sha256

    from fse._backends._numpy import NumpyBackend

    small = Path(job["work"]) / (job["tag"] + "-corpus.json")
    atomic(small, {"timing_texts": texts})
    small_job = {**job, "corpus": {"path": str(small), "sha256": sha256(small)}}
    small_job["batch"] = prepare_batch(small_job)
    check(job.get("wall_deadline"))
    got = pooling(model, small_job)()
    backend = NumpyBackend()
    expected = np.empty_like(got)
    backend.rows(expected_batch, backend.prepare(model._spec.pool_input(), ()), expected)
    return _equal(got, expected)


def _equal(got, expected):
    from _development_encode import output_identity

    proof = output_identity(got)
    if proof != output_identity(expected):
        raise ValueError("preflight adapter shape, dtype or bits differ from reference")
    return proof


def _store(folder, expected):
    import numpy as np

    from fse import VectorStore
    from fse._reference import inverse_norms

    codes = expected.codes if hasattr(expected, "codes") else expected
    arrays = {"vectors": codes, "inv_norms": inverse_norms(codes)}
    if hasattr(expected, "scales"):
        arrays["scales"] = expected.scales
    with VectorStore.open(folder) as store:
        store.verify()
        if len(store) != len(codes):
            raise ValueError("preflight store row count differs")
        for shard, tensors in store._shards:
            for name, array in arrays.items():
                view = array[shard.first_row : shard.first_row + shard.rows]
                if (
                    name not in tensors
                    or tensors[name].dtype != view.dtype
                    or tensors[name].shape != view.shape
                ):
                    raise ValueError("preflight store tensor metadata differs")
                if not np.array_equal(tensors[name].view(np.uint8), view.view(np.uint8)):
                    raise ValueError("preflight store bits differ from reference")
