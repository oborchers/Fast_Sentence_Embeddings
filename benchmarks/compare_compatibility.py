"""Keep each timing series on one implementation and one configuration per side."""

from __future__ import annotations

import json

FIELDS = (
    "comparison_identity_version",
    "commit",
    "runtime_fingerprint",
    "package_location",
    "versions",
    "machine",
    "cpu",
    "python",
    "kind",
    "model",
    "revision",
    "corpus",
    "input_digest",
    "texts",
    "shape",
    "table_dtype",
    "output_dtype",
    "threads",
    "requested_threads",
    "fse_workers",
    "tokenizer_parallelism",
    "parallel_batch_pool",
    "native_multiprocessing",
    "batch_size",
    "torch_num_threads",
    "first_encode_texts",
    "encoder_fingerprint",
    "artifact_provenance",
    "file_bytes",
    "load_source",
    "rust_source_revision",
    "rust_worker_build",
    "timed_passes",
    "query_batch",
    "k",
    "encoder_start",
    "vector_digest",
    "durability",
    "index_settings",
    "suite",
    "preparation_manifest",
)
REQUIRED = (
    "commit",
    "runtime_fingerprint",
    "package_location",
    "versions",
    "machine",
    "python",
    "kind",
    "model",
    "revision",
    "input_digest",
    "table_dtype",
    "output_dtype",
)


def signature(row: dict) -> str:
    if row.get("comparison_identity_version") == 2 and any(key not in row for key in REQUIRED):
        raise ValueError("timing record lacks required executed-runtime/configuration identity")
    values = {key: row[key] for key in FIELDS if key in row}
    if "warm_seconds" in row:
        values["observed_pass_count"] = len(row["warm_seconds"])
    if "datasets" in row:
        values["dataset_identities"] = [
            {
                key: part.get(key)
                for key in (
                    "dataset",
                    "texts",
                    "input_digest",
                    "preparation_manifest",
                    "artifact_identities",
                )
            }
            for part in row["datasets"]
        ]
    return json.dumps(values, sort_keys=True)


def require_same(records: list[dict]) -> None:
    if len({signature(row) for row in records}) > 1:
        raise ValueError(
            "timing series contains incompatible source/runtime/artifact/settings identities"
        )


def require_compatible(pairs: list[dict[str, dict]]) -> None:
    fse = [row for pair in pairs for label, row in pair.items() if label in ("fse-a", "fse-b")]
    require_same(fse)
    others = {label for pair in pairs for label in pair if label not in ("fse-a", "fse-b")}
    if len(others) > 1:
        raise ValueError("timing series changed its counterpart")
    for label in others:
        require_same([pair[label] for pair in pairs])


def identity_status(pairs: list[dict]) -> str:
    return (
        "executed runtime and configuration captured"
        if all(
            row.get("comparison_identity_version") == 2 for pair in pairs for row in pair.values()
        )
        else "historical observation: executed runtime fingerprint not captured, re-measure"
    )
