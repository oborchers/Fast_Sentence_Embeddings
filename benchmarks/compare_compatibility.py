"""Keep each timing series on one implementation and one configuration per side."""

from __future__ import annotations

import json

FIELDS = (
    "comparison_identity_version",
    "commit",
    "runtime_fingerprint",
    "runtime_package",
    "runtime_package_version",
    "runtime_fingerprint_scope",
    "package_location",
    "versions",
    "threadpools",
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


def validate_record_identity(row: dict) -> None:
    if row.get("comparison_identity_version", 0) >= 2 and any(key not in row for key in REQUIRED):
        raise ValueError("timing record lacks required executed-runtime/configuration identity")
    if row.get("comparison_identity_version", 0) >= 3:
        from compare_runtime import expected_package

        if row.get("runtime_package") != expected_package(row["kind"]):
            raise ValueError("counterpart row has the wrong executing backend identity")
        if not row.get("runtime_package_version"):
            raise ValueError("backend runtime package version is missing")


def signature(row: dict) -> str:
    validate_record_identity(row)
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
                    "prediction_digest",
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
    rows = [row for pair in pairs for row in pair.values()]
    if not all(row.get("comparison_identity_version", 0) >= 2 for row in rows):
        return "historical observation: executed runtime fingerprint not captured, re-measure"
    counterparts = [row for row in rows if row.get("kind") != "fse"]
    if any(row.get("comparison_identity_version", 0) < 3 for row in counterparts):
        return (
            "fse runtime captured; counterpart runtime misattributed to fse, re-measure counterpart"
        )
    return "backend-specific executed runtime and configuration captured"


def backend_attribution(row: dict) -> str:
    version = row.get("comparison_identity_version", 0)
    if version < 2:
        return "historical runtime identity not captured"
    if row.get("kind") == "fse":
        return "fse runtime captured"
    if version < 3:
        return "counterpart identity incorrectly points to fse; re-measure"
    validate_record_identity(row)
    return "own backend runtime captured"
