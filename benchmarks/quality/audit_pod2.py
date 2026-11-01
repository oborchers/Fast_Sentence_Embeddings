"""Audit all transferred Pod 2 series and enumerate the exact unmeasured queue."""

import argparse
import json
from pathlib import Path

from pod2_audit_cells import PIN, audit
from pod_speed import POTION, STATIC
from prepare_setfit import SOURCES


def plan():
    result = {
        name: []
        for name in (
            "probe",
            "counterparts",
            "pipelines",
            "float32",
            "int8",
            "classification",
            "search",
            "float16",
        )
    }
    result["probe"] = [f"encode/{POTION}/float32/massive/1/table"]
    for corpus in ("wikitext", "massive"):
        for worker in (1, 32):
            for kind in ("model2vec", "rust", "fastembed"):
                result["counterparts"].append(f"encode/{POTION}/float32/{corpus}/{worker}/{kind}")
                add_pipelines(result["pipelines"], corpus, worker, kind)
    add_tables(result)
    add_classification(result["classification"])
    result["search"] = [
        f"search/{w}/{b}/{k}" for w in (1, 32) for b in (1, 64) for k in ("faiss", "hnswlib")
    ]
    return result


def add_pipelines(rows, corpus, worker, kind):
    for source in ("text", "arrow"):
        for durability in ("full", "none"):
            rows.append(f"pipeline/{corpus}/{worker}/{source}/{durability}/{kind}")


def add_tables(result):
    for dtype in ("float32", "int8", "float16"):
        corpora = ("wikitext", "massive") if dtype == "float32" else ("wikitext",)
        result[dtype] = [
            f"encode/{repo}/{dtype}/{corpus}/{worker}/table"
            for corpus in corpora
            for repo in STATIC
            for worker in (1, 32)
        ]


def add_classification(rows):
    specs = [
        ("setfit", r["dataset"].removeprefix("SetFit/"))
        for r in json.loads(SOURCES.read_text())["datasets"]
    ]
    specs.append(("fineweb", "fineweb"))
    for suite, name in specs:
        for repo in (POTION, "minishlab/potion-base-32M"):
            for worker in (1, 32):
                kinds = (
                    "model2vec",
                    "model2vec-classifier",
                    "tfidf" if suite == "setfit" else "fasttext",
                )
                rows.extend(
                    f"classification/{suite}/{name}/{repo}/{worker}/{kind}" for kind in kinds
                )


def one_stage(name, expected, cells, invocations):
    found = [c for c in cells if c["stage"] == name]
    observed = {c["cell"] for c in found}
    states = [r["status"] for r in invocations if r.get("stage") == name]
    return {
        "stage": name,
        "execution": "complete"
        if "completed" in states
        else ("partial" if "started" in states else "not started"),
        "planned_cells": len(expected),
        "observed_cells": len(observed),
        "valid_cells": sum(c["eligible"] for c in found),
        "invalid_or_incomplete_cells": sum(not c["eligible"] for c in found),
        "not_started": sorted(set(expected) - observed),
        "unexpected": sorted(observed - set(expected)),
    }


def stage_summaries(cells, invocations):
    return [one_stage(name, expected, cells, invocations) for name, expected in plan().items()]


def summarize(root):
    raw = [json.loads(line) for line in (root / "speed-results.jsonl").read_text().splitlines()]
    probe = json.loads((root / "probe.json").read_text())["series_id"]
    groups = {}
    for row in raw:
        groups.setdefault(row["series_id"], []).append(row)
    cells = [audit(identifier, entries, probe) for identifier, entries in groups.items()]
    invocations = [
        json.loads(line) for line in (root / "invocations.jsonl").read_text().splitlines()
    ]
    return {
        "executed_commit": PIN,
        "format": "pod2-independent-audit-v1",
        "source_rows": len(raw),
        "source_series": len(groups),
        "stages": stage_summaries(cells, invocations),
        "cells": cells,
        "rust": rust_annotations(raw),
        "quality_records": sum(1 for _ in (root / "classification-quality.jsonl").open()),
        "bootstrap_returncode": json.loads((root / "bootstrap-invocation.json").read_text())[
            "returncode"
        ],
        "limitations": [
            "Eligibility recomputes paired timing, all-control A/A, input and output-digest stability, own backend identity and worker budget. It does not rerun any timed operation.",
            "Pod2 did not retain full encode arrays for a new cross-library numerical audit. Fresh README panels still use Pod3's explicit full-output checks.",
            "Missing model/joblib weights and input files were not transferred. Their recorded sizes, digests and source manifests are retained, not rehashed from absent files.",
            "Classifier RSS is fresh-parent process high-water memory, not a sum of Model2Vec multiprocessing children.",
            "Pipeline rows omit UTC timestamps. Their order is bounded by dated encode rows in the same unfiltered journal.",
        ],
    }


def rust_annotations(rows):
    operations = [
        (i, row)
        for i, row in enumerate(rows, 1)
        if row.get("event") == "operation" and row["kind"] == "rust"
    ]
    return {
        "affected_rows": 0,
        "total_rows": len(operations),
        "scope": "All executing rows name97b7418. Both compare_rust_worker.measure and pod_pipeline.main set RAYON_NUM_THREADS and TOKENIZERS_PARALLELISM before Rust execution. ea17433 fixes only the new Pod3 large_write_paths path, which Pod2 never executes.",
        "source_files": ["benchmarks/compare_rust_worker.py", "benchmarks/quality/pod_pipeline.py"],
        "annotations": [
            {
                "logical_raw_line": i,
                "series_id": r["series_id"],
                "cell": r["cell"],
                "round": r["round"],
                "bug_affected": False,
                "configured_rayon_threads": r.get("requested_threads", r.get("workers")),
                "runtime_fingerprint": r["runtime_fingerprint"],
                "measured_tokenizer_threads": None,
            }
            for i, r in operations
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.input), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
