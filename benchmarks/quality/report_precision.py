"""Report quality coverage and per-task quantization deltas for one measured setup.

Use the pinned quality environment. This never replaces a missing task with a
subset score. The optional 40-task English mean is labeled with MIND deferred;
the official complete 41-task English mean remains null until every task exists.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import mteb
from protocol import compatible_results


def setup(row: dict) -> str:
    fields = (
        "method_fingerprint",
        "fse_reference_runtime_fingerprint",
        "suite_fingerprint",
        "versions",
        "workers",
        "device",
        "machine",
        "python",
    )
    values = {key: row.get(key) for key in fields}
    values["fse_reference_runtime_fingerprint"] = row.get(
        "fse_reference_runtime_fingerprint", row.get("runtime_fingerprint")
    )
    values["python"] = (values["python"] or "").split(" ", maxsplit=1)[0]
    return json.dumps(values, sort_keys=True)


def mean_complete(rows: dict, tasks: set[str]):
    return (
        statistics.mean(rows[name]["main_score"] for name in sorted(tasks))
        if tasks <= rows.keys()
        else None
    )


def baseline_name(name: str) -> str | None:
    if not name.startswith("fse/"):
        return None
    for dtype in ("float32", "float16", "int8"):
        if name.endswith("-" + dtype):
            return name.removesuffix(dtype) + "float32"
    return None


def matching_runtime(row: dict, baseline: dict) -> bool:
    keys = ("runtime_package", "runtime_package_version", "runtime_fingerprint")
    return all(row.get(key) and row[key] == baseline.get(key) for key in keys)


def task_deltas(rows: dict, baseline: dict) -> list[dict]:
    result = []
    for name, row in sorted(rows.items()):
        other = baseline.get(name)
        matches = other and matching_runtime(row, other)
        matches = matches and row["task_metadata"] == other["task_metadata"]
        matches = (
            matches
            and row.get("source_revision")
            and row.get("source_revision") == other.get("source_revision")
        )
        matches = matches and row.get("float32_parent_fingerprint") == other.get(
            "encoder_fingerprint"
        )
        result.append(
            {
                "task": name,
                "benchmark": row["benchmark"],
                "metric": row["task_metadata"]["main_score"],
                "score": row["main_score"],
                "float32_score": other["main_score"] if matches else None,
                "delta_vs_float32_points": 100 * (row["main_score"] - other["main_score"])
                if matches
                else None,
            }
        )
    return result


def model_records(selected, tasks):
    groups = compatible_results(selected, tasks)
    models = {}
    for key, group in groups.items():
        name = json.loads(key)["model"]
        if name in models:
            raise ValueError(f"multiple incompatible artifact/revision configurations for {name}")
        models[name] = {row["task"]: row for row in group}
    return models


def checked_artifact(name, source, artifacts):
    artifact = artifacts.get(name.removeprefix("fse/") + ".safetensors")
    if artifact and artifact.get("encoder_fingerprint") != source.get("encoder_fingerprint"):
        raise ValueError(f"{name}: artifact manifest differs from the measured fingerprint")
    return artifact


def summarize(rows: list[dict], selected_setup: str, artifacts: dict) -> dict:
    tasks = [
        task
        for benchmark in ("MTEB(eng, v2)", "NanoBEIR")
        for task in mteb.get_benchmark(benchmark).tasks
    ]
    selected = [
        {**row, "python": (row.get("python") or "").split(" ", maxsplit=1)[0]}
        for row in rows
        if setup(row) == selected_setup
    ]
    models = model_records(selected, tasks)
    eng = {task.metadata.name for task in mteb.get_benchmark("MTEB(eng, v2)").tasks}
    nano = {task.metadata.name for task in mteb.get_benchmark("NanoBEIR").tasks}
    result = []
    for name, measured in sorted(models.items()):
        source = next(iter(measured.values()), {})
        base = models.get(baseline_name(name), {})
        deferred = sorted(
            {
                row["task"]
                for row in selected
                if row.get("model") == name and row.get("status") == "deferred"
            }
        )
        artifact = checked_artifact(name, source, artifacts)
        result.append(
            {
                "model": name,
                "source_model": source.get("source_model"),
                "backend_runtime": {
                    key: source.get(key)
                    for key in (
                        "runtime_package",
                        "runtime_package_version",
                        "package_location",
                        "runtime_fingerprint",
                        "runtime_fingerprint_scope",
                    )
                },
                "source_revision": source.get("source_revision"),
                "encoder_fingerprint": source.get("encoder_fingerprint"),
                "artifact": artifact,
                "english_completed": len(eng & measured.keys()),
                "english_required": len(eng),
                "nanobeir_completed": len(nano & measured.keys()),
                "nanobeir_required": len(nano),
                "deferred": deferred,
                "english_full_41_mean": mean_complete(measured, eng),
                "english_40_mean_mind_deferred": mean_complete(
                    measured, eng - {"MindSmallReranking"}
                )
                if deferred == ["MindSmallReranking"]
                else None,
                "nanobeir_full_13_mean": mean_complete(measured, nano),
                "tasks": task_deltas(measured, base),
            }
        )
    return {
        "setup": json.loads(selected_setup),
        "models": result,
        "score_units": "native MTEB score, deltas multiplied by 100 for points",
        "warning": "40-task English means exclude explicitly deferred MIND and are not the full benchmark",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method", help="select a method fingerprint if results contain several setups"
    )
    parser.add_argument("--runtime", help="select the common fse reference runtime fingerprint")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "baseline/dr151_model_artifacts_20261101.json",
    )
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines()]
    selected = [
        row
        for row in rows
        if row.get("method_fingerprint")
        and (not args.method or row["method_fingerprint"] == args.method)
        and (
            not args.runtime
            or row.get("fse_reference_runtime_fingerprint", row["runtime_fingerprint"])
            == args.runtime
        )
    ]
    setups = {setup(row) for row in selected}
    if len(setups) != 1:
        parser.error(
            f"need exactly one compatible setup, found {len(setups)}; select --method/--runtime"
        )
    artifacts = {
        row["artifact"]: row for row in json.loads(args.artifacts.read_text())["artifacts"]
    }
    output = summarize(selected, setups.pop(), artifacts)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
