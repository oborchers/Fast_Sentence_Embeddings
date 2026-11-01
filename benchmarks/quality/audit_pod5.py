"""Audit collected Pod 5 evidence locally, without loading models or running MTEB.

Transfer integrity and experiment completion are separate: a failed experiment can
be safely collected. The coordinator supplies the remote manifest's SHA256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from classic_sources import ALL_SOURCES, FIT_LINES, FIT_REVISION, stem

SOURCES = ("word2vec-google-news-300", "fasttext-crawl-subwords-300", "paragram-300-sl999")
STS = {"BIOSSES", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS17", "STS22.v2", "STSBenchmark"}
BASELINE = Path(__file__).resolve().parents[1] / "baseline"
PIN = "b0bdc3d08544fdecfadeffc8ca5dae303ffd898d"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def document(root, name):
    return json.loads((root / name).read_text())


def transfer_errors(root, expected):
    manifest_path = root / "transfer-manifest.json"
    if digest(manifest_path) != expected:
        return ["transfer-manifest.json: differs from coordinator's remote SHA256"]
    manifest = document(root, manifest_path.name)
    errors, seen = [], set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or entry["path"] in seen
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            errors.append(f"{relative}: unsafe or duplicate manifest path")
            continue
        seen.add(entry["path"])
        if not path.is_file():
            errors.append(f"{relative}: missing")
        elif path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            errors.append(f"{relative}: size or SHA256 mismatch")
    if not seen:
        errors.append("transfer manifest contains no files")
    return errors


def expected_tasks():
    baseline = document(BASELINE, "dr151_quality_paranmt_20261101.json")
    tasks = {row["task"]: row["benchmark"] for row in baseline["models"][0]["tasks"]}
    if len(tasks) != 53 or sum(value == "NanoBEIR" for value in tasks.values()) != 13:
        raise ValueError("committed historical reference does not contain all 53 tasks")
    return tasks


def checked_row(row, source, task, benchmark, artifact, historical):
    fit = row["fit_provenance"]
    expected = {
        "model": f"fse/{stem(source)}-float32",
        "task": task,
        "benchmark": benchmark,
        "status": "completed",
        "source_model": f"fse/{source}",
        "source_revision": ALL_SOURCES[source],
        "table_dtype": "float32",
        "workers": 2,
        "device": "cpu",
        "offline_recipe": "SIF-10",
        "mteb": "2.26.25",
        "method_fingerprint": historical["setup"]["method_fingerprint"],
        "suite_fingerprint": historical["setup"]["suite_fingerprint"],
        "encoder_fingerprint": artifact["encoder_fingerprint"],
        "float32_parent_fingerprint": artifact["encoder_fingerprint"],
    }
    if any(row.get(key) != value for key, value in expected.items()) or row.get("timing_proxy"):
        raise ValueError("task identity or historical evaluation protocol differs")
    score = row["main_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("task score is not finite numeric data")
    reference = historical["models"][0]["artifact"]["preparation_record"]["fit"]
    keys = (
        "corpus",
        "revision",
        "split",
        "paragraphs",
        "tokenized_corpus_xxh3",
        "seed",
        "components",
        "a",
        "solver",
        "gram",
        "tokenizer",
    )
    if any(fit.get(key) != reference[key] for key in keys):
        raise ValueError("fit differs from historical GloVe recipe or tokenized training corpus")
    if fit["revision"] != FIT_REVISION or fit["paragraphs"] != FIT_LINES:
        raise ValueError("unexpected training corpus pin")
    oov = "fse from_gensim vocabulary rows and native OOV ngrams, max_ngrams=40"
    if fit.get("fasttext") != (oov if source.startswith("fasttext-") else None):
        raise ValueError("FastText OOV provenance differs from the historical recipe")
    return score


def model_summary(root, source, tasks, artifacts, historical):
    artifact = next(
        entry for entry in artifacts if entry["artifact"] == f"{stem(source)}-float32.safetensors"
    )
    scores = {}
    for task, benchmark in tasks.items():
        path = root / f"{source}-{task}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 1:
            raise ValueError(f"{path.name}: expected exactly one result")
        scores[task] = checked_row(rows[0], source, task, benchmark, artifact, historical)
    excluded = document(root, f"{source}-excluded.json")
    if (
        excluded.get("reason") != "not computed: needs more memory"
        or excluded.get("main_score") is not None
    ):
        raise ValueError("MIND exclusion is missing or claims a score")
    if not (root / f"{stem(source)}-fit.npz").is_file():
        raise ValueError("fitted state is missing")
    return {
        "source": source,
        "english_completed": 40,
        "english_required": 41,
        "english_full_41_mean": None,
        "english_40_mean_mind_not_computed": statistics.mean(
            scores[t] for t in tasks if tasks[t] == "MTEB(eng, v2)"
        ),
        "nanobeir_full_13_mean": statistics.mean(
            scores[t] for t in tasks if tasks[t] == "NanoBEIR"
        ),
        "STSBenchmark": scores["STSBenchmark"],
        "sts_tasks": {task: scores[task] for task in sorted(STS)},
        "tasks": scores,
    }


def checked_invocations(root, tasks):
    rows = [json.loads(line) for line in (root / "invocations.jsonl").read_text().splitlines()]
    expected = {source + "-" + task for source in SOURCES for task in tasks}
    expected.update("prepare-" + source for source in SOURCES)
    if len(rows) != len(expected) or {row["stage"] for row in rows} != expected:
        raise ValueError("invocation receipts do not cover all 162 stages exactly once")
    for row in rows:
        if row["status"] != "completed" or row["returncode"] != 0:
            raise ValueError(f"{row['stage']}: failed invocation")
        if not (root / (row["stage"] + ".rss.txt")).stat().st_size:
            raise ValueError(f"{row['stage']}: missing RSS evidence")
    memory = document(root, "container-memory.json")
    if not all(
        type(memory.get(key)) is int and memory[key] > 0
        for key in ("memory.max", "memory.current", "memory.peak")
    ):
        raise ValueError("container memory evidence is missing")
    if not (root / "cache-cleanup.jsonl").is_file():
        raise ValueError("owned cache cleanup receipts are missing")


def audit(root, expected_manifest):
    try:
        errors = transfer_errors(root, expected_manifest)
    except (KeyError, ValueError, OSError) as error:
        errors = [str(error)]
    result = {
        "transfer_verified": not errors,
        "experiment_complete": False,
        "errors": errors,
        "models": [],
    }
    if errors:
        return result
    try:
        transfer = document(root, "transfer-manifest.json")
        bootstrap = document(root, "bootstrap-invocation.json")
        if transfer["returncode"] != 0 or not transfer["complete"] or bootstrap["returncode"] != 0:
            raise ValueError(
                "experiment failed or timed out; transferred partial evidence is preserved"
            )
        if (
            bootstrap["pinned_commit"] != PIN
            or document(root, "complete.json")["complete"] is not True
        ):
            raise ValueError("source pin or completion receipt differs")
        artifacts = document(root, "artifact-identities.json")["artifacts"]
        historical = document(BASELINE, "dr151_quality_glove_20261101.json")
        tasks = expected_tasks()
        checked_invocations(root, tasks)
        result["models"] = [
            model_summary(root, source, tasks, artifacts, historical) for source in SOURCES
        ]
        result["experiment_complete"] = True
    except (KeyError, ValueError, OSError, StopIteration) as error:
        result["errors"].append(str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder, args.manifest_sha256)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "models"}), flush=True)
    raise SystemExit(0 if result["experiment_complete"] else 2)


if __name__ == "__main__":
    main()
