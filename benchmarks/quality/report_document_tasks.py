"""Break out measured document-task quality using pinned upstream length metadata.

Length statistics come from the installed, unmodified MTEB package. They are
not fresh measurements of this run's inputs. Every full-task score stays in
the primary quality report. This diagnostic neither evaluates a subset nor
averages different task metrics into a new benchmark score.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from importlib.metadata import distribution
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_artifacts import file_digest  # isort: skip


LENGTH_FIELDS = (
    "min_text_length",
    "average_text_length",
    "max_text_length",
    "total_text_length",
    "unique_texts",
)


def compatible(row: dict, setup: dict) -> bool:
    for key, expected in setup.items():
        value = row.get(key)
        if key == "python":
            value = (value or "").split(" ", maxsplit=1)[0]
        if value != expected:
            return False
    return True


def text_streams(value: dict, prefix: str = "") -> list[dict]:
    streams = []
    for key, item in value.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, dict):
            if "average_text_length" in item:
                streams.append({"stream": path, **{k: item[k] for k in LENGTH_FIELDS if k in item}})
            else:
                streams.extend(text_streams(item, path))
    return streams


def identities(document: dict, rows: list[dict]) -> tuple[dict, dict]:
    models = {model["model"]: model for model in document["models"]}
    metadata, descriptions = {}, {}
    for row in rows:
        if not compatible(row, document["setup"]) or row.get("status") != "completed":
            continue
        name = row["model"]
        if name not in models:
            continue
        model = models[name]
        for key in ("source_model", "source_revision", "encoder_fingerprint"):
            if row.get(key) != model.get(key):
                raise ValueError(f"{name}: raw result differs from the selected quality identity")
        task = row["task"]
        found = row["task_metadata"]
        if task in metadata and metadata[task] != found:
            raise ValueError(f"{task}: incompatible task metadata")
        metadata[task] = found
        descriptions.setdefault(name, set()).add(row["encoder_description"])
    if not metadata:
        raise ValueError("no raw task records match the selected quality configuration")
    if any(len(values) != 1 for values in descriptions.values()):
        raise ValueError("one quality configuration has different encoder descriptions")
    return metadata, {name: next(iter(values)) for name, values in descriptions.items()}


def quality_scores(models: list[dict], task: str, metric: str) -> list[dict]:
    scores = []
    for model in models:
        matches = [row for row in model["tasks"] if row["task"] == task]
        if len(matches) > 1:
            raise ValueError(f"{task}: duplicate task score")
        if matches and matches[0]["metric"] != metric:
            raise ValueError(f"{task}: metric differs from task metadata")
        row = matches[0] if matches else {}
        scores.append(
            {
                "model": model["model"],
                "metric": metric,
                "score": row.get("score"),
                "delta_vs_float32_points": row.get("delta_vs_float32_points"),
                "status": "measured full task" if matches else "pending",
            }
        )
    return scores


def summarize(document: dict, rows: list[dict], root: Path, minimum: float) -> dict:
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("minimum mean character count must be positive")
    metadata, descriptions = identities(document, rows)
    tasks, unavailable = [], []
    for name, meta in sorted(metadata.items()):
        path = root / meta["type"] / f"{name}.json"
        if not path.is_file():
            unavailable.append(name)
            continue
        packaged = json.loads(path.read_text())
        streams = [
            stream
            for split in meta["eval_splits"]
            if split in packaged
            for stream in text_streams(packaged[split], split)
            if stream["average_text_length"] >= minimum
        ]
        if not streams:
            continue
        tasks.append(
            {
                "task": name,
                "type": meta["type"],
                "dataset": meta["dataset"],
                "length_metadata_file": str(path.relative_to(root)),
                "length_metadata_digest": file_digest(path),
                "input_streams": streams,
                "scores": quality_scores(document["models"], name, meta["main_score"]),
            }
        )
    return {
        "setup": document["setup"],
        "minimum_upstream_mean_characters": minimum,
        "length_statistics_origin": "packaged MTEB statistics, not re-measured run inputs",
        "selection": "every measured task with an upstream evaluation text stream meeting the threshold",
        "limitations": [
            "packaged statistics may cover additional subsets or precede the pinned task revision",
            "these means do not measure truncation frequency or model-specific token lengths",
            "scores use each complete task and each model's native input limits",
            "this is a diagnostic breakout, not a new aggregate benchmark or a sampled evaluation",
        ],
        "encoder_descriptions": descriptions,
        "tasks": tasks,
        "missing_packaged_statistics": unavailable,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--minimum-mean-characters", type=float, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.quality.read_text())
    package = distribution("mteb")
    if package.version != "2.26.25" or document["setup"]["versions"]["mteb"] != package.version:
        parser.error("use the matching unmodified pinned MTEB 2.26.25 environment")
    root = Path(package.locate_file("mteb/descriptive_stats"))
    rows = [json.loads(line) for line in args.raw.read_text().splitlines()]
    result = summarize(document, rows, root, args.minimum_mean_characters)
    result["mteb_version"] = package.version
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
