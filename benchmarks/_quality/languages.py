"""Private multilingual summaries: coverage, language tables, zero-shot and reproduction.

Scores stay on the unscaled MTEB main_score scale. A mean exists only where every planned
task of a selection is valid; a benchmark mean only where the selection is the whole
benchmark. Language columns count their tasks; ``language_view`` of MTEB is display only,
so every evaluated subset enters these tables."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from _measurement import sha256

MULTILINGUAL = ("MTEB(Multilingual, v2)", "NanoBEIR-multilingual")
FULL = {"MTEB(Multilingual, v2)": 131, "NanoBEIR-multilingual": 13}
STATUSES = ("valid", "failed", "aborted", "refused", "not_started")


def planned(row):
    filters = row["evaluation_filters"]
    return (row["task"], tuple(filters["hf_subsets"]), tuple(filters["eval_splits"]))


def coverage(rows):
    """Planned against valid tasks of one configuration and benchmark, as sets of
    (task, subsets, splits). Failed, aborted, refused and not started stay apart."""
    plan = {planned(r) for r in rows}
    valid = {planned(r) for r in rows if r["status"] == "valid"}
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES}
    complete = valid == plan and counts["valid"] == len(plan)
    return {"planned": len(plan), **counts, "complete_selection": complete}


def language_scores(folder, row):
    """Per evaluated subset: its languages and main_score, from the raw MTEB result."""
    raw = Path(folder) / (row["cell_id"] + "-mteb.json")
    if sha256(raw) != row["raw_result_sha256"]:
        raise ValueError("raw MTEB result changed after the run")
    result = json.loads(raw.read_text())
    for split, entries in result["scores"].items():
        for entry in entries:
            yield split, entry["hf_subset"], tuple(entry["languages"]), entry["main_score"]


def language_table(folder, rows):
    """Mean per language (same-language subsets) and per language pair (subsets whose
    languages differ), first within a task, then over the tasks that hold it."""
    same, pairs = {}, {}
    for row in rows:
        by_task_same, by_task_pair = {}, {}
        for _split, _subset, languages, score in language_scores(folder, row):
            distinct = list(dict.fromkeys(languages))
            target = by_task_same if len(distinct) == 1 else by_task_pair
            target.setdefault(" | ".join(distinct), []).append(score)
        for target, values in ((same, by_task_same), (pairs, by_task_pair)):
            for language, scores in values.items():
                target.setdefault(language, []).append(statistics.mean(scores))

    def table(values):
        return {
            k: {"tasks": len(v), "of_tasks": len(rows), "mean": statistics.mean(v)}
            for k, v in sorted(values.items())
        }

    return {"languages": table(same), "cross_lingual": table(pairs)}


def benchmark_summary(folder, rows):
    cover = coverage(rows)
    valid = [r for r in rows if r["status"] == "valid"]
    benchmark = rows[0]["benchmark"]
    cover["full_benchmark"] = cover["complete_selection"] and cover["planned"] == FULL.get(
        benchmark
    )
    scores = {r["task"]: r["main_score"] for r in valid}
    summary = {"benchmark": benchmark, "coverage": cover, "scores": scores}
    if not cover["complete_selection"]:
        return {**summary, "selection_mean": None, "benchmark_mean": None}
    types = {}
    for r in valid:
        types.setdefault(r["task_metadata"]["type"], []).append(r["main_score"])
    mean = statistics.mean(scores.values())
    return {
        **summary,
        "selection_mean": mean,
        "selection_mean_type": statistics.mean(statistics.mean(v) for v in types.values()),
        "types": {
            k: {"tasks": len(v), "mean": statistics.mean(v)} for k, v in sorted(types.items())
        },
        "benchmark_mean": mean if cover["full_benchmark"] else None,
        **language_table(folder, valid),
        "training_relation": {r["task"]: r["training_relation"]["state"] for r in valid},
    }


def zero_shot(configurations):
    """Per benchmark: the tasks that every configuration holding that benchmark evaluated
    validly and MTEB confirms as zero-shot for its model, and each configuration's mean over
    exactly that common set. A configuration with an incomplete selection empties it."""
    result = {}
    for benchmark in MULTILINGUAL:
        held = holders(configurations, benchmark)
        if held:
            result[benchmark] = common_set(held)
    return result


def holders(configurations, benchmark):
    return [
        (index, b)
        for index, c in enumerate(configurations)
        for b in c["benchmarks"]
        if b["benchmark"] == benchmark
    ]


def common_set(held):
    common = sorted(set.intersection(*(confirmed(b) for _, b in held)))
    means = [statistics.mean(b["scores"][t] for t in common) for _, b in held] if common else []
    return {
        "configurations": [index for index, _ in held],
        "common_zero_shot_tasks": common,
        "means": means or [None] * len(held),
    }


def confirmed(benchmark):
    return {t for t, s in benchmark.get("training_relation", {}).items() if s == "zero_shot"}


def reproduction(folder, rows):
    """Per task: the largest absolute main_score difference against a published MTEB result
    in ``references/<org>__<model>/<revision>/<Task>.json``, over the shared subsets. The
    published revision may differ from the pin and is reported with the result."""
    found = []
    for row in rows:
        path = reference_of(folder, row) if row["status"] == "valid" else None
        if path is None:
            continue
        published = {
            (split, e["hf_subset"]): e["main_score"]
            for split, entries in json.loads(path.read_text())["scores"].items()
            for e in entries
        }
        ours = {(s, h): v for s, h, _, v in language_scores(folder, row)}
        shared = sorted(set(ours) & set(published))
        found.append(
            {
                "cell_id": row["cell_id"],
                "reference": str(path.relative_to(folder)),
                "reference_sha256": sha256(path),
                "reference_revision": path.parent.name,
                "revision_is_pin": path.parent.name == row["model_revision"],
                "shared_subsets": len(shared),
                "missing_in_reference": len(set(ours) - set(published)),
                "max_abs_difference": max(
                    (abs(ours[k] - published[k]) for k in shared), default=None
                ),
            }
        )
    return found


def reference_of(folder, row):
    model = Path(folder) / "references" / row["model"].replace("/", "__")
    paths = sorted(model.glob("*/" + row["task"] + ".json"))
    if len(paths) > 1:
        raise ValueError(f"several published revisions for {row['cell_id']}")
    return paths[0] if paths else None


# What a cell configures: rows of every status share it, so coverage counts them together.
CONFIGURED = (
    "model",
    "backend",
    "workers",
    "dtype",
    "truncate",
    "truncate_dim",
    "trust_remote_code",
    "publishable",
    "class",
    "machine_environment_sha256",
)
# What a valid row observed: it must be the same for every task of one configuration.
OBSERVED = (
    "model_revision",
    "backend_identity",
    "mteb_version",
    "table_dtype",
    "output_dim",
    "execution",
    "model_files",
)


def identity(rows):
    seen = {json.dumps({k: r.get(k) for k in OBSERVED}, sort_keys=True) for r in rows}
    if len(seen) > 1:
        raise ValueError("one quality configuration observed different model identities")
    return json.loads(seen.pop()) if seen else None


def grouped(rows):
    """Multilingual rows by configuration and benchmark; rows of every status count."""
    groups = {}
    for row in rows:
        if row["kind"] != "mteb" or row.get("functional_only"):
            continue
        if row["benchmark"] in MULTILINGUAL:
            key = json.dumps({k: row.get(k) for k in CONFIGURED}, sort_keys=True)
            groups.setdefault(key, {}).setdefault(row["benchmark"], []).append(row)
    return groups


def multilingual(folder, rows):
    """Configurations with multilingual rows; rows of every status count toward coverage."""
    groups = grouped(rows)
    configurations = []
    for key, by in sorted(groups.items()):
        members = [r for v in by.values() for r in v]
        configurations.append(
            {
                **json.loads(key),
                "identity": identity([r for r in members if r["status"] == "valid"]),
                "benchmarks": [benchmark_summary(folder, r) for _, r in sorted(by.items())],
            }
        )
    every = [r for by in groups.values() for v in by.values() for r in v]
    return {
        "scale": "unscaled MTEB main_score (0 to 1)",
        "configurations": configurations,
        "zero_shot": zero_shot(configurations),
        "reproduction": reproduction(folder, every),
        "probes": probes(rows),
    }


def probes(rows):
    keys = ("cell_id", "model", "status", "reason", "reference", "reference_weight_dtype")
    keys += ("output_dim", "max_abs_difference", "min_cosine", "zero_vectors_fse")
    return [{k: r.get(k) for k in keys} for r in rows if r["kind"] == "probe"]
