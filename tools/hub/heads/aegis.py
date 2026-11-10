"""Pinned Aegis prompt preparation, without loading refusal or response targets."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import unicodedata
import urllib.request
from collections import defaultdict

import numpy as np

SPLITS = ("train", "validation", "test")


def normalize(text):
    """prompt-normalize@1: NFC, collapse whitespace, trim, casefold."""
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sources(work, config):
    raw = work / "raw"
    raw.mkdir(exist_ok=True)
    for name, spec in config["sources"].items():
        path = raw / name
        if not path.exists():
            with urllib.request.urlopen(spec["url"], timeout=120) as response:
                path.write_bytes(response.read())
        if sha256(path) != spec["sha256"]:
            raise ValueError("source checksum mismatch: " + name)


def clean(rows, aart):
    counts = {"raw": len(rows)}
    if any(row["prompt_label_source"] != "human" for row in rows):
        raise ValueError("nonhuman prompt label")
    rows = [
        row
        for row in rows
        if not row["reconstruction_id_if_redacted"] and row["prompt"] != "REDACTED"
    ]
    counts["after_redacted"] = len(rows)
    rows = [row for row in rows if normalize(row["prompt"])]
    counts["after_empty"] = len(rows)
    result = [clean_row(row, aart) for row in rows]
    counts["exact_aart_before_leakage"] = sum(r["source_aart"] for r in result)
    counts["prompt_only_before_leakage"] = sum(not r["has_response"] for r in result)
    return result, counts


def clean_row(row, aart):
    categories = sorted({c.strip() for c in row["violated_categories"].split(",") if c.strip()})
    has_response = bool((row["response"] or "").strip())
    unsafe = row["prompt_label"] == "unsafe"
    if row["prompt_label"] not in ("safe", "unsafe"):
        raise ValueError("unknown prompt label")
    if unsafe and not categories:
        raise ValueError("unsafe row without categories")
    if has_response and bool(categories) != unsafe:
        raise ValueError("response-bearing row violates annotation structure")
    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "normalized": normalize(row["prompt"]),
        "unsafe": unsafe,
        "categories": categories,
        "has_response": has_response,
        "source_aart": row["prompt"] in aart,
    }


def deduplicate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["normalized"]].append(row)
    result, conflicts, removed, merged = [], 0, 0, 0
    for group in groups.values():
        if len({r["unsafe"] for r in group}) > 1:
            conflicts += 1
            removed += len(group)
            continue
        row = dict(min(group, key=lambda r: r["id"]))
        row["categories"] = sorted({c for r in group for c in r["categories"]})
        row["has_response"] = any(r["has_response"] for r in group)
        row["source_aart"] = any(r["source_aart"] for r in group)
        merged += len(group) - 1
        result.append(row)
    return sorted(result, key=lambda r: r["id"]), {
        "duplicate_groups": sum(len(g) > 1 for g in groups.values()),
        "contradictory_groups": conflicts,
        "contradictory_rows_removed": removed,
        "same_label_rows_merged": merged,
    }


def summary(rows, config):
    ids = [r["id"] for r in rows]
    return {
        "rows": len(rows),
        "unsafe": sum(r["unsafe"] for r in rows),
        "prompt_only": sum(not r["has_response"] for r in rows),
        "exact_aart": sum(r["source_aart"] for r in rows),
        "normalized_groups": len({r["normalized"] for r in rows}),
        "ordered_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "positive": {
            label: sum(r["unsafe"] and category in r["categories"] for r in rows)
            for label, category in config["categories"].items()
        },
    }


def prepare(work, config):
    sources(work, config)
    with (work / "raw/aart.csv").open() as handle:
        aart = {r["prompt"] for r in csv.DictReader(handle)}
    data, report = {}, {}
    for split in SPLITS:
        raw = json.loads((work / "raw" / (split + ".json")).read_text())
        data[split], report[split] = clean(raw, aart)
    test = {r["normalized"] for r in data["test"]}
    held = test | {r["normalized"] for r in data["validation"]}
    for split, excluded in (("train", held), ("validation", test)):
        before = len(data[split])
        data[split] = [r for r in data[split] if r["normalized"] not in excluded]
        report[split]["leakage_rows_removed"] = before - len(data[split])
        report[split]["after_leakage"] = len(data[split])
    data["train"], report["train"]["deduplication"] = deduplicate(data["train"])
    for split in SPLITS:
        write(work / (split + "-prepared.json"), data[split])
        report[split]["final"] = summary(data[split], config)
    candidates = [r for r in data["train"] if not r["has_response"] and r["categories"]]
    chosen = random.Random(config["seed"]).sample(candidates, 20)
    write(work / "sample-private.json", chosen)
    report["sample"] = {"seed": config["seed"], "ids": [r["id"] for r in chosen]}
    report["test_access"] = (
        "Planner inspected test counts and structure before training. Preparation reads test "
        "for frozen source filters and leakage exclusion only. No test-driven selection."
    )
    report["normalization"] = config["normalization"]
    write(work / "data-report.json", report)
    return report


def dataset(work, config, split):
    if split not in SPLITS:
        raise ValueError("unknown split")
    rows = json.loads((work / (split + "-prepared.json")).read_text())
    y = np.array([r["unsafe"] for r in rows], dtype=np.int64)
    categories = np.array(
        [
            [r["unsafe"] and c in r["categories"] for c in config["categories"].values()]
            for r in rows
        ],
        dtype=np.int64,
    )
    return rows, {"safety": y, "categories": categories}
