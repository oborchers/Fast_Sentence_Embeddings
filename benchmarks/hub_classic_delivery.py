"""The complete delivery graph: source, selection, fit, models and measurements."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from hub_classic_evidence import digest, validate_build
from hub_classic_sources import DTYPES, SOURCES
from hub_classic_verification import repo_name, verification_rows

CARD_RECIPE = "fse-classic-card@1"
FIT_IDENTITY = ("corpus", "revision", "seed", "paragraphs", "corpus_xxh3", "wordfreq_version")


def read_json(path):
    return json.loads(Path(path).read_text())


def validated_source(models, name, selection_path=None, enriched=False):
    verification = read_json(models / f"{name}-verification.json")
    rows = verification_rows(verification, name)
    fit = read_json(models / f"{name}-fit.json")
    if fit.get("source_repo") != f"fse/{name}" or fit.get("source_revision") != SOURCES[name]:
        raise ValueError("source fit provenance mismatch")
    if selection_path is not None:
        validate_selection(fit, selection_path)
    records = []
    for method, dtype in product(("raw", "sif10"), DTYPES):
        repo = repo_name(name, method, dtype)
        folder = models / repo.split("/")[1]
        record = read_json(models / f"{folder.name}.json")
        _record_identity(record, name, method, dtype, repo)
        validate_build(folder, record, rows[(method, dtype)], fit if method == "sif10" else None)
        if enriched:
            _enriched(folder, record, fit if method == "sif10" else None)
        records.append(record)
    _parents(records)
    return records, verification, fit


def validate_selection(fit, selection_path):
    selection = read_json(selection_path)
    if fit.get("a") != 0.001 or fit.get("components") != 10 or fit.get("paragraphs") != 859346:
        raise ValueError("model fit differs from the required SIF recipe")
    if fit.get("selection_xxh3") != digest(selection_path):
        raise ValueError("model fit does not reference this corpus selection")
    if any(fit.get(key) != selection["fit"].get(key) for key in FIT_IDENTITY):
        raise ValueError("model fit corpus differs from the selected corpus")


def _record_identity(record, name, method, dtype, repo):
    expected = {
        "source": f"fse/{name}",
        "revision": SOURCES[name],
        "method": method,
        "dtype": dtype,
        "repo": repo,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("model build manifest identity mismatch")


def _enriched(folder, record, fit):
    if record.get("card_recipe") != CARD_RECIPE:
        raise ValueError("model card enrichment is missing")
    if f"<!-- {CARD_RECIPE} -->" not in (folder / "README.md").read_text():
        raise ValueError("model card recipe marker is missing")
    if fit is not None and read_json(folder / "preparation.json") != fit:
        raise ValueError("published preparation receipt differs from the validated fit")


def validate_previews(records, cards):
    expected = {record["repo"].split("/")[1] + ".md" for record in records}
    if {p.name for p in cards.glob("*.md")} != expected:
        raise ValueError("card preview inventory is incomplete or unexpected")
    for record in records:
        path = cards / (record["repo"].split("/")[1] + ".md")
        spec = record["files"]["README.md"]
        if path.stat().st_size != spec["bytes"] or digest(path) != spec["xxh3"]:
            raise ValueError("card preview differs from the published model card")


def _parents(records):
    parents = {row["method"]: row for row in records if row["dtype"] == "float32"}
    for row in records:
        parent = parents[row["method"]]
        if (
            row["float32_fingerprint"] != parent["fingerprint"]
            or row["float32_table_fingerprint"] != parent["float32_table_fingerprint"]
        ):
            raise ValueError("dtype variants do not share their recorded float32 model")
