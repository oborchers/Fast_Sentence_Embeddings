"""Committed evidence extraction preserves parent lineage and renderer source links."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def exporter(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "quality"))
    return importlib.import_module("export_readme_panels")


def source_fixture(module):
    row = dict(
        model=module.POTION,
        revision="rev",
        input_digest="texts",
        texts=5352,
        corpus="wikitext",
        machine="pod",
        time_utc="2026-11-01T12:00:00Z",
        requested_threads=16,
        table_dtype="int8",
        output_dtype="float32",
        shape=[5352, 256],
        artifact_provenance=dict(
            float32_parent_fingerprint="parent",
            provenance_verification="embedded source and preparation record",
        ),
    )
    checked = dict(normalize=True, encoder_header={key: "rule" for key in module.TOKENIZER_FIELDS})
    return row, checked


def test_quantized_context_uses_original_float32_parent(exporter):
    row, checked = source_fixture(exporter)
    context = exporter.context([row, dict(row)], checked)
    assert context["parent_artifact"] == "parent"
    assert context["table_dtype"] == "int8"
    assert context["output_dtype"] == "float32"
    assert context["workers"] == 16
    assert context["output_dim"] == 256
    assert context["normalize"] is True
    assert json.loads(context["tokenizer_rule"])["fse.unknown"] == "rule"


def test_changed_or_unverified_parent_rejected(exporter):
    row, checked = source_fixture(exporter)
    other = {
        **row,
        "artifact_provenance": {
            **row["artifact_provenance"],
            "float32_parent_fingerprint": "different",
        },
    }
    with pytest.raises(ValueError, match="changed artifact lineage"):
        exporter.context([row, other], checked)
    row["artifact_provenance"] = {}
    with pytest.raises(ValueError, match="verified Hub"):
        exporter.context([row], checked)


@pytest.mark.parametrize(
    ("commit", "path"),
    [
        ("main", "benchmarks/baseline/raw.jsonl"),
        ("a" * 40, "/tmp/raw"),
        ("a" * 40, "benchmarks/baseline/../../outside"),
    ],
)
def test_source_requires_immutable_repository_evidence(exporter, commit, path):
    with pytest.raises(ValueError, match="publication source"):
        exporter.committed_text(commit, path)


def test_output_preserves_paired_source_and_round_order(exporter):
    source = dict(commit="a" * 40, path="benchmarks/baseline/raw.jsonl", series_id="exact-series")
    row, checked = source_fixture(exporter)
    context = exporter.context([row], checked)
    summary = dict(
        verified_aa_ratio=1.02,
        arms={
            "fse-a": dict(
                runtime_stable=True,
                backend="fse",
                runtime=dict(runtime_fingerprint="fse-code"),
                ordered_texts_per_second=[3.0, 1.0, 2.0],
            )
        },
    )
    result = exporter.panel_row(summary, "fse-a", source, context)
    assert result["texts_per_second"] == [3.0, 1.0, 2.0]
    assert result["source"] == source
    assert result["aa_upper"] == 1.02
    assert result["context"]["parent_artifact"] == "parent"


@pytest.mark.parametrize("labels", [("fse-a",), ("fse-b",), ("fse-a", "model2vec")])
def test_extraction_requires_both_controls_even_if_journal_claims_complete(exporter, labels):
    cell = exporter.case_id(exporter.POTION, "float32", 1, None)
    done = dict(cell=cell, series_id="series", complete=True)
    raw = [
        dict(event="operation", label=label, kind="fse" if label.startswith("fse") else label)
        for label in labels
    ]
    with pytest.raises(ValueError, match="both fse controls"):
        exporter.extract_series(done, raw, {}, "a" * 40, "benchmarks/baseline/raw.jsonl")


def test_extraction_rejects_missing_reference_arm(exporter):
    cell = exporter.case_id(exporter.POTION, "float32", 1, "model2vec")
    raw = [dict(event="operation", label=label, kind="fse") for label in ("fse-a", "fse-b")]
    with pytest.raises(ValueError, match="exact reference backend"):
        exporter.extract_series(dict(cell=cell), raw, {}, "a" * 40, "benchmarks/baseline/raw.jsonl")


def test_context_date_comes_from_accepted_rounds(exporter, monkeypatch):
    base, checked = source_fixture(exporter)
    cell = exporter.case_id(exporter.POTION, "int8", 16, None)
    checked.update(model=exporter.POTION, table_dtype="int8", workers=16)
    raw = [
        dict(
            base,
            event="operation",
            label=label,
            kind="fse",
            round=number,
            time_utc="2026-11-01T23:59:59Z" if number == 0 else "2026-11-02T00:01:00Z",
        )
        for number in range(4)
        for label in ("fse-a", "fse-b")
    ]
    arms = {
        label: dict(
            runtime_stable=True,
            backend="fse",
            runtime=dict(runtime_fingerprint="code"),
            ordered_texts_per_second=[1.0, 2.0, 3.0],
        )
        for label in ("fse-a", "fse-b")
    }
    monkeypatch.setattr(
        exporter,
        "summarize",
        lambda *_args: dict(
            eligible=True, arms=arms, verified_aa_ratio=1.0, accepted_rounds=[1, 2, 3]
        ),
    )
    _, primary, _ = exporter.extract_series(
        dict(cell=cell, series_id="series"),
        raw,
        {cell: checked},
        "a" * 40,
        "benchmarks/baseline/raw.jsonl",
    )
    assert primary["context"]["date"] == "2026-11-02"
