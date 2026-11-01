"""Document diagnostics retain complete-task scores and disclose upstream lengths."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def reporter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/quality"))
    return importlib.import_module("report_document_tasks")


def inputs(tmp_path):
    meta = {
        "name": "task",
        "type": "Retrieval",
        "eval_splits": ["test"],
        "dataset": {"revision": "pin"},
        "main_score": "ndcg_at_10",
    }
    setup = {"method_fingerprint": "method", "python": "3.13.16"}
    score = {"task": "task", "metric": "ndcg_at_10", "score": 0.4, "delta_vs_float32_points": None}
    model = {"model": "example", "source_revision": "pin", "tasks": [score]}
    raw = {
        **setup,
        "model": "example",
        "status": "completed",
        "task": "task",
        "task_metadata": meta,
        "source_revision": "pin",
        "encoder_description": "native limits",
    }
    folder = tmp_path / "Retrieval"
    folder.mkdir()
    (folder / "task.json").write_text(
        json.dumps(
            {
                "test": {
                    "documents_text_statistics": {
                        "average_text_length": 2000,
                        "max_text_length": 100000,
                    }
                },
                "train": {"text_statistics": {"average_text_length": 9000}},
            }
        )
    )
    return {"setup": setup, "models": [model]}, [raw]


def test_report_uses_evaluation_lengths_and_complete_task_metric(reporter, tmp_path):
    document, rows = inputs(tmp_path)
    result = reporter.summarize(document, rows, tmp_path, 1000)
    task = result["tasks"][0]
    assert task["input_streams"] == [
        {
            "stream": "test/documents_text_statistics",
            "average_text_length": 2000,
            "max_text_length": 100000,
        }
    ]
    assert task["scores"][0]["score"] == 0.4
    assert task["scores"][0]["metric"] == "ndcg_at_10"
    assert (
        result["length_statistics_origin"] == "packaged MTEB statistics, not re-measured run inputs"
    )


def test_foreign_configuration_cannot_supply_task_metadata(reporter, tmp_path):
    document, rows = inputs(tmp_path)
    rows[0]["method_fingerprint"] = "other"
    with pytest.raises(ValueError, match="no raw task records match"):
        reporter.summarize(document, rows, tmp_path, 1000)


def test_raw_source_and_report_identity_must_match(reporter, tmp_path):
    document, rows = inputs(tmp_path)
    rows[0]["source_revision"] = "other"
    with pytest.raises(ValueError, match="raw result differs"):
        reporter.summarize(document, rows, tmp_path, 1000)


def test_missing_model_score_stays_pending(reporter, tmp_path):
    document, rows = inputs(tmp_path)
    document["models"].append({"model": "waiting", "tasks": []})
    scores = reporter.summarize(document, rows, tmp_path, 1000)["tasks"][0]["scores"]
    assert scores[1]["status"] == "pending"
    assert scores[1]["score"] is None


def test_multilingual_aggregate_does_not_supply_unrelated_subset_lengths(reporter):
    packaged = {
        "test": {
            "text_statistics": {"average_text_length": 9000},
            "hf_subset_descriptive_stats": {
                "en": {"text_statistics": {"average_text_length": 2000}},
                "de": {"text_statistics": {"average_text_length": 8000}},
            },
        }
    }
    meta = {"eval_splits": ["test"], "eval_langs": {"en": ["eng-Latn"], "de": ["deu-Latn"]}}
    assert reporter.evaluation_streams(packaged, meta) == [
        {
            "stream": "test/hf_subset_descriptive_stats/en/text_statistics",
            "average_text_length": 2000,
        }
    ]
