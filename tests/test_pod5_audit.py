"""Collection integrity never substitutes for complete, comparable quality results."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/quality"))
    return importlib.import_module("audit_pod5")


def manifest(audit, root, entries):
    path = root / "transfer-manifest.json"
    path.write_text(json.dumps({"returncode": 124, "complete": False, "files": entries}))
    return audit.digest(path)


def entry(audit, path):
    return {"path": path.name, "sha256": audit.digest(path), "bytes": path.stat().st_size}


def test_intact_partial_transfer_is_not_a_complete_experiment(audit, tmp_path):
    receipt = tmp_path / "bootstrap-invocation.json"
    receipt.write_text(json.dumps({"returncode": 124}))
    expected = manifest(audit, tmp_path, [entry(audit, receipt)])
    result = audit.audit(tmp_path, expected)
    assert result["transfer_verified"] is True
    assert result["experiment_complete"] is False
    assert "failed or timed out" in result["errors"][0]


def test_remote_manifest_hash_is_required(audit, tmp_path):
    manifest(audit, tmp_path, [])
    assert "coordinator's remote SHA256" in audit.transfer_errors(tmp_path, "0" * 64)[0]


@pytest.mark.parametrize("path", ["../outside", "/absolute", "file"])
def test_unsafe_or_duplicate_transfer_entry_is_rejected(audit, tmp_path, path):
    row = {"path": path, "bytes": 0, "sha256": "0" * 64}
    expected = manifest(audit, tmp_path, [row, row])
    assert any(
        "unsafe or duplicate" in error for error in audit.transfer_errors(tmp_path, expected)
    )


def test_transfer_reports_all_missing_or_corrupt_paths(audit, tmp_path):
    file = tmp_path / "score.jsonl"
    file.write_text("score")
    row = entry(audit, file)
    missing = {"path": "missing.json", "bytes": 0, "sha256": "0" * 64}
    expected = manifest(audit, tmp_path, [row, missing])
    file.write_text("altered")
    errors = audit.transfer_errors(tmp_path, expected)
    assert len(errors) == 2
    assert "mismatch" in errors[0]
    assert "missing" in errors[1]


def test_historical_reference_covers_required_suites(audit):
    tasks = audit.expected_tasks()
    assert len(tasks) == 53
    assert sum(value == "MTEB(eng, v2)" for value in tasks.values()) == 40
    assert tasks.keys() >= audit.STS
    assert "MindSmallReranking" not in tasks


@pytest.fixture
def row_fixture(audit):
    historical = audit.document(audit.BASELINE, "dr151_quality_glove_20261101.json")
    source = audit.SOURCES[0]
    artifact = {"encoder_fingerprint": "test-fingerprint"}
    row = {
        "model": f"fse/SIF-10-{source}-float32",
        "task": "STSBenchmark",
        "benchmark": "MTEB(eng, v2)",
        "status": "completed",
        "source_model": f"fse/{source}",
        "source_revision": audit.ALL_SOURCES[source],
        "table_dtype": "float32",
        "workers": 2,
        "device": "cpu",
        "offline_recipe": "SIF-10",
        "mteb": "2.26.25",
        "main_score": 0.5,
        "method_fingerprint": historical["setup"]["method_fingerprint"],
        "suite_fingerprint": historical["setup"]["suite_fingerprint"],
        "encoder_fingerprint": artifact["encoder_fingerprint"],
        "float32_parent_fingerprint": artifact["encoder_fingerprint"],
        "fit_provenance": historical["models"][0]["artifact"]["preparation_record"]["fit"],
    }
    return row, source, artifact, historical


def test_matching_historical_row_is_accepted(audit, row_fixture):
    row, source, artifact, historical = row_fixture
    assert (
        audit.checked_row(row, source, "STSBenchmark", "MTEB(eng, v2)", artifact, historical) == 0.5
    )


@pytest.mark.parametrize(
    "change",
    [
        {"main_score": float("nan")},
        {"main_score": True},
        {"workers": 16},
        {"timing_proxy": True},
        {"source_revision": "other"},
    ],
)
def test_noncomparable_or_invalid_row_is_rejected(audit, row_fixture, change):
    row, source, artifact, historical = row_fixture
    with pytest.raises(ValueError, match=r"protocol differs|finite numeric"):
        audit.checked_row(
            {**row, **change}, source, "STSBenchmark", "MTEB(eng, v2)", artifact, historical
        )


def alias_fixture(audit, template, historical, tmp_path, monkeypatch):
    from fse._container._hash import Hasher

    corpus_path = tmp_path / "historic-corpus.txt"
    corpus_path.write_text("one  two\nthree four\n")
    raw, canonical = Hasher(), Hasher()
    raw.update(b"one  two\nthree four\n")
    canonical.update(b"one two\nthree four\n")
    template["fit_provenance"] = {
        **template["fit_provenance"],
        "paragraphs": 2,
        "tokenized_corpus_xxh3": canonical.checksum(),
    }
    historical["models"][0]["artifact"]["preparation_record"]["fit"] = {
        **template["fit_provenance"],
        "tokenized_corpus_xxh3": raw.checksum(),
    }
    original_document = audit.document

    def document(root, name):
        if root == audit.BASELINE and name == "dr151_quality_glove_20261101.json":
            return historical
        return original_document(root, name)

    monkeypatch.setattr(audit, "document", document)
    monkeypatch.setattr(audit, "FIT_LINES", 2)
    return corpus_path


@pytest.mark.parametrize("alias", [False, True])
def test_full_collection_requires_every_score(audit, row_fixture, tmp_path, monkeypatch, alias):
    template, _, _, historical = row_fixture
    corpus_path = (
        alias_fixture(audit, template, historical, tmp_path, monkeypatch) if alias else None
    )
    tasks, artifacts, invocations = audit.expected_tasks(), [], []
    for source in audit.SOURCES:
        name = f"SIF-10-{source}-float32.safetensors"
        artifacts.append(
            {
                "artifact": name,
                "encoder_fingerprint": "test-fingerprint",
                "preparation_record": {"fit": template["fit_provenance"]},
            }
        )
        for task, benchmark in tasks.items():
            row = {
                **template,
                "model": f"fse/SIF-10-{source}-float32",
                "source_model": f"fse/{source}",
                "source_revision": audit.ALL_SOURCES[source],
                "task": task,
                "benchmark": benchmark,
                "fit_provenance": {
                    **template["fit_provenance"],
                    "fasttext": "fse from_gensim vocabulary rows and native OOV ngrams, max_ngrams=40"
                    if source.startswith("fasttext-")
                    else None,
                },
            }
            (tmp_path / f"{source}-{task}.jsonl").write_text(json.dumps(row) + "\n")
        for stage in ["prepare-" + source, *(source + "-" + task for task in tasks)]:
            (tmp_path / (stage + ".rss.txt")).write_text("fixture RSS record\n")
            invocations.append({"stage": stage, "status": "completed", "returncode": 0})
        (tmp_path / f"{source}-excluded.json").write_text(
            json.dumps({"reason": "not computed: needs more memory", "main_score": None})
        )
        (tmp_path / f"SIF-10-{source}-fit.npz").write_bytes(b"fixture fitted state")
    for name, value in {
        "bootstrap-invocation.json": {"returncode": 0, "pinned_commit": audit.PIN},
        "complete.json": {"complete": True},
        "artifact-identities.json": {"artifacts": artifacts},
        "container-memory.json": {
            "memory.max": 64 * 2**30,
            "memory.current": 2**30,
            "memory.peak": 2 * 2**30,
        },
    }.items():
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "invocations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in invocations)
    )
    (tmp_path / "cache-cleanup.jsonl").write_text("fixture cleanup record\n")
    receipt = {
        "complete": True,
        "returncode": 0,
        "files": [entry(audit, path) for path in tmp_path.iterdir()],
    }
    path = tmp_path / "transfer-manifest.json"
    path.write_text(json.dumps(receipt))
    expected = audit.digest(path)
    result = audit.audit(tmp_path, expected, corpus_path)
    assert result["experiment_complete"] is True
    assert len(result["models"]) == 3
    assert all(model["english_full_41_mean"] is None for model in result["models"])
    if alias:
        historical["models"][0]["artifact"]["preparation_record"]["fit"]["a"] = 0.5
        rejected = audit.audit(tmp_path, expected, corpus_path)
        assert not rejected["experiment_complete"]
        assert "fit differs" in rejected["errors"][0]
    missing = tmp_path / f"{audit.SOURCES[-1]}-STSBenchmark.jsonl"
    missing.rename(tmp_path / "withheld-score")
    result = audit.audit(tmp_path, expected)
    assert result["transfer_verified"] is False
    assert result["experiment_complete"] is False
