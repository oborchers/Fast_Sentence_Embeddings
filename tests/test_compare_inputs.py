"""Cached extra revisions must never change the comparison corpus."""

import gzip
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("compare_inputs")


def test_massive_uses_exactly_one_pinned_revision(inputs, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(inputs, "MASSIVE_COUNTS", {"en.json.gz": 1})
    repo, revision = inputs.DATASETS["massive"]
    base = tmp_path / ("datasets--" + repo.replace("/", "--")) / "snapshots"
    for rev, text in ((revision, "correct"), ("another-revision", "duplicate")):
        folder = base / rev / "test"
        folder.mkdir(parents=True)
        with gzip.open(folder / "en.json.gz", "wt") as stream:
            stream.write(json.dumps({"text": text}) + "\n")
    assert inputs.texts("massive") == ["correct"]


def test_digest_records_order_and_boundaries(inputs):
    def digest(rows):
        return inputs.corpus_identity("massive", rows)["input_digest"]

    assert digest(["a", "bc"]) != digest(["ab", "c"])
    assert digest(["a", "bc"]) != digest(["bc", "a"])


def test_non_macos_cpu_does_not_run_sysctl(inputs, monkeypatch):
    monkeypatch.setattr(inputs.sys, "platform", "linux")
    monkeypatch.setattr(inputs.platform, "processor", lambda: "")
    monkeypatch.setattr(inputs.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(inputs.subprocess, "run", lambda *_a, **_kw: pytest.fail("sysctl"))
    assert inputs.cpu_name() == "x86_64"


def test_massive_rejects_a_partial_language_cache(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(inputs, "MASSIVE_COUNTS", {"en.json.gz": 1, "de.json.gz": 1})
    folder = tmp_path / "test"
    folder.mkdir()
    with gzip.open(folder / "en.json.gz", "wt") as stream:
        stream.write(json.dumps({"text": "one language only"}) + "\n")
    with pytest.raises(FileNotFoundError, match="incomplete pinned MASSIVE"):
        inputs._massive(tmp_path)


def test_massive_rejects_a_truncated_language_file(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(inputs, "MASSIVE_COUNTS", {"en.json.gz": 2})
    folder = tmp_path / "test"
    folder.mkdir()
    with gzip.open(folder / "en.json.gz", "wt") as stream:
        stream.write(json.dumps({"text": "only one of two rows"}) + "\n")
    with pytest.raises(ValueError, match="unexpected row count"):
        inputs._massive(tmp_path)
