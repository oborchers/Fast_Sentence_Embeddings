"""Moving evidence preserves every measured value retained by compact summaries."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmarks/baseline"


@pytest.fixture
def original():
    manifest = json.loads((BASE / "dr151_evidence_manifest_20261101.json").read_text())
    commit = manifest["evidence_commit"]
    check = subprocess.run(
        ["git", "cat-file", "-e", commit], cwd=ROOT, capture_output=True, check=False
    )
    if check.returncode:
        pytest.skip(
            "Fetch benchmark-evidence-2.0 to verify compact values against archived originals"
        )

    def read(name):
        return subprocess.check_output(
            ["git", "show", commit + ":benchmarks/baseline/" + name], cwd=ROOT
        )

    return read


def test_matrix_keeps_rates_spreads_and_runtime_configuration(original):
    old = json.loads(original("dr151_matrix_summary_20261101.json"))
    compact = json.loads((BASE / "dr151_matrix_summary_20261101.json").read_text())
    assert len(compact["cells"]) == len(old["cells"])
    for source, encoded in zip(old["cells"], compact["cells"], strict=True):
        cell = dict(zip(compact["cell_columns"], encoded, strict=True))
        for key in cell.keys() - {"backends", "identity_status_id", "paired_speed_ratio_status_id"}:
            assert cell[key] == source.get(key)
        for field in ("identity_status", "paired_speed_ratio_status"):
            assert compact["attribution_labels"][cell[field + "_id"]] == source.get(field)
        for label, encoded_arm in cell["backends"].items():
            arm = dict(zip(compact["backend_columns"], encoded_arm, strict=True))
            expected = source["backends"][label]
            for key, value in compact["configurations"][arm["configuration"]].items():
                assert value == expected[key]
            for stat in ("median", "min", "max"):
                assert arm["texts_per_second_" + stat] == expected["texts_per_second"][stat]
            assert arm["peak_rss_bytes"] == expected["peak_rss_bytes"]


def test_artifact_catalog_has_identical_values(original):
    name = "dr151_model_artifacts_20261101.json"
    assert json.loads((BASE / name).read_text()) == json.loads(original(name))


@pytest.mark.parametrize(
    ("folder", "name"),
    [
        ("showcase-100m", "dr151_pod3_showcase_20261101.json"),
        ("large-end-to-end", "dr151_pod3_large_write_20261101.json"),
    ],
)
def test_small_pod3_summaries_preserve_original_bytes(original, folder, name):
    assert (BASE / name).read_bytes() == original(
        "results/dr151-pod3-20261101/" + folder + "/summary.json"
    )


def test_only_historical_1_1_results_are_tracked_on_the_code_branch():
    done = subprocess.run(
        ["git", "ls-files", "benchmarks/baseline/results"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if done.returncode:
        pytest.skip("Git metadata unavailable in this source distribution")
    expected = {
        "lifecycle-1.1.0-m1pro-sif-float32.json",
        "lifecycle-1.1.0-m1pro-sif-int8.json",
        "mteb-1.1.0.jsonl",
        "neighbors-1.1.0.npz",
        "numerics-1.1.0.json",
        "sts-1.1.0.csv",
        "train-1.1.0-m1pro.json",
        "train-dtypes-1.1.0-m1pro.json",
    }
    assert {Path(name).name for name in done.stdout.splitlines()} == expected


def assert_projection(actual, expected):
    """Retained source fields must match exactly, including nested numbers."""
    if isinstance(actual, dict):
        for key, value in actual.items():
            assert key in expected
            assert_projection(value, expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for value, source in zip(actual, expected, strict=True):
            assert_projection(value, source)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "stem",
    ["store_sync", "store_none", "thread_pool", "thread_pool_profiles", "dr190_runpod"],
)
def test_existing_master_studies_preserve_retained_values(original, stem):
    name = stem + "_20261031.json"
    data = (BASE / name).read_bytes()
    assert len(data) < 50_000
    compact = json.loads(data)
    source = json.loads(original(name))
    assert compact.pop("format") == "archived-study-summary-v1"
    raw = compact.pop("raw_evidence")
    manifest = json.loads((BASE / "dr151_evidence_manifest_20261101.json").read_text())
    entry = next(row for row in manifest["files"] if row["path"] == raw["path"])
    archived = original(name)
    assert len(raw["commit"]) == 40
    assert all(character in "0123456789abcdef" for character in raw["commit"])
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", raw["commit"], manifest["evidence_commit"]],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert ancestry.returncode == 0
    pinned = subprocess.check_output(["git", "show", raw["commit"] + ":" + raw["path"]], cwd=ROOT)
    assert pinned == archived
    assert raw["path"] == "benchmarks/baseline/" + name
    assert raw["size_bytes"] == entry["size_bytes"] == len(archived)
    assert raw["sha256"] == entry["sha256"] == hashlib.sha256(archived).hexdigest()
    assert raw["url"] == (
        "https://github.com/oborchers/Fast_Sentence_Embeddings/blob/"
        + raw["commit"]
        + "/benchmarks/baseline/"
        + name
    )
    if stem == "store_sync":
        for cell, old in zip(compact["cells"], source["cells"], strict=True):
            assert cell.pop("discarded_count") == len(old["discarded"])
    elif stem == "store_none":
        assert compact.pop("source_heads") == source["measurements"]["heads"]
        assert compact.pop("measurement_complete") == source["measurements"]["complete"]
    elif stem == "thread_pool_profiles":
        for label, variant in compact["variants"].items():
            hottest = variant.pop("hottest_recorded_functions")
            assert hottest == dict(
                list(source["variants"][label]["all_thread_functions"].items())[:3]
            )
    assert_projection(compact, source)
