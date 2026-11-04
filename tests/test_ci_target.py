"""Explicit PR/SHA binding for optional manual CI, without network calls."""

import copy
import runpy
from pathlib import Path

import pytest

TARGET = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/ci_target.py"))["target"]
SHA = "a" * 40
REPOSITORY = "owner/fse"


def pull_request():
    return {
        "number": 217,
        "state": "open",
        "base": {"repo": {"full_name": REPOSITORY}, "ref": "master"},
        "head": {"sha": SHA, "repo": {"full_name": "fork/fse"}},
    }


def test_exact_current_sha_and_source_repository_are_selected():
    assert TARGET(pull_request(), REPOSITORY, "217", SHA) == {
        "source_repository": "fork/fse",
        "source_ref": SHA,
        "target_pr": "217",
    }


@pytest.mark.parametrize(
    ("number", "sha"),
    [
        ("0", SHA),
        ("-1", SHA),
        ("217/../1", SHA),
        ("217", "a" * 7),
        ("217", "A" * 40),
        ("217", "a" * 39 + "\n"),
    ],
)
def test_malformed_target_is_rejected(number, sha):
    with pytest.raises(ValueError, match="requires"):
        TARGET(pull_request(), REPOSITORY, number, sha)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("number",), 218),
        (("state",), "closed"),
        (("base", "ref"), "other"),
        (("base", "repo", "full_name"), "other/fse"),
        (("head", "sha"), "b" * 40),
        (("head", "repo", "full_name"), "fork/fse\nsource_ref=bad"),
    ],
)
def test_stale_or_mismatched_pr_is_rejected(path, value):
    pr = copy.deepcopy(pull_request())
    node = pr
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match=r"requested|repository"):
        TARGET(pr, REPOSITORY, "217", SHA)
