"""Optional jobs do not hide failures or missing required CI evidence."""

import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/ci_check.py"))
FAILURES = MODULE["failures"]
REQUIRED = MODULE["required"]


def needs(targeted=False, code=True):
    rows = {name: {"result": "success"} for name in MODULE["JOBS"]}
    rows["changes"]["outputs"] = {
        "code": str(code).lower(),
        "native": "true",
        "sdist": "true",
        "st": "true",
        "targeted": str(targeted).lower(),
    }
    return rows


def test_full_ci_requires_all_jobs_and_rejects_skipped_tag_wheels():
    rows = needs()
    assert all(REQUIRED(rows, "push").values())
    assert FAILURES(rows, "push", []) == []
    rows["wheels"]["result"] = "skipped"
    assert FAILURES(rows, "push", []) == ["wheels: skipped"]


def test_intentional_optional_manual_skips_are_accepted():
    rows = needs(targeted=True)
    rows["changes"]["outputs"]["sdist"] = "false"
    for name in ("lint", "coverage", "sdist"):
        rows[name]["result"] = "skipped"
    assert FAILURES(rows, "workflow_dispatch", []) == []
    rows["wheels"]["result"] = "skipped"
    assert FAILURES(rows, "workflow_dispatch", []) == ["wheels: skipped"]


def test_pr_coverage_and_skill_only_master_skips_follow_selection():
    rows = needs()
    rows["coverage"]["result"] = "skipped"
    assert FAILURES(rows, "pull_request", []) == []
    rows = needs(code=False)
    for name in rows.keys() - {"changes"}:
        rows[name]["result"] = "skipped"
    assert FAILURES(rows, "push", []) == []


@pytest.mark.parametrize(
    "result", ["failure", "cancelled", "abandoned", "neutral", "timed_out", "unknown"]
)
def test_started_optional_failures_and_unknown_results_are_never_allowed(result):
    rows = needs(code=False)
    rows["wheels"]["result"] = result
    assert FAILURES(rows, "pull_request", []) == [f"wheels: {result}"]


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "abandoned", "neutral", None])
def test_individual_failed_matrix_entry_is_not_hidden_by_aggregate_success(conclusion):
    rows = needs()
    observed = [
        {
            "name": "wheels (macos-latest, manylinux)",
            "status": "completed",
            "conclusion": conclusion,
        }
    ]
    assert FAILURES(rows, "push", observed)


def test_in_progress_job_is_not_success_and_summary_is_excluded():
    rows = needs()
    observed = [
        {"name": "test (ubuntu-latest, 3.13, latest)", "status": "in_progress", "conclusion": None}
    ]
    assert FAILURES(rows, "push", observed)
    observed[0]["name"] = "check"
    assert FAILURES(rows, "push", observed) == []


def test_missing_selection_and_unexpected_job_inventory_fail_closed():
    rows = needs()
    del rows["changes"]["outputs"]["code"]
    with pytest.raises(ValueError, match="explicit boolean"):
        FAILURES(rows, "push", [])
    rows = needs()
    del rows["wheels"]
    with pytest.raises(ValueError, match="complete expected"):
        FAILURES(rows, "push", [])


def test_failed_only_reruns_keep_previous_successes_and_latest_failure(monkeypatch):
    import json

    old = {
        "name": "wheels (macos-latest, manylinux)",
        "status": "completed",
        "conclusion": "success",
    }
    current = {**old, "conclusion": "abandoned"}
    retained = {
        "name": "test (ubuntu-latest, 3.13, latest)",
        "status": "completed",
        "conclusion": "success",
    }
    answers = iter(
        [
            json.dumps([{"jobs": [old, retained]}]).encode(),
            json.dumps([{"jobs": [current]}]).encode(),
        ]
    )
    monkeypatch.setattr(MODULE["subprocess"], "check_output", lambda _command: next(answers))
    rows = MODULE["inventory"]("owner/repo", "123", 2)
    assert rows == [current, retained]
    assert FAILURES(needs(), "push", rows)
