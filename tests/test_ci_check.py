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
        "lint": str(code and not targeted).lower(),
        "native": "true",
        "sde": "true",
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


@pytest.mark.parametrize("failed_test", [False, True])
def test_failed_only_rerun_ignores_retained_downstream_skips(monkeypatch, failed_test):
    import json

    def completed(name, conclusion):
        return {"name": name, "status": "completed", "conclusion": conclusion}

    test_name = "test (ubuntu-latest, 3.13, latest)"
    previous = [
        completed(test_name, "failure"),
        completed("wheels (macos-latest, manylinux)", "success"),
        completed("release", "skipped"),
        completed("github-release", "skipped"),
    ]
    current = [completed(test_name, "failure" if failed_test else "success")]
    answers = iter(json.dumps([{"jobs": jobs}]).encode() for jobs in (previous, current))
    monkeypatch.setattr(MODULE["subprocess"], "check_output", lambda _command: next(answers))
    observed = MODULE["inventory"]("owner/repo", "123", 2)
    expected = [f"observed {test_name}: completed/failure"] if failed_test else []
    assert FAILURES(needs(), "workflow_dispatch", observed) == expected


@pytest.mark.parametrize("name", ["release", "github-release"])
@pytest.mark.parametrize("conclusion", ["success", "failure", "cancelled", "skipped", None])
def test_downstream_jobs_never_supply_gate_evidence(name, conclusion):
    observed = [{"name": name, "status": "completed", "conclusion": conclusion}]
    rows = needs()
    rows["test"]["result"] = "failure"
    assert FAILURES(rows, "push", observed) == ["test: failure"]


def test_unknown_skipped_job_still_fails_closed():
    observed = [{"name": "unexpected", "status": "completed", "conclusion": "skipped"}]
    assert FAILURES(needs(), "push", observed) == ["observed unexpected: completed/skipped"]


@pytest.mark.parametrize("code", [False, True])
@pytest.mark.parametrize("conclusion", ["success", "skipped", "failure", None])
def test_prefetch_matrix_follows_code_selection(code, conclusion):
    rows = needs(code=code)
    assert REQUIRED(rows, "pull_request")["test-prefetch"] == code
    observed = [
        {
            "name": "test-prefetch (ubuntu-latest, stride 4)",
            "status": "completed",
            "conclusion": conclusion,
        }
    ]
    allowed = conclusion == "success" or (not code and conclusion == "skipped")
    assert bool(FAILURES(rows, "pull_request", observed)) != allowed


def test_required_prefetch_job_cannot_be_missing_or_skipped():
    rows = needs()
    rows["test-prefetch"]["result"] = "skipped"
    assert FAILURES(rows, "pull_request", []) == ["test-prefetch: skipped"]
    del rows["test-prefetch"]
    with pytest.raises(ValueError, match="complete expected"):
        REQUIRED(rows, "pull_request")


@pytest.mark.parametrize("result", ["success", "skipped", "failure"])
def test_documentation_only_requires_successful_lint(result):
    rows = needs(code=False)
    rows["changes"]["outputs"]["lint"] = "true"
    for name in rows.keys() - {"changes", "lint"}:
        rows[name]["result"] = "skipped"
    rows["lint"]["result"] = result
    expected = [] if result == "success" else [f"lint: {result}"]
    assert FAILURES(rows, "pull_request", []) == expected


@pytest.mark.parametrize("value", [None, "invalid", True])
def test_lint_selection_requires_an_explicit_boolean_string(value):
    rows = needs()
    rows["changes"]["outputs"]["lint"] = value
    with pytest.raises(ValueError, match="explicit boolean strings"):
        FAILURES(rows, "pull_request", [])


@pytest.mark.parametrize("sde", [True, False])
@pytest.mark.parametrize("targeted", [True, False])
def test_sde_follows_narrow_input_selection_and_fails_closed(sde, targeted):
    rows = needs(targeted=targeted)
    rows["changes"]["outputs"]["sde"] = str(sde).lower()
    selected = sde
    assert REQUIRED(rows, "workflow_dispatch")["test-sde"] == selected
    rows["test-sde"]["result"] = "skipped"
    assert ("test-sde: skipped" in FAILURES(rows, "workflow_dispatch", [])) == selected
    rows["test-sde"]["result"] = "failure"
    assert "test-sde: failure" in FAILURES(rows, "workflow_dispatch", [])


SDE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/ci_sde_selection.py"))


@pytest.mark.parametrize(
    "path",
    [
        "src/fse/_native.pyx",
        "src/fse/_native_src/kernel/_native_kernel.h",
        "setup.py",
        "tools/check_sde.py",
        "tools/check_isa_isolation.py",
        ".github/workflows/ci.yml",
        "tests/test_native_isa.py",
        "tests/test_conformance_tiles.py",
        "tests/support/tiles.py",
    ],
)
def test_native_and_emulation_inputs_require_sde(path):
    assert SDE["selected"]([path])


@pytest.mark.parametrize(
    "path",
    [
        "src/fse/_store/_reader.py",
        "src/fse/_container/_load.py",
        "src/fse/_encoder/_encode.py",
        "tests/test_store.py",
        "docs/guide.md",
        "uv.lock",
    ],
)
def test_ordinary_python_store_and_document_changes_do_not_require_sde(path):
    assert not SDE["selected"]([path])


@pytest.mark.parametrize("table", ["build-system", "setuptools", "setuptools_scm", "cibuildwheel"])
def test_build_settings_require_sde(table):
    after = (
        {table: {"changed": True}}
        if table == "build-system"
        else {"tool": {table: {"changed": True}}}
    )
    assert SDE["selected"](["pyproject.toml"], {}, after)


@pytest.mark.parametrize(
    "after",
    [
        {"tool": {"ruff": {"line-length": 90}}},
        {"dependency-groups": {"test": ["pytest"]}},
        {"project": {"description": "updated"}},
    ],
)
def test_non_build_pyproject_settings_do_not_require_sde(after):
    assert not SDE["selected"](["pyproject.toml"], {}, after)


def test_missing_pyproject_or_sde_selection_fails_closed():
    assert SDE["selected"](["pyproject.toml"])
    rows = needs()
    del rows["changes"]["outputs"]["sde"]
    with pytest.raises(ValueError, match="explicit boolean"):
        REQUIRED(rows, "pull_request")
