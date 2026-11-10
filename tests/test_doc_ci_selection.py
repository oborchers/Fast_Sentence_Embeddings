"""Execute the real workflow selection so documentation changes cannot skip lint."""

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("jq") is None or not WORKFLOW.exists(),
    reason="workflow selection requires a Unix shell, jq and the checkout workflow",
)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def commit(root):
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "Fixture",
    )


def selection(
    root, event="pull_request", targeted="false", ref="refs/pull/1/merge", before="", base="master"
):
    source = WORKFLOW.read_text()
    step = source.split("      - id: diff\n", 1)[1].split("\n  lint:", 1)[0]
    script = textwrap.dedent(step.split("        run: |\n", 1)[1])
    output = root / "selection.out"
    env = {
        **os.environ,
        "EVENT": event,
        "BASE": base,
        "REF": ref,
        "BEFORE": before,
        "TARGETED": targeted,
        "GITHUB_OUTPUT": str(output),
    }
    subprocess.run(["bash", "-e", "-o", "pipefail", "-c", script], cwd=root, env=env, check=True)
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "initial.md").write_text("initial")
    commit(tmp_path)
    git(tmp_path, "update-ref", "refs/remotes/origin/master", "HEAD")
    return tmp_path


@pytest.mark.parametrize(
    "name",
    [
        "docs/guide.md",
        "guide.md",
        "docs/example.ipynb",
        "recipes/example.ipynb",
        ".agents/skills/example/SKILL.md",
        "src/example.py",
    ],
)
def test_document_and_code_changes_select_lint(repository, name):
    path = repository / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("new")
    commit(repository)
    outputs = selection(repository)
    assert outputs["lint"] == "true"
    if name.endswith(".md"):
        assert outputs["code"] == "false"


def test_document_deletion_selects_lint(repository):
    (repository / "initial.md").unlink()
    commit(repository)
    outputs = selection(repository)
    assert outputs["lint"] == "true"
    assert outputs["code"] == "false"


def test_full_and_targeted_selections_keep_lint_policy(repository):
    assert selection(repository, event="schedule")["lint"] == "true"
    assert selection(repository, event="workflow_dispatch", targeted="true")["lint"] == "false"


def test_deleted_non_markdown_target_selects_lint(repository):
    asset = repository / "docs" / "assets" / "image.svg"
    asset.parent.mkdir(parents=True)
    asset.write_text("svg")
    commit(repository)
    git(repository, "update-ref", "refs/remotes/origin/master", "HEAD")
    asset.unlink()
    commit(repository)
    outputs = selection(repository)
    assert outputs["lint"] == "true"
    assert outputs["code"] == "false"


def test_skill_only_master_push_selects_lint(repository):
    before = git(repository, "rev-parse", "HEAD")
    skill = repository / ".agents" / "skills" / "example" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("skill")
    commit(repository)
    outputs = selection(repository, event="push", ref="refs/heads/master", before=before)
    assert outputs["lint"] == "true"
    assert outputs["code"] == "false"


def test_unknown_pr_base_requires_full_selection(repository):
    outputs = selection(repository, base="missing-base")
    assert all(
        outputs[name] == "true" for name in ("code", "lint", "native", "deps", "st", "sdist")
    )


def test_workflow_change_requires_full_selection(repository):
    workflow = repository / ".github" / "workflows" / "example.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("workflow")
    commit(repository)
    outputs = selection(repository)
    assert all(
        outputs[name] == "true" for name in ("code", "lint", "native", "deps", "st", "sdist")
    )
