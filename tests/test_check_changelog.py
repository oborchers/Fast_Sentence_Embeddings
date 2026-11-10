"""Exercise the committed-PR boundary and the real towncrier release format."""

import runpy
import shutil
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools/check_changelog.py"))
check = MODULE["check"]
main = MODULE["main"]
NOTES = runpy.run_path(str(ROOT / "tools/release_notes.py"))
CONFIG = """[project]
name = "probe"
requires-python = ">=3.11"
dependencies = []

[tool.towncrier]
package = ""
directory = "changelog.d"
filename = "CHANGELOG.md"
template = "changelog.d/template.md"
start_string = "## [Unreleased]\\n"
title_format = "## [{version}] - {project_date}"
issue_pattern = "DR-[0-9]+"
ignore = []

[[tool.towncrier.type]]
directory = "added"
name = "Added"
showcontent = true

[[tool.towncrier.type]]
directory = "fixed"
name = "Fixed"
showcontent = true
"""


@pytest.fixture
def repository(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Changelog test")
    git("config", "user.email", "test@example.invalid")
    git("config", "core.hooksPath", str(tmp_path / "no-hooks"))

    def commit(files):
        for name, text in files.items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if text is None:
                path.unlink()
            else:
                # Fixtures must use repository LF bytes, including on Windows.
                path.write_text(text, encoding="utf-8", newline="\n")
        git("add", ".")
        git("commit", "-qm", "Test change")
        return git("rev-parse", "HEAD")

    base = commit(
        {
            "pyproject.toml": CONFIG,
            "src/example.py": "old = True\n",
            "changelog.d/DR-1.fixed.md": "Existing entry.\n",
            "changelog.d/template.md": (ROOT / "changelog.d/template.md").read_text(),
            "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- Legacy.\n\n## [1.0.0]\n\n- Old.\n",
        }
    )
    return tmp_path, base, commit


@pytest.mark.parametrize(
    "files",
    [
        {"src/example.py": "new = True\n"},
        {"src/example.py": None},
        {"src/example.py": None, "tools/example.py": "old = True\n"},
        {"src/example.py": "new = True\n", "CHANGELOG.md": "# Changed notes\n"},
        {"setup.py": "# Build change\n"},
        {"MANIFEST.in": "include README.md\n"},
        {"pyproject.toml": CONFIG.replace(">=3.11", ">=3.12")},
        {"pyproject.toml": CONFIG.replace("dependencies = []", 'dependencies = ["numpy"]')},
        {"pyproject.toml": CONFIG + '\n[build-system]\nrequires = ["setuptools"]\n'},
        {"pyproject.toml": CONFIG + "\n[tool.setuptools]\ninclude-package-data = true\n"},
        {"pyproject.toml": CONFIG + '\n[tool.setuptools_scm]\nversion_file = "version.py"\n'},
    ],
)
def test_selected_changes_need_new_fragment(repository, files):
    root, base, commit = repository
    commit(files)
    with pytest.raises(ValueError, match="new nonempty"):
        check(root, base)


@pytest.mark.parametrize("fragment", [None, "", " \n", "edited"])
def test_old_or_empty_fragment_does_not_satisfy_gate(repository, fragment):
    root, base, commit = repository
    files = {"src/example.py": "new = True\n"}
    if fragment == "edited":
        files["changelog.d/DR-1.fixed.md"] = "Edited old fragment.\n"
    elif fragment is not None:
        files["changelog.d/DR-2.fixed.md"] = fragment
    commit(files)
    with pytest.raises(ValueError, match="new nonempty"):
        check(root, base)


@pytest.mark.parametrize("kind", ["unknown", "fixed"])
def test_fragment_must_use_ticket_filename(repository, kind):
    root, base, commit = repository
    commit({"src/example.py": "new = True\n", f"changelog.d/123.{kind}.md": "Entry.\n"})
    with pytest.raises(ValueError, match="new nonempty"):
        check(root, base)


@pytest.mark.parametrize(
    "files",
    [
        {"tests/test_example.py": "# Test\n"},
        {"docs/example.md": "Documentation.\n"},
        {"benchmarks/example.py": "# Benchmark\n"},
        {"tools/example.py": "# Tool\n"},
        {"uv.lock": "# Lock change\n"},
        {"pyproject.toml": CONFIG + '\n[dependency-groups]\nlint = ["towncrier"]\n'},
        {"pyproject.toml": CONFIG + '\n[tool.vulture]\nignore_names = ["example"]\n'},
    ],
)
def test_unselected_changes_skip_requirement(repository, files, capsys):
    root, base, commit = repository
    commit(files)
    check(root, base)
    assert "requirement skipped" in capsys.readouterr().out


@pytest.mark.parametrize("path", ["src/example.py", "setup.py", "MANIFEST.in", "pyproject.toml"])
def test_new_fragment_runs_real_towncrier(repository, path):
    pytest.importorskip(
        "towncrier", reason="release tooling is optional in sdist test environments"
    )
    root, base, commit = repository
    text = CONFIG.replace(">=3.11", ">=3.12") if path == "pyproject.toml" else "# Change\n"
    commit({path: text, "changelog.d/DR-2.fixed.md": "Correct the example.\n"})
    check(root, base)


def test_invalid_toml_and_unknown_base_fail(repository, monkeypatch, capsys):
    root, base, commit = repository
    commit({"pyproject.toml": "[broken"})
    with pytest.raises(tomllib.TOMLDecodeError):
        check(root, base)
    monkeypatch.chdir(root)
    assert main(["--base", "missing-reference"]) == 1
    assert "::error::" in capsys.readouterr().err


def test_removed_runtime_table_is_selected(repository):
    root, _, commit = repository
    base = commit({"pyproject.toml": CONFIG + '\n[build-system]\nrequires = ["setuptools"]\n'})
    commit({"pyproject.toml": CONFIG})
    with pytest.raises(ValueError, match="new nonempty"):
        check(root, base)


def test_draft_and_release_build_preserve_legacy_and_published_text(repository):
    pytest.importorskip(
        "towncrier", reason="release tooling is optional in sdist test environments"
    )
    root, _, commit = repository
    # Use the actual production configuration and its complete set of types.
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["towncrier"]
    actual = (ROOT / "pyproject.toml").read_text().split("# --- Changelog fragments ---", 1)[1]
    changelog = root / "CHANGELOG.md"
    original = changelog.read_bytes()
    assert b"\r" not in original
    published = original.split(b"## [1.0.0]", 1)[1]
    commit(
        {
            "pyproject.toml": actual,
            "changelog.d/DR-1.fixed.md": None,
            "changelog.d/DR-2.added.md": "Add a harmless example.\n\nSecond paragraph.\n\n```python\nprint(1)\n```\n",
            "changelog.d/DR-3.fixed.md": "Correct an example.\n",
        }
    )
    args = [
        sys.executable,
        "-m",
        "towncrier",
        "build",
        "--version",
        "2.1.0",
        "--date",
        "2026-11-10",
    ]
    draft = subprocess.check_output([*args, "--draft"], cwd=root, text=True)
    assert draft.startswith("## [2.1.0] - 2026-11-10\n\n### Added\n")
    assert "### Added\n\n- Add a harmless example.\n\n  Second paragraph." in draft
    assert "  ```python\n  print(1)\n  ```" in draft
    assert "### Fixed\n\n- Correct an example." in draft
    assert draft.count("## [2.1.0]") == 1
    assert changelog.read_bytes() == original
    assert (root / "changelog.d/DR-2.added.md").exists()
    subprocess.run([*args, "--yes"], cwd=root, check=True)
    result = changelog.read_bytes()
    # Towncrier reads universal newlines and writes with newline="". With an LF
    # fixture, generated and retained sections must both remain LF on Windows.
    assert b"\r" not in result
    assert result.startswith(
        b"# Changelog\n\n## [Unreleased]\n\n## [2.1.0] - 2026-11-10\n\n### Added\n"
    )
    assert result.split(b"## [1.0.0]", 1)[1] == published
    assert b"- Legacy." in result
    notes = root / "release-notes.md"
    assert NOTES["prepare"]("v2.1.0", changelog, notes) == ("2.1.0", False)
    assert "- Add a harmless example." in notes.read_text()
    assert "- Legacy." in notes.read_text()
    assert not list((root / config["directory"]).glob("DR-*.md"))
    # The production checkout and its real fragments are never changed by this probe.
    assert config["title_format"] == "## [{version}] - {project_date}"


def test_towncrier_errors_are_not_hidden_by_valid_fragment(repository):
    pytest.importorskip(
        "towncrier", reason="release tooling is optional in sdist test environments"
    )
    root, base, commit = repository
    commit(
        {
            "src/example.py": "# Change\n",
            "changelog.d/DR-2.fixed.md": "Valid new entry.\n",
            "changelog.d/unknown.txt": "Invalid entry.\n",
        }
    )
    with pytest.raises(subprocess.CalledProcessError):
        check(root, base)


def test_ci_selection_keeps_source_deletion_when_moved_to_docs(repository):
    if not shutil.which("bash") or not shutil.which("jq"):
        pytest.skip("the Linux CI selection requires bash and jq")
    root, base, commit = repository
    subprocess.run(
        ["git", "-C", str(root), "update-ref", "refs/remotes/origin/master", base], check=True
    )
    commit({"src/example.py": None, "docs/example.md": "old = True\n"})
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    step = workflow.split("      - id: diff\n", 1)[1].split("\n  lint:", 1)[0]
    script = textwrap.dedent(step.split("        run: |\n", 1)[1])
    output = root / "github-output"
    import os

    env = dict(
        os.environ,
        EVENT="pull_request",
        BASE="master",
        REF="refs/pull/1/merge",
        BEFORE="",
        TARGETED="false",
        GITHUB_OUTPUT=str(output),
        RUNNER_TEMP=str(root),
    )
    subprocess.run(["bash", "-e", "-c", script], cwd=root, env=env, check=True)
    assert "code=true" in output.read_text().splitlines()


def test_changelog_fixture_uses_lf_bytes(repository):
    root, _, _ = repository
    content = (root / "CHANGELOG.md").read_bytes()
    assert b"\n\n- Old.\n" in content
    assert b"\r" not in content
