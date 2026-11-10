"""The release tag, CHANGELOG.md section and built files must agree before anything is published."""

import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/release_notes.py"))
version_of = MODULE["version_of"]
is_prerelease = MODULE["is_prerelease"]
section = MODULE["section"]
check_files = MODULE["check_files"]
prepare = MODULE["prepare"]
main = MODULE["main"]

CHANGELOG = """# Changelog

## [Unreleased]

- Next.

## [2.1.0rc1] - 2026-12-01

- Candidate.

## [2.0.0] - 2026-11-02

- First line.

  Continued.

- Second line.

## [1.1.0] - 2026-10-27

- Old.
"""


@pytest.mark.parametrize(("tag", "version"), [("v2.0.0", "2.0.0"), ("v2.1.0rc1", "2.1.0rc1")])
def test_version_of_accepts_release_tags(tag, version):
    assert version_of(tag) == version


@pytest.mark.parametrize(
    "tag",
    [
        "2.0.0",
        "v2.0",
        "v2.0.0.post1",
        "v2.0.0-rc1",
        "fse-v2.0.0",
        "v2.0.0 ",
        "v02.1.0",
        "v2.01.0",
        "v2.1.0rc01",
    ],
)
def test_version_of_rejects_other_tags(tag):
    with pytest.raises(ValueError, match="not of the form"):
        version_of(tag)


def test_prerelease_only_for_suffixed_versions():
    assert is_prerelease("2.1.0rc1")
    assert is_prerelease("3.0.0a2")
    assert is_prerelease("3.0.0b1")
    assert not is_prerelease("2.0.0")


def test_section_returns_the_body_of_exactly_that_version():
    assert section(CHANGELOG, "2.0.0") == "- First line.\n\n  Continued.\n\n- Second line."
    assert section(CHANGELOG, "1.1.0") == "- Old."
    assert section(CHANGELOG, "2.1.0rc1") == "- Candidate."


def test_section_does_not_match_a_prefix_of_a_version():
    with pytest.raises(ValueError, match=r"no section '## \[2\.0\]'"):
        section(CHANGELOG, "2.0")


def test_section_requires_the_section():
    with pytest.raises(ValueError, match="no section"):
        section(CHANGELOG, "2.0.1")


def test_section_refuses_the_unreleased_heading_as_a_version():
    with pytest.raises(ValueError, match="no section"):
        section("## [Unreleased]\n\n- x\n", "2.0.0")


def test_section_refuses_empty_and_duplicate_sections():
    with pytest.raises(ValueError, match="is empty"):
        section("## [2.0.0] - 2026-11-02\n\n## [1.0.0]\n\n- x\n", "2.0.0")
    with pytest.raises(ValueError, match="more than one"):
        section("## [2.0.0]\n\n- a\n\n## [2.0.0]\n\n- b\n", "2.0.0")


def test_check_files_accepts_sdist_and_wheels(tmp_path):
    for name in ["fse-2.0.0.tar.gz", "fse-2.0.0-cp313-cp313-manylinux_2_28_x86_64.whl"]:
        (tmp_path / name).write_bytes(b"")
    assert check_files(tmp_path, "2.0.0") == [
        "fse-2.0.0-cp313-cp313-manylinux_2_28_x86_64.whl",
        "fse-2.0.0.tar.gz",
    ]


def test_check_files_refuses_another_version_or_no_files(tmp_path):
    with pytest.raises(ValueError, match="no built files"):
        check_files(tmp_path, "2.0.0")
    (tmp_path / "fse-2.0.0.tar.gz").write_bytes(b"")
    (tmp_path / "fse-2.0.1-py3-none-any.whl").write_bytes(b"")
    with pytest.raises(ValueError, match=r"fse-2\.0\.1-py3-none-any\.whl"):
        check_files(tmp_path, "2.0.0")


def test_check_files_refuses_a_version_that_only_starts_like_the_tag(tmp_path):
    (tmp_path / "fse-2.0.0rc1.tar.gz").write_bytes(b"")
    with pytest.raises(ValueError, match="do not belong"):
        check_files(tmp_path, "2.0.0")


def test_prepare_writes_notes_and_reports_prerelease(tmp_path):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    notes = tmp_path / "notes.md"
    assert prepare("v2.1.0rc1", changelog, notes) == ("2.1.0rc1", True)
    assert notes.read_text(encoding="utf-8") == "- Candidate.\n"
    assert prepare("v1.1.0", changelog, notes) == ("1.1.0", False)


def test_prepare_writes_nothing_when_a_check_fails(tmp_path):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    notes = tmp_path / "notes.md"
    with pytest.raises(ValueError, match="no section"):
        prepare("v9.9.9", changelog, notes)
    assert not notes.exists()


def test_main_reports_errors_and_sets_outputs(tmp_path, monkeypatch, capsys):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    notes = tmp_path / "notes.md"
    github_output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    args = ["--changelog", str(changelog), "--output", str(notes)]
    assert main(["--tag", "v2.0.0", *args]) == 0
    assert github_output.read_text() == "version=2.0.0\nprerelease=false\n"
    assert main(["--tag", "v2.0.1", *args]) == 1
    assert "::error::CHANGELOG.md has no section" in capsys.readouterr().err
    assert main(["--tag", "latest", *args]) == 1


def test_main_checks_the_built_files_before_writing(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "fse-2.0.1.tar.gz").write_bytes(b"")
    notes = tmp_path / "notes.md"
    args = ["--tag", "v2.0.0", "--changelog", str(changelog), "--output", str(notes)]
    assert main([*args, "--dist", str(dist)]) == 1
    assert not notes.exists()


def test_section_ignores_headings_in_code_blocks():
    text = (
        "## [2.0.0]\n\n- Example:\n\n```markdown\n## [2.0.0]\n## [9.9.9]\n```\n\n"
        "~~~~\n## [1.0.0]\n~~~\n## [1.0.0]\n~~~~\n\n- tail\n\n## [1.1.0]\n\n- x\n"
    )
    assert section(text, "2.0.0").endswith("- tail")
    assert "## [9.9.9]" in section(text, "2.0.0")
    with pytest.raises(ValueError, match="no section"):
        section(text, "9.9.9")
    with pytest.raises(ValueError, match="no section"):
        section(text, "1.0.0")


@pytest.mark.parametrize("name", ["DR-343.changed.md", "unknown.txt", "nested/entry.md"])
def test_prepare_refuses_remaining_fragments_before_writing(tmp_path, name):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG)
    fragment = tmp_path / "changelog.d" / name
    fragment.parent.mkdir(parents=True)
    fragment.write_text("Remaining entry.\n")
    notes = tmp_path / "notes.md"
    with pytest.raises(ValueError, match="unbuilt changelog fragment"):
        prepare("v2.0.0", changelog, notes)
    assert not notes.exists()


def test_prepare_allows_only_fragment_metadata(tmp_path):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG)
    directory = tmp_path / "changelog.d"
    directory.mkdir()
    for name in ("README.md", "template.md.jinja"):
        (directory / name).write_text("Metadata.\n")
    assert prepare("v2.0.0", changelog, tmp_path / "notes.md") == ("2.0.0", False)


@pytest.mark.parametrize("root_link", [False, True])
def test_prepare_refuses_fragment_symlinks(tmp_path, root_link):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG)
    target = tmp_path / "target"
    target.mkdir()
    directory = tmp_path / "changelog.d"
    try:
        if root_link:
            directory.symlink_to(target, target_is_directory=True)
        else:
            directory.mkdir()
            (directory / "README.md").symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable for this account")
    with pytest.raises(ValueError, match=r"symlink|unexpected entry"):
        prepare("v2.0.0", changelog, tmp_path / "notes.md")
    assert not (tmp_path / "notes.md").exists()


def test_fragment_error_does_not_write_cli_outputs(tmp_path, monkeypatch, capsys):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG)
    directory = tmp_path / "changelog.d"
    directory.mkdir()
    (directory / "DR-343.changed.md").write_text("Remaining entry.\n")
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    notes = tmp_path / "notes.md"
    assert main(["--tag", "v2.0.0", "--changelog", str(changelog), "--output", str(notes)]) == 1
    assert "unbuilt changelog fragment" in capsys.readouterr().err
    assert not notes.exists()
    assert not output.exists()
