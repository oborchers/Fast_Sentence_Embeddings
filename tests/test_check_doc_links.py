"""Offline references must fail for missing targets, without executing documents."""

import errno
import json
import subprocess
import sys
from pathlib import Path

import pytest
from check_doc_links import check
from doc_references import markdown_references, notebook_references

TOOL = Path(__file__).resolve().parents[1] / "tools" / "check_doc_links.py"


def inventory(tmp_path, document, files=()):
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content")
    path = tmp_path / "docs" / "guide.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(document)
    return ["docs/guide.md", *files]


@pytest.mark.parametrize(
    "markup",
    [
        "[read](../target.md)",
        "![image](../target.md)",
        "[read][ref]\n\n[ref]: ../target.md",
        "[ref][]\n\n[ref]: ../target.md",
        "[ref]\n\n[ref]: ../target.md",
        '<a href="../target.md">read</a>',
        '| image |\n| --- |\n| <img src="../target.md" /> |',
        "[read](/target.md?download=1#anchor)",
        "`../target.md`",
        "`target.md`",
    ],
)
def test_existing_then_deleted_target_is_an_error(tmp_path, markup):
    paths = inventory(tmp_path, markup, ["target.md"])
    assert check(tmp_path, paths) == []
    (tmp_path / "target.md").unlink()
    errors = check(tmp_path, paths[:1])
    assert len(errors) == 1
    assert "docs/guide.md:1:" in errors[0] or "docs/guide.md:3:" in errors[0]
    assert "missing local target:" in errors[0]


def test_encoded_paths_titles_and_balanced_parentheses(tmp_path):
    paths = inventory(
        tmp_path,
        '[a](../a(b).md "Title")\n[b](<../with space.md>)\n[c](../with%20space.md#x)',
        ["a(b).md", "with space.md"],
    )
    assert check(tmp_path, paths) == []


@pytest.mark.parametrize(
    "example",
    [
        "```md\n[bad](missing.md)\n`../missing.py`\n```",
        "    [bad](missing.md)",
        '<!-- <img src="missing.svg"> -->',
        "`[bad](missing.md)`",
        "`uv run tools/missing.py`",
        "`tests/test_<module>.py`",
        "`tools/*.py`",
        "`/tmp/runtime.json`",
    ],
)
def test_examples_are_not_repository_links(tmp_path, example):
    assert check(tmp_path, inventory(tmp_path, example)) == []


def test_external_targets_and_anchors_need_no_network(tmp_path, monkeypatch):
    import socket

    def refuse(*_args, **_kwargs):
        pytest.fail("checker attempted a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    text = "\n".join(
        f"[external]({target})"
        for target in (
            "https://invalid.test/no.md",
            "//invalid.test/no.md",
            "mailto:x@y.test",
            "#x",
        )
    )
    assert check(tmp_path, inventory(tmp_path, text)) == []


@pytest.mark.parametrize("target", ["../../outside.md", "../../%6futside.md"])
def test_existing_targets_outside_checkout_are_rejected(tmp_path, target):
    root = tmp_path / "checkout"
    root.mkdir()
    (tmp_path / "outside.md").write_text("outside")
    errors = check(root, inventory(root, f"[outside]({target})"))
    assert len(errors) == 1
    assert "escapes checkout" in errors[0]


def notebook(*cells):
    return json.dumps({"nbformat": 4, "cells": list(cells)})


def markdown_cell(source, **kwargs):
    return {"cell_type": "markdown", "source": source, **kwargs}


@pytest.mark.parametrize("source", ["\n[missing](no.md)", ["\n", "[missing](no.md)"]])
def test_notebook_string_and_list_sources_report_cell_and_line(tmp_path, source):
    path = tmp_path / "example.ipynb"
    path.write_text(
        notebook({"cell_type": "code", "source": "[ignore](no.md)"}, markdown_cell(source))
    )
    assert check(tmp_path, [path.name]) == ["example.ipynb:cell 2:2: missing local target: no.md"]


def test_attachments_are_checked_in_their_own_cell(tmp_path):
    path = tmp_path / "example.ipynb"
    cell = markdown_cell(
        "![x](attachment:a%20b.png)", attachments={"a b.png": {"image/png": "data"}}
    )
    path.write_text(notebook(cell))
    assert check(tmp_path, [path.name]) == []
    path.write_text(notebook(cell, markdown_cell("![x](attachment:a%20b.png)")))
    assert check(tmp_path, [path.name]) == [
        "example.ipynb:cell 2:1: missing cell attachment: attachment:a%20b.png"
    ]


@pytest.mark.parametrize(
    "text",
    [
        "{",
        "[]",
        '{"cells": {}}',
        '{"cells": [3]}',
        '{"cells": [{"cell_type": "markdown", "source": 4}]}',
    ],
)
def test_invalid_notebook_is_a_clear_error(tmp_path, text):
    path = tmp_path / "bad.ipynb"
    path.write_text(text)
    errors = check(tmp_path, [path.name])
    assert len(errors) == 1
    assert errors[0].startswith("bad.ipynb:")


def test_notebook_code_and_outputs_are_not_markdown():
    assert (
        notebook_references(
            notebook(
                {"cell_type": "code", "source": "[x](no.md)", "outputs": [{"text": "[x](no.md)"}]}
            )
        )
        == []
    )


def test_reference_line_and_file_path_are_preserved():
    references = markdown_references("heading\n\n`tools/missing.py`\n")
    assert len(references) == 1
    assert references[0].target == "tools/missing.py"
    assert references[0].line == 3
    assert references[0].inline


def test_inline_explicit_paths_never_use_basename_fallback(tmp_path):
    paths = inventory(tmp_path, "`tools/gone.py`", ["src/gone.py", "tools/other.py"])
    assert check(tmp_path, paths) == ["docs/guide.md:1: missing local target: tools/gone.py"]


def test_cli_uses_tracked_files_and_fails_after_target_removal(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    paths = inventory(tmp_path, "[target](../target.md)", ["target.md"])
    subprocess.run(["git", "-C", str(tmp_path), "add", *paths], check=True)
    command = [sys.executable, str(TOOL), "--root", str(tmp_path)]
    good = subprocess.run(command, capture_output=True, text=True, check=False)
    assert good.returncode == 0, good.stderr
    (tmp_path / "target.md").unlink()
    bad = subprocess.run(command, capture_output=True, text=True, check=False)
    assert bad.returncode == 1
    assert "missing local target: ../target.md" in bad.stderr


def example_config(tmp_path, target="output.json", reason="Generated by the documented command"):
    (tmp_path / "pyproject.toml").write_text(
        "[[tool.fse.doc-links.inline-examples]]\n"
        'document = "docs/guide.md"\n'
        f'target = "{target}"\n'
        f'reason = "{reason}"\n'
    )


def test_exact_inline_exception_does_not_suppress_links(tmp_path):
    paths = inventory(tmp_path, "`output.json`\n\n[output](output.json)")
    example_config(tmp_path)
    assert check(tmp_path, paths) == ["docs/guide.md:3: missing local target: output.json"]


def test_unused_inline_exception_fails(tmp_path):
    paths = inventory(tmp_path, "No example here")
    example_config(tmp_path)
    assert check(tmp_path, paths) == ["docs/guide.md: unused inline example: output.json"]


def test_inline_exception_requires_a_reason(tmp_path):
    paths = inventory(tmp_path, "`output.json`")
    example_config(tmp_path, reason="")
    with pytest.raises(ValueError, match="nonempty strings"):
        check(tmp_path, paths)


def symlink(source, target):
    try:
        target.symlink_to(source)
    except OSError as error:
        if (
            error.errno in {errno.EACCES, errno.EPERM, errno.ENOSYS}
            or getattr(error, "winerror", None) == 1314
        ):
            pytest.skip("symlink creation is unavailable or not permitted")
        raise


def test_symlink_target_outside_checkout_is_rejected(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside")
    symlink(outside, root / "escape.md")
    errors = check(root, inventory(root, "[outside](../escape.md)"))
    assert errors == ["docs/guide.md:1: target escapes checkout: ../escape.md"]


def test_source_document_cannot_be_a_symlink_outside_checkout(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside")
    symlink(outside, root / "guide.md")
    assert check(root, ["guide.md"]) == ["guide.md: document escapes checkout"]


def test_root_prefix_does_not_resolve_beside_document(tmp_path):
    paths = inventory(tmp_path, "`tools/gone.py`", ["docs/tools/gone.py"])
    assert check(tmp_path, paths) == ["docs/guide.md:1: missing local target: tools/gone.py"]


def test_deleted_root_directory_is_still_a_file_reference(tmp_path):
    paths = inventory(tmp_path, "`integrations/gone.py`")
    assert check(tmp_path, paths) == ["docs/guide.md:1: missing local target: integrations/gone.py"]


def test_inline_root_path_can_contain_spaces(tmp_path):
    paths = inventory(tmp_path, "`tools/with space.py`", ["tools/with space.py"])
    assert check(tmp_path, paths) == []
    (tmp_path / "tools/with space.py").unlink()
    assert check(tmp_path, paths[:1]) == [
        "docs/guide.md:1: missing local target: tools/with space.py"
    ]


@pytest.mark.parametrize("cell", [{}, {"cell_type": "markdwon", "source": "[x](no.md)"}])
def test_unknown_notebook_cell_type_fails(tmp_path, cell):
    path = tmp_path / "bad.ipynb"
    path.write_text(notebook(cell))
    assert check(tmp_path, [path.name]) == ["bad.ipynb: cell 1 has a missing or unknown cell_type"]


@pytest.mark.parametrize(
    "config",
    [
        "[tool]\nfse = 3\n",
        "[tool.fse]\ndoc-links = 3\n",
        "[tool.fse.doc-links]\ninline-examples = 3\n",
    ],
)
def test_invalid_exception_configuration_fails_clearly(tmp_path, config):
    (tmp_path / "pyproject.toml").write_text(config)
    with pytest.raises(ValueError, match="must be"):
        check(tmp_path, [])


def test_external_link_label_is_not_a_local_file_path(tmp_path):
    assert (
        check(tmp_path, inventory(tmp_path, "[`missing.py`](https://invalid.test/missing.py)"))
        == []
    )
