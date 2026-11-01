"""The network gate executes the README itself, not a rewritten copy of its examples."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "tools/check_readme_examples.py"


def test_python_fences_keep_order_headings_and_source_lines(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# First\n```python\nanswer = 1\n```\n## Next\n```py\nanswer += 1\n```\n")
    parsed = runpy.run_path(str(CHECK))["snippets"](readme)
    assert [(heading, line) for heading, line, _ in parsed] == [("First", 3), ("Next", 7)]
    assert parsed[0][2] == "answer = 1\n"
    assert parsed[1][2] == "answer += 1\n"


@pytest.mark.network
def test_all_readme_python_examples_with_real_models(tmp_path):
    report = tmp_path / "report.json"
    subprocess.run([sys.executable, str(CHECK), "--report", str(report)], cwd=tmp_path, check=True)
    result = json.loads(report.read_text())
    expected = runpy.run_path(str(CHECK))["snippets"](ROOT / "README.md")
    assert len(result["examples"]) == len(expected)
    assert result["status"] == "passed"
    classifier = next(row for row in result["examples"] if "classifier_normalized" in row)
    assert classifier["classifier_normalized"] is False
    assert classifier["folded"] is True
    assert classifier["reload_without_sklearn"] is True


@pytest.mark.parametrize("marker", ["```", "````", "~~~", "~~~~"])
@pytest.mark.parametrize("indent", ["", "   "])
def test_fence_styles_preserve_blank_lines_and_displayed_code(tmp_path, marker, indent):
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# Example\n{indent}{marker}python\n\n{indent}answer = 1\n{indent}{marker}\n"
    )
    parsed = runpy.run_path(str(CHECK))["snippets"](readme)
    assert parsed == [("Example", 3, "\nanswer = 1\n")]


@pytest.mark.parametrize("prefix", ["> ", "- ", "    "])
def test_unsupported_nested_python_fences_fail_instead_of_disappearing(tmp_path, prefix):
    readme = tmp_path / "README.md"
    readme.write_text(f"{prefix}```python\nanswer = 1\n```\n")
    with pytest.raises(ValueError, match="Unsupported nested Python fence"):
        runpy.run_path(str(CHECK))["snippets"](readme)


def test_foreign_fences_do_not_create_false_python_examples(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "~~~~text\n```python\nnot_python\n```\n~~~~\n# Real\n```python\nanswer = 1\n```\n"
    )
    parsed = runpy.run_path(str(CHECK))["snippets"](readme)
    assert parsed == [("Real", 8, "answer = 1\n")]


def test_shorter_backticks_do_not_close_a_long_python_fence(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("````python\n```\n````\n")
    parsed = runpy.run_path(str(CHECK))["snippets"](readme)
    assert parsed == [("README", 2, "```\n")]
