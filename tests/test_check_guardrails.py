import importlib.util
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "check_guardrails.py"

pytestmark = pytest.mark.skipif(not TOOL.exists(), reason="tools/ is not part of this checkout")


@pytest.fixture(scope="module")
def tool():
    # runpy executes the tool without writing bytecode next to it.
    return runpy.run_path(str(TOOL))


def test_compare_accepts_the_ratchet_exactly(tool):
    ratchet = {"a.py::f": {"C901": 12, "cognitive": 20}, "b.py": 600}
    found = {"a.py::f": {"C901": 12, "cognitive": 20}}
    assert tool["compare"](found, {"a.py::f": ratchet["a.py::f"]}, "complexity") == []
    assert tool["compare"]({"b.py": {"lines": 600}}, {"b.py": 600}, "line") == []


@pytest.mark.parametrize(
    ("found", "ratchet", "message"),
    [
        ({"a.py::g": {"C901": 11}}, {}, "exceeds the complexity limit"),
        ({"a.py::f": {"C901": 13}}, {"a.py::f": {"C901": 12}}, "grew from 12 to 13"),
        ({"a.py::f": {"C901": 11}}, {"a.py::f": {"C901": 12}}, "lower its ratchet entry to 11"),
        ({}, {"a.py::f": {"C901": 12}}, "remove it from the ratchet"),
        ({"a.py::f": {"C901": 12, "PLR0912": 13}}, {"a.py::f": {"C901": 12}}, "PLR0912 13 exceeds"),
        ({"a.py::f": {"C901": 12}}, {"a.py::f": {"C901": 12, "PLR0912": 13}}, "PLR0912 now within"),
    ],
)
def test_compare_only_lets_the_ratchet_shrink(tool, found, ratchet, message):
    errors = tool["compare"](found, ratchet, "complexity")
    assert len(errors) == 1
    assert message in errors[0]


def test_qualnames_maps_def_and_decorator_lines(tool):
    source = "def f():\n    pass\n\nclass A:\n    @staticmethod\n    def g():\n        def h():\n            pass\n"
    names = tool["qualnames"](source)
    assert names[1] == "f"
    assert names[5] == names[6] == "A.g"
    assert names[7] == "A.g.h"


def test_long_files_skips_generated_files(tool, tmp_path):
    (tmp_path / "a.py").write_text("x\n" * 11)
    (tmp_path / "b.py").write_text("x\n" * 10)
    (tmp_path / "gen.py").write_text("x\n" * 50)
    found = tool["long_files"](tmp_path, ["a.py", "b.py", "gen.py"], 10, ["gen.py"])
    assert found == {"a.py": 11}


def _fse_tree(tmp_path, files):
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return {
        "native-imports": ["fse._contract"],
        "native-importers": ["src/fse/_text.py"],
        "untested-modules": [],
    }


def test_native_import_rules(tool, tmp_path):
    cfg = _fse_tree(
        tmp_path,
        {
            "src/fse/_native.pyx": "from fse._contract import Batch\nfrom fse._wordvectors import X\nimport scipy\n",
            "src/fse/_text.py": "from fse import _native\n",
            "src/fse/_other.py": "from fse import _native\n",
            "src/fse/_fine.py": "from fse import _contract\n",
        },
    )
    errors = tool["native_import_errors"](tmp_path, cfg)
    assert len(errors) == 3
    assert any("imports fse._wordvectors" in e for e in errors)
    assert any("imports scipy" in e for e in errors)
    assert any("_other.py imports fse._native" in e for e in errors)


def test_every_module_needs_a_test_file(tool, tmp_path):
    cfg = _fse_tree(
        tmp_path,
        {
            "src/fse/__init__.py": "",
            "src/fse/_a.py": "",
            "src/fse/b.py": "",
            "tests/test_init.py": "",
            "tests/test_a.py": "",
        },
    )
    assert tool["untested_modules"](tmp_path, cfg) == ["src/fse/b.py has no tests/test_b.py"]


def test_the_repository_config_parses(tool):
    cfg = tool["load_config"](ROOT)
    assert cfg["max-lines"] == 500
    assert set(cfg["ratchet"]) == {"lines", "functions"}


@pytest.mark.skipif(importlib.util.find_spec("complexipy") is None, reason="needs the lint group")
def test_the_repository_passes_its_guardrails(tool):
    errors, _, _ = tool["check"](ROOT)
    assert errors == []
