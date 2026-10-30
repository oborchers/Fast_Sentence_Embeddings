import importlib.util
import re
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "check_guardrails.py"

pytestmark = pytest.mark.skipif(not TOOL.exists(), reason="tools/ is not part of this checkout")
needs_lint_tools = pytest.mark.skipif(
    importlib.util.find_spec("complexipy") is None or importlib.util.find_spec("ruff") is None,
    reason="needs the lint group",
)


@pytest.fixture(scope="module")
def tool():
    # runpy executes the tool without writing bytecode next to it.
    return runpy.run_path(str(TOOL))


def test_compare_accepts_the_ratchet_exactly(tool):
    entry = {"C901": 12, "cognitive": 20}
    assert tool["compare"]({"a.py::f": dict(entry)}, {"a.py::f": entry}, "complexity") == []
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


def test_the_ratchet_must_not_grow_against_its_base(tool):
    base = {"lines": {"a.py": 600}, "functions": {"a.py::f": {"C901": 12}}}
    assert tool["ratchet_growth"](base, base) == []
    assert tool["ratchet_growth"]({"lines": {"a.py": 590}}, base) == []
    grown = {
        "lines": {"a.py": 601, "b.py": 510},
        "functions": {"a.py::f": {"C901": 12, "PLR0912": 13}},
    }
    errors = tool["ratchet_growth"](grown, base)
    assert len(errors) == 3


def test_qualnames(tool):
    source = (
        "def f():\n    pass\n\nclass A:\n    @staticmethod\n    def g():\n        def h():\n"
        "            pass\n\nif True:\n    def k():\n        pass\n"
    )
    names = tool["qualnames"](source)
    assert names[1] == "f"
    assert names[5] == names[6] == "A.g"
    assert names[7] == "A.g.h"
    assert names[11] == "k"


def test_long_files_skips_generated_files(tool, tmp_path):
    (tmp_path / "a.py").write_text("x\n" * 11)
    (tmp_path / "b.py").write_text("x\n" * 10)
    (tmp_path / "gen.py").write_text("x\n" * 50)
    found = tool["long_files"](tmp_path, ["a.py", "b.py", "gen.py"], 10, ["gen.py"])
    assert found == {"a.py": 11}


def test_extension_imports(tool):
    source = (
        "import numpy as np, scipy\ncimport numpy as np\nfrom libc.stdint cimport int64_t\n"
        "from fse import _contract, _text\nfrom . import _wordvectors\nfrom .._x import y\n"
        '"""\nimport torch\n"""\n# import sklearn\ns = "from gensim import x"\n'
    )
    assert sorted(tool["IMPORTS"]["extension_imports"](source)) == sorted(
        [
            "numpy",
            "scipy",
            "numpy",
            "libc.stdint",
            "fse._contract",
            "fse._text",
            "fse._wordvectors",
            "fse._x",
        ]
    )


def _fse_tree(tmp_path, files):
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return {
        "native-imports": ["fse._contract"],
        "native-importers": ["src/fse/_text.py"],
        "hash-importers": ["src/fse/_container/_hash.py"],
        "hash-modules": ["hashlib", "xxhash"],
        "log-importers": ["src/fse/_log.py"],
        "log-modules": ["logging", "warnings"],
        "log-free": ["src/fse/_dev/_canary.py", "src/fse/_dev/_load.py"],
        "untested-modules": [],
        "forbidden": ["gensim", "scipy", "torch"],
    }


def test_native_import_rules(tool, tmp_path):
    cfg = _fse_tree(
        tmp_path,
        {
            "src/fse/_native.pyx": "from fse._contract import Batch\nfrom fse import _wordvectors\nimport torch\n",
            "src/fse/_text.py": "from fse import _native\n",
            "src/fse/_other.py": "from . import _native\n",
            "src/fse/_sub/_deep.py": "from .._native import pool\n",
            "src/fse/_fine.py": "from fse import _contract\n",
        },
    )
    errors = tool["IMPORTS"]["native_import_errors"](tmp_path, cfg)
    assert len(errors) == 4, errors
    assert any("imports fse._wordvectors" in e for e in errors)
    assert any("imports torch" in e for e in errors)
    assert any("_other.py imports fse._native" in e for e in errors)
    assert any("_deep.py imports fse._native" in e for e in errors)


def test_only_the_harness_imports_a_hash_library(tool, tmp_path):
    cfg = _fse_tree(
        tmp_path,
        {
            "src/fse/_container/_hash.py": "import hashlib\nimport xxhash\n",
            "src/fse/_container/_npy.py": "from . import _hash\n",
            "src/fse/_wordvectors.py": "import hashlib\n",
            "src/fse/_sub/_deep.py": "from xxhash import xxh3_128 as h\n",
            "src/fse/_native_part.pxi": "from hashlib import sha256\n",
            "src/fse/_note.py": '"""import hashlib"""\nhashlibrary = 1\n',
        },
    )
    errors = tool["IMPORTS"]["exclusive_import_errors"](tmp_path, cfg, "hash")
    assert len(errors) == 3, errors
    assert any("_wordvectors.py imports hashlib" in e for e in errors)
    assert any("_deep.py imports xxhash" in e for e in errors)
    assert any("_native_part.pxi imports hashlib" in e for e in errors)


def test_only_the_log_module_imports_logging_and_the_device_thread_never_logs(tool, tmp_path):
    cfg = _fse_tree(
        tmp_path,
        {
            "src/fse/_log.py": "import logging\nimport warnings\n",
            "src/fse/_store.py": "from fse import _log\nimport logging.handlers\n",
            "src/fse/_native_part.pxi": "import warnings\n",
            "src/fse/_dev/_canary.py": "from fse import _log\n",
            "src/fse/_dev/_load.py": "from fse._log import emit\n",
            "src/fse/_dev/_device.py": "from fse import _log\n",
        },
    )
    errors = tool["IMPORTS"]["import_errors"](tmp_path, cfg)
    assert sorted(errors) == [
        "src/fse/_dev/_canary.py imports fse._log; the MLX device thread never logs",
        "src/fse/_dev/_load.py imports fse._log; the MLX device thread never logs",
        "src/fse/_native_part.pxi imports warnings; only ['src/fse/_log.py'] may",
        "src/fse/_store.py imports logging; only ['src/fse/_log.py'] may",
    ]


def test_only_the_loader_of_a_device_backend_imports_its_library(tool, tmp_path):
    files = {
        "src/fse/_backends/_mlx/_load.py": 'import importlib\nmx = importlib.import_module("mlx.core")\n',
        "src/fse/_backends/_mlx/_kernels.py": "import mlx.core as mx\n",
        "src/fse/_backends/_mlx/_device.py": 'mx = __import__("mlx")\n',
        "src/fse/_backends/_cuda/_load.py": "from mlx import core\n",
        "src/fse/_backends/_cuda/_other.py": "import cupy\n",
        "src/fse/_text.py": '"""import mlx"""\nmlxish = importlib.import_module(name)\n',
    }
    files["src/fse/_alias.py"] = 'import importlib as il\nil.import_module("mlx.core")\n'
    files["src/fse/_renamed.py"] = 'from importlib import import_module as load\nload("mlx.core")\n'
    files["src/fse/_keyword.py"] = 'import importlib\nimportlib.import_module(name="mlx.core")\n'
    cfg = _fse_tree(tmp_path, files)
    cfg["device-libraries"] = {"mlx": ["mlx"], "cuda": ["cupy"]}
    errors = tool["IMPORTS"]["device_import_errors"](tmp_path, cfg)
    assert len(errors) == 7, errors
    for name in ("_alias.py", "_renamed.py", "_keyword.py"):
        assert any(f"src/fse/{name} imports mlx" in e for e in errors), (name, errors)
    for name in ("_mlx/_kernels.py imports mlx", "_mlx/_device.py imports mlx"):
        assert any(name in e and "_mlx/_load.py may" in e for e in errors), errors
    assert any("_cuda/_load.py imports mlx" in e for e in errors)
    assert any(
        "_cuda/_other.py imports cupy; only src/fse/_backends/_cuda/_load.py" in e for e in errors
    )


def test_every_module_needs_a_test_file(tool, tmp_path):
    cfg = _fse_tree(
        tmp_path,
        {
            "src/fse/__init__.py": "",
            "src/fse/_a.py": "",
            "src/fse/b.py": "",
            "src/fse/_c/__init__.py": "",
            "src/fse/_c/_d.py": "",
            "tests/test_init.py": "",
            "tests/test_a.py": "",
            "tests/test_c.py": "",
        },
    )
    assert tool["untested_modules"](tmp_path, cfg) == [
        "src/fse/_c/_d.py has no tests/test_c_d.py",
        "src/fse/b.py has no tests/test_b.py",
    ]


def test_the_repository_config_parses(tool):
    cfg = tool["load_config"](ROOT)
    assert cfg["max-lines"] == 500
    assert set(cfg["ratchet"]) == {"lines", "functions"}
    assert {"gensim", "scipy", "torch"} <= set(cfg["forbidden"])
    assert cfg["device-libraries"] == {"mlx": ["mlx"]}
    assert "mlx" in cfg["forbidden"]  # for the extension


@needs_lint_tools
def test_the_repository_passes_its_guardrails(tool):
    errors, _, _ = tool["check"](ROOT)
    assert errors == []


def _repo(tmp_path, body, ratchet=""):
    """A git repository with this pyproject's guardrail settings and one module."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pyproject = (ROOT / "pyproject.toml").read_text()
    start = pyproject.index("\n[tool.ruff.lint.mccabe]\n")
    settings = pyproject[start : pyproject.index("# The ratchet:")]
    # Without the modules of the MLX device thread, which this repository does not have.
    settings = re.sub(r"\nlog-free = \[[^]]*\]\n", "\n", settings)
    contracts = pyproject[pyproject.index("\n[tool.importlinter]\n") :]
    contracts = contracts[: contracts.index("# --- Unused code")]
    (tmp_path / "pyproject.toml").write_text(settings + ratchet + "\n" + contracts)
    (tmp_path / "src" / "fse").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "fse" / "_mod.py").write_text(body)
    (tmp_path / "tests" / "test_mod.py").write_text("")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


BRANCHY = (
    "def f(x):\n"
    + "".join(f"    if x == {i}:\n        return {i}\n" for i in range(8))
    + "    return -1\n"
)


@needs_lint_tools
@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_command_fails_on_a_new_finding_and_passes_with_its_ratchet(tool, tmp_path, capsys):
    # The noqa does not hide the finding.
    repo = _repo(tmp_path, BRANCHY.replace("def f(x):", "def f(x):  # noqa: PLR0911"))
    assert tool["main"]([], root=repo) == 1
    assert "src/fse/_mod.py::f" in capsys.readouterr().out
    ratchet = '\n[tool.fse.guardrails.ratchet.functions]\n"src/fse/_mod.py::f" = { PLR0911 = 9 }\n'
    repo = _repo(tmp_path / "with", BRANCHY, ratchet)
    assert tool["main"]([], root=repo) == 0


@needs_lint_tools
def test_the_command_reports_a_failing_tool(tool, tmp_path, capsys):
    (tmp_path / "pyproject.toml").write_text((ROOT / "pyproject.toml").read_text())
    assert tool["main"]([], root=tmp_path) == 2  # not a git repository
    assert "check_guardrails: git" in capsys.readouterr().err


def test_extension_imports_across_lines_and_against_comment_tricks(tool):
    source = 'from fse import (\n    _contract,\n    _wordvectors,\n)\n# """\nimport torch\n# """\nx = "# not a comment"\n'
    assert tool["IMPORTS"]["extension_imports"](source) == [
        "fse._contract",
        "fse._wordvectors",
        "torch",
    ]


def _commit(repo, message):
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t"}
    env |= {"GIT_COMMITTER_EMAIL": "t@t", "PATH": __import__("os").environ["PATH"]}
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True, env=env)


RATCHET = '\n[tool.fse.guardrails.ratchet.functions]\n"src/fse/_mod.py::f" = {{ PLR0911 = {} }}\n'


@needs_lint_tools
@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_command_rejects_a_ratchet_that_grows_against_its_base(tool, tmp_path, capsys):
    repo = _repo(tmp_path, BRANCHY, RATCHET.format(9))
    _commit(repo, "base")
    # One more return: the finding grows to 10, and the ratchet is raised to match it.
    grown = BRANCHY.replace("    return -1\n", "    if x == 8:\n        return 8\n    return -1\n")
    (repo / "src" / "fse" / "_mod.py").write_text(grown)
    pyproject = (repo / "pyproject.toml").read_text().replace("PLR0911 = 9", "PLR0911 = 10")
    (repo / "pyproject.toml").write_text(pyproject)
    assert tool["main"]([], root=repo) == 0
    assert tool["main"](["--base", "HEAD"], root=repo) == 1
    assert "is new or higher than on the base" in capsys.readouterr().out


@needs_lint_tools
@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_an_empty_base_ratchet_still_forbids_new_entries(tool, tmp_path):
    repo = _repo(tmp_path, "def f():\n    return 1\n", "\n[tool.fse.guardrails.ratchet]\n")
    _commit(repo, "base with an empty ratchet")
    (repo / "src" / "fse" / "_mod.py").write_text(BRANCHY)
    pyproject = (repo / "pyproject.toml").read_text() + RATCHET.format(9)
    (repo / "pyproject.toml").write_text(
        pyproject.replace("\n[tool.fse.guardrails.ratchet]\n", "\n")
    )
    assert tool["main"]([], root=repo) == 0
    assert tool["main"](["--base", "HEAD"], root=repo) == 1


@needs_lint_tools
@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_command_rejects_a_forbidden_native_import(tool, tmp_path, capsys):
    repo = _repo(tmp_path, "def f():\n    return 1\n")
    (repo / "src" / "fse" / "_native_part.pxi").write_text(
        "from fse import (\n    _wordvectors,\n)\n"
    )
    assert tool["main"]([], root=repo) == 1
    assert "_native_part.pxi imports fse._wordvectors" in capsys.readouterr().out


@pytest.mark.parametrize(
    "source",
    [
        'slash = "\\\\"\nimport torch\n',
        "slash = '\\\\'\nimport torch\n",
        '"""ends with a backslash \\\\"""\nimport torch\n',
        "import numpy; import torch\n",
        "if True: import torch\n",
        "import numpy, \\\n    torch\n",
        "cdef int x = 1; from torch import nn\n",
    ],
)
def test_hidden_imports_are_found(tool, source):
    assert any(m.split(".")[0] == "torch" for m in tool["IMPORTS"]["extension_imports"](source)), (
        source
    )


@pytest.mark.parametrize(
    "source",
    [
        's = "import torch"\n',
        "s = 'from torch import x'\n",
        '"""\nimport torch\n"""\n',
        "# import torch\n",
        'quote = "\\""  # import torch\n',
        "importance = 1\nreimport = 2\n",
        'cdef extern from "torch.h":\n    pass\n',
    ],
)
def test_imports_in_strings_and_comments_are_ignored(tool, source):
    assert not any(
        m.split(".")[0] == "torch" for m in tool["IMPORTS"]["extension_imports"](source)
    ), source


@needs_lint_tools
@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_command_finds_an_import_after_an_escaped_quote(tool, tmp_path, capsys):
    repo = _repo(tmp_path, "def f():\n    return 1\n")
    (repo / "src" / "fse" / "_native.pyx").write_text(
        'slash = "\\\\"\nimport numpy; import torch\n'
    )
    (repo / "tests" / "test_native.py").write_text("")
    assert tool["main"]([], root=repo) == 1
    assert "_native.pyx imports torch" in capsys.readouterr().out
