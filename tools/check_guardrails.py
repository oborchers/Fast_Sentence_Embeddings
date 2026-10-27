"""Check the guardrails of the repository against the ratchet in pyproject.toml.

The guardrails ([tool.fse.guardrails]) keep the code reviewable:

- at most ``max-lines`` lines per hand-written source file;
- the ruff complexity limits of [tool.ruff.lint.mccabe] and [tool.ruff.lint.pylint]
  (C901, PLR0911, PLR0912, PLR0913, PLR0915, PLR0917) and a cognitive complexity of at
  most ``max-cognitive-complexity`` per function (complexipy);
- the imports of the native extension, which import-linter cannot see: only
  ``native-importers`` import ``fse._native``, and the extension imports only
  ``native-imports`` from fse;
- a ``tests/test_<module>.py`` for every module of fse.

What exceeded a limit when the guardrails came is the ratchet. It only shrinks: a new
finding fails, a value above its entry fails, a value below its entry or within the
limit fails until the entry is lowered or removed. Inline suppressions are ignored.

    uv run python tools/check_guardrails.py
    uv run python tools/check_guardrails.py --print-ratchet   # the current findings as TOML
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_SUFFIXES = {".py", ".pyx", ".pxd", ".pxi", ".pyi", ".h", ".c"}
RUFF_RULES = ["C901", "PLR0911", "PLR0912", "PLR0913", "PLR0915", "PLR0917"]
_RUFF = [sys.executable, "-m", "ruff", "check", "--output-format", "json", "--exit-zero"]
_RUFF += ["--no-cache", "--quiet"]
_VALUE = re.compile(r"\((\d+) > \d+\)")
_NATIVE_IMPORT = re.compile(r"^\s*(?:from\s+(\S+)\s+c?import|c?import\s+(\S+))", re.MULTILINE)
_FORBIDDEN = ("gensim", "scipy", "smart_open", "psutil", "threadpoolctl", "wordfreq")


def load_config(root: Path) -> dict:
    """The [tool.fse.guardrails] table of the pyproject.toml in root."""
    with open(root / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["fse"]["guardrails"]


def source_files(root: Path, paths: list[str]) -> list[str]:
    """Tracked and untracked, not ignored source files under paths, relative to root."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *paths],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    files = {f for f in out.splitlines() if Path(f).suffix in SOURCE_SUFFIXES}
    return sorted(f for f in files if (root / f).is_file())


def long_files(
    root: Path, files: list[str], max_lines: int, generated: list[str]
) -> dict[str, int]:
    """The line count of every file longer than max_lines, generated files excepted."""
    found = {}
    for name in files:
        if name in generated:
            continue
        lines = len((root / name).read_text(encoding="utf-8").splitlines())
        if lines > max_lines:
            found[name] = lines
    return found


def qualnames(source: str) -> dict[int, str]:
    """The qualified name (Class.method) of every function by the line of its def."""
    names: dict[int, str] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{prefix}{child.name}"
                if not isinstance(child, ast.ClassDef):
                    names[child.lineno] = name
                    for decorator in child.decorator_list:
                        names[decorator.lineno] = name
                visit(child, name + ".")

    visit(ast.parse(source), "")
    return names


def ruff_findings(root: Path, py_files: list[str]) -> dict[str, dict[str, int]]:
    """The ruff complexity findings by path::qualname and rule."""
    if not py_files:
        return {}
    out = subprocess.run(
        [*_RUFF, "--select", ",".join(RUFF_RULES), *py_files],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found: dict[str, dict[str, int]] = {}
    names_by_file: dict[str, dict[int, str]] = {}
    for item in json.loads(out or "[]"):
        path = Path(item["filename"]).resolve().relative_to(root).as_posix()
        if path not in names_by_file:
            names_by_file[path] = qualnames((root / path).read_text(encoding="utf-8"))
        name = names_by_file[path].get(item["location"]["row"], f"line {item['location']['row']}")
        match = _VALUE.search(item["message"])
        found.setdefault(f"{path}::{name}", {})[item["code"]] = int(match.group(1)) if match else 0
    return found


def cognitive_findings(root: Path, py_files: list[str], limit: int) -> dict[str, dict[str, int]]:
    """The functions whose cognitive complexity exceeds limit, by path::qualname."""
    # Imported here: the tests of this tool run without the lint group.
    file_complexity = importlib.import_module("complexipy").file_complexity
    found: dict[str, dict[str, int]] = {}
    for path in py_files:
        result = file_complexity(str(root / path), no_ignore=True, with_plans=False)
        for function in result.functions:
            if function.complexity > limit:
                name = function.name.replace("::", ".")
                found.setdefault(f"{path}::{name}", {})["cognitive"] = function.complexity
    return found


def native_import_errors(root: Path, cfg: dict) -> list[str]:
    """Imports of and by the native extension that break the rules."""
    return [*_extension_import_errors(root, cfg), *_native_importer_errors(root, cfg)]


def _extension_import_errors(root: Path, cfg: dict) -> list[str]:
    errors = []
    allowed = set(cfg["native-imports"])
    for path in sorted((root / "src" / "fse").glob("_native*.p[yx]*")):
        if path.suffix == ".pyi":
            continue
        for match in _NATIVE_IMPORT.finditer(path.read_text(encoding="utf-8")):
            module = (match.group(1) or match.group(2)).rstrip(",")
            if (module.startswith("fse") and module not in allowed) or module.split(".")[
                0
            ] in _FORBIDDEN:
                errors.append(
                    f"{path.relative_to(root)} imports {module}; it may import only {sorted(allowed)}"
                )
    return errors


def _native_importer_errors(root: Path, cfg: dict) -> list[str]:
    errors = []
    for path in sorted((root / "src" / "fse").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in cfg["native-importers"]:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(_imports_native(node) for node in ast.walk(tree)):
            errors.append(f"{rel} imports fse._native; only {cfg['native-importers']} may")
    return errors


def _imports_native(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.startswith("fse._native") for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return module.startswith("fse._native") or (
            module == "fse" and any(alias.name == "_native" for alias in node.names)
        )
    return False


def untested_modules(root: Path, cfg: dict) -> list[str]:
    """Modules of fse without a tests/test_<name>.py."""
    missing = []
    for path in sorted((root / "src" / "fse").glob("*.py")):
        rel = path.relative_to(root).as_posix()
        name = "init" if path.stem == "__init__" else path.stem.lstrip("_")
        if rel not in cfg["untested-modules"] and not (root / "tests" / f"test_{name}.py").exists():
            missing.append(f"{rel} has no tests/test_{name}.py")
    return missing


def compare(found: dict, ratchet: dict, limit_name: str) -> list[str]:
    """The ratchet rules: new and grown findings fail, and improved or stale entries fail
    until their entry is lowered or removed."""
    errors = []
    for key, values in found.items():
        errors += _compare_finding(key, values, ratchet.get(key), limit_name)
    for key, entry in ratchet.items():
        metrics = entry if isinstance(entry, dict) else {"lines": entry}
        stale = [metric for metric in metrics if metric not in found.get(key, {})]
        if stale:
            errors.append(
                f"{key}: {', '.join(stale)} now within the limit; remove it from the ratchet"
            )
    return errors


def _compare_finding(
    key: str, values: dict, entry: dict | int | None, limit_name: str
) -> list[str]:
    if entry is None:
        return [f"{key}: {values} exceeds the {limit_name} limit; split or simplify it"]
    errors = []
    for metric, value in values.items():
        allowed = entry.get(metric) if isinstance(entry, dict) else entry
        if allowed is None:
            errors.append(f"{key}: {metric} {value} exceeds its limit; split or simplify it")
        elif value != allowed:
            change = "grew" if value > allowed else "improved"
            fix = (
                "split or simplify it" if value > allowed else f"lower its ratchet entry to {value}"
            )
            errors.append(f"{key}: {metric} {change} from {allowed} to {value}; {fix}")
    return errors


def check(root: Path) -> tuple[list[str], dict, dict]:
    """Every guardrail error, and the current file and function findings."""
    cfg = load_config(root)
    files = source_files(root, cfg["paths"])
    py_files = [f for f in files if f.endswith(".py") and f not in cfg["generated"]]
    lines = long_files(root, files, cfg["max-lines"], cfg["generated"])
    functions = ruff_findings(root, py_files)
    for key, values in cognitive_findings(root, py_files, cfg["max-cognitive-complexity"]).items():
        functions.setdefault(key, {}).update(values)
    ratchet = cfg.get("ratchet", {})
    errors = [
        *compare({k: {"lines": v} for k, v in lines.items()}, ratchet.get("lines", {}), "line"),
        *compare(functions, ratchet.get("functions", {}), "complexity"),
        *native_import_errors(root, cfg),
        *untested_modules(root, cfg),
    ]
    return errors, lines, functions


def print_ratchet(lines: dict, functions: dict) -> None:
    print("[tool.fse.guardrails.ratchet.lines]")
    for key, value in sorted(lines.items()):
        print(f'"{key}" = {value}')
    print("\n[tool.fse.guardrails.ratchet.functions]")
    for key, values in sorted(functions.items()):
        inner = ", ".join(f"{metric} = {value}" for metric, value in sorted(values.items()))
        print(f'"{key}" = {{ {inner} }}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--print-ratchet", action="store_true", help="print the findings as TOML")
    args = parser.parse_args(argv)
    errors, lines, functions = check(ROOT)
    if args.print_ratchet:
        print_ratchet(lines, functions)
        return 0
    for error in errors:
        print(error)
    if errors:
        print(f"\n{len(errors)} guardrail errors; see AGENTS.md, Guardrails")
        return 1
    print("guardrails pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
