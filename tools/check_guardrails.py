"""Check the guardrails of the repository against the ratchet in pyproject.toml.

The guardrails ([tool.fse.guardrails]; AGENTS.md, Guardrails) keep the code reviewable:

- at most ``max-lines`` lines per hand-written source file under ``paths``;
- the ruff complexity limits of [tool.ruff.lint.mccabe] and [tool.ruff.lint.pylint]
  (C901, PLR0911, PLR0912, PLR0913, PLR0915, PLR0917) and a cognitive complexity of at
  most ``max-cognitive-complexity`` per function (complexipy);
- the imports of and by the native extension, which import-linter cannot see: only
  ``native-importers`` import ``fse._native``, the extension imports only
  ``native-imports`` from fse, and nothing imports the forbidden packages of the
  import-linter contract;
- a test file for every module of fse.

What exceeded a limit when the guardrails came is the ratchet. It only shrinks: a new
finding fails, a value above its entry fails, a value below its entry or within the
limit fails until the entry is lowered or removed. Inline suppressions (``noqa``,
``complexipy: ignore``) are ignored. With ``--base REF``, the ratchet must not have
new entries or higher numbers than in the pyproject.toml of REF.

    uv run python tools/check_guardrails.py
    uv run python tools/check_guardrails.py --base origin/master
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
_RUFF += ["--no-cache", "--quiet", "--ignore-noqa"]
_VALUE = re.compile(r"\((\d+) > \d+\)")
_STRINGS = re.compile(
    r'"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\'|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\''
)
_COMMENT = re.compile(r"#[^\n]*")
_FROM = re.compile(r"^\s*from\s+(\.*[\w.]*)\s+c?import\s+\(?([^)\n]*)", re.MULTILINE)
_IMPORT = re.compile(r"^\s*c?import\s+([^\n]+)", re.MULTILINE)


class ToolError(Exception):
    """A tool the checker runs failed; the message says why."""


def load_pyproject(root: Path) -> dict:
    with open(root / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def load_config(root: Path) -> dict:
    """The [tool.fse.guardrails] table, with the forbidden packages of import-linter."""
    tool = load_pyproject(root)["tool"]
    cfg = dict(tool["fse"]["guardrails"])
    forbidden = [c for c in tool["importlinter"]["contracts"] if c["type"] == "forbidden"]
    cfg["forbidden"] = sorted({m for c in forbidden for m in c["forbidden_modules"]})
    return cfg


def _run(command: list[str], root: Path) -> str:
    try:
        return subprocess.run(command, cwd=root, capture_output=True, check=True).stdout.decode()
    except FileNotFoundError as err:
        raise ToolError(f"{command[0]} is not installed: {err}") from None
    except subprocess.CalledProcessError as err:
        raise ToolError(f"{' '.join(command[:4])} failed:\n{err.stderr.decode()}") from None


def source_files(root: Path, paths: list[str]) -> list[str]:
    """Tracked and untracked, not ignored source files under paths, relative to root."""
    out = _run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *paths], root
    )
    files = {f for f in out.split("\0") if Path(f).suffix in SOURCE_SUFFIXES}
    return sorted(f for f in files if (root / f).is_file())


def long_files(
    root: Path, files: list[str], max_lines: int, generated: list[str]
) -> dict[str, int]:
    """The line count of every file longer than max_lines, generated files excepted."""
    found = {}
    for name in files:
        if name not in generated:
            lines = len((root / name).read_text(encoding="utf-8").splitlines())
            if lines > max_lines:
                found[name] = lines
    return found


def qualnames(source: str) -> dict[int, str]:
    """The qualified name (Class.method) of every function by the lines of its def and
    decorators, also for functions inside if, try or with blocks."""
    names: dict[int, str] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                name = f"{prefix}{child.name}"
                for line in [child.lineno, *(d.lineno for d in child.decorator_list)]:
                    names[line] = name
                visit(child, name + ".")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(ast.parse(source), "")
    return names


def ruff_findings(root: Path, py_files: list[str]) -> dict[str, dict[str, int]]:
    """The ruff complexity findings by path::qualname and rule, noqa ignored."""
    if not py_files:
        return {}
    out = _run([*_RUFF, "--select", ",".join(RUFF_RULES), *py_files], root)
    found: dict[str, dict[str, int]] = {}
    names: dict[str, dict[int, str]] = {}
    for item in json.loads(out or "[]"):
        path = Path(item["filename"]).resolve().relative_to(root).as_posix()
        if path not in names:
            names[path] = qualnames((root / path).read_text(encoding="utf-8"))
        row = item["location"]["row"]
        match = _VALUE.search(item["message"])
        key = f"{path}::{names[path].get(row, f'line {row}')}"
        found.setdefault(key, {})[item["code"]] = int(match.group(1)) if match else 0
    return found


def cognitive_findings(root: Path, py_files: list[str], limit: int) -> dict[str, dict[str, int]]:
    """The functions whose cognitive complexity exceeds limit, by path::qualname."""
    try:
        # Imported here: the tests of this tool run without the lint group.
        file_complexity = importlib.import_module("complexipy").file_complexity
    except ImportError:
        raise ToolError("complexipy is not installed; run `uv sync --group lint`") from None
    found: dict[str, dict[str, int]] = {}
    for path in py_files:
        for function in file_complexity(
            str(root / path), no_ignore=True, with_plans=False
        ).functions:
            if function.complexity > limit:
                found[f"{path}::{function.name.replace('::', '.')}"] = {
                    "cognitive": function.complexity
                }
    return found


def extension_imports(source: str) -> list[str]:
    """Every module a Cython source imports or cimports, strings and comments removed."""
    code = _COMMENT.sub("", _STRINGS.sub('""', source))
    modules = []
    for base, names in _FROM.findall(code):
        if base.rstrip(".") in ("", "fse"):  # from fse import x, from . import x
            modules += ["fse." + n.split(" as ")[0].strip() for n in names.split(",") if n.strip()]
        else:
            # A relative import inside fse names a module of fse.
            modules.append("fse." + base.lstrip(".") if base.startswith(".") else base)
    for names in _IMPORT.findall(code):
        modules += [n.split(" as ")[0].strip() for n in names.split(",") if n.strip()]
    return modules


def _breaks_rules(module: str, cfg: dict) -> bool:
    if module.split(".", maxsplit=1)[0] in cfg["forbidden"]:
        return True
    return module.startswith("fse") and module not in cfg["native-imports"]


def _extension_import_errors(root: Path, cfg: dict) -> list[str]:
    errors = []
    for path in sorted((root / "src" / "fse").rglob("_native*")):
        if path.suffix in (".pyx", ".pxd", ".pxi"):
            for module in extension_imports(path.read_text(encoding="utf-8")):
                if _breaks_rules(module, cfg):
                    allowed = cfg["native-imports"]
                    errors.append(
                        f"{path.relative_to(root)} imports {module}; from fse only {allowed}"
                    )
    return errors


def _absolute(node: ast.ImportFrom, package: str) -> str:
    """The module of a from-import, with a relative one resolved against package."""
    base = node.module or ""
    if not node.level:
        return base
    parent = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
    return f"{parent}.{base}" if base else parent


def imported_modules(tree: ast.AST, module: str) -> list[str]:
    """The absolute names of every module and imported name of a Python module of fse."""
    package = module.rsplit(".", 1)[0]
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = _absolute(node, package)
            names += [base, *(f"{base}.{alias.name}" for alias in node.names)]
    return names


def _native_importer_errors(root: Path, cfg: dict) -> list[str]:
    errors = []
    for path in sorted((root / "src" / "fse").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        module = ".".join(path.relative_to(root / "src").with_suffix("").parts)
        names = imported_modules(ast.parse(path.read_text(encoding="utf-8")), module)
        if rel not in cfg["native-importers"] and any(n.startswith("fse._native") for n in names):
            errors.append(f"{rel} imports fse._native; only {cfg['native-importers']} may")
    return errors


def native_import_errors(root: Path, cfg: dict) -> list[str]:
    """Imports of and by the native extension that break the rules."""
    return [*_extension_import_errors(root, cfg), *_native_importer_errors(root, cfg)]


def test_file_for(path: Path, package: Path) -> str:
    """tests/test_<parts>.py for a module: fse/_a.py is test_a, fse/_b/_c.py is test_b_c,
    fse/_b/__init__.py is test_b, fse/__init__.py is test_init."""
    parts = [
        "init" if p == "__init__" else p.lstrip("_")
        for p in path.relative_to(package).with_suffix("").parts
    ]
    if parts[-1] == "init" and len(parts) > 1:
        parts.pop()
    return "tests/test_" + "_".join(parts) + ".py"


def untested_modules(root: Path, cfg: dict) -> list[str]:
    """Modules of fse, in subpackages too, without their test file."""
    package = root / "src" / "fse"
    missing = []
    for path in sorted(package.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        test = test_file_for(path, package)
        if rel not in cfg["untested-modules"] and not (root / test).exists():
            missing.append(f"{rel} has no {test}")
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
            grew = value > allowed
            fix = "split or simplify it" if grew else f"lower its ratchet entry to {value}"
            errors.append(
                f"{key}: {metric} {'grew' if grew else 'improved'} from {allowed} to {value}; {fix}"
            )
    return errors


def _as_metrics(entry: dict | int | None) -> dict:
    if entry is None:
        return {}
    return entry if isinstance(entry, dict) else {"lines": entry}


def ratchet_growth(ratchet: dict, base: dict) -> list[str]:
    """Entries of ratchet that base does not have, or with higher numbers than in base."""
    errors = []
    for table in ("lines", "functions"):
        old = base.get(table, {})
        for key, entry in ratchet.get(table, {}).items():
            before = _as_metrics(old.get(key))
            for metric, value in _as_metrics(entry).items():
                if metric not in before or value > before[metric]:
                    errors.append(
                        f"ratchet {key}: {metric} {value} is new or higher than on the base"
                    )
    return errors


def base_ratchet(root: Path, ref: str) -> dict:
    """The ratchet in the pyproject.toml of git ref, empty before the guardrails came."""
    tool = tomllib.loads(_run(["git", "show", f"{ref}:pyproject.toml"], root))["tool"]
    return tool.get("fse", {}).get("guardrails", {}).get("ratchet", {})


def check(root: Path, base: str | None = None) -> tuple[list[str], dict, dict]:
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
    old = base_ratchet(root, base) if base else {}
    return [*errors, *(ratchet_growth(ratchet, old) if old else [])], lines, functions


def print_ratchet(lines: dict, functions: dict) -> None:
    print("[tool.fse.guardrails.ratchet.lines]")
    for key, value in sorted(lines.items()):
        print(f'"{key}" = {value}')
    print("\n[tool.fse.guardrails.ratchet.functions]")
    for key, values in sorted(functions.items()):
        inner = ", ".join(f"{metric} = {value}" for metric, value in sorted(values.items()))
        print(f'"{key}" = {{ {inner} }}')


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    """0 if the guardrails pass, 1 for guardrail errors, 2 if a tool failed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--print-ratchet", action="store_true", help="print the findings as TOML")
    parser.add_argument("--base", help="a git ref whose ratchet this one must not exceed")
    args = parser.parse_args(argv)
    try:
        errors, lines, functions = check(root, args.base)
    except ToolError as err:
        print(f"check_guardrails: {err}", file=sys.stderr)
        return 2
    if args.print_ratchet:
        print_ratchet(lines, functions)
        return 0
    for error in errors:
        print(error)
    print(
        f"\n{len(errors)} guardrail errors; see AGENTS.md, Guardrails"
        if errors
        else "guardrails pass"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
