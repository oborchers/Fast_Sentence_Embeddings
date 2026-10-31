"""The import rules of the guardrails that import-linter cannot check (AGENTS.md,
Guardrails), for tools/check_guardrails.py, which loads this file beside it:

- the imports of and by the native extension: only ``native-importers`` import
  ``fse._native``, the extension imports only ``native-imports`` from fse, and nothing
  imports the forbidden packages of the import-linter contract;
- that only ``hash-importers`` import a hash library (``hash-modules``) and only
  ``log-importers`` the logging modules (``log-modules``: logging, warnings), in Python
  and Cython sources;
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# On code without strings, comments and line continuations: an import statement may
# start anywhere (after ; or the : of a one-line block) and ends at ; or the line end.
_FROM = re.compile(r"\bfrom\s+(\.*[\w.]*)\s+c?import\s+(\([^)]*\)|[^;\n]*)")
_IMPORT = re.compile(r"\bc?import\s+([^;\n]+)")


def _string_end(source: str, start: int) -> int:
    """The index after the string literal that starts at start."""
    quote = (
        source[start : start + 3] if source[start : start + 3] in ('"""', "'''") else source[start]
    )
    end = start + len(quote)
    while True:
        end = source.find(quote, end)
        if end < 0:
            return len(source)
        backslashes = len(source[:end]) - len(source[:end].rstrip("\\"))
        if backslashes % 2 == 0:  # an odd number escapes the quote
            return end + len(quote)
        end += 1


def code_only(source: str) -> str:
    """source with string literals emptied and comments removed, in one pass that knows
    whether a quote or a # starts inside a string or a comment."""
    out, i = [], 0
    while i < len(source):
        if source[i] == "#":
            newline = source.find("\n", i)
            i = len(source) if newline < 0 else newline
        elif source[i] in "'\"":
            out.append('""')
            i = _string_end(source, i)
        else:
            out.append(source[i])
            i += 1
    return "".join(out)


def _names(text: str) -> list[str]:
    """The module names of an import list, aliases and parentheses removed."""
    parts = text.strip().strip("()").split(",")
    return [part.split(" as ")[0].strip() for part in parts if part.strip()]


def extension_imports(source: str) -> list[str]:
    """Every module a Cython source imports or cimports, strings and comments removed."""
    code = code_only(source).replace("\\\n", " ")
    modules = []
    for base, names in _FROM.findall(code):
        if base.rstrip(".") in ("", "fse"):  # from fse import x, from . import x
            modules += ["fse." + name for name in _names(names)]
        else:
            # A relative import inside fse names a module of fse.
            modules.append("fse." + base.lstrip(".") if base.startswith(".") else base)
    # The plain imports, with the from-imports (whose names are no modules) taken out.
    for names in _IMPORT.findall(_FROM.sub("", code)):
        modules += _names(names)
    return modules


def _breaks_rules(module: str, cfg: dict) -> bool:
    if module.split(".", maxsplit=1)[0] in cfg["forbidden"]:
        return True
    return module.startswith("fse") and module not in cfg["native-imports"]


def _extension_import_errors(root: Path, cfg: dict) -> list[str]:
    errors = []
    for path in sorted((root / "src" / "fse").rglob("*")):
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


def _source_imports(root: Path, path: Path) -> list[str] | None:
    """The modules a Python or Cython source of fse imports; None for other files."""
    if path.suffix == ".py":
        module = ".".join(path.relative_to(root / "src").with_suffix("").parts)
        return imported_modules(ast.parse(path.read_text(encoding="utf-8")), module)
    if path.suffix in (".pyx", ".pxd", ".pxi"):
        return extension_imports(path.read_text(encoding="utf-8"))
    return None


def exclusive_import_errors(root: Path, cfg: dict, kind: str) -> list[str]:
    """Modules of fse other than ``<kind>-importers`` that import one of ``<kind>-modules``
    (``hash``: a hash library; ``log``: logging or warnings)."""
    importers, modules = cfg[f"{kind}-importers"], set(cfg[f"{kind}-modules"])
    errors = []
    for path in sorted((root / "src" / "fse").rglob("*")):
        rel = path.relative_to(root).as_posix()
        found = sorted({n.split(".")[0] for n in _source_imports(root, path) or []} & modules)
        if found and rel not in importers:
            errors.append(f"{rel} imports {', '.join(found)}; only {importers} may")
    return errors


def import_errors(root: Path, cfg: dict) -> list[str]:
    """Every error of the import rules of this module."""
    return [
        *native_import_errors(root, cfg),
        *exclusive_import_errors(root, cfg, "hash"),
        *exclusive_import_errors(root, cfg, "log"),
    ]
