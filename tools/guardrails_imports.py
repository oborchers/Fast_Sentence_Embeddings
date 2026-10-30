"""The import rules of the guardrails that import-linter cannot check (AGENTS.md,
Guardrails), for tools/check_guardrails.py, which loads this file beside it:

- the imports of and by the native extension: only ``native-importers`` import
  ``fse._native``, the extension imports only ``native-imports`` from fse, and nothing
  imports the forbidden packages of the import-linter contract;
- that only ``hash-importers`` import a hash library (``hash-modules``) and only
  ``log-importers`` the logging modules (``log-modules``: logging, warnings), in Python
  and Cython sources;
- that the modules the MLX device thread runs (``log-free``) do not import ``fse._log``;
- that the device library of every device backend (``device-libraries``: backend name to
  its top-level modules) is imported, by an import statement or ``importlib``, only by
  that backend's ``src/fse/_backends/_<name>/_load.py``.
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


def log_free_errors(root: Path, cfg: dict) -> list[str]:
    """Modules of ``log-free`` (what the MLX device thread runs) that import fse._log."""
    errors = []
    for rel in cfg.get("log-free", []):
        if not (root / rel).is_file():
            errors.append(f"{rel} of log-free is no file; remove it or name its parts")
            continue
        names = _source_imports(root, root / rel) or []
        if any(n == "fse._log" or n.startswith("fse._log.") for n in names):
            errors.append(f"{rel} imports fse._log; the MLX device thread never logs")
    return errors


def _module_argument(node: ast.Call) -> str | None:
    """The literal module name of an import call, positional or ``name=``."""
    values = [*node.args[:1], *(k.value for k in node.keywords if k.arg == "name")]
    literal = [v.value for v in values if isinstance(v, ast.Constant)]
    return literal[0] if literal and isinstance(literal[0], str) else None


def _import_functions(tree: ast.AST) -> set[str]:
    """``__import__``, ``import_module`` and the names it is imported under."""
    names = {"__import__", "import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            names |= {a.asname or a.name for a in node.names if a.name == "import_module"}
    return names


def dynamic_imports(tree: ast.AST) -> list[str]:
    """The modules imported by a call with a literal name (positional or ``name=``):
    ``x.import_module`` (importlib under any alias), ``import_module`` imported under any
    name, and ``__import__``."""
    functions = _import_functions(tree)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    names = [
        _module_argument(c)
        for c in calls
        if getattr(c.func, "attr", None) == "import_module"
        or getattr(c.func, "id", None) in functions
    ]
    return [name for name in names if name is not None]


def device_import_errors(root: Path, cfg: dict) -> list[str]:
    """Modules of fse other than a device backend's ``_load`` that import its library."""
    libraries = {
        module: (backend, f"src/fse/_backends/_{backend}/_load.py")
        for backend, modules in cfg.get("device-libraries", {}).items()
        for module in modules
    }
    errors = []
    for path in sorted((root / "src" / "fse").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        module = ".".join(path.relative_to(root / "src").with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [*imported_modules(tree, module), *dynamic_imports(tree)]
        for top in sorted({n.split(".")[0] for n in names} & set(libraries)):
            backend, loader = libraries[top]
            if rel != loader:
                errors.append(f"{rel} imports {top}; only {loader} may (the {backend} backend)")
    return errors


def import_errors(root: Path, cfg: dict) -> list[str]:
    """Every error of the import rules of this module."""
    return [
        *native_import_errors(root, cfg),
        *exclusive_import_errors(root, cfg, "hash"),
        *exclusive_import_errors(root, cfg, "log"),
        *log_free_errors(root, cfg),
        *device_import_errors(root, cfg),
    ]
