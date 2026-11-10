"""Read caller annotations without importing optional integration frameworks."""

import ast
import importlib
import inspect
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import get_type_hints

import fse
import fse.typing

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_METHODS = {
    "__enter__",
    "__exit__",
    "__getitem__",
    "__iter__",
    "__len__",
    "__getstate__",
    "__setstate__",
}


@dataclass
class Entry:
    name: str
    kind: str
    hints: dict
    written: dict[str, str]
    imports: dict[str, str]
    integration: bool = False
    parameters: tuple[str, ...] = ()
    aliases: dict[str, str] = field(default_factory=dict)
    private_classes: set[str] = field(default_factory=set)


def _public_method(name):
    return not name.startswith("_") or name in PROTOCOL_METHODS or name == "__init__"


def _arguments(node):
    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    return {arg.arg: arg.annotation for arg in args if arg.arg not in ("self", "cls")}


@cache
def _source_nodes(file):
    groups = []

    def scan(body, prefix=""):
        for node in body:
            if isinstance(node, ast.ClassDef):
                scan(node.body, prefix + node.name + ".")
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                groups.append((prefix + node.name, node))

    scan(ast.parse(Path(file).read_text()).body)
    return groups


def _caller_nodes(function):
    """Use code positions: preserving a validation title can rename __qualname__."""
    raw = inspect.unwrap(function)
    file = inspect.getsourcefile(raw)
    assert file is not None, function
    candidates = _source_nodes(file)
    qualname, implementation = next(
        (name, node)
        for name, node in candidates
        if node.name == raw.__name__
        and min([node.lineno, *[d.lineno for d in node.decorator_list]])
        <= raw.__code__.co_firstlineno
        <= node.end_lineno
    )
    overloads = [
        node
        for name, node in candidates
        if name == qualname
        and any(isinstance(d, ast.Name) and d.id == "overload" for d in node.decorator_list)
    ]
    return raw, overloads or [implementation]


def _function_entries(name, function, kind):
    raw, nodes = _caller_nodes(function)
    public = {
        name: getattr(module, name) for module in (fse, fse.typing) for name in module.__all__
    }
    namespace = {**public, **raw.__globals__}
    for index, node in enumerate(nodes):
        annotations = _arguments(node)
        if kind != "constructor":
            annotations["return"] = node.returns
        written = {
            key: ast.unparse(value) for key, value in annotations.items() if value is not None
        }
        # Only repository annotation expressions are evaluated, never caller input.
        holder = type("CallerAnnotations", (), {"__annotations__": written})
        hints = get_type_hints(holder, globalns=namespace, include_extras=True)
        suffix = f"[{index}]" if len(nodes) > 1 else ""
        yield Entry(name + suffix, kind, hints, written, {}, parameters=tuple(_arguments(node)))


def core_entries():
    """Public exports, their own/inherited fse methods, properties and result fields."""
    for module in (fse, fse.typing):
        for name in module.__all__:
            value = getattr(module, name)
            prefix = module.__name__ + "." + name
            if inspect.isfunction(value):
                yield from _function_entries(prefix, value, "function")
            elif inspect.isclass(value) and value.__module__.startswith("fse."):
                yield from _class_entries(prefix, value)


def _class_entries(prefix, cls):
    if issubclass(cls, BaseException):
        return  # Exception methods inherited from builtins are not fse implementations.
    yield from _method_entries(prefix, cls)
    for name, hint in _field_hints(cls).items():
        yield Entry(prefix + "." + name, "field", {"return": hint}, {}, {})


def _method_entries(prefix, cls):
    for name, value in inspect.getmembers_static(cls):
        if not _public_method(name):
            continue
        kind = "constructor" if name == "__init__" else "method"
        if isinstance(value, classmethod | staticmethod):
            value = value.__func__
        if isinstance(value, property):
            kind, value = "property", value.fget
        if not inspect.isfunction(value) or not value.__module__.startswith("fse."):
            continue  # Generated constructors and foreign mixins have no own Python API body.
        if not Path(inspect.unwrap(value).__code__.co_filename).is_file():
            assert kind == "constructor", prefix + "." + name
            continue
        yield from _function_entries(prefix + "." + name, value, kind)


def _field_hints(cls):
    resolved = {}
    for base in reversed(cls.__mro__):
        annotations = {
            name: hint
            for name, hint in inspect.get_annotations(base, eval_str=False).items()
            if not name.startswith("_")
        }
        if annotations:
            # Resolve only public fields, in their declaring class's namespace.
            # Private fields can intentionally reference names from another module.
            declared_fields = type("DeclaredFields", (), {"__annotations__": annotations})
            namespace = vars(importlib.import_module(base.__module__))
            resolved.update(get_type_hints(declared_fields, globalns=namespace))
    return resolved


def _imports(tree):
    return dict(binding for node in tree.body for binding in _import_bindings(node))


def _import_bindings(node):
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            yield alias.asname or root, alias.name if alias.asname else root
    elif isinstance(node, ast.ImportFrom) and node.module:
        for alias in node.names:
            yield alias.asname or alias.name, node.module + "." + alias.name


def _integration_members(cls, classes):
    members = {}
    for base in cls.bases:
        if isinstance(base, ast.Name) and base.id in classes:
            members.update(_integration_members(classes[base.id], classes))
    own = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _public_method(node.name):
            own.setdefault(node.name, []).append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and not node.target.id.startswith("_")
        ):
            own[node.target.id] = [node]
    return members | own


def _exports(tree, classes):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    return [name for name in classes if not name.startswith("_")]


def _integration_entry(full_name, node, imports):
    if isinstance(node, ast.AnnAssign):
        return Entry(
            full_name, "field", {}, {"return": ast.unparse(node.annotation)}, imports, True
        )
    kind = "constructor" if node.name == "__init__" else "method"
    annotations = _arguments(node)
    if kind != "constructor":
        annotations["return"] = node.returns
    written = {key: ast.unparse(value) for key, value in annotations.items() if value is not None}
    return Entry(full_name, kind, {}, written, imports, True, tuple(_arguments(node)))


def _aliases(tree):
    aliases = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            aliases[node.targets[0].id] = ast.unparse(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            aliases[node.target.id] = ast.unparse(node.value)
    return aliases


def _integration_module(file):
    tree = ast.parse(file.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    parts = file.relative_to(next(p for p in file.parents if p.name == "src")).parts[:-1]
    prefix = ".".join(parts)
    for name in _exports(tree, classes):
        assert name in classes, f"Extend surface enumeration for moved export {prefix}.{name}"
        yield from _integration_class_entries(prefix + "." + name, classes[name], classes, tree)


def _integration_class_entries(prefix, cls, classes, tree):
    imports, aliases = _imports(tree), _aliases(tree)
    private = {name for name in classes if name.startswith("_")}
    for member, nodes in _integration_members(cls, classes).items():
        overloads = [node for node in nodes if _overloaded(node, imports)]
        for index, node in enumerate(overloads or nodes):
            suffix = f"[{index}]" if overloads else ""
            entry = _integration_entry(prefix + "." + member + suffix, node, imports)
            entry.aliases, entry.private_classes = aliases, private
            yield entry


def _overloaded(node, imports):
    decorators = getattr(node, "decorator_list", [])
    return any(
        imports.get(ast.unparse(decorator), ast.unparse(decorator)).endswith("overload")
        for decorator in decorators
    )


def integration_entries():
    """Enumerate local exports and inherited adapter methods, without optional imports."""
    for file in sorted((ROOT / "integrations").glob("*/src/**/__init__.py")):
        yield from _integration_module(file)
