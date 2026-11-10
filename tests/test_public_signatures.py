"""Permanent guards for caller annotations, including optional adapter sources."""

import ast
import collections.abc
import importlib
import inspect
import re
import runpy
from collections.abc import Mapping
from typing import Annotated, Any, get_args, get_origin

import numpy as np
import pytest
from numpy.typing import NDArray

import fse
import fse.typing
from fse._validation import PathSequence, TextBatches, TextInput
from support import public_surface
from support.public_surface import core_entries, integration_entries

# ARRAY_DTYPE_ONLY: model labels, mixed heads and storage options select the dtype.
# Unknown NumPy dtype is permitted only at these exact returns/fields, never whole Any.
ARRAY_DTYPE_ONLY = {
    "fse.Classifier.classes_",
    "fse.Classifier.predict",
    "fse.EncodedBatch.rows",
    "fse.EncodedBatch.vectors",
    "fse.Encoder.encode",
    "fse.Heads.predict",
    "fse.HeadsBatch.outputs",
    "fse.MemorySink.result",
    "fse.MemorySink.target",
    "fse.PredictedBatch.outputs",
    "fse.ScoreBatch.outputs",
    "fse.ShardSink.target",
    "fse.Sink.target",
    "fse.VectorBatch.vectors",
    "fse.WordVectors.stored_ngrams",
    "fse.WordVectors.stored_vectors",
}
# STORE_VIEW_METADATA: the documented records carry heterogeneous metadata values.
STORE_VIEW_METADATA = {
    "fse.VectorStore.members",
    "fse.VectorStore.source",
    "fse.ScoreStore.members",
    "fse.ScoreStore.source",
}
# STATIC_MODEL_CONFIG: the model's open configuration has no closed value schema.
STATIC_MODEL_CONFIG = {"fse.StaticModel.config"}
# Adapter releases are separate. These existing framework dict contracts remain visible
# until DR-325 (Integrationspakete ohne privaten fse-Import und mit Exportliste als 0.2.0 vorbereiten).
INTEGRATION_FRAMEWORK_DICTIONARIES = {
    "llama_index.embeddings.fse.FseEmbedding.to_dict",
    "llama_index.embeddings.fse.FseEmbedding.__getstate__",
    "haystack_integrations.components.embedders.fse.FseTextEmbedder.to_dict",
    "haystack_integrations.components.embedders.fse.FseTextEmbedder.run",
    "haystack_integrations.components.embedders.fse.FseTextEmbedder.run_async",
    "haystack_integrations.components.embedders.fse.FseDocumentEmbedder.to_dict",
    "haystack_integrations.components.embedders.fse.FseDocumentEmbedder.run",
    "haystack_integrations.components.embedders.fse.FseDocumentEmbedder.run_async",
}
# INTEGRATION_LOADER_CONFIGURATION: an open dictionary of pass-through loader options.
INTEGRATION_LOADER_CONFIGURATION = {"llama_index.embeddings.fse.FseEmbedding.load_kwargs"}
# INTEGRATION_RECONSTRUCTION_RETURN: existing Haystack factory annotations await that release.
INTEGRATION_RECONSTRUCTION_RETURN = {
    "haystack_integrations.components.embedders.fse.FseTextEmbedder.from_dict",
    "haystack_integrations.components.embedders.fse.FseDocumentEmbedder.from_dict",
}
# STATIC_MODEL_UNSET_DEFAULT: omission uses the saved setting, unlike explicit None.
STATIC_MODEL_UNSET_DEFAULT = {
    ("fse.StaticModel.encode", "max_length"),
    ("fse.StaticModel.from_pretrained", "max_length"),
}


def _plain(annotation):
    return get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation


def _any_kinds(annotation):
    annotation = _plain(annotation)
    if annotation is Any:
        return {"any"}
    bare = _bare_unknowns(annotation)
    if bare is not None:
        return bare
    origin, args = get_origin(annotation), get_args(annotation)
    dtype = _array_unknowns(annotation, origin, args)
    if dtype is not None:
        return dtype
    return _nested_unknowns(origin, args)


def _array_unknowns(annotation, origin, args):
    if origin is NDArray or annotation is NDArray:
        # NumPy 2.5 preserves NDArray as a runtime TypeAliasType.
        scalar = args[0] if args else Any
    elif (origin or annotation) is np.ndarray:
        # NumPy 1.26 expands NDArray to ndarray[Any, dtype[scalar]].
        dtype = get_args(args[1]) if len(args) > 1 else ()
        scalar = dtype[0] if dtype else Any
    else:
        return None
    return {"dtype"} if _any_kinds(scalar) else set()


def _nested_unknowns(origin, args):
    kinds = set()
    for index, arg in enumerate(args):
        if arg is Any and origin in (dict, Mapping):
            kinds.add("mapping_value" if index == 1 else "mapping_key")
        else:
            items = arg if isinstance(arg, list) else [arg]
            for item in items:
                kinds.update(_any_kinds(item))
    return kinds


def _bare_unknowns(annotation):
    origin = get_origin(annotation) or annotation
    if origin in (dict, Mapping) and not get_args(annotation):
        return {"mapping_key", "mapping_value"}
    names = ("Sequence", "Iterable", "Iterator", "Generator", "Callable")
    bare = (list, tuple, set, *(getattr(collections.abc, name) for name in names))
    return {"any"} if origin in bare and not get_args(annotation) else None


def _name(entry):
    return re.sub(r"\[\d+\]$", "", entry.name)


@pytest.fixture(scope="module")
def entries():
    return [*core_entries(), *integration_entries()]


def _unknown_results(entry):
    if not entry.integration:
        return _any_kinds(entry.hints["return"]) if "return" in entry.hints else {"missing"}
    source = entry.written.get("return")
    if source is None:
        return {"missing"}
    node = _expand_integration(entry, source)
    outer = node.value if isinstance(node, ast.Subscript) else node
    mapping = isinstance(outer, ast.Name) and outer.id in ("dict", "Dict", "Mapping")
    has_any = _unknown_ast(node)
    if (isinstance(node, ast.Name) and mapping) or has_any:
        return {"framework_dict"} if mapping else {"any"}
    return set()


class _ExpandAliases(ast.NodeTransformer):
    def __init__(self, entry):
        self.entry = entry
        self.active = set()

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return self.visit(ast.parse(node.value, mode="eval").body)
        return node

    def visit_Subscript(self, node):
        node.value = self.visit(node.value)
        if not (isinstance(node.value, ast.Name) and node.value.id == "Literal"):
            node.slice = self.visit(node.slice)
        return node

    def visit_Name(self, node):
        module, _, name = self.entry.imports.get(node.id, "").rpartition(".")
        if module in ("typing", "typing_extensions"):
            return ast.Name(id=name, ctx=ast.Load())
        if node.id in self.entry.aliases and node.id not in self.active:
            self.active.add(node.id)
            result = self.visit(ast.parse(self.entry.aliases[node.id], mode="eval").body)
            self.active.remove(node.id)
            return result
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and self.entry.imports.get(node.value.id) in (
            "typing",
            "typing_extensions",
        ):
            return ast.Name(id=node.attr, ctx=ast.Load())
        return node


def _expand_integration(entry, source):
    return _ExpandAliases(entry).visit(ast.parse(source, mode="eval").body)


def _unknown_ast(node):
    parts = list(ast.walk(node))
    parameterized = {id(part.value) for part in parts if isinstance(part, ast.Subscript)}
    bare = {
        "dict",
        "Dict",
        "Mapping",
        "list",
        "List",
        "tuple",
        "Tuple",
        "Sequence",
        "Iterable",
        "Iterator",
        "Generator",
        "Callable",
        "set",
        "Set",
    }
    return any(
        isinstance(part, ast.Name)
        and (part.id == "Any" or (part.id in bare and id(part) not in parameterized))
        for part in parts
    )


def _allowed_results(entry):
    name = _name(entry)
    if entry.integration:
        if name in INTEGRATION_RECONSTRUCTION_RETURN:
            return {"missing"}
        if name in INTEGRATION_FRAMEWORK_DICTIONARIES | INTEGRATION_LOADER_CONFIGURATION:
            return {"framework_dict"}
        return set()
    allowed = {"dtype"} if name in ARRAY_DTYPE_ONLY else set()
    if name in STORE_VIEW_METADATA | STATIC_MODEL_CONFIG:
        allowed.add("mapping_value")
    return allowed


def test_public_results_have_only_named_unknown_types(entries):
    used, failures = set(), []
    for entry in entries:
        if entry.kind == "constructor":
            continue  # Constructor calls return their class, not __init__'s None.
        kinds = _unknown_results(entry)
        if kinds and kinds <= _allowed_results(entry):
            used.add(_name(entry))
        elif kinds:
            failures.append((entry.name, kinds))
    assert not failures, failures
    expected = (
        ARRAY_DTYPE_ONLY
        | STORE_VIEW_METADATA
        | STATIC_MODEL_CONFIG
        | INTEGRATION_FRAMEWORK_DICTIONARIES
        | INTEGRATION_LOADER_CONFIGURATION
        | INTEGRATION_RECONSTRUCTION_RETURN
    )
    assert used == expected, ("stale exceptions", expected - used)


def _private_types(annotation, public):
    annotation = _plain(annotation)
    found = set()
    if (
        inspect.isclass(annotation)
        and getattr(annotation, "__module__", "").startswith("fse.")
        and annotation not in public
    ):
        found.add(annotation.__module__ + "." + annotation.__qualname__)
    for arg in get_args(annotation):
        for item in arg if isinstance(arg, list) else [arg]:
            found.update(_private_types(item, public))
    return found


def _core_private_types(entry, public):
    used, failures = set(), []
    for parameter, annotation in entry.hints.items():
        missing = _private_types(annotation, public)
        key = (_name(entry), parameter)
        if missing and key in STATIC_MODEL_UNSET_DEFAULT:
            assert missing == {"fse._encoder._model2vec_rules.Unset"}
            used.add(key)
        elif missing:
            failures.append((entry.name, parameter, missing))
    return used, failures


def _integration_private_types(entry, public):
    failures = []
    for source in entry.written.values():
        for node in ast.walk(_expand_integration(entry, source)):
            if isinstance(node, ast.Name) and node.id in entry.private_classes:
                failures.append((entry.name, source, node.id))
            path = _import_path(entry, node)
            if path and path.startswith("fse."):
                module, _, name = path.rpartition(".")
                missing = _private_types(getattr(importlib.import_module(module), name), public)
                if missing:
                    failures.append((entry.name, source, missing))
    return failures


def _import_path(entry, node):
    if isinstance(node, ast.Name):
        return entry.imports.get(node.id)
    if isinstance(node, ast.Attribute):
        parent = _import_path(entry, node.value)
        return parent + "." + node.attr if parent else None
    return None


def test_annotation_classes_are_publicly_reachable(entries):
    public = {
        value
        for module in (fse, fse.typing)
        for name in module.__all__
        if inspect.isclass(value := getattr(module, name))
    }
    used, failures = set(), []
    for entry in entries:
        allowed, missing = _core_private_types(entry, public)
        used.update(allowed)
        failures.extend(missing)
        if entry.integration:
            failures.extend(_integration_private_types(entry, public))
    assert not failures, failures
    assert used == STATIC_MODEL_UNSET_DEFAULT, ("stale exceptions", used)


# Adapter inputs retain their own framework contracts until their separate release.
INTEGRATION_COMMON_INPUTS_PENDING = {
    "langchain_fse.FseEmbeddings",
    "llama_index.embeddings.fse.FseEmbedding",
    "haystack_integrations.components.embedders.fse.FseTextEmbedder",
    "haystack_integrations.components.embedders.fse.FseDocumentEmbedder",
}
MODEL_REFERENCES = {
    ("fse.Encoder.from_pretrained", "repo_or_path"),
    ("fse.Classifier.from_pretrained", "repo_or_path"),
    ("fse.Regressor.from_pretrained", "repo_or_path"),
    ("fse.StaticModel.from_pretrained", "path"),
}


def _expected_input(name, parameter):
    if (name, parameter) in MODEL_REFERENCES:
        return fse.typing.ModelRef
    if parameter == "batches":
        return _batch_input(name)
    common = {
        "workers": fse.typing.WorkerCount,
        "texts": TextInput,
        "sentences": TextInput,
        "input": TextInput,
        "path": fse.typing.PathLike,
        "paths": PathSequence,
        "cache_dir": fse.typing.PathLike | None,
        "extra_files": dict[str, fse.typing.PathLike] | None,
    }
    return common.get(parameter)


def _batch_input(name):
    predictions = {
        "fse.Classifier.predict_batches",
        "fse.Regressor.predict_batches",
        "fse.Heads.predict_batches",
    }
    return TextBatches | fse.VectorStore if name in predictions else TextBatches


def test_common_input_types(entries):
    failures = []
    for entry in entries:
        if entry.integration:
            owner = entry.name.rsplit(".", 1)[0]
            assert owner in INTEGRATION_COMMON_INPUTS_PENDING, owner
            continue
        if entry.kind == "field":
            continue  # ScoreHead.path is an inference route, not a file-path input.
        for parameter in entry.parameters:
            annotation = entry.hints.get(parameter)
            expected = _expected_input(_name(entry), parameter)
            if expected is not None and _plain(annotation) != _plain(expected):
                failures.append((entry.name, parameter, annotation, expected))
            if (_name(entry), parameter) in MODEL_REFERENCES:
                assert entry.written[parameter] == "ModelRef", entry.name
    assert not failures, failures


def test_scalar_text_and_inference_paths_keep_their_meaning(entries):
    by_name = {entry.name: entry for entry in entries}
    # SCALAR_TOKENIZER_TEXT is one string, not the broad collection/stream input.
    assert by_name["fse.typing.Tokenizer.encode"].hints["sequence"] is str
    # SEMANTIC_INFERENCE_PATH describes arithmetic, not a file input.
    assert by_name["fse.typing.ScoreHead.path"].hints["return"] is str
    routes = get_args(by_name["fse.Heads.paths"].hints["return"])[1]
    assert set(get_args(routes)) == {"folded", "normalized", "unfolded", "standalone"}


@pytest.mark.parametrize(
    ("definition", "unknown"),
    [
        ("from typing import Any as Dynamic\nResult = Dynamic", True),
        ("Result = dict", True),
        ('from typing import Any\nResult = "Any"', True),
        ('from typing import Any\nResult = list["Any"]', True),
        ('from typing import Literal\nResult = Literal["Any"]', False),
    ],
)
def test_adapter_aliases_cannot_hide_unknown_results(tmp_path, monkeypatch, definition, unknown):
    file = tmp_path / "integrations/demo/src/example/__init__.py"
    file.parent.mkdir(parents=True)
    file.write_text(definition + "\nclass Adapter:\n    def run(self) -> Result: ...\n")
    monkeypatch.setattr(public_surface, "ROOT", tmp_path)
    (entry,) = integration_entries()
    assert bool(_unknown_results(entry) - _allowed_results(entry)) is unknown


@pytest.mark.parametrize("annotation", ["_Hidden", '"_Hidden"', 'list["_Hidden"]'])
def test_adapter_aliases_cannot_hide_private_classes(tmp_path, monkeypatch, annotation):
    file = tmp_path / "integrations/demo/src/example/__init__.py"
    file.parent.mkdir(parents=True)
    file.write_text(
        f"class _Hidden: pass\nResult = {annotation}\nclass Adapter:\n    def run(self) -> Result: ...\n"
    )
    monkeypatch.setattr(public_surface, "ROOT", tmp_path)
    (entry,) = integration_entries()
    assert _integration_private_types(entry, set())


def test_core_forward_references_resolve(tmp_path):
    file = tmp_path / "annotations.py"
    file.write_text('from typing import Any\ndef call() -> list["Any"]: ...\n')
    function = runpy.run_path(str(file))["call"]
    (entry,) = public_surface._function_entries("example.call", function, "method")
    assert _unknown_results(entry) == {"any"}


@pytest.mark.parametrize("parameter", ["workers", "path"])
def test_common_parameter_without_annotation_is_visible(tmp_path, parameter):
    file = tmp_path / "annotations.py"
    file.write_text(f"def call({parameter}): ...\n")
    function = runpy.run_path(str(file))["call"]
    (entry,) = public_surface._function_entries("example.call", function, "method")
    assert entry.parameters == (parameter,)
    assert entry.hints.get(parameter) != _expected_input(entry.name, parameter)


def test_adapter_overloads_are_all_visible(tmp_path, monkeypatch):
    file = tmp_path / "integrations/demo/src/example/__init__.py"
    file.parent.mkdir(parents=True)
    file.write_text(
        "from typing import Any, overload\nclass Adapter:\n"
        "    @overload\n    def run(self, x: int) -> Any: ...\n"
        "    @overload\n    def run(self, x: str) -> str: ...\n"
        "    def run(self, x) -> str: ...\n"
    )
    monkeypatch.setattr(public_surface, "ROOT", tmp_path)
    entries = list(integration_entries())
    assert len(entries) == 2
    assert [_unknown_results(entry) for entry in entries] == [{"any"}, set()]


def test_core_quoted_private_class_is_not_public(tmp_path):
    file = tmp_path / "annotations.py"
    file.write_text('class _Hidden: pass\ndef call() -> "_Hidden": ...\n')
    namespace = runpy.run_path(str(file))
    namespace["_Hidden"].__module__ = "fse._example"
    (entry,) = public_surface._function_entries("example.call", namespace["call"], "method")
    assert _private_types(entry.hints["return"], set()) == {"fse._example._Hidden"}
