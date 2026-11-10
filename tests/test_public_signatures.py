"""Permanent guards for caller annotations, including optional adapter sources."""

import ast
import importlib
import inspect
import re
from collections.abc import Mapping
from typing import Annotated, Any, get_args, get_origin

import numpy as np
import pytest

import fse
import fse.typing
from fse._validation import PathSequence, TextBatches, TextInput
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
    origin, args = get_origin(annotation), get_args(annotation)
    if (origin or annotation) is np.ndarray:
        # NumPy 1.26 itself expands NDArray[float32] to ndarray[Any, dtype[float32]].
        # That foreign shape placeholder is not an unknown result dtype.
        dtype = get_args(args[1]) if len(args) > 1 else ()
        return {"dtype"} if not dtype or dtype[0] is Any else set()
    kinds = set()
    for index, arg in enumerate(args):
        if arg is Any and origin in (dict, Mapping):
            kinds.add("mapping_value" if index == 1 else "mapping_key")
        elif isinstance(arg, list):
            for item in arg:
                kinds.update(_any_kinds(item))
        else:
            kinds.update(_any_kinds(arg))
    return kinds


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
    node = ast.parse(source, mode="eval").body
    outer = node.value if isinstance(node, ast.Subscript) else node
    mapping = isinstance(outer, ast.Name) and outer.id in ("dict", "Dict", "Mapping")
    if source == "dict" or "Any" in source:
        return {"framework_dict"} if mapping else {"any"}
    return set()


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
        for node in ast.walk(ast.parse(source, mode="eval")):
            path = entry.imports.get(node.id) if isinstance(node, ast.Name) else None
            if path and path.startswith("fse."):
                module, _, name = path.rpartition(".")
                missing = _private_types(getattr(importlib.import_module(module), name), public)
                if missing:
                    failures.append((entry.name, source, missing))
    return failures


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
    if parameter == "workers":
        return fse.typing.WorkerCount
    if parameter in ("texts", "sentences", "input"):
        return TextInput
    if parameter == "batches":
        if name in (
            "fse.Classifier.predict_batches",
            "fse.Regressor.predict_batches",
            "fse.Heads.predict_batches",
        ):
            return TextBatches | fse.VectorStore
        return TextBatches
    if parameter == "path":
        return fse.typing.PathLike
    if parameter == "paths":
        return PathSequence
    if parameter == "cache_dir":
        return fse.typing.PathLike | None
    if parameter == "extra_files":
        return dict[str, fse.typing.PathLike] | None
    return None


def test_common_input_types(entries):
    failures = []
    for entry in entries:
        if entry.integration:
            owner = entry.name.rsplit(".", 1)[0]
            assert owner in INTEGRATION_COMMON_INPUTS_PENDING, owner
            continue
        if entry.kind == "field":
            continue  # ScoreHead.path is an inference route, not a file-path input.
        for parameter, annotation in entry.hints.items():
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
