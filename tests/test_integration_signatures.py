"""Check written adapter annotations without importing optional frameworks."""

import ast

import pytest

from support import public_surface
from support.public_surface import integration_entries

OWNERS = (
    "langchain_fse.FseEmbeddings",
    "llama_index.embeddings.fse.FseEmbedding",
    "haystack_integrations.components.embedders.fse.FseTextEmbedder",
    "haystack_integrations.components.embedders.fse.FseDocumentEmbedder",
)
INTEGRATION_MODEL_REFERENCES = {
    (
        owner + ".__init__",
        "model_name" if "llama_index" in owner else "model",
    ): "fse.typing.ModelRef"
    for owner in OWNERS
} | {(owner + ".__init__", "workers"): "fse.typing.WorkerCount" for owner in OWNERS}
INTEGRATION_FRAMEWORK_INPUTS = {
    ("langchain_fse.FseEmbeddings." + method, parameter): annotation
    for method, parameter, annotation in (
        ("embed_documents", "texts", "list[str]"),
        ("aembed_documents", "texts", "list[str]"),
        ("embed_query", "text", "str"),
        ("aembed_query", "text", "str"),
    )
} | {
    (owner + "." + method, parameter): annotation
    for owner, parameter, annotation in (
        (OWNERS[2], "text", "str"),
        (OWNERS[3], "documents", "list[haystack.Document]"),
    )
    for method in ("run", "run_async")
}
# Stored fields are normalized, unlike the constructor's broader caller inputs.
INTEGRATION_FIELDS = {
    (OWNERS[1] + ".model_name", "return"): "str",
    (OWNERS[1] + ".workers", "return"): "int | None",
    (OWNERS[1] + ".load_kwargs", "return"): "dict[str, object]",
}


class _Imports(ast.NodeTransformer):
    def __init__(self, imports):
        self.imports = imports

    def visit_Name(self, node):
        return ast.parse(self.imports.get(node.id, node.id), mode="eval").body

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return self.visit(ast.parse(node.value, mode="eval").body)
        return node


def _written(entry, parameter):
    source = entry.written.get(parameter)
    if source is None:
        return None
    return ast.unparse(_Imports(entry.imports).visit(ast.parse(source, mode="eval").body))


def _check(entries, expected):
    used, failures = set(), []
    for entry in entries:
        for parameter in (*entry.parameters, "return"):
            key = (entry.name, parameter)
            if key in expected:
                used.add(key)
                actual = _written(entry, parameter)
                if actual != expected[key]:
                    failures.append((key, actual, expected[key]))
    assert not failures, failures
    assert used == expected.keys(), ("stale entries", expected.keys() - used)


def test_adapter_written_inputs():
    entries = list(integration_entries())
    _check(
        entries, INTEGRATION_MODEL_REFERENCES | INTEGRATION_FRAMEWORK_INPUTS | INTEGRATION_FIELDS
    )
    for entry in entries:
        if entry.kind == "constructor":
            assert (entry.name, "workers") in INTEGRATION_MODEL_REFERENCES


@pytest.mark.parametrize(
    "definition",
    [
        "from wrong import ModelRef\ndef __init__(self, model: ModelRef): ...",
        "from wrong import Other as ModelRef\ndef __init__(self, model: ModelRef): ...",
        "def __init__(self, model): ...",
    ],
)
def test_missing_or_wrong_import_rejected(tmp_path, monkeypatch, definition):
    path = tmp_path / "integrations/demo/src/example/__init__.py"
    path.parent.mkdir(parents=True)
    lines = definition.splitlines()
    imports = "\n".join(lines[:-1])
    path.write_text(imports + "\nclass Adapter:\n    " + lines[-1] + "\n")
    monkeypatch.setattr(public_surface, "ROOT", tmp_path)
    with pytest.raises(AssertionError):
        _check(
            list(integration_entries()),
            {("example.Adapter.__init__", "model"): "fse.typing.ModelRef"},
        )


def test_stale_annotation_entry_rejected():
    with pytest.raises(AssertionError, match="stale entries"):
        _check([], {("removed.Adapter.__init__", "model"): "fse.typing.ModelRef"})


def test_adapter_checks_are_identical():
    files = sorted((public_surface.ROOT / "integrations").glob("*/src/**/_checks.py"))
    assert len(files) == 3
    assert len({file.read_bytes() for file in files}) == 1
