"""Filename matches cannot substitute for bound artifact provenance."""

import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "benchmarks/quality/artifact_manifest.py"
SPEC = importlib.util.spec_from_file_location("artifact_manifest_test", PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_header(path, dtype, fingerprint, revision="pinned"):
    value = {
        "__metadata__": {
            "fse.dtype": dtype,
            "fse.fingerprint": fingerprint,
            "fse.source": json.dumps({"repo": "example/model", "revision": revision}),
        }
    }
    data = json.dumps(value).encode()
    path.write_bytes(struct.pack("<Q", len(data)) + data)


def inputs(tmp_path):
    entries = []
    for dtype in ("float32", "float16", "int8"):
        path = tmp_path / f"model-{dtype}.safetensors"
        write_header(path, dtype, dtype + "-fingerprint")
        entries.append(
            {
                "artifact": path.name,
                "source": "example/model",
                "revision": "pinned",
                "table_dtype": dtype,
                "file_bytes": path.stat().st_size,
            }
        )
    record = {
        "source": "example/model",
        "revision": "pinned",
        "artifacts": [entry["artifact"] for entry in entries],
    }
    (tmp_path / "sources.jsonl").write_text(json.dumps(record) + "\n")
    return {"artifacts": entries}


def test_bound_quantized_artifacts_identify_their_float32_parent(tmp_path):
    bound = MODULE.bind(tmp_path, inputs(tmp_path), {"example/model": "pinned"})
    assert {entry["float32_parent_fingerprint"] for entry in bound["artifacts"]} == {
        "float32-fingerprint"
    }


def test_embedded_revision_must_match_preparation_not_filename(tmp_path):
    artifacts = inputs(tmp_path)
    write_header(tmp_path / "model-int8.safetensors", "int8", "int8-fingerprint", revision="other")
    with pytest.raises(ValueError, match="embedded source differs"):
        MODULE.bind(tmp_path, artifacts, {"example/model": "pinned"})


def test_loaded_fingerprint_must_match_bound_identity(tmp_path):
    bound = MODULE.bind(tmp_path, inputs(tmp_path), {"example/model": "pinned"})
    (tmp_path / "artifact-identities.json").write_text(json.dumps(bound))
    encoder = SimpleNamespace(fingerprint="stale-file", word_vectors=SimpleNamespace(dtype="int8"))
    with pytest.raises(ValueError, match="loaded artifact differs"):
        MODULE.verified_source(tmp_path / "model-int8.safetensors", encoder)
