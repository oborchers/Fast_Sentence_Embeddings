"""Rebuild publication payloads from local prepared tables, never from live cards."""

from __future__ import annotations

import io
import json
import re

from cards import render

import fse
from fse._container import SafetensorsContainer
from fse._container._hash import file_digests
from fse._encoder import _folder

PUBLISHED_SHA256 = "04a4453fecdff450f4bc48b1e46710f4bd12c29f38b411ada405442bb6aade34"
ATTRIBUTES = b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n"


def sha256(path):
    before = path.stat()
    with path.open("rb") as stream:
        result = file_digests(stream, before.st_size)[0]
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"file changed while hashing: {path}")
    return result


def bytes_sha256(data):
    return file_digests(io.BytesIO(data), len(data))[0]


def read_json(path):
    return json.loads(path.read_bytes())


def manifest(path, expected):
    if sha256(path) != expected:
        raise ValueError("publication manifest SHA256 differs")
    result = read_json(path)
    entries = result["repositories"]
    if len(entries) != 30 or len({entry["repo"] for entry in entries}) != 30:
        raise ValueError("expected exactly 30 unique repositories")
    for entry in entries:
        if not re.fullmatch(r"fse/[A-Za-z0-9_-]+", entry["repo"]):
            raise ValueError("unexpected repository name")
        expected_files = {".gitattributes", "README.md", "config.json", "model.safetensors"}
        if entry["method"] == "sif10":
            expected_files.add("preparation.json")
        if set(entry["files"]) != expected_files:
            raise ValueError("unexpected payload inventory")
        if entry["total_bytes"] != sum(spec["bytes"] for spec in entry["files"].values()):
            raise ValueError("payload size differs")
    return result


def check(folder, entry):
    if folder.is_symlink() or {p.name for p in folder.iterdir()} != set(entry["files"]):
        raise ValueError(f"payload inventory differs: {folder}")
    result = {}
    for name, expected in entry["files"].items():
        path = folder / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"payload is not a regular file: {path}")
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if actual != expected:
            raise ValueError(f"payload bytes differ: {path}")
        result[name] = actual
    return result


def card_inputs(models, entry):
    name = entry["repo"].split("/")[1]
    source = entry["source"].split("/")[1]
    record = read_json(models / f"{name}.json")
    verification = read_json(models / f"{source}-verification.json")
    fit = read_json(models / f"{source}-fit.json") if entry["method"] == "sif10" else None
    if any(record[key] != entry[key] for key in ("repo", "source", "method", "dtype")):
        raise ValueError("build record differs from publication identity")
    return record, verification, fit


def render_card(models, entry):
    record, verification, fit = card_inputs(models, entry)
    data = render(models / entry["repo"].split("/")[1], record, verification, fit).encode()
    if {"bytes": len(data), "sha256": bytes_sha256(data)} != entry["files"]["README.md"]:
        raise ValueError(f"generated card differs: {entry['repo']}")
    return data


def build(models, output, entry):
    """Serialize the prepared float32 state and recompute storage variants."""
    record, _, fit = card_inputs(models, entry)
    source_name = entry["source"].split("/")[1]
    parent = models / f"{source_name}-{entry['method']}"
    encoder = fse.Encoder.from_pretrained(parent, mmap=True)
    if encoder.fingerprint != record["float32_fingerprint"]:
        raise ValueError("prepared float32 parent fingerprint differs")
    if entry["dtype"] != "float32":
        encoder = encoder.quantize(entry["dtype"])
    if encoder.fingerprint != entry["metadata"]["fingerprint"]:
        raise ValueError("rebuilt encoder fingerprint differs")
    tensors, metadata, expected = encoder._artifact()
    historical, _ = _folder._header(models / entry["repo"].split("/")[1] / "model.safetensors")
    # Writer provenance is historical payload data, not the version running this rebuild.
    metadata["fse.created_by"] = historical["fse.created_by"]
    folder = output / entry["repo"].split("/")[1]
    folder.mkdir()
    SafetensorsContainer().write(folder / "model.safetensors", tensors, metadata, expected=expected)
    written, _ = _folder._header(folder / "model.safetensors")
    files = {
        "README.md": render_card(models, entry),
        "config.json": _folder.config_bytes(written),
        ".gitattributes": ATTRIBUTES,
    }
    if fit is not None:
        files["preparation.json"] = (json.dumps(fit, indent=2) + "\n").encode()
    for name, data in files.items():
        with (folder / name).open("xb") as stream:
            stream.write(data)
    return folder, check(folder, entry)
