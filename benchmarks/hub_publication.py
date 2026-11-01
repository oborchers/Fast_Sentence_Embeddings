"""Freeze the exact PUBLIC payload and local cold-load probe before human approval.

This module has no Hub writes. The manifest is the artifact to approve. All
payload bytes, including .gitattributes, must be uploaded unchanged afterwards.
"""

from __future__ import annotations

import ast
import io
import json
import re

import numpy as np
from hub_classic_delivery import validated_source
from hub_classic_evidence import benchmark_bytes
from hub_classic_metrics import score
from hub_classic_sources import ATTRIBUTES, LICENSES, SOURCES

import fse
from fse._container._hash import checksum, file_digests

PROBE_PAIRS = 256
TOLERANCE_POINTS = 0.001


def sha256_bytes(data):
    return file_digests(io.BytesIO(data), len(data))[0]


def file_sha256(path):
    before = path.stat()
    with path.open("rb") as stream:
        value = file_digests(stream, before.st_size)[0]
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError("file changed while computing the publication checksum")
    return value


def card_example(card, repo):
    blocks = re.findall(r"```python\r?\n(.*?)\r?\n```", card, flags=re.DOTALL)
    if len(blocks) != 1:
        raise ValueError("a classic card must contain exactly one Python example")
    code = blocks[0]
    expected = f'import fse\nencoder = fse.Encoder.from_pretrained({json.dumps(repo)})\nvectors = encoder.encode(["A sentence."])'
    if ast.dump(ast.parse(code)) != ast.dump(ast.parse(expected)):
        raise ValueError("the card example is not the approved read-only load/encode pattern")
    return code


def probe_inputs(indices):
    lines = [line.split("\t") for line in benchmark_bytes().decode().splitlines()]
    if len(lines) != 1379 or not indices or len(set(indices)) != len(indices):
        raise ValueError("invalid pinned STS-B probe")
    if any(type(index) is not int or not 0 <= index < len(lines) for index in indices):
        raise ValueError("invalid STS-B probe pair index")
    labels = np.asarray([float(lines[index][4]) for index in indices])
    texts = [lines[index][5] for index in indices] + [lines[index][6] for index in indices]
    return labels, texts


def metadata_of(encoder, method, preparation):
    components = 0 if method == "raw" else preparation["components"]
    if method == "sif10" and components != 10:
        raise ValueError("SIF preparation must declare ten folded components")
    return {
        "dtype": encoder.word_vectors.stored_vectors.dtype.name,
        "dimension": encoder.output_dim,
        "vocabulary_size": len(encoder.word_vectors),
        "sif_components": components,
        "fingerprint": encoder.fingerprint,
    }


def local_probe(encoder, indices):
    labels, texts = probe_inputs(indices)
    vectors = encoder.encode(texts, workers=2)
    validate_vectors(vectors, len(texts), encoder.output_dim)
    value = score(labels, vectors)
    if not np.isfinite(value):
        raise ValueError("local STS-B probe Pearson is not finite")
    return {
        "pair_indices": indices,
        "pairs": len(indices),
        "sentences": len(texts),
        "tokenization": "saved fse_words@1 on raw STS-B texts",
        "pearson_x100": value,
        "tolerance_points": TOLERANCE_POINTS,
        "vectors_xxh3": checksum(vectors),
    }


def validate_vectors(vectors, rows, dimension):
    if not isinstance(vectors, np.ndarray) or vectors.shape != (rows, dimension):
        raise ValueError("encoded probe has the wrong shape")
    if vectors.dtype != np.dtype("float32") or not np.isfinite(vectors).all():
        raise ValueError("encoded probe must be finite float32")


def publication_entry(models, record, fit, indices):
    folder = models / record["repo"].split("/")[1]
    encoder = fse.Encoder.from_pretrained(folder, mmap=True)
    example = card_example((folder / "README.md").read_bytes().decode("utf-8"), record["repo"])
    name = record["source"].split("/")[1]
    files = {
        p.name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
        for p in sorted(folder.iterdir())
    }
    return {
        "repo": record["repo"],
        "visibility": "public",
        "source": record["source"],
        "source_revision": record["revision"],
        "method": record["method"],
        "dtype": record["dtype"],
        "source_vector_license": LICENSES[name],
        "local_folder": str(folder),
        "files": files,
        "total_bytes": sum(spec["bytes"] for spec in files.values()),
        "metadata": metadata_of(encoder, record["method"], fit),
        "python_example": example,
        "python_example_sha256": sha256_bytes(example.encode()),
        "probe": local_probe(encoder, indices),
        "publication_hold": None,
    }


def freeze_manifest(models, selection, output):
    if output.exists():
        raise FileExistsError("publication manifests are immutable; choose a new output name")
    indices = sorted(np.random.default_rng(0).choice(1379, PROBE_PAIRS, replace=False).tolist())
    entries = []
    for name in SOURCES:
        records, _, fit = validated_source(models, name, selection, enriched=True)
        for record in records:
            folder = models / record["repo"].split("/")[1]
            if (folder / ".gitattributes").read_bytes() != ATTRIBUTES:
                raise ValueError("public payload requires the exact reviewed LFS attributes")
            entries.append(publication_entry(models, record, fit, indices))
    value = {
        "publication_manifest_version": 1,
        "visibility": "public",
        "approved": False,
        "hub_writes_performed": False,
        "repositories": entries,
        "total_bytes": sum(entry["total_bytes"] for entry in entries),
        "approval_gate": "Oliver approves the exact cards, repo names, licenses and full file inventory before any Hub write. No private staging repos.",
        "post_upload": "Every approved uploaded repo requires a fresh isolated cache, exact SHA256 files, literal public API example, metadata and 512-sentence probe. No Hub repair writes on failure.",
    }
    with output.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    return value


def size_tables(manifest):
    grouped = {}
    for entry in manifest["repositories"]:
        grouped.setdefault((entry["source"], entry["method"]), {})[entry["dtype"]] = entry[
            "total_bytes"
        ]
    lines = [
        "| Source | Method | float32 bytes | float16 bytes | int8 bytes | Total bytes |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (source, method), sizes in sorted(grouped.items()):
        values = [sizes[dtype] for dtype in ("float32", "float16", "int8")]
        lines.append(
            f"| {source} | {method} | {values[0]} | {values[1]} | {values[2]} | {sum(values)} |"
        )
    lines += ["", f"All repository files: **{manifest['total_bytes']} bytes**.", ""]
    return lines


def file_table(manifest):
    lines = ["| Intended PUBLIC repository | File | Bytes | SHA256 |", "|---|---|---:|---|"]
    for entry in manifest["repositories"]:
        for name, spec in entry["files"].items():
            lines.append(f"| {entry['repo']} | `{name}` | {spec['bytes']} | `{spec['sha256']}` |")
    return lines
