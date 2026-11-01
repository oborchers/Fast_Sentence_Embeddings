"""Prepare release cards in a new payload view, preserving every model byte."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
from importlib.metadata import version
from pathlib import Path

from hub_classic_sources import DTYPES, FSE_API_REVISION, SOURCES
from hub_classic_verification import repo_name
from hub_public_readback import run_probe
from hub_publication import (
    card_example,
    file_sha256,
    metadata_of,
    sha256_bytes,
)
from hub_readback_evidence import mkdir_durable, write_atomic

import fse
from fse._container._common import _sync_directory

RELEASE = "2.0.0"
TAG = "v2.0.0"
DEVELOPMENT_NOTICE = (
    "This model requires the fse 2.0 API, currently in development. "
    "The install command pins its source."
)
PINNED_INSTALL = (
    'pip install "git+https://github.com/oborchers/Fast_Sentence_Embeddings.git@'
    + FSE_API_REVISION
    + '"'
)


def release_identity():
    installed = version("fse")
    if installed != RELEASE or fse.__version__ != RELEASE:
        raise ValueError("release cards require an installed fse 2.0.0 release")
    data = subprocess.check_output(
        ["git", "ls-remote", "--exit-code", "origin", f"refs/tags/{TAG}", f"refs/tags/{TAG}^{{}}"],
        text=True,
    )
    refs = dict(line.split()[::-1] for line in data.splitlines())
    commit = refs.get(f"refs/tags/{TAG}^{{}}", refs.get(f"refs/tags/{TAG}", ""))
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("origin has no valid v2.0.0 release tag")
    return {"version": installed, "tag": TAG, "tag_commit": commit}


def render_release_card(text, repo):
    if text.count(PINNED_INSTALL) != 1 or text.count(DEVELOPMENT_NOTICE) != 1:
        raise ValueError("basis card is not the exact development-card usage pattern")
    result = text.replace(PINNED_INSTALL, "pip install fse").replace(
        DEVELOPMENT_NOTICE, "This model requires fse 2.0.0 or later."
    )
    if card_example(result, repo) != card_example(text, repo):
        raise ValueError("release rendering changed the literal Python example")
    return result


def check_files(folder, expected):
    if folder.is_symlink() or {p.name for p in folder.iterdir()} != set(expected):
        raise ValueError("release payload file inventory differs from its basis manifest")
    actual = {}
    for name, spec in expected.items():
        path = folder / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("release payloads require regular files")
        actual[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        if actual[name] != spec:
            raise ValueError(f"release payload differs from its basis manifest: {name}")
    return actual


def check_runtime(entry):
    folder = Path(entry["local_folder"])
    encoder = fse.Encoder.from_pretrained(folder, mmap=True)
    preparation = (
        json.loads((folder / "preparation.json").read_text())
        if entry["method"] == "sif10"
        else None
    )
    if metadata_of(encoder, entry["method"], preparation) != entry["metadata"]:
        raise ValueError("release runtime metadata differs from the approved build")
    probe = run_probe(encoder, entry["probe"])
    if not probe["tolerance_ok"] or not probe["bitwise_ok"]:
        raise ValueError("release runtime does not reproduce the approved local probe")


def release_plan(manifest, output):
    expected = {
        repo_name(name, method, dtype)
        for name in SOURCES
        for method in ("raw", "sif10")
        for dtype in DTYPES
    }
    entries = manifest["repositories"]
    if len(entries) != 30 or {entry["repo"] for entry in entries} != expected:
        raise ValueError("release preparation requires the exact 30 approved repositories")
    plans = []
    for entry in entries:
        folder = Path(entry["local_folder"])
        if output.resolve().is_relative_to(folder.resolve()):
            raise ValueError("release output must be outside every existing payload folder")
        if entry["publication_hold"] is not None:
            raise ValueError("a publication hold remains in the basis manifest")
        check_files(folder, entry["files"])
        check_runtime(entry)
        card = (folder / "README.md").read_bytes()
        if sha256_bytes(card) != entry["files"]["README.md"]["sha256"]:
            raise ValueError("basis card changed after release preflight")
        text = render_release_card(card.decode("utf-8"), entry["repo"])
        plans.append((entry, text))
    return plans


def write_card(path, text):
    with path.open("xb") as stream:
        stream.write(text.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


def release_entry(entry, text, output):
    folder = output / "models" / entry["repo"].split("/")[1]
    mkdir_durable(folder)
    for name in entry["files"]:
        if name != "README.md":
            os.link(Path(entry["local_folder"]) / name, folder / name)
    write_card(folder / "README.md", text)
    _sync_directory(folder)
    after = copy.deepcopy(entry)
    after["local_folder"] = str(folder.resolve())
    after["files"]["README.md"] = {
        "bytes": len(text.encode("utf-8")),
        "sha256": sha256_bytes(text.encode("utf-8")),
    }
    check_files(folder, after["files"])
    after["total_bytes"] = sum(spec["bytes"] for spec in after["files"].values())
    write_card(output / "cards" / f"{folder.name}.md", text)
    return after


def prepare_release(manifest_path, output, expected_sha256):
    if output.exists():
        raise FileExistsError("release outputs are never overwritten; choose a fresh directory")
    basis = manifest_path.read_bytes()
    basis_sha256 = sha256_bytes(basis)
    if basis_sha256 != expected_sha256:
        raise ValueError("basis manifest differs from the approved SHA256")
    identity = release_identity()
    manifest = json.loads(basis)
    plans = release_plan(manifest, output)
    mkdir_durable(output)
    mkdir_durable(output / "models")
    mkdir_durable(output / "cards")
    result = copy.deepcopy(manifest)
    result["repositories"] = [release_entry(entry, text, output) for entry, text in plans]
    result["total_bytes"] = sum(entry["total_bytes"] for entry in result["repositories"])
    result["release"] = {**identity, "basis_manifest_sha256": basis_sha256}
    path = output / "publication-manifest.json"
    write_atomic(path, result)
    receipt = {
        **result["release"],
        "publication_manifest_sha256": file_sha256(path),
        "hub_writes_performed": False,
        "repositories": [
            {
                "repo": entry["repo"],
                "changed_payload_files": ["README.md"],
                "unchanged_model_sha256": entry["files"]["model.safetensors"]["sha256"],
            }
            for entry in result["repositories"]
        ],
    }
    write_atomic(output / "release-transition.json", receipt)
    return result, receipt
