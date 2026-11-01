"""Read-only checks for one already uploaded PUBLIC repo, in a fresh child cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from hub_classic_metrics import score
from hub_publication import (
    TOLERANCE_POINTS,
    card_example,
    file_sha256,
    metadata_of,
    probe_inputs,
    sha256_bytes,
    validate_vectors,
)

from fse._container._hash import checksum

FLAGS = ("files_ok", "load_ok", "metadata_ok", "run_ok")


def check_public_info(entry, revision):
    from huggingface_hub import HfApi

    info = HfApi(token=False).model_info(entry["repo"], revision="main", files_metadata=True)
    if info.id != entry["repo"] or info.private is not False or info.sha != revision:
        raise ValueError("public repo identity, visibility or uploaded revision differs")
    siblings = {file.rfilename: file for file in info.siblings}
    if set(siblings) != set(entry["files"]):
        raise ValueError("public repository file inventory differs from the approved payload")
    model = siblings["model.safetensors"]
    expected = entry["files"]["model.safetensors"]
    if (
        model.lfs is None
        or model.lfs.sha256 != expected["sha256"]
        or model.lfs.size != expected["bytes"]
    ):
        raise ValueError("public model LFS identity differs from the approved payload")


def execute_example(entry, local_folder):
    card = (local_folder / "README.md").read_bytes()
    if sha256_bytes(card) != entry["files"]["README.md"]["sha256"]:
        raise ValueError("local approved card changed")
    code = card_example(card.decode("utf-8"), entry["repo"])
    if (
        code != entry["python_example"]
        or sha256_bytes(code.encode()) != entry["python_example_sha256"]
    ):
        raise ValueError("approved literal Python example changed")
    namespace = {}
    exec(compile(code, "<approved-model-card>", "exec"), namespace)
    encoder = namespace["encoder"]
    validate_vectors(namespace["vectors"], 1, entry["metadata"]["dimension"])
    return encoder


def check_cache_revision(home, repo, revision):
    ref = home / "hub" / ("models--" + repo.replace("/", "--")) / "refs" / "main"
    if not ref.is_file() or ref.read_text().strip() != revision:
        raise ValueError(
            "literal public API load did not use the uploaded revision in its own cache"
        )


def download_and_check(entry, revision, home):
    from huggingface_hub import hf_hub_download

    paths, files = {}, {}
    for name, expected in entry["files"].items():
        path = Path(
            hf_hub_download(
                entry["repo"], name, revision=revision, token=False, cache_dir=home / "hub"
            )
        )
        if not path.resolve().is_relative_to(home.resolve()):
            raise ValueError("download escaped the dedicated empty cache")
        actual = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        if actual != expected:
            raise ValueError(f"downloaded file differs from the approved build: {name}")
        paths[name], files[name] = path, actual
    return paths, files


def run_probe(encoder, expected):
    if expected["tolerance_points"] != TOLERANCE_POINTS:
        raise ValueError("probe tolerance differs from the documented verification tolerance")
    labels, texts = probe_inputs(expected["pair_indices"])
    vectors = encoder.encode(texts, workers=2)
    validate_vectors(vectors, expected["sentences"], encoder.output_dim)
    pearson = score(labels, vectors)
    delta = pearson - expected["pearson_x100"]
    if not np.isfinite(pearson):
        raise ValueError("fresh Hub STS-B probe Pearson is not finite")
    actual_checksum = checksum(vectors)
    return {
        "pearson_x100": pearson,
        "local_pearson_x100": expected["pearson_x100"],
        "difference_points": delta,
        "tolerance_points": TOLERANCE_POINTS,
        "sentences": len(texts),
        "shape": list(vectors.shape),
        "output_dtype": vectors.dtype.name,
        "vectors_xxh3": actual_checksum,
        "expected_vectors_xxh3": expected["vectors_xxh3"],
        "tolerance_ok": abs(delta) <= TOLERANCE_POINTS,
        "bitwise_ok": actual_checksum == expected["vectors_xxh3"],
    }


def check_uploaded_repo(entry, revision, home):
    row = {
        "repo": entry["repo"],
        "uploaded_revision": revision,
        **dict.fromkeys(FLAGS),
        "difference_points": None,
        "hub_writes_performed": False,
    }
    stage = "fresh-cache"
    try:
        if not home.is_dir() or any(home.iterdir()):
            raise ValueError("HF_HOME is not a newly created empty directory")
        row["cache_empty_at_start"] = True
        stage = "files"
        check_public_info(entry, revision)
        stage = "load"
        encoder = execute_example(entry, Path(entry["local_folder"]))
        check_cache_revision(home, entry["repo"], revision)
        row["load_ok"] = True
        row["literal_example_sha256"] = entry["python_example_sha256"]
        stage = "files"
        paths, row["files"] = download_and_check(entry, revision, home)
        row["files_ok"] = True
        stage = "metadata"
        preparation = (
            json.loads(paths["preparation.json"].read_text())
            if entry["method"] == "sif10"
            else None
        )
        row["metadata"] = metadata_of(encoder, entry["method"], preparation)
        if row["metadata"] != entry["metadata"]:
            raise ValueError("public model metadata differs from the local build")
        row["metadata_ok"] = True
        stage = "run"
        row["probe"] = run_probe(encoder, entry["probe"])
        row["difference_points"] = row["probe"]["difference_points"]
        row["run_ok"] = row["probe"]["tolerance_ok"] and row["probe"]["bitwise_ok"]
        if not row["run_ok"]:
            raise ValueError(
                "fresh Hub probe differs from the local reference; see probe measurements"
            )
    except Exception as error:
        row[
            {
                "files": "files_ok",
                "load": "load_ok",
                "metadata": "metadata_ok",
                "run": "run_ok",
            }.get(stage, "load_ok")
        ] = False
        row["error"] = {"stage": stage, "type": type(error).__name__}
        response = getattr(error, "response", None)
        if response is not None:
            row["error"]["http_status"] = getattr(response, "status_code", None)
        if isinstance(error, ValueError):
            row["error"]["detail"] = str(error)
    row["status"] = "passed" if all(row[key] is True for key in FLAGS) else "failed"
    return row
