"""Content identities for benchmark inputs, prepared heads and executed runtime."""

from __future__ import annotations

import json
from pathlib import Path


def file_digest(path: Path) -> str:
    from fse._container._hash import Hasher

    hasher = Hasher()
    with path.open("rb") as stream:
        while block := stream.read(8 * 2**20):
            hasher.update(block)
    return hasher.checksum()


def runtime_identity() -> str:
    import fse
    from fse._container._hash import Hasher

    root = Path(fse.__file__).resolve().parent
    hasher = Hasher()
    paths = [*root.rglob("*.py"), *root.glob("_native*.so"), *root.glob("_native*.pyd")]
    for path in sorted(paths):
        hasher.update(str(path.relative_to(root)).encode("utf-8"))
        hasher.update(file_digest(path).encode("ascii"))
    return hasher.checksum()


def seal_preparation(folder: Path) -> None:
    path = folder / "manifest.json"
    record = json.loads(path.read_text())
    record["artifacts"] = {
        file.name: {"file_bytes": file.stat().st_size, "checksum": file_digest(file)}
        for file in sorted(folder.iterdir())
        if file.is_file() and file != path
    }
    path.write_text(json.dumps(record, indent=2) + "\n")


def validate_preparation(folder: Path, revision: str, evaluation_digest: str) -> dict:
    record = json.loads((folder / "manifest.json").read_text())
    if (
        record["dataset_revision"] != revision
        or record["evaluation_digest"] != evaluation_digest
        or not record.get("artifacts")
    ):
        raise ValueError(f"{folder.name}: classifier preparation does not match this evaluation")
    return record


def checked_encoder(encoder, models: Path, repo: str, dtype: str) -> dict:
    from compare_models import MODELS, artifact_path
    from quality.artifact_manifest import verified_source

    source = verified_source(artifact_path(models, repo, dtype), encoder)
    if (
        source["source_model"] != repo
        or source["source_revision"] != MODELS[repo]
        or source["table_dtype"] != dtype
    ):
        raise ValueError("loaded encoder source differs from the requested pinned model")
    return {**source, "encoder_fingerprint": encoder.fingerprint}


def record_head_encoder(folder: Path, name: str, source: dict) -> None:
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.setdefault("head_encoders", {})[name] = source
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def check_head_encoder(manifest: dict, name: str, source: dict) -> None:
    training = manifest.get("head_encoders", {}).get(name)
    if not training or any(
        training[key] != source[key] for key in ("source_model", "source_revision")
    ):
        raise ValueError(f"{name}: missing or mismatched head training encoder source")
    if training["encoder_fingerprint"] != source["float32_parent_fingerprint"]:
        raise ValueError(f"{name}: encoder is not a descendant of the head training table")


def native_encoder_identity(repo: str, folder: Path) -> dict:
    from compare_models import MODELS

    from fse._container._hash import checksum

    files = {
        str(path.relative_to(folder)): file_digest(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix in (".safetensors", ".json")
    }
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("native encoder identity needs model weights")
    fingerprint = checksum(json.dumps(files, sort_keys=True).encode("utf-8"))
    return {
        "source_model": repo,
        "source_revision": MODELS[repo],
        "encoder_fingerprint": fingerprint,
        "float32_parent_fingerprint": fingerprint,
        "source_files": files,
    }
