"""Prepare exactly the static sources needed by the two pod workloads."""

from __future__ import annotations

import json
from pathlib import Path

from classic_sources import SOURCES
from preparation_catalog import register

HUB = (
    "minishlab/potion-base-8M",
    "minishlab/potion-base-32M",
    "minishlab/potion-retrieval-32M",
    "minishlab/potion-multilingual-128M",
    "sentence-transformers/static-retrieval-mrl-en-v1",
    "sentence-transformers/static-similarity-mrl-multilingual-v1",
)


def hub_model(repo: str, folder: Path, dtypes=("float32",)) -> list[Path]:
    from compare_models import MODELS, artifact_name

    from fse import Encoder

    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / artifact_name(repo, dtype) for dtype in dtypes]
    if any(path.exists() for path in paths):
        raise FileExistsError("preparation refuses existing artifacts")
    encoder = Encoder.from_pretrained(repo, revision=MODELS[repo])
    record = {
        "source": repo,
        "revision": MODELS[repo],
        "recipe": "pinned fse conversion",
        "fit": None,
        "artifacts": [p.name for p in paths],
    }
    parent = encoder.fingerprint
    for dtype, path in zip(dtypes, paths, strict=True):
        current = encoder if dtype == "float32" else encoder.quantize(dtype)
        current.save(path)
        register(folder, path, record, parent)
    return paths


def prepare_mind(folder: Path, cache: Path, workers: int):
    from prepare_classics import prepare

    for repo in HUB:
        hub_model(repo, folder)
    for name in SOURCES:
        prepare(name, folder, cache, workers, ["float32"])
    models = ["fse/" + p.stem for p in folder.glob("*-float32.safetensors")]
    models.sort(
        key=lambda model: (folder / (model.removeprefix("fse/") + ".safetensors")).stat().st_size
    )
    return [*models, "sentence-transformers/all-MiniLM-L6-v2"]


def archive_fit_receipts(folder: Path) -> dict:
    manifest = json.loads((folder / "artifact-identities.json").read_text())
    return {
        row["artifact"]: {
            "fingerprint": row["encoder_fingerprint"],
            "fit": row["preparation_record"].get("fit"),
            "source": row["source"],
            "revision": row["revision"],
        }
        for row in manifest["artifacts"]
    }
